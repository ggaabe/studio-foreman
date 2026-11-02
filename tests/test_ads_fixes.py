"""The fixer's own tests for the ads red-team round, beside the red team's (which pass as written):

  F1  typed facts: a price must be one the facts state as a price, a percentage as a percentage
  F2  the forced aligner Foreman's ASR request loaded leaves with the ASR model; anyone else's stays
  F3  ComfyUI outputs: what /view returned must be the file it should be
  F4  drawn text is measured with the font and fitted to the frame
  F5  engine-reported infrastructure reasons reach the ledger as infrastructure
"""
from __future__ import annotations

import json
import os
import unittest

from foreman import ads, assembly, ledger, studio
from foreman.adapters.comfyui import output_problem
from foreman.adapters.local import InfraFailure, LocalEngine
from foreman.adapters.mlx_serve import MlxServeEngine
from foreman.ads import number_problems, typed_numbers, validate_script
from foreman.engine import RunSpec
from foreman.supervisor import ENGINE_FAILURE_REASONS, ENGINE_INFRA_REASONS

try:
    from ad_fixtures import FACTS, FFMPEG, SCRIPT, TmpCase, ads_config, png, template_data, wait_for, wav
    from fake_mlx_serve import FakeMlxServe
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import FACTS, FFMPEG, SCRIPT, TmpCase, ads_config, png, template_data, wait_for, wav
    from tests.fake_mlx_serve import FakeMlxServe

CAT = studio.catalog(False, ads=True)


def typed_facts(extra: str = "") -> dict:
    text = json.dumps(FACTS, indent=1) + (("\n" + extra) if extra else "")
    t = typed_numbers(text)
    return {"text": text, "numbers": sorted(ads.numbers_in(text)), "money": t["money"], "percent": t["percent"]}


def ad_spec() -> dict:
    from foreman.templates import parse_template
    t = parse_template(template_data())
    return dict(t.ad, id="ad-f", template_id=t.id, seconds=20.0, shots=6, seed=5, brief="x",
                length={"min": 15.0, "max": 30.0}, shot_range={"min": 4, "max": 6}, commercial=True)


class F1TypedFacts(unittest.TestCase):

    def test_numbers_have_kinds(self):
        self.assertEqual(typed_numbers("It is $9.50, or 9.5 dollars, or nine dollars fifty."),
                         {"money": ["9.5"], "percent": []})
        self.assertEqual(typed_numbers("Save 20% or twenty percent, on 3 forms")["percent"], ["20"])
        self.assertEqual(typed_numbers("only 50 cents")["money"], ["0.5"])

    def test_a_percentage_must_be_a_percentage_of_the_facts(self):
        facts = typed_facts("Up to 50 pages.")
        self.assertEqual(number_problems("Each form can have 50 pages.", facts), [])
        self.assertTrue(number_problems("Save 50% on every form.", facts), "50 pages is not 50%")

    def test_a_price_must_be_a_price_of_the_facts(self):
        facts = typed_facts("Up to 2 files per form.")
        self.assertTrue(number_problems("It costs $2 per form.", facts), "2 files is not $2")
        self.assertEqual(number_problems("It costs $9.50 per form.", facts), [])

    def test_the_fixture_script_passes_with_typed_facts(self):
        script, errs = validate_script(SCRIPT, ad_spec(), typed_facts(), ["hero", "talk"], ads_config())
        self.assertEqual(errs, [])
        self.assertIsNotNone(script)


class F2Aligner(TmpCase):

    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.tmp, "home")
        self.art = os.path.join(self.home, "artifacts")
        os.makedirs(self.art)
        self.fake = FakeMlxServe().start()
        self.fake.asr_loads = (studio.ALIGNER,)
        self.eng = MlxServeEngine(self.fake.url, catalog=CAT, model_dir="/fake/models", artifacts_dir=self.art,
                                  mem_probe=lambda: 200.0, read_poll_s=0.02, unload_wait_s=2.0, connect_timeout_s=2.0,
                                  call_timeout_s=2.0, forbidden=studio.FORBIDDEN, ffmpeg=FFMPEG,
                                  input_roots=[self.home], max_resident_models=None,
                                  companions={studio.QWEN3_ASR: (studio.ALIGNER,)})

    def tearDown(self):
        self.eng.close()
        self.fake.stop()
        super().tearDown()

    def transcribe(self):
        t = self.eng.load(studio.QWEN3_ASR)
        self.assertTrue(wait_for(lambda: self.eng.poll_load(t).state == "ready"), self.eng.poll_load(t))
        wav(os.path.join(self.art, "ads", "al", "voice", "track.wav"), 2.0)
        h = self.eng.start(RunSpec("1.1", 1, "job", studio.QWEN3_ASR,
                                   {"goal": "g", "expected_s": 30.0, "workdir": "ads/al",
                                    "ops": [{"op": "transcribe", "out": "t.json", "file_path": "voice/track.wav"}]},
                                   30.0))
        self.assertTrue(wait_for(lambda: self.eng.poll(h).status != "running", 20.0))
        self.assertEqual(self.eng.poll(h).status, "done", self.eng.poll(h))

    def test_an_aligner_foremans_request_loaded_goes_with_the_asr(self):
        self.transcribe()
        self.assertIn(studio.ALIGNER, self.fake.ready, "the fake loaded the aligner for word times")
        self.assertIn(studio.ALIGNER, self.eng.resident_media(), "a resident aligner blocks ComfyUI")
        self.assertTrue(self.eng.unload(studio.QWEN3_ASR))
        self.assertNotIn(studio.ALIGNER, self.fake.ready)

    def test_an_aligner_someone_else_loaded_is_never_unloaded(self):
        self.fake.other_client_load(studio.ALIGNER)
        self.transcribe()
        self.assertTrue(self.eng.unload(studio.QWEN3_ASR))
        self.assertIn(studio.ALIGNER, self.fake.ready, "it was resident before Foreman's request")


class F3Outputs(TmpCase):

    def write(self, name, data: bytes) -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_outputs_are_checked(self):
        good = png(os.path.join(self.tmp, "good.png"))
        self.assertIsNone(output_problem(good, "image/png", "out.png"))
        with open(good, "rb") as f:
            data = f.read()
        self.assertIn("truncated", output_problem(self.write("cut.png", data[: len(data) // 2]), "image/png", "x.png"))
        self.assertIn("text/html", output_problem(self.write("err.png", b"<html>502</html>"), "text/html", "x.png"))
        self.assertIsNotNone(output_problem(self.write("v.mp4", b"<html>oops</html>"), "", "x.mp4"))
        self.assertIsNone(output_problem(self.write("ok.mp4", b"\0\0\0\x18ftypisom" + b"\0" * 64), "video/mp4", "x.mp4"))


class F4Fitting(TmpCase):

    def test_text_is_measured_with_the_font(self):
        try:
            font = assembly.find_font()
        except assembly.FontError as e:
            self.skipTest(str(e))
        w1 = assembly.text_width(font, "iiii", 100)
        w2 = assembly.text_width(font, "WWWW", 100)
        self.assertGreater(w2, 2 * w1, "advance widths come from the font, not a constant")
        self.assertLessEqual(assembly.text_width(font, "www.example.com", 80),
                             assembly.text_width(None, "www.example.com", 80), "the estimate is generous")

    def test_fit_text_shrinks_then_wraps(self):
        lines, size = assembly.fit_text(None, "short", 40, 1000, 20)
        self.assertEqual((lines, size), (["short"], 40))
        lines, size = assembly.fit_text(None, "a much longer call to action line here", 60, 600, 40)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(assembly.text_width(None, ln, size) <= 600 for ln in lines))

    def test_a_long_caption_word_gets_a_smaller_font(self):
        words = [{"text": "Visit", "start": 0.1, "end": 0.5, "line": 0},
                 {"text": "www.a-very-long-address-for-example.com", "start": 0.6, "end": 1.8, "line": 0}]
        path = os.path.join(self.tmp, "c.ass")
        assembly.write_ass(path, words, 540, 960, "Arial")
        with open(path, encoding="utf-8") as f:
            events = [ln for ln in f.read().splitlines() if ln.startswith("Dialogue:")]
        self.assertTrue(any("{\\fs" in e and "www." in e for e in events), events)
        self.assertFalse(any("{\\fs" in e and "Visit" in e for e in events), "only the long word's phrase shrinks")


class F5Reasons(TmpCase):

    def test_engine_infra_reasons_are_infrastructure(self):
        self.assertLessEqual(set(ENGINE_INFRA_REASONS), ledger.INFRA_REASONS)
        self.assertFalse(set(ENGINE_INFRA_REASONS) & ledger.REPEAT_IS_CONTENT, "never content when repeated")
        self.assertEqual(set(ENGINE_FAILURE_REASONS), {"rejected"})

    def test_the_local_engine_reports_an_infra_failure_with_its_reason(self):
        def handler(spec, ctx):
            raise InfraFailure("no font here", reason="engine_unconfigured")
        eng = LocalEngine(CAT, {"ad_assemble": handler}, artifacts_dir=self.tmp, name=studio.LOCAL_ENGINE)
        self.addCleanup(eng.close)
        eng.load(studio.LOCAL)
        h = eng.start(RunSpec("1.1", 1, "ad_assemble", studio.LOCAL, {"goal": "g", "expected_s": 30.0}, 30.0))
        self.assertTrue(wait_for(lambda: eng.poll(h).status != "running", 10.0))
        p = eng.poll(h)
        self.assertEqual((p.status, p.reason), ("failed", "engine_unconfigured"))


if __name__ == "__main__":
    unittest.main()
