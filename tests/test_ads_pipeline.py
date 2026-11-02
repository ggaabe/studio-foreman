"""The ad pipeline's parts on their own:

  P1  the ad template kind and its "ad" block; the planner proposes an ad like any template
  P2  the brain's script: validated by code (shots, lines, facts, banned claims, marks, URLs,
      disclosure), numbers read the way people write them
  P3  timing: ASR word times mapped onto the script text; cuts, 8k+1 frames, padded talk audio
  P4  characters: import, tags, approval, validation, the CLI
  P5  facts: a file or a URL, the end card price checked against them
  P6  assembly with real ffmpeg on synthetic inputs: 1080x1920, duration, audio, -14 LUFS,
      captions burned in (frames with and without differ), the safe zone, the end card, size cap
  P7  the dispatcher: recovery decisions, the ad's lineage checks, the review's decision rules
"""
from __future__ import annotations

import contextlib
import http.server
import io
import json
import os
import threading
import unittest

from foreman import ad_pipeline, ads, assembly, characters, media, paths, studio
from foreman.adapters.local import ContentFailure
from foreman.ads import AdsConfig, numbers_in, plan_shots, validate_script, word_times
from foreman.planner import Planner, PlannerConfig
from foreman.supervisor import AdmissionError
from foreman.templates import TemplateError, parse_template

try:
    from ad_fixtures import (FACTS, FFMPEG, SCRIPT, TmpCase, ads_config, make_home, needs_ffmpeg, png, probe_frame,
                             template_data, wav)
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import (FACTS, FFMPEG, SCRIPT, TmpCase, ads_config, make_home, needs_ffmpeg, png,
                                   probe_frame, template_data, wav)

TAGS = ["hero", "talk", "turnaround", "endcard"]


def ad_spec(**over) -> dict:
    t = parse_template(template_data())
    spec = dict(t.ad, id="ad-1", template_id=t.id, seconds=20.0, shots=6, seed=5, brief="an angle",
                length={"min": 15.0, "max": 30.0}, shot_range={"min": 4, "max": 6}, commercial=True)
    spec.update(over)
    return spec


def facts() -> dict:
    text = json.dumps(FACTS, indent=1)
    return {"text": text, "numbers": sorted(numbers_in(text))}


# ---------------------------------------------------------------- P1

class P1Template(unittest.TestCase):

    def test_an_ad_template_loads_with_its_block(self):
        t = parse_template(template_data())
        self.assertEqual((t.kind, t.family, t.ad["character"], t.ad["end_card"]["price"]), ("ad", "ad", "mascot-a", "$9.50"))
        self.assertEqual(t.params["shots"]["default"], 6)
        self.assertIn("ad", t.describe())
        self.assertNotIn("example.com", json.dumps(t.describe()), "the planner is not shown the address")

    def test_the_examples_are_generic_and_load(self):
        import shutil
        import tempfile
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ex = os.path.join(repo, "examples", "ads")
        from foreman.templates import load_templates
        templates, errors = load_templates(os.path.join(ex, "templates"))
        self.assertEqual(errors, [])
        t = templates["ads.example-service.v1"]
        self.assertEqual((t.kind, t.active), ("ad", False))
        for root, _dirs, files in os.walk(ex):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    text = f.read()
                self.assertNotRegex(text, r"https?://", name)
                self.assertNotRegex(text, r"[\w.+-]+@[\w-]+\.\w+", name)
        with tempfile.TemporaryDirectory() as tmp:
            home = os.path.join(tmp, "home")
            shutil.copytree(os.path.join(ex, "characters"), os.path.join(home, "characters"))
            with open(os.path.join(ex, "characters", "mascot-example", "bible.json")) as f:
                stills = json.load(f)["stills"]
            for st in stills:
                png(os.path.join(home, "characters", "mascot-example", st["file"]))
            self.assertTrue(characters.load_bible(home, "mascot-example").hero())
            shutil.copy(os.path.join(ex, "config.ads.json"), os.path.join(tmp, "config.json"))
            cfg = studio.load_config(os.path.join(tmp, "config.json"), env={"FOREMAN_HOME": home})
            self.assertEqual(cfg.ads_config().talk_seeds, 3)
            self.assertEqual(cfg.comfy_config().plan_gb, 60)
            self.assertIn(studio.COMFYUI, cfg.catalog())
            with open(os.path.join(tmp, "bad.json"), "w") as f:
                json.dump({"ads_options": {"talk_seed": 3}}, f)
            with self.assertRaises(ValueError):
                studio.load_config(os.path.join(tmp, "bad.json"), env={"FOREMAN_HOME": home})
        with open(os.path.join(ex, "assets", "facts.json")) as f:
            nums = numbers_in(f.read())
        self.assertTrue(numbers_in(t.ad["end_card"]["price"]) <= nums, "the example's end card price is in its facts")

    def test_bad_ad_blocks_are_refused_with_every_problem(self):
        bad = template_data()
        bad["ad"] = dict(bad["ad"], facts={"url": "ftp://x"}, end_card={"url": "example.com"}, captures=["../x.png"],
                         unknown=1)
        with self.assertRaises(TemplateError) as e:
            parse_template(bad)
        msg = str(e.exception)
        for part in ("http(s)", "end_card.price", "relative to", "not an ad field"):
            self.assertIn(part, msg)
        with self.assertRaises(TemplateError):
            parse_template(template_data(kind="speak", family="tts"))  # an ad block on another kind
        with self.assertRaises(TemplateError):
            parse_template(template_data(params={"seconds": {"min": 5, "max": 90, "default": 20},
                                                 "shots": {"min": 4, "max": 6, "default": 5}}))
        spam = template_data()
        spam["ad"] = dict(spam["ad"], concept="Cold emails to everyone, then auto-post to social")
        with self.assertRaises(TemplateError):
            parse_template(spam)

    def test_the_planner_proposes_an_ad_and_prices_the_whole_chain(self):
        import tempfile
        from foreman.cli import _ClientEngine
        from foreman.supervisor import Supervisor
        try:
            from ad_fixtures import ManualClock
        except ImportError:  # pragma: no cover
            from tests.ad_fixtures import ManualClock
        with tempfile.TemporaryDirectory() as tmp:
            home = make_home(tmp)
            with open(os.path.join(home, "templates", "ad.json"), "w") as f:
                json.dump(template_data(active=True), f)
            planner = Planner(PlannerConfig(utc=True, min_disk_free_gb=0, artifacts_cap_gb=1e6), home=home,
                              artifacts_dir=os.path.join(home, "artifacts"), commercial_models=studio.COMMERCIAL_OK,
                              disk_probe=lambda: (1000.0, 0.0))
            cat = studio.catalog(False, ads=True)
            sup = Supervisor(os.path.join(tmp, "p.db"), {n: _ClientEngine(n) for n in {s.engine for s in cat.values()}},
                             ManualClock(), studio.policy(), catalog=cat, lead=False, planner=planner)
            try:
                templates = planner.templates()[0]
                now, wall = sup._now(), 1.7e9
                ctx = planner.context(sup, now, wall, templates)
                prop = {"template_id": "ads.service.v1", "consumer": "the owner's ad account (example)",
                        "brief": "A desk-side angle", "params": {"shots": 5}}
                c = planner.check_proposal(sup, prop, templates, {}, wall, ctx)
                self.assertIsInstance(c, dict, c)
                req = c["request"]
                self.assertEqual((req["kind"], req["requires"]), ("ad_facts", studio.LOCAL))
                self.assertEqual(req["payload"]["ad"]["shots"], 5)
                self.assertTrue(req["payload"]["workdir"].startswith("ads/ads.service.v1-"))
                self.assertGreater(c["est_gpu_s"], 1000, "the EV prices the whole ad, not its first step")
                from foreman import retention
                self.assertGreater(retention.estimate_bytes(req), 1e8)
                bad = dict(prop, params={"shots": 9})
                self.assertEqual(planner.check_proposal(sup, bad, templates, {}, wall, ctx)[1], "param_out_of_bounds")
                jid = sup.submit(req["kind"], req["requires"], req["payload"], req["on_success"], req["on_failure"])
                self.assertEqual(sup.store.job(jid).kind, "ad_facts")
            finally:
                sup.close()


# ---------------------------------------------------------------- P2

class P2Script(unittest.TestCase):
    def check(self, obj, **over):
        return validate_script(obj, ad_spec(**over), facts(), TAGS, ads_config())

    def test_the_fixture_script_is_valid_and_normalized(self):
        script, errs = self.check(SCRIPT)
        self.assertEqual(errs, [])
        self.assertEqual([ln["role"] for ln in script["lines"]], ["hook", "body", "body", "body", "cta"])
        self.assertEqual(script["shots"][2], {"k": 2, "kind": "screen", "lines": [2], "visual": "", "still": None,
                                              "capture": "captures/upload.png"})

    def test_each_rule_names_its_problem(self):
        cases = [
            (dict(SCRIPT, body=["It is only $2.99 today.", "x y z.", "a b c."]), "not in the facts"),
            (dict(SCRIPT, body=["Our customers love it.", "x y z.", "a b c."]), "testimonial"),
            (dict(SCRIPT, body=["Guaranteed to arrive in two days.", "x.", "y."]), "guarantee"),
            (dict(SCRIPT, body=["It arrives tomorrow.", "x.", "y."]), "delivery promise"),
            (dict(SCRIPT, body=["This is legally binding everywhere.", "x.", "y."]), "legal advice"),
            (dict(SCRIPT, body=["Better than OtherCo.", "x.", "y."]), "OtherCo"),
            (dict(SCRIPT, body=["See other-site.org for more.", "x.", "y."]), "not the ad's own URL"),
            (dict(SCRIPT, disclosure="Made by people"), "disclosure"),
            (dict(SCRIPT, shots=SCRIPT["shots"][1:]), "hook"),
            (dict(SCRIPT, shots=[dict(s, capture="captures/other.png") if s["kind"] == "screen" else s
                                 for s in SCRIPT["shots"]]), "capture"),
            (dict(SCRIPT, shots=[dict(s, visual="a sign that says SALE") if s["kind"] == "broll" else s
                                 for s in SCRIPT["shots"]]), "on-screen text"),
            (dict(SCRIPT, shots=[dict(s, still="desk") if s["kind"] == "talk" else s for s in SCRIPT["shots"]]),
             "not an approved still"),
            (dict(SCRIPT, shots=SCRIPT["shots"][:-1]), "endcard"),
            (dict(SCRIPT, body=["word " * 40]), "words"),
            (dict(SCRIPT, extra=1), "unknown fields"),
            ("not json", "not a JSON object"),
        ]
        for obj, needle in cases:
            script, errs = self.check(obj)
            self.assertIsNone(script, obj)
            self.assertIn(needle, " | ".join(errs), (needle, errs))

    def test_a_screen_shot_may_describe_the_real_page_it_shows(self):
        # Seen live: the brain described a real capture as "the product's pricing page on a mobile device
        # screen" and the generated-UI rule refused it. A capture's description is never a prompt.
        described = dict(SCRIPT, shots=[dict(s, visual="Static view of the product's pricing page on a phone screen.")
                                         if s["kind"] == "screen" else s for s in SCRIPT["shots"]])
        script, errs = self.check(described)
        self.assertEqual(errs, [])
        self.assertEqual([s["visual"] for s in script["shots"] if s["kind"] == "screen"], [""])
        generated = dict(SCRIPT, shots=[dict(s, visual="Her phone showing the upload page.") if s["kind"] == "broll" else s
                                        for s in SCRIPT["shots"]])
        self.assertIn("product's screens", " | ".join(self.check(generated)[1]))

    def test_length_follows_the_template(self):
        short = dict(SCRIPT, hook="Hi.", body=["Upload it."], cta="Do it at example.com.",
                     shots=[{"kind": "talk", "lines": [0], "visual": "office"}, {"kind": "broll", "lines": [1], "visual": "desk"},
                            {"kind": "talk", "lines": [2], "visual": "office"}, {"kind": "endcard", "lines": []}])
        script, errs = self.check(short)
        self.assertIn("reads in about", " ".join(errs))

    def test_numbers_are_read_the_way_people_write_them(self):
        self.assertEqual(numbers_in("$4.25 per item"), {"4.25"})
        self.assertEqual(numbers_in("four dollars and twenty five cents"), {"4.25"})
        self.assertEqual(numbers_in("four dollars twenty five"), {"4.25"})
        self.assertEqual(numbers_in("1,000 items, 40% off, forty-two pages"), {"1000", "40", "42"})
        self.assertEqual(numbers_in("one tap and you are done"), set(), "'one' reads as a word")
        self.assertEqual(numbers_in("zero fees, twenty five cents"), {"0", "0.25"})


# ---------------------------------------------------------------- P3

class P3Timing(TmpCase):
    def script(self):
        return validate_script(SCRIPT, ad_spec(), facts(), TAGS, ads_config())[0]

    def test_asr_times_land_on_the_script_text(self):
        script = self.script()
        texts = [ln["text"] for ln in script["lines"]]
        asr_words = []
        t = 0.2
        for i, line in enumerate(texts):
            for w in line.split():
                spoken = w.lower().strip(".,?!")
                if spoken == "fifty":
                    continue  # the ASR dropped a word
                asr_words.append({"word": spoken, "start": t, "end": t + 0.25})
                t += 0.3
            t += 0.4
        spans = None
        words, source, spans = word_times(script, {"words": asr_words}, spans, t + 0.3)
        self.assertEqual(source, "asr")
        self.assertEqual([w["text"] for w in words], " ".join(texts).split(), "captions show the script's words")
        self.assertIn("example.com.", [w["text"] for w in words])
        starts = [w["start"] for w in words]
        self.assertEqual(starts, sorted(starts))
        fifty = next(w for w in words if w["text"] == "fifty")
        prev = words[words.index(fifty) - 1]
        self.assertGreaterEqual(fifty["start"], prev["end"] - 1e-6, "the dropped word is interpolated in place")

    def test_without_word_times_lines_are_spread_by_length(self):
        script = self.script()
        words, source, spans = word_times(script, {"text": "x"}, None, 20.0)
        self.assertEqual(source, "estimated")
        self.assertAlmostEqual(spans[-1][1], 20.0, places=2)

    def test_cuts_frames_and_bounds(self):
        script = self.script()
        cfg = ads_config()
        spans = [[0.15 + i * 3.5, 0.15 + i * 3.5 + 3.0] for i in range(5)]
        shots, total, problems = plan_shots(script, spans, spans[-1][1], cfg, ad_spec())
        self.assertEqual(problems, [])
        self.assertEqual(shots[0]["start"], 0.0)
        self.assertAlmostEqual(shots[1]["start"], (spans[0][1] + spans[1][0]) / 2, places=3)
        for s in shots:
            if s.get("frames"):
                self.assertEqual((s["frames"] - 1) % 8, 0)
                self.assertGreaterEqual(s["frames"] / 24.0, s["seconds"])
        self.assertEqual(shots[-1]["kind"], "endcard")
        self.assertAlmostEqual(total, shots[-1]["end"], places=3)
        long_spans = [[0.0, 9.5]] + [[a + 6.5, b + 6.5] for a, b in spans[1:]]
        _, _, problems = plan_shots(script, long_spans, long_spans[-1][1], cfg, ad_spec(length={"min": 15, "max": 60}))
        self.assertTrue(any("at most 8" in p for p in problems))
        big = AdsConfig(**dict(dataclass_dict(cfg), video_width=704, video_height=1280))
        self.assertEqual(ads.frames_for(5.0), 121)
        self.assertEqual(ads.frames_for(7.0), 169)
        self.assertIsNotNone(big)

    def test_talk_audio_is_cut_and_padded_to_the_clip(self):
        src = wav(os.path.join(self.tmp, "track.wav"), 6.0, rate=24000)
        out = os.path.join(self.tmp, "talk.wav")
        self.assertEqual(ads.frames_for(2.5), 65)
        media.cut_wav(src, out, 1.0, 3.5, pad_to_s=ads.frames_for(2.5) / 24.0)
        self.assertAlmostEqual(media.wav_duration(out), 65 / 24.0, places=3)
        spans = media.concat_wavs([src, src], os.path.join(self.tmp, "j.wav"), gap_s=0.3, lead_s=0.15)
        self.assertEqual(spans, [(0.15, 6.15), (6.45, 12.45)])


def dataclass_dict(cfg):
    import dataclasses
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


# ---------------------------------------------------------------- P4

class P4Characters(TmpCase):
    def test_import_tags_approval_and_validation(self):
        src = os.path.join(self.tmp, "folder")
        for n in ("hero.png", "turnaround-sheet.png", "expression_grid.jpg", "kitchen scene.png"):
            png(os.path.join(src, n)) if n.endswith(".png") else png(os.path.join(src, n))
        home = os.path.join(self.tmp, "home")
        b = characters.import_folder(home, src, "host-a", look="a host in a navy cardigan", voice="af_heart")
        self.assertEqual(b.approved(), [], "nothing is approved by an import unless the owner says so")
        tags = {s["file"]: s["tags"] for s in b.data["stills"]}
        self.assertEqual(tags["stills/hero.png"], ["hero"])
        self.assertEqual(tags["stills/turnaround-sheet.png"], ["turnaround"])
        self.assertEqual(tags["stills/expression_grid.jpg"], ["expressions"])
        self.assertEqual(tags["stills/kitchen-scene.png"], ["kitchen-scene"])
        b = characters.approve(home, "host-a", ["hero.png", "kitchen-scene.png"])
        self.assertEqual(len(b.approved()), 2)
        self.assertTrue(b.hero())
        with self.assertRaises(characters.BibleError):
            characters.import_folder(home, src, "host-a")
        with self.assertRaises(characters.BibleError):
            characters.import_folder(home, src, "Bad Id")
        bible = os.path.join(home, "characters", "host-a", "bible.json")
        with open(bible) as f:
            data = json.load(f)
        data["stills"][0]["file"] = "../../outside.png"
        data["voice"] = {"engine": "kokoro", "voice": "not a voice"}
        with open(bible, "w") as f:
            json.dump(data, f)
        with self.assertRaises(characters.BibleError) as e:
            characters.load_bible(home, "host-a")
        self.assertIn("outside", str(e.exception))
        self.assertIn("Kokoro voice", str(e.exception))

    def test_cli_import_list_show(self):
        from foreman import cli
        src = os.path.join(self.tmp, "folder")
        png(os.path.join(src, "hero.png"))
        png(os.path.join(src, "talk.png"))
        ref = wav(os.path.join(self.tmp, "ref.wav"), 2.0)
        home = os.path.join(self.tmp, "home")
        out = io.StringIO()
        old = os.environ.get("FOREMAN_HOME")
        os.environ["FOREMAN_HOME"] = home
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                self.assertEqual(cli.main(["characters", "import", src, "--id", "mascot-b", "--kind", "mascot",
                                           "--look", "a round mascot", "--ref-wav", ref, "--approve"]), 0)
                self.assertEqual(cli.main(["characters", "list"]), 0)
                self.assertEqual(cli.main(["characters", "show", "mascot-b"]), 0)
                self.assertEqual(cli.main(["characters", "show", "nobody"]), 1)
        finally:
            if old is None:
                os.environ.pop("FOREMAN_HOME", None)
            else:
                os.environ["FOREMAN_HOME"] = old
        text = out.getvalue()
        self.assertIn("2 still(s), 2 approved", text)
        self.assertIn("qwen3_clone", text)
        self.assertIn("no character", text)
        self.assertTrue(os.path.exists(os.path.join(home, "characters", "mascot-b", "voice", "reference.wav")))


# ---------------------------------------------------------------- P5

class _Page(http.server.BaseHTTPRequestHandler):
    body = b""
    status = 200
    ctype = "text/html"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(self.status)
        self.send_header("Content-Type", self.ctype)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


class P5Facts(TmpCase):
    def pipeline(self, home):
        return ad_pipeline.AdPipeline(ads_config(), home=home, artifacts_dir=os.path.join(home, "artifacts"),
                                      ffmpeg=FFMPEG, catalog=studio.catalog(False, ads=True),
                                      commercial_models=studio.COMMERCIAL_OK, licenses=studio.LICENSES)

    def ctx(self):
        class C:
            def progress(self, f):
                pass

            def log(self, line):
                pass
        return C()

    def run_facts(self, home, ad):
        from foreman.engine import RunSpec
        p = {"goal": "g", "expected_s": 30, "workdir": f"ads/{ad['id']}", "ad": ad}
        return self.pipeline(home).run_facts(RunSpec("1.1", 1, "ad_facts", studio.LOCAL, p, 30), self.ctx())

    def test_a_facts_file_and_the_end_card_price(self):
        home = make_home(self.tmp)
        art = self.run_facts(home, ad_spec())
        f = paths.read_json(art["path"])
        self.assertIn("9.5", f["numbers"])
        wd = os.path.dirname(art["path"])
        self.assertEqual(paths.read_json(os.path.join(wd, "ad.json"))["picks"]["models"]["voice"], studio.KOKORO)
        stale = ad_spec(id="ad-2", end_card={"price": "$2.99", "url": "example.com", "cta": "x"})
        with self.assertRaises(ContentFailure) as e:
            self.run_facts(home, stale)
        self.assertIn("not in the facts", str(e.exception))

    def test_a_facts_url_is_read_as_text(self):
        home = make_home(self.tmp)
        _Page.body = (b"<html><head><script>var x=1</script></head><body><h1>Pricing</h1><p>Each form costs "
                      b"<b>$9.50</b>&nbsp;all in.</p></body></html>")
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Page)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/pricing"
            art = self.run_facts(home, ad_spec(facts={"url": url}))
            f = paths.read_json(art["path"])
            self.assertIn("Each form costs $9.50 all in.", f["text"])
            self.assertNotIn("var x", f["text"])
            _Page.status = 404
            with self.assertRaises(ContentFailure):
                self.run_facts(home, ad_spec(id="ad-3", facts={"url": url}))
            _Page.status = 503
            with self.assertRaises(RuntimeError):
                self.run_facts(home, ad_spec(id="ad-4", facts={"url": url}))
        finally:
            _Page.status = 200
            srv.shutdown()
            srv.server_close()

    def test_a_commercial_ad_refuses_models_not_cleared(self):
        home = make_home(self.tmp)
        pipe = self.pipeline(home)
        pipe.commercial = frozenset({studio.KOKORO})
        from foreman.engine import RunSpec
        ad = ad_spec()
        with self.assertRaises(ContentFailure) as e:
            pipe.run_facts(RunSpec("1.1", 1, "ad_facts", studio.LOCAL,
                                   {"goal": "g", "expected_s": 30, "workdir": "ads/ad-1", "ad": ad}, 30), self.ctx())
        self.assertIn("commercial", str(e.exception))


# ---------------------------------------------------------------- P6

@needs_ffmpeg
class P6Assembly(TmpCase):

    def setUp(self):
        super().setUp()
        self.ff = assembly.Ffmpeg(FFMPEG, self.tmp)
        self.ff.run(["-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=128x224:r=24:d=3", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "take.mp4"])
        self.ff.run(["-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=3:sample_rate=24000", "-ac", "1",
                     "vo.wav"])
        self.ff.run(["-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=5:sample_rate=48000", "-ac", "2",
                     "music.wav"])
        png(os.path.join(self.tmp, "still.png"), 90, 60)
        self.font = assembly.install_font(assembly.find_font(), os.path.join(self.tmp, "fonts"), "caption")

    def cut(self, W, H, captions=True, max_bytes=40e6, name="out.mp4"):
        segs = [assembly.seg_take(self.ff, "take.mp4", "s0.mp4", 2.0, W, H),
                assembly.seg_still(self.ff, "still.png", "s1.mp4", 1.0, W, H)]
        with open(os.path.join(self.tmp, "price.txt"), "w") as f:
            f.write("$9.50")
        segs.append(assembly.seg_endcard(self.ff, "s2.mp4", 1.0, W, H, "fonts/caption.ttf", [("price.txt", 0.07, 0.45)]))
        assembly.concat(self.ff, segs, "cat.mp4", "list.txt")
        assembly.mix_audio(self.ff, "vo.wav", "music.wav", "mix.wav", 4.0)
        assembly.loudnorm(self.ff, "mix.wav", "audio.wav")
        words = [{"text": "Hello", "start": 0.1, "end": 0.6, "line": 0}, {"text": "there", "start": 0.7, "end": 1.4, "line": 0},
                 {"text": "friend.", "start": 1.5, "end": 2.6, "line": 0}]
        assembly.write_ass(os.path.join(self.tmp, "cap.ass"), words, W, H, self.font["family"], self.font["bold"],
                           end_limit=3.0)
        with open(os.path.join(self.tmp, "label.txt"), "w") as f:
            f.write("AI-generated")
        return assembly.final_encode(self.ff, "cat.mp4", "audio.wav", name, 4.0, max_bytes,
                                     captions="cap.ass" if captions else None, fonts_dir="fonts",
                                     label_file="label.txt", label_font="fonts/caption.ttf", width=W, height=H)

    def test_a_full_size_vertical_cut(self):
        res = self.cut(1080, 1920)
        info = assembly.probe(self.ff, "out.mp4")
        self.assertEqual((info["width"], info["height"]), (1080, 1920))
        self.assertAlmostEqual(info["duration_s"], 4.0, delta=0.1)
        self.assertTrue(info["has_audio"])
        self.assertEqual((info["vcodec"], info["acodec"]), ("h264", "aac"))
        loud = assembly.loudness(self.ff, "out.mp4")
        self.assertAlmostEqual(loud["integrated_lufs"], -14.0, delta=1.0)
        self.assertLessEqual(loud["true_peak_dbtp"], -1.0)
        self.assertLessEqual(res["bytes"], 40e6)
        with open(os.path.join(self.tmp, "out.mp4"), "rb") as f:
            head = f.read(64)
        self.assertEqual(head[4:8], b"ftyp")
        with open(os.path.join(self.tmp, "out.mp4"), "rb") as f:
            data = f.read()
        self.assertLess(data.find(b"moov"), data.find(b"mdat"), "+faststart: the index comes first")

    def test_captions_are_burned_in_and_stay_in_the_safe_zone(self):
        W, H = 270, 480
        self.cut(W, H, captions=True, name="with.mp4")
        self.cut(W, H, captions=False, name="without.mp4")
        a = probe_frame(FFMPEG, os.path.join(self.tmp, "with.mp4"), 1.0, W, H)
        b = probe_frame(FFMPEG, os.path.join(self.tmp, "without.mp4"), 1.0, W, H)
        rows = [sum(abs(x - y) for x, y in zip(a[r * W:(r + 1) * W], b[r * W:(r + 1) * W])) for r in range(H)]
        changed = [r for r, v in enumerate(rows) if v > W * 8]
        self.assertTrue(changed, "the captioned frame differs")
        self.assertGreaterEqual(min(changed) / H, assembly.SAFE_TOP)
        self.assertLessEqual(max(changed) / H, 1 - assembly.SAFE_BOTTOM)
        st = assembly.caption_style(1080, 1920)
        self.assertGreaterEqual(st["margin_v"], 0.2 * 1920)
        self.assertLessEqual(st["margin_v"] + 2 * 1.3 * st["size"], 0.88 * 1920, "two lines stay below the top band")

    def test_the_size_cap_re_encodes_smaller(self):
        orig = assembly.video_kbps
        assembly.video_kbps = lambda *a, **k: 3000  # an estimate that overshoots: the cap must still hold
        try:
            res = self.cut(270, 480, max_bytes=250e3)
        finally:
            assembly.video_kbps = orig
        self.assertLessEqual(res["bytes"], 250e3)
        self.assertGreaterEqual(len(res["attempts"]), 2)
        self.assertGreater(res["attempts"][0]["bytes"], 250e3)

    def test_qa_probes_see_black_and_frozen_video(self):
        self.ff.run(["-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=24:d=1", "-f", "lavfi",
                     "-i", "testsrc=s=64x64:r=24:d=1", "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "bf.mp4"])
        det = assembly.detect(self.ff, "bf.mp4", 0, 2.0)
        self.assertTrue(det["black"])
        self.ff.run(["-loglevel", "error", "-y", "-loop", "1", "-i", "still.png", "-t", "2", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-vf", "scale=64:64", "fz.mp4"])
        self.assertTrue(assembly.detect(self.ff, "fz.mp4", 0, 2.0)["freeze"])
        flat = assembly.gray_frames(self.ff, "bf.mp4", 16, 16, 0, 0.5)
        self.assertLess(assembly.frame_std(flat[0]), 2.0)

    def test_fonts_are_found_and_named(self):
        self.assertTrue(os.path.isfile(assembly.find_font()))
        with self.assertRaises(assembly.FontError):
            old = assembly.FONT_CANDIDATES, assembly.FONT_GLOBS
            assembly.FONT_CANDIDATES, assembly.FONT_GLOBS = (), ()
            try:
                assembly.find_font(["/nope/x.ttf"])
            finally:
                assembly.FONT_CANDIDATES, assembly.FONT_GLOBS = old
        fam, _ = assembly.font_names(self.font["file"])
        self.assertTrue(fam and fam != "caption", "the family comes from the font's name table")

    def test_caption_events_highlight_one_word_at_a_time(self):
        words = [{"text": t, "start": i * 0.4, "end": i * 0.4 + 0.3, "line": 0 if i < 5 else 1}
                 for i, t in enumerate("One two three four, five six seven".split())]
        events = assembly.caption_events(words)
        self.assertEqual(len(events), 7)
        self.assertTrue(all(e[2].count(assembly.HIGHLIGHT) == 1 for e in events))
        chunks = assembly.caption_chunks(words)
        self.assertEqual([len(c) for c in chunks], [4, 1, 2], "phrases break at punctuation and line ends")
        self.assertEqual(assembly._ass_text("a {\\b1}b\\N"), "a (b1)bN")


# ---------------------------------------------------------------- P7

class P7Dispatcher(TmpCase):
    def pipe(self):
        home = make_home(self.tmp)
        return ad_pipeline.AdPipeline(ads_config(), home=home, artifacts_dir=os.path.join(home, "artifacts"),
                                      ffmpeg=FFMPEG)

    def brief(self, reason, kind="ad_shots", attempt=1, detail="x"):
        return {"goal": "g", "source_job_id": 7, "attempt": attempt, "max_attempts": 3,
                "attempted_action": {"kind": kind, "model": studio.LTX, "params": {"workdir": "ads/x", "ops": []},
                                     "on_success": ads.cont("ad_next", "ads/x"), "on_failure": ads.cont("ad_recover", "ads/x")},
                "outcome": {"status": "failed", "reason": reason, "detail": detail}, "context": {"workdir": "ads/x"}}

    def test_recovery_retries_infrastructure_and_gives_up_on_content(self):
        p = self.pipe()
        wd = p.workdir("ads/x")
        d = p.recover(wd, "ads/x", self.brief("crash"), "brain", None, None)
        self.assertEqual(d["verdict"], "retry")
        self.assertEqual(d["submit"][0]["retry_of"], 7)
        self.assertEqual(d["submit"][0]["payload"]["goal"], "g")
        self.assertEqual(p.recover(wd, "ads/x", self.brief("crash", attempt=3), "brain", None, None)["verdict"], "give_up")
        self.assertEqual(p.recover(wd, "ads/x", self.brief("rejected"), "brain", None, None)["verdict"], "give_up")
        self.assertEqual(p.recover(wd, "ads/x", self.brief("cancelled"), "brain", None, None)["verdict"], "give_up")
        with self.assertRaises(ads.AdError):
            p.workdir("../escape")

    def test_the_review_only_accepts_what_it_checked(self):
        rd = ad_pipeline.AdPipeline.review_decision
        self.assertEqual(rd({"verdict": "accept", "facts_ok": True, "character_ok": True}, "", True)["verdict"], "accept")
        self.assertEqual(rd({"verdict": "accept", "facts_ok": True, "character_ok": True}, "", False)["verdict"],
                         "human_review", "no images seen: the character was not checked")
        self.assertEqual(rd({"verdict": "accept", "facts_ok": False, "character_ok": True}, "", True)["verdict"],
                         "human_review")
        self.assertEqual(rd({"verdict": "give_up", "problems": ["states a price the facts do not"]}, "", True)["verdict"],
                         "give_up")
        self.assertEqual(rd(None, "garbage", True)["verdict"], "human_review")

    def test_an_ads_lineage_only_takes_its_own_steps(self):
        import tempfile
        from foreman.store import Store

        class Sup:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            sup = Sup()
            sup.catalog = studio.catalog(False, ads=True)
            sup.store = Store(os.path.join(tmp, "x.db"))

            class Root:
                id = 1
                payload = {"workdir": "ads/a", "ad": {"max_jobs": 3}}
            ok = {"workdir": "ads/a"}
            ads.check_child(sup, Root, "ad_shots", studio.LTX, ok)
            for kind, model, payload in (("render_video", studio.LTX, ok), ("ad_shots", studio.KOKORO, ok),
                                         ("ad_shots", studio.LTX, {"workdir": "ads/b"}), ("ad_shots", "brain", ok)):
                with self.assertRaises(AdmissionError):
                    ads.check_child(sup, Root, kind, model, payload)
            ads.check_child(sup, Root, "ad_stills", studio.COMFYUI, ok)
            sup.store.close()


if __name__ == "__main__":
    unittest.main()
