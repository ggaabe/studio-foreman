"""Red team: the audio gate and pronunciation lexicon (foreman/ads.py, foreman/ad_pipeline.py) and the
reject reason tags on the phone (foreman/phone.py, foreman/ledger.py, foreman/cli.py).

Every test states the behaviour the task asked for; a failing test is a defect, not a test to relax.

  AG1  what the gate accepts as heard: brands mangled by the voice, a brand inside its own address,
       numbers with the wrong unit, currency, order or an extra one, numbers said another way,
       addresses that are not the ad's, empty and garbage transcripts
  AG2  the lexicon: look-alike letters in a display, spellings that add a number, a term inside
       another term's chosen spelling, the voice guard
  AG3  the running ad: the ASR down or answering nothing (the gate never passes silently), bounded
       re-rolls and respellings, no prompt in any gate request (retakes and the mix included), the
       captions keep the display text, music that masks the voice
  AG4  companions: the aligner a failing gate request loaded is still unloaded
  AT1  reason tags: a crash between the verdict and the tag edit, a retract and a new reject on the
       CLI, replays across reject cycles, signed presses in the wrong state, tags written twice,
       oversize tags, the readable reasons and the owner's own note text

Everything is generic (ExampleCo, FormFile, example.com). Nothing here talks to a real server.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from foreman import ad_pipeline, ads, characters, ledger, paths, phone, studio  # noqa: E402
from foreman.ads import gate_tokens, judge_heard, parse_lexicon, spoken_text  # noqa: E402
from foreman.adapters.mlx_serve import MlxServeEngine  # noqa: E402
from foreman.engine import RunSpec  # noqa: E402

import fake_mlx_serve  # noqa: E402
from ad_fixtures import TmpCase, ads_config, make_home, needs_ffmpeg, wait_for, wav  # noqa: E402
from fake_mlx_serve import FakeMlxServe  # noqa: E402
from test_ads_audio_gate import BRAND, HOOK, GateHarness  # noqa: E402
from test_phone import AUDIO_FILES, OWNER  # noqa: E402
from test_phone_reasons import TAG_LABELS, Base, Killed  # noqa: E402

CAT = studio.catalog(False, ads=True)
FORMFILE = {"display": "FormFile", "spoken": "Form File", "respellings": ["Form Fyle"]}


def view(url: str = "example.com") -> dict:
    return {"banned_marks": ["OtherCo"], "banned_words": [], "end_card": {"url": url}}


def lexicon(*entries, url: str = "example.com") -> list:
    problems = []
    out = parse_lexicon(list(entries), "ad.lexicon", problems, view(url))
    assert not problems, problems
    return ads.gate_lexicon({"lexicon": out, "end_card": {"url": url}})


def judge(script: str, heard: str, lex=(), wer_max: float = 0.08) -> dict:
    lex = list(lex)
    return judge_heard(gate_tokens(script, lex), gate_tokens(heard, lex), wer_max, [e["display"] for e in lex])


# ---------------------------------------------------------------- AG1 what the gate accepts as heard

class AG1Brands(unittest.TestCase):
    LINE = "File any form from your phone with FormFile today."

    def heard(self, brand_as: str) -> str:
        return f"file any form from your phone with {brand_as} today"

    def test_a_brand_the_voice_mangled_is_never_heard_as_the_brand(self):
        # the shapes of the live failures: a near word for one half, a vowel changed, an article inserted
        lex = lexicon(FORMFILE)
        for mangled in ("form firm", "form forum", "form a file", "farm file", "foam file", "form fail", "for file",
                        "formfirm.com", "form filed"):
            with self.subTest(mangled):
                res = judge(self.LINE, self.heard(mangled), lex)
                self.assertFalse(res["ok"], res)
                self.assertEqual([m["term"] for m in res["missing_terms"]], ["FormFile"])
        for fine in ("form file", "formfile", "Form-File", "FORM FILE", "form fyle"):  # its spellings
            with self.subTest(fine):
                self.assertTrue(judge(self.LINE, self.heard(fine), lex)["ok"])

    def test_a_plural_the_script_never_said_is_not_the_brand_heard_exactly(self):
        # 'form files' is another phrase (generic paperwork), not the brand; the script has no s
        res = judge(self.LINE, self.heard("form files"), lexicon(FORMFILE))
        self.assertGreater(res["wer"], 0.0, f"'form files' was judged an exact hearing of 'FormFile': {res}")

    def test_the_brand_heard_inside_its_own_address_is_the_brand(self):
        # Qwen3-ASR wrote a correctly said two-word brand as its address ('<brand>.com'), live on the
        # Studio: the brand was said; the gate must not ask for respellings because of it.
        lex = lexicon(dict(BRAND), url="exampleco.com")
        line = "Try ExampleCo today and file your form from your phone in a minute."
        res = judge(line, "try exampleco.com today and file your form from your phone in a minute", lex)
        self.assertEqual(res["missing_terms"], [], "the brand was heard (as its address)")


class AG1Numbers(unittest.TestCase):

    def test_a_percent_heard_as_dollars_is_a_wrong_number(self):
        line = "Get 20% off your first form when you sign up with your phone today."
        for heard in ("get 20 dollars off your first form when you sign up with your phone today",
                      "get twenty dollars off your first form when you sign up with your phone today"):
            with self.subTest(heard):
                res = judge(line, heard)
                self.assertFalse(res["ok"], f"20% said as $20 passed the gate: {res}")

    def test_a_price_heard_as_a_percent_is_a_wrong_number(self):
        res = judge("Pay just $20 for your first form when you sign up with your phone today.",
                    "pay just 20% for your first form when you sign up with your phone today")
        self.assertFalse(res["ok"], f"$20 said as 20% passed the gate: {res}")

    def test_a_price_in_another_currency_is_a_wrong_price(self):
        line = "Pay $9.50 once and file every form from your phone with no printer."
        for heard in ("pay 9.50 euros once and file every form from your phone with no printer",
                      "pay €9.50 once and file every form from your phone with no printer",
                      "pay nine euros fifty once and file every form from your phone with no printer"):
            with self.subTest(heard):
                self.assertFalse(judge(line, heard)["ok"], "dollars said as euros passed the gate")

    def test_two_prices_said_the_wrong_way_round_fail(self):
        line = ("Pay $9.50 for the first form and $5 for the second one, filed from your phone, with no printer, "
                "no stamps and no trip to the post office at all.")
        heard = ("pay $5 for the first form and $9.50 for the second one filed from your phone with no printer no "
                 "stamps and no trip to the post office at all")
        res = judge(line, heard)
        self.assertFalse(res["ok"], f"each price present, but on the wrong form: {res}")

    def test_a_number_the_script_never_says_fails(self):
        line = "Every form costs $9.50 with tax and delivery included for you, today and every day."
        res = judge(line, "every form costs $9.50 with tax and $5 delivery included for you today and every day")
        self.assertFalse(res["ok"], f"the voice added a price ($5) and the gate passed it: {res}")

    def test_one_point_five_million_is_one_and_a_half_million(self):
        line = "Over 1.5 million forms filed for people across the whole country so far."
        res = judge(line, "over one point five million forms filed for people across the whole country so far")
        self.assertTrue(res["ok"], res)

    def test_nine_fifty_is_the_price_nine_dollars_fifty(self):
        # how a price is commonly read aloud ('$9.50' -> 'nine fifty'); README lists it as a known gap
        res = judge("Every form costs $9.50 with tax and delivery included for you.",
                    "every form costs nine fifty with tax and delivery included for you")
        self.assertEqual(res["missing_numbers"], [], res)

    def test_numbers_in_words_digits_and_scales_still_match(self):
        for line, heard in (("It takes 2 minutes to file a form from your phone with no printer needed.",
                             "it takes two minutes to file a form from your phone with no printer needed"),
                            ("Over 10,000 forms filed for people across the whole country so far.",
                             "over ten thousand forms filed for people across the whole country so far"),
                            ("It costs nine dollars and fifty cents with everything included.",
                             "it costs $9.50 with everything included")):
            with self.subTest(heard):
                self.assertTrue(judge(line, heard)["ok"])
        for heard in ("it takes to minutes to file a form from your phone with no printer needed",
                      "it takes 20 minutes to file a form from your phone with no printer needed"):
            with self.subTest(heard):
                self.assertFalse(judge("It takes 2 minutes to file a form from your phone with no printer needed.",
                                       heard)["ok"])


class AG1Addresses(unittest.TestCase):

    def test_the_dot_com_forms_of_the_ads_address(self):
        lex = lexicon()
        line = "Get it done today at example.com."
        for heard in ("get it done today at example.com", "get it done today at example dot com",
                      "Get it done today at Example.Com!", "get it done today at www.example.com",
                      "get it done today at exampledotcom", "get it done today at example-dot-com"):
            with self.subTest(heard):
                self.assertTrue(judge(line, heard, lex)["ok"])
        for heard in ("get it done today at example.co", "get it done today at examples.com",
                      "get it done today at example dot org", "get it done today at example com",
                      "get it done today at example . com"):
            with self.subTest(heard):
                res = judge(line, heard, lex)
                self.assertFalse(res["ok"])
                self.assertEqual([m["term"] for m in res["missing_terms"]], ["example.com"])

    def test_another_domain_that_starts_with_the_address_is_not_the_address(self):
        lex = lexicon()
        line = ("Visit example.com to upload your document, check the details, pay once, and we prepare it and "
                "file it for you today, with no trip and no queue.")
        res = judge(line, line.lower().replace("example.com", "example.com.au").replace(",", ""), lex)
        self.assertFalse(res["ok"], f"'example.com.au' passed as the ad's address example.com: {res}")


class AG1Transcripts(unittest.TestCase):

    def test_empty_and_garbage_transcripts_never_pass(self):
        lex = lexicon(dict(BRAND))
        line = HOOK
        empty = judge(line, "", lex)
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["missing_terms"][0]["heard_as"], "(nothing)")
        for heard in ("   ", ".", "[music]", "♪ ♪ ♪", "ExampleCo", "exampleco exampleco exampleco",
                      "thank you for watching", "lorem ipsum " * 400, "‮" + HOOK[::-1]):
            with self.subTest(heard[:30]):
                self.assertFalse(judge(line, heard, lex)["ok"])


# ---------------------------------------------------------------- AG2 the lexicon

class AG2Lexicon(unittest.TestCase):

    def test_a_display_in_look_alike_letters_is_refused_or_folded(self):
        problems = []
        out = parse_lexicon([{"display": "ЕxampleCo", "spoken": "Example Co"}],  # a Cyrillic E
                            "ad.lexicon", problems, view())
        stored = out[0]["display"] if out else ""
        self.assertTrue(problems or all(ord(c) < 128 for c in stored),
                        f"a display with a Cyrillic letter was kept as {stored!r} (it goes into the gate's reasons, "
                        f"the manifest and the ASR's spelling hints)")

    def test_a_spelling_cannot_make_the_voice_say_a_number(self):
        for entry in ({"display": "ExampleCo", "spoken": "Example Co 50% off"},
                      {"display": "ExampleCo", "spoken": "Example Co", "respellings": ["Example Co for 5 dollars"]},
                      {"display": "$9.50", "spoken": "nineteen fifty"}):
            with self.subTest(entry):
                problems = []
                parse_lexicon([entry], "ad.lexicon", problems, view())
                self.assertTrue(problems, f"accepted a spelling whose numbers are not the display's: {entry}")

    def test_the_voice_guard_refuses_a_spelling_that_adds_a_price(self):
        tmp = self.tmpdir()
        home = make_home(tmp)
        pipe = ad_pipeline.AdPipeline(ads_config(), home=home, artifacts_dir=os.path.join(home, "artifacts"))
        rel = "ads/guard-num"
        wd = pipe.workdir(rel)
        lex = [{"display": "ExampleCo", "spoken": "Example Co, now 50% off", "respellings": []}]
        ad = {"id": "guard-num", "template_id": "t", "seed": 1, "lexicon": lex, "banned_marks": ["OtherCo"],
              "banned_words": [], "end_card": {"url": "example.com", "price": "$9.50"},
              "picks": {"voice": {"engine": "kokoro", "voice": "af_heart"}, "models": {"voice": studio.KOKORO},
                        "voice_track": "lines"}}
        script = {"lines": [{"i": 0, "role": "hook", "text": "Try ExampleCo from your phone."},
                            {"i": 1, "role": "cta", "text": "Go to example.com."}]}
        res = pipe.step_voice(wd, rel, ad, characters.load_bible(home, "mascot-a"), script)
        self.assertEqual(res[0], "fail", f"the voice would say '50% off', which no script line or fact says: {res}")

    def test_a_term_inside_another_terms_chosen_spelling_keeps_that_spelling(self):
        # an owner who also lists a state abbreviation: CO is said 'Colorado' where the script writes CO
        lex = lexicon(dict(BRAND), {"display": "CO", "spoken": "Colorado"})
        said, used = spoken_text("Try ExampleCo in CO today.", lex)
        self.assertEqual(said, "Try Example Co in Colorado today.",
                         "the brand's own spoken spelling 'Example Co' was rewritten")
        self.assertEqual(used.get("ExampleCo"), "Example Co")

    def test_respellings_change_only_the_terms_in_the_voice_text(self):
        lex = lexicon(dict(BRAND))
        said, used = spoken_text("Need a form filed? Try ExampleCo at example.com.", lex, {"0": 1, "1": 1})
        self.assertEqual(said, "Need a form filed? Try Eggs Ample Co at example dot com.")
        self.assertEqual(used, {"ExampleCo": "Eggs Ample Co", "example.com": "example dot com"})

    def tmpdir(self) -> str:
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="foreman-redgate-")
        self.addCleanup(shutil.rmtree, tmp, True)
        return tmp


# ---------------------------------------------------------------- AG3 the running ad

def _read_transcription(handler):
    s = handler.server_state
    n = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(n) if n else b""
    fields, files = fake_mlx_serve.parse_multipart(handler.headers.get("Content-Type", ""), raw)
    with s.lock:
        s.requests.append(("POST", handler.path, dict(fields, file_bytes=len(files.get("file", b"")))))
    return fields


@needs_ffmpeg
class AG3AsrDown(GateHarness):
    """The ASR down, or answering nothing: the gate must never pass silently."""

    def assert_never_passed(self, res):
        self.assertTrue([r for r in self.fake.transcriptions() if r.get("response_format") == "json"],
                        "the gate never asked the ASR: nothing was tested")
        it = self.item(res["item"])
        self.assertNotIn(it["state"], ("review", "accepted"), (it, self.trail()))
        g = self.gate() or {}
        self.assertNotIn(g.get("status"), ("voice_passed", "passed"), g)
        self.assertEqual(self.jobs("ad_timing") + self.jobs("ad_shots") + self.jobs("ad_review"), [], self.trail())
        self.assertFalse(os.path.exists(os.path.join(self.wd, "final", "ad.mp4")))

    def test_an_asr_answering_503_never_lets_the_voice_through(self):
        def down(handler):
            _read_transcription(handler)
            return handler._error(503, "the ASR is not available")
        with mock.patch.object(fake_mlx_serve._Handler, "_transcribe", down):
            res = self.start()
            self.run_until(self.done(res["item"]), timeout=60)
        self.assert_never_passed(res)

    def test_an_asr_answer_without_text_never_lets_the_voice_through(self):
        def no_text(handler):
            _read_transcription(handler)
            return handler._json(200, {"task": "transcribe", "language": "en", "duration": 1.0, "segments": []})
        with mock.patch.object(fake_mlx_serve._Handler, "_transcribe", no_text):
            res = self.start()
            self.run_until(self.done(res["item"]), timeout=60)
        self.assert_never_passed(res)

    def test_an_asr_hearing_nothing_fails_the_ad_and_says_so(self):
        self.fake.asr_text = ""
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=90), self.trail())
        self.assert_never_passed(res)
        it = self.item(res["item"])
        self.assertEqual(it["state"], "failed")
        self.assertIn("audio gate", it["fail_reason"])
        self.assertIn("heard", it["fail_reason"])

    def test_an_asr_down_only_for_the_final_mix_never_lets_the_ad_through(self):
        state = {"timing": False}
        real = fake_mlx_serve._Handler._transcribe

        def mix_down(handler):
            if state["timing"]:
                _read_transcription(handler)
                return handler._error(503, "the ASR is not available")
            return real(handler)

        def watch(audio, fields, _hear=self.fake.hear):
            if fields.get("response_format") == "verbose_json":
                state["timing"] = True  # every request after the word times is the mix gate's
            return _hear(audio, fields)
        self.fake.hear = watch
        with mock.patch.object(fake_mlx_serve._Handler, "_transcribe", mix_down):
            res = self.start()
            self.run_until(self.done(res["item"]), timeout=120)
        it = self.item(res["item"])
        self.assertNotIn(it["state"], ("review", "accepted"), (it, self.trail()))
        self.assertTrue(state["timing"], "the timing step ran")
        self.assertNotEqual((self.gate() or {}).get("status"), "passed")
        self.assertEqual(self.jobs("ad_finish") + self.jobs("ad_review"), [], self.trail())


@needs_ffmpeg
class AG3Bounded(GateHarness):

    def test_respellings_are_tried_once_each_then_the_ad_fails(self):
        many = dict(BRAND, respellings=["Eggs Ample Co", "Egg Sample Co", "Ex Ample Co", "Exam Pull Co"])
        with open(os.path.join(self.home, "templates", "ad.json"), "w") as f:
            from test_ads_audio_gate import brand_template
            t = brand_template()
            t["ad"]["lexicon"] = [many]
            json.dump(t, f)
        self.pipeline.cfg = ads_config(gate_attempts=8)
        self.fake.say = lambda text, body: re.sub(r"(?i)\b(example co|eggs ample co|egg sample co|ex ample co|"
                                                  r"exam pull co)\b", "example cove", text)
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=120), self.trail())
        it = self.item(res["item"])
        self.assertEqual((it["state"], it["fail_class"]), ("failed", "content"), self.trail())
        self.assertIn("never said 'ExampleCo'", it["fail_reason"])
        self.assertIn("example cove", it["fail_reason"])
        takes = self.gate()["units"]["line-0"]["attempts"]
        self.assertEqual([t["forms"]["ExampleCo"] for t in takes],
                         ["Example Co", "Eggs Ample Co", "Egg Sample Co", "Ex Ample Co", "Exam Pull Co"],
                         "each spelling once, in order, then no more")
        hook_takes = [b for b in self.fake.bodies("/v1/audio/speech") if b["input"].startswith("Need a form filed")
                      and "Get it done" not in b["input"]]  # the line, not the full read
        self.assertEqual(len(hook_takes), 5)
        self.assertEqual(self.jobs("ad_timing"), [])


@needs_ffmpeg
class AG3BoundedSeeded(GateHarness):
    clone = True

    def test_seeded_re_rolls_are_bounded_by_the_attempts(self):
        self.pipeline.cfg = ads_config(gate_attempts=6, gate_rerolls=2)
        self.fake.say = lambda text, body: re.sub(r"(?i)\b(example co|eggs ample co)\b", "example cove", text)
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=150), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "failed", self.trail())
        self.assertIn("never said 'ExampleCo'", it["fail_reason"] or "", self.trail())
        g = self.gate()
        takes = g["units"]["line-0"]["attempts"]
        self.assertEqual(len(takes), 6, [t["why"] for t in takes])
        self.assertEqual(len({t["seed"] for t in takes}), 6, "every take a new seed")
        self.assertLessEqual(g["rounds"], 6)
        hook_takes = [b for b in self.fake.bodies("/v1/audio/speech") if b["input"].startswith("Need a form filed")
                      and "Get it done" not in b["input"]]  # the line, not the full read
        self.assertEqual(len(hook_takes), 6)


@needs_ffmpeg
class AG3WholeRun(GateHarness):

    def test_no_prompt_in_any_gate_request_and_the_captions_keep_the_display(self):
        self.fake.mispronounce = {"Example Co": "example cove"}
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        self.assertEqual(self.item(res["item"])["state"], "review", self.trail())
        reqs = self.fake.transcriptions()
        gate = [r for r in reqs if r.get("response_format") == "json"]
        # 5 lines + the track, the retake + the track again, and the final mix
        self.assertEqual(len(gate), 5 + 1 + 1 + 1 + 1, [r.get("response_format") for r in reqs])
        for r in gate:
            self.assertNotIn("prompt", r)
            self.assertNotIn("timestamp_granularities[]", r)
            self.assertNotIn("temperature", r)
            self.assertEqual(r["language"], "en")
        self.assertEqual(reqs[-1].get("response_format"), "json", "the mix is transcribed after the word times")
        with open(os.path.join(self.wd, "final", "captions.ass"), encoding="utf-8") as f:
            ass = f.read()
        self.assertIn("ExampleCo", ass)
        for spoken in ("Example Co", "Eggs Ample Co", "example dot com"):
            self.assertNotIn(spoken, ass, "a spoken spelling reached the captions")
        words = [w["text"] for w in paths.read_json(os.path.join(self.wd, "timing.json"))["words"]]
        self.assertNotIn("Eggs", words)
        self.assertIn("ExampleCo.", words)
        content = json.dumps(self.fake.bodies("/v1/chat/completions")[-1]["messages"])
        said = content.split("Spoken words (the script): ", 1)[1].split("\\n", 1)[0]
        self.assertIn("Try ExampleCo", said, "the brain reads the script as written")
        self.assertNotIn("Eggs Ample", said)
        self.assertIn("audio gate passed", self.item(res["item"])["brain_note"])


@needs_ffmpeg
class AG3Masking(GateHarness):

    def test_music_heard_as_words_inside_the_speech_is_mixed_again_lower(self):
        # The spec: if the mix's WER exceeds the stem's by more than the margin, mix again with the
        # music lower once. Here the music is transcribed as words between every sentence.
        self.fake.asr_unknown.append(lambda t: re.sub(r"([.?!]) ", r"\1 la la la la ", t))
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        mix = self.gate()["mix"]
        first = mix["attempts"][0]
        self.assertGreater(first["wer"], (self.gate()["track"]["wer"] or 0) + 0.05, first)
        self.assertEqual([a["status"] for a in mix["attempts"]], ["failed", "passed"],
                         f"mix WER {first['wer']} against stem WER {self.gate()['track']['wer']}: no re-mix")
        self.assertEqual(len(self.jobs("ad_assemble")), 2)

    def test_a_mix_that_loses_only_the_brand_twice_fails_naming_it(self):
        drop = lambda t: re.sub(r"(?i)example co\b", "", t)  # noqa: E731
        self.fake.asr_unknown.extend([drop, drop])
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual((it["state"], it["fail_class"]), ("failed", "content"), self.trail())
        self.assertIn("never says 'ExampleCo'", it["fail_reason"])
        self.assertEqual(len(self.jobs("ad_assemble")), 2, "one re-mix, not more")
        self.assertEqual(self.jobs("ad_finish") + self.jobs("ad_review"), [])


# ---------------------------------------------------------------- AG4 companions

class AG4Companions(TmpCase):

    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.tmp, "home")
        self.art = os.path.join(self.home, "artifacts")
        os.makedirs(self.art)
        self.fake = FakeMlxServe().start()
        self.fake.asr_loads = (studio.ALIGNER,)
        self.fake._server.handle_error = lambda request, client_address: None  # quiet the deliberate crash
        self.eng = MlxServeEngine(self.fake.url, catalog=CAT, model_dir="/fake/models", artifacts_dir=self.art,
                                  mem_probe=lambda: 200.0, read_poll_s=0.02, unload_wait_s=2.0, connect_timeout_s=2.0,
                                  call_timeout_s=2.0, forbidden=studio.FORBIDDEN, ffmpeg=None,
                                  input_roots=[self.home], max_resident_models=None,
                                  companions={studio.QWEN3_ASR: (studio.ALIGNER,)})
        self.pipe = ad_pipeline.AdPipeline(ads_config(), home=self.home, artifacts_dir=self.art)

    def tearDown(self):
        self.eng.close()
        self.fake.stop()
        super().tearDown()

    def run_gate(self, ops_n=2):
        t = self.eng.load(studio.QWEN3_ASR)
        self.assertTrue(wait_for(lambda: self.eng.poll_load(t).state == "ready"), self.eng.poll_load(t))
        wav(os.path.join(self.art, "ads", "g", "voice", "line-0.wav"), 1.0)
        ops = [self.pipe.gate_op({"language": "en"}, "voice/line-0.wav", f"gate/line-0-a{i}.json", "gate")
               for i in range(ops_n)]
        h = self.eng.start(RunSpec("1.1", 1, "ad_gate", studio.QWEN3_ASR,
                                   {"goal": "g", "expected_s": 30.0, "workdir": "ads/g", "ops": ops}, 30.0))
        self.assertTrue(wait_for(lambda: self.eng.poll(h).status != "running", 20.0))
        return self.eng.poll(h)

    def test_the_aligner_a_failing_gate_request_loaded_is_still_unloaded(self):
        def broken(audio, fields):
            raise RuntimeError("the ASR crashed after loading its aligner")
        self.fake.hear = broken
        poll = self.run_gate()
        self.assertNotEqual(poll.status, "done", "the transcription failed")
        self.assertNotIn(studio.ALIGNER, self.fake.ready, "the aligner the failed request loaded is left resident")
        self.assertIn({"model": studio.ALIGNER}, self.fake.bodies("/v1/unload-model"))
        self.assertIn(studio.QWEN3_ASR, self.eng.resident_models(), "the ASR itself is gpud's to unload")

    def test_the_aligner_is_unloaded_once_and_the_asr_stays(self):
        poll = self.run_gate(ops_n=3)
        self.assertEqual(poll.status, "done", poll)
        self.assertNotIn(studio.ALIGNER, self.fake.ready)
        self.assertEqual(self.fake.bodies("/v1/unload-model").count({"model": studio.ALIGNER}), 1)
        self.assertIn(studio.QWEN3_ASR, self.eng.resident_models())


# ---------------------------------------------------------------- AT1 reason tags

class AT1Crashes(Base):

    def test_a_crash_between_the_reject_and_its_redraw_still_asks_for_the_reasons_after_a_restart(self):
        uid = self.fake.callback(OWNER, OWNER, self.data("draft", self.it["id"], "Reject"),
                                 self.row("draft", self.it["id"])["message_id"])
        with self.crash_in(phone.Phone, "redraw"), self.assertRaises(Killed):
            self.s.poll()
        self.assertEqual(self.ledger_item()["state"], "rejected")
        self.assertEqual(self.s.offset(), uid, "the update is not confirmed: Telegram delivers it again")
        self.restart()
        self.s.step()  # what the restarted service does: outbox, the refresh pass, then the updates
        self.assertEqual(self.counts(), (1, 0, 0), "one verdict")
        self.assertEqual(self.labels(), TAG_LABELS + ["Done", "Undo"],
                         "the refresh pass ran before the redelivered press and the reasons were never asked for")

    def test_a_crash_inside_the_verdict_call_after_the_ledger_write_still_asks_for_the_reasons(self):
        from foreman import cli
        real = cli.apply_verdict

        def dies_after_the_write(*a, **k):
            real(*a, **k)
            raise Killed("kill -9 after the verdict was committed")
        self.fake.callback(OWNER, OWNER, self.data("draft", self.it["id"], "Reject"),
                           self.row("draft", self.it["id"])["message_id"])
        with mock.patch.object(cli, "apply_verdict", dies_after_the_write), self.assertRaises(Killed):
            self.s.poll()
        self.restart()
        self.s.step()
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertIn("Done", self.labels(), "no reason buttons after the restart")


class AT1Elsewhere(Base):

    def test_a_retract_and_a_new_reject_on_the_cli_end_the_phone_picking(self):
        self.tap("Reject")
        self.assertIn("Done", self.labels())
        jid = str(self.it["job_id"])
        self.assertEqual(self.cli("verdict", jid, "retract")[0], 0)
        self.assertEqual(self.cli("verdict", jid, "reject")[0], 0)
        self.s.step()  # the refresh pass
        if "Voice" in self.labels():  # the old picking survived: its buttons still record
            self.tap("Voice")
            self.tap("Done")
        it = self.ledger_item()
        self.assertIsNone(it["verdict_tags"], "the picking of a retracted reject attached its tags to the new one")
        self.assertEqual(self.events("verdict_tagged"), [])
        self.assertNotIn("Done", self.labels())

    def test_a_retract_on_the_cli_before_any_refresh_makes_done_and_tags_do_nothing(self):
        self.tap("Reject")
        self.tap("Voice")
        drawn = dict(self.fake.buttons(self.msg("draft", self.it["id"])["markup"]))
        self.assertEqual(self.cli("verdict", str(self.it["job_id"]), "retract")[0], 0)
        for label in ("Pacing", "Done"):  # no step(): the message still shows the picking
            with self.subTest(label):
                ans = self.press(self.s, "draft", self.it["id"], None, data=drawn[label])
                self.assertNotIn("Recorded", ans.get("text", ""))
        it = self.ledger_item()
        self.assertEqual((it["state"], it["verdict_tags"]), ("review", None))
        self.assertEqual(self.events("verdict_tagged"), [])


class AT1Replays(Base):

    def test_buttons_from_an_earlier_reject_cycle_record_nothing(self):
        self.tap("Reject")
        self.tap("Voice")
        first = self.fake.buttons(self.msg("draft", self.it["id"])["markup"])
        self.tap("Undo")
        self.tap("Reject")
        self.tap("Music")
        before = self.counts()
        for text, data in first:
            with self.subTest(text):
                self.press(self.s, "draft", self.it["id"], None, data=data)
        self.assertEqual(self.counts(), before, "a replay recorded something")
        self.assertEqual(self.ledger_item()["state"], "rejected", "an old Undo retracted the new reject")
        self.assertEqual(self.picks(), ["music"], "an old toggle changed the new picking")
        self.tap("Done")
        self.assertEqual(self.ledger_item()["verdict_tags"], ["music"])

    def test_every_button_of_the_picking_replayed_after_done_records_nothing(self):
        self.tap("Reject")
        self.tap("Voice")
        drawn = self.fake.buttons(self.msg("draft", self.it["id"])["markup"])
        self.tap("Done")
        before = self.counts()
        for text, data in drawn:
            with self.subTest(text):
                self.press(self.s, "draft", self.it["id"], None, data=data)
        self.assertEqual(self.counts(), before)
        it = self.ledger_item()
        self.assertEqual((it["state"], it["verdict_tags"], it["verdict_note"]), ("rejected", ["voice"],
                                                                                   "reasons: voice"))
        self.assertEqual(self.labels(), ["Undo"])

    def test_signed_tag_and_done_presses_in_the_wrong_state_record_nothing(self):
        # Correctly signed data (as if the key leaked into a bug): tags only ever follow a phone reject
        b = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        c = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        self.s.step()
        self.tap("Approve")                                   # accepted
        self.assertEqual(self.cli("verdict", str(b["job_id"]), "reject")[0], 0)  # rejected on the CLI
        self.s.step()
        for it in (self.it, b, c):                            # accepted, CLI-rejected, in review
            row = self.row("draft", it["id"])
            for action, letter in (("t", "b"), ("k", None)):
                with self.subTest(item=it["id"], action=action):
                    data = phone.make_data(self.s.key, OWNER, action, "d", it["id"], row["issue"], letter)
                    self.press(self.s, "draft", it["id"], None, data=data)
                    row = self.row("draft", it["id"])
        self.assertEqual(self.events("verdict_tagged"), [])
        for it in (self.it, b, c):
            self.assertIsNone(ledger.item(self.store, it["id"])["verdict_tags"])
        self.assertEqual(ledger.item(self.store, c["id"])["state"], "review")


class AT1Storage(Base):

    def test_tags_given_twice_replace_and_never_accumulate(self):
        self.tap("Reject")
        self.tap("Voice")
        self.tap("Done")
        jid = str(self.it["job_id"])
        for tags in ("pacing", "pacing", "Pacing, voice", "voice,pacing"):
            code, out = self.cli("verdict", jid, "reject", "--tags", tags)
            self.assertEqual(code, 0, out)
        it = self.ledger_item()
        self.assertEqual(it["verdict_tags"], ["voice", "pacing"])
        self.assertEqual(it["verdict_note"], "reasons: voice, pacing")
        self.assertEqual(it["verdict_note"].count("reasons:"), 1)
        self.assertEqual(len(self.events("verdict_tagged")), 3, "the same tags again log nothing")
        self.assertEqual(len(self.events("verdict_recorded")), 1)
        self.assertEqual(ledger.reject_tag_counts(self.store)["t.speech"],
                         {"rejected": 1, "tagged": 1, "tags": {"voice": 1, "pacing": 1}})

    def test_an_oversize_tags_argument_is_refused_with_a_bounded_message(self):
        huge = ",".join(f"bogus{i}" for i in range(20000))
        with self.assertRaises(ledger.LedgerError) as e:
            ledger.parse_tags(huge)
        self.assertLess(len(str(e.exception)), 2000, f"a {len(str(e.exception))}-character error message")
        code, out = self.cli("verdict", str(self.it["job_id"]), "reject", "--tags", huge)
        self.assertEqual(code, 2)
        self.assertLess(len(out), 4000, f"the CLI printed {len(out)} characters")
        self.assertEqual(self.ledger_item()["state"], "review")
        many = ",".join(["voice"] * 50000 + ["Lip-sync"] * 1000)
        self.assertEqual(ledger.parse_tags(many), ["voice", "lipsync"])

    def test_tampered_stored_tags_never_reach_the_ledger_or_the_note(self):
        self.tap("Reject")
        with self.store.tx():
            self.store.conn.execute("UPDATE phone_messages SET tags = ? WHERE kind = 'draft' AND item = ?",
                                    (json.dumps(["voice", "reasons: script | x" * 100]), self.it["id"]))
        self.tap("Done")
        it = self.ledger_item()
        self.assertIsNone(it["verdict_tags"])
        self.assertNotIn("x", it["verdict_note"] or "")
        with self.store.tx():
            self.store.conn.execute("UPDATE plan_items SET verdict_tags = ? WHERE id = ?",
                                    (json.dumps(["voice", "voice", "<b>", 7, None, "pacing"]), self.it["id"]))
        self.assertEqual(self.ledger_item()["verdict_tags"], ["voice", "pacing"])
        self.assertEqual(ledger.reject_tag_counts(self.store)["t.speech"]["tags"], {"voice": 1, "pacing": 1})

    def test_the_owners_own_note_that_reads_like_reasons_is_never_deleted_by_a_retag(self):
        self.tap("Reject")
        mid = self.row("draft", self.it["id"])["message_id"]
        out = self.say(self.s, "reasons: voice", reply_to=mid)  # the owner's own words, as a reply note
        self.assertIn("Note added", out[-1]["params"]["text"])
        self.tap("Voice")
        self.tap("Done")
        code, out = self.cli("verdict", str(self.it["job_id"]), "reject", "--tags", "music")
        self.assertEqual(code, 0, out)
        note = self.ledger_item()["verdict_note"] or ""
        self.assertEqual(note.split(" | "), ["reasons: voice", "reasons: music"],
                         f"the owner's reply note was deleted with the old reasons: {note!r}")


if __name__ == "__main__":
    unittest.main()
