"""The audio gate and the pronunciation lexicon.

A brand written as one word can be read by a voice as two other words, and a brain that reviews
the finished ad cannot hear it. The gate listens instead: every voice line, the whole voice track
and the final mix are transcribed free-running (Qwen3-ASR, no prompt) and must say every lexicon
term and every number of the script, within a word error rate. The fake MLX-Serve's voice can be
made to say a wrong word, and its ASR hears what each WAV holds.

  G1  the lexicon: validated at template load, merged over the owner's file, sent to the voice as
      the spoken spelling while the script, captions and end card keep the display text
  G2  what counts as heard: case, punctuation, numbers in words or digits, 'dot com' and '.com',
      a term heard as any of its spellings, the word error rate
  G3  the gate in the running ad: a mispronounced brand caught and fixed by a respelling, a seeded
      voice re-rolled first, a persistent failure failing the ad with what was heard, numbers,
      addresses, the voice track, the full read, the record in gate.json and the manifest
  G4  the final mix: music that masks the voice is mixed again lower once, then fails the ad
  G5  the gate's transcriptions: no prompt, no word times, the aligner they loaded unloaded
  G6  the review never accepts an ad whose gate did not pass
"""
from __future__ import annotations

import json
import os
import re
import unittest

from foreman import ad_pipeline, ads, paths, studio
from foreman.adapters.local import InfraFailure
from foreman.adapters.mlx_serve import MlxServeEngine
from foreman.ads import gate_tokens, judge_heard, parse_lexicon, spoken_text
from foreman.engine import RunSpec
from foreman.templates import TemplateError, parse_template

try:
    from ad_fixtures import REVIEW_OK, SCRIPT, TmpCase, ads_config, make_home, needs_ffmpeg, template_data, wait_for, wav
    from fake_mlx_serve import FakeMlxServe
    from test_ads_e2e import AdHarness
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import (REVIEW_OK, SCRIPT, TmpCase, ads_config, make_home, needs_ffmpeg, template_data,
                                   wait_for, wav)
    from tests.fake_mlx_serve import FakeMlxServe
    from tests.test_ads_e2e import AdHarness

CAT = studio.catalog(False, ads=True)
BRAND = {"display": "ExampleCo", "spoken": "Example Co", "respellings": ["Eggs Ample Co"]}
HOOK = "Need a form filed without leaving your desk? Try ExampleCo."
SCRIPT_BRAND = dict(SCRIPT, hook=HOOK)
VIEW = {"banned_marks": ["OtherCo"], "banned_words": [], "end_card": {"url": "example.com"}}


def lex(*entries) -> list:
    problems = []
    out = parse_lexicon(list(entries), "ad.lexicon", problems, VIEW)
    assert not problems, problems
    return ads.gate_lexicon({"lexicon": out, "end_card": {"url": "example.com"}})


def judge(script: str, heard: str, lexicon=(), wer_max: float = 0.08) -> dict:
    lexicon = list(lexicon)
    return judge_heard(gate_tokens(script, lexicon), gate_tokens(heard, lexicon), wer_max,
                       [e["display"] for e in lexicon])


# ---------------------------------------------------------------- G1 the lexicon

class G1Lexicon(unittest.TestCase):

    def test_a_template_lexicon_is_validated_when_it_loads(self):
        good = template_data()
        good["ad"] = dict(good["ad"], lexicon=[BRAND, {"display": "example.com", "spoken": "example dot com",
                                                        "ipa": "ɪɡˈzæmpəl", "arpabet": "IH0 G Z AE1 M P AH0 L"}])
        t = parse_template(good)
        self.assertEqual([e["display"] for e in t.ad["lexicon"]], ["ExampleCo", "example.com"])
        self.assertEqual(t.ad["lexicon"][0]["respellings"], ["Eggs Ample Co"])
        bad = template_data()
        bad["ad"] = dict(bad["ad"], lexicon=[
            BRAND,
            {"display": "Example-Co", "spoken": "Example Co"},            # the same term twice
            {"display": "Markup", "spoken": "Mark *up*"},                 # markup the voice would read
            {"display": "Thing", "respellings": ["OtherCo"]},             # another brand in the voice
            {"display": "Site", "spoken": "rival dot com"},               # another address
            {"display": "Deal", "spoken": "guaranteed deal"},             # a claim
            {"display": "X1", "arpabet": "not phonemes"},
            {"display": "Y1", "respellings": ["a", "b", "c", "d", "e", "f", "g"]},
            {"display": "Z1", "voice": "af_heart"},                       # not a lexicon field
            {"spoken": "no display"}])
        with self.assertRaises(TemplateError) as e:
            parse_template(bad)
        msg = str(e.exception)
        for part in ("markup", "listed twice", "another brand", "not the ad's own URL", "a guarantee", "ARPAbet",
                     "at most 6 spellings", "with a display"):
            self.assertIn(part, msg)
        many = template_data()
        many["ad"] = dict(many["ad"], lexicon=[{"display": f"Term{i}"} for i in range(41)])
        with self.assertRaises(TemplateError):
            parse_template(many)

    def test_the_voice_gets_the_spoken_spelling_and_the_script_keeps_the_display(self):
        lexicon = lex(BRAND)
        said, used = spoken_text("Try ExampleCo today. exampleco does it. ExampleCorp is not it.", lexicon)
        self.assertEqual(said, "Try Example Co today. Example Co does it. ExampleCorp is not it.",
                         "whole words, any case; a longer word is another word")
        self.assertEqual(used, {"ExampleCo": "Example Co"})
        cyr = "Try ЕxаmpleCo today."   # a Cyrillic E and a: they read as ExampleCo
        self.assertEqual(spoken_text(cyr, lexicon)[0], "Try Example Co today.")
        self.assertEqual(spoken_text("Try Example-Co", lexicon)[0], "Try Example Co", "written with a hyphen")
        self.assertEqual(spoken_text("Try ExampleCo", lexicon, {"0": 1})[0], "Try Eggs Ample Co")
        # the address is a term too: said as written, then spelled out; a brand inside it keeps its spelling
        lexicon = ads.gate_lexicon({"lexicon": [BRAND], "end_card": {"url": "exampleco.com"}})
        self.assertEqual(lexicon[-1]["source"], "address")
        self.assertEqual(spoken_text("Visit exampleco.com now.", lexicon)[0], "Visit Example Co.com now.")
        self.assertEqual(spoken_text("Visit exampleco.com now.", lexicon, {"1": 1})[0],
                         "Visit Example Co dot com now.")
        self.assertEqual(ads.advance_spelling(lexicon, {"1": 1}, [1]), {"1": 1, "0": 1},
                         "an address with no spelling left tries the next spelling of the brand inside it")
        self.assertIsNone(ads.advance_spelling(lexicon, {"1": 1, "0": 1}, [1]))

    def test_the_owners_file_is_merged_under_the_template(self):
        tmp = self.tmpdir()
        home = make_home(tmp)
        pipe = ad_pipeline.AdPipeline(ads_config(), home=home, artifacts_dir=os.path.join(home, "artifacts"))
        ad = parse_template(dict(template_data(), ad=dict(template_data()["ad"], lexicon=[BRAND]))).ad
        with open(os.path.join(home, "lexicon.json"), "w") as f:
            json.dump({"entries": [{"display": "exampleco", "spoken": "Ex Ample Co"},
                                   {"display": "FormFile", "spoken": "Form File"}]}, f)
        merged = pipe.lexicon(ad)
        self.assertEqual([(e["display"], e["spoken"], e["source"]) for e in merged],
                         [("ExampleCo", "Example Co", "template"), ("FormFile", "Form File", "owner")],
                         "the template's spelling of a term wins")
        with open(os.path.join(home, "lexicon.json"), "w") as f:
            json.dump([{"display": "Thing", "spoken": "OtherCo"}], f)
        with self.assertRaises(InfraFailure) as e:
            pipe.lexicon(ad)
        self.assertEqual(e.exception.reason, "engine_unconfigured", "the owner's file is configuration")
        self.assertIn("another brand", str(e.exception))
        with open(os.path.join(home, "lexicon.json"), "w") as f:
            f.write("{not json")
        with self.assertRaises(InfraFailure):
            pipe.lexicon(ad)
        os.remove(os.path.join(home, "lexicon.json"))
        self.assertEqual([e["source"] for e in pipe.lexicon(ad)], ["template"], "no file, no owner entries")

    def test_a_spelling_can_never_make_the_voice_say_what_the_script_may_not(self):
        tmp = self.tmpdir()
        home = make_home(tmp)
        from foreman import characters
        pipe = ad_pipeline.AdPipeline(ads_config(), home=home, artifacts_dir=os.path.join(home, "artifacts"))
        rel = "ads/guard"
        wd = pipe.workdir(rel)
        problems = []
        lexicon = parse_lexicon([{"display": "Fast", "spoken": "tomorrow"}], "lexicon", problems, VIEW)
        self.assertEqual(problems, [], "one word alone is no claim")
        ad = {"id": "guard", "template_id": "t", "seed": 1, "lexicon": lexicon, "banned_marks": ["OtherCo"],
              "banned_words": [], "end_card": {"url": "example.com"},
              "picks": {"voice": {"engine": "kokoro", "voice": "af_heart"}, "models": {"voice": studio.KOKORO},
                        "voice_track": "lines"}}
        script = {"lines": [{"i": 0, "role": "hook", "text": "Your form is delivered fast."},
                            {"i": 1, "role": "cta", "text": "Go to example.com."}]}
        res = pipe.step_voice(wd, rel, ad, characters.load_bible(home, "mascot-a"), script)
        self.assertEqual(res[0], "fail")
        self.assertIn("a delivery promise", res[1])

    def test_the_example_owner_file_holds(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        entries, problems = ads.read_lexicon_file(os.path.join(repo, "examples", "ads", "lexicon.json"), VIEW)
        self.assertEqual(problems, [])
        self.assertEqual([e["display"] for e in entries], ["FormFile", "ExampleCo"])
        with open(os.path.join(repo, "examples", "ads", "templates", "ads.example-service.v1.json"),
                  encoding="utf-8") as f:
            t = parse_template(json.load(f))
        merged = ads.merge_lexicon(t.ad["lexicon"], entries)
        self.assertEqual([(e["display"], e["source"]) for e in merged],
                         [("ExampleCo", "template"), ("example.com", "template"), ("FormFile", "owner")])

    def tmpdir(self) -> str:
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="foreman-gate-")
        self.addCleanup(shutil.rmtree, tmp, True)
        return tmp


# ---------------------------------------------------------------- G2 what counts as heard

class G2Heard(unittest.TestCase):

    def test_numbers_in_words_or_digits_are_the_same_number(self):
        line = "It costs nine dollars and fifty cents, twenty percent less, for two million forms."
        self.assertTrue(judge(line, "it costs $9.50, 20% less, for 2,000,000 forms")["ok"])
        self.assertTrue(judge(line, "It costs 9.50 dollars, twenty per cent less, for 2 million forms.")["ok"])
        res = judge(line, "it costs nine dollars and fifteen cents twenty percent less for two million forms")
        self.assertFalse(res["ok"])
        self.assertEqual(res["missing_numbers"], [{"number": "$9.50", "heard_as": "$9.15"}])

    def test_dot_com_and_dot_are_the_same_address(self):
        lexicon = lex()
        line = "Get it done today at example.com."
        for heard in ("get it done today at example.com", "Get it done today at example dot com.",
                      "get it done today at Example.com!", "get it done today at www.example.com"):
            self.assertTrue(judge(line, heard, lexicon)["ok"], heard)
        res = judge(line, "get it done today at example com", lexicon)
        self.assertFalse(res["ok"])
        self.assertEqual(res["missing_terms"][0]["term"], "example.com")

    def test_a_term_heard_as_any_of_its_spellings_is_the_term(self):
        lexicon = lex(BRAND)
        for heard in ("try example co", "Try ExampleCo.", "try example-co", "try eggs ample co", "TRY EXAMPLECO"):
            self.assertTrue(judge("Try ExampleCo.", heard, lexicon)["ok"], heard)
        res = judge("Try ExampleCo.", "try example cove", lexicon)
        self.assertFalse(res["ok"])
        self.assertEqual(res["missing_terms"], [{"idx": 0, "term": "ExampleCo", "heard_as": "example cove"}])
        self.assertFalse(judge("Try ExampleCo and ExampleCo.", "try example co and", lexicon)["ok"],
                         "said twice, heard once")

    def test_the_word_error_rate_and_harmless_differences(self):
        long = "upload your document check the details pay for it and we prepare it and file it for you today"
        self.assertTrue(judge(long, long.replace("details", "detail"))["ok"], "1 of 20 words: 0.05")
        self.assertFalse(judge("We file it for you.", "we filed it for you")["ok"], "1 of 5 words: 0.2")
        self.assertTrue(judge("We file it for you.", "we filed it for you", wer_max=0.25)["ok"], "configurable")
        self.assertTrue(judge("It's easy, you'll see: send an e-mail, OK?", "it is easy you will see send an email okay")
                        ["ok"])
        self.assertEqual(judge("Hello there.", "")["wer"], 1.0)
        garbage = judge("Hello there.", "spam " * 5000)
        self.assertFalse(garbage["ok"])
        self.assertLessEqual(garbage["errors"]["ins"], 4 * 2 + 40, "an endless answer is cut, not aligned whole")


# ---------------------------------------------------------------- G3 the gate in a running ad

def brand_template(**ad_over) -> dict:
    t = template_data()
    t["ad"] = dict(t["ad"], lexicon=[BRAND], **ad_over)
    return t


class GateHarness(AdHarness):
    script = SCRIPT_BRAND

    def setUp(self):
        super().setUp()
        with open(os.path.join(self.home, "templates", "ad.json"), "w") as f:
            json.dump(brand_template(), f)

    def start(self, review=REVIEW_OK):
        self.fake.chat_replies.extend([json.dumps(self.script), review])
        self.res = self.enqueue()
        self.wd = self.res["workdir"]
        return self.res

    def gate(self) -> dict:
        return paths.read_json(os.path.join(self.wd, "gate.json"))

    def until_timing(self):
        ok = self.run_until(lambda: os.path.isfile(os.path.join(self.wd, "timing.json"))
                            or self.item(self.res["item"])["state"] != "queued", timeout=90)
        self.assertTrue(ok, self.trail())

    def speech_inputs(self) -> list:
        return [b["input"] for b in self.fake.bodies("/v1/audio/speech")]


@needs_ffmpeg
class G3Respelling(GateHarness):

    def test_a_mispronounced_brand_is_caught_and_fixed_by_its_next_respelling(self):
        self.fake.mispronounce = {"Example Co": "example cove"}
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "review", (it, self.trail()))
        self.assertEqual(it["brain_verdict"], "accept")
        self.assertIn("audio gate passed", it["brain_note"])
        self.assertIn("1 retake", it["brain_note"])
        g = self.gate()
        self.assertEqual(g["status"], "passed")
        takes = g["units"]["line-0"]["attempts"]
        self.assertEqual([t["wav"] for t in takes], ["voice/line-0.wav", "voice/line-0-a1.wav"])
        self.assertEqual([t["spoken"] for t in takes], [HOOK.replace("ExampleCo", "Example Co"),
                                                        HOOK.replace("ExampleCo", "Eggs Ample Co")])
        self.assertFalse(takes[0]["ok"])
        self.assertEqual(takes[0]["missing_terms"], [{"term": "ExampleCo", "heard_as": "example cove"}])
        self.assertIn("example cove", takes[0]["heard"])
        self.assertTrue(takes[1]["ok"])
        self.assertIn("next spelling", takes[1]["why"])
        self.assertTrue(all(len(u["attempts"]) == 1 for k, u in g["units"].items() if k != "line-0"),
                        "only the line that failed was said again")
        # the voice was given the spoken spellings; the script, the captions and the timing keep the display
        self.assertIn(HOOK.replace("ExampleCo", "Eggs Ample Co"), self.speech_inputs())
        self.assertNotIn(HOOK, self.speech_inputs())
        script = paths.read_json(os.path.join(self.wd, "script.json"))
        self.assertEqual(script["lines"][0]["text"], HOOK)
        timing = paths.read_json(os.path.join(self.wd, "timing.json"))
        self.assertIn("ExampleCo.", [w["text"] for w in timing["words"]])
        with open(os.path.join(self.wd, "final", "captions.ass")) as f:
            self.assertIn("ExampleCo", f.read())
        track = paths.read_json(os.path.join(self.wd, "voice", "track.json"))
        self.assertEqual(track["from"][0], "voice/line-0-a1.wav", "the track is made of the takes that passed")
        # every take is on disk and in the manifest
        for t in takes:
            self.assertTrue(os.path.isfile(os.path.join(self.wd, t["wav"])))
            self.assertTrue(os.path.isfile(os.path.join(self.wd, t["transcript"])))
        manifest = paths.read_json(os.path.join(self.wd, "manifest.json"))
        self.assertEqual(manifest["gate"]["status"], "passed")
        self.assertEqual(len(manifest["gate"]["units"]["line-0"]["attempts"]), 2)
        self.assertEqual(manifest["gate"]["mix"]["status"], "passed")
        self.assertEqual(manifest["review"]["gate"]["status"], "passed")
        self.assertEqual(manifest["gate"]["lexicon"][0]["display"], "ExampleCo")
        kinds = [j.kind for j in self.jobs() if j.requires != "brain"]
        self.assertEqual(kinds.count("ad_voice"), 2)
        self.assertEqual(kinds.count("ad_gate"), 3, "the lines and track, the retake and track, the mix")

    def test_a_brand_never_heard_fails_the_ad_with_what_was_heard(self):
        self.fake.mispronounce = {"Example Co": "example cove", "Eggs Ample Co": "eggs ample cove"}
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=90), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "failed", self.trail())
        self.assertEqual(it["fail_class"], "content")
        self.assertIn("never said 'ExampleCo'", it["fail_reason"])
        self.assertIn("eggs ample cove", it["fail_reason"])
        g = self.gate()
        self.assertEqual(g["status"], "failed")
        self.assertIn("'Example Co', 'Eggs Ample Co'", g["reason"], "the spellings tried")
        self.assertEqual(len(g["units"]["line-0"]["attempts"]), 2, "every spelling once (Kokoro reads the same "
                                                                   "words the same way: no re-roll)")
        self.assertEqual(self.jobs("ad_timing") + self.jobs("ad_stills") + self.jobs("ad_shots"), [],
                         "nothing renders after the voice fails")
        self.assertFalse(os.path.exists(os.path.join(self.wd, "final", "ad.mp4")), "the owner never gets the ad")

    def test_the_attempts_are_bounded(self):
        self.pipeline.cfg = ads_config(gate_attempts=1)
        self.fake.mispronounce = {"Example Co": "example cove"}
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=90), self.trail())
        self.assertEqual(self.item(res["item"])["state"], "failed")
        self.assertEqual(len(self.gate()["units"]["line-0"]["attempts"]), 1)

    def test_a_number_said_wrong_fails_without_a_pointless_retake(self):
        self.fake.mispronounce = {"fifty": "fifteen"}
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=90), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "failed", self.trail())
        self.assertIn("never said the number $9.50", it["fail_reason"])
        self.assertIn("$9.15", it["fail_reason"])
        self.assertEqual(len(self.gate()["units"]["line-3"]["attempts"]), 1)

    def test_numbers_and_addresses_written_the_asrs_way_pass(self):
        self.fake.asr_rewrite = [(r"nine dollars and fifty cents", "$9.50"), (r"example\.com", "Example dot com")]
        self.start()
        self.until_timing()
        g = self.gate()
        self.assertEqual(g["status"], "voice_passed", g.get("reason"))
        self.assertEqual(g["units"]["line-3"]["attempts"][0]["heard"], "it costs $9.50 with everything included.")
        self.assertEqual(g["rounds"], 0, "no retake")

    def test_an_address_heard_without_its_dot_is_said_again_spelled_out(self):
        self.fake.asr_rewrite = [(r"example\.com", "example com")]
        self.start()
        self.until_timing()
        g = self.gate()
        self.assertEqual(g["status"], "voice_passed", g.get("reason"))
        takes = g["units"]["line-4"]["attempts"]
        self.assertEqual([t["spoken"] for t in takes], ["Get it done today at example.com.",
                                                        "Get it done today at example dot com."])
        self.assertEqual(takes[0]["missing_terms"][0]["term"], "example.com")

    def test_a_line_the_whole_track_loses_is_said_again(self):
        # each line alone is heard right; in the whole track the address loses its dot
        self.fake.asr_rewrite = lambda t: t.replace("example.com", "example com") if len(t) > 120 else t
        self.start()
        self.until_timing()
        g = self.gate()
        self.assertEqual(g["status"], "voice_passed", g.get("reason"))
        takes = g["units"]["line-4"]["attempts"]
        self.assertEqual(len(takes), 2)
        self.assertTrue(takes[0]["heard"] and "in_track" in takes[0], takes[0])
        self.assertIn("in the voice track", takes[1]["why"])
        self.assertEqual([r["status"] for r in g["track"]["attempts"]], ["failed", "passed"])


@needs_ffmpeg
class G3SeededVoice(GateHarness):
    clone = True

    def test_a_seeded_voice_is_re_rolled_before_a_respelling(self):
        first = {}

        def say(text, body):
            seed = first.setdefault(text, body.get("seed"))
            return text.replace("Example Co", "example cove") if body.get("seed") == seed else text
        self.fake.say = say
        self.start()
        self.until_timing()
        g = self.gate()
        self.assertTrue(g["seeded"])
        self.assertEqual(g["status"], "voice_passed", g.get("reason"))
        takes = g["units"]["line-0"]["attempts"]
        self.assertEqual(len(takes), 2)
        self.assertEqual(takes[0]["spoken"], takes[1]["spoken"], "the same spelling")
        self.assertNotEqual(takes[0]["seed"], takes[1]["seed"])
        self.assertEqual(takes[1]["why"], "a new seed")
        bodies = [b for b in self.fake.bodies("/v1/audio/speech") if "Example Co" in b["input"] and "?" in b["input"]
                  and len(b["input"]) < 80]
        self.assertEqual([b["seed"] for b in bodies], [takes[0]["seed"], takes[1]["seed"]])


@needs_ffmpeg
class G3FullRead(GateHarness):

    def config(self):
        return ads_config(voice_track="full")

    def test_the_full_read_is_the_track_and_is_said_again(self):
        self.fake.mispronounce = {"Example Co": "example cove"}
        self.start()
        self.until_timing()
        g = self.gate()
        self.assertEqual(list(g["units"]), ["full"])
        self.assertEqual(g["status"], "voice_passed", g.get("reason"))
        self.assertEqual([t["wav"] for t in g["units"]["full"]["attempts"]], ["voice/full.wav", "voice/full-a1.wav"])
        self.assertEqual(paths.read_json(os.path.join(self.wd, "voice", "track.json"))["from"], ["voice/full-a1.wav"])


# ---------------------------------------------------------------- G4 the final mix

@needs_ffmpeg
class G4Masking(GateHarness):

    @staticmethod
    def masked(text: str) -> str:
        """What music louder than the voice leaves of the mix: a whole sentence and the price lost."""
        return re.sub(r"(?i)nine dollars and fifty cents|upload your document and check the details in a minute\.",
                      "", text)

    def test_music_that_masks_the_voice_is_mixed_again_lower(self):
        self.fake.asr_unknown.append(self.masked)  # the first mix loses words; the second is heard whole
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "review", self.trail())
        mix = self.gate()["mix"]
        self.assertEqual([a["status"] for a in mix["attempts"]], ["failed", "passed"])
        self.assertEqual([a["music_db"] for a in mix["attempts"]], [-12.0, -20.0])
        self.assertGreater(mix["attempts"][0]["lost"], mix["attempts"][0]["stem_lost"] + 0.05)
        asm = paths.read_json(os.path.join(self.wd, "final", "assembly.json"))
        self.assertEqual((asm["music_db"], asm["mix_round"]), (-20.0, 1))
        self.assertEqual(len(self.jobs("ad_assemble")), 2)
        self.assertEqual(self.gate()["status"], "passed")

    def test_music_that_still_masks_after_the_re_mix_fails_the_ad(self):
        self.fake.asr_unknown.extend([self.masked, self.masked])
        res = self.start()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "failed", self.trail())
        self.assertEqual(it["fail_class"], "content")
        self.assertIn("music masks the voice", it["fail_reason"])
        self.assertEqual(self.jobs("ad_finish"), [], "no QA, no review: the owner never gets it")
        self.assertEqual(self.jobs("ad_review"), [])


# ---------------------------------------------------------------- G5 the gate's own transcriptions

@needs_ffmpeg
class G5Transcriptions(GateHarness):

    def test_no_prompt_and_no_word_times_in_the_gates_transcriptions(self):
        self.start()
        self.until_timing()
        reqs = self.fake.transcriptions()
        gate = [r for r in reqs if r.get("response_format") == "json"]
        self.assertEqual(len(gate), len(SCRIPT_BRAND["body"]) + 2 + 1, "every line and the track")
        for r in gate:
            self.assertNotIn("prompt", r, "a hint would make the ASR hear the script")
            self.assertNotIn("timestamp_granularities[]", r)
            self.assertEqual(r["language"], "en")
        timing = [r for r in reqs if r.get("response_format") == "verbose_json"]
        self.assertEqual(len(timing), 1)
        self.assertIn("ExampleCo", timing[0]["prompt"], "spelling hints are only for the word times")


class G5Aligner(TmpCase):

    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.tmp, "home")
        self.art = os.path.join(self.home, "artifacts")
        os.makedirs(self.art)
        self.fake = FakeMlxServe().start()
        self.fake.asr_loads = (studio.ALIGNER,)
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

    def gate_job(self, run_id="1.1"):
        t = self.eng.load(studio.QWEN3_ASR)
        self.assertTrue(wait_for(lambda: self.eng.poll_load(t).state == "ready"), self.eng.poll_load(t))
        wav(os.path.join(self.art, "ads", "g", "voice", "line-0.wav"), 1.0)
        g = {"language": "en"}
        ops = [self.pipe.gate_op(g, "voice/line-0.wav", f"gate/line-0-a{i}.json", "gate") for i in range(2)]
        h = self.eng.start(RunSpec(run_id, 1, "ad_gate", studio.QWEN3_ASR,
                                   {"goal": "g", "expected_s": 30.0, "workdir": "ads/g", "ops": ops}, 30.0))
        self.assertTrue(wait_for(lambda: self.eng.poll(h).status != "running", 20.0))
        self.assertEqual(self.eng.poll(h).status, "done", self.eng.poll(h))

    def test_the_aligner_the_gate_loaded_is_unloaded_when_its_job_ends(self):
        self.gate_job()
        self.assertNotIn(studio.ALIGNER, self.fake.ready, "the gate needs no word times: its aligner goes")
        self.assertIn(studio.QWEN3_ASR, self.eng.resident_models(), "the ASR itself stays for gpud to unload")
        self.assertIn({"model": studio.ALIGNER}, self.fake.bodies("/v1/unload-model"))

    def test_an_aligner_someone_else_loaded_stays(self):
        self.fake.other_client_load(studio.ALIGNER)
        self.gate_job()
        self.assertIn(studio.ALIGNER, self.fake.ready)
        self.assertNotIn({"model": studio.ALIGNER}, self.fake.bodies("/v1/unload-model"))


# ---------------------------------------------------------------- G6 the review

class G6Review(TmpCase):
    REVIEW = json.dumps({"verdict": "accept", "facts_ok": True, "character_ok": True, "problems": [], "note": "ok"})

    def setUp(self):
        super().setUp()
        self.home = make_home(self.tmp)
        self.pipe = ad_pipeline.AdPipeline(ads_config(), home=self.home,
                                           artifacts_dir=os.path.join(self.home, "artifacts"))
        self.wd = self.pipe.workdir("ads/rv")
        os.makedirs(os.path.join(self.wd, "final", "qa"), exist_ok=True)
        from foreman import media
        hero = os.path.join(self.wd, "final", "qa", "hero.png")
        with open(hero, "wb") as f:
            f.write(media.encode_png(4, 4, bytes(48)))
        for rel, obj in (("final/qa.json", {"ok": True, "checks": [{"check": "x", "ok": True}], "frames": [],
                                            "hero": "final/qa/hero.png"}),
                         ("ad.json", {"end_card": {"price": "$9.50", "url": "example.com", "cta": "Go"},
                                      "disclosure": "AI-generated", "ai_label": True}),
                         ("script.json", {"lines": [{"i": 0, "role": "hook", "text": "Hello there."}]}),
                         ("facts.json", {"text": "facts", "numbers": []})):
            paths.atomic_json(os.path.join(self.wd, rel), obj)
        self.asked = []

    def review(self, gate):
        if gate is not None:
            paths.atomic_json(os.path.join(self.wd, "gate.json"), gate)
        test = self

        class Client:
            def chat(self, model, messages, ctx, max_tokens=None):
                test.asked.append(messages)
                return G6Review.REVIEW

        class Ctx:
            class spec:
                vision = True
        return self.pipe.review(self.wd, "ads/rv", {}, studio.BRAIN_PRIMARY, Ctx(), Client())

    def test_a_failed_gate_is_given_up_without_asking_the_brain(self):
        d = self.review({"version": 1, "status": "failed", "reason": "audio gate: line 0 never said 'ExampleCo'",
                         "units": {}, "track": {}, "mix": {}})
        self.assertEqual(d["verdict"], "give_up")
        self.assertIn("never said 'ExampleCo'", d["note"])
        self.assertEqual(self.asked, [])
        self.assertEqual(paths.read_json(os.path.join(self.wd, "manifest.json"))["review"]["decision"], "give_up")

    def test_without_a_gate_record_the_brain_cannot_accept(self):
        d = self.review(None)
        self.assertEqual(d["verdict"], "human_review")
        self.assertIn("audio gate did not pass", d["note"])
        self.assertTrue(self.asked)

    def test_a_passed_gate_reaches_the_brain_and_the_note(self):
        d = self.review({"version": 1, "status": "passed", "units": {}, "track": {"wer": 0.0, "heard": "hello there"},
                         "mix": {"wer": 0.04, "lost": 0.0, "heard": "hello there friend"}})
        self.assertEqual(d["verdict"], "accept")
        self.assertTrue(d["note"].startswith("audio gate passed (voice WER 0.00, mix WER 0.04)"), d["note"])
        text = self.asked[0][1]["content"]
        text = text if isinstance(text, str) else " ".join(p.get("text", "") for p in text if isinstance(p, dict))
        self.assertIn("hello there friend", text, "the brain reads what the gate heard in the final mix")
        self.assertIn("Audio gate", text)

    def test_the_decision_rule(self):
        rd = ad_pipeline.AdPipeline.review_decision
        ok = {"verdict": "accept", "facts_ok": True, "character_ok": True}
        self.assertEqual(rd(ok, "", True, gate_ok=True)["verdict"], "accept")
        self.assertEqual(rd(ok, "", True, gate_ok=False)["verdict"], "human_review")


class G6Lineage(unittest.TestCase):

    def test_the_gate_is_an_ad_step_on_the_asr_model(self):
        self.assertIn("ad_gate", ads.JOB_KINDS)
        self.assertEqual(ads.KIND_FAMILIES["ad_gate"], ("asr",))
        self.assertIn("gate", ads.EXPECTED)


if __name__ == "__main__":
    unittest.main()
