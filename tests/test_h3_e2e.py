"""The h3 talk chain end to end: gpud runs whole h3 ads on the fakes (mlx-serve for the brain, the ASR
and the music; a ComfyUI process whose SaveVideo answers H3-like talking takes, each a tone burst per
word it says) with the REAL media worker (foreman/h3/worker.py under this Python, with numpy and
Pillow) and real ffmpeg: the probe, the graphics, the edit and the encode all run.

  H1  the happy path: facts, takes, probe, hearing, music, the edit, the final gate, finish and the
      review; the files (final/ad.mp4, gate.json, timing.json, pick.json, publish.json, the manifest's
      h3 record); one GPU stack at a time (ComfyUI never beside the ASR or its aligner); the owner's
      model never unloaded
  H2  retakes and a re-edit: stray hands, a run-on, a brand the voice cannot say (a new seed, then its
      next spelling), and a beat whose finished voice lost its brand (edited again, safe); the ad
      still reaches the review with the retaken takes in its cut, and the alternates go after the
      owner's verdict
  H3  failures: a beat that never passes (content, the beat named), a commercial template whose graph
      is not cleared (content, before any GPU work), no media python (engine_unconfigured, never the
      template's fault)

The fake ComfyUI renders at the graph's size, so these ads render takes at 128x224 (h3_width,
h3_height); the edit draws its cards on the real 1080x1920 frame. Everything is generic (ExampleCo,
example.com, $9.50).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import socket
import sys
import time
import unittest
from unittest import mock

from foreman import ad_pipeline, ledger, paths, studio
from foreman.adapters.comfyui import ComfyUIEngine
from foreman.adapters.local import LocalEngine
from foreman.adapters.mlx_serve import MlxServeEngine
from foreman.brain_client import BrainClient
from foreman.planner import Planner, PlannerConfig
from foreman.supervisor import Supervisor
from foreman.templates import load_templates

try:
    from ad_fixtures import FFMPEG, REVIEW_OK, ManualClock, TmpCase, ads_config, png
    from fake_mlx_serve import FakeMlxServe, H3Ear
    from test_h3_pipeline import PLAN, h3_block, h3_template
    from test_h3_pipeline import make_home as h3_home
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import FFMPEG, REVIEW_OK, ManualClock, TmpCase, ads_config, png
    from tests.fake_mlx_serve import FakeMlxServe, H3Ear
    from tests.test_h3_pipeline import PLAN, h3_block, h3_template
    from tests.test_h3_pipeline import make_home as h3_home

try:
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    MEDIA = True
except ImportError:  # pragma: no cover
    MEDIA = False
FONTS = all(os.path.isfile(ad_pipeline.H3_FONTS[k]) for k in ("black", "bold"))
needs_worker = unittest.skipUnless(FFMPEG and MEDIA and FONTS, "the h3 media worker needs ffmpeg, numpy, Pillow and "
                                                               "the Arial Black and Arial Bold fonts")
needs_ffmpeg = unittest.skipUnless(FFMPEG, "ffmpeg is not installed here")

HERE = os.path.dirname(os.path.abspath(__file__))
BRAIN = studio.BRAIN_PRIMARY
FOREIGN = ".revision-0123abcd"
MEDIA_MODELS = (studio.QWEN3_ASR, studio.ALIGNER, studio.ACE_STEP)
# How the fake ASR writes what a take says (the hinted and the free-running one alike): the lexicon's
# spoken forms and the brand's respelling as the brand, the address as written.
WRITTEN = {"Example Co": "ExampleCo", "Eggs Ample Co": "ExampleCo", "example dot com": "example.com"}
BRIEF = "Show how easy it is to get a form filed from a phone"
# A shorter plan with only the cutaways every ad has (a punch-in, the end card): one card to draw, for
# the ad that is edited twice.
SHORT = dict(PLAN, beats=[
    PLAN["beats"][0],
    {"line": "Upload it, and ExampleCo files it.", "scene": "desk", "cutaways": [],
     "action": "holding her phone in both hands, she taps it once, then looks up and tells the camera"},
    {"line": "It costs $9.50, all included.", "scene": "kitchen", "cutaways": [],
     "action": "resting her forearms on the counter, she explains with an easy smile"},
    {"line": "Go to example.com and skip the line.", "scene": "kitchen",
     "action": "her hands folded, she says it warmly, pronouncing the web address slowly and clearly, word by word",
     "cutaways": [{"kind": "endcard", "at": "and", "pulse": "line"}]}])


class H3Harness(TmpCase):
    """gpud with the fake mlx-serve, the local engine (its handlers run the real media worker) and a
    fake ComfyUI process, for one h3 ad template."""
    rules = ()              # the fake ComfyUI's rules for this test's takes (tests/fake_comfyui.py)
    commercial_ok = True    # the graph wrapper's commercial_ok
    options = {}            # ads_options over the harness's
    h3 = {}                 # the template's h3 block over the fixture's (h3_block)

    def config(self):
        base = dict(out_width=1080, out_height=1920, media_python=sys.executable, h3_graph="h3_ref2va", h3_width=128,
                    h3_height=224)
        base.update(self.options)
        return ads_config(**base)

    def setUp(self):
        super().setUp()
        self.tmp = os.path.realpath(self.tmp)
        self.home = h3_home(self.tmp, graph={"commercial_ok": self.commercial_ok})
        os.makedirs(os.path.join(self.home, "templates"))
        png(os.path.join(self.home, "assets", "brand", "logo.png"), 220, 64, 170)
        with open(os.path.join(self.home, "templates", "ad.json"), "w") as f:
            json.dump(h3_template(h3_block(**self.h3)), f)
        self.art = os.path.join(self.home, "artifacts")
        os.makedirs(self.art, exist_ok=True)
        self.registry = os.path.join(self.tmp, "comfy-registry.json")
        paths.atomic_json(self.registry, {"rules": [dict(r) for r in self.rules], "used": {}, "takes": []})
        env = mock.patch.dict(os.environ, {"FAKE_COMFY_REGISTRY": self.registry, "FAKE_COMFY_FFMPEG": FFMPEG or "",
                                           "FAKE_COMFY_RUN_S": "0.05"})
        env.start()
        self.addCleanup(env.stop)
        self.fake = FakeMlxServe(sizes={FOREIGN: int(16e9), BRAIN: int(20e9)}).start()
        self.fake.ready[FOREIGN] = int(16e9)
        self.fake.default_id, self.fake.default_promoted = FOREIGN, True
        self.fake.discovered |= {BRAIN, studio.BRAIN_FALLBACK}
        self.fake.music_cap = 90.0
        self.fake.asr_loads = (studio.ALIGNER,)  # word times load the forced aligner beside the ASR
        self.ear = H3Ear(self.registry, WRITTEN)
        self.fake.asr_resolver = self.ear
        self.catalog = studio.catalog(False, ads=True)
        self.pipeline = ad_pipeline.AdPipeline(self.config(), home=self.home, artifacts_dir=self.art, ffmpeg=FFMPEG,
                                               catalog=self.catalog, max_depth=64,
                                               commercial_models=studio.COMMERCIAL_OK, licenses=studio.LICENSES)
        self.brain = BrainClient(self.fake.url, catalog=self.catalog, ads=self.pipeline)
        self.mlx = MlxServeEngine(self.fake.url, catalog=self.catalog, model_dir="/fake/models", artifacts_dir=self.art,
                                  brain=self.brain, mem_probe=lambda: 200.0, read_poll_s=0.02, unload_wait_s=0.5,
                                  connect_timeout_s=2.0, call_timeout_s=2.0, forbidden=studio.FORBIDDEN, ffmpeg=FFMPEG,
                                  input_roots=[self.home], companions={studio.QWEN3_ASR: (studio.ALIGNER,)})
        self.local = LocalEngine(self.catalog, self.pipeline.handlers(), artifacts_dir=self.art, name=studio.LOCAL_ENGINE)
        comfy_dir = os.path.join(self.tmp, "ComfyUI")
        os.makedirs(comfy_dir)
        with open(os.path.join(comfy_dir, "main.py"), "w") as f:
            f.write(f"import sys\nsys.path.insert(0, {HERE!r})\nfrom fake_comfyui import main\n"
                    f"raise SystemExit(main(sys.argv[1:]))\n")
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        self.comfy = ComfyUIEngine(self.catalog, comfy_dir=comfy_dir, python=sys.executable, port=port,
                                   graphs_dir=os.path.join(self.home, "comfyui", "graphs"), artifacts_dir=self.art,
                                   home=self.home, mem_probe=lambda: 200.0, media_probe=self.mlx.resident_media,
                                   start_timeout_s=20.0, stop_wait_s=3.0, poll_s=0.05, name=studio.COMFY_ENGINE)
        self.clock = ManualClock()
        self.planner = Planner(PlannerConfig(utc=True, min_disk_free_gb=0.0, artifacts_cap_gb=1e6), home=self.home,
                               artifacts_dir=self.art, commercial_models=studio.COMMERCIAL_OK,
                               disk_probe=lambda: (1000.0, 0.0))
        self.sup = Supervisor(os.path.join(self.tmp, "foreman.db"),
                              {studio.ENGINE: self.mlx, studio.LOCAL_ENGINE: self.local, studio.COMFY_ENGINE: self.comfy},
                              self.clock, studio.policy(), catalog=self.catalog, pager=lambda *_: None,
                              planner=self.planner)
        self.sup.recover()
        self.together = []      # mlx-serve media models seen resident while the ComfyUI process was up
        self.comfy_ticks = 0    # ticks that saw the ComfyUI process up

    def tearDown(self):
        try:
            self.comfy._stop_process()
        finally:
            self.sup.close()
            self.mlx.close()
            self.local.close()
            self.fake.stop()
            super().tearDown()

    def enqueue(self, brief=BRIEF):
        t = load_templates(os.path.join(self.home, "templates"))[0]["ads.service.v1"]
        return ad_pipeline.enqueue(self.sup, t, brief, None, {}, self.art, self.home,
                                   PlannerConfig(min_disk_free_gb=0.0), force=True)

    def run_until(self, pred, timeout=420.0):
        """Tick gpud until pred(). Supervisor time runs slow while CPU or ComfyUI work runs in real time
        (an edit takes real seconds; its lease is sized in supervisor seconds)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            busy = any(j.state in ("starting", "running", "cancelling") and j.requires in (studio.LOCAL, studio.COMFYUI)
                       for j in self.sup.store.jobs())
            self.clock.t += 0.05 if busy else 1.0
            self.sup.tick()
            if self.comfy.resident_models():
                self.comfy_ticks += 1
                with self.fake.lock:
                    beside = sorted(m for m in self.fake.ready if m in MEDIA_MODELS)
                if beside:
                    self.together.append(beside)
            if pred():
                return True
            time.sleep(0.01)
        return False

    def item(self, iid):
        return ledger.item(self.sup.store, iid)

    def done(self, iid):
        return lambda: self.item(iid)["state"] not in ("queued",)

    def jobs(self, kind=None):
        return [j for j in self.sup.store.jobs() if kind is None or j.kind == kind]

    def kinds(self):
        return [j.kind for j in self.jobs() if j.kind.startswith("ad_") and j.requires != "brain"]

    def trail(self):
        return [(j.id, j.kind, j.state, (j.outcome or {}).get("reason"), str((j.outcome or {}).get("detail"))[:300])
                for j in self.sup.store.jobs()]

    def takes(self) -> list:
        return paths.read_json(self.registry)["takes"]

    def read(self, res, rel):
        return paths.read_json(os.path.join(res["workdir"], rel))

    def assert_one_stack_at_a_time(self):
        """ComfyUI never beside the ASR (or its aligner, or the music model): in gpud's own events, and
        in what mlx-serve held whenever the ComfyUI process was up."""
        self.assertGreater(self.comfy_ticks, 0)
        self.assertEqual(self.together, [], "an mlx-serve media model was resident beside ComfyUI")
        resident, loaded = set(), set()
        for e in self.sup.store.events():
            model = e["data"].get("model")
            if e["kind"] == "load_started":
                resident.add(model)
                loaded.add(model)
                self.assertFalse(studio.COMFYUI in resident and studio.QWEN3_ASR in resident,
                                 ("ComfyUI and the ASR resident together", e))
            elif e["kind"] in ("model_unloaded", "model_evicted"):
                resident.discard(model)
        self.assertLessEqual({studio.COMFYUI, studio.QWEN3_ASR}, loaded)
        self.assertIn(FOREIGN, self.fake.ready, "the owner's model is never unloaded")
        self.assertNotIn(FOREIGN, [b.get("model") for b in self.fake.bodies("/v1/unload-model")])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


@needs_worker
class H1HappyPath(H3Harness):
    # the owner's kind of template: a logo, paint-outs on the captures, a price card re-stacked with a
    # lead (room for its underline) and part of it painted out
    h3 = {"brand": {"name": "ExampleCo", "bg": "#101010", "accent": "#2255CC", "logo": "brand/logo.png"},
          "captures": {"captures/upload.png": {"id": "upload", "note": "the upload page", "crop": [0, 0, 100, 80],
                                               "paint": [[60, 60, 100, 80, "#202020"]], "tap": [10, 50, 90, 70],
                                               "tap_color": "#1d4ed8"}},
          "price_card": {"big": "$9.50", "lines": ["One form, everything included"], "capture": "captures/upload.png",
                         "crop": [0, 0, 100, 60], "underline": [[10, 10, 50, 20]], "lead": 6,
                         "paint": [[70, 40, 100, 60, "#0E1626"]]}}

    def test_one_h3_ad_from_enqueue_to_the_review_queue(self):
        self.fake.chat_replies.extend([json.dumps(PLAN), REVIEW_OK])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "review", (it, self.trail()))
        self.assertEqual(it["brain_verdict"], "accept")
        wd = res["workdir"]
        final = os.path.join(wd, "final", "ad.mp4")
        self.assertEqual(it["artifacts"][0], final, "the phone sends the item's first MP4: the finished ad")
        self.assertNotIn(".mp4", "".join(os.path.basename(a) for a in it["artifacts"][1:]))
        # every step a gpud job, in order: one round of takes, probed, heard, then the cut and its gate
        self.assertEqual(self.kinds(), ["ad_facts", "ad_shots", "ad_probe", "ad_gate", "ad_music", "ad_assemble",
                                        "ad_gate", "ad_finish"], self.trail())
        self.assertEqual([j.requires for j in self.jobs("ad_shots")], [studio.COMFYUI])
        self.assertEqual([j.requires for j in self.jobs("ad_probe")], [studio.LOCAL])
        self.assert_one_stack_at_a_time()
        self.assertNotIn(studio.ALIGNER, self.fake.ready, "the aligner the word times loaded went with the hearing")

        # what ComfyUI was asked for: one take per beat, the line said as the voice is told it
        script = self.read(res, "script.json")
        ad = self.read(res, "ad.json")
        takes = self.takes()
        self.assertEqual(len(takes), 4)
        self.assertEqual([t["seed"] for t in takes], [(ad["seed"] + 51 + k) % 2147483647 for k in range(4)])
        self.assertEqual(takes[3]["spoken"], "Get it done today at example dot com and skip the line.")
        self.assertEqual(takes[2]["spoken"], "It costs nine fifty, with everything included.")
        self.assertIn("Example Co", takes[1]["spoken"])
        self.assertEqual([t["size"] for t in takes], [[128, 224]] * 4)
        self.assertEqual([t["seconds"] for t in takes], [b["seconds"] for b in script["beats"]])

        # the speech gate's record
        g = self.read(res, "gate.json")
        self.assertEqual((g["mode"], g["status"], g["rounds"]), ("h3", "passed", 0))
        self.assertGreater(g["spent_s"], 0.0)
        for k in range(4):
            u = g["units"][f"beat-{k}"]
            self.assertEqual((u["status"], len(u["attempts"])), ("passed", 1), u)
            take = u["attempts"][0]
            self.assertEqual((take["verdict"], take["ok"], take["mp4"]), ("clean", True, f"takes/beat-{k}-t0.mp4"))
            self.assertEqual(len(take["words"]), len(script["beats"][k]["words"]))
            for f in ("mp4", "wav", "probe", "hinted", "transcript"):
                self.assertTrue(os.path.isfile(os.path.join(wd, take[f])), take[f])
        cta = g["units"]["beat-3"]["attempts"][0]
        self.assertTrue(cta["anchor"]["ok"], "the end card's word starts on a voiced onset")
        self.assertEqual((g["track"]["status"], g["mix"]["status"]), ("passed", "passed"))
        self.assertEqual(g["mix"]["wer"], 0.0)
        self.assertEqual(sorted(g["final"]["beats"]), ["b0", "b1", "b2", "b3"])
        self.assertTrue(all(v["ok"] for v in g["final"]["beats"].values()), g["final"])
        self.assertFalse(g["final"]["safe"])
        probe = self.read(res, "takes/beat-3-t0.probe.json")
        self.assertEqual((probe["width"], probe["height"], probe["frames"]), (128, 224, takes[3]["frames"]))
        self.assertEqual(len(probe["voiced_runs"]), len(takes[3]["tokens"]), "one voiced run per word said")
        hinted = [b for b in self.fake.transcriptions() if b.get("timestamp_granularities[]") == "word"]
        self.assertEqual(len(hinted), 4)
        self.assertTrue(all("ExampleCo" in b["prompt"] and "example.com" in b["prompt"] for b in hinted))
        self.assertEqual(len(self.fake.transcriptions()), 4 * 2 + 4 + 1, "two per take, each beat's voice, the mix")

        # the cut: the worker's timeline in Foreman's timing.json, the takes it used, the publish gate
        tl = self.read(res, "final/timeline.json")
        timing = self.read(res, "timing.json")
        self.assertEqual((timing["backend"], timing["source"], timing["fps"]), ("h3", "asr", 24))
        self.assertEqual(timing["total_s"], round(tl["total"], 4))
        kinds = [s["kind"] for s in timing["shots"]]
        for kind in ("talk", "punch", "screen", "still", "price", "endcard"):
            self.assertIn(kind, kinds)
        self.assertEqual(timing["endcard_start"], next(s["start"] for s in timing["shots"] if s["kind"] == "endcard"))
        self.assertAlmostEqual(sum(s["seconds"] for s in timing["shots"]), timing["total_s"], places=3)
        self.assertEqual({w["line"] for w in timing["words"]}, {0, 1, 2, 3})
        self.assertIn("example.com", [w["text"] for w in timing["words"]])
        self.assertEqual(tl["checks"]["frames"]["got"], tl["frames"])
        self.assertTrue(all(p["seen"] for p in tl["checks"]["punch_cuts"]), tl["checks"]["punch_cuts"])
        req = self.read(res, "h3/edit-req.json")
        self.assertEqual((req["price_card"]["lead"], req["price_card"]["paint"]), (6, [[70, 40, 100, 60, "#0E1626"]]))
        self.assertEqual(req["captures"]["upload"]["paint"], [[60, 60, 100, 80, "#202020"]])
        self.assertEqual(req["brand"]["logo_png"], os.path.join(self.home, "assets", "brand", "logo.png"))
        pick = self.read(res, "pick.json")
        self.assertEqual({k: (v["take"], v["alternates"]) for k, v in pick.items()},
                         {str(k): (f"takes/beat-{k}-t0.mp4", []) for k in range(4)})
        pub = self.read(res, "final/publish.json")
        self.assertEqual(pub["ad_sha256"], _sha256(final), "the publish gate belongs to this file")
        self.assertEqual(pub["ai_disclosure"], {"tiktok_aigc_label": None, "youtube_altered_or_synthetic": None,
                                                "meta_ai_info": None})
        self.assertEqual((pub["srt"], pub["attribution"], pub["verified_by"]), ("final/ad.srt", "MiniMax H3", None))
        with open(os.path.join(wd, "final", "ad.srt")) as f:
            self.assertIn("example.com", f.read())
        asm = self.read(res, "final/assembly.json")
        self.assertEqual((asm["label"], asm["edit_round"], asm["h3"]["safe"]), (False, 0, False))
        self.assertTrue(asm["burned_in"]["ok_captions"], asm["burned_in"])
        for gone in ("final/video.mp4", "final/voice.wav", "final/segments"):
            self.assertFalse(os.path.exists(os.path.join(wd, gone)), gone)

        # finish measured the file; the manifest keeps the h3 record
        qa = self.read(res, "final/qa.json")
        self.assertTrue(qa["ok"], [c for c in qa["checks"] if not c["ok"]])
        checks = {c["check"]: c["ok"] for c in qa["checks"]}
        for name in ("frames_exact", "no_freeze", "punch_cuts", "color_bt709", "captions", "loudness"):
            self.assertTrue(checks[name], name)
        self.assertEqual((qa["probe"]["width"], qa["probe"]["height"]), (1080, 1920))
        m = self.read(res, "manifest.json")
        h3 = m["h3"]
        self.assertEqual((h3["graph"], h3["attribution"], h3["commercial_ok"], h3["render"]),
                         ("h3_ref2va", "MiniMax H3", True, [128, 224]))
        self.assertIn("MiniMax H3", h3["license"])
        self.assertEqual(h3["chosen"], {f"b{k}": f"takes/beat-{k}-t0.mp4" for k in range(4)})
        self.assertEqual([b["id"] for b in h3["timeline"]["beats"]], ["b0", "b1", "b2", "b3"])
        self.assertEqual(h3["timeline"]["beats"][1]["cutaways"], ["screen", "still"])
        self.assertEqual(h3["publish"], "final/publish.json")
        self.assertEqual(m["licenses"]["graph:h3_ref2va"], h3["license"])
        self.assertEqual(m["versions"].get("comfyui"), "0.43.0")
        self.assertFalse(m["disclosure"]["label_burned_in"])
        self.assertEqual(m["disclosure"]["attribution"], "MiniMax H3")
        self.assertEqual(m["backends"]["talk"], "h3")
        self.assertEqual([a["n"] for a in m["seeds"]["attempts"]["b2"]], [0])
        self.assertEqual(m["gate"]["status"], "passed")
        self.assertEqual(m["review"]["decision"], "accept")
        self.assertTrue(m["review"]["note"].startswith("speech gate passed (4 beats, 0 retakes; final lines heard "
                                                       "intact; mix WER 0.00)"), m["review"]["note"])
        self.assertTrue(m["review"]["note"].endswith(ad_pipeline.H3_PUBLISH_NOTE), m["review"]["note"])
        self.assertEqual(ad_pipeline.progress(wd)["current"], "done")
        # the owner's view: ads status and ads inspect read an h3 ad's files
        from foreman import cli
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"FOREMAN_HOME": self.home}), contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["--db", os.path.join(self.tmp, "foreman.db"), "ads", "status"]), 0)
            self.assertEqual(cli.main(["--db", os.path.join(self.tmp, "foreman.db"), "ads", "inspect",
                                       str(res["job_id"])]), 0)
        text = out.getvalue()
        self.assertIn("review", text)
        self.assertIn("QA      pass", text)
        self.assertIn("line 3 (cta): Get it done today at example.com and skip the line.", text)


@needs_worker
class H2RetakesAndReedit(H3Harness):
    # beat 0's first take runs on into a word; beat 1's voice cannot say "Example Co" (only its
    # respelling); beat 3's first take (the call to action) shows hands rising into the close-up
    rules = ({"match": "Need a form", "do": "after", "after": "Yeah.", "times": 1},
             {"match": "Example Co", "do": "say", "say": {"Example Co": "Example Ko"}},
             {"match": "skip the line", "do": "hands", "times": 1})

    def test_retakes_a_safe_reedit_and_the_alternates_after_the_verdict(self):
        def lost_brand(info):  # the first hearing of beat 1's finished voice misses its brand
            if not info["words"] and info["n"] == 1 and "Eggs Ample Co" in info["takes"][0]["spoken"] \
                    and len(info["takes"]) == 1:
                return info["text"].replace("ExampleCo ", "")
            return None
        self.ear.hook = lost_brand
        self.fake.chat_replies.extend([json.dumps(SHORT), REVIEW_OK])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "review", (it, self.trail()))
        wd = res["workdir"]
        rnd = ["ad_shots", "ad_probe", "ad_gate"]
        self.assertEqual(self.kinds(), ["ad_facts"] + rnd * 3 + ["ad_music", "ad_assemble", "ad_gate", "ad_assemble",
                                                                 "ad_gate", "ad_finish"], self.trail())
        self.assert_one_stack_at_a_time()

        g = self.read(res, "gate.json")
        self.assertEqual((g["status"], g["rounds"]), ("passed", 2))
        verdicts = {k: [t["verdict"] for t in u["attempts"]] for k, u in g["units"].items()}
        self.assertEqual(len(verdicts["beat-0"]), 2)
        self.assertIn("runs on into ['Yeah']", verdicts["beat-0"][0])
        self.assertEqual(len(verdicts["beat-1"]), 3)
        self.assertTrue(all("never said 'ExampleCo'" in v for v in verdicts["beat-1"][:2]), verdicts["beat-1"])
        self.assertEqual(verdicts["beat-2"], ["clean"])
        self.assertEqual(len(verdicts["beat-3"]), 2)
        self.assertIn("hands visible at", verdicts["beat-3"][0])
        b1 = g["units"]["beat-1"]["attempts"]
        self.assertEqual([t["seed"] - b1[0]["seed"] for t in b1], [0, 1000, 2000])
        self.assertIn("next spelling of 'ExampleCo'", b1[2]["why"])
        self.assertEqual(b1[2]["forms"].get("ExampleCo"), "Eggs Ample Co")
        said = {t["spoken"] for t in self.takes()}
        self.assertIn("Upload it, and Eggs Ample Co files it.", said)
        self.assertEqual(len(self.takes()), 4 + 3 + 1)
        # beat 1's finished voice lost its brand once: edited again with no pause removed, then heard whole
        fin = g["final"]
        self.assertEqual((fin["safe"], fin["round"]), (True, 1))
        self.assertIn("beat 1's finished voice never said 'ExampleCo'", fin["then"])
        self.assertTrue(all(v["ok"] and v["r"] == 1 for v in fin["beats"].values()), fin["beats"])
        edit = self.read(res, "h3/edit-req.json")
        self.assertTrue(edit["safe"])
        self.assertEqual([b["take"] for b in edit["beats"]], ["takes/beat-0-t1.mp4", "takes/beat-1-t2.mp4",
                                                              "takes/beat-2-t0.mp4", "takes/beat-3-t1.mp4"])
        tl = self.read(res, "final/timeline.json")
        self.assertFalse([r for r in tl["checks"]["removals"] if r["ok"]], "a safe edit removes nothing")
        self.assertEqual(self.read(res, "final/assembly.json")["edit_round"], 1)
        self.assertEqual(self.read(res, "final/publish.json")["ad_sha256"], _sha256(os.path.join(wd, "final", "ad.mp4")))
        self.assertTrue(self.read(res, "final/qa.json")["ok"])
        m = self.read(res, "manifest.json")
        self.assertEqual(m["h3"]["chosen"]["b1"], "takes/beat-1-t2.mp4")
        self.assertEqual([a["n"] for a in m["seeds"]["attempts"]["b1"]], [0, 1, 2])
        self.assertTrue(m["review"]["note"].startswith("speech gate passed (4 beats, 4 retakes; final lines heard "
                                                       "intact, edited with no pause removed;"), m["review"]["note"])

        # the alternates (and every file beside them) go after the owner's verdict; the cut's takes stay
        pick = self.read(res, "pick.json")
        self.assertEqual(pick["1"]["alternates"], ["takes/beat-1-t0.mp4", "takes/beat-1-t1.mp4"])
        alts = [a for v in pick.values() for a in v["alternates"]]
        self.assertEqual(len(alts), 4)
        side = [a[:-4] + ext for a in alts for ext in (".wav", ".probe.json", ".mp4.meta.json")] + \
               [f"gate/{os.path.basename(a)[:-4]}{ext}" for a in alts for ext in (".asr.json", ".asr0.json")]
        self.assertTrue(all(os.path.exists(os.path.join(wd, f)) for f in alts + side))
        ledger.record_verdict(self.sup.store, res["job_id"], "reject", 2.0, "not this one", self.sup._now())
        self.planner._last_upkeep = None
        self.clock.t += 120
        self.sup.tick()
        self.assertEqual([f for f in alts + side if os.path.exists(os.path.join(wd, f))], [])
        for v in pick.values():
            for f in (v["take"], v["take"][:-4] + ".wav", v["take"][:-4] + ".probe.json"):
                self.assertTrue(os.path.exists(os.path.join(wd, f)), f)
        self.assertTrue(os.path.exists(os.path.join(wd, "final", "ad.mp4")))


@needs_worker
class H3NeverPasses(H3Harness):
    # beat 2's voice never says its price right; beat 0's first take comes back with no sound
    rules = ({"match": "nine fifty", "do": "say", "say": {"nine fifty": "four fifty"}},
             {"match": "Need a form", "do": "noaudio", "times": 1})
    options = {"h3_attempts": 2}

    def test_a_beat_that_never_passes_fails_the_ad_as_content(self):
        self.fake.chat_replies.extend([json.dumps(PLAN), REVIEW_OK])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=200.0), self.trail())
        it = self.item(res["item"])
        self.assertEqual((it["state"], it["fail_class"]), ("failed", "content"), (it, self.trail()))
        self.assertEqual(self.kinds(), ["ad_facts"] + ["ad_shots", "ad_probe", "ad_gate"] * 2, self.trail())
        g = self.read(res, "gate.json")
        self.assertEqual(g["status"], "failed")
        self.assertTrue(g["reason"].startswith("speech gate: beat 2 (body) never passed after 2 take(s): never said "
                                               "the number $9.50"), g["reason"])
        self.assertIn("four fifty", g["reason"])
        self.assertEqual([len(g["units"][f"beat-{k}"]["attempts"]) for k in range(4)], [2, 1, 2, 1])
        self.assertEqual(len(self.takes()), 6, "only the failing beats were taken again")
        b0 = g["units"]["beat-0"]["attempts"]
        self.assertEqual((b0[0]["verdict"], b0[1]["verdict"], g["units"]["beat-0"]["status"]),
                         ("take: no audio", "clean", "passed"))
        self.assertEqual(len(self.fake.transcriptions()), 2 * (3 + 2), "a take with no sound is never sent to the ASR")
        self.assertFalse(os.path.exists(os.path.join(res["workdir"], "final")), "nothing was edited")
        self.assert_one_stack_at_a_time()


@needs_worker
class H3CommercialGraph(H3Harness):
    commercial_ok = False

    def test_a_commercial_template_needs_a_graph_cleared_for_it(self):
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=60.0), self.trail())
        it = self.item(res["item"])
        self.assertEqual((it["state"], it["fail_class"]), ("failed", "content"), (it, self.trail()))
        facts = self.jobs("ad_facts")
        self.assertEqual([(j.state, j.outcome["reason"]) for j in facts], [("failed", "rejected")], self.trail())
        self.assertIn("commercial_ok", str(facts[0].outcome["detail"]))
        self.assertEqual(self.kinds(), ["ad_facts"], "no GPU work before the license holds")
        self.assertEqual(self.takes(), [])


@needs_ffmpeg
class H3NoMediaPython(H3Harness):
    options = {"media_python": ""}

    def test_no_media_python_is_the_machines_fault(self):
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=60.0), self.trail())
        it = self.item(res["item"])
        facts = self.jobs("ad_facts")
        self.assertTrue(facts, self.trail())
        self.assertEqual({j.outcome["reason"] for j in facts}, {"engine_unconfigured"}, self.trail())
        self.assertIn("media_python", str(facts[0].outcome["detail"]))
        self.assertEqual(self.kinds(), ["ad_facts"] * len(facts), "no GPU work on a machine that cannot edit")
        self.assertEqual((it["state"], it["fail_class"]), ("failed", "infra"), (it, self.trail()))


class H4Seams(unittest.TestCase):
    """Where one builder's part meets another's, piece by piece (no ad runs)."""

    @staticmethod
    def problems(h3) -> list:
        from foreman.templates import TemplateError, parse_template
        try:
            parse_template(h3_template(h3))
        except TemplateError as e:
            return [ln.strip()[2:] for ln in str(e).split("\n")[1:]]
        return []

    def test_the_price_card_takes_a_lead_and_paint_outs_with_its_capture(self):
        from foreman import h3core
        from foreman.templates import parse_template
        card = {"big": "$9.50", "lines": ["One form"], "capture": "captures/upload.png", "crop": [0, 0, 100, 60],
                "lead": 8, "paint": [[70, 40, 100, 60, "#0E1626"]]}
        ad = parse_template(h3_template(h3_block(price_card=card))).ad
        pc = ad["h3"]["price_card"]
        self.assertEqual((pc["lead"], pc["paint"]), (8, [[70, 40, 100, 60, "#0E1626"]]))
        again = []
        self.assertEqual(h3core.parse_h3_block(ad["h3"], again, ad), ad["h3"], "the cleaned block parses unchanged")
        self.assertEqual(again, [])
        self.assertEqual(h3core.h3_box_problems(ad["h3"], {"captures/upload.png": (60, 50)})[-1],
                         "ad.h3.price_card.paint[0] [70, 40, 100, 60] is outside the capture (60x50)")
        self.assertIn("ad.h3.price_card.lead must be an integer 0..200 (capture pixels)",
                      self.problems(h3_block(price_card=dict(card, lead=300))))
        self.assertIn("ad.h3.price_card.paint[0] must be [x0, y0, x1, y1, \"#rrggbb\"] in image pixels",
                      self.problems(h3_block(price_card=dict(card, paint=[[1, 2, 3, 4]]))))
        bare = {"big": "$9.50", "lines": ["One form"], "lead": 8}
        self.assertIn("ad.h3.price_card.crop, underline, lead and paint need a capture",
                      self.problems(h3_block(price_card=bare)))
        plain = parse_template(h3_template(h3_block(price_card={"big": "$9.50", "lines": []}))).ad["h3"]["price_card"]
        self.assertEqual((plain["lead"], plain["paint"]), (0, []))

    def test_hands_count_only_where_the_beat_shows_its_take(self):
        from foreman import h3core
        words = [{"text": w, "start": s, "end": s + 0.3} for w, s in
                 (("Go", 0.5), ("to", 1.0), ("example.com", 1.5), ("and", 2.5), ("skip", 3.0), ("it.", 3.5))]
        beat = {"tail": None, "hold": 1.0,
                "cutaways": [{"kind": "punch", "at_i": 1, "until_i": 2}, {"kind": "endcard", "at_i": 3, "until_i": None}]}
        hits = [0.0, 0.2, 1.1, 2.3, 2.5, 5.9]
        self.assertEqual(h3core.shown_hands(hits, beat, words, [0.5, 3.8, None], 5.0), [0.2, 1.1, 2.3],
                         "before the line's lead-in or under the end card: not shown; under a punch-in "
                         "it is the take")
        self.assertEqual(h3core.shown_hands(hits, dict(beat, tail=0.05, hold=0.0, cutaways=[]), words,
                                            [0.5, 3.8, None], 5.0), [0.2, 1.1, 2.3, 2.5])
        self.assertEqual(h3core.shown_hands(hits, beat, words, None, 5.0), hits, "no span heard: every hit counts")

    @unittest.skipUnless(MEDIA, "the worker needs numpy and Pillow")
    def test_a_card_that_cannot_be_drawn_is_content_and_a_missing_file_the_machine(self):
        from foreman.h3 import edit, ff, graphics

        def layout():
            raise graphics.LayoutError("the headline does not fit in two lines")

        def missing():
            raise FileNotFoundError("display font not found: '/no/font.ttf'")
        with self.assertRaises(ff.ContentError) as e:
            edit._draw("the headline", layout)
        self.assertEqual((e.exception.kind, str(e.exception)),
                         ("content", "the headline: the headline does not fit in two lines"))
        with self.assertRaises(ff.InfraError) as e:
            edit._draw("beat b3: the end card", missing)
        self.assertEqual(e.exception.kind, "infra")
        self.assertIsNotNone(ad_pipeline.H3_UNCONFIGURED.search(str(e.exception)), "a missing font is the setup's")
        self.assertEqual(edit._draw("a card", lambda x, y=0: x + y, 2, y=3), 5)


if __name__ == "__main__":
    unittest.main()
