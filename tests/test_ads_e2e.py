"""The ad pipeline end to end: gpud on the fake mlx-serve (and the local CPU engine) runs one ad
from `ads enqueue` to the review queue: facts, the brain's script, voice, ASR timing, stills,
talking takes (several seeds) and B-roll, music, the ffmpeg cut and its QA, and the vision brain's
check. Then the failure paths: an infrastructure failure retried without using the template's
cap, a take that fails its checks (the others carry on), a script that never validates (a content
failure), and the alternates deleted after the owner's verdict.

  E1  the happy path, one GPU stack at a time, the owner's model never touched
  E2  infrastructure failures and content failures, classified for the ledger
  E3  retention after the verdict; the phone's view of the item
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import unittest

from foreman import ad_pipeline, ledger, paths, studio
from foreman.adapters.comfyui import ComfyUIEngine
from foreman.adapters.local import LocalEngine
from foreman.adapters.mlx_serve import MlxServeEngine
from foreman.brain_client import BrainClient
from foreman.planner import Planner, PlannerConfig
from foreman.supervisor import Supervisor
from foreman.templates import load_templates

try:
    from ad_fixtures import FFMPEG, REVIEW_OK, SCRIPT, ManualClock, TmpCase, ads_config, make_home, needs_ffmpeg
    from fake_mlx_serve import FakeMlxServe
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import (FFMPEG, REVIEW_OK, SCRIPT, ManualClock, TmpCase, ads_config, make_home,
                                   needs_ffmpeg)
    from tests.fake_mlx_serve import FakeMlxServe

BRAIN, FALLBACK = studio.BRAIN_PRIMARY, studio.BRAIN_FALLBACK
FOREIGN = ".revision-0123abcd"


class AdHarness(TmpCase):
    clone = False

    def setUp(self):
        super().setUp()
        self.home = make_home(self.tmp, clone=self.clone)
        self.art = os.path.join(self.home, "artifacts")
        os.makedirs(self.art, exist_ok=True)
        self.fake = FakeMlxServe(sizes={FOREIGN: int(16e9), BRAIN: int(20e9)}).start()
        self.fake.ready[FOREIGN] = int(16e9)
        self.fake.default_id, self.fake.default_promoted = FOREIGN, True
        self.fake.discovered |= {BRAIN, FALLBACK}
        self.fake.speech_cap, self.fake.speech_rate = 6.0, 0.075
        self.fake.music_cap = 4.0
        self.catalog = studio.catalog(False, ads=True)
        self.pipeline = ad_pipeline.AdPipeline(self.config(), home=self.home, artifacts_dir=self.art, ffmpeg=FFMPEG,
                                               catalog=self.catalog, max_depth=64, commercial_models=studio.COMMERCIAL_OK,
                                               licenses=studio.LICENSES)
        self.brain = BrainClient(self.fake.url, catalog=self.catalog, ads=self.pipeline)
        self.mlx = MlxServeEngine(self.fake.url, catalog=self.catalog, model_dir="/fake/models",
                                  artifacts_dir=self.art, brain=self.brain, mem_probe=lambda: 200.0, read_poll_s=0.02,
                                  unload_wait_s=0.5, connect_timeout_s=2.0, call_timeout_s=2.0,
                                  forbidden=studio.FORBIDDEN, ffmpeg=FFMPEG, input_roots=[self.home])
        self.local = LocalEngine(self.catalog, self.pipeline.handlers(), artifacts_dir=self.art,
                                 name=studio.LOCAL_ENGINE)
        self.comfy = ComfyUIEngine(self.catalog, comfy_dir=os.path.join(self.tmp, "no-comfy"), port=1,
                                   artifacts_dir=self.art, home=self.home, name=studio.COMFY_ENGINE)
        self.clock = ManualClock()
        self.pages = []
        self.planner = Planner(PlannerConfig(utc=True, min_disk_free_gb=0.0, artifacts_cap_gb=1e6), home=self.home,
                               artifacts_dir=self.art, commercial_models=studio.COMMERCIAL_OK,
                               disk_probe=lambda: (1000.0, 0.0))
        self.sup = Supervisor(os.path.join(self.tmp, "foreman.db"),
                              {studio.ENGINE: self.mlx, studio.LOCAL_ENGINE: self.local, studio.COMFY_ENGINE: self.comfy},
                              self.clock, studio.policy(), catalog=self.catalog, pager=self.pages.append,
                              planner=self.planner)
        self.sup.recover()

    def config(self):
        return ads_config()

    def tearDown(self):
        self.sup.close()
        self.mlx.close()
        self.local.close()
        self.fake.stop()
        super().tearDown()

    def enqueue(self, brief="Show how easy it is to get a form filed from a desk", **params):
        t = load_templates(os.path.join(self.home, "templates"))[0]["ads.service.v1"]
        return ad_pipeline.enqueue(self.sup, t, brief, None, params, self.art, self.home,
                                   PlannerConfig(min_disk_free_gb=0.0), force=True)

    def run_until(self, pred, timeout=150.0, dt=1.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.clock.t += dt
            self.sup.tick()
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

    def trail(self):
        return [(j.id, j.kind, j.state, (j.outcome or {}).get("reason"), str((j.outcome or {}).get("detail"))[:200])
                for j in self.sup.store.jobs()]


@needs_ffmpeg
class E1HappyPath(AdHarness):

    def test_one_ad_from_enqueue_to_the_review_queue(self):
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        self.fake.asr_text = " ".join([SCRIPT["hook"]] + SCRIPT["body"] + [SCRIPT["cta"]])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "review", (it, self.trail()))
        self.assertEqual(it["brain_verdict"], "accept")
        wd = res["workdir"]
        final = os.path.join(wd, "final", "ad.mp4")
        self.assertEqual(it["artifacts"][0], final, "the phone sends the item's first MP4: the finished ad")
        self.assertNotIn(".mp4", "".join(os.path.basename(a) for a in it["artifacts"][1:]))
        # every step ran as a gpud job, in order, one GPU stack at a time
        kinds = [j.kind for j in self.jobs() if j.kind.startswith("ad_") and j.requires != "brain"]
        order = ["ad_facts", "ad_voice", "ad_timing", "ad_stills", "ad_shots", "ad_music", "ad_assemble", "ad_finish"]
        self.assertEqual([k for k in dict.fromkeys(kinds)], order, kinds)
        self.assertEqual(len(self.fake.video_requests), 2 * 2 + 2, "two talking shots x two seeds, two B-roll takes")
        self.assertEqual(kinds.count("ad_shots"), 2, "takes packed into jobs of at most max_job_s, one kind per job")
        self.assertLessEqual(len(self.fake.ready), 3, "never more than mlx-serve's three slots")
        self.assertIn(FOREIGN, self.fake.ready, "the owner's model is never unloaded")
        self.assertNotIn(FOREIGN, [b.get("model") for b in self.fake.bodies("/v1/unload-model")])
        # the outputs
        qa = paths.read_json(os.path.join(wd, "final", "qa.json"))
        self.assertTrue(qa["ok"], qa)
        self.assertEqual((qa["probe"]["width"], qa["probe"]["height"]), (270, 480))
        manifest = paths.read_json(os.path.join(wd, "manifest.json"))
        for key in ("models", "licenses", "versions", "seeds", "facts", "disclosure", "steps", "qa", "review"):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["disclosure"]["label_burned_in"], True)
        self.assertEqual(manifest["review"]["decision"], "accept")
        self.assertIn("9.5", manifest["facts"]["numbers"])
        self.assertTrue(manifest["steps"], "per-step timings")
        self.assertEqual(sorted(manifest["seeds"]["takes"]), ["0", "1", "3", "4"])
        # captions come from the script text, timed by the ASR words
        timing = paths.read_json(os.path.join(wd, "timing.json"))
        self.assertEqual(timing["source"], "asr")
        self.assertIn("example.com.", [w["text"] for w in timing["words"]])
        with open(os.path.join(wd, "final", "captions.ass")) as f:
            ass = f.read()
        self.assertIn("example.com.", ass)
        # audio-to-video got the line's WAV padded to the clip, and the takes were muxed with it
        a2v = [r for r in self.fake.video_requests if r.get("audio")]
        self.assertEqual(len(a2v), 4)
        for r in a2v:
            self.assertEqual(r["pipeline"], "two_stage")
            self.assertAlmostEqual(r["audio_s"], r["num_frames"] / 24.0, places=2)
            self.assertEqual((r["num_frames"] - 1) % 8, 0)
        broll = [r for r in self.fake.video_requests if not r.get("audio")]
        self.assertTrue(all(r["pipeline"] == "one_stage" and r.get("last_frame_image") for r in broll))
        # stills were edits of the bible's approved stills, never the unapproved one
        img = self.fake.bodies("/v1/images/generations")
        self.assertTrue(img and all(b["mode"] == "edit" and len(b.get("ref_images") or []) <= 3 for b in img))
        # the brain is home and gpud's own media models went away one at a time
        self.assertTrue(self.run_until(lambda: BRAIN in self.mlx.resident_models(), timeout=10))
        ev = [e["data"].get("model") for e in self.sup.store.events("model_unloaded")]
        self.assertNotIn(FOREIGN, ev)
        # ads status / inspect see it
        from foreman import cli
        out = io.StringIO()
        old = os.environ.get("FOREMAN_HOME")
        os.environ["FOREMAN_HOME"] = self.home
        try:
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["--db", os.path.join(self.tmp, "foreman.db"), "ads", "status"]), 0)
                self.assertEqual(cli.main(["--db", os.path.join(self.tmp, "foreman.db"), "ads", "inspect",
                                           str(res["job_id"])]), 0)
        finally:
            if old is None:
                os.environ.pop("FOREMAN_HOME", None)
            else:
                os.environ["FOREMAN_HOME"] = old
        text = out.getvalue()
        self.assertIn("review", text)
        self.assertIn("QA      pass", text)
        self.assertIn("picked seed", text)


class E1Clone(AdHarness):
    clone = True

    @needs_ffmpeg
    def test_a_qwen3_clone_voice_sends_the_reference_wav(self):
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        res = self.enqueue()
        self.assertTrue(self.run_until(lambda: os.path.exists(os.path.join(res["workdir"], "voice", "track.wav")),
                                       timeout=60), self.trail())
        speech = self.fake.bodies("/v1/audio/speech")
        self.assertTrue(speech and all(b.get("ref_audio") and b.get("language") == "English" for b in speech))
        self.assertTrue(all(b["model"].startswith("Qwen3-TTS") for b in speech), [b["model"] for b in speech])


@needs_ffmpeg
class E2Failures(AdHarness):

    def test_an_infrastructure_failure_is_retried_and_resumes_from_the_files(self):
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        self.fake.script["/v1/images/generations"].extend(["ok", "http500"])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "review", self.trail())
        stills = self.jobs("ad_stills")
        self.assertEqual([j.state for j in stills], ["failed", "succeeded"], self.trail())
        self.assertEqual(stills[1].attempt, 2, "the retry is the same task")
        meta = [n for n in os.listdir(os.path.join(res["workdir"], "stills")) if n.endswith(".meta.json")]
        self.assertEqual(len(meta), 6)
        # the first still was not rendered twice: the retry skipped it
        self.assertEqual(len(self.fake.bodies("/v1/images/generations")), 2 + 5)

    def test_a_black_take_is_dropped_and_the_others_carry_the_shot(self):
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        self.fake.script["/v1/video/generations"].extend(["black"])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "review", self.trail())
        pick = paths.read_json(os.path.join(res["workdir"], "pick.json"))
        self.assertEqual(len(pick["0"]["takes"]), 1, "the black take is not a candidate")
        failed = [j for j in self.jobs("ad_shots") if j.state == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].outcome["reason"], "validator")

    def test_a_script_that_never_validates_is_a_content_failure(self):
        bad = dict(SCRIPT, body=SCRIPT["body"][:2] + ["Our customers love it: five stars from thousands of reviews."])
        self.fake.chat_replies.extend([json.dumps(bad), json.dumps(bad)])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"]), timeout=40), self.trail())
        it = self.item(res["item"])
        self.assertEqual(it["state"], "failed")
        self.assertEqual(it["fail_class"], "content")
        state = paths.read_json(os.path.join(res["workdir"], "state.json"))
        self.assertIn("testimonial", state["script_errors"])
        self.assertIn("not in the facts", state["script_errors"], "five is a number the facts do not state")
        self.assertEqual(self.jobs("ad_voice"), [], "nothing renders after a failed script")

    def test_a_take_lost_with_the_engine_is_retried_and_the_ad_finishes(self):
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        self.fake.script["/v1/video/generations"].append("die")
        res = self.enqueue()

        def revive():
            if not self.fake.health_ok and any(j.state in ("failed", "quarantined") for j in self.jobs("ad_shots")):
                self.fake.health_ok = True
            return self.item(res["item"])["state"] != "queued"
        self.assertTrue(self.run_until(revive), self.trail())
        self.assertEqual(self.item(res["item"])["state"], "review", self.trail())
        lost = [j for j in self.jobs("ad_shots") if j.state != "succeeded"]
        self.assertEqual([j.outcome["reason"] for j in lost], ["engine_lost"])
        retry = [j for j in self.jobs("ad_shots") if j.task_key == lost[0].task_key and j.id != lost[0].id]
        self.assertEqual([j.state for j in retry], ["succeeded"], "the same step, retried")


@needs_ffmpeg
class E4ComfyStills(AdHarness):
    """Stills from a ComfyUI graph: gpud starts the (fake) ComfyUI process only after unloading its
    own mlx-serve media model, the graph runs, the server is stopped, and LTX loads after it."""

    def config(self):
        return ads_config(stills_backend="comfyui")

    def setUp(self):
        super().setUp()
        here = os.path.dirname(os.path.abspath(__file__))
        comfy_dir = os.path.join(self.tmp, "ComfyUI")
        os.makedirs(comfy_dir)
        with open(os.path.join(comfy_dir, "main.py"), "w") as f:
            f.write(f"import sys\nsys.path.insert(0, {here!r})\nfrom fake_comfyui import main\n"
                    f"raise SystemExit(main(sys.argv[1:]))\n")
        graphs = os.path.join(self.home, "comfyui", "graphs")
        os.makedirs(graphs)
        with open(os.path.join(graphs, "character_still.json"), "w") as f:
            json.dump({"graph": {"1": {"class_type": "LoadImage", "inputs": {"image": "{{image_1}}"}},
                                 "2": {"class_type": "QwenImageEdit", "inputs": {"prompt": "{{prompt}}", "seed": "{{seed}}",
                                                                                "width": "{{width}}", "height": "{{height}}"}},
                                 "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}},
                       "outputs": ["3"], "license": "Apache-2.0", "commercial_ok": True}, f)
        import socket
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        self.comfy = ComfyUIEngine(self.catalog, comfy_dir=comfy_dir, python=sys.executable, port=port,
                                   graphs_dir=graphs, artifacts_dir=self.art, home=self.home,
                                   mem_probe=lambda: 200.0, media_probe=self.mlx.resident_media, start_timeout_s=20.0,
                                   stop_wait_s=3.0, poll_s=0.05, name=studio.COMFY_ENGINE)
        self.sup.engines[studio.COMFY_ENGINE] = self.comfy

    def tearDown(self):
        try:
            self.comfy._stop_process()
        finally:
            super().tearDown()

    def test_comfy_stills_between_mlx_media_models(self):
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        self.assertEqual(self.item(res["item"])["state"], "review", self.trail())
        stills = self.jobs("ad_stills")
        self.assertTrue(stills and all(j.requires == studio.COMFYUI for j in stills))
        self.assertEqual(self.fake.bodies("/v1/images/generations"), [], "no mlx-serve image model was used")
        order = [(e["kind"], e["data"].get("model")) for e in self.sup.store.events()
                 if e["kind"] in ("load_started", "model_unloaded", "model_evicted")]
        comfy_load = order.index(("load_started", studio.COMFYUI))
        asr_gone = next(i for i, x in enumerate(order) if x[1] == studio.QWEN3_ASR and x[0] != "load_started")
        ltx_load = order.index(("load_started", studio.LTX))
        self.assertLess(asr_gone, comfy_load, "gpud's own media model went first")
        self.assertLess(comfy_load, ltx_load)
        gone = [i for i, x in enumerate(order) if x[1] == studio.COMFYUI and x[0] != "load_started"]
        self.assertTrue(gone and gone[0] < ltx_load, ("ComfyUI is gone before LTX loads", order))
        meta = paths.read_json(os.path.join(res["workdir"], "stills", "shot-0-a.png.meta.json"))
        self.assertEqual((meta["graph"], meta["commercial_ok"]), ("character_still", True))
        manifest = paths.read_json(os.path.join(res["workdir"], "manifest.json"))
        self.assertEqual(manifest["versions"].get("comfyui"), "0.43.0")


class E3Retention(AdHarness):

    @needs_ffmpeg
    def test_alternates_go_after_the_verdict_and_the_final_stays(self):
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        wd = res["workdir"]
        pick = paths.read_json(os.path.join(wd, "pick.json"))
        alts = [a for v in pick.values() for a in v["alternates"]]
        self.assertTrue(alts)
        self.assertTrue(all(os.path.exists(os.path.join(wd, a)) for a in alts))
        ledger.record_verdict(self.sup.store, res["job_id"], "reject", 2.0, "not this one", self.sup._now())
        self.planner._last_upkeep = None
        self.clock.t += 120
        self.sup.tick()
        self.assertFalse(any(os.path.exists(os.path.join(wd, a)) for a in alts), "alternates deleted after the verdict")
        self.assertTrue(os.path.exists(os.path.join(wd, "final", "ad.mp4")))
        self.assertTrue(os.path.exists(os.path.join(wd, "manifest.json")))
        self.assertTrue(all(os.path.exists(os.path.join(wd, v["take"])) for v in pick.values()))
        self.assertTrue(self.sup.store.events("ad_alternates_pruned"))

    @needs_ffmpeg
    def test_the_phone_sends_the_finished_mp4(self):
        from foreman import phone
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        res = self.enqueue()
        self.assertTrue(self.run_until(self.done(res["item"])), self.trail())
        it = self.item(res["item"])
        svc = phone.Phone.__new__(phone.Phone)
        svc.cfg = type("C", (), {"artifacts_dir": self.art})()
        svc.pc = type("P", (), {"upload_max_mb": 50, "video_max_mb": 50, "transcode_timeout_s": 30})()
        svc.ffmpeg = None
        svc.tmp_dir = self.tmp
        attempts, note, _ = svc.plan_media(it)
        self.assertEqual(attempts[0].method, "sendVideo")
        self.assertEqual(attempts[0].path, os.path.join(res["workdir"], "final", "ad.mp4"))
        self.assertEqual((attempts[0].extra or {}).get("width"), 270)


if __name__ == "__main__":
    unittest.main()
