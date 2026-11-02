"""The engines the ad pipeline adds, against the fakes:

  G1  mlx-serve ops: speech (Kokoro and a Qwen3-TTS clone), transcription with word times,
      image edits with references, the full video request (audio-to-video padded and muxed),
      resumable ops, capability checks, LTX's request rules, path confinement, cancel
  G2  the on-demand ComfyUI engine: graph templates, process lifecycle (start, health, /free,
      SIGTERM then SIGKILL, pid-file adoption), uploads, outputs, interrupt, the memory rule
  G3  the local CPU engine: handlers, content failures, cancel kills the subprocess
  G4  gpud: solo_media (ComfyUI never beside an mlx-serve media model), a continuation's
      expected_s, engine-measured checks, and the ledger's 'continue' verdict
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unittest

from foreman import media, paths, studio, validators
from foreman.adapters import mlx_ops
from foreman.adapters.comfyui import ComfyUIEngine, GraphError, load_graph, render_graph
from foreman.adapters.local import ContentFailure, LocalEngine
from foreman.adapters.mlx_serve import MlxServeEngine
from foreman.engine import RunSpec
from foreman.policy import ModelSpec, Policy
from foreman.sim import FakeEngine, SimHost, VirtualClock
from foreman.supervisor import AdmissionError, Supervisor

try:
    from ad_fixtures import FFMPEG, TmpCase, needs_ffmpeg, png, wait_for, wav
    from fake_comfyui import FakeComfy
    from fake_mlx_serve import FakeMlxServe
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import FFMPEG, TmpCase, needs_ffmpeg, png, wait_for, wav
    from tests.fake_comfyui import FakeComfy
    from tests.fake_mlx_serve import FakeMlxServe

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = studio.catalog(False, ads=True)
KOKORO, TTS, ASR, KLEIN, LTX = studio.KOKORO, studio.QWEN3_TTS, studio.QWEN3_ASR, studio.FLUX_KLEIN, studio.LTX


def spec(model, kind="job", run_id="1.1", **payload) -> RunSpec:
    payload.setdefault("goal", "test")
    payload.setdefault("expected_s", 30.0)
    return RunSpec(run_id, int(run_id.split(".")[0]), kind, model, payload, 30.0)


class MlxCase(TmpCase):
    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.tmp, "home")
        self.art = os.path.join(self.home, "artifacts")
        os.makedirs(self.art)
        self.fake = FakeMlxServe().start()
        self.eng = MlxServeEngine(self.fake.url, catalog=CAT, model_dir="/fake/models", artifacts_dir=self.art,
                                  mem_probe=lambda: 200.0, read_poll_s=0.02, unload_wait_s=0.5, connect_timeout_s=2.0,
                                  call_timeout_s=2.0, forbidden=studio.FORBIDDEN, ffmpeg=FFMPEG,
                                  input_roots=[self.home], max_resident_models=None)

    def tearDown(self):
        self.eng.close()
        self.fake.stop()
        super().tearDown()

    def load(self, model):
        t = self.eng.load(model)
        self.assertTrue(wait_for(lambda: self.eng.poll_load(t).state == "ready"), self.eng.poll_load(t))

    def run_job(self, model, timeout=20.0, **payload):
        h = self.eng.start(spec(model, **payload))
        wait_for(lambda: self.eng.poll(h).status != "running", timeout)
        return self.eng.poll(h)


class G1MlxOps(MlxCase):

    def test_kokoro_lines_and_a_clone_from_a_reference_wav(self):
        self.load(KOKORO)
        p = self.run_job(KOKORO, workdir="ads/a1", ops=[
            {"op": "speech", "out": "voice/line-0.wav", "input": "Hello there.", "voice": "af_heart", "speed": 1.1},
            {"op": "speech", "out": "voice/full.wav", "input": "Hello there. Bye now.", "voice": "am_onyx,bm_george"}])
        self.assertEqual(p.status, "done", p)
        self.assertTrue(p.artifact["path"].endswith("voice/full.wav"), "the last op is the job's artifact")
        self.assertGreater(p.artifact["duration_s"], 0)
        self.assertTrue(validators.passed(validators.check("tts", {}, p.artifact)))
        bodies = self.fake.bodies("/v1/audio/speech")
        self.assertEqual([b["voice"] for b in bodies], ["af_heart", "am_onyx,bm_george"])
        self.assertNotIn("out", bodies[0], "Foreman's bookkeeping is never sent")
        self.load(TTS)
        ref = wav(os.path.join(self.home, "characters", "c", "voice", "reference.wav"), 2.0)
        p = self.run_job(TTS, run_id="2.1", workdir="ads/a1", ops=[
            {"op": "speech", "out": "voice/clone.wav", "input": "Cloned line.", "ref_audio_path": ref,
             "language": "English"}])
        self.assertEqual(p.status, "done", p)
        body = self.fake.bodies("/v1/audio/speech")[-1]
        self.assertTrue(body["ref_audio"] and body["language"] == "English" and "voice" not in body)

    def test_transcription_is_multipart_with_word_times(self):
        self.load(ASR)
        src = wav(os.path.join(self.art, "ads", "a2", "voice", "track.wav"), 3.0)
        self.fake.asr_text = "Get it today at example.com."
        p = self.run_job(ASR, workdir="ads/a2", ops=[{"op": "transcribe", "out": "timing/track.json",
                                                       "file_path": "voice/track.wav", "language": "en",
                                                       "prompt": "example.com"}])
        self.assertEqual(p.status, "done", p)
        got = paths.read_json(os.path.join(self.art, "ads", "a2", "timing", "track.json"))
        self.assertEqual([w["word"] for w in got["words"]], ["get", "it", "today", "at", "example.com"])
        req = [r for m, path, r in self.fake.requests if path == "/v1/audio/transcriptions"][-1]
        self.assertEqual((req["response_format"], req["timestamp_granularities[]"], req["language"], req["prompt"]),
                         ("verbose_json", "word", "en", "example.com"))
        self.assertEqual(req["file_bytes"], os.path.getsize(src))

    def test_text_without_the_word_times_a_job_needs_is_the_engines_failure(self):
        """An ASR whose forced aligner is missing answers verbose_json with no words: for the h3 judge
        that is the engine's failure (retried, nothing written), never a take that said nothing."""
        self.load(ASR)
        wav(os.path.join(self.art, "ads", "a2", "voice", "take.wav"), 2.0)
        self.fake.asr_resolver = lambda audio, fields: ("Get it today.", None)
        op = {"op": "transcribe", "out": "gate/take.asr.json", "file_path": "voice/take.wav", "language": "en"}
        out = os.path.join(self.art, "ads", "a2", "gate", "take.asr.json")
        p = self.run_job(ASR, workdir="ads/a2", ops=[dict(op, need_words=True)])
        self.assertEqual((p.status, p.reason), ("failed", "engine_error"), p)
        self.assertIn("no word times", p.error or "")
        self.assertFalse(os.path.exists(out), "nothing a retry would take for the answer")
        req = [r for m, path, r in self.fake.requests if path == "/v1/audio/transcriptions"][-1]
        self.assertNotIn("need_words", req, "Foreman's bookkeeping is never sent")
        p = self.run_job(ASR, run_id="2.1", workdir="ads/a2", ops=[op])
        self.assertEqual(p.status, "done", "without need_words the text alone is an answer")
        self.fake.asr_resolver = lambda audio, fields: ("", None)
        p = self.run_job(ASR, run_id="2.2", workdir="ads/a2", ops=[dict(op, out="gate/quiet.asr.json", need_words=True)])
        self.assertEqual(p.status, "done", "nothing said needs no word times")

    def test_image_edits_with_reference_stills(self):
        self.load(KLEIN)
        hero = png(os.path.join(self.home, "characters", "c", "stills", "hero.png"))
        refs = [png(os.path.join(self.home, "characters", "c", "stills", f"r{i}.png")) for i in range(3)]
        p = self.run_job(KLEIN, workdir="ads/a3", ops=[{"op": "image", "out": "stills/shot-0-a.png", "prompt": "x",
                                                         "size": "64x128", "seed": 5, "image_path": hero,
                                                         "ref_image_paths": refs}])
        self.assertEqual(p.status, "done", p)
        self.assertEqual((p.artifact["width"], p.artifact["height"]), (64, 128))
        body = self.fake.bodies("/v1/images/generations")[-1]
        self.assertEqual((body["mode"], len(body["ref_images"]), body["size"]), ("edit", 3, "64x128"))
        five = [png(os.path.join(self.home, "characters", "c", "stills", f"x{i}.png")) for i in range(5)]
        p = self.run_job(KLEIN, run_id="2.1", workdir="ads/a3", ops=[
            {"op": "image", "out": "stills/too-many.png", "prompt": "x", "image_path": hero, "ref_image_paths": five}])
        self.assertEqual((p.status, p.reason), ("failed", "rejected"), "the server refuses a 5th reference: a 4xx")

    @needs_ffmpeg
    def test_audio_to_video_pads_the_line_and_muxes_it_back(self):
        self.load(LTX)
        line = wav(os.path.join(self.art, "ads", "a4", "voice", "talk-0.wav"), 1.2, rate=24000)
        still = png(os.path.join(self.art, "ads", "a4", "stills", "shot-0-a.png"), 64, 128)
        p = self.run_job(LTX, workdir="ads/a4", ops=[
            {"op": "video", "out": "shots/shot-0-s11.mp4", "prompt": "speaks", "width": 64, "height": 128,
             "num_frames": 33, "seed": 11, "pipeline": "two_stage", "first_frame_path": still,
             "audio_path": "voice/talk-0.wav"}])
        self.assertEqual(p.status, "done", p)
        req = self.fake.video_requests[-1]
        self.assertAlmostEqual(req["audio_s"], 33 / 24.0, places=3)
        self.assertEqual(req["pipeline"], "two_stage")
        meta = paths.read_json(p.artifact["path"] + ".meta.json")
        self.assertEqual((meta["audio"], meta["seed"], meta["ok"], meta["frame_count"]), ("input", 11, True, 33))
        self.assertFalse(any(n.endswith(".rgb") for n in os.listdir(os.path.dirname(p.artifact["path"]))),
                         "raw frames deleted after the mux")
        self.assertFalse(os.path.exists(os.path.join(self.art, "ads", "a4", ".scratch")) and
                         os.listdir(os.path.join(self.art, "ads", "a4", ".scratch")), "scratch cleaned up")
        out = subprocess.run([FFMPEG, "-hide_banner", "-i", p.artifact["path"]], capture_output=True, text=True).stderr
        self.assertIn("Audio:", out)
        self.assertTrue(validators.passed(validators.check("video", {}, p.artifact)), p.artifact)
        self.assertGreaterEqual(os.path.getsize(line), 1)

    @needs_ffmpeg
    def test_a_retried_job_skips_the_ops_already_done(self):
        self.load(LTX)
        ops = [{"op": "video", "out": f"shots/b-{i}.mp4", "prompt": "walks", "width": 64, "height": 128,
                "num_frames": 9, "seed": i, "pipeline": "one_stage", "steps": 8} for i in range(2)]
        self.fake.script["/v1/video/generations"].extend(["ok", "http500"])
        p = self.run_job(LTX, workdir="ads/a5", ops=ops)
        self.assertEqual(p.status, "failed")
        self.assertIsNotNone(mlx_ops.completed(os.path.join(self.art, "ads", "a5", "shots", "b-0.mp4")))
        p = self.run_job(LTX, run_id="1.2", workdir="ads/a5", ops=ops)
        self.assertEqual(p.status, "done", p)
        self.assertEqual([r["seed"] for r in self.fake.video_requests], [0, 1, 1], "op 0 was not rendered again")
        self.assertEqual([o["skipped"] for o in p.artifact["ops"]], [True, None])

    def test_ltx_rules_are_checked_before_sending(self):
        self.assertEqual(mlx_ops.max_frames(704, 1280), 297)
        bad = {"pipeline": "two_stage", "width": 704, "height": 1260, "num_frames": 121}
        self.assertIn("divisible by 64", " ".join(mlx_ops.video_problems(bad)))
        self.assertIn("8k+1", " ".join(mlx_ops.video_problems(dict(bad, height=1280, num_frames=120))))
        self.assertIn("768 MiB", " ".join(mlx_ops.video_problems(dict(bad, height=1280, num_frames=305))))
        self.assertIn("two_stage", " ".join(mlx_ops.video_problems(
            {"pipeline": "one_stage", "width": 704, "height": 1280, "num_frames": 121, "audio_path": "x.wav"})))
        self.assertEqual(mlx_ops.video_problems({"pipeline": "one_stage", "width": 736, "height": 1280,
                                                 "num_frames": 121}), [], "one stage takes a 32 px grid")
        self.load(LTX)
        p = self.run_job(LTX, workdir="ads/a6", ops=[dict(bad, op="video", out="x.mp4", prompt="x")])
        self.assertEqual((p.status, p.reason), ("failed", "rejected"))
        self.assertEqual(self.fake.video_requests, [], "nothing was sent")

    def test_a_model_without_the_capability_is_refused(self):
        self.fake.discovered.add("mlx-community/Qwen3-ASR-1.7B-8bit")
        self.load(ASR)
        p = self.run_job(ASR, workdir="ads/a7", ops=[{"op": "transcribe", "out": "t.json", "file_path": "x.wav"}])
        self.assertEqual(p.status, "failed")
        self.load(KLEIN)
        orig = self.eng.capabilities
        self.eng.capabilities = lambda: {k: {"chat"} for k in orig()}
        p = self.run_job(KLEIN, run_id="2.1", workdir="ads/a7", ops=[{"op": "image", "out": "i.png", "prompt": "x"}])
        self.assertEqual(p.status, "failed")
        self.assertIn("capability", p.error)

    def test_paths_cannot_leave_their_roots(self):
        self.load(KOKORO)
        for ops, wd in (([{"op": "speech", "out": "../../escape.wav", "input": "x"}], "ads/a8"),
                        ([{"op": "speech", "out": "ok.wav", "input": "x"}], "../outside"),
                        ([{"op": "speech", "out": "ok.wav", "input": "x", "ref_audio_path": "/etc/hosts"}], "ads/a8")):
            p = self.run_job(KOKORO, run_id=f"{len(ops)}.{len(wd)}", workdir=wd, ops=ops)
            self.assertEqual((p.status, p.reason), ("failed", "rejected"), (ops, wd, p))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "escape.wav")))
        p = self.run_job(KOKORO, run_id="9.1", workdir="ads/a8", ops=[{"op": "music", "out": "m.wav", "prompt": "x"}])
        self.assertIn("cannot run on", p.error, "a TTS job cannot carry a music op")

    def test_cancel_stops_the_op_in_flight(self):
        self.load(KOKORO)
        self.fake.script["/v1/audio/speech"].append("hang")
        h = self.eng.start(spec(KOKORO, workdir="ads/a9", ops=[{"op": "speech", "out": "x.wav", "input": "x"}]))
        self.assertTrue(wait_for(lambda: self.fake.bodies("/v1/audio/speech")))
        self.eng.cancel(h)
        self.assertTrue(wait_for(lambda: self.eng.poll(h).status == "cancelled"))

    def test_a_request_without_events_gets_a_bounded_heartbeat(self):
        self.load(KLEIN)
        self.fake.image_delay = 0.6
        h = self.eng.start(spec(KLEIN, workdir="ads/b1", ops=[{"op": "image", "out": "i.png", "prompt": "x",
                                                                  "expected_s": 0.4}]))
        seen = []
        end = time.monotonic() + 5
        while time.monotonic() < end and self.eng.poll(h).status == "running":
            seen.append(self.eng.poll(h).progress)
            time.sleep(0.02)
        self.assertEqual(self.eng.poll(h).status, "done")
        self.assertEqual(seen, sorted(seen), "progress never goes back")
        self.assertTrue(any(0 < x < 0.95 for x in seen), "the heartbeat moved while the server worked")
        self.assertLess(max(seen), 0.951, "and it stays inside the op's share")

    def test_resident_media_lists_media_models_whoever_loaded_them(self):
        self.fake.other_client_load("other-flux-model")
        self.fake.ready["some-chat-model"] = int(1e9)
        self.load(KOKORO)
        media_ids = self.eng.resident_media()
        self.assertIn("Kokoro-82M-MLX-Serve", media_ids)
        self.assertIn("other-flux-model", media_ids, "an image model someone else loaded counts")
        self.assertNotIn("some-chat-model", media_ids)


# ---------------------------------------------------------------- G2 ComfyUI

GRAPH = {"graph": {"1": {"class_type": "LoadImage", "inputs": {"image": "{{image_1}}"}},
                   "2": {"class_type": "EditNode", "inputs": {"prompt": "A {{prompt}} portrait", "seed": "{{seed}}",
                                                            "width": "{{width}}", "height": "{{height}}", "image": ["1", 0]}},
                   "9": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "foreman"}}},
         "outputs": ["9"], "defaults": {"width": 64, "height": 96}, "license": "Apache-2.0", "commercial_ok": True}


class G2Graphs(TmpCase):
    def test_placeholders_keep_their_types_and_missing_ones_are_named(self):
        g = render_graph(GRAPH["graph"], {"image_1": "a.png", "prompt": "calm", "seed": 7, "width": 64, "height": 96})
        self.assertEqual(g["2"]["inputs"]["seed"], 7)
        self.assertEqual(g["2"]["inputs"]["prompt"], "A calm portrait")
        self.assertEqual(g["2"]["inputs"]["image"], ["1", 0])
        with self.assertRaises(GraphError) as e:
            render_graph(GRAPH["graph"], {"prompt": "x"})
        self.assertIn("image_1", str(e.exception))

    def test_templates_load_from_the_directory_only(self):
        d = os.path.join(self.tmp, "graphs")
        os.makedirs(d)
        with open(os.path.join(d, "still.json"), "w") as f:
            json.dump(GRAPH, f)
        t = load_graph(d, "still")
        self.assertEqual((t["outputs"], t["commercial_ok"], t["defaults"]["width"]), (["9"], True, 64))
        for bad in ("../still", "/etc/passwd", ".hidden", "missing"):
            with self.assertRaises(GraphError):
                load_graph(d, bad)


class ComfyCase(TmpCase):
    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.tmp, "home")
        self.art = os.path.join(self.home, "artifacts")
        self.comfy_dir = os.path.join(self.tmp, "ComfyUI")
        os.makedirs(self.art)
        os.makedirs(self.comfy_dir)
        with open(os.path.join(self.comfy_dir, "main.py"), "w") as f:
            f.write(f"import sys\nsys.path.insert(0, {HERE!r})\nfrom fake_comfyui import main\nraise SystemExit(main(sys.argv[1:]))\n")
        gd = os.path.join(self.home, "comfyui", "graphs")
        os.makedirs(gd)
        with open(os.path.join(gd, "still.json"), "w") as f:
            json.dump(GRAPH, f)
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.busy = []
        self.free_gb = 200.0
        self.eng = self.engine()

    def engine(self, **kw):
        opts = dict(comfy_dir=self.comfy_dir, python=sys.executable, port=self.port, graphs_dir=None,
                    artifacts_dir=self.art, home=self.home, mem_probe=lambda: self.free_gb,
                    media_probe=lambda: list(self.busy), start_timeout_s=20.0, stop_wait_s=3.0, unload_wait_s=5.0,
                    poll_s=0.05, name=studio.COMFY_ENGINE)
        opts.update(kw)
        opts["graphs_dir"] = opts["graphs_dir"] or os.path.join(self.home, "comfyui", "graphs")
        return ComfyUIEngine(CAT, **opts)

    def tearDown(self):
        try:
            self.eng._stop_process()
        finally:
            super().tearDown()

    def start(self):
        t = self.eng.load(studio.COMFYUI)
        self.assertTrue(wait_for(lambda: self.eng.poll_load(t).state != "loading", 25.0), self.eng.poll_load(t))
        return self.eng.poll_load(t)

    def run_graph(self, run_id="1.1", **op):
        hero = png(os.path.join(self.home, "characters", "c", "stills", "hero.png"))
        base = {"op": "graph", "graph": "still", "out": "stills/s0.png", "images": {"image_1": hero},
                "params": {"prompt": "calm", "seed": 3}}
        base.update(op)
        h = self.eng.start(spec(studio.COMFYUI, run_id=run_id, workdir="ads/c1", ops=[base]))
        wait_for(lambda: self.eng.poll(h).status != "running", 25.0)
        return self.eng.poll(h)


class G2ComfyEngine(ComfyCase):

    def test_only_foremans_own_inputs_and_outputs_are_removed(self):
        """ComfyUI's input and output folders would grow by every take's pictures, voice and video."""
        files = {rel: os.path.join(self.comfy_dir, rel) for rel in (
            "input/foreman-1_1-image_1.png", "input/someone-else.png", "output/video/foreman_h3_00001_.mp4",
            "output/video/theirs_00001_.mp4", "foreman-outside.txt")}
        for path in files.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").close()
        self.eng._remove_own("input", "foreman-1_1-image_1.png")
        self.eng._remove_own("input", "someone-else.png")
        self.eng._remove_own("output", "video/foreman_h3_00001_.mp4")
        self.eng._remove_own("output", "video/theirs_00001_.mp4")
        self.eng._remove_own("input", "../foreman-outside.txt")
        self.eng._remove_own("input", "missing-foreman.png")
        self.assertEqual(sorted(rel for rel, path in files.items() if os.path.exists(path)),
                         ["foreman-outside.txt", "input/someone-else.png", "output/video/theirs_00001_.mp4"])

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("taskpolicy"), "macOS background band")
    def test_comfyui_starts_outside_launchds_background_band(self):
        from foreman.adapters import comfyui
        probe = "import os; print(os.getpriority(4, 0))"  # PRIO_DARWIN_PROCESS: 1 in the background band
        plain = subprocess.run(["taskpolicy", "-b", sys.executable, "-c", probe], capture_output=True, text=True)
        lifted = subprocess.run(["taskpolicy", "-b", sys.executable, "-c", comfyui.UNBACKGROUND, sys.executable,
                                 "-c", probe], capture_output=True, text=True)
        self.assertEqual((plain.stdout.strip(), lifted.stdout.strip()), ("1", "0"), lifted.stderr)

    def test_start_run_free_and_stop_after_the_job(self):
        self.assertEqual(self.eng.resident_models(), set())
        self.assertEqual(self.start().state, "ready")
        self.assertEqual(self.eng.resident_models(), {studio.COMFYUI})
        pid = self.eng._pid
        self.assertTrue(os.path.exists(os.path.join(self.home, "run", "comfyui.json")))
        p = self.run_graph()
        self.assertEqual(p.status, "done", p)
        out = os.path.join(self.art, "ads", "c1", "stills", "s0.png")
        self.assertEqual(media.image_size(out), (64, 96), "the graph's default size came through")
        meta = paths.read_json(out + ".meta.json")
        self.assertEqual((meta["license"], meta["commercial_ok"], meta["comfyui_version"]), ("Apache-2.0", True, "0.43.0"))
        self.assertTrue(wait_for(lambda: not self.eng._alive(pid), 5.0), "exit_after_job stopped the server")
        self.assertEqual(self.eng.resident_models(), set())
        self.assertFalse(os.path.exists(os.path.join(self.home, "run", "comfyui.json")))

    def test_a_retry_skips_a_finished_graph_and_a_bad_graph_is_a_content_failure(self):
        self.eng = self.engine(exit_after_job=False)
        self.start()
        self.assertEqual(self.run_graph().status, "done")
        p = self.run_graph(run_id="1.2")
        self.assertEqual(p.artifact["ops"][0]["skipped"], True)
        p = self.run_graph(run_id="2.1", out="stills/s1.png", graph="missing")
        self.assertEqual((p.status, p.reason), ("failed", "rejected"))
        p = self.run_graph(run_id="3.1", out="stills/s2.png", params={"seed": 1})
        self.assertEqual((p.status, p.reason), ("failed", "rejected"), "a placeholder without a value")

    def test_it_never_starts_beside_an_mlx_serve_media_model_or_without_headroom(self):
        self.busy = ["Kokoro-82M-MLX-Serve"]
        st = self.start()
        self.assertEqual((st.state, st.deferred), ("failed", True))
        self.assertIn("never runs beside", st.error)
        self.busy = []
        self.free_gb = 70.0
        st = self.start()
        self.assertEqual((st.state, st.deferred), ("failed", True))
        self.assertIsNone(self.eng._pid, "nothing was started")

    def test_a_comfyui_foreman_did_not_start_is_never_used_or_stopped(self):
        other = FakeComfy(port=self.port).start()
        try:
            st = self.start()
            self.assertEqual((st.state, st.deferred), ("failed", False))
            self.assertIn("someone else's", st.error)
            self.assertEqual(self.eng.resident_models(), set())
            self.assertTrue(self.eng.unload(studio.COMFYUI))
        finally:
            other.stop()

    def test_sigkill_only_after_sigterm_was_ignored(self):
        os.environ["FAKE_COMFY_IGNORE_TERM"] = "1"
        try:
            self.eng = self.engine(exit_after_job=False, stop_wait_s=1.0)
            self.start()
            pid = self.eng._pid
            self.assertTrue(self.eng.unload(studio.COMFYUI))
            self.assertFalse(self.eng._alive(pid))
        finally:
            os.environ.pop("FAKE_COMFY_IGNORE_TERM", None)

    def test_a_restarted_gpud_adopts_the_server_it_started(self):
        self.eng = self.engine(exit_after_job=False)
        self.start()
        pid = self.eng._pid
        again = self.engine(exit_after_job=False)
        self.assertEqual(again._pid, pid)
        self.assertEqual(again.resident_models(), {studio.COMFYUI})
        self.assertTrue(again.unload(studio.COMFYUI))
        self.assertFalse(self.eng._alive(pid))

    def test_cancel_interrupts_the_graph(self):
        os.environ["FAKE_COMFY_RUN_S"] = "30"
        try:
            self.eng = self.engine(exit_after_job=False)
            self.start()
        finally:
            os.environ.pop("FAKE_COMFY_RUN_S", None)
        hero = png(os.path.join(self.home, "characters", "c", "stills", "hero.png"))
        h = self.eng.start(spec(studio.COMFYUI, workdir="ads/c2", ops=[
            {"op": "graph", "graph": "still", "out": "x.png", "images": {"image_1": hero},
             "params": {"prompt": "p", "seed": 1}, "expected_s": 60}]))
        self.assertTrue(wait_for(lambda: self.eng._runs[h].prompt_id, 10.0))
        self.eng.cancel(h)
        self.assertTrue(wait_for(lambda: self.eng.poll(h).status == "cancelled", 10.0), self.eng.poll(h))
        self.assertFalse(os.path.exists(os.path.join(self.art, "ads", "c2", "x.png")))


# ---------------------------------------------------------------- G3 the local engine

class G3Local(TmpCase):
    def make(self, handler):
        return LocalEngine(CAT, {"ad_facts": handler}, artifacts_dir=self.tmp, name=studio.LOCAL_ENGINE)

    def run_it(self, eng, kind="ad_facts"):
        eng.load(studio.LOCAL)
        h = eng.start(spec(studio.LOCAL, kind=kind))
        wait_for(lambda: eng.poll(h).status != "running", 10.0)
        return h, eng.poll(h)

    def test_success_content_failure_and_crash(self):
        _, p = self.run_it(self.make(lambda s, ctx: (ctx.progress(0.5), {"path": "/x", "exists": True})[1]))
        self.assertEqual((p.status, p.progress), ("done", 1.0))

        def bad(s, ctx):
            raise ContentFailure("the facts say otherwise")
        _, p = self.run_it(self.make(bad))
        self.assertEqual((p.status, p.reason), ("failed", "rejected"))

        def crash(s, ctx):
            raise OSError("disk went away")
        _, p = self.run_it(self.make(crash))
        self.assertEqual((p.status, p.reason), ("failed", None))
        _, p = self.run_it(self.make(crash), kind="ad_unknown")
        self.assertEqual(p.reason, "rejected")

    def test_cancel_kills_its_subprocess(self):
        started = threading.Event()

        def slow(s, ctx):
            started.set()
            ctx.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=60)
            return {"path": "/x", "exists": True}
        eng = self.make(slow)
        eng.load(studio.LOCAL)
        h = eng.start(spec(studio.LOCAL, kind="ad_facts"))
        self.assertTrue(started.wait(5))
        self.assertTrue(wait_for(lambda: eng._runs[h].procs, 5.0))
        proc = eng._runs[h].procs[0]
        eng.cancel(h)
        self.assertTrue(wait_for(lambda: eng.poll(h).status == "cancelled", 10.0))
        self.assertIsNotNone(proc.poll(), "the child is gone")
        self.assertEqual(eng.poll("nope").status, "lost")
        self.assertTrue(eng.unload(studio.LOCAL))
        self.assertEqual(eng.resident_models(), set())


# ---------------------------------------------------------------- G4 gpud

SOLO_CAT = {
    "brain": ModelSpec("brain", "a", 20, 10, "brain", vision=True),
    "tts": ModelSpec("tts", "a", 5, 5, "tts"),
    "video": ModelSpec("video", "a", 40, 10, "video"),
    "cpu": ModelSpec("cpu", "local", 0.5, 1, "cpu"),
    "comfy": ModelSpec("comfy", "b", 60, 10, "comfy", solo_media=True),
}


class G4Supervisor(TmpCase):
    def make(self):
        self.clock = VirtualClock(0.0)
        host = SimHost(256.0)
        from foreman.sim import ScriptedBrain
        brain = ScriptedBrain()
        self.a = FakeEngine("a", self.clock, host, SOLO_CAT, brain=brain)
        self.b = FakeEngine("b", self.clock, host, SOLO_CAT, brain=brain)
        self.l = FakeEngine("local", self.clock, host, SOLO_CAT, brain=brain)
        pol = Policy(text_ladder=("brain",), vision_brain="brain", co_resident_brain="brain")
        self.sup = Supervisor(os.path.join(self.tmp, "g4.db"), {"a": self.a, "b": self.b, "local": self.l}, self.clock,
                              pol, catalog=SOLO_CAT)
        self.sup.recover()

    def tick_until(self, pred, max_s=3000):
        for _ in range(int(max_s)):
            self.clock.advance(1.0)
            self.sup.tick()
            if pred():
                return True
        return False

    def residency(self) -> list:
        return [(e["kind"], e["data"].get("model")) for e in self.sup.store.events()
                if e["kind"] in ("load_started", "model_unloaded")]

    def submit(self, model, kind="job"):
        return self.sup.submit(kind, model, {"goal": "g", "expected_s": 30.0},
                               {"kind": "review", "needs": "any"}, {"kind": "recover", "needs": "any"})

    def test_comfy_never_shares_the_host_with_another_media_model(self):
        self.make()
        j1 = self.submit("tts")
        self.assertTrue(self.tick_until(lambda: self.sup.store.job(j1).state == "succeeded"))
        self.assertIn("tts", self.a.resident_models())
        j2 = self.submit("comfy")
        self.assertTrue(self.tick_until(lambda: self.sup.store.job(j2).state == "succeeded"))
        order = self.residency()
        self.assertLess(order.index(("model_unloaded", "tts")), order.index(("load_started", "comfy")), order)
        self.assertIn("brain", self.a.resident_models(), "the brain stays beside ComfyUI")
        j3 = self.submit("video")
        self.assertTrue(self.tick_until(lambda: self.sup.store.job(j3).state == "succeeded"))
        order = self.residency()
        self.assertLess(order.index(("model_unloaded", "comfy")), order.index(("load_started", "video")), order)
        j4 = self.submit("cpu")
        self.assertTrue(self.tick_until(lambda: self.sup.store.job(j4).state == "succeeded"))
        self.assertIn("video", self.a.resident_models(), "CPU work holds no GPU memory: nothing unloaded for it")

    def test_someone_elses_media_model_makes_comfy_wait(self):
        self.make()
        self.a.load_as_other_client("tts")
        self.assertTrue(self.tick_until(lambda: "tts" in self.sup._resident, 5))
        j = self.submit("comfy")
        self.tick_until(lambda: False, 30)
        self.assertEqual(self.sup.store.job(j).state, "queued")
        self.assertNotIn(("load", "comfy"), [(c, m) for _, c, m in self.b.calls])
        self.assertIn("tts", self.a.resident_models(), "someone else's model is never unloaded")
        self.assertTrue(self.sup.store.events("media_blocked"))

    def test_a_continuation_can_size_its_brain_job(self):
        self.make()
        j = self.sup.submit("job", "tts", {"goal": "g", "expected_s": 30.0},
                            {"kind": "review", "needs": "any", "expected_s": 300}, {"kind": "recover"})
        self.assertTrue(self.tick_until(lambda: any(x.kind == "review" for x in self.sup.store.jobs())))
        review = [x for x in self.sup.store.jobs() if x.kind == "review"][0]
        self.assertEqual(review.payload["expected_s"], 300)
        self.assertEqual(self.sup.p.hard_timeout_s("brain", review.payload), 2 * 1.25 * 300)
        for bad in (0, -5, 1e6, "300"):
            with self.assertRaises(AdmissionError):
                self.sup.submit("job", "tts", {"goal": "g", "expected_s": 30.0},
                                {"kind": "review", "expected_s": bad}, {"kind": "recover"})


class G4Ledger(TmpCase):
    """An ad's lineage has many steps: a take it carried on without, or one crash per step, does not
    turn an infrastructure failure into a content failure."""

    def lineage(self, *steps):
        from foreman.store import Store
        store = Store(os.path.join(self.tmp, f"l{len(steps)}-{time.monotonic_ns()}.db"))
        self.addCleanup(store.close)
        root, jobs = None, []
        for kind, reason, after, task in steps:
            state = "succeeded" if reason is None else "failed"
            jid, _ = store.insert_job(kind=kind, requires=LTX, state=state, payload={}, root_id=root, max_attempts=3,
                                      outcome={"status": state, "reason": reason}, created_at=0.0, task_key=task)
            root = root or jid
            store.update_job(jid, root_id=root)
            if after is not None:
                store.add_continuation(jid, {"kind": "ad_next"}, {"kind": "ad_recover"})
                cid, _ = store.insert_job(kind="ad_recover", requires="brain", state="succeeded", payload={},
                                          outcome={"status": "succeeded", "artifact": {"decision": {"verdict": after}}},
                                          root_id=root, max_attempts=3, created_at=0.0)
                store.mark_fired(jid, "failure" if reason else "success", cid, 0.0)
            jobs.append(store.job(jid))
        return store, jobs

    def test_carried_on_takes_and_one_crash_per_step(self):
        from foreman import ledger
        store, jobs = self.lineage(("ad_shots", "validator", "continue", "t1"), ("ad_shots", None, "continue", "t2"),
                                   ("ad_stills", "crash", "retry", "t3"), ("ad_music", "engine_lost", None, "t4"))
        self.assertEqual(ledger.failure_class(store, jobs)[0], "infra")
        store, jobs = self.lineage(("ad_stills", "crash", "retry", "t3"), ("ad_stills", "crash", None, "t3"))
        self.assertEqual(ledger.failure_class(store, jobs)[0], "content", "the same step crashing again")
        store, jobs = self.lineage(("ad_shots", "validator", "give_up", "t1"))
        self.assertEqual(ledger.failure_class(store, jobs)[0], "content")


class G4Checks(unittest.TestCase):
    def test_engine_checks_can_only_add_failures(self):
        art = {"path": "/x.wav", "exists": True, "duration_s": 2.0, "silence_ratio": 0.0,
               "checks": [{"check": "a.wav:duration", "ok": True, "detail": ""}]}
        self.assertTrue(validators.passed(validators.check("tts", {}, art)))
        art["checks"].append({"check": "b.wav:silence", "ok": False, "detail": "silent"})
        res = validators.check("tts", {}, art)
        self.assertFalse(validators.passed(res))
        self.assertIn("b.wav:silence", validators.failures(res))
        res = validators.check("tts", {}, dict(art, silence_ratio=1.0, checks=[{"check": "x", "ok": True}]))
        self.assertFalse(validators.passed(res), "a passing engine check never excuses a failing validator")
        self.assertFalse(validators.passed(validators.check("cpu", {}, dict(art, checks="nope"))))
        self.assertFalse(validators.passed(validators.check("cpu", {}, dict(art, checks=[{"check": "y", "ok": "yes"}]))))


if __name__ == "__main__":
    unittest.main()
