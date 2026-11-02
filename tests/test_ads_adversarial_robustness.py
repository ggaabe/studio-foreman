"""Red team, robustness and resources lens, for the ad pipeline (foreman/ads.py, ad_pipeline.py,
assembly.py, adapters/mlx_ops.py, adapters/comfyui.py, adapters/local.py).

Every test asserts the behaviour the owner's rules and the design ask for; a failing test is a
defect, not a test to relax. Everything here is generic (an online service, a mascot character,
example.com) and runs against the fakes (tests/fake_mlx_serve.py, tests/fake_comfyui.py) and,
where noted, real ffmpeg on tiny synthetic inputs (skipped without it).

  R1  ComfyUI's process: hung, adopted, orphaned, shut down, the memory rule on a server that
      is already running, and (under gpud) a health blip that must not turn Foreman's own
      server into "someone else's"
  R2  partial files and malformed answers: truncated downloads, non-image outputs, a failed
      write, a killed font copy, orphaned raw frames, a giant ASR answer
  R3  infrastructure versus content: what the owner's caps and the weekly budget may be charged
      for (a missing font, no ffmpeg, a rate-limited facts URL)
  R4  kills: gpud's children die with it, the dispatcher re-runs idempotently, gpud killed at its
      crash points between every pair of steps (and in the middle of a take) renders each thing once
  R5  durations and limits: a 60 s template, the shortest 8k+1 clip, frame rounding drift across
      cuts, LTX's limits, a percent sign on the end card and the AI label, an ordinary URL that
      runs off the end card
  R6  disk: the guard counts an ad for its whole life, and `ads enqueue` counts the ad itself
"""
from __future__ import annotations

import builtins
import errno
import http.server
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest

from foreman import ad_pipeline, ads, assembly, characters, ledger, media, paths, studio
from foreman.adapters import mlx_ops
from foreman.adapters.comfyui import ComfyUIEngine
from foreman.adapters.local import ContentFailure, LocalEngine
from foreman.adapters.mlx_serve import MlxServeEngine
from foreman.ads import AdsConfig, plan_shots
from foreman.engine import RunSpec
from foreman.httpio import json_call
from foreman.planner import Planner, PlannerConfig
from foreman.store import Store
from foreman.supervisor import AdmissionError
from foreman.templates import TemplateError, load_templates, parse_template

try:
    from ad_fixtures import (FFMPEG, REVIEW_OK, SCRIPT, TmpCase, ads_config, make_home, needs_ffmpeg, png,
                             probe_frame, template_data, wait_for, wav)
    from fake_mlx_serve import FakeMlxServe
    from test_ads_e2e import AdHarness
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import (FFMPEG, REVIEW_OK, SCRIPT, TmpCase, ads_config, make_home, needs_ffmpeg, png,
                                   probe_frame, template_data, wait_for, wav)
    from tests.fake_mlx_serve import FakeMlxServe
    from tests.test_ads_e2e import AdHarness

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CAT = studio.catalog(False, ads=True)
GRAPH = {"graph": {"1": {"class_type": "LoadImage", "inputs": {"image": "{{image_1}}"}},
                   "2": {"class_type": "EditNode", "inputs": {"prompt": "{{prompt}}", "seed": "{{seed}}",
                                                            "width": "{{width}}", "height": "{{height}}",
                                                            "image": ["1", 0]}},
                   "9": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}},
         "outputs": ["9"], "defaults": {"width": 64, "height": 96}, "license": "Apache-2.0", "commercial_ok": True}

# A stand-in ComfyUI process (tests/fake_comfyui.py) whose /view can misbehave on request.
COMFY_MAIN = '''import os, socket, sys, urllib.parse
sys.path.insert(0, {here!r})
import fake_comfyui as fc
MODE = os.environ.get("RT_VIEW_MODE", "")
_orig = fc._Handler.do_GET


def do_GET(self):
    url = urllib.parse.urlsplit(self.path)
    if url.path == "/view" and MODE:
        name = (urllib.parse.parse_qs(url.query).get("filename") or [""])[0]
        with self.state.lock:
            data = self.state.outputs.get(name) or b""
        if MODE == "truncate":   # the connection drops halfway through the body
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[: max(1, len(data) // 2)])
            self.wfile.flush()
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return
        if MODE == "html":       # a proxy's error page answered with 200
            body = b"<html><body>502 Bad Gateway</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
    return _orig(self)


fc._Handler.do_GET = do_GET
raise SystemExit(fc.main(sys.argv[1:]))
'''


def spec(model, kind="job", run_id="1.1", **payload) -> RunSpec:
    payload.setdefault("goal", "test")
    payload.setdefault("expected_s", 30.0)
    return RunSpec(run_id, int(run_id.split(".")[0]), kind, model, payload, 30.0)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def alive(pid: int) -> bool:
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def bright(frame: bytes, lo: int = 0, hi: int | None = None, threshold: int = 200) -> int:
    return sum(1 for b in frame[lo:hi] if b > threshold)


# ---------------------------------------------------------------- R1 ComfyUI's process

class ComfyBase(TmpCase):
    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.tmp, "home")
        self.art = os.path.join(self.home, "artifacts")
        self.comfy_dir = os.path.join(self.tmp, "ComfyUI")
        os.makedirs(self.art)
        os.makedirs(self.comfy_dir)
        with open(os.path.join(self.comfy_dir, "main.py"), "w") as f:
            f.write(COMFY_MAIN.format(here=HERE))
        self.graphs = os.path.join(self.home, "comfyui", "graphs")
        os.makedirs(self.graphs)
        with open(os.path.join(self.graphs, "still.json"), "w") as f:
            json.dump(GRAPH, f)
        self.port = free_port()
        self.busy = []
        self.hero = png(os.path.join(self.home, "characters", "c", "stills", "hero.png"))
        self.pidfile = os.path.join(self.home, "run", "comfyui.json")

    def engine(self, **kw) -> ComfyUIEngine:
        opts = dict(comfy_dir=self.comfy_dir, python=sys.executable, port=self.port, graphs_dir=self.graphs,
                    artifacts_dir=self.art, home=self.home, mem_probe=lambda: 200.0,
                    media_probe=lambda: list(self.busy), start_timeout_s=20.0, stop_wait_s=2.0, unload_wait_s=4.0,
                    poll_s=0.05, name=studio.COMFY_ENGINE)
        opts.update(kw)
        eng = ComfyUIEngine(CAT, **opts)
        self.addCleanup(self._stop, eng)
        return eng

    @staticmethod
    def _stop(eng):
        try:
            for r in list(eng._runs.values()):
                r.cancel.set()
            eng._stop_process()
        except Exception:
            pass

    def start(self, eng, env=None):
        old = {}
        for k, v in (env or {}).items():
            old[k] = os.environ.get(k)
            os.environ[k] = v
        try:
            t = eng.load(studio.COMFYUI)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertTrue(wait_for(lambda: eng.poll_load(t).state != "loading", 25.0), eng.poll_load(t))
        return eng.poll_load(t)

    def graph_op(self, out="stills/s0.png", **kw):
        op = {"op": "graph", "graph": "still", "out": out, "images": {"image_1": self.hero},
              "params": {"prompt": "calm", "seed": 3}, "expected_s": 20}
        op.update(kw)
        return op

    def run_graph(self, eng, run_id="1.1", wd="ads/c1", timeout=25.0, **op):
        h = eng.start(spec(studio.COMFYUI, run_id=run_id, workdir=wd, ops=[self.graph_op(**op)]))
        wait_for(lambda: eng.poll(h).status != "running", timeout)
        return eng.poll(h)


class R1ComfyProcess(ComfyBase):

    def test_a_hung_comfyui_process_still_counts_as_resident(self):
        # Foreman's ComfyUI is alive and holds its GPU memory but does not answer /system_stats in
        # time (busy, swapping, wedged). If residency is "answers HTTP" gpud believes it is gone,
        # never unloads it, and loads an mlx-serve media model beside it: the memory rule broken.
        eng = self.engine(exit_after_job=False, call_timeout_s=1.0)
        self.assertEqual(self.start(eng).state, "ready")
        pid = eng._pid
        self.assertEqual(eng.resident_models(), {studio.COMFYUI})
        os.kill(pid, signal.SIGSTOP)
        try:
            seen = eng.resident_models()
        finally:
            os.kill(pid, signal.SIGCONT)
        self.assertIn(studio.COMFYUI, seen,
                      "a live ComfyUI process Foreman started is resident whether or not it answers HTTP "
                      "(fail closed: else gpud loads an mlx-serve media model beside it)")

    def test_a_running_comfyui_rechecks_for_mlx_serve_media_before_each_job(self):
        # The preflight (no mlx-serve media model resident) runs only when Foreman spawns the server.
        # A server that is already up (exit_after_job off, or adopted after a gpud restart) runs the
        # next graph even after someone loaded a media model into mlx-serve.
        eng = self.engine(exit_after_job=False)
        self.assertEqual(self.start(eng).state, "ready")
        self.busy = ["someone-elses-image-model"]
        p = self.run_graph(eng)
        out = os.path.join(self.art, "ads", "c1", "stills", "s0.png")
        self.assertNotEqual(p.status, "done",
                            "a graph never runs while mlx-serve holds a media model (the owner's included)")
        self.assertFalse(os.path.exists(out))

    def test_a_pid_file_from_another_port_does_not_orphan_the_server_foreman_started(self):
        # gpud crashes with its ComfyUI running; the owner changes comfyui_options.port; the next
        # gpud finds the pid file, sees another port, deletes the file and forgets the process:
        # a ComfyUI holding tens of GB that nothing will ever stop.
        a = self.engine(exit_after_job=False)
        self.assertEqual(self.start(a).state, "ready")
        pid = a._pid
        b = self.engine(exit_after_job=False, port=free_port())
        tracked = b._pid == pid or (paths.read_json(self.pidfile) or {}).get("pid") == pid
        self.assertTrue(not alive(pid) or tracked,
                        "the ComfyUI Foreman started is stopped or still tracked, never forgotten while it runs")

    def test_a_main_py_process_foreman_did_not_start_is_never_signalled(self):
        # The pid file names a live process whose command mentions main.py but that is not the
        # session leader Foreman spawned (the pid was reused, or it is another app's main.py).
        # Foreman spawns ComfyUI with start_new_session, so killpg failing means "not ours"; the
        # os.kill fallback signals it anyway.
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "main.py"])
        self.addCleanup(lambda: (sleeper.poll() is None and sleeper.kill(), sleeper.wait()))
        os.makedirs(os.path.dirname(self.pidfile), exist_ok=True)
        paths.atomic_json(self.pidfile, {"pid": sleeper.pid, "port": self.port, "started_wall": time.time(),
                                         "comfy_dir": self.comfy_dir})
        eng = self.engine(exit_after_job=False, stop_wait_s=1.0, unload_wait_s=4.0, call_timeout_s=0.5)
        try:
            eng.unload(studio.COMFYUI)
        except Exception:
            pass
        eng.terminate()
        time.sleep(0.3)
        self.assertIsNone(sleeper.poll(), "only the ComfyUI process Foreman started is ever signalled")

    def test_gpud_shutdown_stops_the_comfyui_it_started(self):
        # Gpud.shutdown() calls close() on every engine. For ComfyUI that only interrupts runs: the
        # server Foreman started keeps its memory while no gpud enforces the memory rule.
        eng = self.engine()
        self.assertEqual(self.start(eng).state, "ready")
        pid = eng._pid
        eng.close()
        self.assertTrue(wait_for(lambda: not alive(pid), 8.0),
                        "a clean gpud shutdown leaves no ComfyUI of Foreman's running")

    def test_an_adopted_server_drops_the_prompts_the_dead_gpud_queued(self):
        # gpud dies while ComfyUI renders a take; the next gpud adopts the server and retries the
        # step. The dead gpud's prompt is still queued or running: two renders of one take, on a
        # GPU budgeted for one.
        a = self.engine(exit_after_job=False)
        self.assertEqual(self.start(a, env={"FAKE_COMFY_RUN_S": "30"}).state, "ready")
        h1 = a.start(spec(studio.COMFYUI, run_id="1.1", workdir="ads/c8", ops=[self.graph_op("stills/t.png")]))
        self.assertTrue(wait_for(lambda: a._runs[h1].prompt_id, 10.0))
        old = a._runs[h1].prompt_id
        b = self.engine(exit_after_job=False)
        self.assertEqual(b._pid, a._pid, "the restarted gpud adopted the server it started")
        t = b.load(studio.COMFYUI)
        self.assertTrue(wait_for(lambda: b.poll_load(t).state != "loading", 10.0))
        h2 = b.start(spec(studio.COMFYUI, run_id="1.2", workdir="ads/c8", ops=[self.graph_op("stills/t.png")]))
        self.assertTrue(wait_for(lambda: b._runs[h2].prompt_id, 10.0))
        _, q = json_call(b.base_url, "GET", "/queue", timeout=3.0)
        live = {str(item[1]) for key in ("queue_running", "queue_pending") for item in (q or {}).get(key) or []}
        self.assertNotIn(old, live, "the dead gpud's prompt is interrupted or removed before the retry renders")

    def test_a_comfyui_that_dies_mid_graph_is_an_infra_failure_and_a_new_one_starts(self):
        eng = self.engine(exit_after_job=False)
        self.assertEqual(self.start(eng, env={"FAKE_COMFY_RUN_S": "30"}).state, "ready")
        pid = eng._pid
        h = eng.start(spec(studio.COMFYUI, run_id="1.1", workdir="ads/c9", ops=[self.graph_op()]))
        self.assertTrue(wait_for(lambda: eng._runs[h].prompt_id, 10.0))
        os.kill(pid, signal.SIGKILL)
        self.assertTrue(wait_for(lambda: eng.poll(h).status != "running", 15.0))
        p = eng.poll(h)
        self.assertEqual(p.status, "failed")
        self.assertNotEqual(p.reason, "rejected", "a dead server is the machine's fault, not the graph's")
        self.assertEqual(eng.resident_models(), set())
        self.assertEqual(self.start(eng).state, "ready", "the next load starts a fresh server")
        self.assertNotEqual(eng._pid, pid)



class R1ComfyUnderGpud(ComfyBase):
    """The real ComfyUI adapter (on the stand-in process) under gpud, beside a simulated mlx-serve."""

    def test_without_a_blip_gpud_stops_its_comfyui_before_an_mlx_media_model(self):
        self.scenario(blip=False)

    def test_a_comfyui_health_blip_does_not_hand_foremans_server_to_someone_else(self):
        # One slow /system_stats answer makes resident_models() report nothing; gpud drops its
        # ownership, and when the server answers again it is "a catalog model gpud did not load":
        # pinned, never unloaded, and (solo_media) blocking every mlx-serve media load for good.
        self.scenario(blip=True)

    def scenario(self, blip: bool):
        from foreman.policy import ModelSpec, Policy
        from foreman.sim import FakeEngine, ScriptedBrain, SimHost, VirtualClock
        from foreman.supervisor import Supervisor
        cat = {"brain": ModelSpec("brain", "a", 20, 10, "brain", vision=True),
               "tts": ModelSpec("tts", "a", 5, 5, "tts"),
               studio.COMFYUI: ModelSpec(studio.COMFYUI, studio.COMFY_ENGINE, 60, 60, "comfy", solo_media=True)}
        clock = VirtualClock(0.0)
        a = FakeEngine("a", clock, SimHost(256.0), cat, brain=ScriptedBrain())
        comfy = ComfyUIEngine(cat, comfy_dir=self.comfy_dir, python=sys.executable, port=self.port,
                              graphs_dir=self.graphs, artifacts_dir=self.art, home=self.home, mem_probe=lambda: 200.0,
                              media_probe=lambda: [], start_timeout_s=20.0, stop_wait_s=2.0, unload_wait_s=4.0,
                              poll_s=0.05, call_timeout_s=1.0, exit_after_job=False, name=studio.COMFY_ENGINE)
        self.addCleanup(self._stop, comfy)
        sup = Supervisor(os.path.join(self.tmp, "g.db"), {"a": a, studio.COMFY_ENGINE: comfy}, clock,
                         Policy(text_ladder=("brain",), vision_brain="brain", co_resident_brain="brain"), catalog=cat)
        self.addCleanup(sup.close)
        sup.recover()

        def tick_until(pred, n):
            for _ in range(n):
                clock.advance(1.0)
                sup.tick()
                if pred():
                    return True
                time.sleep(0.02)
            return False
        cont = ({"kind": "review", "needs": "any"}, {"kind": "recover", "needs": "any"})
        j1 = sup.submit("job", studio.COMFYUI, {"goal": "g", "expected_s": 30.0, "workdir": "ads/s1",
                                                "ops": [self.graph_op()]}, *cont)
        self.assertTrue(tick_until(lambda: sup.store.job(j1).state == "succeeded", 300), sup.store.job(j1).outcome)
        pid = comfy._pid
        self.assertTrue(pid and alive(pid))
        if blip:
            os.kill(pid, signal.SIGSTOP)
            try:
                tick_until(lambda: False, 2)
            finally:
                os.kill(pid, signal.SIGCONT)
        tick_until(lambda: False, 2)
        j2 = sup.submit("job", "tts", {"goal": "g", "expected_s": 30.0}, *cont)
        done = tick_until(lambda: sup.store.job(j2).state in ("succeeded", "failed", "quarantined"), 120)
        blocked = [e["data"] for e in sup.store.events("media_blocked")] + \
            [e["data"] for e in sup.store.events("insufficient_memory")]
        self.assertTrue(done and sup.store.job(j2).state == "succeeded",
                        f"gpud still unloads its own ComfyUI before an mlx-serve media model loads: {blocked[-2:]}")
        self.assertFalse(alive(pid), "and the ComfyUI it started is stopped, not left beside the media model")

# ---------------------------------------------------------------- R2 partial files, malformed answers

class R2ComfyOutputs(ComfyBase):

    def test_a_truncated_view_download_is_never_a_finished_still(self):
        eng = self.engine(exit_after_job=False)
        self.assertEqual(self.start(eng, env={"RT_VIEW_MODE": "truncate"}).state, "ready")
        p = self.run_graph(eng)
        out = os.path.join(self.art, "ads", "c1", "stills", "s0.png")
        if p.status == "done" or mlx_ops.completed(out) is not None:
            with open(out, "rb") as f:
                data = f.read()
            try:
                media.decode_png(data)
            except Exception as e:  # noqa: BLE001
                self.fail(f"a half-downloaded PNG was recorded as a finished output ({len(data)} bytes: {e}); "
                          f"its sidecar makes every retry skip it")

    def test_a_non_image_answer_fails_the_still_and_is_not_kept_as_one(self):
        eng = self.engine(exit_after_job=False)
        self.assertEqual(self.start(eng, env={"RT_VIEW_MODE": "html"}).state, "ready")
        p = self.run_graph(eng)
        out = os.path.join(self.art, "ads", "c1", "stills", "s0.png")
        meta = paths.read_json(out + ".meta.json") or {}
        self.assertFalse(p.status == "done" and meta.get("ok") is not False and media.image_size(out) is None,
                         f"an HTML page saved as stills/s0.png passed as a good still: {meta}")


class R2Writes(TmpCase):

    def test_a_write_that_fails_halfway_leaves_no_temp_file_and_keeps_the_old_one(self):
        d = os.path.join(self.tmp, "w")
        os.makedirs(d)
        target = os.path.join(d, "state.json")
        with open(target, "w") as f:
            f.write("old")

        class Full:
            def __init__(self, path, mode):
                self.f = builtins.open(path, mode)

            def write(self, data):
                self.f.write(data[:3])
                raise OSError(errno.ENOSPC, "No space left on device")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.f.close()

        paths.open = lambda p, m="r", *a, **k: Full(p, m) if "w" in m else builtins.open(p, m, *a, **k)
        try:
            with self.assertRaises(OSError):
                paths.atomic_write(target, b"new content")
        finally:
            del paths.open
        with open(target) as f:
            self.assertEqual(f.read(), "old")
        self.assertEqual(sorted(os.listdir(d)), ["state.json"],
                         "a write that failed (disk full) leaves no partial temp file behind to fill the disk")

    def test_a_font_left_half_copied_by_a_kill_is_replaced(self):
        src = assembly.find_font()
        fonts = os.path.join(self.tmp, "final", "fonts")
        os.makedirs(fonts)
        ext = os.path.splitext(src)[1].lower() or ".ttf"
        with open(src, "rb") as f, open(os.path.join(fonts, "caption" + ext), "wb") as g:
            g.write(f.read(100))  # what a copy killed after 100 bytes leaves
        fi = assembly.install_font(src, fonts, "caption")
        self.assertEqual(os.path.getsize(fi["file"]), os.path.getsize(src),
                         "a retried assembly re-copies a truncated font instead of using it forever")


class R2Asr(unittest.TestCase):

    def test_a_giant_asr_word_list_is_bounded_before_the_alignment(self):
        # The alignment is an O(script words x ASR words) table in pure Python, inside gpud's
        # process (the dispatcher). A 256 MiB ASR answer holds millions of words: 100k already
        # takes seconds and ~400 MB. A 20 s track cannot hold more than a few hundred words.
        script = {"lines": [{"i": i, "role": "body", "text": " ".join(f"word{j}" for j in range(12))}
                            for i in range(5)]}
        asr = {"text": "x", "words": [{"word": f"x{i}", "start": i * 1e-4, "end": i * 1e-4 + 5e-5}
                                      for i in range(100000)]}
        orig = ads.align
        seen = []

        def guarded(a, b):
            seen.append((len(a), len(b)))
            if len(b) > 50 * max(1, len(a)):
                raise AssertionError(f"aligning {len(b)} ASR words against {len(a)} script words")
            return orig(a, b)
        ads.align = guarded
        try:
            try:
                words, _, _ = ads.word_times(script, asr, None, 20.0)
            except AssertionError as e:
                self.fail(f"a malformed ASR answer reached the quadratic alignment unbounded: {e}")
        finally:
            ads.align = orig
        self.assertEqual(len(words), 60)


@needs_ffmpeg
class R2Scratch(TmpCase):

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

    def test_a_retried_take_leaves_no_raw_frames_from_the_killed_attempt(self):
        t = self.eng.load(studio.LTX)
        self.assertTrue(wait_for(lambda: self.eng.poll_load(t).state == "ready"))
        wd = os.path.join(self.art, "ads", "m1")
        dead = os.path.join(wd, ".scratch", "7.1-0-dead")
        os.makedirs(dead)
        with open(os.path.join(dead, "frames.rgb"), "wb") as f:
            f.write(b"\0" * (1 << 20))  # what gpud killed mid-decode leaves (hundreds of MB for real)
        ops = [{"op": "video", "out": "shots/shot-1-s5.mp4", "prompt": "walks", "width": 64, "height": 128,
                "num_frames": 9, "seed": 5, "pipeline": "one_stage", "steps": 8}]
        h = self.eng.start(spec(studio.LTX, run_id="7.2", workdir="ads/m1", ops=ops))
        wait_for(lambda: self.eng.poll(h).status != "running", 20.0)
        self.assertEqual(self.eng.poll(h).status, "done", self.eng.poll(h))
        left = [os.path.join(r, n) for r, _, ns in os.walk(wd) for n in ns if n.endswith(".rgb")]
        self.assertEqual(left, [], "the retry of a step deletes the raw frames its killed attempt left")


# ---------------------------------------------------------------- R3 infrastructure versus content

def assembly_workdir(home: str, art: str, rel: str, seconds: list, card=None, disclosure="AI-generated",
                     words=None) -> str:
    """An ad work directory ready for ad_assemble: screen shots of a still capture (no takes needed)
    then the end card, a voice track and word times."""
    wd = os.path.join(art, rel)
    os.makedirs(os.path.join(wd, "voice"), exist_ok=True)
    png(os.path.join(home, "assets", "captures", "cap.png"), 120, 90, 200)
    bible_root = os.path.join(home, "characters", "mascot-r")
    png(os.path.join(bible_root, "stills", "hero.png"))
    data = {"id": "mascot-r", "name": "Mascot R", "kind": "mascot", "look": "a mascot character",
            "voice": {"engine": "kokoro", "voice": "af_heart"},
            "stills": [{"file": "stills/hero.png", "tags": ["hero"], "approved": True}]}
    paths.atomic_json(os.path.join(wd, "character.json"), {"root": bible_root, "data": data})
    card = card or {"price": "$9.50", "url": "example.com", "cta": "Get it done today"}
    paths.atomic_json(os.path.join(wd, "ad.json"), {"id": "ad-r", "template_id": "ads.service.v1", "end_card": card,
                                                    "disclosure": disclosure, "ai_label": True, "seed": 1,
                                                    "picks": {"talk_seeds": 1}})
    shots, t = [], 0.0
    for k, s in enumerate(seconds):
        shots.append({"k": k, "kind": "screen", "lines": [k], "visual": "", "still": None,
                      "capture": "captures/cap.png", "start": round(t, 3), "end": round(t + s, 3),
                      "seconds": round(s, 3)})
        t += s
    card_at = round(t, 3)
    shots.append({"k": len(seconds), "kind": "endcard", "lines": [], "visual": "", "still": None, "capture": None,
                  "start": card_at, "end": round(card_at + 3.0, 3), "seconds": 3.0})
    total = round(card_at + 3.0, 3)
    wav(os.path.join(wd, "voice", "track.wav"), card_at, rate=24000)
    words = words or [{"text": w, "line": i, "start": round(0.2 + 2 * i, 3), "end": round(0.9 + 2 * i, 3)}
                      for i, w in enumerate(["Hello", "there", "friend.", "Bye."][:len(seconds)])]
    paths.atomic_json(os.path.join(wd, "timing.json"), {"words": words, "source": "asr", "spans": [],
                                                        "track_s": card_at, "shots": shots, "total_s": total,
                                                        "fps": 24, "endcard_start": card_at})
    return wd


class LocalRunner(TmpCase):
    def setUp(self):
        super().setUp()
        self.home = make_home(self.tmp)
        self.art = os.path.join(self.home, "artifacts")
        os.makedirs(self.art, exist_ok=True)

    def run_local(self, pipe, kind, payload, timeout=120.0):
        eng = LocalEngine(CAT, pipe.handlers(), artifacts_dir=self.art, name=studio.LOCAL_ENGINE)
        self.addCleanup(eng.close)
        eng.load(studio.LOCAL)
        h = eng.start(spec(studio.LOCAL, kind=kind, run_id="1.1", **payload))
        self.assertTrue(wait_for(lambda: eng.poll(h).status != "running", timeout), eng.poll(h))
        return eng.poll(h)


class R3Classification(LocalRunner):

    def test_a_missing_font_is_the_machines_fault_not_the_ads(self):
        # FontError becomes ContentFailure, gpud's "rejected": a content failure that uses the
        # template's weekly cap and review budget, for a font file missing on the host.
        assembly_workdir(self.home, self.art, "ads/f1", [2.0, 2.0])
        pipe = ad_pipeline.AdPipeline(ads_config(fonts=["/nonexistent/fonts/x.ttf"]), home=self.home,
                                      artifacts_dir=self.art, ffmpeg=FFMPEG or "ffmpeg-not-needed")
        old = assembly.FONT_CANDIDATES, assembly.FONT_GLOBS
        assembly.FONT_CANDIDATES, assembly.FONT_GLOBS = (), ()
        try:
            p = self.run_local(pipe, "ad_assemble", {"workdir": "ads/f1", "stage": "assemble"})
        finally:
            assembly.FONT_CANDIDATES, assembly.FONT_GLOBS = old
        self.assertEqual(p.status, "failed")
        self.assertIn("font", (p.error or "").lower())
        self.assertNotEqual(p.reason, "rejected",
                            "no font on the host is an infrastructure failure: it must not use the owner's caps")

    def test_without_ffmpeg_the_ad_fails_at_its_first_step_before_any_gpu_work(self):
        # With no ffmpeg the facts step succeeds, and voice, ASR, stills and takes all render
        # (GPU minutes to hours) before the cut fails.
        t = parse_template(template_data())
        req = ads.root_request(t, studio.LOCAL, t.consumers[0], "an angle on filing a form", {"seconds": 20, "shots": 6},
                               "0123456789abcdef")
        pipe = ad_pipeline.AdPipeline(ads_config(), home=self.home, artifacts_dir=self.art, ffmpeg=None,
                                      commercial_models=studio.COMMERCIAL_OK)
        p = self.run_local(pipe, "ad_facts", req["payload"])
        self.assertEqual(p.status, "failed", "an ad that can never be assembled stops before it renders anything")
        if FFMPEG:  # control: the same step with ffmpeg present goes through
            ok = ad_pipeline.AdPipeline(ads_config(), home=self.home, artifacts_dir=self.art, ffmpeg=FFMPEG,
                                        commercial_models=studio.COMMERCIAL_OK)
            self.assertEqual(self.run_local(ok, "ad_facts", req["payload"]).status, "done")

    def test_a_rate_limited_facts_url_is_not_a_content_failure(self):
        class Busy(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = b"Too Many Requests"
                self.send_response(429)
                self.send_header("Retry-After", "30")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Busy)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        pipe = ad_pipeline.AdPipeline(ads_config(), home=self.home, artifacts_dir=self.art, ffmpeg=FFMPEG)
        url = f"http://127.0.0.1:{srv.server_address[1]}/pricing"
        with self.assertRaises(Exception) as cm:
            pipe.fetch_facts({"facts": {"url": url}})
        self.assertNotIsInstance(cm.exception, ContentFailure,
                                 "HTTP 429 is the site asking to come back later: retry it as infrastructure")


# ---------------------------------------------------------------- R4 kills

class R4Children(TmpCase):

    def test_a_killed_gpud_takes_its_ffmpeg_children_with_it(self):
        # launchd stops (or reaps) gpud by signalling its process group. LocalContext.run starts
        # ffmpeg with start_new_session, so a killed gpud's ffmpeg keeps running and keeps writing
        # into the ad's work directory while the next gpud retries the same step.
        pidfile = os.path.join(self.tmp, "child.pid")
        helper = os.path.join(self.tmp, "gpud_stand_in.py")
        with open(helper, "w") as f:
            f.write(f"""import sys, time
sys.path.insert(0, {REPO!r})
from foreman import studio
from foreman.adapters.local import LocalEngine
from foreman.engine import RunSpec
child = "import os, time; open({pidfile!r}, 'w').write(str(os.getpid())); time.sleep(60)"

def handler(spec, ctx):
    ctx.run([sys.executable, "-c", child], timeout=120)
    return {{"path": "/x", "exists": True}}

eng = LocalEngine(studio.catalog(False, ads=True), {{"ad_assemble": handler}}, artifacts_dir={self.tmp!r},
                  name=studio.LOCAL_ENGINE)
eng.load(studio.LOCAL)
eng.start(RunSpec("1.1", 1, "ad_assemble", studio.LOCAL, {{"goal": "g", "expected_s": 30.0}}, 30.0))
time.sleep(120)
""")
        gpud = subprocess.Popen([sys.executable, helper], start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: (gpud.poll() is None and os.killpg(gpud.pid, signal.SIGKILL), gpud.wait()))
        def read_pid():
            try:
                with open(pidfile) as f:
                    return f.read().strip()
            except OSError:
                return ""
        self.assertTrue(wait_for(read_pid, 15.0))
        child = int(read_pid())
        self.addCleanup(lambda: alive(child) and os.kill(child, signal.SIGKILL))
        os.killpg(gpud.pid, signal.SIGKILL)  # launchd's group kill
        gpud.wait()
        gone = wait_for(lambda: not alive(child), 3.0)
        self.assertTrue(gone, "the ffmpeg of a killed gpud is gone too, not racing the retried step on the same files")


class R4Dispatcher(TmpCase):
    """A brain job (the dispatcher) that crashed after writing its state is run again: it must
    submit the same request and not spend another retry seed."""

    def setUp(self):
        super().setUp()
        self.home = make_home(self.tmp)
        self.art = os.path.join(self.home, "artifacts")
        self.pipe = ad_pipeline.AdPipeline(ads_config(), home=self.home, artifacts_dir=self.art, ffmpeg=FFMPEG)
        self.bible = characters.load_bible(self.home, "mascot-a")
        t = parse_template(template_data())
        self.ad = dict(t.ad, id="ad-d", template_id=t.id, seconds=20.0, shots=6, seed=5, brief="x",
                       length={"min": 15.0, "max": 30.0}, shot_range={"min": 4, "max": 6}, commercial=True)
        self.ad["picks"] = self.pipe.picks(self.ad, self.bible)
        self.rel = "ads/d1"
        self.wd = self.pipe.workdir(self.rel)
        shots = [{"k": 0, "kind": "talk", "lines": [0], "visual": "an office", "still": None, "capture": None,
                  "start": 0.0, "end": 2.0, "seconds": 2.0, "frames": 49},
                 {"k": 1, "kind": "endcard", "lines": [], "visual": "", "still": None, "capture": None,
                  "start": 2.0, "end": 5.0, "seconds": 3.0}]
        paths.atomic_json(os.path.join(self.wd, "timing.json"), {"shots": shots, "words": [], "total_s": 5.0,
                                                                 "endcard_start": 2.0})

    def mark(self, rel, ok):
        full = os.path.join(self.wd, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(b"x")
        paths.atomic_json(full + ".meta.json", {"complete": True, "ok": ok})

    def test_a_rerun_after_a_crash_submits_the_same_still_and_take_retries(self):
        self.mark("stills/shot-0-a.png", False)
        first = self.pipe.step_stills(self.wd, self.rel, self.ad, self.bible, None)
        again = self.pipe.step_stills(self.wd, self.rel, self.ad, self.bible, None)
        self.assertEqual(first[0], "job")
        self.assertEqual([o["out"] for o in first[2]["payload"]["ops"]], [o["out"] for o in again[2]["payload"]["ops"]])
        self.assertEqual(self.pipe.state(self.wd)["still_retries"], {"shot-0-a": 1}, "one retry seed, not two")
        self.mark("stills/shot-0-a-r1.png", True)
        for seed in self.pipe.seeds_for(self.ad, {"k": 0, "kind": "talk"}):
            self.mark(f"shots/shot-0-s{seed}.mp4", False)
        first = self.pipe.step_shots(self.wd, self.rel, self.ad, self.bible, None)
        again = self.pipe.step_shots(self.wd, self.rel, self.ad, self.bible, None)
        self.assertEqual([o["out"] for o in first[2]["payload"]["ops"]], [o["out"] for o in again[2]["payload"]["ops"]])
        self.assertEqual(len(self.pipe.state(self.wd)["extra_seeds"]["0"]), 1, "one extra seed, not two")
        self.assertEqual(first[2]["payload"]["ops"][0]["first_frame_path"], "stills/shot-0-a-r1.png",
                         "the take starts from the still that passed")


@needs_ffmpeg
class R4Restarts(AdHarness):
    """gpud killed and restarted on the same database and work directories."""

    def restart(self):
        self.sup.close()
        self.mlx.close()
        self.local.close()
        self.pipeline = ad_pipeline.AdPipeline(self.config(), home=self.home, artifacts_dir=self.art, ffmpeg=FFMPEG,
                                               catalog=self.catalog, max_depth=64,
                                               commercial_models=studio.COMMERCIAL_OK, licenses=studio.LICENSES)
        from foreman.brain_client import BrainClient
        self.brain = BrainClient(self.fake.url, catalog=self.catalog, ads=self.pipeline)
        self.mlx = MlxServeEngine(self.fake.url, catalog=self.catalog, model_dir="/fake/models",
                                  artifacts_dir=self.art, brain=self.brain, mem_probe=lambda: 200.0, read_poll_s=0.02,
                                  unload_wait_s=0.5, connect_timeout_s=2.0, call_timeout_s=2.0,
                                  forbidden=studio.FORBIDDEN, ffmpeg=FFMPEG, input_roots=[self.home])
        self.local = LocalEngine(self.catalog, self.pipeline.handlers(), artifacts_dir=self.art,
                                 name=studio.LOCAL_ENGINE)
        self.comfy = ComfyUIEngine(self.catalog, comfy_dir=os.path.join(self.tmp, "no-comfy"), port=1,
                                   artifacts_dir=self.art, home=self.home, name=studio.COMFY_ENGINE)
        self.planner = Planner(PlannerConfig(utc=True, min_disk_free_gb=0.0, artifacts_cap_gb=1e6), home=self.home,
                               artifacts_dir=self.art, commercial_models=studio.COMMERCIAL_OK,
                               disk_probe=lambda: (1000.0, 0.0))
        from foreman.supervisor import Supervisor
        self.sup = Supervisor(os.path.join(self.tmp, "foreman.db"),
                              {studio.ENGINE: self.mlx, studio.LOCAL_ENGINE: self.local, studio.COMFY_ENGINE: self.comfy},
                              self.clock, studio.policy(), catalog=self.catalog, pager=self.pages.append,
                              planner=self.planner)
        self.sup.recover()

    def quiet_engines(self):
        """No engine thread is working (an in-process engine cannot be killed with gpud)."""
        return not (self.mlx.live_runs() or self.local.live_runs())

    def media_counts(self):
        return {"video": len(self.fake.video_requests), "image": len(self.fake.bodies("/v1/images/generations")),
                "speech": len(self.fake.bodies("/v1/audio/speech")),
                "asr": len([1 for m, p, b in self.fake.requests if p == "/v1/audio/transcriptions"]),
                "music": len(self.fake.bodies("/v1/audio/music-generations"))}

    def test_gpud_restarted_between_every_pair_of_steps_renders_each_thing_once(self):
        # gpud is killed at its own crash points between steps: right after a job's end is
        # recorded (before its continuation fires), right after the continuation fires (before it
        # is dispatched), and right after a job is marked starting (before the engine is asked).
        from foreman.sim import SimulatedCrash
        points = ("after_terminal_recorded", "after_continuation_fired", "after_starting_recorded")
        crashed = set()

        def hook(point):
            if point not in points or not self.quiet_engines():
                return
            key = (point, tuple((j.id, j.state, j.attempt) for j in self.sup.store.jobs()))
            if key not in crashed:
                crashed.add(key)
                raise SimulatedCrash(point)
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        res = self.enqueue()
        self.sup.crash_hook = hook
        restarts = []
        end = time.monotonic() + 300
        while time.monotonic() < end and self.item(res["item"])["state"] == "queued":
            self.clock.t += 1.0
            try:
                self.sup.tick()
            except SimulatedCrash as e:
                restarts.append(str(e))
                self.restart()
                self.sup.crash_hook = hook
            time.sleep(0.01)
        it = self.item(res["item"])
        self.assertEqual(it["state"], "review", (it, restarts, self.trail()))
        self.assertGreaterEqual(len(restarts), 15, ("gpud was killed between the steps", restarts))
        self.assertEqual(self.media_counts(), {"video": 6, "image": 6, "speech": 6, "asr": 1, "music": 1},
                         "every request was made exactly once across the restarts")
        reviews = [j for j in self.jobs("ad_review")]
        self.assertEqual(len(reviews), 1)
        self.assertEqual(it["artifacts"][0], os.path.join(res["workdir"], "final", "ad.mp4"))
        self.assertEqual(len([i for i in ledger.items(self.sup.store) if i["template_id"] == "ads.service.v1"]), 1)
        self.assertTrue(self.run_until(lambda: studio.BRAIN_PRIMARY in self.mlx.resident_models(), timeout=10),
                        "the brain is home")
        self.assertIn(".revision-0123abcd", self.fake.ready, "the owner's model is never unloaded")

    def test_gpud_killed_mid_take_re_renders_only_that_take(self):
        self.fake.chat_replies.extend([json.dumps(SCRIPT), REVIEW_OK])
        self.fake.progress_delay = 0.25
        res = self.enqueue()
        killed = False
        end = time.monotonic() + 240
        while time.monotonic() < end and self.item(res["item"])["state"] == "queued":
            self.clock.t += 1.0
            self.sup.tick()
            running = [j for j in self.sup.store.jobs() if j.kind == "ad_shots" and j.state == "running"]
            if not killed and running:
                run = self.mlx._runs.get(running[0].handle)
                if run is not None and wait_for(lambda: run.status != "running" or run.progress > 0.1, 5.0) \
                        and run.status == "running":
                    self.restart()  # the take is streaming its progress events
                    killed = True
            time.sleep(0.01)
        it = self.item(res["item"])
        self.assertTrue(killed)
        self.assertEqual(it["state"], "review", (it, self.trail()))
        self.assertEqual(len(self.fake.video_requests), 7, "six takes, and the interrupted one once more")
        lost = [j for j in self.jobs("ad_shots") if j.state != "succeeded"]
        self.assertEqual(len(lost), 1, self.trail())
        self.assertIn(lost[0].outcome.get("reason"), ledger.INFRA_REASONS)
        wd = res["workdir"]
        left = [n for r, _, ns in os.walk(wd) for n in ns if n.endswith(".rgb")]
        self.assertEqual(left, [], "no raw frames left behind")


# ---------------------------------------------------------------- R5 durations and limits

class R5Durations(unittest.TestCase):

    def test_a_sixty_second_template_is_refused_or_can_be_made(self):
        # Templates may ask for up to 60 s, but at most 6 shots (5 before the end card) of at most
        # max_shot_s (+0.75 s) each: the longest cut is ~47 s, so a 55-60 s template validates
        # every proposal and then fails every ad at timing, after TTS and ASR, as a content failure.
        data = template_data(params={"seconds": {"min": 55, "max": 60, "default": 60},
                                     "shots": {"min": 4, "max": 6, "default": 6}})
        try:
            t = parse_template(data)
        except TemplateError:
            return  # refused when it is loaded: the owner hears about it once
        cfg = AdsConfig()
        ad = dict(t.ad, length={"min": 55.0, "max": 60.0}, shot_range={"min": 4, "max": 6})
        kinds = ["talk", "broll", "screen", "broll", "talk"]
        script = {"lines": [{"i": i, "role": "body", "text": "word " * int(cfg.max_shot_s * cfg.words_per_s)}
                            for i in range(len(kinds))],
                  "shots": [{"k": k, "kind": kind, "lines": [k], "visual": "x", "still": None, "capture": None}
                            for k, kind in enumerate(kinds)] + [{"k": 5, "kind": "endcard", "lines": []}]}
        line_s = cfg.max_shot_s + 0.75 - 0.55  # every shot as long as plan_shots lets it run (8.75 s)
        spans, pos = [], cfg.lead_s
        for _ in kinds:
            spans.append([pos, pos + line_s])
            pos += line_s + cfg.line_gap_s
        _, total, problems = plan_shots(script, spans, spans[-1][1], cfg, ad)
        self.assertEqual(problems, [], f"the longest plan the rules allow runs {total:.1f} s: a template asking for "
                                       f"55-60 s must be refused at load")

    def test_ltx_limits_hold_at_the_edges(self):
        self.assertEqual(mlx_ops.max_frames(704, 1280), 297)
        self.assertEqual(mlx_ops.video_problems({"pipeline": "two_stage", "width": 704, "height": 1280,
                                                 "num_frames": 297}), [])
        self.assertTrue(mlx_ops.video_problems({"pipeline": "two_stage", "width": 704, "height": 1280,
                                                "num_frames": 305}))
        for s in (0.01, 1.2, 2.0, 4.99, 5.0, 5.01, 7.0, 8.0, 8.75, 12.375):
            n = ads.frames_for(s)
            self.assertEqual((n - 1) % 8, 0, s)
            self.assertGreaterEqual(n / 24.0, s, f"{n} frames cover {s} s")
        self.assertEqual(ads.frames_for(5.0), 121)
        cfg = AdsConfig(video_width=1088, video_height=1920)
        cap = mlx_ops.max_frames(1088, 1920)
        script = {"lines": [{"i": 0, "text": "a"}, {"i": 1, "text": "b"}],
                  "shots": [{"k": 0, "kind": "talk", "lines": [0]}, {"k": 1, "kind": "talk", "lines": [1]},
                            {"k": 2, "kind": "endcard", "lines": []}]}
        _, _, problems = plan_shots(script, [[0.15, 7.5], [7.8, 14.0]], 14.0, cfg, {"length": {"min": 15, "max": 30}})
        self.assertTrue(any(f"{cap} frames" in p for p in problems), problems)

    def test_frames_for_asks_for_the_shortest_8k_plus_1_clip_that_covers_the_shot(self):
        # frames_for rounds 24*s up to a multiple of 8 and then adds 1: for 24*s in (8m, 8m+1] it
        # asks for 8m+9 frames although 8m+1 already cover the shot. Every such take renders 8
        # frames (1/3 s) for nothing, and at a larger video size a shot that fits LTX's 768 MiB is
        # refused: 121 frames (5.04 s) is the cap at 1088x1920, and a 5.02 s shot needs no more.
        for s in (5.01, 5.02, 1.36, 7.02):
            n = ads.frames_for(s)
            shortest = next(k for k in range(9, 1000, 8) if k / 24.0 >= s)
            self.assertEqual(n, shortest, f"{s} s: frames_for asks for {n} frames, {shortest} cover it")
        cfg = AdsConfig(video_width=1088, video_height=1920)
        self.assertEqual(mlx_ops.max_frames(1088, 1920), 121)
        script = {"lines": [{"i": 0, "text": "a"}, {"i": 1, "text": "b"}],
                  "shots": [{"k": 0, "kind": "talk", "lines": [0]}, {"k": 1, "kind": "talk", "lines": [1]},
                            {"k": 2, "kind": "endcard", "lines": []}]}
        spans = [[0.15, 4.87], [5.17, 9.0]]  # the first cut lands at 5.02 s
        shots, _, problems = plan_shots(script, spans, 9.0, cfg, {"length": {"min": 10, "max": 30}})
        self.assertAlmostEqual(shots[0]["seconds"], 5.02)
        self.assertEqual([p for p in problems if "frames" in p], [], "a 5.02 s shot fits in 121 frames")


@needs_ffmpeg
class R5Cuts(LocalRunner):

    def frames(self, path: str) -> int:
        res = subprocess.run([FFMPEG, "-loglevel", "error", "-i", path, "-map", "0:v:0", "-vf", "scale=8:8",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
        return len(res.stdout) // 64

    def test_cut_points_do_not_drift_from_the_plan(self):
        # Each segment is round(seconds * 24) frames on its own, so the rounding error adds up
        # over the cut: the video drifts against the voice track and the captions (both on the
        # plan's absolute times), up to ~0.1 s by the call to action's lip-synced shot.
        seconds = [2.02, 2.02, 2.02, 2.02]
        wd = assembly_workdir(self.home, self.art, "ads/drift", seconds)
        pipe = ad_pipeline.AdPipeline(ads_config(), home=self.home, artifacts_dir=self.art, ffmpeg=FFMPEG)
        p = self.run_local(pipe, "ad_assemble", {"workdir": "ads/drift", "stage": "assemble"}, timeout=180)
        self.assertEqual(p.status, "done", p)
        timing = paths.read_json(os.path.join(wd, "timing.json"))
        planned = int(round(timing["total_s"] * 24))
        got = self.frames(os.path.join(wd, "final", "ad.mp4"))
        self.assertLessEqual(abs(got - planned), 1,
                             f"the cut has {got} frames, the plan {planned}: shot boundaries drifted "
                             f"{(planned - got) / 24.0:.3f} s from the voice and the captions")


@needs_ffmpeg
class R5Text(TmpCase):

    def setUp(self):
        super().setUp()
        self.ff = assembly.Ffmpeg(FFMPEG, self.tmp)
        self.font = assembly.install_font(assembly.find_font(), os.path.join(self.tmp, "fonts"), "caption")
        self.fontfile = "fonts/" + os.path.basename(self.font["file"])

    def card(self, text: str, name: str) -> int:
        with open(os.path.join(self.tmp, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)
        assembly.seg_endcard(self.ff, name + ".mp4", 1.0, 270, 480, self.fontfile, [(name + ".txt", 0.06, 0.5)])
        return bright(probe_frame(FFMPEG, os.path.join(self.tmp, name + ".mp4"), 0.5, 270, 480))

    def test_an_end_card_line_with_a_percent_sign_is_drawn(self):
        plain = self.card("Save 20 today", "plain")
        pct = self.card("Save 20% today", "pct")
        self.assertGreater(plain, 50)
        self.assertGreater(pct, plain // 2, "drawtext expands % in its text file: 'Save 20% today' draws nothing")

    def test_an_ordinary_url_fits_on_the_end_card(self):
        # The end card draws the URL at 4.5% of the height (86 px at 1920) with drawtext, which
        # never wraps or shrinks: a 25-character address runs off both edges of a 1080 px frame
        # (the template allows 80). The address is the point of the end card.
        W, H = 1080, 1920
        url = "www.my-online-service.com"
        with open(os.path.join(self.tmp, "url.txt"), "w", encoding="utf-8") as f:
            f.write(url)
        assembly.seg_endcard(self.ff, "url.mp4", 1.0, W, H, self.fontfile, [("url.txt", 0.045, 0.52)])
        frame = probe_frame(FFMPEG, os.path.join(self.tmp, "url.mp4"), 0.5, W, H)
        rows = range(int(0.45 * H), int(0.6 * H))
        cols = [x for x in range(W) if any(frame[y * W + x] > 200 for y in rows)]
        self.assertTrue(cols)
        self.assertGreaterEqual(min(cols), int(0.03 * W), f"{url!r} is cut off at the left edge")
        self.assertLessEqual(max(cols), int(0.97 * W), f"{url!r} is cut off at the right edge")

    def label(self, text: str, name: str) -> int:
        if not os.path.exists(os.path.join(self.tmp, "v.mp4")):
            self.ff.run(["-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=0x202020:s=270x480:r=24:d=1",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "v.mp4"])
            self.ff.run(["-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=1:sample_rate=48000",
                         "-ac", "2", "a.wav"])
        with open(os.path.join(self.tmp, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)
        assembly.final_encode(self.ff, "v.mp4", "a.wav", name + ".mp4", 1.0, 40e6, label_file=name + ".txt",
                              label_font=self.fontfile, width=270, height=480)
        frame = probe_frame(FFMPEG, os.path.join(self.tmp, name + ".mp4"), 0.5, 270, 480)
        return bright(frame, 0, 270 * int(0.25 * 480))

    def test_the_ai_label_with_a_percent_sign_is_burned_in(self):
        plain = self.label("AI-generated, all synthetic", "plain")
        pct = self.label("AI-generated, 100% synthetic", "pct")
        self.assertGreater(plain, 20)
        self.assertGreater(pct, plain // 2, "the disclosure label vanishes (and QA still says it is burned in)")


# ---------------------------------------------------------------- R6 disk

class R6Disk(TmpCase):

    def test_the_disk_guard_counts_an_ad_for_its_whole_life(self):
        # An ad's root job (its facts) runs for seconds; its takes and cut are written for hours
        # after. pending_gb only counts an item while its root job is queued or running.
        store = Store(os.path.join(self.tmp, "d.db"))
        self.addCleanup(store.close)
        root, _ = store.insert_job(kind="ad_facts", requires=studio.LOCAL, state="succeeded",
                                   payload={"workdir": "ads/x", "plan_bytes": 2e9, "expected_s": 30, "goal": "g"},
                                   max_attempts=3, created_at=0.0)
        store.update_job(root, root_id=root)
        store.insert_job(kind="ad_shots", requires=studio.LTX, state="running", root_id=root, parent_id=root, depth=2,
                         payload={"workdir": "ads/x", "expected_s": 400, "goal": "g", "ops": []}, max_attempts=3,
                         created_at=1.0)
        ledger.insert_item(store, plan_job_id=None, template_id="ads.service.v1", consumer="c", brief="b",
                           params={}, content_hash="h1", job_id=root, state="queued", flags={}, value_usd=1.0,
                           p_ship=0.5, urgency=1.0, value_adj=1.0, ev=None, priority=0, est_gpu_s=1.0,
                           review_minutes=1.0, rationale="r", created_at=0.0, created_wall=0.0,
                           created_day="2026-11-01")
        self.assertGreater(Planner.pending_gb(store), 0.5,
                           "an ad still rendering takes keeps its planned bytes reserved against the disk guard")


class R6Enqueue(AdHarness):

    def test_ads_enqueue_counts_the_ads_own_size_against_the_disk_floor(self):
        # The planner's path adds the ad's estimate to the reading; `ads enqueue` checks the reading
        # alone, so it admits an ad of ~1.6 GB with 0.5 GB above the floor.
        t = load_templates(os.path.join(self.home, "templates"))[0]["ads.service.v1"]
        free = shutil.disk_usage(self.home).free / 1e9
        cfg = PlannerConfig(min_disk_free_gb=max(0.0, free - 0.5), artifacts_cap_gb=1e6)
        with self.assertRaises(AdmissionError):
            ad_pipeline.enqueue(self.sup, t, "Show how easy it is to file a form", None, {}, self.art, self.home,
                                cfg, force=False)


if __name__ == "__main__":
    unittest.main()
