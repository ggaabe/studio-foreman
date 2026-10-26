"""The mlx-serve adapter, the brain client, gpud's process wrapper and the stdlib media analysis,
run against a local fake mlx-serve (tests/fake_mlx_serve.py) and synthetic media.

  M1  media.py: WAV statistics, raw rgb8 frame statistics, PNG encoding
  M2  the SSE reader and the streaming flat-JSON parser (huge complete events, chunk splits)
  M3  MlxServeEngine against the fake server: residency mapping, loads, unloads, runs, faults
  M4  BrainClient: strict decisions, safe fallbacks, prompts with sampled frames
  M5  the supervisor features this deployment needs (on the simulator): foreign models pinned,
      brain evicted for video, engine model-slot caps, the brain asking for a human
  M6  gpud end to end on the fake server: speech + review, video with the brain unloaded and
      brought back, and the dead-man lease firing on a run stalled by fault injection
  M7  CLI, daemon wrapper and the LaunchAgent template
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import plistlib
import random
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import wave

from foreman import media, studio, validators
from foreman.adapters import mlx_serve
from foreman.adapters.mlx_serve import MlxServeEngine, UnloadPending, video_frames_for
from foreman.adapters.sse import Base64FileSink, SseError, SseReader
from foreman.brain_client import BrainClient, decide, parse_json_object
from foreman.engine import RunSpec
from foreman.policy import CATALOG, ModelSpec, Policy
from foreman.sim import ScriptedBrain, Sim, render_request
from foreman.supervisor import Supervisor

try:
    from fake_mlx_serve import FakeMlxServe, tone_wav, video_frames
except ImportError:  # pragma: no cover
    from tests.fake_mlx_serve import FakeMlxServe, tone_wav, video_frames

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOKORO, ACE, LTX = studio.KOKORO, studio.ACE_STEP, studio.LTX
BRAIN, FALLBACK = studio.BRAIN_PRIMARY, studio.BRAIN_FALLBACK
FOREIGN = ".revision-0123abcd"  # a model someone else loaded (snapshot-dir name, as /api/ps shows it)


def wait_for(pred, timeout: float = 5.0, step: float = 0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


def write_wav(path, samples, rate=8000, width=2, channels=1):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="foreman-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------- M1 media

class M1Media(TmpCase):

    def test_wav_stats_duration_level_and_silence(self):
        path = os.path.join(self.tmp, "a.wav")
        with open(path, "wb") as f:
            f.write(tone_wav(2.0, rate=8000, silent_tail=0.5))
        st = media.wav_stats(path)
        self.assertAlmostEqual(st["duration_s"], 2.0, places=3)
        self.assertEqual((st["sample_rate"], st["channels"], st["bits"]), (8000, 1, 16))
        self.assertAlmostEqual(st["silence_ratio"], 0.25, delta=0.03)
        self.assertGreater(st["rms_dbfs"], -20)
        silent = os.path.join(self.tmp, "s.wav")
        write_wav(silent, bytes(16000))
        self.assertEqual(media.wav_stats(silent)["silence_ratio"], 1.0)

    def test_speech_without_a_requested_length_still_needs_a_real_duration(self):
        art = {"path": "/x/output.wav", "exists": True, "duration_s": 0.0, "silence_ratio": 0.0}
        res = validators.check("tts", {"input": "hi"}, art)
        self.assertFalse(validators.passed(res))
        self.assertIn("duration", validators.failures(res))
        self.assertTrue(validators.passed(validators.check("tts", {"input": "hi"}, dict(art, duration_s=1.2))))
        # simulated artifacts without a measurement are judged as before
        self.assertTrue(validators.passed(validators.check("tts", {}, {"path": "/x", "exists": True,
                                                                       "silence_ratio": 0.0})))

    def test_wav_stats_other_sample_formats(self):
        import struct
        loud = [int(20000 * ((-1) ** (i // 20))) for i in range(8000)]
        p8 = os.path.join(self.tmp, "u8.wav")
        write_wav(p8, bytes((v // 256 + 128) for v in loud), width=1)
        p24 = os.path.join(self.tmp, "s24.wav")
        write_wav(p24, b"".join(struct.pack("<i", v << 8)[:3] for v in loud), width=3)
        pst = os.path.join(self.tmp, "st.wav")  # stereo: left silent, right loud
        write_wav(pst, b"".join(struct.pack("<hh", 0, v) for v in loud), channels=2)
        for p in (p8, p24, pst):
            st = media.wav_stats(p)
            self.assertAlmostEqual(st["duration_s"], 1.0, places=2, msg=p)
            self.assertLess(st["silence_ratio"], 0.1, p)
        fl = os.path.join(self.tmp, "f32.wav")  # WAVE_FORMAT_IEEE_FLOAT, which `wave` cannot read
        data = struct.pack("<" + "f" * 8000, *[v / 32768.0 for v in loud])
        fmt = struct.pack("<HHIIHH", 3, 1, 8000, 32000, 4, 32)
        with open(fl, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE"
                    + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data)
        st = media.wav_stats(fl)
        self.assertAlmostEqual(st["duration_s"], 1.0, places=2)
        self.assertLess(st["silence_ratio"], 0.1)
        with open(os.path.join(self.tmp, "junk.wav"), "wb") as f:
            f.write(b"not a wav at all")
        with self.assertRaises(media.MediaError):
            media.wav_stats(os.path.join(self.tmp, "junk.wav"))
        with self.assertRaises(media.MediaError):
            media.wav_stats(os.path.join(self.tmp, "missing.wav"))

    def test_slow_motion_is_not_frozen_but_repeated_frames_are(self):
        # Live Studio, run 22.2: LTX rendered "a hot air balloon rising over a misty valley" with
        # consecutive frames differing by 0.15-0.44 (mean |delta|, 0..255) and frame 0 vs 120 by
        # 4.9. The first threshold (0.5) called that 100 % frozen and the brain re-rendered a
        # good clip. Model it: ~2 in 10 bytes move by 1 each frame, so the mean delta is ~0.2.
        w, h, n = 64, 64, 25
        base = bytes((x * 7 + y * 3) % 200 + 20 for y in range(h) for x in range(w) for _ in range(3))
        slow = os.path.join(self.tmp, "slow.rgb")
        with open(slow, "wb") as f:
            for i in range(n):
                f.write(bytes(min(255, b + (i // 5) + (1 if (j + i) % 10 == 0 else 0)) for j, b in enumerate(base)))
        st = media.video_stats(slow, w, h, n, 24)
        self.assertGreater(st["motion_mean"], 0.1)
        self.assertLess(st["motion_mean"], 0.5)
        self.assertGreater(st["span_diff"], st["motion_mean"])
        self.assertEqual(st["frozen_frame_ratio"], 0.0)
        stuck = os.path.join(self.tmp, "stuck.rgb")
        with open(stuck, "wb") as f:
            f.write(base * n)
        st = media.video_stats(stuck, w, h, n, 24)
        self.assertEqual((st["frozen_frame_ratio"], st["motion_mean"], st["span_diff"]), (1.0, 0.0, 0.0))
        self.assertFalse(validators.passed(validators.check("video", {}, {
            "path": stuck, "exists": True, "black_frame_ratio": 0.0, "frozen_frame_ratio": 1.0})))

    def test_video_stats_black_frozen_and_duration(self):
        w, h = 48, 32
        ok = os.path.join(self.tmp, "ok.rgb")
        with open(ok, "wb") as f:
            f.write(video_frames(w, h, 9))
        st = media.video_stats(ok, w, h, 9, 24)
        self.assertEqual((st["black_frame_ratio"], st["frozen_frame_ratio"]), (0.0, 0.0))
        self.assertAlmostEqual(st["duration_s"], 9 / 24)
        mixed = os.path.join(self.tmp, "mixed.rgb")
        with open(mixed, "wb") as f:
            f.write(video_frames(w, h, 4, "black") + video_frames(w, h, 6, "frozen"))
        st = media.video_stats(mixed, w, h, 10, 24)
        self.assertAlmostEqual(st["black_frame_ratio"], 0.4)
        self.assertGreaterEqual(st["frozen_frame_ratio"], 8 / 9 - 1e-9)  # repeats within both runs
        with self.assertRaises(media.MediaError):
            media.video_stats(ok, w, h, 10, 24)  # truncated: 9 frames on disk

    def test_png_encoder_round_trip_and_samples(self):
        rgb = bytes(random.Random(1).randrange(256) for _ in range(7 * 5 * 3))
        png = media.encode_png(7, 5, rgb)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(media.decode_png(png), (7, 5, 3, rgb))
        grey = media.encode_png(3, 2, b"\x00\x80\xff\x10\x20\x30", channels=1)
        self.assertEqual(media.decode_png(grey)[3], b"\x00\x80\xff\x10\x20\x30")
        w, h, small = media.downscale_rgb(100, 60, video_frames(100, 60, 1), 25)
        self.assertEqual((w, h, len(small)), (25, 15, 25 * 15 * 3))
        path = os.path.join(self.tmp, "v.rgb")
        with open(path, "wb") as f:
            f.write(video_frames(64, 32, 9))
        samples = media.sample_frames_png(path, 64, 32, 9, n=4, max_side=32)
        self.assertEqual([i for i, _ in samples], [0, 3, 5, 8])
        self.assertEqual(media.decode_png(samples[0][1])[:3], (32, 16, 3))
        wav = os.path.join(self.tmp, "t.wav")
        with open(wav, "wb") as f:
            f.write(tone_wav(0.5, rate=8000))
        self.assertEqual(media.decode_png(media.waveform_png(wav, 64, 16))[:3], (64, 16, 1))


# ---------------------------------------------------------------- M2 SSE

class M2Sse(TmpCase):

    def _big_event(self, payload: bytes, escaped: bool = False) -> bytes:
        import base64
        b64 = base64.b64encode(payload)
        if escaped:
            b64 = b64.replace(b"/", b"\\/")
        return (b'data: {"type":"complete","frames":3,"fps":24,"format":"rgb8","data":"' + b64
                + b'","audio_sample_rate":48000,"audio_data":"' + base64.b64encode(b"\x01\x02" * 50) + b'"}\n\n')

    def _reader(self, events):
        tmp = self.tmp

        def open_sink(head):
            self.assertIn(b'"complete"', head)

            def factory(key, fields):
                return Base64FileSink(os.path.join(tmp, key + ".bin")) if key in ("data", "audio_data") else None
            return factory
        return SseReader(lambda f, s: events.append((f, s)), lambda t: events.append(("comment", t)), open_sink,
                         small_limit=1024)

    def test_small_events_comments_and_done(self):
        events = []
        r = self._reader(events)
        stream = (b': keepalive\n\ndata: {"type":"progress","stage":"Denoising","step":1,"total":4}\n\n'
                  b'event: x\ndata: {"type":"error","message":"boom \\"quoted\\""}\n\ndata: [DONE]\n\n')
        for i in range(0, len(stream), 5):  # awkward chunk boundaries
            r.feed(stream[i:i + 5])
        self.assertEqual(events[0], ("comment", "keepalive"))
        self.assertEqual(events[1][0]["step"], 1)
        self.assertEqual(events[2][0]["message"], 'boom "quoted"')
        self.assertEqual(events[3][0]["type"], "[DONE]")

    def test_big_event_is_streamed_to_disk_across_random_chunks(self):
        payload = bytes(random.Random(7).randrange(256) for _ in range(30000))
        for escaped in (False, True):
            events = []
            r = self._reader(events)
            data = self._big_event(payload, escaped)
            rnd, i = random.Random(3), 0
            while i < len(data):
                n = rnd.randrange(1, 900)
                r.feed(data[i:i + n])
                i += n
            fields, streamed = events[-1]
            self.assertEqual((fields["type"], fields["frames"], fields["audio_sample_rate"]), ("complete", 3, 48000))
            self.assertNotIn("data", fields, "a streamed field must not be kept in memory")
            with open(streamed["data"].path, "rb") as f:
                self.assertEqual(f.read(), payload)
            with open(streamed["audio_data"].path, "rb") as f:
                self.assertEqual(f.read(), b"\x01\x02" * 50)

    def test_truncated_or_nested_big_events_raise(self):
        events = []
        r = self._reader(events)
        data = self._big_event(b"x" * 5000)
        with self.assertRaises(SseError):
            r.feed(data[:-6] + b"\n")  # line ends before the object closes
        r2 = SseReader(lambda f, s: None, small_limit=64)
        with self.assertRaises(SseError):
            r2.feed(b'data: {"type":"complete","data":[{"b64_json":"' + b"A" * 200 + b'"}]}\n')
        r3 = SseReader(lambda f, s: None)
        with self.assertRaises(SseError):
            r3.feed(b"data: {not json}\n")


# ---------------------------------------------------------------- M3 engine

def make_fake(**sizes) -> FakeMlxServe:
    fake = FakeMlxServe(sizes={FOREIGN: int(16.3e9), BRAIN: int(21e9), FALLBACK: int(69e9),
                               "Kokoro-82M-MLX-Serve": int(0.36e9), "ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit": int(4e9),
                               "LTX-2.5-MLX-Serve-4bit": int(36e9), **sizes}).start()
    fake.ready[FOREIGN] = int(16.3e9)          # the user's own model, loaded before gpud
    fake.discovered |= {BRAIN, FALLBACK}       # discovered ids; media packs need load-by-path
    return fake


def make_engine(fake, tmp, **kw) -> MlxServeEngine:
    opts = dict(catalog=studio.catalog(True), model_dir="/fake/models", artifacts_dir=os.path.join(tmp, "art"),
                mem_probe=lambda: 200.0, read_poll_s=0.02, unload_wait_s=0.5, connect_timeout_s=2.0,
                call_timeout_s=2.0, forbidden=studio.FORBIDDEN, ffmpeg=None)
    opts.update(kw)
    return MlxServeEngine(fake.url, **opts)


def spec(model, kind="job", run_id="1.1", **payload) -> RunSpec:
    return RunSpec(run_id, int(run_id.split(".")[0]), kind, model, payload, 30.0)


class EngineCase(TmpCase):
    def setUp(self):
        super().setUp()
        self.fake = make_fake()
        self.engine = make_engine(self.fake, self.tmp)

    def tearDown(self):
        self.engine.close()
        self.fake.stop()
        super().tearDown()

    def load(self, model):
        t = self.engine.load(model)
        self.assertEqual(wait_for(lambda: self.engine.poll_load(t).state != "loading"), True)
        self.assertEqual(self.engine.poll_load(t).state, "ready", self.engine.poll_load(t).error)
        return t

    def finish(self, handle, timeout=8.0):
        wait_for(lambda: self.engine.poll(handle).status != "running", timeout)
        return self.engine.poll(handle)


class M3Engine(EngineCase):

    def test_residency_maps_catalog_names_and_reports_foreign_models(self):
        self.assertEqual(self.engine.resident_models(), {FOREIGN})
        self.assertAlmostEqual(self.engine.model_size_gb(FOREIGN), 16.3)
        self.load(KOKORO)   # not discovered: loaded by absolute path, registered under the basename
        self.load(BRAIN)    # discovered: loaded by id
        loads = [b["model"] for b in self.fake.bodies("/v1/load-model")]
        self.assertEqual(loads, ["/fake/models/ddalcu/Kokoro-82M-MLX-Serve", BRAIN])
        self.assertIn("Kokoro-82M-MLX-Serve", self.fake.ready)
        self.assertEqual(self.engine.resident_models(), {FOREIGN, KOKORO, BRAIN})
        self.assertEqual(self.engine.server_id(KOKORO), "Kokoro-82M-MLX-Serve")

    def test_loads_are_idempotent_and_guarded(self):
        t1 = self.load(KOKORO)
        self.assertEqual(self.engine.load(KOKORO), t1, "load of a resident model returns its ticket")
        with self.assertRaises(ValueError):
            self.engine.load(FOREIGN)
        forbidden = "ddalcu/FastH3-Omni-MLX-Serve-8bit"
        eng = make_engine(self.fake, self.tmp, catalog={**studio.catalog(), forbidden: ModelSpec(
            forbidden, "mlx-serve", 60, 30, "video")})
        with self.assertRaises(PermissionError):
            eng.load(forbidden)
        with self.assertRaises(PermissionError):
            eng.start(spec(forbidden, prompt="x"))
        low = make_engine(self.fake, self.tmp, mem_probe=lambda: 45.0)
        t = low.load(ACE)   # 45 GB free - 8 GB < 40 GB headroom
        self.assertEqual(low.poll_load(t).state, "failed")
        self.assertIn("insufficient headroom", low.poll_load(t).error)
        blind = make_engine(self.fake, self.tmp, mem_probe=lambda: None)
        self.assertIn("cannot measure", blind.poll_load(blind.load(ACE)).error)
        self.load(BRAIN)    # foreign + kokoro + brain = mlx-serve's 3 slots
        t = self.engine.load(ACE)
        self.assertEqual(self.engine.poll_load(t).state, "failed")
        self.assertIn("max-resident-models", self.engine.poll_load(t).error)
        self.assertEqual(len(self.fake.bodies("/v1/load-model")), 2, "refused loads never reach the server")

    def test_a_pack_still_downloading_is_never_loaded(self):
        # Live Studio: MLX-Serve.app was still writing Flash-Next's ngram_table.bin.partial (and
        # its .partial.parts) into --model-dir while gpud ran. Loading it then would fail at best.
        models = os.path.join(self.tmp, "models")
        pack = os.path.join(models, *FALLBACK.split("/"))
        os.makedirs(pack)
        for name in ("config.json", "model-00001.safetensors", "ngram_table.bin.partial",
                     "ngram_table.bin.partial.parts"):
            open(os.path.join(pack, name), "w").close()
        eng = make_engine(self.fake, self.tmp, model_dir=models)
        t = eng.load(FALLBACK)
        self.assertEqual(eng.poll_load(t).state, "failed")
        self.assertIn("still downloading", eng.poll_load(t).error)
        self.assertEqual(self.fake.bodies("/v1/load-model"), [], "an incomplete pack never reaches the server")
        for name in ("ngram_table.bin.partial", "ngram_table.bin.partial.parts"):
            os.remove(os.path.join(pack, name))
        self.assertIsNone(eng.incomplete_pack(FALLBACK))
        t = eng.load(FALLBACK)
        self.assertEqual(wait_for(lambda: eng.poll_load(t).state != "loading"), True)
        self.assertEqual(eng.poll_load(t).state, "ready", eng.poll_load(t).error)
        eng.close()

    def test_load_failure_and_unknown_ticket(self):
        self.fake.load_error = (503, "Not enough memory")
        t = self.engine.load(ACE)
        wait_for(lambda: self.engine.poll_load(t).state != "loading")
        self.assertEqual(self.engine.poll_load(t).state, "failed")
        self.assertIn("Not enough memory", self.engine.poll_load(t).error)
        self.assertEqual(self.engine.poll_load("mlx-serve-load-99-dead").state, "failed")

    def test_unload_is_graceful_refuses_foreign_models_and_never_sends_an_empty_id(self):
        self.load(KOKORO)
        with self.assertRaises(PermissionError):
            self.engine.unload(FOREIGN)
        self.assertTrue(self.engine.unload(KOKORO))
        self.assertEqual(self.fake.bodies("/v1/unload-model")[-1], {"model": "Kokoro-82M-MLX-Serve"})
        self.assertTrue(self.engine.unload(ACE), "unloading a model that is not loaded is a no-op success")
        self.assertIn(FOREIGN, self.fake.ready)
        self.assertNotIn("DANGER", [m for m, _, _ in self.fake.requests])

    def test_unload_that_drains_slowly_is_pending_then_done_and_refusal_is_false(self):
        self.load(KOKORO)
        self.fake.unload_delay = 0.8
        with self.assertRaises(UnloadPending):
            self.engine.unload(KOKORO)
        self.assertTrue(wait_for(lambda: self._try_unload(KOKORO), 3.0))
        self.assertEqual(len(self.fake.bodies("/v1/unload-model")), 1, "one request, not one per tick")
        self.fake.unload_delay = 0
        self.load(KOKORO)
        self.fake.unload_error = (500, "Model unload failed")
        self.assertFalse(self.engine.unload(KOKORO))

    def _try_unload(self, model):
        try:
            return self.engine.unload(model)
        except UnloadPending:
            return False

    def test_terminate_always_refuses(self):
        self.assertFalse(self.engine.terminate())

    def test_speech_run_writes_a_wav_artifact_the_validators_accept(self):
        self.load(KOKORO)
        h = self.engine.start(spec(KOKORO, "speak", input="Hello from the Studio", goal="g", expected_s=5))
        self.assertEqual(self.engine.start(spec(KOKORO, "speak", input="again")), h, "start is idempotent on run_id")
        self.assertEqual(self.engine.find("1.1"), h)
        p = self.finish(h)
        self.assertEqual(p.status, "done", p.error)
        art = p.artifact
        self.assertTrue(art["path"].endswith("output.wav") and os.path.exists(art["path"]))
        self.assertGreater(art["duration_s"], 0.5)
        self.assertLess(art["silence_ratio"], 0.1)
        self.assertTrue(art["paths"][0].endswith("waveform.png"))
        self.assertTrue(validators.passed(validators.check("tts", {}, art)))
        body = self.fake.bodies("/v1/audio/speech")[0]
        self.assertEqual(body, {"input": "Hello from the Studio", "model": "Kokoro-82M-MLX-Serve", "stream": True})

    def test_run_refused_when_model_is_not_resident(self):
        h = self.engine.start(spec(KOKORO, "speak", input="hi"))
        p = self.finish(h)
        self.assertEqual(p.status, "failed")
        self.assertIn("not resident", p.error)
        self.assertEqual(self.fake.bodies("/v1/audio/speech"), [], "never sent: the server would use its default")

    def test_music_progress_renews_and_complete_wav(self):
        self.load(ACE)
        self.fake.progress_delay = 0.03
        h = self.engine.start(spec(ACE, "compose_music", prompt="lofi piano", duration_s=12, instrumental=True))
        seen = []
        while self.engine.poll(h).status == "running":
            seen.append(self.engine.poll(h).progress)
            time.sleep(0.01)
        p = self.engine.poll(h)
        self.assertEqual(p.status, "done", p.error)
        self.assertEqual(seen, sorted(seen))
        self.assertGreater(len(set(seen)), 3, "progress moved during the run")
        body = self.fake.bodies("/v1/audio/music-generations")[0]
        self.assertEqual(body["duration_seconds"], 12)
        self.assertNotIn("goal", body)

    def test_progress_strictly_increases_on_every_event_even_when_steps_restart(self):
        run = mlx_serve._Run("h", spec(ACE), "music", ACE)
        values = []
        for stage, step, total in [("encode", 0, 1), ("encode", 1, 1), ("diffuse", 1, 8), ("diffuse", 8, 8),
                                   ("decode", 0, 4), ("decode", 1, 4), ("decode", 4, 4), ("x", None, 0)]:
            self.engine._on_event(run, {"type": "progress", "stage": stage, "step": step, "total": total}, {})
            values.append(run.progress)
        self.assertTrue(all(b > a + 1e-9 for a, b in zip(values, values[1:])), values)
        self.assertLess(values[-1], mlx_serve.EVENT_CAP + 1e-12)

    def test_video_run_streams_frames_to_disk_and_samples_pngs(self):
        self.load(LTX)
        for behaviour in ("ok", "escaped"):
            self.fake.script["/v1/video/generations"].append(behaviour)
            run_id = f"{len(behaviour)}.1"
            h = self.engine.start(spec(LTX, "render_video", run_id, prompt="a fox", width=128, height=96,
                                       num_frames=25, duration_s=1.0))
            p = self.finish(h)
            self.assertEqual(p.status, "done", p.error)
            art = p.artifact
            self.assertEqual((art["width"], art["height"], art["frame_count"], art["fps"]), (128, 96, 25, 24))
            self.assertEqual(os.path.getsize(art["frames_path"]), 128 * 96 * 3 * 25)
            self.assertEqual(art["black_frame_ratio"], 0.0)
            self.assertAlmostEqual(art["duration_s"], 25 / 24)
            self.assertEqual(len(art["samples"]), 4)
            with open(art["samples"][0], "rb") as f:
                self.assertEqual(media.decode_png(f.read())[:2], (128, 96))
            with open(art["meta"]) as f:
                meta = json.load(f)
            self.assertEqual((meta["frame_count"], meta["audio_sample_rate"]), (25, 48000))
            self.assertAlmostEqual(media.wav_stats(art["audio"])["duration_s"], 0.1, places=3)
            self.assertEqual(art["format"], "rgb8", "no ffmpeg: frames + samples, no mp4")
            self.assertTrue(validators.passed(validators.check("video", {"duration_s": 1.0}, art)))
        body = self.fake.bodies("/v1/video/generations")[0]
        self.assertEqual((body["num_frames"], body["width"], body["model"]), (25, 128, "LTX-2.5-MLX-Serve-4bit"))

    def test_black_video_fails_its_validator(self):
        self.load(LTX)
        self.fake.script["/v1/video/generations"].append("black")
        p = self.finish(self.engine.start(spec(LTX, "render_video", prompt="night", width=64, height=32,
                                               num_frames=9)))
        self.assertEqual(p.status, "done")
        results = validators.check("video", {}, p.artifact)
        self.assertFalse(validators.passed(results))
        self.assertIn("black_frames", validators.failures(results))

    def test_video_frames_follow_ltx_8k_plus_1(self):
        self.assertEqual([video_frames_for(s) for s in (0.1, 1, 5, 10)], [9, 25, 121, 241])
        body = self.engine.request_body(spec(LTX, prompt="x", duration_s=5))
        self.assertEqual((body["num_frames"], body["width"], body["height"]), (121, 768, 512))

    def test_error_event_and_http_error_fail_the_run(self):
        self.load(ACE)
        self.fake.script["/v1/audio/music-generations"].extend(["error", "http500"])
        p = self.finish(self.engine.start(spec(ACE, prompt="x", duration_s=10, run_id="1.1")))
        self.assertEqual((p.status, p.error), ("failed", "mlx-serve: generation failed"))
        p = self.finish(self.engine.start(spec(ACE, prompt="x", duration_s=10, run_id="2.1")))
        self.assertEqual(p.status, "failed")
        self.assertIn("HTTP 500", p.error)

    def test_hang_then_cancel_closes_the_connection(self):
        self.load(ACE)
        self.fake.script["/v1/audio/music-generations"].append(("hang", 2))
        h = self.engine.start(spec(ACE, prompt="x", duration_s=10))
        wait_for(lambda: self.engine.poll(h).progress > 0)
        frozen = self.engine.poll(h).progress
        time.sleep(0.2)
        self.assertEqual((self.engine.poll(h).status, self.engine.poll(h).progress), ("running", frozen))
        self.engine.cancel(h)
        self.assertEqual(self.engine.poll(h).status, "cancelled")
        self.assertTrue(wait_for(lambda: self.fake.disconnects, 3.0), "mlx-serve saw the client disconnect")
        with self.assertRaises(LookupError):
            self.engine.cancel("mlx-serve:dead:9.9")

    def test_server_dropping_the_stream_fails_or_loses_the_run(self):
        self.load(ACE)
        self.fake.script["/v1/audio/music-generations"].extend(["drop", "die"])
        p = self.finish(self.engine.start(spec(ACE, prompt="x", duration_s=10, run_id="1.1")))
        self.assertEqual(p.status, "failed")
        self.assertIn("without a complete or error event", p.error)
        p = self.finish(self.engine.start(spec(ACE, prompt="x", duration_s=10, run_id="2.1")))
        self.assertEqual(p.status, "lost", "server stopped answering /health")

    def test_fault_injection_stalls_after_n_events_until_cancelled(self):
        eng = make_engine(self.fake, self.tmp, stall_after=2, stall_kinds=("compose_music",))
        try:
            self.load(ACE)
            self.fake.progress_delay = 0.02
            self.fake.steps = 80  # still streaming when the cancel arrives
            h = eng.start(spec(ACE, "compose_music", prompt="x", duration_s=10))
            wait_for(lambda: any("FAULT INJECTION" in line for line in eng.poll(h).log_tail))
            frozen = eng.poll(h).progress
            time.sleep(0.4)  # the server keeps sending; nothing is read or reported
            self.assertEqual((eng.poll(h).status, eng.poll(h).progress), ("running", frozen))
            eng.cancel(h)
            self.assertEqual(eng.poll(h).status, "cancelled")
            self.assertTrue(wait_for(lambda: self.fake.disconnects, 3.0))
            self.fake.steps, self.fake.progress_delay = 4, 0.0
            other = eng.start(spec(ACE, "other_kind", "2.1", prompt="x", duration_s=10))
            wait_for(lambda: eng.poll(other).status != "running")
            p = eng.poll(other)
            self.assertEqual(p.status, "done", (p.error, p.log_tail))
        finally:
            eng.close()

    def test_a_restarted_process_owns_no_runs(self):
        self.load(ACE)
        self.fake.script["/v1/audio/music-generations"].append("hang")
        h = self.engine.start(spec(ACE, prompt="x", duration_s=10))
        fresh = make_engine(self.fake, self.tmp)
        self.assertIsNone(fresh.find("1.1"))
        p = fresh.poll(h)
        self.assertEqual(p.status, "lost")
        self.assertIn("does not own", p.log_tail[0])
        self.assertTrue(wait_for(lambda: self.fake.bodies("/v1/audio/music-generations"), 3.0))
        time.sleep(0.1)
        self.engine.close()  # the old process "dies": its connection closes
        self.assertTrue(wait_for(lambda: self.fake.disconnects, 3.0))

    def test_decode_heartbeat_is_bounded(self):
        run = mlx_serve._Run("h", spec(LTX), "video", LTX)
        run.decode_grace = 0.3
        self.engine._runs["h"] = run
        self.engine._on_event(run, {"type": "progress", "stage": "Denoising", "step": 30, "total": 30}, {})
        self.engine._on_event(run, {"type": "progress", "stage": "Decoding video", "step": 30, "total": 30}, {})
        a = self.engine.poll("h").progress
        time.sleep(0.1)
        b = self.engine.poll("h").progress
        time.sleep(0.35)
        c = self.engine.poll("h").progress
        time.sleep(0.1)
        d = self.engine.poll("h").progress
        self.assertTrue(mlx_serve.EVENT_CAP <= a < b < c, (a, b, c))
        self.assertEqual(c, d, "after the decode budget the heartbeat stops and the lease can expire")
        self.assertAlmostEqual(d, mlx_serve.DECODE_TOP)

    def test_every_tick_call_is_bounded_when_the_server_hangs(self):
        import socket
        hung = socket.socket()
        hung.bind(("127.0.0.1", 0))
        hung.listen(5)  # accepts connections, never answers
        try:
            eng = MlxServeEngine(f"http://127.0.0.1:{hung.getsockname()[1]}", catalog=studio.catalog(),
                                 call_timeout_s=0.3, mem_probe=lambda: 200.0, artifacts_dir=self.tmp)
            t0 = time.monotonic()
            with self.assertRaises(OSError):
                eng.resident_models()
            self.assertLess(time.monotonic() - t0, 2.0)
        finally:
            hung.close()


# ---------------------------------------------------------------- M4 brain

def brief(kind="speak", model=KOKORO, attempt=1, max_attempts=3, artifacts=(), **outcome):
    return {"goal": "say hi", "attempted_action": {
        "kind": kind, "model": model, "params": {"input": "hi", "expected_s": 5},
        "on_success": {"kind": "review_audio", "needs": "vision"}, "on_failure": {"kind": "recover_audio"}},
        "outcome": {"status": "failed", "reason": "crash", **outcome}, "artifacts": list(artifacts),
        "validator_results": [], "log_tail": ["x"], "attempt": attempt, "max_attempts": max_attempts,
        "source_job_id": 7}


class M4Brain(TmpCase):

    def test_parse_tolerates_fences_and_prose(self):
        self.assertEqual(parse_json_object('```json\n{"verdict": "accept"}\n```'), {"verdict": "accept"})
        self.assertEqual(parse_json_object('Sure! {"verdict": "give_up", "note": "a {b}"} done'),
                         {"verdict": "give_up", "note": "a {b}"})
        self.assertIsNone(parse_json_object("no json here"))
        self.assertIsNone(parse_json_object("[1, 2]"))

    def test_decisions_are_safe(self):
        cat = studio.catalog()
        d = decide(None, brief(), cat, raw="I think it is fine")
        self.assertEqual((d["verdict"], d["submit"]), ("human_review", []))
        self.assertEqual(decide({"verdict": "launch_missiles"}, brief(), cat)["verdict"], "human_review")
        self.assertEqual(decide({"verdict": "accept", "note": "ok"}, brief(), cat),
                         {"verdict": "accept", "note": "ok", "submit": []})
        d = decide({"verdict": "retry", "note": "try a new seed", "changes": {"seed": 5, "requires": "x",
                                                                               "voice": 3}}, brief(), cat)
        req = d["submit"][0]
        self.assertEqual((req["kind"], req["requires"], req["retry_of"]), ("speak", KOKORO, 7))
        self.assertEqual((req["payload"]["seed"], req["payload"]["goal"], req["payload"]["input"]), (5, "say hi", "hi"))
        self.assertNotIn("voice", req["payload"])
        self.assertIn("ignored changes", d["note"])
        self.assertEqual(req["on_failure"], {"kind": "recover_audio"})
        exhausted = decide({"verdict": "retry"}, brief(attempt=3), cat)
        self.assertEqual((exhausted["verdict"], exhausted["submit"]), ("human_review", []))
        broken = brief()
        broken["attempted_action"].pop("on_failure")
        self.assertEqual(decide({"verdict": "retry"}, broken, cat)["verdict"], "human_review")

    def test_review_prompt_attaches_sampled_frames_for_a_vision_brain(self):
        frames = []
        for i in range(6):
            p = os.path.join(self.tmp, f"frame_{i:04d}.png")
            with open(p, "wb") as f:
                f.write(media.encode_png(2, 2, bytes(12)))
            frames.append(p)
        meta = os.path.join(self.tmp, "video.json")
        with open(meta, "w") as f:
            json.dump({"width": 2, "height": 2, "fps": 24, "frame_count": 6, "stats": {"black_frame_ratio": 0}}, f)
        b = brief("render_video", LTX, artifacts=[os.path.join(self.tmp, "frames.rgb"), meta] + frames)
        bc = BrainClient(catalog=studio.catalog(), frames=3)
        msgs = bc.messages("review_render", b, vision=True)
        images = [c for c in msgs[1]["content"] if c["type"] == "image_url"]
        self.assertEqual(len(images), 3)
        self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn('"frame_count": 6', msgs[1]["content"][0]["text"])
        self.assertIn("prompt, seed, steps", msgs[0]["content"])
        text_only = bc.messages("review_render", b, vision=False)
        self.assertIsInstance(text_only[1]["content"], str)
        self.assertIn("no vision", text_only[1]["content"])

    def test_checkpoints_are_deterministic(self):
        bc = BrainClient()
        nxt = {"kind": "render_video", "requires": LTX}
        rs = RunSpec("1.1", 1, "batch_checkpoint", BRAIN, {"brief": {"context": {"next": nxt}}}, 30)
        self.assertEqual(bc(rs, BRAIN)["submit"], [nxt])
        rs = RunSpec("2.1", 2, "brain_checkpoint", BRAIN, {"brief": {}}, 30)
        self.assertEqual(bc(rs, BRAIN)["verdict"], "ack")

    def test_chat_without_context_uses_a_plain_completion(self):
        fake = make_fake()
        try:
            fake.ready[BRAIN] = 1
            fake.chat_replies.append('{"verdict": "give_up", "note": "bad request"}')
            bc = BrainClient(fake.url, catalog=studio.catalog())
            d = bc(RunSpec("1.1", 1, "recover_audio", BRAIN, {"brief": brief()}, 30), BRAIN)
            self.assertEqual(d["verdict"], "give_up")
            body = fake.bodies("/v1/chat/completions")[0]
            self.assertEqual((body["model"], body["enable_thinking"], body["stream"]), (BRAIN, False, False))
            self.assertEqual(body["response_format"], {"type": "json_object"})
        finally:
            fake.stop()


# ---------------------------------------------------------------- M5 supervisor features (simulator)

class M5SupervisorFeatures(TmpCase):

    def sim(self, **policy):
        return Sim(os.path.join(self.tmp, "f.db"), policy=Policy(**policy))

    def test_foreign_models_are_pinned_when_unload_foreign_is_off(self):
        sim = self.sim(unload_foreign=False)
        sim.run_until_quiet(600)
        foreign = "someone-elses-model"
        sim.mlx.catalog = {**CATALOG, foreign: ModelSpec(foreign, "mlx-serve", 16, 10, "llm")}
        sim.mlx._models.add(foreign)
        job = sim.sup.submit(**render_request("a fox", duration_s=4.0, expected_s=120.0))
        self.assertTrue(sim.run_until(lambda: sim.store.job(job).state == "succeeded", 3600))
        sim.run_until_quiet(600)
        self.assertIn(foreign, sim.mlx.resident_models())
        self.assertFalse([c for c in sim.mlx.calls if c[1] == "unload" and c[2] == foreign])

    def test_evicts_brain_unloads_a_brain_that_would_fit(self):
        cat = {**CATALOG, "ltx-2.5": ModelSpec("ltx-2.5", "mlx-serve", 150, 40, "video", evicts_brain=True)}
        sim = Sim(os.path.join(self.tmp, "f.db"))
        sim.sup.catalog = cat
        sim.run_until_quiet(600)
        job = sim.sup.submit(**render_request("a fox", duration_s=4.0, expected_s=120.0))
        sim.run_until(lambda: sim.store.job(job).state == "running", 600)
        self.assertFalse(any(CATALOG[m].is_brain for m in sim.mlx.resident_models()),
                         "no brain resident while the evicting model renders")
        self.assertTrue(sim.run_until_quiet(3600))
        review = [j for j in sim.store.jobs() if j.kind == "review_render"]
        self.assertEqual([j.state for j in review], ["succeeded"])
        self.assertTrue(sim.brain_resident())

    def test_engine_slot_cap_is_respected_by_unloading_gpud_models_first(self):
        sim = self.sim(unload_foreign=False)
        sim.mlx.max_resident_models = 3
        foreign = "someone-elses-model"
        sim.mlx.catalog = {**CATALOG, foreign: ModelSpec(foreign, "mlx-serve", 16, 10, "llm")}
        sim.mlx._models.add(foreign)
        counts = []
        orig = sim.mlx.load

        def load(model):
            t = orig(model)
            counts.append(len(sim.mlx.resident_models()) + 1)
            return t
        sim.mlx.load = load
        tts = sim.sup.submit("speak", "qwen3-tts", {"goal": "hi", "expected_s": 20},
                             {"kind": "review"}, {"kind": "recover"})
        music = sim.sup.submit("compose", "ace-step", {"goal": "tune", "expected_s": 20},
                               {"kind": "review"}, {"kind": "recover"})
        self.assertTrue(sim.run_until_quiet(3600))
        self.assertEqual((sim.store.job(tts).state, sim.store.job(music).state), ("succeeded", "succeeded"))
        self.assertLessEqual(max(counts), 3, f"never more than 3 models in the engine: {counts}")
        self.assertIn(foreign, sim.mlx.resident_models())

    def test_an_unload_still_draining_is_busy_not_an_engine_outage(self):
        # Live Studio, S5: after the lease closed a stalled LTX stream, mlx-serve's unload waited
        # ~12 s for the render to reach its next step boundary. The adapter's UnloadPending was
        # recorded as engine_error / engine_recovered pairs, marking the engine down each tick.
        cat = {**CATALOG, "ltx-2.5": ModelSpec("ltx-2.5", "mlx-serve", 150, 40, "video", evicts_brain=True)}
        sim = Sim(os.path.join(self.tmp, "f.db"))
        sim.sup.catalog = cat
        sim.run_until_quiet(600)
        orig, left = sim.mlx.unload, [3]

        def unload(model):
            if left[0] > 0:
                left[0] -= 1
                raise mlx_serve.UnloadPending(f"unload of {model} still draining")
            return orig(model)
        sim.mlx.unload = unload
        job = sim.sup.submit(**render_request("a fox", duration_s=4.0, expected_s=120.0))
        self.assertTrue(sim.run_until_quiet(3600))
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assertEqual(left[0], 0, "the unload really was pending three times")
        kinds = [e["kind"] for e in sim.store.events()]
        self.assertNotIn("engine_error", kinds)
        self.assertEqual(kinds.count("engine_busy"), 1, "one note per wait, not one per tick")
        self.assertTrue(sim.brain_resident())

    def test_brain_verdict_human_review_pages_a_person(self):
        class Asks(ScriptedBrain):
            def __call__(self, spec, model):
                if spec.kind.startswith("review"):
                    return {"verdict": "human_review", "note": "the fox has five legs"}
                return super().__call__(spec, model)
        sim = Sim(os.path.join(self.tmp, "f.db"), brain=Asks())
        sim.sup.submit(**render_request("a fox", duration_s=4.0, expected_s=120.0))
        self.assertTrue(sim.run_until_quiet(3600))
        self.assertTrue(any("five legs" in (p.get("reason") or "") for p in sim.pages), sim.pages)


# ---------------------------------------------------------------- M6 gpud on the fake server

class ManualClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return 1.7e9 + self.t

    def monotonic(self):
        return self.t


class M6EndToEnd(TmpCase):
    def setUp(self):
        super().setUp()
        self.fake = make_fake()
        self.fake.ready.pop(FOREIGN)
        self.fake.ready[FOREIGN] = int(16.3e9)
        self.catalog = studio.catalog(True)
        self.brain = BrainClient(self.fake.url, catalog=self.catalog)
        self.engine = make_engine(self.fake, self.tmp, brain=self.brain)
        self.clock = ManualClock()
        self.pages = []
        self.sup = Supervisor(os.path.join(self.tmp, "foreman.db"), {"mlx-serve": self.engine}, self.clock,
                              studio.policy(), catalog=self.catalog, pager=self.pages.append)
        self.sup.recover()

    def tearDown(self):
        self.sup.close()
        self.engine.close()
        self.fake.stop()
        super().tearDown()

    def run_until(self, pred, timeout=20.0, dt=1.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.clock.t += dt
            self.sup.tick()
            if pred():
                return True
            time.sleep(0.01)
        return False

    def jobs(self, kind):
        return [j for j in self.sup.store.jobs() if j.kind == kind]

    def submit(self, req):
        return self.sup.submit(req["kind"], req["requires"], req["payload"], req["on_success"], req["on_failure"])

    def test_speech_then_vision_review_with_the_users_model_untouched(self):
        from foreman.cli import speech_request
        job = self.submit(speech_request("Hello from the Studio"))
        self.assertTrue(self.run_until(lambda: [j.state for j in self.jobs("review_audio")] == ["succeeded"]),
                        [(j.kind, j.state, j.outcome) for j in self.sup.store.jobs()])
        self.assertEqual(self.sup.store.job(job).state, "succeeded")
        review = self.jobs("review_audio")[0]
        self.assertEqual(review.outcome["artifact"]["decision"]["verdict"], "accept")
        chat = self.fake.bodies("/v1/chat/completions")[0]
        self.assertEqual(chat["model"], BRAIN)
        parts = chat["messages"][1]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image_url"], "waveform attached for the vision brain")
        self.assertIn(FOREIGN, self.fake.ready, "the user's model is never unloaded")
        self.assertNotIn(FOREIGN, [b.get("model") for b in self.fake.bodies("/v1/unload-model")])

    def test_video_unloads_the_brain_and_the_brain_comes_back_to_review_frames(self):
        from foreman.cli import video_request
        self.assertTrue(self.run_until(lambda: BRAIN in self.engine.resident_models()), "idle: brain comes home")
        req = video_request("a red fox trotting through snow", 1.0, width=128, height=96, expected_s=60)
        job = self.submit(req)
        self.assertTrue(self.run_until(lambda: [j.state for j in self.jobs("review_render")] == ["succeeded"]),
                        [(j.kind, j.state, j.outcome) for j in self.sup.store.jobs()])
        self.assertEqual(self.sup.store.job(job).state, "succeeded")
        ev = [(e["kind"], e["data"].get("model")) for e in self.sup.store.events()
              if e["kind"] in ("model_loaded", "model_unloaded", "job_starting")]
        start = ev.index(("job_starting", LTX))
        self.assertIn(("model_unloaded", BRAIN), ev[:start], "brain unloaded before the render (video_exclusive)")
        self.assertIn(("model_loaded", BRAIN), ev[start:], "brain brought back afterwards")
        chat = self.fake.bodies("/v1/chat/completions")[-1]
        images = [p for p in chat["messages"][1]["content"] if p["type"] == "image_url"]
        self.assertEqual(len(images), 3)
        self.assertLessEqual(len(self.fake.ready), 3)
        self.assertIn(FOREIGN, self.fake.ready)

    def test_dead_man_lease_fires_on_a_stalled_run_and_the_baton_reaches_the_brain(self):
        self.engine.stall_after = 2
        self.fake.progress_delay = 0.02
        self.fake.steps = 400  # still streaming when the lease fires
        self.fake.chat_replies.append('{"verdict": "give_up", "note": "stalled twice is enough"}')
        from foreman.cli import music_request
        job = self.submit(music_request("lofi piano", 12, expected_s=100))  # hard 250 s, no-progress 120 s
        self.assertTrue(self.run_until(lambda: self.sup.store.job(job).state == "running"))
        handle = self.sup.store.job(job).handle
        self.assertTrue(wait_for(lambda: self.engine.poll(handle).log_tail
                                 and "FAULT" in self.engine.poll(handle).log_tail[-1], 5.0))
        self.clock.t += 125.0  # past the 120 s no-progress floor, in supervisor time
        self.assertTrue(self.run_until(lambda: self.sup.store.job(job).state in ("failed", "quarantined"), 10))
        out = self.sup.store.job(job).outcome
        self.assertEqual((out["reason"], out["timeout"], out["cleanup"]), ("timeout", "no_progress", "unloaded"))
        self.assertTrue(wait_for(lambda: "/v1/audio/music-generations" in self.fake.disconnects, 3.0),
                        "the cancel reached mlx-serve as a client disconnect")
        self.assertTrue(self.run_until(lambda: [j.state for j in self.jobs("recover_music")] == ["succeeded"]))
        decision = self.jobs("recover_music")[0].outcome["artifact"]["decision"]
        self.assertEqual(decision["verdict"], "give_up")
        self.assertTrue(any(e["kind"] == "lease_expired" for e in self.sup.store.events()))


# ---------------------------------------------------------------- M7 CLI, daemon, LaunchAgent

class M7Process(TmpCase):

    def env(self):
        return {"FOREMAN_HOME": self.tmp}

    def cli(self, *args):
        from foreman import cli
        out = io.StringIO()
        old = os.environ.get("FOREMAN_HOME")
        os.environ["FOREMAN_HOME"] = self.tmp
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = cli.main(list(args))
        finally:
            if old is None:
                os.environ.pop("FOREMAN_HOME", None)
            else:
                os.environ["FOREMAN_HOME"] = old
        return code, out.getvalue()

    def test_cli_submits_as_a_non_leader_client_and_reports(self):
        from foreman.supervisor import InstanceLock
        code, out = self.cli("submit-speech", "Hello there")
        self.assertEqual(code, 0, out)
        self.assertIn("job 1 queued: speak", out)
        code, out = self.cli("submit-music", "lofi piano", "--seconds", "20")
        self.assertEqual(code, 0, out)
        code, out = self.cli("submit-video", "a fox", "--seconds", "5", "--width", "1280", "--height", "720")
        self.assertEqual(code, 0, out)
        code, out = self.cli("submit-music", "too short", "--seconds", "3")
        self.assertEqual(code, 2)
        self.assertIn("refused", out)
        lock = InstanceLock(os.path.join(self.tmp, "foreman.db.lock"))
        self.assertTrue(lock.acquire(), "the CLI never holds gpud's lock")
        lock.release()
        code, out = self.cli("jobs")
        self.assertIn("render_video", out)
        code, out = self.cli("status")
        self.assertIn('"queued": 3', out)
        code, out = self.cli("events", "--tail", "5")
        self.assertIn("admitted", out)
        code, out = self.cli("cancel", "1")
        self.assertEqual(code, 0, out)
        code, out = self.cli("clear-safe")
        self.assertEqual(code, 0)
        from foreman.store import Store
        st = Store(os.path.join(self.tmp, "foreman.db"))
        video = st.job(3)
        self.assertEqual((video.payload["width"], video.payload["height"], video.payload["num_frames"]),
                         (1280, 704, 121))
        self.assertEqual(st.job(1).state, "failed")
        st.close()

    def test_daemon_runs_ticks_logs_json_and_stops_cleanly(self):
        from foreman.daemon import Gpud, setup_logging
        fake = make_fake()
        try:
            cfg = studio.load_config(env={"FOREMAN_HOME": self.tmp}, base_url=fake.url)
            cfg.tick_s = 0.02
            eng = make_engine(fake, self.tmp, brain=BrainClient(fake.url, catalog=studio.catalog()))
            setup_logging(cfg.log_dir, "INFO", stderr=False)
            box = {}

            def main():  # SQLite connections belong to the thread that opens them, as in gpud
                box["g"] = Gpud(cfg, engine=eng)
                box["code"] = box["g"].run()
            t = threading.Thread(target=main)
            t.start()
            self.assertTrue(wait_for(lambda: "g" in box and box["g"].ticks >= 5, 5.0))
            g = box["g"]
            g.stop.set()
            t.join(5.0)
            self.assertFalse(t.is_alive())
            self.assertEqual(box["code"], 0)
            from foreman.supervisor import InstanceLock
            lock = InstanceLock(cfg.db_path + ".lock")
            self.assertTrue(lock.acquire(), "lock released on shutdown")
            lock.release()
            import logging
            for h in logging.getLogger().handlers:
                h.flush()
            with open(os.path.join(cfg.log_dir, "gpud.log")) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            kinds = {line.get("event") for line in lines}
            self.assertIn("supervisor_start", kinds)
            self.assertTrue(any(line["msg"] == "gpud starting" for line in lines))
        finally:
            import logging
            root = logging.getLogger()
            for h in list(root.handlers):
                root.removeHandler(h)
                h.close()
            fake.stop()

    def test_video_expected_time_follows_studio_measurements(self):
        from foreman import cli
        # measured: 8 MPF 9-12 s, 109 MPF 70-82 s, 217 MPF 177 s (request to artifact on disk)
        for mpf, measured in ((8.0, 12.0), (109.0, 82.0), (217.0, 177.0)):
            exp = cli.video_expected_s(mpf)
            self.assertGreater(exp, 1.5 * measured, mpf)
            self.assertLess(exp, 5.0 * measured, mpf)
        self.assertEqual(cli.video_expected_s(10000.0), 640.0)
        req = cli.video_request("a fox", 5, 1280, 704)
        self.assertEqual((req["payload"]["num_frames"], req["payload"]["width"], req["payload"]["height"]),
                         (121, 1280, 704))
        self.assertAlmostEqual(req["payload"]["expected_s"], cli.video_expected_s(121 * 1280 * 704 / 1e6))

    def test_config_layers(self):
        path = os.path.join(self.tmp, "c.json")
        with open(path, "w") as f:
            json.dump({"base_url": "http://studio:11234", "video_exclusive": True, "stall_kinds": ["render_video"]}, f)
        cfg = studio.load_config(path, env={"FOREMAN_HOME": self.tmp, "FOREMAN_VIDEO_EXCLUSIVE": "0",
                                            "FOREMAN_FAULT_STALL_AFTER": "3"})
        self.assertEqual((cfg.base_url, cfg.video_exclusive, cfg.stall_after, cfg.stall_kinds),
                         ("http://studio:11234", False, 3, ("render_video",)))
        self.assertEqual(cfg.db_path, os.path.join(self.tmp, "foreman.db"))
        self.assertFalse(studio.catalog(False)[LTX].evicts_brain)
        self.assertTrue(studio.catalog()[LTX].evicts_brain)
        self.assertNotIn("ddalcu/FastH3-Omni-MLX-Serve-8bit", studio.catalog())
        self.assertEqual(cfg.policy().progress_timeout_s, 120.0)
        tight = studio.load_config(path, env={"FOREMAN_HOME": self.tmp, "FOREMAN_PROGRESS_TIMEOUT_S": "45"})
        self.assertEqual(tight.policy().progress_timeout_s, 45.0)
        self.assertEqual(tight.policy().text_ladder, studio.policy().text_ladder)
        with open(path, "w") as f:
            json.dump({"nonsense": 1}, f)
        with self.assertRaises(ValueError):
            studio.load_config(path, env={"FOREMAN_HOME": self.tmp})

    def test_launchagent_template_renders_to_a_valid_plist(self):
        script = os.path.join(REPO, "scripts", "install-launchagent.sh")
        res = subprocess.run(["bash", script, "--dry-run"], capture_output=True, timeout=20,
                             env={**os.environ, "FOREMAN_PYTHON": "/usr/bin/python3"})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertNotIn(b"__", res.stdout.split(b"<plist", 1)[1])
        plist = plistlib.loads(res.stdout)
        self.assertEqual(plist["Label"], "com.github.ggaabe.foreman.gpud")
        self.assertEqual(plist["ProgramArguments"], ["/usr/bin/python3", "-m", "foreman.daemon"])
        self.assertEqual((plist["KeepAlive"], plist["ThrottleInterval"], plist["ExitTimeOut"],
                          plist["ProcessType"]), (True, 30, 60, "Background"))
        self.assertEqual(plist["WorkingDirectory"], REPO)
        self.assertTrue(plist["StandardErrorPath"].startswith(os.path.expanduser("~") + "/.foreman/logs/"))
        for name in ("install-launchagent.sh", "uninstall-launchagent.sh"):
            with open(os.path.join(REPO, "scripts", name)) as f:
                text = f.read()
            self.assertNotIn("sudo ", text.replace("no sudo", ""))
            self.assertIn("gui/$(id -u)", text)


if __name__ == "__main__":
    unittest.main()
