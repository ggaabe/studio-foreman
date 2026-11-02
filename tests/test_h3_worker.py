"""The h3 media worker (foreman/h3: probe, edit, audio, captions, ff, the worker entry point):

  W1  the core never imports the worker (numpy, Pillow, foreman.h3) and the worker parses as Python 3.9
  W2  captions and publish files without media: chunks, a beat's position from its first event (B2),
      hidden ranges, solo words, generic alignment, SRT cues that never overlap, the publish gate's hash (B10)
  W3  probe on synthetic takes (ffmpeg-made MP4s with tone bursts): frames, levels, runs, problems, hands (B8),
      skipping a take already probed
  W4  edit on a synthetic two-beat plan: frame-exact output, removals only in silence and only under graphic
      cutaways (B3), captions hidden where asked, the moved beat's every caption, bt709, publish and SRT files
  W5  the worker protocol: progress lines then one JSON result, exit codes 2 (content: B4, B9), 3 (infra), and a
      SIGTERM that leaves no ffmpeg behind

Takes are small (144x256) and the graphics are a stub, so the edit runs in seconds; W2 of the graphics has its
own tests."""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import wave

try:
    from ad_fixtures import FFMPEG
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import FFMPEG

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "foreman", "h3", "worker.py")
FONT = next((p for p in ("/System/Library/Fonts/Supplemental/Arial Black.ttf",
                         "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if os.path.exists(p)), None)


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


MEDIA = _importable("numpy") and _importable("PIL")
needs_media = unittest.skipUnless(MEDIA and FFMPEG, "numpy, Pillow and ffmpeg are needed for the h3 worker")
needs_font = unittest.skipUnless(FONT, "no caption font on this machine")


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

W, H = 144, 256          # take render size for the tests (H3 renders 576x1024; the edit scales either)


# ---------------------------------------------------------------- synthetic takes

def tone_wav(path: str, seconds: float, words, hum=(), rate: int = 32000, freq: float = 300.0) -> str:
    """Mono s16: a tone at -12 dBFS over each word span, a low 200 Hz hum over each hum span (-43 dBFS RMS once
    it is AAC in a take and back at 16 kHz mono: under the -40 dBFS a silent run needs, over the -45 dBFS an edit
    may cut), digital silence elsewhere."""
    amp, low = 0.25 * 32767, 10 ** (-37 / 20) * 32767
    out = bytearray()
    for i in range(int(seconds * rate)):
        t = i / rate
        v = 0.0
        if any(s <= t < e for s, e in words):
            v = amp * math.sin(2 * math.pi * freq * t)
        elif any(s <= t < e for s, e in hum):
            v = low * math.sin(2 * math.pi * 200.0 * t)
        out += struct.pack("<h", int(max(-32768, min(32767, v))))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(out))
    return path


def make_take(path: str, seconds: float, words, hum=(), hands=None, size=(W, H), audio: bool = True,
              silent: bool = False, hands_y: float = 0.82) -> str:
    """An H3-like take: 24 fps h264 yuv420p, a blue checkerboard (never skin-coloured; a punch-in changes it),
    AAC 32 kHz stereo with a tone burst per word. hands=(t0, t1): a skin-coloured block in the bottom band of the
    picture (from hands_y of its height, 0.14 of it tall) from t0 to t1."""
    w, h = size
    sq = "mod(floor(X/8)+floor(Y/8)\\,2)"
    vf = f"geq=r='if({sq}\\,230\\,20)':g='if({sq}\\,120\\,10)':b='if({sq}\\,250\\,60)',"
    vf += f"drawbox=x={w // 4}:y={int(hands_y * h)}:w={w // 2}:h={int(0.14 * h)}:color=0xDCA078:t=fill"
    vf += f":enable='between(t,{hands[0]},{hands[1]})'" if hands else ":enable=0"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=0x2848B0:s={w}x{h}:r=24:d={seconds}"]
    if audio:
        wav_path = path[:-4] + "-src.wav"
        tone_wav(wav_path, seconds, [] if silent else words, hum)
        cmd += ["-i", wav_path]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-frames:v", str(round(seconds * 24))]
    cmd += ["-c:a", "aac", "-ar", "32000", "-ac", "2", "-t", str(seconds)] if audio else ["-an"]
    subprocess.run(cmd + [path], check=True)
    return path


def worker(cwd: str, command: str, req: dict, env=None):
    """Run the worker as the core does: -I, the work directory as cwd. Returns (exit code, progress lines, result)."""
    path = os.path.join(cwd, f"{command}-req.json")
    with open(path, "w") as f:
        json.dump(req, f)
    p = subprocess.run([sys.executable, "-I", WORKER, command, path], cwd=cwd, capture_output=True, text=True,
                       timeout=600, env=env)
    lines = [json.loads(ln) for ln in p.stdout.splitlines() if ln.strip()]
    return p.returncode, lines[:-1], (lines[-1] if lines else None), p.stderr


def corrupt_audio_config(mp4: str) -> None:
    """Break the AAC AudioSpecificConfig in the file's esds box: the stream is listed, it does not decode."""
    with open(mp4, "rb") as f:
        b = bytearray(f.read())
    j = b.find(b"\x05\x80\x80\x80", b.find(b"esds"))
    n = b[j + 4]
    b[j + 5:j + 5 + n] = b"\xf8" + b"\x00" * (n - 1)
    with open(mp4, "wb") as f:
        f.write(bytes(b))


# ---------------------------------------------------------------- W1

def _imports(path: str, package: str) -> list:
    """Absolute dotted names every import statement of a module reaches (relative imports resolved against
    `package`, the module's own package)."""
    out = []
    for node in ast.walk(ast.parse(read(path))):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parent = ".".join(parts[:len(parts) - node.level + 1])
                base = f"{parent}.{base}" if base else parent
            out.append(base)
            out += [f"{base}.{a.name}" for a in node.names]
    return out


class W1CoreNeverImportsTheWorker(unittest.TestCase):
    def test_no_core_module_imports_numpy_pil_or_the_worker(self):
        mods = [(os.path.join(REPO, "foreman", f), "foreman") for f in os.listdir(os.path.join(REPO, "foreman"))
                if f.endswith(".py")]
        mods += [(os.path.join(REPO, "foreman", "adapters", f), "foreman.adapters")
                 for f in os.listdir(os.path.join(REPO, "foreman", "adapters")) if f.endswith(".py")]
        self.assertGreater(len(mods), 20)
        bad = [f"{os.path.relpath(path, REPO)}: {name}" for path, pkg in mods for name in _imports(path, pkg)
               if name.split(".")[0] in ("numpy", "PIL") or name == "foreman.h3" or name.startswith("foreman.h3.")]
        self.assertEqual(bad, [])

    def test_the_scan_sees_relative_imports(self):
        tmp = tempfile.mkdtemp(prefix="foreman-h3-")
        try:
            path = os.path.join(tmp, "m.py")
            with open(path, "w") as f:
                f.write("from . import h3\nfrom ..h3.edit import run\nimport numpy as np\nfrom PIL import Image\n")
            names = _imports(path, "foreman.adapters")
            self.assertIn("foreman.adapters.h3", names)
            self.assertIn("foreman.h3.edit", names)
            self.assertIn("numpy", names)
            self.assertIn("PIL.Image", names)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_worker_parses_as_python_39_with_postponed_annotations(self):
        pkg = os.path.join(REPO, "foreman", "h3")
        for f in sorted(os.listdir(pkg)):
            if not f.endswith(".py") or f == "__init__.py":
                continue
            src = read(os.path.join(pkg, f))
            ast.parse(src, feature_version=(3, 9))
            self.assertNotRegex(src, r"(?m)^\s*match\s+\S.*:\s*$", f)
            self.assertIn("from __future__ import annotations", src, f)

    def test_the_entry_point_and_ffmpeg_helpers_are_stdlib(self):
        """worker.py must start (and report a missing numpy as infrastructure) under any Python."""
        for f in ("ff.py", "worker.py"):
            names = _imports(os.path.join(REPO, "foreman", "h3", f), "foreman.h3")
            self.assertFalse([n for n in names if n.split(".")[0] in ("numpy", "PIL")], f)


# ---------------------------------------------------------------- W2

def _words(spec: str, beat: int = 0, line=None):
    """'word s-e word s-e ...' -> word dicts."""
    p = spec.split()
    out = [{"text": w, "start": float(r.split("-")[0]), "end": float(r.split("-")[1]), "beat": beat}
           for w, r in zip(p[::2], p[1::2])]
    if line is not None:
        for w in out:
            w["line"] = line
    return out


class W2CaptionsAndPublishFiles(unittest.TestCase):
    def setUp(self):
        from foreman.h3 import captions
        self.cap = captions
        self.measure = lambda text: 40.0 * len(text)
        self.tmp = tempfile.mkdtemp(prefix="foreman-h3-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_moved_beat_moves_its_first_caption_too(self):
        """B2: the first event of a beat is written half a frame early and pulled onto the cut, so it starts before
        the beat; the position goes by the words' beat, never by the event's time."""
        ws = _words("Need .1-.4 a .4-.5 form? .5-1.0", 0) + _words("Go 1.05-1.3 to 1.3-1.4 example.com 1.4-2.2", 1)
        ws = self.cap.chunk(ws, self.measure, cuts=[1.0])
        ass = os.path.join(self.tmp, "c.ass")
        ev = self.cap.write_ass(ass, ws, cuts=[1.0], pos={1: [500, 300]}, margins=(100, 180))
        beat1 = [e for e in ev if "Go" in e[2] or "example.com" in e[2]]
        self.assertTrue(beat1 and beat1[0][0] < 1.0, ev)          # pulled before the beat's first frame
        self.assertTrue(all(e[2].startswith("{\\an8\\pos(500,300)}") for e in beat1), beat1)
        self.assertFalse([e for e in ev if e not in beat1 and "pos(" in e[2]])
        style = [ln for ln in read(ass).splitlines() if ln.startswith("Style:")][0]
        self.assertTrue(style.endswith(",7,3,2,100,180,710,1"), style)

    def test_hidden_words_never_show_and_chunks_stop_at_the_hide_edge(self):
        ws = _words("Forms .1-.4 start .4-.7 at .7-.8 $9.50. .8-1.4 One 1.5-1.7 form, 1.7-2.0 checking 2.4-2.8 "
                    "included. 2.8-3.3")
        hide = [(0.75, 2.3)]
        ws = self.cap.chunk(ws, self.measure, hide=hide)
        ev = self.cap.write_ass(os.path.join(self.tmp, "h.ass"), ws, hide=hide)
        shown = " ".join(re.sub(r"\{[^}]*\}", "", t) for _, _, t in ev)
        for word in ("$9.50.", "One", "form,"):
            self.assertNotIn(word, shown)
        self.assertFalse([e for e in ev if e[0] < 2.3 and e[1] > 0.75 + 1 / 48], ev)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(ev, ev[1:])))

    def test_only_solo_words_prices_and_addresses_stand_alone(self):
        ws = _words("Go .1-.2 to .2-.3 it. .3-.5 ExampleCo. .6-1.2 Now 1.3-1.5 relax. 1.5-2.0")
        chunks = lambda solo: [w["line"] for w in self.cap.chunk(ws, lambda t: 10 * len(t), max_w=9999, solo=solo)]
        self.assertEqual(chunks(["ExampleCo"]), [0, 0, 0, 1, 2, 2])  # 'ExampleCo.' may stand alone
        self.assertEqual(chunks([]), [0, 0, 0, 1, 1, 1])              # else it is never a chunk of its own
        self.assertTrue(self.cap._solo("$9.50.") and self.cap._solo("example.com") and not self.cap._solo("form"))

    def test_align_puts_display_words_on_asr_words_without_brand_rules(self):
        asr = [{"word": "Go", "start": .5, "end": .6}, {"word": "to", "start": .6, "end": .7},
               {"word": "Example", "start": .8, "end": 1.1}, {"word": "Co", "start": 1.1, "end": 1.3},
               {"word": "nine", "start": 1.4, "end": 1.6}, {"word": "fifty.", "start": 1.6, "end": 2.0}]
        got = self.cap.align("Go to ExampleCo for $9.50.", asr, offset=10.0, head=0.4, beat=2)
        self.assertEqual([w["text"] for w in got], ["Go", "to", "ExampleCo", "for", "$9.50."])
        self.assertEqual(got[2]["start"], 10.4)                   # 'Example' 'Co' heard as two words
        self.assertAlmostEqual(got[2]["end"], 10.9, delta=0.011)   # (less what the missed 'for' borrows)
        self.assertAlmostEqual(got[4]["start"], 11.0, delta=0.011)
        self.assertEqual(got[4]["end"], 11.6)
        self.assertTrue(got[2]["end"] <= got[3]["start"] <= got[3]["end"] <= got[4]["start"])   # 'for' interpolated
        self.assertEqual({w["beat"] for w in got}, {2})

    @needs_font
    def test_font_family_is_what_libass_matches(self):
        fam = self.cap.font_family(FONT)
        self.assertTrue(fam and fam[0].isalpha(), fam)
        if FONT.endswith("Arial Black.ttf"):
            self.assertEqual(fam, "Arial Black")

    @unittest.skipUnless(MEDIA, "numpy and Pillow are needed to import the edit")
    def test_srt_cues_never_overlap_and_keep_display_words(self):
        from foreman.h3 import edit
        ws = _words("Forms .1-.4 start .4-.7", 0, 0) + _words("$9.50. .75-1.4", 0, 1) + _words("Go 1.45-1.7", 1, 2)
        srt = edit.srt_text(ws)
        cues = re.findall(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})\n(.*)\n", srt)
        t = lambda h, m, s, ms: int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        spans = [(t(*c[:4]), t(*c[4:8]), c[8]) for c in cues]
        self.assertEqual([s[2] for s in spans], ["Forms start", "$9.50.", "Go"])
        self.assertTrue(all(a[1] < b[0] for a, b in zip(spans, spans[1:])), spans)

    @unittest.skipUnless(MEDIA, "numpy and Pillow are needed to import the edit")
    def test_the_publish_gate_belongs_to_one_file(self):
        """B10: a rebuilt ad (another hash) resets what a human filled in; the same file keeps it."""
        from foreman.h3 import edit
        path = os.path.join(self.tmp, "publish.json")
        g = edit.write_publish(path, "ad-1", "a" * 64, "final/ad.srt", "US", "Model X", None)
        self.assertEqual(g["ai_disclosure"], {"tiktok_aigc_label": None, "youtube_altered_or_synthetic": None,
                                              "meta_ai_info": None})
        self.assertEqual((g["ad_sha256"], g["srt"], g["geo"], g["attribution"]), ("a" * 64, "final/ad.srt", "US", "Model X"))
        filled = dict(g, verified_by="owner", srt_uploaded=True)
        with open(path, "w") as f:
            json.dump(filled, f)
        self.assertEqual(edit.write_publish(path, "ad-1", "a" * 64, "final/ad.srt")["verified_by"], "owner")
        again = edit.write_publish(path, "ad-1", "b" * 64, "final/ad.srt")
        self.assertIsNone(again["verified_by"])
        self.assertEqual(load(path)["ad_sha256"], "b" * 64)


# ---------------------------------------------------------------- W3

BEAT_A = [(0.40, 0.70), (0.80, 1.10), (1.20, 1.50), (1.60, 1.90), (2.00, 2.40)]


@needs_media
class W3Probe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="foreman-h3-")
        os.makedirs(os.path.join(self.tmp, "takes"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _req(self, name, **over):
        t = {"mp4": f"takes/{name}.mp4", "wav": f"takes/{name}.wav", "probe": f"takes/{name}.probe.json",
             "render": [W, H], "crop": None, "hands": None}
        t.update(over)
        return t

    def test_probe_measures_a_take_and_skips_it_once_probed(self):
        make_take(os.path.join(self.tmp, "takes", "a.mp4"), 3.0, BEAT_A, hands=(1.0, 1.5))
        code, prog, res, err = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [
            self._req("a", hands={"floor": 0.74, "limit": 30})]})
        self.assertEqual(code, 0, err)
        self.assertTrue(res["ok"] and res["takes"][0]["ok"] and not res["takes"][0]["skipped"], res)
        self.assertTrue(prog and all(0 <= p["progress"] <= 1 for p in prog))
        p = load(os.path.join(self.tmp, "takes", "a.probe.json"))
        self.assertEqual((p["frames"], p["width"], p["height"], p["has_audio"], p["problems"]), (72, W, H, True, []))
        self.assertAlmostEqual(p["duration_s"], 3.0, delta=0.05)
        self.assertIsInstance(p["color"], str)
        self.assertEqual(len(p["db10"]), 300)
        self.assertEqual(len(p["voiced_runs"]), 5, p["voiced_runs"])
        for (s, e), (a, b) in zip(p["voiced_runs"], BEAT_A):
            self.assertLess(abs(s - a), 0.06)
            self.assertLess(abs(e - b), 0.06)
        self.assertTrue(any(abs(a - 2.4) < 0.06 for a, _ in p["silent_runs"]), p["silent_runs"])   # the tail
        self.assertTrue(p["hands"] and all(0.9 <= t <= 1.6 for t in p["hands"]), p["hands"])
        with wave.open(os.path.join(self.tmp, "takes", "a.wav")) as w:
            self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (16000, 1, 2))
        code, _, res, _ = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [self._req("a")]})
        self.assertTrue(code == 0 and res["takes"][0]["skipped"], res)
        os.utime(os.path.join(self.tmp, "takes", "a.mp4"), (time.time() + 5, time.time() + 5))
        code, _, res, _ = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [self._req("a")]})
        self.assertTrue(code == 0 and not res["takes"][0]["skipped"], res)
        self.assertEqual(load(os.path.join(self.tmp, "takes", "a.probe.json"))["hands"], [])  # no hands asked

    def test_hands_the_beat_crops_away_do_not_count(self):
        """B8: the crop is centred when anchor_y is None (as the edit cuts it), and a band it leaves out is not seen."""
        from foreman.h3 import probe
        mp4 = make_take(os.path.join(self.tmp, "takes", "h.mp4"), 2.0, BEAT_A[:2], hands=(0.0, 2.0))
        self.assertTrue(probe.hands_visible(FFMPEG, mp4, W, H))
        self.assertEqual(probe.hands_visible(FFMPEG, mp4, W, H, {"scale": 1.25, "anchor_y": 0}), [])
        self.assertEqual(probe._crop_box(W, H, {"scale": 1.25, "anchor_y": None}), (14, 130, 26, 230))
        self.assertTrue(probe.hands_visible(FFMPEG, mp4, W, H, {"scale": 1.25, "anchor_y": None}))

    def test_audio_that_does_not_decode_fails_the_take_once(self):
        """An audio stream ffmpeg cannot decode (a broken AAC config) leaves no WAV: the probe's problem is final
        and the take is not probed again (the core judges it on that and retakes it)."""
        mp4 = make_take(os.path.join(self.tmp, "takes", "bad.mp4"), 1.5, BEAT_A[:2])
        corrupt_audio_config(mp4)
        code, _, res, err = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [self._req("bad")]})
        self.assertEqual(code, 0, err)
        p = load(os.path.join(self.tmp, "takes", "bad.probe.json"))
        self.assertTrue(p["has_audio"])
        self.assertTrue(p["problems"] and p["problems"][0].startswith("audio does not decode: "), p["problems"])
        self.assertNotIn(self.tmp, p["problems"][0], "ffmpeg's last words, not its temp paths")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "takes", "bad.wav")))
        code, _, res, _ = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [self._req("bad")]})
        self.assertTrue(code == 0 and res["takes"][0]["skipped"] and not res["takes"][0]["ok"], res)

    def test_problems_name_what_is_wrong_with_a_take(self):
        make_take(os.path.join(self.tmp, "takes", "big.mp4"), 1.5, BEAT_A[:2], size=(160, 256))
        make_take(os.path.join(self.tmp, "takes", "mute.mp4"), 1.5, BEAT_A[:2], audio=False)
        make_take(os.path.join(self.tmp, "takes", "quiet.mp4"), 1.5, BEAT_A[:2], silent=True)
        code, _, res, err = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [
            self._req("big"), self._req("mute"), self._req("quiet")]})
        self.assertEqual(code, 0, err)
        big, mute, quiet = (load(os.path.join(self.tmp, "takes", f"{n}.probe.json")) for n in ("big", "mute", "quiet"))
        self.assertEqual(big["problems"], [f"size 160x256 is not the render size {W}x{H}"])
        self.assertEqual(mute["problems"], ["no audio"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "takes", "mute.wav")))
        self.assertTrue(quiet["problems"] and quiet["problems"][0].startswith("silent"), quiet["problems"])
        self.assertEqual([t["ok"] for t in res["takes"]], [False, False, False])
        with open(os.path.join(self.tmp, "takes", "junk.mp4"), "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
        code, _, res, err = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [self._req("junk")]})
        self.assertEqual(code, 0, err)                              # a broken take is a failed take, not a crash
        junk = load(os.path.join(self.tmp, "takes", "junk.probe.json"))
        self.assertIn("no video stream", junk["problems"])
        self.assertFalse(junk["ok"])


# ---------------------------------------------------------------- W4 / W5: the edit

LINE_A = "Need a form filed today?"
LINE_B = "Upload the file and we print it today."
# beat B: a digital-silence pause under the screen insert (removed), a -43 dBFS hum under it (kept: not silent
# by energy), and a long silence before the end card's word (closed: the card starts at the pause)
BEAT_B = [(0.40, 0.70), (0.80, 1.00), (1.10, 1.40), (2.00, 2.30), (2.90, 3.20), (3.30, 3.60), (4.20, 4.50), (4.60, 5.00)]
HUM_B = [(2.30, 2.90)]


def _display(line, spans):
    return [{"text": t, "start": s, "end": e} for t, (s, e) in zip(line.split(), spans)]


class _StubGraphics:
    """Cards and the headline drawn fast: moving test pictures at the output size, bt709, 24 fps."""
    calls = []
    W, H = 1080, 1920

    class LayoutError(ValueError):
        pass

    @classmethod
    def _clip(cls, n, out, ffmpeg, kind, **kw):
        cls.calls.append((kind, n, kw))
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=24",
                        "-frames:v", str(n), "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                        "-video_track_timescale", "12288", "-an", out], check=True)
        return out

    @classmethod
    def screen_insert(cls, capture, n, out, ffmpeg, brand, fonts, fps=24, tap_at=None):
        return cls._clip(n, out, ffmpeg, "screen", tap_at=tap_at, png=capture["png"])

    @classmethod
    def price_insert(cls, card, n, out, ffmpeg, brand, fonts, fps=24, pulse_at=None, underline_at=None):
        return cls._clip(n, out, ffmpeg, "price", pulse_at=pulse_at, underline_at=underline_at)

    @classmethod
    def endcard(cls, card, n, out, ffmpeg, brand, fonts, fps=24, pulse_at=None):
        return cls._clip(n, out, ffmpeg, "endcard", pulse_at=pulse_at, url=card["url"])

    @classmethod
    def headline(cls, text, out_png, brand, fonts, size=76):
        from PIL import Image, ImageDraw
        cls.calls.append(("headline", text, {}))
        im = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
        ImageDraw.Draw(im).rectangle([140, 280, 860, 420], fill=(229, 50, 45, 255))
        im.save(out_png)
        return out_png


def edit_request(tmp: str, **over) -> dict:
    """A two-beat edit request over takes/a.mp4 and takes/b.mp4 (already probed)."""
    caps = os.path.join(tmp, "assets", "upload.png")
    req = {
        "ffmpeg": FFMPEG, "fps": 24, "out": [1080, 1920], "render": [W, H], "ad_id": "ad-test",
        "fonts": {"caption": FONT, "bold": FONT, "display": FONT, "logo": None},
        "brand": {"name": "ExampleCo", "bg": "#0B1220", "accent": "#2563EB", "accent_pressed": "#1D4ED8",
                  "highlight": "#FFD700", "text": "#FFFFFF", "muted": "#E5E7EB", "logo_png": None},
        "captions": {"size": 92, "margin_v": 710, "margins": [100, 180], "solo": ["example.com", "$9.50"]},
        "headline": "File a form from your phone",
        "music": "music/bed.wav", "music_under": 14.0, "bed_lift": 5.0, "lufs": -14.0, "tp": -1.5, "safe": False,
        "captures": {"upload": {"png": caps, "crop": [0, 0, 60, 40], "paint": [], "tap": [10, 10, 30, 20],
                                "tap_color": "#1D4ED8"}},
        "price_card": None,
        "endcard": {"lead": "File it from", "big": "$9.50", "lines": ["One form, everything included"],
                    "footnote": [], "url": "example.com"},
        "stills": {},
        "beats": [
            {"id": "b0", "k": 0, "take": "takes/a.mp4", "wav16": "takes/a.wav", "line": LINE_A,
             "words": _display(LINE_A, BEAT_A), "span": [0.40, 2.40, None], "tail": 0.05, "hold": 0.0,
             "crop": None, "caption_pos": None, "headline": True,
             "cutaways": [{"kind": "punch", "at_i": 2, "until_i": None, "scale": 1.2, "anchor_y": None,
                           "captions": True}]},
            {"id": "b1", "k": 1, "take": "takes/b.mp4", "wav16": "takes/b.wav", "line": LINE_B,
             "words": _display(LINE_B, BEAT_B), "span": [0.40, 5.00, None], "tail": None, "hold": 0.5,
             "crop": {"scale": 1.1, "anchor_y": None}, "caption_pos": [500, 300], "headline": False,
             "cutaways": [{"kind": "screen", "capture": "upload", "at_i": 1, "until_i": 5, "tap_i": 4,
                           "captions": True, "close_pauses": True},
                          {"kind": "endcard", "at_i": 6, "until_i": None, "pulse_i": 7, "captions": False,
                           "close_gap": True, "check_anchor": True}]}],
        "outputs": {"mp4": "final/ad.mp4", "video": "final/video.mp4", "audio": "final/audio.wav",
                    "voice": "final/voice.wav", "ass": "final/captions.ass", "srt": "final/ad.srt",
                    "timeline": "final/timeline.json", "beats_dir": "final/beats", "segments_dir": "final/segments",
                    "publish": "final/publish.json"},
        "encode": {"crf": 16, "maxrate": "16M", "bufsize": "32M", "audio_kbps": 192, "max_mb": 40.0},
        "publish": {"geo": "US", "attribution": "Model X", "note": None}}
    req.update(over)
    return req


def make_plan_inputs(tmp: str) -> None:
    os.makedirs(os.path.join(tmp, "takes"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "assets"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "music"), exist_ok=True)
    make_take(os.path.join(tmp, "takes", "a.mp4"), 2.9, BEAT_A)
    make_take(os.path.join(tmp, "takes", "b.mp4"), 5.4, BEAT_B, hum=HUM_B)
    with open(os.path.join(tmp, "assets", "upload.png"), "wb") as f:
        from foreman import media
        f.write(media.encode_png(4, 4, bytes(48)))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "aevalsrc='0.2*sin(2*PI*220*t)*(0.6+0.4*sin(2*PI*2*t))|0.2*sin(2*PI*330*t)':s=48000:d=16",
                    "-c:a", "pcm_s16le", os.path.join(tmp, "music", "bed.wav")], check=True)
    code, _, res, err = worker(tmp, "probe", {"ffmpeg": FFMPEG, "takes": [
        {"mp4": f"takes/{n}.mp4", "wav": f"takes/{n}.wav", "probe": f"takes/{n}.probe.json", "render": [W, H],
         "crop": None, "hands": None} for n in ("a", "b")]})
    assert code == 0 and res["ok"], (code, res, err)


@needs_media
@needs_font
class W4Edit(unittest.TestCase):
    """One edit run, read from every side."""

    @classmethod
    def setUpClass(cls):
        from unittest import mock
        from foreman.h3 import edit
        cls.tmp = tempfile.mkdtemp(prefix="foreman-h3-")
        make_plan_inputs(cls.tmp)
        _StubGraphics.calls = []
        cls.req = edit_request(cls.tmp)
        cls.progress = []
        cwd = os.getcwd()
        os.chdir(cls.tmp)
        try:
            with mock.patch.object(edit, "_graphics", lambda: _StubGraphics):
                cls.res = edit.run(cls.req, lambda p, note="": cls.progress.append((p, note)))
        finally:
            os.chdir(cwd)
        cls.tl = load(os.path.join(cls.tmp, "final", "timeline.json"))
        cls.ass = read(os.path.join(cls.tmp, "final", "captions.ass"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def f(self, rel):
        return os.path.join(self.tmp, rel)

    def test_the_output_is_frame_exact_bt709_and_measured(self):
        from foreman.h3 import ff
        tl = self.tl
        self.assertEqual(tl["frames"], sum(b["frames"] for b in tl["beats"]))
        self.assertEqual(tl["frames"], sum(s["frames"] for s in tl["segments"]))
        got, dur = ff.decode_count(FFMPEG, self.f("final/ad.mp4"))
        self.assertEqual(got, tl["frames"])
        self.assertAlmostEqual(tl["total"], tl["frames"] / 24)
        self.assertEqual(tl["checks"]["frames"], {"got": got, "want": tl["frames"], "ok": True})
        info = ff.probe(FFMPEG, self.f("final/ad.mp4"))
        self.assertEqual((info["width"], info["height"], info["fps"]), (1080, 1920, 24.0))
        self.assertIn("bt709", info["video_line"])
        self.assertTrue(info["has_audio"])
        self.assertEqual(tl["checks"]["freezes"], [])
        self.assertTrue(self.res["checks_ok"], self.res["checks"])
        self.assertEqual(self.res["frames"], tl["frames"])
        self.assertTrue(self.progress and self.progress[-1][0] >= 0.9)

    def test_segments_tile_the_timeline_and_concat_lines_up(self):
        segs = self.tl["segments"]
        self.assertEqual(segs[0]["frame"], 0)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(a["frame"] + a["frames"], b["frame"])
        self.assertEqual([s["kind"] for s in segs], ["talk", "punch", "talk", "screen", "talk", "endcard"])
        for s in segs:
            self.assertTrue(os.path.isfile(self.f(s["path"])), s)
        punch = segs[1]
        self.assertEqual(self.tl["checks"]["punches"], [punch["start"]])
        self.assertTrue(all(p["seen"] for p in self.tl["checks"]["punch_cuts"]), self.tl["checks"]["punch_cuts"])

    def test_removals_are_silence_under_graphic_cutaways_only(self):
        """B3 and the energy rule: the digital-silence pause under the screen insert goes, the -43 dBFS hum stays,
        the long silence before the end card's word goes and the card starts where that pause began."""
        rem = self.tl["checks"]["removals"]
        cut = [r for r in rem if r["ok"]]
        kept = [r for r in rem if not r["ok"]]
        self.assertEqual(len(cut), 2, rem)
        self.assertTrue(all(r["why"].startswith("silence by energy") for r in cut))
        self.assertTrue(1.40 <= cut[0]["take_s"][0] < cut[0]["take_s"][1] <= 2.00, cut)
        self.assertTrue(3.60 <= cut[1]["take_s"][0] < cut[1]["take_s"][1] <= 4.20, cut)
        self.assertEqual(len(kept), 1, rem)
        self.assertTrue(2.30 <= kept[0]["take_s"][0] and kept[0]["take_s"][1] <= 2.90 and "kept" in kept[0]["why"])
        b1 = self.tl["beats"][1]
        screen, card = b1["cutaways"]
        removed_frames = sum(r["frames"] for r in cut)
        self.assertEqual(b1["frames"], self.tl["notes"]["b1"]["frames"])
        take_frames = card["take_frames"][1]
        self.assertEqual(b1["frames"], take_frames - removed_frames)
        self.assertEqual(screen["frames"], screen["take_frames"][1] - screen["take_frames"][0] - cut[0]["frames"])
        self.assertEqual(card["take_frames"][0], round((cut[1]["take_s"][0] - self.tl["notes"]["b1"]["head"]) * 24))
        self.assertEqual(self.tl["beats"][0]["cutaways"][0]["frames"],
                         self.tl["beats"][0]["cutaways"][0]["take_frames"][1] - self.tl["beats"][0]["cutaways"][0]["take_frames"][0])
        with wave.open(self.f("final/beats/b1.wav")) as w:       # the voice lost exactly the removed frames
            self.assertEqual(w.getnframes(), b1["frames"] * 2000)
        with wave.open(self.f("final/voice.wav")) as w:
            self.assertEqual(w.getnframes(), self.tl["frames"] * 2000)
        with wave.open(self.f("final/audio.wav")) as w:
            self.assertEqual((w.getnframes(), w.getframerate()), (self.tl["frames"] * 2000, 48000))
        with wave.open(self.f("final/beats/b1-16k.wav")) as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (16000, 1))

    def test_event_words_land_inside_their_cutaways(self):
        calls = {k: kw for k, n, kw in _StubGraphics.calls if k in ("screen", "endcard")}
        n = {k: n for k, n, kw in _StubGraphics.calls if k in ("screen", "endcard")}
        self.assertTrue(0 <= calls["screen"]["tap_at"] < n["screen"] / 24)
        self.assertTrue(0 <= calls["endcard"]["pulse_at"] < n["endcard"] / 24)
        self.assertEqual(calls["endcard"]["url"], "example.com")
        self.assertIn(("headline", "File a form from your phone", {}), _StubGraphics.calls)

    def test_captions_hide_under_the_card_and_move_with_their_beat(self):
        hide = self.tl["hide"]
        card = self.tl["beats"][1]["cutaways"][1]
        self.assertEqual(len(hide), 1)
        self.assertAlmostEqual(hide[0][0], card["start"], places=3)
        self.assertAlmostEqual(hide[0][1], card["end"], places=3)
        ev = re.findall(r"Dialogue: 0,(\d+):(\d\d):(\d\d\.\d\d),(\d+):(\d\d):(\d\d\.\d\d),Caption,,0,0,0,,(.*)", self.ass)
        t = lambda h, m, s: int(h) * 3600 + int(m) * 60 + float(s)
        ev = [(t(*e[:3]), t(*e[3:6]), e[6]) for e in ev]
        self.assertTrue(ev)
        self.assertFalse([e for e in ev if e[0] < hide[0][1] and e[1] > hide[0][0] + 0.03], ev)
        shown = " ".join(re.sub(r"\{[^}]*\}", "", e[2]) for e in ev)
        self.assertNotIn("today.", shown.split("?")[-1])           # beat 1's hidden 'it today.'
        b1_start = self.tl["beats"][1]["start"]
        b1 = [e for e in ev if e[0] >= b1_start - 0.2]
        self.assertTrue(b1)
        self.assertTrue(all(e[2].startswith("{\\an8\\pos(500,300)}") for e in b1), b1)
        self.assertLess(min(e[0] for e in b1), b1_start)            # the first one is pulled before the beat: B2
        self.assertFalse([e for e in ev if "pos(" in e[2] and e not in b1])
        self.assertIn("Style: Caption,", self.ass)
        self.assertRegex(self.ass, r"Style: Caption,[^\n]*,7,3,2,100,180,710,1")
        self.assertEqual(self.tl["captions"]["events"], len(ev))
        words = self.tl["words"]
        self.assertEqual([w["text"] for w in words], LINE_A.split() + LINE_B.split())
        self.assertEqual({w["beat"] for w in words}, {0, 1})
        self.assertTrue(all("line" in w for w in words))

    def test_publish_files(self):
        pub = load(self.f("final/publish.json"))
        from foreman.h3 import edit
        self.assertEqual(pub["ad_sha256"], edit.sha256(self.f("final/ad.mp4")))
        self.assertEqual((pub["ad_id"], pub["geo"], pub["srt"], pub["attribution"]),
                         ("ad-test", "US", "final/ad.srt", "Model X"))
        self.assertIsNone(pub["ai_disclosure"]["tiktok_aigc_label"])
        srt = read(self.f("final/ad.srt"))
        self.assertIn("today.", srt)                                   # hidden words are in the SRT
        self.assertTrue(os.path.isfile(self.f("final/video.mp4")))
        left = sorted(os.listdir(self.f("final/segments")))
        self.assertFalse([n for n in left if n.endswith(("-full.mp4", "-punch.mp4", "-take.wav", ".txt", ".png"))
                          or ".tmp-" in n], left)
        self.assertFalse([n for n in os.listdir(self.f("final")) if ".tmp-" in n])

    def test_safe_mode_removes_nothing(self):
        """The re-edit after a final line failed: no pause is closed and no gap either."""
        from foreman.h3 import edit
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            beats = edit.parse(edit_request(self.tmp))["beats"]
            normal, safe = edit.plan_beat(beats[1]), edit.plan_beat(beats[1], safe=True)
        finally:
            os.chdir(cwd)
        self.assertEqual(len(normal["removals"]), 2)
        self.assertEqual((safe["removals"], safe["checks"]), ([], []))
        self.assertEqual(safe["n"], safe["n_take"])
        self.assertEqual(safe["n"] - normal["n"], sum(f1 - f0 for f0, f1, _ in normal["removals"]))
        self.assertEqual(normal["specs"][1]["ta"], normal["removals"][1][0])   # the card starts at the closed gap
        self.assertGreater(safe["specs"][1]["ta"], normal["specs"][1]["ta"])

    def test_the_mix_is_mastered(self):
        m = self.tl["mix"]
        self.assertAlmostEqual(m["I"], -14.0, delta=0.5)
        self.assertLessEqual(m["TP"], -1.0)
        self.assertEqual(m["music_under"], 14.0)
        self.assertEqual(m["samples"], self.tl["frames"] * 2000)


@needs_media
@needs_font
class W5Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="foreman-h3-")
        make_plan_inputs(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _no_final(self):
        return not os.path.exists(os.path.join(self.tmp, "final", "ad.mp4"))

    def test_an_unknown_cutaway_kind_is_refused(self):
        """B4: never silently left out."""
        req = edit_request(self.tmp)
        req["beats"][0]["cutaways"] = [{"kind": "zoom", "at_i": 2}]
        code, _, res, err = worker(self.tmp, "edit", req)
        self.assertEqual((code, res["ok"], res["class"]), (2, False, "content"), err)
        self.assertIn("unknown cutaway kind 'zoom'", res["error"])
        self.assertTrue(self._no_final())

    def test_an_event_word_outside_its_cutaway_is_a_content_failure(self):
        """B9: the tap word must be heard inside the insert, not before it."""
        req = edit_request(self.tmp)
        req["beats"][1]["cutaways"][0]["tap_i"] = 0
        code, prog, res, err = worker(self.tmp, "edit", req)
        self.assertEqual((code, res["class"]), (2, "content"), err)
        self.assertIn("outside the cutaway", res["error"])
        self.assertIn("'Upload'", res["error"])
        self.assertTrue(self._no_final())

    def test_short_music_and_a_missing_ffmpeg_are_infrastructure(self):
        short = os.path.join(self.tmp, "music", "short.wav")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:d=2:sample_rate=48000",
                        short], check=True)
        code, _, res, err = worker(self.tmp, "edit", edit_request(self.tmp, music="music/short.wav"))
        self.assertEqual((code, res["class"]), (3, "infra"), err)
        self.assertIn("music", res["error"])
        code, _, res, _ = worker(self.tmp, "edit", edit_request(self.tmp, ffmpeg=os.path.join(self.tmp, "no-ffmpeg")))
        self.assertEqual((code, res["class"]), (3, "infra"))
        self.assertIn("ffmpeg not found", res["error"])
        code, _, res, _ = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [
            {"mp4": "takes/none.mp4", "wav": "takes/none.wav", "probe": "takes/none.probe.json", "render": [W, H]}]})
        self.assertEqual((code, res["class"]), (3, "infra"))
        self.assertTrue(self._no_final())

    def test_no_music_masters_the_voice_and_a_size_cap_re_encodes(self):
        """One beat, no music: the voice alone at -14 LUFS; over max_mb the encode drops to an ABR that fits."""
        os.makedirs(os.path.join(self.tmp, "solo"), exist_ok=True)
        req = edit_request(self.tmp, music=None, headline=None)
        req["beats"] = req["beats"][:1]
        req["beats"][0]["headline"] = False
        req["outputs"] = {k: v.replace("final/", "solo/") for k, v in req["outputs"].items()}
        req["encode"]["max_mb"] = 0.5
        code, _, res, err = worker(self.tmp, "edit", req)
        self.assertEqual(code, 0, err)
        self.assertEqual((res["encode"]["mode"], len(res["encode"]["attempts"])), ("abr", 2), res["encode"])
        self.assertLessEqual(os.path.getsize(os.path.join(self.tmp, "solo", "ad.mp4")), 0.5e6)
        self.assertGreater(res["encode"]["attempts"][0]["bytes"], 0.5e6)
        tl = load(os.path.join(self.tmp, "solo", "timeline.json"))
        self.assertIsNone(tl["mix"]["music"])
        self.assertAlmostEqual(tl["mix"]["I"], -14.0, delta=0.5)
        self.assertTrue(tl["checks"]["frames"]["ok"])
        req["encode"]["max_mb"] = 0.05                             # no bitrate fits: deleted, infrastructure
        code, _, res, _ = worker(self.tmp, "edit", req)
        self.assertEqual((code, res["class"]), (3, "infra"))
        self.assertIn("over the 0.05 MB cap", res["error"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "solo", "ad.mp4")))

    def test_usage_and_unreadable_requests(self):
        p = subprocess.run([sys.executable, "-I", WORKER, "render", "x.json"], cwd=self.tmp, capture_output=True, text=True)
        self.assertNotIn(p.returncode, (0, 2, 3))
        self.assertFalse(json.loads(p.stdout.splitlines()[-1])["ok"])
        p = subprocess.run([sys.executable, "-I", WORKER, "edit", "missing.json"], cwd=self.tmp, capture_output=True,
                           text=True)
        self.assertEqual(p.returncode, 3)
        with open(os.path.join(self.tmp, "bad.json"), "w") as f:
            f.write("{not json")
        p = subprocess.run([sys.executable, "-I", WORKER, "edit", "bad.json"], cwd=self.tmp, capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)

    def test_sigterm_leaves_no_ffmpeg_behind(self):
        req = edit_request(self.tmp)
        path = os.path.join(self.tmp, "term-req.json")
        with open(path, "w") as f:
            json.dump(req, f)
        p = subprocess.Popen([sys.executable, "-I", WORKER, "edit", path], cwd=self.tmp, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        try:
            first = json.loads(p.stdout.readline())                # "planned": rendering starts now
            self.assertIn("progress", first)
            time.sleep(0.3)
            p.send_signal(signal.SIGTERM)
            out, _ = p.communicate(timeout=60)
        finally:
            if p.poll() is None:
                p.kill()
        self.assertEqual(p.returncode, 128 + signal.SIGTERM)
        self.assertEqual(json.loads(out.splitlines()[-1])["error"], "terminated")
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.killpg(p.pid, 0)                                # anything left in the worker's group?
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail("a process of the worker's group outlived it")
        leftovers = [n for d, _, fs in os.walk(os.path.join(self.tmp, "final")) for n in fs if ".tmp-" in n]
        self.assertEqual(leftovers, [])


# ---------------------------------------------------------------- W6: framing, word ends, fonts, a dead parent

def _skin(rgb) -> int:
    import numpy as np
    a = rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return int(((r > 120) & (r > g + 25) & (g > b) & (r - b > 45)).sum())


def _frames(mp4: str, size=(108, 192)):
    import numpy as np
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", mp4, "-vf", f"scale={size[0]}:{size[1]}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size[1], size[0], 3)


class W6Window(unittest.TestCase):
    def test_a_punch_in_nests_inside_the_beats_crop(self):
        from foreman.h3 import ff
        self.assertEqual(ff.window(576, 1024), (0, 0, 576, 1024))
        self.assertEqual(ff.window(576, 1024, 1.2), (48, 85, 480, 854), "centred, both sizes even")
        beat = ff.window(576, 1024, 1.205, 0)                       # rows 0..850: the hands below are cropped away
        punch = ff.window(576, 1024, 1.2, None, beat)
        self.assertEqual(beat, (49, 0, 478, 850))
        x, y, w, h = punch
        self.assertTrue(beat[0] <= x and x + w <= beat[0] + beat[2] and beat[1] <= y and y + h <= beat[1] + beat[3])
        self.assertLess(w, beat[2], "a punch-in is closer than the beat, even when the beat is cropped")
        self.assertEqual(ff.window(576, 1024, 1.2, 900, beat)[1], 850 - 708, "its anchor is clamped inside the beat")
        self.assertEqual(ff.window(576, 1024, 1.2, 40), (48, 40, 480, 854), "without a beat crop: as before")


@needs_media
class W7Priority(unittest.TestCase):

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("taskpolicy"), "macOS background band")
    def test_the_worker_leaves_launchds_background_band(self):
        """gpud is a launchd Background job: its children, the worker's ffmpeg included, inherit that band."""
        code = ("import os, sys; sys.path.insert(0, %r); from foreman.h3 import worker; "
                "before = os.getpriority(4, 0); worker._leave_background(); print(before, os.getpriority(4, 0))"
                % REPO)
        r = subprocess.run(["taskpolicy", "-b", sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual(r.stdout.split(), ["1", "0"], r.stderr)


class W6Regressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="foreman-h3-")
        os.makedirs(os.path.join(self.tmp, "takes"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def f(self, rel):
        return os.path.join(self.tmp, rel)

    @needs_font
    def test_the_punch_in_shows_no_rows_the_beats_crop_and_the_probe_leave_out(self):
        """A scene crop that keeps hands out of frame (anchor_y 0) and a punch-in on its beat: the probe scans what
        the crop keeps (no hands), and the punch-in must show no more than that."""
        from unittest import mock
        from foreman.h3 import edit
        make_take(self.f("takes/a.mp4"), 2.9, BEAT_A, hands=(0.0, 2.9), hands_y=0.86)
        crop = {"scale": 1.2, "anchor_y": 0}
        code, _, res, err = worker(self.tmp, "probe", {"ffmpeg": FFMPEG, "takes": [
            {"mp4": "takes/a.mp4", "wav": "takes/a.wav", "probe": "takes/a.probe.json", "render": [W, H],
             "crop": crop, "hands": {"floor": 0.74, "limit": 30}}]})
        self.assertEqual(code, 0, err)
        self.assertEqual(load(self.f("takes/a.probe.json"))["hands"], [], "the beat's crop keeps the hands out")
        req = edit_request(self.tmp, music=None, headline=None, captures={})
        req["beats"] = req["beats"][:1]
        req["beats"][0].update(crop=crop, headline=False)
        from foreman.h3 import captions
        burned, real = [], captions.ass_vf

        def spy(ass, fontsdir):                                  # the faces libass is given for the captions
            burned.append(sorted(os.listdir(fontsdir)))
            return real(ass, fontsdir)
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            with mock.patch.object(edit, "_graphics", lambda: _StubGraphics), mock.patch.object(captions, "ass_vf", spy):
                edit.run(req)
        finally:
            os.chdir(cwd)
        self.assertEqual(burned, [[os.path.basename(FONT)]], "the caption font alone")
        self.assertFalse(os.path.exists(self.f("final/segments/fonts")))
        tl = load(self.f("final/timeline.json"))
        punch = next(s for s in tl["segments"] if s["kind"] == "punch")
        frames = _frames(self.f(punch["path"]))
        self.assertEqual(max(_skin(fr) for fr in frames), 0, "the punch-in shows the cropped-away hands")
        talk = next(s for s in tl["segments"] if s["kind"] == "talk")
        self.assertEqual(max(_skin(fr) for fr in _frames(self.f(talk["path"]))), 0)

    def test_the_last_word_never_outlasts_its_beat(self):
        """A take cut before the next heard word: the line's last word ends by the line's acoustic end (not at the
        next word), and no word time leaves its beat (its caption would show over the next beat's picture)."""
        from foreman.h3 import audio, edit
        wav = tone_wav(self.f("takes/a16.wav"), 3.0, [(0.40, 0.70), (0.80, 1.00), (1.05, 1.80)], rate=16000)
        words = [{"text": "Need", "start": 0.40, "end": 0.70}, {"text": "help?", "start": 0.80, "end": 1.00}]
        ws = edit.refine_ends(words, wav, 1.0, 1.25)
        self.assertLessEqual(ws[-1]["end"], 1.0, "the next word's sound is not the line's")
        b = {"id": "b0", "wav16": wav, "span": [0.40, 1.00, 1.25], "tail": None, "hold": 0.0, "words": words,
             "cutaways": []}
        p = edit.plan_beat(b)
        self.assertLess(p["stop"], 1.25)
        self.assertTrue(all(w["e"] <= p["n"] / 24 + 1e-9 and w["s"] <= w["e"] for w in p["shifted"]), (p["shifted"],
                                                                                                         p["n"]))

    def test_the_captions_burn_in_the_face_they_were_measured_with(self):
        """libass loads every face in its fonts directory and matches family + regular weight: the caption font is
        given alone, so Arial Bold.ttf is not drawn as the Arial.ttf beside it."""
        from foreman.h3 import captions, edit
        bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        d = edit._caption_fonts(FONT or bold, self.f("final/segments/fonts")) if FONT else None
        if d:
            self.assertEqual(os.listdir(d), [os.path.basename(FONT)])
        if not (os.path.isfile(bold) and os.path.isfile(os.path.join(os.path.dirname(bold), "Arial.ttf"))):
            self.skipTest("needs Arial Bold.ttf beside Arial.ttf")
        words = [{"text": t, "start": s, "end": s + 0.3, "beat": 0}
                 for t, s in zip("File your form today".split(), (0.0, 0.4, 0.8, 1.2))]
        ws = captions.chunk(words, captions.measurer(bold, 92), max_w=800)
        ass = self.f("c.ass")
        captions.write_ass(ass, ws, font_family=captions.font_family(bold))

        def ink(fontsdir):
            raw = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=24:d=2",
                                  "-vf", captions.ass_vf(ass, fontsdir) + ",select=eq(n\\,30)", "-frames:v", "1",
                                  "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
            return sum(1 for v in raw if v > 128)
        alone = ink(edit._caption_fonts(bold, self.f("fonts-bold")))
        self.assertGreater(alone, 1.2 * ink(os.path.dirname(bold)), "the bold face was drawn")

    def test_the_worker_stops_when_its_parent_dies(self):
        """A media worker leads a process group of its own (a cancel stops its ffmpeg), so launchd ending gpud's
        group does not reach it: it watches its parent, and stops (its ffmpeg too) when the parent is gone."""
        fake = self.f("slow-ffmpeg")
        with open(fake, "w") as f:
            f.write(f"#!/bin/sh\necho $$ > {self.f('ffmpeg.pid')}\nexec sleep 60\n")
        os.chmod(fake, 0o755)
        with open(self.f("takes/a.mp4"), "wb") as f:
            f.write(b"\x00" * 64)
        with open(self.f("probe-req.json"), "w") as f:
            json.dump({"ffmpeg": fake, "takes": [{"mp4": "takes/a.mp4", "wav": "takes/a.wav", "probe": "takes/a.probe.json",
                                                  "render": [W, H]}]}, f)
        parent = subprocess.Popen([sys.executable, "-c", (
            "import subprocess, sys, time\n"
            f"p = subprocess.Popen([sys.executable, '-I', {WORKER!r}, 'probe', 'probe-req.json'], cwd={self.tmp!r}, "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            f"open({self.f('worker.pid')!r}, 'w').write(str(p.pid))\n"
            "time.sleep(120)\n")], start_new_session=True)
        try:
            self.assertTrue(_wait(lambda: os.path.isfile(self.f("ffmpeg.pid")) and os.path.getsize(self.f("ffmpeg.pid"))),
                            "the worker never started its ffmpeg")
            pids = [int(read(self.f(n)).strip()) for n in ("worker.pid", "ffmpeg.pid")]
            os.killpg(parent.pid, signal.SIGKILL)               # launchd ends the job: its process group
            parent.wait()
            self.assertTrue(_wait(lambda: not any(_alive(pid) for pid in pids), 10.0),
                            f"the worker or its ffmpeg outlived their parent: {[p for p in pids if _alive(p)]}")
        finally:
            for pid in ([parent.pid] + [int(read(self.f(n)).strip()) for n in ("worker.pid", "ffmpeg.pid")
                                        if os.path.isfile(self.f(n)) and read(self.f(n)).strip()]):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


def _wait(cond, timeout: float = 15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return bool(cond())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                                          # a zombie waiting for its new parent to reap it
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    except OSError:
        return True
    return bool(out) and not out.startswith("Z")


if __name__ == "__main__":
    unittest.main()
