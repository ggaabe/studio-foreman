"""The h3 talk chain in the ad pipeline (foreman/ad_pipeline.py and friends), on synthetic files.

  S1  the facts step's h3 preflight: the media python, the graph (outputs, placeholders, license),
      the commercial gate, the reference voice, the stills, the capture boxes, the cards against
      the facts; each failure with its class (infrastructure or content)
  S2  the dispatcher round by round: takes (ComfyUI ops), probe (ad_probe), hear (two transcriptions),
      the judge, retakes (seed, spelling), music, the edit, the final gate (a safe re-edit), finish
  S3  the speech gate's record: gate_summary, gate_note, the review's note, the manifest
  S4  retention and the owner's view: pruning a take's sidecars, progress() stages
  S5  the media worker's runner (exit codes and classes) and LocalContext.run(group=True)
  S6  ComfyUI: its environment and a refusal that names a missing model file or node
  S7  characters (a reference voice for any engine) and the generic examples

The media worker is never run here: `ads_options.media_python` points at a fake interpreter (a
small script written by the test) that answers the worker's protocol and writes the files a real
probe or edit would. Everything is generic (ExampleCo, example.com, $9.50).
"""
from __future__ import annotations

import copy
import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import unittest

from foreman import ad_pipeline, ads, characters, h3core, paths, studio
from foreman.adapters import comfyui
from foreman.adapters.local import ContentFailure, InfraFailure, LocalEngine
from foreman.ads import AdError
from foreman.engine import RunSpec
from foreman.templates import load_templates, parse_template

try:
    from ad_fixtures import FFMPEG, TmpCase, ads_config, needs_ffmpeg, png, template_data, wait_for, wav
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import FFMPEG, TmpCase, ads_config, needs_ffmpeg, png, template_data, wait_for, wav

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXAMPLES = os.path.join(REPO, "examples", "ads")
EXAMPLE_GRAPH = os.path.join(EXAMPLES, "comfyui", "graphs", "h3_ref2va.example.json")

FACTS = {"service": "ExampleCo prepares and files a form for you online",
         "price": "$9.50 per form, everything included",
         "how": "Upload a document from your phone, check the details, pay, and it is prepared and filed",
         "site": "example.com"}
BRAND = {"display": "ExampleCo", "spoken": "Example Co", "respellings": ["Eggs Ample Co"]}
TAGS = ("hero", "face", "desk", "kitchen", "payoff")


def h3_block(**over) -> dict:
    b = {"presenter": {"noun": "woman", "subject": "she", "object": "her", "voice": "her own warm voice"},
         "face": "face",
         "scenes": {"desk": {"still": "desk", "framing": "Vertical medium shot from the waist up.",
                             "setting": "She sits at a bright wooden desk", "roles": ["hook", "body"]},
                    "kitchen": {"still": "kitchen", "framing": "Vertical medium close-up from the chest up.",
                                "setting": "She stands at a sunny kitchen counter", "roles": ["body", "cta"],
                                "hands_gate": True, "crop": {"scale": 1.2, "anchor_y": None}}},
         "stills": {"payoff": {"still": "payoff", "note": "a finished form in a folder, no text"}},
         "captures": {"captures/upload.png": {"id": "upload", "note": "the upload page", "crop": [0, 0, 100, 80],
                                              "tap": [10, 50, 90, 70], "tap_color": "#1d4ed8"}},
         "price_card": {"big": "$9.50", "lines": ["One form, everything included"], "capture": "captures/upload.png",
                        "crop": [0, 0, 100, 60], "underline": [[10, 10, 50, 20]]},
         "endcard": {"lead": "Get it done from", "big": "$9.50", "lines": ["Everything included"]},
         "brand": {"name": "ExampleCo", "bg": "#101010", "accent": "#2255CC"},
         "headline": True,
         "readings": {"$9.50": "nine fifty"}}
    b.update(over)
    return b


def h3_template(h3=None, **ad_over) -> dict:
    t = template_data()
    t["ad"] = dict(t["ad"], character="presenter-a", facts={"file": "assets/facts-h3.json"}, ai_label=False,
                   captures=["captures/upload.png"],
                   end_card={"price": "$9.50", "url": "example.com", "cta": "Get it done today"},
                   lexicon=[BRAND], h3=h3 if h3 is not None else h3_block())
    t["ad"].update(ad_over)
    return t


def h3_ad(h3=None, ad_id="ad-h3", commercial=False, **ad_over) -> dict:
    t = parse_template(h3_template(h3, **ad_over))
    return dict(t.ad, id=ad_id, template_id=t.id, seconds=20.0, shots=4, seed=5, brief="filed from a phone",
                length={"min": 15.0, "max": 30.0}, shot_range={"min": 4, "max": 6}, commercial=commercial)


PLAN = {"title": "Filed from your phone", "headline": "File a form from your phone",
        "beats": [
            {"line": "Need a form filed, but no time to stand in line?", "scene": "desk",
             "action": "holding a stack of papers beside her face, she asks the camera with one eyebrow raised",
             "cutaways": [{"kind": "punch", "at": "but"}]},
            {"line": "Upload your document from your phone, and ExampleCo prepares and files it.", "scene": "desk",
             "action": "holding her phone in both hands, she taps it once, then looks up and tells the camera",
             "cutaways": [{"kind": "screen", "capture": "upload", "at": "from", "until": "prepares", "tap": "ExampleCo"},
                          {"kind": "still", "still": "payoff", "at": "prepares"}]},
            {"line": "It costs $9.50, with everything included.", "scene": "kitchen",
             "action": "resting her forearms on the counter, she explains with an easy smile",
             "cutaways": [{"kind": "price", "at": "$9.50", "pulse": "everything"}]},
            {"line": "Get it done today at example.com and skip the line.", "scene": "kitchen",
             "action": "her hands folded, she says it warmly, pronouncing the web address slowly and clearly, word by "
                       "word",
             "cutaways": [{"kind": "endcard", "at": "and", "pulse": "line"}]}],
        "disclosure": "AI-generated"}

# A stand-in for ads_options.media_python: `-I -c "import numpy, PIL"` exits as control.json says;
# `-I <worker.py> probe|edit <request>` writes what the real worker would and answers its protocol.
FAKE_PYTHON = r'''#!{python}
import json, math, os, struct, subprocess, sys, wave
HERE = os.path.dirname(os.path.abspath(__file__))
try:
    ctl = json.load(open(os.path.join(HERE, "control.json")))
except (OSError, ValueError):
    ctl = {{}}
args = sys.argv[1:]
if args[:2] == ["-I", "-c"]:
    sys.stderr.write(ctl.get("import_error", ""))
    sys.exit(int(ctl.get("import_exit", 0)))
_, worker, command, req_path = args
req = json.load(open(req_path))
with open(os.path.join(HERE, "requests.jsonl"), "a") as f:
    f.write(json.dumps({{"command": command, "worker": worker, "cwd": os.getcwd(), "pgid_is_own": os.getpgrp() == os.getpid(),
                        "request": req}}) + "\n")
if ctl.get("stdout") is not None:
    print(ctl["stdout"])
    sys.exit(int(ctl.get("exit", 0)))
if ctl.get("exit"):
    print(json.dumps({{"progress": 0.1, "note": "started"}}))
    print(json.dumps({{"ok": False, "error": ctl.get("error"), "class": ctl.get("class")}}))
    sys.exit(int(ctl["exit"]))


def tone(path, seconds, rate=16000, channels=1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(i / 7.0))) * channels for i in range(n)))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


if command == "probe":
    out = []
    for t in req["takes"]:
        rec = {{"duration_s": 5.0, "frames": 124, "fps": 24.0, "width": t["render"][0], "height": t["render"][1],
               "has_audio": True, "audio_rate": 32000, "color": "tv, bt709, progressive", "db10": [-20.0],
               "voiced_runs": [], "silent_runs": [], "hands": [], "problems": []}}
        rec.update(ctl.get("probes", {{}}).get(t["mp4"], {{}}))
        if rec["has_audio"] and not rec.pop("no_wav", False):   # no_wav: an audio stream that does not decode
            tone(t["wav"], 0.2)
        rec["ok"] = not rec["problems"]
        write_json(t["probe"], rec)
        out.append({{"mp4": t["mp4"], "probe": t["probe"], "ok": rec["ok"], "problems": rec["problems"], "skipped": False}})
    print(json.dumps({{"progress": 1.0, "note": "probed"}}))
    print(json.dumps({{"ok": True, "error": None, "class": None, "takes": out}}))
    sys.exit(0)

o = req["outputs"]
W, H = req["out"]
ff = req["ffmpeg"]
TOTAL = 2.0
for b in req["beats"]:
    tone(os.path.join(o["beats_dir"], b["id"] + ".wav"), 0.5, 48000, 2)
    tone(os.path.join(o["beats_dir"], b["id"] + "-16k.wav"), 0.5)
tone(o["audio"], TOTAL, 48000, 2)
tone(o["voice"], TOTAL, 48000, 2)
os.makedirs(o["segments_dir"], exist_ok=True)
src = ["-f", "lavfi", "-i", "testsrc2=s=%dx%d:r=24:d=0.5" % (W, H), "-f", "lavfi", "-i", "testsrc=s=%dx%d:r=24:d=1.5" % (W, H)]
join = "[0:v][1:v]concat=n=2:v=1[v]"
tags = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
enc = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-r", "24"]
subprocess.run([ff, "-y", "-loglevel", "error"] + src + ["-filter_complex", join, "-map", "[v]"] + enc + tags
               + [o["video"]], check=True)
moved = [b["caption_pos"] for b in req["beats"]]   # the captions: in the bottom band, or where every beat moves them
band = "y=ih*%.4f:w=iw:h=ih*0.05" % (moved[0][1] / H) if all(moved) else "y=ih*0.58:w=iw:h=ih*0.1"
box = join[:-3] + ",drawbox=x=0:" + band + ":color=white:t=fill[v]"
subprocess.run([ff, "-y", "-loglevel", "error"] + src + ["-i", o["audio"], "-filter_complex", box, "-map", "[v]",
               "-map", "2:a"] + enc + tags + ["-c:a", "aac", "-t", "2", o["mp4"]], check=True)
for k in ("ass", "srt"):
    with open(o[k], "w") as f:
        f.write("captions\n")
words = []
n = len(req["beats"])
for i, b in enumerate(req["beats"]):
    for j, w in enumerate(b["line"].split()):
        s = i * TOTAL / n + j * 0.05
        words.append({{"text": w, "start": round(s, 3), "end": round(s + 0.04, 3), "beat": i, "line": j // 3}})
segs = [{{"path": "final/segments/b0-p0.mp4", "beat": "b0", "k": 0, "kind": "talk", "frame": 0, "frames": 12,
          "start": 0.0, "end": 0.5}},
        {{"path": "final/segments/b0-c0.mp4", "beat": "b0", "k": 0, "kind": "punch", "frame": 12, "frames": 12,
          "start": 0.5, "end": 1.0}},
        {{"path": "final/segments/b1-p0.mp4", "beat": "b1", "k": 1, "kind": "talk", "frame": 24, "frames": 12,
          "start": 1.0, "end": 1.5}},
        {{"path": "final/segments/b3-c0.mp4", "beat": "b3", "k": n - 1, "kind": "endcard", "frame": 36, "frames": 12,
          "start": 1.5, "end": 2.0}}]
write_json(o["timeline"], {{"fps": 24, "out": [W, H], "total": TOTAL, "frames": 48, "words": words, "segments": segs,
                            "hide": [[1.5, 2.0]], "cuts": [0.5, 1.0, 1.5],
                            "beats": [{{"id": b["id"], "k": b["k"], "start": 0.0, "end": 0.5, "frames": 12,
                                        "caption_pos": b["caption_pos"], "cutaways": [{{"kind": c["kind"]}}
                                                                                    for c in b["cutaways"]]}}
                                      for b in req["beats"]],
                            "mix": {{"I": -14.0, "TP": -1.6, "LRA": 5.0}}, "notes": {{}},
                            "checks": {{"removals": [{{"beat": "b1", "ok": not req["safe"]}}], "frames": {{"ok": True}},
                                        "freezes": [], "punch_cuts": [], "color": ["bt709"]}},
                            "captions": {{"events": 20, "family": "Arial Black"}}, "encode": {{"crf": 16}}}})
write_json(o["publish"], {{"ad_id": req["ad_id"], "ad_sha256": "x", "attribution": req["publish"]["attribution"]}})
print(json.dumps({{"ok": True, "error": None, "class": None, "total_s": TOTAL, "frames": 48, "checks_ok": True}}))
'''


def make_home(root: str, graph: dict | None = None, ref: bool = True, face_ok: bool = True) -> str:
    """$FOREMAN_HOME with a presenter (hero, face and scene stills; a Kokoro voice with a reference
    recording), a capture, the facts and the h3 graph template."""
    home = os.path.join(root, "home")
    cdir = os.path.join(home, "characters", "presenter-a")
    for i, tag in enumerate(TAGS):
        png(os.path.join(cdir, "stills", f"{tag}.png"), shade=40 + 30 * i)
    voice = {"engine": "kokoro", "voice": "af_heart", "speed": 1.0}
    if ref:
        wav(os.path.join(cdir, "voice", "reference.wav"), 1.0)
        voice["ref_wav"] = "voice/reference.wav"
    bible = {"id": "presenter-a", "name": "Presenter A", "kind": "human", "subject": "the presenter",
             "look": "a woman with shoulder-length brown hair in a plain teal sweater", "voice": voice,
             "stills": [{"file": f"stills/{t}.png", "tags": [t], "approved": face_ok or t != "face"} for t in TAGS]}
    with open(os.path.join(cdir, "bible.json"), "w") as f:
        json.dump(bible, f)
    png(os.path.join(home, "assets", "captures", "upload.png"), 120, 90, 200)
    with open(os.path.join(home, "assets", "facts-h3.json"), "w") as f:
        json.dump(FACTS, f)
    g = load(EXAMPLE_GRAPH)
    g.update(commercial_ok=True)
    g.update(graph or {})
    os.makedirs(os.path.join(home, "comfyui", "graphs"), exist_ok=True)
    with open(os.path.join(home, "comfyui", "graphs", "h3_ref2va.json"), "w") as f:
        json.dump(g, f)
    return home


class Ctx:
    """The local engine's context, enough for a handler: subprocesses run for real."""

    def __init__(self):
        self.runs = []

    def progress(self, f):
        pass

    def log(self, line):
        pass

    def check(self):
        pass

    def run(self, cmd, timeout=1800.0, cwd=None, heartbeat=None, group=False):
        self.runs.append({"cmd": list(cmd), "cwd": cwd, "group": group})
        return subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd, start_new_session=bool(group))


class H3Case(TmpCase):
    def setUp(self):
        super().setUp()
        # The work directories the facts step returns are real paths: so is the home (macOS's TMPDIR
        # is under /var, a symlink to /private/var, and relpath across the two leaves the artifacts).
        self.tmp = os.path.realpath(self.tmp)
        self.fake = os.path.join(self.tmp, "fakepy", "python")
        os.makedirs(os.path.dirname(self.fake))
        with open(self.fake, "w") as f:
            f.write(FAKE_PYTHON.format(python=sys.executable))
        os.chmod(self.fake, 0o755)
        self.control({})

    def control(self, ctl: dict) -> None:
        with open(os.path.join(os.path.dirname(self.fake), "control.json"), "w") as f:
            json.dump(ctl, f)

    def requests(self, command=None) -> list:
        path = os.path.join(os.path.dirname(self.fake), "requests.jsonl")
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        return [r for r in rows if command is None or r["command"] == command]

    def pipe(self, home, max_depth=64, **cfg) -> ad_pipeline.AdPipeline:
        base = dict(media_python=self.fake, h3_graph="h3_ref2va", out_width=1080, out_height=1920)
        base.update(cfg)
        return ad_pipeline.AdPipeline(ads_config(**base), home=home, artifacts_dir=os.path.join(home, "artifacts"),
                                      ffmpeg=FFMPEG or "/no/ffmpeg", catalog=studio.catalog(False, ads=True),
                                      max_depth=max_depth, commercial_models=studio.COMMERCIAL_OK,
                                      licenses=studio.LICENSES)

    def facts_job(self, pipe, ad) -> dict:
        p = {"goal": "g", "expected_s": 30, "workdir": f"ads/{ad['id']}", "ad": ad}
        return pipe.run_facts(RunSpec("1.1", 1, "ad_facts", studio.LOCAL, p, 30), Ctx())

    def ready(self, home=None, **cfg):
        """An h3 ad past its facts step with the plan written: (pipeline, work directory, ad)."""
        home = home or make_home(self.tmp)
        pipe = self.pipe(home, **cfg)
        art = self.facts_job(pipe, h3_ad())
        wd = os.path.dirname(art["path"])
        ad = pipe.load_ad(wd)
        script, errors = pipe.write_script(wd, ad, pipe.bible(wd), paths.read_json(os.path.join(wd, "facts.json")),
                                           "brain", None, Brain([json.dumps(PLAN)]))
        self.assertEqual(errors, "", errors)
        return pipe, wd, ad

    def advance(self, pipe, wd) -> dict:
        return pipe.advance(wd, os.path.relpath(wd, pipe.artifacts_dir), {"context": {}}, "brain", None, None)


class Brain:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, model, messages, ctx, max_tokens=None):
        self.seen.append(messages)
        return self.replies.pop(0)


def say(line: str, t0: float = 0.3, word: float = 0.3, gap: float = 0.15, words=None) -> list:
    """Word times of a clean reading: each word `word` s long, `gap` s of quiet between words."""
    out, t = [], t0
    for w in words or line.split():
        out.append((w, round(t, 3), round(t + word, 3)))
        t += word + gap
    return out


def done_meta(path: str, **extra) -> None:
    paths.atomic_json(path + ".meta.json", dict({"complete": True, "ok": True}, **extra))


def render(wd: str, job: dict, seconds: float = 120.0, ok: bool = True) -> None:
    """What ComfyUI leaves for each take op of a job: the MP4 and its sidecar."""
    for op in job["payload"]["ops"]:
        out = os.path.join(wd, op["out"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        done_meta(out, ok=ok, seconds=seconds, seed=op["seed"], unit=op["unit"], license="test", graph=op["graph"])


def probe_for(beat: dict, words: list, **over) -> dict:
    """The probe of a clean take: long enough for the line, voiced exactly where its words are."""
    end = words[-1][2] if words else 0.0
    rec = {"duration_s": round(max(end + 0.6, float(beat["seconds"])), 3),
           "voiced_runs": [[s, e] for _, s, e in words], "db10": [-20.0]}
    rec.update(over)
    return rec


def hear(wd: str, take: dict, words: list, text=None, hinted_text=None) -> None:
    """The two transcriptions of a take: hinted with word times, and free-running (text only)."""
    said = " ".join(w for w, _, _ in words)
    paths.atomic_json(paths.out_path(wd, take["hinted"]), {"text": hinted_text or said,
                                                           "words": [{"word": w, "start": s, "end": e}
                                                                     for w, s, e in words]})
    done_meta(os.path.join(wd, take["hinted"]))
    paths.atomic_json(paths.out_path(wd, take["transcript"]), {"text": said if text is None else text})
    done_meta(os.path.join(wd, take["transcript"]))


def gate(wd: str) -> dict:
    return paths.read_json(os.path.join(wd, "gate.json"))


def load(path: str):
    with open(path) as f:
        return json.load(f)


def text_of(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


# ---------------------------------------------------------------- S1 the facts step's preflight

class S1Preflight(H3Case):
    def test_an_h3_ad_is_picked_and_records_its_graph_license(self):
        home = make_home(self.tmp, graph={"license": "Model license: test terms", "attribution": "Example Model"})
        pipe = self.pipe(home)
        ctx = Ctx()
        art = pipe.run_facts(RunSpec("1.1", 1, "ad_facts", studio.LOCAL,
                                     {"goal": "g", "expected_s": 30, "workdir": "ads/ad-h3", "ad": h3_ad()}, 30), ctx)
        ad = paths.read_json(os.path.join(os.path.dirname(art["path"]), "ad.json"))
        picks = ad["picks"]
        self.assertEqual(picks["backends"]["talk"], "h3")
        self.assertEqual(picks["models"], {"voice": studio.COMFYUI, "talk": studio.COMFYUI, "asr": studio.QWEN3_ASR,
                                           "music": studio.ACE_STEP, "cpu": studio.LOCAL},
                         "no TTS, stills or B-roll model is named: none runs")
        self.assertEqual(picks["graphs"]["talk"], "h3_ref2va")
        self.assertEqual(picks["h3"], {"width": 576, "height": 1024, "graph": "h3_ref2va",
                                       "license": "Model license: test terms", "attribution": "Example Model",
                                       "commercial_ok": True})
        self.assertEqual(ctx.runs[0]["cmd"], [self.fake, "-I", "-c", "import numpy, PIL"])
        self.assertTrue(pipe.is_h3(ad))

    def test_the_machine_is_infrastructure_and_engine_unconfigured(self):
        home = make_home(self.tmp)
        cases = [("no media python", dict(media_python=""), None, "media_python"),
                 ("numpy missing", {}, {"import_exit": 1, "import_error": "No module named numpy"}, "numpy"),
                 ("no graph", dict(h3_graph="nope"), None, "nope"),
                 ("no python file", dict(media_python=os.path.join(self.tmp, "absent", "python")), None, "cannot run"),
                 ("another frame size", dict(out_width=270, out_height=480), None, "1080x1920")]
        for name, cfg, ctl, needle in cases:
            with self.subTest(name):
                self.control(ctl or {})
                with self.assertRaises(InfraFailure) as e:
                    self.facts_job(self.pipe(home, **cfg), h3_ad())
                self.assertEqual(e.exception.reason, "engine_unconfigured")
                self.assertIn(needle, str(e.exception))
        self.control({})
        for name, edit, needle in (("no license", {"license": ""}, "no license"),
                                   ("no outputs", {"outputs": None}, "output node"),
                                   ("no voice placeholder", {"graph": {k: v for k, v in load(EXAMPLE_GRAPH)["graph"].items()
                                                                       if k != "901"}}, "audio_1")):
            with self.subTest(name):
                home = make_home(os.path.join(self.tmp, name.replace(" ", "-")), graph=edit)
                with self.assertRaises(InfraFailure) as e:
                    self.facts_job(self.pipe(home), h3_ad())
                self.assertEqual(e.exception.reason, "engine_unconfigured")
                self.assertIn(needle, str(e.exception))

    def test_a_commercial_template_needs_the_graphs_commercial_ok(self):
        home = make_home(self.tmp, graph={"commercial_ok": False})
        with self.assertRaises(ContentFailure) as e:
            self.facts_job(self.pipe(home), h3_ad(commercial=True))
        self.assertIn("commercial_ok", str(e.exception))
        self.facts_job(self.pipe(home), h3_ad(ad_id="ad-free"))  # not commercial: the owner's call, runs

    def test_the_example_template_is_commercial_and_refused_with_the_example_graph(self):
        """The example advertises a paid service: it is commercial, and the example graph's commercial_ok false
        stops it at the facts step, before any GPU job, until the owner sets it."""
        templates, _ = load_templates(os.path.join(EXAMPLES, "templates"))
        self.assertTrue(templates["ads.example-h3.v1"].commercial)
        home = make_home(self.tmp, graph=load(EXAMPLE_GRAPH))
        self.assertFalse(load(EXAMPLE_GRAPH)["commercial_ok"])
        with self.assertRaises(ContentFailure) as e:
            self.facts_job(self.pipe(home), h3_ad(commercial=templates["ads.example-h3.v1"].commercial))
        self.assertIn("commercial_ok", str(e.exception))

    def test_the_graph_must_take_its_render_size(self):
        """A graph whose size is fixed inside it renders every take at that size: refused before any take."""
        text = json.dumps(load(EXAMPLE_GRAPH)["graph"]).replace('"{{width}}"', "640").replace('"{{height}}"', "1152")
        home = make_home(self.tmp, graph={"graph": json.loads(text)})
        with self.assertRaises(InfraFailure) as e:
            self.facts_job(self.pipe(home), h3_ad())
        self.assertEqual(e.exception.reason, "engine_unconfigured")
        self.assertIn("width, height", str(e.exception))

    def test_a_relative_media_python_runs_where_the_preflight_ran_it(self):
        """The worker runs in the ad's work directory: the interpreter is resolved once, from gpud's own."""
        rel = os.path.relpath(self.fake, os.getcwd())
        self.assertIn(os.sep, rel)
        pipe, wd, ad = self.ready(media_python=rel)
        self.assertEqual(pipe.media_python(), self.fake)
        job = self.advance(pipe, wd)["submit"][0]
        render(wd, job)
        probe_job = self.advance(pipe, wd)["submit"][0]
        self.assertEqual(probe_job["kind"], "ad_probe")
        pipe.run_probe(RunSpec("2.1", 2, "ad_probe", studio.LOCAL, probe_job["payload"], 30), Ctx())
        self.assertEqual(self.requests("probe")[-1]["cwd"], os.path.realpath(wd))
        bare = self.pipe(make_home(os.path.join(self.tmp, "b")), media_python="sh")
        self.assertEqual(bare.media_python(), os.path.abspath(shutil.which("sh")), "a bare name is looked up on PATH")

    def test_the_character_and_the_template_are_content(self):
        cases = [("no reference voice", dict(ref=False), {}, "reference voice"),
                 ("face not approved", dict(face_ok=False), {}, "ad.h3.face"),
                 ("box outside the capture", {}, {"captures": {"captures/upload.png": {
                     "id": "upload", "crop": [0, 0, 400, 80]}}}, "outside the capture"),
                 ("card price not in the facts", {}, {"endcard": {"big": "$7.25", "lines": []}}, "$7.25")]
        for name, home_kw, h3_over, needle in cases:
            with self.subTest(name):
                home = make_home(os.path.join(self.tmp, name.replace(" ", "-")), **home_kw)
                with self.assertRaises(ContentFailure) as e:
                    self.facts_job(self.pipe(home), h3_ad(h3_block(**h3_over)))
                self.assertIn(needle, str(e.exception))


# ---------------------------------------------------------------- S2 the dispatcher

class S2Dispatcher(H3Case):
    def setUp(self):
        super().setUp()
        self.pipe_, self.wd, self.ad = self.ready()
        self.script = paths.read_json(os.path.join(self.wd, "script.json"))
        self.beats = self.script["beats"]

    def clean_round(self, job, skip=(), probes=None):
        """Render, probe and hear every take of `job` cleanly (beats in `skip` are left to the test)."""
        render(self.wd, job)
        g = gate(self.wd)
        ctl = {"probes": {}}
        said = {}
        for op in job["payload"]["ops"]:
            k = op["shot"]
            words = say(self.beats[k]["line"])
            said[k] = words
            ctl["probes"][op["out"]] = (probes or {}).get(k) or probe_for(self.beats[k], words)
        self.control(ctl)
        d = self.advance(self.pipe_, self.wd)
        self.assertEqual(d["submit"][0]["kind"], "ad_probe", d)
        probe_job = d["submit"][0]
        self.pipe_.run_probe(RunSpec("2.1", 2, "ad_probe", studio.LOCAL, probe_job["payload"], 30), Ctx())
        d = self.advance(self.pipe_, self.wd)
        heard = d["submit"][0] if d["submit"] and d["submit"][0]["kind"] == "ad_gate" else None
        g = gate(self.wd)
        for op in job["payload"]["ops"]:
            k = op["shot"]
            take = g["units"][f"beat-{k}"]["attempts"][-1]
            if k not in skip and heard is not None:
                hear(self.wd, take, said[k], text=self.beats[k]["line"].replace("$9.50", "nine fifty"))
        return d, said

    def test_a_whole_h3_ad_round_by_round(self):
        pipe, wd = self.pipe_, self.wd
        d = self.advance(pipe, wd)
        job = d["submit"][0]
        self.assertEqual((job["kind"], job["requires"], job["payload"]["stage"]), ("ad_shots", studio.COMFYUI, "takes"))
        ops = job["payload"]["ops"]
        self.assertEqual([op["out"] for op in ops], [f"takes/beat-{k}-t0.mp4" for k in range(4)],
                         "one take per beat, four fit one job (4 x 140 s within max_job_s)")
        op = ops[3]
        bible = pipe.bible(wd)
        self.assertEqual(op["images"], {"image_1": bible.approved("kitchen")[0], "image_2": bible.hero(),
                                        "image_3": bible.approved("face")[0], "audio_1": bible.voice_ref()})
        self.assertEqual((op["graph"], op["params"]["width"], op["params"]["height"]), ("h3_ref2va", 576, 1024))
        self.assertEqual(op["seed"], (5 + 51 + 3) % 2147483647)
        self.assertEqual(op["params"]["seconds"], self.beats[3]["seconds"])
        self.assertIn("<d>[English] Get it done today at example dot com and skip the line.</d>", op["params"]["prompt"],
                      "the address is said spelled out first")
        self.assertIn("<d>[English] It costs nine fifty, with everything included.</d>", ops[2]["params"]["prompt"])
        g = gate(wd)
        self.assertEqual((g["mode"], g["status"], sorted(g["units"])), ("h3", "running", [f"beat-{k}" for k in range(4)]))
        self.assertEqual(ad_pipeline.progress(wd)["current"], "takes")
        self.assertEqual(self.advance(pipe, wd)["submit"][0]["payload"]["ops"], ops, "idempotent until rendered")

        # beat 1's take says the brand wrong: heard again with the next spelling... after a reroll first
        d, said = self.clean_round(job, skip=(1,))
        heard = d["submit"][0]
        self.assertEqual(heard["kind"], "ad_gate")
        hops = heard["payload"]["ops"]
        self.assertEqual(len(hops), 8, "two transcriptions per take")
        hinted = next(o for o in hops if o["out"] == "gate/beat-0-t0.asr.json")
        self.assertNotIn("words", hinted, "word times on")
        self.assertIn("ExampleCo", hinted["prompt"])
        free = next(o for o in hops if o["out"] == "gate/beat-0-t0.asr0.json")
        self.assertEqual((free["words"], free["response_format"], free["release_companions"]), (False, "json", True))
        probe_req = self.requests("probe")[0]
        self.assertEqual(probe_req["cwd"], os.path.realpath(wd))
        self.assertTrue(probe_req["pgid_is_own"], "the worker runs in its own process group")
        takes = {t["mp4"]: t for t in probe_req["request"]["takes"]}
        self.assertEqual(takes["takes/beat-3-t0.mp4"]["hands"], {"floor": 0.74, "limit": 30})
        self.assertEqual(takes["takes/beat-3-t0.mp4"]["crop"], {"scale": 1.2, "anchor_y": None})
        self.assertIsNone(takes["takes/beat-0-t0.mp4"]["hands"])
        self.assertEqual(takes["takes/beat-0-t0.mp4"]["render"], [576, 1024])
        g = gate(wd)
        hear(wd, g["units"]["beat-1"]["attempts"][0], said[1],
             text="Upload your document from your phone, and Example Ko prepares and files it.")
        d = self.advance(pipe, wd)
        retake = d["submit"][0]
        self.assertEqual([o["out"] for o in retake["payload"]["ops"]], ["takes/beat-1-t1.mp4"])
        self.assertIn("retake round 1", d["note"])
        g = gate(wd)
        u = g["units"]["beat-1"]
        self.assertEqual((u["n"], u["status"], u["rerolls"]), (1, "pending", 1))
        self.assertEqual(u["attempts"][0]["status"], "failed")
        self.assertIn("ExampleCo", u["attempts"][0]["verdict"])
        self.assertEqual(u["attempts"][1]["seed"], (5 + 51 + 1 + 1000) % 2147483647)
        self.assertEqual(retake["payload"]["ops"][0]["seed"], u["attempts"][1]["seed"])
        self.assertEqual([g["units"][f"beat-{k}"]["status"] for k in (0, 2, 3)], ["passed"] * 3)
        self.assertEqual(g["spent_s"], 480.0)
        self.assertEqual(g["rounds"], 1)

        # the retake passes: the voice is done, music next
        d, _ = self.clean_round(retake)
        d = self.advance(pipe, wd)
        music = d["submit"][0]
        self.assertEqual(music["kind"], "ad_music")
        g = gate(wd)
        self.assertEqual(g["status"], "voice_passed")
        self.assertEqual(g["track"]["status"], "passed")
        self.assertEqual(g["track"]["from"], ["takes/beat-0-t0.mp4", "takes/beat-1-t1.mp4", "takes/beat-2-t0.mp4",
                                              "takes/beat-3-t0.mp4"])
        upper = sum(max(say(b["line"])[-1][2] + 0.6, float(b["seconds"])) + b["hold"] for b in self.beats) + 1.0
        self.assertEqual(music["payload"]["ops"][0]["duration_seconds"], int(max(10, -(-(upper + 1) // 1))))
        self.assertEqual(ad_pipeline.progress(wd)["current"], "music")
        out = os.path.join(wd, music["payload"]["ops"][0]["out"])
        wav(out, 1.0)
        done_meta(out)
        d = self.advance(pipe, wd)
        self.assertEqual(d["submit"][0]["kind"], "ad_assemble")
        self.assertEqual(d["submit"][0]["payload"]["expected_s"], 240.0)

    def test_failures_before_listening_and_their_retakes(self):
        pipe, wd = self.pipe_, self.wd
        job = self.advance(pipe, wd)["submit"][0]
        probes = {0: {"has_audio": False, "problems": ["no audio"], "duration_s": 5.0},
                  3: dict(probe_for(self.beats[3], say(self.beats[3]["line"])), hands=[0.33, 0.5])}
        d, _ = self.clean_round(job, probes=probes)
        heard = d["submit"][0]["payload"]["ops"]
        self.assertNotIn("gate/beat-0-t0.asr.json", [o["out"] for o in heard], "a take without audio is not heard")
        d = self.advance(pipe, wd)
        self.assertEqual(sorted(o["out"] for o in d["submit"][0]["payload"]["ops"]),
                         ["takes/beat-0-t1.mp4", "takes/beat-3-t1.mp4"])
        g = gate(wd)
        self.assertEqual(g["units"]["beat-0"]["attempts"][0]["verdict"], "take: no audio")
        self.assertIn("hands visible", g["units"]["beat-3"]["attempts"][0]["verdict"])

    def test_a_take_whose_audio_does_not_decode_is_judged_and_taken_again(self):
        """The worker records the problem and no WAV: the probe step moves on (it used to ask for the same probe
        until the depth limit), the take fails on its probe and the beat is taken again."""
        pipe, wd = self.pipe_, self.wd
        job = self.advance(pipe, wd)["submit"][0]
        probes = {0: {"problems": ["audio does not decode: Invalid data found when processing input"],
                      "no_wav": True, "duration_s": 5.0}}
        d, _ = self.clean_round(job, probes=probes)
        self.assertEqual(d["submit"][0]["kind"], "ad_gate", "heard, not probed again")
        self.assertFalse(os.path.exists(os.path.join(wd, "takes", "beat-0-t0.wav")))
        self.assertNotIn("gate/beat-0-t0.asr.json", [o["out"] for o in d["submit"][0]["payload"]["ops"]])
        d = self.advance(pipe, wd)
        self.assertEqual([o["out"] for o in d["submit"][0]["payload"]["ops"]], ["takes/beat-0-t1.mp4"])
        self.assertIn("audio does not decode", gate(wd)["units"]["beat-0"]["attempts"][0]["verdict"])
        self.assertEqual(len(self.requests("probe")), 1)

    def test_a_take_of_another_size_is_the_machines_failure(self):
        pipe, wd = self.pipe_, self.wd
        job = self.advance(pipe, wd)["submit"][0]
        render(wd, job)
        self.control({"probes": {"takes/beat-2-t0.mp4": {"width": 640, "height": 1152,
                                                          "problems": ["size 640x1152 is not the render size 576x1024"]}}})
        probe_job = self.advance(pipe, wd)["submit"][0]
        with self.assertRaises(InfraFailure) as e:
            pipe.run_probe(RunSpec("2.1", 2, "ad_probe", studio.LOCAL, probe_job["payload"], 30), Ctx())
        self.assertEqual(e.exception.reason, "engine_unconfigured")
        self.assertIn("640x1152, not the 576x1024 asked", str(e.exception))
        self.assertEqual({u["status"] for u in gate(wd)["units"].values()}, {"pending"}, "no take charged")

    def test_a_take_heard_saying_what_the_line_does_not_is_taken_again(self):
        """H3 makes its own speech: a claim inside the line passes the error rate, not the ad's rules."""
        pipe, wd = self.pipe_, self.wd
        job = self.advance(pipe, wd)["submit"][0]
        d, said = self.clean_round(job, skip=(1,))
        hear(wd, gate(wd)["units"]["beat-1"]["attempts"][0], said[1],
             text="Upload your document from your phone, and ExampleCo prepares and files it, guaranteed.")
        d = self.advance(pipe, wd)
        self.assertEqual([o["out"] for o in d["submit"][0]["payload"]["ops"]], ["takes/beat-1-t1.mp4"])
        verdict = gate(wd)["units"]["beat-1"]["attempts"][0]["verdict"]
        self.assertIn("said what the line does not: a guarantee", verdict)

    def test_a_render_that_produced_nothing_counts_as_a_take(self):
        pipe, wd = self.pipe_, self.wd
        job = self.advance(pipe, wd)["submit"][0]
        render(wd, job, ok=False)
        d = self.advance(pipe, wd)
        self.assertEqual(d["submit"][0]["kind"], "ad_shots", "nothing to probe: every take is taken again")
        g = gate(wd)
        self.assertEqual({u["attempts"][0]["verdict"] for u in g["units"].values()}, {"take: render produced no video"})

    def test_attempts_and_the_render_budget_run_out_as_content(self):
        pipe, wd = self.pipe_, self.wd
        pipe.cfg = ads.AdsConfig(**dict(pipe.cfg.__dict__, h3_attempts=2))
        job = self.advance(pipe, wd)["submit"][0]
        d, said = self.clean_round(job, skip=(2,))
        hear(wd, gate(wd)["units"]["beat-2"]["attempts"][0], said[2], text="It costs four dollars, with nothing.")
        retake = self.advance(pipe, wd)["submit"][0]
        d, said = self.clean_round(retake, skip=(2,))
        hear(wd, gate(wd)["units"]["beat-2"]["attempts"][1], said[2], text="It costs four dollars, with nothing.")
        d = self.advance(pipe, wd)
        self.assertEqual(d["verdict"], "give_up")
        self.assertIn("speech gate: beat 2 (body) never passed after 2 take(s)", d["note"])
        self.assertEqual(gate(wd)["status"], "failed")
        self.assertEqual(self.advance(pipe, wd)["verdict"], "give_up", "a failed gate stays failed")

        pipe2, wd2, _ = self.ready(home=make_home(os.path.join(self.tmp, "b")), h3_budget_s=100.0)
        self.pipe_, self.wd = pipe2, wd2
        job = self.advance(pipe2, wd2)["submit"][0]
        d, said = self.clean_round(job, skip=(2,))
        hear(wd2, gate(wd2)["units"]["beat-2"]["attempts"][0], said[2], text="It costs four dollars.")
        d = self.advance(pipe2, wd2)
        self.assertEqual(d["verdict"], "give_up")
        self.assertIn("render budget is spent", d["note"])

    def test_an_ad_that_cannot_fit_the_depth_limit_is_refused_before_any_take(self):
        pipe, wd, _ = self.ready(home=make_home(os.path.join(self.tmp, "c")), max_depth=30)
        d = self.advance(pipe, wd)
        self.assertEqual(d["verdict"], "give_up")
        self.assertIn("depth limit 30", d["note"])
        self.assertFalse(os.path.exists(os.path.join(wd, "gate.json")))

    def test_the_script_step_uses_the_h3_prompt_and_validator(self):
        home = make_home(os.path.join(self.tmp, "d"))
        pipe = self.pipe(home)
        wd = os.path.dirname(self.facts_job(pipe, h3_ad(ad_id="ad-s"))["path"])
        ad = pipe.load_ad(wd)
        bad = copy.deepcopy(PLAN)
        bad["beats"][3]["line"] = "Get it done today."
        brain = Brain([json.dumps(bad), json.dumps(PLAN)])
        script, why = pipe.write_script(wd, ad, pipe.bible(wd), paths.read_json(os.path.join(wd, "facts.json")),
                                        "brain", None, brain)
        self.assertEqual(brain.seen[0][0]["content"], h3core.H3_SYSTEM)
        self.assertIn("web address", brain.seen[1][-1]["content"], "the repair names the rule")
        self.assertEqual((script["backend"], len(script["beats"])), ("h3", 4))
        self.assertEqual([ln["role"] for ln in script["lines"]], ["hook", "body", "body", "cta"])


@needs_ffmpeg
class S2Edit(H3Case):
    """The edit, the final gate (with one safe re-edit) and finish, from a voice that passed."""

    def setUp(self):
        super().setUp()
        self.pipe_, self.wd, self.ad = self.ready()
        self.script = paths.read_json(os.path.join(self.wd, "script.json"))
        g = self.pipe_.h3_gate(self.wd, self.ad, self.script)
        for b in self.script["beats"]:
            u = g["units"][f"beat-{b['k']}"]
            if b["k"] == 1:  # a failed first take: an alternate
                u["attempts"][0].update(status="failed", ok=False, verdict="runs on")
                u["attempts"].append(h3core.h3_attempt(b, 1, g["lexicon"], {}, None, "a new seed", self.ad))
            take = u["attempts"][-1]
            words = say(b["line"])
            take.update(status="passed", ok=True, span=[words[0][1], words[-1][2], None], wer_inner=0.0,
                        heard=b["line"], words=[{"text": w, "start": s, "end": e} for w, s, e in words])
            u["status"] = "passed"
            for t in u["attempts"]:
                for k in ("mp4", "wav"):
                    wav(os.path.join(self.wd, t[k]), 0.3)
                done_meta(os.path.join(self.wd, t["mp4"]), seconds=100.0)
                paths.atomic_json(os.path.join(self.wd, t["probe"]), {"problems": [], "duration_s": 5.0})
        g.update(status="voice_passed", track={"status": "passed", "wer": 0.0, "wer_inner": 0.0, "lost": 0.0,
                                               "heard": "x"})
        self.pipe_.save_gate(self.wd, g)
        wav(os.path.join(self.wd, "music", "bed.wav"), 1.0)
        done_meta(os.path.join(self.wd, "music", "bed.wav"))

    def assemble(self):
        return self.pipe_.run_assemble(RunSpec("3.1", 3, "ad_assemble", studio.LOCAL,
                                               {"workdir": os.path.relpath(self.wd, self.pipe_.artifacts_dir)}, 240),
                                       Ctx())

    def final_round(self, e, fail_beat=None):
        g = gate(self.wd)
        d = self.advance(self.pipe_, self.wd)
        job = d["submit"][0]
        self.assertEqual(job["kind"], "ad_gate")
        outs = sorted(o["out"] for o in job["payload"]["ops"])
        self.assertEqual(outs, sorted([f"gate/final-b{k}-r{e}.json" for k in range(4)] + [f"gate/mix-r{e}.json"]))
        files = {o["out"]: o["file_path"] for o in job["payload"]["ops"]}
        self.assertEqual(files[f"gate/final-b2-r{e}.json"], "final/beats/b2-16k.wav")
        for b in self.script["beats"]:
            text = "It costs four dollars." if b["k"] == fail_beat else b["line"]
            out = paths.out_path(self.wd, f"gate/final-{b['id']}-r{e}.json")
            paths.atomic_json(out, {"text": text})
            done_meta(out)
        out = os.path.join(self.wd, f"gate/mix-r{e}.json")
        paths.atomic_json(out, {"text": " ".join(ln["text"] for ln in self.script["lines"])})
        done_meta(out)
        return self.advance(self.pipe_, self.wd)

    def test_edit_final_gate_safe_reedit_and_finish(self):
        pipe, wd = self.pipe_, self.wd
        self.assertEqual(self.advance(pipe, wd)["submit"][0]["kind"], "ad_assemble")
        art = self.assemble()
        req = self.requests("edit")[0]
        r = req["request"]
        self.assertEqual(req["worker"], ad_pipeline.H3_WORKER)
        self.assertEqual((r["out"], r["render"], r["safe"], r["music"]), ([1080, 1920], [576, 1024], False, "music/bed.wav"))
        self.assertEqual([b["take"] for b in r["beats"]], ["takes/beat-0-t0.mp4", "takes/beat-1-t1.mp4",
                                                          "takes/beat-2-t0.mp4", "takes/beat-3-t0.mp4"])
        self.assertEqual(r["beats"][3]["hold"], 1.0)
        self.assertEqual(r["beats"][0]["tail"], 0.05)
        self.assertEqual(r["captures"]["upload"]["png"], os.path.realpath(
            os.path.join(pipe.home, "assets", "captures", "upload.png")))
        self.assertEqual(r["price_card"]["big"], "$9.50")
        self.assertEqual(r["endcard"]["url"], "example.com")
        self.assertEqual(set(r["stills"]), {"payoff"})
        self.assertIn("example.com", r["captions"]["solo"])
        self.assertIn("$9.50", r["captions"]["solo"])
        self.assertEqual(r["music_under"], 14.0)
        self.assertEqual(r["encode"]["crf"], 16)
        self.assertEqual(r["headline"], PLAN["headline"])
        self.assertTrue(os.path.isabs(r["fonts"]["caption"]))
        timing = paths.read_json(os.path.join(wd, "timing.json"))
        self.assertEqual(timing["total_s"], 2.0)
        self.assertEqual([s["kind"] for s in timing["shots"]], ["talk", "punch", "talk", "endcard"])
        self.assertEqual(timing["endcard_start"], 1.5)
        self.assertEqual({w["line"] for w in timing["words"]}, {0, 1, 2, 3}, "a word's line is its beat's")
        pick = paths.read_json(os.path.join(wd, "pick.json"))
        self.assertEqual(pick["1"]["take"], "takes/beat-1-t1.mp4")
        self.assertEqual(pick["1"]["alternates"], ["takes/beat-1-t0.mp4"])
        asm = paths.read_json(os.path.join(wd, "final", "assembly.json"))
        for k in ("encoder", "picks", "segments", "font", "caption_events", "label", "burned_in", "loudnorm_first_pass",
                  "total_s", "frames", "music", "music_db", "mix_round", "ffmpeg", "at_wall", "h3"):
            self.assertIn(k, asm)
        self.assertEqual((asm["label"], asm["edit_round"], asm["frames"]), (False, 0, 48))
        self.assertTrue(asm["burned_in"]["ok_captions"])
        self.assertFalse(os.path.exists(os.path.join(wd, "final", "video.mp4")), "the uncaptioned cut is removed")
        self.assertIn(os.path.join(wd, "final", "publish.json"), art["paths"])

        # beat 2's finished voice lost its price: edited again, safe
        d = self.final_round(0, fail_beat=2)
        self.assertEqual(d["submit"][0]["kind"], "ad_assemble")
        g = gate(wd)
        self.assertEqual((g["final"]["safe"], g["final"]["round"]), (True, 1))
        self.assertFalse(g["final"]["beats"]["b2"]["ok"])
        self.assertFalse(os.path.exists(os.path.join(wd, "final", "ad.mp4")))
        self.assertEqual(ad_pipeline.progress(wd)["current"], "assemble")
        self.assemble()
        self.assertTrue(self.requests("edit")[1]["request"]["safe"])
        self.assertEqual(paths.read_json(os.path.join(wd, "final", "assembly.json"))["edit_round"], 1)
        d = self.final_round(1)
        fin = d["submit"][0]
        self.assertEqual((fin["kind"], fin["on_success"]["kind"]), ("ad_finish", "ad_review"))
        g = gate(wd)
        self.assertEqual((g["status"], g["mix"]["status"]), ("passed", "passed"))
        s = pipe.gate_summary(wd)
        self.assertEqual(pipe.gate_note(s), "speech gate passed (4 beats, 1 retake; final lines heard intact, edited "
                                            "with no pause removed; mix WER 0.00)")

        res = pipe.run_finish(RunSpec("4.1", 4, "ad_finish", studio.LOCAL,
                                      {"workdir": os.path.relpath(wd, pipe.artifacts_dir)}, 120), Ctx())
        checks = {c["check"]: c for c in res["checks"]}
        for name in ("frames_exact", "no_freeze", "punch_cuts", "color_bt709"):
            self.assertTrue(checks[name]["ok"], checks[name])
        self.assertEqual([p for p in res["paths"] + [res["path"]] if p.endswith(".mp4")],
                         [os.path.join(wd, "final", "ad.mp4")], "never a second MP4")
        for name in ("ad.srt", "publish.json", "timeline.json"):
            self.assertIn(os.path.join(wd, "final", name), res["paths"])
        m = paths.read_json(os.path.join(wd, "manifest.json"))
        self.assertEqual(m["h3"]["graph"], "h3_ref2va")
        self.assertEqual(m["h3"]["chosen"]["b1"], "takes/beat-1-t1.mp4")
        self.assertEqual([a["n"] for a in m["seeds"]["attempts"]["b1"]], [0, 1])
        self.assertEqual(m["gate"]["final"]["round"], 1)
        self.assertFalse(m["disclosure"]["label_burned_in"])
        self.assertEqual(m["h3"]["publish"], "final/publish.json")

        # a punch that is not a cut is caught on the file
        timing = paths.read_json(os.path.join(wd, "timing.json"))
        timing["shots"][1]["start"] = 0.25
        paths.atomic_json(os.path.join(wd, "timing.json"), timing)
        res = pipe.run_finish(RunSpec("4.2", 4, "ad_finish", studio.LOCAL,
                                      {"workdir": os.path.relpath(wd, pipe.artifacts_dir)}, 120), Ctx())
        self.assertFalse({c["check"]: c for c in res["checks"]}["punch_cuts"]["ok"])

    def test_a_second_final_line_failure_fails_and_masking_gets_one_remix(self):
        pipe, wd = self.pipe_, self.wd
        self.assemble()
        g = gate(wd)
        g["final"]["safe"] = True  # already edited safe once
        pipe.save_gate(wd, g)
        d = self.final_round(0, fail_beat=3)
        self.assertEqual(d["verdict"], "give_up")
        self.assertIn("beat 3 (cta): its finished voice", d["note"])

        self.pipe_, self.wd, self.ad = self.ready(home=make_home(os.path.join(self.tmp, "m")))
        self.setUp_gate()
        pipe, wd = self.pipe_, self.wd
        self.assemble()
        g = self.advance(pipe, wd)
        for b in self.script["beats"]:
            paths.atomic_json(paths.out_path(wd, f"gate/final-{b['id']}-r0.json"), {"text": b["line"]})
            done_meta(os.path.join(wd, f"gate/final-{b['id']}-r0.json"))
        paths.atomic_json(os.path.join(wd, "gate/mix-r0.json"), {"text": "Need a form filed. Example dot com."})
        done_meta(os.path.join(wd, "gate/mix-r0.json"))
        d = self.advance(pipe, wd)
        self.assertEqual(d["submit"][0]["kind"], "ad_assemble")
        g = gate(wd)
        self.assertEqual((g["mix"]["round"], g["mix"]["music_db"], g["final"]["round"], g["final"]["safe"]),
                         (1, -20.0, 1, False))
        self.assemble()
        self.assertEqual(self.requests("edit")[-1]["request"]["music_under"], 22.0)

    def setUp_gate(self):
        self.script = paths.read_json(os.path.join(self.wd, "script.json"))
        g = self.pipe_.h3_gate(self.wd, self.ad, self.script)
        for b in self.script["beats"]:
            u = g["units"][f"beat-{b['k']}"]
            take = u["attempts"][-1]
            words = say(b["line"])
            take.update(status="passed", ok=True, span=[words[0][1], words[-1][2], None], wer_inner=0.0,
                        heard=b["line"], words=[{"text": w, "start": s, "end": e} for w, s, e in words])
            u["status"] = "passed"
            for k in ("mp4", "wav"):
                wav(os.path.join(self.wd, take[k]), 0.3)
            done_meta(os.path.join(self.wd, take["mp4"]), seconds=100.0)
            paths.atomic_json(os.path.join(self.wd, take["probe"]), {"problems": [], "duration_s": 5.0})
        g.update(status="voice_passed", track={"status": "passed", "wer": 0.0, "wer_inner": 0.0, "lost": 0.0})
        self.pipe_.save_gate(self.wd, g)
        wav(os.path.join(self.wd, "music", "bed.wav"), 1.0)
        done_meta(os.path.join(self.wd, "music", "bed.wav"))


@needs_ffmpeg
class S2MovedCaptions(H3Case):
    """Every scene moves its captions (caption_pos): the captions are looked for where they are drawn."""

    def setUp(self):
        super().setUp()
        h3 = h3_block()
        for sc in h3["scenes"].values():
            sc["caption_pos"] = [540, 420]
        home = make_home(self.tmp)
        self.pipe_ = self.pipe(home)
        self.wd = os.path.dirname(self.facts_job(self.pipe_, h3_ad(h3))["path"])
        self.ad = self.pipe_.load_ad(self.wd)
        _, errors = self.pipe_.write_script(self.wd, self.ad, self.pipe_.bible(self.wd),
                                            paths.read_json(os.path.join(self.wd, "facts.json")), "brain", None,
                                            Brain([json.dumps(PLAN)]))
        self.assertEqual(errors, "")
        S2Edit.setUp_gate(self)

    def test_an_ad_whose_every_beat_moves_its_captions_assembles(self):
        self.assertEqual(self.advance(self.pipe_, self.wd)["submit"][0]["kind"], "ad_assemble")
        S2Edit.assemble(self)
        asm = load(os.path.join(self.wd, "final", "assembly.json"))
        self.assertTrue(asm["burned_in"]["ok_captions"], asm["burned_in"])
        tl = load(os.path.join(self.wd, "final", "timeline.json"))
        times, rows = ad_pipeline.AdPipeline.h3_caption_samples(tl, 92)
        self.assertTrue(times)
        self.assertAlmostEqual(rows[0], 420 / 1920 - 0.01, places=4)
        self.assertAlmostEqual(rows[1], (420 + 1.6 * 92) / 1920 + 0.02, places=4)

    def test_caption_samples_prefer_the_bottom_band(self):
        tl = {"out": [1080, 1920], "hide": [[1.5, 2.0]],
              "beats": [{"caption_pos": [500, 300]}, {"caption_pos": None}],
              "words": [{"text": "a", "start": 0.1, "end": 0.3, "beat": 0}, {"text": "b", "start": 1.0, "end": 1.2, "beat": 1},
                        {"text": "c", "start": 1.6, "end": 1.8, "beat": 1}]}
        self.assertEqual(ad_pipeline.AdPipeline.h3_caption_samples(tl), ([1.1], None))
        tl["beats"][1]["caption_pos"] = [500, 900]
        self.assertEqual(ad_pipeline.AdPipeline.h3_caption_samples(tl)[0], [0.2], "the first moved beat's words")
        tl["words"] = tl["words"][2:]
        self.assertEqual(ad_pipeline.AdPipeline.h3_caption_samples(tl), ([], None), "under a card: nothing to sample")


# ---------------------------------------------------------------- S3 the gate's record

class S3GateRecord(H3Case):
    def test_summary_note_and_the_reviews_note(self):
        pipe, wd, ad = self.ready()
        script = paths.read_json(os.path.join(wd, "script.json"))
        g = pipe.h3_gate(wd, ad, script)
        self.assertEqual(pipe.gate_note(pipe.gate_summary(wd)),
                         "speech gate not finished (the finished ad was not heard end to end)")
        g.update(status="failed", reason="speech gate: beat 2 (body) never passed after 3 take(s): x")
        pipe.save_gate(wd, g)
        self.assertEqual(pipe.gate_note(pipe.gate_summary(wd)),
                         "speech gate failed: speech gate: beat 2 (body) never passed after 3 take(s): x")
        g.update(status="passed", mix={"status": "passed", "wer": 0.03, "lost": 0.0, "heard": "all of it"},
                 final={"round": 0, "safe": False, "beats": {"b0": {"ok": True}}})
        pipe.save_gate(wd, g)
        s = pipe.gate_summary(wd)
        self.assertEqual((s["mode"], s["beats"], s["takes"], s["retakes"], s["heard"]), ("h3", 4, 4, 0, "all of it"))
        self.assertEqual(pipe.gate_note(s), "speech gate passed (4 beats, 0 retakes; final lines heard intact; mix WER "
                                            "0.03)")
        os.makedirs(os.path.join(wd, "final"), exist_ok=True)
        paths.atomic_json(os.path.join(wd, "final", "qa.json"), {"ok": True, "checks": [], "frames": [], "hero": None})
        reply = json.dumps({"verdict": "accept", "facts_ok": True, "character_ok": True, "problems": [],
                            "note": "x" * 700})
        d = pipe.review(wd, "ads/ad-h3", {}, "brain", None, Brain([reply]))
        self.assertTrue(d["note"].endswith(ad_pipeline.H3_PUBLISH_NOTE), d["note"])
        self.assertLessEqual(len(d["note"]), 600)
        self.assertEqual(pipe.gate_record(wd)["final"]["round"], 0)


# ---------------------------------------------------------------- S4 retention and progress

class S4Retention(H3Case):
    def test_an_alternates_sidecars_go_with_it_and_nothing_else(self):
        pipe = ad_pipeline.AdPipeline(artifacts_dir=os.path.join(self.tmp, "art"))
        wd = pipe.workdir("ads/r")
        files = ["takes/beat-1-t0.mp4", "takes/beat-1-t0.mp4.meta.json", "takes/beat-1-t0.wav",
                 "takes/beat-1-t0.probe.json", "gate/beat-1-t0.asr.json", "gate/beat-1-t0.asr.json.meta.json",
                 "gate/beat-1-t0.asr0.json", "gate/beat-1-t0.asr0.json.meta.json"]
        kept = ["takes/beat-1-t1.mp4", "takes/beat-1-t1.wav", "takes/beat-1-t10.wav", "takes/beat-1-t0-notes.txt",
                "gate/beat-1-t1.asr.json", "gate/beat-1-t10.asr0.json"]
        for rel in files + kept:
            os.makedirs(os.path.dirname(os.path.join(wd, rel)), exist_ok=True)
            with open(os.path.join(wd, rel), "w") as f:
                f.write("x" * 10)
        paths.atomic_json(os.path.join(wd, "pick.json"), {"1": {"take": "takes/beat-1-t1.mp4", "seed": 1, "score": None,
                                                                "takes": [], "alternates": ["takes/beat-1-t0.mp4"]}})
        self.assertEqual(pipe.prune_alternates(wd), 10 * len(files))
        self.assertEqual([f for f in files if os.path.exists(os.path.join(wd, f))], [])
        self.assertEqual([f for f in kept if not os.path.exists(os.path.join(wd, f))], [])
        self.assertEqual(pipe.prune_alternates(wd), 0, "idempotent")

    def test_progress_follows_the_h3_stages(self):
        pipe, wd, ad = self.ready()
        pr = ad_pipeline.progress(wd)
        self.assertEqual(list(pr["done"]), list(ad_pipeline.H3_STAGES))
        self.assertEqual(pr["current"], "takes")
        os.makedirs(os.path.join(wd, "takes"))
        with open(os.path.join(wd, "takes", "beat-0-t0.mp4"), "wb") as f:
            f.write(b"x")
        self.assertEqual(ad_pipeline.progress(wd)["current"], "probe")
        with open(os.path.join(wd, "takes", "beat-0-t0.probe.json"), "w") as f:
            f.write("{}")
        pr = ad_pipeline.progress(wd)
        self.assertEqual((pr["current"], pr["takes"]), ("hear", ["beat-0-t0.mp4"]))


# ---------------------------------------------------------------- S5 the worker's runner and process groups

class S5Worker(H3Case):
    def test_exit_codes_map_to_failure_classes(self):
        pipe = self.pipe(make_home(self.tmp))
        wd = pipe.workdir("ads/w")
        cases = [({"exit": 2, "error": "a cutaway word was never heard", "class": "content"}, ContentFailure, None),
                 ({"exit": 3, "error": "ffmpeg not found: '/x'", "class": "infra"}, InfraFailure, "engine_unconfigured"),
                 ({"exit": 3, "error": "take missing: takes/beat-0-t0.mp4", "class": "infra"}, InfraFailure,
                  "engine_error"),
                 ({"exit": 1, "error": "KeyError: 'x'"}, RuntimeError, None),
                 ({"stdout": "not json", "exit": 0}, RuntimeError, None)]
        for ctl, cls, reason in cases:
            with self.subTest(ctl):
                self.control(ctl)
                with self.assertRaises(cls) as e:
                    pipe.h3_worker(Ctx(), wd, "probe", {"takes": []}, 15.0)
                if reason:
                    self.assertEqual(e.exception.reason, reason)
                self.assertNotIsInstance(e.exception, ContentFailure if cls is not ContentFailure else InfraFailure)
        self.control({"stdout": json.dumps({"ok": True, "error": None, "class": None, "takes": []}), "exit": 0})
        ctx = Ctx()
        self.assertEqual(pipe.h3_worker(ctx, wd, "probe", {"takes": []}, 15.0)["takes"], [])
        self.assertTrue(ctx.runs[0]["group"])
        self.assertEqual(ctx.runs[0]["cmd"][:4], [self.fake, "-I", ad_pipeline.H3_WORKER, "probe"])
        self.assertEqual(paths.read_json(os.path.join(wd, "h3", "probe-req.json")), {"takes": []})


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


GRANDCHILD = r'''
import subprocess, sys, time
p = subprocess.Popen(["sleep", "30"])
open(sys.argv[1], "w").write(str(p.pid))
time.sleep(float(sys.argv[2]))
'''


class S5ProcessGroup(TmpCase):
    def engine(self, handler):
        return LocalEngine(studio.catalog(False, ads=True), {"ad_probe": handler},
                           artifacts_dir=os.path.join(self.tmp, "art"), name=studio.LOCAL_ENGINE)

    def test_a_cancel_stops_the_workers_children_too(self):
        pidfile = os.path.join(self.tmp, "grandchild.pid")
        started = {}

        def handler(spec, ctx):
            started["ctx"] = ctx
            ctx.run([sys.executable, "-c", GRANDCHILD, pidfile, "30"], timeout=60, group=True)
            return {"path": pidfile}
        eng = self.engine(handler)
        eng.load(studio.LOCAL)
        h = eng.start(RunSpec("9.1", 9, "ad_probe", studio.LOCAL, {}, 30))
        self.assertTrue(wait_for(lambda: os.path.isfile(pidfile) and text_of(pidfile), 10))
        grandchild = int(text_of(pidfile))
        child = started["ctx"]._run.procs[0].pid
        self.assertTrue(_alive(grandchild))
        eng.cancel(h)
        self.assertTrue(wait_for(lambda: eng.poll(h).status == "cancelled", 15))
        self.assertTrue(wait_for(lambda: not _alive(grandchild) and not _alive(child), 10),
                        "the worker and its grandchild are both gone")

    def test_what_a_finished_worker_left_running_is_stopped(self):
        pidfile = os.path.join(self.tmp, "left.pid")
        eng = self.engine(lambda spec, ctx: {"rc": ctx.run([sys.executable, "-c", GRANDCHILD, pidfile, "0"],
                                                           timeout=60, group=True).returncode, "path": pidfile})
        eng.load(studio.LOCAL)
        h = eng.start(RunSpec("9.2", 9, "ad_probe", studio.LOCAL, {}, 30))
        self.assertTrue(wait_for(lambda: eng.poll(h).status == "done", 15), eng.poll(h))
        self.assertTrue(wait_for(lambda: not _alive(int(text_of(pidfile))), 10))


# ---------------------------------------------------------------- S6 ComfyUI

class _Refuse(http.server.BaseHTTPRequestHandler):
    body = b"{}"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        pass


class S6Comfy(TmpCase):
    def engine(self, port=8188, **kw):
        comfy = os.path.join(self.tmp, "ComfyUI")
        os.makedirs(os.path.join(comfy, ".venv", "bin"), exist_ok=True)
        with open(os.path.join(comfy, "main.py"), "w") as f:
            f.write("")
        py = os.path.join(comfy, ".venv", "bin", "python")
        with open(py, "w") as f:
            f.write("")
        gd = os.path.join(self.tmp, "graphs")
        os.makedirs(gd, exist_ok=True)
        shutil.copy(EXAMPLE_GRAPH, os.path.join(gd, "h3.json"))
        return comfyui.ComfyUIEngine(studio.catalog(False, ads=True), comfy_dir=comfy, port=port, graphs_dir=gd,
                                     artifacts_dir=os.path.join(self.tmp, "art"), home=os.path.join(self.tmp, "home"),
                                     name=studio.COMFY_ENGINE, **kw), py

    def test_comfyui_starts_without_foremans_pythonpath_and_with_its_venv_on_path(self):
        seen = {}

        class P:
            pid = 999999

        def popen(cmd, **kw):
            seen.update(kw)
            return P()
        eng, py = self.engine(popen=popen)
        old = dict(os.environ)
        os.environ.update(PYTHONPATH="/somewhere/foreman", PATH="/usr/bin:/bin", FAKE_KNOB="1")
        try:
            eng._spawn()
        finally:
            os.environ.clear()
            os.environ.update(old)
        env = seen["env"]
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["PATH"].split(os.pathsep), [os.path.dirname(py), "/usr/bin", "/bin"])
        self.assertEqual(env["FAKE_KNOB"], "1", "the rest of the environment is kept")
        self.assertTrue(seen["start_new_session"])

    def test_a_refusal_for_a_missing_model_file_or_node_is_the_machines(self):
        lora = {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                "node_errors": {"145": {"class_type": "LoraLoaderModelOnly", "dependent_outputs": ["92"], "errors": [
                    {"type": "value_not_in_list", "message": "Value not in list",
                     "details": "lora_name: 'x.safetensors' not in ['y.safetensors']",
                     "extra_info": {"input_name": "lora_name", "received_value": "x.safetensors"}}]}}}
        node = {"error": {"type": "missing_node_type", "message": "Node 'Reference to video' not found. The custom node "
                                                                  "may not be installed.", "details": "Node ID '#136'"},
                "node_errors": {}}
        old = {"error": {"type": "invalid_prompt", "message": "Cannot execute because node SolAttnPatch does not exist.",
                         "details": "Node ID '#154'"}, "node_errors": {}}
        sampler = {"error": {"type": "prompt_outputs_failed_validation", "message": "x"}, "node_errors": {"123": {
            "class_type": "KSamplerSelect", "errors": [{"type": "value_not_in_list", "message": "Value not in list",
                                                        "details": "sampler_name: 'eulr' not in ['euler']",
                                                        "extra_info": {"input_name": "sampler_name",
                                                                       "received_value": "eulr"}}]}}}
        image = {"error": {"type": "prompt_outputs_failed_validation", "message": "x"},
                 "node_errors": {"137": {"errors": [{"message": "Invalid image file: a.png"}]}}}
        for body, missing in ((lora, True), (node, True), (old, True), (sampler, False), (image, False)):
            with self.subTest(body["error"]["type"]):
                self.assertEqual(bool(comfyui.missing_install(json.dumps(body).encode())), missing)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Refuse)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            eng, _ = self.engine(port=srv.server_address[1])
            run = comfyui._Run("h", RunSpec("1.1", 1, "ad_shots", studio.COMFYUI, {}, 60))
            op = {"op": "graph", "graph": "h3", "params": {"prompt": "p", "seed": 1, "seconds": 4.0, "width": 576,
                                                           "height": 1024, "image_1": "a", "image_2": "b",
                                                           "image_3": "c", "audio_1": "d"}}
            for body, cls in ((lora, comfyui.EngineTrouble), (sampler, comfyui.RunRefused)):
                _Refuse.body = json.dumps(body).encode()
                with self.assertRaises(cls):
                    eng._run_graph(run, op, os.path.join(self.tmp, "out.mp4"), self.tmp, 0.0, 1.0, 60.0, 0)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_the_graph_wrapper_carries_its_attribution(self):
        g = comfyui.load_graph(os.path.dirname(EXAMPLE_GRAPH), "h3_ref2va.example")
        self.assertEqual((g["attribution"], g["commercial_ok"], g["outputs"]), ("MiniMax H3", False, ["92"]))
        self.assertIn("Excluded Territories", g["license"])
        self.assertIsNone(comfyui.load_graph(os.path.join(EXAMPLES, "comfyui", "graphs"), "character_still")
                          ["attribution"])


# ---------------------------------------------------------------- S7 characters and the examples

class S7CharactersAndExamples(TmpCase):
    def test_any_voice_may_name_a_reference_recording(self):
        home = make_home(self.tmp)
        b = characters.load_bible(home, "presenter-a")
        self.assertEqual(b.voice["engine"], "kokoro")
        self.assertIsNone(b.ref_wav(), "Kokoro never reads it")
        self.assertEqual(b.voice_ref(), os.path.join(b.root, "voice", "reference.wav"))
        data = load(os.path.join(b.root, "bible.json"))
        data["voice"]["ref_wav"] = "voice/absent.wav"
        with self.assertRaises(characters.BibleError):
            characters.validate(data, b.root)
        data["voice"]["ref_wav"] = "../outside.wav"
        with self.assertRaises(characters.BibleError):
            characters.validate(data, b.root)
        del data["voice"]["ref_wav"]
        self.assertIsNone(characters.Bible(b.root, characters.validate(data, b.root)).voice_ref())

    def test_the_h3_examples_are_generic_and_hold_together(self):
        templates, errors = load_templates(os.path.join(EXAMPLES, "templates"))
        self.assertEqual(errors, [])
        t = templates["ads.example-h3.v1"]
        self.assertEqual((t.kind, t.active, t.ad["ai_label"], t.commercial), ("ad", False, False, True))
        ad = dict(t.ad, id="ad-x", template_id=t.id, seconds=20.0, shots=4, seed=5, brief="b",
                  length={"min": 15.0, "max": 30.0}, shot_range={"min": 4, "max": 5}, commercial=t.commercial)
        with open(os.path.join(EXAMPLES, "assets", "facts-h3.json")) as f:
            text = json.dumps(json.load(f), indent=1)
        typed = ads.typed_numbers(text)
        facts = {"text": text, "numbers": sorted(ads.numbers_in(text)), "money": typed["money"],
                 "percent": typed["percent"]}
        self.assertEqual(h3core.h3_facts_problems(ad["h3"], facts, ad), [])
        root = os.path.join(EXAMPLES, "characters", "presenter-example")
        with open(os.path.join(root, "bible.json")) as f:
            bible = json.load(f)
        tags = [tg for s in bible["stills"] for tg in s["tags"]]
        script, errs = h3core.validate_h3_script(ad["h3"]["example"], ad, facts, tags, ads.AdsConfig())
        self.assertEqual(errs, [])
        self.assertEqual(bible["voice"]["ref_wav"], "voice/reference.wav")
        g = comfyui.load_graph(os.path.join(EXAMPLES, "comfyui", "graphs"), "h3_ref2va.example")
        text = json.dumps(g["graph"])
        for name in ad_pipeline.H3_PLACEHOLDERS + ("width", "height", "lora"):
            self.assertIn("{{" + name + "}}", text)
        self.assertEqual(g["defaults"], {"lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"})
        self.assertNotIn("ComfySwitchNode\", \"inputs\": {\"switch\": true", text, "constant switches are folded")
        out = comfyui.render_graph(g["graph"], dict(g["defaults"], prompt="p", image_1="a.png", image_2="b.png",
                                                    image_3="c.png", audio_1="v.wav", seed=7, seconds=4.4, width=576,
                                                    height=1024))
        self.assertEqual((out["136"]["inputs"]["width"], out["132"]["inputs"]["value"], out["901"]["inputs"]["audio"]),
                         (576, 4.4, "v.wav"))
        self.assertEqual(out["136"]["inputs"]["ref_images.ref_image_2"], ["902", 0])
        self.assertEqual(out["136"]["inputs"]["ref_audios.ref_audio_0"], ["901", 0])
        with tempfile_dir() as tmp:
            shutil.copy(os.path.join(EXAMPLES, "config.ads.json"), os.path.join(tmp, "config.json"))
            cfg = studio.load_config(os.path.join(tmp, "config.json"), env={"FOREMAN_HOME": tmp})
            self.assertEqual((cfg.ads_config().h3_graph, cfg.ads_config().media_python),
                             ("h3_ref2va", "~/ComfyUI/.venv/bin/python"))


class tempfile_dir:
    def __enter__(self):
        import tempfile
        self.d = tempfile.mkdtemp(prefix="foreman-h3-")
        return self.d

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
