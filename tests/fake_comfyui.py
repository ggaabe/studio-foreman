"""A small imitation of ComfyUI 0.43's HTTP API, for the ComfyUI engine's tests.

Endpoints: GET /system_stats, POST /prompt {"prompt", "client_id"} -> {"prompt_id"}, GET /queue,
POST /queue {"delete": [...]}, GET /history/<id>, GET /view?filename=&subfolder=&type=,
POST /upload/image (multipart "image", "overwrite"), POST /interrupt, POST /free.

A queued prompt "runs" on a worker thread for `run_s` seconds and then writes one PNG per output
node (a node whose class_type starts with Save or Preview), sized by the first width/height inputs
found in the graph. A LoadImage or LoadAudio node must name an uploaded file (else /prompt answers
400 with node_errors, as ComfyUI does). Behaviours per prompt: "ok", "error" (an execution error in
the history), "hang" (runs until interrupted).

A video output node (SaveVideo, VHS_VideoCombine) gets a talking take the way an H3 Ref2VA graph
makes one, encoded by ffmpeg (FAKE_COMFY_FFMPEG, else the one on PATH): the graph's width/height
(scaled to at most 256 px tall, even), 24 fps, 17k+5 frames from its seconds (the PrimitiveFloat),
a blue checkerboard slowly changing colour (never skin-coloured; a punch-in of it is a cut), bt709
tags, and AAC stereo at 32 kHz holding one tone burst per word of the line in the prompt's
`<d>[Language] ...</d>`, >= 0.16 s of silence between words and 0.3 s more after a comma or a full
stop inside the line. Answered as core SaveVideo does: {"images": [{filename .mp4, subfolder
"video"}], "animated": [true]}, /view as video/mp4. Every take has a tone of its own, and the
registry (a JSON file, FAKE_COMFY_REGISTRY) records what each take said and when: {"rules": [...],
"used": {rule index: takes it changed}, "takes": [{"i", "freq", "spoken", "said", "tokens": [[word,
start, end]], "seed", "seconds", "frames", "duration", "size", "acts"}]}. The test writes the rules;
a rule {"match": text in the spoken line (any case; "" matches every take), "times": takes it
changes at most (null: all), "do": ...} makes the take misspeak ("say": {written: said}), add a
word after the line ("after": word, 0.08 s after its last word: a run-on) or before it ("before":
word), show a skin-coloured block in the bottom band for its first 1.2 s ("hands"), go silent
("silent") or have no audio stream ("noaudio").

Run as a script it is a stand-in ComfyUI process: `python fake_comfyui.py --listen H --port P`
(a test writes a main.py that calls main()). FAKE_COMFY_IGNORE_TERM=1 makes it ignore SIGTERM.
"""
from __future__ import annotations

import array
import collections
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from foreman.media import encode_png  # noqa: E402

try:
    from fake_mlx_serve import parse_multipart  # noqa: E402
except ImportError:  # pragma: no cover
    from tests.fake_mlx_serve import parse_multipart  # noqa: E402


H3_LINE = re.compile(r"<d>\[[^\]]*\]\s*(.*?)\s*</d>", re.S)
VIDEO_SAVERS = ("SaveVideo", "VHS_VideoCombine")
H3_RATE = 32000
LEAD, GAP, PAUSE, TAIL, RUNON = 0.25, 0.16, 0.30, 0.40, 0.08
SKIN = "0xC88C64"


def h3_frames(seconds: float) -> int:
    """Frames an H3 graph renders for `seconds` (its node 131: at least 5, rounded up to 17k+5)."""
    n = max(5, int(round(seconds * 24)))
    return n + (5 - n % 17) % 17


def _syllables(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", word.lower())))


def take_spec(graph: dict) -> dict:
    """What a talking-take graph asks for: the spoken line, seconds, seed and render size."""
    spec = {"spoken": None, "seconds": None, "seed": None, "width": None, "height": None}
    for node in graph.values():
        ins = node.get("inputs") or {}
        for v in ins.values():
            m = H3_LINE.search(v) if isinstance(v, str) and spec["spoken"] is None else None
            if m:
                spec["spoken"] = " ".join(m.group(1).split())
        if node.get("class_type") == "PrimitiveFloat" and spec["seconds"] is None \
                and isinstance(ins.get("value"), (int, float)):
            spec["seconds"] = float(ins["value"])
        for k in ("noise_seed", "seed"):
            if spec["seed"] is None and isinstance(ins.get(k), int):
                spec["seed"] = ins[k]
        if spec["width"] is None and isinstance(ins.get("width"), int) and isinstance(ins.get("height"), int):
            spec["width"], spec["height"] = ins["width"], ins["height"]
    spec["spoken"] = spec["spoken"] or "hello world"
    spec["seconds"] = spec["seconds"] or 4.0
    w, h = spec["width"] or 144, spec["height"] or 256
    if h > 256:
        w, h = w * 256 / h, 256
    spec["size"] = [max(16, int(round(w / 2)) * 2), max(16, int(round(h / 2)) * 2)]
    return spec


def _said(spoken: str, acts: list) -> str:
    out = spoken
    for r in acts:
        for written, said in sorted((r.get("say") or {}).items(), key=lambda kv: -len(kv[0])):
            out = re.sub(r"(?i)(?<![\w])" + re.escape(written) + r"(?![\w])", lambda _m, w=said: w, out)
    return out


def layout(tokens: list, total: float, before=None, after=None) -> list:
    """[[word, start, end]]: one burst per word, its length by syllables, inside a take `total` s long."""
    pauses = sum(GAP + (PAUSE if t[-1] in ",.;:?!" else 0.0) for t in tokens[:-1])
    pre = 0.45 if before else 0.0
    room = total - LEAD - pre - TAIL - pauses - (RUNON + 0.2 if after else 0.0)
    per = max(0.05, min(0.30, room / max(1, sum(_syllables(t) for t in tokens))))
    out, t = [], LEAD
    if before:
        out.append([before, round(t, 3), round(t + 0.2, 3)])
        t += pre
    for k, tok in enumerate(tokens):
        d = per * _syllables(tok)
        out.append([tok, round(t, 3), round(t + d, 3)])
        t += d + GAP + (PAUSE if tok[-1] in ",.;:?!" and k < len(tokens) - 1 else 0.0)
    if after:
        s0 = out[-1][2] + RUNON
        out.append([after, round(s0, 3), round(s0 + 0.2, 3)])
    return out


def _bursts_wav(path: str, bursts: list, total: float, freq: float, amp: float) -> None:
    """32 kHz stereo s16: a sine at `freq` in each burst (8 ms fades), silence elsewhere."""
    n = int(round(total * H3_RATE))
    pcm = array.array("h", bytes(4 * n))
    fade = int(0.008 * H3_RATE)
    for _, s0, s1 in bursts:
        a, b = int(s0 * H3_RATE), min(n, int(s1 * H3_RATE))
        for i in range(a, b):
            k = min(i - a, b - 1 - i)
            g = 1.0 if k >= fade else 0.5 - 0.5 * math.cos(math.pi * k / fade)
            v = int(amp * 32767 * g * math.sin(2 * math.pi * freq * i / H3_RATE))
            pcm[2 * i] = pcm[2 * i + 1] = v
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(H3_RATE)
        w.writeframes(pcm.tobytes())


def render_take(graph: dict, registry: str | None, ffmpeg: str | None) -> bytes:
    """An H3-like talking take for `graph` (an MP4's bytes); the registry records what it says."""
    spec = take_spec(graph)
    reg = {}
    if registry and os.path.isfile(registry):
        with open(registry) as f:
            reg = json.load(f)
    rules, used, takes = reg.get("rules") or [], reg.get("used") or {}, reg.get("takes") or []
    acts = []
    for i, r in enumerate(rules):
        if str(r.get("match") or "").lower() not in spec["spoken"].lower():
            continue
        if r.get("times") is not None and int(used.get(str(i), 0)) >= int(r["times"]):
            continue
        used[str(i)] = int(used.get(str(i), 0)) + 1
        acts.append(r)
    dos = {r.get("do") for r in acts}
    said = _said(spec["spoken"], acts)
    frames = h3_frames(spec["seconds"])
    total = frames / 24.0
    before = next((r.get("before") for r in acts if r.get("before")), None)
    after = next((r.get("after") for r in acts if r.get("after")), None)
    tokens = layout(said.split(), total, before, after)
    k = len(takes)
    freq = 620.0 + 115.0 * (k % 20)
    w, h = spec["size"]
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    tmp = tempfile.mkdtemp(prefix="fake-h3-")
    try:
        wav, out = os.path.join(tmp, "voice.wav"), os.path.join(tmp, "take.mp4")
        _bursts_wav(wav, tokens, total, freq, 0.0 if "silent" in dos else 0.3)
        # a still checkerboard whose blue drifts slowly: a punch-in moves every edge (a cut), the
        # take itself never jumps
        m = "mod(floor(X/16)+floor(Y/16),2)"
        vf = (f"format=rgb24,geq=r='24+24*{m}':g='40+50*{m}':b='110+110*{m}+12*sin(2*PI*T/3)',format=yuv420p,"
              "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv")
        if "hands" in dos:
            vf += (f",drawbox=x={int(w * 0.3)}:y={int(h * 0.78)}:w={int(w * 0.4)}:h={int(h * 0.12)}:color={SKIN}@1:"
                   f"t=fill:enable='lt(t,1.2)'")
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
               f"color=c=black:s={w}x{h}:r=24:d={total + 1:.3f}"]
        if "noaudio" not in dos:
            cmd += ["-i", wav]
        cmd += ["-vf", vf, "-map", "0:v"] + (["-map", "1:a"] if "noaudio" not in dos else []) + [
            "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        cmd += (["-c:a", "aac", "-b:a", "96k", "-ar", str(H3_RATE), "-ac", "2"] if "noaudio" not in dos else ["-an"])
        cmd += ["-t", f"{total:.6f}", "-movflags", "+faststart", out]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(out, "rb") as f:
            data = f.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    takes.append({"i": k, "freq": freq, "spoken": spec["spoken"], "said": said,
                  "tokens": tokens, "line": [t[0] for t in tokens if t[0] not in (before, after)],
                  "seed": spec["seed"], "seconds": spec["seconds"], "frames": frames, "duration": round(total, 4),
                  "size": [w, h], "acts": sorted(d for d in dos if d), "wall": time.time()})
    if registry:
        reg.update(rules=rules, used=used, takes=takes)
        with open(registry + ".tmp", "w") as f:
            json.dump(reg, f, indent=1)
        os.replace(registry + ".tmp", registry)
    return data


class FakeComfy:
    def __init__(self, run_s: float = 0.2, host: str = "127.0.0.1", port: int = 0):
        self.run_s = run_s
        self.host, self.port = host, port
        self.inputs: dict = {}          # uploaded name -> bytes
        self.outputs: dict = {}         # filename -> bytes
        self.history: dict = {}
        self.queue: list = []           # [prompt_id] waiting or running
        self.running = None
        self.prompts: dict = {}
        self.behaviours = collections.deque()
        self.interrupted = threading.Event()
        self.freed = 0
        self.requests = []
        self.registry = os.environ.get("FAKE_COMFY_REGISTRY") or None   # the talking takes' registry (JSON)
        self.ffmpeg = os.environ.get("FAKE_COMFY_FFMPEG") or None
        self.lock = threading.RLock()
        self._server = None

    def start(self) -> "FakeComfy":
        fake = self

        class Handler(_Handler):
            state = fake

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._server.server_address[1]}"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def submit(self, graph: dict) -> tuple:
        errors = {}
        for nid, node in graph.items():
            if not isinstance(node, dict) or "class_type" not in node:
                errors[nid] = {"errors": [{"message": "node without class_type"}]}
                continue
            if node["class_type"] == "LoadImage":
                name = (node.get("inputs") or {}).get("image")
                if name not in self.inputs:
                    errors[nid] = {"errors": [{"message": f"Invalid image file: {name}"}]}
            if node["class_type"] == "LoadAudio":
                name = (node.get("inputs") or {}).get("audio")
                if name not in self.inputs:
                    errors[nid] = {"errors": [{"message": f"Invalid audio file: {name}"}]}
        if errors:
            return None, errors
        pid = uuid.uuid4().hex
        with self.lock:
            self.prompts[pid] = graph
            self.queue.append(pid)
            behaviour = self.behaviours.popleft() if self.behaviours else "ok"
        threading.Thread(target=self._run, args=(pid, graph, behaviour), daemon=True).start()
        return pid, None

    def _run(self, pid: str, graph: dict, behaviour: str) -> None:
        end = time.monotonic() + self.run_s
        with self.lock:
            self.running = pid
        while time.monotonic() < end or behaviour == "hang":
            if self.interrupted.is_set() or pid not in self.queue:
                with self.lock:
                    self.interrupted.clear()
                    if pid in self.queue:
                        self.queue.remove(pid)
                    self.history[pid] = {"outputs": {}, "status": {"status_str": "error", "completed": False,
                                                                   "messages": [["execution_interrupted", {}]]}}
                    self.running = None
                return
            time.sleep(0.02)
        with self.lock:
            if behaviour == "error":
                self.history[pid] = {"outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
                    ["execution_error", {"node_id": "1", "exception_message": "out of memory"}]]}}
            else:
                w = h = 64
                for node in graph.values():
                    ins = node.get("inputs") or {}
                    if isinstance(ins.get("width"), int) and isinstance(ins.get("height"), int):
                        w, h = max(8, min(ins["width"], 256)), max(8, min(ins["height"], 256))
                        break
                outs = {}
                for nid, node in graph.items():
                    if node.get("class_type") in VIDEO_SAVERS:
                        name = f"foreman_h3_{pid[:8]}_{nid}_.mp4"
                        try:
                            self.outputs[name] = render_take(graph, self.registry, self.ffmpeg)
                        except Exception as e:  # noqa: BLE001 - reported as ComfyUI reports a node's failure
                            self.history[pid] = {"outputs": {}, "status": {"status_str": "error", "completed": False,
                                                 "messages": [["execution_error", {"node_id": nid,
                                                                                   "exception_message": str(e)[:300]}]]}}
                            break
                        outs[nid] = {"images": [{"filename": name, "subfolder": "video", "type": "output"}],
                                     "animated": [True]}
                    elif str(node.get("class_type", "")).startswith(("Save", "Preview")):
                        name = f"out-{pid[:8]}-{nid}.png"
                        pix = bytes((x * 3 + y) % 256 for y in range(h) for x in range(w) for _ in range(3))
                        self.outputs[name] = encode_png(w, h, pix)
                        outs[nid] = {"images": [{"filename": name, "subfolder": "", "type": "output"}]}
                if pid in self.history:  # a node failed
                    if pid in self.queue:
                        self.queue.remove(pid)
                    self.running = None
                    return
                self.history[pid] = {"prompt": [0, pid, graph], "outputs": outs,
                                     "status": {"status_str": "success", "completed": True, "messages": []}}
            if pid in self.queue:
                self.queue.remove(pid)
            self.running = None


class _Handler(BaseHTTPRequestHandler):
    state: FakeComfy = None
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _raw(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        s = self.state
        url = urllib.parse.urlsplit(self.path)
        with s.lock:
            s.requests.append(("GET", url.path))
        if url.path == "/system_stats":
            return self._json(200, {"system": {"os": "darwin", "comfyui_version": "0.43.0", "python_version": "3.13"},
                                    "devices": [{"name": "mps", "type": "mps"}]})
        if url.path == "/queue":
            with s.lock:
                running = [[0, s.running, {}, {}, []]] if s.running else []
                pending = [[i, p, {}, {}, []] for i, p in enumerate(s.queue) if p != s.running]
            return self._json(200, {"queue_running": running, "queue_pending": pending})
        if url.path.startswith("/history/"):
            pid = urllib.parse.unquote(url.path[len("/history/"):])
            with s.lock:
                entry = s.history.get(pid)
            return self._json(200, {pid: entry} if entry else {})
        if url.path == "/view":
            q = urllib.parse.parse_qs(url.query)
            name = (q.get("filename") or [""])[0]
            with s.lock:
                data = s.outputs.get(name)
            if data is None:
                return self._json(404, {"error": "not found"})
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4" if name.endswith(".mp4") else "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return self._json(404, {"error": "unknown endpoint"})

    def do_POST(self):
        s = self.state
        url = urllib.parse.urlsplit(self.path)
        raw = self._raw()
        with s.lock:
            s.requests.append(("POST", url.path))
        if url.path == "/upload/image":
            fields, files = parse_multipart(self.headers.get("Content-Type", ""), raw)
            disp = raw.split(b"filename=\"", 1)
            name = disp[1].split(b"\"", 1)[0].decode() if len(disp) > 1 else uuid.uuid4().hex + ".png"
            if "image" not in files:
                return self._json(400, {"error": "no image"})
            with s.lock:
                s.inputs[name] = files["image"]
            return self._json(200, {"name": name, "subfolder": "", "type": "input"})
        try:
            body = json.loads(raw.decode() or "{}")
        except ValueError:
            return self._json(400, {"error": "bad json"})
        if url.path == "/prompt":
            graph = body.get("prompt")
            if not isinstance(graph, dict):
                return self._json(400, {"error": {"type": "invalid_prompt", "message": "no prompt"}, "node_errors": {}})
            pid, errors = s.submit(graph)
            if errors:
                return self._json(400, {"error": {"type": "prompt_outputs_failed_validation",
                                                  "message": "Prompt outputs failed validation"}, "node_errors": errors})
            return self._json(200, {"prompt_id": pid, "number": 1, "node_errors": {}})
        if url.path == "/interrupt":
            s.interrupted.set()
            return self._json(200, {})
        if url.path == "/queue":
            with s.lock:
                for pid in body.get("delete") or []:
                    if pid in s.queue:
                        s.queue.remove(pid)
            return self._json(200, {})
        if url.path == "/free":
            with s.lock:
                s.freed += 1
            return self._json(200, {})
        return self._json(404, {"error": "unknown endpoint"})


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--run-s", type=float, default=float(os.environ.get("FAKE_COMFY_RUN_S", "0.2")))
    args, _ = ap.parse_known_args(argv)
    if os.environ.get("FAKE_COMFY_IGNORE_TERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    delay = float(os.environ.get("FAKE_COMFY_START_DELAY", "0"))
    if delay:
        time.sleep(delay)
    fake = FakeComfy(run_s=args.run_s, host=args.listen, port=args.port).start()
    stop = threading.Event()
    if os.environ.get("FAKE_COMFY_IGNORE_TERM") != "1":
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.wait(0.2):
            pass
    finally:
        fake.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
