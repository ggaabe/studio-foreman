"""A small in-process imitation of mlx-serve v26.10.3's HTTP surface, for adapter tests.

It reproduces the shapes the adapter depends on (checked against the server source and the
live Studio): /api/ps (READY models, ':latest' names), /v1/models (+ machine.free_mib),
load by discovered id or absolute path (registered under the basename, or the discovered id
for a discovered pack's own directory; while a load runs, /v1/models lists the id with state
"loading": an assumption, not verified against the server), idempotent unload,
SSE progress / complete / error for speech, music and video, streaming chat completions, and
the fallback to the default model for an unknown id, and /metrics with
`mlx_serve:process_start_time_seconds` (restart() starts a new server process). Per-request
behaviours (hang, error, drop, http500, http400, black or frozen video) are scripted per endpoint.

For the ad pipeline (MLX-Serve 26.11.1 shapes): capabilities per model in /v1/models (audio,
music, video, image, transcription, alignment), speech with a Qwen3-TTS `ref_audio` clone,
multipart /v1/audio/transcriptions answering verbose_json with word times spread over the WAV
(asr_loads: ids it loads beside the ASR model, as mlx-serve may load the forced aligner),
non-streaming /v1/images/generations (edit mode, at most 4 ref_images), and the full video request
checked the way the server checks it (8k+1 frames, a 64 px grid for two-stage and 32 px for one
stage, audio-to-video only two-stage, at most 768 MiB of raw RGB, decodable keyframes and audio).

What the voice says and what the ASR hears are controllable, for the audio gate's tests. Each speech
request is an utterance with a tone of its own (so every take's samples differ), and the fake
remembers what it "said" in it: by default the input, or `say(input, body)`, or the input with the
words in `mispronounce` ({written: said}, whole words, any case) replaced. A transcription hears
the utterances whose samples it finds in the WAV, in order (a line, or the voice track made of
lines); audio it cannot place (the final mix, which ffmpeg re-made) is heard as the longest audio it
did place, passed through the next callable or text queued in `asr_unknown`; `asr_rewrite`
(a callable, or [(regex, replacement)]) then writes the text the way an ASR would (digits for
numbers, '.com'). `asr_text`, when set, is heard for every transcription (the older tests).
`asr_resolver(audio, fields)`, when set, is asked first: it answers (text, words or None), or None
to leave the transcription to the rules above. H3Ear is the one for H3 takes rendered by the fake
ComfyUI: it reads that fake's registry and hears each take by its tone (needs numpy).
"""
from __future__ import annotations

import base64
import collections
import io
import json
import math
import os
import re
import select
import socket
import struct
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def tone_wav(seconds: float, rate: int = 24000, freq: float = 440.0, silent_tail: float = 0.0) -> bytes:
    n = int(seconds * rate)
    quiet = int(silent_tail * rate)
    frames = bytearray()
    for i in range(n):
        v = 0 if i >= n - quiet else int(12000 * math.sin(2 * math.pi * freq * i / rate))
        frames += struct.pack("<h", v)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def video_frames(width: int, height: int, frames: int, mode: str = "ok") -> bytes:
    out = bytearray()
    for f in range(frames):
        if mode == "black":
            out += bytes(width * height * 3)
            continue
        shade = 0 if mode == "frozen" else f * 7
        row = bytearray()
        for x in range(width):
            row += bytes(((x * 3 + shade) % 256, (x + 2 * shade) % 256, (128 + shade) % 256))
        for y in range(height):
            out += bytes((b + y) % 256 for b in row) if mode != "frozen" else row
    return bytes(out)


def capabilities(mid: str, markers=()) -> list:
    """What /v1/models lists for a model id (by its name, like the real server's packs)."""
    low = mid.lower()
    if "ace-step" in low:
        return ["audio", "music"]
    if "ltx" in low:
        return ["video"]
    if "flux" in low or "klein" in low:
        return ["image"]
    if "aligner" in low:
        return ["alignment"]
    if "asr" in low:
        return ["transcription"]
    if any(m in low for m in markers):
        return ["audio"]
    return ["chat", "streaming", "vision"]


def parse_multipart(ctype: str, raw: bytes) -> tuple:
    """({field: value}, {file field: bytes}) of a multipart/form-data body."""
    m = __import__("re").search(r"boundary=([^;]+)", ctype or "")
    fields, files = {}, {}
    if not m:
        return fields, files
    boundary = b"--" + m.group(1).strip().strip('"').encode()
    for part in raw.split(boundary):
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if not part or part.startswith(b"--"):
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        disp = head.decode("latin-1")
        name = __import__("re").search(r'name="([^"]+)"', disp)
        if not name:
            continue
        if 'filename="' in disp:
            files[name.group(1)] = data
        else:
            fields[name.group(1)] = data.decode("utf-8", "replace")
    return fields, files


class FakeMlxServe:
    def __init__(self, sizes=None):
        self.sizes = dict(sizes or {})          # server id (or basename) -> bytes
        self.ready = collections.OrderedDict()  # server id -> bytes (what /api/ps shows)
        self.discovered = set()                 # ids /v1/models lists (discovery)
        self.loading = set()                    # ids a load-model is working on (/v1/models state "loading")
        # mlx-serve v26.10.3 started without --model: the latest chat-capable load is promoted to
        # the default (default_promoted), unless a load said "default": true (setDefault).
        self.default_id = ""
        self.default_promoted = False
        self.media_markers = ("kokoro", "ace-step", "ltx", "flux", "tts", "asr", "aligner", "klein")
        self.speech_rate = 0.06                 # seconds of speech per input character
        self.speech_cap = 3.0                   # ... at most this long
        self.music_cap = 3.0
        self.asr_text = None                    # what the fake "hears" (default: the input it last spoke)
        self.asr_lower = True                   # ASR writes words lower-cased (a URL comes back lower-case)
        self.asr_loads = ()                     # server ids a transcription loads beside the ASR (its aligner)
        self.last_speech = ""
        self.say = None                         # callable(input, body) -> what the voice says (default: the input)
        self.mispronounce = {}                  # {written word or phrase: what the voice says instead}
        self.utterances = []                    # [{"said", "input", "seed", "pcm", "format", "seconds"}]
        self.asr_unknown = collections.deque()  # per transcription of audio it cannot place: callable(text) or text
        self.asr_rewrite = None                 # callable(text) or [(regex, replacement)]: how the ASR writes
        self.asr_resolver = None                # callable(audio, fields) -> (text, words | None) | None: asked first
        self.heard = []                         # [(fields, text)] of every transcription
        self._placed = []                       # [(seconds, text)] of the audio a transcription placed
        self.image_size_cap = 256
        self.image_delay = 0.0
        self.video_requests = []                # summaries of the video requests (fields, decoded sizes)
        self.free_mib = 200000
        self.health_ok = True
        self.load_delay = 0.0
        self.load_error = None                  # (status, message) for the next loads
        self.unload_delay = 0.0
        self.unload_error = None
        self.script = collections.defaultdict(collections.deque)  # endpoint -> behaviours
        self.chat_replies = collections.deque()
        self.requests = []                      # (method, path, body)
        self.disconnects = []                   # endpoints whose client went away mid-stream
        self.progress_delay = 0.0
        self.steps = 6
        self.start_time = 1761456000            # mlx_serve:process_start_time_seconds (unix seconds)
        self.metrics = "text"                   # text: /metrics; json: only /metrics.json; none: neither
        self.lock = threading.Lock()
        self._server = None

    def restart(self, keep=()) -> None:
        """The server process restarts (an update, a crash, its app relaunching it): every model
        is gone (except `keep`, loaded again at startup as --model would) and the start time moves."""
        with self.lock:
            kept = {k: v for k, v in self.ready.items() if k in keep}
            self.ready.clear()
            self.ready.update(kept)
            self.start_time += 3600
            if self.default_id not in self.ready:
                self.default_id, self.default_promoted = "", False

    def other_client_load(self, mid: str, size: int | None = None) -> None:
        """Another client (MLX-Serve.app, a chat window) loads a model behind gpud's back."""
        with self.lock:
            self.ready[mid] = size or self.sizes.get(mid) or self.sizes.get(mid.rsplit("/", 1)[-1]) or int(1e9)

    # ---- lifecycle
    def start(self) -> "FakeMlxServe":
        fake = self

        class Handler(_Handler):
            server_state = fake

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def bodies(self, path: str) -> list:
        return [b for (m, p, b) in self.requests if p == path]

    def transcriptions(self) -> list:
        return [b for (m, p, b) in self.requests if p == "/v1/audio/transcriptions"]

    # ---- what the voice says, what the ASR hears
    def said(self, text: str, body: dict) -> str:
        if self.say is not None:
            return str(self.say(text, body))
        out = text
        for written, spoken in sorted(self.mispronounce.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(r"(?i)(?<![\w])" + re.escape(written) + r"(?![\w])", lambda _m, w=spoken: w, out)
        return out

    def speak(self, text: str, body: dict, seconds: float, silent: bool) -> bytes:
        with self.lock:
            k = len(self.utterances)
        wav = tone_wav(seconds, freq=180.0 + 11.0 * (k % 900), silent_tail=seconds if silent else 0.0)
        with wave.open(io.BytesIO(wav), "rb") as w:
            fmt, pcm = (w.getnchannels(), w.getsampwidth(), w.getframerate()), w.readframes(w.getnframes())
        with self.lock:
            self.utterances.append({"said": self.said(text, body), "input": text, "seed": body.get("seed"),
                                    "pcm": pcm, "format": fmt, "seconds": seconds})
        return wav

    def hear(self, audio: bytes, fields: dict) -> str:
        if self.asr_text is not None:
            return self.asr_text
        try:
            with wave.open(io.BytesIO(audio or b""), "rb") as w:
                fmt, pcm = (w.getnchannels(), w.getsampwidth(), w.getframerate()), w.readframes(w.getnframes())
                seconds = w.getnframes() / float(w.getframerate())
        except Exception:
            fmt, pcm, seconds = None, b"", 0.0
        with self.lock:
            known = [u for u in self.utterances if u["format"] == fmt and u["pcm"]]
        hits = []
        for u in known:
            block = u["format"][0] * u["format"][1]
            pos = pcm.find(u["pcm"])
            while pos >= 0 and pos % block:
                pos = pcm.find(u["pcm"], pos + 1)
            if pos >= 0:
                hits.append((pos, u["said"]))
        if hits:
            text = " ".join(said for _, said in sorted(hits))
            with self.lock:
                self._placed.append((seconds, text))
        else:
            with self.lock:
                placed = sorted(self._placed, key=lambda x: x[0])
            text = placed[-1][1] if placed else (self.last_speech or "hello world")
            with self.lock:
                hook = self.asr_unknown.popleft() if self.asr_unknown else None
            if callable(hook):
                text = hook(text)
            elif isinstance(hook, str):
                text = hook
        rw = self.asr_rewrite
        if callable(rw):
            text = rw(text)
        elif rw:
            for rx, repl in rw:
                text = re.sub(rx, repl, text)
        with self.lock:
            self.heard.append((dict(fields), text))
        return text


_PUNCT = ".,!?;:\"'"


def _bare(word: str) -> str:
    return word.strip(_PUNCT)


def _pcm_mono(audio: bytes):
    """(samples as float in [-1, 1], rate) of a s16 WAV, channels averaged; (None, None) if unreadable."""
    import numpy as np
    try:
        with wave.open(io.BytesIO(audio or b""), "rb") as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except Exception:
        return None, None
    if width != 2 or not raw:
        return None, None
    x = np.frombuffer(raw[:len(raw) // (2 * ch) * 2 * ch], dtype="<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(axis=1), rate


class H3Ear:
    """The fake ASR's ear for H3 takes: an asr_resolver that hears the takes the fake ComfyUI rendered.

    Each take holds one tone (its registry entry's `freq`), one burst per word it said. A recording is
    heard as the takes whose tones it holds, in the order they sound (a take's 16 kHz WAV, a beat's
    finished voice, the final mix with the music under it): their words as an ASR writes them (each
    `written` phrase, any case, as one word: {"example dot com": "example.com"}), joined. A word-timed
    request of one take gets each word's time from its bursts as heard (10 ms frames above -35 dBFS,
    as the probe measures voiced runs), else from the registry. `hook(info)` may rewrite what is heard:
    info {"takes": [registry entries], "words": bool, "n": how many times these takes were heard this
    way before, "text": str}; it returns the text (or None to keep it), for a free-running request."""

    def __init__(self, registry: str, written=None, hook=None, min_amp: float = 0.02):
        self.registry = registry
        self.written = sorted(((p.lower().split(), w) for p, w in (written or {}).items()), key=lambda r: -len(r[0]))
        self.hook = hook
        self.min_amp = min_amp
        self.counts = collections.Counter()
        self.log = []                           # [{"takes": [i], "words": bool, "n", "text"}]
        self.lock = threading.Lock()

    def takes(self) -> list:
        try:
            with open(self.registry) as f:
                return json.load(f).get("takes") or []
        except (OSError, ValueError):
            return []

    def __call__(self, audio: bytes, fields: dict):
        import numpy as np
        takes = self.takes()
        x, rate = _pcm_mono(audio)
        if not takes or x is None:
            return None
        found = self.find(np, x, rate, takes)
        if not found:
            return None
        timed = fields.get("timestamp_granularities[]") == "word"
        words = []
        for t in found:
            tokens = [list(tok) for tok in t["tokens"]]
            if timed and len(found) == 1:
                runs = self.voiced_runs(np, x, rate)
                if len(runs) == len(tokens):
                    for tok, (a, b) in zip(tokens, runs):
                        tok[1], tok[2] = a, b
            words += self.write(tokens)
        text = " ".join(w for w, _, _ in words)
        key = (tuple(t["i"] for t in found), timed)
        with self.lock:
            n = self.counts[key]
            self.counts[key] += 1
        if self.hook is not None and not timed:
            new = self.hook({"takes": found, "words": timed, "n": n, "text": text})
            text = text if new is None else str(new)
        with self.lock:
            self.log.append({"takes": [t["i"] for t in found], "words": timed, "n": n, "text": text})
        return text, ([(_bare(w), a, b) for w, a, b in words if _bare(w)] if timed else None)

    def write(self, tokens: list) -> list:
        out, i = [], 0
        while i < len(tokens):
            for phrase, w in self.written:
                seg = tokens[i:i + len(phrase)]
                if len(seg) == len(phrase) and [_bare(t[0]).lower() for t in seg] == phrase:
                    trail = seg[-1][0][len(seg[-1][0].rstrip(_PUNCT)):]
                    out.append((w + trail, seg[0][1], seg[-1][2]))
                    i += len(phrase)
                    break
            else:
                out.append(tuple(tokens[i]))
                i += 1
        return out

    def find(self, np, x, rate: int, takes: list) -> list:
        """The takes whose tones sound in x, in the order they first sound (40 ms windows, 10 ms hop)."""
        if rate % 16000 == 0 and rate > 16000:
            x, rate = x[::rate // 16000], 16000
        win, hop = int(0.04 * rate), int(0.01 * rate)
        if len(x) < win:
            return []
        nf = 1 + (len(x) - win) // hop
        frames = x[np.arange(win)[None, :] + hop * np.arange(nf)[:, None]] * np.hanning(win)[None, :]
        freqs = np.array([float(t["freq"]) for t in takes])
        basis = np.exp(-2j * np.pi * np.outer(np.arange(win), freqs) / rate)
        amp = np.abs(frames @ basis) * 2.0 / np.hanning(win).sum()
        best, top = amp.argmax(axis=1), amp.max(axis=1)
        first = {}
        for f in range(nf):
            if top[f] >= self.min_amp:
                k = int(best[f])
                first.setdefault(k, []).append(f)
        present = [(fs[0], k) for k, fs in first.items() if len(fs) >= 5]
        return [takes[k] for _, k in sorted(present)]

    @staticmethod
    def voiced_runs(np, x, rate: int, db: float = -35.0, min_s: float = 0.03) -> list:
        hop = int(0.01 * rate)
        n = len(x) // hop
        if not n:
            return []
        rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
        on = 20 * np.log10(rms) > db
        runs, start = [], None
        for f in range(n + 1):
            if f < n and on[f]:
                start = f if start is None else start
            elif start is not None:
                if (f - start) * 0.01 >= min_s - 1e-9:
                    runs.append((round(start * 0.01, 3), round(f * 0.01, 3)))
                start = None
        return runs


class _Handler(BaseHTTPRequestHandler):
    server_state: FakeMlxServe = None
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):  # quiet
        pass

    # ---- helpers
    def _json(self, status: int, obj) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        self._json(status, {"error": {"message": message, "type": kind, "param": None, "code": status}})

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _gone(self) -> bool:
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            if not r:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _send(self, obj) -> bool:
        if self._gone():
            return False
        try:
            self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
            self.wfile.flush()
            return True
        except OSError:
            return False

    def _wait_gone(self, limit: float = 20.0) -> bool:
        end = time.monotonic() + limit
        while time.monotonic() < end:
            if self._gone():
                return True
            time.sleep(0.02)
        return False

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode()) if raw else {}
        except ValueError:
            return {"_raw": raw.decode("utf-8", "replace")}

    # ---- routes
    def do_GET(self):
        s = self.server_state
        with s.lock:
            s.requests.append(("GET", self.path, None))
        if self.path == "/health":
            return self._json(200, {"status": "ok"}) if s.health_ok else self._json(503, {"status": "loading"})
        if self.path == "/metrics":
            if s.metrics != "text":
                return self._error(404, "Unknown endpoint", "not_found")
            text = ("# HELP mlx_serve:requests_running Requests running\n# TYPE mlx_serve:requests_running gauge\n"
                    "mlx_serve:requests_running 0\n"
                    "# HELP mlx_serve:process_start_time_seconds Unix time the server process started\n"
                    "# TYPE mlx_serve:process_start_time_seconds gauge\n"
                    f"mlx_serve:process_start_time_seconds {s.start_time}\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            self.wfile.write(text)
            return
        if self.path == "/metrics.json":
            if s.metrics == "none":
                return self._error(404, "Unknown endpoint", "not_found")
            return self._json(200, {"gauges": {"process_start_time_seconds": s.start_time, "requests_running": 0}})
        if self.path == "/api/ps":
            with s.lock:
                models = [{"name": f"{k}:latest", "model": f"{k}:latest", "size": v, "size_vram": v,
                           "expires_at": "9999-01-01T00:00:00.000Z", "details": {"family": "x"}}
                          for k, v in s.ready.items()]
            return self._json(200, {"models": models})
        if self.path == "/v1/models":
            with s.lock:
                ids = sorted(set(s.ready) | s.discovered)
                if s.default_id in ids:  # mlx-serve lists its default first
                    ids.remove(s.default_id)
                    ids.insert(0, s.default_id)
                ids += sorted(s.loading - set(ids))
                data = [{"id": i, "object": "model", "loaded": i in s.ready,
                         "state": "ready" if i in s.ready else "loading" if i in s.loading else "unloaded",
                         "capabilities": capabilities(i, s.media_markers)} for i in ids]
            return self._json(200, {"object": "list", "data": data,
                                    "machine": {"os": "macos", "free_mib": s.free_mib, "total_mib": 262144}})
        return self._error(404, "Unknown endpoint", "not_found")

    def do_POST(self):
        s = self.server_state
        if self.path == "/v1/audio/transcriptions":
            return self._transcribe()
        body = self._body()
        with s.lock:
            s.requests.append(("POST", self.path, body))
        if self.path == "/v1/images/generations":
            return self._image(body)
        if self.path == "/v1/load-model":
            return self._load(body)
        if self.path == "/v1/unload-model":
            return self._unload(body)
        if self.path == "/v1/chat/completions":
            return self._chat(body)
        if self.path in ("/v1/audio/speech", "/v1/audio/music-generations", "/v1/video/generations"):
            return self._media(self.path, body)
        return self._error(404, "Unknown endpoint", "not_found")

    def _load(self, body):
        s = self.server_state
        target = body.get("model")
        if not isinstance(target, str) or not target:
            return self._error(400, "'model' must be a string")
        if target.startswith("/"):
            # registerByPath: the directory of a discovered pack resolves to its discovered id;
            # any other directory registers under its basename
            own = [d for d in s.discovered if target.rstrip("/").endswith("/" + d)]
            mid = own[0] if own else os.path.basename(target.rstrip("/"))
        elif target in s.discovered or target in s.ready:
            mid = target
        else:
            return self._error(404, "Unknown model id", "model_not_found")
        if s.load_delay:
            with s.lock:
                s.loading.add(mid)
            try:
                time.sleep(s.load_delay)
            finally:
                with s.lock:
                    s.loading.discard(mid)
        if s.load_error:
            return self._error(s.load_error[0], s.load_error[1], "model_load_failed")
        size = s.sizes.get(mid) or s.sizes.get(target) or int(1e9)
        with s.lock:
            s.ready[mid] = size
            chat = not any(m in mid.lower() for m in s.media_markers)
            if body.get("default") is True:
                s.default_id, s.default_promoted = mid, False
            elif chat and (not s.default_id or s.default_promoted) and s.default_id != mid:
                s.default_id, s.default_promoted = mid, True
        return self._json(200, {"model": {"id": mid, "object": "model", "loaded": True, "state": "ready",
                                          "bytes_resident": size, "bytes_on_disk": size}})

    def _unload(self, body):
        s = self.server_state
        mid = body.get("model")
        if not isinstance(mid, str) or not mid or mid == "mlx-serve":
            with s.lock:
                s.requests.append(("DANGER", "unload of the default model", body))
            return self._error(400, "test server: an empty id would unload the default model")
        if s.unload_delay:
            time.sleep(s.unload_delay)
        if s.unload_error:
            return self._error(s.unload_error[0], s.unload_error[1], "model_unload_failed")
        with s.lock:
            known = mid in s.ready or mid in s.discovered
            s.ready.pop(mid, None)
        if not known:
            return self._error(404, "Unknown model id", "model_not_found")
        return self._json(200, {"model": {"id": mid, "object": "model", "loaded": False, "state": "unloaded"}})

    def _resident_or_fallback(self, body) -> bool:
        s = self.server_state
        with s.lock:
            ok = body.get("model") in s.ready
        if not ok:  # mlx-serve falls back to its default (a chat model) for an unknown id
            self._error(400, "Target model does not support this modality. Load the matching image/audio/"
                             "video/3D/decision model and target it by id.")
        return ok

    def _transcribe(self):
        s = self.server_state
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        fields, files = parse_multipart(self.headers.get("Content-Type", ""), raw)
        with s.lock:
            s.requests.append(("POST", self.path, dict(fields, file_bytes=len(files.get("file", b"")))))
        if not self._resident_or_fallback({"model": fields.get("model")}):
            return
        with s.lock:
            for mid in s.asr_loads:  # mlx-serve loading the forced aligner for word times
                s.ready.setdefault(mid, int(1e9))
        audio = files.get("file")
        try:
            with wave.open(io.BytesIO(audio or b""), "rb") as w:
                seconds = w.getnframes() / float(w.getframerate())
        except Exception:
            return self._error(400, "file must be a WAV")
        resolved = s.asr_resolver(audio, fields) if s.asr_resolver is not None else None
        if resolved is not None:  # the resolver writes as a real ASR does: case, punctuation, its own word times
            text, timed = resolved
            with s.lock:
                s.heard.append((dict(fields), text))
            out = {"task": "transcribe", "language": fields.get("language") or "en", "duration": seconds,
                   "text": text, "segments": [{"id": 0, "seek": 0, "start": 0.0, "end": seconds, "text": text}]}
            if fields.get("timestamp_granularities[]") == "word" and timed is not None:
                out["words"] = [{"word": w, "start": round(a, 3), "end": round(b, 3)} for w, a, b in timed]
            return self._json(200, out)
        text = s.hear(audio, fields)
        words = [w for w in text.split() if w]
        if s.asr_lower:
            words = [w.lower() for w in words]
        out = {"task": "transcribe", "language": fields.get("language") or "en", "duration": seconds,
               "text": " ".join(words), "segments": [{"id": 0, "start": 0.0, "end": seconds, "text": " ".join(words)}]}
        if fields.get("timestamp_granularities[]") == "word" and words:
            lead, span = 0.05 * seconds, 0.9 * seconds
            step = span / len(words)
            out["words"] = [{"word": w.strip(".,!?"), "start": round(lead + i * step, 3),
                             "end": round(lead + (i + 0.8) * step, 3)} for i, w in enumerate(words)]
        return self._json(200, out)

    def _image(self, body):
        s = self.server_state
        if not self._resident_or_fallback(body):
            return
        with s.lock:
            behaviour = s.script["/v1/images/generations"].popleft() if s.script["/v1/images/generations"] else "ok"
        if behaviour == "http500":
            return self._error(500, "generation failed", "server_error")
        if s.image_delay:
            time.sleep(s.image_delay)
        refs = body.get("ref_images") or []
        if len(refs) > 4:
            return self._error(400, "ref_images: this model takes at most 4 reference images")
        for b in ([body["image"]] if body.get("image") else []) + list(refs):
            try:
                base64.b64decode(b, validate=True)
            except Exception:
                return self._error(400, "image is not base64")
        if body.get("mode") not in (None, "edit", "variation"):
            return self._error(400, "mode must be edit or variation")
        try:
            w, h = (int(x) for x in str(body.get("size") or "64x64").split("x"))
        except ValueError:
            return self._error(400, "size must be WxH")
        w, h = max(8, min(w, s.image_size_cap)), max(8, min(h, s.image_size_cap))
        seed = int(body.get("seed") or 0)
        pix = bytes(((x * 7 + seed) % 256 if c == 0 else (y * 5 + seed) % 256 if c == 1 else (x + y) % 256)
                    for y in range(h) for x in range(w) for c in range(3))
        from foreman.media import encode_png
        png = encode_png(w, h, pix)
        return self._json(200, {"created": 0, "data": [{"b64_json": base64.b64encode(png).decode()}]})

    def _video_problem(self, body):
        w, h, n = body.get("width", 384), body.get("height", 256), body.get("num_frames", 9)
        pipeline = body.get("pipeline") or "one_stage"
        if not all(isinstance(v, int) for v in (w, h, n)):
            return "width, height and num_frames must be integers"
        if (n - 1) % 8 or n < 9:
            return f"num_frames must be 8k+1, got {n}"
        grid = 64 if str(pipeline).startswith("two_stage") else 32
        if w % grid or h % grid:
            return f"{pipeline} needs width and height divisible by {grid}"
        if body.get("audio") and not str(pipeline).startswith("two_stage"):
            return "audio-to-video requires pipeline two_stage or two_stage_hq"
        if n * w * h * 3 > 768 * (1 << 20):
            return "request exceeds 768 MiB of raw RGB"
        if body.get("last_frame_image") and n < 9:
            return "last_frame_image needs at least 9 frames"
        summary = {k: body.get(k) for k in ("width", "height", "num_frames", "seed", "pipeline", "steps", "decoder",
                                            "prompt", "lora_paths")}
        for key in ("first_frame_image", "last_frame_image", "audio"):
            if body.get(key):
                try:
                    raw = base64.b64decode(body[key], validate=True)
                except Exception:
                    return f"{key} is not base64"
                summary[key] = len(raw)
                if key == "audio":
                    try:
                        with wave.open(io.BytesIO(raw), "rb") as wv:
                            summary["audio_s"] = wv.getnframes() / float(wv.getframerate())
                    except Exception:
                        return "audio must be a WAV"
        with self.server_state.lock:
            self.server_state.video_requests.append(summary)
        return None

    def _media(self, path, body):
        s = self.server_state
        if not self._resident_or_fallback(body):
            return
        if path.endswith("video/generations"):
            problem = self._video_problem(body)
            if problem:
                return self._error(400, problem)
        if path.endswith("speech") and body.get("ref_audio"):
            try:
                with wave.open(io.BytesIO(base64.b64decode(body["ref_audio"], validate=True)), "rb"):
                    pass
            except Exception:
                return self._error(400, "ref_audio must be a base64 WAV")
        if path.endswith("speech"):
            with s.lock:
                s.last_speech = str(body.get("input") or "")
        with s.lock:
            behaviour = s.script[path].popleft() if s.script[path] else "ok"
        if behaviour == "http500":
            return self._error(500, "generation failed", "server_error")
        if behaviour == "http400":
            return self._error(400, "steps must be between 1 and 100")
        self._sse_start()
        video = path.endswith("video/generations")
        total = s.steps
        stages = [("Encoding prompt", 0, total)] + [("Denoising", i, total) for i in range(1, total + 1)] \
            if video else [("encode", 1, 1)] + [("diffuse", i, total) for i in range(1, total + 1)] \
            + [("decode", 1, 1)]
        if path.endswith("speech"):
            stages = []  # Kokoro sends no progress events, only the complete event
        hang_after = None
        if isinstance(behaviour, tuple) and behaviour[0] == "hang":
            hang_after = behaviour[1]
        elif behaviour == "hang":
            hang_after = 0
        for i, (stage, step, tot) in enumerate(stages):
            if hang_after is not None and i >= hang_after:
                break
            if not self._send({"type": "progress", "stage": stage, "step": step, "total": tot}):
                with s.lock:
                    s.disconnects.append(path)
                return
            if s.progress_delay:
                time.sleep(s.progress_delay)
            if behaviour == "error" and i == 1:
                self._send({"type": "error", "message": "generation failed"})
                return
            if behaviour in ("drop", "die") and i == 1:
                if behaviour == "die":
                    s.health_ok = False
                self.connection.shutdown(socket.SHUT_RDWR)
                return
        if hang_after is not None:
            if self._wait_gone():
                with s.lock:
                    s.disconnects.append(path)
            return
        if video:
            self._send({"type": "progress", "stage": "Decoding video", "step": total, "total": total})
            w, h, n = int(body.get("width", 384)), int(body.get("height", 256)), int(body.get("num_frames", 9))
            mode = behaviour if behaviour in ("black", "frozen") else "ok"
            rgb = video_frames(w, h, n, mode)
            pcm = struct.pack("<" + "h" * 2 * 4800, *([1000, -1000] * 4800))
            head = (f'{{"type":"complete","frames":{n},"height":{h},"width":{w},"fps":24,"format":"rgb8",'
                    f'"data":"').encode()
            b64 = base64.b64encode(rgb)
            if behaviour == "escaped":  # Swift-style JSON writers escape every slash
                b64 = b64.replace(b"/", b"\\/")
            tail = (b'","audio_sample_rate":48000,"audio_channels":2,"audio_format":"pcm_s16le","audio_data":"'
                    + base64.b64encode(pcm) + b'"}')
            self._send(head + b64 + tail)
            return
        if path.endswith("speech"):
            seconds = min(s.speech_cap, max(1.0, s.speech_rate * len(body.get("input", "x"))))
            wav = s.speak(str(body.get("input") or ""), body, seconds, behaviour == "silent")
        else:
            seconds = min(s.music_cap, float(body.get("duration_seconds") or max(1.0, 0.06 * len(body.get("input", "x")))))
            wav = tone_wav(seconds, silent_tail=seconds if behaviour == "silent" else 0.0)
        self._send({"type": "complete", "format": "wav", "data": base64.b64encode(wav).decode()})

    def _chat(self, body):
        s = self.server_state
        if not self._resident_or_fallback(body):
            return
        with s.lock:
            reply = s.chat_replies.popleft() if s.chat_replies else '{"verdict":"accept","note":"looks right"}'
        if reply == "hang":
            self._sse_start()
            self._wait_gone()
            return
        if not body.get("stream"):
            return self._json(200, {"id": "x", "object": "chat.completion", "model": body.get("model"),
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                                                 "finish_reason": "stop"}]})
        self._sse_start()
        for i in range(0, len(reply), 7):
            chunk = {"id": "x", "object": "chat.completion.chunk", "model": body.get("model"),
                     "choices": [{"index": 0, "delta": {"content": reply[i:i + 7]}, "finish_reason": None}]}
            if not self._send(chunk):
                return
        self._send(b"[DONE]")
