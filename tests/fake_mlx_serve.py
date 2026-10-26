"""A small in-process imitation of mlx-serve v26.10.3's HTTP surface, for adapter tests.

It reproduces the shapes the adapter depends on (checked against the server source and the
live Studio): /api/ps (READY models, ':latest' names), /v1/models (+ machine.free_mib),
load by discovered id or absolute path (registered under the basename), idempotent unload,
SSE progress / complete / error for speech, music and video, streaming chat completions, and
the fallback to the default model for an unknown id. Per-request behaviours (hang, error,
drop, http500, black or frozen video) are scripted per endpoint.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import math
import os
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


class FakeMlxServe:
    def __init__(self, sizes=None):
        self.sizes = dict(sizes or {})          # server id (or basename) -> bytes
        self.ready = collections.OrderedDict()  # server id -> bytes (what /api/ps shows)
        self.discovered = set()                 # ids /v1/models lists (discovery)
        # mlx-serve v26.10.3 started without --model: the latest chat-capable load is promoted to
        # the default (default_promoted), unless a load said "default": true (setDefault).
        self.default_id = ""
        self.default_promoted = False
        self.media_markers = ("kokoro", "ace-step", "ltx", "flux", "tts")
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
        self.lock = threading.Lock()
        self._server = None

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
                data = [{"id": i, "object": "model", "loaded": i in s.ready,
                         "state": "ready" if i in s.ready else "unloaded",
                         "capabilities": (["audio"] if any(m in i.lower() for m in s.media_markers)
                                          else ["chat", "streaming", "vision"])} for i in ids]
            return self._json(200, {"object": "list", "data": data,
                                    "machine": {"os": "macos", "free_mib": s.free_mib, "total_mib": 262144}})
        return self._error(404, "Unknown endpoint", "not_found")

    def do_POST(self):
        s = self.server_state
        body = self._body()
        with s.lock:
            s.requests.append(("POST", self.path, body))
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
        if s.load_delay:
            time.sleep(s.load_delay)
        if s.load_error:
            return self._error(s.load_error[0], s.load_error[1], "model_load_failed")
        if target.startswith("/"):
            mid = os.path.basename(target.rstrip("/"))
        elif target in s.discovered or target in s.ready:
            mid = target
        else:
            return self._error(404, "Unknown model id", "model_not_found")
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

    def _media(self, path, body):
        s = self.server_state
        if not self._resident_or_fallback(body):
            return
        with s.lock:
            behaviour = s.script[path].popleft() if s.script[path] else "ok"
        if behaviour == "http500":
            return self._error(500, "generation failed", "server_error")
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
        seconds = float(body.get("duration_seconds") or max(1.0, 0.06 * len(body.get("input", "x"))))
        wav = tone_wav(min(seconds, 3.0), silent_tail=min(seconds, 3.0) if behaviour == "silent" else 0.0)
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
