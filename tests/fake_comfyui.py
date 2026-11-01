"""A small imitation of ComfyUI 0.43's HTTP API, for the ComfyUI engine's tests.

Endpoints: GET /system_stats, POST /prompt {"prompt", "client_id"} -> {"prompt_id"}, GET /queue,
POST /queue {"delete": [...]}, GET /history/<id>, GET /view?filename=&subfolder=&type=,
POST /upload/image (multipart "image", "overwrite"), POST /interrupt, POST /free.

A queued prompt "runs" on a worker thread for `run_s` seconds and then writes one PNG per output
node (a node whose class_type starts with Save or Preview), sized by the first width/height inputs
found in the graph. A LoadImage node must name an uploaded file (else /prompt answers 400 with
node_errors, as ComfyUI does). Behaviours per prompt: "ok", "error" (an execution error in the
history), "hang" (runs until interrupted).

Run as a script it is a stand-in ComfyUI process: `python fake_comfyui.py --listen H --port P`
(a test writes a main.py that calls main()). FAKE_COMFY_IGNORE_TERM=1 makes it ignore SIGTERM.
"""
from __future__ import annotations

import collections
import json
import os
import signal
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from foreman.media import encode_png  # noqa: E402

try:
    from fake_mlx_serve import parse_multipart  # noqa: E402
except ImportError:  # pragma: no cover
    from tests.fake_mlx_serve import parse_multipart  # noqa: E402


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
                    if str(node.get("class_type", "")).startswith(("Save", "Preview")):
                        name = f"out-{pid[:8]}-{nid}.png"
                        pix = bytes((x * 3 + y) % 256 for y in range(h) for x in range(w) for _ in range(3))
                        self.outputs[name] = encode_png(w, h, pix)
                        outs[nid] = {"images": [{"filename": name, "subfolder": "", "type": "output"}]}
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
            self.send_header("Content-Type", "image/png")
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
