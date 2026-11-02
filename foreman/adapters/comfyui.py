"""ComfyUIEngine: an on-demand ComfyUI server as a gpud engine, for graphs mlx-serve cannot run.

ComfyUI is not a service that is always there: gpud starts it when a job needs it and stops it
when the job is done, so it never shares the GPU with an mlx-serve media model.

  * Its one catalog model (family "comfy", ModelSpec.solo_media) is "resident" while the server
    process Foreman started is alive (answering HTTP or not: it holds its memory). load() starts it:
        <python> main.py --listen 127.0.0.1 --port <port> --use-pytorch-cross-attention --disable-auto-launch
    from the configured ComfyUI directory (its own venv), logging to <home>/logs/comfyui.log, and
    the load is ready once /system_stats answers. Its pid (and port) is kept in
    <home>/run/comfyui.json so a restarted gpud adopts and can still stop the server it started:
    only a process that is alive, leads its own session, runs main.py and started when the file
    says is adopted, and only its process group is ever signalled. An adopted server's leftover
    prompts are interrupted and removed before the next job.
  * The memory rule: gpud unloads its own mlx-serve media models before the load (solo_media),
    and the load is deferred (gpud retries, and the job fails as an infrastructure failure after
    Policy.load_defer_max_s) while `media_probe()` reports any media model resident in mlx-serve
    (someone else's too: the adapter asks the server itself) or free memory minus the planned
    footprint would fall below `min_headroom_gb`. A server that is already up checks the same
    media rule again before every job and every graph.
  * A job is a list of graph ops: {"op": "graph", "graph": "<name>", "params": {...},
    "images": {"image_1": "<path>"}, "out": "<file in the work directory>"}. The graph is a
    template file <graphs_dir>/<name>.json in ComfyUI's API format with placeholders such as
    {{prompt}}, {{seed}}, {{width}}, {{image_1}} (a string that is exactly one placeholder takes the
    value's type; one inside a longer string is replaced by its text). Optional wrapper:
        {"graph": {...}, "outputs": ["9"], "defaults": {"steps": 8}, "license": "...", "commercial_ok": true,
         "attribution": "..."}   (a credit the model's license requires, recorded with each output)
    Images are uploaded first (POST /upload/image), the graph is queued (POST /prompt), its
    progress is followed by polling /queue and /history/<id> (a bounded heartbeat while it runs),
    and the first file of the output nodes is downloaded (GET /view) to `out`, streamed to a temp
    file and checked (a cut connection, an error page, a truncated image is never kept), with a
    sidecar `<out>.meta.json` that makes the op skip on a retry.
  * After the job: POST /free {"unload_models": true, "free_memory": true}, and (exit_after_job,
    the default) the process is stopped before the job reports done: SIGTERM to its process
    group, a wait, then SIGKILL. Only a ComfyUI Foreman started is ever signalled; a server on
    the port that Foreman did not start makes the load fail (it is someone else's), and
    mlx-serve is never signalled at all.
  * cancel(): POST /interrupt and remove the prompt from the queue. close() (gpud shutting down)
    stops the server too.
"""
from __future__ import annotations

import collections
import copy
import itertools
import json
import logging
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.parse

from .. import media, paths
from ..engine import LoadStatus, Poll, RunSpec
from ..httpio import HttpError, StreamConn, error_message, json_call
from ..policy import number

log = logging.getLogger("foreman.comfyui")

EVENT_CAP = 0.95
MAX_OUTPUT_BYTES = 2 << 30   # one downloaded output at most (a 2 GiB take is far past anything an ad uses)
PLACEHOLDER = re.compile(r"\{\{\s*([a-z][a-z0-9_]{0,40})\s*\}\}")
META = ".meta.json"
# launchd runs gpud as a Background job, and macOS gives that band (throttled CPU and I/O) to every
# child, a new session included: ComfyUI's start-up, model loading and CPU-side work ran several times
# slower in it than from a shell. ComfyUI is started through this, which leaves the band and execs it
# (the process keeps its pid, and its command line is main.py's again).
UNBACKGROUND = ("import os, sys\ntry:\n    os.setpriority(4, 0, 0)  # PRIO_DARWIN_PROCESS: not background\n"
                "except (AttributeError, OSError):\n    pass\nos.execv(sys.argv[1], sys.argv[1:])")


class GraphError(ValueError):
    """A graph template that cannot be rendered (missing file, bad JSON, a value missing)."""


class StopPending(RuntimeError):
    """The ComfyUI process is still exiting; gpud asks again next tick."""

    pending = True


class RunRefused(Exception):
    """The server refused the graph itself (a content failure: Poll.reason 'rejected')."""


class EngineTrouble(Exception):
    """The host or the server failed, not the graph: an mlx-serve media model resident beside a
    running ComfyUI, an answer that is not the file it should be, a stale prompt that will not
    leave the queue. Poll.reason 'engine_error': infrastructure, never charged to the work."""


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- graph templates

def load_graph(graphs_dir: str, name) -> dict:
    """{"graph", "outputs", "defaults", "license", "commercial_ok"} for a template name."""
    if not isinstance(name, str) or not paths.SAFE_NAME.match(name) or name.startswith("."):
        raise GraphError(f"graph name {name!r} must be a plain file name")
    path = os.path.join(os.path.expanduser(graphs_dir or ""), name if name.endswith(".json") else name + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise GraphError(f"graph template {name!r}: cannot read {path}: {e}") from None
    except ValueError as e:
        raise GraphError(f"graph template {name!r}: not JSON: {e}") from None
    if not isinstance(data, dict):
        raise GraphError(f"graph template {name!r} must be a JSON object")
    wrapped = isinstance(data.get("graph"), dict)
    graph = data["graph"] if wrapped else data
    if not graph or not all(isinstance(v, dict) and "class_type" in v for v in graph.values()):
        raise GraphError(f"graph template {name!r} is not an API-format graph (nodes with class_type)")
    outputs = data.get("outputs") if wrapped else None
    if outputs is not None and (not isinstance(outputs, list) or not all(str(o) in graph for o in outputs)):
        raise GraphError(f"graph template {name!r}: outputs must list node ids of the graph")
    defaults = data.get("defaults") if wrapped and isinstance(data.get("defaults"), dict) else {}
    attribution = data.get("attribution") if wrapped else None
    return {"graph": graph, "outputs": [str(o) for o in outputs] if outputs else None, "defaults": defaults,
            "license": data.get("license") if wrapped else None,
            "commercial_ok": data.get("commercial_ok") is True if wrapped else False,
            "attribution": attribution.strip()[:200] if isinstance(attribution, str) and attribution.strip() else None,
            "path": path}


_WEIGHT_FILE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|onnx)\b", re.I)


def missing_install(body) -> str | None:
    """What a POST /prompt refusal says the installation lacks (None when it is about the graph
    itself): a node type no installed pack provides, or a model file that is not on disk (ComfyUI
    answers value_not_in_list for an input naming a weights file it does not have)."""
    try:
        obj = json.loads((body or b"").decode("utf-8", "replace"))
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    err = obj.get("error") if isinstance(obj.get("error"), dict) else {}
    msg = str(err.get("message") or "")
    if err.get("type") == "missing_node_type" or re.search(r"node (type )?\S+ (does not exist|not found)|"
                                                           r"not found\. the custom node", msg, re.I):
        return f"a node type is not installed: {msg[:200]}"
    nodes = obj.get("node_errors") if isinstance(obj.get("node_errors"), dict) else {}
    for nid, entry in nodes.items():
        for e in (entry.get("errors") if isinstance(entry, dict) and isinstance(entry.get("errors"), list) else []):
            if not isinstance(e, dict):
                continue
            extra = e.get("extra_info") if isinstance(e.get("extra_info"), dict) else {}
            value = str(extra.get("received_value") or e.get("details") or "")
            if e.get("type") == "value_not_in_list" and _WEIGHT_FILE.search(value):
                return (f"node {nid} ({entry.get('class_type') or '?'}): a model file is missing: "
                        f"{str(e.get('details') or value)[:200]}")
    return None


def output_problem(path: str, ctype: str, name: str) -> str | None:
    """Why a downloaded output is not the file it should be (None when it is): an error page served
    with 200, a truncated image, a video or audio file without its container's signature."""
    ctype = (ctype or "").split(";")[0].strip().lower()
    if ctype.startswith("text/") or ctype in ("application/json", "application/xhtml+xml"):
        return f"the answer is {ctype}, not a media file"
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(64)
            f.seek(max(0, size - 16))
            tail = f.read(16)
    except OSError as e:
        return f"unreadable: {e}"
    if size <= 0:
        return "empty"
    ext = os.path.splitext(name or path)[1].lower() or os.path.splitext(path)[1].lower()
    if ext == ".png" or head[:8] == b"\x89PNG\r\n\x1a\n":
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return "not a PNG"
        if b"IEND" not in tail:
            return "a truncated PNG (no IEND chunk)"
        return None if media.image_size(path) else "a PNG without a readable header"
    if ext in (".jpg", ".jpeg"):
        if head[:2] != b"\xff\xd8" or b"\xff\xd9" not in tail:
            return "not a complete JPEG"
        return None
    if ext in (".mp4", ".mov", ".m4v"):
        return None if head[4:8] == b"ftyp" else "not an MP4/MOV (no ftyp box)"
    if ext == ".webm":
        return None if head[:4] == b"\x1a\x45\xdf\xa3" else "not a WebM file"
    if ext == ".gif":
        return None if head[:4] == b"GIF8" else "not a GIF"
    if ext == ".wav":
        return None if head[:4] == b"RIFF" and head[8:12] == b"WAVE" else "not a WAV file"
    if ext in (".webp",):
        return None if head[:4] == b"RIFF" and head[8:12] == b"WEBP" else "not a WebP image"
    if head.lstrip()[:1] in (b"<", b"{"):
        return "the answer looks like a page or JSON, not a media file"
    return None


def placeholders(obj) -> set:
    found = set()
    if isinstance(obj, dict):
        for v in obj.values():
            found |= placeholders(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= placeholders(v)
    elif isinstance(obj, str):
        found |= set(PLACEHOLDER.findall(obj))
    return found


def render_graph(graph: dict, values: dict) -> dict:
    """The graph with every {{name}} replaced. Raises GraphError naming the missing values."""
    missing = sorted(placeholders(graph) - set(values))
    if missing:
        raise GraphError(f"the graph needs values for {missing}")

    def sub(v):
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, str):
            m = PLACEHOLDER.fullmatch(v.strip())
            if m:
                return copy.deepcopy(values[m.group(1)])
            return PLACEHOLDER.sub(lambda mm: str(values[mm.group(1)]), v)
        return v
    return sub(graph)


# ---------------------------------------------------------------- the engine

class _Load:
    def __init__(self, ticket: str):
        self.ticket, self.state, self.error, self.deferred = ticket, "loading", None, False


class _Run:
    def __init__(self, handle: str, spec: RunSpec):
        self.handle, self.spec = handle, spec
        self.status, self.progress = "running", 0.0
        self.artifact, self.error, self.reason = None, None, None
        self.log = collections.deque(maxlen=40)
        self.cancel = threading.Event()
        self.prompt_id = None
        self.beat = None
        self.t0 = time.monotonic()


class ComfyUIEngine:
    def __init__(self, catalog, comfy_dir: str = "~/bakeoff/ComfyUI", python: str | None = None, port: int = 8188,
                 host: str = "127.0.0.1", graphs_dir: str = "~/.foreman/comfyui/graphs",
                 artifacts_dir: str = "~/.foreman/artifacts", home: str = "~/.foreman", input_roots=None,
                 min_headroom_gb: float = 40.0, mem_probe=None, media_probe=None, start_timeout_s: float = 240.0,
                 exit_after_job: bool = True, extra_args=(), stop_wait_s: float = 20.0, unload_wait_s: float = 8.0,
                 poll_s: float = 1.0, call_timeout_s: float = 5.0, name: str = "comfyui", popen=None):
        self.name = name
        self.catalog = dict(catalog)
        self._mine = {m for m, s in self.catalog.items() if s.engine == name}
        self.comfy_dir = os.path.expanduser(comfy_dir)
        self.python = os.path.expanduser(python) if python else self._default_python()
        self.host, self.port = host, int(port)
        self._live_port = None         # the port of the server Foreman runs (an adopted one keeps its own)
        self.graphs_dir = os.path.expanduser(graphs_dir)
        self.artifacts_dir = os.path.expanduser(artifacts_dir)
        self.home = os.path.expanduser(home)
        self.input_roots = [self.artifacts_dir] + [os.path.expanduser(r) for r in (input_roots or [self.home])]
        self.min_headroom_gb = float(min_headroom_gb)
        self.mem_probe, self.media_probe = mem_probe, media_probe
        self.start_timeout_s, self.exit_after_job = float(start_timeout_s), bool(exit_after_job)
        self.extra_args = [str(a) for a in (extra_args or ())]
        self.stop_wait_s, self.unload_wait_s = float(stop_wait_s), float(unload_wait_s)
        self.poll_s, self.call_timeout_s = float(poll_s), float(call_timeout_s)
        self._popen = popen or subprocess.Popen
        self._boot = secrets.token_hex(4)
        self._seq = itertools.count(1)
        self._lock = threading.RLock()
        self._proc = None              # Popen of the server Foreman started (this process)
        self._pid = None               # its pid (also one adopted from the pid file)
        self._started_wall = None
        self._loads: dict = {}
        self._runs: dict = {}
        self._by_run: dict = {}
        self._stopping = None          # thread stopping the server
        self._version = None
        self._stale_queue = False      # an adopted server may still run what the dead gpud queued
        self._pidfile = os.path.join(self.home, "run", "comfyui.json")
        self._adopt()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self._live_port or self.port}"

    def _default_python(self) -> str:
        for rel in ("venv/bin/python", ".venv/bin/python", "venv/bin/python3", ".venv/bin/python3"):
            cand = os.path.join(self.comfy_dir, rel)
            if os.path.exists(cand):
                return cand
        return os.path.join(self.comfy_dir, "venv", "bin", "python")

    # ---- the process
    def _alive(self, pid) -> bool:
        if not pid:
            return False
        if self._proc is not None and self._proc.pid == pid:
            return self._proc.poll() is None
        try:  # a child of this process that already exited: reap it (a zombie still answers kill 0)
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                return False
        except ChildProcessError:
            pass
        except OSError:
            pass
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False
        return True

    def _adopt(self) -> None:
        """A ComfyUI an earlier gpud started (pid file) is still Foreman's to stop, on whatever port
        it was started (a changed comfyui_options.port never orphans it): it is adopted when the pid
        is alive, leads its own session (Foreman starts ComfyUI with start_new_session), runs main.py
        and started when the pid file says. Anything else is not Foreman's: the pid file is dropped
        and that process is never signalled."""
        info = paths.read_json(self._pidfile)
        if not isinstance(info, dict):
            return
        pid, port = info.get("pid"), info.get("port")
        if isinstance(pid, int) and isinstance(port, int) and self._alive(pid) and self._ours(pid, info.get("started_wall")):
            self._pid, self._started_wall, self._live_port = pid, info.get("started_wall"), port
            self._stale_queue = True
            log.info("adopted the ComfyUI server Foreman started earlier (pid %d, port %d)", pid, port)
        else:
            self._clear_pidfile()

    @classmethod
    def _ours(cls, pid: int, started_wall) -> bool:
        """`pid` is the ComfyUI Foreman started at `started_wall`, not a reused pid or another app's."""
        try:
            if os.getsid(pid) != pid:  # Foreman's ComfyUI leads its own session; a reused pid rarely does
                return False
        except OSError:
            return False
        if not cls._is_comfy(pid):
            return False
        age = cls._age_s(pid)
        start = number(started_wall)
        return age is not None and start is not None and abs((time.time() - age) - start) <= 120.0

    @staticmethod
    def _age_s(pid: int) -> float | None:
        """Seconds since the process started (ps etime: [[dd-]hh:]mm:ss), or None."""
        try:
            out = subprocess.run(["/bin/ps", "-o", "etime=", "-p", str(pid)], capture_output=True, text=True,
                                 timeout=3, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        m = re.match(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", out)
        if not m:
            return None
        d, h, mi, sec = (int(x) if x else 0 for x in m.groups())
        return float(((d * 24 + h) * 60 + mi) * 60 + sec)

    @staticmethod
    def _is_comfy(pid: int) -> bool:
        try:
            out = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                                 timeout=3, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        return "main.py" in out

    def _write_pidfile(self) -> None:
        os.makedirs(os.path.dirname(self._pidfile), exist_ok=True)
        paths.atomic_json(self._pidfile, {"pid": self._pid, "port": self._live_port or self.port,
                                          "started_wall": self._started_wall, "comfy_dir": self.comfy_dir})

    def _clear_pidfile(self) -> None:
        try:
            os.remove(self._pidfile)
        except OSError:
            pass

    def _healthy(self, timeout: float | None = None) -> dict | None:
        try:
            _, obj = json_call(self.base_url, "GET", "/system_stats", timeout=timeout or min(3.0, self.call_timeout_s))
        except Exception:
            return None
        return obj if isinstance(obj, dict) else {}

    def _spawn(self) -> None:
        main = os.path.join(self.comfy_dir, "main.py")
        if not os.path.isfile(main):
            raise FileNotFoundError(f"no ComfyUI at {self.comfy_dir} (main.py missing): set comfyui_options.dir")
        if not os.path.exists(self.python):
            raise FileNotFoundError(f"ComfyUI's python {self.python} does not exist: set comfyui_options.python")
        logs = os.path.join(self.home, "logs")
        os.makedirs(logs, exist_ok=True)
        out = open(os.path.join(logs, "comfyui.log"), "ab")
        cmd = [self.python, "main.py", "--listen", self.host, "--port", str(self.port),
               "--use-pytorch-cross-attention", "--disable-auto-launch"] + self.extra_args
        log.info("starting ComfyUI: %s (in %s)", " ".join(cmd), self.comfy_dir)
        if sys.platform == "darwin":  # leave launchd's Background band first (see UNBACKGROUND)
            cmd = [self.python, "-c", UNBACKGROUND] + cmd
        try:
            proc = self._popen(cmd, cwd=self.comfy_dir, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True, env=self._env())
        finally:
            out.close()
        with self._lock:
            self._proc, self._pid, self._started_wall = proc, proc.pid, time.time()
            self._live_port, self._stale_queue = self.port, False
        self._write_pidfile()

    def _env(self) -> dict:
        """ComfyUI's environment: gpud's, without its PYTHONPATH (Foreman's checkout is not
        ComfyUI's to import), and with the python's own bin directory first on PATH: custom nodes
        build kernels with tools from the venv (ninja), which launchd's minimal PATH does not have."""
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        bindir = os.path.dirname(self.python)
        rest = [p for p in env.get("PATH", os.defpath).split(os.pathsep) if p and p != bindir]
        env["PATH"] = os.pathsep.join([bindir] + rest)
        return env

    def _signal(self, sig) -> None:
        """Signal the server's process group: ComfyUI and its children only. Foreman's ComfyUI leads
        its own session, so when killpg fails the process is gone or is not Foreman's: nothing else
        is ever signalled (no fallback to the bare pid)."""
        pid = self._pid
        if not pid:
            return
        proc = self._proc
        if proc is None or proc.pid != pid:  # adopted: make sure the pid still is the server Foreman started
            try:
                if os.getsid(pid) != pid:
                    log.warning("pid %d no longer leads its own session: not Foreman's ComfyUI, not signalled", pid)
                    return
            except OSError:
                return
        try:
            os.killpg(pid, sig)
        except OSError as e:
            log.debug("killpg(%d) failed: %s", pid, e)

    def _stop_process(self) -> bool:
        """SIGTERM, wait stop_wait_s, then SIGKILL: only ever the server Foreman started."""
        with self._lock:
            pid = self._pid
        if not pid or not self._alive(pid):
            self._forget()
            return True
        self._signal(signal.SIGTERM)
        end = time.monotonic() + self.stop_wait_s
        while time.monotonic() < end:
            if not self._alive(pid):
                self._forget()
                return True
            time.sleep(0.1)
        log.warning("ComfyUI (pid %d) ignored SIGTERM for %.0fs: SIGKILL (Foreman started it)", pid, self.stop_wait_s)
        self._signal(signal.SIGKILL)
        end = time.monotonic() + 5.0
        while time.monotonic() < end:
            if not self._alive(pid):
                self._forget()
                return True
            time.sleep(0.1)
        return False

    def _forget(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc, self._pid, self._started_wall = None, None, None
            self._live_port, self._stale_queue = None, False
        if proc is not None:
            try:
                proc.wait(timeout=0.1)
            except Exception:
                pass
        self._clear_pidfile()

    def _ours_running(self) -> bool:
        with self._lock:
            pid = self._pid
        return bool(pid and self._alive(pid))

    # ---- Engine interface: residency
    def resident_models(self) -> set:
        """The server is resident while the process Foreman started is alive, whether or not it
        answers HTTP right now (busy, swapping, wedged): it holds its memory either way, and a
        health blip must never make gpud drop it and load an mlx-serve media model beside it (fail
        closed). Only while Foreman is still starting it is it not resident yet."""
        if not self._ours_running():
            return set()
        with self._lock:
            loading = any(op.state == "loading" for op in self._loads.values())
        return set() if loading else set(self._mine)

    def instance_id(self):
        with self._lock:
            return f"pid:{self._pid}:{self._started_wall}" if self._pid else None

    def load(self, model: str) -> str:
        if model not in self._mine:
            raise ValueError(f"{model!r} is not a {self.name} model")
        ticket = f"{self.name}-load-{next(self._seq)}-{self._boot}"
        op = _Load(ticket)
        with self._lock:
            self._loads[ticket] = op
        if self._ours_running():
            op.state = "ready" if self._healthy() is not None else "loading"
            if op.state == "loading":
                threading.Thread(target=self._wait_ready, args=(op,), daemon=True).start()
            return ticket
        if self._healthy() is not None:
            op.state, op.error = "failed", (f"a ComfyUI that Foreman did not start answers on port {self.port}: it is "
                                            f"someone else's (never used or stopped by Foreman); free the port or "
                                            f"configure another one")
            return ticket
        problem = self._preflight(model)
        if problem:
            op.state, op.error, op.deferred = "failed", problem, True
            log.warning("ComfyUI start deferred: %s", problem)
            return ticket
        try:
            self._spawn()
        except Exception as e:
            op.state, op.error = "failed", f"{type(e).__name__}: {e}"
            return ticket
        threading.Thread(target=self._wait_ready, args=(op,), name="foreman-comfyui-start", daemon=True).start()
        return ticket

    def _preflight(self, model: str) -> str | None:
        if self.media_probe is not None:
            try:
                busy = sorted(self.media_probe() or ())
            except Exception as e:
                return f"cannot tell whether mlx-serve holds media models ({type(e).__name__}: {e})"
            if busy:
                return (f"mlx-serve holds media model(s) {busy}: ComfyUI never runs beside an mlx-serve media model "
                        f"(gpud unloads its own first; someone else's must go before ComfyUI starts)")
        if self.mem_probe is not None:
            avail = number(self.mem_probe())
            spec = self.catalog[model]
            if avail is None:
                return "cannot measure free memory before starting ComfyUI"
            if avail - spec.size_gb < self.min_headroom_gb:
                return (f"insufficient headroom for ComfyUI: {avail:.1f} GB free, it is planned at {spec.size_gb:g} GB "
                        f"and {self.min_headroom_gb:g} GB must stay free")
        return None

    def _wait_ready(self, op: _Load) -> None:
        end = time.monotonic() + self.start_timeout_s
        while time.monotonic() < end:
            with self._lock:
                proc = self._proc
            if proc is not None and proc.poll() is not None:
                op.state, op.error = "failed", f"ComfyUI exited with status {proc.returncode} while starting (see logs/comfyui.log)"
                self._forget()
                return
            stats = self._healthy(timeout=2.0)
            if stats is not None:
                self._version = ((stats.get("system") or {}).get("comfyui_version")
                                 if isinstance(stats.get("system"), dict) else None)
                op.state = "ready"
                log.info("ComfyUI is ready (version %s)", self._version)
                return
            time.sleep(min(1.0, self.poll_s))
        op.state, op.error = "failed", f"ComfyUI did not answer /system_stats within {self.start_timeout_s:.0f}s"
        self._stop_process()

    def poll_load(self, ticket: str) -> LoadStatus:
        with self._lock:
            op = self._loads.get(ticket)
        if op is None:
            return LoadStatus("failed", "unknown load ticket (issued by an earlier gpud process)")
        return LoadStatus(op.state, op.error, op.deferred)

    def unload(self, model: str) -> bool:
        if model not in self._mine:
            raise PermissionError(f"refusing to unload {model!r}: not a {self.name} model")
        with self._lock:
            live = [r for r in self._runs.values() if r.status == "running"]
        for r in live:
            self._close(r, "ComfyUI is being stopped")
        if not self._ours_running():
            self._forget()
            return True
        with self._lock:
            stopping = self._stopping
            fresh = stopping is None or not stopping.is_alive()
        if fresh:
            self._free()
            stopping = threading.Thread(target=self._stop_process, name="foreman-comfyui-stop", daemon=True)
            with self._lock:
                self._stopping = stopping
            stopping.start()
        stopping.join(self.unload_wait_s)
        if stopping.is_alive():
            raise StopPending(f"ComfyUI is still exiting (pid {self._pid})")
        return not self._ours_running()

    def terminate(self) -> bool:
        try:
            return self.unload(next(iter(self._mine)))
        except StopPending:
            return False

    def _free(self) -> None:
        try:
            json_call(self.base_url, "POST", "/free", {"unload_models": True, "free_memory": True},
                      timeout=max(5.0, self.call_timeout_s))
        except Exception as e:
            log.warning("ComfyUI /free failed: %s", e)

    # ---- runs
    def start(self, spec: RunSpec) -> str:
        with self._lock:
            handle = self._by_run.get(spec.run_id)
            if handle:
                return handle
            if spec.model not in self._mine:
                raise ValueError(f"{spec.model!r} is not a {self.name} model")
            handle = f"{self.name}:{self._boot}:{spec.run_id}"
            run = _Run(handle, spec)
            self._runs[handle] = run
            self._by_run[spec.run_id] = handle
        threading.Thread(target=self._main, args=(run,), name=f"foreman-comfyui-{spec.run_id}", daemon=True).start()
        return handle

    def find(self, run_id: str):
        with self._lock:
            return self._by_run.get(run_id)

    def poll(self, handle: str) -> Poll:
        with self._lock:
            run = self._runs.get(handle)
            if run is None:
                return Poll("lost", log_tail=("comfyui adapter: this gpud process did not start that run",))
            if run.status == "running" and run.beat is not None:
                t0, grace, lo, hi = run.beat
                frac = min(1.0, (time.monotonic() - t0) / grace) if grace > 0 else 1.0
                run.progress = max(run.progress, lo + (hi - lo) * frac)
            return Poll(run.status, run.progress, run.artifact, run.error, tuple(run.log)[-8:], run.reason)

    def cancel(self, handle: str) -> None:
        with self._lock:
            run = self._runs.get(handle)
        if run is None:
            raise LookupError(f"unknown run handle {handle!r}")
        self._close(run, "cancel requested by gpud")

    def _close(self, run: _Run, why: str) -> None:
        with self._lock:
            if run.status != "running" or run.cancel.is_set():
                return
            run.cancel.set()
            run.log.append(f"[comfyui] {why}")
            pid = run.prompt_id
        try:
            json_call(self.base_url, "POST", "/interrupt", {}, timeout=self.call_timeout_s)
            if pid:
                json_call(self.base_url, "POST", "/queue", {"delete": [pid]}, timeout=self.call_timeout_s)
        except Exception as e:
            log.debug("ComfyUI interrupt failed: %s", e)

    def close(self) -> None:
        """gpud is shutting down: stop the runs, then the server Foreman started (POST /free,
        SIGTERM, SIGKILL), so no ComfyUI of Foreman's holds memory while no gpud enforces the
        memory rule. Runs synchronously on the caller's thread."""
        with self._lock:
            live = [r for r in self._runs.values() if r.status == "running"]
        for r in live:
            self._close(r, "gpud shutting down")
        if self._ours_running():
            self._free()
            self._stop_process()

    def live_runs(self) -> list:
        with self._lock:
            return [r.spec.run_id for r in self._runs.values() if r.status == "running"]

    def _note(self, run: _Run, line: str) -> None:
        with self._lock:
            run.log.append(f"[comfyui] {line}")

    def _finish(self, run: _Run, status: str, artifact=None, error=None, reason=None) -> None:
        with self._lock:
            if run.status != "running":
                return
            run.status, run.artifact, run.error, run.reason = status, artifact, error, reason
            run.beat = None
            if status == "done":
                run.progress = 1.0
            run.log.append(f"[comfyui] {status}" + (f": {error}" if error else ""))

    def _main(self, run: _Run) -> None:
        try:
            art = self._run_ops(run)
            self._after_job(run)
            if run.cancel.is_set():
                raise Cancelled("cancelled")
            self._finish(run, "done", artifact=art)
        except Cancelled:
            self._after_job(run)
            self._finish(run, "cancelled", error="cancelled")
        except (RunRefused, GraphError, paths.PathRefused) as e:
            self._after_job(run)
            self._finish(run, "failed", error=str(e)[:1000], reason="rejected")
        except EngineTrouble as e:
            self._after_job(run)
            if run.cancel.is_set():
                self._finish(run, "cancelled", error="cancelled")
            else:
                log.warning("ComfyUI run %s failed: %s", run.spec.run_id, e)
                self._finish(run, "failed", error=str(e)[:1000], reason="engine_error")
        except Exception as e:  # noqa: BLE001
            self._after_job(run)
            if run.cancel.is_set():
                self._finish(run, "cancelled", error="cancelled")
            else:
                log.warning("ComfyUI run %s failed: %s", run.spec.run_id, e)
                self._finish(run, "failed", error=f"{type(e).__name__}: {e}"[:1000])

    def _after_job(self, run: _Run) -> None:
        """POST /free, then (exit_after_job) stop the server before the job reports its end, so the
        next media model never loads beside it."""
        if not self._ours_running():
            return
        self._free()
        if self.exit_after_job:
            self._note(run, "stopping ComfyUI after the job (exit_after_job)")
            self._stop_process()

    def _run_ops(self, run: _Run) -> dict:
        p = run.spec.payload if isinstance(run.spec.payload, dict) else {}
        ops = p.get("ops")
        if not isinstance(ops, list) or not ops or len(ops) > 64:
            raise RunRefused("a ComfyUI job needs 1 to 64 graph ops")
        if p.get("workdir") is not None:
            wd = paths.workdir(self.artifacts_dir, p["workdir"])
        else:
            wd = paths.workdir(self.artifacts_dir, re.sub(r"[^A-Za-z0-9._-]+", "_", run.spec.run_id) + "-comfy")
        if self._healthy() is None:
            raise ConnectionError("ComfyUI is not answering")
        self._media_check()
        self._clear_stale(run)
        weights = [max(1.0, number(op.get("expected_s")) or 60.0) if isinstance(op, dict) else 1.0 for op in ops]
        total, done = sum(weights), 0.0
        results = []
        for i, (op, w) in enumerate(zip(ops, weights)):
            if run.cancel.is_set():
                raise Cancelled("cancelled")
            if not isinstance(op, dict) or op.get("op") != "graph":
                raise RunRefused(f"op {i} must be a graph op")
            if i:
                self._media_check()
            lo, hi = EVENT_CAP * done / total, EVENT_CAP * (done + w) / total
            out = paths.out_path(wd, op.get("out"))
            meta = paths.read_json(out + META)
            t0 = time.monotonic()
            if isinstance(meta, dict) and meta.get("complete") is True and os.path.isfile(out) \
                    and op.get("skip_existing", True):
                meta = dict(meta, skipped=True)
                self._note(run, f"op {i + 1}/{len(ops)} {op.get('out')}: done already, skipped")
            else:
                meta = self._run_graph(run, op, out, wd, lo, hi, w, i)
                meta.update(complete=True, op="graph", graph=op.get("graph"), out=op.get("out"),
                            seconds=round(time.monotonic() - t0, 2), model=run.spec.model, run_id=run.spec.run_id,
                            comfyui_version=self._version)
                for k in ("seed", "unit", "shot", "role"):
                    if op.get(k) is not None:
                        meta.setdefault(k, op[k])
                paths.atomic_json(out + META, meta)
            meta["path"] = out
            results.append(meta)
            done += w
            with self._lock:
                run.progress = max(run.progress, hi)
        record = os.path.join(wd, f"job-{re.sub(r'[^A-Za-z0-9._-]+', '_', run.spec.run_id)}.json")
        paths.atomic_json(record, {"kind": run.spec.kind, "run_id": run.spec.run_id, "engine": self.name,
                                   "model": run.spec.model, "comfyui_version": self._version,
                                   "ops": [{k: r.get(k) for k in ("op", "graph", "out", "seconds", "skipped", "ok",
                                                                  "seed", "unit")} for r in results],
                                   "finished_wall": time.time()})
        prim = results[-1]
        checks = []
        for r in results:
            for c in r.get("checks") or []:
                checks.append(dict(c, check=f"{os.path.basename(r['path'])}:{c.get('check')}"))
        all_paths = []
        for r in results:
            for x in (r["path"], r["path"] + META):
                if x != prim["path"]:
                    all_paths.append(x)
        art = {"path": prim["path"], "exists": all(os.path.isfile(r["path"]) for r in results),
               "paths": all_paths + [record], "checks": checks, "workdir": wd,
               "ops": [{k: r.get(k) for k in ("op", "graph", "out", "ok", "skipped", "seconds")} for r in results]}
        for k in ("width", "height"):
            if prim.get(k):
                art[k] = prim[k]
        return art

    def _media_check(self) -> None:
        """The memory rule on a server that is already up (exit_after_job off, or adopted after a
        restart): no graph runs while mlx-serve holds a media model, anyone's."""
        if self.media_probe is None:
            return
        try:
            busy = sorted(self.media_probe() or ())
        except Exception as e:
            raise EngineTrouble(f"cannot tell whether mlx-serve holds media models ({type(e).__name__}: {e}): no "
                                f"graph runs until it can be checked") from None
        if busy:
            raise EngineTrouble(f"mlx-serve holds media model(s) {busy}: no ComfyUI graph runs beside an mlx-serve "
                                f"media model")

    def _queue_ids(self) -> tuple:
        _, q = json_call(self.base_url, "GET", "/queue", timeout=self.call_timeout_s)
        q = q if isinstance(q, dict) else {}

        def ids(key):
            return [str(it[1]) for it in q.get(key) or [] if isinstance(it, list) and len(it) > 1]
        return ids("queue_running"), ids("queue_pending")

    def _clear_stale(self, run: _Run, wait_s: float = 60.0) -> None:
        """Before a job: nothing else may be queued or running on Foreman's server (one job at a time).
        A prompt left by a gpud that died (this server was adopted) or by a run that was abandoned is
        interrupted and removed first, so a retried step never renders beside its first attempt."""
        with self._lock:
            mine = {r.prompt_id for r in self._runs.values() if r is not run and r.status == "running" and r.prompt_id}
        try:
            running, pending = self._queue_ids()
        except Exception as e:
            raise EngineTrouble(f"cannot read ComfyUI's queue: {e}") from None
        stale = [p for p in running + pending if p not in mine]
        if not stale:
            with self._lock:
                self._stale_queue = False
            return
        self._note(run, f"removing {len(stale)} stale prompt(s) from ComfyUI's queue first: {stale[:4]}")
        for pid in running:
            if pid in stale:
                try:
                    json_call(self.base_url, "POST", "/interrupt", {"prompt_id": pid}, timeout=self.call_timeout_s)
                except Exception as e:
                    log.debug("ComfyUI interrupt failed: %s", e)
        end = time.monotonic() + wait_s
        while True:
            try:
                running, pending = self._queue_ids()
                left = [p for p in pending if p in stale]
                if left:
                    json_call(self.base_url, "POST", "/queue", {"delete": left}, timeout=self.call_timeout_s)
            except Exception as e:
                log.debug("ComfyUI queue cleanup: %s", e)
                running, pending = stale, []
            if not [p for p in running + pending if p in stale]:
                break
            if run.cancel.is_set():
                raise Cancelled("cancelled")
            if time.monotonic() > end:
                raise EngineTrouble(f"stale prompts {stale[:4]} did not leave ComfyUI's queue within {wait_s:.0f}s")
            time.sleep(min(0.2, self.poll_s))
        with self._lock:
            self._stale_queue = False

    def _remove_own(self, folder: str, rel: str) -> None:
        """Best effort: delete a file Foreman uploaded to ComfyUI's input folder, or an output it has
        downloaded, so neither folder grows by a take's pictures, voice and video on every job. Only a
        name that starts with "foreman" inside that folder of the ComfyUI Foreman runs is touched."""
        root = os.path.realpath(os.path.join(self.comfy_dir, folder))
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(root + os.sep) or not os.path.basename(path).startswith("foreman"):
            return
        try:
            os.remove(path)
        except OSError:
            pass

    def _upload(self, run: _Run, name: str, src: str) -> str:
        ext = os.path.splitext(src)[1].lower() or ".png"
        fname = f"foreman-{re.sub(r'[^A-Za-z0-9_-]+', '_', run.spec.run_id)}-{name}{ext}"
        with open(src, "rb") as f:
            data = f.read()
        from .mlx_ops import multipart
        ctypes = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".wav": "audio/wav",
                  ".mp4": "video/mp4"}
        body, ctype = multipart([("overwrite", "true"), ("type", "input")],
                                [("image", fname, ctypes.get(ext, "application/octet-stream"), data)])
        conn = StreamConn(self.base_url, self.call_timeout_s, poll_s=self.poll_s)
        try:
            status = conn.request("POST", "/upload/image", body, {"Content-Type": ctype}, header_timeout=60.0)
            raw = conn.read_body(1 << 20, silence_s=60.0)
        finally:
            conn.close()
        if status != 200:
            raise RunRefused(f"upload of {name} refused: HTTP {status}: {error_message(raw)}")
        try:
            obj = json.loads(raw.decode("utf-8"))
        except ValueError:
            obj = {}
        uploaded = obj.get("name") or fname
        sub = obj.get("subfolder") or ""
        return f"{sub}/{uploaded}" if sub else uploaded

    def _run_graph(self, run: _Run, op: dict, out: str, wd: str, lo: float, hi: float, weight: float, i: int) -> dict:
        tpl = load_graph(self.graphs_dir, op.get("graph"))
        values = dict(tpl["defaults"])
        params = op.get("params") or {}
        if not isinstance(params, dict):
            raise RunRefused(f"op {i}: params must be an object")
        values.update(params)
        images = op.get("images") or {}
        if not isinstance(images, dict):
            raise RunRefused(f"op {i}: images must be an object of name -> file")
        uploaded = []
        try:
            for name, src in images.items():
                if not PLACEHOLDER.fullmatch("{{" + str(name) + "}}"):
                    raise RunRefused(f"op {i}: image name {name!r} is not a placeholder name")
                real = paths.input_path(src, self.input_roots, wd)
                values[name] = self._upload(run, str(name), real)
                uploaded.append(values[name])
            return self._run_rendered(run, op, tpl, values, out, lo, hi, weight, i)
        finally:
            for rel in uploaded:  # every op uploads its own inputs again
                self._remove_own("input", rel)

    def _run_rendered(self, run: _Run, op: dict, tpl: dict, values: dict, out: str, lo: float, hi: float,
                      weight: float, i: int) -> dict:
        graph = render_graph(tpl["graph"], values)
        client = f"foreman-{run.spec.run_id}"
        try:
            _, obj = json_call(self.base_url, "POST", "/prompt", {"prompt": graph, "client_id": client},
                               timeout=max(10.0, self.call_timeout_s))
        except HttpError as e:
            if 400 <= e.status < 500:
                missing = missing_install(e.body)
                if missing:  # the machine lacks a model file or a node pack: not the graph's fault
                    raise EngineTrouble(f"ComfyUI cannot run the graph {op.get('graph')!r} here: {missing}") from None
                raise RunRefused(f"ComfyUI refused the graph {op.get('graph')!r}: {e.message} "
                                 f"{error_message(e.body)[:300]}") from None
            raise
        prompt_id = (obj or {}).get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError(f"/prompt answered without a prompt_id: {str(obj)[:200]}")
        with self._lock:
            run.prompt_id = prompt_id
            run.beat = (time.monotonic(), max(30.0, 3.0 * weight), lo, lo + 0.95 * (hi - lo))
        self._note(run, f"op {i + 1}: queued graph {op.get('graph')} as {prompt_id}")
        entry = self._wait_history(run, prompt_id, weight)
        with self._lock:
            run.beat, run.prompt_id = None, None
        status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
        if status.get("status_str") == "error":
            msgs = [m for m in status.get("messages") or [] if isinstance(m, list) and m and m[0] == "execution_error"]
            detail = json.dumps(msgs[-1][1])[:400] if msgs else "execution error"
            raise RuntimeError(f"graph {op.get('graph')} failed in ComfyUI: {detail}")
        files = self._output_files(entry.get("outputs") or {}, tpl["outputs"])
        if not files:
            raise RuntimeError(f"graph {op.get('graph')} finished without output files")
        f0 = files[0]
        q = urllib.parse.urlencode({"filename": f0.get("filename", ""), "subfolder": f0.get("subfolder", ""),
                                    "type": f0.get("type", "output")})
        conn = StreamConn(self.base_url, self.call_timeout_s, poll_s=self.poll_s)
        try:
            status_code = conn.request("GET", f"/view?{q}", None, {}, header_timeout=60.0)
            if status_code != 200:
                raise RuntimeError(f"/view answered HTTP {status_code} for {f0.get('filename')}")
            ctype = conn.headers.get("content-type", "")
            # streamed to a temp file beside `out` (renamed when complete): a cut connection, or an
            # answer over the cap, leaves nothing behind that a retry would take for a finished output
            nbytes = conn.save_body(out, MAX_OUTPUT_BYTES, silence_s=60.0)
        except ValueError as e:
            raise EngineTrouble(f"/view for {f0.get('filename')}: {e}") from None
        finally:
            conn.close()
        problem = output_problem(out, ctype, f0.get("filename") or "")
        if problem:
            try:
                os.remove(out)
            except OSError:
                pass
            raise EngineTrouble(f"/view for {f0.get('filename')} did not return the output file: {problem}")
        if (f0.get("type") or "output") == "output":
            self._remove_own("output", os.path.join(f0.get("subfolder") or "", f0.get("filename") or ""))
        meta = {"prompt_id": prompt_id, "file": f0.get("filename"), "bytes": nbytes, "license": tpl["license"],
                "commercial_ok": tpl["commercial_ok"], "content_type": ctype}
        if tpl.get("attribution"):
            meta["attribution"] = tpl["attribution"]
        size = media.image_size(out)
        if size:
            meta.update(width=size[0], height=size[1])
        meta["checks"] = [{"check": "output", "ok": nbytes > 0, "detail": f"{nbytes} bytes, {ctype or 'no type'}"}]
        meta["ok"] = nbytes > 0
        return meta

    @staticmethod
    def _output_files(outputs: dict, wanted) -> list:
        files = []
        for node in (wanted or sorted(outputs)):
            entry = outputs.get(node) if isinstance(outputs, dict) else None
            if not isinstance(entry, dict):
                continue
            for key in ("images", "gifs", "videos", "audio"):
                for f in entry.get(key) or []:
                    if isinstance(f, dict) and f.get("filename"):
                        files.append(f)
        return files

    def _wait_history(self, run: _Run, prompt_id: str, weight: float) -> dict:
        missing_since = None
        while True:
            if run.cancel.is_set():
                raise Cancelled("cancelled")
            if not self._ours_running():
                raise ConnectionError("the ComfyUI process exited while the graph ran")
            try:
                _, hist = json_call(self.base_url, "GET", f"/history/{urllib.parse.quote(prompt_id)}",
                                    timeout=self.call_timeout_s)
            except Exception as e:
                hist = None
                log.debug("ComfyUI /history failed: %s", e)
            entry = hist.get(prompt_id) if isinstance(hist, dict) else None
            if isinstance(entry, dict) and (entry.get("outputs") or (entry.get("status") or {}).get("completed")
                                            or (entry.get("status") or {}).get("status_str") == "error"):
                return entry
            try:
                _, queue = json_call(self.base_url, "GET", "/queue", timeout=self.call_timeout_s)
            except Exception:
                queue = None
            if isinstance(queue, dict):
                ids = {str(item[1]) for key in ("queue_running", "queue_pending") for item in queue.get(key) or []
                       if isinstance(item, list) and len(item) > 1}
                if prompt_id in ids:
                    missing_since = None
                elif missing_since is None:
                    missing_since = time.monotonic()
                elif time.monotonic() - missing_since > 30.0:
                    raise RuntimeError(f"prompt {prompt_id} left the queue without a history entry")
            time.sleep(self.poll_s)
