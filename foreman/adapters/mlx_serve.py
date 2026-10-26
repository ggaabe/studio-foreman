"""MlxServeEngine: foreman.engine.Engine over a running mlx-serve (v26.10.x) HTTP server.

What the adapter relies on (checked against the v26.10.3 source and the live server):

  * GET /api/ps lists READY models only, Ollama shape, names suffixed ':latest'. Models found
    by --model-dir discovery are named '<org>/<repo>'; a model loaded by absolute path is
    registered under its directory basename. Both map back to the catalog name here.
  * POST /v1/load-model {"model": id | absolute path} blocks until the model is ready (or
    fails), so loads run on a worker thread behind a ticket. POST /v1/unload-model is
    idempotent but waits for in-flight generations on that model, so live runs on it are
    cancelled first and the unload itself runs on a worker thread too.
  * A request naming an id the server does not know silently falls back to the server's
    DEFAULT model (live: a speech request for an unregistered Kokoro answered "Target model
    does not support this modality" from the user's chat model), and an unload with an
    empty id unloads the default. So every run first checks /api/ps for its exact id, and
    no request is ever sent without one.
  * Media endpoints with "stream": true send SSE `progress` events per step, then one
    `complete` event carrying base64 output (WAV; or rgb8 frames + pcm_s16le audio for video)
    or an `error` event. The complete event is stream-decoded straight to disk.
  * There is no cancel endpoint: closing the HTTP connection makes mlx-serve abort at the
    next step boundary. A run therefore lives only as long as this process: after a gpud
    restart find() knows nothing and poll() reports the old handle as 'lost'.
  * LTX goes silent between 'Decoding video' and 'complete' (VAE decode, 20 s to minutes).
    The adapter reports a bounded, time-based heartbeat during that one stage, sized from
    frames x pixels (decode_s_per_mpf, 0.6 s per megapixel-frame, at least 60 s; measured on
    the Studio: decode plus the base64 transfer take about 0.16 s per MPF), so the dead-man
    lease does not fire on a healthy decode but still fires on a decode that never ends.

Hard rules for this deployment: terminate() always refuses (mlx-serve belongs to MLX-Serve.app,
so a human decides), unload() refuses any model outside Foreman's catalog, nothing is ever
force-killed, and a load is refused unless free memory (vm_stat free + inactive + speculative,
and the server's own reading) stays `min_headroom_gb` above the model's size.

Every call made from tick() is bounded by a client-side timeout: residency and unload calls
are short HTTP requests, everything long-running happens on daemon threads.
"""
from __future__ import annotations

import binascii
import collections
import itertools
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time

from .. import media, retention
from ..engine import LoadStatus, Poll, RunSpec
from ..httpio import Cancelled, HttpError, Silence, StreamConn, error_message, json_call, split_url, text_call
from ..policy import CATALOG, ModelSpec, number
from .sse import Base64FileSink, SseError, SseReader

log = logging.getLogger("foreman.mlx_serve")

ENDPOINTS = {"tts": "/v1/audio/speech", "music": "/v1/audio/music-generations",
             "video": "/v1/video/generations"}
# Request fields forwarded from a job payload, per family. Everything else in the payload
# (goal, expected_s, duration_s, ...) is Foreman's own bookkeeping and is never sent.
FIELDS = {
    "tts": ("input", "voice", "speed", "seed"),
    "music": ("prompt", "lyrics", "instrumental", "vocal_language", "bpm", "keyscale", "timesignature",
              "seed", "duration_seconds"),
    "video": ("prompt", "num_frames", "width", "height", "seed", "steps", "stage2_steps", "pipeline",
              "cfg_scale", "cfg_audio_scale", "stg_scale", "decoder"),
}
VIDEO_FPS = 24                 # LTX's frame rate in mlx-serve v26.10.3 (fixed server-side)
EVENT_CAP = 0.95               # progress reported for events stays below this
DECODE_TOP = 0.99              # the decode heartbeat climbs from EVENT_CAP to this
FORBIDDEN_PATTERNS = ("FastH3", "MiniMax-H3")  # license excludes US users: never load or run
VIDEO_STATS = ("duration_s", "black_frame_ratio", "frozen_frame_ratio", "mean_luma", "motion_mean", "span_diff")
PARTIAL_SUFFIXES = (".partial", ".parts", ".incomplete", ".download", ".crdownload")  # download in progress


class RunFailed(Exception):
    """The engine answered, and the answer was a failure."""


class RequestRejected(RunFailed):
    """The server refused the request itself (an HTTP 4xx for its parameters, or a request the
    adapter cannot even build): a content failure, reported to gpud as Poll.reason 'rejected'."""


# 4xx answers that say "not now" or "not this server state" rather than "not these parameters".
NOT_REJECTIONS = (404, 408, 409, 423, 429)


def rejected_status(status: int, message: str) -> bool:
    """A 4xx that refuses the request's parameters. Not a 404/408/409/423/429, and not the
    "does not support this modality" answer mlx-serve gives when a request reaches its default
    model because the target is not resident (an eviction between the check and the request)."""
    return 400 <= status < 500 and status not in NOT_REJECTIONS and "modality" not in (message or "").lower()


class UnloadPending(RuntimeError):
    """The unload request is still draining server-side; gpud retries on a later tick. `pending`
    tells the supervisor this is work in progress, not an engine outage."""

    pending = True


def number_or(value, default):
    v = number(value)
    return default if v is None else v


def video_frames_for(duration_s: float, fps: int = VIDEO_FPS) -> int:
    """LTX wants 8k+1 frames: the closest such count to duration_s at fps (at least 9)."""
    k = max(1, int(round(duration_s * fps / 8.0)))
    return 8 * k + 1


def parse_start_time(text: str) -> float | None:
    """`mlx_serve:process_start_time_seconds <unix seconds>` (or any metric ending in
    process_start_time_seconds) from a Prometheus text exposition, or None."""
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        name = parts[0].split("{", 1)[0]
        if name.endswith("process_start_time_seconds") and len(parts) >= 2:
            try:
                value = float(parts[1])
            except ValueError:
                continue
            if number(value) is not None and value > 0:
                return value
    return None


def vm_stat_available_gb(timeout: float = 2.0) -> float | None:
    """(free + inactive + speculative) pages x page size, in GB (1e9), or None off macOS."""
    exe = "/usr/bin/vm_stat"
    if not os.path.exists(exe):
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=timeout, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 16384
    pages = 0
    found = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        mm = re.search(re.escape(label) + r":\s+(\d+)", out)
        if mm:
            pages += int(mm.group(1))
            found += 1
    return pages * page / 1e9 if found else None


class _Load:
    def __init__(self, ticket: str, model: str, timeout_s: float):
        self.ticket, self.model, self.timeout_s = ticket, model, timeout_s
        self.state, self.error = "loading", None
        self.deferred = False   # refused for now (free memory, a model slot): gpud retries later
        self.conn: StreamConn | None = None
        self.started = time.monotonic()


class _Unload:
    def __init__(self, model: str, server_id: str):
        self.model, self.server_id = model, server_id
        self.done = threading.Event()
        self.ok = None          # True / False once done
        self.error = None
        self.unreachable = False
        self.started = time.monotonic()
        self.finished = None


class _Run:
    def __init__(self, handle: str, spec: RunSpec, family: str, server_model: str):
        self.handle, self.spec, self.family, self.server_model = handle, spec, family, server_model
        self.status = "running"
        self.progress = 0.0
        self.artifact = None
        self.error = None
        self.log = collections.deque(maxlen=40)
        self.conn: StreamConn | None = None
        self.cancel = threading.Event()
        self.events = 0
        self.stage = None
        self.decode_at = None
        self.decode_grace = 0.0
        self.stall_after = None
        self.stalled = False
        self.reason = None      # 'rejected' for a request the server refused (Poll.reason)
        self.terminal = None    # None | 'complete' | 'error' (the last SSE event seen)
        self.error_msg = None
        self.complete = None    # (fields, streamed) of the complete event
        self.dir = None
        self._progress_line = False
        self.t0 = time.monotonic()
        self.timings: dict = {}  # stage -> seconds since start when it was first seen

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def note(self, line: str, progress: bool = False) -> None:
        """Append to the log tail; consecutive progress lines of one stage replace each other."""
        text = f"[{self.spec.model.split('/')[-1]}] {line}"
        if progress and self._progress_line and self.log:
            self.log[-1] = text
        else:
            self.log.append(text)
        self._progress_line = progress


class BrainContext:
    """What a brain implementation gets for one run: the exact server id to call, a way to open
    connections that cancel() can close, and a progress hook that renews the lease."""

    def __init__(self, engine: "MlxServeEngine", run: _Run):
        self._engine, self._run = engine, run
        self.base_url = engine.base_url
        self.server_model = run.server_model
        self.spec: ModelSpec = engine.catalog[run.spec.model]

    @property
    def cancelled(self) -> bool:
        return self._run.cancel.is_set()

    def connect(self, poll_s: float = 1.0) -> StreamConn:
        conn = StreamConn(self.base_url, self._engine.connect_timeout_s, poll_s=poll_s)
        with self._engine._lock:
            if self._run.cancel.is_set():
                raise Cancelled("run was cancelled")
            self._run.conn = conn
        return conn

    def progress(self, fraction: float) -> None:
        with self._engine._lock:
            self._engine._bump(self._run, EVENT_CAP * max(0.0, min(1.0, fraction)))

    def log(self, line: str) -> None:
        with self._engine._lock:
            self._run.note(line)


class MlxServeEngine:
    """See the module docstring. `catalog` maps Foreman model names to ModelSpec; models whose
    spec.engine is `name` are this adapter's. Catalog names should be the ids mlx-serve's own
    discovery uses ('<org>/<repo>' under --model-dir)."""

    def __init__(self, base_url: str = "http://127.0.0.1:11234", catalog=None, name: str = "mlx-serve",
                 model_dir: str = "~/.mlx-serve/models", artifacts_dir: str = "~/.foreman/artifacts",
                 brain=None, mem_probe=None, min_headroom_gb: float = 40.0,
                 max_resident_models: int | None = 3, call_timeout_s: float = 5.0,
                 connect_timeout_s: float = 5.0, read_poll_s: float = 1.0, unload_wait_s: float = 8.0,
                 unload_timeout_s: float = 600.0, load_timeout_factor: float = 4.0,
                 max_silence_s: float = 1800.0, decode_s_per_mpf: float = 0.6, min_decode_grace_s: float = 60.0,
                 sample_frames: int = 4, sample_max_side: int = 512, stall_after: int | None = None,
                 stall_kinds=None, forbidden=FORBIDDEN_PATTERNS, ffmpeg: str | None = "auto",
                 preserve_default: bool = True, keep_raw_frames: bool = False):
        split_url(base_url)  # validate early
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.catalog = dict(catalog if catalog is not None else CATALOG)
        self.model_dir = os.path.expanduser(model_dir)
        self.artifacts_dir = os.path.expanduser(artifacts_dir)
        self.brain = brain
        self.mem_probe = mem_probe
        self.min_headroom_gb = float(min_headroom_gb)
        # gpud's planner reads this optional attribute (Supervisor._make_room) so it frees a slot
        # itself rather than letting mlx-serve's LRU evict a model Foreman does not own.
        self.max_resident_models = max_resident_models
        self.call_timeout_s, self.connect_timeout_s = call_timeout_s, connect_timeout_s
        self.read_poll_s, self.unload_wait_s, self.unload_timeout_s = read_poll_s, unload_wait_s, unload_timeout_s
        self.load_timeout_factor, self.max_silence_s = load_timeout_factor, max_silence_s
        self.decode_s_per_mpf, self.min_decode_grace_s = decode_s_per_mpf, min_decode_grace_s
        self.sample_frames, self.sample_max_side = sample_frames, sample_max_side
        self.stall_after = stall_after if stall_after is None else int(stall_after)
        self.stall_kinds = tuple(stall_kinds) if stall_kinds else None
        self.forbidden = tuple(forbidden or ())
        self.ffmpeg = shutil.which("ffmpeg") if ffmpeg == "auto" else ffmpeg
        self.preserve_default = preserve_default
        # After a successful MP4 mux the raw rgb8 frames (~330 MB per 5 s at 704p) are deleted
        # unless this is set; the MP4, sampled PNGs, audio.wav and video.json stay.
        self.keep_raw_frames = bool(keep_raw_frames)
        self._mine ={m for m, s in self.catalog.items() if s.engine == name}
        self._boot = secrets.token_hex(4)
        self._seq = itertools.count(1)
        self._lock = threading.RLock()
        self._loads: dict[str, _Load] = {}
        self._unloads: dict[str, _Unload] = {}
        self._runs: dict[str, _Run] = {}
        self._by_run: dict[str, str] = {}
        self._server_ids: dict[str, str] = {}     # catalog name -> id the server uses for it
        self._sizes: dict[str, float] = {}        # model -> largest resident GB seen on this server instance
        self._resident: set[str] = set()          # last /api/ps reading, mapped
        self._instance: str | None = None         # the server's process start time (/metrics)
        self._instance_retry_at = 0.0             # /metrics unavailable: do not ask again before this
        host = split_url(base_url)[0]
        self._local = host in ("127.0.0.1", "localhost", "::1")
        if self.stall_after is not None:
            log.warning("FAULT INJECTION enabled: media runs%s stop reading mlx-serve after %d progress "
                        "events (exercises the dead-man lease; the run is cancelled by closing its "
                        "connection)", f" of kinds {list(self.stall_kinds)}" if self.stall_kinds else "",
                        self.stall_after)

    # ------------------------------------------------------------------ identity

    def _forbidden(self, model: str) -> bool:
        return any(p.lower() in model.lower() for p in self.forbidden)

    def server_id(self, model: str) -> str:
        with self._lock:
            sid = self._server_ids.get(model) or model
        if not sid or sid == "mlx-serve":
            raise ValueError(f"refusing to address mlx-serve with model id {sid!r} (it means the default model)")
        return sid

    def catalog_name(self, server_id: str) -> str:
        """Catalog name for an id mlx-serve reports; unknown ids are returned unchanged."""
        if server_id in self._mine:
            return server_id
        with self._lock:
            for name, sid in self._server_ids.items():
                if sid == server_id:
                    return name
        base = server_id.rstrip("/").rsplit("/", 1)[-1]
        for name in self._mine:
            if name.rsplit("/", 1)[-1] == base:
                return name
        return server_id

    def model_path(self, model: str) -> str:
        return os.path.join(self.model_dir, *model.split("/"))

    # ------------------------------------------------------------------ residency

    def _get(self, path: str, timeout: float | None = None):
        return json_call(self.base_url, "GET", path, timeout=timeout or self.call_timeout_s)[1]

    def _ps(self) -> dict:
        """{server id: entry} for every READY model mlx-serve reports."""
        obj = self._get("/api/ps")
        models = obj.get("models") if isinstance(obj, dict) else None
        if not isinstance(models, list):
            raise ValueError(f"unexpected /api/ps body: {str(obj)[:200]}")
        out = {}
        for m in models:
            if not isinstance(m, dict):
                continue
            raw = m.get("name") or m.get("model")
            if not isinstance(raw, str) or not raw:
                continue
            out[raw[:-len(":latest")] if raw.endswith(":latest") else raw] = m
        return out

    def resident_models(self) -> set:
        ps = self._ps()
        current, sizes, ids = set(), {}, {}
        for sid, entry in ps.items():
            name = self.catalog_name(sid)
            current.add(name)
            size = number(entry.get("size_vram")) or number(entry.get("size"))
            if size:
                sizes[name] = size / 1e9
            if name in self._mine:
                ids[name] = sid
        with self._lock:
            self._resident = current
            for name, gb in sizes.items():
                self._note_size(name, gb)
            self._server_ids.update(ids)
        return current

    def _note_size(self, model: str, gb: float) -> None:
        """Keep the largest size seen (a render grows LTX by a few GB over its resident weights)."""
        if gb > self._sizes.get(model, 0.0):
            self._sizes[model] = gb

    def model_size_gb(self, model: str) -> float | None:
        """The largest resident size mlx-serve reported for `model` on this server instance."""
        with self._lock:
            return self._sizes.get(model)

    def plan_gb(self, model: str, spec: ModelSpec) -> float:
        """Size for the memory preflight: the catalog size, or the largest size observed on this
        server instance when that is bigger."""
        with self._lock:
            return max(spec.size_gb, self._sizes.get(model, 0.0))

    def instance_id(self) -> str | None:
        """The server process's identity: `mlx_serve:process_start_time_seconds` from /metrics
        (the Unix time it started; it changes only on a restart), else the same gauge from
        /metrics.json. A new value means mlx-serve restarted, so the sizes seen on the old
        process are forgotten. None (never an exception) when neither endpoint gives it; a
        server without them is not asked again for a minute."""
        now = time.monotonic()
        if now < self._instance_retry_at:
            return self._instance
        value, missing = None, False
        try:
            status, body = self._metrics_text()
            if status == 200:
                value = parse_start_time(body)
            missing = status in (404, 501) or value is None
        except Exception as e:  # unreachable: say "unknown"; resident_models reports the outage
            log.debug("could not read /metrics: %s", e)
            return None
        if value is None:
            try:
                obj = self._get("/metrics.json", timeout=min(2.0, self.call_timeout_s)) or {}
                gauges = obj.get("gauges") if isinstance(obj, dict) else None
                value = number((gauges or {}).get("process_start_time_seconds"))
            except Exception:
                value = None
        if value is None:
            if missing:
                self._instance_retry_at = now + 60.0
            return None
        inst = f"start:{value:.0f}"
        with self._lock:
            if self._instance is not None and inst != self._instance:
                log.warning("mlx-serve restarted (process start %s -> %s)", self._instance, inst)
                self._sizes.clear()
            self._instance = inst
        return inst

    def _metrics_text(self) -> tuple:
        return text_call(self.base_url, "/metrics", timeout=min(2.0, self.call_timeout_s))

    def health(self) -> dict:
        return self._get("/health", timeout=min(3.0, self.call_timeout_s)) or {}

    def _server_alive(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    def available_gb(self) -> float | None:
        """Free memory for a load: the smallest of the local vm_stat reading (when the server is
        on this machine) and the server's own /v1/models machine.free_mib."""
        if self.mem_probe is not None:
            v = number(self.mem_probe())
            return v
        vals = []
        if self._local:
            v = vm_stat_available_gb()
            if v is not None:
                vals.append(v)
        try:
            machine = (self._get("/v1/models") or {}).get("machine") or {}
            free = number(machine.get("free_mib"))
            if free is not None:
                vals.append(free * 1048576 / 1e9)
        except Exception as e:
            log.debug("could not read machine.free_mib: %s", e)
        return min(vals) if vals else None

    # ------------------------------------------------------------------ loads

    def _check_model(self, model: str) -> ModelSpec:
        spec = self.catalog.get(model)
        if spec is None or spec.engine != self.name:
            raise ValueError(f"{model!r} is not a {self.name} model in Foreman's catalog")
        if self._forbidden(model):
            raise PermissionError(f"{model} is on the forbidden list (license) and is never loaded or run")
        return spec

    def load(self, model: str) -> str:
        spec = self._check_model(model)
        resident = self.resident_models()
        with self._lock:
            for op in self._loads.values():
                if op.model == model and (op.state == "loading" or (op.state == "ready" and model in resident)):
                    return op.ticket
            ticket = f"{self.name}-load-{next(self._seq)}-{self._boot}"
        op = _Load(ticket, model, max(60.0, spec.load_s * self.load_timeout_factor + 60.0))
        problem, deferred = (None, False) if model in resident else self._preload_problem(model, spec, resident)
        with self._lock:
            self._loads[ticket] = op
            if problem:
                op.state, op.error, op.deferred = "failed", problem, deferred
        if problem:
            log.warning("load of %s %s: %s", model, "deferred" if deferred else "refused", problem)
            return ticket
        threading.Thread(target=self._load_main, args=(op,), name=f"foreman-load-{model}", daemon=True).start()
        return ticket

    def incomplete_pack(self, model: str) -> str | None:
        """A reason to refuse loading `model` when its pack directory (on this machine) still
        holds partial download files, else None. MLX-Serve.app downloads in place
        (`<file>.partial` plus `<file>.partial.parts`), so a pack can sit in --model-dir for
        hours before it is complete. A directory that is not here (a remote server, a test
        double) is not checked."""
        try:
            names = os.listdir(self.model_path(model))
        except OSError:
            return None
        partial = sorted(n for n in names if n.endswith(PARTIAL_SUFFIXES) or ".partial." in n)
        if not partial:
            return None
        return (f"{model} is still downloading ({len(partial)} partial file(s) in {self.model_path(model)}, "
                f"e.g. {partial[0]}); refusing to load an incomplete pack")

    def _preload_problem(self, model: str, spec: ModelSpec, resident: set) -> tuple:
        """(reason to refuse the load, deferred) or (None, False). `deferred` marks a refusal that
        says "not now" rather than "not this model": too little free memory beside the host's other
        tenants, or every mlx-serve slot taken. gpud then keeps a media job waiting and retries."""
        partial = self.incomplete_pack(model)
        if partial:
            return partial, False
        cap = self.max_resident_models
        if cap and len(resident) >= cap:
            return (f"mlx-serve already holds {len(resident)} models (--max-resident-models {cap}); loading "
                    f"{model} would make it LRU-evict one, possibly a model Foreman does not own: "
                    f"{sorted(resident)}"), True
        avail = self.available_gb()
        if avail is None:
            return "cannot measure free memory (vm_stat and /v1/models machine.free_mib both unavailable)", False
        need = self.plan_gb(model, spec)
        if avail - need < self.min_headroom_gb:
            return (f"insufficient headroom: {avail:.1f} GB free, {model} needs ~{need:g} GB and "
                    f"{self.min_headroom_gb:g} GB must stay free"), True
        return None, False

    def _load_target(self, model: str) -> str:
        """The id to send to /v1/load-model: the discovered id if the server lists one, else the
        model's absolute directory (registered under its basename)."""
        try:
            data = (self._get("/v1/models") or {}).get("data") or []
            ids = {e.get("id") for e in data if isinstance(e, dict)}
        except Exception:
            ids = set()
        with self._lock:
            known = self._server_ids.get(model)
        for cand in (known, model):
            if cand and cand in ids:
                return cand
        return self.model_path(model)

    def _server_default(self) -> str | None:
        """mlx-serve's default model. /v1/models has no explicit field: it lists the default first,
        and with no default the most recently used model comes first instead. Only a chat-capable
        model can be the default, so anything else in first place means there is none."""
        data = (self._get("/v1/models") or {}).get("data") or []
        first = data[0] if data and isinstance(data[0], dict) else {}
        sid, caps = first.get("id"), first.get("capabilities") or ()
        return sid if isinstance(sid, str) and sid and "chat" in caps else None

    def _restore_default(self, prior: str | None, loaded: str) -> None:
        """A server started without --model promotes the latest chat-capable load to its default
        model, so loading the brain would silently take over every request that names no model.
        If that happened to someone else's model, point the default back at it. "default": true
        pins it explicitly, so later loads no longer steal it."""
        if not self.preserve_default or not prior or prior == loaded or self.catalog_name(prior) in self._mine:
            return
        try:
            if self._server_default() != loaded or prior not in self._ps():
                return
            json_call(self.base_url, "POST", "/v1/load-model", {"model": prior, "default": True},
                      timeout=max(self.call_timeout_s, 10.0))
            log.info("restored mlx-serve's default model to %r (loading %r had taken it over)", prior, loaded)
        except Exception as e:
            log.warning("could not restore mlx-serve's default model to %r: %s", prior, e)

    def _load_main(self, op: _Load) -> None:
        conn = None
        try:
            try:
                prior_default = self._server_default() if self.preserve_default else None
            except Exception:
                prior_default = None
            target = self._load_target(op.model)
            conn = StreamConn(self.base_url, self.connect_timeout_s, poll_s=self.read_poll_s)
            with self._lock:
                if op.state != "loading":
                    return
                op.conn = conn
            log.info("loading %s via POST /v1/load-model %s", op.model, target)
            status = conn.request("POST", "/v1/load-model", json.dumps({"model": target}).encode(),
                                  {"Content-Type": "application/json"}, header_timeout=op.timeout_s)
            body = conn.read_body(1 << 20, silence_s=30.0)
            if status != 200:
                raise RunFailed(f"HTTP {status}: {error_message(body)}")
            obj = json.loads(body.decode("utf-8"))
            entry = obj.get("model") if isinstance(obj, dict) else None
            sid = entry.get("id") if isinstance(entry, dict) else None
            if not isinstance(sid, str) or not sid:
                raise RunFailed(f"load-model answered without a model id: {body[:200]!r}")
            with self._lock:
                self._server_ids[op.model] = sid
                size = number(entry.get("bytes_resident"))
                if size:
                    self._note_size(op.model, size / 1e9)
                if op.state == "loading":
                    op.state = "ready"
                    self._resident.add(op.model)
            log.info("loaded %s as %r in %.1fs", op.model, sid, time.monotonic() - op.started)
            self._restore_default(prior_default, sid)
        except Cancelled:
            self._fail_load(op, "load aborted (connection closed by gpud)")
        except Silence:
            self._fail_load(op, f"load-model did not answer within {op.timeout_s:.0f}s")
        except Exception as e:
            self._fail_load(op, f"{type(e).__name__}: {e}")
        finally:
            if conn is not None:
                conn.close()

    def _fail_load(self, op: _Load, error: str) -> None:
        with self._lock:
            if op.state == "loading":
                op.state, op.error = "failed", error
        log.warning("load of %s failed: %s", op.model, error)

    def poll_load(self, ticket: str) -> LoadStatus:
        with self._lock:
            op = self._loads.get(ticket)
            if op is None:
                return LoadStatus("failed", "unknown load ticket (issued by an earlier gpud process)")
            return LoadStatus(op.state, op.error, op.deferred)

    # ------------------------------------------------------------------ unload / terminate

    def unload(self, model: str) -> bool:
        if model not in self._mine:
            raise PermissionError(f"refusing to unload {model!r}: it is not a Foreman catalog model")
        sid = self.server_id(model)
        with self._lock:
            for op in self._loads.values():
                if op.model == model and op.state == "loading":
                    op.state, op.error = "failed", "aborted by unload"
                    if op.conn is not None:
                        op.conn.close()
            live = [r for r in self._runs.values() if r.spec.model == model and r.status == "running"]
            job = self._unloads.get(model)
            if job is not None and job.finished is not None and time.monotonic() - job.finished > 30.0:
                job = None  # a result nobody collected (the caller moved on): ask the server again
            if job is None:
                job = _Unload(model, sid)
                self._unloads[model] = job
                threading.Thread(target=self._unload_main, args=(job,), name=f"foreman-unload-{model}",
                                 daemon=True).start()
        for run in live:  # unload waits for in-flight generations: stop ours first
            self._close_run(run, "model is being unloaded")
        if not job.done.wait(self.unload_wait_s):
            waited = time.monotonic() - job.started
            if waited >= self.unload_timeout_s:
                with self._lock:
                    self._unloads.pop(model, None)
                log.error("unload of %s has not finished after %.0fs: treating it as refused", model, waited)
                return False
            raise UnloadPending(f"unload of {model} still draining after {waited:.0f}s")
        with self._lock:
            self._unloads.pop(model, None)
            if job.ok:  # the largest size seen stays: it plans the next load of this model
                self._resident.discard(model)
        if job.unreachable:
            raise ConnectionError(job.error or "mlx-serve unreachable")
        if not job.ok:
            log.error("mlx-serve refused to unload %s: %s", model, job.error)
        return bool(job.ok)

    def _unload_main(self, job: _Unload) -> None:
        try:
            json_call(self.base_url, "POST", "/v1/unload-model", {"model": job.server_id},
                      timeout=self.unload_timeout_s)
            job.ok = True
            log.info("unloaded %s (%s)", job.model, job.server_id)
        except HttpError as e:
            if e.status == 404:  # nothing registered under that id: nothing to unload
                job.ok = True
            else:
                job.ok, job.error = False, str(e)
        except OSError as e:
            job.ok, job.error, job.unreachable = False, f"{type(e).__name__}: {e}", True
        except Exception as e:
            job.ok, job.error = False, f"{type(e).__name__}: {e}"
        finally:
            job.finished = time.monotonic()
            job.done.set()

    def terminate(self) -> bool:
        log.warning("terminate() refused: mlx-serve is run by MLX-Serve.app (MLXCore), not by gpud. Foreman "
                    "never stops or restarts it; a human must decide (SAFE mode pages one).")
        return False

    # ------------------------------------------------------------------ runs

    def start(self, spec: RunSpec) -> str:
        with self._lock:
            handle = self._by_run.get(spec.run_id)
            if handle:
                return handle
        mspec = self._check_model(spec.model)
        handle = f"{self.name}:{self._boot}:{spec.run_id}"
        run = _Run(handle, spec, mspec.family, self.server_id(spec.model))
        if self.stall_after is not None and mspec.family != "brain" and \
                (self.stall_kinds is None or spec.kind in self.stall_kinds):
            run.stall_after = self.stall_after
        with self._lock:
            self._prune()
            self._runs[handle] = run
            self._by_run[spec.run_id] = handle
            run.note(f"start {spec.kind} (run {spec.run_id})")
        threading.Thread(target=self._run_main, args=(run,), name=f"foreman-run-{spec.run_id}", daemon=True).start()
        return handle

    def _prune(self, keep: int = 500) -> None:
        """Forget the oldest finished runs and loads (gpud stops polling a job once it is
        terminal), so a long-lived daemon does not grow without bound."""
        done = [h for h, r in self._runs.items() if r.status != "running"]
        for h in done[:max(0, len(done) - keep)]:
            run = self._runs.pop(h)
            if self._by_run.get(run.spec.run_id) == h:
                del self._by_run[run.spec.run_id]
        finished = [t for t, op in self._loads.items() if op.state != "loading"]
        for t in finished[:max(0, len(finished) - 50)]:
            del self._loads[t]

    def find(self, run_id: str) -> str | None:
        with self._lock:
            return self._by_run.get(run_id)

    def poll(self, handle: str) -> Poll:
        with self._lock:
            run = self._runs.get(handle)
            if run is None:
                return Poll("lost", log_tail=(
                    "mlx-serve adapter: this gpud process does not own that run. Its HTTP stream died with "
                    "the process that started it, and mlx-serve aborts a run whose client disconnects.",))
            if run.status == "running" and run.decode_at is not None and not run.stalled:
                elapsed = time.monotonic() - run.decode_at
                frac = min(1.0, elapsed / run.decode_grace) if run.decode_grace > 0 else 1.0
                beat = EVENT_CAP + (DECODE_TOP - EVENT_CAP) * frac
                if beat > run.progress:
                    run.progress = beat
            return Poll(run.status, run.progress, run.artifact, run.error, tuple(run.log)[-8:], run.reason)

    def cancel(self, handle: str) -> None:
        with self._lock:
            run = self._runs.get(handle)
        if run is None:
            raise LookupError(f"unknown run handle {handle!r} (not started by this gpud process)")
        self._close_run(run, "cancel requested by gpud")

    def _close_run(self, run: _Run, why: str) -> None:
        with self._lock:
            if run.status != "running" or run.cancel.is_set():
                return
            run.cancel.set()
            run.note(f"{why}: closing the HTTP connection (mlx-serve aborts at the next step boundary)")
            if run.complete is None and run.terminal is None:
                run.status, run.error = "cancelled", why
            conn = run.conn
        if conn is not None:
            conn.close()

    def close(self) -> None:
        """Process shutdown: close every connection this process holds (mlx-serve aborts those
        runs at their next step boundary; a restarted gpud records them as interrupted)."""
        with self._lock:
            runs = [r for r in self._runs.values() if r.status == "running"]
            loads = [op for op in self._loads.values() if op.state == "loading" and op.conn is not None]
        for r in runs:
            log.warning("shutdown: abandoning run %s (%s)", r.spec.run_id, r.spec.kind)
            self._close_run(r, "gpud shutting down")
        for op in loads:
            op.conn.close()

    def live_runs(self) -> list:
        with self._lock:
            return [r.spec.run_id for r in self._runs.values() if r.status == "running"]

    # ---- run threads

    def _finish(self, run: _Run, status: str, artifact=None, error: str | None = None,
                reason: str | None = None) -> None:
        with self._lock:
            if run.status != "running":
                return
            run.status, run.artifact, run.error, run.reason = status, artifact, error, reason
            if status == "done":
                run.progress = 1.0
            run.note(f"{status}" + (f": {error}" if error else ""))
            conn = run.conn
        if conn is not None:
            conn.close()
        log.info("run %s %s%s", run.spec.run_id, status, f" ({error})" if error else "")

    def _run_main(self, run: _Run) -> None:
        try:
            if run.family == "brain":
                self._run_brain(run)
            elif run.family in ENDPOINTS:
                self._run_media(run)
            else:
                raise RunFailed(f"no mlx-serve endpoint is wired for model family {run.family!r}")
        except Cancelled:
            self._finish(run, "cancelled", error="cancelled (HTTP connection closed)")
        except RunFailed as e:
            self._finish(run, "failed", error=str(e), reason="rejected" if isinstance(e, RequestRejected) else None)
        except Exception as e:
            if run.cancel.is_set():
                self._finish(run, "cancelled", error="cancelled (HTTP connection closed)")
            else:
                status = "failed" if self._server_alive() else "lost"
                detail = f"{type(e).__name__}: {e}"
                if status == "lost":
                    detail += " (mlx-serve no longer answers /health)"
                log.warning("run %s %s: %s", run.spec.run_id, status, detail)
                self._finish(run, status, error=detail)
        finally:
            conn = run.conn
            if conn is not None:
                conn.close()

    def _check_resident(self, run: _Run) -> None:
        """Pin the exact id mlx-serve has the model under, or refuse: a request naming an id the
        server does not have would be answered by its default model."""
        ps = self._ps()
        sids = [sid for sid in ps if self.catalog_name(sid) == run.spec.model]
        if not sids:
            raise RunFailed(f"{run.spec.model} is not resident in mlx-serve; refusing to send the request "
                            f"(mlx-serve would answer it with its default model)")
        sid = run.server_model if run.server_model in sids else sids[0]
        with self._lock:
            run.server_model = sid
            self._server_ids[run.spec.model] = sid

    def _run_brain(self, run: _Run) -> None:
        if self.brain is None:
            raise RunFailed("no brain client is configured for this engine")
        self._check_resident(run)
        if getattr(self.brain, "wants_context", False):
            decision = self.brain(run.spec, run.spec.model, BrainContext(self, run))
        else:
            decision = self.brain(run.spec, run.spec.model)
        if run.cancel.is_set():
            raise Cancelled("run was cancelled")
        self._finish(run, "done", artifact={"decision": decision, "model": run.spec.model})

    def request_body(self, run_or_spec) -> dict:
        spec = run_or_spec.spec if isinstance(run_or_spec, _Run) else run_or_spec
        family = self.catalog[spec.model].family
        p = spec.payload if isinstance(spec.payload, dict) else {}
        body = {k: p[k] for k in FIELDS.get(family, ()) if p.get(k) is not None}
        if family == "tts":
            text = p.get("input") or p.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RequestRejected("speech job has no 'input' text")
            body["input"] = text
        elif family == "music":
            if "duration_seconds" not in body and number(p.get("duration_s")):
                body["duration_seconds"] = int(round(number(p.get("duration_s"))))
            if not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
                raise RequestRejected("music job has no 'prompt'")
        elif family == "video":
            if "num_frames" not in body:
                body["num_frames"] = video_frames_for(number_or(p.get("duration_s"), 5.0))
            body.setdefault("width", 768)
            body.setdefault("height", 512)
            if not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
                raise RequestRejected("video job has no 'prompt'")
        body["model"] = self.server_id(spec.model)
        body["stream"] = True
        return body

    def _run_media(self, run: _Run) -> None:
        self._check_resident(run)
        body = self.request_body(run)
        body["model"] = run.server_model
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", run.spec.run_id)
        run.dir = os.path.join(self.artifacts_dir, f"{safe}-{self._boot}")
        os.makedirs(run.dir, exist_ok=True)
        if run.family == "video":
            mpf = body["num_frames"] * body["width"] * body["height"] / 1e6
            run.decode_grace = max(self.min_decode_grace_s, self.decode_s_per_mpf * mpf)
        conn = StreamConn(self.base_url, self.connect_timeout_s, poll_s=self.read_poll_s)
        with self._lock:
            if run.cancel.is_set():
                raise Cancelled("cancelled before the request was sent")
            run.conn = conn
            run.note(f"POST {ENDPOINTS[run.family]} " + json.dumps(
                {k: (v if not isinstance(v, str) or len(v) < 80 else v[:77] + "...") for k, v in body.items()}))
        status = conn.request("POST", ENDPOINTS[run.family], json.dumps(body).encode(),
                              {"Content-Type": "application/json", "Accept": "text/event-stream"},
                              header_timeout=self.max_silence_s)
        if status != 200:
            message = error_message(conn.read_body(1 << 20))
            if rejected_status(status, message):
                raise RequestRejected(f"HTTP {status}: {message}")
            raise RunFailed(f"HTTP {status}: {message}")
        ctype = conn.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            raise RunFailed(f"expected an SSE stream, got {ctype!r}: {error_message(conn.read_body(1 << 20))}")
        reader = SseReader(lambda f, s: self._on_event(run, f, s), lambda t: self._on_comment(run, t),
                           lambda head: self._sink_factory(run, head))
        try:
            for chunk in conn.body_chunks(silence_s=self.max_silence_s):
                reader.feed(chunk)
                if run.stalled:
                    # Fault injection: stop reading and reporting. Only closing the connection (the
                    # supervisor's cancel after the lease expires) ends this run.
                    run.cancel.wait()
                    raise Cancelled("fault injection: run was stalled until cancelled")
                if run.terminal is not None:
                    break
        except Silence as e:
            raise RunFailed(f"mlx-serve sent nothing for {self.max_silence_s:.0f}s ({e})") from None
        except SseError as e:
            reader.abort()
            raise RunFailed(f"unparseable SSE stream: {e}") from None
        if run.terminal == "error":
            raise RunFailed(f"mlx-serve: {run.error_msg or 'generation failed'}")
        if run.terminal != "complete":
            raise ConnectionError(f"stream ended after {run.events} progress events without a complete or "
                                  f"error event")
        artifact = self._artifact(run)
        run.timings["artifact"] = round(run.elapsed(), 1)
        artifact["timings_s"] = dict(run.timings)
        log.info("run %s timings %s", run.spec.run_id, run.timings)
        self._finish(run, "done", artifact=artifact)

    def _bump(self, run: _Run, mapped: float) -> None:
        """Strictly increase progress on every observed event (the lease renews only on an
        increase): the mapped step fraction, or a small step toward EVENT_CAP when the step
        counter restarted or repeated."""
        p = run.progress
        run.progress = min(EVENT_CAP, max(p, mapped, p + (EVENT_CAP - p) * 0.02))

    def _on_event(self, run: _Run, fields: dict, streamed: dict) -> None:
        kind = fields.get("type")
        if kind == "progress":
            if run.stalled:
                return
            stage = str(fields.get("stage") or "")
            step, total = number(fields.get("step")), number(fields.get("total"))
            with self._lock:
                run.events += 1
                frac = step / total if step is not None and total else None
                self._bump(run, 0.9 * max(0.0, min(1.0, frac)) if frac is not None else 0.0)
                line = f"{stage} {int(step) if step is not None else '?'}/{int(total) if total else '?'}"
                if stage != run.stage:
                    run.timings.setdefault(stage or "?", round(run.elapsed(), 1))
                    log.info("run %s: %s at +%.1fs", run.spec.run_id, line, run.elapsed())
                run.note(f"{line} (+{run.elapsed():.1f}s)", progress=run.stage == stage)
                if stage != run.stage and stage.lower().startswith("decoding") and run.family == "video":
                    run.decode_at = time.monotonic()
                    run.progress = max(run.progress, EVENT_CAP)
                    run.note(f"{stage}: mlx-serve sends no events until complete; allowing up to "
                             f"{run.decode_grace:.0f}s (bounded decode heartbeat)")
                run.stage = stage
                if run.stall_after is not None and run.events >= run.stall_after:
                    run.stalled = True
                    run.note(f"FAULT INJECTION: stopped reading and reporting after {run.events} progress events")
                    log.warning("fault injection: run %s stalled after %d events", run.spec.run_id, run.events)
        elif kind == "complete":
            with self._lock:
                run.complete = (fields, streamed)
                run.terminal = "complete"
                run.timings["complete"] = round(run.elapsed(), 1)
                run.note(f"complete event received (+{run.elapsed():.1f}s)")
        elif kind == "error":
            with self._lock:
                run.terminal, run.error_msg = "error", str(fields.get("message") or "generation failed")
                run.note(f"ERROR: {run.error_msg}")
        else:
            with self._lock:
                run.note(f"ignored event {str(kind)[:40]}")

    def _on_comment(self, run: _Run, text: str) -> None:
        if run.stalled:
            return
        with self._lock:  # ': keepalive' during a CPU stage: the server is alive and working on it
            p = run.progress
            run.progress = min(EVENT_CAP, p + (EVENT_CAP - p) * 0.005)

    def _sink_factory(self, run: _Run, head: bytes):
        if b'"complete"' not in head:
            return None

        def factory(key: str, fields: dict):
            if key == "data":
                fmt = str(fields.get("format") or "")
                name = "frames.rgb" if fmt == "rgb8" else f"output.{fmt or 'bin'}"
                return Base64FileSink(os.path.join(run.dir, name))
            if key == "audio_data":
                return Base64FileSink(os.path.join(run.dir, "audio.pcm"))
            return None
        return factory

    # ---- artifacts

    def _materialize(self, run: _Run, key: str, filename: str) -> str | None:
        fields, streamed = run.complete
        if key in streamed:
            return streamed[key].path
        value = fields.get(key)
        if isinstance(value, str) and value:
            path = os.path.join(run.dir, filename)
            with open(path, "wb") as f:
                f.write(binascii.a2b_base64(value))
            return path
        return None

    def _artifact(self, run: _Run) -> dict:
        fields, _ = run.complete
        if run.family in ("tts", "music"):
            fmt = str(fields.get("format") or "wav")
            path = self._materialize(run, "data", f"output.{fmt}")
            if path is None:
                raise RunFailed("complete event carried no audio data")
            art = {"path": path, "exists": os.path.getsize(path) > 0, "format": fmt, "bytes": os.path.getsize(path)}
            try:
                stats = media.wav_stats(path)
                art.update({k: stats[k] for k in ("duration_s", "silence_ratio", "rms_dbfs", "peak",
                                                  "sample_rate", "channels")})
                wf = os.path.join(run.dir, "waveform.png")
                with open(wf, "wb") as f:
                    f.write(media.waveform_png(path))
                art["paths"] = [wf]
            except media.MediaError as e:
                art["probe_error"] = str(e)  # validators then fail the silence/duration checks
            return art
        # video
        width, height = int(number_or(fields.get("width"), 0)), int(number_or(fields.get("height"), 0))
        frames, fps = int(number_or(fields.get("frames"), 0)), number_or(fields.get("fps"), VIDEO_FPS)
        frames_path = self._materialize(run, "data", "frames.rgb")
        if frames_path is None:
            raise RunFailed("complete event carried no frames")
        art = {"path": frames_path, "frames_path": frames_path, "exists": True, "format": "rgb8",
               "width": width, "height": height, "fps": fps, "frame_count": frames, "paths": []}
        try:
            art.update({k: v for k, v in media.video_stats(frames_path, width, height, frames, fps).items()
                        if k in VIDEO_STATS})
            samples = []
            for i, png in media.sample_frames_png(frames_path, width, height, frames, self.sample_frames,
                                                  self.sample_max_side):
                p = os.path.join(run.dir, f"frame_{i:04d}.png")
                with open(p, "wb") as f:
                    f.write(png)
                samples.append(p)
            art["samples"] = samples
            art["paths"] += samples
        except media.MediaError as e:
            art["exists"] = os.path.exists(frames_path)
            art["probe_error"] = str(e)
        audio_pcm = self._materialize(run, "audio_data", "audio.pcm")
        if audio_pcm:
            sr = int(number_or(fields.get("audio_sample_rate"), 0))
            ch = int(number_or(fields.get("audio_channels"), 2))
            if sr > 0 and ch > 0:
                wav = os.path.join(run.dir, "audio.wav")
                media.wav_from_pcm_file(audio_pcm, wav, sr, ch)
                os.remove(audio_pcm)
                art["audio"] = wav
                art["paths"].append(wav)
        meta = {"width": width, "height": height, "fps": fps, "frame_count": frames, "format": "rgb8",
                "frames": os.path.basename(frames_path), "audio": os.path.basename(art["audio"]) if art.get("audio") else None,
                "audio_sample_rate": fields.get("audio_sample_rate"), "audio_channels": fields.get("audio_channels"),
                "model": run.spec.model, "run_id": run.spec.run_id,
                "stats": {k: art.get(k) for k in VIDEO_STATS}}
        meta_path = os.path.join(run.dir, "video.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        art["meta"] = meta_path
        art["paths"].insert(0, meta_path)
        mp4 = self._mux(run, frames_path, width, height, fps, art.get("audio")) if self.ffmpeg else None
        if mp4:
            art.update(path=mp4, format="mp4")
            retention.drop_raw_frames(art, self.keep_raw_frames, run.note)
        return art

    def _mux(self, run: _Run, frames_path: str, width: int, height: int, fps: float, wav: str | None) -> str | None:
        """MP4 from the raw frames (+ audio). Apple's hardware encoder first, libx264 if the build
        lacks it or it refuses the job."""
        out = os.path.join(run.dir, "output.mp4")
        head = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", frames_path]
        if wav:
            head += ["-i", wav, "-c:a", "aac", "-shortest"]
        bitrate = max(1_000_000, int(width * height * fps * 0.12))
        encoders = (["-c:v", "h264_videotoolbox", "-b:v", str(bitrate), "-allow_sw", "1"],
                    ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
        tail = ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        for enc in encoders:
            try:
                res = subprocess.run(head + enc + tail, capture_output=True, timeout=600, check=False)
            except (OSError, subprocess.SubprocessError) as e:
                run.note(f"ffmpeg mux failed: {e}")
                return None
            if res.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
                run.note(f"muxed {os.path.basename(out)} with {enc[1]}")
                return out
            run.note(f"ffmpeg {enc[1]} failed: {res.stderr[-300:].decode('utf-8', 'replace')}")
        return None
