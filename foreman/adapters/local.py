"""LocalEngine: CPU work on this Mac under gpud's leases (an ad's facts, assembly and QA).

It holds no GPU memory and loads nothing: its one catalog model (family "cpu") becomes resident
the moment gpud asks for it. Each job kind maps to a handler `fn(spec, ctx) -> artifact`, run on
a worker thread. The context gives the handler what a render has: `ctx.progress(fraction)`
(only an increase renews the lease), `ctx.log(line)` for the log tail, `ctx.cancelled`, and
`ctx.run(cmd)` for subprocesses (ffmpeg), which a cancel stops (SIGTERM, then SIGKILL: these are
Foreman's own children, kept in gpud's process group so they die with it). A handler raises
ContentFailure when the work itself cannot be done (gpud records Poll.reason "rejected", a content
failure), InfraFailure when the host or the network is at fault (Poll.reason engine_error or
engine_unconfigured: infrastructure, never charged to the template); any other exception is a
crash, an infrastructure failure the ad retries. Runs live only as long as this process: after a gpud
restart find() knows nothing and the job is recorded as interrupted.
"""
from __future__ import annotations

import collections
import itertools
import logging
import os
import secrets
import signal
import subprocess
import threading
import time

from ..engine import LoadStatus, Poll, RunSpec

log = logging.getLogger("foreman.local")

EVENT_CAP = 0.95


class ContentFailure(Exception):
    """The job's input or output is wrong (not the machine): reported as a rejected request."""


class InfraFailure(RuntimeError):
    """The machine, its configuration or the network failed, not the work: a font or ffmpeg missing
    on the host, a site that answers 429. Reported as Poll.reason `reason` (engine_error or
    engine_unconfigured), an infrastructure failure however often it repeats: it never uses the
    template's weekly cap or the review budget."""

    def __init__(self, message: str, reason: str = "engine_error"):
        super().__init__(message)
        self.reason = reason if reason in ("engine_error", "engine_unconfigured") else "engine_error"


class Cancelled(Exception):
    pass


class LocalContext:
    def __init__(self, engine: "LocalEngine", run: "_Run"):
        self._engine, self._run = engine, run
        self.spec = run.spec
        self.artifacts_dir = engine.artifacts_dir

    @property
    def cancelled(self) -> bool:
        return self._run.cancel.is_set()

    def check(self) -> None:
        if self._run.cancel.is_set():
            raise Cancelled("cancelled")

    def progress(self, fraction: float) -> None:
        with self._engine._lock:
            p = self._run.progress
            self._run.progress = min(EVENT_CAP, max(p, EVENT_CAP * max(0.0, min(1.0, fraction)),
                                                    p + (EVENT_CAP - p) * 0.01))

    def log(self, line: str) -> None:
        with self._engine._lock:
            self._run.log.append(f"[local] {line}")

    def run(self, cmd: list, timeout: float = 1800.0, cwd: str | None = None, heartbeat: float | None = None):
        """subprocess.run with capture, killed on cancel. `heartbeat` (seconds): while it runs,
        progress creeps toward the next tenth (bounded), so a long ffmpeg pass is not mistaken for a
        stall but one that never ends still is."""
        self.check()
        # In gpud's own process group (no new session): when launchd stops or kills gpud's group,
        # its ffmpeg goes with it instead of racing the retried step on the same files.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, stdin=subprocess.DEVNULL)
        with self._engine._lock:
            self._run.procs.append(proc)
        t0 = time.monotonic()
        start_p = self._run.progress
        try:
            out, err = b"", b""
            while True:
                try:
                    o, e = proc.communicate(timeout=0.5)
                    out, err = out + (o or b""), err + (e or b"")
                    break
                except subprocess.TimeoutExpired:
                    if self._run.cancel.is_set():
                        _stop(proc)
                        raise Cancelled("cancelled") from None
                    if time.monotonic() - t0 > timeout:
                        _stop(proc)
                        raise RuntimeError(f"{os.path.basename(cmd[0])} ran longer than {timeout:.0f}s")
                    if heartbeat:
                        frac = min(1.0, (time.monotonic() - t0) / heartbeat)
                        with self._engine._lock:
                            target = min(EVENT_CAP, start_p + (EVENT_CAP - start_p) * 0.5 * frac)
                            if target > self._run.progress:
                                self._run.progress = target
            return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
        finally:
            with self._engine._lock:
                if proc in self._run.procs:
                    self._run.procs.remove(proc)


def _stop(proc, wait_s: float = 5.0) -> None:
    """SIGTERM a child of ours, then SIGKILL if it lingers (only that process: it shares gpud's
    process group, so the group is never signalled)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


class _Run:
    def __init__(self, handle: str, spec: RunSpec):
        self.handle, self.spec = handle, spec
        self.status, self.progress = "running", 0.0
        self.artifact, self.error, self.reason = None, None, None
        self.log = collections.deque(maxlen=40)
        self.cancel = threading.Event()
        self.procs: list = []
        self.t0 = time.monotonic()


class LocalEngine:
    """See the module docstring. `handlers` maps job kinds to functions; `catalog` entries whose
    engine is `name` are this engine's models (any of them runs any handler)."""

    def __init__(self, catalog, handlers: dict, artifacts_dir: str = "~/.foreman/artifacts",
                 name: str = "foreman-local"):
        self.name = name
        self.catalog = dict(catalog)
        self.handlers = dict(handlers)
        self.artifacts_dir = os.path.expanduser(artifacts_dir)
        self._mine = {m for m, s in self.catalog.items() if s.engine == name}
        self._loaded: set = set()
        self._boot = secrets.token_hex(4)
        self._seq = itertools.count(1)
        self._lock = threading.RLock()
        self._runs: dict = {}
        self._by_run: dict = {}

    # ---- residency: nothing to load
    def resident_models(self) -> set:
        with self._lock:
            return set(self._loaded)

    def load(self, model: str) -> str:
        if model not in self._mine:
            raise ValueError(f"{model!r} is not a {self.name} model")
        with self._lock:
            self._loaded.add(model)
        return f"{self.name}-load-{next(self._seq)}-{self._boot}"

    def poll_load(self, ticket: str) -> LoadStatus:
        return LoadStatus("ready")

    def unload(self, model: str) -> bool:
        if model not in self._mine:
            raise PermissionError(f"refusing to unload {model!r}: not a {self.name} model")
        with self._lock:
            live = [r for r in self._runs.values() if r.spec.model == model and r.status == "running"]
        for r in live:
            self._close(r, "model is being unloaded")
        with self._lock:
            self._loaded.discard(model)
        return True

    def terminate(self) -> bool:
        with self._lock:
            live = [r for r in self._runs.values() if r.status == "running"]
        for r in live:
            self._close(r, "engine terminated")
        with self._lock:
            self._loaded.clear()
        return True

    def instance_id(self) -> str:
        return f"local:{self._boot}"

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
            self._prune()
        threading.Thread(target=self._main, args=(run,), name=f"foreman-local-{spec.run_id}", daemon=True).start()
        return handle

    def _prune(self, keep: int = 200) -> None:
        done = [h for h, r in self._runs.items() if r.status != "running"]
        for h in done[:max(0, len(done) - keep)]:
            r = self._runs.pop(h)
            if self._by_run.get(r.spec.run_id) == h:
                del self._by_run[r.spec.run_id]

    def find(self, run_id: str):
        with self._lock:
            return self._by_run.get(run_id)

    def poll(self, handle: str) -> Poll:
        with self._lock:
            run = self._runs.get(handle)
            if run is None:
                return Poll("lost", log_tail=("local engine: this gpud process did not start that run",))
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
            run.log.append(f"[local] {why}")
            procs = list(run.procs)
        for p in procs:
            _stop(p)

    def close(self) -> None:
        with self._lock:
            live = [r for r in self._runs.values() if r.status == "running"]
        for r in live:
            self._close(r, "gpud shutting down")

    def live_runs(self) -> list:
        with self._lock:
            return [r.spec.run_id for r in self._runs.values() if r.status == "running"]

    def _finish(self, run: _Run, status: str, artifact=None, error=None, reason=None) -> None:
        with self._lock:
            if run.status != "running":
                return
            run.status, run.artifact, run.error, run.reason = status, artifact, error, reason
            if status == "done":
                run.progress = 1.0
            run.log.append(f"[local] {status}" + (f": {error}" if error else ""))
        log.info("local run %s %s%s", run.spec.run_id, status, f" ({error})" if error else "")

    def _main(self, run: _Run) -> None:
        handler = self.handlers.get(run.spec.kind)
        try:
            if handler is None:
                raise ContentFailure(f"the local engine has no handler for job kind {run.spec.kind!r}")
            art = handler(run.spec, LocalContext(self, run))
            if run.cancel.is_set():
                raise Cancelled("cancelled")
            self._finish(run, "done", artifact=art)
        except Cancelled:
            self._finish(run, "cancelled", error="cancelled")
        except ContentFailure as e:
            self._finish(run, "failed", error=str(e)[:1000], reason="rejected")
        except InfraFailure as e:
            if run.cancel.is_set():
                self._finish(run, "cancelled", error="cancelled")
            else:
                self._finish(run, "failed", error=str(e)[:1000], reason=e.reason)
        except Exception as e:  # noqa: BLE001 - a crash is an outcome gpud records, never a dead thread
            if run.cancel.is_set():
                self._finish(run, "cancelled", error="cancelled")
            else:
                log.exception("local run %s raised", run.spec.run_id)
                self._finish(run, "failed", error=f"{type(e).__name__}: {e}"[:1000])
