"""Deterministic simulation: a virtual clock, fake engines with scripted faults, a stateless
scripted brain, and a harness that can kill and restart the supervisor at any point.

Engines are lazy functions of the clock, so they keep "running" while the supervisor is dead.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import struct
import zlib
from dataclasses import dataclass, field

from .engine import LoadStatus, Poll, RunSpec
from .policy import CATALOG, Policy
from .store import ACTIVE
from .supervisor import PLAN_KIND, Supervisor

TIGHT_HOST_GB = 176.0  # 160 GB usable: the video model fits, video + any brain does not


class VirtualClock:
    def __init__(self, start: float = 0.0):
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def monotonic(self) -> float:
        """Virtual time never steps, so it doubles as the monotonic source."""
        return self._t

    def advance(self, dt: float) -> float:
        self._t += dt
        return self._t


class SimulatedCrash(BaseException):
    """Emulates the supervisor process dying. BaseException so no handler swallows it."""


class CrashAt:
    """Crash hook: kill the supervisor the nth time it reaches `point`."""

    def __init__(self, point: str, nth: int = 1):
        self.point, self.nth, self.hits, self.fired = point, nth, 0, False

    def __call__(self, point: str) -> None:
        if point == self.point and not self.fired:
            self.hits += 1
            if self.hits == self.nth:
                self.fired = True
                raise SimulatedCrash(point)


# ---------------------------------------------------------------- fault scripts

@dataclass(frozen=True)
class Behaviour:
    mode: str = "success"         # success | hang | slow | crash | garbage
    at: float = 0.4               # progress fraction where a hang or crash happens
    factor: float = 1.0           # slow: duration multiplier (still reports progress)
    artifact: dict = field(default_factory=dict)  # garbage: artifact overrides
    ignore_cancel: bool = False   # only an unload stops it
    wedged: bool = False          # ignores cancel and unload; only terminate() stops it
    restart_engine: bool = False  # the crash takes the whole engine process down


SUCCESS = Behaviour()


def hang(at: float = 0.4, **kw) -> Behaviour:
    return Behaviour("hang", at=at, **kw)


def slow(factor: float) -> Behaviour:
    return Behaviour("slow", factor=factor)


def crash_mid_job(at: float = 0.5, restart_engine: bool = False) -> Behaviour:
    return Behaviour("crash", at=at, restart_engine=restart_engine)


def garbage(**artifact) -> Behaviour:
    return Behaviour("garbage", artifact=artifact)


_GARBAGE = {"video": {"black_frame_ratio": 1.0}, "music": {"silence_ratio": 1.0},
            "tts": {"silence_ratio": 1.0}, "image": {"width": 0}, "llm": {"exists": False}}
_EXT = {"video": "mp4", "music": "wav", "tts": "wav", "image": "png", "llm": "txt"}


def _tiny_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + chunk(b"IEND", b""))


def _write_files(art: dict, family: str, spec: RunSpec, root: str) -> None:
    """FakeEngine.artifacts_dir: put small real files on disk the way the mlx-serve adapter lays
    a run out (<root>/<run id>-sim/...), so retention can be tested against a real tree."""
    d = os.path.join(root, f"{spec.run_id}-sim")
    os.makedirs(d, exist_ok=True)
    main = os.path.join(d, "output.mp4" if family == "video" else f"output.{_EXT.get(family, 'bin')}")
    with open(main, "wb") as f:
        f.write(b"\x00" * 4096)
    paths = []
    if family == "video":
        for name, data in (("video.json", json.dumps({"run_id": spec.run_id}).encode()), ("audio.wav", b"\x00" * 512),
                           ("frame_0000.png", _tiny_png()), ("frame_0001.png", _tiny_png())):
            with open(os.path.join(d, name), "wb") as f:
                f.write(data)
            paths.append(os.path.join(d, name))
    elif family in ("music", "tts"):
        with open(os.path.join(d, "waveform.png"), "wb") as f:
            f.write(_tiny_png())
        paths.append(os.path.join(d, "waveform.png"))
    art.update(path=main, paths=paths)


def _artifact(family: str, spec: RunSpec, behaviour: Behaviour, root: str | None = None) -> dict:
    p = spec.payload
    art = {"path": f"/sim/out/{spec.run_id}.{_EXT.get(family, 'bin')}", "exists": True}
    if root:
        _write_files(art, family, spec, root)
    if family == "video":
        art.update(duration_s=p.get("duration_s", 5.0), black_frame_ratio=0.0)
    elif family in ("music", "tts"):
        art.update(duration_s=p.get("duration_s", 10.0), silence_ratio=0.02)
    elif family == "image":
        art.update(width=1024, height=1024)
    if behaviour.mode == "garbage":
        art.update(behaviour.artifact or _GARBAGE.get(family, {"exists": False}))
    return art


class SimHost:
    """Unified memory shared by every engine process on the box."""

    def __init__(self, capacity_gb: float):
        self.capacity_gb = capacity_gb
        self.engines: list[FakeEngine] = []
        self.oom: list[tuple] = []

    def used_gb(self) -> float:
        return sum(e.memory_gb() for e in self.engines)


@dataclass
class _Load:
    ticket: str
    model: str
    started: float
    done_at: float
    outcome: str                  # ok | fail | hang | oom
    state: str = "loading"
    error: str | None = None


@dataclass
class _Run:
    handle: str
    spec: RunSpec
    behaviour: Behaviour
    started: float
    duration: float
    state: str = "running"
    steps: int = 0
    progress: float = 0.0
    artifact: dict | None = None
    error: str | None = None
    cancel_at: float | None = None
    lost: bool = False
    log: list = field(default_factory=list)


class FakeEngine:
    """Implements foreman.engine.Engine. One process, possibly many models, scripted faults."""

    STEPS = 20
    CANCEL_LATENCY_S = 2.0

    def __init__(self, name: str, clock: VirtualClock, host: SimHost, catalog=CATALOG, brain=None):
        self.name, self.clock, self.host, self.catalog, self.brain = name, clock, host, catalog, brain
        host.engines.append(self)
        self._models: set[str] = set()
        self._loads: dict[str, _Load] = {}
        self._runs: dict[str, _Run] = {}
        self._by_run: dict[str, str] = {}
        self._job_scripts = collections.defaultdict(collections.deque)
        self._load_scripts = collections.defaultdict(collections.deque)
        self._seq = 0
        self.calls: list[tuple[float, str, str]] = []  # state-changing calls, for assertions
        self.terminate_fails = False
        self.restarts = 0
        self.artifacts_dir: str | None = None  # set: media runs write small real files under it

    # ---- test-side scripting ----
    def script_jobs(self, kind: str, *behaviours: Behaviour) -> None:
        self._job_scripts[kind].extend(behaviours)

    def script_loads(self, model: str, *outcomes: str) -> None:
        """Queue load outcomes for a model: 'ok', 'fail' or 'hang'."""
        self._load_scripts[model].extend(outcomes)

    def evict(self, model: str) -> None:
        """The engine drops a model on its own (memory pressure, idle TTL, ...)."""
        self._sync()
        self.calls.append((self.clock.now(), "evict", model))
        self._models.discard(model)
        for run in self._live_runs(model):
            run.state, run.error = "failed", f"model {model} evicted by engine"
            run.log.append(f"[{model}] ERROR: {run.error}")

    def operator_force_restart(self) -> None:
        """What a human does with kill -9. Not part of the Engine interface."""
        self.calls.append((self.clock.now(), "operator_force_restart", ""))
        self.terminate_fails = False
        self._restart("operator force restart")

    def model_size_gb(self, model: str) -> float | None:
        """Optional Engine extension: footprint of a model gpud's catalog may not know."""
        spec = self.catalog.get(model)
        return spec.size_gb if spec else None

    def memory_gb(self) -> float:
        self._sync()
        loading = sum(self.catalog[l.model].size_gb for l in self._loads.values() if l.state == "loading")
        return sum(self.catalog[m].size_gb for m in self._models) + loading

    # ---- Engine interface ----
    def resident_models(self) -> set[str]:
        self._sync()
        return set(self._models)

    def load(self, model: str) -> str:
        self._sync()
        now = self.clock.now()
        for op in self._loads.values():
            if op.model == model and (op.state == "loading" or (op.state == "ready" and model in self._models)):
                return op.ticket
        self.calls.append((now, "load", model))
        self._seq += 1
        ticket = f"{self.name}-load-{self._seq}"
        size = self.catalog[model].size_gb
        if self.host.used_gb() + size > self.host.capacity_gb:
            self.host.oom.append((now, self.name, model))
            op = _Load(ticket, model, now, now, "oom", state="failed", error="out of memory")
        else:
            outcome = self._load_scripts[model].popleft() if self._load_scripts[model] else "ok"
            load_s = self.catalog[model].load_s * (0.5 if outcome == "fail" else 1.0)
            op = _Load(ticket, model, now, now + load_s, outcome)
        self._loads[ticket] = op
        return ticket

    def poll_load(self, ticket: str) -> LoadStatus:
        self._sync()
        op = self._loads.get(ticket)
        if op is None:
            return LoadStatus("failed", "unknown ticket")
        return LoadStatus(op.state, op.error)

    def unload(self, model: str) -> bool:
        self._sync()
        self.calls.append((self.clock.now(), "unload", model))
        live = self._live_runs(model)
        if any(r.behaviour.wedged for r in live):
            return False
        for run in live:
            run.state = "cancelled"
            run.log.append(f"[{model}] aborted: model unloaded")
        for op in self._loads.values():
            if op.model == model and op.state == "loading":
                op.state, op.error = "failed", "aborted by unload"
        self._models.discard(model)
        return True

    def start(self, spec: RunSpec) -> str:
        self._sync()
        if spec.run_id in self._by_run:
            return self._by_run[spec.run_id]
        now = self.clock.now()
        self.calls.append((now, "start", spec.run_id))
        self._seq += 1
        handle = f"{self.name}-run-{self._seq}"
        script = self._job_scripts[spec.kind]
        behaviour = script.popleft() if script else SUCCESS
        duration = float(spec.payload.get("sim_duration_s") or spec.expected_s)
        run = _Run(handle, spec, behaviour, now, duration)
        if spec.model not in self._models:
            run.state, run.error = "failed", f"model {spec.model} not loaded"
        self._runs[handle] = run
        self._by_run[spec.run_id] = handle
        return handle

    def find(self, run_id: str) -> str | None:
        self._sync()
        return self._by_run.get(run_id)

    def poll(self, handle: str) -> Poll:
        self._sync()
        run = self._runs.get(handle)
        if run is None or run.lost:
            return Poll("lost", log_tail=tuple(run.log[-5:]) if run else ())
        return Poll(run.state, run.progress, run.artifact, run.error, tuple(run.log[-5:]))

    def cancel(self, handle: str) -> None:
        self._sync()
        self.calls.append((self.clock.now(), "cancel", handle))
        run = self._runs.get(handle)
        if run and run.state == "running" and run.cancel_at is None:
            run.cancel_at = self.clock.now()
            run.log.append(f"[{run.spec.model}] cancel requested")

    def terminate(self) -> bool:
        self._sync()
        self.calls.append((self.clock.now(), "terminate", ""))
        if self.terminate_fails:
            return False
        self._restart("terminated by SIGTERM")
        return True

    # ---- internals ----
    def _live_runs(self, model: str) -> list[_Run]:
        return [r for r in self._runs.values() if r.state == "running" and not r.lost and r.spec.model == model]

    def _restart(self, reason: str) -> None:
        self.restarts += 1
        self._models.clear()
        for op in self._loads.values():
            if op.state == "loading":
                op.state, op.error = "failed", reason
        for run in self._runs.values():
            if not run.lost:
                run.lost = True
                run.log.append(f"[{self.name}] process restarted: {reason}")
        self._by_run.clear()

    def _sync(self) -> None:
        now = self.clock.now()
        for op in self._loads.values():
            if op.state == "loading" and op.outcome != "hang" and now >= op.done_at:
                if op.outcome == "fail":
                    op.state, op.error = "failed", "simulated load failure"
                else:
                    op.state = "ready"
                    self._models.add(op.model)
        for run in list(self._runs.values()):
            if run.state == "running" and not run.lost:
                self._advance(run, now)

    def _advance(self, run: _Run, now: float) -> None:
        b, model = run.behaviour, run.spec.model
        if run.cancel_at is not None and not (b.ignore_cancel or b.wedged) \
                and now >= run.cancel_at + self.CANCEL_LATENCY_S:
            run.state = "cancelled"
            run.log.append(f"[{model}] cancelled gracefully at step {run.steps}/{self.STEPS}")
            return
        frac = (now - run.started) / (run.duration * (b.factor if b.mode == "slow" else 1.0))
        if b.mode == "hang":
            frac = min(frac, b.at)
        if b.mode == "crash" and frac >= b.at:
            self._log_steps(run, b.at)
            run.state, run.error = "failed", "Metal command buffer execution failed (simulated)"
            run.log.append(f"[{model}] FATAL: {run.error}")
            if b.restart_engine:
                self._restart("engine process crashed")
            return
        self._log_steps(run, min(frac, 1.0))
        if frac >= 1.0 and b.mode != "hang":
            self._finish(run, model)

    def _log_steps(self, run: _Run, frac: float) -> None:
        target = int(frac * self.STEPS + 1e-9)
        while run.steps < target:
            run.steps += 1
            run.log.append(f"[{run.spec.model}] step {run.steps}/{self.STEPS}")
        run.progress = run.steps / self.STEPS

    def _finish(self, run: _Run, model: str) -> None:
        family = self.catalog[model].family
        if family == "brain":
            try:
                decision = self.brain(run.spec, model) if self.brain else {}
            except Exception as e:  # a buggy brain is just another failed run
                run.state, run.error = "failed", f"brain raised {e!r}"
                return
            run.artifact = {"decision": decision, "model": model}
        else:
            run.artifact = _artifact(family, run.spec, run.behaviour, self.artifacts_dir)
        run.state = "done"
        run.log.append(f"[{model}] done -> {run.artifact.get('path', 'decision')}")


# ---------------------------------------------------------------- the stand-in brain

def render_request(goal: str, *, duration_s: float = 8.0, expected_s: float = 300.0,
                   model: str = "ltx-2.5", **payload) -> dict:
    return {"kind": "render_video", "requires": model,
            "payload": {"goal": goal, "prompt": goal, "duration_s": duration_s, "expected_s": expected_s, **payload},
            "on_success": {"kind": "review_render", "needs": "vision"},
            "on_failure": {"kind": "recover_render"}}


class ScriptedBrain:
    """Stands in for the LLM. A pure function of the job payload (the resume brief): it keeps
    nothing between calls, exactly like a freshly reloaded model. `transcript` is test-side only."""

    def __init__(self, retry_failures: bool = True, chain_forever: bool = False):
        self.retry_failures, self.chain_forever = retry_failures, chain_forever
        self.transcript: list[dict] = []

    def __call__(self, spec: RunSpec, model: str) -> dict:
        brief = spec.payload.get("brief") or {}
        self.transcript.append({"job_id": spec.job_id, "kind": spec.kind, "model": model, "brief": brief})
        if spec.kind == "plan_video":
            render = render_request(spec.payload["goal"], **spec.payload.get("render", {}))
            return {"submit": [render], "note": "planned a render"}
        if spec.kind == "batch_checkpoint":
            nxt = (brief.get("context") or {}).get("next")
            return {"submit": [nxt] if nxt else [], "note": "checkpoint: continue the batch"}
        if spec.kind.startswith("recover"):
            if self.retry_failures and brief["attempt"] < brief["max_attempts"]:
                act = brief["attempted_action"]
                return {"submit": [{"kind": act["kind"], "requires": act["model"],
                                    "payload": {**act["params"], "goal": brief["goal"]},
                                    "on_success": act["on_success"], "on_failure": act["on_failure"],
                                    "retry_of": brief["source_job_id"]}],
                        "note": f"retry after {brief['outcome']['reason']}: {brief['outcome']['detail']}"}
            return {"verdict": "give_up", "note": f"not retrying ({brief['outcome']['reason']})"}
        if spec.kind.startswith("review"):
            clip = brief["artifacts"][0] if brief.get("artifacts") else None
            if self.chain_forever:
                return {"submit": [render_request(brief["goal"])], "verdict": "accept", "note": "again!"}
            return {"verdict": "accept", "note": f"clip looks right: {clip}"}
        return {"verdict": "ack"}


class ScriptedPlanner(ScriptedBrain):
    """Stands in for the planner LLM (and, for every other job kind, for the ScriptedBrain). For
    a plan job it answers the next scripted reply: a dict (sent as JSON text, like a model's
    reply), a raw string (to exercise the strict parser), a callable(snapshot) returning either,
    or an exception to raise (a crashed brain). With the script used up it calls `policy` if one
    is given, else proposes nothing. `plans` (test-side only) records what each tick saw."""

    def __init__(self, *script, policy=None, **kw):
        super().__init__(**kw)
        self.script = collections.deque(script)
        self.policy = policy
        self.plans: list[dict] = []

    def __call__(self, spec: RunSpec, model: str) -> dict:
        if spec.kind != PLAN_KIND:
            return super().__call__(spec, model)
        snap = spec.payload.get("snapshot") or {}
        self.plans.append({"job_id": spec.job_id, "model": model, "snapshot": snap,
                           "prompt": spec.payload.get("prompt") or {}})
        if self.script:
            out = self.script.popleft()
        elif self.policy is not None:
            out = self.policy
        else:
            out = {"proposals": [], "new_template_ideas": []}
        if callable(out) and not isinstance(out, BaseException):
            out = out(snap)
        if isinstance(out, BaseException):
            raise out
        raw = out if isinstance(out, str) else json.dumps(out)
        return {"verdict": "plan", "raw": raw, "model": model}


# ---------------------------------------------------------------- harness

class Sim:
    """Clock + host + two engines + a supervisor that can be killed and respawned (launchd)."""

    def __init__(self, db_path: str, capacity_gb: float = 256.0, policy: Policy | None = None,
                 brain: ScriptedBrain | None = None, planner=None, artifacts_dir: str | None = None):
        self.clock = VirtualClock()
        self.host = SimHost(capacity_gb)
        self.policy = dataclasses.replace(policy or Policy(), host_memory_gb=capacity_gb)
        self.brain = brain or ScriptedBrain()
        self.mlx = FakeEngine("mlx-serve", self.clock, self.host, brain=self.brain)
        self.omlx = FakeEngine("omlx", self.clock, self.host, brain=self.brain)
        self.engines = {"mlx-serve": self.mlx, "omlx": self.omlx}
        self.mlx.artifacts_dir = artifacts_dir
        self.planner = planner  # a foreman.planner.Planner; restart() hands the new gpud this attribute
        self.db_path = db_path
        self.pages: list[dict] = []
        self.restarts = 0
        self.crash_downtime_s = 0.0
        self.samples: list[tuple[float, bool]] = []
        self._actions: list[tuple[float, object]] = []
        self.sup = self._spawn()
        self.sup.recover()

    def _spawn(self) -> Supervisor:
        return Supervisor(self.db_path, self.engines, self.clock, self.policy, pager=self.pages.append,
                          planner=self.planner)

    @property
    def store(self):
        return self.sup.store

    def arm(self, hook) -> None:
        self.sup.crash_hook = hook

    def restart(self, downtime_s: float = 0.0) -> None:
        """Discard the supervisor object; launchd starts a new one on the same DB file."""
        self.sup.close()
        self.restarts += 1
        if downtime_s:
            self.clock.advance(downtime_s)
            self._run_actions()
        self.sup = self._spawn()
        self.sup.recover()

    def at(self, t: float, fn) -> None:
        self._actions.append((t, fn))

    def _run_actions(self) -> None:
        now = self.clock.now()
        due = [a for a in self._actions if a[0] <= now]
        self._actions = [a for a in self._actions if a[0] > now]
        for _, fn in due:
            fn()

    def step(self, dt: float = 1.0) -> None:
        self.clock.advance(dt)
        self._run_actions()
        try:
            self.sup.tick()
        except SimulatedCrash:
            self.restart(self.crash_downtime_s)
        self.samples.append((self.clock.now(), self.brain_resident()))

    def brain_resident(self) -> bool:
        return any(m in CATALOG and CATALOG[m].is_brain for e in self.engines.values() for m in e.resident_models())

    def run_until(self, pred, max_s: float = 7200.0, dt: float = 1.0) -> bool:
        end = self.clock.now() + max_s
        while self.clock.now() < end:
            self.step(dt)
            if pred():
                return True
        return False

    def quiet(self) -> bool:
        """Nothing active, no load in flight, nothing unfired; and either SAFE (queued work waits
        for a human) or an empty queue with a brain at home."""
        st = self.store
        state = st.state()
        if st.jobs(*ACTIVE) or state["loading"] or st.unfired_continuations() or st.unpaged_humans():
            return False
        if state["safe"]:
            return True
        return not st.jobs("queued") and self.brain_resident()

    def run_until_quiet(self, max_s: float = 7200.0, dt: float = 1.0) -> bool:
        return self.run_until(self.quiet, max_s, dt)

    def max_brain_away(self, after: float = 0.0) -> float:
        """Longest stretch (virtual seconds) with no brain resident, counted from `after`."""
        worst, since, prev = 0.0, after, after
        present = False
        for t, brain in self.samples:
            if t < after:
                continue
            if brain and not present and since is not None:
                worst = max(worst, t - since)
            if not brain and present:
                since = prev
            present, prev = brain, t
        if not present and self.samples:
            worst = max(worst, self.samples[-1][0] - since)
        return worst
