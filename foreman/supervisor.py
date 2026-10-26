"""gpud: the always-on supervisor.

CPU-only, holds no GPU memory, restarted by launchd if it dies, and the only component allowed
to load or unload models. It is driven by tick(); everything it needs to carry on lives in
SQLite, so a new Supervisor on the same file resumes from any committed state.

The brain-return guarantee rests on these rules:
  * a job that may run without the brain is admitted only with on_success AND on_failure
    continuations (the "baton"), and every terminal outcome fires exactly one of them;
  * every running job holds a lease that only observed progress renews; a dead lease means
    graceful cancel -> unload -> terminal outcome -> continuation;
  * queued continuations outrank everything (requests are clamped below them), and when idle
    the scheduler reloads the brain;
  * no brainless stretch is scheduled beyond MAX_BRAINLESS_S;
  * nothing an engine, a validator, a pager or the LLM does can kill the supervisor: every
    adapter call, every job and every tick phase runs inside an exception boundary, and a
    failure there becomes a recorded outcome (engine unreachable -> lost -> page; a job gpud
    cannot handle -> failed, so its on_failure still fires), never a crash loop.

Single instance: an exclusive flock on <db>.lock. A second Supervisor on the same file is a
passive standby (tick() and recover() do nothing) until it can take the lock. Every job state
transition is a compare-and-set, and leases are fenced by the leader epoch (boot_count).

Time: leases, deadlines and the brainless cap use supervisor time, a persisted monotonic
timeline (see clock.py), so wall-clock steps neither kill healthy jobs nor blind the dead-man
switch.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
from typing import Callable, Mapping

from . import validators
from .engine import Engine, LoadStatus, Poll, RunSpec
from .policy import CATALOG, ModelSpec, Policy, number
from .store import ACTIVE, TERMINAL, Job, Store, _enc

log = logging.getLogger("foreman")

# Named points where tests can kill the supervisor (see sim.CrashAt).
CRASH_POINTS = (
    "tick_start",
    "after_load_intent",
    "after_load_requested",
    "after_make_room",
    "after_starting_recorded",
    "after_engine_start",
    "after_lease_renewed",
    "after_cancel_recorded",
    "after_cancel_sent",
    "before_cleanup_unload",
    "after_cleanup_unload",
    "after_terminal_recorded",
    "inside_fire_txn",
    "after_continuation_fired",
)

BRIEF_KEYS = ("goal", "attempted_action", "outcome", "artifacts", "validator_results",
              "log_tail", "attempt")
NEEDS = ("any", "vision")
NUMERIC_PAYLOAD_KEYS = ("expected_s", "p90_s", "duration_s", "progress_interval_s")
# Outcomes caused by infrastructure rather than by the job: not charged to max_attempts
# (bounded separately by Policy.max_interruptions).
INFRA_REASONS = ("engine_unreachable",)
# Failure reasons an engine may give for a failed run (Poll.reason); anything else is a "crash".
ENGINE_FAILURE_REASONS = ("rejected",)
# The planner's brain job (foreman/planner.py). It runs only on a brain that is already resident
# (payload.resident_only), is never retried and never pages on failure: its decision is a list of
# proposals the planner validates, never requests gpud admits as they are. Only the planner's own
# tick creates one (idempotency key plan-tick:<n>); admission refuses the kind from anyone else,
# and a plan job without that key is never applied.
PLAN_KIND = "plan"
PLAN_KEY_PREFIX = "plan-tick:"


class AdmissionError(ValueError):
    """The job would weaken a brain-return guarantee (or is malformed), so it is refused."""


class EscalatingAdmissionError(AdmissionError):
    """Refused, and the lineage cannot make progress by itself: a human is paged."""

    def __init__(self, message: str, idem: str):
        super().__init__(message)
        self.idem = idem


class DepthLimitError(EscalatingAdmissionError):
    pass


class LineageClosed(AdmissionError):
    """Refused because the owner (or a kill criterion) closed the lineage: a planned item whose
    template is paused takes no new work. Deliberate, so it pages nobody."""


class LineageBudgetError(EscalatingAdmissionError):
    pass


class EngineUnavailable(Exception):
    """An adapter call raised (or returned garbage). Recorded as an engine-health observation."""

    def __init__(self, engine: str, method: str, error: str):
        super().__init__(f"{engine}.{method}: {error}")
        self.engine, self.method, self.error = engine, method, error


def failed(reason: str, detail: str, log_tail=(), **extra) -> dict:
    return {"status": "failed", "reason": reason, "detail": detail, "artifact": None,
            "validator_results": [], "log_tail": [str(x) for x in log_tail], **extra}


class InstanceLock:
    """One gpud per database: an exclusive, non-blocking flock on '<db>.lock'. The kernel drops
    it when the process dies, so a crashed gpud never leaves a stale lock behind."""

    def __init__(self, path: str):
        self.path, self.fd = path, None

    @property
    def held(self) -> bool:
        return self.fd is not None

    def acquire(self) -> bool:
        if self.fd is not None:
            return True
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:
            pass
        self.fd = fd
        return True

    def release(self) -> None:
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------- adapter result checks
# A malformed result counts as an engine error (EngineUnavailable), never as a gpud crash.

def _model_set(result) -> set[str]:
    if isinstance(result, (str, bytes)) or result is None:
        raise TypeError(f"resident_models() returned {result!r}")
    return {str(m) for m in result}


def _handle(result) -> str:
    if not isinstance(result, str) or not result:
        raise TypeError(f"expected a run handle, got {result!r}")
    return result


def _opt_handle(result) -> str | None:
    return None if result is None else _handle(result)


def _opt_instance(result) -> str | None:
    if result is None:
        return None
    if isinstance(result, bool) or not isinstance(result, (str, int, float)):
        raise TypeError(f"instance_id() returned {result!r}")
    return str(result)[:200] or None


def _opt_id(result) -> str | None:
    return result[:300] if isinstance(result, str) and result and result != "mlx-serve" else None


def _ticket(result) -> str:
    if not isinstance(result, str) or not result:
        raise TypeError(f"expected a load ticket, got {result!r}")
    return result


def _load_status(result) -> LoadStatus:
    state = getattr(result, "state", None)
    if state not in ("loading", "ready", "failed"):
        raise TypeError(f"poll_load() returned {result!r}")
    err = getattr(result, "error", None)
    return LoadStatus(state, None if err is None else str(err), getattr(result, "deferred", False) is True,
                      getattr(result, "in_flight", False) is True)


def _poll(result) -> Poll:
    status = getattr(result, "status", None)
    if not isinstance(status, str):
        raise TypeError(f"poll() returned {result!r}")
    progress = number(getattr(result, "progress", 0.0))
    tail = getattr(result, "log_tail", ()) or ()
    if isinstance(tail, (str, bytes)):
        tail = (tail,)
    err = getattr(result, "error", None)
    reason = getattr(result, "reason", None)
    return Poll(status, progress if progress is not None else 0.0, getattr(result, "artifact", None),
                None if err is None else str(err), tuple(str(x) for x in tail),
                reason if reason in ENGINE_FAILURE_REASONS else None)


def _opt_int(req: dict, key: str) -> int | None:
    value = req.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdmissionError(f"{key} must be an integer, got {value!r}")
    return value


def _jsonable(value, what: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        raise AdmissionError(f"{what} is not JSON-serializable: {e}") from None


def _content_key(kind: str, requires: str, payload: dict) -> str:
    return hashlib.sha256(_enc([kind, requires, payload]).encode()).hexdigest()[:16]


class Supervisor:
    def __init__(self, db_path: str, engines: Mapping[str, Engine], clock, policy: Policy | None = None,
                 catalog: Mapping[str, ModelSpec] | None = None,
                 pager: Callable[[dict], None] | None = None,
                 crash: Callable[[str], None] | None = None,
                 monotonic: Callable[[], float] | None = None, lead: bool = True, planner=None):
        """lead=False makes a pure client (a CLI that only submits or cancels): it never takes
        the single-instance lock, so it cannot keep the real gpud out while it runs. `planner`
        (a foreman.planner.Planner) adds the planner phases to tick()."""
        self.store = Store(db_path)
        self.store.wall_clock = clock.now
        self.engines = dict(engines)
        self.clock = clock
        self.p = policy or Policy()
        self.catalog = dict(catalog or CATALOG)
        self.pager = pager or (lambda page: log.warning("PAGE HUMAN: %s", page))
        self.crash_hook = crash or (lambda point: None)
        self.planner = planner
        self._mono = monotonic or getattr(clock, "monotonic", None)
        self._tb: tuple[float, float, float | None] | None = None  # (T, wall, mono) last read here
        self._tick_t: float | None = None
        self._resident: dict[str, str] = {}   # model -> engine, refreshed from engines every tick
        self._health: dict[str, dict] = dict(self.store.state()["engine_health"] or {})
        self._safe_engine: str | None = None
        self._sizes: dict[str, float] = {}    # adapter-reported sizes of non-catalog models
        st0 = self.store.state()
        self._owners: dict[str, dict] = self.store.owners()  # models gpud itself loaded (model_owners)
        self._instances: dict = dict(st0["instances"] or {})  # engine -> last known server instance id
        self._inst_seen: dict = {}            # engine -> this tick's reading (None: unknown right now)
        self._loading_seen: dict = {}         # engine -> models a load is in flight for (any client)
        self._peaks: dict = dict(st0["observed_gb"] or {})    # model -> largest observed size
        self._room_why: str = ""              # why the last make-room failed (for pages)
        self._ext_errors: dict = {}           # (engine, method) -> error type last noted for an optional hint
        self._noted: dict[str, object] = {}   # de-duplication of repetitive events
        self._idle_since: float | None = None  # queue empty and nothing running since (in-process)
        self._phase_errors: dict[str, int] = {}
        self._recovered = False
        self._superseded = False
        self._epoch: int | None = None
        self._lock = None if db_path == ":memory:" else InstanceLock(db_path + ".lock")
        self._client = not lead
        if self._lock is not None and lead:
            self._lock.acquire()

    @property
    def leader(self) -> bool:
        return not self._client and (self._lock is None or self._lock.held)

    def close(self) -> None:
        try:
            self.store.close()
        finally:
            if self._lock is not None:
                self._lock.release()

    # ------------------------------------------------------------------ public API

    def submit(self, kind: str, requires: str = "brain", payload: dict | None = None,
               on_success: dict | None = None, on_failure: dict | None = None, **opts) -> int:
        req = dict(opts, kind=kind, requires=requires, payload={} if payload is None else payload,
                   on_success=on_success, on_failure=on_failure)
        now = self._now()
        try:
            with self.store.tx():
                return self._admit(req, now)
        except Exception as e:
            err = e if isinstance(e, AdmissionError) else AdmissionError(f"malformed request: {type(e).__name__}: {e}")
            with self.store.tx():
                self.store.event(now, "admission_rejected", None, kind=kind if isinstance(kind, str) else None,
                                 error=str(err))
            if err is e:
                raise
            raise err from e

    def cancel_job(self, job_id: int, reason: str = "cancelled by request", by: str = "request") -> bool:
        """`by` says who cancelled: "request" (the owner, a CLI, a caller) is a deliberate decision;
        "gpud" (gpud draining or switching modes) is an infrastructure failure for the ledger."""
        now = self._now()
        job = self.store.job(job_id)
        if job is None:
            return False
        if job.state == "queued":
            self._record_terminal(job, failed("cancelled", reason, [f"supervisor: cancelled before start ({reason})"],
                                              cancelled_by=by), now)
            return self.store.job(job_id).state != "queued"
        if job.state == "running":
            # Only the leader talks to engines; a standby just records the request.
            return self._begin_cancel(job, failed("cancelled", reason, cancelled_by=by), now, lease_state="revoked",
                                      advance=self.leader and self._recovered)
        return False

    def clear_safe(self) -> None:
        """Human-only: leave SAFE mode and give every brain a fresh set of load attempts."""
        now = self._now()
        with self.store.tx():
            st = self.store.state()
            self.store.set_state(safe=0, safe_reason=None, safe_engine=None, safe_epoch=st["safe_epoch"] + 1,
                                 ladder={})
            self.store.event(now, "safe_mode_cleared", None)
        self._safe_engine = None

    def recover(self) -> None:
        """Startup reconciliation of the DB against what the engines actually report. A standby
        (another gpud holds the lock) does nothing."""
        if not self._lead():
            return
        now = self._now()
        self._tick_t = now
        self._health = dict(self.store.state()["engine_health"] or {})
        with self.store.tx():
            boot = self.store.state()["boot_count"] + 1
            self.store.set_state(boot_count=boot)
            self._persist_time()
            self.store.event(now, "supervisor_start", None, boot=boot, pid=os.getpid())
        self._epoch = boot
        self._recovered = True
        # Everything another gpud (the leader this standby replaces) may have changed since this
        # object was built: provenance, the server instances it saw, the sizes it observed.
        st = self.store.state()
        self._owners = self.store.owners()
        self._instances = dict(st["instances"] or {})
        self._peaks = dict(st["observed_gb"] or {})
        self._sizes, self._inst_seen, self._loading_seen = {}, {}, {}
        self._phase("owners", self._seed_owners, now)
        self._phase("adopt", self._adopt_server_ids, now)
        self._phase("observe", lambda t: self._observe(t, fresh=True), now)
        for job in self.store.jobs("starting", "running"):
            self._guard_job(job, lambda j=job: self._reconcile(j, now, boot=True), now)
        # 'cancelling' jobs resume from their persisted cancel_stage; unfired continuations
        # and in-flight loads are picked up by the normal tick steps.

    def tick(self, now: float | None = None) -> None:
        if not self._lead():
            return
        if not self._recovered:
            self.recover()
        if not self._still_leader():
            return
        now = self._now() if now is None else now
        self._tick_t = now
        with self.store.tx():
            self._persist_time()
        self.crash_hook("tick_start")
        self._phase("observe", lambda t: self._observe(t, fresh=True), now)
        self._phase("loads", self._check_load, now)
        self._phase("running", self._check_running, now)
        self._phase("engines", self._check_engines, now)
        self._phase("fire", self._fire_continuations, now)
        self._phase("orphans", self._clean_orphans, now)
        if self.planner is not None:
            self._phase("planner_upkeep", lambda t: self.planner.upkeep(self, t), now)
            self._phase("planner", lambda t: self.planner.maybe_plan(self, t), now)
        self._phase("schedule", self._schedule, now)
        self._phase("page", self._page_humans, now)
        self._phase("observe", self._observe, now)

    # ------------------------------------------------------------------ leadership, time, boundaries

    def _lead(self) -> bool:
        if self._superseded or self._client:
            return False
        if self._lock is None or self._lock.held:
            return True
        if self._lock.acquire():
            self._recovered = False  # a standby that takes over must reconcile first
            return True
        return False

    def _still_leader(self) -> bool:
        """Leader-epoch fencing: if another instance recovered after us (the lock was broken or
        bypassed), stop acting on this DB."""
        if self._epoch is None or self.store.state()["boot_count"] == self._epoch:
            return True
        log.error("gpud superseded by a newer leader epoch; stepping down")
        if self._lock is not None:
            self._lock.release()  # later ticks may re-acquire it only if the other leader is gone
        else:
            self._superseded = True  # no lock to arbitrate: never take the DB back
        self._recovered = False
        return False

    def _now(self) -> float:
        """Supervisor time: monotonic within this process, re-baselined from the DB across
        restarts. Wall-clock steps do not move it (see clock.py)."""
        wall = float(self.clock.now())
        mono = None
        if self._mono is not None:
            try:
                mono = float(self._mono())
            except Exception:
                mono = None
        step = None
        if self._tb is None:
            st = self.store.state()
            if st["t_now"] is None:
                t = wall
            elif mono is not None and st["t_mono"] is not None and mono >= st["t_mono"]:
                t = st["t_now"] + (mono - st["t_mono"])
            else:
                last_wall = st["t_wall"] if st["t_wall"] is not None else wall
                t = st["t_now"] + max(0.0, wall - last_wall)
        else:
            t0, w0, m0 = self._tb
            if mono is not None and m0 is not None:
                t = t0 + max(0.0, mono - m0)
            else:
                d = wall - w0
                if d < 0 or d > self.p.max_clock_step_s:
                    step, d = d, 0.0
                t = t0 + d
        self._tb = (t, wall, mono)
        if step is not None:
            with self.store.tx():
                self.store.event(t, "clock_step", None, wall_jump_s=step,
                                 note="wall clock stepped; supervisor time did not")
        return t

    def _persist_time(self) -> None:
        if self._tb is not None:
            t, wall, mono = self._tb
            self.store.set_state(t_now=t, t_wall=wall, t_mono=mono)

    def _t(self) -> float:
        return self._tick_t if self._tick_t is not None else self._now()

    def _phase(self, name: str, fn, now: float) -> None:
        """Tick-phase boundary: an exception is logged and recorded, never fatal."""
        try:
            fn(now)
            self._phase_errors.pop(name, None)
        except EngineUnavailable:
            pass  # already recorded as an engine-health observation
        except Exception as e:
            n = self._phase_errors.get(name, 0) + 1
            self._phase_errors[name] = n
            if n == 1 or n % 100 == 0:  # a phase failing every tick must not flood the log either
                log.exception("gpud phase %s failed (%d in a row)", name, n)
                try:
                    with self.store.tx():
                        self.store.event(now, "supervisor_error", None, phase=name, count=n,
                                         error=f"{type(e).__name__}: {e}")
                except Exception:
                    log.exception("could not record supervisor_error")
            if name == "loads" and n >= self.p.max_job_faults:
                self._abandon_load(now, f"gpud could not track the load: {type(e).__name__}: {e}")

    def _guard_job(self, job: Job, fn, now: float, default=None):
        """Job boundary: an engine problem is left to engine-health tracking; any other
        exception is a fault charged to this job, and enough faults fail it."""
        try:
            return fn()
        except EngineUnavailable:
            return default
        except Exception as e:
            log.exception("gpud fault on job %s", job.id)
            self._job_fault(job.id, e, now)
            return default

    def _job_fault(self, job_id: int, error: Exception, now: float) -> None:
        try:
            cur = self.store.job(job_id)
            if cur is None:
                return
            n = (cur.faults or 0) + 1
            with self.store.tx():
                self.store.update_job(job_id, faults=n)
                self.store.event(now, "job_fault", job_id, faults=n, state=cur.state,
                                 error=f"{type(error).__name__}: {error}")
            if n < self.p.max_job_faults:
                return
            detail = f"gpud raised {type(error).__name__} {n} times while handling this job: {error}"
            tail = [f"supervisor: {detail}"]
            if cur.state in ("queued", "starting"):
                self._record_terminal(cur, failed("supervisor_error", detail, tail), now)
            elif cur.state == "running":
                self._begin_cancel(cur, failed("supervisor_error", detail, tail), now, lease_state="revoked")
            elif cur.state == "cancelling":
                outcome = dict(cur.pending_outcome or failed("supervisor_error", detail, tail),
                               cleanup=f"skipped ({detail})")
                self._record_terminal(cur, outcome, now)
            elif cur.state in TERMINAL:
                self._fire_fallback(cur, detail, now)
        except EngineUnavailable:
            pass
        except Exception as e2:
            log.exception("escalating job %s failed", job_id)
            self._force_terminal(job_id, f"{type(error).__name__}: {error}; escalation failed: {e2!r}", now)

    def _force_terminal(self, job_id: int, detail: str, now: float) -> None:
        """Last resort when even recording an outcome raises: quarantine with a minimal outcome
        so the job leaves the active set and a human (and, for brainless jobs, the baton) hears."""
        try:
            cur = self.store.job(job_id)
            if cur is None or cur.state in TERMINAL or cur.state == "awaiting_human":
                return
            outcome = dict(failed("supervisor_error", detail, [f"supervisor: {detail}"]), model=cur.model)
            with self.store.tx():
                if not self.store.update_job(cur.id, expect_state=cur.state, state="quarantined", outcome=outcome,
                                             finished_at=now, pending_outcome=None, cancel_stage=None):
                    return
                self.store.update_lease(cur.id, state="released")
                self.store.event(now, "job_finished", cur.id, kind=cur.kind, state="quarantined",
                                 reason="supervisor_error", detail=detail, model=cur.model, attempt=cur.attempt,
                                 forced=True)
                if not cur.brainless:
                    self._human_review(cur, detail, now, idem=f"review:{cur.id}")
        except Exception:
            log.exception("could not force job %s terminal", job_id)

    def _note(self, now: float, kind: str, key, job_id: int | None = None, **data) -> None:
        """Record a repetitive condition once per distinct key (insufficient memory, ...)."""
        if self._noted.get(kind) == key:
            return
        self._noted[kind] = key
        with self.store.tx():
            self.store.event(now, kind, job_id, **data)

    # ------------------------------------------------------------------ engine calls & health

    def _call(self, name: str | None, method: str, *args, check=None):
        if not name:  # a model with no known engine: nothing to call, nothing to mark unhealthy
            raise EngineUnavailable("?", method, "no engine is known for this model")
        eng = self.engines.get(name)
        if eng is None:  # jobs recorded on an engine that is no longer configured end up 'lost'
            self._engine_error(name, method, "engine is not configured")
            raise EngineUnavailable(name, method, "engine is not configured")
        try:
            result = getattr(eng, method)(*args)
            if check is not None:
                result = check(result)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if getattr(e, "pending", False) is True:
                # The engine answered: the operation is still in progress (an unload draining
                # behind a generation that stops at its next step boundary). Not an outage, so
                # the engine stays healthy; the caller retries next tick, and the adapter bounds
                # the wait (it returns False, and gpud escalates, if it never finishes).
                self._note(self._t(), "engine_busy", (name, method, repr(args)[:200]),
                           engine=name, method=method, detail=err[:300])
                raise EngineUnavailable(name, method, err) from e
            self._engine_error(name, method, err)
            raise EngineUnavailable(name, method, err) from e
        busy = self._noted.get("engine_busy")
        if busy is not None and busy[:2] == (name, method):
            self._noted.pop("engine_busy", None)  # finished: a later wait is noted again
        self._engine_ok(name)
        return result

    def _optional(self, name: str | None, method: str, *args, check=None):
        """An optional engine extension (instance_id, model_size_gb, loading_models, owned_id,
        set_owned_id): None when the adapter has none. These are hints, so an exception or a
        malformed answer is "unknown" (noted once per engine, method and error type), never an
        engine outage: the core calls (resident_models, ...) say whether the engine answers."""
        fn = getattr(self.engines.get(name), method, None) if name else None
        if not callable(fn):
            return None
        try:
            result = fn(*args)
            result = check(result) if check is not None else result
        except Exception as e:
            if self._ext_errors.get((name, method)) != type(e).__name__:
                self._ext_errors[(name, method)] = type(e).__name__
                with self.store.tx():
                    self.store.event(self._t(), "engine_extension_error", None, engine=name, method=method,
                                     error=f"{type(e).__name__}: {e}"[:300],
                                     note="an optional adapter hint failed: treated as unknown")
            return None
        self._ext_errors.pop((name, method), None)
        return result

    def _engine_error(self, name: str, method: str, err: str) -> None:
        h = self._health.get(name)
        if h is not None and h.get("down_since") is not None:
            h["errors"] = h.get("errors", 0) + 1
            h["last_error"] = f"{method}: {err}"[:300]
            return
        now = self._t()
        outage = (h or {}).get("outage", 0) + 1
        self._health[name] = {"down_since": now, "errors": 1, "last_error": f"{method}: {err}"[:300],
                              "lost": False, "outage": outage}
        with self.store.tx():
            self.store.set_state(engine_health=self._health)
            self.store.event(now, "engine_error", None, engine=name, method=method, error=err[:300], outage=outage)

    def _engine_ok(self, name: str) -> None:
        h = self._health.get(name)
        if h is None or h.get("down_since") is None:
            return
        now = self._t()
        down_s, was_lost = now - h["down_since"], bool(h.get("lost"))
        h.update(down_since=None, lost=False)
        with self.store.tx():
            self.store.set_state(engine_health=self._health)
            self.store.event(now, "engine_recovered", None, engine=name, down_s=down_s, was_lost=was_lost,
                             errors=h.get("errors", 0))

    def _avail(self, name: str | None) -> bool:
        """Configured, answering, and not the engine that put gpud in (degraded) SAFE mode."""
        if not name or name not in self.engines or name == self._safe_engine:
            return False
        return (self._health.get(name) or {}).get("down_since") is None

    def _check_engines(self, now: float) -> None:
        for name, h in list(self._health.items()):
            if h.get("down_since") is not None and not h.get("lost") and now - h["down_since"] >= self.p.engine_lost_s:
                self._engine_lost(name, now)

    def _engine_lost(self, name: str, now: float) -> None:
        """An engine failed every call for engine_lost_s: fail what depends on it (so every
        baton fires), remember its runs for cleanup if it comes back, and page a human."""
        h = self._health[name]
        h["lost"] = True
        detail = f"engine {name} unreachable for {now - h['down_since']:.0f}s (last error: {h.get('last_error')})"
        tail = [f"supervisor: {detail}"]
        with self.store.tx():
            self.store.set_state(engine_health=self._health)
            self.store.event(now, "engine_declared_lost", None, engine=name, detail=detail)
            self._human_review(None, f"{detail}; jobs on it were failed and the brain cannot return through it",
                               now, idem=f"engine:{name}:{h.get('outage', 0)}", kind="engine_lost")
            load = self.store.state()["loading"]
            if load and load.get("engine") == name:
                self.store.set_state(loading=None)
                self.store.event(now, "load_abandoned", load.get("for_job"), model=load.get("model"), reason=detail)
        orphans = list(self.store.state()["orphans"] or [])
        for job in self.store.jobs(*ACTIVE):
            if job.engine != name:
                continue
            outcome = dict(job.pending_outcome) if job.state == "cancelling" and job.pending_outcome \
                else failed("engine_unreachable", detail, tail)
            outcome["cleanup"] = "engine_unreachable"
            self._guard_job(job, lambda j=job, o=outcome: self._record_terminal(j, o, now), now)
            if job.brainless and job.model:
                orphans.append({"engine": name, "model": job.model, "job_id": job.id})
        for job in self.store.jobs("queued"):
            if job.brainless and self._engine_name(job.requires) == name:
                self._guard_job(job, lambda j=job: self._record_terminal(j, failed(
                    "engine_unreachable", f"{j.requires} runs on {name}: {detail}", tail), now), now)
        with self.store.tx():
            self.store.set_state(orphans=orphans)

    def _clean_orphans(self, now: float) -> None:
        """Runs gpud gave up on while their engine was unreachable: once it answers again,
        unload their (media) model if nothing live uses it."""
        st = self.store.state()
        orphans = st["orphans"] or []
        if not orphans or (st["safe"] and not st["safe_engine"]):
            return
        live = {j.model for j in self.store.jobs(*ACTIVE)} | {(st["loading"] or {}).get("model")}
        keep = []
        for o in orphans:
            name, model = o.get("engine"), o.get("model")
            if name not in self.engines or model in live:
                continue
            if not self._avail(name):
                keep.append(o)
                continue
            if self._resident.get(model) == name:
                try:
                    self._unload(model, now, reason=f"orphaned run of job {o.get('job_id')}")
                except EngineUnavailable:
                    keep.append(o)
        if keep != orphans:
            with self.store.tx():
                self.store.set_state(orphans=keep)

    # ------------------------------------------------------------------ admission

    def _admit(self, req, now: float, parent: Job | None = None, idem: str | None = None) -> int:
        """Validate and insert one request. Everything the LLM can write is untrusted: anything
        malformed is an AdmissionError, and caller-chosen priority / max_attempts are clamped."""
        p = self.p
        if not isinstance(req, dict):
            raise AdmissionError(f"a request must be an object, got {type(req).__name__}")
        kind = req.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise AdmissionError("job kind is required (a non-empty string)")
        if kind.strip().lower() == PLAN_KIND:
            raise AdmissionError("plan jobs come only from the planner's own ticks; no request or brain "
                                 "decision may submit one")
        requires = req.get("requires")
        requires = "brain" if requires is None else requires
        if not isinstance(requires, str) or not requires:
            raise AdmissionError(f"requires must be a model name or 'brain', got {requires!r}")
        payload = req.get("payload")
        payload = {} if payload is None else payload
        if not isinstance(payload, dict):
            raise AdmissionError(f"payload must be an object, got {type(payload).__name__}")
        payload = dict(payload)
        on_s, on_f = req.get("on_success"), req.get("on_failure")
        if requires != "brain":
            spec = self.catalog.get(requires)
            if spec is None or spec.is_brain:
                raise AdmissionError(f"unknown requirement {requires!r}")
            if spec.engine not in self.engines:
                raise AdmissionError(f"{requires} runs on {spec.engine}, which gpud has no adapter for")
            if spec.size_gb > p.usable_gb:
                raise AdmissionError(f"{requires} needs {spec.size_gb} GB, host has {p.usable_gb} GB usable")
        self._check_payload(requires, payload)
        if parent is not None and self.planner is not None:
            # A planned item's lineage may only retry its own job, and nothing once its template
            # is paused (foreman.planner.Planner.check_child).
            self.planner.check_child(self, parent.root_id or parent.id, kind, requires, payload)
        req_max = _opt_int(req, "max_attempts")
        if req_max is not None and req_max < 1:
            raise AdmissionError("max_attempts must be at least 1")
        max_attempts = min(req_max or p.max_attempts, p.max_attempts)
        req_priority = _opt_int(req, "priority") or 0
        priority = max(p.min_priority, min(req_priority, p.continuation_priority - 1))
        retry_of = _opt_int(req, "retry_of")
        client_key = req.get("idempotency_key")
        if client_key is not None and (not isinstance(client_key, str) or not client_key):
            raise AdmissionError("idempotency_key must be a non-empty string")
        depth = parent.depth + 1 if parent else 0
        root = (parent.root_id or parent.id) if parent else None
        split = None
        if requires != "brain":
            if not (self._cont_spec(on_s) and self._cont_spec(on_f)):
                raise AdmissionError("a job that may run while the brain is unloaded must carry both "
                                     "on_success and on_failure continuations")
            payload, on_s, on_f, split = self._fit_brainless_cap(kind, requires, payload, on_s, on_f)
        content_key = _content_key(kind, requires, payload)
        attempt, task_key, inferred = 1, None, None
        if retry_of is not None:
            src = self.store.job(retry_of)
            if src is None:
                raise AdmissionError(f"retry_of {retry_of} does not exist")
            task_key, max_attempts = src.task_key, src.max_attempts
            attempt = self.store.max_attempt(task_key) + 1
        elif parent is not None:
            # The LLM may resubmit a failing request without retry_of. Within one lineage the
            # same (kind, requires, payload) after a non-success is the same logical task.
            prev = self.store.latest_in_lineage(root, content_key)
            if prev is not None and prev.state != "succeeded" and prev.task_key:
                task_key, max_attempts, inferred = prev.task_key, prev.max_attempts, prev.id
                attempt = self.store.max_attempt(task_key) + 1
        if task_key is not None and attempt > max_attempts:
            raise AdmissionError(f"task {task_key} already used its {max_attempts} attempts")
        if depth > p.max_continuation_depth:
            raise DepthLimitError(f"continuation depth {depth} exceeds {p.max_continuation_depth}",
                                  idem=f"depth:{parent.id}")
        if requires != "brain" and root is not None:
            fails = self.store.lineage_failures(root)
            if fails >= p.max_lineage_failures:
                raise LineageBudgetError(f"lineage {root} already has {fails} failed brainless jobs "
                                         f"(limit {p.max_lineage_failures})", idem=f"lineage:{root}")
        key = idem or (f"client:{client_key}" if client_key is not None else None)
        job_id, created = self._insert_job(
            kind=kind, requires=requires, state="queued", payload=payload,
            parent_id=parent.id if parent else None, root_id=root, depth=depth, task_key=task_key,
            content_key=content_key, attempt=attempt, max_attempts=max_attempts, priority=priority,
            idempotency_key=key, created_at=now)
        if created:
            if requires != "brain":
                self.store.add_continuation(job_id, on_s, on_f)
            self.store.event(now, "admitted", job_id, kind=kind, requires=requires, attempt=attempt, depth=depth)
            if split:
                self.store.event(now, "batch_split", job_id, **split)
            if priority != req_priority or (req_max is not None and req_max > max_attempts and retry_of is None
                                            and inferred is None):
                self.store.event(now, "request_clamped", job_id, priority=[req_priority, priority],
                                 max_attempts=[req_max, max_attempts])
            if inferred is not None:
                self.store.event(now, "retry_inferred", job_id, of=inferred, task_key=task_key, attempt=attempt)
        return job_id

    def _check_payload(self, requires: str, payload: dict) -> None:
        for key in NUMERIC_PAYLOAD_KEYS:
            if payload.get(key) is not None:
                value = number(payload[key])
                if value is None or value <= 0:
                    raise AdmissionError(f"payload.{key} must be a positive number, got {payload[key]!r}")
        needs = payload.get("needs")
        if needs is not None and needs not in NEEDS:
            raise AdmissionError(f"payload.needs must be one of {NEEDS}, got {needs!r}")
        if requires != "brain":
            goal = payload.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                raise AdmissionError("payload.goal is required: it seeds the resume brief")
            if payload.get("expected_s") is None:
                raise AdmissionError("payload.expected_s is required: it sizes the lease")
            if payload.get("items") is not None and not isinstance(payload["items"], list):
                raise AdmissionError("payload.items must be a list")
        _jsonable(payload, "payload")

    @staticmethod
    def _cont_spec(spec) -> bool:
        if not (isinstance(spec, dict) and isinstance(spec.get("kind"), str) and spec["kind"].strip()):
            return False
        if spec.get("needs", "any") not in NEEDS:
            raise AdmissionError(f"continuation needs must be one of {NEEDS}, got {spec.get('needs')!r}")
        if spec.get("context") is not None and not isinstance(spec["context"], dict):
            raise AdmissionError("continuation context must be an object")
        _jsonable(spec, "continuation")
        return True

    def _brainless_budget(self, model: str) -> float:
        """Longest hard timeout a brainless job may have so that unload + load + run + cleanup +
        brain reload still fits inside MAX_BRAINLESS_S."""
        p = self.p
        return (p.max_brainless_s - self.catalog[model].load_s - p.cancel_grace_s
                - self._brain_return_s() - p.tick_slack_s)

    def _brain_return_s(self) -> float:
        return max(self.catalog[m].load_s for m in (self.p.text_ladder[0], self.p.vision_brain))

    def _fit_brainless_cap(self, kind, requires, payload, on_s, on_f):
        hard, budget = self.p.hard_timeout_s(requires, payload), self._brainless_budget(requires)
        if hard <= budget:
            return payload, on_s, on_f, None
        items = payload.get("items")
        if not isinstance(items, list) or len(items) < 2:
            raise AdmissionError(f"worst case {hard:.0f}s exceeds the brainless budget {budget:.0f}s "
                                 f"(MAX_BRAINLESS_S) and the job has no items to split")
        k = int(budget // (hard / len(items)))
        if k < 1:
            raise AdmissionError("a single batch item exceeds the brainless budget")

        def part(lo: int, hi: int) -> dict:
            share = (hi - lo) / len(items)
            out = {**payload, "items": items[lo:hi], "expected_s": payload["expected_s"] * share}
            if payload.get("p90_s"):
                out["p90_s"] = payload["p90_s"] * share
            return out

        # The head chunk runs now; the rest rides in a brain checkpoint continuation, so the
        # brain comes back between chunks and resubmits the remainder (which may split again).
        rest = {"kind": kind, "requires": requires, "payload": part(k, len(items)),
                "on_success": on_s, "on_failure": on_f}
        head_on_s = {"kind": "batch_checkpoint", "needs": "any", "context": {"next": rest}}
        head_on_f = {**on_f, "context": {**(on_f.get("context") or {}), "batch_remaining": rest}}
        info = {"items": len(items), "chunk": k, "hard_s": hard, "budget_s": budget}
        return part(0, k), head_on_s, head_on_f, info

    def _insert_job(self, **cols) -> tuple[int, bool]:
        cols.setdefault("is_continuation", 0)
        job_id, created = self.store.insert_job(**cols)
        if created:
            fill = {}
            if not cols.get("task_key"):
                fill["task_key"] = f"task-{job_id}"
            if not cols.get("root_id"):
                fill["root_id"] = job_id
            if fill:
                self.store.update_job(job_id, **fill)
        return job_id, created

    # ------------------------------------------------------------------ observation

    def _engine_name(self, model: str | None) -> str | None:
        if model in self._resident:
            return self._resident[model]
        spec = self.catalog.get(model)
        return spec.engine if spec else None

    def _size(self, model: str) -> float:
        """Planning size: the catalog's, or the largest size the server reported for the model
        since it started if that is bigger (a model bigger than expected is never under-planned)."""
        spec = self.catalog.get(model)
        peak = self._peaks.get(model)
        seen = self._plausible_gb(peak.get("gb")) if isinstance(peak, dict) else None
        if spec:
            return max(spec.size_gb, seen) if seen else spec.size_gb
        if seen:
            self._sizes[model] = max(self._sizes.get(model, 0.0), seen)
        if model not in self._sizes:
            size = self._plausible_gb(self._optional(self._resident.get(model), "model_size_gb", model))
            if size:
                self._sizes[model] = size
        # unknown footprint: assume it fills the host, so it is unloaded before planning anything
        return self._sizes.get(model, self.p.usable_gb)

    def _plausible_gb(self, value) -> float | None:
        """A size reading gpud may plan with: a finite number above zero that fits in the host's
        memory. Anything else (junk, a glitch reporting more memory than the machine has) is no
        reading at all: it must not wedge every job of the model on memory it cannot exist in."""
        gb = number(value)
        return gb if gb is not None and 0 < gb <= self.p.host_memory_gb else None

    def _is_brain(self, model: str) -> bool:
        spec = self.catalog.get(model)
        return bool(spec and spec.is_brain)

    def _family(self, model: str | None) -> str:
        spec = self.catalog.get(model)
        return spec.family if spec else "unknown"

    def _load_timeout(self, model: str) -> float:
        return self.p.load_timeout_s(self.catalog[model])

    def _usable_brains(self) -> list[str]:
        """Resident brains gpud may send brain jobs to: its own, and (Policy.use_foreign_brains)
        someone else's, read-only: used, never unloaded."""
        return [m for m, e in self._resident.items() if self._is_brain(m) and self._avail(e)
                and (self.p.use_foreign_brains or self._ours(m))]

    # ------------------------------------------------------------------ provenance

    def _ours(self, model: str) -> bool:
        """gpud loaded this model itself (or its load request is in flight). Everything else that
        is resident is someone else's, even a catalog model: never unloaded by gpud."""
        if model in self.p.foreign_models:
            return False
        row = self._owners.get(model)
        if row is None:
            return False
        engine = self._resident.get(model)
        return engine is None or row["engine"] == engine

    def _may_unload(self, model: str) -> bool:
        if model in self.catalog:
            return self._ours(model)
        return self.p.unload_foreign  # a model gpud does not know at all (the simulation's default)

    def _set_owner(self, model: str, **cols) -> None:
        row = dict(self._owners.get(model) or {}, **cols)
        row.setdefault("since", self._t())
        self.store.set_owner(model, **{k: v for k, v in row.items() if k != "model"})
        self._owners[model] = dict(row, model=model)

    def _drop_owner(self, model: str, now: float, reason: str | None = None, event: bool = True) -> None:
        row = self._owners.pop(model, None)
        self.store.drop_owner(model)
        if row is not None and row.get("state") == "owned":
            # The adapter forgets which copy was gpud's: whatever it reports under this name from
            # now on is someone else's (it never stands in for gpud's copy within one reading).
            self._optional(row.get("engine"), "set_owned_id", model, None)
        if row is not None and reason and event:
            self.store.event(now, "ownership_dropped", row.get("job_id"), model=model, engine=row.get("engine"),
                             state=row.get("state"), reason=reason)

    def _adopt_server_ids(self, now: float) -> None:
        """A restarted gpud tells the adapter which copy of each owned model is gpud's (the
        server id recorded when gpud loaded it), so a same-named copy someone else loaded is never
        taken for it (the adapter reports such a copy under its raw server id)."""
        for m, row in sorted(self._owners.items()):
            if row.get("state") == "owned" and row.get("server_id"):
                self._optional(row["engine"], "set_owned_id", m, row["server_id"])

    def _intent_ttl(self, model: str) -> float:
        return (self._load_timeout(model) if model in self.catalog else 0.0) + 60.0

    def _seed_owners(self, now: float) -> None:
        """Once, on a database from before provenance was recorded: a model gpud last loaded (its
        latest model_loaded event, not followed by an unload, eviction or engine restart) is
        gpud's; every other resident model is someone else's."""
        st = self.store.state()
        if st["owners_init"]:
            return
        known = self._known(st)
        last: dict[str, str] = {}
        if known:
            rows = self.store.conn.execute(
                "SELECT kind, data FROM events WHERE kind IN ('model_loaded', 'model_unloaded', 'model_evicted', "
                "'model_appeared', 'engine_restarted') ORDER BY id")
            for r in rows:
                try:
                    data = json.loads(r["data"])
                except ValueError:
                    continue
                if r["kind"] == "engine_restarted":
                    for m in data.get("dropped") or []:
                        last[str(m)] = "engine_restarted"
                elif isinstance(data.get("model"), str):
                    last[data["model"]] = r["kind"]
        owned = sorted(m for m, e in known.items() if m in self.catalog and m not in self.p.foreign_models
                       and last.get(m) == "model_loaded")
        load = st["loading"] or {}
        with self.store.tx():
            for m in owned:
                self._set_owner(m, engine=known[m] or self.catalog[m].engine, state="owned", instance=None,
                                boot=None, since=now, owned_at=now, job_id=None, note="seeded from the event log")
            m = load.get("model")
            if m in self.catalog and m not in owned and m not in self.p.foreign_models:
                # A load the previous gpud had in flight: claimed if it appears (_reconcile_owners).
                self._set_owner(m, engine=load.get("engine") or self.catalog[m].engine, state="loading",
                                instance=None, boot=None, since=now, owned_at=None, job_id=load.get("for_job"),
                                note="seeded: a load in flight")
            self.store.set_state(owners_init=1)
            if known:
                self.store.event(now, "ownership_seeded", None, owned=owned,
                                 foreign=sorted(m for m in known if m not in owned))

    def _check_instance(self, name: str, now: float) -> None:
        """A changed server instance (mlx-serve: a new process start time) means the server
        restarted: what gpud loaded on it is gone, so its ownership records are dropped, and so
        are the sizes observed on the old instance. A reading of None means "unknown right now"
        (no /metrics yet, or none at all): nothing changes, but records made meanwhile carry no
        instance, and an unexpected disappearance then counts as a possible restart
        (_reconcile_owners). Records without an instance adopt the next known one."""
        if not callable(getattr(self.engines.get(name), "instance_id", None)):
            self._inst_seen[name] = None
            return
        inst = self._optional(name, "instance_id", check=_opt_instance)
        self._inst_seen[name] = inst
        old = self._instances.get(name)
        if inst is None or inst == old:
            return
        self._instances[name] = inst
        with self.store.tx():
            self.store.set_state(instances=self._instances)
            load = self.store.state()["loading"] or {}
            # A load gpud sent to the new server (after gpud itself restarted it), or while the
            # identity was unknown, is still tracked by _check_load; one sent to the old server died
            # with it (_check_load fails it).
            tracked = load.get("model") if load.get("engine") == name and load.get("instance") in (None, inst) \
                else None
            dropped = []
            for m, row in sorted(self._owners.items()):
                if row["engine"] != name or m == tracked or row.get("instance") == inst:
                    continue
                if row.get("instance") is None:
                    # Recorded while the identity was unknown, and seen resident since: this server's.
                    self._set_owner(m, instance=inst)
                    continue
                dropped.append(m)
                self._drop_owner(m, now, "recorded on another server instance", event=old is None)
            if old is None:
                return  # first reading: nothing restarted that gpud could know of
            stale = [m for m, p in self._peaks.items() if isinstance(p, dict) and p.get("engine") == name]
            for m in stale:
                self._peaks.pop(m, None)
            self._sizes.clear()  # sizes of non-catalog models: read again from the new server
            self.store.set_state(observed_gb=self._peaks)
            orphans = [o for o in (self.store.state()["orphans"] or []) if o.get("engine") != name]
            self.store.set_state(orphans=orphans)
            self.store.event(now, "engine_instance_changed", None, engine=name, old=old, new=inst,
                             dropped_ownership=dropped,
                             note="the server restarted: nothing resident on it now was loaded by gpud")

    def _observe_sizes(self, name: str, models, now: float) -> None:
        """Track the largest resident size the server reports per model on this instance
        (mlx-serve: /api/ps size_vram), for planning (see _size). A reading larger than the host's
        memory is a glitch: logged once, never planned with."""
        if not callable(getattr(self.engines.get(name), "model_size_gb", None)):
            return
        inst = self._instances.get(name)
        changed, over, odd = [], [], []
        for m in sorted(models):
            raw = number(self._optional(name, "model_size_gb", m))
            if raw is None or raw <= 0:
                continue
            gb = self._plausible_gb(raw)
            if gb is None:
                odd.append((m, raw))
                continue
            gb = math.ceil(gb * 10.0 - 1e-6) / 10.0
            cur = self._peaks.get(m) if isinstance(self._peaks.get(m), dict) else None
            same = cur is not None and cur.get("engine") == name and cur.get("instance") == inst
            if same and (number(cur.get("gb")) or 0.0) >= gb:
                continue
            entry = {"gb": gb, "engine": name, "instance": inst}
            spec = self.catalog.get(m)
            logged = number(cur.get("logged")) if same else None
            if spec and gb > spec.size_gb and (logged is None or gb >= logged + 1.0):
                entry["logged"] = gb
                over.append((m, gb, spec.size_gb))
            elif same and logged is not None:
                entry["logged"] = logged
            self._peaks[m] = entry
            changed.append(m)
        for m, raw in odd:
            self._note(now, "model_size_implausible", (name, m), model=m, engine=name, gb=raw,
                       host_gb=self.p.host_memory_gb, note="more than the host's memory: ignored, not planned with")
        if changed:
            with self.store.tx():
                self.store.set_state(observed_gb=self._peaks)
                for m, gb, plan in over:
                    self.store.event(now, "model_size_observed", None, model=m, engine=name, gb=gb, plan_gb=plan,
                                     note="larger than the catalog size: planned with the observed size")

    def _reconcile_owners(self, now: float, st: dict, current: dict, answered: set) -> None:
        """Ownership follows what the engines report: an owned model gpud sees gone is no longer
        gpud's (if it comes back without a gpud load it is someone else's). A load intent whose
        load gpud no longer tracks (it died with the request in flight, or gpud gave up on a
        request the server may still finish) is claimed if the model appears on the same server
        instance within the load's timeout, else it expires.

        While an engine's identity is unknown (no instance_id, or it reads None), a server restart
        is invisible: another client could reload one of gpud's models before gpud looks again,
        and gpud would take that copy for its own. So when one of gpud's models disappears there
        without gpud unloading it (an eviction, or a restart gpud cannot tell from one), gpud
        stops calling its other models on that engine its own: it never unloads them (the safe
        side; a human is paged if that leaves no room for a brain)."""
        loading = (st.get("loading") or {}).get("model")
        todo, vanished = [], set()
        for m, row in sorted(self._owners.items()):
            eng = row["engine"]
            if eng not in answered:
                continue  # unobservable: keep everything as it was
            if m == loading and row["state"] == "loading":
                continue  # the load is still being tracked (_check_load settles it)
            inst = self._instances.get(eng)
            if row.get("instance") is not None and inst is not None and row["instance"] != inst:
                todo.append(("drop", m, "the server restarted"))
            elif m in self.p.foreign_models:
                todo.append(("drop", m, "listed in foreign_models"))
            elif row["state"] == "owned":
                if current.get(m) != eng:
                    todo.append(("drop", m, "no longer resident"))
                    if self._inst_seen.get(eng) is None:
                        vanished.add(eng)
            elif current.get(m) == eng:
                todo.append(("claim", m, "a load gpud requested completed after gpud lost track of it"))
            elif now - (number(row.get("since")) or now) > self._intent_ttl(m):
                todo.append(("drop", m, "load intent expired: the model never appeared"))
        going = {m for action, m, _ in todo if action == "drop"}
        for m, row in sorted(self._owners.items()):
            if row["engine"] in vanished and row["state"] == "owned" and m not in going:
                todo.append(("drop", m, "one of gpud's models disappeared while the server's identity is unknown: "
                                        "a restart (and someone else's reload of this one) cannot be ruled out"))
        if not todo:
            return
        with self.store.tx():
            for action, m, why in todo:
                if action == "drop":
                    self._drop_owner(m, now, why)
                else:
                    row = self._owners[m]
                    sid = self._optional(row["engine"], "server_id", m, check=_opt_id)
                    self._set_owner(m, state="owned", owned_at=now, instance=self._inst_seen.get(row["engine"]),
                                    server_id=sid, note=why)
                    if sid:
                        self._optional(row["engine"], "set_owned_id", m, sid)
                    self.store.event(now, "model_claimed", row.get("job_id"), model=m, engine=row["engine"], reason=why)

    def _known(self, st: dict) -> dict[str, str]:
        known = dict(st["known_resident"] or {})
        for m, e in known.items():
            if e is None and m in self.catalog:
                known[m] = self.catalog[m].engine
        return known

    def _observe(self, now: float, fresh: bool = False) -> None:
        """Reconcile residency with engine reality (engines may evict on their own; another
        client may load models, catalog models included) and track how long the brain has been
        away. An engine that does not answer keeps its last known residency (conservative for
        memory planning), and its models count as pinned: they cannot be unloaded or used.
        `fresh` (once per tick) also reads each engine's instance id and reported model sizes."""
        st = self.store.state()
        self._safe_engine = st["safe_engine"] if st["safe"] else None
        known = self._known(st)
        current: dict[str, str] = {}
        answered: set = set()
        for name in self.engines:
            try:
                models = self._call(name, "resident_models", check=_model_set)
            except EngineUnavailable:
                current.update({m: e for m, e in known.items() if e == name})
                continue
            answered.add(name)
            for m in sorted(models):
                current.setdefault(m, name)
            if fresh:
                self._check_instance(name, now)
                self._observe_sizes(name, models, now)
                self._loading_seen[name] = set(self._optional(name, "loading_models", check=_model_set) or ())
        self._resident = current
        self._reconcile_owners(now, st, current, answered)
        loading = (st["loading"] or {}).get("model")
        events = [("model_evicted", {"model": m, "engine": known[m]}) for m in sorted(set(known) - set(current))]
        for m in sorted(set(current) - set(known)):
            if m == loading:
                continue
            data = {"model": m, "engine": current[m]}
            if m not in self.catalog:
                data.update(foreign=True, size_gb=self._size(m))
            elif not self._ours(m):
                data.update(foreign=True, size_gb=self._size(m),
                            note="a catalog model gpud did not load: someone else's, never unloaded by gpud")
            events.append(("model_appeared", data))
        updates: dict = {}
        if current != known:
            updates["known_resident"] = current
        brain = bool(self._usable_brains())
        if not brain and st["brainless_since"] is None:
            updates.update(brainless_since=now, brainless_epoch=st["brainless_epoch"] + 1)
            events.append(("brain_departed", {}))
        elif brain and st["brainless_since"] is not None:
            updates["brainless_since"] = None
            events.append(("brain_returned", {"away_s": now - st["brainless_since"],
                                              "brains": sorted(self._usable_brains())}))
        media = any(m in self.catalog and not self._is_brain(m) and self._ours(m) for m in current)  # gpud's own
        mode = ("safe" if st["safe"] else "media+brain" if brain and media else "brain" if brain
                else "brainless" if media else "empty")
        if mode != st["mode"]:
            updates["mode"] = mode
        if updates or events:
            with self.store.tx():
                if updates:
                    self.store.set_state(**updates)
                for kind, data in events:
                    self.store.event(now, kind, None, **data)

    # ------------------------------------------------------------------ loads

    def _check_load(self, now: float) -> None:
        st = self.store.state()
        load = st["loading"]
        if not load:
            return
        name, model = load.get("engine"), load.get("model")
        if model not in self.catalog:
            return self._abandon_load(now, f"unknown model {model!r}")
        limit = self._load_timeout(model)
        sent_to, current = load.get("instance"), self._inst_seen.get(name)
        if sent_to is not None and current is not None and sent_to != current:
            # The server restarted since the request was sent: that load died with it, and a copy
            # resident now was loaded by someone else (its "ready" ticket must not make it gpud's).
            return self._load_failed(load, "the server restarted while the load was in flight", now, definitive=True)
        try:
            status = self._call(name, "poll_load", load["ticket"], check=_load_status)
        except EngineUnavailable:
            return  # cannot see it; it blocks scheduling until the engine answers or is lost
        if status.state == "ready":
            try:
                resident = model in self._call(name, "resident_models", check=_model_set)
            except EngineUnavailable:
                return
            if not resident:  # 'ready' but already gone: a flapping model is a failed load
                return self._load_failed(load, "engine reported the load ready but the model is not resident", now)
            ladder = st["ladder"]
            ladder.get("failures", {}).pop(model, None)
            waited = ladder.get("deferred", {}).pop(model, None)
            known = self._known(st)
            known[model] = name
            extra = {"deferred_s": round(now - float(waited.get("since", now)), 1)} if waited else {}
            sid = self._optional(name, "owned_id", model, check=_opt_id)
            inst = current if current is not None else sent_to
            with self.store.tx():
                self.store.set_state(loading=None, ladder=ladder, known_resident=known)
                self._set_owner(model, engine=name, state="owned", instance=inst, boot=load.get("boot"),
                                owned_at=now, job_id=load.get("for_job"), server_id=sid, note="loaded by gpud")
                self.store.event(now, "model_loaded", load.get("for_job"), model=model, engine=name,
                                 took_s=now - load["started_at"], **extra)
            self._resident[model] = name
            return
        if status.state == "failed" and status.deferred and not self._is_brain(model):
            return self._load_deferred(load, status.error or "the engine deferred the load", now)
        if status.state == "loading" and now - load["started_at"] <= limit:
            return
        reason = status.error or f"load did not finish within {limit:.0f}s"
        definitive = None if status.state == "failed" and not status.in_flight else False
        if status.state == "loading":
            # Abort the stuck load the same way any unload is escalated: graceful unload, then
            # SIGTERM if refused, then SAFE. A refused abort must not be forgotten: the half
            # loaded model may still hold its memory.
            try:
                cleanup = self._unload(model, now, reason="abort stuck load", own_load=True)
            except EngineUnavailable:
                return  # retry the abort next tick
            if cleanup == "wedged":
                known = self._known(self.store.state())
                known[model] = name
                with self.store.tx():
                    self.store.set_state(known_resident=known)
                self._resident[model] = name
            else:
                # Is the request over? Only if the engine now says the load failed and will not
                # complete. An engine that could not abort it (the request already reached the
                # server, which may still finish it) keeps gpud's intent: the model is gpud's if it
                # appears, not someone else's to leave resident forever.
                try:
                    after = self._call(name, "poll_load", load["ticket"], check=_load_status)
                    definitive = None if after.state == "failed" and not after.in_flight else False
                except EngineUnavailable:
                    definitive = False
        self._load_failed(load, reason, now, definitive=definitive)

    def _abandon_load(self, now: float, reason: str) -> None:
        # The load intent (model_owners) stays: whether the request completes is unknown, so the
        # model is claimed if it appears within the load's timeout (_reconcile_owners).
        with self.store.tx():
            load = self.store.state()["loading"]
            if load:
                self.store.set_state(loading=None)
                self.store.event(now, "load_abandoned", load.get("for_job"), model=load.get("model"), reason=reason)

    def _end_intent(self, load: dict, now: float, definitive: bool) -> None:
        """A load that failed. If this process issued it and the engine answered that it failed
        (or refused it before sending), the request is over: the intent goes. Otherwise (a ticket
        from an earlier gpud, a load call that raised, a request gpud gave up on that the server
        may still finish) the intent waits, from now, for _reconcile_owners to claim the model or
        let the intent expire."""
        model = load.get("model")
        row = self._owners.get(model)
        if row is None or row.get("state") != "loading":
            return
        if definitive:
            self._drop_owner(model, now, event=False)
        else:
            self._set_owner(model, since=now, note="gpud gave up on the request; claimed if the model appears")

    def _load_deferred(self, load: dict, reason: str, now: float) -> None:
        """The engine refused a media model's load for now (LoadStatus.deferred: the host lacks
        free memory beside its other tenants, or a model slot). That is not a failed load: the job
        waiting on it stays queued, and the model is not tried again before a backoff
        (Policy.load_defer_s, doubling up to load_defer_max_backoff_s). Without this a transient
        memory dip failed every queued job of that model in turn, seconds apart, and each retry
        unloaded and reloaded other models around it. A wait longer than load_defer_max_s fails
        the job the load was for (its on_failure fires), one job per such wait. A brain's refusal
        is a failed load as before, so the ladder moves to a brain that fits."""
        model = load["model"]
        st = self.store.state()
        ladder = st["ladder"]
        waits = ladder.setdefault("deferred", {})
        w = waits.get(model) if isinstance(waits.get(model), dict) else {}
        since = number(w.get("since"))
        since = now if since is None else since
        n = int(number(w.get("n")) or 0) + 1
        waited = max(0.0, now - since)
        delay = min(self.p.load_defer_max_backoff_s, self.p.load_defer_s * 2 ** min(n - 1, 20))
        for_job = load.get("for_job")
        with self.store.tx():
            if waited >= self.p.load_defer_max_s:
                waits.pop(model, None)  # the next job that needs the model waits afresh
                self._end_intent(load, now, definitive=True)
                self.store.set_state(loading=None, ladder=ladder)
                self.store.event(now, "load_failed", for_job, model=model, attempt=n, reason=reason[:300],
                                 deferred_s=round(waited, 1))
                job = self.store.job(for_job) if for_job else None
                if job and job.state == "queued":
                    self._record_terminal(job, failed(
                        "load_failed", f"{model}: deferred for {waited:.0f}s: {reason}",
                        [f"supervisor: {model} could not be loaded for {waited:.0f}s ({reason})"]), now)
                return
            waits[model] = {"since": since, "n": n, "until": now + delay, "reason": str(reason)[:300]}
            self._end_intent(load, now, definitive=True)  # refused before the request was sent
            self.store.set_state(loading=None, ladder=ladder)
            if n & (n - 1) == 0:  # 1st, 2nd, 4th, 8th, ...: a long wait is logged on a log scale
                self.store.event(now, "load_deferred", for_job, model=model, deferrals=n, retry_in_s=delay,
                                 waited_s=round(waited, 1), reason=str(reason)[:300])

    def _load_failed(self, load: dict, reason: str, now: float, definitive: bool | None = None) -> None:
        """Retry each model once; a brain that fails twice is skipped (the ladder moves down);
        a media model that fails twice fails the job waiting on it (its on_failure fires)."""
        model = load["model"]
        st = self.store.state()
        ladder = st["ladder"]
        fails, skip = ladder.setdefault("failures", {}), ladder.setdefault("skip_until", {})
        n = fails.get(model, 0) + 1
        exhausted = n >= self.p.load_attempts_per_model
        if definitive is None:  # the engine answered "failed" for a ticket this process issued
            definitive = self._epoch is not None and load.get("boot") == self._epoch
        with self.store.tx():
            self._end_intent(load, now, definitive)
            self.store.event(now, "load_failed", load.get("for_job"), model=model, attempt=n, reason=reason)
            if exhausted:
                fails.pop(model, None)
                if self._is_brain(model):
                    skip[model] = now + self.p.ladder_cooldown_s
                    self.store.event(now, "brain_ladder_step", None, skipped=model, until=skip[model])
            else:
                fails[model] = n
            self.store.set_state(loading=None, ladder=ladder)
            if exhausted and not self._is_brain(model) and load.get("for_job"):
                job = self.store.job(load["for_job"])
                if job and job.state == "queued":
                    self._record_terminal(job, failed(
                        "load_failed", f"{model}: {reason}",
                        [f"supervisor: {model} failed to load {n} times ({reason})"]), now)

    def _ladder_ok(self, model: str, now: float, st: dict) -> bool:
        return st["ladder"].get("skip_until", {}).get(model, float("-inf")) <= now

    def _ladder_order(self, needs: str) -> list[str]:
        """Brains gpud may load, in ladder order (never one listed in Policy.foreign_models)."""
        order = ([self.p.vision_brain] if needs == "vision" else []) + list(self.p.text_ladder)
        return [m for m in dict.fromkeys(order) if m in self.catalog and m not in self.p.foreign_models]

    def _others_loading(self, model: str) -> bool:
        """Another client's load of `model` is in flight (the engine's optional loading_models,
        read once per tick). gpud's own intent (a load it sent, or gave up on and may still claim)
        is not someone else's."""
        name = self._engine_name(model)
        if model not in self._loading_seen.get(name, ()):
            return False
        row = self._owners.get(model)
        return not (row is not None and row.get("state") == "loading" and row.get("engine") == name)

    def _pinned(self) -> set[str]:
        """Resident models gpud cannot unload: those on an engine that is unavailable right now,
        every catalog model gpud did not load itself (someone else's: see _ours), and, when
        Policy.unload_foreign is off, every model outside the catalog. All of them still count
        against memory and model slots. A model another client is loading counts too (memory
        only: it is not resident yet)."""
        pinned = {m for m, e in self._resident.items() if not self._avail(e)}
        for m in self._resident:
            if m in self.catalog:
                if not self._ours(m):
                    pinned.add(m)
            elif not self.p.unload_foreign:
                pinned.add(m)
        for name, models in self._loading_seen.items():
            pinned.update(m for m in models if m not in self._resident and self._others_loading(m))
        return pinned

    def _slot_cap(self, engine: str | None) -> int | None:
        """Optional adapter attribute `max_resident_models`: how many models the engine keeps
        before it LRU-evicts one on its own (mlx-serve: 3). gpud frees a slot itself first, so
        the engine never picks the victim (it could pick a model gpud does not own)."""
        cap = number(getattr(self.engines.get(engine), "max_resident_models", None))
        return int(cap) if cap and cap >= 1 else None

    def _fits_alone(self, model: str) -> bool:
        """`model` fits beside everything gpud may not unload: in memory and in its engine's
        model slots."""
        pinned = self._pinned()
        held = sum(self._size(m) for m in pinned if m != model)
        return held + self._size(model) <= self.p.usable_gb and self._slots_ok({model}, pinned)

    def _loadable(self, model: str, now: float, st: dict) -> bool:
        """gpud can load this model now. Not if someone else already has it resident (gpud would
        only take over their copy), is loading it, or it is listed in Policy.foreign_models."""
        if model in self.p.foreign_models or (model in self._resident and not self._ours(model)) \
                or self._others_loading(model):
            return False
        return (model in self.catalog and self._ladder_ok(model, now, st)
                and self._avail(self.catalog[model].engine) and self._fits_alone(model))

    def _ladder_pick(self, needs: str, now: float, st: dict) -> str | None:
        """First brain in ladder order that can be loaded (not skipped, engine answering, fits
        beside pinned memory). While the brain is away, prefer one whose worst-case load still
        lands inside MAX_BRAINLESS_S; failing that, the one with the shortest worst case. A brain
        someone else is loading right now, ranked above every candidate, is waited for instead
        when gpud may use it (Policy.use_foreign_brains): no second brain beside it."""
        order = self._ladder_order(needs)
        cands = [m for m in order if self._loadable(m, now, st)]
        if self.p.use_foreign_brains:
            first = order.index(cands[0]) if cands else len(order)
            if any(self._others_loading(m) and self._ladder_ok(m, now, st) for m in order[:first]):
                return None
        if not cands:
            return None
        since = st["brainless_since"]
        if since is not None:
            away = now - since
            timely = [m for m in cands if away + self._load_timeout(m) <= self.p.max_brainless_s]
            if timely:
                return timely[0]
            return min(cands, key=self._load_timeout)
        return cands[0]

    def _ladder_exhausted(self, needs: str, now: float, st: dict) -> bool:
        """No brain in the ladder can ever come back without a human: each is on cooldown after
        failing its loads, or someone else holds it resident and gpud may not use their copy
        (Policy.use_foreign_brains off; gpud cannot load a second copy of the same model)."""
        def blocked(m: str) -> bool:
            return m in self._resident and not self._ours(m) and not self.p.use_foreign_brains
        return not any(self._ladder_ok(m, now, st) and not blocked(m) for m in self._ladder_order(needs))

    def _begin_load(self, model: str, now: float, for_job: int | None) -> bool:
        """Send a load (True), or skip it (False: nothing was sent, the caller's pass goes on)."""
        name = self._engine_name(model)
        skip = None
        if model in self._resident:
            skip = "resident already"  # someone else's copy: a no-op load would make gpud take it for its own
        elif model in self.p.foreign_models:
            skip = "listed in foreign_models"
        elif self._others_loading(model):
            skip = "another client's load of it is in flight (a second load would join it and look like gpud's)"
        if skip:
            self._note(now, "load_skipped", (model, for_job), job_id=for_job, model=model, ours=self._ours(model),
                       reason=skip)
            return False
        inst = self._inst_seen.get(name)  # None while the server's identity is unknown (never a stale one)
        # Intent first, in its own commit: if gpud dies with the request in flight, the next gpud
        # can still tell that the model it finds resident is its own (_reconcile_owners).
        with self.store.tx():
            self._set_owner(model, engine=name, state="loading", instance=inst, boot=self._epoch, since=now,
                            owned_at=None, job_id=for_job, server_id=None, note="load requested")
        self.crash_hook("after_load_intent")
        try:
            ticket = self._call(name, "load", model, check=_ticket)
        except EngineUnavailable as e:
            self._load_failed({"model": model, "engine": name, "for_job": for_job, "started_at": now,
                               "boot": self._epoch}, f"load request failed: {e.error}", now, definitive=False)
            return True
        with self.store.tx():
            self.store.set_state(loading={"model": model, "engine": name, "ticket": ticket, "started_at": now,
                                          "for_job": for_job, "boot": self._epoch, "instance": inst})
            self.store.event(now, "load_started", for_job, model=model, engine=name,
                             expected_s=self.catalog[model].load_s)
        self.crash_hook("after_load_requested")
        return True

    def _unload(self, model: str, now: float, reason: str, own_load: bool = False) -> str:
        """Graceful unload; if refused, graceful terminate; if that fails, SAFE + page.
        Returns 'unloaded', 'terminated' or 'wedged', or 'foreign' without calling the engine for
        a model gpud did not load (someone else's is never unloaded, whatever the caller wanted).
        `own_load` aborts gpud's own load in flight: its intent is left to _end_intent. There is
        no force-kill path, and no terminate of an engine that also holds models gpud did not
        load (that would take them down too): that is SAFE for the engine and a page. Raises
        EngineUnavailable if the engine does not answer (the caller retries later)."""
        if not own_load and not self._may_unload(model):
            self._note(now, "unload_refused", (model, reason), model=model, reason=reason,
                       note="not loaded by gpud: someone else's model is never unloaded")
            return "foreign"
        name = self._engine_name(model)
        if self._call(name, "unload", model, check=bool):
            self._resident.pop(model, None)
            with self.store.tx():
                known = self._known(self.store.state())
                known.pop(model, None)
                self.store.set_state(known_resident=known)
                if not own_load:
                    self._drop_owner(model, now, event=False)
                self.store.event(now, "model_unloaded", None, model=model, reason=reason)
            return "unloaded"
        with self.store.tx():
            self.store.event(now, "unload_failed", None, model=model, reason=reason)
        others = sorted({m for m, e in self._resident.items() if e == name and m != model and not self._may_unload(m)}
                        | {m for m in self._loading_seen.get(name, ()) if m != model and not self._ours(m)})
        if others:
            self._enter_safe(f"{name} refused to unload {model} ({reason}); terminating it would also take down "
                             f"models gpud did not load ({', '.join(others)}), so gpud leaves it alone: a human "
                             f"must intervene", now, engine=name)
            return "wedged"
        return "terminated" if self._terminate(name, now, f"unload of {model} refused") else "wedged"

    def _terminate(self, name: str, now: float, why: str) -> bool:
        with self.store.tx():
            self.store.event(now, "engine_terminate", None, engine=name, why=why, signal="SIGTERM")
        if self._call(name, "terminate", check=bool):
            gone = {m for m, e in self._resident.items() if e == name}
            for m in gone:
                self._resident.pop(m)
            with self.store.tx():
                known = {m: e for m, e in self._known(self.store.state()).items() if m not in gone and e != name}
                self.store.set_state(known_resident=known)
                for m in [m for m, row in self._owners.items() if row["engine"] == name]:
                    self._drop_owner(m, now, event=False)
                self.store.event(now, "engine_restarted", None, engine=name, dropped=sorted(gone))
            self._check_instance(name, now)  # the restarted server's identity, for what gpud loads next
            return True
        self._enter_safe(f"{name} ignored a graceful terminate ({why}); force-kill is never "
                         f"automatic, a human must intervene", now, engine=name)
        return False

    def _make_room(self, target: set[str], now: float) -> bool:
        """Unload models outside `target` (media first, biggest first) until target fits. Checks
        feasibility first, in memory and in model slots, so it never unloads anything (the brain
        included) for a target that cannot fit because memory or slots are pinned by someone
        else's models, an unreachable engine or a wedged one. `_room_why` says why it failed."""
        usable = self.p.usable_gb
        pinned = self._pinned()
        exclusive = any(m in self.catalog and self.catalog[m].exclusive for m in target)
        pinned_gb = sum(self._size(m) for m in pinned if m not in target)
        if (exclusive and any(m not in target for m in pinned)) or \
                pinned_gb + sum(self._size(m) for m in target) > usable:
            self._room_why = (f"{sorted(target)} needs {sum(self._size(m) for m in target):g} GB beside "
                              f"{pinned_gb:g} GB gpud may not unload ({sorted(pinned)}); {usable:g} GB usable")
            self._note(now, "insufficient_memory", (tuple(sorted(target)), tuple(sorted(pinned))),
                       target=sorted(target), pinned=sorted(pinned), pinned_gb=pinned_gb)
            return False
        if not self._slots_ok(target, pinned):
            name = next((e for e in sorted({self._engine_name(m) for m in target if self._engine_name(m)})
                         if not self._slots_ok({m for m in target if self._engine_name(m) == e}, pinned)), None)
            held = sorted(m for m, e in self._resident.items() if e == name)
            self._room_why = (f"{name} keeps at most {self._slot_cap(name)} models and {sorted(set(held) & pinned)} "
                              f"are not gpud's to unload")
            self._note(now, "insufficient_slots", (name, tuple(sorted(target)), tuple(held)), engine=name,
                       cap=self._slot_cap(name), target=sorted(target), resident=held,
                       pinned=sorted(set(held) & pinned))
            return False
        no_brain = any(m in self.catalog and self.catalog[m].evicts_brain for m in target)
        victims = sorted((m for m in self._resident if m not in target and m not in pinned),
                         key=lambda m: (self._is_brain(m), -self._size(m)))
        # another client's load in flight: not resident yet, but its memory is spoken for
        coming = sum(self._size(x) for x in pinned if x not in self._resident and x not in target)
        for m in victims:
            if m not in self._resident:
                continue  # dropped by an engine restart earlier in this loop
            used = sum(self._size(x) for x in self._resident) + coming
            need = sum(self._size(x) for x in target if x not in self._resident)
            must = exclusive or (m in self.catalog and self.catalog[m].exclusive) or (no_brain and self._is_brain(m))
            if not must and used + need <= usable:
                continue
            try:
                result = self._unload(m, now, reason="make room")
            except EngineUnavailable:
                self._room_why = f"{m} could not be unloaded yet (draining, or the engine does not answer)"
                return False
            if result == "wedged":
                self._room_why = f"{m} could not be unloaded (wedged)"
                return False
        used = sum(self._size(x) for x in self._resident) + coming
        need = sum(self._size(x) for x in target if x not in self._resident)
        if used + need > usable:
            self._room_why = f"{sorted(target)} needs {need:g} GB more, {used:g} of {usable:g} GB in use"
            self._note(now, "insufficient_memory", (tuple(sorted(target)), used, need),
                       target=sorted(target), used=used, need=need)
            return False
        if not self._make_slots(target, pinned, now):
            return False
        self._noted.pop("insufficient_memory", None)
        self._noted.pop("insufficient_slots", None)
        self._room_why = ""
        self.crash_hook("after_make_room")
        return True

    def _make_slots(self, target: set[str], pinned: set[str], now: float) -> bool:
        """Per-engine model-count caps (see _slot_cap): unload gpud's own idle models, media
        first and biggest first, until the target fits under each engine's cap."""
        for name in sorted({e for e in (self._engine_name(m) for m in target) if e}):
            cap = self._slot_cap(name)
            if cap is None:
                continue
            while True:
                held = [m for m, e in self._resident.items() if e == name]
                new = [m for m in target if m not in self._resident and self._engine_name(m) == name]
                if len(held) + len(new) <= cap:
                    break
                cands = sorted((m for m in held if m not in target and m not in pinned),
                               key=lambda m: (self._is_brain(m), -self._size(m)))
                if not cands:
                    self._room_why = f"{name} keeps at most {cap} models and none of {sorted(held)} is gpud's to unload"
                    self._note(now, "insufficient_slots", (name, tuple(sorted(target)), tuple(sorted(held))),
                               engine=name, cap=cap, target=sorted(target), resident=sorted(held))
                    return False
                try:
                    result = self._unload(cands[0], now, reason=f"make room ({name} keeps at most {cap} models)")
                except EngineUnavailable:
                    self._room_why = f"{cands[0]} could not be unloaded yet (draining, or the engine does not answer)"
                    return False
                if result in ("wedged", "foreign"):
                    self._room_why = f"{cands[0]} could not be unloaded ({result})"
                    return False
        return True

    # ------------------------------------------------------------------ running jobs & leases

    @staticmethod
    def _run_id(job: Job) -> str:
        return f"{job.id}.{job.attempt}" + (f".r{job.respawns}" if job.respawns else "")

    def _start(self, job: Job, model: str, now: float) -> None:
        name = self._engine_name(model)
        run_id = self._run_id(job)
        with self.store.tx():
            if not self.store.update_job(job.id, expect_state="queued", state="starting", model=model, engine=name,
                                         run_id=run_id, started_at=now):
                return
            self.store.event(now, "job_starting", job.id, kind=job.kind, model=model, attempt=job.attempt,
                             waited_s=now - job.created_at, continuation=bool(job.is_continuation))
            if job.is_continuation and job.deadline is not None and now > job.deadline + self.catalog[model].load_s:
                self.store.event(now, "brain_return_sla_missed", job.id, late_s=now - job.deadline)
        self.crash_hook("after_starting_recorded")
        spec = RunSpec(run_id, job.id, job.kind, model, job.payload, self.p.expected_s(job.requires, job.payload))
        try:
            handle = self._call(name, "start", spec, check=_handle)
        except EngineUnavailable as e:
            try:  # did the engine take the run before the error? start() is idempotent on run_id
                handle = self._call(name, "find", run_id, check=_opt_handle)
            except EngineUnavailable:
                return  # still 'starting': reconciled when the engine answers, failed if it is lost
            if not handle:  # engine refused; the outcome path still fires a continuation
                self._record_terminal(self.store.job(job.id), failed(
                    "engine_error", f"start failed: {e.error}", [f"supervisor: engine.start raised {e.error}"]), now)
                return
        self.crash_hook("after_engine_start")
        if not self._grant_lease(self.store.job(job.id), handle, now, expect="starting"):
            self._disown(name, handle, job.id, model, now)

    def _disown(self, name: str, handle: str, job_id: int, model: str, now: float) -> None:
        """We started a run for a job that is no longer ours to run (its state moved on)."""
        with self.store.tx():
            self.store.event(now, "stale_start", job_id, handle=handle)
        try:
            self._call(name, "cancel", handle)
        except EngineUnavailable:
            with self.store.tx():
                orphans = list(self.store.state()["orphans"] or [])
                orphans.append({"engine": name, "model": model, "job_id": job_id})
                self.store.set_state(orphans=orphans)

    def _grant_lease(self, job: Job, handle: str, now: float, expect: str, adopted: bool = False,
                     progress: float = 0.0) -> bool:
        hard = self.p.hard_timeout_s(job.requires, job.payload)
        pto = self.p.progress_timeout_for(job.requires, job.payload)
        granted = job.started_at if job.started_at is not None else now
        with self.store.tx():
            if not self.store.update_job(job.id, expect_state=expect, state="running", handle=handle):
                self.store.event(now, "stale_transition", job.id, wanted="running", expected=expect)
                return False
            lease = self.store.lease(job.id)
            if lease is not None and lease["state"] == "active" and lease["granted_at"] == granted:
                self.store.update_lease(job.id, fence=self._epoch or 0)  # re-fenced by this leader
            else:
                self.store.put_lease(job.id, engine=job.engine, model=job.model, granted_at=granted,
                                     hard_deadline=granted + hard, last_progress_at=now, last_progress=progress,
                                     expires_at=min(now + pto, granted + hard), progress_timeout_s=pto,
                                     fence=self._epoch or 0, state="active")
            self.store.event(now, "lease_adopted" if adopted else "lease_granted", job.id, model=job.model,
                             hard_timeout_s=hard, progress_timeout_s=pto, fence=self._epoch)
        return True

    def _reconcile(self, job: Job, now: float, boot: bool) -> None:
        """Settle a job whose engine-side state is unknown to this process."""
        if job.state == "running":
            if not boot:
                return
            try:
                poll = self._call(job.engine, "poll", job.handle, check=_poll) if job.handle else None
            except EngineUnavailable:
                return  # unknown for now; the lease keeps timing it, engine loss fails it
            if poll is not None and poll.status != "lost":
                self._grant_lease(job, job.handle, now, expect="running", adopted=True, progress=poll.progress)
            else:
                self._record_terminal(job, failed(
                    "interrupted", "supervisor restarted and the engine has no live run for this job",
                    poll.log_tail if poll else ["supervisor: run handle unknown to the engine"]), now)
            return
        if job.state != "starting":
            return
        handle = job.handle or self._call(job.engine, "find", job.run_id, check=_opt_handle)
        if handle:
            poll = self._call(job.engine, "poll", handle, check=_poll)
            if poll.status != "lost":
                self._grant_lease(job, handle, now, expect="starting", adopted=True, progress=poll.progress)
            else:
                self._record_terminal(job, failed(
                    "interrupted", "the engine lost the run while gpud was starting it", poll.log_tail), now)
            return
        # The engine never saw the run: gpud died between recording 'starting' and engine.start().
        # That is gpud's failure, not the job's: requeue it without charging an attempt (the same
        # run_id is reused, so a start that did reach the engine later is adopted, not duplicated).
        n = (job.interruptions or 0) + 1
        if n > self.p.max_interruptions:
            self._record_terminal(job, failed(
                "interrupted", f"gpud died {n} times while starting this job; treating it as poison",
                [f"supervisor: start interrupted {n} times"], poison=True), now)
            return
        with self.store.tx():
            if self.store.update_job(job.id, expect_state="starting", state="queued", interruptions=n,
                                     engine=None, model=None, handle=None, started_at=None):
                self.store.event(now, "start_interrupted", job.id, interruptions=n,
                                 note="engine never saw the run; requeued without charging an attempt")

    def _check_running(self, now: float) -> None:
        for job in self.store.jobs(*ACTIVE):
            self._guard_job(job, lambda j=job: self._check_job(j, now), now)

    def _check_job(self, job: Job, now: float) -> None:
        if job.state == "cancelling":
            return self._advance_cancel(job, now)
        if job.state == "starting":
            return self._reconcile(job, now, boot=False)
        try:
            poll = self._call(job.engine, "poll", job.handle, check=_poll)
        except EngineUnavailable:
            poll = None  # unobservable: the dead-man switch still runs on the persisted lease
        if poll is None or poll.status == "running":
            return self._check_lease(job, poll, now)
        if poll.status == "done":
            return self._complete(job, poll, now)
        if poll.status == "failed":
            reason = poll.reason if poll.reason in ENGINE_FAILURE_REASONS else "crash"
            return self._record_terminal(job, failed(reason, poll.error or "engine reported failure", poll.log_tail,
                                                     **self._artifact_fields(job, poll.artifact)), now)
        self._record_terminal(job, failed("engine_lost", f"engine reported '{poll.status}'", poll.log_tail), now)

    def _check_lease(self, job: Job, poll: Poll | None, now: float) -> None:
        lease = self.store.lease(job.id)
        if lease is None:  # defensive: running jobs always get a lease in the same transaction
            self._grant_lease(job, job.handle, now, expect="running", progress=poll.progress if poll else 0.0)
            return
        epoch = self._epoch or 0
        if lease["fence"] != epoch:
            if lease["fence"] > epoch:
                return  # a newer leader owns it; _still_leader() will step us down
            with self.store.tx():
                self.store.update_lease(job.id, fence=epoch)
            lease["fence"] = epoch
        if poll is not None and poll.progress > lease["last_progress"] + 1e-9:  # only progress renews
            pto = lease["progress_timeout_s"] or self.p.progress_timeout_s
            expires = min(now + pto, lease["hard_deadline"])
            with self.store.tx():
                renewed = self.store.update_lease(job.id, owner=epoch, last_progress=poll.progress,
                                                  last_progress_at=now, expires_at=expires)
                if not renewed:
                    self.store.event(now, "stale_lease_renewal", job.id, fence=epoch)
            if not renewed:
                return
            lease.update(last_progress=poll.progress, last_progress_at=now, expires_at=expires)
            self.crash_hook("after_lease_renewed")
        tail = poll.log_tail if poll is not None else ("supervisor: engine unreachable; no progress observed",)
        progress = poll.progress if poll is not None else lease["last_progress"]
        pto = lease["progress_timeout_s"] or self.p.progress_timeout_s
        if now >= lease["hard_deadline"]:
            limit = lease["hard_deadline"] - lease["granted_at"]
            self._begin_cancel(job, failed("timeout", f"hard timeout: running {now - lease['granted_at']:.0f}s "
                                                      f"> {limit:.0f}s", tail, timeout="hard"),
                               now, lease_state="expired")
        elif now - lease["last_progress_at"] >= pto:
            self._begin_cancel(job, failed("timeout", f"no progress for {now - lease['last_progress_at']:.0f}s "
                                                      f"(stuck at {progress:.0%})", tail, timeout="no_progress"),
                               now, lease_state="expired")

    def _begin_cancel(self, job: Job, outcome: dict, now: float, lease_state: str, advance: bool = True) -> bool:
        with self.store.tx():
            if not self.store.update_job(job.id, expect_state="running", state="cancelling", pending_outcome=outcome,
                                         cancel_requested_at=now, cancel_stage="cancel"):
                return False
            self.store.update_lease(job.id, state=lease_state)
            self.store.event(now, "lease_expired" if lease_state == "expired" else "cancel_requested", job.id,
                             reason=outcome["reason"], detail=outcome["detail"])
        self.crash_hook("after_cancel_recorded")
        if advance:
            self._guard_job(job, lambda: self._advance_cancel(self.store.job(job.id), now), now)
        return True

    def _advance_cancel(self, job: Job, now: float) -> None:
        """graceful cancel -> wait up to CANCEL_GRACE_S -> graceful unload -> (terminate only if
        unload is refused) -> record the pending outcome. Never a force-kill. A cancel the engine
        rejects (e.g. 404 for a run it already reaped) skips the grace wait instead of wedging."""
        if job.cancel_stage == "cancel":
            try:
                self._call(job.engine, "cancel", job.handle)
                stage, error = "cancel_sent", None
            except EngineUnavailable as e:
                stage, error = "cancel_failed", e.error
            with self.store.tx():
                if not self.store.update_job(job.id, expect_state="cancelling", cancel_stage=stage):
                    return
                if error is None:
                    self.store.event(now, "cancel_sent", job.id, graceful=True)
                else:
                    self.store.event(now, "cancel_failed", job.id, error=error,
                                     note="going straight to the graceful unload")
            self.crash_hook("after_cancel_sent")
            job = self.store.job(job.id)
        try:
            poll = self._call(job.engine, "poll", job.handle, check=_poll)
        except EngineUnavailable:
            poll = None
        waiting = job.cancel_stage == "cancel_sent" and now - job.cancel_requested_at < self.p.cancel_grace_s
        if waiting and (poll is None or poll.status == "running"):
            return
        self.crash_hook("before_cleanup_unload")
        cleanup = self._unload(job.model, now, reason=f"cleanup after {job.pending_outcome['reason']}")
        self.crash_hook("after_cleanup_unload")
        outcome = dict(job.pending_outcome, cleanup=cleanup)
        if poll is not None:
            if poll.log_tail:
                outcome["log_tail"] = list(poll.log_tail)
            if poll.status in ("done", "failed") and poll.artifact:
                # It finished (or died after writing output) during the cancel grace: keep what it
                # produced in the brief so the brain can inspect it instead of re-rendering blind.
                outcome.update(self._artifact_fields(job, poll.artifact), finished_during_cancel=poll.status)
        self._record_terminal(job, outcome, now)

    def _validate(self, job: Job, artifact) -> list[dict]:
        family = "brain" if job.requires == "brain" else self._family(job.model)
        try:
            return validators.check(family, job.payload, artifact)
        except Exception as e:  # a validator bug is a failed check, not a stuck job
            return [{"check": "validator_error", "ok": False, "detail": f"validator raised {type(e).__name__}: {e}"}]

    def _artifact_fields(self, job: Job, artifact) -> dict:
        if job.requires == "brain" or not isinstance(artifact, dict) or not artifact.get("path"):
            return {}
        return {"artifact": artifact, "validator_results": self._validate(job, artifact)}

    def _complete(self, job: Job, poll: Poll, now: float) -> None:
        results = self._validate(job, poll.artifact)
        ok = validators.passed(results)
        self._record_terminal(job, {
            "status": "succeeded" if ok else "failed", "reason": None if ok else "validator",
            "detail": None if ok else validators.failures(results), "artifact": poll.artifact,
            "validator_results": results, "log_tail": list(poll.log_tail)}, now)

    # ------------------------------------------------------------------ outcomes

    def _record_terminal(self, job: Job, outcome: dict, now: float) -> None:
        """Durably record how a job ended (compare-and-set on the state we observed). The
        continuation is fired separately from this record (outbox pattern), so a crash between
        the two is repaired by the next tick and the UNIQUE idempotency key keeps it single."""
        outcome = {"artifact": None, "validator_results": [], "log_tail": [], "detail": None,
                   **outcome, "model": job.model}
        with self.store.tx():
            cur = self.store.job(job.id)
            if cur is None or cur.state != job.state:
                return
            if cur.requires == "brain":
                state = self._finish_brain_job(cur, outcome, now)
            else:
                state = outcome["status"]
                if state == "failed" and (outcome.get("poison") or
                                          (outcome["reason"] != "cancelled" and cur.attempt >= cur.max_attempts)):
                    state = "quarantined"
                if not self.store.update_job(cur.id, expect_state=cur.state, state=state, outcome=outcome,
                                             finished_at=now, pending_outcome=None, cancel_stage=None):
                    return
            lease = self.store.lease(cur.id)
            if lease and lease["state"] == "active":
                self.store.update_lease(cur.id, state="released")
            self.store.event(now, "job_finished", cur.id, kind=cur.kind, state=state, reason=outcome["reason"],
                             detail=outcome["detail"], model=cur.model, attempt=cur.attempt)
        if not self.store.in_tx:
            self.crash_hook("after_terminal_recorded")

    def _finish_brain_job(self, cur: Job, outcome: dict, now: float) -> str:
        """Brain jobs have no continuation (they ARE the intelligence). Failures are retried on
        the same row up to max_attempts, then quarantined for a human. Infrastructure failures
        (engine unreachable) are requeued without charging an attempt, up to max_interruptions."""
        reset = dict(pending_outcome=None, cancel_stage=None, cancel_requested_at=None)
        if cur.kind == PLAN_KIND:
            return self._finish_plan_job(cur, outcome, now, reset)
        requeue = dict(state="queued", handle=None, run_id=None, engine=None, model=None, started_at=None, **reset)
        if outcome["status"] == "succeeded":
            outcome = self._apply_decision_safely(cur, outcome, now)
            self.store.update_job(cur.id, state="succeeded", outcome=outcome, finished_at=now, **reset)
            return "succeeded"
        reason = outcome["reason"]
        if reason in INFRA_REASONS and not outcome.get("poison") and (cur.interruptions or 0) < self.p.max_interruptions:
            self.store.update_job(cur.id, outcome=outcome, interruptions=(cur.interruptions or 0) + 1,
                                  respawns=(cur.respawns or 0) + 1, **requeue)
            self.store.event(now, "brain_job_requeued", cur.id, reason=reason, interruptions=(cur.interruptions or 0) + 1)
            return "queued"
        if reason != "cancelled" and not outcome.get("poison") and reason not in INFRA_REASONS \
                and cur.attempt < cur.max_attempts:
            self.store.update_job(cur.id, attempt=cur.attempt + 1, outcome=outcome, **requeue)
            self.store.event(now, "brain_job_retry", cur.id, attempt=cur.attempt + 1, reason=reason)
            return "queued"
        state = "failed" if reason == "cancelled" else "quarantined"
        self.store.update_job(cur.id, state=state, outcome=outcome, finished_at=now, **reset)
        if state == "quarantined":
            self._human_review(cur, f"brain job '{cur.kind}' failed {cur.attempt} times "
                                    f"(last: {reason})", now, idem=f"review:{cur.id}",
                               brief=cur.payload.get("brief"))
        return state

    def _finish_plan_job(self, cur: Job, outcome: dict, now: float, reset: dict) -> str:
        """A plan job ends once, either way: no retry (the next tick plans afresh) and no page (the
        planner counts invalid ticks and suspends itself). Its proposals are validated and admitted
        by the planner inside this transaction, in a savepoint: a planner bug is recorded, never
        fatal, and a decision's `submit` is never admitted for a plan job."""
        ok = outcome["status"] == "succeeded"
        ours = str(cur.idempotency_key or "").startswith(PLAN_KEY_PREFIX) and isinstance(cur.payload, dict) \
            and bool(cur.payload.get("resident_only"))
        try:
            with self.store.savepoint():
                if self.planner is None:
                    summary = {"ignored": "no planner is attached to this gpud"}
                elif not ours:
                    summary = {"ignored": "not a planner tick: only plan jobs the planner created are applied"}
                    self.store.event(now, "plan_ignored", cur.id, key=cur.idempotency_key)
                elif ok:
                    summary = self.planner.apply_plan(self, cur, outcome, now)
                else:
                    summary = self.planner.plan_failed(self, cur, outcome, now)
        except Exception as e:
            log.exception("could not apply plan job %s", cur.id)
            summary = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            self.store.event(now, "plan_error", cur.id, error=summary["error"])
            if self.planner is not None and ours:
                # The reply crashed the application: it still costs its tokens and counts as an
                # invalid tick, so a reply that breaks apply cannot repeat for free forever.
                try:
                    with self.store.savepoint():
                        summary.update(self.planner.plan_crashed(self, cur, outcome, now, summary["error"]))
                except Exception:
                    log.exception("could not record the failed plan job %s", cur.id)
        state = "succeeded" if ok else "failed"
        self.store.update_job(cur.id, state=state, outcome=dict(outcome, plan=summary), finished_at=now, **reset)
        return state

    def _apply_decision_safely(self, cur: Job, outcome: dict, now: float) -> dict:
        try:
            with self.store.savepoint():
                return self._apply_decision(cur, outcome, now)
        except Exception as e:
            log.exception("could not apply the decision of job %s", cur.id)
            err = f"decision could not be applied: {type(e).__name__}: {e}"
            self.store.event(now, "decision_error", cur.id, error=err)
            return {**outcome, "children": [], "rejected": [{"kind": None, "error": err}]}

    def _apply_decision(self, cur: Job, outcome: dict, now: float) -> dict:
        """Admit what the brain asked for. Each request is admitted in its own savepoint, and any
        exception (not just AdmissionError) becomes a recorded rejection: LLM output is
        untrusted and must never take the supervisor down or roll back the brain job's record."""
        art = outcome.get("artifact") if isinstance(outcome.get("artifact"), dict) else {}
        decision = art.get("decision") if isinstance(art.get("decision"), dict) else {}
        submit = decision.get("submit")
        children, rejected, escalated = [], [], False

        def reject(kind, error: str, quiet: bool = False) -> None:
            rejected.append(dict({"kind": kind, "error": error}, **({"quiet": True} if quiet else {})))
            self.store.event(now, "admission_rejected", cur.id, kind=kind, error=error)

        if submit is None:
            requests = []
        elif isinstance(submit, list):
            requests = submit
        else:
            requests = []
            reject(None, f"decision.submit must be a list of requests, got {type(submit).__name__}")
        for i, req in enumerate(requests):
            kind = req.get("kind") if isinstance(req, dict) else None
            kind = kind if isinstance(kind, str) else None
            if i >= self.p.max_submits_per_decision:
                reject(kind, f"fan-out limit: at most {self.p.max_submits_per_decision} requests per decision")
                continue
            try:
                with self.store.savepoint():
                    children.append(self._admit(req, now, parent=cur, idem=f"child:{cur.id}.{cur.attempt}.{i}"))
            except Exception as e:
                err = str(e) if isinstance(e, AdmissionError) else f"malformed request: {type(e).__name__}: {e}"
                reject(kind, err, quiet=isinstance(e, LineageClosed))
                if isinstance(e, EscalatingAdmissionError):
                    self._human_review(cur, err, now, idem=e.idem, brief=cur.payload.get("brief"))
                    escalated = True
        if decision.get("verdict") == "human_review":
            # The brain itself asks for a person (it is unsure, or its own output was unusable).
            note = decision.get("note") if isinstance(decision.get("note"), str) else ""
            self._human_review(cur, f"the brain asked for human review: {note}"[:500], now,
                               idem=f"brain-review:{cur.id}.{cur.attempt}", brief=cur.payload.get("brief"))
            escalated = True
        if cur.is_continuation and rejected and not children and not escalated \
                and any(not r.get("quiet") for r in rejected):
            # The baton reached the brain but nothing it asked for can run: do not let the
            # logical task end silently. (A lineage the owner closed by pausing its template
            # ends on purpose: nobody is paged for that.)
            self._human_review(cur, f"every request in the brain's {cur.kind} decision was rejected: "
                               + "; ".join(r["error"] for r in rejected)[:500], now,
                               idem=f"rejected:{cur.id}", brief=cur.payload.get("brief"))
        self.store.event(now, "brain_decision", cur.id, model=cur.model, submitted=children,
                         rejected=len(rejected), verdict=decision.get("verdict"), note=decision.get("note"))
        return {**outcome, "children": children, "rejected": rejected}

    def _brief(self, job: Job, cont: dict | None, context=None) -> dict:
        """Everything a freshly loaded, stateless brain needs to decide what to do next."""
        o = job.outcome or {}
        art = o.get("artifact") if isinstance(o.get("artifact"), dict) else {}
        paths = [art["path"]] if isinstance(art.get("path"), str) and art["path"] else []
        if isinstance(art.get("paths"), list):
            paths += [x for x in art["paths"] if isinstance(x, str)]
        payload = job.payload if isinstance(job.payload, dict) else {}
        return {
            "goal": payload.get("goal"),
            "attempted_action": {
                "kind": job.kind, "model": job.requires,
                "params": {k: v for k, v in payload.items() if k not in ("goal", "brief")},
                "on_success": cont["on_success"] if cont else None,
                "on_failure": cont["on_failure"] if cont else None},
            "outcome": {"status": job.state, "reason": o.get("reason"), "detail": o.get("detail"),
                        "timeout": o.get("timeout"), "cleanup": o.get("cleanup"),
                        "finished_during_cancel": o.get("finished_during_cancel")},
            "artifacts": paths,
            "validator_results": o.get("validator_results") or [],
            "log_tail": [str(x) for x in (o.get("log_tail") or [])] or ["supervisor: no engine output captured"],
            "attempt": job.attempt,
            "max_attempts": job.max_attempts,
            "source_job_id": job.id,
            "task_key": job.task_key,
            "ran": {"model": job.model, "started_at": job.started_at, "finished_at": job.finished_at},
            "context": context,
        }

    def _fire_continuations(self, now: float) -> None:
        for job, cont in self.store.unfired_continuations():
            self._guard_job(job, lambda j=job, c=cont: self._fire_one(j, c, now), now)

    def _fire_one(self, job: Job, cont: dict, now: float) -> None:
        label, spec = ("success", cont["on_success"]) if job.state == "succeeded" else ("failure", cont["on_failure"])
        brief = self._brief(job, cont, context=spec.get("context"))
        with self.store.tx():
            if job.state == "quarantined" or job.depth + 1 > self.p.max_continuation_depth:
                label = "quarantine" if job.state == "quarantined" else "depth_limit"
                reason = (f"task {job.task_key} failed {job.attempt}/{job.max_attempts} attempts "
                          f"(last: {brief['outcome']['reason']})" if label == "quarantine"
                          else f"continuation depth limit {self.p.max_continuation_depth} reached")
                child = self._human_review(job, reason, now, idem=f"cont:{job.id}", brief=brief,
                                           is_continuation=True)
            else:
                child, _ = self._insert_job(
                    kind=spec["kind"], requires="brain", state="queued",
                    payload={"needs": spec.get("needs", "any"), "brief": brief},
                    parent_id=job.id, root_id=job.root_id or job.id, depth=job.depth + 1,
                    max_attempts=self.p.max_attempts, priority=self.p.continuation_priority,
                    deadline=now + self.p.brain_return_sla_s, idempotency_key=f"cont:{job.id}",
                    is_continuation=1, created_at=now)
            self.crash_hook("inside_fire_txn")
            self.store.mark_fired(job.id, label, child, now)
            self.store.event(now, "continuation_fired", job.id, child=child, outcome=label,
                             kind=self.store.job(child).kind)
        self.crash_hook("after_continuation_fired")

    def _fire_fallback(self, job: Job, detail: str, now: float) -> None:
        """The continuation for a terminal job could not be built (gpud kept raising): hand the
        baton to a human instead, so exactly one continuation still fires."""
        if not self.store.continuation(job.id) or self.store.continuation(job.id)["fired_job_id"]:
            return
        payload = job.payload if isinstance(job.payload, dict) else {}
        brief = {"goal": payload.get("goal"), "attempted_action": {"kind": job.kind, "model": job.requires},
                 "outcome": {"status": job.state, "reason": (job.outcome or {}).get("reason"), "detail": detail},
                 "artifacts": [], "validator_results": [], "log_tail": [f"supervisor: {detail}"],
                 "attempt": job.attempt, "max_attempts": job.max_attempts, "source_job_id": job.id}
        with self.store.tx():
            child = self._human_review(job, f"continuation could not be built: {detail}", now,
                                       idem=f"cont:{job.id}", brief=brief, is_continuation=True)
            self.store.mark_fired(job.id, "error", child, now)
            self.store.event(now, "continuation_fired", job.id, child=child, outcome="error", kind="human_review")

    def _human_review(self, source: Job | None, reason: str, now: float, idem: str, brief=None,
                      is_continuation: bool = False, kind: str = "human_review") -> int:
        job_id, created = self._insert_job(
            kind=kind, requires="human", state="awaiting_human",
            payload={"reason": reason, "source_job_id": source.id if source else None, "brief": brief},
            parent_id=source.id if source else None, root_id=(source.root_id or source.id) if source else None,
            depth=source.depth + 1 if source else 0, max_attempts=1, priority=self.p.continuation_priority,
            idempotency_key=idem, is_continuation=int(is_continuation), created_at=now)
        if created:
            self.store.event(now, kind, job_id, reason=reason, source=source.id if source else None)
        return job_id

    def _page_humans(self, now: float) -> None:
        """Pages go through an outbox: at-least-once, never silently dropped. A pager outage is
        retried with backoff and never takes the supervisor down."""
        for job in self.store.unpaged_humans():
            if job.page_retry_at is not None and now < job.page_retry_at:
                continue
            reason = job.payload.get("reason") if isinstance(job.payload, dict) else None
            try:
                self.pager({"job_id": job.id, "kind": job.kind, "reason": reason})
            except Exception as e:
                n = (job.page_failures or 0) + 1
                retry = now + min(self.p.page_backoff_max_s, 5.0 * 2 ** (n - 1))
                with self.store.tx():
                    self.store.update_job(job.id, page_failures=n, page_retry_at=retry)
                    self.store.event(now, "page_failed", job.id, failures=n, retry_at=retry,
                                     error=f"{type(e).__name__}: {e}")
                continue
            with self.store.tx():
                self.store.update_job(job.id, paged_at=now)
                self.store.event(now, "page_human", job.id, reason=reason)

    def _enter_safe(self, reason: str, now: float, engine: str | None = None) -> None:
        """SAFE mode: page a human and stop automatic recovery. With `engine` (one engine is
        wedged and ignored SIGTERM) it is degraded SAFE: that engine is never touched again, no
        media work is dispatched, but a brain may still be brought back on another engine when
        it fits beside the wedged engine's memory, so the intelligence can still come back."""
        st = self.store.state()
        if st["safe"]:
            if st["safe_engine"] and (engine is None or engine != st["safe_engine"]):
                with self.store.tx():  # a second, unrelated failure: escalate to full SAFE
                    self.store.set_state(safe_engine=None, safe_reason=reason)
                    self.store.event(now, "safe_mode_escalated", None, reason=reason)
                    self._human_review(None, f"SAFE mode (full): {reason}", now,
                                       idem=f"safe:{st['safe_epoch']}:full", kind="safe_mode")
                self._safe_engine = None
            return
        with self.store.tx():
            self.store.set_state(safe=1, safe_reason=reason, safe_engine=engine, mode="safe")
            self.store.event(now, "safe_mode_entered", None, reason=reason, engine=engine)
            self._human_review(None, f"SAFE mode: {reason}", now, idem=f"safe:{st['safe_epoch']}", kind="safe_mode")
        self._safe_engine = engine

    # ------------------------------------------------------------------ scheduling

    def _schedule(self, now: float) -> None:
        """One GPU job at a time. Continuations outrank everything; idle means 'bring the brain
        home'. A job that cannot be dispatched right now (its engine is unavailable, memory is
        pinned) does not block the jobs behind it."""
        st = self.store.state()
        if st["loading"] or self.store.jobs(*ACTIVE):
            self._idle_since = None
            return
        if st["safe"] and not st["safe_engine"]:
            return
        degraded = bool(st["safe"])
        queued = self.store.queued_jobs()
        if queued:
            self._idle_since = None
        for job in queued:
            if degraded and job.requires != "brain":
                continue  # degraded SAFE: only the brain may come back; media waits for a human
            if self._guard_job(job, lambda j=job: self._dispatch(j, now, st), now, default=True):
                return
            st = self.store.state()
            if st["loading"] or self.store.jobs(*ACTIVE) or (st["safe"] and not degraded):
                return
        if not queued or not self._usable_brains():
            self._go_home(now, st)
        if not queued and not degraded:
            self._release_idle_media(now)

    def _release_idle_media(self, now: float) -> None:
        """Policy.idle_media_unload_s: once the queue has been empty that long (and a brain is
        home), unload gpud's own idle media models, one per tick, so their memory and engine
        model slots go back to the host and its other users. Off (None) by default."""
        limit = self.p.idle_media_unload_s
        if limit is None or not self._usable_brains():
            return
        if self._idle_since is None:
            self._idle_since = now
            return
        idle = now - self._idle_since
        if idle < limit:
            return
        pinned = self._pinned()
        for m in sorted(self._resident):
            if m in self.catalog and not self._is_brain(m) and m not in pinned and self._avail(self._resident[m]):
                try:
                    self._unload(m, now, reason=f"idle: nothing queued for {idle:.0f}s")
                except EngineUnavailable:
                    pass
                return

    def _dispatch(self, job: Job, now: float, st: dict) -> bool:
        """Try to move one queued job forward. True if something happened (a load started, the
        job started or ended, a checkpoint was inserted); False if it must wait."""
        if job.requires == "brain":
            needs = job.payload.get("needs", "any") if isinstance(job.payload, dict) else "any"
            needs = needs if needs in NEEDS else "any"
            brain = self._resident_brain_for(needs, now, st)
            if brain:
                self._start(job, brain, now)
                return True
            if job.kind == PLAN_KIND or (isinstance(job.payload, dict) and job.payload.get("resident_only")):
                return False  # never load a model for this job (the planner's): wait for a resident brain
            return self._bring_brain(needs, now, job.id, st)
        model = job.requires
        name = self._engine_name(model)
        if model not in self.catalog or name not in self.engines:
            self._record_terminal(job, failed("engine_unconfigured", f"{model} runs on {name}, which gpud has no "
                                              f"adapter for", [f"supervisor: no adapter for {name}"]), now)
            return True
        if not self._avail(name):
            return False
        wait = ((st.get("ladder") or {}).get("deferred") or {}).get(model)
        if isinstance(wait, dict) and model not in self._resident and now < (number(wait.get("until")) or 0.0):
            return False  # the engine deferred this model's load (free memory): wait out the backoff, unload nothing
        if model in self.p.foreign_models and model not in self._resident:
            # Never gpud's to load: only someone else's resident copy could serve it (read-only).
            # Nothing is unloaded for it, and the jobs behind it are not held up.
            return self._media_wait(job, model, now, f"{model} is listed in foreign_models (gpud never loads it) "
                                                      f"and nobody has it resident")
        target, co_brain = self._plan_media(model, now, st)
        if co_brain is None and st["brainless_since"] is not None:
            away, est = now - st["brainless_since"], self._brainless_estimate(job, model)
            if away + est > self.p.max_brainless_s:
                return self._cap_checkpoint(job, away, est, now, st)
        if not self._make_room(target, now):
            return self._media_wait(job, model, now, self._room_why or "gpud could not make room for it")
        for m in [model] + ([co_brain] if co_brain else []):
            if m not in self._resident:
                if self._begin_load(m, now, job.id):
                    self._media_unblock(job, st)
                    return True
                if m == model:
                    return self._media_wait(job, model, now, f"the load of {model} was skipped (see load_skipped)")
                break  # the co-resident brain cannot be loaded right now: run without it
        self._media_unblock(job, st)
        self._start(job, model, now)
        return True

    def _media_wait(self, job: Job, model: str, now: float, why: str) -> bool:
        """A media job gpud cannot move forward for a reason that need not clear by itself (the
        memory or the model slots it needs are held by models gpud may not unload, its model is
        never gpud's to load, ...). It waits without holding up the queue, for at most
        Policy.load_defer_max_s, then fails with load_failed (an infrastructure failure: its
        on_failure baton fires, and a planned item stops counting toward stock). True if the job
        was failed now."""
        st = self.store.state()
        ladder = st["ladder"]
        blocked = ladder.setdefault("blocked", {})
        key = str(job.id)
        ent = blocked.get(key) if isinstance(blocked.get(key), dict) else None
        if ent is None:
            for k in list(blocked):  # forget waits of jobs that moved on
                other = self.store.job(int(k)) if k.isdigit() else None
                if other is None or other.state != "queued":
                    blocked.pop(k, None)
            blocked[key] = {"since": now, "why": why[:300]}
            with self.store.tx():
                self.store.set_state(ladder=ladder)
                self.store.event(now, "media_blocked", job.id, model=model, reason=why[:300],
                                 fails_after_s=self.p.load_defer_max_s)
            return False
        waited = max(0.0, now - (number(ent.get("since")) or now))
        if waited < self.p.load_defer_max_s:
            return False
        blocked.pop(key, None)
        with self.store.tx():
            self.store.set_state(ladder=ladder)
            self._record_terminal(job, failed(
                "load_failed", f"{model}: could not be run for {waited:.0f}s: {why}",
                [f"supervisor: {model} waited {waited:.0f}s ({why})"]), now)
        return True

    def _media_unblock(self, job: Job, st: dict) -> None:
        if str(job.id) in ((st.get("ladder") or {}).get("blocked") or {}):
            with self.store.tx():
                ladder = self.store.state()["ladder"]
                (ladder.get("blocked") or {}).pop(str(job.id), None)
                self.store.set_state(ladder=ladder)

    def _ladder_why(self, needs: str, now: float, st: dict) -> str:
        """Why no brain in the ladder can be loaded right now, for SAFE reasons and pages."""
        parts = []
        for m in self._ladder_order(needs):
            if not self._ladder_ok(m, now, st):
                parts.append(f"{m} failed its loads")
            elif m in self._resident and not self._ours(m):
                parts.append(f"{m} is someone else's" + ("" if self.p.use_foreign_brains else
                                                         " (use_foreign_brains is off)"))
            elif self._others_loading(m):
                parts.append(f"another client is loading {m}")
            elif not self._avail(self.catalog[m].engine):
                parts.append(f"{m}'s engine is unavailable")
            elif not self._fits_alone(m):
                parts.append(f"{m} does not fit beside the memory or model slots gpud may not free")
        skipped = [m for m in ([self.p.vision_brain] if needs == "vision" else []) + list(self.p.text_ladder)
                   if m in self.p.foreign_models]
        if skipped:
            parts.append(f"{', '.join(dict.fromkeys(skipped))} listed in foreign_models")
        return "; ".join(parts) or "the ladder is empty"

    def _brain_blocked(self, now: float, st: dict, why: str) -> None:
        """No brain for Policy.brain_blocked_page_s while gpud cannot bring one back by itself
        (someone else's models hold the memory or the model slots, ...): page a human once per
        absence. An engine outage has its own page (engine_lost); SAFE mode has paged already."""
        since = st.get("brainless_since")
        if since is None or st.get("loading") or st.get("safe") or now - since < self.p.brain_blocked_page_s:
            return
        if self._usable_brains() or not any(self._avail(self.catalog[m].engine) for m in self._ladder_order("any")):
            return
        with self.store.tx():
            self._human_review(None, f"no brain for {now - since:.0f}s and gpud cannot bring one back by itself: "
                                     f"{why}"[:500], now, idem=f"brain-blocked:{st['brainless_epoch']}",
                               kind="brain_blocked")

    def _go_home(self, now: float, st: dict) -> None:
        want = self._ladder_pick("any", now, st)
        if want is None:
            if not self._usable_brains() and self._ladder_exhausted("any", now, st):
                self._enter_safe(f"no brain in the ladder can be loaded ({self._ladder_why('any', now, st)})", now)
            elif not self._usable_brains():
                self._note(now, "brain_unavailable", "home", reason="no brain engine is available or fits")
                self._brain_blocked(now, st, self._ladder_why("any", now, st))
            return
        if want in self._resident and self._avail(self._resident[want]):
            return
        rank = list(dict.fromkeys(list(self.p.text_ladder) + [self.p.vision_brain]))
        if any(m in rank and rank.index(m) <= rank.index(want) for m in self._usable_brains()):
            return  # a brain ranked at least as high is home already (someone else's, used read-only)
        if self._make_room({want}, now):
            self._begin_load(want, now, None)
        elif not self._usable_brains():
            self._brain_blocked(now, st, f"no room for {want}: {self._room_why}")

    def _bring_brain(self, needs: str, now: float, for_job: int, st: dict) -> bool:
        cand = self._ladder_pick(needs, now, st)
        if cand is None:
            if self._ladder_exhausted(needs, now, st):
                self._enter_safe(f"every brain in the ladder failed to load or is not gpud's to load "
                                 f"({self._ladder_why(needs, now, st)})", now)
                return True
            self._note(now, "brain_unavailable", (needs, for_job), job_id=for_job,
                       reason="no brain engine is available or fits; waiting")
            self._brain_blocked(now, st, self._ladder_why(needs, now, st))
            return False
        if needs == "vision" and not self.catalog[cand].vision:
            with self.store.tx():
                self.store.event(now, "brain_degraded", for_job, wanted="vision", using=cand)
        if self._make_room({cand}, now):
            return self._begin_load(cand, now, for_job)
        self._brain_blocked(now, st, f"no room for {cand}: {self._room_why}")
        return False

    def _resident_brain_for(self, needs: str, now: float, st: dict) -> str | None:
        brains = self._usable_brains()
        if not brains:
            return None
        if needs == "vision":
            vision = [m for m in brains if self.catalog[m].vision]
            if vision:
                return vision[0]
            if self._loadable(self.p.vision_brain, now, st):
                return None  # go load the vision brain
        pref = list(self.p.text_ladder) + [self.p.vision_brain]
        return min(brains, key=lambda m: pref.index(m) if m in pref else len(pref))

    def _slots_ok(self, target: set[str], pinned: set[str]) -> bool:
        """`target` fits under every engine's model-count cap beside the models gpud cannot
        unload there (someone else's, or on an engine that does not answer)."""
        for name in {e for e in (self._engine_name(m) for m in target) if e}:
            cap = self._slot_cap(name)
            if cap is None:
                continue
            fixed = [m for m, e in self._resident.items() if e == name and m in pinned and m not in target]
            if len(fixed) + sum(1 for m in target if self._engine_name(m) == name) > cap:
                return False
        return True

    def _plan_media(self, model: str, now: float, st: dict) -> tuple[set[str], str | None]:
        """Target residency for a media job, keeping a brain co-resident whenever it fits, in
        memory and in the engine's model slots beside the models gpud may not unload."""
        spec = self.catalog[model]
        if spec.exclusive or spec.evicts_brain:
            return {model}, None
        pinned = self._pinned()
        room = self.p.usable_gb - self._size(model) - sum(self._size(m) for m in pinned if m != model)
        for b in sorted(self._usable_brains(), key=lambda m: -self._size(m)):
            if self._size(b) <= room and self._slots_ok({model, b}, pinned):
                return {model, b}, b
        co = self.p.co_resident_brain
        if co and co in self.catalog and self._size(co) <= room and self._loadable(co, now, st) \
                and self._slots_ok({model, co}, pinned):
            return {model, co}, co
        return {model}, None

    def _brainless_estimate(self, job: Job, model: str) -> float:
        load = 0.0 if model in self._resident else self.catalog[model].load_s
        return (load + self.p.hard_timeout_s(job.requires, job.payload) + self.p.cancel_grace_s
                + self._brain_return_s() + self.p.tick_slack_s)

    def _cap_checkpoint(self, job: Job, away: float, est: float, now: float, st: dict) -> bool:
        """The brain has been away long enough that the next brainless job could break
        MAX_BRAINLESS_S: insert a brain checkpoint (top priority) and dispatch it now. Keyed by
        the brain-absence epoch (not a timestamp), one per absence; if that checkpoint already
        ran and the brain is still away, bring a brain back anyway rather than wait."""
        with self.store.tx():
            jid, created = self._insert_job(
                kind="brain_checkpoint", requires="brain", state="queued",
                payload={"needs": "any", "brief": {
                    "goal": "checkpoint before more brainless work",
                    "reason": f"brain away {away:.0f}s; job {job.id} ({job.kind}) could keep it away "
                              f"{est:.0f}s more (MAX_BRAINLESS_S={self.p.max_brainless_s:.0f})",
                    "next_job_id": job.id}},
                max_attempts=self.p.max_attempts, priority=self.p.continuation_priority,
                deadline=now + self.p.brain_return_sla_s, idempotency_key=f"cap:{st['brainless_epoch']}",
                created_at=now)
            if created:
                self.store.event(now, "cap_checkpoint", jid, away_s=away, next_job=job.id, est_s=est)
        checkpoint = self.store.job(jid)
        if checkpoint.state == "queued":
            return self._dispatch(checkpoint, now, st)
        return self._bring_brain("any", now, jid, st)
