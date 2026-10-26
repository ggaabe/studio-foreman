"""Red-team tests for Foreman's brain-return invariants (I1-I9).

Each test is an attack the builder's suite does not cover: a hostile or malformed input, an
engine that misbehaves in a way the Engine protocol allows, a clock that jumps, or an
interleaving of two gpud processes. Everything runs on the virtual clock against the fake
engines; nothing sleeps and nothing touches a real model server.

A FAILING test here is a finding. Its docstring names the invariant it breaks and the mechanism.
A passing test is an attack the prototype withstood (kept as a regression guard).

Harness conventions:
  * launchd_run(): any exception (not just SimulatedCrash) escaping the supervisor is treated
    as the gpud process dying. launchd waits ThrottleInterval (10 s) and starts a fresh
    Supervisor on the same DB, which runs recover(). This is what a real deployment does.
  * SkewSim: the supervisor reads a wall clock that the test can step; the engines keep
    physical (monotonic) time, which is what drives real render progress.
"""
from __future__ import annotations

import os
import unittest

from foreman.engine import Poll
from foreman.policy import CATALOG, ModelSpec, Policy
from foreman.sim import (SUCCESS, Behaviour, ScriptedBrain, Sim, SimulatedCrash, crash_mid_job, garbage, hang,
                         render_request, slow)
from foreman.store import ACTIVE, TERMINAL
from foreman.supervisor import AdmissionError, Supervisor

try:  # discovered with -s tests (tests/ on sys.path) or run from the project root
    from test_brain_return import GOAL, SimCase, calls, continuations_of, submit_render
except ImportError:  # pragma: no cover
    from tests.test_brain_return import GOAL, SimCase, calls, continuations_of, submit_render

PRIMARY, FALLBACK, VISION, VIDEO = "qwen38-flash-next", "qwen36-35b-a3b", "qwen38-27b", "ltx-2.5"
IMAGE, DEEPSEEK = "flux2-klein", "deepseek-v4.1-flash-1017"
ENGINE_METHODS = ("resident_models", "load", "poll_load", "unload", "start", "find", "poll", "cancel",
                  "terminate")
LAUNCHD_THROTTLE_S = 10.0


# ---------------------------------------------------------------- attack helpers

def make_unreachable(engine) -> None:
    """Every Engine call raises, as an HTTP adapter does while the server is down or stopped."""
    def refuse(*_a, **_k):
        raise ConnectionRefusedError(f"{engine.name}: connection refused")
    for name in ENGINE_METHODS:
        setattr(engine, name, refuse)


def make_reachable(engine) -> None:
    for name in ENGINE_METHODS:
        engine.__dict__.pop(name, None)


def tolerant_brain_check(sim) -> None:
    """Sim.brain_resident() raises if an engine is unreachable or hosts a non-catalog model;
    the test-side observer must not crash where the supervisor is the thing under test."""
    def resident():
        for e in sim.engines.values():
            try:
                models = e.resident_models()
            except Exception:
                continue
            if any(m in CATALOG and CATALOG[m].is_brain for m in models):
                return True
        return False
    sim.brain_resident = resident


def flap(engine, *models) -> None:
    """The engine drops these models as soon as they finish loading (memory-pressure policy,
    idle TTL of ~0, or a process that dies right after the load and is respawned empty)."""
    original = engine._sync

    def sync():
        original()
        for m in models:
            if m in engine._models:
                engine._models.discard(m)
                engine.calls.append((engine.clock.now(), "evict", m))
    engine._sync = sync


class SkewedClock:
    """Wall clock = physical time + skew (NTP step, RTC correction, manual date change)."""

    def __init__(self, physical):
        self.physical, self.skew = physical, 0.0

    def now(self) -> float:
        return self.physical.now() + self.skew


class SkewSim(Sim):
    """The supervisor reads the (steppable) wall clock; engines and samples keep physical time."""

    def _spawn(self) -> Supervisor:
        if not hasattr(self, "wall"):
            self.wall = SkewedClock(self.clock)
        return Supervisor(self.db_path, self.engines, self.wall, self.policy, pager=self.pages.append)


class FlakyPager(list):
    down = True

    def append(self, page):
        if self.down:
            raise ConnectionError("pager service unreachable")
        super().append(page)


class OneShotBrain(ScriptedBrain):
    """An LLM whose plan_video decision is a fixed (possibly malformed) list of requests."""

    def __init__(self, *requests, **kw):
        super().__init__(**kw)
        self.requests = list(requests)

    def __call__(self, spec, model):
        if spec.kind == "plan_video":
            self.transcript.append({"job_id": spec.job_id, "kind": spec.kind, "model": model, "brief": {}})
            return {"submit": self.requests, "note": "planned"}
        return super().__call__(spec, model)


class AdvCase(SimCase):

    def make_with(self, cls=Sim, capacity=None, policy=None, brain=None):
        self.n += 1
        sim = cls(os.path.join(self.tmp, f"adv-{self.n}.db"), capacity or self.capacity, policy, brain)
        self.addCleanup(lambda: sim.sup.close())
        return sim

    def launchd_run(self, sim, pred, max_s, dt=1.0):
        """Run like production: an exception escaping tick() or recover() kills gpud; launchd
        respawns it after its throttle interval. Returns (pred_reached, crash_reprs)."""
        crashes: list[str] = []
        end = sim.clock.now() + max_s
        dead = False
        while sim.clock.now() < end:
            if dead:
                sim.clock.advance(LAUNCHD_THROTTLE_S)
                sim._run_actions()
                try:
                    sim.sup.close()
                except Exception:
                    pass
                sim.restarts += 1
                sim.sup = sim._spawn()
                try:
                    sim.sup.recover()
                    dead = False
                except Exception as e:
                    crashes.append(repr(e))
                    continue
            try:
                sim.step(dt)
            except Exception as e:
                crashes.append(repr(e))
                dead = True
            if pred():  # the DB and the engines are the truth, whether or not this tick died
                return True, crashes
        return False, crashes

    def crash_loop_msg(self, crashes, what):
        first = crashes[0] if crashes else "no exception"
        return f"{what}; gpud crashed {len(crashes)} times on the same persisted state (first: {first})"


# =====================================================================================
# 1. The supervisor dies on untrusted input and crash-loops (I2, I4, I5: brain never returns)
# =====================================================================================

class A1CrashLoopOnBadInput(AdvCase):
    """Any exception escaping tick() kills gpud. Because the bad input is persisted (a DB row, a
    finished engine run, a resident model), recover() + the next tick hit it again: a
    deterministic crash loop. No continuation fires, the brain is not reloaded, nobody is paged."""

    def test_validator_crashes_on_missing_engine_metadata(self):
        """I2/I9. A real adapter that cannot probe a clip reports black_frame_ratio=None.
        validators.check() does float(None) -> TypeError inside _complete(); the render stays
        'running' in the DB, recover() re-adopts the 'done' run, and the next tick dies again.
        The brain was unloaded for the render (tight host) and never comes back."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", garbage(black_frame_ratio=None))
        job = submit_render(sim.sup)
        ok, crashes = self.launchd_run(sim, sim.quiet, 1800)
        self.assertTrue(ok, self.crash_loop_msg(crashes, f"never settled: job {job} is "
                                                         f"{sim.store.job(job).state}, brain={sim.brain_resident()}"))
        self.assertEqual(sim.store.job(job).outcome["reason"], "validator")
        self.assert_invariants(sim)

    def test_brain_payload_with_non_numeric_duration_crash_loops_at_validation(self):
        """I1/I2. The LLM writes duration_s='8 seconds'. Admission checks only goal and
        expected_s, so the render is admitted, the brain is unloaded, the render finishes, and
        validators.check() raises ValueError on float('8 seconds') on every tick forever."""
        req = render_request(GOAL)
        req["payload"]["duration_s"] = "8 seconds"
        sim = self.make_with(brain=OneShotBrain(req, retry_failures=False))
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        ok, crashes = self.launchd_run(sim, sim.quiet, 1800)
        renders = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertTrue(ok, self.crash_loop_msg(crashes, f"never settled: renders={renders}, "
                                                         f"brain={sim.brain_resident()}"))
        self.assert_invariants(sim)

    MALFORMED_DECISIONS = {
        "expected_s is prose": {"submit": [render_request(GOAL, expected_s="about five minutes")]},
        "submit is an object, not a list": {"submit": render_request(GOAL)},
        "request is a bare string": {"submit": ["render a fox in the snow"]},
        "payload is a string": {"submit": [dict(render_request(GOAL), payload="a fox in the snow")]},
        "retry_of is prose": {"submit": [dict(render_request(GOAL), retry_of="job 4")]},
        "max_attempts is prose": {"submit": [dict(render_request(GOAL), max_attempts="three")]},
        "priority is prose": {"submit": [dict(render_request(GOAL), priority="high")]},
        "requires is a list": {"submit": [dict(render_request(GOAL), requires=[VIDEO])]},
    }

    def test_malformed_brain_decision_crash_loops_admission(self):
        """I1. LLM output is untrusted, but _apply_decision() only catches AdmissionError. Any
        other exception from _admit() (float('about five minutes') in hard_timeout_s, int('job 4'),
        int('three'), dict('a fox'), str.get, an unhashable 'requires', ...) escapes, rolls back
        the brain job's terminal record and kills tick(). The finished plan_video run is
        re-adopted after every restart, so the same decision is re-applied forever: the brain is
        loaded but the system is frozen (later work never runs, nobody is paged)."""
        for name, decision in self.MALFORMED_DECISIONS.items():
            with self.subTest(decision=name):
                class Raw(ScriptedBrain):
                    def __call__(self, spec, model):
                        return decision if spec.kind == "plan_video" else super().__call__(spec, model)

                sim = self.make_with(brain=Raw(retry_failures=False))
                plan = sim.sup.submit("plan_video", payload={"goal": GOAL})
                later = sim.sup.submit("triage_inbox", payload={"goal": "unrelated work queued behind the plan"})
                ok, crashes = self.launchd_run(sim, sim.quiet, 900)
                self.assertTrue(ok, self.crash_loop_msg(
                    crashes, f"never settled: plan is {sim.store.job(plan).state}, "
                             f"later job is {sim.store.job(later).state}"))
                self.assertTrue(sim.store.job(plan).outcome["rejected"], "the bad request should be rejected")

    def test_engine_cancel_that_raises_wedges_the_timeout_path(self):
        """I3. A real HTTP adapter raises on cancel (e.g. 404 for a run the server already
        reaped). _begin_cancel() has committed state='cancelling', cancel_stage='cancel'; every
        tick and every restart re-enters _advance_cancel() -> eng.cancel() -> raises. The lease
        is dead but the job never reaches unload or a terminal record, no continuation fires,
        and the brain (unloaded for the render) never returns."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", hang(0.4))

        def cancel(handle):
            raise LookupError(f"404: no such run {handle}")
        sim.mlx.cancel = cancel
        job = submit_render(sim.sup)
        ok, crashes = self.launchd_run(sim, sim.quiet, 1800)
        self.assertTrue(ok, self.crash_loop_msg(crashes, f"never settled: job {job} is {sim.store.job(job).state}, "
                                                         f"brain={sim.brain_resident()}"))
        self.assert_invariants(sim)

    def test_unknown_model_resident_on_engine_crash_loops_make_room(self):
        """I2/I8. A model that is not in the catalog shows up on mlx-serve (another client used
        its on-demand loading). _observe() records model_appeared, the planner sizes it at 0 GB,
        and _make_room() picks it as the first victim: _unload() -> _engine_for() ->
        catalog[model] -> KeyError. Crash loop; the render never starts, nothing is paged."""
        foreign = "llama-3.3-70b-4bit"
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        tolerant_brain_check(sim)
        sim.mlx.catalog = {**CATALOG, foreign: ModelSpec(foreign, "mlx-serve", 40, 20, "llm")}
        self.settle(sim)                        # primary brain home
        sim.mlx._models.add(foreign)            # appears behind gpud's back
        job = submit_render(sim.sup)
        ok, crashes = self.launchd_run(sim, sim.quiet, 1800)
        self.assertTrue(ok, self.crash_loop_msg(crashes, f"never settled: job {job} is {sim.store.job(job).state}"))
        self.assertEqual(sim.host.oom, [], "planner sized the foreign model at 0 GB and overcommitted")

    def test_job_for_an_unconfigured_engine_unloads_the_brain_then_crash_loops(self):
        """I1/I2. gpud runs without an oMLX adapter (not installed / config drift). Admission
        accepts a DeepSeek job anyway; _schedule() unloads the brain for the exclusive model,
        then _begin_load() -> self.engines['omlx'] -> KeyError. Every restart repeats it: the
        brain is gone and never returns."""
        sim = self.make_with(capacity=256.0, brain=ScriptedBrain(retry_failures=False))
        del sim.engines["omlx"]
        sim.host.engines.remove(sim.omlx)
        sim.restart()
        self.settle(sim)
        try:
            job = sim.sup.submit("long_reasoning", DEEPSEEK, {"goal": "audit the storyboard", "expected_s": 300},
                                 {"kind": "review_answer"}, {"kind": "recover_answer"})
        except AdmissionError:
            return  # acceptable: refused up front
        ok, crashes = self.launchd_run(sim, sim.quiet, 1800)
        self.assertTrue(ok, self.crash_loop_msg(crashes, f"never settled: job {job} is {sim.store.job(job).state}, "
                                                         f"brain resident={sim.brain_resident()}"))

    def test_stopped_omlx_service_blocks_the_brain_on_healthy_mlx_serve(self):
        """I2/I5. The README says unloading DeepSeek 'may mean stopping the oMLX service'. A
        stopped service refuses connections; _observe() calls resident_models() on every engine
        first thing in recover() and tick(), so gpud dies before it can load the brain on the
        perfectly healthy mlx-serve. A brain job queued for mlx-serve never runs."""
        sim = self.make_with()
        tolerant_brain_check(sim)
        make_unreachable(sim.omlx)
        job = sim.sup.submit("triage_inbox", payload={"goal": "answer the user"})
        ok, crashes = self.launchd_run(
            sim, lambda: sim.store.job(job).state == "succeeded" and sim.brain_resident(), 1200)
        self.assertTrue(ok, self.crash_loop_msg(crashes, f"brain job is {sim.store.job(job).state}, "
                                                         f"brain on mlx-serve={sim.brain_resident()}"))

    def test_permanently_dead_engine_pages_a_human(self):
        """I3/I5. mlx-serve dies mid-render and launchd gives up on it. Every poll raises, so gpud
        crash-loops instead of treating the engine as lost; the SAFE/page path is never reached.
        Nobody is told that the brain (which lives on that engine) cannot come back."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.3)
        tolerant_brain_check(sim)
        make_unreachable(sim.mlx)
        ok, crashes = self.launchd_run(sim, lambda: bool(sim.pages) or sim.store.state()["safe"], 3600)
        self.assertTrue(ok, self.crash_loop_msg(crashes, "an hour later no human has been paged"))


# =====================================================================================
# 2. Priority inversion: the baton is not actually top priority (I2, I7)
# =====================================================================================

class A2PriorityInversion(AdvCase):

    def test_brain_assigned_priority_starves_continuations_and_deadlocks_the_cap(self):
        """I2/I7. Continuations get priority 100, but _admit() takes any caller priority as is,
        including one the LLM puts in its decision. With media jobs at priority 1000:
          * a finished render's review waits behind the next render (SLA missed), and
          * once the brain has been away long enough, _cap_checkpoint() inserts a priority-100
            brain_checkpoint, but next_queued() keeps returning the priority-1000 render, the
            checkpoint's idempotency key already exists, and _schedule() returns: deadlock. The
            brain never comes back and nothing is paged."""
        reqs = [dict(render_request(f"{GOAL} (shot {i})"), priority=1000) for i in range(6)]
        sim = self.make_with(brain=OneShotBrain(*reqs, retry_failures=False))
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        settled = sim.run_until_quiet(4 * 3600)
        renders = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertTrue(settled, f"never settled: renders={[(j.id, j.state) for j in renders]}, "
                                 f"queued={[(j.kind, j.priority) for j in sim.store.jobs('queued')]}, "
                                 f"brain resident={sim.brain_resident()}")
        for r in renders:
            self.assert_returned_within_sla(sim, continuations_of(sim.store, r.id)[0])
        self.assertLessEqual(sim.max_brain_away(), sim.policy.max_brainless_s)


# =====================================================================================
# 3. Flapping residency: "loaded" is treated as success even if the model is gone at once
# =====================================================================================

class A3Flapping(AdvCase):

    def test_flapping_primary_brain_is_reloaded_forever_and_work_starves(self):
        """I5/I8. The primary loads, then the engine drops it before gpud can use it.
        _check_load() sees 'ready' and clears the ladder failure count, so the ladder never steps
        down to the (healthy) fallback brain: 100 GB reloads every ~60 s forever, the queued
        brain job never runs, and nobody is paged."""
        sim = self.make_with()
        flap(sim.mlx, PRIMARY)
        job = sim.sup.submit("triage_inbox", payload={"goal": "answer the user"})
        sim.run_until(lambda: sim.store.job(job).state in TERMINAL, 2 * 3600)
        loads = len(calls(sim.mlx, "load", PRIMARY))
        self.assertEqual(sim.store.job(job).state, "succeeded",
                         f"brain job still {sim.store.job(job).state} after 2 h; primary loaded {loads} times")
        self.assertLessEqual(loads, 2 * sim.policy.load_attempts_per_model)

    def test_flapping_media_model_keeps_job_queued_forever(self):
        """I2/I6. The video model 'loads' but is gone before start(). _schedule() just loads it
        again. The job never starts, so it never gets a lease, never fails, and its on_failure
        never fires; MAX_BRAINLESS_S only makes gpud bounce the brain back for empty
        brain_checkpoint jobs (14 of them in 4 h). After 4 h the render is still queued, the
        150 GB model has been loaded 329 times, and nobody has been paged."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        flap(sim.mlx, VIDEO)
        job = submit_render(sim.sup)
        sim.run_until(lambda: sim.store.job(job).state in TERMINAL, 4 * 3600)
        loads = len(calls(sim.mlx, "load", VIDEO))
        self.assertIn(sim.store.job(job).state, TERMINAL,
                      f"render still {sim.store.job(job).state} after 4 h; {VIDEO} loaded {loads} times, "
                      f"{len(sim.store.jobs('awaiting_human'))} humans paged")
        self.assertEqual(len(continuations_of(sim.store, job)), 1)
        self.assertLessEqual(loads, 2 * sim.policy.load_attempts_per_model)


# =====================================================================================
# 4. Poison protection depends on the LLM's cooperation (I6)
# =====================================================================================

class A4PoisonBypass(AdvCase):

    def test_brain_can_raise_max_attempts_past_policy(self):
        """I6. _admit() takes max_attempts from the request, including one the LLM writes into
        its decision. A brain job that always raises is retried on the same row 50 times
        (depth never increases, so the depth limit does not help) instead of max_attempts=3."""
        class Generous(ScriptedBrain):
            def __call__(self, spec, model):
                if spec.kind == "plan_video":
                    return {"submit": [{"kind": "summarize_notes", "payload": {"goal": "summarize"},
                                        "max_attempts": 50}]}
                if spec.kind == "summarize_notes":
                    raise RuntimeError("poisoned prompt")
                return super().__call__(spec, model)

        sim = self.make_with(brain=Generous())
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim, max_s=4 * 3600)
        job = next(j for j in sim.store.jobs() if j.kind == "summarize_notes")
        self.assertEqual(job.state, "quarantined")
        self.assertLessEqual(job.attempt, sim.policy.max_attempts,
                             f"poisoned brain job ran {job.attempt} times before quarantine")

    def test_supervisor_crashes_are_charged_to_the_jobs_poison_budget(self):
        """I4/I6. gpud dies three times right after recording 'starting' (before the engine
        ever saw the run). Each recover() marks the brain job failed(interrupted) and
        _finish_brain_job() charges an attempt, so a perfectly healthy job is quarantined and a
        human paged for gpud's own instability. The same accounting applies to brainless jobs
        retried by the brain."""
        sim = self.make_with()
        hits = [0]

        def hook(point):
            if point == "after_starting_recorded":
                hits[0] += 1
                if hits[0] <= 3:
                    raise SimulatedCrash(point)
        original_restart = sim.restart

        def restart(downtime_s=0.0):
            original_restart(downtime_s)
            sim.arm(hook)
        sim.restart = restart
        sim.arm(hook)
        job = sim.sup.submit("triage_inbox", payload={"goal": "answer the user"})
        self.settle(sim)
        self.assertGreaterEqual(hits[0], 3, "gpud really died three times before the engine saw the run")
        j = sim.store.job(job)
        self.assertEqual(j.state, "succeeded", f"healthy job {j.state} at attempt {j.attempt}: "
                                               f"{j.outcome.get('reason')}; pages={sim.pages}")

    def test_resubmission_without_retry_of_evades_max_attempts(self):
        """I6. Attempts are counted per task_key, which only carries over when the LLM passes
        retry_of. A brain that resubmits the same failing render as a fresh job is bounded only
        by max_continuation_depth: fan-out 1 gives 16 failed renders (~76 min) before the first
        page; with fan-out 2 the depth grows logarithmically (measured: 102 failed + 102 queued
        renders after 8 h, depth 15 of 32, no page)."""
        for fanout in (1, 2):
            with self.subTest(fanout=fanout):
                class Storm(ScriptedBrain):
                    def __call__(self, spec, model):
                        if spec.kind == "recover_render":
                            goal = spec.payload["brief"]["goal"]
                            return {"submit": [render_request(goal) for _ in range(fanout)], "note": "again"}
                        return super().__call__(spec, model)

                sim = self.make_with(brain=Storm())
                sim.mlx.script_jobs("render_video", *[crash_mid_job(0.5)] * 400)
                sim.sup.submit("plan_video", payload={"goal": GOAL})
                bound = 2 * sim.policy.max_attempts

                def failed():
                    return sum(1 for j in sim.store.jobs("failed", "quarantined") if j.kind == "render_video")
                sim.run_until(lambda: bool(sim.pages) or failed() > bound, 4 * 3600)
                self.assertLessEqual(failed(), bound,
                                     f"{failed()} renders of the same goal failed and "
                                     f"{'a human was' if sim.pages else 'no human has been'} paged")


# =====================================================================================
# 5. Wall-clock jumps (I3, I7): leases are measured on clock.now(), not a monotonic clock
# =====================================================================================

class A5ClockJumps(AdvCase):

    def test_forward_wall_clock_step_kills_a_healthy_render(self):
        """I3. 'Slow but progressing jobs are not killed.' A +300 s wall-clock step (NTP
        correction, RTC fix, manual date change) makes now - last_progress_at >= 120 s on the
        next tick without a new denoise step, so a render that is physically progressing
        normally is cancelled and recorded failed(timeout, no_progress). (A larger step also
        trips the hard deadline.)"""
        sim = self.make_with(cls=SkewSim)
        job = submit_render(sim.sup)
        self.running(sim, job, 0.5)
        sim.wall.skew += 300
        self.settle(sim)
        j = sim.store.job(job)
        self.assertEqual(j.state, "succeeded", f"healthy render was killed: {j.outcome.get('detail')}")

    def test_backward_wall_clock_step_lets_a_hang_outlive_the_cap(self):
        """I3/I7. A -3600 s wall-clock step right after a render hangs makes
        now - last_progress_at negative for an hour: the dead-man switch is blind for that long
        and the brain stays away far past MAX_BRAINLESS_S (measured in physical time)."""
        sim = self.make_with(cls=SkewSim, brain=ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", hang(0.4))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.4)
        sim.wall.skew -= 3600
        self.settle(sim, max_s=4 * 3600)
        self.assertEqual(sim.store.job(job).outcome["reason"], "timeout")
        self.assertLessEqual(sim.max_brain_away(), sim.policy.max_brainless_s)


# =====================================================================================
# 6. Two gpud processes on one DB (I4): _grant_lease() has no compare-and-set
# =====================================================================================

class A6TwoSupervisors(AdvCase):

    def test_second_supervisor_recovering_mid_start_lets_the_first_resurrect_a_failed_job(self):
        """I2/I4. gpud A records 'starting' and is about to call engine.start(). A second gpud B
        (launchd/watchdog started it; there is no flock) runs recover(): no handle and find()
        knows nothing, so B marks the job failed(interrupted), fires on_failure, and starts the
        recover brain job on the co-resident vision brain. A then calls start() and
        _grant_lease(), which sets state='running' unconditionally: the failed job comes back to
        life, runs to success, and is recorded a second time. The brain was told 'interrupted'
        and re-renders; the successful clip's review never happens."""
        sim = self.make_with(capacity=256.0)
        job = submit_render(sim.sup)
        done = []

        def hook(point):
            if point == "after_starting_recorded" and not done:
                done.append(True)
                b = Supervisor(sim.db_path, sim.engines, sim.clock, sim.policy, pager=sim.pages.append)
                b.recover()
                b.tick()
                b.close()
        sim.arm(hook)
        self.settle(sim)
        self.assertTrue(done)
        j = sim.store.job(job)
        finishes = [e["data"]["state"] for e in sim.store.events("job_finished", job)]
        cont = sim.store.continuation(job)
        self.assertEqual(len(finishes), 1, f"terminal outcome recorded {len(finishes)} times: {finishes}")
        self.assertEqual(cont["fired_outcome"], "success" if j.state == "succeeded" else "failure",
                         f"job ended {j.state} but its {cont['fired_outcome']} continuation fired")

    def test_two_supervisors_ticking_alternately_do_not_duplicate_anything(self):
        """Passing attack (I4): two live gpuds interleaved at tick granularity on one DB."""
        sim = self.make_with(policy=Policy(max_attempts=6))
        sim.mlx.script_jobs("render_video", hang(0.4), SUCCESS)
        b = Supervisor(sim.db_path, sim.engines, sim.clock, sim.policy, pager=sim.pages.append)
        self.addCleanup(b.close)
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        for _ in range(4 * 3600):
            sim.step()
            b.tick()
            if sim.quiet():
                break
        self.assertTrue(sim.quiet())
        self.assert_invariants(sim)
        self.assertEqual([j.state for j in sim.store.jobs() if j.kind == "review_render"], ["succeeded"])


# =====================================================================================
# 7. Reserved idempotency keys are not reserved (I1, I2)
# =====================================================================================

class A7IdempotencyNamespace(AdvCase):

    def test_client_key_hijacks_the_continuation(self):
        """I2/I9. Internal keys ('cont:<id>', 'safe:<epoch>', 'review:<id>', 'cap:<t>') share one
        UNIQUE namespace with client idempotency keys and admission does not reserve them. A
        client key 'cont:2' makes job 2's continuation resolve to the client's job: the baton
        is 'fired' but no brain job ever receives the resume brief."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.sup.submit("triage_inbox", payload={"goal": "client work"}, idempotency_key="cont:2")
        job = submit_render(sim.sup)
        self.assertEqual(job, 2)
        self.settle(sim)
        child = sim.store.job(sim.store.continuation(job)["fired_job_id"])
        self.assertEqual(child.parent_id, job, f"continuation of {job} resolved to unrelated {child}")
        self.assertEqual(child.payload.get("brief", {}).get("source_job_id"), job)

    def test_client_key_silences_the_safe_mode_page(self):
        """I5. With a client job holding key 'safe:0', _enter_safe() -> _human_review() finds the
        existing row (created=False): no safe_mode job, no page. gpud sits in SAFE mode, the
        brain never comes back, and no human is ever told."""
        sim = self.make_with()
        sim.mlx.script_loads(PRIMARY, "fail", "fail")
        sim.mlx.script_loads(FALLBACK, "fail", "fail")
        sim.sup.submit("triage_inbox", payload={"goal": "client work"}, idempotency_key="safe:0")
        self.settle(sim)
        self.assertTrue(sim.store.state()["safe"])
        self.assertTrue(sim.pages, "gpud entered SAFE mode without paging anyone")


# =====================================================================================
# 8. Stuck load whose abort is refused (I3 / memory planning)
# =====================================================================================

class A8StuckLoad(AdvCase):

    def test_refused_abort_of_a_hung_load_is_ignored_and_memory_is_overcommitted(self):
        """Planner invariant (assert_invariants: no OOM). When a load hangs, _check_load() calls
        eng.unload(model) to abort it and ignores the return value (the _unload() path escalates
        to terminate; this one does not). If the engine cannot abort an in-flight load, the
        100 GB stuck load stays in unified memory, gpud believes it is gone, and the next video
        load overcommits the host."""
        sim = self.make_with()
        eng = sim.mlx
        original_unload = eng.unload

        def unload(model):
            eng._sync()
            if any(op.model == model and op.state == "loading" for op in eng._loads.values()):
                eng.calls.append((eng.clock.now(), "unload_refused", model))
                return False
            return original_unload(model)
        eng.unload = unload
        eng.script_loads(PRIMARY, "hang", "hang")
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        sim.run_until_quiet(4 * 3600)
        self.assertTrue(calls(eng, "unload_refused"))
        self.assertEqual(sim.host.oom, [], "a video load ran into a stuck 100 GB brain load gpud forgot about")


# =====================================================================================
# 9. MAX_BRAINLESS_S under brain-load trouble (I7)
# =====================================================================================

class A9CapUnderLoadFailure(AdvCase):

    def test_cap_overshoot_exceeds_the_documented_one_load_timeout(self):
        """I7. README: the cap can be exceeded 'by up to that [load] timeout'. The budget
        reserves one brain load (60 s), but the ladder allows two primary attempts of up to
        3 x load_s each plus a fallback load. A render sized exactly to the budget that runs to
        its hard timeout, followed by two hung primary loads, keeps the brain away 2085 s:
        more than MAX_BRAINLESS_S + one load timeout (1980 s)."""
        sim = self.make_with()
        sim.mlx.script_jobs("render_video", slow(3.0))
        sim.mlx.script_loads(PRIMARY, "ok", "hang", "hang")
        self.settle(sim)
        t0 = sim.clock.now()
        submit_render(sim.sup, expected_s=666)   # hard timeout 1665 s == the brainless budget
        self.settle(sim, max_s=4 * 3600)
        documented = sim.policy.max_brainless_s + CATALOG[PRIMARY].load_s * sim.policy.load_timeout_factor
        self.assertLessEqual(sim.max_brain_away(after=t0), documented)


# =====================================================================================
# 10. Progress granularity (I3): one global progress_timeout_s for every job
# =====================================================================================

class A10CoarseProgress(AdvCase):

    def test_few_step_model_with_long_steps_is_killed_while_healthy(self):
        """I3. progress_timeout_s is a single 120 s constant, not derived from the job. A
        few-step distilled model (FLUX.2-klein-style, 4 steps) on a 600 s job reports progress
        every 150 s, so a perfectly healthy run is cancelled at 120 s as 'no progress'."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.STEPS = 4
        job = sim.sup.submit("generate_image", IMAGE, {"goal": "poster key art", "expected_s": 600},
                             {"kind": "review_image", "needs": "vision"}, {"kind": "recover_image"})
        self.settle(sim)
        j = sim.store.job(job)
        self.assertEqual(j.state, "succeeded", f"healthy run killed: {j.outcome.get('detail')}")


# =====================================================================================
# 11. Artifacts the brain never hears about (I9)
# =====================================================================================

class A11ArtifactsInBrief(AdvCase):

    def test_engine_failure_with_an_artifact_drops_the_path(self):
        """I9. The engine wrote the clip, then failed in post-processing and reported
        Poll('failed', artifact={...}). _check_running() builds failed('crash', ...) with
        artifact=None, so the brief tells the brain there is nothing to look at."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        original_poll = sim.mlx.poll

        def poll(handle):
            p = original_poll(handle)
            if p.status == "done" and p.artifact and p.artifact.get("path"):
                return Poll("failed", p.progress, p.artifact, "muxer exited 1 after writing the clip", p.log_tail)
            return p
        sim.mlx.poll = poll
        job = submit_render(sim.sup)
        self.settle(sim)
        brief = continuations_of(sim.store, job)[0].payload["brief"]
        self.assertEqual(brief["outcome"]["reason"], "crash")
        self.assertTrue(brief["artifacts"], "the clip exists on disk but the brief lists no artifacts")

    def test_render_that_finishes_during_cancel_grace_is_discarded(self):
        """I9. The render crosses its hard deadline at 99.9% and finishes 1 s later, before the
        graceful cancel lands. _advance_cancel() sees 'done' but records the pending timeout and
        throws the valid artifact away; the brief has no artifact path and no validator results,
        so the brain re-renders from scratch."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        hard = sim.policy.hard_timeout_s(VIDEO, {"expected_s": 300})
        sim.mlx.script_jobs("render_video", Behaviour("slow", factor=(hard + 1) / 300))
        job = submit_render(sim.sup)
        self.settle(sim)
        brief = continuations_of(sim.store, job)[0].payload["brief"]
        self.assertEqual(brief["outcome"]["reason"], "timeout")
        self.assertTrue(brief["artifacts"], "a finished, valid clip was discarded from the brief")


# =====================================================================================
# Passing attacks (regression guards for what the prototype withstood)
# =====================================================================================

class P1ZombieLease(AdvCase):

    def test_zombie_render_that_resumes_progress_after_lease_expiry_cannot_renew(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", hang(0.4))
        job = submit_render(sim.sup)
        self.assertTrue(sim.run_until(lambda: sim.store.job(job).state == "cancelling", 3600))
        before = sim.store.lease(job)
        run = sim.mlx._runs[sim.store.job(job).handle]
        run.behaviour = Behaviour("success", ignore_cancel=True)  # wakes up, reports progress, ignores cancel
        self.settle(sim)
        lease, j = sim.store.lease(job), sim.store.job(job)
        self.assertGreater(run.progress, before["last_progress"], "the zombie really did report progress")
        self.assertEqual((j.state, j.outcome["reason"]), ("failed", "timeout"))
        self.assertEqual(lease["state"], "expired")
        self.assertEqual((lease["last_progress"], lease["last_progress_at"]),
                         (before["last_progress"], before["last_progress_at"]))
        self.assertEqual(len(continuations_of(sim.store, job)), 1)
        self.assert_invariants(sim)


class P2SafeMode(AdvCase):

    def test_safe_mode_survives_restarts_then_clear_brings_the_brain_back(self):
        sim = self.make_with()
        sim.mlx.script_loads(PRIMARY, "fail", "fail")
        sim.mlx.script_loads(FALLBACK, "fail", "fail")
        plan = sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim)
        self.assertTrue(sim.store.state()["safe"])
        n_loads = len(calls(sim.mlx, "load"))
        for downtime in (0, 600, 3 * 3600):
            sim.restart(downtime_s=downtime)
            sim.run_until(lambda: False, 1800, dt=10)
            self.assertTrue(sim.store.state()["safe"], "SAFE mode must persist across gpud restarts")
        self.assertEqual(len(calls(sim.mlx, "load")), n_loads, "no loads while SAFE, even after restarts")
        self.assertEqual([p["kind"] for p in sim.pages], ["safe_mode"], "paged once, not on every boot")
        sim.sup.clear_safe()
        self.settle(sim)
        self.assertEqual(sim.store.job(plan).state, "succeeded")
        self.assertTrue(sim.brain_resident())
        self.assert_invariants(sim)

    def test_clearing_safe_without_fixing_the_wedge_re_enters_safe_and_pages_again(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.terminate_fails = True
        sim.mlx.script_jobs("render_video", hang(0.4, wedged=True))
        job = submit_render(sim.sup)
        self.settle(sim)
        sim.sup.clear_safe()  # the human clears SAFE but has not fixed the engine
        self.settle(sim)
        self.assertTrue(sim.store.state()["safe"])
        self.assertEqual([p["kind"] for p in sim.pages], ["safe_mode", "safe_mode"])
        self.assertEqual(len(calls(sim.mlx, "terminate")), 2, "one SIGTERM per epoch, no loop")
        sim.mlx.operator_force_restart()
        sim.sup.clear_safe()
        self.settle(sim)
        self.assertEqual(continuations_of(sim.store, job)[0].state, "succeeded")
        self.assert_invariants(sim)


class P3CrashInsideEngineCalls(AdvCase):
    """gpud dies after the engine performed a call but before the DB recorded it."""

    def arm_engine(self, sim, name, when):
        original, armed = getattr(sim.mlx, name), [True]

        def wrapped(*args):
            result = original(*args)
            if armed[0] and when(args, result):
                armed[0] = False
                raise SimulatedCrash(f"died right after {name}{args}")
            return result
        setattr(sim.mlx, name, wrapped)
        return armed

    def test_crash_right_after_engine_accepted_a_brain_load(self):
        sim = self.make_with()
        armed = self.arm_engine(sim, "load", lambda a, r: a[0] == PRIMARY)
        plan = sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim)
        self.assertFalse(armed[0])
        self.assertEqual(sim.store.job(plan).state, "succeeded")
        self.assertEqual(len(calls(sim.mlx, "load", PRIMARY)), 2, "boot load reused, one reload after the render")
        self.assert_invariants(sim)

    def test_crash_right_after_a_successful_cleanup_unload(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", hang(0.4))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.4)
        armed = self.arm_engine(sim, "unload", lambda a, r: a[0] == VIDEO and r)
        self.settle(sim)
        self.assertFalse(armed[0])
        self.assertEqual(sim.store.job(job).outcome["reason"], "timeout")
        self.assert_invariants(sim)

    def test_crash_right_after_a_successful_terminate(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", hang(0.4, wedged=True))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.4)
        armed = self.arm_engine(sim, "terminate", lambda a, r: r)
        self.settle(sim)
        self.assertFalse(armed[0])
        self.assertEqual(len(calls(sim.mlx, "terminate")), 1)
        self.assertEqual(sim.store.job(job).outcome["reason"], "timeout")
        self.assert_invariants(sim)

    def test_crash_when_the_engine_first_reports_done(self):
        sim = self.make_with()
        job = submit_render(sim.sup)
        armed = self.arm_engine(sim, "poll", lambda a, r: r.status == "done" and bool(r.artifact.get("path")))
        self.settle(sim)
        self.assertFalse(armed[0])
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assertEqual(sim.store.continuation(job)["fired_outcome"], "success")
        self.assertEqual(len(calls(sim.mlx, "start", f"{job}.1")), 1)
        self.assert_invariants(sim)


class P4Assorted(AdvCase):

    def test_many_brainless_jobs_ending_in_the_same_tick(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        jobs = [submit_render(sim.sup) for _ in range(5)]
        for j in jobs:
            self.assertTrue(sim.sup.cancel_job(j, "batch withdrawn"))
        t0 = sim.clock.now()
        self.settle(sim)
        for j in jobs:
            self.assertEqual(len(continuations_of(sim.store, j)), 1)
        first_brain = sim.store.events("brain_returned")[0]["at"]
        self.assertLessEqual(first_brain - t0, sim.policy.brain_return_sla_s + CATALOG[PRIMARY].load_s)
        self.assert_invariants(sim)

    def test_render_just_under_the_hard_timeout_is_not_killed(self):
        sim = self.make_with()
        sim.mlx.script_jobs("render_video", slow(2.4))  # 720 s against a 750 s hard timeout
        job = submit_render(sim.sup)
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assertEqual(calls(sim.mlx, "cancel"), [])

    def test_downtime_past_the_hard_deadline_kills_once(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", slow(5.0))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.05)
        sim.restart(downtime_s=1000)
        self.settle(sim)
        j = sim.store.job(job)
        self.assertEqual((j.outcome["reason"], j.outcome["timeout"]), ("timeout", "hard"))
        self.assertEqual(len(continuations_of(sim.store, job)), 1)
        self.assert_invariants(sim)

    # NOTE: the red team originally asserted that this outage crashes gpud; the fix made gpud ride
    # it out, so the assertion was inverted to require no crash. Original reasoning kept below.
    # `assertGreater(len(crashes), 0, "the outage really did kill gpud")` required an engine API
    # outage to kill gpud. That is the behavior this suite's own root-cause finding (A1, design
    # concern 1) identifies as the defect: an adapter exception must be recorded as an
    # engine-unreachable observation, not end the process. The fixed supervisor rides out the
    # 30 s outage without dying, so this precondition can no longer hold without reintroducing
    # the bug. Every other assertion here (settles, attempt 1, start() called once, invariants)
    # is kept, unchanged, in test_hardening.H1NoCrashOnOutages
    # .test_transient_engine_outage_is_ridden_out_without_crashing, which also asserts crashes == [].
    def test_transient_engine_outage_mid_render_recovers_without_rerun(self):
        sim = self.make_with()
        job = submit_render(sim.sup)
        self.running(sim, job, 0.3)
        tolerant_brain_check(sim)
        make_unreachable(sim.mlx)
        sim.at(sim.clock.now() + 30, lambda: make_reachable(sim.mlx))
        ok, crashes = self.launchd_run(sim, sim.quiet, 3600)
        self.assertTrue(ok)
        self.assertEqual(crashes, [], "an engine outage must not kill gpud")
        self.assertEqual((sim.store.job(job).state, sim.store.job(job).attempt), ("succeeded", 1))
        self.assertEqual(len(calls(sim.mlx, "start", f"{job}.1")), 1)
        self.assert_invariants(sim)

    # NOTE: the red team originally asserted that a pager outage crashes gpud; the fix retries pages
    # with backoff instead, so the assertion was inverted to require no crash.
    # `assertGreater(len(crashes), 0, "the pager outage really did crash gpud")` required a pager
    # outage to crash gpud. This suite's own design concern says the opposite: "Paging should
    # catch errors, retry with backoff, and never take the supervisor down." The fixed
    # supervisor keeps the page in its outbox and retries it with backoff, so the precondition
    # can no longer hold. Every other assertion here (the brain returns during the outage, the
    # job is quarantined, exactly one human_review page once the pager is back, invariants) is
    # kept, unchanged, in test_hardening.H1NoCrashOnOutages
    # .test_pager_outage_is_retried_with_backoff_without_crashing, which also asserts crashes == [].
    def test_pager_outage_does_not_stop_the_brain_returning(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.pages = FlakyPager()
        sim.restart()
        sim.mlx.script_jobs("render_video", crash_mid_job(0.5))
        job = submit_render(sim.sup, max_attempts=1)

        def settled_but_unpaged():
            st = sim.store
            return (not st.jobs("queued", *ACTIVE) and not st.state()["loading"]
                    and not st.unfired_continuations() and sim.brain_resident())
        ok, crashes = self.launchd_run(sim, settled_but_unpaged, 3600)
        self.assertTrue(ok)
        self.assertEqual(sim.store.job(job).state, "quarantined")
        self.assertEqual(crashes, [], "a pager outage must not crash gpud")
        sim.pages.down = False
        ok, _ = self.launchd_run(sim, sim.quiet, 120)
        self.assertTrue(ok)
        self.assertEqual([p["kind"] for p in sim.pages], ["human_review"])
        self.assert_invariants(sim)


if __name__ == "__main__":
    unittest.main()
