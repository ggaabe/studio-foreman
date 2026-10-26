"""Builder's regression tests for the hardening done in answer to the red team
(tests/test_adversarial.py). Each class guards one mechanism the fix introduced:

  H1  no gpud death on engine or pager outages (siblings of two P4 tests whose "gpud really did
      crash" precondition no longer holds; see the notes on those tests)
  H2  degraded SAFE: a wedged engine does not stop the brain coming back on a healthy one
  H3  single instance (flock standby) and leader-epoch fencing
  H4  supervisor time follows a monotonic source and survives restarts across wall-clock steps
  H5  poison and admission limits the LLM cannot talk its way around
  H6  job-level exception boundary: a gpud bug fails one job, and its baton still fires
  H7  an engine that is lost and comes back: page once, fail what depended on it, clean up
  H8  models the catalog does not know, schema migration of an old database file
  H9  seeded chaos: random adapter exceptions and garbage, malformed LLM decisions and random
      gpud kills never let an exception escape tick(), and every baton still fires

Everything runs on the virtual clock against the fake engines.
"""
from __future__ import annotations

import os
import random
import sqlite3
import types
import unittest

from foreman import validators
from foreman.policy import CATALOG, ModelSpec, Policy
from foreman.sim import (SUCCESS, FakeEngine, ScriptedBrain, Sim, SimulatedCrash, crash_mid_job, garbage, hang,
                         render_request, slow)
from foreman.store import TERMINAL, Store
from foreman.supervisor import Supervisor

try:  # discovered with -s tests (tests/ on sys.path) or run from the project root
    from test_adversarial import (AdvCase, FlakyPager, OneShotBrain, SkewedClock, SkewSim, make_reachable,
                                  make_unreachable, tolerant_brain_check)
    from test_brain_return import GOAL, calls, continuations_of, submit_render
except ImportError:  # pragma: no cover
    from tests.test_adversarial import (AdvCase, FlakyPager, OneShotBrain, SkewedClock, SkewSim, make_reachable,
                                        make_unreachable, tolerant_brain_check)
    from tests.test_brain_return import GOAL, calls, continuations_of, submit_render

PRIMARY, FALLBACK, VIDEO, DEEPSEEK = "qwen38-flash-next", "qwen36-35b-a3b", "ltx-2.5", "deepseek-v4.1-flash-1017"


class H1NoCrashOnOutages(AdvCase):

    def test_transient_engine_outage_is_ridden_out_without_crashing(self):
        """Sibling of P4 test_transient_engine_outage_mid_render_recovers_without_rerun, with all
        of its assertions except the 'gpud really did crash' precondition, plus crashes == []."""
        sim = self.make_with()
        job = submit_render(sim.sup)
        self.running(sim, job, 0.3)
        tolerant_brain_check(sim)
        make_unreachable(sim.mlx)
        sim.at(sim.clock.now() + 30, lambda: make_reachable(sim.mlx))
        ok, crashes = self.launchd_run(sim, sim.quiet, 3600)
        self.assertTrue(ok)
        self.assertEqual(crashes, [], "an engine API outage must not kill gpud")
        self.assertEqual((sim.store.job(job).state, sim.store.job(job).attempt), ("succeeded", 1))
        self.assertEqual(len(calls(sim.mlx, "start", f"{job}.1")), 1)
        self.assertTrue(sim.store.events("engine_error"))
        self.assertTrue(sim.store.events("engine_recovered"))
        self.assertEqual(sim.store.events("engine_declared_lost"), [], "30 s is an outage, not a lost engine")
        self.assertEqual(sim.pages, [])
        self.assert_invariants(sim)

    def test_pager_outage_is_retried_with_backoff_without_crashing(self):
        """Sibling of P4 test_pager_outage_does_not_stop_the_brain_returning, with all of its
        assertions except the 'gpud really did crash' precondition, plus crashes == []."""
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        sim.pages = FlakyPager()
        sim.restart()
        sim.mlx.script_jobs("render_video", crash_mid_job(0.5))
        job = submit_render(sim.sup, max_attempts=1)

        def settled_but_unpaged():
            st = sim.store
            return (not st.jobs("queued", "starting", "running", "cancelling") and not st.state()["loading"]
                    and not st.unfired_continuations() and sim.brain_resident())
        ok, crashes = self.launchd_run(sim, settled_but_unpaged, 3600)
        self.assertTrue(ok)
        self.assertEqual(crashes, [], "a pager outage must not kill gpud")
        self.assertEqual(sim.store.job(job).state, "quarantined")
        self.assertTrue(sim.store.events("page_failed"))
        self.assertTrue(sim.store.unpaged_humans(), "the page is kept for retry, not dropped")
        sim.pages.down = False
        ok, crashes = self.launchd_run(sim, sim.quiet, 120)
        self.assertTrue(ok)
        self.assertEqual(crashes, [])
        self.assertEqual([p["kind"] for p in sim.pages], ["human_review"])
        self.assert_invariants(sim)


class H2DegradedSafe(AdvCase):

    def test_brain_comes_back_beside_a_wedged_engine(self):
        """oMLX wedges with DeepSeek resident and ignores SIGTERM. gpud pages and enters SAFE for
        that engine only: it never touches oMLX again, but the 21 GB fallback brain fits beside
        the stuck 195 GB model on mlx-serve, so the recover continuation still runs."""
        sim = self.make_with(capacity=256.0, brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        sim.omlx.terminate_fails = True
        sim.omlx.script_jobs("long_reasoning", hang(0.4, wedged=True))
        job = sim.sup.submit("long_reasoning", DEEPSEEK, {"goal": "audit the storyboard", "expected_s": 300},
                             {"kind": "review_answer"}, {"kind": "recover_answer"})
        self.settle(sim)
        st = sim.store.state()
        self.assertTrue(st["safe"])
        self.assertEqual(st["safe_engine"], "omlx")
        self.assertEqual(sim.store.job(job).outcome["cleanup"], "wedged")
        self.assertIn(DEEPSEEK, sim.omlx.resident_models(), "the wedged model still holds its memory")
        recover = continuations_of(sim.store, job)[0]
        self.assertEqual((recover.state, recover.model), ("succeeded", FALLBACK))
        self.assertEqual([p["kind"] for p in sim.pages], ["safe_mode"])
        self.assertEqual(len(calls(sim.omlx, "terminate")), 1)
        self.assert_invariants(sim)
        n = len(sim.omlx.calls)
        sim.run_until(lambda: False, 3600, dt=10)
        self.assertEqual(len(sim.omlx.calls), n, "SAFE never pokes the wedged engine")
        sim.omlx.operator_force_restart()
        sim.sup.clear_safe()
        self.settle(sim)
        self.assertFalse(sim.store.state()["safe"])
        self.assertIn(PRIMARY, sim.mlx.resident_models())
        self.assert_invariants(sim)


class H3SingleInstance(AdvCase):

    def test_second_supervisor_is_a_standby_until_the_lock_is_free(self):
        sim = self.make_with(capacity=256.0)
        job = submit_render(sim.sup)
        self.running(sim, job, 0.2)
        boots = sim.store.state()["boot_count"]
        b = Supervisor(sim.db_path, sim.engines, sim.clock, sim.policy, pager=sim.pages.append)
        self.addCleanup(b.close)
        self.assertFalse(b.leader)
        n_events = len(sim.store.events())
        b.recover()
        b.tick()
        self.assertEqual(sim.store.state()["boot_count"], boots, "a standby must not recover")
        self.assertEqual(len(sim.store.events()), n_events, "a standby must not write")
        sim.sup.close()  # the leader dies; the standby takes over on its next tick
        sim.sup = b
        self.settle(sim)
        self.assertTrue(b.leader)
        self.assertEqual(sim.store.state()["boot_count"], boots + 1)
        j = sim.store.job(job)
        self.assertEqual((j.state, j.attempt), ("succeeded", 1))
        self.assertEqual(len(calls(sim.mlx, "start", f"{job}.1")), 1, "taken over, not re-run")
        self.assertEqual(len(sim.store.events("lease_adopted", job)), 1)
        self.assert_invariants(sim)

    def test_client_mode_never_takes_the_lock(self):
        sim = self.make_with()
        sim.sup.close()  # gpud is down (launchd is about to restart it)
        client = Supervisor(sim.db_path, sim.engines, sim.clock, sim.policy, lead=False)
        self.addCleanup(client.close)
        job = client.submit("triage_inbox", payload={"goal": "queued from the CLI"})
        client.tick()
        self.assertFalse(client.leader)
        self.assertEqual(client.store.job(job).state, "queued", "a client never runs anything")
        sim.restart()  # the real gpud still gets the lock while the client is alive
        self.assertTrue(sim.sup.leader)
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")

    def test_superseded_leader_without_a_lock_steps_down_and_stale_renewals_are_fenced(self):
        sim = self.make_with(capacity=256.0)
        job = submit_render(sim.sup)
        self.running(sim, job, 0.2)
        a = sim.sup
        self.addCleanup(a.close)
        a._lock.release()
        a._lock = None  # e.g. the lock file sits on a filesystem where flock is a no-op
        b = Supervisor(sim.db_path, sim.engines, sim.clock, sim.policy, pager=sim.pages.append)
        b.recover()
        self.assertEqual(sim.store.lease(job)["fence"], b._epoch, "the new leader re-fenced the lease")
        self.assertFalse(sim.store.update_lease(job, owner=a._epoch, last_progress=0.99),
                         "a renewal under the old epoch is refused")
        n_events = len(sim.store.events())
        for _ in range(3):
            sim.clock.advance(1)
            a.tick()
        self.assertEqual(len(sim.store.events()), n_events, "the superseded leader did nothing")
        sim.sup = b
        self.settle(sim)
        self.assertEqual((sim.store.job(job).state, sim.store.job(job).attempt), ("succeeded", 1))
        self.assert_invariants(sim)


class MonoSkewedClock(SkewedClock):
    """A steppable wall clock that also offers a monotonic source (physical time)."""

    def monotonic(self) -> float:
        return self.physical.now()


class MonoSkewSim(Sim):
    def _spawn(self) -> Supervisor:
        if not hasattr(self, "wall"):
            self.wall = MonoSkewedClock(self.clock)
        return Supervisor(self.db_path, self.engines, self.wall, self.policy, pager=self.pages.append)


class H4Clocks(AdvCase):

    def test_monotonic_source_ignores_wall_steps_across_restarts(self):
        sim = self.make_with(cls=MonoSkewSim)
        job = submit_render(sim.sup)
        self.running(sim, job, 0.5)
        sim.wall.skew += 86400       # the date jumps a day forward...
        sim.restart(downtime_s=20)   # ...gpud restarts across it...
        sim.wall.skew -= 2 * 86400   # ...then the date jumps two days back
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assertEqual(sim.store.events("clock_step"), [], "with a monotonic source a step is invisible")
        self.assertAlmostEqual(sim.store.state()["t_now"], sim.clock.now(), msg="supervisor time == physical time")
        self.assert_invariants(sim)

    def test_wall_step_without_monotonic_source_is_recorded(self):
        sim = self.make_with(cls=SkewSim)
        job = submit_render(sim.sup)
        self.running(sim, job, 0.5)
        sim.wall.skew -= 3600
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        steps = sim.store.events("clock_step")
        self.assertEqual(len(steps), 1)
        self.assertLess(steps[0]["data"]["wall_jump_s"], -3500)


class H5PoisonAndAdmission(AdvCase):

    def test_lineage_budget_bounds_resubmissions_that_vary_the_payload(self):
        """A brain that changes the payload on every resubmission (so neither retry_of nor the
        content key links the attempts) is still bounded by the per-lineage failure budget."""
        class Shifty(ScriptedBrain):
            def __call__(self, spec, model):
                if spec.kind == "recover_render":
                    brief = spec.payload["brief"]
                    seed = brief["attempted_action"]["params"].get("seed", 0) + 1
                    return {"submit": [render_request(brief["goal"], seed=seed)], "note": "new seed"}
                return super().__call__(spec, model)

        sim = self.make_with(policy=Policy(max_lineage_failures=4), brain=Shifty())
        sim.mlx.script_jobs("render_video", *[crash_mid_job(0.5)] * 50)
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim, max_s=6 * 3600)
        renders = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertEqual(len(renders), 4)
        self.assertTrue(all(j.attempt == 1 for j in renders), "each new payload is a new logical task")
        humans = sim.store.jobs("awaiting_human")
        self.assertEqual(len(humans), 1)
        self.assertIn("lineage", humans[0].payload["reason"])
        self.assertEqual([p["kind"] for p in sim.pages], ["human_review"])
        self.assert_invariants(sim)

    def test_identical_resubmission_without_retry_of_is_counted_as_a_retry(self):
        class Forgetful(ScriptedBrain):
            def __call__(self, spec, model):
                if spec.kind == "recover_render":
                    return {"submit": [render_request(spec.payload["brief"]["goal"])]}
                return super().__call__(spec, model)

        sim = self.make_with(brain=Forgetful())
        sim.mlx.script_jobs("render_video", *[crash_mid_job(0.5)] * 10)
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim, max_s=4 * 3600)
        renders = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertEqual([(j.attempt, j.state) for j in renders], [(1, "failed"), (2, "failed"), (3, "quarantined")])
        self.assertEqual(len({j.task_key for j in renders}), 1)
        self.assertEqual(len(sim.store.events("retry_inferred")), 2)
        self.assert_invariants(sim)

    def test_fan_out_is_limited_per_decision(self):
        reqs = [render_request(f"{GOAL} (shot {i})") for i in range(12)]
        sim = self.make_with(brain=OneShotBrain(*reqs, retry_failures=False))
        plan = sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.assertTrue(sim.run_until(lambda: sim.store.job(plan).state == "succeeded", 600))
        out = sim.store.job(plan).outcome
        limit = sim.policy.max_submits_per_decision
        self.assertEqual(len(out["children"]), limit)
        self.assertEqual(len(out["rejected"]), 12 - limit)
        self.assertTrue(all("fan-out" in r["error"] for r in out["rejected"]))

    def test_requested_priority_and_attempts_are_clamped(self):
        sim = self.make_with()
        req = render_request(GOAL)
        job = sim.sup.submit(req["kind"], req["requires"], req["payload"], req["on_success"], req["on_failure"],
                             priority=10 ** 6, max_attempts=99)
        j = sim.store.job(job)
        self.assertEqual(j.priority, sim.policy.continuation_priority - 1)
        self.assertEqual(j.max_attempts, sim.policy.max_attempts)
        self.assertTrue(sim.store.events("request_clamped", job))

    def test_job_that_kills_gpud_at_every_start_is_quarantined(self):
        """gpud's own deaths are not charged to a job (A4), but a job that kills gpud on every
        start is poison: after max_interruptions it is quarantined instead of looping forever."""
        sim = self.make_with()

        def hook(point):
            if point == "after_starting_recorded":
                raise SimulatedCrash(point)
        original_restart = sim.restart

        def restart(downtime_s=0.0):
            original_restart(downtime_s)
            sim.arm(hook)
        sim.restart = restart
        sim.arm(hook)
        job = sim.sup.submit("triage_inbox", payload={"goal": "a request that kills gpud"})
        self.settle(sim)
        j = sim.store.job(job)
        self.assertEqual(j.state, "quarantined")
        self.assertEqual(j.interruptions, sim.policy.max_interruptions)
        self.assertEqual(j.attempt, 1, "gpud's own deaths are not charged as attempts")
        self.assertEqual([p["kind"] for p in sim.pages], ["human_review"])


class H6JobBoundary(AdvCase):

    def test_gpud_bug_on_one_job_fails_that_job_and_the_baton_still_fires(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.2)
        original = Supervisor._check_lease

        def broken(self_, j, poll, now):
            if j.id == job:
                raise RuntimeError("simulated gpud bug in the lease path")
            return original(self_, j, poll, now)
        sim.sup._check_lease = types.MethodType(broken, sim.sup)
        self.settle(sim)
        j = sim.store.job(job)
        self.assertEqual((j.state, j.outcome["reason"]), ("failed", "supervisor_error"))
        self.assertEqual(len(sim.store.events("job_fault", job)), sim.policy.max_job_faults)
        self.assertEqual(j.outcome["cleanup"], "unloaded", "the run was still cancelled and unloaded")
        self.assertEqual(continuations_of(sim.store, job)[0].kind, "recover_render")
        self.assertTrue(sim.brain_resident())
        self.assert_invariants(sim)

    def test_validator_that_raises_is_a_failed_check(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        original = validators.check

        def boom(family, payload, artifact):
            if family == "video":
                raise ZeroDivisionError("probe bug")
            return original(family, payload, artifact)
        validators.check = boom
        self.addCleanup(setattr, validators, "check", original)
        job = submit_render(sim.sup)
        self.settle(sim)
        j = sim.store.job(job)
        self.assertEqual(j.outcome["reason"], "validator")
        self.assertEqual(j.outcome["validator_results"][0]["check"], "validator_error")
        self.assert_invariants(sim)


class H7EngineLoss(AdvCase):

    def test_lost_engine_pages_once_then_everything_resumes_when_it_returns(self):
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.3)
        tolerant_brain_check(sim)
        make_unreachable(sim.mlx)
        t_down = sim.clock.now()
        sim.at(t_down + 600, lambda: make_reachable(sim.mlx))
        ok, crashes = self.launchd_run(sim, sim.quiet, 4 * 3600)
        self.assertTrue(ok)
        self.assertEqual(crashes, [])
        j = sim.store.job(job)
        self.assertEqual(j.state, "failed")
        self.assertIn(j.outcome["reason"], ("timeout", "engine_unreachable"))
        self.assertEqual(j.outcome["cleanup"], "engine_unreachable")
        lost = sim.store.events("engine_declared_lost")
        self.assertEqual(len(lost), 1)
        self.assertAlmostEqual(lost[0]["at"] - t_down, sim.policy.engine_lost_s, delta=2)
        self.assertEqual([p["kind"] for p in sim.pages], ["engine_lost"])
        recover = continuations_of(sim.store, job)[0]
        self.assertEqual(recover.state, "succeeded", "the baton waited for the engine, then the brain came back")
        self.assertGreater(recover.started_at, t_down + 600)
        self.assertNotIn(VIDEO, sim.mlx.resident_models(), "the orphaned run's model was unloaded")
        self.assert_invariants(sim)


OLD_SCHEMA = """
CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, requires TEXT NOT NULL,
  state TEXT NOT NULL, payload TEXT NOT NULL DEFAULT '{}', outcome TEXT, parent_id INTEGER REFERENCES jobs(id),
  depth INTEGER NOT NULL DEFAULT 0, task_key TEXT, attempt INTEGER NOT NULL DEFAULT 1, max_attempts INTEGER NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0, deadline REAL, idempotency_key TEXT UNIQUE,
  is_continuation INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, started_at REAL, finished_at REAL,
  engine TEXT, model TEXT, run_id TEXT, handle TEXT, pending_outcome TEXT, cancel_requested_at REAL,
  cancel_stage TEXT, paged_at REAL);
CREATE TABLE continuations (job_id INTEGER PRIMARY KEY REFERENCES jobs(id), on_success TEXT NOT NULL,
  on_failure TEXT NOT NULL, fired_outcome TEXT, fired_job_id INTEGER UNIQUE REFERENCES jobs(id), fired_at REAL);
CREATE TABLE leases (job_id INTEGER PRIMARY KEY REFERENCES jobs(id), engine TEXT NOT NULL, model TEXT NOT NULL,
  granted_at REAL NOT NULL, expires_at REAL NOT NULL, hard_deadline REAL NOT NULL, last_progress_at REAL NOT NULL,
  last_progress REAL NOT NULL DEFAULT 0, state TEXT NOT NULL);
CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, kind TEXT NOT NULL, job_id INTEGER,
  data TEXT NOT NULL DEFAULT '{}');
CREATE TABLE supervisor_state (id INTEGER PRIMARY KEY CHECK (id = 1), mode TEXT NOT NULL DEFAULT 'boot',
  safe INTEGER NOT NULL DEFAULT 0, safe_reason TEXT, safe_epoch INTEGER NOT NULL DEFAULT 0, brainless_since REAL,
  ladder TEXT NOT NULL DEFAULT '{}', loading TEXT, known_resident TEXT NOT NULL DEFAULT '[]',
  boot_count INTEGER NOT NULL DEFAULT 0);
INSERT INTO supervisor_state (id, known_resident) VALUES (1, '["qwen38-flash-next"]');
"""


class H8ForeignModelsAndMigration(AdvCase):

    def test_foreign_model_without_a_size_hint_is_assumed_to_fill_the_host(self):
        foreign = "llama-3.3-70b-4bit"
        sim = self.make_with(brain=ScriptedBrain(retry_failures=False))
        tolerant_brain_check(sim)
        sim.mlx.catalog = {**CATALOG, foreign: ModelSpec(foreign, "mlx-serve", 40, 20, "llm")}
        sim.mlx.model_size_gb = None  # an adapter without the optional size hint
        self.settle(sim)
        sim.mlx._models.add(foreign)
        job = submit_render(sim.sup)
        self.settle(sim)
        appeared = [e for e in sim.store.events("model_appeared") if e["data"]["model"] == foreign]
        self.assertEqual(appeared[0]["data"]["size_gb"], sim.policy.usable_gb)
        self.assertNotIn(foreign, sim.mlx.resident_models(), "unknown footprint: evicted before planning")
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assert_invariants(sim)

    def test_database_from_the_first_schema_is_migrated(self):
        path = os.path.join(self.tmp, "old.db")
        conn = sqlite3.connect(path)
        conn.executescript(OLD_SCHEMA)
        conn.close()
        store = Store(path)
        cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(jobs)")}
        self.assertTrue({"root_id", "content_key", "interruptions", "faults"} <= cols)
        self.assertEqual(store.state()["known_resident"], {PRIMARY: None})
        store.close()
        sim = Sim(path, 256.0)
        self.addCleanup(lambda: sim.sup.close())
        plan = sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim)
        self.assertEqual(sim.store.job(plan).state, "succeeded")
        self.assert_invariants(sim)


class ChaosBrain(ScriptedBrain):
    """A brain that sometimes answers with malformed or hostile decisions."""

    def __init__(self, rng):
        super().__init__()
        self.rng = rng

    def __call__(self, spec, model):
        if spec.kind == "plan_video" or self.rng.random() < 0.15:
            return self.rng.choice([
                {"submit": [render_request(GOAL)]},
                {"submit": render_request(GOAL)},
                {"submit": ["junk", 3, None]},
                {"submit": [dict(render_request(GOAL), priority="high", retry_of="job 4")]},
                {"submit": [dict(render_request(GOAL), payload={"goal": GOAL, "expected_s": float("nan")})]},
                {"submit": [{"kind": "summarize", "payload": {"goal": "s"}, "max_attempts": 99}]},
                {1: "mixed key types", "submit": []},
                "not a decision object",
            ])
        return super().__call__(spec, model)


class H9Chaos(AdvCase):
    METHODS = ("resident_models", "load", "poll_load", "unload", "start", "find", "poll", "cancel", "terminate")

    def chaos(self, engine, rng, rate=0.05):
        for name in self.METHODS:
            original = getattr(FakeEngine, name).__get__(engine)

            def wrapped(*args, _orig=original, _name=name):
                r = rng.random()
                if r < rate:
                    raise rng.choice([ConnectionError, TimeoutError, ValueError, KeyError])(f"chaos in {_name}")
                if r < rate * 1.3 and _name in ("poll", "resident_models", "poll_load", "start", "find"):
                    return rng.choice([None, 42, "x", [], {"status": 1}, object()])
                return _orig(*args)
            setattr(engine, name, wrapped)

    def test_no_exception_escapes_and_every_baton_fires(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                sim = self.make_with(capacity=rng.choice([176.0, 256.0]), brain=ChaosBrain(rng))
                sim.brain_resident = lambda s=sim: any(  # observer bypasses the chaos wrappers
                    m in CATALOG and CATALOG[m].is_brain for e in s.engines.values() for m in FakeEngine.resident_models(e))
                for e in sim.engines.values():
                    e.script_jobs("render_video", *[rng.choice([SUCCESS, hang(0.4), crash_mid_job(0.5), garbage(),
                                                                slow(3.0), garbage(black_frame_ratio=None)])
                                                    for _ in range(30)])
                sim.sup.submit("plan_video", payload={"goal": GOAL})

                def kill(point, r=rng):
                    if r.random() < 0.01:
                        raise SimulatedCrash(point)
                original_restart = sim.restart

                def restart(downtime_s=0.0, s=sim, o=original_restart, k=kill):
                    o(downtime_s)
                    s.arm(k)
                sim.restart, sim.crash_downtime_s = restart, 5.0
                sim.arm(kill)
                sim.at(5, lambda s=sim, r=rng: [self.chaos(e, r) for e in s.engines.values()])
                sim.at(3600, lambda s=sim: [e.__dict__.pop(m, None) for e in s.engines.values() for m in self.METHODS])
                for _ in range(4 * 3600):
                    sim.step()  # any Exception escaping the supervisor fails the test here
                    if sim.clock.now() > 3700 and sim.quiet():
                        break
                self.assertTrue(sim.quiet(), "never settled after the chaos stopped")
                st = sim.store
                self.assertEqual(st.events("supervisor_error") + st.events("job_fault"), [],
                                 "chaos must be absorbed as engine errors and rejections, not gpud faults")
                self.assertTrue(st.events("engine_error"))
                for j in st.jobs():
                    if j.brainless:
                        self.assertIn(j.state, TERMINAL)
                        self.assertTrue(st.continuation(j.id)["fired_job_id"], f"baton of {j} never fired")
                fired = [e["job_id"] for e in st.events("continuation_fired")]
                self.assertEqual(len(fired), len(set(fired)))
                self.assertEqual(sim.host.oom, [])


if __name__ == "__main__":
    unittest.main()
