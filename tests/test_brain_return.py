"""Invariant tests for the brain-return mechanism (I1-I9). Everything runs on a virtual clock
against fake engines; nothing sleeps and nothing touches a real model server."""
from __future__ import annotations

import os
import random
import tempfile
import unittest
from collections import defaultdict

from foreman.engine import Engine
from foreman.policy import CATALOG, Policy
from foreman.sim import (SUCCESS, TIGHT_HOST_GB, CrashAt, FakeEngine, ScriptedBrain, Sim, SimulatedCrash,
                         crash_mid_job, garbage, hang, render_request, slow)
from foreman.store import ACTIVE, TERMINAL
from foreman.supervisor import BRIEF_KEYS, CRASH_POINTS, AdmissionError, Supervisor

GOAL = "8s clip: a red fox trotting through fresh snow"
PRIMARY, FALLBACK, VISION, VIDEO = "qwen38-flash-next", "qwen36-35b-a3b", "qwen38-27b", "ltx-2.5"
DEEPSEEK = "deepseek-v4.1-flash-1017"


def submit_render(sup, max_attempts=None, **payload):
    req = render_request(GOAL, **payload)
    opts = {"max_attempts": max_attempts} if max_attempts else {}
    return sup.submit(req["kind"], req["requires"], req["payload"], req["on_success"], req["on_failure"], **opts)


def continuations_of(store, job_id):
    return [j for j in store.children(job_id) if j.is_continuation]


def calls(engine, name, arg=None):
    return [c for c in engine.calls if c[1] == name and (arg is None or c[2] == arg)]


class SimCase(unittest.TestCase):
    capacity = TIGHT_HOST_GB  # default: no brain fits beside the video model

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp, self.n = tmp.name, 0

    def make(self, capacity=None, policy=None, brain=None) -> Sim:
        self.n += 1
        sim = Sim(os.path.join(self.tmp, f"foreman-{self.n}.db"), capacity or self.capacity, policy, brain)
        self.addCleanup(lambda: sim.sup.close())
        return sim

    def settle(self, sim, max_s=7200.0, dt=1.0):
        self.assertTrue(sim.run_until_quiet(max_s, dt), "system never settled")

    def running(self, sim, job_id, min_progress=0.0):
        def pred():
            j = sim.store.job(job_id)
            lease = sim.store.lease(job_id)
            return j.state == "running" and lease and lease["last_progress"] >= min_progress
        self.assertTrue(sim.run_until(pred, 3600), f"job {job_id} never ran")

    # ---- shared invariant checks ----
    def assert_brief(self, brief, job):
        for key in BRIEF_KEYS:
            self.assertIn(key, brief, f"brief for {job} lacks {key}")
        self.assertEqual(brief["goal"], job.payload["goal"])
        self.assertEqual(brief["attempt"], job.attempt)
        self.assertEqual(brief["attempted_action"]["kind"], job.kind)
        self.assertEqual(brief["attempted_action"]["model"], job.requires)
        self.assertEqual(brief["outcome"]["status"], job.state)
        self.assertIsInstance(brief["artifacts"], list)
        self.assertIsInstance(brief["validator_results"], list)
        self.assertTrue(brief["log_tail"], "log tail must never be empty")
        if job.state == "succeeded" or (job.outcome or {}).get("reason") == "validator":
            self.assertTrue(brief["artifacts"])
            self.assertTrue(brief["validator_results"])

    def assert_invariants(self, sim, quiet=True):
        store = sim.store
        jobs = store.jobs()
        conts = defaultdict(list)
        for j in jobs:
            if j.is_continuation:
                conts[j.parent_id].append(j)
        for j in jobs:
            self.assertLessEqual(len(conts[j.id]), 1, f"duplicate continuation for {j}")
            if not j.brainless:
                continue
            if quiet:
                self.assertIn(j.state, TERMINAL, f"lost job {j}")
            if j.state in TERMINAL:
                self.assertEqual(len(conts[j.id]), 1, f"no continuation for {j}")
                self.assertEqual(store.continuation(j.id)["fired_job_id"], conts[j.id][0].id)
                self.assert_brief(conts[j.id][0].payload["brief"], j)
        fired = [e["job_id"] for e in store.events("continuation_fired")]
        self.assertEqual(len(fired), len(set(fired)), "a continuation fired twice")
        for eng in sim.engines.values():
            self.assertFalse({c[1] for c in eng.calls} & {"kill", "force_kill", "sigkill"})
        self.assertEqual(sim.host.oom, [], "planner overcommitted memory")
        if quiet:
            self.assertFalse(store.jobs("queued", *ACTIVE))
            self.assertTrue(sim.brain_resident() or store.state()["safe"], "brain did not come back")

    def assert_returned_within_sla(self, sim, cont):
        self.assertEqual(cont.state, "succeeded", f"{cont} did not run")
        load_s = CATALOG[cont.model].load_s
        self.assertLessEqual(cont.started_at - cont.created_at, sim.policy.brain_return_sla_s + load_s)


class I1Admission(SimCase):
    def test_brainless_job_without_both_continuations_is_rejected(self):
        sim = self.make()
        payload = {"goal": GOAL, "expected_s": 300}
        for on_s, on_f in [(None, None), ({"kind": "review"}, None), (None, {"kind": "recover"}),
                           ({"kind": "review"}, {}), ({}, {"kind": "recover"})]:
            with self.subTest(on_success=on_s, on_failure=on_f):
                with self.assertRaises(AdmissionError):
                    sim.sup.submit("render_video", VIDEO, payload, on_s, on_f)
        self.assertEqual(sim.store.jobs(), [])
        self.assertEqual(len(sim.store.events("admission_rejected")), 5)

    def test_brief_seed_fields_are_required(self):
        sim = self.make()
        ok = ({"kind": "review"}, {"kind": "recover"})
        with self.assertRaises(AdmissionError):
            sim.sup.submit("render_video", VIDEO, {"expected_s": 300}, *ok)          # no goal
        with self.assertRaises(AdmissionError):
            sim.sup.submit("render_video", VIDEO, {"goal": GOAL}, *ok)              # no expected_s
        self.assertGreater(sim.sup.submit("render_video", VIDEO, {"goal": GOAL, "expected_s": 300}, *ok), 0)

    def test_brain_jobs_need_no_continuation(self):
        sim = self.make()
        self.assertGreater(sim.sup.submit("plan_video", payload={"goal": GOAL}), 0)

    def test_brain_cannot_smuggle_in_a_brainless_job(self):
        class Careless(ScriptedBrain):
            def __call__(self, spec, model):
                if spec.kind == "plan_video":
                    return {"submit": [{"kind": "render_video", "requires": VIDEO,
                                        "payload": {"goal": GOAL, "expected_s": 300},
                                        "on_success": {"kind": "review_render"}}]}  # no on_failure
                return super().__call__(spec, model)

        sim = self.make(brain=Careless())
        plan = sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim)
        self.assertEqual([j.kind for j in sim.store.jobs()], ["plan_video"])
        self.assertEqual(sim.store.events("admission_rejected")[0]["job_id"], plan)
        self.assertEqual(sim.store.job(plan).outcome["rejected"][0]["kind"], "render_video")
        self.assert_invariants(sim)


class I2ExactlyOnceAndBrainReturn(SimCase):
    FAULTS = {
        "success": ([SUCCESS], "succeeded", None),
        "crash": ([crash_mid_job(0.5)], "failed", "crash"),
        "engine_process_crash": ([crash_mid_job(0.3, restart_engine=True)], "failed", "engine_lost"),
        "hang": ([hang(0.4)], "failed", "timeout"),
        "slow_past_hard_timeout": ([slow(3.0)], "failed", "timeout"),
        "garbage_output": ([garbage()], "failed", "validator"),
        "hang_ignoring_cancel": ([hang(0.4, ignore_cancel=True)], "failed", "timeout"),
        "wedged_engine": ([hang(0.4, wedged=True)], "failed", "timeout"),
    }

    def check_one_continuation(self, sim, job_id, state, reason):
        job = sim.store.job(job_id)
        self.assertEqual(job.state, state)
        self.assertEqual((job.outcome or {}).get("reason"), reason)
        conts = continuations_of(sim.store, job_id)
        self.assertEqual(len(conts), 1)
        self.assertEqual(conts[0].kind, "review_render" if state == "succeeded" else "recover_render")
        self.assert_returned_within_sla(sim, conts[0])
        self.assertTrue(sim.brain_resident())
        self.assert_invariants(sim)

    def test_every_terminal_outcome_fires_one_continuation_and_brings_brain_back(self):
        for name, (script, state, reason) in self.FAULTS.items():
            with self.subTest(fault=name):
                sim = self.make(brain=ScriptedBrain(retry_failures=False))
                sim.mlx.script_jobs("render_video", *script)
                job = submit_render(sim.sup)
                self.settle(sim)
                self.check_one_continuation(sim, job, state, reason)
                self.assertGreater(sim.max_brain_away(), 0, "brain should really have left (tight host)")

    def test_cancel_while_running(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.2)
        self.assertTrue(sim.sup.cancel_job(job, "user changed their mind"))
        self.settle(sim)
        self.check_one_continuation(sim, job, "failed", "cancelled")

    def test_cancel_while_queued(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.assertTrue(sim.sup.cancel_job(job))
        self.settle(sim)
        self.check_one_continuation(sim, job, "failed", "cancelled")

    def test_media_model_load_failure_and_load_hang(self):
        for outcome in ("fail", "hang"):
            with self.subTest(load=outcome):
                sim = self.make(brain=ScriptedBrain(retry_failures=False))
                sim.mlx.script_loads(VIDEO, outcome, outcome)
                job = submit_render(sim.sup)
                self.settle(sim)
                self.check_one_continuation(sim, job, "failed", "load_failed")
                self.assertEqual(len(calls(sim.mlx, "load", VIDEO)), 2)

    def test_exclusive_deepseek_job_on_omlx(self):
        sim = self.make(capacity=256.0, brain=ScriptedBrain(retry_failures=False))
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim)
        job = sim.sup.submit("long_reasoning", DEEPSEEK, {"goal": "audit the storyboard", "expected_s": 600},
                             {"kind": "review_answer"}, {"kind": "recover_answer"})
        self.assertTrue(sim.run_until(lambda: sim.store.job(job).state == "running", 600))
        self.assertEqual(sim.mlx.resident_models(), set(), "exclusive model must run alone")
        self.settle(sim)
        conts = continuations_of(sim.store, job)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assertEqual(len(conts), 1)
        self.assert_returned_within_sla(sim, conts[0])
        self.assert_invariants(sim)

    def test_full_retry_flow_returns_brain_after_each_brainless_job(self):
        sim = self.make()
        sim.mlx.script_jobs("render_video", hang(0.4), crash_mid_job(0.5), SUCCESS)
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim)
        renders = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertEqual([(j.attempt, j.state) for j in renders], [(1, "failed"), (2, "failed"), (3, "succeeded")])
        for r in renders:
            self.assert_returned_within_sla(sim, continuations_of(sim.store, r.id)[0])
        self.assert_invariants(sim)

    def test_vision_brain_co_resident_needs_no_swap_on_roomy_host(self):
        sim = self.make(capacity=256.0)
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim)
        render = next(j for j in sim.store.jobs() if j.kind == "render_video")
        review = continuations_of(sim.store, render.id)[0]
        self.assertEqual(review.model, VISION)
        self.assertLessEqual(review.started_at - review.created_at, 1.0)  # no load needed
        self.assertFalse([c for c in calls(sim.mlx, "load") if render.started_at <= c[0] <= review.started_at])
        plan = sim.store.job(render.parent_id)
        away = sim.max_brain_away(after=plan.finished_at)  # only the unload/load swap for the render
        self.assertLessEqual(away, CATALOG[VIDEO].load_s + CATALOG[VISION].load_s + 2)
        self.assert_invariants(sim)


class I3LeaseExpiry(SimCase):
    def run_one(self, *script, brain=None, **payload):
        sim = self.make(brain=brain or ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", *script)
        job = submit_render(sim.sup, **payload)
        self.settle(sim)
        return sim, sim.store.job(job)

    def call_names_since(self, sim, t):
        return [c[1] for c in sim.mlx.calls if c[0] >= t and c[1] in ("cancel", "unload", "terminate")]

    def test_no_progress_expires_lease_then_graceful_cancel_then_unload(self):
        sim, job = self.run_one(hang(0.4))
        lease = sim.store.lease(job.id)
        self.assertEqual(job.outcome["timeout"], "no_progress")
        self.assertEqual(lease["state"], "expired")
        expired_at = sim.store.events("lease_expired", job.id)[0]["at"]
        self.assertEqual(expired_at - lease["last_progress_at"], sim.policy.progress_timeout_s)
        self.assertEqual(self.call_names_since(sim, expired_at)[:2], ["cancel", "unload"])
        self.assertNotIn("terminate", self.call_names_since(sim, 0))
        self.assertEqual(job.outcome["cleanup"], "unloaded")

    def test_slow_but_progressing_job_is_not_killed(self):
        sim, job = self.run_one(slow(1.5))
        self.assertEqual(job.state, "succeeded")
        self.assertEqual(calls(sim.mlx, "cancel"), [])

    def test_hard_timeout_kills_job_that_progresses_too_slowly(self):
        sim, job = self.run_one(slow(3.0))
        hard = sim.policy.hard_timeout_s(VIDEO, job.payload)
        self.assertEqual(job.outcome["timeout"], "hard")
        self.assertEqual(sim.store.events("lease_expired", job.id)[0]["at"] - job.started_at, hard)
        self.assertEqual(self.call_names_since(sim, job.started_at)[:2], ["cancel", "unload"])

    def test_cancel_ignored_then_unload_after_grace(self):
        sim, job = self.run_one(hang(0.4, ignore_cancel=True))
        cancel_t = calls(sim.mlx, "cancel")[0][0]
        unload_t = [c[0] for c in calls(sim.mlx, "unload", VIDEO) if c[0] >= cancel_t][0]
        self.assertEqual(unload_t - cancel_t, sim.policy.cancel_grace_s)
        self.assertEqual(job.outcome["cleanup"], "unloaded")
        self.assertEqual(calls(sim.mlx, "terminate"), [])

    def test_wedged_engine_gets_graceful_terminate_never_force_kill(self):
        sim, job = self.run_one(hang(0.4, wedged=True))
        self.assertEqual(self.call_names_since(sim, job.started_at)[:3], ["cancel", "unload", "terminate"])
        self.assertEqual(job.outcome["cleanup"], "terminated")
        self.assertEqual(sim.mlx.restarts, 1)
        self.assertTrue(sim.brain_resident())
        self.assert_invariants(sim)

    def test_engine_that_ignores_terminate_pages_human_and_stops(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.terminate_fails = True
        sim.mlx.script_jobs("render_video", hang(0.4, wedged=True))
        job = submit_render(sim.sup)
        self.settle(sim)
        st = sim.store.state()
        self.assertTrue(st["safe"])
        self.assertEqual(sim.store.job(job).outcome["cleanup"], "wedged")
        self.assertIn("force-kill is never automatic", st["safe_reason"])
        self.assertTrue(any(p["kind"] == "safe_mode" for p in sim.pages))
        recover = continuations_of(sim.store, job)[0]
        self.assertEqual(recover.state, "queued")  # the baton waits for the human
        n_calls = len(sim.mlx.calls)
        sim.run_until(lambda: False, 3 * 3600, dt=10)
        self.assertEqual(len(sim.mlx.calls), n_calls, "SAFE mode must not keep poking the engine")
        sim.mlx.operator_force_restart()  # the human does what gpud never does
        sim.sup.clear_safe()
        self.settle(sim)
        self.assertEqual(sim.store.job(recover.id).state, "succeeded")
        self.assert_invariants(sim)

    def test_interface_has_no_force_kill(self):
        for name in ("kill", "force_kill", "sigkill"):
            self.assertFalse(hasattr(Engine, name))
            self.assertFalse(hasattr(FakeEngine, name))


class I4SupervisorCrashRecovery(SimCase):
    def test_crash_between_job_finished_and_continuation_enqueued(self):
        for outcome in ("success", "hang"):
            with self.subTest(outcome=outcome):
                sim = self.make(brain=ScriptedBrain(retry_failures=False))
                sim.mlx.script_jobs("render_video", SUCCESS if outcome == "success" else hang(0.4))
                job = submit_render(sim.sup)
                self.running(sim, job)
                hook = CrashAt("after_terminal_recorded")
                sim.arm(hook)
                self.settle(sim)
                self.assertTrue(hook.fired)
                self.assertEqual(sim.restarts, 1)
                self.assertEqual(len(continuations_of(sim.store, job)), 1)
                self.assertEqual(len(sim.store.events("continuation_fired", job)), 1)
                self.assert_invariants(sim)

    def test_crash_inside_continuation_transaction_rolls_back_then_fires_once(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.running(sim, job)
        hook = CrashAt("inside_fire_txn")
        sim.arm(hook)
        self.settle(sim)
        self.assertTrue(hook.fired)
        self.assertEqual(len(continuations_of(sim.store, job)), 1)
        self.assert_invariants(sim)

    def test_restart_mid_render_adopts_the_live_run(self):
        sim = self.make()
        job = submit_render(sim.sup)
        self.running(sim, job, 0.5)
        sim.restart(downtime_s=60)
        self.settle(sim)
        j = sim.store.job(job)
        self.assertEqual((j.state, j.attempt), ("succeeded", 1))
        self.assertEqual(len(calls(sim.mlx, "start", f"{job}.1")), 1, "the render must not be re-run")
        self.assertEqual(len(sim.store.events("lease_adopted", job)), 1)
        self.assert_invariants(sim)

    def test_render_that_finished_while_supervisor_was_down_is_harvested(self):
        sim = self.make()
        job = submit_render(sim.sup)
        self.running(sim, job, 0.5)
        sim.restart(downtime_s=600)
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assert_invariants(sim)

    def test_hang_during_downtime_is_caught_after_restart(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        sim.mlx.script_jobs("render_video", hang(0.4))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.4)
        sim.restart(downtime_s=900)
        self.settle(sim)
        self.assertEqual(sim.store.job(job).outcome["reason"], "timeout")
        self.assert_invariants(sim)

    def test_engine_restarted_during_downtime_marks_interrupted_once(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.3)
        sim.sup.close()
        sim.mlx.terminate()  # e.g. the engine was restarted while gpud was also down
        sim.restart(downtime_s=30)
        sim.restart()        # and gpud bounces again immediately
        sim.sup.recover()    # recover() itself is idempotent
        self.settle(sim)
        self.assertEqual(sim.store.job(job).outcome["reason"], "interrupted")
        self.assertEqual(len(continuations_of(sim.store, job)), 1)
        self.assert_invariants(sim)

    def test_crash_during_recovery_itself(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.running(sim, job, 0.3)
        sim.sup.close()
        sim.mlx.terminate()  # the run is gone, so recovery must mark it interrupted
        doomed = Supervisor(sim.db_path, sim.engines, sim.clock, sim.policy, crash=CrashAt("after_terminal_recorded"))
        with self.assertRaises(SimulatedCrash):
            doomed.recover()  # dies after recording 'interrupted', before firing the continuation
        doomed.close()
        sim.restart()
        self.settle(sim)
        self.assertEqual(sim.store.job(job).outcome["reason"], "interrupted")
        self.assertEqual(len(continuations_of(sim.store, job)), 1)
        self.assert_invariants(sim)

    def test_unique_key_backstops_a_lost_fired_marker(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        job = submit_render(sim.sup)
        self.settle(sim)
        first = continuations_of(sim.store, job)[0].id
        sim.store.conn.execute("UPDATE continuations SET fired_job_id = NULL WHERE job_id = ?", (job,))
        sim.restart()
        self.settle(sim)
        self.assertEqual([j.id for j in continuations_of(sim.store, job)], [first])
        self.assertEqual(sim.store.continuation(job)["fired_job_id"], first)

    def test_idempotent_submit(self):
        sim = self.make()
        a = sim.sup.submit("plan_video", payload={"goal": GOAL}, idempotency_key="user-request-42")
        sim.restart()
        b = sim.sup.submit("plan_video", payload={"goal": GOAL}, idempotency_key="user-request-42")
        self.assertEqual(a, b)
        self.assertEqual(len(sim.store.jobs()), 1)

    def demo_flow(self, policy=None):
        sim = self.make(policy=policy or Policy(max_attempts=6))
        sim.mlx.script_jobs("render_video", hang(0.4), SUCCESS)
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        return sim

    def assert_goal_reached(self, sim):
        reviews = [j for j in sim.store.jobs() if j.kind == "review_render"]
        self.assertEqual([j.state for j in reviews], ["succeeded"])
        self.assertEqual(sim.store.jobs("awaiting_human"), [])

    def test_crash_at_every_point(self):
        for point in CRASH_POINTS:
            for nth in (1, 2):
                with self.subTest(point=point, nth=nth):
                    sim = self.demo_flow()
                    hook = CrashAt(point, nth)
                    sim.arm(hook)
                    self.settle(sim)
                    if nth == 1:
                        self.assertTrue(hook.fired, f"crash point {point} never reached")
                    self.assert_invariants(sim)
                    self.assert_goal_reached(sim)

    def test_random_restarts_with_downtime(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                sim = self.demo_flow(Policy(max_attempts=8))
                for _ in range(20000):
                    if rng.random() < 0.03:
                        sim.restart(downtime_s=rng.choice([0, 0, 5, 30, 150]))
                    sim.step()
                    if sim.quiet():
                        break
                self.assertTrue(sim.quiet())
                self.assertGreater(sim.restarts, 5)
                self.assert_invariants(sim)
                self.assert_goal_reached(sim)


class I5BrainLoadLadder(SimCase):
    def plan(self, sim):
        return sim.sup.submit("plan_video", payload={"goal": GOAL})

    def test_primary_retried_once(self):
        sim = self.make()
        sim.mlx.script_loads(PRIMARY, "fail")
        plan = self.plan(sim)
        self.settle(sim)
        self.assertEqual(sim.store.job(plan).model, PRIMARY)
        self.assertEqual(len(sim.store.events("load_failed")), 1)

    def test_fallback_after_primary_fails_twice(self):
        sim = self.make()
        sim.mlx.script_loads(PRIMARY, "fail", "fail")
        plan = self.plan(sim)
        self.settle(sim)
        self.assertEqual(sim.store.job(plan).model, FALLBACK)
        self.assertEqual(sim.store.events("brain_ladder_step")[0]["data"]["skipped"], PRIMARY)
        self.assertEqual(len(calls(sim.mlx, "load", PRIMARY)), 2)
        self.assertFalse(sim.store.state()["safe"])
        self.assert_invariants(sim)

    def test_load_hang_counts_as_failure(self):
        sim = self.make()
        sim.mlx.script_loads(PRIMARY, "hang", "hang")
        plan = self.plan(sim)
        self.settle(sim)
        self.assertEqual(sim.store.job(plan).model, FALLBACK)
        limit = CATALOG[PRIMARY].load_s * sim.policy.load_timeout_factor
        self.assertIn(f"{limit:.0f}s", sim.store.events("load_failed")[0]["data"]["reason"])

    def test_ladder_exhausted_pages_human_enters_safe_and_never_loops(self):
        sim = self.make()
        sim.mlx.script_loads(PRIMARY, "fail", "fail")
        sim.mlx.script_loads(FALLBACK, "fail", "fail")
        plan = self.plan(sim)
        self.settle(sim)
        st = sim.store.state()
        self.assertTrue(st["safe"])
        self.assertEqual([p["kind"] for p in sim.pages], ["safe_mode"])
        self.assertEqual(len(calls(sim.mlx, "load")), 4)
        sim.run_until(lambda: False, 6 * 3600, dt=10)  # well past the ladder cooldown
        self.assertEqual(len(calls(sim.mlx, "load")), 4, "SAFE mode must not retry loads by itself")
        self.assertEqual(sim.store.job(plan).state, "queued", "queued work is kept, not dropped")
        sim.sup.clear_safe()
        self.settle(sim)
        self.assertEqual(sim.store.job(plan).state, "succeeded")
        self.assert_invariants(sim)

    def test_ladder_state_survives_supervisor_restart(self):
        sim = self.make()
        sim.mlx.script_loads(PRIMARY, "fail", "fail", "fail", "fail")
        plan = self.plan(sim)
        self.assertTrue(sim.run_until(lambda: sim.store.events("load_failed"), 600))
        sim.restart()
        self.settle(sim)
        self.assertEqual(len(calls(sim.mlx, "load", PRIMARY)), 2, "a crash loop must not reset the ladder")
        self.assertEqual(sim.store.job(plan).model, FALLBACK)

    def test_vision_brain_failure_degrades_to_text_brain(self):
        sim = self.make()
        sim.mlx.script_loads(VISION, "fail", "fail")
        self.plan(sim)
        self.settle(sim)
        review = next(j for j in sim.store.jobs() if j.kind == "review_render")
        self.assertEqual(review.state, "succeeded")
        self.assertEqual(review.model, PRIMARY)
        self.assertTrue(sim.store.events("brain_degraded"))
        self.assert_invariants(sim)


class I6Poison(SimCase):
    def test_media_job_failing_every_attempt_is_quarantined(self):
        sim = self.make()
        sim.mlx.script_jobs("render_video", *[crash_mid_job(0.5)] * 10)
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim)
        renders = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertEqual([j.state for j in renders], ["failed", "failed", "quarantined"])
        review = continuations_of(sim.store, renders[-1].id)[0]
        self.assertEqual((review.kind, review.state), ("human_review", "awaiting_human"))
        self.assertEqual(review.payload["brief"]["attempt"], 3)
        self.assertEqual([p["kind"] for p in sim.pages], ["human_review"])
        self.assert_invariants(sim)

    def test_brain_job_that_keeps_failing_is_quarantined(self):
        class Exploding(ScriptedBrain):
            def __call__(self, spec, model):
                if spec.kind == "recover_render":
                    raise RuntimeError("poisoned brief")
                return super().__call__(spec, model)

        sim = self.make(brain=Exploding())
        sim.mlx.script_jobs("render_video", crash_mid_job(0.5))
        job = submit_render(sim.sup)
        self.settle(sim)
        recover = continuations_of(sim.store, job)[0]
        self.assertEqual((recover.state, recover.attempt), ("quarantined", sim.policy.max_attempts))
        humans = sim.store.jobs("awaiting_human")
        self.assertEqual(len(humans), 1)
        self.assertEqual(humans[0].payload["brief"]["source_job_id"], job)
        self.assert_invariants(sim)

    def test_depth_limit_stops_an_endless_chain(self):
        sim = self.make(policy=Policy(max_continuation_depth=6), brain=ScriptedBrain(chain_forever=True))
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        self.settle(sim, max_s=6 * 3600)
        jobs = sim.store.jobs()
        self.assertLessEqual(len(jobs), 10)
        self.assertLessEqual(max(j.depth for j in jobs), 7)
        self.assertEqual(len(sim.store.jobs("awaiting_human")), 1)
        self.assert_invariants(sim)


class I7MaxBrainless(SimCase):
    def batch(self, sup, items=6, expected_s=1200.0):
        payload = {"goal": "six shots for the trailer", "items": [f"shot-{i}" for i in range(items)],
                   "expected_s": expected_s}
        return sup.submit("render_video", VIDEO, payload, {"kind": "review_render", "needs": "vision"},
                          {"kind": "recover_render"})

    def test_long_batch_is_split_with_a_brain_checkpoint(self):
        sim = self.make()
        first = self.batch(sim.sup)
        self.assertTrue(sim.store.events("batch_split", first))
        self.settle(sim)
        renders = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertEqual([len(j.payload["items"]) for j in renders], [3, 3])
        self.assertTrue(all(j.state == "succeeded" for j in renders))
        checkpoint = continuations_of(sim.store, renders[0].id)[0]
        self.assertEqual((checkpoint.kind, checkpoint.state), ("batch_checkpoint", "succeeded"))
        self.assertLess(checkpoint.started_at, renders[1].started_at)
        self.assertEqual(continuations_of(sim.store, renders[1].id)[0].kind, "review_render")
        self.assertLessEqual(sim.max_brain_away(), sim.policy.max_brainless_s)
        self.assert_invariants(sim)

    def test_cap_holds_even_when_every_chunk_runs_to_hard_timeout(self):
        sim = self.make()
        sim.mlx.script_jobs("render_video", slow(3.0), slow(3.0), SUCCESS, SUCCESS)  # progressing, too slow
        self.batch(sim.sup)
        self.settle(sim, max_s=4 * 3600)
        self.assertGreater(sim.max_brain_away(), 1200)  # it really was a long absence...
        self.assertLessEqual(sim.max_brain_away(), sim.policy.max_brainless_s)  # ...but within the cap
        self.assert_invariants(sim)

    def test_unsplittable_job_over_the_cap_is_rejected(self):
        sim = self.make()
        with self.assertRaises(AdmissionError):
            submit_render(sim.sup, expected_s=1000)

    def test_scheduler_inserts_checkpoint_when_brain_already_away(self):
        sim = self.make(policy=Policy(max_brainless_s=1200))
        sim.mlx.script_jobs("render_video", slow(3.0), SUCCESS)
        a = submit_render(sim.sup, max_attempts=1)   # will die at its hard timeout -> human review
        b = submit_render(sim.sup)
        self.settle(sim)
        self.assertEqual(sim.store.job(a).state, "quarantined")
        checkpoint = next(j for j in sim.store.jobs() if j.kind == "brain_checkpoint")
        self.assertEqual(checkpoint.state, "succeeded")
        self.assertLess(checkpoint.started_at, sim.store.job(b).started_at)
        self.assertTrue(sim.store.events("cap_checkpoint"))
        self.assertLessEqual(sim.max_brain_away(), 1200)
        self.assert_invariants(sim)


class I8Eviction(SimCase):
    def test_idle_brain_eviction_is_detected_and_brain_reloaded(self):
        sim = self.make()
        self.settle(sim)
        sim.mlx.evict(PRIMARY)
        self.settle(sim)
        evicted = [e for e in sim.store.events("model_evicted") if e["data"]["model"] == PRIMARY]
        self.assertEqual(len(evicted), 1)
        self.assertEqual(len(calls(sim.mlx, "load", PRIMARY)), 2)
        self.assertIn(PRIMARY, sim.mlx.resident_models())

    def test_brain_evicted_mid_decision_is_reloaded_and_job_retried(self):
        sim = self.make()
        plan = sim.sup.submit("plan_video", payload={"goal": GOAL})
        sim.at(75, lambda: sim.mlx.evict(PRIMARY))  # plan_video runs from t=61 to t=91
        self.settle(sim)
        job = sim.store.job(plan)
        self.assertEqual((job.state, job.attempt), ("succeeded", 2))
        self.assertTrue(sim.store.events("brain_job_retry", plan))
        self.assertEqual(sim.store.jobs("awaiting_human"), [])
        self.assert_invariants(sim)

    def test_co_resident_brain_evicted_during_render(self):
        sim = self.make(capacity=256.0)
        job = submit_render(sim.sup)
        self.running(sim, job, 0.3)
        self.assertIn(VISION, sim.mlx.resident_models())
        sim.mlx.evict(VISION)
        self.settle(sim)
        review = continuations_of(sim.store, job)[0]
        self.assertEqual((review.state, review.model), ("succeeded", VISION))
        self.assertEqual(len(calls(sim.mlx, "load", VISION)), 2)
        self.assert_invariants(sim)


class I9ResumeBrief(SimCase):
    def brief_for(self, *script, retry=False):
        sim = self.make(brain=ScriptedBrain(retry_failures=retry))
        sim.mlx.script_jobs("render_video", *script)
        job = submit_render(sim.sup)
        self.settle(sim)
        return sim, continuations_of(sim.store, job)[0].payload["brief"]

    def test_timeout_brief(self):
        _, brief = self.brief_for(hang(0.4))
        self.assertEqual(brief["goal"], GOAL)
        self.assertEqual(brief["attempted_action"]["params"]["prompt"], GOAL)
        self.assertEqual(brief["attempted_action"]["on_failure"], {"kind": "recover_render"})
        self.assertEqual((brief["outcome"]["reason"], brief["outcome"]["timeout"]), ("timeout", "no_progress"))
        self.assertTrue(any("step 8/20" in line for line in brief["log_tail"]))
        self.assertEqual((brief["attempt"], brief["max_attempts"]), (1, 3))
        self.assertEqual(brief["artifacts"], [])

    def test_success_and_validator_failure_briefs(self):
        _, ok = self.brief_for(SUCCESS)
        self.assertTrue(ok["artifacts"][0].endswith(".mp4"))
        self.assertTrue(all(r["ok"] for r in ok["validator_results"]))
        _, bad = self.brief_for(garbage(black_frame_ratio=0.9))
        self.assertTrue(bad["artifacts"])
        self.assertEqual([r["check"] for r in bad["validator_results"] if not r["ok"]], ["black_frames"])

    def test_stateless_brain_rebuilds_the_retry_from_the_brief_alone(self):
        sim, _ = self.brief_for(crash_mid_job(0.5), SUCCESS, retry=True)
        first, retry = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertEqual(retry.payload, first.payload)
        self.assertEqual((retry.task_key, retry.attempt), (first.task_key, 2))
        self.assertEqual(sim.store.continuation(retry.id)["on_success"], sim.store.continuation(first.id)["on_success"])
        self.assert_invariants(sim)


if __name__ == "__main__":
    unittest.main()
