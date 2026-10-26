"""Regressions from the planner's first live run on the Studio (2026-10-26).

The 27B brain planned 15 proposals and gpud admitted 12. The first render and five songs went
through, then free memory dipped below what LTX needs beside the other tenants of the host. The
adapter refused the load, gpud counted the refusal as a failed load, and every queued render
failed in turn, seconds apart, before memory could come back.

  LV1  a media load the engine defers (LoadStatus.deferred: not enough free memory right now)
       keeps its job queued and is retried on a doubling backoff: no load_failed, no failed job,
       no cascade through the queue, and the queued jobs run once memory frees
  LV2  a deferral is bounded: past Policy.load_defer_max_s the job the load was for fails once
       (load_failed), its baton fires once, and the next job waits afresh instead of failing at
       the same moment
  LV3  a brain whose load is deferred still moves down the ladder (deferral is for media only)
  LV4  the mlx-serve adapter marks memory-headroom and model-slot refusals deferred, and an
       incomplete pack or an unmeasurable memory reading not
  LV5  `verdict JOB retract` undoes a verdict given by mistake (it happened live): the item is back
       in the review queue and P(ship) is what it was; a shipped or pruned item refuses

Everything except LV4 runs on the virtual clock against the fake engines.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import unittest

from foreman import ledger
from foreman.policy import Policy
from foreman.sim import ScriptedBrain, ScriptedPlanner, Sim

try:  # discovered with -s tests (tests/ on sys.path) or run from the project root
    from test_brain_return import SimCase, calls, submit_render
    from test_mlx_serve_adapter import ACE, BRAIN, FALLBACK, KOKORO, EngineCase, make_engine, wait_for
    from test_planner import SPEECH, PlannerCase, prop
except ImportError:  # pragma: no cover
    from tests.test_brain_return import SimCase, calls, submit_render
    from tests.test_mlx_serve_adapter import ACE, BRAIN, FALLBACK, KOKORO, EngineCase, make_engine, wait_for
    from tests.test_planner import SPEECH, PlannerCase, prop

PRIMARY, FALLBACK_SIM, VISION, VIDEO = "qwen38-flash-next", "qwen36-35b-a3b", "qwen38-27b", "ltx-2.5"
# The small vision brain is the only one, so it stays resident beside the video model (as the 27B
# does on the Studio) and nothing but the video model's own load is in play.
SMALL_BRAIN = Policy(text_ladder=(VISION,), vision_brain=VISION, co_resident_brain=VISION)


class LV1DeferredLoadWaits(SimCase):
    capacity = 256.0

    def test_deferred_media_load_keeps_the_queue_and_runs_once_memory_frees(self):
        sim = self.make(policy=SMALL_BRAIN)
        self.settle(sim)
        sim.mlx.script_loads(VIDEO, "defer", "defer", "defer", "defer", "ok")
        t0 = sim.clock.now()
        jobs = [submit_render(sim.sup) for _ in range(3)]
        self.settle(sim, 7200)
        for j in jobs:
            job = sim.store.job(j)
            self.assertEqual((job.state, job.attempt), ("succeeded", 1), job.outcome)
        self.assertEqual(sim.store.events("load_failed"), [], "a deferred load is not a failed load")
        self.assertEqual([j.id for j in sim.store.jobs() if j.state == "failed"], [])
        loads = calls(sim.mlx, "load", VIDEO)
        self.assertEqual(len(loads), 5, "retried on the backoff (15, 30, 60, 120 s), not on every tick")
        gaps = [round(b[0] - a[0]) for a, b in zip(loads, loads[1:])]
        for gap, backoff in zip(gaps, (15, 30, 60, 120)):
            self.assertGreaterEqual(gap, backoff)
            self.assertLessEqual(gap, backoff + 3)
        deferred = sim.store.events("load_deferred")
        self.assertEqual([e["data"]["deferrals"] for e in deferred], [1, 2, 4], "logged on a log scale")
        loaded = [e for e in sim.store.events("model_loaded") if e["data"]["model"] == VIDEO]
        self.assertGreaterEqual(loaded[0]["data"]["deferred_s"], 15 + 30 + 60 + 120)
        self.assertGreaterEqual(loaded[0]["at"] - t0, 225)
        self.assertFalse((sim.store.state()["ladder"] or {}).get("deferred"), "the wait ends with the load")
        unloads = [c for c in sim.mlx.calls if c[1] == "unload"]
        self.assertEqual([c for c in unloads if c[2] == VISION], [], "the co-resident brain is never touched")
        self.assert_invariants(sim)

    def test_waiting_media_job_does_not_block_other_work(self):
        sim = self.make(policy=SMALL_BRAIN)
        self.settle(sim)
        sim.mlx.script_loads(VIDEO, *(["defer"] * 6), "ok")
        first = submit_render(sim.sup)
        self.assertTrue(sim.run_until(lambda: sim.store.events("load_deferred"), 60))
        other = submit_render(sim.sup, model="flux2-klein")  # another model fits beside the brain
        self.assertTrue(sim.run_until(lambda: sim.store.job(other).state == "succeeded", 600),
                        "work for another model runs while the video model waits for memory")
        self.assertEqual(sim.store.job(first).state, "queued")
        self.settle(sim, 7200)
        self.assertEqual(sim.store.job(first).state, "succeeded")
        self.assertEqual(sim.store.events("load_failed"), [])
        self.assert_invariants(sim)


class LV2DeferralIsBounded(SimCase):
    capacity = 256.0

    def test_a_wait_past_the_limit_fails_one_job_and_the_next_waits_afresh(self):
        policy = dataclasses.replace(SMALL_BRAIN, load_defer_max_s=600.0)
        sim = self.make(policy=policy, brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        sim.mlx.script_loads(VIDEO, *(["defer"] * 200))
        t0 = sim.clock.now()
        a = submit_render(sim.sup)
        b = submit_render(sim.sup)
        self.assertTrue(sim.run_until(lambda: sim.store.job(a).state != "queued", 3600))
        fail_a = sim.clock.now()
        self.assertEqual(sim.store.job(b).state, "queued", "no cascade: the next job waits afresh")
        self.assertTrue(sim.run_until(lambda: sim.store.job(b).state != "queued", 3600))
        fail_b = sim.clock.now()
        self.settle(sim, 3600)
        for j in (a, b):
            self.assertEqual(sim.store.job(j).state, "failed")
            o = sim.store.job(j).outcome
            self.assertEqual(o["reason"], "load_failed")
            self.assertIn("deferred for", o["detail"])
        self.assertGreaterEqual(fail_a - t0, 600)
        self.assertLessEqual(fail_a - t0, 600 + 300 + 5)
        self.assertGreaterEqual(fail_b - fail_a, 600)
        self.assertEqual(len(sim.store.events("load_failed")), 2)
        self.assertLessEqual(len(calls(sim.mlx, "load", VIDEO)), 16, "about 7 tries per wait, not one per tick")
        recovers = [j for j in sim.store.jobs() if j.kind == "recover_render"]
        self.assertEqual(len(recovers), 2, "each failed job's baton fires once")
        self.assert_invariants(sim)


class LV3BrainDeferralMovesTheLadder(SimCase):
    capacity = 256.0

    def test_a_brain_refused_for_memory_still_steps_down_the_ladder(self):
        sim = self.make()
        sim.mlx.script_loads(PRIMARY, "defer", "defer")
        self.assertTrue(sim.run_until(lambda: FALLBACK_SIM in sim.mlx.resident_models(), 600))
        self.assertEqual(sim.store.events("load_deferred"), [], "brains are never kept waiting")
        self.assertTrue(sim.store.events("brain_ladder_step"))
        self.assertEqual(len([e for e in sim.store.events("load_failed") if e["data"]["model"] == PRIMARY]), 2)


class LV4AdapterMarksDeferrals(EngineCase):

    def poll(self, eng, model):
        t = eng.load(model)
        wait_for(lambda: eng.poll_load(t).state != "loading")
        return eng.poll_load(t)

    def test_memory_and_slot_refusals_are_deferred(self):
        low = make_engine(self.fake, self.tmp, mem_probe=lambda: 45.0)
        try:
            st = self.poll(low, ACE)   # 45 GB free - 8 GB < 40 GB headroom
            self.assertEqual((st.state, st.deferred), ("failed", True))
            self.assertIn("insufficient headroom", st.error)
        finally:
            low.close()
        blind = make_engine(self.fake, self.tmp, mem_probe=lambda: None)
        try:
            st = self.poll(blind, ACE)
            self.assertEqual((st.state, st.deferred), ("failed", False), "an unknown reading is not 'later'")
        finally:
            blind.close()
        self.load(KOKORO)
        self.load(BRAIN)                # the user's model + kokoro + brain = mlx-serve's 3 slots
        st = self.poll(self.engine, ACE)
        self.assertEqual((st.state, st.deferred), ("failed", True))
        self.assertIn("max-resident-models", st.error)
        ok = self.poll(self.engine, KOKORO)
        self.assertEqual((ok.state, ok.deferred), ("ready", False))

    def test_an_incomplete_pack_is_refused_not_deferred(self):
        models = os.path.join(self.tmp, "models")
        pack = os.path.join(models, *FALLBACK.split("/"))
        os.makedirs(pack)
        for name in ("config.json", "weights.safetensors.partial"):
            open(os.path.join(pack, name), "w").close()
        eng = make_engine(self.fake, self.tmp, model_dir=models)
        try:
            st = self.poll(eng, FALLBACK)
            self.assertEqual((st.state, st.deferred), ("failed", False))
            self.assertIn("still downloading", st.error)
        finally:
            eng.close()


class LV5RetractAVerdict(PlannerCase):
    def cli(self, *args):
        from foreman import cli
        out = io.StringIO()
        old = os.environ.get("FOREMAN_HOME")
        os.environ["FOREMAN_HOME"] = self.home
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = cli.main(list(args))
        finally:
            if old is None:
                os.environ.pop("FOREMAN_HOME", None)
            else:
                os.environ["FOREMAN_HOME"] = old
        return code, out.getvalue()

    def test_a_mistaken_verdict_can_be_retracted_until_it_shipped(self):
        self.write(SPEECH)
        brain = ScriptedPlanner({"proposals": [prop("t.speech", brief="first script"),
                                               prop("t.speech", brief="second script"),
                                               prop("t.speech", brief="third script")]})
        sim = Sim(os.path.join(self.home, "foreman.db"), brain=brain,
                  planner=self.planner(interval_s=1e6, min_gap_s=1e9), artifacts_dir=os.path.join(self.home, "artifacts"))
        self.addCleanup(lambda: sim.sup.close())
        self.assertTrue(sim.run_until(lambda: len(ledger.review_queue(sim.store)) == 2, 3600))
        a, b = ledger.review_queue(sim.store)
        code, out = self.cli("verdict", str(a["job_id"]), "accept", "--minutes", "3")
        self.assertIn("P(ship) is now 0.82", out)
        code, out = self.cli("verdict", str(a["job_id"]), "retract", "--note", "meant the other one")
        self.assertEqual(code, 0, out)
        self.assertIn("P(ship) is now 0.80", out, "the prior again: the verdict is forgotten")
        it = ledger.item_for_job(sim.store, a["job_id"])
        self.assertEqual((it["state"], it["verdict"], it["verdict_at"], it["verdict_minutes"]), ("review", None, None, None))
        ev = sim.store.events("verdict_retracted")
        self.assertEqual([(e["job_id"], e["data"]["retracted"], e["data"]["note"]) for e in ev],
                         [(a["job_id"], "accept", "meant the other one")])
        code, out = self.cli("verdict", str(a["job_id"]), "retract")
        self.assertEqual(code, 2, "nothing to retract on an item awaiting a verdict")
        self.assertIn("only an accepted", out)
        code, out = self.cli("verdict", str(b["job_id"]), "reject")
        code, out = self.cli("verdict", str(b["job_id"]), "retract")
        self.assertEqual(code, 0, out)
        self.assertEqual(ledger.item_for_job(sim.store, b["job_id"])["state"], "review")
        code, out = self.cli("verdict", str(b["job_id"]), "accept")
        code, out = self.cli("ship", str(b["job_id"]))
        code, out = self.cli("verdict", str(b["job_id"]), "retract")
        self.assertEqual(code, 2, "a shipped item keeps its verdict")
        self.assertIn("shipped", out)
        self.assertEqual(sorted(i["job_id"] for i in ledger.review_queue(sim.store)), [a["job_id"]])
        code, out = self.cli("verdict", str(a["job_id"]), "reject")
        self.assertEqual(code, 0, out)
        with sim.store.tx():
            ledger.update_item(sim.store, ledger.item_for_job(sim.store, a["job_id"])["id"], pruned_at=sim.sup._now())
        code, out = self.cli("verdict", str(a["job_id"]), "retract")
        self.assertEqual(code, 2)
        self.assertIn("pruned", out)


if __name__ == "__main__":
    unittest.main()
