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

The second round fixes three issues the live run found:

  LV6  provenance: a catalog model someone else loaded is never unloaded by any gpud path (make
       room, slot caps, idle release, cleanup after a timeout, ladder failures, engine-lost orphan
       cleanup), always counted in the memory plan; a model gpud saw go away and that came back
       without a gpud load is someone else's; Policy.foreign_models is never loaded or claimed
  LV7  provenance is durable: it survives a gpud restart, resets when the server restarts (its
       instance id changed), a load intent written before the request lets a gpud that died with
       the request in flight claim the model, an intent whose model never appears expires, and a
       database from before provenance is seeded from its event log
  LV8  someone else's resident brain serves gpud's brain jobs read-only (Policy.use_foreign_brains),
       or not at all when that is off
  LV9  calibrated memory plans (the Studio's measured sizes and load times), the configurable
       headroom, and planning with the largest observed size when it exceeds the catalog's
  LV10 every failure reason is classified infrastructure or content, from the lineage
  LV11 weekly caps, stock and the day's review minutes ignore infrastructure failures, within a
       per-template daily budget, so a failing environment cannot recycle slots forever; items the
       old code settled are classified on the next upkeep (the live load-cascade's slots come back)
  LV12 the report's failed column split infra/content, GPU sinks that exclude items still awaiting
       review, and the CLI's ownership and count notes
  LV13 the mlx-serve adapter: instance id from /metrics, sizes forgotten on a restart, HTTP 4xx as
       a rejected request; gpud on the fake server never unloads a catalog model another client
       loaded, before or after a server restart

Everything except LV4 and LV13 runs on the virtual clock against the fake engines.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import itertools
import json
import os
import re
import tempfile
import time
import unittest

from foreman import ledger, planner as planner_mod, report, studio
from foreman.adapters.mlx_serve import parse_start_time, rejected_status
from foreman.engine import LoadStatus
from foreman.policy import Policy
from foreman.sim import CrashAt, ScriptedBrain, ScriptedPlanner, Sim, garbage, hang, reject
from foreman.store import Store
from foreman.supervisor import ENGINE_FAILURE_REASONS, Supervisor

try:  # discovered with -s tests (tests/ on sys.path) or run from the project root
    from test_adversarial import make_reachable, make_unreachable, tolerant_brain_check
    from test_brain_return import SimCase, calls, continuations_of, submit_render
    from test_mlx_serve_adapter import (ACE, BRAIN, FALLBACK, FOREIGN, KOKORO, LTX, EngineCase, ManualClock, TmpCase,
                                        make_engine, make_fake, spec, wait_for)
    from test_planner import SPEECH, PlannerCase, prop, tpl
except ImportError:  # pragma: no cover
    from tests.test_adversarial import make_reachable, make_unreachable, tolerant_brain_check
    from tests.test_brain_return import SimCase, calls, continuations_of, submit_render
    from tests.test_mlx_serve_adapter import (ACE, BRAIN, FALLBACK, FOREIGN, KOKORO, LTX, EngineCase, ManualClock,
                                              TmpCase, make_engine, make_fake, spec, wait_for)
    from tests.test_planner import SPEECH, PlannerCase, prop, tpl

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


# ===================================================================================== round two
#
# Simulator names (foreman.policy.CATALOG): the primary brain qwen38-flash-next (100 GB), the
# fallback qwen36-35b-a3b (21 GB), the vision brain qwen38-27b (18 GB), video ltx-2.5 (150 GB),
# music ace-step (10 GB), speech qwen3-tts (5 GB), image flux2-klein (14 GB).

MUSIC_M, TTS_M, IMAGE_M = "ace-step", "qwen3-tts", "flux2-klein"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def submit_media(sup, model, kind=None, expected_s=20.0, **payload):
    kind = kind or {"ace-step": "compose_music", "qwen3-tts": "speak", "flux2-klein": "draw"}[model]
    return sup.submit(kind, model, {"goal": f"{kind} something", "expected_s": expected_s, **payload},
                      {"kind": "review"}, {"kind": "recover"})


def unloads(engine, model=None):
    return [c for c in engine.calls if c[1] == "unload" and (model is None or c[2] == model)]


class OwnershipCase(SimCase):
    capacity = 256.0

    def owners(self, sim):
        return {m: r["state"] for m, r in sim.store.owners().items()}

    def assert_never_unloaded(self, sim, *models):
        for m in models:
            self.assertEqual(unloads(sim.mlx, m), [], f"gpud unloaded {m}, which someone else loaded")


class LV6ForeignCatalogModelsAreNeverUnloaded(OwnershipCase):

    def test_make_room_waits_rather_than_unload_a_catalog_brain_someone_else_loaded(self):
        # Live Studio: the fallback brain is in the catalog, but MLX-Serve.app loaded it as its
        # default. A model gpud did not load is pinned and counted, never a make-room victim.
        sim = self.make(capacity=176.0)  # 160 GB usable: video (150) + the foreign fallback (21) do not fit
        self.settle(sim)
        sim.mlx.load_as_other_client(FALLBACK_SIM)
        sim.run_until(lambda: False, 3)
        appeared = [e["data"] for e in sim.store.events("model_appeared") if e["data"]["model"] == FALLBACK_SIM]
        self.assertTrue(appeared and appeared[0]["foreign"], appeared)
        self.assertNotIn(FALLBACK_SIM, self.owners(sim))
        job = submit_render(sim.sup)
        sim.run_until(lambda: False, 300)
        self.assertEqual(sim.store.job(job).state, "queued", "it waits: the room it needs is someone else's")
        short = [e["data"] for e in sim.store.events("insufficient_memory")]
        self.assertTrue(short and FALLBACK_SIM in short[-1]["pinned"], short)
        self.assertIn(PRIMARY, sim.mlx.resident_models(), "nothing is unloaded for a target that cannot fit")
        sim.mlx._models.discard(FALLBACK_SIM)  # the other client unloads its model
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assert_never_unloaded(sim, FALLBACK_SIM)
        self.assertTrue(unloads(sim.mlx, PRIMARY), "gpud's own brain is still gpud's to unload")
        self.assert_invariants(sim)

    def test_the_slot_cap_frees_gpuds_own_models_never_someone_elses(self):
        sim = self.make()
        sim.mlx.max_resident_models = 3
        self.settle(sim)                         # gpud's primary
        sim.mlx.load_as_other_client(FALLBACK_SIM)
        sim.mlx.load_as_other_client(MUSIC_M)    # three slots taken, two of them someone else's
        job = submit_media(sim.sup, TTS_M)
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assertTrue(unloads(sim.mlx, PRIMARY), "gpud gave up its own brain's slot")
        self.assert_never_unloaded(sim, FALLBACK_SIM, MUSIC_M)
        self.assertLessEqual(len(sim.mlx.resident_models()), 3)

    def test_idle_release_and_mode_leave_a_foreign_media_model_alone(self):
        sim = self.make(policy=Policy(idle_media_unload_s=120.0))
        self.settle(sim)
        sim.mlx.load_as_other_client(MUSIC_M)
        job = submit_media(sim.sup, TTS_M)
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assertTrue(sim.run_until(lambda: TTS_M not in sim.mlx.resident_models(), 600), "gpud's own is released")
        sim.run_until(lambda: False, 600)
        self.assertIn(MUSIC_M, sim.mlx.resident_models())
        self.assert_never_unloaded(sim, MUSIC_M)
        self.assertEqual(sim.store.state()["mode"], "brain", "someone else's media model is not gpud's media")

    def test_a_job_runs_read_only_on_a_foreign_model_and_its_timeout_cleanup_skips_the_unload(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        sim.mlx.load_as_other_client(TTS_M)
        sim.mlx.script_jobs("speak", hang(0.3))
        job = submit_media(sim.sup, TTS_M)
        self.settle(sim)
        j = sim.store.job(job)
        self.assertEqual((j.state, j.outcome["reason"], j.outcome["cleanup"]), ("failed", "timeout", "foreign"))
        self.assertEqual(calls(sim.mlx, "load", TTS_M), [], "gpud never loads a model that is resident already")
        self.assert_never_unloaded(sim, TTS_M)
        self.assertIn(TTS_M, sim.mlx.resident_models())
        self.assertEqual(len(continuations_of(sim.store, job)), 1, "the baton still fires once")
        self.assertTrue(sim.store.events("unload_refused"))
        self.assert_invariants(sim)

    def test_the_ladder_skips_a_brain_that_does_not_fit_beside_someone_elses_models(self):
        sim = self.make(capacity=136.0)  # 120 GB usable
        sim.mlx.load_as_other_client(MUSIC_M)
        sim.mlx.load_as_other_client(IMAGE_M)  # 24 GB pinned: the 100 GB primary no longer fits
        self.settle(sim)
        self.assertEqual(calls(sim.mlx, "load", PRIMARY), [])
        self.assertIn(FALLBACK_SIM, sim.mlx.resident_models())
        self.assertEqual(self.owners(sim), {FALLBACK_SIM: "owned"})
        self.assert_never_unloaded(sim, MUSIC_M, IMAGE_M)

    def test_aborting_a_stuck_brain_load_unloads_only_gpuds_own_load(self):
        sim = self.make()
        sim.mlx.load_as_other_client(MUSIC_M)
        sim.mlx.script_loads(PRIMARY, "hang", "hang")
        self.settle(sim, 3600)
        self.assertEqual(len(unloads(sim.mlx, PRIMARY)), 2, "each stuck load of its own is aborted")
        self.assertIn(FALLBACK_SIM, sim.mlx.resident_models(), "the ladder moved on")
        self.assert_never_unloaded(sim, MUSIC_M)
        self.assertEqual(self.owners(sim), {FALLBACK_SIM: "owned"})

    def test_engine_lost_orphan_cleanup_leaves_a_foreign_model(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        sim.mlx.load_as_other_client(TTS_M)
        job = submit_media(sim.sup, TTS_M, expected_s=600.0)  # no-progress limit 240 s > engine_lost_s 120 s
        self.running(sim, job)
        tolerant_brain_check(sim)
        make_unreachable(sim.mlx)
        t_down = sim.clock.now()
        sim.at(t_down + 300, lambda: make_reachable(sim.mlx))
        self.settle(sim, 3600)
        j = sim.store.job(job)
        self.assertEqual((j.state, j.outcome["cleanup"]), ("failed", "engine_unreachable"))
        self.assertEqual(sim.store.state()["orphans"], [], "the orphan is settled")
        self.assertIn(TTS_M, sim.mlx.resident_models(), "someone else's model outlives gpud's orphaned run on it")
        self.assert_never_unloaded(sim, TTS_M)

    def test_a_model_gpud_saw_go_away_is_someone_elses_when_it_comes_back(self):
        sim = self.make(policy=Policy(idle_media_unload_s=120.0))
        self.settle(sim)
        first = submit_media(sim.sup, MUSIC_M)
        self.assertTrue(sim.run_until(lambda: sim.store.job(first).state == "succeeded", 600))
        self.assertEqual(self.owners(sim).get(MUSIC_M), "owned")
        sim.mlx.evict(MUSIC_M)            # mlx-serve dropped it (LRU, memory pressure)
        sim.run_until(lambda: False, 2)
        self.assertNotIn(MUSIC_M, self.owners(sim))
        dropped = [e["data"] for e in sim.store.events("ownership_dropped") if e["data"]["model"] == MUSIC_M]
        self.assertEqual([d["reason"] for d in dropped], ["no longer resident"])
        sim.mlx.load_as_other_client(MUSIC_M)  # ...and another client loads it again
        second = submit_media(sim.sup, MUSIC_M)
        self.settle(sim)
        sim.run_until(lambda: False, 900)
        self.assertEqual(sim.store.job(second).state, "succeeded", "it ran read-only on their copy")
        self.assertEqual(len(calls(sim.mlx, "load", MUSIC_M)), 1, "no second load of a resident model")
        self.assertEqual(unloads(sim.mlx, MUSIC_M), [], "never unloaded once it is someone else's")
        self.assertIn(MUSIC_M, sim.mlx.resident_models())

    def test_foreign_models_listed_by_the_owner_are_never_loaded_unloaded_or_claimed(self):
        sim = self.make(capacity=176.0, policy=Policy(foreign_models=(FALLBACK_SIM,)))
        sim.mlx.load_as_other_client(FALLBACK_SIM)
        self.settle(sim)
        self.assertIn(PRIMARY, sim.mlx.resident_models())
        job = submit_render(sim.sup)
        sim.run_until(lambda: False, 120)
        self.assertEqual(sim.store.job(job).state, "queued")
        self.assert_never_unloaded(sim, FALLBACK_SIM)
        sim.mlx._models.discard(FALLBACK_SIM)
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        other = self.make(policy=Policy(foreign_models=(FALLBACK_SIM,)))
        other.mlx.script_loads(PRIMARY, "fail", "fail")
        self.assertTrue(other.run_until(lambda: other.store.state()["safe"], 3600),
                        "with the primary failing and the fallback not gpud's to load, the ladder ends")
        self.assertEqual(calls(other.mlx, "load", FALLBACK_SIM), [])
        self.assertTrue(other.pages)


class LV7ProvenanceIsDurable(OwnershipCase):

    def test_ownership_survives_a_gpud_restart(self):
        sim = self.make(policy=Policy(idle_media_unload_s=120.0))
        self.settle(sim)
        job = submit_media(sim.sup, MUSIC_M)
        self.assertTrue(sim.run_until(lambda: sim.store.job(job).state == "succeeded", 600))
        sim.restart(downtime_s=30)
        self.assertTrue(sim.sup._ours(MUSIC_M), "read back from model_owners")
        self.assertTrue(sim.run_until(lambda: MUSIC_M not in sim.mlx.resident_models(), 900),
                        "after the restart it is still gpud's to release")
        self.assertEqual(sim.store.events("engine_instance_changed"), [])

    def test_ownership_resets_when_the_server_restarts_while_gpud_is_down(self):
        sim = self.make(policy=Policy(idle_media_unload_s=120.0))
        self.settle(sim)
        job = submit_media(sim.sup, MUSIC_M)
        self.assertTrue(sim.run_until(lambda: sim.store.job(job).state == "succeeded", 600))
        sim.sup.close()                       # gpud is down...
        sim.mlx.server_restart()              # ...the server restarts...
        sim.mlx.load_as_other_client(MUSIC_M)  # ...and someone loads the same model before gpud returns
        sim.mlx.load_as_other_client(PRIMARY)
        sim.restart(downtime_s=60)
        changed = sim.store.events("engine_instance_changed")
        self.assertEqual(len(changed), 1)
        self.assertIn(MUSIC_M, changed[0]["data"]["dropped_ownership"])
        self.assertEqual(self.owners(sim), {})
        sim.run_until(lambda: False, 900)
        self.assertIn(MUSIC_M, sim.mlx.resident_models())
        self.assertEqual(len(unloads(sim.mlx, MUSIC_M)), 0, "continuously 'resident' to gpud, but not gpud's")
        self.assertEqual(calls(sim.mlx, "load", PRIMARY)[1:], [], "their primary serves read-only; gpud loads none")

    def test_ownership_resets_when_the_server_restarts_between_two_ticks(self):
        sim = self.make(policy=Policy(idle_media_unload_s=120.0))
        self.settle(sim)
        job = submit_media(sim.sup, MUSIC_M)
        self.assertTrue(sim.run_until(lambda: sim.store.job(job).state == "succeeded", 600))

        def restart_and_reload():
            sim.mlx.server_restart()
            sim.mlx.load_as_other_client(MUSIC_M)
            sim.mlx.load_as_other_client(PRIMARY)
        sim.at(sim.clock.now() + 1, restart_and_reload)
        sim.run_until(lambda: False, 900)
        self.assertTrue(sim.store.events("engine_instance_changed"))
        self.assertEqual(unloads(sim.mlx, MUSIC_M), [])
        self.assertEqual(self.owners(sim), {})

    def test_a_crash_between_the_intent_and_the_load_request_leaves_nothing_misclassified(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        job = submit_media(sim.sup, MUSIC_M)

        class CrashThenCancel(CrashAt):
            def __call__(hook, point):
                if point == hook.point and not hook.fired:
                    sim.at(sim.clock.now() + 0.5, lambda: sim.sup.cancel_job(job, "nobody wants it now"))
                super().__call__(point)
        sim.arm(CrashThenCancel("after_load_intent"))
        self.settle(sim)
        self.assertEqual(sim.restarts, 1)
        self.assertEqual(calls(sim.mlx, "load", MUSIC_M), [], "the request was never sent")
        self.assertEqual(self.owners(sim).get(MUSIC_M), "loading", "the intent waits out the load timeout")
        sim.run_until(lambda: MUSIC_M not in self.owners(sim), 300)
        expired = [e["data"]["reason"] for e in sim.store.events("ownership_dropped") if e["data"]["model"] == MUSIC_M]
        self.assertEqual(expired, ["load intent expired: the model never appeared"])
        sim.mlx.load_as_other_client(MUSIC_M)
        sim.run_until(lambda: False, 5)
        self.assertFalse(sim.sup._ours(MUSIC_M), "a load after the intent expired is someone else's")

    def test_a_load_in_flight_when_gpud_died_is_claimed_when_the_model_appears(self):
        # The real adapter forgets its tickets with the process: the next gpud polls the old one as
        # "unknown". The intent written before the request lets it claim the model anyway.
        sim = self.make(policy=Policy(idle_media_unload_s=120.0))
        self.settle(sim)
        eng = sim.mlx
        stale = set()
        orig_poll = eng.poll_load

        def poll_load(ticket):
            if ticket in stale:
                return LoadStatus("failed", "unknown load ticket (issued by an earlier gpud process)")
            return orig_poll(ticket)
        eng.poll_load = poll_load
        job = submit_media(sim.sup, MUSIC_M)
        sim.crash_downtime_s = 30.0  # the load (10 s) completes while gpud is down
        sim.arm(CrashAt("after_load_requested"))
        self.assertTrue(sim.run_until(lambda: sim.restarts == 1, 120))
        stale.update(eng._loads)
        self.settle(sim)
        self.assertEqual(sim.store.job(job).state, "succeeded")
        self.assertEqual(len(calls(eng, "load", MUSIC_M)), 1, "claimed, not loaded a second time")
        claimed = [e["data"] for e in sim.store.events("model_claimed")]
        self.assertEqual([c["model"] for c in claimed], [MUSIC_M])
        self.assertEqual(self.owners(sim).get(MUSIC_M), "owned")
        self.assertTrue(sim.run_until(lambda: MUSIC_M not in eng.resident_models(), 900),
                        "gpud's own after all: released when idle")

    def test_a_load_the_engine_answered_as_failed_ends_the_intent_at_once(self):
        sim = self.make(brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        sim.mlx.script_loads(MUSIC_M, "fail", "fail")
        job = submit_media(sim.sup, MUSIC_M)
        self.assertTrue(sim.run_until(lambda: sim.store.job(job).state == "failed", 600))
        self.assertNotIn(MUSIC_M, self.owners(sim), "a definite failure leaves no intent to claim with")
        sim.mlx.load_as_other_client(MUSIC_M)
        sim.run_until(lambda: False, 30)
        self.assertFalse(sim.sup._ours(MUSIC_M))
        self.settle(sim)
        self.assert_never_unloaded(sim, MUSIC_M)

    def test_a_database_from_before_provenance_is_seeded_from_its_event_log(self):
        sim = self.make(capacity=176.0)
        self.settle(sim)                       # gpud loaded the primary (model_loaded)
        sim.mlx.load_as_other_client(FALLBACK_SIM)
        sim.run_until(lambda: False, 3)        # someone else's fallback (model_appeared)
        with sim.store.tx():                   # the database as the previous version left it
            sim.store.conn.execute("DELETE FROM model_owners")
            sim.store.set_state(owners_init=0)
        sim.restart()
        seeded = sim.store.events("ownership_seeded")
        self.assertEqual([(e["data"]["owned"], e["data"]["foreign"]) for e in seeded], [([PRIMARY], [FALLBACK_SIM])])
        self.assertEqual(self.owners(sim), {PRIMARY: "owned"})
        sim.restart()
        self.assertEqual(len(sim.store.events("ownership_seeded")), 1, "seeded once")
        job = submit_render(sim.sup)
        sim.run_until(lambda: False, 120)
        self.assertEqual(sim.store.job(job).state, "queued")
        self.assert_never_unloaded(sim, FALLBACK_SIM)


class LV8ForeignBrainReadOnly(OwnershipCase):

    def test_someone_elses_brain_serves_brain_jobs_and_is_never_loaded_or_unloaded(self):
        sim = self.make(capacity=300.0)
        sim.mlx.load_as_other_client(PRIMARY)
        job = sim.sup.submit("plan_video", payload={"goal": "a fox"})
        self.settle(sim)
        self.assertEqual(sim.store.job(job).model, PRIMARY)
        self.assertEqual(calls(sim.mlx, "load", PRIMARY), [], "their copy is used, not reloaded")
        render = [j for j in sim.store.jobs() if j.kind == "render_video"]
        self.assertEqual([j.state for j in render], ["succeeded"], "the render fits beside their brain")
        self.assert_never_unloaded(sim, PRIMARY)
        self.assertNotIn(PRIMARY, self.owners(sim))
        self.assert_invariants(sim)

    def test_with_use_foreign_brains_off_gpud_brings_its_own_brain(self):
        sim = self.make(capacity=300.0, policy=Policy(use_foreign_brains=False))
        sim.mlx.load_as_other_client(PRIMARY)
        job = sim.sup.submit("plan_video", payload={"goal": "a fox"})
        self.settle(sim)
        self.assertNotEqual(sim.store.job(job).model, PRIMARY)
        self.assertEqual(calls(sim.mlx, "load", PRIMARY), [])
        self.assertTrue(calls(sim.mlx, "load", FALLBACK_SIM), "the ladder brings gpud's own brain")
        self.assert_never_unloaded(sim, PRIMARY)


class LV9CalibratedPlans(OwnershipCase):

    def test_the_studio_catalog_carries_the_measured_sizes_and_load_times(self):
        cat = studio.catalog()
        got = {m: (s.size_gb, s.load_s) for m, s in cat.items()}
        self.assertEqual(got, {studio.BRAIN_PRIMARY: (20, 10), studio.BRAIN_FALLBACK: (77, 70),
                               studio.KOKORO: (1, 3), studio.ACE_STEP: (6, 5), studio.LTX: (40, 10)})
        p = studio.policy()
        self.assertEqual(p.usable_gb, 160.0)
        self.assertLessEqual(cat[studio.LTX].size_gb + cat[studio.BRAIN_PRIMARY].size_gb
                             + cat[studio.BRAIN_FALLBACK].size_gb, p.usable_gb,
                             "LTX, the 27B and someone else's Flash-Next fit the plan together")
        with tempfile.TemporaryDirectory() as home:
            cfg = studio.load_config(env={"FOREMAN_HOME": home, "FOREMAN_MIN_HEADROOM_GB": "24"})
            self.assertEqual((cfg.min_headroom_gb, cfg.policy().usable_gb), (24.0, 176.0))
            with open(os.path.join(home, "config.json"), "w") as f:
                json.dump({"foreign_models": [studio.BRAIN_FALLBACK], "use_foreign_brains": False}, f)
            cfg = studio.load_config(env={"FOREMAN_HOME": home})
            self.assertEqual((cfg.policy().foreign_models, cfg.policy().use_foreign_brains),
                             ((studio.BRAIN_FALLBACK,), False))
            with open(os.path.join(home, "config.json"), "w") as f:
                json.dump({"foreign_models": "not a list"}, f)
            with self.assertRaises(ValueError):
                studio.load_config(env={"FOREMAN_HOME": home})

    def test_a_model_bigger_than_its_catalog_size_is_planned_with_its_observed_size(self):
        # 170 GB usable: video (150 in the catalog) + the 18 GB vision brain fit the static plan;
        # a video model that really takes 160 GB leaves no room for the brain beside it.
        def renders(observed):
            sim = self.make(capacity=186.0)
            if observed:
                sim.mlx.observed_gb[VIDEO] = observed
            self.settle(sim)
            beside = []
            for _ in range(2):
                job = submit_render(sim.sup)
                self.running(sim, job)
                beside.append(VISION in sim.mlx.resident_models())
                self.settle(sim)
                self.assertEqual(sim.store.job(job).state, "succeeded")
            self.assertEqual(sim.host.oom, [])
            return sim, beside
        control, beside = renders(None)
        self.assertEqual(beside, [True, True], "as planned from the catalog: the vision brain stays beside")
        self.assertEqual(control.store.events("model_size_observed"), [])
        sim, beside = renders(160.0)
        self.assertEqual(beside, [False, False], "planned with the observed 160 GB: no room for the brain")
        self.assertEqual(sim.sup._size(VIDEO), 160.0)
        seen = [e["data"] for e in sim.store.events("model_size_observed")]
        self.assertEqual([(d["model"], d["gb"], d["plan_gb"]) for d in seen], [(VIDEO, 160.0, 150)], "logged once")
        sim.sup.close()
        sim.mlx.server_restart()
        sim.restart()
        self.assertEqual(sim.sup._size(VIDEO), 150, "sizes seen on the old server instance are forgotten")

    def test_the_adapter_preflight_admits_ltx_with_88_gb_free_and_uses_a_larger_observed_size(self):
        fake = make_fake(**{"LTX-2.5-MLX-Serve-4bit": int(45e9)})
        self.addCleanup(fake.stop)
        with tempfile.TemporaryDirectory() as tmp:
            free = [88.0]
            eng = make_engine(fake, tmp, catalog=studio.catalog(), mem_probe=lambda: free[0])
            try:
                t = eng.load(LTX)  # 88 - 40 >= 40: before the calibration (60 GB) this waited for an hour
                wait_for(lambda: eng.poll_load(t).state != "loading")
                self.assertEqual(eng.poll_load(t).state, "ready", eng.poll_load(t).error)
                eng.resident_models()
                self.assertEqual(eng.plan_gb(LTX, studio.catalog()[LTX]), 45.0, "observed 45 GB > the 40 GB plan")
                self.assertTrue(eng.unload(LTX))
                free[0] = 84.0     # 84 - 40 >= 40, but 84 - 45 < 40
                t = eng.load(LTX)
                st = eng.poll_load(t)
                self.assertEqual((st.state, st.deferred), ("failed", True))
                self.assertIn("needs ~45 GB", st.error)
            finally:
                eng.close()


class LV10FailureClassification(unittest.TestCase):
    """Every way a planned item's lineage can fail, classified from the jobs themselves."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(os.path.join(tmp.name, "c.db"))
        self.addCleanup(self.store.close)

    def lineage(self, *media):
        """media: (state, reason, extra outcome fields, review) per media job, where review is
        None, ("brain", state, verdict) or ("human",)."""
        root, jobs = None, []
        for state, reason, extra, review in media:
            outcome = {"status": state, "reason": reason, **(extra or {})}
            jid, _ = self.store.insert_job(kind="render_video", requires="ltx-2.5", state=state, payload={},
                                           outcome=outcome, root_id=root, max_attempts=3, created_at=0.0)
            root = root or jid
            self.store.update_job(jid, root_id=root)
            self.store.add_continuation(jid, {"kind": "review"}, {"kind": "recover"})
            if review:
                if review[0] == "human":
                    cid, _ = self.store.insert_job(kind="human_review", requires="human", state="awaiting_human",
                                                   payload={}, root_id=root, max_attempts=1, created_at=0.0)
                else:
                    out = {"status": review[1], "reason": None if review[1] == "succeeded" else "timeout",
                           "artifact": {"decision": {"verdict": review[2]}}}
                    cid, _ = self.store.insert_job(kind="review_render", requires="brain", state=review[1],
                                                   payload={}, outcome=out, root_id=root, max_attempts=3,
                                                   created_at=0.0)
                self.store.mark_fired(jid, "success", cid, 0.0)
            jobs.append(self.store.job(jid))
        return jobs

    def cls(self, *media):
        return ledger.failure_class(self.store, self.lineage(*media))[0]

    def test_every_media_failure_reason(self):
        for reason in sorted(ledger.INFRA_REASONS):
            with self.subTest(reason=reason):
                self.assertEqual(self.cls(("failed", reason, None, None)), "infra")
        for reason in ("validator", "rejected"):
            with self.subTest(reason=reason):
                self.assertEqual(self.cls(("failed", reason, None, None)), "content")
        cases = {"gpud": "infra", "request": "content", "template_paused": "content", None: "content"}
        for by, want in cases.items():
            with self.subTest(cancelled_by=by):
                self.assertEqual(self.cls(("failed", "cancelled", {"cancelled_by": by} if by else {}, None)), want)
        self.assertEqual(self.cls(("quarantined", "load_failed", None, None)), "infra", "attempts used up on infra")
        self.assertEqual(self.cls(("failed", "something_new", None, None)), "content", "unknown reasons count")

    def test_reviews_and_mixed_lineages(self):
        for verdict in ("retry", "give_up", "reject"):
            with self.subTest(verdict=verdict):
                self.assertEqual(self.cls(("succeeded", None, None, ("brain", "succeeded", verdict))), "content")
        self.assertEqual(self.cls(("succeeded", None, None, ("brain", "quarantined", None))), "infra",
                         "the brain could not review: the environment's failure")
        self.assertEqual(self.cls(("succeeded", None, None, ("human",))), "content")
        self.assertEqual(self.cls(("failed", "validator", None, None), ("failed", "load_failed", None, None)), "content",
                         "a content failure anywhere in the lineage counts")
        self.assertEqual(self.cls(("failed", "interrupted", None, None), ("failed", "load_failed", None, None)), "infra")

    def test_every_failure_reason_gpud_can_record_is_classified(self):
        known = ledger.INFRA_REASONS | ledger.CONTENT_REASONS
        found = set()
        for name in ("supervisor.py", "planner.py"):
            with open(os.path.join(REPO, "foreman", name)) as f:
                text = f.read()
            found |= set(re.findall(r'failed\(\s*"([a-z_]+)"', text))
            found |= set(re.findall(r'"reason": "([a-z_]+)"', text))
        self.assertTrue({"load_failed", "interrupted", "timeout", "cancelled", "engine_unreachable"} <= found, found)
        self.assertEqual(found - known, set(), "a failure reason with no classification")
        self.assertLessEqual(set(ENGINE_FAILURE_REASONS), ledger.CONTENT_REASONS)
        self.assertEqual(ledger.INFRA_REASONS & ledger.CONTENT_REASONS, frozenset())

    def test_a_rejected_request_is_recorded_as_rejected_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = Sim(os.path.join(tmp, "r.db"), brain=ScriptedBrain(retry_failures=False))
            try:
                sim.mlx.script_jobs("render_video", reject())
                job = submit_render(sim.sup)
                self.assertTrue(sim.run_until_quiet(3600))
                o = sim.store.job(job).outcome
                self.assertEqual(o["reason"], "rejected")
                self.assertIn("HTTP 400", o["detail"])
            finally:
                sim.sup.close()


AD = tpl(SPEECH, id="t.ad", caps={"per_week": 2, "max_stock": 2, "two_week_use": 2}, review_minutes=10, value_usd=100)


class ItemsCase(PlannerCase):

    def make_sim(self, *loads, jobs=(), **over):
        self.n += 1
        tick = itertools.count(1)
        per_tick = over.pop("per_tick", 2)
        tid = over.pop("tid", "t.ad")

        def propose(_snap):
            n = next(tick)
            return {"proposals": [prop(tid, brief=f"ad script tick {n} item {k}") for k in range(per_tick)]}
        brain = ScriptedPlanner(policy=propose, retry_failures=False)
        cfg = dict(review_max_minutes_day=20.0, min_gap_s=60.0)
        cfg.update(over)
        sim = Sim(os.path.join(self.home, "foreman.db") if self.n == 1 else os.path.join(self.tmp, f"i{self.n}.db"),
                  brain=brain, planner=self.planner(**cfg))
        self.addCleanup(lambda: sim.sup.close())
        if loads:
            sim.mlx.script_loads("qwen3-tts", *loads)
        if jobs:
            sim.mlx.script_jobs("speak", *jobs)
        return sim

    def ctx(self, sim):
        templates, _ = sim.sup.planner.templates()
        return sim.sup.planner.context(sim.sup, sim.sup._now(), float(sim.clock.now()), templates)

    def reasons(self, sim):
        return [r["reason"] for r in ledger.rejections_since(sim.store, float("-inf"))]


class LV11InfraFailuresDoNotUseCaps(ItemsCase):

    def test_infra_failures_hand_their_slots_and_review_minutes_back(self):
        self.write(AD)
        sim = self.make_sim("fail", "fail", "fail", "fail")
        self.assertTrue(sim.run_until(lambda: len(ledger.items(sim.store)) >= 4, 4 * 3600))
        self.assertTrue(sim.run_until(lambda: "weekly_cap" in self.reasons(sim), 4 * 3600))
        self.assertTrue(sim.run_until(lambda: len(ledger.review_queue(sim.store)) == 2, 4 * 3600))
        its = ledger.items(sim.store)
        self.assertEqual(len(its), 4, "two failed for infrastructure, two more admitted in their place")
        failed, ok = its[:2], its[2:]
        for it in failed:
            self.assertEqual((it["state"], it["fail_class"], it["counted"]), ("failed", "infra", 0))
            self.assertTrue(it["count_note"].startswith("not counted: infrastructure failure"), it["count_note"])
        self.assertEqual([i["state"] for i in ok], ["review", "review"])
        ctx = self.ctx(sim)
        self.assertEqual((ctx["week"]["t.ad"], ctx["stock"]["t.ad"], ctx["review_minutes_today"]), (2, 2, 20.0),
                         "only the two produced items count: per_week, max_stock and the day's minutes")
        self.assertNotIn("review_minutes_today", self.reasons(sim)[:2])
        events = [e["data"] for e in sim.store.events("item_failed")]
        self.assertEqual([(d["fail_class"], d["counted"]) for d in events], [("infra", False), ("infra", False)])
        stats = ledger.template_stats(sim.store, sim.sup.planner.templates()[0], sim.sup._now())["t.ad"]
        self.assertEqual((stats["week"], stats["7d"]["failed_infra"], stats["7d"]["failed_content"]), (2, 2, 0))

    def test_content_failures_still_use_the_cap(self):
        self.write(AD)
        for behaviour in (garbage(), reject()):
            with self.subTest(behaviour=behaviour.mode):
                sim = self.make_sim(jobs=(behaviour, behaviour))
                self.assertTrue(sim.run_until(lambda: "weekly_cap" in self.reasons(sim), 4 * 3600))
                self.assertTrue(sim.run_until(lambda: len(ledger.items(sim.store, "failed")) == 2, 4 * 3600))
                its = ledger.items(sim.store)
                self.assertEqual([(i["state"], i["fail_class"], i["counted"]) for i in its],
                                 [("failed", "content", 1)] * 2)
                self.assertEqual(self.ctx(sim)["week"]["t.ad"], 2)

    def test_a_persistently_failing_environment_is_bounded_by_the_daily_budget(self):
        self.write(AD)
        sim = self.make_sim(*(["fail"] * 400))
        sim.run_until(lambda: False, 8 * 3600, 2.0)
        its = ledger.items(sim.store)
        self.assertEqual(len(its), 4, "per_week (2) uncounted, then two more that count, then the cap")
        self.assertEqual([i["counted"] for i in its], [0, 0, 1, 1])
        self.assertIn("are used up", its[2]["count_note"])
        self.assertIn("weekly_cap", self.reasons(sim))
        sim.run_until(lambda: False, 86400.0, 30.0)  # a new day: a fresh budget, but the week is full
        self.assertEqual(len(ledger.items(sim.store)), 4)
        media = [j for j in sim.store.jobs() if j.brainless]
        self.assertEqual(len(media), 4, "one media job per item (the recover brain gave up)")
        self.assertLessEqual(len(calls(sim.mlx, "load", "qwen3-tts")), 8, "two load attempts per job")

    def test_the_budget_can_be_set(self):
        self.write(AD)
        sim = self.make_sim(*(["fail"] * 400), infra_budget_per_day=1)
        sim.run_until(lambda: False, 8 * 3600, 2.0)
        self.assertEqual([i["counted"] for i in ledger.items(sim.store)], [0, 1, 1])

    def test_items_the_previous_version_settled_are_classified_and_their_slots_restored(self):
        # The live cascade: six planned ad clips failed their loads in 40 s and used up the week's
        # six slots. Their rows carry no classification; the next upkeep classifies them from their
        # lineage and the week's slots come back. Nothing edits the database by hand.
        self.write(tpl(SPEECH, id="t.ad-clip", caps={"per_week": 6, "max_stock": 6, "two_week_use": 6},
                       review_minutes=1))
        sim = self.make_sim(*(["fail"] * 12), tid="t.ad-clip", per_tick=6, review_max_minutes_day=30.0)
        self.assertTrue(sim.run_until(lambda: len(ledger.items(sim.store)) == 6, 3600))
        planner_mod.set_paused(sim.store, True, "the old gpud: no further tick before the upgrade", sim.sup._now())
        self.assertTrue(sim.run_until(lambda: len(ledger.items(sim.store, "failed")) == 6, 4 * 3600))
        with sim.store.tx():  # the rows as the old code left them
            sim.store.conn.execute("UPDATE plan_items SET fail_class = NULL, counted = NULL, count_note = NULL")
        self.assertEqual(self.ctx(sim)["week"]["t.ad-clip"], 6, "before: the week's cap is used up")
        sim.step()
        classified = sim.store.events("item_classified")
        self.assertEqual(len(classified), 6)
        self.assertTrue(all(e["data"]["fail_class"] == "infra" and not e["data"]["counted"] for e in classified))
        self.assertEqual(self.ctx(sim)["week"].get("t.ad-clip", 0), 0, "after: the six slots are back")
        planner_mod.set_paused(sim.store, False, None, sim.sup._now())
        planner_mod.plan_now(sim.store, sim.sup._now())
        self.assertTrue(sim.run_until(lambda: len(ledger.review_queue(sim.store)) == 6, 4 * 3600))
        self.assertEqual(self.ctx(sim)["week"]["t.ad-clip"], 6)
        sim.step()
        self.assertEqual(len(sim.store.events("item_classified")), 6, "classified once")
        text = report.build(sim.store, sim.sup.planner.templates()[0], sim.sup._now(), 0.0)
        self.assertIn("| active | 6/0 |", text)
        self.assertIn("6 infrastructure failure(s) in the window did not count", text)


class LV12ReportAndCli(ItemsCase):

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

    def test_report_splits_failures_and_skips_items_awaiting_review_as_sinks(self):
        self.write(AD)
        sim = self.make_sim("fail", "fail", "fail", "fail", jobs=(garbage(),), review_max_minutes_day=60.0,
                            per_tick=3)
        self.assertTrue(sim.run_until(lambda: len(ledger.review_queue(sim.store)) == 1
                                      and len(ledger.items(sim.store, "failed")) == 3, 6 * 3600))
        templates = sim.sup.planner.templates()[0]
        text = report.build(sim.store, templates, sim.sup._now(), 0.0)
        self.assertIn("| GPU h | P(ship) | status | failed infra/content |", text)
        self.assertIn("| active | 2/1 |", text)
        sinks = text.split("## Top GPU sinks with no use", 1)[1].split("##", 1)[0]
        waiting = ledger.review_queue(sim.store)[0]
        self.assertNotIn(f"| {waiting['job_id']} |", sinks, "an item awaiting a verdict is not a sink yet")
        content = [i for i in ledger.items(sim.store, "failed") if i["fail_class"] == "content"][0]
        self.assertIn(f"| {content['job_id']} |", sinks, "the garbage render used GPU time for nothing")
        ledger.record_verdict(sim.store, waiting["job_id"], "reject", None, None, sim.sup._now())
        text = report.build(sim.store, templates, sim.sup._now(), 0.0)
        self.assertIn(f"| {waiting['job_id']} |", text.split("## Top GPU sinks with no use", 1)[1].split("##", 1)[0])

    def test_cli_shows_count_notes_and_ownership(self):
        self.write(AD)
        sim = self.make_sim("fail", "fail", "fail", "fail")
        self.assertTrue(sim.run_until(lambda: len(ledger.review_queue(sim.store)) == 2, 4 * 3600))
        sim.mlx.load_as_other_client(MUSIC_M)
        sim.run_until(lambda: False, 3)
        code, out = self.cli("review", "--all")
        self.assertEqual(code, 0, out)
        self.assertIn("infra failure; not counted: infrastructure failure", out)
        code, out = self.cli("status")
        self.assertIn("qwen38-flash-next (gpud)", out)
        self.assertIn("ace-step (not gpud)", out)


class LV13AdapterProvenance(EngineCase):

    def test_instance_id_comes_from_metrics_and_a_restart_forgets_the_sizes(self):
        self.assertEqual(self.engine.instance_id(), f"start:{self.fake.start_time}")
        self.load(LTX)
        self.engine.resident_models()
        self.assertEqual(self.engine.model_size_gb(LTX), 36.0)
        self.fake.restart()
        self.assertNotEqual(self.engine.instance_id(), f"start:{self.fake.start_time - 3600}")
        self.assertIsNone(self.engine.model_size_gb(LTX), "sizes seen on the old process are forgotten")
        self.fake.metrics = "json"
        eng = make_engine(self.fake, self.tmp)
        try:
            self.assertEqual(eng.instance_id(), f"start:{self.fake.start_time}", "/metrics.json fallback")
        finally:
            eng.close()
        self.fake.metrics = "none"
        eng = make_engine(self.fake, self.tmp)
        try:
            self.assertIsNone(eng.instance_id())
            n = len(self.fake.requests)
            self.assertIsNone(eng.instance_id())
            self.assertEqual(len(self.fake.requests), n, "a server without /metrics is not asked every tick")
        finally:
            eng.close()
        self.assertEqual(parse_start_time("# x\nmlx_serve:process_start_time_seconds 1761456000\n"), 1761456000.0)
        self.assertEqual(parse_start_time('process_start_time_seconds{job="x"} 12.5'), 12.5)
        self.assertIsNone(parse_start_time("mlx_serve:requests_running 0"))

    def test_a_4xx_for_the_parameters_is_rejected_a_5xx_is_not(self):
        self.load(LTX)
        for n, (behaviour, reason) in enumerate((("http400", "rejected"), ("http500", None)), 1):
            with self.subTest(behaviour=behaviour):
                self.fake.script["/v1/video/generations"].append(behaviour)
                h = self.engine.start(spec(LTX, kind="render_video", run_id=f"{n}.1", prompt="a fox",
                                           num_frames=9, width=64, height=64))
                p = self.finish(h)
                self.assertEqual((p.status, p.reason), ("failed", reason), p.error)
        self.assertTrue(rejected_status(422, "bad steps"))
        for status, msg in ((404, "Unknown model id"), (429, "busy"), (400, "Target model does not support this modality")):
            self.assertFalse(rejected_status(status, msg), (status, msg))


class LV13GpudOnTheFakeServer(TmpCase):
    def setUp(self):
        super().setUp()
        self.fake = make_fake()
        self.catalog = studio.catalog(True)
        from foreman.brain_client import BrainClient
        self.engine = make_engine(self.fake, self.tmp, brain=BrainClient(self.fake.url, catalog=self.catalog))
        self.clock = ManualClock()
        self.sup = Supervisor(os.path.join(self.tmp, "foreman.db"), {"mlx-serve": self.engine}, self.clock,
                              studio.policy(), catalog=self.catalog, pager=[].append)
        self.sup.recover()

    def tearDown(self):
        self.sup.close()
        self.engine.close()
        self.fake.stop()
        super().tearDown()

    def run_until(self, pred, timeout=20.0, dt=1.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.clock.t += dt
            self.sup.tick()
            if pred():
                return True
            time.sleep(0.01)
        return False

    def jobs(self, kind):
        return [j for j in self.sup.store.jobs() if j.kind == kind]

    def submit(self, req):
        return self.sup.submit(req["kind"], req["requires"], req["payload"], req["on_success"], req["on_failure"])

    def unloaded(self):
        return [b.get("model") for b in self.fake.bodies("/v1/unload-model")]

    def test_slot_pressure_never_picks_the_catalog_brain_another_client_loaded(self):
        # Live Studio: the user's model, MLX-Serve.app's Flash-Next (a catalog model) and gpud's 27B
        # fill mlx-serve's three slots. A speech job needs a slot: before provenance the biggest
        # idle "gpud" brain, Flash-Next, was the victim.
        self.assertTrue(self.run_until(lambda: BRAIN in self.engine.resident_models()), "gpud's brain comes home")
        self.fake.other_client_load(FALLBACK)
        self.assertTrue(self.run_until(lambda: FALLBACK in self.engine.resident_models()))
        from foreman.cli import speech_request
        job = self.submit(speech_request("Hello from the Studio"))
        self.assertTrue(self.run_until(lambda: [j.state for j in self.jobs("review_audio")] == ["succeeded"], 30),
                        [(j.kind, j.state, j.outcome) for j in self.sup.store.jobs()])
        self.assertEqual(self.sup.store.job(job).state, "succeeded")
        self.assertNotIn(FALLBACK, self.unloaded())
        self.assertIn(BRAIN, self.unloaded(), "gpud freed a slot with its own brain")
        self.assertIn(FALLBACK, self.fake.ready)
        self.assertEqual(self.sup.store.owners().get(BRAIN, {}).get("instance"), f"start:{self.fake.start_time}")

    def test_after_a_server_restart_a_reloaded_brain_is_someone_elses(self):
        self.assertTrue(self.run_until(lambda: BRAIN in self.engine.resident_models()))
        self.assertTrue(self.sup._ours(BRAIN))
        self.fake.restart(keep=(FOREIGN,))
        self.fake.other_client_load(BRAIN)  # the app loads the same brain again
        self.assertTrue(self.run_until(lambda: self.sup.store.events("engine_instance_changed")))
        self.assertFalse(self.sup._ours(BRAIN))
        from foreman.cli import video_request
        job = self.submit(video_request("a red fox trotting through snow", 1.0, width=128, height=96, expected_s=60))
        self.assertTrue(self.run_until(lambda: [j.state for j in self.jobs("review_render")] == ["succeeded"], 30),
                        [(j.kind, j.state, j.outcome) for j in self.sup.store.jobs()])
        self.assertEqual(self.sup.store.job(job).state, "succeeded")
        self.assertNotIn(BRAIN, self.unloaded(), "video_exclusive would unload gpud's brain, never theirs")
        self.assertEqual(self.jobs("review_render")[0].model, BRAIN, "their brain reviewed it, read-only")


if __name__ == "__main__":
    unittest.main()
