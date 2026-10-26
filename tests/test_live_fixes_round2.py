"""Regressions for the fixes that answered the red team of the live fixes
(tests/test_live_fixes_adversarial.py) and the design concerns it raised.

  LR1  the mlx-serve adapter: a load abandoned after its request reached the server is reported
       in flight; another client's load in flight is visible; a restarted gpud re-adopts the
       server id of its own copy, so the other client's same-named copy is never taken for it;
       a size no Mac could hold is ignored
  LR2  the supervisor: optional adapter hints that fail never mark the engine down; sizes of
       models outside the catalog are forgotten with the server instance; a media job that cannot
       run because someone else's models hold the memory or slots it needs (or because its model
       is listed in foreign_models) waits without holding up the queue and fails after
       load_defer_max_s; make-room never unloads gpud's own brain for a target whose model slots
       can never be freed; the planner never routes work to a model listed in foreign_models
  LR3  failure classes: a crash or timeout that repeats in a lineage counts; a review refused for
       its request counts; the weekly ceiling on uncounted infrastructure failures; a brief whose
       item failed for infrastructure reasons (and did not count) may be proposed again
  LR4  Studio config: foreign_models names are checked and normalized, the headroom is bounded
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest

from foreman import ledger, studio
from foreman.adapters.mlx_serve import OTHER_COPY
from foreman.httpio import json_call
from foreman.policy import Policy
from foreman.sim import ScriptedBrain, ScriptedPlanner, Sim
from foreman.store import Store
from foreman.supervisor import Supervisor

try:  # discovered with -s tests (tests/ on sys.path) or run from the project root
    from test_brain_return import calls, continuations_of, submit_render
    from test_live_fixes import IMAGE_M, MUSIC_M, OwnershipCase, submit_media, unloads
    from test_mlx_serve_adapter import (ACE, BRAIN, FOREIGN, KOKORO, EngineCase, ManualClock, TmpCase, make_engine,
                                        make_fake, wait_for)
    from test_planner import SPEECH, VIDEO as VIDEO_TPL, PlannerCase, prop, tpl
except ImportError:  # pragma: no cover
    from tests.test_brain_return import calls, continuations_of, submit_render
    from tests.test_live_fixes import IMAGE_M, MUSIC_M, OwnershipCase, submit_media, unloads
    from tests.test_mlx_serve_adapter import (ACE, BRAIN, FOREIGN, KOKORO, EngineCase, ManualClock, TmpCase,
                                              make_engine, make_fake, wait_for)
    from tests.test_planner import SPEECH, VIDEO as VIDEO_TPL, PlannerCase, prop, tpl

PRIMARY, FALLBACK_SIM, VIDEO = "qwen38-flash-next", "qwen36-35b-a3b", "ltx-2.5"
KOKORO_BASE = KOKORO.rsplit("/", 1)[-1]


class LR1Adapter(EngineCase):

    def test_a_load_abandoned_after_its_request_reached_the_server_is_in_flight(self):
        self.fake.load_delay = 0.8
        t = self.engine.load(ACE)
        self.assertTrue(wait_for(lambda: self.fake.loading, 3.0), "the request reached the server")
        self.assertTrue(self.engine.unload(ACE))          # gpud aborts it (nothing registered yet: 404, ok)
        st = self.engine.poll_load(t)
        self.assertEqual((st.state, st.in_flight), ("failed", True), st.error)
        self.assertIn("may still finish", st.error)
        self.assertTrue(wait_for(lambda: "ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit" in self.fake.ready, 3.0),
                        "and the server finishes it")
        low = make_engine(self.fake, self.tmp, mem_probe=lambda: 10.0)
        try:
            st = low.poll_load(low.load(KOKORO))           # refused before anything was sent
            self.assertEqual((st.state, st.in_flight), ("failed", False))
        finally:
            low.close()

    def test_another_clients_load_in_flight_is_visible(self):
        self.fake.load_delay = 0.6
        other = threading.Thread(target=lambda: json_call(self.fake.url, "POST", "/v1/load-model", {"model": BRAIN},
                                                          timeout=5.0))
        other.start()
        self.assertTrue(wait_for(lambda: self.engine.loading_models() == {BRAIN}, 3.0))
        other.join()
        self.assertEqual(self.engine.loading_models(), set())
        st = self.engine.poll_load(self.engine.load(BRAIN))
        self.assertEqual(st.state, "failed", "their copy is resident: gpud sends no load that would claim it")

    def test_a_copy_under_the_catalog_name_is_not_gpuds_when_gpuds_copy_has_another_id(self):
        # gpud loaded Kokoro by path (not discovered yet: registered under the basename). After a
        # rescan another client loads the discovered id, which IS the catalog name. A restarted
        # adapter that is told which copy is gpud's reports theirs under a label.
        self.load(KOKORO)
        self.assertEqual(self.engine.owned_id(KOKORO), KOKORO_BASE)
        self.fake.discovered.add(KOKORO)
        self.fake.other_client_load(KOKORO)
        fresh = make_engine(self.fake, self.tmp)
        try:
            fresh.set_owned_id(KOKORO, KOKORO_BASE)
            self.assertEqual(fresh.resident_models(), {FOREIGN, KOKORO, f"{KOKORO} {OTHER_COPY}"})
            self.assertEqual(fresh.server_id(KOKORO), KOKORO_BASE)
            self.assertTrue(fresh.unload(KOKORO))
            self.assertIn(KOKORO, self.fake.ready, "their copy stays")
            self.assertNotIn(KOKORO_BASE, self.fake.ready)
            with self.assertRaises(PermissionError):
                fresh.unload(f"{KOKORO} {OTHER_COPY}")
        finally:
            fresh.close()

    def test_a_size_no_mac_could_hold_is_ignored(self):
        self.fake.other_client_load(KOKORO_BASE, size=int(5e15))
        self.engine.resident_models()
        self.assertIsNone(self.engine.model_size_gb(KOKORO))
        self.assertEqual(self.engine.plan_gb(KOKORO, studio.catalog()[KOKORO]), 1)


class LR1GpudRestart(TmpCase):
    """The server id of gpud's copy is recorded in model_owners and handed to the next adapter."""

    def make(self):
        from foreman.brain_client import BrainClient
        catalog = studio.catalog()
        engine = make_engine(self.fake, self.tmp, brain=BrainClient(self.fake.url, catalog=catalog))
        sup = Supervisor(os.path.join(self.tmp, "foreman.db"), {"mlx-serve": engine}, self.clock,
                         studio.policy(idle_media_unload_s=None), catalog=catalog, pager=[].append)
        sup.recover()
        return engine, sup

    def run_until(self, sup, pred, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.clock.t += 1.0
            sup.tick()
            if pred():
                return True
            time.sleep(0.01)
        return False

    def test_a_restarted_gpud_keeps_its_own_copy_apart_from_someone_elses(self):
        from foreman.media_requests import speech_request
        self.fake, self.clock = make_fake(), ManualClock()
        self.addCleanup(self.fake.stop)
        engine, sup = self.make()
        req = speech_request("Hello from the Studio")
        job = sup.submit(req["kind"], req["requires"], req["payload"], req["on_success"], req["on_failure"])
        self.assertTrue(self.run_until(sup, lambda: sup.store.job(job).state == "succeeded", 30))
        self.assertEqual(sup.store.owners()[KOKORO]["server_id"], KOKORO_BASE)
        sup.close()
        engine.close()
        self.fake.discovered.add(KOKORO)
        self.fake.other_client_load(KOKORO)           # someone else's copy, under the catalog name
        engine, sup = self.make()
        self.addCleanup(engine.close)
        self.addCleanup(sup.close)
        self.assertEqual(engine.owned_id(KOKORO), KOKORO_BASE, "re-adopted from model_owners")
        self.assertTrue(sup._ours(KOKORO))
        self.assertTrue(sup._pinned() >= {f"{KOKORO} {OTHER_COPY}"})
        self.assertTrue(sup._unload(KOKORO, sup._now(), "test") == "unloaded")
        self.assertEqual([b["model"] for b in self.fake.bodies("/v1/unload-model")], [KOKORO_BASE])
        self.assertIn(KOKORO, self.fake.ready)


class LR2Supervisor(OwnershipCase):

    def test_optional_hints_that_fail_never_mark_the_engine_down(self):
        sim = self.make()
        self.settle(sim)
        sim.mlx.instance_id = lambda: {"not": "an id"}

        def size(_model):
            raise RuntimeError("size probe broke")
        sim.mlx.model_size_gb = size
        sim.run_until(lambda: False, 100)
        self.assertEqual(sim.store.events("engine_error"), [])
        self.assertEqual(sim.store.events("engine_recovered"), [])
        noted = {e["data"]["method"] for e in sim.store.events("engine_extension_error")}
        self.assertEqual(noted, {"instance_id", "model_size_gb"})
        self.assertEqual(len(sim.store.events("engine_extension_error")), 2, "noted once each, not every tick")
        self.assertIn(PRIMARY, sim.mlx.resident_models())

    def test_sizes_of_models_outside_the_catalog_are_forgotten_with_the_server(self):
        sim = self.make()
        self.settle(sim)
        sim.mlx.observed_gb["users-chat-model"] = 30.0
        sim.mlx.load_as_other_client("users-chat-model")
        sim.run_until(lambda: False, 3)
        self.assertEqual(sim.sup._size("users-chat-model"), 30.0)
        sim.mlx.server_restart()
        sim.mlx.observed_gb["users-chat-model"] = 10.0
        sim.mlx.load_as_other_client("users-chat-model")
        sim.run_until(lambda: False, 3)
        self.assertTrue(sim.store.events("engine_instance_changed"))
        self.assertEqual(sim.sup._size("users-chat-model"), 10.0)

    def test_a_media_job_blocked_by_someone_elses_memory_fails_after_the_bound(self):
        sim = self.make(capacity=176.0, policy=Policy(load_defer_max_s=600.0),
                        brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        sim.mlx.load_as_other_client(FALLBACK_SIM)     # 21 GB that gpud may not free: video (150) no longer fits
        render = submit_render(sim.sup)
        sim.run_until(lambda: False, 30)
        speech = submit_media(sim.sup, "qwen3-tts")
        self.assertTrue(sim.run_until(lambda: sim.store.job(speech).state == "succeeded", 300),
                        "the queue behind the blocked render keeps moving")
        self.assertTrue(sim.run_until(lambda: continuations_of(sim.store, render), 900))
        j = sim.store.job(render)
        self.assertEqual((j.state, j.outcome["reason"]), ("failed", "load_failed"))
        self.assertIn("could not be run for", j.outcome["detail"])
        self.assertEqual(len(continuations_of(sim.store, render)), 1, "its baton fires once")
        self.assertTrue(sim.store.events("media_blocked"))
        self.assert_never_unloaded(sim, FALLBACK_SIM)
        self.assertEqual(unloads(sim.mlx, PRIMARY), [], "nothing was unloaded for a render that could not fit")

    def test_make_room_never_unloads_the_brain_for_slots_it_can_never_get(self):
        # One model slot, held by someone else's music model. The render fits in memory once the
        # brain is gone, but never in the slots: unloading the brain first (and reloading it on the
        # next tick, over and over) would be churn for nothing.
        sim = self.make(capacity=176.0, policy=Policy(load_defer_max_s=600.0),
                        brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        sim.mlx.max_resident_models = 1
        sim.mlx.load_as_other_client(MUSIC_M)
        render = submit_render(sim.sup)
        sim.run_until(lambda: False, 300)
        self.assertEqual(unloads(sim.mlx, PRIMARY), [])
        self.assertEqual(sim.store.job(render).state, "queued")
        self.assertTrue(sim.store.events("insufficient_slots"))
        sim.run_until(lambda: sim.store.job(render).state != "queued", 600)
        self.assertEqual(sim.store.job(render).outcome["reason"], "load_failed")
        self.assert_never_unloaded(sim, MUSIC_M, PRIMARY)

    def test_a_job_for_a_model_in_foreign_models_waits_then_fails_and_nothing_is_unloaded(self):
        sim = self.make(capacity=176.0, policy=Policy(foreign_models=(VIDEO,), load_defer_max_s=600.0),
                        brain=ScriptedBrain(retry_failures=False))
        self.settle(sim)
        render = submit_render(sim.sup)
        self.assertTrue(sim.run_until(lambda: sim.store.job(render).state != "queued", 900))
        self.assertEqual(sim.store.job(render).outcome["reason"], "load_failed")
        self.assertIn("foreign_models", sim.store.job(render).outcome["detail"])
        self.assertEqual(calls(sim.mlx, "load", VIDEO), [])
        self.assertEqual(unloads(sim.mlx, PRIMARY), [])

    def test_someone_elses_resident_copy_of_a_model_in_foreign_models_serves_its_job(self):
        sim = self.make(policy=Policy(foreign_models=(IMAGE_M,)))
        self.settle(sim)
        sim.mlx.load_as_other_client(IMAGE_M)
        job = submit_media(sim.sup, IMAGE_M)
        self.assertTrue(sim.run_until(lambda: sim.store.job(job).state == "succeeded", 600))
        self.assertEqual(calls(sim.mlx, "load", IMAGE_M), [])
        self.assert_never_unloaded(sim, IMAGE_M)


class LR2PlannerRouting(PlannerCase):

    def test_the_planner_never_routes_work_to_a_model_in_foreign_models(self):
        self.write(tpl(VIDEO_TPL, id="t.clip"))
        planner = self.planner()
        t = planner.templates()[0]["t.clip"]
        for foreign, want in (((), VIDEO), ((VIDEO,), None)):
            with self.subTest(foreign_models=foreign):
                sim = Sim(os.path.join(self.tmp, f"r{len(foreign)}.db"), policy=Policy(foreign_models=foreign))
                self.addCleanup(sim.sup.close)
                self.assertEqual(planner.model_for(sim.sup, t), want)


class LR3Classes(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(os.path.join(tmp.name, "c.db"))
        self.addCleanup(self.store.close)

    def lineage(self, *media, review=None):
        root, jobs = None, []
        for reason in media:
            outcome = {"status": "failed", "reason": reason}
            jid, _ = self.store.insert_job(kind="render_video", requires="ltx-2.5", state="failed", payload={},
                                           outcome=outcome, root_id=root, max_attempts=3, created_at=0.0)
            root = root or jid
            self.store.update_job(jid, root_id=root)
            jobs.append(self.store.job(jid))
        if review is not None:
            jid, _ = self.store.insert_job(kind="render_video", requires="ltx-2.5", state="succeeded", payload={},
                                           outcome={"status": "succeeded"}, root_id=root, max_attempts=3,
                                           created_at=0.0)
            self.store.add_continuation(jid, {"kind": "review"}, {"kind": "recover"})
            cid, _ = self.store.insert_job(kind="review_render", requires="brain", state="quarantined", payload={},
                                           outcome={"status": "failed", "reason": review}, root_id=root or jid,
                                           max_attempts=3, created_at=0.0)
            self.store.mark_fired(jid, "success", cid, 0.0)
            jobs.append(self.store.job(jid))
        return jobs

    def cls(self, *media, review=None):
        return ledger.failure_class(self.store, self.lineage(*media, review=review))

    def test_a_crash_or_timeout_that_repeats_in_the_lineage_counts(self):
        self.assertEqual(self.cls("crash")[0], "infra", "once: the environment's")
        self.assertEqual(self.cls("timeout", "load_failed")[0], "infra")
        for pair in (("crash", "crash"), ("timeout", "crash"), ("timeout", "load_failed", "timeout")):
            with self.subTest(lineage=pair):
                cls, why = self.cls(*pair)
                self.assertEqual(cls, "content", why)
                self.assertIn("failed the engine again", why)

    def test_a_review_refused_for_its_request_counts_and_one_that_could_not_run_does_not(self):
        self.assertEqual(self.cls(review="rejected")[0], "content")
        self.assertEqual(self.cls(review="timeout")[0], "infra")


class LR3Budgets(PlannerCase):

    def test_uncounted_infrastructure_failures_are_capped_per_week_too(self):
        # infra_budget_per_day 5, the template's per_week 2: failures on days 1, 3 and 5 of one
        # week. Each day's budget has room, but only per_week of them go uncounted in 7 days; the
        # one on day 9 is outside that window again.
        self.write(tpl(SPEECH, id="t.wk", caps={"per_week": 2, "max_stock": 2, "two_week_use": 2}))
        planner = self.planner(infra_budget_per_day=5)
        templates = planner.templates()[0]
        day = 86400.0
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "w.db"))
            ids = []
            for n, at in enumerate((1 * day, 3 * day, 5 * day, 9 * day)):
                with store.tx():
                    ids.append(ledger.insert_item(store, template_id="t.wk", consumer="weekly demo reel",
                                                  brief=f"b{n}", content_hash=f"h{n}", job_id=100 + n,
                                                  state="failed", fail_class="infra", created_at=at - 60,
                                                  finished_at=at))

            class _Sup:
                pass
            sup = _Sup()
            sup.store = store
            planner.classify_failed(sup, 10 * day, templates)
            got = [ledger.item(store, i)["counted"] for i in ids]
            notes = [ledger.item(store, i)["count_note"] for i in ids]
            store.close()
        self.assertEqual(got, [0, 0, 1, 0])
        self.assertIn("in 7 days are used up", notes[2])

    def test_a_brief_that_failed_for_infrastructure_reasons_may_be_proposed_again(self):
        t = tpl(SPEECH, id="t.again", caps={"per_week": 5, "max_stock": 5, "two_week_use": 5})
        self.write(t)
        brief = "the same script, read again"
        brain = ScriptedPlanner(policy=lambda _snap: {"proposals": [prop("t.again", brief=brief)]},
                                retry_failures=False)
        sim = Sim(os.path.join(self.tmp, "again.db"), brain=brain,
                  planner=self.planner(review_max_minutes_day=60.0, min_gap_s=60.0))
        self.addCleanup(lambda: sim.sup.close())
        sim.mlx.script_loads("qwen3-tts", "fail", "fail")   # the first item's load fails twice
        self.assertTrue(sim.run_until(lambda: len(ledger.review_queue(sim.store)) == 1, 4 * 3600))
        its = ledger.items(sim.store)
        self.assertEqual([(i["brief"], i["state"]) for i in its], [(brief, "failed"), (brief, "review")])
        first, second = its
        self.assertEqual((first["fail_class"], first["counted"]), ("infra", 0))
        self.assertEqual(first["content_hash"], f"{second['content_hash']}~{first['id']}", "kept under a retired hash")
        self.assertNotEqual(first["job_id"], second["job_id"])
        snap = sim.sup.planner.snapshot(sim.sup, sim.sup._now(), float(sim.clock.now()),
                                        sim.sup.planner.templates()[0], {})
        marks = [(o["state"], o.get("may_repeat", False)) for o in snap["recent_outputs"]]
        self.assertEqual(marks, [("review", False), ("failed", True)])
        sim.run_until(lambda: False, 3600, 5.0)
        dup = [r for r in ledger.rejections_since(sim.store, float("-inf")) if r["reason"] == "duplicate"]
        self.assertTrue(dup, "once it was produced, the same brief is a duplicate again")
        self.assertEqual(len(ledger.items(sim.store)), 2)


class LR4StudioConfig(unittest.TestCase):

    def config(self, data, **env):
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "config.json"), "w") as f:
                json.dump(data, f)
            return studio.load_config(env={"FOREMAN_HOME": home, **env})

    def test_foreign_models_are_normalized_or_refused(self):
        cfg = self.config({"foreign_models": [studio.BRAIN_FALLBACK.rsplit("/", 1)[-1] + ":latest", studio.LTX]})
        self.assertEqual(cfg.policy().foreign_models, (studio.BRAIN_FALLBACK, studio.LTX))
        with self.assertRaises(ValueError) as e:
            self.config({"foreign_models": ["my-own-chat-model"]})
        self.assertIn("not a catalog model", str(e.exception))

    def test_a_headroom_that_leaves_no_room_for_the_brain_is_refused_with_the_limit(self):
        self.assertEqual(self.config({}, FOREMAN_MIN_HEADROOM_GB="180").policy().usable_gb, 20.0)
        with self.assertRaises(ValueError) as e:
            self.config({}, FOREMAN_MIN_HEADROOM_GB="181")
        self.assertIn("At most 180 GB", str(e.exception))


if __name__ == "__main__":
    unittest.main()
