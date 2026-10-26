"""Regression tests for the fixes made after the second planner red team (beyond the red team's
own tests in test_planner_adversarial.py). Each test names the design concern it covers.

  X1  a reply that crashes its own application is an invalid tick and still pays its tokens
  X2  a reply listing hundreds of junk proposals is an invalid tick, not 31,000 rejection rows
  X3  inside a planned item's lineage the brain may only retry the planned job
  X4  plan jobs the planner did not create are never applied and never load a model
  X5  a paused template's queued work is cancelled quietly: no page, the item fails, no review
  X6  an engine-side loss of the plan job (brain evicted) is not an invalid tick
  X7  the disk guard counts what the work admitted earlier in the same tick may write
  X8  an MP4 whose box structure is truncated keeps the raw frames
  X9  accepted items not yet shipped count as value for kill_low_value
  X10 a wall-clock step back does not refill the day's token budget
  X11 a same-size template edit with the old mtime is still picked up
  X12 text hygiene: folding, printable output, URLs and invisible characters
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import unittest
from unittest import mock

from foreman import ledger, planner as planner_mod, retention, sim as sim_mod
from foreman.policy import CATALOG, Policy
from foreman.sim import ScriptedPlanner, crash_mid_job
from foreman.store import TERMINAL
from foreman.supervisor import PLAN_KIND
from foreman.templates import dir_signature, has_url, load_templates, strip_urls
from foreman.text import fold, printable, strip_invisible

try:
    from test_planner_adversarial import (DAY, SPEECH, VIDEO, AdvCase, JumpClock, fake_item, plan_jobs, plan_of,
                                          prop, rejected_reasons, tpl)
except ImportError:  # pragma: no cover - run from the repository root
    from tests.test_planner_adversarial import (DAY, SPEECH, VIDEO, AdvCase, JumpClock, fake_item, plan_jobs,
                                                plan_of, prop, rejected_reasons, tpl)


class X1CrashingReplies(AdvCase):
    def test_a_reply_that_crashes_apply_counts_as_invalid_and_is_charged(self):
        self.write(SPEECH)
        with mock.patch.object(planner_mod, "parse_plan", side_effect=RuntimeError("validator bug")):
            sim = self.sim({"proposals": []}, {"proposals": []}, {"proposals": []},
                           interval_s=120.0, min_gap_s=1e9)
            jobs = self.wait_plans(sim, 3)
        self.assertIn("error", plan_of(jobs[0]))
        self.assertEqual(len(sim.store.events("plan_error")), 3)
        self.assertEqual(len(sim.store.events("plan_invalid")), 3, "plan_error counts toward the invalid streak")
        st = ledger.planner_state(sim.store)
        self.assertIsNotNone(st["suspended_until"], "three crashing replies suspend the planner and page")
        prompts = sum(e["data"]["prompt_tokens"] for e in sim.store.events("planner_tick"))
        self.assertGreater(st["tokens_used"], prompts, "the replies were charged although apply rolled back")


class X2JunkFloods(AdvCase):
    def test_hundreds_of_junk_proposals_are_one_invalid_tick(self):
        self.write(SPEECH)
        junk = '{"proposals": [' + ",".join(["0"] * 31000) + "]}"
        self.assertLess(len(junk), planner_mod.PlannerConfig().max_reply_chars)
        sim = self.sim(junk)
        job = self.wait_plans(sim, 1)[0]
        self.assertIn("proposals (a reply may list at most", plan_of(job)["invalid"])
        self.assertEqual(ledger.rejections_since(sim.store, float("-inf")), [])
        self.assertEqual(sim.store.events("plan_rejected"), [])


class X3PlannedLineages(AdvCase):
    def _sim_with_recover(self, submit, crashes: int = 1):
        class Recover(ScriptedPlanner):
            def __call__(self, spec, model):
                if spec.kind.startswith("recover"):
                    brief = spec.payload["brief"]
                    act = brief["attempted_action"]
                    base = {"kind": act["kind"], "requires": act["model"], "on_success": act["on_success"],
                            "on_failure": act["on_failure"], "retry_of": brief["source_job_id"]}
                    params = dict(act["params"], goal=brief["goal"])
                    return {"verdict": "retry", "note": "variants", "submit": submit(base, params)}
                return super().__call__(spec, model)
        self.write(SPEECH)
        brain = Recover({"proposals": [prop("t.speech", brief="the planned script")]})
        sim = self.sim(brain=brain, interval_s=1e6, min_gap_s=1e9)
        sim.mlx.script_jobs("speak", *[crash_mid_job() for _ in range(crashes)])
        job = self.wait_plans(sim, 1)[0]
        root = plan_of(job)["admitted"][0]["job_id"]
        sim.run_until_quiet(3600)
        return sim, root

    def test_only_retries_of_the_planned_job_are_admitted(self):
        def submit(base, params):
            return [dict(base, kind="render_video", requires="ltx-2.5",
                         payload={"goal": "unvetted", "prompt": "an unvetted render", "expected_s": 60}),
                    dict(base, payload=dict(params, input="cold email outreach script for every lead")),
                    dict(base, payload=dict(params, speed=5.0)),
                    dict(base, payload=dict(params, seed=7, speed=1.1))]
        sim, root = self._sim_with_recover(submit)
        children = [j for j in sim.store.jobs() if j.brainless and j.root_id == root and j.id != root]
        self.assertEqual(len(children), 1, [(j.kind, j.payload) for j in children])
        self.assertEqual((children[0].payload["seed"], children[0].payload["speed"]), (7, 1.1))
        self.assertEqual(children[0].payload["input"], "the planned script", "the words stay as admitted")
        errors = [e["data"]["error"] for e in sim.store.events("admission_rejected")]
        self.assertTrue(any("may only retry its own speak" in e for e in errors), errors)
        self.assertTrue(any("may not change ['input']" in e for e in errors), errors)
        self.assertTrue(any("may not change ['speed']" in e for e in errors), errors)

    def test_a_planned_lineage_runs_at_most_max_attempts_media_jobs(self):
        seeds = itertools.count(100)

        def submit(base, params):  # a new seed and no retry_of: a new logical task every time
            return [dict({k: v for k, v in base.items() if k != "retry_of"}, payload=dict(params, seed=next(seeds)))]
        sim, root = self._sim_with_recover(submit, crashes=8)  # every attempt crashes
        media = [j for j in sim.store.jobs() if j.brainless and j.root_id == root]
        self.assertEqual(len(media), Policy().max_attempts)
        errors = [e["data"]["error"] for e in sim.store.events("admission_rejected")]
        self.assertTrue(any("already ran 3 media jobs" in e for e in errors), errors)


class X4RoguePlanJobs(AdvCase):
    def test_a_plan_job_without_the_planners_key_is_ignored(self):
        self.write(SPEECH)
        brain = ScriptedPlanner({"proposals": [prop("t.speech", brief="from a rogue plan row")]})
        sim = self.sim(brain=brain, interval_s=1e6, first_tick_s=1e6, min_gap_s=1e9)
        with sim.store.tx():  # a row written by an older gpud, before admission refused the kind
            jid, _ = sim.sup._insert_job(kind=PLAN_KIND, requires="brain", state="queued",
                                         payload={"needs": "any", "prompt": {"system": "s", "user": "u"}},
                                         max_attempts=1, priority=0, idempotency_key="child:9.1.0",
                                         created_at=sim.sup._now())
        sim.run_until(lambda: sim.store.job(jid).state in TERMINAL, 900)
        sim.run_until_quiet(900)
        self.assertEqual(ledger.items(sim.store), [])
        self.assertEqual([e["job_id"] for e in sim.store.events("load_started") if e["job_id"] == jid], [])
        self.assertIn("ignored", (sim.store.job(jid).outcome or {}).get("plan") or {"ignored": "never ran"})

    def test_the_owner_cannot_submit_a_plan_job_either(self):
        sim = self.sim()
        from foreman.supervisor import AdmissionError
        for kind in (PLAN_KIND, " Plan "):
            with self.assertRaises(AdmissionError):
                sim.sup.submit(kind, "brain", {"needs": "any"})


class X5PausedTemplates(AdvCase):
    def test_cancelled_planned_work_fails_quietly_and_skips_review(self):
        self.write(SPEECH)
        sim = self.sim({"proposals": [prop("t.speech") for _ in range(3)]}, interval_s=1e6, min_gap_s=1e9)
        self.wait_plans(sim, 1)
        ledger.set_template_paused(sim.store, "t.speech", True, "owner", "enough of these", sim.sup._now())
        sim.run_until_quiet(3600)
        items = ledger.items(sim.store)
        self.assertEqual({i["state"] for i in items}, {"failed"})
        self.assertTrue(all("paused" in (i["fail_reason"] or "") for i in items), [i["fail_reason"] for i in items])
        self.assertEqual(ledger.review_queue(sim.store), [])
        self.assertEqual(sim.store.jobs("awaiting_human"), [], "a deliberate pause pages nobody")


class X6EngineSideLosses(AdvCase):
    def test_an_evicted_brain_is_not_an_invalid_tick(self):
        self.write(SPEECH)
        sim = self.sim({"proposals": [prop("t.speech")]}, interval_s=1e6, min_gap_s=1e9)
        self.assertTrue(sim.run_until(lambda: any(j.state == "running" for j in plan_jobs(sim)), 600))
        plan = [j for j in plan_jobs(sim) if j.state == "running"][0]
        for m in sorted(sim.mlx.resident_models()):
            if CATALOG[m].is_brain:
                sim.mlx.evict(m)
        self.assertTrue(sim.run_until(lambda: sim.store.job(plan.id).state in TERMINAL, 600))
        self.assertEqual(sim.store.events("plan_invalid"), [])
        self.assertEqual(ledger.planner_state(sim.store)["invalid_streak"], 0)
        failed = sim.store.events("plan_job_failed")
        self.assertEqual([(e["data"]["counted"], e["data"]["brain_gone"]) for e in failed], [(False, True)])


class X7DiskReservations(AdvCase):
    def test_one_tick_cannot_fill_the_last_gigabyte(self):
        self.write(VIDEO)
        self.disk = [300.4, 1.0]  # 0.4 GB above the 300 GB floor; one 5 s 768x512 render is ~0.17 GB
        sim = self.sim({"proposals": [prop("t.video", "render_video", brief=f"shot {k}") for k in range(5)]})
        job = self.wait_plans(sim, 1)[0]
        self.assertEqual(len(plan_of(job)["admitted"]), 2)
        self.assertEqual([r for _, r in rejected_reasons(job)], ["disk_guard"] * 3)
        self.assertIn("reserved for admitted work", plan_of(job)["rejected"][0]["detail"])
        admitted = sim.store.job(plan_of(job)["admitted"][0]["job_id"])
        self.assertGreater(retention.estimate_bytes({"kind": admitted.kind, "payload": admitted.payload}), 1e8)


def _box(kind: bytes, body: bytes) -> bytes:
    return (8 + len(body)).to_bytes(4, "big") + kind + body


class X8Mp4Check(AdvCase):
    def test_truncated_or_incomplete_mp4_keeps_the_raw_frames(self):
        good = _box(b"ftyp", b"isom\x00\x00\x02\x00isom") + _box(b"moov", b"\x00" * 32) + _box(b"mdat", b"\x01" * 400)
        cases = {"good": (good, True), "truncated": (good[:-100], False), "no moov": (
            _box(b"ftyp", b"isom\x00\x00\x02\x00") + _box(b"mdat", b"\x01" * 40), False),
                 "empty": (b"", False), "not iso-bmff": (b"mp4data", None)}
        for label, (data, want) in cases.items():
            with self.subTest(label):
                d = os.path.join(self.tmp, label.replace(" ", "_"))
                os.makedirs(d)
                mp4, frames = os.path.join(d, "output.mp4"), os.path.join(d, "frames.rgb")
                with open(mp4, "wb") as f:
                    f.write(data)
                with open(frames, "wb") as f:
                    f.write(b"\x00" * 64)
                self.assertEqual(retention.mp4_complete(mp4), want)
                art = retention.drop_raw_frames({"path": mp4, "frames_path": frames, "paths": [frames]}, False)
                self.assertEqual(os.path.exists(frames), want is False, label)
                self.assertEqual(bool(art.get("raw_frames_deleted")), want is not False)


class X9KillCriteria(AdvCase):
    def test_accepted_items_shipped_later_are_not_low_value(self):
        self.write(SPEECH)
        sim = self.sim(interval_s=1e6)
        for day in range(16):  # all accepted, none shipped yet (shipped in a batch at month end)
            it = fake_item(sim.store, "t.speech", day * DAY + 1, review_minutes=3.0, value_usd=20.0)
            ledger.record_verdict(sim.store, it["job_id"], "accept", None, None, day * DAY + 100)
        templates = load_templates(os.path.join(self.home, "templates"))[0]
        self.assertEqual(ledger.kill_check(sim.store, templates, 16 * DAY, sim.planner.cfg), [])


class X10TokenDays(AdvCase):
    def test_a_wall_clock_step_back_does_not_refill_the_budget(self):
        self.write(SPEECH)
        with mock.patch.object(sim_mod, "VirtualClock", JumpClock):
            sim = self.sim(interval_s=600.0, max_reply_tokens=500)
        self.wait_plans(sim, 1)
        st = ledger.planner_state(sim.store)
        first = sim.store.events("planner_tick")[0]["data"]["prompt_tokens"]
        sim.planner.cfg = dataclasses.replace(sim.planner.cfg, token_budget=st["tokens_used"] + first + 250)
        sim.run_until(lambda: False, 700)
        self.assertEqual(len(plan_jobs(sim)), 1)
        sim.clock.offset -= DAY  # the wall clock steps back a day; supervisor time does not
        sim.run_until(lambda: False, 1300)
        self.assertEqual(len(plan_jobs(sim)), 1, "a clock step must not hand out a second budget for the day")
        self.assertEqual(ledger.planner_state(sim.store)["tokens_day"], "1970-01-01")


class X11TemplateEdits(AdvCase):
    def test_a_same_size_edit_with_the_old_mtime_is_picked_up(self):
        path = os.path.join(self.home, "templates", "t.speech.json")
        self.write(tpl(SPEECH, caps={"per_week": 9, "max_stock": 50, "two_week_use": 50}))
        before = os.stat(path)
        sig = dir_signature(os.path.join(self.home, "templates"))
        tmp = path + ".new"
        with open(tmp, "w") as f:
            json.dump(tpl(SPEECH, caps={"per_week": 1, "max_stock": 50, "two_week_use": 50}), f)
        os.utime(tmp, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(tmp, path)  # an editor or sync tool: same size, same mtime, new inode
        self.assertEqual(os.stat(path).st_size, before.st_size)
        self.assertNotEqual(dir_signature(os.path.join(self.home, "templates")), sig)


class X12TextHygiene(unittest.TestCase):
    def test_fold_printable_and_urls(self):
        self.assertEqual(fold("Ｔｈｅ week​ly up­date﻿"), "The weekly update")
        self.assertEqual(strip_invisible("⁠‍"), "")
        self.assertEqual(printable("a\x1b]52;c;x\x07\rb‮\nc"), "a\\x1b]52;c;x\\x07\\x0db\\u202e\\nc")
        for s in ("acme.edu/x", "bit.ly/abc", "10.0.0.5:8080", "ｅｖｉｌ．ｃｏｍ/x", "evil​.com", "a@b.io",
                  "localhost:3000"):
            self.assertTrue(has_url(s), s)
        for s in ("e.g. a sting", "U.S. recap", "10.5 s", "and/or", "3.5/5 stars", "the end.Then"):
            self.assertFalse(has_url(s), s)
        self.assertEqual(strip_urls("see evil-site.com/brief now"), "see [link removed] now")


if __name__ == "__main__":
    unittest.main()
