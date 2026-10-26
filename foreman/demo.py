"""Narrated run of the brain-return mechanism on a virtual clock.

    python3 -m foreman.demo                 # render hangs, lease expires, brain recovers
    python3 -m foreman.demo --fault crash   # or: garbage, wedged, none
"""
from __future__ import annotations

import argparse
import os
import tempfile

from .policy import CATALOG
from .sim import SUCCESS, TIGHT_HOST_GB, Sim, crash_mid_job, garbage, hang

FAULTS = {"hang": hang(0.4), "crash": crash_mid_job(0.5), "garbage": garbage(), "wedged": hang(0.4, wedged=True)}
GOAL = "8s clip: a red fox trotting through fresh snow"


def _line(e: dict, store) -> str | None:
    d, j = e["data"], e["job_id"]
    job = store.job(j) if j else None
    k = e["kind"]
    if k == "supervisor_start":
        return f"gpud started (boot {d['boot']}); no model loaded yet"
    if k == "admitted":
        if job.brainless:
            c = store.continuation(j)
            return (f"job {j} {d['kind']} admitted for {d['requires']} with the baton: "
                    f"on_success={c['on_success']['kind']}, on_failure={c['on_failure']['kind']}")
        return f"job {j} {d['kind']} admitted (needs the brain)"
    if k == "load_started":
        why = "" if j else " (idle: bringing the primary brain home)"
        return f"loading {d['model']} on {d['engine']} (~{d['expected_s']:.0f}s){why}"
    if k == "model_loaded" and not CATALOG[d["model"]].is_brain:
        return f"{d['model']} loaded"
    if k == "model_unloaded":
        return f"unloaded {d['model']} ({d['reason']})"
    if k == "brain_departed" and e["at"] > 0:
        return "brain is no longer resident; what happens next is carried by the queued job's continuations"
    if k == "brain_returned":
        if e["at"] - d["away_s"] <= 0:
            return f"brain ready ({', '.join(d['brains'])})"
        return f"brain is back ({', '.join(d['brains'])}) after {d['away_s']:.0f}s away"
    if k == "job_starting":
        extra = f", waited {d['waited_s']:.0f}s for the brain" if d["continuation"] else ""
        return f"job {j} {d['kind']} running on {d['model']} (attempt {d['attempt']}{extra})"
    if k == "lease_granted" and job.brainless:
        return (f"job {j} lease: renew by progress within {d['progress_timeout_s']:.0f}s, "
                f"hard limit {d['hard_timeout_s']:.0f}s")
    if k == "lease_expired":
        return f"job {j} lease expired: {d['detail']}"
    if k == "cancel_sent":
        return f"job {j} graceful cancel sent (no force-kill)"
    if k == "engine_terminate":
        return f"{d['engine']} refused to unload; sending graceful SIGTERM"
    if k == "job_finished" and d["state"] != "queued":
        why = f" ({d['reason']}: {d['detail']})" if d["reason"] else ""
        return f"job {j} {d['kind']} {d['state']}{why}"
    if k == "continuation_fired":
        return f"job {j} hands the baton to job {d['child']} {d['kind']} (resume brief attached)"
    if k == "brain_decision":
        sub = f" -> submitted {d['submitted']}" if d["submitted"] else ""
        return f"job {j} brain ({d['model']}) decided: {d['note'] or d['verdict']}{sub}"
    if k in ("page_human", "safe_mode_entered"):
        return f"PAGE HUMAN: {d['reason']}"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fault", choices=[*FAULTS, "none"], default="hang")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        sim = Sim(os.path.join(tmp, "foreman.db"), capacity_gb=TIGHT_HOST_GB)
        if args.fault != "none":
            sim.mlx.script_jobs("render_video", FAULTS[args.fault])
        sim.mlx.script_jobs("render_video", SUCCESS)
        print(f"Foreman demo: {sim.policy.usable_gb:.0f} GB usable, so the video model cannot share memory "
              f"with any brain.\nFault on the first render: {args.fault}. Virtual clock; nothing sleeps.\n")
        sim.sup.submit("plan_video", payload={"goal": GOAL})
        settled = sim.run_until_quiet(4 * 3600)
        store = sim.store
        for e in store.events():
            text = _line(e, store)
            if text:
                print(f"  t={e['at']:6.0f}s  {text}")

        recover = next((t for t in sim.brain.transcript if t["kind"] == "recover_render"), None)
        if recover:
            b = recover["brief"]
            params = b["attempted_action"]["params"]
            print(f"\nWhat the reloaded brain read (job {recover['job_id']} on {recover['model']}); it keeps no memory:")
            print(f"  goal              {b['goal']}")
            print(f"  attempted_action  {b['attempted_action']['kind']} on {b['attempted_action']['model']} "
                  f"(duration_s={params.get('duration_s')}, expected_s={params.get('expected_s')})")
            print(f"  outcome           {b['outcome']['status']}: {b['outcome']['reason']} - {b['outcome']['detail']}")
            print(f"  artifacts         {b['artifacts']}")
            print(f"  validator_results {[(r['check'], r['ok']) for r in b['validator_results']]}")
            print(f"  log_tail          {' | '.join(b['log_tail'][-3:])}")
            print(f"  attempt           {b['attempt']} of {b['max_attempts']}")

        fired = store.events("continuation_fired")
        first_ready = store.events("brain_returned")[0]["at"]  # ignore the cold boot
        print(f"\nSettled: {settled}. Continuations fired: {len(fired)} (one per brainless job). "
              f"Longest brain absence: {sim.max_brain_away(after=first_ready):.0f}s "
              f"(MAX_BRAINLESS_S={sim.policy.max_brainless_s:.0f}). Brain resident now: {sim.brain_resident()}.")
        sim.sup.close()
    return 0 if settled else 1


if __name__ == "__main__":
    raise SystemExit(main())
