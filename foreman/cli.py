"""Foreman command line: submit work to gpud, look at its state, and give the owner's verdicts.

    python3 -m foreman.cli submit-speech "Hello from the Studio" [--voice af_heart]
    python3 -m foreman.cli submit-music "warm lo-fi piano, rain" --seconds 30 [--lyrics TEXT]
    python3 -m foreman.cli submit-video "a red fox trotting through snow" --seconds 5 [--width 768 --height 512]
    python3 -m foreman.cli status [--live]
    python3 -m foreman.cli jobs [--all]
    python3 -m foreman.cli events [--tail 30] [--job ID]
    python3 -m foreman.cli cancel JOB_ID
    python3 -m foreman.cli clear-safe

The planner's loop (see README "Planner"):

    python3 -m foreman.cli review [--all]                 items awaiting your verdict, with their files
    python3 -m foreman.cli verdict JOB accept|reject [--minutes M] [--note TEXT] [--tags voice,pacing]
    python3 -m foreman.cli verdict JOB retract [--note TEXT]   undo a verdict given by mistake
    python3 -m foreman.cli ship JOB [--usd X]             you used it (and what it earned)
    python3 -m foreman.cli resolve [PAGE ...] [--note TEXT]   close pages (no id: list open pages)
    python3 -m foreman.cli templates [list | pause ID | unpause ID]
    python3 -m foreman.cli planner status|pause|resume|plan-now [--note TEXT]
    python3 -m foreman.cli report [--days 7] [--out PATH]  Markdown scoreboard

Ads (see README "Ads") and the characters they star:

    python3 -m foreman.cli ads status                     every ad, where it stands
    python3 -m foreman.cli ads inspect JOB|AD_ID          its steps, takes, QA and files
    python3 -m foreman.cli ads enqueue TEMPLATE --brief TEXT [--seconds S --shots N --seed X] [--force]
    python3 -m foreman.cli ads models [--rescan]          the configured model per role vs /v1/models
    python3 -m foreman.cli characters list | show ID
    python3 -m foreman.cli characters import DIR --id ID [--voice af_heart | --ref-wav F] [--look TEXT] [--approve]
    python3 -m foreman.cli characters approve ID FILE... [--tags hero,talk] [--unapprove]

Phone approvals over Telegram (see README "Phone approvals"; the service is foreman.phone):

    python3 -m foreman.cli phone setup [--no-wait]        bot token (hidden prompt or stdin), then pairing
    python3 -m foreman.cli phone status | test | resend | unpair [--keep-token]

The CLI never talks to mlx-serve (except `status --live`, which only reads /api/ps) and never
takes gpud's lock: it opens the same SQLite file as a non-leader client (Supervisor(lead=False))
and goes through the supervisor's public submit(), so every admission rule applies. The running
gpud picks the job up on its next tick.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

from . import ledger, planner, report, studio
from .clock import SystemClock
from .media_requests import music_request, speech_request, video_expected_s, video_request  # noqa: F401
from .supervisor import AdmissionError, Supervisor
from .templates import load_templates
from .text import printable as _p


class _ClientEngine:
    """Admission only checks that an engine is configured; a client never calls it."""

    def __init__(self, name: str = studio.ENGINE):
        self.name = name

    def __getattr__(self, item):
        raise AttributeError(f"the CLI does not talk to engines (asked for {item!r})")


def _client(cfg) -> Supervisor:
    if not os.path.exists(os.path.dirname(cfg.db_path) or "."):
        os.makedirs(os.path.dirname(cfg.db_path), exist_ok=True)
    catalog = cfg.catalog()
    engines = {name: _ClientEngine(name) for name in sorted({s.engine for s in catalog.values()})}
    return Supervisor(cfg.db_path, engines, SystemClock(), studio.policy(), catalog=catalog, lead=False)


def _when(wall) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall)) if wall else "-"


def _short(model) -> str:
    return _p(str(model).split("/")[-1]) if model else "-"


# ---------------------------------------------------------------- submissions
# The builders live in foreman.media_requests (the planner shares them); re-exported here.

def _submit(cfg, req: dict) -> int:
    sup = _client(cfg)
    try:
        job = sup.submit(req["kind"], req["requires"], req["payload"], req["on_success"], req["on_failure"])
    finally:
        sup.close()
    print(f"job {job} queued: {req['kind']} on {req['requires']} "
          f"(expected {req['payload']['expected_s']:.0f}s; gpud picks it up on its next tick)")
    return job


# ---------------------------------------------------------------- views

def cmd_status(cfg, live: bool) -> None:
    sup = _client(cfg)
    try:
        st = sup.store.state()
        jobs = sup.store.jobs()
        pst = planner.status(sup.store, sup._now())
        review_n = len(ledger.review_queue(sup.store))
        owners = sup.store.owners()
    finally:
        sup.close()
    by_state: dict = {}
    for j in jobs:
        by_state[j.state] = by_state.get(j.state, 0) + 1
    print(f"db               {cfg.db_path}")
    print(f"mode             {st['mode']}" + (f"  SAFE: {st['safe_reason']}" if st["safe"] else ""))
    resident = sorted(st["known_resident"] or {})
    print("resident         " + (", ".join(
        f"{_short(m)} ({'gpud' if (owners.get(m) or {}).get('state') == 'owned' else 'loading' if m in owners else 'not gpud'})"
        for m in resident) or "-"))
    if st["loading"]:
        ld = st["loading"]
        print(f"loading          {_short(ld.get('model'))} (for job {ld.get('for_job')})")
    if st["brainless_since"] is not None and st["t_now"] is not None:
        print(f"brain away       {st['t_now'] - st['brainless_since']:.0f}s")
    health = {k: v for k, v in (st["engine_health"] or {}).items() if v.get("down_since") is not None}
    if health:
        print(f"engines down     {json.dumps(health)}")
    print(f"gpud boots       {st['boot_count']}  (last tick {_when(st['t_wall'])})")
    print(f"jobs             {json.dumps(by_state, sort_keys=True)}")
    print(f"planner          {pst['state']}; review queue {review_n} item(s); open pages {by_state.get('awaiting_human', 0)}")
    active = [j for j in jobs if j.state in ("starting", "running", "cancelling", "queued", "awaiting_human")]
    for j in active[-15:]:
        print(f"  {j.id:>5} {j.state:<15} {_p(j.kind, 40):<16} {_short(j.requires):<34} attempt {j.attempt}")
    if live:
        from .httpio import json_call
        try:
            _, ps = json_call(cfg.base_url, "GET", "/api/ps", timeout=5.0)
            names = [m.get("name") for m in (ps or {}).get("models", [])]
            print(f"mlx-serve /api/ps {names}")
        except Exception as e:
            print(f"mlx-serve /api/ps unreachable: {type(e).__name__}: {e}")


def cmd_jobs(cfg, show_all: bool) -> None:
    sup = _client(cfg)
    try:
        jobs = sup.store.jobs()
    finally:
        sup.close()
    if not show_all:
        jobs = jobs[-40:]
    print(f"{'id':>5} {'state':<15} {'kind':<16} {'requires':<34} {'att':>3} {'parent':>6}  reason")
    for j in jobs:
        o = j.outcome or {}
        reason = o.get("reason") or ""
        if j.requires == "brain" and isinstance(o.get("artifact"), dict):
            d = o["artifact"].get("decision") or {}
            reason = f"{d.get('verdict')}: {str(d.get('note') or '')[:60]}"
        elif j.state == "awaiting_human":
            reason = str((j.payload or {}).get("reason") or "")[:80]
        print(f"{j.id:>5} {j.state:<15} {_p(j.kind, 40):<16} {_short(j.requires):<34} {j.attempt:>3} "
              f"{j.parent_id if j.parent_id else '-':>6}  {_p(reason, 120)}")


def cmd_events(cfg, tail: int, job: int | None) -> None:
    sup = _client(cfg)
    try:
        events = sup.store.events(job_id=job) if job is not None else \
            sup.store.events_after(max(0, sup.store.last_event_id() - tail), tail)
    finally:
        sup.close()
    for e in events[-tail:]:
        data = json.dumps(e["data"], sort_keys=True, default=str)
        print(f"{_when(e['wall'])} #{e['id']:<6} {e['kind']:<22} job={e['job_id'] if e['job_id'] else '-':<5} "
              f"{data[:220]}")


# ---------------------------------------------------------------- the owner's loop

def _templates(cfg) -> tuple:
    return load_templates(os.path.join(cfg.home, "templates"))


def _files(paths) -> list:
    """(label, path) for an item's artifacts, the ones a person opens first."""
    order = {".mp4": "MP4", ".wav": "WAV", ".png": "PNG", ".json": "JSON", ".rgb": "RAW"}
    out = []
    for p in paths or []:
        label = order.get(os.path.splitext(p)[1].lower(), "FILE")
        out.append((label, p))
    rank = {"MP4": 0, "WAV": 1, "PNG": 2, "JSON": 3, "RAW": 4, "FILE": 5}
    return sorted(out, key=lambda x: rank[x[0]])


def cmd_review(cfg, show_all: bool) -> None:
    sup = _client(cfg)
    try:
        now = sup._now()
        queue = ledger.review_queue(sup.store)
        recent = [i for i in ledger.items(sup.store) if i["state"] != "review"][-15:] if show_all else []
        tags = ledger.reject_tag_counts(sup.store)
    finally:
        sup.close()
    if not queue:
        print("review queue is empty")
    # Briefs and brain notes are untrusted text (inbound note -> LLM -> brief): every field is
    # printed through text.printable, so no escape sequence or carriage return reaches the terminal.
    for it in queue:
        produced = it.get("produced_at")
        age = (now - (produced if produced is not None else now)) / 3600.0
        print(f"job {it['job_id']:<5} {_p(it['template_id'], 80)}  for {_p(repr(it['consumer']), 120)}  "
              f"waiting {age:.1f} h  (review ~{it['review_minutes'] or 0:g} min)")
        print(f"    brief: {_p(it['brief'], 200)}")
        if it.get("brain_verdict"):
            print(f"    brain: {_p(it['brain_verdict'], 40)}"
                  + (f": {_p(it['brain_note'], 200)}" if it.get("brain_note") else ""))
        for label, path in _files(it.get("artifacts")):
            print(f"    {label:<4} {_p(path)}")
        print(f"    -> python3 -m foreman.cli verdict {it['job_id']} accept|reject [--minutes M] [--note TEXT] "
              "[--tags voice,pacing]")
    for it in recent:
        extra = it.get("verdict_note") or it.get("fail_reason") or ""
        print(f"  [{it['state']:<8}] job {it['job_id']:<5} {_p(it['template_id'], 80)}: {_p(it['brief'], 80)}"
              + (f"  ({_p(extra, 80)})" if extra else ""))
        if it["state"] == "failed" and it.get("count_note"):
            print(f"             {it.get('fail_class') or '?'} failure; {_p(it['count_note'], 200)}")
    tagged = {tid: c for tid, c in sorted(tags.items()) if c["tagged"]}
    if tagged:  # what the owner said was wrong with the rejects that stand, per template
        print("reject reasons by template:")
        for tid, c in tagged.items():
            print(f"  {_p(tid, 80)}: {ledger.tag_summary(c)}")


# The owner's actions, without printing. The phone service (foreman/phone.py) records verdicts,
# retracts, shipments, notes and page resolutions through these same functions, so both paths
# share the ledger's validation, its events (the phone's carry by="phone") and the kill criteria.

def apply_verdict(cfg, job_id: int, verdict: str, minutes=None, note=None, by=None, tags=None) -> dict:
    """accept | reject | retract. `tags`: a reject's reasons (ledger.parse_tags). Returns {"item",
    "p_ship", "killed"}; raises ledger.LedgerError."""
    templates, _ = _templates(cfg)
    sup = _client(cfg)
    try:
        now = sup._now()
        if verdict == "retract":
            if tags:
                raise ledger.LedgerError("a retract takes no reason tags")
            it = ledger.retract_verdict(sup.store, job_id, note, now, by=by)
            killed = []
        else:
            it = ledger.record_verdict(sup.store, job_id, verdict, minutes, note, now, by=by, tags=tags)
            killed = ledger.kill_check(sup.store, templates, now, cfg.planner_config()) if templates else []
        t = templates.get(it["template_id"])
        p = ledger.posterior(t, ledger.items(sup.store, template=t.id))[2] if t else None
    finally:
        sup.close()
    return {"item": it, "p_ship": p, "killed": killed}


def apply_ship(cfg, job_id: int, usd=None, by=None) -> dict:
    sup = _client(cfg)
    try:
        return ledger.record_ship(sup.store, job_id, usd, sup._now(), by=by)
    finally:
        sup.close()


def apply_tags(cfg, job_id: int, tags, by=None) -> dict:
    """Set the reasons of a reject already recorded (ledger.tag_verdict: idempotent)."""
    sup = _client(cfg)
    try:
        return ledger.tag_verdict(sup.store, job_id, tags, sup._now(), by=by)
    finally:
        sup.close()


def apply_note(cfg, job_id: int, note: str, window_s: float, by=None, ref=None) -> dict:
    """Add a note to a verdict given at most window_s ago (supervisor time); `ref` makes it
    idempotent (see ledger.annotate_verdict)."""
    sup = _client(cfg)
    try:
        return ledger.annotate_verdict(sup.store, job_id, note, sup._now(), window_s, by=by, ref=ref)
    finally:
        sup.close()


def apply_resolve(cfg, page_id: int, note=None, by=None) -> None:
    sup = _client(cfg)
    try:
        ledger.resolve_page(sup.store, page_id, note, sup._now(), by=by)
    finally:
        sup.close()


def apply_planner(cfg, action: str, note=None) -> dict:
    """status | pause | resume | plan-now; returns planner.status() after the action."""
    sup = _client(cfg)
    try:
        now = sup._now()
        if action == "pause":
            planner.set_paused(sup.store, True, note, now)
        elif action == "resume":
            planner.set_paused(sup.store, False, note, now)
        elif action == "plan-now":
            planner.plan_now(sup.store, now)
        return planner.status(sup.store, now)
    finally:
        sup.close()


def cmd_verdict(cfg, job_id: int, verdict: str, minutes, note, tags=None) -> None:
    if tags is not None:  # the reasons for a reject: with the verdict, or added to one already recorded
        tags = ledger.parse_tags(tags)
        if not tags:
            raise ledger.LedgerError("--tags needs one or more of: " + ", ".join(t for t, _n in ledger.REASON_TAGS))
        if verdict != "reject":
            raise ledger.LedgerError("--tags gives the reasons for a reject (verdict JOB reject --tags voice,pacing)")
        sup = _client(cfg)
        try:
            cur = ledger.item_for_job(sup.store, job_id)
        finally:
            sup.close()
        if cur is not None and cur["state"] == "rejected":
            if minutes is not None or note is not None:
                raise ledger.LedgerError(f"job {job_id} is already rejected: only --tags can be added to its verdict")
            it = apply_tags(cfg, job_id, tags)
            print(f"job {job_id}: reasons recorded for its reject ({_p(it['template_id'], 80)}): "
                  + ", ".join(ledger.TAG_NAMES[t] for t in it["verdict_tags"] or []))
            return
    res = apply_verdict(cfg, job_id, verdict, minutes, note, tags=tags)
    it, p = res["item"], res["p_ship"]
    if verdict == "retract":
        print(f"job {job_id}: verdict retracted; {_p(it['template_id'], 80)} item back in the review queue"
              + (f"; P(ship) is now {p:.2f}" if p is not None else ""))
        return
    print(f"job {job_id}: {verdict} recorded for {it['template_id']} ({it['verdict_minutes']:g} review min)"
          + (f"; P(ship) is now {p:.2f}" if p is not None else "")
          + (f"; {ledger.reasons_text(it['verdict_tags'])}" if it.get("verdict_tags") else ""))
    for tid, reason, detail in res["killed"]:
        print(f"template {tid} paused by a kill criterion: {reason} ({detail})")


def cmd_ship(cfg, job_id: int, usd) -> None:
    it = apply_ship(cfg, job_id, usd)
    print(f"job {job_id}: shipped ({it['template_id']})" + (f", ${usd:.2f}" if usd is not None else ""))


def cmd_resolve(cfg, ids, note) -> int:
    sup = _client(cfg)
    try:
        if not ids:
            pages = sup.store.jobs("awaiting_human")
            if not pages:
                print("no open pages")
            for j in pages:
                reason = str((j.payload or {}).get("reason") or "")
                print(f"page {j.id:<5} {_p(j.kind, 40):<18} {_p(reason, 160)}")
            return 0
        now, failed = sup._now(), 0
        for pid in ids:
            try:
                ledger.resolve_page(sup.store, pid, note, now)
                print(f"page {pid}: resolved")
            except ledger.LedgerError as e:
                failed += 1
                print(f"page {pid}: {e}", file=sys.stderr)
    finally:
        sup.close()
    return 1 if failed else 0


def cmd_templates(cfg, action: str, tid, note) -> int:
    templates, errors = _templates(cfg)
    sup = _client(cfg)
    try:
        now = sup._now()
        if action in ("pause", "unpause"):
            if not tid:
                print(f"templates {action} needs a template id", file=sys.stderr)
                return 2
            if tid not in templates and tid not in ledger.template_states(sup.store):
                print(f"no template {tid!r}", file=sys.stderr)
                return 1
            ledger.set_template_paused(sup.store, tid, action == "pause", "owner" if action == "pause" else None,
                                       note, now)
            print(f"template {tid}: {action}d")
            return 0
        stats = ledger.template_stats(sup.store, templates, now)
        states = ledger.template_states(sup.store)
    finally:
        sup.close()
    print(f"{'template':<34} {'kind':<14} {'status':<26} {'P(ship)':>7} {'week':>6} {'stock':>7} "
          f"{'7d prod/acc/rej/ship':>21} {'usd 28d':>8}")
    for t_id in sorted(set(templates) | set(stats)):
        t, s, ps = templates.get(t_id), stats.get(t_id, {}), states.get(t_id) or {}
        status = (f"paused: {ps.get('reason')}" if ps.get("paused") else "not loaded" if t is None
                  else "active" if t.active else "inactive")
        week = f"{s.get('week', 0)}/{t.per_week}" if t else str(s.get("week", 0))
        stock = f"{s.get('stock', 0)}/{min(t.max_stock, int(t.two_week_use))}" if t else str(s.get("stock", 0))
        w = s.get("7d") or {}
        flow = f"{w.get('produced', 0)}/{w.get('accepted', 0)}/{w.get('rejected', 0)}/{w.get('shipped', 0)}"
        p = s.get("p_ship")
        print(f"{_p(t_id, 80):<34} {(t.kind if t else '-'):<14} {_p(status, 26):<26} "
              f"{('%.2f' % p) if p is not None else '-':>7} "
              f"{week:>6} {stock:>7} {flow:>21} {(s.get('28d') or {}).get('usd', 0):>8.2f}")
        if ps.get("paused") and ps.get("detail"):
            print(f"    {_p(ps['detail'], 300)}")
    if not templates and not stats:
        print(f"(no templates in {os.path.join(cfg.home, 'templates')}; see examples/templates)")
    for e in errors:  # a template error quotes the owner's file: printed line by line, escaped
        lines = str(e).splitlines() or [""]
        print(f"error: {_p(lines[0], 400)}")
        for line in lines[1:]:
            print(_p(line, 400))
    return 1 if errors else 0


def cmd_planner(cfg, action: str, note) -> None:
    st = apply_planner(cfg, action, note)
    if action == "pause":
        print("planner paused (gpud keeps running the queue; `planner resume` to plan again)")
    elif action == "resume":
        print("planner resumed (suspension and invalid streak cleared)")
    elif action == "plan-now":
        print("plan-now recorded: gpud plans on its next tick once a brain is home and the GPU is free"
              + ("" if st["state"] == "active" else f" (but the planner is {st['state']}: resume it first)"))
    due = st["next_due_in_s"]
    print(f"planner          {st['state']}" + (f" ({_p(st['paused_note'], 200)})" if st.get("paused_note") else ""))
    print(f"interval         {(st['interval_s'] or 0) / 60:.0f} min (empty ticks in a row: {st['empty_streak']}, "
          f"invalid in a row: {st['invalid_streak']}, suspensions: {st['suspensions']})")
    print("next tick        " + ("-" if due is None else "due now" if due <= 0 else f"in {due / 60:.0f} min")
          + (" (plan-now pending)" if st["force"] else "") + (f" (trigger: {st['trigger']})" if st["trigger"] else ""))
    print(f"ticks            {st['ticks']} ({st['ticks_7d']} in 7 days); last plan job "
          + (f"{st['last_plan_job']['id']} {st['last_plan_job']['state']}" if st["last_plan_job"] else "-"))
    print(f"tokens           {st['tokens_used']} on {st['tokens_day'] or '-'}")


def cmd_report(cfg, days: float, out) -> str:
    templates, _ = _templates(cfg)
    sup = _client(cfg)
    try:
        text = report.build(sup.store, templates, sup._now(), time.time(), days)
    finally:
        sup.close()
    path = out or os.path.join(cfg.home, "reports", f"scoreboard-{time.strftime('%Y-%m-%d')}.md")
    report.write(text, path)
    print(f"report written to {path}")
    return path


# ---------------------------------------------------------------- ads and characters

def cmd_ads(cfg, args) -> int:
    from . import ad_pipeline
    if args.action == "status":
        sup = _client(cfg)
        try:
            items = ad_pipeline.ad_items(sup.store, cfg.artifacts_dir)
        finally:
            sup.close()
        if not items:
            print("no ads yet (ads enqueue TEMPLATE --brief TEXT, or an active ad template for the planner)")
        for it in items:
            pr = it["progress"] or {}
            last = it["last_job"] or {}
            print(f"job {it['job_id']:<5} item {it['item']:<4} {it['state']:<9} {_p(it['template_id'], 60)}  "
                  f"step {pr.get('current', '?'):<9} last {_p(last.get('kind'), 20)} {last.get('state', '-')}")
            print(f"    brief: {_p(it['brief'], 160)}")
            if pr.get("final"):
                print(f"    MP4  {_p(pr['final'])}")
            if it["state"] == "failed":
                print(f"    {it.get('fail_class') or '?'} failure: {_p(it.get('fail_reason') or '', 200)}")
        return 0
    if args.action == "inspect":
        if not args.target:
            print("ads inspect needs a job id or an ad id", file=sys.stderr)
            return 2
        sup = _client(cfg)
        try:
            items = ad_pipeline.ad_items(sup.store, cfg.artifacts_dir)
            events = None
            hit = next((i for i in items if str(i["job_id"]) == args.target or i["ad_id"] == args.target), None)
            if hit:
                events = [e for e in sup.store.events(job_id=hit["job_id"])][-5:]
        finally:
            sup.close()
        if hit is None:
            print(f"no ad {args.target!r}", file=sys.stderr)
            return 1
        wd = hit["workdir"]
        print(f"ad {hit['ad_id']} (job {hit['job_id']}, item {hit['item']}, {hit['state']})  {_p(wd)}")
        pr = hit["progress"] or {"done": {}}
        print("steps   " + "  ".join(f"{k}:{'done' if v else '-'}" for k, v in pr["done"].items()))
        from . import paths as _paths
        script = _paths.read_json(os.path.join(wd, "script.json")) if wd else None
        if isinstance(script, dict):
            for ln in script.get("lines", []):
                print(f"  line {ln['i']} ({ln['role']}): {_p(ln['text'], 200)}")
            for sh in script.get("shots", []):
                print(f"  shot {sh['k']} {sh['kind']:<7} lines {sh['lines']} {_p(sh.get('visual') or '', 120)}")
        pick = _paths.read_json(os.path.join(wd, "pick.json")) if wd else None
        for k, v in (pick or {}).items():
            print(f"  shot {k}: picked seed {v.get('seed')} (score {v.get('score')}); takes "
                  + ", ".join(f"{t['seed']}={t['score']}" for t in v.get("takes", [])))
        qa = _paths.read_json(os.path.join(wd, "final", "qa.json")) if wd else None
        if isinstance(qa, dict):
            print("QA      " + ("pass" if qa.get("ok") else "FAIL") + ": "
                  + "; ".join(f"{c['check']} {'ok' if c['ok'] else 'FAIL'} ({_p(c['detail'], 60)})" for c in qa["checks"]))
        state = _paths.read_json(os.path.join(wd, "state.json"), {}) if wd else {}
        for st in (state or {}).get("steps", [])[-12:]:
            print(f"  job {st.get('job_id')}: {st.get('kind')} {st.get('status')} ({st.get('reason')}) "
                  f"{st.get('seconds')}s")
        if (state or {}).get("script_errors"):
            print(f"  script problems: {_p(state['script_errors'], 300)}")
        for e in events or []:
            print(f"  event {e['kind']} {json.dumps(e['data'], default=str)[:160]}")
        for name in ("final/ad.mp4", "manifest.json"):
            if wd and os.path.exists(os.path.join(wd, name)):
                print(f"  {name:<14} {_p(os.path.join(wd, name))}")
        return 0
    if args.action == "enqueue":
        templates, errors = _templates(cfg)
        t = templates.get(args.target or "")
        if t is None:
            print(f"no loaded template {args.target!r}" + (f" ({len(errors)} template file(s) have errors)" if errors else ""),
                  file=sys.stderr)
            return 1
        sup = _client(cfg)
        try:
            res = ad_pipeline.enqueue(sup, t, args.brief, args.consumer,
                                      {"seconds": args.seconds, "shots": args.shots, "seed": args.seed},
                                      cfg.artifacts_dir, cfg.home, cfg.planner_config(), force=args.force)
        finally:
            sup.close()
        print(f"ad {res['ad_id']} queued as job {res['job_id']} (item {res['item']}); gpud runs it step by step. "
              f"Work directory: {res['workdir']}")
        return 0
    if args.action == "models":
        from .httpio import json_call
        try:
            if args.rescan:
                json_call(cfg.base_url, "POST", "/v1/models/rescan", {}, timeout=30.0)
            _, obj = json_call(cfg.base_url, "GET", "/v1/models", timeout=5.0)
        except Exception as e:
            print(f"mlx-serve unreachable: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        listed = {e.get("id"): set(e.get("capabilities") or []) for e in (obj or {}).get("data") or [] if isinstance(e, dict)}
        need = {"tts": "audio", "tts_clone": "audio", "asr": "transcription", "image": "image", "video": "video",
                "music": "music"}
        ads_cfg = cfg.ads_config()
        for role, cap in need.items():
            m = ads_cfg.models.get(role)
            base = (m or "").rsplit("/", 1)[-1]
            sid = m if m in listed else next((i for i in listed if i and i.rsplit("/", 1)[-1] == base), None)
            caps = listed.get(sid)
            state = "missing" if sid is None else ("ok" if caps is None or cap in caps else f"lacks {cap!r}")
            print(f"{role:<10} {_p(m or '-', 60):<60} {state}" + (f" (caps {sorted(caps)})" if caps else ""))
        for i, caps in sorted(listed.items(), key=lambda x: str(x[0])):
            print(f"  server: {_p(i, 70):<70} {sorted(caps)}")
        return 0
    return 2


def cmd_characters(cfg, args) -> int:
    from . import characters
    try:
        if args.action == "list":
            found = characters.list_bibles(cfg.home)
            if not found:
                print(f"no characters in {characters.characters_dir(cfg.home)} (characters import DIR --id ID)")
            for cid, b, err in found:
                if b is None:
                    print(f"{_p(cid, 64):<24} ERROR {_p(err, 300)}")
                    continue
                d = b.describe()
                print(f"{cid:<24} {_p(d['name'] or '', 30):<30} {d['kind'] or '-':<7} stills {d['approved']}/{d['stills']} "
                      f"approved, tags {d['tags']}, voice {d['voice'].get('engine')}" + ("" if d["hero"] else "  (NO HERO)"))
            return 0
        if args.action == "show":
            b = characters.load_bible(cfg.home, args.id or "")
            d = b.describe()
            print(json.dumps(d, indent=1))
            for st in b.data.get("stills", []):
                print(f"  {'approved' if st.get('approved') else 'pending ':<8} {','.join(st.get('tags', [])):<24} "
                      f"{_p(os.path.join(b.root, st['file']))}")
            return 0
        if args.action == "import":
            if not args.dir or not args.id:
                print("characters import needs DIR and --id", file=sys.stderr)
                return 2
            b = characters.import_folder(cfg.home, args.dir, args.id, name=args.name, kind=args.kind, look=args.look,
                                         subject=args.subject, voice=args.voice, speed=args.speed, ref_wav=args.ref_wav,
                                         language=args.language, approve=args.approve, overwrite=args.overwrite)
            d = b.describe()
            print(f"character {b.id}: {d['stills']} still(s), {d['approved']} approved, tags {d['tags']} -> {b.root}")
            if not d["hero"]:
                print("note: no approved still is tagged hero; ads need one (characters approve ID FILE --tags hero)")
            return 0
        if args.action == "approve":
            b = characters.approve(cfg.home, args.id or "", args.files or [],
                                   tags=[t for t in (args.tags or "").split(",") if t] or None,
                                   approved=not args.unapprove)
            print(f"character {b.id}: approved stills now {b.describe()['approved']}")
            return 0
    except characters.BibleError as e:
        print(f"characters: {e}", file=sys.stderr)
        return 1
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m foreman.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config file (default: $FOREMAN_HOME/config.json)")
    ap.add_argument("--db", dest="db_path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit-speech", help="Kokoro text to speech")
    s.add_argument("text")
    s.add_argument("--voice")
    s.add_argument("--speed", type=float)
    s.add_argument("--expected-s", type=float)
    m = sub.add_parser("submit-music", help="ACE-Step text to music")
    m.add_argument("prompt")
    m.add_argument("--seconds", type=float, required=True)
    m.add_argument("--lyrics")
    m.add_argument("--seed", type=int)
    m.add_argument("--expected-s", type=float)
    v = sub.add_parser("submit-video", help="LTX-2.5 text to video")
    v.add_argument("prompt")
    v.add_argument("--seconds", type=float, required=True)
    v.add_argument("--width", type=int, default=768)
    v.add_argument("--height", type=int, default=512)
    v.add_argument("--steps", type=int)
    v.add_argument("--seed", type=int)
    v.add_argument("--expected-s", type=float)
    st = sub.add_parser("status", help="supervisor state")
    st.add_argument("--live", action="store_true", help="also read mlx-serve's /api/ps")
    j = sub.add_parser("jobs", help="recent jobs")
    j.add_argument("--all", action="store_true")
    e = sub.add_parser("events", help="the event log")
    e.add_argument("--tail", type=int, default=30)
    e.add_argument("--job", type=int)
    c = sub.add_parser("cancel", help="cancel a queued or running job")
    c.add_argument("job_id", type=int)
    sub.add_parser("clear-safe", help="leave SAFE mode (a human decision)")
    r = sub.add_parser("review", help="items awaiting your verdict")
    r.add_argument("--all", action="store_true", help="also show recently decided items")
    vd = sub.add_parser("verdict", help="accept or reject a reviewed item (updates P(ship)); retract undoes one")
    vd.add_argument("job_id", type=int)
    vd.add_argument("verdict", choices=("accept", "reject", "retract"))
    vd.add_argument("--minutes", type=float, help="review minutes you spent (default: the template's estimate)")
    vd.add_argument("--note")
    vd.add_argument("--tags", help="reject: what was wrong, comma-separated (" + ", ".join(
        t for t, _n in ledger.REASON_TAGS) + "); on a rejected item, sets its reasons")
    sh = sub.add_parser("ship", help="record that an accepted item was used")
    sh.add_argument("job_id", type=int)
    sh.add_argument("--usd", type=float, help="what it earned (default: the template's value_usd)")
    rs = sub.add_parser("resolve", help="resolve pages (human_review, engine_lost, ...); no id lists them")
    rs.add_argument("page_ids", type=int, nargs="*")
    rs.add_argument("--note")
    tp = sub.add_parser("templates", help="templates with stats, caps and pauses")
    tp.add_argument("action", nargs="?", default="list", choices=("list", "pause", "unpause"))
    tp.add_argument("template_id", nargs="?")
    tp.add_argument("--note")
    pl = sub.add_parser("planner", help="planner status and control")
    pl.add_argument("action", nargs="?", default="status", choices=("status", "pause", "resume", "plan-now"))
    pl.add_argument("--note")
    rp = sub.add_parser("report", help="write the Markdown scoreboard")
    rp.add_argument("--days", type=float, default=7.0)
    rp.add_argument("--out")
    ad = sub.add_parser("ads", help="ads: status, inspect, enqueue one from a template, models")
    ad.add_argument("action", choices=("status", "inspect", "enqueue", "models"))
    ad.add_argument("target", nargs="?", help="inspect: a job id or an ad id; enqueue: a template id")
    ad.add_argument("--brief", help="enqueue: this ad's angle on the template's concept")
    ad.add_argument("--consumer")
    ad.add_argument("--seconds", type=float)
    ad.add_argument("--shots", type=int)
    ad.add_argument("--seed", type=int)
    ad.add_argument("--force", action="store_true", help="enqueue past the weekly cap and the disk guard")
    ad.add_argument("--rescan", action="store_true", help="models: POST /v1/models/rescan first")
    ch = sub.add_parser("characters", help="character bibles under $FOREMAN_HOME/characters")
    ch.add_argument("action", choices=("list", "show", "import", "approve"))
    ch.add_argument("id_or_dir", nargs="?", help="show/approve: the character id; import: the folder")
    ch.add_argument("files", nargs="*", help="approve: still file names")
    ch.add_argument("--id", dest="char_id")
    ch.add_argument("--name")
    ch.add_argument("--kind", choices=("human", "mascot"))
    ch.add_argument("--look")
    ch.add_argument("--subject")
    ch.add_argument("--voice", help="a Kokoro voice id (or a blend 'am_onyx,bm_george')")
    ch.add_argument("--speed", type=float)
    ch.add_argument("--ref-wav", dest="ref_wav", help="a reference WAV for a Qwen3-TTS voice clone")
    ch.add_argument("--language")
    ch.add_argument("--tags", help="approve: comma-separated tags to set")
    ch.add_argument("--approve", action="store_true", help="import: approve every still (you checked them)")
    ch.add_argument("--unapprove", action="store_true")
    ch.add_argument("--overwrite", action="store_true")
    ph = sub.add_parser("phone", help="phone approvals over Telegram: setup, status, test, resend, unpair")
    ph.add_argument("action", choices=("setup", "status", "test", "resend", "unpair"))
    ph.add_argument("--no-wait", action="store_true",
                    help="setup: print the pairing link and return (the phone service completes the pairing)")
    ph.add_argument("--timeout", type=float, help="setup: wait at most this many seconds for the pairing")
    ph.add_argument("--keep-token", action="store_true", help="unpair: keep the bot token for a new pairing")
    args = ap.parse_args(argv)
    cfg = studio.load_config(args.config, db_path=args.db_path)
    try:
        if args.cmd == "submit-speech":
            _submit(cfg, speech_request(args.text, args.voice, args.speed, args.expected_s))
        elif args.cmd == "submit-music":
            _submit(cfg, music_request(args.prompt, args.seconds, args.lyrics, args.seed, args.expected_s))
        elif args.cmd == "submit-video":
            _submit(cfg, video_request(args.prompt, args.seconds, args.width, args.height, args.steps, args.seed,
                                       args.expected_s))
        elif args.cmd == "status":
            cmd_status(cfg, args.live)
        elif args.cmd == "jobs":
            cmd_jobs(cfg, args.all)
        elif args.cmd == "events":
            cmd_events(cfg, max(1, args.tail), args.job)
        elif args.cmd == "cancel":
            sup = _client(cfg)
            try:
                ok = sup.cancel_job(args.job_id, "cancelled from the CLI")
            finally:
                sup.close()
            print(f"job {args.job_id}: {'cancel recorded (gpud finishes it on its next tick)' if ok else 'not cancellable'}")
            return 0 if ok else 1
        elif args.cmd == "clear-safe":
            sup = _client(cfg)
            try:
                sup.clear_safe()
            finally:
                sup.close()
            print("SAFE mode cleared; every brain gets a fresh set of load attempts")
        elif args.cmd == "review":
            cmd_review(cfg, args.all)
        elif args.cmd == "verdict":
            cmd_verdict(cfg, args.job_id, args.verdict, args.minutes, args.note, args.tags)
        elif args.cmd == "ship":
            cmd_ship(cfg, args.job_id, args.usd)
        elif args.cmd == "resolve":
            return cmd_resolve(cfg, args.page_ids, args.note)
        elif args.cmd == "templates":
            return cmd_templates(cfg, args.action, args.template_id, args.note)
        elif args.cmd == "planner":
            cmd_planner(cfg, args.action, args.note)
        elif args.cmd == "report":
            cmd_report(cfg, max(1.0, args.days), args.out)
        elif args.cmd == "ads":
            return cmd_ads(cfg, args)
        elif args.cmd == "characters":
            if args.action == "import":
                args.dir, args.id = args.id_or_dir, args.char_id
            else:
                args.id = args.id_or_dir or args.char_id
            return cmd_characters(cfg, args)
        elif args.cmd == "phone":
            from . import phone
            try:
                if args.action == "setup":
                    return phone.cli_setup(cfg, wait=not args.no_wait, timeout=args.timeout)
                if args.action == "status":
                    return phone.cli_status(cfg)
                if args.action == "test":
                    return phone.cli_test(cfg)
                if args.action == "resend":
                    return phone.cli_resend(cfg)
                return phone.cli_unpair(cfg, keep_token=args.keep_token)
            except phone.PhoneError as e:
                print(f"phone: {e}", file=sys.stderr)
                return 1
            except (OSError, sqlite3.Error) as e:  # an unreadable file or database: a message, not a traceback
                print(f"phone: {phone.redact(f'{type(e).__name__}: {e}')}", file=sys.stderr)
                return 1
    except AdmissionError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except ledger.LedgerError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
