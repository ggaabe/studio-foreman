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
    python3 -m foreman.cli verdict JOB accept|reject [--minutes M] [--note TEXT]
    python3 -m foreman.cli ship JOB [--usd X]             you used it (and what it earned)
    python3 -m foreman.cli resolve [PAGE ...] [--note TEXT]   close pages (no id: list open pages)
    python3 -m foreman.cli templates [list | pause ID | unpause ID]
    python3 -m foreman.cli planner status|pause|resume|plan-now [--note TEXT]
    python3 -m foreman.cli report [--days 7] [--out PATH]  Markdown scoreboard

The CLI never talks to mlx-serve (except `status --live`, which only reads /api/ps) and never
takes gpud's lock: it opens the same SQLite file as a non-leader client (Supervisor(lead=False))
and goes through the supervisor's public submit(), so every admission rule applies. The running
gpud picks the job up on its next tick.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import ledger, planner, report, studio
from .clock import SystemClock
from .media_requests import music_request, speech_request, video_expected_s, video_request  # noqa: F401
from .supervisor import AdmissionError, Supervisor
from .templates import load_templates


class _ClientEngine:
    """Admission only checks that an engine is configured; a client never calls it."""

    name = studio.ENGINE

    def __getattr__(self, item):
        raise AttributeError(f"the CLI does not talk to engines (asked for {item!r})")


def _client(cfg) -> Supervisor:
    if not os.path.exists(os.path.dirname(cfg.db_path) or "."):
        os.makedirs(os.path.dirname(cfg.db_path), exist_ok=True)
    return Supervisor(cfg.db_path, {studio.ENGINE: _ClientEngine()}, SystemClock(), studio.policy(),
                      catalog=studio.catalog(cfg.video_exclusive), lead=False)


def _when(wall) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall)) if wall else "-"


def _short(model) -> str:
    return str(model).split("/")[-1] if model else "-"


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
    finally:
        sup.close()
    by_state: dict = {}
    for j in jobs:
        by_state[j.state] = by_state.get(j.state, 0) + 1
    print(f"db               {cfg.db_path}")
    print(f"mode             {st['mode']}" + (f"  SAFE: {st['safe_reason']}" if st["safe"] else ""))
    print(f"resident         {', '.join(_short(m) for m in sorted(st['known_resident'] or {})) or '-'}")
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
        print(f"  {j.id:>5} {j.state:<15} {j.kind:<16} {_short(j.requires):<34} attempt {j.attempt}")
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
        print(f"{j.id:>5} {j.state:<15} {j.kind:<16} {_short(j.requires):<34} {j.attempt:>3} "
              f"{j.parent_id if j.parent_id else '-':>6}  {reason}")


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
    finally:
        sup.close()
    if not queue:
        print("review queue is empty")
    for it in queue:
        age = (now - (it.get("produced_at") or now)) / 3600.0
        print(f"job {it['job_id']:<5} {it['template_id']}  for {it['consumer']!r}  "
              f"waiting {age:.1f} h  (review ~{it['review_minutes'] or 0:g} min)")
        print(f"    brief: {it['brief'][:200]}")
        if it.get("brain_verdict"):
            print(f"    brain: {it['brain_verdict']}" + (f": {it['brain_note'][:200]}" if it.get("brain_note") else ""))
        for label, path in _files(it.get("artifacts")):
            print(f"    {label:<4} {path}")
        print(f"    -> python3 -m foreman.cli verdict {it['job_id']} accept|reject [--minutes M] [--note TEXT]")
    for it in recent:
        extra = it.get("verdict_note") or it.get("fail_reason") or ""
        print(f"  [{it['state']:<8}] job {it['job_id']:<5} {it['template_id']}: {it['brief'][:80]}"
              + (f"  ({extra[:80]})" if extra else ""))


def cmd_verdict(cfg, job_id: int, verdict: str, minutes, note) -> None:
    templates, _ = _templates(cfg)
    sup = _client(cfg)
    try:
        now = sup._now()
        it = ledger.record_verdict(sup.store, job_id, verdict, minutes, note, now)
        t = templates.get(it["template_id"])
        killed = ledger.kill_check(sup.store, templates, now, cfg.planner_config()) if templates else []
        p = ledger.posterior(t, ledger.items(sup.store, template=t.id))[2] if t else None
    finally:
        sup.close()
    print(f"job {job_id}: {verdict} recorded for {it['template_id']} ({it['verdict_minutes']:g} review min)"
          + (f"; P(ship) is now {p:.2f}" if p is not None else ""))
    for tid, reason, detail in killed:
        print(f"template {tid} paused by a kill criterion: {reason} ({detail})")


def cmd_ship(cfg, job_id: int, usd) -> None:
    sup = _client(cfg)
    try:
        it = ledger.record_ship(sup.store, job_id, usd, sup._now())
    finally:
        sup.close()
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
                print(f"page {j.id:<5} {j.kind:<18} {reason[:160]}")
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
        print(f"{t_id:<34} {(t.kind if t else '-'):<14} {status[:26]:<26} {('%.2f' % p) if p is not None else '-':>7} "
              f"{week:>6} {stock:>7} {flow:>21} {(s.get('28d') or {}).get('usd', 0):>8.2f}")
        if ps.get("paused") and ps.get("detail"):
            print(f"    {ps['detail']}")
    if not templates and not stats:
        print(f"(no templates in {os.path.join(cfg.home, 'templates')}; see examples/templates)")
    for e in errors:
        print(f"error: {e}")
    return 1 if errors else 0


def cmd_planner(cfg, action: str, note) -> None:
    sup = _client(cfg)
    try:
        now = sup._now()
        if action == "pause":
            planner.set_paused(sup.store, True, note, now)
        elif action == "resume":
            planner.set_paused(sup.store, False, note, now)
        elif action == "plan-now":
            planner.plan_now(sup.store, now)
        st = planner.status(sup.store, now)
    finally:
        sup.close()
    if action == "pause":
        print("planner paused (gpud keeps running the queue; `planner resume` to plan again)")
    elif action == "resume":
        print("planner resumed (suspension and invalid streak cleared)")
    elif action == "plan-now":
        print("plan-now recorded: gpud plans on its next tick once a brain is home and the GPU is free"
              + ("" if st["state"] == "active" else f" (but the planner is {st['state']}: resume it first)"))
    due = st["next_due_in_s"]
    print(f"planner          {st['state']}" + (f" ({st['paused_note']})" if st.get("paused_note") else ""))
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
    vd = sub.add_parser("verdict", help="accept or reject a reviewed item (updates P(ship))")
    vd.add_argument("job_id", type=int)
    vd.add_argument("verdict", choices=("accept", "reject"))
    vd.add_argument("--minutes", type=float, help="review minutes you spent (default: the template's estimate)")
    vd.add_argument("--note")
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
            cmd_verdict(cfg, args.job_id, args.verdict, args.minutes, args.note)
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
    except AdmissionError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except ledger.LedgerError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
