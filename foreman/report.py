"""The scoreboard (`python3 -m foreman.cli report`): what the planner's work earned against what it
cost. Markdown, written under $FOREMAN_HOME/reports/ (never into the repository).

Per template: produced, passed by the brain, accepted and rejected by the owner, shipped, dollars,
review minutes, GPU-hours, P(ship). Then GPU-hours and the idle share, the biggest GPU sinks that
nobody used, rejections by reason code, templates paused by the kill criteria, and the planner's
ideas for new templates.
"""
from __future__ import annotations

import collections
import os
import time

from . import ledger
from .planner import REASONS, status as planner_status
from .text import printable

DAY = 86400.0


def _h(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f}"


def _cell(text) -> str:
    """Untrusted text (briefs, ideas, details) as one Markdown table cell or list item: escaped
    (no control characters or line breaks), no pipes, no HTML."""
    return printable(text).replace("|", "/").replace("<", "&lt;").replace(">", "&gt;")


def build(store, templates: dict, now: float, wall: float, days: float = 7.0, cfg=None) -> str:
    window = days * DAY
    lo = now - window
    gpu = ledger.gpu_usage(store, now, window)
    stats = ledger.template_stats(store, templates, now, gpu["by_root"])
    all_items = ledger.items(store)
    paused = ledger.template_states(store)
    pst = planner_status(store, now)
    first = store.conn.execute("SELECT MIN(created_at) AS t FROM jobs").fetchone()["t"]
    span = min(window, max(0.0, now - first)) if first is not None else 0.0
    roots = {i["job_id"] for i in all_items}
    planner_s = sum(s for root, s in gpu["by_root"].items() if root in roots)
    plan_jobs = store.conn.execute("SELECT id, started_at, finished_at FROM jobs WHERE kind = 'plan' "
                                   "AND started_at IS NOT NULL AND (finished_at IS NULL OR finished_at >= ?)", (lo,))
    plan_s = sum(max(0.0, min(r["finished_at"] if r["finished_at"] is not None else now, now) - max(r["started_at"], lo))
                 for r in plan_jobs)
    key = "7d" if days <= 7 else "28d"
    out = ["# Foreman scoreboard", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M', time.localtime(wall))} local time. "
           f"Window: the last {days:g} days (template columns use the {key} ledger window).", ""]
    out += ["## Planner", "",
            f"- state: **{pst['state']}**" + (f" ({pst['paused_note']})" if pst.get("paused_note") else ""),
            f"- ticks: {pst['ticks_7d']} in the last 7 days, {pst['ticks']} in all; interval now "
            f"{(pst['interval_s'] or 0) / 60:.0f} min after {pst['empty_streak']} empty tick(s) in a row",
            f"- tokens on {pst['tokens_day'] or '-'}: {pst['tokens_used']}", ""]
    out += ["## Templates", "",
            "| template | admitted | produced | brain ok | accepted | rejected | shipped | usd | review min | "
            "GPU h | P(ship) | status |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tid in sorted(stats):
        s = stats[tid][key]
        ps = paused.get(tid) or {}
        t = templates.get(tid)
        status = (f"paused: {ps.get('reason')}" if ps.get("paused") else
                  "not loaded" if t is None else "active" if t.active else "inactive")
        out.append(f"| {_cell(tid)} | {s['admitted']} | {s['produced']} | {s['brain_accepted']} | {s['accepted']} | "
                   f"{s['rejected']} | {s['shipped']} | {s['usd']:.2f} | {s['review_min']:g} | {s['gpu_h']:.2f} | "
                   f"{stats[tid]['p_ship'] if stats[tid]['p_ship'] is not None else '-'} | {_cell(status)} |")
    if not stats:
        out.append("| (no templates and no items yet) | | | | | | | | | | | |")
    total_usd = sum(stats[t][key]["usd"] for t in stats)
    total_min = sum(stats[t][key]["review_min"] for t in stats)
    out += ["", f"Shipped value ${total_usd:.2f} for {total_min:g} review minutes"
            + (f" (${total_usd / total_min:.2f} per minute)." if total_min else "."), ""]
    idle = 1.0 - gpu["total"] / span if span > 0 else 1.0
    out += ["## GPU", "",
            f"- GPU-hours: {_h(gpu['total'])} (planner items {_h(planner_s)}, plan ticks {_h(plan_s)}, "
            f"other {_h(max(0.0, gpu['total'] - planner_s - plan_s))})",
            f"- idle share: {idle:.0%} of the {span / 3600.0:.1f} h observed (idle is a correct outcome)", ""]
    sinks = []
    for it in all_items:
        if it["state"] in ("accepted", "shipped"):
            continue
        s = gpu["by_root"].get(it["job_id"], 0.0)
        if s > 0:
            sinks.append((s, it))
    sinks.sort(key=lambda x: (-x[0], x[1]["id"]))
    out += ["## Top GPU sinks with no use", ""]
    if sinks:
        out += ["| job | template | GPU min | state | brief |", "|---|---|---|---|---|"]
        for s, it in sinks[:5]:
            out.append(f"| {it['job_id']} | {_cell(it['template_id'])} | {s / 60.0:.1f} | {it['state']} | "
                       f"{_cell(it['brief'][:80])} |")
    else:
        out.append("None.")
    out.append("")
    rej = collections.Counter(r["reason"] for r in ledger.rejections_since(store, lo))
    out += ["## Rejected proposals by reason", ""]
    if rej:
        out += ["| reason | count | meaning |", "|---|---|---|"]
        for reason, n in sorted(rej.items(), key=lambda x: (-x[1], x[0])):
            out.append(f"| {reason} | {n} | {_cell(REASONS.get(reason, ''))} |")
    else:
        out.append("None.")
    out.append("")
    killed = [(tid, s) for tid, s in sorted(paused.items()) if s["paused"]]
    out += ["## Paused templates", ""]
    if killed:
        for tid, s in killed:
            out.append(f"- {_cell(tid)}: {_cell(s['reason'])}" + (f" ({_cell(s['detail'])})" if s.get("detail") else ""))
    else:
        out.append("None.")
    out.append("")
    ideas = [i for e in store.events("plan_ideas") if e["at"] >= lo for i in (e["data"].get("ideas") or [])]
    out += ["## New template ideas from the planner (for the owner to consider)", ""]
    out += [f"- {_cell(i)}" for i in dict.fromkeys(ideas)] or ["None."]
    out.append("")
    return "\n".join(out)


def write(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
