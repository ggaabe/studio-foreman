"""The value ledger behind the planner: one row per admitted proposal (plan_items), the owner's
verdicts, what shipped, review minutes, GPU time per lineage, and the per-template Beta
posteriors and kill criteria computed from them.

Everything here is a function of the database, so gpud and the CLI (a non-leader client on the
same file) see the same numbers. Item state changes are compare-and-set, like job transitions:

    queued    admitted; its gpud lineage (the job, retries, the brain's review) is still running
    review    the media succeeded and the brain's review accepted it or asked for a human:
              it waits for the owner's verdict (the review queue)
    accepted  the owner accepted it               -> shipped (the owner used it; usd recorded)
    rejected  the owner rejected it               (artifacts pruned after 7 days)
    failed    the lineage ended without output the brain would pass on (artifacts pruned after 14 days)
    expired   it waited 14 days without a verdict (artifacts pruned)

P(ship) per template is Beta(prior.alpha + accepts, prior.beta + rejects): only the owner's
verdicts move it, never the brain's review and never the planner.
"""
from __future__ import annotations

import json
import os

from .policy import number
from .store import TERMINAL, Job

ITEM_JSON = ("params", "flags", "artifacts", "artifact_dirs")
STOCK_STATES = ("queued", "review", "accepted")
LIVE = ("queued", "starting", "running", "cancelling")
DAY = 86400.0
MAX_USD = 1e9  # what one shipped item earned, at most (a bigger number is a typo)


def _enc(v):
    return json.dumps(v, sort_keys=True, default=str)


def _item(row) -> dict:
    d = dict(row)
    for k in ITEM_JSON:
        if d.get(k) is not None:
            d[k] = json.loads(d[k])
    return d


# ---------------------------------------------------------------- items

def insert_item(store, **cols) -> int:
    for k in ITEM_JSON:
        if k in cols:
            cols[k] = _enc(cols[k])
    names, marks = ", ".join(cols), ", ".join("?" * len(cols))
    cur = store.conn.execute(f"INSERT INTO plan_items ({names}) VALUES ({marks})", tuple(cols.values()))
    return cur.lastrowid


def update_item(store, item_id: int, expect_state=None, **cols) -> bool:
    for k in ITEM_JSON:
        if k in cols:
            cols[k] = _enc(cols[k])
    sets = ", ".join(f"{k} = ?" for k in cols)
    sql, args = f"UPDATE plan_items SET {sets} WHERE id = ?", [*cols.values(), item_id]
    if expect_state is not None:
        states = (expect_state,) if isinstance(expect_state, str) else tuple(expect_state)
        sql += f" AND state IN ({', '.join('?' * len(states))})"
        args += list(states)
    return store.conn.execute(sql, args).rowcount == 1


def items(store, *states, template: str | None = None) -> list:
    sql, args = "SELECT * FROM plan_items WHERE 1=1", []
    if states:
        sql += f" AND state IN ({', '.join('?' * len(states))})"
        args += list(states)
    if template is not None:
        sql += " AND template_id = ?"
        args.append(template)
    return [_item(r) for r in store.conn.execute(sql + " ORDER BY id", args)]


def item(store, item_id: int) -> dict | None:
    row = store.conn.execute("SELECT * FROM plan_items WHERE id = ?", (item_id,)).fetchone()
    return _item(row) if row else None


def item_for_job(store, job_id: int) -> dict | None:
    """The item a job belongs to: any job of its lineage (the planned job, a retry, the brain's
    review) leads back to it."""
    job = store.job(job_id)
    if job is None:
        return None
    root = job.root_id or job.id
    row = store.conn.execute("SELECT * FROM plan_items WHERE job_id = ?", (root,)).fetchone()
    return _item(row) if row else None


def hash_exists(store, content_hash: str) -> bool:
    return store.conn.execute("SELECT 1 FROM plan_items WHERE content_hash = ?", (content_hash,)).fetchone() is not None


def record_rejection(store, plan_job_id, at: float, template_id, reason: str, detail: str | None) -> None:
    store.conn.execute("INSERT INTO plan_rejections (plan_job_id, at, template_id, reason, detail) VALUES (?, ?, ?, ?, ?)",
                       (plan_job_id, at, template_id[:80] if isinstance(template_id, str) else None, reason,
                        None if detail is None else str(detail)[:500]))


def rejections_since(store, since: float) -> list:
    rows = store.conn.execute("SELECT * FROM plan_rejections WHERE at >= ? ORDER BY id", (since,))
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- planner state & template pauses

def planner_state(store) -> dict:
    return dict(store.conn.execute("SELECT * FROM planner_state WHERE id = 1").fetchone())


def set_planner_state(store, **cols) -> None:
    sets = ", ".join(f"{k} = ?" for k in cols)
    store.conn.execute(f"UPDATE planner_state SET {sets} WHERE id = 1", tuple(cols.values()))


def template_states(store) -> dict:
    return {r["template_id"]: dict(r) for r in store.conn.execute("SELECT * FROM template_state")}


def set_template_paused(store, template_id: str, paused: bool, reason: str | None, detail: str | None,
                        now: float) -> None:
    """Pause (by the owner or a kill criterion) or unpause. Unpausing restarts the history the kill
    criteria look at, so a template the owner brought back is judged on what it does next."""
    with store.tx():
        cur = template_states(store).get(template_id)
        since = now if not paused else (cur or {}).get("since")
        store.conn.execute(
            "INSERT INTO template_state (template_id, paused, reason, detail, at, since) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(template_id) DO UPDATE SET paused = excluded.paused, reason = excluded.reason, "
            "detail = excluded.detail, at = excluded.at, since = excluded.since",
            (template_id, int(bool(paused)), reason, detail, now, since))
        store.event(now, "template_paused" if paused else "template_unpaused", None, template=template_id,
                    reason=reason, detail=detail)


def paused_ids(store) -> dict:
    return {tid: s for tid, s in template_states(store).items() if s["paused"]}


# ---------------------------------------------------------------- value

def posterior(template, its) -> tuple:
    """(alpha, beta, mean) of P(ship): the template's prior plus the owner's verdicts."""
    acc = sum(1 for i in its if i.get("verdict") == "accept")
    rej = sum(1 for i in its if i.get("verdict") == "reject")
    a, b = template.alpha + acc, template.beta + rej
    return a, b, a / (a + b)


def shipped_value(it: dict) -> float:
    if it.get("shipped_at") is None:
        return 0.0
    return float(it["usd"]) if it.get("usd") is not None else float(it.get("value_usd") or 0.0)


def realized_value(it: dict) -> float:
    """What an item is worth to the kill criteria: its shipped value, or for an item the owner
    accepted and has not shipped yet (work is often shipped in batches), the template's value_usd."""
    if it.get("shipped_at") is not None:
        return shipped_value(it)
    if it.get("verdict") == "accept":
        return float(it.get("value_usd") or 0.0)
    return 0.0


# ---------------------------------------------------------------- GPU time

def gpu_usage(store, now: float, window_s: float) -> dict:
    """GPU-busy seconds inside [now - window_s, now]: every job that held the GPU (brain and
    media; one runs at a time), in total and per lineage root."""
    lo = now - window_s
    rows = store.conn.execute(
        "SELECT id, root_id, kind, started_at, finished_at FROM jobs WHERE started_at IS NOT NULL "
        "AND requires != 'human' AND (finished_at IS NULL OR finished_at >= ?)", (lo,))
    total, by_root = 0.0, {}
    for r in rows:
        end = r["finished_at"] if r["finished_at"] is not None else now
        d = max(0.0, min(end, now) - max(r["started_at"], lo))
        if d <= 0:
            continue
        total += d
        root = r["root_id"] or r["id"]
        by_root[root] = by_root.get(root, 0.0) + d
    return {"total": total, "by_root": by_root, "window_s": window_s}


# ---------------------------------------------------------------- following an item's lineage

def artifact_paths(job: Job) -> list:
    art = (job.outcome or {}).get("artifact") if isinstance(job.outcome, dict) else None
    if not isinstance(art, dict):
        return []
    out = []
    for p in [art.get("path")] + list(art.get("paths") or []):
        if isinstance(p, str) and p and p not in out:
            out.append(p)
    return out


def settle(store, it: dict) -> dict | None:
    """Where an admitted item's lineage ended up, or None while it is still in progress."""
    rows = store.conn.execute("SELECT * FROM jobs WHERE root_id = ? ORDER BY id", (it["job_id"],)).fetchall()
    media = [j for j in (Job(r) for r in rows) if j.brainless]
    if not media:
        job = store.job(it["job_id"])
        if job is None:
            return {"state": "failed", "fail_reason": "its gpud job no longer exists", "artifact_dirs": []}
        media = [job]
    latest = media[-1]
    dirs = sorted({os.path.dirname(p) for j in media for p in artifact_paths(j)})
    if latest.state not in TERMINAL:
        return None
    cont = store.continuation(latest.id)
    if not cont or not cont.get("fired_job_id"):
        return None
    child = store.job(cont["fired_job_id"])
    if child is None or child.state in LIVE:
        return None
    art = (child.outcome or {}).get("artifact") if isinstance(child.outcome, dict) else None
    decision = art.get("decision") if isinstance(art, dict) and isinstance(art.get("decision"), dict) else {}
    verdict = decision.get("verdict") if child.requires == "brain" else None
    note = decision.get("note") if isinstance(decision.get("note"), str) else None
    common = {"output_job_id": latest.id, "review_job_id": child.id, "brain_verdict": verdict,
              "brain_note": (note or "")[:600] or None, "artifact_dirs": dirs}
    if latest.state == "succeeded" and child.state == "succeeded" and verdict in ("accept", "human_review"):
        return dict(common, state="review", artifacts=artifact_paths(latest))
    if latest.state != "succeeded":
        o = latest.outcome or {}
        why = f"{latest.state} ({o.get('reason')}): {str(o.get('detail') or '')[:160]}"
    elif child.requires == "human":
        why = f"handed to a human: {str((child.payload or {}).get('reason') or '')[:160]}"
    else:
        why = f"the brain's review said {verdict}: {(note or '')[:160]}"
    return dict(common, state="failed", fail_reason=why[:300], artifacts=artifact_paths(latest))


def review_queue(store) -> list:
    return sorted(items(store, "review"),
                  key=lambda i: (i["produced_at"] if i.get("produced_at") is not None else 0.0, i["id"]))


# ---------------------------------------------------------------- the owner's actions

class LedgerError(ValueError):
    pass


def _item_or_raise(store, job_id: int) -> dict:
    it = item_for_job(store, job_id)
    if it is None:
        raise LedgerError(f"job {job_id} is not a planner item (see `review` for the items awaiting a verdict)")
    return it


def resolve_page(store, page_id: int, note: str | None, now: float) -> None:
    job = store.job(page_id)
    if job is None:
        raise LedgerError(f"no job {page_id}")
    if job.requires != "human":
        raise LedgerError(f"job {page_id} ({job.kind}) is not a page to a human")
    with store.tx():
        if not store.update_job(page_id, expect_state="awaiting_human", state="resolved", finished_at=now,
                                outcome={"status": "resolved", "note": note, "resolved_at": now}):
            raise LedgerError(f"page {page_id} is {store.job(page_id).state}, not awaiting a human")
        store.event(now, "page_resolved", page_id, kind=job.kind, note=note)


def record_verdict(store, job_id: int, verdict: str, minutes: float | None, note: str | None, now: float) -> dict:
    if verdict not in ("accept", "reject"):
        raise LedgerError("a verdict is accept or reject")
    if minutes is not None and (number(minutes) is None or minutes < 0 or minutes > 600):
        raise LedgerError("--minutes must be a number between 0 and 600")
    it = _item_or_raise(store, job_id)
    if it["state"] != "review":
        raise LedgerError(f"item {it['id']} (job {it['job_id']}) is {it['state']}, not awaiting a verdict")
    mins = float(it["review_minutes"] or 0.0) if minutes is None else float(minutes)
    with store.tx():
        if not update_item(store, it["id"], expect_state="review", state="accepted" if verdict == "accept" else "rejected",
                           verdict=verdict, verdict_at=now, verdict_minutes=mins, verdict_note=note):
            raise LedgerError(f"item {it['id']} changed while recording the verdict; try again")
        store.event(now, "verdict_recorded", it["job_id"], item=it["id"], template=it["template_id"],
                    verdict=verdict, minutes=mins)
        if it.get("review_job_id"):  # the brain asked for a person about this item: that is this verdict
            for page in store.children(it["review_job_id"]):
                if page.requires == "human" and page.state == "awaiting_human":
                    store.update_job(page.id, expect_state="awaiting_human", state="resolved", finished_at=now,
                                     outcome={"status": "resolved", "note": f"verdict {verdict} recorded",
                                              "resolved_at": now})
                    store.event(now, "page_resolved", page.id, kind=page.kind, note=f"verdict {verdict}")
    return item(store, it["id"])


def record_ship(store, job_id: int, usd: float | None, now: float) -> dict:
    if usd is not None and (number(usd) is None or usd < 0 or usd > MAX_USD):
        raise LedgerError(f"--usd must be a finite number in 0..{MAX_USD:g}")
    it = _item_or_raise(store, job_id)
    if it["state"] != "accepted":
        hint = " (record `verdict JOB accept` first)" if it["state"] == "review" else ""
        raise LedgerError(f"item {it['id']} (job {it['job_id']}) is {it['state']}, not accepted{hint}")
    with store.tx():
        if not update_item(store, it["id"], expect_state="accepted", state="shipped", shipped_at=now, usd=usd):
            raise LedgerError(f"item {it['id']} changed while recording the shipment; try again")
        store.event(now, "item_shipped", it["job_id"], item=it["id"], template=it["template_id"], usd=usd)
    return item(store, it["id"])


# ---------------------------------------------------------------- stats & kill criteria

def template_stats(store, templates: dict, now: float, gpu_by_root: dict | None = None) -> dict:
    gpu_by_root = gpu_by_root if gpu_by_root is not None else gpu_usage(store, now, 28 * DAY)["by_root"]
    by_t: dict = {}
    for it in items(store):
        by_t.setdefault(it["template_id"], []).append(it)
    out = {}
    for tid in sorted(set(templates) | set(by_t)):
        its = by_t.get(tid, [])

        def window(days: float, its=its) -> dict:
            lo = now - days * DAY
            produced = [i for i in its if i.get("produced_at") is not None and i["produced_at"] >= lo]
            judged = [i for i in its if i.get("verdict_at") is not None and i["verdict_at"] >= lo]
            shipped = [i for i in its if i.get("shipped_at") is not None and i["shipped_at"] >= lo]
            return {"admitted": sum(1 for i in its if i["created_at"] >= lo), "produced": len(produced),
                    "brain_accepted": sum(1 for i in produced if i.get("brain_verdict") == "accept"),
                    "accepted": sum(1 for i in judged if i["verdict"] == "accept"),
                    "rejected": sum(1 for i in judged if i["verdict"] == "reject"),
                    "shipped": len(shipped), "usd": round(sum(shipped_value(i) for i in shipped), 2),
                    "review_min": round(sum(float(i.get("verdict_minutes") or 0) for i in judged), 1),
                    "gpu_h": round(sum(gpu_by_root.get(i["job_id"], 0.0) for i in its
                                       if i["created_at"] >= lo) / 3600.0, 3)}
        t = templates.get(tid)
        post = posterior(t, its) if t is not None else None
        out[tid] = {"7d": window(7), "28d": window(28),
                    "stock": sum(1 for i in its if i["state"] in STOCK_STATES),
                    "week": sum(1 for i in its if i["created_at"] >= now - 7 * DAY),
                    "awaiting_review": sum(1 for i in its if i["state"] == "review"),
                    "p_ship": round(post[2], 4) if post else None,
                    "verdicts": sum(1 for i in its if i.get("verdict"))}
    return out


def kill_check(store, templates: dict, now: float, cfg) -> list:
    """Pause templates that are not earning their review time (code, never the LLM):
    acceptance below cfg.kill_min_acceptance after cfg.kill_min_verdicts verdicts; nothing accepted or
    shipped in cfg.kill_unused_days; value per review-minute below cfg.kill_min_value_per_min over
    cfg.kill_value_days (shipped value, plus value_usd for accepted items not shipped yet).
    Returns [(template_id, reason, detail)] for the templates paused now."""
    states = template_states(store)
    paused = []
    for tid in sorted(templates):
        st = states.get(tid) or {}
        if st.get("paused"):
            continue
        since = st.get("since")
        since = float("-inf") if since is None else since
        its = [i for i in items(store, template=tid) if i["created_at"] >= since]
        if not its:
            continue
        judged = [i for i in its if i.get("verdict")]
        reason = detail = None
        n = len(judged)
        acc = sum(1 for i in judged if i["verdict"] == "accept")
        unused_lo = now - cfg.kill_unused_days * DAY
        if n >= cfg.kill_min_verdicts and acc / float(n) < cfg.kill_min_acceptance:
            reason, detail = "kill_low_acceptance", f"{acc}/{n} accepted ({acc / float(n):.0%} < {cfg.kill_min_acceptance:.0%})"
        elif min(i["created_at"] for i in its) <= unused_lo and since <= unused_lo and not any(
                (i.get("verdict") == "accept" and i.get("verdict_at") is not None and i["verdict_at"] >= unused_lo)
                or (i.get("shipped_at") is not None and i["shipped_at"] >= unused_lo) for i in its):
            reason, detail = "kill_unused_30d", f"nothing it produced was accepted or used in {cfg.kill_unused_days:g} days"
        elif judged:
            lo = now - cfg.kill_value_days * DAY
            recent = [i for i in judged if i["verdict_at"] >= lo]
            first = min(i["verdict_at"] for i in judged)
            minutes = sum(float(i.get("verdict_minutes") or 0) for i in recent)
            if first <= lo and recent and minutes > 0:
                value = sum(realized_value(i) for i in recent)
                if value / minutes < cfg.kill_min_value_per_min:
                    reason = "kill_low_value"
                    detail = (f"${value:.2f} shipped or accepted for {minutes:.0f} review minutes over "
                              f"{cfg.kill_value_days:g} days (${value / minutes:.2f}/min < "
                              f"${cfg.kill_min_value_per_min:g}/min)")
        if reason:
            set_template_paused(store, tid, True, reason, detail, now)
            paused.append((tid, reason, detail))
    return paused
