"""SQLite (WAL) persistence. The database is the supervisor's only memory: a fresh Supervisor
on the same file must be able to carry on from any committed state.

Every job state transition is a compare-and-set (`update_job(..., expect_state=...)`), so a
stale writer (a second gpud, or a code path working from an old snapshot) can never move a job
out of a state it did not observe. Lease renewals are additionally fenced by the leader epoch.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

TERMINAL = ("succeeded", "failed", "quarantined")
ACTIVE = ("starting", "running", "cancelling")
_JOB_JSON = ("payload", "outcome", "pending_outcome")
_STATE_JSON = ("ladder", "loading", "known_resident", "engine_health", "orphans")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  kind                TEXT NOT NULL,
  requires            TEXT NOT NULL,              -- 'brain' | a media model | 'human'
  state               TEXT NOT NULL,              -- queued starting running cancelling succeeded
                                                  -- failed quarantined awaiting_human resolved
  payload             TEXT NOT NULL DEFAULT '{}',
  outcome             TEXT,
  parent_id           INTEGER REFERENCES jobs(id),
  root_id             INTEGER,                    -- lineage root (the client submission)
  depth               INTEGER NOT NULL DEFAULT 0,
  task_key            TEXT,                       -- retries of one logical task share it
  content_key         TEXT,                       -- hash of (kind, requires, payload)
  attempt             INTEGER NOT NULL DEFAULT 1,
  max_attempts        INTEGER NOT NULL,
  interruptions       INTEGER NOT NULL DEFAULT 0, -- gpud/engine-side interruptions (not charged)
  respawns            INTEGER NOT NULL DEFAULT 0, -- engine-side requeues (new run id suffix)
  faults              INTEGER NOT NULL DEFAULT 0, -- gpud exceptions while handling this job
  priority            INTEGER NOT NULL DEFAULT 0,
  deadline            REAL,
  idempotency_key     TEXT UNIQUE,                -- client keys are stored as 'client:<key>'
  is_continuation     INTEGER NOT NULL DEFAULT 0,
  created_at          REAL NOT NULL,
  started_at          REAL,
  finished_at         REAL,
  engine              TEXT,
  model               TEXT,
  run_id              TEXT,
  handle              TEXT,
  pending_outcome     TEXT,                       -- outcome to record once cancel cleanup ends
  cancel_requested_at REAL,
  cancel_stage        TEXT,                       -- cancel | cancel_sent | cancel_failed
  paged_at            REAL,
  page_failures       INTEGER NOT NULL DEFAULT 0,
  page_retry_at       REAL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent_id);

CREATE TABLE IF NOT EXISTS continuations (
  job_id        INTEGER PRIMARY KEY REFERENCES jobs(id),
  on_success    TEXT NOT NULL,
  on_failure    TEXT NOT NULL,
  fired_outcome TEXT,                             -- success | failure | quarantine | depth_limit | error
  fired_job_id  INTEGER UNIQUE REFERENCES jobs(id),
  fired_at      REAL
);

CREATE TABLE IF NOT EXISTS leases (
  job_id             INTEGER PRIMARY KEY REFERENCES jobs(id),
  engine             TEXT NOT NULL,
  model              TEXT NOT NULL,
  granted_at         REAL NOT NULL,
  expires_at         REAL NOT NULL,
  hard_deadline      REAL NOT NULL,
  last_progress_at   REAL NOT NULL,
  last_progress      REAL NOT NULL DEFAULT 0,
  progress_timeout_s REAL,                        -- per job (floor: Policy.progress_timeout_s)
  fence              INTEGER NOT NULL DEFAULT 0,  -- leader epoch that owns the lease
  state              TEXT NOT NULL                -- active | expired | revoked | released
);

CREATE TABLE IF NOT EXISTS events (
  id     INTEGER PRIMARY KEY AUTOINCREMENT,
  at     REAL NOT NULL,                           -- supervisor (monotonic) time
  wall   REAL,                                    -- wall clock, for humans
  kind   TEXT NOT NULL,
  job_id INTEGER,
  data   TEXT NOT NULL DEFAULT '{}'
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
  BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
  BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;

CREATE TABLE IF NOT EXISTS supervisor_state (
  id              INTEGER PRIMARY KEY CHECK (id = 1),
  mode            TEXT NOT NULL DEFAULT 'boot',
  safe            INTEGER NOT NULL DEFAULT 0,
  safe_reason     TEXT,
  safe_epoch      INTEGER NOT NULL DEFAULT 0,
  safe_engine     TEXT,                           -- SAFE caused by one wedged engine (degraded)
  brainless_since REAL,
  brainless_epoch INTEGER NOT NULL DEFAULT 0,
  ladder          TEXT NOT NULL DEFAULT '{}',
  loading         TEXT,
  known_resident  TEXT NOT NULL DEFAULT '{}',     -- {model: engine}
  engine_health   TEXT NOT NULL DEFAULT '{}',
  orphans         TEXT NOT NULL DEFAULT '[]',
  boot_count      INTEGER NOT NULL DEFAULT 0,     -- also the leader epoch (fencing token)
  t_now           REAL,                           -- last supervisor time
  t_wall          REAL,                           -- wall clock at t_now
  t_mono          REAL                            -- monotonic source reading at t_now
);
INSERT OR IGNORE INTO supervisor_state (id) VALUES (1);

-- The planner (foreman/planner.py). Its own state is persisted here so a restarted gpud (or a
-- fresh Planner object) carries on with the same schedule, backoff, budget and suspension.
CREATE TABLE IF NOT EXISTS planner_state (
  id               INTEGER PRIMARY KEY CHECK (id = 1),
  paused           INTEGER NOT NULL DEFAULT 0,   -- the owner paused it (CLI)
  paused_note      TEXT,
  suspended_until  REAL,                         -- supervisor time; set after repeated invalid ticks
  suspensions      INTEGER NOT NULL DEFAULT 0,
  invalid_streak   INTEGER NOT NULL DEFAULT 0,
  empty_streak     INTEGER NOT NULL DEFAULT 0,   -- consecutive ticks that admitted nothing
  interval_s       REAL,                         -- current interval (base, doubled by backoff)
  next_due_at      REAL,
  last_tick_at     REAL,
  tick_seq         INTEGER NOT NULL DEFAULT 0,   -- plan job idempotency: plan:<seq>
  force            INTEGER NOT NULL DEFAULT 0,   -- plan-now from the CLI
  trigger          TEXT,                         -- pending event trigger
  last_event_id    INTEGER NOT NULL DEFAULT 0,   -- events scanned for triggers
  last_review_size INTEGER,
  tokens_day       TEXT,                         -- day (of supervisor time) the token count belongs to
  tokens_used      INTEGER NOT NULL DEFAULT 0,
  engine_fail_streak INTEGER NOT NULL DEFAULT 0  -- plan jobs lost to the engine in a row (backoff, not invalid)
);
INSERT OR IGNORE INTO planner_state (id) VALUES (1);

-- One row per admitted proposal: the value ledger. job_id is the gpud job it became (the root of
-- its lineage: retries and the brain's reviews share root_id). Times are supervisor time; the
-- *_wall / created_day columns are for humans and daily budgets.
CREATE TABLE IF NOT EXISTS plan_items (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  plan_job_id     INTEGER,
  template_id     TEXT NOT NULL,
  consumer        TEXT NOT NULL,
  brief           TEXT NOT NULL,
  params          TEXT NOT NULL DEFAULT '{}',
  content_hash    TEXT NOT NULL UNIQUE,
  job_id          INTEGER UNIQUE,
  state           TEXT NOT NULL,                 -- queued review accepted rejected shipped failed expired
  flags           TEXT NOT NULL DEFAULT '{}',
  value_usd       REAL,
  p_ship          REAL,
  urgency         REAL,
  value_adj       REAL,
  ev              REAL,
  priority        INTEGER,
  est_gpu_s       REAL,
  review_minutes  REAL,
  deadline_wall   REAL,
  hard_deadline   INTEGER NOT NULL DEFAULT 0,
  reserved_slot   INTEGER NOT NULL DEFAULT 0,    -- admitted into a reserved hard-deadline review slot
  rationale       TEXT,
  created_at      REAL NOT NULL,
  created_wall    REAL,
  created_day     TEXT,
  produced_at     REAL,                          -- media succeeded and the brain's review passed it on
  output_job_id   INTEGER,
  review_job_id   INTEGER,
  brain_verdict   TEXT,
  brain_note      TEXT,
  artifacts       TEXT NOT NULL DEFAULT '[]',
  artifact_dirs   TEXT NOT NULL DEFAULT '[]',
  finished_at     REAL,
  fail_reason     TEXT,
  verdict         TEXT,                          -- the owner's: accept | reject
  verdict_at      REAL,
  verdict_minutes REAL,
  verdict_note    TEXT,
  shipped_at      REAL,
  usd             REAL,
  pruned_at       REAL
);
CREATE INDEX IF NOT EXISTS plan_items_state ON plan_items(state);
CREATE INDEX IF NOT EXISTS plan_items_template ON plan_items(template_id);

CREATE TABLE IF NOT EXISTS plan_rejections (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  plan_job_id INTEGER,
  at          REAL NOT NULL,
  template_id TEXT,
  reason      TEXT NOT NULL,                     -- a planner.REASONS code
  detail      TEXT
);
CREATE INDEX IF NOT EXISTS plan_rejections_at ON plan_rejections(at);

CREATE TABLE IF NOT EXISTS template_state (
  template_id TEXT PRIMARY KEY,
  paused      INTEGER NOT NULL DEFAULT 0,
  reason      TEXT,                              -- owner | kill_low_acceptance | kill_unused_30d | kill_low_value
  detail      TEXT,
  at          REAL,
  since       REAL                               -- kill criteria only count history after this
);
"""

# Columns added after the first schema; _migrate() adds them to an older database file.
_MIGRATIONS = {
    "jobs": [("root_id", "INTEGER"), ("content_key", "TEXT"),
             ("interruptions", "INTEGER NOT NULL DEFAULT 0"), ("respawns", "INTEGER NOT NULL DEFAULT 0"),
             ("faults", "INTEGER NOT NULL DEFAULT 0"), ("page_failures", "INTEGER NOT NULL DEFAULT 0"),
             ("page_retry_at", "REAL")],
    "leases": [("progress_timeout_s", "REAL"), ("fence", "INTEGER NOT NULL DEFAULT 0")],
    "events": [("wall", "REAL")],
    "supervisor_state": [("safe_engine", "TEXT"), ("brainless_epoch", "INTEGER NOT NULL DEFAULT 0"),
                         ("engine_health", "TEXT NOT NULL DEFAULT '{}'"),
                         ("orphans", "TEXT NOT NULL DEFAULT '[]'"),
                         ("t_now", "REAL"), ("t_wall", "REAL"), ("t_mono", "REAL")],
    "planner_state": [("engine_fail_streak", "INTEGER NOT NULL DEFAULT 0")],
    "plan_items": [("reserved_slot", "INTEGER NOT NULL DEFAULT 0")],
}


def _enc(value):
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:  # mixed-type keys from an untrusted decision: keep the data, drop the sort
        return json.dumps(value, default=str)


class Job:
    """Snapshot of one jobs row with JSON columns decoded."""

    def __init__(self, row: sqlite3.Row):
        for key in row.keys():
            value = row[key]
            if key in _JOB_JSON and value is not None:
                value = json.loads(value)
            setattr(self, key, value)

    @property
    def brainless(self) -> bool:
        return self.requires not in ("brain", "human")

    def __repr__(self) -> str:
        return f"Job({self.id} {self.kind} {self.requires} {self.state})"


class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()
        self._depth = 0
        self._sp = 0
        self.wall_clock = None  # callable; set by the supervisor so events carry wall time too
        self.closed = False

    def _migrate(self) -> None:
        tables = {r["name"] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, cols in _MIGRATIONS.items():
            if table not in tables:
                continue
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols:
                if name not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self.conn.executescript(SCHEMA)
        self.conn.execute("CREATE INDEX IF NOT EXISTS jobs_lineage ON jobs(root_id, content_key)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS jobs_task ON jobs(task_key)")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.conn.close()

    @property
    def in_tx(self) -> bool:
        return self._depth > 0

    @contextmanager
    def tx(self):
        """BEGIN IMMEDIATE ... COMMIT; an exception (including a simulated crash) rolls back.
        Nested use joins the outer transaction."""
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self._depth = 0
            self.conn.execute("ROLLBACK")
            raise
        self._depth = 0
        self.conn.execute("COMMIT")

    @contextmanager
    def savepoint(self):
        """A nested unit of work: an exception rolls back only what happened inside it, and the
        enclosing transaction carries on. Outside a transaction it behaves like tx()."""
        if not self._depth:
            with self.tx():
                yield
            return
        self._sp += 1
        name = f"sp{self._sp}"
        self.conn.execute(f"SAVEPOINT {name}")
        self._depth += 1
        try:
            yield
        except BaseException:
            self.conn.execute(f"ROLLBACK TO {name}")
            self.conn.execute(f"RELEASE {name}")
            raise
        else:
            self.conn.execute(f"RELEASE {name}")
        finally:
            self._depth -= 1

    # ---- jobs ----
    def insert_job(self, **cols) -> tuple[int, bool]:
        """Insert unless idempotency_key exists. Returns (id, created)."""
        for key in _JOB_JSON:
            if key in cols:
                cols[key] = _enc(cols[key])
        names, marks = ", ".join(cols), ", ".join("?" * len(cols))
        cur = self.conn.execute(
            f"INSERT INTO jobs ({names}) VALUES ({marks}) ON CONFLICT(idempotency_key) DO NOTHING",
            tuple(cols.values()))
        if cur.rowcount == 1:
            return cur.lastrowid, True
        row = self.conn.execute("SELECT id FROM jobs WHERE idempotency_key = ?",
                                (cols["idempotency_key"],)).fetchone()
        return row["id"], False

    def update_job(self, job_id: int, expect_state: str | tuple | None = None, **cols) -> bool:
        """Update a job row. With expect_state this is a compare-and-set: it only applies if the
        row is still in that state (or one of those states). Returns whether a row changed."""
        for key in _JOB_JSON:
            if key in cols:
                cols[key] = _enc(cols[key])
        sets = ", ".join(f"{k} = ?" for k in cols)
        sql, args = f"UPDATE jobs SET {sets} WHERE id = ?", [*cols.values(), job_id]
        if expect_state is not None:
            states = (expect_state,) if isinstance(expect_state, str) else tuple(expect_state)
            sql += f" AND state IN ({', '.join('?' * len(states))})"
            args += list(states)
        return self.conn.execute(sql, args).rowcount == 1

    def job(self, job_id: int) -> Job | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job(row) if row else None

    def jobs(self, *states: str) -> list[Job]:
        if states:
            marks = ", ".join("?" * len(states))
            rows = self.conn.execute(f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY id", states)
        else:
            rows = self.conn.execute("SELECT * FROM jobs ORDER BY id")
        return [Job(r) for r in rows]

    def children(self, parent_id: int) -> list[Job]:
        rows = self.conn.execute("SELECT * FROM jobs WHERE parent_id = ? ORDER BY id", (parent_id,))
        return [Job(r) for r in rows]

    def queued_jobs(self) -> list[Job]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE state = 'queued' "
            "ORDER BY priority DESC, COALESCE(deadline, 1e300), id")
        return [Job(r) for r in rows]

    def next_queued(self) -> Job | None:
        jobs = self.queued_jobs()
        return jobs[0] if jobs else None

    def latest_in_lineage(self, root_id: int, content_key: str) -> Job | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE root_id = ? AND content_key = ? ORDER BY id DESC LIMIT 1",
            (root_id, content_key)).fetchone()
        return Job(row) if row else None

    def max_attempt(self, task_key: str) -> int:
        row = self.conn.execute("SELECT MAX(attempt) AS m FROM jobs WHERE task_key = ?", (task_key,)).fetchone()
        return int(row["m"] or 0)

    def lineage_failures(self, root_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE root_id = ? AND requires NOT IN ('brain', 'human') "
            "AND state IN ('failed', 'quarantined')", (root_id,)).fetchone()
        return int(row["n"])

    def unpaged_humans(self) -> list[Job]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE state = 'awaiting_human' AND paged_at IS NULL ORDER BY id")
        return [Job(r) for r in rows]

    # ---- continuations ----
    def add_continuation(self, job_id: int, on_success: dict, on_failure: dict) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO continuations (job_id, on_success, on_failure) VALUES (?, ?, ?)",
            (job_id, _enc(on_success), _enc(on_failure)))

    @staticmethod
    def _cont(row) -> dict:
        d = dict(row)
        d["on_success"], d["on_failure"] = json.loads(d["on_success"]), json.loads(d["on_failure"])
        return d

    def continuation(self, job_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM continuations WHERE job_id = ?", (job_id,)).fetchone()
        return self._cont(row) if row else None

    def unfired_continuations(self) -> list[tuple[Job, dict]]:
        marks = ", ".join("?" * len(TERMINAL))
        rows = self.conn.execute(
            f"SELECT c.job_id FROM continuations c JOIN jobs j ON j.id = c.job_id "
            f"WHERE c.fired_job_id IS NULL AND j.state IN ({marks}) ORDER BY c.job_id", TERMINAL)
        return [(self.job(r["job_id"]), self.continuation(r["job_id"])) for r in rows.fetchall()]

    def mark_fired(self, job_id: int, outcome: str, fired_job_id: int, at: float) -> None:
        self.conn.execute(
            "UPDATE continuations SET fired_outcome = ?, fired_job_id = ?, fired_at = ? "
            "WHERE job_id = ? AND fired_job_id IS NULL", (outcome, fired_job_id, at, job_id))

    # ---- leases ----
    def put_lease(self, job_id: int, **cols) -> None:
        cols["job_id"] = job_id
        names, marks = ", ".join(cols), ", ".join("?" * len(cols))
        self.conn.execute(f"INSERT OR REPLACE INTO leases ({names}) VALUES ({marks})", tuple(cols.values()))

    def update_lease(self, job_id: int, owner: int | None = None, **cols) -> bool:
        """With owner, only an active lease fenced by that leader epoch is updated (renewals): a
        stale gpud cannot renew a lease a newer leader has taken over."""
        sets = ", ".join(f"{k} = ?" for k in cols)
        sql, args = f"UPDATE leases SET {sets} WHERE job_id = ?", [*cols.values(), job_id]
        if owner is not None:
            sql, args = sql + " AND state = 'active' AND fence = ?", args + [owner]
        return self.conn.execute(sql, args).rowcount == 1

    def lease(self, job_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM leases WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    # ---- events (append-only audit log) ----
    def event(self, at: float, kind: str, job_id: int | None = None, /, **data) -> None:
        wall = None
        if self.wall_clock is not None:
            try:
                wall = float(self.wall_clock())
            except Exception:
                wall = None
        self.conn.execute("INSERT INTO events (at, wall, kind, job_id, data) VALUES (?, ?, ?, ?, ?)",
                          (at, wall, kind, job_id, json.dumps(data, sort_keys=True, default=str)))

    def events(self, kind: str | None = None, job_id: int | None = None) -> list[dict]:
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if kind is not None:
            sql, args = sql + " AND kind = ?", args + [kind]
        if job_id is not None:
            sql, args = sql + " AND job_id = ?", args + [job_id]
        rows = self.conn.execute(sql + " ORDER BY id", args)
        return [dict(r, data=json.loads(r["data"])) for r in rows]

    def events_after(self, after_id: int, limit: int = 1000) -> list[dict]:
        """Events with id > after_id, oldest first (log shipping and `events --tail`)."""
        rows = self.conn.execute("SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?", (after_id, limit))
        return [dict(r, data=json.loads(r["data"])) for r in rows]

    def last_event_id(self) -> int:
        row = self.conn.execute("SELECT MAX(id) AS m FROM events").fetchone()
        return int(row["m"] or 0)

    # ---- singleton supervisor state ----
    def state(self) -> dict:
        d = dict(self.conn.execute("SELECT * FROM supervisor_state WHERE id = 1").fetchone())
        for key in _STATE_JSON:
            if d[key] is not None:
                d[key] = json.loads(d[key])
        if isinstance(d["known_resident"], list):  # pre-migration format: a bare list of models
            d["known_resident"] = {m: None for m in d["known_resident"]}
        return d

    def set_state(self, **cols) -> None:
        for key in _STATE_JSON:
            if key in cols:
                cols[key] = _enc(cols[key])
        sets = ", ".join(f"{k} = ?" for k in cols)
        self.conn.execute(f"UPDATE supervisor_state SET {sets} WHERE id = 1", tuple(cols.values()))
