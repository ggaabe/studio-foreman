"""gpud: the always-on Foreman supervisor process for a Mac running mlx-serve.

    python3 -m foreman.daemon                  # run in the foreground (launchd runs exactly this)
    python3 -m foreman.daemon --once           # one recover() + tick(), then exit (smoke test)

It takes the single-instance lock on <db>.lock, runs recover() against what mlx-serve reports,
then calls tick() about once a second. Logs are JSON lines in <home>/logs/gpud.log (rotated),
and every supervisor event is mirrored there as it is committed. Pages (human_review,
safe_mode, engine_lost) are logged at WARNING and appended to <home>/pages.jsonl.

SIGTERM / SIGINT end the loop after the current tick, close the HTTP connections this process
holds (mlx-serve aborts those runs at their next step boundary; the next gpud records them as
interrupted and fires their on_failure batons), release the lock and exit 0. A watchdog thread
exits the process (status 70) if one tick runs longer than watchdog_s; gpud holds no GPU
memory, so launchd restarting it strands nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time

from . import studio
from .brain_client import BrainClient
from .clock import SystemClock
from .supervisor import Supervisor

log = logging.getLogger("foreman")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}",
               "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            out.update(extra)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(log_dir: str, level: str = "INFO", stderr: bool = True) -> None:
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(os.path.join(log_dir, "gpud.log"), maxBytes=20 << 20, backupCount=5)
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)
    if stderr:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(JsonFormatter())
        sh.setLevel(logging.WARNING)
        root.addHandler(sh)


class Pager:
    """At-least-once delivery target for the supervisor's page outbox: a WARNING log line and
    one JSON line per page in <home>/pages.jsonl (tail it, or hook a notifier to it)."""

    def __init__(self, path: str):
        self.path = path

    def __call__(self, page: dict) -> None:
        log.warning("PAGE HUMAN: %s", page.get("reason"), extra={"fields": {"page": page}})
        with open(self.path, "a") as f:
            f.write(json.dumps(dict(page, wall=time.time()), default=str) + "\n")


class Gpud:
    def __init__(self, cfg: studio.Config, engine=None, clock=None, planner=None):
        self.cfg = cfg
        os.makedirs(cfg.home, exist_ok=True)
        os.makedirs(cfg.artifacts_dir, exist_ok=True)
        catalog = studio.catalog(cfg.video_exclusive)
        if engine is None:
            brain = BrainClient(cfg.base_url, catalog=catalog)
            engine = studio.build_engine(cfg, brain=brain)
        self.engine = engine
        if planner is None and cfg.planner:
            planner = studio.build_planner(cfg)
        self.sup = Supervisor(cfg.db_path, {studio.ENGINE: engine}, clock or SystemClock(), cfg.policy(),
                              catalog=catalog, pager=Pager(os.path.join(cfg.home, "pages.jsonl")),
                              planner=planner)
        self.stop = threading.Event()
        self._tick_started: float | None = None
        self._last_event = self.sup.store.last_event_id()
        self.ticks = 0

    def ship_events(self) -> None:
        for e in self.sup.store.events_after(self._last_event, 500):
            self._last_event = e["id"]
            level = logging.WARNING if e["kind"] in (
                "page_human", "safe_mode_entered", "engine_declared_lost", "lease_expired", "human_review",
                "supervisor_error", "job_fault", "insufficient_memory", "insufficient_slots", "planner_suspended",
                "plan_invalid", "plan_error", "template_paused", "template_error", "unload_refused",
                "engine_instance_changed", "model_size_observed") else logging.INFO
            log.log(level, "event %s", e["kind"], extra={"fields": {
                "event": e["kind"], "event_id": e["id"], "job_id": e["job_id"], "at": e["at"], "data": e["data"]}})

    def watchdog(self) -> None:
        limit = self.cfg.watchdog_s
        while not self.stop.wait(5.0):
            started = self._tick_started
            if started is not None and time.monotonic() - started > limit:
                log.critical("tick has run for %.0fs (> watchdog %.0fs); exiting so launchd restarts gpud",
                             time.monotonic() - started, limit)
                logging.shutdown()
                os._exit(70)

    def run(self, once: bool = False) -> int:
        st = self.sup.store.state()
        log.info("gpud starting", extra={"fields": {
            "pid": os.getpid(), "db": self.cfg.db_path, "mlx": self.cfg.base_url, "leader": self.sup.leader,
            "video_exclusive": self.cfg.video_exclusive, "boot_count": st["boot_count"],
            "fault_stall_after": self.cfg.stall_after, "fault_kinds": self.cfg.stall_kinds,
            "progress_timeout_s": self.sup.p.progress_timeout_s, "planner": self.sup.planner is not None,
            "keep_raw_frames": self.cfg.keep_raw_frames}})
        if not self.sup.leader:
            log.warning("another gpud holds %s.lock: standing by until it exits", self.cfg.db_path)
        if self.cfg.watchdog_s and not once:
            threading.Thread(target=self.watchdog, name="gpud-watchdog", daemon=True).start()
        try:
            self._tick_started = time.monotonic()
            self.sup.recover()
            self._tick_started = None
            self.ship_events()
            while not self.stop.is_set():
                t0 = time.monotonic()
                self._tick_started = t0
                try:
                    self.sup.tick()
                except Exception:  # tick() has its own boundaries; this is the last line of defense
                    log.exception("tick raised")
                finally:
                    self._tick_started = None
                self.ticks += 1
                self.ship_events()
                if once:
                    break
                self.stop.wait(max(0.0, self.cfg.tick_s - (time.monotonic() - t0)))
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        live = getattr(self.engine, "live_runs", lambda: [])()
        if live:
            log.warning("shutting down with live runs %s: their connections close now and mlx-serve aborts them; "
                        "the next gpud records them as interrupted", live)
        try:
            close = getattr(self.engine, "close", None)
            if close:
                close()
        finally:
            try:
                self.ship_events()
            except Exception:
                pass
            self.sup.close()
            log.info("gpud stopped after %d ticks", self.ticks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m foreman.daemon", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config file (default: $FOREMAN_HOME/config.json)")
    ap.add_argument("--db", dest="db_path")
    ap.add_argument("--base-url", help="mlx-serve URL (default http://127.0.0.1:11234)")
    ap.add_argument("--video-exclusive", dest="video_exclusive", action="store_true", default=None,
                    help="unload the brain while video renders (the default for this deployment)")
    ap.add_argument("--no-video-exclusive", dest="video_exclusive", action="store_false",
                    help="keep a brain co-resident with LTX when it fits")
    ap.add_argument("--fault-stall-after", dest="stall_after", type=int,
                    help="fault injection: media runs stop reading after N progress events")
    ap.add_argument("--log-level")
    ap.add_argument("--once", action="store_true", help="recover, tick once, exit")
    args = ap.parse_args(argv)
    cfg = studio.load_config(args.config, db_path=args.db_path, base_url=args.base_url,
                             video_exclusive=args.video_exclusive, stall_after=args.stall_after,
                             log_level=args.log_level)
    setup_logging(cfg.log_dir, cfg.log_level)
    gpud = Gpud(cfg)

    def _stop(signum, _frame):
        log.info("signal %d: stopping after this tick", signum)
        gpud.stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return gpud.run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
