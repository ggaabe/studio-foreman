"""Entry point of the h3 media worker: `<media_python> -I <repo>/foreman/h3/worker.py <command> <request.json>`.

The core never imports this package (it needs numpy and Pillow, which the core's Python may not have);
it runs this file as a subprocess under a configured interpreter, with the ad's work directory as the
current directory. Request paths are relative to that directory unless absolute.

Protocol: progress lines `{"progress": 0..1, "note": "..."}` on stdout, then the LAST line is one JSON
object `{"ok": true|false, "error": str|null, "class": "content"|"infra"|null, ...results}`. Exit codes:
0 ok; 2 content failure (the takes or the plan cannot make a good ad); 3 infrastructure (ffmpeg, a font,
an input file, numpy or Pillow missing, music shorter than the ad); anything else is a crash, which the
core retries. The worker leads its own process group and, on SIGTERM, terminates the group (its ffmpeg
children) before it exits, so a cancelled job leaves nothing running. Being in a group of its own, it is out of
the reach of whoever stops its parent's group (launchd ending gpud): it watches its parent instead, and when the
parent is gone (gpud died; the step will be retried on the same files) it stops the same way.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback

COMMANDS = ("probe", "edit")
EXIT = {"content": 2, "infra": 3}
PARENT_POLL_S = 0.5         # how often the worker looks for its parent
GRACE_S = 5.0               # then how long the SIGTERM path may take before the group is killed outright


class Terminated(BaseException):
    """SIGTERM arrived: unwind (temp files are removed on the way out) and exit."""


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def _progress(p, note="") -> None:
    _emit({"progress": round(max(0.0, min(1.0, float(p))), 3), "note": str(note)[:200]})


def _lead_group() -> bool:
    """Make this process the leader of its own process group (already true when started with a new session)."""
    try:
        if os.getpgrp() != os.getpid():
            os.setpgid(0, 0)
    except OSError:
        pass
    return os.getpgrp() == os.getpid()


def _on_term(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if os.getpgrp() == os.getpid():          # never signal a group this process does not lead
        try:
            os.killpg(os.getpid(), signal.SIGTERM)
        except OSError:
            pass
    raise Terminated()


def _watch_parent(parent: int, poll_s: float = PARENT_POLL_S) -> threading.Thread:
    """A daemon thread that stops this worker once its parent is gone (the parent id changes when the parent dies:
    the worker is re-parented). It takes the SIGTERM path (the group's ffmpeg stopped, temp files removed); if that
    has not ended the process within GRACE_S, the whole group is killed."""
    def watch():
        while os.getppid() == parent:
            time.sleep(poll_s)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except OSError:
            pass
        time.sleep(GRACE_S)
        try:
            if os.getpgrp() == os.getpid():
                os.killpg(os.getpid(), signal.SIGKILL)
        except OSError:
            pass
        os._exit(128 + signal.SIGTERM)

    t = threading.Thread(target=watch, name="parent-watch", daemon=True)
    t.start()
    return t


def _fail(error, cls, code):
    _emit({"ok": False, "error": error, "class": cls})
    return code


def _leave_background() -> None:
    """launchd runs gpud as a Background job, and macOS gives that band (throttled CPU and I/O) to every
    child: the edit ran about five times slower in it than from a shell. Leave it before any ffmpeg
    starts, so the encodes inherit normal priority."""
    if sys.platform == "darwin":
        try:
            os.setpriority(4, 0, 0)  # PRIO_DARWIN_PROCESS, this process: not background
        except (AttributeError, OSError):
            pass


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parent = os.getppid()
    _leave_background()
    _lead_group()
    signal.signal(signal.SIGTERM, _on_term)
    if parent > 1:
        _watch_parent(parent)
    if len(argv) != 2 or argv[0] not in COMMANDS:
        return _fail(f"usage: worker.py {{{'|'.join(COMMANDS)}}} request.json", None, 64)
    command, req_path = argv
    try:
        try:
            with open(req_path, encoding="utf-8") as f:
                req = json.load(f)
        except OSError as e:
            return _fail(f"cannot read the request: {e}", "infra", 3)
        except ValueError as e:
            return _fail(f"bad request: not JSON ({e})", "content", 2)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from foreman.h3 import ff
            mod = __import__(f"foreman.h3.{command}", fromlist=["run"])
        except ImportError as e:             # numpy or Pillow missing under this interpreter
            return _fail(f"the media python cannot load the worker: {e}", "infra", 3)
        try:
            result = mod.run(req, _progress)
        except ff.WorkerError as e:
            return _fail(str(e), e.kind, EXIT[e.kind])
        out = {"ok": True, "error": None, "class": None}
        out.update(result or {})
        _emit(out)
        return 0
    except Terminated:
        return _fail("terminated", "infra", 128 + signal.SIGTERM)
    except Exception as e:                   # a crash: the core retries the step
        traceback.print_exc(file=sys.stderr)
        return _fail(f"{type(e).__name__}: {e}"[:2000], None, 1)


root = _repo_root()

if __name__ == "__main__":
    sys.path.insert(0, root)
    sys.exit(main())
