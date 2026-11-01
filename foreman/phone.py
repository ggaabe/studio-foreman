"""Phone approvals over Telegram: the owner reviews drafts and answers pages from a phone.

    python3 -m foreman.phone              # the service (its own LaunchAgent; launchd runs exactly this)
    python3 -m foreman.phone --once       # one pass: send what is due, read updates once, exit

    python3 -m foreman.cli phone setup    # bot token (hidden prompt, or stdin), checked with getMe, then pairing
    python3 -m foreman.cli phone status | test | unpair [--keep-token] | resend

A separate process from gpud, so a Telegram outage (or a bug here) can never affect scheduling.
It opens the same SQLite file as a non-leader client (WAL, busy timeout) and records verdicts,
retracts, shipments, notes and page resolutions only through the CLI's functions
(`cli.apply_*`): the ledger's code paths, with the same validation, the same events (carrying
by="phone") and the same kill criteria. It never touches gpud, its lock or an engine, so a gpud
restart is just another writer on the file.

What it sends to the paired chat, each exactly once (the phone_messages table, UNIQUE(item, kind)):

  * every item entering the review queue: the MP4 (sendVideo, up to video_max_mb), else a sampled
    frame as a document; speech and music as an M4A (ffmpeg) for sendAudio, else the WAV as a
    document; nothing above the Bot API's 50 MB upload limit; text only when the files are gone.
    The caption is plain text (no parse_mode): template, job, the brief, the brain's note, review
    minutes. Buttons: Approve / Reject; then Used it / Undo, or Undo.
  * every page (human_review, engine_lost, SAFE mode, planner suspended, ...) with a Resolve button;
  * optionally (off by default) a daily digest.

Security model
  * The bot token, the pairing and the HMAC key live in $FOREMAN_HOME/secrets/telegram.json
    (directory 0700, file 0600), never in config.json or the database. Every log line, exception
    message and stored error passes through `redact`.
  * Pairing binds exactly one private chat and one user: a one-time code (15 minutes, at most 5
    wrong tries, burned if it shows up in a group) arriving as /start <code> from the deep link.
  * Updates from any other chat or user are ignored without a reply and logged without content.
  * Buttons carry compact HMAC-signed data bound to the chat, the draft or page, the action and
    the message's issue counter: a forged, garbled, replayed or stale button records nothing, and
    the ledger's compare-and-set states are the final guard, so neither a double tap nor a race
    with the CLI can record a second verdict.
  * Only regular files strictly inside the artifacts directory are uploaded (never a symlink).
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import fcntl
import getpass
import hashlib
import hmac
import json
import logging
import logging.handlers
import os
import re
import secrets
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass

from . import ledger, planner as planner_mod, retention, studio
from .daemon import JsonFormatter
from .phone_config import API_BASE, TOKEN_RE, PhoneConfig
from .phone_config import TOKEN_SCAN as _TOKEN_SCAN
from .policy import number
from .store import Store
from .supervisor import InstanceLock
from .templates import load_templates, strip_urls
from .text import CONTROL_RE, strip_invisible

LOG = logging.getLogger("foreman.phone")

SECRETS_FILE = "telegram.json"
_BOT_PATH = re.compile(r"/bot\d[^/\s'\"]*")
USERNAME_RE = re.compile(r"[A-Za-z0-9_]{3,64}")
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I: easy to read off a terminal
CODE_LEN = 10
PAIR_RE = re.compile(r"^/(start|pair)(?:@([A-Za-z0-9_]+))?\s+([A-Za-z0-9_-]{1,64})$")
MB = 1e6
MAX_RESPONSE = 8 << 20          # bytes read from one Bot API answer, at most
CAPTION_MAX = 1024              # Telegram's caption limit (UTF-16 code units)
BODY_MAX = 880                  # the caption without its status line
ANSWER_MAX = 190                # answerCallbackQuery text (Telegram: 200)
TERMINAL_DRAFT = ("shipped", "expired", "failed", "gone", "rejected-pruned")


# ---------------------------------------------------------------- redaction

_SECRET_VALUES: list = []       # every secret this process has seen (bounded), longest first


def register_secret(value) -> None:
    """Remember a secret (the token, the signing key, a pairing code) so `redact` removes it from
    any text, also in its URL-encoded form."""
    if not isinstance(value, str) or len(value) < 8:
        return
    for v in (value, urllib.parse.quote(value, safe="")):
        if v not in _SECRET_VALUES:
            _SECRET_VALUES.append(v)
    del _SECRET_VALUES[:-24]
    _SECRET_VALUES.sort(key=len, reverse=True)


def redact(text) -> str:
    """`text` with every known secret and anything shaped like a bot token removed."""
    s = str(text)
    for v in list(_SECRET_VALUES):
        if v in s:
            s = s.replace(v, "<redacted>")
    s = _TOKEN_SCAN.sub("<redacted-token>", s)
    return _BOT_PATH.sub("/bot<redacted>", s)


class RedactingJsonFormatter(JsonFormatter):
    """gpud's JSON-lines format, with `redact` applied to the whole finished line (message,
    fields and traceback): the token cannot reach a log file, whoever logs it."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            line = super().format(record)
        except Exception as e:  # a bad format string must not lose the line
            line = json.dumps({"level": record.levelname, "msg": f"unformattable log record: {type(e).__name__}"})
        return redact(line)


def setup_logging(log_dir: str, level: str = "INFO", stderr: bool = True) -> None:
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(os.path.join(log_dir, "phone.log"), maxBytes=10 << 20, backupCount=3)
    fh.setFormatter(RedactingJsonFormatter())
    root.addHandler(fh)
    if stderr:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(RedactingJsonFormatter())
        sh.setLevel(logging.WARNING)
        root.addHandler(sh)


def _install_excepthooks() -> None:
    """An exception nobody caught is printed redacted (launchd keeps stderr in a file)."""
    import traceback

    def hook(exc_type, exc, tb):
        sys.stderr.write(redact("".join(traceback.format_exception(exc_type, exc, tb))))

    sys.excepthook = hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = lambda args: hook(args.exc_type, args.exc_value, args.exc_traceback)


# ---------------------------------------------------------------- errors

class PhoneError(Exception):
    """Something the owner can fix; the CLI prints it."""

    def __init__(self, message: str):
        super().__init__(redact(message))


class TelegramError(Exception):
    """A Bot API call failed. `code` is Telegram's error_code (or the HTTP status); None for a
    network failure (TelegramNetworkError). The message is always redacted."""

    def __init__(self, method: str, code, description, retry_after=None):
        self.method = str(method)
        self.code = code if isinstance(code, int) else None
        self.description = redact(description)[:300]
        ra = number(retry_after)
        self.retry_after = float(ra) if ra is not None and ra >= 0 else None
        super().__init__(f"{self.method}: {self.code if self.code is not None else 'network'}: {self.description}")

    @property
    def retriable(self) -> bool:
        return self.code is None or self.code in (409, 429) or self.code >= 500

    @property
    def auth(self) -> bool:
        return self.code in (401, 404)

    @property
    def not_modified(self) -> bool:
        return self.code == 400 and "not modified" in self.description.lower()


class TelegramNetworkError(TelegramError):
    def __init__(self, method: str, description: str):
        super().__init__(method, None, description)


# ---------------------------------------------------------------- the Bot API over urllib

def _form_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))[:100] or "file"


def multipart(fields: dict, files: list, progress=None) -> tuple:
    """(content type, length, chunks): a multipart/form-data body written by hand (the standard
    library has no encoder). `files` is [(field, filename, content type, path)]; the file is read
    in 1 MB pieces while it is sent, so an upload is never held in memory whole, and exactly the
    size measured here is sent (a file that shrinks meanwhile fails the request)."""
    boundary = "foreman-" + secrets.token_hex(16)
    parts = []
    for name, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="{_form_name(name)}"\r\n\r\n'.encode()
        parts.append((head + str(value).encode("utf-8", "replace") + b"\r\n", None))
    for field, filename, ctype, path in files:
        head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{_form_name(field)}"; '
                f'filename="{_form_name(filename)}"\r\nContent-Type: {ctype}\r\n\r\n').encode()
        parts.append((head, (path, os.path.getsize(path))))
    tail = f"--{boundary}--\r\n".encode()
    length = sum(len(h) + (f[1] + 2 if f else 0) for h, f in parts) + len(tail)

    def chunks():
        for head, f in parts:
            yield head
            if f is None:
                continue
            path, size = f
            left = size
            with open(path, "rb") as fh:
                while left > 0:
                    b = fh.read(min(1 << 20, left))
                    if not b:
                        raise OSError("a file got shorter while it was being sent")
                    left -= len(b)
                    if progress is not None:
                        progress()
                    yield b
            yield b"\r\n"
        yield tail

    return f"multipart/form-data; boundary={boundary}", length, chunks()


class BotApi:
    """The few Bot API methods Foreman uses, over urllib with a timeout on every call. Every
    failure is a TelegramError whose text has the token removed; the original exception (whose
    text or URL may carry it) is never chained."""

    def __init__(self, token: str, base: str = API_BASE, timeout: float = 20.0, upload_timeout: float = 180.0):
        register_secret(token)
        self._token = token
        self.base = str(base).rstrip("/")
        host = (urllib.parse.urlsplit(self.base).hostname or "").lower()
        handlers = [urllib.request.ProxyHandler({})] if host in ("127.0.0.1", "localhost", "::1") else []
        self._opener = urllib.request.build_opener(*handlers)
        self.timeout, self.upload_timeout = float(timeout), float(upload_timeout)
        self.progress = None  # called for every MB an upload sends (the phone's watchdog heartbeat)

    def __repr__(self) -> str:
        return f"BotApi({self.base!r})"

    def call(self, method: str, params: dict | None = None, files: list | None = None, timeout: float | None = None):
        url = f"{self.base}/bot{self._token}/{method}"
        err = None
        status, raw = None, b""
        try:
            if files:
                ctype, length, body = multipart(params or {}, files, self.progress)
                headers = {"Content-Type": ctype, "Content-Length": str(length)}
            else:
                body = json.dumps(params or {}, ensure_ascii=False).encode("utf-8", "replace")
                headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                status, raw = resp.status, resp.read(MAX_RESPONSE)
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                raw = e.read(1 << 20)
            except Exception:
                raw = b""
        except (OSError, ValueError, ArithmeticError, LookupError) as e:  # URLError, timeouts, TLS, ...
            err = TelegramNetworkError(method, f"{type(e).__name__}: {getattr(e, 'reason', None) or e}")
        except Exception as e:  # http.client.HTTPException and anything else the transport raises
            err = TelegramNetworkError(method, f"{type(e).__name__}: {e}")
        if err is not None:  # raised outside the except clause: nothing chains the original exception
            raise err
        try:
            data = json.loads(raw.decode("utf-8", "replace")) if raw else None
        except ValueError:
            data = None
        if not isinstance(data, dict):
            if status is not None and status >= 400:
                raise TelegramError(method, status, f"HTTP {status} without a Bot API answer")
            raise TelegramNetworkError(method, "the answer was not a Bot API JSON object")
        if not data.get("ok"):
            params_ = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
            raise TelegramError(method, data.get("error_code") or status, str(data.get("description") or "error"),
                                params_.get("retry_after"))
        return data.get("result")

    # ---- methods
    def get_me(self) -> dict:
        res = self.call("getMe")
        if not isinstance(res, dict):
            raise TelegramNetworkError("getMe", "getMe answered without a bot")
        return res

    def get_updates(self, offset: int, timeout: float, limit: int = 50) -> list:
        res = self.call("getUpdates", {"offset": int(offset), "timeout": int(max(0, timeout)), "limit": int(limit),
                                       "allowed_updates": ["message", "callback_query"]},
                        timeout=max(0.0, timeout) + self.timeout)
        return res if isinstance(res, list) else []

    def send_message(self, chat_id, text: str, markup=None, reply_to=None) -> dict:
        p = {"chat_id": chat_id, "text": text, "link_preview_options": {"is_disabled": True}}
        if markup is not None:
            p["reply_markup"] = markup
        if reply_to:
            p["reply_parameters"] = {"message_id": int(reply_to), "allow_sending_without_reply": True}
        res = self.call("sendMessage", p)
        return res if isinstance(res, dict) else {}

    def send_file(self, method: str, field: str, chat_id, path: str, caption: str, markup=None, extra=None,
                  ctype: str = "application/octet-stream") -> dict:
        fields = {"chat_id": chat_id, "caption": caption, "reply_markup": markup}
        fields.update(extra or {})
        res = self.call(method, fields, files=[(field, os.path.basename(path), ctype, path)],
                        timeout=self.upload_timeout)
        return res if isinstance(res, dict) else {}

    def answer_callback(self, callback_id, text: str | None = None, alert: bool = False) -> None:
        p = {"callback_query_id": str(callback_id)}
        if text:
            p["text"] = text[:ANSWER_MAX]
        if alert:
            p["show_alert"] = True
        self.call("answerCallbackQuery", p)

    def edit(self, chat_id, message_id, media: str, text: str, markup) -> None:
        if media == "text":
            self.call("editMessageText", {"chat_id": chat_id, "message_id": int(message_id), "text": text,
                                          "reply_markup": markup, "link_preview_options": {"is_disabled": True}})
        else:
            self.call("editMessageCaption", {"chat_id": chat_id, "message_id": int(message_id), "caption": text,
                                             "reply_markup": markup})

    def edit_markup(self, chat_id, message_id, markup) -> None:
        self.call("editMessageReplyMarkup", {"chat_id": chat_id, "message_id": int(message_id), "reply_markup": markup})


# ---------------------------------------------------------------- secrets

def new_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))


def new_key() -> str:
    return secrets.token_hex(32)


def code_mac(key: str, code: str) -> str:
    return hmac.new(bytes.fromhex(key), b"pair:" + code.strip().upper().encode(), hashlib.sha256).hexdigest()


class SecretStore:
    """$FOREMAN_HOME/secrets/telegram.json: directory 0700, file 0600, written atomically under an
    flock. Keys: token, bot_id, bot_username, hmac_key, pairing {mac, expires_wall, failures},
    chat_id, user_id, paired_wall, burned (why the last code stopped being valid)."""

    def __init__(self, home: str):
        self.dir = os.path.join(home, "secrets")
        self.path = os.path.join(self.dir, SECRETS_FILE)
        self.lock_path = os.path.join(self.dir, "telegram.lock")

    def _ensure_dir(self) -> None:
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        st = os.lstat(self.dir)
        if not os.path.isdir(self.dir) or os.path.islink(self.dir):
            raise PhoneError(f"{self.dir} is not a directory")
        if st.st_uid != os.getuid():
            raise PhoneError(f"{self.dir} belongs to another user")
        if st.st_mode & 0o777 != 0o700:
            os.chmod(self.dir, 0o700)

    def signature(self):
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    def exists(self) -> bool:
        return os.path.lexists(self.path)

    def load(self) -> dict | None:
        try:
            st = os.lstat(self.path)
        except FileNotFoundError:
            return None
        if os.path.islink(self.path) or not os.path.isfile(self.path):
            raise PhoneError(f"{self.path} is not a regular file; refusing to read it")
        if st.st_uid != os.getuid():
            raise PhoneError(f"{self.path} belongs to another user; refusing to read it")
        if st.st_mode & 0o077:
            os.chmod(self.path, 0o600)
            LOG.warning("the secrets file was readable by other users; it is 0600 now")
        try:
            with open(self.path) as f:
                data = json.load(f)
        except ValueError:
            raise PhoneError(f"{self.path} is not valid JSON; run `phone setup` again") from None
        if not isinstance(data, dict):
            raise PhoneError(f"{self.path} is not a JSON object; run `phone setup` again")
        register_secret(data.get("token"))
        register_secret(data.get("hmac_key"))
        return data

    @contextmanager
    def locked(self):
        self._ensure_dir()
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _write(self, data: dict) -> None:
        self._ensure_dir()
        tmp = os.path.join(self.dir, f".{SECRETS_FILE}.{os.getpid()}.{secrets.token_hex(4)}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                fd = None
                json.dump(data, f, indent=1, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def update(self, fn) -> dict:
        """Read, change (fn returns the new dict, or None to delete the file), write; under the lock."""
        with self.locked():
            cur = self.load() or {}
            new = fn(dict(cur))
            if new is None:
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
                return {}
            self._write(new)
            return new


def paired(d: dict | None) -> bool:
    return bool(d and d.get("token") and isinstance(d.get("chat_id"), int) and isinstance(d.get("user_id"), int))


def pairing_open(d: dict | None, now: float) -> bool:
    p = (d or {}).get("pairing")
    return bool(d and d.get("token") and isinstance(p, dict) and number(p.get("expires_wall")) is not None
                and now <= p["expires_wall"])


# ---------------------------------------------------------------- signed callback data

ACTIONS = {"a": "approve", "r": "reject", "u": "used", "z": "undo", "s": "resolve"}
KINDS = {"d": "draft", "p": "page"}
_DATA_RE = re.compile(r"^([aruzs])([dp])([0-9a-z]{1,12})\.([0-9a-z]{1,8})\.([A-Za-z0-9_-]{16})$")


def _b36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    n = int(n)
    if n <= 0:
        return "0"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out


def _sig(key: str, chat_id, action: str, kind: str, ref: int, issue: int) -> str:
    msg = f"{chat_id}|{action}{kind}{ref}|{issue}".encode()
    return base64.urlsafe_b64encode(hmac.new(bytes.fromhex(key), msg, hashlib.sha256).digest()[:12]).decode()


def make_data(key: str, chat_id, action: str, kind: str, ref: int, issue: int) -> str:
    """At most about 40 bytes (Telegram allows 64): action, kind, id and issue in base 36, then a
    96-bit HMAC over them and the chat id."""
    return f"{action}{kind}{_b36(ref)}.{_b36(issue)}.{_sig(key, chat_id, action, kind, ref, issue)}"


def parse_data(key: str, chat_id, data) -> tuple | None:
    """(action, kind, ref, issue) for data this pairing signed, else None (forged, garbled, from an
    earlier pairing, or for an action that does not exist on that kind)."""
    if not isinstance(data, str) or len(data) > 64 or not key:
        return None
    m = _DATA_RE.match(data)
    if not m:
        return None
    action, kind, ref, issue, sig = m.group(1), m.group(2), int(m.group(3), 36), int(m.group(4), 36), m.group(5)
    if (kind == "p") != (action == "s"):
        return None
    if not hmac.compare_digest(sig, _sig(key, chat_id, action, kind, ref, issue)):
        return None
    return ACTIONS[action], KINDS[kind], ref, issue


# ---------------------------------------------------------------- captions (plain text, never parse_mode)

def utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le", "replace")) // 2


def fit(s: str, limit: int) -> str:
    """`s` cut to at most `limit` UTF-16 code units (how Telegram counts), with an ellipsis."""
    if utf16_len(s) <= limit:
        return s
    out, n = [], 0
    for ch in s:
        w = 2 if ord(ch) > 0xFFFF else 1
        if n + w > limit - 1:
            break
        out.append(ch)
        n += w
    return "".join(out).rstrip() + "\u2026"


# Terminal escape sequences (CSI, OSC up to its terminator, and two-byte escapes), removed whole.
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[ -/]*[0-~]|\x9b[0-?]*[ -/]*[@-~]")


def clean(text, limit: int) -> str:
    """Untrusted text as one plain line: terminal escape sequences, control characters, line
    breaks and invisible or bidi formatting characters removed, whitespace collapsed, lone
    surrogates replaced, at most `limit`. No parse_mode is ever set, so HTML or Markdown in it is
    shown as written."""
    s = "" if text is None else str(text)
    s = s.encode("utf-8", "replace").decode("utf-8")
    s = _ANSI.sub(" ", s)
    s = strip_invisible(s, " ")
    s = CONTROL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return fit(s, limit)


def untrusted(text, limit: int) -> str:
    """clean() plus links removed (a brain note or page reason must not become a tappable link)."""
    s = clean(text, max(limit * 2, 64))
    try:
        s = strip_urls(s)
    except Exception:
        pass
    return clean(s, limit)


def _mb(n: float) -> str:
    return f"{n / MB:.1f} MB"


def draft_body(it: dict, template, files_note: str) -> str:
    minutes = it.get("review_minutes")
    minutes = float(minutes) if number(minutes) is not None else (template.review_minutes if template else None)
    lines = [f"Draft for review: job {it['job_id']} ({clean(it.get('template_id'), 64)})",
             f"For: {clean(it.get('consumer'), 80)}",
             f"Brief: {untrusted(it.get('brief'), 360)}"]
    if it.get("brain_verdict"):
        note = untrusted(it.get("brain_note"), 200) if it.get("brain_note") else ""
        lines.append(f"Brain: {clean(it['brain_verdict'], 20)}" + (f": {note}" if note else ""))
    if minutes is not None:
        lines.append(f"Review: about {minutes:g} min")
    if files_note:
        lines.append(files_note)
    lines.append("Reply to this message to add a note to your verdict.")
    return fit("\n".join(lines), BODY_MAX)


def private_body(it: dict, template) -> str:
    """A draft of a template flagged private: nothing about its content leaves the Mac."""
    minutes = it.get("review_minutes")
    minutes = float(minutes) if number(minutes) is not None else (template.review_minutes if template else None)
    lines = [f"Draft for review: job {it['job_id']} (private)",
             "Its template is flagged private, so its brief, notes and files stay on the Studio: review it "
             "there (python3 -m foreman.cli review)."]
    if minutes is not None:
        lines.append(f"Review: about {minutes:g} min")
    return "\n".join(lines)


def is_private(it: dict | None, template) -> bool:
    flags = (it or {}).get("flags") if isinstance((it or {}).get("flags"), dict) else {}
    return bool(flags.get("private")) or bool(template is not None and getattr(template, "private", False))


PAGE_LABELS = {"human_review": "a person is needed", "engine_lost": "engine lost", "safe_mode": "SAFE mode",
               "planner_suspended": "planner suspended", "brain_blocked": "no brain can come back"}


def page_body(job, private: bool = False) -> str:
    payload = job.payload if isinstance(job.payload, dict) else {}
    label = PAGE_LABELS.get(job.kind, clean(job.kind, 40))
    reason = ("It is about work of a template flagged private: the details stay on the Studio "
              "(python3 -m foreman.cli resolve)." if private else untrusted(payload.get("reason") or "", 600))
    lines = [f"Foreman page {job.id}: {label}", reason]
    if job.kind == "safe_mode":
        lines.append("Resolve closes this page; leaving SAFE mode still needs `python3 -m foreman.cli clear-safe` "
                     "on the Studio.")
    elif job.kind == "planner_suspended":
        lines.append("The suspension ends by itself; /resume ends it now.")
    return fit("\n".join(x for x in lines if x), BODY_MAX)


def draft_view(it: dict | None) -> str:
    if it is None:
        return "gone"
    if it["state"] == "rejected" and it.get("pruned_at") is not None:
        return "rejected-pruned"
    return str(it["state"])


DRAFT_BUTTONS = {"review": (("Approve", "a"), ("Reject", "r")), "accepted": (("Used it", "u"), ("Undo", "z")),
                 "rejected": (("Undo", "z"),)}
DRAFT_STATUS = {"accepted": "Approved. Tap Used it once you have used it.", "rejected": "Rejected.",
                "rejected-pruned": "Rejected; its files are pruned, so it can no longer be undone.",
                "shipped": "Used.", "expired": "Expired: no verdict within 14 days, so its files were pruned.",
                "failed": "No longer available.", "gone": "This draft no longer exists."}
DESCRIBE = {"review": "waiting for your verdict", "accepted": "approved", "rejected": "rejected",
            "rejected-pruned": "rejected and pruned", "shipped": "used", "expired": "expired", "failed": "failed",
            "gone": "gone", "awaiting_human": "open", "resolved": "resolved"}
ALLOWED = {"approve": ("review",), "reject": ("review",), "used": ("accepted",), "undo": ("accepted", "rejected")}


def status_line(kind: str, view: str, last_action: str | None) -> str:
    if kind == "page":
        return "" if view == "awaiting_human" else ("Resolved." if view == "resolved" else f"Closed ({view}).")
    if view == "review":
        return "Undone: back in the review queue." if last_action == "undo" else ""
    return DRAFT_STATUS.get(view, f"Now {view}.")


# ---------------------------------------------------------------- media

def _ctype(path: str) -> str:
    return {".mp4": "video/mp4", ".m4a": "audio/mp4", ".wav": "audio/wav", ".png": "image/png",
            ".json": "application/json"}.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


@dataclass
class Attempt:
    method: str
    field: str
    path: str
    media: str            # video | audio | document
    note: str = ""
    extra: dict = dataclasses.field(default_factory=dict)


def transcode_m4a(ffmpeg: str, wav: str, out_dir: str, timeout: float) -> str | None:
    """A phone-friendly AAC/M4A copy of a WAV (in the phone's own temp directory), or None."""
    os.makedirs(out_dir, mode=0o700, exist_ok=True)
    out = os.path.join(out_dir, f"draft-{secrets.token_hex(4)}.m4a")
    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", wav, "-vn", "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", out]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as e:
        LOG.warning("ffmpeg could not make an M4A: %s", type(e).__name__)
        res = None
    if res is not None and res.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
        return out
    if res is not None:
        LOG.warning("ffmpeg could not make an M4A (exit %s)", res.returncode)
    try:
        os.unlink(out)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------- the service

class Phone:
    """The phone service. `step()` is one pass (secrets, outbox, button refresh, digest, one
    getUpdates) and returns how long to wait before the next; `run()` loops it. Every Bot API,
    database or handler error is caught at a boundary and turned into a backoff, never a crash."""

    def __init__(self, cfg: studio.Config, clock=time.time, sleep=time.sleep, api_factory=None, ffmpeg="config",
                 lead: bool = True):
        self.cfg = cfg
        self.pc: PhoneConfig = cfg.phone_config()
        self.clock, self.sleep = clock, sleep
        self.api_factory = api_factory or (lambda token: BotApi(token, self.pc.api_base, self.pc.request_timeout_s,
                                                                self.pc.upload_timeout_s))
        self.ffmpeg = studio.find_ffmpeg(cfg.ffmpeg, cfg.home) if ffmpeg == "config" else ffmpeg
        self.secrets = SecretStore(cfg.home)
        self.tmp_dir = os.path.join(cfg.home, "phone", "tmp")
        self.lock = InstanceLock(os.path.join(cfg.home, "phone.lock")) if lead else None
        self.store: Store | None = None
        self.api: BotApi | None = None
        self.d: dict | None = None
        self._sig = object()
        self._token = None
        self.hold_until = 0.0          # a 429 asked for quiet until then (wall time)
        self._fails = 0                # consecutive failed passes (backoff)
        self._retries: dict = {}       # update_id -> transient failures (bounded)
        self._next_refresh = 0.0
        self._more = False
        self._noted: dict = {}         # de-duplicated warnings
        self._step_started: float | None = None
        self._net_down = False
        self._watch_every = 5.0
        self.stop = threading.Event()

    # ------------------------------------------------------------ plumbing

    def db(self) -> Store:
        if self.store is None or self.store.closed:
            d = os.path.dirname(self.cfg.db_path)
            if d:
                os.makedirs(d, exist_ok=True)
            self.store = Store(self.cfg.db_path)
        return self.store

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        if self.lock is not None:
            self.lock.release()

    def _note_once(self, key: str, level: int, msg: str, *args) -> None:
        text = msg % args if args else msg
        if self._noted.get(key) != text:
            self._noted[key] = text
            LOG.log(level, msg, *args)

    def reload(self) -> dict | None:
        """Re-read the secrets file when it changed (phone setup or unpair while the service runs)."""
        sig = self.secrets.signature()
        if sig == self._sig:
            return self.d
        self.d = self.secrets.load() if sig is not None else None  # a PhoneError leaves _sig: read again next time
        self._sig = sig
        token = (self.d or {}).get("token")
        if token != self._token:
            self._token = token
            self.api = self.api_factory(token) if token else None
            if self.api is not None:
                self.api.progress = self._touch
        return self.d

    def _touch(self) -> None:
        """Progress inside a pass (an upload moving): the watchdog measures time since the last one."""
        if self._step_started is not None:
            self._step_started = time.monotonic()

    @property
    def paired(self) -> bool:
        return paired(self.d)

    @property
    def chat_id(self):
        return (self.d or {}).get("chat_id")

    @property
    def key(self) -> str:
        return (self.d or {}).get("hmac_key") or ""

    @property
    def username(self) -> str:
        return str((self.d or {}).get("bot_username") or "")

    def _state(self) -> dict:
        return dict(self.db().conn.execute("SELECT * FROM phone_state WHERE id = 1").fetchone())

    def _set_state(self, **cols) -> None:
        sets = ", ".join(f"{k} = ?" for k in cols)
        with self.db().tx():
            self.db().conn.execute(f"UPDATE phone_state SET {sets} WHERE id = 1", tuple(cols.values()))

    def offset(self) -> int:
        return int(self._state()["update_offset"] or 0)

    def _row(self, kind: str, ref: int) -> dict | None:
        r = self.db().conn.execute("SELECT * FROM phone_messages WHERE kind = ? AND item = ?", (kind, ref)).fetchone()
        return dict(r) if r else None

    def _row_by_message(self, message_id) -> tuple:
        """(row, current?) for a message id: the latest message of a draft or page, or an older copy."""
        if not isinstance(message_id, int):
            return None, False
        r = self.db().conn.execute("SELECT * FROM phone_messages WHERE message_id = ?", (message_id,)).fetchone()
        if r:
            return dict(r), True
        for r in self.db().conn.execute("SELECT * FROM phone_messages WHERE old_ids != '[]'"):
            try:
                if message_id in json.loads(r["old_ids"]):
                    return dict(r), False
            except ValueError:
                continue
        return None, False

    def _update_row(self, row_id: int, expect: dict | None = None, **cols) -> bool:
        sets = ", ".join(f"{k} = ?" for k in cols)
        sql, args = f"UPDATE phone_messages SET {sets} WHERE id = ?", [*cols.values(), row_id]
        for k, v in (expect or {}).items():
            sql += f" AND {k} = ?"
            args.append(v)
        with self.db().tx():
            return self.db().conn.execute(sql, args).rowcount == 1

    def _event(self, kind: str, **data) -> None:
        """An event in gpud's append-only log, in supervisor time (through a non-leader client)."""
        from . import cli
        sup = cli._client(self.cfg)
        try:
            with sup.store.tx():
                sup.store.event(sup._now(), kind, None, **data)
        finally:
            sup.close()

    def clear_messages(self) -> None:
        with self.db().tx():
            self.db().conn.execute("DELETE FROM phone_messages")
            self.db().conn.execute("UPDATE phone_state SET blocked = 0, hint_at = NULL WHERE id = 1")

    # ------------------------------------------------------------ the loop

    def _lead(self) -> bool:
        return self.lock is None or self.lock.held or self.lock.acquire()

    def backoff(self, retry_after: float | None = None) -> float:
        self._fails += 1
        if retry_after is not None:
            return max(1.0, min(float(retry_after), 3600.0))
        return min(self.pc.backoff_max_s, 2.0 ** (self._fails - 1))

    def guard(self, fn, *args) -> float | None:
        """Run fn; a failure becomes a delay (and is logged, redacted), never an exception. A
        failure that means Telegram cannot be talked to right now sets _net_down for the pass."""
        try:
            return fn(*args)
        except TelegramError as e:
            self._set_error(str(e))
            if e.retriable or e.auth:
                self._net_down = True
            if e.code == 429:
                self.hold_until = self.clock() + (e.retry_after or 5.0)
                LOG.warning("Telegram asked to slow down (retry after %ss)", e.retry_after)
                return self.backoff(e.retry_after or 5.0)
            if e.auth:
                self._note_once("auth", logging.ERROR, "Telegram rejected the bot token (%s); revoked? run "
                                "`python3 -m foreman.cli phone setup`", e.code)
                self._fails += 1
                return self.pc.backoff_max_s
            if e.code == 409:
                self._note_once("conflict", logging.WARNING, "another client reads this bot's updates "
                                "(phone setup, a second service or a webhook): waiting")
                self._fails += 1
                return max(self.pc.idle_s, 30.0)
            LOG.warning("Bot API call failed: %s", e)
            return self.backoff()
        except sqlite3.Error as e:
            LOG.warning("database error: %s: %s", type(e).__name__, e)
            if self.store is not None:
                try:
                    self.store.close()
                except Exception:
                    pass
                self.store = None
            return self.backoff()
        except PhoneError as e:
            self._note_once("phone", logging.ERROR, "%s", e)
            self._fails += 1
            return max(self.pc.idle_s, 60.0)
        except Exception as e:
            LOG.exception("phone pass failed: %s", type(e).__name__)
            return self.backoff()

    def _set_error(self, text: str) -> None:
        try:
            self._set_state(last_error=redact(text)[:300])
        except Exception:
            pass

    def step(self) -> float:
        now = self.clock()
        if not self._lead():
            self._note_once("standby", logging.INFO, "another phone service holds phone.lock: standing by")
            return self.pc.idle_s
        self._noted.pop("standby", None)
        try:
            self.reload()
        except PhoneError as e:
            self._note_once("secrets", logging.ERROR, "%s", e)
            return self.pc.idle_s
        try:
            self._set_state(last_poll_wall=now)
        except sqlite3.Error:
            pass
        if self.api is None:
            self._note_once("idle", logging.INFO, "no bot token: run `python3 -m foreman.cli phone setup`")
            return self.pc.idle_s
        if not self.paired:
            if pairing_open(self.d, now):
                delay = self.guard(self.poll_pairing)
                return self._settle(delay)
            self._note_once("idle", logging.INFO, "not paired: run `python3 -m foreman.cli phone setup`")
            return self.pc.idle_s
        self._noted.pop("idle", None)
        # Each part has its own boundary. Only a failure to reach Telegram ends the pass early: a
        # bug or a bad row in the outbox must never stop the owner's button presses being read.
        self._net_down = False
        delays = []
        if now >= self.hold_until and not self._blocked():
            parts = [self.send_outbox]
            if now >= self._next_refresh:
                self._next_refresh = now + self.pc.refresh_s
                parts.append(self.refresh)
            if self.pc.digest:
                parts.append(self.maybe_digest)
            for fn in parts:
                d = self.guard(fn)
                if d is not None:
                    delays.append(d)
                if self._net_down:
                    break
        if not self._net_down:
            d = self.guard(self.poll)
            if d is not None:
                delays.append(d)
        return self._settle(max(delays) if delays else None)

    def _settle(self, delay) -> float:
        if delay is None:
            self._fails = 0
            self._noted.pop("auth", None)
            self._noted.pop("conflict", None)
            return 0.0
        return float(delay)

    def _blocked(self) -> bool:
        try:
            return bool(self._state()["blocked"])
        except sqlite3.Error:
            return False

    def watchdog(self) -> None:
        """Exit (status 70, launchd restarts the service) when a pass made no progress for
        watchdog_s: every call has a timeout, so this only catches a hang nothing else can."""
        limit = self.pc.watchdog_s
        while not self.stop.wait(self._watch_every):
            started = self._step_started
            if limit and started is not None and time.monotonic() - started > limit:
                LOG.critical("a phone pass has run for %.0fs (> watchdog %.0fs); exiting so launchd restarts it",
                             time.monotonic() - started, limit)
                logging.shutdown()
                os._exit(70)

    def run(self, once: bool = False) -> int:
        LOG.info("phone service starting", extra={"fields": {"pid": os.getpid(), "db": self.cfg.db_path,
                                                             "ffmpeg": bool(self.ffmpeg)}})
        self._clean_tmp()
        if self.pc.watchdog_s and not once:
            threading.Thread(target=self.watchdog, name="phone-watchdog", daemon=True).start()
        try:
            while not self.stop.is_set():
                self._step_started = time.monotonic()
                try:
                    delay = self.step()
                except Exception as e:  # step() has its own boundaries; this is the last line of defense
                    LOG.exception("phone step raised: %s", type(e).__name__)
                    delay = self.backoff()
                finally:
                    self._step_started = None
                if once:
                    break
                if delay > 0:
                    self.stop.wait(delay)
        finally:
            self.close()
            LOG.info("phone service stopped")
        return 0

    def _clean_tmp(self) -> None:
        try:
            for name in os.listdir(self.tmp_dir):
                if name.startswith("draft-"):
                    os.unlink(os.path.join(self.tmp_dir, name))
        except OSError:
            pass

    # ------------------------------------------------------------ pairing

    def poll_pairing(self):
        """While a code is open: only a /start <code> or /pair <code> message can do anything.
        Updates after the one that paired are the paired chat's and are handled as such."""
        updates = self.api.get_updates(self.offset(), self.pc.poll_timeout_s)
        for u in updates:
            uid = u.get("update_id") if isinstance(u, dict) else None
            if not isinstance(uid, int) or uid < self.offset():
                continue
            if self.paired:
                self._handle_one(u)
            else:
                try:
                    self.pair_update(u)
                except (sqlite3.Error, PhoneError):
                    raise
                except Exception as e:
                    LOG.warning("a pairing update raised %s; skipped", type(e).__name__)
            self._set_state(update_offset=uid + 1)
        return None

    def pair_update(self, u: dict) -> bool:
        """True when this update paired the chat. Only a /start <code> or /pair <code> message counts."""
        msg = u.get("message") if isinstance(u.get("message"), dict) else None
        text = msg.get("text") if msg else None
        m = PAIR_RE.match(text.strip()) if isinstance(text, str) else None
        if not m:
            return False
        if m.group(2) and self.username and m.group(2).lower() != self.username.lower():
            return False
        code = m.group(3)
        chat = msg.get("chat") if isinstance(msg.get("chat"), dict) else {}
        user = msg.get("from") if isinstance(msg.get("from"), dict) else {}
        now = self.clock()
        result = {}

        def change(d: dict):
            p = d.get("pairing")
            if not d.get("token") or not isinstance(p, dict) or paired(d):
                result["why"] = "no pairing pending"
                return d
            if number(p.get("expires_wall")) is None or now > p["expires_wall"]:
                result["why"] = "expired"
                d["pairing"], d["burned"] = None, "the code expired"
                return d
            ok = hmac.compare_digest(code_mac(d["hmac_key"], code), str(p.get("mac") or ""))
            if chat.get("type") != "private":
                result["why"] = "group"
                if ok:  # seen by everyone in that chat: it must not pair anyone any more
                    d["pairing"], d["burned"] = None, "the code was sent in a group chat"
                return d
            if user.get("is_bot") or not isinstance(user.get("id"), int) or not isinstance(chat.get("id"), int):
                result["why"] = "not a person"
                return d
            if not ok:
                p["failures"] = int(p.get("failures") or 0) + 1
                result["why"] = "wrong code"
                result["failures"] = p["failures"]
                if p["failures"] >= self.pc.pairing_max_failures:
                    d["pairing"], d["burned"] = None, f"{p['failures']} wrong codes"
                return d
            d.update(chat_id=chat["id"], user_id=user["id"], paired_wall=now, pairing=None, burned=None,
                     hmac_key=new_key())
            result["paired"] = True
            return d

        self.d = self.secrets.update(change)
        self._sig = self.secrets.signature()
        register_secret(self.d.get("hmac_key"))
        if not result.get("paired"):
            why = result.get("why")
            if why == "wrong code":
                LOG.warning("a wrong pairing code arrived (%d of %d allowed)", result.get("failures", 0),
                            self.pc.pairing_max_failures)
            elif why == "group":
                LOG.warning("a pairing attempt from a group chat was refused%s",
                            "; the code is burned (run phone setup again)" if self.d.get("burned") else "")
            else:
                LOG.info("a pairing attempt was ignored (%s)", why)
            return False
        self.clear_messages()
        self._event("phone_paired", bot=self.username)
        LOG.warning("paired with one private chat; drafts and pages go there now")
        try:
            self.api.send_message(self.chat_id, "Paired with Foreman. Drafts for review and pages will arrive here.\n"
                                  + HELP)
        except TelegramError as e:
            LOG.warning("could not send the pairing confirmation: %s", e)
        return True

    # ------------------------------------------------------------ updates

    def poll(self):
        timeout = 0 if self._more else self.pc.poll_timeout_s
        updates = self.api.get_updates(self.offset(), timeout)
        self._set_state(last_ok_wall=self.clock())
        for u in updates:
            uid = u.get("update_id") if isinstance(u, dict) else None
            if not isinstance(uid, int):
                continue
            if uid < self.offset():
                continue  # handled already (redelivered)
            try:
                self._handle_one(u)
            except sqlite3.OperationalError:  # the database is busy: try this update again, a few times
                n = self._retries.get(uid, 0) + 1
                self._retries[uid] = n
                if len(self._retries) > 100:
                    self._retries.clear()
                if n < 3:
                    raise
                LOG.error("update %d kept failing on a busy database; skipped", uid)
            self._retries.pop(uid, None)
            self._set_state(update_offset=uid + 1)
        return None

    def _handle_one(self, u: dict) -> None:
        """One update, inside its own boundary (a poison update is logged and skipped, its offset
        still advances); only a busy database is raised so the update is tried again."""
        try:
            self.handle(u)
        except sqlite3.OperationalError:
            raise
        except TelegramError as e:
            LOG.warning("update %s: Bot API call failed: %s", u.get("update_id"), e)
            if e.code == 429:
                self.hold_until = self.clock() + (e.retry_after or 5.0)
        except Exception as e:
            LOG.exception("update %s raised %s; skipped", u.get("update_id"), type(e).__name__)

    def _authorized(self, chat, user) -> bool:
        return (isinstance(chat, dict) and isinstance(user, dict) and chat.get("type") == "private"
                and chat.get("id") == self.chat_id and user.get("id") == self.d.get("user_id")
                and not user.get("is_bot"))

    def handle(self, u: dict) -> None:
        if isinstance(u.get("callback_query"), dict):
            return self.on_callback(u["callback_query"])
        if isinstance(u.get("message"), dict):
            return self.on_message(u["message"])
        LOG.info("ignored an update of another type")

    def _say(self, text: str, reply_to=None, markup=None) -> dict:
        return self.api.send_message(self.chat_id, text, markup=markup, reply_to=reply_to)

    def on_message(self, msg: dict) -> None:
        chat, user = msg.get("chat"), msg.get("from")
        if not self._authorized(chat, user):
            ctype = chat.get("type") if isinstance(chat, dict) else None
            LOG.info("ignored a message from a chat or user that is not paired (%s chat)",
                     clean(ctype, 20) or "unknown")
            return
        if self._blocked():
            self._set_state(blocked=0)
            LOG.info("the paired chat talks to the bot again: sending resumes")
        text = msg.get("text")
        mid = msg.get("message_id")
        if not isinstance(text, str) or not text.strip():
            return self._hint(mid)
        reply = msg.get("reply_to_message")
        if text.strip().startswith("/"):
            return self.on_command(text.strip(), mid)
        if isinstance(reply, dict) and isinstance(reply.get("message_id"), int):
            return self.on_reply(reply["message_id"], text, mid)
        self._hint(mid)

    def _hint(self, mid) -> None:
        now = self.clock()
        last = self._state()["hint_at"]
        if last is not None and now - last < self.pc.hint_every_s:
            return
        self._set_state(hint_at=now)
        self._say("Sorry, I only understand a few commands.\n" + HELP, reply_to=mid)

    def on_command(self, text: str, mid) -> None:
        cmd = text.split()[0].lower()
        if "@" in cmd:
            cmd, _, bot = cmd.partition("@")
            if self.username and bot != self.username.lower():
                return  # a command for another bot
        if cmd in ("/start", "/help"):
            self._say(HELP, reply_to=mid)
        elif cmd == "/pair":
            self._say("This chat is already paired.", reply_to=mid)
        elif cmd == "/queue":
            res = self.resend_pending()
            if res["drafts"] or res["pages"]:
                self._say(f"Sent again: {res['drafts']} draft(s), {res['pages']} page(s)."
                          + (f" {res['failed']} could not be sent." if res["failed"] else ""), reply_to=mid)
            else:
                self._say("Nothing is waiting for you.", reply_to=mid)
        elif cmd == "/status":
            self._say(self.status_text(), reply_to=mid)
        elif cmd in ("/pause", "/resume"):
            from . import cli
            action = cmd[1:]
            st = cli.apply_planner(self.cfg, action, note="from the phone")
            self._say("Planner paused: gpud keeps running the queue. /resume to plan again." if action == "pause"
                      else f"Planner resumed ({st['state']}).", reply_to=mid)
        else:
            self._hint(mid)

    # ------------------------------------------------------------ notes (replies to a draft)

    def on_reply(self, target_mid: int, text: str, mid) -> None:
        from . import cli
        row, _ = self._row_by_message(target_mid)
        if row is None or row["kind"] != "draft":
            self._say("Notes can only be added to drafts: reply to a draft message.", reply_to=mid)
            return
        note = clean(text, 500)
        it = ledger.item(self.db(), row["item"])
        if it is None:
            self._say("That draft no longer exists; the note was not saved.", reply_to=mid)
            return
        if it["state"] == "review":
            joined = fit(f"{row['note']} | {note}" if row.get("note") else note, 1000)
            self._update_row(row["id"], note=joined)
            self._say(f"Noted: it goes with your verdict on job {it['job_id']}.", reply_to=mid)
            LOG.info("note kept for draft %d until its verdict", it["id"])
        elif it["state"] in ("accepted", "rejected", "shipped"):
            try:
                cli.apply_note(self.cfg, it["job_id"], note, self.pc.note_window_s, by="phone")
            except ledger.LedgerError as e:
                self._say(clean(str(e), 300), reply_to=mid)
                return
            self._say(f"Note added to your verdict on job {it['job_id']}.", reply_to=mid)
            LOG.info("note added to the verdict on draft %d", it["id"])
        else:
            self._say(f"Job {it['job_id']} is {DESCRIBE.get(draft_view(it), it['state'])}; the note was not saved.",
                      reply_to=mid)

    # ------------------------------------------------------------ buttons

    def keyboard(self, kind: str, ref: int, issue: int, view: str) -> dict:
        if kind == "draft":
            buttons = DRAFT_BUTTONS.get(view, ())
        else:
            buttons = (("Resolve", "s"),) if view == "awaiting_human" else ()
        row = [{"text": t, "callback_data": make_data(self.key, self.chat_id, a, kind[0], ref, issue)}
               for t, a in buttons]
        return {"inline_keyboard": [row] if row else []}

    def view_of(self, kind: str, ref: int) -> str:
        if kind == "draft":
            return draft_view(ledger.item(self.db(), ref))
        job = self.db().job(ref)
        return "gone" if job is None else str(job.state)

    def _answer(self, cq_id, text: str) -> None:
        try:
            self.api.answer_callback(cq_id, text)
        except TelegramError as e:  # "query is too old" and the like: the owner still sees the edit
            LOG.info("could not answer a button press: %s", e)

    def _strip(self, message_id) -> None:
        if not isinstance(message_id, int):
            return
        try:
            self.api.edit_markup(self.chat_id, message_id, {"inline_keyboard": []})
        except TelegramError as e:
            if not e.not_modified:
                LOG.info("could not remove the buttons of an old message: %s", e)

    def redraw(self, row: dict, view: str | None = None, last_action: str | None = None, **cols) -> dict | None:
        """Draw the row's message again for the current state: a new issue (old buttons go stale),
        the status line and the buttons that state allows. Terminal states close the row."""
        view = view or self.view_of(row["kind"], row["item"])
        issue = row["issue"] + 1
        last = last_action if last_action is not None else row.get("last_action")
        terminal = (view in TERMINAL_DRAFT) if row["kind"] == "draft" else view != "awaiting_human"
        state = "closed" if terminal else row["state"]
        if not self._update_row(row["id"], expect={"issue": row["issue"]}, issue=issue, shown=view, state=state,
                                last_action=last, **cols):
            return None  # someone else drew it meanwhile
        text = row.get("body") or ""
        line = status_line(row["kind"], view, last)
        if line:
            text = f"{text}\n\n{line}"
        if row.get("message_id"):
            try:
                self.api.edit(self.chat_id, row["message_id"], row.get("media") or "text", fit(text, CAPTION_MAX),
                              self.keyboard(row["kind"], row["item"], issue, view))
            except TelegramError as e:
                if e.retriable:  # the refresh pass draws it again (shown no longer matches any state)
                    self._update_row(row["id"], shown=None)
                    raise
                if not e.not_modified:
                    LOG.info("could not edit %s %d: %s", row["kind"], row["item"], e)
                    if "not found" in e.description.lower() or "can't be edited" in e.description.lower():
                        self._update_row(row["id"], state="closed")
        return self._row(row["kind"], row["item"])

    def on_callback(self, cq: dict) -> None:
        from . import cli
        msg = cq.get("message") if isinstance(cq.get("message"), dict) else {}
        user, chat = cq.get("from"), msg.get("chat")
        cq_id = cq.get("id")
        if not self._authorized(chat, user):
            LOG.info("ignored a button press from a chat or user that is not paired")
            return
        mid = msg.get("message_id")
        parsed = parse_data(self.key, self.chat_id, cq.get("data"))
        if parsed is None:
            LOG.warning("rejected a button whose data is not signed by this pairing (forged, garbled or old)")
            self._answer(cq_id, "This button is not valid any more.")
            known, current = self._row_by_message(mid)
            if known is not None and current and known["state"] == "sent":
                self.redraw(known)  # the message is ours: give it buttons this pairing signed
            else:
                self._strip(mid)
            return
        action, kind, ref, issue = parsed
        row = self._row(kind, ref)
        if row is None or row["state"] not in ("sent", "closed") or not row.get("message_id"):
            self._answer(cq_id, "This message is not known any more.")
            self._strip(mid)
            return
        if mid != row["message_id"]:
            self._answer(cq_id, "This is an older copy: use the latest message for it.")
            self._strip(mid)
            return
        view = self.view_of(kind, ref)
        if issue != row["issue"]:
            same = row.get("last_action") == action and view in self._after(action)
            self._answer(cq_id, f"Already done: {DESCRIBE.get(view, view)}." if same
                         else "This message was out of date; it shows the current state now.")
            self.redraw(row, view)
            return
        if view != row["shown"]:
            self._answer(cq_id, f"This changed meanwhile: it is {DESCRIBE.get(view, view)} now.")
            self.redraw(row, view)
            return
        if kind == "page":
            return self._resolve(cq_id, row, ref)
        it = ledger.item(self.db(), ref)
        if view not in ALLOWED[action]:
            self._answer(cq_id, f"Not possible: this draft is {DESCRIBE.get(view, view)}.")
            self.redraw(row, view)
            return
        try:
            if action in ("approve", "reject"):
                res = cli.apply_verdict(self.cfg, it["job_id"], "accept" if action == "approve" else "reject",
                                        None, row.get("note") or None, by="phone")
                p, killed = res["p_ship"], res["killed"]
                answer = ("Approved." if action == "approve" else "Rejected.") + (
                    f" P(ship) for {clean(it['template_id'], 64)} is now {p:.2f}." if p is not None else "")
            elif action == "undo":
                res = cli.apply_verdict(self.cfg, it["job_id"], "retract", by="phone")
                p, killed = res["p_ship"], []
                answer = "Undone: back in the review queue." + (f" P(ship) is {p:.2f} again." if p is not None else "")
            else:
                cli.apply_ship(self.cfg, it["job_id"], None, by="phone")
                killed, answer = [], "Recorded as used."
        except ledger.LedgerError as e:
            self._answer(cq_id, clean(str(e), ANSWER_MAX))
            self.redraw(row)
            return
        LOG.info("%s recorded from the phone for draft %d (job %d)", action, it["id"], it["job_id"])
        self._answer(cq_id, fit(answer, ANSWER_MAX))
        self.redraw(row, last_action=action, acted_at=self.clock(),
                    note=None if action in ("approve", "reject") else row.get("note"))
        for tid, reason, detail in killed:
            self._say(f"Template {clean(tid, 64)} was paused by a kill criterion: {clean(reason, 40)} "
                      f"({clean(detail, 300)}).")

    @staticmethod
    def _after(action: str) -> tuple:
        return {"approve": ("accepted", "shipped"), "reject": ("rejected", "rejected-pruned"), "used": ("shipped",),
                "undo": ("review",), "resolve": ("resolved",)}.get(action, ())

    def _resolve(self, cq_id, row: dict, page_id: int) -> None:
        from . import cli
        job = self.db().job(page_id)
        try:
            cli.apply_resolve(self.cfg, page_id, "resolved from the phone", by="phone")
        except ledger.LedgerError as e:
            self._answer(cq_id, clean(str(e), ANSWER_MAX))
            self.redraw(row)
            return
        LOG.info("page %d resolved from the phone", page_id)
        extra = " SAFE mode itself still needs clear-safe on the Studio." if job and job.kind == "safe_mode" else ""
        self._answer(cq_id, "Resolved." + extra)
        self.redraw(row, last_action="resolve", acted_at=self.clock())

    # ------------------------------------------------------------ the outbox

    def _templates(self) -> dict:
        return load_templates(os.path.join(self.cfg.home, "templates"))[0]

    def _page_private(self, job) -> bool:
        """A page raised inside the lineage of a planned item whose template is flagged private."""
        try:
            it = ledger.item_for_job(self.db(), job.id)  # the page's lineage root is the item's job
        except Exception:
            it = None
        return it is not None and is_private(it, self._templates().get(it.get("template_id")))

    def _item_pages(self) -> set:
        """Pages the brain raised about a draft: the draft carries them (its verdict resolves them)."""
        out = set()
        for it in ledger.items(self.db(), "review"):
            if it.get("review_job_id"):
                out.update(j.id for j in self.db().children(it["review_job_id"]) if j.requires == "human")
        return out

    def _open_pages(self) -> list:
        skip = self._item_pages()
        return [j for j in self.db().jobs("awaiting_human") if j.id not in skip]

    def _claim(self, kind: str, ref: int, resend: bool = False) -> tuple | None:
        """(claim token, row before the claim) when this process may send it now, else None."""
        now = self.clock()
        claim = secrets.token_hex(8)
        conn = self.db().conn
        with self.db().tx():
            cur = conn.execute("INSERT INTO phone_messages (item, kind, state, claim, claimed_at, created_at, attempts) "
                               "VALUES (?, ?, 'sending', ?, ?, ?, 1) ON CONFLICT(item, kind) DO NOTHING",
                               (ref, kind, claim, now, now))
            if cur.rowcount == 1:
                return claim, None
            row = dict(conn.execute("SELECT * FROM phone_messages WHERE kind = ? AND item = ?", (kind, ref)).fetchone())
            due = row["state"] == "pending" and (row["next_try_at"] is None or row["next_try_at"] <= now)
            stale = row["state"] == "sending" and (row["claimed_at"] or 0) < now - self.pc.stale_claim_s
            again = resend and row["state"] in ("sent", "closed", "failed", "pending")
            if not (due or stale or again):
                return None
            if stale:
                LOG.warning("%s %d was being sent by a process that stopped; sending it again (an earlier copy "
                            "may exist)", kind, ref)
            ok = conn.execute("UPDATE phone_messages SET state = 'sending', claim = ?, claimed_at = ?, "
                              "attempts = attempts + 1 WHERE id = ? AND state = ? AND COALESCE(claim, '') = ?",
                              (claim, now, row["id"], row["state"], row["claim"] or "")).rowcount == 1
            return (claim, row) if ok else None

    def _release(self, kind: str, ref: int, claim: str, prev: dict | None, err: Exception) -> None:
        """A send that failed: back to where it was, with a retry time (or failed for good)."""
        now = self.clock()
        row = self._row(kind, ref)
        if row is None or row["claim"] != claim:
            return
        retriable = isinstance(err, TelegramError) and (err.retriable or err.code == 403) or \
            isinstance(err, (OSError, sqlite3.OperationalError))
        if prev is not None and prev["state"] in ("sent", "closed") and prev.get("message_id"):
            state = prev["state"]  # a resend that failed: the earlier message still stands
        elif retriable or row["attempts"] < 5:
            state = "pending"
        else:
            state = "failed"
        if isinstance(err, TelegramError) and err.code == 429:
            wait = err.retry_after or 5.0
        else:
            wait = min(self.pc.backoff_max_s, 5.0 * 2 ** max(0, row["attempts"] - 1))
        self._update_row(row["id"], expect={"claim": claim}, state=state, claim=None, next_try_at=now + wait,
                         error=redact(f"{type(err).__name__}: {err}")[:300])
        if isinstance(err, TelegramError) and err.code == 403:
            self._set_state(blocked=1)
            LOG.warning("the paired chat refuses messages from the bot (blocked?); sending stops until it writes again")

    def send_outbox(self):
        """Send what entered the review queue and the open pages, each once, a few per pass."""
        sent = 0
        self._more = False
        work = [("draft", it["id"], it) for it in ledger.review_queue(self.db())]
        if self.pc.pages:
            work += [("page", j.id, j) for j in self._open_pages()]
        for kind, ref, obj in work:
            row = self._row(kind, ref)
            if row is not None and row["state"] in ("sent", "closed", "failed"):
                continue
            if sent >= self.pc.max_sends_per_pass:
                self._more = True
                break
            got = self._claim(kind, ref)
            if got is None:
                continue
            if sent and self.pc.send_interval_s:
                self.sleep(self.pc.send_interval_s)
            self.deliver(kind, obj, got[0], got[1])
            sent += 1
        return None

    def deliver(self, kind: str, obj, claim: str, prev: dict | None) -> dict:
        """Send one draft or page under a claim and record it. A Telegram failure that is worth
        retrying puts the row back (with a retry time) and is raised, so the pass stops."""
        ref = obj["id"] if kind == "draft" else obj.id
        try:
            if kind == "draft":
                view = draft_view(obj)
                media, msg, body = self._send_draft(obj, (prev or {}).get("issue", 0) + 1, view)
            else:
                view = str(obj.state)
                body = page_body(obj, self._page_private(obj))
                msg = self._say(body, markup=self.keyboard("page", ref, (prev or {}).get("issue", 0) + 1, view))
                media = "text"
        except BaseException as e:
            self._release(kind, ref, claim, prev, e if isinstance(e, Exception) else RuntimeError("stopped"))
            raise
        mid = msg.get("message_id") if isinstance(msg, dict) else None
        old = json.loads((prev or {}).get("old_ids") or "[]")
        if prev is not None and prev.get("message_id"):
            old = (old + [prev["message_id"]])[-10:]
        issue = (prev or {}).get("issue", 0) + 1
        now = self.clock()
        cols = dict(state="sent", claim=None, message_id=mid if isinstance(mid, int) else None, media=media, body=body,
                    issue=issue, shown=view, sent_at=now, next_try_at=None, error=None, old_ids=json.dumps(old))
        for attempt in range(3):  # the message is out: recording it must not lose to a busy database
            try:
                ok = self._update_row(self._row(kind, ref)["id"], expect={"claim": claim}, **cols)
                break
            except sqlite3.OperationalError:
                if attempt == 2:
                    raise
                self.sleep(0.5)
        if not ok:
            LOG.warning("%s %d was claimed by another process while it was sent", kind, ref)
        if prev is not None and prev.get("message_id"):
            self._strip(prev["message_id"])  # the older copy's buttons are stale now
        LOG.info("sent %s %d as %s", kind, ref, media)
        return self._row(kind, ref)

    def _files(self, it: dict) -> tuple:
        """(usable files, how many were refused): regular files strictly inside the artifacts
        directory (resolved), never a symlink and never a file with a second hard link (which could
        be some other file's contents under an artifact's name)."""
        ok, refused = [], 0
        for p in it.get("artifacts") or []:
            if not isinstance(p, str) or not p:
                continue
            if not os.path.lexists(p):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or not retention._inside(p, self.cfg.artifacts_dir):
                refused += 1
                continue
            ok.append(p)
        return ok, refused

    def plan_media(self, it: dict) -> tuple:
        """([Attempt], note for the caption, temp files to delete) for a draft's files."""
        files, refused = self._files(it)
        limit, vmax = self.pc.upload_max_mb * MB, self.pc.video_max_mb * MB
        by_ext: dict = {}
        for p in files:
            by_ext.setdefault(os.path.splitext(p)[1].lower(), []).append(p)
        pngs = [p for p in by_ext.get(".png", []) if os.path.basename(p) != "waveform.png"]
        mp4 = (by_ext.get(".mp4") or [None])[0]
        is_video = bool(mp4 or any(os.path.basename(p) == "video.json" for p in files)
                        or any(os.path.basename(p).startswith("frame_") for p in pngs))
        attempts, temps, note = [], [], ""
        frame = sorted(pngs)[len(pngs) // 2] if pngs else None
        if not files:
            note = ("Files: refused (outside the artifacts directory)." if refused
                    else "Files: no longer on the Studio (pruned or moved); this is the text only.")
            return attempts, note, temps
        if is_video:
            size = os.path.getsize(mp4) if mp4 else None
            if mp4 and size <= vmax:
                attempts.append(Attempt("sendVideo", "video", mp4, "video", f"Video: MP4, {_mb(size)}.",
                                        dict(self._video_meta(files), supports_streaming=True)))
                if size <= limit:
                    attempts.append(Attempt("sendDocument", "document", mp4, "document", f"Video: MP4 as a file, {_mb(size)}."))
            why = (f"the MP4 is {_mb(size)}, over the {self.pc.video_max_mb:g} MB phone limit" if mp4 and size > vmax
                   else "no MP4 (ffmpeg was not available when it rendered)" if not mp4 else "Telegram refused the MP4")
            if frame and os.path.getsize(frame) <= limit:
                attempts.append(Attempt("sendDocument", "document", frame, "document",
                                        f"Video: one sampled frame; {why}; the full clip is on the Studio."))
            note = f"Video: {why}; the clip is on the Studio."
            return attempts, note, temps
        wav = (by_ext.get(".wav") or [None])[0]
        if wav:
            wsize = os.path.getsize(wav)
            m4a = transcode_m4a(self.ffmpeg, wav, self.tmp_dir, self.pc.transcode_timeout_s) if self.ffmpeg else None
            if m4a:
                temps.append(m4a)
                if os.path.getsize(m4a) <= limit:
                    attempts.append(Attempt("sendAudio", "audio", m4a, "audio",
                                            f"Audio: M4A for the phone ({_mb(os.path.getsize(m4a))}); the WAV is on the Studio.",
                                            {"title": f"Foreman job {it['job_id']}"}))
            if wsize <= limit:
                attempts.append(Attempt("sendDocument", "document", wav, "document",
                                        f"Audio: the WAV as a file ({_mb(wsize)})"
                                        + ("" if self.ffmpeg else "; no ffmpeg for an M4A") + "."))
            note = (f"Audio: the WAV is {_mb(wsize)}, over Telegram's {self.pc.upload_max_mb:g} MB limit; it is on the Studio."
                    if wsize > limit else "Audio: Telegram refused the file; it is on the Studio.")
            wf = [p for p in by_ext.get(".png", []) if os.path.basename(p) == "waveform.png"]
            if wsize > limit and wf and os.path.getsize(wf[0]) <= limit:
                attempts.append(Attempt("sendDocument", "document", wf[0], "document",
                                        f"Audio: its waveform; the WAV is {_mb(wsize)}, over Telegram's "
                                        f"{self.pc.upload_max_mb:g} MB limit, and is on the Studio."))
            return attempts, note, temps
        small = [p for p in files if os.path.getsize(p) <= limit]
        if small:
            attempts.append(Attempt("sendDocument", "document", small[0], "document", "File attached."))
        return attempts, "Files: none Telegram can show; they are on the Studio.", temps

    def _video_meta(self, files: list) -> dict:
        meta = [p for p in files if os.path.basename(p) == "video.json"]
        if not meta:
            return {}
        try:
            with open(meta[0]) as f:
                v = json.load(f)
            w, h, fps, n = (number(v.get(k)) for k in ("width", "height", "fps", "frame_count"))
            out = {}
            if w and h and 0 < w <= 10000 and 0 < h <= 10000:
                out.update(width=int(w), height=int(h))
            if fps and n and fps > 0:
                out["duration"] = max(1, int(round(n / fps)))
            return out
        except (OSError, ValueError, AttributeError):
            return {}

    def _send_draft(self, it: dict, issue: int, view: str) -> tuple:
        template = self._templates().get(it.get("template_id"))
        markup = self.keyboard("draft", it["id"], issue, view)
        if is_private(it, template):  # no brief, no brain note, no file: only that it waits, and the buttons
            body = private_body(it, template)
            return "text", self._say(body, markup=markup), body
        attempts, note, temps = self.plan_media(it)
        try:
            refusal = None
            for att in attempts:
                body = draft_body(it, template, att.note)
                try:
                    msg = self.api.send_file(att.method, att.field, self.chat_id, att.path, fit(body, CAPTION_MAX),
                                             markup, att.extra, _ctype(att.path))
                    return att.media, msg, body
                except TelegramError as e:
                    if e.retriable or e.code in (401, 403, 404):
                        raise
                    refusal = e.description  # 400, 413: try the next way of sending it
                    LOG.warning("Telegram refused %s for draft %d (%s); trying the next way", att.method, it["id"], e)
                except OSError as e:  # the file went away or shrank while being read
                    refusal = f"the file could not be read ({type(e).__name__})"
                    LOG.warning("draft %d: %s could not read its file; trying the next way", it["id"], att.method)
            if refusal is not None:
                note = f"Files: Telegram refused them ({clean(refusal, 80)}); they are on the Studio."
            body = draft_body(it, template, note)
            return "text", self._say(fit(body, CAPTION_MAX), markup=markup), body
        finally:
            for p in temps:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def resend_pending(self) -> dict:
        """/queue and `phone resend`: every draft still awaiting a verdict and every open page goes
        out again as a new message (its older copy's buttons stop working)."""
        out = {"drafts": 0, "pages": 0, "failed": 0}
        work = [("draft", it["id"], it) for it in ledger.review_queue(self.db())]
        if self.pc.pages:
            work += [("page", j.id, j) for j in self._open_pages()]
        n = 0
        for kind, ref, obj in work[:self.pc.resend_max]:
            got = self._claim(kind, ref, resend=True)
            if got is None:
                continue
            if n and self.pc.send_interval_s:
                self.sleep(self.pc.send_interval_s)
            n += 1
            try:
                self.deliver(kind, obj, got[0], got[1])
                out["drafts" if kind == "draft" else "pages"] += 1
            except TelegramError as e:
                out["failed"] += 1
                if e.retriable or e.code == 403:
                    break
        return out

    def refresh(self):
        """Redraw messages whose draft or page changed elsewhere (a CLI verdict, a retract, the
        retention pruning, a page resolved on the Studio), a few per pass."""
        n = 0
        rows = self.db().conn.execute("SELECT * FROM phone_messages WHERE state = 'sent' AND kind IN ('draft', 'page') "
                                      "ORDER BY id").fetchall()
        for r in rows:
            row = dict(r)
            view = self.view_of(row["kind"], row["item"])
            if view == row["shown"]:
                continue
            self.redraw(row, view)
            n += 1
            if n >= self.pc.max_refresh_per_pass:
                break
        return None

    # ------------------------------------------------------------ status and digest

    def status_text(self) -> str:
        from . import cli
        sup = cli._client(self.cfg)
        try:
            store = sup.store
            now = sup._now()
            st = store.state()
            pst = planner_mod.status(store, now)
            queue = ledger.review_queue(store)
            pages = store.jobs("awaiting_human")
            templates = load_templates(os.path.join(self.cfg.home, "templates"))[0]
            stats = ledger.template_stats(store, templates, now) if templates else {}
        finally:
            sup.close()
        lines = ["Foreman status"]
        age = time.time() - st["t_wall"] if st.get("t_wall") else None
        lines.append(f"gpud: {clean(st['mode'], 20)}" + (f", SAFE: {untrusted(st['safe_reason'], 160)}" if st["safe"] else "")
                     + (f"; last tick {age:.0f} s ago" if age is not None else "; no tick recorded yet"))
        due = pst.get("next_due_in_s")
        lines.append(f"planner: {pst['state']}" + (f", next tick in {max(0, due) / 60:.0f} min"
                                                   if due is not None and pst["state"] == "active" else ""))
        minutes = sum(float(i.get("review_minutes") or 0) for i in queue)
        lines.append(f"review queue: {len(queue)} draft(s), about {minutes:g} min; open pages: {len(pages)}")
        caps = []
        for tid, t in sorted(templates.items()):
            if not t.active:
                continue
            s = stats.get(tid) or {}
            caps.append(f"{clean(tid, 40)} {s.get('week', 0)}/{t.per_week} this week, stock {s.get('stock', 0)}/"
                        f"{min(t.max_stock, int(t.two_week_use))}")
        if caps:
            lines.append("caps: " + "; ".join(caps[:8]) + (" ..." if len(caps) > 8 else ""))
        return fit("\n".join(lines), 3500)

    def maybe_digest(self):
        now = self.clock()
        lt = time.localtime(now)
        if lt.tm_hour < self.pc.digest_hour:
            return None
        day = lt.tm_year * 10000 + lt.tm_mon * 100 + lt.tm_mday
        row = self._row("digest", day)
        if row is not None and row["state"] != "pending":
            return None
        got = self._claim("digest", day)
        if got is None:
            return None
        try:
            msg = self._say(self.digest_text())
        except BaseException as e:
            self._release("digest", day, got[0], got[1], e if isinstance(e, Exception) else RuntimeError("stopped"))
            raise
        row = self._row("digest", day)
        self._update_row(row["id"], state="closed", claim=None, message_id=msg.get("message_id"), media="text",
                         sent_at=now)
        return None

    def digest_text(self) -> str:
        from . import cli
        sup = cli._client(self.cfg)
        try:
            store, now = sup.store, sup._now()
            queue = ledger.review_queue(store)
            pages = store.jobs("awaiting_human")
            templates = load_templates(os.path.join(self.cfg.home, "templates"))[0]
            stats = ledger.template_stats(store, templates, now)
            pst = planner_mod.status(store, now)
        finally:
            sup.close()
        week = [s["7d"] for s in stats.values()]
        acc, rej = sum(w["accepted"] for w in week), sum(w["rejected"] for w in week)
        shipped, usd = sum(w["shipped"] for w in week), sum(w["usd"] for w in week)
        minutes = sum(float(i.get("review_minutes") or 0) for i in queue)
        return (f"Foreman daily summary\nWaiting for you: {len(queue)} draft(s) (about {minutes:g} min), "
                f"{len(pages)} open page(s)\nLast 7 days: {acc} accepted, {rej} rejected, {shipped} used (${usd:g})\n"
                f"Planner: {pst['state']}\n/queue sends the waiting drafts again.")


HELP = ("Commands: /queue sends the drafts and pages waiting for you again, /status shows gpud and the planner, "
        "/pause and /resume pause or resume the planner, /help shows this. Reply to a draft to add a note to "
        "your verdict.")


# ---------------------------------------------------------------- the CLI (python3 -m foreman.cli phone ...)

def read_token(stdin=None) -> str:
    """The bot token: from stdin when it is not a terminal (a pipe or a file), else a hidden prompt."""
    stdin = sys.stdin if stdin is None else stdin
    try:
        interactive = stdin.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if interactive:
        text = getpass.getpass("Telegram bot token from @BotFather (input hidden): ")
    else:
        text = stdin.readline(4096)
    text = (text or "").strip()
    register_secret(text)
    if not TOKEN_RE.fullmatch(text):
        raise PhoneError("that does not look like a bot token (digits, a colon, then about 35 letters, digits, "
                         "'-' or '_'); copy it again from @BotFather")
    return text


def _mask(n) -> str:
    return f"...{str(n)[-4:]}" if n is not None else "-"


def _when(wall) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(wall)) if wall else "-"


def cli_setup(cfg, wait: bool = True, timeout: float | None = None, out=print, stdin=None, clock=time.time,
              sleep=time.sleep) -> int:
    pc = cfg.phone_config()
    store = SecretStore(cfg.home)
    token = read_token(stdin)
    api = BotApi(token, pc.api_base, pc.request_timeout_s, pc.upload_timeout_s)
    try:
        me = api.get_me()
    except TelegramError as e:
        if e.auth:
            raise PhoneError("Telegram rejected that token: copy it again from @BotFather (/mybots, API Token)")
        raise PhoneError(f"could not reach Telegram to check the token: {e}")
    username = str(me.get("username") or "")
    if not USERNAME_RE.fullmatch(username) or not isinstance(me.get("id"), int) or not me.get("is_bot", True):
        raise PhoneError("getMe did not describe a bot with a username")
    code, key = new_code(), new_key()
    register_secret(code)
    now = clock()
    expires = now + pc.pairing_ttl_s
    old = store.load() if store.exists() else None
    store.update(lambda d: {"token": token, "bot_id": me["id"], "bot_username": username, "hmac_key": key,
                            "pairing": {"mac": code_mac(key, code), "expires_wall": expires, "failures": 0,
                                        "created_wall": now},
                            "chat_id": None, "user_id": None, "paired_wall": None, "burned": None, "version": 1})
    out(f"token accepted for @{username}; saved in {store.path} (mode 0600, directory 0700)")
    if paired(old):
        out("the previous pairing is replaced: its chat gets nothing more, and its buttons stop working")
    out("pair your phone now: open this link on it and tap Start")
    out(f"    https://t.me/{username}?start={code}")
    out(f"or send this to @{username}:  /pair {code}")
    out(f"the code works once, for one private chat, until {_when(expires)} (15 minutes)")
    if not wait:
        out("the phone service completes the pairing when the code arrives (python3 -m foreman.phone)")
        return 0
    deadline = expires if timeout is None else min(expires, now + float(timeout))
    svc = Phone(cfg, clock=clock, sleep=sleep, ffmpeg=None)
    try:
        if svc._lead():
            out("waiting for the code (Ctrl-C stops waiting; the code stays valid until it expires)")
            while clock() <= deadline:
                svc.reload()
                if svc.paired:
                    break
                if not pairing_open(svc.d, clock()):
                    break
                delay = svc.guard(svc.poll_pairing)
                if delay:
                    sleep(min(delay, max(0.0, deadline - clock()), 5.0))
        else:
            out("the phone service is running: it completes the pairing; waiting for it")
            while clock() <= deadline:
                d = store.load()
                if paired(d) or not pairing_open(d, clock()):
                    break
                sleep(1.0)
    except KeyboardInterrupt:
        out(f"stopped waiting; the code stays valid until {_when(expires)} and the phone service can complete it")
        return 1
    finally:
        svc.close()
    d = store.load()
    if paired(d):
        out(f"paired: one private chat (user {_mask(d['user_id'])}) receives drafts and pages from now on")
        out("next: install the phone service (scripts/install-launchagent.sh phone) and try `phone test`")
        return 0
    why = (d or {}).get("burned") or "the code expired before it was used"
    raise PhoneError(f"not paired: {why}; run `phone setup` again")


def cli_status(cfg, out=print) -> int:
    store = SecretStore(cfg.home)
    pc = cfg.phone_config()
    d = store.load()
    now = time.time()
    if not d or not d.get("token"):
        out("token            not configured: run `python3 -m foreman.cli phone setup`")
    else:
        out(f"token            configured for @{clean(d.get('bot_username'), 64)} (bot id {d.get('bot_id')})")
    try:
        mode_f = os.stat(store.path).st_mode & 0o777 if store.exists() else None
        mode_d = os.stat(store.dir).st_mode & 0o777 if os.path.isdir(store.dir) else None
    except OSError:
        mode_f = mode_d = None
    if mode_f is not None:
        ok = mode_f == 0o600 and mode_d == 0o700
        out(f"secrets          {store.path} (file {mode_f:04o}, directory {mode_d:04o})" + ("" if ok else "  WARNING: expected 0600 / 0700"))
    if paired(d):
        out(f"paired           yes, since {_when(d.get('paired_wall'))} (user {_mask(d.get('user_id'))}, private chat)")
    elif pairing_open(d, now):
        out(f"paired           no; a pairing code is open until {_when(d['pairing']['expires_wall'])}")
    else:
        out("paired           no" + (f" (last code: {clean(d.get('burned'), 80)})" if d and d.get("burned") else ""))
    lock = InstanceLock(os.path.join(cfg.home, "phone.lock"))
    running = not lock.acquire()
    lock.release()
    db = Store(cfg.db_path)
    try:
        st = dict(db.conn.execute("SELECT * FROM phone_state WHERE id = 1").fetchone())
        counts = {(r["kind"], r["state"]): r["n"] for r in db.conn.execute(
            "SELECT kind, state, COUNT(*) AS n FROM phone_messages GROUP BY kind, state")}
        waiting = len(ledger.review_queue(db))
    finally:
        db.close()
    beat = st.get("last_poll_wall")
    out(f"service          {'running' if running else 'not running'}"
        + (f"; last pass {now - beat:.0f} s ago" if beat else "")
        + ("" if running else "  (scripts/install-launchagent.sh phone)"))

    def n(kind, *states):
        return sum(counts.get((kind, s), 0) for s in states)
    out(f"drafts           {n('draft', 'sent', 'closed')} sent, {n('draft', 'pending', 'sending')} to send, "
        f"{n('draft', 'failed')} failed; {waiting} awaiting a verdict")
    out(f"pages            {n('page', 'sent', 'closed')} sent, {n('page', 'pending', 'sending')} to send")
    out(f"updates offset   {st.get('update_offset')}" + ("; the chat blocked the bot" if st.get("blocked") else ""))
    if st.get("last_error"):
        out(f"last error       {clean(st['last_error'], 200)}")
    out(f"api              {pc.api_base}; digest {'on' if pc.digest else 'off'}; pages {'on' if pc.pages else 'off'}")
    return 0


def _paired_service(cfg) -> Phone:
    svc = Phone(cfg, lead=False)
    svc.reload()
    if svc.api is None:
        svc.close()
        raise PhoneError("no bot token: run `python3 -m foreman.cli phone setup` first")
    if not svc.paired:
        svc.close()
        raise PhoneError("not paired yet: run `python3 -m foreman.cli phone setup` and tap the link")
    return svc


def cli_test(cfg, out=print) -> int:
    svc = _paired_service(cfg)
    try:
        try:
            msg = svc._say("Foreman test message: phone approvals are set up. Nothing to do.")
        except TelegramError as e:
            raise PhoneError(f"the test message was not sent: {e}")
    finally:
        svc.close()
    out(f"test message sent (message {msg.get('message_id')})")
    return 0


def cli_resend(cfg, out=print) -> int:
    svc = _paired_service(cfg)
    try:
        res = svc.resend_pending()
    finally:
        svc.close()
    out(f"sent again: {res['drafts']} draft(s), {res['pages']} page(s)"
        + (f"; {res['failed']} failed (see `phone status`)" if res["failed"] else ""))
    return 1 if res["failed"] else 0


def cli_unpair(cfg, keep_token: bool = False, out=print) -> int:
    store = SecretStore(cfg.home)
    d = store.load()
    if not d:
        out("nothing to unpair: no phone is configured")
        return 0
    if paired(d):
        try:
            BotApi(d["token"], cfg.phone_config().api_base, 10.0).send_message(
                d["chat_id"], "This chat is no longer paired with Foreman.")
        except TelegramError:
            pass  # best effort: the chat is unpaired either way
    if keep_token:
        store.update(lambda x: dict(x, chat_id=None, user_id=None, paired_wall=None, pairing=None,
                                    hmac_key=new_key(), burned=None))
    else:
        store.update(lambda x: None)
    svc = Phone(cfg, lead=False)
    try:
        svc.clear_messages()
        svc._event("phone_unpaired", kept_token=bool(keep_token))
    finally:
        svc.close()
    out("unpaired: no chat is bound, and every button sent before stops working"
        + ("; the token is kept (phone setup pairs again)" if keep_token else "; the token is deleted"))
    out("to revoke the token itself: send /revoke to @BotFather and pick the bot")
    return 0


# ---------------------------------------------------------------- the service's entry point

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m foreman.phone", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config file (default: $FOREMAN_HOME/config.json)")
    ap.add_argument("--db", dest="db_path")
    ap.add_argument("--log-level")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    args = ap.parse_args(argv)
    _install_excepthooks()
    cfg = studio.load_config(args.config, db_path=args.db_path, log_level=args.log_level)
    setup_logging(cfg.log_dir, cfg.log_level)
    svc = Phone(cfg)

    def _stop(signum, _frame):
        LOG.info("signal %d: stopping after this pass", signum)
        svc.stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return svc.run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
