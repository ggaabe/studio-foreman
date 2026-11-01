"""The phone service's settings (`phone_options` in config.json), apart from foreman/phone.py so
that gpud can validate config.json without ever importing the Telegram code.

The bot token is never a setting: it lives in $FOREMAN_HOME/secrets/telegram.json (`phone setup`),
and a key or value here that looks like one is refused when the configuration is loaded.
"""
from __future__ import annotations

import dataclasses
import ipaddress
import re
import urllib.parse
from dataclasses import dataclass

from .policy import number

API_BASE = "https://api.telegram.org"
TOKEN_RE = re.compile(r"\d{3,20}:[A-Za-z0-9_-]{30,80}")
# Anything token-shaped, also URL-encoded (':' as %3A), wherever it appears in a string.
TOKEN_SCAN = re.compile(r"(?<![0-9])\d{3,20}(?::|%3[Aa])[A-Za-z0-9_-]{30,}")
BOT_API_UPLOAD_MB = 50.0          # the Bot API's upload limit (api.telegram.org)


def _secretish(text) -> bool:
    """A config key or value that may be a pasted bot token: never echoed back."""
    s = str(text)
    return "token" in s.lower() or bool(TOKEN_SCAN.search(s)) or bool(re.search(r"[:%][A-Za-z0-9_-]{20,}", s))


def _loopback(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_api_base(v) -> str:
    """The Bot API base URL: https, or plain http only to this machine (the fake in tests, a local
    Bot API server). Every call carries the token in its URL path, so http to anywhere else would
    send it over the network in clear."""
    if not isinstance(v, str) or not re.match(r"^https?://[^\s]+$", v) or _secretish(v):
        raise ValueError("phone option api_base must be an https URL (http only for 127.0.0.1 / localhost)")
    parts = urllib.parse.urlsplit(v)
    if not parts.hostname or parts.username is not None or parts.password is not None:
        raise ValueError("phone option api_base must be an https URL with a host and no user name")
    if parts.scheme == "http" and not _loopback(parts.hostname):
        raise ValueError("phone option api_base must use https for a host other than this machine: every call "
                         "carries the bot token in its URL")
    return v


@dataclass(frozen=True)
class PhoneConfig:
    api_base: str = API_BASE          # the Bot API (a local fake in tests)
    poll_timeout_s: float = 20.0      # getUpdates long poll
    request_timeout_s: float = 20.0   # every other call (plus the long poll for getUpdates)
    upload_timeout_s: float = 180.0
    send_interval_s: float = 1.0      # between two sends to the chat (Telegram: about one per second)
    max_sends_per_pass: int = 5
    refresh_s: float = 60.0           # how often buttons are checked against changes made elsewhere (CLI)
    max_refresh_per_pass: int = 10
    note_window_s: float = 1800.0     # a reply adds a note to a verdict given at most this long ago
    video_max_mb: float = 45.0        # sendVideo up to this; a bigger MP4 sends a frame as a document
    upload_max_mb: float = 50.0       # the Bot API's upload limit: nothing bigger is ever sent (at most 50)
    pages: bool = True                # send pages (human_review, engine_lost, SAFE mode, ...)
    digest: bool = False              # a daily summary
    digest_hour: int = 9              # local hour after which the digest goes out
    pairing_ttl_s: float = 900.0
    pairing_max_failures: int = 5
    idle_s: float = 30.0              # not configured, not paired or standing by: look again this often
    backoff_max_s: float = 300.0
    watchdog_s: float = 600.0         # a pass without progress this long: exit and let launchd restart it
    stale_claim_s: float = 1800.0     # a send claimed this long ago by a process that died: send it again
    transcode_timeout_s: float = 120.0
    hint_every_s: float = 600.0       # at most one polite answer to text it does not understand per this
    resend_max: int = 20              # /queue and `phone resend`: at most this many messages

    @classmethod
    def from_dict(cls, data) -> "PhoneConfig":
        data = dict(data or {})
        names = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted((str(k) for k in set(data) - set(names)), key=str)
        # The message lands in gpud's launchd log: a key that may be a pasted token is never echoed.
        if any(_secretish(k) for k in unknown):
            raise ValueError("the bot token never goes in config.json: run `python3 -m foreman.cli phone setup` "
                             "(it is kept in $FOREMAN_HOME/secrets/telegram.json)")
        if unknown:
            shown = [re.sub(r"[^A-Za-z0-9_.-]", "?", k)[:40] for k in unknown[:10]]
            raise ValueError(f"unknown phone options {shown} (known: {sorted(names)})")
        for k, v in data.items():
            if isinstance(v, str) and TOKEN_SCAN.search(v):
                raise ValueError(f"phone option {k} holds what looks like a bot token: tokens go in "
                                 f"$FOREMAN_HOME/secrets/telegram.json (`phone setup`), never in config.json")
            default = names[k].default
            if isinstance(default, bool):
                if not isinstance(v, bool):
                    raise ValueError(f"phone option {k} must be true or false, got {v!r}")
            elif isinstance(default, (int, float)):
                if number(v) is None or v < 0:
                    raise ValueError(f"phone option {k} must be a number >= 0, got {v!r}")
            else:
                check_api_base(v)
        if "digest_hour" in data and not 0 <= data["digest_hour"] <= 23:
            raise ValueError("phone option digest_hour must be an hour 0..23")
        if "upload_max_mb" in data and data["upload_max_mb"] > BOT_API_UPLOAD_MB:
            raise ValueError(f"phone option upload_max_mb must be at most {BOT_API_UPLOAD_MB:g} (the Bot API's limit)")
        return dataclasses.replace(cls(), **data)
