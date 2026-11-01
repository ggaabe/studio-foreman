"""The phone service's settings (`phone_options` in config.json), apart from foreman/phone.py so
that gpud can validate config.json without ever importing the Telegram code.

The bot token is never a setting: it lives in $FOREMAN_HOME/secrets/telegram.json (`phone setup`),
and a key or value here that looks like one is refused when the configuration is loaded.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from .policy import number

API_BASE = "https://api.telegram.org"
TOKEN_RE = re.compile(r"\d{3,20}:[A-Za-z0-9_-]{30,80}")
# Anything token-shaped, also URL-encoded (':' as %3A), wherever it appears in a string.
TOKEN_SCAN = re.compile(r"(?<![0-9])\d{3,20}(?::|%3[Aa])[A-Za-z0-9_-]{30,}")


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
    upload_max_mb: float = 50.0       # the Bot API's upload limit: nothing bigger is ever sent
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
        unknown = sorted(set(data) - set(names))
        if any("token" in str(k).lower() for k in unknown):
            raise ValueError("the bot token never goes in config.json: run `python3 -m foreman.cli phone setup` "
                             "(it is kept in $FOREMAN_HOME/secrets/telegram.json)")
        if unknown:
            raise ValueError(f"unknown phone options {unknown} (known: {sorted(names)})")
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
            elif not isinstance(v, str) or not re.match(r"^https?://[^\s]+$", v):
                raise ValueError(f"phone option {k} must be an http(s) URL, got {v!r}")
        if "digest_hour" in data and not 0 <= data["digest_hour"] <= 23:
            raise ValueError("phone option digest_hour must be an hour 0..23")
        return dataclasses.replace(cls(), **data)
