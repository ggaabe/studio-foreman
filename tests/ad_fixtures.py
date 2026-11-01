"""Shared fixtures for the ad pipeline tests: a generic character bible, a generic ad template,
facts, a valid script reply, and gpud on the fake servers. Everything here is generic (an online
service, a mascot character, example.com)."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
import wave

from foreman import media, studio
from foreman.ads import AdsConfig

FFMPEG = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg",
                                                     os.path.expanduser("~/.foreman/bin/ffmpeg")) if os.path.exists(p)), None)
needs_ffmpeg = unittest.skipUnless(FFMPEG, "ffmpeg is not installed here")

FACTS = {"service": "An online service that prepares and files a form for you",
         "price": "$9.50 per form, everything included",
         "how": "Upload a document, check the details, pay, and it is prepared and filed",
         "site": "example.com"}

SCRIPT = {
    "hook": "Need a form filed without leaving your desk?",
    "body": ["Upload your document and check the details in a minute.",
             "We prepare it and file it for you.",
             "It costs nine dollars and fifty cents with everything included."],
    "cta": "Get it done today at example.com.",
    "shots": [{"kind": "talk", "lines": [0], "visual": "a bright home office, soft window light"},
              {"kind": "broll", "lines": [1], "visual": "the mascot taps a phone and smiles, slow push-in"},
              {"kind": "screen", "lines": [2], "capture": "captures/upload.png"},
              {"kind": "broll", "lines": [3], "visual": "the mascot holds a finished form, gentle handheld camera"},
              {"kind": "talk", "lines": [4], "visual": "the same office, medium close-up"},
              {"kind": "endcard", "lines": []}],
    "disclosure": "AI-generated",
}

REVIEW_OK = json.dumps({"verdict": "accept", "facts_ok": True, "character_ok": True, "problems": [],
                        "note": "facts and character match"})


def png(path: str, w: int = 48, h: int = 64, shade: int = 90) -> str:
    pix = bytes(((x * 5 + shade) % 256 if c == 0 else (y * 3 + shade) % 256 if c == 1 else shade)
                for y in range(h) for x in range(w) for c in range(3))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(media.encode_png(w, h, pix))
    return path


def wav(path: str, seconds: float = 1.0, rate: int = 16000, freq: float = 220.0) -> str:
    import math
    n = int(seconds * rate)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)))
    return path


def make_home(root: str, clone: bool = False) -> str:
    """$FOREMAN_HOME with a mascot bible (approved hero/talk/turnaround stills), a capture, facts
    and an ad template."""
    home = os.path.join(root, "home")
    cdir = os.path.join(home, "characters", "mascot-a")
    for name, shade in (("hero", 90), ("talk", 120), ("turnaround", 60), ("endcard", 150), ("unapproved", 30)):
        png(os.path.join(cdir, "stills", f"{name}.png"), shade=shade)
    voice = {"engine": "kokoro", "voice": "am_onyx,bm_george", "speed": 1.0}
    if clone:
        wav(os.path.join(cdir, "voice", "reference.wav"), 3.0)
        voice = {"engine": "qwen3_clone", "ref_wav": "voice/reference.wav", "language": "English"}
    bible = {"id": "mascot-a", "name": "Mascot A", "kind": "mascot", "subject": "the mascot",
             "look": "a friendly cartoon-like mascot character in a blue cap and a yellow vest",
             "keep": ["blue cap", "yellow vest"], "style": "natural light", "voice": voice,
             "stills": [{"file": "stills/hero.png", "tags": ["hero"], "approved": True},
                        {"file": "stills/talk.png", "tags": ["talk"], "approved": True},
                        {"file": "stills/turnaround.png", "tags": ["turnaround"], "approved": True},
                        {"file": "stills/endcard.png", "tags": ["endcard"], "approved": True},
                        {"file": "stills/unapproved.png", "tags": ["desk"], "approved": False}]}
    with open(os.path.join(cdir, "bible.json"), "w") as f:
        json.dump(bible, f)
    png(os.path.join(home, "assets", "captures", "upload.png"), 120, 90, 200)
    os.makedirs(os.path.join(home, "assets"), exist_ok=True)
    with open(os.path.join(home, "assets", "facts.json"), "w") as f:
        json.dump(FACTS, f)
    os.makedirs(os.path.join(home, "templates"), exist_ok=True)
    with open(os.path.join(home, "templates", "ad.json"), "w") as f:
        json.dump(template_data(), f)
    return home


def template_data(**over) -> dict:
    t = {
        "id": "ads.service.v1", "kind": "ad", "family": "ad",
        "consumer": ["the owner's ad account (example)"], "revenue_line": "marketing",
        "value_usd": 40, "prior": {"alpha": 2, "beta": 2},
        "caps": {"per_week": 5, "max_stock": 5, "two_week_use": 5},
        "params": {"seconds": {"min": 15, "max": 30, "default": 20}, "shots": {"min": 4, "max": 6, "default": 6}},
        "brief_max_chars": 400, "flags": {"commercial": True}, "review_minutes": 3, "active": False,
        "ad": {"concept": "Getting a form filed from a phone is easy and the price is clear.",
               "character": "mascot-a", "facts": {"file": "assets/facts.json"},
               "cta": "Get it done today at example.com.",
               "end_card": {"price": "$9.50", "url": "example.com", "cta": "Get it done today", "still": "endcard"},
               "disclosure": "AI-generated", "ai_label": True, "captures": ["captures/upload.png"],
               "banned_marks": ["OtherCo"], "music": {"prompt": "warm upbeat acoustic", "bpm": 100},
               "spellings": ["example.com"]},
    }
    t.update(over)
    return t


def ads_config(**over) -> AdsConfig:
    base = dict(models=dict(studio.AD_MODELS), video_width=64, video_height=128, out_width=270, out_height=480,
                talk_seeds=2, max_job_s=600.0, expected={"talk": 60.0, "broll": 60.0, "still": 20.0})
    base.update(over)
    return AdsConfig(**base)


def probe_frame(ffmpeg: str, path: str, t: float, w: int, h: int) -> bytes:
    res = subprocess.run([ffmpeg, "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                          "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True)
    return res.stdout


class ManualClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return 1.7e9 + self.t

    def monotonic(self):
        return self.t


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="foreman-ads-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def wait_for(pred, timeout: float = 10.0, step: float = 0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()
