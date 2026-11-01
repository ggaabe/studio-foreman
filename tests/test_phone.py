"""Phone approvals over Telegram (foreman/phone.py), against a local fake Bot API (tests/fake_telegram.py).

  PH1  setup and pairing: the token (stdin or a hidden prompt) checked with getMe and kept in a
       0600 file; the right code pairs one private chat and one user; a wrong, expired or group
       code does not; a second user is ignored; the CLI waits itself or lets a running service pair
  PH2  each draft and page is sent exactly once, across restarts and a crash mid-send
  PH3  media: MP4 by sendVideo, a frame when it is too big, M4A via ffmpeg or the WAV as a
       document, the size guard, a 400 falling back, pruned files, files outside the artifacts root
  PH4  captions are plain text: control, bidi and invisible characters removed, links defused,
       no parse_mode, Telegram's length limit
  PH5  buttons record through the ledger: approve, reject, used, undo, with their events (by=phone),
       P(ship) and the kill criteria
  PH6  forged, stale, replayed and old-copy buttons, state changed elsewhere, double taps
  PH7  a reply to a draft adds a note to the verdict, within the window
  PH8  commands: /queue /status /pause /resume /help, unknown text answered once
  PH9  everything from another chat or user is ignored without a reply and logged without content
  PH10 429 and network backoff, an offline Bot API, a busy database, offset persistence
  PH11 the token never reaches a log line, an exception, the database, config.json or the CLI's output
  PH12 pages with Resolve, refreshed buttons, the digest, CLI status/test/resend/unpair, the LaunchAgent

Nothing here talks to the real Telegram API or sleeps longer than the fake's short long-poll waits.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import plistlib
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_telegram import FakeTelegram  # noqa: E402

from foreman import cli, ledger, phone, studio  # noqa: E402
from foreman.store import Store  # noqa: E402
from foreman.supervisor import InstanceLock  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = "6123456789" + ":" + "AAHfakeTokenForForemanTests_only-01234"  # fake; split so no scanner takes it for one
OWNER, OTHER = 424242, 515151
CODE = "ABCDEFGH23"

SPEECH = {"id": "t.speech", "kind": "speak", "family": "tts", "consumer": ["weekly demo reel", "team update"],
          "revenue_line": "internal", "value_usd": 20, "prior": {"alpha": 8, "beta": 2},
          "caps": {"per_week": 50, "max_stock": 50, "two_week_use": 50},
          "params": {"voice": {"allowed": ["af_heart", "am_michael"], "default": "af_heart"}},
          "review_minutes": 1, "active": True}
VIDEO = {"id": "t.video", "kind": "render_video", "family": "video", "consumer": "pitch deck for a named prospect",
         "revenue_line": "new_business", "value_usd": 100, "prior": {"alpha": 5, "beta": 5},
         "caps": {"per_week": 50, "max_stock": 50, "two_week_use": 50},
         "params": {"seconds": {"min": 1, "max": 5, "default": 5}, "size": {"allowed": ["768x512"], "default": "768x512"}},
         "review_minutes": 2, "active": True}
VIDEO_FILES = ("output.mp4", "frame_0000.png", "frame_0060.png", "frame_0120.png", "video.json", "audio.wav")
AUDIO_FILES = ("output.wav", "waveform.png")


class Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


class Tty(io.StringIO):
    def isatty(self):
        return True


class PhoneCase(unittest.TestCase):
    options: dict = {}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="foreman-phone-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.home = os.path.join(self.tmp, "home")
        self.art = os.path.join(self.home, "artifacts")
        os.makedirs(os.path.join(self.home, "templates"))
        os.makedirs(self.art)
        self.fake = FakeTelegram(TOKEN)
        self.addCleanup(self.fake.close)
        self.write_config()
        for t in (SPEECH, VIDEO):
            with open(os.path.join(self.home, "templates", t["id"] + ".json"), "w") as f:
                json.dump(t, f)
        self.store = Store(self.cfg.db_path)
        self.addCleanup(self.store.close)
        self.clock = Clock(time.time())
        self.n = 0
        self.log = io.StringIO()   # what a log file would hold (the redacting formatter)
        self.raw: list = []        # the records as logged, before redaction
        handler = logging.StreamHandler(self.log)
        handler.setFormatter(phone.RedactingJsonFormatter())
        raw = self

        class Raw(logging.Handler):
            def emit(self, record):
                raw.raw.append(logging.Formatter().format(record))

        lg = logging.getLogger("foreman.phone")
        self._old_level = lg.level
        lg.setLevel(logging.INFO)
        rawh = Raw()
        lg.addHandler(handler)
        lg.addHandler(rawh)
        self.addCleanup(lg.removeHandler, handler)
        self.addCleanup(lg.removeHandler, rawh)
        self.addCleanup(lg.setLevel, self._old_level)
        patcher = mock.patch.object(phone, "new_code", return_value=CODE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_config(self, **options):
        opts = {"api_base": self.fake.url, "poll_timeout_s": 0, "send_interval_s": 0, "refresh_s": 0,
                "watchdog_s": 0, "request_timeout_s": 5}
        opts.update(self.options)
        opts.update(options)
        with open(os.path.join(self.home, "config.json"), "w") as f:
            json.dump({"ffmpeg": "", "phone_options": opts}, f)
        self.cfg = studio.load_config(env={"FOREMAN_HOME": self.home})

    # ---- helpers
    def svc(self, ffmpeg=None, **kw) -> phone.Phone:
        p = phone.Phone(self.cfg, clock=self.clock, sleep=lambda s: None, ffmpeg=ffmpeg, **kw)
        self.addCleanup(p.close)
        return p

    def cli(self, *args, stdin: str | None = None) -> tuple:
        out = io.StringIO()
        old_home, old_stdin = os.environ.get("FOREMAN_HOME"), sys.stdin
        os.environ["FOREMAN_HOME"] = self.home
        sys.stdin = io.StringIO(stdin if stdin is not None else "")
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = cli.main(list(args))
        finally:
            sys.stdin = old_stdin
            if old_home is None:
                os.environ.pop("FOREMAN_HOME", None)
            else:
                os.environ["FOREMAN_HOME"] = old_home
        return code, out.getvalue()

    def setup(self, token=TOKEN) -> str:
        code, out = self.cli("phone", "setup", "--no-wait", stdin=token + "\n")
        self.assertEqual(code, 0, out)
        return out

    def paired_svc(self, **kw) -> phone.Phone:
        self.setup()
        s = self.svc(**kw)
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        self.assertEqual(s.step(), 0.0)
        self.assertTrue(s.paired, self.log.getvalue())
        return s

    def item(self, files=VIDEO_FILES, template="t.video", kind="render_video", state="review", sizes=None,
             brief="a red fox trotting through fresh snow", brain_note="clean motion, the fox stays in frame",
             brain_verdict="accept", run_dir=None, **cols) -> dict:
        self.n += 1
        run = run_dir or os.path.join(self.art, f"run-{self.n}")
        os.makedirs(run, exist_ok=True)
        paths = []
        for name in files:
            p = name if os.path.isabs(name) else os.path.join(run, name)
            if not os.path.isabs(name):
                with open(p, "wb") as f:
                    if name == "video.json":
                        f.write(json.dumps({"width": 768, "height": 512, "fps": 24, "frame_count": 121}).encode())
                    else:
                        f.write((name.encode() + b"\x00\r\n") * ((sizes or {}).get(name, 64) // (len(name) + 3) + 1))
            paths.append(p)
        now = time.time()
        with self.store.tx():
            jid, _ = self.store.insert_job(kind=kind, requires="m", state="succeeded", payload={}, max_attempts=1,
                                           created_at=now, idempotency_key=f"phone-test:{self.n}")
            self.store.update_job(jid, root_id=jid)
            iid = ledger.insert_item(self.store, template_id=template, consumer="pitch deck for a named prospect",
                                     brief=brief, content_hash=f"h{self.n}", job_id=jid, state=state, created_at=now,
                                     produced_at=now, review_minutes=2.0, value_usd=100.0, brain_verdict=brain_verdict,
                                     brain_note=brain_note, artifacts=paths, artifact_dirs=[run], **cols)
        return ledger.item(self.store, iid)

    def page(self, kind="safe_mode", reason="SAFE mode: the brain-load ladder is exhausted", parent=None) -> int:
        self.n += 1
        with self.store.tx():
            jid, _ = self.store.insert_job(kind=kind, requires="human", state="awaiting_human",
                                           payload={"reason": reason, "source_job_id": parent}, parent_id=parent,
                                           max_attempts=1, created_at=time.time(), idempotency_key=f"page-test:{self.n}")
        return jid

    def row(self, kind, ref) -> dict:
        r = self.store.conn.execute("SELECT * FROM phone_messages WHERE kind = ? AND item = ?", (kind, ref)).fetchone()
        return dict(r) if r else None

    def msg(self, kind, ref) -> dict:
        return self.fake.messages[self.row(kind, ref)["message_id"]]

    def data(self, kind, ref, label) -> str:
        return dict(self.fake.buttons(self.msg(kind, ref)["markup"]))[label]

    def press(self, s, kind, ref, label, user=OWNER, chat=OWNER, data=..., message_id=None) -> dict:
        data = data if data is not ... else self.data(kind, ref, label)
        mid = message_id if message_id is not None else self.row(kind, ref)["message_id"]
        n = len(self.fake.calls("answerCallbackQuery"))
        self.fake.callback(chat, user, data, mid)
        s.poll()
        answers = self.fake.calls("answerCallbackQuery")[n:]
        return answers[-1]["params"] if answers else {}

    def say(self, s, text, user=OWNER, chat=OWNER, chat_type="private", reply_to=None) -> list:
        n = len(self.fake.sends())
        self.fake.message(chat, user, text, chat_type=chat_type, reply_to=reply_to)
        s.poll()
        return self.fake.sends()[n:]

    def events(self, kind) -> list:
        return self.store.events(kind)

    def secrets(self) -> dict:
        with open(os.path.join(self.home, "secrets", "telegram.json")) as f:
            return json.load(f)


# ---------------------------------------------------------------- PH1 setup and pairing

class PH1SetupAndPairing(PhoneCase):
    def test_setup_checks_the_token_keeps_it_0600_and_the_right_code_pairs(self):
        out = self.setup()
        self.assertIn(f"https://t.me/foreman_test_bot?start={CODE}", out)
        self.assertIn(f"/pair {CODE}", out)
        self.assertNotIn(TOKEN, out)
        self.assertEqual(self.fake.calls()[0]["method"], "getMe")
        sdir = os.path.join(self.home, "secrets")
        self.assertEqual(stat.S_IMODE(os.stat(sdir).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(sdir, "telegram.json")).st_mode), 0o600)
        d = self.secrets()
        self.assertEqual(d["token"], TOKEN)
        self.assertIsNone(d["chat_id"])
        self.assertNotIn(CODE, json.dumps(d), "only an HMAC of the code is kept")
        with open(os.path.join(self.home, "config.json")) as f:
            self.assertNotIn(TOKEN, f.read())
        s = self.svc()
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        self.assertEqual(s.step(), 0.0)
        d = self.secrets()
        self.assertEqual((d["chat_id"], d["user_id"]), (OWNER, OWNER))
        self.assertIsNone(d["pairing"])
        confirm = self.fake.last("sendMessage")["params"]
        self.assertEqual(confirm["chat_id"], OWNER)
        self.assertIn("Paired with Foreman", confirm["text"])
        self.assertEqual(len(self.events("phone_paired")), 1)
        # the code is used up: nobody else can pair with it
        self.fake.message(OTHER, OTHER, f"/start {CODE}")
        s.step()
        self.assertEqual(self.secrets()["user_id"], OWNER)
        self.assertEqual([r for r in self.fake.sends() if r["params"]["chat_id"] in (OTHER, str(OTHER))], [])

    def test_pair_command_and_bot_mention_work_too(self):
        self.setup()
        s = self.svc()
        self.fake.message(OWNER, OWNER, f"/pair@foreman_test_bot {CODE.lower()}")
        s.step()
        self.assertTrue(s.paired)

    def test_wrong_codes_count_and_burn_the_code(self):
        self.setup()
        s = self.svc()
        for k in range(4):
            self.fake.message(OTHER, OTHER, f"/start WRONGCODE{k}")
        s.step()
        self.assertFalse(s.paired)
        self.assertEqual(self.secrets()["pairing"]["failures"], 4)
        self.fake.message(OTHER, OTHER, "/start WRONGCODE9")
        self.fake.message(OWNER, OWNER, f"/start {CODE}")  # too late: the fifth wrong code burned it
        s.step()
        self.assertFalse(s.paired)
        self.assertIsNone(self.secrets()["pairing"])
        self.assertIn("wrong codes", self.secrets()["burned"])
        self.assertEqual(self.fake.sends(), [], "no reply to a stranger, nor to anyone while unpaired")

    def test_an_expired_code_pairs_nobody(self):
        self.setup()
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        self.clock.t += 16 * 60
        s = self.svc()
        self.assertEqual(s.step(), s.pc.idle_s, "no open code: the service does not even read updates")
        self.assertFalse(s.paired)
        self.assertEqual(self.fake.calls("getUpdates"), [])
        # a pairing that reads the message after expiry burns the code instead of pairing
        s2 = self.svc()
        s2.reload()
        self.clock.t -= 16 * 60
        s2.api.get_updates(0, 0)
        self.clock.t += 16 * 60
        self.assertFalse(s2.pair_update(self.fake.updates[0]))
        self.assertEqual(self.secrets()["burned"], "the code expired")
        code, out = self.cli("phone", "status")
        self.assertIn("paired           no (last code: the code expired)", out)

    def test_a_group_chat_is_refused_and_burns_the_code(self):
        self.setup()
        s = self.svc()
        self.fake.message(-100777, OWNER, f"/start {CODE}", chat_type="group")
        s.step()
        self.assertFalse(s.paired)
        self.assertIn("group", self.secrets()["burned"])
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        s.step()
        self.assertFalse(s.paired, "a code seen in a group pairs nobody")
        self.assertEqual(self.fake.sends(), [])

    def test_a_second_user_is_ignored_after_pairing(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        approve = self.data("draft", it["id"], "Approve")
        before = len(self.fake.sends())
        for text in ("/status", "/pause", f"/start {CODE}", "hello there"):
            self.fake.message(OTHER, OTHER, text)
        self.fake.callback(OTHER, OTHER, approve, self.row("draft", it["id"])["message_id"])
        s.step()
        self.assertEqual(len(self.fake.sends()), before)
        self.assertEqual(self.fake.calls("answerCallbackQuery"), [])
        self.assertEqual(ledger.item(self.store, it["id"])["state"], "review")
        self.assertFalse(ledger.planner_state(self.store)["paused"])
        self.assertEqual(self.secrets()["user_id"], OWNER)

    def test_bad_token_is_refused_and_nothing_is_saved(self):
        code, out = self.cli("phone", "setup", "--no-wait", stdin="6123456789:" + "Z" * 35 + "\n")
        self.assertEqual(code, 1)
        self.assertIn("Telegram rejected that token", out)
        self.assertNotIn("Z" * 35, out)
        self.assertFalse(os.path.exists(os.path.join(self.home, "secrets", "telegram.json")))
        code, out = self.cli("phone", "setup", "--no-wait", stdin="not a token at all\n")
        self.assertEqual(code, 1)
        self.assertIn("does not look like a bot token", out)
        self.assertNotIn("not a token at all", out)

    def test_hidden_prompt_when_stdin_is_a_terminal(self):
        with mock.patch.object(phone.getpass, "getpass", return_value=TOKEN) as gp:
            self.assertEqual(phone.read_token(Tty()), TOKEN)
        gp.assert_called_once()
        self.assertNotIn(TOKEN, gp.call_args[0][0])

    def test_cli_waits_and_pairs_by_itself(self):
        self.write_config(poll_timeout_s=1)
        lines, result = [], {}
        t = threading.Thread(target=lambda: result.setdefault("code", phone.cli_setup(
            self.cfg, wait=True, timeout=20, out=lines.append, stdin=io.StringIO(TOKEN + "\n"))))
        t.start()
        for _ in range(200):
            if any("waiting for the code" in x for x in lines):
                break
            time.sleep(0.02)
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        t.join(20)
        self.assertEqual(result.get("code"), 0, lines)
        self.assertTrue(any(x.startswith("paired: one private chat") for x in lines), lines)
        self.assertEqual(self.secrets()["chat_id"], OWNER)
        self.assertGreater(phone.Phone(self.cfg, ffmpeg=None).offset(), 0)

    def test_cli_lets_a_running_service_pair(self):
        s = self.svc()
        self.assertTrue(s._lead(), "the service holds phone.lock")
        lines, result = [], {}
        t = threading.Thread(target=lambda: result.setdefault("code", phone.cli_setup(
            self.cfg, wait=True, timeout=20, out=lines.append, stdin=io.StringIO(TOKEN + "\n"),
            sleep=lambda x: time.sleep(0.02))))
        t.start()
        for _ in range(200):
            if any("the phone service is running" in x for x in lines):
                break
            time.sleep(0.02)
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        s.step()
        t.join(20)
        self.assertEqual(result.get("code"), 0, lines)
        self.assertTrue(s.paired)
        self.assertEqual(self.fake.calls("getUpdates")[0]["params"]["offset"], 0)
        self.assertEqual(len(self.fake.calls("getUpdates")), 1, "only the service read updates")


# ---------------------------------------------------------------- PH2 exactly once

class PH2ExactlyOnce(PhoneCase):
    def test_each_draft_and_page_once_across_restarts(self):
        s = self.paired_svc()
        a, b = self.item(), self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        pg = self.page()
        s.step()
        sends = [r for r in self.fake.sends() if "Paired" not in str(r["params"].get("text"))]
        self.assertEqual([r["method"] for r in sends], ["sendVideo", "sendDocument", "sendMessage"])
        s.step()
        s.close()
        s2 = self.svc()  # a restarted service
        s2.step()
        s2.step()
        self.assertEqual(len(self.fake.sends()), 4, "nothing is sent twice")
        c = self.item()
        s2.step()
        self.assertEqual(len(self.fake.sends()), 5)
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM phone_messages").fetchone()[0], 4)
        for kind, ref in (("draft", a["id"]), ("draft", b["id"]), ("page", pg), ("draft", c["id"])):
            r = self.row(kind, ref)
            self.assertEqual((r["state"], r["issue"]), ("sent", 1))
            self.assertIsInstance(r["message_id"], int)

    def test_a_send_cut_short_by_a_crash_is_sent_again_once(self):
        s = self.paired_svc()
        it = self.item()
        with self.store.tx():  # a process claimed it and died before recording the answer
            self.store.conn.execute("INSERT INTO phone_messages (item, kind, state, claim, claimed_at, created_at) "
                                    "VALUES (?, 'draft', 'sending', 'dead', ?, ?)", (it["id"], self.clock.t, self.clock.t))
        s.step()
        self.assertEqual(len(self.fake.calls("sendVideo")), 0, "a fresh claim is someone else's send in flight")
        self.clock.t += s.pc.stale_claim_s + 1
        s.step()
        s.step()
        self.assertEqual(len(self.fake.calls("sendVideo")), 1)
        self.assertEqual(self.row("draft", it["id"])["state"], "sent")
        self.assertIn("sending it again", self.log.getvalue())

    def test_more_than_a_pass_goes_out_over_several_passes(self):
        self.write_config(max_sends_per_pass=2)
        s = self.paired_svc()
        for _ in range(5):
            self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        s.step()
        self.assertEqual(len(self.fake.calls("sendDocument")), 2)
        self.assertEqual(self.fake.last("getUpdates")["params"]["timeout"], 0, "more to send: no long poll")
        s.step()
        s.step()
        self.assertEqual(len(self.fake.calls("sendDocument")), 5)


# ---------------------------------------------------------------- PH3 media

class PH3Media(PhoneCase):
    def test_mp4_goes_out_by_sendvideo_with_its_size_and_buttons(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        r = self.fake.last("sendVideo")
        p, f = r["params"], r["files"]["video"]
        with open(os.path.join(self.art, "run-1", "output.mp4"), "rb") as fh:
            self.assertEqual(f["data"], fh.read())
        self.assertEqual((f["filename"], f["content_type"]), ("output.mp4", "video/mp4"))
        self.assertEqual((p["chat_id"], p["supports_streaming"], p["width"], p["height"], p["duration"]),
                         (str(OWNER), "true", "768", "512", "5"))
        self.assertNotIn("parse_mode", p)
        self.assertEqual([t for t, _ in self.fake.buttons(p["reply_markup"])], ["Approve", "Reject"])
        cap = p["caption"]
        for want in (f"job {it['job_id']}", "t.video", "Brief: a red fox trotting through fresh snow",
                     "Brain: accept: clean motion, the fox stays in frame", "Review: about 2 min", "Video: MP4"):
            self.assertIn(want, cap)

    def test_an_mp4_over_the_limit_sends_a_frame_as_a_document(self):
        self.write_config(video_max_mb=0.0005)
        s = self.paired_svc()
        self.item(sizes={"output.mp4": 4000})
        s.step()
        self.assertEqual(self.fake.calls("sendVideo"), [])
        r = self.fake.last("sendDocument")
        self.assertEqual(r["files"]["document"]["filename"], "frame_0060.png", "the middle sampled frame")
        self.assertIn("over the 0.0005 MB phone limit", r["params"]["caption"])
        self.assertIn("one sampled frame", r["params"]["caption"])

    def test_a_refused_video_falls_back_to_a_document_then_to_text(self):
        s = self.paired_svc()
        a = self.item()
        self.fake.fail("sendVideo", 400, "Bad Request: wrong file identifier/HTTP URL specified")
        s.step()
        self.assertEqual(self.fake.last("sendDocument")["files"]["document"]["filename"], "output.mp4")
        self.assertEqual(self.row("draft", a["id"])["media"], "document")
        b = self.item()
        self.fake.fail("sendVideo", 400, "Bad Request: IMAGE_PROCESS_FAILED")
        self.fake.fail("sendDocument", 413, "Request Entity Too Large", times=2)
        s.step()
        last = self.fake.last("sendMessage")["params"]
        self.assertIn("Telegram refused them", last["text"])
        self.assertEqual([t for t, _ in self.fake.buttons(last["reply_markup"])], ["Approve", "Reject"])
        self.assertEqual(self.row("draft", b["id"])["media"], "text")

    def test_speech_becomes_m4a_with_ffmpeg(self):
        ff = os.path.join(self.tmp, "ffmpeg")
        with open(ff, "w") as f:
            f.write('#!/bin/sh\nfor last; do :; done\nprintf "fake-m4a-bytes" > "$last"\n')
        os.chmod(ff, 0o755)
        s = self.paired_svc(ffmpeg=ff)
        self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        s.step()
        r = self.fake.last("sendAudio")
        self.assertEqual(r["files"]["audio"]["data"], b"fake-m4a-bytes")
        self.assertTrue(r["files"]["audio"]["filename"].endswith(".m4a"))
        self.assertEqual(r["files"]["audio"]["content_type"], "audio/mp4")
        self.assertIn("M4A for the phone", r["params"]["caption"])
        self.assertEqual(os.listdir(os.path.join(self.home, "phone", "tmp")), [], "the M4A is deleted once sent")

    def test_speech_without_ffmpeg_or_with_a_failing_one_is_the_wav_as_a_document(self):
        bad = os.path.join(self.tmp, "ffmpeg-bad")
        with open(bad, "w") as f:
            f.write("#!/bin/sh\nexit 1\n")
        os.chmod(bad, 0o755)
        for ff in (None, bad):
            with self.subTest(ffmpeg=ff):
                self.fake.requests.clear()
                s = self.paired_svc(ffmpeg=ff) if ff is None else self.svc(ffmpeg=ff)
                self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
                s.step()
                r = self.fake.last("sendDocument")
                self.assertEqual(r["files"]["document"]["filename"], "output.wav")
                self.assertEqual(self.fake.calls("sendAudio"), [])
                if ff is None:
                    self.assertIn("no ffmpeg for an M4A", r["params"]["caption"])
                s.close()

    def test_nothing_over_the_upload_limit_is_sent(self):
        self.write_config(upload_max_mb=0.001)
        s = self.paired_svc()
        self.item(files=AUDIO_FILES, template="t.speech", kind="speak", sizes={"output.wav": 5000, "waveform.png": 50})
        s.step()
        r = self.fake.last("sendDocument")
        self.assertEqual(r["files"]["document"]["filename"], "waveform.png")
        self.assertIn("over Telegram's 0.001 MB limit", r["params"]["caption"])
        self.item(files=("output.wav",), template="t.speech", kind="speak", sizes={"output.wav": 5000})
        s.step()
        self.assertIn("over Telegram's 0.001 MB limit", self.fake.last("sendMessage")["params"]["text"])
        for req in self.fake.sends():
            for f in req["files"].values():
                self.assertLessEqual(len(f["data"]), 1000)

    def test_pruned_files_send_the_text_with_a_note(self):
        s = self.paired_svc()
        it = self.item()
        shutil.rmtree(os.path.join(self.art, "run-1"))
        s.step()
        r = self.fake.last("sendMessage")["params"]
        self.assertIn("no longer on the Studio", r["text"])
        self.assertIn(f"job {it['job_id']}", r["text"])
        self.assertEqual(self.row("draft", it["id"])["media"], "text")

    def test_files_outside_the_artifacts_root_are_never_uploaded(self):
        secret = os.path.join(self.tmp, "id_rsa.png")
        with open(secret, "wb") as f:
            f.write(b"PRIVATE KEY MATERIAL")
        link = os.path.join(self.art, "run-x", "frame_0001.png")
        os.makedirs(os.path.dirname(link))
        os.symlink(secret, link)
        hard = os.path.join(self.art, "run-x", "output.mp4")
        os.link(secret, hard)
        s = self.paired_svc()
        self.item(files=(secret, link, hard, os.path.join(self.art, "..", "config.json")))
        s.step()
        for req in self.fake.requests:
            for f in req["files"].values():
                self.assertNotIn(b"PRIVATE KEY", f["data"])
        self.assertIn("refused (outside the artifacts directory)", self.fake.last("sendMessage")["params"]["text"])


    def test_a_private_template_keeps_its_work_off_telegram(self):
        private = dict(SPEECH, id="t.private", flags={"private": True})
        with open(os.path.join(self.home, "templates", "t.private.json"), "w") as f:
            json.dump(private, f)
        s = self.paired_svc()
        a = self.item(template="t.private", brief="the client's confidential script", brain_note="secret note")
        b = self.item(template="t.video", flags={"private": True}, brief="flagged when it was admitted")
        s.step()
        for it in (a, b):
            m = self.msg("draft", it["id"])
            self.assertEqual(m["kind"], "text")
            self.assertIn(f"job {it['job_id']} (private)", m["text"])
            self.assertNotIn(it["brief"], m["text"])
            self.assertNotIn("secret note", m["text"])
            self.assertEqual([t for t, _ in self.fake.buttons(m["markup"])], ["Approve", "Reject"])
        self.assertEqual([r for r in self.fake.requests if r["files"]], [], "no file left the Mac")
        self.press(s, "draft", a["id"], "Approve")
        self.assertEqual(ledger.item(self.store, a["id"])["state"], "accepted")
        with self.store.tx():  # a page inside that lineage: its reason quotes the work
            pid, _ = self.store.insert_job(kind="human_review", requires="human", state="awaiting_human",
                                           payload={"reason": "the brain asked: the confidential script is off"},
                                           root_id=a["job_id"], parent_id=a["job_id"], max_attempts=1,
                                           created_at=time.time(), idempotency_key="private-page")
        s.step()
        text = self.msg("page", pid)["text"]
        self.assertNotIn("confidential", text)
        self.assertIn("flagged private", text)


# ---------------------------------------------------------------- PH4 captions

class PH4Captions(PhoneCase):
    def test_untrusted_text_is_plain_clean_and_short(self):
        s = self.paired_svc()
        brief = ("Fox \x1b]52;c;ZXZpbA==\x07 <b>bold</b> *md* [x](y) \u202eevil\u202c zero\u200bwidth\r\nline two "
                 + "long " * 600)
        it = self.item(brief=brief, brain_note="looks fine, see https://evil.example/x and \x1b[2J")
        s.step()
        p = self.fake.last("sendVideo")["params"]
        cap = p["caption"]
        for bad in ("\x1b", "\x07", "\u202e", "\u202c", "\u200b", "\r", "https://", "evil.example", "ZXZpbA==",
                    "[2J"):
            self.assertNotIn(bad, cap)
        self.assertIn("<b>bold</b> *md* [x](y)", cap, "shown as written: no parse_mode")
        self.assertIn("[link removed]", cap)
        self.assertNotIn("parse_mode", p)
        self.assertLessEqual(phone.utf16_len(cap), phone.CAPTION_MAX)
        self.assertIn("\u2026", cap)
        self.assertIn(f"job {it['job_id']}", cap)
        page = self.page(kind="human_review", reason="the brain asked for human review: \x1b[31mred\x1b[0m "
                                                      "visit www.example.com now")
        s.step()
        text = self.msg("page", page)["text"]
        self.assertNotIn("\x1b", text)
        self.assertNotIn("www.example.com", text)

    def test_clean_and_fit(self):
        self.assertEqual(phone.clean("a\x00b\x1fc\u2028d\u00a0e", 50), "a b c d e")
        self.assertEqual(phone.clean("x \x1b]52;c;ZXZpbA==\x07 \x1b[2J\x1b[31mred\x1b[0m \x1bc \x1b(B y", 80), "x red y")
        self.assertEqual(phone.utf16_len(phone.fit("\U0001F98A" * 600, 1024)), 1023)
        self.assertEqual(phone.clean("\ud800 lone", 20), "? lone")


# ---------------------------------------------------------------- PH5 buttons through the ledger

class PH5Buttons(PhoneCase):
    def test_approve_used_reject_undo_record_through_the_ledger(self):
        s = self.paired_svc()
        a = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        b = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        s.step()
        ans = self.press(s, "draft", a["id"], "Approve")
        self.assertEqual(ledger.item(self.store, a["id"])["state"], "accepted")
        self.assertIn("Approved. P(ship) for t.speech is now 0.82", ans["text"])
        ev = self.events("verdict_recorded")
        self.assertEqual((len(ev), ev[0]["data"]["verdict"], ev[0]["data"]["by"], ev[0]["job_id"]),
                         (1, "accept", "phone", a["job_id"]))
        self.assertEqual(ev[0]["data"]["minutes"], 2.0, "the item's review minutes, as the CLI default")
        m = self.msg("draft", a["id"])
        self.assertEqual([t for t, _ in self.fake.buttons(m["markup"])], ["Used it", "Undo"])
        self.assertIn("Approved. Tap Used it", m["text"])
        self.assertEqual(self.row("draft", a["id"])["issue"], 2)
        ans = self.press(s, "draft", a["id"], "Used it")
        self.assertEqual(ledger.item(self.store, a["id"])["state"], "shipped")
        self.assertEqual(self.events("item_shipped")[0]["data"]["by"], "phone")
        self.assertEqual(self.fake.buttons(self.msg("draft", a["id"])["markup"]), [])
        self.assertEqual(self.row("draft", a["id"])["state"], "closed")
        ans = self.press(s, "draft", b["id"], "Reject")
        self.assertIn("Rejected. P(ship) for t.speech is now 0.75", ans["text"])
        self.assertEqual([t for t, _ in self.fake.buttons(self.msg("draft", b["id"])["markup"])], ["Undo"])
        ans = self.press(s, "draft", b["id"], "Undo")
        self.assertIn("back in the review queue", ans["text"])
        self.assertIn("0.82", ans["text"])
        self.assertEqual(ledger.item(self.store, b["id"])["state"], "review")
        ret = self.events("verdict_retracted")
        self.assertEqual((ret[0]["data"]["retracted"], ret[0]["data"]["by"]), ("reject", "phone"))
        m = self.msg("draft", b["id"])
        self.assertEqual([t for t, _ in self.fake.buttons(m["markup"])], ["Approve", "Reject"])
        self.assertIn("Undone: back in the review queue.", m["text"])
        self.press(s, "draft", b["id"], "Approve")
        code, out = self.cli("templates")  # the CLI's view agrees: 10 / 12 after two accepts
        self.assertIn("0.83", out)

    def test_kill_criteria_run_after_a_phone_verdict(self):
        s = self.paired_svc()
        for k in range(19):
            it = self.item(files=(), template="t.speech", kind="speak")
            ledger.record_verdict(self.store, it["job_id"], "reject", None, None, time.time())
        last = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        s.step()
        self.press(s, "draft", last["id"], "Reject")
        self.assertTrue(ledger.template_states(self.store)["t.speech"]["paused"])
        self.assertEqual(ledger.template_states(self.store)["t.speech"]["reason"], "kill_low_acceptance")
        self.assertIn("was paused by a kill criterion: kill_low_acceptance", self.fake.last("sendMessage")["params"]["text"])

    def test_verdicts_work_while_gpud_holds_its_lock_and_the_phone_never_takes_it(self):
        gpud = InstanceLock(self.cfg.db_path + ".lock")
        self.assertTrue(gpud.acquire())
        self.addCleanup(gpud.release)
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.press(s, "draft", it["id"], "Approve")
        self.assertEqual(ledger.item(self.store, it["id"])["state"], "accepted")
        gpud.release()  # gpud restarts
        self.assertTrue(gpud.acquire(), "the phone never held gpud's lock")
        self.press(s, "draft", it["id"], "Undo")
        self.assertEqual(ledger.item(self.store, it["id"])["state"], "review")


# ---------------------------------------------------------------- PH6 stale, forged, replayed, double taps

class PH6StaleAndForged(PhoneCase):
    def setUp(self):
        super().setUp()
        self.s = self.paired_svc()
        self.it = self.item()
        self.s.step()

    def verdicts(self) -> int:
        return len(self.events("verdict_recorded"))

    def test_forged_and_foreign_data_record_nothing(self):
        good = self.data("draft", self.it["id"], "Approve")
        forged = good[:-1] + ("A" if good[-1] != "A" else "B")
        other_key = phone.make_data(phone.new_key(), OWNER, "a", "d", self.it["id"], 1)
        other_chat = phone.make_data(self.s.key, OTHER, "a", "d", self.it["id"], 1)
        wrong_kind = good.replace("ad", "sd", 1)
        for data in (forged, other_key, other_chat, wrong_kind, "garbage", "x" * 300, "", None):
            with self.subTest(data=data):
                ans = self.press(self.s, "draft", self.it["id"], None, data=data)
                self.assertIn("not valid", ans.get("text", ""))
        self.assertEqual(self.verdicts(), 0)
        self.assertEqual(ledger.item(self.store, self.it["id"])["state"], "review")
        self.assertEqual([t for t, _ in self.fake.buttons(self.msg("draft", self.it["id"])["markup"])],
                         ["Approve", "Reject"], "its own message keeps valid buttons")
        self.assertIn("not signed by this pairing", self.log.getvalue())

    def test_a_replayed_or_stale_button_never_records_a_second_verdict(self):
        approve = self.data("draft", self.it["id"], "Approve")
        reject = self.data("draft", self.it["id"], "Reject")
        self.press(self.s, "draft", self.it["id"], "Approve")
        self.press(self.s, "draft", self.it["id"], "Undo")
        for data in (approve, reject):
            ans = self.press(self.s, "draft", self.it["id"], None, data=data)
            self.assertIn("out of date", ans["text"])
        self.assertEqual(self.verdicts(), 1)
        self.assertEqual(len(self.events("verdict_retracted")), 1)
        self.assertEqual(ledger.item(self.store, self.it["id"])["state"], "review")
        self.assertEqual([t for t, _ in self.fake.buttons(self.msg("draft", self.it["id"])["markup"])],
                         ["Approve", "Reject"])

    def test_double_tap_is_idempotent(self):
        approve = self.data("draft", self.it["id"], "Approve")
        mid = self.row("draft", self.it["id"])["message_id"]
        self.fake.callback(OWNER, OWNER, approve, mid)
        self.fake.callback(OWNER, OWNER, approve, mid)
        self.s.poll()
        self.assertEqual(self.verdicts(), 1)
        answers = [a["params"]["text"] for a in self.fake.calls("answerCallbackQuery")]
        self.assertEqual(len(answers), 2)
        self.assertTrue(answers[0].startswith("Approved."))
        self.assertEqual(answers[1], "Already done: approved.")

    def test_a_verdict_given_elsewhere_refreshes_instead_of_recording_again(self):
        code, out = self.cli("verdict", str(self.it["job_id"]), "accept")
        self.assertEqual(code, 0, out)
        ans = self.press(self.s, "draft", self.it["id"], "Approve")
        self.assertIn("changed meanwhile: it is approved now", ans["text"])
        self.assertEqual(self.verdicts(), 1)
        self.assertEqual([t for t, _ in self.fake.buttons(self.msg("draft", self.it["id"])["markup"])],
                         ["Used it", "Undo"])
        # and the refresh pass catches such changes by itself
        code, out = self.cli("verdict", str(self.it["job_id"]), "retract")
        self.s.step()
        self.assertEqual([t for t, _ in self.fake.buttons(self.msg("draft", self.it["id"])["markup"])],
                         ["Approve", "Reject"])

    def test_pruned_expired_or_gone_items_explain_and_record_nothing(self):
        self.press(self.s, "draft", self.it["id"], "Reject")
        with self.store.tx():
            ledger.update_item(self.store, self.it["id"], pruned_at=time.time())
        ans = self.press(self.s, "draft", self.it["id"], "Undo")
        self.assertIn("rejected and pruned", ans["text"])
        self.assertEqual(self.events("verdict_retracted"), [])
        self.assertEqual(self.fake.buttons(self.msg("draft", self.it["id"])["markup"]), [])
        b = self.item()
        self.s.step()
        approve = self.data("draft", b["id"], "Approve")
        with self.store.tx():
            ledger.update_item(self.store, b["id"], state="expired")
        ans = self.press(self.s, "draft", b["id"], None, data=approve)
        self.assertIn("expired", ans["text"])
        self.assertEqual(self.verdicts(), 1)
        self.assertIn("Expired", self.msg("draft", b["id"])["text"])

    def test_an_older_copy_after_queue_is_stale(self):
        old_mid = self.row("draft", self.it["id"])["message_id"]
        approve = self.data("draft", self.it["id"], "Approve")
        self.say(self.s, "/queue")
        new = self.row("draft", self.it["id"])
        self.assertNotEqual(new["message_id"], old_mid)
        self.assertEqual(json.loads(new["old_ids"]), [old_mid])
        self.assertEqual(self.fake.buttons(self.fake.messages[old_mid]["markup"]), [], "the old copy lost its buttons")
        ans = self.press(self.s, "draft", self.it["id"], None, data=approve, message_id=old_mid)
        self.assertIn("older copy", ans["text"])
        self.assertEqual(self.verdicts(), 0)
        self.press(self.s, "draft", self.it["id"], "Approve")
        self.assertEqual(self.verdicts(), 1)

    def test_a_press_from_another_user_in_the_paired_chat_is_ignored(self):
        approve = self.data("draft", self.it["id"], "Approve")
        self.press(self.s, "draft", self.it["id"], None, data=approve, user=OTHER)
        self.press(self.s, "draft", self.it["id"], None, data=approve, chat=OTHER, user=OWNER)
        self.assertEqual(self.fake.calls("answerCallbackQuery"), [])
        self.assertEqual(self.verdicts(), 0)


# ---------------------------------------------------------------- PH7 notes

class PH7Notes(PhoneCase):
    def test_a_reply_before_the_verdict_goes_with_it(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        mid = self.row("draft", it["id"])["message_id"]
        out = self.say(s, "the fox \x1b[2J looks great", reply_to=mid)
        self.assertIn("goes with your verdict", out[-1]["params"]["text"])
        self.assertEqual(out[-1]["params"]["reply_parameters"]["message_id"], self.fake.next_message_id - 1)
        self.press(s, "draft", it["id"], "Approve")
        self.assertEqual(ledger.item(self.store, it["id"])["verdict_note"], "the fox looks great")
        self.assertIsNone(self.row("draft", it["id"])["note"])
        self.assertNotIn("looks great", self.log.getvalue(), "the note is not logged")

    def test_a_reply_after_the_verdict_within_the_window_and_after_it(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        mid = self.row("draft", it["id"])["message_id"]
        self.press(s, "draft", it["id"], "Reject")
        out = self.say(s, "too dark", reply_to=mid)
        self.assertIn("Note added", out[-1]["params"]["text"])
        self.assertEqual(ledger.item(self.store, it["id"])["verdict_note"], "too dark")
        self.assertEqual(self.events("verdict_noted")[0]["data"]["by"], "phone")
        with self.store.tx():
            ledger.update_item(self.store, it["id"], verdict_at=time.time() - 3 * 3600)
        out = self.say(s, "and too short", reply_to=mid)
        self.assertIn("older than 30 minutes", out[-1]["params"]["text"])
        self.assertEqual(ledger.item(self.store, it["id"])["verdict_note"], "too dark")
        page = self.page()
        s.step()
        out = self.say(s, "noted", reply_to=self.row("page", page)["message_id"])
        self.assertIn("only be added to drafts", out[-1]["params"]["text"])


# ---------------------------------------------------------------- PH8 commands

class PH8Commands(PhoneCase):
    def test_status_pause_resume_help_queue(self):
        s = self.paired_svc()
        self.item()
        s.step()
        text = self.say(s, "/status")[-1]["params"]["text"]
        for want in ("Foreman status", "gpud: boot", "planner: active", "review queue: 1 draft(s), about 2 min",
                     "open pages: 0", "t.video 1/50 this week, stock 1/50"):
            self.assertIn(want, text)
        self.assertIn("Planner paused", self.say(s, "/pause")[-1]["params"]["text"])
        self.assertTrue(ledger.planner_state(self.store)["paused"])
        self.assertEqual(self.events("planner_paused")[0]["data"]["note"], "from the phone")
        self.assertIn("planner: paused", self.say(s, "/status")[-1]["params"]["text"])
        self.assertIn("Planner resumed (active)", self.say(s, "/resume@foreman_test_bot")[-1]["params"]["text"])
        self.assertFalse(ledger.planner_state(self.store)["paused"])
        self.assertIn("/queue", self.say(s, "/help")[-1]["params"]["text"])
        out = self.say(s, "/queue")
        self.assertEqual([r["method"] for r in out], ["sendVideo", "sendMessage"])
        self.assertIn("Sent again: 1 draft(s), 0 page(s).", out[-1]["params"]["text"])
        self.assertEqual(self.say(s, "/status@some_other_bot"), [], "a command for another bot")

    def test_queue_with_nothing_waiting_and_unknown_text(self):
        s = self.paired_svc()
        self.assertIn("Nothing is waiting", self.say(s, "/queue")[-1]["params"]["text"])
        first = self.say(s, "what is this?")
        self.assertIn("only understand a few commands", first[-1]["params"]["text"])
        self.assertEqual(self.say(s, "still there?"), [], "one polite answer per 10 minutes")
        self.assertEqual(self.say(s, "/frobnicate"), [])
        self.clock.t += 601
        self.assertEqual(len(self.say(s, "hello?")), 1)


# ---------------------------------------------------------------- PH9 unauthorized updates

class PH9Unauthorized(PhoneCase):
    def test_other_chats_users_and_groups_are_ignored_and_not_logged(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        before = len(self.fake.sends())
        self.fake.message(OTHER, OTHER, "/status secret-text-from-a-stranger")
        self.fake.message(-100555, OWNER, "/pause group-secret", chat_type="group")
        self.fake.message(-100556, OWNER, "/pause", chat_type="supergroup")
        self.fake.message(OWNER, OWNER, "/pause", is_bot=True)
        self.fake.callback(-100555, OWNER, self.data("draft", it["id"], "Approve"),
                           self.row("draft", it["id"])["message_id"], chat_type="group")
        self.fake.push({"edited_message": {"message_id": 1, "chat": {"id": OWNER, "type": "private"},
                                           "from": {"id": OWNER}, "text": "/pause"}})
        self.fake.push({"channel_post": {"message_id": 2, "chat": {"id": -100, "type": "channel"}, "text": "x"}})
        s.step()
        self.assertEqual(len(self.fake.sends()), before)
        self.assertEqual(self.fake.calls("answerCallbackQuery"), [])
        self.assertFalse(ledger.planner_state(self.store)["paused"])
        self.assertEqual(ledger.item(self.store, it["id"])["state"], "review")
        logs = self.log.getvalue() + "\n".join(self.raw)
        for secret in ("secret-text-from-a-stranger", "group-secret", str(OTHER), "-100555"):
            self.assertNotIn(secret, logs)
        self.assertIn("ignored a message from a chat or user that is not paired (private chat)", logs)
        self.assertIn("(group chat)", logs)
        self.assertIn("ignored an update of another type", logs)


# ---------------------------------------------------------------- PH10 backoff, offline, offset

class PH10Robustness(PhoneCase):
    def test_429_honours_retry_after(self):
        s = self.paired_svc()
        pg = self.page()
        self.fake.fail("sendMessage", 429, "Too Many Requests: retry after 7", retry_after=7)
        self.assertEqual(s.step(), 7.0)
        r = self.row("page", pg)
        self.assertEqual(r["state"], "pending")
        self.assertAlmostEqual(r["next_try_at"], self.clock.t + 7, delta=0.01)
        n = len(self.fake.calls())
        s.step()
        self.assertEqual([c["method"] for c in self.fake.calls()[n:]], ["getUpdates"], "quiet until retry_after")
        self.clock.t += 8
        s.step()
        self.assertEqual(self.row("page", pg)["state"], "sent")
        self.fake.fail("getUpdates", 429, "Too Many Requests: retry after 3", retry_after=3)
        self.assertEqual(s.step(), 3.0)

    def test_network_errors_back_off_exponentially_and_recover(self):
        s = self.paired_svc()
        it = self.item()
        self.fake.offline = True
        delays = [s.step() for _ in range(5)]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 8.0, 16.0])
        self.assertEqual(self.row("draft", it["id"])["state"], "pending")
        self.assertIn("network", phone.Phone(self.cfg, ffmpeg=None).db().conn.execute(
            "SELECT last_error FROM phone_state").fetchone()[0])
        self.fake.offline = False
        self.clock.t += 600
        self.assertEqual(s.step(), 0.0)
        self.assertEqual(self.row("draft", it["id"])["state"], "sent")
        self.assertEqual(s._fails, 0)

    def test_an_unreachable_api_and_a_revoked_token_never_crash(self):
        self.write_config(api_base="http://127.0.0.1:9")  # nothing listens on the discard port
        s = self.paired_svc_offline()
        self.assertGreater(s.step(), 0)
        s.close()
        self.write_config()
        s2 = self.svc()
        self.fake.token = "6123456789:" + "R" * 35  # revoked: the fake answers 401 to ours
        self.assertEqual(s2.step(), s2.pc.backoff_max_s)
        self.assertIn("Telegram rejected the bot token", self.log.getvalue())

    def paired_svc_offline(self):
        os.makedirs(os.path.join(self.home, "secrets"), mode=0o700)
        st = phone.SecretStore(self.home)
        st.update(lambda d: {"token": TOKEN, "bot_username": "foreman_test_bot", "hmac_key": phone.new_key(),
                             "chat_id": OWNER, "user_id": OWNER})
        self.item()
        return self.svc()

    def test_offsets_persist_and_a_poison_update_does_not_block_the_rest(self):
        s = self.paired_svc()
        ids = [self.fake.message(OWNER, OWNER, "/help") for _ in range(3)]
        with mock.patch.object(phone.Phone, "on_command", side_effect=[RuntimeError("boom"), None, None]):
            s.poll()
        self.assertEqual(s.offset(), ids[-1] + 1)
        self.assertIn("raised RuntimeError; skipped", self.log.getvalue())
        s.close()
        s2 = self.svc()
        s2.step()
        self.assertEqual(self.fake.last("getUpdates")["params"]["offset"], ids[-1] + 1)

    def test_a_busy_database_retries_the_update_instead_of_skipping_it(self):
        s = self.paired_svc()
        uid = self.fake.message(OWNER, OWNER, "/help")
        calls = {"n": 0}
        real = phone.Phone.on_command

        def flaky(self_, text, mid):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return real(self_, text, mid)
        with mock.patch.object(phone.Phone, "on_command", flaky):
            self.assertEqual(s.step(), 1.0)
            self.assertEqual(s.offset(), uid, "not advanced: the update is tried again")
            self.assertEqual(s.step(), 0.0)
        self.assertEqual(s.offset(), uid + 1)
        self.assertEqual(calls["n"], 2)

    def test_run_once_and_a_raising_pass_are_contained(self):
        s = self.paired_svc()
        with mock.patch.object(phone.Phone, "send_outbox", side_effect=ValueError("unexpected")):
            self.assertEqual(s.step(), 1.0)
        self.assertEqual(s.run(once=True), 0)

    def test_a_broken_outbox_never_stops_button_presses(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.fake.callback(OWNER, OWNER, self.data("draft", it["id"], "Approve"), self.row("draft", it["id"])["message_id"])
        with mock.patch.object(phone.Phone, "send_outbox", side_effect=KeyError("a bug")):
            self.assertEqual(s.step(), 1.0)
        self.assertEqual(ledger.item(self.store, it["id"])["state"], "accepted", "the press was read anyway")

    def test_a_failed_edit_is_drawn_again_by_the_refresh_pass(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.fake.fail("editMessageCaption", 502, "Bad Gateway")
        self.press(s, "draft", it["id"], "Approve")
        self.assertEqual(ledger.item(self.store, it["id"])["state"], "accepted")
        self.assertEqual([t for t, _ in self.fake.buttons(self.msg("draft", it["id"])["markup"])], ["Approve", "Reject"])
        self.assertIsNone(self.row("draft", it["id"])["shown"])
        s.step()
        self.assertEqual([t for t, _ in self.fake.buttons(self.msg("draft", it["id"])["markup"])], ["Used it", "Undo"])

    def test_watchdog_exits_on_a_hung_pass_and_uploads_keep_it_quiet(self):
        self.write_config(watchdog_s=0.2)
        s = self.paired_svc()
        exits = []

        def fake_exit(code):
            exits.append(code)
            s.stop.set()
        s._watch_every = 0.01
        beats = []
        api = phone.BotApi(TOKEN, self.fake.url)
        api.progress = lambda: beats.append(1)
        big = os.path.join(self.art, "big.mp4")
        with open(big, "wb") as f:
            f.write(b"\0" * (2 * (1 << 20) + 5))
        api.send_file("sendDocument", "document", OWNER, big, "c")
        self.assertEqual(len(beats), 3, "one heartbeat per MB sent")
        s._step_started = time.monotonic()
        s._touch()
        self.assertGreater(s._step_started, time.monotonic() - 0.1)
        with mock.patch.object(phone.os, "_exit", fake_exit), mock.patch.object(phone.logging, "shutdown"):
            s._step_started = time.monotonic() - 5
            t = threading.Thread(target=s.watchdog)
            t.start()
            t.join(5)
        self.assertEqual(exits, [70])
        self.assertIn("exiting so launchd restarts it", self.log.getvalue())


# ---------------------------------------------------------------- PH11 the token stays secret

class PH11TokenSecrecy(PhoneCase):
    def test_token_never_in_logs_exceptions_db_config_or_output(self):
        outputs = [self.setup()]
        s = self.svc()
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        s.step()
        self.item()
        self.page()
        self.fake.fail("sendVideo", 400, f"Bad Request: echoing /bot{TOKEN}/sendVideo back")
        self.fake.fail("sendMessage", 429, "slow down", retry_after=1)
        s.step()
        self.clock.t += 5
        self.fake.offline = True
        s.step()
        self.fake.offline = False
        with mock.patch.object(phone.Phone, "on_command", side_effect=ValueError(f"boom {TOKEN}")):
            self.fake.message(OWNER, OWNER, "/help")
            s.step()
        for args in (("phone", "status"), ("phone", "test"), ("phone", "resend")):
            outputs.append(self.cli(*args)[1])
        errors = []
        for api in (phone.BotApi(TOKEN, "http://127.0.0.1:9"), phone.BotApi(TOKEN, self.fake.url + "/nope"),
                    phone.BotApi(TOKEN, "http://127.0.0.1:99999"), phone.BotApi(TOKEN[:-1] + "\n", self.fake.url),
                    phone.BotApi(TOKEN, "ftp://127.0.0.1")):
            try:
                api.get_me()
            except phone.TelegramError as e:
                errors.append(e)
                self.assertIsNone(e.__context__)
                self.assertIsNone(e.__cause__)
        self.assertEqual(len(errors), 5)
        outputs.append(self.cli("phone", "unpair")[1])
        texts = {"log": self.log.getvalue(), "outputs": "\n".join(outputs),
                 "errors": "\n".join(f"{e} {e!r} {e.description}" for e in errors)}
        with open(self.cfg.db_path, "rb") as f:
            texts["db"] = f.read().decode("latin-1")
        for suffix in ("-wal", "-shm"):
            if os.path.exists(self.cfg.db_path + suffix):
                with open(self.cfg.db_path + suffix, "rb") as f:
                    texts["db" + suffix] = f.read().decode("latin-1")
        with open(os.path.join(self.home, "config.json")) as f:
            texts["config"] = f.read()
        for where, text in texts.items():
            with self.subTest(where=where):
                self.assertNotIn(TOKEN, text)
                self.assertNotIn(TOKEN.split(":")[1], text)
        self.assertIn("<redacted", texts["log"], "the echo and the exception were redacted, not dropped")
        self.assertFalse(os.path.exists(os.path.join(self.home, "secrets", "telegram.json")))

    def test_redact_catches_encoded_and_unknown_tokens(self):
        other = "987654321:" + "Q" * 35
        self.assertNotIn("Q" * 35, phone.redact(f"url /bot{other}/getMe and {other.replace(':', '%3A')}"))
        phone.register_secret(TOKEN)
        self.assertEqual(phone.redact(f"x{TOKEN}y"), "x<redacted>y")

    def test_config_refuses_a_token(self):
        with open(os.path.join(self.home, "config.json"), "w") as f:
            json.dump({"phone_options": {"bot_token": TOKEN}}, f)
        with self.assertRaises(ValueError) as cm:
            studio.load_config(env={"FOREMAN_HOME": self.home})
        self.assertIn("never goes in config.json", str(cm.exception))
        with open(os.path.join(self.home, "config.json"), "w") as f:
            json.dump({"phone_options": {"api_base": f"https://api.telegram.org/bot{TOKEN}"}}, f)
        with self.assertRaises(ValueError) as cm:
            studio.load_config(env={"FOREMAN_HOME": self.home})
        self.assertNotIn(TOKEN, str(cm.exception))
        with open(os.path.join(self.home, "config.json"), "w") as f:
            json.dump({"telegram_token": TOKEN}, f)
        with self.assertRaises(ValueError):
            studio.load_config(env={"FOREMAN_HOME": self.home})

    def test_loose_permissions_are_tightened(self):
        self.setup()
        path = os.path.join(self.home, "secrets", "telegram.json")
        os.chmod(path, 0o644)
        os.chmod(os.path.dirname(path), 0o755)
        phone.SecretStore(self.home).update(lambda d: d)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode), 0o700)


# ---------------------------------------------------------------- PH12 pages, digest, CLI, LaunchAgent

class PH12PagesDigestCli(PhoneCase):
    def test_pages_once_with_resolve_and_refresh_when_resolved_elsewhere(self):
        s = self.paired_svc()
        safe = self.page()
        lost = self.page(kind="engine_lost", reason="engine mlx-serve lost")
        s.step()
        m = self.msg("page", safe)
        self.assertIn(f"Foreman page {safe}: SAFE mode", m["text"])
        self.assertIn("clear-safe", m["text"])
        self.assertEqual([t for t, _ in self.fake.buttons(m["markup"])], ["Resolve"])
        ans = self.press(s, "page", safe, "Resolve")
        self.assertIn("Resolved. SAFE mode itself still needs clear-safe", ans["text"])
        self.assertEqual(self.store.job(safe).state, "resolved")
        ev = self.events("page_resolved")
        self.assertEqual((ev[0]["job_id"], ev[0]["data"]["by"]), (safe, "phone"))
        self.assertEqual(self.fake.buttons(self.msg("page", safe)["markup"]), [])
        code, out = self.cli("resolve", str(lost), "--note", "fixed")
        self.assertEqual(code, 0, out)
        s.step()
        self.assertEqual(self.fake.buttons(self.msg("page", lost)["markup"]), [])
        self.assertIn("Resolved.", self.msg("page", lost)["text"])
        s.step()
        self.assertEqual(len([r for r in self.fake.sends() if "Foreman page" in str(r["params"].get("text"))]), 2)

    def test_a_brain_page_about_a_draft_rides_on_the_draft(self):
        s = self.paired_svc()
        with self.store.tx():
            rj, _ = self.store.insert_job(kind="review_render", requires="brain", state="succeeded", payload={},
                                          max_attempts=1, created_at=time.time(), idempotency_key="rv")
        it = self.item(brain_verdict="human_review", review_job_id=rj)
        pg = self.page(kind="human_review", reason="the brain asked for human review: the voice", parent=rj)
        s.step()
        self.assertIsNone(self.row("page", pg))
        self.press(s, "draft", it["id"], "Approve")
        self.assertEqual(self.store.job(pg).state, "resolved", "the verdict resolved it, as on the CLI")

    def test_pages_can_be_turned_off(self):
        self.write_config(pages=False)
        s = self.paired_svc()
        self.page()
        s.step()
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM phone_messages").fetchone()[0], 0)

    def test_digest_is_off_by_default_and_once_a_day_when_on(self):
        def digests():
            return [r for r in self.fake.sends() if "daily summary" in str(r["params"].get("text"))]
        s = self.paired_svc()
        s.step()
        s.close()
        self.assertEqual(digests(), [])
        self.write_config(digest=True, digest_hour=0)
        self.item()
        for _ in range(2):  # restarts do not send it again
            s = self.svc()
            s.step()
            s.step()
            s.close()
        self.assertEqual(len(digests()), 1)
        self.assertIn("Waiting for you: 1 draft(s)", digests()[0]["params"]["text"])
        self.clock.t += 86400
        s = self.svc()
        s.step()
        self.assertEqual(len(digests()), 2)

    def test_cli_status_test_resend_unpair(self):
        code, out = self.cli("phone", "status")
        self.assertIn("not configured", out)
        self.assertIn("service          not running", out)
        code, out = self.cli("phone", "test")
        self.assertEqual(code, 1)
        self.assertIn("no bot token", out)
        s = self.paired_svc()
        it = self.item()
        s.step()
        code, out = self.cli("phone", "status")
        for want in ("configured for @foreman_test_bot", "paired           yes", "(file 0600, directory 0700)",
                     "drafts           1 sent, 0 to send, 0 failed; 1 awaiting a verdict", "service          running"):
            self.assertIn(want, out)
        self.assertNotIn(str(OWNER), out, "the user id is masked")
        code, out = self.cli("phone", "test")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.fake.last("sendMessage")["params"]["text"][:21], "Foreman test message:")
        code, out = self.cli("phone", "resend")
        self.assertIn("sent again: 1 draft(s), 0 page(s)", out)
        approve = self.data("draft", it["id"], "Approve")
        code, out = self.cli("phone", "unpair", "--keep-token")
        self.assertEqual(code, 0, out)
        self.assertIn("no longer paired", self.fake.last("sendMessage")["params"]["text"])
        d = self.secrets()
        self.assertEqual((d["token"], d["chat_id"]), (TOKEN, None))
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM phone_messages").fetchone()[0], 0)
        self.assertEqual(len(self.events("phone_unpaired")), 1)
        self.assertEqual(s.step(), s.pc.idle_s, "unpaired: the service idles")
        # pair again: the old buttons were signed with a key that is gone
        s.close()
        s2 = self.paired_svc()
        s2.step()
        self.fake.callback(OWNER, OWNER, approve, self.row("draft", it["id"])["message_id"])
        s2.poll()
        self.assertEqual(self.events("verdict_recorded"), [])
        self.assertIn("not valid", self.fake.last("answerCallbackQuery")["params"]["text"])
        code, out = self.cli("phone", "unpair")
        self.assertFalse(os.path.exists(os.path.join(self.home, "secrets", "telegram.json")))
        self.assertIn("the token is deleted", out)
        self.assertIn("/revoke", out)

    def test_launchagent_template_for_the_phone(self):
        script = os.path.join(REPO, "scripts", "install-launchagent.sh")
        res = subprocess.run(["bash", script, "phone", "--dry-run"], capture_output=True, timeout=20,
                             env={**os.environ, "FOREMAN_PYTHON": "/usr/bin/python3"})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertNotIn(b"__", res.stdout.split(b"<plist", 1)[1])
        plist = plistlib.loads(res.stdout)
        self.assertEqual(plist["Label"], "com.github.ggaabe.foreman.phone")
        self.assertEqual(plist["ProgramArguments"], ["/usr/bin/python3", "-m", "foreman.phone"])
        self.assertEqual((plist["KeepAlive"], plist["ThrottleInterval"], plist["ProcessType"]), (True, 30, "Background"))
        self.assertTrue(plist["StandardErrorPath"].endswith("/.foreman/logs/phone.stderr.log"))
        self.assertEqual(sorted(plist["EnvironmentVariables"]), ["FOREMAN_HOME", "PYTHONPATH", "PYTHONUNBUFFERED"])
        gpud = plistlib.loads(subprocess.run(["bash", script, "--dry-run"], capture_output=True, timeout=20).stdout)
        self.assertEqual(gpud["Label"], "com.github.ggaabe.foreman.gpud")
        bad = subprocess.run(["bash", script, "bogus", "--dry-run"], capture_output=True, timeout=20)
        self.assertEqual(bad.returncode, 2)
        un = os.path.join(REPO, "scripts", "uninstall-launchagent.sh")
        with open(un) as f:
            text = f.read()
        self.assertIn('gpud|phone) SERVICE="$arg"', text)
        self.assertNotIn("sudo ", text.replace("no sudo", ""))

    def test_the_phone_is_its_own_process_and_gpud_does_not_import_it(self):
        code = "\n".join((
            "import sys, foreman.daemon, foreman.supervisor, foreman.planner, foreman.cli",
            "from foreman import studio",
            "cfg = studio.load_config(env={'FOREMAN_HOME': %r})" % self.home,
            "print('foreman.phone' in sys.modules, cfg.phone_config().api_base == %r)" % self.fake.url))
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30, cwd=REPO)
        self.assertEqual(res.stdout.strip(), b"False True", res.stderr)


if __name__ == "__main__":
    unittest.main()
