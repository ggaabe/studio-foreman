"""Adversarial tests for phone approvals over Telegram (red team). Each test names the invariant it attacks.

  PA1  strangers who learn the bot's username: /start guesses, brute force, groups, edits, forwards, bots
  PA2  callback data: forged, modified, truncated, unicode, overlong, other items, replayed after a full cycle
  PA3  races: two taps, a CLI verdict mid-tap, gpud holding the write lock, state changed elsewhere,
       a kill after the ledger write and before the offset commit
  PA4  exactly once: a kill between Telegram accepting a send and the database recording it, a lost
       response, two services on one machine
  PA5  offsets: poison updates (always busy, raising while pairing), a failure in the middle of a batch
  PA6  the token: config keys, a plaintext API base, reprs, echoed error bodies, the CLI's stderr
  PA7  captions: injection in every untrusted field, forged lines, links Telegram would make tappable,
       Telegram's length limits
  PA8  files: symlinked directories, sibling prefixes, FIFOs, a file swapped after the check, the
       size guard, corrupt sidecars, a file pruned mid-send
  PA9  429 storms and network partitions
  PA10 the secrets file: corrupt, unreadable, symlinked, mistyped, loose permissions
  PA11 /pause and /resume abuse
  PA12 reply-to-note abuse

Fake Bot API only (tests/fake_telegram.py, and two small subclasses here); deterministic: crash
points are patched in, time is the test's virtual clock. Nothing here changes foreman/ or test_phone.py.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import logging
import os
import shutil
import socket
import sqlite3
import stat
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_telegram import FakeTelegram  # noqa: E402
from test_phone import AUDIO_FILES, CODE, OTHER, OWNER, SPEECH, TOKEN, PhoneCase  # noqa: E402

from foreman import cli, ledger, phone, studio  # noqa: E402

SECRET_HALF = TOKEN.split(":")[1]
# Shaped like a real token: test_phone.TOKEN contains the word "Token", which config's name check catches.
PLAIN_TOKEN = "7012345678" + ":" + "AAEq9Zx7Lm3Rt5Vb1Nc8Kd2Wf6Hj4Pg0YsQ"  # fake; split so no scanner takes it for one
NEWCHAT = 626262


class Killed(BaseException):
    """kill -9 at a chosen point: nothing in foreman/phone.py catches it, as nothing can catch SIGKILL."""


class ScriptedTelegram(FakeTelegram):
    """The fake Bot API, plus two ways real networks fail: `lose` carries a call out and then drops
    the connection before answering (a partition at the worst moment); `raw` answers a method with
    raw bytes (an HTML error page from a proxy or a load balancer)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lose: list = []
        self.raw: dict = {}

    def _reply(self, h, status, obj):
        method = self.requests[-1]["method"] if self.requests else None
        if status == 200 and method in self.lose:
            self.lose.remove(method)
            h.close_connection = True
            try:
                h.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return
        if method in self.raw:
            code, body = self.raw.pop(method)
            h.send_response(code)
            h.send_header("Content-Type", "text/html")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return
        return super()._reply(h, status, obj)


class Base(PhoneCase):
    # ---- helpers on top of test_phone.PhoneCase
    def use_fake(self, cls):
        self.fake.close()
        self.fake = cls(TOKEN)
        self.addCleanup(self.fake.close)
        self.write_config()

    def push_message(self, chat, user, text, chat_type="private", date=None, **extra) -> int:
        with self.fake.cond:
            self.fake.next_message_id += 1
            mid = self.fake.next_message_id
        msg = {"message_id": mid, "date": int(time.time()) if date is None else int(date), "text": text,
               "chat": {"id": chat, "type": chat_type}, "from": {"id": user, "is_bot": False, "first_name": "X"}}
        msg.update(extra)
        return self.fake.push({"message": msg})

    @contextlib.contextmanager
    def crash_in(self, owner, attr, when=lambda *a, **k: True):
        """Raise Killed the first time owner.attr is called (and `when` agrees)."""
        real = getattr(owner, attr)
        armed = {"on": True}

        def boom(*a, **k):
            if armed["on"] and when(*a, **k):
                armed["on"] = False
                raise Killed(f"kill -9 in {attr}")
            return real(*a, **k)
        with mock.patch.object(owner, attr, boom):
            yield armed
        self.assertFalse(armed["on"], f"the crash point {attr} was never reached")

    def restart(self, s) -> phone.Phone:
        s.close()
        s2 = self.svc()
        s2.reload()
        return s2

    def counts(self) -> tuple:
        return (len(self.events("verdict_recorded")), len(self.events("verdict_retracted")),
                len(self.events("item_shipped")))

    def verdicts(self) -> int:
        return len(self.events("verdict_recorded"))

    def state_of(self, it) -> str:
        return ledger.item(self.store, it["id"])["state"]

    def buttons_of(self, kind, ref) -> list:
        return [t for t, _ in self.fake.buttons(self.msg(kind, ref)["markup"])]

    def outgoing_texts(self) -> list:
        out = []
        for r in self.fake.requests:
            for k in ("text", "caption"):
                v = r["params"].get(k)
                if v:
                    out.append(str(v))
        return out

    def uploads(self) -> list:
        return [f for r in self.fake.requests for f in r["files"].values()]

    def sends_to(self, chat) -> list:
        return [r for r in self.fake.sends() if str(r["params"].get("chat_id")) == str(chat)]

    def secrets_path(self) -> str:
        return os.path.join(self.home, "secrets", "telegram.json")


# ---------------------------------------------------------------- PA1 strangers

class PA1Strangers(Base):
    def test_guesses_get_no_reply_leave_no_trace_and_a_few_do_not_stop_the_owner(self):
        self.setup()
        s = self.svc()
        for text in ("/start", "/start ABCDEFGH22", "/pair zzzzzzzzzz", "/start@foreman_test_bot ABCDEFGH24",
                     "/start@another_bot ABCDEFGH23", "hello", "/help", "/status", "/start " + "Q" * 64):
            self.fake.message(OTHER, OTHER, text)
        s.step()
        self.assertFalse(s.paired)
        self.assertEqual(self.fake.sends(), [], "no oracle: a stranger gets no answer whatever the guess")
        self.assertEqual(self.secrets()["pairing"]["failures"], 4)
        logs = self.log.getvalue() + "\n".join(self.raw)
        for guess in ("ABCDEFGH22", "zzzzzzzzzz", "ABCDEFGH24", "Q" * 64, str(OTHER)):
            self.assertNotIn(guess, logs)
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        s.step()
        self.assertTrue(s.paired)
        self.assertEqual((self.secrets()["chat_id"], self.secrets()["user_id"]), (OWNER, OWNER))
        self.assertEqual(self.sends_to(OTHER), [])

    def test_a_strangers_wrong_guesses_cannot_lock_the_owner_out_of_pairing(self):
        """The code has ~50 bits, so 5 guesses cannot find it, but the global 5-failure burn lets anyone
        who knows the bot's @username cancel every code the owner creates, as fast as they make them."""
        self.setup()
        s = self.svc()
        for k in range(6):
            self.fake.message(OTHER + k, OTHER + k, f"/start WRONG{k:05d}")
        s.step()
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        s.step()
        self.assertTrue(s.paired, "strangers' guesses burned the owner's code: "
                                  f"{self.secrets().get('burned')!r}")

    def test_messages_sent_before_the_code_existed_do_not_count_against_it(self):
        """Telegram keeps undelivered updates for a day: guesses queued before `phone setup` (or the
        owner's taps on an older link) are read first and burn the new code before it is used."""
        old = self.clock.t - 3600
        for k in range(5):
            self.push_message(OTHER, OTHER, f"/start OLDGUESS{k:02d}", date=old)
        self.setup()
        s = self.svc()
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        s.step()
        self.assertTrue(s.paired, "a backlog older than the code burned it: "
                                  f"{self.secrets().get('burned')!r}")

    def test_edits_channel_posts_and_callbacks_carrying_the_code_neither_pair_nor_burn(self):
        self.setup()
        s = self.svc()
        self.fake.push({"edited_message": {"message_id": 1, "date": 0, "text": f"/start {CODE}",
                                           "chat": {"id": OTHER, "type": "private"},
                                           "from": {"id": OTHER, "is_bot": False}}})
        self.fake.push({"channel_post": {"message_id": 2, "date": 0, "text": f"/start {CODE}",
                                         "chat": {"id": -100123, "type": "channel"}}})
        self.fake.callback(OTHER, OTHER, f"/start {CODE}", 1)
        self.fake.push({"my_chat_member": {"chat": {"id": -100124, "type": "group"},
                                           "from": {"id": OTHER, "is_bot": False}}})
        s.step()
        self.assertFalse(s.paired)
        self.assertIsNotNone(self.secrets()["pairing"], "not burned: none of these is a message to the bot")
        self.assertEqual(self.fake.sends(), [])
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        s.step()
        self.assertTrue(s.paired)
        for r in self.fake.calls("getUpdates"):
            self.assertEqual(r["params"]["allowed_updates"], ["message", "callback_query"])

    def test_a_bot_account_cannot_pair_or_burn_the_code(self):
        self.setup()
        s = self.svc()
        self.fake.message(OTHER, OTHER, f"/start {CODE}", is_bot=True)
        s.step()
        self.assertFalse(s.paired)
        self.assertIsNotNone(self.secrets()["pairing"])
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        s.step()
        self.assertTrue(s.paired)
        self.assertEqual(self.secrets()["user_id"], OWNER)

    def test_after_pairing_strangers_cannot_repair_probe_or_even_rewrite_the_secrets(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        before, sig = self.secrets(), phone.SecretStore(self.home).signature()
        n = len(self.fake.sends())
        for text in (f"/start {CODE}", f"/pair {CODE}", "/start GUESS12345", "/queue", "/status", "/pause",
                     "/resume", "/help"):
            self.fake.message(OTHER, OTHER, text)
        owner = {"id": OWNER, "is_bot": False, "first_name": "Owner"}
        self.push_message(OTHER, OTHER, "/pause", forward_from=owner, forward_date=0,
                          forward_origin={"type": "user", "date": 0, "sender_user": owner})
        self.push_message(OTHER, OTHER, "nice", reply_to_message={"message_id": self.row("draft", it["id"])["message_id"]})
        for _ in range(200):  # a flood
            self.fake.message(OTHER + 1, OTHER + 1, "/start AAAAAAAAAA")
        for _ in range(6):
            s.step()
        self.assertEqual(len(self.fake.sends()), n)
        self.assertEqual(self.fake.calls("answerCallbackQuery"), [])
        self.assertEqual(self.secrets(), before)
        self.assertEqual(phone.SecretStore(self.home).signature(), sig, "a stranger's /start never rewrites the file")
        self.assertFalse(ledger.planner_state(self.store)["paused"])
        self.assertIsNone(ledger.item(self.store, it["id"])["verdict_note"])
        self.assertIsNone(self.row("draft", it["id"])["note"])
        self.assertEqual(s._retries, {}, "nothing about strangers is kept in memory")


# ---------------------------------------------------------------- PA2 callback data

class PA2CallbackData(Base):
    def setUp(self):
        super().setUp()
        self.s = self.paired_svc()
        self.a = self.item()
        self.b = self.item()
        self.s.step()

    def untouched(self):
        self.assertEqual(self.counts(), (0, 0, 0))
        self.assertEqual(len(self.events("page_resolved")), 0)
        for it in (self.a, self.b):
            self.assertEqual(self.state_of(it), "review")

    def test_modified_data_never_acts(self):
        good = self.data("draft", self.a["id"], "Approve")
        head, issue, sig = good.split(".")
        ref, b_ref = head[2:], phone._b36(self.b["id"])
        nxt = phone._b36(int(issue, 36) + 1)
        variants = {
            "reject with approve's MAC": f"rd{ref}.{issue}.{sig}",
            "used with approve's MAC": f"ud{ref}.{issue}.{sig}",
            "undo with approve's MAC": f"zd{ref}.{issue}.{sig}",
            "resolve on a draft": f"sd{ref}.{issue}.{sig}",
            "page kind": f"sp{ref}.{issue}.{sig}",
            "another item": f"ad{b_ref}.{issue}.{sig}",
            "the next issue": f"ad{ref}.{nxt}.{sig}",
            "issue zero": f"ad{ref}.0.{sig}",
            "MAC cut to 15": good[:-1],
            "MAC cut to 8 and padded": f"{head}.{issue}.{sig[:8]}{'A' * 8}",
            "MAC with base64 padding": good[:-1] + "=",
            "MAC flipped case": f"{head}.{issue}.{sig.swapcase()}",
            "no MAC": f"{head}.{issue}.",
            "Arabic-Indic digit": f"ad\u0661.{issue}.{sig}",
            "fullwidth action": "\uff41" + good[1:],
            "upper case": good[:2].upper() + good[2:],
            "65 characters": good + "A" * (65 - len(good)),
            "overlong": good + "A" * 400,
            "NUL appended": good + "\x00",
            "leading space": " " + good,
            "zero width inside": good[:3] + "\u200b" + good[3:],
            "another key": phone.make_data(phone.new_key(), OWNER, "a", "d", self.a["id"], int(issue, 36)),
            "the chat id as a string key": phone.make_data(self.s.key, str(OWNER) + " ", "a", "d", self.a["id"],
                                                           int(issue, 36)),
            "an int": 12345, "a list": [good], "an object": {"data": good}, "true": True, "null": None,
        }
        for name, data in variants.items():
            with self.subTest(name):
                ans = self.press(self.s, "draft", self.a["id"], None, data=data)
                self.assertNotIn("Approved", ans.get("text", ""))
                self.assertNotIn("Rejected", ans.get("text", ""))
        self.untouched()
        self.assertEqual(self.buttons_of("draft", self.a["id"]), ["Approve", "Reject"], "its own message keeps working")
        self.press(self.s, "draft", self.a["id"], "Approve")
        self.assertEqual(self.counts(), (1, 0, 0))

    def test_non_canonical_spellings_of_a_genuine_button_do_no_more_than_the_button(self):
        """int(x, 36) accepts leading zeros and '$' accepts a trailing newline, so 'ad01.1.MAC' and
        'ad1.1.MAC\\n' verify. They must do exactly what the genuine button does: one verdict, on
        that item only."""
        good = self.data("draft", self.a["id"], "Approve")
        head, issue, sig = good.split(".")
        for data in (f"{head[:2]}0{head[2:]}.{issue}.{sig}", good + "\n", f"{head}.0{issue}.{sig}", good):
            self.press(self.s, "draft", self.a["id"], None, data=data)
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertEqual(self.state_of(self.a), "accepted")
        self.assertEqual(self.state_of(self.b), "review")

    def test_a_genuine_button_pressed_on_another_drafts_message_records_nothing(self):
        approve_a = self.data("draft", self.a["id"], "Approve")
        ans = self.press(self.s, "draft", self.b["id"], None, data=approve_a)
        self.assertIn("older copy", ans["text"])
        self.untouched()

    def test_every_button_ever_drawn_replayed_after_a_full_cycle_adds_nothing(self):
        seen = []

        def grab():
            seen.extend(d for _, d in self.fake.buttons(self.msg("draft", self.a["id"])["markup"]))
        grab()
        for label in ("Approve", "Undo", "Reject", "Undo", "Approve", "Used it"):
            self.press(self.s, "draft", self.a["id"], label)
            grab()
        self.assertEqual(self.counts(), (3, 2, 1))
        self.assertEqual(self.state_of(self.a), "shipped")
        for data in seen * 2:
            self.press(self.s, "draft", self.a["id"], None, data=data)
        self.assertEqual(self.counts(), (3, 2, 1))
        self.assertEqual(self.state_of(self.a), "shipped")
        self.assertEqual(self.fake.buttons(self.msg("draft", self.a["id"])["markup"]), [])


# ---------------------------------------------------------------- PA3 races

class PA3Races(Base):
    def test_a_cli_verdict_landing_mid_tap_wins_and_the_tap_records_nothing(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        real = cli.apply_verdict

        def racing(cfg, job_id, verdict, *a, **kw):
            if kw.get("by") == "phone":  # the owner, on the Studio, a moment before the phone's write
                real(cfg, job_id, "reject")
            return real(cfg, job_id, verdict, *a, **kw)
        with mock.patch.object(cli, "apply_verdict", racing):
            ans = self.press(s, "draft", it["id"], "Approve")
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertNotIn("by", self.events("verdict_recorded")[0]["data"], "the CLI's verdict, not the phone's")
        self.assertEqual(self.state_of(it), "rejected")
        self.assertIn("not awaiting a verdict", ans["text"])
        self.assertEqual(self.buttons_of("draft", it["id"]), ["Undo"])

    def test_approve_and_reject_tapped_at_once_record_one_verdict(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        approve, reject = self.data("draft", it["id"], "Approve"), self.data("draft", it["id"], "Reject")
        mid = self.row("draft", it["id"])["message_id"]
        for data in (approve, reject, approve, reject):
            self.fake.callback(OWNER, OWNER, data, mid)
        s.poll()
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertEqual(self.state_of(it), "accepted")
        answers = [a["params"]["text"] for a in self.fake.calls("answerCallbackQuery")]
        self.assertEqual(len(answers), 4, "every press is answered")
        self.assertTrue(answers[0].startswith("Approved."))
        self.assertEqual(answers[2], "Already done: approved.")

    def test_a_tap_while_gpud_holds_the_write_lock_waits_and_records_once(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        gpud = sqlite3.connect(self.cfg.db_path, isolation_level=None, check_same_thread=False, timeout=10)
        self.addCleanup(gpud.close)
        gpud.execute("BEGIN IMMEDIATE")  # gpud mid-tick (or restarting) holds the write lock
        timer = threading.Timer(0.4, gpud.execute, ("COMMIT",))
        timer.start()
        self.addCleanup(timer.cancel)
        ans = self.press(s, "draft", it["id"], "Approve")
        timer.join(5)
        self.assertTrue(ans.get("text", "").startswith("Approved."), ans)
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertEqual(self.buttons_of("draft", it["id"]), ["Used it", "Undo"])

    def test_a_database_busy_for_longer_than_the_retries_drops_the_tap_but_never_half_records_it(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.fake.callback(OWNER, OWNER, self.data("draft", it["id"], "Approve"), self.row("draft", it["id"])["message_id"])
        with mock.patch.object(cli, "apply_verdict", side_effect=sqlite3.OperationalError("database is locked")):
            for _ in range(4):
                s.step()
        self.assertEqual(self.counts(), (0, 0, 0))
        self.assertEqual(self.state_of(it), "review")
        self.assertEqual(self.buttons_of("draft", it["id"]), ["Approve", "Reject"], "the buttons still work")
        self.press(s, "draft", it["id"], "Approve")
        self.assertEqual(self.counts(), (1, 0, 0))

    def test_taps_after_the_state_changed_elsewhere_never_act(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.press(s, "draft", it["id"], "Approve")
        used = self.data("draft", it["id"], "Used it")
        self.assertEqual(self.cli("verdict", str(it["job_id"]), "retract")[0], 0)
        ans = self.press(s, "draft", it["id"], None, data=used)
        self.assertIn("changed meanwhile", ans["text"])
        self.assertEqual(self.state_of(it), "review")
        self.assertEqual(self.buttons_of("draft", it["id"]), ["Approve", "Reject"])
        reject = self.data("draft", it["id"], "Reject")
        self.assertEqual(self.cli("verdict", str(it["job_id"]), "accept")[0], 0)
        self.assertEqual(self.cli("ship", str(it["job_id"]))[0], 0)
        ans = self.press(s, "draft", it["id"], None, data=reject)
        self.assertIn("used now", ans["text"])
        self.assertEqual(self.state_of(it), "shipped")
        self.assertEqual(self.counts(), (2, 1, 1))
        self.assertEqual(self.fake.buttons(self.msg("draft", it["id"])["markup"]), [])
        # the item's row deleted outright
        gone = self.item()
        s.step()
        approve = self.data("draft", gone["id"], "Approve")
        with self.store.tx():
            self.store.conn.execute("DELETE FROM plan_items WHERE id = ?", (gone["id"],))
        ans = self.press(s, "draft", gone["id"], None, data=approve)
        self.assertIn("gone", ans["text"])
        self.assertEqual(self.counts(), (2, 1, 1))
        # pruned after a reject: Undo is no longer possible, from any copy
        pr = self.item()
        s.step()
        self.press(s, "draft", pr["id"], "Reject")
        undo = self.data("draft", pr["id"], "Undo")
        with self.store.tx():
            ledger.update_item(self.store, pr["id"], pruned_at=time.time())
        self.press(s, "draft", pr["id"], None, data=undo)
        self.press(s, "draft", pr["id"], None, data=undo)
        self.assertEqual(self.state_of(pr), "rejected")
        self.assertEqual(self.counts(), (3, 1, 1))

    def test_a_kill_after_the_ledger_write_and_before_the_offset_commit_replays_harmlessly(self):
        s = self.paired_svc()
        it = self.item()
        pg = self.page()
        s.step()
        for label, kind, ref in (("Approve", "draft", it["id"]), ("Undo", "draft", it["id"]),
                                 ("Resolve", "page", pg)):
            with self.subTest(label):
                uid = self.fake.callback(OWNER, OWNER, self.data(kind, ref, label), self.row(kind, ref)["message_id"])
                with self.crash_in(phone.Phone, "redraw"), self.assertRaises(Killed):
                    s.poll()
                self.assertEqual(s.offset(), uid, "the offset is committed only after the update is handled")
                s = self.restart(s)
                s.poll()  # Telegram delivers the same update again
                self.assertEqual(s.offset(), uid + 1)
        self.assertEqual(self.counts(), (1, 1, 0))
        self.assertEqual(len(self.events("page_resolved")), 1)
        self.assertEqual(self.state_of(it), "review")
        self.assertEqual(self.buttons_of("draft", it["id"]), ["Approve", "Reject"])

    def test_a_kill_after_a_reply_note_and_before_the_offset_commit_does_not_double_the_note(self):
        s = self.paired_svc()
        before, after = self.item(), self.item()
        s.step()
        # a note before the verdict (kept by the phone until the verdict)
        self.fake.message(OWNER, OWNER, "too dark", reply_to=self.row("draft", before["id"])["message_id"])
        with self.crash_in(phone.Phone, "_say"), self.assertRaises(Killed):
            s.poll()
        s = self.restart(s)
        s.poll()
        self.press(s, "draft", before["id"], "Approve")
        # a note after the verdict (ledger.annotate_verdict)
        self.press(s, "draft", after["id"], "Reject")
        self.fake.message(OWNER, OWNER, "too short", reply_to=self.row("draft", after["id"])["message_id"])
        with self.crash_in(phone.Phone, "_say"), self.assertRaises(Killed):
            s.poll()
        s = self.restart(s)
        s.poll()
        with self.subTest("before the verdict"):
            self.assertEqual(ledger.item(self.store, before["id"])["verdict_note"], "too dark")
        with self.subTest("after the verdict"):
            self.assertEqual(ledger.item(self.store, after["id"])["verdict_note"], "too short")
            self.assertEqual(len(self.events("verdict_noted")), 1)

    def test_a_kill_between_a_new_pairing_and_clearing_the_old_chats_messages(self):
        """The owner moves to a new phone: phone setup again, /start from the new chat. A kill right
        after the secrets file names the new chat (before phone_messages is cleared) must not leave
        the drafts already sent to the old chat marked as sent for good."""
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.assertEqual(self.row("draft", it["id"])["state"], "sent")
        s.close()
        self.setup()
        s = self.svc()
        self.fake.message(NEWCHAT, NEWCHAT, f"/start {CODE}")
        with self.crash_in(phone.Phone, "clear_messages"), self.assertRaises(Killed):
            s.step()
        self.assertEqual(self.secrets()["chat_id"], NEWCHAT)
        s = self.restart(s)
        for _ in range(3):
            s.step()
        to_new = [r for r in self.sends_to(NEWCHAT)
                  if f"job {it['job_id']}" in str(r["params"].get("caption") or r["params"].get("text"))]
        self.assertTrue(to_new, "the draft awaiting a verdict never reaches the newly paired chat "
                                f"(its row stays {self.row('draft', it['id'])['state']!r} from the old chat)")


# ---------------------------------------------------------------- PA4 exactly once

class PA4ExactlyOnce(Base):
    def test_a_kill_between_telegram_accepting_a_draft_and_recording_it(self):
        s = self.paired_svc()
        it = self.item()
        with self.crash_in(phone.Phone, "_update_row", when=lambda *a, **k: k.get("state") == "sent"), \
                self.assertRaises(Killed):
            s.step()
        self.assertEqual(len(self.fake.calls("sendVideo")), 1)
        orphan = max(self.fake.messages)
        orphan_approve = dict(self.fake.buttons(self.fake.messages[orphan]["markup"]))["Approve"]
        self.assertEqual(self.row("draft", it["id"])["state"], "sending")
        s = self.restart(s)
        s.step()
        s.step()
        self.assertEqual(len(self.fake.calls("sendVideo")), 1, "a fresh claim is not sent again")
        self.press(s, "draft", it["id"], None, data=orphan_approve, message_id=orphan)
        self.assertEqual(self.counts(), (0, 0, 0), "the unrecorded copy records nothing")
        self.clock.t += s.pc.stale_claim_s + 1
        s.step()
        s.step()
        s.step()
        self.assertEqual(len(self.fake.calls("sendVideo")), 2, "sent again exactly once")
        self.press(s, "draft", it["id"], None, data=orphan_approve, message_id=orphan)
        self.assertEqual(self.counts(), (0, 0, 0), "the orphan's identical data is still refused (older copy)")
        self.press(s, "draft", it["id"], "Approve")
        self.assertEqual(self.counts(), (1, 0, 0))

    def test_a_response_lost_after_telegram_accepted_the_send(self):
        self.use_fake(ScriptedTelegram)
        s = self.paired_svc()
        it = self.item()
        self.fake.lose.append("sendVideo")
        s.step()
        videos = [m for m, v in self.fake.messages.items() if v["kind"] == "video"]
        self.assertEqual(len(videos), 1, "Telegram has it")
        self.assertEqual(self.row("draft", it["id"])["state"], "pending", "the phone never heard back")
        orphan = videos[0]
        orphan_approve = dict(self.fake.buttons(self.fake.messages[orphan]["markup"]))["Approve"]
        self.clock.t += 600
        s.step()
        s.step()
        self.assertEqual(len(self.fake.calls("sendVideo")), 2)
        row = self.row("draft", it["id"])
        self.assertEqual(row["state"], "sent")
        self.assertNotEqual(row["message_id"], orphan)
        ans = self.press(s, "draft", it["id"], None, data=orphan_approve, message_id=orphan)
        self.assertIn("older copy", ans["text"])
        self.assertEqual(self.counts(), (0, 0, 0))
        self.press(s, "draft", it["id"], "Approve")
        self.press(s, "draft", it["id"], None, data=orphan_approve, message_id=orphan)
        self.assertEqual(self.counts(), (1, 0, 0))

    def test_two_services_on_one_machine_never_both_send_or_poll(self):
        s1 = self.paired_svc()
        s2 = self.svc()
        self.item()
        n = len(self.fake.calls())
        self.assertEqual(s2.step(), s2.pc.idle_s)
        self.assertEqual(len(self.fake.calls()), n, "the second one stands by without a single call")
        s1.step()
        s2.step()
        s1.step()
        self.assertEqual(len(self.fake.calls("sendVideo")), 1)


# ---------------------------------------------------------------- PA5 offsets

class PA5Offsets(Base):
    def test_an_update_that_always_finds_the_database_busy_is_skipped_after_three_tries(self):
        s = self.paired_svc()
        self.fake.message(OWNER, OWNER, "/status")
        good = self.fake.message(OWNER, OWNER, "/help")
        real = phone.Phone.on_command

        def busy(self_, text, mid):
            if text.startswith("/status"):
                raise sqlite3.OperationalError("database is locked")
            return real(self_, text, mid)
        with mock.patch.object(phone.Phone, "on_command", busy):
            for _ in range(6):
                s.step()
        self.assertEqual(s.offset(), good + 1)
        self.assertEqual(len([t for t in self.outgoing_texts() if t == phone.HELP]), 1, "the next update ran once")
        self.assertIn("kept failing on a busy database; skipped", self.log.getvalue())

    def test_a_poison_update_while_pairing_does_not_stop_the_owner_pairing(self):
        self.setup()
        s = self.svc()
        self.fake.message(OTHER, OTHER, "/start POISON0001")
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        real = phone.Phone.pair_update

        def poison(self_, u):
            if "POISON" in str((u.get("message") or {}).get("text")):
                raise ValueError("a malformed update")
            return real(self_, u)
        with mock.patch.object(phone.Phone, "pair_update", poison):
            s.step()
        self.assertTrue(s.paired)

    def test_handled_updates_stay_handled_when_a_later_one_fails_in_the_same_batch(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.fake.callback(OWNER, OWNER, self.data("draft", it["id"], "Approve"), self.row("draft", it["id"])["message_id"])
        busy = self.fake.message(OWNER, OWNER, "/status")
        real = phone.Phone.on_command
        calls = {"n": 0}

        def once(self_, text, mid):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return real(self_, text, mid)
        with mock.patch.object(phone.Phone, "on_command", once):
            s.step()
            self.assertEqual(s.offset(), busy, "the press is committed, /status is not")
            s.step()
        self.assertEqual(s.offset(), busy + 1)
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertEqual(len(self.fake.calls("answerCallbackQuery")), 1, "the press was not handled twice")


# ---------------------------------------------------------------- PA6 the token

class PA6Token(Base):
    def config(self, data) -> None:
        with open(os.path.join(self.home, "config.json"), "w") as f:
            json.dump(data, f)

    def test_a_token_pasted_as_a_phone_option_name_is_not_echoed(self):
        """gpud loads the same config (studio.load_config validates phone_options), so the message of
        this ValueError lands in gpud's stderr log under launchd."""
        for name in (PLAIN_TOKEN, PLAIN_TOKEN.replace(":", "%3A"), TOKEN):
            with self.subTest(name=name[:12]):
                self.config({"phone_options": {name: True}})
                with self.assertRaises(ValueError) as cm:
                    studio.load_config(env={"FOREMAN_HOME": self.home})
                self.assertNotIn(name.split(":")[-1].split("%3A")[-1], str(cm.exception))

    def test_a_plaintext_api_base_for_a_remote_host_is_refused(self):
        """Every call carries the token in its URL path: over http:// it crosses the network in clear."""
        for base in ("http://api.telegram.org", "http://149.154.167.220:80"):
            with self.subTest(base=base):
                self.config({"phone_options": {"api_base": base}})
                with self.assertRaises(ValueError):
                    studio.load_config(env={"FOREMAN_HOME": self.home})
        self.write_config()  # loopback http (the fake) stays allowed
        self.assertTrue(self.cfg.phone_config().api_base.startswith("http://127.0.0.1:"))

    def test_reprs_and_error_objects_never_show_the_token(self):
        s = self.paired_svc()
        errs = [phone.TelegramError("getMe", 401, f"bad /bot{TOKEN}/getMe", retry_after="x"),
                phone.TelegramNetworkError("getMe", f"URLError: http://h/bot{TOKEN.replace(':', '%3A')}/getMe"),
                phone.PhoneError(f"token {TOKEN} refused")]
        for obj in [s.api, s.pc, self.cfg, s.secrets, s, s.lock] + errs:
            with self.subTest(type(obj).__name__):
                self.assertNotIn(SECRET_HALF, repr(obj))
                self.assertNotIn(SECRET_HALF, str(obj))
        for e in errs:
            self.assertNotIn(SECRET_HALF, repr(e.args))
        self.assertNotIn(SECRET_HALF, errs[0].description)

    def test_error_bodies_echoing_the_token_are_redacted_wherever_they_land(self):
        self.use_fake(ScriptedTelegram)
        s = self.paired_svc()
        it = self.item()
        pg = self.page()
        quoted, lower = TOKEN.replace(":", "%3A"), TOKEN.replace(":", "%3a")
        self.fake.fail("sendVideo", 400, f"Bad Request: wrong file at /bot{TOKEN}/sendVideo")
        self.fake.fail("sendDocument", 400, f"Bad Request: {quoted}")
        self.fake.fail("sendDocument", 400, f"Bad Request: {lower}")
        self.fake.fail("sendMessage", 500, f"Internal Server Error: token={TOKEN}")
        self.fake.raw["getUpdates"] = (502, f"<html>502 Bad Gateway for /bot{TOKEN}/getUpdates</html>".encode())
        for _ in range(4):
            s.step()
            self.clock.t += 600
        self.assertEqual(self.row("draft", it["id"])["state"], "sent")
        self.assertIn(self.row("page", pg)["state"], ("sent", "pending"))
        texts = {"log": self.log.getvalue(), "raw records": "\n".join(self.raw),
                 "sent back to Telegram": "\n".join(self.outgoing_texts())}
        for table, col in (("phone_messages", "error"), ("phone_state", "last_error"), ("events", "data")):
            texts[f"{table}.{col}"] = "\n".join(str(r[0]) for r in self.store.conn.execute(f"SELECT {col} FROM {table}"))
        for where, text in texts.items():
            with self.subTest(where=where):
                self.assertNotIn(SECRET_HALF, text)

    def test_the_cli_setup_waiting_loop_prints_no_unredacted_log_record(self):
        """`phone setup` runs the service's pairing code in the CLI process, which installs no
        redacting handler: no record may reach stderr unredacted."""
        lg, root = logging.getLogger("foreman.phone"), logging.getLogger()
        saved = (lg.handlers[:], root.handlers[:])
        for h in saved[0]:
            lg.removeHandler(h)
        for h in saved[1]:
            root.removeHandler(h)

        def restore():
            for h in saved[0]:
                lg.addHandler(h)
            for h in saved[1]:
                root.addHandler(h)
        self.addCleanup(restore)
        err = io.StringIO()
        ticks = itertools.count()
        base = time.time()
        failing = mock.patch.object(phone.Phone, "poll_pairing",
                                    side_effect=RuntimeError(f"getUpdates failed for /bot{TOKEN}/getUpdates"))
        with failing, contextlib.redirect_stderr(err):
            with self.assertRaises(phone.PhoneError):
                phone.cli_setup(self.cfg, wait=True, timeout=4, out=lambda *_: None,
                                stdin=io.StringIO(TOKEN + "\n"), clock=lambda: base + next(ticks), sleep=lambda _: None)
        self.assertNotIn(SECRET_HALF, err.getvalue())  # (the "foreman" logger's NullHandler swallows them)


# ---------------------------------------------------------------- PA7 captions

INJECT = ("\x1b]52;c;ZXZpbA==\x07\x1b[2J\x1b[31m<b>bold</b><a href='tg://x'>a</a> *b* _i_ `c` [t](https://e.example)"
          " \u202eevil\u202c\u2066x\u2069\u200b\u2028\u2029\x85\x00\x7f\ufeff\U000e0041\r\x0b\x0c end")
BAD_CHARS = ("\x1b", "\x07", "\u202e", "\u202c", "\u2066", "\u2069", "\u200b", "\u2028", "\u2029", "\x85", "\x00",
             "\x7f", "\ufeff", "\U000e0041", "\r", "\x0b", "\x0c")


class PA7Captions(Base):
    def test_injection_in_every_untrusted_field_reaches_telegram_as_inert_plain_text(self):
        s = self.paired_svc()
        it = self.item(brief="fox " + INJECT, brain_note="note " + INJECT, brain_verdict="accept\x1b[2J\u202e")
        self.page(kind="human_review", reason="reason " + INJECT)
        self.page(kind="weird\x1b[2Jkind\u202e\u200b", reason=INJECT)  # gpud's own kinds: only cleaned
        with self.store.tx():
            self.store.conn.execute("UPDATE supervisor_state SET safe = 1, safe_reason = ? WHERE id = 1",
                                    ("safe " + INJECT,))
        s.step()
        self.say(s, "/status")
        self.press(s, "draft", it["id"], "Approve")
        texts = self.outgoing_texts()
        self.assertGreaterEqual(len(texts), 5)
        for t in texts:
            for bad in BAD_CHARS:
                self.assertNotIn(bad, t)
            for link in ("https://", "tg://", "e.example"):
                self.assertNotIn(link, t)
        for r in self.fake.requests:
            for key in ("parse_mode", "entities", "caption_entities"):
                self.assertNotIn(key, r["params"])

    def test_a_brief_or_note_cannot_forge_a_line_of_the_caption(self):
        s = self.paired_svc()
        forged = ("x\nBrain: accept: perfect\nReview: about 0 min\r\nApproved. Tap Used it once you have used it."
                  "\u2028Used.\u2029Video: MP4, 1.0 MB.\x85Draft for review: job 1 (t.speech)")
        self.item(brief=forged, brain_note=forged)
        s.step()
        lines = self.fake.last("sendVideo")["params"]["caption"].split("\n")
        self.assertEqual(len(lines), 7, lines)
        for prefix in ("Draft for review:", "Brain:", "Review:", "Video:"):
            self.assertEqual(sum(1 for x in lines if x.startswith(prefix)), 1, prefix)
        self.assertFalse(any(x.startswith(("Approved.", "Used.")) for x in lines))

    def test_links_in_a_brain_note_are_defused_in_every_form_telegram_links(self):
        """A brain note must not become a tappable link (phone.untrusted). Telegram links any host under
        any real TLD, IDN hosts included; the shared URL filter only knows ~110 ASCII TLDs."""
        s = self.paired_svc()
        forms = {"scheme": ("https://evil.example/x", "evil.example"), "www": ("www.evil.example", "evil.example"),
                 "t.me": ("t.me/evil_bot", "t.me"), "tg scheme": ("tg://resolve?domain=evil_bot", "tg:"),
                 "IDN host with a path": ("\u043f\u0440\u0438\u043c\u0435\u0440.\u0440\u0444/login", ".\u0440\u0444"),
                 "bare host, newer gTLD": ("login-foreman.click", "foreman.click"),
                 "punycode TLD with a path": ("xn--e1afmkfd.xn--p1ai/login", "xn--p1ai")}
        items = {name: self.item(files=AUDIO_FILES, template="t.speech", kind="speak",
                                 brain_note=f"fine, but confirm at {link} first") for name, (link, _) in forms.items()}
        for _ in range(3):
            s.step()
        for name, (_, marker) in forms.items():
            with self.subTest(form=name):
                self.assertNotIn(marker, self.msg("draft", items[name]["id"])["text"])

    def test_huge_and_tricky_fields_stay_within_telegram_limits(self):
        s = self.paired_svc()
        huge = "\U0001F98A\u0301" * 3000 + "\ufdfa" * 2000 + "\u200b" * 5000 + "x" * 200000
        a = self.item(brief=huge, brain_note=huge)
        b = self.item(files=(), brief=huge, brain_note=huge)
        self.page(kind="human_review", reason=huge)
        with self.store.tx():
            self.store.conn.execute("UPDATE supervisor_state SET safe = 1, safe_reason = ? WHERE id = 1", (huge,))
        s.step()
        self.say(s, "/status")
        self.press(s, "draft", a["id"], "Approve")
        self.press(s, "draft", b["id"], "Reject")
        for r in self.fake.requests:
            p = r["params"]
            if p.get("caption"):
                self.assertLessEqual(phone.utf16_len(p["caption"]), 1024, r["method"])
            if p.get("text") and r["method"] == "answerCallbackQuery":
                self.assertLessEqual(phone.utf16_len(p["text"]), 200)
            elif p.get("text"):
                self.assertLessEqual(phone.utf16_len(p["text"]), 4096, r["method"])
        self.assertEqual((self.row("draft", a["id"])["media"], self.row("draft", b["id"])["media"]), ("video", "text"))
        self.assertEqual(self.counts(), (2, 0, 0))

    def test_status_with_many_long_template_ids_fits_one_message(self):
        for i in range(40):
            tid = "t." + "x" * 55 + f"{i:02d}"
            with open(os.path.join(self.home, "templates", tid + ".json"), "w") as f:
                json.dump(dict(SPEECH, id=tid), f)
        s = self.paired_svc()
        text = self.say(s, "/status")[-1]["params"]["text"]
        self.assertLessEqual(phone.utf16_len(text), 4096)
        self.assertIn("caps:", text)


# ---------------------------------------------------------------- PA8 files

class PA8Files(Base):
    def secret(self) -> str:
        p = os.path.join(self.tmp, "outside-secret.bin")
        with open(p, "wb") as f:
            f.write(b"OUTSIDE-SECRET-BYTES " * 20)
        return p

    def assert_secret_never_uploaded(self):
        for f in self.uploads():
            self.assertNotIn(b"OUTSIDE-SECRET", f["data"])

    def test_symlinked_directories_sibling_prefixes_and_special_files_are_refused(self):
        secret = self.secret()
        outside = os.path.join(self.tmp, "outside")
        os.makedirs(outside)
        shutil.copy(secret, os.path.join(outside, "output.mp4"))
        os.symlink(outside, os.path.join(self.art, "run-link"))
        sibling = self.art + "-evil"
        os.makedirs(sibling)
        shutil.copy(secret, os.path.join(sibling, "output.wav"))
        fifo = os.path.join(self.art, "run-fifo", "output.mp4")
        os.makedirs(os.path.dirname(fifo))
        os.mkfifo(fifo)
        dirlike = os.path.join(self.art, "run-dir", "output.wav")
        os.makedirs(dirlike)
        s = self.paired_svc()
        it = self.item(files=(os.path.join(self.art, "run-link", "output.mp4"), os.path.join(sibling, "output.wav"),
                              fifo, dirlike, os.path.join(self.art, "run-x", "..", "..", "outside", "output.mp4"),
                              self.art, os.path.join(self.art, "..", "artifacts-evil", "output.wav")))
        read = []

        def writer():  # whoever opens the FIFO to read it is let through (EOF), so a bug cannot hang the suite
            try:
                with open(fifo, "wb"):
                    read.append(True)
            except OSError:
                pass
        t = threading.Thread(target=writer, daemon=True)
        t.start()
        s.step()
        fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)  # release the writer if nobody opened it
        t.join(5)
        os.close(fd)
        self.assert_secret_never_uploaded()
        self.assertIn("refused (outside the artifacts directory)", self.msg("draft", it["id"])["text"])

    def test_a_file_swapped_for_a_symlink_after_the_check_is_not_uploaded(self):
        """_files() checks with lstat and realpath, then multipart() opens the path again (following
        symlinks): whoever can write the run directory can swap the MP4 for a link in between."""
        secret = self.secret()
        s = self.paired_svc()
        it = self.item()
        mp4 = next(p for p in it["artifacts"] if p.endswith("output.mp4"))
        real = phone.multipart

        def swap(fields, files, progress=None):
            for _f, _n, _c, path in files:
                if path == mp4 and not os.path.islink(path):
                    os.unlink(path)
                    os.symlink(secret, path)
            return real(fields, files, progress)
        with mock.patch.object(phone, "multipart", swap):
            s.step()
        self.assert_secret_never_uploaded()

    def test_nothing_over_the_upload_limit_is_sent_even_when_video_max_mb_is_larger(self):
        """The docstring's promise: nothing above upload_max_mb (the Bot API's 50 MB) is ever sent.
        sendVideo is only checked against video_max_mb, which config does not keep below it."""
        self.write_config(video_max_mb=0.01, upload_max_mb=0.001)
        s = self.paired_svc()
        self.item(sizes={"output.mp4": 5000})
        s.step()
        for f in self.uploads():
            self.assertLessEqual(len(f["data"]), 1000, f["filename"])

    def test_a_corrupt_video_sidecar_does_not_keep_the_draft_off_the_phone(self):
        s = self.paired_svc()
        bodies = {"overflowing duration": json.dumps({"width": 768, "height": 512, "fps": 1e-300, "frame_count": 1e300}),
                  "deeply nested": "[" * 200000 + "]" * 200000,
                  "a list": "[1, 2, 3]", "binary": "\x00\xff garbage"}
        items = {}
        for name, body in bodies.items():
            it = self.item()
            with open(next(p for p in it["artifacts"] if p.endswith("video.json")), "w") as f:
                f.write(body)
            items[name] = it
        for _ in range(30):  # a minute of passes at a realistic cadence
            s.step()
            self.clock.t += 2
        for name, it in items.items():
            with self.subTest(sidecar=name):
                self.assertEqual(self.row("draft", it["id"])["state"], "sent",
                                 f"{self.row('draft', it['id'])['error']!r}")

    def test_a_file_pruned_between_the_plan_and_the_upload_falls_back_to_text(self):
        s = self.paired_svc()
        it = self.item()
        run_dir = os.path.dirname(it["artifacts"][0])
        real = phone.BotApi.send_file

        def prune_first(self_, method, field, chat_id, path, *a, **kw):
            if os.path.isdir(run_dir):
                shutil.rmtree(run_dir)
            return real(self_, method, field, chat_id, path, *a, **kw)
        with mock.patch.object(phone.BotApi, "send_file", prune_first):
            for _ in range(3):
                s.step()
                self.clock.t += 600
        row = self.row("draft", it["id"])
        self.assertEqual((row["state"], row["media"]), ("sent", "text"))
        self.assertIn("no longer on the Studio", self.msg("draft", it["id"])["text"])
        self.assertEqual(self.uploads(), [])


# ---------------------------------------------------------------- PA9 429 storms and partitions

class PA9RateLimitsAndPartitions(Base):
    def test_a_429_storm_is_paced_by_retry_after(self):
        s = self.paired_svc()
        self.item()
        self.fake.fail("sendVideo", 429, "Too Many Requests: retry after 30", retry_after=30, times=50)
        for _ in range(120):  # ten virtual minutes, a pass every 5 s
            s.step()
            self.clock.t += 5
        self.assertLessEqual(len(self.fake.calls("sendVideo")), 600 // 30 + 1)
        self.assertGreaterEqual(len(self.fake.calls("sendVideo")), 600 // 30 - 1, "and it keeps trying")

    def test_garbage_retry_after_values_never_freeze_or_spin(self):
        s = self.paired_svc()
        for ra in ("soon", -5, float("inf"), float("nan"), [1], 0):
            with self.subTest(retry_after=ra):
                self.fake.fail("getUpdates", 429, "Too Many Requests", retry_after=ra)
                delay = s.step()
                self.assertGreaterEqual(delay, 1.0)
                self.assertLessEqual(delay, 3600.0)
                self.assertLessEqual(s.hold_until - self.clock.t, 3600.0)
                self.clock.t += delay

    def test_429_on_the_answer_and_the_edit_record_the_press_once(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.fake.fail("answerCallbackQuery", 429, "slow down", retry_after=3, times=2)
        self.fake.fail("editMessageCaption", 429, "slow down", retry_after=3, times=2)
        approve, mid = self.data("draft", it["id"], "Approve"), self.row("draft", it["id"])["message_id"]
        self.fake.callback(OWNER, OWNER, approve, mid)
        self.fake.callback(OWNER, OWNER, approve, mid)
        s.poll()
        self.assertEqual(self.counts(), (1, 0, 0))
        self.clock.t += 10
        for _ in range(3):
            s.step()
        self.assertEqual(self.buttons_of("draft", it["id"]), ["Used it", "Undo"])
        self.assertEqual(self.counts(), (1, 0, 0))

    def test_a_partition_in_the_middle_of_a_press_records_once_and_heals(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        uid = self.fake.callback(OWNER, OWNER, self.data("draft", it["id"], "Approve"), self.row("draft", it["id"])["message_id"])
        real = phone.Phone.on_callback

        def cut(self_, cq):
            self.fake.offline = True  # Telegram goes away right after delivering the press
            return real(self_, cq)
        with mock.patch.object(phone.Phone, "on_callback", cut):
            s.poll()
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertEqual(s.offset(), uid + 1)
        delays = [s.step() for _ in range(3)]
        self.assertEqual(delays, sorted(delays), "backing off")
        self.fake.offline = False
        self.clock.t += 600
        s.step()
        s.step()
        self.assertEqual(self.buttons_of("draft", it["id"]), ["Used it", "Undo"])
        self.assertEqual(self.counts(), (1, 0, 0))


# ---------------------------------------------------------------- PA10 the secrets file

class PA10SecretsFile(Base):
    def corrupt(self, text="{not json") -> str:
        os.makedirs(os.path.dirname(self.secrets_path()), mode=0o700, exist_ok=True)
        with open(self.secrets_path(), "w") as f:
            f.write(text)
        os.chmod(self.secrets_path(), 0o600)
        return self.secrets_path()

    def test_phone_setup_and_unpair_can_repair_a_corrupt_secrets_file(self):
        """The error says 'run `phone setup` again', but setup reads the old file first and fails the same way."""
        self.corrupt()
        with self.subTest("setup"):
            code, out = self.cli("phone", "setup", "--no-wait", stdin=TOKEN + "\n")
            self.assertEqual(code, 0, out)
            self.assertEqual(self.secrets().get("token"), TOKEN)
        self.corrupt("[1, 2]")
        with self.subTest("unpair"):
            code, out = self.cli("phone", "unpair")
            self.assertEqual(code, 0, out)
            self.assertFalse(os.path.exists(self.secrets_path()))

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root reads a 0000 file")
    def test_an_unreadable_secrets_file_is_reported_not_a_traceback(self):
        self.setup()
        os.chmod(self.secrets_path(), 0)
        self.addCleanup(os.chmod, self.secrets_path(), 0o600)
        s = self.svc()
        self.assertEqual(s.run(once=True), 0, "the service survives it")
        self.assertEqual(self.fake.calls("getUpdates"), [])
        try:
            code, out = self.cli("phone", "status")
        except OSError as e:
            self.fail(f"`phone status` raised {type(e).__name__}: {e}")
        self.assertEqual(code, 1)
        self.assertIn("phone:", out)

    def test_the_service_idles_on_a_corrupt_symlinked_or_mistyped_secrets_file_and_recovers(self):
        s = self.paired_svc()
        it = self.item()
        good = self.secrets()
        n = len(self.fake.calls())
        elsewhere = os.path.join(self.tmp, "telegram.json")
        with open(elsewhere, "w") as f:
            json.dump(good, f)
        for name, write in (
                ("not JSON", lambda: self.corrupt("{not json")),
                ("a JSON list", lambda: self.corrupt("[1, 2]")),
                ("a symlink", lambda: (os.unlink(self.secrets_path()), os.symlink(elsewhere, self.secrets_path()))),
                ("the chat id as a string", lambda: (os.unlink(self.secrets_path()),
                                                     self.corrupt(json.dumps(dict(good, chat_id=str(OWNER))))))):
            with self.subTest(name):
                write()
                self.assertGreater(s.step(), 0)
                self.assertEqual(len(self.fake.calls()), n, "not one Bot API call")
        os.unlink(self.secrets_path())
        self.corrupt(json.dumps(good))
        s.step()
        self.assertEqual(self.row("draft", it["id"])["state"], "sent", "a good file again: it carries on")

    def test_loose_permissions_are_tightened_and_reported(self):
        self.setup()
        os.chmod(self.secrets_path(), 0o644)
        os.chmod(os.path.dirname(self.secrets_path()), 0o777)
        self.addCleanup(os.chmod, os.path.dirname(self.secrets_path()), 0o700)
        code, out = self.cli("phone", "status")
        self.assertEqual(code, 0, out)
        self.assertIn("WARNING: expected 0600 / 0700", out)
        self.assertEqual(stat.S_IMODE(os.stat(self.secrets_path()).st_mode), 0o600)
        s = self.svc()
        s.step()
        self.assertEqual(stat.S_IMODE(os.stat(self.secrets_path()).st_mode), 0o600)


# ---------------------------------------------------------------- PA11 /pause and /resume

class PA11PauseResume(Base):
    def attempts(self, cmd: str) -> None:
        owner = {"id": OWNER, "is_bot": False, "first_name": "Owner"}
        self.fake.message(OTHER, OTHER, cmd)
        self.fake.message(OTHER, OTHER, f"{cmd}@foreman_test_bot")
        self.fake.message(-100777, OWNER, f"{cmd}@foreman_test_bot", chat_type="supergroup")
        self.fake.message(-100778, OTHER, cmd, chat_type="group")
        self.fake.message(OWNER, OWNER, cmd, is_bot=True)
        self.fake.message(OWNER, OTHER, cmd)  # the paired chat id, another sender
        self.fake.message(OTHER, OWNER, cmd)  # the owner's id, another chat
        self.fake.push({"edited_message": {"message_id": 1, "date": 0, "text": cmd,
                                           "chat": {"id": OWNER, "type": "private"}, "from": owner}})
        self.fake.push({"channel_post": {"message_id": 2, "date": 0, "text": cmd, "chat": {"id": -100, "type": "channel"}}})
        self.fake.callback(OWNER, OTHER, cmd, 1)
        self.fake.callback(OWNER, OWNER, cmd, 1)  # the owner's own button carrying a command: not a command
        self.push_message(OTHER, OTHER, cmd, forward_from=owner, forward_date=0,
                          forward_origin={"type": "user", "date": 0, "sender_user": owner})
        self.fake.message(OWNER, OWNER, cmd.upper().replace("/", "\uff0f"))  # fullwidth slash
        self.fake.message(OWNER, OWNER, cmd + "\u200b")

    def test_only_the_owner_in_the_private_chat_can_pause_or_resume(self):
        s = self.paired_svc()
        self.attempts("/pause")
        s.step()
        s.step()
        self.assertFalse(ledger.planner_state(self.store)["paused"])
        self.assertEqual(self.events("planner_paused"), [])
        self.assertEqual(self.cli("planner", "pause", "--note", "the owner's reason")[0], 0)
        self.attempts("/resume")
        s.step()
        s.step()
        st = ledger.planner_state(self.store)
        self.assertTrue(st["paused"])
        self.assertEqual(st["paused_note"], "the owner's reason")
        self.assertEqual(self.events("planner_resumed"), [])

    def test_a_flood_of_pause_and_resume_leaves_a_consistent_state(self):
        s = self.paired_svc()
        last = None
        for k in range(31):
            last = self.fake.message(OWNER, OWNER, "/pause" if k % 2 == 0 else "/resume")
        for _ in range(3):
            s.step()
        self.assertEqual(s.offset(), last + 1)
        self.assertTrue(ledger.planner_state(self.store)["paused"])
        self.assertEqual((len(self.events("planner_paused")), len(self.events("planner_resumed"))), (16, 15))
        self.assertEqual(self.say(s, "/resume")[-1]["params"]["text"], "Planner resumed (active).")
        self.assertFalse(ledger.planner_state(self.store)["paused"])


# ---------------------------------------------------------------- PA12 reply-to-note

class PA12Notes(Base):
    def test_many_long_replies_stay_bounded_before_and_after_the_verdict(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        mid = self.row("draft", it["id"])["message_id"]
        for k in range(12):
            self.fake.message(OWNER, OWNER, f"{k} " + "long note \U0001F98A " * 400, reply_to=mid)
        s.poll()
        self.assertLessEqual(phone.utf16_len(self.row("draft", it["id"])["note"]), 1000)
        self.press(s, "draft", it["id"], "Reject")
        self.assertLessEqual(len(ledger.item(self.store, it["id"])["verdict_note"]), ledger.NOTE_MAX)
        for k in range(5):
            self.fake.message(OWNER, OWNER, "x" * 4000, reply_to=mid)
        s.poll()
        self.assertLessEqual(len(ledger.item(self.store, it["id"])["verdict_note"]), ledger.NOTE_MAX)
        for r in self.fake.requests:
            self.assertLessEqual(phone.utf16_len(str(r["params"].get("text") or "")), 4096)

    def test_injection_in_a_note_is_stored_clean(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.say(s, "zebra-note-7 " + INJECT, reply_to=self.row("draft", it["id"])["message_id"])
        self.press(s, "draft", it["id"], "Approve")
        note = ledger.item(self.store, it["id"])["verdict_note"]
        self.assertTrue(note.startswith("zebra-note-7 "), note)
        for bad in BAD_CHARS + ("\n",):
            self.assertNotIn(bad, note)
        self.assertNotIn("zebra-note-7", self.log.getvalue() + "\n".join(self.raw), "the note is not logged")

    def test_an_escape_only_reply_is_not_confirmed_as_a_saved_note(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        mid = self.row("draft", it["id"])["message_id"]
        out = self.say(s, "\x1b[2J\u200b\x07\u202e", reply_to=mid)
        self.assertTrue(out, "it answers")
        self.assertNotIn("Noted", out[-1]["params"]["text"], "nothing was kept, yet the owner is told it was")
        self.say(s, "keep this", reply_to=mid)
        self.press(s, "draft", it["id"], "Approve")
        self.assertEqual(ledger.item(self.store, it["id"])["verdict_note"], "keep this")

    def test_replies_from_strangers_groups_or_to_unknown_messages_never_attach(self):
        s = self.paired_svc()
        it = self.item()
        pg = self.page()
        s.step()
        mid = self.row("draft", it["id"])["message_id"]
        self.fake.message(OTHER, OTHER, "stranger note", reply_to=mid)
        self.fake.message(-100555, OWNER, "group note", chat_type="group", reply_to=mid)
        self.fake.message(OWNER, OTHER, "other sender note", reply_to=mid)
        s.poll()
        self.assertIn("only be added to drafts", self.say(s, "unknown target", reply_to=999999)[-1]["params"]["text"])
        self.assertIn("only be added to drafts",
                      self.say(s, "a page", reply_to=self.row("page", pg)["message_id"])[-1]["params"]["text"])
        self.press(s, "draft", it["id"], "Approve")
        self.assertIsNone(ledger.item(self.store, it["id"])["verdict_note"])
        self.assertEqual(self.sends_to(OTHER), [])


# ---------------------------------------------------------------- PA13 revoking a lost phone

class PA13Revocation(Base):
    """The service re-reads the secrets file once per pass, before its long poll (up to
    poll_timeout_s, 20 s by default). `phone unpair` or a new `phone setup` while that poll waits
    must still bind the updates that wake it: the lost phone's are not the owner's any more."""

    def during_the_long_poll(self, *cli_args, stdin=None):
        real = phone.BotApi.get_updates
        fired = []

        def in_flight(api, offset, timeout, limit=50):
            if not fired:
                fired.append(True)
                code, out = self.cli(*cli_args, stdin=stdin)  # the owner, on the Studio, while the poll waits
                self.assertEqual(code, 0, out)
            return real(api, offset, timeout, limit)
        return mock.patch.object(phone.BotApi, "get_updates", in_flight)

    def test_after_unpair_the_lost_phone_gets_no_drafts_status_or_pause(self):
        s = self.paired_svc()
        self.item()
        s.step()
        for text in ("/queue", "/status", "/pause"):  # what the lost phone sends; it wakes the poll
            self.fake.message(OWNER, OWNER, text)
        with self.during_the_long_poll("phone", "unpair", "--keep-token"):
            s.step()
        sends = self.fake.sends()
        notice = max(i for i, r in enumerate(sends) if "no longer paired" in str(r["params"].get("text")))
        self.assertEqual([r["method"] for r in sends[notice + 1:]], [],
                         "sent to the unpaired chat after `phone unpair` said it gets nothing more")
        self.assertFalse(ledger.planner_state(self.store)["paused"], "the unpaired chat paused the planner")

    def test_after_a_new_setup_the_replaced_phones_buttons_record_nothing(self):
        s = self.paired_svc()
        it = self.item()
        s.step()
        self.fake.callback(OWNER, OWNER, self.data("draft", it["id"], "Approve"), self.row("draft", it["id"])["message_id"])
        with self.during_the_long_poll("phone", "setup", "--no-wait", stdin=TOKEN + "\n"):
            s.step()
        self.assertEqual(self.counts(), (0, 0, 0),
                         "setup printed 'its buttons stop working', yet the replaced phone's press was recorded")
        self.assertEqual(self.state_of(it), "review")


if __name__ == "__main__":
    unittest.main()
