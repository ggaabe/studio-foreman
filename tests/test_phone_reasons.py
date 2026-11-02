"""Reject reason tags on the phone (foreman/phone.py, foreman/ledger.py), against the fake Bot API.

  RT1  Reject records the verdict exactly once, through the same ledger call as before; then the
       message's buttons become the reason tags, Done and Undo
  RT2  a tap toggles a tag (a check mark), only on the message; Done stores the tags with the
       verdict (verdict_tags, and the note's readable reasons) and the message goes back to Undo
  RT3  Undo, before or after Done, clears the tags; presses after it do nothing
  RT4  forged, garbled, re-signed and misplaced tag data record nothing
  RT5  double taps, a tap on an older drawing, a crash between the ledger and the offset commit
  RT6  a press after unpair or after a new pairing does nothing
  RT7  reply notes still attach; a reject changed elsewhere ends the picking
  RT8  the CLI: verdict JOB reject --tags, review and report show tag counts per template
  RT9  the ledger: parse_tags, reject_tag_counts, template_stats, the migration of an old database

Nothing here talks to the real Telegram API.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_phone import AUDIO_FILES, CODE, OTHER, OWNER, PhoneCase  # noqa: E402

from foreman import cli, ledger, phone, report  # noqa: E402
from foreman import store as store_mod  # noqa: E402
from foreman.store import Store  # noqa: E402
from foreman.templates import load_templates  # noqa: E402

TAG_LABELS = ["Script", "Voice", "Pronunciation", "Music", "Lip-sync", "Faces/consistency", "Stills", "Motion",
              "B-roll", "Captions/text", "Pacing", "End card", "Other"]
CHECK = "✓ "


class Killed(BaseException):
    """kill -9 at a chosen point (nothing in foreman/ catches a BaseException that is not an Exception)."""


class Base(PhoneCase):
    def setUp(self):
        super().setUp()
        self.s = self.paired_svc()
        self.it = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        self.s.step()

    # ---- helpers
    def labels(self, it=None) -> list:
        return [t for t, _ in self.fake.buttons(self.msg("draft", (it or self.it)["id"])["markup"])]

    def ledger_item(self, it=None) -> dict:
        return ledger.item(self.store, (it or self.it)["id"])

    def tap(self, label, it=None, **kw) -> dict:
        return self.press(self.s, "draft", (it or self.it)["id"], label, **kw)

    def picks(self, it=None):
        return phone.row_tags(self.row("draft", (it or self.it)["id"]))

    def counts(self) -> tuple:
        return (len(self.events("verdict_recorded")), len(self.events("verdict_retracted")),
                len(self.events("verdict_tagged")))

    def restart(self) -> None:
        self.s.close()
        self.s = self.svc()
        self.s.reload()

    @contextlib.contextmanager
    def crash_in(self, owner, attr):
        real = getattr(owner, attr)
        armed = {"on": True}

        def boom(*a, **k):
            if armed["on"]:
                armed["on"] = False
                raise Killed(f"kill -9 in {attr}")
            return real(*a, **k)
        with mock.patch.object(owner, attr, boom):
            yield
        self.assertFalse(armed["on"], f"the crash point {attr} was never reached")


# ---------------------------------------------------------------- RT1 the verdict, then the tags

class RT1RejectThenTags(Base):
    def test_reject_records_once_through_the_same_call_then_offers_the_reason_tags(self):
        with mock.patch.object(cli, "apply_verdict", wraps=cli.apply_verdict) as spy:
            ans = self.tap("Reject")
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args, mock.call(self.cfg, self.it["job_id"], "reject", None, None, by="phone"),
                         "the same ledger call as before the tags existed")
        self.assertIn("Rejected. P(ship) for t.speech is now 0.73", ans["text"])  # Beta(8, 2) and one reject
        self.assertIn("What was wrong?", ans["text"])
        ev = self.events("verdict_recorded")
        self.assertEqual((len(ev), ev[0]["data"]["verdict"], ev[0]["data"]["by"]), (1, "reject", "phone"))
        self.assertNotIn("tags", ev[0]["data"])
        it = self.ledger_item()
        self.assertEqual((it["state"], it["verdict_tags"]), ("rejected", None))
        self.assertEqual(self.labels(), TAG_LABELS + ["Done", "Undo"])
        markup = self.msg("draft", self.it["id"])["markup"]
        self.assertEqual([len(r) for r in markup["inline_keyboard"]], [3, 3, 3, 3, 1, 2])
        self.assertIn(phone.PICKING, self.msg("draft", self.it["id"])["text"])
        self.assertEqual(self.picks(), [])
        row = self.row("draft", self.it["id"])
        for text, data in self.fake.buttons(markup):  # every button signed by this pairing for this drawing
            parsed = phone.parse_data(self.s.key, OWNER, data)
            self.assertIsNotNone(parsed, text)
            self.assertLessEqual(len(data.encode()), 64)
            self.assertEqual(parsed[1:4], ("draft", self.it["id"], row["issue"]))
        tags = [phone.parse_data(self.s.key, OWNER, d)[4] for _, d in self.fake.buttons(markup)[:13]]
        self.assertEqual(tags, [t for t, _n in ledger.REASON_TAGS])

    def test_with_reject_tags_off_reject_behaves_as_before(self):
        self.write_config(reject_tags=False)
        s = self.svc()
        s.reload()
        ans = self.press(s, "draft", self.it["id"], "Reject")
        self.assertNotIn("What was wrong", ans["text"])
        self.assertEqual(self.labels(), ["Undo"])
        self.assertIsNone(self.picks())
        self.assertNotIn(phone.PICKING, self.msg("draft", self.it["id"])["text"])

    def test_approve_and_cli_rejects_never_start_picking(self):
        b = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        self.s.step()
        self.tap("Approve")
        self.assertEqual(self.labels(), ["Used it", "Undo"])
        self.assertEqual(self.cli("verdict", str(b["job_id"]), "reject")[0], 0)
        self.s.step()  # the refresh pass redraws it: rejected elsewhere, so no picking
        self.assertEqual(self.labels(b), ["Undo"])
        self.assertIsNone(self.picks(b))


# ---------------------------------------------------------------- RT2 toggles and Done

class RT2TogglesAndDone(Base):
    def test_taps_toggle_on_the_message_only_and_done_stores_them_with_the_verdict(self):
        self.tap("Reject")
        ans = self.tap("Voice")
        self.assertIn("Voice: marked", ans["text"])
        self.assertEqual(self.picks(), ["voice"])
        self.assertIn(CHECK + "Voice", self.labels())
        self.tap("Pacing")
        self.tap("Lip-sync")
        ans = self.tap(CHECK + "Lip-sync")
        self.assertIn("Lip-sync: unmarked", ans["text"])
        self.assertEqual(self.picks(), ["voice", "pacing"])
        labels = self.labels()
        self.assertEqual([x for x in labels if x.startswith(CHECK)], [CHECK + "Voice", CHECK + "Pacing"])
        self.assertIn("Lip-sync", labels)
        self.assertIsNone(self.ledger_item()["verdict_tags"], "nothing reaches the ledger before Done")
        self.assertEqual(self.counts(), (1, 0, 0))
        ans = self.tap("Done")
        self.assertEqual(ans["text"], "Recorded with the reject: voice, pacing.")
        it = self.ledger_item()
        self.assertEqual(it["verdict_tags"], ["voice", "pacing"])
        self.assertEqual(it["verdict_note"], "reasons: voice, pacing")
        self.assertEqual((it["state"], it["verdict"]), ("rejected", "reject"))
        ev = self.events("verdict_tagged")
        self.assertEqual((len(ev), ev[0]["data"]["tags"], ev[0]["data"]["by"], ev[0]["job_id"]),
                         (1, ["voice", "pacing"], "phone", self.it["job_id"]))
        self.assertEqual(self.counts(), (1, 0, 1), "one verdict, one set of reasons")
        self.assertEqual(self.labels(), ["Undo"], "back to the normal post-verdict state")
        self.assertIsNone(self.picks())
        self.assertIn("Rejected; reasons: voice, pacing.", self.msg("draft", self.it["id"])["text"])
        stored = self.store.conn.execute("SELECT verdict_tags FROM plan_items WHERE id = ?",
                                         (self.it["id"],)).fetchone()
        self.assertEqual(json.loads(stored[0]), ["voice", "pacing"], "a structured column, JSON")

    def test_done_with_nothing_picked_records_no_reasons(self):
        self.tap("Reject")
        ans = self.tap("Done")
        self.assertIn("no reasons recorded", ans["text"])
        self.assertEqual(self.counts(), (1, 0, 0))
        it = self.ledger_item()
        self.assertEqual((it["verdict_tags"], it["verdict_note"]), (None, None))
        self.assertEqual(self.labels(), ["Undo"])
        self.assertIn("Rejected.", self.msg("draft", self.it["id"])["text"])

    def test_every_tag_can_be_picked_and_the_note_stays_within_its_limit(self):
        mid = self.row("draft", self.it["id"])["message_id"]
        self.say(self.s, "x" * 600, reply_to=mid)  # a long note kept for the verdict
        self.say(self.s, "y" * 600, reply_to=mid)
        self.tap("Reject")
        for label in TAG_LABELS:
            self.tap(label)
        self.tap("Done")
        it = self.ledger_item()
        self.assertEqual(it["verdict_tags"], [t for t, _n in ledger.REASON_TAGS])
        self.assertLessEqual(len(it["verdict_note"]), ledger.NOTE_MAX)
        self.assertTrue(it["verdict_note"].endswith(" | " + ledger.reasons_text(it["verdict_tags"])),
                        "the reasons are kept whole; the owner's text is shortened")
        self.assertTrue(it["verdict_note"].startswith("x" * 100))


# ---------------------------------------------------------------- RT3 Undo

class RT3Undo(Base):
    def test_undo_after_done_clears_the_tags_and_a_new_reject_picks_afresh(self):
        self.tap("Reject")
        self.tap("Voice")
        self.tap("Done")
        ans = self.tap("Undo")
        self.assertIn("back in the review queue", ans["text"])
        it = self.ledger_item()
        self.assertEqual((it["state"], it["verdict_tags"], it["verdict_note"]), ("review", None, None))
        self.assertIsNone(self.store.conn.execute("SELECT verdict_tags FROM plan_items WHERE id = ?",
                                                  (self.it["id"],)).fetchone()[0], "SQL NULL, not 'null'")
        self.assertEqual(self.labels(), ["Approve", "Reject"])
        self.tap("Reject")
        self.assertEqual(self.picks(), [])
        self.assertNotIn(CHECK + "Voice", self.labels())
        self.tap("Music")
        self.tap("Done")
        self.assertEqual(self.ledger_item()["verdict_tags"], ["music"])
        self.assertEqual(self.counts(), (2, 1, 2))

    def test_undo_while_picking_and_every_tag_button_pressed_after_it_does_nothing(self):
        self.tap("Reject")
        self.tap("Voice")
        drawn = self.fake.buttons(self.msg("draft", self.it["id"])["markup"])
        self.tap("Undo")
        self.assertEqual(self.ledger_item()["state"], "review")
        self.assertIsNone(self.picks())
        before = self.counts()
        for text, data in drawn:
            with self.subTest(text):
                ans = self.press(self.s, "draft", self.it["id"], None, data=data)
                self.assertIn("Already done" if text == "Undo" else "out of date", ans["text"])
        self.assertEqual(self.counts(), before)
        it = self.ledger_item()
        self.assertEqual((it["state"], it["verdict_tags"]), ("review", None))
        self.assertIsNone(self.picks())
        self.assertEqual(self.labels(), ["Approve", "Reject"])
        self.assertEqual(self.events("verdict_tagged"), [])


# ---------------------------------------------------------------- RT4 forged data

class RT4Forged(Base):
    def test_forged_garbled_and_re_signed_tag_data_record_nothing(self):
        self.tap("Reject")
        row = self.row("draft", self.it["id"])
        voice = self.data("draft", self.it["id"], "Voice")
        head, issue, letter, sig = voice.split(".")
        ref, iss = self.it["id"], row["issue"]
        key = self.s.key
        variants = {
            "MAC flipped": f"{head}.{issue}.{letter}.{sig[:-1]}{'A' if sig[-1] != 'A' else 'B'}",
            "another tag letter, the same MAC": f"{head}.{issue}.k.{sig}",
            "no tag letter": f"{head}.{issue}.{sig}",
            "two tag letters": f"{head}.{issue}.{letter}.k.{sig}",
            "upper-case letter": f"{head}.{issue}.{letter.upper()}.{sig}",
            "a letter beyond the tags, signed": phone.make_data(key, OWNER, "t", "d", ref, iss, "z"),
            "a tag on Done, signed": phone.make_data(key, OWNER, "k", "d", ref, iss, "b"),
            "a tag on Reject, signed": phone.make_data(key, OWNER, "r", "d", ref, iss, "b"),
            "a tag toggle on a page, signed": phone.make_data(key, OWNER, "t", "p", ref, iss, "b"),
            "another key": phone.make_data(phone.new_key(), OWNER, "t", "d", ref, iss, "b"),
            "another chat": phone.make_data(key, OTHER, "t", "d", ref, iss, "b"),
            "leading zero": f"td0{head[2:]}.{issue}.{letter}.{sig}",
            "65 characters": voice + "A" * (65 - len(voice)),
            "NUL appended": voice + "\x00",
            "garbage": "tb.1.b.xxxxxxxxxxxxxxxx",
            "an int": 7, "null": None,
        }
        for name, data in variants.items():
            with self.subTest(name):
                ans = self.press(self.s, "draft", self.it["id"], None, data=data)
                self.assertIn("not valid", ans.get("text", ""))
        self.assertEqual(self.picks(), [], "nothing toggled")
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertEqual(self.labels(), TAG_LABELS + ["Done", "Undo"], "the genuine buttons are still drawn")
        self.tap("Voice")  # and still work
        self.assertEqual(self.picks(), ["voice"])

    def test_a_genuine_tag_button_on_another_drafts_message_records_nothing(self):
        b = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        self.s.step()
        self.tap("Reject")
        self.tap("Reject", it=b)
        voice_a = self.data("draft", self.it["id"], "Voice")
        ans = self.press(self.s, "draft", b["id"], None, data=voice_a)
        self.assertIn("older copy", ans["text"])
        self.assertEqual((self.picks(), self.picks(b)), ([], []))
        self.assertEqual(self.labels(b), TAG_LABELS + ["Done", "Undo"])

    def test_a_press_from_another_user_or_chat_is_ignored(self):
        self.tap("Reject")
        voice = self.data("draft", self.it["id"], "Voice")
        n = len(self.fake.calls("answerCallbackQuery"))
        self.press(self.s, "draft", self.it["id"], None, data=voice, user=OTHER)
        self.press(self.s, "draft", self.it["id"], None, data=voice, chat=OTHER, user=OWNER)
        self.assertEqual(len(self.fake.calls("answerCallbackQuery")), n)
        self.assertEqual(self.picks(), [])


# ---------------------------------------------------------------- RT5 double taps, stale drawings, crashes

class RT5DoubleTapsAndCrashes(Base):
    def test_a_double_tap_toggles_once_and_the_next_genuine_tap_still_counts(self):
        self.tap("Reject")
        voice = self.data("draft", self.it["id"], "Voice")
        mid = self.row("draft", self.it["id"])["message_id"]
        self.fake.callback(OWNER, OWNER, voice, mid)
        self.fake.callback(OWNER, OWNER, voice, mid)
        n = len(self.fake.calls("answerCallbackQuery"))
        self.s.poll()
        answers = [a["params"]["text"] for a in self.fake.calls("answerCallbackQuery")[n:]]
        self.assertEqual(len(answers), 2)
        self.assertIn("Voice: marked", answers[0])
        self.assertIn("older drawing", answers[1])
        self.assertIn("Voice is marked", answers[1])
        self.assertEqual(self.picks(), ["voice"])
        issue = self.row("draft", self.it["id"])["issue"]
        self.tap("Pacing")  # the stale tap did not move the issue on: the buttons on screen work
        self.assertEqual(self.picks(), ["voice", "pacing"])
        self.assertEqual(self.row("draft", self.it["id"])["issue"], issue + 1)

    def test_a_tap_on_an_older_drawing_changes_nothing_and_says_so(self):
        self.tap("Reject")
        old = dict(self.fake.buttons(self.msg("draft", self.it["id"])["markup"]))
        self.tap("Voice")
        ans = self.press(self.s, "draft", self.it["id"], None, data=old["Pacing"])  # tapped before the edit arrived
        self.assertIn("changed nothing: Pacing is not marked", ans["text"])
        self.assertEqual(self.picks(), ["voice"])
        ans = self.press(self.s, "draft", self.it["id"], None, data=old["Done"])
        self.assertIn("changed nothing", ans["text"])
        self.assertEqual(self.counts(), (1, 0, 0), "an old Done stores nothing")
        self.tap("Pacing")
        self.assertEqual(self.picks(), ["voice", "pacing"])

    def test_a_double_tapped_reject_records_one_verdict_and_keeps_picking(self):
        reject = self.data("draft", self.it["id"], "Reject")
        mid = self.row("draft", self.it["id"])["message_id"]
        self.fake.callback(OWNER, OWNER, reject, mid)
        self.fake.callback(OWNER, OWNER, reject, mid)
        self.s.poll()
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertEqual(self.fake.calls("answerCallbackQuery")[-1]["params"]["text"], "Already done: rejected.")
        self.assertEqual(self.picks(), [])
        self.tap("Voice")
        self.assertEqual(self.picks(), ["voice"])

    def test_a_double_tapped_done_records_once(self):
        self.tap("Reject")
        self.tap("Voice")
        done = self.data("draft", self.it["id"], "Done")
        mid = self.row("draft", self.it["id"])["message_id"]
        self.fake.callback(OWNER, OWNER, done, mid)
        self.fake.callback(OWNER, OWNER, done, mid)
        self.s.poll()
        self.assertEqual(self.counts(), (1, 0, 1))
        self.assertEqual(self.fake.calls("answerCallbackQuery")[-1]["params"]["text"], "Already done: rejected.")
        self.assertEqual(self.ledger_item()["verdict_note"], "reasons: voice")
        self.assertEqual(self.labels(), ["Undo"])

    def test_a_kill_after_done_reached_the_ledger_replays_harmlessly(self):
        self.tap("Reject")
        self.tap("Voice")
        uid = self.fake.callback(OWNER, OWNER, self.data("draft", self.it["id"], "Done"),
                                 self.row("draft", self.it["id"])["message_id"])
        with self.crash_in(phone.Phone, "redraw"), self.assertRaises(Killed):
            self.s.poll()
        self.assertEqual(self.ledger_item()["verdict_tags"], ["voice"], "the ledger has it")
        self.assertEqual(self.s.offset(), uid, "the update is not confirmed")
        self.restart()
        self.s.poll()  # Telegram delivers the same Done again
        self.assertEqual(self.s.offset(), uid + 1)
        self.assertEqual(self.counts(), (1, 0, 1))
        self.assertEqual(self.ledger_item()["verdict_note"], "reasons: voice", "the reasons once")
        self.assertEqual(self.labels(), ["Undo"])

    def test_a_kill_after_the_reject_reached_the_ledger_still_asks_for_the_reasons(self):
        uid = self.fake.callback(OWNER, OWNER, self.data("draft", self.it["id"], "Reject"),
                                 self.row("draft", self.it["id"])["message_id"])
        with self.crash_in(phone.Phone, "redraw"), self.assertRaises(Killed):
            self.s.poll()
        self.assertEqual(self.ledger_item()["state"], "rejected")
        self.assertEqual(self.labels(), ["Approve", "Reject"], "the buttons were never redrawn")
        self.restart()
        with mock.patch.object(cli, "apply_verdict", wraps=cli.apply_verdict) as spy:
            self.s.poll()  # the same press, delivered again
        self.assertEqual(spy.call_count, 0, "nothing is recorded again")
        self.assertEqual(self.s.offset(), uid + 1)
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertIn("What was wrong?", self.fake.calls("answerCallbackQuery")[-1]["params"]["text"])
        self.assertEqual(self.labels(), TAG_LABELS + ["Done", "Undo"])
        self.tap("Voice")
        self.tap("Done")
        self.assertEqual(self.ledger_item()["verdict_tags"], ["voice"])
        # a reject given on the CLI meanwhile is not the phone's: no picking for it
        b = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        self.s.step()
        reject = self.data("draft", b["id"], "Reject")
        self.assertEqual(self.cli("verdict", str(b["job_id"]), "reject")[0], 0)
        ans = self.press(self.s, "draft", b["id"], None, data=reject)
        self.assertIn("changed meanwhile", ans["text"])
        self.assertEqual(self.labels(b), ["Undo"])

    def test_a_kill_after_a_toggle_was_recorded_replays_as_a_stale_tap(self):
        self.tap("Reject")
        uid = self.fake.callback(OWNER, OWNER, self.data("draft", self.it["id"], "Voice"),
                                 self.row("draft", self.it["id"])["message_id"])
        with self.crash_in(phone.BotApi, "edit"), self.assertRaises(Killed):
            self.s.poll()
        self.assertEqual(self.picks(), ["voice"])
        self.restart()
        self.s.poll()
        self.assertEqual(self.s.offset(), uid + 1)
        self.assertEqual(self.picks(), ["voice"], "toggled once, not on and off again")
        self.assertIn(CHECK + "Voice", self.labels(), "the buttons on screen are put back")

    def test_a_busy_database_on_done_records_nothing_half(self):
        self.tap("Reject")
        self.tap("Voice")
        self.fake.callback(OWNER, OWNER, self.data("draft", self.it["id"], "Done"),
                           self.row("draft", self.it["id"])["message_id"])
        with mock.patch.object(cli, "apply_tags", side_effect=sqlite3.OperationalError("database is locked")):
            for _ in range(4):
                self.s.step()
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertEqual(self.picks(), ["voice"], "still picking")
        self.tap("Done")
        self.assertEqual(self.counts(), (1, 0, 1))


# ---------------------------------------------------------------- RT6 unpair

class RT6Unpair(Base):
    def test_a_press_after_unpair_does_nothing(self):
        self.tap("Reject")
        voice = self.data("draft", self.it["id"], "Voice")
        done = self.data("draft", self.it["id"], "Done")
        mid = self.row("draft", self.it["id"])["message_id"]
        code, out = self.cli("phone", "unpair", "--keep-token")
        self.assertEqual(code, 0, out)
        n = len(self.fake.calls("answerCallbackQuery"))
        for data in (voice, done, voice):
            self.fake.callback(OWNER, OWNER, data, mid)
        self.s.poll()
        self.assertEqual(len(self.fake.calls("answerCallbackQuery")), n, "an unpaired chat gets no answer")
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertIsNone(self.ledger_item()["verdict_tags"])
        self.assertIsNone(self.row("draft", self.it["id"]), "the unpairing forgot the messages")
        # paired again (a new key): the old chat's buttons are not valid any more
        self.setup()
        self.fake.message(OWNER, OWNER, f"/start {CODE}")
        self.s.step()
        self.assertTrue(self.s.paired)
        for data in (voice, done):
            ans = self.press(self.s, "draft", self.it["id"], None, data=data, message_id=mid)
            self.assertIn("not valid", ans.get("text", ""))
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertIsNone(self.ledger_item()["verdict_tags"])

    def test_an_unpair_between_the_check_and_done_records_nothing(self):
        self.tap("Reject")
        self.tap("Voice")
        real = self.s._still_bound

        def unpaired_meanwhile(binding):
            self.cli("phone", "unpair", "--keep-token")
            return real(binding)
        with mock.patch.object(self.s, "_still_bound", unpaired_meanwhile):
            self.tap("Done")
        self.assertEqual(self.counts(), (1, 0, 0))
        self.assertIsNone(self.ledger_item()["verdict_tags"])


# ---------------------------------------------------------------- RT7 notes and changes elsewhere

class RT7NotesAndElsewhere(Base):
    def test_reply_notes_before_and_after_the_reject_go_with_the_reasons(self):
        mid = self.row("draft", self.it["id"])["message_id"]
        out = self.say(self.s, "too dark", reply_to=mid)
        self.assertIn("goes with your verdict", out[-1]["params"]["text"])
        self.tap("Reject")
        self.assertEqual(self.ledger_item()["verdict_note"], "too dark")
        out = self.say(self.s, "the hum at 0:04", reply_to=mid)
        self.assertIn("Note added", out[-1]["params"]["text"])
        self.tap("Voice")
        self.tap("Music")
        self.tap("Done")
        it = self.ledger_item()
        self.assertEqual(it["verdict_note"], "too dark | the hum at 0:04 | reasons: voice, music")
        self.assertEqual(it["verdict_tags"], ["voice", "music"])
        out = self.say(self.s, "and too long", reply_to=mid)  # still inside the note window
        self.assertIn("Note added", out[-1]["params"]["text"])
        self.assertEqual(self.ledger_item()["verdict_note"],
                         "too dark | the hum at 0:04 | reasons: voice, music | and too long")

    def test_a_retract_or_pruning_elsewhere_ends_the_picking(self):
        self.tap("Reject")
        voice = self.data("draft", self.it["id"], "Voice")
        self.assertEqual(self.cli("verdict", str(self.it["job_id"]), "retract")[0], 0)
        self.s.step()  # the refresh pass redraws it
        self.assertEqual(self.labels(), ["Approve", "Reject"])
        self.assertIsNone(self.picks())
        ans = self.press(self.s, "draft", self.it["id"], None, data=voice)
        self.assertIn("out of date", ans["text"])
        self.assertEqual(self.counts(), (1, 1, 0))
        # rejected on the phone again, then pruned by retention while picking
        self.tap("Reject")
        voice = self.data("draft", self.it["id"], "Voice")
        with self.store.tx():
            ledger.update_item(self.store, self.it["id"], pruned_at=time.time())
        ans = self.press(self.s, "draft", self.it["id"], None, data=voice)
        self.assertIn("rejected and pruned", ans["text"])
        self.assertEqual(self.fake.buttons(self.msg("draft", self.it["id"])["markup"]), [])
        self.assertEqual(self.row("draft", self.it["id"])["state"], "closed")
        self.assertEqual(self.counts(), (2, 1, 0))

    def test_a_queue_resend_and_the_refresh_pass_leave_the_picking_alone(self):
        b = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        self.s.step()
        self.tap("Reject")
        self.tap("Voice")
        self.say(self.s, "/queue")  # sends b again; the rejected draft is not in the queue
        self.s.step()
        self.assertEqual(self.picks(), ["voice"])
        self.assertIsNone(self.picks(b))
        self.tap("Done")
        self.assertEqual(self.ledger_item()["verdict_tags"], ["voice"])


# ---------------------------------------------------------------- RT8 the CLI

class RT8Cli(Base):
    def test_verdict_reject_with_tags_review_and_report(self):
        b = self.item(files=AUDIO_FILES, template="t.speech", kind="speak")
        c = self.item(template="t.video")
        code, out = self.cli("verdict", str(b["job_id"]), "reject", "--tags", "voice, Lip-sync,b-roll", "--note", "meh")
        self.assertEqual(code, 0, out)
        self.assertIn("reasons: voice, lip-sync, B-roll", out)
        it = ledger.item(self.store, b["id"])
        self.assertEqual(it["verdict_tags"], ["voice", "lipsync", "broll"])
        self.assertEqual(it["verdict_note"], "meh | reasons: voice, lip-sync, B-roll")
        self.assertEqual(self.events("verdict_recorded")[0]["data"]["tags"], ["voice", "lipsync", "broll"])
        for args, why in ((("accept", "--tags", "voice"), "reasons for a reject"),
                          (("reject", "--tags", "voice,hairdo"), "unknown reason tag"),
                          (("reject", "--tags", " , "), "needs one or more"),
                          (("retract", "--tags", "voice"), "reasons for a reject")):
            with self.subTest(args):
                code, out = self.cli("verdict", str(c["job_id"]), *args)
                self.assertEqual(code, 2, out)
                self.assertIn(why, out)
        self.assertEqual(ledger.item(self.store, c["id"])["state"], "review")
        # on the phone too, then reasons added on the CLI to a reject that has none
        self.tap("Reject")
        self.tap("Done")
        code, out = self.cli("verdict", str(self.it["job_id"]), "reject", "--tags", "pacing,voice")
        self.assertEqual(code, 0, out)
        self.assertIn("reasons recorded for its reject", out)
        self.assertEqual(self.ledger_item()["verdict_tags"], ["voice", "pacing"])
        self.assertEqual(len(self.events("verdict_recorded")), 2, "no second verdict")
        code, out = self.cli("verdict", str(self.it["job_id"]), "reject", "--tags", "pacing")  # replaced
        self.assertEqual(self.ledger_item()["verdict_note"], "reasons: pacing")
        code, out = self.cli("verdict", str(self.it["job_id"]), "reject", "--tags", "pacing", "--note", "x")
        self.assertEqual(code, 2, out)
        self.assertIn("already rejected", out)
        code, out = self.cli("verdict", str(self.it["job_id"]), "reject")
        self.assertEqual(code, 2, out)
        self.assertIn("not awaiting a verdict", out)
        code, out = self.cli("review")
        self.assertEqual(code, 0, out)
        self.assertIn("reject reasons by template:", out)
        self.assertIn("t.speech: voice 1, lip-sync 1, B-roll 1, pacing 1 (2 of 2 reject(s) tagged)", out)
        path = os.path.join(self.tmp, "scoreboard.md")
        code, out = self.cli("report", "--out", path)
        self.assertEqual(code, 0, out)
        with open(path) as f:
            text = f.read()
        self.assertIn("## Reject reasons", text)
        self.assertIn("| t.speech | 2 | 2 | voice 1, lip-sync 1, B-roll 1, pacing 1 |", text)
        code, out = self.cli("verdict", str(b["job_id"]), "retract")
        self.assertEqual(code, 0, out)
        it = ledger.item(self.store, b["id"])
        self.assertEqual((it["verdict_tags"], it["verdict_note"]), (None, None))
        code, out = self.cli("review")
        self.assertIn("t.speech: pacing 1 (1 of 1 reject(s) tagged)", out)

    def test_review_and_report_say_nothing_new_without_tags(self):
        code, out = self.cli("review")
        self.assertNotIn("reject reasons", out)
        templates, _ = load_templates(os.path.join(self.home, "templates"))
        text = report.build(self.store, templates, time.time(), time.time())
        self.assertIn("## Reject reasons", text)
        self.assertIn("None.", text.split("## Reject reasons")[1].split("##")[0])


# ---------------------------------------------------------------- RT9 the ledger

class RT9Ledger(Base):
    def test_parse_tags(self):
        self.assertEqual(ledger.parse_tags("Pacing, voice,VOICE, lip sync, end-card, faces/consistency,captions/text"),
                         ["voice", "lipsync", "faces", "captions", "pacing", "endcard"])
        self.assertEqual(ledger.parse_tags(["B-roll", "other"]), ["broll", "other"])
        self.assertEqual(ledger.parse_tags(None), [])
        self.assertEqual(ledger.parse_tags(""), [])
        for bad in ("voice,nope", ["voice", 3], "✓ voice"):
            with self.subTest(bad):
                with self.assertRaises(ledger.LedgerError):
                    ledger.parse_tags(bad)
        self.assertEqual(len(ledger.REASON_TAGS), 13)
        self.assertEqual(set(phone.TAG_CODES.values()), set(ledger.TAG_NAMES))

    def test_counts_per_template_for_the_planner_and_the_reports(self):
        now = time.time()
        a = self.item(files=(), template="t.speech", kind="speak")
        b = self.item(files=(), template="t.speech", kind="speak")
        c = self.item(files=(), template="t.video")
        d = self.item(files=(), template="t.video")
        ledger.record_verdict(self.store, a["job_id"], "reject", None, None, now - 10 * 86400, tags=["voice", "pacing"])
        ledger.record_verdict(self.store, b["job_id"], "reject", None, None, now, tags="voice")
        ledger.record_verdict(self.store, c["job_id"], "reject", None, None, now)
        ledger.record_verdict(self.store, d["job_id"], "reject", None, None, now, tags=["motion"])
        ledger.retract_verdict(self.store, d["job_id"], None, now)
        with self.assertRaises(ledger.LedgerError):
            ledger.record_verdict(self.store, d["job_id"], "accept", None, None, now, tags=["motion"])
        counts = ledger.reject_tag_counts(self.store)
        self.assertEqual(counts["t.speech"], {"rejected": 2, "tagged": 2, "tags": {"voice": 2, "pacing": 1}})
        self.assertEqual(counts["t.video"], {"rejected": 1, "tagged": 0, "tags": {}})
        self.assertEqual(ledger.reject_tag_counts(self.store, since=now - 86400)["t.speech"]["tags"], {"voice": 1})
        self.assertEqual(list(ledger.reject_tag_counts(self.store, template="t.video")), ["t.video"])
        templates, _ = load_templates(os.path.join(self.home, "templates"))
        stats = ledger.template_stats(self.store, templates, now)
        self.assertEqual(stats["t.speech"]["reject_tags"], {"voice": 2, "pacing": 1})
        self.assertNotIn("reject_tags", stats["t.speech"]["7d"], "the planner's ledger windows are unchanged for now")
        self.assertEqual(ledger.tag_summary(counts["t.speech"]), "voice 2, pacing 1 (2 of 2 reject(s) tagged)")

    def test_tag_verdict_is_idempotent_and_only_for_rejects(self):
        now = time.time()
        a = self.item(files=(), template="t.speech", kind="speak")
        with self.assertRaises(ledger.LedgerError):
            ledger.tag_verdict(self.store, a["job_id"], ["voice"], now)
        ledger.record_verdict(self.store, a["job_id"], "accept", None, None, now)
        with self.assertRaises(ledger.LedgerError):
            ledger.tag_verdict(self.store, a["job_id"], ["voice"], now)
        ledger.retract_verdict(self.store, a["job_id"], None, now)
        ledger.record_verdict(self.store, a["job_id"], "reject", None, "flat", now)
        for _ in range(3):
            it = ledger.tag_verdict(self.store, a["job_id"], ["voice"], now, by="phone")
        self.assertEqual(len(self.events("verdict_tagged")), 1)
        self.assertEqual(it["verdict_note"], "flat | reasons: voice")
        it = ledger.tag_verdict(self.store, a["job_id"], ["music", "voice"], now)
        self.assertEqual(it["verdict_note"], "flat | reasons: voice, music")
        it = ledger.tag_verdict(self.store, a["job_id"], [], now)
        self.assertEqual((it["verdict_tags"], it["verdict_note"]), (None, "flat"))
        with self.store.tx():  # a corrupt stored value reads as no tags, never raises
            self.store.conn.execute("UPDATE plan_items SET verdict_tags = '{\"x\": 1' WHERE id = ?", (a["id"],))
        self.assertIsNone(ledger.item(self.store, a["id"])["verdict_tags"])

    def test_an_old_database_gains_the_columns(self):
        path = os.path.join(self.tmp, "old.sqlite")
        old_schema = "\n".join(line for line in store_mod.SCHEMA.splitlines()
                               if not line.lstrip().startswith(("verdict_tags ", "tags ")))
        self.assertNotIn("verdict_tags", old_schema)
        conn = sqlite3.connect(path)
        conn.executescript(old_schema)
        conn.execute("INSERT INTO plan_items (template_id, consumer, brief, content_hash, state, created_at, verdict) "
                     "VALUES ('t.speech', 'x', 'y', 'h-old', 'rejected', 1.0, 'reject')")
        conn.commit()
        conn.close()
        st = Store(path)
        self.addCleanup(st.close)
        cols = {r["name"] for r in st.conn.execute("PRAGMA table_info(plan_items)")}
        self.assertIn("verdict_tags", cols)
        self.assertIn("tags", {r["name"] for r in st.conn.execute("PRAGMA table_info(phone_messages)")})
        self.assertIsNone(ledger.items(st)[0]["verdict_tags"])
        self.assertEqual(ledger.reject_tag_counts(st), {"t.speech": {"rejected": 1, "tagged": 0, "tags": {}}})


if __name__ == "__main__":
    unittest.main()
