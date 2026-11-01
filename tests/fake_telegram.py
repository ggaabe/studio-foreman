"""A local stand-in for the Telegram Bot API (stdlib http.server on 127.0.0.1), for tests/test_phone.py.

It answers the methods foreman/phone.py uses the way Telegram does: POST /bot<token>/<method>
with a JSON or multipart/form-data body (parsed with the standard library's email parser, an
encoder-independent check of the hand-written multipart), {"ok": true, "result": ...} or
{"ok": false, "error_code", "description", "parameters": {"retry_after"}}. A wrong token is
401 Unauthorized. getUpdates confirms everything below `offset`, returns what is left and, with
a timeout, waits (at most `max_wait` seconds) for the next update. Tests queue updates with
message() / callback(), script failures with fail(), and drop connections with `offline`.
Nothing leaves 127.0.0.1.
"""
from __future__ import annotations

import email.parser
import email.policy
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_multipart(ctype: str, body: bytes) -> tuple:
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"MIME-Version: 1.0\r\nContent-Type: " + ctype.encode() + b"\r\n\r\n" + body)
    params, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        data = part.get_payload(decode=True) or b""
        if filename is not None:
            files[name] = {"filename": filename, "content_type": part.get_content_type(), "data": data}
        else:
            params[name] = data.decode("utf-8")
    return params, files


class FakeTelegram:
    def __init__(self, token: str, username: str = "foreman_test_bot", bot_id: int = 777000):
        self.token, self.username, self.bot_id = token, username, bot_id
        self.requests: list = []          # every authorised call: {"method", "params", "files"}
        self.raw_paths: list = []         # request paths as received (they carry the token, like Telegram's)
        self.updates: list = []
        self.next_update_id = 5000
        self.next_message_id = 100
        self.messages: dict = {}          # message_id -> what the bot sent (and edited)
        self.failures: dict = {}          # method -> [(status, description, retry_after)]
        self.offline = False              # drop every connection without an answer
        self.max_wait = 2.0
        self.cond = threading.Condition()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                fake._serve(self)

            do_GET = do_POST

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        with self.cond:
            self.cond.notify_all()
        self.server.shutdown()
        self.server.server_close()

    # ---- test controls
    def push(self, update: dict) -> int:
        with self.cond:
            self.next_update_id += 1
            update = dict(update, update_id=self.next_update_id)
            self.updates.append(update)
            self.cond.notify_all()
            return self.next_update_id

    def message(self, chat_id: int, user_id: int, text: str, chat_type: str = "private", reply_to=None,
                is_bot: bool = False) -> int:
        with self.cond:
            self.next_message_id += 1
            mid = self.next_message_id
        msg = {"message_id": mid, "date": int(time.time()), "text": text,
               "chat": {"id": chat_id, "type": chat_type, "first_name": "Owner"},
               "from": {"id": user_id, "is_bot": is_bot, "first_name": "Owner"}}
        if reply_to is not None:
            msg["reply_to_message"] = {"message_id": reply_to, "chat": msg["chat"], "date": 0}
        return self.push({"message": msg})

    def callback(self, chat_id: int, user_id: int, data, message_id: int, chat_type: str = "private") -> int:
        cq = {"id": f"cq{self.next_update_id + 1}", "chat_instance": "x", "data": data,
              "from": {"id": user_id, "is_bot": False, "first_name": "Owner"},
              "message": {"message_id": message_id, "date": 0, "chat": {"id": chat_id, "type": chat_type}}}
        return self.push({"callback_query": cq})

    def fail(self, method: str, status: int, description: str, retry_after=None, times: int = 1) -> None:
        self.failures.setdefault(method, []).extend([(status, description, retry_after)] * times)

    def calls(self, *methods) -> list:
        return [r for r in self.requests if not methods or r["method"] in methods]

    def sends(self) -> list:
        return self.calls("sendMessage", "sendVideo", "sendAudio", "sendDocument")

    def last(self, *methods) -> dict:
        got = self.calls(*methods)
        return got[-1] if got else {}

    @staticmethod
    def buttons(markup) -> list:
        """[(text, callback_data)] of an inline keyboard (a dict, or its JSON in a multipart field)."""
        if isinstance(markup, str):
            markup = json.loads(markup)
        return [(b["text"], b["callback_data"]) for row in (markup or {}).get("inline_keyboard", []) for b in row]

    # ---- serving
    def _reply(self, h, status: int, obj) -> None:
        body = json.dumps(obj).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def _serve(self, h) -> None:
        length = int(h.headers.get("Content-Length") or 0)
        body = h.rfile.read(length) if length else b""
        if self.offline:
            h.close_connection = True
            try:
                h.connection.shutdown(2)
            except OSError:
                pass
            return
        self.raw_paths.append(h.path)
        m = re.match(r"^/bot([^/]+)/(\w+)$", h.path)
        if not m:
            return self._reply(h, 404, {"ok": False, "error_code": 404, "description": "Not Found"})
        token, method = m.group(1), m.group(2)
        if token != self.token:
            return self._reply(h, 401, {"ok": False, "error_code": 401, "description": "Unauthorized"})
        ctype = h.headers.get("Content-Type", "")
        files = {}
        if ctype.startswith("multipart/form-data"):
            params, files = parse_multipart(ctype, body)
        else:
            params = json.loads(body.decode() or "{}")
        self.requests.append({"method": method, "params": params, "files": files})
        queue = self.failures.get(method)
        if queue:
            status, desc, retry = queue.pop(0)
            obj = {"ok": False, "error_code": status, "description": desc}
            if retry is not None:
                obj["parameters"] = {"retry_after": retry}
            return self._reply(h, status, obj)
        handler = getattr(self, "m_" + method, None)
        if handler is None:
            return self._reply(h, 404, {"ok": False, "error_code": 404, "description": "Not Found: method not found"})
        try:
            result = handler(params, files)
        except _ApiError as e:
            return self._reply(h, e.status, {"ok": False, "error_code": e.status, "description": e.description})
        self._reply(h, 200, {"ok": True, "result": result})

    # ---- methods
    def m_getMe(self, p, f):
        return {"id": self.bot_id, "is_bot": True, "first_name": "Foreman", "username": self.username}

    def m_getUpdates(self, p, f):
        offset, timeout = int(p.get("offset") or 0), float(p.get("timeout") or 0)
        limit = int(p.get("limit") or 100)
        deadline = time.monotonic() + min(timeout, self.max_wait)
        with self.cond:
            while True:
                if offset:
                    self.updates = [u for u in self.updates if u["update_id"] >= offset]
                if self.updates or time.monotonic() >= deadline:
                    return [dict(u) for u in self.updates[:limit]]
                self.cond.wait(max(0.0, deadline - time.monotonic()))

    def _send(self, p, f, kind: str):
        chat = p.get("chat_id")
        if chat in (None, ""):
            raise _ApiError(400, "Bad Request: chat_id is empty")
        with self.cond:
            self.next_message_id += 1
            mid = self.next_message_id
        markup = p.get("reply_markup")
        if isinstance(markup, str):
            markup = json.loads(markup)
        msg = {"message_id": mid, "date": int(time.time()), "chat": {"id": int(chat), "type": "private"}}
        if kind == "text":
            if len(str(p.get("text") or "")) > 4096:
                raise _ApiError(400, "Bad Request: message is too long")
            msg["text"] = p.get("text")
        else:
            if len(str(p.get("caption") or "")) > 1024:
                raise _ApiError(400, "Bad Request: message caption is too long")
            msg["caption"] = p.get("caption")
            field = next(iter(f.values()), None)
            if field is None or not field["data"]:
                raise _ApiError(400, "Bad Request: file must be non-empty")
            msg[kind] = {"file_id": f"file{mid}", "file_size": len(field["data"]), "file_name": field["filename"]}
        self.messages[mid] = {"kind": kind, "text": msg.get("text") or msg.get("caption"), "markup": markup,
                              "params": p, "files": f}
        return msg

    def m_sendMessage(self, p, f):
        return self._send(p, f, "text")

    def m_sendVideo(self, p, f):
        return self._send(p, f, "video")

    def m_sendAudio(self, p, f):
        return self._send(p, f, "audio")

    def m_sendDocument(self, p, f):
        return self._send(p, f, "document")

    def m_answerCallbackQuery(self, p, f):
        return True

    def _edit(self, p, text_key=None):
        mid = int(p.get("message_id") or 0)
        cur = self.messages.get(mid)
        if cur is None:
            raise _ApiError(400, "Bad Request: message to edit not found")
        markup = p.get("reply_markup")
        new_text = p.get(text_key) if text_key else cur["text"]
        if new_text == cur["text"] and markup == cur["markup"]:
            raise _ApiError(400, "Bad Request: message is not modified: specified new message content and reply "
                                 "markup are exactly the same as a current content and reply markup of the message")
        cur["text"], cur["markup"] = new_text, markup
        return {"message_id": mid}

    def m_editMessageText(self, p, f):
        if self.messages.get(int(p.get("message_id") or 0), {}).get("kind") not in (None, "text"):
            raise _ApiError(400, "Bad Request: there is no text in the message to edit")
        return self._edit(p, "text")

    def m_editMessageCaption(self, p, f):
        if self.messages.get(int(p.get("message_id") or 0), {}).get("kind") == "text":
            raise _ApiError(400, "Bad Request: there is no caption in the message to edit")
        return self._edit(p, "caption")

    def m_editMessageReplyMarkup(self, p, f):
        return self._edit(p)


class _ApiError(Exception):
    def __init__(self, status: int, description: str):
        self.status, self.description = status, description
