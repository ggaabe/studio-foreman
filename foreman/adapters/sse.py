"""Incremental Server-Sent-Events reader that never holds a huge event in memory.

mlx-serve ends a media stream with one `data: {"type":"complete",...}` line that carries the
whole result as base64: a WAV for audio, raw rgb8 frames plus pcm_s16le audio for video (about
440 MB of text for a 5 s 720p clip). Small events (progress, error) are buffered and parsed with
json.loads. An event line that grows past `small_limit` bytes is parsed on the fly instead, as
a flat JSON object: scalar fields are collected, and string fields the caller asks to stream
(`data`, `audio_data`) go straight into a sink, typically a base64 decoder writing to a file.
"""
from __future__ import annotations

import binascii
import json

SMALL_LIMIT = 256 * 1024
MAX_SMALL_STRING = 1 << 20

_WS = b" \t\r\n"
_ESCAPES = {ord("/"): b"/", ord('"'): b'"', ord("\\"): b"\\", ord("n"): b"\n", ord("t"): b"\t",
            ord("r"): b"\r", ord("b"): b"\b", ord("f"): b"\f"}


class SseError(ValueError):
    pass


class Base64FileSink:
    """Decodes base64 text in arbitrary pieces and appends the bytes to a file."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "wb")
        self._left = b""
        self.nbytes = 0

    def write(self, text: bytes) -> None:
        data = self._left + text
        cut = len(data) // 4 * 4
        if cut:
            out = binascii.a2b_base64(data[:cut])
            self._f.write(out)
            self.nbytes += len(out)
        self._left = data[cut:]

    def close(self) -> None:
        if self._f.closed:
            return
        try:
            if self._left.strip(b"="):
                raise SseError(f"{self.path}: base64 payload ends mid-quantum ({len(self._left)} chars left)")
        finally:
            self._f.close()

    def abort(self) -> None:
        if not self._f.closed:
            self._f.close()


class FlatJsonStream:
    """Parses one flat JSON object fed in pieces. `open_sink(key, scalars)` may return a sink
    for a string-valued key (scalars holds the fields parsed so far); other strings are kept,
    up to MAX_SMALL_STRING bytes. Nested objects or arrays raise SseError."""

    def __init__(self, open_sink):
        self.open_sink = open_sink
        self.fields: dict = {}
        self.streamed: dict = {}
        self._state = "start"
        self._key = bytearray()
        self._val = bytearray()
        self._sink = None
        self._esc = None  # pending escape bytes inside a streamed string
        self._skip = False  # a backslash ended the previous piece of a kept string or key

    def _scan(self, data: bytes, i: int, n: int) -> int:
        """Index of the closing quote of a kept string from i, or -1 if it is not in this piece.
        Escapes are skipped, including one split across two pieces."""
        j = i
        if self._skip:
            self._skip = False
            j += 1
        while j < n:
            c = data[j]
            if c == 0x5C:
                if j + 1 >= n:
                    self._skip = True
                    return -1
                j += 2
                continue
            if c == 0x22:
                return j
            j += 1
        return -1

    @property
    def done(self) -> bool:
        return self._state == "done"

    def feed(self, data: bytes) -> None:
        i, n = 0, len(data)
        while i < n:
            st = self._state
            if st == "done":
                if data[i:].strip(_WS):
                    raise SseError("trailing data after the JSON object")
                return
            if st in ("start", "key_or_end", "colon", "value", "after_value", "key_next"):
                c = data[i]
                if c in _WS:
                    i += 1
                    continue
                if st == "start":
                    if c != 0x7B:
                        raise SseError("event is not a JSON object")
                    self._state = "key_or_end"
                elif st in ("key_or_end", "key_next"):
                    if c == 0x7D and st == "key_or_end":
                        self._state = "done"
                    elif c == 0x22:
                        self._key = bytearray()
                        self._state = "key"
                    else:
                        raise SseError(f"expected a key, got {chr(c)!r}")
                elif st == "colon":
                    if c != 0x3A:
                        raise SseError("expected ':'")
                    self._state = "value"
                elif st == "value":
                    if c == 0x22:
                        key = self._key.decode("utf-8", "replace")
                        self._sink = self.open_sink(key, self.fields) if self.open_sink else None
                        self._val = bytearray()
                        self._esc = None
                        self._state = "str"
                    elif c in (0x7B, 0x5B):
                        raise SseError("nested values are not supported in a streamed event")
                    else:
                        self._val = bytearray([c])
                        self._state = "scalar"
                elif st == "after_value":
                    if c == 0x2C:
                        self._state = "key_next"
                    elif c == 0x7D:
                        self._state = "done"
                    else:
                        raise SseError(f"expected ',' or '}}', got {chr(c)!r}")
                i += 1
                continue
            if st == "key":
                j = self._scan(data, i, n)
                if j < 0:
                    self._key += data[i:n]
                    return
                self._key += data[i:j]
                self._key = bytearray(json.loads(b'"' + bytes(self._key) + b'"').encode("utf-8"))
                self._state = "colon"
                i = j + 1
                continue
            if st == "scalar":
                j = i
                while j < n and data[j] not in b",}" and data[j] not in _WS:
                    j += 1
                self._val += data[i:j]
                if j >= n:
                    return
                self.fields[self._key.decode("utf-8")] = json.loads(bytes(self._val))
                self._state = "after_value"
                i = j
                continue
            if st == "str":
                i = self._string(data, i, n)
                continue
            raise SseError(f"bad parser state {st}")  # pragma: no cover

    def _string(self, data: bytes, i: int, n: int) -> int:
        sink = self._sink
        if sink is None:
            # Small string: keep the raw (escaped) bytes and let json decode them at the end.
            j = self._scan(data, i, n)
            if j < 0:
                self._val += data[i:n]
                if len(self._val) > MAX_SMALL_STRING:
                    raise SseError(f"field {self._key.decode()!r} is too large to keep in memory")
                return n
            self._val += data[i:j]
            self.fields[self._key.decode("utf-8")] = json.loads(b'"' + bytes(self._val) + b'"')
            self._state = "after_value"
            return j + 1
        while i < n:
            if self._esc is not None:
                self._esc += data[i:i + 1]
                i += 1
                lead = self._esc[0]
                if lead == ord("u"):
                    if len(self._esc) < 5:
                        continue
                    sink.write(chr(int(self._esc[1:5], 16)).encode("utf-8"))
                elif lead in _ESCAPES:
                    sink.write(_ESCAPES[lead])
                else:
                    raise SseError(f"bad escape \\{chr(lead)}")
                self._esc = None
                continue
            q = data.find(b'"', i)
            b = data.find(b"\\", i)
            stop = min(x for x in (q, b, n) if x >= 0)
            if stop > i:
                sink.write(data[i:stop])
            if stop == n:
                return n
            if stop == b:
                self._esc = bytearray()
                i = stop + 1
                continue
            # closing quote
            key = self._key.decode("utf-8")
            sink.close()
            self.streamed[key] = sink
            self._sink = None
            self._state = "after_value"
            return stop + 1
        return n

    def abort(self) -> None:
        if self._sink is not None:
            self._sink.abort()
        for s in self.streamed.values():
            try:
                s.abort()
            except Exception:
                pass


class SseReader:
    """Feed raw bytes; get callbacks per event.

    on_event(fields: dict, streamed: dict) for each `data:` line (streamed maps keys to the
    sinks that received them; it is empty for small events), on_comment(text) for `:` lines
    (mlx-serve's keepalive). `open_sink(first_bytes)` returns a sink factory for one big event,
    or None to refuse it (then SseError is raised)."""

    def __init__(self, on_event, on_comment=None, open_sink=None, small_limit: int = SMALL_LIMIT):
        self.on_event, self.on_comment, self.open_sink = on_event, on_comment, open_sink
        self.small_limit = small_limit
        self._line = bytearray()
        self._mode = "line"      # line | data | big | skip
        self._stream = None
        self.bytes_seen = 0

    def feed(self, chunk: bytes) -> None:
        self.bytes_seen += len(chunk)
        i, n = 0, len(chunk)
        while i < n:
            nl = chunk.find(b"\n", i)
            end = n if nl < 0 else nl
            piece = chunk[i:end]
            if self._mode == "big":
                self._stream.feed(piece)
            else:
                self._line += piece
                if self._mode == "line" and len(self._line) >= 5:
                    self._mode = "data" if self._line.startswith(b"data:") else "skip"
                if self._mode == "data" and len(self._line) > self.small_limit:
                    self._go_big()
            if nl < 0:
                return
            self._end_line()
            i = nl + 1

    def _go_big(self) -> None:
        payload = bytes(self._line[5:]).lstrip()
        factory = self.open_sink(payload[:256]) if self.open_sink else None
        self._stream = FlatJsonStream(factory)
        self._mode = "big"
        self._line = bytearray()
        self._stream.feed(payload)

    def _end_line(self) -> None:
        mode, line = self._mode, bytes(self._line).rstrip(b"\r")
        self._mode, self._line = "line", bytearray()
        if mode == "big":
            stream, self._stream = self._stream, None
            if not stream.done:
                stream.abort()
                raise SseError("big event ended before its JSON object closed")
            self.on_event(stream.fields, stream.streamed)
            return
        if not line:
            return
        if line.startswith(b":"):
            if self.on_comment:
                self.on_comment(line[1:].decode("utf-8", "replace").strip())
            return
        if line.startswith(b"data:"):
            text = line[5:].strip()
            if text == b"[DONE]":
                self.on_event({"type": "[DONE]"}, {})
                return
            try:
                obj = json.loads(text.decode("utf-8"))
            except ValueError as e:
                raise SseError(f"unparseable event: {text[:200]!r} ({e})") from None
            self.on_event(obj if isinstance(obj, dict) else {"value": obj}, {})
        # other SSE fields (event:, id:, retry:) carry nothing mlx-serve uses

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.abort()
