"""Small HTTP helpers with a hard client-side timeout on every call.

Two shapes, both standard library only:

  * `json_call()` is urllib with an explicit timeout and proxies disabled, for the short calls
    gpud makes from inside tick() (GET /api/ps, POST /v1/unload-model, ...). A call that hangs
    raises after `timeout` seconds; gpud records that as "engine unreachable", never a stall.
  * `StreamConn` is a raw-socket HTTP/1.1 client for the long calls (model loads, SSE media
    renders, streaming chat). They run on worker threads, read with a short socket timeout so
    the thread can notice a cancel, and `close()` from any thread aborts them at once. Closing
    the connection is also how mlx-serve is told to stop a generation: it has no cancel
    endpoint and aborts at the next step boundary when the client disconnects.

http.client's file-object reader is unusable after one read timeout ("cannot read from timed
out object"), which is why the streaming side talks to the socket directly.
"""
from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HttpError(Exception):
    """A non-2xx answer. `message` is the server's error text when it sent one."""

    def __init__(self, status: int, message: str, body: bytes = b""):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.message, self.body = status, message, body


class Cancelled(Exception):
    """The connection was closed on purpose (a cancel) while a call was reading it."""


class Silence(Exception):
    """No byte arrived for longer than the caller's silence budget."""


def error_message(body: bytes) -> str:
    """The `error.message` of an OpenAI-style error body, else a short excerpt."""
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return body[:300].decode("utf-8", "replace").strip() or "no body"
    if isinstance(obj, dict):
        err = obj.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(err, str):
            return err
        if isinstance(obj.get("message"), str):
            return obj["message"]
    return json.dumps(obj)[:300]


def json_call(base_url: str, method: str, path: str, body=None, timeout: float = 5.0):
    """(status, decoded JSON or None). Raises HttpError for a non-2xx status and OSError (or
    a subclass, including socket.timeout) when the server cannot be reached in time."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read(64 << 20)
            status = resp.status
    except urllib.error.HTTPError as e:
        try:
            raw = e.read(1 << 20)
        except Exception:
            raw = b""
        raise HttpError(e.code, error_message(raw), raw) from None
    except urllib.error.URLError as e:
        reason = e.reason if isinstance(e.reason, BaseException) else OSError(str(e.reason))
        raise reason from None
    if not raw:
        return status, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise ValueError(f"{method} {path}: response is not JSON ({e})") from None


def text_call(base_url: str, path: str, timeout: float = 2.0, limit: int = 1 << 20) -> tuple:
    """GET a plain-text resource (a Prometheus /metrics page): (status, text). A non-2xx answer
    is returned as its status with the body text, not raised; OSError when unreachable in time."""
    req = urllib.request.Request(base_url.rstrip("/") + path, method="GET", headers={"Accept": "text/plain"})
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.status, resp.read(limit).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            raw = e.read(1 << 16)
        except Exception:
            raw = b""
        return e.code, raw.decode("utf-8", "replace")
    except urllib.error.URLError as e:
        reason = e.reason if isinstance(e.reason, BaseException) else OSError(str(e.reason))
        raise reason from None


def split_url(base_url: str):
    u = urllib.parse.urlsplit(base_url)
    if u.scheme != "http":
        raise ValueError(f"only plain http:// is supported, got {base_url!r}")
    return u.hostname or "127.0.0.1", u.port or 80, (u.path or "").rstrip("/")


class StreamConn:
    """One HTTP/1.1 request over a raw socket. Thread-safe close(); everything else belongs to
    the thread that created it."""

    def __init__(self, base_url: str, connect_timeout: float = 5.0, poll_s: float = 1.0):
        self.host, self.port, self.prefix = split_url(base_url)
        self.connect_timeout, self.poll_s = connect_timeout, poll_s
        self._sock = None
        self._lock = threading.Lock()
        self._closed = False
        self._buf = b""
        self.status = None
        self.headers: dict = {}
        self.sent = False  # the whole request reached the socket (the server may act on it)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with self._lock:
            self._closed = True
            sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
                header_timeout: float | None = None) -> int:
        """Send the request and read the status line and headers. Returns the status."""
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        with self._lock:
            if self._closed:
                sock.close()
                raise Cancelled("closed before the request was sent")
            self._sock = sock
        sock.settimeout(self.poll_s)
        lines = [f"{method} {self.prefix}{path} HTTP/1.1", f"Host: {self.host}:{self.port}",
                 "Connection: close", "Accept: */*"]
        for k, v in (headers or {}).items():
            lines.append(f"{k}: {v}")
        if body is not None:
            lines.append(f"Content-Length: {len(body)}")
        payload = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + (body or b"")
        try:
            sock.settimeout(max(self.connect_timeout, 5.0))
            sock.sendall(payload)
            self.sent = True
            sock.settimeout(self.poll_s)
        except OSError as e:
            if self._closed:
                raise Cancelled("closed while sending") from None
            raise e
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self._recv(header_timeout)
            if not chunk:
                raise ConnectionError("server closed the connection before sending headers")
            head += chunk
            if len(head) > 1 << 16 and b"\r\n\r\n" not in head:  # the body may arrive in the same read
                raise ConnectionError("response headers too large")
        head, _, self._buf = head.partition(b"\r\n\r\n")
        status_line, *header_lines = head.decode("latin-1").split("\r\n")
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise ConnectionError(f"bad status line {status_line!r}")
        self.status = int(parts[1])
        for line in header_lines:
            k, _, v = line.partition(":")
            self.headers[k.strip().lower()] = v.strip()
        return self.status

    def _recv(self, silence_s: float | None = None) -> bytes:
        """Next bytes from the socket; b'' at EOF. Raises Cancelled after close(), Silence when
        nothing arrived for silence_s seconds."""
        waited = 0.0
        while True:
            sock = self._sock
            if sock is None or self._closed:
                raise Cancelled("connection closed")
            try:
                return sock.recv(1 << 20)
            except socket.timeout:
                waited += self.poll_s
                if silence_s is not None and waited >= silence_s:
                    raise Silence(f"no data for {waited:.0f}s") from None
            except OSError:
                if self._closed:
                    raise Cancelled("connection closed") from None
                raise

    def body_chunks(self, silence_s: float | None = None):
        """Yield the response body (identity, content-length or chunked) as it arrives."""
        te = self.headers.get("transfer-encoding", "").lower()
        length = self.headers.get("content-length")
        if "chunked" in te:
            yield from self._chunked(silence_s)
            return
        remaining = int(length) if length is not None and length.isdigit() else None
        buf, self._buf = self._buf, b""
        while True:
            if buf:
                if remaining is not None:
                    buf = buf[:remaining]
                    remaining -= len(buf)
                yield buf
            if remaining is not None and remaining <= 0:
                return
            buf = self._recv(silence_s)
            if not buf:
                return

    def _chunked(self, silence_s):
        buf, self._buf = self._buf, b""

        def need(n_or_line):
            nonlocal buf
            while True:
                if n_or_line is None:
                    i = buf.find(b"\r\n")
                    if i >= 0:
                        line, buf = buf[:i], buf[i + 2:]
                        return line
                elif len(buf) >= n_or_line:
                    out, buf = buf[:n_or_line], buf[n_or_line:]
                    return out
                more = self._recv(silence_s)
                if not more:
                    raise ConnectionError("connection closed inside a chunked body")
                buf += more

        while True:
            size = int(need(None).split(b";")[0].strip() or b"0", 16)
            if size == 0:
                return
            remaining = size
            while remaining:
                if not buf:
                    more = self._recv(silence_s)
                    if not more:
                        raise ConnectionError("connection closed inside a chunk")
                    buf += more
                piece, buf = buf[:remaining], buf[remaining:]
                remaining -= len(piece)
                yield piece
            need(2)

    def read_body(self, limit: int = 1 << 20, silence_s: float | None = 30.0) -> bytes:
        out = bytearray()
        for chunk in self.body_chunks(silence_s):
            out += chunk
            if len(out) >= limit:
                break
        return bytes(out[:limit])
