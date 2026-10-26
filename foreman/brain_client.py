"""BrainClient: the real brain behind gpud's brain jobs, over mlx-serve's chat completions.

Contract (the one ScriptedBrain in sim.py implements): `brain(spec, model[, ctx]) -> decision`,
where spec.payload["brief"] is the resume brief (the brain keeps no state of its own) and the
decision is what Supervisor._apply_decision reads: {"verdict", "note", "submit": [requests]}.

The model is never asked to write job requests itself. It answers a small strict JSON schema
(verdict, note, optional parameter changes) and this module builds any resubmission from the
brief, so a model that hallucinates a field cannot create a malformed or unbounded job:

  accept        the output matches the goal; nothing more to do
  retry         resubmit the attempted action (optionally with whitelisted changes, e.g. a new
                prompt or seed) as a retry of the same task; refused past max_attempts
  give_up       the failure is permanent; nothing is resubmitted
  human_review  unsure, or something looks wrong: gpud pages a person

Malformed output, an unknown verdict, or a retry that cannot be built all become
human_review with a note saying why. Transport errors (server gone, timeout) raise, so the
supervisor retries the brain job on its normal ladder.

For review jobs whose artifact has frames, 2-4 sampled PNG frames are attached as image_url
data URIs when the chosen brain has vision; audio reviews attach the waveform PNG and the
measured WAV statistics.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re

from . import media
from .adapters.sse import SseReader
from .httpio import Cancelled, error_message, json_call
from .policy import number

log = logging.getLogger("foreman.brain")

VERDICTS = ("accept", "retry", "give_up", "human_review")
# Per model family, the request fields the brain may change on a retry, and their types.
CHANGES = {
    "tts": {"input": str, "voice": str, "speed": float, "seed": int},
    "music": {"prompt": str, "lyrics": str, "instrumental": bool, "seed": int, "bpm": int, "keyscale": str},
    "video": {"prompt": str, "seed": int, "steps": int},
}
MAX_TEXT = 4000

SYSTEM = """You are the deciding brain of Foreman, a supervisor that runs GPU media jobs (speech, \
music, video) on a Mac Studio. You are stateless: the resume brief is everything you know about \
the task. Deterministic validators already measured the output; trust their numbers.

Answer with exactly one JSON object and nothing else:
{"verdict": "accept" | "retry" | "give_up" | "human_review",
 "note": "<one or two sentences explaining the decision>",
 "changes": {<optional: parameters to change on a retry>}}

Rules:
- accept: the job succeeded and the output plausibly matches the goal.
- retry: only when attempt < max_attempts and running it again (optionally with changes) could \
plausibly fix the problem. Allowed change keys: CHANGE_KEYS.
- give_up: the failure is permanent (bad request, impossible goal) or retries are exhausted.
- human_review: you are unsure, the output looks wrong in a way a retry will not fix, or \
anything looks unsafe.
Never invent file paths, models or parameters that are not listed."""


def _png_data_uri(path: str, max_bytes: int = 2 << 20) -> str | None:
    try:
        if os.path.getsize(path) > max_bytes:
            return None
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _family_of(catalog, model) -> str | None:
    spec = catalog.get(model) if catalog else None
    return spec.family if spec else None


def parse_json_object(text: str):
    """The first JSON object in a model reply (tolerates code fences and stray prose)."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    start = t.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except ValueError:
                        break
        start = t.find("{", start + 1)
    return None


def _clean_changes(changes, family: str | None) -> tuple:
    """(accepted changes, list of dropped keys)."""
    allowed = CHANGES.get(family or "", {})
    out, dropped = {}, []
    if not isinstance(changes, dict):
        return out, ([] if changes in (None, {}) else ["<not an object>"])
    for k, v in changes.items():
        typ = allowed.get(k)
        if typ is None:
            dropped.append(k)
            continue
        if typ is str and isinstance(v, str) and v.strip() and len(v) <= MAX_TEXT:
            out[k] = v
        elif typ is bool and isinstance(v, bool):
            out[k] = v
        elif typ in (int, float) and number(v) is not None:
            out[k] = int(v) if typ is int else float(v)
        else:
            dropped.append(k)
    return out, dropped


def decide(reply: dict | None, brief: dict, catalog=None, raw: str = "") -> dict:
    """Turn the model's reply into a supervisor decision. Never raises."""
    try:
        return _decide(reply, brief, catalog, raw)
    except Exception as e:  # defensive: a bug here must still yield a safe decision
        log.exception("decision construction failed")
        return {"verdict": "human_review", "note": f"brain client error: {type(e).__name__}: {e}", "submit": []}


def _decide(reply, brief, catalog, raw):
    if not isinstance(reply, dict):
        snippet = (raw or "").strip().replace("\n", " ")[:200]
        return {"verdict": "human_review", "note": f"brain output was not a JSON object: {snippet!r}", "submit": []}
    verdict = reply.get("verdict")
    note = reply.get("note") if isinstance(reply.get("note"), str) else ""
    note = note.strip()[:600]
    if verdict not in VERDICTS:
        return {"verdict": "human_review", "note": f"brain returned an unknown verdict {verdict!r}: {note}"[:600],
                "submit": []}
    if verdict != "retry":
        return {"verdict": verdict, "note": note, "submit": []}
    brief = brief if isinstance(brief, dict) else {}
    act = brief.get("attempted_action") if isinstance(brief.get("attempted_action"), dict) else {}
    attempt, max_attempts = number(brief.get("attempt")), number(brief.get("max_attempts"))
    if not act.get("kind") or not act.get("model") or not isinstance(act.get("on_success"), dict) \
            or not isinstance(act.get("on_failure"), dict):
        return {"verdict": "human_review", "note": f"brain wanted a retry but the brief cannot rebuild the job: {note}",
                "submit": []}
    if attempt is not None and max_attempts is not None and attempt >= max_attempts:
        return {"verdict": "human_review",
                "note": f"brain wanted a retry but attempts are used up ({attempt:g}/{max_attempts:g}): {note}",
                "submit": []}
    changes, dropped = _clean_changes(reply.get("changes"), _family_of(catalog, act["model"]))
    params = dict(act.get("params") or {})
    params.update(changes)
    params["goal"] = brief.get("goal")
    request = {"kind": act["kind"], "requires": act["model"], "payload": params,
               "on_success": act["on_success"], "on_failure": act["on_failure"]}
    if isinstance(brief.get("source_job_id"), int):
        request["retry_of"] = brief["source_job_id"]
    extra = f" (changed: {sorted(changes)})" if changes else ""
    if dropped:
        extra += f" (ignored changes: {sorted(dropped)})"
    return {"verdict": "retry", "note": (note + extra)[:700], "submit": [request]}


class BrainClient:
    """Callable brain. With a BrainContext (passed by MlxServeEngine) it streams the chat over a
    connection the run can cancel and reports token progress so the lease renews."""

    wants_context = True

    def __init__(self, base_url: str = "http://127.0.0.1:11234", catalog=None, max_tokens: int = 600,
                 temperature: float = 0.2, timeout_s: float = 140.0, frames: int = 3, server_ids=None):
        self.base_url = base_url.rstrip("/")
        self.catalog = catalog or {}
        self.max_tokens, self.temperature, self.timeout_s = max_tokens, temperature, timeout_s
        self.frames = max(0, min(4, frames))
        self.server_ids = server_ids or {}
        self.transcript: list = []   # last few prompts/replies, for debugging (bounded)

    # ---- the contract
    def __call__(self, spec, model: str, ctx=None) -> dict:
        payload = spec.payload if isinstance(spec.payload, dict) else {}
        brief = payload.get("brief") if isinstance(payload.get("brief"), dict) else {}
        kind = spec.kind
        if kind == "batch_checkpoint":   # deterministic: continue the split batch
            nxt = (brief.get("context") or {}).get("next") if isinstance(brief.get("context"), dict) else None
            return {"verdict": "continue", "note": "checkpoint: continue the batch", "submit": [nxt] if nxt else []}
        if kind == "brain_checkpoint":   # gpud brought the brain back between brainless jobs
            return {"verdict": "ack", "note": "brain checkpoint between brainless jobs", "submit": []}
        vision = bool(getattr(ctx.spec, "vision", False)) if ctx is not None else \
            bool(getattr(self.catalog.get(model), "vision", False))
        messages = self.messages(kind, brief, vision)
        if ctx is not None:
            content = messages[-1].get("content")
            n = sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url") \
                if isinstance(content, list) else 0
            ctx.log(f"asking {ctx.server_model} ({'vision' if vision else 'text'}) with {n} image(s) attached")
        raw = self.chat(model, messages, ctx)
        reply = parse_json_object(raw)
        decision = decide(reply, brief, self.catalog, raw)
        self.transcript = (self.transcript + [{"kind": kind, "model": model, "reply": raw[:2000],
                                               "decision": decision}])[-10:]
        return decision

    # ---- prompt
    def messages(self, kind: str, brief: dict, vision: bool) -> list:
        act = brief.get("attempted_action") if isinstance(brief.get("attempted_action"), dict) else {}
        family = _family_of(self.catalog, act.get("model"))
        keys = ", ".join(sorted(CHANGES.get(family or "", {}))) or "none"
        system = SYSTEM.replace("CHANGE_KEYS", keys)
        if kind.startswith("review"):
            task = "REVIEW: the job finished. Decide whether its output is acceptable."
        elif kind.startswith("recover"):
            task = "RECOVER: the job failed. Decide whether to retry, give up, or ask a human."
        else:
            task = f"Job kind {kind!r}: decide what to do next."
        summary = {
            "goal": brief.get("goal"),
            "attempted_action": {"kind": act.get("kind"), "model": act.get("model"),
                                 "params": {k: v for k, v in (act.get("params") or {}).items()
                                            if k not in ("items",)}},
            "outcome": brief.get("outcome"),
            "validator_results": brief.get("validator_results"),
            "log_tail": (brief.get("log_tail") or [])[-8:],
            "attempt": brief.get("attempt"), "max_attempts": brief.get("max_attempts"),
            "artifacts": brief.get("artifacts"),
            "context": brief.get("context"),
        }
        measured = self._measure(brief, family)
        if measured:
            summary["measured"] = measured
        text = f"{task}\n\nResume brief:\n{json.dumps(summary, indent=1, default=str)[:12000]}"
        images = self._images(brief) if vision and kind.startswith("review") else []
        if images:
            text += f"\n\n{len(images)} image(s) attached: " + (
                "sampled video frames in order." if family == "video" else "the audio waveform.")
        elif kind.startswith("review") and family == "video":
            text += "\n\nNo frames are attached (this brain has no vision); judge from the measurements."
        content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in images]
        return [{"role": "system", "content": system}, {"role": "user", "content": content if images else text}]

    def _measure(self, brief: dict, family: str | None) -> dict:
        out = {}
        for p in brief.get("artifacts") or []:
            if not isinstance(p, str):
                continue
            if p.endswith(".wav") and family in ("tts", "music") and os.path.exists(p):
                try:
                    out["audio"] = {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in media.wav_stats(p).items()}
                except media.MediaError as e:
                    out["audio_error"] = str(e)
            if p.endswith("video.json") and os.path.exists(p):
                try:
                    with open(p) as f:
                        meta = json.load(f)
                    out["video"] = {k: meta.get(k) for k in ("width", "height", "fps", "frame_count", "stats")}
                except (OSError, ValueError) as e:
                    out["video_error"] = str(e)
        return out

    def _images(self, brief: dict) -> list:
        pngs = [p for p in (brief.get("artifacts") or []) if isinstance(p, str) and p.endswith(".png")]
        frames = [p for p in pngs if os.path.basename(p).startswith("frame_")]
        chosen = frames if frames else pngs[:1]
        if len(chosen) > self.frames:
            idx = media.sample_indices(len(chosen), self.frames)
            chosen = [chosen[i] for i in idx]
        uris = [_png_data_uri(p) for p in chosen]
        return [u for u in uris if u]

    # ---- transport
    def _server_model(self, model: str, ctx) -> str:
        if ctx is not None:
            return ctx.server_model
        return self.server_ids.get(model, model)

    def chat(self, model: str, messages: list, ctx=None) -> str:
        body = {"model": self._server_model(model, ctx), "messages": messages, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "enable_thinking": False,
                "response_format": {"type": "json_object"}}
        if ctx is None:
            _, obj = json_call(self.base_url, "POST", "/v1/chat/completions", dict(body, stream=False),
                               timeout=self.timeout_s)
            return self._content(obj)
        return self._stream(dict(body, stream=True), ctx)

    @staticmethod
    def _content(obj) -> str:
        try:
            msg = obj["choices"][0]["message"]
            return msg.get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"unexpected chat completion body: {str(obj)[:200]}") from None

    def _stream(self, body: dict, ctx) -> str:
        conn = ctx.connect()
        parts, tokens = [], [0]
        errors = []

        def on_event(fields, _streamed):
            if fields.get("type") == "[DONE]":
                return
            if isinstance(fields.get("error"), dict):
                errors.append(error_message(json.dumps(fields).encode()))
                return
            for ch in fields.get("choices") or []:
                delta = ch.get("delta") if isinstance(ch, dict) else None
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
                    tokens[0] += 1
                    ctx.progress(min(1.0, tokens[0] / float(self.max_tokens)))

        try:
            status = conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                                  {"Content-Type": "application/json", "Accept": "text/event-stream"},
                                  header_timeout=self.timeout_s)
            if status != 200:
                raise RuntimeError(f"chat completion HTTP {status}: {error_message(conn.read_body(1 << 20))}")
            reader = SseReader(on_event)
            for chunk in conn.body_chunks(silence_s=self.timeout_s):
                reader.feed(chunk)
        except Cancelled:
            raise
        finally:
            conn.close()
        if errors:
            raise RuntimeError(f"chat completion failed: {errors[0]}")
        ctx.log(f"brain replied with {tokens[0]} chunks")
        return "".join(parts)
