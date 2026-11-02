"""Multi-request media jobs ("ops") on mlx-serve.

An ad needs many requests of one model in a row: a line of speech per script line, a still per
shot, a take per seed. Each such step is ONE gpud job whose payload lists its requests:

    {"goal": ..., "expected_s": ..., "workdir": "ads/<id>",
     "ops": [{"op": "speech", "out": "voice/line-0.wav", "input": "...", "voice": "af_heart"}, ...]}

and the adapter runs them in order under the job's single lease, cancel and provenance, exactly
like a render: the progress of each request maps into its share of the job (weighted by the
op's `expected_s`), requests without progress events (images, transcription) get a bounded
time-based heartbeat, and closing the connection cancels the request in flight.

Ops, the endpoint and capability each needs, and the fields forwarded (anything else in an op is
Foreman's own bookkeeping and is never sent):

    speech      /v1/audio/speech               audio          input voice speed seed language,
                                                              ref_audio_path -> ref_audio (Qwen3-TTS clone)
    music       /v1/audio/music-generations    music          prompt lyrics instrumental bpm seed duration_seconds ...
    transcribe  /v1/audio/transcriptions       transcription  multipart: file_path, language, prompt; verbose_json
                                                              with word timestamps unless words is false
    image       /v1/images/generations         image          prompt size seed steps mode strength ...,
                                                              image_path -> image, ref_image_paths -> ref_images
    video       /v1/video/generations          video          prompt width height num_frames seed steps pipeline
                                                              decoder cfg_scale cfg_audio_scale stg_scale stage2_steps
                                                              lora_paths lora_scales, first_frame_path, last_frame_path,
                                                              audio_path -> audio (padded with silence to exactly
                                                              num_frames / 24 s; that padded WAV is muxed onto the MP4)

The model's family must match every op (a job on the TTS model only speaks), and when
/v1/models lists capabilities the model must have the op's capability. Video requests are
checked against LTX's rules before they are sent (8k+1 frames, a 64 px grid for the two-stage
pipelines and 32 px for one stage, audio-to-video only two-stage, at most 768 MiB of raw RGB per
request), so a request the server would refuse costs nothing.

Outputs: `out` is relative to the work directory (inside the artifacts root; paths.py). Each op
writes its file and then a sidecar `<out>.meta.json` (stats, seed, checks, how long it took):
the sidecar is the completion marker, so a retried job skips the ops that already finished
(resumable). Inputs (`*_path`) must lie under the artifacts root or the owner's Foreman home.
Raw frames are deleted once the MP4 is written (keep_raw_frames keeps them), and a job first
deletes the scratch directories no live run owns (a killed gpud's raw frames). A take must come
back at the size and frame count asked for. A model without the capability its op needs, no
ffmpeg or no free disk is the environment's failure (Poll.reason engine_unconfigured or
engine_error), never the request's.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import secrets
import shutil
import time

from .. import media, paths, validators
from ..httpio import Cancelled, Silence, StreamConn, error_message
from ..policy import number
from . import mlx_serve as _ms  # mlx_serve imports this module lazily, so this import is never circular
from .sse import Base64FileSink, SseError, SseReader

OPS = {
    "speech": {"family": "tts", "path": "/v1/audio/speech", "cap": "audio", "expected_s": 20.0},
    "music": {"family": "music", "path": "/v1/audio/music-generations", "cap": "music", "expected_s": 90.0},
    "transcribe": {"family": "asr", "path": "/v1/audio/transcriptions", "cap": "transcription", "expected_s": 30.0},
    "image": {"family": "image", "path": "/v1/images/generations", "cap": "image", "expected_s": 60.0},
    "video": {"family": "video", "path": "/v1/video/generations", "cap": "video", "expected_s": 180.0},
}
OP_FAMILIES = frozenset(v["family"] for v in OPS.values())
FORWARD = {
    "speech": ("input", "voice", "speed", "seed", "language"),
    "music": ("prompt", "lyrics", "instrumental", "vocal_language", "bpm", "keyscale", "timesignature", "seed",
              "duration_seconds"),
    "image": ("prompt", "size", "seed", "steps", "mode", "strength", "guidance_scale", "negative_prompt",
              "transparent", "lora_paths", "lora_scales"),
    "video": ("prompt", "num_frames", "width", "height", "seed", "steps", "stage2_steps", "pipeline", "cfg_scale",
              "cfg_audio_scale", "stg_scale", "decoder", "lora_paths", "lora_scales"),
    "transcribe": ("language", "prompt", "temperature"),
}
VIDEO_PATH_KEYS = ("first_frame_path", "last_frame_path", "audio_path", "mux_audio_path")
PIPELINES = ("one_stage", "two_stage", "two_stage_hq")
LTX_MAX_RGB_BYTES = 768 << 20      # the server refuses a request whose raw RGB would exceed this
MAX_OPS = 64
MAX_REFS = 8
FPS = 24
META = ".meta.json"


def _err(msg: str, rejected: bool = False):
    return (_ms.RequestRejected if rejected else _ms.RunFailed)(msg)


def max_frames(width: int, height: int) -> int:
    """The longest 8k+1 clip LTX takes at width x height (768 MiB of raw RGB per request)."""
    per = width * height * 3
    if per <= 0:
        return 0
    n = LTX_MAX_RGB_BYTES // per
    return int((n - 1) // 8 * 8 + 1) if n >= 9 else 0


def video_problems(op: dict) -> list:
    """Why LTX would refuse this video op (empty when it is fine)."""
    out = []
    w, h, n = number(op.get("width")), number(op.get("height")), number(op.get("num_frames"))
    pipeline = op.get("pipeline") or "one_stage"
    if pipeline not in PIPELINES:
        out.append(f"pipeline must be one of {PIPELINES}, got {pipeline!r}")
    if w is None or h is None or w <= 0 or h <= 0 or w != int(w) or h != int(h):
        out.append(f"width and height must be positive integers, got {op.get('width')!r}x{op.get('height')!r}")
        return out
    grid = 64 if str(pipeline).startswith("two_stage") else 32
    if int(w) % grid or int(h) % grid:
        out.append(f"{pipeline} needs width and height divisible by {grid}, got {int(w)}x{int(h)}")
    if n is None or n != int(n) or n < 9 or (int(n) - 1) % 8:
        out.append(f"num_frames must be 8k+1 and at least 9, got {op.get('num_frames')!r}")
    elif int(n) * int(w) * int(h) * 3 > LTX_MAX_RGB_BYTES:
        out.append(f"{int(n)} frames at {int(w)}x{int(h)} exceed LTX's 768 MiB of raw RGB per request "
                   f"(at most {max_frames(int(w), int(h))} frames at this size)")
    if op.get("audio_path") and not str(pipeline).startswith("two_stage"):
        out.append("audio-to-video needs pipeline two_stage or two_stage_hq")
    return out


def wants_ops(run) -> bool:
    p = run.spec.payload if isinstance(run.spec.payload, dict) else {}
    if isinstance(p.get("ops"), list):
        return True
    if run.family in ("asr", "image"):
        return True
    if run.family == "video" and any(p.get(k) for k in VIDEO_PATH_KEYS):
        return True
    return run.family == "tts" and bool(p.get("ref_audio_path"))


def payload_ops(run) -> list:
    """The op list of a job: its `ops`, or the one request a single-request job describes."""
    p = run.spec.payload if isinstance(run.spec.payload, dict) else {}
    if p.get("ops") is not None:
        return p["ops"]
    op = {k: v for k, v in p.items() if k not in ("goal", "expected_s", "p90_s", "progress_interval_s", "duration_s",
                                                   "brief", "needs", "template_id", "consumer", "workdir")}
    kind = {"tts": "speech", "asr": "transcribe", "image": "image", "video": "video", "music": "music"}[run.family]
    op["op"] = kind
    op.setdefault("out", {"speech": "output.wav", "transcribe": "transcript.json", "image": "image.png",
                          "video": "output.mp4", "music": "output.wav"}[kind])
    if kind == "transcribe" and "file_path" not in op and isinstance(p.get("file"), str):
        op["file_path"] = p["file"]
    return [op]


def _meta_path(out: str) -> str:
    return out + META


def completed(out: str):
    """The sidecar of an op that already finished (its output and sidecar both exist), else None."""
    meta = paths.read_json(_meta_path(out))
    if isinstance(meta, dict) and os.path.isfile(out) and meta.get("complete") is True:
        return meta
    return None


def _b64_file(engine, path: str, wd: str) -> str:
    try:
        real = paths.input_path(path, engine.input_roots, wd)
    except paths.PathRefused as e:
        raise _err(str(e), rejected=True) from None
    with open(real, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _input(engine, path: str, wd: str) -> str:
    try:
        return paths.input_path(path, engine.input_roots, wd)
    except paths.PathRefused as e:
        raise _err(str(e), rejected=True) from None


# ---------------------------------------------------------------- the job

def run_ops(engine, run) -> dict:
    """Run every op of the job; returns the job's artifact (gpud validates it)."""
    engine._check_resident(run)
    p = run.spec.payload if isinstance(run.spec.payload, dict) else {}
    ops = payload_ops(run)
    if not isinstance(ops, list) or not ops or len(ops) > MAX_OPS:
        raise _err(f"a job needs 1 to {MAX_OPS} ops, got {len(ops) if isinstance(ops, list) else ops!r}", True)
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or op.get("op") not in OPS:
            raise _err(f"op {i} must be an object with op one of {sorted(OPS)}", True)
        if OPS[op["op"]]["family"] != run.family:
            raise _err(f"op {i} ({op['op']}) cannot run on {run.spec.model} (a {run.family} model)", True)
        if op["op"] == "video":
            bad = video_problems(op)
            if bad:
                raise _err(f"op {i}: " + "; ".join(bad), True)
    try:
        if p.get("workdir") is not None:
            wd = paths.workdir(engine.artifacts_dir, p["workdir"])
        else:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", run.spec.run_id)
            wd = paths.workdir(engine.artifacts_dir, f"{safe}-{engine._boot}")
    except paths.PathRefused as e:
        raise _err(str(e), True) from None
    run.dir = wd
    clean_scratch(engine, wd)
    caps = engine.capabilities()
    have = caps.get(run.server_model)
    for op in ops:
        need = OPS[op["op"]]["cap"]
        if have is not None and need not in have:
            raise _ms.EnvironmentFailure(
                f"{run.spec.model} (server id {run.server_model!r}) does not list the {need!r} capability in "
                f"/v1/models (it lists {sorted(have)}): check the configured model for this role (ads_options.models)")
    weights = [max(1.0, number(op.get("expected_s")) or OPS[op["op"]]["expected_s"]) for op in ops]
    total = sum(weights)
    results, done = [], 0.0
    for i, (op, w) in enumerate(zip(ops, weights)):
        if run.cancel.is_set():
            raise Cancelled("run was cancelled")
        lo, hi = _ms.EVENT_CAP * done / total, _ms.EVENT_CAP * (done + w) / total
        try:
            out = paths.out_path(wd, op.get("out"))
        except paths.PathRefused as e:
            raise _err(f"op {i}: {e}", True) from None
        meta = completed(out) if op.get("skip_existing", True) else None
        t0 = time.monotonic()
        if meta is not None:
            meta = dict(meta, skipped=True)
            with engine._lock:
                run.note(f"op {i + 1}/{len(ops)} {op['op']} {op.get('out')}: done already, skipped")
        else:
            with engine._lock:
                run.note(f"op {i + 1}/{len(ops)} {op['op']} -> {op.get('out')}")
            meta = RUNNERS[op["op"]](engine, run, op, out, wd, lo, hi, i)
            meta.update(complete=True, op=op["op"], out=op.get("out"), seconds=round(time.monotonic() - t0, 2),
                        model=run.spec.model, server_model=run.server_model, run_id=run.spec.run_id)
            for k in ("seed", "unit", "shot", "role"):
                if op.get(k) is not None:
                    meta.setdefault(k, op[k])
            paths.atomic_json(_meta_path(out), meta)
        meta["path"] = out
        results.append(meta)
        done += w
        with engine._lock:
            engine._bump(run, hi)
            run.timings[f"op{i}"] = round(run.elapsed(), 1)
    return _artifact(engine, run, wd, ops, results)


def _artifact(engine, run, wd: str, ops: list, results: list) -> dict:
    primary = next((i for i, op in enumerate(ops) if op.get("primary")), len(ops) - 1)
    prim = results[primary]
    out_paths, checks = [], []
    for r in results:
        for x in [r["path"], _meta_path(r["path"])] + list(r.get("samples") or []):
            if isinstance(x, str) and x not in out_paths:
                out_paths.append(x)
        for c in r.get("checks") or []:
            if isinstance(c, dict):
                checks.append(dict(c, check=f"{os.path.basename(r['path'])}:{c.get('check')}"))
    record = os.path.join(wd, f"job-{re.sub(r'[^A-Za-z0-9._-]+', '_', run.spec.run_id)}.json")
    try:
        version = (engine.health() or {}).get("version")
    except Exception:
        version = None
    paths.atomic_json(record, {"kind": run.spec.kind, "run_id": run.spec.run_id, "engine": engine.name,
                               "model": run.spec.model, "server_model": run.server_model, "server_version": version,
                               "ops": [{k: r.get(k) for k in ("op", "out", "seconds", "skipped", "ok", "seed", "unit")}
                                       for r in results],
                               "timings_s": dict(run.timings), "finished_wall": time.time()})
    art = {"path": prim["path"], "exists": all(os.path.isfile(r["path"]) for r in results),
           "paths": [x for x in out_paths if x != prim["path"]] + [record], "checks": checks, "workdir": wd,
           "ops": [{k: r.get(k) for k in ("op", "out", "ok", "skipped", "seconds")} for r in results],
           "timings_s": dict(run.timings)}
    for k in ("duration_s", "silence_ratio", "rms_dbfs", "peak", "sample_rate", "channels", "width", "height",
              "fps", "frame_count", "samples", "words") + _ms.VIDEO_STATS:
        if prim.get(k) is not None:
            art[k] = prim[k]
    if prim.get("op") == "video":
        art["format"] = "mp4"
    return art


# ---------------------------------------------------------------- transport

def _open(engine, run) -> StreamConn:
    conn = StreamConn(engine.base_url, engine.connect_timeout_s, poll_s=engine.read_poll_s)
    with engine._lock:
        if run.cancel.is_set():
            raise Cancelled("cancelled before the request was sent")
        run.conn = conn
    return conn


def _refused(status: int, conn) -> Exception:
    message = error_message(conn.read_body(1 << 20))
    if _ms.rejected_status(status, message):
        return _ms.RequestRejected(f"HTTP {status}: {message}")
    return _ms.RunFailed(f"HTTP {status}: {message}")


def _request_json(engine, run, path: str, body: bytes, ctype: str, lo: float, hi: float, grace: float,
                  limit: int = 256 << 20) -> dict:
    """A request answered by one JSON body (no progress events): a bounded heartbeat climbs from
    lo toward hi over `grace` seconds while the server works, then stops (the lease fires on a
    request that never answers)."""
    conn = _open(engine, run)
    with engine._lock:
        run.beat = (time.monotonic(), max(1.0, grace), lo, hi)
    try:
        status = conn.request("POST", path, body, {"Content-Type": ctype, "Accept": "application/json"},
                              header_timeout=engine.max_silence_s)
        if status != 200:
            raise _refused(status, conn)
        raw = conn.read_body(limit, silence_s=engine.max_silence_s)
    except Silence as e:
        raise _ms.RunFailed(f"mlx-serve sent nothing for {engine.max_silence_s:.0f}s ({e})") from None
    finally:
        with engine._lock:
            run.beat = None
            run.conn = None
        conn.close()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise _ms.RunFailed(f"{path}: the answer is not JSON ({e}): {raw[:200]!r}") from None
    if not isinstance(obj, dict):
        raise _ms.RunFailed(f"{path}: expected a JSON object, got {type(obj).__name__}")
    return obj


def _request_stream(engine, run, path: str, body: dict, scratch: str, lo: float, hi: float,
                    decode_grace: float = 0.0) -> tuple:
    """A streaming (SSE) request: progress events map into [lo, hi), and the complete event's
    big fields are decoded straight to files in `scratch`. Returns (fields, {key: file path})."""
    conn = _open(engine, run)
    state = {"terminal": None, "complete": None, "error": None, "events": 0, "stage": None}
    span = hi - lo

    def on_event(fields, streamed):
        kind = fields.get("type")
        if kind == "progress":
            stage = str(fields.get("stage") or "")
            step, total = number(fields.get("step")), number(fields.get("total"))
            frac = max(0.0, min(1.0, step / total)) if step is not None and total else None
            with engine._lock:
                state["events"] += 1
                p = run.progress
                mapped = lo + 0.9 * span * frac if frac is not None else p
                run.progress = min(lo + 0.995 * span, max(p, mapped, p + (lo + 0.9 * span - p) * 0.02))
                if stage != state["stage"]:
                    run.timings.setdefault(stage or "?", round(run.elapsed(), 1))
                    run.note(f"{stage} (+{run.elapsed():.1f}s)")
                    if stage.lower().startswith("decoding") and decode_grace > 0:
                        run.progress = max(run.progress, lo + 0.9 * span)
                        run.beat = (time.monotonic(), decode_grace, run.progress, lo + 0.995 * span)
                    state["stage"] = stage
        elif kind == "complete":
            state["complete"], state["terminal"] = (fields, streamed), "complete"
        elif kind == "error":
            state["terminal"], state["error"] = "error", str(fields.get("message") or "generation failed")

    def on_comment(_text):
        with engine._lock:
            p = run.progress
            run.progress = min(lo + 0.995 * span, p + (lo + 0.995 * span - p) * 0.005)

    def open_sink(head):
        if b'"complete"' not in head:
            return None

        def factory(key, fields):
            if key == "data":
                fmt = str(fields.get("format") or "")
                return Base64FileSink(os.path.join(scratch, "frames.rgb" if fmt == "rgb8" else f"output.{fmt or 'bin'}"))
            if key == "audio_data":
                return Base64FileSink(os.path.join(scratch, "audio.pcm"))
            return None
        return factory

    try:
        status = conn.request("POST", path, json.dumps(body).encode(),
                              {"Content-Type": "application/json", "Accept": "text/event-stream"},
                              header_timeout=engine.max_silence_s)
        if status != 200:
            raise _refused(status, conn)
        ctype = conn.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            raise _ms.RunFailed(f"expected an SSE stream, got {ctype!r}: {error_message(conn.read_body(1 << 20))}")
        reader = SseReader(on_event, on_comment, open_sink)
        try:
            for chunk in conn.body_chunks(silence_s=engine.max_silence_s):
                reader.feed(chunk)
                if state["terminal"] is not None:
                    break
        except Silence as e:
            reader.abort()
            raise _ms.RunFailed(f"mlx-serve sent nothing for {engine.max_silence_s:.0f}s ({e})") from None
        except SseError as e:
            reader.abort()
            raise _ms.RunFailed(f"unparseable SSE stream: {e}") from None
        except BaseException:
            reader.abort()  # a cancel or a dropped connection: close the half-written frame file
            raise
    finally:
        with engine._lock:
            run.beat = None
            run.conn = None
        conn.close()
    if state["terminal"] == "error":
        raise _ms.RunFailed(f"mlx-serve: {state['error']}")
    if state["terminal"] != "complete":
        raise ConnectionError(f"stream ended after {state['events']} progress events without a complete or error event")
    fields, streamed = state["complete"]
    files = {k: s.path for k, s in streamed.items()}
    for key, name in (("data", None), ("audio_data", "audio.pcm")):
        if key in files or not isinstance(fields.get(key), str) or not fields.get(key):
            continue
        fmt = str(fields.get("format") or "")
        name = name or ("frames.rgb" if fmt == "rgb8" else f"output.{fmt or 'bin'}")
        target = os.path.join(scratch, name)
        with open(target, "wb") as f:
            f.write(binascii.a2b_base64(fields[key]))
        files[key] = target
    return fields, files


def clean_scratch(engine, wd: str) -> int:
    """Delete the scratch directories (raw frames, padded audio) that no live run of this engine
    owns: what a gpud killed mid-decode left behind, hundreds of MB per take. Returns bytes freed."""
    root = os.path.join(wd, ".scratch")
    if not os.path.isdir(root) or os.path.islink(root):
        return 0
    with engine._lock:
        live = [re.sub(r"[^A-Za-z0-9._-]+", "_", r.spec.run_id) + "-" for r in engine._runs.values()
                if getattr(r, "status", None) == "running"]
    freed = 0
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if any(name.startswith(prefix) for prefix in live) or os.path.islink(path) or not paths.inside(path, wd):
            continue
        for dirpath, _, files in os.walk(path):
            for f in files:
                try:
                    freed += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        shutil.rmtree(path, ignore_errors=True)
    return freed


def _scratch(wd: str, run, i: int) -> str:
    d = os.path.join(wd, ".scratch", f"{re.sub(r'[^A-Za-z0-9._-]+', '_', run.spec.run_id)}-{i}-{secrets.token_hex(2)}")
    os.makedirs(d, exist_ok=True)
    return d


def _body(op: dict, kind: str, model: str) -> dict:
    body = {k: op[k] for k in FORWARD[kind] if op.get(k) is not None}
    body["model"] = model
    return body


def _audio_meta(path: str) -> dict:
    try:
        st = media.wav_stats(path)
    except media.MediaError as e:
        return {"probe_error": str(e), "ok": False,
                "checks": [{"check": "audio", "ok": False, "detail": str(e)}]}
    meta = {k: st[k] for k in ("duration_s", "silence_ratio", "rms_dbfs", "peak", "sample_rate", "channels")}
    checks = [{"check": "duration", "ok": st["duration_s"] > 0, "detail": f"{st['duration_s']:.2f}s"}]
    meta.update(checks=checks, ok=all(c["ok"] for c in checks))
    return meta


# ---------------------------------------------------------------- ops

def _op_speech(engine, run, op, out, wd, lo, hi, i) -> dict:
    text = op.get("input") or op.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _err(f"op {i}: speech needs 'input' text", True)
    body = _body(op, "speech", run.server_model)
    body["input"] = text
    if op.get("ref_audio_path"):
        body["ref_audio"] = _b64_file(engine, op["ref_audio_path"], wd)
    body["stream"] = True
    scratch = _scratch(wd, run, i)
    try:
        fields, files = _request_stream(engine, run, OPS["speech"]["path"], body, scratch, lo, hi)
        src = files.get("data")
        if not src:
            raise _ms.RunFailed("the complete event carried no audio")
        os.replace(src, out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    meta = _audio_meta(out)
    meta.update(voice=op.get("voice"), clone=bool(op.get("ref_audio_path")))
    return meta


def _op_music(engine, run, op, out, wd, lo, hi, i) -> dict:
    if not isinstance(op.get("prompt"), str) or not op["prompt"].strip():
        raise _err(f"op {i}: music needs a 'prompt'", True)
    body = _body(op, "music", run.server_model)
    body["stream"] = True
    scratch = _scratch(wd, run, i)
    try:
        _, files = _request_stream(engine, run, OPS["music"]["path"], body, scratch, lo, hi)
        if not files.get("data"):
            raise _ms.RunFailed("the complete event carried no audio")
        os.replace(files["data"], out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    meta = _audio_meta(out)
    silent = number(meta.get("silence_ratio"))
    meta["checks"] = list(meta.get("checks") or []) + [{
        "check": "silence", "ok": silent is not None and silent <= validators.MAX_SILENCE_RATIO,
        "detail": f"silence_ratio={meta.get('silence_ratio')!r}"}]
    meta["ok"] = all(c["ok"] for c in meta["checks"])
    return meta


def multipart(fields: list, files: list) -> tuple:
    """(body, content type) for multipart/form-data: fields [(name, value)], files
    [(name, filename, content type, bytes)]."""
    boundary = "foreman" + secrets.token_hex(12)
    parts = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, filename, ctype, data in files:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                     f'Content-Type: {ctype}\r\n\r\n'.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _op_transcribe(engine, run, op, out, wd, lo, hi, i) -> dict:
    src = _input(engine, op.get("file_path"), wd)
    with open(src, "rb") as f:
        audio = f.read()
    fields = [("model", run.server_model), ("response_format", op.get("response_format") or "verbose_json")]
    if op.get("words", True):
        fields.append(("timestamp_granularities[]", "word"))
    for k in FORWARD["transcribe"]:
        if op.get(k) is not None:
            fields.append((k, str(op[k])))
    body, ctype = multipart(fields, [("file", "audio.wav", "audio/wav", audio)])
    try:
        seconds = media.wav_duration(src)
    except media.MediaError:
        seconds = 30.0
    grace = max(30.0, 3.0 * (number(op.get("expected_s")) or OPS["transcribe"]["expected_s"]), 2.0 * seconds)
    before = engine._companions_resident(run.spec.model) if getattr(engine, "companions", None) else set()
    try:
        obj = _request_json(engine, run, OPS["transcribe"]["path"], body, ctype, lo, hi, grace)
    finally:
        if getattr(engine, "companions", None):
            engine.note_companions(run.spec.model, before)
    words = obj.get("words") if isinstance(obj.get("words"), list) else []
    paths.atomic_json(out, obj)
    ok = isinstance(obj.get("text"), str)
    checks = [{"check": "transcript", "ok": ok, "detail": f"{len(words)} timed words"}]
    return {"words": len(words), "text_chars": len(obj.get("text") or ""), "audio_s": round(seconds, 3),
            "checks": checks, "ok": ok}


def _op_image(engine, run, op, out, wd, lo, hi, i) -> dict:
    if not isinstance(op.get("prompt"), str) or not op["prompt"].strip():
        raise _err(f"op {i}: image needs a 'prompt'", True)
    body = _body(op, "image", run.server_model)
    if op.get("image_path"):
        body["image"] = _b64_file(engine, op["image_path"], wd)
        body.setdefault("mode", "edit")
    refs = op.get("ref_image_paths") or []
    if not isinstance(refs, list) or len(refs) > MAX_REFS:
        raise _err(f"op {i}: ref_image_paths must be a list of at most {MAX_REFS} files", True)
    if refs:
        body["ref_images"] = [_b64_file(engine, r, wd) for r in refs]
    grace = max(30.0, 3.0 * (number(op.get("expected_s")) or OPS["image"]["expected_s"]))
    obj = _request_json(engine, run, OPS["image"]["path"], json.dumps(body).encode(), "application/json", lo, hi, grace)
    data = obj.get("data")
    b64 = data[0].get("b64_json") if isinstance(data, list) and data and isinstance(data[0], dict) else None
    if not isinstance(b64, str) or not b64:
        raise _ms.RunFailed(f"the image answer carried no b64_json: {str(obj)[:200]}")
    try:
        png = base64.b64decode(b64)
    except (binascii.Error, ValueError) as e:
        raise _ms.RunFailed(f"the image is not base64: {e}") from None
    paths.atomic_write(out, png)
    size = media.png_size(png) or media.image_size(out)
    w, h = (size or (0, 0))
    checks = [{"check": "dimensions", "ok": w > 0 and h > 0, "detail": f"{w}x{h}"}]
    return {"width": w, "height": h, "mode": body.get("mode"), "refs": len(refs), "checks": checks,
            "ok": all(c["ok"] for c in checks), "prompt": op["prompt"][:300]}


def _op_video(engine, run, op, out, wd, lo, hi, i) -> dict:
    if not engine.ffmpeg:
        raise _ms.EnvironmentFailure("video ops write MP4s and need ffmpeg (FOREMAN_FFMPEG / ffmpeg in config)")
    if not isinstance(op.get("prompt"), str) or not op["prompt"].strip():
        raise _err(f"op {i}: video needs a 'prompt'", True)
    w, h, n = int(op["width"]), int(op["height"]), int(op["num_frames"])
    need = n * w * h * 3 * 1.3 + 1e9
    free = shutil.disk_usage(wd).free
    if free < need:
        raise _ms.EnvironmentFailure(f"not enough free disk for {n} frames at {w}x{h}: {free / 1e9:.1f} GB free, "
                                     f"{need / 1e9:.1f} GB needed", reason="engine_error")
    body = _body(op, "video", run.server_model)
    body.update(width=w, height=h, num_frames=n, pipeline=op.get("pipeline") or "one_stage")
    scratch = _scratch(wd, run, i)
    mux_wav, mux_from = None, None
    try:
        for key, field in (("first_frame_path", "first_frame_image"), ("last_frame_path", "last_frame_image")):
            if op.get(key):
                body[field] = _b64_file(engine, op[key], wd)
        if op.get("audio_path"):
            src = _input(engine, op["audio_path"], wd)
            fed = src
            if op.get("pad_audio", True):
                fed = os.path.join(scratch, "audio-in.wav")
                try:
                    media.pad_wav(src, fed, n / float(FPS))
                except media.MediaError as e:
                    raise _err(f"op {i}: cannot pad the audio: {e}", True) from None
            with open(fed, "rb") as f:
                body["audio"] = base64.b64encode(f.read()).decode("ascii")
            if op.get("mux_audio", "input") == "input":
                mux_wav, mux_from = fed, "input"
        body["stream"] = True
        mpf = n * w * h / 1e6
        grace = max(engine.min_decode_grace_s, engine.decode_s_per_mpf * mpf)
        fields, files = _request_stream(engine, run, OPS["video"]["path"], body, scratch, lo, hi, decode_grace=grace)
        frames_path = files.get("data")
        if not frames_path:
            raise _ms.RunFailed("the complete event carried no frames")
        fw, fh = int(number(fields.get("width")) or w), int(number(fields.get("height")) or h)
        fn, fps = int(number(fields.get("frames")) or n), number(fields.get("fps")) or FPS
        meta = {"width": fw, "height": fh, "frame_count": fn, "fps": fps, "seed": op.get("seed"),
                "pipeline": body["pipeline"], "requested_frames": n}
        try:
            meta.update({k: v for k, v in media.video_stats(frames_path, fw, fh, fn, fps).items()
                         if k in _ms.VIDEO_STATS})
            samples = []
            base = os.path.splitext(out)[0]
            for idx, png in media.sample_frames_png(frames_path, fw, fh, fn, int(op.get("samples") or 3),
                                                    engine.sample_max_side):
                sp = f"{base}-f{idx:04d}.png"
                paths.atomic_write(sp, png)
                samples.append(sp)
            meta["samples"] = samples
        except media.MediaError as e:
            meta["probe_error"] = str(e)
        if mux_wav is None and op.get("mux_audio", "generated") == "generated" and files.get("audio_data"):
            sr = int(number(fields.get("audio_sample_rate")) or 0)
            ch = int(number(fields.get("audio_channels")) or 2)
            if sr > 0 and ch > 0:
                mux_wav, mux_from = os.path.join(scratch, "audio.wav"), "generated"
                media.wav_from_pcm_file(files["audio_data"], mux_wav, sr, ch)
        saved, run.dir = run.dir, scratch
        try:
            mp4 = engine._mux(run, frames_path, fw, fh, fps, mux_wav)
        finally:
            run.dir = saved
        if not mp4:
            raise _ms.RunFailed("ffmpeg could not write the MP4 (see the log tail)")
        os.replace(mp4, out)
        meta["audio"] = mux_from
        if engine.keep_raw_frames:
            os.replace(frames_path, os.path.splitext(out)[0] + ".rgb")
            meta["raw_frames"] = os.path.splitext(out)[0] + ".rgb"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    meta["duration_s"] = fn / float(fps) if fps else None
    black, frozen = number(meta.get("black_frame_ratio")), number(meta.get("frozen_frame_ratio"))
    checks = [
        {"check": "black_frames", "ok": black is not None and black <= validators.MAX_BLACK_FRAME_RATIO,
         "detail": f"black_frame_ratio={meta.get('black_frame_ratio')!r}"},
        {"check": "frozen_frames", "ok": frozen is not None and frozen <= validators.MAX_FROZEN_FRAME_RATIO,
         "detail": f"frozen_frame_ratio={meta.get('frozen_frame_ratio')!r}"},
        {"check": "frames", "ok": fn == n, "detail": f"got {fn} frames, asked for {n}"},
        {"check": "size", "ok": (fw, fh) == (w, h), "detail": f"got {fw}x{fh}, asked for {w}x{h}"},
    ]
    meta.update(checks=checks, ok=all(c["ok"] for c in checks))
    return meta


RUNNERS = {"speech": _op_speech, "music": _op_music, "transcribe": _op_transcribe, "image": _op_image,
           "video": _op_video}
