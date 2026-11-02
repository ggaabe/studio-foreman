"""The worker's `probe` command: what a rendered talking take is, measured.

For each take: the 16 kHz mono WAV the speech gate transcribes, and a probe JSON with the video's
real size, decoded frame count and colour tags, the audio's 10 ms levels, its voiced runs (the
anchor gate's onsets) and silent runs (the only places an edit may close), and the times stray
hands show in the bottom band of the picture the beat keeps. A take whose probe JSON is newer than
its MP4 is skipped, so a retried job redoes only what is missing.
"""
from __future__ import annotations

import os

import numpy as np

from . import audio, ff
from .ff import ContentError, InfraError

SILENT_DB = -45.0           # a take whose loudest 10 ms is under this has no line in it
HANDS_FPS = 6


def _crop_box(w, h, crop):
    """(x0, x1, y0, y1) of the native take a beat's crop keeps, exactly as the edit cuts it (ff.window: centred
    when anchor_y is None, both sizes rounded to even). A punch-in on the beat is cut inside this box, so the box
    covers every row the beat's picture shows."""
    if not crop:
        return 0, w, 0, h
    x0, y0, cw, ch = ff.window(w, h, crop.get("scale") or 1.0, crop.get("anchor_y"))
    return x0, x0 + cw, y0, y0 + ch


def hands_visible(ffmpeg, mp4, w, h, crop=None, floor=0.74, limit=30, fps=HANDS_FPS):
    """Times (6 fps) where stray hands show: more than `limit` skin-coloured pixels in the native take's bottom
    band (rows below `floor` of the height, under the neck in a close-up) inside the part of the frame the beat's
    crop keeps. Frames are decoded at the take's real size. H3 close-ups can open with smeared hands rising into
    frame; the rule (r > 120, r > g + 25, g > b, r - b > 45) is tuned to that, not to every skin tone."""
    raw = ff.run(ffmpeg, "-i", mp4, "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-").stdout
    per = w * h * 3
    n = len(raw) // per
    if not n:
        return []
    fr = np.frombuffer(raw[:n * per], np.uint8).reshape(n, h, w, 3)
    x0, x1, y0, y1 = _crop_box(w, h, crop)
    lo, hi = max(int(floor * h), y0), y1
    if hi <= lo:
        return []
    band = fr[:, lo:hi, x0:x1].astype(np.int16)
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    hits = ((r > 120) & (r > g + 25) & (g > b) & (r - b > 45)).sum(axis=(1, 2))
    return [round(i / fps, 2) for i, c in enumerate(hits) if c > limit]


def _why(err) -> str:
    """ffmpeg's last words about a failure (not the temp paths before them)."""
    lines = [ln.strip() for ln in str(err).splitlines() if ln.strip()]
    return (lines[-1] if lines else str(err))[-160:]


def _fresh(probe_json, mp4, wav) -> bool:
    try:
        t = os.stat(probe_json).st_mtime_ns
        if t < os.stat(mp4).st_mtime_ns:
            return False
    except OSError:
        return False
    old = ff.read_json(probe_json)
    if not isinstance(old, dict):
        return False
    # A take with problems is judged on them (an audio stream that does not decode leaves no WAV): probing
    # the same file again would find the same.
    return not old.get("has_audio") or os.path.isfile(wav) or bool(old.get("problems"))


def probe_take(ffmpeg, take: dict) -> dict:
    """Probe one take (paths already resolved) and write its WAV and probe JSON. Returns the probe record."""
    mp4, wav, out = take["mp4"], take["wav"], take["probe"]
    if not os.path.isfile(mp4):
        raise InfraError(f"take missing: {mp4}")
    rw, rh = take["render"]
    info = ff.probe(ffmpeg, mp4)
    rec = {"duration_s": info["duration_s"], "frames": 0, "fps": info["fps"], "width": info["width"],
           "height": info["height"], "has_audio": info["has_audio"], "audio_rate": info["audio_rate"],
           "color": info["color"], "pix_fmt": info["pix_fmt"], "db10": [], "voiced_runs": [], "silent_runs": [],
           "hands": [], "problems": []}
    problems = rec["problems"]
    if not info["has_video"] or not info["width"]:
        problems.append("no video stream")
    else:
        try:
            rec["frames"], _ = ff.decode_count(ffmpeg, mp4)
        except ff.FfmpegError as e:
            problems.append(f"video does not decode: {_why(e)}")
        if (info["width"], info["height"]) != (rw, rh):
            problems.append(f"size {info['width']}x{info['height']} is not the render size {rw}x{rh}")
    if not info["has_audio"]:
        problems.append("no audio")
    else:
        try:
            with ff.atomic(wav) as tmp:
                ff.run(ffmpeg, "-i", mp4, "-vn", "-ac", 1, "-ar", 16000, "-c:a", "pcm_s16le", tmp)
            db = audio.db10(wav)
        except ff.FfmpegError as e:
            problems.append(f"audio does not decode: {_why(e)}")
            db = None
        if db is not None:
            rec["db10"] = [round(float(v), 1) for v in db]
            rec["voiced_runs"] = [[round(s, 2), round(e, 2)] for s, e in audio.voiced_runs(wav, db=db)]
            rec["silent_runs"] = [list(r) for r in audio.silent_runs(wav, -40.0, 0.25)]
            loudest = float(db.max()) if len(db) else -120.0
            if loudest < SILENT_DB:
                problems.append(f"silent (loudest 10 ms {loudest:.1f} dBFS)")
    hands = take.get("hands")
    if hands and rec["frames"] and info["width"]:
        rec["hands"] = hands_visible(ffmpeg, mp4, info["width"], info["height"], take.get("crop"),
                                     float(hands.get("floor", 0.74)), int(hands.get("limit", 30)))
    rec["ok"] = not problems
    ff.write_json(out, rec)
    return rec


def _take(t, i):
    if not isinstance(t, dict):
        raise ContentError(f"bad request: takes[{i}] is not an object")
    for k in ("mp4", "wav", "probe"):
        if not isinstance(t.get(k), str) or not t[k]:
            raise ContentError(f"bad request: takes[{i}].{k} must be a path")
    r = t.get("render")
    if not (isinstance(r, list) and len(r) == 2 and all(isinstance(v, int) and v > 0 for v in r)):
        raise ContentError(f"bad request: takes[{i}].render must be [width, height]")
    crop = t.get("crop")
    if crop is not None and not (isinstance(crop, dict) and isinstance(crop.get("scale", 1.0), (int, float))
                                 and float(crop.get("scale", 1.0)) >= 1.0):
        raise ContentError(f"bad request: takes[{i}].crop must be {{scale >= 1, anchor_y}} or null")
    hands = t.get("hands")
    if hands is not None and not isinstance(hands, dict):
        raise ContentError(f"bad request: takes[{i}].hands must be an object or null")
    return dict(t, mp4=os.path.abspath(t["mp4"]), wav=os.path.abspath(t["wav"]), probe=os.path.abspath(t["probe"]))


def run(req: dict, progress=lambda p, note="": None) -> dict:
    ffmpeg = ff.need_ffmpeg(req.get("ffmpeg"))
    takes = req.get("takes")
    if not isinstance(takes, list):
        raise ContentError("bad request: takes must be a list")
    takes = [_take(t, i) for i, t in enumerate(takes)]
    out = []
    for i, t in enumerate(takes):
        if _fresh(t["probe"], t["mp4"], t["wav"]):
            rec, skipped = ff.read_json(t["probe"]), True
        else:
            rec, skipped = probe_take(ffmpeg, t), False
        out.append({"mp4": os.path.relpath(t["mp4"]), "probe": os.path.relpath(t["probe"]), "ok": bool(rec.get("ok")),
                    "problems": rec.get("problems") or [], "skipped": skipped})
        progress((i + 1) / max(1, len(takes)), f"probed {os.path.basename(t['mp4'])}")
    return {"takes": out}
