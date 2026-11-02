"""ffmpeg helpers for the h3 worker, and the worker's error classes.

Every call runs the ffmpeg binary the request names (never one looked up on PATH). Sub-ranges of
a clip are cut with `select` on frame numbers plus a forced 24 fps output and the 12288 mp4
timescale every segment shares: `trim` and `-fps_mode passthrough` both drop frames on ffmpeg 7.1,
and concat -c copy only lines up when every piece has the same timescale. Frame counts come from a
full decode (`framecrc`), never from the container's header.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess

FPS = 24
TIMESCALE = 12288          # 24 fps x 512: what the mp4 muxer picks for 1/24, forced so pieces concat
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-an"]
TO_BT709 = "scale=out_color_matrix=bt709:out_range=tv"       # RGB sources: the matrix the tags claim
BT709_TAGS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
BT709_FLAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


class WorkerError(Exception):
    """A failure the worker reports with a class: "content" (exit 2) or "infra" (exit 3)."""
    kind = "infra"


class ContentError(WorkerError):
    """The takes or the plan cannot make a good ad (a cutaway word outside its cutaway, ...)."""
    kind = "content"


class InfraError(WorkerError):
    """The machine is missing something: ffmpeg, a font, an input file, enough music."""
    kind = "infra"


class FfmpegError(RuntimeError):
    """ffmpeg ran and failed (a crash of the step, retried by the core)."""


def _cmd(ffmpeg, args) -> list:
    return [str(ffmpeg)] + [str(a) for a in args]


def call(ffmpeg, args, data=None, check=True) -> subprocess.CompletedProcess:
    """Run ffmpeg with `args` as given (bytes out); a binary that cannot start is infrastructure."""
    try:
        p = subprocess.run(_cmd(ffmpeg, args), input=data, capture_output=True)
    except OSError as e:
        raise InfraError(f"cannot run ffmpeg {ffmpeg!r}: {e}") from e
    if check and p.returncode:
        tail = p.stderr.decode(errors="replace")[-1500:]
        raise FfmpegError(f"ffmpeg failed ({p.returncode}): {tail}")
    return p


def run(ffmpeg, *args, data=None) -> subprocess.CompletedProcess:
    """ffmpeg -y, quiet, no stdin interaction, then `args`."""
    return call(ffmpeg, ["-y", "-hide_banner", "-nostdin", "-loglevel", "error"] + list(args), data)


def info(ffmpeg, *args) -> str:
    """ffmpeg's stderr at the default log level (stream lines, filter reports); never raises on exit."""
    return call(ffmpeg, ["-hide_banner", "-nostdin"] + list(args), check=False).stderr.decode(errors="replace")


def need_ffmpeg(ffmpeg) -> str:
    if not isinstance(ffmpeg, str) or not ffmpeg or not os.path.isfile(ffmpeg) or not os.access(ffmpeg, os.X_OK):
        raise InfraError(f"ffmpeg not found: {ffmpeg!r}")
    return ffmpeg


def probe(ffmpeg, path) -> dict:
    """What `ffmpeg -i` says about a file: duration_s, width, height, fps, vcodec, pix_fmt, color
    (the stream's colour tags, e.g. "tv, bt709, progressive"), has_video, has_audio, audio_rate,
    audio_channels, and the video stream line."""
    err = info(ffmpeg, "-i", path)
    out = {"duration_s": None, "width": None, "height": None, "fps": None, "vcodec": None, "pix_fmt": None,
           "color": None, "has_video": False, "has_audio": False, "audio_rate": None, "audio_channels": None,
           "video_line": None}
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        out["duration_s"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 3)
    v = re.search(r"Stream #[^\n]*?Video: ([^\n]*)", err)
    if v:
        line = v.group(1)
        out.update(has_video=True, video_line="Video: " + line, vcodec=line.split()[0].strip(","))
        pm = re.search(r", ([a-z0-9_]+)(?:\(([^)]*)\))?, (\d+)x(\d+)", line)
        if pm:
            out.update(pix_fmt=pm.group(1), color=pm.group(2), width=int(pm.group(3)), height=int(pm.group(4)))
        fm = re.search(r"([\d.]+) fps", line)
        if fm:
            out["fps"] = float(fm.group(1))
    a = re.search(r"Stream #[^\n]*?Audio: ([^\n]*)", err)
    if a:
        out["has_audio"] = True
        rm = re.search(r"(\d+) Hz, ([^,]+)", a.group(1))
        if rm:
            out["audio_rate"], out["audio_channels"] = int(rm.group(1)), rm.group(2).strip()
    return out


def decode_count(ffmpeg, path) -> tuple:
    """(frames, seconds) of the first video stream, by decoding every frame (framecrc)."""
    p = call(ffmpeg, ["-hide_banner", "-nostdin", "-v", "error", "-i", path, "-map", "0:v:0", "-f", "framecrc", "-"])
    tb, pts = 1.0, []
    for ln in p.stdout.decode(errors="replace").splitlines():
        if ln.startswith("#tb 0:"):
            num, den = ln.split(":")[1].split("/")
            tb = int(num) / int(den)
        elif ln and not ln.startswith("#"):
            f = ln.split(",")
            pts.append((int(f[2]), int(f[3])))
    if not pts:
        return 0, 0.0
    return len(pts), (pts[-1][0] + pts[-1][1] - pts[0][0]) * tb


def freezes(ffmpeg, path, noise=0.003, dur=1.0) -> list:
    """Start times of every freeze freezedetect finds over the whole video."""
    err = info(ffmpeg, "-i", path, "-map", "0:v:0", "-vf", f"freezedetect=n={noise}:d={dur}", "-f", "null", "-")
    return [float(t) for t in re.findall(r"freeze_start: ([\d.]+)", err)]


def scene_cuts(ffmpeg, path, threshold=8) -> list:
    """[(time, score)] of every scene change scdet reports."""
    err = info(ffmpeg, "-i", path, "-map", "0:v:0", "-vf", f"scdet=threshold={threshold}", "-f", "null", "-")
    return [(float(t), float(s)) for s, t in re.findall(r"lavfi\.scd\.score: ([\d.]+), lavfi\.scd\.time: ([\d.]+)", err)]


def window(w, h, scale=1.0, anchor_y=None, within=None) -> tuple:
    """(x, y, cw, ch): the part of a w x h take a crop keeps, 1/scale of it (both sizes even), centred across,
    its top at anchor_y native px (centred when None). `within` (another window) nests it: a punch-in on a beat
    that has a crop of its own is that crop scaled again, its top clamped inside it, so it never shows rows the
    beat's crop leaves out (the rows the probe's hands scan skips)."""
    bx, by, bw, bh = within if within else (0, 0, w, h)
    s = max(1.0, float(scale or 1.0))
    cw, ch = int(round(bw / s / 2)) * 2, int(round(bh / s / 2)) * 2
    y = by + (bh - ch) // 2 if anchor_y is None else max(by, min(by + bh - ch, int(anchor_y)))
    return bx + (bw - cw) // 2, y, cw, ch


def frames_of(ffmpeg, src, a, b, out) -> str:
    """Frames [a, b) of a 24 fps clip as their own segment."""
    run(ffmpeg, "-i", src, "-vf", f"select='between(n\\,{a}\\,{b - 1})',setpts=N/({FPS}*TB)", "-r", FPS, *X264,
        "-video_track_timescale", TIMESCALE, out)
    return out


def concat_list(paths, list_file) -> str:
    """A concat-demuxer list naming each file by absolute path (quotes escaped)."""
    lines = []
    for p in paths:
        q = os.path.abspath(p).replace("'", "'\\''")
        lines.append(f"file '{q}'\n")
    write_text(list_file, "".join(lines))
    return list_file


# ---------------------------------------------------------------- atomic outputs

def remove(path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def tmp_for(path) -> str:
    """A temp name beside `path` with the same extension (ffmpeg picks the format from it)."""
    d, base = os.path.split(path)
    stem, ext = os.path.splitext(base)
    return os.path.join(d, f".{stem}.tmp-{os.getpid()}{ext}")


@contextlib.contextmanager
def atomic(path):
    """Yield a temp path to write; it replaces `path` only when the block finishes, and is removed if
    the block fails (or the worker is cancelled)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = tmp_for(path)
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        remove(tmp)
        raise


def write_text(path, text: str) -> None:
    with atomic(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)


def _finite(obj):
    """JSON without NaN or Infinity (a window with no music measures -inf)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):     # numpy scalars
        return _finite(obj.item())
    return obj


def write_json(path, obj) -> None:
    write_text(path, json.dumps(_finite(obj), indent=1, sort_keys=True, default=str))


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default
