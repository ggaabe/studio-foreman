"""Finished vertical ads with ffmpeg alone (no Remotion), and the measurements that check them.

The cut is built in passes, each a plain ffmpeg run on files in the ad's work directory (run
from that directory with relative paths, so no filter argument needs escaping):

  1. one segment per shot, normalized to the output size (1080x1920 by default), 24 fps and an
     exact frame count: a generated take is scaled to cover and cropped; a screen capture is
     fitted over a blurred copy of itself; a still gets a slow push-in (zoompan on a 2x copy); the
     end card is a still or a plain background with the price, URL and CTA drawn in the middle;
  2. the segments are joined with hard cuts (concat demuxer, stream copy);
  3. the audio: the voice track, the music bed ducked under it (sidechaincompress keyed by the
     voice), mixed, then loudnorm in two passes to -14 LUFS integrated and -1.5 dBTP;
  4. the final encode burns in the word-highlight ASS captions (libass with an explicit fontsdir:
     the Studio's ffmpeg has no fontconfig) and the small always-visible "AI-generated" label
     (drawtext with an explicit fontfile), H.264 (VideoToolbox, libx264 as the fallback) + AAC,
     +faststart, at a bitrate that keeps the file under max_mb (re-encoded smaller if not).

Captions stay inside the central safe zone: never over the top 12% or the bottom 20% of the
frame (bottom-centre alignment with a 24% bottom margin and at most two lines).

Fonts: find_font() resolves a real font file at runtime (configured paths first, then the usual
macOS locations) and raises FontError with the places it looked if there is none.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import struct
import subprocess

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)
FONT_GLOBS = ("/Library/Fonts/*.ttf", "/System/Library/Fonts/Supplemental/*.ttf", "/System/Library/Fonts/*.ttf")
SAFE_TOP, SAFE_BOTTOM = 0.12, 0.20   # caption-free bands, as fractions of the height
CAPTION_MARGIN_V = 0.24               # bottom margin of the caption block (above the bottom band)
CAPTION_SIZE = 0.036                  # caption font size, fraction of the height (69 px at 1920)
HIGHLIGHT = "&H0000D7FF&"             # ASS colours are &HAABBGGRR: gold
WHITE = "&H00FFFFFF&"


class AssemblyError(RuntimeError):
    pass


class FontError(AssemblyError):
    pass


# ---------------------------------------------------------------- running ffmpeg

class Ffmpeg:
    """An ffmpeg binary plus how to run it (a LocalContext.run that a cancel stops, or
    subprocess.run), always from the work directory."""

    def __init__(self, binary: str, cwd: str, runner=None, timeout: float = 1800.0, note=None):
        if not binary:
            raise AssemblyError("ffmpeg is required to assemble an ad (set ffmpeg in config.json)")
        self.binary, self.cwd, self.timeout = binary, cwd, timeout
        self._runner = runner
        self.note = note or (lambda line: None)

    def run(self, args: list, check: bool = True, heartbeat: float | None = None):
        cmd = [self.binary, "-hide_banner", "-nostdin"] + [str(a) for a in args]
        if self._runner is not None:
            res = self._runner(cmd, timeout=self.timeout, cwd=self.cwd, heartbeat=heartbeat)
        else:
            res = subprocess.run(cmd, capture_output=True, timeout=self.timeout, cwd=self.cwd, check=False)
        if check and res.returncode != 0:
            tail = (res.stderr or b"")[-1200:].decode("utf-8", "replace")
            raise AssemblyError(f"ffmpeg failed ({res.returncode}): {' '.join(str(a) for a in args)[:300]}\n{tail}")
        return res

    def path(self, rel: str) -> str:
        """A file named relative to the work directory, as this process sees it."""
        return rel if os.path.isabs(rel) else os.path.join(self.cwd, rel)

    def stderr(self, args: list) -> str:
        res = self.run(args, check=False)
        return (res.stderr or b"").decode("utf-8", "replace")

    def version(self) -> str:
        try:
            res = subprocess.run([self.binary, "-version"], capture_output=True, timeout=10, check=False)
            return (res.stdout or b"").decode("utf-8", "replace").splitlines()[0][:200]
        except (OSError, subprocess.SubprocessError, IndexError):
            return "unknown"


# ---------------------------------------------------------------- fonts

def find_font(preferred=()) -> str:
    looked = []
    for cand in list(preferred or ()) + list(FONT_CANDIDATES):
        if not cand:
            continue
        cand = os.path.expanduser(cand)
        looked.append(cand)
        if os.path.isfile(cand):
            return cand
    for pattern in FONT_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
        looked.append(pattern)
    raise FontError("no font file found for captions and the end card (ffmpeg here has no fontconfig, so a real "
                    "file is needed): set ads_options.fonts to a .ttf path. Looked at: " + ", ".join(looked[:12]))


def font_names(path: str) -> tuple:
    """(family, subfamily) from a TrueType/OpenType font's name table (the first face of a
    collection), or (file stem, "") when it cannot be read."""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, "rb") as f:
            data = f.read(64 << 20)
        base = struct.unpack(">I", data[12:16])[0] if data[:4] == b"ttcf" else 0
        num = struct.unpack(">H", data[base + 4: base + 6])[0]
        name_off = None
        for i in range(num):
            rec = data[base + 12 + 16 * i: base + 28 + 16 * i]
            if rec[:4] == b"name":
                name_off = struct.unpack(">I", rec[8:12])[0]
                break
        if name_off is None:
            return stem, ""
        _fmt, count, strings = struct.unpack(">HHH", data[name_off: name_off + 6])
        found = {}
        for i in range(count):
            pid, eid, lang, nid, length, off = struct.unpack(">HHHHHH", data[name_off + 6 + 12 * i: name_off + 18 + 12 * i])
            if nid not in (1, 2, 16, 17):
                continue
            raw = data[name_off + strings + off: name_off + strings + off + length]
            if pid == 3 or pid == 0:
                text = raw.decode("utf-16-be", "replace")
                rank = 0 if (pid == 3 and lang == 0x409) else 1
            elif pid == 1:
                text = raw.decode("mac_roman", "replace")
                rank = 2
            else:
                continue
            if nid not in found or rank < found[nid][0]:
                found[nid] = (rank, text)
        family = (found.get(16) or found.get(1) or (0, stem))[1].strip() or stem
        sub = (found.get(17) or found.get(2) or (0, ""))[1].strip()
        return family, sub
    except (OSError, struct.error, IndexError, ValueError):
        return stem, ""


def install_font(src: str, fonts_dir: str, name: str) -> dict:
    """Copy a font into the ad's fonts directory under a filter-safe name; returns
    {"path" (relative to the work directory), "family", "bold"}."""
    os.makedirs(fonts_dir, exist_ok=True)
    ext = os.path.splitext(src)[1].lower() or ".ttf"
    dst = os.path.join(fonts_dir, f"{name}{ext}")
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    family, sub = font_names(dst)
    return {"file": dst, "family": family, "bold": "bold" in sub.lower() or "black" in sub.lower(), "source": src}


# ---------------------------------------------------------------- probing

def probe(ff: Ffmpeg, path: str) -> dict:
    """Duration, size, frame rate and streams from `ffmpeg -i` (no ffprobe needed)."""
    text = ff.stderr(["-i", path])
    out = {"duration_s": None, "width": None, "height": None, "fps": None, "has_audio": False, "has_video": False,
           "vcodec": None, "acodec": None}
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        out["duration_s"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    for line in text.splitlines():
        if "Stream #" in line and "Video:" in line and not out["has_video"]:
            out["has_video"] = True
            out["vcodec"] = line.split("Video:", 1)[1].strip().split()[0].strip(",")
            sz = re.search(r"\b(\d{2,5})x(\d{2,5})\b", line)
            if sz:
                out["width"], out["height"] = int(sz.group(1)), int(sz.group(2))
            fr = re.search(r"([\d.]+) fps", line)
            if fr:
                out["fps"] = float(fr.group(1))
        elif "Stream #" in line and "Audio:" in line and not out["has_audio"]:
            out["has_audio"] = True
            out["acodec"] = line.split("Audio:", 1)[1].strip().split()[0].strip(",")
    return out


def loudness(ff: Ffmpeg, path: str) -> dict:
    """Integrated loudness (LUFS), true peak (dBTP) and loudness range (LU) from ebur128."""
    text = ff.stderr(["-i", path, "-map", "0:a:0", "-filter:a", "ebur128=peak=true", "-f", "null", "-"])
    summary = text[text.rfind("Summary:"):] if "Summary:" in text else text

    def grab(pattern):
        m = re.search(pattern, summary)
        if not m:
            return None
        try:
            v = float(m.group(1))
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return {"integrated_lufs": grab(r"I:\s+(-?[\d.]+|-inf) LUFS"), "true_peak_dbtp": grab(r"Peak:\s+(-?[\d.]+|-inf) dBFS"),
            "lra_lu": grab(r"LRA:\s+(-?[\d.]+) LU")}


def detect(ff: Ffmpeg, path: str, start: float = 0.0, duration: float | None = None,
           black_min_s: float = 0.25, freeze_min_s: float = 1.0) -> dict:
    """Black and frozen intervals (seconds from `start`) from blackdetect and freezedetect."""
    args = ["-ss", f"{start:.3f}"] + (["-t", f"{duration:.3f}"] if duration else []) + [
        "-i", path, "-map", "0:v:0", "-vf",
        f"blackdetect=d={black_min_s}:pix_th=0.10,freezedetect=n=-60dB:d={freeze_min_s}", "-f", "null", "-"]
    text = ff.stderr(args)
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", text)]
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", text)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", text)]
    freeze = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else (duration or s)
        freeze.append((s, e))
    return {"black": black, "freeze": freeze}


def gray_frames(ff: Ffmpeg, path: str, width: int, height: int, start: float = 0.0,
                duration: float | None = None, fps: float = 24.0) -> list:
    """Frames as width x height 8-bit grey byte strings (a small decode for statistics)."""
    args = ["-loglevel", "error", "-ss", f"{start:.3f}"] + (["-t", f"{duration:.3f}"] if duration else []) + [
        "-i", path, "-vf", f"fps={fps:g},scale={width}:{height}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    res = ff.run(args)
    data = res.stdout or b""
    size = width * height
    return [data[i: i + size] for i in range(0, len(data) - size + 1, size)]


def frame_std(frame: bytes) -> float:
    n = len(frame)
    if not n:
        return 0.0
    mean = sum(frame) / n
    return math.sqrt(sum((b - mean) ** 2 for b in frame) / n)


def extract_png(ff: Ffmpeg, path: str, t: float, out: str, max_side: int = 512) -> str:
    """One frame at `t` seconds as a PNG whose long side is at most max_side (for the vision brain)."""
    ff.run(["-loglevel", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", path, "-frames:v", "1", "-vf",
            f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))':flags=area", out])
    return out


def scale_png(ff: Ffmpeg, src: str, out: str, max_side: int = 512) -> str:
    ff.run(["-loglevel", "error", "-y", "-i", src, "-frames:v", "1", "-vf",
            f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))':flags=area", out])
    return out


def fit_still(ff: Ffmpeg, src: str, out: str, width: int, height: int) -> str:
    """A still scaled to cover width x height and centre-cropped (a keyframe at the video size)."""
    ff.run(["-loglevel", "error", "-y", "-i", src, "-frames:v", "1", "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height},setsar=1",
            out])
    return out


# ---------------------------------------------------------------- captions

def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\\", "").replace("{", "(").replace("}", ")")).strip()


def caption_chunks(words: list, max_words: int = 4, max_chars: int = 24) -> list:
    """Words grouped into short on-screen phrases (never across script lines; a phrase ends after
    punctuation): [[word, ...], ...]."""
    chunks, cur = [], []
    for w in words:
        if not _ass_text(w.get("text", "")):
            continue
        if cur and (w.get("line") != cur[-1].get("line") or len(cur) >= max_words
                    or len(" ".join(x["text"] for x in cur + [w])) > max_chars):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?;:,]$", w["text"].strip()):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def caption_events(words: list, end_limit: float | None = None, hold_s: float = 0.25, **kw) -> list:
    """[(start, end, ASS text)]: one event per word, showing its phrase with that word highlighted."""
    chunks = caption_chunks(words, **kw)
    events = []
    for ci, chunk in enumerate(chunks):
        nxt = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else None
        stop = chunk[-1]["end"] + hold_s
        if nxt is not None:
            stop = min(stop, nxt)
        if end_limit is not None:
            stop = min(stop, end_limit)
        for wi, w in enumerate(chunk):
            a = w["start"] if wi else chunk[0]["start"]
            b = chunk[wi + 1]["start"] if wi + 1 < len(chunk) else stop
            if b - a < 0.02:
                continue
            parts = []
            for xi, x in enumerate(chunk):
                t = _ass_text(x["text"])
                parts.append(f"{{\\c{HIGHLIGHT}}}{t}{{\\c{WHITE}}}" if xi == wi else t)
            events.append((a, b, " ".join(parts)))
    return events


def caption_style(width: int, height: int) -> dict:
    size = max(10, int(round(CAPTION_SIZE * height)))
    return {"size": size, "margin_v": int(round(CAPTION_MARGIN_V * height)),
            "margin_lr": int(round(0.08 * width)), "outline": max(1, int(round(size * 0.07))),
            "shadow": max(0, int(round(size * 0.03)))}


def write_ass(path: str, words: list, width: int, height: int, family: str, bold: bool = True,
              end_limit: float | None = None) -> int:
    """Write the captions file; returns the number of events."""
    st = caption_style(width, height)
    events = caption_events(words, end_limit=end_limit)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Caption,{family.replace(',', ' ')},{st['size']},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"{-1 if bold else 0},0,0,0,100,100,0,0,1,{st['outline']},{st['shadow']},2,{st['margin_lr']},{st['margin_lr']},"
        f"{st['margin_v']},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for a, b, text in events:
        lines.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Caption,,0,0,0,,{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(events)


# ---------------------------------------------------------------- segments

X264_SEG = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-an"]


def seg_take(ff: Ffmpeg, src: str, out: str, seconds: float, width: int, height: int, fps: int = 24) -> str:
    """A generated take: cover-scaled, cropped, exactly `seconds` long (last frame held if short)."""
    n = max(1, int(round(seconds * fps)))
    ff.run(["-loglevel", "error", "-y", "-i", src, "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height},"
            f"setsar=1,fps={fps},tpad=stop_mode=clone:stop_duration={seconds + 1:.3f}",
            "-frames:v", str(n)] + X264_SEG + [out], heartbeat=30.0)
    return out


def _pushin(ff: Ffmpeg, still: str, out: str, seconds: float, width: int, height: int, fps: int, push: float) -> str:
    n = max(1, int(round(seconds * fps)))
    zoom = f"1+{push:.4f}*on/{max(1, n - 1)}"
    ff.run(["-loglevel", "error", "-y", "-i", still, "-vf",
            f"scale={2 * width}:{2 * height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={2 * width}:{2 * height},zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={n}:s={width}x{height}:fps={fps},setsar=1",
            "-frames:v", str(n)] + X264_SEG + [out], heartbeat=30.0)
    return out


def seg_still(ff: Ffmpeg, src: str, out: str, seconds: float, width: int, height: int, fps: int = 24,
              push: float = 0.06) -> str:
    """A still with a slow push-in."""
    return _pushin(ff, src, out, seconds, width, height, fps, push)


def _fit_over_blur(width: int, height: int) -> str:
    fw, fh = int(width * 0.92) // 2 * 2, int(height * 0.92) // 2 * 2
    return (f"split=2[a][b];[a]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
            f"boxblur=20:2,eq=brightness=-0.08[bg];[b]scale={fw}:{fh}:force_original_aspect_ratio=decrease,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def seg_screen(ff: Ffmpeg, src: str, out: str, seconds: float, width: int, height: int, fps: int = 24,
               is_video: bool = True, push: float = 0.05) -> str:
    """A real screen capture fitted over a blurred copy of itself; a still capture also gets the
    slow push-in."""
    n = max(1, int(round(seconds * fps)))
    if is_video:
        ff.run(["-loglevel", "error", "-y", "-i", src, "-filter_complex",
                f"[0:v]{_fit_over_blur(width, height)},fps={fps},tpad=stop_mode=clone:stop_duration={seconds + 1:.3f}[v]",
                "-map", "[v]", "-frames:v", str(n)] + X264_SEG + [out], heartbeat=30.0)
        return out
    composed = os.path.splitext(out)[0] + "-composed.png"
    ff.run(["-loglevel", "error", "-y", "-i", src, "-filter_complex", f"[0:v]{_fit_over_blur(width, height)}[v]",
            "-map", "[v]", "-frames:v", "1", composed])
    try:
        return _pushin(ff, composed, out, seconds, width, height, fps, push)
    finally:
        try:
            os.remove(ff.path(composed))
        except OSError:
            pass


def seg_endcard(ff: Ffmpeg, out: str, seconds: float, width: int, height: int, fontfile: str, lines: list,
                background: str | None = None, color: str = "#1d3557", fps: int = 24) -> str:
    """The end card: a still (darkened a little) or a plain colour, with `lines` drawn centred:
    [(text file relative to the work directory, size as a fraction of the height, y as a fraction)]."""
    n = max(1, int(round(seconds * fps)))
    draws = []
    for textfile, size, y in lines:
        fs = max(8, int(round(size * height)))
        draws.append(f"drawtext=fontfile={fontfile}:textfile={textfile}:fontsize={fs}:fontcolor=white:"
                     f"x=(w-text_w)/2:y={int(y * height)}-text_h/2:box=1:boxcolor=black@0.45:"
                     f"boxborderw={max(4, fs // 3)}")
    chain = ",".join(draws) if draws else "null"
    if background:
        args = ["-loglevel", "error", "-y", "-loop", "1", "-framerate", str(fps), "-i", background, "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,"
                f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.25:t=fill,{chain}"]
    else:
        if not re.match(r"^#[0-9a-fA-F]{6}$", color or ""):
            color = "#1d3557"
        args = ["-loglevel", "error", "-y", "-f", "lavfi", "-i",
                f"color=c=0x{color[1:]}:s={width}x{height}:r={fps}:d={seconds + 1:.3f}", "-vf", f"setsar=1,{chain}"]
    ff.run(args + ["-frames:v", str(n)] + X264_SEG + [out], heartbeat=30.0)
    return out


def concat(ff: Ffmpeg, segments: list, out: str, list_file: str) -> str:
    with open(os.path.join(ff.cwd, list_file), "w") as f:
        for s in segments:
            f.write(f"file '{s}'\n")
    ff.run(["-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out])
    return out


# ---------------------------------------------------------------- audio

def mix_audio(ff: Ffmpeg, voice: str, music: str | None, out: str, total_s: float, music_db: float = -12.0,
              duck_threshold: float = 0.02, duck_ratio: float = 8.0, duck_attack_ms: float = 20.0,
              duck_release_ms: float = 400.0) -> str:
    """The voice over the whole length, and the music bed ducked under it, as one 48 kHz WAV."""
    T = f"{total_s:.3f}"
    vo = f"[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:{T},asetpts=N/SR/TB"
    if not music:
        ff.run(["-loglevel", "error", "-y", "-i", voice, "-filter_complex", f"{vo}[out]", "-map", "[out]",
                "-c:a", "pcm_s16le", out])
        return out
    fade = max(0.0, total_s - 1.5)
    graph = (f"{vo},asplit=2[vo][sc];"
             f"[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:{T},asetpts=N/SR/TB,"
             f"volume={music_db:g}dB,afade=t=in:d=0.4,afade=t=out:st={fade:.3f}:d=1.5[m];"
             f"[m][sc]sidechaincompress=threshold={duck_threshold:g}:ratio={duck_ratio:g}:attack={duck_attack_ms:g}:"
             f"release={duck_release_ms:g}[duck];"
             f"[vo][duck]amix=inputs=2:duration=first:normalize=0[out]")
    ff.run(["-loglevel", "error", "-y", "-i", voice, "-i", music, "-filter_complex", graph, "-map", "[out]",
            "-c:a", "pcm_s16le", out])
    return out


def loudnorm(ff: Ffmpeg, src: str, out: str, lufs: float = -14.0, tp: float = -1.5, lra: float = 11.0) -> dict:
    """Two-pass loudnorm to `lufs` integrated and `tp` true peak. Returns the first pass's
    measurement."""
    target = f"I={lufs:g}:TP={tp:g}:LRA={lra:g}"
    text = ff.stderr(["-i", src, "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-"])
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.S)
    measured = {}
    if m:
        try:
            measured = json.loads(m[-1])
        except ValueError:
            measured = {}
    keys = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    ok = all(k in measured and re.match(r"^-?\d+(\.\d+)?$", str(measured[k]).strip()) for k in keys)
    if ok:
        second = (f"loudnorm={target}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
                  f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
                  f"offset={measured['target_offset']}:linear=true")
    else:  # silence or a parse problem: one dynamic pass
        second = f"loudnorm={target}"
    ff.run(["-loglevel", "error", "-y", "-i", src, "-af", second, "-ar", "48000", "-c:a", "pcm_s16le", out])
    return measured


# ---------------------------------------------------------------- the final encode

def video_kbps(seconds: float, max_bytes: float, audio_kbps: int = 160, cap: int = 8000, floor: int = 150) -> int:
    budget = 0.85 * max_bytes * 8 / 1000.0 / max(1.0, seconds)
    return int(max(floor, min(cap, budget - audio_kbps)))


def final_encode(ff: Ffmpeg, video: str, audio: str, out: str, seconds: float, max_bytes: float,
                 captions: str | None = None, fonts_dir: str | None = None, label_file: str | None = None,
                 label_font: str | None = None, width: int = 1080, height: int = 1920, audio_kbps: int = 160,
                 encoders=("h264_videotoolbox", "libx264")) -> dict:
    """Burn in captions and the label, encode H.264 + AAC with +faststart under max_bytes."""
    chain = []
    if captions:
        chain.append(f"ass={captions}" + (f":fontsdir={fonts_dir}" if fonts_dir else ""))
    if label_file and label_font:
        fs = max(8, int(round(0.022 * height)))
        chain.append(f"drawtext=fontfile={label_font}:textfile={label_file}:fontsize={fs}:fontcolor=white@0.92:"
                     f"x={int(0.05 * width)}:y={int(0.13 * height)}:box=1:boxcolor=black@0.35:boxborderw={max(3, fs // 4)}")
    chain.append("format=yuv420p")
    kbps = video_kbps(seconds, max_bytes, audio_kbps)
    attempts = []
    order = list(encoders)
    for _round in range(4):
        last_err = None
        for enc in order:
            venc = (["-c:v", "h264_videotoolbox", "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.3)}k",
                     "-bufsize", f"{kbps * 2}k", "-allow_sw", "1", "-profile:v", "high"] if enc == "h264_videotoolbox"
                    else ["-c:v", "libx264", "-preset", "medium", "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.2)}k",
                          "-bufsize", f"{kbps}k", "-profile:v", "high"])
            res = ff.run(["-loglevel", "error", "-y", "-i", video, "-i", audio, "-filter_complex",
                          f"[0:v]{','.join(chain)}[v]", "-map", "[v]", "-map", "1:a:0"] + venc +
                         ["-c:a", "aac", "-b:a", f"{audio_kbps}k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart",
                          "-t", f"{seconds:.3f}", out], check=False, heartbeat=120.0)
            if res.returncode == 0 and os.path.exists(ff.path(out)) and os.path.getsize(ff.path(out)) > 0:
                size = os.path.getsize(ff.path(out))
                attempts.append({"encoder": enc, "kbps": kbps, "bytes": size})
                if size <= max_bytes:
                    return {"encoder": enc, "video_kbps": kbps, "bytes": size, "attempts": attempts}
                # Over the cap: aim lower, and let libx264 (which keeps to its rate) try next.
                audio_bytes = audio_kbps * 1000 / 8.0 * seconds
                share = max(0.05, (max_bytes * 0.9 - audio_bytes) / max(1.0, size - audio_bytes))
                kbps = max(100, int(kbps * min(0.9, share)))
                order = [e for e in order if e == "libx264"] or order
                break
            last_err = (res.stderr or b"")[-800:].decode("utf-8", "replace")
            attempts.append({"encoder": enc, "kbps": kbps, "error": last_err[-300:]})
        else:
            raise AssemblyError(f"no H.264 encoder worked: {last_err}")
    raise AssemblyError(f"could not get the ad under {max_bytes / 1e6:.0f} MB: {attempts}")
