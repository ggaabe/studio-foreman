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
     +faststart, at a bitrate that keeps the file under max_mb (re-encoded smaller if not; an
     MP4 still over the cap is deleted, never left as if finished).

Drawn text (the end card, the label) is drawn verbatim (drawtext expansion=none: '20%' is text)
and measured with the font's own advance widths first: a line too wide for the frame is shrunk,
or broken in two. A caption phrase whose word cannot wrap (a long web address) is drawn smaller.
Shot boundaries are cut on the plan's absolute frame numbers (round(t * 24)), so rounding never
adds up across shots against the voice track and the captions.

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
LABEL_SIZE, LABEL_MIN_SIZE = 0.022, 0.018  # the AI label's font size (shrunk to fit, never below the minimum)
LABEL_Y = 0.13                        # its top, just below the top band
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


def _same_file(a: str, b: str) -> bool:
    """`b` exists and holds the same bytes as `a`."""
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                x, y = fa.read(1 << 20), fb.read(1 << 20)
                if x != y:
                    return False
                if not x:
                    return True
    except OSError:
        return False


# ---------------------------------------------------------------- text widths (fitting drawn text)

class FontMetrics:
    """Advance widths from a TrueType/OpenType font (the first face of a collection): cmap formats
    4 and 12 and hmtx, read with the standard library. Text drawn with drawtext or libass is
    measured with the font it is drawn with, so a line that would run off the frame is shrunk (or
    wrapped) before it is drawn."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = f.read(64 << 20)
        base = struct.unpack(">I", data[12:16])[0] if data[:4] == b"ttcf" else 0
        num = struct.unpack(">H", data[base + 4: base + 6])[0]
        tables = {}
        for i in range(num):
            rec = data[base + 12 + 16 * i: base + 28 + 16 * i]
            tables[rec[:4]] = struct.unpack(">I", rec[8:12])[0]
        head, hhea, hmtx, cmap = (tables.get(t) for t in (b"head", b"hhea", b"hmtx", b"cmap"))
        if None in (head, hhea, hmtx, cmap):
            raise ValueError("not a TrueType/OpenType font with head, hhea, hmtx and cmap")
        self.upem = struct.unpack(">H", data[head + 18: head + 20])[0] or 1000
        self.n_metrics = struct.unpack(">H", data[hhea + 34: hhea + 36])[0]
        self.data, self.hmtx = data, hmtx
        self.sub = None
        n_sub = struct.unpack(">H", data[cmap + 2: cmap + 4])[0]
        best = None
        for i in range(n_sub):
            pid, eid, off = struct.unpack(">HHI", data[cmap + 4 + 8 * i: cmap + 12 + 8 * i])
            fmt = struct.unpack(">H", data[cmap + off: cmap + off + 2])[0]
            rank = {(3, 10): 0, (0, 4): 1, (3, 1): 2, (0, 3): 3}.get((pid, eid), 9)
            if fmt in (4, 12) and (best is None or rank < best[0]):
                best = (rank, cmap + off, fmt)
        if best is None:
            raise ValueError("no usable cmap subtable")
        self.sub, self.fmt = best[1], best[2]
        self._cache: dict = {}

    def glyph(self, cp: int) -> int:
        d, o = self.data, self.sub
        if self.fmt == 12:
            n = struct.unpack(">I", d[o + 12: o + 16])[0]
            for i in range(n):
                a, b, g = struct.unpack(">III", d[o + 16 + 12 * i: o + 28 + 12 * i])
                if a <= cp <= b:
                    return g + cp - a
            return 0
        if cp > 0xFFFF:
            return 0
        segs = struct.unpack(">H", d[o + 6: o + 8])[0] // 2
        ends = o + 14
        starts = ends + 2 * segs + 2
        deltas = starts + 2 * segs
        ranges = deltas + 2 * segs
        for i in range(segs):
            end = struct.unpack(">H", d[ends + 2 * i: ends + 2 * i + 2])[0]
            if cp > end:
                continue
            start = struct.unpack(">H", d[starts + 2 * i: starts + 2 * i + 2])[0]
            if cp < start:
                return 0
            delta = struct.unpack(">h", d[deltas + 2 * i: deltas + 2 * i + 2])[0]
            ro = struct.unpack(">H", d[ranges + 2 * i: ranges + 2 * i + 2])[0]
            if ro == 0:
                return (cp + delta) & 0xFFFF
            addr = ranges + 2 * i + ro + 2 * (cp - start)
            g = struct.unpack(">H", d[addr: addr + 2])[0]
            return (g + delta) & 0xFFFF if g else 0
        return 0

    def advance(self, ch: str) -> float:
        """The advance of one character in em units."""
        cp = ord(ch)
        if cp not in self._cache:
            g = self.glyph(cp)
            idx = min(g, max(0, self.n_metrics - 1))
            adv = struct.unpack(">H", self.data[self.hmtx + 4 * idx: self.hmtx + 4 * idx + 2])[0]
            self._cache[cp] = adv / float(self.upem) if g else _guess_em(ch)
        return self._cache[cp]


_METRICS: dict = {}


def _guess_em(ch: str) -> float:
    """A generous width for a character of a bold sans font (when the font cannot be read): text
    measured with it is never wider on screen than this says."""
    if ch in " .,:;'!|il\u2019":
        return 0.36
    if ch in "ftrIj-()[]/":
        return 0.48
    if ch in "mwMW@%":
        return 0.98
    if ch.isupper() or ch.isdigit():
        return 0.76
    return 0.66


def text_width(font_file: str | None, text: str, size: float) -> float:
    """How wide `text` is at `size` px with `font_file` (a generous estimate without one)."""
    fm = None
    if font_file:
        if font_file not in _METRICS:
            try:
                _METRICS[font_file] = FontMetrics(font_file)
            except (OSError, ValueError, struct.error, IndexError):
                _METRICS[font_file] = None
        fm = _METRICS[font_file]
    em = sum(fm.advance(ch) if fm is not None else _guess_em(ch) for ch in str(text))
    return em * float(size)


def fit_text(font_file: str | None, text: str, size: int, max_w: float, min_size: int, max_lines: int = 2) -> tuple:
    """([lines], size): `text` at `size` if it fits in max_w px, else shrunk (down to min_size),
    else broken into at most `max_lines` lines at spaces and sized so the widest fits."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    w = text_width(font_file, text, size)
    if w <= max_w or not text:
        return [text], int(size)
    shrunk = int(size * max_w / w)
    words = text.split(" ")
    if shrunk >= min_size or len(words) < 2 or max_lines < 2:
        return [text], max(4, shrunk)
    best = None
    for cut in range(1, len(words)):
        a, b = " ".join(words[:cut]), " ".join(words[cut:])
        widest = max(text_width(font_file, a, size), text_width(font_file, b, size))
        if best is None or widest < best[0]:
            best = (widest, [a, b])
    widest, lines = best
    return lines, int(min(size, size * max_w / widest)) if widest > max_w else int(size)


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read(4096)
    except (OSError, UnicodeDecodeError):
        return None


def install_font(src: str, fonts_dir: str, name: str) -> dict:
    """Copy a font into the ad's fonts directory under a filter-safe name; returns
    {"path" (relative to the work directory), "family", "bold"}."""
    os.makedirs(fonts_dir, exist_ok=True)
    ext = os.path.splitext(src)[1].lower() or ".ttf"
    dst = os.path.join(fonts_dir, f"{name}{ext}")
    if not _same_file(src, dst):  # missing, or left half-copied by a killed run: copy it again, atomically
        tmp = f"{dst}.tmp-{os.getpid()}"
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
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


def gray_frame(ff: Ffmpeg, path: str, width: int, height: int, t: float) -> bytes | None:
    """The frame at `t` seconds as width x height 8-bit grey, or None."""
    res = ff.run(["-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", path, "-frames:v", "1", "-vf",
                  f"scale={width}:{height}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                 check=False)
    data = res.stdout or b""
    return data[:width * height] if res.returncode == 0 and len(data) >= width * height else None


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


def caption_chunks(words: list, max_words: int = 4, max_chars: int = 24, measure=None, avail=None) -> list:
    """Words grouped into short on-screen phrases (never across script lines; a phrase ends after
    punctuation): [[word, ...], ...]. A word too wide for the frame (a long web address) is a
    phrase of its own, so only it is drawn smaller."""
    chunks, cur = [], []
    for w in words:
        if not _ass_text(w.get("text", "")):
            continue
        wide = measure is not None and avail and measure(_ass_text(w["text"])) > avail
        if cur and (wide or w.get("line") != cur[-1].get("line") or len(cur) >= max_words
                    or len(" ".join(x["text"] for x in cur + [w])) > max_chars):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if wide or re.search(r"[.!?;:,]$", w["text"].strip()):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def caption_events(words: list, end_limit: float | None = None, hold_s: float = 0.25, measure=None, avail=None,
                   size: int | None = None, **kw) -> list:
    """[(start, end, ASS text)]: one event per word, showing its phrase with that word highlighted.
    No event starts or lasts past end_limit (the end card). A phrase whose widest word does not fit
    in `avail` px at the caption size is drawn with a smaller font ({\\fs}) so nothing is clipped."""
    chunks = caption_chunks(words, measure=measure, avail=avail, **kw)
    events = []
    for ci, chunk in enumerate(chunks):
        nxt = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else None
        stop = chunk[-1]["end"] + hold_s
        if nxt is not None:
            stop = min(stop, nxt)
        if end_limit is not None:
            stop = min(stop, end_limit)
        prefix = ""
        if measure is not None and avail and size:
            widest = max(measure(_ass_text(x["text"])) for x in chunk)
            if widest > avail:
                prefix = f"{{\\fs{max(8, int(size * avail / widest))}}}"
        for wi, w in enumerate(chunk):
            a = w["start"] if wi else chunk[0]["start"]
            b = chunk[wi + 1]["start"] if wi + 1 < len(chunk) else stop
            if end_limit is not None:
                if a >= end_limit:
                    break
                b = min(b, end_limit)
            if b - a < 0.02:
                continue
            parts = []
            for xi, x in enumerate(chunk):
                t = _ass_text(x["text"])
                parts.append(f"{{\\c{HIGHLIGHT}}}{t}{{\\c{WHITE}}}" if xi == wi else t)
            events.append((a, b, prefix + " ".join(parts)))
    return events


def caption_style(width: int, height: int) -> dict:
    size = max(10, int(round(CAPTION_SIZE * height)))
    return {"size": size, "margin_v": int(round(CAPTION_MARGIN_V * height)),
            "margin_lr": int(round(0.08 * width)), "outline": max(1, int(round(size * 0.07))),
            "shadow": max(0, int(round(size * 0.03)))}


def write_ass(path: str, words: list, width: int, height: int, family: str, bold: bool = True,
              end_limit: float | None = None, font_file: str | None = None) -> int:
    """Write the captions file; returns the number of events. Words are measured with `font_file`
    (a generous estimate without it) so a phrase never runs past the side margins."""
    st = caption_style(width, height)
    avail = width - 2 * st["margin_lr"]
    events = caption_events(words, end_limit=end_limit, measure=lambda t: text_width(font_file, t, st["size"]),
                            avail=avail, size=st["size"])
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


def _frames(seconds: float, fps: int, frames: int | None) -> int:
    return max(1, int(frames)) if frames else max(1, int(round(seconds * fps)))


def seg_take(ff: Ffmpeg, src: str, out: str, seconds: float, width: int, height: int, fps: int = 24,
             frames: int | None = None) -> str:
    """A generated take: cover-scaled, cropped, exactly `frames` frames (default: `seconds` at
    `fps`; the last frame is held if the take is short)."""
    n = _frames(seconds, fps, frames)
    ff.run(["-loglevel", "error", "-y", "-i", src, "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height},"
            f"setsar=1,fps={fps},tpad=stop_mode=clone:stop_duration={n / float(fps) + 1:.3f}",
            "-frames:v", str(n)] + X264_SEG + [out], heartbeat=30.0)
    return out


def _pushin(ff: Ffmpeg, still: str, out: str, seconds: float, width: int, height: int, fps: int, push: float,
            frames: int | None = None) -> str:
    n = _frames(seconds, fps, frames)
    zoom = f"1+{push:.4f}*on/{max(1, n - 1)}"
    ff.run(["-loglevel", "error", "-y", "-i", still, "-vf",
            f"scale={2 * width}:{2 * height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={2 * width}:{2 * height},zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={n}:s={width}x{height}:fps={fps},setsar=1",
            "-frames:v", str(n)] + X264_SEG + [out], heartbeat=30.0)
    return out


def seg_still(ff: Ffmpeg, src: str, out: str, seconds: float, width: int, height: int, fps: int = 24,
              push: float = 0.06, frames: int | None = None) -> str:
    """A still with a slow push-in."""
    return _pushin(ff, src, out, seconds, width, height, fps, push, frames)


def _fit_over_blur(width: int, height: int) -> str:
    fw, fh = int(width * 0.92) // 2 * 2, int(height * 0.92) // 2 * 2
    return (f"split=2[a][b];[a]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
            f"boxblur=20:2,eq=brightness=-0.08[bg];[b]scale={fw}:{fh}:force_original_aspect_ratio=decrease,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def _fill_from_top(width: int, height: int) -> str:
    # a phone-shaped capture covers the frame; what is cut is its bottom, never its headline
    return (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height}:(iw-{width})/2:0,setsar=1")


PORTRAIT_CAPTURE = 0.75  # width/height at or below this fills the 9:16 frame instead of floating over a blur


def seg_screen(ff: Ffmpeg, src: str, out: str, seconds: float, width: int, height: int, fps: int = 24,
               is_video: bool = True, push: float = 0.05, frames: int | None = None) -> str:
    """A real screen capture: a portrait (phone) still fills the frame from its top, anything else is
    fitted over a blurred copy of itself; a still capture also gets the slow push-in."""
    n = _frames(seconds, fps, frames)
    if is_video:
        ff.run(["-loglevel", "error", "-y", "-i", src, "-filter_complex",
                f"[0:v]{_fit_over_blur(width, height)},fps={fps},"
                f"tpad=stop_mode=clone:stop_duration={n / float(fps) + 1:.3f}[v]",
                "-map", "[v]", "-frames:v", str(n)] + X264_SEG + [out], heartbeat=30.0)
        return out
    from .media import image_size
    size = image_size(ff.path(src))
    portrait = bool(size and size[1] and size[0] / float(size[1]) <= PORTRAIT_CAPTURE)
    composed = os.path.splitext(out)[0] + "-composed.png"
    ff.run(["-loglevel", "error", "-y", "-i", src, "-filter_complex",
            f"[0:v]{_fill_from_top(width, height) if portrait else _fit_over_blur(width, height)}[v]",
            "-map", "[v]", "-frames:v", "1", composed])
    try:
        return _pushin(ff, composed, out, seconds, width, height, fps, push, n)
    finally:
        try:
            os.remove(ff.path(composed))
        except OSError:
            pass


def _drawtext(fontfile: str, textfile: str, fs: int, x: str, y: str, color: str, box: str, border: int) -> str:
    # expansion=none: the text is drawn as written ('20%' and '%{pts}' are text, not expansions)
    return (f"drawtext=fontfile={fontfile}:textfile={textfile}:expansion=none:fontsize={fs}:fontcolor={color}:"
            f"x={x}:y={y}:box=1:boxcolor={box}:boxborderw={border}")


def fitted_lines(ff: Ffmpeg, fontfile: str, textfile: str, fs: int, max_w: float, min_fs: int) -> list:
    """[(text file, font size)]: the text file as one line if it fits in max_w px (shrunk down to
    min_fs if need be), else its text broken into two lines written beside it. A file that cannot
    be read is drawn as it is."""
    text = _read_text(ff.path(textfile))
    if text is None:
        return [(textfile, fs)]
    lines, size = fit_text(ff.path(fontfile), text, fs, max_w, min_fs)
    if len(lines) == 1:
        return [(textfile, size)]
    out = []
    stem = os.path.splitext(textfile)[0]
    for i, line in enumerate(lines):
        rel = f"{stem}-l{i}.txt"
        with open(ff.path(rel), "w", encoding="utf-8") as f:
            f.write(line)
        out.append((rel, size))
    return out


# Over a still, the end card's text sits in this band (fractions of the height): below the
# character's face and above the bottom strip where TikTok and Reels draw their own buttons.
ENDCARD_BAND = (0.57, 0.83)
ENDCARD_LINES_OVER_STILL = (("price", 0.068, 0.64), ("url", 0.042, 0.72), ("cta", 0.038, 0.78))
ENDCARD_LINES_PLAIN = (("price", 0.075, 0.40), ("url", 0.045, 0.52), ("cta", 0.04, 0.60))


def seg_endcard(ff: Ffmpeg, out: str, seconds: float, width: int, height: int, fontfile: str, lines: list,
                background: str | None = None, color: str = "#1d3557", fps: int = 24, frames: int | None = None) -> str:
    """The end card: a still (darkened a little) or a plain colour, with `lines` drawn centred:
    [(text file relative to the work directory, size as a fraction of the height, y as a fraction)].
    Each line is measured with the font and shrunk (or broken in two) to stay inside the frame."""
    n = _frames(seconds, fps, frames)
    draws = []
    for textfile, size, y in lines:
        fs = max(8, int(round(size * height)))
        parts = fitted_lines(ff, fontfile, textfile, fs, 0.88 * width, max(8, int(round(0.6 * fs))))
        step = 1.25 * parts[0][1]
        top = y * height - step * (len(parts) - 1) / 2.0
        for i, (tf, pfs) in enumerate(parts):
            draws.append(_drawtext(fontfile, tf, pfs, "(w-text_w)/2", f"{int(top + i * step)}-text_h/2", "white",
                                   "black@0.45", max(4, pfs // 3)))
    chain = ",".join(draws) if draws else "null"
    if background:
        # the character stays clear above a darkened band that holds the text (ENDCARD_BAND)
        top, bottom = ENDCARD_BAND
        args = ["-loglevel", "error", "-y", "-loop", "1", "-framerate", str(fps), "-i", background, "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,"
                f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.10:t=fill,"
                f"drawbox=x=0:y=ih*{top}:w=iw:h=ih*{bottom - top:.3f}:color=black@0.55:t=fill,{chain}"]
    else:
        if not re.match(r"^#[0-9a-fA-F]{6}$", color or ""):
            color = "#1d3557"
        args = ["-loglevel", "error", "-y", "-f", "lavfi", "-i",
                f"color=c=0x{color[1:]}:s={width}x{height}:r={fps}:d={n / float(fps) + 1:.3f}", "-vf",
                f"setsar=1,{chain}"]
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
        fs = max(8, int(round(LABEL_SIZE * height)))
        parts = fitted_lines(ff, label_font, label_file, fs, 0.90 * width, max(8, int(round(LABEL_MIN_SIZE * height))))
        y = int(LABEL_Y * height)
        for tf, pfs in parts:
            chain.append(_drawtext(label_font, tf, pfs, str(int(0.05 * width)), str(y), "white@0.92", "black@0.35",
                                   max(3, pfs // 4)))
            y += int(round(1.3 * pfs))
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
            _remove(ff.path(out))
            raise AssemblyError(f"no H.264 encoder worked: {last_err}")
    _remove(ff.path(out))  # an MP4 over the cap is not a finished ad: never leave one behind
    raise AssemblyError(f"could not get the ad under {max_bytes / 1e6:.0f} MB: {attempts}")


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def changed_share(a: bytes, b: bytes, width: int, rows: tuple, threshold: int = 40) -> float:
    """The share of pixels in rows [r0, r1) of two grey frames that differ by more than threshold."""
    r0, r1 = rows
    lo, hi = r0 * width, min(len(a), len(b), r1 * width)
    if hi <= lo:
        return 0.0
    return sum(1 for x, y in zip(a[lo:hi], b[lo:hi]) if abs(x - y) > threshold) / float(hi - lo)


def burned_in(ff: Ffmpeg, plain: str, final: str, width: int, height: int, caption_times: list,
              label: bool, rows: tuple | None = None) -> dict:
    """Did the captions and the label actually get drawn? Frames of the final cut are compared with
    the same frames of the cut before captions and label (drawtext or libass can fail silently: a
    font libass cannot match, text drawtext would not render). `rows`: (top, bottom) as fractions of
    the height where the sampled captions sit (default: the caption band above the bottom band).
    Returns {"captions": share or None, "label": share or None, "ok_captions", "ok_label"}."""
    w = min(width, 360) // 2 * 2
    h = max(2, int(round(height * w / float(width)))) // 2 * 2
    cap_rows = (int((1 - SAFE_BOTTOM - 0.36) * h), int((1 - SAFE_BOTTOM + 0.02) * h))
    if rows is not None:
        top, bottom = max(0.0, float(rows[0])), min(1.0, float(rows[1]))
        cap_rows = (int(top * h), max(int(top * h) + 1, int(round(bottom * h))))
    label_rows = (int((LABEL_Y - 0.01) * h), int((LABEL_Y + 2.8 * LABEL_SIZE + 0.01) * h))
    caps, labs = [], []
    for t in list(caption_times[:3]) or ([0.5] if label else []):
        a, b = gray_frame(ff, plain, w, h, t), gray_frame(ff, final, w, h, t)
        if not a or not b:
            continue
        if caption_times:
            caps.append(changed_share(a, b, w, cap_rows))
        labs.append(changed_share(a, b, w, label_rows))
    cap = max(caps) if caps else None
    lab = max(labs) if labs else None
    return {"captions": None if cap is None else round(cap, 5), "label": None if lab is None else round(lab, 5),
            "ok_captions": cap is not None and cap >= 0.002, "ok_label": (not label) or (lab is not None and lab >= 0.001),
            "frames": len(caps), "size": [w, h]}
