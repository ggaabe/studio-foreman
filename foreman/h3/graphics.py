"""Ad graphics for the h3 edit: the screen insert, the price card, the end card and the hook headline.

Every frame is drawn with Pillow and piped as rawvideo rgb24 into ffmpeg (libx264 veryfast crf 16,
yuv420p, an explicit bt709 matrix and bt709 tags, no audio, exactly n frames, checked by decoding).
No clip sits still: each is rendered per frame on the shared brand background (the brand's bg
under a soft radial glow of its accent that slowly grows), its content pushed in continuously,
with motion events (a tap, a pulse, an underline wipe) keyed to word onsets given in seconds from
the clip's first frame.

Everything site-specific comes in as data, nothing is detected: the palette and the logo from
`brand` ({name, bg, accent, accent_pressed, highlight, text, muted, logo_png}, colours "#rrggbb"),
the font files from `fonts` ({display, bold, logo}), and every region of a page capture as boxes
in the capture's own pixels ([x0, y0, x1, y1], x1 and y1 exclusive): the crop, the paint-outs
(claims the page makes that the ad must not show), the button a tap presses, the words an
underline marks. Real page crops sit on rounded cards with a hairline edge; the logo is the
brand's own PNG, or else a plain pill with the brand name (never cropped from a capture).

Drawn pixels stay inside the platform-safe zone x 60-940, y 270-1240 on every frame and every
layout is centred on x 500 (the right edge carries the platforms' buttons); the screen insert,
which keeps its captions, also stays above the caption band (y 1110).

Errors: texts or boxes that cannot be drawn as given raise LayoutError (a content problem); a
missing font, capture, logo or ffmpeg raises FileNotFoundError; an encode that fails or does not
decode to n frames raises RuntimeError.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
SAFE = (60, 270, 940, 1240)    # platform-safe zone: no drawn pixel outside it
CX = 500                       # layouts centre on x 500
GLOW = (500, 640)              # centre of the brand glow (and of the end card's push)
CAPTION_TOP = 1110             # the caption band starts below this y
K = 2                          # supersampling of the pushed layers
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
INK = 24                       # a capture pixel this far (sum of |dRGB|) from the card colour is ink
_fonts, _faces, _glow = {}, {}, {}


class LayoutError(ValueError):
    """Texts or boxes that cannot be drawn as given (the template's content, not the machine)."""


# ---------------------------------------------------------------- palette, fonts, logo

def _rgb(v, what: str) -> tuple:
    if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v):
        return tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))
    raise LayoutError(f"{what}: {v!r} is not a #rrggbb colour")


def _mix(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def palette(brand: dict) -> dict:
    """RGB tuples of the brand colours, plus the tones derived from them: `pill` (the logo pill's
    body, bg 3.7% of the way to text), `border` (card and pill hairlines, bg 12.2% to text),
    `headline` (the hook headline's box, the accent 6% lighter) and `logo_text` (the name on a
    drawn pill, the accent 40% lighter)."""
    if not isinstance(brand, dict):
        raise LayoutError("brand must be an object")
    p = {}
    for key in ("bg", "accent", "accent_pressed", "highlight", "text", "muted"):
        if key not in brand:
            raise LayoutError(f"brand: missing {key!r}")
        p[key] = _rgb(brand[key], f"brand.{key}")
    p["pill"] = _mix(p["bg"], p["text"], 0.037)
    p["border"] = _mix(p["bg"], p["text"], 0.122)
    p["headline"] = _mix(p["accent"], WHITE, 0.06)
    p["logo_text"] = _mix(p["accent"], WHITE, 0.40)
    return p


def _need(path, what: str) -> str:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path!r}")
    return path


def _face(path: str, prefer=("demibold", "semibold", "bold", "medium")):
    """(file, face index) for a font file. In a collection (.ttc/.otc) the first face whose style
    name is in prefer (spaces and case ignored) wins, else face 0."""
    _need(path, "font")
    if not path.lower().endswith((".ttc", ".otc")):
        return path
    if path not in _faces:
        styles = []
        for i in range(64):
            try:
                styles.append(ImageFont.truetype(path, 20, index=i).getname()[1].replace(" ", "").lower())
            except (OSError, ValueError):
                break
        _faces[path] = next((styles.index(s) for s in prefer if s in styles), 0)
    return (path, _faces[path])


def _fontset(fonts: dict) -> dict:
    """{display, bold, logo} as font paths or (collection, face) pairs, each checked to exist."""
    if not isinstance(fonts, dict):
        raise LayoutError("fonts must be an object")
    out = {"display": _need(fonts.get("display"), "display font"), "bold": _need(fonts.get("bold"), "bold font")}
    out["logo"] = _face(fonts["logo"]) if fonts.get("logo") else out["bold"]
    return out


def _font(path, size):
    """path is a font file or a (collection file, face index) pair."""
    key = (path, int(round(size)))
    if key not in _fonts:
        p, idx = path if isinstance(path, tuple) else (path, 0)
        _fonts[key] = ImageFont.truetype(p, key[1], index=idx)
    return _fonts[key]


def logo(brand: dict, fonts: dict, height, max_w=None) -> Image.Image:
    """The brand's logo as an RGBA sprite `height` px tall (narrower than max_w, shrinking if it has
    to): brand.logo_png scaled, else a plain pill (bg-toned body, hairline edge) with brand.name in
    the logo font. None when the brand has neither."""
    height = int(round(height))
    png = brand.get("logo_png")
    if png:
        im = Image.open(_need(png, "logo")).convert("RGBA")
        w = max(1, round(im.width * height / im.height))
        if max_w and w > max_w:
            height, w = max(1, round(height * max_w / w)), int(max_w)
        return im.resize((w, height), Image.LANCZOS)
    name = " ".join(str(brand.get("name") or "").split())
    if not name:
        return None
    return _name_pill(name, palette(brand), _fontset(fonts)["logo"], height, max_w)


def _name_pill(name: str, pal: dict, font, height: int, max_w=None) -> Image.Image:
    ss = 4
    h = height * ss
    size = height * 0.40 * ss
    pad = h * 0.34
    while True:
        f = _font(font, size)
        l, _, r, _ = f.getbbox(name, anchor="ls")
        if not max_w or (2 * pad + r - l) / ss <= max_w or size <= 8 * ss:
            break
        size -= ss
    cap = -f.getbbox("H", anchor="ls")[1]
    w = int(math.ceil((2 * pad + r - l) / ss)) * ss
    bw = max(1, round(height / 55))
    im = Image.new("RGBA", (w, h), pal["border"] + (255,))
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), (h - 1) / 2, fill=255)
    im.putalpha(m)
    body = Image.new("RGBA", (w - 2 * bw * ss, h - 2 * bw * ss), pal["pill"] + (255,))
    m = Image.new("L", body.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, body.width - 1, body.height - 1), (body.height - 1) / 2, fill=255)
    body.putalpha(m)
    im.alpha_composite(body, (bw * ss, bw * ss))
    ImageDraw.Draw(im).text((pad - l, (h + cap) / 2), name, font=f, fill=pal["logo_text"], anchor="ls")
    return im.resize((w // ss, height), Image.LANCZOS)


# ---------------------------------------------------------------- encode and measure

def _run_info(path: str, ffmpeg: str) -> str:
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", path], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    return r.stderr


def _decode_count(path: str, ffmpeg: str):
    """Frames and duration of the first video stream, by decoding every frame with ffmpeg -f framecrc."""
    r = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-map", "0:v:0", "-f", "framecrc", "-"],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
    tb, pts = 1.0, []
    for ln in r.stdout.splitlines():
        if ln.startswith("#tb 0:"):
            num, den = ln.split(":")[1].split("/")
            tb = int(num) / int(den)
        elif ln and not ln.startswith("#"):
            f = ln.split(",")
            pts.append((int(f[2]), int(f[3])))
    if not pts:
        return 0, 0.0
    return len(pts), (pts[-1][0] + pts[-1][1] - pts[0][0]) * tb


def probe(path: str, ffmpeg: str) -> dict:
    """Decoded frame count and duration of a clip plus what ffmpeg reports about its first video
    stream: {frames, duration, size (w, h), pix_fmt, color [range, space, ...], rate, audio}.
    ffmpeg only (the Studio's ffmpeg has no ffprobe beside it)."""
    frames, dur = _decode_count(path, ffmpeg)
    info = _run_info(path, ffmpeg)
    m = re.search(r"Video: [^,]*, (\w+)(?:\(([^)]*)\))?, (\d+)x(\d+)", info)
    rate = re.search(r"Video: .*?([\d.]+) fps", info)
    return {"frames": frames, "duration": dur,
            "size": (int(m.group(3)), int(m.group(4))) if m else None,
            "pix_fmt": m.group(1) if m else None,
            "color": [s.strip() for s in (m.group(2) or "").split(",") if s.strip()] if m else [],
            "rate": rate.group(1) if rate else None,
            "audio": bool(re.search(r"Stream #\S+.*: Audio:", info))}


def freezes(path: str, ffmpeg: str, noise: float = 0.003, dur: float = 1.0) -> list:
    """[(start, end)] of the stills ffmpeg freezedetect finds in a clip (end None if it runs to the end)."""
    r = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", path, "-map", "0:v:0",
                        "-vf", f"freezedetect=n={noise}:d={dur}", "-f", "null", "-"],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
    out = []
    for ln in r.stderr.splitlines():
        if "freeze_start:" in ln:
            out.append([float(ln.split("freeze_start:")[1].split()[0]), None])
        elif "freeze_end:" in ln and out:
            out[-1][1] = float(ln.split("freeze_end:")[1].split()[0])
    return [tuple(f) for f in out]


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _encode(frame, n, out: str, ffmpeg: str, fps) -> str:
    """Pipe frame(i) for i < n into libx264 (RGB to yuv420p with the bt709 matrix, tv range, bt709
    tags; the mp4 timescale every h3 segment shares, so concat -c copy lines up) and check the
    result decodes to n frames of n/fps s. Written to a temp file first, so `out` is either complete
    or untouched."""
    n = _count(n)
    _need(ffmpeg, "ffmpeg")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    root, ext = os.path.splitext(out)
    tmp = f"{root}.tmp-{os.getpid()}{ext or '.mp4'}"
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-frames:v", str(n),
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                  "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-color_range", "tv", "-an", "-r", str(fps), "-video_track_timescale", str(512 * int(fps)),
           "-movflags", "+faststart", "-f", "mp4", tmp]
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            try:
                for i in range(n):
                    im = frame(i)
                    if im.size != (W, H) or im.mode != "RGB":
                        raise RuntimeError(f"frame {i} is {im.mode} {im.size}, want RGB {W}x{H}")
                    p.stdin.write(im.tobytes())
                p.stdin.close()
            except BrokenPipeError:  # ffmpeg died: its exit code and stderr say why
                try:
                    p.stdin.close()
                except OSError:
                    pass
            rc = p.wait()
        except BaseException:
            p.kill()
            p.wait()
            try:
                p.stdin.close()
            except OSError:
                pass
            _remove(tmp)
            raise
        if rc:
            err.seek(0)
            msg = err.read().decode("utf-8", "replace").strip()[-600:]
            _remove(tmp)
            raise RuntimeError(f"ffmpeg failed on {out} (exit {rc}): {msg}")
    try:
        frames, dur = _decode_count(tmp, ffmpeg)
        if frames != n or abs(dur - n / fps) > 1e-3:
            raise RuntimeError(f"{out}: {frames} frames / {dur} s, want {n} / {n / fps}")
        os.replace(tmp, out)
    except BaseException:
        _remove(tmp)
        raise
    return out


def _count(n) -> int:
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError(f"frame count must be a positive integer, got {n!r}")
    return int(n)


def _at(t, fps):
    """Frame index of an event t s after the clip's first frame (None stays None)."""
    return None if t is None else int(round(t * fps))


# ---------------------------------------------------------------- drawing helpers

def _text(text, path, size, fill, stroke=0, sfill=BLACK):
    """Ink-tight RGBA sprite of one line; returns (img, baseline y in img, cap height)."""
    f = _font(path, size)
    l, t, r, b = f.getbbox(text, anchor="ls", stroke_width=stroke)
    im = Image.new("RGBA", (r - l, b - t), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((-l, -t), text, font=f, fill=fill, anchor="ls",
                            stroke_width=stroke, stroke_fill=sfill)
    return im, -t, -f.getbbox("H", anchor="ls")[1]


def _fit(text, path, size, maxw, stroke=0):
    """Largest size <= size at which the line's ink is at most maxw wide."""
    while size > 8:
        l, _, r, _ = _font(path, size).getbbox(text, anchor="ls", stroke_width=stroke)
        if r - l <= maxw:
            break
        size -= 1
    return size


def _rmask(w, h, r, ss=4):
    """Antialiased rounded-rectangle alpha mask."""
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def _ease_out(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def _ease_back(u, c=1.70158):
    u = min(max(u, 0.0), 1.0)
    return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2


def _pulse(i, f0, amp, frames=8):
    """Scale 1 -> 1 + amp -> 1 over frames frames from frame f0 (1 outside it or when f0 is None)."""
    if f0 is None or not 0 <= i - f0 < frames:
        return 1.0
    return 1 + amp * math.sin(math.pi * (i - f0) / (frames - 1))


def _fade(spr, a):
    if a >= 1:
        return spr
    s = spr.copy()
    s.putalpha(spr.getchannel("A").point(lambda v: int(v * a + 0.5)))
    return s


def _put(layer, spr, cx, cy, s=1.0):
    """Alpha-composite spr scaled by s, centred at (cx, cy) in layer px."""
    if s != 1:
        spr = spr.resize((max(1, round(spr.width * s)), max(1, round(spr.height * s))), Image.LANCZOS)
    layer.alpha_composite(spr, (int(round(cx - spr.width / 2)), int(round(cy - spr.height / 2))))


def _card(img, pad, fill, edge, k=1, radius=24, border=2):
    """img on a rounded card: fill colour, pad px around it, a border px edge (all px times k)."""
    w, h = img.width + 2 * pad * k, img.height + 2 * pad * k
    c = Image.new("RGBA", (w, h), edge + (255,))
    c.putalpha(_rmask(w, h, radius * k))
    inner = Image.new("RGBA", (w - 2 * border * k, h - 2 * border * k), fill + (255,))
    inner.putalpha(_rmask(inner.width, inner.height, (radius - border) * k))
    c.alpha_composite(inner, (border * k, border * k))
    c.alpha_composite(img.convert("RGBA"), (pad * k, pad * k))
    return c


def brand_bg(i, n, brand) -> Image.Image:
    """Frame i of n of the shared background: the brand bg under a soft radial glow of its accent
    (18% at its centre, sigma 400 px at (500, 640)) that grows 1.00->1.05 over the clip. A fixed
    +-1 LSB TPDF dither keeps the glow from banding without adding frame-to-frame noise. brand is
    the brand dict or its palette()."""
    pal = brand if "border" in brand else palette(brand)
    if not _glow:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _glow["d2"] = (xx - GLOW[0]) ** 2 + (yy - GLOW[1]) ** 2
        rng = np.random.default_rng(7)
        _glow["dither"] = rng.random((H, W, 1), dtype=np.float32) - rng.random((H, W, 1), dtype=np.float32)
    z = 1 + 0.05 * i / max(1, n - 1)
    g = 0.18 * np.exp(-_glow["d2"] / (2 * (400 * z) ** 2))
    bg = np.array(pal["bg"], np.float32)
    a = bg + g[..., None] * (np.array(pal["accent"], np.float32) - bg) + _glow["dither"] + 0.5
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def _push(layer, z, cx, cy):
    """The K-times supersampled RGBA layer scaled by z about (cx, cy) in frame px, at frame size."""
    box = (K * (cx - cx / z), K * (cy - cy / z), K * (cx + (W - cx) / z), K * (cy + (H - cy) / z))
    return layer.resize((W, H), Image.LANCZOS, box=box)


def _frame(i, n, pal, layer, z, c):
    """Brand background frame i of n with the layer pushed by z about c over it."""
    im = brand_bg(i, n, pal)
    top = _push(layer, z, *c)
    im.paste(top, (0, 0), top)
    return im


def _ripple(im, cx, cy, r, a):
    """Filled white circle of radius r at alpha a, antialiased, over im."""
    ss, R = 4, int(math.ceil(r)) + 2
    lay = Image.new("L", (2 * R * ss, 2 * R * ss), 0)
    ImageDraw.Draw(lay).ellipse(((R - r) * ss, (R - r) * ss, (R + r) * ss, (R + r) * ss), fill=int(round(255 * a)))
    lay = lay.resize((2 * R, 2 * R), Image.LANCZOS)
    x, y = int(round(cx)) - R, int(round(cy)) - R
    im.paste(WHITE, (x, y, x + 2 * R, y + 2 * R), lay)


# ---------------------------------------------------------------- captures

def _box(v, size, what: str) -> tuple:
    """[x0, y0, x1, y1] (x1, y1 exclusive) as ints, checked to lie inside an image of size (w, h)."""
    if not (isinstance(v, (list, tuple)) and len(v) == 4
            and all(isinstance(c, int) and not isinstance(c, bool) for c in v)):
        raise LayoutError(f"{what}: want [x0, y0, x1, y1] as 4 integers, got {v!r}")
    x0, y0, x1, y1 = v
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise LayoutError(f"{what} {list(v)} is not inside the {size[0]}x{size[1]} capture")
    return tuple(v)


def _inside(b, box) -> bool:
    return box[0] <= b[0] and box[1] <= b[1] and b[2] <= box[2] and b[3] <= box[3]


def _capture(png, crop, paint, what: str):
    """(the capture with its paint-outs filled, the crop box in its px)."""
    im = Image.open(_need(png, f"{what} capture")).convert("RGB")
    box = _box(crop, im.size, f"{what} crop") if crop else (0, 0, im.width, im.height)
    if paint:
        dr = ImageDraw.Draw(im)
        for j, p in enumerate(paint):
            if not (isinstance(p, (list, tuple)) and len(p) == 5):
                raise LayoutError(f"{what} paint {j}: want [x0, y0, x1, y1, '#rrggbb'], got {p!r}")
            x0, y0, x1, y1 = _box(list(p[:4]), im.size, f"{what} paint {j}")
            dr.rectangle((x0, y0, x1 - 1, y1 - 1), fill=_rgb(p[4], f"{what} paint {j} colour"))
    return im, box


def _page_colour(a, box, w=8) -> tuple:
    """The page colour around a crop: the median of the ring w px wide just outside it (the crop's
    own outermost px when it fills the capture)."""
    x0, y0, x1, y1 = box
    hh, ww = a.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - w), max(0, y0 - w), min(ww, x1 + w), min(hh, y1 + w)
    parts = [a[Y0:y0, X0:X1], a[y1:Y1, X0:X1], a[y0:y1, X0:x0], a[y0:y1, x1:X1]]
    ring = np.concatenate([p.reshape(-1, 3) for p in parts])
    if not len(ring):
        c = a[y0:y1, x0:x1]
        ring = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    return tuple(int(v) for v in np.median(ring, 0))


def _pressed(crop, btn, pressed, pal):
    """Copy of the crop with the button btn (crop px) shifted to its pressed colour; antialiased
    edges and the label keep their share (each pixel moves by its share of the button's chroma,
    so a white or dark label stays put). The button colour is the median of its box."""
    c = np.asarray(crop).astype(np.float32).copy()
    base = np.median(c[btn[1]:btn[3], btn[0]:btn[2]].reshape(-1, 3), 0)
    if pressed is None:
        near = np.abs(base - np.array(pal["accent"], np.float32)).sum() <= 24
        target = np.array(pal["accent_pressed"] if near else _mix(tuple(base), BLACK, 0.16), np.float32)
    else:
        target = np.array(_rgb(pressed, "tap_color"), np.float32)
    x0, y0 = max(0, btn[0] - 2), max(0, btn[1] - 2)
    x1, y1 = min(c.shape[1], btn[2] + 2), min(c.shape[0], btn[3] + 2)
    s = c[y0:y1, x0:x1]
    chroma = float(base.max() - base.min())
    if chroma >= 32:
        w = np.clip((s.max(2) - s.min(2)) / chroma, 0, 1)[..., None]
    else:  # a grey button: only its own colour moves
        w = (np.abs(s - base).sum(2) <= INK).astype(np.float32)[..., None]
    c[y0:y1, x0:x1] = s + w * (target - base)
    return Image.fromarray(np.clip(c + 0.5, 0, 255).astype(np.uint8), "RGB")


def _runs(flags, gap=1):
    """[start, end) runs of True, joining runs separated by fewer than gap False values."""
    idx = np.where(flags)[0]
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            out.append((int(s), int(p) + 1))
            s = i
        p = i
    out.append((int(s), int(p) + 1))
    return out


def _price_crop(card: dict):
    """The price card's capture crop: (image, card colour, underlines [(x0, baseline y, x1)] in that
    image's px). The card colour is the crop's median colour. With `lead` > 0 the crop is cut into
    its text lines (runs of rows holding ink against the card colour, cut midway between lines) and
    re-stacked with lead px of card colour between them, glyphs untouched (room for an underline
    to clear the next line)."""
    im, box = _capture(card.get("png"), card.get("crop"), card.get("paint"), "price card")
    crop = im.crop(box)
    a = np.asarray(crop).astype(np.int16)
    fill = tuple(int(v) for v in np.median(a.reshape(-1, 3), 0))
    lead = card.get("lead") or 0
    if isinstance(lead, bool) or not isinstance(lead, int) or not 0 <= lead <= 200:
        raise LayoutError(f"price card lead: want an integer 0..200 (capture px), got {lead!r}")
    cuts = []
    if lead:
        lines = _runs((np.abs(a - np.array(fill)).sum(2) > INK).any(1))
        cuts = [(lines[i][1] + lines[i + 1][0]) // 2 for i in range(len(lines) - 1)]
        bands = [0] + cuts + [crop.height]
        out = Image.new("RGB", (crop.width, crop.height + lead * len(cuts)), fill)
        for i in range(len(bands) - 1):
            out.paste(crop.crop((0, bands[i], crop.width, bands[i + 1])), (0, bands[i] + i * lead))
        crop = out
    unders = []
    for j, u in enumerate(card.get("underline") or []):
        b = _box(u, im.size, f"price card underline {j}")
        if not _inside(b, box):
            raise LayoutError(f"price card underline {j} {list(b)} is not inside the crop {list(box)}")
        y = b[3] - 1 - box[1]  # the last row above the baseline, in crop px, then past the leads above it
        unders.append((b[0] - box[0], y + 1 + lead * sum(1 for c in cuts if c <= y), b[2] - box[0]))
    return crop, fill, unders


# ---------------------------------------------------------------- screen insert

def screen_layout(capture: dict, brand: dict, fonts: dict, zoom=1.03, k=1):
    """Screen insert layout before the push, drawn k times larger, positions in frame px:
    (pill or None, pill xy, card, card xy, card size, tap centre or None, pressed card or None,
    push centre).

    The card is the capture's crop (paint-outs filled) on a rounded card the page's own colour;
    it and the logo above it are scaled so the whole stack, pushed by zoom, stays inside
    x 74-926, y 276-1106 (the caption band starts below 1110).
    """
    pal = palette(brand)
    im, box = _capture(capture.get("png"), capture.get("crop"), capture.get("paint"), "screen")
    crop = im.crop(box)
    btn = None
    if capture.get("tap"):
        t = _box(capture["tap"], im.size, "screen tap")
        if not _inside(t, box):
            raise LayoutError(f"screen tap {list(t)} is not inside the crop {list(box)}")
        btn = (t[0] - box[0], t[1] - box[1], t[2] - box[0], t[3] - box[1])
    page = _page_colour(np.asarray(im).astype(np.int16), box)
    m = 4
    top, bot, xl, xr = 272 + m, CAPTION_TOP - m, 70 + m, 930 - m
    pill = logo(brand, fonts, 62 * k, 700 * k)
    ph, gap, pad = (62, 22, 14) if pill is not None else (0, 0, 14)
    s = min(((bot - top) / zoom - ph - gap - 2 * pad) / crop.height,
            (min(xr - CX, CX - xl) * 2 / zoom - 2 * pad) / crop.width)
    cw, ch = round(crop.width * s), round(crop.height * s)
    if cw < 1 or ch < 1:
        raise LayoutError(f"screen crop {list(box)} is too thin to draw")
    srcs = [crop] + ([_pressed(crop, btn, capture.get("tap_color"), pal)] if btn else [])
    cards = [_card(c.resize((cw * k, ch * k), Image.LANCZOS), pad, page, pal["border"], k) for c in srcs]
    cy = (top + bot) / 2
    y0 = cy - (ph + gap + ch + 2 * pad) / 2
    pxy = (CX - pill.width / k / 2, y0 + (ph - pill.height / k) / 2) if pill is not None else None
    cxy = (CX - (cw + 2 * pad) / 2, y0 + ph + gap)
    bc = None
    if btn:
        bc = (cxy[0] + pad + (btn[0] + btn[2]) / 2 * s, cxy[1] + pad + (btn[1] + btn[3]) / 2 * s)
    return pill, pxy, cards[0], cxy, (cw + 2 * pad, ch + 2 * pad), bc, (cards[1] if btn else None), (CX, cy)


def _screen_push(n, fps=24):
    """Total push of an n-frame screen insert: 0.03 (1.00->1.03), but never slower than 0.03 per
    2.25 s. Its flat card moves least of the three clips: at a fixed 1.03 an insert over ~4.5 s
    reads as a freeze (freezedetect n=0.003 d=1.0 hits at 144 frames)."""
    return 0.03 * max(1.0, (n - 1) / (2.25 * fps))


def screen_frames(capture: dict, n, brand: dict, fonts: dict, fps=24, tap_at=None):
    """frame(i) -> RGB image of the screen insert (see screen_insert)."""
    n = _count(n)
    pal = palette(brand)
    zt = _screen_push(n, fps)  # the layout is fitted to the last frame's zoom
    pill, pxy, card, cxy, _, bxy, pcard, c = screen_layout(capture, brand, fonts, zoom=1 + zt, k=K)
    if tap_at is not None and bxy is None:
        raise LayoutError("a tap time was given but the capture has no tap box")
    layers = []
    for cd in (card, pcard) if pcard is not None else (card,):
        lay = Image.new("RGBA", (W * K, H * K), (0, 0, 0, 0))
        if pill is not None:
            lay.alpha_composite(pill, (round(pxy[0] * K), round(pxy[1] * K)))
        lay.alpha_composite(cd, (round(cxy[0] * K), round(cxy[1] * K)))
        layers.append(lay)
    rf = _at(tap_at, fps)

    def frame(i):
        z = 1 + zt * i / max(1, n - 1)
        k = -1 if rf is None else i - rf
        im = _frame(i, n, pal, layers[1 if 0 <= k < 4 else 0], z, c)
        if 0 <= k < 8:  # tap ripple on the button: radius 60->95, alpha 0.45->0
            bx, by = bxy
            _ripple(im, c[0] + (bx - c[0]) * z, c[1] + (by - c[1]) * z,
                    60 + 35 * _ease_out(k / 8), 0.45 * (1 - k / 8))
        return im
    return frame


def screen_insert(capture: dict, n, out: str, ffmpeg: str, brand: dict, fonts: dict, fps=24, tap_at=None) -> str:
    """The brand logo over a real page crop on a card, on the brand background, push 1.00->1.03
    (faster past 55 frames, see _screen_push); tap_at s after the first frame: a white tap ripple on
    the tap box (8 frames) and the button pressed (4 frames).

    capture: {png (abs), crop [x0,y0,x1,y1]|None (None: the whole capture), paint [[x0,y0,x1,y1,
    "#rrggbb"], ...]|None, tap [x0,y0,x1,y1]|None, tap_color "#rrggbb"|None (the pressed colour;
    default the brand's accent_pressed for an accent-coloured button, else the button 16% darker)}.
    Returns out."""
    n = _count(n)
    return _encode(screen_frames(capture, n, brand, fonts, fps, tap_at), n, out, ffmpeg, fps)


# ---------------------------------------------------------------- price card

def price_layout(card: dict, brand: dict, fonts: dict, k=1):
    """Price insert sprites drawn k times larger, positions in frame px: (items, push centre).

    items are (name, RGBA, x, y): "price" is the big price that pops, ("line", j) the qualifier
    lines, "card" the capture crop on its card, ("underline", j) the bars wiped in under the
    underline boxes. Without a capture the price and lines are centred on the glow.
    """
    pal, fs = palette(brand), _fontset(fonts)
    big = " ".join(str(card.get("big") or "").split())
    if not big:
        raise LayoutError("price card: empty big text")
    lines = [" ".join(str(s).split()) for s in card.get("lines") or []]
    if any(not s for s in lines) or len(lines) > 4:
        raise LayoutError(f"price card: want 0-4 non-empty lines, got {card.get('lines')!r}")
    items = []
    size = _fit(big, fs["display"], 220, 830, 9)
    spr, base, cap = _text(big, fs["display"], size * k, pal["highlight"], 9 * k)
    y = 300
    items.append(("price", spr, CX - spr.width / k / 2, y))
    yb = y + base / k
    last_base = yb
    for j, line in enumerate(lines):
        s, b, c = _text(line, fs["bold"], _fit(line, fs["bold"], 50, 780) * k, pal["text"])
        top = yb + 62 + j * 68
        items.append((("line", j), s, CX - s.width / k / 2, top + (c - b) / k))
        last_base = top + c / k
    if not card.get("png"):
        if card.get("underline"):
            raise LayoutError("price card: underline boxes need a capture")
        dy = GLOW[1] - (y + last_base) / 2
        return [(nm, s, x, yy + dy) for nm, s, x, yy in items], (CX, GLOW[1])
    crop, fill, unders = _price_crop(card)
    cw, pad = 820, 24
    ctop = last_base + 64
    sc = (cw - 2 * pad) / crop.width
    room = 1200 - ctop - 2 * pad  # the card's bottom, pushed 1.04 about the centre, stays above y 1240
    if crop.height * sc > room:
        sc = room / crop.height
        cw = crop.width * sc + 2 * pad
    img = crop.resize((max(1, round(crop.width * sc * k)), max(1, round(crop.height * sc * k))), Image.LANCZOS)
    cimg = _card(img, pad, fill, pal["border"], k)
    items.append(("card", cimg, CX - cw / 2, ctop))
    for j, (u0, ub, u1) in enumerate(unders):
        uw = max(1, round((u1 - u0) * sc))
        bar = Image.new("RGBA", (uw * k, 6 * k), pal["accent"] + (255,))
        bar.putalpha(_rmask(uw * k, 6 * k, 3 * k))
        items.append((("underline", j), bar, CX - cw / 2 + pad + u0 * sc, ctop + pad + ub * sc + 6))
    return items, (CX, (y + ctop + cimg.height / k) / 2)


def price_frames(card: dict, n, brand: dict, fonts: dict, fps=24, pulse_at=None, underline_at=None):
    """frame(i) -> RGB image of the price insert (see price_insert)."""
    n = _count(n)
    pal = palette(brand)
    items, c = price_layout(card, brand, fonts, K)
    pl = card.get("pulse_line", 0)
    names = {nm for nm, _, _, _ in items}
    if pulse_at is not None and ("line", pl) not in names:
        raise LayoutError(f"a pulse time was given but the price card has no line {pl!r} to pulse")
    if underline_at is not None and ("underline", 0) not in names:
        raise LayoutError("an underline time was given but the price card has no underline box")
    base = Image.new("RGBA", (W * K, H * K), (0, 0, 0, 0))
    price, pulse, bars = None, None, []
    for name, s, x, y in items:
        if name == "price":
            price = (s, x * K, y * K)
        elif name == ("line", pl):
            pulse = (s, x * K, y * K)
        elif name[0] == "underline":
            bars.append((s, x * K, y * K))
        else:
            base.alpha_composite(s, (round(x * K), round(y * K)))
    pf, uf = _at(pulse_at, fps), _at(underline_at, fps)

    def frame(i):
        lay = base.copy()
        s, x, y = price  # pop 0.85 -> ~1.01 -> 1.00 over frames 0-3, full opacity throughout
        _put(lay, s, x + s.width / 2, y + s.height / 2, 0.85 + 0.15 * _ease_back(i / 3) if i < 3 else 1)
        if pulse is not None:
            s, x, y = pulse
            _put(lay, s, x + s.width / 2, y + s.height / 2, _pulse(i, pf, 0.08))
        u = 1.0 if uf is None else _ease_out((i - uf + 1) / 6)
        for s, x, y in bars if u > 0 else ():  # wiped in left to right over 6 frames from underline_at
            lay.alpha_composite(s.crop((0, 0, max(1, round(s.width * u)), s.height)), (round(x), round(y)))
        return _frame(i, n, pal, lay, 1 + 0.04 * i / max(1, n - 1), c)
    return frame


def price_insert(card: dict, n, out: str, ffmpeg: str, brand: dict, fonts: dict, fps=24,
                 pulse_at=None, underline_at=None) -> str:
    """The big price (pop-in, full opacity) and its qualifier lines from frame 0 over the real
    capture crop on a card, push 1.00->1.04; pulse_at s: line pulse_line pulses 1.00->1.08->1.00
    over 8 frames; underline_at s: 6 px accent bars under the underline boxes wipe in left to
    right over 6 frames (None: drawn from frame 0).

    card: {big, lines [0-4], png (abs)|None, crop|None, paint|None, lead (capture px, default 0),
    underline [[x0,y0,x1,y1], ...]|None (y1 at the words' baseline; the bar hangs 6 px below it),
    pulse_line (default 0)}. Returns out."""
    n = _count(n)
    return _encode(price_frames(card, n, brand, fonts, fps, pulse_at, underline_at), n, out, ffmpeg, fps)


# ---------------------------------------------------------------- end card

def endcard_layout(card: dict, brand: dict, fonts: dict, k=1, fps=24):
    """End card elements drawn k times larger, in frame px: list of (name, RGBA, x, y, start s).

    Everything starts at 0 s at full opacity except the footnote lines, which fade in over frames
    3-8 (start 3 / fps)."""
    pal, fs = palette(brand), _fontset(fonts)
    big = " ".join(str(card.get("big") or "").split())
    url = " ".join(str(card.get("url") or "").split())
    if not big or not url:
        raise LayoutError("end card: big and url are required")
    lines = [" ".join(str(s).split()) for s in card.get("lines") or []]
    foot = [" ".join(str(s).split()) for s in card.get("footnote") or []]
    if any(not s for s in lines + foot):
        raise LayoutError("end card: blank line")
    els = []
    pill = logo(brand, fonts, 110 * k, 840 * k)
    if pill is not None:
        els.append(("logo", pill, CX - pill.width / k / 2, 290 + (110 - pill.height / k) / 2, 0.0))

    def line(name, text, path, size, fill, top, stroke=0, t0=0.0):
        s, _, _ = _text(text, path, _fit(text, path, size, 840, stroke) * k, fill, stroke * k)
        els.append((name, s, CX - s.width / k / 2, top - stroke, t0))
        return top - stroke + s.height / k
    lead = " ".join(str(card.get("lead") or "").split())
    if lead:
        line("lead", lead, fs["bold"], 56, pal["text"], 473)
    line("price", big, fs["display"], 200, pal["highlight"], 542, 9)
    for j, q in enumerate(lines):
        line(("line", j), q, fs["bold"], 46, pal["muted"], 748 + 58 * j)
    utop = 768 + 58 * len(lines)
    size = _fit(url, fs["display"], 72, 840 - 2 * 46)
    t, tb, tc = _text(url, fs["display"], size * k, pal["text"])
    ph, pw = 130 * k, t.width + 2 * 46 * k
    pill_url = Image.new("RGBA", (pw, ph), pal["accent"] + (255,))
    pill_url.putalpha(_rmask(pw, ph, ph / 2 - 0.5))
    pill_url.alpha_composite(t, (round((pw - t.width) / 2), round((ph - tc) / 2 - (tb - tc))))
    els.append(("url", pill_url, CX - pw / k / 2, utop, 0.0))
    bottom = utop + 130
    for j, f in enumerate(foot):
        bottom = line(("foot", j), f, fs["bold"], 40, pal["text"], utop + 166 + 50 * j, t0=3 / fps)
    if bottom > 1212:  # pushed 1.04 about the glow (frame 0) the bottom must stay above y 1240: more lines
        els = _shrink(els, (1212 - 290) / (bottom - 290), (CX, 290), k)  # than two and two shrink the card
    return els


def _shrink(els, s, c, k):
    """Elements (name, RGBA drawn k times larger, x, y, t0) scaled by s about c (frame px)."""
    out = []
    for nm, spr, x, y, t0 in els:
        cx, cy = x + spr.width / k / 2, y + spr.height / k / 2
        spr = spr.resize((max(1, round(spr.width * s)), max(1, round(spr.height * s))), Image.LANCZOS)
        out.append((nm, spr, c[0] + (cx - c[0]) * s - spr.width / k / 2, c[1] + (cy - c[1]) * s - spr.height / k / 2, t0))
    return out


def _endcard_push(n, fps=24):
    """Total push after the settle: 0.03 (1.00->1.03), but never slower than 0.03 per 2.25 s. Held
    long, the end card's slow push reads as a freeze (freezedetect n=0.003 d=1.0 hits at 240 frames,
    and its d=0.5 windows already reach 0.8 s at 192)."""
    return 0.03 * max(1.0, (n - 7) / (2.25 * fps))


def endcard_frames(card: dict, n, brand: dict, fonts: dict, fps=24, pulse_at=None):
    """frame(i) -> RGB image of the end card (see endcard)."""
    n = _count(n)
    pal = palette(brand)
    els = endcard_layout(card, brand, fonts, K, fps)
    zt = _endcard_push(n, fps)
    s = min(1.0, 1.04 / (1 + zt))  # a push past 1.04 shrinks the card about the glow to stay in the safe zone
    if s < 1:
        els = _shrink(els, s, GLOW, K)
    base = Image.new("RGBA", (W * K, H * K), (0, 0, 0, 0))
    late = []
    for name, s, x, y, t0 in els:
        if name == "url" or t0 > 0:
            late.append((name, s, x * K, y * K))
        else:
            base.alpha_composite(s, (round(x * K), round(y * K)))
    pf = _at(pulse_at, fps)

    def frame(i):
        lay = base.copy()
        for name, s, x, y in late:
            if name == "url":
                _put(lay, s, x + s.width / 2, y + s.height / 2, _pulse(i, pf, 0.06))
            elif i >= 3:  # footnotes: 1/6 at frame 3 .. full at frame 8
                lay.alpha_composite(_fade(s, min(1.0, (i - 2) / 6)), (round(x), round(y)))
        if i <= 5:  # settle 1.04 -> 1.00, ease-out, then a slow push 1.00 -> 1.03 to the last frame
            z = 1.04 - 0.04 * _ease_out(i / 5)
        else:
            z = 1 + zt * (i - 6) / max(1, n - 7)
        return _frame(i, n, pal, lay, z, GLOW)
    return frame


def endcard(card: dict, n, out: str, ffmpeg: str, brand: dict, fonts: dict, fps=24, pulse_at=None) -> str:
    """Brand end card: logo, the lead line, the big price, the qualifier lines and the URL pill at
    full opacity from frame 0 (settle 1.04->1.00 over frames 0-5), the footnote lines fading in
    over frames 3-8, push 1.00->1.03 from frame 6 (faster past 61 frames, see _endcard_push);
    pulse_at s: the URL pill pulses 1.00->1.06->1.00.

    card: {lead, big, lines, footnote, url}. Returns out."""
    n = _count(n)
    return _encode(endcard_frames(card, n, brand, fonts, fps, pulse_at), n, out, ffmpeg, fps)


# ---------------------------------------------------------------- headline

def _wrap2(text, path, size, maxw):
    """One line if it is short, else the two-line split with the narrowest longer line."""
    f = _font(path, size)

    def wid(s):
        b = f.getbbox(s, anchor="ls")
        return b[2] - b[0]
    w = text.split()
    if len(w) < 2 or wid(text) <= min(maxw, 640):
        return [text], wid(text)
    best = min(([" ".join(w[:i]), " ".join(w[i:])] for i in range(1, len(w))),
               key=lambda ls: max(map(wid, ls)))
    return best, max(map(wid, best))


def headline(text: str, out_png: str, brand: dict, fonts: dict, size=76) -> str:
    """Transparent 1080x1920 PNG: the hook headline (<= 2 lines) in the display font, text colour,
    on a rounded box of the headline tone with a soft shadow, top y 280, centre x 500.

    size is the largest font size; it shrinks until the box and its visible shadow fit the safe
    zone, and raises LayoutError below 40 px. Written atomically; returns out_png.
    """
    pal, fs = palette(brand), _fontset(fonts)
    padx, pady, shadow = 44, 32, 22  # shadow: how far its visible (alpha > 8) blur reaches past the box
    text = " ".join(str(text or "").split())
    if not text:
        raise LayoutError("empty headline")
    maxw = SAFE[2] - SAFE[0] - 2 * shadow - 2 * padx
    while True:
        lines, lw = _wrap2(text, fs["display"], size, maxw)
        if lw <= maxw:
            break
        if size <= 40:
            raise LayoutError(f"headline does not fit two lines of 40 px inside the safe zone: {text!r}")
        size -= 2
    f = _font(fs["display"], size)
    cap = -f.getbbox("H", anchor="ls")[1]
    pitch = round(size * 1.18)
    bw, bh = lw + 2 * padx, pady + cap + pitch * (len(lines) - 1) + pady + round(size * 0.06)
    x0, y0 = round(CX - bw / 2), 280
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + 8, x0 + bw, y0 + bh + 8), 26, fill=110)
    img.putalpha(sh.filter(ImageFilter.GaussianBlur(14)))
    box = Image.new("RGBA", (bw, bh), pal["headline"] + (255,))
    box.putalpha(_rmask(bw, bh, 26))
    img.alpha_composite(box, (x0, y0))
    dr = ImageDraw.Draw(img)
    for j, ln in enumerate(lines):
        dr.text((CX, y0 + pady + cap + j * pitch), ln, font=f, fill=pal["text"], anchor="ms")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    tmp = f"{out_png}.tmp-{os.getpid()}"
    try:
        img.save(tmp, format="PNG")
        os.replace(tmp, out_png)
    except BaseException:
        _remove(tmp)
        raise
    return out_png
