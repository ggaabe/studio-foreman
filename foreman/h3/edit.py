"""The worker's `edit` command: the v6 recipe's edit, driven by a request.

Every beat is its chosen take on a 24 fps grid, trimmed to the line's acoustic bounds (+tail, +hold),
minus removed frame spans; picture, voice, word times and graphics events all map through the same
spans. Cutaways cover parts of a beat's picture while the voice runs on (J/L cuts on word onsets): a
punch-in (crop-then-scale of the same take), a screen insert, a price card, the end card, or a
text-free still with a push-in. Pauses are removed only under graphic cutaways and only where the
take is silent by energy (its loudest 10 ms under -45 dBFS); the worker never transcribes. Then the
beats' voices are joined, the music laid under them and the sum mastered, word-highlight captions
written and burned, the file encoded bt709, a sidecar SRT and the publish gate written, and the
result measured (frames, freezes, punch cuts, colour) for the timeline.

Fixes over the recipe: a beat's caption position goes by beat, not by event time (B2); removals lie
only inside graphic cutaways, cutaways are sorted and never overlap, so a punch slice always matches
its voice (B3); an unknown cutaway kind is refused (B4); an event word (tap, pulse, underline) must
fall inside its cutaway (B9); the publish gate is keyed to the file's hash (B10); RGB sources are
converted with the bt709 matrix their tags claim (B11).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import wave

import numpy as np

from . import audio, captions, ff
from .ff import FPS, ContentError, InfraError

SR = 48000
SPF = SR // FPS                         # voice samples per picture frame
KINDS = ("punch", "screen", "price", "endcard", "still")
GRAPHIC = ("screen", "price", "endcard", "still")
EVENTS = {"screen": (("tap_i", "tap_at", 0.0),),
          "price": (("pulse_i", "pulse_at", 0.04), ("underline_i", "underline_at", 0.08)),
          "endcard": (("pulse_i", "pulse_at", 0.0),),
          "punch": (), "still": ()}     # (index key, graphics argument, lead s before the word's onset)
OUTPUTS = ("mp4", "video", "audio", "voice", "ass", "srt", "timeline", "beats_dir", "segments_dir", "publish")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SILENCE_DB = -45.0                      # a pause is cut only when its loudest 10 ms is under this
CUT_LEAD = 0.08                         # the picture leaves this long before a cutaway's word


class RequestError(ContentError):
    """A malformed request: retrying the same request cannot help."""


def _graphics():
    try:
        from . import graphics
    except ImportError as e:
        raise InfraError(f"the graphics module cannot load: {e}") from e
    return graphics


def _draw(what, fn, *args, **kw):
    """A graphics call with its failures in the worker's classes: texts or boxes that cannot be drawn
    as given are the template's (content), a missing font, capture or logo the machine's (infra)."""
    graphics = _graphics()
    try:
        return fn(*args, **kw)
    except graphics.LayoutError as e:
        raise ContentError(f"{what}: {e}") from e
    except FileNotFoundError as e:
        raise InfraError(f"{what}: {e}") from e


# ---------------------------------------------------------------- request

def _need(ok, msg):
    if not ok:
        raise RequestError(f"bad request: {msg}")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _pair(v, what, even=False):
    _need(isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) and x > 0 for x in v),
          f"{what} must be [width, height]")
    _need(not even or all(x % 2 == 0 for x in v), f"{what} must be even")
    return int(v[0]), int(v[1])


def _input(path, what):
    """An input file (absolute, or relative to the work directory); a missing one is infrastructure."""
    _need(isinstance(path, str) and path and "\x00" not in path, f"{what} must be a path")
    full = os.path.abspath(path)
    if not os.path.isfile(full):
        raise InfraError(f"{what} missing: {path}")
    return full


def _output(rel, what):
    _need(isinstance(rel, str) and rel and not os.path.isabs(rel)
          and ".." not in rel.replace("\\", "/").split("/"), f"{what} must be a relative path inside the work directory")
    return os.path.abspath(rel)


def _index(c, key, n, what, lo=0):
    v = c.get(key)
    if v is None:
        return None
    _need(isinstance(v, int) and not isinstance(v, bool) and lo <= v < n, f"{what}.{key} must be a word index in [{lo}, {n})")
    return v


def _cutaway(c, j, words, req, where):
    what = f"{where}.cutaways[{j}]"
    _need(isinstance(c, dict), f"{what} must be an object")
    kind = c.get("kind")
    _need(kind in KINDS, f"{what}: unknown cutaway kind {kind!r}")          # B4: never silently skipped
    n = len(words)
    at = _index(c, "at_i", n, what)
    _need(at is not None, f"{what}.at_i is required")
    out = {"kind": kind, "at_i": at, "until_i": _index(c, "until_i", n, what, at + 1),
           "captions": c.get("captions") is not False, "close_pauses": bool(c.get("close_pauses")),
           "close_gap": bool(c.get("close_gap")), "capture": c.get("capture"), "still": c.get("still")}
    for key, _, _ in EVENTS[kind]:
        out[key] = _index(c, key, n, what)
    if kind == "punch":
        _need(not (out["close_pauses"] or out["close_gap"]), f"{what}: a punch-in never removes pauses")
        out["scale"] = float(c.get("scale") if c.get("scale") is not None else 1.2)
        _need(1.0 <= out["scale"] <= 4.0, f"{what}.scale must be in [1, 4]")
        ay = c.get("anchor_y")
        _need(ay is None or _num(ay), f"{what}.anchor_y must be a number or null")
        out["anchor_y"] = ay
    elif kind == "screen":
        _need(out["capture"] in req["captures"], f"{what}: capture {out['capture']!r} is not in captures")
    elif kind == "price":
        _need(req["price_card"] is not None, f"{what}: no price_card in the request")
    elif kind == "still":
        _need(out["still"] in req["stills"], f"{what}: still {out['still']!r} is not in stills")
        push = c.get("push") if c.get("push") is not None else 0.06
        _need(_num(push) and 0 <= push <= 0.5, f"{what}.push must be in [0, 0.5]")
        out["push"] = float(push)
    _need(not out["close_gap"] or out["until_i"] is None, f"{what}: close_gap needs a cutaway that runs to the beat's end")
    return out


def _beat(b, i, req):
    where = f"beats[{i}]"
    _need(isinstance(b, dict), f"{where} must be an object")
    bid = b.get("id")
    _need(isinstance(bid, str) and SAFE_ID.match(bid), f"{where}.id must be a plain name")
    line = b.get("line")
    _need(isinstance(line, str) and line.split(), f"{where}.line must be text")
    ws = b.get("words")
    _need(isinstance(ws, list) and len(ws) == len(line.split()),
          f"{where}.words must be the line's {len(line.split())} display words with take times")
    words = []
    for k, w in enumerate(ws):
        _need(isinstance(w, dict) and isinstance(w.get("text"), str) and _num(w.get("start")) and _num(w.get("end")),
              f"{where}.words[{k}] must be {{text, start, end}}")
        s0 = max(0.0, float(w["start"]))                 # an aligner's zero or inverted span is noise, not a failure
        words.append({"text": w["text"], "start": s0, "end": max(s0, float(w["end"]))})
    span = b.get("span")
    _need(isinstance(span, list) and len(span) == 3 and _num(span[0]) and _num(span[1]) and span[1] >= span[0] >= 0
          and (span[2] is None or _num(span[2])), f"{where}.span must be [start, end, next|null]")
    tail, hold = b.get("tail"), b.get("hold") or 0.0
    _need(tail is None or (_num(tail) and tail >= 0), f"{where}.tail must be a number or null")
    _need(_num(hold) and 0 <= hold <= 10, f"{where}.hold must be in [0, 10]")
    crop = b.get("crop")
    _need(crop is None or (isinstance(crop, dict) and _num(crop.get("scale", 1.0)) and crop.get("scale", 1.0) >= 1
                           and (crop.get("anchor_y") is None or _num(crop.get("anchor_y")))),
          f"{where}.crop must be {{scale >= 1, anchor_y}} or null")
    pos = b.get("caption_pos")
    _need(pos is None or (isinstance(pos, list) and len(pos) == 2 and all(_num(v) for v in pos)),
          f"{where}.caption_pos must be [x, y] or null")
    cuts = b.get("cutaways") or []
    _need(isinstance(cuts, list), f"{where}.cutaways must be a list")
    cuts = [_cutaway(c, j, words, req, where) for j, c in enumerate(cuts)]
    for j in range(1, len(cuts)):           # B3: sorted and never overlapping, by word
        p = cuts[j - 1]
        _need(p["until_i"] is not None and p["until_i"] <= cuts[j]["at_i"],
              f"{where}: cutaways must be in line order and not overlap")
    return {"id": bid, "k": b.get("k", i), "take": _input(b.get("take"), f"{where}.take"),
            "wav16": _input(b.get("wav16"), f"{where}.wav16"), "line": line, "words": words,
            "span": [float(span[0]), float(span[1]), None if span[2] is None else float(span[2])],
            "tail": None if tail is None else float(tail), "hold": float(hold),
            "crop": None if crop is None else {"scale": float(crop.get("scale", 1.0)), "anchor_y": crop.get("anchor_y")},
            "caption_pos": None if pos is None else [pos[0], pos[1]], "headline": bool(b.get("headline")),
            "cutaways": cuts}


def parse(req) -> dict:
    """The request with every path resolved and every field checked (RequestError, InfraError)."""
    _need(isinstance(req, dict), "the request must be an object")
    r = {"ffmpeg": ff.need_ffmpeg(req.get("ffmpeg"))}
    _need(req.get("fps", FPS) == FPS, f"fps must be {FPS}")
    r["out"] = _pair(req.get("out", [1080, 1920]), "out", even=True)
    _need(isinstance(req.get("ad_id"), str) and req["ad_id"], "ad_id must be text")
    r["ad_id"] = req["ad_id"]
    fonts = req.get("fonts")
    _need(isinstance(fonts, dict), "fonts must be an object")
    r["fonts"] = dict(fonts)
    for k in ("caption", "bold", "display"):
        r["fonts"][k] = _input(fonts.get(k), f"font {k}")
    if fonts.get("logo") is not None:
        r["fonts"]["logo"] = _input(fonts["logo"], "font logo")
    brand = req.get("brand")
    _need(isinstance(brand, dict), "brand must be an object")
    r["brand"] = dict(brand)
    if brand.get("logo_png") is not None:
        r["brand"]["logo_png"] = _input(brand["logo_png"], "brand logo")
    cap = req.get("captions") or {}
    _need(isinstance(cap, dict), "captions must be an object")
    margins = cap.get("margins", [100, 180])
    _need(isinstance(margins, list) and len(margins) == 2 and all(isinstance(v, int) and v >= 0 for v in margins)
          and sum(margins) < r["out"][0] - 100, "captions.margins must be [left, right]")
    solo = cap.get("solo") or []
    _need(isinstance(solo, list) and all(isinstance(s, str) for s in solo), "captions.solo must be a list of words")
    r["captions"] = {"size": cap.get("size", 92), "margin_v": cap.get("margin_v", 710), "margins": margins, "solo": solo}
    _need(_num(r["captions"]["size"]) and 8 <= r["captions"]["size"] <= 400, "captions.size must be in [8, 400]")
    _need(_num(r["captions"]["margin_v"]) and 0 <= r["captions"]["margin_v"] < r["out"][1], "captions.margin_v")
    hl = req.get("headline")
    _need(hl is None or (isinstance(hl, str) and hl.strip()), "headline must be text or null")
    r["headline"] = hl
    r["music"] = None if req.get("music") is None else _input(req["music"], "music")
    for k, d in (("music_under", 14.0), ("bed_lift", 5.0), ("lufs", -14.0), ("tp", -1.5)):
        v = req.get(k, d)
        _need(_num(v), f"{k} must be a number")
        r[k] = float(v)
    r["safe"] = bool(req.get("safe"))
    caps = req.get("captures") or {}
    _need(isinstance(caps, dict), "captures must be an object")
    r["captures"] = {}
    for cid, c in caps.items():
        _need(isinstance(c, dict), f"captures[{cid!r}] must be an object")
        r["captures"][cid] = dict(c, png=_input(c.get("png"), f"capture {cid}"))
    pc = req.get("price_card")
    _need(pc is None or isinstance(pc, dict), "price_card must be an object or null")
    r["price_card"] = None if pc is None else dict(pc, png=None if pc.get("png") is None else _input(pc["png"], "price card capture"))
    _need(isinstance(req.get("endcard"), dict), "endcard must be an object")
    r["endcard"] = dict(req["endcard"])
    stills = req.get("stills") or {}
    _need(isinstance(stills, dict), "stills must be an object")
    r["stills"] = {sid: _input(p, f"still {sid}") for sid, p in stills.items()}
    outs = req.get("outputs")
    _need(isinstance(outs, dict) and all(k in outs for k in OUTPUTS), f"outputs must name {', '.join(OUTPUTS)}")
    r["outputs"] = {k: _output(outs[k], f"outputs.{k}") for k in OUTPUTS}
    enc = req.get("encode") or {}
    _need(isinstance(enc, dict), "encode must be an object")
    r["encode"] = {"crf": enc.get("crf", 16), "maxrate": str(enc.get("maxrate", "16M")),
                   "bufsize": str(enc.get("bufsize", "32M")), "audio_kbps": enc.get("audio_kbps", 192),
                   "max_mb": enc.get("max_mb", 40.0)}
    _need(isinstance(r["encode"]["crf"], int) and 0 <= r["encode"]["crf"] <= 51, "encode.crf must be 0..51")
    _need(isinstance(r["encode"]["audio_kbps"], int) and 32 <= r["encode"]["audio_kbps"] <= 512, "encode.audio_kbps")
    _need(_num(r["encode"]["max_mb"]) and r["encode"]["max_mb"] > 0, "encode.max_mb must be positive")
    pub = req.get("publish") or {}
    _need(isinstance(pub, dict), "publish must be an object")
    r["publish"] = {"geo": pub.get("geo"), "attribution": pub.get("attribution"), "note": pub.get("note")}
    beats = req.get("beats")
    _need(isinstance(beats, list) and beats, "beats must be a non-empty list")
    r["beats"] = [_beat(b, i, r) for i, b in enumerate(beats)]
    ids = [b["id"] for b in r["beats"]]
    _need(len(set(ids)) == len(ids), "beat ids must be unique")
    return r


# ---------------------------------------------------------------- the beat plan (no rendering)

def refine_ends(words, wav16k, line_end, next_start=None, thresh_db=-35.0):
    """Each word ends at the last 10 ms frame above thresh_db before the next word starts (the line's last word:
    before its acoustic end, or the next heard word when that comes first: the energy up to the next word is that
    word's, not the line's), never earlier than the aligner said: an aligner can pack a spoken address into a span
    that ends before its last syllable is said."""
    db = audio.db10(wav16k)
    out = []
    for i, w in enumerate(words):
        s0, e0 = float(w["start"]), float(w["end"])
        if i + 1 < len(words):
            lim = float(words[i + 1]["start"])
        else:
            lim = line_end if next_start is None else min(float(line_end), float(next_start))
        loud = [f for f in range(int(s0 * 100), min(len(db), int(lim * 100))) if db[f] > thresh_db]
        out.append(dict(w, end=max(e0, min(lim - 0.02, (loud[-1] + 1) / 100)) if loud else e0))
    return out


def plan_beat(b, safe=False):
    """Where beat b's take is cut: bounds, take frames, cutaway frame ranges, verified removals, event times and
    output word times (relative to the beat's first output frame). Raises ContentError when the take cannot be
    edited as planned (a cutaway word outside its cutaway, a cutaway with no frames)."""
    bid, wav16 = b["id"], b["wav16"]
    w0, w1, wn = b["span"]
    head, stop, info = audio.acoustic_bounds(wav16, w0, w1, wn)
    if b["tail"] is not None and info.get("acoustic_end"):
        stop = min(stop, info["acoustic_end"] + b["tail"])
    ws = refine_ends(b["words"], wav16, info.get("acoustic_end") or stop, wn)
    n_take = int(-(-(stop - head + b["hold"]) * FPS // 1))
    tf = lambda t: int(round((t - head) * FPS))             # take time -> take frame
    specs = []
    for c in b["cutaways"]:
        t_at = ws[c["at_i"]]["start"]
        ta = tf(t_at - CUT_LEAD)
        te = tf(ws[c["until_i"]]["start"] - CUT_LEAD) if c["until_i"] is not None else n_take
        if not 0 <= ta < te <= n_take:
            raise ContentError(f"beat {bid}: the {c['kind']} cutaway at {ws[c['at_i']]['text']!r} has no frames "
                               f"inside the take's cut (frames {ta}-{te} of {n_take})")
        specs.append({"c": c, "ta": ta, "te": te, "t_at": t_at})
    for j in range(1, len(specs)):
        if specs[j]["ta"] < specs[j - 1]["te"]:
            raise ContentError(f"beat {bid}: cutaways {j - 1} and {j} overlap in time")
    # removals: pauses under graphic cutaways (close_pauses) and a long pause before an end-of-beat cutaway (close_gap)
    cands, checks = [], []
    if not safe:
        runs = audio.silent_runs(wav16, -40.0, 0.25)
        for j, sp in enumerate(specs):
            c, ta, te = sp["c"], sp["ta"], sp["te"]
            if c["close_pauses"]:
                for rs, re_ in runs:
                    if head + ta / FPS + 0.10 <= rs and re_ <= head + te / FPS - 0.10:
                        f0, f1 = int(-(-(rs + 0.075 - head) * FPS // 1)), int((re_ - 0.075 - head) * FPS)
                        if f1 - f0 >= 2:
                            cands.append((f0, f1, j, "pause"))
            if c["close_gap"]:
                for rs, re_ in runs:
                    if sp["t_at"] - 0.25 <= re_ <= sp["t_at"] + 0.05 and re_ - rs >= 0.40:
                        f0, f1 = int(-(-(rs + 0.10 - head) * FPS // 1)), int((re_ - 0.05 - head) * FPS)
                        if f1 - f0 >= 2:
                            cands.append((f0, f1, j, "gap"))
    kept = []
    for f0, f1, j, how in sorted(cands, key=lambda r: r[0]):
        lvl = audio.span_level(wav16, head + f0 / FPS, head + f1 / FPS)
        if lvl["max_db"] < SILENCE_DB:
            ok, why = True, f"silence by energy (max {lvl['max_db']:.1f} dBFS)"
        else:
            ok, why = False, f"not silence by energy (loudest 10 ms {lvl['max_db']:.1f} dBFS): pause kept"
        if ok and kept and f0 < kept[-1][1]:
            ok, why = False, "overlaps another removal"
        if ok and how == "gap" and j and f0 < specs[j - 1]["te"]:
            ok, why = False, "would start the cutaway inside the one before it: pause kept"
        checks.append({"beat": bid, "take_s": [round(head + f0 / FPS, 3), round(head + f1 / FPS, 3)],
                       "frames": f1 - f0, "ok": ok, "why": why})
        if ok:
            kept.append((f0, f1, j))
            if how == "gap":
                specs[j]["ta"] = f0                          # the picture leaves at the start of the closed pause
    # B3: every removal lies inside a graphic cutaway; cutaways sorted and apart; removals apart
    for f0, f1, j in kept:
        sp = specs[j]
        if not (sp["c"]["kind"] in GRAPHIC and sp["ta"] <= f0 < f1 <= sp["te"]):
            raise RuntimeError(f"beat {bid}: removal {f0}-{f1} outside its cutaway {sp['ta']}-{sp['te']}")
    for j in range(1, len(specs)):
        if specs[j]["ta"] < specs[j - 1]["te"]:
            raise RuntimeError(f"beat {bid}: cutaways overlap after removals")
    removed = sum(f1 - f0 for f0, f1, _ in kept)
    gone = lambda f: sum(min(f, f1) - f0 for f0, f1, _ in kept if f > f0)   # removed frames before take frame f
    of = lambda f: f - gone(f)                                               # take frame -> beat output frame
    ot = lambda t: (t - head) - gone(int(round((t - head) * FPS))) / FPS     # take time -> beat output seconds
    n = n_take - removed
    for sp in specs:
        c = sp["c"]
        a, e = of(sp["ta"]), of(sp["te"])
        sp.update(a=a, e=e, m=e - a, events={})
        if sp["m"] < 1:
            raise ContentError(f"beat {bid}: the {c['kind']} cutaway has no frames left")
        for key, arg, lead in EVENTS[c["kind"]]:
            i = c.get(key)
            if i is None:
                continue
            rel = ot(ws[i]["start"]) - a / FPS               # seconds from the cutaway's first frame
            if not 0 <= rel < sp["m"] / FPS:                 # B9
                raise ContentError(f"beat {bid}: the {c['kind']} cutaway's {key[:-2]} word {ws[i]['text']!r} is "
                                   f"heard at {rel:.2f}s, outside the cutaway (0-{sp['m'] / FPS:.2f}s)")
            sp["events"][arg] = round(max(0.0, rel - lead), 4)
    shifted, last = [], -math.inf
    for w in ws:                                              # no word lies in a removed span: those are silent
        # and none outlasts the beat: its voice stops at `stop`, the next beat's picture starts after it
        ws_, we_ = min(w["start"], stop), min(w["end"], stop)
        s = ws_ - gone(int(round((ws_ - head) * FPS))) / FPS
        e = we_ - gone(int(round((we_ - head) * FPS))) / FPS
        s = max(s, last)
        e = max(e, s)
        last = s
        shifted.append({"text": w["text"], "s": s - head, "e": e - head})
    return {"head": head, "stop": stop, "info": info, "words": ws, "n_take": n_take, "n": n, "specs": specs,
            "removals": kept, "checks": checks, "shifted": shifted}


# ---------------------------------------------------------------- pictures

def talk_clip(ffmpeg, src, size_in, head, n, out, size_out, punch=1.0, anchor_y=None, overlay=None, within=None):
    """The take from `head`, n frames at the output size with a little grain. `punch` crops the native take before
    scaling (centred, top at anchor_y native px, default centred), so a punch-in is a real change of framing; past
    the take's end the last frame holds. `within`: the beat's own crop ({scale, anchor_y}): a punch-in is cut
    inside it (ff.window), a closer framing of what the beat shows, never rows its crop keeps out. `overlay`: a
    full-frame RGBA PNG (the headline) on every frame, converted with the bt709 matrix (B11)."""
    W, H = size_in
    box = ff.window(W, H, within.get("scale") or 1.0, within.get("anchor_y")) if within else None
    x, y, w, h = ff.window(W, H, punch, anchor_y, box)
    hold = max(4.0, n / FPS + 1.0)
    vf = (f"crop={w}:{h}:{x}:{y},scale={size_out[0]}:{size_out[1]}:flags=lanczos,setsar=1,fps={FPS},"
          f"noise=alls=5:allf=t+u,tpad=stop_mode=clone:stop_duration={hold:g}")
    if overlay:
        ff.run(ffmpeg, "-ss", f"{head:.3f}", "-i", src, "-i", overlay, "-filter_complex",
               f"[0:v]{vf}[v];[1:v]{ff.TO_BT709},format=yuva420p[o];[v][o]overlay=0:0", "-frames:v", n, *ff.X264, out)
    else:
        ff.run(ffmpeg, "-ss", f"{head:.3f}", "-i", src, "-vf", vf, "-frames:v", n, *ff.X264, out)
    return out


def still_clip(ffmpeg, png, n, out, size_out, push=0.06):
    """A still, cover-fitted to the frame, with a smooth sub-pixel push-in; converted with the bt709 matrix (B11)."""
    from PIL import Image
    extent = getattr(getattr(Image, "Transform", Image), "EXTENT")
    bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    OW, OH = size_out
    img = Image.open(png).convert("RGB")
    iw, ih = img.size
    cw, ch = (iw, iw * OH / OW) if iw / ih < OW / OH else (ih * OW / OH, ih)
    cw, ch = min(cw, iw), min(ch, ih)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
           "-vf", f"{ff.TO_BT709},format=yuv420p,noise=alls=4:allf=t+u,{ff.BT709_TAGS}", *ff.X264, *ff.BT709_FLAGS, out]
    try:
        p = subprocess.Popen([str(x) for x in cmd], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        raise InfraError(f"cannot run ffmpeg {ffmpeg!r}: {e}") from e
    try:
        for i in range(n):
            q = i / max(1, n - 1)
            q = q * q * (3 - 2 * q)
            z = 1 + push * q
            w, h = cw / z, ch / z
            x, y = (iw - w) / 2, (ih - h) / 2
            p.stdin.write(img.transform((OW, OH), extent, (x, y, x + w, y + h), bicubic).tobytes())
        p.stdin.close()
    except BaseException:
        p.kill()
        p.wait()
        raise
    err = p.stderr.read().decode(errors="replace")
    if p.wait():
        raise ff.FfmpegError(f"still clip encode failed: {err[-1000:]}")
    return out


# ---------------------------------------------------------------- voice

def _read_pcm(path):
    with wave.open(path) as w:
        if w.getframerate() != SR or w.getsampwidth() != 2:
            raise RuntimeError(f"{path}: not 48 kHz s16")
        ch = w.getnchannels()
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch).copy()


def _write_pcm(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.astype(np.int16).tobytes())


def cut_voice(whole, removals, n, out):
    """The beat's voice on the take grid with the removed frames cut out (silence, 5 ms fades): exactly n frames."""
    x = _read_pcm(whole).astype(np.float32)
    keep = np.ones(len(x), bool)
    fade = int(0.005 * SR)
    for f0, f1, _ in removals:
        s0, s1 = f0 * SPF, f1 * SPF
        x[max(0, s0 - fade):s0] *= np.linspace(1, 0, s0 - max(0, s0 - fade))[:, None]
        x[s1:s1 + fade] *= np.linspace(0, 1, min(fade, len(x) - s1))[:, None]
        keep[s0:s1] = False
    y = x[keep][: n * SPF]
    if len(y) != n * SPF:
        raise RuntimeError(f"{out}: {len(y)} voice samples for {n} frames ({n * SPF})")
    with ff.atomic(out) as tmp:
        _write_pcm(tmp, y)
    return out


# ---------------------------------------------------------------- publish files

def _srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def srt_text(words) -> str:
    """One cue per caption chunk (hidden ones included: the SRT is for the platform's own captions), the display
    words as written; a cue ends 0.2 s after its last word or 1 ms before the next cue, so cues never overlap."""
    chunks = {}
    for w in words:
        chunks.setdefault((w["beat"], w["line"]), []).append(w)
    order = sorted(chunks.values(), key=lambda c: c[0]["start"])
    out = []
    for i, ws in enumerate(order, 1):
        end = ws[-1]["end"] + 0.2
        if i < len(order):
            end = min(end, order[i][0]["start"] - 0.001)
        out.append(f"{i}\n{_srt_time(ws[0]['start'])} --> {_srt_time(end)}\n{' '.join(w['text'] for w in ws)}\n\n")
    return "".join(out)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_publish(path, ad_id, digest, srt_rel, geo=None, attribution=None, note=None) -> dict:
    """The publish gate a human fills (each platform's AI-content toggle, the SRT uploaded). It belongs to one file:
    an existing gate for the same hash is kept as the human left it; one for another file is replaced (B10)."""
    old = ff.read_json(path)
    if isinstance(old, dict) and old.get("ad_sha256") == digest:
        return old
    gate = {"ad_id": ad_id, "ad_sha256": digest,
            "ai_disclosure": {"tiktok_aigc_label": None, "youtube_altered_or_synthetic": None, "meta_ai_info": None},
            "geo": geo, "srt": srt_rel, "srt_uploaded": None, "attribution": attribution,
            "verified_by": None, "verified_at": None,
            "note": note or ("An AI-generated presenter: set every platform's AI-content disclosure at upload, then "
                             "fill this in. Not ready until it is.")}
    ff.write_json(path, gate)
    return gate


# ---------------------------------------------------------------- the edit

def _encode(r, video, audio_wav, ass, fontsdir, total, out):
    """The final encode (v6's: libx264 medium CRF, bt709, AAC 48 kHz, exact length), then ABR re-encodes while the
    file is over max_mb; still over after three: deleted, infrastructure."""
    e = r["encode"]
    vf = f"{captions.ass_vf(ass, fontsdir)},{ff.BT709_TAGS},format=yuv420p"
    cap, akbps = float(e["max_mb"]) * 1e6, int(e["audio_kbps"])
    rate, attempts, kbps = ["-crf", e["crf"], "-maxrate", e["maxrate"], "-bufsize", e["bufsize"]], [], None
    for i in range(4):
        with ff.atomic(out) as tmp:
            ff.run(r["ffmpeg"], "-i", video, "-i", audio_wav, "-vf", vf, "-map", "0:v", "-map", "1:a",
                   "-c:v", "libx264", "-preset", "medium", *rate, "-g", 48, "-r", FPS, *ff.BT709_FLAGS,
                   "-c:a", "aac", "-b:a", f"{akbps}k", "-ar", SR, "-movflags", "+faststart", "-t", f"{total:.3f}", tmp)
        size = os.path.getsize(out)
        attempts.append({"mode": "crf" if kbps is None else "abr", "video_kbps": kbps, "bytes": size})
        if size <= cap:
            return {"encoder": "libx264", "mode": attempts[-1]["mode"], "crf": e["crf"], "video_kbps": kbps,
                    "bytes": size, "attempts": attempts}
        kbps = int(0.85 * cap * 8 / 1000 / total - akbps) if kbps is None else int(kbps * cap / size * 0.95)
        if kbps < 150:
            break
        rate = ["-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.2)}k", "-bufsize", f"{2 * kbps}k"]
    ff.remove(out)
    raise InfraError(f"the ad is {attempts[-1]['bytes'] / 1e6:.1f} MB, over the {e['max_mb']} MB cap "
                     f"after {len(attempts)} encodes")


def _caption_fonts(font, fonts_dir):
    """A fonts directory holding the caption font alone: libass loads every face in the directory it is given and
    matches the style's family at regular weight, so the caption font's own folder (Arial Bold.ttf beside
    Arial.ttf) would draw the captions in another face than the one they were measured with."""
    shutil.rmtree(fonts_dir, ignore_errors=True)
    os.makedirs(fonts_dir)
    try:
        shutil.copyfile(font, os.path.join(fonts_dir, os.path.basename(font)))
    except OSError as e:
        raise InfraError(f"cannot copy the caption font {font}: {e}") from e
    return fonts_dir


def _seg(path, bid, k, kind, f0, f1):
    return {"path": os.path.relpath(path), "beat": bid, "k": k, "kind": kind, "frame": f0, "frames": f1 - f0,
            "start": round(f0 / FPS, 4), "end": round(f1 / FPS, 4)}


def run(req: dict, progress=lambda p, note="": None) -> dict:
    r = parse(req)
    ffmpeg, OW, OH = r["ffmpeg"], r["out"][0], r["out"][1]
    o = r["outputs"]
    seg_dir, beats_dir = o["segments_dir"], o["beats_dir"]
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(beats_dir, exist_ok=True)
    wants_headline = bool(r["headline"]) and any(b["headline"] for b in r["beats"])
    drawn = any(c["kind"] in ("screen", "price", "endcard") for b in r["beats"] for c in b["cutaways"])
    graphics = _graphics() if drawn or wants_headline else None
    if graphics is not None and (OW, OH) != (graphics.W, graphics.H):
        raise InfraError(f"the cards are drawn on a {graphics.W}x{graphics.H} frame and the ad is {OW}x{OH}: set "
                         f"ads_options out_width and out_height to {graphics.W}x{graphics.H} for an h3 ad")

    # 1. plan every beat before rendering anything (a bad plan fails in seconds, not after the encodes)
    plans, frame = [], 0
    for i, b in enumerate(r["beats"]):
        p = plan_beat(b, r["safe"])
        p["start_f"] = frame
        frame += p["n"]
        plans.append(p)
    total_f = frame
    total = total_f / FPS
    if r["music"] is not None:
        have = ff.probe(ffmpeg, r["music"])["duration_s"]
        if have is None or have + 0.02 < total:
            raise InfraError(f"music is {have} s for a {total:.2f} s ad")
    progress(0.05, "planned")

    # 2. pictures and voice per beat
    headline = None
    if wants_headline:
        headline = os.path.join(seg_dir, "headline.png")
        _draw("the headline", graphics.headline, r["headline"], headline, r["brand"], r["fonts"])
    segs, voices, words_all, hide, speech, cuts, notes, punches, checks = [], [], [], [], [], [], {}, [], {"removals": []}
    scratch, beats_out = [], []
    for i, (b, p) in enumerate(zip(r["beats"], plans)):
        bid, start_s, sf = b["id"], p["start_f"] / FPS, p["start_f"]
        tinfo = ff.probe(ffmpeg, b["take"])
        if not tinfo["width"]:
            raise ContentError(f"beat {bid}: the take has no readable video")
        size_in = (tinfo["width"], tinfo["height"])
        ov = headline if b["headline"] else None
        bc = b["crop"] or {}
        of = _mapper(p)
        full = talk_clip(ffmpeg, b["take"], size_in, p["head"], p["n_take"], os.path.join(seg_dir, f"{bid}-full.mp4"),
                         (OW, OH), punch=bc.get("scale", 1.0), anchor_y=bc.get("anchor_y"), overlay=ov)
        scratch.append(full)
        pos_t, cut_out = 0, []
        for j, sp in enumerate(p["specs"]):
            c, ta, te = sp["c"], sp["ta"], sp["te"]
            if ta > pos_t:                                    # talk pieces never contain removals
                piece = ff.frames_of(ffmpeg, full, pos_t, ta, os.path.join(seg_dir, f"{bid}-p{j}.mp4"))
                segs.append(_seg(piece, bid, i, "talk", sf + of(pos_t), sf + of(ta)))
            out, m = os.path.join(seg_dir, f"{bid}-c{j}.mp4"), sp["m"]
            ev = sp["events"]
            if c["kind"] == "punch":
                pf = talk_clip(ffmpeg, b["take"], size_in, p["head"], p["n_take"], os.path.join(seg_dir, f"{bid}-punch.mp4"),
                               (OW, OH), punch=c["scale"], anchor_y=c["anchor_y"], overlay=ov, within=b["crop"])
                scratch.append(pf)
                ff.frames_of(ffmpeg, pf, ta, te, out)
                punches.append(start_s + sp["a"] / FPS)
            elif c["kind"] == "screen":
                _draw(f"beat {bid}: the screen insert", graphics.screen_insert, r["captures"][c["capture"]], m, out,
                      ffmpeg, r["brand"], r["fonts"], fps=FPS, tap_at=ev.get("tap_at"))
            elif c["kind"] == "price":
                _draw(f"beat {bid}: the price card", graphics.price_insert, r["price_card"], m, out, ffmpeg,
                      r["brand"], r["fonts"], fps=FPS, pulse_at=ev.get("pulse_at"), underline_at=ev.get("underline_at"))
            elif c["kind"] == "endcard":
                _draw(f"beat {bid}: the end card", graphics.endcard, r["endcard"], m, out, ffmpeg, r["brand"], r["fonts"],
                      fps=FPS, pulse_at=ev.get("pulse_at"))
            elif c["kind"] == "still":
                still_clip(ffmpeg, r["stills"][c["still"]], m, out, (OW, OH), c["push"])
            else:                                             # parse() refuses these; never append nothing (B4)
                raise RequestError(f"bad request: unknown cutaway kind {c['kind']!r}")
            if not os.path.isfile(out):
                raise RuntimeError(f"beat {bid}: the {c['kind']} cutaway wrote no file")
            segs.append(_seg(out, bid, i, c["kind"], sf + sp["a"], sf + sp["e"]))
            t0, t1 = start_s + sp["a"] / FPS, start_s + sp["e"] / FPS
            if not c["captions"]:
                hide.append((t0, t1))
            cuts += [t0, t1]
            cut_out.append({"kind": c["kind"], "start": round(t0, 4), "end": round(t1, 4), "frames": m,
                            "take_frames": [ta, te], "events": ev, "captions": c["captions"]})
            pos_t = te
        if pos_t < p["n_take"]:
            if pos_t == 0:
                segs.append(_seg(full, bid, i, "talk", sf, sf + p["n"]))
                scratch.remove(full)
            else:
                piece = ff.frames_of(ffmpeg, full, pos_t, p["n_take"], os.path.join(seg_dir, f"{bid}-p{len(p['specs'])}.mp4"))
                segs.append(_seg(piece, bid, i, "talk", sf + of(pos_t), sf + p["n"]))
        # voice: the whole take at the target loudness on the take grid, then the removed frames cut out
        whole = os.path.join(seg_dir, f"{bid}-take.wav")
        scratch.append(whole)
        au = audio.take_audio(b["take"], p["head"], p["stop"] - p["head"], p["n_take"] / FPS, whole, ffmpeg, lufs=r["lufs"])
        bwav = os.path.join(beats_dir, f"{bid}.wav")
        cut_voice(whole, p["removals"], p["n"], bwav)
        with ff.atomic(os.path.join(beats_dir, f"{bid}-16k.wav")) as tmp:
            ff.run(ffmpeg, "-i", bwav, "-ac", 1, "-ar", 16000, "-c:a", "pcm_s16le", tmp)
        voices.append(bwav)
        checks["removals"] += p["checks"]
        notes[bid] = {"head": p["head"], "stop": p["stop"], "bounds": p["info"], "audio": au, "frames": p["n"],
                      "start": round(start_s, 4), "take": os.path.relpath(b["take"]),
                      "removed": [[round(p["head"] + f0 / FPS, 3), round(p["head"] + f1 / FPS, 3)]
                                  for f0, f1, _ in p["removals"]]}
        dw = [{"text": w["text"], "start": round(w["s"] + start_s, 3), "end": round(w["e"] + start_s, 3), "beat": i}
              for w in p["shifted"]]
        words_all += dw
        speech += [(w["start"], w["end"]) for w in dw]
        cuts.append(start_s)
        beats_out.append({"id": bid, "k": b["k"], "take": os.path.relpath(b["take"]), "line": b["line"],
                          "start": round(start_s, 4), "end": round(start_s + p["n"] / FPS, 4), "frames": p["n"],
                          "head": p["head"], "stop": p["stop"], "caption_pos": b["caption_pos"],
                          "headline": bool(ov), "cutaways": cut_out})
        progress(0.05 + 0.55 * (i + 1) / len(r["beats"]), f"beat {bid}: {p['n']} frames")

    # 3. join, mix, captions, encode
    vlist = ff.concat_list([os.path.abspath(s["path"]) for s in segs], os.path.join(seg_dir, "video.txt"))
    alist = ff.concat_list(voices, os.path.join(seg_dir, "voice.txt"))
    scratch += [vlist, alist]
    with ff.atomic(o["video"]) as tmp:
        ff.run(ffmpeg, "-f", "concat", "-safe", 0, "-i", vlist, "-c", "copy", tmp)
    with ff.atomic(o["voice"]) as tmp:
        ff.run(ffmpeg, "-f", "concat", "-safe", 0, "-i", alist, "-c:a", "pcm_s16le", tmp)
    progress(0.65, "joined")
    with ff.atomic(o["audio"]) as tmp:
        try:
            if r["music"] is not None:
                mixed = audio.mix(o["voice"], r["music"], speech, total, tmp, ffmpeg, last_word_end=speech[-1][1],
                                  music_under=r["music_under"], bed_lift=r["bed_lift"], lufs=r["lufs"], tp=r["tp"])
            else:
                mixed = audio.master(o["voice"], total, tmp, ffmpeg, lufs=r["lufs"], tp=r["tp"])
        except ValueError as e:
            raise InfraError(str(e)) from e
    progress(0.75, "mixed")
    cuts = sorted(set(round(c, 4) for c in cuts if 0 < c < total))
    cap = r["captions"]
    ml, mr = cap["margins"]
    family = captions.font_family(r["fonts"]["caption"])
    words_all = captions.chunk(words_all, captions.measurer(r["fonts"]["caption"], cap["size"]), max_w=OW - ml - mr,
                               cuts=cuts, hide=hide, solo=cap["solo"])
    pos = {i: b["caption_pos"] for i, b in enumerate(r["beats"]) if b["caption_pos"]}       # B2: by beat
    with ff.atomic(o["ass"]) as tmp:
        events = captions.write_ass(tmp, words_all, hide=hide, font_family=family, size=cap["size"],
                                    margin_v=cap["margin_v"], margins=(ml, mr), fps=FPS, cuts=cuts, pos=pos, play=(OW, OH))
    progress(0.8, "captions")
    fonts_dir = _caption_fonts(r["fonts"]["caption"], os.path.join(seg_dir, "fonts"))
    try:
        enc = _encode(r, o["video"], o["audio"], o["ass"], fonts_dir, total, o["mp4"])
    finally:
        shutil.rmtree(fonts_dir, ignore_errors=True)
    progress(0.9, "encoded")

    # 4. publish files: the SRT (display words), the AI-disclosure gate keyed to this file
    ff.write_text(o["srt"], srt_text(words_all))
    digest = sha256(o["mp4"])
    pub = r["publish"]
    write_publish(o["publish"], r["ad_id"], digest, os.path.relpath(o["srt"]), pub["geo"], pub["attribution"], pub["note"])

    # 5. measured on the file (reported; the core's finish measures again on its own)
    got, _ = ff.decode_count(ffmpeg, o["mp4"])
    checks["frames"] = {"got": got, "want": total_f, "ok": got == total_f}
    checks["freezes"] = ff.freezes(ffmpeg, o["mp4"])
    scenes = ff.scene_cuts(ffmpeg, o["mp4"], 8)
    checks["punches"] = [round(t, 4) for t in punches]
    checks["punch_cuts"] = [{"at": round(t, 3), "seen": any(abs(s - t) < 0.05 for s, _ in scenes)} for t in punches]
    vline = ff.probe(ffmpeg, o["mp4"])["video_line"]
    checks["color"] = [vline] if vline else []
    progress(0.97, "checked")
    timeline = {"fps": FPS, "out": [OW, OH], "beats": beats_out, "segments": segs, "words": words_all,
                "hide": [list(h) for h in hide], "cuts": cuts, "total": total, "frames": total_f, "mix": mixed,
                "notes": notes, "checks": checks, "captions": {"events": len(events), "family": family,
                                                               "font": r["fonts"]["caption"]},
                "encode": enc, "ad_sha256": digest}
    ff.write_json(o["timeline"], timeline)
    for path in scratch:
        ff.remove(path)
    if headline:
        ff.remove(headline)
    ok = checks["frames"]["ok"] and not checks["freezes"] and all(p["seen"] for p in checks["punch_cuts"])
    return {"total_s": round(total, 4), "frames": total_f, "bytes": enc["bytes"], "caption_events": len(events),
            "checks_ok": ok, "checks": {"frames": checks["frames"], "freezes": checks["freezes"],
                                        "punch_cuts": checks["punch_cuts"], "color": checks["color"]},
            "mix": {k: mixed.get(k) for k in ("I", "TP", "LRA", "under_lu", "limited_db")},
            "encode": enc, "outputs": {k: os.path.relpath(v) for k, v in o.items()}}


def _mapper(p):
    """take frame -> beat output frame for plan p (the same mapping plan_beat uses)."""
    kept = p["removals"]
    return lambda f: f - sum(min(f, f1) - f0 for f0, f1, _ in kept if f > f0)
