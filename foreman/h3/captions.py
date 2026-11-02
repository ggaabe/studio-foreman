"""Burned word-highlight captions for an h3 ad: the script's display words are the text, the take's
word times give the times.

align() puts display words on timed ASR words; chunk() groups words into 1-4 word chunks; write_ass()
writes one event per word, the word in the highlight colour, each chunk kept on its own shot at
picture cuts, hidden under ranges that carry their own text (a price card, the end card), and a beat's
events moved where the plan says. Nothing here knows a brand: the words that may stand alone in a
chunk (an address, a price, a product name) come in as `solo`.
Standard library and Pillow only (Pillow only in measurer); ffmpeg is not called here (ass_vf() gives
the filter string for the encode).
"""
from __future__ import annotations

import difflib
import math
import re
import struct

ACTIVE, WHITE = r"{\c&H0000D7FF&}", r"{\c&H00FFFFFF&}"
CLOSE = "\"')]”’"  # closing marks that may follow a word's punctuation
FUNC = set("the to a an at and for of or your but in on with from by my our no if than that before after into".split())
WEAK_END, WEAK_START = set("you we they i he she is are will can".split()), {"it", "them"}  # mild: subject|verb, verb|it
HOLD, LEAD, GAP = 0.25, 0.5, 0.7  # last word holds +0.25 s; a first event before 0.5 s starts at 0.00; pause that splits
CUT_TOL, CUT_PULL = 0.05, 0.15  # chunk ending <= cut + 0.05 s stops on the cut; one starting <= 0.15 s after starts on it
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
INF = float("inf")
SOLO_RX = re.compile(r"\$\d|\w\.(com|org|net|io)\b", re.I)   # a price or an address may always stand alone


def _num(n):
    """Spoken words for an integer as a price or number is said: 950 -> nine fifty, 2026 -> twenty twenty six."""
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    if n % 1000 == 0 and n < 10 ** 6:
        return _num(n // 1000) + ["thousand"]
    if n % 100 == 0 and n < 10000:
        return _num(n // 100) + ["hundred"]
    if n < 10000:
        return _num(n // 100) + (["oh"] if n % 100 < 10 else []) + _num(n % 100)
    return [ONES[int(c)] for c in str(n)]


def tokens(text):
    """Comparable spoken tokens: 'P.D.F.' -> pdf, '$9.50'/'9.50'/'9:50'/'950' -> nine fifty, '.com' -> dot com,
    hyphens split."""
    t = text.lower()
    t = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m[0].replace(".", ""), t)
    t = re.sub(r"\$?(\d+)[.:](\d\d)\b", lambda m: " ".join(_num(int(m[1])) + (
        [] if m[2] == "00" else (["oh"] if m[2][0] == "0" else []) + _num(int(m[2])))), t)
    t = re.sub(r"\d+", lambda m: " ".join(_num(int(m[0]))), t)
    t = re.sub(r"\.(com|org|net|io)\b", r" dot \1", t)
    return re.findall(r"[a-z]+(?:'[a-z]+)?", t)


def _parts(w):
    """(leading punctuation, core, trailing punctuation) of a script word."""
    m = re.match(r"^([\"'(\[“‘]*)(.*?)([.,!?;:\"')\]”’…]*)$", w)
    return m[1], m[2], m[3]


def _spread(lo, hi, weights):
    """Split [lo, hi] into contiguous spans in proportion to weights."""
    tot, out, t = sum(weights) or 1, [], lo
    for w in weights:
        out.append((t, t + (hi - lo) * w / tot)); t = out[-1][1]
    return out


def _asr_tokens(asr_words, joins=()):
    """[(token, start, end)] for ASR words, a multi-token word ('Example.com', '$9.50') split by length. Two or
    three consecutive tokens whose concatenation is a token of the display words (a name the ASR heard as two
    words) become that one token."""
    out = []
    for w in asr_words:
        tk = tokens(w["word"])
        out += [(t, s, e) for t, (s, e) in zip(tk, _spread(w["start"], w["end"], [len(t) for t in tk]))]
    merged, i = [], 0
    while i < len(out):
        for k in (3, 2):
            part = out[i:i + k]
            cat = "".join(t for t, _, _ in part)
            if len(part) == k and cat in joins and not all(t in joins for t, _, _ in part):
                merged.append((cat, part[0][1], part[-1][2])); i += k
                break
        else:
            merged.append(out[i]); i += 1
    return merged


def _fill(spans, weights):
    """Times for display words the ASR missed, interpolated between their neighbours."""
    n, i = len(spans), 0
    if not any(spans):
        return [(k / 3, (k + 1) / 3) for k in range(n)]
    while i < n:
        if spans[i]:
            i += 1; continue
        j = i
        while j < n and not spans[j]:
            j += 1
        p, q, m = spans[i - 1] if i else None, spans[j] if j < n else None, j - i
        if p is None:
            lo, hi = max(0.0, q[0] - 0.3 * m), q[0]
        elif q is None:
            lo, hi = p[1], p[1] + 0.3 * m
        else:
            lo, hi = p[1], q[0]
            short = 0.12 * m - (hi - lo)
            if short > 0:  # no room: borrow from both neighbours, at most half of each
                lo, hi = max((p[0] + p[1]) / 2, lo - short / 2), min((q[0] + q[1]) / 2, hi + short / 2)
                spans[i - 1], spans[j] = (p[0], lo), (hi, q[1])
        spans[i:j] = _spread(lo, hi, weights[i:j])
        i = j
    return spans


def align(display, asr_words, offset=0.0, head=0.0, beat=0):
    """Display words [{"text", "start", "end", "beat"}] for a line (a string split on spaces, or its display
    words), timed by ASR words ({"word", "start", "end"}) matched on spoken tokens (difflib); a misheard run is
    spread at most 0.6 s per token, missed words are interpolated. t_out = t_asr - head + offset."""
    words = display.split() if isinstance(display, str) else list(display)
    disp = [(w, tokens(_parts(w)[1])) for w in words]
    a = [(k, t) for k, (_, tk) in enumerate(disp) for t in tk]
    b = _asr_tokens(asr_words, {t for _, t in a})
    times = [None] * len(a)
    sm = difflib.SequenceMatcher(a=[t for _, t in a], b=[t for t, _, _ in b], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            times[i1:i2] = [x[1:] for x in b[j1:j2]]
        elif op == "replace":  # misheard run: spread its time, at most 0.6 s per script token
            s = b[j1][1]
            times[i1:i2] = _spread(s, min(b[j2 - 1][2], s + 0.6 * (i2 - i1)), [len(t) for _, t in a[i1:i2]])
    spans = [None] * len(disp)
    for (k, _), tm in zip(a, times):
        if tm:
            spans[k] = (min(spans[k][0], tm[0]), max(spans[k][1], tm[1])) if spans[k] else tm
    spans = _fill(spans, [max(1, len(t)) for t, _ in disp])
    out, last = [], -INF
    for (text, _), (s, e) in zip(disp, spans):
        s = max(s, last); e = max(e, s); last = s
        out.append({"text": text, "start": round(s - head + offset, 3), "end": round(e - head + offset, 3), "beat": beat})
    return out


def _core(text):
    return _parts(text)[1].lower()


def _solo(text, solo=frozenset()):
    """True for the words that may stand alone in a chunk: the `solo` words (a name, the address, a price), or
    anything that reads as a price or an address."""
    return _core(text) in solo or bool(SOLO_RX.search(text))


def _cost(ws, i, j, measure, max_w, cuts, forced_end, solo=frozenset()):
    """Badness of ws[i:j] as one chunk; inf when it is too wide."""
    part, n = ws[i:j], j - i
    if n > 1 and measure(" ".join(w["text"] for w in part)) > max_w:
        return INF
    last = part[-1]["text"]
    end = forced_end or not last[-1:].isalnum()  # a cut or punctuation follows
    c = {1: 3, 2: 2 if end else 3, 3: 0, 4: 1}.get(n, 1)
    for k, (w, v) in enumerate(zip(part, part[1:])):
        p = w["text"].rstrip(CLOSE)[-1:]
        c += 500 if p in ".?!…" else 300 if p in ",;:" and k >= 1 else 0
        c += 50 * (v["start"] - w["end"] > GAP) + 50 * sum(w["start"] < t <= v["start"] for t in cuts)
    word, first = (re.sub(r"\W", "", x["text"].lower()) for x in (part[-1], part[0]))
    c += 0 if end else 1000 if word in FUNC else 30 if word in WEAK_END else 0
    c += 20 if i and ws[i - 1]["text"][-1:].isalnum() and first in WEAK_START else 0
    c += max(0.0, 1 - (ws[j]["start"] - part[-1]["end"]) / 0.15) if j < len(ws) else 0  # prefer breaks at her pauses
    return c + (1000 if n == 1 and not _solo(last, solo) else 0)


def chunk(words, measure, max_w=800, max_words=4, cuts=(), hide=(), solo=()):
    """Copies of words with "line" (chunk id) set. Breaks after . ? ! (and , once the chunk has 2+ words), else at 3-4
    words or max_w px of measure(text); never ends on the/to/a/at/and/for/of/or/your/but...; no lone word but a
    `solo` word, an address or a price. Chunks never span a beat change or a hide edge (a word is hidden if it starts
    in a hide range), and only there may a chunk end on a function word; crossing a picture cut time in cuts is
    avoided."""
    solo = frozenset(_core(s) for s in solo)
    words = [dict(w) for w in words]
    hid = [any(s <= w["start"] < e for s, e in hide) for w in words]
    line, r0 = 0, 0
    for r1 in range(1, len(words) + 1):
        if r1 < len(words) and words[r1].get("beat", 0) == words[r1 - 1].get("beat", 0) and hid[r1] == hid[r1 - 1]:
            continue
        ws, n = words[r0:r1], r1 - r0
        best, back = [0.0] + [INF] * n, [0] * (n + 1)
        for j in range(1, n + 1):
            for i in range(max(0, j - max_words), j):
                c = best[i] + _cost(ws, i, j, measure, max_w, cuts, j == n, solo)
                if c < best[j]:
                    best[j], back[j] = c, i
        bounds, j = [], n
        while j > 0:
            bounds.append((back[j], j)); j = back[j]
        for i, j in reversed(bounds):
            for w in ws[i:j]:
                w["line"] = line
            line += 1
        r0 = r1
    return words


def _clip(s, e, hide):
    """Pieces of [s, e) outside every hide range."""
    out = [(s, e)]
    for hs, he in hide:
        out = [p for a, b in out for p in ((a, min(b, hs)), (max(a, he), b)) if p[1] > p[0]]
    return out


def _ts(cs):
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _esc(t):
    return t.replace("\\", "/").replace("{", "(").replace("}", ")")


def _pos_tag(xy):
    return "{\\an8\\pos(%s,%s)}" % (f"{xy[0]:g}", f"{xy[1]:g}")


def write_ass(path, words, hide=(), font_family="Arial Black", size=92, margin_v=710, margins=(140, 140), fps=24,
              cuts=(), pos=None, play=(1080, 1920)):
    """One Dialogue per word showing its chunk ("line") with that word in #FFD700, from its start to the next word's
    start; a chunk's last word holds +0.25 s or to the next chunk. A word starting inside a hide range is never shown
    (not even as part of a visible chunk), and no frame that touches a hide range gets a caption. Picture cuts (times
    in cuts, snapped to frames) keep captions on their own shot: a chunk stops on the frame of the first cut after its
    first frame that all its words end by (+0.05 s), so it never runs onto the next shot and a chunk with one word
    across a cut stops at the cut after that; a chunk whose first word starts within 0.15 s after a cut starts on it;
    a chunk a cut falls cleanly between two words of (the later word shown from the cut's frame on) is shown as two.
    Only a single word spanning a cut is drawn across it. Times are snapped to whole frames at fps and written half a
    frame early, so each frame shows exactly the event due at its start time. pos {beat: (x, y)}: every event of that
    beat's chunks is drawn top-centred at (x, y) (chosen by the words' beat, so a chunk pulled before its beat's first
    frame moves too). `path` is written as given (the caller makes it atomic). Returns the events [(start, end, text)]
    in seconds."""
    pos = pos or {}
    hid = lambda w: any(s <= w["start"] < e for s, e in hide)
    fr = lambda t: math.ceil(t * fps - 1e-6)  # first frame at or after t
    cf = sorted({round(t * fps) for t in cuts})  # each cut as the first frame of the new shot
    stop_at = lambda part: next((f for f in cf if f > fr(part[0]["start"]) and  # first cut after the part's first
                                 max(x["end"] for x in part) <= f / fps + CUT_TOL), INF)  # frame its words end by
    chunks = []
    for w in words:  # a chunk cut by a hide edge shows only its visible words, one cut cleanly by a picture cut splits
        if (chunks and chunks[-1][0]["line"] == w["line"] and hid(chunks[-1][0]) == hid(w)
                and fr(w["start"]) < stop_at(chunks[-1])):
            chunks[-1].append(w)
        else:
            chunks.append([w])
    block = [(math.floor(s * fps + 1e-6), fr(e)) for s, e in hide]  # frames that touch a hide range
    ev = []
    for c, part in enumerate(chunks):
        if hid(part[0]):
            continue
        nxt = chunks[c + 1][0]["start"] if c + 1 < len(chunks) else INF
        stop = stop_at(part)
        s0 = part[0]["start"]
        pull = max((f for f in cf if f / fps - 1e-6 <= s0 <= f / fps + CUT_PULL), default=INF)
        xy = pos.get(part[0].get("beat", 0))
        pre = _pos_tag(xy) if xy else ""
        for k, w in enumerate(part):
            end = part[k + 1]["start"] if k + 1 < len(part) else w["end"] + HOLD
            text = pre + " ".join(ACTIVE + _esc(x["text"]) + WHITE if m == k else _esc(x["text"])
                                  for m, x in enumerate(part))
            ev.append([min(fr(w["start"]), pull if k == 0 else INF), min(fr(min(end, nxt)), stop), text])
    if ev and ev[0][0] < LEAD * fps:
        ev[0][0] = 0
    out = []
    for s, e, text in ev:  # in frames
        for a, b in _clip(s, e, block):
            a = max(a, out[-1][1] if out else 0)
            if b > a:
                out.append((a, b, text))
    cs = lambda f: max(0, round((f - 0.5) * 100 / fps))  # centiseconds between frame f-1 and frame f
    out = [(cs(a), cs(b), t) for a, b, t in out]
    ml, mr = margins
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: {play[0]}\nPlayResY: {play[1]}\nWrapStyle: 2\n"
                "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, "
                "SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
                "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                f"Style: Caption,{font_family},{size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,"
                f"7,3,2,{ml},{mr},{margin_v},1\n\n[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        f.writelines(f"Dialogue: 0,{_ts(a)},{_ts(b)},Caption,,0,0,0,,{t}\n" for a, b, t in out)
    return [(a / 100, b / 100, t) for a, b, t in out]


def _sfnt(d):
    """(byte offset of the first face, {tag: offset}) of a TrueType/OpenType file or collection."""
    base = struct.unpack(">I", d[12:16])[0] if d[:4] == b"ttcf" else 0
    n = struct.unpack(">H", d[base + 4:base + 6])[0]
    return base, {d[base + 12 + 16 * i:base + 16 + 16 * i]: struct.unpack(">I", d[base + 20 + 16 * i:base + 24 + 16 * i])[0]
                  for i in range(n)}


def _win_em(font_file):
    """(unitsPerEm, winAscent + winDescent) of a TrueType/OpenType font."""
    with open(font_file, "rb") as f:
        d = f.read()
    _, tab = _sfnt(d)
    upm = struct.unpack(">H", d[tab[b"head"] + 18:tab[b"head"] + 20])[0]
    wa, wd = struct.unpack(">HH", d[tab[b"OS/2"] + 74:tab[b"OS/2"] + 78])
    return upm, wa + wd


def font_family(font_file):
    """The family name libass matches a Style's Fontname against (name id 1, else 16, else the full name)."""
    with open(font_file, "rb") as f:
        d = f.read()
    _, tab = _sfnt(d)
    off = tab[b"name"]
    count, strings = struct.unpack(">HH", d[off + 2:off + 6])
    found = {}
    for i in range(count):
        pid, eid, lid, nid, ln, o = struct.unpack(">6H", d[off + 6 + 12 * i:off + 18 + 12 * i])
        if nid not in (1, 4, 16):
            continue
        raw = d[off + strings + o:off + strings + o + ln]
        if pid in (0, 3):
            text, rank = raw.decode("utf-16-be", "replace"), 0 if lid == 0x409 else 1
        elif pid == 1:
            text, rank = raw.decode("mac_roman", "replace"), 2 if lid == 0 else 3
        else:
            continue
        found.setdefault(nid, []).append((rank, text.strip()))
    for nid in (1, 16, 4):
        if found.get(nid):
            return min(found[nid])[1]
    raise ValueError(f"{font_file}: no family name")


def measurer(font_file, size=92, border=7, shadow=3):
    """measure(text) -> px width of the ink libass draws for a centred line at ASS size `size` (it scales
    winAscent+winDescent to `size`, so Arial Black 92 has a 65.2 px em), border, shadow and 1 px antialias included,
    taken as twice the wider half: the shadow falls right only, so a chunk with measure <= 800 stays inside margins
    that leave 800 px."""
    from PIL import ImageFont
    upm, win = _win_em(font_file)
    f, k = ImageFont.truetype(font_file, upm), size / win

    def measure(text):
        adv, (x0, _, x1, _) = f.getlength(text), f.getbbox(text)
        return 2 * max((adv / 2 - x0) * k + border, (x1 - adv / 2) * k + border + shadow) + 2
    return measure


def ass_vf(ass_path, fontsdir):
    """The -vf filter that burns ass_path with fonts from fontsdir, escaped for an ffmpeg filtergraph."""
    esc = lambda v: re.sub(r"([\\',;\[\]])", r"\\\1", re.sub(r"([\\':])", r"\\\1", v))
    return f"ass=filename={esc(ass_path)}:fontsdir={esc(fontsdir)}"
