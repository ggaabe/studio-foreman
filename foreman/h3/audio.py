"""Dialogue cuts and the music mix for an h3 ad (the v6 recipe's audio, tuned on real takes).

WAVs are 48 kHz stereo s16 unless noted. ffmpeg is always passed in as a path (never looked up on
PATH). Only the standard library and numpy are used.

Public:
  acoustic_bounds(wav16k, first_start, last_end, next_start=None) -> (onset, stop, info)
  silent_runs(wav, thresh_db=-40.0, min_s=0.25, frame_ms=10) -> [(start, end), ...]
  span_level(wav, a, b, frame_ms=10) -> {'max_db', 'rms_db', 'frames_over_-35'}
  db10(wav16k) -> 10 ms RMS dBFS per frame;  voiced_runs(wav16k, thresh_db=-35.0, min_s=0.03) -> [(s, e)]
  take_audio(src, head, keep, total, out, ffmpeg, eq=None, lufs=None, max_gain=6.0) -> dict
  speech_regions(speech, last_word_end=None) -> [[a, b], ...]
  duck_curve(n, regions, duck_db=-10.0, fade_out=0.7) -> per-sample gain
  mix(voice, music, speech, total, out, ffmpeg, ..., music_under=14.0, bed_lift=5.0, win=1.0) -> dict
  master(voice, total, out, ffmpeg, lufs=-14.0, tp=-1.5) -> dict   (no music: the voice alone)
  loudness(x, ffmpeg) -> {'I', 'thresh', 'LRA', 'TP', 'M'}
"""
from __future__ import annotations

import json
import re
import struct
import subprocess
import wave

import numpy as np

from .ff import FfmpegError, InfraError

SR = 48000
FRAME = 0.01                 # analysis frame (s)
SIL_DB, LOUD_DB = -40.0, -30.0


# ---------------------------------------------------------------- helpers

def _run(cmd, data=None):
    """Run a command; return (stdout bytes, stderr text); raise on failure."""
    try:
        p = subprocess.run([str(c) for c in cmd], input=data, capture_output=True)
    except OSError as e:
        raise InfraError(f"cannot run {cmd[0]}: {e}") from e
    if p.returncode:
        raise FfmpegError(f"{cmd[0]} failed:\n{p.stderr.decode(errors='replace')[-1500:]}")
    return p.stdout, p.stderr.decode(errors="replace")


def _read_wav(path):
    """PCM WAV -> (float64 array (n, ch), rate)."""
    with wave.open(str(path)) as w:
        ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, "<i2") / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4") / 2147483648.0
    else:
        raise ValueError(f"{path}: {8 * sw}-bit WAV not supported")
    return x.reshape(-1, ch), sr


def _write_wav(path, x):
    """float (n, 2) -> 48 kHz s16 WAV."""
    y = np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(y.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


def _wav_bytes(x):
    """float (n, ch) -> in-memory 32-bit float WAV, for piping into ffmpeg."""
    x = np.ascontiguousarray(x, "<f4")
    ch, size = x.shape[1], x.nbytes
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + size, b"WAVE", b"fmt ", 16, 3,
                      ch, SR, SR * ch * 4, ch * 4, 32, b"data", size)
    return hdr + x.tobytes()


def _fit(x, n):
    """Pad with silence or trim to exactly n samples."""
    x = np.asarray(x, np.float64)
    if len(x) >= n:
        return x[:n].copy()
    return np.vstack([x, np.zeros((n - len(x), x.shape[1]))])


def _is_mono(src, ffmpeg):
    try:
        p = subprocess.run([str(ffmpeg), "-hide_banner", "-nostdin", "-i", str(src)], capture_output=True)
    except OSError as e:
        raise InfraError(f"cannot run {ffmpeg}: {e}") from e
    m = re.search(r"Audio:.*?Hz, ([^,\n]+)", p.stderr.decode(errors="replace"))
    if not m:
        raise RuntimeError(f"{src}: no audio stream")
    return m.group(1).strip() in ("mono", "1 channels")


def _decode(src, ffmpeg, af=""):
    """Any audio file -> float64 (n, 2) at 48 kHz. Mono is duplicated at unity
    (ffmpeg's own mono->stereo upmix is -3 dB)."""
    up = "pan=stereo|c0=c0|c1=c0" if _is_mono(src, ffmpeg) else "aformat=channel_layouts=stereo"
    chain = f"aresample={SR}:first_pts=0,{up}" + (f",{af}" if af else "")
    out, _ = _run([ffmpeg, "-nostdin", "-v", "error", "-i", src, "-vn", "-af", chain,
                   "-f", "f32le", "-c:a", "pcm_f32le", "-"])
    return np.frombuffer(out, "<f4").reshape(-1, 2).astype(np.float64)


def _frames_db(x, sr, frame=FRAME):
    """Mono signal -> RMS dBFS per `frame` s frame from sample 0 (floor -120)."""
    f = int(round(sr * frame))
    k = len(x) // f
    rms = np.sqrt((x[:k * f].reshape(k, f) ** 2).mean(1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def _clip_runs(x, level):
    """Runs of >= 3 flat samples at >= 0.999 * level, summed over channels."""
    n = 0
    for c in np.asarray(x).T:
        hot = np.abs(c) >= 0.999 * level
        pair = hot[1:] & hot[:-1] & (np.abs(np.diff(c)) < 1e-6)
        d = np.diff(np.concatenate(([0], pair.astype(np.int8), [0])))
        n += int(((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) >= 2).sum())
    return n


def _last_json(text):
    return json.loads(text[text.rfind("{"):text.rfind("}") + 1])


def _limit(x, ceiling_db, win=0.005):
    """Look-ahead peak limiter: 4x-oversampled peaks, gain = min over a 5 ms
    window, smoothed over 2.5 ms, so every true peak lands under ceiling_db.
    Returns (limited x, max reduction dB, per-sample gain)."""
    n = len(x)
    X = np.fft.rfft(x, axis=0)
    up = np.fft.irfft(X, 4 * n, axis=0) * 4
    pk = np.abs(up).max(1).reshape(n, 4).max(1)
    g = np.minimum(1.0, 10 ** (ceiling_db / 20) / np.maximum(pk, 1e-9))
    w = round(win * SR) | 1
    g = np.lib.stride_tricks.sliding_window_view(np.pad(g, w // 2, constant_values=1.0), w).min(1)
    h = w // 2 | 1
    g = np.convolve(np.pad(g, h // 2, mode="edge"), np.ones(h) / h, mode="valid")
    return x * g[:, None], round(float(-20 * np.log10(g.min())), 2), g


def loudness(x, ffmpeg):
    """BS.1770 via ffmpeg ebur128 for a path or a float (n, 2) array at 48 kHz.
    Returns {'I': LUFS, 'thresh': gating threshold, 'LRA': LU, 'TP': dBTP,
    'M': [(t, momentary LUFS), ...]}, each M the 400 ms window ending at t (floor -120:
    ebur128 logs silent blocks as nan or -inf, and they must not drop out)."""
    if isinstance(x, np.ndarray):
        inp, data = ["-f", "wav", "-i", "pipe:0"], _wav_bytes(x)
    else:
        inp, data = ["-i", x], None
    _, err = _run([ffmpeg, "-nostdin", "-hide_banner", "-v", "verbose", *inp,
                   "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-"], data)
    M = [(float(t), max(float(m), -120.0) if m != "nan" else -120.0) for t, m in   # silence logs nan/-inf
         re.findall(r"t:\s*([\d.]+)\s+TARGET:\S+\s+LUFS\s+M:\s*(-?[\d.]+|-?inf|nan)", err)]
    s = err[err.rfind("Summary:"):]
    get = lambda k: float(re.search(k + r":\s*(-?inf|-?[\d.]+)", s).group(1))
    return {"I": get("I"), "thresh": get("Threshold"), "LRA": get("LRA"), "TP": get("Peak"), "M": M}


# ---------------------------------------------------------------- 1. bounds

def _voiced_runs(active, gap=6, min_frames=3):
    """Frame runs above threshold, bridging gaps under 60 ms; drop runs < 30 ms."""
    runs = []
    for k in np.flatnonzero(active):
        if runs and k - runs[-1][1] < gap:
            runs[-1][1], runs[-1][2] = k + 1, runs[-1][2] + 1
        else:
            runs.append([k, k + 1, 1])
    return [(s, e) for s, e, c in runs if c >= min_frames]


def acoustic_bounds(wav16k, first_start, last_end, next_start=None, asr_slack=0.25):
    """Energy-based cut points for one line of a take (16 kHz mono ASR WAV).

    Returns (onset, stop, info), seconds rounded to 3 decimals. info has:
      acoustic_end  first 10 ms frame at/after last_end-0.05 that begins 60 ms < -40 dBFS
      runon         time of the first later frame > -30 dBFS (run-on fragment) or None
      onset_rule    'energy' (voiced run found in the 0.30 s look-back), 'window'
                    (energy runs through the whole look-back) or 'asr' (none: first_start)
      limit         which rule set stop: 'tail', 'runon', 'next', 'next-energy' or 'eof'
      asr_early     True when ASR put next_start inside the line's own sounding tail
                    (acoustic_end within asr_slack after next_start); the next word is
                    then bounded by its energy onset - 0.06 instead of next_start - 0.08.
    Energy cannot move the onset later than first_start: 16 kHz audio loses most /s/
    energy, so ASR word starts are kept as the latest possible onset.
    """
    x, sr = _read_wav(wav16k)
    db = _frames_db(x.mean(1), sr)
    nf, dur = len(db), len(x) / sr
    pad = np.concatenate([db, np.full(6, -120.0)])
    up = lambda t: max(0, int(np.ceil(t / FRAME - 1e-6)))   # first frame starting at/after t

    # onset
    lo, hi = up(first_start - 0.30), int(first_start / FRAME + 1e-6)
    runs = _voiced_runs(db > SIL_DB)
    starts = [s for s, e in runs if lo <= s <= hi]
    if starts:
        t_on, rule = min(starts) * FRAME, "energy"
    elif any(s < lo and e > hi for s, e in runs):
        t_on, rule = lo * FRAME, "window"
    else:
        t_on, rule = first_start, "asr"
    onset = round(max(0.0, float(t_on) - 0.10), 3)

    # acoustic end and run-on
    k = up(last_end - 0.05)
    while k < nf and not (pad[k:k + 6] < SIL_DB).all():
        k += 1
    ae = min(k * FRAME, dur)
    loud = np.flatnonzero(db[k:] > LOUD_DB)
    runon = float(k + loud[0]) * FRAME if len(loud) else None
    lims = {"tail": ae + 0.20, "eof": dur}
    if runon is not None:
        lims["runon"] = runon - 0.06
    asr_early = False
    if next_start is not None:
        if next_start - 0.08 < ae <= next_start + asr_slack:
            asr_early = True
            nxt = np.flatnonzero(db[k:] > SIL_DB)
            lims["next-energy"] = (float(k + nxt[0]) * FRAME if len(nxt) else dur) - 0.06
        else:
            lims["next"] = next_start - 0.08
    limit = min(lims, key=lims.get)
    stop = round(min(max(lims[limit], onset + 0.1), dur), 3)
    info = {"acoustic_end": round(ae, 3), "runon": None if runon is None else round(runon, 3),
            "onset_rule": rule, "limit": limit, "asr_early": asr_early}
    return onset, stop, info


def silent_runs(wav, thresh_db=-40.0, min_s=0.25, frame_ms=10):
    """Silences of a WAV (any rate, mono or stereo): runs of consecutive frame_ms frames
    (on the file's own grid from sample 0, RMS over all channels) that are all below
    thresh_db dBFS and last at least min_s. Returns [(start, end), ...] in seconds,
    3 decimals. Only these runs may be closed; ASR word ends are not boundaries."""
    x, sr = _read_wav(wav)
    f = int(round(sr * frame_ms / 1000)) / sr
    q = _frames_db(np.sqrt((x ** 2).mean(1)), sr, frame_ms / 1000) < thresh_db
    d = np.diff(np.concatenate(([0], q.astype(np.int8), [0])))
    return [(round(float(s * f), 3), round(float(e * f), 3)) for s, e in
            zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if (e - s) * f >= min_s - 1e-9]


def span_level(wav, a, b, frame_ms=10):
    """Level of [a, b) s of a WAV, to refuse removing anything that is not silence.
    max_db: the loudest frame_ms window, slid one sample at a time (a burst cannot hide
    across two grid frames); rms_db: RMS of the whole span; frames_over_-35: how many
    frame_ms frames from a (a last partial frame counts) are over -35 dBFS. RMS is over
    all channels; floor -120 dB."""
    x, sr = _read_wav(wav)
    i, j = max(0, round(a * sr)), min(len(x), round(b * sr))
    if j <= i:
        raise ValueError(f"{wav}: empty span [{a}, {b})")
    p = (x[i:j] ** 2).mean(1)
    f = min(len(p), max(1, round(sr * frame_ms / 1000)))
    c = np.concatenate(([0.0], np.cumsum(p)))
    db = lambda e: 10 * np.log10(np.maximum(e, 1e-12))
    fr = db(np.array([p[k:k + f].mean() for k in range(0, len(p), f)]))
    return {"max_db": round(float(db((c[f:] - c[:-f]).max() / f)), 2),
            "rms_db": round(float(db(p.mean())), 2),
            "frames_over_-35": int((fr > -35).sum())}


def db10(wav16k):
    """10 ms RMS dBFS frames of a mono 16 kHz WAV, from sample 0 (the scale the take gates read)."""
    with wave.open(str(wav16k)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    hop = sr // 100
    return 20 * np.log10(np.sqrt(np.mean(x[: len(x) // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-9)


def voiced_runs(wav16k, thresh_db=-35.0, min_s=0.03, db=None):
    """(start, end) runs of 10 ms frames above thresh_db (the anchor gate's onsets)."""
    db = db10(wav16k) if db is None else db
    runs, on = [], None
    for i, v in enumerate(list(db) + [-120.0]):
        if v > thresh_db and on is None:
            on = i
        elif v <= thresh_db and on is not None:
            if (i - on) / 100 >= min_s:
                runs.append((on / 100, i / 100))
            on = None
    return runs


# ---------------------------------------------------------------- 2. take audio

def take_audio(src, head, keep, total, out, ffmpeg, eq=None, lufs=None, max_gain=6.0):
    """Cut [head, head+keep] of a take's audio (any file ffmpeg reads) to a 48 kHz
    stereo s16 WAV of exactly round(total*48000) samples: clamp to full scale (so
    clipped tops are flat), volume=-3dB, adeclip, optional eq filter string (e.g.
    'highshelf=f=5000:g=3'), 20 ms fade-in, 40 ms fade-out ending at keep, silence
    after. If the source ends before head+keep, the fade-out ends at the source end
    instead (short_s says by how much).
    lufs: measure the integrated loudness (ebur128) of the kept, faded span and apply
    gain = lufs - I, capped at +/-max_gain dB, so every take sits at the same level; if
    that would put the true peak over -1 dBTP the span is peak-limited to -1.2 instead
    (limited_db says by how much). Spans with no gated loudness (< 0.4 s) are left.
    Returns {'samples', 'clip_runs_in', 'clip_runs_out', 'peak_db', 'short_s',
    'lufs_before', 'gain_db', 'lufs_after', 'tp_db', 'limited_db'} (lufs_* and tp_db
    are None without lufs)."""
    if not 0.06 < keep <= total + 1e-9:
        raise ValueError(f"need 0.06 < keep <= total (keep={keep}, total={total})")
    if head < 0:
        raise ValueError(f"head must be >= 0 (head={head})")
    clamp = "aformat=sample_fmts=s16"
    raw = _decode(src, ffmpeg, clamp)
    x = _decode(src, ffmpeg, clamp + ",volume=-3dB:precision=float,adeclip" + (f",{eq}" if eq else ""))
    a, k, n = round(head * SR), round(keep * SR), round(total * SR)
    fi, fo = round(0.020 * SR), round(0.040 * SR)
    k_src = min(k, len(x) - a)                 # samples the source really has
    if k_src < fi + fo:
        raise ValueError(f"{src}: only {max(k_src, 0) / SR:.3f} s of audio after head={head}")
    seg = x[a:a + k_src].copy()
    seg[:fi] *= np.linspace(0, 1, fi, endpoint=False)[:, None]
    seg[k_src - fo:] *= np.linspace(1, 0, fo)[:, None]
    lv = {"lufs_before": None, "gain_db": 0.0, "lufs_after": None, "tp_db": None, "limited_db": 0.0}
    if lufs is not None:
        m = loudness(seg, ffmpeg)
        lv["lufs_before"] = m["I"]
        if m["I"] > -70:
            gain = float(np.clip(lufs - m["I"], -max_gain, max_gain))
            seg *= 10 ** (gain / 20)
            if m["TP"] + gain > -1.0:
                seg, lv["limited_db"], _ = _limit(seg, -1.2)
            m = loudness(seg, ffmpeg)
            lv["gain_db"] = round(gain, 2)
        lv["lufs_after"], lv["tp_db"] = m["I"], m["TP"]
    _write_wav(out, _fit(seg, n))
    with wave.open(str(out)) as w:
        got = w.getnframes()
    if got != n:
        raise RuntimeError(f"{out}: {got} samples, expected {n}")
    return {"samples": got,
            "clip_runs_in": _clip_runs(raw[a:a + k], 1.0),
            "clip_runs_out": _clip_runs(x[a:a + k], 10 ** (-3 / 20)),
            "peak_db": round(float(20 * np.log10(max(np.abs(seg).max(), 1e-9))), 2),
            "short_s": round((k - k_src) / SR, 4), **lv}


# ---------------------------------------------------------------- 3. mix

def speech_regions(speech, pre=0.12, post=0.25, gap=0.8, last_word_end=None):
    """Word (start, end) pairs -> [start-pre, end+post] regions merged across gaps
    under `gap`. With last_word_end, words starting after it are ignored and no
    region runs past last_word_end + post."""
    out = []
    for s, e in sorted(speech):
        if last_word_end is not None and s >= last_word_end:
            continue
        a, b = s - pre, e + post
        if last_word_end is not None:
            b = min(b, last_word_end + post)
        if out and a - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def duck_curve(n, regions, duck_db=-10.0, down=0.12, up=0.40, fade_in=0.01, fade_out=0.7):
    """Per-sample music gain: 0 dB outside regions, duck_db inside, ramps (linear
    in dB) `down` s starting at each region start and `up` s starting at each end,
    a linear fade-in and a qsin fade-out (cos 0..pi/2) over the last fade_out s.
    Regions whose ramps overlap take the deeper gain (duck_db <= 0)."""
    t, d = np.arange(n) / SR, np.zeros(n)
    for a, b in regions:
        b = max(b, a + down)
        d = np.minimum(d, np.interp(t, [a, a + down, b, b + up], [0.0, duck_db, duck_db, 0.0]))
    g = 10 ** (d / 20)
    k = min(n, round(fade_in * SR))
    g[:k] *= np.linspace(0, 1, k, endpoint=False)
    k = min(n, round(fade_out * SR))
    if k:
        g[n - k:] *= np.cos(np.linspace(0, np.pi / 2, k))
    return g


def _blocks(x, ffmpeg):
    """Momentary loudness as arrays (block centres s, LUFS): 400 ms blocks every 100 ms."""
    M = loudness(x, ffmpeg)["M"]
    return np.array([t - 0.2 for t, _ in M]), np.array([q for _, q in M])


def _gated(B, w0, w1, floor=-70.0, share=0.0):
    """BS.1770 gated loudness of [w0, w1] from _blocks: the blocks lying inside it (the
    blocks centred in it when it is under ~0.6 s), gates `floor` LUFS (-70) and -10 LU.
    -inf if none pass, or fewer than `share` of the blocks pass the floor."""
    c, q = B
    sel = (c >= w0 + 0.19) & (c <= w1 - 0.19)
    if sel.sum() < 2:
        sel = np.abs(c - (w0 + w1) / 2) <= max(0.06, (w1 - w0) / 2)
    q = q[sel]
    q = q[q > floor] if len(q) and (q > floor).mean() >= share else q[:0]
    if not len(q):
        return -np.inf
    e = 10 ** (q / 10)
    e = e[q > 10 * np.log10(e.mean()) - 10]
    return round(float(10 * np.log10(e.mean())), 2)


def _split(a, b, win):
    e = np.linspace(a, b, max(1, round((b - a) / win)) + 1)
    return [[float(p), float(q)] for p, q in zip(e[:-1], e[1:])]


def _level(m, frame=0.05, med=1.05, ramp=0.3, cap=20.0):
    """Per-sample gain (dB) that takes a signal's slow level changes out (builds, drops) and
    keeps what is faster than ~1 s (beats, phrasing): RMS per 50 ms frame, median over ~1 s
    (a step stays a step, beats drop out), mean over `ramp` s (the step becomes a ramp that
    long), relative to the median of the whole, capped at +/-cap dB. Frames more than 25 dB
    under the median (a stop, silence) take their neighbours' level, so a hole in the music
    is left a hole instead of being boosted into its edges."""
    f = round(frame * SR)
    d = _frames_db(np.sqrt((m ** 2).mean(1)), SR, frame)
    ok = d > np.median(d) - 25
    if ok.any():
        d = np.interp(np.arange(len(d)), np.flatnonzero(ok), d[ok])
    k = round(med / frame) | 1
    d = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(d, k // 2, mode="edge"), k), 1)
    k = round(ramp / frame)
    d = np.convolve(np.pad(d, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    d = np.clip(np.median(d) - d, -cap, cap)
    return np.interp(np.arange(len(m)), np.arange(len(d)) * f + f / 2, d)


def _bed_curve(v, m, regions, total, ffmpeg, under, lift, fade_out, win=1.0, down=0.3, up=0.4,
               up_last=0.25, tol=0.75, fix=0.9):
    """Music gain (dB per sample, no fades) from measurement. The music is first levelled
    (_level: its own builds and drops come out as 0.3 s ramps, so a step inside a window
    cannot pass through), then each speech region is cut into ~win s windows; a window's
    target music loudness is its gated voice loudness - under (a window with no speech,
    < region - 10 LU, uses the region's). Outside the regions the
    bed targets the neighbouring in-speech target + lift (the lower neighbour between two
    regions); the tail bed is measured up to the fade-out and held to the end. Gains start
    at target - the music's own loudness in the window, are smoothed within +/-tol dB of
    that (the flattest curve the band allows), joined linearly between window centres
    (flat to the group edges), ramp `down` s into a region, `up` s out (`up_last` after
    the last), then measured again: a speech window more than `fix` LU off is pulled back
    to 0.6 LU off, a bed window (>= 0.4 s) more than 0.3 LU off its measured neighbour +
    lift is put on it; up to 5 rounds (a window where the music stops is left out and
    takes its neighbours' gain). Returns (gain dB, windows, control points, rounds);
    the window gains and control points are on top of the levelling."""
    lev = _level(m)
    m = m * 10 ** (lev / 20)[:, None]
    Bv, Bm = _blocks(v, ffmpeg), _blocks(m, ffmpeg)
    wins, groups = [], []
    for a, b in regions:
        lr, idx = _gated(Bv, a, b), []
        for w in _split(a, b, win):
            lv = _gated(Bv, *w)
            pause = not lv > lr - 10
            idx.append(len(wins))
            wins.append({"t": w, "kind": "speech", "pause": pause, "target": (lr if pause else lv) - under})
        groups.append((a, b, idx))
    beds = []
    if regions[0][0] - down > 0.05:
        beds.append((0.0, regions[0][0] - down, [regions[0][0] - down], [groups[0][2][0]]))
    for j in range(len(regions) - 1):
        s0, s1 = regions[j][1] + up, regions[j + 1][0] - down
        if s1 - s0 > 0.05:
            beds.append((s0, s1, [s1], [groups[j][2][-1], groups[j + 1][2][0]]))
    s0 = regions[-1][1] + up_last
    if s0 < total - 0.05:
        beds.append((s0, total, [max(total - fade_out, min(s0 + 0.4, total))], [groups[-1][2][-1]]))
    for s0, s1, (m1,), ref in beds:
        t_ref, idx = min(wins[i]["target"] for i in ref), []
        for w in _split(s0, m1, win):
            idx.append(len(wins))
            wins.append({"t": w, "kind": "bed", "pause": False, "ref": ref, "target": t_ref + lift})
        groups.append((s0, s1, idx))
    mus = np.array([_gated(Bm, *w["t"]) for w in wins])
    mref = float(np.median(mus[np.isfinite(mus)])) if np.isfinite(mus).any() else -20.0
    mus = np.array([_gated(Bm, *w["t"], floor=mref - 15, share=0.5) for w in wins])
    for w, x in zip(wins, mus):        # the music stops for most of it: neighbours' gain, no feedback
        w["stop"] = not np.isfinite(x)
    for s0, s1, idx in groups:
        T = np.array([wins[i]["target"] for i in idx])
        G = T - mus[idx]
        ok = np.isfinite(G)
        G = np.interp(np.arange(len(G)), np.flatnonzero(ok), G[ok]) if ok.any() else T - mref
        g = G.copy()
        for _ in range(20):
            p = np.pad(g, 1, mode="edge")
            g = np.clip((p[:-2] + 2 * p[1:-1] + p[2:]) / 4, G - tol, G + tol)
        for i, x in zip(idx, g):
            wins[i]["g"] = float(np.clip(x, -40.0, 12.0))
    t = np.arange(len(v)) / SR
    for rounds in range(1, 7):
        P = sorted(p for s0, s1, idx in groups for p in
                   [(s0, wins[idx[0]]["g"])] + [(sum(wins[i]["t"]) / 2, wins[i]["g"]) for i in idx]
                   + [(s1, wins[idx[-1]]["g"])])
        gdb = np.interp(t, [p[0] for p in P], [p[1] for p in P])
        if rounds == 6:
            break
        Bg, moved = _blocks(m * 10 ** (gdb / 20)[:, None], ffmpeg), False
        meas = [_gated(Bg, *w["t"]) for w in wins]
        for w, mu in zip(wins, meas):
            if w["stop"]:
                continue
            if w["kind"] == "speech":
                e, lim, to = w["target"] - mu, fix, 0.6
            elif w["t"][1] - w["t"][0] >= 0.4:      # a shorter bed is ramps only: left open-loop
                e, lim, to = min(meas[i] for i in w["ref"]) + lift - mu, 0.3, 0.0
            else:
                continue
            if np.isfinite(e) and abs(e) > lim:
                w["g"], moved = float(np.clip(w["g"] + e - np.sign(e) * to, -40.0, 12.0)), True
        if not moved:
            break
    return gdb + lev, wins, P, rounds


def _window_report(y, v, wins, regions, m0, ffmpeg, m=None):
    """Measure a written mix: music isolated as y - a*v (a by least squares on v, or on
    v and the gained music m when given), then per window
    voice/music gated loudness and music-under-voice (speech) or lift over the lower
    neighbouring speech window's music (bed), and per region (from r0 + 0.12, the first
    word) the music's momentary swing and the part of it the gain adds (music M minus the
    raw music's M)."""
    X = np.stack([v.ravel()] + ([] if m is None else [m.ravel()]), 1)
    a = float(np.linalg.lstsq(X, y.ravel(), rcond=None)[0][0])
    Bv, Bm, B0 = _blocks(a * v, ffmpeg), _blocks(y - a * v, ffmpeg), _blocks(m0, ffmpeg)
    rows = []
    for w in wins:
        r = {"t": [round(w["t"][0], 2), round(w["t"][1], 2)], "kind": w["kind"],
             "gain_db": round(w.get("g", 0.0), 2), "music": _gated(Bm, *w["t"])}
        if w["kind"] == "speech":
            r.update(voice=_gated(Bv, *w["t"]), pause=w["pause"], stop=w.get("stop", False))
            r["under"] = round(r["voice"] - r["music"], 2)
        else:
            r["lift"] = round(r["music"] - min(_gated(Bm, *wins[i]["t"]) for i in w["ref"]), 2)
        rows.append(r)
    swing, gsw = [], []
    for r0, r1 in regions:
        sel = (Bm[0] >= r0 + 0.12 + 0.19) & (Bm[0] <= r1 - 0.19) & (Bm[1] > -70) & (B0[1] > -70)
        swing.append(round(float(np.ptp(Bm[1][sel])), 2) if sel.sum() > 1 else 0.0)
        gsw.append(round(float(np.ptp((Bm[1] - B0[1])[sel])), 2) if sel.sum() > 1 else 0.0)
    return rows, swing, gsw, round(a, 4)


def _master(s0, n, out, ffmpeg, lufs=-14.0, tp=-1.5):
    """Master a float (n, 2) sum to lufs LUFS / tp dBTP and write exactly n samples to out.

    Pass 1 measures with ebur128 (loudnorm's own first-pass input_i read 0.3 LU low, which left
    the linear pass 2 at -13.7); pass 2 is loudnorm linear=true. If the linear gain would push
    the true peak over tp, the sum is limited and measured again; repeated, since limiting lowers
    the loudness and so raises the gain needed. loudnorm only stays linear when the measured LRA
    is in (0, target LRA]; with linear=true the LRA target has no other use, so it is raised to
    the measured LRA rather than let loudnorm fall back to dynamic mode (which missed I by up to
    2 LU in tests). Returns (written y, loudnorm pass-2 json, limited dB, limiter gain, LRA
    target, loudnorm's own sample count)."""
    s = s0
    p1, limited, lg = loudness(s, ffmpeg), 0.0, np.ones(n)
    for _ in range(10):
        lift = lufs - p1["I"]
        if p1["TP"] + lift <= tp - 0.1:
            break
        s, limited, lg = _limit(s0, tp - 0.4 - lift)
        p1 = loudness(s, ffmpeg)
    else:
        raise RuntimeError(f"limiter did not converge (I {p1['I']}, TP {p1['TP']})")
    lra_in = min(max(p1["LRA"], 0.1), 49.9)            # loudnorm's LRA target tops out at 50
    lra_t = max(7.0, lra_in + 0.1)
    base = [ffmpeg, "-nostdin", "-hide_banner", "-f", "wav", "-i", "pipe:0", "-af"]
    ln = (f"loudnorm=I={lufs:g}:TP={tp:g}:LRA={lra_t:.1f}:measured_I={p1['I']}:measured_TP={p1['TP']}"
          f":measured_LRA={lra_in}:measured_thresh={p1['thresh']}"
          f":linear=true:print_format=json,aresample={SR}")
    raw, err = _run(base + [ln, "-f", "f32le", "-c:a", "pcm_f32le", "-"], _wav_bytes(s))
    p2 = _last_json(err)
    y = np.frombuffer(raw, "<f4").reshape(-1, 2).astype(np.float64)
    ln_len = len(y)
    y = _fit(y, n)
    _write_wav(out, y)
    y, _ = _read_wav(out)
    return y, p2, limited, lg, lra_t, ln_len


def _ln_keys(p2):
    return {k: p2.get(k) for k in ("normalization_type", "input_i", "input_tp", "output_i", "output_tp",
                                   "target_offset")}


def master(voice, total, out, ffmpeg, lufs=-14.0, tp=-1.5):
    """The voice alone (no music), mastered exactly as mix() masters its sum; exactly total s."""
    n = round(total * SR)
    v = _fit(_decode(voice, ffmpeg), n)
    y, p2, limited, _, lra_t, ln_len = _master(v, n, out, ffmpeg, lufs, tp)
    fin = loudness(out, ffmpeg)
    return {"samples": len(y), "I": fin["I"], "TP": fin["TP"], "LRA": fin["LRA"], "loudnorm": _ln_keys(p2),
            "loudnorm_out_samples": ln_len, "limited_db": limited, "lra_target": round(lra_t, 1),
            "music": None, "music_under": None, "bed_lift": None}


def mix(voice, music, speech, total, out, ffmpeg, duck_db=-10.0, music_lufs=-20.0,
        fade_out=0.7, last_word_end=None, music_under=14.0, bed_lift=5.0, win=1.0, lufs=-14.0, tp=-1.5):
    """Lay the music under the voice and master to lufs LUFS / tp dBTP (-14 / -1.5).

    voice, music: audio files (music from offset 0; ValueError if it is shorter than
    total). speech: (start, end) word times in program seconds; last_word_end defaults to
    the last word's end. Regions are speech_regions(...) with the last one ending at
    last_word_end + 0.05.
    music_under (LU): the music gain is set from measurement (_bed_curve): the music's own
    builds and drops are levelled out (0.3 s ramps), then ~win s windows of each region keep
    it music_under +/-1 LU below the voice; the bed outside the regions sits bed_lift LU
    above the neighbouring in-speech level; ramps 0.3 s in, 0.4 s out, 0.25 s after the
    last word; fade_out is shortened (not below 0.5 s) when that is needed to leave 0.4 s
    of full bed before it (fade_out_s says what was used).
    music_under=None: the old constant duck (music to music_lufs, duck_curve(duck_db)).
    Either way: 10 ms fade-in, qsin fade-out over the last fade_out s, sum, then _master:
    measure (ebur128), peak-limit only if a linear gain would break tp, loudnorm
    linear=true with those measurements, pad/trim to exactly total s.
    Returns measurements of the written file: I, TP, LRA, loudnorm, limited_db,
    lra_target, regions, windows (per window: voice, music, under or lift, mean music
    gain_db), under_lu [min, max] over speech windows (not pauses or music stops),
    bed_lift_lu (max), bed_full_s (full bed after the last ramp, before the fade),
    fade_out_s, music_gain_db ([min, max] in speech; the single normalising gain with
    music_under=None), music_swing_lu (momentary range of the isolated music inside each
    region), duck_swing_lu (the part of it the gain adds),
    min_ramp_s / max_slope_db_s of the gain, music_under_voice_lu over region interiors,
    tail_rms_db over [total-1.0, total-0.7], max_zero_run_ms, max_step_down_db (10 ms
    frames, last 2 s)."""
    n = round(total * SR)
    v = _fit(_decode(voice, ffmpeg), n)
    m0 = _decode(music, ffmpeg)
    if len(m0) < n:
        raise ValueError(f"{music}: {len(m0) / SR:.2f} s of music for a {total:.2f} s program "
                         "(the end would be digital silence)")
    m0 = m0[:n]
    m_in = loudness(m0, ffmpeg)["I"]
    regions = speech_regions(speech, last_word_end=last_word_end)
    wins, P, rounds, up_last = [], [], 0, 0.40
    if music_under is not None and regions:
        lwe = max(e for _, e in speech) if last_word_end is None else last_word_end
        regions[-1][1] = min(regions[-1][1], lwe + 0.05)
        regions = [[max(a, 0.0), min(b, total)] for a, b in regions if min(b, total) - max(a, 0.0) > 0.05]
    if music_under is not None and regions:
        # keep >= 0.4 s of full bed between the 0.25 s ramp-up and the fade (shorten the fade, not below 0.5 s)
        fade_out = min(fade_out, max(0.5, total - regions[-1][1] - 0.25 - 0.4))
        gdb, wins, P, rounds = _bed_curve(v, m0, regions, total, ffmpeg, music_under, bed_lift, fade_out, win)
        up_last = 0.25
        g = 10 ** (gdb / 20)
        k = min(n, round(0.01 * SR))
        g[:k] *= np.linspace(0, 1, k, endpoint=False)
        k = min(n, round(fade_out * SR))
        if k:
            g[n - k:] *= np.cos(np.linspace(0, np.pi / 2, k))
    else:
        gdb = np.full(n, music_lufs - m_in)
        g = 10 ** (gdb / 20) * duck_curve(n, regions, duck_db, fade_out=fade_out)
    m = m0 * g[:, None]
    s0 = v + m
    y, p2, limited, lg, lra_t, ln_len = _master(s0, n, out, ffmpeg, lufs, tp)

    fin = loudness(out, ffmpeg)
    vl = v * lg[:, None]                   # the voice as it is in the sum (after any limiting)
    rows, swings, duck_sw, a = _window_report(y, vl, wins, regions, m0, ffmpeg, m * lg[:, None])
    vm, mm = a * vl, y - a * vl
    inner = [(max(0, round((r0 + 0.12) * SR)), min(n, round(r1 * SR))) for r0, r1 in regions]
    idx = np.concatenate([np.arange(i, j) for i, j in inner if j > i]) if inner else np.arange(0)
    lv = loudness(vm[idx], ffmpeg)["I"] if len(idx) else float("nan")
    lm = loudness(mm[idx], ffmpeg)["I"] if len(idx) else float("nan")
    und = [r["under"] for r in rows if r["kind"] == "speech" and not (r["pause"] or r["stop"])]
    lifts = [r["lift"] for r in rows if r["kind"] == "bed"]
    ramps = [q[0] - p[0] for p, q in zip(P, P[1:]) if abs(q[1] - p[1]) >= 1.0]
    core = slice(round(0.01 * SR), n - round(fade_out * SR))       # gain without the fades
    gd, in_sp = 20 * np.log10(np.maximum(g, 1e-9)), np.zeros(n, bool)
    for r0, r1 in regions:
        in_sp[round(r0 * SR):round(r1 * SR)] = True
    in_sp[:round(0.01 * SR)] = False
    gsp = gd[in_sp] if in_sp.any() else gd[core]
    for r, w in zip(rows, wins):                   # the total music gain (levelling included)
        r["gain_db"] = round(float(gd[round(w["t"][0] * SR):round(w["t"][1] * SR)].mean()), 2)

    mono = y.mean(1)
    fdb = _frames_db(mono, SR)
    last2 = fdb[-200:]
    zero = np.all(y == 0, axis=1).astype(np.int8)
    d = np.diff(np.concatenate(([0], zero, [0])))
    zrun = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max(initial=0)
    seg = y[round((total - 1.0) * SR):round((total - 0.7) * SR)]
    return {
        "samples": len(y), "I": fin["I"], "TP": fin["TP"], "LRA": fin["LRA"],
        "loudnorm": _ln_keys(p2),
        "loudnorm_out_samples": ln_len, "limited_db": limited, "lra_target": round(lra_t, 1),
        "music_in_lufs": m_in,
        "music_gain_db": [round(float(gsp.min()), 2), round(float(gsp.max()), 2)] if wins
        else round(music_lufs - m_in, 2),
        "regions": [[round(r0, 3), round(r1, 3)] for r0, r1 in regions],
        "music_under": music_under, "bed_lift": bed_lift if wins else None, "gain_rounds": rounds,
        "windows": rows,
        "under_lu": [min(und), max(und)] if und else None,
        "bed_lift_lu": max(lifts) if lifts else None,
        "bed_full_s": round(total - fade_out - regions[-1][1] - up_last, 3) if regions else None,
        "fade_out_s": round(fade_out, 3),
        "min_ramp_s": round(min(ramps), 3) if ramps else None,
        "max_slope_db_s": round(float(np.abs(np.diff(gd[core][::480])).max(initial=0) * 100), 1),
        "voice_lufs_speech": lv, "music_lufs_speech": lm,
        "music_under_voice_lu": round(lv - lm, 2),
        "music_swing_lu": swings, "duck_swing_lu": duck_sw,
        "lsq_gain": [a, a],
        "tail_rms_db": round(float(10 * np.log10(max((seg ** 2).mean(), 1e-12))), 2),
        "max_zero_run_ms": round(float(1000 * zrun / SR), 2),
        "max_step_down_db": round(float(np.max(last2[:-1] - last2[1:], initial=0)), 2),
    }
