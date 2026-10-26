"""Standard-library artifact analysis: the measurements the validators need, taken from the
files the engine actually wrote rather than from anything the engine claims.

  * WAV: duration, RMS level, peak and silence ratio (a manual RIFF reader, so 8/16/24/32-bit
    PCM and 32-bit float all work; `wave` writes the WAV files gpud produces itself).
  * Raw rgb8 video (what mlx-serve's /v1/video/generations returns): black-frame ratio,
    frozen-frame ratio and duration = frames / fps, from a fixed-size pixel sample per frame so
    a 300 MB clip is measured in well under a second of pure Python.
  * A PNG encoder (zlib + struct) for frame samples and audio waveforms the vision brain reads.

Nothing here raises on a merely bad file: the analysis functions raise MediaError, which the
adapter turns into a failed artifact check (a silent Metal failure can hand back zero-filled
buffers with no error, so these numbers are the only proof a render is real).
"""
from __future__ import annotations

import array
import math
import operator
import os
import struct
import sys
import wave
import zlib

SILENCE_DBFS = -50.0          # a 50 ms window quieter than this counts as silent
SILENCE_WINDOW_S = 0.05
BLACK_PIXEL_LEVEL = 26        # ~10 % of full scale, like ffmpeg blackdetect's pix_th=0.10
BLACK_PICTURE_RATIO = 0.98    # ... and a frame is black when 98 % of its pixels are that dark
# Mean |delta| (0..255, sampled pixels) below which a frame repeats the previous one. Measured on
# the Studio: a slow but real LTX shot (a balloon rising over mist, 1280x704) moves 0.15-0.44 per
# frame, so the first guess of 0.5 called it 100 % frozen; repeated frames measure 0.
FROZEN_MEAN_DIFF = 0.1
PIXEL_SAMPLES = 4096          # sampled pixels per frame
AUDIO_SAMPLE_BUDGET = 2_000_000


class MediaError(ValueError):
    """The file is missing, truncated or not in a format this module understands."""


# ---------------------------------------------------------------- WAV

def _read_wav(path: str):
    """(sample_rate, channels, bits, is_float, data_bytes) from a RIFF/WAVE file."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                raise MediaError(f"{path}: not a RIFF/WAVE file")
            fmt = None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
                if cid == b"fmt ":
                    raw = f.read(size)
                    if len(raw) < 16:
                        raise MediaError(f"{path}: short fmt chunk")
                    tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", raw[:16])
                    if tag == 0xFFFE and len(raw) >= 26:  # WAVE_FORMAT_EXTENSIBLE: real tag in the GUID
                        tag = struct.unpack("<H", raw[24:26])[0]
                    fmt = (tag, channels, rate, bits)
                elif cid == b"data":
                    if fmt is None:
                        raise MediaError(f"{path}: data chunk before fmt chunk")
                    # A streaming writer may leave the size as 0 or 0xFFFFFFFF: read to EOF then.
                    data = f.read() if size in (0, 0xFFFFFFFF) else f.read(size)
                    tag, channels, rate, bits = fmt
                    if tag not in (1, 3):
                        raise MediaError(f"{path}: unsupported WAV format tag {tag}")
                    if channels < 1 or rate < 1 or bits not in (8, 16, 24, 32):
                        raise MediaError(f"{path}: unsupported WAV layout {channels}ch {rate}Hz {bits}bit")
                    return rate, channels, bits, tag == 3, data
                else:
                    f.seek(size + (size & 1), os.SEEK_CUR)
    except OSError as e:
        raise MediaError(f"{path}: {e}") from None
    raise MediaError(f"{path}: no data chunk")


def _samples(data: bytes, bits: int, is_float: bool, stride: int):
    """Every `stride`-th sample as floats in [-1, 1] (an array, so slicing stays in C)."""
    if is_float:
        if bits != 32:
            raise MediaError(f"unsupported float WAV width {bits}")
        a = array.array("f")
        a.frombytes(data[: len(data) // 4 * 4])
        if sys.byteorder == "big":
            a.byteswap()
        return a[::stride], 1.0
    if bits == 8:  # unsigned
        b = data[::stride]
        return array.array("h", (x - 128 for x in b)), 128.0
    if bits == 24:  # keep the top 16 bits of each sample: plenty for level and silence
        n = len(data) // 3
        hi = bytearray(n * 2)
        hi[0::2] = data[1: n * 3: 3]
        hi[1::2] = data[2: n * 3: 3]
        data, bits = bytes(hi), 16
    code = "h" if bits == 16 else "i"
    a = array.array(code)
    width = a.itemsize
    a.frombytes(data[: len(data) // width * width])
    if sys.byteorder == "big":
        a.byteswap()
    return a[::stride], 32768.0 if bits == 16 else 2147483648.0


def _stride(total: int, channels: int) -> int:
    """Sample stride that keeps the work bounded and still visits every channel (a stride that
    is a multiple of the channel count would only ever read the left channel)."""
    stride = max(1, total // AUDIO_SAMPLE_BUDGET)
    if stride > 1 and channels > 1 and stride % channels == 0:
        stride += 1
    return stride


def wav_stats(path: str, silence_dbfs: float = SILENCE_DBFS, window_s: float = SILENCE_WINDOW_S) -> dict:
    """Duration, level and silence ratio of a WAV file."""
    rate, channels, bits, is_float, data = _read_wav(path)
    frame_bytes = channels * bits // 8
    frames = len(data) // frame_bytes
    if frames == 0:
        return {"duration_s": 0.0, "sample_rate": rate, "channels": channels, "bits": bits,
                "rms": 0.0, "rms_dbfs": -120.0, "peak": 0.0, "silence_ratio": 1.0}
    total = frames * channels
    stride = _stride(total, channels)
    samples, scale = _samples(data[: frames * frame_bytes], bits, is_float, stride)
    n = len(samples)
    per_window = max(1, int(rate * window_s * channels / stride))
    threshold = 10 ** (silence_dbfs / 20.0) * scale
    thr2 = threshold * threshold
    silent = windows = 0
    sumsq_total = 0.0
    peak = 0.0
    for lo in range(0, n, per_window):
        chunk = samples[lo: lo + per_window]
        if not chunk:
            break
        sq = math.fsum(map(operator.mul, chunk, chunk))
        sumsq_total += sq
        windows += 1
        if sq / len(chunk) < thr2:
            silent += 1
        peak = max(peak, max(chunk), -min(chunk))
    rms = math.sqrt(sumsq_total / n) / scale if n else 0.0
    return {"duration_s": frames / float(rate), "sample_rate": rate, "channels": channels, "bits": bits,
            "rms": rms, "rms_dbfs": 20 * math.log10(rms) if rms > 0 else -120.0,
            "peak": min(1.0, peak / scale), "silence_ratio": silent / windows if windows else 1.0}


def write_wav(path: str, pcm_s16le: bytes, sample_rate: int, channels: int) -> None:
    """Wrap interleaved little-endian 16-bit PCM into a WAV file."""
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_s16le[: len(pcm_s16le) // (2 * channels) * 2 * channels])


def wav_from_pcm_file(pcm_path: str, wav_path: str, sample_rate: int, channels: int,
                      chunk: int = 1 << 22) -> None:
    """Streamed version of write_wav for a PCM file on disk (video soundtracks)."""
    with wave.open(wav_path, "wb") as w, open(pcm_path, "rb") as src:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        block = 2 * channels
        carry = b""
        while True:
            data = src.read(chunk)
            if not data:
                break
            data = carry + data
            cut = len(data) // block * block
            w.writeframes(data[:cut])
            carry = data[cut:]


def waveform_png(path: str, width: int = 512, height: int = 128) -> bytes:
    """A grey min/max envelope of a WAV file, for a vision model that cannot listen."""
    rate, channels, bits, is_float, data = _read_wav(path)
    frame_bytes = channels * bits // 8
    frames = len(data) // frame_bytes
    stride = _stride(frames * channels, channels)
    samples, scale = _samples(data[: frames * frame_bytes], bits, is_float, stride)
    img = bytearray(b"\x10" * (width * height))
    mid = height // 2
    for x in range(width):
        img[mid * width + x] = 0x50
    n = len(samples)
    if n:
        per_col = max(1, n // width)
        for x in range(width):
            col = samples[x * per_col: (x + 1) * per_col]
            if not col:
                break
            hi, lo = max(col) / scale, min(col) / scale
            y0 = max(0, min(height - 1, int(mid - hi * (mid - 1))))
            y1 = max(0, min(height - 1, int(mid - lo * (mid - 1))))
            for y in range(min(y0, y1), max(y0, y1) + 1):
                img[y * width + x] = 0xE0
    return encode_png(width, height, bytes(img), channels=1)


# ---------------------------------------------------------------- raw rgb8 video

def _frame_reader(path: str, width: int, height: int, frame_count: int):
    size = width * height * 3
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise MediaError(f"bad video geometry {width}x{height}x{frame_count}")
    try:
        have = os.path.getsize(path)
    except OSError as e:
        raise MediaError(f"{path}: {e}") from None
    if have < size * frame_count:
        raise MediaError(f"{path}: {have} bytes, expected {size * frame_count} for "
                         f"{frame_count} frames of {width}x{height} rgb8 (truncated)")
    return size


def _sample_pixels(frame: bytes, width: int, height: int):
    """(R, G, B) byte strings of an evenly strided pixel sample."""
    stride = max(1, (width * height) // PIXEL_SAMPLES)
    step = 3 * stride
    return frame[0::step], frame[1::step], frame[2::step]


def video_stats(path: str, width: int, height: int, frame_count: int, fps: float) -> dict:
    """Black and frozen frame ratios plus duration for raw interleaved rgb8 frames."""
    size = _frame_reader(path, width, height, frame_count)
    dark_level = BLACK_PIXEL_LEVEL * 3
    black = frozen = 0
    prev = first = None
    luma_sum = motion_sum = 0.0
    with open(path, "rb") as f:
        for _ in range(frame_count):
            frame = f.read(size)
            r, g, b = _sample_pixels(frame, width, height)
            n = len(r)
            sums = list(map(operator.add, map(operator.add, r, g), b))
            dark = sum(1 for s in sums if s < dark_level)
            if n and dark / n >= BLACK_PICTURE_RATIO:
                black += 1
            luma_sum += sum(sums) / (3.0 * n) if n else 0.0
            cur = r + g + b
            if prev is not None and len(prev) == len(cur):
                diff = sum(map(abs, map(operator.sub, cur, prev))) / float(len(cur) or 1)
                motion_sum += diff
                if diff < FROZEN_MEAN_DIFF:
                    frozen += 1
            if first is None:
                first = cur
            prev = cur
    fps = float(fps) if fps else 0.0
    span = (sum(map(abs, map(operator.sub, prev, first))) / float(len(prev) or 1)
            if prev is not None and first is not None and len(prev) == len(first) else 0.0)
    return {"frame_count": frame_count, "width": width, "height": height, "fps": fps,
            "duration_s": frame_count / fps if fps > 0 else None,
            "black_frames": black, "black_frame_ratio": black / frame_count,
            "frozen_frames": frozen, "frozen_frame_ratio": frozen / max(1, frame_count - 1) if frame_count > 1 else 0.0,
            "mean_luma": luma_sum / frame_count,
            # how much the picture moves: mean |delta| between consecutive frames, and first vs last
            "motion_mean": motion_sum / (frame_count - 1) if frame_count > 1 else 0.0,
            "span_diff": span}


def read_frame(path: str, width: int, height: int, index: int) -> bytes:
    size = width * height * 3
    with open(path, "rb") as f:
        f.seek(index * size)
        frame = f.read(size)
    if len(frame) != size:
        raise MediaError(f"{path}: frame {index} is truncated")
    return frame


def downscale_rgb(width: int, height: int, rgb: bytes, max_side: int) -> tuple:
    """Nearest-neighbour downscale by an integer factor so the long side is <= max_side."""
    k = max(1, math.ceil(max(width, height) / float(max_side)))
    if k == 1:
        return width, height, rgb
    nw, nh = (width + k - 1) // k, (height + k - 1) // k
    out = bytearray(nw * nh * 3)
    row_bytes, step = width * 3, 3 * k
    for j, y in enumerate(range(0, height, k)):
        row = rgb[y * row_bytes: (y + 1) * row_bytes]
        o = j * nw * 3
        out[o + 0: o + nw * 3: 3] = row[0::step]
        out[o + 1: o + nw * 3: 3] = row[1::step]
        out[o + 2: o + nw * 3: 3] = row[2::step]
    return nw, nh, bytes(out)


def sample_indices(frame_count: int, n: int) -> list:
    if frame_count <= 0 or n <= 0:
        return []
    if n == 1 or frame_count == 1:
        return [frame_count // 2]
    n = min(n, frame_count)
    return sorted({round(i * (frame_count - 1) / (n - 1)) for i in range(n)})


def sample_frames_png(path: str, width: int, height: int, frame_count: int, n: int = 4,
                      max_side: int = 512) -> list:
    """[(frame index, PNG bytes)] for n evenly spaced frames, downscaled for a vision model."""
    _frame_reader(path, width, height, frame_count)
    out = []
    for i in sample_indices(frame_count, n):
        w, h, rgb = downscale_rgb(width, height, read_frame(path, width, height, i), max_side)
        out.append((i, encode_png(w, h, rgb)))
    return out


# ---------------------------------------------------------------- PNG

def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(width: int, height: int, pixels: bytes, channels: int = 3, level: int = 6) -> bytes:
    """8-bit grey (1), RGB (3) or RGBA (4) pixels, rows top to bottom, no filtering."""
    color = {1: 0, 3: 2, 4: 6}.get(channels)
    if color is None:
        raise ValueError(f"unsupported channel count {channels}")
    stride = width * channels
    if width <= 0 or height <= 0 or len(pixels) < stride * height:
        raise ValueError(f"{len(pixels)} bytes do not hold {width}x{height}x{channels}")
    raw = bytearray()
    for y in range(height):
        raw += b"\x00"
        raw += pixels[y * stride: (y + 1) * stride]
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(bytes(raw), level))
            + _chunk(b"IEND", b""))


def decode_png(data: bytes) -> tuple:
    """(width, height, channels, pixels) for the unfiltered PNGs encode_png writes (tests and
    sanity checks only; not a general PNG decoder)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise MediaError("not a PNG")
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        n = struct.unpack(">I", data[pos: pos + 4])[0]
        kind, body = data[pos + 4: pos + 8], data[pos + 8: pos + 8 + n]
        crc = struct.unpack(">I", data[pos + 8 + n: pos + 12 + n])[0]
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc:
            raise MediaError(f"bad CRC in {kind!r}")
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    if ihdr is None:
        raise MediaError("no IHDR")
    w, h, depth, color = ihdr[:4]
    channels = {0: 1, 2: 3, 6: 4}[color]
    raw = zlib.decompress(idat)
    stride = w * channels
    rows = []
    for y in range(h):
        line = raw[y * (stride + 1): (y + 1) * (stride + 1)]
        if line[0] != 0:
            raise MediaError("filtered rows are not supported by this decoder")
        rows.append(line[1:])
    return w, h, channels, b"".join(rows)
