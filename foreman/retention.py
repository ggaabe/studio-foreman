"""Artifact retention and disk guards. Autonomy without retention fills the disk: one 5 s 704p
clip is about 330 MB of raw rgb8 frames.

  * After a successful MP4 mux the adapter deletes frames.rgb (keeping the sampled PNGs,
    audio.wav, video.json and output.mp4), unless keep_raw_frames is set. A failed mux keeps it,
    and so does an MP4 whose box structure is truncated or incomplete (mp4_complete).
  * Planner items: artifacts of rejected items are pruned 7 days after the verdict, and those of
    items nobody reviewed (expired in the review queue, or failed) after 14 days.
  * The planner admits nothing while the internal disk has less than min_disk_free_gb free or the
    artifacts directory holds more than artifacts_cap_gb, counting what the work still in flight
    and the work admitted earlier in the same tick may write (estimate_bytes): the reading is
    cached, and admitted work writes later.

Deletion is confined to directories strictly inside the artifacts root (resolved, no symlinks),
so a path in the database can never point it anywhere else.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

from . import ledger
from .policy import number

DAY = 86400.0


def drop_raw_frames(art: dict, keep_raw: bool, note=lambda line: None) -> dict:
    """Called by the adapter once output.mp4 exists. Removes the raw frames file (the MP4, the
    sampled PNGs, audio.wav and video.json stay) unless keep_raw; records what it did in the
    artifact and in video.json."""
    frames = art.get("frames_path")
    mp4 = art.get("path")
    complete = mp4_complete(mp4) if isinstance(mp4, str) else False
    if complete is False and isinstance(frames, str) and os.path.exists(frames) and not keep_raw:
        note(f"kept raw frames: {os.path.basename(str(mp4))} is not a complete MP4")
    if keep_raw or complete is False or not isinstance(frames, str) or not os.path.exists(frames):
        if isinstance(frames, str) and frames not in art.setdefault("paths", []):
            art["paths"].append(frames)
        return art
    try:
        size = os.path.getsize(frames)
        os.remove(frames)
    except OSError as e:
        note(f"could not delete raw frames after the mux: {e}")
        if frames not in art.setdefault("paths", []):
            art["paths"].append(frames)
        return art
    art["frames_path"] = None
    art["raw_frames_deleted"] = True
    art["paths"] = [p for p in art.get("paths", []) if p != frames]
    note(f"deleted raw frames ({size / 1e6:.0f} MB) after the MP4 mux (keep_raw_frames is off)")
    meta = art.get("meta")
    if isinstance(meta, str) and os.path.exists(meta):
        try:
            with open(meta) as f:
                data = json.load(f)
            data.update(frames=None, raw_frames_deleted=True, mp4=os.path.basename(art.get("path") or ""))
            with open(meta, "w") as f:
                json.dump(data, f, indent=1)
        except (OSError, ValueError) as e:
            note(f"could not update video.json: {e}")
    return art


def mp4_complete(path: str):
    """True when `path` is an ISO-BMFF file (MP4) whose top-level boxes are all whole and include
    ftyp, moov and mdat; False when it is missing, empty, truncated or lacks one of them; None when
    it does not start with an ISO-BMFF box at all (not something this check can judge, e.g. a test
    stand-in for ffmpeg), in which case the caller trusts ffmpeg's exit status."""
    try:
        if os.path.islink(path) or not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if size <= 0:
            return False
        seen, pos = set(), 0
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8 or head[4:8] != b"ftyp":
                return None
            while pos < size:
                f.seek(pos)
                hdr = f.read(8)
                if len(hdr) < 8:
                    return False
                box, kind = int.from_bytes(hdr[:4], "big"), hdr[4:8]
                if box == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        return False
                    box = int.from_bytes(ext, "big")
                elif box == 0:
                    box = size - pos  # the last box runs to the end of the file
                if box < 8 or pos + box > size:
                    return False
                seen.add(kind)
                pos += box
        return {b"ftyp", b"moov", b"mdat"} <= seen
    except OSError:
        return False


def estimate_bytes(request: dict) -> float:
    """What one admitted request may write under the artifacts directory, at most: a render's raw
    rgb8 frames (deleted after a good mux, kept if the mux fails or keep_raw_frames is set) plus
    its MP4, PNG samples and audio; a song's WAV; speech is small."""
    payload = request.get("payload") if isinstance(request, dict) else None
    payload = payload if isinstance(payload, dict) else {}
    kind = request.get("kind") if isinstance(request, dict) else None

    def num(key, default):
        v = number(payload.get(key))
        return v if v is not None and v >= 0 else default
    if kind == "render_video":
        frames = num("num_frames", 121.0)
        raw = num("width", 768.0) * num("height", 512.0) * 3.0 * frames
        return raw * 1.2 + 48000 * 2 * 2 * num("duration_s", frames / 24.0)
    if kind == "compose_music":
        return 48000 * 2 * 2 * num("duration_s", 30.0) * 1.1
    return 2e6


def _inside(path: str, root: str) -> bool:
    try:
        real, base = os.path.realpath(path), os.path.realpath(root)
    except (OSError, ValueError):
        return False
    return real != base and os.path.commonpath([real, base]) == base


def _size(path: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def prune_dirs(dirs, root: str) -> dict:
    """Delete each run directory that lies strictly inside `root` (never a symlink, never the root
    itself, never outside it). Returns {"removed", "skipped", "bytes"}."""
    removed, skipped, freed = [], [], 0
    for d in dirs or []:
        if not isinstance(d, str) or not d:
            continue
        if not root or os.path.islink(d) or not _inside(d, root):
            skipped.append(d)
            continue
        if not os.path.isdir(d):
            continue  # already gone
        freed += _size(d)
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d)
    return {"removed": removed, "skipped": skipped, "bytes": freed}


def prune_due(store, now: float, root: str, cfg) -> list:
    """Prune planner items whose retention ran out. Returns [(item id, why, bytes)]."""
    done = []

    def prune(it: dict, why: str, new_state: str | None = None) -> None:
        if new_state:  # leave the review queue first, so a verdict racing this sees a settled state
            with store.tx():
                if not ledger.update_item(store, it["id"], expect_state=it["state"], state=new_state):
                    return
                store.event(now, "item_expired", it["job_id"], item=it["id"], template=it["template_id"])
        res = prune_dirs(it.get("artifact_dirs") or [], root)
        with store.tx():
            ledger.update_item(store, it["id"], pruned_at=now)
            store.event(now, "artifacts_pruned", it["job_id"], item=it["id"], why=why, removed=len(res["removed"]),
                        skipped=len(res["skipped"]), bytes=res["bytes"])
        done.append((it["id"], why, res["bytes"]))

    def since(it: dict, key: str) -> float:
        t = it.get(key)
        return now if t is None else t  # explicit: a timestamp of 0.0 is a time, not "missing"

    for it in ledger.items(store, "rejected"):
        if it.get("pruned_at") is None and since(it, "verdict_at") <= now - cfg.prune_rejected_days * DAY:
            prune(it, f"rejected {cfg.prune_rejected_days:g}+ days ago")
    for it in ledger.items(store, "review"):
        if since(it, "produced_at") <= now - cfg.prune_unreviewed_days * DAY:
            prune(it, f"never reviewed in {cfg.prune_unreviewed_days:g} days", new_state="expired")
    for it in ledger.items(store, "failed"):
        if it.get("pruned_at") is None and since(it, "finished_at") <= now - cfg.prune_unreviewed_days * DAY:
            prune(it, f"failed {cfg.prune_unreviewed_days:g}+ days ago")
    return done


class DiskProbe:
    """(internal disk free GB, artifacts dir GB). Walking a big artifacts tree is not free, so the
    reading is cached for cache_s seconds (real time)."""

    def __init__(self, home: str, artifacts_dir: str, cache_s: float = 600.0):
        self.home, self.artifacts_dir, self.cache_s = home, artifacts_dir, cache_s
        self._cached = None
        self._at = None

    def __call__(self) -> tuple:
        t = time.monotonic()
        if self._cached is not None and self._at is not None and t - self._at < self.cache_s:
            return self._cached
        path = self.home if os.path.exists(self.home) else os.path.expanduser("~")
        free = shutil.disk_usage(path).free / 1e9
        art = _size(self.artifacts_dir) / 1e9 if os.path.isdir(self.artifacts_dir) else 0.0
        self._cached, self._at = (free, art), t
        return self._cached


def disk_reading(probe) -> tuple:
    """((free GB, artifacts GB), None), or (None, why) when the probe fails or its reading is not
    two finite, non-negative numbers (NaN compares False with everything, so it must not reach a
    threshold check)."""
    try:
        free, art = probe()
        free, art = float(free), float(art)
    except Exception as e:
        return None, f"disk probe failed: {type(e).__name__}: {e}"
    if not (math.isfinite(free) and math.isfinite(art)) or free < 0 or art < 0:
        return None, f"disk probe gave an unusable reading ({free!r} GB free, {art!r} GB of artifacts)"
    return (free, art), None


def disk_verdict(reading: tuple, cfg, reserved_gb: float = 0.0) -> tuple:
    """(blocked, detail) for a reading, after `reserved_gb` more is written (work already admitted
    this tick but not on disk yet)."""
    free, art = reading[0] - reserved_gb, reading[1] + reserved_gb
    extra = f" after {reserved_gb:.1f} GB reserved for admitted work" if reserved_gb >= 0.05 else ""
    if free < cfg.min_disk_free_gb:
        return True, f"internal disk has {free:.0f} GB free{extra} (< {cfg.min_disk_free_gb:g} GB)"
    if art > cfg.artifacts_cap_gb:
        return True, f"artifacts use {art:.0f} GB{extra} (> cap {cfg.artifacts_cap_gb:g} GB)"
    return False, f"{free:.0f} GB free, artifacts {art:.0f} GB{extra}"


def disk_guard(probe, cfg) -> tuple:
    """(blocked, detail). A probe that fails blocks: unknown disk state is not a reason to fill it."""
    reading, err = disk_reading(probe)
    if reading is None:
        return True, err
    return disk_verdict(reading, cfg)
