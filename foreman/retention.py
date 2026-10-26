"""Artifact retention and disk guards. Autonomy without retention fills the disk: one 5 s 704p
clip is about 330 MB of raw rgb8 frames.

  * After a successful MP4 mux the adapter deletes frames.rgb (keeping the sampled PNGs,
    audio.wav, video.json and output.mp4), unless keep_raw_frames is set. A failed mux keeps it.
  * Planner items: artifacts of rejected items are pruned 7 days after the verdict, and those of
    items nobody reviewed (expired in the review queue, or failed) after 14 days.
  * The planner admits nothing while the internal disk has less than min_disk_free_gb free or the
    artifacts directory holds more than artifacts_cap_gb.

Deletion is confined to directories strictly inside the artifacts root (resolved, no symlinks),
so a path in the database can never point it anywhere else.
"""
from __future__ import annotations

import json
import os
import shutil
import time

from . import ledger

DAY = 86400.0


def drop_raw_frames(art: dict, keep_raw: bool, note=lambda line: None) -> dict:
    """Called by the adapter once output.mp4 exists. Removes the raw frames file (the MP4, the
    sampled PNGs, audio.wav and video.json stay) unless keep_raw; records what it did in the
    artifact and in video.json."""
    frames = art.get("frames_path")
    if keep_raw or not isinstance(frames, str) or not os.path.exists(frames):
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

    for it in ledger.items(store, "rejected"):
        if it.get("pruned_at") is None and (it.get("verdict_at") or now) <= now - cfg.prune_rejected_days * DAY:
            prune(it, f"rejected {cfg.prune_rejected_days:g}+ days ago")
    for it in ledger.items(store, "review"):
        if (it.get("produced_at") or now) <= now - cfg.prune_unreviewed_days * DAY:
            prune(it, f"never reviewed in {cfg.prune_unreviewed_days:g} days", new_state="expired")
    for it in ledger.items(store, "failed"):
        if it.get("pruned_at") is None and (it.get("finished_at") or now) <= now - cfg.prune_unreviewed_days * DAY:
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


def disk_guard(probe, cfg) -> tuple:
    """(blocked, detail). A probe that fails blocks: unknown disk state is not a reason to fill it."""
    try:
        free, art = probe()
        free, art = float(free), float(art)
    except Exception as e:
        return True, f"disk probe failed: {type(e).__name__}: {e}"
    if free < cfg.min_disk_free_gb:
        return True, f"internal disk has {free:.0f} GB free (< {cfg.min_disk_free_gb:g} GB)"
    if art > cfg.artifacts_cap_gb:
        return True, f"artifacts use {art:.0f} GB (> cap {cfg.artifacts_cap_gb:g} GB)"
    return False, f"{free:.0f} GB free, artifacts {art:.0f} GB"
