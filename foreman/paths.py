"""Path confinement for files engines write and read on Foreman's behalf.

Job payloads live in the database and name files: a work directory under the artifacts root,
outputs inside it, inputs (a character's stills, a reference voice, a screen capture) under the
owner's directories. Every such name is resolved here, with symlinks followed, and refused unless
it stays strictly inside the root it belongs to, so a payload can never make an engine write or
upload a file anywhere else.
"""
from __future__ import annotations

import json
import os
import re

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PathRefused(ValueError):
    """A path outside the directory it must stay in (or not a regular file where one is needed)."""


def inside(path: str, root: str) -> bool:
    """`path` resolves to somewhere strictly inside `root` (both with symlinks followed)."""
    try:
        real, base = os.path.realpath(path), os.path.realpath(root)
    except (OSError, ValueError):
        return False
    return real != base and os.path.commonpath([real, base]) == base


def _relative(rel) -> str:
    if not isinstance(rel, str) or not rel.strip() or "\x00" in rel:
        raise PathRefused(f"not a relative path: {rel!r}")
    if os.path.isabs(rel) or any(part in ("..", "") for part in rel.replace("\\", "/").split("/")):
        raise PathRefused(f"{rel!r} must be a plain relative path (no '..', no leading '/')")
    return rel


def workdir(artifacts_dir: str, rel) -> str:
    """The absolute work directory `rel` names under the artifacts root (created if missing)."""
    path = os.path.join(os.path.realpath(artifacts_dir), _relative(rel))
    if not inside(path, artifacts_dir) and not inside(os.path.dirname(path), artifacts_dir):
        raise PathRefused(f"work directory {rel!r} is outside {artifacts_dir}")
    os.makedirs(path, exist_ok=True)
    if not inside(path, artifacts_dir):
        raise PathRefused(f"work directory {rel!r} resolves outside {artifacts_dir}")
    return path


def out_path(wd: str, rel) -> str:
    """An output file `rel` inside the work directory `wd` (its parent directories created)."""
    path = os.path.join(wd, _relative(rel))
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    if not (inside(path, wd) and (parent == wd or inside(parent, wd))):
        raise PathRefused(f"output {rel!r} resolves outside the work directory")
    if os.path.islink(path):
        raise PathRefused(f"output {rel!r} is a symlink")
    return path


def input_path(path, roots, wd: str | None = None) -> str:
    """A readable regular file: absolute, or relative to the work directory, and inside one of
    `roots` once resolved."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise PathRefused(f"not a file path: {path!r}")
    full = path if os.path.isabs(path) else os.path.join(wd or "", _relative(path))
    if not any(r and inside(full, r) for r in roots):
        raise PathRefused(f"{path!r} is outside the directories Foreman reads inputs from")
    real = os.path.realpath(full)
    if not os.path.isfile(real):
        raise PathRefused(f"{path!r} is not a file")
    return real


def atomic_write(path: str, data: bytes) -> None:
    """Write `data` to `path` all at once: the old file stays until the new one is complete, and a
    write that fails (a full disk, a kill) leaves no partial temp file behind."""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_json(path: str, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=str).encode())


def read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default
