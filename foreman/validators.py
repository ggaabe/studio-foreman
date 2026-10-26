"""Deterministic artifact checks. They run on metadata, so a failing check is a hard fact the
brain can read in the resume brief, not an opinion.

Engine metadata is untrusted: a missing or non-numeric field is a failed check with a detail,
never an exception. An exception here would leave the job stuck in 'running' forever."""
from __future__ import annotations

from .policy import number

MAX_BLACK_FRAME_RATIO = 0.2
MAX_SILENCE_RATIO = 0.5


def check(family: str, payload: dict, artifact) -> list[dict]:
    results: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    if artifact is not None and not isinstance(artifact, dict):
        add("artifact", False, f"engine returned a {type(artifact).__name__}, not an artifact object")
        return results
    art = artifact or {}
    payload = payload if isinstance(payload, dict) else {}
    if family == "brain":
        add("decision", isinstance(art.get("decision"), dict), "brain returned a decision object")
        return results

    path = art.get("path")
    add("exists", bool(art.get("exists") and isinstance(path, str) and path), f"path={path}")
    raw_want = payload.get("duration_s")
    if family in ("video", "music", "tts") and raw_want is not None:
        want = number(raw_want)
        got = number(art.get("duration_s"))
        if want is None or want <= 0:
            add("duration", False, f"requested duration {raw_want!r} is not a positive number")
        else:
            tol = max(0.5, 0.1 * want)
            add("duration", got is not None and abs(got - want) <= tol,
                f"got {art.get('duration_s')!r}s, want {want:g}s +/- {tol:g}s")
    if family == "video":
        ratio = number(art.get("black_frame_ratio"))
        add("black_frames", ratio is not None and ratio <= MAX_BLACK_FRAME_RATIO,
            f"black_frame_ratio={art.get('black_frame_ratio')!r}"
            + ("" if ratio is not None else " (missing or not a number: clip could not be probed)"))
    if family in ("music", "tts"):
        ratio = number(art.get("silence_ratio"))
        add("silence", ratio is not None and ratio <= MAX_SILENCE_RATIO,
            f"silence_ratio={art.get('silence_ratio')!r}"
            + ("" if ratio is not None else " (missing or not a number: audio could not be probed)"))
    if family == "image":
        w, h = number(art.get("width")), number(art.get("height"))
        add("dimensions", bool(w and h and w > 0 and h > 0), f"{art.get('width')!r}x{art.get('height')!r}")
    return results


def passed(results: list[dict]) -> bool:
    return all(r["ok"] for r in results)


def failures(results: list[dict]) -> str:
    return "; ".join(f"{r['check']}: {r['detail']}" for r in results if not r["ok"])
