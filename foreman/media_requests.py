"""Request builders for the three media kinds (speech, music, video), shared by the CLI and the
planner. Each returns a complete gpud request: kind, model, payload (with the goal and expected
duration the resume brief and the lease need) and both continuations (the baton).

`model` defaults to the Studio deployment's pack for that family; the planner passes the model it
resolved from the catalog it runs with (the simulator's catalog uses other names).
"""
from __future__ import annotations

from . import studio
from .supervisor import AdmissionError

KIND_FAMILY = {"speak": "tts", "compose_music": "music", "render_video": "video"}


def speech_request(text: str, voice: str | None = None, speed: float | None = None,
                   expected_s: float | None = None, model: str | None = None, seed: int | None = None) -> dict:
    payload = {"goal": f"speak: {text[:120]}", "input": text,
               "expected_s": expected_s or 30.0 + 0.05 * len(text)}
    if voice:
        payload["voice"] = voice
    if speed:
        payload["speed"] = speed
    if seed is not None:
        payload["seed"] = seed
    return {"kind": "speak", "requires": model or studio.KOKORO, "payload": payload,
            "on_success": {"kind": "review_audio", "needs": "vision"},
            "on_failure": {"kind": "recover_audio"}}


def music_request(prompt: str, seconds: float, lyrics: str | None = None, seed: int | None = None,
                  expected_s: float | None = None, model: str | None = None, instrumental: bool | None = None,
                  bpm: int | None = None) -> dict:
    if not 10 <= seconds <= 600:
        raise AdmissionError("ACE-Step takes 10 to 600 seconds")
    payload = {"goal": f"{seconds:g}s of music: {prompt[:120]}", "prompt": prompt, "duration_s": float(seconds),
               "expected_s": expected_s or 45.0 + 1.5 * seconds}
    if lyrics:
        payload["lyrics"] = lyrics
    else:
        payload["instrumental"] = True if instrumental is None else bool(instrumental)
    if seed is not None:
        payload["seed"] = seed
    if bpm is not None:
        payload["bpm"] = int(bpm)
    return {"kind": "compose_music", "requires": model or studio.ACE_STEP, "payload": payload,
            "on_success": {"kind": "review_audio", "needs": "vision"},
            "on_failure": {"kind": "recover_music"}}


def video_expected_s(mpix_frames: float) -> float:
    """Expected LTX-2.5 4-bit render time (request to artifact on disk) for frames x megapixels,
    with headroom for GPU contention. Measured on the Studio (M5 Ultra, other inference services
    sharing the GPU), distilled 8-step schedule:
        49 frames 512x320 (8 MPF)       9-12 s
        121 frames 1280x704 (109 MPF)   70-82 s   (denoise ~6 s/step, decode + transfer ~17 s)
        241 frames 1280x704 (217 MPF)   177 s     (denoise ~15 s/step, decode + transfer ~35 s)
    This gives about 1.9x those times (53 s, 166 s, 309 s); the lease's hard timeout is 2.5x it."""
    return min(640.0, 45.0 + 1.0 * mpix_frames + 0.001 * mpix_frames ** 2)


def video_request(prompt: str, seconds: float, width: int = 768, height: int = 512, steps: int | None = None,
                  seed: int | None = None, expected_s: float | None = None, model: str | None = None) -> dict:
    from .adapters.mlx_serve import VIDEO_FPS, video_frames_for
    if seconds <= 0 or seconds > 20:
        raise AdmissionError("video length must be in (0, 20] seconds")
    w, h = max(64, int(width) // 32 * 32), max(64, int(height) // 32 * 32)  # LTX works on a 32 px grid
    frames = video_frames_for(seconds)
    mpix_frames = w * h * frames / 1e6
    payload = {"goal": f"{seconds:g}s {w}x{h} clip: {prompt[:120]}", "prompt": prompt, "width": w, "height": h,
               "num_frames": frames, "duration_s": frames / float(VIDEO_FPS),
               "expected_s": expected_s or video_expected_s(mpix_frames)}
    if steps:
        payload["steps"] = int(steps)
    if seed is not None:
        payload["seed"] = seed
    return {"kind": "render_video", "requires": model or studio.LTX, "payload": payload,
            "on_success": {"kind": "review_render", "needs": "vision"},
            "on_failure": {"kind": "recover_render"}}
