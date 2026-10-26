"""Foreman command line: submit work to gpud and look at its state.

    python3 -m foreman.cli submit-speech "Hello from the Studio" [--voice af_heart]
    python3 -m foreman.cli submit-music "warm lo-fi piano, rain" --seconds 30 [--lyrics TEXT]
    python3 -m foreman.cli submit-video "a red fox trotting through snow" --seconds 5 [--width 768 --height 512]
    python3 -m foreman.cli status [--live]
    python3 -m foreman.cli jobs [--all]
    python3 -m foreman.cli events [--tail 30] [--job ID]
    python3 -m foreman.cli cancel JOB_ID
    python3 -m foreman.cli clear-safe

The CLI never talks to mlx-serve (except `status --live`, which only reads /api/ps) and never
takes gpud's lock: it opens the same SQLite file as a non-leader client (Supervisor(lead=False))
and goes through the supervisor's public submit(), so every admission rule applies. The running
gpud picks the job up on its next tick.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import studio
from .clock import SystemClock
from .policy import number
from .supervisor import AdmissionError, Supervisor


class _ClientEngine:
    """Admission only checks that an engine is configured; a client never calls it."""

    name = studio.ENGINE

    def __getattr__(self, item):
        raise AttributeError(f"the CLI does not talk to engines (asked for {item!r})")


def _client(cfg) -> Supervisor:
    if not os.path.exists(os.path.dirname(cfg.db_path) or "."):
        os.makedirs(os.path.dirname(cfg.db_path), exist_ok=True)
    return Supervisor(cfg.db_path, {studio.ENGINE: _ClientEngine()}, SystemClock(), studio.policy(),
                      catalog=studio.catalog(cfg.video_exclusive), lead=False)


def _when(wall) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall)) if wall else "-"


def _short(model) -> str:
    return str(model).split("/")[-1] if model else "-"


# ---------------------------------------------------------------- submissions

def speech_request(text: str, voice: str | None = None, speed: float | None = None,
                   expected_s: float | None = None) -> dict:
    payload = {"goal": f"speak: {text[:120]}", "input": text,
               "expected_s": expected_s or 30.0 + 0.05 * len(text)}
    if voice:
        payload["voice"] = voice
    if speed:
        payload["speed"] = speed
    return {"kind": "speak", "requires": studio.KOKORO, "payload": payload,
            "on_success": {"kind": "review_audio", "needs": "vision"},
            "on_failure": {"kind": "recover_audio"}}


def music_request(prompt: str, seconds: float, lyrics: str | None = None, seed: int | None = None,
                  expected_s: float | None = None) -> dict:
    if not 10 <= seconds <= 600:
        raise AdmissionError("ACE-Step takes 10 to 600 seconds")
    payload = {"goal": f"{seconds:g}s of music: {prompt[:120]}", "prompt": prompt, "duration_s": float(seconds),
               "expected_s": expected_s or 45.0 + 1.5 * seconds}
    if lyrics:
        payload["lyrics"] = lyrics
    else:
        payload["instrumental"] = True
    if seed is not None:
        payload["seed"] = seed
    return {"kind": "compose_music", "requires": studio.ACE_STEP, "payload": payload,
            "on_success": {"kind": "review_audio", "needs": "vision"},
            "on_failure": {"kind": "recover_music"}}


def video_request(prompt: str, seconds: float, width: int = 768, height: int = 512, steps: int | None = None,
                  seed: int | None = None, expected_s: float | None = None) -> dict:
    from .adapters.mlx_serve import VIDEO_FPS, video_frames_for
    if seconds <= 0 or seconds > 20:
        raise AdmissionError("video length must be in (0, 20] seconds")
    w, h = max(64, int(width) // 32 * 32), max(64, int(height) // 32 * 32)  # LTX works on a 32 px grid
    frames = video_frames_for(seconds)
    mpix_frames = w * h * frames / 1e6
    payload = {"goal": f"{seconds:g}s {w}x{h} clip: {prompt[:120]}", "prompt": prompt, "width": w, "height": h,
               "num_frames": frames, "duration_s": frames / float(VIDEO_FPS),
               "expected_s": expected_s or min(640.0, 90.0 + 6.0 * mpix_frames)}
    if steps:
        payload["steps"] = int(steps)
    if seed is not None:
        payload["seed"] = seed
    return {"kind": "render_video", "requires": studio.LTX, "payload": payload,
            "on_success": {"kind": "review_render", "needs": "vision"},
            "on_failure": {"kind": "recover_render"}}


def _submit(cfg, req: dict) -> int:
    sup = _client(cfg)
    try:
        job = sup.submit(req["kind"], req["requires"], req["payload"], req["on_success"], req["on_failure"])
    finally:
        sup.close()
    print(f"job {job} queued: {req['kind']} on {req['requires']} "
          f"(expected {req['payload']['expected_s']:.0f}s; gpud picks it up on its next tick)")
    return job


# ---------------------------------------------------------------- views

def cmd_status(cfg, live: bool) -> None:
    sup = _client(cfg)
    try:
        st = sup.store.state()
        jobs = sup.store.jobs()
    finally:
        sup.close()
    by_state: dict = {}
    for j in jobs:
        by_state[j.state] = by_state.get(j.state, 0) + 1
    print(f"db               {cfg.db_path}")
    print(f"mode             {st['mode']}" + (f"  SAFE: {st['safe_reason']}" if st["safe"] else ""))
    print(f"resident         {', '.join(_short(m) for m in sorted(st['known_resident'] or {})) or '-'}")
    if st["loading"]:
        ld = st["loading"]
        print(f"loading          {_short(ld.get('model'))} (for job {ld.get('for_job')})")
    if st["brainless_since"] is not None and st["t_now"] is not None:
        print(f"brain away       {st['t_now'] - st['brainless_since']:.0f}s")
    health = {k: v for k, v in (st["engine_health"] or {}).items() if v.get("down_since") is not None}
    if health:
        print(f"engines down     {json.dumps(health)}")
    print(f"gpud boots       {st['boot_count']}  (last tick {_when(st['t_wall'])})")
    print(f"jobs             {json.dumps(by_state, sort_keys=True)}")
    active = [j for j in jobs if j.state in ("starting", "running", "cancelling", "queued", "awaiting_human")]
    for j in active[-15:]:
        print(f"  {j.id:>5} {j.state:<15} {j.kind:<16} {_short(j.requires):<34} attempt {j.attempt}")
    if live:
        from .httpio import json_call
        try:
            _, ps = json_call(cfg.base_url, "GET", "/api/ps", timeout=5.0)
            names = [m.get("name") for m in (ps or {}).get("models", [])]
            print(f"mlx-serve /api/ps {names}")
        except Exception as e:
            print(f"mlx-serve /api/ps unreachable: {type(e).__name__}: {e}")


def cmd_jobs(cfg, show_all: bool) -> None:
    sup = _client(cfg)
    try:
        jobs = sup.store.jobs()
    finally:
        sup.close()
    if not show_all:
        jobs = jobs[-40:]
    print(f"{'id':>5} {'state':<15} {'kind':<16} {'requires':<34} {'att':>3} {'parent':>6}  reason")
    for j in jobs:
        o = j.outcome or {}
        reason = o.get("reason") or ""
        if j.requires == "brain" and isinstance(o.get("artifact"), dict):
            d = o["artifact"].get("decision") or {}
            reason = f"{d.get('verdict')}: {str(d.get('note') or '')[:60]}"
        elif j.state == "awaiting_human":
            reason = str((j.payload or {}).get("reason") or "")[:80]
        print(f"{j.id:>5} {j.state:<15} {j.kind:<16} {_short(j.requires):<34} {j.attempt:>3} "
              f"{j.parent_id if j.parent_id else '-':>6}  {reason}")


def cmd_events(cfg, tail: int, job: int | None) -> None:
    sup = _client(cfg)
    try:
        events = sup.store.events(job_id=job) if job is not None else \
            sup.store.events_after(max(0, sup.store.last_event_id() - tail), tail)
    finally:
        sup.close()
    for e in events[-tail:]:
        data = json.dumps(e["data"], sort_keys=True, default=str)
        print(f"{_when(e['wall'])} #{e['id']:<6} {e['kind']:<22} job={e['job_id'] if e['job_id'] else '-':<5} "
              f"{data[:220]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m foreman.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config file (default: $FOREMAN_HOME/config.json)")
    ap.add_argument("--db", dest="db_path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit-speech", help="Kokoro text to speech")
    s.add_argument("text")
    s.add_argument("--voice")
    s.add_argument("--speed", type=float)
    s.add_argument("--expected-s", type=float)
    m = sub.add_parser("submit-music", help="ACE-Step text to music")
    m.add_argument("prompt")
    m.add_argument("--seconds", type=float, required=True)
    m.add_argument("--lyrics")
    m.add_argument("--seed", type=int)
    m.add_argument("--expected-s", type=float)
    v = sub.add_parser("submit-video", help="LTX-2.5 text to video")
    v.add_argument("prompt")
    v.add_argument("--seconds", type=float, required=True)
    v.add_argument("--width", type=int, default=768)
    v.add_argument("--height", type=int, default=512)
    v.add_argument("--steps", type=int)
    v.add_argument("--seed", type=int)
    v.add_argument("--expected-s", type=float)
    st = sub.add_parser("status", help="supervisor state")
    st.add_argument("--live", action="store_true", help="also read mlx-serve's /api/ps")
    j = sub.add_parser("jobs", help="recent jobs")
    j.add_argument("--all", action="store_true")
    e = sub.add_parser("events", help="the event log")
    e.add_argument("--tail", type=int, default=30)
    e.add_argument("--job", type=int)
    c = sub.add_parser("cancel", help="cancel a queued or running job")
    c.add_argument("job_id", type=int)
    sub.add_parser("clear-safe", help="leave SAFE mode (a human decision)")
    args = ap.parse_args(argv)
    cfg = studio.load_config(args.config, db_path=args.db_path)
    try:
        if args.cmd == "submit-speech":
            _submit(cfg, speech_request(args.text, args.voice, args.speed, args.expected_s))
        elif args.cmd == "submit-music":
            _submit(cfg, music_request(args.prompt, args.seconds, args.lyrics, args.seed, args.expected_s))
        elif args.cmd == "submit-video":
            _submit(cfg, video_request(args.prompt, args.seconds, args.width, args.height, args.steps, args.seed,
                                       args.expected_s))
        elif args.cmd == "status":
            cmd_status(cfg, args.live)
        elif args.cmd == "jobs":
            cmd_jobs(cfg, args.all)
        elif args.cmd == "events":
            cmd_events(cfg, max(1, args.tail), args.job)
        elif args.cmd == "cancel":
            sup = _client(cfg)
            try:
                ok = sup.cancel_job(args.job_id, "cancelled from the CLI")
            finally:
                sup.close()
            print(f"job {args.job_id}: {'cancel recorded (gpud finishes it on its next tick)' if ok else 'not cancellable'}")
            return 0 if ok else 1
        elif args.cmd == "clear-safe":
            sup = _client(cfg)
            try:
                sup.clear_safe()
            finally:
                sup.close()
            print("SAFE mode cleared; every brain gets a fresh set of load attempts")
    except AdmissionError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
