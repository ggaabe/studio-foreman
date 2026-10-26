"""Tunable constants and the model catalog.

Model sizes and load times are planning estimates for the simulation, not measurements.
Replace them with numbers measured on the real Mac Studio before trusting the planner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    engine: str            # engine process that hosts it ("mlx-serve" or "omlx")
    size_gb: float         # resident footprint used for memory planning
    load_s: float          # expected cold-load time
    family: str            # brain | video | music | tts | image | llm
    vision: bool = False
    exclusive: bool = False  # must be the only model resident on the host
    evicts_brain: bool = False  # no brain may stay resident while this model runs (other models may)

    @property
    def is_brain(self) -> bool:
        return self.family == "brain"


def _catalog(*specs: ModelSpec) -> dict[str, ModelSpec]:
    return {s.name: s for s in specs}


CATALOG = _catalog(
    ModelSpec("qwen38-flash-next", "mlx-serve", 100, 60, "brain"),          # primary brain (+ n-gram table)
    ModelSpec("qwen36-35b-a3b", "mlx-serve", 21, 15, "brain"),              # fallback brain
    ModelSpec("qwen38-27b", "mlx-serve", 18, 15, "brain", vision=True),     # small vision brain
    ModelSpec("ltx-2.5", "mlx-serve", 150, 40, "video"),
    ModelSpec("ace-step", "mlx-serve", 10, 10, "music"),
    ModelSpec("qwen3-tts", "mlx-serve", 5, 5, "tts"),
    ModelSpec("flux2-klein", "mlx-serve", 14, 10, "image"),
    ModelSpec("deepseek-v4.1-flash-1017", "omlx", 195, 120, "llm", exclusive=True),
)


def number(value) -> float | None:
    """A finite real number, or None. bool and numeric strings are not numbers here: LLM output
    is untrusted, and '8 seconds' must be refused at admission, not crash a validator later. An
    integer too large for a float (JSON allows 10**400) is not a number either."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class Policy:
    host_memory_gb: float = 256.0
    host_reserve_gb: float = 16.0          # macOS, gpud, page cache headroom
    progress_timeout_s: float = 120.0      # PROGRESS_TIMEOUT_S: floor of a lease's no-progress limit
    progress_timeout_fraction: float = 0.4  # ... or this share of the job's expected duration
    progress_timeout_steps: float = 2.0    # ... or this many declared progress intervals
    hard_timeout_factor: float = 2.0       # HARD_TIMEOUT_FACTOR: hard timeout = factor * p90
    p90_factor: float = 1.25               # p90 estimate when a job only gives expected_s
    cancel_grace_s: float = 30.0           # wait after a graceful cancel before unloading
    max_brainless_s: float = 1800.0        # MAX_BRAINLESS_S: longest scheduled brain absence
    brain_return_sla_s: float = 60.0       # BRAIN_RETURN_SLA_S: continuation start deadline (+ load)
    max_attempts: int = 3                  # MAX_ATTEMPTS per logical task (also a ceiling for requests)
    max_continuation_depth: int = 32       # MAX_CONTINUATION_DEPTH: lineage hops before human review
    max_lineage_failures: int = 12         # failed brainless jobs per lineage before human review
    max_submits_per_decision: int = 8      # fan-out limit for one brain decision
    max_interruptions: int = 5             # gpud/engine-side interruptions a job may absorb uncharged
    max_job_faults: int = 3                # gpud exceptions on one job before it is failed
    load_timeout_factor: float = 3.0       # a load slower than factor * load_s counts as failed
    load_attempts_per_model: int = 2       # 2 = "retry once", then move down the ladder
    ladder_cooldown_s: float = 3600.0      # a brain that failed its attempts is skipped this long
    text_ladder: tuple[str, ...] = ("qwen38-flash-next", "qwen36-35b-a3b")  # then page human + SAFE
    vision_brain: str = "qwen38-27b"
    co_resident_brain: str | None = "qwen38-27b"  # kept beside media models when memory allows
    continuation_priority: int = 100       # requests are clamped below this
    min_priority: int = -1000
    brain_job_expected_s: float = 30.0
    tick_slack_s: float = 5.0
    engine_lost_s: float = 120.0           # an engine failing every call this long is 'lost' -> page
    max_clock_step_s: float = 30.0         # wall-clock jump treated as a step (no monotonic source)
    page_backoff_max_s: float = 60.0       # pager retry backoff ceiling
    unload_foreign: bool = True            # False: models outside the catalog are pinned, never unloaded
    # Provenance: gpud records every model it loads (Supervisor, table model_owners). A catalog model
    # resident without gpud having loaded it is someone else's: never unloaded, always counted.
    use_foreign_brains: bool = True        # someone else's resident brain may serve gpud's brain jobs (read-only)
    foreign_models: tuple = ()             # catalog models that are always someone else's: never loaded or unloaded
    idle_media_unload_s: float | None = None  # unload gpud's idle media models after the queue is empty this long
    load_defer_s: float = 15.0             # a media load the engine deferred (free memory) is retried after this,
    load_defer_max_backoff_s: float = 300.0  # ... doubling up to this,
    load_defer_max_s: float = 3600.0       # ... and the job waiting on it fails once the wait exceeds this
    # (also the bound for a media job gpud cannot make room for, because the memory or the model
    # slots it needs are held by models gpud may not unload, or whose model is in foreign_models)
    brain_blocked_page_s: float = 900.0    # no brain this long while gpud cannot bring one back
                                           # (someone else's models hold the memory or slots): page

    @property
    def usable_gb(self) -> float:
        return self.host_memory_gb - self.host_reserve_gb

    def expected_s(self, requires: str, payload: dict) -> float:
        if requires == "brain":
            return number(payload.get("expected_s")) or self.brain_job_expected_s
        value = number(payload.get("expected_s"))
        if value is None or value <= 0:
            raise ValueError(f"expected_s must be a positive number, got {payload.get('expected_s')!r}")
        return value

    def hard_timeout_s(self, requires: str, payload: dict) -> float:
        p90 = number(payload.get("p90_s")) or self.expected_s(requires, payload) * self.p90_factor
        return self.hard_timeout_factor * p90

    def progress_timeout_for(self, requires: str, payload: dict) -> float:
        """Per-job no-progress limit: never below the global floor, scaled with the job's expected
        duration (a 4-step image model reports progress rarely), widened by a declared progress
        interval, and never above the hard timeout."""
        t = max(self.progress_timeout_s, self.progress_timeout_fraction * self.expected_s(requires, payload))
        interval = number(payload.get("progress_interval_s"))
        if interval:
            t = max(t, self.progress_timeout_steps * interval)
        return min(t, self.hard_timeout_s(requires, payload))

    def load_timeout_s(self, spec: ModelSpec) -> float:
        return spec.load_s * self.load_timeout_factor
