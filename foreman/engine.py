"""The Engine adapter interface.

gpud talks to every model server (mlx-serve, oMLX) only through this interface. FakeEngine in
sim.py implements it for tests; a real adapter would implement it over the server's API.

There is deliberately no kill() / force-kill method. The strongest automatic action is
terminate(), the graceful SIGTERM analog (launchd restarts the engine afterwards). If that
fails, the supervisor enters SAFE mode and pages a human.

Adapters may raise. gpud treats any exception from an adapter call as an "engine unreachable"
observation for that engine (never as a reason to die): the call is retried on later ticks, the
dead-man switch keeps running on the persisted lease, and an engine that fails every call for
Policy.engine_lost_s is declared lost (its jobs fail, their on_failure continuations fire, and a
human is paged). An adapter for a service that gpud stops on purpose should report it as having
no resident models rather than raising.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RunSpec:
    run_id: str          # "<job id>.<attempt>"; start() must be idempotent on it
    job_id: int
    kind: str
    model: str
    payload: dict
    expected_s: float


@dataclass(frozen=True)
class Poll:
    status: str          # running | done | failed | cancelled | lost
    progress: float = 0.0  # monotonic, 0..1 (e.g. denoise steps done / total)
    artifact: dict | None = None
    error: str | None = None
    log_tail: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LoadStatus:
    state: str           # loading | ready | failed
    error: str | None = None
    # failed only for now: the engine refused the load because the host lacks free memory (or a
    # model slot) at the moment. gpud keeps a media job waiting instead of failing it.
    deferred: bool = False


class Engine(Protocol):
    name: str

    def resident_models(self) -> set[str]:
        """Models actually loaded right now (the engine may evict on its own)."""

    def load(self, model: str) -> str:
        """Begin loading; returns a ticket. Idempotent while that model is loading or loaded."""

    def poll_load(self, ticket: str) -> LoadStatus: ...

    def unload(self, model: str) -> bool:
        """Graceful unload; aborts runs on that model. False if the engine will not comply."""

    def start(self, spec: RunSpec) -> str:
        """Start a run and return a handle. Same run_id returns the same handle."""

    def find(self, run_id: str) -> str | None:
        """Handle for a run the engine still knows about (used after a supervisor restart)."""

    def poll(self, handle: str) -> Poll: ...

    def cancel(self, handle: str) -> None:
        """Graceful cancel request (stop after the current step)."""

    def terminate(self) -> bool:
        """Graceful process stop (SIGTERM analog); launchd restarts it. False if it did not exit."""

    # Optional: def model_size_gb(self, model: str) -> float | None
    #   Resident footprint of a model the catalog does not know (loaded by another client). Without
    #   it gpud assumes such a model fills the host, so it is unloaded before anything is planned.
