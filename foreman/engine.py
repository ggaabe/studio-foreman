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
    # Why a failed run failed, when the engine knows: "rejected" means the server refused the
    # request itself (an HTTP 4xx for its parameters), a content failure. None: gpud records "crash".
    reason: str | None = None


@dataclass(frozen=True)
class LoadStatus:
    state: str           # loading | ready | failed
    error: str | None = None
    # failed only for now: the engine refused the load because the host lacks free memory (or a
    # model slot) at the moment. gpud keeps a media job waiting instead of failing it.
    deferred: bool = False
    # failed on the client's side only: the request reached the server and was abandoned (gpud
    # aborted it, the answer never came), so the server may still finish the load. gpud keeps its
    # load intent and claims the model if it appears, instead of calling it someone else's.
    in_flight: bool = False


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
    #   Resident footprint of a model as the server reports it (the largest seen since the server
    #   started). For a model the catalog does not know (loaded by another client) it is the only
    #   size gpud has; without it such a model is assumed to fill the host. For a catalog model gpud
    #   plans with the larger of the catalog size and this reading.
    #
    # Optional: def instance_id(self) -> str | None
    #   Identity of the running server process (mlx-serve: its process start time from /metrics).
    #   A change means the server restarted: every model gpud loaded on it is gone, and whatever is
    #   resident now was loaded by someone else. None when unknown (never the last value read).
    #   Without it (or while it reads None) gpud cannot see a restart, so an unexpected
    #   disappearance of one of its models makes it disown the others on that engine.
    #
    # Optional: def loading_models(self) -> set[str]
    #   Models a load is in flight for on the server, by any client. gpud waits for another
    #   client's load instead of sending its own (a second load joins the first, and gpud would
    #   take the copy for its own).
    #
    # Optional: def owned_id(self, model) -> str | None, set_owned_id(self, model, server_id)
    #   For a server that can hold one model under several ids: the id of the copy gpud's load
    #   made, recorded in model_owners and handed back to a restarted adapter.
    #
    # The optional extensions are hints: an exception from one, or a malformed answer, counts as
    # "unknown" (noted once as engine_extension_error), never as an engine outage.
