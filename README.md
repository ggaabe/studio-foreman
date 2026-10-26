# Foreman

A prototype of the GPU-residency supervisor ("gpud") for a Mac Studio running local models.
Its job is to make sure **the deciding brain always comes back**. If the brain has to be
unloaded so a big media model fits, the brain returns when that job ends, whether it succeeded,
failed, hung, crashed, produced garbage, or the supervisor itself died along the way. When it
returns, it knows what to do next. When it cannot return by itself, a human is paged.

The supervisor core is proven in simulation: a virtual clock, fake engines with scripted faults,
and a scripted stand-in for the LLM. On top of that core, `foreman/adapters/mlx_serve.py` drives a
real [mlx-serve](https://github.com/ddalcu/mlx-serve) server and `foreman/brain_client.py` makes
the LLM decisions through it (see [Running on a Mac with mlx-serve](#running-on-a-mac-with-mlx-serve)).
A [planner](#planner) lets the resident brain propose what to make from owner-written templates;
code validates and prices every proposal, and the owner's verdicts are the only feedback it learns
from. Everything uses only the Python standard library (3.9 or newer; tested on 3.9.6 and 3.12). The
test suite makes no network calls outside 127.0.0.1 and starts no real model server.

## Run it

```bash
cd foreman
python3 -m unittest discover -s tests -v     # 355 tests (plus subtests), about 75 to 100 seconds
python3 -m foreman.demo                       # narrated timeline: render hangs, brain recovers
python3 -m foreman.demo --fault crash         # also: garbage, wedged, none
```

The suite reports `OK`. Two red-team tests originally demanded that gpud crash on an engine or
pager outage; they now assert the opposite, and one subtest of the live-fixes red team pins a
documented gap instead of demanding a fix. See [Inverted red-team tests](#inverted-red-team-tests).

## Layout

| File | What it holds |
|---|---|
| `foreman/supervisor.py` | `Supervisor`: `submit`, `tick`, `recover`, `cancel_job`, `clear_safe`; the instance lock |
| `foreman/store.py` | SQLite (WAL) schema, migrations and data access: jobs, continuations, leases, append-only events, supervisor_state. Every job transition is a compare-and-set |
| `foreman/policy.py` | Constants (`Policy`), the model catalog (sizes and load times are estimates), `number()` |
| `foreman/clock.py` | `SystemClock` (wall time plus `time.monotonic()`) and the supervisor-time rules |
| `foreman/engine.py` | The `Engine` adapter interface a real mlx-serve or oMLX adapter would implement |
| `foreman/validators.py` | Deterministic artifact checks: exists, duration, black frames, silence, dimensions |
| `foreman/sim.py` | `VirtualClock`, `FakeEngine` with fault scripts, `ScriptedBrain`, the `Sim` harness |
| `foreman/demo.py` | `python3 -m foreman.demo` |
| `foreman/adapters/mlx_serve.py` | `MlxServeEngine`: the `Engine` interface over mlx-serve's HTTP API (loads, SSE runs, cancel by disconnect, fault injection) |
| `foreman/adapters/sse.py` | Incremental SSE reader; streams a huge `complete` event's base64 straight to disk |
| `foreman/httpio.py` | urllib calls with hard timeouts, and a raw-socket streaming client that any thread can close |
| `foreman/media.py` | Stdlib artifact analysis: WAV duration/RMS/silence, rgb8 black/frozen frames, a PNG encoder |
| `foreman/brain_client.py` | `BrainClient`: chat-completion decisions from the resume brief, with sampled frames for vision |
| `foreman/studio.py` | The Mac Studio deployment: catalog, policy, config loading |
| `foreman/daemon.py` | `python3 -m foreman.daemon`: gpud itself (lock, recover, 1 s ticks, JSON logs, SIGTERM) |
| `foreman/cli.py` | `python3 -m foreman.cli`: submit speech/music/video, status, jobs, events, cancel, clear-safe; the owner's loop: review, verdict, ship, resolve, templates, planner, report |
| `foreman/planner.py` | `Planner`: tick cadence and backoff inside gpud, the input snapshot and prompt, the strict plan parser, the validator (reason codes in `REASONS`), EV and priority, suspension and the token budget |
| `foreman/templates.py` | Template schema and loader (helpful errors), banned classes, goals.md and the inbox, request building from a proposal |
| `foreman/ledger.py` | The value ledger: planner items, the review queue, verdicts and shipments, Beta posteriors, 7/28-day stats, kill criteria, page resolution |
| `foreman/retention.py` | Raw-frame deletion after the MP4 mux, pruning of rejected and unreviewed items, the disk guard |
| `foreman/report.py` | The Markdown scoreboard behind `cli report` |
| `foreman/text.py` | Text hygiene for untrusted strings: folding (NFKC, invisible characters removed) for hashes and filters, and `printable` for terminal and report output |
| `foreman/media_requests.py` | Speech, music and video request builders (with their batons), shared by the CLI and the planner |
| `examples/` | A generic `goals.md` and four inactive example templates (no real client names) |
| `deploy/`, `scripts/` | LaunchAgent template and install/uninstall scripts (no sudo) |
| `tests/test_brain_return.py` | Invariants I1 to I9 (the original 48 tests, unchanged) |
| `tests/test_adversarial.py` | The red team's 37 attacks (two inverted to require no crash; see below) |
| `tests/test_hardening.py` | Regression tests for the hardening, including a seeded chaos test |
| `tests/test_mlx_serve_adapter.py` | The adapter, brain client, media analysis, CLI and daemon against a local fake mlx-serve (`tests/fake_mlx_serve.py`) |
| `tests/test_planner.py` | P1 to P14: templates, parser, every validator reason, EV ordering, budgets across many ticks, no model loads, backoff, suspension, tokens, idempotency, verdicts and kills, retention, untrusted inbound text, CLI and wiring |
| `tests/test_planner_adversarial.py` | The planner red team's 46 attacks (A to U), all passing unchanged |
| `tests/test_planner_fixes.py` | X1 to X12: regression tests for the design concerns the planner red team raised |
| `tests/test_live_fixes.py` | LV1 to LV13: regressions from the planner's first live run (deferred media loads, retracting a verdict, model provenance, calibrated memory plans, infrastructure failures that do not use caps) |
| `tests/test_live_fixes_adversarial.py` | The red team's 30 attacks on those three fixes (RP provenance, RC calibration, RF failure classes); one subtest adjusted, see below |
| `tests/test_live_fixes_round2.py` | LR1 to LR4: regressions for the answers to that red team and its design concerns |

## How the brain comes back

1. **gpud is CPU-only and always on.** It holds no GPU memory, launchd restarts it, and it is the
   only thing that loads or unloads models. Its memory lives in SQLite, so a fresh process on the
   same file picks up from any committed state.
2. **The baton.** A job that might run while the brain is unloaded (any media or oMLX job) is
   admitted only if it carries both an `on_success` and an `on_failure` continuation. When the job
   ends, exactly one of them becomes a queued brain job with a **resume brief**: goal, attempted
   action (including its continuations, so it can be resubmitted as is), outcome, artifact paths,
   validator results, log tail, and attempt count. Artifacts are kept even when the job failed
   after writing them, or finished during its cancel grace. The brain keeps no state of its own;
   the brief is everything it gets.
3. **Leases (dead-man switch).** Each running job holds a lease, and only *observed* progress
   renews it. No progress for the job's progress timeout, a runtime past `hard_timeout_factor` x
   p90, an engine crash, or a failed validator ends the job. The lease keeps timing even while
   the engine does not answer. For timeouts the sequence is: graceful cancel, wait up to
   `cancel_grace_s`, graceful unload, `terminate()` (SIGTERM) only if the unload is refused,
   record the failure, fire `on_failure`. A cancel the engine rejects (for example a 404 for a
   run it already reaped) skips the grace wait instead of wedging the job. There is no
   force-kill anywhere in the interface. If even SIGTERM fails, gpud pages a human and enters
   SAFE mode.
4. **Brain-return rule.** Continuations get top priority (every request's priority is clamped
   below theirs) and a deadline of `brain_return_sla_s`. When the queue is empty, gpud reloads
   the primary brain ("home"). No brainless stretch is scheduled past `max_brainless_s`: long
   batches are split at admission with a `batch_checkpoint` continuation between chunks, and if
   the brain is already away the scheduler inserts a `brain_checkpoint` (one per absence) and
   dispatches it at once. While the brain is away, the ladder prefers a brain whose worst-case
   load still lands inside the cap. When memory allows, the small vision brain stays loaded next
   to the media model, so reviewing a clip needs no swap.
5. **Brain-load ladder.** Retry the primary once, then fall back to the smaller brain, then page a
   human and enter SAFE mode. A load counts as failed if it times out, errors, or reports "ready"
   while the model is not resident (a flapping model). Aborting a stuck load is escalated like any
   unload (refused, then SIGTERM, then SAFE), so a half-loaded model is never forgotten. Ladder
   state is persisted, so a crash loop cannot reset it.
6. **SAFE mode.** Ladder exhaustion means full SAFE: no loads at all until `clear_safe()`. A
   single wedged engine means *degraded* SAFE: that engine is never touched again and no media
   work is dispatched, but a brain may still be loaded on another engine if it fits beside the
   wedged engine's memory (for example the 21 GB fallback next to a stuck 195 GB DeepSeek).
7. **Poison protection.** Each logical task gets `max_attempts` (a request cannot raise it).
   A resubmission with the same kind, model and payload inside the same lineage counts as a
   retry even without `retry_of`. Each lineage may have at most `max_lineage_failures` failed
   brainless jobs, and each decision at most `max_submits_per_decision` requests. At the limits
   the job is quarantined (or the request refused) and a human is paged. `max_continuation_depth`
   stops endless chains. gpud's own deaths are not charged to a job: a run the engine never saw
   is requeued, up to `max_interruptions`, after which the job is treated as poison.
8. **Recovery.** `recover()` reconciles the DB with what the engines report. A run the engine
   still knows about is adopted under its existing lease (re-fenced to the new leader epoch). A
   run the engine lost becomes `failed(interrupted)` and fires its `on_failure` once. A job gpud
   recorded as starting but the engine never saw is requeued with the same run id. Recording an
   outcome and firing its continuation are separate commits (outbox pattern). A crash between
   them is repaired on the next tick, and the atomic mark-fired step plus the `UNIQUE`
   idempotency key keep the continuation single.

## Nothing kills gpud

The red team showed that any exception escaping `tick()` became a deterministic crash loop,
because the bad input was persisted and `recover()` walked into it again. Now:

- **Every adapter call** goes through one wrapper. An exception, or a malformed result (a poll
  without a status, a non-iterable model list), is an *engine-unreachable observation* for that
  engine. Its models are counted as pinned in memory (last known residency) and cannot be used
  or unloaded. Its jobs keep their lease timing. An engine that fails every call for
  `engine_lost_s` (120 s) is **lost**: its jobs and the queued jobs that need it are failed (so
  their batons fire), a human is paged once per outage, and its runs are remembered as orphans
  whose models are unloaded when it answers again. When it answers, everything resumes on its
  own; no `clear_safe()` is needed.
- **Every job** is handled inside a boundary. An unexpected exception is a fault charged to that
  job; after `max_job_faults` the job is failed with reason `supervisor_error` (a running job is
  still cancelled and unloaded first), so its `on_failure` fires. If even that raises, the job is
  force-quarantined.
- **Every tick phase** is inside a boundary too, and the pager is retried with backoff.
- **LLM output is untrusted.** Admission validates types (numbers must be real numbers, payloads
  objects, `requires` a string, and so on). Each request in a decision is admitted inside its own
  SQLite savepoint, and any exception becomes a recorded rejection. If every request in a
  continuation's decision is rejected, a human is paged rather than letting the task end
  silently.
- **Validators** treat missing or non-numeric metadata as a failed check, never an exception.

## Single instance, fencing and time

- **Lock.** gpud takes an exclusive, non-blocking `flock` on `<db>.lock` when it starts. The
  kernel drops the lock when the process dies, so a crash never leaves a stale lock. A second
  Supervisor on the same file is a passive standby: `tick()` and `recover()` do nothing until it
  can take the lock, and then it recovers before acting. A CLI that only submits work should use
  `Supervisor(..., lead=False)` so it never takes the lock.
- **Leader epoch.** Every `recover()` increments `boot_count`, which serves as the leader epoch.
  At each tick the leader checks that the epoch is still its own. If another instance recovered
  after it (the lock was bypassed, for example on a filesystem where `flock` does nothing), it
  steps down. Without a lock it then stays down for good.
- **Compare-and-set.** Every job state transition is `UPDATE ... WHERE state = <expected>`. A
  stale writer cannot resurrect a job that is already terminal. If a start loses that race, the
  run it started is disowned and cancelled.
- **Fencing tokens.** Each lease records the epoch that owns it. Renewals are
  `UPDATE ... WHERE fence = <my epoch>`, and adoption after a restart re-fences the lease.
- **Supervisor time.** Leases, load timeouts, cancel grace, SLAs, ladder cooldowns and the
  brainless cap all use a persisted, monotonic timeline, not the wall clock. Within a process it
  advances with `clock.monotonic()`. Across restarts it is re-baselined from the persisted
  `(time, wall, monotonic)` triple: by the monotonic delta while that timeline is continuous
  (same boot), otherwise by the wall delta, never negative. Without a monotonic source, a
  wall-clock jump larger than `max_clock_step_s` inside one process is ignored and recorded as a
  `clock_step` event. Events store both supervisor time (`at`) and wall time (`wall`). The
  brainless checkpoint is keyed by a brain-absence counter, not by a timestamp.

## Invariants

`tests/test_brain_return.py` (I1 to I9) plus the attacks in `tests/test_adversarial.py`:

| | Invariant | Tests |
|---|---|---|
| I1 | A brainless job without both continuations, or without the goal and expected duration the brief needs, is rejected, including when the brain submits it. Malformed requests (wrong types, prose numbers, an unconfigured engine) are rejected and recorded, never fatal | `I1Admission`, `A1` malformed decisions, unconfigured engine |
| I2 | After any terminal outcome (success, crash, engine-process crash, hang, slow past hard timeout, garbage, unprobeable output, cancel while running or queued, media load failure, hang or flap, exclusive DeepSeek job, engine lost), exactly one continuation is enqueued, a brain comes back, and the continuation starts within `brain_return_sla_s` + brain load time once a brain can be loaded | `I2ExactlyOnceAndBrainReturn`, `A1`, `A3`, `H7` |
| I3 | No progress or a hard timeout leads to graceful cancel, then unload, then terminate only if needed, and never a force-kill. The job is marked `failed(timeout)`. A slow but progressing job is not killed, including a few-step model (the progress timeout is per job: the largest of the 120 s floor, 40 % of the expected duration, and 2 declared progress intervals, capped at the hard timeout) and including wall-clock steps | `I3LeaseExpiry`, `A1` cancel raises, `A5`, `A10` |
| I4 | Killing the supervisor at each of 14 named crash points, at random ticks with downtime, mid-render, between "job finished" and "continuation enqueued", and during `recover()` itself loses no job and never duplicates a continuation. A second gpud on the same DB cannot resurrect a terminal job | `I4SupervisorCrashRecovery`, `A6`, `H3` |
| I5 | Primary retried once, then fallback, then page + SAFE. No load loop (a flapping brain counts as a failed load), and the ladder survives restarts. Every way into SAFE pages a human | `I5BrainLoadLadder`, `A3`, `A7`, `P2` |
| I6 | Max attempts leads to quarantine + `human_review`, and a request cannot raise `max_attempts`. Identical resubmissions without `retry_of` are counted. The per-lineage failure budget bounds varied resubmissions. A poisoned brain job is quarantined, and the depth limit ends infinite chains. gpud's own deaths are not charged | `I6Poison`, `A4`, `H5` |
| I7 | The brain is never away longer than `max_brainless_s` when brain loads take their nominal time, even when every chunk of a split batch runs into its hard timeout, even when callers set priorities, and even across wall-clock steps. If loads fail, see the gap below | `I7MaxBrainless`, `A2`, `A5`, `A9` |
| I8 | Autonomous eviction of an idle, busy, or co-resident brain is detected and the brain is reloaded when needed. Models gpud does not know are sized by the adapter (or assumed to fill the host) and unloaded before planning (simulation default; the Studio pins them). A catalog model gpud did not load itself is someone else's: never unloaded (not even by restarting its engine), always counted (see [Which models are gpud's](#which-models-are-gpuds)) | `I8Eviction`, `A1` foreign model, `H8`, `LV6` to `LV8`, `RP`, `LR1`, `LR2` |
| I9 | Every resume brief is complete. A stateless brain can rebuild the retry from the brief alone. Artifacts written before a failure, or finished during the cancel grace, are in the brief | `I9ResumeBrief`, `A11`, checks on every continuation in `assert_invariants` |
| I10 | No adapter exception, malformed adapter result, validator error, malformed decision, pager outage or gpud fault on one job escapes `tick()`. Each becomes a recorded outcome | `A1`, `H1`, `H6`, `H9` (seeded chaos with random gpud kills) |

`assert_invariants` runs after most scenarios. It checks that no brainless job is left
non-terminal, that each brainless job has exactly one continuation, that no `continuation_fired`
event is duplicated, that no force-kill call was made, that the planner never overcommitted
memory, and that a brain is resident (or the system is in SAFE mode) at the end.

## Inverted red-team tests

Two of the red team's attacks, `P4Assorted.test_transient_engine_outage_mid_render_recovers_without_rerun`
and `P4Assorted.test_pager_outage_does_not_stop_the_brain_returning`, originally opened with
`assertGreater(len(crashes), 0, "... really did crash gpud")`. That precondition demanded the
behavior the red team's own root-cause finding identified as the defect: an adapter or pager
exception killing the process. The fixed gpud rides out both outages, so those assertions were
inverted to `assertEqual(crashes, [])`. Every other assertion in them is unchanged.
`test_hardening.H1NoCrashOnOutages` covers the same ground independently.

`test_live_fixes_adversarial.RP4RestartGpudCannotSee`, subtest `unchanged`, asserted that gpud
never releases a model another client reloaded inside one tick of a server restart whose identity
reads the same before and after. With a readable identity that does not change, the restart is
indistinguishable from no restart; the only trace is that another of gpud's models disappeared,
which is what a routine LRU eviction looks like. Passing would mean disowning all of gpud's models
on every eviction under a trusted identity, which pins gpud's own media models (memory it then
cannot free: the over-planning this round fixed) until a human or the server removes them. That
subtest now asserts the documented gap (the release happens); the `missing` subtest (no identity
at all), where gpud does disown its other models after such a disappearance, is unchanged and
passes. Every other attack in that file passes as written.

## Mapping to the real engines

The mlx-serve adapter exists (next section); an oMLX adapter does not. The points below are the
requirements any adapter has to meet. How the mlx-serve adapter meets them is described in
[Running on a Mac with mlx-serve](#running-on-a-mac-with-mlx-serve).

- Every call must have a client-side timeout (a few seconds). gpud calls adapters synchronously
  from `tick()`, so a call that hangs blocks the supervisor. A timeout that raises is handled as
  "engine unreachable".
- `resident_models()`: ask the server which models are loaded. Poll it every tick and never cache
  it, because both servers may evict on their own. A service that gpud stops on purpose (for
  example oMLX after DeepSeek) should be reported as having no models, not raise. Otherwise it is
  declared lost after `engine_lost_s` and a human is paged.
- Optional `model_size_gb(model)`: the footprint of a model the catalog does not know (loaded by
  another client). Without it, such a model is assumed to fill the host. A reading larger than the
  host's memory is a glitch and is ignored (`model_size_implausible`).
- Optional `instance_id()`: the server process's identity (a change means it restarted). Without
  it, or while it reads `None`, gpud cannot see a restart (see [Which models are gpud's](#which-models-are-gpuds)).
- Optional `loading_models()`: models a load is in flight for, by any client, so gpud waits for
  another client's load instead of joining it. Optional `owned_id(model)` / `set_owned_id(model,
  id)`: for a server that can hold one model under several ids, the id of gpud's own copy.
- Optional extensions are hints: one that raises or returns garbage counts as "unknown" (noted once
  as `engine_extension_error`) and never marks the engine down.
- `load(model)` / `poll_load(ticket)`: if the server's load call blocks, run it on a worker thread
  and expose a ticket. A load that takes longer than `load_timeout_factor` x `load_s` counts as
  failed. A failed `LoadStatus` with `in_flight` says the request reached the server and was given
  up on client-side (aborted, timed out), so the server may still finish it.
- `unload(model)`: the server's graceful unload, which must also abort an in-flight load. Return
  `False` when it refuses or hangs.
- `start(spec)` / `find(run_id)`: recovery depends on starting being idempotent on `run_id`. If the
  server cannot take a client-supplied id, the adapter must persist a `run_id -> server job id` map
  itself, written before the request is sent.
- `poll(handle)`: progress has to be monotonic and move during long renders (for LTX, denoise steps
  completed out of the total). This is what keeps the lease alive. A job whose steps are far
  apart should declare `payload.progress_interval_s`. Report a run the server no longer knows as
  `lost`.
- `cancel(handle)`: the server's graceful cancel (stop after the current step).
- `terminate()`: a graceful SIGTERM of the engine process via its launchd job, after which
  launchd restarts it. Never SIGKILL. That decision stays with a human, which is why SAFE mode
  exists.
- gpud itself would run as a launchd agent with `KeepAlive`, using a SQLite file on local disk
  (so `flock` works) and `foreman.clock.SystemClock`. `tick()` would be called about once a
  second.

## Running on a Mac with mlx-serve

gpud runs on the same Mac as an already running mlx-serve (v26.10.x, port 11234). It never
starts, stops or restarts that server: on the reference Studio, MLX-Serve.app owns it. It needs
only `/usr/bin/python3` (3.9 or newer) and the model packs under `~/.mlx-serve/models`.

### The Studio deployment (`foreman/studio.py`)

| Role | Model (mlx-serve id) | Measured resident | Plan GB | Measured load | Plan load s | Notes |
|---|---|---|---|---|---|---|
| primary brain, vision | `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` | 18.2 GB (load-model reports 16.95) | 20 | 2 to 6 s, up to 9 | 10 | qwen3_5 with a vision tower: reviews get sampled frames |
| fallback brain | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw` | 75.3 GB | 77 | not measured | 70 | text only |
| speech | `ddalcu/Kokoro-82M-MLX-Serve` | about 0.4 GB | 1 | about 1 s | 3 | `/v1/audio/speech`, WAV |
| music | `ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit` | about 4 GB pack | 6 | 1 to 2 s | 5 | `/v1/audio/music-generations`, 10 to 600 s |
| video | `ddalcu/LTX-2.5-MLX-Serve-4bit` | 33.1 GB, 35.6 GB during a 5 s 1280x704 render | 40 | 3 to 5 s | 10 | `/v1/video/generations`, rgb8 frames + pcm audio, 24 fps |

Model names are the ids mlx-serve's `--model-dir` discovery gives the packs (`<org>/<repo>`).
Packs downloaded after the server started are not discovered until a rescan, so the adapter then
loads them by absolute path, which registers them under the directory basename; both names map
back to the catalog. FastH3-Omni (MiniMax-H3 derived; its license excludes US users) is not in
the catalog, and the adapter refuses to load or run anything matching it.

**Plan sizes are calibrated.** The sizes are what mlx-serve's `/api/ps` reported as `size_vram`
on the Studio (2026-10-26), and for LTX the peak seen during a live render, plus a modest margin;
load times are the measured wall times with room to spare (gpud counts a load as failed after 3x
its plan time). The earlier estimates (brain 24 GB / 25 s, LTX 60 GB / 45 s, Flash-Next 74 GB /
70 s) were too loose: LTX's 60 GB plus the 40 GB headroom made renders wait up to an hour while
88 to 110 GB were free. A model can still be bigger than its plan (a longer clip, another
quantization): gpud and the adapter's preflight both plan with the largest size mlx-serve reported
for the model since the server started whenever that exceeds the catalog number (`model_size_observed`
is logged when it does), and forget those readings when the server restarts. A reading no machine
could hold (more than the host's memory; the adapter's limit is 1 TB) is ignored and logged once
(`model_size_implausible`): one glitch must not keep every job of that model waiting on memory.

The policy reserves 96 GB of the 256 GB (macOS 16 GB, other inference services that live outside
mlx-serve 40 GB, and the `min_headroom_gb` margin, 40 GB by default), so 160 GB are planned: the
27B, LTX and someone else's Flash-Next fit together (20 + 40 + 77 GB). `min_headroom_gb` (config,
or `FOREMAN_MIN_HEADROOM_GB`) sets both the adapter's free-memory preflight and the policy's
reserve; a headroom that leaves less than the 27B brain's 20 GB to plan with (more than 180 GB) is
refused at startup. Models in mlx-serve that gpud did not load (anything the user or another app
loaded, catalog models included) are **pinned**: they count against memory and model slots and are
never unloaded (next section).

### Which models are gpud's

gpud records every model it loads itself, durably, in the `model_owners` table, together with
the mlx-serve server instance it loaded it on (`mlx_serve:process_start_time_seconds` from
`/metrics`, which changes only when the server restarts; `/metrics.json` is the fallback). A
resident model gpud did not load is **someone else's**, even when it is in the catalog: it is
never unloaded or evicted by any gpud path (make room, the model-slot cap, idle release, the
cleanup after a timeout, an aborted ladder load, the engine-lost orphan cleanup), and it always
counts in the memory and slot plan. On the Studio this is MLX-Serve.app's default model, which is
the catalog's fallback brain.

- **Intent first.** Before the load request is sent, gpud commits a `loading` intent. If gpud dies
  with the request in flight, the next gpud claims the model when it appears on the same server
  instance within the load's timeout plus 60 s (`model_claimed`); otherwise the intent expires.
  A load the engine answered as failed (or refused before sending) ends its intent at once, so a
  copy another client loads right after is theirs. A load gpud gave up on after the request
  reached the server (its 3x timeout, then the abort; the adapter reports it `in_flight`) keeps its
  intent from that moment: the server may still finish it, and the model is then gpud's to
  release, not a leak nobody owns.
- **One copy is gpud's.** mlx-serve can hold one catalog model under two ids (the discovered
  `<org>/<repo>` and the basename of a same-named directory someone loaded by path). The adapter
  reports only the copy gpud's own load made under the catalog name (its id is recorded as
  `server_id` in `model_owners` and handed back to the adapter after a gpud restart); any other copy
  is reported under its raw id (the discovered id as `<id> (another copy)`), a model outside the
  catalog: pinned, sized, a slot of its own. If gpud's copy goes, no alias stands in for it, and
  unloads and runs address gpud's id. A load of a model someone else already has resident is
  refused by the adapter (mlx-serve would only answer "ready" for their copy), and gpud waits for
  another client's load in flight instead of joining it (`loading_models`; this needs mlx-serve to
  list the model as `loading` in `/v1/models`, which is assumed, not verified).
- **Ownership ends** when gpud unloads the model, when it observes the model gone
  (`ownership_dropped`), or when the server instance changes (`engine_instance_changed`): after a
  restart nothing resident was loaded by gpud, even a model with the same name. A model that comes
  back without a gpud load is someone else's. While the identity is unknown (no `/metrics`, or a
  read that failed; the adapter never reports the previous server's identity for it), loads are
  recorded without an instance and adopt the next one read, and an unexpected disappearance of one
  of gpud's models makes gpud disown its others on that server: a restart plus someone else's
  reload cannot be ruled out, so they are never unloaded (the safe side; a human is paged if that
  leaves no room for a brain).
- **Never by force, never by restart.** gpud's escalation of a refused unload is `terminate()`
  (SIGTERM, launchd restarts the engine). It is not used on an engine that also holds models gpud
  did not load, or that someone else is loading: that would take them down too. gpud enters SAFE
  for that engine and pages instead. (mlx-serve's adapter refuses terminate anyway.)
- **Someone else's brain** may serve gpud's brain jobs read-only (`use_foreign_brains`, on by
  default; the planner may then plan on it too). gpud never loads a second copy of a model that is
  resident already, so a media job whose model someone else has loaded runs on their copy, read-only;
  a timeout's cleanup then skips the unload (`cleanup: foreign`). With `use_foreign_brains` off, a
  brain someone else holds resident is not gpud's to load either: if that leaves no brain in the
  ladder, gpud enters SAFE and pages.
- **Owner override.** `"foreign_models": [...]` in `config.json` lists catalog models that are
  always someone else's: gpud never loads, unloads or claims them (it may still use a resident one
  as a brain, or run a media job on a resident one). Entries may be written the way mlx-serve shows
  them (the basename, with or without `:latest`, or the pack's path) and are normalized to the
  catalog name; a name that matches no catalog model stops gpud at startup. The planner never
  routes work to such a model, and a job for one that nobody has resident waits (nothing is
  unloaded for it, the queue behind it moves) and fails after an hour. Leaving the fallback brain
  out of the ladder this way means a failing primary ends in SAFE mode instead.
- **Waiting is bounded.** A media job that cannot run because the memory or the model slots it
  needs are held by models gpud may not unload waits without holding up the jobs behind it
  (`media_blocked`), and fails with `load_failed` after `load_defer_max_s` (1 h), so its baton fires
  and a planned item stops counting toward stock. gpud checks slots before it unloads anything,
  so it never gives up its own brain for a model that could not get a slot anyway. When no brain
  can come back for `brain_blocked_page_s` (15 min) because of such models (or another client's
  load that never finishes), a human is paged once per absence (`brain_blocked`).
- **Upgrade.** A database from before provenance is seeded once from its event log: a resident
  model whose latest event is gpud's own `model_loaded` is gpud's, any other is someone else's
  (`ownership_seeded`). `status` marks each resident model `gpud` or `not gpud`.

By default the 27B vision brain stays resident while LTX renders: it fits beside LTX (about
18 + 36 GB), a review starts with no reload, and mlx-serve's GenYield lets it answer chat
between denoise steps. Measured on the Studio, a 5 s 1280x704 render took 76 s with the brain
resident against 68-82 s with it unloaded, and the review started 0 s after the render.
`"video_exclusive": true` unloads the brain for every render instead, which exercises the
brain-return path each time (that is how the live scenarios below were run) or frees memory on
a smaller machine.

Seven small supervisor features back this deployment. None changes the simulation (its 104
original tests are untouched): the policy and adapter ones are off by default, and no scripted
brain returns `human_review` or defers a load.

- `Policy.unload_foreign=False` pins every resident model outside the catalog. Catalog models gpud
  did not load are pinned whatever this says (provenance, above).
- `ModelSpec.evicts_brain` (set on LTX by `video_exclusive`) makes the planner unload a brain
  even when it would fit, and never plan a co-resident brain for that model.
- An adapter may expose `max_resident_models`. The planner then unloads gpud's own idle models
  (media first, biggest first) until the target fits under that count.
- A brain decision with `verdict: "human_review"` pages a person (`kind: human_review`).
- `Policy.idle_media_unload_s` (300 s here): once the queue has been empty that long, gpud
  unloads its own idle media models one per tick, so only the brain (and the user's models) stay
  in mlx-serve's slots and memory.
- An adapter exception with `pending = True` (the mlx-serve adapter's `UnloadPending`) means
  "still working on it": the caller retries next tick and the engine stays healthy.
- A load whose `LoadStatus` is `deferred` (the adapter's memory or slot refusal) keeps a media
  job queued: the model is retried after `Policy.load_defer_s` (15 s), doubling up to
  `load_defer_max_backoff_s` (5 min), nothing is unloaded for it in between, other models' jobs
  run meanwhile, and only a wait over `load_defer_max_s` (1 h) fails the job it was for, one job
  per such wait. A brain's deferred load is a failed load, so the ladder still moves. Found live:
  a memory dip below LTX's headroom failed six queued renders in 40 s, and a seventh two minutes
  later (see [Planner live on the Studio](#planner-live-on-the-studio-2026-10-26)).

### Run it

```bash
cd ~/src/studio-foreman
python3 -m foreman.daemon --once                    # smoke test: recover, one tick, exit
python3 -m foreman.daemon                           # gpud in the foreground (Ctrl-C stops it)

python3 -m foreman.cli submit-speech "Hello from the Studio"
python3 -m foreman.cli submit-music "warm lo-fi piano, rain on the window" --seconds 30
python3 -m foreman.cli submit-video "a red fox trotting through fresh snow" --seconds 5 --width 768 --height 512
python3 -m foreman.cli status --live                # mode, residency, jobs; --live also reads /api/ps
python3 -m foreman.cli jobs                         # brain verdicts and failure reasons per job
python3 -m foreman.cli events --tail 40             # the append-only event log
python3 -m foreman.cli cancel 12                    # graceful: gpud closes the stream, then unloads
python3 -m foreman.cli clear-safe                   # a human leaving SAFE mode
```

The owner's side of the planner (`review`, `verdict`, `ship`, `resolve`, `templates`, `planner`,
`report`) is described in [Planner](#planner).

Each submission carries its baton: speech and music finish with `review_audio` (the vision
brain sees the waveform and the measured WAV statistics) or `recover_audio` / `recover_music`;
video finishes with `review_render` (2 to 4 sampled frames attached) or `recover_render`. The CLI
opens the same SQLite file as a non-leader client (`Supervisor(lead=False)`), so admission rules
apply and it never takes gpud's lock.

State lives in `~/.foreman/`: `foreman.db` (plus `foreman.db.lock`), `logs/gpud.log` (JSON lines,
rotated; every supervisor event is mirrored there), `pages.jsonl` (one line per page to a human)
and `artifacts/<run id>-<process>/`. The planner adds `templates/`, `goals.md`, `inbox/` and
`reports/` (see [Planner](#planner)). Audio runs leave `output.wav` and `waveform.png`. Video runs
leave `video.json` (width, height, fps, frame count, measured stats), `frame_NNNN.png` samples,
`audio.wav`, and `output.mp4` when ffmpeg is found (see Known gaps for the `ffmpeg` setting). The
raw rgb8 frames (`frames.rgb`, about 330 MB for a 5 s 704p clip) are deleted once the MP4 mux
succeeds, unless `keep_raw_frames` is set; without ffmpeg, or when the mux fails, they stay. Other
artifacts are deleted only for planner items, on the retention schedule in [Planner](#retention).

Configuration, later sources winning: defaults, `~/.foreman/config.json` (keys are the `Config`
fields in `foreman/studio.py`, for example `{"video_exclusive": false}`), environment variables
(`FOREMAN_HOME`, `FOREMAN_DB`, `FOREMAN_MLX_URL`, `FOREMAN_MODEL_DIR`, `FOREMAN_VIDEO_EXCLUSIVE`,
`FOREMAN_MIN_HEADROOM_GB`, `FOREMAN_MAX_RESIDENT_MODELS`, `FOREMAN_FAULT_STALL_AFTER`,
`FOREMAN_FAULT_KINDS`, `FOREMAN_PROGRESS_TIMEOUT_S`, `FOREMAN_KEEP_RAW_FRAMES`, `FOREMAN_PLANNER`,
`FOREMAN_PLANNER_INTERVAL_S`, ...), then command-line flags. Provenance has two keys:
`foreign_models` (catalog models that are always someone else's) and `use_foreign_brains`
(default true); `min_headroom_gb` (default 40) is the free memory kept beside every load.

The adapter refuses to load a pack whose directory still holds partial download files
(`*.partial`, `*.partial.parts`, ...): MLX-Serve.app downloads in place, and on the Studio the
fallback brain was still downloading while gpud ran.

### As a LaunchAgent

`deploy/com.github.ggaabe.foreman.gpud.plist` is a user LaunchAgent template: `KeepAlive` true
(gpud holds no GPU memory, so restarting it however it exited strands nothing),
`ThrottleInterval` 30, `ExitTimeOut` 60, `ProcessType` Background, stdout and stderr under
`~/.foreman/logs`. Nothing installs it automatically.

```bash
scripts/install-launchagent.sh --dry-run   # print the rendered plist
scripts/install-launchagent.sh             # render, plutil -lint, launchctl bootstrap gui/$(id -u)
scripts/uninstall-launchagent.sh           # launchctl bootout, remove the plist (keeps ~/.foreman)
```

Neither script uses sudo. On SIGTERM gpud finishes its tick, closes its HTTP streams and exits.
mlx-serve aborts a run whose client disconnected, so a restarted gpud finds those runs gone,
records them as `failed(interrupted)` and fires their `on_failure` batons. A watchdog thread exits
the process (status 70) if one tick ever runs longer than `watchdog_s` (300 s), and launchd
restarts it.

### How the adapter maps the Engine interface onto mlx-serve

Checked against the v26.10.3 source and, read-only, against the live server:

- **Every call made from `tick()` has a client-side timeout.** Residency is one
  `GET /api/ps` (5 s). Loads, runs and unloads run on worker threads. Streaming uses a
  raw-socket client that reads with a 1 s socket timeout, so it notices a cancel and any thread
  can close it (`http.client` is unusable after one read timeout).
- `resident_models()`: `GET /api/ps` lists READY models (names end in `:latest`; `expires_at`
  is ignored). Sizes come from `size_vram` and feed `model_size_gb()`. Read every tick, because
  mlx-serve evicts on its own. Each catalog model is reported once, for the copy gpud loaded (or
  the only copy); a second copy under another id is reported under its raw id
  ([Which models are gpud's](#which-models-are-gpuds)).
- `instance_id()`: `mlx_serve:process_start_time_seconds` from `/metrics` (`/metrics.json` as a
  fallback); a server without either is not asked again for a minute, and reads `None` meanwhile,
  never the identity of the server that was running before. `loading_models()`: `/v1/models`
  entries whose `state` is `loading`.
- **Your default model stays yours.** An mlx-serve started without `--model` promotes the latest
  chat-capable load to its default, so loading the brain would silently capture every request
  that names no model (your apps, the "mlx-serve" alias). Before each load the adapter notes the
  default (the first entry of `/v1/models`, if it is chat-capable). If the load took it over
  from a model Foreman does not own, and that model is still resident, the adapter puts it back
  with `POST /v1/load-model {"model": <yours>, "default": true}`. That pins it, so later loads
  cannot take it again. Turn this off with `preserve_default=False`.
- `load()`: `POST /v1/load-model` blocks until the model is ready, so it runs on a thread
  behind a ticket. Before sending it, the adapter refuses the load when free memory minus the
  model's size would drop below 40 GB, or when mlx-serve already holds `--max-resident-models`
  (3) models. Such a refusal says "not now", so its ticket is marked `deferred`: a brain's
  deferred load is a failed load (the ladder moves to a brain that fits), but a media job keeps
  waiting in the queue and gpud tries the load again after 15 s, doubling up to 5 min, without
  unloading anything in between; only a wait over 1 h fails the job (`Policy.load_defer_*`). An
  incomplete pack or an unreadable memory probe is refused outright (a failed load). Free memory
  is `vm_stat` free + inactive + speculative pages x page size, and the server's own
  `machine.free_mib`; the smaller wins. Slots are counted per server model, so two copies of one
  catalog model hold two. gpud's planner also reads `max_resident_models` and frees a slot itself,
  unloading its own idle models (media first), so mlx-serve's LRU never picks a victim that
  Foreman does not own. A model someone else already has resident is never "loaded" (that
  load-model would only answer ready for their copy), and a request that reached the server and
  then got no answer (aborted by gpud, timed out, connection broken) is reported `in_flight`.
- `unload()`: `POST /v1/unload-model` is idempotent, but it waits for in-flight generations on
  that model, so the adapter first closes its own streams on it. A drain that takes longer than
  8 s raises `UnloadPending` (gpud notes `engine_busy` and retries on the next tick). Only a refusal, or a drain longer than 600 s,
  returns False. It refuses outright to unload a model outside the catalog. An empty or
  `mlx-serve` id would unload the server's *default* model, so one is never sent.
- `start()`: a worker thread checks `/api/ps` for the exact id first. An unknown id would make
  mlx-serve **silently answer with its default model** (verified live: a speech request for an
  unregistered Kokoro came back with "Target model does not support this modality" from the
  user's chat model). Then it POSTs the media request with `"stream": true`. Only whitelisted
  payload fields are forwarded (`input`/`voice` for speech, `prompt`/`duration_seconds`/`lyrics`
  for music, `prompt`/`num_frames`/`width`/`height`/`steps`/`seed` for video). `goal`,
  `expected_s` and other Foreman fields stay in Foreman. Brain runs call the brain client instead.
- `poll()`: every SSE `progress` event (and `: keepalive` comment) strictly increases the
  reported progress, which is what renews the lease. Step fractions map into [0, 0.9], and a
  restarted or repeated counter still moves it a little. LTX goes silent between
  `Decoding video` and `complete` (VAE decode plus the base64 transfer: 17 s for 5 s at 1280x704,
  35 s for 10 s, measured). For that one stage the adapter reports a time-based heartbeat from
  0.95 to 0.99 over a budget of 0.6 s per megapixel-frame (at least 60 s, about 4x the measured
  time), then stops. A healthy decode keeps its lease, and a decode that never ends still trips
  the dead-man switch.
- The `complete` event (base64 WAV, or base64 rgb8 frames plus `pcm_s16le` audio, about 440 MB
  of text for 5 s at 720p) is parsed as a stream. Small events are `json.loads`-ed. A line past
  256 KB is parsed incrementally as a flat JSON object, and its `data` / `audio_data` strings are
  base64-decoded straight into files. Then `foreman.media` measures the files and the
  validators check those measurements, never the engine's claims: duration, black-frame ratio,
  frozen-frame ratio, silence ratio.
- `error` events and HTTP errors fail the run with the server's message. A stream that ends
  without a terminal event fails it, and is reported `lost` if `/health` no longer answers.
- `cancel()`: mlx-serve has no cancel endpoint; closing the connection makes it abort at the
  next step boundary (not during the VAE decode). The run reports `cancelled` at once.
- `find()` / `poll()` only know runs this process started. After a gpud restart the old stream
  is gone and mlx-serve has aborted the run, so the old handle polls `lost` and the supervisor
  records it as interrupted. A `starting` job is requeued uncharged, as before.
- `terminate()` always returns False. mlx-serve belongs to MLX-Serve.app, so the supervisor's
  escalation ends in SAFE mode and a page. Nothing is ever force-killed.

The brain client (`foreman/brain_client.py`) streams `POST /v1/chat/completions` over a
connection the run can cancel. It sends `enable_thinking: false`,
`response_format: {"type": "json_object"}` and the resume brief, and reports token progress. The
model answers a small schema: `accept`, `retry` (with whitelisted changes such as a new prompt
or seed), `give_up` or `human_review`. The client builds the actual resubmission from the brief
itself, as a `retry_of` the source job, refused once attempts are used up. Malformed output, an
unknown verdict or an impossible retry becomes `human_review`, which the supervisor now turns into
a page. `batch_checkpoint` and `brain_checkpoint` are handled deterministically without the LLM.

### Fault injection: the dead-man lease against the real engine

```bash
FOREMAN_FAULT_STALL_AFTER=3 FOREMAN_FAULT_KINDS=compose_music python3 -m foreman.daemon
```

A media run then stops reading the stream and stops reporting progress after 3 progress events
(runs that emit none, like Kokoro, are unaffected). The lease expires after the job's
no-progress timeout (at least 120 s). gpud cancels by closing the connection, so mlx-serve
aborts at its next step boundary. gpud then unloads the model, records `failed(timeout)` and
fires `on_failure`, so the brain comes back with the brief. The engine is never harmed: it only
sees a client that went quiet and then hung up. Do not leave the variables set in a LaunchAgent.

### Live on the Studio (2026-10-26)

Every scenario below ran against the real mlx-serve 26.10.3 on the M5 Ultra, with
`video_exclusive` on, the user's own model resident and pinned, and Darkbloom serving on the
same GPU. Times are request to artifact on disk; brain absence is `brain_departed` to
`brain_returned`.

| Scenario | What happened |
|---|---|
| Speech (Kokoro) | Load 1 s, render 3 s, WAV validated (4.8 s, 27 % silence). The vision brain reviewed the waveform PNG and accepted |
| Music (ACE-Step, 12 and 15 s) | Load 1 to 2 s, render 2 to 4 s, duration and silence validated, accepted |
| Video, brain unloaded | 2 s 512x320: 12 s. 5 s 1280x704: 70 to 82 s. 10 s 1280x704 (the 768 MB raw cap): 177 s, an ~870 MB `complete` event streamed to disk. Brain away 20 s, 80 to 90 s and 186 s. The brain reviewed 3 sampled frames |
| Rejected request (`steps` 500) | HTTP 400 from mlx-serve. `on_failure` fired once, the brain read the brief and retried with `steps` 50 (attempt 2, rendered). Its review saw a motorboat where the prompt asked for a paper boat and asked for a human (page). No loop |
| Dead-man lease (fault injection) | The run stopped reporting after 2 events; the lease expired 60 s later (FOREMAN_PROGRESS_TIMEOUT_S=60); gpud closed the stream; mlx-serve logged `generation cancelled — client disconnected`; the unload drained in 5 to 12 s; `on_failure` fired once; the brain was back 2 to 3 s later. The brain retried twice, then the task was quarantined at 3/3 and a human paged |
| gpud SIGTERM mid-render | Both by hand and under the LaunchAgent (launchd restarted gpud in 0.2 s): the next gpud recorded the run `failed(interrupted)` once, fired `on_failure` once and had the brain back 3 to 8 s after starting. The retry rendered and was accepted |

Loads measured: 27B brain 2 to 9 s, LTX 3 to 4.5 s, ACE-Step 1 to 2 s, Kokoro 1 s. Idle gpud
uses about 90 MB of RAM and well under 1 % of a core, and records nothing while idle.

The live runs found and fixed: a pack still downloading must not be loaded; a slow but real
shot was flagged 100 % frozen (threshold recalibrated, motion stats added); a draining unload
marked the engine down every tick; the video time and decode budgets were 4 to 20 times too
loose; idle media models held mlx-serve's slots forever; reinstalling the LaunchAgent could
leave gpud stopped.

### Known gaps of the mlx-serve adapter

- A run lives only as long as the gpud process that started it. Any gpud restart, including
  a planned one, aborts in-flight renders; they come back as `interrupted` failures for the
  brain to retry.
- The memory and slot checks are point-in-time. Another client can load a model between the
  check and the load, and mlx-serve may then evict by its own LRU rule. The next tick sees the
  eviction (`model_evicted`) and the brain is reloaded when needed.
- An unload that waits behind a generation is retried each tick. The adapter's `UnloadPending`
  carries `pending = True`, so the supervisor records one `engine_busy` note per wait instead of
  marking the engine down (live: about 12 s while LTX reached its next step boundary).
- A model loaded by absolute path is matched to the catalog by directory basename. That only
  names it: whether it is Foreman's is decided by provenance, and of two copies under different ids
  only the one gpud loaded is reported as the catalog model.
- Provenance cannot tell two loads of the same id apart. If another client loads a model while
  gpud's own load intent for it is open (the request in flight, gpud just restarted mid-load, or a
  load gpud gave up on that may still complete), gpud takes that copy for its own; the same goes
  for a second gpud on the same server whose load joins the first, when mlx-serve does not list
  the model as `loading` in `/v1/models` (that state is assumed, not verified against the server).
  If the owner loads the very id gpud loaded, it is a no-op for mlx-serve and gpud cannot see it:
  gpud may later release that model when idle. If the server restarts between gpud's once-a-tick
  instance check and a load request, gpud treats its own load as someone else's: it never unloads
  it (the safe side).
- Without a readable identity (no `/metrics`, or while it does not answer) a restart is invisible.
  gpud then disowns its other models on that server whenever one of its models disappears
  unexpectedly (the safe side), at the price of pinning its own models after a plain eviction. An
  identity that reads the same across a restart (a broken `/metrics`) is trusted: a restart plus
  another client's reload inside one tick is then taken for gpud's own model.
- If the user's own models fill every mlx-serve slot, gpud cannot load a brain. It records
  `insufficient_slots` and waits rather than unload them, and pages a human once the brain has been
  away for `brain_blocked_page_s` (15 min).
- Plan sizes and load times in `foreman/studio.py` are calibrated on the Studio (table above);
  Flash-Next's load time was not measured and keeps its 70 s estimate. The video `expected_s` and
  the decode budget are calibrated too, with other inference services sharing the GPU; heavier
  contention than that slows renders further.
- A run that a gpud restart interrupts comes back as `failed(interrupted)` and the brain's retry
  counts as an attempt (only a job the engine never saw is requeued uncharged). Three restarts in
  the middle of the same task's renders quarantine it.
- mlx-serve answers a request that fails validation (for example `steps` outside [1, 100]) with
  HTTP 400 before any SSE event. The adapter reports it as `Poll.reason = "rejected"` and the
  supervisor records reason `rejected` (a content failure for the ledger); the message is in the
  brief's detail. 401, 403, 405, 407 (the caller or the endpoint was refused: an auth proxy, an API
  key, another server version), 404, 408, 409, 423, 429 and the "does not support this modality"
  answer (a request that reached mlx-serve's default model) are not rejections: they stay `crash`.
- mlx-serve writes a media stream's headers only when the job reaches its inference thread. A
  request queued behind another client's long generation shows no progress, so it can hit the
  no-progress timeout (at least 120 s) before it starts.
- Without ffmpeg there is no MP4, only raw frames, PNG samples and a WAV. `ffmpeg` in the config
  (or `FOREMAN_FFMPEG`) sets the binary; `auto` checks the PATH, `$FOREMAN_HOME/bin` and Homebrew,
  because launchd gives gpud a minimal PATH. A no-sudo install: `pip3 install --user
  imageio-ffmpeg` and link its binary to `~/.foreman/bin/ffmpeg`. Muxing uses Apple's
  `h264_videotoolbox` encoder and falls back to `libx264`.

## Planner

gpud's scheduler decides **when and where** work runs. The planner, an LLM on whatever brain is
already resident, only proposes **what** to make, and only as instances of templates the owner
wrote. Code decides whether a proposal is admitted, what it is worth and where it queues. The two
loops share nothing but the database: the planner cannot load a model, start a job, set a
priority, raise a cap or add a template, and an admitted proposal is an ordinary gpud job with
its review and recover batons. Idle is a correct outcome; the prompt says so, and the scoreboard
reports the idle share without trying to raise it.

The planner is on by default (`"planner": false` in `config.json` or `FOREMAN_PLANNER=0` turns it
off) and does nothing until an active template exists in `$FOREMAN_HOME/templates/`.

### How it decides

1. **When a tick runs.** Every `interval_s` (30 min); sooner on events: a media job finished, a
   verdict or shipment was recorded, a template was unpaused, a page was resolved, or the review
   queue dropped below 10 (event ticks at most every 5 min); and on `planner plan-now`. After the
   third tick in a row that admits nothing the interval doubles, and it doubles again with each
   further empty tick, up to 4 h; one admission resets it to 30 min. A fresh database ticks 60 s
   after gpud starts.
2. **When it does not.** A due tick waits while the owner paused the planner, while it is
   suspended, in SAFE mode, while the GPU is busy (a load or a job is running), while a
   continuation is queued, and **while no brain is resident: the planner never causes a model
   load.** The plan job is a brain job marked `resident_only`: the scheduler starts it on the
   resident brain or not at all, and one still queued after 5 min is cancelled. Only the
   planner's tick creates plan jobs (key `plan-tick:<n>`): gpud's admission refuses the `plan`
   kind from any request or brain decision (a review or recover decision cannot start a planning
   round past the pause, the cadence, the token budget or the suspension), the scheduler never
   loads a model for a `plan` job whatever its payload says, and a plan job without the
   planner's key is never applied. A tick is used up
   without asking the brain when no template is active and unpaused, when the disk guard holds,
   or when the day's token budget is spent. Each skip reason is logged once when it starts, so an
   idle planner adds nothing to the event log.
3. **What it sees.** Every tick rebuilds its input from the database and files; no conversation
   is kept. `<goals>` (goals.md), `<templates>` (the active ones: consumers, value, review
   minutes, param bounds, P(ship), caps left this week, stock against max_stock and two weeks of
   use), `<recent_outputs>` (the last 30 briefs, to avoid repeats), `<ledger>` (7 and 28 days per
   template: admitted, produced, brain-accepted, accepted, rejected, shipped, dollars, review
   minutes, GPU-hours), `<state>` (local time and weekday, review items and minutes left today,
   GPU-hours and the speculative and filler shares), then one `<untrusted>` block per inbox note.
   Data is escaped so no text can close or open a tag, and the inbox file name in the `source`
   attribute is escaped too (quotes included), so a file name cannot add an attribute of its own.
4. **What it may answer.** One JSON object, nothing else:
   `{"proposals": [{"template_id", "consumer", "brief", "params", "deadline"?, "value_adj"?,
   "rationale"?}], "new_template_ideas": [up to 3 strings]}`. Any other top-level key, a non-list
   `proposals`, more than ten times `max_proposals` entries in it, prose around the JSON (a code
   fence is tolerated), anything the JSON parser chokes on (30,000 nested brackets raise a
   RecursionError, not a ValueError) or an empty reply makes the whole tick invalid. Any other
   key inside a proposal rejects that proposal (`unknown_field`). Each proposal is validated
   under its own exception boundary: whatever it contains (a 400-digit integer, a year-1
   deadline in local time), it can only reject itself (`malformed_proposal` at worst), never the
   rest of the tick. Ideas are kept for the owner with every URL (in any form the brief filter
   knows) and control character removed, and shown in the report; they never become templates.
5. **Validation (code).** Each proposal is checked in order and rejected with the first reason
   code that applies (`foreman.planner.REASONS`): unknown, inactive or paused template; no
   consumer, a generic one ("anyone", "youtube", ...) or one the template does not list; no brief
   (a brief of only invisible characters is none), a brief over the template's `brief_max_chars`,
   a brief with control characters (terminal escapes, carriage returns), a brief with a URL
   (a scheme, `www.`, an e-mail address, an IP address, localhost, a host followed by a path or
   port such as `acme.edu/x` or `bit.ly/abc`, or a bare host under a common top-level domain), or
   one in a banned class;
   a param the template does not declare, of the wrong type or outside its bounds or whitelist;
   a deadline that is not a date or has passed; a `value_adj` that is not a number; no model of
   the template's family, or a commercial template routed to a model off the commercial-use list
   (`studio.COMMERCIAL_OK`); a content hash already proposed; EV of zero or less; more than 20
   proposals in one tick. Briefs, consumers and the filters are compared on folded text
   (`foreman.text`): NFKC, so fullwidth letters are plain ones, with invisible characters (zero
   width spaces and joiners, the byte-order mark, soft hyphens, bidi controls, fillers) removed,
   and the banned-class and URL filters also try the spelling in which an invisible character is
   a space. Invisible characters are not stored in the brief.
6. **Value (code, never the LLM).**
   `EV = value_usd x P(ship) x urgency x value_adj - $2/min x review_minutes - $0.25/h x GPU time`.
   P(ship) is the mean of Beta(prior.alpha + accepts, prior.beta + rejects): only the owner's
   verdicts move it, never the brain's review and never the planner. Urgency is 2 for a deadline
   within a day, 1.5 within 3 days, 1.25 within 7, else 1. `value_adj` is clamped to 0.5..2. GPU
   time is the media job's expected duration plus its review.
7. **Budgets, in EV order.** The valid proposals are sorted by EV and admitted while they fit:
   the template's `per_week` cap (items admitted in the last 7 days), `max_stock` and
   `two_week_use` (items queued, awaiting review or accepted but not yet shipped), at most 15
   items queued or awaiting the owner, at most 30 review minutes created per day, speculative
   work at most 20 % and filler at most 15 % of the week's GPU time (measured, plus the estimates
   of work still queued). A template flagged `hard_deadline` whose proposal is due within 7 days
   takes one of 3 reserved review slots first, outside the 15 items and the day's minutes; the
   reserved slots hold at most 3 such items at a time and 3 a day, so deadlines the LLM writes
   lift the budget by those 3 items and no more. Once they are taken, hard-deadline work competes
   for the normal budget. The disk guard is checked again here, counting what the items still in
   flight and the ones admitted earlier in the same tick may write (a render's raw frames are
   about 170 MB for 5 s at 768x512), because the disk reading is cached for 10 minutes. Each
   admitted proposal is a gpud job whose priority comes from its EV inside the
   band -99..-1: behind the owner's own CLI submissions and the brain's retries (priority 0), and
   far behind continuations (100). The caps and the day's review minutes count items queued or in
   flight, items produced, and items that failed for **content** reasons; an item that failed for
   **infrastructure** reasons does not count, within a daily budget (next section).
8. **Errors.** Three invalid ticks in a row (unparseable or schema-violating output, a reply
   whose application raised (`plan_error`), or a plan job that timed out or failed while its
   brain stayed resident) suspend the planner for 1 h and page a human (`planner_suspended`).
   gpud keeps running the queue meanwhile; the suspension ends by itself, or with
   `planner resume`. A plan job that was cancelled, interrupted by a gpud restart or lost with
   its engine does not count; neither does one whose brain was evicted mid-plan (memory
   pressure, an idle TTL), after which the next tick comes at the usual cadence. An interrupted
   or unreachable one is retried 2 min later, the delay doubling with each such failure in a
   row. A reply whose application raised still pays its tokens. The token budget is 2M per day,
   estimated as characters / 4 of the prompt plus the reply; a tick starts only if its prompt
   plus the longest reply it allows (3000 tokens) still fits, and a reply is charged at most that
   reservation (the engine is asked for no more; a longer one is logged as
   `plan_reply_over_limit`), so the day's budget is never overrun. Days, for the token budget and
   the review minutes, are those of supervisor time, so a wall-clock step does not hand out a
   second budget for the same day. If the owner pauses the planner while a plan job runs, its
   reply is discarded (`plan_discarded`) and nothing is admitted.

### The review queue and the owner's verdicts

An admitted item follows its gpud lineage (the job, the brain's retries, the brain's review).
When the media succeeded and the brain's review accepted it or asked for a human, the item
enters the **review queue**; any other ending marks it failed with the reason. The owner then
gives a verdict, which is the only thing that teaches the planner anything:

```bash
python3 -m foreman.cli review [--all]                     # items awaiting a verdict: brief, brain note, MP4/WAV/PNG paths
python3 -m foreman.cli verdict 41 accept --minutes 3      # or reject; --note TEXT; minutes default to the template's estimate
python3 -m foreman.cli verdict 41 retract                 # undo a mistaken verdict (not after ship or pruning)
python3 -m foreman.cli ship 41 --usd 60                   # you used it (usd defaults to the template's value_usd)
python3 -m foreman.cli resolve                            # list open pages; resolve 12 13 --note "old scenario tests"
python3 -m foreman.cli templates                          # stats, caps, stock, P(ship), paused; templates pause|unpause ID
python3 -m foreman.cli planner status                     # or pause [--note] | resume | plan-now
python3 -m foreman.cli report [--days 7] [--out PATH]     # Markdown scoreboard, default $FOREMAN_HOME/reports/
```

`review`, `resolve`, `templates`, `jobs` and `status` print every untrusted string (briefs, brain
notes, consumers, page reasons, job kinds a brain decision chose) through `foreman.text.printable`:
control characters, line breaks and invisible or bidi formatting characters are shown as
escapes, so a brief carrying an OSC 52 clipboard write, a carriage return or a cursor move cannot
overwrite or forge what the owner reads before a verdict. The report escapes the same way.
`--minutes` and `--usd` refuse NaN and infinity.

A verdict on an item the brain flagged `human_review` also resolves that page. `verdict JOB
retract` puts an accepted (not yet shipped) or rejected (not yet pruned) item back in the review
queue; P(ship) and the kill criteria are computed from the items, so they forget the verdict, and
a `verdict_retracted` event follows the original one in the log. The scoreboard
lists, per template, what was admitted, produced, passed by the brain, accepted, rejected and
shipped, with dollars, review minutes, GPU-hours, P(ship) and failures split
infrastructure/content (and how many infrastructure failures did not count); then GPU-hours and
the idle share, the five biggest GPU sinks nobody used (items still awaiting a verdict, or still
in flight, are not sinks yet), rejections by reason code, paused templates and the planner's
ideas.

**Kill criteria (code).** A template is paused, logged as `template_paused` and listed in the
report, when its acceptance is below 25 % after at least 20 verdicts, when nothing it produced
was accepted or shipped in 30 days, or when value per review minute stayed below $2 over 2 weeks
(shipped value, plus `value_usd` for items accepted but not shipped yet, since work is often
shipped in batches). They are checked every minute inside gpud and after each verdict.
`templates unpause ID` brings a template back and restarts the history the criteria judge.

**A paused template stops producing**, whether the owner or a kill criterion paused it. Its
queued media jobs are cancelled before the scheduler can start them (the recover baton still
fires, as for any terminal job, and its retry is refused); an item whose media finishes after
the pause is marked failed instead of entering the review queue. A job already running is left
to finish. None of this pages anyone: the pause was deliberate.

**Planned lineages only retry.** Inside a planned item's lineage, a review or recover decision
may only resubmit the planned job: the same kind and model, the same payload apart from the seed,
the timing estimates and the template's own tunable params inside its bounds (voice and speed for
speech, bpm and instrumental for music, steps for video), and at most `max_attempts` (3) media
jobs in all. A new kind of job, other words, another model or a longer clip would be work that
never went through a template, its caps, the review budget or P(ship).

### Infrastructure failures do not use caps

When a planned item's lineage ends without output for review, the ledger classifies the failure
from its jobs (`ledger.failure_class`):

| Class | Failures | Counts against caps and review minutes |
|---|---|---|
| infra | the load failed, or was refused for memory until its deferral expired, or gpud could not make room for it beside models it may not unload for an hour (`load_failed`); the engine was unreachable or lost (`engine_unreachable`, `engine_lost`, `engine_error`, `engine_unconfigured`); a gpud restart interrupted it (`interrupted`); gpud raised on it (`supervisor_error`); the run crashed or stalled server-side once (`crash`, `timeout`; also an HTTP 5xx or 401/403/407 answer, and an SSE `error` event); gpud cancelled it (`cancelled` with `cancelled_by: gpud`; no gpud path does this today: a drain or restart shows up as `interrupted`); the brain's review of good output could not run | no, within the template's daily and weekly budgets |
| content | a validator failed (black, frozen or silent output: `validator`), also on output that finished during a timeout's cancel grace; the server rejected the request's parameters with a 4xx (`rejected`); a crash or timeout that happened again in the same lineage (the brain's retry of the same request failed the engine the same way); the brain's review turned the output down, or was itself refused for its request (`rejected`); the owner (or a paused template) cancelled it; anything not listed above | yes |

A lineage with any content failure is content. Weekly caps (`per_week`), `max_stock` and
`two_week_use` (which count only items queued, awaiting review or accepted anyway) and the day's
review minutes leave an infrastructure failure out; its GPU time still counts toward the
speculative and filler shares, for the time it actually ran (a load that failed ran nothing).

**Bounded.** At most `infra_budget_per_day` infrastructure failures per template per day, and at
most `infra_budget_per_week` in any 7 days (planner options; 0, the default, means the template's
own `per_week` for both, so a rolling week admits at most per_week counted items plus per_week
uncounted ones), go uncounted; beyond that they count again, so a persistently failing environment
fills the week's cap instead of recycling it. Each planned item also keeps gpud's retry limits (at
most `max_attempts` media jobs per lineage). Every failed item records `fail_class`, whether it
`counted`, and why (`count_note`, also on the `item_failed` event); `review --all` prints it.

**The same brief again.** An item that failed for infrastructure reasons and did not count
produced nothing, so its brief may be proposed again: `<recent_outputs>` marks it `may_repeat`,
and admitting it again retires the old row's content hash (`<hash>~<item id>`, history kept) and
gives the new job its own idempotency key. Any other earlier item still makes the brief a
`duplicate`.

**Restoring slots after an upgrade.** Items the previous version settled carry no
classification. gpud's next upkeep classifies them from their lineages, oldest failure first, with
the same rule and budget (`item_classified`). On the Studio this hands back the six ad-clip slots
the live run's load cascade (jobs 70 to 75) used up before the memory fix: no database edit.

### Templates and goals

Templates are JSON files in `$FOREMAN_HOME/templates/` (one per file), written by the owner and
read-only to everything else. A template that fails validation is not loaded; `templates` and
gpud's log name the file and every problem, with "did you mean" hints for misspelt keys, so a
typo cannot silently weaken a limit. `examples/templates/` has four generic ones, all inactive:
copy one, name your real consumer, then set `active` to true.

```json
{
  "id": "narration.weekly-update.v1",
  "kind": "speak",
  "family": "tts",
  "consumer": "weekly team update video",
  "revenue_line": "internal_communications",
  "value_usd": 15,
  "prior": {"alpha": 8, "beta": 2},
  "caps": {"per_week": 2, "max_stock": 2, "two_week_use": 2},
  "params": {
    "voice": {"allowed": ["af_heart", "am_michael"], "default": "af_heart"},
    "speed": {"min": 0.85, "max": 1.15, "default": 1.0}
  },
  "style_prefix": "",
  "brief_max_chars": 900,
  "flags": {"speculative": false, "filler": false, "commercial": false, "private": true},
  "review_minutes": 2,
  "active": false,
  "notes": "The brief is the exact script to read."
}
```

| Key | Meaning |
|---|---|
| `id` | 2 to 64 characters of `a-z 0-9 . _ -` |
| `kind`, `family` | `speak`/`tts`, `compose_music`/`music` or `render_video`/`video`; `model` optionally pins a catalog model of that family |
| `consumer` | Who uses the output: a string or a list. A proposal must name one of them exactly. Generic audiences are refused |
| `revenue_line`, `value_usd` | What the work is for, and what one accepted item is worth |
| `prior` | Beta prior on P(ship), for example 0.8 (8, 2) for committed client work, 0.5 for pitch samples, 0.3 for ad variants, 0.05 (1, 19) for speculative media |
| `caps` | `per_week`, `max_stock` (integers) and `two_week_use` (expected use in two weeks) |
| `params` | What a proposal may vary, each with bounds: `speak` voice, speed, seed; `compose_music` seconds, seed, instrumental, bpm; `render_video` seconds, size (`WxH` whitelist), steps, seed. Bounds must lie inside what the engine accepts; music and video need a default length (and size) |
| `style_prefix` | Prepended to the brief to form the model prompt |
| `brief_max_chars` | Longest brief a proposal may carry (default 600, at most 2000) |
| `flags` | `speculative` and `filler` (GPU shares), `commercial` (license gate), `hard_deadline` (reserved review slots), `private` (recorded; Foreman has no cloud lane, so nothing leaves the Mac anyway) |
| `review_minutes` | The owner's minutes to review one item: its cost in EV and the daily review budget |
| `active`, `notes` | Only active templates are offered to the planner; notes are shown to it |

`$FOREMAN_HOME/goals.md` is the owner's plain-words statement of what matters this month
(`examples/goals.md` is a generic starting point). `$FOREMAN_HOME/inbox/` holds inbound notes
(`.txt`, `.md`, `.json`; the newest 10, 2000 characters each), for example a client's brief. They
reach the prompt only as `<untrusted>` data.

### Safety rules

- **Templates are the only source of work.** Nothing in a reply, the goals or the inbox can add a
  template, activate one, or change a cap, prior, value or bound: those come only from the
  owner's files, and the validator checks every proposal against them.
- **Banned outright.** No template may exist, and no brief may be admitted, for platform-payout
  content farms, uncommissioned media with no named consumer, mass SEO pages, AI-to-AI
  promotion, autonomous outreach, or anything outward-facing. A template with keys such as
  `publish_to`, `upload`, `send` or `webhook` does not load. Foreman makes files for a human to
  review; nothing in it posts, uploads, sends or spends.
- **Untrusted inbound text** is escaped, wrapped and labelled as data; it can tell the planner
  what someone needs, never what it is allowed to do. On the way out it is escaped again: the
  CLI and the report never emit a control character from a brief, a note or a file name.
- **No model loads, no outward actions, no spending**; one GPU job at a time, as before.
- **License gate.** A template flagged `commercial` is admitted only on a model listed in
  `studio.COMMERCIAL_OK` (Kokoro, ACE-Step, LTX-2.5 here). Re-check that list when a model or its
  license changes (LTX-2.x needs a paid license above its revenue threshold).
- **Idempotent.** Each tick is a job keyed `plan-tick:<n>`; each admitted proposal is keyed by
  its content hash (template, normalised brief, merged params), unique in the ledger and as the
  job's idempotency key. The brief is normalised by folding (NFKC, invisible characters removed,
  case folded, spacing collapsed) and number params are canonical floats, so copies that differ
  only by a zero-width space, a byte-order mark, a soft hyphen, fullwidth letters or `10` versus
  `10.0` are one item. A crash while a plan is applied rolls the whole application back, and
  the re-run admits each proposal once without asking the brain again.

### Retention

- After a successful MP4 mux the adapter deletes `frames.rgb` and keeps `output.mp4`, the sampled
  PNGs, `audio.wav` and `video.json` (which records the deletion), unless `keep_raw_frames` is
  true (`FOREMAN_KEEP_RAW_FRAMES=1`). A failed or missing mux keeps the raw frames, and so does
  an MP4 whose top-level boxes are truncated or lack `ftyp`, `moov` or `mdat`
  (`retention.mp4_complete`; ffmpeg exiting 0 is not enough).
- The run directories of rejected items are pruned 7 days after the verdict. Items that waited 14
  days in the review queue expire and are pruned; failed items are pruned 14 days after they
  ended. Deletion is confined to directories strictly inside the artifacts root (resolved, never
  a symlink), so a path in the database cannot point it anywhere else.
- **Disk guard.** The planner admits nothing, and logs why, while the internal disk has less than
  300 GB free or the artifacts directory holds more than `artifacts_cap_gb` (150 GB). A probe that
  fails, or gives anything but two finite non-negative numbers (a NaN compares false with every
  threshold), blocks too. The reading is cached for 10 minutes; admission counts what the work in
  flight and the work admitted earlier in the same tick may write against it.

Planner settings go in `config.json` under `planner_options` (fields of
`foreman.planner.PlannerConfig`), for example
`{"planner_options": {"interval_s": 3600, "token_budget": 1000000, "artifacts_cap_gb": 100}}`;
unknown or malformed options stop gpud at startup with the list of known ones.

### Planner invariants

| | Invariant | Tests |
|---|---|---|
| PL1 | Only active, unpaused, owner-written templates produce work. Inbound text, goals and replies cannot add templates or change caps; a plan decision's `submit` is ignored; no request or brain decision can submit a plan job; a paused template's queued work does not start and its retries are refused; inside a planned lineage the brain may only retry the planned job | `P1`, `P4`, `P13`, adversarial `E`, `R`, `X3`, `X4`, `X5` |
| PL2 | Planning never causes a model load: no resident brain means no tick, a `plan` job never loads a model, and the load/unload sequence of a scenario is identical with and without the planner | `P6`, adversarial `L`, `R` |
| PL3 | Every rejected proposal is recorded with a reason code from `REASONS`, in `plan_rejections` and the event log; a proposal can only reject itself, never the rest of its tick | `P3`, `assert_logged`, adversarial `A`, `I` |
| PL4 | Weekly caps, stock, the review budget (with reserved hard-deadline slots) and the speculative and filler shares hold across many ticks; when nothing qualifies, nothing runs. Infrastructure failures do not use the caps or the review minutes, within per-template daily and weekly budgets; content failures do, and so does a crash or timeout that repeats in a lineage | `P5`, `LV10`, `LV11`, `RF`, `LR3` |
| PL5 | Admission is in EV order, EV <= 0 is never admitted, and planned work queues behind the owner's own submissions | `P4` |
| PL6 | Backoff doubles after 3 empty ticks up to the cap and resets on an admission; 3 invalid ticks (a reply that crashes its own application included) suspend and page while the queue keeps running; an engine-side loss is not an invalid tick; the daily token budget holds, whatever the reply's length and whatever the wall clock does | `P7`, `P8`, `P9`, adversarial `A`, `P`, `X1`, `X2`, `X6`, `X10` |
| PL7 | No duplicate job from a re-run tick, a crash inside the application, a fresh gpud, invisible or fullwidth characters, or numerically equal params | `P10`, adversarial `B`, `K` |
| PL8 | Only human verdicts move P(ship); the kill criteria pause a template | `P11` |
| PL9 | Raw frames are deleted only after a successful mux of a complete MP4; pruning follows the schedule and stays inside the artifacts root; the disk guard blocks admission, also on an unusable reading, and counts what admitted work may write | `P12`, adversarial `D`, `F`, `X7`, `X8` |
| PL10 | One bad template file never takes the other templates, the kill criteria or the pruning down; untrusted text never reaches the owner's terminal or report as control characters | adversarial `G`, `T`, `X12` |

`tests/test_planner_adversarial.py` is the red team's (46 attacks, all passing unchanged);
`tests/test_planner_fixes.py` covers the design concerns it raised (`X1` to `X12`).

### Known gaps of the planner

- The tests prove that whatever the brain answers is parsed, validated and priced strictly, not
  that its proposals are good. Their quality depends on the templates, the goals and the model.
- The banned-class check on briefs is a keyword filter on folded text. It catches the plain
  phrasings, also behind invisible or fullwidth characters, but not a determined paraphrase or a
  homoglyph (a Cyrillic letter that looks Latin); the real guard is that templates are the
  owner's and that nothing in Foreman can act outside the Mac. The URL filter likewise knows the
  usual forms; a bare host name under an unusual top-level domain with no path is not caught.
- A deadline is whatever the LLM writes. It raises urgency (up to 2x EV) and, on a template
  flagged `hard_deadline`, opens the reserved review slots; the slots are bounded (3 at a time,
  3 a day) and the prompt says to give a deadline only when the goals or an inbound note state
  one, but nothing checks it against them.
- Work the owner submits by hand keeps gpud's usual retry baton: a review or recover decision in
  that lineage may still submit other jobs (bounded by the fan-out, depth and lineage-failure
  limits), outside the templates and the ledger. Only planned lineages are restricted to retries.
- GPU shares count every GPU job of the week (brain reviews and plan ticks included), measured
  for finished work and estimated for queued work. On a fresh system with no other GPU use, any
  speculative or filler item would be 100 % of the week, so none is admitted until committed
  work has run.
- The daily review budget counts the estimated minutes of items admitted that day, not the
  minutes the owner actually spent (those are recorded with the verdict and used by the kill
  criteria and the report).
- The plan job runs at continuation priority, so a continuation created while it runs waits for
  it (up to its 300 s hard timeout).
- Only planner items are in the ledger. Work submitted by hand is outside the review queue,
  P(ship) and the kill criteria.
- Deadlines and days are local time (the days of supervisor time, so a wall-clock step does not
  move them, but a time-zone change does). A template file edited while gpud runs is picked up
  within a minute (by name, size, mtime, ctime and inode); one that stops validating drops out
  and its error is logged.
- The tests use `ScriptedPlanner` and `FakeEngine` (the adapter test `FakeMlxServe`); the live
  run below is the only check against the real Qwen3.8-27B.
- The failure classification is by reason code and lineage. A single `timeout` or `crash` is taken
  for the environment's (a GPU fault and a request that crashes the engine look the same once); a
  second one in the lineage counts. A brain review that crashed or timed out on the output (the
  vision brain choking on its frames) is taken for the environment's too; only a refused review
  request (`rejected`) counts.
- Proposals are admitted in EV order, so a speculative proposal is measured against the week's
  GPU time before the cheaper committed work admitted later in the same tick is counted; it may
  be refused for its share in one tick and admitted in the next.
- Someone else's resident brain (the Studio's MLX-Serve.app default) can serve planning and
  reviews read-only (`use_foreign_brains`); turn that off to keep gpud's jobs on its own brains.

### Planner live on the Studio (2026-10-26)

Five owner templates (pitch-pack video clips and music beds for a prospective client, ad-test
clips and music beds for the owner's own service, and an internal calibration render), all
commercial, on the 27B vision brain that stays resident (the fallback brain, loaded by another
client, and other inference services shared the host).

| Tick | What happened |
|---|---|
| 1 (plan-now) | 68 s on the resident 27B, 1879 prompt tokens. 15 proposals: 12 admitted in EV order (1 pitch clip at EV $94, 2 pitch songs at $16, 6 ad clips at $5, 3 ad songs at $2.5), 2 pitch clips refused for `speculative_share` (22 % > 20 %) |
| between ticks | The first clip rendered in 76 s, its raw frames were deleted after the MP4 mux, and the brain's review accepted it from 4 sampled frames; five songs rendered and were accepted. Then free memory beside the host's other tenants dropped to about 96 GB, below LTX's 60 + 40 GB, and before the fix every queued clip failed its load in turn, six in 40 s (now the jobs wait: see the deferred loads above) |
| 2 (job finished) | 1 pitch clip admitted, 1 refused for its share; it failed the same way |
| 3, 4 (after the fix) | The third pitch clip rendered and passed the brain's review; with an inbox note asking for it, two calibration renders were admitted, rendered in 8 s each and passed |
| 5 to 8 | Every weekly cap used: the brain proposed nothing, or one clip refused as a `duplicate` of an earlier brief. Interval 30 -> 60 -> 120 min after the third and fourth empty tick |

Two calibration items took test verdicts (accept and ship, reject): P(ship) went from 0.80 to
0.82 to 0.75. Every other item waits for the owner. Eight plan jobs started on the resident
brain with 0 s wait; no model was loaded for any of them, and no brain was loaded at all during
the run. Each tick cost 2 to 68 s of brain time and 1.9 to 2.8k prompt tokens (23k tokens for the
day). Renders kept only the MP4, PNG samples, WAV and video.json: 0.5 to 3 MB per clip instead of
about 330 MB of raw frames.

The run left three issues, fixed since (`LV6` to `LV13`): the cascade's failed loads used up the
ad clips' weekly cap (they are infrastructure failures now and stop counting on gpud's first
upkeep after the upgrade); LTX's 60 GB plan kept renders waiting while 88 to 110 GB were free (40
GB now, from the measured 33 to 36 GB); and the fallback brain someone else had loaded counted as
gpud's (provenance now records what gpud loads). A red team then attacked the three fixes
(`tests/test_live_fixes_adversarial.py`, 30 attacks); its findings are fixed and covered by
`LR1` to `LR4`: same-named copies of a model under two server ids, loads gpud gave up on that the
server finished, restarts gpud could not identify, a standby taking over, two gpuds on one server,
owner overrides written the way mlx-serve shows them, no brain gpud may use, implausible sizes,
an unbounded headroom, garbage that finished during a timeout's cancel, and auth refusals counted
as content.

## Known gaps

- **Cap overshoot when brain loads fail.** The brainless budget reserves one nominal brain load
  (60 s). While the brain is away, the ladder prefers a brain whose worst-case load still fits
  inside the cap. If loads fail anyway, the absence can exceed `max_brainless_s` by the load
  timeouts of the failing attempts made after the reserve is used up. In the worst case every
  remaining ladder attempt hangs: about 385 s over the cap for a text continuation (fallback
  twice at 45 s, then primary twice at 180 s, less the ~65 s reserve) and about 475 s for one
  that needs the vision brain (which is tried twice first). After that the ladder is exhausted
  and gpud pages a human and enters SAFE.
- One GPU job runs at a time. Even when the vision brain is co-resident, it does not review a clip
  while the next render runs. When several continuations are created together (fan-in), each
  waits for the previous one, so later ones can miss the per-continuation SLA (recorded as
  `brain_return_sla_missed`). The brain itself does come back in time.
- The brain-return deadline counts from when the GPU is free. A continuation created while another
  job holds the GPU waits for that job. The wait is still bounded by that job's lease and
  `max_brainless_s`, or by `engine_lost_s` if the engine stops answering.
- `max_brainless_s` is enforced at admission even on a host where the vision brain could stay
  co-resident. A single render whose hard timeout exceeds the budget (`expected_s` above about
  666 s, or about 634 s for DeepSeek) cannot be admitted unless it has `items` to split. There is
  no override for a human-approved long job.
- A memory reading for an engine that does not answer is its last known residency. If the
  process actually died and freed its memory, gpud stays conservative and may refuse loads on
  other engines until the engine answers again or is declared lost.
- Degraded SAFE brings only a brain back. Media work waits for a human. Engine loss does not
  resubmit anything when the engine returns: the brain is told through `on_failure` and decides.
- When a decision is rejected, a human is paged (for continuations). The brain does not get a
  feedback job carrying the error.
- Logical-task identity without `retry_of` is a hash of `(kind, requires, payload)` within one
  lineage. A brain that changes the payload on every resubmission is bounded only by
  `max_lineage_failures` (12), not by `max_attempts`.
- `flock` is advisory and local. On a filesystem where it does nothing, the leader-epoch check
  is the only protection, and an instance can still finish the tick it is in when it is
  superseded. Compare-and-set transitions and fencing keep that from corrupting state, but an
  engine call can be repeated.
- Without a monotonic source, a long legitimate stall inside one process (over
  `max_clock_step_s`) is treated as a clock step, so up to that much time is not counted, and a
  backward wall step during downtime means the downtime is not counted. Use `SystemClock`. On
  macOS `time.monotonic()` does not advance during system sleep, and neither do the engines.
- Memory planning uses catalog sizes, or the largest size an engine reported for a model since it
  started when that is bigger. Activation and KV-cache growth beyond what was already observed are
  not modeled. The Studio's sizes and load times are measured (`studio.py`); the simulation's in
  `policy.py` are estimates.
- Batch checkpoints count toward `max_continuation_depth` (default 32), so very long batches need a
  higher limit.
- In simulation the validators read simulated metadata. With mlx-serve they read what
  `foreman.media` measured on the files (sampled pixels and 50 ms audio windows, not a full
  ffprobe-style blackdetect / silencedetect).
- Pages are delivered at least once and may repeat after a crash. `cli resolve` closes a page
  (`human_review`, `engine_lost`, `planner_suspended`, ...) and records the note; it does not
  undo or retry anything, and SAFE mode still needs `clear-safe`.
