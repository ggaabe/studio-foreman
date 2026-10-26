# Foreman

A prototype of the GPU-residency supervisor ("gpud") for a Mac Studio running local models.
Its job is to make sure **the deciding brain always comes back**. If the brain has to be
unloaded so a big media model fits, the brain returns when that job ends, whether it succeeded,
failed, hung, crashed, produced garbage, or the supervisor itself died along the way. When it
returns, it knows what to do next. When it cannot return by itself, a human is paged.

Everything here runs in simulation: a virtual clock, fake engines with scripted faults, and a
scripted stand-in for the LLM. It uses only the Python standard library (3.9 or newer; tested on 3.9.6 and 3.12), makes no
network calls, and starts no real model server.

## Run it

```bash
cd foreman
python3 -m unittest discover -s tests -v     # 104 tests (plus subtests), about 20 seconds
python3 -m foreman.demo                       # narrated timeline: render hangs, brain recovers
python3 -m foreman.demo --fault crash         # also: garbage, wedged, none
```

The suite reports `OK`. Two red-team tests originally demanded that gpud crash on an engine or
pager outage; they now assert the opposite. See [Inverted red-team tests](#inverted-red-team-tests).

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
| `tests/test_brain_return.py` | Invariants I1 to I9 (the original 48 tests, unchanged) |
| `tests/test_adversarial.py` | The red team's 37 attacks (two inverted to require no crash; see below) |
| `tests/test_hardening.py` | Regression tests for the hardening, including a seeded chaos test |

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
| I4 | Killing the supervisor at each of 13 named crash points, at random ticks with downtime, mid-render, between "job finished" and "continuation enqueued", and during `recover()` itself loses no job and never duplicates a continuation. A second gpud on the same DB cannot resurrect a terminal job | `I4SupervisorCrashRecovery`, `A6`, `H3` |
| I5 | Primary retried once, then fallback, then page + SAFE. No load loop (a flapping brain counts as a failed load), and the ladder survives restarts. Every way into SAFE pages a human | `I5BrainLoadLadder`, `A3`, `A7`, `P2` |
| I6 | Max attempts leads to quarantine + `human_review`, and a request cannot raise `max_attempts`. Identical resubmissions without `retry_of` are counted. The per-lineage failure budget bounds varied resubmissions. A poisoned brain job is quarantined, and the depth limit ends infinite chains. gpud's own deaths are not charged | `I6Poison`, `A4`, `H5` |
| I7 | The brain is never away longer than `max_brainless_s` when brain loads take their nominal time, even when every chunk of a split batch runs into its hard timeout, even when callers set priorities, and even across wall-clock steps. If loads fail, see the gap below | `I7MaxBrainless`, `A2`, `A5`, `A9` |
| I8 | Autonomous eviction of an idle, busy, or co-resident brain is detected and the brain is reloaded when needed. Models gpud does not know are sized by the adapter (or assumed to fill the host) and unloaded before planning | `I8Eviction`, `A1` foreign model, `H8` |
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

## Mapping to the real engines (NOT IMPLEMENTED)

No real adapter exists yet. A real one implements `foreman.engine.Engine`. The points below are
requirements and open questions. They are not claims about the current mlx-serve v26.10.3 or
oMLX 0.7.2 APIs, which still need to be checked against each server's actual endpoints.

- Every call must have a client-side timeout (a few seconds). gpud calls adapters synchronously
  from `tick()`, so a call that hangs blocks the supervisor. A timeout that raises is handled as
  "engine unreachable".
- `resident_models()`: ask the server which models are loaded. Poll it every tick and never cache
  it, because both servers may evict on their own. A service that gpud stops on purpose (for
  example oMLX after DeepSeek) should be reported as having no models, not raise. Otherwise it is
  declared lost after `engine_lost_s` and a human is paged.
- Optional `model_size_gb(model)`: the footprint of a model the catalog does not know (loaded by
  another client). Without it, such a model is assumed to fill the host.
- `load(model)` / `poll_load(ticket)`: if the server's load call blocks, run it on a worker thread
  and expose a ticket. A load that takes longer than `load_timeout_factor` x `load_s` counts as
  failed.
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
- Memory planning uses catalog sizes. Activation and KV-cache growth are not modeled, and the sizes
  and load times in `policy.py` are estimates, not measurements.
- Batch checkpoints count toward `max_continuation_depth` (default 32), so very long batches need a
  higher limit.
- The validators read simulated metadata. Real ones would need ffprobe-style probing (blackdetect,
  silencedetect), which is outside the standard library.
- Pages are delivered at least once and may repeat after a crash. `human_review` and
  `engine_lost` jobs have no resolve API yet; only `clear_safe()` exists.
