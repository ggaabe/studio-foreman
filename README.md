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
Everything uses only the Python standard library (3.9 or newer; tested on 3.9.6 and 3.12). The
test suite makes no network calls outside 127.0.0.1 and starts no real model server.

## Run it

```bash
cd foreman
python3 -m unittest discover -s tests -v     # 147 tests (plus subtests), about 30 seconds
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
| `foreman/adapters/mlx_serve.py` | `MlxServeEngine`: the `Engine` interface over mlx-serve's HTTP API (loads, SSE runs, cancel by disconnect, fault injection) |
| `foreman/adapters/sse.py` | Incremental SSE reader; streams a huge `complete` event's base64 straight to disk |
| `foreman/httpio.py` | urllib calls with hard timeouts, and a raw-socket streaming client that any thread can close |
| `foreman/media.py` | Stdlib artifact analysis: WAV duration/RMS/silence, rgb8 black/frozen frames, a PNG encoder |
| `foreman/brain_client.py` | `BrainClient`: chat-completion decisions from the resume brief, with sampled frames for vision |
| `foreman/studio.py` | The Mac Studio deployment: catalog, policy, config loading |
| `foreman/daemon.py` | `python3 -m foreman.daemon`: gpud itself (lock, recover, 1 s ticks, JSON logs, SIGTERM) |
| `foreman/cli.py` | `python3 -m foreman.cli`: submit speech/music/video, status, jobs, events, cancel, clear-safe |
| `deploy/`, `scripts/` | LaunchAgent template and install/uninstall scripts (no sudo) |
| `tests/test_brain_return.py` | Invariants I1 to I9 (the original 48 tests, unchanged) |
| `tests/test_adversarial.py` | The red team's 37 attacks (two inverted to require no crash; see below) |
| `tests/test_hardening.py` | Regression tests for the hardening, including a seeded chaos test |
| `tests/test_mlx_serve_adapter.py` | The adapter, brain client, media analysis, CLI and daemon against a local fake mlx-serve (`tests/fake_mlx_serve.py`) |

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

## Running on a Mac with mlx-serve

gpud runs on the same Mac as an already running mlx-serve (v26.10.x, port 11234). It never
starts, stops or restarts that server: on the reference Studio, MLX-Serve.app owns it. It needs
only `/usr/bin/python3` (3.9 or newer) and the model packs under `~/.mlx-serve/models`.

### The Studio deployment (`foreman/studio.py`)

| Role | Model (mlx-serve id) | Plan GB | Notes |
|---|---|---|---|
| primary brain, vision | `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` | 24 | qwen3_5 with a vision tower: reviews get sampled frames |
| fallback brain | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw` | 74 | text only |
| speech | `ddalcu/Kokoro-82M-MLX-Serve` | 1 | `/v1/audio/speech`, WAV |
| music | `ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit` | 8 | `/v1/audio/music-generations`, 10 to 600 s |
| video | `ddalcu/LTX-2.5-MLX-Serve-4bit` | 60 | `/v1/video/generations`, rgb8 frames + pcm audio, 24 fps |

Model names are the ids mlx-serve's `--model-dir` discovery gives the packs (`<org>/<repo>`).
Packs downloaded after the server started are not discovered until a rescan, so the adapter then
loads them by absolute path, which registers them under the directory basename; both names map
back to the catalog. FastH3-Omni (MiniMax-H3 derived; its license excludes US users) is not in
the catalog, and the adapter refuses to load or run anything matching it. Plan sizes are on-disk
size plus activation headroom; they are estimates, not measurements.

The policy reserves 96 GB of the 256 GB (macOS, other inference services that live outside
mlx-serve, and a 40 GB margin), so 160 GB are planned. Models in mlx-serve that are not in the
catalog (anything the user loaded) are **pinned**: they count against memory and model slots and
are never unloaded (`Policy.unload_foreign=False`).

`video_exclusive` (default **on** for this deployment) unloads the brain while LTX renders, so
every render exercises the brain-return path: the review job reloads the vision brain, which
then looks at the frames. The production-friendlier setting is `"video_exclusive": false`: the
27B vision brain fits beside LTX, a review needs no reload, and mlx-serve's GenYield lets a
co-resident brain answer chat between denoise steps (at some cost in render speed).

Six small supervisor features back this deployment. None changes the simulation (its 104
original tests are untouched): the policy and adapter ones are off by default, and no scripted
brain returns `human_review`.

- `Policy.unload_foreign=False` pins every resident model outside the catalog.
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

Each submission carries its baton: speech and music finish with `review_audio` (the vision
brain sees the waveform and the measured WAV statistics) or `recover_audio` / `recover_music`;
video finishes with `review_render` (2 to 4 sampled frames attached) or `recover_render`. The CLI
opens the same SQLite file as a non-leader client (`Supervisor(lead=False)`), so admission rules
apply and it never takes gpud's lock.

State lives in `~/.foreman/`: `foreman.db` (plus `foreman.db.lock`), `logs/gpud.log` (JSON lines,
rotated; every supervisor event is mirrored there), `pages.jsonl` (one line per page to a human)
and `artifacts/<run id>-<process>/`. Audio runs leave `output.wav` and `waveform.png`. Video runs
leave `frames.rgb` (raw rgb8), `video.json` (width, height, fps, frame count, measured stats),
`frame_NNNN.png` samples, `audio.wav`, and `output.mp4` only when `ffmpeg` is on the PATH (the
Studio has none). Artifacts are never deleted automatically; a 5 s 720p clip is about 330 MB raw.

Configuration, later sources winning: defaults, `~/.foreman/config.json` (keys are the `Config`
fields in `foreman/studio.py`, for example `{"video_exclusive": false}`), environment variables
(`FOREMAN_HOME`, `FOREMAN_DB`, `FOREMAN_MLX_URL`, `FOREMAN_MODEL_DIR`, `FOREMAN_VIDEO_EXCLUSIVE`,
`FOREMAN_MIN_HEADROOM_GB`, `FOREMAN_MAX_RESIDENT_MODELS`, `FOREMAN_FAULT_STALL_AFTER`,
`FOREMAN_FAULT_KINDS`, `FOREMAN_PROGRESS_TIMEOUT_S`, ...), then command-line flags.

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
  mlx-serve evicts on its own.
- `load()`: `POST /v1/load-model` blocks until the model is ready, so it runs on a thread
  behind a ticket. Before sending it, the adapter refuses the load (a failed ticket, which moves
  the brain ladder or fails the media job) when free memory minus the model's size would drop
  below 40 GB, or when mlx-serve already holds `--max-resident-models` (3) models. Free memory
  is `vm_stat` free + inactive + speculative pages x page size, and the server's own
  `machine.free_mib`; the smaller wins. gpud's planner also reads `max_resident_models` and frees
  a slot itself, unloading its own idle models (media first), so mlx-serve's LRU never picks a
  victim that Foreman does not own.
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
- A model loaded by absolute path is matched to the catalog by directory basename. Someone else
  loading the same pack under the same basename would be taken for Foreman's.
- If the user's own models fill every mlx-serve slot, gpud cannot load a brain. It records
  `insufficient_slots` and waits rather than unload them, and it does not page for this.
- Plan sizes and load times in `foreman/studio.py` are still estimates (they are generous: the
  measured loads below are 1 to 9 s). The video `expected_s` and the decode budget are now
  calibrated on the Studio, with other inference services sharing the GPU; heavier contention
  than that slows renders further.
- A run that a gpud restart interrupts comes back as `failed(interrupted)` and the brain's retry
  counts as an attempt (only a job the engine never saw is requeued uncharged). Three restarts in
  the middle of the same task's renders quarantine it.
- mlx-serve answers a request that fails validation (for example `steps` outside [1, 100]) with
  HTTP 400 before any SSE event. The adapter fails the run with the server's message, and the
  supervisor records it with the generic reason `crash`; the message is in the brief's detail.
- mlx-serve writes a media stream's headers only when the job reaches its inference thread. A
  request queued behind another client's long generation shows no progress, so it can hit the
  no-progress timeout (at least 120 s) before it starts.
- No MP4 without ffmpeg: the artifacts are raw frames, PNG samples and a WAV.

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
- In simulation the validators read simulated metadata. With mlx-serve they read what
  `foreman.media` measured on the files (sampled pixels and 50 ms audio windows, not a full
  ffprobe-style blackdetect / silencedetect).
- Pages are delivered at least once and may repeat after a crash. `human_review` and
  `engine_lost` jobs have no resolve API yet; only `clear_safe()` exists.
