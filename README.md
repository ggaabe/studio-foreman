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
from. [Phone approvals](#phone-approvals) put the review queue and the pages on the owner's phone
through a Telegram bot, as a separate service that cannot affect scheduling. [Ads](#ads) turn an
owner template into a finished vertical ad (character, voice, lip-sync, B-roll, captions, music,
end card) through a chain of gpud jobs, with an on-demand ComfyUI engine beside mlx-serve.
Everything uses only
the Python standard library (3.9 or newer; tested on 3.9.6 and 3.12). The
test suite makes no network calls outside 127.0.0.1 and starts no real model server.

## Run it

```bash
cd foreman
python3 -m unittest discover -s tests -v     # 927 tests (plus subtests), about 15 minutes
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
| `foreman/phone.py` | `python3 -m foreman.phone`: phone approvals over Telegram, its own process (Bot API client over urllib, hand-written multipart, secrets, pairing, signed buttons, the outbox, the `cli phone` commands) |
| `foreman/phone_config.py` | `PhoneConfig` (`phone_options`), apart so gpud validates config.json without importing the phone code |
| `foreman/text.py` | Text hygiene for untrusted strings: folding (NFKC, invisible characters removed) for hashes and filters, and `printable` for terminal and report output |
| `foreman/media_requests.py` | Speech, music and video request builders (with their batons), shared by the CLI and the planner |
| `foreman/ads.py` | Ads: `AdsConfig`, the template's ad block, the root request, the lineage check, the script validator, numbers, word timing and cuts |
| `foreman/ad_pipeline.py` | `AdPipeline`: the dispatcher (`ad_next`, `ad_recover`, `ad_review`), the facts, assembly and QA steps, take scoring, the manifest, `enqueue` and the CLI's views |
| `foreman/assembly.py` | ffmpeg assembly and probes: segments, ASS captions, end card, ducking, loudnorm, the size-capped final encode, fonts, black/frozen/loudness checks |
| `foreman/characters.py` | Character bibles: validation, import from a folder, approval |
| `foreman/paths.py` | Path confinement for the files engines read and write |
| `foreman/adapters/mlx_ops.py` | Multi-request jobs on mlx-serve: speech (Kokoro, Qwen3-TTS clone), transcription, images, the full video request |
| `foreman/adapters/comfyui.py` | `ComfyUIEngine`: ComfyUI started on demand, graph templates, uploads, outputs, interrupt, /free, stop |
| `foreman/adapters/local.py` | `LocalEngine`: CPU work (an ad's facts, assembly, QA) under gpud's leases |
| `foreman/h3core.py` | The h3 talk chain's pure logic: the template's h3 block, the brain's beat plan, what the voice is told, beat seconds, the H3 prompt, the judge of a take, retakes |
| `foreman/h3/` | The h3 media worker (numpy + Pillow, run as a subprocess under `ads_options.media_python`, never imported by the core): probe, edit, audio, captions, graphics |
| `examples/` | A generic `goals.md` and four inactive example templates (no real client names); `examples/ads/`: generic ad templates (an LTX mascot ad, an h3 presenter ad), character bibles, facts files, ComfyUI graph templates (the format, and an H3 Ref2VA take graph) and config |
| `deploy/`, `scripts/` | LaunchAgent templates (gpud, phone) and install/uninstall scripts (no sudo) |
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
| `tests/test_live_integration.py` | LI1 to LI2: regressions from verifying those fixes on the Studio (the owner's default is a catalog model; seeded models get their server id) |
| `tests/test_ads_engines.py` | G1 to G4: the mlx-serve ops, the ComfyUI engine (a fake ComfyUI process, `tests/fake_comfyui.py`), the local engine, solo_media, a continuation's expected_s, engine checks, failure classes |
| `tests/test_ads_pipeline.py` | P1 to P7: the ad template and the planner, the script validator, timing, characters, facts, assembly with real ffmpeg, the dispatcher |
| `tests/test_ads_e2e.py` | E1 to E4: gpud on the fakes runs whole ads: the happy path, failures, retention, ComfyUI stills between mlx-serve media models |
| `tests/test_h3_core.py` | H1 to H11: the h3 chain's pure logic: the h3 block, the beat plan, spellings and readings, seconds, the prompt, alignment, the judge, retakes, the number fix, what may reach the prompt, the picture and the speech gate, a scan that no owner data is in the repo (its terms come from a private file named by `FOREMAN_PRIVATE_TERMS`; skipped without it), and H12, the second review's fixes |
| `tests/test_h3_pipeline.py` | S1 to S7: the h3 chain in the dispatcher on synthetic files (a fake media worker): the facts preflight and its failure classes, every round's step (a take whose audio does not decode, one of another size, one heard saying a claim), the final gate's safe re-edit, captions that every beat moves, finish's file checks, the gate record, retention, process groups, ComfyUI's environment and refusals, the examples |
| `tests/test_h3_worker.py`, `tests/test_h3_graphics.py` | The media worker's probe and edit, and its graphics, under the local python with numpy and Pillow (skipped without them): a punch-in inside the beat's crop, word times inside their beat, the caption font alone for libass, a worker whose parent dies |
| `tests/test_h3_e2e.py` | H1 to H4: gpud runs whole h3 ads on the fakes (a fake ComfyUI process rendering talking takes, the fake ASR hearing them by their tones) with the real media worker and ffmpeg: the happy path, retakes and a safe re-edit, retention, the failure classes, and the seams between the parts |
| `tests/test_phone.py` | PH1 to PH12: phone approvals against a local fake Bot API (`tests/fake_telegram.py`): pairing, exactly-once sends, media paths, captions, buttons through the ledger, stale and forged buttons, notes, commands, unauthorized updates, backoff, offsets, the token's secrecy, pages, digest, CLI, LaunchAgent |

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

The ads red team (`tests/test_ads_adversarial_robustness.py`, `tests/test_ads_adversarial_compliance.py`,
96 tests) passes as written: none of its assertions was changed (one fixture gained an audio gate
record, below).

The audio gate and reason tags red team (`tests/test_gate_tags_adversarial.py`, 41 tests) passes as
written: none of its tests was changed, including the two whose causes were already known (a price
read "nine fifty", and the music-masking check comparing word error rates as the task asked rather
than lost words only). Two older tests changed with its fixes:
`test_ads_audio_gate.G6Review.test_without_a_gate_record_the_brain_cannot_accept` asserted that a
review with no gate record goes to `human_review`, which still put an ad nobody heard in front of
the owner (one of the red team's design concerns). It is now
`test_without_a_gate_record_or_a_checked_mix_the_ad_never_reaches_the_owner`: such an ad is given
up without asking the brain, and so is one whose final mix was never checked.
`test_ads_adversarial_compliance.R8Injection.test_the_facts_cannot_close_the_untrusted_block_in_the_review_prompt`
built a finished ad without a gate record, so the review now gives up before writing the prompt
the test inspects; its fixture gained a passed `gate.json`, and its assertion is unchanged.

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

One of the implementation's own tests changed meaning after the phone red team:
`test_phone.PH1SetupAndPairing.test_wrong_codes_count_and_burn_the_code` asserted that five wrong
pairing codes burn the code for everyone. `test_phone_adversarial.PA1Strangers` showed that this
lets anyone who knows the bot's @username cancel every code the owner makes, for no gain against
guessing (the code has about 50 bits). It is now
`test_wrong_codes_count_and_lock_out_their_sender_not_the_owner`: the five wrong codes lock out
their sender (the right code from that sender is refused too), the code survives, and the owner
pairs. All 55 attacks in `test_phone_adversarial.py` pass as written.

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
the catalog, and the adapter refuses to load or run anything matching it. (The ComfyUI H3 graph of
the h3 talk chain is gated by its graph template's license fields instead: see "H3 talking shots".)

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
  (`ownership_seeded`). The previous version recorded no server id, so a seeded model gets the id
  the server reports it under on gpud's first look (and the adapter is told that copy is gpud's).
  `status` marks each resident model `gpud` or `not gpud`.

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
`report`) is described in [Planner](#planner); the same verdicts from a phone in
[Phone approvals](#phone-approvals).

Each submission carries its baton: speech and music finish with `review_audio` (the vision
brain sees the waveform and the measured WAV statistics) or `recover_audio` / `recover_music`;
video finishes with `review_render` (2 to 4 sampled frames attached) or `recover_render`. The CLI
opens the same SQLite file as a non-leader client (`Supervisor(lead=False)`), so admission rules
apply and it never takes gpud's lock.

State lives in `~/.foreman/`: `foreman.db` (plus `foreman.db.lock`), `logs/gpud.log` (JSON lines,
rotated; every supervisor event is mirrored there), `pages.jsonl` (one line per page to a human)
and `artifacts/<run id>-<process>/`. The planner adds `templates/`, `goals.md`, `inbox/` and
`reports/` (see [Planner](#planner)); the phone service adds `secrets/telegram.json`,
`logs/phone.log`, `phone.lock` and `phone/tmp/` (see [Phone approvals](#phone-approvals)). Audio runs leave `output.wav` and `waveform.png`. Video runs
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
`~/.foreman/logs`. macOS gives a Background job's children its throttled CPU and I/O band, so the
ComfyUI that gpud starts and the h3 media worker leave that band first (an edit there ran about five
times slower than from a shell). Nothing installs it automatically. The phone service has its own template
(`deploy/com.github.ggaabe.foreman.phone.plist`, see [Phone approvals](#phone-approvals)).

```bash
scripts/install-launchagent.sh --dry-run   # print the rendered plist
scripts/install-launchagent.sh             # render, plutil -lint, launchctl bootstrap gui/$(id -u)
scripts/uninstall-launchagent.sh           # launchctl bootout, remove the plist (keeps ~/.foreman)
scripts/install-launchagent.sh phone       # the same for the phone service (uninstall-launchagent.sh phone)
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
  cannot take it again. "Does not own" is provenance, not the catalog: on the Studio the default
  is MLX-Serve.app's Flash-Next, the catalog's fallback brain, and it is restored like any other
  model; only a copy gpud's own load made is left alone. Turn this off with `preserve_default=False`.
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
python3 -m foreman.cli verdict 41 reject --tags voice,pacing   # what was wrong ("Reasons for a reject")
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
infrastructure/content (and how many infrastructure failures did not count); the owner's reasons
for the rejects (reason tags per template); then GPU-hours and
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
| `flags` | `speculative` and `filler` (GPU shares), `commercial` (license gate), `hard_deadline` (reserved review slots), `private` (Foreman has no cloud lane; the phone service sends only a notice and the buttons for its drafts, never the brief, the brain's note or a file) |
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
  review; nothing in it posts, uploads, sends or spends. The one exception is the owner's own
  [phone service](#phone-approvals), which sends drafts and pages to the single private chat the
  owner paired, and nowhere else (never a private template's content).
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

## Phone approvals

`foreman/phone.py` puts the review queue and the pages on the owner's phone through a Telegram
bot: each draft arrives with its MP4 or audio, and Approve / Reject (then Used it / Undo) record
the verdict. It is its own process and its own LaunchAgent, not part of gpud. It reads and writes
the same SQLite file as a non-leader client (WAL, 5 s busy timeout), never takes gpud's lock and
never talks to an engine, so a Telegram outage, a slow upload or a bug in it cannot delay or crash
scheduling, and a gpud restart is just another writer on the file. gpud does not even import it
(`foreman/phone_config.py` holds the settings gpud validates).

Every verdict, retract, shipment, note and page resolution goes through the CLI's own functions
(`cli.apply_verdict`, `apply_ship`, `apply_note`, `apply_resolve`, `apply_planner`), so a press on
the phone is a `verdict JOB accept` typed on the Studio: the same validation, the same
compare-and-set item states, the same events (with `"by": "phone"`), the same posterior and the
same kill criteria afterwards.

### Setting it up

1. In Telegram, message **@BotFather**, send `/newbot`, choose a display name and a username
   ending in `bot`, and copy the token it answers with (digits, a colon, about 35 characters).
2. On the Studio, run `python3 -m foreman.cli phone setup` and paste the token at the hidden
   prompt (or pipe it in: `pbpaste | python3 -m foreman.cli phone setup`). Setup checks it with
   `getMe` and saves it in `~/.foreman/secrets/telegram.json` (directory 0700, file 0600).
3. Setup prints a link, `https://t.me/<your bot>?start=<code>`, and the code. Open the link on
   the phone and tap **Start** (or send `/pair <code>` to the bot). The code works once, for 15
   minutes, and only from a private chat. Setup waits and prints `paired` when it arrives. Taps on
   an older link (sent before this code existed) are ignored: tap the new one.
4. Install the service: `scripts/install-launchagent.sh phone` (`com.github.ggaabe.foreman.phone`,
   KeepAlive, logs in `~/.foreman/logs/phone.log`). Then `python3 -m foreman.cli phone test` sends
   a harmless message.

If the service already runs, `phone setup` leaves the pairing to it (`--no-wait` returns right
after printing the link). Without a token or a pairing the service idles and looks again every 30 s.
`phone setup` also repairs a secrets file that is corrupt, unreadable or a symlink (it replaces
it), and `phone unpair` removes one.

```bash
python3 -m foreman.cli phone setup [--no-wait] [--timeout S]   # token, getMe, pairing
python3 -m foreman.cli phone status        # token configured (never shown), paired, service, counts, last error
python3 -m foreman.cli phone test          # a harmless message to the paired chat
python3 -m foreman.cli phone resend        # every draft awaiting a verdict and every open page, again
python3 -m foreman.cli phone unpair [--keep-token]
scripts/install-launchagent.sh phone [--dry-run]       # scripts/uninstall-launchagent.sh phone
python3 -m foreman.phone [--once]                      # the service in the foreground
```

### What arrives

| What | When | How |
|---|---|---|
| A draft | an item enters the review queue | The MP4 by `sendVideo` up to 45 MB (`video_max_mb`); a bigger one, or a render without an MP4, sends its middle sampled frame as a document. Speech and music as an AAC M4A made with ffmpeg (the `ffmpeg` setting, as for the mux) for `sendAudio`, else the WAV as a document. Nothing above the Bot API's 50 MB upload limit is ever sent (a WAV that big sends its waveform). A 400 or 413 for a file falls back to the next way, down to text only; files already pruned send the text with a note |
| A page | `human_review`, `engine_lost`, `safe_mode`, `planner_suspended`, `brain_blocked`, ... | Text with a Resolve button. A page the brain raised about a draft rides on that draft (its verdict resolves it, as on the CLI). Resolving a SAFE-mode page does not leave SAFE mode: that is still `clear-safe` on the Studio |
| A digest | once a day after `digest_hour`, if `digest` is on (off by default) | Drafts and minutes waiting, open pages, the week's accepts, rejects and uses |

Each draft and page is one row in `phone_messages` (`UNIQUE(item, kind)`), claimed before the send
and recorded after it, so it goes out once across restarts of either process. A send cut short by
a crash between the two goes out again once its claim is 30 minutes old: only then can a second
copy exist (Telegram has no idempotency key).

The caption is plain text and is sent without `parse_mode`, so HTML or Markdown in a brief shows as
written: the job id and template, the consumer, the brief (at most 360 characters), the brain's
verdict and note, the review minutes, how the files went out, and "Reply to this message to add a
note to your verdict." Terminal escape sequences, control characters, line breaks and invisible or
bidi characters are removed. In the brief, the brain's note and page reasons, anything Telegram
could make tappable is defused: URLs and anything host-shaped (any script, any TLD, punycode
included; a file name such as `config.json` stays) become `[link removed]`, and `@name` and
`/word` are spaced apart (`@ name`, `/ word`) so they are neither a mention nor a command. Link
previews are off, and the caption stays inside Telegram's 1024 characters.

### Buttons, notes and commands

- **Approve / Reject** record `verdict JOB accept|reject` (the review minutes default to the
  item's estimate, as on the CLI). After Approve the message offers **Used it** (`ship JOB`, at the
  template's value) and **Undo** (`verdict JOB retract`); after Reject, the reasons (below), then
  **Undo**. The answer shows the new P(ship); a template a kill criterion paused is announced in
  the chat.
- **Reasons for a reject.** Reject records the verdict exactly as before (once, the same `verdict
  JOB reject`); then the message's buttons ask what was wrong: **Script, Voice, Pronunciation,
  Music, Lip-sync, Faces/consistency, Stills, Motion, B-roll, Captions/text, Pacing, End card,
  Other**, then **Done** and **Undo**. A tap toggles a reason (a check mark shows it) and only
  changes the message; **Done** stores the picked reasons with the verdict (the ledger's
  `verdict_tags`, and "reasons: voice, pacing" appended to the verdict note; a `verdict_tagged`
  event) and the message goes back to **Undo**, which also clears them. Done with nothing picked
  stores nothing. A tap on an older drawing of the buttons (a double tap, or a tap before the last
  edit reached the phone) changes nothing and says so; the buttons on screen keep working. An Undo,
  a retract on the CLI or the pruning ends the picking, and its buttons do nothing after it. The
  picking belongs to that very Reject (its `verdict_recorded` event, kept as
  `phone_messages.pick_of`), so a retract and a new reject on the CLI end it too, even before the
  next refresh: its tags never attach to the new reject. A Reject recorded just before the service
  stopped (the verdict committed, the buttons never redrawn) asks for the reasons on the service's
  first pass after the restart, and the press Telegram delivers again then records nothing. On the
  CLI, `verdict JOB reject --tags voice,pacing` gives them with the verdict (on an item already
  rejected, it sets them, replacing the earlier reasons in the note; an owner's own note that reads
  the same stays). An unknown tag is refused naming a few of them, however many were given. `review`
  and `report` show the reasons per template, and `ledger.reject_tag_counts` counts them for the
  planner to read later (the planner's prompt does not carry them yet). `"reject_tags": false` in
  `phone_options` turns the question off.
- The message's status line and buttons follow the item: a verdict, retract or pruning done
  elsewhere (the CLI, retention) redraws it within a minute (`refresh_s`).
- **A reply to a draft** adds a note: kept until the verdict while the draft waits, added to the
  verdict (a `verdict_noted` event) within 30 minutes of it (`note_window_s`), refused after. Each
  reply counts once: a reply delivered again after a crash (its update not yet confirmed) adds
  nothing, and a reply that is empty once control characters are removed is refused, not "noted".
- Forwarded messages are never commands, notes or pairing codes, even from the owner.
- `/queue` sends every draft still awaiting a verdict and every open page again (the older copy
  loses its buttons). `/status` shows gpud's mode and last tick, the planner, the review queue and
  the week's caps per template. `/pause` and `/resume` pause or resume the planner only. `/help`
  lists them. Other text gets one polite pointer to the commands per 10 minutes.

### Security

- **Secrets.** The token, the pairing state and the HMAC key live only in
  `secrets/telegram.json`, written atomically under a lock; a looser mode is tightened back to
  0600 / 0700 when it is read (the directory too). config.json refuses a `phone_options` key or
  value that looks like a token without echoing it (gpud validates the same file, and its stderr is
  a log), and an `api_base` that is plain http to anything but this machine (the token is in every
  request's URL). The token is removed from every log line (the formatter redacts the whole finished line,
  tracebacks included), every exception (a Bot API failure is raised without the urllib exception,
  whose text or URL can carry it), every stored error and the CLI's output. The pairing code is
  kept only as an HMAC.
- **Pairing** binds exactly one private chat id and one user id. A sender with five wrong codes
  is ignored for the rest of that code's life (kept only as an HMAC of its id); strangers cannot
  burn the owner's code. The right code seen in a group burns it; a group, a channel, a bot, a
  forwarded message or a message dated before the code was made can never pair.
- **Unpairing and re-pairing apply at once.** The secrets file is read again before every update
  and right before every verdict, pause or resend, so a `phone unpair` or a new `phone setup` while
  the long poll waits also covers the updates that wake it: the lost phone's /queue, /status, /pause
  and button presses do nothing. Which pairing `phone_messages` belongs to is recorded with it, so
  a crash between a new pairing and clearing the old chat's messages heals on the next pass.
- **Everyone else is ignored.** Messages and button presses from another chat or user, any group,
  any bot, and every update type other than a message or a button press get no reply and are
  logged without their content.
- **Signed buttons.** Button data is about 40 bytes: the action, the draft or page, the message's
  issue counter (a reason toggle also its reason's letter) and a 96-bit HMAC-SHA256 over them and
  the chat id, keyed by the pairing (a new key at every pairing and unpairing). Forged or garbled
  data, data from an earlier pairing, a press on an older copy, a press on an older drawing of the
  buttons (the counter moved on) or on an item whose state changed meanwhile records nothing: it is
  answered with what happened and the message is redrawn for the current state. Only the canonical
  spelling of the data verifies (no leading zeros, nothing after the MAC). Every send, a resend
  after a crash included, draws a new issue, so two copies never carry the same data. The ledger's
  compare-and-set states are the last guard, so a double tap or a race with the CLI records one
  verdict.
- **Files.** Only regular files strictly inside the artifacts directory are uploaded (resolved;
  never a symlink, never a file with a second hard link). The upload opens the file itself without
  following links and checks it is the very file that was checked (device and inode), so a file
  swapped for a link afterwards uploads nothing; ffmpeg gets the checked WAV on stdin with the WAV
  demuxer forced, never a path. Nothing above 50 MB is uploaded whatever the settings
  (`upload_max_mb` is at most 50 and `video_max_mb` is held under it).

**Privacy.** Drafts (the MP4 or audio, the brief, the brain's note), your notes and pages pass
through Telegram's servers (bot chats are not end-to-end encrypted) and stay in the chat history on
the phone. A template flagged `private` keeps its work off Telegram: its drafts arrive as a notice
with the job id and the buttons only (no brief, brain note or file), and pages raised inside its
lineage carry no details. To keep everything off Telegram, stop the service
(`scripts/uninstall-launchagent.sh phone`).

**Revoking.** `python3 -m foreman.cli phone unpair` tells the chat it is unpaired, deletes the
token and the binding (`--keep-token` keeps the token to pair again) and invalidates every button
already sent, effective for the very next update the service handles. To kill the token itself, send `/revoke` to @BotFather and pick the bot; the service
then logs that Telegram rejected its token and waits for a new `phone setup`.

### Robustness

- Long polling (`getUpdates`, 20 s). The offset is persisted in `phone_state` after each update is
  handled, so a restart neither loses nor repeats one. Each update has its own boundary: one that
  raises is logged and skipped (its offset advances); one that meets a busy database is tried
  again, up to three times.
- Network errors back off exponentially (1 s, doubling to 5 min); a 429 waits its `retry_after`
  (at most an hour, whatever it says); a 409 (someone else reads the updates) waits; a 401 (a
  revoked token) is logged once and the service waits for a new token. A file that vanishes or
  is swapped mid-send is not an outage: the next way of sending the draft is tried. A press that
  keeps meeting a busy database is answered ("nothing was recorded, tap again"). A new bot (another
  bot id) reads its updates from the start: each bot numbers its updates on its own. Only a failure to reach Telegram ends a pass early: a bug or a bad
  row in the outbox never stops the owner's presses being read. A draft that keeps failing for
  another reason is marked failed after five tries (`phone status`; `/queue` tries it again).
- A 403 (the owner blocked the bot) stops sending until the chat writes again or presses a button.
- A watchdog thread exits the service (status 70; launchd restarts it) when a pass makes no
  progress for 10 minutes; an upload reports progress for every MB it sends.
- Memory is bounded: answers are read up to 8 MB, uploads are streamed in 1 MB pieces, and the
  outbox and the refresh handle a few messages per pass.

Settings go under `phone_options` in config.json (fields of `foreman.phone_config.PhoneConfig`),
for example `{"phone_options": {"digest": true, "digest_hour": 8, "pages": false}}`;
`FOREMAN_PHONE_API` overrides the Bot API's base URL. The bot token is never a setting.

### Known gaps of the phone service

- It has not run against the real Telegram API: the tests use `tests/fake_telegram.py`, which
  follows the Bot API documentation (methods, error shapes, offsets, `retry_after`) but is not
  Telegram. The first live run should be `phone setup` and `phone test` on the Studio.
- HTTPS uses the interpreter's CA store (`/usr/bin/python3` reads `/private/etc/ssl/cert.pem`); a
  proxy configured in the environment or in macOS is used.
- A crash between a successful send and its record sends that draft again, once, after the claim
  goes stale (30 min).
- A note replied before a verdict that is then given on the CLI stays with the phone: it only goes
  with a verdict given by button.
- A press is answered after the verdict is recorded; while Telegram is slow the button spins.
- Reasons picked after a Reject are kept on the message until Done: reasons left without Done are
  not stored (the verdict is). Each toggle redraws the buttons, so a second reason tapped before
  the first edit reached the phone is not counted (its answer says so; tap it again).
- Defusing host-shaped text in briefs and notes is deliberately broad: a word like `Node.js` or
  `final.mix` becomes `[link removed]` (Telegram links about 1500 TLDs, and that list changes); a
  few file extensions that are no TLD (`json`, `wav`, `png`, ...) are kept.
- Upgrading from the first phone version (no pairing id recorded) forgets the sent messages once
  and sends the drafts and pages awaiting the owner again.

## Ads

Finished vertical ads instead of single clips: 15 to 30 seconds (about 46 at most, see below), 9:16, four to six
shots of 2 to 8 seconds, one recurring character (a mascot or a human host) who speaks the hook
and the call to action lip-synced to a voice-over, B-roll of the character, real screen captures
of the product (never generated UI), captions timed word by word, a music bed ducked under the
voice, an end card with the real price and address, and an AI disclosure. Which tools make each
part is configuration (`ads_options`, `comfyui_options`), not code.

An ad is an owner template of kind `ad`. The planner proposes ads like any other template, and
`cli ads enqueue` runs one on demand. Either way the ad is one planned item whose lineage is a
chain of gpud jobs, so every step runs under gpud's leases, progress, cancel, provenance and
exactly-once failure handling, and the finished MP4 reaches the review queue and the phone like
any draft.

### The chain

The root job reads the facts. Every job after it is submitted by the ad's dispatcher, a
deterministic brain step (`ad_next`, `ad_recover`) that runs as the previous job's continuation,
reads what is already on disk in the ad's work directory and submits the next job. A step whose
outputs exist is skipped, so a crash, a retry or a gpud restart resumes from the files.

| Step | Job kind | Runs on | Writes |
|---|---|---|---|
| facts | `ad_facts` | local CPU engine | `facts.json` (the facts URL fetched, or the facts file read), `ad.json`, `character.json` |
| script | inside `ad_next` | the resident brain | `script.json`: hook, body, call to action and 4-6 shots, validated by code |
| voice | `ad_voice` | Kokoro, or Qwen3-TTS cloning the bible's reference WAV | `voice/line-<i>.wav`, `voice/full.wav` (and `-a<n>.wav`, the audio gate's retakes) |
| gate | `ad_gate` | Qwen3-ASR, free-running (no prompt) | `gate/line-<i>-a<n>.json`, `gate/track-r<n>.json`, `gate.json`, `voice/track.wav` |
| timing | `ad_timing` | Qwen3-ASR (word times from the forced aligner) | `timing.json`: the script's words with times, cuts, 8k+1 frame counts, `voice/talk-<k>.wav` |
| stills | `ad_stills` | image edit on mlx-serve, a ComfyUI graph, or the approved still itself | `stills/shot-<k>-a.png` (and `-b.png`, B-roll's last frame) |
| shots | `ad_shots` | LTX: talking shots two-stage audio-to-video, N seeds each; B-roll one-stage first/last frame | `shots/shot-<k>-s<seed>.mp4` (+ sidecars with stats) |
| music | `ad_music` | ACE-Step, instrumental, the ad's length | `music/bed.wav` |
| assemble | `ad_assemble` | local CPU engine, ffmpeg | `final/ad.mp4`, `captions.ass`, `assembly.json`, `pick.json` |
| mix gate | `ad_gate` | Qwen3-ASR, free-running | `gate/mix-r<n>.json` (and once more after a re-mix with the music lower) |
| finish | `ad_finish` | local CPU engine | `final/qa.json`, `final/video.json`, `manifest.json` |
| review | `ad_review` | the vision brain | its verdict; the item goes to review (accept or human_review) or fails (give_up) |

gpud runs one job at a time, so the GPU holds one media stack at a time in that order, with the
brain resident beside it as for renders. mlx-serve's three model slots (the brain, the owner's
default, one media model) make gpud unload each media model before the next. The work directory
is `$FOREMAN_HOME/artifacts/ads/<ad id>/`; `cli ads inspect` shows it.

**The script.** The brain answers a strict JSON object. Code checks it: line and shot counts and
lengths (the template's bounds, about 2.6 spoken words a second, 2 to 8 seconds a shot), shot
kinds `talk|broll|screen|endcard`, every line covered once and in order, the hook and the call to
action as talking shots, screen shots only from the template's captures, still tags only from the
bible's approved stills, and the disclosure copied exactly. The spoken lines: no stage directions
or markup (`*waves*`, `(smiling)`), no emoji, no text-direction controls, no word spelled with
look-alike letters of another script; every number in `facts.json` however it is written
(digits, words, `$.99`, `USD7`, `2 million`, `10k`, `one dollar`, `a hundred dollars`), and a
price or a percentage must be one the facts state as a price or a percentage, not just any number
on the page; no testimonials, reviews, ratings, quotes, invented customers or the host speaking as
a customer; no guarantees, refunds or deadlines ("by Friday", "the same day", "within 24 hours");
no legal advice ("by law", "the court will accept"); no other brand's mark (the template's
`banned_marks`, matched across case, spaces, hyphens and look-alike letters, and no trademark
symbols); no address but the ad's own, written or spoken ("rival dot com"). The visuals (the image
and video prompts): no on-screen text or signs, no prices, no generated product screens or pages
(only screen shots of real captures show the product), no other brand, no address. Checks read
the text folded (NFKC, invisible characters removed, look-alike letters as Latin), and the lines
are stored exactly as they were checked, so the voice and the captions say what passed. An invalid
script gets one repair attempt with the problems listed (an h3 plan two); one still invalid after that is a content failure.
The facts and every other quoted text reach the brain inside one `<untrusted>` block whose markers
the text cannot close.

**The pronunciation lexicon.** A voice reads a brand written as one word the way it reads any
unknown word, and a brain that reviews the finished ad cannot hear it: a brand was read as two other
words in live ads, while the same voice and ASR got it right every time once it was respelled as two
words. The template's `lexicon` lists such terms, `[{"display": "ExampleCo", "spoken": "Example Co",
"respellings": ["Eggs Ample Co"], "ipa": "...", "arpabet": "..."}]`, and the owner's
`$FOREMAN_HOME/lexicon.json` (a list of entries, or `{"entries": [...]}`) is merged under it (the
template's entry for a term wins). The script, the captions, the end card and every check keep the
display text; only what the voice is given changes: each term, found as a whole word in any case,
with or without spaces or hyphens inside and with look-alike letters read as Latin, becomes its
`spoken` spelling (default: the display), and the `respellings` (best first) are what the audio
gate tries next. The ad's own address is a term too (said as written, then spelled out, "example
dot com"), and a brand written inside the address is said with its own spoken spelling there too
(and its respellings are tried once the address's own run out); only an address is spelled again
inside, so another term's chosen spelling is said as given (a `CO` entry never turns "Example Co"
into "Example Colorado"). Every spelling is
checked like a spoken line when the template loads (no markup, symbols or look-alike letters, no
claim, other brand or other address; at most 40 entries of at most 6 respellings), and the display
too (no look-alike letters: it goes into the gate's reasons, the manifest and the timing's spelling
hints). A spelling says exactly the display's numbers, perhaps another way ("Form 1099" may be said
"Form ten ninety nine"), never another one: "Example Co, 50% off" is refused, and so is a display
that is a price or a percentage (a number is never a term: the gate checks it as a number). The
voice step checks every line again as it will be said: a line whose spoken text would state a
number its display does not fails the ad before the voice runs. The owner's file
is checked when an ad's facts step reads it, and one that does not hold stops the ad there, before
any GPU work, as the machine's configuration (`engine_unconfigured`), not as content. `ipa` and
`arpabet` are kept for reference and the manifest: no MLX-Serve voice is known to read them.

**The audio gate** (code, on the ASR model: `ad_gate`). Every voice line, then the voice track, then
the final mix, is transcribed free-running by Qwen3-ASR (`/v1/audio/transcriptions`,
`response_format` json, the ad's language, no prompt and no spelling hints, no word times) and
compared with the script. Both are normalized the same way: case, punctuation and contractions; a
number in words or in digits is one number with its unit ("nine dollars and fifty cents" is "$9.50",
"twenty percent" is "20%", "2 million" and "two million" are "2,000,000", "one point five million"
is "1.5 million"; a price the script states, read as "nine fifty", is that price, and a number read
in digit groups, "twenty twenty six" or "ten ninety nine", is the number); "dot com" is ".com"; and
a lexicon term heard as any of its spellings is the term (heard with an "s" after it, "form files",
it is the term and a word error). A take passes when every lexicon term of its line was heard as
often as the line says it; every number was heard with its unit (a price in its currency: "$9.50"
heard as "9.50 euros" is another price; a percentage as a percentage: "20%" heard as "20 dollars" is
another number) and in the line's order (two prices swapped fail); no number the line does not state
was heard (a voice that adds "$5" fails); and its word error rate is at most `gate_wer_max` (0.08).
The ad's address must be heard as itself: "example.com.au" or "shop.example.com" is another address.
A brand heard written as its own address ("<brand>.com" for a correctly said brand, as Qwen3-ASR
writes it at times) is the brand. The forced aligner's word times are never a check: they assume the
script. A take that fails is said again: with a new seed when the voice varies by seed (Qwen3-TTS;
`gate_rerolls` per spelling), then with the next respelling of each term it did not hear, up to
`gate_attempts` (4) takes of a line; Kokoro reads the same words the same way, so a line it misreads
with no spelling left (a wrong number, say) fails at once. When every line passes alone, the track
made of the passing takes is checked too, and a line it loses is said again the same way. Each line
gets all its takes even when lines are found failing one at a time (alone, then in the track):
retake rounds are bounded per line, and as a safety net by the lines times `gate_attempts` and by
the ad's continuation depth (a round that would take the ad past it fails the gate, naming the
line). A line that still fails fails the ad as content, before anything renders, with a reason that
names the term and what was heard ("audio gate: line 0 never said 'ExampleCo': heard 'example cove'
(2 take(s), said as 'Example Co', 'Eggs Ample Co')"): the owner never gets that ad. After the cut,
the final mix is transcribed the same way: it must still say every term and number, and its word
error rate may not exceed the voice track's by more than `gate_mix_margin` (0.05): words the music
hid and the music heard as words between them both count (music masking the voice), while words an
ASR writes over the music alone before the voice starts or after it ends (the end card) do not. If
the mix fails, the ad is mixed once more with the music `gate_remix_db` (8 dB) lower and checked again; a
second loss is a content failure. Every take (what the voice was given, its seed, why it was taken,
what was heard, the word error rate, the terms and numbers it missed), every track and mix check,
and the verdicts are kept in `gate.json` and the gate's files in `gate/`, and go into the manifest.
The transcriptions need no word times, so a companion model their requests loaded (the forced
aligner) is unloaded as soon as the gate's job ends, through the adapter's companion record (one
that was resident before is never touched).

**Timing.** The voice track is the line takes the audio gate passed, joined with short pauses
(`voice_track: "lines"`, the default; `"full"` uses the full read instead, and the gate then judges
and retakes the full read). Qwen3-ASR transcribes it with word times (here, and only here, the
template's `spellings` and the lexicon's display terms are given as spelling hints), and the
times are mapped onto the script's own words by an edit-distance alignment, so captions show the
script ("example.com", "$9.50") and never the ASR's spelling; a word the ASR dropped is
interpolated. The ASR answer is untrusted: one with more words than the track could hold is
ignored (estimated timing instead), and word times outside the track, out of the script's order
or outside their line are dropped and interpolated, so one bad time never drags the rest of a line.
A take asks for the shortest 8k+1 frame count that covers its shot. Cuts fall in the middle of the pause between two shots' lines. A talking shot gets
the track's slice for its line, padded with silence to exactly `num_frames / 24` seconds, as
audio-to-video needs; that padded WAV is also what is muxed onto the take.

**Takes.** Each talking shot is rendered with `talk_seeds` seeds (default 3); all takes are kept
and the assembly picks one with a scored heuristic: how well the motion around the mouth follows
the voice's loudness (correlation over frames), how much the picture moves without freezing, and
no black frames. A take or still that fails its checks (black, frozen, wrong size) is recorded and
the ad carries on with the others; when a shot has no good take left, one more seed is tried
(`talk_retry_seeds`), then it is a content failure.

**Assembly** (`foreman/assembly.py`, ffmpeg only): every shot scaled to 1080x1920 and cut hard
on its boundaries, on the plan's absolute frame numbers (so rounding never drifts against the voice
and the captions); screen captures fitted over a blurred copy of themselves, stills with a slow
push-in; captions as ASS with the spoken word highlighted, inside the central safe zone (never
over the top 12% or the bottom 20%), burned in with libass and an explicit `fontsdir`; the end
card (an approved still tagged as the template says, or a plain colour) with the price, the
address and the call to action drawn with an explicit `fontfile`; a small "AI-generated" label on
every frame when the template asks for it. Drawn text is verbatim (`expansion=none`: "20%" is
text) and measured with the font's own advance widths, so a line too wide for the frame is shrunk
or broken in two, and a caption word that cannot wrap (a long address) is drawn smaller; the music bed ducked under the voice with
`sidechaincompress`, then two-pass `loudnorm` to -14 LUFS and -1.5 dBTP; H.264 (VideoToolbox,
libx264 as the fallback) and AAC with `+faststart`, re-encoded smaller until it is under
`max_mb` (40 MB, at most 45: the phone service's sendVideo limit) so Telegram's sendVideo takes it;
an MP4 still over the cap is deleted, never left as if finished. ffmpeg on the Studio has no
fontconfig, so a real font file is resolved at runtime (`ads_options.fonts`, then the usual macOS
paths); a missing one (or no ffmpeg) stops the ad at its facts step, before any GPU work, as an
infrastructure failure. After the cut, sampled frames of the final are compared with the same
frames before captions and label: text that did not show is not a finished ad (retried as an
infrastructure failure).

**QA** (`ad_finish`): resolution, duration, an AAC track, size, integrated loudness and true
peak (ebur128), black intervals (blackdetect) before the end card, frozen intervals (freezedetect)
inside generated shots, flat frames (a NaN-poisoned render decodes to a flat picture), finite
measurements, and the captions and the label measured as drawn (the frame comparison above, not a
flag). Then `ad_review` runs on the vision brain: it reads
the facts, the spoken words, what the audio gate heard in the final mix, the gate's verdict, the end
card and the QA, and compares sampled frames with the bible's hero still. It can accept only when
the facts and the character both check out, it saw the images and the audio gate passed; an ad whose
gate did not pass is given up without asking it: one the gate failed, and one nobody heard end to
end (no gate record, or the final mix never checked), so the owner never gets an unheard ad.
Otherwise the ad goes to the owner flagged (`human_review`) or fails (`give_up`). The item's note
(and so the phone's caption) starts with the gate's result ("audio gate passed (voice WER 0.00, mix
WER 0.02, 1 retake(s))").

**The manifest** (`manifest.json`): models per role and their licenses, versions (mlx-serve,
ComfyUI, ffmpeg), the backends and graphs, every seed (stills, takes with their scores and which
one was picked, music), the disclosure flags, a snapshot of the facts with their hash, the
script, the shots, the QA, the audio gate (the lexicon used and every take, track and mix check),
every step's job and duration, and the review.

**Failures.** An infrastructure failure of a step (a crash, a timeout, an engine lost, a load that
never fit, a restart) is retried by `ad_recover` with the same request; its finished outputs are
kept. Engines say when the host is at fault (Poll.reason `engine_error` or `engine_unconfigured`: no
ffmpeg or font, a model without the capability its role needs, a facts site answering 429 or 5xx, an
mlx-serve media model beside a running ComfyUI, a download that is not the file, an mlx-serve answer
that says it is busy or unavailable (HTTP 429, 502, 503), a transcription answer without its text,
and an ASR that hears no word in any of a gate job's two or more recordings though none is silent),
and those stay infrastructure however often they repeat; gpud now keeps that reason from the
engine's poll (it used to record every such failure as a crash, which a repeat turned into content).
So an ASR outage fails the ad as infrastructure, never as the template's content, and the gate never
passes on a transcription it did not get. The ledger classifies the item as usual, per logical task,
so infrastructure failures do not use the template's weekly cap or the review budget, and a take the
ad carried on without does not make a later infrastructure failure count as content. The dispatcher
keeps three retries' worth of continuation hops free: an ad whose planned jobs would leave fewer is
refused at its timing step.

**Retention.** Raw frames are deleted after each take is muxed, scratch files after each op (and
a job first deletes the scratch a killed attempt left), and the assembly's intermediate segments
after the cut. Writes are atomic and a failed one leaves no temp file. After the owner's verdict
the takes the cut did not use are deleted; the final, the picked takes, the stills, voice, music
and the manifest stay until the item's own retention prunes the whole work directory. The disk
guard (the planner's and `ads enqueue`'s) counts an ad's estimated size for as long as any job of
its lineage is live, plus what other items in flight may still write.

**Characters at run time.** The facts step snapshots the bible with a hash of every approved
still. Each later step reads the owner's live `bible.json` again: a still whose approval was
withdrawn, or whose bytes changed under the same name, is no longer used.

### ComfyUI on demand

Some graphs (character stills from an image-edit model, reference-conditioned shots) run in
ComfyUI, which has its own Python and holds its own GPU memory. `foreman/adapters/comfyui.py` is a
gpud engine that starts ComfyUI only for the job that needs it:

    <comfyui_options.python> main.py --listen 127.0.0.1 --port 8188 --use-pytorch-cross-attention --disable-auto-launch

from `comfyui_options.dir`, waits for `GET /system_stats`, uploads the input images, queues the
graph (`POST /prompt`), follows it on `/queue` and `/history/<id>` with a bounded heartbeat,
downloads the output (`/view`, streamed to a temp file and checked: a cut connection, an error
page or a truncated image is never kept; its own uploads and the downloaded output are then deleted from
ComfyUI's input and output folders), then `POST /free` and stops the process (SIGTERM, a
wait, SIGKILL only for the ComfyUI Foreman started). The server counts as resident while that
process is alive, answering or not (a busy or wedged ComfyUI still holds its memory). A restarted
gpud adopts it from `run/comfyui.json` on whatever port it was started, but only a process that is
alive, leads its own session, runs `main.py` and started when the file says; its leftover prompts
are interrupted and removed before the next job. A server that is already up checks the media
rule again before every job and graph, and a gpud shutdown stops it. Its catalog model is `solo_media`: gpud unloads its own
mlx-serve media models before starting it and stops it before loading any other media model, and
the adapter refuses to start (deferred, then an infrastructure failure) while mlx-serve holds any
media model, someone else's included, or free memory minus `plan_gb` would fall under the
headroom. A ComfyUI that Foreman did not start is never used or signalled, and mlx-serve is never
signalled at all.

Graphs are JSON templates in ComfyUI's API format in `comfyui_options.graphs_dir` (default
`$FOREMAN_HOME/comfyui/graphs`), with placeholders such as `{{prompt}}`, `{{seed}}`,
`{{width}}`, `{{height}}`, `{{num_frames}}`, `{{image_1}}`: a string that is exactly one
placeholder takes the value's type. The optional wrapper `{"graph": ..., "outputs": ["9"],
"defaults": {...}, "license": "...", "commercial_ok": true}` names the output node and the
license; a commercial template refuses a graph without `commercial_ok`. See
`examples/ads/comfyui/graphs/` for the format.

### Characters

Bibles live under `$FOREMAN_HOME/characters/<id>/` and are never committed: `bible.json` (look,
traits to keep, style notes, the voice: a Kokoro voice or blend, or a Qwen3-TTS clone from a
reference WAV) and stills tagged by pose or scene (`hero`, `turnaround`, `expressions`, `talk`,
`endcard`, any other scene name). The pipeline only uses approved stills; the hero is the
reference every edit starts from and the QA compares against.

```bash
python3 -m foreman.cli characters import ~/Desktop/mascot-stills --id mascot-a --kind mascot \
    --look "a round blue mascot in a yellow cap" --voice am_onyx,bm_george     # stills unapproved
python3 -m foreman.cli characters approve mascot-a hero.png talk.png turnaround.png
python3 -m foreman.cli characters list
python3 -m foreman.cli characters show mascot-a
```

### Templates and running one

`examples/ads/templates/ads.example-service.v1.json` is a generic ad template (inactive), with a generic
bible (`examples/ads/characters/`), facts file (`examples/ads/assets/facts.json`) and config
(`examples/ads/config.ads.json`). Its `ad`
block: `concept`, `character`, `facts` (`{"url": ...}` or `{"file": "assets/facts.json"}`: a file
under `$FOREMAN_HOME/assets`, never a secret or the configuration), `cta`, `end_card` (`price`,
`url`, `cta`, `still`, `background`), `disclosure` (it must say the ad is AI-generated),
`ai_label`, `captures` (files under `$FOREMAN_HOME/assets`), `banned_marks`, `music`, `style`,
`language`, `spellings` (terms the timing ASR should spell your way), `lexicon` (see "The
pronunciation lexicon"), `talk_seeds`. Its `params` are
`seconds` (15 up to what the shots allow), `shots` (4 to 6) and `seed`. Every shot but the end card
lasts at most `max_shot_s` (8 s) and the end card 3 s, so six shots run about 46 s at most: a
template whose `seconds.max` is longer than its `shots.max` can run is refused when it loads. The
end card's price must state a price, its call to action is held to the script's claim and brand
rules, and when the facts are read the price must be a price the facts state, every number of the
call to action must be in the facts, and the address must be on the facts URL's site (so a stale
template fails before anything renders).

```bash
python3 -m foreman.cli ads enqueue ads.example-service.v1 --brief "how fast it is from a phone" [--seconds 20 --shots 5]
python3 -m foreman.cli ads status
python3 -m foreman.cli ads inspect <job id | ad id>
python3 -m foreman.cli ads models [--rescan]     # the configured model per role against /v1/models
```

### Configuration

`config.json` keys (see `examples/ads/config.ads.json`):

- `ads_options.models`: the catalog model per role (`tts`, `tts_clone`, `asr`, `image`, `video`,
  `music`, `cpu`, `comfy`). Model ids live in the catalog (`foreman/studio.py`) and this map; the
  adapter checks each op against the model's capabilities in `GET /v1/models`.
- `stills_backend` (`mlx`, `comfyui`, `direct`), `talk_backend`, `broll_backend` (`mlx`,
  `comfyui`) and their graphs (`stills_graph`, `talk_graph`, `broll_graph`).
- `talk_seeds`, `talk_retry_seeds`, `broll_retry_seeds`, `still_retry_seeds`; `talk_pipeline`
  (`two_stage`, `two_stage_hq`), `talk_steps`, `talk_stage2_steps`, `talk_decoder`,
  `talk_cfg_scale`, `talk_cfg_audio_scale`, `broll_steps`, `broll_last_frame`, `image_steps`,
  `lora_paths` / `lora_scales`.
- `video_width` / `video_height` (704x1280: two-stage needs a 64 px grid, and at most 297 frames
  fit LTX's 768 MiB per request at that size), `out_width` / `out_height` (1080x1920).
- `voice_track`, `line_gap_s`, `endcard_s`, `words_per_s`, `min_shot_s`, `max_shot_s`,
  `music_db`, `duck_threshold`, `duck_ratio`, `loudness_lufs`, `true_peak_db`, `max_mb`, `fonts`,
  `endcard_color`, `push_in`.
- `expected`: seconds per unit (`still`, `talk`, `broll`, `line`, `full`, `asr`, `gate` (one
  gate transcription), `music`, `assemble`, `finish`, `script`, `review`, `dispatch`), and
  `max_job_s` (600): a step's units are packed into jobs up to that expected time, so every lease
  stays inside gpud's brainless budget.
- The audio gate: `gate_wer_max` (0.08), `gate_attempts` (4 takes of a line at most),
  `gate_rerolls` (1 new seed per spelling, for a voice that varies by seed), `gate_mix_margin`
  (0.05), `gate_remix_db` (8); `lexicon_file` (default `$FOREMAN_HOME/lexicon.json`; see
  `examples/ads/lexicon.json`).
- `characters_dir`, `assets_dir`, `commercial_models`, `max_jobs`, `keep_segments`.
- `comfyui_options`: `dir`, `python`, `port`, `graphs_dir`, `plan_gb` (the footprint gpud plans
  ComfyUI with, 60 by default: it fits beside the brain and the owner's default model in gpud's
  memory plan; set it to the bake-off graphs' measured peak), `start_timeout_s`, `exit_after_job`,
  `extra_args`.

The deployment policy allows 64 continuation hops per lineage (an ad is up to about twenty steps,
two hops each, plus retries, the audio gate's retake rounds and one re-mix); the dispatcher refuses
an ad whose planned jobs would not fit.

### H3 talking shots

A template whose `ad` block has an `h3` block runs the **h3 chain** instead of the voice, timing,
stills and shots steps: a human presenter says every beat on camera. Each beat is one talking take
of MiniMax H3 Ref2VA in ComfyUI (pictures: the beat's scene still, the hero still and a face
close-up; audio: the character's own reference voice, so the voice is one person's across beats;
the line inside the structured prompt, a static camera). The take is the picture and the voice:
no TTS runs. Code hears every take, keeps the first that says its line cleanly, takes a beat again
when it does not, and the edit lays the screen capture, the price card, a still and the end card
over parts of the beats while the voice goes on (J and L cuts on word onsets). The chain is chosen
per template, so a mascot template on LTX keeps running beside it.

| Step | Job kind | Runs on | Writes |
|---|---|---|---|
| facts | `ad_facts` | local CPU engine | as above, plus the h3 checks below; `ad.json` picks `backends.talk = "h3"` |
| script | inside `ad_next` | the resident brain | `script.json`: lines, one talk shot per beat, and the beats (scene, action, cutaways on words of the line, seconds, seed) |
| takes | `ad_shots` | ComfyUI (`comfyui/graphs`), the h3 graph | `takes/beat-<k>-t<n>.mp4` (+ the adapter's sidecar) |
| probe | `ad_probe` | local CPU engine, the media worker | `takes/beat-<k>-t<n>.wav` (16 kHz mono), `.probe.json` (levels, voiced runs, stray hands) |
| hear | `ad_gate` | Qwen3-ASR: hinted with word times, and free-running | `gate/beat-<k>-t<n>.asr.json`, `.asr0.json`; the verdicts in `gate.json` |
| music | `ad_music` | ACE-Step | `music/bed.wav` (as long as the takes could make the cut) |
| assemble | `ad_assemble` | local CPU engine, the media worker | `final/ad.mp4`, `final/audio.wav`, `captions.ass`, `ad.srt`, `timeline.json`, `final/beats/`, `publish.json`, `assembly.json`; `timing.json`, `pick.json` |
| final gate | `ad_gate` | Qwen3-ASR, free-running | `gate/final-<beat>-r<n>.json` per beat, `gate/mix-r<n>.json` |
| finish | `ad_finish` | local CPU engine | `final/qa.json` (with the h3 checks), `final/video.json`, `manifest.json` |
| review | `ad_review` | the vision brain | as above |

A round of takes is one take for every beat still pending, packed into ComfyUI jobs (`max_job_s`
over `expected.h3_take`, 140 s), so ComfyUI starts once per round; then one probe job and one
hearing job. **The judge** (`foreman/h3core.py`, code): the take must have audio, the render's size
and most of the length asked; the free-running transcript must say every lexicon term and number
of the line and no other number inside it, within `gate_wer_max`; the word-timed transcript must
start with the line (nothing said before it), not run on into another word within `h3_runon_s`
(0.15 s), and hear most of it; a cut word (the end card's) must start on a real voiced onset after
quiet (`h3_anchor_tol_s`, `h3_anchor_gap_s`: a forced aligner packs a spoken address short and
slides the next words early); a scene with `hands_gate` must show no stray hands in its bottom band
in the frames the cut shows of the take (from just before the line to its sound's end and hold, not under a
card or still);
the line must end before the take does; and what the take was heard saying may not add what the
ad's rules refuse and its line does not (a claim, another brand or address, a banned word: H3 makes its
own speech, and one inserted word stays within `gate_wer_max` on a long line; the final gate applies
the same rule to each beat's finished voice). A take whose probe found a problem (no audio, an audio
stream that does not decode, silence) is judged on it without being heard. A take of another size
than `h3_width` x `h3_height` stops the ad as the machine's failure (`engine_unconfigured`: the graph
does not take its size from `{{width}}` and `{{height}}`), not the template's. A render ComfyUI fails
on, or that leaves no file, is an infrastructure failure of the takes job, retried as it was (same
seeds), never a failed take. A failed beat is taken again with seed + 1000 n and, once
it was re-rolled `gate_rerolls` times for a term it did not say, that term's next spelling, up to
`h3_attempts` (3) takes and `h3_budget_s` (1200) render seconds per ad; then the ad fails as content,
naming the beat and what was heard. The dispatcher refuses an ad whose worst case (every beat taken
`h3_attempts` times) would not fit the continuation depth. A take's voice is told the line with the
lexicon's spellings (the ad's address spelled out first: "example dot com") and the template's
`readings` ("nine fifty" for $9.50); the speech gate must take what the voice is told for the line
word for word, or the ad fails before any render. An ASR may write a price said without a currency word
as a bare number ("9.50"): every judge (the take's, each finished voice's and the mix's) reads a bare
number equal to a reading's price as that price, and any other number as it was heard. Nothing else in
the prompt may speak for it: the
brain's action is held to the visual rules plus no claim, banned word, markup or quoted speech, the
template's presenter, framing and setting likewise, and every field is stripped of the prompt's own
markup (`<`, `>`, `[`, `]`, `{`, `}`), so the line is the prompt's one `<d>` dialogue tag.

**The edit** keeps each beat's take on a 24 fps grid trimmed to the line's acoustic bounds, cuts
the cutaways in on word onsets (a punch-in is cut inside the beat's own `crop`, so it never shows rows
the crop, and the hands check, leave out; no word time leaves its beat, so no caption runs over the
next beat's picture), removes pauses only under graphic cutaways and only where the take
is silent by energy, sets each beat's voice to -14 LUFS, lays the music `h3_music_under` (14) LU
under the voice (`h3_bed_lift` between lines), burns word-highlight captions (Arial Black,
hidden under the price card and the end card; libass is given the caption font alone, so it draws the
face the lines were measured with), encodes bt709 x264 (`h3_crf`) within `max_mb`, and
writes a sidecar SRT. **The final gate** hears each beat's finished voice again: one that lost a word
(an edit that cut "dot com") makes the ad be edited once more with no pause removed and no gap
closed; a second loss fails the ad. The mix is judged as above against the takes, with one re-mix
(the music `gate_remix_db` further under). **Finish** adds what the file must show: exactly the
timeline's frame count, no frozen second anywhere, a scene cut at every punch-in, bt709 tags (the
assembly checks that the captions show where they are drawn: in the caption band, or at the moved
position when every beat sets `caption_pos`). No AI
label is ever burned into the picture: an h3 template must say `"ai_label": false`, no drawn text (the
headline, the cards, the brand's name) and no line may label the ad as AI-made, and
`final/publish.json`, keyed to the finished file's hash, is the gate a person fills in before
posting (each platform's AI-content disclosure, the region, the SRT uploaded, the attribution); the
review's note ends by saying so.

**The template's `h3` block** (owner data, never in the repo; checked when the template loads, and
against the bible, the files and the facts at the facts step):
`presenter` (`noun`, `subject`, `object` for the prompt, `voice`: what the reference recording
holds, `voice_style`), `face` (the approved still tag of the face close-up), `scenes` (2 to 12:
`still`, `framing`, `setting`, `roles`, `hands_gate`, `crop`, `caption_pos`, `punch_anchor_y`),
`stills` (cutaway pictures: `still`, `note`), `captures` (PNG captures of `ad.captures` with `id`,
`note`, `crop`, `paint` boxes, the `tap` button and its colour), `price_card` (`big`, `lines`, a
capture `crop`, `underline` boxes, `pulse_line`, a `lead` of card-coloured pixels stacked between the
crop's text lines so an underline clears the next one, `paint` boxes), `endcard` (`lead`, `big`,
`lines`, `footnote`; `big` is the card's largest line: a price, a phone number or a short call, and
like every drawn text its numbers must be in the facts),
`brand` (`name`, colours, an optional `logo` PNG under assets), `headline` (true: the brain writes
one; a string; null), `readings`, `caption` (`size`, `margin_v`, `margins`), `punch_scale`,
`hold_cta_s`, `tail_hook_s`, `picture_style` and `soundscape` (the H3 prompt's look and sound; by
default the recipe's realistic photographic style and quiet room tone, so a cartoon mascot outdoors
says its own), `example` (a finished plan shown to the brain) and `notes`. See
`examples/ads/templates/ads.example-h3.v1.json` (generic: ExampleCo, example.com, $9.50), its
facts `examples/ads/assets/facts-h3.json` and presenter `examples/ads/characters/presenter-example/`.
The character's voice names its reference recording (`voice.ref_wav`, allowed with any engine).

**The plan.** The brain writes the beats (line, scene, action, cutaways on words of the line), with the
template's `example` as a worked plan, and code checks them (`validate_h3_script`); an h3 plan gets two
repairs with the problems listed before the ad fails as content. A screen or price card or a still stays up
for at least three words, the call to action goes on for at least two words after the web address (the end card
cuts in on a word after it), and a tap, pulse or underline whose word the cutaway cannot draw (outside it,
or on a card with nothing to press or underline) is left out and noted in `script.json` `dropped`, not a
reason to throw the plan away. An action may say a phone's screen faces away from the camera: that is
what the visual rule wants, not a product screen; and she may state the price: that is how the line is said,
not a price in the picture.

**The media worker** (`foreman/h3/`: probe, edit, audio, captions, graphics) needs numpy and Pillow,
which gpud's Python does not have, so the core never imports it: it runs
`<ads_options.media_python> -I foreman/h3/worker.py probe|edit h3/<command>-req.json` in the ad's
work directory, in a process group of its own (a cancel stops its ffmpeg too; out of the reach of
launchd ending gpud's group, it watches its parent and stops the same way when gpud is gone, so a
retried step never races it on the same files). `media_python` is resolved once to an absolute path
(a bare name on `PATH`, a relative path from gpud's directory): the preflight checks what the worker
runs. Its last stdout line
is its answer; exit 2 is a content failure, 3 the machine's (an infrastructure failure), anything
else a crash that is retried. On the Studio `media_python` is ComfyUI's venv python.

**Configuration** (`ads_options`): `media_python`, `h3_graph` (`h3_ref2va`), `h3_width` /
`h3_height` (576x1024, multiples of 32), `h3_attempts`, `h3_budget_s`, `h3_music_under`,
`h3_bed_lift`, `h3_crf`, `h3_hands_floor` / `h3_hands_limit`, `h3_anchor_tol_s`, `h3_anchor_gap_s`,
`h3_runon_s`, and `expected.h3_take`, `h3_probe`, `h3_edit`. The facts step stops an h3 ad before any
GPU work when the machine is not ready (`engine_unconfigured`: `media_python` missing or without
numpy and Pillow, `out_width` / `out_height` other than 1080x1920 (the frame the cards, the headline
and the captions are drawn for), the graph template missing, without outputs, a needed placeholder or a license
text, the fonts, character files outside `$FOREMAN_HOME`) or the template or character does not
hold (content: no reference voice, the face or a scene still not approved, a capture or the logo
missing, a box outside its capture, a card's number or price not in the facts).

**The graph and its license.** `examples/ads/comfyui/graphs/h3_ref2va.example.json` is the API
graph of a macOS ComfyUI H3 workflow (the int8 transformer, the turbo 4-step LoRA, Spectrum and
SolAttn), its constant switches folded and unused nodes dropped, with placeholders `{{prompt}}`,
`{{image_1}}` to `{{image_3}}`, `{{audio_1}}`, `{{seed}}`, `{{seconds}}`, `{{width}}`, `{{height}}`
and `{{lora}}`; the graph rounds the frames up to 17k+5 itself. The MiniMax H3 Community License
grants rights only outside its Excluded Territories, which are the European Union, the United
Kingdom, the Republic of Korea and the United States: there, neither the weights nor their outputs
may be used without a separate license from MiniMax. A commercial product or service using it must
display "MiniMax H3", and large commercial users need MiniMax's written authorization. Its Acceptable
Use Policy (Exhibit A) also forbids publishing outputs in a public environment "without clearly and
prominently disclosing" that they are machine-generated (item 12) and using them for false online
engagement such as fake reviews (item 8); the license encourages a "Powered by MiniMax H3" notice on a
product built with it (Section III.3). Whether each platform's AI-content toggle (Foreman never burns a
label into the picture) is a clear and prominent enough disclosure, and where the attribution goes, is
also the owner's call. Foreman does not decide whether the license allows a given use: the owner does,
after reading it. The h3 graph
template must state a `license` (required), may carry an `attribution` (recorded in every take's
sidecar, the manifest and `final/publish.json`), and a commercial template refuses it unless its
wrapper says `commercial_ok: true`, which only the owner sets. The example graph says `commercial_ok:
false`, and the example template, an ad for a paid service, is commercial: the pair is refused at the
facts step until the owner decides. mlx-serve's FastH3 builds stay refused by model id (`studio.FORBIDDEN`); that guard is about
model ids, not this graph.

### Known gaps of the ad pipeline

- Not run against the real servers yet: the request shapes follow MLX-Serve 26.11.1's docs and
  measurements, and the fakes reproduce them. Expected times (`expected`) are placeholders until
  the bake-off measures two-stage lip-sync, image edits and ComfyUI graphs on the Studio. The
  owner asked for MLX-Serve to be upgraded first; that is a change on the Studio and is the first
  item of the live checklist, after which `cli ads models --rescan` re-checks every role's
  capabilities (a renamed capability is reported as `engine_unconfigured`, not as content).
- The new catalog sizes (Qwen3-TTS, Qwen3-ASR with its aligner, FLUX.2 klein, ComfyUI's
  `plan_gb`) are estimates, not measurements. `plan_gb` stays 60 (it must fit gpud's plan beside
  the brain and the owner's default model); a bf16 graph with its text encoders may need more.
- Whether mlx-serve loads the forced aligner as a separate resident model is not verified. If it
  does, an aligner that Foreman's own ASR request loaded is unloaded with the ASR model (one that
  was resident before is never touched); a fourth resident model may still make mlx-serve's LRU
  evict one.
- ComfyUI audio inputs (a talking-shot graph) are uploaded through `/upload/image`; whether the
  installed nodes read them from there is not verified.
- The take score is a heuristic (mouth-region motion against the voice's loudness): it ranks
  takes, it does not measure lip-sync accuracy. The vision brain's character check sees four
  small images.
- The banned-claim, number, mark and URL checks are patterns: they catch the usual phrasings
  (and the red team's), not every paraphrase. Brands not listed in `banned_marks` are only caught
  by the brain's review. A bare spelled-out "one" counts as a number only before a unit or scale
  ("one dollar", "one hundred"). The brain's final review and the owner's verdict are the backstops.
- Facts fetched from a URL are the page's text; prices rendered by JavaScript are not seen (use
  a facts file then).
- The model for each role is configuration (a catalog model), checked against `/v1/models`
  capabilities before every request; gpud does not pick a different model by itself, because it
  only loads catalog models it can account for (`cli ads models` shows the mapping).
- A template's length limit is checked with the default `max_shot_s` (8 s); a deployment that
  raises `ads_options.max_shot_s` is not seen when templates load.
- The disk guard counts an ad's whole estimate while it runs, without subtracting what it has
  already written (conservative).
- The audio gate hears with the same ASR that gives the word times, so a word that ASR always
  mishears looks like a voice error: such a line fails the ad (raise `gate_wer_max`, or add the
  word to the lexicon with a spelling both get right). Its normalization covers the usual forms
  (numbers in words or digits with their unit, ".com", contractions, compounds such as "e-mail"),
  not every one ("nine oh five" is two numbers; a price read "nine fifty" counts only where the
  script states that price), and a lexicon term that is also a common phrase counts wherever the
  phrase is said. A bare count of one is compared as the word "one" (an ASR writes "one minute"
  as "1 minute"), so a voice that drops or changes it is only a word error, unless it says
  another number. The music-masking check compares the ASR's view of the
  mix with its view of the voice track; it does not measure the loudness of the voice against the
  music. Whether MLX-Serve's Qwen3-ASR loads the forced aligner for a plain JSON transcription is
  not verified (the gate unloads it either way), and whether its Kokoro reads `ipa` or `arpabet`
  is not known (they are not sent). The tests use the fake MLX-Serve, whose voice says what a test
  tells it and whose ASR hears what each WAV holds; the gate has not run against the Studio yet.
- The h3 chain has not run end to end through gpud on the Studio yet: its recipe ran by hand (the
  v6 ad), and the tests drive the dispatcher on synthetic files and the worker on synthetic takes.
  `expected.h3_take` (140 s) is the recipe's measured render time; ComfyUI's `plan_gb` (60) is about
  what an H3 take needs beside the resident brains and should be measured through Foreman. The
  judge's thresholds (most of the line heard: 70% of its words; the alignment costs) are judgement
  calls tuned on the recipe's takes. `final/publish.json` leaves the region (`geo`) and every
  platform toggle for a person to fill in.

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
