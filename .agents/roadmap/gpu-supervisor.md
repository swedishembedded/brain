# gpu-supervisor - roadmap

Scope: A killable supervisor for un-cancellable device operations.

A killable, supervised owner for the device operations that cannot be
cancelled: the successor to `backend_api::hardware::bounded`'s "time out
and abandon the worker thread" fallback, written down after the
cross-process device-init lock landed (`.agents/knowledge/` #73, #74) and
the question "what would an Erlang/OTP system have done here" was asked and
answered honestly.

**Status: the supervisor, the breaker and their executor integration are
built and verified, and `brain serve` runs its executor with them; one
device path runs behind them end to end. In-process device work (every
served model today) still runs under `bounded`.** The table
under "Milestones" says what is done and what is open, with the reason.

The cross-process half of the device-init lock was reverted
(`.agents/knowledge/` #79): a host-wide lock made one process's ordinary
device work stall an unrelated process on unrelated, idle hardware.
`device_init_lock` is in-process only. The worker below does not
reintroduce cross-process serialisation: each worker takes the in-process
lock for its own threads only, so a process touching card A is never
blocked by a process touching card B.

---

## What was broken, and what is fixed

`bounded` runs the un-cancellable call on its own thread and gives up on it
after `--gpu-wait-s`. That converts an unattributed infinite hang into a
named, timed failure. It does **not** reclaim anything: the abandoned
thread is still inside the driver, holding its file descriptors, mappings
and driver contexts, and there is no safe way to kill a thread mid-`ioctl`
from inside its own process.

Three properties were missing, and OTP names all three:

1. **Exclusive ownership instead of shared access behind a lock.** Built:
   a supervised instance owns its worker process outright, and a device's
   lane is the only thread that talks to it.
2. **A genuinely killable unit for the foreign call.** Built: a child
   process, killed with `SIGKILL` to its whole process group and reaped, so
   the kernel reclaims every fd, mapping and driver context it held.
3. **A supervisor with an explicit restart and give-up policy.** Built: the
   per-device circuit breaker, shared by every caller through one registry.

Why the device cannot simply be created in a child and handed back: a
`VkDevice` / `wgpu::Device` is a userspace object graph in the creating
process's address space, with no `SCM_RIGHTS`-style transfer. Whatever
process creates the device is the only one that can use it, so the worker
owns the whole device lifetime and the parent holds only a handle to the
worker.

---

## What is built

| piece | where | what it does |
|---|---|---|
| `Worker` | `crates/gpu-supervisor/src/worker.rs` | `execve` of a worker program in its own process group with `PR_SET_PDEATHSIG = SIGKILL`; one `AF_UNIX` stream socket pair as its stdin/stdout; one line request and one line answer per call with a fresh id (a stale answer is discarded); an explicit deadline per request as a socket read timeout (no helper thread); on a missed deadline, end of stream or malformed reply, `SIGKILL` to the group and a bounded reap; a worker still in uninterruptible driver sleep after the bound is handed back, never waited on forever; drop sends the group `SIGKILL` and hands the worker to the process's reaper thread (`brain-gpu-reaper`), so dropping an instance (an eviction on the scheduler's thread) never waits for an exit |
| `Breaker` | `crates/gpu-supervisor/src/breaker.rs` | per-device circuit breaker with time passed in: a fault opens it with an immediate probe (a restart); the third fault within 600 s trips it, after which a probe runs every 300 s and two clean probes close it (`BreakerPolicy`, every value a parameter) |
| `Health` | `crates/gpu-supervisor/src/health.rs` | the registry every supervised call goes through: refuses work for a quarantined device without sending anything, records each fault once, logs the trip once, parks unreaped workers and holds the device's probes back until they are reaped, runs probes in fresh workers |
| executor integration | `crates/residency/src/executor.rs` (`start_supervised`, `assign`, `refuse_quarantined`, `start_probes`, `lane_loop`) | quarantined devices are excluded from placement like busy ones; work no healthy device can ever host is refused once the breaker tripped; a group that faulted its device fails and its claim (budget, slot, instance) is unwound; due probes run on the device's own lane, which is busy meanwhile; an in-process wait past `--gpu-wait-s` (`backend_api::hardware::WAIT_TIMEOUT_MARK`) quarantines the card too, whether it panicked, failed the activation or came back as a job's error |
| `SupervisedModel` | `crates/residency/src/supervised.rs` | a resident model whose instance is a worker process: `open <device>` at activation, `run <action> <params>` per job, `probe <device>` from the executor; a lost worker answers every later job of its batch with the same definitive error |
| `brain-gpu-worker` | `crates/gpu-worker` | the real worker: opens card `gpu<i>` through `backend-wgpu`, compiles the probe kernel, and answers `probe` with a real dispatch whose result it checks |
| `brain serve` supervision | `crates/serving/src/supervision.rs`, `ServeKnobs::supervision`, `crates/cli` `serve_supervision` | the serving executor runs with `Supervision`; probes run in the `brain-gpu-worker` found beside the `brain` executable (bounded by `--gpu-wait-s`); a missing or non-executable worker is named at startup and probes become `Probe::Elapsed` (the breaker paces readmission without touching the card); shipped in the Linux release artifacts and the `.deb` (`/usr/bin/brain-gpu-worker`) |

Choices the design above left open, and what was chosen:

* **Transport.** A stream socket pair with newline framing, not
  `SOCK_SEQPACKET`: std offers only stream and datagram pairs, a stream
  gives end of stream on peer death (a datagram pair does not), and with
  one writer per direction a frame cannot interleave. Bounded replies
  (`MAX_REPLY_BYTES`) keep a broken worker from growing the buffer.
* **Where the breaker lives.** State in `gpu-supervisor` (layer 1, beside
  `backend-api`, so every backend can share one policy); the placement
  decision in the residency executor, which owns placement.
* **One worker per card or per instance.** Per instance (one card and one
  kernel set), owned by the instance; residency keys on instances, and the
  lane already serialises a card's work.
* **Restart.** A fault below the trip count makes the probe due at once;
  the probe is a fresh worker opening the device, so a passing probe is
  the restart. The faulted request is failed, never retried.

## Verification

* `cargo test -p brain-gpu-supervisor` (14 integration tests, 1 unit test): a hung request fails
  within its deadline with the worker and its whole process group gone
  from `/proc` and no thread left behind; an abort is reported with its
  signal and reaped; an open breaker refuses work until a probe passes; a
  hung probe is killed and keeps the device out; another device serves
  while one is wedged; an `err` answer is not a fault; a garbled reply is
  a protocol fault; a request that would break the framing is refused
  unsent; dropping a worker reaps it (through the reaper, which takes a
  still-running process without blocking the caller and reaps it, no
  zombie, once it exits); the breaker's restart, trip, window
  and spaced-probe rules. The fakes are a POSIX shell that hangs in a
  grandchild or aborts.
* `cargo test -p brain-residency --test it supervision` (7 tests): through
  the executor, a hung run fails by its deadline while the other card
  serves and the restart probe readmits the device and the claim is
  unwound; a crash in activation keeps the card out (failing probes trip
  the breaker instead of spinning) while work is placed on the other card;
  work only a tripped device could host is refused at once; with no worker
  (`Probe::Elapsed`), a job error carrying the `--gpu-wait-s` mark
  quarantines the card, a restart is readmitted at once, a trip refuses work
  until the spaced readmissions pass; a multi-device model whose placement
  has a tripped card is refused at once while single-device work keeps
  running on the healthy card; a resident on a tripped card whose teardown
  never finishes is moved: the next job is served on the healthy card, the
  old instance is dropped on neither the dispatcher nor a lane, and its
  memory stays charged until its teardown ends; a faulted instance whose
  teardown wedges leaves the scheduler answering and placing other work.
* `cargo test -p brain-serving --lib supervision` (2 tests): the worker
  beside the server executable runs the probes, bounded by `--gpu-wait-s`; a
  missing or non-executable worker is named and the breaker is kept.
* `bats tests/e2e/deb.bats`: the package ships `/usr/bin/brain-gpu-worker`
  executable beside `/usr/bin/brain`.
* `cargo test -p brain-residency --test it supervisor_model` (2 tests): an
  exhaustive model of lanes, dispatcher, worker, the real `Breaker` and a
  resident instance that must leave a tripped card, with teardowns that may
  wedge (about 1.4 million states); nine deliberate bugs are each caught,
  among them never moving the resident, releasing its memory before its
  teardown ends, and tearing it down on the dispatcher.
* `cargo test -p brain-gpu-worker` (3 tests, real card, skip with a printed
  reason where no card opens): a job through the executor and a supervised
  worker dispatches on gpu0 and returns the checked sum; a quarantined card
  is readmitted by a real probe; the per-request cost (below).

## Milestones

| milestone | status |
|---|---|
| **M1** worker binary with `open`/`probe`; supervised spawn, kill, reap; breaker | done and verified (above) |
| executor lanes honour health; stuck activation or run frees the lane and unwinds the claim | done and verified for supervised instances |
| in-process `bounded` wedges quarantine the card until a supervised probe passes | done (`note_in_process_wedge`); the abandoned thread itself is still not reclaimed, which is what M2 removes |
| the serving process starts its executor with `Supervision` | done: see "`brain serve` supervision" above. Served models still run in process, so what a probe readmits is a card whose in-process waits timed out; the wedged thread itself is reclaimed only by M2 |
| **M2** a served model executing inside the worker | open, redesigned below (request-level hosting); not started, because the first step is a decision about which program hosts model code |
| **M3** measure the per-request round trip against real device work | measured for the protocol (below); per-request hosting passes, per-dispatch proxying does not |
| **M4** make it the default; delete the abandon-the-thread path from `bounded` | open: gated on every served model hosting in a worker |

## Models executing inside the worker

### What the round trip costs

`a_supervised_request_costs_a_socket_round_trip_on_top_of_its_dispatch`
(`crates/gpu-worker/tests/it/main.rs`), on a P40 under a host load of about
130 on 48 cores: a request the worker answers without device work has a
median round trip of 18 us (2000 requests); one carrying a dispatch of the
probe kernel, its buffer creation and a 16-byte read back takes 148 us
(200 requests). The test prints both and bounds the empty round trip at
5 ms.

### Request-level hosting, not a proxied backend

The design above put `Alloc`/`Write`/`Dispatch`/`Read` behind the protocol,
one round trip per dispatch. A decode step is hundreds of dispatches, so
at 18 us each that adds milliseconds per token: per-dispatch proxying fails
its own M3 gate. The worker instead hosts the whole model, and the protocol
carries jobs, as `SupervisedModel` already does (`open`, `run <action>
<params>`): one round trip per job instead of one per dispatch, which
against a job of milliseconds of device work is noise. Device memory then never crosses the boundary; only a job's
inputs and outputs do, and those are JSON except for blobs.

### The first real model, and what blocks it

The smallest real served path is the decision models
(`catalog::resident_decision`): JSON in and out, no blobs, no streaming,
serial batches, small checkpoints. Moving them needs, in
order:

1. **Which program hosts model code.** The worker must link the model.
   `brain-gpu-worker` is a layer-2 crate that links one backend; linking a
   model family makes it a model-layer binary, and linking the catalog makes
   it a second copy of `brain` (about 160 MB in the fast profile). The
   options are `brain` re-executing itself in a worker mode (one binary
   holds every model; it needs an internal entry point that is not a
   capability, which the CLI rules do not cover yet), a model-linking worker
   shipped beside `brain`, or one worker per family. Choosing is the first
   step; the recommendation is the first, since M4 ends with every model in
   a worker.
2. **A generic host loop** (`residency::supervised`, the worker side of the
   protocol over any `ResidentModel`): `open <device>` activates on that
   device, `run` calls `Instance::run`, `probe` opens and checks the device.
3. **Activation outside the catalog.** `DecisionResident` and the
   `on_device` scoped device selection it activates through live in
   `crates/catalog`; the host needs them below the catalog (`on_device` in
   `residency`, the resident beside the decisions crate).
4. **A serve flag** choosing worker hosting for a family, and the parent's
   `SupervisedModel` carrying that family's manifest, instance key, estimate
   and `check_job` unchanged.

What a decision model does not need but every other family does, and so
blocks M4: progress frames (streaming tokens: an `<id> progress <json>`
line the parent forwards while the answer is pending), cancellation (a
request is in flight alone, so a cancel needs a signal or a second socket),
late admission into a running group (`run_open`'s `JobFeed` across the
socket), blobs (images, audio: the `memfd` mapping above, now per job
rather than per buffer), adapters attached while serving (the adapter
registry lives in the parent), multi-device instances (several cards in one
worker), and memory: each worker holds its own device context and its own
host copy of converted weights, which in-process instances share; that cost
per card has to be measured before more than one family moves.
| moving a resident off a quarantined card | done: once the card's breaker trips (before that the restart probe is due at once, cheaper than a rebuild), an idle resident a queued job needs is abandoned when a healthy device could hold it (`move_off_quarantined`, `ResidencyManager::abandon_resident`): it leaves residency at once, so the claim builds it afresh on the healthy card, and it is dropped on a thread of its own (`drop_elsewhere`), keeping its memory charged until the drop reports back (`Msg::Dropped`, `release_abandoned`). A failed group's instance (`unwind_failed`, `unwind_failed_multi`) goes the same way, so no teardown through a suspect driver runs on the dispatcher. A teardown that never returns holds one thread and its memory, nothing else. A resident no healthy card can hold is refused while the trip lasts |
| multi-device models under quarantine | done: a multi-device model has one placement (its resident devices, else the ones `estimate_multi` names; `ResidencyManager::multi_placement`), and `refuse_quarantined` refuses its jobs once any device of it is quarantined with a tripped breaker; before a trip they wait for the restart probe like single-device work |
| Windows | open: no `SIGKILL` or process groups; `Worker::spawn` reports unsupported off Unix. `TerminateProcess` and a job object are the equivalent and need a platform seam |
