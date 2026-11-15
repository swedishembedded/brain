# gpu-supervisor - roadmap

Scope: A killable supervisor for un-cancellable device operations.

A killable, supervised owner for the device operations that cannot be
cancelled: the successor to `backend_api::hardware::bounded`'s "time out
and abandon the worker thread" fallback, written down after the
cross-process device-init lock landed (`.agents/knowledge/` #73, #74) and
the question "what would an Erlang/OTP system have done here" was asked and
answered honestly.

**Status: the supervisor, the breaker and their executor integration are
built and verified, and `brain serve` runs its executor with them. The
decision models and the LFM2.5 encoders can be served from a killable
worker (`brain serve --worker-hosted decision,lfm`), which is `brain`
re-executing itself in an internal worker mode; that is off by default. Every
other served model still runs in process under `bounded`.** The table under "Milestones" says what is
done and what is open, with the reason.

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
| `Worker` | `crates/gpu-supervisor/src/worker.rs` | `execve` of a worker program in its own process group with `PR_SET_PDEATHSIG = SIGKILL`; one `AF_UNIX` stream socket pair as its stdin/stdout; one line request and one line answer per call with a fresh id (a stale frame is discarded); while a request is in flight (`exchange`) both sides send frames about it under its id (`progress`, `cancel`, and kinds a `Conversation` agrees on), and a frame may pass descriptors it declares (`ok+<n>`); an explicit deadline per request as a socket read timeout (no helper thread), which the worker's frames renew when the caller asks (`Exchange::renew`); on a missed deadline, end of stream or malformed reply, `SIGKILL` to the group and a bounded reap; a worker still in uninterruptible driver sleep after the bound is handed back, never waited on forever; drop sends the group `SIGKILL` and hands the worker to the process's reaper thread (`brain-gpu-reaper`), so dropping an instance (an eviction on the scheduler's thread) never waits for an exit |
| `Breaker` | `crates/gpu-supervisor/src/breaker.rs` | per-device circuit breaker with time passed in: a fault opens it with an immediate probe (a restart); the third fault within 600 s trips it, after which a probe runs every 300 s and two clean probes close it (`BreakerPolicy`, every value a parameter) |
| `Health` | `crates/gpu-supervisor/src/health.rs` | the registry every supervised call goes through: refuses work for a quarantined device without sending anything, records each fault once, logs the trip once, parks unreaped workers and holds the device's probes back until they are reaped, runs probes in fresh workers |
| executor integration | `crates/residency/src/executor.rs` (`start_supervised`, `assign`, `refuse_quarantined`, `start_probes`, `lane_loop`) | quarantined devices are excluded from placement like busy ones; work no healthy device can ever host is refused once the breaker tripped; a group that faulted its device fails and its claim (budget, slot, instance) is unwound; due probes run on the device's own lane, which is busy meanwhile; an in-process wait past `--gpu-wait-s` (`backend_api::hardware::WAIT_TIMEOUT_MARK`) quarantines the card too, whether it panicked, failed the activation or came back as a job's error |
| `wire` | `crates/gpu-supervisor/src/wire.rs` | frames with descriptors beside them (`SCM_RIGHTS`, close-on-exec, truncation refused), sealed memfds for bulk data (`seal`, `read_sealed`: refuses an unsealed or oversized descriptor), `take_worker_socket` (the worker's end: stdin must be a socket, then stdin reads `/dev/null` and stdout goes to stderr so model code cannot write into the protocol) |
| `SupervisedModel` | `crates/residency/src/supervised.rs` | a resident whose instance is a worker process; it wraps the model's own resident, which answers the manifest, instance key, estimate and job checks in the serving process. Activation sends `hello <PROTOCOL>` then `open <device> <spec>`; each group of jobs `run_batch` is handed is one `batch[+<n>] <action> <k> <jobs>` exchange with the group's blobs in one memfd, each job's progress forwarded to its own caller (renewing the deadline) and each caller's cancellation sent as `cancel <job>`; a single job is a group of one; `run_open` sends the group as `group`, whose worker reaches the executor's `JobFeed` through `take`/`joined`, `done` and `close`/`closed` frames (an action whose worker-side `run_open` declined is remembered and batched directly); an envelope or answer too long for a line moves into a memfd. A lost worker answers every later job of its batch with the same definitive error; a worker's `err panicked:` answer is raised again on the lane, so the group fails and the instance is rebuilt exactly as for an in-process panic, with no quarantine |
| host loop | `crates/residency/src/supervised/host.rs` (`host`, `host_on`), frame JSON in `supervised/codec.rs` | the worker side over any `ResidentModel` a `HostedFamily` rebuilds from a `HostSpec`: refuses (exit 2, nothing served) unless stdin is a socket and the first request is `hello` with this protocol; `open` activates on the named device; `batch` runs the instance's own `run_batch` over the group under `catch_unwind`, so a model that batches batches in the worker; a reader thread registers one cancel token per job as it reads the group and fires only the one a `cancel <job>` names, and registers a joined job's token as it reads the `joined` frame; `group` runs the instance's `run_open` with a `WorkerFeed` whose `take` and `close` are round trips to the parent; the request loop is the socket's only writer |
| worker hosting | `WorkerHosting` (residency), `serving::supervision::worker_hosting`, `catalog::hosted_families`, `crates/cli` (`--worker-hosted`, `--worker-deadline-s`, the `__worker` arm in `main`) | `brain serve --worker-hosted FAMILY` wraps that family's residents in `SupervisedModel`s whose worker is `/proc/self/exe <device flags> __worker`: the server's own executable (an upgrade on disk never mixes versions) with the device flags `select_backend` consumed (`crate::device_flags`). A family no worker can host stops the server at startup. `decisions::resident` (moved out of the catalog) and `catalog::resident_lfm` (scanned encoders, through `ScanSettings::worker_hosting` and `Built::Hostable` in `serving::model_dir`) are the certified families; `residency::on_device` (moved out of the catalog) is how it binds its card in either process |
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

* `cargo test -p brain-gpu-supervisor` (22 integration tests, 1 unit test): a hung request fails
  within its deadline with the worker and its whole process group gone
  from `/proc` and no thread left behind; an abort is reported with its
  signal and reaped; an open breaker refuses work until a probe passes; a
  hung probe is killed and keeps the device out; another device serves
  while one is wedged; an `err` answer is not a fault; a garbled reply is
  a protocol fault; a request that would break the framing is refused
  unsent; dropping a worker reaps it (through the reaper, which takes a
  still-running process without blocking the caller and reaps it, no
  zombie, once it exits); the breaker's restart, trip, window
  and spaced-probe rules; progress frames reach the caller in order and a
  cancel reaches the worker naming the request in flight; an earlier
  request's progress is discarded; an answer declaring descriptors that
  never came is a protocol fault; a frame kind the request's conversation
  knows is answered under the request's id, and without one it is a protocol
  fault; a renewing deadline lets a request that keeps reporting outlive it,
  a fixed one ends the same request as wedged, and a silent worker under a
  renewing one is killed at it; each frame gets its own descriptors and an
  8 MiB blob travels in a memfd; an unsealed or oversized descriptor is
  refused. The fakes are a POSIX shell that hangs in a grandchild or aborts.
* `cargo test -p brain-residency --test it supervision` (9 tests): through
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
  teardown wedges leaves the scheduler answering and placing other work; a
  card that fits a hosted instance but not its worker's device context
  refuses it at once naming the budget, and with a second card that fits
  both the instance is built there.
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
* `cargo test -p brain-gpu-worker` (2 tests, real card, skip with a printed
  reason where no card opens): a quarantined card is readmitted by a real
  probe; the per-request cost (below).
* `cargo test -p brain-residency --test hosted` (10 tests; the test binary is
  its own worker, `harness = false`): an 8 MiB blob and a 2 MiB parameter
  cross in memfds, the caller's scope reaches the worker; three progress
  frames arrive in order and a cancellation fired from a progress callback
  ends a waiting job with `cancelled` while the worker serves on; a worker
  killed with `SIGKILL` mid-job fails that job within the deadline naming
  the signal, quarantines its card, and the next job is built in a new
  worker on the other card; a panic in model code fails the job, records no
  fault and is rebuilt in a new worker; a group of 40 jobs crosses as one
  request whose single `run_batch` call in the worker sees all of it, each
  job's progress, result and blob (more blobs than a frame carries
  descriptors) return under its own index, and the one job asked to cancel
  is cancelled while the others answer; an open group runs in the worker's
  `run_open`: jobs that join while it runs cross as it takes them, each job
  is finished through the parent's feed as it is done, a refused close
  keeps the group going, a late job's cancellation reaches it, and a model
  that only batches closed groups declines having done nothing; a worker
  killed mid-group fails every job the group had not finished (the joined
  one included) once, keeps the finished ones' answers and quarantines its
  card; the jobs waiting behind an open group cross to the worker to be
  screened, and only the one its model can never run is refused through the
  parent's feed, by its ticket and with the model's reason; a job that keeps
  reporting runs
  past a 600 ms run deadline in the same worker, and one that goes silent
  fails at it and quarantines its card; the worker mode refuses a
  non-socket stdin, a first request that is not `hello`, another protocol
  version and an unknown family.
* `cargo test -p brain-residency --test it host_exchange` (2 tests): an
  exhaustive model of one lane's jobs crossing the worker socket with
  progress, caller cancellation, a deadline that progress renews, and a
  worker that may die or wedge, with jobs sent one per request, two to a
  `batch` request, or as an open `group` that up to two late jobs join
  through `take`/`joined`, that screens the jobs still waiting through
  `screen`/`waiting`/`refused` (one late job may never run), finish through
  `done` and end through `close`/`closed` (about 1.3 million states over up
  to three jobs and one fault): every job answered once, `cancelled` only
  when its caller asked and never for a job that ran to completion, no
  cancellation that reached the worker dropped, a deadline kills only a
  silent worker, a job refused only if it can never run and never joined, a
  job that can never run never joining once the worker screened it, no late
  job left waiting for a group that ended healthy, and from every state an
  end is reachable (no cycle starves a job); fifteen
  deliberate bugs are each caught (registering the token when the run
  starts rather than when it is read, clearing it unconditionally after the
  answer, answering `cancelled` when the cancel is sent, freeing the lane at
  end of stream without an answer, a deadline counted from the request, a
  deadline restarted at every poll, a cancel firing every job of its group,
  a token registered for only the group's first job, a joined job's token
  registered when `take` returns rather than when `joined` is read, a fault
  answering finished jobs again, a fault forgetting the joined jobs, a
  worker ending its group on `closed false`, a worker refusing every job it
  is shown, a lane dropping the verdicts, a lane answering a refused job
  and leaving it waiting).
* `cargo test -p brain-cli --test it worker_hosted` (4 tests, the three
  device ones skip without a GPU): the worker mode is absent from `brain
  help` and refused from a terminal; a tiny Laya and a tiny decide
  checkpoint answer in a worker as in process (every number within 1e-5,
  the measured `latency_ms` aside) and the job ran in a worker this process
  started; a worker killed between jobs fails the next job at once and
  records one fault on its card, and once that card is tripped the job
  after is built on the other card; the per-job cost (below).
* `cargo test -p brain-cli --test it worker_hosted_lfm` (4 tests, each
  skips without a GPU, the healthy-card one without two): an LFM2.5 encoder
  batches a group in its worker and answers as in process, a worker killed
  mid-stream fails its group and quarantines its card, a dead worker's next
  job is built on the healthy card, and the per-job, per-group and per-card
  costs (below). `cargo test -p brain-serving --lib
  a_scanned_encoder_runs_in_a_worker`: with `--worker-hosted lfm` a scanned
  encoder's activation goes to the worker command, not to its weights.
* `cargo test -p brain-serving --lib supervision` (3 tests) adds: worker
  hosting re-executes the server with its device flags and `__worker` for
  the families named, and refuses a family no worker can host.

## Milestones

| milestone | status |
|---|---|
| **M1** worker binary with `open`/`probe`; supervised spawn, kill, reap; breaker | done and verified (above) |
| executor lanes honour health; stuck activation or run frees the lane and unwinds the claim | done and verified for supervised instances |
| in-process `bounded` wedges quarantine the card until a supervised probe passes | done (`note_in_process_wedge`); the abandoned thread itself is still not reclaimed, which is what M2 removes |
| the serving process starts its executor with `Supervision` | done: see "`brain serve` supervision" above. Served models still run in process, so what a probe readmits is a card whose in-process waits timed out; the wedged thread itself is reclaimed only by M2 |
| **M2** a served model executing inside the worker | done for the decision models and the LFM2.5 encoders (`--worker-hosted decision,lfm`, off by default): request-level hosting in `brain`'s own worker mode, verified above; what each next family needs is below |
| **M3** measure the per-request round trip against real device work | measured for the protocol, a decision job, an encoder job and an encoder group, and a worker's memory per card (below); per-request hosting passes, per-dispatch proxying does not |
| **M4** make it the default; delete the abandon-the-thread path from `bounded` | open: gated on every served model hosting in a worker; the decision and LFM2.5 families are certified, every other family needs the items below |
| moving a resident off a quarantined card | done: once the card's breaker trips (before that the restart probe is due at once, cheaper than a rebuild), an idle resident a queued job needs is abandoned when a healthy device could hold it (`move_off_quarantined`, `ResidencyManager::abandon_resident`): it leaves residency at once, so the claim builds it afresh on the healthy card, and it is dropped on a thread of its own (`drop_elsewhere`), keeping its memory charged until the drop reports back (`Msg::Dropped`, `release_abandoned`). A failed group's instance (`unwind_failed`, `unwind_failed_multi`) goes the same way, so no teardown through a suspect driver runs on the dispatcher. A teardown that never returns holds one thread and its memory, nothing else. A resident no healthy card can hold is refused while the trip lasts |
| multi-device models under quarantine | done: a multi-device model has one placement (its resident devices, else the ones `estimate_multi` names; `ResidencyManager::multi_placement`), and `refuse_quarantined` refuses its jobs once any device of it is quarantined with a tripped breaker; before a trip they wait for the restart probe like single-device work |
| Windows | open: no `SIGKILL` or process groups; `Worker::spawn` reports unsupported off Unix. `TerminateProcess` and a job object are the equivalent and need a platform seam |

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

### Which program hosts model code

`brain` itself, re-executed in an internal worker mode (`brain <device
flags> __worker`, `serving::supervision::WORKER_MODE`): one binary holds
every model, so no second model-linking binary (about 160 MB in the fast
profile) has to be built, shipped and kept in step, and M4 ends with every
model in a worker anyway. The mode is not a capability or a command; the
CLI rules for it are `.agents/rules/sdk-design.md` section 15.

Health probes stay in `brain-gpu-worker`: a probe has to test the card, not
a model build, and that program opens a card with one backend and a probe
kernel, links no model code and does not change when a model does.

### The first real model: decision models

`decisions::resident` (moved out of the catalog, beside the decisions
crate) registers each checkpoint through `WorkerHosting::serve`, so with
`--worker-hosted decision` its activation starts a worker that rebuilds
the same `DecisionResident` from `{id, dir}` and activates it on the
assigned card through `residency::on_device`; the family model (inventory,
`auto` selection) touches no device and stays in process. The parent keeps
the resident's manifest, instance key, estimate and `check_job` unchanged.

### What a job costs in a worker

`a_worker_hosted_job_costs_a_round_trip_more_than_in_process`
(`crates/cli/tests/it/worker_hosted.rs`), on a P40 with the tiny random
decide checkpoint, the same `decide` job through the executor 60 times
after a warm-up, host load average 70 to 86 on 48 cores, five runs: median
0.94 to 1.29 ms in process, 1.30 to 1.50 ms in a worker, so 0.21 to 0.37 ms
more per job. That is more than the protocol's own 18 us round trip: the
job's envelope and outcome are encoded and parsed once more each, and the
worker hands each request from its reader thread to its request loop, two
thread wake-ups on a loaded host. Against a real model's milliseconds to
seconds of device work per job it is small; against a sub-millisecond job
it is a third. Not yet measured: how much of it is the hand-off (the
request loop could read the socket itself while idle).

### The first batching family: LFM2.5 encoders

`catalog::resident_lfm::hosted` rebuilds an encoder from `{id, weights,
tokenizer, batch}` (`LfmResident::host_spec`); the model-dir scan builds a
scanned encoder as `Built::Hostable`, which `ScanSettings::worker_hosting`
wraps in a `SupervisedModel` when it covers `lfm`. The encoder's own
`run_batch` (equal-length jobs, `--max-batch` per forward, progress after
each forward) runs in the worker on the whole group. An encoder supplied on
demand (`StoreSupplier`, `serving::model_dir::resident_for_local`) is still
built in process: that path takes no hosting yet.

`crates/cli/tests/it/worker_hosted_lfm.rs` (4 tests, P40s): six equal-length
jobs at two per forward answer in a worker as in process (every embedding
value within 1e-5, the same three `batch i/3` progress steps per job, so
one `run_batch` of three forwards in both); a worker killed with `SIGKILL`
at the first progress of a 64-job group (31 forwards still to run) fails
all 64 naming the signal, records one fault and quarantines the card, and
the instance answers later jobs at once; through the executor a worker
killed between jobs fails the next job, and once the card is tripped the
job after is built in a new worker on the other card.

### What an encoder costs in a worker

`a_worker_hosted_encoder_costs_a_round_trip_per_group_and_a_context_per_card`,
on a P40 with the tiny random encoder (`d_model` 64, three layers), host
load average 2 to 5 on 48 cores, eight runs:

* one `embed` job through the executor, median of 60 after a warm-up: 1.59
  to 1.77 ms in process, 1.73 to 1.90 ms in a worker, 0.04 to 0.31 ms more;
* an eight-job group straight on the instance (four forwards, 32 progress
  frames, eight results each with an embeddings blob), median of 30: 4.84
  to 5.50 ms in process, 5.50 to 6.18 ms in a worker, 0.17 to 1.0 ms more.
  The extra cost grows with the frames a group sends (each is a socket
  write, a JSON encode and parse and a thread wake-up on each side), not
  with the jobs' device work.

### What a worker costs per card

The same test reads the cards' used memory (`nvidia-smi`) before and after
a build, with no other work on them, and the worker's resident set: the
first encoder instance in a process that had no card open adds 150 MiB of
device memory (the device context: a second instance beside it adds
nothing measurable at this model size), and the same encoder as a worker
adds 154 MiB, steady over five runs. So a worker costs one device context
per card per worker, about 150 MiB on a P40, that in-process instances of
the same card share, plus about 283 MiB of host memory (its own process:
the `brain` binary's mappings, its runtime and its own copy of what the
model keeps on the host, which for a large checkpoint is its converted
weights). A card that hosts N workers pays N contexts, and placement
charges each: a hosted resident's estimate is its model's plus
`supervised::WORKER_DEVICE_CONTEXT_BYTES` (`MemCost::with_device_context`,
added only where the instance holds device memory), so `hot_cost` and
every placement, eviction plan and release see it. The constant is the
measured context rounded up, and the test above fails when a worker's
context grows past it. The worker's host memory is charged the same way:
the estimate also carries `supervised::WORKER_HOST_BYTES`
(`MemCost::with_process_host`), which the manager charges to the host
budget beside the card (`single_charges`, refunded through the same list
on every eviction, unwind and abandonment) whenever the host is budgeted,
the rule a multi-device model's host bytes follow. A host that cannot
hold it is refused before any eviction (`TooLarge` past the host's
ceiling, else `Starved`: nothing is evicted for host memory, so the
claim fails now and is retried on the next request rather than waiting),
and `plan` reports the same refusal. The constant is the measured
resident set rounded up, and the same test fails when a worker holds
more. What it does not cover: a large checkpoint's host copy in the
worker grows with the model, and a model's own `ram` is not charged for a
GPU-placed instance, in process or hosted.

### What each next family needs

Built for every family: progress frames (a chunk blob included) that renew
the run deadline (`Exchange::renew`: it bounds a job's silence, so a
streaming job of minutes runs on; only the job's own frames renew it, never
a heartbeat thread that would beat on while the job is stuck), real
batching (a group is one `batch` request and the worker's own `run_batch`
runs it, with per-job progress, cancellation and results), late joins
(an open group is one `group` request whose `run_open` reaches the
executor's `JobFeed` through `take`/`joined`, `done` and `close`/`closed`
frames; each
`take` is one round trip, about the protocol's 18 us, paid between decode
steps), screening the jobs waiting to join (`screen` answered `waiting`
with the jobs not screened yet, the worker's instance checks each and
`refused` has the parent's feed turn the doomed ones away at once; one more
round trip per step for a family that screens every step), caller cancellation, blobs in and out through memfds, envelopes and answers of
any size, panics answered as job errors.

* **Adapters.** The adapter registry, bank and release following live in
  the serving process, and every family that serves adapters (the Qwen3 and
  Qwen3.8 decoders, through `llmserve::adapter_registry` and its
  `AdapterSync`) resolves the `stamp`ed selection against that registry
  inside `run_open`. The `stamp` already runs in the parent and the selection
  crosses as a param; what the worker lacks is the registry itself. It needs
  an `attach`/`detach` request pair the parent sends to every live worker of
  the base when the registry changes (with the adapter weights in a memfd or
  a path the worker reads), the worker's own registry rebuilt from those
  requests, and a group refresh (`OpenGroup::Refresh`) triggered by them, so
  a running group picks up a new adapter as it does in process.
* **Multi-device instances.** `MultiDeviceResidentModel` has no hosted form.
  It needs a `HostedFamily` variant that rebuilds a multi-device resident and
  an `open` that names a device list (`open gpu0,gpu1 <spec>`), so one worker
  owns every card of the instance (a device object graph cannot be shared
  between processes); a `SupervisedMultiModel` registered through
  `register_multi`; and fault attribution per card: the worker reports which
  card a failing driver call was on (a `fault <device>` frame before it
  dies, or the device in its panic answer), so only that card is
  quarantined, and a silent death quarantines every card of the instance,
  since it cannot be attributed.
* **Memory accounting per device.** The worker's device context is
  charged as one constant measured on a P40; a card whose driver holds a
  larger context needs it measured there (a probe worker could report its
  own context's footprint at startup). The worker's host memory is one
  constant too, measured with the tiny encoder: a hosted large checkpoint
  keeps its converted weights in the worker, which the constant does not
  cover (the worker could report its resident set after `open`, or the
  estimate could add the model's own `ram`), and a host short of memory
  refuses the claim rather than evicting idle residents to make room.
* **Models supplied on demand.** `resident_for_local` (the `StoreSupplier`
  path that builds a model a request names but the scan did not serve)
  builds in process; it needs the scan's `worker_hosting`.
* **The Qwen3 embedding resident.** It is a stateless `ProviderResident`
  over `QwenProvider`, which loads on first use and keeps the model inside
  the provider; hosting it needs a `HostedFamily` that rebuilds the provider
  from its weights and tokenizer, and its estimate to charge the card.
* **Metrics, demotion.** `Instance::metrics`, `demote` and `promote` are not
  forwarded; a hosted instance reports nothing and is only ever dropped.
