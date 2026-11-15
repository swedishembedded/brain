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
decision models, the LFM2.5 encoders and the Qwen3 decoder with its
adapters can be served from a killable worker (`brain serve --worker-hosted
decision,lfm,qwen3`), which is `brain`
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
* `cargo test -p brain-residency --test hosted` (13 tests; the test binary is
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
  fails at it and quarantines its card; the instance's metrics come back on
  the `open` answer and on every later answer (the worker's, not the
  parent's), a demotion and a promotion run in the worker that holds the
  instance (a demoted one runs nothing, a promoted one is the same instance
  in the same worker), a demotion the model refuses fails without a fault,
  and a promotion onto another card rebuilds the instance there in a new
  worker; the adapters a base offers in the serving process reach its
  worker: one offered before the build routes the first job, one offered
  while an open group runs reaches the worker before the job that joins
  after it and the idle group refreshes its routes for it, a withdrawn one
  and one never offered are answered as no model, and one whose file does
  not hold its digest is refused by the worker; a model across two cards
  runs in one worker that owns both, registered through `register_multi`
  and charged a worker's context on each card, a fault its worker reports
  on one card quarantines that card alone whether the worker then dies or
  answers (the instance is lost), and a death nothing was reported for
  quarantines both; the worker mode refuses a
  non-socket stdin, a first request that is not `hello`, another protocol
  version and an unknown family.
* `cargo test -p brain-residency --lib demotion_that_keeps
  a_warm_instance_that_kept` (2 tests): an instance whose demotion keeps a
  device context on its card (`MemCost::context`) stays Warm on that card
  charged the context and nothing on the host, the eviction planner counts
  a Hot one's demotion as freeing only the rest (counting all of it would
  over-commit the card), evicting the Warm one again drops it, and a Warm
  one is promoted back on its own card when it fits there, before an
  emptier card.
* `cargo test -p brain-residency --test it hosted_adapter` (2 tests): an
  exhaustive model of a hosted base's adapter offered and withdrawn at any
  time while the lane runs open groups that late jobs join, with a worker
  that may die and be replaced (about 22 thousand states over up to three
  changes, two groups, two late jobs and one fault): a group starts with
  the worker's registry at least as new as the lane brought it before
  sending it, a job joins only after every change made before it was handed
  out reached the worker, the worker's registry is always one the parent
  offered and never goes back, and at the end a live worker holds what its
  parent believes it told it; five deliberate bugs are each caught (no
  sync before a group, changes sent after `joined`, a late frame dropped, a
  late frame handed to the request loop, a new worker's feed not started
  over). The fourth is why the reader applies a late frame itself: handed
  to the request loop, a frame about a group that just ended can land after
  a newer frame the reader applied.
* `cargo test -p brain-residency --test it hosted_multi` (2 tests): an
  exhaustive model of fault attribution for a model hosted across two cards
  (about 2.3 thousand states over up to two requests, two reports, one
  worker fault, rebuilds and readmissions): no request is sent while a card
  is quarantined or to a lost instance, once a request ends a card its
  worker blamed is quarantined, a death without a blame quarantines every
  card, and no unblamed card is quarantined when the worker blamed one;
  five deliberate bugs are each caught (a silent death blaming only the
  first card, fault frames ignored, a fault in an answered request not
  recorded, admission asking only the first card, reports sent after the
  answer).
* `cargo test -p brain-gpu-supervisor --test it several_devices` (1 test): a
  worker owning two devices is refused work once either is quarantined, its
  hang is recorded against the device `blame` names (never one it does not
  hold) and its crash against both, and the unreaped worker is held by
  every faulted device.
* `cargo test -p brain-residency --test it hosted_tier` (2 tests): an
  exhaustive model of a hosted instance moving between tiers on two cards
  beside another tenant (about 2.5 thousand states over up to three jobs,
  two demotions and one worker fault): no card holds more than it is
  charged (but for the context of a worker the lane is about to kill, whose
  charge moved with the claim), a job runs only on a Hot instance in a
  worker on the card it was placed on, the parent's metrics are the
  worker's between requests, and at the end every card holds exactly what
  it is charged and no worker outlives its instance; five deliberate bugs
  are each caught (a demoted instance charged to the host, a Warm instance
  handed over Hot without a promotion, a promotion onto another card sent
  to the old worker, the new worker opened before the old one is killed,
  the metrics of a new worker's `open` not kept).
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
* `cargo test -p brain-cli --test it worker_hosted_qwen3` (5 tests, the
  store's `Qwen/Qwen3-0.6B`, each skips without it or a GPU): a batch of
  four greedy generations (exact decode, so a row's tokens do not depend on
  what shares its step) decodes the same token ids, text and streamed deltas
  in a worker as in process; a generation cancelled at its third delta
  finishes `cancelled` after a handful of tokens and the same worker answers
  the next; a worker killed with `SIGKILL` at the fifth streamed token fails
  that generation naming signal 9, records one fault and quarantines the
  card, and the next generation is built in a new worker; an adapter offered
  through `LiveAdapters` while a base generation streams in the worker is
  used by the next request, which decodes the same tokens as in process and
  names the release the serving process offers (id, digest, version), and an
  unoffered adapter name is refused; the per-request, per-token and host
  costs (below). `cargo test -p brain-catalog --lib
  a_resident_rebuilt_from_its_host_spec_sizes_the_identical_engine`: for
  every flag combination the policy tests cover, on a budgeted card, the
  resident rebuilt from its host spec has the original's manifest, estimate
  and spec, while the same configuration resolved again without the budget
  sizes another context; an unknown field, a zero batch and an unknown GEMM
  precision are refused (mutation-checked: a dropped KV pool size and a
  plan that accepts unknown fields are each caught). `cargo test -p
  brain-llmserve --lib
  a_worker_mirror_offers_what_the_serving_registry_offers_under_its_versions`
  (mutation-checked: numbering in the mirror, skipping the digest check).
  `cargo test -p brain-serving --lib
  a_scanned_qwen3_base_and_its_adapters_run_in_a_worker_when_its_family_is_hosted`:
  with `--worker-hosted qwen3` activating a scanned base or an adapter view
  over it goes to the worker, and both are placed as the hosted base.
  Through HTTP on a release binary: `brain serve --worker-hosted qwen3`
  over the model store answers a streamed `/v1/chat/completions` for
  `Qwen/Qwen3-0.6B` from its worker.
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
| **M2** a served model executing inside the worker | done for the decision models, the LFM2.5 encoders and the Qwen3 decoder with its adapters (`--worker-hosted decision,lfm,qwen3`, off by default): request-level hosting in `brain`'s own worker mode, verified above; what each next family needs is below |
| **M3** measure the per-request round trip against real device work | measured for the protocol, a decision job, an encoder job and an encoder group, a worker's memory per card, and Qwen3 generation (time to first token, decode speed, host memory; below); per-request hosting passes, per-dispatch proxying does not |
| **M4** make it the default; delete the abandon-the-thread path from `bounded` | open: gated on every served model hosting in a worker; the decision, LFM2.5 and Qwen3 families are certified, every other family needs the items below |
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
demand (`StoreSupplier`, `serving::model_dir::resident_for_local`) is hosted
the same way: the supplier carries the server's `WorkerHosting`, and
`resident_for_local` serves a `Built::Hostable` through it
(`an_encoder_supplied_on_demand_runs_in_a_worker_when_its_family_is_hosted`
in `brain-serving`, beside the scan's own test).

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

### The first text-generation family: Qwen3

A `QwenResident` is built from one resolved plan (`resident_llm::plan::
QwenPlan`): the serving process resolves the context, the KV pool, the
weight tier and every other choice from the operator's flags and its cards'
budgets, and the host spec carries that plan as JSON (each tier type in its
flag spelling), so the worker builds the engine the server sized without
knowing a budget (`QwenResident::host_spec`, `from_host_spec`, refusing an
unknown field). With `--worker-hosted qwen3` (`QwenServeConfig::
worker_hosting`, which the executor, the model-dir scan and an on-demand
supply set) the resident holds its hosted form itself: a `SupervisedModel`
over a twin built from the same plan and sharing its adapter registry,
through which `estimate` and `activate` go. An `AdapterView`, which places
and builds through its base, is therefore built in the base's worker and
charged as the base, and every concrete handle the adapter flags, the
watcher and the scan hold (`Serving::qwen`, the scan's bases,
`LiveAdapters`) stays the one resident. The registry is the hosted base's
`AdapterSource`; in the worker `catalog::resident_llm::hosted` rebuilds the
resident and mirrors what it is offered into the resident's own registry
(`llmserve::adapter_registry::AdapterMirror`: the file is read, bytes that
are not the offered digest are refused, the serving process's version is
kept, so a reply names the release as it would in process; the attach
request carries that version, protocol 5).

### What a generation costs in a worker

`a_worker_hosted_qwen3_costs_a_round_trip_per_request_and_little_per_token`,
`Qwen/Qwen3-0.6B` at fp32 weights and int8 KV, context 2048, exact decode,
on one P40, host load average 4 to 15 on 48 cores, six runs:

* one stream of 64 tokens: time to first token (median of three) 57.6 to
  61.4 ms in process, 58.2 to 58.5 ms in a worker; decode 97.2 to 97.6
  tokens/s in process, 96.9 to 97.2 in a worker. A worker adds about 0.5 ms
  per request and about 0.05 ms per token (0.5 percent): one progress frame
  per token, against about 10 ms of device work per token.
* a batch of eight streams of 64 tokens: decode 536 to 627 tokens/s in
  process and 530 to 625 in a worker, median time to first token 131 to
  230 ms and 141 to 227 ms; the spread between runs (when the eight join the
  open group) is larger than any difference between the two.
* the worker's resident set is 370 MiB, of which 126 MiB is its own
  (anonymous and shared memory) and the rest file pages of the `brain`
  executable and the driver's libraries, which the serving process maps
  too; placement charges `WORKER_HOST_BYTES` (320 MiB), which the test
  requires to cover the worker's own memory. The weights are uploaded tile
  by tile and not kept on the host, so this does not grow with the
  checkpoint the way an encoder's host copy would.

What is not covered yet for Qwen3: the base digest a reply names is
looked up again in each worker (cached on disk by file identity, so usually
at once); a reply before it is known names none, as one in process does
before the server's own lookup finishes.

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
any size, panics answered as job errors, adapters: a base hosted with an
`AdapterSource` (`SupervisedModel::with_adapters`) tells each of its
workers every change of the adapters the serving process offers, as
`attach {id, digest, path}`/`detach <id>` requests before a group
(`sync_adapters`) and as frames while one runs, sent at each poll and
before the reply to each `take` (`JobsTalk::adapter_changes`), so a job that
joins after a change never reaches the worker first; the worker's reader
applies every adapter frame about a request it already read at once (in
the order they were sent) to the `AdapterSink` its family's `HostedModel`
returned, and the model refreshes its routes from it as it does in process.
A worker is told lazily, before it next runs anything, not the moment the
registry changes: a worker that runs nothing needs no adapter, and no
thread has to push to workers that may be busy. A new worker of an instance
is told everything again (`AdapterFeed::reset`). Then multi-device instances: a `HostedFamily` returns `HostedModel::multi` for a
`MultiDeviceResidentModel`, `SupervisedMultiModel` (registered through
`register_multi`, `WorkerHosting::serve_multi`) charges each card its
model's share plus a worker's device context
(`MultiDeviceCost::with_device_context`), and its activation opens every
card of the placement in one worker (`open gpu0,gpu1 <spec>`); faults are
attributed per card: model code in a worker calls `report_device_fault`
for the card a failing driver call was on, the worker sends it as a
`<id> fault <device> <why>` frame before its next frame, and the parent
quarantines that card alone (`Health::exchange_on` with the devices the
frames named; `Blamed::record` when the worker answers anyway, which loses
the instance), while a death no card was named for quarantines every card
the worker held, and work is admitted only while every card admits it. And metrics and tiers: every
answer carries the instance's `metrics` after the request, which the parent
keeps (the dispatcher's poll never reaches a worker); `demote warm|cold`
runs the instance's own demotion in its worker, bounded by the run deadline
because the dispatcher waits for it (an in-process demotion it waits for
unbounded); `promote <device>` runs its promotion there when the card is
the worker's, and onto another card the old worker is killed first and a
new one opened. A worker cannot give its device context back while it
lives, so a demoted hosted instance stays Warm on its card charged that
context (`MemCost::context`, set by `SupervisedModel::estimate` only for a
model that demotes), the planner counts a Hot one's demotion as freeing
only the rest, and a promotion goes back to that card first.

* **A multi-device family.** The hosting is built (above); no served
  multi-device family is hosted yet. Each needs its `HostedFamily` returning
  `HostedModel::multi`, its registration through
  `WorkerHosting::serve_multi`, and, for per-card attribution, a call to
  `report_device_fault` where its lane sees a driver call fail on one card
  (a `--gpu-wait-s` timeout of that card's stage); without one, a fault
  quarantines every card of the instance, as in process.
* **Memory accounting per device.** The worker's device context is
  charged as one constant measured on a P40; a card whose driver holds a
  larger context needs it measured there (a probe worker could report its
  own context's footprint at startup). The worker's host memory is one
  constant too, measured with the tiny encoder: a hosted large checkpoint
  keeps its converted weights in the worker, which the constant does not
  cover (the worker could report its resident set after `open`, or the
  estimate could add the model's own `ram`), and a host short of memory
  refuses the claim rather than evicting idle residents to make room.
* **The Qwen3 embedding resident.** It is a stateless `ProviderResident`
  over `QwenProvider`, which loads on first use and keeps the model inside
  the provider; hosting it needs a `HostedFamily` that rebuilds the provider
  from its weights and tokenizer, and its estimate to charge the card.
