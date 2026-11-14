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
| `Worker` | `crates/gpu-supervisor/src/worker.rs` | `execve` of a worker program in its own process group with `PR_SET_PDEATHSIG = SIGKILL`; one `AF_UNIX` stream socket pair as its stdin/stdout; one line request and one line answer per call with a fresh id (a stale answer is discarded); an explicit deadline per request as a socket read timeout (no helper thread); on a missed deadline, end of stream or malformed reply, `SIGKILL` to the group and a bounded reap; a worker still in uninterruptible driver sleep after the bound is handed back, never waited on forever; drop terminates and reaps |
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

* `cargo test -p brain-gpu-supervisor` (14 tests): a hung request fails
  within its deadline with the worker and its whole process group gone
  from `/proc` and no thread left behind; an abort is reported with its
  signal and reaped; an open breaker refuses work until a probe passes; a
  hung probe is killed and keeps the device out; another device serves
  while one is wedged; an `err` answer is not a fault; a garbled reply is
  a protocol fault; a request that would break the framing is refused
  unsent; dropping a worker reaps it; the breaker's restart, trip, window
  and spaced-probe rules. The fakes are a POSIX shell that hangs in a
  grandchild or aborts.
* `cargo test -p brain-residency --test it supervision` (4 tests): through
  the executor, a hung run fails by its deadline while the other card
  serves and the restart probe readmits the device and the claim is
  unwound; a crash in activation keeps the card out (failing probes trip
  the breaker instead of spinning) while work is placed on the other card;
  work only a tripped device could host is refused at once; with no worker
  (`Probe::Elapsed`), a job error carrying the `--gpu-wait-s` mark
  quarantines the card, a restart is readmitted at once, a trip refuses work
  until the spaced readmissions pass.
* `cargo test -p brain-serving --lib supervision` (2 tests): the worker
  beside the server executable runs the probes, bounded by `--gpu-wait-s`; a
  missing or non-executable worker is named and the breaker is kept.
* `bats tests/e2e/deb.bats`: the package ships `/usr/bin/brain-gpu-worker`
  executable beside `/usr/bin/brain`.
* `cargo test -p brain-residency --test it supervisor_model` (2 tests): an
  exhaustive model of lanes, dispatcher, worker and the real `Breaker`
  (about half a million states); six deliberate bugs are each caught.
* `cargo test -p brain-gpu-worker` (2 tests, real card, skip with a printed
  reason where no card opens): a job through the executor and a supervised
  worker dispatches on gpu0 and returns the checked sum; a quarantined card
  is readmitted by a real probe.

## Milestones

| milestone | status |
|---|---|
| **M1** worker binary with `open`/`probe`; supervised spawn, kill, reap; breaker | done and verified (above) |
| executor lanes honour health; stuck activation or run frees the lane and unwinds the claim | done and verified for supervised instances |
| in-process `bounded` wedges quarantine the card until a supervised probe passes | done (`note_in_process_wedge`); the abandoned thread itself is still not reclaimed, which is what M2 removes |
| the serving process starts its executor with `Supervision` | done: see "`brain serve` supervision" above. Served models still run in process, so what a probe readmits is a card whose in-process waits timed out; the wedged thread itself is reclaimed only by M2 |
| **M2** buffers and dispatch behind the protocol for `backend-wgpu` (`Alloc`/`Write`/`Dispatch`/`Read` with a `memfd` mapping for bulk tensors), selected by a flag | open: no model runs in a worker yet; `brain-gpu-worker` serves the probe kernel only |
| **M3** measure the per-dispatch round trip against a real dispatch batch on the target cards | open: gated on M2 |
| **M4** make it the default; delete the abandon-the-thread path from `bounded` | open: gated on M3 |
| moving a resident off a quarantined card | open: its instance would be dropped on the dispatcher thread through a driver that may be the wedged one, so its jobs wait for the restart probe or are refused once the breaker trips |
| multi-device models under quarantine | open: they are excluded from quarantined cards like any group, but `refuse_quarantined` does not refuse them, so behind a tripped breaker they wait for the 300 s probe instead of failing |
| Windows | open: no `SIGKILL` or process groups; `Worker::spawn` reports unsupported off Unix. `TerminateProcess` and a job object are the equivalent and need a platform seam |
