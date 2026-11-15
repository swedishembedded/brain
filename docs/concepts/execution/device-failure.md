# Device failure

A call into a GPU driver (creating a device, waiting on a fence, a dispatch)
cannot be interrupted once it has started. This page describes what brain
does when such a call never returns or the process making it dies, and what
a caller sees.

## Work in the serving process

Device work that runs inside the brain process is bounded, not killable:

- Every wait on a device has a ceiling, `--gpu-wait-s` (30 s by default <!-- perf-number: the flag's default, a configured ceiling not a measurement -->). A
  wait past it fails with an error naming the operation and the flag, instead
  of hanging the process.
- The thread stuck in the driver cannot be killed safely from inside its own
  process, so it is left behind, still holding what it had opened, until the
  driver returns or the process exits.
- The job fails with that error, its device lane is freed and the model's
  claim (slot, instance) is released. The failed instance is torn down in
  the background, so a teardown that blocks in the driver holds up nothing
  else; its memory is counted as used until the teardown finishes. The next
  job for the model builds a fresh instance.

## Device health in `brain serve`

`brain serve` keeps a health record per card. A card whose device wait ran
past `--gpu-wait-s` is quarantined: no new work is placed on it, work goes to
another card that can hold it, and the card is readmitted once a health
probe passes. The probe runs in `brain-gpu-worker`, a separate program that
`brain serve` looks for in its own directory (the release packages install
it beside `brain`; `make build/release` builds both into `target/release/`).
At startup `brain serve` prints one line saying which form it runs in:

- `device supervision: health probes run in <path>`: a probe is a fresh
  worker process opening the card and running one checked dispatch, bounded
  by `--gpu-wait-s`.
- `device supervision without a worker: <why>`: the worker is missing or not
  executable. Cards are still quarantined after a wait timeout, but are
  readmitted on the schedule below without a probe, so a card that keeps
  wedging is kept out after its third fault in ten minutes instead of costing
  every request a full wait.

## Models hosted in a worker

`brain serve --worker-hosted FAMILY` runs every model of that family in a
supervised worker instead of in the serving process: a child process that
owns one card's device and the model's instance on it, and answers the
jobs its card is given: a group a model batches is batched in the worker,
and a model that admits requests into a running batch admits them there.
The worker is `brain` itself, started by the server in an internal mode
with the server's own device flags (`--backend`, `--gpu-wait-s`, ...);
it is not a command and refuses to run when started by hand. A model answers
the same in a worker as in process: its inputs, outputs, progress (streamed
tokens and steps) and cancellation cross the process boundary, and its
images and audio travel in shared memory rather than over the socket.
Its metrics are reported as in process, and a model that can leave its card
without being rebuilt (demoted when another model needs the room) does so
inside its worker. The worker keeps its device context on the card while
the model is demoted, so that context (about 150 MiB on a P40 <!-- perf-number: measured device context of a worker, see the gpu-supervisor ledger -->) stays counted
against the card until the model is promoted back there or dropped.

Two families can be hosted this way: `decision` (the decide and Laya
models) and `lfm` (LFM2.5 encoders served from the model store, on disk at
startup or fetched on demand with `--autofetch`). Every other
family runs in the serving process; naming one in the
flag stops the server at startup with the list of families that can be
hosted. `--worker-deadline-s SECS` (900 by default <!-- perf-number: the flag's default, a configured bound not a measurement -->) bounds how
long a worker may take to load its model, and how long a job may go without
reporting progress: a job that streams tokens or steps runs as long as it
keeps reporting, and one that goes silent for that long counts as wedged.

What a worker changes when something goes wrong:

- **A wedged or crashed driver call ends the worker, not the server.** The
  job it was running fails at once (or at its deadline) with an error naming
  the card, the job and what became of the worker; the worker is killed and
  reaped, so the driver state it held is released, and the card is
  quarantined as described below. The job is not retried.
- **The next job is built in a new worker,** on a healthy card if the
  quarantined one is kept out.
- **A bug in model code is not a device fault.** A model that panics in a
  worker fails its job the way it would in process: its instance is rebuilt
  for the next job and the card is not quarantined.

## Supervised work

Device work in a supervised worker (a model hosted as above, or one an
embedder serves through `residency::supervised::SupervisedModel` on an
executor started with `residency::Executor::start_supervised`) runs in a
child process that owns one card's device and answers one request at a
time. Then:

- **Every request has a deadline.** For a job it bounds the worker's
  silence: each progress report starts it over. A request that misses it, or
  a worker that dies, fails the job within the deadline with an error that names the
  device, the operation, the deadline or the signal, and the worker's
  process id. The job is not retried: a dispatch is not idempotent.
- **The worker is terminated, not abandoned.** Its whole process group is
  sent `SIGKILL` and reaped, so the kernel reclaims every file descriptor,
  mapping and driver context it held. A worker the kernel cannot release yet
  (it is still inside the driver) is polled until it is, and the device is
  not probed before then. A worker never outlives the process that started
  it.
- **The device is quarantined.** No new work is placed on it; work goes to
  another device that can hold it. Other devices keep serving throughout.
  A model loaded on a device whose breaker tripped (below) is moved: its
  next job builds it afresh on a healthy device, and the old copy is torn
  down in the background, its memory still counted until that finishes.
- **A health probe readmits it.** After a first or second fault in ten
  minutes, a probe (a fresh worker opening the device and running one
  checked dispatch) runs at once, and a pass readmits the device. A third
  fault in that window trips the device's circuit breaker: it is probed
  every five minutes and readmitted after two passing probes. While tripped,
  work that no healthy device can hold fails at once with an error saying
  the device is quarantined and why, instead of waiting. A model sharded
  over several cards needs every one of them, so its work fails once any of
  its cards is tripped.
- **A model sharded over several cards has one worker for all of them.**
  When that worker dies, every one of its cards is quarantined, unless the
  model told the worker which card failed: then only that card is, and the
  others keep serving other work.
- **In-process waits count too.** When the executor runs with supervision,
  a device wait past `--gpu-wait-s` in the serving process also quarantines
  that card until a probe in a fresh worker passes.

Supervised workers need a Unix host (Linux): they rely on process groups and
`SIGKILL`.
