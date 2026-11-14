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

## Supervised work

Device work can also run in a supervised worker: a child process
(`brain-gpu-worker`) that owns one card's device and answers one request at a
time. An embedder starts the executor with supervision
(`residency::Executor::start_supervised`) and serves the model through
`residency::supervised::SupervisedModel`. Then:

- **Every request has a deadline.** A request that misses it, or a worker
  that dies, fails the job within the deadline with an error that names the
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
- **In-process waits count too.** When the executor runs with supervision,
  a device wait past `--gpu-wait-s` in the serving process also quarantines
  that card until a probe in a fresh worker passes.

Supervised workers need a Unix host (Linux): they rely on process groups and
`SIGKILL`.
