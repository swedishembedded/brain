# gpu-supervisor - roadmap

A killable, supervised owner for the device operations that cannot be
cancelled. This is the designed successor to
`backend_api::hardware::bounded`'s "time out and abandon the worker thread"
fallback, written down after the cross-process device-init lock landed
(`.agents/rules/lessons.md` #73, #74) and the question "what would an
Erlang/OTP system have done here" was asked and answered honestly.

**Status: designed, not implemented.** Nothing in this file is code today.
What shipped instead is the smaller, complete fix it builds on: one shared
cross-thread device-init lock plus one shared wall-clock bound, in
`backend_api::hardware`, used by every crate that opens a device.

The cross-process half of that lock was reverted (`.agents/rules/lessons.md`
#79): a host-wide lock made one process's ordinary device work stall an
unrelated process on unrelated, idle hardware. `device_init_lock` is
in-process only now. Any future revival of cross-process coordination -
including the per-card worker design below - has to be scoped so that a
process touching card A is never blocked by a process touching card B; the
design below predates that finding and its cross-process serialisation
claims (the child worker sharing `device_init_lock` "against any process
that has not been migrated yet") need re-deriving against it, not assuming
it still holds.

---

## What is actually still broken

`bounded` runs the un-cancellable call on its own thread and gives up on it
after `BRAIN_GPU_WAIT_S`. That converts an unattributed infinite hang into a
named, timed failure, which is a real improvement and is why it shipped. It
does **not** reclaim anything:

* The abandoned thread is still inside the driver, still holding whatever
  file descriptors, mappings and driver contexts the call had opened.
* It still holds the device-init lock it acquired, so a wedge can starve
  device creation for every OTHER THREAD of that same process until it
  exits. This no longer crosses processes (`device_init_lock` is in-process
  only, see the status note above), but it is still real within one
  process.
* There is no safe way to fix this from inside the process. Forcibly killing
  a thread mid-`ioctl` leaves the allocator and the driver's own bookkeeping
  in states that do not agree with each other, which is the same class of
  corruption `DeviceShared::drop` already refuses to risk (it leaks a faulted
  device rather than destroying it).

Three properties are missing, and OTP names all three:

1. **Exclusive ownership instead of shared access behind a lock.** Today N
   call sites each touch the driver, correctly ordered by a lock. A
   `gen_server` owns the resource outright; everyone else sends it a message.
   There is no lock to get right because there is nothing to race.
2. **A genuinely killable unit for the foreign call.** A Port or a dirty
   scheduler is an OS-level thing the VM can tear down. `SIGKILL` on a child
   process unconditionally reclaims every fd, mapping and driver context that
   child held - the kernel does it, the wedged code does not have to
   cooperate. That is strictly stronger than abandoning a thread.
3. **A supervisor with an explicit restart and give-up policy**, rather than
   a timeout constant re-invented per crate by whoever hit the problem next.

---

## Why the obvious shape does not work

"Create the device in a child process and hand the handle back" is not
available. A `VkDevice` / `wgpu::Device` is a userspace object graph in the
creating process's address space: a pointer table into a `dlopen`ed ICD, with
per-process driver state behind it. There is no `SCM_RIGHTS`-style transfer
for it. Whatever process calls `vkCreateDevice` is the only process that can
use the result.

So the child either owns the whole device lifetime, or the split buys nothing.

---

## Design: `brain-gpu-worker`

One helper binary per physical card, spawned by the parent, owning that
card's device for its whole lifetime. The parent holds no device handle at
all; it holds a `WorkerHandle` and speaks a request/response protocol.

### Process lifecycle

* Spawned lazily on first use of a card, `execve` of a dedicated binary (not
  a `fork` of the parent, and not a re-exec of the current executable - a
  test binary re-executing itself would re-run the test suite).
* One `SOCK_SEQPACKET` `AF_UNIX` socketpair, inherited on fd 3. Datagram
  framing, so a partial write can never desynchronise the stream, and
  `SOCK_SEQPACKET` closes on peer death, which is the parent's liveness
  signal.
* Bulk tensor payloads go through a `memfd` shared mapping passed once at
  startup, not through the socket: the protocol carries offsets and lengths,
  never megabytes.
* The child takes the same `backend_api::hardware::device_init_lock` around
  its own device creation, guarding its own threads exactly as any other
  brain process would. It does NOT serialise against other worker
  processes or other parents - per the status note above, cross-process
  device-creation ordering is deliberately not this lock's job, and a
  design that reintroduces it has to solve the same-card-vs-different-card
  distinction this roadmap did not originally account for.

### Protocol

Request/response, one in flight per worker, every request carrying a
monotonic id so a late reply from a restarted worker is discarded rather than
mistaken for the current one.

| message | direction | payload |
|---|---|---|
| `Open { card }` | to worker | canonical device identity |
| `Opened { caps }` | from worker | the `backend_api` capability record |
| `Compile { kernels }` | to worker | name + WGSL per kernel |
| `Alloc { bytes, usage }` / `Freed { id }` | both | buffer ids, never pointers |
| `Write { id, offset, memfd_range }` | to worker | offsets into the shared mapping |
| `Dispatch { batch }` | to worker | the existing `Step` list, ids substituted for handles |
| `Read { id, range }` | to worker | result written back into the mapping |
| `Poll { deadline }` | to worker | the bounded fence wait |
| `Wedged { what }` | from worker | the child noticed its own timeout first |

Every message is data. Nothing that only means something inside one address
space crosses the boundary.

### Supervision

* Each request has a deadline (the existing `BRAIN_GPU_WAIT_S` ladder).
* Deadline exceeded: `SIGKILL` the worker, `waitpid` it, and **the kernel
  reclaims the driver context** - the property the thread version cannot
  offer. The in-process device-init lock the child held is released the
  instant the process is gone, same as any other process exit.
* Restart policy, per card: restart on the first two wedges within 600s,
  replaying `Open` + `Compile` (both are pure functions of data the parent
  still holds; in-flight buffer contents are lost and the request that
  triggered the restart is failed, not retried, because a compute dispatch is
  not idempotent from the caller's point of view).
* **Circuit breaker**: a third wedge inside the same 600s window marks that
  card offline in `gpu_core::devices`. Placement stops proposing it, the
  breaker logs one line naming the card and the window, and a half-open probe
  (one `Open` + one trivial dispatch) is attempted every 300s. Two clean
  probes close the breaker. Without this, a card whose driver is genuinely
  wedged costs every subsequent request a full `BRAIN_GPU_WAIT_S` before
  failing - the failure is bounded but the cumulative cost is not.

### What it would cost

The parent-side `Backend` impl is mechanical (the trait is already ids and
`Step` lists, not pointers). The expensive parts are: a new binary and its
build/packaging, the shared-mapping allocator, and the fact that every
existing test that builds a `Gpu` would start depending on a child process
being spawnable from the test environment. That last point is the real risk
and the reason this is a roadmap item and not a patch.

---

## Milestones

* **M1** - `brain-gpu-worker` binary; `Open`/`Opened`/`Compile` only. Parent
  still creates its own device for real work; the worker is used solely as a
  preflight probe under supervision. Proves the spawn/kill/reap loop and the
  breaker with no change to how compute runs.
* **M2** - buffers and dispatch move behind the protocol for ONE backend
  (`backend-wgpu`), selected by an opt-in env knob so the default path is
  untouched while it is measured.
* **M3** - measure. A per-dispatch round trip over a `SEQPACKET` socket is
  order-microseconds and a real dispatch batch is order-milliseconds, but
  that has to be shown on this hardware, not assumed. Gate M4 on it.
* **M4** - make it the default; delete the abandon-the-thread path from
  `backend_api::hardware::bounded` and leave only the supervised form.

## Open questions

* Does the breaker belong in `backend_api::hardware` (next to the lock, so
  every backend shares one policy) or in `gpu_core::devices` (next to
  placement, which is what has to act on it)? Probably the former for the
  state and the latter for the decision.
* One worker per card, or one per (card, kernel set)? Pipeline compilation is
  the slow part of `Open`, and residency already keys on kernel sets.
* Windows has no `SIGKILL`; `TerminateProcess` is the equivalent and needs a
  small platform seam. (`device_init_lock` itself is a plain
  `std::sync::Mutex` now, so no cross-platform file-lock equivalent is
  needed for it.)
