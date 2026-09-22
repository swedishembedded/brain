<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 73. A thread-only lock does not protect a resource several PROCESSES share, and an un-cancellable FFI call needs a bounded wait even when its own doc already names the hazard

A full `make test` hung three times running, each time on a different real
GPU test (`backend-wgpu::upload_flush`, then `gpu-core::read_staging_reuse`),
past the Makefile's own 2400s deadlock timeout with zero attribution. The
temptation was to call this "a known NVIDIA driver flake" and move on -
`backend-wgpu/src/lib.rs`'s own doc on `DeviceShared` already says "several
concurrent Vulkan devices on a single card deadlocked the test suite roughly
half the time" and a sibling defect (`fastvlm`'s exit segfault) is already
recorded elsewhere as accepted-external on the same driver. That would have
been the wrong call: `/proc/<pid>/task/*/wchan` on the actually-hung process
showed the spinning thread at `wchan=0`, `state=R`, climbing `utime` - a real
userspace wedge, not the `futex_do_wait` "all threads blocked" signature the
existing `init_lock()` Mutex was built to fix, and not a symptom that gets to
be waved off just because the driver has a documented flaky history.

Root cause, found by reading git history for what changed around the
symptom rather than guessing: `init_lock()` is a `std::sync::Mutex`, which
only orders THREADS inside one address space. Every `tests/*.rs` file
compiles to its own OS process, so two test binaries - or two `brain serve`
instances - creating/destroying Vulkan devices on the same physical card had
ZERO coordination between them despite the Mutex's existence. Confirmed by
reproducing the exact hang with `cargo test -p brain-backend-wgpu --release
--tests` run repeatedly (2 of 5 runs hung before any fix).

Fix, in two layers, because the first alone was not enough: (1)
`cross_process_gpu_lock()`, a `flock(2)` on a well-known file acquired
alongside the existing Mutex, extends the same serialisation across every
process on the host - this reduced but did NOT eliminate the hang (still 2/5
under stress), because a lock only guarantees no two holders run at once, it
cannot bound how long the driver's own asynchronous teardown of a
just-exited holder's device takes before the next `vkCreateDevice` is safe.
(2) `bounded()`/`bounded_for()`: device/adapter creation is a synchronous FFI
call into the Vulkan loader and ultimately an `ioctl` into a proprietary
kernel module, which cannot be cancelled mid-flight from Rust - so it runs on
its own thread, and the caller waits on a channel with a timeout instead of
blocking directly, turning a wedge into a named, bounded, reported failure
(`BRAIN_GPU_WAIT_S`) instead of an unattributed infinite hang. 20/20 stress
runs clean with both layers; 2/5 hung with neither.

**Rules that follow.** A `std::sync::Mutex`/`RwLock` protects threads, never
processes - if the resource a lock guards (a physical device, a port, a
file) is reachable from more than one process on the same host, the lock
needs a process-crossing form (`flock`, a named semaphore) or it is
serialising nothing that matters. And: an operation that calls into
proprietary/external code with no cancellation path (an FFI call, a driver
ioctl, a blocking library call with no timeout parameter) needs the SAME
bounded-wait treatment this codebase already gives dispatch/submit/teardown
waits (`gpu_wait_timeout`/`BRAIN_GPU_WAIT_S`) - "we already have a timeout
convention for waits on this resource" is a reason to extend it to a new
call site that lacks one, not a reason to assume every call site already has
it. A pre-existing doc comment describing "this class of bug happens" is a
lead to chase with `/proc`, git history and a stress-repro loop, not a
license to file a new instance of it under the same accepted-external bucket
without checking whether THIS one is actually the same failure.
