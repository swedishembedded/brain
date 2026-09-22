<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 74. Fixing a shared-hardware hazard in ONE crate leaves it open in every other crate that touches the same hardware - the fix has to live where all of them can reach it

Lesson #73 fixed `backend-wgpu`: a cross-process `flock` around device
create/destroy plus a bounded wait on the un-cancellable driver call. That
was a correct fix for the crate it lived in and a stress-verified one (20/20
clean). It was also, on its own, an incomplete answer to the actual question,
because `cross_process_gpu_lock` and `bounded` were private functions in
`crates/backend-wgpu/src/lib.rs` and the P40s in this box are not
backend-wgpu's private property.

An audit of every crate that opens real hardware found the same class of
defect sitting open next door, on the SAME cards:

* `crates/vulkan`'s `VkContext::new_inner` called `vkCreateDevice` with **no
  lock of any kind** - not a mutex, not a file lock - and `Drop` destroyed
  the device the same way, on the same physical cards `backend-wgpu` had just
  been taught to serialise. Two of its fence waits were `u64::MAX`, the exact
  unbounded wait `backend-vulkan`'s own `gpu_wait_timeout_ns` doc says makes
  a wedged queue unkillable instead of reported.
* `crates/backend-vulkan`'s `try_new` had no bound on device creation at all,
  and carried a private, character-for-character re-transcription of the
  `BRAIN_GPU_WAIT_S` ladder - a third copy.
* `crates/npu` built a fresh runtime `Core` per session with no lock, while
  also mutating the process-global `LD_LIBRARY_PATH` on the way in.

None of that was discoverable from `backend-wgpu`. There was nothing shared
to reach for, so each crate had independently invented a different partial
answer or none.

The fix is `backend_api::hardware`: one cross-thread AND cross-process
device-init lock, one `BRAIN_GPU_WAIT_S` parse, one bounded-worker helper,
in the one crate every backend already depends on. Two things fell out of
making it shared that a per-crate copy never had to face. First, the lock
must be **re-entrant per thread**: `flock` locks an open file description, so
a second `open` + `LOCK_EX` from the same process blocks against the first
exactly as a foreign process would (verified with a `try_lock` probe:
`WouldBlock`), and any layering at all would self-deadlock without it.
Second, the lock must be **keyed by device class**: the NPU is physically
separate silicon, and a single global lock would have queued every NPU
compile behind an unrelated GPU device creation. Note also where the shared
code could NOT go - `gpu-core` is the facade *above* the backends and depends
on them, so it is not their common ancestor no matter how much it looks like
the "shared GPU crate".

**Rules that follow.** When a defect is "this crate races another process for
a shared external resource", the fix is not done when that crate passes: the
question to answer next is "what else on this machine touches the same
resource, and what is ITS story", and the honest answer is usually "there
isn't one". Grep for the resource (`ash::Entry`, `create_device`, the vendor
runtime's constructor), not for the symptom. And put the corrected primitive
where every current and future toucher can reach it - a private function that
solves the problem correctly for one crate is a template for N divergent
copies, which is how this codebase ended up with three transcriptions of one
timeout ladder and one card protected by two locks that did not know about
each other.
