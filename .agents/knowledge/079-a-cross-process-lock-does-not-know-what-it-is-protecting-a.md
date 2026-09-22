<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 79. A cross-process lock does not know what it is protecting a caller FROM - it protects the caller from every OTHER caller, including ones it has nothing to do with

Lesson #73 gave `backend_api::hardware::device_init_lock` a host-wide
`flock(2)`, reasoning that two PROCESSES creating/destroying Vulkan devices on
one physical card race the same driver hazard two THREADS do. True, and the
fix still made things worse in production: a real deployment started a second
`brain` process targeting a completely different, IDLE card while the first
was mid-generation on its own card, and the second hung - not because the two
processes wanted the same hardware, but because they took the same lock FILE
regardless of which card either of them was touching. One process's ordinary,
long-lived device work made an unrelated process on unrelated silicon wait
with no way for either side to see why, which from the outside reads as
"brain just hangs."

Keying the lock per physical device (PCI bus id, UUID) was the first fix
attempted and is the wrong fix, not an incomplete one: it still assumes brain
should be the thing deciding which processes may touch a GPU at the same
time. It should not. A host that runs several brain processes side by side
already has to know which processes exist and what they want - that is
scheduling, and scheduling belongs to whatever embeds brain, not to a
library function three call sites deep in `WgpuBackend::new`. Brain's own
job is narrower: don't let two THREADS of the SAME process race the same
non-reentrant loader, which is a real, local, fully brain-controlled
invariant no external scheduler can see or fix from outside.

**Fix**: `device_init_lock`/`device_class_lock` is now `std::sync::Mutex`
only - no `flock`, no lock file, no cross-process claim anywhere in the
name or the doc. Every call site (`backend-wgpu`, `crates/vulkan`,
`backend-vulkan`, `crates/npu`) keeps working unchanged, because the
in-process guarantee they actually need was never the part that was wrong.
[`bounded`]/`try_bounded_for` (the wall-clock bound on a call that cannot be
cancelled) is untouched - that mechanism bounds ONE call regardless of what
else is or is not contending for the device, and removing the cross-process
lock does not reduce its value.

**Rule going forward**: a lock's scope must match the scope of the actual
conflict, not the widest scope that would ALSO prevent it. Two callers
racing a non-reentrant loader inside one address space is a `Mutex`
question. Whether two independent processes get simultaneous, safe access
to two independent pieces of hardware is a deployment/orchestration
question, and answering it inside the library removes the answer the
embedder needed to give.
