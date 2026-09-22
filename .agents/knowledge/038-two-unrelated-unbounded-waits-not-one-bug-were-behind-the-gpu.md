<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 38. Two unrelated unbounded waits, not one bug, were behind "the GPU hangs"

Lesson #37 above named the CPU-side symptom (a roofline probe stuck for
hours) but explicitly left the root cause untracked. Revisiting it: the
kernel itself was innocent (`roof_fma.wgsl` takes its iteration count
through a *uniform*, not a specialization constant, so there is no per-rung
recompile; the first rung is only a few GFLOP, trivially fast on any real
device) - the "hours at near-zero CPU" was always a BLOCK, not slow
arithmetic, confirming #37's own reasoning without yet finding what was
blocking.

Reading `gpu_core::roof`'s three calibration loops
(`measure_compute`/`measure_int8`/`measure_bandwidth`) and both GPU
backends' wait implementations side by side surfaced **two independent
defects**:

1. **The calibration loop itself had no ceiling of any kind** - no
   wall-clock deadline, no total-work budget, only the opt-in
   `BRAIN_NO_ROOF=1` escape hatch #37 already documented. A backend that
   stalls on ANY single dispatch inside that loop (for any reason) blocks
   forever, because nothing bounds how long the loop is willing to wait
   before giving up and reporting "unmeasured."
2. **Both GPU backends' waits were LITERALLY unbounded** -
   `backend-vulkan`'s `wait_for_fences(&[fence], true, u64::MAX)` and
   `backend-wgpu`'s five `poll(PollType::wait_indefinitely())` call sites
   had no timeout and no device-lost handling. A wedged queue therefore
   blocks the calling thread forever BY CONSTRUCTION - this is why past
   hangs presented as unkillable-without-SIGKILL rather than as a reported
   error the caller could catch and act on.

Fix: a wall-clock deadline on every calibration loop (`BRAIN_ROOF_BUDGET_S`,
returning "unmeasured" on expiry - the SAME contract `ensure`'s doc already
promised for an unprobeable device, so no caller needed to change); the
probe now defaults OFF on the CPU device class specifically, since #37's own
root cause was never fully bisected and CPU is the one backend that
reproduced it; both backends' waits now have a finite deadline
(`BRAIN_GPU_WAIT_S`) and report which submit wedged rather than retrying
silently. Validated on real GPU hardware: the full gradcheck suite ran clean
after the fix, zero hangs.

**A separate, related finding surfaced while validating this fix, and later
corrected once a debugger was actually attached.** Running the roofline
test battery hung, and was initially assumed to be a known
concurrent-device-creation deadlock (`gpu_core::testgpu`'s own module doc
warns about a "every thread parked in `futex_do_wait`" failure mode) -
plausible given `gpu.new_like(PROBE_KERNELS)` runs per `measure()` call,
`/proc/<pid>/task/*/wchan` showed `futex_do_wait`, and `BRAIN_GPU_WAIT_S=5`
did not resolve it. That diagnosis was WRONG. The general shape held -
"the GPU hangs" is not always one bug, and `wchan` inspection correctly
ruled OUT the bounded-wait class - but `wchan` alone was not enough to find
the real cause, because a `Mutex::lock()` self-deadlock and a driver worker
thread both park in `futex_do_wait`; distinguishing them needs an actual
backtrace.

Environments that restrict `ptrace` to a process's own parent (`ptrace_scope=1`)
block attaching to an already-running process - but that restriction only
gates non-parent attach; launching the debugger as the process's own parent
(`gdb ./binary`, sending it commands, interrupting the child mid-hang to force
a prompt) sidesteps it entirely and is available even when live-attach is
blocked. The resulting backtrace showed `roof::ensure` holding `CACHE`'s
lock across a call into `measure(gpu)`, and `measure()` unconditionally
calling `gpu.caps()` (to gate the int8 probe), which locks the SAME `CACHE`
via `roof::known()` - `std::sync::Mutex` is not reentrant, so this was a
guaranteed, deterministic, single-threaded deadlock with zero dependency on
concurrency, device count, or the driver at all. It reproduced specifically
on the first test in the roofline suite to call `ensure()` (not `measure()`
directly) against a cold cache, and is masked in almost all ordinary use
because the on-disk persist store is normally already warm from a prior run,
so `ensure()` returns before ever reaching the self-deadlocking branch.
Fixed: `ensure()` now takes and releases `CACHE`'s lock at each access point,
never across `measure()`.

Re-validating Vulkan device sharing once the deadlock no longer masked it
found a REAL, separate, third defect: `VulkanBackend` had never implemented
`Backend::share`/`new_like`/`downgrade` (all default to `None`/no-op), so
`gpu_core::testgpu::dev`'s "one shared device per process" pool never
actually shared anything on Vulkan - every call silently built a whole new
device. Fixing that (proper `Arc`-shared `VkContext`/`VkPipelineSet`, see
`crates/backend-vulkan/src/lib.rs`) then surfaced a FOURTH defect one layer
down: `VulkanBackend`'s command-buffer recording and `VkContext::run_cmd`/
`dispatch` all touch a SHARED `command_pool`/`queue` with no synchronization,
and the Vulkan spec requires host access to both to be externally
synchronized - two threads sharing a device via the newly-working `share()`
reproduced a REAL `ERROR_DEVICE_LOST` within seconds
(`crates/gpu-core/tests/device_sharing.rs::concurrent_shared_handles_do_not_deadlock`,
unreliable before a fix, clean after adding `VkContext::queue_lock`, held
across every allocate-record-submit-wait-free sequence).

**The lesson under the lesson**: `wchan`-only triage can correctly rule OUT
one hypothesis (here, the bounded-wait class) while still pointing at the
WRONG mechanism for what remains - a self-deadlock and a driver stall
produce the identical `futex_do_wait` signature. When a debugger is even
partially available (here: blocked for attach, but not for launch-as-child),
use it before writing down a driver-level conclusion; a wrong "it's the
driver's fault, needs a kernel-level fix" diagnosis can stand undisturbed
for an entire investigation if nothing pushes back on it.
