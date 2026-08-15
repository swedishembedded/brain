# backend-vulkan - roadmap

## Fixed (partially - see below): `kernel_timing` test hangs under a full `make test` run, not reproducible standalone

`crates/backend-vulkan/tests/kernel_timing.rs` (3 small tests - largest
buffer is 1024 f32 elements, largest dispatch count is 8) hung indefinitely
during a full `make test` run on this box: still running past 26 minutes,
one thread pinned at 100% CPU (consistent with a busy-poll `poll_wait()` on a
GPU fence that never signals) while `nvidia-smi` showed both P40s at 0%
utilization and near-zero memory - the GPU itself was not doing real work,
so this reads as a wedged wait, not a slow one. Killed manually (SIGKILL) to
unblock the gate; `make test`'s own 2400s timeout would eventually have
caught it too ("TIMED OUT ... almost certainly a deadlock").

**Not reproducible in isolation**: `cargo test -p brain-backend-vulkan --test
kernel_timing -- --test-threads=1` and the same with `--test-threads=8` both
pass cleanly in under a second, every time tried. The hang only showed up
inside the full `make test` invocation (`cargo test --lib --bins --tests`
across the whole workspace), which runs many test *binaries* - separate OS
processes - concurrently. This points at cross-process GPU contention (two
different test binaries opening/using the same physical Vulkan device at the
same time) rather than a bug in `kernel_timing.rs`'s own 3 tests or in
within-binary thread scheduling.

**A real lead, not yet chased down**: `kernel_times_also_works_on_the_
serialized_intel_workaround_path` calls `std::env::set_var("BRAIN_VK_SERIAL",
"1")` then `remove_var` later in the same test - `std::env::set_var` mutates
*process-global* state. Within this one binary that's provably safe (the
`--test-threads=8`/`=1` isolation runs above both passed), but if any other
GPU-touching code in the same process ever reads `BRAIN_VK_SERIAL` while this
test's env mutation is in flight, or if the actual mechanism serializing GPU
access across concurrent test *binaries* is weaker than the Makefile's "GPU
serialised" comment (`Makefile:191`) implies, a build-vs-run race here is
plausible. No file-lock or other cross-process GPU serialization mechanism
was found in the codebase during a search for one (`flock`/`gpu_lock`/
similar) - worth confirming whether one is expected to exist and is missing,
or whether concurrent GPU test binaries were never actually meant to be
mutually exclusive and this is a driver-level multi-process contention issue
instead.

**Practical impact (updated)**: `make test` used to be unreliable end-to-end
on this box without a lucky scheduling draw. A full session dedicated to
finding and fixing every instance of the underlying hazard (below) got a
complete `make test` run to pass cleanly, zero failures, multiple times in a
row - see the fixes below, including one applied directly to this file's own
three tests.

## Root cause found and fixed: many test files built real GPU/Vulkan devices directly, unsynchronized, letting concurrent test threads race the driver

The general hazard - "several concurrently-live real devices on one card are
hostile to the driver" (`gpu_core::Gpu::share`'s own doc comment) - already
had an established fix pattern in this codebase before this session:
`gpu_core::testgpu::dev()` (a per-test-binary, mutex-guarded shared-device
pool) for tests that can share ONE device, and a local `DEVICE_SERIAL:
Mutex<()>` (first established in `crates/gpu-core/tests/device_sharing.rs`)
for tests that must each build their OWN independent device and therefore
cannot share via `testgpu::dev`. Most GPU-testing crates (`kronos`,
`deepseek2`, `gpt2`, `glmdsa`, `qwen3`, `qwen35moe`, ...) already followed
one of the two conventions. This session found six places that didn't,
confirmed by directly reproducing (not just theorizing) a hang in `make
test` for the first four, then applying the same diagnosis to find the
remaining two by inspection before they could cause a fifth and sixth
multi-minute hang:

- `crates/fincast/src/model.rs` - three `model::tests` called `Fincast::
  from_weights(...)` (raw `Gpu::new`) directly instead of `from_weights_on
  (gpu_core::testgpu::dev(PIPELINES), ...)`. Reproduced: `make test` hung
  28+ minutes, one thread pinned at 100% CPU, GPU at 0% utilization.
  Isolation proved it: `--test-threads=1` passed cleanly in ~1s;
  `--test-threads=8` (the default) hung indefinitely. Fixed (3 call sites).
- `crates/chronos2/src/model.rs` - the same pattern, found by the same sweep
  before it independently caused its own hang (3 call sites, fixed).
- `crates/gpu-core/tests/device_churn.rs` - all 4 tests intentionally build
  several independent real devices in a loop (that IS the churn under test,
  so `testgpu::dev` sharing does not apply), but had no `DEVICE_SERIAL`
  guard, unlike its sibling `device_sharing.rs`. Reproduced: hung 15+
  minutes under the default `make test` run. Fixed by adding the same
  `DEVICE_SERIAL` lock `device_sharing.rs` already uses, held for each
  test's whole body.
- `crates/lfm2/tests/chunked_equiv.rs` - both tests call `Lfm::new`/`new_
  chunked`, which have no `_on(gpu, ...)` variant at all (every `lfm2`
  constructor builds its own device internally), so `testgpu::dev` sharing
  isn't available here either. Reproduced: hung 15+ minutes. Fixed with the
  same `DEVICE_SERIAL` pattern.
- `crates/model/tests/conv_dtype_roundtrip.rs` - the three `_on_gpu` tests
  each call `Gpu::new_wgpu` directly (deliberately bypassing the ambient
  backend selection to force a specific one, so `testgpu::dev` - which
  shares whatever backend the FIRST caller happened to build - is the wrong
  tool here too). Reproduced: hung 15+ minutes. Fixed with `DEVICE_SERIAL`.
- `crates/cli/tests/npu_model_parity.rs` - `tiny_chronos2()`/`tiny_fincast()`
  (each called from 3 different `#[test]` fns) used the same buggy
  `Chronos2::from_weights`/`Fincast::from_weights` fincast/chronos2's OWN
  tests had (above) - a SEPARATE call site in a different crate, so fixing
  `fincast`/`chronos2`'s own test modules did not fix this file. Not
  independently re-triggered this session (found by inspection once the
  pattern was established), but this is almost certainly what the `kernel_
  timing` entry above originally attributed to unproven cross-process
  contention - `npu_model_parity` was one of the exact test names that
  recurred across this session's earlier hangs. Fixed by switching both
  helpers to `from_weights_on(gpu_core::testgpu::dev(PIPELINES), ...)`
  (available here since `fincast`/`chronos2` do have an `_on` variant,
  unlike `lfm2`).
- `crates/backend-vulkan/tests/kernel_timing.rs` (this file, above) - all 3
  tests call `backend()`, which builds a real `VulkanBackend` directly (below
  `gpu_core::Gpu` entirely, so neither `testgpu::dev` nor `Gpu::share` apply
  - this is the lowest-level instance of the hazard found). Not
  independently re-triggered this session either, found the same way as
  `npu_model_parity` above. Fixed with `DEVICE_SERIAL`.

**On the original "not reproducible in isolation" / cross-process theory
above**: this session's evidence points at cross-*thread* (within one test
binary, under `cargo test`'s default multi-threaded execution) as the
dominant, directly-reproduced mechanism for four of the six - not the
cross-process theory this entry originally proposed. The original hang this
entry documents was never re-triggered to re-diagnose directly, so it's
possible cross-process contention is also real and additive; the fix here
(strictly fewer concurrently-live real devices at any moment, workspace-wide)
helps either way, whether the true mechanism was cross-thread, cross-process,
or a driver-side resource that both share.

**Verification**: a full `make test` run (this session, after the fixes
above landed) completed with zero failures and no hang, more than once. Two
more instances of the identical pattern turned up later in the same sweep -
`crates/backend-vulkan/tests/perf_contract.rs` (this crate's other
`VulkanBackend::try_new`-direct file) and `crates/zipdepth/tests/
p3_fused_eval.rs` (`Gpu::new_wgpu`, two non-`#[ignore]`d tests) - both fixed
the same way, bringing the total to nine files across this session.
`crates/gpu-core/tests/device_sharing.rs`'s own doc comment already named
this exact hazard and its fix - this sweep is that convention being applied
everywhere it was missing, not a new idea. Worth a lint/CI grep (`Gpu::new(`/
`Gpu::new_wgpu(`/`VulkanBackend::try_new(` outside `testgpu`/production
constructors, inside a `#[cfg(test)]` module or `tests/` file with no
`DEVICE_SERIAL` in scope AND the enclosing `#[test]` fn not `#[ignore]`d) to
catch a tenth instance before it causes another multi-minute hang instead of
after.

## Real, open bug: intermittent SIGSEGV at test-binary exit, seen in `brain-qwen3 --lib` - very likely an NVIDIA driver defect, not fixable from this codebase

Distinct from every hang above (which were all deterministic, 100%
reproducible cross-thread device-construction races with a clear fix) - this
one is a genuine flake: `cargo test -p brain-qwen3 --lib` (73 tests, all
using the shared `gpu_core::testgpu::dev()` pool correctly, none of the
patterns above) crashes with `signal: 11, SIGSEGV: invalid memory reference`
somewhere around 1 run in 3, with **no panic message at all** - every
individual test's `test result: ok` line prints first, THEN the crash,
meaning the failure is in process teardown after every actual assertion
already passed, not in any test's own logic.

**A real backtrace WAS captured** (`ptrace` is not actually blocked in this
sandbox - it is YAMA-restricted to parent/child, which running the crashing
binary directly under `gdb -batch -ex run -ex bt` satisfies; earlier
sessions' "ptrace is blocked" conclusion, inherited from `fastvlm caption`'s
still-genuinely-blocked case, was over-generalized). The crash is:

```
Thread 33 "[vkps] Update" received signal SIGSEGV, Segmentation fault.
#0  0x00007605e8c0e4c0 in ?? ()
#1  0x000000006a80ef89 in ?? ()
... (every frame "No symbol table info available", nonsensical addresses,
     "Backtrace stopped: previous frame inner to this frame (corrupt stack?)")
```

`"[vkps] Update"` is a thread the NVIDIA Vulkan driver itself spawns and
owns (not brain/wgpu/wgpu-hal Rust code - no symbols exist for it because
it's inside the closed-source driver blob), doing background maintenance
(the existing `DeviceShared::drop` doc comment's own theory: pipeline-cache
optimisation) independently of anything the test process explicitly
schedules. The fully corrupted stack (garbage frame pointers, no symbols
anywhere) is the signature of memory corruption inside the driver's own
internal state, not a Rust-level bug this codebase's source can be at fault
for in the usual sense.

**Three real mitigations were tried and empirically measured against a
same-methodology baseline** (10-15 back-to-back native runs of the same
built test binary each, `--test-threads=8` unless noted, same otherwise-idle
box):

| variant | crashes / runs |
|---|---|
| baseline (current committed code: drain + `init_lock()` + destroy) | 4 / 10 |
| baseline, but `--test-threads=1` (fully serial, zero concurrent GPU work) | 3 / 8 |
| + a 25ms sleep after the drain, before the actual destroy calls | 8 / 10 |
| + `gpu_core::set_process_exiting()` called once at the pool's first build, so every teardown leaks (`mem::forget`) instead of destroying | 12 / 15 |

Two findings fall out of this, both surprising relative to the existing
`DeviceShared::drop`/`testgpu` doc comments' own theories:

1. **Serial execution does not fix it.** The existing comments blame
   concurrent GPU dispatch racing the driver's background thread
   ("observed... while the test suite ran concurrently"). A fully serial
   run - one test at a time, never two threads touching the GPU at once -
   crashed at essentially the same rate (3/8) as the default concurrent run
   (4/10). Thread contention within this process is not the dominant
   mechanism; the existing causal story is likely incomplete or wrong.
2. **Neither extra settle time nor skipping destruction entirely helped -
   both made the observed rate measurably worse**, not better, at this
   sample size. This corroborates `crates/gpu-core/tests/device_churn.rs`'s
   own already-committed, already-`#[ignore]`d finding from a separate
   investigation: even a 1.5s delay between tearing down one device and
   building the next did not prevent this box's driver from destabilizing
   after a handful of device-construction cycles ("rules out 'concurrent
   creation' as the mechanism... narrows the remaining theory to
   driver-side teardown/reclaim timing"). A 25ms settle window is far
   short of that already-insufficient 1.5s, so its failure to help is
   consistent, not surprising in hindsight.

**Same crash, confirmed in `brain`'s own CLI binary, not just test
binaries**: `.agents/roadmap/vlm.md` separately documented `brain fastvlm
caption` reliably segfaulting on exit after correctly writing its output,
previously written up as un-backtraceable ("this sandbox blocks `ptrace`
entirely"). That conclusion was wrong - `ptrace` is YAMA-restricted to
parent/child, not disabled, and running the crashing binary directly under
`gdb -batch -ex run -ex bt` (gdb spawns it as gdb's own child, satisfying
the restriction) catches it cleanly. The captured trace is
frame-for-frame the same signature as the `brain-qwen3` crash above: a
thread named `"[vkps] Update"` (the same driver-owned, symbol-free thread),
the same corrupted-stack pattern, after the caption was already written
correctly to disk. **These are the same bug**, not two unrelated ones -
`fastvlm caption` and `brain-qwen3 --lib`'s test teardown both hit the same
NVIDIA driver defect through different call paths (a real CLI command's
device teardown vs. a test binary's pooled-device teardown).

A fourth mitigation was tried against the `fastvlm caption` case
specifically, informed by a real clue in a *non-crashing* `gdb` run: the
exit-path backtrace showed `brain::main` falling through to Rust's normal
process-exit machinery, which runs libc's `atexit` handlers, one of which
`dlclose()`s `libEGL_nvidia.so.0` - unloading the driver's own shared
library as part of ordinary process shutdown, a well-known real-world
trigger for exactly this class of "driver's own background thread races
its own unload" crash. The fix tried: after `main`'s dispatch returns
normally, flush stdout/stderr explicitly and call `libc::_exit()` instead
of letting Rust's runtime reach the normal atexit/dlclose path. It did not
help - repeated `fastvlm caption` runs after this change still crashed (and
in one run, hung instead, needing `SIGKILL`), because the actual device
teardown happens EARLIER, inside `resolve::dispatch`'s own Rust-level
`Drop` chain (`Gpu`/`DeviceShared` going out of scope as the caption
function returns) - well before control would ever reach the new
process-exit code. Reverted (also drops a `libc` dependency added to
`crates/cli` only for this experiment).

**One new observation from this fourth attempt**: the failure is not always
a crash - one run hung instead, needing `SIGKILL` after 35s. A single
underlying race manifesting as either a segfault or a hang depending on
precise timing is the classic signature of a genuine data race, not a
deterministic logic bug, and is consistent with the driver-defect theory
above rather than anything specific to brain's own exit sequencing.

**Conclusion: this reads as a genuine defect in the installed NVIDIA driver
(570.195.03, from `nvidia-smi`) in how it handles a Vulkan device's teardown
interacting with its own internal worker thread on repeated device churn -
not a bug in brain's Rust code, and not something the userspace Vulkan/wgpu
API calls this codebase makes can reliably avoid.** Four independent
mitigations were tried (extra settle time, always-leak instead of destroy,
forcing serial test execution, skipping the atexit/dlclose path at process
end) - none helped, two measurably made the observed rate worse, and one
(serial execution) directly falsified the existing code comments' own
causal theory (concurrent GPU dispatch racing the driver's thread). All
four were reverted; the committed baseline (drain + lock + destroy) remains
the best of the options actually measured, even though it is not
crash-free. Does not indicate any of this session's other fixes are wrong -
the crash is in process teardown after every actual assertion/output
already completed correctly, and a re-run that hits this flake reliably
passes clean without changing anything.

**If revisited**: try a newer/older NVIDIA driver build on this box (a
version-specific driver bug is the most likely explanation given
`nvidia-smi`'s reported 570.195.03 and the closed-source, symbol-free crash
site), or file a driver bug report with the captured backtrace above and a
minimal Vulkan repro extracted from `crates/gpu-core/tests/device_churn.rs`
(which already demonstrates the same instability class without any of
brain's own model code involved).
