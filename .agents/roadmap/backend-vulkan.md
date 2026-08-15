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

## Real, open bug: intermittent SIGSEGV at test-binary exit, seen in `brain-qwen3 --lib`

Distinct from every hang above (which were all deterministic, 100%
reproducible cross-thread device-construction races with a clear fix) - this
one is a genuine flake: `cargo test -p brain-qwen3 --lib` (73 tests, all
using the shared `gpu_core::testgpu::dev()` pool correctly, none of the
patterns above) crashes with `signal: 11, SIGSEGV: invalid memory reference`
roughly 1 run in 3, with **no panic message at all** - every individual
test's `test result: ok` line prints first, THEN the crash, meaning the
failure is in process teardown after every actual assertion already passed,
not in any test's own logic. Confirmed non-deterministic across repeated identical runs on an
otherwise-idle box (no other GPU process competing): clean, crash, clean,
across three back-to-back attempts - not proof of an exact rate at this
sample size, but enough to rule out "always crashes" or "never crashes."

**This matches a hazard `gpu_core::WeakGpu`'s own doc comment already names
and partially defends against**: "a device that survives into process exit
(leaked in a static, or torn down from an `atexit` hook) crashes the NVIDIA
driver's worker threads during teardown... in-test drops never crash; a
leaked static crashed intermittently; atexit teardown crashed every run."
`testgpu::dev()`'s weak-reference pool exists specifically so a pooled
device dies with its last real handle DURING a test, never surviving to
process exit - but this crash's timing (strictly after every test's own
`ok`, during the test harness's own final cleanup) is exactly the "survives
to process exit" danger zone that doc comment describes, at roughly the
"leaked static" crash RATE it measured ("intermittent", not "every run").
Something in this crate's 73-test suite - which builds many distinct
`testgpu::dev()` pool entries, one per distinct kernel-list key, all
potentially still weakly-referenced right up to the test harness's own exit
- is not draining as cleanly as the "dies in-test" design assumes. Not yet
narrowed to which specific test/kernel-list combination is the last one
standing when this fires, since the crash is silent (no panic, no test name
attached) and non-deterministic (bisecting by disabling tests would need
many repeated runs per candidate to get a confident signal at a ~33% base
rate).

**Not further root-caused this session**: like `fastvlm caption`'s
documented segfault-on-exit (`.agents/roadmap/vlm.md`), this is a SIGSEGV
with no Rust panic to backtrace, and this sandbox blocks `ptrace` entirely
(even for a locally-owned process), so no core dump or live debugger
attachment was possible. Given the crash is in process teardown after every
assertion already passed, it does not indicate any of this session's actual
fixes are wrong - `make test` runs that hit this flake and are re-run
without changing anything reliably pass clean, which is why this was not
treated as a blocker for the fixes already verified and committed. A real,
open bug, not silently dropped.
