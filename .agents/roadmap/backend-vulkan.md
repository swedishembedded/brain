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

## The "GPU device lost" crash: deferred reclaim freed buffers live descriptor sets still named

Hardware: 2x Tesla P40 (GP102, sm_61, 24 GB, no NVLink, PCIe), driver
570.195.03. Repro: `cargo test -p brain-ltxv --test int8_compute
dit_forward_stays_close_with_int8_compute_dispatch` under `BRAIN_DEVICE=vulkan`
- 1.6 s, tiny 2-layer random-weight config, no real checkpoint, 100%
reproducible. The expensive real-weight LTX generation was never needed.

**Found with validation layers, on the first run.** They are not installed on
this box (`/usr/share/vulkan/explicit_layer.d` has only `VkLayer_INTEL_nullhw`
and `VkLayer_MESA_overlay`), but the loader is 1.4.304 and the Ubuntu package
extracts and runs without root or any code change:

```
apt-get download vulkan-validationlayers && dpkg-deb -x *.deb vvl
VK_ADD_LAYER_PATH=vvl/usr/share/vulkan/explicit_layer.d \
LD_LIBRARY_PATH=vvl/usr/lib/x86_64-linux-gnu \
VK_LOADER_LAYERS_ENABLE='*validation*' <the test binary>
```

`VK_LOADER_LAYERS_ENABLE` force-enables the layer through the loader, so
neither `VkInstance` creation site needs a debug path added. The first run
said exactly what was wrong:

```
VUID-vkCmdDispatch-None-08114: the descriptor VkDescriptorSet 0x125...[Set 0,
Binding 3, Index 0] is using buffer VkBuffer 0x11a... that is invalid or has
been destroyed.
```

**Root cause.** `VkOwnedBuffer::drop` buries a buffer rather than freeing it;
`VkContext::reclaim_dead` destroys it once the device is provably done. The
whole safety condition was `pending_steps`, a count of dispatches that had
reached `VulkanBackend::submit` (`crates/backend-vulkan/src/lib.rs`, the
`steps_recorded` call in `submit`). That is incremented too late. A descriptor
set starts naming a raw `vk::Buffer` when it is **written**, in `record()`,
which happens while the step is still being *built*. So a caller that built a
batch of steps, dropped a scratch buffer, and only then submitted left the
counter at zero with live sets still pointing at the scratch - and the next
flush of an empty pending list destroyed it underneath them. Every
`read`/`write`/`poll_wait` reaches that path, and so does `submit` itself when
`clears` is non-empty, because it flushes BEFORE counting its own steps.

A trace of the repro made it quantitative: **45 descriptor sets recorded, then
one reclaim destroying 34 buffers those exact sets named, then the submit.**
The dispatches read freed device memory, which this card reports as
`VK_ERROR_DEVICE_LOST`. Disabling reclaim entirely made the test pass, which
confirmed the mechanism before any fix was written.

**Fix.** Track the reference precisely rather than counting steps:
`VkContext::set_names` records the handles each live set names, at
descriptor-write time; `set_released` retires a transient set after the flush
that ran it has fence-waited; `reclaim_dead` refuses to destroy a buried buffer
any live set still names and leaves it buried for the next reclaim. Keyed per
set, not counted globally, so a caller-held `step_buf` step pins only its own
operands instead of switching reclaim off device-wide - the omni streaming and
per-token reclaim paths keep working unchanged.

**Gate.** `crates/backend-vulkan/tests/deferred_reclaim.rs`. It asserts the
invariant directly - a still-named buffer is observably not destroyed
(`buried_bytes`), the dispatch that names it computes the right answer, and an
unreferenced sibling IS still reclaimed - rather than gating on "the process
did not crash", which a use-after-free is only sometimes impolite enough to do.
Both tests fail without the fix (`buried 0 B, expected at least 4096 B`).
Verified: 10/10 clean repro runs, zero validation errors, and
`cargo test -p brain-backend-vulkan -p brain-backend-wgpu -p brain-gpu-core
--tests` green.

## Then the real surprise: the native Vulkan backend was running at HALF the wgpu backend's arithmetic throughput

With the crash gone, `gpu_core::roof` could finally measure this backend. Best
of several runs on an otherwise idle box, same card, same WGSL:

| backend | fp32 FMA | packed int8 (DP4A) | DRAM |
|---|---:|---:|---:|
| `backend-wgpu` | 10.62 TFLOP/s | 43.2 TOP/s | 287 GB/s |
| `backend-vulkan` (before) | 5.05 TFLOP/s | 20.4 TOP/s | 287 GB/s |
| `backend-vulkan` (after) | 10.65 TFLOP/s | 43.3 TOP/s | 287 GB/s |

Bandwidth matching exactly while BOTH compute probes came in at ~2.1x placed
the gap in code generation, not in dispatch geometry, occupancy or submit
overhead. Device timestamp queries confirmed real GPU execution time (246 ms vs
132 ms for the identical dispatch), so it was not host-side batching either.

**Root cause.** `crates/vulkan/src/shader.rs` passed
`naga::back::spv::Options::default()`, which sets `force_loop_bounding: true`
(a guard counter decremented every loop iteration) and
`BoundsCheckPolicies::default()` = `Restrict` (a clamp on every buffer access).
`backend-wgpu` deliberately turns both off - `create_shader_module_trusted` with
`ShaderRuntimeChecks::unchecked()`, which wgpu-hal translates into exactly those
two naga settings - and `backend-cpu` makes the same call with Cranelift's
`MemFlags::trusted()`. Only the Vulkan path was paying, and nothing measured it
because nothing had ever rooflined this backend.

Fixed by mirroring the decision and reading the SAME `BRAIN_GPU_CHECKED`
switch, plus enabling `robustBufferAccess` on the logical device so an
out-of-range access is bounded by hardware rather than undefined - the backstop
wgpu relies on when its own robustness cap selects `BoundsCheckPolicy::
Unchecked`. LTX-2.5 int8 parity is unchanged to nine decimals afterwards, on
both the synthetic fixture and the real 22B Q8_0 block 0
(cosine 0.996300183 vs the wgpu-measured 0.996303655), under validation layers
with zero errors.

**Measured roofline for this box** (`gpu_core::roof`, GPU-timestamp based),
against the GP102 playbook's theoretical 11.758 TFLOP/s / 346 GB/s / ~47 TOP/s:

| roof | `backend-wgpu` | `backend-vulkan` | % of spec |
|---|---:|---:|---:|
| fp32 FMA | 10 697 GFLOP/s | 10 638 GFLOP/s | **91.0% / 90.5%** |
| DRAM bandwidth | 287.9 GB/s | 286.8 GB/s | **83.2% / 82.9%** |
| packed int8 DP4A | 43 604 GOP/s | 43 416 GOP/s | **92.8% / 92.4%** |

Best of three runs each on a fully idle box (both cards at 0 MiB, 0%), the two
backends agreeing within ~1% on every roof after the fix above.

Two of the three sit in the 85-90% band sec27 says to expect; DRAM at 83% is
slightly under it and is the same number on both backends, so it is the card,
not the backend.

**A defect this comparison exposed.** `roof`'s memo, both in memory and on
disk, was keyed on the adapter description alone. A roof is a property of the
(device, backend) PAIR - as the table above shows, by a factor of two - so
whichever backend measured first published its number as the other's, and every
"% of roof" on the other backend was wrong by that ratio. Both caches now key
on the backend too.

## What is STILL slower on `backend-vulkan`, and why the backend swap is not yet the win

Real 22B checkpoint, `ltxv_bench streamed 8 3520 1024 1 1`, warm (cache-hit,
device-resident) call, `BRAIN_PROFILE=1`:

| stage | wgpu | native vulkan |
|---|---:|---:|
| block submit+wait (sum over 8 layers) | 3608 ms | **6813 ms** |
| activation/context/adaLN upload per forward | 1393 ms | 1757 ms |

So the real block stack is still **~1.9x slower** on the native Vulkan backend
even with the rooflines matched. That is NOT kernel efficiency - the kernels are
the same WGSL and now the same measured peak - so it is the backend's execution
model. The prime suspect, not yet confirmed: `flush()` is a blocking
submit + fence wait, `write`/`write_at`/`read` each force one, and
`submit(clears, steps)` runs `run_clears` as its own separate blocking
`begin_cmd`/`end_and_wait` round trip, where wgpu records the clear into the
same encoder. A real forward does many of these per layer. **Chasing that is the
next piece of work on this backend**, and until it is done `--device vulkan`
buys memory, not speed.

Two further gaps found while measuring, both real and both open:

* At 48 layers with device residency the native Vulkan path dies in
  `crates/vulkan/src/context.rs`'s `allocate_memory` `.expect(...)` - a hard
  panic with an opaque message, at only ~4.6 GiB resident, where the wgpu path
  reports a recoverable `Out of Memory`. Whatever the allocation failure is
  (it is well short of the card), the backend should surface it as an error
  rather than panicking.
* `ltxv_bench` prints no per-kernel table on this backend: `VkProfile` is
  per-`VulkanBackend` handle, so a `share()`d handle's dispatches are not
  accumulated into the handle the bench reads. The device timestamps
  themselves work (`tests/kernel_timing.rs` gates them); the aggregation
  across shared handles does not.

## Problem 2: wgpu's 2.00x resident overhead - root cause found, fix is NOT in wgpu

sec35 recorded the effect and correctly attributed it to the wgpu backend. The
mechanism is now identified, with hardware evidence.

Re-confirmed unchanged (`crates/gpu-core/tests/vram_overhead.rs`, this pass):
wgpu 2.00x with upload at both 256 MiB and 1024 MiB, 2.00x with 64 MiB
chunking, 1.00x allocate-only, and native Vulkan 1.00x.

**This card's Vulkan memory types** (dumped through `ash` - `vulkaninfo`
needs an X server and cannot run here):

```
heap 0: 24.00 GiB DEVICE_LOCAL          <- VRAM
heap 1: 138.40 GiB (no flags)           <- system RAM
type  8: heap=1  HOST_VISIBLE | HOST_COHERENT
type  9: heap=1  HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
type 10: heap=0  DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
```

Type 10 is host-visible device-local memory drawn from the **full 24 GiB VRAM
heap** - not from a 256 MiB non-ReBAR BAR window, which is what the older
"non-ReBAR Pascal" framing assumed.

`wgpu-hal/src/vulkan/device.rs::create_buffer` maps any `MAP_WRITE` buffer -
which is every `Queue::write_buffer` staging buffer - to
`gpu_allocator::MemoryLocation::CpuToGpu`, and `gpu-allocator` resolves that to
preferred flags `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL`. On this card
that is type 10. **Every byte ever uploaded therefore allocates an equal number
of bytes of staging IN VRAM**, and `gpu-allocator` pools freed blocks rather
than returning `VkDeviceMemory` to the driver - which is why the cost tracks
*cumulative* bytes written, why chunking cannot bound it, why allocate-only is
1.00x, and why brain's native Vulkan backend (one shared, bounded staging
buffer) is 1.00x. All five observations fall out of that one mechanism.

**Why no fix landed.** The fix is a memory-type *policy* decision and it does
not live in wgpu's own code: `MemoryLocation` has no "host-preferred upload
staging" variant, so wgpu-hal cannot express the right request, and
`CpuToGpu`'s preference for `DEVICE_LOCAL` is `gpu-allocator`'s - a third-party
crate. Correcting it properly means either a new `MemoryLocation` variant
upstream in `gpu-allocator` or wgpu-hal selecting the memory type itself, and
then arguing the trade (write-combined system RAM vs VRAM staging) for
integrated and ReBAR parts too, where the current choice is right.

Separately, the fork at `github.com/swedishembedded/wgpu` (cloned alongside
this repo under `[path/to/applications]/wgpu`, `trunk` at `6d1bb1959`) is
**wgpu 30.0.0**, while this workspace builds `wgpu = "29"` from crates.io - a major
version apart. Both stop conditions the task named were met, so **brain's
dependency was deliberately left untouched** at crates.io 29; there is no
half-applied swap to revert (`Cargo.toml` is unmodified).

**Practical consequence for placement**: sec35's advice stands but its
reasoning should be updated - the 2x is not a property of non-ReBAR Pascal, it
is wgpu staging landing in a host-visible VRAM heap that this card happens to
expose at full size.

## 2026-08-21: the 2.00x is fixed, in wgpu, and it measures 1.00x

Hardware: 2x Tesla P40 (GP102, 24 GB, driver 570.195.03), the same box every
number above was taken on. Baseline re-confirmed on an idle box before
touching anything, so the before/after pair is one session's own measurement
rather than a comparison against a remembered number:

| probe (1024 MiB logical) | before | after |
|---|---:|---:|
| `wgpu-256mib` | 2.00x | **1.00x** |
| `wgpu-1024mib` | 2.00x | **1.00x** |
| `wgpu-1024mib-nocopysrc` | 2.00x | **1.00x** |
| `wgpu-1024mib-alloconly` | 1.00x | 1.00x |
| `wgpu-1024mib-chunked64` | 2.00x | **1.00x** |
| `native-vulkan-1024mib` | 1.00x | 1.00x |

The wgpu backend now costs exactly what it allocates, matching this repo's own
native Vulkan backend on the same card. A 22 GB checkpoint uploaded through
`Queue::write_buffer` no longer needs 44 GB of a 24 GB card to be resident.

### One correction to the section above, from reading the allocator's source

That section attributes the cumulative cost to `gpu-allocator` "pooling freed
blocks rather than returning `VkDeviceMemory` to the driver". Only half of that
is true, and the half that matters is different.
`gpu_allocator::vulkan::MemoryType::free` destroys a block as soon as it is
empty whenever the block is dedicated, or is a general block that is not the
last one - and any allocation larger than the configured memblock size (which
`wgpu-hal` derives from `wgt::MemoryHints`: 8-64 MiB of device / 4-32 MiB of
host memory under `MemoryUsage`, which is what `crates/backend-wgpu` selects
unless `BRAIN_GPU_MEM_PERF=1`) gets exactly such a dedicated block. A 1 GiB
staging buffer is therefore created and destroyed per upload, not pooled.

What the probe actually measures is that the staging copy is **resident at the
same time as its destination**: `wgpu_core` holds the staging buffer in its
pending-writes temporaries until the submission consuming it retires, so peak
resident is 2N for an N-byte upload, and a loader staging tensor after tensor
pays that peak for the whole load. That also explains the chunked probe, which
the old framing had to hand-wave: 64 MiB chunks do not help because all of the
chunks are live at once, not because a pool refuses to release them. The
practical consequence for placement is identical, so no earlier conclusion
changes - but the next person reading this should not go looking for a pool
that is not there.

### The fix, and why it is four lines rather than a staging-buffer rewrite

The previous section had the mechanism right and the conclusion wrong. It
concluded that correcting the memory-type *policy* needed either a new
`gpu_allocator::MemoryLocation` variant upstream or wgpu-hal selecting memory
types itself, because `MemoryLocation` has no "host-preferred staging" variant
and `CpuToGpu`'s `DEVICE_LOCAL` preference is a third-party crate's policy.
Both are true. Neither is necessary, because the memory type is not the only
input to `gpu-allocator`'s choice: the caller also supplies
`vk::MemoryRequirements::memory_type_bits`, the set of types the allocation is
*allowed* to use, and `gpu-allocator` intersects its preference with that set
before falling back (`gpu_allocator::vulkan::Allocator::allocate`, the
`mem_loc_preferred_bits` / `mem_loc_required_bits` pair). Clearing the
device-local types out of `memory_type_bits` for exactly the buffers that
should not be device local makes `CpuToGpu`'s preferred search find nothing and
fall through to its required `HOST_VISIBLE | HOST_COHERENT`, i.e. system RAM,
with no new API and no change to `gpu-allocator` at all.

So the whole fix is in `wgpu-hal/src/vulkan`: a second memory-type mask
computed once at device creation (`adapter.rs`, host-visible types that are not
device local), stored on `Device` (`mod.rs`), and applied in `create_buffer`
(`device.rs`) to upload staging buffers only.

Three conditions keep it from changing placement anywhere it should not, which
is the part that needed thought rather than code:

* **Only pure upload staging.** Usage must be `MAP_WRITE` and nothing beyond
  `COPY_SRC` - which is exactly and only what `wgpu_core::resource::
  StagingBuffer::new` allocates for every `Queue::write_buffer`/`write_texture`.
  Such a buffer is written once by the CPU and read once by the transfer queue;
  no shader ever touches it, so device-local placement buys it nothing. A
  mappable uniform/storage/vertex buffer that a shader DOES read directly still
  gets `CpuToGpu`, which is the right answer on an integrated or
  resizable-BAR part and is where the ledger's "the current choice is right for
  some hardware" caution actually applies.
* **Only when the device has somewhere else to put it.** The mask is applied
  only if it is non-empty for this allocation. A unified integrated GPU, where
  every host-visible type is also `DEVICE_LOCAL`, computes an empty mask and is
  untouched - no behaviour change and, importantly, no allocation failure from
  masking away every candidate.
* **Readback is not involved.** `MAP_READ` maps to `GpuToCpu`, which already
  prefers host memory and wants `HOST_CACHED`.

Note what the fix does NOT rely on: any attempt to tell a "small non-ReBAR BAR
window" apart from "the whole VRAM heap". That distinction is not visible
through the Vulkan API on this driver (it reports one 24 GiB
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` type either way), and it is not
the right question. The right question is what the buffer is FOR, which the
usage flags answer exactly.

### Where the code is

* **wgpu**: `github.com/swedishembedded/wgpu`, branch
  `v29-staging-host-memory`, one commit (`bc0a87788`, "vulkan: stage uploads in
  host memory, not in the VRAM heap"), cut from the upstream **v29.0.4 tag**
  (`e99f5305d`), +81 lines across `wgpu-hal/src/vulkan/{adapter,device,mod}.rs`
  and `CHANGELOG.md`. **This branch exists only in the local clone**: this
  session had no push credentials for the fork, and per standing instruction
  surfaced that rather than attempting the push. Until it is pushed, the
  `[patch]` in the workspace root `Cargo.toml` stays commented out.
* **gpu-allocator**: cloned for verification alongside the other checkouts and
  read, **not modified**. Confirmed to be Traverse-Research's crate, version
  0.28.0 from the crates.io registry - the same source `wgpu-hal` 29.0.4 and
  the fork's own manifest both name. No fork of it is needed: see above for why
  the fix does not require a new `MemoryLocation`.
* **brain**: the `[patch.crates-io]` note in the workspace root `Cargo.toml`,
  `.cargo/config.toml.example` plus a `.gitignore` entry for the real
  `.cargo/config.toml`, and this file. Nothing in `crates/` changed except the
  gate below.

### The version gap, resolved by not having one

The fork's `trunk` is wgpu 30.0.0 and this workspace pins `wgpu = "29"`, which
is what stopped the previous session. But upstream keeps a **`v29` release
branch**, whose head is the `v29.0.4` tag - byte-identical, verified by diff,
to the `wgpu-hal` 29.0.4 source this workspace was already building from
crates.io. Branching the fix from there means the patched dependency resolves
as 29.0.4 exactly, so `crates/backend-wgpu` needs no API migration, no v29->v30
diff had to be evaluated, and no risk was taken with a major version bump. The
v30 line does not enter into it.

### How it is wired, and how both halves were verified

`paths` in `.cargo/config.toml` (gitignored, machine-specific) and
`[patch.crates-io]` in the workspace root `Cargo.toml` (committed, portable)
are two different mechanisms for the same override, and both were exercised
rather than reasoned about:

* The `paths` override resolves the whole wgpu tree from a local checkout and
  is what every measurement in this section was taken through. Cargo warns on
  every invocation that the override "has altered the original list of
  dependencies" (a crate's in-repo manifest never lists exactly what its
  published form does); the warning is inherent to the mechanism, cargo says it
  may become a hard error some day, and `.cargo/config.toml.example` documents
  the fallback if it ever does.
* The `[patch.crates-io]` git entry was verified to resolve for real - against
  a `file://` URL on the local clone, since the fork branch is not pushed yet -
  and pulls `wgpu`, `wgpu-core`, `wgpu-hal`, `wgpu-types` and `naga` from the
  one entry, all five landing in `Cargo.lock` at 29.0.4. Swapping that URL for
  the GitHub one changes nothing else.
* With both present, the `paths` override wins, which is the intended
  precedence: a developer editing the dependency builds against their working
  tree while everyone else builds against the pinned rev.

### The gate

`crates/gpu-core/tests/vram_overhead.rs` keeps its shape and gains one probe:
**host-to-device upload throughput**, best of three 1 GiB uploads with
allocation excluded. Where staging lives decides the resident cost and the
upload cost at once - host memory costs no VRAM but adds a DMA hop - and a
change that won on one side while quietly losing on the other would be no win,
so the file now measures both instead of asserting one and assuming the other.
Its stale "the fix is `--device vulkan`, not a wgpu-level change" conclusion is
replaced by what actually happened.

`cargo test -p brain-gpu-core -p brain-backend-wgpu --lib --tests` against the
patched dependency: **89 passed, 0 failed**, no regressions.

### The cost side: this is a trade, not a free win

The upload-throughput probes were added to this gate together with the fix,
precisely so the other half of the trade could not be assumed. They found a
real regression, and it is large enough that the fix is deliberately left
OPT-IN (the `[patch]` is commented out) rather than made the default.

Best of three 1 GiB uploads, destination allocation excluded, P40, idle box
for the unpatched arm:

| upload shape | unpatched (VRAM staging) | patched (host staging) |
|---|---:|---:|
| one `write_f32` of 1 GiB | 1.11 GB/s | 0.54 GB/s |
| `write_f32_chunked`, 64 MiB | 1.18 GB/s | 0.46 GB/s |
| `write_f32_chunked`, 4 MiB (`paramstore::UPLOAD_CHUNK_WORDS`, the real path) | 1.16 GB/s | 0.43 GB/s |
| 4 MiB chunks, submitting after each | not measured | 0.40 GB/s |

The patched numbers are stable at 0.4-0.55 GB/s across five separate runs and
across every granularity, including runs where the box was contended and runs
where it was not, so this is a deterministic cost and not measurement noise.

**A hypothesis that was tested and is wrong.** The obvious explanation was
volume of live staging: `wgpu_core` allocates a fresh staging buffer per
`write_buffer` and holds all of them until the next submission, so 1 GiB of
upload means 1 GiB of staging live at once - which in host memory is 1 GiB the
driver must page-pin per upload, where the same thing in VRAM is nearly free.
If that were the whole story, submitting after every chunk (bounding live
staging to one 4 MiB chunk, which is what this repo's own native Vulkan
backend effectively does with its single reused staging buffer) would recover
the throughput. It does not: 0.40 GB/s, no better than not bounding it. That
last row is dominated by its own 256 submit+fence round trips, so the
experiment is inconclusive rather than a clean refutation - but it definitely
does not support the pinning-volume story, and the cost is at least partly the
host memory path itself on this driver.

**What is NOT explained.** The native Vulkan backend asks for the same
`HOST_VISIBLE | HOST_COHERENT` properties, and its `find_memory_type` takes the
same first match - memory type 8 - yet uploads at 1.84-3.41 GB/s through it.
Same card, same memory type, 4-8x the throughput. The difference must be in
how the upload is issued (one persistently reused staging buffer, mapped once,
versus a fresh allocate + map + unmap per `write_buffer`), not in where the
memory lives. Nailing that down is the next piece of work, and it is the piece
that would turn this trade into an unqualified win.

**So the honest summary of the trade, on this hardware:**

* Loading a large model: strongly positive. A 22 GB checkpoint needed 44 GB of
  peak VRAM through wgpu and therefore could not load on a 24 GB card at all;
  now it needs 22 GB and fits. The load itself costs roughly 51 s instead of
  19 s - a one-time cost against a capability that did not exist before.
* An upload-heavy inner loop: negative. `.agents/roadmap`'s own LTX numbers
  above put activation/context/adaLN upload at 1393 ms per forward on wgpu;
  at these rates that becomes several seconds per forward. That is a real
  per-step regression, not a one-time one.

Which is why the committed default is unchanged and the fix is one uncomment
away, with this table next to it. The decision of when to turn it on is a
placement decision per workload, and it should be made with these numbers in
hand rather than by a patch silently changing under a benchmark.

**State this session left the box in**: `.cargo/config.toml` was used for
every measurement above and then removed, so a build here resolves the
unmodified crates.io wgpu 29.0.4 exactly as before. That is deliberate on two
counts - the throughput table above means the patch should be a considered
choice rather than a silent default, and another agent was benchmarking on
this box at the time, where halving upload throughput under their measurements
would have been worse than unhelpful. Turning it back on is
`cp .cargo/config.toml.example .cargo/config.toml` and editing the one path.

## 2026-08-21 (later the same day): the upload regression is gone, and it was two bugs, not one

The section above left host-memory staging shipping OPT-IN, because it halved
upload throughput (1.16 -> 0.43 GB/s at the 4 MiB chunk size real weight
upload uses) and named the cause as "not settled". It is settled. There were
two independent causes, neither of them the memory placement itself, and both
are now fixed. The fix is better than the pre-fix baseline on **both** axes,
so it is no longer a trade to weigh per workload.

Hardware for everything below: **Tesla P40** (GP102, 24 GiB, driver
570.195.03), two of them in one box, 138 GiB host RAM. The box was NOT idle -
another agent was running real LTX-2.5 generations on one or both cards
throughout - which is called out per measurement where it matters.

### Cause 1: host memory is expensive to ALLOCATE; device memory is not

This is the whole of the wgpu-side regression, and it is arithmetic rather
than a hypothesis. Measured directly with a standalone raw-`ash` probe (no
wgpu in the path at all), `vkAllocateMemory` + `vkMapMemory` on this card:

| block size | host-visible, NOT device local (type 8) | device local + host visible (type 10) |
|---|---:|---:|
| 4 MiB | 7.7 ms | 2.2 ms |
| 16 MiB | 28.2 ms | 0.6 ms |
| 64 MiB | 126.9 ms | 1.3 ms |
| 256 MiB | 544.8 ms | 4.8 ms |

Host allocation is **linear in size at about 0.5 GB/s** - the driver has to
commit and pin the pages so the GPU can DMA from them. Device-local
allocation is flat in size and ~100x cheaper, because the driver is handing
back an address range it already owns.

Half a gigabyte per second is *slower than the upload the pages exist to
carry*. So once staging moved to host memory, and with `gpu-allocator`
sub-allocating from 64 MiB blocks on a non-device-local heap and destroying a
block as soon as it empties, page pinning stopped being a setup step and
became the entire cost of uploading. The arithmetic closes exactly: 1 GiB of
staging needs 16 fresh 64 MiB host blocks, 16 x (126.9 + 9.3) ms = **2.18 s**,
against a total measured upload time of 2.50 s. ~87% of the upload was
`vkAllocateMemory`. The same 1 GiB through the old `DEVICE_LOCAL` placement,
whose allocations are nearly free, took 0.88 s.

The same probe priced the fix before any of it was written: with the staging
buffers created once and reused, the same 256 MiB upload went from 0.28 GB/s
to 5.33-7.66 GB/s depending on shape.

### Cause 2: `backend-wgpu` was never submitting upload-only work at all

This one is in this repo, not in wgpu, and it is the reason the first fix
appeared to do nothing when measured through `vram_overhead.rs`.

`Queue::write_buffer` does not talk to the GPU. wgpu copies the bytes into a
staging buffer it allocates for the call and records a copy into its
pending-writes encoder; nothing reaches the device, and **no staging buffer
can be reclaimed**, until a `queue.submit` carries that encoder.
`WgpuBackend::flush_inner` returned early whenever the pending DISPATCH list
was empty - treating "nothing to do" as a property of dispatches alone - so a
phase that only uploads never submitted.

Traced, not deduced. With the Vulkan staging allocator instrumented
underneath a 1 GiB chunked upload, the event log is one unbroken run of 1536
allocations followed by one unbroken run of 1536 frees at device teardown,
with **zero** interleaving:

```
EV 0    TAKE hit=false parked=0          buckets=0
...                                     (1536 takes, every one a miss)
EV 1536 PARK parked=4294967296          buckets=4
...                                     (1536 parks, all at the end)
```

A staging pool cannot recycle a buffer that is never released, so the hit
rate was exactly 0%. It also means a model load held every staging buffer it
had ever allocated live simultaneously - which is its own contribution to the
original 2.00x resident finding, independent of where the memory lived.

Fixed in `crates/backend-wgpu/src/lib.rs`: a `writes_pending` flag on
`DeviceShared` (the queue is shared, so a flush through any handle must submit
them), and `flush` submits when writes are outstanding and no dispatch is.
An idle flush still submits nothing, because `poll_wait` calls flush once per
step and a queue round trip in each would be its own regression. Both halves
are pinned by `crates/backend-wgpu/tests/upload_flush.rs`, which was confirmed
red before the fix and green after.

### What was built in the fork

The branch `v29-staging-host-memory` is **two commits on top of the v29.0.4
tag, one per problem fixed** (it was rebuilt from three, and from messages
naming this box's specific card, so upstream-facing text stays about the
behaviour rather than about our hardware):

* **`a9fe884bd` - stage uploads in host memory, not in the VRAM heap.** The
  placement fix from the previous session, unchanged in substance.
* **`c7187db0c` - reuse upload staging buffers instead of pinning pages per
  write.** A per-device `StagingBufferPool` in `wgpu-hal/src/vulkan`.
  `destroy_buffer` parks a staging buffer keyed by size class instead of
  destroying it; `create_buffer` takes one back out, already allocated, bound
  and mapped. This is the same strategy as this repo's own
  `crates/vulkan/src/context.rs::with_staging` - one buffer, allocated and
  mapped once, written and copied from over and over - ported to wgpu-hal's
  own abstractions rather than copied.
  A full pool **evicts by age rather than turning arrivals away**, and that
  detail is not cosmetic. Refusing them was tried first and made the pool
  worthless under a mixed workload: 25 large one-shot staging buffers (a 1 GiB
  `write_f32` makes one) took the whole capacity, then 1582 consecutive 4 MiB
  parks were refused and all 1607 later takes missed. A 0% hit rate with the
  pool nominally full. Evicting the least recently parked buffer instead lets
  the pool follow the workload.

Design points that took the actual thought:

* **No synchronization of its own, and this is not laziness.**
  `destroy_buffer` already destroys the buffer and frees its memory
  immediately, so the caller must ALREADY guarantee no submission references
  it; handing it back out instead is strictly weaker than destroying it. The
  `wgpu_hal::Device` trait documents exactly this latitude - backends "are
  allowed to allocate GPU memory for buffers from allocation pools, and this
  call is permitted to simply return `buffer`'s storage to that pool". No
  fence, barrier or deferred-free list is needed, and adding one would have
  been cargo-culting the earlier "device lost" bug rather than reasoning about
  this one.
* **It cannot hold device memory.** Only buffers actually steered into
  host-visible non-device-local memory are eligible, so the pool can never
  reintroduce the resident cost `bc0a87788` removed.
* **It cannot leak application-visible bytes.** Eligibility also requires
  `MemoryFlags::TRANSIENT`, which `wgpu_core::resource::StagingBuffer::new`
  sets and nothing else in the workspace does. A buffer an application created
  itself with the same usage is never recycled, so it can never observe a
  previous one's contents.
* **It is bounded by the application's own peak.** A buffer only enters the
  pool by having been handed out first, and a new one is only allocated when
  every parked buffer of its class is in use, so parked-plus-live staging
  never exceeds the most the application ever had live at once. The
  `MemoryHints` fraction of the host heap on top of that is a backstop, not
  the operative bound.
* **Size classes** double below 64 MiB and step by 64 MiB above it, so one
  upload's buffer matches the next one's request without rounding a 700 MiB
  request up to 1 GiB. Three unit tests in `wgpu-hal` pin that a class is
  never smaller than the request, that the waste is bounded, and that a class
  size maps to itself.

### The numbers

**`crates/gpu-core/tests/vram_overhead.rs`, 1 GiB, best of three** - the
project's own gate, so this is the row that counts. Final column taken on a
**completely idle box**, both cards at 0 MiB / 0% util:

| upload shape | stock wgpu 29.0.4 | + host staging alone | + reuse + flush fix |
|---|---:|---:|---:|
| one `write_f32` of 1 GiB | 1.11 GB/s | 0.50 GB/s | **3.76 GB/s** |
| `write_f32_chunked`, 64 MiB | 1.18 GB/s | 0.48 GB/s | **4.96 GB/s** |
| `write_f32_chunked`, 4 MiB (the real weight path) | 1.16 GB/s | 0.43 GB/s | **4.70 GB/s** |
| 4 MiB chunks, submitting per chunk | not measured | 0.51 GB/s | **5.06 GB/s** |

So the shape that matters, the one real weight upload uses, went from
1.16 GB/s before any of this to **4.70 GB/s** - about **4.05x the original**,
and roughly 11x the intermediate state that made the fix ship opt-in.

**Resident cost: unchanged at 1.00x, on every probe.** That same idle run:

```
wgpu-256mib             256 gpu0    256    1.00x
wgpu-1024mib           1024 gpu0   1024    1.00x
wgpu-1024mib-alloconly 1024 gpu0   1026    1.00x
wgpu-1024mib-chunked64 1024 gpu0   1026    1.00x
native-vulkan-1024mib  1024 gpu1   1024    1.00x
native-vulkan-drop-x4  1024 gpu1      0    0.00x
backend fix: wgpu=1.00x native-vulkan=1.00x (delta 0.00x)
```

The stock path was 2.00x. So the 4x throughput comes on top of the halved
VRAM, not instead of it - which is the whole reason this stops being a trade.

**A caveat for whoever runs this next.** These are `nvidia-smi` deltas, so
they are only meaningful on a quiet box. While another agent's LTX
generations were allocating and freeing gigabytes on both cards, the
secondary ratios read 7.61x, 5.45x, -6.41x, 33.12x and similar nonsense, and
`native-vulkan-drop-x4` failed its assertion twice on pure contamination -
it passes cleanly at 0 MiB whenever the box is actually idle. The
`wgpu-1024mib` and `backend fix` rows were the only two that held steady
regardless of load. Do not read a failure of this file as a regression
without checking `nvidia-smi` first.

**Cross-checked independently.** A standalone wgpu program built directly
against the fork (no brain in the path) measured the same three arms on the
idle card, 1 GiB, best of three: stock 3.05 / 2.48 / 2.49 GB/s at 4 / 64 /
1024 MiB chunks, host staging alone 0.46 / 0.35 / 0.45, and with the pool
3.00 / 3.45 / 5.19 - rising to 4.65-5.33 GB/s at 4 MiB once the pool is warm
past the first two passes. The first upload into a cold pool still pays the
pinning once (~0.44 GB/s); nothing after it does.

### State of the `[patch]`

**Still commented out, and for exactly one reason, which is no longer about
whether the fix is worth having:** the fork branch is not pushed.
`git ls-remote` on `github.com/swedishembedded/wgpu` lists `trunk` and nothing
else, so uncommenting the `[patch.crates-io]` git entry as it stands would
fail every build here and in CI at dependency RESOLUTION - before a single
line compiles. This pass was explicitly asked to hold off on pushing, so the
entry stays commented with the note next to it rewritten to say so. Push the
branch, uncomment the one line, done; nothing else needs to change.

Everything above was measured through `.cargo/config.toml`'s `paths` override
(gitignored, machine-local, template in `.cargo/config.toml.example`) pointed
at the local wgpu fork checkout ([path/to/wgpu]). **That override was left in
place**,
unlike the previous session which removed it - the fix is now strictly better
on both axes, so a build on this box picking it up is a benefit rather than a
silent regression under someone else's benchmark. Delete the file to build
against stock crates.io wgpu again.

### End to end, and the test-suite results

`cargo test -p brain-gpu-core -p brain-backend-wgpu --tests`: **all green**,
including the two new `upload_flush.rs` cases. `cargo test -p brain-ltxv
--tests`: **52 passed, 0 failed** across 20 test binaries - the whole
production graphics path under every model in the repo. `make release`
builds clean against the patched dependency.

One test needs the same warning as the resident probes above.
`roofline::measuring_twice_agrees` failed once during a run that overlapped a
neighbouring generation, on two back-to-back compute measurements disagreeing
(7019 vs 2707 GFLOP/s) - while the very next test in the same binary measured
7106. It passes 6/6 on a quiet box. It measures dispatch throughput, which
nothing in this work touches; treat a failure there as "check `nvidia-smi`
first", same as the resident rows.

A real generation through the production path with the fix active: 25 frames
at 512x512, real VAE decode, **11.2 s total** (denoise 0.6 s = 0.138 s/forward,
VAE 10.3 s), valid mp4 written. So the graphics path is correct end to end,
not merely green under tests.

The full **LTX-2.5 22B** generation was attempted at the same shape and did
NOT complete: it reached a 20.9 GiB peak on one 24 GiB card and then hit
`wgpu error: Out of Memory`. That is the weight-residency sizing question in
`crates/ltxv` (the Q8_0 DiT is 23.6 GB on its own, so a single card needs the
streaming/multi-card path), not a staging question - and that code was being
actively changed in the working tree at the time by another agent, so it was
left alone rather than tuned blind. Worth noting which direction the fix
pushes it, though: the ledger above records that this checkpoint needed ~44 GB
of peak VRAM through unpatched wgpu and so could not be attempted on a 24 GB
card at all. Reaching 20.9 GiB before running out is the halved staging cost
doing its job; what is left is a residency window, not a 2x.
