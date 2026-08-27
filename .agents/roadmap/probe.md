# probe - roadmap

`model::probe` (the per-accelerator, per-dtype GEMM capability measurement
whale's `whale status` publishes) and `gpu_core::roof` (the measured silicon
ceiling every "% of peak" divides by).

## Fixed: both probes were timing an idle GPU, not the GPU

**Symptom.** On a Meteor Lake box (Intel Arc iGPU, Mesa 25.0.7 ANV, 22-thread
Cranelift CPU JIT) `whale status` reported the iGPU at 56-65 GFLOP/s fp32
while the CPU JIT reported 138-157 GFLOP/s. An integrated Arc beaten by the
host CPU at fp32 is not a plausible reading of that silicon.

**Root cause.** Neither probe brought the device to its operating point before
timing. An integrated GPU parks at its frequency floor when idle and takes
**seconds** of continuous work to reach the clock it runs a job at; the probe
gave each `(device, dtype)` 150 ms and took the best of two dispatches, so
every GPU row was the *idle clock*. The ramp, same kernel and shape and
buffers throughout (`matmul_reg2`, `[256,2048] x [2048,2048]ᵀ`, one dispatch
per timed iteration, `poll_wait`-bracketed):

| t | GFLOP/s | `gt_act_freq_mhz` |
|---|---|---|
| first dispatch out of idle | 119 | 350 |
| 1.0 s | 475 | 750 |
| 3.0 s | 943 | 1800 |
| 6.0 s | 1078 | 2150 |

The achieved rate tracks the driver's own reported clock about 1:1. The floor
is 100 MHz and the ceiling 2250 MHz, so the available range is over 20x, and
a sub-second probe reads the bottom of it.

`gpu_core::roof` had the identical bug with a worse consequence, because it
**caches and persists** what it measures: a cold probe on this box reported
542 GFLOP/s fp32 and 11.3 GB/s DRAM, while `matmul_reg2` - a real memory-fed
GEMM, which cannot exceed the silicon - was measured at 1243 GFLOP/s on the
same device. A roof below a real kernel's rate is impossible, and nothing
checked for it.

**Fix.**

* `model::probe::warm_up` + `Plan::warmup` (default 3 s, once per device, not
  once per tier). `sweep_ramped` returns the `WarmUp` beside the tiers so the
  ramp is reported rather than hidden. `Plan` also went from best-of-2 in
  150 ms to best-of-8 in 250 ms per tier; `reps` was already a ceiling the
  budget cuts short, so a slow device is unaffected.
* `gpu_core::roof::warm_up` (default 2 s, `BRAIN_ROOF_WARMUP_S=0` restores the
  old behaviour) before any rung.
* `crates/gpu-core/tests/roofline.rs::real_work_can_never_beat_the_roof` pins
  the invariant that would have caught it: `matmul_reg2`'s measured rate
  against the measured roof, on the same device, both ramped.
* `crates/model/tests/probe_gemm.rs::the_probe_reports_the_operating_point_not_the_idle_clock`
  pins the probe side: where the device demonstrably climbed during the ramp,
  the published number must be above the cold first dispatch.

**Result** (same box, `cargo test --release -p brain-model --test probe_gemm
-- --nocapture`, whole-machine profile, machine otherwise idle):

| device | f32 | bf16 | f16 | i8 | q4 |
|---|---|---|---|---|---|
| igpu (wgpu) | 765 GFLOP/s | 874 | 842 | 1442 GOP/s | 184 |
| cpu (Cranelift JIT) | 195 GFLOP/s | 13.8 | 7.7 | -- | -- |

against 56-65 / 63-77 / 53-63 / 100-125 / 14-22 before. The measured roof rose
from 542 to 814 GFLOP/s fp32 and from 11.3 to 31.0 GB/s DRAM in the same
conditions, and `matmul_reg2` reads 92% of that roof at `1024³`.

## This box's numbers move with the package power budget - record the state

`throttle_reason_pl1` is `1` under any sustained load here and
`punit_req_freq_mhz` clamps to its 800 MHz floor, so the iGPU's achievable
rate is a function of what the CPU is doing. Measured spreads on one
otherwise-unchanged tree:

* A trivial `bash` sysfs-polling loop on the host held the GPU at 800 MHz
  requested and cost roughly 3x on the same GEMM.
* With the package at 100 C after an hour of benchmarking (and a concurrent
  PyTorch job on the CPU), the same `machine` profile read 163 GFLOP/s fp32
  for the iGPU and 35 GFLOP/s for the CPU - both roughly 5x down from the
  idle-box figures above, with the fix in place either way.
* `roofline.rs::measuring_twice_agrees` had to widen from a 0.25 to a 0.5
  spread band: two back-to-back `roof::measure` calls read 443 and 258
  GFLOP/s, each reproducible alone. That is the chassis, not the probe.

So a single figure from this class of machine is only meaningful with the
package state attached, and an A/B whose two arms sit either side of a
heating event measures the chassis.

## Not the cause, checked and ruled out

* **The cooperative-matrix kernel is not in any measured or served path.**
  `crates/vulkan` (`brain-vulkan`) is a demo crate: `vulkan::matmul` and its
  GLSL `matmul_coopmat.comp` are reached only from `moe pid vk-info` and
  `moe pid vk-matmul`, both behind the cli's optional `vulkan-coopmat`
  feature. `brain-backend-vulkan` - the real, runtime-selectable
  `--device vulkan` backend - imports only `vulkan::context` and
  `vulkan::shader` and compiles the ordinary WGSL kernels through naga. So the
  long-standing `no glslc/glslangValidator on PATH` build warning never
  degraded inference or the benchmark, which is why installing the toolchain
  changed no measured number. `build.rs`'s warning now says that, and names
  the packages, instead of pointing at a `README_VULKAN.md` that does not
  exist in this tree.
* **`step_sliced` vs `step`.** `Ops::matmul` binds sub-ranges; measured at
  `[256,2048] x [2048,2048]ᵀ` and `2048³` the two are within noise (752 vs 754,
  790 vs 779 GFLOP/s).
* **A taller shape ladder.** Adding `(512,2048,2048)` and `(1024,2048,2048)`
  rungs raised f32 (631 to 1111 GFLOP/s in one interleaved round) but starved
  the tiers that come after it in the same budget: i8 fell from 1226 to 408
  GOP/s and q4 from 166 to 57. Rejected at this budget; the three-rung ladder
  stands.

## Left to do

* **A ceiling that survives the thermal state.** whale caches one sweep's
  result for 15 minutes; on a part whose rate swings 5x with package
  temperature, "the best of the last N sweeps" is arguably the honest reading
  of the word *ceiling*, and the current single-sweep value is not. Deliberately
  not done in the same change as the ramp.
* **A real `VK_KHR_cooperative_matrix` path.** This Intel part advertises the
  extension (`cooperativeMatrix = true`, compute stage). The existing GLSL
  kernel is NVIDIA-shaped and unwired; wiring a coop-matrix GEMM into
  `brain-backend-vulkan` is a project, not a fix, and it would be the first
  kernel in the tree whose source of truth is not WGSL.
* **`matmul_reg2`'s register tile is sized for a P40.** 64 fp32 accumulators
  per thread at `@workgroup_size(256)` is comfortable on a 255-register
  NVIDIA SM and is most of an Intel Xe thread's GRF. At `1024³` on a ramped
  device it reads 92% of the measured FMA roof, so there may be little left
  here, but a smaller-tile sibling has not been measured on this part.
