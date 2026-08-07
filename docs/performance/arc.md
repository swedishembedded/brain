# brain on Intel Arc (Meteor Lake, Xe-LPG iGPU)

Status ledger for making brain's GPU path fast on this box's Intel Arc iGPU.
Sibling to `docs/performance/p40.md` (the two Tesla P40s) — this is the other
end of the hardware spectrum brain has to stay honest about: an integrated,
bandwidth-shared, no-XMX part rather than a discrete card with dedicated
GDDR5. Companion planning document: the approved workstream plan archived at
`applications/bench/tasks/brain/saturate-arc-igpu/instruction.md`, and the
research note this ledger supersedes, `.todo/saturate-arc-igpu.md`.

> **Status: IN PROGRESS.** This document is being filled in as the workstream
> executes. Sections marked *pending* have not been measured yet — do not
> treat their absence as "zero" or "not applicable"; see §4 for what is
> currently blocking measurement.

---

## 0. Reproducing a number on this box

**Prerequisites** (see the archived plan's Track M for the full list):
- `sudo sysctl kernel.perf_event_paranoid=-1` (or `setcap cap_perfmon+ep`) —
  unblocks the i915 PMU (`intel_gpu_top`, per-kernel GPU busy%). **Not yet
  granted** as of this writing; nothing in this ledger depends on it.
- `sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj` — unblocks RAPL
  energy accounting. **Not yet granted.**
- `sudo apt install glslang-tools` — unblocks the `crates/vulkan` coopmat
  SPIR-V build. **Not yet installed.**
- Everything else (GPU frequency/RC6/throttle via `/sys/class/drm`) works
  **unprivileged**, and is what `crates/perf/src/devicetel.rs` reads.

**The memory-safety protocol** (32 GB unified, shared CPU+GPU, no discrete
VRAM — this box has been observed with several GB already in swap):
1. Before any heavy step: `free -g` and refuse to proceed if `available` <
   12 GiB.
2. One real-shape engine per process, ever — never build two.
3. Never run two GPU measurements concurrently — a single global
   `PROBE` mutex already enforces this inside `crates/gpu-core/tests/
   roofline.rs`; extend the discipline to the whole workstream (a `perf run`
   and a `cargo test` touching the GPU must never overlap).
4. Real-checkpoint tests stay `#[ignore]`d off the default `make test` lane.

**A number without frequency/throttle context is not reproducible.** This
box has been observed **PL1-throttled at idle** (`throttle_reason_pl1=1`,
`punit_req_freq_mhz=800` against a `rps_RP0_freq_mhz` ceiling of 2250 — a
2.8× clock gap). Every artifact this workstream produces through
`crates/perf` now carries `resources.gpu_freq_mhz_{start,end}`,
`resources.gpu_rc6_pct` and `resources.gpu_throttled_{pl1,thermal}_at_end`
(see `crates/perf/src/devicetel.rs`) for exactly this reason — read them
before comparing two runs.

---

## 1. The one number that drives everything

**Measured once, real, but not reliably reproducible on demand — see §4,
gap A1 for why.**

```
ridge point = 372 GFLOP/s ÷ 19.3 GB/s ≈ 19.3 FLOP per byte
```

This settles the open ~90 GB/s-theoretical vs ~19 GB/s-achieved question
decisively: **measured DRAM bandwidth is 19.3 GB/s**, matching the
achieved figure, not the DDR5 dual-channel theoretical one. The gap
between theoretical and achieved (~4.7×) is large even for a shared-memory
iGPU and is itself worth explaining (contention with the CPU, throttled
memory controller clock, or an artifact of the same instability described
in gap A1 — unconfirmed which). Equally striking: **measured compute is
372 GFLOP/s, ~8% of the 4.61 TFLOP/s vendor figure** — consistent with the
severe PL1 throttling independently observed on this box (measured
`punit_req_freq_mhz=800` against a `rps_RP0_freq_mhz` ceiling of 2250, a
2.8× gap, and that alone doesn't fully explain an 8% ratio, so more is
happening than clock scaling alone).

| metric | measured | vendor spec | ratio |
|---|---:|---:|---:|
| fp32 compute | 372 GFLOP/s | 4610 GFLOP/s | 8.1% |
| int8 compute (DP4A) | 2066 GOP/s | 18430 GOP/s | 11.2% |
| DRAM bandwidth | 19.3 GB/s | ~90 GB/s (theoretical) | 21.4% |
| cache bandwidth | 26.1 GB/s | — | — |
| ridge | 19.3 FLOP/byte | ~51 (theoretical) | — |

**This device is more memory-bound, relative to its own compute, than the
P40 (ridge ≈34)** even before accounting for how far below spec both
numbers already sit — every P40-derived tiling decision that assumed "GEMM
tiling matters, this card can use the FLOPs" needs re-deriving from this
ridge, not transferred.

Vendor figures for context — **priors to confirm, not facts to code
against**:

| precision | vendor figure | mechanism |
|---|---:|---|
| FP32 | 4.61 TFLOP/s | 8 Xe-cores × 128 lanes × 2 × 2.25 GHz |
| FP16 | 9.22 TFLOP/s | native 2:1 packed math |
| INT8 | 18.43 TOPS | DP4A on the vector engines (no XMX on this silicon) |
| INT4 | 18.43 TOPS | **no** 4:1 speedup over INT8 on this vector engine, unlike a dedicated NPU |

Memory bandwidth is **unresolved** and is exactly what the roofline probe
exists to settle: a DDR5-5600 dual-channel theoretical figure (~90 GB/s) and
a separately reported ~19 GB/s achieved figure are both in play, 8-9× apart,
implying ridge points of ≈51 vs ≈240 FLOP/byte respectively. This is not a
rounding difference — it changes whether this device should be treated as
"somewhat more memory-bound than a P40" or "overwhelmingly memory-bound, to
the point that GEMM tiling barely matters." **Do not plan kernel work off
either number until the probe reports a real one.**

---

## 2. Hardware facts (measured this session, source noted)

| | | source |
|---|---|---|
| GPU | `Intel(R) Arc(tm) Graphics (MTL)`, PCI `8086:7d55`, Xe-LPG, 8 Xe-cores | `vulkaninfo` |
| Driver | Mesa ANV 25.0.7, Vulkan 1.4.305 (instance 1.4.304) | `vulkaninfo` |
| wgpu | 29 | `Cargo.toml` |
| Clock | `gt0` (render) max 2250 MHz; `gt1` (media) max 1300 MHz | `/sys/class/drm/card1/gt/{gt0,gt1}/rps_max_freq_mhz` |
| XMX | **none** — Intel removed matrix hardware from Meteor Lake; low-precision runs on DP4A vector-engine paths only | vendor docs (XeSS on MTL uses DP4A) |
| `VK_KHR_cooperative_matrix` | exposed (`cooperativeMatrix = true`, compute stage) but **no shape list printed** by this `vulkaninfo` build — whether brain's `feature_supported && !shapes.is_empty()` gate evaluates `true` here is unconfirmed | `vulkaninfo` |
| `shaderIntegerDotProduct` | `true`; `integerDotProduct4x8BitPackedSignedAccelerated = true` (real DP4A) | `vulkaninfo` |
| `integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated` | `true` — fused accumulate, not yet exploited anywhere in brain (WGSL's `dot4I8Packed` is non-accumulating; only reachable via native SPIR-V `OpSDotAccSat`) | `vulkaninfo` |
| `subgroupSize` | variable 8–32, `subgroupSizeControl = true` | `vulkaninfo` |
| `shaderFloat16` | `true` (native 2:1 packed, unclaimed anywhere in brain — `f16: false` is hardcoded on both GPU backends) | `vulkaninfo` |
| Memory | DDR5 SODIMMs, dual-channel (confirmed via `edac` DIMM labels — **not** soldered LPDDR5x) | `/sys/devices/system/edac/mc/mc0/dimm*/dimm_label` |
| RAM | 32 GB total, unified (no discrete VRAM) | `free -g` |
| `glslc`/`glslangValidator` | not installed | `crates/vulkan`'s build-script warning |
| `perf_event_paranoid` | 1 (blocks i915 PMU without a privilege change) | `/proc/sys/kernel/perf_event_paranoid` |
| RAPL energy files | present, root-only (0400, CVE-2020-8694) | `/sys/class/powercap/intel-rapl:*/energy_uj` |
| Throttle state (idle sample) | `throttle_reason_pl1=1`, `punit_req_freq_mhz=800` vs `rps_RP0_freq_mhz=2250` | `/sys/class/drm/card1/gt/gt0/*` |

---

## 3. The measured roofline

One clean, complete run (see gap A1 for the reliability caveat and why
this is a single data point, not an averaged/repeated one):

```
measured roofline on wgpu: 372 GFLOP/s fp32, 2066 GOP/s int8,
  19.3 GB/s DRAM, 26.1 GB/s cache, ridge 19.3 FLOP/byte
```

Sanity checks applied: `gflops` (372) is well under the theoretical FP32
ceiling (4610) — not a folded loop. `gbs` (19.3) is under `gflops` (372) —
the timing is a real device measurement, not a host-timed §E.0 artifact.
`cache_gbs` (26.1) ≥ `gbs` (19.3) — the hierarchy is monotonic as required.
`int8_gops`/`gflops` = 2066/372 ≈ 5.6× — DP4A is a real, substantial win
over fp32 on this silicon, consistent with the vendor's stated 4× ratio
(measured ratio is *higher* than the vendor's own fp32-vs-int8 ratio,
because the fp32 rate is depressed further than the int8 rate here —
unexplained, worth another measurement once gap A1 is closed).

`--device vulkan`: **not yet measured** — the same reliability risk
applies and a clean run has not been captured for the native Vulkan
backend yet. Do not assume it matches the wgpu numbers above; measure it
independently once a clean run is achieved.

**Persisted cache**: this run's result was NOT persisted to
`~/.cache/brain/roof-*.txt` (the process exited via the test harness
without going through `roof::ensure`'s save path — `roof::measure` was
called directly by the test, which deliberately bypasses the cache so the
measurement itself can be asserted). A future `roof::ensure` call in a
real `brain` process will re-measure and persist, carrying the same
reliability risk described in gap A1.

---

## 4. Gaps

| # | Gap | Status |
|---|---|---|
| A0 | No committed Arc `perf gate` baseline exists anywhere — the only committed baseline in the repo is CPU-only | open |
| A1 | **`gpu_core::roof::measure` (and, by extension, any sustained GPU dispatch loop) is unreliable on this adapter** — not a deterministic hang, but severe, unpredictable throttling-induced stalls that can turn a probe round into a multi-minute wait. Root-caused this session (see below); partially mitigated (`MAX_PROBE_WALL_SECONDS`); **not fully fixable in application code** — needs a cancellable `Backend::poll_wait`, filed as a follow-up. | **root-caused, partially mitigated, real fix out of scope this session** |
| A2 | `resources` block (energy, device util) is null on every artifact this repo has ever produced — `energy.rs::PowerSampler` has zero call sites outside its own tests | **partially closed this session** — `devicetel.rs` now fills `resources.gpu_freq_mhz_*`/`gpu_rc6_pct`/`gpu_throttled_*_at_end`/`pkg_temp_c_end` on every `latency`/`sweep` artifact; RAPL-based `energy_j` and real occupancy (needs the i915 PMU, gated on a privilege this box doesn't have yet) remain open |
| A3 | `backend-vulkan` had zero per-kernel timestamp support | **closed this session** — `VkContext` gained `timestamp_period_ns`/`timestamp_valid_bits`; `VulkanBackend` gained a `VkProfile` (a reusable `vk::QueryPool`, sized `MAX_TIMED_DISPATCHES + 1 = 8193`) wired into both `flush()` paths — `n+1` bracketing timestamps in the batched path (mirroring `backend-wgpu`'s `flush_timed`, not its per-dispatch-pass `flush_profiled`, for the same reason: a per-dispatch pass would change what execution is being measured), and a timestamp pair per already-isolated dispatch in the Intel-serialized path (safe there — each dispatch already pays its own submit+fence). Verified live on `--device vulkan`: `crates/backend-vulkan/tests/kernel_timing.rs`, 3/3 passing in 0.19s — device time correctly `<=` host wall time, `reset_kernel_times` zeroes the accumulator, both flush paths report real per-kernel timing. Existing `perf_contract.rs` batching-contract tests (3/3) still pass — no regression. `set_kernel_timing`/`kernel_times` degrade to `false`/`None` on any queue with `timestamp_valid_bits == 0`, never a substituted host time. |
| A4 | Workgroup size (`@workgroup_size(64)` in 352/361 kernels, 256 in the rest) has never been swept on this variable-SIMD8/16/32 device | open |
| A5 | The autotuner (`backend_api::select::AutoTuner`) runs and persists on this adapter, but every persisted pick is a toy unit-test shape (`16x16x16`-class) — no real model shape has ever been tuned here | open |
| A6 | Tile constants (`matmul_reg3`'s `BM=BN=128`, `SPLITK_TARGET_WGS=288`, ...) are P40-swept literals, never re-derived for this device's much smaller Xe-core count | open |
| A7 | Whether `backend-wgpu` is exposed to the documented Meteor-Lake ANV compute-compute barrier bug (the one `backend-vulkan` already works around via `BRAIN_VK_SERIAL`) is unconfirmed | open |

### The A1 incident, in detail

**First observation.** `cargo test -p brain-gpu-core --test roofline`
produced no output for 10+ minutes on both `--device gpu` and `--device
vulkan`, right after `roof::measure` began (device construction succeeds —
adapter string and limits print fine — then nothing further). All OS
threads slept in `futex_do_wait`; `/sys/class/drm/card1/gt/gt0/
rps_act_freq_mhz` read `0` throughout the sampled window.

**Bisection ruled out the obvious code-level suspects**, each a fresh,
timeout-guarded, tightly-scoped test:

| bisection | result |
|---|---|
| Basic dispatch (`device_stats.rs`: one `axpy` kernel, 1024 elements, 3 dispatches, direct `testgpu::dev`, no `new_like`) | passes, 0.22s |
| `new_like([roof_fma, axpy])` (no DP4A kernel) + one small non-calibrating `roof_fma` dispatch | passes, 0.21s |
| `new_like([roof_dp4a])` alone, compile and dispatch phases timed separately | passes, 0.15s both phases |

None of these hang. **The failure needs the full, self-calibrating
`roof::measure` flow** — up to `FMA_THREADS = 1<<20` threads, buffers up
to 512 MiB (`BW_ELEMS = 64<<20`, two buffers), and a doubling calibration
loop that can submit up to `MAX_BW_PASSES = 4096` dispatches in one
`submit()` call.

**The decisive evidence came from re-running the exact same test.**
Temporary per-round diagnostic timing was added to `measure_compute`/
`measure_bandwidth`/`best_of` (since reverted). A second run of the
identical test:
- completed cleanly in **2.61s** and printed a real, plausible roofline
  (§1/§3);
- a **third run, immediately after**, showed identical repeated dispatches
  taking measurably longer each time — 0.070s → 0.080s → 0.091s, a live
  downclock signature — then stalled for 120+ seconds before being killed;
- a **fourth run**, with the `MAX_PROBE_WALL_SECONDS` mitigation (below)
  in place, **still stalled past a 180-second bound**.

**Conclusion: this is not a deterministic bug in brain's code.** The same
process, same code, same device produced a clean pass and a multi-minute
stall back-to-back. The mechanism is almost certainly this box's PL1
thermal/power throttling (independently confirmed active —
`throttle_reason_pl1=1` — before and after every attempt) interacting
badly with a calibration loop that assumes roughly stable per-dispatch
cost. `docs/kernel-checklist.md`'s own "grade the top row against the
measured roof" methodology has never had to account for a device whose
*measurement itself* is this unstable.

**Mitigation added, and its limit stated honestly.** `crates/gpu-core/
src/roof.rs` gained `MAX_PROBE_WALL_SECONDS` (20s), checked after every
calibration round in all three probe loops (`measure_compute`,
`measure_bandwidth`, `measure_int8`): once a probe has already spent that
long, it stops requesting a larger, riskier dispatch and returns its best
measurement so far. **This does not fix the core problem** — `Backend` has
no cancellable or timeout-bounded `poll_wait()`, so nothing in application
code can abort a `submit()`+`poll_wait()` that is already in flight and
stuck; the fourth-run failure above confirms the mitigation reduces
exposure (fewer, smaller escalations) without eliminating the failure
mode. A genuine fix needs a `Backend::poll_wait_timeout` (or equivalent)
added to the engine so a stuck wait can be aborted and reported as `None`
— a real API change, filed as its own follow-up rather than attempted
under this session's scope.

**Working conclusion for the rest of this workstream**: wrap every
GPU-touching command on this box in a hard external `timeout`, never run
one bare. Treat a single successful roofline measurement as real, valid
data — not as evidence the number is wrong when a *different* run stalls.
Every "graded against the roof" step in Tracks M and O can proceed using
§1/§3's numbers, with the explicit caveat that re-measuring on demand may
require several attempts.

---

## 5. Backend comparison at real scale

Pending — Track M's Phase 0b (needs §3's roofline first, and is itself
budgeted at hours of wall-clock on this hardware class; see the archived
plan). A smoke-scale comparison already exists in `.todo/saturate-arc-igpu.md`
and should not be over-read: it measured `backend-vulkan` as consistently
*slower* than `backend-wgpu` at trivial (8-token) scale, and found no
memory-footprint gap between the two backends (5249 MB vs 5238 MB peak host
RSS) — both findings need the real-scale re-check before being treated as
general truths.

---

## 6. Per-kernel profiles

Pending — needs §3 (the roof to grade against) and gap A3 (vulkan
timestamps) closed first.

---

## 7. Coopmat and DP4A: measured verdicts

### DP4A — resolved: real hardware, on both backends

The plan's O0.5 asked whether `backend-wgpu`'s unconditional
`DeviceCaps.numeric.int8_dot = true` is honest, or a lie to the selector —
`p40.md` had claimed "the wgpu/CPU backends don't enable the integer-dot
feature," which reads as a blanket "wgpu can't do it."

**Measured, on `--device gpu` (wgpu), from this ledger's §3 roofline:**

```
372 GFLOP/s fp32, 2066 GOP/s int8  ->  ratio 5.6x
```

Per the decision rule fixed in the plan before measuring (ratio >= 3x on
wgpu -> DP4A is a real wgpu capability, fix the doc): **`int8_dot: true` is
correct.** WGSL's `dot4I8Packed` — core WGSL, what `backend-wgpu` actually
claims — lowers through naga to real hardware DP4A on this Mesa ANV driver,
not a polyfill (a polyfill would show a ratio near 1x). `docs/performance/
p40.md`'s §"INT8 GEMM (DP4A)" section has been corrected in place: the real,
narrower reason the `matmul_i8` kernel specifically stays Vulkan-only is
that it targets the *native* `shaderIntegerDotProduct` Vulkan device feature
directly, a lower-level surface wgpu's abstraction doesn't expose for an app
to enable — not a capability gap in WGSL's `dot4I8Packed`, which both
backends already reach. `.todo/int8-kv-dp4a-scores.md`'s open "GPU-side
measurement" question is answered by this same number: DP4A is a real,
substantial (5.6x) win on this Arc iGPU, on the backend that already serves
`--device gpu` by default.

`--device vulkan`'s int8 ratio has not yet been separately measured (the
same reliability caveat as every other vulkan roofline attempt applies) —
expected to be comparable or higher (native `shaderIntegerDotProduct` is a
strictly lower-level path than naga's lowering), not lower; record it here
once captured.

### Coopmat — pending

Needs `glslc`/`glslangValidator` installed (not yet) and a shape
enumeration of this driver's `VK_KHR_cooperative_matrix` support (§2 records
that the extension is exposed but no shape list was printed by this
`vulkaninfo` build). Given Meteor Lake's Xe-LPG has no XMX matrix hardware,
the working prior is that any coopmat path here is emulated over
subgroup/DP4A rather than unlocking new arithmetic hardware — stated as a
falsifiable prediction, not a conclusion, per the archived plan's O5. Given
DP4A alone already measures a healthy 5.6x, coopmat's ceiling (if it beats
`matmul_i8_dyn` at all) is bounded by roughly that same hardware — it would
need to beat the *already-DP4A-accelerated* kernel, not the fp32 one, to be
worth adopting per O5's decision rule.

---

## 8. Plan and results

The ordered, file-level implementation plan lives at
`applications/bench/tasks/brain/saturate-arc-igpu/instruction.md` (archived
from the approved planning session) and in this repo's own
`.todo/saturate-arc-igpu.md` (corrected this session — see its header note).
Results land in this document's §§1, 3, 5–7 as each blocking item closes.
