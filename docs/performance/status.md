# `brain perf` — implementation ledger

Design: [`benchmarking.md`](benchmarking.md). This file tracks what is built.

## P1 — harness core + Tier-1 scenarios ✅

**Crate `crates/perf`** (`brain-perf`, lib `perf`), 59 unit tests.

| Module | What |
|---|---|
| `stats` | exact nearest-rank percentiles (P50/95/99/99.9), best-of-N + spread, Jain fairness |
| `env` | the result fingerprint: commit/dirty, device, backend, **adapter + `adapter_is_software`**, CPU model/cores, RAM, OS, build profile, perf-relevant env flags |
| `target` | the `PerfTarget` seam (`submit`/`step`/`busy`/`counters`) + `Emission` timeline |
| `workload` | the 8 standard shapes, length distributions, 4 arrival processes, SLOs, `scaled()` for small devices |
| `driver` | single-threaded load driver: submits per the arrival process, records the timeline |
| `metrics` | `ReqRecord` → TTFA/IAL/TPOA/E2E/queue; `Summary` with goodput and SLO attainment |
| `schema` | the `brain.perf/1` artifact + `valid`/`invalid_reason` |
| `report` | one-run render, and `compare` with unit-refusal / invalid-exclusion / axis warnings |
| `scenarios` | `latency`, `throughput`, `serve`, `sweep` + `Options` |
| `targets` | `CapabilityTarget` (any `capability::Provider`) and `PagedLlmTarget` (paged serving engine) |

Three CLI targets: `fake` (harness self-check), **`qwen-synth:<L>x<D>x<H>[xV]`**
(the real paged engine on random weights — same kernels/KV/batching, no
checkpoint needed, so hardware comparison works on any machine), and
`qwen:<weights>` (a real checkpoint).

**CLI** `brain perf {list,run,compare}` (`crates/cli/src/perf_cli.rs`).
**Make** `perf`, `perf/<scenario>`, `perf/compare`, `perf/smoke`.

### Engine changes this required

- **`qwen::serve::Scheduler::step_report`** — `step()` returns only *completed*
  requests, so nothing downstream could observe when a sequence was admitted or
  when each token appeared, making TTFT and ITL uncomputable. `step_report`
  additionally reports admissions and per-sequence token deltas; `step` is now a
  thin wrapper over the same `step_inner`, so behaviour is unchanged. Two tests
  pin it: every token accounted exactly once, and outputs identical to `step`.
- **`Scheduler::free_blocks`** — KV pressure alongside the latencies.
- **`gpu_core::adapter_info` / `backend_name` / `discrete_gpu_count`**, backed by
  `backend_wgpu::adapter_desc` — the adapter was previously *logged to stderr and
  discarded*, so no result could record what hardware produced it.

### Verified

- 59 `perf` unit tests + 16 `qwen` lib tests, green in parallel.
- End-to-end on the `fake` target: `perf run {latency,throughput,serve,sweep}`,
  artifact written, `perf compare` ranks, warns on differing axes, refuses to
  rank across units, excludes invalid runs.
- **Not** verified against a real model: no checkpoint is available in this
  workspace, so the `qwen:<weights>` target path is compiled and unit-covered but
  has not produced a real measurement.

### Known limitations

- The correctness gate (P2) does not exist yet, so every artifact currently
  reports `correctness.passed: null` and the renderer prints
  *"correctness gate did not run — result is unverified"*. This is deliberate:
  an ungated run must never be mistaken for a verified one.
- `resources` (device utilisation, energy) and most of `memory` are `null` — the
  counters do not exist in the engine yet.
- `latency` currently runs through the same closed-loop driver as `serve` with a
  fixed level. A true in-process fixed-batch path that bypasses the driver is
  worth adding when it starts mattering for regression signal.

## Fixed along the way (pre-existing)

- **SIGSEGV in every debug-profile test run that built GPU devices on more than
  one thread.** wgpu's default `InstanceFlags` are `from_build_config()`, which
  enables `DEBUG | VALIDATION` whenever `debug_assertions` is set — so `cargo
  test` silently turned on the Vulkan validation layers and `VK_EXT_debug_utils`
  object naming while `make release` did not. On a software Vulkan ICD
  (lavapipe) `vkSetDebugUtilsObjectNameEXT` faults inside `libvulkan.so.1`.
  Instance flags are now **opt-in** via `BRAIN_GPU_VALIDATION=1`, so debug and
  release build the same instance. Backend construction is also serialised
  (`init_lock`), since concurrent entry into Mesa's EGL/GL loader is unsafe.
- **Multi-GPU tests faulted instead of skipping** on single-GPU / GPU-less boxes:
  seven `dp_parity` / `shard_parity` / `shard_microbatch` / `tensor_parallel`
  tests assumed cards 0..n exist and died inside the driver. They now gate on
  `gpu_core::discrete_gpu_count()` and skip with a message.

## Planned

| Phase | Contents |
|---|---|
| **P2** | `startup`; the **correctness gate** wired into every scenario (reusing `make parity` / `gradcheck`); `perf gate` + committed hard-floor baselines; `--profile edge` scaled matrix |
| **P3** | `residency` ★ (multi-model catalogue > memory, Zipf popularity, eviction regret) and `kvcache` (session lifecycle under KV pressure) |
| **P4** | `placement` (CPU/GPU/Vulkan/NPU, `placement_efficiency`), `mixed` (traffic-class isolation), `overload` (admission control), `cancel` |
| **P5** | `frontend`, `soak`, `faults`, `energy` |

Engine work these depend on, in dependency order:

1. `capability::Provider` adoption by `qwen`, `yolo`, `depth`, `tts` — each makes
   its model benchmarkable through `CapabilityTarget` with no new benchmark code.
2. Cancellation as an engine concept (a request abortable mid-decode) → `cancel`.
3. A pluggable admission policy → `overload`.
4. KV/eviction counters on the paged engine and on `residency` → `kvcache`,
   `residency`.
5. Device and energy counters in `gpu-core` → `resources`, `energy`.
