# `brain perf` — implementation ledger

Design: [`benchmarking.md`](benchmarking.md). This file tracks what is built.

## P1–P5 — all 14 scenarios ✅

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

## First real findings (2×Tesla P40, 48-core host)

The suite's first run on real hardware found a decode bottleneck that is not
in the GPU at all.

**Batching barely helped.** `qwen-synth:8x512x8` (46.8M params, fp32),
decode-bound (8-token prompt, 64-token output), `--device gpu0`:

| concurrency | out tok/s | TTFA p99 | IAL p99 |
|---:|---:|---:|---:|
| 1 | 36.4 | 63 ms | 32 ms |
| 4 | 46.9 | 211 ms | 91 ms |
| 16 | 50.3 | 811 ms | 279 ms |
| 32 | 45.4 | 972 ms | 410 ms |

Throughput gains 1.4× from concurrency 1→16 and then *regresses*, while
latency grows 15×. A 47M fp32 model at ~36 tok/s is roughly **1% of the
P40's fp32 peak** — overhead-bound, not compute-bound.

**Cause: the LM head ran on the host, single-threaded.**
`qwen::serve::Engine::logits` is a scalar Rust matmul over
`[vocab, d_model]` — 16.4M multiply-accumulates per sequence per token at
vocab 32k — executed once per sequence per decode step while the GPU sits
idle. Batch size multiplies that host cost linearly, which is exactly why
batching stops paying.

Confirmed causally by varying **only** vocab (same shape, same device):

| vocab | out tok/s | IAL p50 |
|---:|---:|---:|
| 4 000 | 126.3 | 18.3 ms |
| 8 000 | 100.7 | 23.3 ms |
| 32 000 | 40.6 | 70.3 ms |

`IAL ≈ 11 ms + 1.86 µs per 1000 vocab`, so at vocab 32k about **85% of each
decode step was the host-side head**.

### Fixed: the head now runs on the device

`matmul` (parallel over every one of `bsz × vocab` outputs) followed by a new
`argmax_row` kernel, so the hidden state never leaves the device and only
`bsz` indices are read back instead of a `[bsz, vocab]` block. Prefill keeps
the host head — it runs once per request, not once per token.

Same commands, same hardware, `qwen-synth:8x512x8`, `--device gpu0`:

| concurrency | before | after | gain |
|---:|---:|---:|---:|
| 1 | 36.4 | **74.2** | 2.0× |
| 4 | 46.9 | **155.3** | 3.3× |
| 16 | 50.3 | **234.9** | 4.7× |
| 32 | 45.4 | **290.5** | 6.4× |

Batching now scales **3.9× from concurrency 1→32** (was 1.25× and
*regressing* past 16), and the workload meets its SLO at low concurrency
instead of never — `sweep` reports a real max-sustainable point of
concurrency 4 at 155 tok/s goodput.

Vocab sensitivity is largely gone, which is the direct confirmation:

| vocab | before | after |
|---:|---:|---:|
| 4 000 | 126.3 | **176.9** |
| 8 000 | 100.7 | **168.7** |
| 32 000 | 40.6 | **114.6** |

8× vocab now costs 1.6× IAL (10.8 → 17.7 ms) instead of 3.8×.

Correctness is pinned by `device_head_argmax_matches_the_host_head`, and the
pre-existing token-for-token reference tests
(`batched_serving_matches_reference`,
`scheduler_dynamic_admission_matches_reference`) still pass — the engine
generates identical text, faster. Verified on both backends: the same WGSL
runs on the CPU backend via the Cranelift JIT.

**Still on the table**: at 290 tok/s for a 47M fp32 model the P40 is still far
from its ~11 TFLOP/s peak, so the next bottleneck is worth finding — likely
per-step dispatch/sync overhead, since IAL p99 stays ~48 ms at batch 32.

*(Numbers are `--target qwen-synth`, i.e. random weights — valid for cost,
meaningless for output quality.)*

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

## Tier 2 — what each scenario can and cannot see today

All ten are implemented and run. Each is explicit about its limits: where a
metric needs an engine capability that does not exist, the field is `null` and
the artifact carries a `notes` string. A confident number nobody measured is
worse than an honest gap.

| scenario | measures today | blocked on |
|---|---|---|
| `startup` | device init / weights / first prefill, cold vs second build | a pipeline cache — there is no warm path, and the two rows matching is the evidence |
| `mixed` | per-class goodput, P99 TTFA/IAL, **normalised slowdown** vs an isolated baseline, Jain fairness | — |
| `overload` | capacity, then 0.5–4× offered load; goodput, queue P99, wasted admissions | a pluggable **admission policy**: nothing is ever rejected, so policies cannot be compared |
| `cancel` | abort latency, block reclaim, waste, neighbour interference | an async client-disconnect path (transport-level) |
| `kvcache` | admission pressure: KV stalls and stall time under a working set 3× the pool | **prefix caching / cross-request block reuse** — without it hit-rate and eviction-regret are structurally null |
| `residency` ★ | warm/cold TTFA, weight-cache hit rate, **eviction regret**, per-model fairness under a shifting Zipf | wiring to real `ResidentModel`s for true load latency (activation is modelled from size) |
| `placement` | `placement_efficiency` = best combined / best single | nothing — but `--device` is process-global, so it analyses per-device artifacts |
| `frontend` | JSONL encode/decode and chat-template cost; host cores per saturated device | tokenizer + media inputs for the remaining stages |
| `faults` | device-OOM injection, detection, no silent corruption | a **multi-rank harness** for worker death, hung ranks, collective timeouts, corrupt KV |
| `soak` | throughput/latency/memory/KV series; drift per hour | nothing — but it **refuses to extrapolate** below 600 s |

Cross-cutting: `fidelity` (greedy-token gate; a failing run is written
`valid: false` and excluded from `compare`) and `energy` (external power
sampling — an unreadable meter yields `null`, never a fabricated zero).

### Second wave of findings (2×Tesla P40)

- **`residency`: 64% eviction regret.** 24 models at 4× overcommit under a Zipf
  load whose popularity shifts mid-run: LRU evicts models wanted again almost
  immediately about two thirds of the time, hit rate 48%, fairness 0.494 with
  every model still served. That is the "bad policy" signature rather than the
  "cache too small" one — the distinction the regret metric exists to draw.
- **`startup`: there is no warm start.** A second engine built in the same
  process costs the same as the first (~3.1 s vs ~3.5 s, i.e. no reuse), because
  every `Engine` constructs its own `Gpu` and recompiles every WGSL pipeline.
  Device init ~0.3 s, weights ~1.8 s, first prefill ~1.0 s.
- **`frontend`: the host is not the bottleneck.** JSONL + templating cost is
  negligible against a ~290 tok/s device rate (well under 0.01 cores), so the
  ceiling found earlier is genuinely in the engine.
- **`cancel` is clean**: zero waste, zero leaked blocks, neighbours unharmed.
- **An engine crash, found by `kvcache`.** A prompt longer than
  `max_blocks_per_seq * block_size` wrote **past its row of the block table**
  (index 82944 of 82944) — silently corrupting the next sequence's mapping
  before it panicked. Now `Engine::max_seq_len` bounds it and the scheduler
  rejects oversized requests at admission, reporting them in
  `StepReport.rejected`, so the queue keeps moving instead of crashing or
  blocking forever.

## Still planned

1. `capability::Provider` adoption by `qwen`, `yolo`, `depth`, `tts` — each makes
   its model benchmarkable through `CapabilityTarget` with no new benchmark code.
2. A pluggable admission policy → makes `overload` able to *compare* policies.
3. Prefix caching / block reuse → makes `kvcache`'s hit-rate and eviction-regret
   real rather than structurally null.
4. A pipeline cache → gives `startup` a warm path worth measuring.
5. Device utilisation counters in `gpu-core` → fills the `resources` block.
6. `perf gate` + committed hard-floor baselines, and the fidelity gate wired
   into every scenario rather than available to them.
