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


### Third wave: decode-regime kernels + the fidelity gate (all landed)

Same sweep, same hardware (`qwen-synth:8x512x8`, Tesla P40, decode-heavy):

| concurrency | session start | + device head | + prefill budget | **+ decode kernels** | total |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.4 | 74.2 | 72.6 | **154.0** | 4.2× |
| 4 | 46.9 | 155.3 | 154.5 | **381.7** | 8.1× |
| 16 | 50.3 | 234.9 | 234.1 | **632.7** | 12.6× |
| 32 | 45.4 | 290.5 | 289.6 | **865.5** | **19.1×** |

IAL p99 at batch 32: 410 ms → **11.8 ms**. The interactive SLO now holds
through concurrency 16 at 632 tok/s goodput (session start: no level at all).

What landed: `matmul_gemv` (workgroup-per-column, W streamed once across all
rows — the 67.5% kernel), `rmsnorm_rows` (workgroup-per-row — the 16.6%
kernel), `argmax_part/final` (two-stage reduction — the 10.3% kernel), each
selected per dispatch by row count so training/prefill shapes keep the
per-element kernels. Gated off the CPU backend, whose JIT barrier-split model
mis-executes them and whose native AVX2 fast paths already cover the regime —
`Backend::kind()` makes the selection explicit.

**Every number above is gated**: the fidelity check (batched-vs-sequential
greedy through the same engine) now runs inside every Tier-1 scenario and
reported `greedy_token_match: 1.0` — the artifacts are marked valid, not
"unverified".


### Fourth wave: policy seams (landed)

- **AdmissionPolicy** (`qwen::serve`): `UnboundedQueue` (default, historical),
  `MaxQueueDepth`, `DeadlineAware` (EWMA-fed). Consulted at submit; typed
  `RejectReason` in `StepReport::rejected`. Policies are pure functions of
  `QueueState` — unit-tested with no engine. Remaining: a `--admission` flag so
  `perf overload` compares the ladder per policy.
- **EvictionPolicy** (`residency::place`): `Lru` + `CostAware` (GDSF,
  `uses x bytes / age`); `plan_eviction_with`, benchmark scores the real code
  via `--policy`. Hit rate 50.0% -> 54.3% on identical seeds; cheap-before-
  expensive eviction pinned by test. Event-counted regret is metric-limited at
  4x overcommit — a bytes-weighted regret metric is the follow-up.

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

## Mitigations

Every finding above has a designed mitigation with a root cause, an API sketch
and a verification criterion in **[`mitigations.md`](mitigations.md)**, organised
around reaching *any* device's ceiling rather than this box's.

Profiling shows decode is **64% GPU-busy**, so it is not host stalling. Three
readings, in increasing order of importance:

1. The hot kernels (`matmul` 67.5%, `rmsnorm` 16.6%, `argmax_row` 10.3%) all
   parallelise over output *rows* — right for training, catastrophic for decode.
   `rmsnorm` runs **8 threads on 3840 cores** at batch 8.
2. Decode is **memory-bound, not compute-bound**: at M=8 arithmetic intensity is
   ~0.25 FLOP/byte whatever the tiling, so *tiling cannot fix it*.
3. Therefore the first-order lever is **bytes moved**, and it is the one lever
   every device class shares. `matmul_i8` (DP4A, 4×) and `qwen::q8` already
   exist — but `serve.rs` uses int8 for the **KV cache only, never for weights**.

The structural blockers are that `Backend` exposes almost no device capability,
kernel choice is fixed at model-construction time, tile sizes are literals in
281 hand-written files, and the fp32-only invariant — which is what buys the
old-GPU/WebGPU guarantee — is simultaneously the ceiling on modern hardware.
`mitigations.md` Part I addresses those; Part II expresses each finding through
them.

## Closed since (suite reliability + duplication)

- **Test-suite deadlock and exit-crash fixed** (commit 2fea497): one device per
  process via explicit `share`/`new_like`/`WeakGpu`; the weak-pool test fixture
  (`gpu_core::testgpu`) lets the device die with its last in-process handle —
  the only teardown shape this NVIDIA driver survives. 30/30 parallel runs
  clean at 8 and 48 test threads (was ~50% deadlock).
- **rayon centralised** (5b0e539): `backend_cpu::par` is the one home for
  host-parallel loops; `grep -l '^rayon' crates/*/Cargo.toml` returns exactly
  `backend-cpu`.
- **Host math centralised** (508d84d): `model::hostmath`, parity-tested against
  the WGSL kernels through the CPU backend.
- **ONNX emission centralised** (0f660d7): `npu::topo::TopoBase` — one DSL and
  one rmsnorm/layernorm/silu emitter across the eight topology builders.

## Closed since (the portability spine + the remaining serving mitigations)

One session (commits 4ca517b..07f28e2), in roadmap order:

- **S1 `DeviceCaps`** (4ca517b): a queryable capability model on every backend —
  class, limits, subgroup width, unified memory, and a `NumericSupport` tier
  list where every value is queried, never assumed (fast-f16 stays `false`
  until measured: Pascal exposes f16 at 1/64 rate). Recorded machine-readable
  in every perf artifact's env block. Also fixed en route: the wgpu and Vulkan
  backends never overrode `max_storage_binding_bytes`, so every caller saw a
  fixed ~2 GiB instead of the device's real limit.
- **S2 `KernelSelector`** (ff2612c): one pure, memoised policy for which kernel
  variant runs per (op, shape, caps) — the scattered `kind() != "cpu"` regime
  tests are gone, and the CPU JIT's inability to run workgroup reductions is an
  honest capability (`workgroup_reductions`), not a string compare.
- **A0 int8 serving weights** (5675a08): `--weights-int8` / target suffix
  `:i8w`; the 7 per-layer linears + LM head quantize at load (per-channel,
  packed 4/u32), activations quantize per forward, fp32 copies are not kept.
  Measured on the P40 (decode_heavy): +32% throughput at c=16 (753 vs 570
  tok/s) and TTFA p99 631 ms vs 2386 ms — but SLOWER at c=1 (78 vs 127),
  because a 128×128 tile is mostly idle at m=1. That gap forced the next item:
- **`matmul_i8_gemv`** (same commit): the decode-shaped packed GEMV; int8 c=1
  now 131 tok/s. The GEMV/tile crossover is measured at m≈8 and lives in the
  selector (`I8_GEMV_MAX_ROWS`) — refining it per-device is S5's job.
- **F1/F2 warm start** (5675a08): `Engine::from_map_on` (one device per
  process, engines share via `new_like`) + a persisted per-adapter driver
  pipeline cache. `startup`'s two identical rows finally separate: device
  953.9 → 239.7 ms, total 4117 → 2761 ms.
- **Admission comparison** (5675a08): `overload --admission
  {unbounded|depth:N|deadline:ms}` installs real engine policies; rejections
  flow through `EmissionKind::Rejected` — terminal, never goodput, excluded
  from the SLO denominator. An uninstallable policy is an error, not a silent
  fallback. `residency` adds **bytes-weighted** eviction regret (a good policy
  makes the regretted evictions the cheap ones) and fixes a regret-attribution
  bug: a re-request now resolves its own eviction entry, not the most recent.
- **A4 on-device decode window** (42ab2fa): decode feeds the argmax back and
  advances paged metadata on the device for up to 4 tokens per host
  round-trip; windows engage only when nothing is waiting, and a mid-window
  EOS's surplus K/V rolls back. fp32 c=1 127 → 137 tok/s, c=4 286 → 327.
  Also implemented `dot4I8Packed` in the CPU JIT (it previously failed
  `Jit::new` hard on any kernel using it).
- **D prefix cache** (ed3f516): full prompt blocks indexed by (parent physical
  block, token ids) — exact by construction, no hash to collide; prefill
  adopts the longest chain and computes only the tail; LRU eviction under
  pool pressure, live sequences outrank the cache. `kvcache`'s hit-rate
  counters are now real (`kv_prefix_hit_rate`, null until measured). Writing
  its property test exposed a real hole: token ids were never validated, and
  an out-of-vocab id sent the embedding gather out of bounds — silent garbage
  under the trusted kernels. Now a typed `RejectReason::InvalidToken` at
  admission plus a prefill backstop.
- **testgpu adoption** (07f28e2): qwen/gpt/tts/speaker test construction on
  the shared weak-pool device (qwen proven at `--test-threads=8` on the P40);
  codec deliberately stays on its production CPU pin.

## Still planned

1. `capability::Provider` adoption by `qwen`, `yolo`, `depth`, `tts` — each makes
   its model benchmarkable through `CapabilityTarget` with no new benchmark code.
2. **S3** parameterised kernels (WGSL `override` / template step) and **S5**
   the autotuner-as-selector — the measured refinement of every boundary the
   static policy hard-codes (the i8 GEMV/tile crossover first).
3. Device utilisation counters in `gpu-core` → fills the `resources` block (K).
4. `perf gate` + committed hard-floor baselines (J2); `FaultSink` injection
   points (G); real tokenizer/media stages in `frontend` (I); multi-model
   `placement` reframing (H).
5. Raise the global `TEST_THREADS` once the remaining GPU-test crates
   (glm, moe, vision, depth, …) adopt `gpu_core::testgpu`.
