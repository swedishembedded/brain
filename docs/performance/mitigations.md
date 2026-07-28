# Mitigation plan: portable peak compute utilisation

Every finding in [`status.md`](status.md), with a root cause established from
profiling, a designed mitigation, and how it will be verified.

**The plan is organised around portability, not around the card in this box.**
A first draft of this document optimised for a Tesla P40 — a specific fp32
GEMV kernel, a specific tile size. That is the wrong shape of answer: brain's
value is one engine across CPU, old and modern discrete GPUs, integrated GPUs,
NPUs and WebGPU, and a fix that only helps one of those is a regression in
everything the project is for. So Part I is the *structural* work that lets any
device reach its own ceiling, and Part II is the individual findings expressed
through those seams.

Two rules apply throughout:

1. **Nothing lands without its gate.** Kernel-math changes run `make gradcheck`
   and `make parity`; serving changes keep the token-for-token reference tests
   green. A performance change that alters output is a bug, not a trade-off.
2. **The benchmark that found it is the benchmark that closes it.** Each item
   names the `brain perf` scenario and the number that must move.

---

## Part 0 — What the profile actually says

`BRAIN_PROFILE=1`, `qwen-synth:8x512x8`, batch 8, Tesla P40:

```
GPU kernel time total 1402.5 ms  =  64.3% of wall
  matmul                947.0 ms  2047 calls  (67.5%)   0.46 ms/call
  rmsnorm               232.3 ms  1188 calls  (16.6%)   0.20 ms/call
  argmax_row            144.1 ms    31 calls  (10.3%)   4.65 ms/call
  everything else        79.1 ms                        ( 5.6%)
dispatches 6182, submits 36, readbacks 36
```

The GPU is *busy* (64% of wall), so this is not host stalling. Three readings,
in increasing order of importance:

**(a) The kernels are shaped for training.** Every hot kernel parallelises over
output rows or elements — right when M = batch × seq is thousands, catastrophic
when M is the number of concurrent sequences. `rmsnorm` is one thread per row:
**8 threads on 3840 cores** at batch 8.

**(b) Decode is memory-bound, not compute-bound.** A `[8,512]×[512,512]` matmul
at 0.46 ms is 9.1 GFLOP/s — 0.08% of the P40's ~11 TFLOP/s — but also only
~7 GB/s against 346 GB/s of bandwidth. At M=8 the arithmetic intensity is ~0.25
FLOP/byte *no matter how you tile it*. **Tiling cannot fix a bandwidth problem.**

**(c) Therefore the first-order lever is bytes moved, and it is portable.**
Every device in brain's range — CPU, integrated GPU, discrete GPU, NPU, WebGPU —
is memory-bound during decode, because decode streams the whole weight set per
token. Halving the bytes helps all of them. Widening the math helps only the
devices that have wider math. That ordering is what makes this plan portable
rather than P40-shaped.

And the repo already has the machinery: `matmul_i8` (DP4A, documented as *"the
P40's fastest inference path"*, 4× the MACs of fp32 FMA and 4× fewer weight
bytes) and `qwen::q8` (per-channel symmetric int8 weight quantisation, already
used by the z-image encoder). **`serve.rs` uses int8 for the KV cache only —
never for weights.** The serving engine runs fp32 weights on a card whose
fastest path is int8.

---

## Part I — Structural gaps that cap every device

These are the reasons brain cannot currently reach *any* device's ceiling. They
are prerequisites: without them, each per-device optimisation is another
hand-written kernel and another `if` in a backend.

### S1. There is no device capability model

`backend_api::Backend` exposes exactly one property of the hardware —
`max_storage_binding_bytes()`. Nothing else: not compute-unit count, not
bandwidth class, not shared-memory size, not subgroup width, not whether the
device has fast f16, int8 dot product, or cooperative matrices. A kernel
selector cannot make a good decision it has no inputs for.

```rust
/// What the device can actually do. Queried once, cached on the backend.
#[derive(Clone, Debug)]
pub struct DeviceCaps {
    pub class: DeviceClass,          // Cpu | IntegratedGpu | DiscreteGpu | Npu | Browser
    pub compute_units: u32,          // SMs / CUs / cores
    pub max_workgroup_size: u32,
    pub workgroup_mem_bytes: u32,
    pub subgroup_size: Option<u32>,  // None = not exposed (WebGPU baseline)
    pub unified_memory: bool,        // no host<->device copy cost
    pub peak_bandwidth_gbs: Option<f32>,
    pub numeric: NumericSupport,     // see S4
}
```

Values come from the backend: wgpu `AdapterInfo` + `Limits` + `Features`, the
CPU backend from `std::thread::available_parallelism` and detected ISA, the NPU
from OpenVINO's device query. Where a value is unknowable it is `Option::None`
and the selector must cope — an unknown capability is never assumed present.

**Practice.** One struct, plain data, `Clone`; no trait objects in the hot path.
Backends fill it in `new()`. It is also exactly what `brain perf` should record
in its `env` block, which today captures the adapter *string* but nothing
machine-readable.

### S2. Kernel selection is fixed at model-construction time

`KernelIds` is a struct of `usize` pipeline indices chosen once when a model is
built. There is no way to choose a different implementation for a different
shape, let alone a different device — so the decode path and the training path
run the *same* kernel despite being different regimes, which is finding (a).

```rust
/// One logical operation, independent of how it is implemented.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op { MatMul, RmsNorm, ArgMaxRow, Rope, GqaScores, /* … */ }

/// The shape that decides which variant wins.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpShape { pub m: u32, pub n: u32, pub k: u32, pub dtype: Dtype }

pub trait KernelSelector: Send + Sync {
    /// Pick a registered variant for this op/shape on this device.
    fn select(&self, op: Op, shape: OpShape, caps: &DeviceCaps) -> KernelId;
}
```

Selection happens once per distinct `(Op, OpShape)` and is memoised — decode
shapes are few and fixed, so this is a `HashMap` lookup amortised to nothing,
not a per-dispatch branch.

**Practice.** The selector is a trait so a test can install a deterministic one
(`AlwaysReference`) and pin behaviour; `Op`/`OpShape` are `Hash + Eq` so the
memo table is trivial; the default impl is a pure function of its arguments and
therefore unit-testable with no device at all.

**This is the seam the CPU backend already wants.** It currently collapses
`conv2d_tiled`/`conv_act_tiled`/`conv2d` to one native path by *name matching
inside the backend*, with a comment saying "the tiling only helps the GPU". That
knowledge belongs in a selector, not in each backend's `if` ladder.

### S3. Kernel variants are hand-written files with literal tile sizes

There are 281 `.wgsl` files and every tile size, workgroup size and unroll factor
is a literal. Tuning a tile per device means writing another file. That does not
scale to (ops × shapes × devices), and it is why there is one `matmul_reg` sized
for a discrete GPU and nothing for anything else.

**Mitigation — parameterise, don't multiply.** WGSL `override` declarations
(pipeline-overridable constants, supported by naga and wgpu) let one source
carry `override TILE_M: u32 = 8;` and be specialised at pipeline creation. Where
`override` is not enough (array sizes must be constant-expressions), a small
`kernels::template` step substitutes before compilation and caches by
`(source, params)`.

```wgsl
override TILE_M: u32 = 8u;
override TILE_N: u32 = 64u;
override SPLIT_K: u32 = 4u;
```

**Practice.** WGSL stays the single source of truth — this adds *parameters* to
a kernel, it does not fork it. The unparameterised default must reproduce
today's behaviour exactly, so the change is provably inert until a selector
chooses otherwise. The const-list regeneration (`make kernels-regen`) is
unchanged.

### S4. The fp32-only invariant is the ceiling — and it is doing a real job

> *fp32 only, core compute only — no atomics, no subgroups, no f16*

This is what makes the same kernels run on a 2016 GPU and in a browser, and it
is the reason the project can claim what it claims. It is also, stated plainly,
**the reason no modern device can reach its ceiling**: tensor cores need
f16/bf16, the P40's 4× path needs int8 DP4A, and fast reductions want subgroups.

Resolving this by relaxing the invariant would forfeit the portability. The
resolution is to make it a **floor rather than a ceiling**:

```rust
/// Numeric paths a device supports, in increasing capability.
pub struct NumericSupport {
    pub f32: bool,          // always true — the portable baseline and the oracle
    pub int8_dot: bool,     // DP4A / dot4I8Packed  (Pascal+, most modern GPUs)
    pub f16: bool,          // fast f16 arithmetic (NOT true on P40: 1/64 rate)
    pub coop_matrix: bool,  // tensor cores / matrix cores
}
```

Rules that keep this honest:

* **The fp32 portable path is never removed.** It is the WebGPU/old-GPU
  guarantee *and* the numerical reference every other tier is checked against.
* **A tier is opt-in per device, chosen by measured capability**, never by
  assumption. `f16: false` on a P40 is not a detail — using f16 there would be
  **64× slower**, which is exactly why "just use half precision" is the wrong
  portable answer.
* **Every tier is parity-gated against fp32** by extending `make parity`, which
  already exists to compare CPU/Vulkan/NPU. A tier that cannot demonstrate
  equivalence within a declared tolerance does not ship.
* **Quantisation is a numeric tier, not a separate feature.** `qwen::q8` and
  `matmul_i8` already exist; they belong behind this seam so every model gets
  them, rather than being wired per-model as today.

### S5. Nothing is measured on the actual device

Even with capabilities and variants, the right choice is empirical: a selector's
model of a device is always approximate. The established answer is to measure
once and remember.

```rust
/// Time each registered variant for a shape, keep the winner, persist by device.
pub struct AutoTuner { cache: TuneCache }
```

Keyed by `(adapter identity, driver version, kernel source hash, Op, OpShape)`,
persisted under a cache dir, invalidated when any key component changes. First
run pays a few milliseconds per distinct shape; every later run is free. A
missing or stale cache is *ignored*, never trusted.

**Practice.** The tuner is a `KernelSelector` implementation, so it composes with
S2 rather than complicating it. `BRAIN_NO_AUTOTUNE=1` forces the static
selector, which is what CI and reproducible benchmarking use — an autotuned
result must never make a benchmark unreproducible.

### S6. Weight layout is fixed row-major

Weights are read-only after load, so their layout is free to change once, at
load time, to whatever the chosen kernel streams fastest — swizzled for
coalescing, pre-packed for DP4A, blocked for a CPU's cache. Today every kernel
reads `[N,K]` row-major regardless of device.

**Mitigation.** A `WeightLayout` chosen alongside the kernel variant, applied
during `ParamStore` construction. The transform is pure, deterministic and
testable independently of any device.

### S7. Execution is eager and per-op; the graph seam is unused

Every op is a separate dispatch: 6182 dispatches for 36 decode steps. Fusion
(norm+matmul, QKV into one GEMM, SwiGLU's gate/up) cuts both dispatch count and
the round-trips to memory that dominate a bandwidth-bound regime.

`backend_api::GraphBackend` — `compile(onnx) -> run()` — **already exists and
nothing implements it.** That is the correct seam for whole-graph devices (the
Intel NPU via OpenVINO, and any future compiler-style target), and it is the
structural answer to "how do NPUs fit", which per-op dispatch cannot express.

Two independent pieces: (i) fuse the obvious eager sequences behind the S2
selector; (ii) implement `GraphBackend` for the NPU path so `crates/npu` stops
being a bespoke pipeline.

---

## Part II — The findings, through those seams

### A. Decode at ~1% of the card

**A0 (new, first). Int8 weights in the serving path.** The single largest and
most portable win: 4× fewer weight bytes in a bandwidth-bound regime, plus 4×
the MACs on any device with `int8_dot`. `matmul_i8` and `qwen::q8` exist; the
work is wiring them behind S4's numeric tier and adding a `--weights-int8` (or
capability-driven default) to the engine. On devices without DP4A the same
quantisation still halves-or-better the bytes moved, so it wins there too.

*Verification:* `perf sweep --workload decode_heavy` tok/s, and a `fidelity` gate
run — quantisation **will** change some argmaxes, so the acceptance criterion is
a declared token-agreement threshold, not exactness. This is precisely why the
gate must land first (item J1).

**A1. `rmsnorm_rows`** — one workgroup per row: 64 threads accumulate strided
partial sums into workgroup memory, **one** `workgroupBarrier()`, thread 0 folds
64 partials. 64× the parallelism and coalesced loads. Selected by row count via
S2, so large-M training keeps today's kernel.

**A2. `matmul` for skinny M** — one workgroup per output block, threads split K
with coalesced consecutive-k reads, `x` (tiny at decode) staged in workgroup
memory and reused across all M rows. Tile sizes come from S3 `override`s so the
same source serves a 3840-core discrete GPU, a 96-EU integrated GPU and a CPU.

**A3. `argmax_row`** — our own kernel, one thread per row, 10.3% of decode time.
Split into `argmax_part` → `argmax_final` following the existing
`max_abs_part`/`max_abs_final` idiom. No barrier at all, so every backend is
trivially fine.

**A4. One host round-trip per token** — `readbacks 36`, one per step, each a full
flush and fence; 36% of wall is outside kernels. With `argmax_row` already
producing the token on-device, feed it back as the next step's input without a
round-trip, reading back every *k* steps for stop-checking. Re-measure before
attempting double-buffering.

> **Barrier constraint.** `wgsl-cpu::split_at_barrier` supports **exactly one
> top-level `workgroupBarrier()`**, no nesting. A `log2(n)` tree reduction would
> not compile on the CPU backend. Every reduction above therefore uses one
> barrier plus a short serial fold, or the two-dispatch split — the repo's
> existing idiom.

### B. Head-of-line blocking: prefill starves decode

TTFA p99 grows 15× and IAL p99 10× from concurrency 1→32 while throughput is
flat, and the `interactive` SLO was met at *no* concurrency level. Cause:
`Scheduler::step_inner` runs a **full prefill for every fitting waiting request**
before a single decode step, so a burst of 32 arrivals performs 32 prompt
forwards back-to-back while every running sequence waits.

Mitigation: a `PrefillBudget { max_tokens_per_step, max_admissions_per_step }`
checked during admission, so decode runs every iteration and arrivals are
absorbed over several. Chunked prefill already exists, so long prompts can later
be split *across* iterations too. Recorded in the artifact's `target.config`.

*Verification:* TTFA p99 sub-linear in concurrency on `perf sweep --workload
interactive`; `perf mixed` shows the interactive class's `normalised_slowdown`
fall with `batch` throughput unchanged. Reference tests must stay green —
admission order changes *when* tokens appear, never *which*.

### C. No admission policy

`perf overload` reports 0 rejections at 4× capacity. Add an `Admission` trait
(`admit(&PendingRequest, &QueueState) -> Decision`) with `UnboundedQueue` as the
default so nothing changes unless a policy is installed; `MaxQueueDepth`,
`DeadlineAware`, `TokenBudget` as alternatives. `StepReport::rejected` already
carries rejections; its reason becomes a typed enum rather than a `String`.
Policies are pure functions of `QueueState` and unit-testable with no engine.

### D. No prefix cache

`kv_hit_rate` and `eviction_regret` are structurally null. Two steps:
(i) refcounted blocks in `BlockAllocator` — also enables copy-on-write branching;
(ii) chained per-block prefix hashing with a `HashMap<PrefixHash, PhysBlock>`, so
`prefill` computes only the unmatched tail. Invariant: **a cache hit must produce
byte-identical KV**, enforced by a warm-vs-cold identical-output test and a
random-prefix property test. Prefix reuse must show *no* decode-rate change — it
is a prefill optimisation.

### E. Residency: 64% eviction regret

`place::plan_eviction` is strict LRU, which has no notion of reload *cost* (4 GB
vs 200 MB) or *popularity*, so under a shifting Zipf it repeatedly evicts the
just-cooled head. Add an `EvictionPolicy` trait scoring
`recency × hit_rate × reload_cost`; keep `Lru` available so the benchmark
compares them. `plan_eviction` stays pure over `(cost, budgets, residents,
policy, now)` — pass `now` in rather than calling `Instant::now()`, so it remains
testable without a clock.

### F. No warm start

A second `Engine` costs the same as the first: each builds its own `Gpu` and
recompiles all pipelines. (i) `Engine::from_map_on(Arc<Gpu>, …)` so a serving
process shares one device — this also removes a full device init per `residency`
activation; (ii) `wgpu::PipelineCache` persisted, keyed by adapter + driver +
kernel-source hash, a stale blob ignored rather than trusted.

### G. Faults: 1 of 5 injectable

A `FaultSink` trait behind a `fault-injection` feature, compiled out of release
so there is no cost and no `unsafe`. Multi-rank faults need a real multi-process
harness (`model::netcollective` provides the transport); until it exists they
must keep reporting *skipped*, never passing.

### H. Placement has nothing to place

Single-model cross-device execution does not exist — `--device gpu,cpu` makes
both schedulable for *different* models. Near term, point `perf placement` at
multi-*model* placement, which is the thing brain actually does, and say so in
`notes`. Longer term, inference-side pipeline parallelism via the existing
`model::shard` `Shardable` seam is what makes per-layer placement meaningful.

### I. Frontend stages unmeasured

Wire the real `data::bpe` / `qwen_tokenizer` and media-decode paths in. The
current "host is not the bottleneck" conclusion is only safe for the stages
measured — and it will change once A lands and the device rate rises.

### J. No correctness gate, no regression gate

(1) Wire `fidelity` into the scenarios: reference run at batch 1 on the same
device, greedy streams compared, opt-out rather than opt-in. **This must precede
the kernel and quantisation work** — it is the safety net for exactly the failure
mode A0–A2 risk. (2) `brain perf gate` against committed baselines as *hard
floors* on best-of-N, following `scripts/wm-perf-gate.sh`; tight deltas flap on
shared boxes.

### K. `resources` is null

`backend-wgpu` already counts submits/dispatches/readbacks for `BRAIN_PROFILE`;
promote them from an stderr dump to a queryable `DeviceStats`, which fills most
of the block nearly free. True utilisation needs vendor counters and stays
`null`.

### L. `capability::Provider` adoption

Only `demo`, `imageops`, `zimage` implement it, so `CapabilityTarget` — the seam
that makes any model benchmarkable with no new benchmark code — covers almost
nothing. Implementing it for `qwen`, `yolo`, `depth`, `tts` is mechanical and
each immediately gains every Tier-1 scenario. Highest leverage per line here.

---

## Part III — What "peak" means per device class

The point of Part I is that these targets differ, and the engine must be able to
express all of them from one source.

| Class | Primary lever | Numeric tier | Trap to avoid |
|---|---|---|---|
| **Old discrete GPU** (P40, Pascal) | bandwidth: int8 weights | `int8_dot` (DP4A, 4×) | f16 is **1/64 rate** — using it is 64× slower |
| **Modern discrete GPU** (Turing+, RDNA, Blackwell) | tensor/matrix cores | `coop_matrix`, `f16`/bf16 | fp32-only forfeits nearly all of the card |
| **Integrated GPU** (Arc, Iris, Apple) | unified memory — no PCIe copies | `f16` usually, `int8_dot` often | tiles sized for 3840 cores thrash a 96-EU device |
| **CPU** (Cranelift JIT + AVX2/512/NEON) | cache blocking + SIMD width | `f32`, `int8` via VNNI | GPU tiling is actively harmful; the backend already collapses it |
| **NPU** (OpenVINO whole-graph) | whole-graph compile, int8 | `int8` | per-op dispatch cannot express it — needs `GraphBackend` |
| **WebGPU / browser** | the portability floor | `f32` only | anything assuming subgroups or f16 breaks the guarantee |

The one lever common to every row is **bytes moved**, which is why A0 leads.

---

## Ordering

Sequenced so each step is independently verifiable and the risky work is fenced.

| # | Item | Why here | Size |
|---|---|---|---|
| 1 | **J1** fidelity gate wired in | the safety net for everything numeric; must precede A0 | S |
| 2 | **A3** `argmax_row` split | our own regression, 10% of decode, no new seams needed | S |
| 3 | **A1** `rmsnorm_rows` | 16.6%, proves the single-barrier reduction pattern on both backends | S |
| 4 | **B** prefill budget | fixes the user-visible SLO failure, zero kernel risk | S–M |
| 5 | **S1** `DeviceCaps` | prerequisite for everything portable; pure data | S–M |
| 6 | **S2** `KernelSelector` | the seam A2/A0 and the CPU backend all need | M |
| 7 | **A0** int8 weights in serving | biggest portable win; needs J1 + S1 + S2 to be safe and device-aware | M |
| 8 | **S3** parameterised kernels | stops per-device tuning multiplying files | M |
| 9 | **A2** skinny-M matmul | the 67.5%, now expressible once for all devices | M–L |
| 10 | **L** `Provider` adoption | unblocks benchmarking every other model | M |
| 11 | **S5** autotuner | converts S1–S3 into measured, per-device choices | M |
| 12 | **F1** shared `Gpu`, **C** admission, **E** eviction policy | independent, contained | M each |
| 13 | **A4** on-device sampling, **S6** weight layout, **S7** fusion + `GraphBackend` | after the balance shifts; re-measure first | M–L |
| 14 | **D** prefix cache | large win for agent/RAG shapes, highest correctness risk | L |
| 15 | **F2**, **G**, **H**, **I**, **J2**, **K** | small and independent | S each |

Items 1–4 are worth doing as one block: cheap, independently verifiable, and
together they should move both the throughput and the latency numbers that
currently fail every SLO. Items 5–9 are the portability spine — after them, a
new device class is a `DeviceCaps` implementation and a tuning entry, not a
rewrite.
