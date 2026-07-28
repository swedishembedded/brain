# Mitigation plan for the benchmark findings

Every finding in [`status.md`](status.md), with a root cause established from
evidence, a designed mitigation, and how it will be verified. Ordered by
value-per-effort at the end.

Two rules apply to everything below:

1. **Nothing lands without its gate.** Kernel-math changes run `make gradcheck`
   and `make parity`; serving changes keep the token-for-token reference tests
   (`batched_serving_matches_reference`,
   `scheduler_dynamic_admission_matches_reference`) green. A performance change
   that alters output is a bug, not a trade-off.
2. **The benchmark that found it is the benchmark that closes it.** Each
   mitigation names the `brain perf` scenario and the number that must move, so
   "fixed" is a measurement rather than an opinion.

### Engine invariants the designs must respect

| Constraint | Consequence for these designs |
|---|---|
| WGSL is the single source of truth | no host-only fast path; a kernel change must run on **both** backends |
| fp32, no atomics, no subgroups, no f16 | reductions cannot use atomics — use the existing split-kernel idiom |
| Workgroup memory **and** `workgroupBarrier()` are allowed | proven by `matmul_reg`, `conv2d_tiled`, `flash_attn_bidir` |
| The CPU JIT supports **exactly one top-level barrier**, no nesting (`wgsl-cpu::split_at_barrier`) | a tree reduction with `log2(n)` barriers **will not compile on CPU**. Use one barrier + a short serial finish, or two dispatches (`max_abs_part` → `max_abs_final`) |
| ≤8 storage buffers per kernel, single bind group | budget bindings before adding parameters |

---

## A. Decode runs at ~1% of the card — the kernels are shaped for training

**Evidence.** `BRAIN_PROFILE=1`, `qwen-synth:8x512x8`, batch 8, Tesla P40:

```
GPU kernel time total 1402.5 ms  =  64.3% of wall
  matmul                947.0 ms  2047 calls  (67.5%)   0.46 ms/call
  rmsnorm               232.3 ms  1188 calls  (16.6%)   0.20 ms/call
  argmax_row            144.1 ms    31 calls  (10.3%)   4.65 ms/call
  everything else        79.1 ms                        ( 5.6%)
dispatches 6182, submits 36, readbacks 36
```

A `[8,512] × [512,512]` matmul is 4.2 MFLOP; at 0.46 ms that is **9.1 GFLOP/s
against the P40's ~11 TFLOP/s** — 0.08% of peak — and ~7 GB/s against 346 GB/s
of bandwidth. The GPU is *busy* (64% of wall), so this is not host stalling: the
kernels themselves are slow at these shapes.

**Root cause — one cause, three symptoms.** Every hot kernel parallelises over
**output rows or elements**. That is right for training (M = batch × seq is
thousands) and catastrophic for decode, where M is the number of concurrent
sequences (8–32):

* `rmsnorm` is *one thread per row* (`if (t >= p.seq_len) { return; }`, then a
  `d_model` loop) — at batch 8 that is **8 threads on 3840 cores**;
* `matmul` is one thread per output element with a scalar K-loop and
  uncoalesced weight reads — at M=8 it is a GEMV with 0.25 FLOP/byte;
* `argmax_row` (added with the device head) has the same flaw — one thread per
  row scanning 32k logits. **This one is ours and regressed nothing, but it is
  10% of decode time and must be fixed.**

`matmul_reg` exists but is a 128×128 register-tiled tile designed for the
*compute-bound* regime; at M=8 it wastes 15/16 of every tile. Decode is a
different regime and needs kernels designed for it, not a bigger tile.

### A1. `rmsnorm_rows` — one workgroup per row

Each workgroup owns one row: 64 threads each accumulate a strided partial sum of
squares into `var<workgroup> partial: array<f32, 64>`, **one** `workgroupBarrier()`,
then thread 0 sums the 64 partials and the workgroup writes the normalised row.
64× more parallelism per row and coalesced loads, within the single-barrier
budget the CPU JIT allows.

```wgsl
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let row = wg.x; let t = li.x;
    var acc = 0.0;
    for (var c = t; c < p.d_model; c = c + 64u) { let v = x[row * p.d_model + c]; acc = acc + v * v; }
    partial[t] = acc;
    workgroupBarrier();                       // exactly one, top level
    // thread 0 finishes 64 partials serially — cheaper than a second barrier
    ...
}
```

Selected by row count: keep the existing kernel for large M (where per-element
parallelism already fills the device) and dispatch the row kernel below a
threshold. The choice lives in `model::block::rmsnorm_fwd` behind a
`KernelIds`-style id, so no model code changes.

### A2. `matmul_gemv` — skinny-M, weight-streaming

For M ≤ ~16 the arithmetic intensity is fixed at ~0.25 FLOP/byte no matter the
tiling: the kernel is **memory-bound on streaming W**, so the only lever is
reading W at full bandwidth. One workgroup per output column block; the 64
threads split the K dimension, each accumulating `x[m,k]*W[n,k]` over a strided
slice with **coalesced consecutive-k reads**; one barrier; thread 0 (or a short
serial fold) combines. `x` is tiny (M×K) and is read into workgroup memory once
and reused across all M rows, which is what turns M separate GEMVs into one pass
over W.

Expected: from ~7 GB/s toward the card's achievable streaming bandwidth. Even
reaching 25% of 346 GB/s is a >10× improvement on the dominant kernel.

`matmul.wgsl` stays as the reference implementation and the gradcheck oracle;
the new kernel is selected by shape, exactly as `conv2d`'s fused/register-tiled
paths are today.

### A3. `argmax_row` — partial/final split

Follow the established reduction idiom (`max_abs_part` → `max_abs_final`):
`argmax_part` reduces each row in `P` chunks to `P` (value, index) pairs;
`argmax_final` folds `P` pairs per row. No barrier at all, so both backends are
trivially fine, and the tie-break rule (lowest index wins) is preserved by
comparing index on equal values.

### A4. One host round-trip per token

`readbacks 36` = one per decode step: the argmax indices. Each is a full flush
plus a fence wait, and 36% of wall is not in kernels. Two mitigations, in order:

1. **Sample on-device** — with `argmax_row` already producing the next token,
   feed it back as the next step's input *without* a host round-trip. This needs
   the token buffer to be device-resident and `embed` to read it, removing the
   per-step sync entirely for greedy decode. Sequences still need host visibility
   for stop-checks, so read back every *k* steps (`k` configurable, default 1 →
   opt-in) or read asynchronously one step behind.
2. **Double-buffer the step** — record step *n+1* while *n*'s readback is in
   flight. Bigger change; only worth it if (1) leaves a measurable gap.

Do (1) first and re-measure before committing to (2).

**Verification.** `make gradcheck` + `make parity`; a new
`crates/kernels/tests/` case asserting each new kernel matches its reference
kernel elementwise on random shapes including M=1 and M not a multiple of 64;
then `perf sweep --workload decode_heavy` — the number that must move is
**output tok/s at concurrency 32 (now 290)** and **`matmul` share of
`BRAIN_PROFILE`**.

---

## B. Head-of-line blocking: prefill starves decode

**Evidence.** TTFA p99 grows 230 ms → 3413 ms (15×) and IAL p99 37 ms → 382 ms
(10×) from concurrency 1 → 32, while throughput stays flat.

**Root cause.** `Scheduler::step_inner` admits in a `while` loop that runs a
**full prefill for every fitting waiting request** before a single decode step:

```rust
while self.running.len() < self.max_running {
    ...
    let hidden = self.eng.prefill(&mut table, &req.prompt);   // whole prompt
    ...
}
// only then: one batched decode over the running set
```

So a burst of 32 arrivals performs 32 prompt forwards back-to-back; every
already-running sequence waits for all of them. This is textbook head-of-line
blocking, and it is why the `interactive` SLO was met at no concurrency level.

**Mitigation — a prefill budget per iteration.**

```rust
/// Admission policy for one scheduler iteration.
pub struct PrefillBudget {
    /// Max prompt tokens prefilled before yielding to decode.
    pub max_tokens_per_step: u32,
    /// Max requests admitted per iteration, whatever their size.
    pub max_admissions_per_step: usize,
}
```

Admission stops once the budget is spent, so decode runs every iteration and the
remaining arrivals are admitted over the following ones. Chunked prefill already
exists (`max_prefill`), so a long prompt can also be *split across* iterations
rather than blocking one — the natural follow-up once the budget lands.

Default: `max_tokens_per_step ≈ 2 × max_prefill`, tunable, and recorded in the
perf artifact's `target.config` so a run states the policy it used.

**Verification.** `perf sweep --workload interactive` must show TTFA p99 growing
sub-linearly with concurrency, and `perf mixed` must show the `interactive`
class's `normalised_slowdown` fall while `batch` throughput is unchanged. The
reference tests must still pass token-for-token — admission order changes *when*
tokens are produced, never *which*.

---

## C. No admission policy — the server cannot shed load

**Evidence.** `perf overload` reports 0 rejections at every offered load up to
4× capacity, because the scheduler queues without bound.

**Mitigation — an `Admission` trait, defaulting to today's behaviour.**

```rust
pub trait Admission: Send + Sync {
    /// Called once per waiting request per iteration.
    fn admit(&mut self, req: &PendingRequest, state: &QueueState) -> Decision;
}

pub enum Decision { Admit, Defer, Reject { reason: RejectReason } }
```

`QueueState` carries queue depth, oldest wait, free blocks and observed mean
service time — enough for `MaxQueueDepth`, `DeadlineAware` and `TokenBudget`
without any of them reaching into the scheduler. `UnboundedQueue` is the default
impl, so existing behaviour is unchanged unless a policy is installed.

Rejections already flow through `StepReport::rejected` (added for the capacity
fix), so the plumbing exists. `RejectReason` becomes an enum rather than the
current `String` — a typed reason is matchable by callers and by the benchmark,
and `thiserror` gives it a `Display` for the transport.

**Verification.** `perf overload` gains a real ladder per policy: `rejections`
non-zero past 1×, `waste_fraction` falling, and `rejection_accuracy` reportable.
Unit tests drive each policy against a synthetic `QueueState` with no engine at
all — the policies are pure functions of state, which is the point of the seam.

---

## D. No prefix cache — `kvcache` can only measure stalls

**Evidence.** `perf kvcache` reports 120 KV stalls / 3.4 s at 3× overcommit, and
`kv_hit_rate`/`eviction_regret` are *structurally* null: every request computes
its own KV and releases it on completion.

**Mitigation — content-addressed block sharing, in two steps.**

1. **Refcounted blocks.** `BlockAllocator` gains a refcount per physical block;
   `BlockTable::release` decrements rather than frees. This alone enables
   copy-on-write forking (an agent branching a conversation), and is a small,
   well-contained change to a module that is already unit-tested.
2. **Prefix hashing.** Hash each *full* block's token ids (rolling hash over the
   block, chained to the previous block's hash so a hash identifies a whole
   prefix). A `HashMap<PrefixHash, PhysBlock>` maps a matched prefix to existing
   blocks; `prefill` skips the matched prefix and computes only the tail.

Correctness is the risk, so the invariant is explicit: **a cache hit must
produce byte-identical KV to computing it**. Enforced by a test that runs the
same prompt twice with the cache warm and cold and asserts identical generated
tokens, plus a random-prefix property test.

**Verification.** `perf kvcache` cold vs warm: `kv_hit_rate` becomes non-null,
`recomputed_artifacts` falls, TTFA on the `shared_prefix` workload drops. Prefix
reuse must show **no** decode-rate change — it is a prefill optimisation, and a
benchmark claiming otherwise is measuring wrong.

---

## E. Residency: 64% eviction regret

**Evidence.** 24 models at 4× overcommit under a Zipf load whose popularity
shifts mid-run: two-thirds of evictions are of models wanted again almost
immediately; hit rate 48%, Jain fairness 0.494.

**Root cause.** `place::plan_eviction` is strict LRU: it walks `lru_on(device)`
and evicts oldest-first until the deficit is covered. LRU has no notion of
*cost* (a 4 GB model costs 4 s to reload; a 200 MB one costs 0.2 s) and no
notion of *popularity* (a model at the head of a Zipf distribution will be
wanted again within seconds). Under a shifting distribution it evicts the
just-cooled head repeatedly.

**Mitigation — cost-aware scoring (GDSF-style), behind a policy seam.**

```rust
pub trait EvictionPolicy: Send + Sync {
    /// Lower score = evict first.
    fn score(&self, e: &ResidentEntry, now: Instant) -> f64;
}
```

Default becomes `CostAware`:

```text
score = recency_weight(last_use)  ×  hit_rate(key)  ×  reload_cost_bytes
```

so a small, cold, cheap-to-reload model is evicted before a large, hot,
expensive one. `Lru` stays available and is what the benchmark compares against
— the *point* is to have both and show the difference.

Two supporting changes:
* track a per-key hit counter and last-use in `Residents` (it already stores
  `last_use`; add `uses` and `bytes` is already in `MemCost`);
* keep `plan_eviction` pure over `(cost, budgets, residents, policy)` so it stays
  unit-testable without threads or a clock — the existing design is right and
  should not be given a `Instant::now()` dependency; pass `now` in.

**Verification.** `perf residency` with `--policy lru` vs `--policy cost-aware`
on the identical seed: regret must fall materially and fairness rise, with
aggregate goodput no worse. A regression test pins a hand-built scenario where
LRU provably evicts wrong and the new policy does not.

---

## F. No warm start — every engine recompiles every pipeline

**Evidence.** A second `Engine` in the same process costs the same as the first
(~3.1 s vs ~3.5 s): device init ~0.3 s, weights ~1.8 s, first prefill ~1.0 s.

**Root cause.** `Engine::from_map` calls `Gpu::new(PIPELINES)`, and every `Gpu`
creates its own instance, adapter, device and compiles all 18 pipelines.

**Mitigation — two independent changes.**

1. **Share the device.** `Engine::from_map_on(gpu: Arc<Gpu>, …)` alongside the
   owning constructor. A serving process builds one `Gpu` and hands it to every
   engine; the existing constructor keeps working by creating one and wrapping
   it. This is the change that matters for multi-model serving, and it composes
   with `residency`, which today pays a full device init per activation.
2. **Persist compiled pipelines.** `wgpu::PipelineCache` (available in wgpu 29;
   `PipelineCacheDescriptor` + `get_data()`) seeded from a file under a cache
   dir, keyed by adapter name + driver version + a hash of the kernel sources.
   A stale or foreign cache must be *ignored*, never trusted — wgpu validates
   the blob, and the key makes a mismatch impossible to act on.

**Verification.** `perf startup` gains a genuine warm row: `device_init_ms` in
the warm block must fall to near zero with a shared device, and cold
`device_init_ms` must fall with a populated pipeline cache. The scenario already
prints both rows; today they match, and that is the before-picture.

---

## G. Faults: only 1 of 5 injectable

**Evidence.** `perf faults` injects device OOM; worker death, hung ranks,
collective timeouts and corrupt KV transfers are reported *skipped*.

**Mitigation — a fault-injection seam, compiled out of release by default.**

```rust
#[cfg(feature = "fault-injection")]
pub trait FaultSink: Send + Sync {
    fn before(&self, point: FaultPoint) -> Result<(), InjectedFault>;
}
```

`FaultPoint` names the places worth breaking (`KernelDispatch`, `BufferAlloc`,
`WeightRead`, `CollectiveOp`, `KvTransfer`). Call sites are a single `?`-style
check; with the feature off the trait and its calls vanish, so there is **no
release-build cost** and no `unsafe`.

Multi-rank faults need a harness that spawns real ranks; that is a separate
piece of work (`crates/model::netcollective` already gives the transport). Until
it exists the scenario must keep reporting them as *skipped* — never as passing.

**Verification.** `perf faults` `injected` rises from 1 to 5 single-process
points, `all_acceptable` stays true, and a deliberate regression (a swallowed
error) must flip it false.

---

## H. Placement: nothing to place

**Evidence.** `perf placement` reports `placement_efficiency: null` — with only
single-device artifacts there is no combined run to compare.

**Root cause.** Deeper than the benchmark: brain has no cross-device execution
for a *single* model instance. `--device gpu,cpu` makes both schedulable for
*different models* (via residency), which is real, but one model runs on one
device.

**Mitigation — measure the capability that exists, and name the one that does not.**

1. **Now:** `perf placement` compares multi-*model* placement — run `residency`
   with `--device gpu0`, `gpu0,gpu1`, and `gpu,cpu`, and the efficiency number
   becomes meaningful for the thing brain actually does. This needs no engine
   change and is the honest version of the metric today.
2. **Later:** single-model cross-device execution is pipeline parallelism, which
   `crates/model::shard` (`Pipeline<M>` over `Shardable`) already implements for
   training. Exposing it for inference is the real work; only then does
   per-*layer* placement efficiency mean anything.

Until (2), the scenario's `notes` must say the efficiency is over model
placement, not layer placement.

---

## I. Frontend: stages unmeasured

**Evidence.** Only JSONL encode/decode and templating are timed; tokenise,
detokenise, image decode and audio resample are absent (correctly reported as
absent rather than as free). Host cost is negligible at 290 tok/s — but that
conclusion is only safe for the stages measured.

**Mitigation.** Take an optional `--tokenizer` and media inputs, and time the
real `data::bpe` / `data::qwen_tokenizer` paths and the `zimage`/`audio` decode
paths. Straightforward; the value is that the "host is not the bottleneck"
conclusion becomes complete, and it will *change* once A lands and the device
rate rises 10×.

---

## J. No correctness gate, no regression gate

**Evidence.** Every artifact reports `correctness.passed: null`; the renderer
prints "result is unverified". There are no committed baselines.

**Mitigation.**

1. **Wire `fidelity` into the scenarios.** Before the measured run, generate a
   reference at batch 1 on the same device; after, compare greedy streams. This
   is the check that would catch a "fast" kernel that changes the argmax — the
   exact failure mode the A-series work risks. Cost is one short run, so it is
   opt-out (`--no-gate`) rather than opt-in.
2. **`brain perf gate`** against `scripts/perf-baselines.json`: hard floors on
   best-of-N, not tight deltas, following `scripts/wm-perf-gate.sh` — a shared
   box throttles and tight deltas flap. Committed baselines are refreshed
   deliberately, never automatically.

---

## K. `resources` is entirely null

No device-utilisation counters exist. `gpu-core` should expose a small
`DeviceStats { submits, dispatches, bytes_uploaded, bytes_read }` — the
`BRAIN_PROFILE` counters already exist in `backend-wgpu`; promoting them to a
queryable struct (rather than an stderr dump at exit) fills most of the block
and is nearly free. Real utilisation needs vendor counters and can stay `null`.

---

## L. `capability::Provider` adoption

Only `demo`, `imageops` and `zimage` implement it, so `CapabilityTarget` — the
seam that makes *any* model benchmarkable with no new benchmark code — covers
almost nothing. Implementing `Provider` for `qwen`, `yolo`, `depth` and `tts` is
mechanical, and each one immediately gains every Tier-1 scenario. This is the
highest leverage-per-line item in the plan and should not wait for the engine
work.

---

## Ordering

| # | Item | Why here | Rough size |
|---|---|---|---|
| 1 | **A3** `argmax_row` split | our own regression, 10% of decode, self-contained | S |
| 2 | **A1** `rmsnorm_rows` | 16.6% of decode, single-barrier pattern proves out the approach | S |
| 3 | **B** prefill budget | fixes the *user-visible* failure (SLO met at no concurrency) with no kernel risk | S–M |
| 4 | **J1** fidelity gate wired in | must precede A2 — it is the safety net for kernel work | S |
| 5 | **A2** `matmul_gemv` | the 67.5%, and the biggest single win | M–L |
| 6 | **L** `Provider` adoption | unblocks benchmarking every other model | M |
| 7 | **F1** shared `Gpu` | unblocks multi-model serving and cuts residency activation cost | M |
| 8 | **C** admission seam | makes `overload` able to compare policies | M |
| 9 | **E** cost-aware eviction | 64% regret, contained change behind a seam | M |
| 10 | **A4** on-device sampling | the remaining 36% host share, after A1–A3 shift the balance | M |
| 11 | **D** prefix cache | large win for agent/RAG shapes, but the highest correctness risk | L |
| 12 | **F2**, **G**, **H1**, **I**, **J2**, **K** | smaller, independent | S each |

A1–A3 and B are worth doing first as a block: they are the cheapest, they are
independently verifiable, and together they should move both the throughput and
the latency numbers that currently fail every SLO.
