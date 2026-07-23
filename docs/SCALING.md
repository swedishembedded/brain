# Scaling brain across GPUs (and, later, machines)

How to train and run models on more than one device — the three parallelism
dimensions brain implements, when to use each, what to expect, and how they
compose toward multi-node clusters.

This is the umbrella. Depth lives in:
- [`DATAPARALLEL.md`](DATAPARALLEL.md) — **Data** parallelism (replicate + all-reduce)
- [`SHARDING.md`](SHARDING.md) — **Pipeline** parallelism (split layers) + micro-batching
- [`TENSOR_PARALLEL.md`](TENSOR_PARALLEL.md) — **Tensor** parallelism (split one op) + the
  transport (collectives), the process **grid**, and the **planner**

Design notes and the reference papers (Megatron, GPipe, PipeDream/1F1B, ZeRO,
PTD-P) are under `resources/dp/`.

## The three dimensions at a glance

| dimension | splits | per-step communication | primary purpose | brain API |
|---|---|---|---|---|
| **Data (D)** | the batch | one gradient all-reduce | throughput (model fits one card) | `model::DataParallel<M>` |
| **Pipeline (P)** | the layers | one residual per stage boundary | capacity (weights > one card) | `model::Pipeline<M>` |
| **Tensor (T)** | one matmul | two all-reduces per layer | capacity for a layer too wide for one card | `Collective` + per-model weave |

They are **orthogonal** and compose: `DP( PP( TP(model) ) )`. On this 2-GPU box
only one dimension can exceed 1, so composition is validated on the degenerate
configs; the code is written for a `(t, p, d)` grid of any size (§Grid).

### Which to reach for

1. **Model fits on one card, want it faster** → **Data parallel**. Simplest, and
   the only one that raises throughput here. Speedup grows with grad-accumulation
   (the gradient sync is a fixed per-step cost).
2. **Model's weights don't fit on one card** → **Pipeline** (split layers across
   cards; cheapest communication — one residual per cut). Add **micro-batching**
   to overlap the stages.
3. **A single layer is too wide for one card** (huge `d_model`/`d_ff`) → **Tensor
   parallel**, but keep the degree **small** — it all-reduces *per layer*. Use the
   least TP that makes the layer fit while keeping local GEMMs large (§Planner).
4. **Very large model** → combine: TP within a node (fast link), PP across nodes,
   DP to scale the batch.

## Transport-agnostic by design (cluster-ready)

Every dimension moves data through one seam — `model::Collective`
(`all_reduce` / `all_gather` / `reduce_scatter` / `broadcast`), a per-rank
(NCCL/MPI-shaped) interface. Today the only implementation is `HostCollective`,
which stages through host RAM (the right transport for a single box with no
NVLink). **A future compute-node/cluster transport is a new `Collective` impl over
sockets — not a rewrite of the sharding code.** The `Grid` (§Grid) maps ranks to
`(tensor, pipeline, data)` coordinates and hands each dimension its collective
group, so the layout and the transport are independent.

## Hardware reality on this box (2× Tesla P40, **no NVLink**)

The interconnect is PCIe via host RAM. That shapes what helps:

- **DP**: the 2.4 GB (0.6B model) gradient all-reduce is the fixed cost; the fused
  host optimiser reads grads once and updates once. **1.34–1.58×** measured
  (rising with grad-accumulation).
- **PP**: cross-card traffic is one small residual per cut — cheap. Sharding is
  bit-exact and distributes weights evenly (auto-placed). Micro-batching overlaps
  stages for **1.26×** on top.
- **TP**: an all-reduce *per layer*. Activations are small (~2 MB) but there are
  `2·n_layers` of them, so on PCIe TP is **latency-bound** — implemented for
  **capacity and correctness**, not speed. Real TP speedups need NVLink/NVSwitch.

Rule of thumb here: **DP for speed, PP for capacity, TP only when a layer itself
won't fit.**

## How to use it

### Data parallel (any model)
```rust
use model::{Batch, DataParallel};
let mut dp = DataParallel::<Qwen>::new(cfg, batch, seqlen, &init, &[0, 1]);
dp.zero_grads();
dp.forward_backward(&microbatches);                 // concurrent across cards
dp.adamw_step(step, lr, wd, Some(1.0), 1.0 / microbatches.len() as f32);
dp.save("out.weights");
```

### Pipeline parallel (any `Shardable` model — gpt, qwen)
```rust
use model::{Batch, Pipeline};
let mut pipe = Pipeline::<Qwen>::new(cfg, batch, seqlen, &init, &[0, 1]); // cuts auto-placed
// one batch:
let loss = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
pipe.backward();
pipe.adamw_step(step, lr, wd, Some(1.0), 1.0 / grad_accum as f32);
// or micro-batched (overlapped stages, bounded activation memory):
let loss = pipe.train_step(&microbatches, step, lr, wd, Some(1.0));
```

### Planning the tensor-parallel degree
```rust
use model::{plan_tp, Hardware, ModelShape};
let plan = plan_tp(&hw, &shape);   // smallest TP that fits, keeping GEMMs large
println!("{} -> {}", plan.degree, plan.note);
```

## Expectations (what "correct" and "fast" mean here)

**Correctness is bit-exact.** Every dimension is validated against the
single-device result: DP grads `rel ≤ 1.3e-7`, PP loss+grads `rel 0.00e0`, TP
fwd/bwd `rel ≤ 2e-7` (differences are only floating-point summation order in the
all-reduce). This is the gate — a parallel run must reproduce the serial run.

**Speed** is bounded by the interconnect, not the compute:
- `fwd+bwd` genuinely overlaps across cards (DP ~3×, PP micro-batch ~2×).
- End-to-end is capped by the fixed per-step sync (no NVLink), so speedups are
  modest on 2 cards and improve with more grad-accumulation and more pipeline
  stages.

## Composition & the grid

Ranks form a `(tp, pp, dp)` **grid** (`model::Grid`), laid out TP-fastest so
tensor-parallel peers are adjacent (they need the tightest coupling). Each
dimension's peer set is a `Collective` group (`model::LocalGroups`). This is the
Megatron PTD-P placement and is the same whether ranks are threads on one box or
processes across a cluster. See [`TENSOR_PARALLEL.md`](TENSOR_PARALLEL.md).

## Status & roadmap

**Done + bit-exact-validated:** the transport (collectives), the layout (grid +
groups), the TP planner, and all three dimension mechanics — DP (generic, all 9
models), PP (generic `Shardable`, auto-placed, micro-batched), TP (MLP + attention,
forward *and* backward).

**Remaining (integration, not new primitives):**
- **Per-model TP weave** — wire the proven col/row-parallel linears + all-reduces
  into a full model's `forward`/`backward` (per-model, like the `Shardable` weave).
- **Unified 3D executor** — assign each rank its grid coord and run
  TP-in-stage → PP-across-stages → DP-across-replicas over `LocalGroups`.
- **More `Shardable` models** — glm (MLA+MoE+MTP), moe; seq2seq is encoder-decoder
  (needs a two-boundary shard model); non-transformers use DP only.
- **Network `Collective`** — the socket transport for multi-machine clusters.
