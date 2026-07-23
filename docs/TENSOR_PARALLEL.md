# Tensor parallelism, collectives, the grid, and the planner

The third parallelism dimension and the machinery that composes all three. Tensor
parallelism splits **one tensor operation across devices** (`Y = XW` → each GPU
computes `X·Wᵢ`, combined by a collective) — distinct from the intra-GPU
kernel/workgroup tiling (brain's reg2 GEMM), which is how each *local* shard is
then executed efficiently. See [`SCALING.md`](SCALING.md) for the umbrella.

Grounded in Megatron-LM and PTD-P (`resources/dp/`).

## The collective layer (`model::collective`)

Everything moves data through `Collective` — a per-rank (NCCL/MPI-shaped)
interface:

| op | meaning |
|---|---|
| `all_reduce(rank, x)` | element-wise **sum** of every rank's `x`; all get the total |
| `all_gather(rank, x)` | concat every rank's `x` in rank order |
| `reduce_scatter(rank, x)` | sum, then each rank keeps its `1/world` slice |
| `broadcast(rank, x, root)` | root's `x` to all |

Reductions sum in a fixed rank order, so results are deterministic /
bit-reproducible. `HostCollective` implements it in-process via a barrier + shared
host staging (the transport for a single box, no NVLink). **A networked cluster is
a new `Collective` impl over sockets; call sites don't change.**

## Tensor-parallel transformer layer (Megatron `f`/`g`)

Per layer, TP degree `k`, with the residual stream **replicated** across the `k`
ranks (only the inside of attention/MLP is sharded):

**MLP** `Z = GeLU(X·Wfc)·Wproj`:
- `Wfc` **column-parallel** — rank *i* owns `ff/k` hidden columns, computes
  `Yᵢ = GeLU(X·Wfcᵢ)`. No communication; the intermediate stays sharded.
- `Wproj` **row-parallel** — rank *i* owns `ff/k` input rows, computes a partial
  `Zᵢ = Yᵢ·Wprojᵢ` (`[m,d]`). **One all-reduce** sums the partials → `Z`.

**Attention** (heads are independent):
- QKV **column-parallel by head** — rank *i* owns `n_heads/k` heads (its q/k/v
  rows), runs attention on them. No communication.
- Output projection **row-parallel** — partial `[m,d]` per rank + **one
  all-reduce**.

So **2 all-reduces per layer forward** (attn + MLP). In the **backward** the
conjugate `f` operator all-reduces the **input gradient** `dX` at each
column-parallel entry; the weight-shard gradients (`dWfc`, `dWproj`) are **local**
(no communication) and reassemble to the single-GPU gradient.

### Validated bit-exact (`crates/model/tests/tensor_parallel.rs`, 2×P40)

| what | vs single-GPU |
|---|--:|
| MLP forward | rel 9.95e-8 |
| attention forward (head-split) | rel 8.66e-8 |
| training: `dX` (all-reduced) | rel 2.03e-7 |
| training: `dWfc`, `dWproj` (local shards) | **rel 0.00e0** |

The small residuals are only floating-point summation order in the all-reduce.

**Gotcha found by TDD:** build the per-rank `Gpu`s **sequentially before
threading** — `BRAIN_GPU_INDEX` is process-global and wgpu device init is not
concurrency-safe; setting it inside the threads deadlocks (same discipline as
`Pipeline`/`DataParallel`).

## The 3D process grid (`model::grid`)

A world of `tp·pp·dp` ranks maps to `(tensor, pipeline, data)` coordinates,
**TP-fastest**:

```text
rank = (dp_rank · pp + pp_rank) · tp + tp_rank
```

so ranks `0..tp` are one tensor-parallel group. This is the Megatron PTD-P
placement: **TP needs the tightest coupling** (an all-reduce per layer → same
node/NVLink, lowest rank stride), **pipeline** crosses nodes (a residual per
boundary), **data** is outermost (one grad all-reduce per step).

- `Grid::{coord, rank, tp_group, pp_group, dp_group}` — the portable rank math.
- `LocalGroups` — the in-process realisation: one `HostCollective` per group in
  each dimension, so rank *r* reaches its peers via `lg.tp(r)` / `pp(r)` / `dp(r)`
  (returns the group's collective + the rank's local index). A cluster builds the
  same per-group communicators over sockets.

## The tensor-parallel planner (`model::plan`)

TP has a real per-layer communication cost, so more of it is not always better.
The cost model:

```text
T_total(t) = T_local(t) + T_comm(t) + T_sync(t)
```

Raising `t` cuts `T_local` (FLOPs/`t`) but adds `T_comm` (a ring all-reduce per
layer: `bytes/bandwidth + latency`) and, once the local dim drops below the reg2
tile (`gemm_min_dim = 128`), lowers GEMM efficiency.

`plan_tp` is **capacity-first** (Megatron practice): pick the **smallest** degree
that makes the model fit while keeping local GEMMs large — TP is for *fitting*, and
data/pipeline parallelism chase throughput. It reports when memory forces a degree
that shrinks GEMMs below the tile. `tp_step_secs` exposes the cost model directly
(unit-tested: free communication ⇒ more TP strictly faster; a slow high-latency
link makes TP a net loss).

## Extending: weaving TP into a full model

The mechanic and transport are proven; making a whole model tensor-parallel is a
per-model weave (like the `Shardable` pipeline weave):

1. Give the model a TP context (`rank`, `world`, its `Collective`).
2. Shard the four projection weights: QKV + FC **column-parallel**, output-proj +
   down-proj **row-parallel**. Keep embeddings, layernorms, and the residual
   stream **replicated**.
3. Insert **one all-reduce** after the attention output projection and one after
   the MLP down projection (forward); the conjugate `dX` all-reduce in the
   backward.
4. Gate it exactly like the single-device path so `world == 1` is byte-identical.

Then a unified executor assigns each rank a `Grid` coordinate and runs
TP-in-stage → PP-across-stages → DP-across-replicas over `LocalGroups`.
