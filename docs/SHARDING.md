# Multi-GPU model sharding (pipeline parallelism)

Split a model's **layers** across several GPUs so one whose weights exceed a
single card fits across the pool. Generic over any [`Shardable`] model:
[`model::Pipeline<M>`](../crates/model/src/shard.rs).

## Why pipeline (not tensor) parallel

In brain's transformers the only tensor crossing a layer boundary is the residual
stream — one `[b·t·d_model]` slab. Everything else (Q/K/V, attention scores, MLP
activations) is layer-local. So partitioning by **contiguous layer ranges** means

- each GPU allocates weights + activations for **only its layers** (plus the
  embedding on stage 0 and the final-norm+lm_head+cross-entropy on the last stage);
- the cross-GPU traffic is a single residual per cut per pass (a few MB),
  host-staged — **no NVLink required** (the P40 box has none).

Crucially the residual has the **same width at every layer**, so the transfer
cost is identical wherever you cut. Placement is therefore purely a
*load-balancing* problem, not a transfer-minimising one.

## Generic: the `Shardable` seam

Splitting a model's execution graph is inherently per-architecture, but it's a
*small* seam. A model implements [`model::Shardable`]:

- `shard_cost(cfg, b, t)` — a per-layer / embed / head cost model (parameter
  counts) used to place the cuts;
- `new_shard(cfg, b, t, init, shard)` — build one stage: with `Shard::whole` this
  is the single-device path **byte-for-byte**; a partial shard allocates weights
  (via the model's `shard_param_list`) and activations only for its layers, and
  the `n·vocab` logits buffers only on the head stage;
- `run_forward_stage` / `run_backward_stage` and four residual-transfer methods
  (`read_out_res` / `write_in_res` / `read_in_dres` / `write_out_dres`);
- `replicated_params()` — names of any tied weight replicated across stages
  (Qwen's `tok.weight`); empty for untied models (GPT).

Everything else — orchestration, the optimiser, and cut placement — is generic in
`model::shard`. Implemented for **Qwen** (tied `tok.weight`) and **GPT** (untied
`lm_head`); the same ~50-line impl fits glm / moe / seq2seq.

## Automatic placement (`plan_balanced`)

`Pipeline::new` calls `plan_balanced`, which partitions the layers into
`gpus.len()` contiguous stages so the **maximum per-stage cost is minimised**
(exact DP). Because the embed and head stages carry the big embedding / lm_head,
they are given **fewer layers** — balancing memory across cards, which maximises
the model size that fits and evens out the pipeline. (Unit-tested in
`model/src/shard.rs`: a 12-layer model with a heavy embedding over 3 GPUs puts
more layers on the middle stage; even split when there is no endpoint weight.)

## Optimiser

The fused host optimiser gathers each parameter's gradient from its owning
stage(s) — **summed** for a replicated tied weight — runs one AdamW update with a
**global** grad-norm clip (rayon, so the tied replicas stay bit-identical), and
writes the new weights back. Stages keep only weight+grad on-GPU
(`BRAIN_OFFLOAD_ADAM`), moments in host RAM — sharding a large model's weights
across GPUs *and* its optimiser state into RAM.

## Correctness (2 P40s)

`crates/{qwen,gpt}/tests/shard_parity.rs` — auto-placed 2-stage pipeline vs the
single-device model:

| model | tied? | forward loss rel | worst grad rel |
|---|---|--:|--:|
| qwen | yes (`tok.weight`) | 0.00e0 | 0.00e0 |
| gpt  | no                 | 0.00e0 | 0.00e0 |

Both are **bit-exact**, and a sharded overfit run reduces the loss.

`crates/qwen/tests/integration_qwen3.rs::qwen3_shard_real_2gpu` — the real 0.6B:

```
layers split: 2 stages over 28 layers
loss  single-GPU=12.172951  2-GPU-sharded=12.172951  rel=0.00e0
per-card memory: 0, 3254 MiB | 1, 3251 MiB
```

Weights distribute evenly across both cards (auto-placement balanced the tied
endpoints), bit-exact against the single-device forward.

## Running

```rust
use model::{Batch, Pipeline};
let mut pipe = Pipeline::<Qwen>::new(cfg, batch, seqlen, &init, &[0, 1]); // auto-placed
pipe.zero_grads();
let loss = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
pipe.backward();
pipe.adamw_step(step, lr, wd, Some(1.0), 1.0 / grad_accum as f32);
```

`Pipeline::with_shards` bypasses auto-placement with explicit `Shard`s.

## Micro-batching (concurrent stages)

`Pipeline::pipelined_fwd_bwd(microbatches)` runs a **GPipe** schedule: stages run
**concurrently** (one thread per GPU, connected by channels), so while stage *i*
processes microbatch *k*, stage *i+1* processes *k−1* — overlapping the cards
instead of the plain one-batch path's 1/p duty cycle. Activations are
**re-materialised** in the backward (each stage keeps only per-microbatch *input
residuals*, `[b·t·d]`, and recomputes its forward before its backward), so
activation memory is `O(p · b·t·d)` regardless of the microbatch count — this is
what lets brain's single-activation-buffer stages pipeline at all. Grounded in
GPipe / PipeDream-Flush (`resources/dp/`).

Bit-exact to sequential grad-accumulation (`crates/qwen/tests/shard_microbatch.rs`:
worst grad rel 1.11e-7). Real 0.6B, 4 microbatches across 2 P40s:

```
naive sequential pipeline:  7073 ms/step
concurrent GPipe+recompute: 5604 ms/step   speedup 1.26x
```

The overlap beats the extra recompute forward even on a 2-stage pipeline; the win
grows with pipeline depth (bubble ≈ (p−1)/m). `train_step` wraps it with the fused
optimiser (grads averaged by 1/m).

## Composition & limits

- Composes with [data-parallel](DATAPARALLEL.md): shard a large model across a
  group, replicate the group data-parallel (2D parallelism). See
  `resources/dp/README.md` for the full 3D (TP×PP×DP) design.
- `plan_balanced` balances memory (parameter counts); a FLOP-weighted cost would
  balance latency instead — same DP, different `ShardCost`.
- Full 1F1B interleaving (multiple chunks/device) would shrink the bubble further
  at the cost of more communication — a scheduling refinement on this base.
