# Multi-GPU model sharding (pipeline parallelism)

Split a model's **layers** across several GPUs so one whose weights exceed a
single card fits across the pool. Implemented for Qwen3 in
[`crates/qwen/src/shard.rs`](../crates/qwen/src/shard.rs) as `Pipeline`.

## Why pipeline (not tensor) parallel

In brain's decoder the only tensor crossing a layer boundary is the residual
stream `res` — one `[b·t·d_model]` slab. Everything else (Q/K/V, attention
scores, MLP activations) is layer-local. So partitioning by **contiguous layer
ranges** means:

- each GPU allocates weights + activations for **only its layers** (plus the
  embedding on stage 0 and final-norm+lm_head+cross-entropy on the last stage);
- the cross-GPU traffic is a single residual per cut per pass (a few MB),
  host-staged — **no NVLink required** (the P40 box has none).

Tensor parallelism (splitting each matmul) would need an all-reduce *per layer*
over the interconnect — a bad trade on PCIe-only hardware.

## How a stage is built

A `Shard { start, end, embed, head, gpu_index }` describes one stage. The
existing `Qwen` model is parameterised by it (`Qwen::new_shard`): with the
default `Shard::whole`, the single-device path is **byte-for-byte unchanged**;
with a partial shard it

- allocates weights only for `blocks.start..end` (via `shard_param_list`),
  keeping non-owned residual/layer buffers as size-1 dummies so the model's
  absolute `res[l]` indexing is preserved;
- runs the token embedding only if `embed`, the head epilogue only if `head`
  (the `logits`/`d_logits` buffers — `n·vocab`, ~311 MB at vocab 152k — exist
  only on the head stage, where most of the saving is).

`Pipeline::new(cfg, b, t, init, train, &[gpu0, gpu1, …])` builds one stage per
GPU index (repeats allowed — `&[0,0]` puts two stages on one card, useful for
testing). The full weights live once in host RAM; each stage uploads only its
slice to its GPU.

## Forward / backward / optimiser

- **Forward**: stage 0 embeds → runs its layers → `res[end]` is read to host and
  written into stage 1's `res[start]` → … → the last stage runs the head and
  returns the loss.
- **Backward**: the last stage computes the CE gradient → its `dres[start]` is
  carried back to the previous stage's `dres[end]` → … → stage 0 runs `emb_bwd`.
- **Tied embedding** (`tok.weight` used by both the embedding on stage 0 and the
  lm_head on the last stage) is **replicated** on those two stages. Each
  contributes half the tied gradient; the optimiser **sums** the two replicas'
  grads, computes **one global grad-norm** (the tied weight counted once) and
  applies **one clip coefficient** on every stage — so the replicas stay
  bit-identical, exactly the single-device tied-weight math.
- **Optimiser**: sharded training uses the host **offload** AdamW (moments in
  RAM); `OffloadAdam::step_with_scale` applies the globally-reduced scale. This
  pairs naturally with sharding — a large model splits its *weights* across GPUs
  and its *optimiser state* into RAM.

## Correctness

`tests/shard_parity.rs` (tiny model, 2 stages on GPUs 0+1):

- forward loss **rel 0.00e0** vs single-device;
- **every** per-parameter gradient **rel 0.00e0** (incl. the tied `tok.weight`
  sum);
- a sharded overfit run reduces the loss (3.16 → 1.40 in 50 steps).

`tests/integration_qwen3.rs::qwen3_shard_real_2gpu` (the real 0.6B):

```
=== Qwen3-0.6B sharded across 2 P40s (pipeline-parallel) ===
  layers split: 2 stages over 28 layers
  loss  single-GPU=12.172951  2-GPU-sharded=12.172951  rel=0.00e0
  per-card memory: 0, 1636 MiB | 1, 1887 MiB
```

The weights genuinely distribute across both cards (≈ half each; the head stage
holds slightly more for the lm_head + tied replica), bit-exact against the
single-device forward.

## Running

```rust
use qwen::{Pipeline, QwenConfig};
let pipe = Pipeline::new(cfg, batch, seqlen, &init, /*train=*/true, &[0, 1]);
let loss = pipe.forward(&x, &y);
pipe.backward();
pipe.adamw_step(step, lr, wd, Some(1.0), grad_accum_inv);
pipe.save("out.weights");
```

Two stages on both P40s while one card is busy: `SHARD_TEST_GPUS=1,1` pins both
stages to GPU 1 (still exercises the full cross-stage transfer path).

## Limitations / next steps

- **Bubble**: stages run sequentially (stage 1 waits for stage 0). For a single
  batch this is correct but leaves each GPU idle part of the time.
  Micro-batch pipelining (GPipe/1F1B) would overlap stages for throughput — a
  scheduling optimisation on top of this correct base, not a correctness change.
- **Balance**: layers are split evenly by count. The head stage does more work
  (lm_head over the full vocab); an imbalance-aware split could even out latency.
- LoRA sharding uses the same seam but the pipeline optimiser currently drives
  the offload (full-FT) path; a LoRA-adapter optimiser across stages is a small
  follow-on.
