# Data-parallel training across GPUs

A **full replica** of the model on each GPU, each processing a different slice of
the step's micro-batches **concurrently**, then a gradient all-reduce so every
replica applies the identical update. This is the *throughput* path (a training
speedup) for models that fit on one card; it composes with the pipeline
[sharding](SHARDING.md) *capacity* path. Implemented for Qwen3 in
[`crates/qwen/src/dataparallel.rs`](../crates/qwen/src/dataparallel.rs) as
`DataParallel`.

## The design that actually wins on this box (2× P40, no NVLink)

Naive data-parallel (separate all-reduce, then a per-replica optimiser) is
*slower* than one GPU here — the 2.4 GB gradient sync and the optimiser dominate.
Two findings shaped the working design:

1. **Never clip on-GPU for a 152k-vocab tied model.** The on-GPU AdamW's grad
   clip uses `gradnorm_sq`, a single-threaded reduction over the 155M-row
   `tok.weight` gradient — **~30 s/step**. The offload optimiser computes the same
   norm on the host with rayon in a fraction of that. So the replicas hold only
   **weight+grad** on-GPU (`Role::Offload`); the moments live in host RAM.

2. **Fuse the all-reduce into the optimiser.** Because the optimiser already
   pulls every gradient to the host, do the cross-replica **sum there**, run
   **one** host AdamW update (shared state — all replicas are identical), and
   broadcast the new weights back. Reading grads *once* and updating *once* — vs a
   separate all-reduce plus a per-replica optimiser — is the difference between
   0.75× and a real speedup. Both cards' reads and writes are overlapped (one
   thread per replica), so the PCIe transfers run in parallel.

`Backend: Send + Sync` on native makes the per-card threads sound; each thread
takes a disjoint `&mut Qwen` (the model is `Send`, not `Sync`).

## Correctness

`tests/dp_parity.rs`: running `K` micro-batches split across 2 replicas +
all-reduce produces the identical accumulated gradient as one GPU running the
same `K` with grad-accum `K` — **worst per-parameter grad rel 1.10e-7** (the 1e-7
is float summation order in the reduce). This also confirms the backward pass
accumulates into the grad buffers.

## Speedup (real 0.6B, block 512, `tests/integration_qwen3.rs::qwen3_dataparallel_speedup`)

| micro-batches/step | single-GPU | 2-GPU DP | end-to-end | fwd+bwd (compute) |
|--:|--:|--:|--:|--:|
| 4  | 10338 ms | 7730 ms | **1.34×** | 2.96× (7085→2394 ms) |
| 8  | 17670 ms | 11159 ms | **1.58×** | 2.40× (14103→5880 ms) |

The **fwd+bwd compute parallelises ~3×** across the two cards. End-to-end is
capped by the **fixed** fused reduce+optimiser cost (~5.3 s/step — PCIe-bound, no
NVLink), which is paid **once per optimiser step regardless of micro-batch
count**. So higher grad-accumulation amortises it and the speedup climbs toward
the compute limit. Per-replica VRAM is weight+grad only (~9.4 GB for 0.6B at
block 512), moments in host RAM.

## Running

```rust
use qwen::{DataParallel, QwenConfig};
let mut dp = DataParallel::new(cfg, batch, seqlen, &init, &[0, 1]); // one replica per GPU
dp.zero_grads();
dp.forward_backward(&microbatches);          // concurrent across cards
dp.adamw_step(step, lr, wd, Some(1.0), 1.0 / microbatches.len() as f32);
dp.save("out.weights");
```

## Where each multi-GPU mode fits

- **Data-parallel** (this) — model fits one card, want it *faster*. Speedup grows
  with grad-accumulation; capped by the PCIe gradient sync (NVLink would lift it).
- **Pipeline sharding** ([SHARDING.md](SHARDING.md)) — model too big for one card.
  Distributes weights; bit-exact; no speedup on its own.
- They **compose**: shard a large model across a group of GPUs, replicate the
  group data-parallel — 2D parallelism. The seams (`Pipeline`, `DataParallel`,
  the host-staged transfer + fused optimiser) are all in place for it.
