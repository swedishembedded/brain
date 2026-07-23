# brain — workspace architecture

brain is a Cargo **workspace** of small, single-responsibility crates. The
engine is pure Rust + raw WGSL, **fp32-only / core-compute-only** (single bind
group, ≤4 storage buffers/kernel, `@workgroup_size(64)`, no atomics/subgroups/f16)
so the exact same kernels run on old desktop GPUs and on WebGPU in the browser.

## Crate graph

```
                         kernels  (WGSL source of truth; pure strings, no deps)
                            │
        ┌──────────────┬────┴───────┬───────────────┐
        ▼              ▼            ▼                ▼
     gpu-core      paramstore     optim           (every model crate)
   (device,        (param/grad/  (AdamW +
    dispatch,       Adam bufs)    grad clip)
    readback)         │   ▲          │
        ▲             └───┘          │
        └──────────────┬─────────────┘
                       │
   checkpoint ◄────────┤  (weight container + manifest/SHA-256 + expert shards;
   (portable I/O)      │   target-agnostic, no wgpu)
                       │
   autodiff ───────────┤  (shared SSA forward-cache / reverse-mode scaffolding)
                       │
        ┌──────────────┼───────────────┬───────────────┐
        ▼              ▼               ▼               ▼
       moe            gpt          timeseries          pid
   (sparse MoE)   (dense GPT)   (float-IO, MSE)   (control xfmr + data)
        │              │               │               │
        └──────┬───────┴───────┬───────┘               │
               ▼               ▼                        │
          federated          eval                       │
   (shard split/assemble,  (perplexity, exact-match,    │
    train-scope, router      routing/ablation/          │
    integration, anchor      marginal-utility)          │
    + router losses)          ▲                          │
               │              │                          │
        data ──┴──────────────┴──────────────────────────┤
   (char + BPE tokenizers, dataset generators,            │
    loaders, masking/alignment, normalization)            │
                                                          │
   gradcheck  (finite-difference gradient checker — the   │
              backprop correctness gate; replaces the     │
              former PyTorch oracle)                      │
                                                          ▼
   cli (`brain` binary) ── aggregates every model + data + eval + gradcheck
   web (`brain_web`, wasm32+WebGPU) ── PID inference demo (gpu-core, checkpoint,
        optim, paramstore, pid); compiles to nothing off wasm32
   vulkan (optional, `vulkan-coopmat`) ── ash + naga coopmat matmul; NOT a
        default member, so default `cargo build` stays pure-wgpu
```

## Multi-GPU scaling (in `brain-model`)

The `model` crate carries the architecture-agnostic scaling layer, written once
over the `Model` trait so every model gets it:

- `collective` — transport-agnostic collectives (`Collective` + `HostCollective`);
  the seam a future network/cluster transport plugs into.
- `grid` — the `(tensor, pipeline, data)` process `Grid` + `LocalGroups`
  (rank↔coord mapping, per-group collectives; Megatron PTD-P layout).
- `parallel` — `DataParallel<M>` (replicate + fused all-reduce optimiser).
- `shard` — `Pipeline<M>` over the `Shardable` seam (split layers, auto-placed,
  micro-batched).
- `plan` — `plan_tp` tensor-parallel planner (capacity-first cost model).

Tensor parallelism's mechanic (col/row-parallel MLP + attention, fwd+bwd) is
validated in `crates/model/tests/tensor_parallel.rs`. See
[`SCALING.md`](SCALING.md), [`DATAPARALLEL.md`](DATAPARALLEL.md),
[`SHARDING.md`](SHARDING.md), [`TENSOR_PARALLEL.md`](TENSOR_PARALLEL.md).

## Conventions

- **WGSL is the source of truth.** All kernels live in `crates/kernels/wgsl/*.wgsl`
  and are embedded as `pub const`s + a `src(name)` lookup. Every consumer
  references them by name; no kernel text is duplicated.
- **Default build is pure wgpu.** `crates/vulkan` is excluded from
  `default-members`; build it with `cargo build -p brain-vulkan` or enable the
  cli `vulkan-coopmat` feature.
- **The web surface is additive.** `brain-web` is the only wasm/WebGPU crate and
  is empty off `wasm32`; the native `brain` binary never depends on it.
- **Backprop is gated by `gradcheck`** (finite differences), not by a Python
  oracle — brain is self-contained and pure-Rust.
- Short lib names (`gpu_core`, `kernels`, `moe`, `pid`, …) on `brain-*` packages.

## Build & test

```bash
cargo build                 # native default-members (pure wgpu)
cargo test                  # full suite (set MOE_SKIP_GPU_TESTS=1 with no GPU)
cargo build -p brain-vulkan # optional coopmat path
cargo build -p brain-web --target wasm32-unknown-unknown --features webgpu
```
