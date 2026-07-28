# brain — workspace architecture

brain is a Cargo **workspace** of ~60 small, single-responsibility crates. The
engine is pure Rust + raw WGSL, **fp32-only / core-compute-only** (single bind
group, ≤8 storage buffers/kernel, no atomics/subgroups/f16, `@workgroup_size(64)`
apart from the register-tiled matmuls at 256) so the exact same kernels run on
old desktop GPUs and on WebGPU in the browser.

`AGENTS.md` is the routing guide (which crate does what, task → file); this
document is the *shape* of the workspace: how the layers stack and what may
depend on what.

## Crate graph

Six layers. Each may depend only on layers above it.

```
 ─── 1. kernels ────────────────────────────────────────────────────────────
   kernels        WGSL source of truth (281 files -> pub consts + src(name));
                  pure strings, no deps. Nothing below duplicates kernel text.

 ─── 2. the accelerator seam ───────────────────────────────────────────────
   backend-api    Backend/GraphBackend traits + neutral buffer/step handles
                  + registry. A NEW BACKEND DEPENDS ONLY ON THIS.
        ▲   ▲   ▲
        │   │   └── backend-vulkan   ash + naga (WGSL -> SPIR-V)
        │   └────── backend-cpu      wgsl-cpu (naga IR -> Cranelift JIT),
        │                            rayon across cores, AVX2 fast paths
        └────────── backend-wgpu     wgpu: Vulkan/Metal/DX12/GL/WebGPU  [default]
                            │
   gpu-core       ◄─────────┘  device facade: ONE Gpu/DeviceBuffer/Step API.
                  Picks a backend at runtime (--device / BRAIN_DEVICE); all
                  three are compiled into every native build. This is the ONLY
                  thing abstracted — there is no per-backend model code.

 ─── 3. training substrate ─────────────────────────────────────────────────
   paramstore ⇄ optim      param/grad/Adam buffers; AdamW + global grad clip
   checkpoint              .weights container + manifest/SHA-256 + expert
                           shards; target-agnostic, no wgpu, no fs on wasm
   autodiff                shared SSA forward-cache scaffolding (placeholder)
   data                    char + GPT-2 BPE tokenizers, dataset generators,
                           loaders, masking/alignment, normalization
   model                   architecture-agnostic Model trait + generic trainer,
                           shared block builders (block.rs, vit.rs), paged KV,
                           and the multi-GPU parallelism layer (see below)

 ─── 4. models ─────────────────────────────────────────────────────────────
   decoder LMs    gpt  qwen  moe  glm  pid  seq2seq  autoencoder  timeseries
   vision / 3D    yolo ─┐            depth ─┐        mirror   splat
                        └─ vision ───┘  (shared conv blocks, fold_bn)
   diffusion      diffusion + dit + vae ──> zimage  (+ qwen text encoder)
   audio          audio ──> codec, speaker ──> tts   (tts reuses qwen)
   forecasting    forecast (seam) <── chronos2  kronos  fincast ;  fcbench
   world models   wm-core <── wm-diamond, wm-genie ;  wm-display (SDL)

 ─── 5. cross-cutting services ─────────────────────────────────────────────
   federated      vertical expert shard split/assemble, hash-verified manifests
   eval           perplexity + exact-match (LM) and mAP/precision/recall (det)
   bench          model-agnostic architecture-evaluation battery (DecoderLm seam)
   gradcheck      finite-difference backprop gate; replaces the PyTorch oracle
   onnx ──> npu   ONNX export + INT8 PTQ + OpenVINO Intel-NPU runtime. NOT a
                  gpu-core backend — a separate whole-graph compile path.
   capture        V4L2 webcam (ioctl FFI, YUYV->RGB)

 ─── 6. serving & front-ends ───────────────────────────────────────────────
   capability     typed ActionSpec manifests — the ONE dispatch shape for CLI
                  (`brain do`) and the event API alike
   residency      weight tiering GPU/RAM/disk (LRU + budget) + job scheduling
   events/hfsm ──> runtime      JSONL event protocol + event-driven HSM controller
   server         one protocol, three transports: stdio, TCP, unix socket
   dbus           zbus surface over capability::Registry (fd-passed frames)
   cli            the `brain` binary — aggregates everything
   web            wasm32+WebGPU PID demo; compiles to nothing off wasm32
   vulkan         optional coopmat matmul (`vulkan-coopmat`); NOT a default
                  member, so the default build stays pure-wgpu
```

Distinct from `backend-vulkan`: `crates/vulkan` is the older cooperative-matrix
*acceleration* experiment, while `backend-vulkan` is a full eager backend behind
`backend-api`. Only the latter is a default member.

## Multi-GPU scaling (in `brain-model`)

The `model` crate carries the architecture-agnostic scaling layer, written once
over the `Model` trait so every model gets it:

- `collective` — transport-agnostic collectives (`Collective` + `HostCollective`);
  the seam every transport plugs into.
- `netcollective` — `Collective` over TCP (coordinator-star topology). Swapping
  it in turns single-box multi-GPU training into a cluster with no change to the
  drivers, the grid, or any model; reductions stay bit-reproducible (fixed rank
  order), same as the host transport.
- `distributed` — SPMD data-parallel training on the `Collective` seam: full
  replica per rank, all-reduce the gradient, identical optimiser step, so
  replicas stay bit-identical with no separate broadcast.
- `grid` — the `(tensor, pipeline, data)` process `Grid` + `LocalGroups`
  (rank↔coord mapping, per-group collectives; Megatron PTD-P layout).
- `parallel` — `DataParallel<M>` (replicate + fused all-reduce optimiser).
- `shard` — `Pipeline<M>` over the `Shardable` seam (split layers, auto-placed,
  micro-batched).
- `plan` — `plan_tp` tensor-parallel planner (capacity-first cost model).
- `paged` — paged KV-cache foundation (block allocator + `BlockTable`) that the
  serving engines build on.

Tensor parallelism's mechanic (col/row-parallel MLP + attention, fwd+bwd) is
validated in `crates/model/tests/tensor_parallel.rs`. See
[overview](scaling/overview.md), [data parallel](scaling/data-parallel.md),
[pipeline sharding](scaling/pipeline-sharding.md),
[tensor parallel](scaling/tensor-parallel.md).

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

Always go through the Makefile — it wraps cargo with the project's expected
flags and targets.

```bash
make build                  # debug build, native default-members
make release                # optimized ./target/release/brain
make test                   # full suite (MOE_SKIP_GPU_TESTS=1 with no GPU;
                            # BRAIN_DEVICE=cpu runs the whole suite GPU-free)
make gradcheck              # finite-difference backprop gate
make parity                 # cross-backend parity gate: CPU == Vulkan == NPU
make kernels-regen          # after adding/removing a crates/kernels/wgsl/*.wgsl
make web/dev                # wasm32 + WebGPU demo (crates/web)
```

The two non-default targets are built explicitly:

```bash
cargo build -p brain-vulkan # optional coopmat path
cargo build -p brain-web --target wasm32-unknown-unknown --features webgpu
```
