# brain - workspace architecture

brain is a Cargo **workspace** of many small, single-responsibility crates
(currently in the neighborhood of 90). The engine is pure Rust + raw WGSL,
**core-compute-only** (single bind group, <=8 storage buffers/kernel, no
atomics/subgroups, no `shader-f16` extension, `@workgroup_size(64)` apart from
the register-tiled matmuls at 256) so the exact same kernels run on old desktop
GPUs and on WebGPU in the browser. The portable WGSL path is the reference and
the fallback; it is never the only path:

- **Activations and master parameters are fp32** on the portable path. Reduced
  *storage* tiers are decoded inline by pure-integer WGSL, needing no device
  feature: `Weight::{F32, BF16, F16, I8, Q4, NF4, KQuant}` (`model::ops`),
  with `Weight::upload` the one capability-gated place a tier is chosen
  (`want.promote(caps.numeric)`). bf16/f16 here are packed storage, not f16
  arithmetic.
- **int8 is a first-class compute path**, not storage-only: DP4A dot-product
  WGSL kernels (`matmul_i8*.wgsl`, `moe_linear_gated_i8.wgsl`,
  `crates/kernels/wgsl/`) drive the int8 paged KV cache, the `--precision int8`
  inference paths and INT8 PTQ for the Intel NPU export.
- **Tensor cores are the default where the device has them**, selected on
  queried capability through the gpu-core provider seam (cuBLAS TF32/bf16 for
  GEMMs, fp16 `mma` in paged flash prefill, int8 tensor-core GEMM); the fp32
  WGSL/CUDA-core path is what a gradient check or parity gate pins
  (`GemmPrecision::Fp32Reference`).
- **Training state is fp32 today**: `paramstore` holds weight, gradient and
  Adam moments as fp32 buffers, `optim::offload` keeps fp32 host moments, and
  placement charges 4 bytes per element except packed int8. A frozen base may
  be held reduced (bf16, int8, NF4) only through `Weight`, i.e. for adapter
  training. There is no FP8, NVFP4 or MX storage tier and no low-precision
  full fine-tune yet; the planned placement of those concepts is in "Where the
  quantized load/train concepts live" below.

`AGENTS.md` is the routing guide (which crate does what, task → file); this
document is the *shape* of the workspace: how the layers stack and what may
depend on what.

## Crate graph

Six layers. Each may depend only on layers above it (and on its own layer).
The table below is generated from `scripts/gates/crate-layers.map`, which
`scripts/gates/check-crate-layers.sh` enforces for every workspace package;
where the prose diagram that follows disagrees with it, the table wins and the
diagram is stale.

<!-- crate-layers:begin (generated from scripts/gates/crate-layers.map; edit the map, then run scripts/gates/crate-layers.py --write-doc) -->
```text
layer 1 - kernels, architecture registry, device-neutral contracts
    arch, backend-api, kernels, kernels-cuda, memauth, seed
layer 2 - accelerator seam: backends and the gpu-core facade
    backend-cpu, backend-cuda, backend-vulkan, backend-wgpu, cuda-aot, gpu-core, vulkan, wgsl-
    cpu, wgsl-cuda
layer 3 - training substrate and shared contracts
    atif, audio, audit, camera, capability, checkpoint, data, diffusion, dit, events, forecast,
    gguf, model, modelref, modelstore, neuro, onnx, optim, paramstore, promote, residency, rlcd,
    search, shutdown, solve, stats, survival, testutil, timeseries, trace, vae, vision,
    weightset
layer 4 - models
    arcface, bisenet, campplus, captioner, chronos2, clip, codeformer, connectome, controlnet,
    cosyvoice, decide, decisions, deepseek2, deepseek2ocr, deepseekocr2, deepseekvl, diamond,
    ecapatdnn, eval, fastvlm, fincast, florence2, flux1, flux2, fly, flybody, gemma4,
    genieredux, glmdsa, gpt2, horizon, imaging, imgpipe, instantid, januspro, kronos, lfm2,
    llava, llmserve, lpips, ltxv, mimi, minicpmv, minimaxh3, minimaxmusic3, modernbert,
    moondream3, mujoco, mvs, nemotronasr, npu, pulid, qwen3, qwen35, qwen35moe, qwen3asr,
    qwen3omnimoe, qwen3tts, qwen3vl, qwen3vlmoe, recon, rrdbnet, s3dit, s3tokenizer, sam1, sam2,
    scrfd, sdxlunet, sfm, splat, supir, t5encoder, timesfm3, toyautoencoder, toymoe, toypid,
    toyseq2seq, viewport, vqgan, wan, wm-core, wm-display, worldmirror2, yolov8, zipdepth
layer 5 - cross-cutting evaluation and tooling
    bench, capture, fcbench, federated, gradcheck, modelcost, rl
layer 6 - serving, front ends, SDK, samples
    apiserve, appopts, brain, braintop, capability-mock, catalog, cli, dbus, hfsm, loader, perf,
    runtime, sample-decision-arena, sample-decision-doom, sample-decision-intents, sample-
    decision-json, sample-decision-patterns, sample-decision-rubiks, sample-decision-salesagent,
    sample-decision-triage, sample-fly-interactive, sample-imagegen-batch, sample-imagegen-
    generate, sample-learning-reader, sample-learning-rlcd, sample-learning-visualclick, sample-
    reconstruction-splat, sample-study-chat, sample-study-document, sample-study-timeline,
    sample-text-embed, sample-text-embed-train, sample-text-encoder-finetune, sample-text-
    retrieve, sample-text-retrieve-json, sample-tts-clone, server, serving, web
```
<!-- crate-layers:end -->


```
 ─── 1. kernels & architecture registry ────────────────────────────────────
   kernels        WGSL source of truth (hundreds of files -> pub consts +
                  src(name)); pure strings, no deps. Nothing below duplicates
                  kernel text.
   arch           the canonical architecture registry (`brain_arch::ARCHS`):
                  one row per architecture - id, display name, domain, HF
                  class names, GGUF spelling, default checkpoint ref. Zero
                  deps, wasm-safe - the CLI resolver, GGUF import, HF fetch
                  and ModelCard all read this ONE table instead of their own
                  drifting copy. Naming rule: `id` is `[a-z0-9]+` (no
                  underscore/hyphen) - llama.cpp's `LLM_ARCH_*` vocabulary
                  lowercased where an entry exists, else a brain-defined name
                  in the same style. Dependency-free like `kernels`, and
                  sits below `checkpoint` (layer 3) in the stack - every
                  layer above it, `checkpoint` and `model` included, is free
                  to depend on it.

 ─── 2. the accelerator seam ───────────────────────────────────────────────
   backend-api    Backend/GraphBackend traits + neutral buffer/step handles
                  + registry. A NEW BACKEND DEPENDS ONLY ON THIS.
        ▲   ▲   ▲   ▲
        │   │   │   └── backend-cuda   CUDA Driver API (libcuda.so.1 dlopen'ed
        │   │   │                      at run time - no build-time CUDA dep).
        │   │   │                      wgsl-cuda (naga IR -> CUDA C++) + NVRTC,
        │   │   │                      compiled lazily per kernel on first
        │   │   │                      dispatch; kernels-cuda holds the
        │   │   │                      hand-written tier and its registry
        │   │   └────── backend-vulkan ash + naga (WGSL -> SPIR-V)
        │   └────────── backend-cpu    wgsl-cpu (naga IR -> Cranelift JIT),
        │                              rayon across cores, AVX2 fast paths
        └────────────── backend-wgpu   wgpu: Vulkan/Metal/DX12/GL/WebGPU [default]
                            │
   gpu-core       ◄─────────┘  device facade: ONE Gpu/DeviceBuffer/Step API.
                  Picks a backend at runtime (--device / --backend /
                  BRAIN_DEVICE); all four are compiled into every native
                  build. This is the ONLY thing abstracted - there is no
                  per-backend model code.

 ─── 3. training substrate ─────────────────────────────────────────────────
   paramstore ⇄ optim      param/grad/Adam buffers; AdamW + global grad clip
   checkpoint              .safetensors container + manifest/SHA-256 + expert
                           shards; target-agnostic, no wgpu, no fs on wasm.
                           `checkpoint::packed` is the quantized-tensor seam:
                           TensorDescriptor/QuantEncoding (what bytes mean),
                           the codec registry (reference tile decode/encode),
                           PackedTensorReader/TensorSink (bounded tiled I/O,
                           never a whole tensor), MemoryBudget (shared staging
                           accounting) and the Preserve/Repack/Convert load
                           executor with its LoadPlan report. Model crates
                           supply names and shapes, never codecs or loaders.
   autodiff                shared SSA forward-cache scaffolding (placeholder)
   data                    char + GPT-2 BPE tokenizers, dataset generators,
                           loaders, masking/alignment, normalization
   model                   architecture-agnostic Model trait + generic trainer,
                           shared block builders (block.rs, vit.rs), paged KV,
                           the KV storage descriptor (kv_tier: element codec x
                           scale hierarchy; f32/bf16/int8/fp8-E4M3 tiers) with
                           its generic calibration interface (kvcalib), and
                           the multi-GPU parallelism layer (see below)
   promote                 the model-agnostic promote/reject decision: the
                           Environment/Verifier reward seam, the exact paired
                           sign test, the four-bar gate over already-scored
                           pairs, and the document/fact probe contract. A
                           LEAF - it depends on no other brain crate at all.
                           It sits HERE, not next to `rl` at layer 5, because
                           a layer-4 model crate has to be able to gate its
                           own candidate adapter: with the gate up there,
                           `brain-qwen3 -> brain-rl -> brain-bench ->
                           brain-qwen3` was a real Cargo cycle. `rl` and
                           `bench` re-export from here rather than owning
                           anything, so no caller changed and nothing is
                           duplicated; `scripts/gates/check-crate-layers.sh`
                           fails the build if this closure ever climbs.
   rlcd                    RLCD (reinforcement learning for calibrated
                           decisions): proper scoring rules over soft target
                           distributions, and (as later work lands) cost
                           matrices/Bayes risk/value-of-information and
                           calibration metrics. A LEAF for the same reason as
                           `promote`: this is pure `&[f32]` host arithmetic
                           with no model dependency, but it used to live
                           inside `brain-decide` (layer 4), so no OTHER
                           decision-capable model crate (e.g. `modernbert`'s
                           Laya head) could reach it without depending on a
                           sibling model crate. `decide` re-exports it as
                           `decide::loss` rather than owning it, so no caller
                           changed and no logic exists twice; the same
                           `check-crate-layers.sh` gate covers it.
   survival                survival estimation and evaluation arithmetic:
                           weighted Kaplan-Meier, the censoring distribution,
                           Aalen-Johansen, IPCW concordance and Brier score,
                           D-calibration, calibration at a horizon. A LEAF for
                           the same reason: pure f64 host arithmetic that a
                           model crate, an application or a sample evaluates
                           time-to-event predictions with, without a model.
   seed                    where a run's seed comes from: `random_seed()` for a
                           run whose caller gave none and `parse_seed()` for
                           one they did. std only, no dependencies: a LEAF so
                           that `data` (which re-exports it as
                           `data::rng::random_seed`), `capability`, `appopts`
                           and `forecast` (which depends on no other brain
                           crate) resolve an unset seed the same way. Every
                           default seed in the workspace is random and is
                           reported where it is drawn.

 ─── 4. models ─────────────────────────────────────────────────────────────
   decoder LMs    gpt2  qwen3  qwen35moe  toymoe  glmdsa  toypid  toyseq2seq  toyautoencoder  timeseries
   vision / 3D    yolov8 ─┐         zipdepth ─┐     worldmirror2   splat
                        └─ vision ───┘  (shared conv blocks, fold_bn)
   diffusion      diffusion + dit + vae ──> s3dit  (+ qwen3 text encoder)
   audio          audio ──> mimi, ecapatdnn ──> qwen3tts   (qwen3tts reuses qwen3)
   forecasting    forecast (seam) <── chronos2  kronos  fincast ;  fcbench
   world models   wm-core <── diamond, genieredux ;  wm-display (SDL)

 ─── 5. cross-cutting services ─────────────────────────────────────────────
   federated      vertical expert shard split/assemble, hash-verified manifests
   eval           perplexity + exact-match (LM) and mAP/precision/recall (det)
   bench          model-agnostic architecture-evaluation battery (DecoderLm seam)
   gradcheck      finite-difference backprop gate; replaces the PyTorch oracle
   onnx ──> npu   ONNX export + INT8 PTQ + OpenVINO Intel-NPU runtime. NOT a
                  gpu-core backend - a separate whole-graph compile path.
   capture        V4L2 webcam (ioctl FFI, YUYV->RGB)

 ─── 6. serving & front-ends ───────────────────────────────────────────────
   capability     typed ActionSpec manifests - the ONE dispatch shape for CLI
                  (`brain <arch> <verb>`) and the event API alike
   capability-mock a deterministic, weight-free Provider (gradient images,
                  moving-gradient video, sine-tone PCM, derived text/bytes) -
                  mirrors any real Manifest 1:1 or builds one by hand, so an
                  in-process capability consumer never has to load real
                  weights to exercise the contract
   catalog        the served-model catalog: manifest + weight-free provider
                  ctor + residency adapter for every registered model (~70
                  crates), in ONE list, depending on nothing CLI-local, so
                  any process can serve a model under a memory budget
   serving        what a serving process stands on: the machine probe, the
                  one shared `residency::Executor` built from the catalog and
                  a models directory, models-directory discovery; `cli`
                  calls it instead of owning the assembly
   residency      weight tiering GPU/RAM/disk (LRU + budget) + job scheduling
   loader         the resolver core, models-directory lookup, GPU/CPU
                  placement, and default-checkpoint auto-fetch (behind an
                  explicit DownloadPolicy) - depending on nothing CLI-local,
                  same split as `catalog`; `cli` calls it instead of owning
                  the implementation, and it is what a public SDK loads a
                  model through without linking `cli` itself
   events/hfsm ──> runtime      JSONL event protocol + event-driven HSM controller
   server         one protocol, three transports: stdio, TCP, unix socket
   dbus           zbus surface over capability::Registry (fd-passed frames)
   sdk (pkg `brain`)  the public embeddable SDK facade - `ImagePipeline` over
                  flux2 AND s3dit via `loader`'s resolver (dispatching
                  internally on the resolved architecture, uniform past
                  that), no CLI process and no capability-dispatch machinery
                  in the loop. Package name is a deliberate, sole exception
                  to `brain-<short>` (commented in root Cargo.toml and in
                  the crate's own Cargo.toml).
   cli            the `brain` binary - aggregates everything
   web            wasm32+WebGPU toypid demo; compiles to nothing off wasm32
   vulkan         optional coopmat matmul (`vulkan-coopmat`); NOT a default
                  member, so the default build stays pure-wgpu
```

Distinct from `backend-vulkan`: `crates/vulkan` is the older cooperative-matrix
*acceleration* experiment, while `backend-vulkan` is a full eager backend behind
`backend-api`. Only the latter is a default member.

## Multi-GPU scaling (in `brain-model`)

The `model` crate carries the architecture-agnostic scaling layer, written once
over the `Model` trait so every model gets it:

- `collective` - transport-agnostic collectives (`Collective` + `HostCollective`);
  the seam every transport plugs into.
- `netcollective` - `Collective` over TCP (coordinator-star topology). Swapping
  it in turns single-box multi-GPU training into a cluster with no change to the
  drivers, the grid, or any model; reductions stay bit-reproducible (fixed rank
  order), same as the host transport.
- `distributed` - SPMD data-parallel training on the `Collective` seam: full
  replica per rank, all-reduce the gradient, identical optimiser step, so
  replicas stay bit-identical with no separate broadcast.
- `grid` - the `(tensor, pipeline, data)` process `Grid` + `LocalGroups`
  (rank↔coord mapping, per-group collectives; Megatron PTD-P layout).
- `parallel` - `DataParallel<M>` (replicate + fused all-reduce optimiser).
- `shard` - `Pipeline<M>` over the `Shardable` seam (split layers, auto-placed,
  micro-batched).
- `plan` - `plan_tp` tensor-parallel planner (capacity-first cost model).
- `paged` - paged KV-cache foundation (block allocator + `BlockTable`) that the
  serving engines build on.

Tensor parallelism's mechanic (col/row-parallel MLP + attention, fwd+bwd) is
validated in `crates/model/tests/tensor_parallel.rs`. See
[overview](../../docs/concepts/scaling/overview.md), [data parallel](../../docs/concepts/scaling/data-parallel.md),
[pipeline sharding](../../docs/concepts/scaling/pipeline.md),
[tensor parallel](../../docs/concepts/scaling/tensor-parallel.md).

## Where the quantized load/train concepts live

The quantized-training programme (`.agents/roadmap/quantized-training.md`)
adds concepts that must land in one layer each; this is their placement. An
entry names a target, not a shipped capability - check the roadmap's status
before relying on one.

| concept | layer / crate | rule |
|---|---|---|
| codec decode/encode, quantized GEMM tiles, fused optimizer-update kernels | 1 `kernels`, `kernels-cuda` | kernel text lives only here; portable WGSL reference first |
| kernel choice from operand descriptors + device caps | 2 `gpu-core` provider | the one selection seam (`check-kernel-selection.sh`); no model crate picks a kernel by format |
| tensor descriptor, quantization encoding, codec registry, packed tensor reader/sink, memory budget, load plan executor, declarative name/layout mapping (`remap.rs`) | 3 `checkpoint` | the lowest layer that needs a shared type owns it; no wgpu dependency |
| trainable parameter with independent storage/gradient/optimizer-state dtype, optimizer state formats and storage, transactional state | 3 `paramstore`, `optim` | one AdamW definition; state formats are data, not new optimizers per format |
| `Weight` variants and generic quantized forward/input-grad/weight-grad ops, adapter runtime (definition, execution, optimization), KV storage descriptor, saved-activation policy, generic trainers and objectives, parameter manifest and capability traits | 3 `model` | extend `model::ops`, `adapter::sites`, `model::training`, `model::conformance` rather than adding parallel seams |
| graph, parameter manifest (roles, adapter targets), declarative checkpoint mapping, I/O and objectives, custom ops | 4 model crates | nothing else: no codec, loader, adapter, optimizer or trainer code |
| residency, serving, sdk | 5-6 | consume `MemoryBudget`/`LoadPlan` and the generic trainers; never re-implement them |

Acceptance test of the layering: adding a storage format, adapter kind,
optimizer or objective touches no model crate, and adding a model touches no
codec, adapter, optimizer or trainer code. No upward dependencies and no
duplicated descriptors (the KV tier adopts the checkpoint crate's encoding
rather than keeping its own).

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
  oracle - brain is self-contained and pure-Rust.
- Short lib names (`gpu_core`, `kernels`, `toymoe`, `toypid`, …) on `brain-*` packages.

## Build & test

Always go through the Makefile - it wraps cargo with the project's expected
flags and targets.

```bash
make build/debug                  # debug build, native default-members
make build/release                # optimized ./target/release/brain
make test                   # full suite (MOE_SKIP_GPU_TESTS=1 with no GPU;
                            # BRAIN_DEVICE=cpu runs the whole suite GPU-free)
make test/gradcheck              # finite-difference backprop gate
make test/parity                 # cross-backend parity gate: CPU == Vulkan == NPU
make generate/kernels/registry          # after adding/removing a crates/kernels/wgsl/*.wgsl
make web/dev                # wasm32 + WebGPU demo (crates/web)
```

The two non-default targets are built explicitly:

```bash
cargo build -p brain-vulkan # optional coopmat path
cargo build -p brain-web --target wasm32-unknown-unknown --features webgpu
```
