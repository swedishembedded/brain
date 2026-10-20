# quantized training, PEFT and memory-bounded loading - roadmap

Goal: any brain model that exposes compatible operations and objectives can use
any registered adapter method or full fine-tuning over any compatible checkpoint
precision, on any backend that has an optimized path or a verified portable
fallback, with memory bounded by contract. Adding a storage format, adapter
kind, optimizer or objective touches no model crate; adding a model touches no
codec, loader, adapter, optimizer or trainer code. Where each concept lives is
in `.agents/rules/architecture.md` ("Where the quantized load/train concepts
live"); the loader inventory this plan starts from is knowledge entry 337; the
per-architecture matrix of what is offered today, with the test behind each
cell, is `docs/reference/training-capabilities.md`.

## Invariants (each one has a gate or a test named below)

1. **A quantized checkpoint is never implicitly expanded to full precision**
   during load, inference, adapter training, full training or offload/reload.
   Loading has exactly three operations: Preserve (copy packed bytes), Repack
   (lossless layout change) and Convert (a numeric change under an explicit
   policy, in bounded tiles, recorded in a load report). Staging and scratch
   budgets are configured and independent of the largest tensor. An unsupported
   source/destination pair is a clear error, never a silent fp32 detour.
2. **Four independent precision policies**: storage, compute, gradient
   accumulation, optimizer state (plus an optional master copy). No policy
   implies another.
3. **Mixed precision is the normal case.** Every tensor keeps its own descriptor
   through load, residency, forward, backward, optimizer update, save and
   reload. Kernel choice comes from operand formats, shape and device
   capability in the shared op and provider layers, never from a model-crate
   conditional. Fused tensors and expert sets with mixed formats are split into
   compatible groups or explicitly repacked, never promoted to fp32.
4. **No silent precision change.** `auto` policies take allowed ranges and
   report every choice. Optimizer selection is explicit, never substituted.
5. **Placement charges physical destination bytes**, from the tensor's
   descriptor, not an fp32 element count.
6. **A model supplies only** its graph, parameter manifest (names, shapes, roles,
   adapter targets), declarative checkpoint mapping, I/O and objectives, and
   custom ops.
7. **Every low-precision path ships a measured accuracy check** against a
   higher-precision reference with format-specific tolerances (absolute error,
   relative L2, cosine, non-finite count). Portable WGSL is the reference and
   fallback; tensor-core paths are the default where the device has them.

## Enforcement (terminal-runnable, in `make check` and the pre-commit hooks)

| gate | what it holds | allowlist (shrink-only, reason per entry) |
|---|---|---|
| `check-crate-layers.sh` + `crate-layers.map` | every workspace package on a layer; no upward dependency; the architecture crate table is generated from the map | `crate-layers.allow` |
| `check-whole-tensor-loads.sh` | no new whole-tensor fp32 read (`with_tensor`, `tensor_f32`, `read_model_dir`, `checkpoint::load`, in-memory derived checkpoints) in production code | `whole-tensor-loads.allow` |
| `check-format-logic.sh` (also run by `check-kernel-selection.sh`) | no `Weight::<format>`, `kernels_cuda::` or codec call in a model crate | `format-logic.allow` |
| `check-adapter-naming.sh` | `.lora_a`/`.lora_b` spelled only inside `model::adapter` | `adapter-naming.allow` |
| `check-placement-accounting.sh` | no new caller of fp32-element sizing helpers | `placement-accounting.allow` |

A migration that removes a call site deletes or lowers its allowlist entry in
the same commit. A new violation is a design defect, not an allowlist addition.

## Status at the time of writing (read from source, not from memory)

- Done: bf16, int8 (group-32) and NF4 frozen bases for adapter training in
  `qwen35` (`Weight::{BF16, NF4}`, the decoded-weight scope) and bf16 and int8
  in `qwen3`; an int8-base device trainer for `flux2`; a 27B NF4 and bf16 LoRA
  run on one card from an FP8 directory streamed tensor by tensor; the
  base-tier choice from device capability (`choose_base_tier`); tensor-core
  GEMMs selected by capability; `Weight::KQuant` execution and its host
  relayout (no loader builds it yet).
- Not done: a tensor descriptor and codec registry as the single description of
  stored bytes (the first piece, `TensorDescriptor`/`QuantEncoding` in the
  checkpoint crate, has just landed and has no consumer); a packed bounded
  reader; any FP8, NVFP4 or MX `Weight`; a quantized or FP8 full fine-tune;
  optimizer state below fp32 or off host RAM; a dtype-aware trainable
  parameter; any adapter kind but LoRA.

## Phases

### Phase 1 - describe bytes and load them bounded (checkpoint, layer 3)
`TensorDescriptor` and `QuantEncoding` (element type, packing, block geometry,
scale hierarchy: global, block, channel; physical versus logical layout); a
codec registry with a CPU reference tile decoder for f32, bf16, f16, FP8
(e4m3, e5m2, block-scaled), int8, NF4 and the GGUF quant types (reusing the
existing decoders); a packed tensor reader with no whole-tensor fallback over
mmap safetensors, sharded HF directories and GGUF; a tensor sink; a shared
memory budget (host staging high-water, pinned memory, file-page residency
released after consumption); a tiled Preserve/Repack/Convert executor that
produces a load plan reporting source and destination encoding, peak staging
and "intermediate full tensors: none".
Gate: a tensor much larger than the staging budget loads within budget; no
read path materializes a whole tensor; an FP8 to NF4 tiled conversion equals the
reference; one budget shared by two concurrent loads holds; a mid-load failure
releases everything. Then the model loaders (`qwen35` FP8 and GGUF sources, the
`Weight::upload` callers, `DerivedCheckpoint`) move onto the planner and the
whole-tensor allowlist shrinks to the reference modes.

### Phase 2 - generic quantized execution (model::ops, provider, kernels)
For every `Weight` variant: forward `Y = X W^T`, input gradient `dX = dY W` and
weight gradient `dW = dY^T X` into a chosen gradient dtype, through bounded
tile decode feeding the tensor-core GEMM by default, with WGSL portable and CPU
reference paths. Generalize the NF4 forward and input-gradient kernels into
this contract rather than per-format copies. Add block-scaled FP8 (E4M3, E5M2)
storage; later NVFP4, MXFP4 and MXFP8 (native NVFP4 tensor-core execution is
Blackwell-only; other devices use the portable path).
Gate: one frozen-base adapter-gradient test across F32, BF16, NF4, I8, KQuant
and FP8 with per-format tolerances against the decoded fp32 reference.

### Phase 3 - generic adapter runtime (model::adapter)
Split the adapter abstraction into definition (parameter specs per target,
initialization, serialization schema), execution (forward and backward over any
base `Weight`) and optimization (parameter groups); the device wiring is
parameter-set driven with no suffix hard-coding. Device-wired rsLoRA, LoRA+ and
LoRA-FA; the serving adapter bank moves behind the adapter site abstraction so
every architecture serves adapters as overlays, without folding into the base;
the catalog's adapter registry stops importing a model crate's types. Then
DoRA, LoKr (never a dense Kronecker product), IA3, and PiSSA/EVA/LoftQ
initialization; LoHa, OFT, VeRA later.
Gate: finite-difference gradient per kind; fused versus unfused parity; save,
reload and overlay parity on two architectures.

### Phase 4 - generic objectives and trainers (model::training, sdk)
Full and preference fine-tuning generic over `Model` plus the adapter sites,
with no per-model dispatch in the sdk; held-out scoring over `Model`; the
chunked-head cross-entropy conformance check enabled; the conformance subject
takes a tensor/direction budget; all supported tiers registered. GRPO and
distillation generic.
Acceptance: the same workflow runs on `qwen3` (NF4, FP8, int8), `qwen35`,
`qwen35moe` and `flux2`.

### Phase 5 - full fine-tuning state (paramstore, optim)
A trainable parameter with independent value storage, gradient dtype, optimizer
state dtype and optional master copy (none, bf16, fp32). A bounded-tile
parameter update (decode tile, update, re-encode) with fp32, bf16 and 8-bit
block-scaled moments and optional stochastic rounding; one AdamW definition
(CPU reference, WGSL, fused CUDA), with Lion, SGD-momentum and Adafactor on the
same state manager. Optimizer state streamed from disk in bounded ranges with a
transactional checkpoint (weights, moments, step, schedule, RNG; atomic commit)
replacing the host fp32 triple. Then the generic full trainer: bf16, then FP8
full fine-tune, layer-streamed with all gradients computed against one
parameter version, streamed low-precision export, resume, then INT8/INT4
quantization-aware training.
Gates: memory ceiling independent of parameter count (measured staging
high-water); resume parity; 8-bit moments against fp32 AdamW on a small fixed
task. Real-model acceptance: quantized load without fp32 materialization, every
parameter updated, a fixed staging budget, export in the chosen format, resume,
serve, held-out and retention against a higher-precision baseline.

### Phase 6 - KV storage and the memory planner
Replace the closed KV tier enum with a storage descriptor built on the Phase 1
encoding; add an FP8 E4M3 KV tier (append, paged decode, flash prefill,
byte-preserving offload) with attention parity against fp32 and measured
capacity against bf16 and int8; a generic attention calibration interface.
INT4/FP4 KV only after evidence. One global planner over weights, gradients,
optimizer state, activations, KV, staging and transfers as separate categories,
choosing in this order: preserve quantized weights, then optimizer-state
formats, then recompute, then lower activation precision (validated), then
offload, then compact KV for rollouts, then a smaller micro-batch; it refuses
before loading anything if no authorized combination fits. Backward ops declare
their saved tensors with a policy (keep, quantize, recompute, offload); recurrent
state, norm and softmax statistics stay high precision; memory-efficient
attention backward never forms the full probability matrix.

### Phase 7 - native low-precision tensor-core paths and research
Native FP8 and NVFP4 tensor-core GEMMs through the provider, LoHa/OFT/VeRA,
masterless low-bit training.

## Experiments to run once the phases land
A: full fine-tune with fp32 versus int8 versus streamed int8 AdamW versus
Adafactor. B: long-context activation policies. C: rollouts with bf16, int8 and
FP8 KV. Each reports persistent memory, peak scratch, file-backed resident
memory, device memory and host memory separately.
