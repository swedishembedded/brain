# quantized training, PEFT and memory-bounded loading - roadmap

Scope: Quantized training, PEFT and memory-bounded loading over any checkpoint precision.

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
| `check-training-capabilities.sh` | the capability page is generated from `training-capabilities.tsv`; every named test exists and is a live `#[test]`; every registered conformance tier and SDK chat fine-tuner has a row | none (the declaration is the source) |
| `check-model-flags.sh` | no `--<architecture>-<option>` flag in production code; options are model-agnostic (`model::policy`) or derived from the memory plan | `model-flags.allow` (the deprecated aliases) |

A migration that removes a call site deletes or lowers its allowlist entry in
the same commit. A new violation is a design defect, not an allowlist addition.

## Status (reconciled against origin/main on 2026-10-21)

Each item is done (D), partial (P) or open (O), with the commit that landed it.
The per-cell matrix and its tests are `docs/reference/training-capabilities.md`.

### Done
- D Descriptor and codec: `TensorDescriptor`/`QuantEncoding`, codec registry, bounded packed
  reader, sink and tiled Preserve/Repack/Convert executor (8cbdec63a, f52600c8e, b94a78726);
  `SafetensorsSink` writes a model tile by tile in each descriptor's encoding (56ae14f05);
  `convert_to_file` is the one-off bounded conversion into a packed file (c5ddbfa02), and
  qwen35 `write_converted_base` uses it so a streamed NF4 base never requantizes per step
  (32976b9b5).
- D Generic frozen-base forward, dX and dW for every `Weight` tier including FP8, NVFP4 and
  MX (3ec33bb7e, 456190d2d, 892e263a1); `Ops::read_host` reads any tier back through the
  bounded tile decode (cdf3dc135).
- D Adapter runtime: definition, execution and optimization split; rsLoRA, LoRA+, LoRA-FA,
  IA3, LoKr, DoRA (922d326b4, 27602fa70, 7d3de820d, 37da9d4ef); generic serving bank for LoRA.
- D Base formats: LoRA bases are declared by the model and chosen by one `--base-format` flag
  (da2cf8f8c); `ChatFineTune::base_format` and `PreferenceFineTune::base_format` spell it for
  every architecture (b646cbbce, cc4abb2df); every declared base trains through `train_lora`
  and an undeclared one is refused by name (09fd9bd0e, 9c7bdb9de); the qwen35 FP8 base trains
  like its fp32 decoding (022079253); qwen3 trains over NF4 and FP8.
- D Objectives and acceptance: DPO, GRPO and top-K distillation are `ModelFit` visitors in
  `rl::fit` (06b285b65); the four-architecture acceptance workflow is `model::acceptance`
  over qwen3, qwen35, qwen35moe and flux2 (3bb497b6a); conformance C8 holds the weighted loss
  to the chunked-head check (ae94ecd6b).
- D Full fine-tuning state: Adafactor, logarithmic 8-bit state, bf16 device moments and the
  `TrainState` seam (6d3edf527); resident full fine-tune of qwen3 with bf16, FP8 and int8/NF4
  quantization-aware weights, an off-device master, `Ops::weight_grad`, deferred publication,
  planner refusal, streamed export and transactional resume (b87dd8e0d, 4ce2d9357); one typed
  `FitOpts` precision, the planner narrowing optimizer state only when authorized, and exact
  resume for every full fine-tune (24ada03fa, adab58541).
- D Vocabulary tables in the resident state: the embedding table and an untied LM head are
  `ResidentRole::Table` tensors (bf16 whatever the linears' encoding, master and moments off the
  device, bf16 or f32 gradient), resident for the whole step and updated after the backward in a
  layer-streamed one; the head's bf16 gradient is produced in bounded vocabulary runs
  (`HeadGrad`), the embedding gradient scatters over the distinct looked-up rows
  (`Ops::embedding_grad`, a padded list a tape replays), the host update reads gradients through
  a bounded window (`Ops::read_range`), and a tensor wider than one storage binding is refused by
  name. Wired into qwen3 and qwen35: at the 27B shape this removes the fp32 table value, gradient
  and two moments (16 bytes a parameter, about 40 GB) from the device.
- D Generic optimizer-state precision: every model with a `ParamStore` and the device `Optim`
  exposes them (`Model::optimizer_state`) and the fit loop applies the run's state format before
  the first step (`trainprec::apply_train_precision`, `ParamStore::apply_state_policy`); the bf16
  and int8 moment kernels are one list (`optim::MOMENT_KERNELS`); one shared spec
  (`trainprec::check_optimizer_state_formats`) runs per model.
- D Generic host-held optimizer state: `moments=host` (or a host-only format such as log8) moves
  every trainable tensor's moments to the host optimizer inside `Optim` (`ParamStore::offload_all`,
  no device moment buffer left), exact resume commits it for any model
  (`Optim::host_commit`, `trainprec::check_resume_with_host_moments`), the default
  `optimized_params`/`read_moments`/`write_moments` of every model read the store in any format,
  and data-parallel replicas and pipeline stages release their unused device moments; qwen3's own
  offload optimizer is gone.
- D Device memory telemetry: `Gpu::memory_usage` (live, reserved, peak of live since
  `reset_memory_peak`) from the backend's own accounting on wgpu and CUDA; every fit step reports
  it (`StepWork::device_memory`, the SDK record's `device_peak_bytes`/`device_reserved_bytes`, the
  qwen35 progress line).
- D Row-compacted MoE training activations: `MoeActs::compact` holds each expert's activations for
  its routed rows only (`rows * top_k` instead of `rows * n_experts` rows a layer, 32x fewer at
  top-8 of 256), `experts_fwd_train` and `moe_layer_bwd` gather the rows and scatter the outputs
  and gradients back (`rows_scatter_add.wgsl`); gradchecked and equal to the dense path
  (`model/tests/moe_block_gradcheck.rs`), wired into every MoE trainer: qwen35moe, and deepseek2
  and glmdsa, whose recorded passes are now `model::moe::PassSeg`s (fixed dispatches and each MoE
  layer's experts, built per pass); glmdsa's own per-expert MoE training code is gone.
- D Activation recompute through the run's precision (`activations=keep|recompute|bf16|fp8`,
  `Model::set_saved_policy`): deepseek2, glmdsa, gpt2, lfm2 and horizon share one activation buffer set per layer kind
  and replay each layer (routed experts included) before its backward, bit-equal to keeping every
  layer (`model::saved::check_recompute`); qwen3 and qwen35 answer from their build; spectrallm, ouro
  and huginn always rebuild each layer and refuse keep (`check_always_recomputes`); every other
  model keeps every layer and refuses by name. Narrow boundaries through the shared
  `model::saved::Residuals` (`check_quantized_boundaries`): gpt2, deepseek2, glmdsa, lfm2, horizon, spectrallm, ouro, huginn.
  qwen35moe (eager) keeps no layer's activations past the forward and replays each; its narrow-boundary spec runs
  with every token routed to every expert (fp8 flips a fixed fraction of top-2 choices against f32). Open: the
  vision trainers.
- D Layer streaming for any `ParamStore` model (`weights=f32,window=N`): `ResidentValue::F32` (the value its own
  master), `ops::DeviceIo`/`StoreDevice` (the state on the model's own handle), `model::tape::LayerTape` (per-layer
  segments), `model::streamed::StreamedStore` (converts a built store, aliases its entries to the window, drives
  the shared `resident_ft::adamw_step`), `Model::set_resident` from `apply_train_precision`. Specs:
  `streamed::check_layer_streaming`, `trainprec::check_exact_resume`. gpt2, spectrallm, deepseek2 and glmdsa (their
  `PassSeg` passes through `streamed::LayeredPass`) and qwen35moe (eager: its passes report the edges) and lfm2 adopt it.
  ouro and huginn refuse a window by name (each layer runs in every step of the recurrence). Open: horizon;
  Construction-time streaming (`Role::Streamed`, `StreamedStore::attach`, `Model::new_resident`,
  `check_streamed_construction`): gpt2, lfm2, spectrallm, deepseek2, glmdsa, qwen35moe. Open: horizon.
- D Memory: `model::saved` saved-activation policy (keep, recompute, bf16/fp8 boundaries)
  priced in memplan and wired into qwen3 and qwen35 LoRA training (a4a1242bc); flash attention
  backward without `[H,T,T]` probabilities for qwen35moe (fc6df1080) and flux2, including the
  bidirectional kernels and `AttentionPath::{Flash,Auto}` (ce1456612, 5fce3f645, 4f3f5f9d0);
  the streamed trainer draws scratch from a replay arena (bd495d218) and reports layer wait
  against compute time (971757bfd).
- D Serving: residents build from the stored checkpoint view, serve it packed and park it
  packed when demoted (d3878f0cf, 61e30dc56, 37ab12718); FP8 KV tier with a descriptor
  (9ae3ea5ca, 412dce436, fcf0c7690); `model::memplan` with the lever order (c135b6f86).
- D Gates: G3 format-logic also matches `Weight::FP8` and `Weight::Micro` (7e7b6e86d).

### Partial
- P Layer-streamed full fine-tune: qwen3 and qwen35 train every weight with the stored values on
  the host, a bounded window of layers on the device, per-layer updates between the backward
  segments and a device-measured global clip (exact two-pass or running), bit-equal to the
  resident step on tiny models and loss-equal on Qwen3-0.6B; a real 27B-class run is not done.
  qwen35moe and glmdsa still hold fp32 resident weights (see Open).
- P LoRA bases: qwen35 and qwen3 take any declared format; DoRA is only over fp32-resident
  bases (verify the refusal path before relying on it); the serving bank is LoRA-only.
- P Memory planner levers: CompactKv, ConvertWeights, OptimizerState and the saved-tensor
  policy have consumers; Offload and Microbatch levers do not.
- P Whole-tensor loads: the LLM residents moved to packed sources; the allowlist still holds
  the non-LLM crates and `TensorSource` still defaults to whole tensors.
- P Mixed-format models: per-tensor tier from the descriptor on upload and refusal of mixed
  fused concats are done; an end-to-end test with FP8, NF4, bf16 and f32 tensors in one
  model through load, forward, backward, save and reload is missing.
- P Speed of the packed paths: the NF4 loader is measured at about 2.6x the FP8 loader, but
  end-to-end NF4 training step time against preserved FP8 is not shown at 2x (see Open).

### Open (not done; do not read any item above as covering these)
- O Layer-streamed full fine-tune of a 27B-class model on a real checkpoint (the mechanism is
  proven on tiny models and Qwen3-0.6B; the update is still host-bound, device Adafactor and
  device master updates are not done).
- O Pinned staging plus a dedicated transfer stream (`Pool::Pinned` has no consumer).
- O NF4 at least 2x faster than preserved FP8 end to end: measured 1.25x to 1.4x on a
  contended card; only the loader reaches 2.6x. Needs an uncontended measurement.
- O flux2 int8 base onto `Ops` (its int8 trainer still has its own path).
- O qwen35 int8 and int4 LoRA bases (qwen35 serves int8/int4 but trains over bf16, NF4, FP8).
- O FP4 and INT4 KV tiers (FP8 KV is done; the rest waits for evidence).
- O Native FP8 and NVFP4 tensor-core GEMMs through the provider (portable decode path today).
- O Removal of the deprecated `--<architecture>-<option>` aliases (`model-flags.allow`).
- O PiSSA, EVA and LoftQ initialization; LoHa, OFT and VeRA.
- O Full fine-tune on qwen35moe and glmdsa over resident packed weights (qwen35 has it). qwen35moe
  first needs bf16 storage variants of its expert training kernels (`model::moe` reads fp32 expert
  buffers).
- O Device memory by category (weights, gradients, moments, activations, scratch) against the
  planner's demands; `Gpu::memory_usage` reports live, reserved and peak bytes per step (wgpu and
  CUDA; Vulkan and the CPU JIT report none) but not what they hold. The freed-block cache trades
  throughput for about 14 GiB of reserved memory at 8K tokens (knowledge 310, 311).
- O Training state sharded over two devices (optimizer state and masters split by tensor, global
  clip over both); only the layer pipeline splits a model across devices today.
- O The value read of `DeviceTiles` (commit, export, a tensor without a master) still decodes a
  whole tensor on the host; gradients are read through a bounded window.


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
Status: DPO, GRPO and top-K distillation are `ModelFit` visitors (`rl::fit`) and
run on qwen3 (bf16, int8, NF4, FP8) and qwen35 (bf16, NF4, FP8); the acceptance
workflow is `model::acceptance` (`LoraArch`/`LoraStep`), registered for all four
architectures in `rl/tests/lora_acceptance.rs`. A LoRA trainer declares its base
formats (`model::policy::TrainBases`), one `--base-format` flag parses them, and
`LoraRequest::base` carries the choice through the generic trainers.

### Phase 5 - full fine-tuning state (paramstore, optim)
Status: the state manager, Adafactor, logarithmic 8-bit state, bf16 device moments, resident
and QAT weights with an off-device master, deferred publication, streamed export and exact
resume are landed (see the Status section above and `docs/guides/training/optimizer-state.md`).
Open: the same trainer over qwen35moe and glmdsa, a real 27B-class layer-streamed run, and
device kernels for the host-side master update.
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

## Migration map for the files the phases rewrite first

Each row says what the file does today and the change that removes it from the
gate allowlists (or, where the file is exempt from a gate, what the gate cannot
see). Entries in the allowlists carry the same fixes.

| file | today | change (phase) |
|---|---|---|
| `checkpoint/src/lib.rs` `TensorSource` | `with_tensor_chunks` and `with_tensor_range` default to the whole tensor; no packed reader; exempt from the whole-tensor gate | packed tensor reader with no whole-tensor fallback; defaults removed so an unbounded wrapper fails to compile (1) |
| `checkpoint/src/weightio.rs` `DerivedCheckpoint`, `Renamed`, `InMemory` | f32 and `U32` only, advertises F32, clones | descriptor-carrying derived views that pass packed bytes through (1) |
| `checkpoint/src/mmap.rs` | whole-tensor `tensor_f32` reachable from production loaders | production callers move to tiled reads; the accessor stays for reference modes only (1) |
| `qwen35/src/fp8_source.rs` | HF naming, FP8 scale discovery, dequantization and reading in one model-crate type | FP8 codec in the registry; the type becomes a declarative name map over the shared reader (1) |
| `qwen35/src/base_source.rs` | rejects GGUF as a training base; the base format is `model::policy::TrainBases` (F32, bf16, NF4, FP8), frozen-only below fp32 | any registered codec accepted; base precision from the descriptor policy (1, 5) |
| `qwen35/src/stream_train.rs` | widens each layer's FP8 base to fp32 | packed-base backward through the generic quantized op (2) |
| `paramstore/src/dtype.rs` | charges one fp32 per element (packed int8 as four) | physical bytes from the tensor descriptor; the planner shares it (1, 6) |
| `optim/src/offload.rs` | `OffloadSlot = (String, Vec<f32> x3)` host master, m, v | bounded-tile state manager with independent state dtypes and a transactional on-disk form (5) |
| `model/src/adapter/device.rs` | `is_adapter_param` hard-codes `.lora_a`/`.lora_b` | parameter-set driven (3); until then the adapter-naming gate keeps all other spellings out |
| `model/src/kv_tier.rs` | closed `KvTier {F32, Bf16, Int8}` | storage descriptor on the shared encoding, FP8 tier added (6) |

## Experiments to run once the phases land
A: full fine-tune with fp32 versus int8 versus streamed int8 AdamW versus
Adafactor. B: long-context activation policies. C: rollouts with bf16, int8 and
FP8 KV. Each reports persistent memory, peak scratch, file-backed resident
memory, device memory and host memory separately.
