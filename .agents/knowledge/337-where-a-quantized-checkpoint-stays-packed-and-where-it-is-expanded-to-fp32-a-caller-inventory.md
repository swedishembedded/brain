<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 337. Where a quantized checkpoint stays packed and where it is expanded to fp32: a caller inventory

Read at 4af5f59bf (2026-10-20). Purpose: before any loader is moved onto a bounded, packed-preserving
path, know which callers keep bytes packed, which expand one tensor, which expand the whole model, and
where precision is chosen. Invariant used throughout: a quantized checkpoint is never implicitly
expanded to full precision; loading is Preserve (copy packed bytes), Repack (lossless layout change),
or Convert (numeric change under an explicit policy, bounded tiles, recorded). Nothing below is a
measurement; every class is read from the code and cites where.

## 1. The reading API and what each entry point costs

`checkpoint::TensorSource` (`crates/checkpoint/src/lib.rs`) is the only seam loaders use.

| entry point | contract | cost on the real implementors |
|---|---|---|
| `with_tensor` | f32 slice of the WHOLE tensor | `MmapSafetensors` and `MmapGguf` decode the whole tensor into one `Vec<f32>` (`mmap.rs` `tensor_f32`, `gguf.rs` `with_tensor`); a bf16/fp8/quantized tensor is expanded to 4 bytes per element |
| `with_tensor_chunks` | f32 chunks of at most `max_elems` | bounded only where overridden: `MmapSafetensors`, `MmapGguf`, `RemapSource` (whole-tensor kinds only), `Qwen35Source` (FP8 bands). The trait default is whole-tensor-as-one-chunk, so `HashMap`, `AdapterOverlay`-style wrappers over a non-overriding source, `DitWeights`, `Concat`/`RowPermute` fetches and every transformed leaf (`GdnFixSource`) silently fall back to the whole tensor |
| `with_tensor_range` | f32 elements `[start, start+len)` | default decodes the whole tensor and slices; only `MmapGguf` (`tensor_range`) and `RemapSource` slices are range-bounded. `MmapSafetensors::tensor_f32_range` exists as an inherent method, not a trait override |
| `raw_words` | zero-copy `u32` view | only when stored as F32 or packed-int8 `U32` and 4-byte aligned; bf16/fp8/GGUF-quant return `None` |
| `with_tensor_u32_chunks` | bounded packed-int8 words | brain's own int8 layout only (`PACKED_INT8_LAYOUT`); no other packed format has a bounded reader |
| `raw_blocks` | borrowed GGUF block bytes + `BlockLayout` | the only Preserve/Repack primitive for GGUF; `None` for safetensors, for any value-transforming wrapper, and for `Concat`/`RowPermute` |

There is no reader that yields packed FP8, NF4, MX or NVFP4 blocks. `DerivedCheckpoint`
(`weightio.rs`): `tensor_f32` and `tensor_u32` only; `Renamed` advertises every tensor as `F32`
and reads through the source's whole-tensor f32 decode; `InMemory` holds f32 and clones on read; so a
derived view over a quantized source expands it. Its callers: `qwen3tts::import` and
`ecapatdnn::import` (`Renamed`), `yolov8::import` and `clip::import` (`InMemory`), `mimi::import`
(`CodecView`), `qwen3omnimoe::int8_thinker_resident` (`Int8View`, passes `U32` through).
`WeightReader::tensor` (whole tensor, f32) is the eager accessor behind `qwen35::import`
(`import_layer`, `import_mtp`), `qwen35::stream` (norm, embedding rows through
`tensor_f32_range`), `ltxv`/`vqgan`/`deepseek2ocr`/`januspro` imports and `sdxlunet::import`
(`tensor_f32`).

## 2. Classification of loaders

Classes: **P** preserve or repack, packed bytes reach the device; **T** per-tensor fp32 (one tensor
at a time on the host as f32, then uploaded at a chosen tier); **W** whole-model fp32 on the host
(a map or container of every tensor); **D** whole-model fp32 resident on the device
(`ParamStore`); **H** host fp32 trainer or optimizer state.

### P - packed bytes survive
- `gguf::int8_direct::try_i8_rect` and `model::int8::q8_0_rect`: a Q8_0 GGUF tensor becomes brain
  packed int8 as a byte repack (exact, no fp32). Reached through `model::int8::quantize_from`,
  `upload_quantized`, `upload_rect`. Callers: `qwen3/src/q8.rs`, `qwen35` head and MTP
  (`stream.rs:1220`, `int8_gguf_resident.rs:2981`, `int8_gguf_resident/mtp.rs:169`, `dflash2.rs`),
  `qwen35moe/src/q8.rs` (`quantize_from`), `wan/src/block.rs`, `gemma4/src/block.rs`,
  `s3dit/src/block.rs`, `ltxv/src/block.rs`, `flux2/src/model.rs` (`lin_rect`, via
  `DitWeights::try_i8_rect`, which declines for a LoRA-touched tensor).
  Falls back to class T for every non-Q8_0 source: `quantize_from` then decodes with
  `with_tensor_chunks` (bounded by `UPLOAD_CHUNK_WORDS`) and requantizes each row block to int8, an
  implicit Convert (Q4_K, bf16, fp8 -> int8) with no policy object and no record. A sub-rectangle
  that is not Q8_0-block aligned (`quantize_rect_from`) materializes the whole tensor as f32.
- `paramstore::upload::Uploader::packed` and `raw_words`: brain-native packed-int8 `U32` tensors and
  F32 tensors are copied word for word in bounded writes (`UPLOAD_CHUNK_WORDS`).
- `gguf::kquant::try_kq_rect` builds the canonical K-quant device layout (host relayout, no
  arithmetic on weights) and `Weight::KQuant` executes it, but NO model loader calls it: it is used
  by `crates/gguf/tests/kquant.rs`, `model/tests/matmul_kq.rs` and `ops_kquant.rs` only. The
  comments in `qwen3`, `qwen35`, `qwen35moe` model constructors say the same ("this crate never
  builds a `Weight::KQuant`"). Q4_K/Q5_K/Q6_K GGUF tensors therefore reach the device only by
  decode to f32 and requantize.
- NF4 and bf16 weights (`Weight::NF4`, `Weight::BF16`) are never loaded packed from a file: both are
  produced by `Weight::upload` from f32 (section 2 T). There is no NF4 or FP8 checkpoint reader.

### T - one tensor at a time as f32, then `Weight::upload`
`Weight::upload(ops, raw, n, k, want)` (`model/src/ops.rs`) takes an f32 slice of the whole
`[n, k]` tensor and packs it (F32, BF16, F16, I8, Q4, NF4); it cannot take a chunk or a packed
source. The host peak is therefore one full tensor in f32 (the 248320 x 5120 embedding or head
is 5 GB in f32). Callers:
- `qwen3/src/model.rs:1376,1407` (the seven linears per layer and the half-precision tables),
  `qwen35/src/model.rs:1703` (`upload` closure over `TierPolicy`), `qwen35moe/src/model.rs:1558`
  (hard-coded `Dtype::I8` when `quant(name)`), `qwen3/src/serve.rs` (13 sites, from the class W map),
  `dflash2.rs:370`.
- `qwen35::fp8_source::Qwen35Source::with_tensor` (FP8 base for the 27B LoRA/QLoRA run): the FP8
  blocks are dequantized band by band (`WHOLE_TENSOR_BAND_ELEMS`) but assembled into one whole f32
  `Vec` for the callback; `with_tensor_chunks` is band-bounded. The 27B NF4 and bf16 LoRA bases are
  built this way (host peak about one tensor, knowledge 234).
- `paramstore::new_with_roles_src` / `Uploader::tensor`: bounded chunks to a device f32 buffer, so the
  host side is bounded but the device side is class D.
- `s3dit`, `wan`, `ltxv` (`block.rs` `with_tensor` ... `quantize_weight_q4` at `block.rs:1606`,
  `load_*_tensors_from_source` at `block.rs:3186,3216`, `dit.rs:1168,1524`, `av_stream.rs:110`,
  `pipeline.rs:971`), `gemma4` (`block.rs:177,221`, `model.rs:228,260,266`), `minimaxh3`
  (`block.rs:503-616`, `load_dev`), `moondream3`, `glmdsa`, `lfm2` importers (a writer sink fed
  one whole tensor per call), `qwen3/src/export.rs` (`dtype.encode` over one whole tensor).
- `model::shard::Shardable::new_shard_from` default: collects every tensor of the stage into a host
  `HashMap` of f32 (class W for the stage); only models that override it stream.
- Transient device expansion, bounded: `Ops::decoded_weight_scope` decodes each whole NF4 weight to
  bf16 once per layer pass (2 * n * k bytes per distinct weight while the scope lives). This is the
  one place an NF4 weight becomes a wider tensor on the device. It is a layer-bounded scratch, not a
  model copy.
- Per-layer fp32 expansion in the streamed trainer: `qwen35/src/stream_train.rs`
  `load_layer_f32` calls `import_layer` (the layer's FP8 tensors as f32 `HashMap`) then
  `build_layer_f32` uploads `Weight::F32` for the layer, and `build_head_f32_from_mmap` writes the
  head chunk-wise into an f32 device buffer. Bounded per layer, but the frozen base is widened 4x
  (FP8) per layer on every pass; `Weight::F32` is the only base this trainer accepts
  (`train_base.rs` rejects I8/Q4/KQuant for training).

### W - whole model as f32 on the host
- `qwen3::serve::Engine::tensors_from` builds a `HashMap<String, Vec<f32>>` of the whole model.
  Callers: `Engine::load`, `catalog/src/resident_llm.rs:873` (`engine_tensors`, every Qwen3 resident
  activation), `qwen3/src/eval.rs:61`, `qwen3/src/rollout.rs`, `januspro/src/t2i.rs:151`,
  `deepseekvl/src/model.rs:163`, `cli/src/perf_longctx.rs:426`. `Engine::from_map*` then uploads at a
  tier, so int8/bf16 serving of Qwen3 holds a full fp32 copy in RAM during build.
- `checkpoint::load` / `Container::by_role("")`: whole container, f32. Callers:
  `catalog/src/resident_qwen35.rs:243` and `resident_qwen35moe.rs` (`Engine::from_map(cfg, &weights,
  ...)`, brain-native safetensors only), the qwen3 fine-tune resume path (`finetune.rs:78`).
- `checkpoint::safetensors::read_model_dir` returns every tensor as f32: `qwen35::import::import_dir`
  (the 27B would need about 108 GB), `qwen3vl/src/model.rs:359`, `ltxv/src/pipeline.rs:128`,
  `minimaxh3/src/import.rs`, `worldmirror2`, `t5encoder`, `kronos`, `fincast` imports.
- `Qwen3 serve Engine::fold_adapter` (`sdk/src/text.rs:357,378`): reads each adapted linear as f32,
  folds, writes back through `write_weight` (an fp32 base only, by its own doc).
- Eval/rollout/`read_weight`/`write_weight` on the `Model` trait (`model/src/lib.rs:303`) exchange
  `Vec<f32>` of one named tensor.

### D - whole model as fp32 on the device
`paramstore::ParamStore` allocates every parameter as an fp32 buffer; a trainable parameter also
gets grad, Adam m and v buffers (all fp32); `Role::Frozen` still holds the fp32 weight. About fifty
crates build one (`grep ParamStore::new` under `crates/`; the large ones are `qwen3/model.rs`,
`qwen35/model.rs`, `qwen35moe/model.rs`, `glmdsa`, `deepseek2`, `gpt2`, `fastvlm`, `clip`).
`paramstore/src/dtype.rs` `device_bytes` charges every dtype 4 bytes per element except packed-int8
`U32`; every placement and footprint decision (`qwen3/src/footprint.rs`, `memauth`) inherits that
charge, so a reduced tier is accounted at fp32 where it goes through `ParamStore` and at its true
size only where it goes through `Weight`. The quantized linears are excluded from the `ParamStore`
(`is_quantizable_linear` in `qwen35/src/model.rs`), so a LoRA build over a bf16/NF4/int8 base holds
packed linears plus an fp32 store of norms, embeddings (unless `table_tier`), head and adapters.

### H - host fp32 trainer or optimizer state
- `optim/src/offload.rs` `OffloadAdam`: `OffloadSlot = (String, Vec<f32> x3)` master weights, m, v
  for every `Role::Offload` parameter. Qwen3 full fine-tune (`finetune.rs:94,108`,
  `with_adam_offloaded`) trains in fp32 only (`Dtype::F32`) and exports fp32; there is no
  low-precision full-FT path anywhere (knowledge 336).
- `flux2/src/finetune.rs` host trainer (`--trainer host`): fp32 frozen base plus an effective fp32
  copy; reference only (the device trainer is the default).
- `model::lora` fold paths (`fold_adapter_into`, `flux2::lora::fold_adapters`): f32 per tensor or per
  stage; a fold into an int8 base requantizes lossily (`flux2/tests/lora_requant_int8.rs`).

## 3. Where precision is chosen

Per model crate (conditionals that choose a storage dtype in model code):
- `qwen3/src/model.rs`: `quantized` bool plus a `Dtype`; `Qwen::new_lora_dt_gemm`; trainers accept
  F32, BF16 and I8 only (`sdk/src/finetune.rs` `qwen3_tier`; NF4 is not exposed for qwen3).
- `qwen35/src/model.rs` + `train_base.rs`: `TierPolicy` per leaf (substring rules), `table_tier`
  for embedding/head, `BaseDtype` {F32, Bf16, Nf4} in `base_source.rs`; `BaseDtype::is_frozen_only`.
- `qwen35moe/src/model.rs`: `quant(name)` selects `Dtype::I8` for the nine mixer linears and the
  untied head; MoE experts are separate (`moe_expert_names`); no per-leaf policy, no bf16/NF4.
- `flux2/src/devgrad.rs`: its own `BasePrecision {F32, Int8}`; `flux2/src/model.rs` carries
  `Precision` plus a per-label keep-fp32 list; `ltxv` `QTier`; `s3dit`/`wan`/`gemma4` bool or enum
  arms per block loader. `sdk/src/finetune.rs` `choose_base_tier` picks the tier from device
  capability and a memory estimate (knowledge 336).
- Shared: `Weight::upload` (`want.promote(caps.numeric)` is the one capability gate; refuses FP8,
  NVFP4, MX, KQuant), `Ops::matmul`/`Ops::matmul_dx` dispatch on the `Weight` variant (kernel choice
  stays in the shared op layer), `GemmPrecision` (`set_gemm_precision`), `model::kv_tier::KvTier`
  {F32, Bf16, Int8} (closed enum). `Weight` has F32, BF16, F16, I8, Q4, NF4, KQuant; no FP8 or
  block-scaled float variant.
- `model::adapter::device::is_adapter_param` hard-codes the `.lora_a` and `.lora_b` suffixes; the
  only `AdapterKind` implementation is `LoraPair`.

## 4. Top findings, in priority order
1. `with_tensor_chunks` and `with_tensor_range` default to whole-tensor: every wrapper that does not
   override both is silently unbounded; `checkpoint::srccheck` checks value agreement of the
   entry points, not that the memory bound holds.
2. No packed FP8/NF4/MX reader exists. FP8 is dequantized in `qwen35::fp8_source`, a model crate that
   also owns HF naming, scale discovery and conversion.
3. `Weight::upload` takes a whole f32 tensor; a bounded packed-to-packed path exists only for Q8_0
   -> int8 and brain-native int8.
4. Q4_K/Q5_K/Q6_K have a tested relayout and an executing `Weight::KQuant`, but no loader uses it.
5. Whole-model fp32 host maps remain in Qwen3 serving (`tensors_from`), the qwen35/qwen35moe catalog
   residents (`checkpoint::load`) and every `read_model_dir` importer.
6. `ParamStore`/`device_bytes` charge fp32 for everything but packed int8, so placement over-charges
   quantized parameters that live in the store and the optimizer state is always fp32 on device or
   host.
7. The streamed qwen35 trainer widens the frozen base to fp32 per layer; the resident trainer holds
   bf16/NF4 packed but needs the decoded-weight scratch per layer.
8. `qwen35/src/base_source.rs` rejects GGUF as a training base (`open_base`); the GGUF route
   (`int8_gguf_resident`) is serving-only and converts every non-Q8_0 tensor to int8.
9. mmap bounds the heap but file-backed pages accumulate in resident memory (knowledge 42);
   `advise_drop` is called per tensor by `ParamStore`, `qwen3/src/model.rs` and `minimaxh3`, not by
   every loader.
