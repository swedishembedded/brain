# qwen35 - roadmap

Qwen3.8-27B (HF `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type:
"qwen3_5"`; llama.cpp `general.architecture = "qwen35"`) - the **dense** sibling of
`crates/qwen35moe` (`LLM_ARCH_QWEN35MOE`). Same hybrid 3:1 Gated-DeltaNet : gated-GQA
mixer split (`full_attention_interval=4`), same partial/M-RoPE, same
per-head-interleaved sigmoid attention-output gate - but a plain dense SwiGLU MLP
instead of the 256-expert MoE, plus a single-layer MTP head sharing
`embed_tokens`/`lm_head`, and a spliced Qwen3-VL-style vision tower (27 blocks,
hidden 1152, no DeepStack) reusing `crates/qwen3vl` unchanged. Weights ship as
DeepSeek-V3-style blockwise FP8 (E4M3, 128x128 `weight_scale_inv` in BF16), one
safetensors shard per decoder layer (`layers-{0..63}.safetensors`) plus
`outside.safetensors` (embed/lm_head/final-norm/whole vision tower, all BF16) and
`mtp.safetensors`. `reasoning_effort` (xhigh/medium/low) is a chat-template
system-prompt injection only - no architectural surface.

Unlike qwen35moe's port, `transformers.models.qwen3_5` (torch 2.13 / transformers
5.14.1) is installed on the development machine, so this port is gated by a real
`porting.md` §5 parity ladder against real reference goldens, not structural
correctness alone. The one part with no available reference is the MTP head:
`transformers`' own loader discards `mtp.*` on load
(`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so it is implemented
structurally and never parity-claimed here.

## Done

- [x] M1: hoist the zero-coupling GDN scratch allocators (`gdn_chunk_size`,
  `GdnScratchBufs`/`GdnScratchTrainBufs`/`GdnBwdScratchBufs`) out of
  `crates/qwen35moe/src/model.rs` into `crates/model/src/gdn.rs`, migrate qwen35moe.
- [x] M2: reference goldens (`tools/goldens/qwen35_dump_reference.py`) - tiny
  all-dims-distinct text config, plus vision at real dims (depth 27/hidden 1152).
- [x] M3: `crates/qwen35` config/param-manifest/init.
- [x] M4: FP8 blockwise import (`crates/model/src/fp8.rs`, safetensors `F8_E4M3`
  support, two-way tensor coverage, the `(1+w)` RMSNorm fold applied on direct
  HF-safetensors import - unlike qwen35moe, which imports only from GGUF where
  llama.cpp's conversion already bakes the fold in).
- [x] M5: text-only forward at tiny dims, climbing parity rungs 1-3 against the M2
  goldens (cosine + rel_l2 + max_abs at every stage).
- [x] M6: backward + `gradcheck::check_qwen35` (both mixer types; wide init on
  `in_proj_qkv`/`conv1d` from day one per lessons.md #40, plus a no-zero-FD
  assertion so the same hollow-gradcheck failure mode cannot recur unnoticed).
- [x] M7: MTP head (`Qwen35::run_mtp_forward`/`mtp_backward` in
  `crates/qwen35/src/model.rs`; `mtp.layers.0.*` is one full Gated-Attention
  decoder layer reusing `layer_gqa_fwd`/`mlp_fwd` unchanged via weight-name-prefix
  parameterization, unlike glmdsa's position-wise-only MTP block). Gated by
  `gradcheck::check_qwen35_mtp` (every `mtp.*` tensor, seed 8, zero tolerance
  failures) plus `crates/qwen35/tests/mtp_convergence.rs` (a real training loop
  still reduces the combined loss; zeroing `mtp.fc_e`/`mtp.fc_h` moves the loss,
  proving the head is load-bearing rather than merely wired). Structural only -
  no reference oracle available here, see below.
- [x] M8: LoRA (rank/alpha adapters on all 12 targetable leaves -
  `crate::config::lora_targets()`: qwen35moe's 9 GDN/GQA projections plus this
  model's own dense-MLP `gate`/`up`/`down`, which qwen35moe never targets since
  its MLP is MoE) + full finetune (`crates/qwen35/src/finetune.rs`, mirrors
  `qwen3::finetune` - `Mode::FullOffload`/`Mode::Lora`, genuinely new surface
  for this family, qwen35moe has none). `Qwen35::lora_fwd`/`proj_bwd`'s LoRA
  branch mirror `qwen35moe::model::Qwen35`'s exactly (two-matmul + `AXPY`
  forward fusion, frozen-base dX-only backward + `A`/`B` adapter grads).
  Gated by `gradcheck::check_qwen35_lora` (every `.lora_a`/`.lora_b` tensor
  across both mixer types, zero tolerance failures across 6 probed seeds);
  `tests/lora_freezes_base.rs` (a real training loop changes only the
  adapters, every frozen base weight bit-identical); `tests/lora_roundtrip.rs`
  (adapters survive a save+reload cycle with real optimizer steps, reloaded
  logits reproduce the trained model and differ from the untrained base by a
  real margin, plus a `lora`-key-absent checkpoint loads as a plain model);
  `tests/convergence.rs` (full finetune and LoRA both drive a fixed batch's
  loss down, plus a cyclic-sequence memorization floor).
- [x] M9: vision tower splice. `crates/qwen35/src/vl.rs` (near-verbatim from
  `crates/qwen35moe/src/vl.rs`, adapted to this crate's explicit-`Gpu`
  constructor convention) composes `qwen3vl::encoder::{VisionEncoder,
  PatchMerger}` - reused completely unchanged - with a new embedding-splice
  seam added to `crates/qwen35/src/model.rs`
  (`enable_mm_splice`/`write_img_embeds`/`read_d_img_embeds`/
  `write_mrope_tables`, wired via `model::vlm::{splice_fwd,splice_bwd}` right
  after the token-embed gather / right before the final `tok.weight`
  backward), mirroring `qwen35moe::model::Qwen35`'s own splice seam exactly.
  Real-dims parity IS achieved (contrary to this file's earlier "no
  independent oracle" note for the composite, which is a separate, narrower
  claim - see below): `tools/goldens/qwen35_vision_dump_reference.py`
  freshly constructs the REAL `transformers.models.qwen3_5.
  Qwen3_5VisionModel` at its real dims (depth=27, hidden=1152, confirmed
  byte-for-byte identical to `VisionConfig::qwen3_omni()` apart from
  `out_hidden_size`/`deepstack_indexes`, checked field by field against the
  installed reference before writing the dumper) with random weights under a
  fixed seed - no checkpoint download needed, matching this milestone's own
  "real dims, random weights" scope (full real-WEIGHT vision parity is
  M10's job). `tests/vision_parity.rs` gates `VisionEncoder`/`PatchMerger`
  (patch embed, two tapped blocks, pre-merger hidden, post-merger output)
  against that golden: cosine 1.0000000000, max_abs ~3e-5, rel_l2 ~3e-6 at
  every stage, both backends - the vision tower reuse is numerically
  correct at this model's real scale. `tests/vl.rs` covers what genuinely
  has no oracle (the full composite - vision tower + splice + decoder - at
  random weights has none): end-to-end finite loss, the splice is
  load-bearing at both the `Qwen35Vl` level (perturbing pixels moves the
  loss - a uniform pixel shift is a poor probe here, since the vision
  blocks' own LayerNorms are shift-invariant by construction, so the test
  uses a structurally different random draw instead) and, at much higher
  margin, directly at the decoder-level splice mechanism in isolation
  (`enable_mm_splice`/`write_img_embeds` with two deliberately large-margin
  explicit embeddings), the splice backward gradient is nonzero and finite,
  and the M-RoPE positions for the image run match `get_rope_index`'s own
  independent computation exactly.
- [x] M11: CLI, caps, residency, serving, docs. Turned out to need more than
  a port: `crates/qwen35` had no incremental single-token KV-cache decode, no
  sharding constructor, and no paged-serving engine before `caps.rs`/
  `serve.rs`/`shard.rs`/CLI/residency could be built. Landed in three
  commits: (1) decode-step core - `Qwen35::step`/`reset_decode_cache`/
  `run_decode_step`/`layer_gdn_decode_step`/`layer_gqa_decode_step` reuse the
  ALREADY-shared `model::block::gqa_decode_step`/`model::gdn::
  {gdn_recurrent_step, gdn_causal_conv1d_step}` primitives (the same ones
  qwen35moe's own decode step calls), written fresh with this crate's own
  weight-name-prefix convention (from M7) rather than duplicating
  qwen35moe's layer-index-keyed copy - more directly reusable if qwen35moe
  migrates onto it later (a smaller M12 follow-up now); `Qwen35::new_shard`
  + `model::Shardable` (`crates/qwen35/src/shard.rs`) with `run_forward`/
  `backward` retrofitted with shard-conditional gating, verified behaviorally
  identical on the whole-shard path by the full existing test suite (all
  passed unchanged) - `cfg.mtp` requires a whole shard (asserted at
  construction, MTP needs `res[n_layers]` and the shared `lm_head`, both
  only valid there). `decode_step.rs` proves `step()` reproduces
  `logits_all()` position-for-position (worst maxabs ~1e-7); `shard_parity.rs`
  mirrors qwen35moe's own gate (self-skips on this box, 0 discrete GPUs).
  (2) `crate::sample` (generation over `step`), `crate::serve::Engine` (a
  single-GPU `PagedDecoder`: real per-block-id GQA KV pool, a private
  `GdnSlot` map keyed by `BlockTable::blocks()[0]` for the GDN state the
  trait has no parameter for), `crate::caps` (the `capability::Provider`) -
  all three mirror qwen35moe's own modules almost verbatim (the decode-step
  orchestration they wrap is architecture-identical); `serve.rs`/
  `sample_generate.rs` tests confirm the paged engine reproduces `step()`'s
  decode token-for-token on both backends. (3) `crates/cli/src/{qwen35_cli,
  resident_qwen35}.rs`, `catalog.rs` `ModelEntry`, `resident.rs::
  build_executor` arm, docs (`docs/models/qwen35.md`, README + index.md +
  AGENTS.md rows - not the quickstart, per scope).
  **Found and fixed a real pre-existing bug while wiring `model_dir.rs`'s
  family dispatch**: `qwen35moe`'s own checkpoint save/import/LoRA paths
  stamped `ModelCard` id/family `"brain/qwen35"`/`"qwen35"` (a leftover from
  before this dense sibling existed) - directly colliding with this crate's
  own correct family. Fixed at the source (`qwen35moe::model::Qwen35::save`,
  `qwen35moe::import::import_mmap`, `qwen35moe::lora::save_adapter`,
  `qwen35moe::config::to_json`'s `"model"` field) to stamp `"qwen35moe"`
  instead, verified against qwen35moe's own full test suite (unchanged) plus
  the CLI's full test suite (unchanged) before adding `crates/qwen35`'s own
  `"qwen35"` arm to `model_dir.rs`.
  Two deliberate scope cuts from qwen35moe: no `precision`/int8 param in
  `caps.rs` (no `q8.rs` for this crate, not in the approved M11 scope); no
  `export` subcommand in the CLI (no NPU/ONNX export path for this arch, an
  already-recorded gap); `reasoning_effort` is NOT wired into `caps.rs` -
  neither `qwen3::chat` nor `qwen35moe::caps` implement it today (only
  `enable_thinking`, a plain bool), and no verified Qwen3.8 prompt-injection
  convention was found to implement it against without guessing, so it is
  recorded as a gap below rather than fabricated.

- [x] M12 part A: migrated qwen35moe off its private `crate::q8::Qwen35Q8`
  mixer API onto `model::ops::{Ops,Act,Weight}`, mirroring `crates/qwen3`'s own
  migration. `q8.rs` narrowed to the MoE-expert path only (`Q8GqaLayer`/
  `Q8GdnLayer`/`Q8Mixer`/`mm8` removed; `is_i8_linear`/`quant`/`Lin8`/
  `Lin8Expert` unchanged); `model.rs` gained `ops: Ops` +
  `weights: HashMap<String, Weight>` and an `ops_linear` dispatcher,
  `layer_gdn_fwd`/`layer_gqa_fwd` rewritten to go through it (backward is
  untouched - it always reads fp32 `ParamStore` directly, same as before).
  `PIPELINES` became a `pipelines()` function (mirroring `qwen3::model::
  pipelines`) since `Ops::new` requires the full façade kernel set registered,
  not just the tiers a model plans to use.
  **Found and fixed a real pre-existing bug**: `Qwen35Config::tiny()`'s
  `linear_value_head_dim: 5` made `linear_value_dim() = 30`, not a multiple
  of 4 - harmless under the old `mm8`-only int8 path (which never touched
  `tiny()`), but `Ops::act` quantizes its activation input eagerly and
  unconditionally on EVERY build (fp32 included), so this broke every fp32
  forward through the mixer too. Fixed by changing it to 2 (giving
  `linear_value_dim() = 12`), preserving the pairwise-distinct-dims invariant.
  Also found: `Weight::upload`'s caps-based dtype promotion means an int8
  build's mixer weights now silently demote to fp32 on a backend without DP4A
  support (e.g. this engine's CPU JIT) - `model_i8_smoke.rs`'s CPU test used
  to assert a hard panic there; rewritten to assert the (now correct) fp32
  fallback behavior instead, matching `qwen3`'s already-established contract.
  Full `qwen35moe` test suite (all 60+ tests across lib + 9 integration
  files) and `check_qwen35moe`/`check_qwen35moe_a_log_elementwise`/
  `check_qwen35moe_lora` gradcheck all pass unchanged in shape.
- [x] M12 part B: hoisted the GDN and gated-GQA mixer orchestration into
  `crates/model` as two new modules, `model::gdn_mixer` and `model::gqa_mixer`
  (new `brain-audio` dependency on `brain-model`, for the shared conv1d
  fwd/bwd `gdn_mixer_fwd`/`gdn_mixer_bwd` reuse). Each exposes one fwd/bwd
  function pair taking explicit `*Ids`/`*Shape`/`*Weights`/`*Grads` structs
  (no config trait) plus the layer's ALREADY-projected activations
  (`mixed_qkv`/`bproj`/`aproj`/`z` for GDN; `q_full`/`k`/`v` for GQA) and
  returning the pre-`out_proj`/`o_proj` activation - the exact boundary
  `model::block`'s own doc states ("linear projections stay in the model").
  Both `qwen35` and `qwen35moe`'s `layer_gdn_fwd`/`layer_gqa_fwd`/
  `gdn_mixer_bwd`/`gqa_mixer_bwd` now call these directly, keeping only their
  own projection dispatch (LoRA, and for `qwen35moe`, `model::ops::Weight`
  int8) local; `GdnLayerActs`/`GqaLayerActs` in both crates slimmed to a thin
  wrapper around the hoisted `GdnMixerActs`/`GqaMixerActs` plus the one
  locally-needed buffer (`gated`/`ctx_gated`). Gated by new
  `crates/model/tests/gdn_mixer_equivalence.rs`: since `model` cannot depend
  on either downstream crate, it proves the real cross-crate risk instead -
  the same shared function resolved through two INDEPENDENTLY, DIFFERENTLY
  ORDERED pipeline registrations (mirroring the fact that `qwen35`'s and
  `qwen35moe`'s own local kernel-index numbering never agrees) produces
  bit-identical fwd+bwd output from identical input. Verified via the full
  `qwen35`/`qwen35moe`/`model` test suites (all pass unchanged) plus
  `check_qwen35`/`check_qwen35_a_log_elementwise`/`check_qwen35_lora`/
  `check_qwen35_mtp` and `check_qwen35moe`/`check_qwen35moe_a_log_elementwise`/
  `check_qwen35moe_lora` gradcheck (all pass, values unchanged from before
  the hoist).

## Not yet done

- [ ] M10: real-weight streaming parity (fetch the 30.9 GB FP8 checkpoint;
  per-layer streaming forward parity for layers {0, 3, 63}; full real-weight
  parity of the vision tower; embed/lm_head spot checks).
- [ ] M13: performance pass (profile-first; native device-side FP8 GEMM only if
  the profile says arithmetic is the limiter, not before).

## Recorded gaps (this development machine has no discrete GPU and 18 GiB usable RAM)

- No whole-model 27B forward, no whole-model torch reference, no e2e generation or
  perplexity number on real weights - unreachable at 27B vs 18 GiB with no
  discrete GPU. Rungs 4-5 of the parity ladder are out of reach here.
- No multi-GPU shard parity (`discrete_gpu_count() == 0` self-skips it) - and note
  qwen35moe's own `shard_parity.rs` does not run on this machine either, so any
  claim it protects a refactor here is a claim about a different machine.
- No int8 tier at all for this crate (no `q8.rs`, unlike qwen35moe) - not in
  the approved M11 scope; `caps.rs` has no `precision` param as a result.
- No serving throughput/latency or residency measurement on real weights.
- MTP head: structurally implemented, **no reference oracle** (see above) -
  gradchecked and overfit-tested, never parity-claimed.
- Vision + decoder fused end-to-end on real weights is not runnable (needs both
  towers resident simultaneously).
- No NPU (`NpuModel`) implementation this port - the firmware blocker on this
  exact host is diagnosed separately, not re-run here.
- `crate::serve::Engine` (M11) is single-GPU, one truly-active sequence at a
  time on the GPU (real continuous batching at admission/scheduling, never
  batched dispatch), no prefix-cache reuse, no chunked/batched prefill, and
  never loads a LoRA adapter (the adapters train and gradient-check
  correctly; folding a trained adapter into the serving path - like
  `qwen3::lora::fold_adapter_into` - has no counterpart here yet) - matches
  qwen35moe's own `serve.rs` scope exactly, not a this-box limitation.
- `reasoning_effort` (xhigh/medium/low) is not wired into `caps.rs`: no
  verified Qwen3.8 prompt-injection convention was found to implement it
  against - only `enable_thinking` (reused from `qwen3::chat`) is real.

Never write an intermediate full-precision whole-model file (~108 GB) - quantized
device buffers must be built directly from the compressed FP8 checkpoint, same
constraint as qwen35moe.
