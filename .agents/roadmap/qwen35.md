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
- [x] M10: real-weight streaming parity, under a self-imposed 16 GB RAM
  ceiling. Fetched only the shards this milestone's own tests ever read
  (~7.6 GB: `layers-{0,3,63}.safetensors`, `outside.safetensors`,
  `mtp.safetensors`, tokenizer files) rather than the full 30.9 GB - the
  other 61 layer shards are never touched by any test here. New
  `qwen35::import::import_layer` streams exactly one `layers-{l}.safetensors`
  shard via `checkpoint::mmap::MmapSafetensors`, dequantizing only that
  layer's own FP8 tensors (never the whole checkpoint, never a full-model
  `HashMap` - ~108 GB dequantized, impossible on this box regardless of
  streaming). `crates/qwen35/tests/real_weight_streaming.rs` (4 `#[ignore]`d
  tests, self-skip loudly without `BRAIN_QWEN35_DIR`) drives the
  already-hoisted `model::gdn_mixer`/`model::gqa_mixer` (M12) directly on a
  standalone `Gpu` - never constructs a `Qwen35` (that path still assumes
  every layer's weights are present, which a real 27B checkpoint on this
  18 GiB box cannot do; a recorded gap, not something this milestone works
  around). Layers 0 (Gated DeltaNet), 3 and 63 (gated GQA, first and last
  full-attention layer) each match `tools/goldens/
  qwen35_dump_real_layer_reference.py`'s dump of the REAL `transformers.
  models.qwen3_5.Qwen3_5DecoderLayer` forward at cosine=1.000000000,
  rel_l2≈1e-6, peak RSS 2.45 GiB (measured via `brain_testutil::mem`, well
  under the 16 GB ceiling). `embed_tokens`/`lm_head` row spot check
  (`tools/goldens/qwen35_dump_embed_lm_head_rows.py`) matches exactly
  (cosine=1.0, max_abs=0.0), peak RSS 2.37 GiB. Vision-tower real-weight
  parity is NOT done (see gaps below) - the two decoder-mixer types plus
  embed/lm_head were this milestone's load-bearing claim; the vision splice
  already has real-dims (random-weight) parity from M9.
  **Found and fixed two real, confirmed pre-existing bugs, both only
  reachable with actual checkpoint bytes:**
  (1) `checkpoint::mmap::MmapSafetensors` (`open`/`tensor_f32`/
  `with_tensor_chunks`) never got F8_E4M3 support when `crate::safetensors::
  parse`'s eager path gained it - `try_dtype_width` didn't know the dtype's
  byte width, so `open()` failed outright ("unknown dtype 'F8_E4M3'") on any
  file containing one, before `decode_into` (which also lacked an F8_E4M3
  arm) was ever reached. Blocked the streaming mmap path entirely for this
  (or any) real blockwise-FP8 checkpoint. Fixed both functions, mirroring
  `crate::safetensors::parse`'s own F8_E4M3/F8_E5M2 split exactly, with two
  new regression tests. (2) `qwen35::import::classify`'s embedding-tensor
  name was wrong: `model.embed_tokens.weight` (claimed "confirmed" from
  `quantization_config.modules_to_not_convert`, which the embedding was
  never actually a candidate for, since it is never FP8-quantized) versus
  the real checkpoint's actual `model.language_model.embed_tokens.weight`.
  `import_dir`'s own two-way coverage check would have caught this the
  moment `import_dir` ran against the real checkpoint (never a silent
  wrong-tensor placement) - fixed the name, corrected the module doc's
  now-falsified "confirmed" claim, added a regression test pinning the fix.
- [x] M13: performance pass. New `crates/qwen35/src/bin/qwen35_bench.rs`
  profiles one GDN layer, one GQA layer, and the dense SwiGLU MLP at
  `Qwen35Config::qwen38_27b()`'s real dims (random weights - cost depends on
  shape, not values), reporting wall-clock, dispatch count
  (`Gpu::stats()`), and (where the backend supports it) a per-kernel device
  timestamp breakdown, graded against this device's own measured roofline.
  Measured on this box's only available backend, an Intel Arc iGPU
  (Vulkan) - no discrete GPU here, so every number below is iGPU-relative,
  not a datacenter projection - at `T=128`:
  ```
  measured roofline: 3930 GFLOP/s, 50.9 GB/s DRAM
  GDN layer:  55.4 ms/rep, 185 dispatches/rep   (29.7 GFLOP offline)
  GQA layer:   0.2 ms/rep,  15 dispatches/rep   (26.8 GFLOP offline)
  dense MLP:   0.2 ms/rep,   4 dispatches/rep   (68.5 GFLOP offline)
  ```
  **Conclusion: arithmetic is NOT the limiter anywhere in this model on this
  backend - a native device-side FP8 GEMM is explicitly out of scope for
  this pass, per the task's own rule.** GQA and the dense MLP are already
  sub-millisecond and, if anything, dispatch-count-bound at these tiny
  per-call FLOP totals (matmul throughput is nowhere near saturated at 15-4
  dispatches). GDN is ~270x slower than GQA despite having FEWER offline
  GFLOP (29.7 vs 26.8) and only ~12x more dispatches (185 vs 15) - the ratio
  does not track FLOPs OR dispatch count, it tracks the chunked
  recurrence's SEQUENTIAL data dependency: `gdn_chunk_cumsum_step`/
  `gdn_ut_step` fire 126 times each (once per chunk-position, strictly
  ordered), and each such dispatch pays a real host round-trip/launch-
  latency cost this iGPU's driver cannot hide behind other queued work,
  since the next dispatch cannot even be RECORDED correctly until the
  previous one's result is known. This is confirmed by the CPU JIT
  backend's own numbers at the same shape, where cost tracks FLOPs directly
  instead (`BRAIN_DEVICE=cpu`, no per-dispatch launch cost to dominate):
  GDN 1890 ms, GQA 1239 ms, dense MLP 3203 ms (slowest, matching its
  largest FLOP count) - a completely different ranking, the tell that the
  GPU numbers above are latency-bound, not compute-bound.
  If future performance work continues on this model, the higher-leverage
  target the profile actually points at is REDUCING GDN's per-chunk
  dispatch count (fusing consecutive sequential steps in `model::gdn::
  gdn_chunk_fwd`) - a separate, larger climb than this pass's own scope,
  and one that would also benefit `qwen35moe` (the same shared chunk
  recurrence, M1/M12).
  **Noted but not chased down** (out of this pass's scope): the per-kernel
  DEVICE TIMESTAMP breakdown (`gpu.kernel_times()`) reported all-zero on the
  first two `report()` calls in one process and then, on a later run at
  `T=512`, returned obviously-corrupted values (~1.5e15 ms) for the third
  call in sequence - some cross-call state issue in this Vulkan backend's
  timestamp query handling, not something this pass's own wall-clock +
  dispatch-count evidence (which was clean and consistent both times)
  depends on. Recorded here rather than silently worked around.

- [x] M14: int8 (DP4A) weight tier. Unlike qwen35moe's own `q8.rs` (a
  bespoke, model-owned quantizer), this model dispatches every one of its 12
  per-layer mixer/MLP linears (5 GDN `in_proj_{qkv,z,b,a}`/`out_proj`, 4 GQA
  `{q,k,v,o}_proj`, 3 MLP `gate`/`up`/`down`) through the shared
  `model::ops::{Ops, Weight}` façade - see `crate::model::is_i8_linear`'s own
  doc for the exact leaf-name list, and the module doc's "int8 (DP4A)
  inference tier" note for the kernel wiring
  (`max_abs_row`/`quant_pack`/`matmul_i8_dyn`, appended to what is now
  `crate::model::pipelines()` - a `OnceLock`-cached function, replacing the
  old `PIPELINES` const so the int8 façade kernels and the bf16/f16 dtype
  variants `Ops::REQUIRED_KERNELS` demands can be appended without a second
  hand-maintained list). `Qwen35::new_i8`/`Qwen35::new_on_i8` build the
  quantized instance (inference-only - asserted mutually exclusive with
  `new_train_on`'s LoRA/full-finetune path); `caps.rs` gained a `precision`
  manifest param (`fp32`/`int8`) wired into the `qwen35 generate` action's
  `Hot` cache key, so switching precision rebuilds the resident model. The KV
  cache itself stays fp32 always (no int8 KV path, matching qwen35moe's own
  documented scope note).
  **Verification, exact test names and measured numbers:**
  - `cargo test -p brain-qwen35 --lib --bins --tests` (`--test-threads=8`):
    18 test binaries, all green, 0 failed (including the two new ones
    below) - `int8_real_weight_sanity.rs`'s 2 tests self-skip (`ignored`)
    without `BRAIN_QWEN35_DIR`, `real_weight_streaming.rs`'s pre-existing 4
    likewise.
  - `crates/qwen35/tests/model_i8_smoke.rs` (fresh tiny-init weights, no
    real checkpoint needed) - `--nocapture` output:
    `tiny_cfg_clears_the_int8_packing_bar` ok;
    `int8_model_excludes_quantized_names_from_the_fp32_param_store` ok;
    `int8_forward_tracks_fp32_within_quant_tolerance_default_backend`
    (Intel Arc iGPU, Vulkan): cosine=0.999999999 rel_l2=0.000033401;
    `int8_forward_matches_fp32_almost_exactly_on_cpu_backend_full_demotion`
    (CPU JIT - `Weight::upload` demotes int8 requests it can't execute back
    to fp32, so this is a same-arithmetic sanity check, not a quantization
    check): cosine=1.000000000 rel_l2=0.000000000;
    `int8_forward_covers_the_mtp_head_when_mtp_is_enabled`: cosine=0.999999999.
  - `crates/qwen35/tests/int8_real_weight_sanity.rs`, run against the REAL
    `Qwen/Qwen3.8-27B-FP8` checkpoint (`BRAIN_QWEN35_DIR=/data/workspace/
    resources/qwen3.8 cargo test -p brain-qwen35 --test
    int8_real_weight_sanity -- --ignored --nocapture`): both
    `layer_0_gated_delta_net_int8_tracks_fp32_on_real_weights` and
    `layer_3_gated_gqa_int8_tracks_fp32_on_real_weights` pass - every one of
    the 15 real-weight leaves checked (8 in layer 0's GDN mixer + MLP, 7 in
    layer 3's GQA mixer + MLP) scores cosine in [0.999928306, 0.999951862]
    against the fp32 tier on the SAME real dequantized weight values (not a
    full-model forward - see that file's own doc for why a real 27B forward
    still can't be built on this box). Peak RSS stayed well under the 16 GB
    self-imposed ceiling (one layer streamed at a time, same discipline as
    M10's `real_weight_streaming.rs`).
  - Full regression sweep, all green: `cargo test -p brain-gradcheck --lib`
    (51 tests, including `qwen35_analytic_grads_match_finite_differences`,
    `qwen35_lora_analytic_grads_match_finite_differences`,
    `qwen35_mtp_analytic_grads_match_finite_differences`,
    `qwen35_a_log_elementwise_grads_match_finite_differences` - the int8
    wiring only touches the forward-only `new_i8`/`new_on_i8` paths, but the
    shared `ops_linear` dispatch point sits inline in the fp32/LoRA forward
    too, so these confirm backward is unaffected); `cargo test -p
    brain-gradcheck` (full integration suite, 20 test binaries); `cargo test
    -p brain-model` (35 test binaries, including
    `gdn_mixer_equivalence.rs`'s 2 hoisted-mixer cross-pipeline tests);
    `cargo clippy -p brain-qwen35 -p brain-qwen35moe -p brain-gradcheck -p
    brain-model --all-targets -- -D warnings` exits 0, 0 warnings; `make
    clippy` (whole-workspace ratchet gate) exits 0 at its 0-warning
    baseline; `make build` (dev profile, whole workspace) succeeds; `make
    gradcheck` succeeds (20 test binaries, 0 failed).
  Found and fixed one real bug while doing this: `crates/cli/src/
  qwen35_cli.rs`'s `infer` subcommand still imported the old `qwen35::
  model::PIPELINES` const directly - the `pipelines()` function this
  milestone introduced (a `OnceLock`-cached appender, needed so the int8
  façade kernels and `Ops::REQUIRED_KERNELS`'s dtype-variant kernels have
  somewhere to be added without a second hand-maintained kernel-index list)
  broke that one caller. Fixed by switching the CLI to `pipelines()` too;
  caught by the whole-workspace `make clippy` gate, not by the
  crate-scoped one (a scoped clippy/test run cannot see a caller living in
  a different crate).
  **Not done, left for a future pass**: no int8 KV cache; no int8 tier
  wired into `crate::serve::Engine` (the paged serving path still always
  builds fp32); the CLI's `qwen35 infer` subcommand (as opposed to the
  `qwen35 generate` capability action) has no `--precision` flag of its
  own, so it never builds an int8 model directly, only via the caps path.

## Not yet done

Nothing - all milestones (M0-M14) are complete. Remaining scope is the
recorded gaps below, none of which are achievable on this development
machine (no discrete GPU, 18 GiB usable RAM), plus M14's own "not done"
items just above.

## Recorded gaps (this development machine has no discrete GPU and 18 GiB usable RAM)

- No whole-model 27B forward, no whole-model torch reference, no e2e generation or
  perplexity number on real weights - unreachable at 27B vs 18 GiB with no
  discrete GPU. Rungs 4-5 of the parity ladder are out of reach here.
- No multi-GPU shard parity (`discrete_gpu_count() == 0` self-skips it) - and note
  qwen35moe's own `shard_parity.rs` does not run on this machine either, so any
  claim it protects a refactor here is a claim about a different machine.
- No serving throughput/latency or residency measurement on real weights.
- MTP head: structurally implemented, **no reference oracle** (see above) -
  gradchecked and overfit-tested, never parity-claimed.
- Vision + decoder fused end-to-end on real weights is not runnable (needs both
  towers resident simultaneously).
- Vision tower's OWN standalone real-weight parity (M10's original plan
  bullet 2, `outside.safetensors`'s `model.visual.*`, ~0.9 GB) was not done -
  M10 landed the two decoder-mixer types (both now cosine=1.0 against the
  real reference) plus embed/lm_head; the vision tower still only has M9's
  real-DIMS/random-weight parity, not real-weight parity.
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
