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

- [x] M15: sliding-window streaming forward (`crates/qwen35/src/stream.rs`) -
  the piece that lets a real 64-layer/27B forward run on THIS box at all: a
  new `stream::run`/`StreamState` drives one forward pass over every real
  decoder layer holding only a small `weightset::WeightSet`-scheduled window
  of layers' weights resident at once, re-reading the rest from disk
  (`crate::import::import_layer`, unchanged, M10) as the schedule advances -
  `Qwen35::new_*` all still need every layer resolved in one host `HashMap`
  first, which M14's own doc already flagged as impossible at 27B on this
  box (~27 GB even fully int8, far past available RAM). Deliberately
  narrow scope: proves the streaming PLUMBING is numerically correct and
  memory-bounded - no CLI/serve wiring, no tokenizer/generation loop, no MTP
  (left to M16/M17).
  **Design, with the real numbers behind each choice:**
  - Downloaded the 61 previously-missing `layers-*.safetensors` shards from
    `Qwen/Qwen3.8-27B-FP8` (`hf download ... --include "layers-*.safetensors"
    --local-dir`, resumed the existing partial local-dir download rather than
    re-fetching anything already present) - all 64 `layers-{0..63}.
    safetensors` now present and non-empty under `BRAIN_QWEN35_DIR`, ~29 GB
    total on disk (24 GB of layer shards + the pre-existing `outside.
    safetensors`/`mtp.safetensors`), 284 GB free disk remaining afterward -
    the download never came close to the disk-pressure abort condition.
  - `weightset::WeightSet` is used purely for hit/miss + slot-index
    bookkeeping (`WeightSet::advance`), never as a fixed-shape buffer pool -
    unlike `crates/s3dit/src/dev.rs`'s `WindowedPhase` precedent (homogeneous
    block shapes, safe to overwrite a slot's buffers in place), this model's
    layers are NOT homogeneous (`full_attention_interval=4`: 3-in-4 layers
    are Gated DeltaNet with 5 quantizable leaves, 1-in-4 are GQA with 4). Each
    slot instead holds an `Option<OwnedStreamedLayer>` (an enum over
    `OwnedGdnLayer`/`OwnedGqaLayer`) that is DROPPED and rebuilt fresh on
    every miss, mirroring `qwen3omnimoe::generate::run_layers`'s drop/rebuild
    discipline (including its two hard-won lessons: force-reclaim device
    memory via one throwaway `gpu.read(_, 1)` right after dropping an evicted
    slot and before the next slot uploads; never let a layer's output borrow
    that layer's own weight buffers).
  - Per-layer int8 (DP4A) footprint, computed directly from
    `Qwen35Config::qwen38_27b()`'s real dims (packed-int8 bytes + per-row f32
    scale, summed over all 12 quantizable leaves - 4 GQA + 3 MLP for a Full
    layer, 5 GDN + 3 MLP for a Linear layer): **372.48 MB/GQA layer, 383.47
    MB/GDN layer** (max ≈ 384 MB/layer, matching the ≈380 MB/layer this
    milestone's own plan estimated) - the non-quantized fp32 aux tensors
    (norms, GDN's `A_log`/`dt_bias`/gated-norm weight, GQA's `q_norm`/
    `k_norm`) add well under 1 MB/layer, negligible against that. Window
    budget: **4 slots** (`weightset::Schedule::cyclic(64, 1)` +
    `weightset::CyclicScan { lookahead: 1 }`, so 3 slots pin permanently for
    this single pass and 1 rotates) → ≈1.53 GB worst-case device-resident
    weight bytes, comfortably inside the ~11 GiB available RAM this iGPU's
    "device" buffers share with the host. Measured (not projected) peak RSS
    for the full 64-layer chain: **1.93 GiB** - well under both the 4-slot
    estimate above and this milestone's own 6 GiB test-asserted ceiling.
  - Initial residual: a small fixed-seed synthetic `[n, d_model]` vector
    (`stream::seed_residual`, `data::rng::Lcg`), not the real embedding
    table - `embed_tokens`/`lm_head` are each `[248320, 5120]`, **5.09 GB
    (4.74 GiB) dequantized to f32**, confirmed against the real checkpoint's
    own tensor shapes; materializing even one would burn most of this box's
    available RAM for a value this milestone's gate does not need (real
    embedding/lm_head/tokenizer is M16's job). A real chain of real layer
    WEIGHTS still transforms this input exactly as it would a real embedded
    token row - what this milestone's gates actually check.
  **Verification, exact test names and measured numbers:**
  - `crates/qwen35/tests/streaming_forward.rs`, run against the real
    checkpoint (`BRAIN_QWEN35_DIR=/data/workspace/resources/qwen3.8 cargo
    test -p brain-qwen35 --test streaming_forward -- --ignored --nocapture
    --test-threads=1`), all 4 pass in 4488.19 s total (dominated by the full
    64-layer chain - real per-layer disk streaming + host FP8 dequant + int8
    CPU-side quantization 64 times over, not a throughput-tuned path, and
    this run also shared the box with other concurrent processes):
    - `layer_0_streamed_matches_the_real_reference` (Gated DeltaNet):
      cosine=0.999104643 rel_l2=0.043262 max_abs=2.7555
    - `layer_3_streamed_matches_the_real_reference` (GQA, first full-attn
      layer): cosine=0.998473474 rel_l2=0.055302 max_abs=0.9961
    - `layer_63_streamed_matches_the_real_reference` (GQA, last full-attn
      layer): cosine=0.999086663 rel_l2=0.043479 max_abs=6.2053
    - all three clear the test's 0.99 floor (streaming introduces no new
      error source of its own vs. `real_weight_streaming.rs`'s already-proven
      fp32 path - same `import_layer`, same mixer math - only int8's
      already-measured one, M14's own per-leaf cosine range
      [0.9999283, 0.9999519] propagated through a whole layer's chain)
    - `full_chain_streams_all_64_real_layers_within_budget`: all 64 real
      layers streamed, output finite, **peak RSS 1.93 GiB** (budget 6 GiB) -
      a bounded-correctness-and-memory smoke gate only, no whole-model
      reference exists on any machine this workspace has access to.
  - `cargo test -p brain-qwen35 --lib --bins --tests -- --test-threads=8`:
    19 test binaries, all green, 0 failed (unchanged from M14 plus the new
    `streaming_forward.rs`'s 4 tests, which self-skip as `ignored` here).
  - `cargo clippy -p brain-qwen35 --all-targets -- -D warnings` exits 0;
    `make clippy` (whole-workspace ratchet gate) exits 0 at its 0-warning
    baseline; `make build` (dev profile, whole workspace) succeeds; `make
    gradcheck` succeeds (`cargo test -p brain-gradcheck`, 20 test binaries,
    0 failed - `stream.rs` only touches inference-only forward paths, but
    this confirms the shared `model::gdn_mixer`/`model::gqa_mixer`/`model::
    ops` machinery it reuses is unaffected).
  **Not done, left for later milestones (explicitly out of this one's
  scope)**: no CLI/serve wiring; no tokenizer or real generation loop; MTP
  untouched; no multi-token decode loop (`Schedule::cyclic(64, 1)` is a
  single pass); no real embedding/lm_head (M16).

### M15 perf follow-up: profiling the 75-minute streaming pass

M15's own full-chain test measured 4488.19 s (~74.8 min) wall-clock for the
whole 64-layer streamed pass, as a single end-to-end number with no stage
breakdown. This follow-up profiled `import_layer`'s two host-side stages in
isolation (`crates/qwen35/src/bin/import_profile.rs`, real `[5120, 17408]`
`mlp.down_proj` FP8 tensor off the real checkpoint,
`/data/workspace/resources/qwen3.8/layers-5.safetensors`, 89.1M elements) and
fixed what the numbers actually showed was slow:

- **Real disk throughput on this box** (re-measured, not trusted from a
  stale session number): `dd if=layers-5.safetensors of=/dev/null bs=4M
  iflag=direct` (O_DIRECT, bypasses page cache) = **1.6 GB/s** (384 MB shard
  in 0.237 s). The stale ~1.3 GB/s figure undersold it slightly.
- **`checkpoint::mmap::MmapSafetensors::tensor_f32`** (raw FP8-byte decode,
  `import_layer`'s own call, inside `crates/checkpoint/src/mmap.rs`): **~5.0
  s for 89.1M elements (~17-20 Melem/s)** - by far the dominant cost, **~90%
  of one tensor's total import time**, not the two functions this task
  guessed at. This is inside `mmap.rs`, out of this fix's scope (owned by
  concurrent M15/M16-adjacent work on that file at profiling time) - flagged
  here as a real, measured finding for a follow-up, not fixed.
- **`model::fp8::dequant_block128`** (before): 1061 ms/call, 84 Melem/s.
  **After** (fix below): 158 ms/call, 563 Melem/s - **6.7x**.
- **`model::int8::quantize_weight`** (before): 1292 ms/call, 69 Melem/s.
  **After** (fix below): 355 ms/call, 251 Melem/s - **3.6x**.

**Fix** (`crates/model/src/fp8.rs`, `crates/model/src/int8.rs`, native/non-
wasm32 builds only - wasm32 keeps the identical-arithmetic sequential loop):
- `dequant_block128`'s inner loop recomputed the column-block index
  (`c / block`, an integer division) once per ELEMENT (17408 times/row for
  the real `down_proj` shape) instead of once per BLOCK (136 times/row) -
  restructured to walk column blocks and multiply a contiguous slice by one
  scalar per block, which both removes 128x the divisions and gives LLVM a
  loop shape it auto-vectorizes on its own (no hand SIMD intrinsics needed).
  Rows are independent writes, so they also fan out across
  `backend_cpu::par::rows_mut` (the workspace's one shared rayon pool -
  `backend-cpu/src/par.rs`'s own doc: "rayon lives in exactly one crate").
- `quantize_weight`'s per-row absmax + int8 pack is likewise row-independent
  (own scale, own output words) - fanned out via `backend_cpu::par::map`,
  matching the exact `Vec<(Vec<u32>, T)>`-then-reassemble shape
  `crates/kronos/src/generate.rs` already uses for the same "independent
  row, two owned outputs" case.
- Both changes preserve the EXACT same per-element floating-point order
  (only the block-index arithmetic and the row schedule changed, never the
  multiply/round/clamp itself) - `cargo test -p brain-model` (all 143 lib
  tests + every integration suite) stayed green throughout, including the
  exact-equality dequant tests and the sign/round-trip int8 tests.

**Validation:**
- Isolated before/after (`import_profile`, 5 reps): see the two functions'
  numbers above.
- Partial-layer re-run (`crates/qwen35/src/bin/import_layer_bench.rs`,
  real layers 0-5 via the real `import_layer` call `stream.rs` itself
  makes): 128.664 s / 6 layers = 21.4 s/layer avg (noisy - this box also had
  concurrent unrelated cargo processes competing for disk/CPU during the
  run, one layer spiked to 47 s), extrapolating to **~22.9 min for a full
  64-layer import** vs the prior full-pass 74.8 min. A full 64-layer re-run
  was not repeated (impractical at ~75 min/attempt for a tight edit-measure
  loop) per this task's own instruction to prefer the partial substitute.
  Consistent with the profile: since mmap decode is ~90% of a tensor's
  import cost and is untouched by this fix, the realistic end-to-end
  improvement from THIS fix alone is real but partial (roughly the
  dequant+quantize share of the total, not the dominant 75-minute cost).
- `cargo test -p brain-model` (full): pass, 0 failed.
- `cargo test -p brain-qwen35moe --test model_i8_smoke`: pass, 6/6 (int8
  path spot-check for a crate sharing `model::int8::quantize_weight`).
- `cargo test -p brain-qwen35 --lib import::`: pass, 11/11 (import.rs itself
  is unmodified, but exercises `dequant_block128` directly).
- `cargo clippy -p brain-model --all-targets -- -D warnings`: clean.
- `make build` / `make gradcheck`: see this task's own commit for the
  confirmed run.

**Not fixed by this task (explicitly out of scope at the time):**
`checkpoint::mmap::MmapSafetensors`'s raw-byte decode was the real majority
cost (~90% of a big FP8 tensor's import time here) - flagged as the natural
next step for whoever owns `mmap.rs` next. Fixed immediately below, once
`mmap.rs` was free of concurrent M16 work.

### mmap.rs FP8 decode fix: the actual dominant cost

Root cause: `checkpoint::safetensors::e4m3fn_to_f32` (the per-byte E4M3
decode `mmap.rs::decode_into`'s `F8_E4M3` arm calls once per element) computed
its value from scratch every call, including two `f32::powi` calls - a
branchy, non-inlined function invoked 89.1 million times for one real
`down_proj` tensor. But an E4M3 byte has only 256 possible values, so the
whole function is a pure lookup: rewrote it to build a `[f32; 256]` table
once (`std::sync::OnceLock`, same lazy-init pattern this crate's own
`pipelines()` functions already use) from the original scalar formula
(kept, renamed `e4m3fn_to_f32_scalar`, called only 256 times total now) and
index into it thereafter. Bit-for-bit identical output (it's the same
function, just memoized) - `cargo test -p brain-checkpoint` (80 lib + 16
`torchpt` tests, including the exact-value E4M3 edge-case assertions) stayed
green.

**Measured, real checkpoint, same tensor as the profiling task above**
(`import_profile`, `layers-5.safetensors`, `mlp.down_proj`, 89.1M elements):
`MmapSafetensors::tensor_f32` (FP8 decode): **408.04 ms (218.4 Melem/s)**,
down from ~5.0 s (~17-20 Melem/s) - roughly **6-12x**. Full-tensor stage
breakdown after this fix: decode 58.2%, dequant 9.3%, quantize 32.5% (of
700.71 ms total) - decode is no longer the dominant stage at all, though
still the largest single one.

**End-to-end validation** (`import_layer_bench`, real layers 0-5, same real
checkpoint): 17.962 s / 6 layers = 2.994 s/layer avg, extrapolating to
**~3.19 min for a full 64-layer import** - down from the original 74.8 min
(**~23x** total) and from the prior fix's own 22.9 min estimate (**~7x**
further). This directly bounds the real cost of every future real-checkpoint
streaming run in this repo (M15's/M16's own ignored tests, and any future
milestone that streams the real checkpoint) - a decode step that previously
cost ~28-32 min (M16's own measured number) should now cost roughly a few
minutes, though this was not re-measured via a full `generate_streaming`
re-run (impractical to repeat for every fix in a tight loop - the isolated
+ partial-layer numbers above are the validated evidence).

Verification: `cargo test -p brain-checkpoint` (96 tests), `cargo test -p
brain-model` (full), `cargo test -p brain-qwen35 --lib` (30 tests, including
`stream::tests::gdn_end_padding_does_not_change_real_position_outputs`),
`cargo clippy -p brain-checkpoint --all-targets -- -D warnings`, `make
build`, `make gradcheck` (20 suites) - all clean.

### M16: real end-to-end generation (real prompt -> real tokenizer -> streaming engine -> real lm_head -> real sampling -> real text)

Extends M15's synthetic-input streaming forward into real, human-readable
generation: a real prompt, the real `Qwen/Qwen3.8-27B-FP8` tokenizer, real
embedding rows, the same 64-real-layer sliding-window streaming forward
(now driven per decode step over the growing prompt+generated sequence,
not a single synthetic pass), a real resident int8 `lm_head`, and real
sampling (greedy and temperature/top-k/top-p) - producing a transcript a
human can read directly.

**New `checkpoint::mmap::MmapSafetensors` accessor:**
```rust
pub fn tensor_f32_range(&self, name: &str, start_elem: usize, len_elem: usize) -> Option<Vec<f32>>
```
Decodes a flat, row-major element range straight off the mmap (byte-sliced
by the tensor's own dtype width), reusing the same `decode_into` helper
`tensor_f32`/`with_tensor_chunks` already share - never touches bytes
outside the requested range, unlike `tensor_f32` (whole tensor) or
`with_tensor_chunks` (scans from offset 0). `None` on an out-of-bounds
range (no panic), including a `start_elem` near `usize::MAX` (checked add).
Used to pull ONE embedding-table row (`O(d_model)` bytes) per prompt/
generated token, instead of decoding or scanning the whole `[248320, 5120]`
table. Two new tests in `crates/checkpoint/src/mmap.rs`: range reads match
the corresponding slice of a full decode; out-of-bounds (partial overlap,
past-the-end, `usize::MAX`, unknown name) all return `None` cleanly.

**`crates/qwen35/src/stream.rs` additions:**
- `load_layer` split into `load_layer` (disk shard -> host map, via
  `import_layer`, unchanged) + a new private `build_layer` (host map ->
  device `OwnedStreamedLayer`) - so a test can drive the device-upload half
  against synthetic weights (`crate::init::init_weights`) with no
  checkpoint on disk.
- `run`'s per-layer windowed load/evict loop factored into a private
  `stream_all_layers(state, dir, cfg, xres0, n, window_budget) ->
  DeviceBuffer`, shared by `run` (M15's synthetic-input gate, unchanged
  behavior) and the new `generate` (below).
- `pub fn generate(dir: &Path, cfg: &Qwen35Config, tokenizer_path: &Path, prompt: &str, max_new: usize, temperature: f32, top_k: usize, top_p: f32, window_budget: u32, seed: u64) -> Result<String, String>`:
  tokenizes `prompt` (`data::qwen_tokenizer::QwenBpe`, mirrors
  `crate::caps::GenerateAction::run`'s own tokenizer-present path and its
  `<|im_end|>`/`<|endoftext|>` EOS fallback); each decode step re-embeds
  the WHOLE growing (prompt + generated-so-far) sequence via real
  `tensor_f32_range` rows off `outside.safetensors`'s
  `model.language_model.embed_tokens.weight`, end-pads to the next
  multiple of 64 (`GDN_DECODE_CHUNK` - the reference GDN chunk size) with
  dummy token id `0` (design decision 2: `model::gdn`'s chunked recurrence
  asserts `t % chunk == 0`), runs `stream_all_layers` over the padded
  sequence, reads back ONLY the last real position's hidden state, applies
  the model's final `norm.weight` RMSNorm (read + `(1+w)`-folded via
  `crate::import::fold_plain_rmsnorm_weights`, reusing the exact same fold
  every other plain-RMSNorm weight in this crate gets) and a resident int8
  `lm_head` (quantized ONCE before the decode loop, kept resident for the
  whole call - never re-quantized per step), and samples via the now-
  `pub(crate)` `crate::sample::sample_logits`/`argmax`. No persistent
  incremental KV/GDN state between decode steps (design decision 1) - every
  step is a full non-incremental forward over the whole sequence so far,
  since every step already re-streams every layer's weights from disk
  regardless (the dominant, unavoidable per-step cost).
- `quantize_i8_from_mmap_rows`: quantizes `lm_head.weight`/
  `embed_tokens.weight` (`[248320, 5120]`, plain BF16, ~4.74 GiB
  dequantized to f32) to int8 (~1.18 GiB packed) directly from the mmap in
  row-chunks (`model::int8::quantize_weight`'s scale is per-ROW, so
  chunking never crosses a scale boundary), writing straight into a
  pre-sized device buffer via `Gpu::write_at`/`write_f32_at` - **never**
  holding the whole dequantized `[n, k]` f32 array in host RAM at once.
  Chosen over the simpler one-shot `Weight::upload` after checking `free
  -h` on this shared box at the time: only ~8.3 GiB "available" RAM (~22
  GiB already in use by other concurrent sessions), too close to the 4.74
  GiB one-shot peak for comfort.
- New test `gdn_end_padding_does_not_change_real_position_outputs`
  (fast, CPU backend, NOT `#[ignore]`d - no real checkpoint needed):
  directly proves the causality assumption design decision 2 depends on,
  against the REAL code path (`StreamState::build_layer` +
  `layer_forward`, not a hand-rolled parallel replay) - two residual
  streams that agree on the first 5 rows but differ arbitrarily in a
  64-row padded tail produce IDENTICAL layer output on those first 5 rows
  (max abs diff < 1e-5), while the padded tail itself genuinely differs
  (so the test could actually catch a leak). **Passed** - confirms GDN's
  intra-chunk strict-lower-triangular masking and the causal
  (left-padded, `pad: kw-1`) depthwise conv1d really do keep end-padding
  from leaking backward into real positions, matching what `model::gdn`'s
  and `model::gdn_mixer`'s own module docs already claimed.

**`crates/qwen35/src/sample.rs`:** `argmax`/`sample_logits` bumped from
private to `pub(crate)` - reused directly by `stream::generate` instead of
duplicated.

**`crates/qwen35/src/caps.rs`:** new boolean param `streaming` (default
`false`) on the existing `generate` action. `streaming=true` routes
`GenerateAction::run` through `crate::stream::generate` (new
`GenerateAction::run_streaming`) instead of building a resident `Qwen35`;
`weights` is then the CHECKPOINT DIRECTORY (not a single `.safetensors`
file), a tokenizer is required, and `messages`/`chat`/`tools`/`stop` are
not supported (the streaming engine takes a plain prompt string, not a
chat request - a recorded scope gap, not an oversight). `window_budget` is
fixed at 4 (M15's own measured-safe value), config fixed at
`Qwen35Config::qwen38_27b`. The `streaming=false` (default) path is
byte-for-byte unchanged. No "hot" resident-model cache for the streaming
path - every call re-streams every layer regardless, so there is nothing
to keep warm.

**Real generated transcripts** (`crates/qwen35/tests/generate_streaming.rs`,
`#[ignore]`d, `BRAIN_QWEN35_DIR`-gated, run with `BRAIN_QWEN35_DIR=
/data/workspace/resources/qwen3.8 cargo test -p brain-qwen35 --test
generate_streaming -- --ignored --nocapture --test-threads=1`), prompt
`"The capital of France is"`, `max_new=2`, `window_budget=4`:

- **GREEDY** (`temperature=0`): output `" Paris."` - 64.1 min total, 32.0
  min/decode step (2 steps).
- **SAMPLED** (`temperature=0.8, top_k=40, top_p=0.9, seed=42`): output
  `" Paris."` - 55.8 min total, 27.9 min/decode step (2 steps). Landed on
  the SAME text as greedy - not required (the two settings' outputs are
  never asserted equal), but unsurprising: "Paris" is evidently a very
  high-confidence next token here, so top-p=0.9/top-k=40 sampling still
  concentrates on it most of the time.
- Both real, coherent, grammatically correct continuations of the prompt -
  the actual proof this milestone exists to produce, not a synthetic
  smoke value.
- Whole test: 7193.28 s (~119.9 min) for both settings, 4 decode-forward
  passes total. Per-step cost (~28-32 min) is noticeably faster than M15's
  own 74.8-min full-pass measurement - consistent with the M15 perf
  follow-up's `dequant_block128`/`quantize_weight` speedups (see that
  section above) having already landed before this milestone ran, plus
  warm page-cache reads across the two settings' repeated layer-shard
  access in one process.
- Peak RSS across the whole run (both settings, one process): **2.74
  GiB** (`brain_testutil::mem`) - well inside this box's available RAM,
  consistent with M15's own per-layer streaming footprint plus the ~1.18
  GiB resident int8 `lm_head`.

**Verification:**
- `cargo test -p brain-checkpoint` (new `tensor_f32_range` tests included):
  96/96 pass (80 lib + 16 torchpt).
- `cargo test -p brain-qwen35` (non-ignored, includes the new causality
  test): all pass, 0 failed.
- `cargo test -p brain-qwen35 --test generate_streaming -- --ignored
  --nocapture --test-threads=1` against the real checkpoint: pass (see
  transcripts above).
- `cargo clippy -p brain-qwen35 -p brain-checkpoint --all-targets -- -D
  warnings`: clean.
- `make build` (whole workspace, dev profile): succeeds.
- `make gradcheck`: succeeds (20 test binaries, 0 failed) - this milestone
  only touches inference-only forward paths, but confirms the shared
  `model::gdn`/`model::gdn_mixer`/`model::gqa_mixer`/`model::ops`/
  `model::int8` machinery it reuses is unaffected.

**Not done, left for later milestones (explicitly out of this one's
scope):** throughput tuning (the ~30-min/decode-step cost is unoptimized
by design, deferred to a later milestone that gates the residency policy
by real `brain-perf` measurement); persistent incremental KV/GDN decode
state across steps (design decision 1 - every step re-runs a full forward
over the whole growing sequence instead); `docs/models/qwen35.md` (M20);
`messages`/`chat`/tools support in the streaming `caps.rs` path; any
transcript longer than a couple of tokens (impractical at this per-step
cost - not attempted, and not a bug).

## Not yet done

Nothing - all milestones (M0-M16) are complete. Remaining scope is the
recorded gaps below, none of which are achievable on this development
machine (no discrete GPU, 18 GiB usable RAM), plus M14's/M15's/M16's own
"not done" items just above.

## Recorded gaps (this development machine has no discrete GPU and 18 GiB usable RAM)

- No whole-model torch reference at 27B (no perplexity number on real
  weights, no rung 4-5 parity ladder comparison) - unreachable with no
  discrete GPU regardless of streaming. M16's `stream::generate` DOES now
  produce real, human-readable end-to-end generation (real prompt -> real
  tokenizer -> real embeddings -> all 64 real streamed layers -> real
  lm_head -> real sampling -> real text - see that milestone's own section
  above for verbatim transcripts), closing the "no e2e generation" half of
  this gap; what remains unreachable here is a NUMBER to compare against
  (perplexity, or a token-by-token match against a real HF `generate()`
  run) and anything past a couple of tokens (~30 min/decode step at this
  milestone's own measured, unoptimized-by-design streaming cost makes a
  longer transcript impractical on this box, not merely undesirable).
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
