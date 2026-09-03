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
    `Qwen/Qwen3.8-27B-FP8` checkpoint (`BRAIN_QWEN35_DIR=[path/to/qwen3.8]
    cargo test -p brain-qwen35 --test
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
    checkpoint (`BRAIN_QWEN35_DIR=[path/to/qwen3.8] cargo
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
`[path/to/qwen3.8]/layers-5.safetensors`, 89.1M elements) and
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
[path/to/qwen3.8] cargo test -p brain-qwen35 --test
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

### M17: MTP-accelerated greedy streaming decode ("confirm, advance, speculate")

The key insight: since M16's streaming decode re-pays the SAME fixed
per-pass weight-streaming cost (all 64 real layers, from disk, every decode
step) regardless of how many token positions that pass computes output
for, running the MTP head (`crate::model::Qwen35::run_mtp_forward`,
training-only until this landed, `mtp.*` real weights never imported
before this either) as one extra, CHEAP, resident-weights computation
within the SAME pass that produces the main next-token prediction lets one
streaming pass yield genuine progress on TWO tokens instead of one - an
I/O-bound-regime-specific win (amortizing the fixed per-pass cost over more
confirmed tokens), not `qwen3::serve::spec_decode`'s FLOPs-bound kind.

**Phase 1 - real-weight MTP import** (`crates/qwen35/src/import.rs`):
`classify()`'s per-layer leaf-rename table hoisted into a new
`classify_layer_leaf(leaf, ty)` (returns a bare suffix, no `blocks.{l}.`
prefix), so the new `pub fn import_mtp(reader, cfg, block) ->
Result<HashMap<String, Vec<f32>>, String>` can reuse it for
`mtp.layers.0.*` (architecturally a plain `Full`-layer leaf set) instead of
hand-duplicating the rename rules. `classify()` itself had always
deliberately dropped every `mtp.*` tensor (this module's own doc, "a
DELIBERATE out-of-scope drop") - no real-weight MTP import existed
anywhere in this crate before this milestone, despite `mtp.safetensors`
being downloaded and the MTP architecture being gradchecked against
synthetic weights since M7.
- Splits the real checkpoint's ONE fused `mtp.fc.weight [d, 2d]` into
  `mtp.fc_e.weight [d,d]` (columns `[0,d)`) and `mtp.fc_h.weight [d,d]`
  (columns `[d,2d)`) - exactly the order `config.rs`'s own `param_list()`
  doc comment already committed to. **No external oracle exists to check
  this column order against** (confirmed again this session: the installed
  `transformers.models.qwen3_5` loader ignores every `mtp.*` key on load,
  `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`) - this reuses the
  ONE convention the already-gradchecked forward/backward math assumes
  rather than inventing an unverifiable second one. **Validated structurally,
  not numerically**: on the real checkpoint, `fc_e`/`fc_h` are genuinely
  different sub-tensors (not an aliasing bug), both finite, both the
  expected `[5120, 5120]` shape - that is the full extent of what is
  checkable here.
- `fold_plain_rmsnorm_weights` extended (judgment call, documented in its
  own doc comment) to fold MTP's own three norms
  (`mtp.pre_fc_norm_embedding.weight`/`mtp.pre_fc_norm_hidden.weight`/
  `mtp.norm.weight`) by EXACT name, matched by analogy with every other
  plain (non-gated) norm in this checkpoint - MTP has no gated norm
  anywhere in its own architecture (its only mixer is GQA-shaped, never
  Gated DeltaNet), so the gated-norm reparameterization that motivates the
  ONE exception (`linear_attn.norm`) never applies here.
- Two-way coverage validated against `Qwen35Config { mtp: true, ..cfg
  }.param_list()`'s own `mtp.*` subset, mirroring `import_dir`'s own
  discipline; every real tensor either classified onto an expected name or
  a loud, by-name error (missing `mtp.fc.weight`, unrecognized leaf, or any
  tensor left over after classification).
- New tests: `import_mtp_splits_fc_folds_norms_and_matches_full_two_way_coverage`
  + `import_mtp_missing_fc_weight_errors_by_name` (fast, synthetic fixture,
  real on-disk safetensors via `checkpoint::mmap::MmapSafetensors`, no real
  checkpoint needed) plus a new `#[ignore]`d, `BRAIN_QWEN35_DIR`-gated
  `crates/qwen35/tests/import_mtp_real_weight.rs`.
  **Real-checkpoint result** (`mtp.safetensors`, `BRAIN_QWEN35_DIR=
  [path/to/qwen3.8] cargo test -p brain-qwen35 --test
  import_mtp_real_weight -- --ignored --nocapture`): all 16 expected
  `mtp.*` tensors present with the exact `param_list()` shapes, all finite,
  `fc_e != fc_h` confirmed on real data - **pass**, 19.18 s.

**Phase 2/3 - wiring into `crate::stream`** (`crates/qwen35/src/stream.rs`):
- `OwnedMtpLayer`: reuses `OwnedGqaLayer` directly for `mtp.layers.0.*`'s
  self-attn+MLP leaves (real dims confirmed to match a main-stack `Full`
  layer's exactly), loaded ONCE per `generate()` call via a new
  `StreamState::load_mtp` (the SAME `Weight::upload(..., Dtype::I8)` int8
  path the 64 main layers use for the quantizable leaves; `fc_e`/`fc_h`
  and the three norm vectors stay fp32-resident via the same
  `Weight::upload` façade requesting `Dtype::F32` - genuinely negligible
  memory next to a single main layer's own streamed footprint, not worth
  quantizing).
- New free functions `single_position_mrope`/`mtp_mixer_forward`/
  `mtp_forward` reimplement `run_mtp_forward`'s own math (embed candidate
  token -> two pre-norms -> `fc_e(en)+fc_h(hn)` -> one GQA-shaped decoder
  layer via the SAME shared `model::gqa_mixer::gqa_mixer_fwd` the main-layer
  GQA path already calls -> `mtp.norm` -> the already-resident int8
  `lm_head`) for a SINGLE row (`n=1`) rather than the whole training batch
  `run_mtp_forward` computes over - sound specifically because of the
  decode loop's own causality argument below (a wrong MTP guess never
  corrupts correctness, only wastes one pass's worth of speculation).
- `generate`/`generate_with_stats` gained a `use_mtp: bool` parameter
  (`generate` is now a thin wrapper over `generate_with_stats`, which
  additionally returns the real number of `stream_all_layers` passes
  issued - the instrumentation gate 2 needed). `use_mtp: true` with a
  non-zero `temperature` is a loud `Err`, never a silent fallback: **scoped
  to GREEDY decoding only** - verifying a stochastic MTP draft against a
  stochastic target needs rejection-sampling machinery this milestone
  deliberately does not build (unlike `qwen3::serve::spec_decode`, which
  never needs it either, since it drafts deterministically then verifies
  against the target's OWN greedy choice - but a temperature-sampled
  target makes "the correct next token" ill-defined for a single
  verification check here). `crate::caps`'s `generate` action gained a
  matching `use_mtp` manifest param (default `false`).
- New private `generate_mtp_accelerated`: the "confirm, advance, speculate"
  decode loop. Maintains a confirmed prefix plus at most one PENDING
  (MTP-guessed, unverified) tail token; every pass feeds the streaming
  forward the confirmed history + the one pending tail (end-padded, same
  as M16), then reads main-head logits at the last-confirmed position
  (always - the model's own true prediction, independent of whatever
  `pending` guessed) and, ONLY on a match, a SECOND time at the
  now-confirmed pending position too - free, since causal attention
  guarantees that row's output is identical to what a genuine serial
  continuation would have produced there (its own input really was the
  now-known-correct token). A fresh pending guess for the next round comes
  from one `mtp_forward` call against THIS pass's own already-computed
  hidden state, never an extra pass. On a mismatch, the wrong pending token
  is simply never included in the next pass's input sequence - no
  persistent-KV rollback needed (unlike `qwen3::serve::spec_decode`'s
  `model::paged::truncate`), a deliberate simplification M16's own
  growing-prefix-recompute architecture (no persistent KV/GDN state across
  passes at all) already enables. `head_logits` extracted as a shared
  helper both the plain loop and this loop call, so gate 1's byte-identical
  claim is true by construction (one implementation of "apply the final
  norm + project to vocab logits" for either path to possibly diverge on),
  not just by argument.

**Phase 4 gates** (`crates/qwen35/tests/generate_streaming_mtp.rs`,
`#[ignore]`d, `BRAIN_QWEN35_DIR`-gated, `BRAIN_QWEN35_DIR=
[path/to/qwen3.8] cargo test -p brain-qwen35 --test
generate_streaming_mtp -- --ignored --nocapture --test-threads=1`), same
prompt `"The capital of France is"`, `max_new=4`, `window_budget=4`,
`seed=20260819`, greedy (`temperature=0.0`) both paths:

- **Plain** (`use_mtp=false`): output `" Paris.\nThe"` - **4 passes**,
  79.1 min total.
- **MTP-accelerated** (`use_mtp=true`): output `" Paris.\nThe"` - **3
  passes**, 52.4 min total.
- **Gate 1 (exact-match determinism): PASS** - `assert_eq!(mtp_text,
  plain_text)` held: both paths produced the BYTE-IDENTICAL token
  sequence, confirming the causality argument above holds in the real
  code (the MTP head's own guesses never change WHAT text comes out, only
  how many streaming passes it costs).
- **Gate 2 (real pass-count reduction): PASS** - measured ratio
  **plain/MTP = 4/3 = 1.333x**, not the ~2x upper bound (this checkpoint's
  real MTP head guessed correctly on 1 of the 3 pending predictions it
  made against this real prompt/prefix - "somewhere close to but not
  necessarily exactly 2x", exactly as scoped; the real number, not
  assumed).
- Whole test: 7890.64 s (~131.5 min) for both settings together.
- **Per-pass cost was measured much higher here (~19.8 min/pass plain,
  ~17.5 min/pass MTP-accelerated) than the ~3-4-minute/pass figure the
  M15 perf follow-up's OWN doc flagged as an untested extrapolation** ("not
  re-measured via a full `generate_streaming` re-run" - that section's own
  words). This run's own process accounting (`ps`/`free`/`loadavg`) showed
  heavy concurrent CPU contention for a large share of its wall-clock time
  - this milestone's own `cargo test -p brain-gradcheck --lib` (full suite,
  ~14 min) and `make gradcheck` (release rebuild + full suite, ~20 min)
  ran concurrently on the same shared, multi-session box, competing for
  the same CPU cores the per-layer FP8/int8 host-side dequant work needs.
  Recorded here as the likely cause, not confirmed by an isolated
  re-measurement (impractical to repeat at this per-run cost) - a gap in
  measurement rigor, not evidence the earlier throughput fixes regressed.

**Verification:**
- `cargo test -p brain-qwen35 --lib --bins --tests`: all green, 0 failed
  (13 new `import::tests::*`/`import_mtp_*` unit tests included).
- `cargo test -p brain-qwen35 --test import_mtp_real_weight -- --ignored
  --nocapture` and `--test generate_streaming_mtp -- --ignored --nocapture
  --test-threads=1` against the real checkpoint: both pass (numbers above).
- `cargo test -p brain-gradcheck --lib qwen35 -- --nocapture` (targeted
  re-confirmation) AND the full `cargo test -p brain-gradcheck --lib` (51
  tests) AND `make gradcheck` (20 suites, release): all green, 0 failed -
  **`qwen35_mtp_analytic_grads_match_finite_differences` explicitly
  re-confirmed passing in all three runs**, proving this milestone's new
  import/streaming-decode code never touched `model.rs`'s shared
  forward/backward math.
- `cargo clippy -p brain-qwen35 --all-targets -- -D warnings`: clean.
- `make build` (whole workspace, dev profile): succeeds.

**Not done, left for later milestones (explicitly out of this one's
scope):** sampled (non-greedy) MTP-accelerated decode (needs
rejection-sampling machinery - see Phase 2/3 above); MTP wired into
`crate::caps`'s manifest for anything beyond the plain boolean pass-through
(no dedicated response field reporting pass counts/speculation hit-rate to
an HTTP caller); an isolated, non-contended re-measurement of real
per-pass cost (see Phase 4's own note above); `docs/models/qwen35.md`
(M20, carried over from M16).

### M18: streaming LoRA fine-tuning - forward AND backward through the real 64-layer checkpoint

M8's LoRA is real but needs every layer's weights resolved in one host
`HashMap` first (`Qwen35::new_train_on`) - impossible at 27B on this box.
M15-M17's streaming forward proves inference can run with only a small
window of layers resident, but never computes a gradient anywhere. This
milestone builds the missing piece: a streaming-aware LoRA trainer
(`crates/qwen35/src/stream_train.rs`) that streams every layer's weights
TWICE per step - once forward, once in reverse for backward (`dx = d_out @
W_frozen` through each frozen layer, no `dW` for the base - only the tiny
LoRA adapters need gradients/Adam moments, and those stay fully resident for
the whole run) - roughly doubling the per-step weight-I/O cost of a
forward-only pass. That is the accepted, deliberate cost model here, not
something this milestone tries to avoid.

**Step 0 - backward through a streamed weight: what's actually there.**
Read `model::ops::Ops` (`matmul`/`matmul_dx`/`matmul_dw`) and
`model::dispatch` in full. Finding: `Ops::matmul_dx`/`Ops::matmul_dw` (the
B10 backward-of-matmul pair) exist and are gradient-checked
(`gradcheck::check_matmul_bf16_weight`), but are deliberately scoped to
`F32`/`BF16` weights only - `Ops::matmul_dx`'s own doc: "F16/I8/Q4
backward-through-the-weight is a real, reachable follow-up, not attempted
here"; `bind_matmul_dx` panics for `I8`/`Q4`. Backward through an
int8-quantized weight is not wired up ANYWHERE in this tree (no `Ops`
method, no `model::dispatch` primitive, no model crate's own hand-dispatch
either - `crates/qwen3/src/model.rs::proj_bwd`'s LoRA branch, the closest
precedent, only ever reads `self.w(wname)`, always fp32 by construction on
that crate's own resident trainer).

Extending that kernel family to int8 would be genuine new kernel work, out
of proportion for this milestone. **Resolution, verified by reading the
code rather than assumed:** `crate::import::import_layer` already
dequantizes every FP8 tensor to f32 in host RAM BEFORE `crate::stream::
StreamState::build_layer` optionally packs it down to int8 for the
(inference-only) streaming forward - the f32 values already exist as a
byproduct of every streamed layer load, at zero extra host compute. This
milestone's own layer loader (`stream_train::build_layer_f32`/
`load_layer_f32`) simply skips that packing step and uploads the SAME
already-dequantized f32 values as `Weight::F32` - forward dispatches through
`Ops::matmul` (which already handles F32 fine, same as int8), and backward's
frozen-base `dx`/the tiny LoRA-internal `A`/`B` matmuls dispatch the SAME
`matmul_dx`/`matmul_dw`/`matmul` kernel NAMES `crate::model::Qwen35::
proj_bwd`'s own LoRA branch already uses (`stream_train::proj_bwd_streamed`
is a direct kernel-dispatch transcription of that method, substituting the
streamed `Weight::F32` buffer and a small resident `LoraStore` for `self.w`/
`self.ps`) - no new kernel, no `Ops` extension, reusing already-proven,
already-gradchecked dispatch. Trade-off: streamed per-layer weights are ~4x
larger resident (fp32 vs the inference path's int8) for the run's whole
lifetime - accepted, since the dominant cost either way is disk I/O, not
device memory (M15's own finding still holds).

**Design.** `StreamTrainer` mirrors `StreamState` (window budget, drop/
rebuild-per-slot discipline, `model::gdn_mixer`/`model::gqa_mixer` for the
mixer math) plus: a resident `LoraStore` (`paramstore::ParamStore` +
`optim::Optim`, sized for ONLY the `.lora_a`/`.lora_b` tensors -
`crate::init::init_lora_only` - reusing the SAME AdamW dispatch graph
`Qwen35::new_train_on` already builds, just much smaller); a resident
`lm_head`/`embed_tokens` table (loaded once, never re-streamed - `Ops::
matmul_dx` needs F32/BF16, so this table cannot be the inference path's
int8 tier either). A forward pass streams all 64 layers ascending, applying
the LoRA delta at each targeted leaf (`stream_train::{gdn,gqa}_layer_forward_lora`,
`mlp_forward_lora` - direct transcriptions of `Qwen35::layer_gdn_fwd`/
`layer_gqa_fwd`/`mlp_fwd`) and caching ONLY the residual stream at each
layer boundary (`xres_cache[l]`, `[n, d_model]`, one buffer per layer - a
few hundred KB to a few MB at a realistic tiny training `n`, well under a
GB for all 64 layers at real `d_model=5120`) - not the larger per-leaf
`x`/`a=x@Aᵀ` activations the milestone's own plan first proposed. Backward
re-streams all 64 layers in REVERSE, and for each one RECOMPUTES its own
forward internals (mixer/MLP activations) from the cached `xres_cache[l]`
plus the freshly re-streamed (frozen, deterministic) weight, rather than
caching those larger internals across the whole run - standard activation-
checkpointing, justified here specifically because backward already
re-streams every layer's weights regardless (the dominant cost), so
recomputing costs no extra weight I/O, only compute already implicitly
budgeted for. This uses LESS host memory than the plan's own literal
per-leaf-activation proposal while proving the exact same correctness
property; documented as a deliberate, judgment-call deviation, not
silently done.

**Step 1 - tiny-scale equivalence gate (the hard correctness gate).**
`crates/qwen35/tests/stream_train_equivalence.rs`, fast, NOT `#[ignore]`d
(part of the default `cargo test -p brain-qwen35` run), `Qwen35Config::
tiny()` scale (4 layers, 3 GDN + 1 GQA), CPU JIT backend for BOTH trainers
(no cross-backend floating-point-order noise), `window_budget=2` (forces a
real mid-pass eviction in both the forward AND the reverse-order backward
loop, not just the boundary pin), fp32 throughout (this trainer's only
supported tier, so the comparison is exact-not-just-close by construction,
no int8 quantization noise to muddy the proof) - `crate::init::init_weights`
builds ONE init map fed to BOTH the existing resident `Qwen35::new_train_on`
and the new `StreamTrainer::new_synthetic`, guaranteeing byte-identical
starting weights (a real, initially-missed subtlety: `init_weights` and
`init_lora_only` draw from DIFFERENT RNG streams even at the same seed, so
`LoraStore::new` was designed to take the adapter values in directly rather
than reseed a second generator). Two tests:
- `streaming_lora_trainer_matches_the_resident_trainer_exactly`: 3 steps on
  an identical fixed batch, comparing loss AND every `.lora_a`/`.lora_b`
  gradient (before the AdamW step) AND every adapter weight (after it) at
  each step, tolerance 1e-4. **Real measured numbers (`--nocapture`):** loss
  trajectory resident vs. streaming: step 1 `3.432276`/`3.432276`, step 2
  `3.432078`/`3.432078`, step 3 `3.430849`/`3.430849` - identical to 6
  decimal places at every step; every gradient and post-AdamW weight
  comparison passed under 1e-4 (in practice far tighter - the printed
  losses show effectively bit-identical trajectories). **PASS.**
- `streaming_lora_trainer_reduces_loss_on_a_fixed_batch`: an independent,
  narrower gate (real learning signal, not just replay-matching) - 8 steps,
  loss `[3.3819466, 3.3811188, 3.3586473, 3.2662919, 3.060675, 2.936966,
  2.8580084, 2.767636]`, strictly decreasing. **PASS.**

**Step 2 - training dataset via `Qwen/Qwen3-0.6B`.** Generated with the
already-shipped `brain qwen3 infer` CLI against the locally-cached
`Qwen/Qwen3-0.6B` brain checkpoint, PLAIN completion mode (not `--chat`) -
`--chat` was tried first and hit a real, pre-existing, unrelated bug
(`crates/qwen3/src/sample.rs:179`, `scaled[b].partial_cmp(&scaled[a]).unwrap()`
panics on a `None` - a NaN logit somewhere in that crate's chat-template
path; not investigated further, out of this milestone's own crate/scope,
plain-completion mode is unaffected and was used instead). Five prompts,
`temperature=0.7 top-k=40 max-new=90`, one distinct seed per prompt
(1001..1005):
```
brain qwen3 infer --weights <Qwen3-0.6B brain checkpoint> --tokenizer <its tokenizer.json> \
    --prompt "<PROMPT>" --max-new 90 --temp 0.7 --top-k 40 --seed <SEED>
```
Prompts: "Explain in a few sentences why the sky is blue.", "Write a short
paragraph describing a quiet morning in a mountain village.", "Give three
tips for staying focused while studying.", "Describe what a lighthouse
keeper's daily routine might look like.", "Summarize, in your own words,
why bees are important for ecosystems." Output written to
`[path/to/qwen35_finetune]/GENERATION.txt` (full log: settings + prompts +
raw completions, verbatim, human-inspectable) and
`[path/to/qwen35_finetune]/corpus.txt` (just the generated text,
concatenated - the actual training corpus this milestone's Step 3 tokenizes
- 436 words). Both files live at the WORKSPACE root's `resources/` (a
sibling of this repo, matching M15's own `resources/qwen3.8/` convention),
not inside this git repo.

**Step 3 - a real short streamed LoRA fine-tune on the real checkpoint.**
`crates/qwen35/tests/stream_train_real.rs` (`#[ignore]`d, `BRAIN_QWEN35_DIR`-
gated) is the checked-in gate; a new standalone binary,
`crates/qwen35/src/bin/stream_train_step.rs`, was ALSO built and is what
this milestone's own real numbers below came from - the interactive
development environment this landed in kills a background process after
roughly 45-50 minutes of wall-clock regardless of the timeout requested,
shorter than one combined run (BEFORE-gen + one real step already sums to
~54 min, and the whole run is ~106 min) - the standalone binary splits the
identical `StreamTrainer` calls into separate short (<40 min) processes,
checkpointing the tiny LoRA adapter state to a small safetensors file
between them (`--phase before|step|after`, `--adapter-in`/`--adapter-out`).
This is an artifact of THIS session's own process-lifetime constraint, not
a change to what is trained or measured - confirmed by the fact that a
combined attempt run TWICE independently (before being killed both times)
reproduced the IDENTICAL step-1 loss (`2.417521`) and the IDENTICAL BEFORE
transcript both times, and a third, clean run of the BEFORE phase via the
split binary reproduced it a THIRD time.

Real config: `Qwen35Config::qwen38_27b()` (real 64-layer, `d_model=5120`,
`vocab=248320`), LoRA rank 4 / alpha 8 on all 12 targetable leaves,
`window_budget=2`, `n=16` training tokens (the first 16 tokens of Step 2's
`corpus.txt` under the real tokenizer, shifted-by-one next-token targets),
`lr=0.05`. **A real, measured device-memory constraint discovered here, not
assumed:** the resident `lm_head` at fp32 (`248320×5120×4` bytes ≈ 4.74
GiB, one contiguous buffer) exceeds this box's Vulkan/wgpu adapter's real
`max_buffer_size` (2047 MiB - a hard driver/hardware limit, confirmed by an
actual `wgpu::Validation Error` on the first real attempt, not a host-RAM
shortage) - int8 (~1.18 GiB) and even bf16 (~2.37 GiB) both still exceed it
too. Fix: run training on the CPU JIT backend (`Gpu::new_cpu`) instead of
this box's default GPU adapter - it has no per-buffer ceiling (plain host
allocations), and per M13's own profiling this workload's real cost is
dominated by disk I/O + host-side FP8 dequant regardless of which backend
runs the matmuls, so this is a genuinely well-justified choice for this
specific bottleneck, not a downgrade of convenience.

**Real measured numbers** (CPU JIT backend, `stream_train_step`, prompt
`"The capital of France is"`, `max_new=3`, greedy):
- Trainer construction (resident `lm_head`+`norm.weight` load from
  `outside.safetensors`): 6.5-7.6 s.
- **BEFORE** (zero-init adapter - LoRA's own "starts as an exact no-op"
  invariant, so this IS the base model's real behavior): `17.4-18.1 min`
  across three independent runs, output **byte-identical every time**:
  `"The capital of France is" -> " Paris.\n"` - matches M16's/M17's own
  real-checkpoint transcripts for the same prompt.
- **Step 1**: `loss=2.417521`, `35.71 min` (confirmed identical across two
  independent attempts before the split-binary run).
- **Step 2**: `loss=0.071535`, `35.58 min` - a 33x drop from step 1, a real,
  clear, honest "loss decreasing" signal (the milestone's own gate; not a
  target loss VALUE, which this run does not claim to reach any particular
  one of). Only 2 real steps were run (not 3) given the ~36 min/step real
  cost measured live - "a handful of steps... single digits... is
  completely fine and expected" per this milestone's own scope.
- **AFTER** (the just-trained adapter): `17.40 min`, output
  `"The capital of France is" -> "emelemelemel"` - visibly, dramatically
  different from BEFORE's `" Paris."` - the qualitative check this
  milestone's own gate calls for, unambiguously satisfied. Honestly
  reported as-is, not softened: the output is garbled, not a coherent
  improvement - an expected, real consequence of training a rank-4 adapter
  hard (loss 2.42→0.07 in 2 steps) on a 16-token sequence with no relation
  to the "capital of France" prompt at `lr=0.05`, not an infrastructure
  bug. The adapter genuinely, visibly changed the model's behavior, which
  is exactly what this gate checks for - it does not check for the
  fine-tune being well-behaved or production-quality (explicitly out of
  scope: "do not attempt anything resembling a real production fine-tune
  run").
- Total real wall-clock for Step 3 (before + step1 + step2 + after,
  summed across the split runs): **≈106 minutes.**

**Verification:**
- `cargo test -p brain-qwen35 --lib --bins --tests -- --test-threads=8`:
  all green, 0 failed (27 `test result: ok` lines across the lib +
  every integration test binary), including `stream_train_equivalence.rs`'s
  2 new non-ignored tests and `stream_train_real.rs`'s 1 new test
  self-skipping (`ignored`) without `BRAIN_QWEN35_DIR`.
- `cargo test -p brain-gradcheck --lib` (53 tests, full suite): all green, 0
  failed - `qwen35_analytic_grads_match_finite_differences`,
  `qwen35_lora_analytic_grads_match_finite_differences`,
  `qwen35_mtp_analytic_grads_match_finite_differences` all re-confirmed
  unaffected (this milestone builds a new PARALLEL streaming path, `model.
  rs`'s own backward math is untouched).
- `cargo clippy -p brain-qwen35 --all-targets -- -D warnings`: clean.
- `make build` / `make gradcheck`: both succeed.
- Every real-checkpoint number above is measured, not projected; the split-
  binary/single-test discrepancy is a session-environment artifact,
  documented above, not a correctness concern.

**Files touched:** `crates/qwen35/src/stream_train.rs` (new), `crates/
qwen35/src/bin/stream_train_step.rs` (new), `crates/qwen35/src/stream.rs`
(widened several existing private helpers - `get`, `idx`, `kernel_ids`,
`gdn_mixer_ids`, `gqa_mixer_ids`, `pad_to_gdn_chunk`, `embed_rows`,
`read_final_norm`, `GDN_DECODE_CHUNK` - to `pub(crate)` so `stream_train.rs`
can reuse them directly; zero behavior change to any existing inference
path), `crates/qwen35/src/lib.rs` (`pub mod stream_train;`), `crates/
qwen35/tests/stream_train_equivalence.rs` (new), `crates/qwen35/tests/
stream_train_real.rs` (new), `[path/to/qwen35_finetune]/
GENERATION.txt` + `corpus.txt` (new, outside this git repo).

**Not done, left for later milestones (explicitly out of this one's
scope):** no int8 (or bf16) backward-through-the-weight - the streamed
frozen base is always fp32 in this trainer, ~4x the inference path's
device-memory footprint per resident layer; no full-parameter (non-LoRA)
streamed fine-tune (explicitly ruled out by this milestone's own plan - the
frozen base needing no gradient/optimizer state is the whole reason LoRA is
tractable here at all); no incremental/persistent decode state in
`generate_greedy` (same non-incremental, full-resequence-every-step design
`crate::stream::generate` already uses, for the same reason); no int8/bf16
resident head fallback for GPU-backend training (this box's own
`max_buffer_size` made CPU the only working backend for training at real
vocab scale - a different box with a larger single-buffer limit, or a
future multi-buffer-sharded head, could lift this); no
`crates/cli`/`caps.rs` wiring for streaming training (this milestone is the
trainer + its own gates, not a serving/CLI surface).

**Follow-up investigated and closed: can the resident `lm_head` be chunked
so `stream_train`'s GPU-backend path stops hitting `max_buffer_size`?**
Read `Ops::matmul`/`Ops::matmul_dx`/`Ops::matmul_dw` (`crates/model/src/
ops.rs`) and the WGSL they dispatch (`matmul.wgsl`, `matmul_dx.wgsl`) end to
end, and checked every existing "chunk" helper in this tree
(`checkpoint::mmap`/`paramstore::upload`'s `with_tensor_chunks`,
`gpu_core::write_f32_chunked`, `model::gdn`'s recurrence chunking,
`model::vit`'s attention chunking) for a reusable pattern - none of them
split ONE logical tensor across several PHYSICAL device buffers; they all
either stream host bytes into one still-whole device buffer (bounding host
RAM, not device allocation size) or chunk a compute loop, not a buffer.
`build_head_f32_from_mmap` (this milestone's own loader) already does the
former for the `lm_head`'s host-side load - it still ends up as ONE `n*k`
device buffer.

The real blocker: `matmul.wgsl` writes `out[row*p.n + col]` and
`matmul_dx.wgsl` reads `dy[row*p.n + nn]`, where `p.n` is exactly the
dispatch's own output/reduction width - there is no separate row-stride
parameter. Chunking the vocab dimension across, say, 3 sub-`max_buffer_size`
weight buffers (`4.74 GiB / 2047 MiB` ≈ 2.4, so >= 3 chunks of ~83k vocab
rows each) would need each chunk's matmul to write into (forward) or read
from (backward) the correct COLUMN range of one full-width `[n_tokens,
vocab]` logits/`d_logits` buffer - which neither kernel's index arithmetic
supports without a new row-stride parameter (or a column
scatter/gather kernel, which also does not exist - `region_copy.wgsl`
indexes `src`/`dst` with the SAME stride on both sides, so it cannot
reindex a compact per-chunk buffer into a strided super-buffer; `row_scatter.
wgsl` scatters whole ROWS by index, not partial-width columns). That is
real, new `crates/kernels` WGSL work - explicitly out of scope for this
investigation (the task deliberately excluded `crates/kernels` unless read-
only) and disproportionate to what is a pure training-throughput win, not a
correctness gap: the CPU backend already trains this exact model correctly
end to end (this milestone's own real-checkpoint run, above).

This is also not a one-off: `crates/sam1/tests/parity.rs` and `crates/
deepseek2/tests/common/real_lm.rs` hit the identical class of problem (a
resident real-weight buffer bigger than one device allocation can be on
this box's Vulkan adapter) and both resolve it the same way - pin the CPU
backend for the affected pass. That is this engine's established answer to
"one tensor exceeds one device buffer's ceiling," not a workaround specific
to qwen35. No code changed as a result of this investigation (Option B was
correct); `docs/models/qwen35.md`'s "Why training runs on the CPU backend,
not the GPU" section now records the same reasoning.

### M19: measure M15's residency-window choice + wire qwen35 into `crates/perf`

Two genuinely separate pieces, plus a real, measured performance fix found
along the way while investigating why Piece B's one real baseline run cost as
much as it did.

**Piece A - was `stream.rs`'s `CyclicScan{lookahead:1}`/budget-4 choice ever
actually measured against qwen35's real per-layer byte-cost profile?**
`crates/perf/src/scenarios/weights.rs` already benchmarked `CyclicScan`/`Lru`/
`AllResident` for Z-Image-Turbo's 34 uniformly-counted blocks; this milestone
extended it (additively - `Run`/`run` for Z-Image untouched) with a second,
byte-weighted arm: `ByteRun`/`run_qwen35`/`drive_bytes`, driving the SAME real
`weightset::WeightSet` code over qwen35's real 64-layer int8 byte-cost profile
(`Qwen35Config::layer_i8_bytes`, new - GDN 383,467,904 bytes vs GQA
372,482,048 bytes, a real but small ~3% spread, pinned by a dedicated test).
Real measured numbers (`weights-qwen35` scenario, pure host bookkeeping, no
GPU/checkpoint - `brain perf run weights-qwen35`, and the crate's own tests),
8 passes, at every budget tested:

```
budget   cyclic churn   lru count-overhead   lru bytes-overhead
  2         1.000              1.016               1.016
  4         1.000              1.049               1.050
  8         1.000              1.123               1.123
 16         1.000              1.306               1.307
 32         1.000              1.939               1.941
```

`CyclicScan` is exactly optimal (`1.0`, both metrics) at every budget - Bélády
on a known schedule, confirmed rather than assumed, and confirmed that
qwen35's real ~3% per-layer byte heterogeneity does not change the ranking
(count- and byte-weighted overhead agree within 0.2 percentage points
everywhere tested). `Lru`'s real disadvantage GROWS with budget (+4.9% at 4,
+93.9% at 32) - the multi-pass caching benefit `CyclicScan`'s persistent pin
gives scales with how much of the model the window can hold.

**The honest caveat that measurement needed**: `stream_all_layers` builds a
brand-new `WeightSet` on EVERY call (`Schedule::cyclic(64, 1)` - a single,
non-repeating pass), and `generate`'s decode loop re-invokes it fresh per
token (no persistent state across decode steps, `stream.rs`'s own
already-documented design decision 1). A single, non-repeating pass never
revisits any group, so today, at `passes=1`, EVERY policy loads all 64 layers
from disk exactly once - the leaderboard above is real and correctly proves
`CyclicScan` is the right choice for any future design that DOES persist a
window across passes, but it currently buys nothing over `Lru` because there
is no cross-call persistence yet to exploit. Given that, plus M13's own
profiling (GDN's dispatch-latency-bound cost, not weight I/O), there was no
measured case for moving `WINDOW_BUDGET` off 4 - confirmed, not changed.
`stream.rs`'s `LOOKAHEAD` doc and `caps.rs`'s `run_streaming` doc now carry
these real numbers instead of asserting "conservative and safe" unmeasured.

**Piece B - wire qwen35 into `crates/perf` as a `PerfTarget`.** `qwen35`
does not fit the `build_glm`/`ExecutorTarget`/`*Resident` convention every
other LLM target in `crates/cli/src/perf_cli.rs` uses: `crate::
resident_qwen35::Qwen35Resident` (which mirrors `build_glm`'s pattern
exactly) is built on `qwen35::serve::Engine`, which - like every `Qwen35::
new_*` constructor - needs every layer's weights resolved in one host
`HashMap` first, impossible at this config's real 27B size on this box
(~24.4 GB even fully int8, per Piece A's own per-layer byte numbers ×64).
`crate::stream::generate` is the only path that can run the real checkpoint
here at all, and it deliberately has no persistent resident instance in the
first place (a fresh streaming pass every call) - there is nothing for a
`ResidentModel`/`Executor` to keep warm. Since `qwen35::caps::Qwen35Provider`
already exposes this path as a plain `capability::Provider` action
(`generate`, `streaming: true`), `perf::targets::CapabilityTarget` - the
adapter that turns any `Provider` into a benchmarkable target with zero new
benchmark code, previously never wired to a real CLI target - is the correct
fit, not a forced one. New: `build_qwen35_stream`/`qwen35-stream:
<checkpoint-dir>:<tokenizer.json>` in `perf_cli.rs`. Cost: no independent
correctness gate (`CapabilityTarget`'s own documented limitation - one code
path, so a two-run comparison would check an optimisation against itself);
this target's artifacts stay honestly `correctness: not_checked()`.

Given the real ~17-40 min/pass cost, only the cheapest, single-pass-implying
scenario was run for real: `latency` at `concurrency=1`, `--requests 1
--warmup 0 --output 1` (`startup` was checked and ruled out - it is hard-wired
to `qwen3::serve::Engine`/`SynthSpec`, no generalization to a `Provider`
seam without disproportionate rework). ONE real call to `crate::stream::
generate` (`max_new=1`, real tokenizer, real prompt "The capital of France
is", real streamed forward over all 64 real layers, real int8 lm_head, real
greedy sampling) against `BRAIN_QWEN35_DIR`:

```
e2e_ms p50 = 1,086,513 ms  (18.11 minutes, single real pass, this box, this session's load)
```

Baseline recorded at `scripts/gates/qwen35-perf-baselines/
qwen35-stream-latency-cpu22-gpu0.json` via `brain perf gate <candidate>
--baseline <path> --update` (first baseline for this model, matching
`qwen-serving-perf-gate.sh`'s established bootstrap convention).

**A real, measured performance fix found investigating that number.** The
18-minute figure prompted profiling ONE real layer's own load breakdown
(`crates/qwen35/tests/stream_profile.rs`, new, `#[ignore]`d, real checkpoint):
mmap open ~1-5 ms, `import_layer` (host FP8 dequant) ~11-23 s, quantize+GPU
upload ~4-7 s, GPU forward compute ~0.05-1 s - `import_layer` was ~70-83% of
a real layer's ~16-28 s total, utterly dwarfing GPU compute (consistent with,
and explaining, why `qwen35_bench.rs`'s own M13 GPU-only numbers - a few
seconds for all 64 layers combined - are nowhere near the real per-pass
cost). Root cause: `checkpoint::safetensors`/`checkpoint::mmap`'s F8_E4M3
byte decode (`raw.iter().map(|&b| e4m3fn_to_f32(b)).collect()`) was
single-threaded, unlike the already-parallel `model::fp8::dequant_block128`/
`model::int8::quantize_weight` downstream of it - the earlier LUT
optimization (commit `068d2ff53`) made each element's own cost O(1) but never
fanned the WALK over tens of millions of elements out across cores. Fixed:
new `checkpoint::safetensors::decode_e4m3_bytes`, native builds route through
`backend_cpu::par::each_mut` (the same pool/policy `dequant_block128`/
`quantize_weight` already use), wasm32 keeps the sequential loop (no thread
pool there). Validated with an isolated, disk-I/O-free in-memory
micro-benchmark (`crates/checkpoint/tests/e4m3_decode_bench.rs`, new,
`#[ignore]`d, no checkpoint needed): **10.58x speedup** (1909 ms -> 180 ms,
380 MB synthetic buffer, 22 cores), output verified bit-identical (via
`.to_bits()`, not `==`, since E4M3's one reserved NaN encoding makes a plain
float `assert_eq!` spuriously fail on ~1/128 of a random buffer). Re-verified
correct on the REAL checkpoint after the fix: `real_weight_streaming.rs`'s
ignored real-weight-vs-golden tests (layer 0/3/63 + embed/lm_head) still pass
at cosine=1.000000000 unchanged.

**Also checked (validated, no code change needed)**: whether
`crates/backend-cpu`'s AVX2 fast path is actually engaged for the CPU JIT
training path's (`stream_train_step.rs`, forced onto CPU by the fp32
lm_head/`max_buffer_size` constraint) matmul-heavy dispatch, or silently
falling back to scalar. Measured directly (`qwen35_bench mlp 128 3` on
`BRAIN_DEVICE=cpu`, `BRAIN_NO_FASTCONV=1` vs default): 12504 ms/rep -> 4229
ms/rep, a real 2.96x - already active by default on this box (AVX2 present,
`BRAIN_NO_FASTCONV` unset), confirmed rather than assumed.

**Honestly out of scope, not attempted**: a cross-pass persistent weight
cache (letting `generate`'s decode loop share ONE `WeightSet` across steps,
turning Piece A's currently-inert leaderboard into a real per-step saving).
Investigated and NOT built: this box's usable RAM (~18 GiB) is smaller than
the checkpoint's own on-disk footprint (~24-30 GB), so neither host page
cache nor a bounded device-side window can hold the whole model resident
across steps regardless of policy - the dominant real cost above `import_
layer`'s own (now-parallelized) CPU decode work is disk I/O for data that
does not fit in RAM either way, which a caching policy cannot remove, only a
future bigger-RAM box or faster storage can. Restructuring the decode loop's
weight-set lifecycle to attempt it anyway would be a real, separate
engineering effort this milestone's own real-run budget could not afford to
validate properly (each iteration costing a real ~17-40 min pass) without
disproportionate risk to a delicate streaming decode path - left as a
follow-up, not rushed.

**Verification**: `cargo test -p brain-perf` (all green, including the new
`weights::tests::qwen35_*` suite); `cargo test -p brain-qwen35 --lib --bins
--tests` (all green, unchanged in shape) and, separately,
`real_weight_streaming.rs --ignored` against the real checkpoint (all 4
green, cosine=1.0 unchanged - the correctness re-check the E4M3 decode
parallelization needed); `cargo test -p brain-cli` (all green); `cargo
clippy -p brain-perf -p brain-qwen35 -p brain-checkpoint --all-targets -- -D
warnings` (clean); the same command WITH `-p brain-cli` added surfaces only
the 19 already-recorded, pre-existing `crates/ltxv` errors (pulled in
because `brain-cli` depends on it; not touched by, or related to, this
milestone); `make build` and `make gradcheck` both pass.

### M20: GGUF import (`crates/qwen35/src/gguf_import.rs`), including MTP - on a NEW box (2x P40, 48 AVX2 cores, 184 GiB RAM)

A real community GGUF exists for this model: `unsloth/Qwen3.8-27B-GGUF`
(Q8_0, imatrix-calibrated, `general.architecture = "qwen35"`, 866 tensors,
248320-token embedded gpt2-BPE tokenizer + chat template). Unlike the FP8
HF-safetensors route ([`crate::import`]), this importer **imports the MTP
head** - `qwen35moe`'s own GGUF importer drops its MTP block, but this
checkpoint's MTP tensors (`blk.64.*` + its `nextn.*` extras) are right there
in the file, and the streaming driver was extended
(`gguf::import::to_st_into`) to let a bespoke pass (the `nextn.eh_proj.weight`
column split, the one reshape no `Mapped` variant expresses) write into the
SAME output file as the generic per-tensor loop, so MTP costs no second
checkpoint or second coverage contract.

The leaf-name vocabulary (`attn_q.weight`, `ffn_gate.weight`,
`ssm_alpha.weight`, …) was extracted into `gguf::leaf` as a shared `Role`
enum and retrofitted onto `qwen3`/`qwen35moe`'s own GGUF importers in the
same change (zero behavior change - their existing bit-for-bit/coverage
gates stayed green) so this port's leaf table is the third USE of that
vocabulary, not a third re-transcription of it.

**Verified against the real file** (not just the tiny synthetic fixture):
`config_from_gguf` derives every dimension correctly from the real header
(vocab 248320, d=5120, ff=17408, 24/4 heads, head_dim 256, 48 GDN + 16 GQA
layers, `mrope_section=[11,11,10]` recovered from the KV's 4-slot array with
its always-zero 4th entry asserted, not just dropped) and the embedded
tokenizer round-trips a real sentence exactly
(`config_and_tokenizer_extract_from_the_real_checkpoint`, gated on
`BRAIN_QWEN35_GGUF`, real run: 1.13 s, header/tokenizer-only).

**A real, measured norm-fold trap, resolved empirically rather than
assumed**: [`crate::import::fold_plain_rmsnorm_weights`] undoes HF's
zero-init `(1+w)` RMSNorm convention on the safetensors route. Reading real
norm weights directly off the GGUF (`blk.0.attn_norm.weight` in `[0.89,
1.05]`, `output_norm.weight` in `[1.6, 2.0]`) showed llama.cpp's conversion
has ALREADY applied that fold - so `gguf_import.rs` must NOT (and does not)
call it a second time. Had this been assumed instead of measured, the
GGUF route would have silently corrupted every plain norm on the real
checkpoint while passing every synthetic-fixture test (whose norm values are
arbitrary and cannot expose the direction of a fold).

**A full offline `brain import-gguf` conversion of the real file could NOT be
run on this box**: the roadmap's own standing rule two paragraphs below
("never write an intermediate full-precision whole-model file, ~108 GB") is
not theoretical here - this box has 53 GB free (296 GB disk, 244 GB already
used). The synthetic-fixture round-trip
(`import_gguf_covers_the_main_stack_and_the_mtp_head_with_no_norm_fold`,
including a value-level check of the `eh_proj` column split) plus the
real-header config/tokenizer test above are what stands in for it. This
means `Qwen35Importer::loads_directly()` is honestly `false` (the trait
default) for now - the model has NO practical route to a servable checkpoint
yet until a resident reads the Q8_0 bytes DIRECTLY (M21, tracked in "Not yet
done" below); registering the offline converter was still worth doing (one
line in `crates/cli/src/gguf_import.rs`'s table, gated by its own dispatch
tests) since it is the honest, generic answer for any future smaller
quantization tier or a bigger box.

**A real, pre-existing defect this port's investigation surfaced (fixed in
the same effort, different file)**: `crates/cli/src/model_dir.rs`'s
`resident_for` synthesized a GGUF card's `family` from `general.architecture`
verbatim and dispatched it through the SAME match as brain-native checkpoint
families - so a raw `qwen35`/`qwen35moe` GGUF, dropped into the model
directory, was routed straight into a resident whose `Engine` cannot open
GGUF bytes at all, instead of the actionable `brain import-gguf` hint. Fixed
with a gate before the match, allowlisting only the families that read GGUF
themselves; see lesson 64 in `.agents/rules/lessons.md`.

### M21 (PARTIAL): a resident that reads the Q8_0 GGUF directly, across two GPUs, with no fp32 intermediate

`crates/qwen35/src/int8_gguf_resident.rs`, model id
`unsloth/Qwen3.8-27B-Q8_0`, registered via
`crates/cli/src/resident_qwen35.rs::multi_gpu_gguf_from_env` and
`Executor::register_multi`.

**Status, plainly: the PLUMBING works and is gated green, the LOADING is now
proven correct against an external oracle and gated green, and the OUTPUT is
still wrong and is gated RED.** The first two gates live in
`crates/qwen35/tests/gguf_resident_real.rs`
(`a_real_two_card_load_runs_end_to_end` and
`the_two_card_stack_continues_a_factual_prompt_correctly`), kept separate so
a reader can tell which question is settled; the loading gate is
`crates/qwen35/tests/gguf_reference_parity_real.rs` (see "An external oracle
now exists" below).

Composed, not written: `checkpoint::gguf::MmapGguf` (already a
`checkpoint::TensorSource`) -> `checkpoint::remap::RemapSource` over a
`Fetch::Whole` plan built from `gguf_import::classify` itself ->
`Qwen35::new_i8_shard` per card -> `model::shard::plan_fewest_devices` for
the split -> `Qwen35::run_decode_step`'s `input_override` seam for the
cross-card residual, whose correctness is already gated bit-for-bit at tiny
scale by `model.rs`'s `two_shard_int8_decode_matches_the_whole_shard_model`.

**Measured on the real file, this box** (2x Tesla P40, 2 GiB/card reserve,
cap 512):

- placement: **27.05 GiB total - layers 0..34 on gpu0 (13.67 GiB), 34..64 on
  gpu1 (13.38 GiB)**; one card correctly reported infeasible rather than
  attempted. Real peak was ~14.3 GiB/card (staging + pipelines on top of the
  planned weights), inside the 2 GiB reserve.
- cold load: **545.7 s** from cold page cache, **~40-65 s** warm, for the
  whole 29 GB file (dequantize each Q8_0 leaf out of the mapping,
  re-quantize to brain's group-wise INT8, upload).
- throughput: **3.83 tok/s prefill, 3.93 tok/s decode** (they are the same
  one-token-per-pass primitive here, so the near-equality is expected, and
  the split is now reported by `Instance::metrics`).
- greedy decoding is bit-stable across requests on one instance.

**The endpoints do not live in a shard, and cannot.** The first real load
died in `paramstore::upload`: `tok.weight` is 5_085_593_600 bytes as fp32
and this card's `max_buffer_size` is 4_292_870_144; both `[248320, 5120]`
tables are also 2.4x the 2047 MiB storage-BINDING limit. So the stages are
built `embed: false, head: false` and the resident owns both ends - the
embedding is read one row at a time from the mapping
(`MmapGguf::tensor_range`, new, the GGUF twin of
`MmapSafetensors::tensor_f32_range`) and the head is INT8 via
`stream::quantize_i8_rows` + `stream::head_logits_on`, both generalized out
of `crate::stream`, which had already reached the same conclusion about the
same two tensors. See `.agents/rules/lessons.md` #69.

**A real, measured import defect this found and fixed:** llama.cpp's
converter stores `ssm_a = -exp(A_log)`, brain's `gdn_decay_gate.wgsl` wants
`A_log`. Importing verbatim made the GDN decay gate up to 260x too strong -
the recurrent state was annihilated every step, so the model kept ~one token
of context and echoed the prompt (`"The capital city of France is"` ->
`" France is France is France is"`). Fixed in the SHARED driver
(`gguf::import::Mapped::Transformed` + `ElemOp::LnNeg`), in this crate's
importer AND in `qwen35moe`'s (which had copied the same wrong assumption),
and in the resident's streaming loader (`SsmALogFix`, which also has to
refuse the zero-copy `raw_words` path or the transform is bypassed). Both
crates' synthetic fixtures wrote `ssm_a` POSITIVE - a value llama.cpp cannot
emit - so they could not have caught it; they now write it negative and
`ElemOp::LnNeg` refuses a non-negative input. See lesson #70.

**A second, quieter defect found on the way:** `Qwen35Config::layer_i8_bytes`
- the number this placement is built on, and the one `crates/perf`'s
`weights` scenario reports - charged `n*4` for a leaf's scales when
`model::ops::Weight::I8` has been group-wise (`n*k/8`) for a long time. It
under-counted every layer by 12.5%, 3.4 GB across the model. Fixed and now
gated against the real `model::int8::quantize_weight` output rather than
against its own pinned numbers; lesson #68.

**What is still wrong, and what has been ruled out.** After the `ssm_a` fix
the output changed but is still not the model's: `"The capital city of
France is"` -> `"..\n\n\n\n..."`. `BRAIN_QWEN35_GGUF_DEBUG=1` dumps the
per-card residual RMS and the head's top-5, and shows a model that predicts
sensibly at short range (it continues `"Kal"` with `"man"`, and predicts
`" city"` after `"The capital"`) and degrades as context grows - i.e. the
long-range mechanisms are degraded, not the graph.

Ruled out, each by a gate that now exists:

| Hypothesis | Evidence against |
|---|---|
| the cross-card split | `model.rs`'s `two_shard_int8_decode_matches_the_whole_shard_model` - bit-exact |
| the int8 DECODE tape | new `tests/decode_step.rs::int8_decode_step_matches_int8_full_prefill_{cpu,default_backend}` - int8 decode vs int8 prefill, worst maxabs 1.0e-7 (CPU) / 6.0e-8 (P40), 0 argmax mismatches. This closes the other half of lesson #67, which fixed the panic but left the VALUES ungated |
| the int8 tier itself, at depth | new `tests/gguf_i8_vs_fp32_real.rs` builds the SAME real layers twice from the SAME GGUF bytes (`Qwen35::new_fp32_shard_src`, also new) - worst cosine 0.9963 at 4 layers, 0.9862 at 8, 0.9877 at 12: real loss, but not collapsing with depth. And M16 above already got `" Paris."` out of 64 real layers at this same INT8 tier from the FP8 checkpoint |
| the embedding row read | `MmapGguf::tensor_range` is gated bit-identical against a whole-tensor decode incl. mid-block starts, on Q8_0 |
| the `(1+w)` norm folds | every norm's real value distribution is consistent with llama.cpp having already folded the plain norms and NOT folded the gated `ssm_norm` (which `Qwen3_5RMSNormGated` does not reparameterize - `crate::import`'s own doc) |
| an `ssm_alpha`/`ssm_beta` swap | tried on the real file: strictly worse (`"I"` then EOS) |

Two rows of that table have since been corrected by measurement, and are
kept above only so the correction is legible: the int8 row's "not collapsing
with depth" was an extrapolation from a depth-8 gate and is WRONG (see "A
real defect this DID find" below), and the whole framing that followed it -
that the loader must be mis-reading a tensor - is disproven next.

**An external oracle now exists, and it clears the loader.** The claim above
("the remaining defect is a difference between what this GGUF route loads
and what the safetensors route loads") is now DISPROVEN.
`tools/goldens/qwen35_gguf_reference_forward.py` is a second, independent
implementation of this exact architecture - pure CPython, no torch/numpy/
transformers/llama.cpp, transcribed from the published `modeling_qwen3_5`
reference module - that parses the GGUF header, dequantizes Q8_0 itself and
runs the real decoder over the real bytes. Against it, `Qwen35::
new_fp32_shard_src` fed by `int8_gguf_resident::shard_source` is
**bit-equal**: cosine 1.000000 and relative L2 0.0000 at 1, 4, 8 and 12 real
layers across 4 real decode positions, with the residual's `sum` (a
projection no permutation preserves) agreeing to 6 significant digits. The
first four layers of that comparison are now a standing gate,
`tests/gguf_reference_parity_real.rs`.

Every remaining layout question was settled from the FILE's own statistics
rather than by assumption, and each agrees with what the loader does:

| Question | How the file itself answers it |
|---|---|
| is `attn_q` per-head `[query\|gate]`-interleaved, or globally chunked? | interleaved: averaged over heads, the FIRST 256 rows of each 512-row block carry the rotary break at row 64 that `attn_k` also has (ratio 0.93 vs 1.00 for the second block, corr 0.41-0.54 against `attn_k`'s profile vs ~0.0) - the second block is the gate, which has no rotary structure |
| did llama.cpp apply its LLaMA-style q/k row permute (half-split -> adjacent-pair RoPE)? | no: `attn_k`'s row magnitude is a function of `d mod 32` over the 64 rotary channels (low `[0,16)`, high `[16,32)`, repeated), i.e. `d` pairs with `d+32`; an adjacent-pair permute would make it a function of `floor(d/2)`, and even/odd rows would differ (they do not) |
| are `attn_k`/`attn_v` swapped? | no: `attn_k`'s per-position profile has the rotary break at 64 (ratio 0.68-0.87), `attn_v`'s is flat (0.99) |
| `ssm_conv1d` layout and which tap is the current token | `[channel][tap]`, tap fastest, tap `K-1` current: the whole-tensor magnitude profile has its one regime change at flat index 16384 = `4096 channels * 4 taps`, exactly the k->v channel boundary, which the `[tap][channel]` reading cannot produce |
| `attn_qkv` order | `q\|k\|v`: the row-scale profile breaks at 2048/4096, matching `ssm_conv1d`'s own channel structure |
| is the block order scrambled? | no: `post_attention_norm` rises monotonically 0.78 -> 1.36 with depth, `attn_q_norm` 1.23 -> 1.66, `ssm_dt.bias` max grows with depth - a lexicographic or shuffled block order would be jagged |
| is `output.weight` really the head (not tied to `token_embd`)? | different bytes, and projecting the final residual through `token_embd` instead gives pure garbage (`"ipped"`, CJK fragments) |

**A real defect this DID find: the int8 tier collapses along the SEQUENCE,
not just with depth.** `gguf_i8_vs_fp32_real.rs` measures 8 layers and
concluded "real loss, but not collapsing"; that extrapolation was wrong,
because the resident decodes one token at a time through a PERSISTENT GDN
recurrent state, so W8A8 error compounds along the sequence as well as along
depth. Measured against the reference oracle at 32 real layers (half the
model), same tokens, same driver:

| position | cosine(int8, fp32) | rel. L2 |
|---|---|---|
| 0 | 0.9888 | 0.17 |
| 1 | 0.9098 | 0.44 |
| 2 | 0.7988 | 0.60 |
| 3 | 0.8856 | 0.49 |

At 8 and 16 layers the same positions are still 0.988-0.999, which is why a
depth-8 gate cannot see this. M16's `" Paris."` at "the same INT8 tier" is
NOT a counter-example: `stream::generate` re-runs the whole sequence through
chunked prefill every step and therefore never carries quantization error in
a recurrent state at all.

**What is still open.** The exact fp32 computation of these weights - brain's
and the independent reference's, which agree bit-for-bit - does not answer
the question either: for `"The capital city of France is"` the reference's
own top-10 is `" France"/" a"/" capital"/" is"/" the"/" "/" in"/" at"/
" around"/" of"`, with `" Paris"` at rank 80 (logit 9.47 against 15.25).
Grammar is intact, facts are not, and the distribution flattens as context
grows. Attention is not the culprit (probing layers 3/31/63 at position 5
shows peaked, content-dependent, per-head-varying distributions, and
disabling RoPE barely moves them - at theta 1e7 the angles over 6 positions
are tiny), nor is the GDN decay (measured per-head `exp(g)` on real
activations is 0.0013..1.0 with medians rising 0.47 -> 0.99 with depth, i.e.
long memory, not annihilation), nor per-layer contribution health (GQA layers
contribute 0.28-0.73 of the residual, GDN 0.07-0.65, both growing smoothly
with depth). Three whole-model reference variants were run end to end and
none recovers the answer: GDN `q`/`k` swapped (invisible at position 0 by
symmetry, so the exact shape of the symptom - and still wrong), the GDN
recurrent state force-zeroed each step, and a `<|endoftext|>` BOS/attention
sink prepended (stopped after 5 of its 7 positions - every one of them was
still topped by a prompt token, so it was not going to turn). So the
remaining defect is either in the community
conversion itself (unverifiable here: no llama.cpp binary, no
`convert_hf_to_gguf.py`, no `gguf` python module, and the FP8 safetensors
checkpoint does not fit the free disk) or in a convention the two
implementations share and the file's own statistics cannot distinguish - the
GDN value-head grouping (`repeat_interleave` vs tile) and an
`ssm_alpha`/`ssm_beta` swap are the two such conventions left, and both are
statistically symmetric. A second `qwen35`-architecture GGUF from a
different quantizer would settle it in one run.

**Deliberately not in this milestone**: MTP (requires a whole shard,
asserted in `new_impl_on`), vision, LoRA, batched prefill, and more than one
sequence per dispatch.

### M22 - decode throughput: first honest profile, 3.94 -> 7.44 tok/s on the real two-card resident

**Status: done.** Measured on the real `unsloth/Qwen3.8-27B-Q8_0` GGUF across
2x Tesla P40, 64 real layers, INT8, `n = 1`. Every number below is a
WHOLE-PASS number from `qwen35_decode_profile` on the production flush path;
the per-kernel tables it also prints are used only to rank.

**There was no decode profiler.** `qwen35_bench` prices one layer at prefill
widths on random weights, which cannot answer where a served token goes: at
`T = 1` the model is memory-bound on its own weights and the ranking is a
different ranking. `crates/qwen35/src/bin/qwen35_decode_profile.rs` drives the
real resident and reports the whole-pass rate, the per-kernel table (all
stages merged), and the weight-streaming roof the rate is judged against.

| | tok/s | ms/token |
|---|---|---|
| baseline (start of M22) | 3.94 | 253.6 |
| + `rmsnorm_rows` on the decode tape | 7.19 | 139.1 |
| + M-RoPE dedup and an explicit per-layer flush | **7.44** | **134.4** |

**1.89x.** Baseline artifact:
`scripts/gates/qwen35-perf-baselines/qwen35-resident-int8-cpu48-gpu2.json`.

**What the profile actually said.** The top row was not the GEMM:

```
rmsnorm                 117.478 ms/token   210 calls   48.2%
matmul_i8_gemv_reg      103.076 ms/token   497 calls   42.3%
bmm                       5.780 ms/token    96 calls    2.4%
```

RMSNorm cost more than the entire 27 GB int8 weight stream underneath it.
`rmsnorm.wgsl` assigns thread `t` row `t`, so a one-row decode norm runs a
5120-element reduction on ONE thread of a 3840-core card with every 32-byte
sector serving a single useful float. The coalesced `rmsnorm_rows` already
existed, `backend_api::select` already preferred it, and `block::rms_variant`
was already the seam three other models selected it through; qwen35 dispatched
the reference by hardcoded index and never asked. 19.4x on the kernel, 1.82x
on the pass. Gated by `tests/rmsnorm_variant_agreement.rs` against a HOST
reference (the swap is not bit-identical - 64 partials fold in a different
order, agreeing to ~3e-6 - which is why it sits at the call site and not in
`gpu_core::upgrade`).

**The expensive surprise** is lesson #75: deduplicating the per-GQA-layer
M-RoPE upload, which is strictly less work, made the pass 1.45x SLOWER,
because `Gpu::write*` flushes the pending queue and those 32 uploads per token
had been the only thing giving the pass any host/device overlap. One explicit
`Gpu::flush()` per decoder layer recovered it and made the dedup free.
Measured cadences: 1/layer 7.47, 2/layer 7.31, 1/2 layers 7.44.

**Where it stops, and why that is the honest end of this phase.** The pass is
now device-bound: 131.6 ms/token of device time against 134.4 ms/token of wall
clock. `matmul_i8_gemv_reg` is 78% of it at 103 ms/token, streaming ~14.5 GB
per card per token at roughly 280 GB/s, which is about 81% of a P40's 346 GB/s
theoretical DRAM roof. Per the kernel checklist section F.2, a top row already
at its memory roof cannot be fixed by a kernel change; the only levers left are
to move fewer bytes (INT4 - a precision change, not a speed change) or to batch
independent rows over the same weights (MTP self-speculative decode, or a
second concurrent sequence - both separate phases). Everything else in the
table put together is 28 ms/token, so even reducing ALL of it to zero would
reach only about 9.5 tok/s.

**Hypotheses from the pre-M22 survey that the profile KILLED**, recorded
because a dead hypothesis is worth as much as a confirmed one:

* *"8 of 11 per-layer decode GEMMs dispatch the naive `MATMUL` instead of the
  selector."* Stale - already fixed. Every projection in
  `layer_gdn_decode_step`/`layer_gqa_decode_step`/`mlp_fwd` goes through
  `ops_linear` -> `Ops::matmul` -> `select`, and at `m = 1` with an INT8
  weight that lands on `matmul_i8_gemv`, upgraded to
  `matmul_i8_gemv_reg#MREG=1`. The profile shows that name and no `matmul`.
* *"The LM head runs on the host (`hostmath::matvec_par` over
  `[248320, 5120]`)."* True of `serve::Engine`, NOT of this resident, which
  has always used `stream::head_logits_on` with an INT8 device weight.
* *"~800 `submit()` calls per token."* Real, and free. `Gpu::submit` on this
  backend appends to a pending list under a mutex; it is not a queue
  submission. Collapsing them would have bought nothing - and see lesson #75
  for how the opposite turned out to be true of the calls that DO submit.
* *"~1200 `create_buffer`/destroy pairs per token; `Gpu::scratch_scope` is
  unused."* Real and still unused, but host time is now ~2% of the pass
  (134.4 wall vs 131.6 device), so the whole remaining prize is under 3
  ms/token. Not worth the aliasing-contract risk at this point; revisit only
  if the device side is ever cut enough to expose it.

**Correctness.** Unchanged, and separately gated. Every real-weight gate stays
green (`gguf_reference_parity_real`, `gguf_i8_vs_fp32_real`, `decode_step`,
`shard_parity`, `two_shard_int8_decode_matches_the_whole_shard_model`), and
`the_two_card_stack_continues_a_factual_prompt_correctly` stays RED for the
reason M21 records - the int8-along-the-sequence compounding of lesson #72,
unrelated to anything here. One consequence worth noting: the e2e plumbing
gate used to assert that the generated text had more than three distinct
characters, and that check was a coin flip on known-garbage output (it passed
on `"Give one"` while the red gate's own `"..\n\n..."` would have failed it).
A legitimate 1e-6 reduction-order change flipped it, so it was removed with a
comment pointing at the reference-comparing gates that catch broken plumbing
properly, and at the condition for restoring it.

### M23 (DONE): the root cause of the RED output gate, found and fixed - GGUF-stored GDN value heads are GROUP-MAJOR, brain wants SUB-MAJOR

**Status: `the_two_card_stack_continues_a_factual_prompt_correctly` is GREEN.**
`"The capital city of France is"` -> `" Paris. Paris is the largest city in"`
on the real two-card INT8 resident, at 7.57 tok/s (M22's throughput, unaffected).

M21 left this RED with the candidate space narrowed to two conventions the
file's own statistics could not distinguish: "the GDN value-head grouping
(`repeat_interleave` vs tile) and an `ssm_alpha`/`ssm_beta` swap". This
milestone ran the decisive experiment M21 named but could not: a direct,
per-tensor diff against the Qwen3.8-27B **FP8 safetensors** checkpoint - the
KNOWN-GOOD side, since the same INT8 tier over 64 real layers already
produced `" Paris."` from FP8 (M16) - refetched once disk space and a working
HF login made that possible (`crates/qwen35/tests/gguf_vs_fp8_weights_real.rs`,
a permanent diagnostic; `crates/qwen35/tests/gguf_vs_fp8_permutation_search.rs`,
the exploratory harness that found and confirmed the exact transform).

**The finding.** llama.cpp's GGUF conversion stores EVERY GDN leaf indexed by
VALUE HEAD - `ssm_a` (`A_log`), `ssm_dt.bias` (`dt_bias`),
`ssm_conv1d.weight`'s v-channels, `attn_qkv.weight`'s v-rows,
`ssm_alpha.weight`/`ssm_beta.weight`'s rows, `attn_gate.weight`'s rows,
`ssm_out.weight`'s columns - in GROUP-MAJOR order (`index = group*num_k_heads
+ key_head`, i.e. a key head's `group` repeats sit `num_k_heads` apart,
strided), not brain's (and the reference HF model's) SUB-MAJOR order
(`index = key_head*group + repeat`, i.e. a key head's repeats are adjacent).
Every head's own decay/bias/projection was individually plausible - finite,
in-range, nothing an output-shape or NaN check could catch - just applied to
the WRONG head's key/value state: grammatically fluent, factually wrong
output that degrades with context length, exactly the M21 symptom.

**Why the earlier oracle comparison could not see this.** `gguf_reference_
parity_real.rs`'s independent CPython reader (`tools/goldens/
qwen35_gguf_reference_forward.py`) agreed bit-for-bit with brain's own
(pre-fix) reading - M21 read that agreement as "the loader is proven
correct". It wasn't: the oracle's own `kh = j // group` line paired
repeat_interleave-ordered K/Q with the SAME un-degrouped, group-major A/dt_bias
brain read - two independent implementations sharing the identical wrong
assumption, not independent evidence. Two SEPARATE, real bugs compounded
into one symptom, and both had to be found and fixed to close it (see below).

**The decisive signal.** Three of the EIGHT affected leaves - `A_log`,
`dt_bias`, `ssm_conv1d.weight` - are small enough that llama.cpp stores them
unquantized (no Q8_0 rounding noise on top), so the comparison against FP8
was exact: cosine **1.0000000**, not merely close, for the single group-major
-> sub-major transpose tried. The other five (`in_proj_qkv`/`in_proj_a`/
`in_proj_b`/`in_proj_z`/`out_proj` - Q8_0-quantized, real rounding noise on
top) resolved to cosine 0.9996-1.0000 under the exact same transform - the
first attempt at these landed on a backwards divmod and looked unresolved,
corrected once the small leaves' exact match proved the transform itself was
right and the bug was in the harness, not the hypothesis.

**The fix, and a fix to the fix.** `crate::int8_gguf_resident::GdnHeadOrder`
(wrapped by the existing `SsmALogFix` streaming source - name kept, scope
widened): one `src_head(h)` formula, applied at three granularities -
`degroup_heads` for the two flat `[num_v_heads]` vectors, `degroup_rows` for
row-block leaves (an optional q/k prefix `conv1d.weight`/`in_proj_qkv.weight`
carry, `in_proj_z.weight` does not, and `in_proj_a.weight`/`in_proj_b.weight`
are one-scalar-per-head - `head_dim=1` - not `linear_value_head_dim`-wide
blocks), `degroup_cols` for `out_proj.weight`'s column-block axis. The FIRST
version of this fix shipped with `in_proj_a`/`in_proj_b` never wired into the
dispatcher at all - `degroup_rows` hardcoded `self.head_dim` (128) with no way
to express these two leaves' `head_dim=1` shape, so they were silently left
group-major. The real end-to-end gate passed anyway (this specific 8-token
greedy continuation was not sensitive enough to catch it), but a re-run of
the M23.1 diff at MORE layers (0/1/3/31/32, as more of the FP8 checkpoint
downloaded) caught it directly: `in_proj_a`/`in_proj_b` still showed low
cosine at layer 1 where every other leaf was clean. `degroup_rows` gained a
`head_dim` parameter instead of always reading `self.head_dim`, both leaves
were wired in, and the diff re-run now shows cosine 1.0000 for both at every
tested layer. All four permutation methods are gated by hand-computed
synthetic unit tests (`degroup_heads_matches_the_hand_computed_transpose`,
`degroup_rows_at_zero_offset_matches_the_hand_computed_block_permutation`,
`degroup_rows_leaves_the_prefix_before_row_offset_untouched`,
`degroup_rows_at_head_dim_one_matches_the_hand_computed_row_permutation`,
`degroup_cols_matches_the_hand_computed_block_permutation_per_row`), not only
against real weights - the lesson: a real-weight gate that happens to pass is
not proof every affected leaf was found; the synthetic tests and the
per-tensor diff are what actually enumerate them. `crate::gguf_import::
classify` (the offline converter) is **not yet updated** - the resident's
streaming path was the priority since it is what actually serves the real
checkpoint; the offline path is deprecated (`brain import-gguf` forwards to
it) and cannot even run a full conversion of the real file on this box
(~108 GB, M20's own note) - tracked as open work, not silently skipped.

**Confirms M21's own "along-the-sequence" finding was partly this bug.**
`gguf_i8_vs_fp32_real.rs` (int8 vs fp32, both from the SAME now-fixed GGUF
bytes) now holds a stable cosine 0.997-0.999 across all 8 positions at 8 real
layers - no degradation trend at all, where M21 measured a real collapse
(0.9888 at position 0 dropping to 0.7988 by position 2, at 32 layers). The
32-layer re-run to confirm the same holds at that depth hit an unrelated
VRAM ceiling on this box (fp32 + int8 built simultaneously at 32 layers
does not fit one 24 GiB P40) - open, not contradicted.

**Open follow-up, honestly RED.** `gguf_reference_parity_real.rs`'s pinned
digest was regenerated against the fixed CPython oracle, but the oracle
still disagrees with brain's Rust computation by a shrinking-but-real margin
(pos 0: rms 0.6952 in Rust, the pinned `EXPECT` differs) - very likely one
more missed or misordered call site in the ~450-line reference script, not a
defect in the two independently-passing checks (real correct generation;
the direct per-tensor diff above). Left honestly failing rather than pinned
to whatever the script currently prints. See that test file's own doc for
the full accounting.

Verified real-time on this box (2x P40, 48 cores, 184 GiB RAM):
`the_two_card_stack_continues_a_factual_prompt_correctly` completed in 69.6 s
warm-cache (cold load ~260 s the first run after this fix landed, consistent
with M21's own cold-load figures - the fix changes correctness, not load
time or throughput).

### M24 (DONE): the Q4 (W4A8) weight tier, wired into dense qwen35 - one card, correct output, ~1.4-1.7x M22's throughput

M22 left decode device-bound at 81% of the P40 DRAM roof with one lever
left: stream fewer weight bytes. Brain already had a parity-gated INT4
(W4A8) tier (`model::ops::Weight::Q4`, `model::int4::quantize_weight_q4`,
four q4 kernels) that qwen35 registered but could never select - `Qwen35::
new_impl_on` took `i8: bool`, which cannot express a third tier.

**Phase 0 (blocking measurement, done first).** An earlier, un-templated
benchmark found `matmul_q4_gemv_reg` losing to the plain `matmul_q4_gemv` at
m=1..16 - the same occupancy pathology `matmul_i8_gemv_reg` was built to
fix, appearing to recur for Q4. Re-measured at qwen35's OWN real decode
shapes with the correct per-`m` `MREG`-bucketed build
(`crates/model/tests/matmul_q4_speed_bench.rs::
gemv_vs_gemv_reg_at_qwen35_decode_shapes`): `_reg` won at every one of the
five shapes, 1.55-1.88x. Wired into `gpu_core::upgrade`'s transparent seam,
gated by a fresh bit-identity + integer-oracle test
(`crates/gpu-core/tests/q4_gemv_reg_upgrade.rs`) mirroring the I8 row's own.

**The wiring.** `TierPolicy` (`model::ops`) - a per-leaf substring-matched
`Dtype`, generalizing the single `Dtype` `qwen3::Qwen::new_shard_dt` takes -
replaces the `i8: bool`. `Qwen35::new_impl_on`'s signature widened; every
existing public constructor (`new_i8`, `new_i8_shard`, `new_on_i8`, ...)
kept as a one-line alias over the new `new_shard_dt`/`new_on_dt` - zero
external call-site breakage, confirmed by the full existing suite staying
green throughout. The per-layer upload loop now runs off
`Qwen35Config::layer_leaves`, the SAME table `layer_weight_bytes` folds for
its cost estimate - the byte formula and the uploader can no longer drift
the way `layer_i8_bytes` once did (lesson #68). `int8_gguf_resident.rs`
threads the policy through placement and a new `BRAIN_QWEN35_GGUF_TIER` env
var; `qwen35_decode_profile` reads it and stamps it into the baseline
artifact's `target` field.

**Real numbers, this box, this checkpoint (2x P40, cap 512, real prompt):**

| Tier | Cards | Resident bytes | tok/s (decode) | vs M22 I8 baseline |
|---|---|---|---|---|
| I8 (M22 baseline) | 2 | 27.05 GiB | 7.44-7.57 | 1.00x |
| Uniform Q4 | **1** | 15.71 GiB | 10.96 (`qwen35_decode_profile`, 16 steps) / 12.46 (real `generate` call, 8 steps) | **1.47-1.65x** |
| Policy C (Q4 MLP, F32 GDN gates) | **1** | 15.79 GiB | 10.89 / 12.44 | **1.46-1.65x** |

Both real-`generate` numbers come from
`crates/qwen35/tests/gguf_resident_real.rs::
the_q4_tier_continues_the_same_factual_prompt_on_one_card`, which ALSO
checks output correctness: both policies produce the byte-identical
`" Paris. Paris is the largest city in"` the I8 tier does. Policy C - the
recommended default, holding the two GDN state-sensitive gate projections
(`in_proj_a`/`in_proj_b`, ~94 MB total) at F32 - costs 0.6% of uniform Q4's
speed for a real quality hedge; both were measured rather than assumed
necessary, and both currently produce identical text at `max_new=8`, greedy.

The single-card fit is a standing arithmetic gate, not a one-off measurement:
`int8_gguf_resident::tests::a_q4_mlp_tier_with_gdn_gates_held_at_f32_fits_
one_24gb_card` and its uniform-Q4 sibling both assert `plan_by_capacity`
returns exactly one stage, the same way `the_real_model_needs_two_24gb_
cards_and_fits_them` pins I8's two-card requirement.

**What this unlocks, not yet built:** MTP self-speculative decode (asserted
whole-shard-only today - `cfg.mtp` requires `shard.is_whole`, which a 2-card
split can never satisfy but a 1-card Q4 build now could; M17 measured a
1.333x pass-count reduction on top, which would compose multiplicatively),
and a second concurrent sequence on the freed card (bandwidth-bound decode
makes a second sequence nearly free per-token). Both real milestones, not
attempted here - see `.agents/roadmap/qwen35.md`'s own "Not yet done" for
the four concrete prerequisites MTP-on-this-path needs.

**Not yet done from the original M24 plan:** a Q4 `lm_head` (saves ~0.6 GB,
needs a `quantize_q4_rows` sibling to `stream::quantize_i8_rows` - the head
stays INT8 in every measurement above); `qwen35moe` adoption of
`TierPolicy` (the GdnSlot/DecodeCaches hoist into `crates/model` that would
let it share this work without a third copy is itself still open); the
offline `gguf_import.rs` converter path.

### M25 - chunked prefill: 41 -> 873 prefill tok/s, and the end of the per-token replay

**Status: done.** `crates/qwen35/src/serve.rs::Engine::prefill` replayed a
prompt ONE TOKEN AT A TIME - a full `Qwen35::run_decode_step` (its ~40
dispatches per layer, its `m = 1` GEMV against every weight in the model) per
prompt token. That module's own doc named it as deliberately deferred. It is
no longer deferred: the prompt is now consumed in bounded rounds of
`MAX_PREFILL_TOKENS = 256`, one dispatch shape per layer per round.

**The three seams, and what each of them actually needed.**

1. **GDN recurrent state - already threadable, never threaded.**
   `model::gdn::gdn_chunk_fwd` has taken explicit `initial_state`/`final_state`
   parameters all along; `gdn_mixer_fwd` simply allocated both fresh, so every
   whole-sequence forward started from zero. That is right for training and
   wrong for round 2 of a prefill. `gdn_mixer_stream_fwd` (new, and
   `gdn_mixer_fwd` is now a `cont: None` wrapper around it, so no existing
   call site moved) binds the caller's persistent buffer as `initial_state` -
   safe, because `gdn_chunk_fwd` reads it exactly once, in its own seeding
   copy - and copies `final_state` back after the loop rather than aliasing
   the two across a dispatch.
2. **GDN conv history - NOT something `gdn_chunk_fwd` returns.** This was the
   part worth tracing carefully. The whole-sequence path expresses causality
   as `conv1d_fwd`'s left `pad = K-1`, which IS the zero history and cannot
   express "continue from these K-1 real values". And the tail is not in any
   of the mixer's outputs: it is the last `K-1` rows of the round's own
   `mixed_qkv`, the conv's INPUT, before the conv or the SiLU. So the streaming
   arm prepends the history rows to the round's rows (`concat2` on a flat
   token-major concatenation), convolves the `K-1+t` extended input with
   `pad = 0` (exactly `t` outputs, each with real left context), and writes the
   extended input's own last `K-1` rows back as the next round's history -
   which is correct even when `t < K-1`, where some of those rows are still
   the inherited history. The layout conversion is two `nchw_nlc`/`nlc_nchw`
   dispatches: `gdn_causal_conv1d_step`'s `hist` is channel-major `[C, K-1]`,
   oldest tap first, and the round's rows are token-major.
3. **GQA attention against a growing cache - no new kernel needed.**
   `model::block::gqa_chunk_step` (new) is `gqa_decode_step` generalised from
   one query row to `n`: bulk-fill rows `start..start+n` of the flat per-layer
   cache, then attend each row against `0..=start+i`. Two findings made this
   cheap. First, `kv_cache_fill`'s single `kv_append` dispatch only reaches
   rows `0..n` (that kernel writes at `row*width`, so widening `width` to the
   chunk pins the destination to a multiple of the chunk length) - the general
   form is `splice.wgsl`, `dst[base+i] = src[i]`, which is exactly a bulk fill
   at an arbitrary row offset (`kv_cache_fill_at`). Second, causality here is
   per-row (`seq_lens[i] = start+i+1`), not a `j <= i` index comparison, which
   is precisely the contract `paged_decode_scores_batched` /
   `decode_softmax_batched` / `paged_decode_apply_batched` already implement -
   and two of those three were ALREADY compiled into this crate's pipeline
   list to satisfy `Ops::REQUIRED_KERNELS` ("compiled, never dispatched"). A
   flat `[cap, kv_dim]` cache is the degenerate one-block case of a paged pool
   (`block_size = cap`, one block, `max_bt = 1`), so their slot arithmetic
   reduces to plain flat addressing with no indirection. qwen3's block-table
   machinery was deliberately NOT ported - this engine has one physical block
   per sequence and needs none of it.

**What did NOT fit: the fused flash prefill.** `paged_flash_prefill.wgsl` is
the right long-term kernel (no materialised score slab at all) and its host
tape is the same one built here, but its shared-memory tiles hardcode
`HD = 128` and this model's `head_dim` is 256. So the triad stays, and with it
a `[chunk, n_heads, pos+chunk]` scores/probs scratch - about 100 MB at a
4096-token context and about 1.2 GB at 48K with `chunk = 256`. That is the
remaining barrier to a genuinely 48K-token prefill, and it is a kernel-side
fix (a 256-wide tile variant, or a `LANES`/`CH` retune), not a host-side one.

**Measured, on synthetic weights at the real PER-LAYER shape.** The real 27B
cannot be measured through `serve::Engine` on this box at all: that engine
builds a plain FP32 `Qwen35`, and the real 64-layer FP32 model is ~108 GB
against two 24 GB P40s (this is also why the fp32 `lm_head` alone, 5.09 GB,
is past `max_buffer_size` - the reason the REAL-weight path is
`int8_gguf_resident`, which is not this milestone's target). So
`crates/qwen35/src/bin/qwen35_prefill_profile.rs` prices both replay shapes on
the same instance at the real per-layer dims (`d_model 5120`, `ff 17408`, 24
query / 4 KV heads of 256, 48 GDN value heads of 128, `full_attention_interval
4` so the layer-type mix is exact) with the layer count and vocabulary scaled
to fit. Single P40, fp32, 512-token prompt, chunk 256:

| layers | per-token replay | chunked prefill | speedup |
|---|---|---|---|
| 4 | 41.2 tok/s (24.3 ms/tok) | **873.3 tok/s** (1.15 ms/tok) | **21.2x** |
| 8 | 21.2 tok/s (47.2 ms/tok) | **433.4 tok/s** (2.31 ms/tok) | **20.5x** |

Both paths halve with depth, so the ratio is depth-independent across the
range measured. At 2048 prompt tokens (4 layers) it is 40.2 -> 759.5 tok/s,
18.9x - the chunked path loses a little as the attention it must recompute per
round grows, which is the `MAX_PREFILL_TOKENS` scratch story above showing up
in time as well as in bytes.

`MAX_PREFILL_TOKENS = 256` is measured, not guessed (512-token prompt, 4
layers): chunk 64 -> 501 tok/s, 128 -> 838, 256 -> 873, 512 -> 847. The curve
is flat past 256 and the scratch keeps growing, so 256 is where it stops
paying.

(Every figure in this section was measured before M26 bounded a round's
in-flight transient memory with `DRAIN_EVERY_N_LAYERS`, which costs 1.4% at
this shape and 3.5% at 2048 prompt tokens: 873.3 re-measures as 840.0, 433.4
as 425.2, 759.5 as 732.8. The ratios and the conclusion are unchanged - see
M26 for why the drain is worth that.)

**Correctness.** Two new gates, both spec-level, and the bounds in both are
measured against deliberately broken implementations rather than picked:

* `crates/qwen35/tests/chunked_prefill.rs` -
  `chunked_prefill_matches_token_by_token_replay_{cpu,default_backend}` and
  `whole_prompt_single_chunk_matches_token_by_token_replay`. A 14-token prompt
  at chunk 4/8/16, then three further single-token `step` calls on each path:
  the chunked prefill must leave EXACTLY the decode state the per-token replay
  leaves, so the continuation matches. Correct: 0 (CPU JIT) / 3.7e-9 (wgpu).
  Broken: KV rows filled at offset 0 -> 6.2e-3; the per-query causal mask
  flattened -> 3.1e-3; chunk-RELATIVE instead of absolute M-RoPE positions ->
  1.4e-3. The bound is 1e-5. Note the third one: `decode_step.rs`'s own 2e-3
  logits bound would have PASSED a wrong-RoPE-position prefill, which is why
  this file does not reuse that number.
* `crates/model/tests/gdn_mixer_stream.rs` -
  `threading_the_stream_state_across_rounds_matches_the_whole_sequence_forward`.
  Rounds of 3+5 (with different internal chunk sizes) against one 8-row
  forward, comparing the mixer's own output. Correct: 2.0e-7. Dropped
  recurrent state: 0.34. Dropped conv history: 0.64. Bound 1e-5.

  This second gate exists because the first one CANNOT carry the GDN state
  claim. Measured: on random weights, a chunked prefill that carried no
  recurrent state at all between rounds moved `qwen35`'s final hidden state by
  5e-7 - the residual stream and the final RMSNorm dilute the mixer's
  contribution that far - and passed a 2e-3 end-to-end bound comfortably. Two
  further fixture facts came out of chasing that: the shipped
  `init::init_weights` (reference init: `dt_bias = 1`, `A ~ U(0,16)`) gives a
  per-token state decay of about `e^-10`, i.e. no memory to test at all, so
  the fixture retunes the gate to a ~0.98/token decay; and a config with a
  SINGLE GQA layer cannot see a round's internal masking at all (a round's
  non-final rows feed nothing that outlives the round), so the fixture widens
  `tiny()` to eight layers to get a second one.

Every pre-existing qwen35 and model gate stays green, including
`decode_step.rs` (the per-token tape is untouched) and `serve.rs`.

**Not done here, deliberately - closed by M26 below.** `int8_gguf_resident.rs`
still replayed per token after this milestone. It is the REAL-weight path, so
it is where this speedup is ultimately worth the most, but it is multi-stage
(pipeline-parallel across two cards) and its cross-stage seam -
`run_decode_step`'s `input_override` - is `[d_model]`, one row. A chunked
resident needs that seam widened to `[n, d_model]`. `run_prefill_chunk`
asserted `shard.embed && shard.head` precisely so that gap was loud rather
than silent; it still does, but it is now a thin wrapper over the sharded
primitive M26 added rather than the only chunked primitive there is.

### M26 (DONE): chunked prefill on the REAL path - 262.8 s -> 26.5 s of prompt replay on the two-card 27B resident

M25 built the chunked primitive and measured it at 21x, but only ever ran it
on synthetic weights: `serve::Engine` builds a plain FP32 `Qwen35`, and the
real 64-layer FP32 model is ~108 GB against two 24 GB P40s. The checkpoint
this box actually serves goes through `int8_gguf_resident`, and that path was
still replaying the prompt one token at a time. Measured, real, before this
milestone: a 1731-token prompt took **262.8 s to prefill (6.6 tok/s)**. After:
**26.5 s (65.4 tok/s), 9.9x**, with the same greedy continuation.

**What the seam actually needed.** M25's `run_prefill_chunk` asserts
whole-model, and its `input_override`-equivalent is not exposed for a chunk at
all in a sharded configuration. The generalization is exactly the one
`run_decode_step` already makes over shards, one row at a time, lifted to `n`:

* `Qwen35::run_prefill_chunk_stage` is M25's body with two branches instead of
  two asserts - gather through `tok.weight` only when `shard.embed` (otherwise
  the input is an `[n, d_model]` `input_override`), and through the final
  `norm.weight` only when `shard.head` (otherwise the raw last-layer residual
  block is what the next stage wants). The per-layer GDN/GQA chunk math is
  untouched and unmoved; the `DecodeCaches` state contract is M25's, verbatim.
* `run_prefill_chunk` is now a wrapper: assert whole-model, call the stage
  form, split off the last row. `serve::Engine` and
  `tests/chunked_prefill.rs` are behaviourally unchanged, and that file's
  three gates still print the same 0 / 3.7e-9 they did.
* `prefill_chunk_stage` host-stages the result, the exact counterpart of
  `decode_step_stage` - one host round trip per stage per ROUND where the old
  path paid one per stage per TOKEN. At two stages and 1731 tokens that is
  3462 round trips replaced by 28.
* On the resident: `embed_rows` gathers a whole round's embedding rows
  straight from the GGUF mapping (one `MmapGguf::tensor_range` per row - an
  arbitrary token set names non-contiguous rows of a `[vocab, d_model]` table,
  so there is no wider range to ask for; the saving is downstream, not here),
  and `stack_prefill_chunk` threads the `[n, d_model]` carry through every
  stage, projecting only the round's last row through the head.

**The one thing M25's numbers did not transfer: how much memory a round is
allowed to have in flight.** Swept on the real checkpoint, 2x P40, 1731-token
prompt, peak card occupancy sampled from `nvidia-smi`:

| chunk | prefill | tok/s | peak VRAM |
|---|---|---|---|
| (per token) | 262.8 s | 6.6 | - |
| 64 | 36.5 s | 47.4 | 16.2 GiB |
| 128 | 26.6 s | 65.0 | 17.8 GiB |
| 192 | 26.4 s | 65.4 | 20.3 GiB |
| 256 | out of memory | - | >24 GiB |

`serve::Engine`'s own 256 did not run. The reason is not the GQA score slab
M25 already accounted for: EVERY intermediate a layer allocates stays alive
until something drains the queue, and a round drained only at its terminal
readback, so a stage held all ~32 of its layers' worth at once at the real 27B
widths (`d_model` 5120, `ff` 17408, `conv_dim` 10240) - about 5 GiB on top of
13.5 GiB of resident INT8 weights.

That is a DEPTH cost, not a chunk cost, and capping the chunk is the wrong
lever for it - the Q4 tier fits ONE card, where a stage is 64 layers deep and
overruns the same budget at half the chunk (which is exactly how this surfaced:
the single-card `decode_throughput_at_a_real_long_context` gate went red while
the two-card one was green). So the fix went where the defect was:
`run_prefill_chunk_stage`'s layer loop now BLOCKS on the device every
`DRAIN_EVERY_N_LAYERS = 4` layers instead of only flushing, which is what lets
wgpu actually reclaim the layers already behind it. Live transients become
`4 * per_layer` regardless of shard depth, and the table becomes:

| chunk | prefill | tok/s | peak VRAM |
|---|---|---|---|
| **256** | **26.5 s** | **65.4** | **15.3 GiB** |
| 512 | 27.6 s | 62.8 | 16.6 GiB |

256 is now both the fastest measured round size and cheaper in memory than 64
was without the drain, so the resident and `serve::Engine` agree on the
constant after all. The drain costs host/device overlap, priced with
`bin/qwen35_prefill_profile` (synthetic weights, real per-layer dims, single
P40, 512-token prompt, chunk 256): 851.9 tok/s draining never, 840.0 every 4
layers (-1.4%), 807.8 every layer (-5.2%); at 2048 prompt tokens the every-4
cost is -3.5% (759.5 -> 732.8). M25's headline 873/433 figures shift by that
much and no more, which is the price of a prefill whose memory no longer
scales with model depth.

Note also what the M25 scope note got wrong in the other direction: it
predicted this would need `t`-sized per-stage buffers on every card. It did
not - every buffer a round touches is allocated per call from `n`, so the
stages stay built at `b = t = 1`.

**Correctness.** Two gates, one synthetic and one on the real 27B:

* `qwen35::model::tests::two_shard_chunked_prefill_matches_token_by_token_replay`
  \- RED before the primitive existed. Two stages (cut at layer 5, so a GQA
  layer and GDN layers sit on BOTH sides), a 14-token prompt in rounds of 4
  (4+4+4+2, ragged last round), then three further single-token steps through
  the same two stages, against the same prompt replayed one token at a time.
  Worst maxabs 3.7e-9 against a 1e-5 bound; a stage-1 seam deliberately broken
  to seed itself with zeros instead of `input_override` measures 1.86. Stage 1
  is fed a deliberately WRONG `token_id` throughout, so "it used the seam and
  not the token" is a checked fact rather than an assumption.
* `crates/qwen35/tests/gguf_resident_real.rs::prefill_throughput_at_a_real_long_context`
  \- the measurement above, and a real-weight correctness gate in the same
  run: the 1731-token prompt ends on M23's factual cue, and the answer must
  still be Paris after 14 rounds of chunked prefill. It is (`" Paris. It
  is"`). All seven real-checkpoint gates in that file stay green.

**A round is only worth issuing if the tier's GEMM is tiled - and Q4's is
not.** This was found by measurement, not by reading: with the chunked path
wired unconditionally, the single-card `decode_throughput_at_a_real_long_
context` gate went from ~2.5 minutes to over 18, and its own metrics said why.
Real checkpoint, uniform Q4, ONE P40, 1555-token prompt:

| replay | prefill | tok/s |
|---|---|---|
| per token | 152.1 s | 10.2 |
| rounds of 256 | 1108.3 s | 1.40 |

A **7.3x regression** from the same host change that is a 9.9x win at INT8.
The cause is one line of dispatch: `Ops::bind` maps `(PackedInt8, Q4)` to
`matmul_q4_dyn.wgsl`, which is by its own header "the correct-first, non-tiled
q4 GEMM ... one thread per output element", while `(PackedInt8, I8)` maps to
the 128x128 register-tiled `matmul_i8_dyn`. At `m = 1` Q4 instead gets
`matmul_q4_gemv`, which is coalesced and workgroup-reducing - so for Q4 the
per-token tape is genuinely the faster one, and a round is pure loss.

`Qwen35::chunked_prefill_is_profitable` (true unless any weight is `Q4`)
therefore selects the tape in `Qwen35GgufInstance::replay_prompt`, which both
`generate` and `profile_decode` go through so a profiler can never warm up on
a tape production would not take. The two tapes leave IDENTICAL state, gated
bit-for-bit at tiny scale, so this is a cost choice and never a behavioural
one.

**The fix for that is already built and one binding away.** `matmul_q4_dyn_reg`
(128x128 register-tiled, `dot4I8Packed`-based) landed in the kernel-performance
ledger's M5.5, measured 2.02x at `m = 32` rising to 12.56x at `m = 2048`
against the naive kernel and proven bit-identical to it - and was deliberately
left "wired nowhere yet ... once a real model dispatches this kernel enough to
need it". This milestone is that model: binding it in `model::ops::Ops` (and
adding it to `Ops::REQUIRED_KERNELS`, which every model's pipeline list must
then register) would let the Q4 tier take the chunked path too. That is a
cross-model change with its own gates, scoped to that ledger's Phase 1, and is
deliberately NOT done here - the fallback keeps Q4 exactly as fast as it was
while INT8 gets its 9.9x.

**Still not done, and still deliberately.** The flash-attention prefill kernel
M25 named (`paged_flash_prefill.wgsl`, tiles capped at `head_dim` 128 against
this model's 256) is what would let the round size stop being bounded by a
materialised score slab, and it is what a genuinely long-context prefill on
this resident needs. MTP and multi-sequence batching remain out of scope for
this resident for the reasons its own module doc gives.

### M27 (DONE): YaRN long-context RoPE scaling (`max_position_embeddings: 262144` was a dead config field, now wired)

`Qwen35Config::max_position_embeddings` was read nowhere at runtime before
this milestone - RoPE always used the plain, unscaled
`theta.powf(-2d/head_dim)` frequency regardless of position, so real usable
context was capped by whatever `BRAIN_QWEN35_CTX`/`BRAIN_QWEN35_GGUF_CTX`
happened to be set to (a serving-side VRAM/KV-cache budget), never by the
checkpoint's actual 262144-token training window.

YaRN (arXiv 2309.00071) landed as a **generic, model-agnostic** module,
`model::yarn` (`crates/model/src/yarn.rs`) - not folded into
`qwen3vl::mrope` despite that being the crate that already builds this
model's M-RoPE tables, because "any model can reuse it" is the whole point
and `crates/model` is this workspace's actual architecture-agnostic seam.
`model::yarn::scaled_inv_freq(dim, theta, &YarnConfig)` returns the
per-channel scaled `inv_freq` plus the attention-magnitude correction
(`attention_factor`, YaRN's `mscale`); `qwen3vl::mrope::mrope_tables` became
a thin wrapper over a new `mrope_tables_scaled`, which takes that pair as an
optional `(inv_freq, attention_factor)` override instead of always deriving
`inv_freq` inline from `theta` - every other `mrope_tables` caller in the
tree (qwen3vl itself, qwen3omnimoe) is unaffected, both by construction (the
`None` path is the exact original arithmetic, `attention_factor` folds in as
a literal `* 1.0`) and by a dedicated regression test pinning
`mrope_tables_scaled(.., None)` byte-identical to `mrope_tables`.

qwen35 opts in through a new `Qwen35Config::rope_scaling: Option<model::yarn
::YarnConfig>`, parsed from a checkpoint's `config.json` exactly like
`mrope_section` already is - a `rope_scaling: {"type": "yarn", "factor":
..., "original_max_position_embeddings": ...}` key (optional `beta_fast`/
`beta_slow`/`attention_factor` overrides too); absent or any other `type`
value means `None`, i.e. today's plain unscaled RoPE. Both `mrope_tables`
call sites (whole-sequence prefill and single-step decode) now go through
`mrope_tables_scaled` via a `Qwen35Config::yarn_scaling()` helper. GGUF
import (`gguf_import.rs`) sets `rope_scaling: None` unconditionally -
llama.cpp's own rope-scaling KV convention is not read yet, so a GGUF
checkpoint stays unscaled until that is wired too; the HF-style
`config.json` path was this milestone's scope.

Regression proof that an unconfigured checkpoint (every checkpoint that
exists today) is untouched: `golden_parity.rs`'s real-`transformers`
reference comparison still passes at the same tolerance, and the new
`crates/qwen35/tests/yarn_rope_scaling.rs` adds a config with
`rope_scaling: Some(YarnConfig::new(1.0, ..))` - the YaRN code path is
genuinely exercised, just requesting no real extension - and asserts its
`logits_all` output is bit-for-bit identical to `rope_scaling: None`. The
same file proves the scaling actually does something: a `factor = 3.0`,
`original_max_position_embeddings = 6` config decoded across `tiny()`'s
24-token `block_size` measurably diverges from the unscaled baseline once
past position 6, and a full decode run under real scaling never panics or
produces a non-finite hidden state.

## Not yet done


- **M18 follow-up: `stream_train_real.rs` asserts on loss only, never on the
  generations it prints.** That file's single test
  (`short_real_streaming_lora_finetune_reduces_loss_and_shifts_generation`)
  prints a BEFORE and an AFTER greedy generation from the real 27B
  checkpoint, but its only gates are that the losses are finite
  and that the last one is below the first - the test's own name promises a
  generation shift that nothing in it checks. Closing it needs one of: an
  expected-phrase check specific to `resources/qwen35_finetune/corpus.txt`'s
  content, or a perplexity-on-held-out-corpus-lines comparison
  (BEFORE vs AFTER) - both of which require encoding domain knowledge of that
  corpus that this file does not currently carry. Blocked on this box either
  way: the test is `BRAIN_QWEN35_DIR`-gated on the ~54 GB HF-safetensors FP8
  checkpoint, which is not present here (only the community GGUF is) and
  cannot be fetched - `df -h /` on this box moves between 41 GB and 58 GB
  free as the build tree churns, i.e. never reliably more than the download
  alone, never mind headroom beside it. A change to the assertions cannot be
  validated against weights that are not here, so the test is left as-is and
  the gap recorded instead.

M0-M20 are complete; M21's real-checkpoint output-correctness gate was RED
from M21 through the start of M23 and is now GREEN - see M23's own section
above for the fix and M24's for the Q4 (W4A8) tier work that followed it.
`gguf_reference_parity_real.rs`'s CPython oracle re-derivation is still
honestly RED, tracked as M23 follow-up (see that test file's own doc).
Otherwise, remaining scope is the recorded gaps below, none of which are
achievable on the ORIGINAL development machine (no discrete GPU, 18 GiB
usable RAM) this ledger was written against, plus M14's/M15's/M16's/M17's/
M18's/M19's own "not done" items just above. M20 was validated on a
different box (2x Tesla P40, 48 AVX2 cores, 184 GiB RAM) - see its own
section for what that box could and could not do.

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
- MTP head: structurally implemented, **no reference oracle** - and this is
  now CHECKED rather than assumed: the published `modeling_qwen3_5` and
  `modeling_qwen3_5_moe` reference modules do not implement the MTP head at
  all, they drop it on load (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`),
  so no amount of reference-module access closes this gap; only a real
  checkpoint's own MTP predictions could. The head is therefore
  gradchecked and overfit-tested, never parity-claimed. M17 landed the
  first real-weight import (`crate::import::import_mtp`) and wired it into
  a real greedy-decode speedup (`crate::stream::generate`'s `use_mtp`
  path) - both still without any external numerical reference to validate
  the head's own predictions against (the "no reference oracle" gap is
  about MTP's OWN correctness as a predictor, not about whether the
  overall decode loop it feeds into stays correct - M17's gate 1 proves
  the latter unconditionally, by causality, regardless of the former).
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
- `reasoning_effort` (xhigh/medium/low), `tool_choice` and `preserve_thinking`
  ARE now wired into `caps.rs` and the shared chat path
  (`qwen3::chat::parse_request`, reused by `qwen35moe`), each validated
  against the REAL Qwen3.8 chat template (the resources dir
  `tokenizer_config.json`, executed through `data::chat_template` and
  cross-checked against the hand-port in `data`'s
  `chat_template_cross_check.rs`, `matches_qwen_chat_qwen38_*`). The
  hand-ported Qwen3.8 template flavor itself
  (`data::qwen_chat::TemplateFlavor::Qwen38`, render-exact and
  cross-validated) IS now wired end-to-end: `template_flavor` is a real
  request param (`qwen3.8` default for this model, `qwen3` opt-out),
  `parse_request` renders the flavor (making `preserve_thinking` live at
  serve time), `ParsedRequest` carries the flavor plus the prefilled-open-
  `<think>` state, `SeqState`'s scanner starts reasoning-open and parses the
  3.8 XML `<function=...>` wire form, and `apiserve` forwards
  `template_flavor` from the OpenAI chat surface (`chat_template_kwargs`
  nested or top-level). This crate's two tokenizer entry points
  (`caps.rs`'s GenerateAction and the GGUF resident instance) inject the
  `qwen3.8` default before the shared parse, so the model is served by its
  own template without asking.
- No cross-pass persistent weight cache in `stream::generate`'s decode loop
  (M19's own investigation) - this box's usable RAM is smaller than the
  checkpoint's on-disk footprint, so the win is capped by disk I/O regardless
  of caching policy; a genuinely bigger-RAM box is the real prerequisite, not
  more code here.

Never write an intermediate full-precision whole-model file (~108 GB) - quantized
device buffers must be built directly from the compressed FP8 checkpoint, same
constraint as qwen35moe.
