# Qwen3-Omni-30B-A3B — status ledger

Chronological, measured-only. Development box: 2× Tesla P40 (24 GB, Pascal,
non-ReBAR) + Xeon E5-2690 v3 ×2 (48 threads, AVX2, no VNNI/AMX) + 184 GB RAM,
**no NPU**. Weights staged on the 93 GB tmpfs at `/tmp/.X11-unix/brain` during
development (see "Facts" below); the model store proper is
`$BRAIN_MODELS_DIR`.

## Facts (2026-08-07)

- `Qwen/Qwen3-Omni-30B-A3B-Instruct`: `Qwen3OmniMoeForConditionalGeneration`,
  35.25 B params, 70.5 GB bf16, 28 010 tensors across sharded safetensors
  (`model.safetensors.index.json`, `total_size` 70 519 637 090 B).
- Tensor-count breakdown by top-level prefix: `thinker.model` 18 866,
  `talker.model` 7 942, `thinker.audio_tower` 525, `thinker.visual` 351,
  `code2wav.decoder` 118, `code2wav.pre_transformer` 89,
  `talker.code_predictor` 86, `code2wav.upsample` 22, small heads/projections
  (`talker.hidden_projection`, `talker.text_projection`, `code2wav.code_embedding`,
  `talker.codec_head`, `thinker.lm_head`) 11 total.
- Config confirmed (dumped from the released `config.json`, see
  `readme.md` for the architecture table): Thinker text 48L/hidden 2048/
  128 experts top-8/moe_inter 768/no shared expert/`use_qk_norm`/vocab 152064/
  M-RoPE `[24,20,20]` interleaved/theta 1e6. Audio tower 32L/d_model 1280/
  20 heads/ffn 5120/128 mel bins/`n_window_infer` 800/out 2048. Vision tower
  depth 27/hidden 1152/16 heads/inter 4304/patch 16/temporal_patch 2/
  `spatial_merge` 2/DeepStack `[8,16,24]`/`gelu_pytorch_tanh`/out 2048. Talker
  text 20L/hidden 1024/128 experts top-6/shared_expert 768/vocab 3072 (codec
  ids)/`accept_hidden_layer` 24. Code predictor 5L/`num_code_groups` 16/
  vocab 2048. Code2Wav: RVQ 16 quantizers (1 semantic@4096 + 15 acoustic@2048),
  8L sliding-window(72) GQA pre-transformer hidden 1024/inter 3072, ConvNeXt
  upsample `[2,2]`, SEANet decoder `[8,5,4,3]`, total upsample 1920 → 24 kHz
  (12.5 Hz code rate). Speakers `chelsie:2301 ethan:2302 aiden:2303`.
- Preprocessing confirmed: Whisper mel (128 bins, n_fft 400, hop 160, 16 kHz
  sampling, `n_samples` 4 800 000 max); Qwen2VL image processor (`smart_resize`,
  mean/std 0.5, `min_pixels` 3136 / `max_pixels` 12845056, `merge_size` 2).
- License: `license:other` (Qwen custom), not gated, not private.
- Memory: int8-quantized full parameter count (~35 GB) exceeds a single P40's
  24 GB; brain's documented non-ReBAR 2× resident-per-buffer cost
  (`docs/lessons.md` §14) would push that to ~70 GB against 48 GB total across
  both cards if unaddressed — this is why M1 (below) gates M2 onward.

## Plan of record

`/home/user/.claude/plans/plan-complete-and-feature-woolly-raven.md` (17
milestones: M0 goldens → M1 VRAM investigation → M2 sparse MoE core → M3
config/import → M4 audio tower → M5 vision tower → M6 thinker → M7 talker →
M8 code2wav → M9 residency wiring → M10 D-Bus → M11 OpenAI → M12 Anthropic →
M13 brain-py → M14 examples/omni.py → M15 NPU → M16 profiling → M17 testdata
cleanup).

## Done

- **M0 — facts + docs skeleton** (2026-08-07). This ledger, `readme.md`,
  `.todo/qwen3-omni.md`, `docs/manifest.txt` entry. Config dumped and cross-checked
  against the transformers reference implementation
  (`transformers/models/qwen3_omni_moe/{modular,modeling}_qwen3_omni_moe.py`,
  installed version 5.14.1) — `Qwen3OmniMoeCode2Wav.forward`/`chunked_decode`
  and `Qwen3OmniMoeForConditionalGeneration.generate` read in full to confirm
  the Talker's `_get_talker_user_parts`/`_get_talker_assistant_parts` embedding
  splice and the `chunked_decode(chunk_size=300, left_context_size=25)`
  contract before any Rust is written, per the porting playbook §0-1.

- **M1 — 2× resident-VRAM investigation, RESOLVED** (2026-08-07).
  `crates/gpu-core/tests/vram_overhead.rs` measured the doubling directly via
  `nvidia-smi` deltas around known allocations (P40 ×2): allocation alone is
  1.00x; any upload under the **default wgpu backend** is exactly 2.00x,
  independent of `COPY_SRC`/`COPY_DST` usage flags and independent of upload
  chunk size (added `write_at`/`write_f32_chunked` to `backend_api::Backend`
  and all three native backends to test chunking — did not help, ruling out
  "a staging belt sized to the biggest call"). The same probe on **brain's own
  native Vulkan backend** (`crates/backend-vulkan`, `--device vulkan`)
  measured a clean **1.00x** — its `with_staging` reuses one shared, bounded
  staging buffer; wgpu-core's does not get freed. Full write-up:
  `docs/lessons.md` #34.
  **Decision: `--device vulkan` for the Omni residency path.** int8 thinker
  (~30 GB) now fits GPU-resident, sharded across both P40s at true 1x
  (~15 GB/card via `crates/qwen/src/shard.rs`) — the hybrid CPU-expert fallback
  in the plan of record is not needed. The native Vulkan backend's coop-matrix
  kernel is unavailable on this box (no `glslc`/`glslangValidator`, scalar
  fallback used), so M16 must measure real throughput on `--device vulkan`
  before any change to the *default* backend elsewhere in brain — this
  decision is scoped to Omni's placement, not an engine-wide default flip.
  M2 (sparse MoE) proceeds targeting GPU residency for both Thinker and Talker
  experts.

- **M2a — sparse MoE core, forward pass, fp32** (2026-08-07). `model::moe`
  (`crates/model/src/moe.rs`): `router_fwd` (reuses `router_gate.wgsl`
  unchanged — Thinker/Talker both use plain softmax top-k, not glm's
  sigmoid/bias/group-limited variant) + `expert_fwd`, a per-expert step built
  on one new kernel, `moe_linear_gated.wgsl` (`matmul.wgsl` with a per-row
  gate early-exit before the K-reduction — skips a non-routed row's FLOPs
  instead of computing and discarding it). Bumped `router_gate{,_train}.wgsl`'s
  `MAX_EXPERTS` 64→128 (Omni's Thinker and Talker are both 128-expert).
  Verified exact (max_abs_diff < 1e-5, effectively bit-identical) against a
  dense oracle built from the SAME kernels `crates/glm`'s `Mlp::Moe` arm uses
  (`matmul`→`silu_mul`→`matmul`→`scale_add`, evaluated densely) — on all three
  backends (wgpu, native Vulkan, CPU JIT) —
  `crates/model/tests/moe_sparse_parity.rs`.
  Pre-existing, unrelated finding while running `make clippy`: this branch
  carries 192 warnings against a 183 baseline with EVERY change in this
  workstream stashed out (`crates/cli/{resident_forecast,resident_llm,
  resident_mock,splat_cli,supply,wm_cli}.rs`) — confirmed not caused by this
  work; not in scope to fix here, noted for whoever picks it up.

- **M2b — sparse MoE core, int8** (2026-08-07). `expert_fwd_i8` + one new
  kernel, `moe_linear_gated_i8.wgsl` — the int8/DP4A counterpart of
  `moe_linear_gated.wgsl`, deliberately in the SAME naive (one thread per
  output element, no workgroup tiling) tier rather than gating
  `matmul_i8_dyn`/`matmul_i8_gemv`: those stage rows into workgroup-shared
  memory across a `workgroupBarrier()`, and a per-thread early return that not
  every thread in the workgroup reaches ahead of that barrier is undefined
  behaviour in WGSL. A safely-gated TILED int8 kernel needs row compaction —
  the same gather/prefix-sum work M2a already deferred, for the same
  atomics-forbidden reason. Reuses `model::int8::quantize_weight` (weight
  packing, unchanged) and `quant_rows_steps` (dynamic per-token activation
  quantization, unchanged) — the shared input `x` is quantized ONCE per layer
  before the expert loop (every expert reads the same `xq`/`sx`); the
  post-SiLU hidden `h` is necessarily requantized per-expert since it differs
  per expert. Verified against the fp32 sparse path (M2a) at rel-L2 0.0084
  (asserted < 0.02) on a random synthetic MoE —
  `crates/model/tests/moe_sparse_i8_parity.rs`. Also on CPU (Cranelift JIT):
  passes, which means `dot4I8Packed` and this whole kernel (no barriers) ARE
  CPU-JIT-compatible — corrected the kernel's `@cpu` metadata from an
  over-cautious `no` (copied from `matmul_i8_dyn`'s genuinely-barriered case)
  to `yes` after confirming.
  **Still deferred**: row-compaction/gather-scatter (both tiers), a TILED
  int8 kernel, backward pass (M2c), `crates/glm` migration (M2d).

- **M3a — `crates/omni` config** (2026-08-07). New crate, `omni::config`:
  `AudioConfig`/`VisionConfig` (same shape as `qwen-asr`/`qwenvl`'s configs,
  at Omni's scale), `MoeTextConfig` (shared by Thinker `thinker_config.text_config`
  and Talker `talker_config.text_config` — both plain-softmax top-k, not
  glm's sigmoid/bias/group-limited router), `ThinkerConfig`, `TalkerConfig`
  (reuses `tts::config::MtpConfig::from_json` unchanged for
  `code_predictor_config` — same path, same shape), `Code2WavConfig` (same
  shape as `codec::config::CodecConfig`, extended for Omni's wider
  pre-transformer and mean-pooled multi-codebook input), `OmniConfig`.
  Cross-checked two ways: an inline structurally-faithful sample (unit tests,
  no checkpoint needed) AND a real-checkpoint test
  (`crates/omni/tests/real_config.rs`, `BRAIN_OMNI_HF_DIR`-gated, skips
  cleanly without weights) run against the actual downloaded
  `config.json` — every field in "Facts" above reproduced exactly from the
  real file, not a hand-copied sample.
  Added `brain-qwenvl` and `brain-omni` to `[workspace.dependencies]`
  (`qwenvl` had never been depended on by anything before this).

- **M3b (part 1) — the `family_of_architecture` routing landmine, fixed**
  (2026-08-07). `Qwen3OmniMoeForConditionalGeneration` contains `"qwen"` as a
  substring, so `modelstore::plan::family_of_architecture`'s naive
  first-match scan would have silently routed an Omni fetch to the dense
  `qwen` HF importer (which would then fail, or worse, partially import a
  family it cannot represent). Fixed by checking `"omni"` before `"qwen"` in
  the scan order; regression test
  (`family::tests::omni_architecture_does_not_fall_through_to_qwen`) asserts
  both the real HF class name and the plain-Qwen3 case it must not affect.
  `cli::supply::convert` gained an explicit `"omni"` arm (a clear
  "not wired yet" error, distinct from the drift-detection catch-all it would
  otherwise have hit) — `cli::model_dir::resident_for`'s existing catch-all
  was already honest for an unrecognized family, no change needed there.

- **M3b — HF -> brain name mapper** (2026-08-07). `omni::import`: pure
  `&str -> Option<String>` mapping functions per component
  (`map_audio`/`map_vision`/`map_thinker`/`map_talker`/`map_code_predictor`/
  `map_code2wav`, composed as `hf_to_brain`), following
  `qwenvl`/`qwen_asr`'s existing naming conventions so M4/M5's shared-encoder
  hoist is a config bump, not a second copy. `code_predictor` maps by
  IDENTITY (no rename) since `omni::config::TalkerConfig` already reuses
  `tts::config::MtpConfig::from_json` unchanged — matching names means M7 can
  load it with `tts::mtp` directly. Audio's `self_attn.{q,k,v}_proj` fuse
  into one `qkv` tensor (`fuse_audio_qkv`), matching
  `qwen_asr::import::map_audio_encoder`'s existing fused layout.
  Validated three ways: (1) unit tests against real tensor-name samples from
  the checkpoint; (2) `crates/omni/tests/real_import_coverage.rs`
  (`BRAIN_OMNI_HF_DIR`-gated) — **every one of the real checkpoint's 28 010
  tensor names maps successfully with zero brain-name collisions**, checked
  directly against `model.safetensors.index.json`, not a hand-picked sample;
  (3) real tensor SHAPES pulled from shard 15 independently confirm the
  config assumptions — `code2wav.code_embedding.weight` is `[32768, 1024]`
  = `codebook_size(2048) * num_quantizers(16) x hidden_size(1024)`, exactly
  as `Code2WavConfig` models it; `pre_transformer` mlp/attn shapes match too.

## Disk-capacity finding, and how it was resolved (M3c)

`checkpoint::weightio::StWriter::write` was f32-only — every existing
importer (`qwen`, `glm`, `lfm`) writes an on-disk brain checkpoint at f32,
then quantizes to int8 transiently at RESIDENCY-LOAD time (one tensor of
host f32 at a time, dropped after upload — see `crates/qwen/src/q8.rs`
`Q8::build`, `crates/zimage/src/block.rs`). For Qwen3-Omni's 70.5 GB bf16
checkpoint, that convention would mean an on-disk f32 checkpoint of **~141 GB**
(measured: `70 519 637 090 * 2`) — fitting on NEITHER of this box's
filesystems (93 GB tmpfs, 71 GB free on the 296 GB overlay), and not fixable
by streaming harder (the ~141 GB DESTINATION file itself needs one
filesystem that size, regardless of how the source is read).

**Resolved by extending `checkpoint::weightio` for int8-native storage**
(user's explicit choice over building against the old f32 convention and
deferring the disk problem): `Dtype` enum (`F32`/`U32`), `StWriter::create_mixed`
(per-tensor dtype in the plan; `create` is now a thin `Dtype::F32`-everywhere
wrapper, byte-identical to its pre-existing behavior — every current f32
importer is untouched) and `StWriter::write_u32` (writes
`model::int8::quantize_weight`'s packed output as-is, no repacking).
Verified independently via the raw `safetensors` crate (not brain's own
reader, so the test cannot pass merely because a writer and a reader share a
bug) — `crates/checkpoint/src/weightio.rs`
`create_mixed_writes_packed_int8_alongside_f32_scale`. Full checkpoint suite
(52 tests) and a full workspace rebuild both stayed green.

`omni::import::import_as` now streams the real orchestration: HF -> brain
name mapping (M3b) + the audio qkv fuse (buffered per-layer across
out-of-order arrival — a real bug was caught and fixed here: naively calling
`.take()` inside a tuple pattern unconditionally clears the field even when
the pattern doesn't match, silently discarding a partial q/k/v arrival) +
`should_quantize` (every rank-2 weight with a K dimension divisible by 4 —
attention/expert/router/embedding projections — gets int8; norms, biases,
layer-scales, and anything not meeting that shape stay f32, automatically,
not by a per-family special case) + `checkpoint::weightio::StWriter::create_mixed`/
`write_u32`. Result: **~35 GB on-disk checkpoint**, fits the 93 GB tmpfs.
Tested end to end against a synthetic HF checkpoint (small but structurally
real, matching `tts::import`'s own precedent) —
`crates/omni/src/import.rs` `streams_qkv_fuse_and_quantizes_2d_weights`:
verifies the qkv fuse concatenation order, that a quantized tensor's packed+
scale bytes are fewer than its f32-equivalent AND dequantize within
`quantize_weight`'s own documented tolerance, that a 1-D tensor is kept exact
f32 with no scale sibling, and that `code_predictor` reaches the output with
an unchanged name — all via the raw `safetensors` crate, independent of
`checkpoint::weightio`'s own reader.
**Not yet done**: a full 70 GB source -> ~35 GB int8-native run against the
real checkpoint (would need the remaining 11 of 15 shards, ~55 GB more,
downloaded — not attempted this session; the mechanism is proven on real
partial data (`real_import_coverage.rs`, all 28 010 real names) and on a
synthetic full pipeline, which is the honest boundary of what this session
covers).

- **M0 (finished) — real goldens, all 6 components, against the actual
  checkpoint** (2026-08-07). `tools/goldens/omni_dump_reference.py` ran end
  to end against real weights (shard 1 has `audio_tower`+`visual`; shard 15
  has `code2wav`+`code_predictor`) — every component produced finite,
  non-NaN activations. Four real bugs found and fixed along the way, each
  informative about the reference implementation, none affecting brain's own
  Rust design:
  - `layer0`: the checkpoint stores one `gate_proj`/`up_proj`/`down_proj` per
    expert (matching `model::moe`'s own per-expert design exactly — no brain
    change needed), but the transformers module class wants ONE stacked
    `gate_up_proj [E, 2*ff, d]` / `down_proj [E, d, ff]` parameter — a
    transformers-internal loading convention `from_pretrained` applies
    automatically that a raw `state_dict` load must replicate by hand
    (`fuse_experts` in the dumper). Router logits/scores/indices come from
    hooking `mlp.gate` (`Qwen3OmniMoeThinkerTextTopKRouter`), not `mlp`
    itself — `Qwen3OmniMoeThinkerTextSparseMoeBlock.forward` returns only the
    combined hidden state. Verified sane: top-8 expert weights sum to 1.0
    (`norm_topk_prob`).
  - `vision`: `BaseModelOutputWithDeepstackFeatures`'s real field is
    `deepstack_features`, not the guessed `deepstack_feature_lists`.
  - `rope`: the `__new__`-bypass stub (skips allocating the full module tree
    to compute `get_rope_index` with no weights) also needs
    `spatial_merge_size` set by hand — `get_rope_index` reads
    `self.spatial_merge_size` directly, not through `self.config`.
  - `talkcp`: the code predictor's `forward` only consumes `inputs_embeds`
    directly on its "prefill" branch, which triggers on
    `inputs_embeds.shape[1] > 1` — a single-step `[1,1,h]` call falls through
    to a `generation_steps`-indexed `input_ids` embedding lookup instead
    (and crashes with `input_ids=None`). Fixed by calling with the smallest
    real prefill shape, `[1,2,h]` (hidden + codebook-0 embedding, predicting
    codebook-1).
  Goldens (756 KB total, 6 files) copied to
  `/data/workspace/resources/brain-goldens/omni/` (the `BRAIN_GOLDEN_MIRROR`
  default) and wired into `scripts/data/fetch-testdata.sh` (`golden_tree
  "omni" "golden/omni"`, alongside the existing `lfm`/`qwen`/`vae`/`zimage`/
  `esrgan` entries) — `make fetch/testdata` now restores
  `testdata/golden/omni/` for anyone, not just this box.

- **M4 — audio tower (AuT), EXACT parity** (2026-08-07). Confirmed the
  reuse the plan called for is complete as-is, with zero new encoder code:
  `qwen_asr::config::AudioEncoderConfig::qwen3_omni()` (a new preset —
  `num_mel_bins`/`downsample_hidden`/`output_dim`/`n_window`/`n_window_infer`/
  `max_pos` are IDENTICAL between Qwen3-ASR and Omni's audio tower, confirmed
  against the real config; only `d_model`/`n_heads`/`ffn_dim`/`n_layers`
  differ) plugged directly into the existing, unmodified
  `qwen_asr::encoder::AudioEncoder`. Real weights (shard 1) streamed via
  `checkpoint::mmap::MmapSafetensors` (selective single-shard reads — the
  full 15-shard `WeightReader::open_hf_dir` can't open with only 4 of 15
  shards on disk) through the same `hf_to_brain`/qkv-fuse path `import_as`
  uses, run through `AudioEncoder::encode` on the SAME `golden_mel` input
  formula the Python dumper used, compared against the real golden's
  `hidden` (the audio embeds, post-projector) —
  `crates/omni/tests/audio_parity.rs`:
  **cosine 1.000000 / max_abs 0.000002 (wgpu), cosine 1.000000 / max_abs
  0.000001 (CPU JIT)** — real bit-level parity, not a synthetic shape check.
  `qwen_asr`'s own test suite stayed green (the new preset added no risk to
  the existing Qwen3-ASR path).

- **M5 — vision tower, EXACT parity (image path; video not yet covered)**
  (2026-08-07). `qwenvl::config::VisionConfig::qwen3_omni()` (a new preset —
  `gelu_pytorch_tanh` was already `qwenvl`'s own activation choice, no kernel
  change; `num_position_embeddings` isn't a literal field in Omni's released
  config — it expresses `image_size`/`patch_size` instead — but derives to
  the identical 2304, confirmed against the real checkpoint's
  `thinker.visual.pos_embed.weight` shape `[2304, 1152]`) plugged into the
  existing, unmodified `qwenvl::encoder::VisionEncoder`/`PatchMerger`.
  **One real naming bug found and fixed**: Omni's HF merger tensors use
  `ln_q`/`mlp.{0,2}` (an `nn.Sequential(Linear, GELU, Linear)`, so index 1 is
  the weightless activation), not Qwen3-VL-4B's `norm`/`linear_fc1`/
  `linear_fc2` — both are the same LayerNorm->Linear->GELU->Linear shape, so
  `omni::import`'s `merger_leaf` now maps both onto `PatchMerger`'s actual
  target keys (`ln`/`fc1`/`fc2`); `patch_embed.proj.*`/`pos_embed.weight`
  also needed the same segment-stripping `qwenvl::import` already does, to
  `patch_embed.*`/`pos_embed`.
  Verified against real weights (shard 1) via the same
  `checkpoint::mmap::MmapSafetensors` selective-read pattern
  `audio_parity.rs` established —
  `crates/omni/tests/vision_parity.rs`: **cosine 1.000000 on the raw
  per-patch encoder output (max_abs 0.012, on values with std ~15 — a real
  bit-level match, not a synthetic shape check) AND on all three DeepStack
  taps (max_abs 0.000003–0.000007) — on both wgpu and the CPU JIT.** Along
  the way, a golden-shape discrepancy clarified the reference model's own
  boundary: `Qwen3OmniMoeVisionEncoder.forward`'s `last_hidden_state` is the
  RAW pre-merger ViT output, while `deepstack_features` are ALREADY merged
  internally — two different stages returned by the same call, not the same
  stage twice. The primary `PatchMerger` ran on real weights too (shape +
  finite-output checked) but has no golden of its own to compare against yet
  (this golden's "hidden" is pre-merger, per the above).
  `qwenvl`'s own 38-test suite stayed green.
  **Video (t>1) is explicitly NOT covered** — `VisionEncoder::encode_with_taps`
  is documented single-frame (`t=1`) only; temporal patching for video is
  real remaining work, not yet started.

## M6 design note (investigated, not yet implemented)

Before writing any Thinker decoder code, checked how `crates/tts`'s Talker —
the one other model in brain declaring M-RoPE with `interleaved: true,
mrope_section: [24,20,20]`, the EXACT same shape as Thinker's — actually
handles it, since getting this wrong silently would be a very expensive bug
to chase later. Answer, already proven in production
(`crates/tts/src/talker.rs`'s own doc comment): for any token stream where
all three M-RoPE axes (temporal/height/width) carry the SAME position index
— true for pure text, and true for a pure audio stream — `interleaved`
M-RoPE with a 3-way section split **collapses exactly to `qwen`'s ordinary
half-split RoPE** (`rotate_half`, θ = 1e6). No new kernel, no interleaving
math, for that case. The Talker reuses `qwen::Qwen` (`tie_embeddings=false`,
`enable_mrope()`) directly and gets this for free. My own M0 `layer0` golden
is pure text (9 tokens, no image/audio), so `Qwen3OmniMoeThinkerTextModel`'s
own default position handling (confirmed by reading `forward`'s source: no
`position_ids` passed → `arange(seq_len)` broadcast to all 4 axes) means
**this golden's attention needs only plain `qwen`-style RoPE too** — the
real M-RoPE (temporal/height/width genuinely diverging across an
image/video/audio span) only has to be built for mixed-modality prompts,
which is a real but narrower piece of work than "M-RoPE from scratch."

What's genuinely new for M6, given the above: `qwen::Qwen`'s internals
(sharding, LoRA, int8, KV-cache all interleaved in one large constructor —
`crates/qwen/src/model.rs`) are not modular enough to swap out just its
dense SwiGLU MLP for `model::moe`'s sparse one without real surgery, unlike
`crates/glm`, which already carries an `Mlp::Dense`/`Mlp::Moe` enum seam at
exactly this point (evaluated densely today, per M2's own status). Two
honest paths, neither started: (a) give `qwen::Qwen` the same `Mlp` seam
`glm` has, so Thinker (and TTS's own future MoE Talker, M7) can reuse the
attention/RoPE/QK-norm/KV-cache machinery that's already correct and tested,
switching only the FFN call; or (b) a new, leaner decoder in `crates/omni`
built directly from `model::block`'s shared primitives
(`rmsnorm_fwd`/`rope_fwd`/`gqa_fwd`) + `model::moe`, forward-only, no
sharding/LoRA/KV-cache — closer to `qwenvl::parity`'s minimal partial-depth
test harness than to production `qwen::Qwen`. (a) is more in the spirit of
"evolve core, don't duplicate" but touches a large, heavily-relied-on file;
(b) is faster to land correctly but is a second (if leaner) forward
implementation of the same attention math. Real M-RoPE (the
image/video/audio-divergent case) and the `model::vlm::splice_fwd`
multimodal splice are additional, separate pieces on top of either.

- **M6a — Thinker decoder layer, exact parity, pure-text path** (2026-08-07).
  `crates/omni/src/thinker.rs`: `layer_fwd`, path (b) from the design note
  above — a new, leaner forward built directly from `model::block`'s shared
  primitives + `model::moe`, not a `qwen::Qwen` modification. Real-weight test
  `crates/omni/tests/thinker_layer_parity.rs` (layer 0, 9-token pure-text
  prompt) now passes on both `--device vulkan` and `--device cpu`, checked at
  every stage: `xmid` (post-attention residual), router logits, the dense
  post-topk-renorm gate, the routed expert-id SET, and the full layer output —
  **cosine 1.000000, max_abs ≤ 2e-6 on all four**, plus a fully independent
  host-side (no GPU kernels) recomputation of one row's MoE combine, used as a
  third witness during debugging.

  Two real, load-bearing bugs found and fixed on the way to that result — both
  pre-existing before this milestone, exposed because Thinker is the first
  brain model with `n_experts` in the 65-128 range:

  1. **`crates/omni/src/thinker.rs`'s own pipeline table wired `rope` to
     `kernels::ROPE`** (`rope.wgsl` — interleaved adjacent-pair rotation,
     hardcoded base 10000) **instead of `kernels::ROPE_BASE`**
     (`rope_base.wgsl` — half-split `rotate_half`, host-supplied `theta`) —
     the kernel `qwen::Qwen` and this module's own doc comment both rely on
     for the M-RoPE-collapses-to-plain-RoPE case. A copy-paste-shaped mistake
     local to this new file, not a pre-existing engine bug. Symptom: `xmid`
     cosine 0.951 (a small-but-real divergence, not a wipeout — still a valid
     rotation, just the wrong one).
  2. **`crates/kernels/wgsl/router_gate.wgsl` and `router_gate_train.wgsl`
     had a stale fixed-size array**: `MAX_EXPERTS` was bumped 64→128 (per this
     workstream's own earlier commit) but the shader-local `var prob: array<f32,
     64>` / `var used: array<bool, 64>` were never bumped to match, so every
     expert index ≥ 64 wrote out of bounds during the softmax/top-k/renorm
     loop — silently wrong for the top half of a 128-expert router (WGSL has
     no bounds-checking panic; out-of-bounds writes just corrupt whatever is
     adjacent). Fixed to `array<f32, 128>` / `array<bool, 128>` in both files,
     `make kernels-regen && make kernels-table` run after. This is a
     pre-existing engine-level bug (not new to M6) that would have silently
     miscomputed routing for **any** ≥65-expert MoE model using
     `router_gate.wgsl`/`router_gate_train.wgsl` — Thinker/Talker are the only
     current callers at that scale, and Talker (M7) was not yet exercised, so
     this is caught before it could affect a shipped result. `crates/glm` is
     unaffected (uses the separate `router_gate_sigmoid.wgsl`). No existing
     unit test caught it because `crates/model/tests/moe_sparse_parity.rs`
     uses a synthetic `n_experts: 8` shape (well under 64).

  A third apparent divergence (after both fixes, `out` cosine 0.76 against
  the golden's `hidden` tensor) turned out to be a **test/golden bug, not a
  code bug**: `Qwen3OmniMoeThinkerTextModel.forward` always applies its
  top-level `self.norm` (the final decoder-stack RMSNorm) after the layer
  loop, even truncated to 1 layer for the golden dump — so `last_hidden_state`
  is `model.norm(layer0_output)`, not layer 0's own raw output that a single
  `layer_fwd` call actually produces. Fixed by adding a `layer_out` tensor to
  the `layer0` golden (`tools/goldens/omni_dump_reference.py`, a forward hook
  on `model.norm`'s input) and comparing against that instead — the right
  fix, since `layer_fwd` deliberately does not include a stack-level final
  norm (that belongs to the caller composing all 48 layers, not to one layer).

  Debugging method worth recording for the next milestone: three independent
  witnesses (GPU kernel path, a from-scratch host-side Rust recomputation
  using the same mmap'd real weights, and the golden itself) agreeing with
  each other but not with a fourth (the wrong golden tensor) is what
  localized this to "wrong comparison target," not "wrong math" — worth
  reaching for before assuming the newest code is the buggy component.

  Not yet done: 3-axis M-RoPE for genuinely divergent image/video/audio
  positions, and the `model::vlm::splice_fwd` multimodal splice — this
  milestone validates the pure-text (and, by the design note's argument,
  pure-audio) collapse case only. The full 48-layer composed loop and a real
  "describe this image" generation are also not yet built.

- **M6b — real 3-axis M-RoPE + full decoder composition** (2026-08-07).
  `crates/omni/src/thinker.rs::layer_fwd` now takes the table-driven
  `model::block::rope2d_fwd` path unconditionally (fed a `[n, head_dim/2]`
  `cos`/`sin` table from `qwenvl::mrope::{get_rope_index, mrope_tables}` —
  already-implemented, already-tested code that predates this milestone and
  was simply not yet wired in), rather than keeping a separate "plain RoPE"
  analytic fast path alongside a "real M-RoPE" path that would never be
  reached. One RoPE call site now serves pure text, pure audio, and a mixed
  image/video/audio prompt alike — the M6a design note's own collapse
  argument made this the natural next step, and it retires the exact class of
  bug M6a found (a pipeline table pointing `rope` at the wrong kernel):
  there is now only one kernel to wire correctly, not two.
  `thinker_layer_parity.rs` re-verified at cosine 1.000000 on all four
  stages with the real table (previously the sequential-position analytic
  path); `model::block::rope2d_fwd` is a thin hoist of the exact dispatch
  `qwen::Qwen::rope2d_step` already uses for Qwen3-VL (not a new kernel), so
  no new device math was written.

  `crates/omni/src/thinker.rs::decode` composes `w.layers.len()` `layer_fwd`
  calls residual-to-residual, then the top-level `model.norm` `layer_fwd`
  deliberately omits — the piece that turns "one validated layer" into "the
  actual decoder stack." `thinker_decode.rs` (new, no real weights, no
  `#[ignore]`) proves the composition byte-for-byte against a hand-chained
  oracle (call `layer_fwd` N times + one more `rmsnorm_fwd`, by hand, in the
  test) on both backends — the porting playbook's tiny-config-before-real-
  weights tier. A real-weight 48-layer parity run is out of scope per the
  plan's own parity-bar decision ("component goldens + e2e smoke", not a
  full 48-layer tensor-exact reference) and is what M14's real generation
  smoke test will exercise instead.

  Multimodal splice is deliberately NOT implemented as thinker-local code:
  `model::vlm::splice_fwd` (already covered by its own direct test,
  `crates/model/src/vlm.rs`) is exposed via `thinker::SPLICE`
  (`thinker_pipelines()`'s kernel index) for a caller to invoke on the
  embedding buffer before calling `decode` — `decode` has no opinion on how
  its `x` input was assembled, matching `qwen::Qwen::write_img_embeds`'s
  contract. Wiring an actual vision/audio-embedding caller (the audio/vision
  towers from M4/M5 feeding into this) is M9's job, not M6's — M6's scope was
  the decoder + M-RoPE + splice *seam*, not a full multimodal generation
  driver.

## Not started

M2c (backward + gradcheck, deferred — see M2 note above), M2d (glm migration,
deferred), M5's video path, M7 through M17. See the plan file. M6 itself is
now believed complete (decoder + real M-RoPE + composed loop + splice seam,
all validated); the actual splice call site, a KV-cache/decode loop, and a
real multimodal generation are M9/M14's job per the design note above, not
additional M6 scope.

**Standing constraint for M9 (`OmniResident`)**: fetch/import/load must all
stay mmap-backed, streaming one tensor at a time, matching the pattern
already audited end to end for this model and already established engine-wide
(`crates/qwen/src/model.rs`'s `from_reader_inference`/`_decode`, `docs
comment: "peak host ≈ one tensor"`) —
- **Fetch**: shard-incremental (download one shard → convert+quantize →
  delete shard), already implemented this way in M3's `import_as` per the
  disk-capacity finding above; never buffer a whole shard, let alone the
  whole 70.5 GB checkpoint, in RAM.
- **Process** (M3, done): `crates/omni/src/import.rs::import_as` reads via
  `checkpoint::weightio::WeightReader`/`checkpoint::mmap::MmapSafetensors`
  (real `memmap2::Mmap`, OS page cache, header-only parse on open, one
  tensor's f32 expansion materialized per `tensor_f32` call — confirmed by
  reading `crates/checkpoint/src/mmap.rs`) and writes via `StWriter`'s
  `BufWriter<File>` with `seek`-per-tensor (`crates/checkpoint/src/weightio.rs`)
  — already the minimum-copy shape; the only unavoidable copy per tensor is
  the dtype conversion itself (bf16 -> f32, or f32 -> packed int8).
- **Load** (M9, not started): when `OmniResident` is built, open the ~35 GB
  int8 checkpoint via `WeightReader::open` (mmap) and upload one tensor at a
  time to GPU-resident buffers, exactly like `Qwen::from_reader_inference`
  does today — do not `checkpoint::safetensors::read`/`fs::read` the whole
  file into a `Vec` first. This is the point where thinker/talker/code2wav
  weight lifetime (`docs/lessons.md §14`'s per-turn build/drop schedule)
  intersects the mmap discipline: dropping a resident stage's GPU buffers
  should not require re-reading its source bytes from disk if the OS page
  cache still holds them, which is a property mmap gives for free and an
  explicit `fs::read` does not.

## Honesty notes

- No NPU device run has happened or will happen on this box (M15's scope is
  explicitly capped at CPU-side OpenVINO parity + ONNX graph validation).
- No number in this file is a projection; everything above is either read
  directly from the released `config.json`/index, or will be a `brain perf`/
  `gpu_core::profile` measurement once code exists.
- **M6a was verified with targeted tests, not a full green `make test`**
  (2026-08-07): the fast lane hung at `crates/gpu-core/tests/roofline.rs`'s
  `caps_expose_the_roofs_only_after_something_measured_them` (reproduced
  twice, including in isolation — a genuine hang, not a flake). Root cause:
  `nvidia-smi --query-compute-apps` showed ~10.7 GB of GPU memory owned by
  PIDs no longer resolvable in this container, almost certainly orphaned
  Vulkan contexts from cargo-test processes killed earlier this session (one
  attempt used an out-of-range Bash-tool `timeout`, SIGTERM'd mid-run) — a
  wedged queue/fence from leaked GPU state, in a file this milestone never
  touched, not a regression from M6a's code. What M6a's own change *was*
  verified against, green on both backends: `cargo test -p brain-model -p
  brain-omni -p brain-kernels` (incl. `moe_sparse_parity`/`moe_sparse_i8_parity`),
  `thinker_layer_parity` on both `--device vulkan` and `--device cpu`
  (cosine 1.000000 on xmid/router_logits/gate/out), and `make clippy` shows
  zero new warnings in every file this milestone touched (net warning count
  dropped from +18 over baseline to +9, entirely pre-existing `crates/cli`
  debt from before this session). Re-run the full `make test` once GPU state
  clears (a fresh boot, or once whatever process is holding that memory
  exits) before relying on this as a complete regression check.
