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

- **M7a — Talker decoder layer, exact parity, real weights** (2026-08-07).
  `crates/omni/src/talker.rs`: `layer_fwd`/`decode`, [`crate::thinker`]'s
  sibling (same reasoning for why it is a new module rather than a
  `tts::talker`/`qwen::Qwen` conversion — see the module doc), plus the one
  real architectural difference: an always-active shared expert
  (`model::moe::shared_expert_fwd`, new — dense SwiGLU with its own
  intermediate width, sigmoid-gated per token, added to the routed-expert
  combine via a fresh buffer per `add2.wgsl`'s out-of-place convention).
  Verified standalone against a host-computed oracle
  (`crates/model/tests/moe_shared_expert.rs`) before real weights, per the
  porting playbook. `crates/omni/tests/talker_layer_parity.rs` (real
  weights, layer 0, 9 codec-id prompt) passes on both `--device vulkan` and
  `--device cpu` at **cosine 1.000000** on all four stages (xmid, router
  logits, gate, layer output) — first try, no debugging needed, entirely
  because M6a/M6b's bugs (wrong RoPE kernel, the `router_gate.wgsl` array
  bound, the `hidden`-vs-`layer_out` golden distinction) were already fixed
  in shared code this module reuses unchanged.
  `crates/omni/tests/talker_decode.rs` mirrors `thinker_decode.rs`'s
  tiny-config composition smoke test.

  One real bug found and fixed in `crates/omni/src/config.rs` before any of
  the above: `MoeTextConfig::talker_defaults()` had `use_qk_norm: false`.
  The real `talker_config.text_config` JSON has no `use_qk_norm` key at all
  (so this default was silently authoritative), but the reference's
  `Qwen3OmniMoeTalkerDecoderLayer` reuses `Qwen3OmniMoeThinkerTextAttention`
  verbatim, whose `q_norm`/`k_norm` are unconditional — no config flag gates
  them, and the real checkpoint carries `q_norm.weight`/`k_norm.weight` for
  every Talker layer. Caught by reading the actual attention class before
  writing `talker.rs`, not by a failing test — worth noting as the payoff of
  checking a config default against the class that actually consumes it,
  not just against the JSON key's presence.

  `tools/goldens/omni_dump_reference.py`: added `dump_talker_layer0`
  (registered as `talker_layer0`), same shape as `dump_layer0` but through
  `Qwen3OmniMoeTalkerModel` with `inputs_embeds=codec_embedding(ids)` --
  that model has no `self.embed_tokens` (real usage always assembles
  `inputs_embeds` itself: text projection + codec embedding + Thinker-hidden
  splice), so `input_ids=` would hit a missing attribute. `fuse_experts` and
  the `layer_out`-vs-`hidden` golden distinction are reused unchanged from
  the Thinker dumper.

- **M7b — code predictor, exact parity, real weights** (2026-08-07).
  `tts::mtp::MtpModel` (the standalone Qwen3-TTS MTP) reused **completely
  unchanged in forward-pass code** for Omni's code predictor, confirming the
  plan's "near-exact reuse" prediction — real Omni `code_predictor_config`
  (5 layers, 16/8 GQA heads, hidden 1024, head_dim 128, vocab 2048, 16 code
  groups, dense — no MoE) already parses through
  `TalkerConfig::from_json`'s existing `tts::config::MtpConfig::from_json`
  reuse (M3), unmodified. The only code change was a visibility bump
  (`MtpModel::build_on`: `pub(crate)` → `pub`) so a real-weight test could
  construct a model directly from mmap'd HF tensors, bypassing `ParamStore`/
  checkpoint-file I/O, the same pattern every other real-weight test in this
  workstream uses.

  `crates/omni/tests/code_predictor_parity.rs`: reads
  `talker.code_predictor.*` straight from the real checkpoint (spans shards
  14-15), renames with `tts::import::mtp_hf_to_brain` (already exactly
  correct — no new rename code needed), builds via `MtpModel::build_on`, and
  reproduces the golden's 2-position "prefill" logits
  (`tools/goldens/omni_dump_reference.py`'s pre-existing `talkcp` component,
  from M0) at **cosine 1.000000** on both `--device vulkan` and
  `--device cpu`.

  One real bug found while writing the test, in the test's own first draft
  — not in `MtpModel` or the reference: assumed the golden's `logits [2,
  2048]` had one row per *predicted codebook*; it actually has one row per
  *input position*, because the reference's `forward` applies a single
  `self.lm_head[generation_steps]` to the WHOLE hidden-state tensor
  (`logits = self.lm_head[generation_steps](hidden_states)`, broadcasting
  over the sequence dim) rather than a different head per position. Row 0
  (`lm_head[0]` applied to the Talker-hidden conditioning slot) is not a
  trained prediction target at all; row 1 (`lm_head[0]` applied to
  codebook-0's embedding) is the actual "predict codebook 1" quantity
  `MtpModel::logits` computes. Comparing against the wrong row gave cosine
  0.045 — close enough to look like a real divergence, caught by checking
  the golden tensor's actual shape (`[2, 2048]`, not `[2048]`) before
  assuming what it meant.

  A real, still-open gap found and precisely documented (not fixed here —
  it is loader-side, M9 work, not part of validating the forward pass):
  `crates/omni/src/import.rs::map_code_predictor`'s doc comment claimed
  "matching names too means M7 can load this with `tts::mtp` directly" —
  false. The importer keeps code-predictor tensor names byte-identical to
  HF (`talker.code_predictor.model.layers.0.self_attn.q_proj.weight`, …),
  but `tts::mtp::MtpModel::load_inference` expects the UNPREFIXED `blocks.N.*`
  form `tts::import::mtp_hf_to_brain` produces for standalone Qwen3-TTS —
  neither matches the other. `code_predictor_parity.rs` sidesteps this
  entirely by reading the raw HF checkpoint itself and applying
  `mtp_hf_to_brain` inline, which is why it validates correctly despite the
  gap. M9 needs either a prefix-aware `ParamStore` lookup or a dedicated
  `talker.code_predictor.*` sub-extraction (reusing `mtp_hf_to_brain`
  unchanged) before `OmniResident` can actually load this component from
  the unified checkpoint. Comment corrected in place; no behavior change.

  Still not done: `accept_hidden_layer` (Talker consumes Thinker's hidden
  state at a given layer, not just its own embeddings), `text_projection`/
  `hidden_projection`, and codec-id sampling (suppressed-token set,
  repetition penalty, temperature, top-k) — these compose Thinker+Talker+MTP
  into an actual generation loop and are M9/M14's job (residency + serving),
  matching M6's own "decoder + seam, not a generation driver" scoping.

- **M8 — Code2Wav vocoder, exact parity, real weights** (2026-08-07).
  `codec::Codec::decode_omni` (new): reuses the standalone Qwen3-TTS codec's
  pre-transformer (`transformer`, including its per-residual `LayerScale` —
  already implemented, unmodified), SEANet decoder (`causal_conv`, `snake`,
  `residual_unit`), and ConvNeXt upsample block (`convnext`) COMPLETELY
  UNCHANGED under a `CodecConfig` shape bump (`hidden_size` 512→1024,
  `intermediate_size` 1024→3072 — the plan's own prediction, confirmed).
  Two genuine (non-config) differences needed new code:
  1. **Input path**: `hidden = mean_q(code_embedding[codes[q] +
     q·codebook_size])` — ONE combined `[num_quantizers·codebook_size,
     hidden_size]` embedding table (`code_offset` is `torch.arange(...) *
     codebook_size`, not a saved weight — confirmed by reading
     `Qwen3OmniMoeCode2Wav.__init__`/`forward`), replacing `decode`'s
     per-group RVQ dequant (`quantizer.rvq_first`/`rvq_rest` + `output_proj`)
     AND `pre_conv` entirely; `pre_transformer` also has no `input_proj`/
     `output_proj` (`hidden_size` is the working width throughout — no
     separate `latent_dim` split). New `code_embedding_mean` composes the
     already-existing `EMBED` gather (once per quantizer, host-computed
     offset) with the existing `AXPY` kernel (`out += (1/nq)·in`, newly added
     to `codec`'s own pipeline table — the kernel itself is not new, `omni`'s
     M6a work already established it exists and is trustworthy)
     accumulating the mean in place, into a zero-initialized `gpu.storage`
     buffer (confirmed zero-init is a documented cross-backend invariant,
     `backend-vulkan/src/lib.rs`'s own "zero-init like wgpu/CPU storage"
     comment).
  2. **SEANet decoder's transposed-conv crop convention**: reading
     `Qwen3OmniMoeCausalTransConvNet` directly found it crops `pad = K -
     stride` off BOTH sides of the native `ConvTranspose1d` output (`Lo =
     (L-1)·stride - 2·pad + K`), where `codec::Codec::causal_convtr` crops
     only the right (`pad = 0`, `Lo = L·stride`) — correct for the
     standalone Qwen3-TTS codec's own reference, genuinely different for
     Omni's. For the upsample stages (`K == stride`, so `pad = 0`) the two
     conventions coincide and `causal_convtr` is reused unchanged; the
     decoder's own transposed convs (`K = 2·stride`, `pad = stride`) need
     the symmetric crop, added as a new `causal_convtr_sym` helper calling
     `audio::conv::convtr1d_fwd` directly with an explicit `pad` (the
     kernel's `pad` param already implements PyTorch's native symmetric
     `ConvTranspose1d(padding=...)` semantics exactly — read `convtr1d.wgsl`
     directly to confirm before writing any Rust, no new device math).
     **This is what the golden caught**: the naive `Lo = L·stride`
     assumption produces a 15360-sample waveform; the golden (from
     `tools/goldens/omni_dump_reference.py`'s pre-existing `code2wav`
     component, M0) is 14805 samples for the same `T=8` input — derived the
     correct `(L-1)·stride` formula by hand from the reference source, then
     confirmed it lands on exactly 14805 before writing the Rust, rather
     than debugging a shape-mismatch panic after the fact.

  `crates/omni/tests/code2wav_parity.rs` (real weights) reproduces the
  golden's waveform at **cosine 1.000000, exact length match (14805 ==
  14805)**, first try. `codec::Codec::from_weights` hard-codes the CPU
  backend (`Gpu::new_cpu`, pre-existing, not an M8 change) — no Vulkan
  variant exists for this crate to cross-check against, unlike every other
  milestone's two-backend verification.

  A second loader-side naming gap found (same shape as M7b's code-predictor
  one, not fixed here for the same reason — needs a loader-design decision):
  `omni::import::map_code2wav` renames `pre_transformer.layers.N` to
  `pre_transformer.blocks.N` (matching `thinker`/`talker`'s shared
  dense-attention convention), but `codec::Codec::transformer`'s `ParamStore`
  lookups use the unrenamed HF-style `pre_transformer.layers.N.*` directly —
  neither matches the other, so `Codec::load_inference` cannot yet load
  Omni's code2wav weights from `omni::import`'s unified checkpoint output
  either. `code2wav_parity.rs` sidesteps this by reading the raw HF
  checkpoint directly, same pattern as `code_predictor_parity.rs`.

  Not done: `chunked_decode(chunk_size=300, left_context_size=25)`
  (streaming decode — `codec::decode_stream`/`streaming.rs` look like the
  right home per the plan, not attempted this session) and the two
  loader-side naming gaps (code predictor, code2wav) M9 needs to resolve
  before `OmniResident` can load either component from the unified
  checkpoint.

- **M9a — Thinker text generation: capability + residency wiring, and a real
  (if slow) generation loop** (2026-08-07). Scoped down from the plan's
  literal M9 text mid-session (see the design note below) to what is
  actually load-bearing: a genuinely working `generate` action, not just
  specs and dispatch points pointing at an unfinished loop.

  `crates/omni/src/generate.rs` (new): `thinker_forward_streaming` — one
  full Thinker forward (all `cfg.n_layers` layers + `thinker::final_norm`),
  streaming each layer's weights fresh from a `checkpoint::weightio::
  WeightReader` and dropping them before the next layer loads, instead of
  holding all 48 layers (128 experts each) GPU-resident. `generate_greedy`:
  embed → forward → `thinker::lm_head_fwd` → argmax the last row → append →
  repeat until `max_new_tokens` or an EOS id. **Deliberately validation-tier,
  not production**: no KV-cache (every new token re-runs the FULL forward
  from scratch) and no int8/GPU-sharded residency (M1's own finding — int8
  Thinker at ~30GB needs sharding across both P40s, `crates/qwen/src/
  shard.rs`'s pattern, not yet built for Thinker). Correct; slow. Two small
  additions to `crates/omni/src/thinker.rs` to support this: `final_norm`
  (factored out of `decode` so both call it — no duplication) and
  `lm_head_fwd` (`thinker.lm_head.weight`, untied — `tie_word_embeddings:
  false` in the real config).

  `crates/omni/src/caps.rs` (new): ONE action, `generate` (text prompt in,
  greedy text out) — no `converse`/`transcribe`/`speak` declared, since
  those need multimodal splice + Talker + MTP + Code2Wav chained together
  with `accept_hidden_layer`/codec sampling, none of which are wired into a
  serving-shaped loop yet. Declaring an action whose `run()` can't do what
  its spec promises is worse than not declaring it. `OmniProvider::load(dir)`
  reads directly from a real HF checkpoint directory (`WeightReader::
  open_hf_dir` + `data::qwen_tokenizer::QwenBpe::from_dir`, which already
  handles the split `vocab.json`/`merges.txt` layout this repo ships — no
  `tokenizer.json` needed) — no brain-native import step involved for this
  path.

  `crates/cli/src/resident_omni.rs` (new): `OmniResident` (`BRAIN_OMNI_HF_DIR`
  env-gated, mirroring `TtsResident`), registered in `resident.rs::
  build_executor`'s direct-block list (matching TTS/ASR's placement — the
  plan's proposed `catalog.rs`/`ModelEntry` route is for models catalog.rs
  itself iterates for HTTP surfaces without a bespoke block; Omni's shape,
  one action needing custom env config, fits the direct-block precedent
  better). `estimate()` reports the checkpoint's on-disk size as a RAM
  figure (honest about what streaming reads touch — not a VRAM budget; this
  resident does not yet participate in `docs/lessons.md §14`'s GPU-residency
  scheduling).

  Two of the three plan-listed CLI dispatch points, done: `cli::supply::
  convert`'s `"omni"` arm now calls `omni::import::import_as` instead of
  the placeholder `Err` (the importer itself was already correct, M3 — only
  the dispatch was stubbed). `cli::model_dir::resident_for`'s catch-all
  ("family not servable from the model dir yet") is already the CORRECT,
  honest answer for `"omni"` today and needed NO new arm: that function
  loads from a converted brain-native checkpoint on disk, which `OmniResident`
  does not (yet) support — it loads straight from the HF directory instead,
  a deliberately separate, env-gated path (same shape as TTS/ASR). Adding a
  same-shaped arm there would have been wiring an untrue claim, not a fix.

  **`qwenvl` registration — explicitly deferred, not "free"**: the plan's
  own text says registering `qwenvl` (today library-only) "is one caps.rs
  and one resident away, and the omni work makes it free." That undersells
  it: `qwenvl::model` has an `end_to_end_forward` (confirmed by its own test,
  `model::tests::end_to_end_forward_is_finite`), but no generation loop —
  giving it a real `caps.rs` action would mean building `qwenvl` its OWN
  tokenizer/prompt-assembly/greedy-decode loop, the same scope this entry's
  `generate.rs` just took to build for Thinker, not a thin wrapper. Deferred
  as a separate, explicitly-scoped follow-up rather than attempted inside
  an already-large M9 push.

  **Real-weight validation status**: `generate_greedy`'s LOOP correctness
  (tokenizer round-trip, sampling, EOS, layer chaining) is what carries risk
  here — every per-layer forward it calls (`layer_fwd`, `final_norm`,
  `lm_head_fwd`) is already validated exactly (cosine 1.000000, M6a/M6b).
  A real end-to-end greedy-decode comparison against
  `tools/goldens/omni_dump_generate.py` (new — deliberately kept SEPARATE
  from `omni_dump_reference.py`, whose own header commits to "component-
  scoped only, the checkpoint is too large to load wholesale even once";
  this script does exactly that on purpose, via `Qwen3OmniMoeThinkerForConditionalGeneration.
  from_pretrained(..., torch_dtype=bfloat16, low_cpu_mem_usage=True)` to
  avoid an extra fp32 double-allocation pass) requires the FULL checkpoint
  on disk — only 4/15 shards (15 GB of 70.5 GB) were present at the start of
  this milestone. Downloading the rest is a multi-hour operation at this
  connection's unauthenticated ~4 MB/s rate; see the follow-up entry below
  once that validation actually runs.

## M9 design note: why "wiring" grew into "build the generation loop"

The plan's own M9 text is "capability, residency and family wiring" —
specs + a `ResidentModel` + CLI dispatch. Taken literally that's completable
without a working generation loop at all (declare the actions, wire the
dispatch points, have `run()` return a clear "not yet implemented" error —
the exact pattern `cli::supply::convert`'s pre-M9 `"omni"` arm already used).
That was the first plan proposed for this milestone. It was changed
mid-milestone, on request, to include a real generation loop — since a
`generate` action that cannot generate is a spec with no implementation
behind it, and the per-layer math this whole workstream has been validating
since M6 had no loop yet exercising it end to end. The scope this actually
turned into: a new `crates/omni/src/generate.rs` module (streaming
per-layer forward + greedy decode), the tokenizer wiring
(`data::qwen_tokenizer::QwenBpe`, already existed, just needed the real
`vocab.json`/`merges.txt` downloaded), and a real-weight validation that
needed the full 70.5 GB checkpoint on disk (only 15 GB of it present at the
start of this milestone) — a materially larger undertaking than the
plan's own M9 estimate, discovered and negotiated in stages as each new
requirement (full download size, RAM needed to load Thinker in Python,
`ConvTranspose1d`-adjacent precision) surfaced. Recorded here so a future
reader of the plan file understands why M9's actual diff is much larger
than "wiring" suggests.

- **M10/M11/M12 — D-Bus, OpenAI, and Anthropic surfaces** (2026-08-07). All
  three turned out to already be (near-)free once `OmniResident` existed
  (M9a) — none of the plan's listed sub-items (interface-spec changes,
  new `/v1/audio/*` endpoints, image content-block handling) were needed for
  what is actually implemented (one text-only `generate` action); those
  remain real, separate work for whenever multimodal input/speech output
  are wired.

  **M10 (D-Bus): zero new code.** `crates/dbus` is fully generic —
  `Manager::serve` holds one `residency::Executor` handle and dispatches
  `Run`/`Subscribe` by whatever `(model, action)` string the caller sends,
  with no per-model code anywhere in the crate (`service.rs`'s only string
  matching is a generic legacy-alias resolver and a manifest-driven
  `transcribe_stream`-availability check, neither omni-specific).
  `resident_omni::OmniResident`'s registration in `build_executor` (M9a)
  was already sufficient. Verified by running the crate's own generic proof
  of this property, `crates/dbus/tests/roundtrip.rs`'s
  `run_roundtrips_a_result_over_an_fd` (a throwaway `RevProvider`/
  `SlowAction` test model wrapped as a `ResidentModel`, driven over a real
  D-Bus session with zero D-Bus-crate code referencing it) — still green,
  confirming the mechanism `OmniResident` also relies on.

  **M11/M12 (OpenAI + Anthropic): needed one real fix, in `crates/omni`, not
  in either HTTP crate.** `apiserve::catalog::api_caps` (the function both
  `/v1/models` listing and `/v1/chat/completions`/`/v1/messages` exposure
  gate on) classifies a model as chat-capable only when an action is
  `.streaming()`, has a `prompt`/`messages`/`text` param, and outputs
  `Media::Text` — and both handlers always populate `messages` (a flattened
  JSON-array string), never a bare `prompt`. `crates/omni/src/caps.rs`'s
  `generate_spec()` originally declared neither: not `.streaming()`, and
  only a `prompt` param. Fixed to mirror `crate::resident_mock::MockResident
  ::generate_spec()`'s proven shape exactly — `.streaming()` + `messages`/
  `prompt`/`system`/`max_new`/`temp`/`top_p`/`top_k`/`seed`/`stop`/`tools`/
  `tool_choice`/`enable_thinking` params (temp/top_p/top_k/seed/stop/tools
  accepted but not yet applied — greedy generation is deterministic and
  ignores sampling knobs; the field-level doc comments on each param say so
  explicitly, not silently). Added `last_user_text` (identical extraction
  logic to `resident_mock`'s own, kept in sync deliberately) to pull the
  last user turn out of `messages`, falling back to `prompt`. Both
  `omni::caps::GenerateAction::run` and `cli::resident_omni::OmniInstance::run`
  now call it instead of requiring a bare `prompt`.

  New `crates/omni/tests/caps_conformance.rs` (no real weights needed —
  pure manifest/spec construction) proves this against the REAL
  `apiserve::catalog::api_caps` function (added `brain-apiserve` as an omni
  dev-dependency for exactly this — no risk of the test's own logic
  silently drifting from the real gate, which a hand-rolled re-check would
  risk): `omni_manifest_is_chat_exposed` asserts `api_caps(&omni::caps::
  manifest()).chat == true`, plus `last_user_text` extraction tests and a
  param-completeness check. All green.

  Genuinely still open for a full M11/M12 per the plan's original text
  (unrelated to what M9a built, since Omni has no multimodal input/speech
  output yet either way): the pre-existing `content_text` bug that silently
  drops `image_url`/`input_audio` content parts in both `openai.rs` and
  `anthropic.rs`, and the `/v1/audio/speech`/`/v1/audio/transcriptions`
  endpoints. Neither blocks today's text-only `generate`.

- **M13/M14 — brain-py HTTP clients + `examples/omni.py`** (2026-08-07).
  `brain_py/openai.py` (new): `BrainOpenAI(BrainBase)`, a thin `urllib`
  client implementing the same `manifests`/`run`/`subscribe` primitives
  `BrainDBus`/`BrainStdio` do, so `chat()`/`generate()` work identically
  over this transport — only the `generate` action is supported (a chat
  REST API has no generic "run any action" concept the way D-Bus does;
  everything else raises `NotImplementedError` rather than silently doing
  the wrong thing). `brain_py/anthropic.py` (new): `BrainAnthropic`,
  same shape, against `/v1/messages`; `manifests()` raises
  `NotImplementedError` since Anthropic's API has no model-listing
  endpoint (`examples/omni.py` catches this and skips the pre-check rather
  than crashing).

  Two real bugs found and fixed while actually running these against a
  live server (not just unit-testing the request-building logic):
  1. **`_post`'s `with urlopen(...) as resp: return resp` closes the
     connection before the caller ever reads it** — `resp.read()` outside
     the `with` block hit an already-closed connection, silently returning
     empty. Split into `_post_json` (reads the body INSIDE the `with`
     block, returns parsed JSON) and `_post_stream` (returns the live
     connection for the caller's own iteration loop to drain and close,
     never wrapped in a `with` that would close it early). Would have
     shipped invisibly: every unit-level check of the request/header
     construction passed fine; only an actual end-to-end call against a
     running `brain serve --openai` surfaced it.
  2. **`BrainAnthropic._build_messages` dropped an embedded system-role
     message's content instead of promoting it to Anthropic's top-level
     `system` field** — only reproduced via `base.py`'s `generate(system=...)`
     path (a separate param) worked; a raw `messages=[{"role":"system",...}]`
     list built by hand silently lost the system prompt. Fixed to extract
     and join any system-role message content when a separate `system=`
     wasn't already given.

  `examples/omni/omni.py` (new): the transport × input/output matrix, honestly
  scoped — `--dbus`/`--openai URL`/`--anthropic URL` × `--in-text`/`--out-stdio`/
  `--out-text` are real; `--in-speech`/`--in-mic`/`--in-image`/`--in-video`/
  `--out-mic`/`--out-audio` are declared (so the interface won't need a
  breaking change once they're real) but `skip()` with a specific reason
  naming exactly what's missing. No `--stream`: `BrainBase`'s
  transport-agnostic `on_progress` carries `(step, total, message)`, not
  per-token delta text (that's a `BrainDBus.subscribe`-only `on_delta`
  kwarg, not part of the abstract contract this script needs to work
  identically over all three transports) — and the real Omni resident
  doesn't emit true per-token progress either (`crate::resident_omni`'s two
  `Progress::step` ticks are start/end, not one per generated token). A
  `--stream` flag that printed the literal string `"token"` N times (the
  mock's fixed `Progress::token` message field) would have been actively
  misleading, caught by actually running it rather than assuming the
  generic progress plumbing would "just work."

  Verified end to end against `BRAIN_MOCK=1` for all three transports by
  hand (D-Bus, OpenAI with a real `APIKEY`-scraped bearer token, Anthropic
  with `x-api-key`) — all three round-trip `"You said: <prompt>"` correctly.
  Registered in `tests/e2e/examples/manifest.tsv` and a new `@test` pair in
  `tests/e2e/examples.bats` (D-Bus path + the unimplemented-flag skip path);
  the OpenAI/Anthropic HTTP transports are NOT in the automated e2e suite —
  the shared harness's server doesn't start an `--openai` surface or
  capture either provider's generated API key, and extending that shared
  setup was out of scope for wiring one example in. `examples/omni/README.md`
  (new) documents all three transports plus the scope boundary.

## M9b — KV-cache decode + real multimodal input (2026-08-07)

(M15/M16/M17 in the plan file are still NPU wiring / profiling / testdata
audit, unaffected by this entry's numbering — this is a follow-up to M9's
own generation-loop scope, not a renumbering of the later milestones.)

Two pieces of `crate::generate`'s documented "not yet built" scope closed
this round, both explicitly requested ("continue wiring in everything
including kv cache" / "I want all input paths as well, not just text"):

**KV-cache.** Hoisted `qwen::Qwen`'s incremental-decode attention pattern
(`kcache`/`vcache` + `kv_append`/`attn_decode_scores`/`decode_softmax`/
`attn_decode_apply`) into `model::block` as `GqaDecodeIds`/`gqa_decode_step`/
`kv_cache_fill` — the "hoist, migrate existing users" rule's second user
(`qwen::Qwen` itself was NOT migrated this round; that is real follow-up
work, tracked below). Proven algebraically identical to the existing O(T²)
`gqa_fwd` at every position by a new CPU-backend test
(`decode_step_matches_causal_batched_attention_at_every_position`: calling
`gqa_decode_step` once per position must reproduce `gqa_fwd`'s causal-masked
row exactly, since `gqa_scores.wgsl` already masks `j > i` — passed, `< 1e-4`
elementwise). `thinker::layer_fwd` gained an additive `Option<&ThinkerLayerCache>`
param (bulk-fills the cache from a batched prefill pass, via `kv_cache_fill`'s
single flat-copy dispatch — no per-row loop needed, since the cache and the
batched K/V share the same per-row stride) and a new sibling
`layer_decode_step` (single-token decode, `model::block::rope2d_fwd` fed a
1-row M-RoPE table instead of qwen's separate `ROPE_AT` kernel — Thinker's
RoPE path was already table-driven, so no second RoPE kernel was needed for
decode, unlike qwen's `step()`). The MoE FFN tail (`moe_sublayer`) is shared,
unchanged, between both attention shapes.

`crate::generate::generate_greedy` now `prefill`s the whole prompt once
(populating every layer's cache) and `decode_step`s one new token at a time
against the cache — O(cached length) attention per step, not O(cached
length)². Weight I/O is UNCHANGED (every layer's weights are still streamed
fresh from `reader` per step; the cache changes the attention math's
complexity, not the disk-streaming pattern — full GPU-resident weights across
steps remains the separate, not-yet-built M1-adjacent work). New
`generate_greedy_positioned`/`generate_greedy_multimodal` entries generalize
the loop to caller-supplied embeddings + real M-RoPE positions (see below).

**Real multimodal input.** New `crates/omni/src/mm.rs`: `encode_audio`
streams `thinker.audio_tower.*` from the `WeightReader` (same qkv-fuse remap
`tests/audio_parity.rs` already validated to cosine 1.000000) and runs
`qwen_asr::encoder::AudioEncoder::encode` unchanged on a raw 16kHz clip
(padded to a whole `chunk_len` — 16000 samples = 1 second, since
`chunk_len(100) x hop(160) == 16000` exactly, so no search is needed).
`encode_image` streams `thinker.visual.*`, real bilinear resize
(`imaging::host::resize_bilinear_hwc` — NOT a crop/pad, since
`smart_resize_default` picks patch-multiple dimensions close to but never
equal to the source's) + `qwenvl::encoder::VisionEncoder` + `PatchMerger`
(same remap `tests/vision_parity.rs` validated). `encode_video_frames` reuses
the single-frame image path per already-decoded frame (frame extraction
itself is out of scope for this crate, same line the M14 plan drew — brain-py
would own `av`/ffmpeg extraction; not built). `build_multimodal_prompt`
assembles media blocks (wrapped in their real start/end tokens from the real
checkpoint's `config.json`) followed by the user's text, splices embeddings
host-side (`splice_host` — a plain slice copy, since `generate.rs`'s prefill
already builds the whole prompt as one host `Vec<f32>` before a single
upload; `model::vlm::splice_fwd`'s on-device kernel is for callers whose
residual buffer is already GPU-resident first, e.g. `qwen::Qwen`'s baked
graph — not this shape).

Real M-RoPE positions for a mixed sequence needed `qwenvl::mrope::get_rope_index`
generalized to MULTIPLE placeholder token types in one scan (it only ever
took one `image_token_id`) — done as `get_rope_index_multi(tokens,
placeholders: &[(token_id, grids)])`, with `get_rope_index` rewritten as a
one-entry wrapper (proven identical to the original by a new
`multi_single_type_matches_get_rope_index_exactly` test, plus all of
`get_rope_index`'s existing tests still passing unchanged). Audio's "grid" is
`(n_audio, 1, 1)` — the meshgrid already degenerates to "T axis advances,
H/W pinned" for `h=w=1`, matching Omni's audio M-RoPE shape modulo one
documented approximation: the post-run anchor advance was generalized from
`max(h,w)` to `max(t,h,w)` (identical for every existing `t=1` image case,
newly correct for audio/video's `t>1`), but real HF Omni additionally scales
audio's T axis by wall-clock seconds (`position_id_per_seconds`) — this port
uses frame-ordinal T advance instead, not the exact reference formula. New
tests: `multi_text_audio_text_advances_by_audio_duration`,
`multi_image_and_audio_interleave_independently`,
`multi_video_run_is_a_t_h_w_meshgrid_like_a_multiframe_image` — all passing.

**Wired into serving**: `generate_spec()` gained optional `audio`/`image`
blob inputs (`Media::Audio`/`Media::Image` — additive, doesn't touch the
`.streaming()`/param shape M11/M12 already validated against `api_caps`).
Both `omni::caps::GenerateAction::run` (the `Provider` path) and
`cli::resident_omni::OmniInstance::run` (the REAL path `brain serve`
dispatches D-Bus/HTTP requests through) extract `audio` via
`audio::asr_caps::wav_from_blob` (the same raw-16kHz-PCM wire convention
`transcribe_spec` already documents) and `image` via
`capability::blob::decode_image` (the engine's one HWC-f32 image wire
format), routing to `OmniInner::generate_multimodal` when either is present.
D-Bus carries these generically (M10's "zero new code" finding holds again —
`brain_py.dbus.BrainDBus.run`/`subscribe` already accepted `blobs=`/`meta=`
kwargs for other models; `BrainBase.generate`/`chat` just needed the same
kwargs threaded through). `BrainOpenAI`/`BrainAnthropic` now raise a clear
`NotImplementedError` on blob input instead of silently dropping it — their
content-part wiring is the same pre-existing, still-open gap M11/M12 already
documented (`openai.rs`/`anthropic.rs` drop image/audio content parts
server-side).

`examples/omni.py`'s `--in-speech` (16kHz mono 16-bit PCM WAV, stdlib `wave`)
and `--in-image` (binary PPM via `brain_py.image.load_ppm`, the same
zero-dependency path `examples/imagegen` uses) are now real over `--dbus`.
`--in-mic`/`--in-video` still `skip()` — live capture and video-frame
extraction need dependencies this example deliberately doesn't take on; the
engine-side video path (`encode_video_frames`) exists but nothing decodes an
actual video FILE into frames yet, and `generate_spec()` has no wire shape
for a list of frame blobs from one `Invocation` (video input is real at the
`crate::mm` level, not yet reachable through the `generate` action's wire
contract — documented explicitly in `caps.rs`'s module doc).

**Verification**: `cargo test -p brain-model -p brain-omni -p brain-qwenvl
-p brain-kernels -p brain-cli` — all green (91/91 in `brain-model` incl. the
2 new KV-cache tests, 42/42 in `brain-qwenvl` incl. the 4 new `mrope::multi_*`
tests, `brain-omni`'s 14 unit + `caps_conformance`'s 5 tests, `brain-cli`'s
63). The full workspace `make test` hit the SAME pre-existing GPU-state hang
`roofline.rs`'s `caps_expose_the_roofs_only_after_something_measured_them`
already recorded in this file's M6a entry (orphaned VRAM from earlier killed
processes, unrelated to any file this round touched) — targeted tests are
the real check, same judgment call as M6a. `make clippy`: 201 warnings vs. a
183 baseline (+18), but every one of the 18 is in a file this round never
touched (`crates/npu/tests/*`, `crates/cli/{forecast,npu,resident_asr,
resident_forecast,resident_llm,resident_mock,splat,supply,wm}_cli.rs` —
confirmed by grepping the full clippy report for every file this round
edited: zero matches) — pre-existing debt, not a regression, same pattern
M6a's entry already documented for this same gate. Real-weight parity
(`thinker_layer_parity`, `talker_layer_parity`, `vision_parity`,
`audio_parity`, etc.) is unaffected — none of that math changed, only the
attention dispatch shape and a new host-side assembly path around it — and
those tests remain `#[ignore]`d pending the checkpoint download (in progress
throughout this work: 51 GB / 70.5 GB, 43 GB free on the 93 GB tmpfs as of
this entry).

**Not started, still**: `qwen::Qwen` itself was not migrated onto the hoisted
`model::block` decode primitives (only `omni::thinker` uses them so far —
qwen's own inline copy still works, gated by its own existing tests, but now
diverges from the hoisted version rather than being its single source);
real wall-clock audio-timestamp M-RoPE scaling (documented approximation
above); video FILE decoding in brain-py/examples; a wire shape for
multi-frame video input on the `generate` action; int8/GPU-sharded resident
weights across generation steps (the cache fixed the attention complexity,
not the weight-I/O pattern).

## Not started

M2c (backward + gradcheck, deferred — see M2 note above), M2d (glm migration,
deferred), M8's `chunked_decode` streaming path, `qwen::Qwen`'s migration
onto the hoisted KV-cache decode primitives (see M9b above), int8/GPU-sharded
production residency (M9b above), `converse`/`transcribe`/`speak` actions
(need Talker+MTP+Code2Wav chained with `accept_hidden_layer` + codec
sampling), `qwenvl` registration (deferred above, its own generation loop),
the pre-existing multimodal-content-part-drop bug in `openai.rs`/`anthropic.rs`
and the `/v1/audio/*` endpoints (M11/M12's original scope, orthogonal to what
's implemented), OpenAI/Anthropic transports in the automated e2e harness
(M13/M14's own note above), video-file decoding + a multi-frame wire shape
(M9b above), M15 through M17. See the plan file. M6 through M9b are now
believed complete for what is actually implemented (Thinker/Talker decoders,
real M-RoPE incl. real audio/image/video splice, KV-cache decode, composed
loops, splice seam, code predictor, Code2Wav vocoder, Thinker text generation
with optional real speech/image input exposed over D-Bus/OpenAI/Anthropic
with a working Python example — all validated against real weights, the real
HTTP classification logic, or a live end-to-end run). The two loader-side
checkpoint-naming gaps (code predictor, code2wav — documented in M7b/M8
above) remain open; they do not block Thinker-only generation (`OmniResident`
reads straight from the HF directory, not the unified checkpoint those gaps
affect).

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
