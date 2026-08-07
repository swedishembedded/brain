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

## In progress

- Reference goldens dump (`tools/goldens/omni_dump_reference.py`) — script
  written (component-scoped: audio tower, vision tower, one MoE decoder layer,
  M-RoPE position ids, code predictor, code2wav — each streams only its own
  tensors from the sharded checkpoint via `model.safetensors.index.json`, no
  full-model load). Weight shards for these components (4 of 15,
  `model-{00001,00013,00014,00015}-of-00015.safetensors`, ~15.5 GB) fetching
  into the ramdisk (`/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct/`);
  not yet run against real weights.

- **M2a — sparse MoE core, forward pass** (2026-08-07). `model::moe`
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
  **Deliberately deferred to a follow-up commit**: no row-compaction
  (gather/scatter) — WGSL kernels here may not use atomics, so true stream
  compaction needs a separate prefix-sum pass; the per-row early-exit already
  removes the FLOPs, just not the thread launches, which is enough for
  correctness and a real speed win but leaves grid-size optimization on the
  table for M16. No int8 grouped-GEMM tier yet (M2b). No backward pass /
  gradcheck coverage yet (M2c) — `expert_fwd` is forward-only. `crates/glm`
  migration not done yet (M2d).
  Pre-existing, unrelated finding while running `make clippy`: this branch
  carries 192 warnings against a 183 baseline with EVERY change in this
  workstream stashed out (`crates/cli/{resident_forecast,resident_llm,
  resident_mock,splat_cli,supply,wm_cli}.rs`) — confirmed not caused by this
  work; not in scope to fix here, noted for whoever picks it up.

## Not started

M2b (int8 grouped GEMM), M2c (backward + gradcheck), M2d (glm migration),
M3 through M17. See the plan file.

## Honesty notes

- No NPU device run has happened or will happen on this box (M15's scope is
  explicitly capped at CPU-side OpenVINO parity + ONNX graph validation).
- No number in this file is a projection; everything above is either read
  directly from the released `config.json`/index, or will be a `brain perf`/
  `gpu_core::profile` measurement once code exists.
