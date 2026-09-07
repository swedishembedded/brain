# minimaxh3 - roadmap

MiniMax-H3: a 33B joint video+audio rectified-flow diffusion transformer -
**one packed self-attention stack**, not a two-stream architecture like
`ltxv`. A single 50-block transformer denoises video and audio latent rows
in the same sequence, under **two independent shifted-sigma Euler schedules**
(`shift=12` video, `shift=3` audio) driven from one forward per step, with
no CFG pass (guidance-distilled). Conditioning comes from a Qwen3-VL text
encoder truncated to **decoder layer 50** (confirmed from source, see below),
refined by 2 token-refiner blocks (plain pre-norm, no AdaLN, no RoPE) before
packing. Video: a causal 3D VAE (24 latent channels) + `1x2x2` patchify.
Audio: a DAC-lineage encoder + BigVGAN decoder (32 latent channels, 32kHz
mono - see Phase 4 detail; the checkpoint's own shipped code never encodes
audio, only decodes). Released tasks: `t2va` (text-to-video+audio), `fl2va`
(first/last-frame-conditioned), `ref2va` (up to 12 mixed image/video/audio
references) - `t2va`/`fl2va` share one transformer partition (`transformer`),
`ref2va` uses a separately-trained second instance of the SAME architecture
(`transformer_ref`).

The port follows `.agents/rules/porting.md` in order: facts before code,
reference goldens before Rust, two-way import coverage, tiny-config smoke,
then the five-rung parity ladder, never skipped. The phases below are the
approved implementation plan for this port.

## Reference material - the actual authority, not prose

**`diffusers==0.40.0` (a normal, stable PyPI release - `pip install
diffusers` gets it, no dev/pre-release needed) ships COMPLETE, real
MiniMax-H3 support.** This is now the primary math authority for this port,
confirmed by direct import and by reading the installed source (session
venv, `torch==2.14.0+cu130`, `transformers==5.16.1`):

| Class | File |
|---|---|
| `MiniMaxH3Transformer3DModel` | `diffusers/models/transformers/transformer_minimax_h3.py` |
| `MiniMaxH3Blocks` (+ the `t2va`/`fl2va`/`ref2va` step classes) | `diffusers/modular_pipelines/minimax_h3/{modular_blocks_minimax_h3,before_denoise,before_encoder,encoders,decoders,denoise,references}.py` |
| `MiniMaxH3Scheduler` | `diffusers/schedulers/scheduling_minimax_h3.py` |
| `AutoencoderKLMiniMaxH3` (video) | `diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py` |
| `AutoencoderKLMiniMaxH3Audio` | `diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py` |
| `MiniMaxH3ModularPipeline` | `diffusers/modular_pipelines/minimax_h3/modular_pipeline.py` |

All Apache-2.0 (`Copyright 2025/2026 The MiniMax Team and The HuggingFace
Team`). This supersedes the checkpoint's own legacy `FL2VA/model_index.json`
(a `diffusers==0.32.2`-era `MiniMaxH3Pipeline` with a custom
`MiniMaxH3Qwen3VLHFEncoder` wrapper class that is NOT what this port reads
from - the modern `MiniMaxH3ModularPipeline` uses plain
`Qwen3VLForConditionalGeneration` directly and does the layer-50 truncation
itself, in `encoders.py`, not in a wrapper class). MiniMax's own
(community-licensed) repository is not needed and not read.

`transformer_minimax_h3.py` and `modular_blocks_minimax_h3.py` +
`before_denoise.py` have been read in full; `scheduling_minimax_h3.py`,
`autoencoder_kl_minimax_h3{,_audio}.py`, `denoise.py`, `encoders.py`, and
`MiniMaxH3Ref2VAPrepareLayoutStep`'s full body are not yet read - see
"Convention questions" for exactly what is settled vs still open.

## Checkpoint layout - real numbers, not an estimate

The full `MiniMaxAI/MiniMax-H3` repo is **498.5GB** (confirmed via the HF
tree API, not a guess), because the model ships **twice**, once per
supported diffusers API generation:

| Layout | Size | What it is |
|---|---|---|
| Root (`transformer/`, `transformer_ref/`, `text_encoder/`, `vae/`, `audio_vae/`, `tokenizer/`, `processor/`, `scheduler/`, `audio_scheduler/`) | ~210GB | The `diffusers>=0.36.0.dev0` `MiniMaxH3ModularPipeline` layout (confirmed identical to `modular_model_index.json`) - **one copy of everything**, `transformer`/`transformer_ref` share the same text encoder/VAEs. **This is the layout the port targets and the layout `diffusers==0.40.0` actually loads.** |
| `FL2VA/` + `Ref2VA/` | ~288GB | The older `diffusers==0.32.2` `MiniMaxH3Pipeline` layout - each partition fully self-contained (duplicates the ~78GB of shared components). **Not used by this port; deleted from disk (see below).** |

**Current local disk state** (`BRAIN_MINIMAXH3_DIR` root, this session's
local checkout - see `[path/to/checkout]` in place of the real machine path):
- `text_encoder/`: **10 of 14 shards present** (`00001-00009`, `00014` +
  `config.json`) - moved from the now-deleted `FL2VA/text_encoder/`, **verified
  byte-identical first** (LFS oid + size match exactly against the root
  manifest) before moving, so this was a safe rename, not a guess. Missing:
  shards `00010-00013` + the small tokenizer/processor/config files.
  `text_encoder`/`video_vae`/`audio_vae` are shared by BOTH `transformer` and
  `transformer_ref` in this layout (one `modular_model_index.json` lists each
  once) - the "shared across partitions?" question from the old FL2VA/Ref2VA
  framing is moot under the root layout.
- `transformer/`, `transformer_ref/`, `vae/`, `audio_vae/` (remaining ~19.5GB
  of text_encoder, plus `tokenizer/`/`processor/`/`scheduler/`/
  `audio_scheduler/`/`model_index.json`): **not yet fetched** - the download
  command was handed to the user (blocked from running `hf download` directly
  by the auto-mode permission classifier) and has not been confirmed run yet.
- The old `FL2VA/` legacy-layout tree (including its own audio_vae, which was
  confirmed NOT byte-identical to the root's - different LFS oid AND a
  32-byte size difference, a different export) has been deleted; freed 45GB.

**`H3Recipe` (`crates/modelstore/src/recipe.rs`) and the real-checkpoint test
paths in `crates/minimaxh3/src/import.rs` still assume the old FL2VA/Ref2VA
nested layout and need updating to the root flat layout - deferred until the
download above completes (explicit user instruction), tracked in "Recorded
gaps."**

**The unsloth GGUF release** (`unsloth/MiniMax-H3-GGUF`) is real and
confirmed **pruned, not just quantized**: Q8_0 is 21.44GB per transformer
variant vs ~66GB bf16 - close to what dropping `adaln_proj`'s 13B (see below)
would produce, an independent signal Phase 8's AdaLN-precompute plan is
pointed the right direction. Not fetched - bf16 is what porting.md's
parity-proving import needs; GGUF is a later serving-tier option once the
port is real-weight-parity-gated.

## Verified ground truth

### From the real checkpoint's own config/safetensors headers
- `model_index.json` (root): `sigma_shift_scales: {video: 12.0, audio: 3.0}`.
- `transformer/config.json`: `hidden_size=5376`, `num_layers=50`,
  `token_refiner_num_layers=2` (diffusers: `num_refiner_layers`),
  `num_attention_heads=56`, `attention_head_dim=128` (QKV width
  `56*128=7168 != 5376`), `ffn_hidden_size=14336`, `latents_dim=24` (video),
  `audio_latents_dim=32`, `patch_size=[1,2,2]`, `text_dim=5120`,
  `timestep_input_dim=256`, `time_embed_hidden_size=5376`,
  `time_embed_dim=2688`, `adaln_out_features=96768`,
  `final_adaln_out_features=10752`, `rope_inv_freq_len=16`.
- Real transformer tensor names (BF16 except norms/embed/rope, which are
  F32): `blocks.N.qkv_proj.weight [21504,5376]` (fused QKV, confirms the
  installed source's `to_qkv` fused-projections branch is what the real
  checkpoint uses), `attn.{q,k}_norm.weight [128]`,
  `attn.out_proj.weight [5376,7168]`, `mlp.fc1.weight [28672,5376]` (SwiGLU),
  `mlp.fc2.weight [5376,14336]`, `norm{1,2}.weight [5376]`,
  `adaln_proj.linear.{weight [96768,2688],bias}` per block,
  `condition_proj.{weight [5376,5120],bias}` (diffusers: `context_embedder`),
  `time_embedder.proj_{in,out}`, `token_refiner.blocks.{0,1}.*` +
  `token_refiner.final_norm.weight`, `video_patch_proj.weight [5376,96]`
  (diffusers: `proj_in`), `audio_patch_proj.weight [5376,32]` (diffusers:
  `audio_proj_in`), `rope.inv_freq [16]`.
- **`adaln_proj` is 260M params/block x 50 blocks = 13.0B of the 33B total**
  - AdaLN precompute (Phase 8) is load-bearing for fitting this model.
- `audio_vae/`: `latent_channels=32`, `sample_rate=32000`, `output_channel=2`
  (stereo, L/R independently through a shared mono VAE), real per-channel
  `latents_mean`/`latents_std`.
- `text_encoder/`: Qwen3-VL, `hidden_size=5120`, `num_hidden_layers=64`,
  `num_attention_heads=64`, `num_key_value_heads=8`, `head_dim=128`,
  `intermediate_size=25600`, `rope_theta=5000000`, M-RoPE
  `mrope_section=[24,20,20]` interleaved. Vision tower: `depth=27`,
  `hidden_size=1152`, `num_heads=16`, `deepstack_visual_indexes=[8,16,24]`,
  `out_hidden_size=5120`, `patch_size=16`, `spatial_merge_size=2`.

### From reading `transformer_minimax_h3.py` (the transformer block math, exactly)

- **The 18 `adaln_proj` modulation vectors, solved**:
  `MINIMAX_H3_MODALITY_NUM = 3` (0=video, 1=text, 2=audio - every row of the
  packed sequence, including TEXT rows, gets AdaLN-modulated). `adaln_proj`
  is one `Linear(time_embed_dim, 6 * hidden_size * 3)` producing the
  standard AdaLN-Zero sextet (`shift_msa, scale_msa, gate_msa, shift_mlp,
  scale_mlp, gate_mlp`) **times 3 modalities** = 18. Table row layout:
  `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]`, addressed per row of the
  packed sequence by `adaln_indices = timestep_indices * 3 + token_tags`.
  My original hypothesis (2 modalities x 2 sublayers x 3) was WRONG in
  mechanism though right in ballpark - worth remembering as a reminder that
  guessing this instead of reading the source would have shipped a real bug.
- **Timestep conditioning is per-ROW, not per-clip-scalar**: `timestep`
  holds the *distinct* timestep values present in one forward
  (`(num_timesteps,)`), and every row of the packed sequence carries an
  INDEX into it (`timestep_indices`, `(seq_len,)`) - true diffusion-forcing,
  confirming the ltxv-style "do not assume scalar-per-stream" default was
  the right one to default to. The porting.md SS7 modulation-fold shortcut
  does NOT apply - the unfolded per-row form is required, exactly as
  `crate::schedule`'s module doc already assumed defensively.
- **RoPE, exactly**: one shared `inv_freq` buffer of `rope_freq_dim=16`
  frequencies (`1/(10000^(arange(0,32,2)/32))`). For each axis `(t,h,w)`:
  `freqs_axis = position_id[axis] * inv_freq` (16 values); concatenate the 3
  axes -> 48; concatenate with itself -> 96 (`rotate_half` convention). Only
  the LEADING 96 of `head_dim=128` channels are rotated; the remaining 32
  pass through unrotated. This is genuinely more than "one shared table
  sliced per axis" - the pass-through tail was not something the config
  numbers alone would reveal.
- **`norm_out` (final AdaLN) is per-TIMESTEP only, not per-modality**:
  `MiniMaxH3AdaLayerNormOut.forward` is called with `timestep_indices`
  directly (never multiplied by `MINIMAX_H3_MODALITY_NUM`), and its
  `linear` produces `2*hidden_size` (not `x3`) - confirms
  `final_adaln_out_features=10752=2x5376` is shift+scale shared across
  video/text/audio at a given timestep, asymmetric with every other AdaLN
  site in the model. Not something the config numbers alone implied.
- **Token refiner has NO AdaLN and NO RoPE at all** - plain pre-norm
  (`nn.RMSNorm`) self-attention + SwiGLU feed-forward, residual, 2 layers,
  a final `RMSNorm`. Simpler than block Phase 5's plan assumed (which
  guessed it might share the main block's AdaLN/RoPE machinery).
- **Attention**: `to_q`/`to_k`/`to_v` (or fused `to_qkv`, matching the real
  checkpoint), no biases anywhere in attention (q/k/v/out all `bias=False`),
  per-head `RMSNorm(dim_head=128)` QK-norm (`norm_q`/`norm_k`, matching real
  `q_norm`/`k_norm` tensor names), output via `to_out[0]` (Linear, no bias)
  + `to_out[1]` (Dropout p=0, no-op at inference). Full self-attention over
  the ENTIRE packed sequence, no mask, no cross-attention anywhere in the
  model.
- **FFN**: diffusers' generic `FeedForward(hidden_size, inner_dim=14336,
  activation_fn="swiglu", bias=False)` - the exact internal gate/up split
  order inside that generic class is not yet independently confirmed (low
  priority, a diffusers-generic utility, not MiniMax-H3-specific).
- **AdaLN activation order**: `linear(silu(temb))` - SiLU BEFORE the linear
  projection, both for the per-block `adaln_proj` and for `norm_out`.
- **Mixed precision** (informs the reference goldens' dtype handling, not
  brain's own all-fp32 compute): `_keep_in_fp32_modules = ["proj_in",
  "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope"]` -
  everything else, INCLUDING `context_embedder` and every `adaln_proj`, runs
  bf16 in the reference.
- Block math, exactly (AdaLN-Zero, per-row modulation):
  ```
  shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_proj(temb)  # indexed by adaln_indices per row
  h = h + gate_msa[adaln_indices] * attn(norm1(h) * (1+scale_msa[adaln_indices]) + shift_msa[adaln_indices], rope)
  h = h + gate_mlp[adaln_indices] * ff(norm2(h) * (1+scale_mlp[adaln_indices]) + shift_mlp[adaln_indices])
  ```
- `time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True,
  downscale_freq_shift=0)` (diffusers' standard sinusoidal embed, exact
  flags matter for bit parity) -> `time_embedder =
  TimestepEmbedding(256, time_embed_dim=5376, out_dim=2688)`, shared by
  every block's `adaln_proj` and by `norm_out`.
- `_cp_plan` (context-parallel sharding plan) is a real blueprint for how
  MiniMax intends this to scale across GPUs - directly relevant to Phase 12
  later, not read in detail yet.

### From reading `modular_blocks_minimax_h3.py` + `before_denoise.py` (packing, layout, row order)

- **Text-encoder truncation depth, confirmed exactly**: "the hidden state
  MiniMax-H3 conditions on ... read after the **50th** decoder layer of the
  Qwen3-VL conditioner" (`MiniMaxH3AutoTextEncoderStep`'s own docstring).
  The pasted doc that started this port claimed this; it is now independently
  confirmed from source, not merely repeated.
- **A keyframe/reference's own vision-block rows are tagged modality=VIDEO
  in the text encoder's output, not TEXT** - `text_token_tags` is not a
  constant; genuinely surprising, would not have been guessed.
- **`t2va`/`fl2va` packed row order, exactly**: `[text | keyframe
  conditions | target audio | target video]`
  (`MiniMaxH3PrepareLayoutStep.build_packed_sequence`). `video_indices` is
  the CONCATENATION of the keyframe-condition range and the target-video
  range (skipping over the audio block sandwiched between them in raw
  position) - video/audio/text index arrays are not always contiguous.
  `t2va` has zero keyframe-condition rows (`keyframe_anchors=()`).
- **Position-id (RoPE `t,h,w`) construction, exactly**:
  - text rows: `t = row index (0..num_text_tokens)`, `h=w=0`.
  - keyframe condition rows: `t = anchor_time` (`"first"` ->
    `num_text_tokens`; `"last"` -> a specific summed value, see below),
    `(h,w) = frame_grid` (the full spatial grid for that one frame).
  - target audio rows (channel-major): `t = num_text_tokens +
    arange(num_audio_latents)` (repeated per channel), `h=0` (never set),
    `w` pinned to the width grid's extremes - first channel at
    `width_grid[0]`, remaining channel(s) at `width_grid[-1]`.
  - target video rows: `t` from `_temporal_position_grid` - NON-UNIFORM
    spacing, `5/3 * (1,4,4,4,4)` cyclically per latent frame (mirrors the
    video VAE's 17-pixel-frame-to-5-latent-frame grouping); `(h,w) =
    frame_grid`, repeated per frame.
  - spatial grid: aspect-normalized, `dim/patch` coordinates centered on
    `[0, 32)` via `np.linspace(..., endpoint=False)` in **float64**
    (reproduced exactly, not approximated - torch's own `linspace` computes
    a different grid than numpy's `endpoint=False` form at the same
    arguments).
- **A genuinely fine-grained numerical trap, worth its own line**: the
  `"last"` keyframe anchor's rotary time is computed via **numpy's pairwise
  summation** of the per-frame time spans, specifically to bit-match the
  reference; the analogous audio-soundtrack span sum in the `ref2va` layout
  (not yet fully read) uses **sequential** summation instead - the module's
  own comment states the two orders differ in the last ULP from 16 latent
  frames onward. Any Rust port of either sum must reproduce the SAME
  summation order as its own call site, not assume they match each other.
- `ref2va`'s layout (`MiniMaxH3Ref2VAPrepareLayoutStep`) is structurally
  parallel (one condition block per reference, its own row range) but its
  full body has not been read yet - see open questions.

## Convention questions

### Settled from source (see citations above - not repeated here)
- ~~18 `adaln_proj` modulation vectors~~ - solved: 3 modalities x 6 AdaLN-Zero params.
- ~~per-row vs per-clip timestep~~ - solved: per-row, via `timestep_indices` into a compact distinct-values vector.
- ~~packed-sequence row order~~ - solved for t2va/fl2va: `[text | keyframe conditions | target audio | target video]`.
- ~~`rope.inv_freq` axis construction~~ - solved: one shared 16-freq table, 3 axes concatenated then doubled, only the leading 96/128 head_dim channels rotated.
- ~~text-encoder truncation layer~~ - solved: layer 50, exactly.
- ~~fl2va keyframe conditioning convention~~ - solved: extra video rows (`condition_start:audio_start`), not a mask.
- ~~shared vs partition-specific text_encoder/video_vae/audio_vae~~ - moot under the root layout: genuinely shared (one `modular_model_index.json` lists each once for both `transformer` and `transformer_ref`).
- ~~the dual schedule's exact convention~~ - solved, and the placeholder it replaced was WRONG (see Phase 7 detail): no `num_train_timesteps`, no `invert_sigmas`, `linspace(1,0,N)` with the terminal folded into the count, `step()` is a data-ward `x0=x_t+sigma*v` blend by a separately-sourced ratio, not Euler.
- ~~`video_vae`'s causal-encoder/non-causal-decoder conventions~~ - solved: strict zero-pad causal convs (no cross-chunk cache) + separate symmetric reflect spatial padding on the encoder side, confirmed 36-layer non-causal ViT decoder. See Phase 6 detail.
- ~~the Qwen3-VL layer-50 extraction call site~~ - solved: stock `transformers` `output_hidden_states=True` + generic hook-based capture (no MiniMax-H3-specific forward hook), `hidden_states[50]` is pre-final-norm (the helper's own `ValueError` guard exists specifically to keep it that way), and brain's existing `res[0]`=embed/`res[l]`=block-`l-1`-output/pre-norm convention in `qwen3::Qwen::encode_hidden` already matches this exactly - `encode_hidden(tokens, 50)` needs no off-by-one adjustment. Presentation is plain token concatenation, no chat template, vision content as inline `<|vision_start|>...<|vision_end|>` runs.

### Still open - settle before claiming the relevant phase's parity
- [ ] `ref2va`'s full packed layout beyond the structural skeleton already
      seen (`MiniMaxH3Ref2VAPrepareLayoutStep`'s full body, `denoise.py`'s
      `MiniMaxH3Ref2VADenoiseStep`, the audio-reference summation-order
      trap noted above) - read once at a research pass, not yet implemented
      (Phase 9 deliberately deferred it - see the Phases list).
- [ ] diffusers' generic `FeedForward(activation_fn="swiglu")`'s exact
      internal gate/up split order - low priority, not MiniMax-H3-specific.

## Phases (see the approved plan for full detail)

- [x] Phase 0 - ledger, arch registration, crate skeleton, license gate
- [~] Phase 1 - reference oracle: **torch/diffusers/transformers installed
      and confirmed to have full real MiniMax-H3 support** (`diffusers
      0.40.0`, stable PyPI); the actual golden-dump script (`tools/
      minimaxh3_dump_reference.py`) has not been written yet - most of what
      it would settle has already been read directly from source instead
      (see "Convention questions" - settled section), which is a stronger
      form of ground truth than a golden dump for STRUCTURE, though real
      numeric goldens (for the parity ladder's later rungs) still need the
      script written and run.
- [x] Phase 2 - fetch recipe for the two-level partitioned checkpoint
      (`brain pull` only) - **now targets the WRONG layout** (the old
      FL2VA/Ref2VA nested shape); needs updating to the root flat layout,
      deferred until the current download completes (see Recorded gaps).
- [x] Phase 3 - Qwen3-VL `encode_hidden`/`encode_hiddens` extension
- [x] Phase 4 - audio VAE **decoder** port: structurally complete,
      real-checkpoint-validated, AND now real-weight NUMERIC parity-gated
      against the checkpoint's own shipped reference (see below)
- [x] Phase 5 - H3 DiT core: tiny-config numeric parity against the real
      `diffusers` reference, all taps cosine 1.0000000000 (see detail below).
      Real-checkpoint import (splitting the fused `qkv_proj`) not built yet.
- [x] Phase 6 - video VAE: tiny-config numeric parity against the real
      reference, all taps cosine 1.0000000000. Real-weight validation
      written but blocked on `vae/`'s download (see detail below).
- [x] Phase 7 (revised) - dual rectified-flow schedulers - **the placeholder
      assumptions this phase shipped with were WRONG in a load-bearing way**,
      caught and fixed while building Phase 9 (see "A placeholder that was
      wrong" below). `crate::schedule` now matches the real
      `MiniMaxH3Scheduler` exactly.
- [x] Phase 8 - AdaLN precompute checkpoint transform: numerically verified
      identical to running `adaln_proj` directly; real-config extrapolation
      lands at 20.2B backbone params, matching the roadmap's own earlier
      ~19.3B estimate (see detail below).
- [x] Phase 9 (t2va/fl2va only) - packed-sequence layout, RoPE position-ids
      (including the numpy-pairwise-summation reproduction for the "last"
      keyframe anchor), and the full denoise-loop orchestration. `ref2va`
      **deliberately not implemented** - its layout is structurally
      different enough (per-reference sub-blocks, a second/sequentially-
      summed rotary-span convention, per-reference-kind limits) to need its
      own layout builder rather than a generalization of this one; the
      DiT/scheduler/VAE machinery it would reuse is otherwise ready.
- [x] Phase 10 (gradcheck scaffold only) - host f64 forward + analytic
      backward, `check_minimaxh3`/`check_minimaxh3_conditioning` both green.
      LoRA/device-trainer not built yet.
- [x] Phase 11 - capability/residency/D-Bus serving contract for
      `t2va`/`fl2va`. Real DiT weight import and a real (non-stub) Qwen3-VL
      text encoder are both still open - see Recorded gaps.
- [x] First real end-to-end `t2va` generation, real weights throughout -
  `crates/minimaxh3/examples/generate_t2va.rs`, run against the full
  checkpoint (`text_encoder/` 63GB, `transformer/` 66GB, `vae/` 10GB, all
  streamed): real Qwen3-VL prompt encoding -> real 50-layer/33B DiT (fp32,
  no reduced-precision tier yet) denoise loop (4 steps, 128x128, 124
  frames) -> real video VAE + audio vocoder decode -> muxed to MP4 (ffmpeg,
  no distro package available in this environment, a pip-installed static
  binary used instead). Total wall clock 3181s (~53min): text encoder
  ~140s, DiT streaming-load ~15-20min (dominates - reading+converting 66GB
  fp32-promoted-from-bf16 with no reduced-precision path), the 3 real
  model evaluations + both VAE decodes + mux the remainder. Peak cgroup
  memory rode the 150GB container cap repeatedly DURING DiT load (anon
  climbed to ~142GB at one point, reclaimable page cache from the
  freshly-completed 76GB of downloads absorbing the difference each time)
  before plateauing ~140GB resident once loading finished - survived every
  time, never killed, confirming the streaming+`advise_drop` fixes and the
  container's reactive reclaim are sufficient together, but with very
  little margin at this checkpoint's real fp32 size on this host's current
  cap. Output is mechanically valid (real audio, non-degenerate: -22.9dB
  mean / -10.8dB peak, not silence; real video, structured not
  noise-garbage) but not yet visually recognizable - expected at only 3
  real denoising steps for a rectified-flow model (20-50 is typical), not
  a correctness concern. Next real-quality attempt should either raise
  `num_inference_steps`/canvas (linearly more DiT-forward wall-clock, no
  new memory risk since it's the SAME loaded model) or wire Phase 8's
  already-verified AdaLN precompute into this loading path first (cuts
  DiT to ~20.2B params, both memory pressure AND the load-time-dominated
  wall-clock) - the precompute transform itself is only verified against
  extrapolated dimensions today, never run against these real weights.
- [ ] Phase 12 - streaming-overlap engine (separate, measured perf phase) -
      `_cp_plan` in the real transformer source is a useful blueprint,
      not yet studied in detail. Deliberately not attempted this session:
      nothing is real-weight-parity-gated yet, which is this phase's own
      stated precondition.

## Phase 4 detail: audio VAE decoder

`crates/minimaxh3/src/vocoder.rs` + `import.rs` - the BigVGAN-topology
decoder half of the audio VAE only (`DacAudioVAE.decode`: `dec_in_proj` then
the BigVGAN stack). **Encode is out of scope** for this milestone - the
checkpoint's own shipped `dac_audio_vae.py` only implements `.decode()` at
all; `encoder`/`mean_proj`/`logs_proj`/`pre_block` (173 of the checkpoint's
1087 tensors) are present for training-time compatibility but dead in the
shipped inference-only forward, and neither reads them.

This reparameterizes an existing in-workspace BigVGAN v2 / AMP1 / anti-
aliased-SnakeBeta vocoder port (same topology, different config numbers) -
genuinely new work was the H3-specific config/manifest and three real,
checked naming differences from that precedent: `decoder.ups.{i}.0.*` (not
flat `ups.{i}.*` - `BigVGAN.__init__` wraps each stage's transposed conv in
its own one-element `ModuleList`), `decoder.resblocks.{idx}.activations.{0..5}`
flat (not split `acts1.{d}`/`acts2.{d}` - `AMPBlock1` stores one 6-element
list and slices it in its own forward), and a top-level `dec_in_proj` that
is a PLAIN (non-weight-normalized) 1x1 conv, unlike every other conv in the
file.

**Verified without any Python oracle** (real checkpoint, no torch needed for
any of this):
- `tensor_manifest()`'s 779 post-fold tensor count matches an independent
  Python count from the real safetensors header exactly (914 raw tensors
  under `decoder.*`+`dec_in_proj.*`, folding each `weight_g`/`weight_v` pair
  to one `.weight`).
- `import_audio_vae_decoder` achieves real two-way coverage against the
  actual 578MB checkpoint: every claimed tensor present and correctly
  shaped, the 173 unclaimed tensors are exactly the expected out-of-scope
  `encoder`/`mean_proj`/`logs_proj`/`pre_block` prefixes (asserted, not
  assumed).
- `decode()` runs the FULL real-scale graph (1024-wide, 7 stages, 21
  resblocks) against the real imported weights end to end on this
  hardware's CPU backend: finite, correctly `[-1,1]`-clamped, non-trivial
  output, ~6s including import.
- A weight-free tiny-config smoke test also exercises the same graph
  end-to-end; its first version had a real bug (channel width `24` right-
  shifted to 0 after 7 halvings, `24 >> 7 == 0`), caught by its own
  "output must not be trivially all-zero" assertion - fixed by widening to
  `128`. Worth recording: the assertion did its job.

**Real NUMERIC parity, now proven** against the checkpoint's own shipped
`minimax_h3_audio_vae.py`/`dac_audio_vae.py` reference (loaded directly via
`from_pretrained`, not reimplemented - `tools/minimaxh3_audio_vae_dump_reference.py`),
on a deterministic synthetic `[32, 6]` latent (seed 7): worst-tap cosine
**1.0000000000**, worst-tap max abs error **4.172e-6** (`tap_conv_pre`; the
end-to-end `waveform` tap itself is 1.192e-6), comfortably inside float32
noise and well past the porting.md floor (stage cosine >= 0.9999). Checked at
FOUR points, not just end-to-end (`crate::vocoder::decode_with_taps`'s new
`DecodeTaps`): `dec_in_proj` output, `conv_pre` output, the first
upsample+resblock-average stage output, and the final waveform -
`crates/minimaxh3/src/import.rs::decode_matches_the_real_reference_numerically`.
No bug was found in the existing port - every tap passed on the first real
run, meaning the earlier "structurally complete, real-checkpoint-validated"
claim (weight orientation, SwiGLU/gate order n/a here - this decoder has no
SwiGLU, the antialiasing filter scale, weight_norm fold, resblock averaging)
was already numerically correct, not merely finite. Fixture:
`testdata/golden/minimaxh3/audio_vae/{minimaxh3_audio_vae.safetensors,manifest.json}`
(gitignored under `/testdata/`, like every other golden in this workspace -
regenerate with the dumper above). The dumper's own two self-checks (a
hand-rolled stage-by-stage replay vs. the library's own `vae.decode(z)`, and
a freshly re-instantiated model) both landed at exactly `0.0` max-abs
difference before anything was written, so the golden's own correctness
does not rest on trusting either code path alone.

## Phase 5 detail: DiT core

`crates/minimaxh3/src/{config,rope,block,model}.rs` - the block math settled
in "From reading transformer_minimax_h3.py" above, implemented and gated
against a real `tools/minimaxh3_dit_dump_reference.py` dump (the actual
`diffusers` `MiniMaxH3Transformer3DModel` class, tiny config, seeded random
weights). All 8 taps (token-refiner output, RoPE cos/sin, block-0
input/attn-out/output, video/audio outputs) land at cosine 1.0000000000,
worst max_abs 5.960e-7 - float32 noise. **No bug needed fixing** - every tap
passed on the first real run. `H3Transformer::load` takes a `Tensors` source
named after the reference module's own attribute paths (`to_q`/`to_k`/`to_v`
split, not the checkpoint's fused `qkv_proj`) - **splitting the real
checkpoint's fused QKV the same way `mlp.fc1.weight` is already split is not
built yet** (tracked in Recorded gaps below).

One deliberate internal-addressing deviation, proven not to change any
observable output (parity is still 1.0): the reference's `adaln_proj`
output reshapes timestep-major (`[timestep_idx*3+modality, 6*hidden]`); this
port addresses six per-param tables modality-major
(`[modality*num_timesteps+timestep_idx, hidden]`,
`adaln_indices = token_tags*num_timesteps + timestep_indices`) so each
`(modality, param)` weight slice is a contiguous row range needing no
interleaving/scatter kernel this workspace has no primitive for.

`dit::rope`/`dit::adaln` end up unused: `dit::rope` is interleaved-pair
rotation, H3 needs half-split `rotate_half` (a bespoke `rope.rs` instead,
built on the existing `rope2d_partial` kernel, which already implements the
half-split/pass-through-tail contract exactly); `dit::adaln`'s `RowTable` is
for a modulation vector shared by a whole forward, and H3's genuinely varies
per packed-sequence row, so the row-varying gather is built directly on the
existing `embed` (row-gather) kernel instead. Batch size is fixed at 1
(multi-item batching is a documented, structurally straightforward
follow-up, not built).

## Phase 6 detail: video VAE

`crates/minimaxh3/src/video_vae.rs` - `AutoencoderKLMiniMaxH3`: a causal 3D
CNN encoder (6 stages, spatial x16/temporal x4, **strict zero-pad causal
convs with separate symmetric REFLECT spatial padding** - genuinely zero-pad
causal, not a Wan-style cross-chunk `feat_cache`; each clip-length chunk runs
independently, re-padded from scratch) paired with a **non-causal 36-layer
ViT decoder** (confirmed correct, word for word, against the class's own
docstring and `decoder_num_layers=36` default - the ledger's earlier
secondhand claim held up this time). New kernel: `pad2d_reflect.wgsl` (no
reflect-pad primitive existed; every prior zero-pad convention in this
workspace was insufficient). Gated against a real `tools/
minimaxh3_video_vae_dump_reference.py` dump (real `diffusers`
`AutoencoderKLMiniMaxH3`, tiny config): all 5 taps (encode/decode, single-
and multi-clip incl. the cross-fade blend) cosine 1.0000000000, worst
max_abs 1.021e-6. **No bug in the ported math** - the two real bugs caught
were in the PORT'S OWN SUPPORT CODE before any golden ran: the cross-fade
`_blend` helper's output length (equals `b`'s length, not
`a_len+b_len-overlap`, caught by re-reading source before implementing) and
`pad2d_reflect.wgsl`'s first draft using a `fn reflect(...)` helper (this
workspace's CPU JIT backend rejects user-defined WGSL function calls -
inlined instead, caught by the weight-free smoke test's first run). Spatial
tiling is explicitly not implemented (the golden is dumped `use_tiling=False`
too, so the comparison is honest - the reference's own docs state tiled and
untiled decode are not numerically equivalent). **Real-weight validation is
written but blocked**: `vae/` has not finished downloading.

## Phase 7 detail: the schedule placeholder was wrong, now fixed

Phase 7 shipped `crate::schedule::DualSchedule` built on
`diffusion::scheduler::FlowMatchEulerScheduler` as an explicitly-flagged
placeholder, with every open question honestly listed in its own module doc.
Reading the real `MiniMaxH3Scheduler` (`scheduling_minimax_h3.py`, 284
lines) while building Phase 9 showed every one of those assumptions was
wrong in a load-bearing way:

- **No `num_train_timesteps` exists anywhere in the class.** Timesteps are
  `t = 1 - sigma` directly in `[0,1]` (`t=1` is clean), never on a
  `sigma*1000` scale.
- **No `invert_sigmas` or equivalent** - the sigma grid is unconditionally
  `linspace(1, 0, num_inference_steps)`, with the **terminal `0.0` counted
  as part of the requested step count itself** (then `unique_consecutive`-
  deduplicated after the shift), not `FlowMatchEulerScheduler`'s
  append-a-terminal-sigma-afterward shape - `len(sigmas)` and every interior
  value genuinely differ between the two.
- **`step()` is not Euler `x_next = x + dt*v`.** It recovers a data-ward
  `x0 = x_t + sigma_from_timestep * v` (note the **`+` sign** - MiniMax-H3's
  transformer predicts a data-ward velocity, the opposite of diffusers'
  usual `x0 = x_t - sigma*v`), using a sigma recovered from the `timestep`
  argument, then blends `x_t`/`x0` by `ratio = sigma_next/sigma` pulled from
  the **separate, deliberately-not-round-tripped** sigma grid (the
  reference's own comment: for sigma < 0.5 the float32 round trip
  `1-(1-sigma)` is not exact, so the two sources are kept apart on purpose).
- Video (`shift=12`) and audio (`shift=3`) are **two instances of the same
  one class**, not two subclasses - modality is pure runtime config state
  (`self._shift`), settable at construction or via `set_shift()`.
- The scheduler itself has **no per-row timestep concept at all** - one
  scalar `_step_index`/`sigma`/`sigma_next` per `step()` call, applied by
  broadcast to the whole tensor. Per-row noise levels are entirely the
  transformer's/pipeline's responsibility upstream; only the `x0` estimate
  (via `sigma_from_timestep`) can vary per-row through broadcasting - the
  actual Euler blend ratio is one shared scalar regardless.

`crate::schedule` was rewritten from scratch against the real source
(`H3Scheduler` + `DualSchedule`, `step()` now takes an explicit `step_index`
argument rather than hidden mutable state); `precompute_adaln.rs`'s one call
site was updated for the new `set_timesteps(&[f32])` signature. Worth
recording as its own lesson: a documented, explicitly-flagged placeholder is
still a placeholder - it does not become correct by virtue of being honest
about being unverified, and this one was caught only because a later phase
happened to read the real source before relying on it.

## Phase 8 detail: AdaLN precompute

`crates/minimaxh3/src/precompute_adaln.rs` - given a fixed denoise step
count N, precomputes every block's `adaln_proj` output over the full
`(3 modalities x N steps)` grid into a small table, dropping the 260M-
param/block `adaln_proj.linear.{weight,bias}` entirely. Verified two ways:
numeric identity against an independent nested-loop f64 reference at every
grid point (worst max_abs < 1e-4, float32-noise level, sharing no code with
either the device path or the module's own `linear_rows` call), and byte
accounting - at the real config's exact dimensions (extrapolated, no real
weights loaded), `adaln_proj` totals **exactly 13,010,457,600 params**
(260,209,152/block x 50), matching the roadmap's independently-derived
"13.0B" line exactly; at a representative N=50 steps the precomputed tables
total only 241.9M params, **removing ≈12.77B of the 13.0B** and landing the
backbone at **≈20.2B params** - in the same neighborhood as this roadmap's
earlier ~19.3B estimate (the residual gap is bf16-vs-f32 storage and the
much smaller `time_embedder`/`norm_out` tensors, out of this phase's scope).
One documented placeholder: text rows' grid timestep reuses the video
schedule's value (no independent third schedule exists), flagged as not
affecting this phase's own correctness claim - an exact cache of whatever
grid it is built over - but real for Phase 9's eventual authority on what
that grid should be.

## Phase 9 detail: t2va/fl2va pipelines

`crates/minimaxh3/src/pipeline.rs` - packed-sequence layout
(`build_packed_sequence`, the `[text | keyframe conditions | target audio |
target video]` order settled above), the full RoPE position-id construction
per row type, and the `DualSchedule`-driven denoise loop tying together the
DiT, both VAEs and the vocoder. The reference's exact `_spatial_position_grid`
(`np.linspace(..., endpoint=False)`) and the "last" keyframe anchor's
**numpy pairwise-summation** algorithm (base case <=8 sequential, 8-wide
unrolled accumulation up to 128, balanced-tree combine, recursive halving
above 128 - transcribed from numpy's own `pairwise_sum` in `loops.c.src`,
not simplified to a sequential sum) are both reproduced exactly, because the
reference's own comment states pairwise vs. sequential summation order
diverges in the last ULP from 16 elements onward. `t2va`/`fl2va` both run
end to end at tiny config, weight-free, in well under 5s: text conditioning
-> packed layout -> denoise loop over the real `H3Transformer` -> video VAE
decode -> vocoder decode, finite/correctly-shaped/non-trivial/correctly-
clamped throughout - the first test in the crate exercising every Phase
5/6/7/9 piece together. `ref2va` is not implemented (see the Phases list
above for why).

Two real gaps surfaced only by full composition, both documented in
`pipeline.rs` itself: (1) `qwen3vl::Qwen3Vl::encode_hidden_with_image`/
`splice_vision` support exactly one spliced image at a construction-fixed
row position, so 2-keyframe `fl2va` TEXT conditioning cannot place both
keyframes in one forward yet - the DiT-side keyframe conditioning ROWS
(which actually anchor the video) are unaffected, since they never touch
the text encoder; (2) the reference's 5-15 second duration bound forces even
the tiny-config smoke test to ~124 real frames (~450 packed rows) despite
every weight/channel dimension being tiny - "tiny" does not imply "short
sequence" in this crate.

## Phase 10 detail: training scaffold

`crates/minimaxh3/src/grad.rs` - a host f64/f32-generic forward + hand-
derived analytic backward for the whole DiT core, independently re-derived
from the device kernel graph (shares no differentiable code with it, per
this workspace's own FD-oracle-independence rule). `crates/gradcheck`'s
`check_minimaxh3` (per-tensor directional FD, all 51 parameter tensors,
worst relative error ~4.9e-8) and `check_minimaxh3_conditioning` (per-entry
elementwise FD on the shared/folded AdaLN sites - `time_embedder`, `norm_out`,
every block's `adaln_proj` - 784 entries, worst abs 2.36e-10) are both green,
both comfortably inside porting.md's 1e-4/1e-3 floors.

**One real bug found and fixed**, cleanly localized by the FD failure
pattern itself: the token-refiner block's backward called `attn_bwd` with
the pre-QK-norm cached buffers instead of the post-norm ones attention
forward actually used (the main block's cache already had this right; the
refiner block's cache dropped the post-norm buffers to save a copy, not
realizing `attn_bwd` still needed them). Before the fix, `check_minimaxh3`
failed exactly on `context_embedder.*` and the refiner block's own
`norm_q`/`norm_k`/`to_q`/`to_k`/`norm1` params (33-50% relative error) while
everything downstream of the fix point passed - the FD gate did exactly the
job it exists to do. LoRA and a device trainer are not built yet.

## Phase 11 detail: serving

`crates/minimaxh3/src/caps.rs` (extended from the Phase 0 license-gate-only
stub) + `crates/cli/src/resident_minimaxh3.rs` (new) + a `catalog`/`resident.rs`
registration - `t2va`/`fl2va` manifest actions, both `.streaming()`, both
license-gated at every entry point. The two-latent-extent instance-key
problem this architecture is the first in the workspace to need (a video
latent extent AND an independent audio latent extent, two different closed
forms over one aligned frame count, not one derived from the other the way
`ltxv`'s audio track is): the instance key names both explicitly
(`"{task}:{video_latent_frames}x{h}x{w}:{num_audio_latents}[@{dtype}]"`), but
- copying `ltxv`'s real, separate, worth-copying pattern - the actual
resident `H3Transformer` is held ONE PER DEVICE in `Arc<Mutex<...>>`,
DECOUPLED from the instance key, because `H3Transformer::forward` takes a
packed sequence of any length per call (no compiled graph is sized to a
request shape the way `wan::pipeline::HotDit` genuinely is) - switching
request shape never forces a 33B reload. D-Bus needed no surface extension:
`Run`/`Subscribe` are already fully generic over `Invocation`/`Outcome`/
`Media::{Video,Audio}`, exercised today by `wan`/`ltxv`.

`h3_dit_param_count` derives the real DiT's parameter count from config
numbers alone, cross-checked against this roadmap's own independently-
confirmed "adaln_proj ~13.0B of 33B" line - lands at ~33.05B, matching.

## Phase 12 detail: DiT forward performance - measured, not guessed

The first end-to-end real-weight run put ONE DiT forward at **~9.7 minutes**
(50 blocks, `BRAIN_DEVICE=cpu`, 128x128 canvas / 124 frames, fp32, 48
threads on 2x Xeon E5-2690 v3 - Haswell-EP, **AVX2+FMA, no AVX-512**). Where
that time actually went was measured per kernel at the real per-layer shapes
before anything was changed, with `crates/minimaxh3/examples/dit_roofline.rs`
(one block's weights, caller-chosen sequence length, so the profiling loop is
seconds rather than the ~50 minutes a real generation takes).

**Two of the three attention kernels had no native CPU path at all.** Of the
thirteen kernels `crate::block` dispatches, only `matmul` and `silu_mul`
carried `@cpu native`; `attn_scores_qk`, `attn_softmax_bidir` and
`attn_apply_full` ran as Cranelift-JIT **scalar** code, one output element per
invocation. `attn_apply_full` measured **5.4 GFLOP/s** - a third of the whole
forward in one kernel.

**And the GEMM was bandwidth-bound, not compute-bound.** `fast_ops::matmul_abt`
walked the whole of B once per row of A, so a thread holding `R` rows moved
`R·n·k·4` bytes; at the FFN shape that is a 308 MB weight streamed 20x per
thread. It measured ~130 GFLOP/s against a ~2 TFLOP/s AVX2 peak - i.e. at
essentially 100% of memory bandwidth and ~7% of FLOPs, where neither more
threads nor a wider vector could have helped.

### What changed

1. `crates/backend-cpu`: native CPU paths for the bidirectional self-attention
   trio, routed into the EXISTING cross-attention fast ops (`attn_apply_full`
   is the `tq==tk`, zero-offset case of `attn_apply_cross`; `attn_softmax_bidir`
   of `attn_softmax_cross`), plus one new `fast_ops::attn_scores_qk` sharing
   `attn_scores_cross`'s packing helper. `attn_scores_qk`'s `causal != 0` arm
   deliberately stays on the JIT rather than being emulated.
2. `fast_ops::matmul_abt`: a 3x4 register-blocked microkernel (12 accumulators
   + 3 A vectors + 1 B temporary = exactly Haswell's 16 `ymm`, so the inner
   loop spills nothing and the 12 independent FMA chains cover the 5-cycle FMA
   latency) under a **column-outer** nest that holds a B tile resident and
   sweeps a thread's A rows through it. Selected by shape, not unconditionally:
   only when B is both the bigger operand and too big to stay cached
   (`n >= rows.max(4) && n*k > 512Ki` floats). Every shape that fails that test
   keeps the original kernel AND the original chunking, unchanged.

### Measured, same-host confirmation

Per-kernel, `dit_roofline 960 5`, `BRAIN_DEVICE=cpu`, machine otherwise idle
(the numbers below are GFLOP/s; both columns measured back to back on the
same build tree, only `crates/backend-cpu` differing):

| op (real per-layer shape, seq 960)   | before | after | x    |
|--------------------------------------|-------:|------:|-----:|
| `matmul` q/k/v `[S,5376]x[7168,5376]T`  | 138.4 | 404.1 | 2.92 |
| `matmul` to_out `[S,7168]x[5376,7168]T` | 151.0 | 362.3 | 2.40 |
| `matmul` fc1 `[S,5376]x[14336,5376]T`   | 132.5 | 304.5 | 2.30 |
| `matmul` fc2 `[S,14336]x[5376,14336]T`  | 136.3 | 195.1 | 1.43 |
| `attn_scores_qk`                        |  13.7 | 132.6 | 9.68 |
| `attn_apply_full`                       |   5.4 |  88.0 | 16.4 |
| `attn_softmax_bidir`                    |   5.0 |   7.5 | 1.48 |
| `adaln_proj` (host `linear_rows`)       |  35.0 |  36.5 | 1.04 |

Summed over 50 blocks that is **453.0 s -> 147.8 s per forward, 3.07x**
(against the ~9.7 min a real forward measured; the probe models the block
stack, not `proj_in`/`proj_out`, the token refiner, or the scatter/gather).
Correctness gate: `model::tests::tiny_config_matches_the_real_reference_
numerically` stays at **worst cosine 1.0000000000** over all 8 taps, and
`matmul_shape_bench` reports **max rel err 0.00e0** between the two nests at
every shape.

### `adaln_proj` precompute is a MEMORY lever, not a speed one

Phase 8's claim stands exactly as written, but it is worth recording what it
is and is not worth at inference. `adaln_proj` is 13.0B of the 33B params, yet
at `num_timesteps = 2` its per-forward cost is a skinny `n=2` mat-vec, not a
GEMM: **28.5 ms per block, 1.4 s per forward - 0.9% of the post-change
forward**. Its value is the 52 GB of resident fp32 host memory it removes
(1.04 GB/block x 50), which is what decides whether this model can be placed
on a GPU at all - not the FLOPs.

### GPU placement: measured, and blocked on the kernel, not the VRAM

`dit_roofline` was run against a real Tesla P40 (`BRAIN_DEVICE=vulkan`). The
first run reported 57 TFLOP/s for a 5376x7168 GEMM - roughly 5x the card's
entire fp32 peak, because `submit` only queues work and the harness was
timing submission. With a readback fencing each timed region, the honest
numbers are:

| op                | P40 (fenced) | CPU (after) |
|-------------------|-------------:|------------:|
| `matmul` q/k/v    |    18.2 GF/s |   404.1 GF/s |
| `matmul` fc1      |    19.6 GF/s |   304.5 GF/s |
| `attn_apply_full` |   456.1 GF/s |    88.0 GF/s |
| `rope2d_partial`  |    40.6 GF/s |     3.9 GF/s |

So the P40 is **~20x SLOWER than the optimized CPU on the GEMMs** and several
times faster on everything else. The cause is kernel selection, not the
hardware: `crate::block::linear` dispatches `matmul.wgsl`, which is one thread
per output element with a serial k-loop and no tiling or workgroup memory -
`m·n·k·2·4` bytes of global traffic (592 GB for one FFN GEMM). At 19.6 GFLOP/s
it is running at ~0.2% of the card's ~11.8 TFLOP/s fp32 peak. This workspace
already ships `matmul_tiled`/`matmul_reg`/`matmul_reg2`/`matmul_reg3` for
exactly this, and `backend-cpu` already routes all of them to the same native
path, so switching the DiT's dispatch is not a fork.

**The order of work is therefore the opposite of the obvious one**: fixing GPU
matmul kernel selection comes FIRST, before any VRAM/sharding effort, because
until it lands a perfectly-sharded model would run slower than the CPU does
today.

### Fixed: `crate::block::linear` now routes through `model::block::pick_gemm`

Root cause, confirmed by reading the kernel source rather than assumed from
the "no tiling" framing above: `matmul.wgsl` indexes `col = idx % n`, so
adjacent GPU threads in a warp read weight rows `k` floats apart - a fully
uncoalesced access pattern, not merely an untiled one, which is why the
measured throughput (~0.2% of the card's fp32 peak) is far below what
"untiled but coalesced" would cost. This was never a hardware or kernel-
authoring gap: `model::block::pick_gemm` (backed by `backend_api::select`'s
`Op::MatMul` policy, which already answers `RegisterTiled` for exactly this
crate's shapes - large `m`, wide `n`) is the SAME seam ~15 other model
crates (`qwen3`, `wan`, `clip`, `t5encoder`, `vae`, `sam1`, `gpt2`, ...)
already call with `matmul`/`matmul_reg3` as their naive/tiled pair.
`minimaxh3` is the only DiT-family crate that never adopted it - `crate::
block::linear` hardcoded `K_MATMUL` instead. Fixed by registering
`matmul_reg3` (`matmul_reg2`'s tiling with its two Pascal shared-memory
bank-conflict patterns removed - the P40-tuned variant, exactly this box's
hardware) in the crate's own `KERNELS` table and routing `linear`'s kernel
choice + dispatch geometry through `pick_gemm(m, n, K_MATMUL, K_MATMUL_REG3,
false)`. On CPU this changes nothing observable (`backend-cpu`'s native
fast path already treats `matmul`/`matmul_reg3` identically by kernel name -
confirmed via a full real-weight layer-by-layer re-run, worst cosine
unchanged at 0.9999999764). On a real GPU it is the actual fix: the
existing, already-fenced, already parity-checked `crates/gpu-core/tests/
bench_matmul.rs` (`--ignored`) measures the SAME naive-vs-tiled gap on this
box's own P40 at comparable large shapes (naive ISN'T H3-exact but is the
same kernel/pattern):

| shape (closest proxy to H3's linears) | naive (Vulkan) | reg2 (wgpu, tiled) | CPU AVX2 (this box, post-optimization) |
|---|---:|---:|---:|
| square 2048                     | 36.6 GF/s |  3996 GF/s (34.0% of peak) | 717 GF/s |
| glm mla-ish (k=6144, closest to fc2's k=14336) | 22 GF/s | 3026 GF/s (25.7% of peak) | 348 GF/s |

So the tiled kernel is not merely "no longer 20x slower than CPU" - once
dispatched correctly it is **4-12x FASTER than the already-optimized CPU
path** at comparable shapes, which is the outcome the "P40 is 12 TFLOP/s
fp32 / 47 TOP/s int8" datasheet numbers always implied was available.
`dit_roofline`'s own per-kernel probe (which still hardcodes the naive
`matmul` kernel independently of this fix, to keep measuring the "before"
baseline) hit `BRAIN_GPU_WAIT_S`'s 30s default timeout at H3's real
`seq_len=960` widths - not a device wedge, just the naive kernel legitimately
taking that long per call at this scale, further confirming how severe the
gap was. Re-measuring `dit_roofline` itself at the exact H3 shapes with the
fix wired through it (not just `bench_matmul`'s proxy shapes) is not yet
done - the roofline example's own kernel table would need widening to
register `matmul_reg3` and call `pick_gemm`, mirroring `crate::block::linear`.

None of this unlocks a real GPU generation on its own: the 33B fp32 DiT
still does not fit in 48GB across the two P40s (see the VRAM table below) -
that still needs Phase 8's AdaLN precompute wired into loading and a
bf16/int8 storage tier on `Ctx::upload`, neither of which exist yet.

The VRAM budget for when that is done (2x24 GB = 48 GB):

| tier | resident params | bytes | fits? |
|------|-----------------|-------|-------|
| fp32, as-is                    | 33.05B | 132 GB | no |
| fp32, AdaLN precomputed        | 20.2B  |  81 GB | no |
| bf16/f16 storage, precomputed  | 20.2B  |  40 GB | only just - ~4 GB left for activations, and `scores`+`probs` alone are 2.8 GB at a 256px canvas |
| int8, precomputed              | 20.2B  |  20 GB | comfortably, and the P40 has real DP4A |

`Ctx::upload` is fp32-only today (no `Weight`-enum tier like `crates/qwen3`'s)
and Phase 8's precompute is a checkpoint transform not yet wired into `load` -
both are prerequisites for a RESIDENT multi-card DiT, and neither is started.
Placement itself is no longer one of them: see the next section.

### Multi-GPU: the block stack is placed across every schedulable card

The streaming denoise loop opened exactly ONE device, so on this two-card box
every block of every step ran on gpu0 while gpu1 stayed at 1 MiB used for the
whole generation - and gpu0 was the card that ran out of memory. That is now
`crate::dit_shard`: `model::StreamPlan` (the model-agnostic half of
`model::shard`, cut by the same `plan_balanced` exact DP the resident
pipelines in `gpt2`/`ltxv`/`qwen35`/`minimaxmusic3` already use) gives each
schedulable card a contiguous block range, and
`H3Transformer::forward_streaming_with_taps` runs one `crate::block::Ctx` per
stage. Only the residual stream crosses a cut, host-staged - lossless for
f32, so a split plan is bit-identical to the single-stage one
(`model::tests::a_two_stage_split_is_numerically_identical_to_the_single_
stage_path` gates that on any machine, GPU-less included).

Streaming and sharding COMPOSE - each card still streams its own range one
block at a time, and this is deliberately not "there is room now, load
everything". What that buys and what it does not:

* It does **not** halve the per-card peak. A streamed stage's live footprint
  is one block's weights (~2.6 GB fp32) plus that step's activations and
  scratch, and every stage pays that whatever its range length. Splitting
  changes how MANY blocks a card runs, not what one block costs.
* It does move the endpoint weights apart: the token refiner (2 full-width
  blocks, ~2.8 GB fp32, loaded per forward) is on the first stage only and
  `norm_out`/`proj_out` on the last, where before one card held both.
* It halves each card's per-step block uploads and its share of the
  allocator churn that `crate::block::load_block_streaming`'s own doc
  records as the failure mode (Vulkan suballocator fragmentation, invisible
  to `nvidia-smi`'s reserved-byte counter).
* It does **not** speed up one denoise step: with no CFG pass (H3 is
  guidance-distilled) there is one sample in flight, so the stages run
  strictly in sequence. A pipelined schedule needs two independent forwards
  to overlap, which this architecture does not have. Weight PREFETCH is the
  real overlap available here - card `i+1` uploading its first blocks while
  card `i` computes - and is not implemented.

Left for a follow-up, deliberately, not forgotten: **the video VAE decode is
still single-card**. `video_vae::decode_clip_untiled`'s 36-layer loop has the
same shape as the DiT's block loop and would take a `StreamPlan` the same
way, but the VAE's better multi-card shape is its OUTER loop - `decode`/
`decode_clip_in` process spatial TILES that share no state, which is
embarrassingly parallel across cards rather than a pipeline, and is a
different mechanism (`ltxv`/`minimaxmusic3`'s `devplan` CFG-split shape) from
the layer sharding done here. Neither is wired; the DiT was the memory-hungry,
user-visible half and went first.

## Recorded gaps (kept current)

- ~~The DiT's GPU path dispatches the untiled `matmul.wgsl` and therefore
  measures ~20x slower than the optimized CPU path on a P40~~ - fixed:
  `crate::block::linear` now picks its kernel via `model::block::pick_gemm`
  (the same seam ~15 other model crates already use), which selects
  `matmul_reg3` for this crate's large/wide shapes - see the GPU-placement
  detail above for the measured before/after. `BRAIN_DEVICE=vulkan` is no
  longer a pessimization for the matmul family; `dit_roofline`'s own probe
  still needs widening to exercise the fix at H3's exact shapes (it
  currently measures only the pre-fix naive baseline).
- Eight of the thirteen kernels `crate::block` dispatches still run as
  Cranelift-JIT scalar code on CPU (`rmsnorm_eps`, `embed`, `row_scatter`,
  `bias_add`, `rope2d_partial`, `mul`, `add2`, `gate_row`). Together they are
  ~9 s of the 147.8 s post-change forward (~6%) - worth a native pass, but no
  longer where the time is.
- `matmul` fc2 (`k=14336`) reaches 195 GFLOP/s against fc1's 304: at that `k`
  the 3x4 tile's working set (3 A rows + 4 B rows = 401 KB) exceeds the 256 KB
  L2, so it wants a `k`-blocked accumulating variant. Not attempted - it needs
  the microkernel to accumulate into C across k-chunks rather than write it.
- `H3Transformer::forward` always computes the parity taps
  (`forward_full` reads back block 0's output and the packed input
  unconditionally, ~62 MB of device->host copies per forward). Free to skip on
  CPU, but a real pipeline stall once a GPU path exists.
- ~~`H3Recipe`/`import.rs` assume the old FL2VA/Ref2VA layout~~ - both
  retargeted to the root flat layout; `H3Recipe` also had a real ordering
  bug fixed alongside it (it would have lost to `ZimageRecipe`'s own
  `matches`, since H3's real repo carries all four of Z-Image's role dirs as
  a subset of its own nine).
- `crates/cli/src/supply.rs::convert` has no case for recipe id
  `"minimaxh3"` - a completed `brain pull` is not yet turned into a servable
  manifest that way; `minimaxh3::import` reading `BRAIN_MINIMAXH3_DIR`
  directly remains the supported path.
- ~~Real DiT weight import does not exist~~ - **settled from source, not
  needed at all**: the original concern assumed the real checkpoint's QKV
  was fused (`qkv_proj`, the OLD FL2VA legacy layout's own shape) and would
  need splitting to match `H3Transformer::load`'s `to_q`/`to_k`/`to_v`
  naming. Direct inspection of the real, downloaded root-layout
  `transformer/diffusion_pytorch_model-00001-of-00014.safetensors` header
  shows the checkpoint ALREADY stores split `to_q`/`to_k`/`to_v` (and every
  other tensor `H3Transformer::load` names - `adaln_proj`, `ff.net.{0,2}`,
  `proj_in`/`proj_out`, `token_refiner.refiner_blocks.{i}`, all confirmed
  byte-for-byte against the real header) - zero conversion code needed, only
  the sharded `read_model_dir` loader `crate::caps::read_tensors` already
  uses. Not yet run end to end only because `transformer/` (3 of 14 shards
  so far) has not finished downloading - once it does, `LoadedWeights::load`
  should work directly, no new import function to write.
- ~~No real Qwen3-VL text encoder is wired into serving~~ - `crate::caps::
  build_text_encoder`/`encode_text_real`/`text_conditioning` wire a real
  `qwen3vl::Qwen3Vl` + tokenizer into every `t2va`/`fl2va` entry point,
  falling back to the stub only when no real checkout is present (never
  silently on a real load failure). `text_encoder/`'s download completed
  (63GB bf16) and hit a real host OOM on the first two load attempts
  (`f32`, then `int8` destination - see below); fixed by streaming the
  decoder's weights instead of materializing them eagerly.
- ~~Real Qwen3-VL text-encoder load OOMs the host~~ - root-caused, not
  guessed: `qwen3vl::Qwen3Vl::from_hf` used
  `checkpoint::safetensors::read_model_dir`, which decodes EVERY tensor to
  f32 before `Qwen::new_shard_dt_decode` allocates a single destination
  byte - for `text_encoder/`'s 63GB-bf16-on-disk checkpoint that is an
  unavoidable ~126GB source materialization, independent of the caller's
  requested destination dtype. Two real runs under a 150GB container cap
  confirm this: switching the destination from f32 to int8 (which should
  roughly halve a "2x-during-construction" peak, the original hypothesis)
  made no measurable difference - both climbed to the identical ~149GB
  ceiling on nearly the same timeline and were SIGKILLed, because both were
  still dominated by the same dtype-independent source decode, never
  reaching the point where the destination dtype would matter. Fixed by
  extending `Qwen3Vl::from_hf` to stream the decoder
  (`model.language_model.*`, the dominant byte share) through
  `checkpoint::weightio::WeightReader` + a new
  `qwen3vl::import::decoder_source` (a `checkpoint::remap::RemapSource`),
  the same mechanism `qwen3::import::hf_shard_source` already gives
  FLUX.2's text encoder - never brain infrastructure that didn't already
  exist, just not yet wired into `Qwen3Vl`. The vision tower/mergers stay
  on the eager path (a small fraction of the checkpoint's bytes; streaming
  them would not move the peak). Re-run under the same 150GB cap climbed
  smoothly and predictably instead of rocketing to the kill point -
  confirms the fix, see this entry's own commit for the measured curve.
  General lesson for any future large-checkpoint loader in this tree:
  `read_model_dir`'s "decode-dtype-matches-source, not destination" design
  makes picking a cheaper destination dtype alone NOT a memory fix when the
  source itself already exceeds budget - the loader has to stream, not
  just the destination has to shrink.
- ~~Streaming decoder's per-layer QUANTIZED linears loop never released
  mmap pages~~ - `qwen3::model::new_impl`'s int8/f16/bf16 weight-build loop
  (the one that reads the 7 per-layer linears - QKVO + MLP gate/up/down,
  the overwhelming majority of a transformer's bytes) is a SEPARATE code
  path from `paramstore::new_with_roles_src`'s own loop (the one the first
  streaming fix above patched), calls the unbounded
  `TensorSource::with_tensor`, and had no `advise_drop` call at all. Fixed
  by adding one, mirroring the paramstore-loop fix.
- ~~Even truncated to 50/64 layers, memory still climbed to ~140GB on a
  build that should need ~25GB at int8~~ - **root cause was NOT page
  cache** (see `.agents/rules/lessons.md` #87 for the general lesson):
  `/proc/<pid>/smaps_rollup` showed `Pss_File` at ~132MB (the streaming fix
  above was already working) against `Anonymous`/`Private_Dirty` at ~90GB -
  real heap growth. `gpu_core::select::Dtype::I8.promote(&numeric)` demotes
  to `Dtype::F32` when `numeric.int8_dot` is false, and the CPU JIT
  backend's `int8_dot` IS false (`backend-cpu`'s own `Caps`: "the
  multi-barrier packed-int8 GEMMs are outside the JIT's single-barrier
  model, and there is no VNNI fast path yet") - so `build_text_encoder`'s
  `Dtype::I8` request had been silently landing as fp32 (4x the intended
  bytes) through every real-weight attempt in this port's history so far,
  including the very first "does int8 help vs fp32" A/B that found "no
  difference" (there was none to find - both requests built the identical
  fp32 model). Fixed: `caps::best_linear_dtype()` now queries
  `Gpu::caps().numeric` and picks int8 only where `int8_dot` is genuinely
  available, else bf16 (CPU's `bf16_storage` IS true - a real, honest 2x,
  not a second silent no-op) instead of fp32.
- The CPU backend has NO real int8 compute path at all (`int8_dot: false`,
  no VNNI fast path yet) - `minimaxh3`'s text encoder (and anything else on
  this host wanting int8's real ~4x-vs-fp32 density) can only actually get
  it by running on the two Tesla P40s, which is not yet attempted for this
  component. The P40s sit idle for every real-weight run in this ledger so
  far; whether a 50-layer-truncated ~25GB int8 shard fits one 24GB card, or
  needs splitting across both, is unmeasured.
- Measured, real-weight confirmation of the two memory fixes above, same
  50-layer-truncated `text_encoder/` load, same 150GB container cap, same
  test (`text_conditioning_uses_the_real_encoder_when_weights_are_present`),
  each run's peak read from `/sys/fs/cgroup/memory.current`:
  | Build | Peak cgroup memory | Wall clock |
  |---|---|---|
  | Whole 64 layers, decode_only, requested int8 (silently fp32) | ~149GB (right at the cap) | 615.86s (then hit an unrelated `forward_steps` panic on decode_only - see the `encode_hidden` fix above) |
  | Whole 64 layers, decode_only, requested int8 (silently fp32), `encode_hidden` fixed | ~149GB (plateaued at the cap) | 615.86s, passed |
  | Truncated 50 layers, batched, requested int8 (silently fp32) | ~139GB | 269.47s, passed |
  | Truncated 50 layers, batched, requested int8, mmap `advise_drop` added to the per-layer-linears loop | ~139GB (page cache was never the dominant cost - confirmed via `smaps_rollup`) | 263-269s, passed |
  | Truncated 50 layers, batched, HONEST bf16 (`caps::best_linear_dtype` queries `Gpu::caps().numeric` instead of hardcoding int8) | ~103GB | 263.59s, passed |
  Truncation (64→50 layers) cut wall clock by more than half on its own
  (batched forward vs. the sequential per-token decode loop the FIRST row's
  `encode_hidden` fix made correct but never fast). The bf16 fix is the
  only row that actually moved peak memory - confirming int8 was never
  real on this backend for any earlier row, including the ones that looked
  like an "int8 vs fp32, no difference" A/B.
- `t2va`/`fl2va`'s denoise loop threads neither cancellation nor per-step
  progress yet (`wan`/`ltxv` both poll `inv.cancel` per step; this one does
  neither) - `.streaming()` on the manifest currently means "long-running"
  only.
- `fl2va`'s 2-keyframe TEXT conditioning (not the DiT-side conditioning
  rows, which are unaffected) needs `qwen3vl::Qwen3Vl` to splice two images
  into one presentation - it currently supports exactly one (see Phase 9
  detail).
- `video_vae/` not yet downloaded; Phase 6's real-checkpoint validation
  (written, gated on `BRAIN_MINIMAXH3_DIR`) is blocked on it.
- ~~`crate::caps::read_tensors` (the DiT loader) still uses eager
  `read_model_dir`, will OOM once `transformer/` finishes downloading~~ -
  fixed: `crate::caps::open_dit_reader` replaces it with a lazy, mmap-backed
  `checkpoint::weightio::WeightReader::open_hf_dir`; `H3Transformer::load`
  (resident serving) and the new `H3Transformer::forward_streaming_with_taps`
  (validation/one-shot use) both take `&dyn checkpoint::TensorSource`. Block
  loading itself was extracted into shared free functions in `block.rs`
  (`load_dev`/`load_host`/`load_fc1`/`load_attn`/`load_block`/
  `load_refiner_block`) so both callers load identically.
- ~~The DiT was never validated against REAL trained weights, only
  tiny-config random ones~~ - `tools/minimaxh3_dit_real_dump_reference.py`
  loads the REAL installed `diffusers` `MiniMaxH3Transformer3DModel` at the
  real checkpoint's full width/depth (33,122,992,896 params, confirmed) and
  dumps taps at block 0/mid(25)/last(49) plus rope/temb/refiner/final
  outputs from a small 9-row synthetic packed sequence; `model::tests::
  dit_matches_the_real_reference_numerically_layer_by_layer` compares all
  11 against `H3Transformer::forward_streaming_with_taps` (one block
  resident at a time - **never the eager whole-model load**, a hard
  requirement for validation as much as for serving: even the smaller
  10-row-sequence test would otherwise pull all 50 blocks/~132GB fp32
  resident just to check numbers). Result: worst cosine 0.9999999620 (at
  `output_audio`), everything else 0.999999969-1.000000000, all at
  float32-noise-level `rel_l2`/`max_abs` - the full stack (real-weight
  loading, tensor-name/role mapping, refiner, RoPE, AdaLN indexing, all 50
  blocks, both output heads) is numerically right, not merely
  "produces plausible-looking output". Peak RSS during the whole 148s run:
  6GB. One real bug caught and fixed along the way: `rope::build_tables`
  intentionally returns only the `half = 3*rope_freq_dim`-wide table (see
  its own doc), but the reference's own `rope.forward` doubles that to
  `2*half` before `cos`/`sin` (the `rotate_half` convention - channel `m`
  and `m+half` share one angle); the tiny-config parity test already
  accounted for this (`first_half_cols`), the new real-weight test
  initially did not and failed with a raw length mismatch (432 vs 864)
  until the same helper was applied - a test-construction bug, not a model
  bug, but worth recording since it is exactly the kind of shape mismatch
  that could otherwise be misread as a real numeric failure.
- **Why "brain already has heavily optimized kernels" did not apply
  automatically to the DiT's attention**: `backend-cpu`'s native-fast-path
  routing (`CpuBackend::new`'s `FastIdx`) is a plain runtime string lookup
  (`names.iter().position(|n| n == k)`) keyed on kernel-source names a
  model's own `KERNELS` table supplies - there is no compile-time or
  typed-identifier mechanism preventing a name mismatch; a WGSL kernel's own
  `@cpu native` header tag is documentation only; nothing cross-checks it
  against `FastIdx` at build or test time. `minimaxh3::block.rs` dispatches
  `attn_scores_qk`/`attn_softmax_bidir`/`attn_apply_full` (the bidirectional
  self-attention trio a packed-sequence DiT needs), which `FastIdx` simply
  had no entries for until this session's optimization pass - the JIT still
  compiled and ran them (no crash, no warning), just one output element per
  invocation. Fixed cleanly, not by duplicating kernel math: `attn_scores_qk`
  is a 2-line wrapper over the same `scores_packed` helper `attn_scores_cross`
  already used; `attn_softmax_bidir`/`attn_apply_full` dispatch straight into
  the pre-existing `attn_softmax_cross`/`attn_apply_cross` functions (the
  `tq==tk`, zero-offset case of the general cross-attention math) - zero new
  fast-path implementations, only 3 new name-table entries + ~50 lines of
  dispatch. The 3 WGSL files themselves needed only a header-tag fix
  (`@cpu yes` -> `@cpu native`); they are not redundant/removable - they
  remain the real GPU dispatch source and the CPU JIT fallback. Remaining
  gap: `crates/backend-cpu/tests/matmul_family_native_fastpath.rs` guards
  only the multi-barrier hard-panic case; nothing yet guards this silent
  "JIT-compilable but no native path" case in general, so the same class of
  bug (a new kernel name a model adopts, with no matching `FastIdx` entry)
  can recur for the next model with no test failure, only a silent 10-20x
  slowdown - worth a generic cross-check test (parse every kernel's
  `@cpu native` tag, assert a live `FastIdx`/equivalent entry exists for
  every model's own `KERNELS` table) but not written this session.
- `ref2va` is not implemented (Phase 9's own deliberate scope cut - its
  layout needs its own builder, not a generalization of t2va/fl2va's); the
  DiT/scheduler/VAE machinery it would reuse is otherwise ready once
  `transformer_ref/` finishes downloading.
- The golden-dump script Phase 1 originally named
  (`tools/minimaxh3_dump_reference.py`) never got written as one script -
  instead, three per-component dumpers now exist and are in real use
  (`tools/minimaxh3_{audio_vae,dit,video_vae}_dump_reference.py`), each
  gated by its own crate test. `ref2va`'s and the scheduler's own dumpers
  (if any is still needed beyond `crate::schedule`'s existing closed-form
  tests) do not exist yet.
