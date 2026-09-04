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

## Recorded gaps (kept current)

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
  silently on a real load failure). Validated structurally (fallback
  behavior, error propagation); the real-encoder numeric path itself is
  still gated on `text_encoder/`'s download, which HAS now completed
  (`text_conditioning_uses_the_real_encoder_when_weights_are_present`
  should now run for real rather than skip - re-check once this ledger
  entry is next touched).
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
