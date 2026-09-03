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

### Still open - settle before claiming the relevant phase's parity
- [ ] The dual schedule's base (pre-shift) sigma spacing, `num_train_timesteps`
      and `invert_sigmas` - `scheduling_minimax_h3.py` not yet read;
      `crate::schedule::DualSchedule` still assumes the Z-Image/FLUX.2
      defaults as a documented placeholder.
- [ ] `video_vae`'s exact causal-3D-encoder / non-causal-ViT-decoder
      conventions - `autoencoder_kl_minimax_h3.py` not yet read; per the
      wan/ltxv precedent, do not assume either one's chunking/caching/
      padding/norm-axis conventions transfer.
- [ ] `ref2va`'s full packed layout beyond the structural skeleton already
      seen (`MiniMaxH3Ref2VAPrepareLayoutStep`'s full body, `denoise.py`'s
      `MiniMaxH3Ref2VADenoiseStep`, the audio-reference summation-order
      trap noted above) - not yet read in detail.
- [ ] The Qwen3-VL layer-50 extraction call site itself
      (`modular_pipelines/minimax_h3/encoders.py`) - confirms exactly how
      `hidden_states[50]` is read (pre/post final-norm at that depth,
      whether M-RoPE/DeepStack apply identically to brain's existing
      `qwen3vl::Qwen3Vl::encode_hidden` path) - not yet read.
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
- [~] Phase 4 - audio VAE **decoder** port: structurally complete and
      real-checkpoint-validated, numeric parity still open (see below)
- [ ] Phase 5 - H3 DiT core, tiny-config -> real-weight parity ladder.
      Structurally ready to start: the block math, AdaLN indexing, RoPE and
      t2va/fl2va packing are now fully known from source (see above).
- [ ] Phase 6 - video VAE (gated on `video_vae/` download AND on reading
      `autoencoder_kl_minimax_h3.py`)
- [x] Phase 7 - dual rectified-flow schedulers (`DualSchedule`) - base sigma
      spacing/`num_train_timesteps`/`invert_sigmas` still assumed, pending
      `scheduling_minimax_h3.py`
- [ ] Phase 8 - AdaLN precompute checkpoint transform
- [ ] Phase 9 - t2va / fl2va / ref2va pipelines - t2va/fl2va's packing is
      now fully known; ref2va's is not yet
- [ ] Phase 10 - training: gradcheck -> LoRA -> device trainer
- [ ] Phase 11 - capability / residency / D-Bus serving contract
- [ ] Phase 12 - streaming-overlap engine (separate, measured perf phase) -
      `_cp_plan` in the real transformer source is a useful blueprint,
      not yet studied in detail

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

**Still open, explicitly not claimed:** numeric parity against the real
reference implementation. The oracle is now installed (Phase 1), but the
audio VAE's own golden dump has not been written/run yet - "runs and stays
finite at real scale" remains the honest ceiling on what is verified so far.

## Recorded gaps (kept current)

- **`H3Recipe` (`modelstore::recipe`) and `crates/minimaxh3/src/import.rs`'s
  real-checkpoint test paths both assume the old FL2VA/Ref2VA nested
  layout** - the download strategy changed to the root flat
  (`transformer/`+`transformer_ref/`+shared `text_encoder/`+`vae/`+
  `audio_vae/`) layout mid-session once its ~290GB of redundancy was
  discovered. Updating both is explicitly deferred until the current
  download (handed to the user, not yet confirmed run) completes.
- `crates/cli/src/supply.rs::convert` has no case for recipe id
  `"minimaxh3"` - a completed `brain pull` is not yet turned into a servable
  manifest that way; `minimaxh3::import` reading `BRAIN_MINIMAXH3_DIR`
  directly remains the supported path.
- `video_vae/` not yet downloaded; `autoencoder_kl_minimax_h3.py` not yet
  read. Phase 6 is blocked on both.
- The `Ref2VA/` legacy partition was never downloaded (correctly - it was
  redundant under the layout switch); `transformer_ref/` (the root-layout
  equivalent) is part of the pending download. `ref2va`'s full packing logic
  (`MiniMaxH3Ref2VAPrepareLayoutStep`, `denoise.py`) is not yet read.
- The golden-dump script (`tools/minimaxh3_dump_reference.py`) has not been
  written. Most structural questions it would have settled were instead
  settled by reading source directly (see "Convention questions"); real
  numeric goldens for the parity ladder still need it.
