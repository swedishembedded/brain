# ltxv - roadmap

LTX-2.5: a **two-stream** (video + audio) PixArt-alpha-style DiT, 48 layers, video
stream 32 heads x 128 = 4096 dim / audio stream 32 x 64 = 2048 dim, coupled every
block by bidirectional audio<->video cross-attention, with **per-token timesteps**
(diffusion forcing: `timesteps = denoise_mask * sigma`, shape `(B,T,1)`) driving a
PixArt adaLN-single modulation table. Text conditioning is a separate cross-attention
(SDXL/Wan topology, never concatenated into the sequence) from a 12B Gemma-4 encoder.
A causal 3D video VAE (stride T8/H32/W32, 128 latent channels) ships with two
decoders - a conv mirror and a 3D neighborhood-attention "diffusion decoder"
(`NADiffusionDecoder`, zero convolutions) - plus a 2D-causal-conv audio VAE (log-mel
in, continuous Gaussian latent), a BigVGAN+snakebeta vocoder with a bandwidth-
extension stage, two latent upscalers, and a duration-prediction head. `ltxv` names
the family (LTX-2.3/2.4/2.5 share one `AVTransformer3DModel` class and one GGUF
architecture tag), the release is a config.

The port follows `.agents/rules/porting.md` in order, staged **video -> audio ->
DiffVAE/DFR** (see "Staging" below) because the video path alone is already a full
port's worth of new structure (per-token modulation, split-RoPE, the embeddings
connector) and audio adds a second VAE+vocoder+cross-attention surface on top.

Reference material: official repo `github.com/Lightricks/LTX-2` (math authority,
public even though the HF weights are gated) cloned under
`resources/ltxv/source/`; real checkpoint headers of `Lightricks/LTX-2.5` (gated,
needs an HF account with access accepted and `hf auth login`) are the
tensor-naming + exact-config authority - **the code repo's own module defaults
are for the now-superseded 19B checkpoints**, not 2.5; an LTX-2.3 GGUF header
(`unsloth/LTX-2.3-GGUF`) is the third opinion / GGUF-detection authority.

## Validation policy for the large components

The 22B DiT (42 GB bf16 / ~88 GB fp32) and the 12B Gemma-4 text encoder (26 GB
bf16) are large enough that many machines - including the one this port was
started on - cannot build or run them at all, not merely slowly. Policy for
this port:

- Every component small enough to run on modest hardware (both video VAEs, the
  audio VAE, the vocoder, both latent upscalers, the duration head - about
  4.5 GB total) gets **real-weight parity gates**, same bar as every other port
  in this repo.
- The DiT, the embeddings connector, and Gemma-4 get **tiny-config** (few layers,
  small dims) **layer-by-layer parity** against the reference, proving the op
  sequence and every convention (chunk order, RoPE layout, table indexing) is
  correct at a scale that fits everywhere, plus **import coverage asserted
  against the real 4349-tensor / 686-tensor headers** even without a forward at
  that size.
- Full 22B / 12B real-weight validation is an explicit, tracked gap (see below),
  not a silent omission - `brain:validate-existing-model` should re-run it the
  day this lands on hardware that can hold it.

## Not yet done

- [ ] **`ltxv` arch row + resource fetch** - `crates/arch/src/lib.rs` `ARCHS` row
      (`id`/`gguf` both `"ltxv"`, `hf: &["AVTransformer3DModel"]`,
      `default_ref: Lightricks/LTX-2.5`), this ledger, `docs/models/ltxv.md`,
      weights under `resources/ltxv/weights/` (the runnable-everywhere subset
      only), source under `resources/ltxv/source/`.
- [ ] **Shared DiT hoist** - `crates/dit` currently owns only RoPE despite its doc
      claiming adaLN/patchify/QK-norm, so `wan`/`s3dit`/`flux2` each re-implement
      PixArt timestep embedding and patchify/unpatchify. Hoist a shared
      `timestep_embedding`, `patchify`/`unpatchify`, and a **per-token** adaLN
      table helper (Wan's and s3dit's folds are both token-INDEPENDENT and would
      silently produce the wrong result if reused for LTX) into `crates/dit`, and
      migrate `wan` + `s3dit` onto them in the same change - no per-model copies,
      per [[brain-evolve-core-for-models]].
- [x] **`LTX2Scheduler`** (`diffusion::scheduler::ltx2_sigmas`, token-count-
      dependent Flux-style shift + terminal stretch to 0.1) added to
      `crates/diffusion/src/scheduler.rs` next to the existing `flow_shift`/
      `time_shift_exponential`, not inside `crates/ltxv`. Validated against
      `testdata/golden/ltxv/schedule/` across all 10 dumped
      `(tokens, steps, base_shift, max_shift, stretch, terminal)` cases -
      worst max abs deviation 1.79e-7 (`crates/diffusion/tests/
      ltxv_schedule_parity.rs`), plus the three real `DISTILLED_*_SIGMAS`
      constant tables (`LTX2_DISTILLED_SIGMAS`/`LTX2_STAGE2_DISTILLED_SIGMAS`/
      `LTX2_TDP_DISTILLED_SIGMAS`) matched bit-exactly against source. The
      rectified-flow `euler_ancestral_step` (`EulerAncestralDiffusionStep`)
      landed alongside it, structurally unit-tested (no golden numbers exist
      for it - only the schedule was dumped).
- [x] **Reference goldens** - `tools/goldens/ltxv_vae_dump_reference.py` (real
      conv-decoder VAE weights, encoder+decoder stage taps, per-channel-stats
      round trip self-validated, round-trip cosine 0.992-0.996),
      `tools/goldens/ltxv_dit_dump_reference.py` (tiny video-only config, every
      real-config FLAG set correctly, adaLN row order pinned against source,
      fresh-instantiation + batch-independence + RoPE-unit-rotation
      self-validated), `tools/goldens/ltxv_audio_dump_reference.py` (real audio
      VAE + base vocoder weights, mel front end cross-checked two independent
      ways bit-exact, round-trip cosine 0.998; BWE deliberately out of scope),
      `tools/goldens/ltxv_schedule_dump_reference.py` (`LTX2Scheduler` sigma
      vectors + the real `DISTILLED_SIGMA_VALUES` read from source, cross-
      checked against an independent fp64 numpy reimplementation). All four run
      real weights or tiny configs on CPU; goldens land in the gitignored
      `testdata/golden/ltxv/` (regenerate locally, never committed). The
      diffusion (NA) video decoder and audio BWE stage are explicitly deferred
      to their own later milestones, not dumped here.
- [x] **Video VAE** (`crates/ltxv/src/vae3d.rs` over `vae::blocks3d`) - encoder +
      conv decoder, real weights, parity at cosine >= 0.999999 on both the
      Vulkan and CPU-JIT backends (`crates/ltxv/tests/vae_parity.rs`, 7 tests:
      encoder/decoder/round-trip at 9 and 17 frames, plus import coverage).
      Two new kernels (`space_to_depth3d`/`depth_to_space3d`, channel-outer
      3-axis resample - genuinely new semantics, not covered by the existing
      2D `pixel_shuffle`), everything else (`PixelNorm`, the group-mean skip)
      reuses existing `blocks3d` kernels via a synthesized gain/eps or a
      channel-slice-and-average composition. Two real bugs found and fixed
      during the port: the encoder's pre-shuffle frame-0 duplication was
      applied twice instead of once (space_to_depth divisibility panic), and
      the outer pixel `patchify`/`unpatchify` boundary uses a DIFFERENT
      channel sub-order (width-then-height) than the internal
      space-to-depth/depth-to-space resample (height-then-width) - conflating
      the two produced a "structurally right, numerically off" 0.982 cosine
      with every per-block tap bit-exact up to the last op. The NA diffusion
      decoder and general overlapping-tile chunked encode/decode remain out
      of scope, deferred to the DFR milestone.
- [x] **Video-stream DiT** (`LTXModelType::VideoOnly`) - `config.rs`/`rope.rs`/
      `block.rs`/`dit.rs`, tiny (2-layer) config, every real-LTX-2.5 flag set
      correctly. Parity against the tiny golden at cosine >= 0.999999 on every
      tap, most at 1.000000000 (`crates/ltxv/tests/dit_parity.rs`). RoPE's
      exact construction (fractional position -> [-1,1], geometric bands,
      front zero-pad, band-major/axis-minor, per-head sequential chunking of
      one shared table) was verified against the golden's real `rope_cos`/
      `rope_sin` numbers with an independent host reimplementation before any
      device dispatch was written. Rotation reuses the existing `rope2d`
      kernel (table-driven rotate-half), dispatched once per head since each
      head reads a different sub-table - `rope_neox` was the first guess and
      is refuted (it computes its angle analytically from a scalar position,
      no table input, so it cannot express LTX's band/axis construction). The
      per-token adaLN combine reuses `dit::adaln::add_table` at `rows=T`
      (generalized for exactly this caller); the per-token gate reuses the
      existing `gate_row` kernel (`rows_per_cond=1`) unchanged. No new
      kernels. Real-22B-checkpoint import, the audio stream, the
      audio<->video cross-attention, and the text encoder are explicit,
      tracked gaps - only the tiny-config op sequence is proven so far.
- [x] **Pipeline + CLI + serving contract (M4)** - `brain ltxv t2v` produces a
      playable mp4 end to end: real `LTX2Scheduler` schedule + rectified-flow
      ancestral Euler denoise loop + CFG fold (`crates/ltxv/src/pipeline.rs`)
      over the tiny video-only DiT (M3) with FRESH RANDOM WEIGHTS
      (`dit::random_tiny_weights`, seeded - there is no real 22B checkpoint to
      load), decoded through the real causal 3D video VAE (M2). This is an
      explicit SMOKE TEST of the pipeline WIRING, not a generation-quality
      claim - recorded loudly in `crates/ltxv/src/pipeline.rs`'s module doc,
      the same way M1/M2/M3 recorded their own out-of-scope pieces. Full
      serving contract: `crates/ltxv/src/caps.rs` (manifest/`t2v` action,
      weights-free manifest test, cancellation polled once per denoise step),
      `crates/cli/src/ltxv_cli.rs` + `resident_ltxv.rs` (residency adapter -
      deliberately holds NOTHING resident: the DiT is free to rebuild and the
      VAE is read fresh per call, same precedent as `wan`'s own "VAE never
      cached alongside the DiT" choice), registered in `resolve.rs`'s
      `ARCH_HANDLERS` (`brain ltxv t2v` runs the dedicated CLI module, taking
      precedence over generic capability dispatch, the same routing `wan`
      uses - confirmed with `brain caps brain/ltxv`, which lists the manifest
      correctly without a `catalog.rs` entry) and wired into
      `resident.rs::build_executor` (env-gated on `BRAIN_LTXV_VAE`). Verified
      end to end: `brain ltxv t2v --prompt "a cat walking on a beach" --frames
      9 --width 64 --height 64 --steps 4 --seed 42 --device cpu` against the
      real `ltx-2.5-video-vae-conv-bf16.safetensors` produced a real, playable
      64x64 h264 mp4, 9 frames at 8 fps, ~4.9 KB, in ~54s wall clock (VAE
      decode on the CPU-JIT backend dominates at this size). Two simplifications,
      both documented inline where they land: the guidance fold is CFG-only
      (no STG/audio-video joint guidance/rescale - those need machinery this
      port hasn't built), and there is no real text encoder yet
      (`pipeline::context_stub` folds the prompt's hash into a seeded, purely
      structural context tensor). The real 22B DiT import and the Gemma-4 text
      encoder (see the next two bullets) are the tracked gaps this milestone
      exists to isolate, not to close. GGUF importer entry for `ltxv` is NOT
      part of this milestone - deferred until real-weight DiT import lands.
- [x] **`crates/gemma4` text encoder** - own arch row (`Source::Brain`; the
      `transformers.models.gemma4_unified` reference is itself dated 2026, too
      recent to check against a llama.cpp `LLM_ARCH_*` spelling - flagged for
      re-verification rather than guessed at). Config/rope/block/model ported
      against the real, INSTALLED `transformers.models.gemma4_unified`
      package (a genuine authoritative reference, not reverse-engineered from
      config alone). Tiny (6-layer, the real 5:1 sliding:full ratio's minimal
      instance) config parity at cosine 1.000000000 on every tap, including
      BOTH structurally different attention paths separately (a GQA sliding-
      window layer and an MQA `attention_k_eq_v` global layer with dual RoPE
      bases) and the LTX-specific 49-hidden-state `aggregate_embed`
      projection. Zero new kernels - reuses `gqa_scores_win`/`gqa_apply`/
      `rope2d`/`rmsnorm_eps`/`gelu` exactly. Two real bugs found and fixed:
      the real `scaling=1.0` (not the kernel's built-in `1/sqrt(head_dim)`)
      fixed exactly (not approximately) by folding `sqrt(head_dim)` into
      `q_norm`'s uploaded weight, exploiting that RMSNorm-then-RoPE is linear
      in a uniform per-vector scalar; and a RoPE kernel mismatch on the
      `full_attention`/`partial_rotary_factor` layers - `rope2d_partial` was
      the natural guess but pairs channels at the rotated sub-block's own
      half-point, while Gemma-4's real `rotate_half` always pairs at the
      FULL head's half-point regardless of how few frequencies are nonzero;
      caught at cosine 0.77 on that one tap while the unaffected sliding-layer
      tap was already at 1.0, fixed by widening the table to `head_dim/2`
      with zero-padded identity columns past `rope_angles` and reusing plain
      `rope2d` for both layer types. Real-weight (12B/26 GB) parity remains
      an explicit, tracked gap - needs a machine that can hold it.
- [x] **Audio VAE + base vocoder** (`crates/ltxv/src/{audio_vae,vocoder}.rs`) -
      real weights, real parity: encoder/decoder/round-trip and the vocoder
      all at cosine 1.0000000000 against `testdata/golden/ltxv/audio/`. 2D
      causal conv (`[channels, time, mel_bins]`), several conventions genuinely
      differ from the video VAE (asymmetric zero-pad on time vs the video
      VAE's replicate padding, a real strided downsample conv rather than
      space-to-depth, `PixelNorm` at eps 1e-6 not 1e-8, a per-(channel,freq)
      not per-channel bottleneck affine) - see `audio_vae.rs`'s module doc.
      The vocoder is BigVGAN v2/AMP1 (snakebeta + the anti-aliased
      `Activation1d` up/downsample against checkpoint-supplied Kaiser-sinc
      filter buffers). Zero new kernels either file - `pad2d`/`crop2d`/
      `conv_bias_reg`/`l2norm_scale` (audio VAE) and `conv1d`/`convtr1d`/
      `snake_beta`/`axpy` (vocoder, reusing `crates/mimi`'s established
      pattern) cover everything. Bandwidth-extension (48kHz upsampling,
      needs the checkpoint-basis STFT) is explicitly out of scope, same as
      the goldens that back this.
- [x] **Audio DiT stream + bidirectional A<->V cross-attention**
      (`LtxAudioDitConfig`/`LtxAvDitConfig` in `config.rs`, `LtxAvBlock` in
      `block.rs`, `LtxAvDit` in `dit.rs`) - extends the video-only DiT (M3)
      rather than duplicating it; the video-only path and its existing test
      are untouched (confirmed byte-for-byte via `git diff` on that test
      file). Exact op order pinned against `transformer.py`'s
      `BasicAVTransformerBlock.forward` directly: video self-attn+text-CA
      runs fully, then audio self-attn+text-CA runs fully, THEN A2V/V2A off
      a shared pre-AV snapshot of both (so direction order does not bias the
      result), THEN both MLPs. A2V/V2A both run at the AUDIO stream's head
      geometry (asymmetric Q/O projections on the video side); the two
      per-block `[5,dim]` AV tables (video-side, audio-side) use a REVERSED
      row order vs the main 9-row table (`scale,shift` not `shift,scale`,
      rows 0-1 for the A2V direction, rows 2-3 for V2A, row 4 the gate) -
      traced call-by-call against `get_av_ca_ada_values`'s two call sites,
      not assumed from the roadmap's own earlier prose summary. The A2V
      gate is driven by the VIDEO table's row 4 at the CROSS (audio)
      modality's scalar sigma; V2A's gate is the AUDIO table's row 4 at
      video's sigma - the asymmetry already recorded above, now implemented.
      `crate::rope::ltx_rope_tables` generalized from a hardcoded 3-axis
      construction to axis-count-generic, since audio's 1-axis self RoPE
      and the shared cross-modal time-only RoPE are both instances of the
      same math, not a separate construction. Parity: every new tap >=
      0.999999951 cosine (`crates/ltxv/tests/av_dit_parity.rs`) against a
      new golden (`tools/goldens/ltxv_av_dit_dump_reference.py`, a sibling
      of the video-only dumper, not a modification of it). One open
      judgment call, flagged inline rather than silently assumed:
      `audio_ff_bias` has no independently-verified real-checkpoint value
      on this ledger (only video's `ff_bias=false` is confirmed) and is set
      to `false` as the consistent assumption pending verification.
      `av_ca_timestep_scale_multiplier` is a plain config field, not
      hardcoded, for the same reason (metadata reportedly `1000.0` vs. the
      class default `1`, not confirmed empirically - see the "Convention
      questions" section above).
- [ ] **Training** - `grad.rs`/`modelgrad.rs` generic over `trait Fp`,
      `gradcheck::check_ltxv{,_lora}`, LoRA in the ComfyUI key layout, finetune,
      single- and batch-overfit-to-zero gates.
- [ ] **NA diffusion decoder, upscalers, duration head, DFR** - the one genuinely
      new kernel family (3D neighborhood/windowed attention), gated against the
      reference's own eager tiled-masked-SDPA fallback before any fast kernel.
- [ ] **NPU export, INT8, sharding, optimization pass** - only after parity is
      frozen, per porting.md sec10.

## Convention questions settled from source, not experiment

Recorded here as they're pinned by tests, so this section grows as milestones
land. Known traps already identified from reading (not yet test-pinned):

- adaLN table row order is **shift, scale, gate** (self-attn rows 0-2, MLP rows
  3-5, optional text-CA q/kv rows 6-8) - but the **A<->V** table order is
  **scale, shift** (reversed), and its **gate** is driven by the *cross*
  modality's scalar sigma while scale/shift come from *this* modality's
  per-token timestep.
- QK-RMSNorm is over the full `inner_dim` (not per-head), learnable, applied
  **before** RoPE.
- Only one RMSNorm sits between self-attn and text cross-attn (the "fused
  re-norm" - there is no separate `norm2`); the text-CA output is added with
  **no gate** unless `cross_attention_adaln` is set (LTX-2.5: it is).
- The final output norm is **LayerNorm**, not RMSNorm, and uses
  `embedded_timestep` (the raw 256->D MLP output), not the 6D modulation vector.
- RoPE is **split/rotate-half** (GPT-NeoX style), not brain's interleaved
  `rope_interleave_table` layout - bridged by permuting q/k projection (and
  q_norm/k_norm) weight rows at import time, since RMSNorm over `inner_dim` is
  permutation-equivariant. No new kernel needed; must be pinned by a test before
  trusting it.
- Video VAE temporal padding is **replicate frame 0**, not Wan's zero-pad;
  spatial padding is `zeros` on the encoder, `reflect` on the conv decoder.
  `PixelNorm` (channel-RMS, no learnable gain) uses eps **1e-8** in the resnet/
  `conv_norm_out` path vs **1e-6** everywhere `build_normalization_layer` is used
  (the audio VAE) - both appear in the same checkpoint family, do not unify them.
- Video VAE down/upsample is space-to-depth / depth-to-space with a
  parameter-free group-mean skip, **not** strided/transposed conv, and there is
  **no cross-chunk feature cache** (unlike Wan) - tiling uses overlapping tiles
  with trapezoidal 1-D blend masks instead.
- Audio VAE has **no attention anywhere** at the real config
  (`attn_resolutions: []`, `mid_block_add_attention: false`) despite the
  reference code supporting it.
- The vocoder's bandwidth-extension stage needs an STFT, but it is implemented
  as a **conv1d against checkpoint-supplied DFT+Hann bases** - no ISTFT anywhere
  in the whole audio stack, which is what keeps this portable without a GPU FFT
  kernel.
- The Gemma-4 text projection consumes **all 49 hidden states** (embedding +
  48 layers) concatenated per token (`Linear(3840*49 -> 4096/2048)`), not just
  the last layer or a fixed offset like `s3dit`'s Qwen3 usage - confirmed by the
  `188160 = 3840*49` shape in the real header, must be pinned by a test.
- The real 22B/2.5 config differs from the code repo's own module defaults (which
  describe the superseded ~19B checkpoints) in several fields the checkpoint
  header settles empirically: `ff_bias: false` (repo default `true`),
  `use_prompt_adaln_single: false`, `cross_attention_adaln: true`,
  `caption_proj_before_connector: true`, `use_keyframes_abs_pos_embedding: true`,
  `av_ca_timestep_scale_multiplier: 1000.0` in the checkpoint metadata (the repo
  source computes `sigma*1` for the A<->V gate via a *separate* multiplier field
  read from config, not the hardcoded `1` a stale reading of the code might
  suggest - verify empirically before building on either number).
- No int8 "convrot" quantization format exists in the reference repo despite the
  HF filename (`comfy-int8-convrot`) implying one - the real formats are
  blockwise FP8, blockwise FP6, and NVFP4. Layers upstream never quantizes:
  `patchify_proj`, every `*adaln_single*`, `caption_projection`, `proj_out`, and
  their `audio_` twins, `to_gate_logits`, `scale_shift_table` - the same list
  applies when brain adds its own INT8 path in the optimization milestone.

## Recorded gaps (kept current)

- Full 22B DiT and 12B Gemma-4 real-weight parity: not run, needs hardware that
  can hold the full checkpoints.
- NPU device execution: not yet diagnosed for this port; treat as unproven
  until a specific host and blocker (or lack of one) are recorded.
- `vae::blocks3d` has no backward (`blocks/grad.rs` only covers the 2D builder),
  so video-VAE fine-tuning is out of scope until that lands separately.
- Multi-device residents (needed for a sharded 22B DiT across multiple cards)
  do not show up in `braintop` - a pre-existing gap noted in the serving-
  contract exploration, not new to this port.
- Image-to-video, IC-LoRA pipelines, and the `DubIt` speaker-identity pipeline
  are out of scope for this port.

## Scope that collapsed once the reference was read

- The HF filenames imply an "int8-convrot" quantization scheme; grepping the
  full source tarball found zero references to convolution-rotation/Hadamard
  quantization actually wired into any load path - see the INT8 note above.
- `modality_tiling.py` looked like it might be the audio/video mixing mechanism
  from its name; it is spatial/temporal tiling of a video-only token sequence
  for tiled inference, unrelated to the A<->V cross-attention that actually
  couples the streams.
