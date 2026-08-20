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
- [x] **Training (video-only DiT)** - `crates/ltxv/src/{grad,modelgrad,lora,
      finetune}.rs`, `gradcheck::{check_ltxv,check_ltxv_conditioning}`, the
      `wan::grad` template. Scoped to `LtxDitConfig`/`LtxBlock`/`LtxDit`
      only - the audio-extended `LtxAvDitConfig`/`LtxAvBlock`/`LtxAvDit`
      (M6b) has no training support yet, a tracked gap. Two real structural
      differences from Wan's own oracle required new primitives rather than
      reuse: LTX's modulation/gate are per-token `[T,dim]` (Wan's are
      per-block `[dim]`), and RoPE reads per-head sub-tables (Wan's is one
      shared table) - see `grad.rs`'s module doc for the gradient-duality
      argument this implies for the adaLN-table split. Gates: block FD
      3.03e-10 (bound 1e-4), model FD ~7.9e-8 (bound 1e-3), the conditioning
      elementwise-fold gate 2.01e-10, host f32 forward vs the GPU-dispatched
      `LtxDit::forward` at cosine 1.0000000000 (max_abs 1.49e-6) - the
      load-bearing cross-check that the from-scratch host reimplementation
      computes the SAME thing as the already-correct GPU path, not a
      restatement of the same code. LoRA: bit-exact no-op at init, fold-vs-
      apply bit-equal after real training steps, standalone descent
      0.377658->0.170922 over 40 steps. Whole-model overfit (Adam, 400
      steps): loss 1.08134715->0.00000051.
- [x] **Latent upscalers + duration head** (`crates/ltxv/src/{upsampler,
      duration_head}.rs`) - real weights, real parity, cosine 1.0 throughout.
      Both real checkpoints (spatial x2, temporal x2) share the same
      `dims: 3` architecture down to `nn.Conv3d` everywhere - only the
      middle `upsampler` stage and `mid_channels` differ. `ResBlock`'s op
      order is genuinely unusual (activation AFTER the residual add, not
      before - the opposite of the video VAE's own resnet block). Zero new
      kernels: the per-frame `Conv2d`+`PixelShuffleND(2)` collapses exactly
      to `Builder3d::depth_to_space` at `pt=1` (a real mathematical
      equivalence, not an approximation - see `upsampler.rs`'s doc), and
      `GroupNorm` reuses the existing `gn_part`/`gn_stats2`/`gn_apply`
      kernels (previously wired only into the 2D builder) via the same
      `kernels_with` extension idiom `sdxlunet`/`vqgan` already use. The
      duration head's `nn.MultiheadAttention` was decomposed by hand and
      cross-checked against a from-scratch second derivation inside the
      golden dumper itself (cosine 1.0-1e-6) before trusting either as the
      Rust port's target. The checkpoint's own `rational_resampler: true`
      metadata field is dead code upstream (the `elif temporal_upsample`
      branch never reads it) - noted, not implemented.
- [x] **NA diffusion decoder** (`crates/ltxv/src/na_decoder.rs`) - the one
      genuinely new kernel family this whole port needs (3D
      neighborhood/windowed self-attention), gated against the reference's
      own eager tiled-masked-SDPA fallback (`fallback_na/eager.py`) before
      any Rust was written. Real weights
      (`ltx-2.5-video-vae-bf16.safetensors`'s `decoder.*` tree - a DIFFERENT
      file from the conv decoder's `-conv-bf16`), real parity at cosine
      1.000000000 on every tap (`crates/ltxv/tests/na_decoder_parity.rs`):
      stages 1-4 (the deterministic context path: `NA(RMSNorm) -> SwiGLU`
      pre-norm blocks x4 stages + `LinearPixelShuffleUpsample`, at a
      (3,7,7) latent - the smallest volume stage 0's own kernel allows,
      producing a real (17,56,56,256) context), AND stage 5 (all 8
      `CombinedDiffusionNABlock`s, context injection + AdaLN-Zero
      scale/shift modulation, at a DECOUPLED synthetic (13,13,13) context/
      noised-pixel input - see `tools/goldens/ltxv_na_decoder_dump_
      reference.py`'s doc for why the diffusion tap does not chain from the
      real (17,56,56) context: a naive gather-then-dense NA kernel at that
      volume dispatches ~284M score threads, fine for one Python reference
      run, excessive for a routinely-rerun Rust test fixture - correctness
      is proven at both scales via the same code path, just not composed
      end to end in one golden). Two new kernels
      (`na3d_scores`/`na3d_apply.wgsl`, gather-the-window-then-dense-attend,
      per-axis inward-shifting bounds computed inline - `gqa_scores_win`/
      `attn_decode_scores_win` are 1-D sequence-position windows, checked
      and refuted first) plus one more (`pixel_shuffle3d_cl.wgsl`,
      channels-last `LinearPixelShuffleUpsample` - same `(c,p1=T,p2=H,
      p3=W)` sub-order as `vae::blocks3d`'s internal resample kernels,
      confirmed against the einops string, genuinely NOT `crate::patchify`'s
      outer-boundary convention which this decoder ALSO uses unchanged for
      the noised-pixel patchify/unpatchify boundary - both conventions
      appear in this one decoder, verified independently rather than
      assumed from precedent either way); everything else (Linear/RMSNorm/
      SwiGLU/the PixArt timestep embedder/the per-token-table adaLN combine)
      reuses existing kernels/host-math (`matmul`, `rmsnorm_eps`, `silu_mul`,
      `attn_softmax_cross` reused unmodified for the NA softmax since NATTEN
      windows are never masked, `rope_interleave_table` for this module's
      genuinely-interleaved-pair RoPE, `dit::timestep::pixart_timestep_
      embed`, `dit::adaln::add_table`). The real checkpoint's
      `default_num_inference_steps=1` + `model_output_type="x0"` collapse
      the usual multi-step Euler sampling loop to exactly ONE stage-5
      forward at `t=1.0` whose output IS the x0 prediction directly - so
      the "sampling loop" milestone stage and the "single block forward"
      stage are the SAME real-weight-parity-proven tap for this checkpoint,
      not two. The "legacy static gates folded into Linear weights" the
      class docstring mentions is confirmed EMPIRICALLY dead for this
      checkpoint two independent ways: `DiffusionNABlock._modulation` only
      reads 4 of `scale_shift_table`'s 7 rows (scale/shift for attn/mlp;
      the 3 gate rows are computed then discarded) and NEITHER residual
      function (`combined/attn.py::full`, `combined/mlp.py::residual_mlp`)
      multiplies by a gate anywhere; separately, `model_configurator.
      _read_diff_vae_gates` (the production loader's own gate-fold
      pre-read) returns an EMPTY dict against the real header - no
      `gate_msa`/`gate_mlp`/`gate_ctx` sibling tensors exist to fold, so
      brain's own importer needs no analogous fold step. General tiling
      (`diffusion_tiling.py`'s overlapping-tile trapezoidal blend), the
      CHUNKED/BLACKWELL_DSL block variants, and multi-step Euler sampling
      (moot for this checkpoint) remain explicit, tracked gaps - see below.
- [x] **DFR geometry + a smoke-level multi-stage pipeline** (`crates/ltxv/src/
      dfr.rs`, `pipeline::generate_dfr`, `brain ltxv dfr`, plus a `dfr`
      `capability::Action` in `caps.rs`/`resident_ltxv.rs` alongside `t2v` -
      the serving contract's obligation 1 is explicit that a CLI subcommand
      must never be the ONLY entry point, so `dfr` is reachable through
      `brain do brain/ltxv dfr` and `Subscribe` over D-Bus too, not just the
      dedicated CLI module) - the real,
      weight-free bookkeeping (`resolve_canvas`, `pixel_to_latent_index`,
      `keyframe_slots` building the `keyframes_mask` seam
      `dit::LtxDit::forward` has accepted since M3, `tile_ranges`/
      `stitch_tile_latents` for the overlapping-tile temporal-round stitch,
      `target_frame_count`) is unit-tested with most cases pinned against a
      LIVE run of the reference `dfr_layout.py` functions, not hand-derived
      (`crates/ltxv/src/dfr.rs`'s own test doc comments record the exact
      `python3 -c "..."` invocation each number came from). `generate_dfr`
      wires it into a real end-to-end pipeline: half-res base generation with
      appended keyframe-slot tokens, a REAL spatial x2 latent upscale
      (M8a's `upsampler.rs`, applied to both the video and its slots), a
      full-res detailing pass that re-noises the real upscaled result via
      `GaussianNoiser`'s own `torch.lerp(seed, noise, sigma0)` formula (not a
      fresh unrelated noise draw), and 0-2 real temporal x2 upsample rounds
      (M8a's temporal upsampler) with tile-based re-noise + stitch, ending in
      the same real VAE conv-decoder decode `generate` uses. Still the tiny
      random-weight DiT and stub text-context seed M4 established - no real
      22B checkpoint exists to load. Explicit, tracked gaps beyond the
      already-recorded IC-LoRA absence: no per-token/partial-strength
      anchor-keyframe carry-forward across temporal-round seams (real DFR
      pins seam keyframes at `strength=0.95` via per-token timesteps; this
      pipeline broadcasts one scalar sigma to every token, same limit
      `denoise`'s own doc already records for M4 - `TileRange::
      anchor_kf_global` still computes the real anchor position for a future
      milestone to wire in), the NA diffusion decoder (M8b) is not wired in
      as an alternative decode path (its tiling/scale contract differs
      enough to be a separate integration), no real distilled-schedule sigma
      tables (every stage uses the same generic `ltx2_sigmas` M4 already
      proved), and keyframe-slot RoPE positions use this port's existing
      plain-integer-latent-grid convention rather than upstream's
      fps-normalized units (see `dfr.rs`'s module doc). General overlapping-
      tile chunked NA-decoder decode, the CHUNKED/BLACKWELL_DSL block
      variants, and multi-step Euler sampling (all recorded under M8b above)
      are unaffected by this milestone.
- [x] **INT8 storage for the video-only DiT's weights** (`crates/ltxv/src/
      int8.rs`, new) - STORAGE format only, per porting.md sec10 point 6 (a
      precision change is not a speed change until profiling says arithmetic
      is the limiter): no new WGSL kernel, no compute-time DP4A activation
      quantization, and no change to `LtxDit::forward`/`LtxAvDit::forward`'s
      dispatch path. Reuses `model::int8::{quantize_weight,dequantize_weight}`
      (the shared per-channel symmetric int8 primitives zimage's DiT / the
      Qwen encoder-decoder / FLUX.2's DiT already use for the same purpose).
      `is_never_quantized` implements this port's own already-recorded
      "upstream never quantizes" list (`patchify_proj`, every
      `*adaln_single*` table, `caption_projection`, `proj_out`,
      `to_gate_logits`, `scale_shift_table` - every variant, including the
      `audio_`/`av_ca_`-prefixed twins, matched by the same substring since
      the prefix sits in front of it) by substring match against the REAL
      tensor names `dit_tensor_manifest` emits, pinned by a test against
      exact real names (`crates/ltxv/tests/int8_storage.rs`).
      `quantize_tensors`/`dequantize_tensors` split/rejoin a `Tensors` weight
      map; every eligible `[n,k]` projection (`to_q`/`to_k`/`to_v`/`to_out.0`,
      `ff.net.0.proj`/`ff.net.2`) round-trips at a worst per-tensor cosine of
      0.999975 (20 eligible tensors, tiny config, `k%4==0` required for
      `model::int8`'s packing width) - never-quantized/1D tensors pass through
      `full` untouched, checked bit-for-bit. The test that actually matters -
      the SAME tiny `LtxDit` forward run twice, once at plain f32 and once
      with every eligible weight round-tripped through int8 storage first -
      lands at final-output cosine 0.999999995 and every per-block output
      cosine >= 0.999999995 (2-layer, dim-64 tiny config; the modulation/
      patchify/output tables staying at full f32 keeps int8 noise from
      compounding across more than a couple of projections). Gap: not wired
      into any real checkpoint importer's load path yet (`dit::
      load_tiny_weights`/`random_tiny_weights` still produce plain f32), and
      whether this port ever wants a compute-time int8 kernel for the DiT at
      all is an open question this milestone did not settle.
- [x] **Pipeline-parallel sharding for the video-only DiT** (`model::Shardable`
      for `crate::dit::LtxDit`, `crates/ltxv/src/shard.rs` new) - wires
      `LtxDit` onto the generic `model::shard`/`plan_balanced`/`model::
      Pipeline` seam `qwen3`/`gpt2`/`qwen35moe` already use for their own
      transformer stacks. `LtxAvDit` (audio+video) is explicitly OUT of scope
      - its bidirectional cross-attention couples the two streams every
      block, a materially bigger seam than one block stack. Every stage loads
      its own copy of the (small) `adaln_single.*` weights and independently
      recomputes the per-token adaLN table and RoPE tables from the shared
      batch, rather than shipping those over the wire - only the residual `x`
      crosses a stage boundary, matching `model::shard`'s own "residual only"
      contract; `patchify_proj.*`/`keyframes_abs_pos_embedding` are
      embed-stage-only and `scale_shift_table`/`proj_out.*` are head-stage-
      only (`crate::dit::shard_owns_weight`). `crate::dit::forward_blocks` (a
      new block-range helper `LtxDit::forward`'s own `[0,num_layers)` loop was
      refactored to call, no behavior change - confirmed by every pre-existing
      `crates/ltxv` parity/gradcheck/overfit test still passing unchanged)
      is the single source of truth both the ordinary and the sharded forward
      path share. `Model`'s diffusion-shaped batch (`crate::dit::DitBatch`)
      does not fit `model::Batch`'s LM/seq2seq/image-splice variants, so
      `Model::set_batch` is a documented no-op and the real seam is
      `LtxDit::load_shard_batch` - the same shortcut `s3dit::train::
      ZTrainModel` already takes for its own diffusion batch. Tested
      (`crates/ltxv/tests/shard_parity.rs` + `crates/ltxv/src/shard.rs`'s own
      unit tests): `shard_cost` fed to `plan_balanced` produces a well-formed
      partition (contiguous, complete, exactly one embed/one head stage, no
      empty stage) for both the tiny test config AND the real 22B config's
      shape (48 layers, `inner_dim` 4096) at 2/4/8 stages - a plan can be
      COMPUTED for the real model even though it cannot be built or run on
      this port's hardware (see "Validation policy" above); `new_shard`
      genuinely loads only its block range's weight subset (a 1-of-2-layer
      shard's total float count is strictly less than the whole model's, and
      a whole shard's parameter set equals the full manifest exactly); the
      single-shard degenerate case (one stage owns every block) matches
      non-sharded `LtxDit::forward` at cosine=1.000000000, max_abs=0.000e0;
      a genuine two-stage split (real block-range partition, boundary handed
      off through `write_in_res`/`read_out_res`, run sequentially on a
      single GPU) composes to the same cosine=1.000000000, max_abs=0.000e0
      against the same reference. Explicit, tracked gaps: no real multi-device
      execution was or could be performed at this milestone (only one GPU
      was available at the time - the two-stage test proves the
      boundary-handoff logic is correct, not that two real cards agree);
      `LtxDit` has no backward/training pass at
      all (`Shardable::run_backward_stage`/`read_in_dres`/`write_out_dres`
      and `Model::backward`/`read_grad`/`write_weight` all `unimplemented!()`
      loudly, matching this repo's own honest-loud-stub precedent rather than
      a silent wrong-zero-gradient); `model::Pipeline<LtxDit>` type-checks but
      is not a usable end-to-end entry point (`Pipeline::forward` drives a
      stage through `Model::set_batch`, a no-op here - the tests above
      hand-drive `Shardable`'s methods directly instead, the same limitation
      `s3dit::train::ZTrainModel` already documents for itself);
      `residency::multi::MultiDeviceResidentModel` (inference-time placement)
      is untouched, a separate later lift.
- [x] **Performance profiling pass** (`crates/ltxv/src/bin/ltxv_bench.rs`,
      new) - per porting.md sec10's mandatory profile-before-touching-code
      discipline, on the shared `gpu_core::profile`/`gpu_core::roof` facility
      (`crates/vqgan/src/bin/vqgan_bench.rs`'s precedent, graded against this
      DEVICE's own measured roofline - an Intel Arc integrated GPU, not a
      hardcoded discrete-card peak). Two small additive enablers:
      `LtxBlock::build_steps` (chains N blocks' already-recorded steps into
      ONE combined submit over a device-resident buffer, without touching
      `LtxBlock::forward`) and `LtxVaeDecoder::steps()`/`::gpu()` (read
      accessors onto an already-built decode graph), plus a `gate_row` cost
      formula in `crates/gpu-core/src/cost.rs` so the DiT pass's whole-pass
      rate is computable. DiT table: real width (`inner_dim=4096`, 32 heads),
      1 of 48 real layers at 64 tokens/32 context (108 dispatches) - forced
      down from the planned 8-layer/512-token shape by genuine GPU
      contention (three parallel agents plus a coordinator all profiling/
      testing on the same single GPU at once; confirmed via `/proc/<pid>/wchan
      == drm_syncobj_array_wait_timeout`, a real fence wait, not a bug) that
      also made absolute timings untrustworthy this pass (the identical
      108-dispatch graph measured 5.01 s vs 14.02 s back to back) and revealed
      the device-timestamp-query kernel attribution on that machine is unreliable
      (one run attributed 1.17M ms to a 2-call kernel against a 14 s whole-
      pass total) - both recorded as killed hypotheses, not findings about the
      kernels. `matmul` (the 8 attention projections + 2 FFN matmuls per
      block) is essentially the whole pass by dispatch-count share at this
      token count, the expected shape when `O(tokens x dim^2)` GEMMs dwarf
      `O(tokens^2)` self-attention at small token counts - no kernel-kind
      stood out as a share-based target given the contention caveat, so
      nothing was changed in the DiT path. VAE table: real weights
      (`ltx-2.5-video-vae-conv-bf16.safetensors`), 9 frames at 128x128 (457
      dispatches, this table trustworthy in absolute terms too - it ran
      isolated after the contended DiT attempts were killed) - a "nothing to
      fix, already fixed upstream" finding: `wan.md`'s own Perf section
      documents fixing exactly this shape of defect for Wan's VAE (`conv3d`
      dominating at ~96-98%, far under roofline) via an `im2col3d_at`+
      `matmul_reg3`+`nlc_bias_nchw` GEMM lowering in the SHARED `vae::
      blocks3d` crate; `ltxv`'s video VAE decoder is built on that same
      shared crate and inherited the fix for free (the lowered path is 66.7%
      of this pass combined, direct `conv3d` only 10.4% - the 6 surviving
      direct-conv dispatches are the low-`Cout` convs the `GEMM_CONV3D_MIN_
      COUT` guard correctly keeps direct, same pattern `wan.md` records
      post-fix). Recorded follow-ups, not attempted this pass (out of a
      "profiling infrastructure only" scope): six kernel kinds in the VAE
      table (`im2col3d_at`, `conv3d`, `l2norm_scale`, `nlc_bias_nchw`,
      `depth_to_space3d`, `add_chan_bcast`) have no `gpu_core::cost` formula
      yet, so the VAE pass's whole-pass GFLOP/s rate reads unavailable; a
      re-run of the DiT bench at the originally planned 8-layer/512-token
      shape on an uncontended device; investigating the device-timestamp-query
      path before trusting DEVICE-timed (not machine-clock-bracketed)
      per-kernel numbers on this backend.
- [x] **NPU export - deliberate scope exclusion, not a silent omission.**
      `crates/npu/src/lib.rs`'s `NpuModel` trait has exactly three real
      implementors (`DepthNpuModel`, `Chronos2NpuModel`, `FincastNpuModel`),
      all sharing one shape: the host does patching/tokenizing/embedding in
      plain Rust, and only a SMALL, FIXED-SHAPE core is exported/compiled onto
      the NPU. A 48-layer/4096-dim video DiT (coupled every block to a second
      2048-dim audio stream in the full `LtxAvDit`) has no equivalent small
      fixed core to peel off - the block stack IS the model, at a width/depth
      that dwarfs every existing NPU target in this repo by two-plus orders of
      magnitude (the forecast/depth cores are single-digit-to-low-double-digit
      MB graphs; this DiT's real checkpoint is 42 GB bf16). NPUs are
      structurally an edge-inference target for models far smaller than a 22B
      video DiT in this repo's own portfolio; this model's real deployment
      targets are GPU/CPU. NPU firmware is also already a recorded, diagnosed-
      elsewhere blocker (`.agents/roadmap/dtype.md`:
      `/dev/accel/accel0` is present and `Inventory::probe().npus == 1`, but
      the firmware is not functional, so `NpuConfig{allow_fallback:true}`
      silently retargets OpenVINO's GPU plugin instead of erroring) - not
      re-diagnosed here, cited as the already-settled reason a working NPU
      path could not even be smoke-tested regardless of the
      scale question. If NPU export is ever revisited for `ltxv`, the open
      question is architectural first: is there ANY small fixed-shape core
      worth exporting (the embeddings connector, or the duration head - both
      already small, real-weight components) versus the 22B block stack,
      which is not a candidate under this trait's existing pattern at all.
- [x] **Real-weight parity ladder for the 22B DiT, reduced depth** - the first
      time REAL 22B weights (not tiny random ones) were touched at all, on the
      real `ltx-2.5-22b-distilled-transformer-Q8_0.gguf` (23.6 GB, 4349
      tensors, verified header-first). Two separate gates, per this port's own
      "quantization vs port correctness must never alias" design (porting.md
      §5):
      1. **Quantization exactness** (`crates/checkpoint/src/gguf.rs`'s new
         `MmapGguf::raw_tensor_bytes` accessor + `crates/ltxv/tests/
         gguf_quant_real.rs`, real-file-gated on `BRAIN_LTXV_DIT`/the model
         store) - `checkpoint::gguf::deq_q8_0` vs. a from-spec Q8_0 dequant
         reimplemented independently in the test itself (34-byte blocks,
         `int8 * f16 scale`, no shared code with the crate's own decoder),
         on three real Q8_0 2D projections (block 0's `attn1.to_q`/`ff.net.2`
         and block **47**'s `attn1.to_v` - the first, an early, and the LAST
         block, so an off-by-block-index bug in name construction can't hide
         behind block 0 alone) plus `patchify_proj.weight` (ships F32 in this
         checkpoint, sanity-checked at the other dtype). Result: **exact**,
         `max_abs == 0.0` on every tensor, both dtypes.
      2. **Port correctness, reduced depth** (new golden dumper `tools/
         goldens/ltxv_real_dit_dump_reference.py`, sibling of the tiny
         dumpers; extends `crates/ltxv/tests/dit_parity.rs` with
         `real_weight::ltxv_real_dit_tiny_layers_matches_reference`) - the
         official `LTXModel(model_type=VideoOnly, num_layers=2)` at REAL
         width (`inner_dim=4096`, 32 heads x 128, gated attention ON), its
         `num_layers` constructor arg alone giving a truncated 2-of-48-layer
         model for free (no reference-internals surgery needed), loaded with
         the real GGUF's own first-two-blocks weight subset via the dumper's
         own independent Q8_0/F32 decoder (not brain's Rust code at all - see
         the dumper's module doc for why that's what keeps this gate distinct
         from #1 above), replayed against `LtxDit::forward` fed the IDENTICAL
         real weight subset read straight off the same GGUF via `checkpoint::
         gguf::MmapGguf` + `dit::dit_tensor_manifest` (no fixture holds the
         weights themselves - only the ~2.2 MB of inputs/taps does. Result,
         every tap: `adaln_table`/`embedded_timestep`/`block.0.out`/
         `block.1.out`/`out` at cosine 1.000000000, `b0_attn1_out`/
         `b0_attn2_out`/`b0_ff_out` at cosine >= 0.999999998, `rope_cos`/
         `rope_sin` at cosine >= 0.999999810 (the same `MIN_COS=0.999999` bar
         `dit_parity.rs`'s existing tiny-config tests already use, not a
         loosened one). Shape: grid `(2,2,2)` -> 8 tokens, context_len 6 - the
         SAME small shape `ltxv_dit_dump_reference.py`'s `TINY_CONFIG`/`GRID`
         already establish, per this task's own small-first constraint;
         real-weight-subset load ~25 s, forward ~19 s on the CPU-JIT
         path. Dumper run itself (real GGUF -> golden) took ~3.5 min, mostly
         a pure-Python Q8_0 decode loop over ~2.7 GB of real Q8_0 blocks - the
         one step in this milestone that ran past the "a few minutes" target,
         noted rather than hidden; every OTHER step (both Rust tests, the
         quant-exactness dequant of the whole real tensor set) finished in
         well under a minute.

      One real bug found and fixed by this work, not a numbers-only pass:
      `crate::dit::dit_tensor_manifest` (video-only) never listed
      `to_gate_logits.{weight,bias}` at all (only `av_dit_tensor_manifest`'s
      `push_attn` did), so `random_tiny_weights`/`shard_weights`/`import_dit`
      would have silently built or accepted a WRONG weight set for any gated
      config - invisible until this milestone tried to load a real
      `apply_gated_attention: true` config's weights through that exact path
      and `LtxBlock::on`'s `tget("...to_gate_logits...")` would have panicked
      "missing weight". Fixed by routing `dit_tensor_manifest`'s per-attn loop
      through the SAME `push_attn` helper `av_dit_tensor_manifest` already
      uses (both `to_gate_logits` names are already on `crate::int8::
      is_never_quantized`'s substring list, so this changes no int8-
      eligibility classification); `crates/ltxv/src/modelgrad.rs`'s
      `params_and_grads_cover_the_whole_manifest_in_the_same_order` test
      updated to explicitly exclude `to_gate_logits` from the equality check
      it makes, with a comment recording WHY (gated-attention backward is not
      implemented by that training path - a pre-existing, now-explicit gap,
      not a new one this fix introduced).

      Zero new kernels, zero changes to `LtxBlock`/`LtxDit::forward`'s op
      sequence - this milestone is entirely new tests plus the one manifest
      fix above. Explicit, tracked gaps (see "Recorded gaps" below for the
      full breakdown): only 2 of the real checkpoint's 48 layers, only the
      VIDEO stream (no audio stream, no A<->V cross-attention, `LtxAvDit`
      untouched), `use_embeddings_connector: false` for this gate specifically
      (both embeddings connectors' own real-weight parity is unattempted -
      8 more real-width layers plus 128 learnable registers apiece, a
      materially bigger check than the block-stack gate above), Q8_0 only
      (Q4_K_M untested at this milestone), and no
      int8/int4 COMPUTE parity (Phase 5 territory - there is no compute-time
      int8 kernel yet for the DiT to compare against, so the plan's
      "int8-vs-fp32 at the model level" comparison is deferred wholesale,
      not attempted at a reduced bar).
- [x] **Audio golden regenerated via a librosa-backed `torchaudio` shim**
      (`tools/goldens/torchaudio_shim/`) - the audio VAE/vocoder dumper
      (`tools/goldens/ltxv_audio_dump_reference.py`) needs `torchaudio` only
      for one self-check (`torchaudio.transforms.MelSpectrogram` at
      `mel_scale="slaney", norm="slaney"`, cross-checked against the
      dumper's own primary mel computation), and a real environment can end
      up with a CUDA-linked `torchaudio` wheel that will not load against a
      CPU-only torch build. Rather than leave `testdata/golden/ltxv/audio/
      audio.safetensors` unregeneratable, a small stand-in package
      (`tools/goldens/torchaudio_shim/torchaudio/`) reimplements exactly
      that one call using `librosa.filters.mel(..., htk=False,
      norm="slaney")` + `librosa.stft` - the documented interoperability
      target `torchaudio`'s own slaney mode is designed to reproduce.
      Run with `PYTHONPATH=tools/goldens/torchaudio_shim` prepended when a
      real `torchaudio` will not import; the dumper itself is unmodified.
      Verified, not assumed: the dumper's own self-check reported
      `max_abs == 0.0` between the shim and its primary computation (i.e.
      bit-exact agreement with the algorithm the shim stands in for), and
      all 6 `crates/ltxv/tests/audio_parity.rs` tests (previously silently
      skipping - the golden was simply absent, a lesson-#1-shaped gap of its
      own) now pass for real under `BRAIN_REQUIRE_FIXTURES=1`.
- [x] **Int8/int4 compute path + AV pipeline-parallel sharding, run for real
      on two physical GPUs** - a CAPACITY milestone (porting.md sec10: a
      precision change is not a speed claim), making the 22B DiT small
      enough to fit available VRAM and proving the pipeline-parallel seam on
      real hardware for the first time.

      Compute-time int8/int4 (`crates/ltxv/src/block.rs::KERNELS` gains
      `max_abs_row`/`quant_pack`/`matmul_i8_dyn` for i8, `matmul_q4_dyn` for
      i4, dispatched through the shared `model::ops` DP4A recipe every other
      quantizing model already uses; modelled on `wan::block`'s `QTier`,
      adapted to this crate's 10-linear block shape). `crate::int8::
      is_never_quantized` stays the single authoritative exclusion list.
      Verified at tiny scale (`crates/ltxv/tests/int8_compute.rs::
      dit_forward_stays_close_with_int8_compute_dispatch`) AND against one
      real block (block 0) of the actual Q8_0 checkpoint
      (`real_q8_0_block0_int8_compute_matches_fp32`).

      Device-bytes measurement at REAL 22B dims (`crates/ltxv/tests/
      device_bytes_real.rs`, lesson #34 - a memory saving nothing measures
      is not a claim): computed from the real config's own tensor-size
      breakdown, not assumed. Measured ratio lands in a real (3.5, 4.0)x
      band - never the flat theoretical 4x an all-eligible model would get,
      since `is_never_quantized`'s exclusions keep a real fraction of
      parameters at fp32.

      AV sharding extended from `LtxDit` to `LtxAvDit`
      (`crates/ltxv/src/shard.rs`) - the A<->V coupling means BOTH streams'
      residuals cross a stage boundary together, not one. Same proof ladder
      `LtxDit`'s own `Shardable` impl already established, single-process
      first (`crates/ltxv/tests/av_shard_parity.rs`): single-shard
      degenerate case bit-exact, a genuine two-stage block-range split
      boundary-handed-off correctly.

      Then run for real (`crates/ltxv/tests/av_shard_2gpu_real.rs`): stage 0
      pinned to `gpu0`, stage 1 to `gpu1` via `gpu_core::devices::with_gpu`
      (the same thread-local placement mechanism `model::shard::Pipeline`
      already uses for construction-time placement, applied here to
      forward-time placement since `LtxAvDit` opens its device fresh per
      `run_stage_forward` call), the boundary residual crossing as a real
      host `Vec<f32>` round trip - what actually crosses a PCIe boundary
      between two distinct devices, unlike a same-device buffer that never
      leaves VRAM. **Passed for real on two physical GPUs** - closes the
      "two real cards agreeing is unverified" gap for the AV path.

      Explicit, tracked gaps: the 2-GPU run uses the small synthetic
      `tiny_gated` config (2 layers), proving the MECHANISM, not a
      real-22B-weight claim - a real-checkpoint-weight version needs a
      GGUF-streaming int8 shard loader this pass did not build; no
      compute-time int8 for the embeddings connectors or the NA decoder;
      deep kernel performance tuning is a later phase, not attempted here.
- [x] **Real weights wired into `brain ltxv t2v` for the first time** -
      `pipeline::generate` had used `random_tiny_weights`/`context_stub`
      exclusively through every earlier milestone (deliberately - each one
      was scoped to leave this file untouched). `Paths` gains optional
      `dit`/`text_encoder` roles (`--dit`/`$BRAIN_LTXV_DIT`, `--text-encoder`/
      `$BRAIN_LTXV_TEXT_ENCODER`); `dit_config_from_name` accepts
      `"ltx25_22b"` alongside `"tiny"`. Real DiT weights load via
      `crate::gguf_src::LtxvGgufSource` + a new streamed loader
      (`load_head_tensors_from_source`/`forward_q_streamed`) that keeps only
      the small non-block tensors resident and streams each of the 48
      `transformer_blocks.*` fresh from the GGUF, quantizes it to int8 on the
      way to the device, and drops it - the whole 22B model is never
      materialized as host fp32 (88 GB), which the existing `forward_q`
      would have required. Real text conditioning
      (`real_text_context`, `crates/gemma4`'s tokenizer + `Gemma4Model` +
      `AggregateEmbed`) is wired as an independent switch.

      One real, general bug found and fixed along the way, not specific to
      this milestone's own weights: `crate::block::EmbeddingsConnector`
      requires its input sequence length to be an EXACT multiple of its own
      register count (register substitution tiles registers round-robin over
      invalid positions), but neither a real tokenized prompt's length nor
      `GenOpts::context_len`'s fixed stub size is naturally shaped that way -
      `LtxDitConfig::ltx25_22b` has `connector_num_learnable_registers: 128`,
      so any non-128-multiple context length panics the moment a
      connector-enabled real config's forward actually runs (never exercised
      before this milestone - every prior real-weight test used a
      `use_embeddings_connector: false` tiny config or fed the connector
      directly at an already-correct length). Fixed with `padded_context_len`,
      shared by both the stub-context and real-text-context branches.

      **This took six real attempts, each surfacing a genuinely distinct
      problem - recorded honestly, not smoothed over:**
      1. A stale `target/release/brain` binary was run against new source
         changes without rebuilding first - the "success" it appeared to
         show was the OLD code path, not the new wiring. Caught by checking
         the log's own printed description against what the new code should
         have printed.
      2. A real compile error: `denoise()` gained a `context_valid: &[f32]`
         parameter but two call sites inside `generate_dfr` were not updated
         - `cargo build -p brain-ltxv` failed outright (6 errors). An earlier
         status report had claimed the build passed; it had not been
         actually re-run after the last edit.
      3. The `EmbeddingsConnector` register-multiple requirement above,
         first seen as a hard panic (`seq_len 8/11 must be a multiple of
         num_registers 128`) before its cause was understood.
      4. The real Gemma-4-12B encoder's own forward pass measured too slow
         on the CPU path to fit inside a smoke-test time budget - rather
         than let it run unbounded, real text conditioning was deprioritized
         for THIS milestone's actual generation run in favor of the stub
         context (real DiT + real VAE + stub context, not full real
         end-to-end) - a tracked gap, not silently dropped; the wiring
         itself (`real_text_context`) is real and built, just not exercised
         in the run that finally produced output.
      5. The first-ever real 22B int8 denoise step measured **398.19 s** -
         almost certainly dominated by one-time GPU shader-compile/pipeline
         warm-up rather than steady-state cost: the next clean runs measured
         **195.6-203.9 s/step** consistently, roughly half. Both numbers are
         real measurements from real hardware, kept rather than only citing
         the faster one - this port is entirely unoptimized (a later phase's
         job), so neither number should be read as a performance claim.
      6. `diffusion::scheduler::ltx2_sigmas`'s "stretch" step produces `NaN`
         at `steps == 1` specifically: with only one non-terminal sigma
         entry (always exactly `1.0`, the ramp's own first value),
         `scale = (1 - 1.0) / (1 - terminal) = 0`, and the subsequent
         division is `0/0`. **Not fixed in `ltx2_sigmas` itself** - worked
         around by using `--steps 2` for the final successful run instead of
         `--steps 1`. A real, narrow, tracked gap: `ltx2_sigmas` should
         either guard `scale == 0` explicitly or skip the stretch when fewer
         than 2 non-zero entries exist, and a regression test for the
         `steps == 1` case is still owed.

      The successful run: `brain ltxv t2v --prompt "a cat walking on a
      beach" --frames 9 --width 64 --height 64 --steps 2 --seed 42
      --dit-config ltx25_22b --device gpu` against the real
      `ltx-2.5-22b-distilled-transformer-Q8_0.gguf` + the real conv video
      VAE + the stub text context, wall clock **409.9 s total** (build
      10.03 s, denoise 395.2 s = 197.6 s/forward x 2, VAE decode 4.6 s),
      wrote a real, valid 64x64/9-frame/8fps mp4. Explicitly a WIRING smoke
      test, the same framing this module's doc has carried since the
      original tiny-weight M4 milestone - 2 denoise steps and a stub prompt
      are not a generation-quality claim, and real-weight DiT parity is
      still only proven at reduced depth (the earlier "real-weight parity
      ladder" milestone, 2 of 48 layers).
- [x] **Training for the audio+video DiT (Phase 7)** - `crates/ltxv/src/
      {av_grad,av_modelgrad,av_lora,av_finetune}.rs`,
      `gradcheck::{check_ltxv_av,check_ltxv_av_conditioning}`,
      `crates/ltxv/tests/{av_block_grad,av_overfit,av_lora_train,
      av_lora_reload,av_concept_learning}.rs`. Closes the gap the M7
      training milestone's own doc named ("the audio-extended
      `LtxAvDitConfig`/`LtxAvBlock`/`LtxAvDit` has no training support
      yet").

      Reused, unchanged, first: `crate::grad`'s video-only
      `block_forward`/`block_backward` were split into the two composable
      phases `crate::block`'s own device path already draws as separate
      functions (`self_attn_and_text_ca_fwd`/`_bwd`, `mlp_fwd`/`_bwd`) -
      a behaviour-preserving refactor, verified byte-identical by
      re-running `block_grad.rs`/`overfit.rs`/`lora_train.rs` before any
      new AV code existed. `crate::av_grad` calls both phases twice (video
      stream, audio stream) and adds only what's genuinely new: the
      audio<->video cross-attention. Two real structural differences from
      every attention module already in this crate: `CrossAttnW` is a
      non-square `AttnW` twin (A2V/V2A project between DIFFERENT-width
      streams, always at the audio stream's head geometry -
      `q_dim != kv_dim != inner_dim` in general), and the residual gate is
      a THIRD point on `crate::grad`'s own per-forward/per-token gate
      spectrum (`gate_bcast`/`gate_bcast_bwd`, new) - ONE row shared by
      every token, driven by the OTHER modality's scalar sigma, not a
      per-token gate. `crate::av_modelgrad` adds the six shared-shape
      `AdaLayerNormSingle` timestep MLPs the AV model carries (one
      `TsMlpW`/`ts_mlp_forward`/`ts_mlp_bwd` implementation, reused six
      times: each stream's own 9-row table, the two 4-row AV scale/shift
      tables, the two 1-row AV gate tables) and the two per-stream output
      stages. Audio's FFN carries bias where video's does not
      (`dit::push_ff`'s doc - a real per-stream asymmetry, not a
      simplification) - handled with a small biased `mlp_b_fwd`/`mlp_b_bwd`
      twin rather than genericising `Lin`/`LinNB` together.

      Gates, all at `LtxAvDitConfig::tiny` (ungated - `to_gate_logits` and
      both embeddings connectors stay out of scope, the same line M3/M7
      already drew): block FD worst error 2.998e-10 over 82 tensors (bound
      1e-4, `av_block_grad.rs`, video+audio weight tensors plus every
      external input including the four model-shared AV conditioning
      tables); model FD (`check_ltxv_av`) and the elementwise conditioning
      fold gate (`check_ltxv_av_conditioning`, covering both streams' main
      adaLN tables plus all four AV cross-modal tables, read TWICE per
      block each) both pass at (atol=1e-6, rtol=1e-4), the same bound
      `check_ltxv`/`check_ltxv_conditioning` already hold the video-only
      path to; whole-AV-model overfit (Adam, 400 steps): loss
      0.78321021->0.00000013. This whole training path is pure host math
      with no GPU dispatch at all (same as the video-only path) - lesson
      #5's "run every gradcheck on both backends" has no separate CPU/GPU
      surface to exercise here.

      LoRA targets 28 leaves per block: both streams' `attn1`/`attn2`
      q/k/v/o + FFN (10 leaves each, matching the video-only path's own
      10), plus all four A<->V cross-attention projections
      (`audio_to_video_attn`/`video_to_audio_attn` q/k/v/o, 8 more) -
      included deliberately, not merely for symmetry (`av_lora.rs`'s own
      doc): a concept that manifests as a video/audio correlation can only
      be learned through those four modules. Bit-exact no-op at init,
      fold-vs-apply bit-equal after real training steps, standalone
      descent 0.316691->0.151351 over 40 steps (rank 4, lr 3e-3). A
      fresh-OS-process save/reload round trip (`av_lora_reload.rs`, a
      self-respawn-the-compiled-test-binary trick, lesson #23) closes the
      loop one step further than the `qwen3::tests::lora_roundtrip`
      precedent, which closes it with a fresh struct in the SAME process:
      live-vs-reloaded 0.000e0 (bit-exact), live-vs-base 3.035e-2 (a real,
      measured margin the reload must reproduce).

      A synthetic procedural dataset with exact ground truth
      (`data::gen_clips`, already used by `wan`'s own LoRA gates - reused
      as-is, not reinvented, per this port's own reuse-first convention): a
      magenta triangle orbiting a white dot (concept) vs. a cyan square
      bouncing between two walls (distractor), rendered at 24x24, encoded
      into the DiT's own token-latent shape by a fixed, never-trained
      random projection (`crate::av_finetune`'s own doc explains why - no
      real VAE-latent distribution exists for this tiny random-init model
      to be calibrated against). A LoRA trained ONLY on concept clips (8
      clips, 80 steps, rank 4) moves GENERATED OUTPUT (a one-step
      flow-matching denoise from pure noise, not loss - lesson #3: finite
      differences prove the derivative, never the objective) toward the
      concept centroid and away from a held-out distractor centroid, on a
      HELD-OUT caption string never seen during training: mean score
      base=+0.00243 -> adapted=+0.00408 (delta +0.00165,
      `av_concept_learning.rs`).

      A stronger training budget (200 steps, rank 8, lr 8e-3) was tried
      FIRST and FLIPS the sign (delta -0.082, adapted output moves AWAY
      from the concept) - the same over-training collapse
      `wan::tests::finetune_ab`'s own `#[ignore]`d G2 gate documents at
      real scale ("collapses ... does not recover by step 250"), sharper
      here because `caption_context` gives each distinct caption string an
      INDEPENDENT random embedding (no real text encoder exists in this
      crate's training scope yet, so there is no semantic bridge between
      the training caption and the held-out one) - enough steps memorises
      "this exact context vector -> this exact latent" rather than a
      direction that generalises to an unrelated context draw. Recorded as
      a refuted hypothesis with its own number, not silently dropped.

      Scoped down to numeric-only, deliberately: no video files were
      produced for this milestone. This tiny AV DiT's token-latent space
      has no relationship to the real VAE's calibrated distribution (fit
      to the real 22B checkpoint, which has no training/backward path in
      this crate at all - `crate::int8`'s compute path is inference-only),
      so decoding this test's synthetic latents back to pixels would only
      demonstrate that the projection is invertible, not that the LoRA
      learned anything about video content.

      Explicit, tracked gaps: gated attention's backward (`to_gate_logits`)
      and both embeddings connectors' training are not implemented, the
      same scope line M3/M7 already drew for the video-only path;
      `AvModelWeights::from_tensors` exists (checkpoint-name-keyed import
      into the training weight layout, mirroring `ModelWeights::
      from_tensors`) but no real 22B GGUF-to-training-weights bridge was
      built - this milestone trains only at `LtxAvDitConfig::tiny`;
      `model::Shardable`'s own backward seam (`run_backward_stage`/
      `read_in_dres`/`write_out_dres`) remains unimplemented for
      `LtxAvDit`, same as `LtxDit`'s own recorded gap above - this
      host-math training path is separate and does not build on it.

      **Closing the validation plan's own "review: compare base vs
      finetuned clips" gate**: re-ran `av_concept_learning.rs` in a later
      session and confirmed the same numbers hold (base=+0.00243,
      adapted=+0.00408, delta +0.00165), then asked directly whether to (a)
      accept this numeric gate as the review evidence, (b) build a real
      training/backward path for the 22B int8 checkpoint so a real video
      comparison becomes possible, or (c) decode the tiny model's adapted
      output anyway with a clear "not representative" label. Decision:
      accept the numeric gate - it is the real evidence this milestone
      produces, and a real-weight LoRA video comparison is new scope beyond
      what this port's own plan asked for (synthetic clips with exact
      ground truth, not real-weight training), not a shortfall of this
      milestone as built.
- [x] **Performance, driven by measurement (Phase 8)** - supersedes the
      contention-ruined M9 profiling entry above with a clean, uncontended
      re-run plus three verified kernel-selector fixes and the first-ever
      measured attribution of the real ~200s/step number Phase 6 reported.

      **Cost-formula prerequisite** (`crates/gpu-core/src/cost.rs`) - 25 new
      match arms (`na3d_scores`, `na3d_apply`, `conv3d`/`conv3d_dx`/
      `conv3d_dw`, `im2col3d_at`, `space_to_depth3d`, `depth_to_space3d`,
      `pixel_shuffle3d_cl`, `l2norm_scale`, `nlc_bias_nchw`,
      `add_chan_bcast`, the `rope`/`rope_neox`/`rope_train`/
      `rope_train_bwd`/`rope_partial`/`rope_partial_bwd`/`rope_sub`/
      `rope_interleave_table` family, `gelu_erf`, `geglu_shift`,
      `snake_beta`, plus `attn_scores_cross_kt`/`kv_k_headt` once the
      optimization pass below adopted them), each derived from its own
      WGSL `struct Params` and loop structure, not guessed. Also fixed a
      real bug in `covers()`'s own probe: it built an all-ones params slice
      only 16 words long, so `conv3d`/`conv3d_dx`/`conv3d_dw`/
      `im2col3d_at` (19-field `Params` structs) silently reported
      UNCOVERED even with a correct formula in place - widened to 32
      words. `FLOOR` raised 150 -> 225 of 416 kernels
      (`cost::tests::cost_coverage_over_the_kernel_tree_never_regresses`,
      `cost::tests::ltxv_kernel_costs` for the hand-computed formulas).

      **Re-run on an uncontended device** (`ltxv_bench dit`, GPU idle before
      every run per `nvidia-smi`/`ps aux`) - roofline measured twice back
      to back, exact agreement both times (10542 GFLOP/s, 287.4 GB/s DRAM,
      lesson #27); clock/temperature tracked throughout (idle 33-41C at
      544 MHz, boosted to 1303-1531 MHz / 40-56 W under sustained load, no
      throttle observed - nowhere near the 90C/999MHz throttle point this
      repo's own roadmap history records elsewhere). Also fixed the bench's
      own drift bug first (`crates/ltxv/src/bin/ltxv_bench.rs`):
      `real_video_dit_config` hardcoded `apply_gated_attention: false`
      (stale from before gated attention was implemented), silently
      profiling a REDUCED op sequence instead of the real one - deleted in
      favour of `LtxDitConfig::ltx25_22b()` with only `num_layers`
      overridden, and the module doc's `tokens=512` claim (vs the code's
      actual `1024` default) corrected to match.

      **Fix 1 - the fp32 GEMM selector** (`crates/ltxv/src/block.rs::
      linear`). `linear()` hardcoded the naive one-thread-per-output
      `K_MATMUL` unconditionally, for every projection in the fp32
      reference tier (`LtxBlock`/`LtxAvBlock` - `LtxDit::forward`,
      gradcheck, `ltxv_bench dit`) - it never went through the shared
      `model::block::gemm_variant`/`GemmVariants` selector
      `crates/wan/src/block.rs::linear` already uses for the identical
      10-linear block shape. Measured before (1 layer, 256 tokens, 128
      context): `matmul` was 99.7% of an 8221.69 ms whole pass at 14.4
      GFLOP/s - 0.1% of the measured roof. Fix: registered
      `matmul_reg3`/`matmul_gemv` in `LtxBlock::KERNELS`, wired `linear()`
      through `gemm_variant(..., gpu.caps().workgroup_reductions ?
      Fast{gemv,tiled} : Reference(K_MATMUL), m, n)`, mirroring `wan`'s
      `Sel`/`GemmVariants` exactly. Measured after: same shape, whole pass
      67.39 ms - **122x** - `matmul_reg3` at 3435.6-4674.7 GFLOP/s (33-44%
      of roof) depending on shape. The int8 tier (`qlinear`, the real
      generation path) was never affected: it always dispatched the fast
      tiled `matmul_i8_dyn` unconditionally (DP4A has no naive sibling to
      fall back to), so this bug was confined to the fp32 reference tier -
      training (`crate::grad`), gradcheck, parity tests, and this bench.

      **Fix 2 - `attn_scores_cross` -> `attn_scores_cross_kt`**
      (`crates/ltxv/src/block.rs::attn_scores_kt`, new small helper).
      Re-profiling the WHOLE PASS after fix 1 (8 layers, 1024 tokens, 256
      context) surfaced a NEW #1: `attn_scores_cross` at 54.4% of a
      3471.18 ms pass, 45.2 GFLOP/s (0.4% of roof). `attn_scores_cross.
      wgsl`'s own doc names the fix (K parallelises over the wrong axis for
      coalescing; transpose K once via `kv_k_headt`, read it via
      `attn_scores_cross_kt`) - the exact defect and exact fix
      `crates/wan/src/block.rs` already carries for the identical kernel
      (measured there at "91 GFLOP/s, 0.77% of fp32 peak"). Zero new WGSL
      (kernels.md sec F.3): reused both existing sibling kernels. Applied
      to BOTH `attention()` (fp32) and `attention_q()` (int8) - they share
      one score/softmax/apply trio, only the four projections differ
      between tiers, so this fix reaches the real 22B int8 production path
      too, not just the fp32 bench. Measured: `attn_scores_cross_kt` at
      161.24 ms/16 calls, 534.8-535.4 GFLOP/s (5.1% of roof) vs 1907.35 ms
      before - **11.8x** on this kernel; whole pass 3471.18 -> 1758.68 ms.
      Dead-code cleanup: `K_ATTN_SCORES`/`attn_scores_cross` removed
      entirely from `LtxBlock::KERNELS` (indices renumbered) once grep
      confirmed every attention call site in the crate - self-attn, text
      cross-attn, both A<->V cross-attention directions, the connector -
      now exclusively uses the kt path; nothing left registers a pipeline
      this crate never dispatches.

      **Fix 3 - `attn_softmax_cross` -> `softmax_rows`**
      (`crates/ltxv/src/block.rs::attn_softmax`, new small helper).
      Re-profiling again after fix 2 surfaced the next #1:
      `attn_softmax_cross` at 25.3% of the (now smaller) pass, 4.5 GFLOP/s
      (2.1% of roof). `softmax_rows.wgsl`'s own doc names
      `attn_softmax_cross` as exactly the kernel it exists to replace (one
      WORKGROUP per row instead of one thread), gated on
      `DeviceCaps::workgroup_reductions` the same way `crates/wan/src/
      block.rs::Sel.softmax_rows` already gates it. Zero new WGSL again.
      Measured: `softmax_rows` at 20.02 ms/16 calls, 100.6 GFLOP/s (46.6%
      of roof) vs 448.6 ms before - **22.4x**; whole pass 1758.68 -> 1333.87
      ms. Combined effect of fixes 2+3 alone at this shape: 3471.18 ->
      1333.87 ms (2.6x), on top of fix 1's separately-measured 122x at the
      smaller shape. `matmul_reg3` is the dominant kernel again after all
      three fixes (63.9% of the pass, 44.3% of roof) - already near this
      kernel's established ceiling elsewhere in the repo and not chased
      further (no faster registered sibling found; a genuinely new tiled
      variant would be needed, out of scope for a reuse-first pass).

      **Correctness gate for all three fixes**: unchanged - all 99
      `brain-ltxv` lib tests, plus `dit_parity`/`av_dit_parity`/
      `int8_compute`/`block_grad`/`av_block_grad`/`host_forward_parity`
      (including `int8_compute::real_q8_0_block0_int8_compute_matches_fp32`,
      a REAL Q8_0-checkpoint comparison), pass byte-for-byte identically
      before and after each fix - every fix changed dispatch selection
      only, never the math.

      **The real ~200s/step, attributed** (not guessed): a new
      `ltxv_bench streamed [layers] [tokens] [ctx_len]` mode
      (`crates/ltxv/src/bin/ltxv_bench.rs`, needs `BRAIN_LTXV_DIT`) drives
      the REAL production path (`forward_q_streamed`, int8, the real Q8_0
      GGUF) directly, and new `gpu_core::profile::stage_time`
      instrumentation inside `forward_q_streamed`
      (`crates/ltxv/src/dit.rs`) splits one forward call into: patchify +
      keyframes (host), the adaLN-single table (host), RoPE table build
      (host, f64), `open_device` (fresh `Gpu` + pipeline compile),
      embeddings-connector routing, and per-layer sums of block GGUF
      read+dequant / int8 quantize+upload / GPU forward+wait. Measured
      against the real Q8_0 checkpoint at 2 then 4 real `transformer_blocks`
      (both uncontended, `t=128` tokens, `ctx_len=64`): the adaLN table is
      FLAT at ~21 s regardless of layer count (a NEW finding - never
      previously measured, since it sits outside the per-layer loop
      entirely); GGUF read+dequant scales linearly at ~1546 ms/layer; int8
      quantize+upload at ~1802 ms/layer; GPU forward+wait at ~89 ms/layer -
      all three confirmed linear across the two layer counts. Extrapolated
      to the real 48 layers: adaLN 21.0 s (11.3%), GGUF read+dequant
      74.2 s (39.8%), int8 quantize+upload 86.5 s (46.4%), GPU
      forward+wait 4.3 s (2.3%), misc 0.36 s (0.2%) - **186.3 s total**,
      within ~5-8% of the real measured ~196-204 s/step from Phase 6 (the
      residual plausibly explained by the real generation's larger token
      count pushing up the two token-dependent terms, adaLN and GPU
      forward+wait, past this probe's `t=128`).

      This directly answers the task's own (a)/(b)/(c) question:
      block-streaming re-read+re-quantize (`forward_q_streamed`'s own doc
      already flagged this architectural cost) is the dominant term at
      **~86%** of the real per-step cost (hypothesis b) - NOT GPU compute
      (hypothesis a, ~2-3%, and now measurably fast after the three fixes
      above) - plus a previously-unknown, now well-attributed "something
      else" (hypothesis c): `ada_layer_norm_single`'s call into `linear()`
      (`crates/ltxv/src/dit.rs`) is a naive, unthreaded, unblocked
      triple-nested scalar loop that re-streams a ~604 MB weight matrix
      from host RAM once per output ROW (not once total) - flat ~21 s per
      forward call regardless of layer count, ~11% of the real total.

      **Not implemented this pass** (tracked, not silently dropped - both
      genuinely out of scope for a kernel-selector pass): (1) caching
      quantized block weights across the ~20-50 denoise steps of one
      generation run - the highest-value fix by a wide margin (it would
      remove essentially all of the ~86% streaming-I/O share, since the
      SAME weights are re-read and re-quantized from the SAME immutable
      checkpoint file every single step), but a genuine architectural
      change (in-memory residency for ~22 GB of int8 weights spanning the
      whole generation loop, touching `pipeline.rs`'s denoise loop and
      `RealDit`'s lifetime) too large to attempt safely inside this pass;
      (2) parallelizing/blocking `ada_layer_norm_single`'s naive host
      linear - a cheap, well-diagnosed, NOT-yet-attempted ~21 s/step
      (~11%) win, deliberately left as a scoped follow-up rather than a
      same-pass speculative rewrite of a function shared with other call
      sites in `dit.rs`.

      **`brain perf` integration** (`crates/cli/src/perf_cli.rs::
      build_ltxv`, the `ltxv`/`ltxv:` `strip_prefix` arm) - modeled on
      `build_wan` exactly: shape parsing+validation (`frames = 1 + 8k`,
      width/height multiples of 32), `Paths::from_env` with explicit
      existence checks, `TargetInfo` with params/quant/config axes
      (frames/width/height/steps/dit_config/engine), `ExecutorTarget::
      new_streaming` over the pre-existing `LtxvResident`
      (`crates/cli/src/resident_ltxv.rs`) and `ltxv::caps`. Defaults to
      `dit_config="tiny"` (fast, random weights) unless `BRAIN_LTXV_DIT` is
      set (then `"ltx25_22b"`) - deliberate given the ~186 s/step
      real-config cost measured above, documented in the target's own doc
      comment so a future reader does not mistake it for an oversight.

      Fixed an adjacent stale capability manifest while wiring this
      (`crates/ltxv/src/caps.rs`): `DIT_CONFIGS` only ever advertised
      `"tiny"`, even though the real 22B path (`RealDit`/
      `forward_q_streamed`) has been wired and working since real weights
      first landed in `brain ltxv t2v` - the `t2v` action's generic
      capability-dispatch path (`ActionSpec::validate`, the same path
      D-Bus `Subscribe`/`brain do` use) would have REJECTED
      `dit_config=ltx25_22b` even though the bespoke CLI and
      `dit_config_from_name` both already accept it - a real functional
      gap, not just a stale comment, that this perf target's own
      construction surfaced. Split into `T2V_DIT_CONFIGS`
      (`["tiny","ltx25_22b"]`) and `DFR_DIT_CONFIGS` (`["tiny"]` only -
      `generate_dfr` still always builds `random_tiny_weights` regardless
      of `dit_config`, confirmed by reading its body; advertising
      `ltx25_22b` there would have promised a real-weight DFR run this
      crate cannot yet produce). Gate: `caps::tests` (8/8, plus a new
      dfr-side decode check).

      Verified, not just built: the three shape-validation error paths
      (`brain perf run latency --target ltxv:8x64x64x4` etc, 3/3 correct
      messages) and a full `brain ltxv t2v` smoke generation (bypassing
      the perf/residency path entirely) at the tiny config, which
      completed cleanly post-fix - denoise 0.63-0.71 s/step, VAE decode
      67.3 s, a real mp4 written - confirming the three kernel-selector
      fixes above are correct on the real generation path, not only the
      synthetic bench.

      Found here, fixed later in shared infrastructure: device opening can
      fail to match the expected physical adapter by PCI id ("wgpu
      enumerated 0 adapters while looking for 'Tesla P40'"), falling back
      to a software adapter whose 128 MiB
      `max_storage_buffer_binding_size` is too small for even the smallest
      real-VAE decode buffer. This pass reproduced it only through
      `residency::Executor`'s GPU lane and concluded it was specific to
      that lane, because the shorter direct-CLI runs tried at the time did
      not open enough devices to reach it. That conclusion was wrong: a
      full real-weight `brain ltxv t2v` (real 22B DiT + real Gemma-4 text
      encoder, 8 real denoise steps, then VAE decode) reproduces it
      exactly, crashing at the first device opened after denoising. The
      cause is not placement logic at all - see the closed entry under
      "Recorded gaps" and the `backend-wgpu`/`vulkan` doc comments.

      **Baseline + gate** (`scripts/gates/ltxv-perf-gate.sh` +
      `scripts/gates/ltxv-perf-baselines/ltxv-tiny-9x64x64x4-cpu.json`),
      modeled exactly on `scripts/gates/qwen-serving-perf-gate.sh`:
      `latency` scenario (not `serve` - `LtxvInstance::run_batch` runs
      requests sequentially, no concurrent batching to gate the way
      qwen's admission path does; never `sweep`, whose curve artifact
      carries no flat metric `perf gate` can read), floor 0.5, SKIPs (exit
      0) when `BRAIN_LTXV_VAE` is unset. Deliberately `--device cpu`
      (sidesteps the residency GPU-adapter gap above entirely; VAE decode
      cost is device-independent per the M4 precedent, so this costs no
      real signal). Measured: `--update` and a fresh gate run both
      completed in ~70-90 s wall time (VAE decode dominated, the tiny
      DiT's own cost trivial); mutation-verified per kernels.md sec F.8:
      inflating a copy of the baseline's throughput fields 180x makes
      `brain perf gate` correctly report FAIL against the SAME candidate
      that PASSes against the real baseline - the gate actually gates.

      **CPU path - a negative result.** Profiling above (both the GPU
      kernel table and the `forward_q_streamed` stage-time breakdown)
      never showed `conv3d`, `na3d_*`, `rope*`, `gelu_erf`, `geglu_shift`,
      `snake_beta`, or `gate_row` as a measurable share of either GPU
      dispatch time or the real per-step host-side cost. Both real
      host-side costs this pass found - `ada_layer_norm_single`'s naive
      `linear()` (~11% of the real per-step total) and the dominant GGUF
      read+dequant/quantize cost (~86%) - are outside this kernel list
      entirely (checkpoint I/O and a host-side timestep-MLP projection,
      neither a WGSL-portable elementwise/conv/attention kernel
      `backend-cpu` could plausibly need an AVX microkernel for). Per this
      phase's own instruction to only add fast paths for a measured
      bottleneck, NO AVX2/AVX-512 microkernels were added - `backend-cpu`
      remains without a fast path for any of these seven kernels,
      unchanged, now backed by a measured reason instead of an unmeasured
      assumption. No AVX-512-capable CPU is available for this pass
      regardless (a Haswell-generation Xeon E5-2690 v3, AVX2 only, per
      `/proc/cpuinfo`) - matching this repo's own `row_abt_avx512` honesty
      precedent, had there been a kernel worth adding one for.

      **NPU - reconciled, not silently skipped.** `/dev/accel/accel*`
      confirmed absent again this pass. The M9 entry above already
      recorded NPU as a deliberate architectural scope exclusion for the
      22B DiT (no small fixed-shape core to peel off, unlike this repo's
      forecast/depth NPU targets). Reconciling that exclusion with the
      original validation task's "best effort" instruction: "best effort"
      was satisfied by that M9 analysis identifying the SPECIFIC
      architectural reason NPU does not apply here, not a time/resource
      constraint a better effort could close - and this phase's own
      real-weight measurements reinforce it: the real bottleneck found
      here (checkpoint I/O/dequant plus a host-side matmul) is not compute
      the `NpuModel` trait's small-fixed-shape-core pattern could address
      either, even setting the 22B-scale objection aside. No further NPU
      work follows from this phase's findings.
- [x] **Model-specific optimization, exact win 1: parallelize `ada_layer_
      norm_single`'s host `linear()` (Phase 9)** - `crates/ltxv/src/dit.rs::
      linear`, `crates/ltxv/Cargo.toml` (new `brain-backend-cpu` dependency).
      Closes the Phase 8 tracked gap named above: `ada_layer_norm_single`'s
      call into this function (the `[t,4096]x[36864,4096]^T` 9-row adaLN
      table, ~11% of the real per-step total) was a naive, single-core,
      unthreaded scalar loop. Fix per kernels.md §F.3 (grep for a faster
      sibling before writing anything new): `wan::model::linear` already
      solved this exact formula/layout with `backend_cpu::par::rows_mut`
      (row-parallel, each row's own dot product left as a straight
      sequential accumulation) - reused verbatim rather than reimplemented.
      Every output element accumulates in the identical order regardless of
      thread count, so the result is bit-identical to the old serial form,
      not merely close - the gate every exact win in this phase uses.

      **Correctness**: `dit_parity`/`av_dit_parity`/`block_grad`/
      `av_block_grad`/`host_forward_parity` (including the real-weight
      `real_weight::ltxv_real_dit_tiny_layers_matches_reference` test, real
      Q8_0 GGUF, 2 of 48 real layers) all green, every cosine/max_abs number
      unchanged from its pre-fix value (e.g. `adaln_table` cosine
      1.000000000 both before and after).

      **Measured** (`ltxv_bench streamed`, 4 real layers/128 tokens/64
      context, real `ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, both P40s
      idle at 0 MiB/0% before each run per `nvidia-smi`): the adaLN-single
      table stage 21284 ms -> 5889 ms then 6311 ms across two repeated runs
      (~3.4-3.6x, reproducible), whole streamed-forward wall time 35.59s ->
      19.40s/19.91s (~1.8x at this small shape, where the flat per-call
      adaLN cost is a much larger share of only 4 layers than it is of the
      real 48). Extrapolated the same way Phase 8's own attribution did (the
      adaLN cost is flat regardless of layer count, the streaming costs
      scale linearly): the real ~186.3s/step estimate becomes ~171.2s/step
      (~1.09x on the whole real per-step total) - a real, exact,
      bit-identical win, but a modest share of the ~200s/step total because
      the ~86% GGUF-streaming cost (a separate, NOT-yet-attempted item, see
      below) still dominates.

      Bandwidth-bound, not thread-count-bound, and recorded as such rather
      than claimed as a full fix: 48 cores were available but only ~3.5x was
      measured, because the naive loop's own defect (re-walking the whole
      ~604 MB weight matrix once per output row) is still present per-thread
      - row-parallelizing it hides the serial-core cost behind aggregate
      DRAM bandwidth, it does not remove the redundant re-reads. A
      blocked/tiled rewrite that reads each weight row once regardless of
      row count remains an unattempted further win, tracked below.
- [x] **Model-specific optimization, exact win 2: host-side per-generation
      block-weight cache (Phase 9)** - the single highest-value fix Phase 8
      identified and explicitly declined to attempt ("too large to attempt
      safely"), now closed with a lower-risk design than the one that was
      ruled out. `crates/ltxv/src/block.rs` (`CachedQLinear`/
      `CachedQAttnWeights`/`CachedQFfWeights`/`CachedQBlockWeights` - each
      existing `Q*::upload` split into a CPU-only `quantize_host` (GGUF
      fp32 in, packed int8/int4 bytes + scale out, no device touched) and a
      device-only `from_cached` (uploads already-quantized bytes, no CPU
      compute) - plus `LtxBlockQ::on_cached`, the same construction `on`
      itself now composes from), `crates/ltxv/src/dit.rs::
      forward_q_streamed` (gains a `block_cache: &RefCell<Vec<Option<
      CachedQBlockWeights>>>` parameter; its per-layer loop checks the
      cache before reading/quantizing a block and populates it on a miss),
      `crates/ltxv/src/pipeline.rs::RealDit` (owns the cache in a `RefCell`
      field, shared via the SAME reference across every one of a
      generation's forward calls - both CFG branches, every denoise step),
      `crates/ltxv/src/bin/ltxv_bench.rs` (`streamed`'s new `reuse_cache`
      argument, demonstrating the cache-miss vs cache-hit shape in one
      harness).

      **Why this design sidesteps Phase 8's own risk assessment**: Phase 8
      declined a "device-resident weights across the whole generation loop"
      design as an architectural change too large to attempt safely inside
      a kernel-selector pass, and `forward_q_streamed`'s own doc already
      records that reusing ONE `Gpu` handle across calls was tried once and
      measured WORSE (ran out of device memory a fresh device open does
      not). This design caches the already-quantized bytes on the HOST
      (plain `Vec`s in a `RefCell`, never a `DeviceBuffer`) and still opens
      a fresh `Gpu` on every forward call exactly as before - the "fresh
      device every call" constraint that made the earlier device-resident
      idea unsafe is completely untouched; only the CPU-side GGUF-read and
      int8/int4-pack work is skipped on a cache hit, and re-upload of the
      cached bytes to whichever fresh device this call opens still happens
      every time.

      **Correctness, bit-identical, not approximate**: `model::int8::
      quantize_weight`/`model::int4::quantize_weight_q4` are pure functions
      of the checkpoint's own immutable weight bytes, so a cached result and
      a freshly recomputed one are the same bytes by construction - caching
      skips redundant work, it does not change any number. Gated by a new
      `crates/ltxv/tests/block_weight_cache.rs` (4 tests, all green): a
      synthetic (`LtxDitConfig::tiny()`, always-runs, no fixture) forward
      that populates an empty cache is asserted `max_abs == 0.0` against an
      independent cache-free forward, then a SECOND call on the SAME
      (now-populated) cache is asserted `max_abs == 0.0` against the same
      reference - proving the cache-hit path computes the identical
      function, not merely a close one; a second synthetic test proves a
      cache shared across TWO DIFFERENT contexts (the real `cond`/`uncond`
      CFG shape) still produces each branch's own correct, independent
      output, catching a stale-cache-entry or wrong-layer-index bug class a
      same-context-only test cannot see; a real-weight-gated test (real
      Q8_0 GGUF, 2 layers, t=8/context_len=6, this task's own "start small"
      shape budget) repeats the bit-identical proof on the actual production
      path. Every pre-existing gate re-run unchanged and green: `dit_parity`/
      `av_dit_parity`/`block_grad`/`av_block_grad`/`host_forward_parity`/
      `int8_compute`/`int8_storage` plus the FULL 24-file
      `cargo test -p brain-ltxv --tests` suite, 0 failures.

      **Measured** (`ltxv_bench streamed <layers> 128 64 1`, real
      `ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, both P40s idle at
      0 MiB/0% before each run per `nvidia-smi`, no other GPU process):

      | shape | call 1 (cache miss) | call 2 (cache hit, same cache) | speedup |
      |---|---:|---:|---:|
      | 4 layers | 20.43-20.92 s | 4.80-6.49 s | 3.1-4.4x |
      | 8 layers | 34.96 s (4.37 s/layer) | 5.53 s (0.69 s/layer) | 6.3x |

      Per-layer stage rates from the 8-layer run (the larger, more reliable
      sample): GGUF read+dequant 1649.5 ms/layer, int8 quantize 1612.4
      ms/layer - BOTH exactly 0.0 ms on a cache-hit call, not merely small
      (the two costs Phase 8 attributed as ~86% of one real step); GPU
      upload+forward+wait 195.0 ms/layer on a cache hit (the honest
      remaining floor - upload of already-quantized bytes plus the actual
      device forward). Extrapolated to the real 48 layers the same way
      Phase 8's own attribution extrapolated (these rates are linear in
      layer count, confirmed by the 4-vs-8-layer ratio agreeing with
      layer-count doubling to within ~10%): a cache-MISS forward (the
      generation's first step) costs ~172.8 s - consistent with the ~171.2 s
      estimate exact-win-1 (the `linear()` parallelization) computed
      independently, a cross-check both estimates agree on; a cache-HIT
      forward (every step after the first) costs ~14.6 s - a **~11.8x**
      reduction per step past the first.

      Whole-generation extrapolation for the real distilled schedule (8
      steps, `LTX2_DISTILLED_SIGMAS`): at `guidance<=1.0` (1 forward/step,
      this crate's own CLI default) the pre-Phase-9 baseline is 8x186.3s =
      1490.4s (~24.8 min); post-Phase-9 (both exact wins together) is
      172.8s + 7x14.6s = 275.0s (~4.6 min) - a **~5.4x** whole-generation
      speedup. At `guidance>1.0` (CFG on, 16 forward calls, cond and uncond
      SHARE one `RealDit`/cache so only the very first of the 16 calls is
      ever a cache miss): baseline 16x186.3s = 2980.8s (~49.7 min); post
      172.8s + 15x14.6s = 391.8s (~6.5 min) - a **~7.6x** whole-generation
      speedup. Both numbers are extrapolations from the measured per-layer
      rates above, not a direct 48-layer/full-generation timing (this
      task's own "no long real-weight generation runs" constraint), stated
      as such.

      **Memory, measured not assumed** (lesson: a memory saving is not
      measured by anything unless someone measures it):
      `CachedQBlockWeights::byte_len` sums the REAL heap bytes one cached
      block holds (not `size_of`, which would only see `Vec` headers); the
      real-weight test above reads it off the actual cache populated by a
      real forward call and extrapolates x48: **270.1 MB/block measured ->
      12.96 GB for the full 48-layer model**, comfortably inside this class
      of hardware's 184 GiB RAM and well under the ~22 GB figure earlier
      milestones estimated from the checkpoint's own on-disk size (this is
      the packed int8 host footprint specifically, a related but distinct
      number, now actually measured rather than inferred).

      **Not attempted this pass, tracked**: (1) a colder cache surviving
      ACROSS separate `brain ltxv t2v` process invocations (this cache is
      per-`RealDit`-instance, i.e. per generation call, matching this
      crate's own "the DiT is free to rebuild" residency choice recorded at
      M4 - a process-lifetime or on-disk pre-quantized cache is a separate,
      bigger lift); (2) caching the adaLN-single table across the cond/
      uncond branches of one denoise step (both branches share the same
      sigma, so the table is IDENTICAL between them and is currently still
      recomputed twice when CFG is on) - a real, smaller, independent exact
      win this pass did not implement, left as a scoped follow-up; (3)
      caching the text cross-attention K/V projections themselves (constant
      across every step because `context` never changes) - subsumed by this
      milestone's WEIGHT-level cache for the read/quantize cost, but the
      qquant_linear ACTIVATION compute for `attn2.wk`/`attn2.wv` still runs
      on every cache-hit call since only the weight upload was cached, not
      the resulting K/V tensors - a further, smaller win on top of this one,
      not attempted.
- [x] **Temporal caching (TeaCache-style) - killed by measurement, not
      attempted as code (Phase 9)**. The published technique reuses a
      block's residual across a step when the timestep-conditioning signal
      changed too little to matter, keyed on a cheap proxy for that signal
      (the adaLN-single table / PixArt timestep embedding, exactly what
      this crate's own `ada_layer_norm_single` computes). Before writing any
      caching code, the real question was measured directly: how much does
      that real conditioning signal actually change between the 8 REAL
      steps of `LTX2_DISTILLED_SIGMAS` - this crate's own real checkpoint
      IS the distilled few-step schedule, so that is the only regime that
      matters here, per this task's own instruction not to force a
      synthetic result on a longer schedule this crate does not actually
      run.

      A throwaway probe (real `adaln_single.emb.timestep_embedder.*`
      weights off the real Q8_0 GGUF, `dit::timestep::pixart_timestep_embed`
      at each of the 8 real sigmas, mean\|Δembed\| between consecutive steps
      normalised by mean\|embed\|) measured the relative modulation-signal
      change at EVERY ONE of the 8 real step transitions:

      | step | sigma -> sigma | relative Δembed |
      |---|---|---:|
      | 0 | 1.00000 -> 0.99375 | 0.7775 |
      | 1 | 0.99375 -> 0.98750 | 0.5281 |
      | 2 | 0.98750 -> 0.98125 | 0.8042 |
      | 3 | 0.98125 -> 0.97500 | 0.6634 |
      | 4 | 0.97500 -> 0.90938 | 1.0419 |
      | 5 | 0.90938 -> 0.72500 | 0.8283 |
      | 6 | 0.72500 -> 0.42188 | 0.5630 |
      | 7 | 0.42188 -> 0.00000 | 0.8644 |

      This REFUTES a hypothesis the raw sigma values alone would have
      suggested: steps 0-3's own RAW sigma deltas are tiny (0.6-0.7% of the
      schedule's total range - `diffusion::scheduler::LTX2_DISTILLED_SIGMAS`
      starts with four near-identical high-noise values before the schedule
      takes its real large steps), which reads like exactly the "barely
      changed, safe to reuse" case a caching scheme wants. The MEASURED
      conditioning signal says otherwise: the PixArt sinusoidal timestep
      embedding is not smooth in sigma the way a plain linear input would
      be (a known property of Fourier/sinusoidal positional-style
      embeddings, sharpened further by the SiLU/two-layer-MLP
      nonlinearity), so even steps 0-3's tiny sigma deltas produce a
      relative embedding change of 0.53-0.80 - the SAME order of magnitude
      as steps 4-7's much larger sigma jumps (0.56-1.04), not smaller.
      Every one of the 8 transitions sits at 0.53-1.04, roughly 3-10x above
      TeaCache's own published safe-reuse threshold (~0.1-0.2 relative L1 on
      a UNIFORM, dozens-of-steps schedule this crate's real checkpoint does
      not use).

      **Killed, not scoped out**: this is a genuine negative result with a
      real measured number behind it, not a time-budget deferral - the
      real 8-step distilled schedule this crate's real checkpoint actually
      runs has NO step transition whose own conditioning signal changes
      little enough to make residual reuse a safe approximation by any
      threshold in the published technique's own range. Implementing the
      caching machinery anyway would either reuse nothing (a wasted
      complexity+quality-gate cost for zero speedup) or reuse residuals
      across a signal change several times the technique's own safety
      margin (a real, measured quality risk on an already-few-step
      distilled schedule this port cannot afford to degrade further). No
      real generation run was needed to reach this conclusion - the
      schedule's own structure, measured directly off the real checkpoint's
      real conditioning weights, already answers the question the
      generation-quality gate would have asked.
- [x] **Spatial masking / spatiotemporal cubes (Sliding-Tile-Attention-style)
      - scoped out this pass, not killed, on an analytic crossover, per
      §F.2's own "ask what the top row is running at" (Phase 9)**. Checked
      §F.3 first, as instructed: `na3d_scores`/`na3d_apply` (this crate's
      own NA diffusion decoder kernels) already accept a `[t*h*w, heads,
      head_dim]` row-major, query-major/head-minor Q/K/V layout with a
      `(kt,kh,kw)` window - and `crate::pipeline::grid_positions` (the DiT's
      own RoPE position builder) already emits tokens in EXACTLY that
      frame-major/height-mid/width-minor order (confirmed by reading both,
      not assumed) - so the kernel fit is real, not hypothetical: reusing
      these two kernels for the DiT's self-attention (`attn1`) would need
      no new WGSL, matching this task's own explicit hint to check them
      first.

      The reason this was not implemented is a measured cost question, not
      an implementation-difficulty one. Phase 8's own real-width profile
      already answers §F.2's question for the shapes it measured:
      `matmul_reg3` (the block's GEMM) is 63.9% of the pass at 44.3% of
      roof, while `attn_scores_cross_kt` is 5.1% of roof and a small
      fraction of the pass at `t=1024`/`context_len=256` - GEMM is the top
      row, not attention, at every real-width shape measured so far.
      Extending that with an analytic FLOP-count crossover (self-attention
      scores+apply scale as `~4·T²·dim`; the block's ten GEMMs scale as
      `~28·T·dim²`; `T` = video tokens, `dim` = 4096) and Phase 8's own
      measured per-kernel throughput (`matmul_reg3` ~4000 GFLOP/s,
      `attn_scores_cross_kt` ~535 GFLOP/s, used as the best available proxy
      for a windowed kernel's likely rate - no na3d throughput was measured
      on THIS hardware this pass, so this is an extrapolation from a
      structurally similar kernel, not a direct measurement) puts the
      wall-clock crossover - the token count where self-attention's own
      cost first matches the block's GEMM cost - at roughly **several
      thousand tokens** (order-of-magnitude ~4000-6000, not a precise
      number given the throughput proxy above). That is close enough to a
      plausible real-quality target (121 frames at 512x512 is `T=4096` at
      this checkpoint's own stride) that the honest answer is "uncertain,"
      not "never matters" the way the TeaCache finding above could be -
      unlike that finding, this one is NOT a killed hypothesis.

      **What would settle it, and why it was not done this pass**: a real
      measurement of the block's kernel-share profile at a real target
      resolution (`T` in the several-thousand range) is the only way to
      know whether attention has actually become the top row there - and
      building that shape means either a real multi-thousand-token forward
      pass or a synthetic-weight bench at that same token count, either of
      which is a materially bigger GPU-time spend than this task's own
      "small shapes first, no long real-weight generation runs" constraint
      allows for a single pass. Scoped out explicitly rather than
      attempted on a guess: implementing windowed attention and its own
      quality gate (STA is an approximation, needing its own quality
      threshold per this task's own instruction) against a cost question
      that is still genuinely open would be exactly the "confident
      hypothesis, profile disagrees" failure mode `.agents/rules/kernels.md`
      §E's own table exists to warn against.
- [x] **Cross-modal / codec-side optimization - scoped out, no real target
      exists yet to optimize (Phase 9)**. This task's own instruction was to
      pursue this only after confirming the AV forward is parity-proven, and
      only if the higher-value items above left budget - checked before
      spending any of that budget: `crate::pipeline::generate`/
      `generate_dfr` (the only two real-checkpoint generation entry points
      this crate has) both construct `RealDit` over `LtxDitConfig`/`LtxDit`
      exclusively; `LtxAvDit` (the audio+video model the A<->V cross-modal
      coupling actually lives in) has NO real-weight streaming counterpart
      to `forward_q_streamed` anywhere in this crate (confirmed by grep -
      the only real-weight-capable forward path streams video-only blocks).
      Phase 4's "AV forward is parity-proven" refers to forward-CORRECTNESS
      at tiny/synthetic config (`av_dit_parity.rs`), not a real-checkpoint
      generation path that actually runs - one does not exist to load real
      AV weights into at all yet (the "Recorded gaps" section already
      states this: AV training/inference both stop at `LtxAvDitConfig::
      tiny`, no real 22B AV-checkpoint bridge was built).

      There is therefore nothing for a cross-modal optimization to
      optimize: no real A<->V forward this crate can run today, proven or
      not. This is a stronger precondition failure than the ordering rule
      this task cited ("optimize only after the forward is proven") - it is
      "there is no real forward to optimize at all," a prerequisite that
      building the real-weight AV generation path itself (a real feature,
      not an optimization) would need to close first, and which is out of
      this optimization phase's own scope. Not pursued, no budget spent on
      it beyond this check.
- [x] **Observability: `--trace-ltxv <0-5>`, the first consumer of the new
      workspace tracing crate**. This port repeatedly needed the same
      breadcrumbs by hand - which layer, cache hit or miss, how long - and
      re-derived them with throwaway `eprintln!`s more than once, because
      the workspace had no logging facility at all (no `tracing`, no `log`
      in any crate's manifest before this). `crates/trace` adds the generic
      mechanism (a family registry mapping a short name onto the crates it
      covers, `tracing`'s own five levels plus off, text or JSON, stdout or
      a file); `ltxv` is its first real consumer and the proof it works end
      to end.

      Instrumented: `pipeline::generate`/`generate_dfr`/`denoise` and
      `dit::forward_q_streamed` as `#[instrument]` spans, plus
      `caps::LtxvAction::run` - the served path, whose start/success/failure
      previously existed nowhere at all because D-Bus and HTTP have no
      terminal to print to. Level 1 is every failure path with its numbers
      (including how many of how many values went non-finite, at which
      sigma); level 2 is the conditions that make a run silently not what it
      looks like (random-weight tiny DiT, stub text context, an ignored
      `--steps`, a cancellation naming its phase and step); 3 phase
      boundaries and durations; 4 per denoise step and per host stage; 5
      every individual forward and EVERY transformer block with its
      cache hit/miss and its load/quantize/GPU milliseconds.

      That last one is specifically the breadcrumb Phase 8's attribution
      needed and had to reconstruct from summed stage totals: a total can
      only say the block cache saved time on average, while the per-layer
      line makes ONE anomalous layer visible.

      `gpu_core::profile::stage_time`/`BRAIN_PROFILE` is deliberately
      untouched - the perf gate parses its stage totals - so
      `forward_q_streamed` now reports the same timings through both
      mechanisms. Consolidating them is a separate decision, explicitly not
      taken here.

      Verified on a real run (9 frames, 64x64, 2 steps, real VAE, not a
      unit test): level 5 emits the full labelled stream with span context,
      level 2 emits exactly the two "this is not a real model" warnings,
      level 0 and no flag emit zero bytes, and `--trace-format json` writes
      lines that all parse with `target`/`level`/`span` as real JSON
      members.

- [x] **Whole-generation profiling of the REAL path, and four exact wins
      (Phase 10)** - the first end-to-end attribution of an actual
      `brain ltxv t2v` run at real weights, rather than of one
      `forward_q_streamed` call in isolation. It answered a question Phase 9
      left open (its isolated bench predicted ~275s of denoise; the real run
      measured 440s) and, more usefully, found that denoise was never the
      majority of the wall clock at all.

      **Method**: `--trace-ltxv 5` on a real run (9 frames, 64x64, real
      22B Q8_0 DiT, real Gemma-4 text encoder, real VAE, 8 distilled-schedule
      steps, `guidance<=1.0` so 1 forward/step), both P40s idle at 0 MiB/0%
      before each run per `nvidia-smi`. This is the instrumentation Phase 9's
      own entry asked for and is why no timing had to be reconstructed by
      hand this pass.

      **The measured baseline, 964.3s in-process (973.5s wall)**:

      | stage | secs | share |
      |---|---:|---:|
      | DiT GGUF head load ("build transformer") | 10.8 | 1.1% |
      | **Gemma-4 text encode** | **491.7** | **51.0%** |
      | denoise, 8 steps | 440.4 | 45.7% |
      | VAE decode | 21.1 | 2.2% |

      The text encode had NEVER been measured: `Timings` does not carry it,
      so every previously reported ltxv number ("build 58.7s, denoise 412.1s,
      vae 20.9s") silently excluded the single largest stage. That is the
      whole reason the reported parts never summed to the reported total.

      Denoise splits into step 1 (365.4s, every block a cache miss) and steps
      2-8 (9.7-12.8s each, 74.1s total). Phase 9's extrapolations were wrong
      in BOTH directions and the errors partly cancelled: it predicted
      ~172.8s for the miss step (real: 365.4s) and ~14.6s per hit step (real:
      ~10.6s). Step 1's own split at 48 real layers: GGUF read+dequant
      251.5s, int8 quantize 96.9s, GPU upload+forward+wait 7.7s.

      **Root cause of the divergence, measured not assumed: the storage
      reads at ~58-68 MB/s cold.** A `dd` of 8 GiB from an uncached region of
      the real checkpoint measured 58.4 MB/s; a second file measured
      70.6 MB/s; a re-read of the same region once page-cached measured
      4.3 GB/s. Sixteen parallel readers measured 60 MB/s - the device does
      NOT respond to queue depth, so no amount of read-side concurrency moves
      it. A real generation reads ~50 GB cold (26.3 GB text encoder + ~22 GB
      of DiT blocks), which is ~800s of unavoidable I/O wait and by itself
      most of the run. Confirmed independently by the process accounting:
      the baseline run spent 7m12s of CPU across 16m14s of wall clock.
      Phase 9's isolated bench never saw this because repeated runs left its
      four layers warm in page cache - the bench was measuring dequant CPU
      where the real run measures disk.

      **This reframes the task**: at real scale the DiT's GPU compute is
      7.7-9.2s of a ~964s run. There was no kernel to fix. Per kernels.md
      §F.2, the top row was checked against the roof and the answer was that
      the top rows are not compute at all. Four exact wins followed, all
      bit-identical, all with their own mutation-verified gate:

      **Win 1 - row-parallel int8/int4 weight quantization**
      (`model::int8::quantize_weight`, `model::int4::quantize_weight_q4`,
      new `backend_cpu::par::chunks_mut`). Both quantizers walked their
      output rows on one core of 48. Per-output-row by construction (row r's
      scale is `max|w[r,:]|/q_max`, row r's words read only row r), so the
      split cannot move a value. **96.9s -> 6.8s and 6.6s across two real
      runs, 14.3x, reproducible**; on `ltxv_bench streamed 4 8 128 1`,
      6517.9ms -> 554.7ms with the forward's printed output stats unchanged
      to every digit. Gated against a serial oracle transcribed from the doc
      comments' own formulas, over five shapes including a single row and
      non-dividing row counts; mutation-verified by hoisting the max fold to
      per-tensor (the obvious wrong parallelization, and invisible to a
      cosine-only check since it rescales uniformly). In `crates/model`, so
      every model with an int8/int4 tier inherits it (§F.7).

      **Win 2 - block-parallel GGUF dequantization**
      (`checkpoint::gguf::deq_blocks`). Warm-cache read+dequant 1573ms/layer
      -> 1130ms/layer. Deliberately recorded as the SMALL win it is: what
      remains is allocating and filling ~1.8GB of fp32 per block, and on a
      cold cache the stage is I/O-bound anyway, so the parallelism has
      little to return here. Its real value is the gate that came with it -
      every pre-existing dequant test in that file feeds exactly ONE block,
      which is structurally blind to block ordering (lesson #4). The new
      multi-block tests compare by BIT PATTERN against a per-block oracle
      (random-byte fixtures legitimately decode NaN scales, and NaN != NaN
      would fail a value comparison between byte-identical results) and fail
      when the block order is reversed.

      **Win 3 - map safetensors instead of slurping, and decode dtypes in
      parallel** (`checkpoint::safetensors::read`/`parse`). `read` called
      `std::fs::read`, keeping a second anonymous copy of a 26.3 GB file on
      the heap beside the fp32 tensors being built from it. Peak RSS
      ~80GB -> 73GB. The point is not the memory: an anonymous copy of
      bytes that are already in page cache EVICTS page cache, on a machine
      where the checkpoints are a large fraction of RAM and the storage
      behind them runs at 58 MB/s. The dtype conversion (billions of
      independent 2-byte decodes) was also single-core and is now split
      through one `decode_elems` helper. Text encode measured 491.7s ->
      383.6s and 291.8s across two runs - improving, but with a spread far
      wider than the change itself, because the stage is ~438s of disk at
      the measured rate. Recorded as "masked by I/O", not claimed as a
      clean speedup.

      **Win 4 - cache the embeddings-connector routing across a generation**
      (`ltxv::block::GenerationCache`, `crate::dit::forward_q_streamed`,
      `crate::pipeline::RealDit`). The connector reads only `context`,
      `context_valid` and `context_len`, all fixed for a generation once the
      prompt is encoded, yet ran on every step: 8 transformer layers whose
      fp32 weights (~6.4 GB at the real width) were re-uploaded to that
      call's fresh device each time. **2537-4839 ms/step -> 1.0-7.6 ms on a
      hit**, ~22.6s per generation, and stable across runs unlike the
      I/O stages around it. Exact for the same reason the block-weight cache
      is: identical inputs through a pure function. Keyed on the FULL
      context, validity mask and length, never a step index or a hash,
      because CFG runs two branches against one `RealDit`. Gated on
      `tiny_gated` (plain `tiny` disables the connector, so the existing
      cache tests could not reach this path at all) including a case where
      only the VALIDITY MASK differs; mutation-verified by dropping the
      context from the key. Both per-generation caches now live in one
      `GenerationCache` so `forward_q_streamed` stops growing a
      `&RefCell<..>` parameter per cached thing.

      **Whole-generation result, real runs, same 9-frame/64x64 real-weight
      shape** (kernels.md §F.1: the whole pass is the truth, never the
      per-stage table):

      | run | total | build | text encode | denoise | vae |
      |---|---:|---:|---:|---:|---:|
      | before | 964.3s | 10.8 | 491.7 | 440.4 | 21.1 |
      | after (1) | 672.7s | 8.0 | 383.6 | 254.6 | 26.2 |
      | after (2) | 704.6s | 30.5 | 291.8 | 363.7 | 18.3 |

      **~1.40x end to end (964.3s -> 672.7/704.6s), with the output mp4
      byte-for-byte IDENTICAL across all three runs** (same md5, same seed,
      same prompt, same real weights) - the strongest available proof that
      every win here was exact rather than merely close.

      The two `after` runs differ by 4.7% purely from disk luck, so the
      honest way to read the headline is the paired one: run (2) drew a
      WORSE step-1 disk read than the baseline did (272.4s vs 251.5s) and
      still finished 259.7s faster. The improvement is therefore at least
      that much and is not an artifact of a lucky cache.

      **One reproducible partial regression, recorded rather than hidden.**
      The block upload+forward+wait bucket on the FIRST cache-hit step (step
      2) went 7.2s -> 14.4s and 14.2s, in both `after` runs; steps 3-8 are
      unchanged within noise (6.4-8.7s against 5.7-7.2s). It is once per
      generation, ~7s, against the connector cache's ~22.6s, so steps 2-8 in
      total still improved (74.1s -> 68.0s) and the whole-pass number is what
      the change is judged by (lesson #21). The likely mechanism is that step
      2 is the first call that reads all 48 cached blocks back out of host
      RAM after step 1 wrote them, and it used to be preceded on the same
      fresh device by the connector's own multi-gigabyte allocate/free cycle;
      not chased further, and stated as a hypothesis rather than a
      measurement, because nothing here measured it.

      **Tracked, deliberately not attempted** (each real, none silently
      dropped): (1) the ~614s of cold checkpoint I/O per run is the dominant
      remaining term and no code change removes it - the levers are reading
      fewer bytes (the Q4_K_M checkpoint is 15.7 GB against Q8_0's 23.6 GB,
      a quality decision, not a perf one) or not re-reading across runs (an
      encoded-context or pre-quantized-weight cache keyed on
      checkpoint identity, which would make a REPEAT validation run skip the
      whole 491.7s text encode - the single highest-value remaining item for
      iteration speed, and a design decision rather than an optimization);
      (2) `load_block_tensors_from_source` copies each tensor a second time
      (`d.to_vec()` over a `Vec` the GGUF reader just allocated), ~1.8 GB of
      needless allocation and memcpy per block, ~86 GB per generation -
      removing it needs a `TensorSource` seam that can hand over ownership;
      (3) the Q8_0 -> fp32 -> int8 round trip materializes ~1.8 GB of fp32
      per block only to compress it straight back to ~0.46 GB - fusing
      dequant into quantize per row would be bit-identical and would delete
      that traffic entirely, but it is a streaming-pipeline restructure, not
      a loop change; (4) the per-step block GPU upload (~6.5-8.9s/step,
      re-uploading ~13 GB of already-quantized bytes to a fresh device every
      step) is now the largest remaining per-step cost, and is bounded by
      `forward_q_streamed`'s own "fresh `Gpu` per call" design, which its
      doc records was measured worse to remove.

      **Content quality - checked, nothing to report as a bug.** The task
      asked for a sanity check only. The decoded frames are finite, with
      real dynamic range and no saturation (luma average 176-220, per-frame
      low 132-168, high 201-208), no NaN or non-finite value anywhere in the
      run, and no warning or error in the trace beyond the documented
      "`--steps` is ignored for the distilled schedule" one. At this shape
      the DiT sees 2x2x2 = 8 tokens for the entire scene, so abstract output
      is expected and is not evidence of a defect. No correctness anomaly
      found; nothing changed in that direction.

### Phase 11 - the text encoder: a generic quantizer, an int8 tier, and a context cache

Phase 10 measured the Gemma-4 text encode at 51% of a real generation's wall
clock and recorded it as never having been measured before. This phase
attacked it, and the first thing it did was measure the stage's own INSIDE,
because "the text encode is slow" does not say which of three completely
different fixes applies.

**Method**: `--trace-ltxv 4` on the same real run Phase 10 used (9 frames,
64x64, real 22B Q8_0 DiT, real Gemma-4 text encoder, real VAE, 8
distilled-schedule steps, `guidance<=1.0`), both P40s idle at 0 MiB / 0%
before each run, and the encoder + DiT + VAE files explicitly evicted from
page cache before each run (`posix_fadvise(DONTNEED)`) so both arms measure
a genuinely cold read rather than one arm measuring the page cache.

#### What the stage is actually made of (measured, first time)

| sub-stage | secs | share |
|---|---:|---:|
| checkpoint read + bf16->f32 decode + import | 474.7 | 90.4% |
| 48-layer tower forward (11 tokens, fp32) | 45.9 | 8.7% |
| aggregate-embed projection | 0.6 | 0.1% |
| **text encode total** | **524.8** | |

Whole run, fully cold: **1069.3 s** (build 48.3, text encode 524.8,
denoise 470.8, VAE 25.3, unattributed 0.1). Note the parts now sum to the
total - Phase 10's entry recorded that they did not, and `Timings` now
carries `text_encode` plus an `unattributed` remainder so a future missing
stage is visible rather than silent.

**The stage is 90% I/O**, at 52.8 MiB/s over 26.26 GB - the same cold
storage rate Phase 10 measured independently with `dd`. That single number
reorders every candidate fix:

- reading FEWER BYTES is the whole game on a first run;
- making the arithmetic faster can address at most the 45.9 s the forward
  costs, so an int8 compute tier is the SMALLEST of the three levers here,
  not the largest - the opposite of the intuition that motivated it;
- not reading the bytes AT ALL is worth more than both, for the repeat-run
  workflow this pipeline is actually used in.

All three were built, and they are complementary rather than alternatives.

#### After: three levers, measured separately

Same shape, same prompt, same protocol, page cache evicted for the encoder,
DiT and VAE before each run.

| stage | bf16 + fp32 | Q8_0 + int8 | Q8_0 + int8, cache hit |
|---|---:|---:|---:|
| build transformer | 48.34 | 50.88 | 46.11 |
| **text encode** | **524.8** | **304.3** | **0.011** |
| denoise, 8 steps | 470.8 | 365.6 | 383.5 |
| VAE decode | 25.3 | 27.4 | 25.9 |
| unattributed | 0.1 | 0.2 | 0.1 |
| **total** | **1069.3** | **748.3** | **455.6** |

End to end **1069.3 s -> 748.3 s (1.43x)** on a first encode of a prompt,
and **-> 455.6 s (2.35x)** on any later run of the same one. Text encode
alone **524.8 -> 304.3 s (1.72x)**, then **0.011 s** - a 4.2 MB cache entry
standing in for a 26.3 GB read and a 12B forward.

The three arms' build/denoise/VAE columns differ by a few percent in both
directions, which is the run-to-run spread of a cold-storage-bound pipeline
and not a signal; only the text-encode column is a controlled comparison.

Sub-stage split of the encode, which is where the reasoning lives:

| sub-stage | bf16 + fp32 | Q8_0 + int8 |
|---|---:|---:|
| encoder opened (read/decode/import) | 474.71 | 24.20 |
| 48-layer forward | 45.94 | 278.37 |
| aggregate-embed | 0.64 | 0.70 |

The two forwards are NOT comparable in isolation and the table would lie if
read that way: the streamed forward CONTAINS the per-layer checkpoint read
that the eager path had already paid for under "opened". The total is the
honest comparison. The residual 278 s is still mostly reading ~11.5 GB of
layer weights, not arithmetic - consistent with the 90%-I/O finding, and the
reason the remaining lever is fusing the Q8_0-to-fp32-to-int8 round trip
rather than a faster GEMM.

**Denoise also dropped 105 s, and nothing in the DiT path changed.** The
honest explanation is memory, not compute: the eager encoder peaked at
74.5 GB RSS and the streamed one at roughly a quarter of that, so the DiT's
own streaming read has far more page cache left to work with. Recorded as a
secondary effect rather than claimed as a denoise optimization.

#### Correctness of the int8 tier

Real 12B weights, both arms reading the SAME Q8_0 file so the tier is the
only variable (comparing an int8 GGUF against a bf16 safetensors forward
would confound the arithmetic with the storage format):

| comparison | cosine | rel_l2 |
|---|---:|---:|
| layer 0, `sliding_attention` | 0.997754001 | 0.067 |
| layer 5, `full_attention` (MQA, `k_eq_v`) | 0.998706203 | 0.051 |
| **whole encoder, 48 layers, `last_hidden_state`** | **0.999915592** | **0.019** |

The whole-encoder number being BETTER than either single layer is not a
mistake: per-layer quantization errors are independent and partly cancel
down the stack, and the model's own final `norm` removes what is left of the
scale. Both layer types are gated on purpose - they are structurally
different graphs (head_dim 256 vs 512, 8 KV heads vs 1, a real `v_proj` vs
none), and the `full_attention` path has the least redundancy for an error
to hide in.

Peak process RSS over a whole generation, measured from `/proc/<pid>/status`
rather than derived: **71.05 GiB eager -> 13.11 GiB streamed**, and most of
what remains is reclaimable mmap pages rather than heap.

#### What was reused rather than written

- `checkpoint::quant::quantize` already implemented every ggml block
  encoder, including Q8_0. The generic converter adds a policy, a plan and a
  streaming writer around it, not new quantization math.
- `model::int8::quantize_weight`'s packed layout and the
  `max_abs_row`/`quant_pack`/`matmul_i8_dyn` trio are what `ltxv`'s own
  quantized blocks already dispatch. **Zero new kernels.**
- `model::hostmath::matvec_par` for the aggregate-embed projection.
- `gpu_core::cache_dir()` for the context cache's location, made `pub`
  rather than copied.
- `import::classify` for the GGUF loader's name space, exposed as
  `canonical_weight_name`/`is_recognized_non_weight` so the two loaders
  cannot drift on which tensors exist.

#### What was refuted or found the hard way

- **The int8 parity gate's first version was worthless against the most
  likely bug.** See lesson #47.
- **`AggregateEmbed` was not "host glue too small to matter".** It is
  `Linear(188160 -> 4096)` - 770 M multiply-accumulates per token - and ran
  as a scalar loop on one core.

#### Closed from Phase 10's own gap list

- ~~"The real text-encode stage is invisible to `Timings`"~~ - closed.
  `Timings::text_encode` plus `Timings::unattributed(wall)` printed as its
  own row on both CLI lines; the baseline run above reports 0.1 s
  unattributed, so the parts now sum to the total.
- ~~"a cache of the ENCODED TEXT CONTEXT ... Not attempted: it is a
  cache-invalidation design decision, not an optimization"~~ - attempted,
  and the design decision made rather than deferred. The invalidation
  question is answered by not relying on the digest at all: it is a
  filename, the full key material lives in the entry, and load compares it
  field by field, so a collision is a miss and never a wrong context. That
  is what makes it safe on by default.

#### Tracked gaps this phase leaves

- **The streamed encode still round-trips Q8_0 -> fp32 -> int8 per layer.**
  `Gemma4GgufSource` hands out f32 (that is what `TensorSource` is), and
  `Proj::upload` immediately re-quantizes it. Fusing the two per row would
  be bit-identical and would delete both the intermediate fp32 and a large
  share of the remaining 278 s. Same shape as the identical gap Phase 10
  recorded for the DiT's own streamed blocks, and it wants the same
  `TensorSource` seam - deliberately not bolted on as a defaulted trait
  method.
- **`embed_tokens` is decoded whole to gather a handful of rows.** The
  table is `[262144, 3840]`; a real prompt touches a few rows of it, and the
  loader dequantizes ~1 GB of Q8_0 to 4 GB of f32 to do so. A row-range read
  on `MmapGguf` (the twin of `MmapSafetensors::tensor_f32_range`) would fix
  it. Not measured separately, so its share of the 278 s is unknown - it is
  a known inefficiency, not a quantified one.
- **The whole-encoder int8-vs-fp32 comparison is opt-in, not a default
  gate.** `real_q8_0_whole_encoder_int8_matches_fp32` needs
  `BRAIN_GEMMA4_FULL_PARITY=1` because it reads 13 GiB twice; the default
  gate is per-layer, matching `ltxv`'s own precedent.
- **The aggregate-embed head is loaded as 3 GB of f32 every encode**
  (19.1 s of the 24.2 s open). It is a genuine `[4096, 188160]` GEMM
  operand, so it is a candidate for the device and for int8, but at one
  encode per generation it was not worth the complexity this pass.
- **The cache is never evicted.** Entries are ~4 MB each and keyed on
  prompt plus encoder identity, so a long-lived workflow accumulates them.
  No size cap, no LRU, no `brain` verb to clear it - deliberately, since a
  policy nobody asked for is worse than a documented directory, but it is a
  real gap rather than an oversight.

### Phase 12 - self-attention becomes flash attention (the O(T²) crossover, measured)

Phase 8's per-kernel work was done at T=1024/ctx 256, where `attn1` and
`attn2` cost about the same. This phase re-profiled at the REAL 720p latent
token count and the ranking had completely changed - which is the whole point
of `.agents/rules/kernels.md` §F.9.

**Method**: `BRAIN_LTXV_DIT=<real 22B distilled Q8_0 GGUF>
./target/release/ltxv_bench streamed 8 3520 1024 1`, one Tesla P40, idle
before each run, both arms the SAME command on the same checkpoint. T=3520 is
the real 720p/25-frame/32-stride grid (`lat_t=4, lh=h/32, lw=w/32`). Numbers
below are the **cache-hit** call (call 2), the shape every step of a
generation past the first has. VRAM is `nvidia-smi --loop-ms 200` peak across
the run.

**Before** - GPU kernel time 5556.4 ms per 8-layer forward:

    attn_apply_cross       2095.8 ms   16 calls  (37.7%)
    attn_scores_cross_kt   1994.1 ms   16 calls  (35.9%)
    matmul_i8_dyn           903.8 ms   80 calls  (16.3%)
    softmax_rows            281.6 ms   16 calls  ( 5.1%)

Self-attention alone (the 8 `attn1` calls, separated from the 8 `attn2` calls
that share those kernel slots) was ~3382 ms of that 5556 ms. It is the only
op in the block that is O(T²) in the video token count; everything else is
O(T), so it overtakes the whole block once T passes a few thousand - exactly
the analytic crossover an earlier pass predicted and declined to act on
without a measurement.

**The change**: `attn1` (and the embeddings connector's own self-attention,
same construction) now dispatches `model::block::flash_bidir_fwd` over a
`pack_qkv` slab instead of the materialized `attn_scores_cross_kt` ->
`softmax_rows` -> `attn_apply_cross` chain. No new kernel and no new cost
formula: all five kernels already existed, already gated, already measured on
this hardware. `attn2` and the A<->V cross-attention are NOT self-attention
(different key row set) and keep the trio unchanged. Gated on
`DeviceCaps::workgroup_reductions`, so `BRAIN_DEVICE=cpu` still takes the
exact trio - which stays the reference definition of the math.

**After** - GPU kernel time 2614.8 ms per 8-layer forward:

    matmul_i8_dyn           907.1 ms   80 calls  (34.7%)
    attn_apply_cross        483.2 ms    8 calls  (18.5%)   <- attn2 only now
    attn_scores_cross_kt    444.4 ms    8 calls  (17.0%)   <- attn2 only now
    flash_attn_bidir_reg2   432.5 ms    8 calls  (16.5%)   <- all of attn1
    pack_qkv                 10.4 ms    8 calls  ( 0.4%)

| | before | after |
|---|---|---|
| self-attention, 8 layers | ~3382 ms | 443 ms (**7.6x**) |
| GPU kernel time, 8 layers | 5556.4 ms | 2614.8 ms (**2.13x**) |
| wall, 8 layers (cache hit) | 101.41 s (12.68 s/layer) | 93.86 s (11.73 s/layer) |
| peak VRAM | 18189 MiB | 16522 MiB |

**1080p (T=8160) went from impossible to routine.** The materialized path
needs a `[32, 8160, 8160]` fp32 score slab; this device reports
`max_buffer_size` 4094 MiB, and the old code fails at buffer creation, not
slowly:

    wgpu error: Validation Error
      In Device::create_buffer
        Buffer size 8522956800 is greater than the maximum buffer size (4292870144)

The flash path runs the same shape in 303.8 ms of self-attention at 16650 MiB
peak. `pack_qkv`'s slab is 3*T*inner_dim floats - 173 MB at T=3520 - against
two 1.55 GiB slabs.

**The fused path is also MORE ACCURATE than the one it replaces**, which is
the finding worth keeping. Random operands cannot see it: both arms agree to
~1.6e-7 at the exact production geometry (heads=32, head_dim=128, T=3520).
But when every token row is identical - which is precisely what
`ltxv_bench streamed`'s all-zero latent produces after patchify+bias - the
answer is analytically exact (`ctx == v`, uniform softmax) and the two arms
separate:

    identical rows, t=3520:  max|flash - exact| = 2.4e-6
                             max|materialized - exact| = 4.7e-5

`attn_apply_cross` sums T equal positive terms sequentially in one thread and
drifts by O(T·eps) coherently across every row; the flash kernel's blocked
online accumulation does not. This is why the real-checkpoint int8 bench's
output statistics move by ~1e-3 relative between the two arms even though the
attention itself is exact to 1e-7 - the drift being removed is the OLD path's.
That was chased down rather than assumed: a reduction-order-only perturbation
(swapping `softmax_rows` for `attn_softmax_cross`) moves the same output by
only ~4e-6, which killed the "int8 quantization amplifies any perturbation"
hypothesis and forced the real explanation out.

**Gates** (`crates/ltxv/src/block.rs`'s own `tests` module, both
mutation-verified - `k_off`/`v_off` swapped and `head_dim - 1`, each caught at
1e-1 against a 1e-5 bar):
- `flash_self_attention_matches_the_materialized_reference_and_a_host_oracle`
  - five shapes (query-tile tails, both parity fixtures' geometries, the real
  head_dim=128) x three implementations (flash, the materialized trio, a host
  f64 oracle), `max_abs` asserted alongside cosine because cosine is
  scale-invariant and cannot see a wrong `1/sqrt(hd)` (lesson #2).
- `flash_self_attention_beats_the_materialized_trio_on_long_sequence_accuracy`
  - the identical-rows case above, against the analytic answer, asserting the
  ORDERING (fused must not be less accurate than the trio) and that the flash
  path was actually selected, so a silent fallback cannot pass it vacuously.

Every pre-existing gate still passes unchanged, on both backends:
`dit_parity`, `av_dit_parity`, `host_forward_parity`, `connector_real_parity`,
`streamed_vs_eager_real`, `block_grad` and the rest of `-p brain-ltxv`, and
the same set again under `BRAIN_DEVICE=cpu` (which is what proves the
non-cooperative fallback branch, §F.4). `int8_compute` fails under
`BRAIN_DEVICE=cpu` before and after this change alike - `matmul_i8_dyn` has no
CPU JIT - so the int8 tier remains GPU-only, unchanged.

**§F.9: the bottleneck moved, and it is no longer on the GPU.** The same
profiling run makes the next target unambiguous, and it is not a kernel:

    stage forward_q_streamed: adaLN-single table (host): 76281.3 ms

That is a per-CALL host cost, independent of layer count (identical at 1 layer
and at 8), and at T=3520 it is ~81% of the 93.86 s wall for 8 layers. It is
`dit.rs`'s host `linear` behind `ada_layer_norm_single`, a
`[3520, 4096] x [36864, 4096]^T` GEMM - 531 GFLOP running at ~7 GFLOP/s on the
host while a 10.5 TFLOP/s card sits idle. Until that moves to the device, GPU
kernel work is a minority of a real step and further kernel optimization has a
small ceiling. Recorded here rather than acted on: it is a different phase.

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

- Full 22B DiT real-weight parity: **partially closed, connector gap now
  closed too**. The "real-weight parity ladder for the 22B DiT, reduced
  depth" milestone proved quantization exactness (Q8_0, exact) and
  video-stream port correctness at REAL width, 2 of 48 `transformer_blocks`,
  gated attention ON, `use_embeddings_connector: false`. Two follow-up
  real-weight diagnostics (triggered by a user-reported "pure noise, not a
  real image" real generation and a request to only trust upstream, not this
  crate's own doc comments) closed two of that milestone's own named gaps:
  1. **`video_embeddings_connector`'s own real-weight parity** (real width:
     dim=4096, 8 layers, 128 registers, gated attention ON, loaded straight
     off the real Q8_0 GGUF, `tools/goldens/ltxv_real_connector_dump_
     reference.py` instantiating the reference's own `Embeddings1DConnector`
     directly) - `crates/ltxv/tests/connector_real_parity.rs`: cosine
     0.999999973 against the reference on a synthetic-but-real-shaped
     `[128, 4096]` input (20 real + 108 register-substituted positions,
     right-padded). Every block's cross-attention routes through this
     module's output in the real pipeline, so this was the single
     highest-blast-radius untested component in the whole port.
  2. **`crate::dit::forward_q_streamed` (what `RealDit::forward` actually
     calls, i.e. what every real `brain ltxv t2v --dit-config ltx25_22b` run
     executes) had NEVER been checked against anything at real weights** -
     every other real-weight gate in this crate replays against the EAGER
     `LtxDit::forward`/`forward_q`; `forward_q_streamed`'s only prior
     coverage (`block_weight_cache.rs`) used `random_tiny_weights`.
     `crates/ltxv/tests/streamed_vs_eager_real.rs` runs both paths on the
     SAME real Q8_0 weights, SAME config (gated attention + embeddings
     connector both ON, int8 compute tier), SAME inputs: bit-identical
     (cosine=1.000000000, max_abs=0.0).

  Still NOT run: all 48 layers at once, the audio stream, the A<->V
  cross-attention (`LtxAvDit`), the Q4_K_M quant tier.
- 12B Gemma-4 real-weight parity: **closed at reduced depth**. Same
  trigger as above. `tools/goldens/gemma4_real_dump_reference.py` builds
  the REAL reference `Gemma4UnifiedTextModel` at real width (hidden=3840,
  head_dim=256 sliding / 512 full, GQA 8kv-heads sliding / MQA+`k_eq_v` 1kv-
  head full, `partial_rotary_factor=0.25` on the full/global RoPE table
  only) on the first 6 of the real checkpoint's 48 layers (5 sliding + 1
  full, the real 5:1 `sliding_window_pattern`'s minimal instance), loaded
  from the real 26 GB bf16 checkpoint (`gemma4-12b-with-proj-ltx-2.5-bf16.
  safetensors`). `crates/gemma4/tests/real_weight_parity.rs` replays the
  golden's own `input_ids` through `gemma4::Gemma4Model::forward`: BOTH RoPE
  tables, BOTH attention types' own self-attention output (sliding/GQA AND
  full/MQA+k_eq_v, the two structurally different paths), EVERY one of the 7
  `hidden_states` entries, and `last_hidden_state` all match the reference
  at cosine=1.000000000 (max_abs on the order of 1e-5 to 1.6e-2, consistent
  with fp32 accumulation drift across 6 real layers, not a port bug). This
  closes the "needs hardware that can hold the checkpoint" excuse for good -
  the checkpoint has been on disk and loadable since the Q8_0 conversion
  work - and the "genuinely unattempted" status this gap carried since the
  very first exploration of this port. Not run: the full 48 layers at once
  (compute-bound on CPU-only torch, not a scope question - the reduced-depth
  6 layers already exercise BOTH structurally different attention paths, so
  a bug specific to layer count rather than layer TYPE is the only thing
  left unruled-out) and the LTX-specific aggregate-embed projection's own
  real-weight numbers (its shape is sized for the real 49-state tuple,
  out of scope for a 6-layer/7-state reduced run - already parity-proven at
  tiny scale, `gemma4::tests::parity::gemma4_tiny_matches_reference`).

  Also found and fixed in the same pass, independent of any of the above:
  `real_text_context`'s `context_len` was computed as the real prompt's own
  token count rounded up to the nearest multiple of the connector's register
  count (128) - for a typical short prompt, 128. The reference's own
  `gemma_assets.py::TOKENIZER_MAX_LENGTH = 1024` pads (or truncates) every
  prompt to a FIXED 1024 regardless of its own length; the connector routes
  the FULL 1024-wide sequence through cross-attention on the real checkpoint,
  never a prompt-length-dependent shape. A short prompt was silently getting
  roughly 1/8th the context length the checkpoint was calibrated against.
  Fixed (`crates/ltxv/src/pipeline.rs`); confirmed via a real generation
  that content became visibly more prompt-relevant once the stale text-
  context cache this bug had populated was cleared and regenerated.
- NPU device execution: `ltxv` gets no `NpuModel` implementation at all this
  port (see the M9 perf entry's NPU write-up above for the full reasoning) -
  the firmware-not-functional blocker is separately
  diagnosed in `.agents/roadmap/dtype.md`, not re-run here.
- `vae::blocks3d` has no backward (`blocks/grad.rs` only covers the 2D builder),
  so video-VAE fine-tuning is out of scope until that lands separately.
- Multi-device residents (needed for a sharded 22B DiT across multiple cards)
  do not show up in `braintop` - a pre-existing gap noted in the serving-
  contract exploration, not new to this port.
- Image-to-video, IC-LoRA pipelines, and the `DubIt` speaker-identity pipeline
  are out of scope for this port.
- `examples/videogen/` is `wan`-authored and generic enough to drive `brain/
  ltxv`'s `t2v` action as-is (same param names, `--model brain/ltxv` plus
  explicit `--width`/`--height` compatible with the 32-stride VAE), but its
  own CLI defaults (416x240) are wan's, not a multiple of ltxv's stride, and
  it has no `dfr` coverage (a different action name, a different size-stride
  rule, and `dfr`'s own `temporal_upsample_rounds` param) - a dedicated
  example/README for `dfr` has not been added.
- INT8 storage for the DiT (`crate::int8`) is not wired into any real
  checkpoint importer's load path, and whether this port ever wants a
  compute-time int8 kernel (vs. storage-format-only) is unsettled - that
  needs the DiT's own performance profile to say arithmetic is the
  bottleneck first, per porting.md sec10 point 6.
- Real multi-device pipeline-parallel execution: **closed for `LtxAvDit`**,
  still open for `LtxDit` (video-only). The "int8/int4 compute + AV
  sharding" milestone above ran `LtxAvDit`'s two-stage split on two REAL
  physical GPUs for the first time (`crates/ltxv/tests/av_shard_2gpu_real.rs`)
  and it agreed with the single-process reference - but only at a small
  synthetic `tiny_gated` config, not real checkpoint weights (that needs a
  GGUF-streaming int8 shard loader, a tracked gap of its own). `LtxDit`'s own
  `model::Shardable` impl has still only been run single-process (the
  single-shard degenerate case and a sequential two-stage boundary-handoff
  test) - its two-real-GPU proof was not repeated separately since
  `LtxAvDit` is the superset path.
- `LtxDit` has no backward/training pass through the `model::Shardable`/
  `model::Pipeline` seam - `crate::grad`/`crate::modelgrad`'s existing
  host-math training path is separate and does not build on it. `LtxAvDit`
  now has an equivalent host-math training path (`crate::av_grad`/
  `crate::av_modelgrad`, the "training for the audio+video DiT" milestone
  above) but the SAME `Shardable`/`Pipeline`-seam gap applies to it too -
  neither DiT's device-sharded backward is implemented.
- Gated attention's (`to_gate_logits`) backward and both embeddings
  connectors' training are not implemented for either DiT - the host-math
  training paths (video-only and AV) both train only the ungated,
  connector-disabled config point (`LtxDitConfig::tiny`/
  `LtxAvDitConfig::tiny`, not `tiny_gated`). No real-checkpoint
  GGUF-to-training-weights bridge exists for the AV DiT either -
  `AvModelWeights::from_tensors` reads a checkpoint-name-keyed tensor map
  (mirroring `ModelWeights::from_tensors`), but nothing feeds it the real
  22B AV weights yet, so AV training is proven only at tiny/synthetic
  scale.
- Real-checkpoint weight caching across a generation run: the Phase 8
  performance entry above measured that `forward_q_streamed` re-reads and
  re-quantizes all 48 real blocks from the GGUF on EVERY forward call
  (~86% of the real ~200s/step) - **closed in Phase 9** (see below): a
  HOST-side (not device-VRAM-resident) cache of already-quantized block
  bytes, shared across a generation's forward calls, sidesteps the
  device-memory risk that made this "too large to attempt safely" in
  Phase 8 (lesson #35's wgpu resident-buffer overhead applies to DEVICE
  VRAM, not host RAM, which this class of hardware has 184 GiB of). Not
  fully closed: the remaining cache-hit cost (~14.6s/step extrapolated,
  down from ~186s) is now GPU-upload + adaLN-table-recompute-dominated,
  and the FIRST forward of every generation still pays the full
  uncached cost (~172.8s extrapolated) - a colder-cache-across-generations
  scheme (e.g. process-lifetime residency across multiple `brain ltxv t2v`
  invocations) was not attempted, out of scope for a single generation's
  own pipeline.
- `ada_layer_norm_single`'s host-side `linear()` call was a naive,
  unthreaded, unblocked scalar loop re-streaming its ~604 MB weight matrix
  from host RAM once per output row (Phase 8's flat ~21s/forward
  measurement, ~11% of the real per-step total) - **closed in Phase 9**
  (see below): row-parallelized, bit-identical, ~3.5x on the stage itself.
  Not fully closed: the fix is bandwidth-, not thread-count-, bound (48
  cores measured only ~3.5x, since every thread still re-walks the same
  604 MB matrix), so a blocked/tiled rewrite that avoids the redundant
  re-reads entirely remains a further, unattempted win.
- ~~The residency executor's GPU-lane device-opening path can fail to match
  the expected physical adapter by PCI id, falling back to a software
  adapter with too small a `max_storage_buffer_binding_size` for even the
  smallest real-VAE decode buffer (Phase 8's entry above).~~ **Closed** -
  root-caused and fixed in `backend-wgpu`/`vulkan`/`backend-vulkan`, which
  own the real cause (repeated Vulkan instance create/destroy makes the
  loader unload and reload the ICD until it stops resolving
  `vkCreateInstance`, after which the process sees no cards at all). It was
  never residency-specific: any code that opens a device per forward call
  hits it, which is why `brain ltxv t2v` reproduced it too once the run was
  long enough. See those crates' `shared_instance`/`instance` doc comments
  for the mechanism and `crates/gpu-core/tests/device_churn.rs` +
  `crates/backend-wgpu/tests/adapter_enumeration.rs` for the gates.
  `scripts/gates/ltxv-perf-gate.sh` still runs `--device cpu`, now by
  choice of measurement target rather than to sidestep a bug.
- `brain perf`'s `ltxv:` target measures only the tiny random-weight
  config by default; the real 22B checkpoint's per-step cost (Phase 8's
  entry above) makes it unsuitable for a routine gate, so no committed
  baseline exists yet for `dit_config=ltx25_22b` - a deliberate,
  separately-scheduled measurement whenever it is needed, not a default
  one.
- ~~Cold checkpoint I/O is the dominant remaining cost of a real
  generation and no code change removes it ... a cache of the ENCODED TEXT
  CONTEXT ... Not attempted: it is a cache-invalidation design decision,
  not an optimization.~~ **Both levers taken in Phase 11**, which also
  confirmed the diagnosis by measuring the text encode as 90% I/O. Fewer
  bytes: the encoder is quantized to Q8_0 and streamed (26.26 -> 14.09 GB
  on disk, and no whole-checkpoint fp32 expansion at all). Not re-reading
  them: `crate::text_cache`, on by default, with the invalidation question
  answered by verifying the stored key rather than trusting a digest. Cold
  I/O is still the dominant cost of what remains - the streamed encode's
  own 278 s is mostly reading layer weights - so this is a large dent, not
  a closure.
- ~~The real text-encode stage is invisible to `Timings` ... the
  `Timings` struct still does not carry it.~~ **Closed in Phase 11**:
  `Timings::text_encode` plus `Timings::unattributed(wall)`, printed as its
  own row by both `brain ltxv` CLI lines, so a stage nobody has
  instrumented yet shows up as a number instead of as a silent gap.
- `load_block_tensors_from_source` copies every streamed tensor a second
  time (`d.to_vec()` over a `Vec` the GGUF reader has just allocated) -
  ~1.8 GB of redundant allocation and memcpy per block, ~86 GB per real
  generation. Closing it needs a `checkpoint::TensorSource` seam that can
  transfer ownership rather than lend a slice; deliberately not added as a
  defaulted trait method in Phase 10 (lesson #30: a default trait method is
  a silent opt-out).
- The streamed int8 tier decodes Q8_0 to fp32 and immediately re-quantizes
  it to int8, materializing ~1.8 GB of fp32 per block to produce ~0.46 GB.
  Fusing the two per row would be bit-identical and would delete that
  traffic, but it restructures the load path into a streaming pipeline
  rather than changing a loop, so Phase 10 left it tracked.

## Scope that collapsed once the reference was read

- The HF filenames imply an "int8-convrot" quantization scheme; grepping the
  full source tarball found zero references to convolution-rotation/Hadamard
  quantization actually wired into any load path - see the INT8 note above.
- `modality_tiling.py` looked like it might be the audio/video mixing mechanism
  from its name; it is spatial/temporal tiling of a video-only token sequence
  for tiled inference, unrelated to the A<->V cross-attention that actually
  couples the streams.
