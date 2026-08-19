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

- Full 22B DiT real-weight parity: **partially closed** - the "real-weight
  parity ladder for the 22B DiT, reduced depth" milestone above proved
  quantization exactness (Q8_0, exact) and video-stream port correctness at
  REAL width but only 2 of 48 `transformer_blocks`, gated attention ON,
  `use_embeddings_connector: false`. Still NOT run, and still needing either
  more host RAM/time or the int8 compute path (Phase 5) to attempt cheaply:
  all 48 layers at once, the audio stream, the A<->V cross-attention
  (`LtxAvDit`), either embeddings connector's own real-weight parity, the
  Q4_K_M quant tier, and any int8/int4 COMPUTE-path comparison (no such
  kernel exists yet for the DiT - `crate::int8` is storage-format-only).
- 12B Gemma-4 real-weight parity: not run. The 26 GB bf16 checkpoint has
  since been fetched and verified byte-exact + structurally against the real
  header (`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`) - the "needs
  hardware that can hold the checkpoint" reason no longer applies.
  Genuinely unattempted, out of scope for the DiT real-weight milestone
  above (which did not touch `crates/gemma4`) - remains open for a
  dedicated pass.
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

## Scope that collapsed once the reference was read

- The HF filenames imply an "int8-convrot" quantization scheme; grepping the
  full source tarball found zero references to convolution-rotation/Hadamard
  quantization actually wired into any load path - see the INT8 note above.
- `modality_tiling.py` looked like it might be the audio/video mixing mechanism
  from its name; it is spatial/temporal tiling of a video-only token sequence
  for tiled inference, unrelated to the A<->V cross-attention that actually
  couples the streams.
