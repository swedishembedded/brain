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

### Phase 13 - the block-weight cache stops dying with the generation

Phase 9 built a host-side cache of each block's already-quantized bytes and
scoped it to ONE `generate()` call: `RealDit` owned it and dropped it when the
generation finished. That removed ~86% of every denoise step past the first.
Phase 10 then measured what it did NOT remove: the first step of a real run
cost 365.4 s of a 964.3 s run, because this box's storage reads the checkpoint
at ~58-70 MB/s cold and every generation re-read all ~22 GB of DiT blocks from
scratch. Two back-to-back generations seconds apart paid it twice, for bytes
that had not changed.

The cache's contents were never a property of a generation. `model::int8::
quantize_weight` is a pure function of the checkpoint's immutable bytes, so the
correct scope is the CHECKPOINT and the correct lifetime is however long a
memory ceiling allows - which is what this phase implements.

**The change** (`crates/ltxv/src/weightcache.rs`, new; `crates/ltxv/src/
block.rs`, `dit.rs`, `pipeline.rs`; `crates/cli/src/resident_ltxv.rs`):

* **Keyed on checkpoint identity**, not on a generation: path + byte length +
  mtime, which is exactly the identity `ltxv::text_cache::Key` already carries
  for the text encoder - reused rather than re-invented - plus the block index
  and the quant tier. Two generations against one file share every entry; a
  replaced or re-quantized file at the same path shares none.
* **Held in a process-wide registry**, so the store outlives the `RealDit`, the
  `generate()` call and the resident instance alike. `GenerationCache` is now a
  handle onto it; `GenerationCache::default()` still means "a private,
  unregistered store", which is what every existing test and `ltxv_bench` call
  site already wanted, so none of them changed meaning.
* **`Sync`, not `RefCell`.** An `RwLock` over the slot table with `Arc` per
  entry, so a reader takes no lock across its (multi-hundred-millisecond)
  device upload. The concurrent two-GPU CFG dispatch that needs this is a LATER
  phase and was deliberately not attempted here; the point is only that the
  cache no longer blocks it, which is pinned by a `Send + Sync` assertion and a
  multi-threaded hammer test rather than left as an intention.
* **Governed, not merely affordable.** The store runs under a byte budget
  derived from `memauth::limits().ram_total` - the process-wide ceiling the
  `--limit-vram-total`/`--limit-ram-total` milestone published - taking two
  thirds of it and leaving a third for everything else a generation holds in
  host RAM (head tensors, encoded context, VAE weights, pixel buffers). With no
  ceiling published the budget is `None`, i.e. exactly today's behaviour: a
  guessed default would change how every existing run behaves in order to
  govern something nobody asked to govern.
* **Evicted by `residency::place::CostAware`** - the same GDSF policy
  (`uses * bytes / age`) the residency manager scores whole model instances
  with, driven through the same `residency::lru::Entry`, rather than a second
  bespoke LRU that would have to be re-tuned separately. `brain-ltxv` gains a
  `brain-residency` dependency for this; `residency` depends only on
  `capability`/`memauth`, so it is not a cycle.
* **The connector half is now bounded** (4 entries, least-recently-used
  dropped). Unbounded was affordable when the cache died with the generation;
  at process lifetime every new prompt would have added a few megabytes that
  nothing ever removed.

**`LtxvResident` becomes the first production implementor of `demote`/
`promote`.** It holds a handle onto the same store the pipeline resolves by
path from inside `generate()`, so `demote` releases memory the pipeline is
really using rather than a private copy. Two honesty notes recorded rather than
glossed:

1. `Instance::demote`'s contract says `Warm` releases DEVICE buffers and keeps
   host bytes. LTX has no device buffers to release between calls -
   `forward_q_streamed` opens a fresh `Gpu` per forward and drops everything
   before returning, a design its own doc records as deliberate and measured.
   An LTX instance's entire reclaimable footprint IS host RAM, so a `demote`
   that released "device buffers" would release nothing while the manager
   charged a Warm cost and believed it had made progress. `demote` therefore
   releases the block cache and `estimate_at` reports the honest post-demote
   number. This is safe for a reason no other model can claim: the entries are
   a pure function of immutable checkpoint bytes, so dropping one costs time
   and nothing else. `promote` is a lazy no-op - re-filling ~13 GB eagerly
   would block the manager's worker thread for minutes to do work the request
   itself does incrementally.
2. `estimate`'s HOST figure was wrong before this phase and is now real. It
   charged `manifest_bytes(dit_tensor_manifest(cfg))` - the fp32 manifest size,
   ~62 GB at the 22B config - for a checkpoint this path never materializes at
   all. It now charges what a real run holds: the head tensors, ONE block's
   transient fp32 expansion, and the block cache, the last from
   `block::cached_block_bytes` - a closed form over what `quantize_host` really
   builds, pinned against a really-quantized block at both tiers and both gate
   settings, not a `file_size * 1.3` guess. The VRAM figure keeps the
   pre-existing conservative manifest number: a streamed forward's peak VRAM is
   dominated by activation buffers that follow the latent token count, and
   deriving that honestly is its own piece of work (tracked below).

**Correctness gates** (`crates/ltxv/tests/block_weight_cache.rs`, extending the
4 tests Phase 9 left, now 8; plus 5 in `weightcache`'s own module and 7 in
`resident_ltxv`):

- `a_second_generation_reuses_the_first_generations_entries_bit_identically` -
  generation A populates a store through a handle it then drops; generation B,
  with a DIFFERENT prompt/latent, resolves a fresh handle from the path alone
  and must record `num_layers` hits and ZERO misses, with output bit-identical
  (`max_abs == 0.0`) to a cache-free forward. A different checkpoint identity
  must start empty.
- `eviction_under_a_tight_ram_ceiling_repopulates_correctly` - the ceiling is
  sized off a REALLY measured block (two blocks' worth against a four-layer
  model, so eviction is forced and provably partial), eviction is asserted to
  have happened, the cache is asserted to stay under budget, and then the
  claim that matters: a SECOND forward, which necessarily misses on the evicted
  layers, re-reads and re-quantizes them and is still bit-identical. A block
  larger than the whole budget is not retained and the forward is still exact.
- `cached_block_bytes_matches_a_real_measured_block` - the closed-form
  footprint must equal a really-quantized block's own `byte_len()`, so a change
  to what the cache stores cannot silently make every residency estimate wrong.
- `demote_releases_the_shared_block_cache_and_promote_returns_to_hot` - demote
  clears the store the PIPELINE reads (not a private copy); a model with no
  real checkpoint refuses both rather than claiming progress it cannot make.
- `the_cache_is_send_and_sync_so_concurrent_cfg_dispatch_is_not_blocked` and a
  multi-threaded accounting test.

Every pre-existing gate re-run and green: the FULL `cargo test -p brain-ltxv
--tests` (124 lib + all 27 integration binaries, 0 failures), `-p
brain-residency` (80), `-p brain-cli` (151+ across its binaries), including
`dit_parity`, `av_dit_parity`, `host_forward_parity`, `streamed_vs_eager_real`,
`connector_real_parity` and the real-weight
`real_checkpoint_cached_forward_is_bit_identical_to_an_uncached_one`.

**Measured on the REAL resident path** - not the one-shot `brain ltxv t2v` CLI,
which by design holds nothing. Nothing in this workspace drove `LtxvResident`
end to end with real weights before, so the harness is new and permanent
(`resident_ltxv.rs::two_real_generations_share_one_warm_checkpoint_cache`,
`#[ignore]`d):

    BRAIN_LTXV_VAE=<ltx-2.5-video-vae-conv-bf16.safetensors> \
    BRAIN_LTXV_DIT=<ltx-2.5-22b-distilled-transformer-Q8_0.gguf> \
    BRAIN_LTXV_TEXT_ENCODER=<gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf> \
      cargo test --release -p brain-cli --bins -- --ignored --nocapture \
      two_real_generations_share

Two generations, DIFFERENT prompts, 9 frames at 64x64, 8 distilled-schedule
steps, `guidance<=1.0` (1 forward/step), one Tesla P40 idle before the run,
through ONE resident instance:

| | wall | block-cache misses | hits | held |
|---|---:|---:|---:|---:|
| generation 1 (cold cache) | 245.3 s | 48 (every block) | 336 | 48 blocks, 12.96 GB |
| generation 2 (warm, new prompt) | 179.2 s | **0** | 384 | unchanged |

The zero is the load-bearing number: generation 2 hit the cache on its very
first layer of its very first step, which is precisely what a per-generation
cache could never do. 384 = 48 layers x 8 steps, so every access was a hit.
The measured per-block footprint is 270.1 MB at the real 22B width (x48 =
12.96 GB), which the closed-form estimator reproduces exactly.

**The wall-clock delta understates the win, and is reported that way on
purpose.** The checkpoint was already page-cache-warm from earlier test runs in
the same session, so generation 1's 48 misses cost ~58 s of warm read+quantize
rather than the ~250 s of cold disk Phase 10 measured with a `dd`-verified
~58-70 MB/s. On a genuinely cold checkpoint the second generation saves that
larger figure instead. What does NOT depend on page-cache state is the miss
count, which is why the gate asserts on it.

**Not attempted, deliberately:**

* **Concurrent CFG dispatch across the two cards.** A later phase. This phase's
  only obligation to it was not to leave a `!Sync` type in the way, discharged
  above.
* **`crates/weightset`'s `CyclicScan`** - checked, and declined with a reason.
  Its Belady-style planner is the right planner for a fixed WINDOW OF DEVICE
  SLOTS over equally-shaped weight groups with a known visitation schedule, and
  a denoise loop's `Schedule::cyclic(n_blocks, steps)` is exactly that
  schedule. But this cache is host-side, its entries are variable-sized
  already-quantized blobs rather than uniform device slots, and the constraint
  governing it is a BYTE budget shared with every other model in the process -
  which is what `residency`'s own `EvictionPolicy` exists for. Using two
  eviction rules in one workspace to save one indirection was the worse trade.
  `weightset` therefore still has zero production consumers.
* **A cache surviving across separate processes** (an on-disk pre-quantized
  block store). Phase 9 tracked it; it is still open. The text encoder's own
  output cache already does this at a different level.
* **The VRAM half of `LtxvResident::estimate`** - see honesty note 2 above.

### Phase 14 - the adaLN-single host stage, and a correction to Phase 12's attribution

Phase 12 closed by naming the next bottleneck: `stage forward_q_streamed:
adaLN-single table (host): 76281.3 ms`, ~81% of the 93.86 s wall for 8 layers
at the real 720p token count, and attributed it to `dit.rs`'s host `linear`
behind `ada_layer_norm_single` - the `[3520,4096] x [36864,4096]^T` GEMM.

**That attribution was wrong, and finding out cost one measurement.** Before
changing the GEMM, its own shape was benchmarked in isolation
(`backend-cpu/tests/host_gemm.rs::tile_sweep_at_the_real_adaln_shape`, the
exact `[3520,4096] x [36864,4096]^T` operands): the naive row-parallel loop
runs it in **14.39 s at 73.9 GFLOP/s** - not 76 s. Something else in the same
timing bracket was five times bigger, and instrumenting the two halves
separately named it immediately: `ada_layer_norm_single` called
`dit::timestep::pixart_timestep_embed` **once per row, in a serial loop**, and
that helper's two linears (`dit::timestep::linear1`) are single-row matvecs on
ONE core. At T=3520 that is `3520 x (256x4096 + 4096x4096)` = 6.3e10
multiply-adds on one core of 48. Measured: **60709.1 ms**, i.e. 80% of the
stage, against the GEMM's 15137.8 ms.

This is the same lesson §F.2/§F.9 already carry, arriving through a different
door: the profiler named a STAGE, a plausible story was told about which line
inside it dominated, and the story was wrong by 4x. The cheap check - time the
suspected line on its own before optimizing it - is what turned a 1.7x fix into
a 7.5x one.

**Two exact wins, both bit-identical, in the order the measurement dictated.**

**Win 1 - batch the timestep embedder** (`crates/ltxv/src/dit.rs::
ada_layer_norm_single`). Every row is independent and they all read the SAME
two weight matrices, so `rows` serial single-core matvecs are two ordinary
`[rows,in] x [out,in]^T` GEMMs with an elementwise SiLU between them. Only the
`[rows,256]` sinusoid table is genuinely per-row, and that is row-parallel.
Bit-identical because `dit::timestep::linear1` accumulates `bias` then
`+= x*w` over ascending `k`, one f32 add at a time, and so does `linear`:
every output element is the same sequence of the same roundings.
`dit::timestep::pixart_timestep_embed` itself is UNTOUCHED - the batching is
local to this call site, so `wan`/`s3dit`/`flux` keep their exact current
behaviour.

**Win 2 - a blocked host GEMM** (`crates/backend-cpu/src/host_gemm.rs`, new).
Per kernels.md §F.3 the tree was checked first: `backend_cpu::fast_ops::
matmul_abt` is the AVX2 kernel for this exact `A @ Bt` shape and is several
times faster still, but it splits `k` across eight lanes and uses FMA, so it is
NOT bit-identical to the loop this path is gated against and cannot be dropped
in. `wan::model::linear` is the same naive loop. So this is new code, hoisted
to `backend-cpu` rather than kept in `ltxv` because the identical shape recurs
(`wan::model::linear`, `ltxv::dit::linear`) - additive, with the naive loop
kept beside it as the reference definition of the arithmetic.

It fixes two defects in one nest. The memory one Phase 9 recorded and declined:
the naive order reads the ENTIRE 604 MB weight matrix once per OUTPUT ROW, and
row-parallelizing hid that behind aggregate bandwidth instead of removing it -
~2.1 TB of DRAM traffic for 5.3e11 MACs. A register block of 8 output rows
against one weight row cuts the re-reads eightfold. The quieter one: `acc +=
x*w` on a single accumulator is a loop-carried dependency on a ~4-cycle f32
add, so a core retires ~1 MAC per 4 cycles no matter what the bandwidth is;
8 independent accumulator chains fill that latency.

**Bit-identity is structural, not a tolerance**: every output element is still
`bias` then `+= x[m,k]*w[n,k]` for ascending `k`, one f32 add at a time.
Nothing is reassociated, no accumulator is split and recombined, no FMA
replaces a separate multiply and add, no SIMD lane sums a partial range of `k`.
The blocking changes only which element is computed when, which IEEE-754 does
not observe.

**The tile sweep** (measured, not guessed - `tile_sweep_at_the_real_adaln_shape`,
48-thread Xeon E5-2690 v3, real shape), and it pointed the opposite way from
the obvious intuition:

| m-tile | secs | GFLOP/s | vs naive |
|---|---:|---:|---:|
| naive (row-parallel) | 14.39 | 73.9 | 1.00x |
| **8** | **8.32** | **127.8** | **1.73x** |
| 16 | 8.83 | 120.4 | 1.63x |
| 32 | 9.32 | 114.1 | 1.54x |
| 64 | 11.44 | 92.9 | 1.26x |
| 128 | 11.79 | 90.2 | 1.22x |
| 256 | 16.22 | 65.5 | 0.89x |

Monotone rather than U-shaped because at tile 8 the kernel is already
ARITHMETIC-bound: 127.8 GFLOP/s is ~1 scalar MAC per core-cycle, the ceiling
for a non-reassociating f32 multiply-then-add. Extra weight reuse buys nothing
past that, while the tile's slice of `x` (`tile * K * 4` bytes - 512 KB already
at tile 8 with K=4096) grows past this core's 256 KB L2 and starts costing.
Tile 256 is SLOWER than the loop it replaces.

**Gates.** `backend-cpu/tests/host_gemm.rs`: bit-pattern comparison (not a
tolerance, and not `assert_eq!` on f32, which would call two NaNs unequal)
against the naive loop over 8 shapes x 11 tile sizes x with/without bias,
covering tiles that do not divide `rows`, tails shorter than the register
block, single-row inputs (the `coeff=1` AV gate tables) and `in_dim = 0`.
Mutation-verified on the bug class blocking actually invites: a deliberately
reassociated (even/odd split-accumulator) reduction must differ in bit pattern
on most elements, and the blocked kernel must be on the reference's side of
that line. `ltxv::dit::tests::batched_adaln_timestep_embedding_is_bit_identical
_to_the_per_row_form` does the same for win 1 against the exact per-row loop it
replaces, at four `(rows, dim, coeff)` shapes including `rows=1`.

Every pre-existing gate re-run and green: the FULL `cargo test -p brain-ltxv
--tests`, `-p brain-backend-cpu`, `-p brain-residency`, `-p brain-cli`,
including `dit_parity`, `av_dit_parity`, `host_forward_parity`, `block_grad`,
`av_block_grad`, `streamed_vs_eager_real` and the real-weight parity tests.

**Measured**, same command both arms, same box, same session, one Tesla P40
idle before each run, real `ltx-2.5-22b-distilled-transformer-Q8_0.gguf`:

    BRAIN_PROFILE=1 BRAIN_LTXV_DIT=<real Q8_0 22B> \
      ./target/release/ltxv_bench streamed 8 3520 1024 1

Call 2 (the cache-hit call - the shape every step of a generation past the
first has):

| | before | after | |
|---|---:|---:|---:|
| adaLN timestep embedder (host) | 60709.1 ms | 1258.3 ms | **48.2x** |
| adaLN table GEMM (host) | 15137.8 ms | 8777.0 ms | **1.72x** |
| **adaLN-single stage total** | **75846.9 ms** | **10039.6 ms** | **7.6x** |
| wall, 8 layers, cache hit | 93.98 s (11.75 s/layer) | 28.98 s (3.62 s/layer) | **3.24x** |
| wall, 8 layers, cache miss | 112.30 s | 42.43 s | 2.65x |

Reproducible: a second after-run measured 10229.9 ms / 28.85 s. The before arm
reproduces Phase 12's own published 76281.3 ms / 93.86 s to within 1%, which is
what makes the two phases' numbers comparable.

**The strongest evidence that this changed no number is in the bench's own
output.** `ltxv_bench` prints the forward's output statistics, and across both
arms at the real 22B int8 width they are identical to every digit printed:

    len=450560 mean=0.192827 std=2.603959 min=-7.937798 max=7.273794 nonfinite=0

The adaLN stage is now 35% of a cache-hit call rather than 81%, and the split
of a real step has changed shape: GPU upload+forward+wait (17.9 s) is now the
largest single item.

**Not attempted, and why:**

* **Vectorizing the blocked GEMM.** The bit-identical way to do it is to
  vectorize ACROSS `M` - eight output rows in one AVX2 lane group, each lane
  still summing its own `k` sequentially, with a separate `_mm256_mul_ps` and
  `_mm256_add_ps` rather than an FMA (which rounds once instead of twice). That
  would lift the ~1 MAC/core-cycle scalar ceiling by ~8x and take the table
  GEMM from 8.8 s toward ~1 s. It needs a transposed pack of `x` and is a
  larger change than this pass's remit; the memory win was taken first and the
  arithmetic left scalar. Recorded as a real, available, exact win.
* **Moving the adaLN table to the GPU.** Phase 12 suggested it ("a 10.5
  TFLOP/s card sits idle"). Still open, and now a smaller prize than it looked:
  the stage is 10.0 s rather than 76.3 s, and the card is not idle during a
  cache-hit call - it is the largest item.
* **Switching `wan::model::linear` to the blocked kernel.** `crates/wan` is
  out of this pass's scope. The kernel is in `backend-cpu` precisely so that
  adoption is a one-line change when someone measures Wan's own shapes.
* **Deduplicating identical timesteps.** A real denoise step passes the same
  sigma for nearly every one of the 3520 tokens (only frozen conditioning
  tokens differ), so the embedder could compute 1-2 distinct rows instead of
  3520 and broadcast - bit-identical, and another ~50x on top of win 1. Not
  taken: it makes the cost input-dependent in a way the batched GEMM does not,
  and win 1 already reduced this stage to 1.3 s. Tracked.

### Phase 15 - the second card starts existing

Phases 12-14 made ONE card faster. This phase makes the second one exist. Every
stage of a generation was single-device - the Gemma-4 encode, the denoise loop,
the VAE decode - and inside the loop the conditional and unconditional
classifier-free-guidance forwards ran one after the other on that same card. On
the two-P40 box this port is developed on, that is a 24 GB card at 0.0%
utilization for the entire run, which is not an inference from the code: it is
what `nvidia-smi` recorded, below.

Three pieces, in the order a measurement dictated - including one piece that the
measurement cancelled.

#### 0 - the measurement that cancelled the expensive piece

The original plan carried a large speculative item: a GGUF-streaming int8 SHARD
loader, splitting the real 22B DiT's 48 blocks across two cards, because the
1080p token count might not fit one 24 GB board. Phase 12's flash-attention
number for 1080p (~16.6 GiB) was measured on the self-attention kernel ALONE in
an isolated bench, not on a real forward, so it could not settle the question.

Measured, real `ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, all 48 real
layers, int8 compute, T=8160 (`lat_t=4, lh=34, lw=60` - 1080p), one Tesla P40,
sampled at 200 ms:

    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits --loop-ms=200 &
    BRAIN_GPU_INDEX=0 BRAIN_LTXV_DIT=<real Q8_0 22B> \
      ./target/release/ltxv_bench streamed 48 8160 1024 1

| | wall | peak VRAM on gpu0 |
|---|---:|---:|
| call 1 (cache miss, all 48 blocks) | 329.9 s | **16 650 MiB = 16.26 GiB** |
| call 2 (cache hit, all 48 blocks) | 246.7 s | same |

16.26 of 24.00 GiB, i.e. **7.7 GiB of headroom on ONE card at the highest
resolution this pipeline targets**. Both calls printed identical output
statistics (`len=1044480 mean=0.043124 std=0.698590 min=-1.510173
max=1.889706 nonfinite=0`), which is the cache's own exactness re-confirmed at
a token count Phase 13 never ran.

So the shard loader is not needed, and item 3 below is declined. But the same
sweep found something that IS a real 1080p blocker, and it is not the DiT:

| stage | shape | peak VRAM | fits a 24 GiB P40? |
|---|---|---:|---|
| DiT, 48 layers, int8 streamed | T=8160 (1080p) | 16 650 MiB | yes |
| VAE decode (`ltxv_bench vae`) | 25 frames @ 1920x1088 | **OOM** (>21 718 MiB observed before the abort) | **no** |
| VAE decode | 9 frames @ 1920x1088 | 15 186 MiB | yes |
| VAE decode | 25 frames @ 1280x704 (720p) | 16 564 MiB | yes |
| VAE decode | 49 frames @ 1280x704 | **OOM** (>23 930 MiB) | **no** |

The binding constraint at 1080p is the **conv VAE decoder's un-tiled activation
buffers**, not the transformer. That matters for what to build next: sharding
the DiT across two cards would not have moved this number at all. The fix is
the reference's own overlapping-tile chunked decode, which this ledger already
tracks as an open gap ("general overlapping-tile chunked encode/decode remain
out of scope, deferred to the DFR milestone"). Until it lands, 1080p is capped
at 9 pixel frames and 720p at 25 - stated here so a benchmark pass does not
discover it as a crash.

#### 1 - concurrent CFG dispatch, one branch per card

When `guidance > 1.0` every denoise step runs two DiT forwards at the SAME
latent: one against the prompt's context, one against the empty prompt's. They
share no intermediate value; the only thing that reads both is the host-side
fold `uncond + guidance·(cond - uncond)` after both return. Two independent
forwards is exactly the shape two cards want, and it needs no weight sharding
whatsoever.

**The change** (`crates/ltxv/src/devplan.rs`, new; `pipeline.rs`; `caps.rs`;
`crates/cli/src/resident_ltxv.rs`):

* **`DevicePlan`** names three placements - `text`, `cond`, `uncond` - and
  resolves against `gpu_core::devices::ambient_compute_set()`, the same
  `--device`/`BRAIN_DEVICE` resolution every other placement decision in this
  workspace goes through. Reading `gpus()` directly would have let a
  `--device gpu0` run schedule onto a card the operator excluded. Fewer than
  two schedulable cards, or the CPU backend, resolves to `Single` - byte for
  byte the old behaviour, no threads spawned.
* **The base card is the CURRENT selection**, not a hardcoded 0, so a
  generation running inside the residency executor's `with_gpu`-scoped lane
  keeps its assigned card as `cond` and borrows only the other one.
* **`Denoiser::forward_cfg_pair`** is where a denoiser states whether its two
  branches can be placed independently. The default is the sequential pair
  every call site already had, which is the only correct answer for `LtxDit`:
  it holds ONE `Gpu` built at construction, so dispatching its forward from a
  differently-scoped thread would not move a single byte. `RealDit` overrides
  it, because `forward_q_streamed` opens a fresh `Gpu` INSIDE every call - the
  property, already documented and measured in Phase 13, that makes scoping
  the call enough to move the whole forward.
* **`StepInputs`** groups the seven per-step arguments both branches share, so
  "run this pair, wherever" is one call rather than a nine-argument closure
  written twice. `Denoiser::forward` lost six parameters in the process.
* **The text encoder is pinned to the card the conditional forward will not
  use.** It finishes before denoising starts, so this is not about overlap: it
  is about not leaving the 12B encoder's device footprint on the card that is
  about to hold the denoise loop's activations.

**Why nothing else in `RealDit` needed a lock.** The task of checking this was
explicit, and the answer is per-field rather than "the cache is `Sync`, so we
are fine": `src` is a `MmapGguf` (an immutable mapping), `head` is an owned
tensor map never written after construction, `cfg`/`device`/`place` are `Copy`
data, and `cache` is Phase 13's `RwLock`-over-slots store that hands out `Arc`s
and holds no lock across a device upload. The connector half of that cache is
read by both branches and they genuinely differ there (two different contexts,
so two of the store's four connector slots) - `MAX_CONNECTOR_ENTRIES = 4` is
exactly why neither branch evicts the other's.

**Gate: bit-identical, not "close".** Two gates, because the tiny one runs in
milliseconds on every `cargo test` and the real one costs eight minutes:

- `pipeline::tests::the_concurrent_cfg_pair_is_bit_identical_to_the_sequential_
  one` drives the REAL dispatch function (`dispatch_cfg_pair`, extracted as a
  free function precisely so the gate cannot test a copy of it) with a
  `PerCallDeviceDit` - a tiny-config denoiser that builds a fresh `LtxDit`, and
  therefore a fresh `Gpu`, inside every forward. Real kernels on real cards, at
  a config that needs no fixture. Compared on BIT PATTERNS, not `assert_eq!` on
  `f32` (which calls two NaNs unequal), and it additionally asserts the two
  branches differ from each other - a gate where `cond == uncond` would pass
  even for a dispatch that ran the conditional forward twice.
- `pipeline::tests::the_cfg_step_routes_through_the_pair_method` counts pair
  dispatches, so a refactor that quietly went back to two bare `forward` calls
  fails instead of silently making the placement dead code.
- `crates/ltxv/tests/cfg_parallel.rs` (`#[ignore]`d), the real-weight half: two
  full 22B generations, same seed/prompt/shape, differing only in the plan,
  compared byte for byte over every decoded frame - plus an assertion that the
  clip is not frozen, since any two runs of a frozen generator agree.

**Measured**, real Q8_0 22B checkpoint, 9 frames at 64x64, `guidance = 5.0`
(so 2 forwards x 8 distilled steps = 16 forwards), deterministic sampler, two
Tesla P40s, `nvidia-smi --query-gpu=index,utilization.gpu,memory.used
--loop-ms=200` throughout:

| arm | wall | denoise | gpu0 busy | **gpu1 busy** | both at once | peak MiB gpu0/gpu1 |
|---|---:|---:|---:|---:|---:|---:|
| warm-up, cold cache (discarded) | 229.3 s | 218.9 s | 54.7% | **0.0%** | 0.0% | 12844 / 1 |
| sequential, one card | 146.6 s | 135.2 s | 80.6% | **0.0%** | 0.0% | 2277 / 1 |
| concurrent, two cards | **75.6 s** | **63.8 s** | 77.2% | **56.5%** | **55.7%** | 2493 / 663 |

**1.94x wall, 2.12x on the denoise loop itself**, and bit-identical output. The
second card goes from a measured 0.0% - not "low", zero, in every one of 733
samples - to busy in 56.5% of samples, with both cards busy simultaneously in
55.7% of them.

**The warm-up row is why this phase re-ran its own gate.** The first version
timed the sequential arm against the concurrent one with no warm-up and reported
**4.0x**. That number was mostly Phase 13's cache: the first generation against
a checkpoint pays a ~230 s cold read that has nothing to do with placement. The
honest figure is half of it. This is the §F.2/F.9 lesson again - the first
measurement of a change is usually measuring something else - so the warm-up is
now part of the gate rather than a discipline someone has to remember.

#### 2 - concurrent admission of independent generations

`LtxvInstance::run_batch` was a serial `.map()`, and its doc comment said so:
nothing in this pipeline batches a denoise loop across prompts, because each
request has its own latent, its own schedule and its own step count. That
reasoning is right about BATCHING and wrong about throughput - N independent
generations do not need to be one graph, they need to be on N cards.

`residency::executor` already runs per-device lanes, so two models on two cards
overlap today. What it cannot do is spread ONE model's batch, because same-key
jobs group onto one lane by design. The missing piece is not a second
scheduler; it is a way for a `run_batch` implementation to say "these are
independent, spread them".

**`residency::devpool::DevicePool`** (new, 5 unit tests, no GPU code in it at
all - `brain-residency` still depends only on `capability` + `memauth`):

* `run_all(n, job, events)` runs `job(index, device)` for `0..n` across the
  pool, **at most one at a time per device**, and returns results in REQUEST
  order regardless of completion order (gated - a pool returning completion
  order would pair request 0's answer with request 3's caller).
* Work is claimed by an `AtomicUsize` cursor rather than pre-partitioned, so a
  card that finishes early takes the next waiting request instead of idling
  behind a slow neighbour. Per-request cost really does vary here (a longer
  clip, a colder cache).
* **Progress is delivered over an mpsc channel and replayed on the CALLING
  thread**, so a caller's `&mut dyn FnMut(usize, Progress)` sink needs neither
  a lock nor a `Send` bound - the classic reason to move progress over a
  channel instead of sharing the sink, and the same `thread::scope` + channel
  idiom `model::shard::Pipeline::pipelined_fwd_bwd` established.
* **One request per device is a memory fact, not a tuning knob**: item 0
  measured a real DiT forward at 16.26 GiB and a 720p VAE decode at 16.18 GiB
  on a 24 GiB board. Two on one card is a hard `wgpu` out-of-memory abort, not
  a slowdown. The ceiling is gated by a live counter (`at_most_one_request_per_
  device_is_ever_in_flight`), not asserted from the code's shape.
* A one-request batch or a one-device pool runs INLINE - no threads, no
  channel, no reordering, and each event delivered before the next request
  starts, gated separately. The overwhelmingly common shape must not be made
  slower or less debuggable by a pool it does not need.

**`resident_ltxv.rs`** builds its pool from `ambient_compute_set()` with the
residency-assigned card FIRST (so a batch of one runs exactly where residency
placed it), and gives each concurrent request `DevicePlan::Single` on its own
card - item 1's two-card CFG split would have every request reaching for both
cards. `Instance::run`'s body moved to a `run_on(paths, device, plan, ...)`
that takes only shared references, which is what makes running several of them
at once safe rather than merely convenient.

The Tier-1 sharing is by construction and needed no new code: the block cache
is keyed on the CHECKPOINT (Phase 13), so N concurrent generations against one
file pay the cold read once between them. Each card still uploads to its own
device from the same `Arc<CachedQBlockWeights>` host bytes.

**Measured**, real 22B Q8_0 checkpoint through the REAL resident path
(`resident_ltxv.rs::concurrent_generations_share_one_cache_and_overlap_across_
the_cards`, `#[ignore]`d), four different prompts, 9 frames at 64x64,
`guidance = 1.0`, two Tesla P40s, warm cache on every timed arm:

    BRAIN_LTXV_VAE=<...> BRAIN_LTXV_DIT=<real Q8_0 22B> \
      cargo test --release -p brain-cli --bins -- --ignored --nocapture \
      concurrent_generations_share

| | wall | vs serial | throughput | gpu0 busy | **gpu1 busy** | both at once | peak MiB gpu0/gpu1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 request (baseline) | 82.6 s | - | 1.00x | 72.8% | **0.0%** | 0.0% | 2277 / 1 |
| **2 concurrent** | **78.0 s** | 165.2 s | **2.12x** | 75.1% | **54.6%** | 49.0% | 2277 / 12839 |
| **4 concurrent** | **156.9 s** | 330.3 s | **2.11x** | 75.1% | **57.1%** | 45.7% | 12844 / 12841 |

Two concurrent generations cost 0.95x what ONE costs - the second is
effectively free, which is what "the second card was doing nothing" means
numerically. Four cost 1.90x of one rather than 4x, exactly the 2-wide
ceiling: requests 3 and 4 wait for a card, deliberately, because admitting
them would abort. Every request produced a non-empty clip and the block-cache
miss count did not move by one across either arm - the assertion that N
concurrent requests share the warm checkpoint rather than each re-reading
23.6 GB of it.

**Not attempted, deliberately:**

* **A second admission policy.** `residency::admission`'s edge concurrency
  ceiling and admit deadlines are unchanged and still the front door; this
  adds capacity BEHIND them, where a batch has already been admitted onto a
  lane. Writing a second policy to shed on would have given the workspace two
  answers to "is the server full".
* **Sizing the pool from live VRAM rather than one-per-card.** The honest
  input for that is a per-request VRAM estimate that follows the token count,
  and `LtxvResident::estimate`'s VRAM half is still the conservative manifest
  figure Phase 13 recorded as an open gap. One-per-card is correct for every
  shape measured here and never over-admits; refining it needs that gap closed
  first.
* **Cross-process sharing of the block cache.** Still open, still tracked from
  Phase 9.

#### 3 - the GGUF-streaming int8 DiT shard loader: declined, with the number

**Not built, and item 0 is the reason.** A real 48-layer forward at the 1080p
token count peaks at 16.26 GiB on a 24 GiB card - the thing a shard loader
would split fits, with 7.7 GiB to spare, and it fits *because* Phase 12's flash
attention removed the `[heads,T,T]` score matrix that used to make it not fit.
Building a streaming int8 shard loader for the real 22B weights would have been
a large piece of engineering aimed at a constraint that no longer exists.

The MECHANISM is not in doubt and does not need re-proving:
`crates/ltxv/tests/av_shard_2gpu_real.rs` already runs a real two-GPU sharded
forward with real cross-device residual handoff at the synthetic `tiny_gated`
config, and `qwen3omnimoe`'s int8 layer-sharded Thinker does it against a real
30B checkpoint. What was missing was a REASON, and the measurement says there
isn't one at any resolution this pipeline supports.

If a future checkpoint or a longer clip does exceed one card, the number to
re-measure first is the one above, with the same command.

**Also not attempted, deliberately:**

* **Tiling the VAE decode.** Item 0 shows it, not the DiT, is what fails at
  1080p/25 frames. It is a real, now-quantified piece of work, and it is a
  different piece of work from this phase's (the reference's overlapping-tile
  decode with trapezoidal blend masks, already tracked). Doing it here would
  have meant shipping two unrelated changes as one.
* **Overlapping the text encode with the denoise loop.** It cannot be: the
  denoise loop's first forward needs the encoded context. Pinning the encoder
  to the other card is all the placement freedom that stage has.
* **Splitting a SINGLE forward across two cards** (tensor or pipeline
  parallel). That is the shard loader, declined above, and it would also
  introduce a cross-device reduction - the one thing that would make
  bit-identity a real question instead of a trivially satisfied one.
* **Three or more concurrent CFG branches.** There are exactly two.

#### What a benchmark pass should know before it runs

* **1080p is capped at 9 pixel frames** and **720p at 25**, by the VAE decoder,
  not the DiT - see item 0's table. `ltxv_bench vae 1 49 704 1280` and
  `ltxv_bench vae 1 25 1088 1920` both abort with a `wgpu` out-of-memory.
* **`--limit-vram-total` is a process-wide TOTAL across all cards**, not a
  per-card ceiling, and a concurrent pair charges it twice. A run sized for one
  card (`--limit-vram-total 20G` on a two-card box) will now be refused where
  it previously serialized. Either raise it or set `BRAIN_LTXV_CFG_PARALLEL=0`.
* **The first generation against a checkpoint is not comparable to the second.**
  Every number above is a warm-cache number and says so; a benchmark that times
  arm A cold against arm B warm will report roughly double the truth, which is
  the mistake this phase made once and now gates against.
* **`--start-frame` is unaffected by any of this.** Image conditioning happens
  before the denoise loop and is device-agnostic; the concurrent path carries
  the same `Frozen` mask and the same per-token timesteps, and the bit-identity
  gate runs the full `generate()` including that path.
* **Two concurrent generations with the SAME prompt** can both miss the
  on-disk text-context cache and both write the same entry. `text_cache::load`
  validates the stored key and the shapes, so a torn write reads back as a MISS
  and never as a wrong context - the cost is a redundant encode, not a wrong
  clip. Unusual enough not to be worth a lock file; recorded rather than left
  to be discovered.

#### Gates, all green

New: 4 in `ltxv::devplan`, 2 in `ltxv::pipeline` (the bit-identity gate and the
routing gate), 5 in `residency::devpool`, 1 in `resident_ltxv` (the pool's
shape), plus two `#[ignore]`d real-weight harnesses
(`ltxv/tests/cfg_parallel.rs`, `resident_ltxv::concurrent_generations_share_
one_cache_and_overlap_across_the_cards`) that are permanent, not one-off
scripts.

Every pre-existing gate re-run and green across `-p brain-ltxv -p
brain-residency -p brain-cli --lib --bins --tests`, including `dit_parity`,
`av_dit_parity`, `host_forward_parity`, `streamed_vs_eager_real`,
`connector_real_parity`, `block_weight_cache` and `av_shard_2gpu_real`.

### Phase 16 - the VAE decoder stops being the 1080p ceiling

Phase 15's item 0 measured that the thing blocking a full 1080p clip is not
the DiT (16.26 of 24 GiB, comfortable) but the conv VAE decoder's un-tiled
activation buffers, and deliberately did not fix it, so as not to ship two
unrelated changes as one. This phase is that fix: the reference's own
overlapping-tile decode with trapezoidal blend masks, ported and measured.

#### 0 - which axis actually drives the OOM (it is neither)

Phase 15's table read as though frame count and resolution were separate
constraints ("1080p is capped at 9 pixel frames and 720p at 25"). Measured
properly, they are one constraint: peak decode VRAM tracks the **output pixel
volume `frames x H x W`** and is close to blind to how that volume splits
between the axes.

Real `ltx-2.5-video-vae-conv-bf16.safetensors`, whole (un-tiled)
`LtxVaeDecoder`, one Tesla P40 (24576 MiB), wgpu backend, peak sampled at
200 ms:

    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits --loop-ms=200 &
    BRAIN_GPU_INDEX=0 BRAIN_GPU_WAIT_S=900 \
      BRAIN_LTXV_VAE=<real conv VAE> ./target/release/ltxv_bench vae 1 <frames> <height> <width>

| shape | Mpx | peak MiB | wall |
|---|---:|---:|---:|
| 9f @1920x1088 | 18.80 | 15186 | 65.0 s |
| 25f @1280x704 | 22.53 | 16564 | 71.6 s |
| 25f @1408x768 | 27.03 | 19089 | 100.0 s |
| 33f @1280x704 | 29.74 | 20494 | 107.5 s |
| 9f @2560x1440 | 33.18 | 23785 | 131.8 s |
| **17f @1920x1088** | **35.51** | **24264** | 128.8 s |
| **33f @1408x768** | **35.68** | **23813** | 124.3 s |
| 49f @1280x704 | 44.24 | **wgpu Out of Memory** (23348 at abort) | - |
| 25f @1920x1088 | 52.22 | **wgpu Out of Memory** (21719 at abort) | - |

The three bolded-adjacent rows are the experiment: 33.18, 35.51 and 35.68 Mpx
split as `9 x 3.69M`, `17 x 2.09M` and `33 x 1.08M` - a 3.7x spread in frame
count at the same volume - and they land at 23785 / 24264 / 23813 MiB, within
500 MiB of each other. The fit is roughly **3.9 GiB + 598 MiB per Mpx**, so a
24 GiB card runs out just past **~35 Mpx**. Tiling either axis works; what
matters is the product.

**A correction to Phase 15's own table.** Phase 15 recorded 49f @1280x704 and
25f @1920x1088 as OOM, and that is right, but it also implied the mid shapes
were unreachable. They were not: `25f @1408x768`, `33f @1280x704`,
`9f @2560x1440` and `33f @1408x768` all *complete*. What they hit first was
`BRAIN_GPU_WAIT_S`, the 30 s GPU-submit watchdog - a whole-clip decode at
those shapes is one submit that takes 100-130 s, so it aborts with
"poll_wait: GPU submit did not complete within 30s -- device likely wedged",
which is not an out-of-memory and reads nothing like one. Raising the
watchdog is what turned four "OOM" rows into four measurements, and it is why
the cliff above is a measured number rather than a bracket. A benchmark pass
that sees that panic at a large VAE shape should raise the watchdog before
concluding anything about memory.

#### 1 - what was read from the reference, and what it actually says

Ported from the cloned `github.com/Lightricks/LTX-2` under
`resources/ltxv/source/`, not invented:

* `packages/ltx-core/src/ltx_core/tiling.py` - `compute_trapezoidal_mask_1d`
  (the blend is **linear-ramp trapezoidal**), `split_by_size`,
  `split_temporal_causal`, and the `masks_are_complementary` /
  `compute_summed_weights` pair.
* `packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py` -
  `map_spatial_slice` / `map_temporal_slice`, the latent-interval to
  pixel-range mappings.
* `packages/ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py` -
  `ConvVideoDecoder._prepare_tiles` / `tiled_decode`, the orchestration.
* `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py` -
  `_CONV_AUTO_LONG_SIDE = (768, 64)` and `_CONV_AUTO_FRAMES = (80, 24)`, what
  `AUTO_TILING` resolves to for **the conv VAE**, plus
  `TileSizeConfig.from_long_side`'s aspect coupling.

`diffusion_tiling.py` was read too and is **the wrong file for this decoder**
- it is the DiffVAE (neighborhood-attention) decoder's tiling, with its own
VRAM-budget search, its own stage-4/5 halo arithmetic and explicitly
`causal_temporal=False` ("DiffVAE temporal tiling deliberately skips ConvVAE
causal split/mask tricks"). Following it would have produced a plausible
tiler with the wrong temporal geometry for the conv decoder this port
actually has. The conv path's authority is `video_vae.py` +
`conv_video_decoder.py`, and those are what got ported.

Three conventions that had to be read rather than guessed:

1. **The two axes use DIFFERENT ramp conventions.** Spatial masks are built
   with `left_starts_from_0=False` (fade-in `i/(r+1)`, never reaching 0);
   the temporal mask uses `True` (fade-in `i/r`, starting at exactly 0).
2. **The temporal split is causal**: `split_temporal_causal` shifts every
   tile after the first back by one latent cell and widens its left ramp by
   one. That extra cell is the "sacrificial first sample", and it exists
   because `map_temporal_slice` maps a latent interval to `1 + (end-1)*scale`
   pixel frames, not `len*scale` - the `1 + 8k` frame rule. Without both
   halves of that asymmetry the temporal masks do not sum to 1.
3. **The overlap is 64 px / 24 frames, and it is deliberately SMALLER than
   the receptive field.** This decoder's spatial receptive field is ~15
   latent cells (6 at the latent grid + 2.5 at 2x + 5.5 at 4x + 1.1 at 8x,
   summing every kernel-3 conv at the resolution it runs at) while a 1080p
   latent is 34 cells tall, so no overlap that still saves memory can cover
   it and **no halo-and-crop scheme can be exact either**. Blending is what
   upstream ships precisely because exactness is unreachable here.

#### 2 - what was built

**`crates/vae/src/tiling3d.rs`** (new, 12 unit tests, pure host geometry, no
GPU code): `trapezoidal_mask_1d`, `Interval`, `split_by_size`,
`split_temporal_causal`, `map_spatial` / `map_temporal`, `AxisPlan`,
`TilePlan3d` and `Blender`. It lives in `crates/vae` rather than
`crates/ltxv` because it is scale-parameterised throughout and `crates/vae`
already owns `blocks3d`, the shared 3D-causal-VAE primitives both `ltxv` and
`wan` build on - `wan`'s VAE has the same `1 + k*scale` frame rule at stride
4 and could adopt it unchanged.

*Not* put in `crates/imaging`, despite that crate's doc claiming "tiling":
`imaging::tiling` is **halo tiles with disjoint cores**, and its own module
doc explains that it does not blend because each output pixel comes from
exactly one tile. That is a different accumulation contract, not a 2D version
of this one, and (per point 3 above) it is the contract that cannot work
here. Its `TilePlan` is also 2D `Rect` geometry with no time axis.

**One design difference from the reference, stated because it is a
difference:** upstream computes `masks_are_complementary` and *skips* the
divisor when the masks partition unity, falling back to a dense
`compute_summed_weights` when they do not. This port always divides. The
tiles are the full cartesian product of the per-axis splits and each tile's
mask is the outer product of its three 1-D masks, so the accumulated weight
factors exactly - `W(t,h,w) = Wt(t)*Wh(h)*Ww(w)` - and the divisor is three
1-D vectors, never a dense `[T,H,W]` buffer. When the masks do partition
unity the divisor is exactly 1.0 and the division is a bit-pattern no-op
(gated: the one-tile case is bit-identical). When they do not - a short final
tile clamps its own ramp, which the reference permits - dividing is the
correct answer rather than an unnormalised seam. One path, no branch to get
wrong.

**`crates/ltxv/src/vae3d.rs`**: `LtxVaeTiling` (pixel-unit layout + `auto()`,
the port of `from_long_side` including Python's round-half-to-even),
`should_tile` (the policy), `WHOLE_DECODE_MAX_PIXELS` (the measured
constant), and `LtxVaeTiledDecoder`. Tiles are grouped by latent SHAPE and
one graph is built per shape, used for every tile of that shape, and dropped
before the next shape's is built - the "fresh resources per unit of work"
pattern `RealDit::forward_q_streamed` already established, and the reason
peak VRAM is one tile's rather than the clip's. A `split_by_size` cover has
at most four distinct spatial shapes however many tiles it has, so this is 4
weight uploads for 9 tiles, not 9.

**`crates/ltxv/src/pipeline.rs`**: both decode call sites (`generate` and
`generate_dfr`) now route through one `decode_video` helper that picks the
path. Everything this port shipped before stays bit-for-bit on the exact
whole path - `WHOLE_DECODE_MAX_PIXELS = 24_000_000` sits above 9f@1080p
(18.8 Mpx) and 25f@720p (22.5 Mpx) and below 25f@1080p (52.2 Mpx).
`BRAIN_LTXV_VAE_TILE=1`/`0` overrides in either direction, which is also how
the two paths get compared.

#### 3 - measured

**The shape this phase exists for**, real weights, one Tesla P40, wgpu:

| 25 frames @1920x1088 | whole path | tiled path |
|---|---|---|
| result | **wgpu Out of Memory** | **completes** |
| peak VRAM | 21719 MiB at abort | **8985 MiB** |
| wall | - | 68.7 s |
| tiles | - | 9 (3x3 spatial, temporal untiled) |
| overlap waste | - | 1.192x |

8985 MiB against a 24576 MiB card is **63% headroom**, and the whole-path fit
above predicted 9087 MiB for a 8.6 Mpx tile - within 1.2% of what the run
actually took, which is the fit being a model rather than a curve drawn
through points.

**49 frames @1280x704**, the other shape Phase 15 recorded as not fitting,
handled by the same mechanism with no special case (7 latent frames is still
under the 10-latent temporal tile, so again only the spatial axes split):

| 49 frames @1280x704 | whole path | tiled path |
|---|---|---|
| result | **wgpu Out of Memory** | **completes** |
| peak VRAM | 23348 MiB at abort | **12302 MiB** |
| wall | - | 59.2 s |
| tiles | - | 4 (2x2 spatial, temporal untiled) |
| overlap waste | - | 1.145x |

**Cost of tiling**, measured where BOTH paths run - 9 frames @1920x1088, the
production 3x3 tile geometry, same latent, same card:

| | wall | peak VRAM |
|---|---:|---:|
| whole | 23.6 s | 15186 MiB |
| tiled | 29.1 s | (one tile's) |

**1.23x wall**, against an overlap waste of 1.192x - i.e. the overhead is
almost exactly the redundant pixels the overlap requires, and the per-shape
graph grouping keeps the extra weight uploads off the critical path.

#### 4 - the correctness gates, and what each one can actually see

The instruction for this work asked for tiled-vs-whole agreement at cosine
>= 0.999999, this crate's usual bar. **That bar is not physically reachable
here and the reason is structural, not a porting defect**: with a ~15-cell
receptive field and a 2-cell overlap, two tiles genuinely see different
context in the seam, and upstream never claims otherwise. Asserting it would
have meant asserting something false. So the claim is split into pieces that
are each exactly true:

**Exact, and gated as exact:**

* `vae_tiling::a_single_tile_plan_is_bit_identical_to_the_whole_decode` -
  real weights, a plan whose tiles exceed every axis, latent extents
  deliberately all different (2 x 2 x 3, so an axis swap cannot hide):
  **cosine 1.000000000, rel_l2 0.0000e0, max_abs 0.0000e0**, and compared on
  BIT PATTERNS, not `==` on f32. The whole tiling machinery - slice, per-shape
  graph build, mask, accumulate, divide - is exact.
* `vae::tiling3d::the_blend_reconstructs_a_known_volume_exactly` - cut a known
  volume into a genuinely 3D-split plan's tiles, feed the pieces back through
  `Blender`, require the result to equal the original: worst `|delta| < 1e-5`.
  Here the "decoder" is the identity, so a mask, slice or divisor bug cannot
  hide behind the receptive-field approximation. This is the gate that
  actually proves the blend.
* `vae::tiling3d::the_temporal_masks_partition_unity_after_the_causal_shift`
  and `the_spatial_masks_partition_unity` - per-axis weights within 1e-6 of 1.

**Approximate, and gated with a measured band:**

* `vae_tiling::a_real_split_agrees_with_the_whole_decode_away_from_a_broken_
  port` - 9 tiles at 9f/256x256, a deliberately harsh split (128 px tiles on
  a 256 px image, 2.25x waste) so it runs in seconds:
  **cosine 0.999093484, rel_l2 4.2641e-2, max_abs 1.6697e-1**.
* `vae_tiling::real_1080p::the_production_1080p_tile_geometry_agrees_with_the_
  whole_decode` (`#[ignore]`d) - the number that describes production: the
  real `auto(1088, 1920)` 3x3 cover at 9 frames, where the whole path still
  fits: **cosine 0.999765097, rel_l2 2.1676e-2, max_abs 2.5191e-1**.
* `vae_tiling::the_blend_beats_a_hard_cut_at_the_same_tile_geometry` - the
  same tiles stitched with no mask and no divisor (later tile overwrites
  earlier): **cosine 0.992828795, rel_l2 1.2401e-1, max_abs 5.4389e-1**. The
  blend is ~2.9x better in rel_l2 and ~3.3x better in max_abs at identical
  geometry, so it is doing work rather than decorating.

**What the approximate gate CANNOT see, established by breaking the code and
re-running rather than by assertion.** This is worth recording because the
obvious assumption is wrong twice:

* Deleting the blend divisor entirely: **no measurable effect** (cosine
  0.999093484, unchanged to nine digits). At this geometry the masks
  partition unity, so the divisor is 1.0.
* Building the SPATIAL mask with the temporal ramp convention: **no
  measurable effect** (cosine 0.999098052). Because this port always divides
  by the accumulated weight, any positive ramp shape renormalises into a
  valid partition of unity. That is a genuine robustness property of the
  always-divide choice - and it is also why the ramp CONVENTION cannot be
  gated from an end-to-end comparison at all.
* Flipping the convention on the TEMPORAL axis: **caught**, at deviation
  0.0556 against a 1e-6 bound, by
  `the_temporal_masks_partition_unity_after_the_causal_shift`, and by
  `the_blend_reconstructs_a_known_volume_exactly`.

So the end-to-end gate's real job is narrow and is documented as such in the
test file: it bounds gross structural error (decoding the wrong sub-volume,
stitching to the wrong offset, losing a tile). The exact properties live in
`tiling3d`'s unit tests and in the one-tile bit-identity gate.

**No whole-path result exists for the shapes that motivated this**, so
`vae_tiling::real_1080p::a_full_25_frame_1080p_clip_decodes_on_one_card` and
its 49-frame/720p twin (both `#[ignore]`d) gate what can be checked without
one: it completes, every value is finite, the output has real dynamic range
(25f@1080p: min -0.8327, max 0.6655, std 0.1573 - not a flat or degenerate
image), and **no tile boundary is a gradient outlier**. That last one is the
seam check: mean absolute horizontal gradient per output column, probed in a
window around the centre of each interior tile's fade-in region, against the
median column. Measured **1.06x the median** at 25f@1080p and **1.08x** at
49f@1280x704, against a 6x bound. A visible seam is exactly a gradient spike
at a known column, so this observable is the one that would catch a blend
that had silently degenerated.

#### 5 - what was explicitly NOT done

* **Tiled ENCODE.** `VideoEncoder.tiled_encode` is a real thing upstream and
  is not ported. Nothing in this pipeline encodes a clip large enough to need
  it - `--start-frame` encodes ONE frame - and the encode-side geometry
  differs (`map_temporal_interval_to_latent` / `map_spatial_interval_to_latent`
  build RECTANGULAR masks, not trapezoidal, and upstream validates a 16-frame
  / 64-px minimum overlap there that decode does not). Porting it on
  speculation would have been a second untested surface.
* **The DiffVAE (`NADiffusionDecoder`) tiling.** Different decoder, different
  file, still out of scope - see item 1.
* **A VRAM-budget search for the tile size.** Upstream has one
  (`recommended_decode_tiling_config`) but only for the DiffVAE; the conv
  path's own auto layout is aspect-only and that is what got ported. Sizing
  tiles from live free VRAM needs the per-request VRAM estimate that
  `LtxvResident::estimate` still lacks (a gap Phase 13 opened and Phase 15
  re-recorded), and inventing a second answer to "how much fits" while that
  is open would give the workspace two.
* **Reusing `crate::dfr`'s `tile_ranges`/`stitch_tile_latents`.** Read first,
  as instructed. They do not transfer: DFR tiles the TEMPORAL axis of a
  latent mid-diffusion and stitches by dropping a lead-in prefix and
  concatenating - no overlap blend, no masks, no spatial axes, and a seam
  list driven by keyframe positions rather than a memory budget. The shape is
  genuinely different, not a specialisation.
* **Reusing `crates/wan`'s VAE chunking.** Also read first. Wan solves the
  temporal axis with a cross-chunk `FeatCache` that makes chunked decode
  EXACTLY equal to whole-clip decode - which works because Wan's decoder is
  causal. LTX's conv decoder runs `causal=False` (this checkpoint's
  `causal_decoder: false`), so every conv pads symmetrically and reads a
  future frame; there is no cache that makes a chunk exact. Wan also does not
  address the spatial axes at all, and the spatial axes are half of what the
  measurement above says drives the OOM.
* **Making the tiled path the default at every shape.** It is a lossy path.
  Every shape that fits keeps the exact one.

#### 6 - end to end, on the real checkpoints

A real generation, not a decode in isolation - the REAL Q8_0 22B DiT + REAL
Gemma-4 text encoder + REAL conv VAE, `--start-frame` a real PNG, on **one**
Tesla P40, the default **wgpu** backend:

    BRAIN_GPU_INDEX=0 BRAIN_GPU_WAIT_S=1800 \
    BRAIN_LTXV_VAE=<conv VAE> BRAIN_LTXV_DIT=<Q8_0 22B> \
    BRAIN_LTXV_TEXT_ENCODER=<Gemma-4 Q8_0> \
      ./target/release/brain -v ltxv t2v \
        --prompt "a red fox trotting through tall grass at golden hour, cinematic" \
        --frames 25 --width 1920 --height 1088 --fps 24 \
        --guidance 1.0 --seed 7 --dit-config ltx25_22b \
        --start-frame out/sdxl-fox.png --device gpu0 --output-path out.mp4

**It completes**, `rc=0`, and writes a real 1920x1088 / 25-frame / 24 fps
h264 file (verified with `ffprobe`: `nb_frames=25`, `width=1920`,
`height=1088`).

| stage | wall |
|---|---:|
| build | 8.0 s |
| text encode | 0.1 s (Phase 11's on-disk context cache, warm) |
| denoise, 8 forwards at 8160 tokens | 2091.8 s (261.5 s/forward) |
| **VAE decode (tiled)** | **70.1 s** |
| other | 6.0 s |
| **total** | **2176.0 s** |

**Peak VRAM: 16651 MiB on gpu0, 1 MiB on gpu1.** Two things in that one line:

* `--device gpu0` really did confine the run to one card (gpu1 never moved
  off idle), so this is a genuine single-P40 result.
* 16651 MiB is the **DiT's** peak, matching Phase 15's isolated 16650 MiB.
  The tiled decode's own plateau is 8652 MiB, sampled directly, and then
  releases to 14 MiB - **entirely underneath the DiT's high-water mark**. The
  VAE decode is no longer the binding constraint at this shape; it is no
  longer even the second-largest allocation.

**The same command on the pre-change binary is the control**, run first and
by accident (the CLI had not been rebuilt after the pipeline was wired -
worth recording, because the failure looked exactly like a bug in the new
code and was not). It reached the identical point and died:
`wgpu error: Out of Memory`, peak 23313 MiB, **after all 8 denoise steps
completed**. The 200 ms VRAM trace of that run is the clean before/after:
the DiT releases to 12 MiB at the end of denoising, then a SINGLE monotonic
climb 570 -> 23313 MiB aborts - the whole-clip decode allocating one graph -
where the new path shows a per-tile plateau at 8652 MiB instead. That run
also confirms, independently, that text-encode + denoise WITH image
conditioning at this shape peaks at 16645 MiB and releases cleanly, i.e. the
VAE decode was the sole end-to-end blocker.

#### Gates, all green

New: 12 in `vae::tiling3d`, 6 in `ltxv::vae3d`, 3 routine + 3 `#[ignore]`d
real-weight tests in `crates/ltxv/tests/vae_tiling.rs`.

Every pre-existing gate re-run and green: `cargo test --release -p brain-ltxv
--lib --tests` (136 unit + every integration suite, including `dit_parity`,
`av_dit_parity`, `vae_parity`, `host_forward_parity`,
`streamed_vs_eager_real`, `connector_real_parity`, `block_weight_cache`,
`na_decoder_parity`, `upsampler_parity` and `av_shard_2gpu_real`) and
`cargo test --release -p brain-vae --lib --tests`.

#### What a benchmark pass should know before it runs

* **1080p is no longer capped at 9 frames, and 720p is no longer capped at
  25.** Phase 15's cap note is superseded: the tiled path runs 25f@1080p at
  8985 MiB and 49f@1280x704, both on one card.
* **The whole-path ceiling is ~35 Mpx on a 24 GiB card**, not a frame count
  and not a resolution - see item 0's table before assuming either.
* **A "GPU submit did not complete within 30s" panic at a large VAE shape is
  the watchdog, not memory.** Raise `BRAIN_GPU_WAIT_S`. A whole-clip decode
  is one submit and takes 100-130 s at 35 Mpx.
* **Tiled decode is lossy by construction** (item 4). Any A/B that compares a
  1080p clip against an older 720p one is comparing an approximate decode
  against an exact one as well as two resolutions.
* **`BRAIN_LTXV_VAE_TILE=1`** forces the tiled path at shapes that fit, which
  is the supported way to measure the two against each other.

### Phase 17 - the x0 conversion runs at the token's OWN timestep

Every real `--start-frame` clip this port has ever produced had a visible
colour defect: frame 0 came out over-saturated, the frames right after it
washed out to roughly the model's unconditioned level, and the colour then
climbed back to a HIGHER plateau for the second half of the clip and stayed
there. The shape was reported as an anomaly because nothing about
"conditioning influence decays with distance from the anchor" predicts a
clip whose LAST frames look more conditioned than its middle ones.

The whole shape is one line, and the line is not in the conditioning code.

#### What the numbers actually said

Mean HSV saturation per decoded frame (64x64 downsample, `colorsys.rgb_to_hsv`
averaged over the frame), real 22B Q8_0 DiT + real Gemma-4 encoder + real conv
VAE, one Tesla P40, `--start-frame [path/to/malinois.png]` (a real photo of a
Belgian Malinois, at the clip's own resolution):

| run | f0 | trough | late plateau |
|---|---:|---:|---:|
| 512x512, g=3.0, seed 42 | 0.555 | 0.311 @ f5 | 0.50 |
| 1280x704, g=3.0, seed 42 | 0.554 | 0.294 @ f6 | 0.46 |
| 1280x704, g=1.0, seeds 100/101 | 0.556 | 0.31 | 0.47 |
| plain t2v, no stills, 512x512 | 0.330 | (flat) | 0.31 |

Two measurements that had not been taken re-framed the whole thing:

* **the conditioning still's OWN mean saturation is 0.461**, and
* a `--start-frame X --end-frame X` run (the APPENDED-keyframe mechanism,
  which never overwrites latent frame 0) is **flat at 0.463 for all 25
  frames**.

So the "elevated late plateau" is not elevated - 0.46-0.50 is the correct
level, the one the source image and the appended-keyframe path both sit at.
The defect is at the other end: **frame 0, the one frame that is supposed to
BE the still, was 20% over-saturated** (0.555 vs 0.461), and the trough was
the causal VAE decoder's temporal receptive field smearing that over-driven
latent frame across its neighbours. A 25-frame clip is 4 latent frames, so
`f0 / f1-8 / f9-16 / f17-24`; the V bottoms out in the middle of latent frame
1 and is gone by latent frame 3, exactly the reach of a corrupted latent
frame 0.

#### Root cause

`crates/ltxv/src/pipeline.rs`, the denoise loop:

    let mut denoised = to_denoised(&latent, &velocity, sigma);   // scalar

The reference does this conversion **inside the model wrapper**, not in the
sampler - `ltx_core/model/transformer/model.py`, `X0Model.forward`:

    denoised_video = to_denoised(video.latent, vx, video.timesteps)

and `Modality.timesteps` is `timesteps_from_mask(denoise_mask, sigma)` =
`denoise_mask * sigma`, shape `(B, T, 1)`, broadcast over the `(B, T, C)`
latent's channel axis (`ltx_pipelines/utils/helpers.py`). It is **per token**.

For plain text-to-video the two are the same number on every token, because
`denoise_mask` is all ones - which is why `dit_parity`, `av_dit_parity`,
`host_forward_parity`, `streamed_vs_eager_real`, `motion_real` and the t2v
control were all green through this. The moment anything is frozen they are
not: a `--start-frame` anchor is announced at timestep **0**, where the
reference's conversion is the IDENTITY, and brain instead subtracted a
full-strength `velocity * sigma` from an already-clean token.

Within the loop that error is invisible - `post_process_latent` re-pins the
frozen tokens after every step. It survives in exactly one place: the
**terminal step**. `samplers._ancestral_euler_denoising_loop` short-circuits
`sigma_next == 0` to the x0 estimate with no re-pin (brain matches this, and
matched it before this phase). So the very last thing that happens to the
anchor before it is decoded is the one thing that corrupts it.

**Which means the defect exists only under the ancestral sampler**, and that
is not a footnote - it cost this phase a real-weight run to learn. The
deterministic loop (`samplers._step_state`) re-pins the x0 ESTIMATE *before*
the step formula touches it, so a bad x0 conversion at a frozen token is
overwritten by `clean` and vanishes. The ancestral loop re-pins the STEPPED
latent instead and hands the terminal step's raw estimate straight out. The
first version of this phase's real-weight gate copied `motion_real.rs`'
`eta = 0.0` for reproducibility, and consequently **passed against the
defect at bit-identical numbers** (frame-0 delta 5.33 either way). `eta`
had to become `1.0` - `ANCESTRAL_ETA`, what `ltx_pipelines.distilled` runs
for every checkpoint at or above 2.5, what `GenOpts::default` sets, and what
every one of the buggy generations above used. Nothing is given up: the
renoise draw is `data::rng::Rng` seeded from `GenOpts::seed`, so `eta = 1` is
as run-to-run deterministic as `eta = 0`.

The size of the corruption is a constant, which is why the curve was
seed-, guidance- and resolution-independent. A rectified-flow model at t=0
predicts `v ~= -x0`, so `x0_wrong = clean - (-clean)*sigma_terminal =
(1 + sigma_terminal)*clean`, and `LTX2_DISTILLED_SIGMAS`' last non-zero entry
is **0.421875**. The anchor latent was ~1.42x too large on every run.

#### How it was confirmed before anything was changed

A weight-free controlled experiment, no DiT involved at all: encode the
conditioning still as a 25-frame static clip through the real conv VAE, then
decode it twice - once unchanged, once with **latent frame 0 alone multiplied
by 1.421875** - and measure the same saturation curve.

| pixel frame | 0 | 2 | 5 | 7 | 9 | 12 | 14 | 20 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE probe, correct LF0 | 0.464 | 0.464 | 0.463 | 0.464 | 0.467 | 0.463 | 0.459 | 0.459 | 0.467 |
| VAE probe, LF0 x 1.421875 | 0.554 | 0.453 | 0.350 | 0.307 | 0.369 | 0.451 | 0.490 | 0.468 | 0.469 |
| the real buggy 512x512 run | 0.555 | 0.422 | 0.311 | 0.320 | 0.372 | 0.483 | 0.500 | 0.498 | 0.503 |

One scale error on one latent frame reproduces the frame-0 over-saturation,
the V-trough AND the "unexplained late-clip rise" - on a clip that has no
generated content in it whatsoever. That is what made this a root cause
rather than a plausible story.

#### The fix

`to_denoised` now takes the per-token `timesteps` slice and the channel width
and broadcasts the way `X0Model.forward` does; the denoise loop hands it the
SAME `timesteps` vector it just handed the model. One file,
`crates/ltxv/src/pipeline.rs`.

Nothing else moves. The CFG fold stays on the velocity: `to_denoised` is
affine in `v` with the same per-token coefficient for both branches, so
folding on `v` then converting is identical to the reference's
convert-then-fold on x0. `generate_dfr` passes `frozen: None` at every one of
its four `denoise` call sites, so its timesteps are uniform and its output is
bit-identical to before.

#### Re-verified on a real clip

The SAME settings that produced the first row of the table above, re-run
against the fixed sampler - 512x512, 25 frames, seed 42, guidance 3.0, same
prompt, same `--start-frame`, real 22B Q8_0 + real Gemma-4 + real conv VAE,
one Tesla P40 pair, 417.9 s (`denoise 397.8 s = 24.862 s/forward`):

| frame | 0 | 3 | 5 | 7 | 9 | 12 | 16 | 20 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | **0.555** | 0.357 | **0.311** | 0.320 | 0.372 | 0.483 | 0.502 | 0.498 | 0.503 |
| after | **0.469** | 0.453 | 0.454 | 0.455 | 0.462 | 0.469 | 0.488 | 0.492 | 0.500 |

Frame 0 is now within 1.7% of the conditioning still's own 0.461 (it was 20%
high). The trough is gone - the deepest dip is 0.453, 3% below frame 0,
against a 33% collapse to the unconditioned baseline before. The
"unexplained late-clip rise" is gone with it: what is left is a smooth,
monotone 0.46 -> 0.50 drift across 25 frames as the generated content moves
away from a real-photo anchor, which is the behaviour the plain-t2v control
always predicted and which nothing in the reference says should not happen.

#### Gates

* `pipeline::tests::a_frozen_token_survives_the_terminal_step_exactly` - the
  real gate, weight-free, microseconds. Drives the whole loop on the real
  schedule's own terminal pair `[1.0, 0.421875, 0.0]` with a denoiser that
  returns a non-zero velocity (the fix must not depend on the model
  conveniently predicting zero at a clean token) and asserts the frozen token
  ends at **exactly** its clean content, under both `eta = 0` and `eta = 1`.
  Verified RED first: it failed at `4.578125`, i.e. `5.0 - 0.421875`, which is
  the defect's own arithmetic.
* `pipeline::tests::to_denoised_is_per_token_and_collapses_to_the_scalar_form_when_nothing_is_frozen`
  - pins both halves of the contract, including that the unconditioned path
  is unchanged.
* `crates/ltxv/tests/anchor_real.rs` (new, `#[ignore]`d, ~6 min) - the
  perceptual half: two real 22B generations at 384x192 / 9f (one
  unconditioned to produce the anchor, the same way `motion_real.rs` builds
  its own, so no image fixture and no resize can be blamed; one conditioned
  on it), asserting the clip's first decoded frame reproduces the
  conditioning still. **Calibrated by running it both ways**, same shape,
  same seed, same sampler, nothing else changed:

  | | frame-0 saturation | ratio vs the still (0.3037) | frame-0 delta |
  |---|---:|---:|---:|
  | scalar-sigma x0 conversion | 0.4522 | **1.489** | 12.84 |
  | per-token x0 conversion | 0.3050 | **1.004** | 2.67 |

  Bounds 1.08 and 7.0. The full saturation curve tells the same story:
  `0.452 0.369 0.322 0.282 0.261 0.234 0.220 0.201 0.192` defective against
  `0.305 0.299 0.297 0.293 0.291 0.288 0.288 0.286 0.286` fixed.

`cargo test --release -p brain-ltxv --tests`: **138 unit + every integration
suite green**, `dit_parity`, `av_dit_parity`, `host_forward_parity`,
`streamed_vs_eager_real`, `vae_parity`, `vae_tiling`, `na_decoder_parity`,
`upsampler_parity`, `connector_real_parity`, `block_weight_cache`,
`cfg_parallel` and the `image_conditioning_tests` /
`conditioned_latent_tests` modules included. *(Run with `BRAIN_LTXV_DIT`
UNSET: `an_explicit_vae_path_beats_the_environment_variable` asserts
`Paths::resolve` finds no DiT, so exporting the real checkpoint paths fails
it. Pre-existing, unrelated, left alone.)*

#### What was ruled out on the way, and stays ruled out

Recorded so the next investigation does not re-walk it: the conditioning
builders are correct. `conditioned_latent`'s start-only branch touches
exactly `[0, lh*lw)` in the denoise mask, in `clean` and in the initial-latent
mix; `post_process_latent` re-pins exactly that range and skips every
`mask == 1.0` token; `keyframes_mask` is the unconditional first-latent-frame
marker `VideoLatentTools._first_frame_keyframes_mask` builds and is not
aliased anywhere; the mechanism split (`VideoConditionByLatentIndex` for
`frame_idx == 0`, `VideoConditionByKeyframeIndex` otherwise) matches
`helpers.combined_image_conditionings` line for line; and the VAE's
`per_channel_statistics` normalize/un-normalize is applied on both the encode
and the decode side (`vae3d.rs:487-490` and `:573-576`). The appended-keyframe
control measuring flat at the source image's own saturation is the empirical
statement of all of that at once.

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

- **`brain ltxv upscale`'s multi-pass continuity has no real-weight
  measurement, and the command has no capability action.** Phase 21 added
  post-hoc 2x upscaling of a finished clip, sharing `upscale_and_refine` with
  the internal two-stage path so the Phase 19 un-normalize defect cannot recur
  in a second copy. Its independent-segment plan turned out to be a real defect
  and Phase 25 replaced it with `longform::window_plan`'s rolling latent
  context. Gated weight-free (the plan) and on the real VAE + real spatial
  upscaler with the tiny DiT (wiring across a real seam, CPU). No `ltx25_22b`
  run of the fixed path has happened: the code path is exercised, the QUALITY
  of its output is unmeasured and unclaimed, and the multi-pass seam is argued
  from Phase 22's own measurement of the same mechanism rather than measured
  again here with `clipmetric::blowup_ratio`. The reduced context a dense
  output grid forces (2 latent frames at 2560x1408, against the reference's 8)
  is a compromise no number in this repo justifies - only the full 8 is cited.
  Separately, `upscale` is CLI-only - it would be the first action here to
  take an input BLOB rather than parameters alone, and that shape is
  undesigned. See Phase 21 item 3 and Phase 25.

- **An anchor position is not routed to a long-form window.** Phase 26 added
  `--mid-frame`/`--mid-frame-at`, a third conditioning still at an arbitrary
  interior pixel frame, composable with `--start-frame`/`--end-frame` and
  correct across both stages of a two-stage run. It is **refused** for a
  multi-window or multi-scene clip, alongside the `--end-frame` refusals Phases
  22 and 24 wrote. What is missing is one piece of routing: find the window
  whose emitted frame range contains the requested pixel frame, re-express the
  position in that window's own frame numbering, and decide what an anchor
  landing inside a carried latent context means - `denoise_stage` refuses a
  still and a context together outright today, and an appended guide does not
  actually collide with a context the way an overwrite does, so that refusal is
  broader than it needs to be. No real-weight run exists for any three-anchor
  generation either; the gates are arithmetic plus a tiny-DiT CPU wiring test.
  See Phase 26.

- **Single-stage generation past the distilled schedule's token count**:
  **closed in Phase 19**. `generate` ran `LTX2_DISTILLED_SIGMAS` at the
  requested resolution; `ltx_pipelines.distilled` only ever runs it at
  `width // 2, height // 2` and then refines at full size with
  `STAGE_2_DISTILLED_SIGMAS`. Past ~6k video tokens the one-stage form
  disintegrates the END of the clip (blowup ratio 14.66 at 8160 tokens
  against 1.03-1.06 at 1024-5600). Now routed by `should_two_stage` /
  `SINGLE_STAGE_MAX_TOKENS`. **Still open: STG, joint audio/video guidance
  and CFG rescale** - upstream's `MultiModalGuiderParams` carries
  `stg_scale` and `rescale_scale` (0.45 for the HQ preset) and this port
  folds plain CFG only, which is a pre-existing simplification this module's
  own doc already records, not something Phase 19 introduced.

- **The latent upscalers were called in the wrong latent space**: **closed in
  Phase 19**, and it was pre-existing rather than new.
  `ltx_core.model.upsampler.model.upsample_video` un-normalizes with the
  VAE's `per_channel_statistics`, upsamples, and re-normalizes; all three
  call sites in this crate (the new two-stage one plus `generate_dfr`'s
  spatial video, keyframe-slot and temporal rounds) called
  `LatentUpsampler::upsample` bare, costing half the latent's variance. The
  reason it survived a green parity suite is worth carrying forward:
  `upsampler_parity.rs` computed `max_abs`, PRINTED it, and asserted on
  COSINE alone, and cosine cannot see a scale error. It now asserts both.
  **Note that no test drives `generate_dfr` end to end**, so the DFR half of
  that fix is gated by the helper's own test, not by a whole-pipeline run.

- **The LTX int8 tier does not run on `backend-vulkan` at all.** Every
  attempt panics with `GPU device lost while waiting for a submit to
  complete` (`crates/backend-vulkan/src/lib.rs`'s `wait_for_fences`), at
  every shape tried and independently of device residency: 48 layers at
  T=3520 with a full resident window (19192 MiB), the same with residency
  off (**6769 MiB**, nowhere near a memory limit), two layers at 512 tokens,
  and `cargo test -p brain-ltxv --test int8_compute` (both tests). Found
  while checking Phase 18's backend-aware residency budget, which reserves
  far less on this backend precisely because it does NOT carry wgpu's
  measured 2.00x per-uploaded-buffer resident cost - so this is the backend
  where device residency should pay MOST, and the budget for it is written,
  gated and unverifiable against real weights until this is fixed. Not
  diagnosed further: it predates Phase 18 and is a backend defect, not a
  model one.

- Overlapping-tile chunked VAE **decode**: **closed in Phase 16**. The video
  VAE milestone left "general overlapping-tile chunked encode/decode ... out
  of scope, deferred to the DFR milestone", and Phase 15's item 0 measured it
  as the real 1080p blocker. `vae::tiling3d` + `ltxv::vae3d::
  LtxVaeTiledDecoder` port `ltx_core.tiling` + `ConvVideoDecoder.tiled_decode`;
  25 frames at 1920x1088 now decodes at 8985 MiB where the whole path was a
  hard `wgpu` out-of-memory. **Still open: tiled ENCODE**
  (`VideoEncoder.tiled_encode`), deliberately - nothing in this pipeline
  encodes a clip large enough to need it, and its geometry genuinely differs
  (RECTANGULAR masks, a validated 16-frame/64-px minimum overlap). Also still
  open: tiling the NA diffusion decoder (`diffusion_tiling.py`, a different
  decoder), and sizing tiles from live free VRAM rather than the reference's
  aspect-only auto layout - the latter is blocked on the per-request VRAM
  estimate `LtxvResident::estimate` still lacks, below.
- Tiled decode is **lossy by construction, and is not claimed otherwise**.
  The conv decoder's spatial receptive field is ~15 latent cells against a
  64-px (2-cell) overlap, so no memory-saving tiling can be exact and no
  halo-and-crop variant can be either - this is upstream's own trade, not a
  porting defect. Measured at the production 1080p geometry: cosine
  0.999765097 / rel_l2 2.1676e-2 / max_abs 2.5191e-1 against the whole
  decode. The tiling MACHINERY is exact (one-tile plans are bit-identical);
  see Phase 16 item 4 for which gate can see what, including the two
  deliberate breaks that end-to-end comparison provably CANNOT detect.
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
  still open for `LtxDit` (video-only) - and phase 36 measured that closing it
  is not worth doing for LATENCY (1.02x on a warm forward, against 2.11x of
  throughput the same second card already returns as an independent-request
  lane). Read that phase before picking this up. The "int8/int4 compute + AV
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
  own pipeline. **The cross-generation half is closed in Phase 13**: the
  cache is now keyed on CHECKPOINT identity and held in a process-wide
  registry under a `memauth`-derived byte budget with `residency`'s own
  cost-aware eviction, so a second generation against the same file starts
  warm on its first layer (measured through `LtxvResident`: 48 block misses
  on generation 1, ZERO on generation 2). Still open: survival across
  separate PROCESS invocations, which needs an on-disk pre-quantized block
  store.
- `ada_layer_norm_single`'s host-side `linear()` call was a naive,
  unthreaded, unblocked scalar loop re-streaming its ~604 MB weight matrix
  from host RAM once per output row (Phase 8's flat ~21s/forward
  measurement, ~11% of the real per-step total) - **closed in Phase 9**
  (see below): row-parallelized, bit-identical, ~3.5x on the stage itself.
  Not fully closed at the time: the fix was bandwidth-, not
  thread-count-, bound (48 cores measured only ~3.5x, since every thread
  still re-walks the same 604 MB matrix), so a blocked/tiled rewrite that
  avoids the redundant re-reads entirely remained a further, unattempted
  win. **Closed in Phase 14**: `backend_cpu::host_gemm::blocked_linear`,
  register- and cache-blocked, bit-identical, 14.39s -> 8.32s on the
  isolated real shape (73.9 -> 127.8 GFLOP/s, which is the ~1 scalar
  MAC/core-cycle ceiling, so it is now arithmetic-bound rather than
  bandwidth-bound). Phase 14 also corrected this entry's own premise: at
  T=3520 the `linear()` call was only 15.1s of the 75.8s stage - the other
  60.7s was `ada_layer_norm_single`'s SERIAL per-row timestep embedder,
  fixed in the same phase. Still open: a bit-identical vectorization ACROSS
  `M` (AVX2 lanes over output rows, each lane still summing its own `k`),
  which would lift the scalar ceiling by ~8x.
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

### Phase 18 - the forward stops being a PCIe benchmark

Phase 17 left a real 48-layer 720p forward at **111.9 s wall against 15.5 s of
GPU kernel time - 13.9% of the wall was compute.** This phase is the
architectural fix for the other 86%, and the first thing it did was kill the
hypothesis it was commissioned on.

#### 0 - the premise was wrong, and one measurement said so

The brief was: `forward_q_streamed` opens a fresh `Gpu` per call and re-uploads
all 48 already-quantized blocks (~13 GB) on every one of a generation's 8-16
forwards, so device residency is the fix. The design is real and the
re-uploading is real. Its cost was not what anyone thought.

Instrumenting the one bucket that used to hide all of it (`block GPU
upload+forward+wait`, 99.3 s of the 111.9 s) split it four ways. Real
`ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, 48 layers, T=3520, cache-warm,
one Tesla P40, `BRAIN_PROFILE=1 ltxv_bench streamed 48 3520 1024 1`:

| stage | ms | share of 111.9 s |
|---|---:|---:|
| **block adaLN combine+slice (host, per block)** | **35 966** | **32.1%** |
| block submit + output readback (contains all 15.5 s of GPU kernel) | 22 430 | 20.0% |
| block activation+modulation upload & record | 14 253 | 12.7% |
| **block parity-tap readback, DISCARDED** | **11 651** | **10.4%** |
| adaLN-single model table (host, once per call) | 10 269 | 9.2% |
| block WEIGHT upload to device | **~9 000** | ~8% |
| RoPE build / patchify / connector / open_device | 1 100 | 1.0% |

The weight re-upload - the thing this phase was chartered to remove - was
**~9 s of 99 s**. What actually dominated:

1. **`dit::adaln::add_table` + `slice_mod`, on the host, once per BLOCK.** At
   the real width that is a `[3520, 36864]` f32 combine (519 MB written) plus
   nine `[3520, 4096]` slices, then nine 57.7 MB uploads - per block, times 48,
   times every step, for a table whose only per-block input is a **147 KB**
   `scale_shift_table`. ~1 GB of host memcpy and ~519 MB of PCIe per block:
   **~25 GB of PCIe per forward.**
2. **`LtxBlockQ::forward` read back its three parity taps on every block** -
   `attn1_out`/`attn2_out`/`ff_out`, each a full `[t, dim]` - and
   `forward_q_streamed` threw all three away. 173 MB per block, **8.3 GB per
   forward**, 11.7 s, for values nothing read.

This is §F.2 arriving through the same door Phase 14 recorded it: the profiler
named a stage, a plausible story was told about which line inside it dominated,
and the story was wrong by an order of magnitude. The cheap check - split the
bucket before optimizing it - is what turned a ~9 s fix into a ~64 s one.

#### 1 - what was built

**`crates/ltxv/src/devres.rs`** (new) - the device-residency lifecycle:
`DitSession` (one card, one generation: one `Gpu` held open, plus an optional
`BlockWindow`), `BlockWindow` (a fixed number of device SLOTS over the model's
blocks), the VRAM budget, and `run_blocks`, which is now the ONE block-stack
implementation for every residency mode.

**`crates/weightset` gets its first production consumer.** Phase 13 checked it
for the HOST cache and correctly declined (variable-sized blobs under a shared
byte budget across an unbounded set of checkpoints - `residency`'s
`EvictionPolicy` problem). This is the other problem, the one its module doc
describes: *a fixed-size window of device slots over a model's weight groups*
visited *in an order known exactly in advance*. A denoise loop is
`Schedule::cyclic(48, passes)` against a known slot budget, the slots are
uniform (every block is the same shape), and `CyclicScan` pins the longest
prefix and rotates the tail by furthest-next-use. Used as-is; no second window
was written. The cursor is the layer index taken against a 2-pass schedule -
from position `l` the remaining `[l, 2n)` range holds every group exactly once
more, which is the whole lookahead Bélády needs, and deriving it from `l`
rather than a running counter means an aborted forward cannot desynchronise it.

**`crates/kernels/wgsl/adaln_row.wgsl`** (new - the only kernel this phase
adds). `out[r,d] = tbl[row*D+d] + tab[r*NR*D + row*D + d]`, optionally
`1.0 +` that. §F.3 was done first and the tree genuinely has no fit:
`bias_add` broadcasts a row but only IN PLACE, `region_copy` preserves the
source layout rather than compacting a strided row out of it, and
`add_chan_bcast`/`broadcast_add_hw` are NCHW spatial ops. Coalesced by
construction (consecutive threads walk consecutive `d`, contiguous in both
operands), no reduction, no shared memory, no barrier - a pure streaming op,
which is the right shape for something whose entire job is to not be PCIe.
Measured at **211 ms per 48-layer forward**, 1.3% of GPU kernel time, against
the 36.0 s + ~25 GB it replaces.

**`crates/ltxv/src/block.rs`**: `ModBufs` (the device-side twin of `Mod`) with
two fills - `upload` (host combine+slice, the reference definition of the
arithmetic, still what every tap-producing/parity path takes) and `derive`
(nine `adaln_row` dispatches from a per-FORWARD table upload plus the block's
resident `scale_shift_table`). `LtxBlockQ` gains `sst_buf` (147 KB, resident
with the weights) and `forward_prod` (modulation derived on device, no taps,
context uploaded once per forward). `forward_on`/`forward_timed` split the
per-call SCRATCH handle from the block's own - a resident block's activations
must not be charged against `memauth` through its long-lived weight handle,
since a `Gpu`'s grants are released when the HANDLE drops, not the buffer.
`BlockTimings` makes the four-way split above a permanent, readable number
instead of one this phase had to add by hand.

**`crates/ltxv/src/dit.rs`**: `forward_q_streamed_in(session, ...)` is the
entry point production takes; `forward_q_streamed` is now a thin wrapper over
it with a session that keeps nothing - same function, one body, and the arm
the bit-identity gate compares against. **`crates/ltxv/src/pipeline.rs`**:
`RealDit` holds one session PER CARD, keyed on `devices::current_gpu()`, so
the concurrent CFG pair gets two sessions on two cards and `Single` gets one;
`generate`'s existing `drop(dit)` before the VAE decode releases both.

#### 2 - the VRAM finding that decided the whole budget

**Device residency of the weights is NOT where the win is on this hardware, and
the number that says so is a wgpu defect this repo had already measured.**

`crates/gpu-core/tests/vram_overhead.rs` records that on a non-ReBAR Pascal
card under the default wgpu backend every uploaded storage buffer costs
**2.00x** its size resident, and that brain's own native Vulkan backend
measures 1.00x. `.agents/rules/kernels.md` sec D records the other half: that
staging is only retired by a **blocking readback**. Both bit, in order:

* Pre-filling 48 blocks with no drain reached **24392 MiB of a 24576 MiB card**
  and aborted (`wgpu error: Out of Memory`).
* With a one-word drain per block (`paramstore::upload::Uploader::drain`'s
  trick, reused) the resident cost fell to a measured **297 MiB/block**, i.e.
  1.1x - the doubling is drainable after all, which the earlier measurement did
  not separate.
* But wgpu's allocator pool is **elastic and greedy**: with nothing else
  holding memory it grows to 16522 MiB at T=3520 and stays there; under
  pressure from long-lived allocations it works in 5702 MiB. It does not back
  off when an allocation fails, it errors. So the pool must be BUDGETED at its
  greedy size, not its need.
* And the order matters: filling the window lazily, block by block, lost the
  race against the pool and aborted at 24009 MiB by block 28. `weightset`'s own
  `slot_contents` doc says a caller must load the initial pins; what this phase
  added is WHEN - `DitSession::prefill` runs before the RoPE tables, the
  connector, or any block has allocated anything.

The measured slots/peak relation at T=3520 is linear and now published:
`peak_MiB ~= 16522 + 285 * resident_blocks`. The shipped reserve
(`activation_reserve_bytes`) is fitted to the greedy plateau plus real
headroom, which gives 20 resident blocks at 720p and a 22087 MiB peak.

**A killed hypothesis, with its number:** chaining the activations across all
48 blocks (leaving `x` on the card, one upload and one readback per FORWARD
instead of per block) removes ~5.6 GB of round trip and saves about **3 s**. It
also removes the one blocking readback per block that makes the pool shrink,
and the pool then goes from 5.7 GiB to 16.5 GiB. Ten GiB of pool is worth ~35
resident blocks. Reverted; `forward_prod` keeps the per-block readback and says
why.

**And the residency win itself, measured both ways at 720p:** 0 resident
blocks 52.09 s, 24 resident blocks 51.83 s, 25 resident blocks 47.90 s at a
23623 MiB peak. Residency is worth a few seconds here, not tens - because near
a full card the driver starts spilling new allocations (the same per-forward
adaLN upload costs 0.51 s at 0 resident blocks and 7.5 s at 25). The mechanism
is right and is kept; on this backend, at this resolution, it is a minority of
the win and is budgeted conservatively rather than maximised.

#### 3 - measured, 720p (T=3520), 48 real layers, cache-warm, one Tesla P40

`BRAIN_GPU_INDEX=0 BRAIN_GPU_WAIT_S=1800 BRAIN_PROFILE=1
BRAIN_LTXV_DIT=<real Q8_0 22B> ./target/release/ltxv_bench streamed 48 3520 1024 1 [resident]`,
`nvidia-smi --query-gpu=index,memory.used --loop-ms=200` throughout.

| | before (Phase 17) | after |
|---|---:|---:|
| wall, 48 layers, cache-hit call | **111.86 s** | **50.48 s** (2.22x) |
| GPU kernel time (timestamp queries) | 15 536 ms | 16 196 ms |
| **GPU compute share of wall** | **13.9%** | **32.1%** |
| block adaLN combine+slice (host) | 35 966 ms | **0 ms** |
| discarded parity-tap readback | 11 651 ms | **0 ms** |
| block weight upload to device | ~9 000 ms | 6 206 ms (20 of 48 resident) |
| peak VRAM | 16 522 MiB | 22 087 MiB |
| resident blocks | 0 | 20 |

Output statistics identical to every digit printed across every arm of every
run in this phase - `len=450560 mean=0.060893 std=0.683607 min=-1.330047
max=1.784977 nonfinite=0` - which is the same evidence Phase 14 used and the
same standard.

**1080p (T=8160), same command at `streamed 48 8160 1024 1`**, against Phase
15's own published number for the identical shape:

| | before (Phase 15) | after |
|---|---:|---:|
| wall, 48 layers, cache-hit call | **246.7 s** | **151.40 s** (1.63x) |
| GPU kernel time | 47 313 ms | 47 313 ms |
| **GPU compute share of wall** | **19.2%** | **31.3%** |
| resident blocks | 0 | 0 (the budget correctly declines at this shape) |
| peak VRAM | 16 650 MiB | 23 878 MiB |

Output statistics identical to Phase 15's own published line for this shape -
`len=1044480 mean=0.043124 std=0.698590 min=-1.510173 max=1.889706
nonfinite=0` - which is bit-identity re-confirmed at a token count the tiny
gates never reach. The peak is higher and the reason is named rather than
glossed: the per-forward `[t, 9*dim]` adaLN table is 1.2 GB at this token
count, and wgpu's pool grows into whatever is left. It fits, with 698 MiB
spare, and a benchmark pass should know that number is thin.

**GPU compute is now the largest single item in a real forward.** The runner-up
is `ada_layer_norm_single`'s own model-level host GEMM at 10.2 s (20%), which
Phase 14 already published an available bit-identical AVX2-across-`M` fix for
and which is now the next target; then the per-forward adaLN table upload
(7.5 s), which is only that expensive because residency leaves the card nearly
full.

#### 4 - the Vulkan backend: budgeted for, and separately broken

The 2.00x cost above is wgpu's, not the hardware's, so the budget is
**backend-aware**: `activation_reserve_bytes(t, backend)` reserves 2.0 MiB/token
on `backend-vulkan` (which recycles transients explicitly at every flush and
measures 1.00x resident) against 5.2 MiB/token on wgpu, so the SAME card is
budgeted the **whole 48-block window at 720p** there instead of 20. Gated
(`the_slot_policy_is_bounded_by_the_layer_count_and_shrinks_as_tokens_grow`
asserts the Vulkan budget is strictly larger and full at 720p).

That budget could not be confirmed on real weights, because **the LTX int8
tier does not currently run on `backend-vulkan` at all**, and that is
pre-existing rather than anything this phase introduced. Measured, in
increasing order of decisiveness:

* `BRAIN_DEVICE=vulkan ltxv_bench streamed 48 3520 1024 1 1`, 48 resident:
  `GPU device lost while waiting for a submit to complete` at 19192 MiB.
* The same command with residency OFF (`BRAIN_LTXV_RESIDENT_BLOCKS=0`): the
  same panic at **6769 MiB**, nowhere near any memory limit.
* `streamed 2 512 128 1 1` - two layers, 512 tokens: the same panic.
* `BRAIN_DEVICE=vulkan cargo test -p brain-ltxv --test int8_compute`: both
  tests, neither of which touches `adaln_row`, `devres` or any of this phase's
  code, fail with the same panic.

So the failure is in the backend and predates this work. Recorded as a gap
below rather than chased here. The consequence worth stating: on hardware where
`backend-vulkan` does work, this phase's residency window should pay
considerably MORE than the 720p numbers above show, because the 2.00x and the
greedy pool are both wgpu's.

#### 5 - correctness

Bit-identity is the whole contract: this phase changes WHEN and BY WHAT ROUTE
bytes reach the card, never what a kernel reads. `adaln_row` reproduces
`add_table`'s operand order (`tbl[i] + tab[..]`) and `slice_mod`'s `1.0 + x`
exactly - one f32 add, then one more - so it is bit-identical rather than
close.

New: `crates/ltxv/tests/device_residency.rs` (5 + 1 `#[ignore]`d real-weight),
4 in `ltxv::devres`. Compared on BIT PATTERNS, never `==` on `f32`:

* `a_device_resident_forward_is_bit_identical_to_the_streaming_one` - three
  forwards at DIFFERENT latents through one resident session vs. the transient
  path, plus the residency assertions that stop it passing vacuously (every
  block uploads exactly once for the whole session; every visit of every
  forward is a hit) and an assertion that the three forwards really do differ.
* `on_device_modulation_is_bit_identical_to_the_host_combine_and_slice` - the
  gate the other two CANNOT provide, since both run the device path: the
  chained stack against the EAGER `LtxDit::forward_q`, which goes through
  `add_table` + `slice_mod` on the host. **Mutation-verified twice**: swapping
  two of `MOD_ROWS`' nine row indices fails it, and deleting the `plus_one` add
  fails it at element 0 (`-3.613038e-2` vs `-3.761481e-2`).
* `a_narrow_window_is_bit_identical_and_still_uploads_less_than_streaming` -
  the graceful-degrade path is production code, so it is gated: bit-identical,
  and the upload count is asserted EXACTLY (`pinned + tail * passes`), not
  bounded.
* `a_zero_slot_session_falls_back_to_streaming_and_is_still_exact`.
* `the_slot_policy_never_over_promises` + the `devres` unit tests - the budget
  must be monotone in the token count, never exceed the layer count, reach zero
  rather than a negative count, and budget Vulkan strictly above wgpu.

Every pre-existing gate re-run and green: the FULL `cargo test --release
-p brain-ltxv -p brain-residency -p brain-weightset --lib --tests`, including
`dit_parity`, `av_dit_parity`, `host_forward_parity`, `streamed_vs_eager_real`,
`connector_real_parity`, `block_weight_cache`, `vae_parity`, `vae_tiling`,
`na_decoder_parity`, `upsampler_parity`, `av_shard_2gpu_real`, `cfg_parallel`
and `anchor_real`.

#### 6 - the graceful path, and what is charged against what

Nothing here may abort where it could degrade:

* The slot budget is `(usable_vram - activation_reserve(t, backend)) /
  cached_block_bytes`, clamped to the layer count. `usable_vram` is the
  `memauth` authority's live headroom when `--limit-vram-total` is published
  and the card's own DEVICE_LOCAL heap otherwise.
* Under a published ceiling the headroom is **divided by the schedulable card
  count**, because the flag is a process-wide TOTAL and the two concurrent CFG
  branches build their sessions at the same moment on two threads - without the
  division both would plan a full window and the loser would be refused
  mid-upload.
* `can_charge_a_block` re-checks the LIVE headroom immediately before each
  ~270 MB weight upload and degrades that block to streaming (traced at `warn`,
  counted in `ResidencyStats::refusals`) rather than letting `Gpu::storage`'s
  infallible facade panic. Free when no ceiling is published.
* **The window never takes more than a quarter of the card**, whatever the
  token count says, and that cap was found the hard way rather than reasoned.
  A generation is not just its denoise loop: `pipeline::generate` runs the
  Gemma-4 text encode before it and the VAE decode after it, each on its OWN
  `Gpu`, and a fresh wgpu device cannot reuse the pool a dropped one left
  behind - so weights this loop released are not usefully free to the next
  stage. At a SMALL token count the reserve above is tiny, so the policy
  granted all 48 blocks (~13 GB), the denoise loop finished normally, and then
  the VAE decode's own device aborted with `wgpu error: Out of Memory` at
  **24211 MiB of a 24576 MiB card** on a 9-frame 64x64 clip - a shape with no
  memory problem of its own whatsoever. Caught by re-running Phase 15's
  `cfg_parallel.rs` real-weight gate, which is the only thing in this crate
  that drives a whole generation end to end; the isolated forward benchmarks
  every other number here comes from cannot see it, because they stop before
  the decode.
* Fewer slots than layers is not a failure mode, it is the design: `CyclicScan`
  pins the prefix and streams the tail, still bit-identical. Zero slots is
  exactly the pre-residency behaviour.
* `BRAIN_LTXV_RESIDENT_BLOCKS=<n|0>` overrides the computed count - the bisect
  handle every measurement in this entry used.

#### 7 - what was explicitly NOT done

* **Device residency for the fp32 `LtxDit` path** (`forward_blocks`/
  `forward_blocks_q`). Those are the parity-bisect entry points: they retain
  every block's output and every tap by design, are never on the production
  path, and `LtxDit` is `!Sync`. Making them resident would risk the parity
  ladder to speed up something no generation runs.
* **DFR.** Checked rather than assumed: `crates/ltxv/src/dfr.rs` is pure host
  geometry with no device code at all, and `pipeline::generate_dfr` builds a
  tiny random-weight `LtxDit`, never `forward_q_streamed`. It does not have
  this pattern, so nothing was changed there.
* **Moving `ada_layer_norm_single`'s model-level table to the GPU.** It is now
  the largest non-compute item (10.2 s, 20%) and a GPU GEMM would be ~0.3 s AND
  delete the 7.5 s upload - but it reassociates, so it is not bit-identical and
  is a numerics decision, not an optimization. Phase 14's own available
  bit-identical AVX2-across-`M` win is the in-scope next step.
* **Deduplicating identical timestep rows.** Phase 14 tracked it; it would
  collapse both the 10.2 s host GEMM and the 7.5 s upload to near zero,
  bit-identically, since a real step has 1-2 distinct rows. Still tracked, now
  with a second reason to want it.
* **Making `shift_kv`/`one_plus_scale_kv` device-resident.** They are per-block
  `[ctx_len, dim]` broadcasts (33.6 MB/block, ~1.6 GB/forward - the ~6% of the
  modulation traffic this phase did not remove). Resident they would cost
  1.6 GB of VRAM, and deriving them with `adaln_row` against a zero table would
  turn `v` into `v + 0.0`, which differs from `v` for `-0.0`. Left alone rather
  than trade a bit-identity guarantee for 6%.
* **Fixing `backend-vulkan`.** See item 4: real, pre-existing, out of scope.


### Phase 19 - the clip stops falling apart before it ends

A real 1080p generation - real Q8_0 22B DiT, real Gemma-4 encoder, real conv
VAE, `--start-frame` a real photo, guidance 3.0, seed 42 - produced a clip
that is correct for its first ~0.7 s and whose last several frames are
visibly warped and smeared. Reported directly, at 1080p only: the identical
prompt, image, seed and pipeline at 720p and 512x512 are clean. Two bugs, one
of them pre-existing and silently passing its own parity gate.

#### 0 - the observable, because none of the existing ones could see this

Every gate in this crate was green through it. `vae_tiling`'s were green
because the decoder is innocent. `motion_real` was green because the clip DID
move - `peak_excursion` is a FLOOR, and a clip that runs away from frame 0
scores BETTER on it the worse it disintegrates. `dit_parity`,
`host_forward_parity`, `streamed_vs_eager_real` were green because a single
forward against a golden is correct at any token count. The output statistics
(min/max/std/nonfinite) were all normal.

What separates the two cases is purely TEMPORAL, and it is one number:
**`clipmetric::blowup_ratio`**, the largest frame-to-frame difference over the
MEDIAN one, on a fixed 128x128 box downsample (so a 720p run and a 1080p run
are comparable). A clip with steady motion holds it near 1 *whatever its own
pace is*, because the median tracks that pace; a clip that comes apart at one
point pushes it into double digits. Scored across every real clip this
session had on disk, it separates cleanly - everything healthy 1.02-1.06,
one outlier at 14.66 - and it costs nothing to compute.

#### 1 - isolating the decoder from the latent

The first question was whether Phase 16's tiled decode was at fault (1080p is
the only shape that needs it) or the DiT's own latent. Four independent
answers, all pointing the same way:

1. **Geometry.** At 25 frames/1920x1088 the auto plan is 3x3 SPATIAL with the
   temporal axis untiled - `AUTO_FRAMES = (80, 24)` is a 10-latent temporal
   tile over 4 latent frames, so `split_temporal_causal` returns one interval
   mapping to all 25 output frames with an all-ones mask (already gated by
   `a_25_frame_clip_needs_no_temporal_split`; the production log says
   `tiles=9`). A spatial-only blend is time-invariant. It cannot produce a
   defect confined to the last 7 of 25 frames.
2. **Measured, on real content.** The same real 720p latent decoded whole and
   tiled: blowup 1.04 vs 1.03, agreeing to <= 0.06 in 0-255 units on every
   frame. The tiled path introduces no temporal instability.
3. **Spatial signature.** Comparing the worst frame against the last clean
   one, per 64-px column band, the tile seam bands (px 704-767, 1408-1471)
   measure 38.2 and 32.8 against a band median of 33.2; the row seams
   (384-447, 768-831) measure 36.7/37.3 against a row median of 34.5. The
   LARGEST band, 47.3, is at columns 448-511 - the interior of tile 0. The
   damage tracks the content, not the geometry.
4. **The decisive one.** Dump the real final latent
   (`BRAIN_LTXV_LATENT_DUMP`), take a centre crop small enough to fit, and
   decode it through the WHOLE path - the tiled decoder never runs. The
   blowup reproduces at **17.43** (higher than the full clip's 14.66, because
   the crop is centred where the damage is worst). Then decode the same crop
   with latent frame 3 replaced by latent frame 2: **1.31**. The defect
   survives deleting the decoder and vanishes when one latent frame's content
   is swapped out.

So: the latent. And its statistics name the failure exactly. Per-latent-frame
standard deviation and adjacent-frame distance, same prompt/seed/image
throughout:

| request | tokens | LF0 | LF1 | LF2 | LF3 | adjacent \|delta\| |
|---|---:|---:|---:|---:|---:|---|
| 960x544 | 2040 | 1.070 | 0.960 | 1.013 | **1.069** | 0.535, 0.451, 0.423 |
| 1280x704 | 3520 | 1.074 | 0.987 | 1.032 | **1.077** | 0.467, 0.419, 0.405 |
| 1920x1088 | 8160 | 1.067 | 0.977 | 1.006 | **0.911** | 0.404, 0.386, **0.630** |

A healthy latent's last frame recovers to ~1.07 and its adjacent distances
DECREASE monotonically. The 1080p latent inverts both: the last frame's
variance collapses while it sits 1.63x further from its neighbour than the
pair before it. In pixels that is a discontinuity plus progressive smearing -
high-frequency energy (mean |Laplacian|, 512x512 gray) is flat at 14.8-15.4
for frames 0-17 and then falls 11.5, 11.8, 10.9, 11.3, 10.3, 10.8, **8.7**.
Detail LOSS, not residual noise, which is what ruled out "under-denoised".

Two things this ruled out that were worth ruling out. **The device-residency
window is not involved**: the run's own log says `device residency planned
slots=0` at this token count, so Phase 18's `BlockWindow`/`CyclicScan` never
engages here. **There is no phantom end-frame conditioning**:
`conditioning_block_count(start=true, end=false)` is 0, the only two uses of
`frames - 1` sit inside `end_frame.is_some()` branches, and the run logs
`appended_blocks=0`, `tokens == base_tokens == 8160`, `frozen_tokens=2040` -
exactly one latent frame, the FIRST. The end of the clip is not conditioned,
handled specially, or indexed anywhere; only its generated content is wrong.

#### 2 - root cause A: one stage where the reference runs two

`ltx_pipelines.distilled.DistilledPipeline.__call__` never runs
`DISTILLED_SIGMAS` at the requested resolution. It runs it at
`width // 2, height // 2`, carries that latent up with the spatial x2
upscaler, and spends three more DETERMINISTIC steps
(`STAGE_2_DISTILLED_SIGMAS`, from sigma 0.909375) refining at full size -
unconditionally, for every shape, with the same weights and the same LoRAs in
both stages. So the distilled table is only ever asked to build structure
from noise at a QUARTER of the output's tokens, and upstream's largest
shipped preset (`LTX_2_3_HQ_PARAMS`, 1088x1920 out) puts that at 544x960 =
2040 tokens.

This port ran ONE stage at the full requested resolution. That is fine while
the token count stays near what the table was distilled for and is not fine
past it. The bracket, everything but the resolution held fixed:

| request | video tokens | blowup ratio |
|---|---:|---:|
| 512x512 | 1024 | 1.06 |
| 960x544 | 2040 | 1.03 |
| 1280x704 | 3520 | 1.04 |
| 1600x896 | 5600 | 1.04 |
| **1920x1088** | **8160** | **14.66** |

`SINGLE_STAGE_MAX_TOKENS = 6144` is set BETWEEN the largest measured-good
count and the measured-broken one - the same discipline
`vae3d::WHOLE_DECODE_MAX_PIXELS` uses - so **every shape this port already
ran keeps its exact behaviour** and only the one that disintegrates changes
path. Where in `(5600, 8160)` the real cliff sits is not measured and the
constant does not pretend to know; it only has to separate them.
`should_two_stage` additionally requires both axes on a multiple of 64 (so
halving lands on the VAE's 32-px stride - upstream asserts the same in
`assert_resolution(..., is_two_stage=True)`) and a real distilled config.
`BRAIN_LTXV_TWO_STAGE=1`/`0` overrides.

`generate`'s denoise body became `denoise_stage`, resolution-parametric with
an optional re-noised seed, so one body serves a single-stage run and both
stages of a two-stage one. **A single-stage run is bit-identical to before
this phase**: the stage seed salt is 0, so `seeded_noise(o.seed ^ 0)` and
`o.seed ^ 0x4e_4f_49_53_45 ^ 0` are the original expressions. That is not
only an argument from the code - a full real 25-frame 1280x704 generation
(real 22B Q8_0 DiT, real Gemma-4 encoder, real conv VAE, same
`--start-frame`, seed 42, guidance 3.0) run before and after produces a
final latent with the **same md5**, `998e25a13b5c59b93515402ce4fce990`, and
the same clip metric to three decimals (median 3.401, max 3.580, blowup
1.05).

#### 3 - root cause B: the latent upscalers were never un-normalized around

Two-stage alone did not fix it. It removed the disintegration - frames 17-24
went flat - and replaced it with a clip blurred to **2.9** mean
high-frequency energy against ~13.7 for a healthy one, whose final latent had
LF0 at std 1.067 (the pinned still) and LF1-3 at 0.71/0.64/0.62.

`ltx_core.model.upsampler.model.upsample_video`:

```python
latent = video_encoder.per_channel_statistics.un_normalize(latent)
latent = upsampler(latent)
latent = video_encoder.per_channel_statistics.normalize(latent)
```

Both latent upscalers run in **raw VAE latent space**, not the normalized
diffusion space, and upstream's `VideoUpsampler` builds a whole video ENCODER
alongside the upscaler for no other purpose than to reach those statistics.
This port called `LatentUpsampler::upsample` directly. Measured on a real
stage-1 latent through the real x2 spatial upscaler:

| per-latent-frame std | LF0 | LF1 | LF2 | LF3 |
|---|---:|---:|---:|---:|
| input (normalized) | 1.070 | 0.960 | 1.013 | 1.069 |
| bare call (WRONG) | 0.504 | 0.524 | 0.530 | 0.465 |
| un-normalized around | **1.014** | **0.919** | **0.994** | **1.074** |

**Why the existing gate could not see it, which is the transferable lesson.**
`upsampler_parity.rs`'s `report` computed `max_abs`, PRINTED it, and asserted
on **cosine alone** - and cosine is scale-invariant, so a port returning
exactly `k * golden` passes at cosine 1.000000000 for any `k`. The port
itself is exact (max_abs 6.7e-6..3.4e-5 on every tap, re-confirmed); what was
wrong was the space it was called in, and the one observable that would have
caught it was the one being printed instead of asserted. `report` now asserts
`max_abs <= 1e-3` as well.

This bug was **pre-existing and not confined to the new path**: all three
upscaler call sites had it, two of them in `generate_dfr` (the spatial video
and keyframe-slot upscales) plus its temporal rounds. All three now go
through one `upsampler::upsample_video` helper, and
`vae3d::per_channel_statistics` is exposed for it. `generate_dfr`'s VAE
import moved ahead of its first upscale so the statistics are available
there; the decode reuses it, so the file is still read once.

#### 4 - measured, end to end, on the real bug shape

Same command both ways - real Q8_0 22B DiT, real Gemma-4 encoder, real conv
VAE, `--start-frame` the same real photo, `--frames 25 --width 1920 --height
1088 --fps 24 --guidance 3.0 --seed 42`, two Tesla P40s, wgpu:

| | before (one stage) | two stage, upscaler not un-normalized | **after (both fixes)** |
|---|---:|---:|---:|
| **blowup ratio** | **14.66** | 8.64 | **1.02** |
| frame-to-frame, first half | 1.4-1.6 | 3.0-7.0 | 9.5-11.3 |
| frame-to-frame, last 7 | **15.9 -> 23.2** | 3.1-4.5 | 11.0-11.4 |
| median frame-to-frame | 1.58 | 3.59 | **11.11** |
| high-frequency energy, mean f1-24 | 13.81 | 2.91 | **12.20** |
| high-frequency energy at f24 | **8.7** | 2.5 | **11.9** |
| peak VRAM | 16651 MiB | - | ~9.6 GiB |
| wall | 1263 s | - | 953 s |

The reproduction is exact: re-running the reported recipe on this session's
own binary reproduced `median 1.580, max 23.171, blowup 14.66` - the same
numbers to three decimals as the clip that was reported.

Three things in the "after" column beyond the disintegration being gone.
The clip MOVES - median frame-to-frame went 1.58 -> 11.11, i.e. the
near-frozen first 17 frames were part of the same defect, not a separate
one. Sharpness is flat across the clip (13.4 / 12.0 / 11.9 at frames 1 / 12 /
24) rather than collapsing at the end. And it is 1.3x FASTER, because eight
steps at 8160 tokens costs more than eight at 2040 plus three at 8160.

The middle column is worth keeping: two-stage ALONE traded the
disintegration for a clip blurred to a fifth of the reference's detail, and
that is what pointed at the second bug. Shipping after the first fix would
have replaced a visible defect with a subtler one.

Confirmed by eye as well as by metric, which is the standard this defect
earned: the last frame's "P40" lettering is crisp and legible, the wings are
intact, the dog is anatomically coherent with readable collar and fur, and
the road markings are sharp - against garbled lettering, smeared wings and a
disintegrated animal before.

#### 5 - gates

Weight-free, always run:

* `pipeline::tests::the_stage_policy_matches_the_shapes_that_were_measured` -
  the four measured-good shapes must keep taking one stage, the measured-bad
  one must take two. Guarded against a stray `BRAIN_LTXV_TWO_STAGE` export.
* `pipeline::tests::the_stage_policy_refuses_a_shape_it_cannot_halve_and_a_
  config_it_does_not_apply_to` - the non-token conditions.
* `clipmetric`'s three unit tests - the metric itself, against a steady pan,
  a late blowup, two resolutions of the same content, and a static clip
  (where the ratio is undefined and must read 1.0, not "unstable").
* `latentdump`'s two - the dump round-trips bit-for-bit and rejects a file
  that is not one.

Real weights, seconds:

* `upsampler_parity::the_upscaler_is_un_normalized_around_exactly_as_the_
  reference_does_it` - `upsample_video` is exactly
  `normalize . upsample . un_normalize`, compared on BIT PATTERNS, AND
  differs measurably from the bare call (the half that stops the gate passing
  if the helper ever quietly becomes `upsample` again). Written as the exact
  composition rather than as a variance claim on purpose: the variance
  invariant only holds for a latent that really lives in the diffusion
  distribution, and an i.i.d. draw is not one - asserting it on synthetic
  input fails for a reason that has nothing to do with the bug.
* The `max_abs` bound added to `report`, above.

Real weights, minutes, `#[ignore]`d:

* `clip_stability_real::a_real_1080p_clip_does_not_disintegrate_before_it_
  ends` - one real 1080p generation, `blowup_ratio < 4.0`, plus a floor on
  the median so a frozen clip cannot pass vacuously. `#[ignore]`d for cost,
  not confidence: the defect only exists above the token ceiling and a token
  count that large IS a full generation - there is no small shape that
  reproduces it, because the token count is the variable.

#### 6 - tooling this needed, kept

`BRAIN_LTXV_LATENT_DUMP=<path>` writes the final latent from `decode_video`
before any decoder touches it, and `ltxv_bench decode <latent>
<whole|tiled> [h0 h1 w0 w1]` re-decodes a dump through an explicitly chosen
path over an optional latent-cell crop, printing the per-latent-frame
statistics and the frame-to-frame curve.
`BRAIN_LTXV_DECODE_LF_SUBST=dst=src` overwrites one latent frame with
another and `BRAIN_LTXV_DECODE_UPSAMPLE=<path>` runs the real upscaler first.
Together those turned every question in item 1 from a 22-minute regeneration
into a one-minute decode, and they are why the crop-whole-decode experiment
was possible at all at a shape whose whole-clip decode does not fit.

#### 7 - what a benchmark pass should know

* **1080p is a two-stage shape now.** It needs
  `BRAIN_LTXV_UPSAMPLER_SPATIAL`; without it `generate` errors and names the
  variable rather than producing a broken clip.
* **It is also cheaper.** Stage 1 at 2040 tokens plus three refinement steps
  at 8160 beats eight steps at 8160.
* **The 25-frame 1080p single-stage shape was marginal on VRAM** and this
  removes that too: peak fell from 16651 MiB to ~9.6 GiB. Three of six
  attempts at the old shape died with `wgpu error: Out of Memory` at up to
  23121 MiB of 24576 while steps 2-7 sat at 11-12.3 GiB - wgpu's greedy pool,
  which Phase 18 already recorded as elastic and non-backing-off at this
  token count. A run that OOMs at this shape is not necessarily a leak.
* **Score clips with `blowup_ratio`, not by eye.** It is one number, it is
  resolution-independent, and it is the only thing in this crate that sees
  this class of defect.

### Phase 20 - the prompt starts reaching the picture

Plain text-to-video ignored its prompt. Not "loosely followed" - ignored:
two unrelated captions, everything else identical, decoded to the same
picture of the same person. Reported against the real stack (real Q8_0 22B
DiT, real Gemma-4 12B Q8_0 encoder, real conv VAE) and reproduced here on the
same box, two Tesla P40s.

The defect was never in this crate's conditioning plumbing. It was in
`crates/gemma4`, in the one module of the text path that had been built from
a checkpoint header instead of from source.

#### 0 - the observable, and what it ruled out first

The report framed this as a divergence between the image-conditioned path
(`frozen: Some(..)`, which appeared to follow prompts) and the plain one
(`frozen: None`, which did not). Reading `generate`/`denoise_stage`/`denoise`
refutes that outright: `ctx_cond`/`ctx_uncond`/`context_valid`/`context_len`
are built ONCE in `generate`, before the stage plan exists, and reach
`StepInputs` through the same fields whatever `o.start_frame` is. Image
conditioning changes `latent`, `timesteps`, `positions`, `keyframes_mask` and
`denoise_t_count`; it cannot change the text context, and no `frozen.is_some()`
test guards anything on the context path. `context_stub` is likewise selected
only by `paths.text_encoder.is_none()`, which is a checkpoint being absent,
not a start frame.

So the question became "does the context discriminate between prompts AT
ALL", and that is answerable without a GPU, because `crate::text_cache`
already writes every encoded context to disk. Over the session's cached
contexts (`[1024, 4096]` each, 40-ish valid rows):

* every pair of DISTINCT caption token rows within one prompt sat at cosine
  **0.9963**;
* the mean row of *"a bright red vintage convertible car driving fast through
  an empty desert highway at sunset"* and the mean row of *"a slow pan across
  a snowbound pine forest"* sat at cosine **0.99984** of each other;
* `||mean row||` was **975** against a mean residual of **49**.

Every row of every prompt's context was the same vector plus 5% of noise.
The DiT's cross-attention was being handed a constant, so it sampled its
unconditional prior - which is exactly what "a generic cinematic drama scene
that has nothing to do with the prompt" is.

Confirmed end to end before touching anything: the two prompts above at
512x512 / 25 frames / seed 7 / guidance 1.0, no image conditioning, produced
the same close-up of the same woman's face, whole-clip mean absolute pixel
delta **6.67** of 255.

#### 1 - the root cause, and how it got in

`gemma4::AggregateEmbed::forward` - LTX's own
`text_embedding_projection.video_aggregate_embed`, `Linear(3840*49 -> 4096)` -
concatenated the 49 raw hidden states per token and applied the linear.

`tools/goldens/gemma4_dump_reference.py` said in as many words why: "there is
no reference implementation to import ... This is a documented judgment call,
not a confirmed detail: the real module's internal structure (whether it has
a bias, an extra norm before the linear, etc.) is not derivable from a
tensor-name/shape header alone". The premise was false. The module is
`ltx_core.text_encoders.gemma.feature_extractor.FeatureExtractorV2`, it is in
`resources/ltxv/source/`, and it does three things the guess did not:

1. **per-token, per-STATE RMS normalization** over the hidden axis
   (`norm_and_concat_per_token_rms`: `x * rsqrt(mean(x^2) + 1e-6)`, no
   learned weight, no mean subtraction, independently for each of the 49
   states);
2. **an interleaved column order** - `torch.stack(hidden_states, dim=-1)`
   gives `[T, D, L]` and `.reshape(B, T, D*L)` flattens it `d`-major,
   `l`-minor, so input column `d*n_states + k` is state `k`'s coordinate
   `d`. The port had `k*hidden + d`: `n_states` contiguous blocks. That is a
   permutation of the weight matrix's columns, on its own enough to make the
   output unrelated to the caption;
3. **`_rescale_norm`** - `* sqrt(out_dim / hidden_size)`, 1.0328 at the real
   config.

(1) is why the output was near-constant: the 49 raw Gemma states differ in
magnitude by orders of magnitude, so an un-normalized concatenation is
dominated by whichever states are largest, and their token-to-token variation
is small next to their common component. (2) is why what little signal
survived was not the caption's.

A fourth, smaller divergence on the same path: `LTXGemmaTokenizer.
tokenize_with_weights` prepends `<bos>` unconditionally and says why in its
class doc - "Gemma 3 already emits it via post_processor; Gemma 4 does not,
so we prepend". `data::qwen_tokenizer::QwenBpe` is deliberately
template-free, so nothing added it and every prompt was encoded one token
short, each caption token sitting one position early. Confirmed from the
cached contexts before it was fixed: row 0 differed between prompts starting
"a ..." and "A Belgian ...", which a shared leading BOS could not do.

#### 2 - the fix

* `gemma4::AggregateEmbed::forward` applies the normalization, the
  interleaved order and the rescale. The reference additionally zeroes padded
  positions; this function only ever sees a prompt's real tokens (`ltxv::
  pipeline` pads afterwards, and the connector overwrites the padded tail
  with its learnable registers), so there is no mask to thread.
* `ltxv::pipeline::real_text_context`'s `tokenize` strips the prompt and
  prepends the tokenizer's own `<bos>`, looked up by content. A checkpoint
  that declares none keeps the old behaviour and says so.
* `tools/goldens/gemma4_dump_reference.py` grew `feature_extractor_v2`, a
  transcription of the reference's input transform, and its module doc now
  records that the guess was wrong rather than leaving the old justification
  standing.
* `text_cache::Key` grew `encode_revision` (`ENCODE_REVISION = 2`). Every
  other key field describes an INPUT; this one describes the function. Without
  it the disk cache - whose whole header is an argument about never serving a
  wrong context - would have gone on serving pre-fix contexts for every prompt
  already encoded this session, against an unchanged checkpoint.

#### 3 - after, on the same box

Same two prompts, same seed, same shape, nothing else changed:

| | row-to-row cos within a prompt | cos(mean_A, mean_B) | `\|\|mean row\|\|` vs residual | clip delta |
|---|---:|---:|---:|---:|
| before | 0.9963 | 0.99984 | 975 vs 49 | 6.67 |
| after | 0.4676 | 0.8828 | 202 vs 219 | **100.57** |

The caption went from a 5% residual to the dominant component of its own
context.

What the clips show, looked at rather than scored:

* *"a bright red vintage convertible car driving fast through an empty desert
  highway at sunset, dust clouds behind it"* - a red vintage convertible seen
  from behind, driving away down an empty two-lane desert highway, the sun low
  on the horizon between distant hills, dust kicked up along the roadside.
  Every clause of the prompt is in the frame.
* *"a slow pan across a snowbound pine forest"* - deep snow, pine trunks,
  snow-laden branches, a slow lateral drift.

Before the fix both of these were the same woman's face.

#### 4 - gates

Weight-free, milliseconds:

* `gemma4::model::the_projection_rms_normalizes_every_layer_slice_and_rescales`
  - scaling one token's one state slice by any positive factor must not change
  that token's output (the defining property of the per-state norm, and
  exactly what a plain concatenate-then-project lacks), the surviving scale is
  `sqrt(out_dim/hidden)`, and the one-hot weight rows pin the interleaved
  column order.
* `gemma4::model::an_all_zero_layer_slice_stays_finite` - the `+1e-6` inside
  the rsqrt, which is load bearing for a zero state.
* `gemma4`'s `gemma4_tiny_matches_reference` `aggregate_out` tap now pins the
  real formula rather than the guess, at the suite's own cosine 0.999999 bar.
  The fixture was regenerated; the dumper is the source of truth for it.
* `text_cache::every_key_field_changes_the_digest` covers `encode_revision`.

Real weights, `#[ignore]`d:

* `prompt_adherence_real::two_unrelated_prompts_do_not_produce_the_same_clip`
  - two full real generations at 384x192 / 9 frames, no image conditioning,
  guidance 1.0 so the clip is the conditional branch's own answer and not a
  CFG difference. Measured 80.04 against a floor of 20.0. This is the
  perceptual half, and it is what proves the weight-free gates above are
  testing something a viewer can see.

#### 5 - what this says about the rest of the port

Every parity gate in `crates/ltxv` was green throughout. They had to be: a
DiT forward against a golden is correct at any context, and the context was
structurally perfect - right shape, right validity mask, right padding, right
connector routing. It just carried no information. Shape-level parity cannot
see a semantic-level defect, and "the module's internals are not derivable
from the header" is a claim to go and check in the reference, not a licence to
guess. The reference was already vendored in `resources/ltxv/source/` the
whole time.

### Phase 21 - the upscaler stops being reachable only from inside a generation

The spatial x2 latent upscaler has been in this crate since the DFR milestone
and load-bearing since Phase 19, where it became the middle of the reference's
two-stage generation. But it was reachable only from inside `generate`: there
was no way to point it at a clip that had already finished rendering. Real
usage wants exactly that - a batch of segments rendered overnight at 1280x704,
and a decision the next morning about which ones are worth 2560x1408.

`brain ltxv upscale --input clip.mp4 --output-path clip_2x.mp4 --prompt "..."`
is that entry point. What it does is not new work: VAE-encode, official x2
latent upscale, refine on `STAGE_2_DISTILLED_SIGMAS`, VAE-decode. The point of
the phase is that it is not a SECOND copy of that.

#### 0 - the shared unit, chosen so the Phase 19 defect cannot come back twice

Phase 19 closed two bugs, and the second one - the latent upscalers called in
the wrong latent space, costing half the latent's variance - is the kind that
survives a green parity suite. It had three call sites when it was found, and
all three were wrong. A fourth call site written from scratch here would have
been a fourth chance to get the sandwich wrong, in the one place with no
golden to check it against.

So `generate`'s two-stage tail was extracted whole. `upscale_and_refine(sc,
&Refine { .. })` is the un-normalize/upsample/re-normalize sandwich plus the
refinement `denoise_stage`, and BOTH the two-stage generation path and the new
standalone command call it - the two-stage branch of `generate` is now that one
call. The same extraction pulled out `build_denoiser` and `build_context`
verbatim, since the standalone path needs the same DiT and the same text
context and neither is upscale-specific. Nothing about WHAT any of them does
changed; `denoise_stage`'s own seed salt (`0x5332`), eta (0) and schedule are
the expressions Phase 19 left.

Two supporting changes fell out of it, both narrowing rather than widening:

* `LtxVaeTiledDecoder` borrows its weights instead of owning them, and so does
  `decode_video`. A standalone upscale decodes one clip per segment against one
  set of VAE weights; owning them would have cost a ~3 GB host copy per
  segment. The tiled path already held them for its whole lifetime, so this is
  strictly less copying, not more.
* `Paths::resolve` takes the spatial upscaler as a fourth optional argument.
  On `t2v` it stays environment-only (there it is a fixed member of the
  checkpoint set, not a choice); `upscale`, whose whole subject is that
  network, takes `--upsampler-spatial`.

#### 1 - the length ceiling, and why it is segmentation rather than an error

An upscaled clip has FOUR times the video tokens per frame of its input. A
105-frame 1280x704 clip refines at 12320 tokens today; the same clip upscaled
is 49280. That is not a quality question (`SINGLE_STAGE_MAX_TOKENS` is about
building structure from noise, and refinement starts from content) - it is a
per-forward allocation question. The adaLN table alone is `[t, 9*4096]` fp32 =
147456 bytes per token, against a `max_storage_buffer_binding_size` this box's
Tesla P40 reports as **2047 MiB**, which one table crosses at t ~= 14556. That
figure is the adapter's own, logged by `backend-wgpu` on every run, not a
constant anyone wrote down.

`REFINE_MAX_TOKENS = 12288` sits below the crossover with room for the other
per-forward slabs and above 8160, the largest refinement token count this
crate has a recorded real run at (Phase 19's two-stage 1080p). It is DERIVED,
not measured, and its doc says so. It bounds only the new command; `generate`'s
stage 2 is untouched and still runs whatever the requested resolution implies.

Past it, `refine_segments` splits the clip rather than refusing it. The
overlap is not a tuning choice: every clip the causal VAE can represent has
`1 + 8k` frames, so `n` DISJOINT segments of that shape sum to `n + 8*sum(k_i)`,
which is a legal `1 + 8K` clip only for `n == 1`. Sharing exactly one frame at
each boundary closes it for any `n` - `sum(1 + 8k_i) - (n-1) = 1 + 8*sum(k_i)` -
and the shared frame is re-rendered by the later segment, whose refinement pass
is the one it shares with the frames that follow it. Segments are split as
evenly as the arithmetic allows, so a clip never ends on a 9-frame stub refined
with less temporal context than everything before it.

**The seam is real, is not blended, and is reported.** Each segment refines
independently, so fine detail can step where two meet. That is a bounded,
visible artefact announced on stderr and in the tracing log, which is a
different class of thing from the silent end-of-clip disintegration Phase 19
documents - and it is the honest trade for refining a clip longer than one
pass can hold. A clip that fits is never split, and a shape that cannot be
split into anything runnable is refused before a single weight is read.

#### 2 - what is gated, and what deliberately is not

Two claims, in `crates/ltxv/tests/upscale.rs`:

1. **The segment plan**, weight-free and always running: a clip that fits is
   exactly one segment; a clip that does not splits into segments that are
   each `1 + 8k`, each under the ceiling, and which reassemble to exactly the
   input frames in order; an unrepresentable frame count or an ungrowable grid
   is an error, not a plan. This is the gate that stands between a long clip
   and an hour of wasted device time.
2. **The wiring, end to end** on the real conv VAE and the real x2 spatial
   upscaler (tiny random-weight DiT, CPU): a 9-frame 64x64 clip comes back
   9-frame 128x128, not flat, having passed through the `spatial upscale`
   phase and `LTX2_STAGE2_STEPS` refinement steps. 112 s on CPU.

What is NOT re-gated: the un-normalize sandwich. `upscale` reaches it through
the same `upscale_and_refine` the two-stage path uses, and it already has an
exact gate in `upsampler_parity.rs`. Asserting it again here would be gating a
second copy - and the whole point of the phase is that there is no second copy.

The CLI's own `--help`/parser self-check gained an `upscale` arm and one extra
assertion the existing ones do not have: the token ceiling quoted in the help
text must equal `REFINE_MAX_TOKENS`. A help text that promises a number the
code no longer enforces is worse than one that promises nothing.

#### 3 - what this phase does NOT claim

* **No real-weight end-to-end run.** Both cards were saturated by unrelated
  in-flight generations for the whole of this work, so every validation above
  is CPU or weight-free. `upsampler_parity`'s
  `the_upscaler_is_un_normalized_around_exactly_as_the_reference_does_it`
  could not be re-run either - it hardcodes `Some("gpu")` and aborts with
  `wgpu error: Out of Memory` under contention, which is where the 2047 MiB
  binding figure above came from. The remaining item is one real run against
  `ltx25_22b`; nothing about the code path is unexercised, but the quality of
  what comes out of it is unmeasured and is not claimed.
* **No seam measurement.** The multi-segment path's boundary artefact is
  argued from construction, not measured - measuring it needs a real clip long
  enough to segment, which needs a card. `clipmetric::blowup_ratio` is the
  metric that would see it, since a detail step at a known frame index is
  exactly the temporal discontinuity it was built for in Phase 19.
* **No capability action.** `upscale` is CLI-only. It would be the first
  action in this model to take an input BLOB (a whole video file) rather than
  parameters alone, and that action shape has not been designed. Recorded
  here and in `docs/models/ltxv.md`'s support table rather than quietly
  skipped, because "a bespoke CLI subcommand is never the only entry point" is
  this repo's own serving contract.
* **`--factor` accepts only 2.** The official checkpoint is an x2 network and
  this command runs that network; there is no resampler behind the flag and it
  says so rather than silently rounding.

### Phase 22 - a clip stops being one window long

Every generation this crate has ever run was one denoising window. The ceiling
is real hardware: the embeddings-connector and adaLN slabs of one forward cross
this box's `max_storage_buffer_binding_size` somewhere above 14000 video
tokens, which at 1280x704 is about 15 latent frames - 113 pixel frames, 4.7
seconds at 24 fps. Anything longer was produced by hand, by chaining: decode
window N, take its literal last RGB frame, VAE-encode that one frame as window
N+1's `--start-frame`, generate, concatenate the mp4s.

That chain is continuous in POSITION and discontinuous in VELOCITY, and the
difference is the whole phase. A single pixel frame says where everything is
and nothing about what was moving, in which direction, or how fast. The model
is handed a still and asked to start a clip from it, so it starts a clip from
it - picking whatever motion the prompt and the seed suggest, which is not
necessarily the motion that was already happening. Watched on real chained
output this session: motion changing at the seam, stuttering at the seam, and
one continuation that ran visibly BACKWARDS.

`brain ltxv t2v --frames 481` now generates the clip as several windows and
carries the previous window's own last LATENT frames across each boundary -
sliced out of the denoised latent before anything was ever decoded, frozen at
sigma 0 while only the new frames denoise around them. No pixel round trip at
any internal seam.

#### 0 - the mechanism was already here, pinned to one frame

`--start-frame` freezes exactly one latent frame: `denoise_mask = 0` over its
`lh*lw` tokens, the initial latent there set to the encoded still rather than
noise (`GaussianNoiser`'s `lerp(clean, noised, denoise_mask)` at mask 0 IS
`clean`), the per-token timestep therefore 0 (Phase 17's fix - the conversion
`to_denoised` performs at timestep 0 is the identity), and `post_process_latent`
re-pinning it after every step. That is `VideoConditionByLatentIndex(latent_idx
= 0, strength = 1.0)`.

The reference's own class asserts only `(B, C, H, W)` on that latent - **the
frame count is free** - and writes `clean_latent[:, start:stop]` /
`denoise_mask[:, start:stop]` over whatever range it covers. So the
generalization from "freeze 1 latent frame" to "freeze K" is not a new
mechanism; it is the same one with the count unpinned. `Stage` gained an
optional `LatentContext { chw, frames }` and `denoise_stage` one branch that
builds the mask, the clean buffer and the initial latent over `frames *
lh * lw` tokens instead of `lh*lw`. Nothing in `denoise`, `to_denoised`,
`post_process_latent` or `euler_ancestral_step` changed at all.

#### 1 - K = 8 latent frames, and the derivation that does NOT set it

The first hypothesis was that K falls out of the causal VAE: pick the decoder's
temporal receptive field, so a window's new frames decode as they would have in
one long clip. **That hypothesis is dead, and it is worth recording why.**

This checkpoint's decoder is `causal_decoder: false`. Every one of its 42
kernel-3 temporal convolutions pads SYMMETRICALLY, so decoding one latent frame
depends on frames on both sides of it. Summing the convolutions at the temporal
resolution each runs at - 6 at the latent grid, 5 at 2x, 9 at 4x, 22 at 8x -
gives `6 + 5/2 + 9/4 + 22/8 = 13.5` latent frames of radius; an exact integer
index walk gives `i-14 … i+14`. (The same summation reproduces this crate's
already-published spatial figure of ~15 cells exactly, which is what says the
method is right.) A rolling window cannot supply 14 latent frames of LOOKAHEAD
at any price, because those frames do not exist yet. No K makes a seam decode
exactly, and `LtxVaeTiling`'s own temporal overlap - 3 latent frames, upstream's
`_CONV_AUTO_FRAMES = (80, 24)` - already carries the same admission for the
same reason, with the tiling docs saying outright that covering the receptive
field is unreachable and blending is the accepted lossy trade.

What K actually decides is how much MOTION HISTORY the diffusion model is
conditioned on, which is a conditioning question, not a convolution one - and
there the reference has an answer. `packages/ltx-trainer/configs/
video_extend_lora.yaml` is the official LTX-2 video-extension recipe, and its
one video condition is:

```yaml
conditions:
  - type: prefix
    # For prefix conditioning, N latent frames correspond to (N - 1) * 8 + 1
    # pixel frames. temporal_boundary=8 means 57 pixel frames are used as prefix.
    temporal_boundary: 8
```

with validation samples that spell the same number the other way round
(`num_frames: 57`). `ltx_trainer.training_strategies.flexible.
PrefixConditionConfig` documents `temporal_boundary` as "Number of temporal
units for prefix region. For video: number of latent frames", and
`_compute_temporal_mask` places those units at the FRONT of the token sequence
(`mask[:, :num_tokens] = 1.0`). That is exactly the layout this phase builds.

So `CONTEXT_LATENT_FRAMES = 8`, `CONTEXT_FRAMES = 57`. Not 9 - "9" is the
shortest legal CLIP (`1 + 8k`, 2 latent frames), a different number that
pattern-matches. The DiT imposes no additional minimum: its attention is global
over the whole window, so a frozen prefix is visible to every generated token
however far away it sits.

#### 2 - the window arithmetic, and why it is not `refine_segments`

`refine_segments` (Phase 21) splits an existing clip into `1 + 8k` segments
that overlap by exactly ONE PIXEL frame, because `n` disjoint `1 + 8k` segments
sum to `n + 8*sum(k_i)`, legal only for `n == 1`. `window_plan` closes the same
arithmetic differently and does not need the overlap: a continuation window's
decode produces `1 + 8*(context + new - 1)` frames, of which the leading
`1 + 8*(context - 1)` belong to the carried context and are dropped, so the
window contributes exactly `8 * new`. Window 0 contributes `1 + 8*(new - 1)`.
The sum is `1 + 8*(new_0 - 1 + sum_{i>0} new_i)` for any number of windows.
The two functions share the one line that turns a token ceiling into a latent
frame count and nothing else, and they are deliberately two functions.

| | `refine_segments` | `window_plan` |
|---|---|---|
| subject | a clip that already exists | a clip being generated |
| seam carries | one re-rendered PIXEL frame | K clean LATENT frames |
| seam artefact | detail can step | motion is continuous by construction |
| sizing | segments equal | **window 0 largest**, rest equal |

That last row is the one non-obvious choice. Window 0 is the only window that
builds structure from noise with no history at all, and the only one whose
budget is not already spent on a context - and making it the largest is what
GUARANTEES it can hand its successor a full K frames. An even split cannot: at
a 15-latent-frame budget with an 8-frame context, an even three-way split gives
window 0 six latent frames and window 1 would silently carry a truncated
context. The continuation windows are then equal to each other, so a clip never
ends on a stub.

`LONGFORM_MAX_TOKENS = 13200` is the per-window ceiling. Derived the same way
`REFINE_MAX_TOKENS` is - the adaLN table is `[t, 9*4096]` fp32 = 147456 bytes
per token against 2047 MiB, crossing at t ~= 14556 - and then pinned to a
MEASURED point rather than left at the derivation: 13200 is 113 frames at
1280x704 (15 x 22 x 40), the largest window this crate has a recorded real
generation at. Every shape already known to run keeps running unsplit, and no
window is ever planned at a token count nothing has ever run at.
`BRAIN_LTXV_LONGFORM_MAX_TOKENS` overrides it, which is also how a card with a
different binding size gets a usable plan.

The context is not free and the docs say the number: a continuation window
spends `K * lh * lw` before it generates anything. At 1280x704 that is 7040 of
13200 tokens, so a window generates 5-7 new latent frames instead of 15. Long
form costs roughly twice the device time per output second. That is the price
of the history, and it is the thing being bought.

#### 3 - two-stage windows carry TWO contexts

A plan past `SINGLE_STAGE_MAX_TOKENS` takes the reference's two-stage shape,
exactly as a single-window request of the same size already does. Stage 1 runs
at half resolution, and that is where the window's structure - and therefore
its motion - is decided; a context applied only to stage 2's three refinement
steps would not carry motion at all.

The decision is made ONCE, from the plan's widest window, and applies to every
window. Deciding it per window would let one clip be built two different ways
half way through, and - worse - a single-stage window produces no
half-resolution latent at all, so a two-stage window following one would hand
its stage 1 either nothing or a stale tail from two windows back. One bool for
the plan removes the whole class.

So a continuation window freezes the previous window's stage-1 latent tail at
half resolution during stage 1, and its final full-resolution latent tail
during stage 2. Both are genuine latents that the previous window really
produced at that exact resolution. The alternative - spatially downsampling the
full-res tail for stage 1 - was rejected: there is no x0.5 latent downsampler
in the LTX-2.5 checkpoint set, and inventing one would put content into stage 1
that no LTX-2.5 component ever produced. The rolling state is therefore two
latent slabs of at most K frames each, and it does not grow with the clip's
length.

`Refine` gained a `seed_salt` alongside its existing `0x5332`, because two
windows of one clip refining with the same noise would repeat each other.

#### gates

Weight-free, always run:

* `crates/ltxv/tests/longform.rs` - `the_carried_context_is_the_references_own_eight_latent_frames`
  (K and its pixel spelling are the reference's, not drift);
  `a_clip_that_fits_one_window_carries_no_context` (13200 tokens is one
  window, nothing carried, no seam);
  `a_request_longer_than_one_window_rolls_a_latent_context_across_every_seam`
  (481 frames at 1280x704: every window under the ceiling, every window a
  legal `1 + 8k` decode, every continuation window carrying a FULL context its
  predecessor could actually supply, and the windows reassembling to exactly
  481 frames in order);
  `an_impossible_request_is_refused_up_front`;
  `the_carried_tail_is_the_previous_windows_own_last_latent_frames` (the carry
  is a slice, per channel, per frame, per cell - a transpose or an off-by-one
  cannot pass).
* `ltxv::pipeline::tests::a_frozen_prefix_of_latent_frames_survives_the_whole_trajectory` -
  the other half of the continuity chain, and the one that lives in the
  sampler: several frames, several tokens per frame, several channels, all
  frozen, all coming out of the whole trajectory bit-identical at eta 0 AND
  eta 1, with every step announcing the prefix at timestep 0 and everything
  else at the schedule's sigma. Together with the carry test this closes
  "window n's last K latent frames ARE window n+1's first K".
* `ltxv::longform::tests` - the window's own frame accounting is
  self-consistent (`decoded == emitted + dropped`, and a continuation window
  drops exactly a K-latent-frame clip's own length).

Real weights, seconds (real conv VAE, tiny random-weight DiT, CPU):

* `longform.rs::real_weights::a_multi_window_request_comes_back_as_one_clip_of_the_requested_length` -
  41 frames at 64x64 under a forced 20-token ceiling, so the plan really
  splits: one clip of exactly 41 frames comes back, not flat, with one VAE
  decode per window and every window's steps in the timing. A WIRING claim.

#### 4 - what this phase does NOT claim

* **No real-weight long generation.** Both Tesla P40s were saturated by
  unrelated in-flight generations for the whole of this work, so nothing here
  ran against `ltx25_22b`. The code path is exercised end to end on CPU with
  the real VAE, and the seam's LATENT continuity is exact by construction and
  gated as such - but that the resulting clip's MOTION looks continuous to a
  viewer is argued, not measured. The measurement to run is
  `clipmetric::frame_to_frame_diffs` across a known seam index: the naive
  last-frame chain should show a spike there and the rolling-context path
  should not.
* **The first carried latent frame is presented as one pixel frame wide, and
  really covers eight.** A window is handed to the model as an ordinary clip
  with its own time origin, so `real_pixel_positions`' causal fix gives its
  latent frame 0 the `[0, 1/fps)` bound the causal VAE's genuine first frame
  has, and `keyframes_mask` marks it. The carried frame sitting in that slot
  came from the middle of the previous window and covers 8 pixel frames of real
  time. One of K frames is affected and it is the oldest; every later context
  frame is consistent. The alternative - global-timeline positions across the
  whole clip - would put continuation windows at frame-axis positions no clip
  the model was trained on ever starts at, and was not attempted.
* **No blending, no overlap-add, at the seam.** The context frames are decoded
  (they have to be - a latent frame cannot be decoded without its neighbours,
  and feeding them is what makes the PIXEL seam continuous) and then dropped.
  Nothing is averaged. Given the ±14-latent-frame receptive field above, the
  frames just after a seam are decoded with 8 latent frames of real history
  where a single-window decode would have had more; that residual is not
  measured.
* **The output clip is assembled in host RAM.** The rolling LATENT state does
  not grow with duration - that was the design requirement and it holds - but
  `Video::frames` still accumulates every decoded frame before the caller
  encodes anything, which is ~1.3 GB for 20 seconds at 1280x704 and scales
  linearly. Streaming windows straight into the encoder is the obvious next
  step and was not taken here; it changes `generate`'s return type, which is
  the serving contract's shape as well as the CLI's.
* **`--end-frame` is refused for a multi-window clip**, rather than silently
  applying to the last window. The latent context and an appended keyframe
  block both want to be what a window is pinned to, and "the clip ends on this
  still" over a rolling plan has not been designed.
* **Not a distilled long-form model.** The reference ships prefix conditioning
  as a LoRA FINE-TUNING recipe for video extension, not as a base-model
  inference path. The mechanism is the base model's own (it is `--start-frame`'s
  freezing, widened) and 8 latent frames is the prefix size that recipe uses,
  but a base checkpoint with no extension LoRA is being asked for something it
  was not explicitly distilled for. It is strictly more information than one
  re-encoded pixel frame; it is not a claim that the model was trained to
  consume it.

### Phase 23 - long-form, re-derived from the reference, and its first real run

Phase 22 shipped rolling-window long-form generation and closed with a list of
what it did NOT claim, headed by "No real-weight long generation". This phase
is that run - and an independent re-derivation of Phase 22's conditioning
decisions from the vendored reference rather than from Phase 22's reading of
it, because a decision only ever checked against the argument that produced it
has not been checked.

Both of Phase 22's load-bearing conditioning claims survive, and are now
sourced to the reference's own call chain rather than to one config file. One
claim elsewhere in `denoise` - older than Phase 22 - was wrong about the
reference and is fixed. And the real run found a defect Phase 22 could not
have seen.

#### 0 - freezing at exactly sigma 0 is what the reference does, checked call by call

The vendored tree is the official LTX-2 repository, so the inference side of
prefix conditioning can be READ instead of inferred. End to end:

* the trainer's own validation sampler builds a prefix as
  `VideoConditionByLatentIndex(latent=..., strength=1.0, latent_idx=0)`;
* that item writes `clean_latent[:, start:stop]` and sets
  `denoise_mask[:, start:stop] = 1.0 - strength`, i.e. exactly `0`;
* `GaussianNoiser.__call__` is `lerp(latent, noise, noise_scale)` then
  `lerp(clean_latent, that, denoise_mask)`. **At mask 0 the initial latent is
  the clean content exactly** - no forward noise, no augmentation term, no
  sigma floor;
* `timesteps_from_mask` is `denoise_mask * sigma`, so a prefix token's
  per-token timestep is `0`;
* both denoising loops re-pin it every step.

Training agrees and is NARROWER than the literature: the flexible strategy's
intrinsic-condition mask is BINARY, the update is `noisy = m*clean +
(1-m)*noisy` with `timesteps = (1-mask)*timesteps` and the region excluded from
the loss, and the video-extension recipe sets `probability: 1.0`. That recipe
has never seen a prefix at any noise level except zero.

So "should the carried context get a little forward noise" is answered no, and
not as a matter of taste.

#### 1 - where the noise-on-context idea comes from, and why it does not apply

The instinct is real and has a literature, but it is two literatures and only
one of them is this one.

**Family A - per-frame independent noise levels.** Diffusion Forcing samples an
independent noise level per token and FEEDS that level to the network, so a
context at level zero is in distribution by construction; its transformer
descendant states the same thing as "noise as masking", with history frames at
level 0 and therefore unmasked. Rolling Diffusion is the same family with the
level a fixed function of position in the window, and its already-committed
frames sit at exactly zero. LTX-2's per-token-timestep conditioning is this
family, and the prescription is: match training, and training put the context
at zero. The family does use non-zero context noise, but in GUIDANCE - an
unconditional branch built by masking history with complete noise, and a
fractional variant at an intermediate level - both terms in an expression whose
base is still the clean-history conditional.

**Family B - noise as a distrust knob.** Cascaded diffusion's conditioning
augmentation, and the autoregressive video work that inherited it, noise the
conditioning because it is a previous stage's IMPERFECT output; the level is
sampled during training, fed to the model, and pinned at inference. The benefit
only exists if the model was trained that way.

The carried context here IS a model's own previous output, so family B's drift
argument applies - but this checkpoint's extension recipe is family A, and
bolting a family-B augmentation onto it at inference is a train/test mismatch in
the other direction. It has also been measured and lost: FramePack ablates
"noisy history" against its own method and reports 1146 Elo against 1221, with
the mechanism named exactly - reducing reliance on the history interrupts error
accumulation "at the cost of aggravating forgetting". The supported ways to
attack drift, if it shows up, are to train for it (sample a prefix noise level
and feed it, which would also unlock fractional history guidance) or to change
the sampling structure. Not to noise the prefix unilaterally.

#### 2 - window-local positions are the correct answer, not a shortcut

Phase 22 recorded this as a rejected alternative. The reference and the
literature both make it stronger than that.

The reference's own tiled temporal pipeline remaps every tile's keyframe
positions into tile-local coordinates before denoising it, and caps its
conditioning frame rate independently of playback so RoPE time never leaves the
trained distribution. Prefix training likewise builds positions for the whole
clip starting at zero, with the prefix occupying frames `0..N-1` - so the
extension recipe has only ever seen a prefix at positions `0..N-1`. The
streaming-video literature agrees from the failure side: the papers that chain
windows with a globally growing frame index report indices leaving the range
RoPE was trained over, and both published fixes are forms of window-relative
re-indexing. `real_pixel_positions` being rebuilt per window, from zero, is the
behaviour that matches training; the cost is on the other branch.

What remains a deviation is the one Phase 22 already recorded and it is
unchanged: the carried frame in local latent slot 0 is read as a
single-pixel-frame latent while it really covers eight. One of K frames, the
oldest, and the seven against the seam are consistent. The reference does the
same wherever it slices mid-stream latents into a tile, and handles it by
discarding the affected output - which is what a continuation window's dropped
context frames already do here.

#### 3 - a correction to `Frozen`'s doc: the ancestral loop masks the x0 estimate too

`Frozen`'s doc claimed the reference's two loops differ in WHERE the mask is
applied - deterministic Euler on the x0 estimate, ancestral Euler on the
stepped latent only, "the x0 estimate is left alone". The second half is wrong.
`samplers._ModalityStep.from_modality_result` runs `post_process_latent` on the
x0 estimate for every ancestral step, terminal one included, and the loop THEN
masks the stepped latent again after the renoise term. Two applications, not
one moved. `denoise` implemented only the second.

Invisible at `denoise_mask == 0` (the x0 conversion runs at that token's own
zero timestep, where it is the identity, so the estimate already IS the clean
content) and at `denoise_mask == 1` (the blend is the identity) - which is why
every gate passed either way and why nothing that ships changes by one bit:
`--start-frame` at full strength and the long-form latent context are both
exactly 0 or 1. `--conditioning-strength` below 1 under the default `eta = 1.0`
is where it was a different trajectory from the reference's. Fixed, gated at a
strength in between by
`a_partially_conditioned_token_is_pulled_to_its_clean_content_under_both_samplers`
- one terminal step, so what comes out IS the masked x0 estimate with no
further arithmetic to hide a missing blend. Red at `eta = 1.0` (9.75 against
9.875) and green at `eta = 0.0` before the fix, which is the shape a gate for
this has to have.

#### 4 - the seam, measured

Phase 22's own list opened with "that the resulting clip's MOTION looks
continuous to a viewer is argued, not measured", and named the measurement.
It has run: 121 frames at 384x192 split into exactly two windows with the real
8-latent-frame context, four real 22B window generations, 577 s on one P40.

| arm | seam ratio | distance from 1.0 |
|---|---|---|
| rolling latent context | **0.99** | 0.01 |
| last-frame chaining | 0.85 | 0.15 |

The gate had to be corrected first, and the correction matters more than the
numbers. It compared raw magnitude (`rolling < chained`), but 1.0 is the
target, not 0: at 1.0 the seam transitions exactly like a typical frame, BELOW
1.0 is a freeze - the artefact naive chaining produces - and above is a jump.
On these very numbers the old assertion would have FAILED a near-perfect 0.99
against a chain arm that stalled to 0.85, i.e. rewarded the defect it exists to
detect. It now compares distance from 1.0.

#### 5 - the first real-weight long-form run, and the defect it found

337 frames at 1280x704, real 22B distilled checkpoint, one `--start-frame`.
The planner did exactly what it should: five windows (one of 15 new latent
frames, then four of 8 carried plus 7 new), the two-stage shape chosen once for
the plan, 66 progress phases which is `1 + 5 * (8 + 4 + 1)`. Windows 1 and 2
completed. Window 3 aborted in its refinement stage with `wgpu error: Out of
Memory`, 3100 s in.

**It is not a leak, and that was worth measuring rather than assuming.** A
controlled reproduction - 249 frames at 384x192 with the token ceiling forced to
1008 and two-stage forced on, so the plan is four two-stage windows and the run
is ~10 minutes - sampled nvidia-smi every 5 s and segmented per window:

| | window 1 | window 2 | window 3 | window 4 |
|---|---|---|---|---|
| peak, before | 13911 | 13911 | 13911 | 13911 |
| floor, before | 2507 | 5961 | 5769 | - |
| peak, after | 8145 | 8145 | 8145 | 8145 |
| floor, after | 79 | 95 | 10 | 95 |

Flat, and the peak bit-identical every window. Nothing accumulates on the
device across the rolling loop.

**What it is** is the height of that steady-state peak. Peak VRAM during a
window is dominated by the DiT's RESIDENT WEIGHT WINDOW being alive while
something else opens its own device. `generate` never has that problem and its
own comment says why - it drops the whole denoiser before the VAE decode - and
`devres::planned_slots` already caps the resident window at a quarter of the
card for exactly this collision, recording the measurement that put the cap
there: a decode's own device aborting at 24211 MiB of a 24576 MiB card, on a
shape with no memory problem of its own. A window loop cannot drop the
denoiser, so before this phase it held it, across every window's x2 upscaler
build and every window's decode.

`Denoiser` gained `release_devices` - a no-op by default, on `RealDit` a drain
of its per-card session map that drops each card's open `Gpu` and the resident
window with it. `generate_long` calls it twice per window: before the upscaler
build (nearly free - stage 2 is a different token count, so the window had to
be rebuilt for it anyway, and the new session's slot count is now planned from
stage 2's OWN token count instead of inheriting stage 1's much smaller one) and
before the decode. `upscale`'s segment loop had the identical hazard and got
the identical call. The weights are not re-read; they come back from the
checkpoint-scoped host cache the `RealDit` still holds. The run got FASTER
(539.6 s against 637.1 s, 11.008 against 11.787 s per forward).

The floor dropping from ~5.9 GB to ~0.1 GB is the release really happening.

#### what this phase does NOT claim

* **The 1280x704 failure is not reproduced.** The controlled shape reproduces
  the collision and its fix, not the abort. The recurring peak there is 5766
  MiB lower than it was, which is the headroom the aborting allocation did not
  have - but whether that is sufficient is what a re-run answers, not this.
  A slow fragmentation ratchet inside the long-lived DiT device would be
  invisible to nvidia-smi (which reports reserved, not used-within-pool) and is
  consistent with an abort at the third window rather than the first; releasing
  the session twice per window resets that allocator either way, so the change
  addresses it without proving it was there.
* **Host memory still grows with duration.** Phase 22 recorded it and it is
  untouched: `Video::frames` accumulates every decoded frame before the caller
  encodes anything. That is ~950 MB for this clip and cannot produce a wgpu
  device OOM, but it is the thing that actually scales with length.
* **No always-run gate on the release.** `generate_long` builds its own
  denoiser, so the wiring is not injectable and the claim is a resource one.
  It is gated the way this crate already gates VRAM findings - a recorded
  measurement - not by a test.
* **`BRAIN_GPU_INDEX` does not pin on a multi-GPU box.** A schedulable set of
  more than one card makes `ComputeSet::apply_backend` clear the registry's
  ambient selection, "including an inherited BRAIN_GPU_INDEX", by its own
  comment. `BRAIN_DEVICE=gpu1` is the pin that holds, and the seam gate's doc
  now says so.

### Phase 24 - a clip stops being one scene long

Phases 22 and 23 made a clip longer than one denoising window into one
continuous SHOT: every continuation window is hard-conditioned on the previous
window's own last 8 latent frames, frozen at sigma 0, so motion survives the
seam. That is the right answer to the question it answers, and it is the wrong
answer to a different one. A single `--frames`/`--prompt` call has no way to
become a different scene: the prompt is shared by every window, and every
window after the first is pinned to real content it must continue. "Generate
two minutes that starts in a harbour and ends at sea" was not expressible; the
workflow was N separate commands and `ffmpeg concat` by hand, which is exactly
the last-frame chaining Phase 22 removed, one level up.

`brain ltxv t2v` now takes a repeated `--scene <frames>:<prompt>`. Inside a
scene nothing changes at all. **At a scene boundary the rolling context
resets** - the next scene's first window carries `context == 0`, exactly like
the first window of any plan - so the new scene is driven by its own prompt and
by nothing of the old one.

```text
brain ltxv t2v --dit-config ltx25_22b --width 768 --height 448 --fps 24 \
  --output-path story.mp4 \
  --scene 121:"a fishing boat leaves a harbour at dawn, camera tracking" \
  --scene 121:"the open sea under heavy rain, waves breaking over the bow" \
  --scene  57:"a close-up of a gull on a wet railing"
```

#### 0 - a multi-scene call is the existing machinery run N times

`generate_scenes` is `generate_long` in a loop over scenes, and deliberately
contains no second window loop, no second sampler, and no second stage
decision. The reset is not a feature that had to be built: a fresh
`generate_long` call's window 0 already carries nothing, which IS the reset.
What the new function adds is three things and they are all bookkeeping - the
plan over all scenes before the first weight is read, a per-scene seed, and the
concatenation.

That shape also means Phase 23's VRAM release still does the work: each scene's
`generate_long` builds its own denoiser and drops it at the end, so the card is
fully released between scenes as well as between windows. The rebuild is not a
re-read - `block::GenerationCache::for_checkpoint` is keyed on the checkpoint
file, not on the call, so scene 2 attaches the warm host weight cache scene 1
filled. The text encode genuinely does run again per scene, and has to: the
prompt changed, which is the entire point.

`longform::scene_plan` is `window_plan` once per scene with each window's
`first_frame` offset into the whole clip. Validating every scene up front
matters at this scale: a five-scene request whose fourth scene is unplannable
should fail in milliseconds, not after three scenes of device time, and the
refusal names which scene.

Seeds are `seed ^ SCENE_SEED_SALT * scene_index`, so two scenes never draw the
same initial noise. Multiplied rather than XORed so that scene 0 keeps the
caller's seed exactly - a one-scene request stays bit-for-bit the run it
already was, and `generate_scenes` hands a one-scene call straight to
`generate_long` for the same reason `generate_long` hands a one-window request
straight to `generate`.

#### 1 - the CLI shape, and the two things it refuses

`--scene <frames>:<prompt>`, repeatable - the house spelling for both halves
(`flux2_cli`'s `--ref` is the repeatable-push precedent, `split_once` on `:`
the compound-value one). The separator is the FIRST colon and only the first,
because "then: the camera pans" is a sentence.

`--scene` cannot be combined with `--prompt`/`--frames`. Those two ARE the
single-scene spelling; a command carrying both would have two ways to say where
the clip starts and no obvious answer for what `--frames` means next to three
scenes that each have their own. `--frames` has a default, so the parser tracks
whether it was actually given rather than inferring it from the value.

`--end-frame` is refused for a multi-scene clip, for the reason Phase 22
refused it for a multi-window one: the last window of a multi-scene plan
belongs to the last scene, which is not what "the clip ends on this still"
means. `--start-frame` conditions the first scene's opening and nowhere else.

#### 2 - hard cut only, and why a per-boundary soft anchor was NOT added

The obvious extension is an optional per-boundary `--start-frame`-style anchor:
carry ONE re-encoded pixel frame across a scene cut, for a softer transition.
It is not implemented, and the reason is not effort.

A single re-encoded pixel frame is precisely the conditioning Phase 22 measured
and rejected - continuous in position, discontinuous in velocity, seam ratio
0.85 against the rolling context's 0.99 - and its characteristic artefact is
that the model starts a clip from a still and re-invents the motion. Applying
it at a scene boundary would reintroduce that artefact at the one place where
the content is supposed to change anyway, and would half-pin a new scene to the
old scene's last composition, which is the constraint the boundary exists to
remove. A caller who wants the content to continue is asking for ONE scene, and
says so by writing one. The genuinely different feature - one continuous shot
whose PROMPT evolves while the latent context keeps rolling - is not this one,
has its own literature (see below), and has not been designed.

#### 3 - what the lineage actually documents about long-video degradation

The second motivation for this feature was the worry that a long single-scene
generation drifts into a repeating loop. Checked against the papers rather than
asserted, and the honest answer is more specific than the worry.

**Attested, strongly: drift.** FramePack (arXiv 2504.12626 v1/v2, section 1)
defines it - "'drifting' refers to the iterative degradation of visual quality
due to error accumulation over time (also called exposure bias)". That paper
was rewritten in v3 (October 2025, different title, different author list,
re-run numbers), so every FramePack quote in this phase is pinned to v1/v2 -
v3 says "the degradation", drops "iterative", and renumbers its tables. Self
Forcing (arXiv 2506.08009) states its own limit outright: "quality degradation
remains observable when generating videos substantially longer than those seen
during training". CausVid (arXiv 2412.07772) is the same failure attacked by
distillation.

**Attested, and directly about THIS conditioning: stagnation, not looping.**
DFoT / History Guidance (arXiv 2502.06764, section 5) is the closest thing in
the literature to an analysis of a clean-history-conditioned model: "a major
failure mode of HG-v under high guidance scales is the generation of overly
static videos with minimal motion. This occurs because HG-v encourages
consistency with history, leading to a trivial solution of simply copying the
most recent history frame." Its Figure 5 caption is the trade-off in one line:
"Vanilla history guidance trades off dynamics · diversity for quality ·
consistency." StreamingT2V (arXiv 2403.14773) reports the same mode from the
other side, in its introduction: "all assessed image-to-video methods produce
video stagnation or strong quality degradation when applied autoregressively by
conditioning on the last frame of the preceding chunk". Its CAM/APM modules
exist to fix stagnation and appearance forgetting respectively - the abstract
splits them as a "short-term memory block ... leading to consistent chunk
transitions" and a "long-term memory block ... to prevent the model from
forgetting the initial scene". StreamingT2V drives its whole long video from
ONE prompt; it has no per-chunk prompt mechanism.

**Barely attested: literal content repetition.** No paper in the core lineage -
StreamingT2V, Diffusion Forcing, Rolling Diffusion, CausVid, Self Forcing,
FramePack - says the content loops or repeats. Where it IS named as a mechanism
is the streaming-with-attention-sink line, and it has a name there:
"sink-collapse", where "the generated content repeatedly reverts to the sink
frame, resulting in abrupt scene resets and cyclic motion patterns" (LoL,
arXiv 2601.16914, abstract), attributed to a conflict between RoPE's periodic
structure and multi-head attention; DySink (arXiv 2605.21028) states the same
mechanism as "RoPE-induced phase re-alignment can homogenize inter-head
attention and cause sink collapse, where content regresses toward sink frames".
VideoSSM (arXiv 2512.04519) corroborates the symptom without the term - sink
frames "reduce drift but cause content repetition and a lack of dynamism".
**That mechanism needs two things this port does not have**: a persistent
early-frame KV sink retained across the whole rollout, and monotonically
growing RoPE frame indices. Phase 23 section 2 established that
`real_pixel_positions` is rebuilt per window from zero, and the carry here is
the immediately preceding window's tail, not a pinned anchor. So the one
citable prediction of literal looping does not describe this architecture, and
"the scenery will repeat" should not be asserted as a known failure of it.
"Attractor state" is not this literature's vocabulary at all and is not used
here.

**What the lineage's own fix is, and that this phase does not adopt it.** Every
method that stabilises long rollouts adds noise BACK into the history rather
than freezing it: Diffusion Forcing rolls out "using the previous latent
associated with slightly 'noisy tokens' for some small noise level 0 < k << K"
(section 3.3), DFoT's fractional history guidance partially masks - which that
paper's appendix A.2 defines as partially noising - the history and reports
that "guiding with lower frequencies [...] consistently increases dynamics
while maintaining quality" (section 6.3), and FramePack's own related-work
section groups both as "noise scheduling and augmentation in history frames
modify noise levels at specific timesteps, video times, or image frequencies to
create causal computation or anti-drifting effects. These methods generally
reduce dependency on past frames." Phase 23 already settled why this port does
not do that unilaterally - LTX-2's own extension recipe trains the prefix at
exactly zero and nothing else - and that is unchanged here. It is recorded
because it is the lever this phase deliberately did not pull.

**Per-scene prompting itself is attested prior art, and a hard switch is a
known baseline.** FIFO-Diffusion and MTVG "propose switching directly from one
prompt to the next during video generation, resulting in noticeable
inconsistencies between scenes" (as characterised by arXiv 2412.17254,
"Enhancing Long Video Generation Consistency without Tuning") - which is a
defect when the cut is unintended and is the intended output here. Everything
else in that survey keeps the visual history and varies how the TEXT binds:
Gen-L-Video (Wang et al. 2023) interpolates between prompts over transitional
frames, FreeNoise (2310.15169) does the same through its Motion Injection, MinT
(2412.05263) binds each event to a time span with ReRoPE, Mask2DiT (2503.19881)
masks text-to-segment attention while keeping cross-segment vision, MEVG
(2312.04086) anchors the next event on the previous clip's last frame. Dropping
the visual history at a prompt change is not something any of them describes;
it is the correct thing for a genuine scene CUT and is not a substitute for the
"one shot, evolving prompt" feature those papers build.

**The reference has nothing.** Every LTX-2 inference pipeline takes a single
`prompt: str` - no `nargs`, no append, no prompt schedule; the only sequence of
prompts anywhere is `PromptEncoder`'s positive/negative CFG pair. The only
scene-detection code in the tree, `packages/ltx-trainer/scripts/
split_scenes.py`, chops TRAINING footage into clips. There is no documented
statement about inference-time extension, chunking, per-scene prompting, or a
maximum practical duration.

So the claim this phase makes about problem 2 is the narrow one: multi-scene
does not prevent drift or stagnation within a scene, and nothing here measures
how long one scene survives. What it does is put the length of any single
autoregressive chain under the caller's control, and give them a reason to end
one that is not "hope".

#### gates

Weight-free, always run (`crates/ltxv/tests/longform.rs`):

* `a_multi_scene_plan_resets_the_carried_context_at_every_scene_boundary` -
  three scenes at 1280x704 (481 + 241 + 113 frames, the last deliberately short
  enough to be ONE window): every scene's first window carries 0, every later
  window in a scene carries the full 8, each scene emits exactly its own
  length, a context reset happens once per scene and nowhere else, and the
  flattened plan reassembles to one continuous 835-frame clip with no
  duplicate and no gap.
* `a_scene_spec_is_a_frame_count_a_colon_and_the_whole_rest_of_the_prompt` -
  the separator is the first colon only, a prompt keeps its own colons, and a
  scene with no prompt / no count / no separator is refused.
* `an_impossible_scene_is_refused_and_named` - the refusal says which scene.
* `ltxv_cli::tests::every_flag_the_parser_accepts_is_documented` - `--scene`
  joins the existing help/parser self-check.

Real weights, seconds (real conv VAE, tiny random-weight DiT, CPU):

* `a_two_scene_request_is_one_clip_whose_second_scene_owes_nothing_to_its_first` -
  the direct observable for a reset, rather than an assertion about the plan.
  A two-scene call (41 + 25 frames at 64x64 under a forced 20-token ceiling, so
  scene 1 really spans two windows) comes back as one 66-frame clip whose
  first 41 frames are byte-identical to scene 1 generated ALONE at its own
  seed, and whose last 25 are byte-identical to scene 2 generated alone at
  its. Anything crossing the boundary - a carried latent, a leaked sampler
  state, a shared noise draw - breaks that equality. One VAE decode per window
  across both scenes, and the aggregated timings account for both scenes' steps.

#### 4 - the first real multi-scene run

178 frames at 384x192, real 22B distilled checkpoint, real Gemma-4 encoder, one
P40, the token ceiling forced to 1008 so scene 1 really splits. 543.4 s
(build 13.8, text encode 124.1, denoise 374.9 at 15.620 s/forward, vae 29.5),
one 178-frame mp4.

The plan ran as written - `scenes=2 windows=3 frames=178`, then scene 1 as two
windows (`carried=0 new=14`, then `carried=8 new=2`) and scene 2 as its own
single window. Both halves are visible in one run: the intra-scene carry
happening inside scene 1, and the reset at the boundary.

The cut is where it was planned, measured rather than watched. Per-frame mean
luma difference across the assembled clip (`blend=difference` + `signalstats`):

| | value |
|---|---|
| median frame-to-frame difference | 4.99 |
| largest difference that is not the boundary | 6.90 (frame 53) |
| difference at frame 120 -> 121 | **116.95** |

One spike, 23x the clip's own median and 17x the next largest, at exactly the
frame the plan puts the boundary at. Scene 1 is a tracking shot across an open
field; scene 2 is a dark rain-covered window with blurred city lights behind
it. Nothing of the first scene survives into the second, which is the whole
request.

#### what this phase does NOT claim

* **No quality claim about a cut.** The gates prove the boundary is a reset and
  the clip is one file of the right length. Whether a hard cut between two
  prompts is what a viewer wants to see is a directorial question, and no
  metric here measures it. Phase 23's `clipmetric` seam ratio is the wrong
  instrument on purpose: at a scene boundary a spike is the intended output,
  so the boundary measurement in section 4 is reported as a spike against the
  clip's own median rather than as a ratio to be minimised.
* **No prompt-adherence claim.** Section 4's first scene asked for a Belgian
  Malinois and the checkpoint produced a human runner, correctly tracked across
  an open field. That is the model answering the prompt its own way and it says
  nothing about this phase either direction - what the run demonstrates is that
  each scene denoises against its OWN text context and that the second scene
  owes nothing to the first.
* **Nothing is measured about how long one scene survives.** The drift and
  stagnation literature above is cited, not reproduced. This phase adds the
  lever, not a number for where to set it.
* **A multi-scene run costs a text encode and a denoiser build per scene.** The
  weight cache makes the rebuild cheap and the encode is unavoidable, but a
  many-short-scene clip pays a fixed per-scene overhead a single long clip does
  not.
* **Not on the capability surface.** `ltxv::caps`'s `t2v` action still takes one
  prompt; `--scene` is a CLI flag only. Putting a scene list into an
  `ActionSpec` is a separate design (a repeated typed parameter, and a progress
  contract across scenes) and was not attempted.
* **Host memory still grows with duration**, and now with the whole multi-scene
  clip rather than one scene's: `generate_scenes` concatenates every scene's
  decoded frames before returning. Phase 22 recorded this for one clip and it
  is unchanged in kind; a multi-scene request just makes it easier to reach.
* **No per-boundary soft anchor**, and no "one shot, evolving prompt" mode. See
  section 2 and the last paragraph of section 3 for what each would be and why
  neither is this.

### Phase 25 - an upscaled clip stops being several clips

A real 217-frame 1280x704 clip went through `brain ltxv upscale --factor 2`
and came back 217 frames at 2560x1408, correct in every dimension and wrong in
the only way that matters: it plays as five or six different clips cut
together, not as one clip with more detail. The frame count check passed
because the arithmetic was right. The output was unusable because nothing
crossed a boundary.

#### 0 - the count, and why it is not a "seam"

Phase 21's `refine_segments` split a too-long refinement into consecutive
`1 + 8k` segments sharing ONE pixel frame at each boundary, and its own ledger
entry called that "forced, not tuned" with a seam that is "real, unblended".
Both halves of that are true and neither is what went wrong. What went wrong is
that the shared frame carried nothing: segment `n + 1` VAE-encoded its own
range **of the ORIGINAL clip**, upscaled it, and refined it with
`context: None` and `seed_salt: 0`. Segment `n`'s refined output was never an
input to anything. The overlap existed only so `n` segments of `1 + 8k` frames
could sum to a `1 + 8K` clip; the earlier copy of the shared frame was
discarded. Zero bits crossed a boundary - not a re-encoded picture, which is
Phase 22's naive baseline, but nothing at all.

That is worse than a detail step, and the sigma table says why. A refinement
pass starts at `LTX2_STAGE2_DISTILLED_SIGMAS[0] = 0.909375`, and
`denoise_stage`'s partial re-noise is `lerp(seed, noise, sigma0)` - so a pass
keeps **9%** of the content it was handed and re-derives the rest. Two passes
over adjacent content, with independent noise and no shared history, do not
produce the same clip twice. They produce two clips.

And there were not two. At 2560x1408 the latent grid is 44 x 80 = **3520
tokens per latent frame**, so `REFINE_MAX_TOKENS = 12288` holds
`12288 / 3520 = 3` latent frames per pass. 217 frames is `k = 27`, `k_max = 2`,
so `ceil(27 / 2) = ` **14** segments (thirteen of 17 frames, one of 9). Fourteen
independent renderings of one clip, boundaries every ~0.7 s at 24 fps. "Five or
six" was a generous count.

#### 1 - the fix is a function this crate already had

Phases 22 and 23 answered exactly this question for generation and measured the
answer: carrying the previous window's own last `CONTEXT_LATENT_FRAMES = 8`
real latent frames, frozen at sigma 0, lands a seam at ratio **0.99** against
an ideal 1.0, where re-encoding one decoded pixel frame lands at **0.85**
(`longform.rs::seam_real`). `upscale` was written before that existed and was
never migrated.

So `refine_segments` is deleted and `refine_plan` is `longform::window_plan`
with one line in front of it. The upscale loop is now `generate_long`'s window
loop with the generation replaced by an encode: per pass, VAE-encode the source
range `Window::source_first_frame()` names, carry it up with the x2 upscaler,
overwrite the leading `context` latent frames with the previous pass's own
refined latent (`Refine::context`, the `LatentContext` field Phase 22 added and
this call site passed `None` to), refine, `carry_tail` the result, decode, drop
`Window::dropped_frames()` leading pixels. Same planner, same carry, same
freeze, same Phase 23 `release_devices` before each decode. Nothing about
windowed continuity is implemented twice.

`Window::source_first_frame` is the one thing refinement needs that generation
does not: `first_frame - dropped_frames()`, the input frame a pass starts
READING at, since a pass has to fetch the pixels its carried context decodes
back to rather than inventing them. Its own arithmetic makes the range a pass
reads end exactly where that pass's output ends, which the gate asserts
directly.

Two smaller things came with it. `UpscaleOpts` gained
`max_refine_tokens` - `LongOpts::max_window_tokens`'s counterpart, and what
lets the CPU wiring gate below run a real multi-pass plan at 64x64. And the
per-pass seed salt stopped being `0`: every pass drew identical refinement
noise, which was not the cause of anything but was not intended either. It is
now `REFINE_SEED_SALT * pass_index`, multiplied rather than XORed for
`SCENE_SEED_SALT`'s reason - pass 0 keeps the caller's seed exactly, so a clip
that fits one pass is bit for bit the run it already was.

#### 2 - where full reuse stops: the context does not always fit

`window_plan` REFUSES a grid with no room for `context + 1` latent frames, and
for generation that is the right answer - the caller picked the resolution and
a smaller one is available. Refinement has neither escape. Its grid is the
input's grid times the factor squared, and at 2560x1408 a pass holds three
latent frames total, so eight carried ones is not tight, it is impossible.
Refusing would refuse the case the feature exists for.

`longform::fitted_context` is that one line: `want.min(max_lat - 1)`, an error
only when even `max_lat < 2`. The plan carries the most history the ceiling
leaves room for and keeps one frame to refine, and `upscale` warns with both
numbers when it had to shrink. This is a compromise nothing in this repo
measures - `CONTEXT_LATENT_FRAMES = 8` is the reference's own
`temporal_boundary` and no smaller value is cited anywhere - and it is recorded
as a compromise rather than presented as a tuning.

**It costs passes, and the number is not small.** A pass spends `context` of
its budget before it refines anything, so the DiT work per emitted frame scales
as `max_lat / (max_lat - context)`. For the 217-frame clip above:

| output | tokens/latent frame | latent frames a pass | passes | carried | latent frames of DiT work |
|---|---|---|---|---|---|
| 1280x704 | 880 | 13 | 4 | 8 (full) | 52 |
| 1920x1088 | 2040 | 6 | 23 | 5 | 138 |
| 2560x1408 | 3520 | 3 | **26** | **2** | **78** |
| 2560x1408, Phase 21 | 3520 | 3 | 14 | 0 | 41 |

The user's own case is 26 passes against 14, **1.90x** the refinement forwards,
for a clip that was previously not worth keeping. The denser the output grid
the worse that ratio gets, which is why `--context-frames` is on the command:
`--context-frames 1` carries a single latent frame and costs roughly what the
uncarried plan did, and the default is the largest the grid allows.
`REFINE_MAX_TOKENS` was NOT raised to buy budget back - it is derived from the
2047 MiB binding size, explicitly conservative, and moving it needs a card and
a measurement, neither of which this phase had.

#### gates

Weight-free, always run (`crates/ltxv/tests/upscale.rs`):

* `a_clip_too_long_to_refine_in_one_pass_carries_real_latent_context_across_every_seam` -
  the exact shape that produced the defect (217 frames, 44 x 80 out). Every
  pass after the first carries a non-zero context, every pass carries the same
  amount, every predecessor can actually supply it, every pass is under the
  ceiling and decodes `1 + 8k` frames, every pass's source range is inside the
  clip and ends where that pass's own output ends, and the passes reassemble to
  exactly 0..217 with no duplicate and no gap. A plan whose continuation passes
  carry nothing fails this test, which is what Phase 21's plan did.
* `the_carried_context_shrinks_to_the_grid_rather_than_vanishing_or_refusing` -
  1280x704 out takes the full 8; 2560x1408 out takes exactly `max_lat - 1 = 2`
  rather than erroring or silently carrying none.
* `a_clip_that_fits_is_one_pass_and_no_seam` / `an_impossible_request_is_refused_up_front` -
  Phase 21's own two claims, kept.

Real weights, 52 s (real conv VAE, real x2 spatial upscaler, tiny
random-weight DiT, CPU):

* `a_multi_pass_upscale_is_one_clip_that_carries_its_own_latent_context` -
  25 frames 64x64 -> 128x128 with the token ceiling forced to 48 so the plan is
  really two passes carrying two latent frames, the way `longform.rs`'s own
  wiring gate forces one. Comes back 25 frames at 128x128, not flat, fps
  intact, with the upscaler and a decode run once per pass and
  `LTX2_STAGE2_STEPS` steps per pass. The carry is load-bearing rather than
  incidental here: `upscale` refuses to run a pass whose planned context and
  whose predecessor's carried tail disagree, so a run in which nothing was
  carried fails with that error instead of passing.

Not re-gated, deliberately: that `carry_tail` is a bit-exact slice (gated in
`longform.rs`), that a frozen prefix survives the sampler
(`pipeline::tests::a_frozen_prefix_of_latent_frames_survives_the_whole_trajectory`),
and that the upscaler is un-normalized around correctly
(`upsampler_parity.rs`). `upscale` reaches all three through the same code
`generate_long` does; asserting them again would gate second copies that do not
exist.

#### what this phase does NOT claim

* **No real-weight run.** Per the constraint this work was done under, nothing
  here touched a card. The mechanism is the one Phases 22/23 measured at 0.99
  on the same seam question, applied to the same freeze through the same
  functions - but the specific claim "an upscaled 217-frame clip now looks like
  one clip" is unverified and belongs to the follow-up run.
* **No seam metric for refinement.** `clipmetric::blowup_ratio` /
  `frame_to_frame_diffs` are the instruments, and a before/after on a real clip
  is what would turn "argued from Phase 22's measurement" into a number of this
  path's own. Two real 2560x1408 upscales is what it costs.
* **The shrunken context is unjustified by any number.** See section 2. Eight
  latent frames is the only cited figure; two is what fits.
* **The pass count rises**, by 1.90x on the case that motivated the phase and
  more on denser grids. That is stated in section 2's table, not hidden.
* **Nothing changed about a clip that fits one pass** - one window, no context,
  seed unsalted, identical output.
* **Still CLI-only, still no capability action.** Unchanged from Phase 21.
### Phase 26 - a clip stops being anchored only at its ends

`--start-frame` pins where a clip begins and `--end-frame` pins where it ends,
and between them the model is on its own. Over 9 frames that is fine. Over 121
frames with a moving camera it is the whole problem: the two anchors are 120
frames apart, everything in between is unconstrained, and there was no way to
say "and it should look like THIS half way through". The request was a third
anchor - first, middle and last in ONE generation pass.

```text
brain ltxv t2v --dit-config ltx25_22b --frames 121 --width 768 --height 448 \
  --fps 24 --output-path boat.mp4 \
  --prompt "a fishing boat crosses the harbour mouth, camera tracking left" \
  --start-frame dawn.png --mid-frame midway.png --end-frame open-sea.png
```

#### 0 - the reference was already generic, and so was the port

The feature was described as something official LTX-2.5 supports natively.
Checked before building - the vendored reference (`resources/ltxv/source`)
first, then what ships around it - and the finding is more useful than "yes":
**there is no first/middle/last feature anywhere in the lineage. There is a
conditioning item whose position is arbitrary, and three of them is what a
"three-keyframe workflow" is.**

`ltx_core.conditioning.types.keyframe_cond.VideoConditionByKeyframeIndex` -
which this port has used for `--end-frame` since Phase 19 - does exactly this
with its `frame_idx`:

```python
positions[:, 0, ...] += self.frame_idx
if self.num_pixel_frames == 1:
    positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
positions = positions.to(dtype=torch.float32)
positions[:, 0, ...] /= latent_tools.fps
```

A raw pixel-frame offset added to the RoPE time coordinate of an APPENDED token
block. No `// 8`, no snapping, no rounding, and no bounds check - the class
validates nothing about `frame_idx` at all. Its sibling
`VideoConditionByLatentIndex` (`--start-frame`'s in-place overwrite) is equally
unpinned: `start_token = get_token_count(target_shape._replace(frames=self.
latent_idx))`, an arbitrary `latent_idx`, again unchecked. The "first/last" in
`ltx_pipelines.utils.helpers.combined_image_conditionings` is a CALLER's `if
img.frame_idx == 0`, not a property of either item. Conditionings reach the
sampler as a plain `list[ConditioningItem]` applied in order by
`state_with_conditionings`, with no cap, and the reference's own CLI spells the
generic form directly: `--image PATH FRAME_IDX STRENGTH`, repeatable
(`ltx_pipelines.utils.args.ImageAction`).

Grepping the vendored tree for `middle frame`, `mid frame`, `MiddleFrame`,
`three keyframe`, `first, middle` returns nothing, and it contains no ComfyUI
workflow JSONs at all. What "First/Middle/Last Frame" names in the community
workflows is three chained `LTXVAddGuide` nodes, and that node's schema is the
same generic thing: `frame_idx` `min=-9999, max=9999`, "Negative values are
counted from the end of the video", and its `execute` calls `append_keyframe`
for every guide including `frame_idx == 0` (ComfyUI
`comfy_extras/nodes_lt.py`). Its one snapping rule is explicitly gated on
MULTI-frame guides - `if guide_length > 1 and frame_idx != 0` - so a single
still is never snapped.

So the hypothesis this phase started from was right, and it is stronger than
"probably": `--start-frame` and `--end-frame` were not two features, they were
two hardcoded call sites (`0` and `frames - 1`) of one mechanism that never
cared. The denoising side needed nothing - `Frozen`, `timesteps_from_mask`, the
per-token x0 conversion Phase 17 fixed and `post_process_latent` are all per
token and index-agnostic already.

#### 1 - what actually changed, which is less than it sounds

`conditioned_latent` stops branching on WHICH stills were given and branches on
HOW MANY. One still at frame 0 and nothing else is still image-to-video and
still overwrites latent frame 0 in place; every other request collects its
stills into a `Vec<(pixel_frame, tokens)>` in timeline order and hands the whole
list to `append_image_conditioning`, which already took a `blocks` slice of any
length and already wrote one appended block per entry. Two of the three old
branches were that loop with the loop written out, and they are gone.
`conditioning_block_count` gains its third bool and keeps returning `0` for the
one request that appends nothing.

That rule - a lone start still overwrites, anything else appends - is not a
convenience. `image_conditionings_by_adding_guiding_latent` wraps EVERY image
including `frame_idx == 0`, and `KeyframeInterpolationPipeline` is the pipeline
built for this shape. Adding a middle anchor to a `--start-frame` run therefore
RELEASES latent frame 0, which is gated
(`a_mid_anchor_moves_the_start_still_from_an_overwrite_to_a_guide`): pinning one
instant twice, once in place and once as a guide, is what that shape avoids.
Every combination that already worked - start-only, end-only, both - is
byte-for-byte the run it was.

**The two-stage path needed no work, and that is worth stating rather than
assuming.** Image conditioning is applied inside `denoise_stage`, which is the
body BOTH stages run (`upscale_and_refine` calls it), and each stage encodes the
stills at its own `st.width`/`st.height`. `o.frames` is the clip's, not the
stage's, so the anchor resolves to the same pixel frame at half resolution and
at full. A mid anchor survives stage 1, the x2 latent upscale and the refinement
for the same reason `--end-frame` already did.

#### 2 - where the middle is, and why nothing is snapped to the x8 grid

`--mid-frame-at <N>` takes a pixel frame. Left off, the position is
`(frames - 1) / 2` - and that is the reference's own number rather than the
obvious one: `ltx_pipelines.utils.helpers.evenly_spaced_keyframe_positions(1,
num_frames)` is `torch.linspace(0, num_frames - 1, 3).round()[1:-1]`, `[60]`
for a 121-frame clip. Every legal clip length is `1 + 8k`, so `frames - 1` is
even and the division is exact; nothing here depends on which way a tie rounds.

**The middle index does NOT have to land on a latent-frame boundary**, which is
the first thing this phase expected to have to solve. The constraint does not
exist. An appended guide is not a slot on the generated video's latent grid -
it is extra tokens carrying their own RoPE position - so there is no grid for
it to land on. The `1 + 8k` rule constrains the clip's LENGTH, which `generate`
already enforces. The reference agrees twice over: the keyframe item snaps
nothing (section 0), and the only pixel-to-latent mapping in the whole tree,
`ltx_pipelines.dfr_layout.pixel_to_latent_index`, *raises* on a position that is
not already on the x8 border rather than rounding to it - and is used only for
DFR's own generated-keyframe grid. So `--mid-frame-at 37` on a 121-frame clip is
legal and is taken verbatim.

`latent_frame_containing` exists anyway, reported in one `tracing::info!` line
and never enforced: it says which latent frame's own pixel span the anchor's
instant falls in (frame 8 of 16, for pixel frame 60), which is what a reader of
the log and a future window plan both want to know.

Refused: `at == 0` and `at >= frames - 1`. Those two instants are what the other
flags already name, and a clip needs three frames to have an interior at all -
the same condition the reference raises on (`num_frames < num_keyframes + 2`).

#### 3 - the CLI, and what it will not do

`--mid-frame <path>` and `--mid-frame-at <N>`, composable with both existing
anchors, and `mid_frame`/`mid_frame_at` on the `t2v` capability action next to
the two that were already there. The first line of a run reports the RESOLVED
frame rather than the flag, so a caller who left the position off is told which
frame it landed on before the run rather than after.

`--mid-frame` is **refused for a multi-window clip and for a multi-scene one**,
next to the `--end-frame` refusals Phases 22 and 24 wrote, and for a reason of
its own on top of theirs: its position is a pixel frame of the WHOLE clip.
Routing it means finding the window whose emitted range covers it,
re-expressing the position in that window's own frame numbering, and deciding
what an anchor landing inside a carried context means - that context is content
the previous window already generated, and `denoise_stage` refuses a still and a
context together outright today. That is a design, not a wiring change, and it
is named in "what this phase does NOT claim" rather than left as a silent gap.

While the anchor flags were being extended, `ltxv_cli::tests::
every_flag_the_parser_accepts_is_documented` turned out never to have listed
`--start-frame`, `--end-frame`, `--conditioning-strength` or `--context-frames`
- four flags that could have drifted out of `--help` with no test noticing.
They are in the list now, with the two new ones.

#### gates

Weight-free, always run:

* `pipeline::mid_anchor_frame_tests::the_default_position_is_the_references_own_single_interior_keyframe`
  - the default lands on `evenly_spaced_keyframe_positions`' own answer at four
  clip lengths, and the latent frame it falls in is cross-checked against
  `real_pixel_positions`' own bounds so the two formulas cannot drift apart.
* `pipeline::mid_anchor_frame_tests::an_explicit_position_is_taken_verbatim_and_only_the_ends_are_refused`
  - 37 on a 121-frame clip is NOT on the x8 border and is accepted, which is
  section 2's finding stated as a test; frame 0, the last frame and anything
  past the clip are refused.
* `pipeline::conditioned_latent_tests::three_anchors_append_three_guiding_blocks_at_their_own_instants`
  - three blocks, each holding its own image, at instants 0, 4 and 8, with every
  token of the generated video still denoising freely.
* `...::a_lone_mid_anchor_appends_one_guiding_block_at_its_own_instant` and
  `...::a_mid_anchor_moves_the_start_still_from_an_overwrite_to_a_guide` - the
  mechanism does not need company, and adding it to a `--start-frame` run
  releases latent frame 0.
* `pipeline::tests::a_frozen_range_survives_the_whole_trajectory_wherever_it_sits`
  - Phase 22's `a_frozen_prefix_of_latent_frames_survives_the_whole_trajectory`,
  generalised and renamed. It froze a PREFIX, which is the one layout that
  cannot fail if a step only ever re-pins from the start of the sequence. It now
  freezes three ranges at once - a carried head, one latent frame in the
  INTERIOR of the generated video, and an appended guiding block past its end -
  and asserts all three come out bit-identical at eta 0 and eta 1, with the
  timestep announcement checked per token rather than per side. Verified to
  bite: truncating `post_process_latent`'s loop to the first half of the
  sequence leaves the old prefix assertion green and fails this one.

Real weights, minutes (real conv VAE, tiny random-weight DiT, CPU -
`crates/ltxv/tests/anchors.rs`):

* `three_simultaneous_anchors_each_reach_the_denoiser` - 17 frames at 64x64,
  four generations off one seed. Changing the middle still's PIXELS changes the
  clip; moving it from frame 8 to frame 12 changes the clip; dropping it changes
  the clip. Each of those is a way for an anchor to be silently dropped -
  unencoded, appended at the wrong position, overwritten by the next block - and
  each is a run that must not come back equal.
* `the_default_mid_position_is_the_one_the_reference_would_pick` - leaving
  `--mid-frame-at` off is bit-for-bit the same clip as naming frame 8 of a
  17-frame request, through the whole pipeline rather than at the arithmetic.
* `longform.rs`'s
  `an_anchor_a_multi_window_plan_cannot_honour_is_refused_rather_than_ignored`
  - both `--mid-frame` and `--end-frame` on a multi-window plan come back as
  errors naming the flag. Costs no generation (the refusal is raised before any
  weight is read) and exists because a caller who supplied a still and got a
  clip that ignored it would have no way to tell. It is the first gate this
  crate has on the `--end-frame` refusal Phase 22 wrote.

#### what this phase does NOT claim

* **No real-weight run, and no quality claim.** Nothing here was generated with
  `ltx25_22b`. The gates prove three anchors are real inputs to the denoiser at
  the instants they were pointed at; whether a middle anchor makes a long clip
  hold together better is exactly the question they cannot answer, and no number
  in this entry is a measurement of output quality. `anchor_real.rs`'s
  perceptual gate still covers `--start-frame` only and was not extended.
* **Multi-window and multi-scene are unsupported, not untested.** The refusals
  are explicit and name why. The named follow-up: **route a clip-wide anchor
  position to the window that covers it** - find the window whose emitted range
  contains the pixel frame, re-express the position in that window's own frame
  numbering, and decide what an anchor landing inside a carried context means.
  Until that exists, a caller who wants a middle anchor on a long clip generates
  the piece they want to anchor as its own request.
* **The strength knob is still global.** `--conditioning-strength` applies to
  every still given. The reference's `--image PATH FRAME_IDX STRENGTH` is
  per-image, and a middle anchor is the first case where per-anchor strength has
  an obvious use (pin the ends hard, guide the middle softly). Not built.
* **One mid anchor, not N.** The internals are a list and the reference caps
  nothing, so N is now a CLI shape question rather than a mechanism one - but
  `--mid-frame` takes one path, and a repeated `<frame>:<path>` spelling was not
  designed.
* **The same-image warning applies here too.** Phase 19 measured that the same
  still at both ends produces a static clip, because that request has a correct
  trivial answer. A middle anchor identical to either end is the same trap over
  a shorter span, and nothing refuses it.
* **Nothing refuses two anchors at the same instant.** `--mid-frame-at` can name
  a frame another anchor already covers; the reference validates nothing there
  either, and neither does this.


### Phase 27 - the per-token adaLN table stops being 3519 copies of one row

The largest remaining non-compute item in a real forward was
`ada_layer_norm_single`, on the HOST: a `[3520,4096] x [36864,4096]ᵀ` GEMM
plus a `[3520,256] -> [3520,4096] -> [3520,4096]` timestep embedder, once per
forward, followed by a 519 MB upload of the result. Phase 14 took it from
75.8 s to 10.0 s and closed by naming the next step; this is that step.

#### 0 - the premise, checked in the code before anything was built

Every row of that table is a function of ONE scalar - that token's timestep -
and nothing else. `pipeline::denoise` builds the timestep vector in exactly
two places (`crates/ltxv/src/pipeline.rs`, the `frozen` match):

* no conditioning: `vec![sigma; t]` - literally one value, `t` times;
* conditioned: `f.mask.iter().map(|&m| m * sigma)`, where `mask` is
  `denoise_mask` - `1.0` on a generated token, `0.0` on a frozen one
  (an image anchor, a long-form window's carried context), or `1 - strength`
  on a `--conditioning-strength` block.

So the distinct-row count is **1** for plain text-to-video and **2** for
everything anchored or carried - never 3520, and never input-dependent in a
way that is hard to characterise. `--conditioning-strength` is global, so even
three stills at three instants add only the one extra value. That was read off
the code, not assumed, and it is what makes the rest of this entry a work
DELETION rather than a trade.

#### 1 - deduplicate generically, do not special-case "uniform"

`dit::adaln::RowTable` (new, in the shared `dit` crate next to `add_table`)
holds a per-token table as its DISTINCT rows plus a `[t]` u32 row map, with
`distinct_rows()` keying on `f32::to_bits`.

The alternative - detect "all timesteps equal", take a fast path, otherwise
run the old one - was rejected before it was written. It gives up the whole
win on exactly the shapes a fallback exists for: an image-anchored clip and
every long-form continuation window have TWO distinct rows, and a
uniform-only fast path sends both back to full cost. Deduplication wins ~1750x
there instead of 1x, and it removes the fallback branch (and its test debt)
altogether - `distinct == t` degrades continuously to precisely the old cost,
which is also how the "before" arm below is measured.

Bit-identity is structural: equal input BITS produce the identical sequence of
roundings, and `backend_cpu::host_gemm::blocked_linear` gives every output
element `bias` then `+= x*w` for ascending `k` regardless of how many rows
accompany it (its own gate, Phase 14). Keying on bits rather than `==` is the
part that licenses this - `0.0 == -0.0` is true on two different inputs and
`NaN == NaN` is false on the same one.

`adaln_row.wgsl` grows a fourth storage binding, `map: array<u32>`, and reads
`tab[map[r] * NR * D + off]`. Everything else about it is unchanged, including
the operand order that makes it bit-identical to the host
`add_table` + `slice_mod` form.

#### 2 - measured, both arms, same binary, same box, one idle P40

`ltxv_bench streamed` grew a `distinct_timesteps` argument for exactly this:
`1` is a real text-to-video step, `<tokens>` is what the stage cost before the
dedup existed. Both arms are therefore the SAME build, back to back, with no
tree drift between them.

    BRAIN_LTXV_DIT=<real Q8_0 22B> \
      ./target/release/ltxv_bench streamed 48 3520 1024 1 1 {3520|1}

48 real layers, 3520 video tokens (25 frames at 1280x704), 1024 context,
int8 compute, ONE resident session, `call 2` (the cache-hit call - the shape
every forward of a generation past the first has). Best of 4 runs each; the
first call of each run is the warm-up and never enters the statistics.

| warm forward, 48 layers | no dedup (3520 distinct) | 1 distinct (a real step) | |
|---|---:|---:|---:|
| adaLN timestep embedder (host) | 1268.7 ms | 21.1 ms | **60.1x** |
| adaLN table GEMM (host) | 8851.8 ms | 177.8 ms | **49.8x** |
| **adaLN-single stage total** | **10243.4 ms** | **222.0 ms** | **46.1x** |
| model-level adaLN upload | 519 MB | 147 KB + 14 KB map | **3400x** |
| **whole warm forward** | **47.91 s** | **36.41 s** | **1.32x** |

At 8 layers, same command with `8` in place of `48`, and with the middle
column that decides whether the generic dedup was worth choosing over a
uniform special case:

| warm forward, 8 layers | no dedup (3520) | 2 distinct (anchored, long-form) | 1 distinct (plain t2v) |
|---|---:|---:|---:|
| adaLN-single stage total | 10302.0 ms | 608.6 ms | 199.2 ms |
| whole warm forward | 16.40 s | 6.64 s | 6.15 s |

**That middle column is the whole argument.** A "detect uniform, else fall
back" design would put every image-anchored clip and every long-form
continuation window in the LEFT column - 10.3 s and a 519 MB upload per
forward. Deduplicating puts them within half a second of the best case
instead. The layer-count independence is the other point: the stage runs once
per forward however deep the stack is, which is why the 8-layer and 48-layer
stage numbers agree.

**The parts sum to the whole.** At 8 layers the two arms differ by 10.25 s of
wall, of which 10.10 s is the adaLN stage and 0.15 s is the per-forward
upload bucket; residual 0.00 s. At 48 layers the wall difference (11.50 s)
exceeds the stage difference (10.02 s) by 1.5 s, which is inside the
run-to-run spread of `block weight upload` at that depth (4.6-6.2 s across
these runs, a partial window re-uploading its tail).

**Nothing moved in the output.** `ltxv_bench` prints the forward's own output
statistics, and across all four runs of each arm they reproduce to every digit
printed. The uniform arm's line is character-for-character the one Phase 14
and Phase 19 published for this shape:

    len=450560 mean=0.060893 std=0.683607 min=-1.330047 max=1.784977 nonfinite=0

**Confirmed by a real generation, on a second harness.** Same shape through
the actual CLI - real Q8_0 22B DiT, real Gemma-4 encoder, real conv VAE, one
P40:

    brain -v --device gpu0 ltxv t2v --dit-config ltx25_22b \
      --prompt "a fishing boat crosses the harbour mouth at dawn, camera tracking left" \
      --frames 25 --width 1280 --height 704 --fps 24 --steps 8 --seed 42 \
      --output-path clip.mp4

    ltxv: 393.2s total  (build 8.17s, text encode 0.1s, denoise 355.5s
                         = 44.437s/forward, vae 28.9s, other 0.6s)

Per-step, differenced out of the printed running averages: 104.00 (cold),
then 37.94, 36.53, 35.21, 34.72, 35.40, 35.82, 34.38 - **best-of-7 warm
forward 34.38 s**. The profiling pass that opened this phase measured the same
command at **45.63 s** best-of-7 before it. That delta (11.25 s) matches the
controlled two-arm bench delta (11.50 s) to a quarter of a second, on two
independent harnesses, which is the cross-validation that matters here.

#### 3 - a correction to two numbers this crate was carrying

`ada_layer_norm_single` carried, in a comment, "the bulk of the ~76 s this
stage cost per forward call (measured: the table GEMM alone is ~14 s of it)".
An independent profile of the same command at the same shape measured 10.2 s
(GEMM 8.9, embedder 1.4), and the two read as a 7.5x contradiction.

Both are right and they measure different trees: the ~76 s / ~14 s figures
are the PRE-Phase-14 state, quoted in a comment written by the commit that
removed it. Re-derived here: 10243.4 ms is the correct pre-Phase-27 figure,
for ONE call site (`adaln_single` in `forward_q_streamed_in`, once per
forward - the AV `av_ca_*` families are on no production path), wrapping the
timestep embedder plus the table GEMM plus the `ts_scaled` map. The comment is
gone; the general lesson is `.agents/rules/lessons.md` #54.

**And `stage_time` nests.** `forward_q_streamed: adaLN-single table (host)`
encloses the two `ada_layer_norm_single: ...` lines printed just above it
(measured residual: 5.1 ms in 10302.0), so summing all three double-counts the
stage. `gpu_core::profile::stage_time` prints one line per call and has no
nesting structure of its own, so the enclosing label now says it is a TOTAL
and names what it nests.

#### 4 - gates, and what each one would have missed alone

* `dit::adaln`'s unit tests: uniform, two interleaved, all distinct, and
  signed zero as two keys rather than one.
* `ltxv::dit::tests::batched_adaln_timestep_embedding_is_bit_identical_to_the_
  per_row_form`, extended from four shapes to four shapes x three timestep
  patterns, comparing bit patterns against the per-row
  `dit::timestep::pixart_timestep_embed` loop.
* `crates/gpu-core/tests/adaln_row_gather.rs` (new): the kernel itself against
  the host arithmetic, over a distinct-count ladder, on BOTH backends. It
  lives in `gpu-core` because `ltxv`'s only production caller of this kernel
  is int8, which the CPU JIT cannot dispatch at all (`matmul_i8_dyn` has more
  than one top-level barrier) - so `ltxv`'s own tests can never run
  `adaln_row` on `backend-cpu`, and a kernel declared `@cpu yes` needs a gate
  that actually runs it there.
* `device_residency.rs::on_device_modulation_is_bit_identical_to_the_host_
  combine_and_slice`, extended to the same ladder. This is the end-to-end one:
  the eager arm still materialises a DENSE `[t, 9*dim]` table and indexes it
  by token, so it is an independent statement of which row belongs to which
  token. That is why the eager paths were deliberately NOT converted to the
  compact form - two arms sharing one row map cannot gate the row map.

**Mutation-verified, on the failure mode this change actually has.** Rotating
the device gather by one token (`map[(r + 1) % R]`):

| mutation | uniform case | two interleaved | caught by |
|---|---|---|---|
| gather rotated by one token | **PASSES** | **FAILS** | bit patterns, GPU and CPU alike |
| dedup collapses every key to row 0 | passes the bit compare (both arms share the bug) | - | the ladder's own "different patterns must give different answers" guard, and `dit::adaln`'s unit tests |

The first row is the whole argument for the ladder: **with one distinct row
every scatter is the same scatter**, so a suite whose cases were all uniform
would have shipped a broken scatter green. The second row is the argument for
the setup guard - a mutation upstream of BOTH arms is invisible to a
two-arm comparison and needs a test that the arms disagree when they should.

Restored and re-run green after each.

#### 5 - what this phase does NOT claim

* **The whole-run number is one arm, not two.** 393.2 s is measured, after.
  The 547.0 s it is set against was measured before, on a tree that has since
  moved (two other agents were editing `ltxv` and `cli` files during this
  pass) and on a run whose text encode was not necessarily cache-warm - this
  one's was, at 0.1 s. Of the 153.8 s difference, **90 s is attributable**
  (8 warm forwards x the 11.25 s the same run's own per-step curve shows) and
  the remaining ~64 s is not attributed. The forward-level delta is the number
  to trust; the whole-run pair is a sanity check, not an A/B. Making a real
  `t2v` run take the old path requires reverting the change, so a true
  whole-run A/B was not attempted against a tree being edited underneath it.
* **The activation reserve was not re-fitted.** `devres::activation_reserve_
  bytes`'s `PER_TOKEN_WGPU` was fitted to a measured plateau that INCLUDED the
  519 MB adaLN table (and wgpu's 2.00x resident cost on it). Removing the
  table should buy roughly a gigabyte of that plateau back, i.e. two or three
  more resident blocks at 720p - but re-fitting it is Phase 18's measurement
  procedure, not an edit, and it was not run. Recorded as a real, available
  win.
* **The nine `[t, dim]` modulation buffers are untouched.** `ModBufs::derive`
  still allocates 9 x 57.7 MB of device scratch per block at T=3520 and fills
  them with `t` copies of one row whenever the table has one distinct row.
  Making the CONSUMERS broadcast instead would remove ~519 MB of device
  traffic per block, but it changes `gate_row`/`norm_mod`'s contracts across
  several kernels rather than one, and this pass stopped at the host and the
  PCIe bus.
* **`ada_layer_norm_single`'s AV call sites gain nothing measurable.** They
  dedup too (the code is shared), but the AV DiT is on no production path and
  its gate tables already ran at `rows = 1`.
* **The host GEMM at `u = 1` is single-threaded.** `blocked_linear`
  parallelises over row TILES, so one distinct row is one tile is one core
  streaming the 604 MB weight matrix - 178-387 ms, the whole remaining cost of
  this stage. Column-parallelising it for small `m` would be bit-identical and
  is worth ~0.3 s of a 36 s forward; measured and left, since it belongs in
  `backend-cpu`'s own sweep rather than here.

#### 6 - two things audited in the same files, one of which was a false alarm

**`matmul_gemv`'s registration in `crates/ltxv/src/block.rs` is NOT dead, and
removing it would have been a silent corruption.** The claim under audit was
that no ltxv shape can reach it, because `model::block::gemm_variant` picks it
only at `m <= 32` and every ltxv linear passes a token count (3520/1024/128),
so the six specialised `matmul_gemv_reg#MREG=n` pipelines `gpu_core::upgrade::
expand` appends at every device open are pure cost. The first half is true of
PRODUCTION shapes and is why it appears in no profile. The second half is
false, and the measurement says so directly - `linear()` instrumented to print
its selected kernel index, over `cargo test -p brain-ltxv --test shard_parity
-- --nocapture`:

    112 PROBE_LINEAR m=6 kind=16 wg=true
     28 PROBE_LINEAR m=4 kind=16 wg=true

Kind 16 is `K_MATMUL_GEMV`. 140 dispatches in ONE test binary: the tiny-config
parity ladder runs text cross-attention at `context_len` 3, 4 and 6, and the
K/V projections there pass `m = context_len`. Since the registration is what
maps index 16, removing it would re-point every one of those at
`kv_k_headt` - a live kernel with different bindings, which panics on a GPU
backend and, per `.agents/rules/lessons.md` #53, silently reads out of bounds
on `backend-cpu`.

The same argument fails for the other diffusion crates too, checked rather
than assumed: `flux1` (`gemv: Some(K_MATMUL_GEMV)`, and `gemm_variant`'s own
doc records that flux1 is the crate that LEARNED to route skinny-M there),
`wan` (per-request modulation at `rows = 1`), `pulid` (its registration
carries a comment naming its own `m = 1` `id_map` chain and `m = 32` `to_kv`
injections) and `instantid` all register it and all have a skinny-M shape.
`sdxlunet` mentions it only in its bench's roofline table. **Dead
registrations found: zero, of five crates checked.** The generalisable form:
"no PRODUCTION shape dispatches it" and "no shape dispatches it" are different
claims, and a kernel index is load-bearing for the whole list whether or not
anything dispatches it.

**`ltxv_bench` now prints the DEFECT block.** It was the only `*_bench` in the
tree calling `print_top` alone, where `mm3_bench`, `wan_bench`, `qwen_bench`,
`unet_bench` and `vqgan_bench` all follow it with `p.defects(r, 5.0)` and print
the rows. That is why this model's roof-floor defects were invisible to its own
harness: a table that ranks kernels but never says which are BELOW their roof
leaves the roofline arithmetic to the reader. Both of this bench's profile
sites (`dit` and `vae`) now go through one `report()` helper with the same
shape and wording the other five use.

It paid for itself on the first run. `./target/release/ltxv_bench dit 2 2 512
128`, against this P40's measured 10517 GFLOP/s / 287.5 GB/s roofline:

    DEFECT  rmsnorm_eps    4.7% of its memory roof (floor 35%) - 10.7% of this pass

`rmsnorm_eps` is the one-thread-per-row form; `rmsnorm_rows` is the
cooperative workgroup-per-row sibling this tree already carries and which
`model::block::rms_variant` already knows how to select. That is kernels.md
§F.3's exact shape - a fast sibling a model never learned about - and it is
now a visible row rather than an arithmetic exercise. NOT fixed in this phase:
it wants its own crossover sweep and its own gate, and the share it can return
has to be re-measured at the real 3520-token width (this reading is at 512).
Recorded as the next target.

### Phase 28 - text cross-attention stops materializing a score slab

Phase 12 fused `attn1` and left `attn2` on the materialized
`attn_scores_cross_kt` -> `softmax_rows` -> `attn_apply_cross` trio, on the
stated grounds that the flash family "cannot express" a different key row set.
That was true of the kernels that existed; it was never a statement about the
algorithm. This phase writes the one kernel that can.

It also closes Phase 27's own recorded next target (`rmsnorm_eps` vs
`rmsnorm_rows`), which turned out to be one line once measured at the real
width.

#### 0 - re-measure first, because the previous ranking was taken at the wrong width

Phase 12 and Phase 27 both profiled at `T = 3520`. A real generation at this
crate's own standard test shape - 25 frames at 1280x704 - is `T = 13200`, and
the ranking there is not the ranking at 3520: self-attention is O(T²) and
everything else is O(T), so every share moves. Optimizing a shape nobody
generates at is wasted work (§F.1's "the group table is an UPPER BOUND" has a
sibling: a table taken at the wrong SHAPE is not an upper bound of anything).

**Method**: `BRAIN_PROFILE=1 BRAIN_LTXV_DIT=<real 22B distilled Q8_0 GGUF>
./target/release/ltxv_bench streamed 8 13200 1024 1 1 1`, one Tesla P40,
`nvidia-smi` confirming both cards idle BEFORE each run and never sampled
during one. The numbers below are the **cache-hit** arm (call 2 - the shape
every forward of a generation past the first has), taken as the DIFFERENCE
between the cumulative kernel table printed at the end of call 2 and the one
printed at the end of call 1, since `BRAIN_PROFILE`'s tables are cumulative
over the process.

**Before** - 14471.6 ms of GPU kernel time per 8-layer cache-hit forward:

    flash_attn_bidir_reg2   5761.9 ms    8 calls  (39.8%)
    matmul_i8_dyn           3223.8 ms   80 calls  (22.3%)
    attn_apply_cross        1815.4 ms    8 calls  (12.5%)
    attn_scores_cross_kt    1640.3 ms    8 calls  (11.3%)
    rmsnorm_eps             1060.2 ms   56 calls  ( 7.3%)
    softmax_rows             230.2 ms    8 calls  ( 1.6%)
    matmul_reg3               60.9 ms   32 calls  ( 0.4%)

Against this card's measured 10517 GFLOP/s / 287.5 GB/s roofline, the
cross-attention trio is a DEFECT by this repo's own 5%-of-both-roofs rule:
`attn_scores_cross_kt` does 110.8 GFLOP in 205.0 ms per layer (540 GFLOP/s,
5.1% of the compute roof) and `attn_apply_cross` the same 110.8 GFLOP in
226.9 ms (488 GFLOP/s, 4.6%), while writing and re-reading a
`[32, 13200, 1024]` fp32 score slab and its probabilities twin - 1.73 GB each.
`flash_attn_bidir_reg2` in the SAME block does 2855 GFLOP per layer at 37.7%
of the same roof, doing 13x the arithmetic in 1.6x the time.

#### 1 - the recorded diagnosis for `flash_attn_bidir_reg2` is WRONG, and it was checked before it was believed

A note carried into this session blamed that kernel's 37.7% on
`head_dim = 64` against a 128-wide compile-time tile, i.e. half of every tile
zero-filled. That is not what this model dispatches. `LtxDitConfig::
ltx25_22b()` is `inner_dim 4096` / `num_heads 32`, so `head_dim()` is 128 -
exactly `HD`, no zero fill at all - and `ltxv_bench dit`'s own banner prints
`head_dim 128` at the real config. A `head_dim`-specialised variant would
therefore be a `kernels::template` knob that compiles to the same kernel.

What the kernel is actually near is a shared-memory ISSUE ceiling, not a
padding waste: its two inner loops each retire 8 fused multiply-adds per
`vec4` shared load, and a Pascal SM issues four times as many FFMA lanes per
clock as LSU lanes, so the shared traffic is the same order of magnitude as
the arithmetic. Raising the ratio means more query rows per thread, and the
48 KiB of workgroup memory it already declares is the Vulkan/NVIDIA limit
exactly - a third row needs the tile geometry rebuilt AND lands `q0..q2` /
`o0..o2` at ~192 registers before anything else, which is the spill this whole
family exists to avoid. NOT attempted here, and recorded as "near its
structural ceiling for this tiling" rather than as a target.

#### 2 - `flash_attn_cross_reg2`, the kernel that did not exist

§F.3 first, and the grep came back genuinely empty: `flash_attn_bidir_*` all
derive both tile counts from one `tcols` and read one fused `[t, 3*d_model]`
slab; `flash_attn_causal_gqa` has separate q/k/v but still one `tcols` and a
`j > i` mask. Neither can express two independent lengths.

The new kernel is `flash_attn_bidir_reg2` with exactly two changes - three
separate buffers with their own strides/offsets, and `t_dec`/`t_enc` split -
and nothing else: same BR=128 two-query-row register block, same vec4 shared
tiles, same software-pipelined K/V staging, same two barriers per tile, same
lane/bank ownership. It needs no `pack_qkv`: `attention()` already produces
q, k and v as three plain `[rows, inner_dim]` buffers, which is the operand
shape.

The seam is shared, not local (§F.7): `model::block::{flash_cross_supported,
flash_cross_step, FlashCrossLayout}` alongside the existing
`flash_bidir_fwd`, gated on queried `DeviceCaps` (workgroup reductions,
`max_workgroup_size >= 256`, `workgroup_mem_bytes >= 49152`) and never on a
backend name, with `BRAIN_NO_FLASH_CROSS=1` as the A/B switch the measurement
below was taken with (§F.6: without it a sweep on a capable device compares
the fused path against itself and reports a meaningless 1.00x). Any model with
a cross-attention trio can adopt it in one call; `ltxv` is the first.

**After** - 11273.0 ms per 8-layer cache-hit forward:

    flash_attn_bidir_reg2   5750.9 ms    8 calls  (51.0%)
    matmul_i8_dyn           3213.9 ms   80 calls  (28.5%)
    rmsnorm_eps             1088.6 ms   56 calls  ( 9.7%)
    flash_attn_cross_reg2    486.8 ms    8 calls  ( 4.3%)

Text cross-attention, all four kernels of it (`kv_k_headt` +
`attn_scores_cross_kt` + `softmax_rows` + `attn_apply_cross`, 3688.5 ms) is
now one 486.8 ms dispatch - **7.58x** - and it runs at 3642 GFLOP/s, 34.6% of
the compute roof, against 4.6-5.1% for the two kernels it replaces. It also
stops allocating 3.46 GB of score+probability slab per layer plus the 16.8 MB
key-minor transpose scratch.

#### 3 - `rmsnorm_rows`, which had been sitting in the tree the whole time

Phase 27 ended by recording `rmsnorm_eps` as a §F.3 case and declining to fix
it without a real-width measurement. At 13200 tokens it is 1060.2 ms per
8-layer forward, 7.3% of GPU kernel time, and it is the one-thread-per-row
form: thread `t` owns row `t` and walks all 4096 floats of it, so a warp's 32
loads are 16 KB apart and each fetched sector serves one useful float.

`rmsnorm_rows` is the coalesced workgroup-per-row twin. It takes the SAME
three buffers and the SAME `[d, rows, eps]` Params, `model::block::
rms_variant` already implements the selection rule (`backend_api::select`'s
`Op::RmsNorm`, keyed on `DeviceCaps`), and `wan` and `flux2` both already
register it. Adopting it in `ltxv` is one registration and one call site.

**After** - 10308.8 ms per 8-layer cache-hit forward:

    flash_attn_bidir_reg2   5754.0 ms    8 calls  (55.8%)
    matmul_i8_dyn           3225.0 ms   80 calls  (31.3%)
    flash_attn_cross_reg2    486.7 ms    8 calls  ( 4.7%)
    rmsnorm_rows             108.6 ms   56 calls  ( 1.1%)

**9.76x** on that row, for a one-line selection change - which is the whole
point of the meta-rule: the expensive defect in this repo is not a slow
kernel, it is a fast kernel a later model never learned about.

#### 4 - the whole-pass numbers, and why they are believable

| | before | after cross | after cross + rmsnorm |
|---|---|---|---|
| text cross-attention, 8 layers | 3688.5 ms | 486.8 ms (**7.58x**) | 486.7 ms |
| RMSNorm, 8 layers | 1060.2 ms | 1088.6 ms | 108.6 ms (**9.76x**) |
| GPU kernel time, 8 layers | 14471.6 ms | 11273.0 ms | 10308.8 ms (**1.404x**) |
| wall, 8 layers (cache hit) | 27.88 s | 23.61 s | 23.20 s (**1.202x**) |

The three runs are separate processes minutes apart, so the rows that did NOT
change are the control: `flash_attn_bidir_reg2` measured 5761.9 / 5750.9 /
5754.0 ms and `matmul_i8_dyn` 3223.8 / 3213.9 / 3225.0 ms across them, a
spread under 0.4% in both cases. A contended card would have moved those
first, so the deltas on the rows that did change are the change.

Wall moves less than device time because a cache-hit forward at this width
also spends ~3.2 s uploading activations/context/adaLN and ~1.4 s building
RoPE tables on the host, neither of which this phase touches. Those are now
the next targets, ahead of any kernel: `rope2d`'s table build is host f64 and
the upload is per-forward.

#### 5 - gates, and what each mutation proved

New, in `crates/ltxv/src/block.rs`'s own `tests` module:

- `flash_cross_attention_matches_the_materialized_reference_and_a_host_oracle`
  - six shapes x three implementations (the fused kernel, the materialized
  trio, a host f64 oracle). The shapes are chosen so no single index confusion
  survives: `nq > nk` AND `nq < nk` both appear (a length swap cannot pass by
  symmetry), `nq` both a multiple and not a multiple of the 128-row query
  tile, `nk` both a multiple and not a multiple of the 16-row key tile and
  once smaller than one whole tile, `head_dim` at the real 128 and at widths
  that leave the tile zero-filled. `max_abs` AND `rel_l2` are asserted
  alongside cosine, never cosine alone (lesson #2 - cosine is scale-invariant).
- `fused_cross_attention_averages_the_right_v_rows_under_a_uniform_softmax`
  - the V-INDEX gate the random test cannot give. `q` is zero, so every score
  is exactly 0, the softmax is exactly uniform, and the answer is ANALYTIC:
  the per-channel mean of `v`. `v`'s rows are deliberately unequal, so reading
  `v` at the wrong stride or by the query row lands elsewhere entirely.

The reference arm is now reached by calling `attn_context_materialized`
directly rather than by passing `self_attn = false`. That steering was safe
only while cross-attention had no fused kernel; keeping it would have compared
one fused kernel against another and never touched the reference - the exact
"a sweep cannot see below its own threshold, so it will confirm whatever you
wrote" failure of §F.6, in gate form.

**Mutation-verified, six mutations, each RED then restored GREEN** (measured
at heads=32 / head_dim=128 / nq=300 / nk=128 against a 1e-5 bar):

| mutation | max_abs | cosine | rel_l2 | caught by |
|---|---|---|---|---|
| K read transposed (row/channel swapped) | 1.259e-1 | 0.8998 | 4.477e-1 | the oracle test |
| K and V operands swapped | 3.678e-1 | 0.00542 | 1.397e0 | BOTH; the v-mean test at 4.802e1 vs a 1e-3 bar |
| online-softmax `corr` rescale dropped | 1.846e-1 | 0.9858 | 2.163e-1 | the oracle test |
| score scale multiplied by 1.02 | 1.936e-3 | 0.999979 | 6.864e-3 | the oracle test |
| `nq`/`nk` swapped at the dispatch site | 2.633e-1 | 0.6527 | 8.411e-1 | BOTH; v-mean at 4.801e1 |
| `rmsnorm_rows` dispatched at `rows` threads | - | - | - | `dit_parity`, all three cases |

Two things that table says and a single mutation would not. First, the v-mean
test is genuinely orthogonal, not a duplicate: it passed unchanged under the
transposed-K, dropped-rescale and wrong-scale mutations (with `q = 0` the
scores are irrelevant) and caught the two that touch V. Second, the SCALE
mutation is the one that comes closest to slipping through: it moves cosine
only to 0.999979, which is four orders of magnitude less alarming than the
transposed-K mutation's 0.8998 while being a far more plausible bug, and
`rel_l2` at 6.864e-3 is what states its size honestly. A gate on cosine with a
looser floor would have shipped it.

Pre-existing gates re-run and confirmed from PRINTED output (a skipped test
reports as a pass in this repo, so "it passed" is not evidence):
`dit_parity` (all three, including `real_weight::ltxv_real_dit_tiny_layers_
matches_reference` against the real Q8_0 GGUF - `b0_attn2_out`, which is
exactly the kernel this phase replaced, at cosine 1.000000000 / max_abs
3.278e-7), `host_forward_parity` (cosine 1.0000000000, max_abs 1.550e-6),
`shard_parity` (both cases bit-identical, max_abs 0.000e0),
`streamed_vs_eager_real` (cosine 1.000000000, max_abs 0.000e0),
`connector_real_parity` (cosine 0.999999973) and `int8_compute` (final output
cosine 0.999999989, and the real Q8_0 block-0 case at 0.996303608).

#### 6 - recorded, not done

* **`devres::activation_reserve_bytes` is now over-fitted, and it was ALREADY
  saturated at the standard test shape.** Its wgpu slope was fitted to a
  measured VRAM plateau at T=3520 and its Vulkan slope derived analytically
  from "`attn2`'s `[heads, t, context_len]` score+probability pair plus ~20
  activation buffers" - and that pair no longer exists. 3.46 GB per in-flight
  layer has come free. But the bigger finding is what the same profiling run
  prints without being asked:

      [call 2] DEVICE RESIDENCY: slots=0 device_hits=0 device_uploads=0

  Zero resident blocks, at `resident=1`, at the width a real generation runs.
  The wgpu slope extrapolates to tens of gigabytes at 13200 tokens, so
  `card - reserve` underflows to nothing and the policy declines residency
  entirely - meaning every block's weights cross to the device on every
  forward, at exactly the shape the residency machinery was built for. A
  linear fit taken at one token count is not a model of a pool whose growth is
  not linear in that count.

  Deliberately NOT re-fitted here, and not guessed at: under-reserving costs a
  driver-level abort, so a re-fit needs a plateau sweep ACROSS token counts
  (and a re-measured plateau now that the slab is gone), not one point and an
  argument. Recorded as the next target, ahead of any kernel - it is worth
  more than the remaining kernel headroom and it is not a kernel problem.
* **The A<->V cross-attentions now take the fused path too** (they are
  `!self_attn`), at the audio stream's `head_dim = 64`, which leaves half of
  every 128-wide tile zero-filled. Correct - the gate covers head_dim 8, 16
  and 64 - but not measured at AV scale, because nothing wires the AV DiT into
  a pipeline yet. If that lands, measure before assuming the fused path wins
  there too (§C5: a tile sized for the worst case is a cost at every other
  case).
* **The cross ladder has ONE rung.** `flash_cross_supported` is a bool, not a
  `flash_bidir_variant`-style walk, because there is no 16 KiB sibling. A
  device under 48 KiB of workgroup memory keeps the trio. If a second rung is
  ever written, that bool becomes a ladder and every caller inherits it.

### Phase 29 - the sound stops ending at the first window seam

Phase 22 made a clip longer than one denoising window into one continuous
shot; the audio wiring that landed after it worked, and worked only for a clip
that fits ONE window. At 1280x704 that is a hard ceiling of 15 latent frames =
113 frames = 4.71 s at 24 frames a second, because `LONGFORM_MAX_TOKENS` is
13200 and a latent frame there is 22 x 40 = 880 video tokens. A ten-second
request with `--audio` was refused outright, in the CLI, with a message saying
the seam had not been designed. It has now.

The mechanism is the video half's, unchanged: the previous window's own last
audio tokens, sliced out of the denoised latent before anything is decoded
(`audio::carry_tail`), written over the head of the next window's audio
sequence and frozen at sigma 0 - `denoise_mask == 0`, per-token timestep 0,
re-pinned by `post_process_latent` on both the x0 estimate and the stepped
latent, exactly the two applications the video's carried prefix gets. The
windows contribute tokens to ONE audio latent, decoded once when the loop
ends. Two rolling slabs of `context_tokens x 128` floats are the whole of the
new state; at the default context that is 59 tokens = 30 KB, and it does not
grow with the clip.

#### 1 - the alignment rule, and why it constrains the WINDOW PLAN

The two streams do not share a time resolution and cannot be made to. One
video latent frame is `VAE_TEMPORAL_SCALE = 8` pixel frames, so it is
`8 * LATENT_RATE / fps = 200 / fps` audio tokens - `25/3` at 24 frames a
second. A window seam is a re-basing of both streams onto the new window's own
time origin, and it is exact only when both streams shift by the SAME amount
of time:

    video shift = (origin(w+1) - origin(w)) / fps   seconds
    audio shift = (tokens(w) - carried)  / LATENT_RATE   seconds

Setting them equal and clearing denominators gives the whole rule as an
integer identity, `pixels_advanced * LATENT_RATE == tokens_advanced * fps`,
which has an integer solution for `tokens_advanced` iff `200 * advance` is
divisible by `fps`. So an audio-visual plan may only advance by multiples of
`audio::window_latent_frame_quantum(fps)` - 3 at 24 and 30 frames a second, 1
at 25, 1 at anything that divides 200. `longform::window_plan_aligned` places
seams accordingly and `audio::audio_plan` re-derives the layout from the
finished plan and REFUSES rather than rounding.

Three things fall out of it, and the third is the one that took a wrong turn
first:

* **The carried count is then forced, and it is the SAME at every seam**:
  `ac = ta_prev - 200*advance/fps`. Because window 0's advance is also a whole
  quantum, `ta_prev` and the context's own span have the same fractional part,
  so the difference of their `round`s is exactly the difference of their
  arguments - which makes `ac` equal to `latent_frames(dropped_frames, fps)`,
  i.e. the audio token count of the carried prefix considered as a clip in its
  own right. The rule and the obvious guess coincide, but only *given* the
  quantum; without it they differ by up to a token and nothing says so.
* **The totals then close exactly**: window 0 contributes `ta_0`, every later
  window `200*new/fps`, and those sum to `round(frames/fps*LATENT_RATE)` - the
  clip's own token count, the same number a single-window clip of that length
  would carry. The decode is therefore the same `(4*ta - 3) * HOP_LENGTH`
  samples it always was and the container's two stream durations do not move.
* **The LAST window must NOT be constrained.** Constraining every window makes
  `k_total` congruent to `context - 1` modulo the quantum, which refuses most
  legal `1 + 8k` lengths outright - 241 frames at the default 8-frame context
  is one of them. The last window has no successor and hands nothing across,
  so leaving it free costs nothing and is what makes any length plannable.
  This was the first design and it was wrong; the arithmetic above is what
  said so.

The refusal path is real and is reachable: at a frame rate whose quantum does
not fit the grid (23 frames a second at 1280x704 needs 8 carried + 23 new
latent frames against a 15-frame budget) the plan is refused before any weight
is read, with `audio::quantum_note` naming the ratio, the quantum, and the
three things that would work. Frame rates that divide 200 need no quantum at
all.

#### 2 - what the gates are, and what each mutation proved

`crates/ltxv/tests/audio_seam.rs`, 8 tests, all arithmetic on the SAME
functions the pipeline calls - `audio::positions`, `pipeline::
real_pixel_positions`, `audio::audio_plan`, `longform::window_plan_aligned` -
over a sweep of 4 frame rates x 8 multi-window lengths at the real 22 x 40
latent grid, plus a 120-frame-rate backstop.

| mutation | what it models | caught by |
|---|---|---|
| `context_tokens() + 1` | the carried span computed one token long | the TOTAL check in `audio_plan` (125 tokens where a 121-frame clip is 126), which fails 4 of the 8 tests |
| `AudioPlan.context + 1`, total left correct | **the audio context shifted by one token at a seam** - this change's real failure mode, and the one the total cannot see | the POSITION gate: "carried audio token 0 ends at 0.01 in its new window and 1.97 in its old one, a shift of 2 - the sound is -0.04 s away from the picture it belongs to"; also the integer shift identity (1200 vs 1176) and the `a.context` equality |
| `window_latent_frame_quantum() -> 1` | the alignment rule removed | 5 of 8, including the refusal test and the quantum's own definition |
| `carry_tail` returns the FIRST k tokens | the tail taken from the wrong end | only the carry-tail identity test - which is why that test exists separately from the arithmetic |
| `latent_frames(w.emitted_frames())` for a window's own token count | a plausible confusion of the two frame counts a window has | 4 of 8 |

The second row is the one to read. A one-token context shift keeps every token
count self-consistent, so a gate on counts alone passes it; what fails is
comparing POSITION VALUES between the two windows' real RoPE tables, which is
what turns "the numbers add up" into "the sound is where the picture is".

#### 3 - the per-window cost of the seam, and what actually costs

Carrying the audio adds no per-window weight traffic at all: the rolling state
is two `59 x 128` slabs, the per-window work is one `seeded_noise` of
`ta_w * 128` values and two `copy_from_slice`s, and the audio VAE + vocoder
are read and run ONCE for the whole clip rather than per window. The audio
stream's share of a forward's tokens is 109 against 3080 at stage 1 and 76
against 12320 at stage 2 - the cost of `--audio` is the AV BLOCK, not the
audio tokens, and that block is a separate workstream.

Four per-window costs that were in the long-form path and are now not, all of
them the same class (work at a seam that could have been carried or cached):

* the **spatial x2 latent upscaler was re-read and re-imported off disk once
  per window** (and once per pass in `upscale`) - a ~1 GB checkpoint plus its
  bf16 expansion, every seam. It is now a `SpatialUpsampler` cache filled at
  most once per run, and filled LAZILY, so an audio-visual generation does not
  hold its expansion beside the host-fp32 transformer through a stage that
  never touches it. Only `LatentUpsampler::build` is genuinely per window.
* the **audio VAE checkpoint was read twice** (once for the VAE, once for the
  vocoder, because `StTensor` is not `Clone`). One read and a `partition` on
  the prefix each importer already selects by hands both their half.
* the **audio stream's text projection was copied per stage** into
  `AudioState`; it is borrowed now.
* the **projected caption was cloned** out of `TextContext` in `generate`, and
  **every decoded frame was cloned** in `ltxv_cli` to hand it to the encoder -
  both are moves now. The frame clone is the clip's whole pixel buffer.
* `stage2_sigmas` was rebuilt per window and is now resolved once, before the
  first forward, so a refinement schedule the run cannot spell costs
  milliseconds rather than a window of device time.

#### 4 - recorded, NOT done

* **No real multi-window audio-visual generation has completed on this box,
  and the waveform verification is DEFERRED rather than dropped.** Seam
  continuity in the samples (max sample-to-sample step at each seam against
  the 99.9th percentile elsewhere) and the plain audio statistics (peak,
  clipped count, RMS dBFS, longest near-silent run, L/R correlation) are NOT
  measured and are not claimed. They are owed on a real multi-window clip once
  the AV block has a quantized path: a seam figure taken on the host-fp32 path
  would have to be retaken anyway, because int8 changes the samples it is
  measured on.

  The measurement HARNESS is ready and was validated before it was trusted, on
  a synthetic clip with a click planted at the exact seam sample: 9.72x the
  99.9th-percentile step on the clicked channel against 0.57x on the clean
  one. The seam's sample index is not estimated - audio token `i` covers mel
  frames `[4i-3, 4i+1)` and each is `HOP_LENGTH` samples, so the first sample
  window `k+1` contributed is `(4*T - 3) * HOP_LENGTH` for the cumulative
  token count `T` before it.

* **Two attempts, and they did NOT fail the same way - the first diagnosis was
  wrong and is recorded here so it is not repeated.** Attempt one died with
  `wgpu error: Out of Memory` on the third forward of window 0 stage 1, which
  looked like a per-forward leak in the AV path. It was not. A single-window
  control at the same resolution ran that same third forward clean, and
  sampling `nvidia-smi` through it shows the AV forward oscillating between
  roughly 0.7 and 5.9 GiB of a 24 GiB card with no growth across forwards. The
  OOM was a co-tenant on the same card, not a requirement of this path - which
  also says the host-fp32 AV path has no headroom against anything else
  sharing its GPU. Attempt two was killed deliberately, not by a fault: it
  held ~90 GB of RSS on a box with no swap, and the fp32 baseline the
  quantized-AV workstream needs costs ~84 GB, so the two could not be resident
  at once.

* The end-to-end claims this phase makes are therefore the arithmetic ones,
  which are exhaustively gated, plus the plan resolution - `--frames 241
  --width 1280 --height 704 --fps 24 --audio` now reports "4 windows, +251
  audio tokens (10.04s of 16 kHz stereo)" where it used to refuse.
* **A multi-scene clip still refuses `--audio`**, now from the library rather
  than only the CLI. A scene boundary deliberately carries nothing, so the
  sound would restart at every cut AND the per-scene token counts do not sum
  to the clip's own. That is a design question about what sound should do at a
  visual cut, not an arithmetic one.

### Phase 30 - the audio-visual forward stops being a host-fp32 upload benchmark

The audio-visual DiT block had no quantized, streamed or device-resident path
at all: no `LtxAvBlockQ`, no AV cached-weight type, no AV session. An
`--audio` run therefore expanded the whole 21.0 B-parameter checkpoint to host
fp32 and re-uploaded every block on every forward. What that cost, recorded
before this phase: 96.7 GB of RSS, nine minutes before the first forward, the
card at 0% while the host churned, and a joint forward 5.4x a video-only one
at 19-24% GPU utilisation. It was upload-bound, not compute-bound - audio and
the two cross-modal attentions are only 29% of the parameters.

`LtxAvBlockQ`, `CachedQAvBlockWeights` and `AvDitSession` close that: the same
28-linear int8 tier the video block uses applied to both streams and both
cross-modal directions, the same host-RAM `GenerationCache`, and the same
Belady-planned device residency window over a shared generic `BlockWindow<B>`.

#### 1 - what a forward costs now, at the width a real generation runs at

One Tesla P40 (24576 MiB, wgpu/Vulkan), real
`ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, all 48 layers, int8, ctx 1024,
one distinct timestep, device-resident session, T = 13200 video tokens
(`longform::LONGFORM_MAX_TOKENS`: 15 latent frames of a 1280x704 clip, 113
frames at 24 fps) and its own 118 audio tokens. Harness:
`ltxv_bench streamed 48 13200 1024 1 1 1 3` and
`ltxv_bench streamed-av 48 13200 1024 118 1 1 1 3`, four calls each, the first
warm call discarded as warm-up, best of the remaining two. Both cards idle
before each run, nothing sampled during one, no build running.

| | video-only | audio+video | AV / video |
|---|---:|---:|---:|
| first forward (cold: GGUF read+dequant+quantize of all 48 blocks, plus the prefill) | 163.59 s | 197.80 s | 1.21x |
| warm forward, best of 2 | **93.59 s** | **111.15 s** | **1.19x** |
| of which DEVICE (kernel timestamp queries) | 61.88 s | 68.24 s | 1.10x |
| of which HOST | 31.70 s | 42.91 s | 1.35x |
| device share of wall (kernel time / wall - MEASURED, from timestamp queries) | 66.1% | 61.4% | |
| `nvidia-smi utilization.gpu`, mean over a warm forward (a separate UNTIMED observation run) | 85.7% | 83.5% | |
| peak VRAM over a whole run (same observation run) | 18092 MiB | 21082 MiB | |
| resident blocks the policy granted | 23 of 48 | 16 of 48 | |
| int8 bytes per block (`cached_block_bytes` / `cached_av_block_bytes`) | 257.6 MiB | 371 MiB | 1.44x |
| host peak RSS (`/proc/self/status` `VmHWM`) | 36884 MiB | 52591 MiB | 1.43x |

**The BEFORE arm, reproduced at the same width rather than cited.** The
host-fp32 audio-visual path still exists (`ltxv_bench av 13200 1024 113 24`,
`av_stream::AvDenoiser`) and was run on the same card on the same day, as an
untimed observation (nvidia-smi polled throughout):

| | host fp32 (before) | int8 streamed + resident (now) | |
|---|---:|---:|---:|
| weight load + dequant, before any forward | 88.9 s | (none - folded into forward 1) | |
| forward, best of 2 | 478.9 s | 111.15 s | **4.31x faster** |
| host peak RSS | 113071 MiB | 52591 MiB | **2.15x smaller** |
| `nvidia-smi utilization.gpu`, mean over one forward | 38.4% | 83.5% | |
| peak VRAM | 15814 MiB | 21082 MiB | |
| AV forward / video-only int8 forward | 5.12x | **1.19x** | |

The 5.4x this workstream started from reproduces (5.12x against today's
video-only forward). The 96.7 GB of RSS recorded for it turns out to have been
an under-count: measured here it is 110.4 GiB, and the load is 88.9 s rather
than nine minutes only because the 23.6 GB GGUF was already warm in page
cache. Two of the four before-numbers therefore move, both in the direction
that makes the fp32 path look worse, not better.

**The ratio is 1.19x, not 5.4x.** A correction to the previous pass's own
last measurement, which reported 1.15x: the video-only half reproduced exactly
(it reported 108.0 s, this pass measured 108.60 s on the same code), the
audio-visual half did not - three warm calls measured 127.70 / 130.49 / 129.94
s against its 123.7 s. Trust the reproduced pair, not the single sample.

RSS is the other half of the claim and it moved as intended: 51.4 GiB peak for
the whole audio-visual run against the 96.7 GB the host-fp32 path needed. The
weights themselves are 17828 MiB of int8 across 48 AV blocks where the fp32
expansion was ~84 GB; most of the remaining RSS is the mmapped 23.6 GB GGUF,
which is reclaimable page cache rather than anonymous memory.

The two utilisation figures are different measurements of different things and
both are reported because neither substitutes for the other: the device share
is `Gpu::kernel_times`' accumulated DEVICE time over the call's wall clock -
the fraction of the forward in which a kernel of this process was executing -
while `nvidia-smi utilization.gpu` is the fraction of 1-second sampling
windows in which ANY kernel was running, which is necessarily the larger of
the two and says nothing about how full the card was while it ran. The gap
between them (66% against 86%) is the small-dispatch tail this port has a lot
of: 2176 `rope2d` dispatches a forward, none of them long.

There is no separate load phase to time any more. The GGUF read+dequant
(68.5 s), the int8 quantize (9.5 s) and the resident prefill are all inside
the FIRST forward, which is why the first forward is 197.80 s and every later
one is 111.15 s.

#### 2 - the debt this phase removed, and what it was worth

**The activation crossed PCIe twice per BLOCK.** `DitSession::run_blocks` and
its AV twin chained `x` between blocks as a HOST `Vec<f32>`: every block
uploaded a `[t, dim]` activation, ran, and read the whole thing back. At
T=13200 that is 216 MB up and 216 MB down per block, 48 blocks, ~20.8 GB of
round trip per forward whose only purpose was to arrive back where it started.
`run_blocks`'s own doc claimed the opposite ("`x` is uploaded ONCE, chained
block to block as a device buffer, and read back once") - the doc was aspirational,
the code was not.

An earlier pass had measured chaining and REJECTED it: leaving `x` on the card
removes the one blocking readback per block that makes wgpu's allocator pool
shrink, and the pool then grew several-fold, costing more resident blocks than
the traffic was worth. That measurement was right and its conclusion was too
strong. The drain needs the readback to be BLOCKING, not to be BIG:
`backend_wgpu`'s `read` is `flush` + `map_async` + a bounded `poll_wait`
whatever `n` is, and only the copy scales. So `forward_prod_dev` chains the
device buffer and drains with a ONE-WORD read off the block's own output -
exactly the probe `DitSession::prefill` already used between weight uploads.

Measured on the same harness, same day, same card:

| | video-only warm | audio+video warm |
|---|---:|---:|
| before | 108.60 s | 129.94 s |
| after | 93.59 s | 111.15 s |
| | **-13.8%** | **-14.5%** |

and the split says exactly where it went (video, warm, per forward):

| stage | before | after |
|---|---:|---:|
| block submit+wait (contains the per-block readback) | 80.02 s | 63.71 s |
| activation/context/adaLN record+upload | 15.51 s | 14.30 s |
| device->host readback (per forward) | 0.00 s | 0.37 s |

Device kernel time did not move (61.84 s -> 61.88 s), which is the whole
signature of a host/PCIe change; the DiT output's mean/std/min/max are
identical to six decimals on both streams, and the numeric gates below cover
the rest. The readback was the expensive half by an order of magnitude: 10.4 GB
down cost ~16 s (~0.65 GB/s) where 10.4 GB up cost ~1.2 s.

**Two smaller items on the same path**, both in `self_attn_and_text_ca_q`'s
prompt scale/shift broadcast: `1.0 + scale` was recomputed per ELEMENT of a
`[context_len, dim]` buffer rather than once per row, and both broadcasts were
handed to `write_buffer` as one whole-payload call, which on a non-ReBAR card
needs a staging allocation the size of the payload - the same reason
`devres`'s own table uploads are chunked.

**Three stale references to a `LtxBlockQ::forward_chained` that does not
exist** (in `block.rs` and `devres.rs`) now name `forward_prod_dev`.

**A red test.** `device_residency.rs::the_slot_policy_never_over_promises`
still asserted `slots == 0` at 1080p - the exact behaviour the reserve re-fit
had just removed - so `cargo test -p brain-ltxv` was failing before this pass
touched anything. It now asserts the property that matters (a NONEMPTY,
partial window at every width a real generation uses, and a window that keeps
shrinking past them), mirroring `devres`'s own unit test.

#### 2b - where the DEVICE half goes, and why it is not the next target

The per-kernel table at the real width (video, T=13200, `BRAIN_PROFILE`,
cumulative over four calls) is dominated by two rows and nothing else is close:

| kernel | share of device time |
|---|---:|
| `flash_attn_bidir_reg2` (self-attention) | 55.7% |
| `matmul_i8_dyn` (the ten int8 linears) | 31.1% |
| `flash_attn_cross_reg2` (text cross-attention) | 4.7% |
| everything else, largest row `rmsnorm_rows` | <= 1.1% each |

Graded against this box's measured roofline (10517 GFLOP/s fp32, 43560 GOP/s
int8 DP4A), per warm forward: the self-attention does 1.37e14 FLOP in ~34.5 s,
i.e. about 38% of the fp32 roof, and the int8 linears do 3.0e14 OP in ~19.3 s,
about 36% of the DP4A roof. Neither is anywhere near a defect floor and
neither has an obvious algorithmic change left, so at this width the device
half is close to structural and the HOST half - 31.7 s of a 93.6 s video
forward, 42.9 s of a 111.2 s audio-visual one - is where the remaining wins
are. That is what section 6 lists.

Note this ranking is only true at the real width. The same profile at 880
tokens puts `matmul_reg3` (an fp32 helper) at 71% of device time and
`flash_attn_bidir_reg2` at 5.5% - the exact inversion that makes a
small-token-count profile useless for deciding what to optimise.

#### 3 - the residency arm: what the reserve actually is at real width

`devres::activation_reserve_bytes`'s wgpu per-token slope was fitted at
T=3520 and extrapolated, so at production width `card - reserve` underflowed
and the policy declined residency ENTIRELY (`slots = 0`) - on the video path
as well as the audio-visual one. The re-fit is confirmed working at real
width: the policy grants 23 of 48 blocks on the video path and 16 of 48 on the
audio-visual one at T=13200, and both run to completion with no driver-level
abort.

The plateau sweep, run as UNTIMED OBSERVATION (nvidia-smi polled at 1 Hz
throughout, so its wall figures are not headline numbers), 48 layers, int8,
policy-chosen slot count, peak VRAM the maximum sample on card 0:

| video tokens | slots granted | peak VRAM | resident weights | non-weight peak | warm forward |
|---:|---:|---:|---:|---:|---:|
| 3520 | 23 | 16728 MiB | 5925 MiB | 10803 MiB | 23.34 s |
| 6160 | 23 | 15192 MiB | 5925 MiB | 9267 MiB | 41.40 s |
| 8800 | 23 | 16728 MiB | 5925 MiB | 10803 MiB | 59.87 s |
| 11440 | 23 | 16728 MiB | 5925 MiB | 10803 MiB | 79.43 s |
| 13200 | 23 | 18092 MiB | 5925 MiB | 12167 MiB | 94.22 s |
| 16720 | 5 | 15658 MiB | 1288 MiB | 14370 MiB | 129.29 s |

**Read the peak column, not the token column.** Peak VRAM is FLAT across a
3.25x range of token counts and then goes DOWN at the widest one, where the
policy holds the FEWEST weights. It is not a function of the token count -
wgpu's allocator pool sizes itself to what is free rather than to what is
needed, so most of the per-token slope the reserve encodes is absorbed by the
pool rather than paid for. That is the same non-monotonicity the constant's
own doc already records from one point; this is it measured across the axis
the fit is taken over.

Two things the sweep DOES pin down, and both say the fit is sound:

* **Where each constraint binds.** The slot count is 23 at every token count
  up to and including 13200, which is exactly `card / MAX_CARD_FRACTION_DENOM
  / cached_block_bytes` - the quarter-of-the-card cap that exists so the VAE
  decode after the denoise loop has room. The RESERVE only takes over past
  that: at 16720 tokens it grants 5. So over the whole supported single-window
  range the reserve's job is to be large enough not to bind and small enough
  not to underflow, and it is both. The failure this replaced - `slots = 0` at
  every width past 720p - cannot recur without the reserve exceeding a whole
  card.
* **How much margin the reserve carries near and past the crossover.**
  `activation_reserve_bytes` against the observed non-weight peak: 16294 MiB
  vs 10803 at 11440 (1.51x), 18565 vs 12167 at 13200 (1.53x), 23106 vs 14370
  at 16720 (1.61x). That is the "roughly half again as margin" the constant's
  doc claims, confirmed rather than asserted, and it is the right side to err
  on - under-reserving is a driver-level abort, over-reserving costs resident
  blocks the graceful partial-window path picks up.

**So the reserve is NOT re-fitted in this pass, deliberately.** A re-fit would
be fitting a slope to a peak that does not vary with the variable being
fitted, over a range where the fitted quantity is not the binding constraint,
and the one place it does bind it already carries the intended margin. What
WOULD justify revisiting it: a card whose quarter is bigger than what the
reserve leaves, which moves the crossover inside the supported range instead
of at its top edge. That is a different card, not a different fit.

One thing the sweep does NOT cover, and it is a real gap rather than a
footnote: `activation_reserve_bytes` takes the VIDEO token count and models
the video-only working set. An audio-visual forward carries a second stream
and two cross-modal attentions, and its non-weight peak at T=13200 measures
15146 MiB against the video path's 12167 - so the same reserve carries 1.23x
margin there where it carries 1.53x on the video path. It still fits (21082
MiB peak of a 24576 MiB card) and it fits for a reason that is not the
reserve's doing: an AV block is 1.44x a video block, so the same budget
arithmetic grants 16 slots instead of 23 and the smaller window happens to
leave the extra room. That is a coincidence of this card and this block-size
ratio, not a property of the policy.

One hazard the sweep found and did not have to fix: below roughly 7000 tokens
the reserve is SMALLER than the observed non-weight peak (5.93 GiB of reserve
at 3520 against a 10.5 GiB observed pool), and only the card-fraction cap
keeps the window from being planned on that basis. It is not a live defect -
the pool is elastic and shrinks under exactly the pressure a full window
applies, which is why 3520 runs clean at 23 slots - but the cap is doing
load-bearing work there, not just VAE-decode work.

#### 4 - the int8 audio-visual tier's parity, and what each gate is worth

Every comparison asserts cosine AND relative L2. Cosine alone is scale
invariant, and this phase measured that directly rather than asserting it (see
the mutation table).

| comparison | cosine | rel_l2 | floors |
|---|---:|---:|---|
| tiny gated config, int8 vs fp32 AV block, video out | 0.9999999996 | 3.03e-5 | 0.9995 / 3e-2 |
| the same, audio out | 0.9999999996 | 2.80e-5 | 0.9995 / 3e-2 |
| the same, raw A2V output (pre-gate) | 0.9999676182 | 8.43e-3 | 0.999 / 5e-2 |
| the same, raw V2A output (pre-gate) | 0.9999742007 | 7.39e-3 | 0.999 / 5e-2 |
| the same, video attn1 out | 0.9998416057 | 1.80e-2 | 0.999 / 5e-2 |
| REAL Q8_0 block 0, int8 vs fp32, video out | 0.9985762873 | 5.63e-2 | 0.99 / 1.5e-1 |
| the same, audio out | 0.9988666645 | 4.83e-2 | 0.99 / 1.5e-1 |
| the same, raw A2V output | 0.9995120969 | 3.13e-2 | 0.98 / 2e-1 |
| the same, raw V2A output | 0.9997432730 | 2.31e-2 | 0.98 / 2e-1 |
| REAL Q8_0, WHOLE streamed int8 forward vs eager fp32, video out | 0.9975000074 | 7.78e-2 | 0.99 / 1.5e-1 |
| the same, audio out | 0.9981436150 | 6.15e-2 | 0.99 / 1.5e-1 |

The real-weight block-0 figure (0.9986) sits where the video-only int8 tier's
own does (0.9963) - an int8 tier is not expected to reach 1.0, and the floors
are set below what a clean run measures rather than at it.

The last two rows are a gate that did not exist: `av_forward_q_streamed_in` -
the function a real audio-visual generation dispatches - was called by ONE
bench binary and by nothing else in the workspace. Everything between the
blocks (both patchifies, the six model-level adaLN row tables and their row
maps, the four RoPE table sets, both embeddings connectors, the resident
window's rotation, both output stages) had no coverage at all. It now runs
against the eager fp32 `LtxAvDit::forward` on real weights, with a deliberately
narrow resident window so the rotation is on the tested path, plus a
bit-identity check that a warm forward reproduces the cold one exactly.

#### 5 - the mutations, and which metric caught which

Every gate relied on above was mutation-verified: mutation applied, RED
confirmed, mutation removed, GREEN confirmed.

| mutation | modelled fault | RED in | caught by |
|---|---|---|---|
| A2V `gate_row` replaced by a plain add | the A<->V CROSS-ATTENTION gate always OPEN | `a_closed_cross_modal_gate_...`; real-weight block 0 | the CLOSED arm's bit equality (two different audio latents moved the video output by max_abs 1.9e-3 where the answer must be exactly 0); and cosine 0.595 on real weights |
| `to_gate_logits` gate skipped in `attention_q` | every per-head attention gate always OPEN | tiny block gate; real-weight block 0; whole streamed forward | COSINE on the tiny config's `video attn1_out` tap (0.99812), 0.709 on real block 0, -0.014 on the whole forward |
| int8 block outputs scaled by 1.05 | a systematic gain (a wrong global dequant scale) | tiny block gate | **rel_l2 ALONE** (5.0e-2 > 3.0e-2) with cosine unchanged in all ten printed digits - the direct demonstration that a cosine-only gate passes a systematically wrong result |
| A2V video scale row derived without its `1.0 +` | a device-side modulation ROUTING slip | `device_derived_av_modulation_...` | BIT equality. The change is max_abs 5.7e-5; the whole-forward tolerance gate moved from 0.9975000 to 0.9974601 and passed. No tolerance gate can see it |
| the block-to-block activation chain not advanced | this phase's own change, mis-wired | `streamed_vs_eager_real` (video, cosine 0.709); the new AV streamed gate (cosine 0.697) | cross-implementation comparison ONLY. `a_device_resident_forward_is_bit_identical_to_the_streaming_one` PASSED - both its arms carry the same fault |

Two of those rows are the reason this suite keeps an exact gate and an
analytic gate beside its tolerance gates. An always-open cross-modal gate does
NOT move the tiny-config tolerance gate past a floor sized for a lossy tier
(it moves `video out` rel_l2 from 3.0e-5 to 6.0e-3), and a modulation routing
slip does not move any tolerance gate at all.

#### 6 - recorded, NOT done

* **The next host item is measured and located: device buffer ALLOCATION
  churn.** After the chaining fix, `activation/context/adaLN record+upload`
  is still 14.3 s of a 93.6 s warm video forward. It is not the prompt
  broadcast and it is not the activation: it scales with the TOKEN count
  (4.73 s at 3520, 7.67 at 6160, 10.02 at 8800, 11.77 at 11440), i.e. ~1.5 s
  fixed plus ~0.9 ms per token. That is the ~60 `Gpu::storage` calls each
  block makes for `[t, dim]` and `[t, 4*dim]` scratch, 48 blocks per forward,
  every one a fresh (zero-initialised) wgpu buffer. The fix is a size-keyed
  scratch pool reused across blocks; it needs an aliasing-safe design and its
  own gate (two live buffers must never share a slot), so it is a workstream
  and not a tail-end edit.
* **The prompt scale/shift broadcast is still per block per forward.** Both
  `[context_len, dim]` buffers are a pure function of the block's own
  `prompt_scale_shift_table` and the context width, so they are constant for a
  whole generation; at ctx 1024 they are 32 MiB per block, ~1.5 GiB per
  forward. Caching them on a resident block costs that much VRAM (worth
  several resident blocks); expanding them on the DEVICE from the `[2, dim]`
  row costs nothing but needs a broadcast dispatch this crate does not have a
  clean seam for yet. Deliberately left, with the arithmetic, rather than
  guessed at.
* **The audio-visual path has no training support** and no bandwidth
  extension - unchanged by this phase. It also had no real end-to-end
  generation recorded on this box, which phase 31 closes: nothing in this
  phase was reachable from a generation, because `pipeline::build_denoiser`
  still routed every `--audio` request to the host-fp32 arm.
* The AV forward has no CFG-parallel two-card measurement; every figure above
  is one card, one branch.

### Phase 31 - `--audio` actually takes the quantized path, and a real multi-window clip has sound

Phase 30 built `LtxAvBlockQ`/`CachedQAvBlockWeights`/`AvDitSession` and phase
29 made the audio latent cross a window seam. Neither was reachable from a
generation: `pipeline::build_denoiser` still routed every `--audio` request to
`av_stream::AvDenoiser` and logged "the audio stream has no streamed/quantized
path yet", which had stopped being true one commit earlier. A real `--audio`
run's own banner said `REAL checkpoint audio+video DiT (host fp32, both
streams)`, which is how the gap was found - not by reading the code.

#### 1 - what was wired

`pipeline::RealAvDit` is `RealDit`'s twin, field for field: the open
`LtxvGgufSource`, the non-block head tensors
(`dit::load_av_head_tensors_from_source`), the CHECKPOINT-scoped
`block::GenerationCache`, and a `Mutex<HashMap<card, Arc<AvDitSession>>>`
built on that card's first forward. `Denoiser::forward_av` resolves the
session and calls `dit::av_forward_q_streamed_in`; `Denoiser::release_devices`
drains the map, which is what a window boundary and the stage-1 -> stage-2
boundary already call.

Two structural things came with it rather than being left as comments:

* **One step struct, not two.** `av_stream::AvStepInputs` was deleted and
  `AvDenoiser::forward` now takes the same `dit::AvStreamedStep` the quantized
  arm takes, built by one private `pipeline::av_step`. The two arms are an A/B
  over one caller and the fields they read are four pairs of same-typed slices
  (two latents, two timestep vectors, two position sets, two text
  projections); a struct per arm is a place where one arm can be handed the
  audio member where the video one belongs and still produce a plausible clip.
* **The banner is derived, not guessed.** `ltxv_cli` used to spell the arm out
  itself from `(dit_config, audio)`. It now calls `pipeline::av_arm_label`,
  the same pure function `build_denoiser` routes on, so the line a reader
  checks and the object the run denoises with cannot disagree.

#### 2 - what happened to the host-fp32 arm, and why it was kept

`av_stream::AvDenoiser` is NOT dead and is not deleted. It is a thin driver
over `LtxAvDit`, which is precisely the eager fp32 model the tiny-config
parity suite proves against
(`av_dit_parity.rs::ltxv_av_dit_tiny{,_gated}_matches_reference`) and that
`real_q8_0_av_streamed_forward_matches_the_eager_fp32_forward` compares the
quantized forward to; it is also the "before" column of every measurement in
phase 30 (`ltxv_bench av`). Deleting the driver would leave that model with
no way to be run at a real config from a generation, which is what a
measurement and a driver-level fallback both need.

It is now reachable only behind `BRAIN_LTXV_AV_FP32=1`
(`pipeline::av_fp32_reference`), the same shape as `BRAIN_NO_FLASH_CROSS` and
kept for the same three reasons that variable's own doc gives: an A/B needs a
switch or it compares the fast path against itself, a reference definition of
the math has to stay runnable, and a driver-level surprise needs a fallback.

The reader-cannot-tell problem is solved by making the arm a DERIVED label
rather than a documented convention: `pipeline::av_arm_label(real, audio,
fp32)` is a pure function, `build_denoiser` routes on the same predicate, and
`ltxv_cli`'s banner prints the label instead of spelling the arm out itself.
The previous banner was hand-written from `(dit_config, audio)` and had gone
stale the moment the routing changed underneath it - which is exactly the
failure this replaces.

#### 3 - the memory guard follows the switch

`AvWeights::fits_in_host_memory` refused a machine that could not hold the
fp32 expansion, and `check_audio_request` called it unconditionally - so an
`--audio` run on the quantized path was budgeted against a requirement it no
longer has, and the message named a gap ("the AV stream has no
streamed/quantized path yet") that had closed.

`pipeline::check_av_host_memory(cfg, fp32)` now picks the requirement of the
arm that will run. The quantized figure is DERIVED, never written down:
`block::cached_av_block_bytes` (the same function the residency policy sizes a
slot with) times the layer count, plus the manifest minus its
`transformer_blocks.` prefix as fp32. The fp32 arm's own guard is unchanged
except for its message, which now names the switch that asked for the
expansion so it can never read as "brain cannot generate audio on this
machine". `available_host_bytes` is shared rather than copied: two arms of one
feature must not disagree about how much memory the box has.

#### 4 - the window and two-stage boundaries

A clip holds ONE `RealAvDit` and hands it three kinds of boundary: step to
step (nothing moves), stage 1 to stage 2 (the video token count roughly
quadruples), and window k to k+1 (the token count may be unchanged while every
position moves, because a continuation window re-bases both streams onto its
own time origin).

The memory side was already in the loop and needed nothing new:
`generate_long` calls `Denoiser::release_devices` before each VAE decode and
between the two stages, and `RealAvDit`'s implementation drains its session
map exactly as `RealDit`'s does, so the next session's slot count is planned
from the shape it will actually run rather than inherited.

The CORRECTNESS side is not the same question and got its own gate:
`av_dit_parity.rs::a_reused_av_session_is_exact_across_a_window_or_stage_boundary`
runs three shapes through ONE narrow (1-slot, so the rotation is on the tested
path) session and compares each, BIT for bit, against a session built fresh
for that shape. Releasing is a memory decision; a clip's correctness must not
depend on when a card happened to be handed back.

The shape LADDER is the point, and one rung of it is not obvious: two of the
three shapes have the SAME token count and differ only in their positions,
which is what a window seam produces. Without that rung the gate cannot see a
`RopeCache` key that hashes geometry without positions - the exact "silently
reusing another shape's rotation produces plausible video" failure that
cache's own doc warns about.

#### 5 - the mutations, and which metric caught which

| mutation | modelled fault | RED in | caught by |
|---|---|---|---|
| `av_step` builds `v_context` from the AUDIO projection | one arm handed the other stream's text conditioning - four such pairs exist and every one is two same-typed slices | `an_av_step_carries_each_streams_own_inputs` | slice equality, in milliseconds and with no checkpoint |
| `rope_key` hashes geometry but not `positions` | a window seam re-bases both streams onto a new time origin at an unchanged token count, so the cache serves the PREVIOUS window's rotation | `a_reused_av_session_is_exact_across_a_window_or_stage_boundary`, shape 1 only | BIT equality against a fresh session (max_abs 7.5e-2). Shape 0 and shape 2 both PASS - only the same-token-count/different-positions rung sees it |

Not touched, and therefore not re-verified here: the cross-modal gate path.
This phase adds no numerics - it routes an existing forward - so
`a_closed_cross_modal_gate_makes_the_gated_stream_independent_of_the_other`,
`device_derived_av_modulation_is_bit_identical_to_the_host_uploaded_form` and
the real-weight block-0 comparison are unchanged and were re-run as-is.

#### 6 - the run: a real multi-window audio-visual clip, end to end

The verification phase 29 deferred, taken on the path phase 30 built. One
Tesla P40 (`--device gpu0`, both cards idle before the run, nothing sampled
during it except `/proc/<pid>/status` for RSS, which never touches the
device), real `ltx-2.5-22b-distilled-transformer-Q8_0.gguf` + real
`gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf` + real video/audio VAEs + the real
spatial x2 upscaler, no build running:

```
brain --device gpu0 ltxv t2v --dit-config ltx25_22b --audio \
  --prompt "a blacksmith hammers glowing steel on an anvil, rhythmic metallic
            clangs ringing out, sparks flying in a dark forge" \
  --frames 121 --width 1280 --height 704 --fps 24 --seed 7
```

121 frames is the shortest shape that crosses a seam at this resolution: a
single window caps at 113. **The run's own banner, which is the evidence that
the quantized arm is what executed:**

```
ltxv (REAL checkpoint audio+video DiT (int8 compute, both streams), real VAE,
real Gemma-4 text encoder): 121 frames at 1280x704, 8 distilled-schedule
(fixed) steps x 1 forward(s) of 12320 tokens, eta 1, guidance 1, seed 7
[two-stage: 3080 tokens at 640x352, then 3 refinement steps at 12320 tokens]
[2 windows, 57-frame rolling latent context], +126 audio tokens (5.04s of
16 kHz stereo, denoised jointly)
```

It used to read `(host fp32, both streams)`, and that line is how the gap this
phase closes was found in the first place.

**Wall clock and where it went** (spans between the run's own trace
timestamps, `--trace-ltxv 4`; they sum to the wall clock rather than to a
fraction of it):

| span | seconds |
|---|---:|
| open the GGUF + load the AV head tensors | 11.8 |
| text encode (served from the on-disk context cache) | 1.1 |
| window 0 stage 1, 3080 video + 109 audio tokens, 8 steps | 325.7 |
| upscaler load + stage-2 setup | 6.4 |
| window 0 stage 2, 12320 video + 109 audio tokens, 3 steps | 315.3 |
| window 0 VAE decode, 105 frames, 8 tiles | 164.3 |
| window 1 stage 1, 2200 video + 76 audio tokens, 8 steps | 196.8 |
| stage-2 setup | 5.6 |
| window 1 stage 2, 8800 video + 76 audio tokens, 3 steps | 212.5 |
| window 1 VAE decode (73 frames, 4 tiles) + audio VAE + vocoder | 95.1 |
| **end to end** | **1345.0** |

Host peak RSS over the whole run, `VmHWM` sampled from `/proc`: **56.0 GiB**.
Peak VRAM was NOT sampled continuously; two single `nvidia-smi` reads during
stage 2 showed 20153 MiB of the 24576 MiB card, which is an observation and
not a peak.

**Read `secs_per_step` in this pipeline's traces as a RUNNING MEAN, not a
per-step cost.** It is `elapsed / steps_done`, so a stage's last value times
its step count is its span, and quoting the first value as "the step time" is
wrong by the whole warm-up. The marginal times, first step (which pays the
GGUF read, the int8 quantize and the resident prefill) separated from the
warm ones:

| stage | first step | warm marginal, mean of the rest |
|---|---:|---:|
| w0 stage 1, 3080 + 109 tokens | 119.0 s | **29.5 s** (7 steps) |
| w0 stage 2, 12320 + 109 tokens | 107.3 s | **104.0 s** (2 steps) |
| w1 stage 1, 2200 + 76 tokens | 31.9 s | **23.6 s** (7 steps) |
| w1 stage 2, 8800 + 76 tokens | 73.7 s | **69.4 s** (2 steps) |

**Against the arm it replaced, at one identical shape.** The host-fp32 attempt
recorded in phase 29 ran the same prompt, seed, resolution and frame count on
the same card and got six steps into window 0 stage 1 before it was killed, so
one span is directly comparable and the rest is not measured:

| | host fp32 | quantized | |
|---|---:|---:|---:|
| transformer ready (fp32 expansion vs GGUF open + head tensors) | 142.2 s | 11.8 s | **12.1x** |
| stage 1 warm marginal step, 3080 video + 109 audio tokens | 161.9 s | 29.5 s | **5.49x** |
| host RSS | killed at ~90 GB, whole-model expansion logged at 78 GiB | 56.0 GiB peak | |

The 5.49x is a whole-pipeline figure at a real shape and it is larger than
phase 30's 4.31x forward-only ratio, because a real stage-1 forward is 3080
tokens rather than 13200: the fp32 arm's cost is dominated by re-uploading
every block's fp32 weights, which does not shrink with the token count, while
the quantized arm's does.

**The residency policy, at each of the four shapes the clip presented it
with** - the session is released at every window and stage boundary and
re-planned from the shape it is about to run, which is exactly what
`RealAvDit::release_devices` is for:

| stage | video tokens | slots granted | reserve | prefill |
|---|---:|---:|---:|---:|
| w0 stage 1 | 3080 | 16 of 48 | 5509 MiB | 27.2 s (cold: GGUF read + quantize) |
| w0 stage 2 | 12320 | 16 of 48 | 17429 MiB | 3.0 s (warm host cache) |
| w1 stage 1 | 2200 | 16 of 48 | 4374 MiB | 4.4 s |
| w1 stage 2 | 8800 | 16 of 48 | 12888 MiB | 3.1 s |

An AV block is 371 MiB of int8, so 16 slots is the quarter-of-the-card cap
binding at every one of them, exactly as phase 30 predicted. The prefill cost
after the first is the re-upload only: the weights come back from the host
cache, never from the checkpoint.

**The waveform, on the model's OWN samples.** The clip was generated a SECOND
time with `ffmpeg` shimmed off `PATH`, so the CLI takes its
"never throw away a generation for want of an encoder" branch and writes the
sound as 16-bit PCM beside numbered frames. Both runs are the same seed,
prompt and shape and agree to five decimals on every seam figure, so the AAC
mux is not hiding anything; the numbers below are the lossless ones (the
second run's wall clock was 1330.9 s against the first's 1345.0 s, the
difference being the mux the second one does not do).

The harness is the one phase 29 left ready, re-validated before it was
trusted rather than taken on faith: a synthetic clip at this clip's own token
layout with a sign flip planted at the exact seam sample reads **6.18x** the
99.9th-percentile step on the clicked channel against **0.95x** on the clean
one, and the reported "exact boundary step" equals the maximum in the window,
i.e. the index lands where the plant is. The seam's sample index is derived,
not searched: audio token `i` covers mel frames `[4i-3, 4i+1)` and each is
`HOP_LENGTH = 160` samples, so the first sample window `k+1` contributed is
`(4*T - 3) * HOP_LENGTH` for the cumulative token count `T` before it. Here
window 0 contributed 109 tokens and window 1 the remaining 17, so the seam is
sample **69280** of 80667.

**Seam continuity - the seam is SMOOTHER than ordinary audio, by a factor of
about 23:**

| | 99.9th pct step elsewhere | max step within +/-16 of the seam | ratio |
|---|---:|---:|---:|
| left | 0.658533 | 0.028839 | **0.044x** |
| right | 0.692439 | 0.028839 | **0.042x** |

That is the direction the comparable stage of another model in this tree
measured, and the opposite of a click. It is also OFFSET-ROBUST, which
matters because a codec frame or an off-by-a-few in the derivation would move
the index: widening the window to +/-160, +/-512 and +/-1024 samples the
maxima are 0.082x, 0.198x and 0.410x of the ambient 99.9th percentile
respectively. Nothing anywhere near the seam reaches the ambient distribution,
let alone exceeds it.

**Plain audio statistics** (16 kHz stereo, 80667 samples per channel):

| | left | right |
|---|---:|---:|
| peak | 0.9765 | 1.0000 |
| clipped samples (`abs >= 0.999`) | 0 | 1 |
| RMS | -21.96 dBFS | -21.86 dBFS |
| longest run below -60 dBFS | 0.7 ms | 0.4 ms |

One sample of 80667 on the right channel saturates the f32 to i16 conversion.
It is reported rather than rounded away; a single-sample full-scale excursion
is inaudible and is not evidence of a broken vocoder, but a clip that starts
producing them in quantity would be.

**The stereo field is real and narrow**: L/R correlation **0.9840** - neither
1.0 (mono) nor 0.0 (unrelated) - with `max|L-R| = 0.2031`,
`rms(L-R) = -36.83 dBFS`, and only 307 of 80667 sample pairs identical. Two
channels that were the same buffer would have correlation exactly 1.0 and
80667 identical pairs.

**The two streams cover the same time window, exactly.** The audio latent's
length is `round(frames / fps * LATENT_RATE) = round(121/24*25) = 126` tokens,
which is what the two windows contribute (109 + 17), so nothing drifts at the
seam. The decode is `(4*126 - 3) * 160 = 80160` samples = 5.0100 s, three mel
frames short of the clip by construction (the causal audio VAE's first latent
frame covers one mel frame, not four); the last sample is held over that gap
to 80667 samples = **5.041667 s**, which is exactly `121 / 24`. The run's own
line says so (`decoded_seconds=5.010 video_seconds=5.0417`), and the muxed
container agrees: video `duration=5.041667`, audio `duration=5.042000`, a
0.33 ms difference that is one AAC frame's granularity and not a drift.

#### 7 - recorded, NOT done

* **The audio-visual path still has no training support and no bandwidth
  extension.** Unchanged by this phase, which routes an existing forward and
  adds no numerics.
* **The AV forward still has no CFG-parallel two-card path.** `RealAvDit`
  deliberately does not override `Denoiser::forward_cfg_pair`: a joint forward
  does not go through that seam (both streams' inputs and both outputs would
  have to cross it), and the real distilled checkpoint denoises at guidance 1,
  where no pair is issued at all. The host weight cache is now shared and
  `Sync` the way `RealDit`'s is, so the obstacle is the trait method's shape
  and not the type - but it IS a trait change, not a tail-end edit.
* **`activation_reserve_bytes` still models the VIDEO-only working set.**
  Phase 30 measured what that costs on the AV path and the arithmetic is
  unchanged here; it fits for a reason that is a coincidence of this card and
  this block-size ratio rather than a property of the policy.
* **`--audio` still refuses a multi-scene clip**, for the design reason phase
  29 recorded, not an arithmetic one.

---

### Phase 32 - the DP4A audit: where the int8 path really issues `dot4I8Packed`, and the shared-load ceiling that capped it

**The question this phase was asked**: does brain's int8 tier genuinely execute
as W8A8 DP4A, or does it store weights as int8 and dequantize them into slow
Pascal fp32 GEMMs? The answer for THIS model is the first one, everywhere it
can - and the audit is recorded here because the profile it produced is what
identified the real ceiling, which was not where the question assumed.

#### 0 - DP4A is genuinely emitted, on BOTH GPU backends, and that is measured

`naga` lowers WGSL `dot4I8Packed` to SPIR-V `OpSDot` with
`PackedVectorFormat4x8Bit` only when the writer is given the
`DotProduct`/`DotProductInput4x8BitPacked` capabilities; otherwise it emits a
four-way `BitFieldSExtract` polyfill and the kernel silently runs at a small
fraction of the rate its name implies. Both of this workspace's GPU backends
supply the capability: `wgpu-hal`'s Vulkan adapter pushes the four
`DotProduct*KHR` capabilities whenever its `shader_integer_dot_product` private
cap is set, and `crates/vulkan`'s own context queries
`shaderIntegerDotProduct` + `integerDotProduct4x8BitPackedSignedAccelerated`,
enables the feature, and reports it as `DeviceCaps::numeric.int8_dot`.

The decisive evidence is not the source reading, it is the roofline cache on
this box, which holds a record per (device, backend) PAIR:

| backend | fp32 GFLOP/s | int8 GOP/s | ratio |
|---|---:|---:|---:|
| `backend-wgpu` | 10517.5 | 43559.7 | 4.14x |
| `backend-vulkan` | 10542.0 | 43705.6 | 4.15x |

A polyfilled `dot4I8Packed` cannot produce a 4.14x ratio over fp32 FMA - it is
roughly a dozen integer instructions per four MACs. So the instruction is real
on both backends. Note what this does NOT say: the recorded gap "the LTX int8
tier does not run on `backend-vulkan` at all" is a `wait_for_fences` device-loss
defect in that backend, NOT a missing DP4A path. DP4A works there; the model
path hangs for an unrelated reason.

#### 1 - the coverage map, by OPERATION rather than by kernel name

At `ltx25_22b`, the ten quantizable linears per block (`attn1`/`attn2`'s
`to_q`/`to_k`/`to_v`/`to_out.0` plus `ff.net.0.proj`/`ff.net.2`) all dispatch
`matmul_i8_dyn` unconditionally - there is no shape below which they fall back
to `matmul_reg3` or to the naive `matmul`, and `qlinear` has no fp32 arm at
all. Activations are quantized per token every forward
(`max_abs_row` -> `quant_pack`), and `max_abs_row` is transparently upgraded
to the coalesced `max_abs_rows` by `gpu_core::upgrade`, so the prep is not
paying the one-thread-per-row penalty either.

What stays fp32, and why each one is right or wrong:

| operation | kernel | DP4A | verdict |
|---|---|---|---|
| the ten block linears | `matmul_i8_dyn` | yes | correct |
| self-attention QK/AV | `flash_attn_bidir_reg2` | no | the real gap - see 4 |
| text cross-attention QK/AV | `flash_attn_cross_reg2` | no | same |
| `to_gate_logits` + the per-head gate expand | `matmul_reg3` | no | never-quantized by design; measured at 61.2 ms of a 9615.9 ms forward |
| the embeddings connector's own linears | `matmul_reg3` | no | once per generation, not per step |
| norms / RoPE / adaLN / GELU / gating | fp32 | no | all memory-bound; DP4A is not the lever |

So the model is NOT "int8 storage with fp32 arithmetic". Every GEMM that can
be packed is packed and is computed packed.

#### 2 - the arithmetic split, counted from the architecture

Per video block at token count `T`, context `C`, `dim = 4096`:
int8 MACs are `14*T*dim^2 + 2*C*dim^2`; fp32 attention MACs are
`2*T^2*dim + 2*T*C*dim`; the fp32 gate pair is `2*T*dim*heads`. At
`T = 13200`, `C = 1024`, 48 layers:

| tier | ops per forward | share |
|---|---:|---:|
| int8 (DP4A) GEMM | 3.010e14 | **67.0%** |
| fp32 self-attention | 1.370e14 | 30.5% |
| fp32 cross-attention | 1.063e13 | 2.4% |
| fp32 gating | 6.64e11 | 0.1% |

Over a whole 10 s 1280x704 24 fps generation (8 stage-1 steps at 6820 tokens +
3 stage-2 steps at 27280 tokens, video stream only, no CFG pair) the same
count gives **3.11 POP of int8 GEMM and 2.17 POP of fp32 attention, 5.28 POP
total** - the DP4A share falls to **59%** because self-attention is O(T^2)
and stage 2 is four times as wide. An independently-derived estimate of
5.65 POP split 3.55/2.10 including the audio stream agrees with this to
within 7% on the total and ~12% on each half, which is the audio stream's
own contribution.

The ceiling, if every eligible op used DP4A, is ~99.9% - attention is the
entire remainder.

#### 3 - `matmul_i8_dyn` was at ~36% of the DP4A roof, and the cause was the shared-load instruction ratio

**Measured first, hypotheses second** (`qwen_bench gemm8`, best-of-7, both
cards idle, `max|Δ| == 0` against a host i32 reference at every shape):

    n sweep  (m=4096, k=4096)   128 -> 6030 GOP/s, 512 -> 9662, 2048 -> 12453, 4096 -> 12847
    k sweep  (m=4096, n=2048)  1024 -> 11124, 4096 -> 14160, 8192 -> 14151, 16384 -> 13581

Both curves rise to a plateau and stay there. That kills two hypotheses at
once: a DRAM-bound tile gets WORSE as `n` grows (the A tile is re-read `n/BN`
times, and the naive traffic count at `4096^3` is ~1.14 GB in 10.7 ms = 107
GB/s against a 287 GB/s roof, so it was never close), and a prologue/epilogue
overhead would shrink as `k` grows. Flat in both is an on-chip issue ceiling.

The ceiling is the shared-memory load INSTRUCTION rate, and the reason is
specific to DP4A. The kernel was a faithful port of `matmul_reg3`'s layout:
k-major shared tiles, one scalar `u32` shared load per operand per k-step, an
8x8 register block - 64 MACs per 16 shared loads, i.e. four MACs per load
instruction. That is the right ratio for FFMA. But `dot4I8Packed` retires four
times the MACs per instruction from the same shared word, and a Pascal SM
issues four times as many math lanes per clock as load-store lanes, so at that
mix the math and load-store ceilings coincide exactly and every inefficiency
lands on the memory side.

**The fix**: shared tiles become k-group-MINOR and are read as `vec4<u32>` -
four k-groups per load instruction, sixteen DP4A per load instead of four.
Same 128x128 tile, same 8x8 register block, same three barriers, same
`Params`, same bindings, same dispatch geometry, so **no call site in any model
changes**. The stride is padded in vec4 units (an unpadded one puts a
tx-group's sixteen lanes on a quarter of the banks); only the A operands are
hoisted into registers across the inner unroll, because hoisting both sides
pushes the register block past two workgroups per SM.

Bit-identical, not merely close: the accumulation is INTEGER, so re-associating
the k axis is exact.

**A/B on ONE idle card (gpu1), same binary shape, best-of-7:**

| m, k, n | before GOP/s (% roof) | after GOP/s (% roof) | speedup |
|---|---:|---:|---:|
| 4096, 4096, 1024 | 10769 (24.8%) | 14558 (33.5%) | 1.35x |
| 4096, 4096, 2048 | 12362 (28.5%) | 15435 (35.5%) | 1.25x |
| 4096, 4096, 4096 | 12819 (29.5%) | 16052 (37.0%) | 1.25x |
| 4096, 16384, 2048 | 14158 (32.6%) | 18468 (42.5%) | 1.30x |
| 4096, 16384, 4096 | 14453 (33.3%) | 17809 (41.0%) | 1.23x |
| 1024, 4096, 16384 | 12770 (29.4%) | 16043 (36.9%) | 1.26x |

The baseline was also measured on gpu0 while idle and agreed with the gpu1
baseline to within 0.2% at `4096^3` (12847 vs 12819), so the card is not the
variable.

#### 4 - the whole-pass number, and the control rows that make it believable

`BRAIN_PROFILE=1 BRAIN_LTXV_DIT=<real 22B Q8_0> BRAIN_DEVICE=gpu1
./target/release/ltxv_bench streamed 8 13200 1024 1 1 1`, cache-hit arm
(call 2 cumulative table minus call 1's), both cards idle at start:

| kernel | before (Phase 28) | after | ratio |
|---|---:|---:|---:|
| `flash_attn_bidir_reg2` | 5754.0 ms | 5722.3 ms | 1.006 (**control**) |
| `flash_attn_cross_reg2` | 486.7 ms | 486.0 ms | 1.001 (**control**) |
| `rmsnorm_rows` | 108.6 ms | 108.8 ms | 0.998 (**control**) |
| `matmul_i8_dyn` | 3225.0 ms | **2562.7 ms** | **1.258x** |
| GPU kernel time, 8 layers | 10308.8 ms | **9615.9 ms** | **1.072x** |

Three unchanged rows agreeing to within 0.6% across two sessions is what makes
the fourth row a measurement of the change rather than of the machine.

Graded against roof at real width, per 8-layer cache-hit forward:
`matmul_i8_dyn` now does 5.016e13 int8 OP in 2.5627 s = 19573 GOP/s,
**44.9% of the 43560 GOP/s DP4A roof**, up from 35.7%. `flash_attn_bidir_reg2`
does 2.2838e13 FLOP in 5.7223 s = 3991 GFLOP/s, 37.9% of the 10517 GFLOP/s
fp32 roof (unchanged, as it must be).

**Read the last row honestly**: a 1.26x on the int8 GEMM buys 1.07x on the
device pass, because self-attention is 59.5% of it. That is the finding, not a
disappointment - the int8 half of this model is now closer to its roof than the
fp32 half is to its own.

#### 5 - the gates, and which mutation caught which

`qwen_bench gemm8` asserts `max|Δ| == 0` against a host i32 reference, so the
gate is on the BITS, with no tolerance to tune. Ragged shapes were added to the
sweep specifically because the vec4 staging has a partial-quad path that full
shapes never reach: `(m,k,n)` of `(1,4,1)`, `(3,12,5)`, `(100,12,37)`,
`(130,20,200)`, `(257,260,129)`, `(128,32,128)`, `(129,36,127)`,
`(7,1024,3072)`, `(1,4096,4096)` - all exact.

**Mutation**: drop the third lane of the ragged-tail staging branch.

| shape | what it exercises | mutated result |
|---|---|---|
| `k=12` (kg=3, tail quad has 3 valid lanes) | the dropped lane | **RED**, `max|Δ|` 3.007e2, `max_rel` 1.326e2 |
| `k=36` (kg=9, tail quad has 1 valid lane) | a tail that never reaches lane 2 | green - correctly blind |
| `k=4096` (kg=1024, no tail at all) | the main path | green - correctly blind |

So the ragged shapes in the list are load-bearing and the attribution is
exact; a suite of only full-quad shapes would have shipped the bug.

Cross-model parity: every crate that registers `matmul_i8_dyn` was re-run -
`ltxv`, `wan`, `flux1`, `flux2`, `s3dit`, `qwen3`, `qwen35`, `qwen35moe`,
`qwen3omnimoe`, plus `gpu-core`/`model`/`kernels`/`backend-api` - all green.

#### 6 - recorded, NOT done

* **int8 attention is the whole remaining prize and was NOT attempted.**
  Self-attention is 59.5% of the device pass at real width and 30.5% of the
  arithmetic. If QK and AV both ran DP4A at the fraction of roof the GEMM now
  reaches, attention would drop roughly four-fold and the device pass would
  fall by about 1.7x. The obstacle is quality, not expressibility: this tier's
  real-weight block-0 parity already sits at cosine 0.9963 (video) / 0.9986
  (AV) with attention in fp32, and per-token int8 on Q/K without per-channel
  K smoothing is the documented way that collapses. The usual mitigation -
  int8 QK with an fp16 PV accumulate - degenerates on Pascal, where f16 is a
  1/64-rate path, into int8 QK with an fp32 PV, which halves the available
  win. The honest first step is therefore int8 QK ONLY, scores dequantized for
  the online softmax, PV left fp32 - worth about 1.26x on the device pass by
  the same profile - gated on cosine AND rel_l2 against the fp32-attention arm
  at real width before anything ships. No number here is a measurement of an
  int8-attention implementation, because none exists.

* **`matmul_i8_gemv` still carries BOTH defects its fp32 sibling had fixed.**
  It declares `array<i32, 2048>` sized for its `m <= 32` worst case (8 KB of
  workgroup memory at every `m`) and accumulates with a read-modify-WRITE into
  that array inside the k loop - checklist C5 and C6 exactly, the two the fp32
  `matmul_gemv` -> `matmul_gemv_reg` upgrade exists to fix, and no int8 twin of
  that upgrade exists. It is live on the int8 serving decode path
  (`qwen3::serve`), `flux1`'s m=1 modulation GEMVs and `qwen35moe`. The recipe
  is already written: a GPU-only `_reg` sibling with an `MREG` template knob
  plus one `gpu_core::upgrade` row, and for int8 the rewrite is bit-identical
  for the same reason this phase's is. Not attempted here.

  Related catalogue integrity note: `scripts/build/kernelmeta.py::opt` returns
  5 for ANY kernel containing `dp4a`, so `matmul_i8_gemv`'s `@opt 5` is derived
  from the instruction, not from a register block. The one row that should have
  flagged this kernel rates it top tier for the wrong reason.

* **`moe_linear_gated_i8` issues DP4A on the NAIVE tier** (`@opt 2`, one thread
  per output element, serial inner reduction) and says why in its own header:
  a per-row early exit for an unrouted row cannot coexist with a workgroup
  barrier, so tiling it needs row compaction. This is the largest remaining
  RATE gap in the workspace's DP4A story after attention - it is what
  `moondream3`'s 1280 expert tensors and `qwen3omnimoe`'s Thinker experts run
  on, and for the Thinker the experts are ~60% of the per-token GEMM
  arithmetic. Not touched by this phase.

* **`qwen3omnimoe`'s int8 Thinker stores attention/router/`lm_head` weights as
  int8 and dequantizes them to fp32 on upload** (`model::int8::
  upload_dequantized`), because those linears have no int8 dispatch path -
  only the MoE experts do. At the 48-layer, hidden-2048, GQA-32/4 Thinker
  config that is ~40% of the per-token GEMM arithmetic left in fp32, and about
  3.6 GB of avoidable device bytes across the two-card split. This is the one
  place in the workspace that genuinely matches "stores weights as int8 and
  dequantizes them into fp32 GEMMs".

* **`minimaxmusic3`'s DiT int8 tier (`dit_int8`) is storage-only** and
  dequantizes to f32 before `dit::from_tensors` - the same shape, in a shipped
  model. Its Global LLM half does take the real packed path.

* **`ltxv::attention_q` quantizes the SAME activation three times** in
  self-attention (`to_q`, `to_k`, `to_v` all take `norm_x`) and twice in cross
  attention (`to_k`/`to_v` both take `enc_hidden`). `qwen3::q8::Q8::quant`
  already documents the right shape ("call once per distinct input"). Measured
  before deciding: `max_abs_rows` + `quant_pack` together are 174.4 ms of a
  9615.9 ms forward, so the redundant share is about 0.3% and hoisting it is
  not worth a change to a crate mid-workstream. Recorded so it is not
  re-derived.

* **The int8 paged-KV decode kernels dequantize keys to f32 on read and are
  RIGHT to.** `paged_decode_scores_i8_batched` has an arithmetic intensity of
  about 2 ops/byte against an int8 machine balance of ~151 ops/byte
  (43560 GOP/s over 287 GB/s): it is DRAM-bound by two orders of magnitude, so
  DP4A cannot help. int8 there is a pool-size tier, and calling it a missing
  DP4A path would be a misreading.

* **The recorded "LTX int8 tier does not run on `backend-vulkan` at all" no
  longer reproduces at the scope it names.** `BRAIN_DEVICE=vulkan
  ./target/release/deps/int8_compute-* --test-threads=1 --nocapture` prints
  `adapter: Tesla P40 (Vulkan compute, ash + naga WGSL->SPIR-V)` - so it really
  is the native backend, not a silent wgpu fallback - and both tests pass,
  including `real_q8_0_block0_int8_compute_matches_fp32`, the REAL-weight
  block-0 int8-vs-fp32 comparison. The gap's own text lists exactly that test
  binary as failing. What is NOT retested here is the large end of that claim
  (48 layers at T=3520, a whole streamed forward), so the bullet should be
  narrowed rather than deleted. It was never a DP4A-availability claim in any
  case: `backend-vulkan` measures 43705.6 GOP/s of `dot4I8Packed`, marginally
  ahead of wgpu's 43559.7.

### Phase 33 - the host stops spending a third of every forward in the Vulkan allocator

Phase 30 ended with the honest split: a warm video forward was 66% device
time and 34% host, and named `Gpu::storage` churn as the next item at ~14.3 s
of 93.6 s. The full attribution says the churn is roughly TWICE that, because
only its allocation half had ever been measured - buffers were also being
DESTROYED per block, outside every timing span the code had.

#### 0 - where the host time actually goes, with nothing left over

Attribution of one warm video forward at production width (48 layers,
T = 13200, ctx 1024, int8, device-resident session, one distinct timestep;
`ltxv_bench streamed 48 13200 1024 1 1 1 3`, best warm call). Wall 88.3 s,
device 58.1 s (kernel timestamp queries), host 30.2 s. Every row is a span
that does not overlap any other row, so they sum to the forward:

| host item | s | share of host |
|---|---:|---:|
| device buffer **allocation** - the block's ~74 temporaries, 48 blocks | 12.04 | 39.8% |
| device buffer **destruction** - the same set, dropped at each block's end | 10.53 | 34.8% |
| `submit` + poll host residual (block submit+wait minus device kernel time) | 1.84 | 6.1% |
| block weight upload (the 25 blocks residency could not keep) | 1.67 | 5.5% |
| graph recording (`Gpu::step` bind groups + the per-forward writes) | 1.65 | 5.4% |
| output stage - host LayerNorm + modulate + `proj_out` over `[t, dim]` | 0.88 | 2.9% |
| block loop residual (window `acquire`, the `after_block` callback) | 0.71 | 2.4% |
| adaLN-single table (host) | 0.40 | 1.3% |
| device -> host readback of the final activation (206 MiB) | 0.34 | 1.1% |
| patchify (host linear) | 0.13 | 0.4% |
| per-forward upload of `x` / the adaLN table / the context | 0.08 | 0.3% |
| embeddings connector (cache hit) | 0.01 | 0.0% |
| **unaccounted** | **0.00** | **0.0%** |

Two of those rows are measurements the code could not previously make and had
to be added as temporary probes (since removed): the block loop was timed
between `record_upload`, `compute` and `readback`, and the DESTRUCTION of a
block's scratch happened after all three of those spans closed, so it was
invisible. `output_stage` and `DitSession::prefill` were likewise outside
every span; `output_stage` is now permanently instrumented, `prefill`
measured zero on a warm resident call and did not earn a line.

The allocation/recording split is derived rather than directly measured: the
`activation/context/adaLN record+upload` stage is 13.72 s before and 1.68 s
after, and the arena is the only difference, so 12.04 s of that stage was
allocation and 1.68 s is what recording and the writes really cost. A
process-wide `Gpu::storage` counter agreed independently (12.52 s over 4544
calls, of which 3552 are block scratch).

**`perf` says the same thing, from outside the process.**
`kernel.perf_event_paranoid` was lowered to 1 for this pass;
`perf record -F 199 --call-graph dwarf`, `--delay` set past the cold call so
only warm forwards are sampled, on a `RUSTFLAGS="-C debuginfo=1"` build (not
committed - the shipped profile is unchanged):

Sampled at 4 layers, not 48 - the same real token width, a shorter run, and
the shares are what is being compared rather than the seconds:

| symbol, cumulative | arena off | arena on |
|---|---:|---:|
| `gpu_allocator::vulkan::MemoryBlock::new` | 5.60% | below the 2% cut |
| `gpu_allocator::vulkan::Allocator::free` | 4.66% | below the 2% cut |
| `__GI___ioctl` | 10.21% | 4.17% |
| `__GI_munmap` | 3.14% | below the 2% cut |
| `ltxv::dit::output_stage` | 6.46% | 6.72% |
| the frame the GPU wait unwinds into | 77.43% | 83.00% |

The last row is not host stall and must not be read as one: it is the process
blocked in `device.poll(Wait)` while the card computes, which is why its share
RISES when the host does less work. The wall-minus-device split already counts
it as device time.

**The per-layer cost is linear, checked rather than assumed.** Same width,
same everything, 4 vs 8 layers on the un-pooled arm: host 3.70 s -> 5.84 s,
i.e. 0.535 s per layer plus 1.56 s fixed, which extrapolates to 27.3 s at 48
against 29.7 s measured. The 2.4 s shortfall is exactly the row that does not
exist at 4 or 8 layers - at those depths every block is resident, so nothing
is re-uploaded. Per-BLOCK host cost scales; the fixed part does not.

#### 1 - the fix: a replay arena on the device handle, not a pool in this crate

`gpu_core::scratch::Arena`, entered by `Gpu::scratch_scope`. Every block
dispatches the identical shape sequence, so the arena simply remembers, in
call order, what a scope asked for and hands the same buffers back next time.
Nothing is created and nothing is destroyed once the sequence has run once.
It lives on the shared device facade rather than in `crates/ltxv`, so a model
opts in by wrapping its per-iteration body in one line; `crates/ltxv/src/
block.rs` has exactly two such lines, one per stream pair
(`LtxBlockQ::forward_prod_dev`, `LtxAvBlockQ::forward_prod_dev`).

**Why it cannot alias a live operand, which is the whole design question.**
Three parts, and only the middle one is the arena's own:

* **Inside a scope, nothing is issued twice.** The cursor only advances, so
  two operands of one dispatch cannot collide however the caller behaves.
* **Across scopes, a handle the caller KEPT blocks reuse.** `DeviceBuffer` is
  an `Arc`, so `DeviceBuffer::is_unique` - the arena's copy being the last one
  - answers "does any caller still name this allocation". A slot that fails
  the test is not reused; the arena allocates a fresh buffer and takes the
  slot over. That is what makes the one value a block stack deliberately
  carries forward, the CHAINED ACTIVATION, correct with no special case: it is
  still held by the caller, so its slot is re-allocated and the previous
  block's output is never written over. The cost is ~1 allocation per block
  instead of ~74, and it is self-correcting rather than a hand-maintained
  exemption list.
* **The device being finished is the CALLER's half**, and it is why this is an
  opt-in scope and not a change to `Gpu::storage`'s meaning. The refcount test
  above cannot see submitted work: a recorded `Step` does not hold a
  `DeviceBuffer` clone (the wgpu backend's step is a `BindGroup`, which keeps
  the native buffer alive by a different path), so a scope must not be
  re-entered until the previous one has been DRAINED. `forward_prod_dev`
  already ends in a blocking one-word read of its own output, which is
  `flush` + `map_async` + a bounded `poll_wait`. That distinction was checked
  in `backend-wgpu`, not assumed - the first version of this module's doc
  claimed the refcount covered dispatches too, which would have made the
  argument circular.

A slot whose requested size differs is likewise re-allocated, so a caller
whose sequence is not in fact identical degrades to plain allocation rather
than binding a buffer too small for its dispatch.

The arena also removes an accounting bug it did not set out to fix: `Gpu`'s
`memauth` grants are pushed per allocation and released only when the HANDLE
drops, so 4544 allocations a forward pushed 4544 grant records. Only a fresh
allocation is charged now.

#### 2 - measured, both arms, same binary, same box, one idle P40

`BRAIN_LTXV_NO_SCRATCH_POOL=1` selects the un-pooled arm. Four calls per run,
the first warm call discarded as warm-up, best of the remaining two. Both
cards idle before each run, nothing sampled during one, no build running.

**Video only** (`ltxv_bench streamed 48 13200 1024 1 1 1 3`):

| | before | after | |
|---|---:|---:|---|
| warm forward, best of 2 | 87.72 s | **64.56 s** | **1.36x** |
| of which DEVICE (kernel timestamps) | 57.99 s | 58.17 s | **+0.3%** |
| of which HOST | 29.72 s | 6.39 s | -78.5% |
| device share of wall | 66.1% | 90.1% | |
| first (cold) forward | 160.24 s | 140.06 s | 1.14x |
| host peak RSS | 36789 MiB | 36756 MiB | |
| resident blocks the policy granted | 23 of 48 | 23 of 48 | |

**Audio + video** (`ltxv_bench streamed-av 48 13200 1024 118 1 1 1 3`):

| | before | after | |
|---|---:|---:|---|
| warm forward, best of 2 | 101.60 s | **72.98 s** | **1.39x** |
| of which DEVICE | 63.63 s | 63.87 s | **+0.4%** |
| of which HOST | 37.97 s | 9.12 s | -76.0% |
| device share of wall | 62.6% | 87.5% | |
| first (cold) forward | 197.94 s | 166.73 s | 1.19x |
| host peak RSS | 52531 MiB | 52597 MiB | |
| resident blocks the policy granted | 16 of 48 | 16 of 48 | |
| peak VRAM (separate UNTIMED observation, `nvidia-smi -l 1`) | 21082 (phase 30) | 21079 MiB | |

**The before arm is measured, not cited, and it does not reproduce phase 30's
absolute numbers.** Phase 30 recorded 93.59 s wall / 61.88 s device / 31.70 s
host for the same video command; the same code measures 87.72 / 57.99 / 29.72
here. Both halves are ~6% faster than they were, so this is the box or the
toolchain and not a change in this workstream - and the number this pass is
about, the host SHARE, reproduces to the decimal: 33.9% then, 33.9% now. That
is why the before column above is a run and not a quotation.

Device kernel time not moving is the control, and it did not move on either
stream. Peak VRAM did not move either, which is the answer to the obvious
worry about holding a whole block's scratch across the stack: the arena holds
exactly ONE such set, and one set was already live at once during recording -
the allocator was destroying and recreating the same footprint 48 times per
forward rather than keeping it.

The residency grant is unchanged on both streams (23/48 and 16/48), so
`devres::activation_reserve_bytes` still fits and nothing about the phase-30
residency arm needs re-deriving.

#### 3 - the remaining host half, ranked

After the arena, the video forward's 6.4 s of host time at 48 layers splits
(stage timers, warm call):

| | s |
|---|---:|
| block weight upload - the 25 blocks residency could not keep, ~6.4 GB | 2.42 |
| graph recording (`Gpu::step` bind groups) | 1.62 |
| output stage (host LayerNorm + modulate + `proj_out`) | 1.00 |
| device -> host readback | 0.35 |
| adaLN-single table (host) | 0.15 |
| patchify (host) | 0.11 |
| everything else, incl. the per-forward arena refill | ~0.7 |

None of these is worth 4% of the forward on its own. The item that IS worth
more than all of them together is listed in section 5.

#### 4 - the gates, and which mutation caught which

`crates/gpu-core/tests/scratch_arena.rs` gates the arena's five structural
properties on `DeviceBuffer::alloc_id`; `crates/ltxv/tests/scratch_pool.rs`
gates the forward's OUTPUT BITS in both arms, at the tiny config and - when
the checkpoint is present - at the real 22B block's own allocation sequence.
Bits, not cosine: the arena changes when device memory is allocated and
nothing else, so `assert_eq!` on `to_bits` is the statement the code actually
makes, and a tolerance would be a weaker one.

Every guard was mutated separately, because they mask each other:

| mutation | what went red | what stayed green, and why that is the finding |
|---|---|---|
| drop the `is_unique` check | `a_buffer_still_held_by_the_caller_is_never_recycled` | BOTH ltxv bit gates. At ltxv's current dispatch order the only buffer that escapes a scope is the chained activation, and it is dead by the time the next block's last dispatch rewrites its slot - so the guard is defending a property this model does not currently depend on. Worth knowing rather than claiming a catch |
| never advance the cursor | `a_released_buffer_comes_back_in_the_next_scope`, `a_buffer_still_held_...` | the ltxv bit gates: the mutation degrades the arena to plain allocation, which is a performance regression and not a correctness one |
| hand back the PREVIOUS slot, guard intact | the same two | the ltxv bit gates - the uniqueness guard absorbs it, which is direct evidence the guard is load-bearing |
| hand back the previous slot AND drop the guard | **both ltxv bit gates**, 2048 of 2048 output words differing | nothing. This is the mutation that proves the bit gate can see aliasing |
| drop the size check | `a_changed_size_is_re_allocated_not_reused`, but ONLY after that test was rewritten | it first passed with the size check deleted, because it kept the small buffer alive and the UNIQUENESS guard refused the slot before the size check was ever consulted. The test now releases the buffer and asserts on the arena's held words |

`cargo test -p brain-ltxv` is green (284 passed, 0 failed), including the
real-weight gates that run through the new path:
`dit_parity::real_weight::ltxv_real_dit_tiny_layers_matches_reference`,
`int8_compute::real_q8_0_block0_int8_compute_matches_fp32` and
`streamed_vs_eager_real`. `device_residency::real_weight::a_resident_real_
checkpoint_forward_is_bit_identical_to_the_streaming_one` is `#[ignore]`d in
the fast lane and was run explicitly with `BRAIN_LTXV_DIT` set: green.

#### 5 - recorded, NOT done

* **The forward is now 90% device-bound, and the way past that is to stop
  draining after every block.** The per-block blocking read exists so wgpu's
  allocator pool shrinks (phase 30) - a reason the arena has just removed,
  since nothing is being allocated or freed per block any more. Without it the
  host's remaining 6.4 s would overlap the device's 58 s instead of adding to
  it, which is worth about another 1.1x. It is NOT a small change: the arena's
  second condition is that a scope may not be re-entered before the previous
  one has drained, so removing the drain needs alternating arenas (drain block
  `l-1` before entering block `l+1`'s scope) and its own gate. Deliberately
  left with the arithmetic rather than attempted at the end of a pass.
* **The arena is rebuilt once per forward.** `DitSession::device_for_call`
  hands out a fresh `Gpu::share` per call, and the arena lives on the handle,
  so the first block of every forward pays ~74 allocations (~0.7 s of the
  remaining 6.4 s). Keeping one long-lived scratch handle on the session would
  recover it, but the session is what two concurrent CFG branches share, and
  one arena behind two concurrent forwards is exactly the aliasing this design
  refuses. It needs a per-branch handle, not a cached share.
* **The output stage is host math at production width** - a LayerNorm, a
  per-token modulate and a `[t, dim] x [dim, out]` linear over 13200 tokens,
  1.00 s per forward on the host with the card idle. Every kernel it needs
  exists; it is the one remaining host stage in this forward that has no
  reason to be one.
* **The 25 non-resident blocks are now the largest single host row** (2.42 s,
  ~6.4 GB per forward). That is a residency-budget question, not an allocator
  one: peak VRAM was 18.1 GiB (video) of 24 GiB at phase 30 and the arena did
  not move it on the arm that WAS re-observed here (audio+video, 21082 ->
  21079 MiB), so the reserve may have room the phase-30 re-fit left on the
  table. The video arm's peak was not re-observed and is still phase 30's
  number.
* **Only `crates/ltxv` opts into the arena.** Every other block-stack model in
  the workspace has the same shape - an identical dispatch sequence per block,
  a chained activation, a drain - and none of them has been measured for it.

### Phase 34 - the VAE decode stops paying for two permutes per norm, and for eight weight uploads per clip

Phase 33 attributed the DiT forward and left the VAE alone. It is the second
largest item in a real run and the largest that had never been optimised: on a
10 s clip (241 frames, 1280x704, audio) the measured stage split is
`build 10.3 | text encode 171.3 | denoise 2328.6 | VAE 645.7 | audio 5.9`,
i.e. the decode is 20.4% of the run.

#### 0 - the decode-only harness, and where the time actually goes

`ltxv_bench decode <latent.bin> tiled` at the real clip geometry - a
`[128, 31, 22, 40]` latent, which is 241 frames at 1280x704 - on one idle
Tesla P40, `BRAIN_PROFILE=1`. The latent is synthetic (a decode's cost is a
function of its shapes, not of the values), real VAE weights.

That shape takes the TILED path: 16 tiles in **8 distinct latent shapes**
(the temporal axis splits 4 ways at `1 + 8k`, each spatial axis 2), and the
plan's **overlap waste is 1.502x** - not the 1.192x phase 16 recorded, which
was a 25-frame clip whose temporal axis did not split at all.

Baseline, before any change (wall 357.9 s, device 306.7 s summed over the
eight per-shape devices, host the remainder):

| item | s | share of wall |
|---|---:|---:|
| DEVICE (kernel timestamp queries) | 306.7 | 85.7% |
| host | 51.2 | 14.3% |

and the device half, per kernel kind, summed over all 16 tiles:

| kernel | s | share of device |
|---|---:|---:|
| `matmul_reg3` | 170.04 | 55.4% |
| `im2col3d_at` | 69.25 | 22.6% |
| `nchw_nlc` | 22.83 | 7.4% |
| `l2norm_scale` | 21.73 | 7.1% |
| `nlc_nchw` | 13.09 | 4.3% |
| `nlc_bias_nchw` | 3.87 | 1.3% |
| `concat2` | 3.15 | 1.0% |
| `silu` | 1.44 | 0.5% |
| `add2` | 0.98 | 0.3% |
| everything else | 0.37 | 0.1% |

**Three of those rows are one operation.** `nchw_nlc` + `l2norm_scale` +
`nlc_nchw` is `Builder3d::pixel_norm`, dispatched 37 times per decode, and
together they are **18.8% of the device time**. Against the card's own
measured roofline (`ltxv_bench vae 2 81 416 768`, the largest tile shape:
10450 GFLOP/s, 287.0 GB/s) the same pass reports the repo's roof-floor defect
rule firing on two of them - `nchw_nlc` at **6.2%** of its memory roof
(floor 35%) and `l2norm_scale` at **11.7%** of its compute roof (floor 30%) -
with the whole pass at **26.5%** of roof.

#### 1 - why a norm was three kernels, and why one is enough

`pixel_norm` (and `rms_norm`, the Wan VAE's learned-gain sibling - the same
three dispatches with a different `(gain, eps)` pair) normalises over the
CHANNEL axis of an `[C, T, H, W]` volume, which is the SLOWEST-varying axis.
The row-oriented `l2norm_scale` needs its rows contiguous, so the composed
form permutes into `[THW, C]`, normalises, and permutes back.

Both permutes are pure strided movement: `nchw_nlc` gathers
`x[(n*C+ch)*HW + l]` with `ch` varying fastest, so a warp's lanes land `HW`
floats apart and each fetched sector serves one useful float. The composition
pays that sector amplification TWICE in order to spare the middle kernel from
paying it once. And the middle kernel is worse than it looks: `l2norm_scale`
gives one thread each OUTPUT element, so every one of a row's `C` threads
redoes that row's whole sum of squares - its op count scales as `C` per
element, which is why it reads as compute-bound at 11.7% of a compute roof.

This is the same argument `layernorm2d` already settled for the LayerNorm
family (`crates/vision/src/blocks.rs`'s `LayerNorm2d` records that
measurement, and `.agents/rules/kernels.md` §E lists "composing several
coalesced stages beats a fused kernel" as a KILLED hypothesis). The L2/RMS
twin did not exist, so this phase wrote it: **`l2norm_scale2d`**, one
invocation per spatial position walking `C` at stride `T*H*W`, barrier-free
and array-free so `backend-cpu` JITs it.

It is **bit-identical**, not merely close, and that is a property of the
construction rather than luck: the permutes are exact rearrangements, and both
arms fold a position's sum of squares over ASCENDING channel index, so the
fused kernel performs the identical sequence of roundings on the identical
values. Every gate below therefore asserts on BITS.

The dispatch decision lives in ONE new private `Builder3d::chan_l2norm`, which
`pixel_norm` and `rms_norm` both call - so `crates/wan`'s causal VAE inherits
it without a line changing in that crate. `crates/ltxv/src/audio_vae.rs` kept
its own byte-identical composed copy of the same trio; it now dispatches the
fused kernel too, because a private composed copy left in one model is exactly
how the next model inherits the slow form.

`BRAIN_VAE3D_SPLIT_NORM=1` selects the composed arm.

#### 2 - the second finding: eight weight uploads and eight devices per clip

`LtxVaeTiledDecoder::decode_with` builds one graph per distinct tile SHAPE and
drops it before the next, so peak VRAM is one tile's rather than the clip's
(phase 16). Each of those builds also called `Gpu::open` - a fresh adapter,
queue and one shader compile per kernel - and re-uploaded the whole decoder,
~1.6 GB at fp32, through `Builder3d`'s per-builder weight memo.

At 25 frames that cost was paid up to four times and phase 16 measured it as
noise against the tiles. At 241 frames the cover has EIGHT shapes, and the
first build measures 3.68 s against a recording-only build of ~0.30 s: the
uploads alone are ~24 s of host time with the card idle for all of it.

`Builder3d::with_weights` / `finish_keeping_weights` let a caller carry the
device weight memo from one graph to the next, and `LtxVaeDecoder::build_on`
takes a device the caller owns. The tiled decoder now opens ONE device and
threads one memo through all eight builds. Peak VRAM is unchanged, and the
reason is structural rather than hopeful: the weights and one tile's
activations were already co-resident during every build, and the activations
are still dropped before the next shape's are allocated. Only the weights
survive, and they are the same buffers that used to be re-uploaded.

`BRAIN_LTXV_VAE_NO_SHARED_WEIGHTS=1` restores the per-shape device, and it is
what the measurement below is against. Same clip, same binary, the
`graph build` stage timer:

| 241 frames @1280x704, 16 tiles in 8 shapes | `NO_SHARED_WEIGHTS=1` | shared |
|---|---:|---:|
| graph build (8 shapes) | 31.96 s | **5.83 s** |
| graph drop (8 shapes) | 7.03 s | **1.86 s** |
| device open | inside the builds | 0.27 s, once |
| wall | 315.0 s | **283.7 s** |
| **DEVICE (kernel timestamps)** | **255.75 s** | **257.57 s** |

**Device time not moving is the control**, and it did not: 255.75 vs 257.57 s,
0.7% apart, on a change that only decides where a weight buffer came from.
Everything the change is worth - 26.1 s of build plus 5.2 s of teardown - is
host time the card was idle for, and it is 1.11x on the stage's wall clock.

Per shape, the un-shared builds measure 3.97 / 3.96 / 3.79 / 4.13 / 3.41 s and
the shared ones 3.68 (the first, which really does upload) then 0.31 / 0.26 /
0.28 / 0.24 / 0.34 / 0.31 / 0.40 s. The first build being the same in both
arms is the point: every later one falls to what RECORDING costs.

#### 3 - measured: the norm arms, same binary, same box, one idle P40

`ltxv_bench decode <241f latent> tiled` on gpu1, `BRAIN_PROFILE=1`,
`BRAIN_GPU_WAIT_S=1800`. Both arms carry the shared device/weights of item 2,
so the ONLY difference between these two columns is which norm ran:

**Three arms, run back to back in that order**, because a second card in the
same chassis was under continuous load from an unrelated job all session and
gpu1 sat at 89 C throttled to 1354-1392 MHz against gpu0's 1531 MHz. A single
before/after pair on a drifting die is not a measurement, so the fused arm was
run twice, once at each end:

| 241 frames @1280x704, 16 tiles | fused (1st) | `SPLIT_NORM=1` (2nd) | fused (3rd) |
|---|---:|---:|---:|
| wall | 283.7 s | 358.3 s | 300.3 s |
| of which DEVICE (kernel timestamps) | 257.57 s | 332.49 s | 273.27 s |
| of which HOST | 26.1 s | 25.8 s | 27.0 s |
| the channel norm's kernels | **2.21 s** | 61.82 s | **2.28 s** |
| `matmul_reg3` | 174.03 s | 184.12 s | 183.77 s |
| `im2col3d_at` | 71.44 s | 76.24 s | 77.00 s |

**The third arm is what makes the second column readable.** Against the arm
run immediately before it, in the same thermal state, the convolution rows
agree to 0.2% (`matmul_reg3` 183.77 vs 184.12) and 1% (`im2col3d_at` 77.00 vs
76.24) - so the convolutions did not move, and the ~5% they appeared to move
between arms 1 and 2 was the die cooling down between the session's start and
that point. What DID move is one stage:

| matched-thermal comparison (arm 3 vs arm 2) | | |
|---|---:|---|
| the channel norm's kernels | 61.82 s -> 2.28 s | **27.1x**, 59.5 s removed |
| DEVICE | 332.49 s -> 273.27 s | **1.217x** |
| wall | 358.3 s -> 300.3 s | **1.193x** |
| HOST | 25.8 s -> 27.0 s | unchanged, and that is the CONTROL |

**Host time not moving is the control**, and it did not: a device-side kernel
fusion must leave the host half alone. On a cool die the same change measured
283.7 s wall against the 358.3 s split arm (1.26x), which is the best observed
rather than the attributable number.

At one tile shape, where a roofline is available
(`ltxv_bench vae 2 81 416 768`, latent `[128, 11, 13, 24]`, both arms of the
same binary, roofline measured identically in both runs at 10450 GFLOP/s /
287.0 GB/s):

| | composed | fused |
|---|---:|---:|
| whole pass | 23922 ms | 20272 ms |
| whole pass, % of roof | 26.5% | **30.3%** |
| roof-floor DEFECT rows | `nchw_nlc` 6.2% of memory roof, `l2norm_scale` 11.7% of compute roof | **none** |
| `l2norm_scale2d` | - | 60.8% of its memory roof |

Both defect rows the repo's own floor rule was flagging are gone, and the
kernel that replaced them sits at a normal fraction of its bandwidth roof.

#### 4 - the host half, attributed with nothing left over (fused arm 1)

Stage spans added to `LtxVaeDecoder::decode` and
`LtxVaeTiledDecoder::decode_with` (permanent, `BRAIN_PROFILE`-gated). None of
them overlaps another, so they sum to the call - the fused arm above:

| host item | s | share of wall |
|---|---:|---:|
| `submit` + device wait (this is the DEVICE, listed so the rows sum) | 258.07 | 91.0% |
| pixel readback, 16 tiles | 6.04 | 2.1% |
| graph build - weight upload + recording, 8 shapes | 5.83 | 2.1% |
| host `unpatchify`, 16 tiles | 5.01 | 1.8% |
| blend accumulate (host), 16 tiles | 4.92 | 1.7% |
| graph drop (device teardown), 8 shapes | 1.86 | 0.7% |
| blend divide (host) | 0.89 | 0.3% |
| device open | 0.27 | 0.1% |
| latent slice (host), 16 tiles | 0.04 | 0.0% |
| latent upload, 16 tiles | 0.00 | 0.0% |
| **unaccounted** | **0.8** | **0.3%** |
| **wall** | **283.7** | |

`submit + device wait` is 258.07 s against 257.57 s of summed kernel
timestamps, so the host residual inside that span is 0.5 s: on this path the
card really is the whole of it.

The first graph build measures 3.68 s and every later one ~0.30 s, which is
what item 2's fix buys per shape - eight shapes on their own devices would be
~29 s of upload plus eight device opens, against 6.1 s.

#### 5 - the gates, and which mutation caught which

Both changes are BIT-identical claims, so every gate asserts on `to_bits`, not
on a cosine floor - and deliberately not on cosine alone, which is scale
invariant and would score a uniformly mis-scaled image perfect.

New:

* `crates/vae/tests/blocks3d_norm.rs` - checkpoint-free, both norms
  (`pixel_norm` and `rms_norm`, the two `(gain, eps)` pairs into the one
  dispatch site) at a shape whose four extents are all different, on the
  default device AND explicitly on `backend-cpu` (the kernel declares
  `@cpu yes`, and only a run on the CPU JIT holds that claim up). Prints the
  differing-word count so a reader can see it ran.
* `crates/ltxv/tests/vae_parity.rs::the_fused_channel_norm_changes_no_bit_of_
  a_real_weight_decode` - the REAL 170-tensor checkpoint, 17 frames, both
  arms in one process: 0 of 208896 decoded words differ.
* `crates/ltxv/tests/vae_tiling.rs::sharing_one_device_and_one_weight_set_
  across_tile_shapes_changes_no_bit` - a deliberately MULTI-shape cover (a
  one-tile plan would prove nothing, since the first build uploads either
  way), both arms of `BRAIN_LTXV_VAE_NO_SHARED_WEIGHTS`.
* `gpu_core::cost` gained an `l2norm_scale2d` row with a hand-computed
  expectation, and the kernel catalogue regenerated.

Every guard was mutated separately:

| mutation | what went red | what it proves |
|---|---|---|
| fold the sum of squares over DESCENDING `c` | both `blocks3d_norm` tests, 2709 of 5040 words | the gate sees a pure re-association, not just a gross error - which is the whole reason the claim is "bit-identical" rather than "close" |
| drop the per-channel gain `g[c]` from the scale pass | both `blocks3d_norm` tests | the gain is really applied and really compared |
| index the channel axis as if it were contiguous (the NLC assumption) | the CPU-backend test **SIGSEGV**s; the GPU arm would have read out of range | worth recording as a caught mutation of a different KIND: on `backend-cpu` an out-of-range index is a crash, not a wrong number, so a JIT-backed gate can catch an indexing bug the bounds-checked GPU arm would only show as garbage |
| make a weight-memo HIT hand back some other tensor's buffer | `sharing_one_device_and_one_weight_set_across_tile_shapes_changes_no_bit` | the sharing gate can see a mis-keyed memo, which is the only way this change can be wrong |
| the sharing gate's own SHAPE-COUNT guard | it fired, on the first geometry tried | recorded because it is the finding: an 8x8 latent under a 4-cell tile splits into three EQUAL tiles, so the obvious geometry to copy from the gate above would have compared the shared arm against itself and passed forever. The guard is why that was caught rather than shipped |

Pre-existing gates re-run and green, with the printed numbers unchanged to
every digit this crate records:

* `vae_parity` - 8 passed / 0 failed / 1 ignored: encoder, decoder and round
  trip at 9 and 17 frames against the dumped goldens, all at cosine
  1.000000000, plus the explicit `backend-cpu` run.
* `vae_tiling` - 4 passed / 0 failed / 3 ignored, and the approximate gate
  reproduces phase 16's numbers exactly: `9-tile tiled vs whole: cosine
  0.999093484, rel_l2 4.2641e-2, max_abs 1.6697e-1`, hard cut 0.992828795 /
  1.2401e-1 / 5.4389e-1. Nine identical digits on a lossy path is a stronger
  statement about "nothing moved" than the pass/fail is.
* `cargo test -p brain-ltxv` - every test binary green.
* `cargo test -p brain-wan --test vae_parity` - 9 passed / 0 failed. Wan's
  causal VAE calls `Builder3d::rms_norm` and therefore took the fused kernel
  with no change in that crate at all; this is the gate that says so.
* `cargo test -p brain-vae -p brain-gpu-core` - green, including the new
  `l2norm_scale2d` cost row and the coverage floor.
* `make clippy` - exit 0, 0 warnings (baseline 0). `make kernels-table/check` -
  up to date, 431 kernels, all fields declared.

#### 6 - recorded, NOT done - and the largest one is not a kernel

Re-profiled after the fix, the decode's device half is:

| kernel | share of device |
|---|---:|
| `matmul_reg3` | 67.6% |
| `im2col3d_at` | 27.7% |
| everything else | 4.7% |

with `matmul_reg3` at ~42-46% of the card's measured compute roof and
`im2col3d_at` at ~51-58% of its memory roof. Neither is a defect; both are
structural, and the ranked ways past them are:

* **The overlap waste is 1.502x at this clip geometry, and that multiplies
  BOTH rows above.** Upstream's conv-VAE auto layout
  (`_CONV_AUTO_LONG_SIDE = (768, 64)`, `_CONV_AUTO_FRAMES = (80, 24)`) tiles a
  `[31, 22, 40]` latent as 4 temporal x 2 x 2, and the temporal axis is where
  the waste is: a 3-cell overlap on a 10-cell tile is ~43% extra per interior
  tile, against ~18% (height) and ~9% (width) for a 2-cell spatial overlap on
  a 13- and 24-cell tile. One third of every second the card spends in this
  stage decodes pixels the blend then averages away. A brute-force search over
  `(t, h, w)` tile sizes at upstream's own overlaps, constrained to a per-tile
  pixel volume that is known to fit, bottoms out at **1.30-1.32x** - so roughly
  12% of the whole stage is available from tile SIZING alone, with no kernel
  written. It is NOT taken here, deliberately: phase 16 recorded "a VRAM-budget
  search for the tile size" as out of scope because sizing tiles from live free
  VRAM needs the per-request VRAM estimate `LtxvResident::estimate` still
  lacks, and inventing a second answer to "how much fits" while that is open
  would give the workspace two. The arithmetic is here so the next pass does
  not have to re-derive it.
* **`im2col3d_at` exists only to feed `matmul_reg3`.** It materialises
  `27 x Cin` floats per output position, which the GEMM then reads back - so
  the lowering pays roughly twice the im2col volume in DRAM traffic that an
  implicit-GEMM conv (staging the im2col patch directly into the GEMM's shared
  tile) would not pay at all. That is a real new kernel with a register-tiled
  inner loop, not a selection fix, and it is the largest single kernel-level
  item left.
* **The 16 tiles are decoded strictly one after another and the host stages
  between them do not overlap the card.** Readback + `unpatchify` + blend is
  ~16 s of the 284, all of it with the queue empty. Double-buffering tile `i`'s
  host work against tile `i+1`'s submit would recover most of it, and the
  arena's own precedent applies: the decoder graph is static and re-submitted
  per tile, so the only thing that has to alternate is the output buffer.
* **The encoder is untouched.** `LtxVaeEncoder` uses the same `pixel_norm`, so
  it inherited the fused kernel for free, but nothing in this port encodes a
  clip large enough to profile.

#### 7 - what this is worth on a real run, and what it is not

The stage this phase touched was 645.7 s of a 3162.6 s clip. Nothing here
changes the DiT, so the honest projection is the decode's own ratio applied to
that line and nothing else: at the matched-thermal 1.217x on device time and
the same host half, the decode lands near 530 s and the run near 3047 s -
about 3.6% end to end, against 20.4% of the run spent in the stage. The stage
is now 91% device-bound and the device half is 95% two kernels, so the next
1.2x on it is item 6's tile geometry, not another fusion.

**A caveat on absolute seconds in this section.** The decode-only harness
measures 357.9 s for the pre-change path where the real pipeline reported
645.7 s for the same clip geometry, and that gap is not explained here. The
harness decodes a synthetic latent through `LtxVaeTiledDecoder` directly, so
it excludes whatever the pipeline does around the call (the weight import is
outside the harness's timer) and it ran with the other card idle. Every ratio
in this phase is a same-binary, same-harness A/B and is unaffected; the
absolute 645.7 s is the pipeline's own number and re-deriving it needs a run
this phase deliberately did not do.

### Phase 35 - the block loop stops waiting for the block it just submitted

Phase 33 left the warm video forward 90% device-bound and named the remaining
item: the per-block BLOCKING one-word read. It exists so wgpu retires its
staging and its pool does not grow (phase 30 measured chaining without it and
correctly rejected it), and the arena had just removed the reason it had to be
the last thing a block does. This phase moves it.

#### 0 - the change is a reordering, and it needs no second arena

A block body is three phases: RECORD (bind groups, plus this block's weight
upload when residency could not keep it), SUBMIT, WAIT. Recording touches no
device memory, so the wait does not have to sit between this block's submit
and the next block's recording. It only has to sit between the previous
block's submit and THIS block's submit:

```text
  serial:    record(l)  submit(l)  wait(l)          record(l+1) submit(l+1) wait(l+1)
  pipelined: record(l)  wait(l-1)  submit(l) flush  record(l+1) wait(l)     submit(l+1) flush
```

The device sees the identical submissions in the identical order with at most
one block in flight either way. What changes is where the host is while the
card works. `ltxv::block::block_pipeline` is the switch
(`BRAIN_LTXV_NO_PIPELINE=1` opts out), two call sites, one per stream pair.

**The recorded design sketch was alternating arenas, and that turned out to be
unnecessary.** The sketch assumed the aliasing window is "recording block `l+1`
while block `l` runs". It is not: block `l+1` only takes the arena's HANDLES
during recording, and every dispatch that writes one of those buffers is
submitted after the wait that completes block `l`. One arena is enough, and a
second would have cost a whole block's scratch in device memory for nothing.
`gpu_core::scratch`'s contract is restated to the condition that is actually
required - drained before the new scope's dispatches are SUBMITTED, not before
the new scope is entered.

#### 1 - the instrument had to be fixed first, and that is the reusable finding

`ltxv_bench` turns `BRAIN_PROFILE` on, which routes every flush through
`backend_wgpu`'s `flush_timed`. That path resolved its query sets and then
MAPPED the resolve target to read the ticks - and mapping blocks until the
submission completes. So the profiler that reports "device share of wall" was
itself a full device round trip per flush: with it on, the pipelined arm
measured exactly as fast as the serial one, because the profiler had put the
drain back. The readback is now deferred (`resolve_ticks`, folded in on any
deliberate read of the accumulator) and the flush returns while the card works.

A second, worse instance of the same class was created by that fix and found
by measurement: `Drop for WgpuBackend` reports its profile, so resolving from
`dump_profile_now` made EVERY handle drop wait for the queue - including the
`Gpu::share` an evicted resident block holds, dropped once per streamed block
in the middle of a forward. A destructor is outside every span the caller
times, so it did not appear in any stage; it read as the pipelining simply not
working, and the stage table blamed the weight upload. A temporary probe around
the block loop's own phases is what found it:

| span, warm forward, 8 layers, T = 13200, nothing resident | serial | pipelined, bug present |
|---|---:|---:|
| `forward_prod_dev`'s own timed spans | 10306 ms | 545 ms |
| dropping the block (untimed by any stage) | 96 ms | 9801 ms |

Only the LAST handle on a device resolves from `Drop` now. Both properties are
gated in `crates/backend-wgpu/tests/kernel_timing.rs`.

#### 2 - measured, both arms, same binary, one idle P40 (gpu0)

`ltxv_bench streamed 48 13200 1024 1 1 1 <reps>`, real distilled Q8_0 weights,
device-resident session, one distinct timestep. The first warm call is the
warm-up and is excluded; the headline is the best of the rest.

| | before | after | |
|---|---:|---:|---|
| warm forward, best | 65.44 s | **62.49 s** | **1.047x** |
| of which DEVICE (kernel timestamps) | 58.22 s | 58.24 s | **+0.03%** |
| of which HOST | 7.22 s | 4.24 s | -41% |
| device share of wall | 89.0% | **93.2%** | |
| first (cold) forward, best of the runs taken | 138.87 s | 106.21 s | 1.31x |
| peak VRAM (separate UNTIMED observation, `nvidia-smi -l 1`) | 18092 MiB | 18555 MiB | +463 MiB |
| host peak RSS | 36792 MiB | 36863 MiB | +71 MiB |
| resident blocks the policy granted | 23 of 48 | 23 of 48 | |

Device kernel time not moving is the control and it did not move. Peak VRAM is
+2.6% on a 24576 MiB card and, over six consecutive forwards, it steps to
18555 MiB during the first warm one and then stays there exactly - the extra is
one further block's upload staging in flight, not a pool that grows.

Where the wall went, warm call, same run:

| stage | before | after |
|---|---:|---:|
| block submit+wait (sum over layers) | 58371 ms | 56288 ms |
| block weight upload (the 25 non-resident blocks) | 2343 ms | 1685 ms |
| activation/context/adaLN record+upload | 2627 ms | 1673 ms |
| device -> host readback | 340 ms | 1478 ms |
| output stage (host) | 875 ms | 912 ms |

The readback grows because it is now where the LAST block's device tail is
waited for; `submit+wait` correspondingly holds 47 of the 48 blocks. The two
in-loop host rows are what the card now runs underneath.

**The two effects, separated** (`BRAIN_LTXV_RESIDENT_BLOCKS` forces the window
size, 8 layers, same token width, best warm call):

| | before | after |
|---|---:|---:|
| every block resident, so nothing streams | 12.18 s | 11.80 s |
| nothing resident, so every block uploads | 12.99 s | 12.55 s |

Device time was 9.64-9.66 s in all four. Both the graph recording and the
weight upload overlap; neither is a special case.

#### 3 - how far ahead to run, answered with the memory number

One block, and the evidence is what the next block costs. Deleting the wait
altogether (the arena and the pipelining otherwise unchanged) runs the whole
48-block stack ahead of the card:

| | drain kept | drain deleted |
|---|---:|---:|
| warm forward, best | 62.49 s | 61.82 s |
| peak VRAM | 18555 MiB | **23222 MiB** of 24576 |
| host peak RSS | 36863 MiB | 41472 MiB |

1.1% more wall for 4.7 GB of VRAM and 4.6 GB of host RSS, on a card with
1354 MiB left over - and the audio+video path already peaks at 21082 MiB with
the drain, so the same trade does not fit there at all. This is phase 30's
rejected experiment, now with the number attached. Deeper lookahead is also not
where anything is left: at one block the card is busy 1.21 s per block against
0.09 s of host work, so the host is already idle 93% of the loop.

#### 4 - the gates, and which mutation caught which

`crates/ltxv/tests/block_pipeline.rs` gates the forward's OUTPUT BITS across
the full cross product of the two switches (pipelining x arena), at the tiny
config and at the real 22B block's own shapes, against the arm with both off.
Bits, not cosine: this reorders nothing arithmetically.

| mutation | what went red | what stayed green, and why that is the finding |
|---|---|---|
| delete the drain from the pipelined arm | **nothing** | both bit gates, at both widths. Not a blind gate: see the row below. wgpu-core prepends a "(wgpu internal) Transit" pass to EVERY submission, inserting barriers from the DEVICE-GLOBAL tracker, and a storage buffer is never in `ordered_uses_mask` - so a recycled buffer's old and new users are ordered by the backend across submissions, and there is no aliasing left for a bit gate to see. The drain is load-bearing for MEMORY (section 3), not for arithmetic |
| hand back the previous arena slot AND drop the uniqueness guard | **both** `block_pipeline` gates | nothing. This is the control that proves the bit gate can see arena aliasing under the pipelined arm, which is what makes the row above a statement about wgpu rather than about the gate |
| `flush_timed` resolves its own ticks (the pre-change behaviour) | `a_timed_flush_returns_before_the_device_has_finished` | it also took `dropping_a_shared_handle_does_not_wait_for_the_device` red, by its own precondition assert: the flush had already drained, so there was no outstanding work left for the drop to wait on. That guard exists so the drop test cannot pass vacuously, and it fired |
| `Drop` resolves on every handle, shares included | `dropping_a_shared_handle_does_not_wait_for_the_device` | the other two. This is the exact bug section 1 describes, and nothing but a wall-clock measurement caught it before the test existed |
| resolve only the NEWEST deferred batch | `deferring_the_timestamp_readback_loses_no_dispatch` | the other two |

#### 5 - recorded, NOT done

* **The output stage is still host math at production width** - a LayerNorm, a
  per-token modulate and a `[t, dim] x [dim, out]` linear over 13200 tokens,
  ~0.9 s per forward with the card idle. It is now the largest single host row
  that pipelining cannot hide, because it runs after the block loop with
  nothing left to overlap. Every kernel it needs exists.
* **The final readback is the second** (~1.5 s), and most of that is the last
  block's device tail rather than the 206 MiB copy. Only a deeper pipeline
  across DENOISE STEPS could hide the copy itself.
* **The arena is still rebuilt once per forward** (phase 33's item, unchanged):
  `DitSession::device_for_call` hands out a fresh `Gpu::share` per call.
* **The audio+video path takes the same switch and was not re-measured here.**
  The code is the same reordering in `LtxAvBlockQ::forward_prod_dev` and the
  bit gate covers the video stream only; the AV arm's own before/after numbers
  are open.
* **Only `crates/ltxv` pipelines.** Every other block-stack model in the
  workspace has the same shape, and the reordering needs no per-model
  machinery beyond the three backend calls.

### Phase 36 - the second card, priced: why splitting ONE denoise forward across two P40s cannot pay

Phase 35 left the warm video forward 93.2% device-bound, so the host is no
longer the constraint and the single card is. Meanwhile gpu1 sits at 0% for a
whole generation, and phase 15's CFG-parallel path has nothing to split: the
distilled schedule denoises at `guidance = 1.0`, where no unconditional
forward is issued at all.

This phase asks whether a two-card split of one forward pays, and the answer
is no by a wide margin. Nothing was sharded; what was built is the
measurement that decides it, plus one general fix the measurement found.

#### 0 - the structural fact that caps the whole idea

A pipeline split over the 48-block stack (blocks `0..k` on one card, `k..48`
on the other) has **exactly one activation in flight**, so it has no
parallelism at all - it is a 100% bubble, not a small one:

* `guidance = 1.0` on the distilled schedule, so there is no conditional /
  unconditional pair to place on two cards (`pipeline::generate`'s default,
  and `cfg_on = guidance > 1.0` never fires);
* denoise step `n+1` consumes step `n`'s whole output;
* a long-form window conditions on its predecessor's tail.

So `wall = stage0 + crossing + stage1`, each card idle exactly while the other
works, total device time unchanged. Splitting can only win what it removes,
and the only thing it removes is **weight streaming**: two cards hold two
residency windows, so nearly the whole stack becomes resident. That effect is
measurable today, on one card, without building anything.

#### 1 - the residency prize, measured on its own

`ltxv_bench streamed 48 13200 1024 1 1 1 <reps>`, real
`ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, int8, one distinct timestep,
device-resident session, gpu0, both cards verified idle BEFORE each run and
`nvidia-smi` never sampled during one. First warm call excluded as warm-up;
headline is best of the rest.

| arm | warm forward, best | device (kernel timestamps) | host | uploads / forward |
|---|---:|---:|---:|---:|
| policy window, 23 of 48 slots | **62.33 s** | 58.08 s (93.2%) | 4.25 s | 26 |
| `BRAIN_LTXV_RESIDENT_BLOCKS=0` | 63.79 s | 58.18 s (90.7%) | 5.56 s | 48 |

**22 block uploads cost 1.46 s**, and device kernel time did not move (0.17%)
- the control that says this is a host/PCIe difference and not a different
amount of compute.

That difference is the same size as the one sharding would buy, and it is the
whole prize. A 24-block half-stack with the same policy gets 23 slots for 24
blocks, so `CyclicScan` pins 22 and rotates 2: **4 uploads per forward across
both cards instead of 26**, i.e. 22 removed - the identical quantity measured
above.

Note the cold call goes the other way: 105.15 s with the window against
81.03 s without it, because a prefill of 23 blocks is serialised ahead of the
block loop. Residency is a warm-path win that costs the first forward.

#### 2 - what a stage boundary costs, on two real cards

`crates/gpu-core/tests/pcie_handoff.rs` (new, `#[ignore]`d hardware probe):
one real `[13200, 4096]` fp32 activation - 206 MiB, exactly what crosses a
pipeline cut - through the paths this engine actually uses, best of 3 with a
warm-up, both cards idle.

| direction | ms | GB/s |
|---|---:|---:|
| host -> device (`write_f32_chunked`, 1 MiB chunks) | 36.8 | 5.87 |
| device -> host (`Gpu::read`), before section 3 | 264.2 | 0.82 |
| device -> host, after section 3 | 208.5 | 1.04 |
| card 0 -> card 1, whole handoff, after section 3 | 201.8 | 1.07 effective |

So a two-stage pipeline costs about **0.20 s per forward** for its one
crossing.

#### 3 - the readback is not bus-bound, and that is the reusable finding

0.82 GB/s against 5.87 GB/s in the other direction is not a PCIe asymmetry.
A temporary probe splitting `WgpuBackend::read` into its three phases, same
206 MiB payload:

| phase | ms |
|---|---:|
| `copy_buffer_to_buffer` + submit + poll (the actual bus transfer) | 35 |
| `get_mapped_range()` -> `.to_vec()` (host memcpy into a FRESH allocation) | 150 |
| the same memcpy into a PRE-FAULTED sink | 22 |
| `unmap` | 0.01 |

The bus does 5.9 GB/s in both directions. **The cost is the destination
`Vec`'s first-touch page faults** - 206 MiB of freshly mmapped pages, faulted
and zeroed as the memcpy writes them, at roughly a seventh of the rate the
same copy runs at once the pages exist. Reading the mapped range is not slow;
writing brand-new anonymous pages is.

Two consequences, one fixed here and one recorded:

* **Fixed: the staging buffer is now reused.** `read` allocated a fresh
  `MAP_READ` buffer per call, and pinning host pages is the expensive half of
  allocating one - the same finding that turned upload staging into a recycled
  buffer (workspace `Cargo.toml`'s dependency-override notes) applied to the
  direction that had never been looked at. `DeviceShared` keeps one, grown on
  demand and capped at 512 MiB; guarded by the `io` lock every read already
  takes; `BRAIN_GPU_NO_READ_STAGING_REUSE=1` opts out. Worth 264.2 -> 208.5 ms
  on the probe. On the LTXV forward it is one readback per forward, i.e. below
  the wall-clock noise floor - claimed at the probe, not at the model. The
  full 48-layer control run confirms exactly that: **62.42 s best of 3 against
  the 62.33 s baseline** (inside the 62.41 / 62.45 / 62.42 / 63.30 spread of
  the run itself), device kernel time **58.08 s in both**, and output
  `mean=0.044928 std=0.679524 min=-1.391593 max=1.823861 nonfinite=0`
  identical to six decimals in both arms.
* **And it exposed a latent alignment bug that had nothing to do with it.**
  Three sites cast a mapped buffer's bytes in place
  (`cast_slice::<u8, u64>` for the timestamp resolve, `<u8, f32>` for the
  readback). A mapped pointer is `memory_block.mapped_ptr +
  suballocation.offset`, and keeping one buffer alive moved gpu-allocator's
  packing enough that the timestamp-resolve buffer landed 4-mod-8: a real 22B
  run died mid-forward with
  `cast_slice>TargetAlignmentGreaterAndInputNotAligned`, in the profiler,
  minutes in, after every stage of the first forward had printed. All three
  sites now widen the DESTINATION instead
  (`backend_wgpu::copy_pod_from_bytes`), which has no alignment precondition
  at all. Gated deterministically without a GPU by copying a payload out of a
  byte slice at every source offset in `0..8`.
* **Not fixed: the destination allocation.** `Gpu::read` returns a fresh
  `Vec<f32>` by contract, so the 150 ms cannot be amortised without a
  `read_into`-shaped seam and per-call-site adoption. Left undone deliberately
  - it is worth ~0.2% on the LTXV forward, and it only becomes decisive for a
  scheme that crosses the boundary per BLOCK (section 4).

#### 4 - the three alternatives, priced against the same numbers

| scheme | latency, one clip | second card | notes |
|---|---:|---|---|
| today, one card | 62.33 s | idle | |
| **pipeline-parallel over blocks** | **~61.1 s (1.02x)** | ~50% busy | `-1.46 s` residency, `+0.20 s` crossing. Bit-identical. |
| sequence-parallel (split tokens, exchange K/V per block) | ~43 s (1.45x) | ~100% busy | 48 crossings; upper bound, assumes the exchange does not overlap compute |
| the same, with the section-3 destination fix | ~38 s (1.65x) | ~100% busy | |
| two independent requests, one per card | 62.33 s each | 100% busy | already shipped, phase 15 item 2, **2.11x throughput** |

**Pipeline-parallel is declined, and this time the reason is a measurement
rather than a memory ceiling** (phase 15 item 3 declined the same loader
because the model fit one card; that argument is unchanged and this one is
additional). 1.02x is not merely marginal - it is *negative* for a server,
because it spends the second card on one request to save 2%, where the
existing device pool spends it on a second request and returns 2.11x
throughput. Any single-request split has to beat that trade, and a 100% bubble
cannot.

**Sequence-parallel is the only split that can.** Each card owns half the
tokens and runs every block; per block the two cards exchange their K and V
halves (`[6600, 4096]` each, 206 MiB per card per block) and each computes
attention for its own queries against the full key set. It would be
**bit-identical**, which is worth stating because it is not obvious: every
per-token operation (norms, modulation, both GEMM halves, the FFN) is
unchanged by which other tokens accompany it; `matmul_i8_dyn`'s activation
scales are per ROW; and a query row's flash-attention reduction visits the
same key tiles in the same order regardless of how the query rows are split.
Its cost is 48 boundary crossings, which is exactly why section 3's
destination-allocation number is the gate on it: 9.7 s of transfer today
against 4.5 s with the fix, on ~29 s of halved device time.

**And even that is second in line.** At this width self-attention is ~60% of
device time at ~38% of the fp32 roof, against a DP4A roof 4.1x higher. An int8
attention path is a bigger win than two-card sequence-parallelism, on one
card, with no transfer to hide and no second device to occupy.

#### 5 - gates, and which mutation caught which

`crates/backend-wgpu/tests/read_staging_reuse.rs` (5 tests) covers the reuse.
Both arms of `BRAIN_GPU_NO_READ_STAGING_REUSE` are green.

| mutation | what went red | what stayed green |
|---|---|---|
| map the whole cached buffer (`slice(..)`) instead of `slice(0..want)` | `a_shorter_read_after_a_longer_one_returns_only_its_own_bytes` | the other four - a stale tail is invisible to a growing or equal-size read, which is why the shrinking ladder exists |
| never re-copy into a reused buffer | three of the five | `a_zero_length_read_is_empty` |
| `retain_read_staging` becomes a no-op (reuse silently reverted) | `a_loop_over_one_shape_pins_its_staging_pages_once` | **all four correctness tests** - and that is the finding: every correctness assertion passes just as well for a `read` that allocates every time, so the mechanism needed its own observable (`WgpuBackend::read_staging_allocations`) or a revert would show up only as a wall clock nobody watches |
| `copy_pod_from_bytes` goes back to `cast_slice` | both `mapped_copy_tests` cases, with the exact panic the real run produced | everything else, on this box - which is the point: the trigger is an allocator packing decision, so the property has to be gated at every source offset rather than waited for |

`pcie_handoff.rs` asserts only what cannot be hardware-dependent: a rate above
any host bus this engine targets (the §E.0 host-timing failure), a zero-time
transfer, and - on the two-card probe - that the receiving card holds exactly
the sending card's bytes, so a handoff cannot measure fast by moving nothing.

#### 6 - recorded, NOT done

* **No sharded real-checkpoint loader was built.** The tracked gap ("a
  real-checkpoint-weight version needs a GGUF-streaming int8 shard loader")
  stays open, now with a measured reason not to close it for latency. If it is
  built, it should be built for sequence-parallelism, which needs replicated
  weights on both cards rather than a layer split.
* **The composition in section 4's second row is arithmetic**, not a run: both
  of its terms are measured (1.46 s residency, 0.20 s crossing) but their sum
  was not observed on a sharded forward, because no sharded forward exists.
* **The sequence-parallel row is an upper bound** and assumes the K/V exchange
  does not overlap compute and that device time halves exactly.
* **`Gpu::read`'s destination allocation** (section 3) is the single highest
  leverage item for any future multi-device split, and is untouched.
* **A pre-existing hang, found while running the suite and reproduced on
  unmodified `main`**: `crates/backend-wgpu/tests/upload_flush.rs` spins at
  100% CPU forever when its two tests run concurrently, because each builds
  its own `WgpuBackend` and two Vulkan devices in one process is the deadlock
  this driver has. `--test-threads=1` passes in under a second. The default
  `TEST_THREADS=8` reaches it. Not fixed here - it is unrelated to this
  phase - but it is a live trap for anyone running the suite.

### Phase 37 - IC-LoRA reference conditioning, and the character-swap claim it does not support

Asked for an identity-preserving character swap in an existing clip, driven by
a description of LTX-2.5 as having "structural control layers built natively
into the inference pipeline" (a "Canny IC-LoRA" node), identity injected through
the image-to-video path, and a diffusion decoder that "dynamically paints the
target actor's face onto the stuntman's moving body". Verified first. Most of
that is wrong, and the parts that are right are not arranged the way it says.

#### 1 - what the reference actually has

* **`ICLoraPipeline` is real, and it is bring-your-own-adapter.** Its own
  docstring (`packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py:60-69`) says
  "The specific IC-LoRA model should be provided via the loras parameter", and
  `main()` passes whatever `--lora` gave it. Nothing structural is "built into"
  the pipeline: `--video-conditioning` only appends reference tokens, and
  `reference_video_cond.py:15-18` states that attending across them is what the
  ADAPTER was trained to do. Feeding reference tokens with no matching IC-LoRA
  is an out-of-distribution sequence, not a weaker control.
* **`IC` is `In-Context`, not "Identity & Composition"** - the trainer config
  header (`packages/ltx-trainer/configs/v2v_ic_lora.yaml`) and every Lightricks
  model card say so. Identity is not simply absent, though, and the precise
  shape of the problem is worth carrying: **the reference slot holds exactly one
  thing, and each adapter is trained for one reading of it.** Union-Control
  reads it as a structure signal; `LTX-2.3-22b-IC-LoRA-Ingredients` (tagged
  `character-consistency`, `reference-sheet`) reads it as a character/prop
  reference sheet. "Lock the choreography AND inject the actor" therefore wants
  two incompatible trained semantics in one slot. Ingredients also GENERATES
  from its sheet rather than preserving an input clip, and has no LTX-2.5 build.
  No path anywhere here accepts a face crop or a per-subject embedding.
* **There is no Canny control mode in the pipelines package.** The only
  case-insensitive "canny" hits under `ltx-pipelines` are the word "uncanny"
  inside negative prompts. Canny appears in the TRAINER as a way to build your
  own reference dataset (`packages/ltx-trainer/scripts/compute_reference.py`,
  `docs/dataset-preparation.md:437` and its `Lightricks/Canny-Control-Dataset`).
* **The diffusion decoder cannot paint a face.** `DiffusionVideoDecoder`'s
  entire signal input is the latent: `decode_video(latent, tiling_config,
  generator)` (`packages/ltx-core/.../video_vae/diffusion_video_decoder.py`,
  `forward`/`decode_video`). It has no identity, text or image input of any
  kind. Everything about who is in the frame is already fixed by the DiT.

#### 2 - released weights: the finding that decides the whole task

Enumerated all 57 `Lightricks` HF models. **LTX-2.5 has exactly ONE IC-LoRA:
`LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`, a detailer.** The structural
control adapters exist only for older generations - `LTX-2.3-22b-IC-LoRA-Union-
Control` (its card: "Control Type: Union conditioning - Canny + Depth + Pose"),
`LTX-2-19b-IC-LoRA-{Canny,Depth,Pose,Union}-Control`, and 13b ones for 0.9.7 -
and the repo README is explicit that "a LoRA only works with the model it was
trained on". This crate builds `ltx25_22b` only, so none of them applies.

Placebo-checked every candidate by range-fetching the safetensors header and
then real weight slices (never a full download). A LoRA at default init has
`lora_B` exactly zero; all of these are genuinely trained:

| checkpoint | rank | `lora_B` std | kurtosis | frac_zero |
| --- | --- | --- | --- | --- |
| `LTX-2.3-22b-IC-LoRA-Union-Control` | 64 | 7.9e-3 | 3.57 | 0.0 |
| `LTX-2-19b-IC-LoRA-Canny-Control` | 64 | 5.7e-3 | 3.63 | 0.0 |
| `LTX-2-19b-IC-LoRA-Pose-Control` | 64 | 5.7e-3 | 3.82 | 0.0 |
| `yuvraj108c/LTX-2.5-22b-IC-LoRA-BBox-Control` (third party) | 32 | 1.2e-2 | 1.26 | 0.0 |

(`LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` is gated and returned 403 to an
authenticated range request; its licence was not accepted on this box.) All
960-tensor files, 48 blocks x 10 projections, `attn1/attn2/ff` - the same shape
`lora.rs` already applies. Kurtosis is nowhere near the 1.801 of a uniform
default init, which is the check that caught the `pulid.md` placebo.

So: **an identity-preserving character swap is not achievable**, and the nearest
real thing - structure-preserving v2v that keeps choreography, camera and set
while the new character comes from the PROMPT - is not achievable on LTX-2.5
either, for want of a published adapter. That is a weights fact, not a code one.

#### 3 - what was built anyway, and why it is not speculative

`ltxv::refcond` ports the conditioning mechanism itself: reference-token
positions (`get_pixel_coords` re-expressed in the target frame, with the
`downscale_factor` spatial stretch and the `temporal_scale_factor` re-spacing +
translate + clamp), the frozen `1 - strength` denoise mask, the never-marked
keyframe mask, and `downsample_mask_video_to_latent`. This is the primitive any
IC-LoRA needs, whoever trains it, and it is pure geometry - so unlike the 22B
DiT it can be pinned down exactly on this box, which is the point.

* Gated by `tools/goldens/ltxv_refcond_dump_reference.py`, a **live run** of the
  official `VideoConditionByReferenceLatent`, its attention-strength wrapper and
  the real `iclora_utils` source - not a transcription. `torchaudio`'s native
  library is broken here, so the one unrelated import that drags it in is
  stubbed; the function bodies that produce the goldens are the reference's own.
* `crates/ltxv/tests/refcond_parity.rs` asserts **cosine AND rel_l2** on every
  tap. Cosine alone is scale-invariant and this path multiplies by
  `downscale_factor` and divides by `fps / S` - a wrong scale is the likeliest
  defect here and precisely the one cosine cannot see, the same trap
  `upsampler_parity.rs` already records.
* **The attention mask is stored factored.** The reference materialises a dense
  `(N+M)^2` matrix; the golden asserts it is exactly reconstructible from the
  `M`-vector of per-token weights plus `build_attention_mask`'s block structure,
  so this crate keeps `M` numbers. At 1280x704x121 the dense form would be
  hundreds of gigabytes for a mask that carries a few thousand distinct values.

#### 3b - the gate, mutation-verified

Every mutation below was applied to `refcond.rs`, the gate run, and the source
restored. The third row is the one that matters:

| mutation | caught by |
| --- | --- |
| `downscale_factor` spatial stretch dropped | cosine 0.999998545 (only just under the bound) |
| reference re-spaced at `fps` instead of `fps / S` | cosine 0.999947848 |
| **temporal translate's `clamp(min=0)` dropped** | **rel_l2 4.808e-4 ONLY - cosine was 0.999999884, i.e. ABOVE the 0.999999 bound, so a cosine-only gate would have passed this** |
| denoise mask `strength` instead of `1 - strength` | cosine 0.707106781 |
| mask temporal window off by one | cosine 0.999963354 |
| mask first latent frame taken from pixel frame 1, not 0 | cosine 0.997793604 |
| factored mask's reference rows flattened to 1.0 | cosine 0.933849528 |

The third row is this repo's cosine-only lesson reproduced on new code before
anything relied on the gate: the clamp only bites the first latent frame of a
temporally-scaled reference, so it moves a handful of the 288 values and barely
rotates the vector while changing its magnitude - exactly the defect class
cosine is blind to. One mutation attempt (`let _causal = ()` after the causal
copy) was discarded as a no-op rather than recorded as a gate weakness.

#### 4 - the script, and the pin

`examples/videogen/character_swap.sh` produces the two control signals a swap
needs - a Canny structure reference (ffmpeg `edgedetect`, the same signal
`compute_reference.py` builds) and a character-pin mask video - and then STOPS,
naming the missing adapter. It does not call a generation action that does not
exist.

The pin is where honesty cost something. A point on frame 0 propagated through
the clip needs video segmentation, and `crates/sam2` is the **image path only**
- its own module doc puts the memory bank, memory attention and temporal object
pointer explicitly out of scope. So the script takes a point PER KEYFRAME
(`PIN="640,300@0;700,320@48"`), segments each with the real `sam2 segment`
action, and holds each mask until the next keyframe. That is the nearest true
thing to "click the stuntman once"; a single point is correct only for a
locked-off shot.

#### 5 - recorded, NOT done

* **No `brain ltxv v2v` action.** With no LTX-2.5 control adapter to load it
  would be unrunnable and unvalidatable, so `refcond` is wired into no pipeline
  yet. The append helper takes the same `(base_t, base_positions,
  base_keyframes_mask, ...)` shape `append_image_conditioning` does, so the
  wiring is mechanical when an adapter exists.
* **No IC-LoRA TRAINING path.** `v2v_ic_lora.yaml` wants a `reference_latents/`
  dataset; the trainer side of that (reference latents in the loss, not just at
  inference) is untouched here.
* **The PIN branch was never run end to end** - `BRAIN_SAM2_WEIGHTS` is not on
  this box, so `sam2 segment` was not invoked. The ffmpeg branches and the
  keyframe-hold logic were both run and checked; the SAM 2.1 call is argued from
  its action spec (`sam2::caps`, `points` as `'x,y;…'` in source pixels, a
  `mask` blob out), not from a run.
* **Masked / inpainting conditioning is the better next route, and is also not
  built.** `ltx_core.conditioning.types.mask_cond.VideoConditionByMask` is real
  (unmasked positions pinned to the clean source latent, masked positions
  denoised normally) and, unlike IC-LoRA, is not architecturally dependent on an
  adapter having been trained to attend across appended tokens - so it is the
  one mechanism here that could replace a region of an EXISTING clip while
  keeping the rest of the frame bit-exact, rather than merely structurally
  similar. Two caveats found while checking it: NO shipped inference pipeline
  uses it (`grep VideoConditionByMask` across `ltx-pipelines` is empty; it is
  reachable only through the trainer's `video_inpainting_lora.yaml`), and the
  released in/out-painting adapter is again LTX-2.3 only
  (`LTX-2.3-22b-IC-LoRA-In-Outpainting`). brain has no masked conditioning at
  all. The mask this phase's script already produces is exactly its input.
* **No real-weight run of anything in this phase.** It is all geometry, gated
  against a live reference; the 22B DiT gap above is untouched and unaffected.
