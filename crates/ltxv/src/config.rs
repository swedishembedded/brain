// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The video-only DiT's configuration - every FLAG that changes the op
//! sequence, transcribed from `tools/goldens/ltxv_dit_dump_reference.py`'s
//! `TINY_CONFIG` (cross-checked against `manifest.json`'s `run.tiny_config`,
//! which this module treats as authoritative over any other transcription -
//! checkpoint/dump reality always wins over prose, per this port's own
//! porting playbook).
//!
//! Every real-LTX-2.5-config flag is set to its real value even at toy
//! dims, which is what makes a tiny-config parity test meaningful: it proves
//! the op sequence this crate implements, not a simplified one. This crate
//! implements exactly ONE point in the flag matrix -
//! `cross_attention_adaln: true`, `use_prompt_adaln_single: false`,
//! `use_middle_indices_grid: true`, `apply_gated_attention: false` - and
//! [`LtxDitConfig::assert_supported`] panics loudly if a future caller ever
//! constructs a config outside that point, rather than silently running the
//! wrong op sequence.

/// The video-only DiT's shape + op-sequence configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtxDitConfig {
    /// `num_attention_heads * attention_head_dim`.
    pub inner_dim: u32,
    pub num_heads: u32,
    pub num_layers: u32,
    /// Real VAE latent channel count (128 for the real checkpoint too).
    pub in_channels: u32,
    pub out_channels: u32,
    /// `== inner_dim` for this milestone (`caption_projection: None`, see
    /// the dumper's module doc for why - the incoming context is already at
    /// `inner_dim` width, no projection module exists inside the
    /// transformer for the real LTX-2.5 config either).
    pub cross_attention_dim: u32,
    /// `false` for the real LTX-2.5 config (the class default is `true`).
    pub ff_bias: bool,
    /// `true` for the real LTX-2.5 config (the class default is `false`) -
    /// gates the 9-row (vs. 6-row) adaLN table and the text-cross-attention
    /// AdaLN modulation path this crate implements.
    pub cross_attention_adaln: bool,
    /// `false` for the real LTX-2.5 config (the class default is `true`) -
    /// text K/V modulation is a static per-block table, not a timestep-MLP.
    /// This crate only implements `false`; see [`LtxDitConfig::assert_supported`].
    pub use_prompt_adaln_single: bool,
    pub use_keyframes_abs_pos_embedding: bool,
    pub norm_eps: f32,
    pub positional_embedding_theta: f64,
    /// `(frame, height, width)` RoPE position normalizers.
    pub positional_embedding_max_pos: [u32; 3],
    pub timestep_scale_multiplier: u32,
    /// `true` for the real config - RoPE is evaluated at the midpoint of
    /// each token's `[start, end)` patch bounds. This crate only implements
    /// `true`; see [`LtxDitConfig::assert_supported`].
    pub use_middle_indices_grid: bool,
    /// `true` for the real LTX-2.5 config - every `Attention` module (self-,
    /// text-cross-, and (on the AV config) both audio<->video cross-attention
    /// directions) gets a per-head `2*sigmoid(gate)` multiply on its
    /// attention CONTEXT (post `attn_apply`, pre `to_out.0`), gate computed
    /// from the module's OWN raw input (`to_gate_logits`, `[heads, q_dim]`+
    /// bias) - `ltx_core...attention.Attention.forward`/`ops.
    /// PytorchGatedAttention` (`resources/ltxv/source/packages/ltx-core/src/
    /// ltx_core/model/transformer/{attention,ops}.py`). ONE flag, shared by
    /// every module of BOTH streams (`transformer.py`'s
    /// `BasicAVTransformerBlock.__init__` passes `video.apply_gated_attention`
    /// to `attn1`/`attn2`/`audio_to_video_attn` and
    /// `audio.apply_gated_attention` to `audio_attn1`/`audio_attn2`/
    /// `video_to_audio_attn`, but `model.py`'s `LTXModel.__init__` derives
    /// BOTH `TransformerConfig`s from the SAME single `apply_gated_attention`
    /// constructor arg - no independent per-stream value exists in the
    /// reference).
    pub apply_gated_attention: bool,
    /// `true` for the real LTX-2.5 config - gates BOTH embeddings
    /// connectors' own self-attention (`attn1`) the SAME way, via the SAME
    /// `to_gate_logits` mechanism as [`Self::apply_gated_attention`] - but a
    /// SEPARATE config key in the reference
    /// (`Embeddings1DConnectorConfigurator`/
    /// `AudioEmbeddings1DConnectorConfigurator.from_metadata`, both reading
    /// `connector_apply_gated_attention`, `embeddings_connector.py`), not
    /// tied to the main DiT's own flag.
    pub connector_apply_gated_attention: bool,
    /// `video_embeddings_connector.transformer_1d_blocks` depth - real value
    /// 8. Read but not yet consumed by `forward()`: the connector's own
    /// forward pass is a later milestone (see `crate::dit::av_dit_tensor_
    /// manifest`'s doc); this field only makes the real checkpoint's tensor
    /// count representable.
    pub connector_num_layers: u32,
    /// `video_embeddings_connector`'s own attention head count - real value
    /// 32 (`connector_num_attention_heads`).
    pub connector_num_attention_heads: u32,
    /// `video_embeddings_connector`'s own per-head dim - real value 128
    /// (`connector_attention_head_dim`); `connector_num_attention_heads *
    /// connector_attention_head_dim` is the connector's own working width
    /// (4096 for the real checkpoint, same as `inner_dim` - not a structural
    /// requirement, just true of this checkpoint).
    pub connector_attention_head_dim: u32,
    /// `video_embeddings_connector.learnable_registers` row count - real
    /// value 128 (`connector_num_learnable_registers`).
    pub connector_num_learnable_registers: u32,
    /// Single-axis RoPE max-pos normalizer for the connector's own attention
    /// - `[max_pos]`, real value `[4096]` (`connector_positional_embedding_
    /// max_pos`).
    pub connector_positional_embedding_max_pos: [u32; 1],
    /// Whether the connector applies a final norm to its output - real value
    /// `true` (`connector_norm_output`).
    pub connector_norm_output: bool,
    /// Whether the caption projection runs before the connector consumes its
    /// output - real value `true` (`caption_proj_before_connector`).
    pub caption_proj_before_connector: bool,
    /// Whether [`crate::dit::LtxDit::forward`]/[`crate::dit::LtxAvDit::
    /// forward`] route the given raw `context` through
    /// `video_embeddings_connector`/`audio_embeddings_connector`
    /// ([`crate::connector`]) before the block stack consumes it - real
    /// value `true`. Unlike every other `connector_*` field (read but not
    /// consumed until this flag was added), this one gates the actual forward
    /// path: `false` (the existing tiny configs' setting) reproduces their
    /// original behavior exactly (`context` used as-is, no connector
    /// weights read). There is no equivalent reference field - in
    /// `ltx_core`, the connector is a standalone module the PIPELINE applies
    /// to the text encoder's output before ever calling `LTXModel.forward`
    /// (confirmed by `model_configurator.py`: neither `LTXModelConfigurator`
    /// nor `LTXVideoOnlyModelConfigurator` pass an embeddings-connector
    /// module into `LTXModel` at all), so this crate introduces the flag
    /// itself to make that same "does the DiT's own `context` input still
    /// need the connector applied" question representable in ONE place a
    /// caller can read, rather than pushing it onto every caller.
    pub use_embeddings_connector: bool,
}

impl LtxDitConfig {
    /// `inner_dim / num_heads`.
    pub fn head_dim(&self) -> u32 {
        assert_eq!(self.inner_dim % self.num_heads, 0, "inner_dim {} not a multiple of num_heads {}", self.inner_dim, self.num_heads);
        self.inner_dim / self.num_heads
    }

    /// `connector_num_attention_heads * connector_attention_head_dim` -
    /// `video_embeddings_connector`'s own working width.
    pub fn connector_inner_dim(&self) -> u32 {
        self.connector_num_attention_heads * self.connector_attention_head_dim
    }

    /// Rows of the per-block `scale_shift_table` / the adaLN-single raw
    /// output: `ADALN_NUM_BASE_PARAMS(6) + (3 if cross_attention_adaln)`
    /// (`ltx_core.model.transformer.adaln.adaln_embedding_coefficient`).
    pub fn adaln_rows(&self) -> u32 {
        6 + if self.cross_attention_adaln { 3 } else { 0 }
    }

    /// Panics if this config is outside what this crate's forward
    /// implements (see this module's doc). Every field asserted here is a
    /// field the block/model forward would otherwise silently compute a
    /// DIFFERENT (and wrong) op sequence for if it disagreed - not a
    /// cosmetic check. `apply_gated_attention` is deliberately NOT asserted
    /// here (either value is now supported, see [`crate::block::attention`]'s
    /// doc) - the earlier panic on `true` is gone now that the per-head
    /// `2*sigmoid(gate)` multiply is implemented.
    pub fn assert_supported(&self) {
        assert!(self.cross_attention_adaln, "ltxv M3 only implements cross_attention_adaln=true");
        assert!(!self.use_prompt_adaln_single, "ltxv M3 only implements use_prompt_adaln_single=false (static prompt_scale_shift_table, no timestep MLP)");
        assert!(self.use_middle_indices_grid, "ltxv M3 only implements use_middle_indices_grid=true (RoPE at patch midpoints)");
    }

    /// `tools/goldens/ltxv_dit_dump_reference.py`'s `TINY_CONFIG` - 2 layers,
    /// `inner_dim` 64 (4 heads x 16), every flag at its real-LTX-2.5 value.
    /// Cross-checked field by field against `testdata/golden/ltxv/dit/
    /// manifest.json`'s `run.tiny_config`.
    pub fn tiny() -> LtxDitConfig {
        LtxDitConfig {
            inner_dim: 64,
            num_heads: 4,
            num_layers: 2,
            in_channels: 128,
            out_channels: 128,
            cross_attention_dim: 64,
            ff_bias: false,
            cross_attention_adaln: true,
            use_prompt_adaln_single: false,
            use_keyframes_abs_pos_embedding: true,
            norm_eps: 1e-6,
            positional_embedding_theta: 10000.0,
            positional_embedding_max_pos: [20, 2048, 2048],
            timestep_scale_multiplier: 1000,
            use_middle_indices_grid: true,
            apply_gated_attention: false,
            connector_apply_gated_attention: false,
            connector_num_layers: 2,
            connector_num_attention_heads: 2,
            connector_attention_head_dim: 8,
            connector_num_learnable_registers: 4,
            connector_positional_embedding_max_pos: [64],
            connector_norm_output: true,
            caption_proj_before_connector: true,
            use_embeddings_connector: false,
        }
    }

    /// `tools/goldens/ltxv_dit_dump_reference.py`'s `TINY_GATED_CONFIG` -
    /// [`Self::tiny`] with `apply_gated_attention`/
    /// `connector_apply_gated_attention`/`use_embeddings_connector` all
    /// turned on (the real-LTX-2.5 values [`Self::tiny`] deliberately leaves
    /// off), at dims chosen so every axis differs from every other (lesson
    /// #4: `heads=3`/`head_dim=8` main attention vs `connector_heads=4`/
    /// `connector_head_dim=6` connector attention - both factor the SAME
    /// `inner_dim=24` differently, so a transpose between the two head
    /// splits cannot hide).
    pub fn tiny_gated() -> LtxDitConfig {
        LtxDitConfig {
            inner_dim: 24,
            num_heads: 3,
            num_layers: 2,
            in_channels: 128,
            out_channels: 128,
            cross_attention_dim: 24,
            ff_bias: false,
            cross_attention_adaln: true,
            use_prompt_adaln_single: false,
            use_keyframes_abs_pos_embedding: true,
            norm_eps: 1e-6,
            positional_embedding_theta: 10000.0,
            positional_embedding_max_pos: [20, 64, 96],
            timestep_scale_multiplier: 1000,
            use_middle_indices_grid: true,
            apply_gated_attention: true,
            connector_apply_gated_attention: true,
            connector_num_layers: 2,
            connector_num_attention_heads: 4,
            connector_attention_head_dim: 6,
            connector_num_learnable_registers: 3,
            connector_positional_embedding_max_pos: [50],
            connector_norm_output: true,
            caption_proj_before_connector: true,
            use_embeddings_connector: true,
        }
    }

    /// The real LTX-2.5 22B video-stream config, transcribed field-by-field
    /// from the GGUF's own embedded `config` KV (`AVTransformer3DModel`,
    /// range-read and parsed against the real 4349-tensor header - see
    /// `crate::dit::av_dit_tensor_manifest`'s doc). `apply_gated_attention`/
    /// `connector_apply_gated_attention`/`use_embeddings_connector` are all
    /// `true` and, as of the milestone that implemented the per-head
    /// `2*sigmoid(gate)` multiply and both embeddings connectors' forward
    /// pass, [`Self::assert_supported`] no longer rejects this config.
    pub fn ltx25_22b() -> LtxDitConfig {
        LtxDitConfig {
            inner_dim: 4096,
            num_heads: 32,
            num_layers: 48,
            in_channels: 128,
            out_channels: 128,
            cross_attention_dim: 4096,
            ff_bias: false,
            cross_attention_adaln: true,
            use_prompt_adaln_single: false,
            use_keyframes_abs_pos_embedding: true,
            norm_eps: 1e-6,
            positional_embedding_theta: 10000.0,
            positional_embedding_max_pos: [20, 2048, 2048],
            timestep_scale_multiplier: 1000,
            use_middle_indices_grid: true,
            apply_gated_attention: true,
            connector_apply_gated_attention: true,
            connector_num_layers: 8,
            connector_num_attention_heads: 32,
            connector_attention_head_dim: 128,
            connector_num_learnable_registers: 128,
            connector_positional_embedding_max_pos: [4096],
            connector_norm_output: true,
            caption_proj_before_connector: true,
            use_embeddings_connector: true,
        }
    }

    /// The real **LTX-2.3** 22B video-stream config - [`Self::ltx25_22b`]
    /// with exactly TWO fields changed, and nothing else.
    ///
    /// LTX-2.3, 2.4 and 2.5 are the same `AVTransformer3DModel` class behind
    /// the same `ltxv` GGUF architecture tag, so "the release is a config" is
    /// a claim this crate can check rather than assume - and it holds. Both
    /// releases' 22B checkpoint headers were range-read (headers only, never
    /// weights: the safetensors 8-byte little-endian JSON length followed by
    /// the JSON) and diffed tensor by tensor:
    ///
    /// * 2.5 carries `keyframes_abs_pos_embedding` (`[1, 4096]`); 2.3 does
    ///   NOT -> [`Self::use_keyframes_abs_pos_embedding`] is `false` here.
    /// * 2.3 carries 96 video-FFN bias tensors 2.5 does not
    ///   (`transformer_blocks.{0..47}.ff.net.0.proj.bias` and
    ///   `.ff.net.2.bias`) -> [`Self::ff_bias`] is `true` here.
    /// * Every other tensor name is shared, with ZERO shape mismatches
    ///   across all 4348 of them: 4349 (2.5) - 1 + 96 = 4444 (2.3).
    ///
    /// Neither flag is present in 2.3's own embedded `config` KV, so the
    /// tensor list is what settles them - and the reference agrees
    /// independently: `LTXAudioVideoModelConfigurator.from_metadata` reads
    /// `ff_bias=config.get("ff_bias", True)` and
    /// `use_keyframes_abs_pos_embedding=config.get(
    /// "use_keyframes_abs_pos_embedding", False)` (`ltx_core...transformer.
    /// model_configurator`), whose absent-key defaults are exactly the two
    /// values the 2.3 header shows. Two independent authorities, same answer.
    ///
    /// Every OTHER field is transcribed from 2.3's own `config` KV and is
    /// value-for-value identical to 2.5's, which is why this constructor is
    /// written as a struct update of [`Self::ltx25_22b`] rather than a second
    /// copy of 26 numbers that could drift.
    ///
    /// **Untested against real LTX-2.3 weights.** No LTX-2.3 checkpoint has
    /// ever been downloaded or forwarded by this crate, at any size. What IS
    /// proven: header coverage (the manifest this config generates matches
    /// the real 4444-tensor header name for name and shape for shape, see
    /// `crate::import`'s own tests) and the op sequence, which is shared with
    /// LTX-2.5 line for line and tiny-config parity-gated there. What is NOT:
    /// any number produced from real 2.3 weights.
    ///
    /// Two consequences worth knowing at this call site rather than
    /// somewhere else:
    ///
    /// * `ff_bias: true` puts a bias on the VIDEO stream's FFN for the first
    ///   time in this crate. The biased-FFN machinery is well exercised
    ///   (`audio_ff` and both embeddings connectors are biased on every real
    ///   LTX-2.5 run, same code), but the video stream reaching it is new -
    ///   every golden and every checkpoint loaded so far is `ff_bias: false`.
    /// * A 2.3 checkpoint still cannot be RUN end to end, for a reason
    ///   outside this crate: LTX-2.3's text encoder is Gemma 3 12B, not
    ///   Gemma-4, and `crates/gemma4` has no Gemma-3 path. The projection
    ///   geometry is identical (`188160 = 3840 * 49` either way), so the DiT
    ///   side is ready and the encoder is not. Read this config as
    ///   "2.3 is loadable and forwardable", never "2.3 is runnable".
    pub fn ltx23_22b() -> LtxDitConfig {
        LtxDitConfig { ff_bias: true, use_keyframes_abs_pos_embedding: false, ..LtxDitConfig::ltx25_22b() }
    }
}

/// The audio stream's shape configuration - the audio-side counterpart of
/// [`LtxDitConfig`], narrower per-head-dim than video (real LTX-2.5: 64 vs
/// video's 128) but the SAME head COUNT (real config: 32 both streams) -
/// that equality is what keeps the shared cross-modal RoPE table's per-head
/// split consistent regardless of which stream's preprocessor built it, see
/// [`LtxAvDitConfig::assert_supported`] and `crate::rope`'s doc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtxAudioDitConfig {
    pub inner_dim: u32,
    pub num_heads: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    /// Doubles as (a) audio's own text-cross-attention context width
    /// (`audio_attn2.context_dim`) and (b) the shared cross-modal (A2V/V2A)
    /// RoPE table + attention geometry width (`ltx_core...model.LTXModel`'s
    /// single `audio_cross_attention_dim` field feeds both uses) - asserted
    /// `== inner_dim` in [`LtxAvDitConfig::assert_supported`], the same
    /// "caption already at inner_dim, no projection module" judgment call
    /// `config.rs`'s module doc records for video's own `cross_attention_dim`
    /// (true of the real LTX-2.5 checkpoint's 2048==2048, not a structural
    /// requirement of the reference class).
    pub cross_attention_dim: u32,
    pub ff_bias: bool,
    /// Single-axis (time only) RoPE max-pos normalizer - `[max_pos]`, class
    /// default `[20]`.
    pub positional_embedding_max_pos: [u32; 1],
    /// `audio_embeddings_connector`'s own attention head count - real value
    /// 32 (`audio_connector_num_attention_heads`). See [`LtxDitConfig::
    /// connector_num_attention_heads`]'s doc for the "read, not yet
    /// consumed" caveat, which applies identically here.
    pub connector_num_attention_heads: u32,
    /// `audio_embeddings_connector`'s own per-head dim - real value 64
    /// (`audio_connector_attention_head_dim`).
    pub connector_attention_head_dim: u32,
}

impl LtxAudioDitConfig {
    /// `inner_dim / num_heads`.
    pub fn head_dim(&self) -> u32 {
        assert_eq!(self.inner_dim % self.num_heads, 0, "audio inner_dim {} not a multiple of num_heads {}", self.inner_dim, self.num_heads);
        self.inner_dim / self.num_heads
    }

    /// `connector_num_attention_heads * connector_attention_head_dim` -
    /// `audio_embeddings_connector`'s own working width.
    pub fn connector_inner_dim(&self) -> u32 {
        self.connector_num_attention_heads * self.connector_attention_head_dim
    }

    /// `tools/goldens/ltxv_av_dit_dump_reference.py`'s `TINY_CONFIG` audio
    /// half - `inner_dim` 32 (4 heads x 8), proportionally narrower than the
    /// video tiny config's 64 (half), same head COUNT (4) as video.
    pub fn tiny() -> LtxAudioDitConfig {
        LtxAudioDitConfig {
            inner_dim: 32,
            num_heads: 4,
            in_channels: 128,
            out_channels: 128,
            cross_attention_dim: 32,
            ff_bias: false,
            positional_embedding_max_pos: [20],
            connector_num_attention_heads: 2,
            connector_attention_head_dim: 4,
        }
    }

    /// `tools/goldens/ltxv_av_dit_dump_reference.py`'s `TINY_GATED_CONFIG`
    /// audio half - `inner_dim` 12 (3 heads x 4, SAME head COUNT as
    /// [`LtxDitConfig::tiny_gated`]'s video half per [`LtxAvDitConfig::
    /// assert_supported`]'s invariant - `head_dim` must stay EVEN and
    /// `inner_dim/2` a multiple of `num_heads`, the RoPE split's own
    /// divisibility requirement, `crate::rope::ltx_rope_tables`'s doc),
    /// connector factored as `2 heads x 6` (different factorization of the
    /// SAME 12 vs. main attention's `3 x 4` - lesson #4, catches a
    /// heads/head_dim transpose between the two attention geometries).
    pub fn tiny_gated() -> LtxAudioDitConfig {
        LtxAudioDitConfig {
            inner_dim: 12,
            num_heads: 3,
            in_channels: 128,
            out_channels: 128,
            cross_attention_dim: 12,
            ff_bias: false,
            positional_embedding_max_pos: [20],
            connector_num_attention_heads: 2,
            connector_attention_head_dim: 6,
        }
    }

    /// The real LTX-2.5 22B audio-stream config - `audio_num_attention_heads:
    /// 32`, `audio_attention_head_dim: 64` (`inner_dim` 2048),
    /// `audio_cross_attention_dim: 2048` (`== inner_dim`, matching
    /// [`LtxAvDitConfig::assert_supported`]'s invariant),
    /// `audio_positional_embedding_max_pos: [20]`. See [`LtxDitConfig::
    /// ltx25_22b`]'s doc for provenance.
    pub fn ltx25() -> LtxAudioDitConfig {
        LtxAudioDitConfig {
            inner_dim: 2048,
            num_heads: 32,
            in_channels: 128,
            out_channels: 128,
            cross_attention_dim: 2048,
            ff_bias: false,
            positional_embedding_max_pos: [20],
            connector_num_attention_heads: 32,
            connector_attention_head_dim: 64,
        }
    }

    /// The real **LTX-2.3** 22B audio-stream config - byte-for-byte
    /// [`Self::ltx25`] except [`Self::ff_bias`], which follows the video
    /// stream's (see [`LtxDitConfig::ltx23_22b`]).
    ///
    /// The audio FFN's bias tensors (`transformer_blocks.N.audio_ff.net.0.
    /// proj.bias` / `.net.2.bias`) are present in BOTH releases' headers, so
    /// this field does not describe them and the manifest does not read it -
    /// `crate::dit::push_ff`'s doc records that audio's FFN bias is a
    /// per-instance fact taken off the tensor map, never off a config flag.
    /// The reference does expose a separate `audio_ff_bias` key
    /// (`model_configurator.py`, default `True`), but NEITHER release's
    /// `config` KV sets it, so there is no checkpoint value to transcribe;
    /// mirroring the video stream keeps the one flag this struct carries
    /// consistent with the release it names instead of inventing a third
    /// answer.
    pub fn ltx23() -> LtxAudioDitConfig {
        LtxAudioDitConfig { ff_bias: true, ..LtxAudioDitConfig::ltx25() }
    }
}

/// The bundled audio<->video DiT configuration - `LTXModelType::AudioVideo`.
/// Video and audio each keep their OWN adaLN-single conditioning (own
/// `scale_shift_table`, own timestep MLP), own self-/text-cross-attention,
/// coupled every block by bidirectional cross-attention (`crate::block`'s
/// doc has the exact op order and adaLN table layout).
///
/// This milestone implements exactly the ONE real-config point the
/// reference's own `LTXModelConfigurator` asserts every real checkpoint
/// satisfies (`check_config_value(config, "use_audio_video_cross_attention",
/// True)`, `check_config_value(config, "av_cross_ada_norm", True)`) - the
/// reference `LTXModel` class does not even expose these as constructor
/// knobs, only as configurator-side asserts, so unlike [`LtxDitConfig::
/// assert_supported`]'s per-field panics there is no "off" path to reject
/// here; [`LtxAvDitConfig::assert_supported`] instead pins the cross-stream
/// geometry invariants this milestone's op sequence actually depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtxAvDitConfig {
    pub video: LtxDitConfig,
    pub audio: LtxAudioDitConfig,
    /// `ltx_core...model.LTXModel.av_ca_timestep_scale_multiplier` - scales
    /// the CROSS modality's scalar sigma feeding the A2V/V2A gate MLP (see
    /// `crate::block`'s doc). Real checkpoint value UNVERIFIED (this port's
    /// roadmap ledger: metadata reportedly carries `1000.0` vs. the
    /// reference class's own default `1` - not confirmed empirically), so
    /// this crate takes it as a plain config field rather than hardcoding
    /// either number.
    pub av_ca_timestep_scale_multiplier: f32,
}

impl LtxAvDitConfig {
    /// `max(video.positional_embedding_max_pos[0],
    /// audio.positional_embedding_max_pos[0])` -
    /// `ltx_core...model.LTXModel.__init__`'s `cross_pe_max_pos`, the
    /// normalizer for the SHARED cross-modal (A2V/V2A) time-only RoPE table.
    pub fn cross_pe_max_pos(&self) -> u32 {
        self.video.positional_embedding_max_pos[0].max(self.audio.positional_embedding_max_pos[0])
    }

    /// Panics if this config is outside what this milestone's op sequence
    /// implements: every [`LtxDitConfig::assert_supported`] video invariant,
    /// plus the two AV-specific geometry invariants the block/RoPE forward
    /// would otherwise silently compute the WRONG cross-modal shapes for -
    /// see [`LtxAudioDitConfig::cross_attention_dim`]'s doc and this
    /// struct's doc for why each one matters.
    pub fn assert_supported(&self) {
        self.video.assert_supported();
        assert_eq!(self.audio.cross_attention_dim, self.audio.inner_dim, "ltxv AV milestone assumes audio.cross_attention_dim == audio.inner_dim (see LtxAudioDitConfig::cross_attention_dim's doc)");
        assert_eq!(
            self.audio.num_heads, self.video.num_heads,
            "ltxv AV milestone requires equal head COUNT across streams (see LtxAudioDitConfig's doc) - only per-head dim differs, matching the real LTX-2.5 config's own 32/32 split"
        );
    }

    /// `tools/goldens/ltxv_av_dit_dump_reference.py`'s `TINY_CONFIG` - video
    /// half identical to [`LtxDitConfig::tiny`], audio half
    /// [`LtxAudioDitConfig::tiny`], every AV flag at its real-LTX-2.5
    /// structural value (see this struct's doc); `av_ca_timestep_scale_
    /// multiplier` picked as a non-1 value (not the unverified real number,
    /// see that field's doc) so a parity test that hardcodes the class
    /// default `1` instead of reading the config would fail loudly.
    pub fn tiny() -> LtxAvDitConfig {
        LtxAvDitConfig { video: LtxDitConfig::tiny(), audio: LtxAudioDitConfig::tiny(), av_ca_timestep_scale_multiplier: 3.0 }
    }

    /// `tools/goldens/ltxv_av_dit_dump_reference.py`'s `TINY_GATED_CONFIG` -
    /// [`LtxDitConfig::tiny_gated`] + [`LtxAudioDitConfig::tiny_gated`],
    /// `av_ca_timestep_scale_multiplier` a THIRD distinct non-1 value (not
    /// [`Self::tiny`]'s `3.0`) so the two golden configs cannot be confused.
    pub fn tiny_gated() -> LtxAvDitConfig {
        LtxAvDitConfig { video: LtxDitConfig::tiny_gated(), audio: LtxAudioDitConfig::tiny_gated(), av_ca_timestep_scale_multiplier: 5.0 }
    }

    /// The real LTX-2.5 22B AV config - [`LtxDitConfig::ltx25_22b`] +
    /// [`LtxAudioDitConfig::ltx25`] + `av_ca_timestep_scale_multiplier:
    /// 1000.0`, confirmed against the real GGUF's embedded config KV (this
    /// port's roadmap ledger previously flagged this value as UNVERIFIED;
    /// range-reading the real header settles it).
    pub fn ltx25() -> LtxAvDitConfig {
        LtxAvDitConfig { video: LtxDitConfig::ltx25_22b(), audio: LtxAudioDitConfig::ltx25(), av_ca_timestep_scale_multiplier: 1000.0 }
    }

    /// The real **LTX-2.3** 22B AV config - [`LtxDitConfig::ltx23_22b`] +
    /// [`LtxAudioDitConfig::ltx23`], same `av_ca_timestep_scale_multiplier:
    /// 1000.0` as 2.5 (both releases' `config` KV carry that number
    /// explicitly and identically).
    ///
    /// Nothing in this crate hardcodes it: [`crate::import::
    /// av_dit_config_from_kv`] derives the config from whichever checkpoint
    /// is actually loaded, so 2.3 vs 2.5 selection is a property of the FILE,
    /// not of a caller-chosen enum. This constructor exists so tests, the
    /// FLOPs model and the shape ledger can name 2.3's real widths without a
    /// checkpoint on disk - the same role [`Self::ltx25`] already plays.
    pub fn ltx23() -> LtxAvDitConfig {
        LtxAvDitConfig { video: LtxDitConfig::ltx23_22b(), audio: LtxAudioDitConfig::ltx23(), av_ca_timestep_scale_multiplier: 1000.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_config_matches_the_golden_manifest() {
        let c = LtxDitConfig::tiny();
        c.assert_supported();
        assert_eq!(c.head_dim(), 16);
        assert_eq!(c.adaln_rows(), 9);
    }

    #[test]
    #[should_panic(expected = "cross_attention_adaln")]
    fn assert_supported_rejects_unimplemented_flags() {
        let mut c = LtxDitConfig::tiny();
        c.cross_attention_adaln = false;
        c.assert_supported();
    }

    #[test]
    fn av_tiny_config_matches_the_golden_manifest() {
        let c = LtxAvDitConfig::tiny();
        c.assert_supported();
        assert_eq!(c.audio.head_dim(), 8);
        assert_eq!(c.cross_pe_max_pos(), 20);
    }

    #[test]
    fn tiny_gated_config_is_supported_and_distinct_from_tiny() {
        let c = LtxDitConfig::tiny_gated();
        c.assert_supported();
        assert_eq!(c.head_dim(), 8);
        assert_eq!(c.connector_inner_dim(), 24);
        assert!(c.apply_gated_attention);
        assert!(c.connector_apply_gated_attention);
        assert!(c.use_embeddings_connector);
        assert_ne!(c.inner_dim, LtxDitConfig::tiny().inner_dim);
    }

    #[test]
    fn av_tiny_gated_config_is_supported() {
        let c = LtxAvDitConfig::tiny_gated();
        c.assert_supported();
        assert_eq!(c.audio.head_dim(), 4);
        assert_eq!(c.audio.connector_inner_dim(), 12);
        assert_ne!(c.av_ca_timestep_scale_multiplier, LtxAvDitConfig::tiny().av_ca_timestep_scale_multiplier);
    }

    #[test]
    #[should_panic(expected = "equal head COUNT")]
    fn av_assert_supported_rejects_mismatched_head_counts() {
        let mut c = LtxAvDitConfig::tiny();
        c.audio.num_heads = 2;
        c.assert_supported();
    }

    /// Pins every real-LTX-2.5-22B field to the value transcribed from the
    /// GGUF's own embedded `config` KV (this module's doc / `crate::dit::
    /// av_dit_tensor_manifest`'s doc) - guards against a future GGUF-KV-
    /// parsing constructor silently drifting from these checkpoint-derived
    /// numbers. `assert_supported` IS called here (unlike before gated
    /// attention/the connectors had a forward pass) - this config is now
    /// fully supported, not merely representable.
    #[test]
    fn ltx25_config_matches_the_real_checkpoint_header() {
        let v = LtxDitConfig::ltx25_22b();
        v.assert_supported();
        assert_eq!(v.inner_dim, 4096);
        assert_eq!(v.num_heads, 32);
        assert_eq!(v.head_dim(), 128);
        assert_eq!(v.num_layers, 48);
        assert_eq!(v.in_channels, 128);
        assert_eq!(v.out_channels, 128);
        assert_eq!(v.cross_attention_dim, 4096);
        assert!(!v.ff_bias);
        assert!(v.cross_attention_adaln);
        assert!(!v.use_prompt_adaln_single);
        assert!(v.use_keyframes_abs_pos_embedding);
        assert_eq!(v.norm_eps, 1e-6);
        assert_eq!(v.positional_embedding_theta, 10000.0);
        assert_eq!(v.positional_embedding_max_pos, [20, 2048, 2048]);
        assert_eq!(v.timestep_scale_multiplier, 1000);
        assert!(v.use_middle_indices_grid);
        assert!(v.apply_gated_attention);
        assert_eq!(v.connector_num_layers, 8);
        assert_eq!(v.connector_num_attention_heads, 32);
        assert_eq!(v.connector_attention_head_dim, 128);
        assert_eq!(v.connector_inner_dim(), 4096);
        assert_eq!(v.connector_num_learnable_registers, 128);
        assert_eq!(v.connector_positional_embedding_max_pos, [4096]);
        assert!(v.connector_norm_output);
        assert!(v.caption_proj_before_connector);
        assert!(v.connector_apply_gated_attention);
        assert!(v.use_embeddings_connector);

        let a = LtxAudioDitConfig::ltx25();
        assert_eq!(a.inner_dim, 2048);
        assert_eq!(a.num_heads, 32);
        assert_eq!(a.head_dim(), 64);
        assert_eq!(a.in_channels, 128);
        assert_eq!(a.out_channels, 128);
        assert_eq!(a.cross_attention_dim, 2048);
        assert!(!a.ff_bias);
        assert_eq!(a.positional_embedding_max_pos, [20]);
        assert_eq!(a.connector_num_attention_heads, 32);
        assert_eq!(a.connector_attention_head_dim, 64);
        assert_eq!(a.connector_inner_dim(), 2048);

        let av = LtxAvDitConfig::ltx25();
        assert_eq!(av.video, v);
        assert_eq!(av.audio, a);
        assert_eq!(av.av_ca_timestep_scale_multiplier, 1000.0);
        assert_eq!(av.cross_pe_max_pos(), 20);
    }

    /// The real LTX-2.3 22B config, at REAL widths, against the real
    /// checkpoint header - the shape counterpart of
    /// [`ltx25_config_matches_the_real_checkpoint_header`] above.
    ///
    /// Both releases' 22B safetensors headers and the LTX-2.3 GGUF header
    /// were range-read (metadata only) and diffed tensor by tensor. This
    /// test pins the result: LTX-2.3 is LTX-2.5 with TWO fields changed and
    /// **nothing else**. The "nothing else" half is what a struct-update
    /// constructor could quietly lose - so rather than restating 26 numbers
    /// (which would drift with `ltx25_22b` and prove nothing), it undoes
    /// exactly the two documented changes and demands the result be equal to
    /// the 2.5 config by `PartialEq` over EVERY field. A third divergence
    /// creeping into either constructor fails here.
    #[test]
    fn ltx23_is_ltx25_with_exactly_two_flags_changed() {
        let v23 = LtxDitConfig::ltx23_22b();
        let v25 = LtxDitConfig::ltx25_22b();

        // The two that really differ, in the direction the headers show.
        assert!(v23.ff_bias, "LTX-2.3 carries transformer_blocks.N.ff.net.*.bias");
        assert!(!v25.ff_bias, "LTX-2.5 does not");
        assert!(!v23.use_keyframes_abs_pos_embedding, "LTX-2.3 has no keyframes_abs_pos_embedding tensor");
        assert!(v25.use_keyframes_abs_pos_embedding, "LTX-2.5 has one");

        // ...and nothing else does.
        let undone = LtxDitConfig { ff_bias: false, use_keyframes_abs_pos_embedding: true, ..v23 };
        assert_eq!(undone, v25, "LTX-2.3 and LTX-2.5 must differ in exactly ff_bias + use_keyframes_abs_pos_embedding");

        // Real widths, spelled out rather than inherited, so a refactor of
        // `ltx25_22b` that changed a width would fail here too and not just
        // silently carry into 2.3.
        assert_eq!(v23.inner_dim, 4096);
        assert_eq!(v23.num_heads, 32);
        assert_eq!(v23.head_dim(), 128);
        assert_eq!(v23.num_layers, 48);
        assert_eq!(v23.cross_attention_dim, 4096);
        assert_eq!(v23.connector_inner_dim(), 4096);
        assert_eq!(v23.adaln_rows(), 9);
        v23.assert_supported();

        // Audio stream: same two-flag story, audio-side.
        let a23 = LtxAudioDitConfig::ltx23();
        assert!(a23.ff_bias);
        assert_eq!(LtxAudioDitConfig { ff_bias: false, ..a23 }, LtxAudioDitConfig::ltx25());
        assert_eq!(a23.inner_dim, 2048);
        assert_eq!(a23.head_dim(), 64);
        assert_eq!(a23.connector_inner_dim(), 2048);

        let av23 = LtxAvDitConfig::ltx23();
        av23.assert_supported();
        assert_eq!(av23.video, v23);
        assert_eq!(av23.audio, a23);
        assert_eq!(av23.av_ca_timestep_scale_multiplier, LtxAvDitConfig::ltx25().av_ca_timestep_scale_multiplier);
        assert_eq!(av23.cross_pe_max_pos(), 20);
        assert_ne!(av23, LtxAvDitConfig::ltx25());
    }
}
