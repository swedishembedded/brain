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
//! the op sequence this crate implements, not a simplified one. This
//! milestone (M3) implements exactly ONE point in the flag matrix -
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
    /// `false` for the real LTX-2.5 config - the per-head `2*sigmoid(gate)`
    /// multiply is NOT implemented here; see [`LtxDitConfig::assert_supported`].
    pub apply_gated_attention: bool,
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

    /// Panics if this config is outside the ONE flag combination M3
    /// implements (see this module's doc). Every field asserted here is a
    /// field the block/model forward would otherwise silently compute a
    /// DIFFERENT (and wrong) op sequence for if it disagreed - not a
    /// cosmetic check.
    pub fn assert_supported(&self) {
        assert!(self.cross_attention_adaln, "ltxv M3 only implements cross_attention_adaln=true");
        assert!(!self.use_prompt_adaln_single, "ltxv M3 only implements use_prompt_adaln_single=false (static prompt_scale_shift_table, no timestep MLP)");
        assert!(self.use_middle_indices_grid, "ltxv M3 only implements use_middle_indices_grid=true (RoPE at patch midpoints)");
        assert!(!self.apply_gated_attention, "ltxv M3 does not implement the per-head 2*sigmoid(gate) attention multiply");
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
            connector_num_layers: 2,
            connector_num_attention_heads: 2,
            connector_attention_head_dim: 8,
            connector_num_learnable_registers: 4,
            connector_positional_embedding_max_pos: [64],
            connector_norm_output: true,
            caption_proj_before_connector: true,
        }
    }

    /// The real LTX-2.5 22B video-stream config, transcribed field-by-field
    /// from the GGUF's own embedded `config` KV (`AVTransformer3DModel`,
    /// range-read and parsed against the real 4349-tensor header - see
    /// `crate::dit::av_dit_tensor_manifest`'s doc). `apply_gated_attention:
    /// true` here means [`Self::assert_supported`] panics on this config -
    /// intentional: this milestone makes the real config REPRESENTABLE, not
    /// runnable end to end (gated attention's forward is a later milestone).
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
            connector_num_layers: 8,
            connector_num_attention_heads: 32,
            connector_attention_head_dim: 128,
            connector_num_learnable_registers: 128,
            connector_positional_embedding_max_pos: [4096],
            connector_norm_output: true,
            caption_proj_before_connector: true,
        }
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

    /// The real LTX-2.5 22B AV config - [`LtxDitConfig::ltx25_22b`] +
    /// [`LtxAudioDitConfig::ltx25`] + `av_ca_timestep_scale_multiplier:
    /// 1000.0`, confirmed against the real GGUF's embedded config KV (this
    /// port's roadmap ledger previously flagged this value as UNVERIFIED;
    /// range-reading the real header settles it).
    pub fn ltx25() -> LtxAvDitConfig {
        LtxAvDitConfig { video: LtxDitConfig::ltx25_22b(), audio: LtxAudioDitConfig::ltx25(), av_ca_timestep_scale_multiplier: 1000.0 }
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
    /// numbers. `apply_gated_attention: true` means `assert_supported` is
    /// deliberately NOT called here (see [`LtxDitConfig::ltx25_22b`]'s doc).
    #[test]
    fn ltx25_config_matches_the_real_checkpoint_header() {
        let v = LtxDitConfig::ltx25_22b();
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
}
