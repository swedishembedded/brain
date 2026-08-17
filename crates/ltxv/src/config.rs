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
}

impl LtxDitConfig {
    /// `inner_dim / num_heads`.
    pub fn head_dim(&self) -> u32 {
        assert_eq!(self.inner_dim % self.num_heads, 0, "inner_dim {} not a multiple of num_heads {}", self.inner_dim, self.num_heads);
        self.inner_dim / self.num_heads
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
        }
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
}
