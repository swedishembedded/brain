// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3 DiT core configuration - the real checkpoint's numbers
//! (`transformer/config.json`, cross-checked against `transformer_minimax_h3
//! .py`'s own `__init__` defaults, both read directly) plus a
//! [`H3TransformerConfig::tiny`] constructor for weight-free smoke tests and
//! the real-weight parity harness.
//!
//! Swedish Embedded AB implements this MiniMax-H3 DiT core port for its
//! clients. If your team needs expertise in porting large diffusion
//! transformers to new inference stacks, you can procure our services by
//! sending an email to info@swedishembedded.com.

/// `MINIMAX_H3_MODALITY_NUM` - every row of the packed sequence (including
/// TEXT rows) carries one of these three tags and gets its own AdaLN
/// modulation parameters per timestep (`transformer_minimax_h3.py`'s own
/// module-level constant).
pub const MODALITY_NUM: u32 = 3;
/// `token_tags` value for a video row.
pub const TAG_VIDEO: u32 = 0;
/// `token_tags` value for a text row.
pub const TAG_TEXT: u32 = 1;
/// `token_tags` value for an audio row.
pub const TAG_AUDIO: u32 = 2;

/// The H3 DiT core's shape/hyperparameter surface - field names and defaults
/// match `MiniMaxH3Transformer3DModel.__init__`'s own keyword arguments
/// exactly (`diffusers/models/transformers/transformer_minimax_h3.py`), so a
/// diff against the reference constructor call is a diff against this
/// struct's [`Default`] impl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H3TransformerConfig {
    pub num_attention_heads: u32,
    /// Per-head channel width. `num_attention_heads * attention_head_dim`
    /// (the QKV width, `inner_dim`) is DELIBERATELY LARGER than
    /// `hidden_size` in MiniMax-H3 (7168 vs 5376 at the real config) - not a
    /// misconfiguration, confirmed from the real checkpoint's own tensor
    /// shapes.
    pub attention_head_dim: u32,
    /// The packed sequence's residual-stream width.
    pub hidden_size: u32,
    pub num_layers: u32,
    /// Token refiner block count (diffusers: `num_refiner_layers`; the real
    /// checkpoint's own config key is `token_refiner_num_layers`).
    pub num_refiner_layers: u32,
    /// SwiGLU inner width (the checkpoint's own `ffn_hidden_size`).
    pub ffn_dim: u32,
    /// Video latent channel count (pre-patchify).
    pub in_channels: u32,
    pub audio_in_channels: u32,
    /// `(t, h, w)` patch used to pack video latents into rows.
    pub patch_size: [u32; 3],
    /// Text conditioning width (the Qwen3-VL hidden size H3 conditions on).
    pub text_dim: u32,
    /// Sinusoidal timestep embedding width (`time_proj`'s `num_channels`).
    pub freq_dim: u32,
    /// `time_embedder`'s inner (first-linear-output) width.
    pub time_embed_hidden_dim: u32,
    /// `time_embedder`'s output width - the input width of every AdaLN
    /// projection (`adaln_proj`, `norm_out.linear`).
    pub time_embed_dim: u32,
    /// Rotary frequencies PER AXIS. The `(t, h, w)` axes share one
    /// `inv_freq` buffer of this length; `2 * 3 * rope_freq_dim` of
    /// `attention_head_dim` channels get rotated, the rest pass through.
    pub rope_freq_dim: u32,
    pub rope_theta: f32,
    /// Epsilon of `norm1`/`norm2` (pre-attention/pre-FFN RMSNorm) and of the
    /// token refiner's own norms.
    pub norm_eps: f32,
    /// Epsilon of the per-head query/key RMSNorm.
    pub qk_norm_eps: f32,
    /// Epsilon of the token refiner's `final_norm` and of `norm_out`.
    pub final_norm_eps: f32,
}

impl Default for H3TransformerConfig {
    fn default() -> H3TransformerConfig {
        H3TransformerConfig::real()
    }
}

impl H3TransformerConfig {
    /// The real `MiniMaxAI/MiniMax-H3` `transformer/config.json` numbers -
    /// read directly from the checkpoint header and from
    /// `transformer_minimax_h3.py`'s own `__init__` defaults, not estimated.
    pub fn real() -> H3TransformerConfig {
        H3TransformerConfig {
            num_attention_heads: 56,
            attention_head_dim: 128,
            hidden_size: 5376,
            num_layers: 50,
            num_refiner_layers: 2,
            ffn_dim: 14336,
            in_channels: 24,
            audio_in_channels: 32,
            patch_size: [1, 2, 2],
            text_dim: 5120,
            freq_dim: 256,
            time_embed_hidden_dim: 5376,
            time_embed_dim: 2688,
            rope_freq_dim: 16,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    /// A small config for weight-free smoke tests and the real-weight
    /// numeric parity harness, at the REAL config's own proportions rather
    /// than arbitrary round numbers: `inner_dim (128) != hidden_size (20)`
    /// mirrors the real `7168 != 5376` asymmetry, and `rotary_dim (24) =
    /// 0.75 * attention_head_dim (32)` mirrors the real `96/128` rotated
    /// fraction exactly (`rope_freq_dim` scales down 16 -> 4, so `2*3*4=24`
    /// of `32` channels rotate, matching the real `2*3*16=96` of `128`).
    pub fn tiny() -> H3TransformerConfig {
        H3TransformerConfig {
            num_attention_heads: 4,
            attention_head_dim: 32,
            hidden_size: 20,
            num_layers: 2,
            num_refiner_layers: 1,
            ffn_dim: 64,
            in_channels: 3,
            audio_in_channels: 5,
            patch_size: [1, 2, 2],
            text_dim: 12,
            freq_dim: 16,
            time_embed_hidden_dim: 40,
            time_embed_dim: 24,
            rope_freq_dim: 4,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    /// The attention QKV width - `num_attention_heads * attention_head_dim`,
    /// larger than [`Self::hidden_size`] at the real config (see that
    /// field's own doc).
    pub fn inner_dim(&self) -> u32 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// `proj_in`'s input width - one patchified video latent row
    /// (`in_channels * prod(patch_size)`).
    pub fn video_patch_dim(&self) -> u32 {
        self.in_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
    }

    /// The number of `attention_head_dim` channels RoPE rotates per head -
    /// `2 * 3 * rope_freq_dim` (3 axes, each contributing `rope_freq_dim`
    /// angles, doubled by the `rotate_half` convention). The remaining
    /// `attention_head_dim - rotary_dim` channels pass through unrotated.
    pub fn rotary_dim(&self) -> u32 {
        2 * 3 * self.rope_freq_dim
    }

    /// `adaln_proj.linear`'s output width for ONE block -
    /// `6 * hidden_size * MODALITY_NUM` (the AdaLN-Zero sextet, times 3
    /// modalities). The real checkpoint's own `adaln_out_features`.
    pub fn adaln_out_features(&self) -> u32 {
        6 * self.hidden_size * MODALITY_NUM
    }

    /// `norm_out.linear`'s output width - `2 * hidden_size` (shift+scale
    /// only, shared across every modality at a given timestep - see
    /// `crate::model`'s own doc for why this is asymmetric with every other
    /// AdaLN site in the model). The real checkpoint's own
    /// `final_adaln_out_features`.
    pub fn final_adaln_out_features(&self) -> u32 {
        2 * self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_config_matches_the_checkpoint_header_numbers() {
        let cfg = H3TransformerConfig::real();
        assert_eq!(cfg.inner_dim(), 7168, "56*128");
        assert_ne!(cfg.inner_dim(), cfg.hidden_size, "QKV width != hidden_size at the real config, by design");
        assert_eq!(cfg.video_patch_dim(), 96, "24*1*2*2 - matches proj_in [5376,96]");
        assert_eq!(cfg.rotary_dim(), 96, "2*3*16 - matches the roadmap's confirmed 96/128 split");
        assert_eq!(cfg.adaln_out_features(), 96768, "6*5376*3, matches adaln_proj.linear.weight [96768,2688]");
        assert_eq!(cfg.final_adaln_out_features(), 10752, "2*5376, matches final_adaln_out_features");
    }

    #[test]
    fn tiny_config_preserves_the_real_proportions() {
        let cfg = H3TransformerConfig::tiny();
        assert_ne!(cfg.inner_dim(), cfg.hidden_size, "tiny config must also exercise inner_dim != hidden_size");
        assert!(cfg.rotary_dim() < cfg.attention_head_dim, "tiny config must also exercise the RoPE pass-through tail");
        assert_eq!(cfg.rotary_dim() * 4, cfg.attention_head_dim * 3, "24/32 == 96/128 == 0.75, the real rotated fraction");
    }
}
