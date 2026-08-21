// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CausalMaskedDiffWithXvec` (CosyVoice 2) flow decoder configuration: the
//! `UpsampleConformerEncoder` + `CausalConditionalDecoder` (UNet CFM
//! estimator) dims, read verbatim from `cosyvoice2.yaml`'s `flow:` block and
//! cross-checked against the real `flow.pt` tensor shapes (see
//! `crate::flow_import`).
//!
//! CosyVoice 3's `CausalMaskedDiffWithDiT` (no encoder, a DiT estimator
//! instead of this UNet) is a deliberate follow-up milestone, not implemented
//! here - see `resources/cosyvoice/source/cosyvoice/flow/flow.py`'s
//! `CausalMaskedDiffWithDiT` for the delta.

/// `UpsampleConformerEncoder` dims (`encoder:` block of `cosyvoice2.yaml`).
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub d_model: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub ff_dim: u32,
    pub num_blocks: u32,
    pub num_up_blocks: u32,
    /// `eps` for `ConformerEncoderLayer`'s own `norm_mha`/`norm_ff` (distinct
    /// from the embed/`after_norm`/`CausalBlock1D` LayerNorms below, which use
    /// `1e-5` - verified from the real source, not assumed uniform).
    pub layer_norm_eps: f32,
    /// `eps` for `LinearNoSubsampling`'s embed LayerNorm and `after_norm`.
    pub outer_norm_eps: f32,
    pub pre_lookahead_len: u32,
}

/// `CausalConditionalDecoder` (the UNet CFM estimator) dims (`decoder.estimator:`).
#[derive(Clone, Debug)]
pub struct EstimatorConfig {
    /// `x(80) + mu(80) + spks(80) + cond(80)`.
    pub in_channels: u32,
    pub mel_channels: u32,
    pub channels: u32,
    pub attention_head_dim: u32,
    pub num_heads: u32,
    /// Transformer blocks per down/mid/up stage (`n_blocks`).
    pub blocks_per_stage: u32,
    pub num_mid_stages: u32,
    /// `SinusoidalPosEmb`'s scale (`1000`, matcha's own default) and
    /// `TimestepEmbedding`'s hidden width (`channels * 4`).
    pub time_embed_dim: u32,
    /// `LayerNorm`/`GroupNorm`-free everywhere in the causal variant: every
    /// norm here is a plain `LayerNorm(eps=1e-5)` (`CausalBlock1D`,
    /// `BasicTransformerBlock.norm1/norm3`) - one constant, not two, unlike
    /// the encoder above.
    pub norm_eps: f32,
}

/// The whole `CausalMaskedDiffWithXvec` configuration.
#[derive(Clone, Debug)]
pub struct FlowConfig {
    pub input_size: u32,
    pub output_size: u32,
    pub spk_embed_dim: u32,
    pub vocab_size: u32,
    pub token_mel_ratio: u32,
    pub encoder: EncoderConfig,
    pub estimator: EstimatorConfig,
    /// `inference_cfg_rate` - the CFG weight `solve_euler` uses to combine the
    /// conditional/unconditional branches (`(1+rate)*cond - rate*uncond`).
    pub inference_cfg_rate: f32,
    /// Default Euler step count `flow.inference()` always passes (`n_timesteps=10`).
    pub n_timesteps: u32,
    /// Length of the fixed `CausalConditionalCFM.rand_noise` buffer
    /// (`50 * 300`, `[1, 80, 15000]`) - see `crate::flow`'s noise-asset doc.
    pub rand_noise_len: u32,
}

impl FlowConfig {
    /// The real `FunAudioLLM/CosyVoice2-0.5B` flow decoder configuration.
    pub fn cosyvoice2() -> FlowConfig {
        FlowConfig {
            input_size: 512,
            output_size: 80,
            spk_embed_dim: 192,
            vocab_size: 6561,
            token_mel_ratio: 2,
            encoder: EncoderConfig {
                d_model: 512,
                heads: 8,
                head_dim: 64,
                ff_dim: 2048,
                num_blocks: 6,
                num_up_blocks: 4,
                layer_norm_eps: 1e-12,
                outer_norm_eps: 1e-5,
                pre_lookahead_len: 3,
            },
            estimator: EstimatorConfig {
                in_channels: 320,
                mel_channels: 80,
                channels: 256,
                attention_head_dim: 64,
                num_heads: 8,
                blocks_per_stage: 4,
                num_mid_stages: 12,
                time_embed_dim: 1024,
                norm_eps: 1e-5,
            },
            inference_cfg_rate: 0.7,
            n_timesteps: 10,
            rand_noise_len: 50 * 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosyvoice2_matches_the_real_yaml_and_checkpoint() {
        let cfg = FlowConfig::cosyvoice2();
        assert_eq!(cfg.input_size, 512);
        assert_eq!(cfg.output_size, 80);
        assert_eq!(cfg.spk_embed_dim, 192);
        assert_eq!(cfg.vocab_size, 6561);
        assert_eq!(cfg.token_mel_ratio, 2);
        assert_eq!(cfg.encoder.heads * cfg.encoder.head_dim, cfg.encoder.d_model);
        assert_eq!(cfg.estimator.num_heads * cfg.estimator.attention_head_dim, 512);
        assert_eq!(cfg.estimator.in_channels, cfg.estimator.mel_channels * 4);
        assert_eq!(cfg.estimator.time_embed_dim, cfg.estimator.channels * 4);
        assert_eq!(cfg.rand_noise_len, 15000);
    }
}
