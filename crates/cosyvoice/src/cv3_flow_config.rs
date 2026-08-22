// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CausalMaskedDiffWithDiT` (CosyVoice 3) flow decoder configuration: no
//! encoder at all (verified absent, not merely unused, by reading
//! `resources/cosyvoice/source/cosyvoice/flow/flow.py`'s
//! `CausalMaskedDiffWithDiT.__init__`/`.inference` end to end) - condition
//! assembly is a bare `PreLookaheadLayer` + `repeat_interleave`, and the CFM
//! estimator is a 22-layer adaLN-zero `DiT`, not `crate::flow_config`'s UNet.
//! Every number below is read verbatim from the real
//! `resources/cosyvoice/weights3/cosyvoice3.yaml`'s `flow:` block and
//! cross-checked against the real `flow.pt` tensor shapes (see
//! `crate::cv3_flow_import`).

/// The `DiT` estimator's dims (`decoder.estimator:` block).
#[derive(Clone, Debug)]
pub struct DitConfig {
    pub dim: u32,
    pub depth: u32,
    pub heads: u32,
    pub dim_head: u32,
    pub ff_mult: u32,
    /// `mel_dim` / `output_size` - both the noised-input and predicted-velocity
    /// channel count (80).
    pub mel_dim: u32,
    /// `mu_dim` - the token/lookahead ("text_embed") channel count fed to
    /// `InputEmbedding` (80, since `CausalMaskedDiffWithDiT` has no encoder to
    /// project up to a wider width the way CosyVoice 2's does).
    pub mu_dim: u32,
    /// `spk_dim` - the speaker-embedding channel count (80).
    pub spk_dim: u32,
    /// `TimestepEmbedding`'s `SinusPositionEmbedding` width (256, the F5-TTS
    /// default - distinct from `dim`).
    pub freq_embed_dim: u32,
    /// `CausalConvPositionEmbedding`'s kernel size (31) and group count (16).
    pub conv_pos_kernel: u32,
    pub conv_pos_groups: u32,
    /// `RotaryEmbedding`'s base (`10000`, `x_transformers` default).
    pub rope_theta: f32,
    /// `LayerNorm(elementwise_affine=False)`'s eps used throughout
    /// `AdaLayerNormZero`/`AdaLayerNormZero_Final`/`DiTBlock.ff_norm` (`1e-6`
    /// - NOT the `1e-5` CosyVoice 2's UNet uses).
    pub norm_eps: f32,
}

impl DitConfig {
    pub fn inner_dim(&self) -> u32 {
        self.heads * self.dim_head
    }

    /// `InputEmbedding.proj`'s input width: `mel_dim*2 + mu_dim + spk_dim`
    /// (`x`, `cond` - both `mel_dim`-wide - plus `text_embed`/`mu` and
    /// `spks`).
    pub fn input_embed_in(&self) -> u32 {
        self.mel_dim * 2 + self.mu_dim + self.spk_dim
    }

    pub fn ff_hidden(&self) -> u32 {
        self.dim * self.ff_mult
    }
}

/// The whole `CausalMaskedDiffWithDiT` configuration.
#[derive(Clone, Debug)]
pub struct Cv3FlowConfig {
    /// `input_size` - the `input_embedding` table's embedding width (80, NOT
    /// CosyVoice 2's 512 - the token embedding is never widened by an
    /// encoder here).
    pub input_size: u32,
    pub output_size: u32,
    pub spk_embed_dim: u32,
    pub vocab_size: u32,
    pub token_mel_ratio: u32,
    pub pre_lookahead_len: u32,
    /// `PreLookaheadLayer(in_channels=80, channels=1024,
    /// pre_lookahead_len=3)`'s hidden width.
    pub pre_lookahead_channels: u32,
    pub dit: DitConfig,
    pub inference_cfg_rate: f32,
    pub n_timesteps: u32,
    pub rand_noise_len: u32,
}

impl Cv3FlowConfig {
    /// The real `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` flow decoder configuration.
    pub fn cosyvoice3() -> Cv3FlowConfig {
        Cv3FlowConfig {
            input_size: 80,
            output_size: 80,
            spk_embed_dim: 192,
            vocab_size: 6561,
            token_mel_ratio: 2,
            pre_lookahead_len: 3,
            pre_lookahead_channels: 1024,
            dit: DitConfig {
                dim: 1024,
                depth: 22,
                heads: 16,
                dim_head: 64,
                ff_mult: 2,
                mel_dim: 80,
                mu_dim: 80,
                spk_dim: 80,
                freq_embed_dim: 256,
                conv_pos_kernel: 31,
                conv_pos_groups: 16,
                rope_theta: 10000.0,
                norm_eps: 1e-6,
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
    fn cosyvoice3_matches_the_real_yaml_and_checkpoint() {
        let cfg = Cv3FlowConfig::cosyvoice3();
        assert_eq!(cfg.input_size, 80, "CV3's input_embedding is 80-wide, not CV2's 512");
        assert_eq!(cfg.output_size, 80);
        assert_eq!(cfg.spk_embed_dim, 192);
        assert_eq!(cfg.vocab_size, 6561);
        assert_eq!(cfg.token_mel_ratio, 2);
        assert_eq!(cfg.pre_lookahead_len, 3);
        assert_eq!(cfg.pre_lookahead_channels, 1024);
        assert_eq!(cfg.dit.inner_dim(), 1024);
        assert_eq!(cfg.dit.input_embed_in(), 320);
        assert_eq!(cfg.dit.ff_hidden(), 2048);
        assert_eq!(cfg.rand_noise_len, 15000);
        assert_eq!(cfg.n_timesteps, 10);
        assert_eq!(cfg.inference_cfg_rate, 0.7);
    }
}
