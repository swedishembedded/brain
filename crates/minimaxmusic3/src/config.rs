// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Configs for MiniMax Music 3's four brain-native components (the fifth,
//! the Global LLM, is a real Qwen3-8B and reuses [`qwen3::QwenConfig`]
//! directly - no local config type for it).
//!
//! Every `::tiny()` here is mutually consistent: `ConditionEncoderConfig`'s
//! `out_dim` == `DitConfig`'s `condition_dim`, `DitConfig`'s `in_channels` ==
//! `VocoderConfig`'s `latent_channels`, and `DepthDecoderConfig`'s
//! `hidden_size` == `ConditionEncoderConfig`'s `condition_hidden_dim` (both
//! are the Global LLM's hidden width) - the same invariants the real
//! checkpoint's shapes satisfy, confirmed against the real `config.json`
//! files under each of `condition_encoder/`, `rvq_depth_decoder/`,
//! `transformer/` and `vocoder/`. `num_codebooks` (depth decoder) and
//! `num_condition_layers` (condition encoder) are the same structural count
//! (1 semantic + `num_codebooks - 1` residual per-frame hidden states) and
//! must always agree.

/// The RVQ depth decoder: a 4-layer causal transformer that autoregressively
/// predicts the residual codebooks `c1..c{num_codebooks-1}` within one audio
/// frame, and owns the embedding table for those residual codes.
#[derive(Clone, Copy, Debug)]
pub struct DepthDecoderConfig {
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub intermediate_size: u32,
    pub audio_vocab_size: u32,
    pub num_codebooks: u32,
    pub max_position_embeddings: u32,
}

impl DepthDecoderConfig {
    /// The released checkpoint's real dims (`rvq_depth_decoder/config.json`).
    pub fn real() -> DepthDecoderConfig {
        DepthDecoderConfig {
            hidden_size: 4096,
            num_layers: 4,
            num_attention_heads: 16,
            intermediate_size: 6144,
            audio_vocab_size: 1024,
            num_codebooks: 8,
            max_position_embeddings: 16,
        }
    }

    /// Tiny config for gradcheck/parity fixtures. `hidden_size` matches
    /// [`ConditionEncoderConfig::tiny`]'s `condition_hidden_dim`;
    /// `num_codebooks` matches its `num_condition_layers`.
    pub fn tiny() -> DepthDecoderConfig {
        DepthDecoderConfig {
            hidden_size: 8,
            num_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            audio_vocab_size: 5,
            num_codebooks: 4,
            max_position_embeddings: 9,
        }
    }
}

/// Softmax-mixes the `num_condition_layers` per-frame hidden states (the
/// Global LLM's + each depth-decoder step's), projects, and nearest-resamples
/// from the 25 Hz autoregressive frame rate onto the Flow-VAE latent
/// timeline.
#[derive(Clone, Copy, Debug)]
pub struct ConditionEncoderConfig {
    pub condition_hidden_dim: u32,
    pub num_condition_layers: u32,
    pub out_dim: u32,
    pub input_sampling_rate: u32,
    pub input_hop_length: u32,
    pub output_sampling_rate: u32,
    pub output_hop_length: u32,
}

impl ConditionEncoderConfig {
    /// The released checkpoint's real dims (`condition_encoder/config.json`).
    pub fn real() -> ConditionEncoderConfig {
        ConditionEncoderConfig {
            condition_hidden_dim: 4096,
            num_condition_layers: 8,
            out_dim: 2048,
            input_sampling_rate: 24000,
            input_hop_length: 960,
            output_sampling_rate: 44100,
            output_hop_length: 512,
        }
    }

    pub fn tiny() -> ConditionEncoderConfig {
        ConditionEncoderConfig {
            condition_hidden_dim: 8,
            num_condition_layers: 4,
            out_dim: 6,
            input_sampling_rate: 24000,
            input_hop_length: 960,
            output_sampling_rate: 44100,
            output_hop_length: 512,
        }
    }
}

/// The flow-matching DiT: denoises Flow-VAE audio latents conditioned on the
/// condition encoder's frame-aligned output. `rotary_dim < attention_head_dim`
/// is a PARTIAL rotary embedding - only the first `rotary_dim` dims of each
/// head rotate (`kernels::ROPE_PARTIAL`).
#[derive(Clone, Copy, Debug)]
pub struct DitConfig {
    pub in_channels: u32,
    pub condition_dim: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub attention_head_dim: u32,
    pub ff_inner_dim: u32,
    pub rotary_dim: u32,
    pub fourier_embedding_dim: u32,
}

impl DitConfig {
    /// The released checkpoint's real dims (`transformer/config.json`).
    pub fn real() -> DitConfig {
        DitConfig {
            in_channels: 128,
            condition_dim: 2048,
            num_layers: 36,
            num_attention_heads: 32,
            attention_head_dim: 64,
            ff_inner_dim: 8192,
            rotary_dim: 32,
            fourier_embedding_dim: 256,
        }
    }

    /// `in_channels` matches [`VocoderConfig::tiny`]'s `latent_channels`;
    /// `condition_dim` matches [`ConditionEncoderConfig::tiny`]'s `out_dim`.
    pub fn tiny() -> DitConfig {
        DitConfig {
            in_channels: 4,
            condition_dim: 6,
            num_layers: 2,
            num_attention_heads: 2,
            attention_head_dim: 4,
            ff_inner_dim: 16,
            rotary_dim: 2,
            fourier_embedding_dim: 8,
        }
    }
}

/// The Flow-VAE (DAC-style) waveform decoder: decodes flow-matched latents of
/// shape `(batch, latent_channels, length)` into a stereo waveform - the two
/// audio channels are decoded as two folded `latent_channels / 2` streams.
#[derive(Clone, Debug)]
pub struct VocoderConfig {
    pub latent_channels: u32,
    pub decoder_input_dim: u32,
    pub decoder_hidden_dim: u32,
    pub upsampling_ratios: Vec<u32>,
    pub sampling_rate: u32,
}

impl VocoderConfig {
    /// The released checkpoint's real dims (`vocoder/config.json`).
    pub fn real() -> VocoderConfig {
        VocoderConfig {
            latent_channels: 128,
            decoder_input_dim: 1024,
            decoder_hidden_dim: 1536,
            upsampling_ratios: vec![8, 8, 4, 2],
            sampling_rate: 44100,
        }
    }

    /// `latent_channels` matches [`DitConfig::tiny`]'s `in_channels`.
    /// `decoder_hidden_dim` must be divisible by `2^len(upsampling_ratios)`
    /// (each stage halves the channel count) - `16 / 2^2 = 4`.
    pub fn tiny() -> VocoderConfig {
        VocoderConfig {
            latent_channels: 4,
            decoder_input_dim: 8,
            decoder_hidden_dim: 16,
            upsampling_ratios: vec![2, 2],
            sampling_rate: 8000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_configs_are_shape_consistent_with_each_other() {
        let cond = ConditionEncoderConfig::tiny();
        let dit = DitConfig::tiny();
        let vocoder = VocoderConfig::tiny();
        let depth = DepthDecoderConfig::tiny();

        assert_eq!(cond.out_dim, dit.condition_dim);
        assert_eq!(dit.in_channels, vocoder.latent_channels);
        assert_eq!(depth.hidden_size, cond.condition_hidden_dim);
        assert_eq!(depth.num_codebooks, cond.num_condition_layers);
        assert_eq!(
            vocoder.decoder_hidden_dim % (1 << vocoder.upsampling_ratios.len()),
            0,
            "decoder_hidden_dim must halve evenly across every upsample stage"
        );
    }

    #[test]
    fn real_configs_are_shape_consistent_with_each_other() {
        let cond = ConditionEncoderConfig::real();
        let dit = DitConfig::real();
        let vocoder = VocoderConfig::real();
        let depth = DepthDecoderConfig::real();

        assert_eq!(cond.out_dim, dit.condition_dim);
        assert_eq!(dit.in_channels, vocoder.latent_channels);
        assert_eq!(depth.hidden_size, cond.condition_hidden_dim);
        assert_eq!(depth.num_codebooks, cond.num_condition_layers);
    }
}
