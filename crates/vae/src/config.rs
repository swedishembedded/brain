// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AutoencoderKL` config (the diffusers `config.json` fields brain needs).

/// The subset of the diffusers `AutoencoderKL` config that determines the
/// decoder graph. Z-Image / FLUX-dev VAE: 16 latent channels, `[128,256,512,
/// 512]` block channels (→ 8× spatial), 2 layers/block, 32 groups, `silu`.
#[derive(Clone, Debug, PartialEq)]
pub struct VaeConfig {
    pub in_channels: u32,
    pub out_channels: u32,
    pub latent_channels: u32,
    /// Encoder channel schedule, low→high res. The decoder walks it reversed.
    pub block_out_channels: Vec<u32>,
    pub layers_per_block: u32,
    pub norm_num_groups: u32,
    /// GroupNorm epsilon (diffusers VAE default 1e-6).
    pub norm_eps: f32,
    /// Whether the mid block has a self-attention layer (Z-Image/FLUX: true).
    pub mid_block_add_attention: bool,
    pub scaling_factor: f32,
    pub shift_factor: f32,
}

impl VaeConfig {
    /// Parse the fields we need from a diffusers `vae/config.json` value.
    pub fn from_json(v: &serde_json::Value) -> VaeConfig {
        let u = |k: &str, d: u32| v.get(k).and_then(|x| x.as_u64()).map(|x| x as u32).unwrap_or(d);
        let f = |k: &str, d: f32| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(d);
        let block_out_channels = v
            .get("block_out_channels")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|e| e.as_u64().map(|n| n as u32)).collect())
            .unwrap_or_else(|| vec![128, 256, 512, 512]);
        VaeConfig {
            in_channels: u("in_channels", 3),
            out_channels: u("out_channels", 3),
            latent_channels: u("latent_channels", 16),
            block_out_channels,
            layers_per_block: u("layers_per_block", 2),
            norm_num_groups: u("norm_num_groups", 32),
            // diffusers ResnetBlock2D / conv_norm_out use eps 1e-6 for the VAE.
            norm_eps: f("norm_eps", 1e-6),
            mid_block_add_attention: v
                .get("mid_block_add_attention")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            scaling_factor: f("scaling_factor", 0.3611),
            shift_factor: f("shift_factor", 0.1159),
        }
    }

    /// Decoder channel schedule (block_out reversed): the per-up-block output
    /// channel count, high→low res. E.g. `[128,256,512,512] → [512,512,256,128]`.
    pub fn reversed_channels(&self) -> Vec<u32> {
        let mut c = self.block_out_channels.clone();
        c.reverse();
        c
    }

    /// Total spatial upscale factor of the decoder (`2^(num_blocks-1)`): one
    /// nearest-2× upsample per up-block except the last. `[128,256,512,512]` → 8.
    pub fn upscale_factor(&self) -> u32 {
        1 << (self.block_out_channels.len() as u32 - 1)
    }
}
