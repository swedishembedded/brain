// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AutoencoderKL` config (the diffusers `config.json` fields brain needs).

/// The subset of the diffusers `AutoencoderKL` config that determines the
/// decoder graph. Z-Image / FLUX-dev VAE: 16 latent channels, `[128,256,512,
/// 512]` block channels (an eightfold spatial downscale), 2 layers/block, 32 groups, `silu`.
/// FLUX.2 (`AutoencoderKLFlux2`) keeps the conv net and adds: 32 latent
/// channels, 1×1 `quant_conv`/`post_quant_conv` at the latent boundary, and a
/// 2×2 latent pixel-unshuffle normalized by frozen BatchNorm stats
/// (`crate::latent`).
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
    /// 1×1 `quant_conv` (2·latent → 2·latent) after the encoder `conv_out`
    /// (SDXL/SD1.x and FLUX.2: true; Z-Image/FLUX.1: false). Defaults to TRUE
    /// when the config omits it — see [`VaeConfig::from_json`].
    pub use_quant_conv: bool,
    /// 1×1 `post_quant_conv` (latent → latent) before the decoder `conv_in`
    /// (SDXL/SD1.x and FLUX.2: true; Z-Image/FLUX.1: false). Defaults to TRUE
    /// when the config omits it — see [`VaeConfig::from_json`].
    pub use_post_quant_conv: bool,
    /// Latent pixel-unshuffle patch `[pi, pj]` (FLUX.2: `[2,2]` → the DiT sees
    /// `prod(patch)·latent` channels). `[1,1]` = no packing.
    pub patch_size: [u32; 2],
    /// Eval-mode BatchNorm epsilon for the packed-latent normalization
    /// (FLUX.2: 1e-4). Unused when `patch_size == [1,1]`.
    pub batch_norm_eps: f32,
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
            // DEFAULT TRUE, because that is `AutoencoderKL.__init__`'s default
            // (`use_quant_conv: bool = True, use_post_quant_conv: bool = True`)
            // and a config.json only carries the keys it OVERRIDES. Every model
            // that omits them — the whole SDXL/SD1.x family — therefore means
            // true, while the models that want them off (FLUX.1, Z-Image) say
            // `false` explicitly. Defaulting to false silently dropped SDXL's
            // `post_quant_conv`, a 4x4 mixing of the latent channels: the decode
            // stayed in a plausible [-1,1] range and was UNCORRELATED with the
            // reference (cosine -0.03), so the picture had roughly the right
            // structure and unusable colour.
            use_quant_conv: v.get("use_quant_conv").and_then(|x| x.as_bool()).unwrap_or(true),
            use_post_quant_conv: v
                .get("use_post_quant_conv")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            patch_size: v
                .get("patch_size")
                .and_then(|x| x.as_array())
                .and_then(|a| match a.as_slice() {
                    [i, j] => Some([i.as_u64()? as u32, j.as_u64()? as u32]),
                    _ => None,
                })
                .unwrap_or([1, 1]),
            batch_norm_eps: f("batch_norm_eps", 1e-4),
        }
    }

    /// FLUX.2 (Klein/dev) `AutoencoderKLFlux2` preset: 32 latent channels,
    /// quant/post-quant 1×1 convs, 2×2 latent packing with BatchNorm-stat
    /// normalization (eps 1e-4). Latent scale/shift are identity — FLUX.2
    /// normalizes via the checkpoint's `bn.running_{mean,var}` instead.
    pub fn flux2() -> VaeConfig {
        VaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 32,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            norm_eps: 1e-6,
            mid_block_add_attention: true,
            scaling_factor: 1.0,
            shift_factor: 0.0,
            use_quant_conv: true,
            use_post_quant_conv: true,
            patch_size: [2, 2],
            batch_norm_eps: 1e-4,
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
