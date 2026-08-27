// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Convolutional KL image autoencoder — the `AutoencoderKL` used by Z-Image,
//! FLUX.2, and HiDream-I1 to compress images into the latent space the DiT
//! denoises.
//!
//! Decoder-first: latent-diffusion inference only needs `decode` (the VAE runs
//! frozen), so that is what lands first and is parity-gated against the
//! reference. The encoder, KL diagonal-Gaussian sampling, and the trainable
//! `Model` impl (with hand-written backward + gradcheck) land alongside as the
//! training path is wired.
//!
//! Everything is composed from brain's existing kernels — conv (`conv_bias_reg`),
//! GroupNorm (`gn_stats`/`gn_apply`), SiLU, residual add, nearest upsample, and
//! the bidirectional self-attention trio — the same primitives the DIAMOND UNet
//! (`crates/wm-diamond`) already validates.
//!
//! [`blocks3d`] is the sibling builder for **3D causal video** autoencoders
//! (`[C, T, H, W]` volumes, causal `conv3d`, cross-chunk `FeatCache`). It is
//! deliberately a sibling and not a widening of [`blocks`]: five consumers
//! depend on the latter's `(prefix, c, h, w, x)` signatures and none of them has
//! a time axis. The two share the kernel set, not the builder.

pub mod blocks;
pub mod blocks3d;
pub mod config;
pub mod decoder;
pub mod latent;
pub mod tiling3d;

pub use config::VaeConfig;
pub use decoder::{
    decoder_device_bytes, decoder_device_bytes_for_pixels, decoder_weight_bytes, encoder_device_bytes,
    encoder_device_bytes_for_pixels, encoder_weight_bytes,
    device, level_bytes_per_pixel, Tensors, VaeDecoder, VaeEncoder,
};
