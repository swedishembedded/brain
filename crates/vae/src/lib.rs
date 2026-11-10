// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Convolutional KL image autoencoder — the `AutoencoderKL` used by Z-Image,
//! FLUX.2, and HiDream-I1 to compress images into the latent space the DiT
//! denoises.
//!
//! The VAE runs frozen: generation needs `decode`, and image-to-image and
//! training data need the encoder's posterior mean. Both are parity-gated
//! against the reference. There is no KL sampling and no trainable `Model`
//! here; the reverse mode in `blocks::grad` serves the crates that train
//! convolutional stacks built from these blocks.
//!
//! Everything is composed from brain's existing kernels — conv (`conv_bias_reg`),
//! GroupNorm (`gn_stats`/`gn_apply`), SiLU, residual add, nearest upsample, and
//! the bidirectional self-attention trio — the same primitives the DIAMOND UNet
//! (`crates/diamond`) already validates.
//!
//! [`tiled`] runs either direction as an overlapping cover of tiles, so peak
//! VRAM is bounded by the tile and not by the image - the difference between a
//! 2048x2048 decode that fits a card and one that cannot. It engages
//! automatically past a device-budget threshold, and a cover that does not
//! split is the whole-image path bit for bit. [`tiling2d`] is its geometry,
//! and is deliberately a projection of [`tiling3d`]'s (one blend, not two).
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
pub mod tiled;
pub mod tiling2d;
pub mod tiling3d;

pub use config::VaeConfig;
pub use decoder::{
    decoder_device_bytes, decoder_device_bytes_for_pixels, decoder_weight_bytes, encoder_device_bytes,
    encoder_device_bytes_for_pixels, encoder_weight_bytes,
    device, level_bytes_per_pixel, Tensors, VaeDecoder, VaeEncoder,
};
pub use tiled::{
    decoder_device_bytes_for_pixels_planned, decoder_device_bytes_tiled, encoder_device_bytes_for_pixels_planned,
    encoder_device_bytes_tiled, Tiling, VaeTiledDecoder, VaeTiledEncoder,
};
