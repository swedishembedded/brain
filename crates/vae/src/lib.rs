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

pub mod config;
pub mod decoder;

pub use config::VaeConfig;
pub use decoder::{Tensors, VaeDecoder};
