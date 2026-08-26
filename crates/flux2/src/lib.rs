// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 Klein (4B / 9B) in brain.
//!
//! The Black Forest Labs FLUX.2 Klein family: an MMDiT-style rectified-flow
//! transformer (double-stream img/txt blocks with joint attention, then
//! single-stream parallel blocks) over the FLUX.2 autoencoder's 128-channel
//! latent, conditioned on three concatenated Qwen3 hidden states. The klein
//! variants are step- + guidance-distilled (4 Euler steps, no CFG, no guidance
//! embedding); the `base` variants are the undistilled weights with the same
//! tensor layout (50 steps + real CFG).
//!
//! Canonical tensor names are the BFL reference names (`double_blocks.N.…`,
//! `single_blocks.N.…`); the diffusers `transformer/` layout is remapped onto
//! them at import ([`import::import_diffusers`]).
//!
//! Training: [`grad`] (FD-gradchecked block fwd+bwd reference) → [`modelgrad`]
//! (whole-model host fwd+bwd under the rectified-flow velocity-MSE loss, incl.
//! the conditioning path) → [`lora`] (frozen-base low-rank adapters over the
//! fused checkpoint layout) → [`finetune`] (captioned-image folder →
//! trained adapter, host f32 trainer).

pub mod caps;
pub mod config;
pub mod finetune;
pub mod grad;
pub mod import;
pub mod lora;
pub mod model;
pub mod modelgrad;
pub mod pipeline;

pub use config::Flux2Config;
pub use import::{import_bfl, import_diffusers, Tensors};
pub use model::{position_ids, Flux2Model, Precision, Sample, KERNELS};
pub use pipeline::{AdapterSpec, BatchRequest, GenOpts, Paths, Pipeline};
