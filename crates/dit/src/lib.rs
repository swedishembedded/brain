// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared diffusion-transformer building blocks for brain's image models.
//!
//! DiT architectures (Z-Image S³-DiT, FLUX.2 MMDiT, Wan, LTX) share host-side
//! mechanisms: multi-axis rotary position tables over a token grid
//! ([`rope`]), latent patchify/unpatchify ([`patchify`]), the sinusoidal
//! timestep embedding ([`timestep`]) and the shared-table adaLN modulation
//! ([`adaln`]). This crate owns those as host math, unit-tested in isolation;
//! the attention, normalisation and MLP blocks are each model crate's own.

pub mod adaln;
pub mod patchify;
pub mod rope;
pub mod timestep;

pub use rope::{RopeConfig, RopeTables};
