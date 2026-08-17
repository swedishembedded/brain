// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared diffusion-transformer building blocks for brain's image models.
//!
//! DiT architectures (Z-Image S³-DiT, FLUX.2 MMDiT, HiDream) share a common set
//! of mechanisms — multi-axis rotary position encoding over a token grid,
//! adaLN-zero timestep modulation, sinusoidal timestep embedding, latent
//! patchify/unpatchify, QK-normalized attention, and SwiGLU MLPs. This crate
//! owns those, so each per-model crate (`zimage`, `flux2`, `hidream`) is a thin
//! assembly over them, the same way the model crates compose `crates/model`.
//!
//! Host-math pieces (which need no device) land first and are unit-tested in
//! isolation; the device blocks build on `gpu_core`/`kernels`/`wm-core` as they
//! are wired to each model's forward.

pub mod adaln;
pub mod patchify;
pub mod rope;
pub mod timestep;

pub use rope::{RopeConfig, RopeTables};
