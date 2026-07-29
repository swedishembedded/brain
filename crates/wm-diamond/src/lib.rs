// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAMOND (Diffusion As a Model Of eNvironment Dreams, NeurIPS 2024) —
//! the Atari-100k EDM diffusion world model, playable through
//! [`wm_core::WorldModel`].
//!
//! - [`config`]: architecture + parameter manifest (reference names).
//! - [`cond`]: host-side EDM conditioners, Fourier/action/cond-MLP path,
//!   AdaGroupNorm gamma/beta production, Karras sigmas.
//! - [`model`]: the UNet as one pre-recorded brain kernel graph.
//! - [`import`]: torch `.pt` -> `.weights` with full-coverage validation.
//! - [`play`]: the context ring + Euler denoising loop behind the trait.
//! - [`npu`]: fp32 ONNX export + the OpenVINO (Intel NPU) playback path.
//!
//! Reference: the DIAMOND repo (github.com/eloialonso/diamond, MIT).
//! Parity fixtures: `make wm-fixtures` (docs/models/world-models/fixtures.md).

pub mod cond;
pub mod config;
pub mod import;
pub mod model;
pub mod npu;
pub mod play;
pub mod train;

pub use config::DiamondConfig;
pub use model::{DiamondUNet, Tensors};
pub use play::DiamondWorldModel;
