// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sparse Mixture-of-Experts Transformer.
//!
//! - [`model`] — inference + generation (RMSNorm, RoPE, top-k experts, tied head).
//! - [`train`] — full training: forward + backprop + AdamW as WGSL kernels
//!   (numerically gated by the finite-difference `brain-gradcheck`).
//!
//! - [`caps`] - the served actions: `train`, `eval`, `generate`.

/// The epsilon of every RMSNorm in this model. It is brain's own
/// architecture (no upstream checkpoint), so the value is fixed here once
/// rather than read from a config.
pub const RMS_EPS: f32 = 1e-6;

pub mod caps;
pub mod model;
pub mod train;

pub use model::*;
