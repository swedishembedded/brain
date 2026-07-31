// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sparse Mixture-of-Experts Transformer.
//!
//! - [`model`] — inference + generation (RMSNorm, RoPE, top-k experts, tied head).
//! - [`train`] — full training: forward + backprop + AdamW as WGSL kernels
//!   (numerically gated by the finite-difference `brain-gradcheck`).
//!
//! The model's public CLI entry points ([`run_generate`], [`run_train`],
//! [`run_eval`]) are re-exported here for the `brain` binary.

pub mod model;
pub mod train;

pub use model::*;
