// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Encoder-decoder Transformer (ADR 0001 §5), a first-class [`model::Model`].
//!
//! - [`model`] — forward + backprop as WGSL dispatches ([`model::Seq2Seq`]).
//! - [`init`] — weight initialization (GPT-2 style scaled residual projections).

pub mod init;
pub mod model;

pub use init::init_weights;
pub use model::{Seq2Seq, Seq2SeqConfig, IGNORE};
