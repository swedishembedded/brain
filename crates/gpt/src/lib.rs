// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense GPT decoder Transformer (nanogpt parity).
//!
//! - [`model`] — forward + backprop as WGSL dispatches ([`model::Gpt`]).
//! - [`init`] — nanogpt-style weight initialization.

pub mod init;
pub mod model;
pub mod sample;
pub mod train;

pub use init::init_weights;
pub use model::{Gpt, GptConfig};
pub use train::{train, TrainOpts};
