// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder (LiquidAI): bidirectional hybrid short-conv/attention
//! encoder with a tied masked-LM head.
//!
//! Reference: `LiquidAI/LFM2.5-Encoder-{230M,350M}` (HF), imported 1:1 and
//! parity-gated against the released fp32 checkpoints.

pub mod caps;
pub mod config;
pub mod import;
pub mod init;
pub mod model;
pub mod train;

pub use config::{LayerType, LfmConfig};
pub use model::Lfm;
