// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Florence-2's BART-style shared encoder-decoder text model - see
//! [`lm`]'s module doc for the top-level orchestration.

pub mod attn;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod lm;

pub use config::{BartConfig, POSITION_OFFSET};
pub use decoder::{Decoder, DecoderKernelIds};
pub use encoder::{Encoder, EncoderKernelIds};
pub use lm::{Florence2Lm, Florence2LmKernelIds};
