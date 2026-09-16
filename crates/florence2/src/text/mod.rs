// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Florence-2's BART-style shared encoder-decoder text model - see
//! [`lm`]'s module doc for the top-level orchestration.

pub mod attn;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod lm;
pub mod lora;

pub use attn::{BartAttnBwdIds, BartAttnBwdScratch};
pub use config::{BartConfig, POSITION_OFFSET};
pub use decoder::{Decoder, DecoderBwdKernelIds, DecoderKernelIds};
pub use encoder::{Encoder, EncoderBwdKernelIds, EncoderKernelIds};
pub use lm::{Florence2Lm, Florence2LmKernelIds};
pub use lora::{LoraCfg, LoraCtx, LoraKernelIds, LoraScratch};
