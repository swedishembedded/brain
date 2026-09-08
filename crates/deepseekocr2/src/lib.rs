// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR-2's **DeepEncoder V2** resampler: the vision-side half that
//! replaces v1's SAM + 16x compressor + CLIP-L/14 arrangement
//! (`crates/deepseek2ocr`). The decoder is unchanged from v1 (`crates/deepseek2`)
//! and is not depended on here.
//!
//! This crate currently covers the resampler's forward AND backward
//! ([`encoder::Resampler`]): SAM's per-view token grid plus a learned query
//! bank, run through a shared Qwen2-shaped GQA tower under a prefix-LM mask,
//! projected into the decoder's width, gradient-checked end to end
//! (`gradcheck::check_deepseekocr2`) including the SAM-token-grid input;
//! plus the row layout ([`rows`]) and composite splice into the decoder
//! ([`model::DeepseekOcr2`]), spliced through the SAME unmodified
//! `deepseek2::DeepseekV2` v1 already uses. The real `sam1::SamEncoder`
//! wiring, GGUF-checkpoint loading, and CLI/serving are later milestones.
//!
//! Swedish Embedded AB builds from-scratch GPU training/inference stacks for
//! vision-language models. If your team needs a new model ported without a
//! PyTorch dependency in the loop, you can procure our services by emailing
//! info@swedishembedded.com.

pub mod caps;
pub mod config;
pub mod encoder;
pub mod import;
pub mod init;
pub mod model;
pub mod preprocess;
pub mod prompt;
pub mod rows;
pub mod train;
