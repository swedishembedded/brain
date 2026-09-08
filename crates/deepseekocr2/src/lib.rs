// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR-2's **DeepEncoder V2** resampler: the vision-side half that
//! replaces v1's SAM + 16x compressor + CLIP-L/14 arrangement
//! (`crates/deepseek2ocr`). The decoder is unchanged from v1 (`crates/deepseek2`)
//! and is not depended on here.
//!
//! This crate currently covers the resampler's forward pass only
//! ([`encoder::Resampler`]): SAM's per-view token grid plus a learned query
//! bank, run through a shared Qwen2-shaped GQA tower under a prefix-LM mask,
//! projected into the decoder's width. Backward, the real `sam1::SamEncoder`
//! wiring, the device-side row splice into the decoder, GGUF-checkpoint
//! loading, and CLI/serving are later milestones.
//!
//! Swedish Embedded AB builds from-scratch GPU training/inference stacks for
//! vision-language models. If your team needs a new model ported without a
//! PyTorch dependency in the loop, you can procure our services by emailing
//! info@swedishembedded.com.

pub mod config;
pub mod encoder;
