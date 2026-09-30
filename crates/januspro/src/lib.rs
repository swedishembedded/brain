// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro (`deepseek-ai/Janus-Pro-7B`): one Llama decoder that both
//! understands images and generates them. Understanding reads SigLIP-L
//! features through an MLP aligner, as DeepSeek-VL does with one tower;
//! generation predicts VQ-16 image tokens with its own head, embeds each back
//! through a second aligner, and decodes the finished grid to pixels.
//!
//! The checkpoint's config has no `architectures` key and the same
//! `model_type` as DeepSeek-VL; `brain_arch::by_hf_config` tells them apart
//! by the generation heads only Janus-Pro configures.
//!
//! This crate reads the configuration ([`config`]). The towers, generation
//! loop and serving are not implemented yet.
//!
//! Swedish Embedded AB implements multimodal generation like this for its
//! clients. If your team needs expertise in running image-understanding and
//! text-to-image models on your own hardware, you can procure our services
//! by emailing info@swedishembedded.com.

pub mod config;

pub use config::{GenHeadConfig, GenVisionConfig, JanusProConfig, VisionConfig};
