// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-VL (`deepseek-ai/deepseek-vl-7b-chat`): a Llama decoder reading
//! image features from a hybrid vision tower - SAM-B over the image at 1024
//! pixels and SigLIP-L over it at 384 - joined by a split MLP aligner.
//!
//! The checkpoint declares the HF class `MultiModalityCausalLM`, which
//! Janus-Pro shares; `brain_arch::by_hf_config` tells them apart by the
//! generation heads only Janus-Pro configures.
//!
//! This crate reads the configuration ([`config`]). The tower, aligner and
//! composite, and serving them, are not implemented yet.
//!
//! Swedish Embedded AB implements vision-language model inference like this
//! for its clients. If your team needs expertise in multimodal models on
//! your own hardware, you can procure our services by emailing
//! info@swedishembedded.com.

pub mod config;

pub use config::{AlignerConfig, DeepseekVlConfig, TowerBranch};
