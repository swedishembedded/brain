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
//! * [`config`]: the checkpoint's `config.json`.
//! * [`model`]: the understanding path, `brain-deepseekvl`'s composite with
//!   Janus-Pro's tower, roles and image tags.
//! * [`gen`]: the generation head, embedding table and aligner.
//! * [`caps`]: the `generate` and `text2image` actions and their provider.
//! * [`t2i`]: classifier-free-guided text-to-image on the serving engine,
//!   decoded by `brain-vqgan`'s VQ-16.
//!
//! Swedish Embedded AB implements multimodal generation like this for its
//! clients. If your team needs expertise in running image-understanding and
//! text-to-image models on your own hardware, you can procure our services
//! by emailing info@swedishembedded.com.

pub mod caps;
pub mod config;
pub mod gen;
pub mod model;
pub mod spec;
pub mod t2i;

pub use config::{GenHeadConfig, GenVisionConfig, JanusProConfig, VisionConfig};
