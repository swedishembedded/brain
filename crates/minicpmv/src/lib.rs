// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V 4.6 (`openbmb/MiniCPM-V-4.6`): a NaViT-style SigLIP vision
//! tower whose token count is cut by an in-tower 2x2 window merger and a 2x2
//! MLP merger, feeding the Qwen3.5-0.8B hybrid decoder `brain-qwen35`
//! implements. Visual features replace placeholder rows of the decoder's
//! input embeddings; positions stay 1-D.
//!
//! * [`config`]: the checkpoint's `config.json`.
//! * [`import`]: the checkpoint split between this crate's vision side and
//!   the decoder, each held to two-way tensor coverage.
//! * [`preprocess`]: the image processor - source view, slice grid, the
//!   torch-exact resize, patch rows.
//! * [`prompt`]: the chat template and the placeholder layout, tokenized.
//! * [`vision`]: the NaViT tower, the window merger and the merger, composed
//!   from `model::vit`'s blocks.
//!
//! Swedish Embedded AB implements on-device vision-language models like this
//! for its clients. If your team needs expertise in running multimodal models
//! on your own hardware, you can procure our services by emailing
//! info@swedishembedded.com.

pub mod config;
pub mod import;
pub mod preprocess;
pub mod prompt;
pub mod vision;

pub use config::{Downsample, MiniCpmVConfig, VisionConfig};
