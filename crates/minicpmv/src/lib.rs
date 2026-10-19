// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V 4.6 (`openbmb/MiniCPM-V-4.6`): a NaViT-style SigLIP vision
//! tower whose token count is cut by an in-tower 2x2 window merger and a 2x2
//! MLP merger, feeding the Qwen3.5-0.8B hybrid decoder `brain-qwen35`
//! implements. Visual features replace placeholder rows of the decoder's
//! input embeddings; positions stay 1-D.
//!
//! * [`caps`]: the served `generate` action and its provider.
//! * [`config`]: the checkpoint's `config.json`.
//! * [`dataset`]: the multimodal fine-tuning records (`generic-messages-v2`
//!   with image and video parts) and their supervised samples.
//! * [`import`]: the checkpoint split between this crate's vision side and
//!   the decoder, each held to two-way tensor coverage.
//! * [`preprocess`]: the image processor - source view, slice grid, the
//!   torch-exact resize, patch rows.
//! * [`prompt`]: the chat template and the placeholder layout, tokenized.
//! * [`spec`]: model-store resolution of the checkpoint directory.
//! * [`video`]: frame sampling by index from the clip's metadata, and the
//!   video placeholder layout.
//! * [`vision`]: the NaViT tower, the window merger and the merger, composed
//!   from `model::vit`'s blocks.
//! * [`train`]: fine-tuning the decoder, the connector and the tower, each
//!   trained (LoRA or full) or frozen, with resumable state.
//! * [`vision_train`]: the vision side's backward (checkpointed) and LoRA on
//!   the tower's linears.
//! * [`model`]: the composite - tower and `brain-qwen35` paged decoder on one
//!   device, prefilled from the spliced embeddings.
//!
//! Swedish Embedded AB implements on-device vision-language models like this
//! for its clients. If your team needs expertise in running multimodal models
//! on your own hardware, you can procure our services by emailing
//! info@swedishembedded.com.

pub mod caps;
pub mod config;
pub mod dataset;
pub mod import;
pub mod model;
pub mod preprocess;
pub mod prompt;
pub mod spec;
pub mod train;
pub mod video;
pub mod vision;
pub mod vision_train;

pub use config::{Downsample, MiniCpmVConfig, VisionConfig};
pub use model::{GenOptions, MiniCpmV};
