// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL — a ViT vision encoder + PatchMerger + DeepStack in front of a Qwen3
//! text decoder driven from spliced image embeddings (`qwen::Qwen::enable_mm_splice`).
//!
//! Assembled from brain's existing building blocks: the shared `model::vit` block
//! builder for the ViT, the whole `qwen` decoder for the text side (interleaved
//! M-RoPE swaps its analytic RoPE for the table-driven `rope2d` kernel), and the
//! `model::vlm` splice seam to inject the projected vision tokens. No net-new
//! device kernels — see docs in each stage.
//!
//! This crate is being built incrementally (config → M-RoPE → ViT → merger →
//! DeepStack → import). Today: configuration.

pub mod config;
pub mod encoder;
pub mod import;
pub mod model;
pub mod mrope;
pub mod preprocess;
pub mod vision;

pub use model::Qwen3Vl;

pub use config::{Qwen3VlConfig, VisionConfig};
