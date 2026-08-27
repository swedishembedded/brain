// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FastVLM (Apple) — a FastViTHD hybrid conv/attention vision encoder + an
//! `mlp2x_gelu` projector in front of a Qwen2 decoder driven from spliced image
//! embeddings (LLaVA-style, image token id `-200`).
//!
//! Reuse plan: the decoder is the existing `qwen` decoder with two config toggles
//! (QK-norm **off**, qkv **bias on** — the Qwen2-vs-Qwen3 deltas) and the shared
//! `model::vlm` splice seam; FastViTHD is composed from brain's existing conv
//! kernels (no net-new device kernels) as block builders. Built incrementally —
//! today: configuration.

pub mod config;
pub mod encoder;
pub mod model;
pub mod import;
pub mod vision_import;
/// `capability::Provider` — the image-caption action (`brain do fastvlm caption`).
pub mod caps;
pub mod captioner;
#[cfg(test)]
mod parity;
#[cfg(test)]
mod train_smoke;

pub use config::{FastVitHdConfig, FastVlmConfig};
