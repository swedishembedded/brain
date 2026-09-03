// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL-30B-A3B (`qwen3vlmoe`): `qwen3vl`'s ViT + PatchMerger + DeepStack
//! vision tower spliced onto a `qwen3omnimoe`-shaped MoE text decoder (GQA +
//! QK-norm + RoPE + top-k sparse MoE, no shared expert).
//!
//! **Scope of this crate today**: architecture-name recognition
//! ([`import::GGUF_ARCHITECTURE`]) and a config/composite-struct SHAPE
//! ([`config::Qwen3VlMoeConfig`], [`model::Qwen3VlMoe`]) verified against the
//! REAL released `config.json` and proven to compose correctly on synthetic
//! tiny configs. This is NOT a real-checkpoint-verified port - no
//! `Qwen3-VL-30B-A3B` checkpoint (safetensors or GGUF) was available to
//! import against in this environment. See `model`'s and `import`'s module
//! docs for exactly what is and is not claimed, and this workspace's
//! vision-language roadmap log for the open items this leaves.

pub mod config;
pub mod import;
pub mod model;
