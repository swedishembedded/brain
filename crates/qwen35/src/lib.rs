// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5/3.8-27B: a dense hybrid Gated-DeltaNet / GQA decoder Transformer,
//! forward + backprop as hand-written WGSL compute kernels - the dense
//! sibling of `crates/qwen35moe` (llama.cpp registers the two as separate
//! architectures, `LLM_ARCH_QWEN35` vs `LLM_ARCH_QWEN35MOE`, despite sharing
//! one HF `model_type`, `"qwen3_5"`).
//!
//! See `crates/qwen35/src/config.rs` for the architecture summary. Built
//! against the real `Qwen/Qwen3.8-27B-FP8` checkpoint's `config.json` and
//! the installed `transformers.models.qwen3_5` reference module - unlike
//! `qwen35moe`, which was ported with no reference available, this port is
//! gated by real per-stage parity against goldens dumped by
//! `tools/goldens/qwen35_dump_reference.py`.

pub mod caps;
pub mod config;
pub mod finetune;
pub mod gguf_import;
pub mod import;
pub mod int8_gguf_resident;
pub mod init;
pub mod model;
pub mod sample;
pub mod serve;
pub mod shard;
pub mod stream;
pub mod stream_train;
pub mod vl;

pub use config::{LayerType, Qwen35Config};
pub use model::Qwen35;
