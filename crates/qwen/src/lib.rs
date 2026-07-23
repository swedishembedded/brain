// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3 dense decoder Transformer for brain: pure Rust + WGSL, fp32, runs on
//! the shared `gpu_core` engine (wgpu or the native CPU JIT). See `model.rs` for
//! the forward/backprop dispatch graph and `config.rs` for the architecture.

pub mod config;
pub mod import;
pub mod init;
pub mod model;
pub mod sample;
pub mod finetune;
pub mod toolcall_eval;

pub use config::{LoraCfg, QwenConfig};
pub use init::init_weights;
pub use model::{Qwen, IGNORE};
