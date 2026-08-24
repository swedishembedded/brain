// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3 dense decoder Transformer for brain: pure Rust + WGSL, fp32, runs on
//! the shared `gpu_core` engine (wgpu or the native CPU JIT). See `model.rs` for
//! the forward/backprop dispatch graph and `config.rs` for the architecture.

pub mod caps;
pub mod chat;
pub mod config;
pub mod eval;
pub mod import;
pub mod init;
pub mod model;
pub mod q8;
pub mod sample;
pub mod serve;
pub mod finetune;
pub mod lora;
pub mod shard;
pub mod toolcall_eval;

pub use config::{LoraCfg, QwenConfig};
pub use init::init_weights;
pub use model::{Qwen, Shard, IGNORE};
/// The weight **storage tier** [`Qwen::new_shard_dt`] takes, re-exported so a
/// caller naming one does not have to depend on `gpu-core` directly.
pub use gpu_core::select::Dtype;
/// Generic multi-GPU training (see [`::model`]); use as `Pipeline::<Qwen>::new(..)`
/// / `DataParallel::<Qwen>::new(..)`.
pub use ::model::{DataParallel, Pipeline};
