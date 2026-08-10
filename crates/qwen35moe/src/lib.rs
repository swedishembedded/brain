// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5-35B-A3B: a hybrid Gated-DeltaNet / GQA sparse-MoE decoder
//! Transformer, forward + backprop as hand-written WGSL compute kernels.
//!
//! See `crates/qwen35moe/src/config.rs` for the architecture summary and
//! `docs/models/qwen35/status.md` (once created) for the porting ledger.
//! Built against the real `Qwen/Qwen3.5-35B-A3B` checkpoint's
//! `config.json`/`modeling_qwen3_5_moe.py` (see
//! `/data/workspace/resources/qwen3.5/` for the reference sources).

pub mod config;
pub mod import;
pub mod init;
pub mod model;
pub mod q8;

pub use config::{LayerType, Qwen35Config};
pub use model::Qwen35;
