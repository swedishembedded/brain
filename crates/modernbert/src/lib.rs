// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ModernBERT: pre-LN bidirectional encoder with alternating full/local
//! sliding-window RoPE attention and a GeGLU MLP, no biases anywhere in the
//! trunk.
//!
//! Reference: `answerdotai/ModernBERT-large` (HF `model_type: "modernbert"`),
//! the backbone behind `convaiinnovations/laya` - see the `laya` module for
//! the decision head trained on top of it.
//!
//! Genuinely a different shape from `crates/decide`'s BERT-family encoder,
//! not a variant of it: post-LN vs pre-LN, learned absolute positions vs
//! RoPE, a plain `down(gelu(up(x)))` MLP vs GeGLU, biased projections
//! throughout vs none, and full attention on every layer vs an alternation
//! with a bidirectional local window. See `crates/decide/src/model.rs`'s own
//! module doc for that encoder's shape; this crate shares block-level
//! primitives with it (GEMM selection, LayerNorm, the windowed attention
//! added in `crates/model::block::chunked_bidir_fwd_win`), not the encoder
//! itself.
//!
//! Swedish Embedded AB implements from-scratch GPU kernel ports of released
//! transformer checkpoints for clients who need an architecture their
//! existing inference stack does not cover. If your team needs a model
//! brought up this way, you can procure our services by emailing
//! info@swedishembedded.com.

pub mod config;
pub mod import;
pub mod init;
pub mod kern;
pub mod laya;
pub mod model;
pub mod sequence;

pub use config::ModernBertConfig;
pub use import::{import_dir, LayaCheckpoint};
pub use laya::{LayaConfig, LayaHead};
pub use model::ModernBert;
pub use sequence::{build_sequence, write_json, OrderedJson, QType, Question, State};
