// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS Talker + MTP code predictor.
//!
//! The Talker is a Qwen3-style dense decoder (RMSNorm, GQA, RoPE, SwiGLU) that
//! predicts codebook-0 acoustic tokens conditioned on projected text embeddings
//! and a speaker embedding; the MTP (multi-token-prediction) code predictor is a
//! small 5-layer Qwen3 block that fills residual codebooks 1..15 from codebook-0.
//!
//! Reuses `crates/qwen` (decoder backbone, import, vocab-tiling, LoRA) and
//! `crates/model::block` builders; the new structure is multi-codebook +
//! text-projection embeddings, a dual-track step graph, and a KV-cache seam.

pub mod config;
pub mod gen;
pub mod gen_kv;
pub mod gen_kv_mtp;
pub mod import;
pub mod mtp;
pub mod pipeline;
pub mod prompt;
pub mod sft;
pub mod talker;

pub use config::{MtpConfig, TalkerConfig};
pub use sft::{ce, ce_batch, FinetuneOpts, MultiCodebookLabels};
pub use gen::TalkerGen;
pub use gen_kv::CpuTalker;
pub use gen_kv_mtp::CpuMtp;
pub use mtp::MtpModel;
pub use pipeline::{GenOpts, TtsPaths};
pub use prompt::{Prompt, TtsSpecials};
pub use talker::{TalkerModel, TextProjection};
