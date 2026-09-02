// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GGUF → brain-native import: the parts that are the same for every model.
//!
//! `checkpoint::gguf` is the *file format* layer - mmap, header parse, KV map,
//! and per-tensor dequantization of every F32/F16/BF16 and legacy/k-quant
//! block type. This crate is the layer above it: turning a file into a
//! **model**. Three things are shared, and none of them are per-architecture:
//!
//! - [`kv::ArchKv`] - GGUF's standardized `{arch}.{suffix}` hyperparameter
//!   convention read as a declarative list, with the raw file one method call
//!   away for everything that does not follow the convention.
//! - [`import`] - the streaming import driver: classify, dequantize one tensor
//!   at a time, write, and prove two-way coverage (nothing planned missing,
//!   nothing in the source unaccounted for).
//! - [`leaf`] - llama.cpp's own per-block leaf-name vocabulary (GQA, dense
//!   FFN, MoE FFN, Gated-DeltaNet/SSM), shared by every decoder-LM importer so
//!   `attn_q.weight`/`ffn_gate.weight`/`ssm_alpha.weight` are spelled once,
//!   not re-transcribed per model.
//! - [`int8_direct`] - a Q8_0 tensor straight into brain's packed-int8
//!   layout as a byte repack (no dequantize-then-requantize), now that
//!   `model::int8::GROUP` matches Q8_0's own block size.
//! - [`kquant`] - the six-format generalization of [`int8_direct`]:
//!   Q4_K/Q5_K/Q6_K/Q5_0/Q4_0/Q8_0 straight into brain's canonical device
//!   K-quant layout (packed codes plus interleaved per-group scale/min),
//!   with no dequantize-then-requantize detour for any of them.
//! - [`route`] - "which model is this file", answered once for every consumer:
//!   `general.architecture` resolved against the canonical architecture
//!   registry, plus the secondary `clip.projector_type` discriminator that a
//!   multimodal projector needs (every mmproj file declares `clip`). It
//!   resolves the architecture; the table of what to DO with each one lives
//!   with the model crates, in `cli::gguf_import`, because that is the only
//!   layer that can see them.
//!
//! A model supplies only its own decisions: a hyperparameter struct with a
//! `param_list`, and a tensor-name classifier. [`deepseek_ocr`] (a decoder)
//! and [`deepseek_ocr_vision`] (a four-stage vision tower) are both of those,
//! in full, in a few hundred lines each; `qwen35moe` keeps its own config in
//! its own crate and calls [`import`] directly.

pub mod deepseek_ocr;
pub mod deepseek_ocr_vision;
pub mod import;
pub mod int8_direct;
pub mod kquant;
pub mod kv;
pub mod leaf;
pub mod route;

pub use import::{ImportStats, Mapped};
pub use int8_direct::try_i8_rect;
pub use kquant::{try_kq_rect, KqLayout};
pub use kv::{architecture, ArchKv};
pub use leaf::{role, Role};
pub use route::{route, route_path, Route};
