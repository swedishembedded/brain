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
//! - [`registry`] - dispatch on `general.architecture`, plus a secondary
//!   discriminator where the architecture alone is ambiguous (every mmproj
//!   file declares `clip`).
//!
//! A model supplies only its own decisions: a hyperparameter struct with a
//! `param_list`, and a tensor-name classifier. [`deepseek_ocr`] (a decoder)
//! and [`deepseek_ocr_vision`] (a four-stage vision tower) are both of those,
//! in full, in a few hundred lines each; `qwen35moe` keeps its own config in
//! its own crate and calls [`import`] directly.

pub mod deepseek_ocr;
pub mod deepseek_ocr_vision;
pub mod import;
pub mod kv;
pub mod registry;

pub use import::{ImportStats, Mapped};
pub use kv::{architecture, ArchKv};
pub use registry::{import_gguf, ArchEntry, ARCHITECTURES};
