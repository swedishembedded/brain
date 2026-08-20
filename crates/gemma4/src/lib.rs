// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gemma-4 (unified text tower) - the text encoder LTX-2.5 conditions its
//! DiT on. The real checkpoint
//! (`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, 12B params / 26 GB
//! bf16) is a UNIFIED text+vision+audio model, but LTX-2.5 only ever uses it
//! as a TEXT encoder - this crate implements ONLY the text-only forward path
//! through the decoder-layer stack (no vision tower, no audio tower, no
//! image/video/audio token handling).
//!
//! This crate follows the exact pattern `crates/ltxv`'s tiny-config DiT
//! uses: a real reference implementation
//! (`transformers.models.gemma4_unified`, ported faithfully at TINY dims,
//! every real-LTX-2.5-config FLAG set correctly, gated by parity against
//! goldens dumped from the real Python reference at that same tiny size
//! (`tools/goldens/gemma4_dump_reference.py`). This crate proves the OP
//! SEQUENCE (the 5:1 sliding/full attention alternation, the dual RoPE
//! bases, the `attention_k_eq_v` global-layer variant, the 49-hidden-state
//! aggregate-embed projection) at tiny scale; [`import`] adds the real
//! 12B/686-tensor checkpoint's config and two-way tensor-coverage import
//! (shape-level, off the checkpoint's own header - real-WEIGHT per-stage
//! forward parity at full scale remains a tracked gap, see that module's
//! doc and this crate's roadmap entry for why).
//!
//! - [`config`] - every FLAG that changes the op sequence, tiny + real dims.
//! - [`rope`] - the two RoPE table constructions (sliding/`default`,
//!   full/`proportional`) and which existing kernel each reuses.
//! - [`block`] - one decoder layer's forward as a device kernel graph (both
//!   layer types, the `attention_k_eq_v` variant, the two kernel-contract
//!   facts found while wiring existing kernels to this architecture).
//! - [`model`] - the full model forward + the LTX-specific aggregate-embed
//!   projection.
//! - [`import`] - the real checkpoint's tensor manifest and two-way-coverage
//!   import (`model.`-prefix strip, out-of-scope tower / asset-blob
//!   routing).
//! - [`tokenizer`] - extracting the real checkpoint's embedded
//!   `tokenizer_json` byte tensor and parsing it as a BPE tokenizer.

pub mod block;
pub mod config;
pub mod import;
pub mod model;
pub mod rope;
pub mod tokenizer;

pub use config::{Gemma4Config, LayerType};
pub use import::{gemma4_tensor_manifest, import_gemma4};
pub use model::{load_tiny_weights, AggregateEmbed, Gemma4Model, Gemma4Output};
pub use tokenizer::{extract_tokenizer_json_bytes, load_tokenizer};
