// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaVA-1.5-13B - a CLIP-L/14@336 vision tower spliced into a Vicuna-1.5
//! (LLaMA-2 13B) decoder by a two-layer `mlp2x_gelu` projector.
//!
//! `crates/supir`'s optional captioner: `--no_llava` (an empty caption,
//! replaced by a user-supplied prompt) is a supported upstream SUPIR path, so
//! nothing here ever touches the diffusion graph - this crate only ever
//! emits a string, through [`captioner::Captioner`] and the
//! [`capability::Provider`] in [`caps`], the same seam `crates/fastvlm` and
//! `crates/qwen3vl` satisfy today. `crates/supir` links none of this
//! directly; a caller composes it through a [`capability::Registry`]
//! (the `imgpipe` pattern - see that crate's `caps.rs`).
//!
//! Reuse plan, closely mirroring `crates/fastvlm`: the vision tower is the
//! EXISTING `clip::model::ClipVision` (no new vision graph - `config`'s
//! `clip_l336()` preset), the decoder is the EXISTING `qwen3::Qwen` with
//! `enable_mm_splice`/`PrefillInput::Embed` (no new splice machinery - that
//! seam already generalises past FastVLM's own use of it), and `int8`
//! precision reuses `Qwen::new_shard_i8` as-is (no new quantization code).
//! What is genuinely new: the config presets, the `mlp2x_gelu` projector at
//! LLaVA's widths, the `vicuna_v1` template, the tokenizer
//! (`data::llama_bpe`, its own crate), and the import name mapping.

pub mod caps;
pub mod captioner;
pub mod config;
pub mod import;
pub mod model;
pub mod prompt;
pub mod template;

pub use config::LlavaConfig;
