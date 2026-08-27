// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaVA-1.5-13B - a CLIP-L/14@336 vision tower (penultimate layer, patch
//! features only, CLS dropped) spliced into a Vicuna-1.5 (LLaMA-2 13B) decoder
//! by a two-layer `mm_projector`, over the `vicuna_v1` conversation template.
//!
//! This is the id-reservation crate, not the port. Registering an
//! architecture before any of its code exists fixes the crate directory
//! name, the package name, and the CLI word in one place before anything
//! else is written. Brought into this workspace as `crates/supir`'s optional
//! image captioner - SUPIR's reference pipeline auto-captions the
//! low-quality image with LLaVA to build the restoration prompt, but an
//! empty caption entirely replaced by a user-supplied prompt is a supported
//! upstream path, so `supir` depends on this crate only through
//! `capability::Registry`, never directly.
//!
//! ## Status
//!
//! Implementation has not started. The plan is a LLaMA byte-fallback BPE
//! tokenizer (a sibling of the existing CLIP BPE), a CLIP-L/14@336 vision
//! preset over the existing plain pre-LN ViT graph, a LLaMA-2 13B config
//! preset over the existing Qwen-shaped decoder graph (plain multi-head
//! attention, no QK-norm, no attention/MLP bias - a config this decoder
//! already supports, not new capability), then this crate's own
//! `mm_projector` and token splice, two-way import, a parity gate, INT8
//! (fp32 13B is around 52 GB), and the serving contract's `caption` action.
