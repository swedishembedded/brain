// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLIP-family text and image encoders: CLIP-L, OpenCLIP-bigG and EVA-CLIP
//! behind one config-driven graph.
//!
//! * [`config`] - the reference configurations and canonical tensor manifests.
//! * [`import`] - checkpoint import with two-way coverage validation.
//! * [`init`] - random init for the towers (tests / gradient checks only).
//! * [`model`] - the graphs (`ClipText`, `EvaVision`, `ClipVision`).
//!
//! Three graphs, and the reason there are three rather than one is worth
//! stating: `ClipText` is causal and token-indexed; `EvaVision` is EVA02, whose
//! `inner_attn_ln` and SwiGLU-with-`ffn_ln` sublayers `model::vit`'s block
//! builder cannot express; `ClipVision` is a **vanilla** pre-LN ViT, which that
//! builder expresses exactly - so it composes `model::vit::vit_block_fwd_cached`
//! / `vit_block_bwd` and adds no second block graph.
//!
//! `ClipVision` is also where the **`PatchSource` seam** lives: DeepSeek-OCR
//! injects its SAM branch's compressed feature map as CLIP's patch tokens,
//! bypassing the conv patch embedding entirely, and that bypass is a first-class
//! (and bit-identity-tested) API rather than a per-model branch.
//!
//! The **text** and **vanilla-CLIP image** towers are trainable:
//! `ClipText::new_train_on` / `ClipVision::new_train_on` add the reverse pass
//! over the same forward. The EVA image tower is still forward-only, as are the
//! `capability` Provider / residency adapter / D-Bus surface and the CLIP BPE
//! tokenizer (which belongs in `crates/data` next to the GPT-2 and Qwen BPEs,
//! not here).

pub mod caps;
pub mod config;
pub mod import;
pub mod init;
pub mod model;
