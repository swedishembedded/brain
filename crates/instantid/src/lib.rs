// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! InstantID: SDXL + a face-keypoint ControlNet + IP-Adapter-FaceID decoupled
//! cross-attention keyed by the ArcFace embedding.
//!
//! ```text
//!   photo ─► scrfd detect + arcface align/embed ─► 512-d ─► Resampler ─► 16 x 2048 ID tokens
//!                                                                          │
//!   photo ─► 5 face keypoints ─► crates/controlnet ─► residuals ─► SDXL UNet
//!                                                                  │  every attn2 site:
//!                                                                  └─ hidden = text_attn
//!                                                                       + scale * ip_attn(ID)
//! ```
//!
//! **Nothing here re-implements a model brain already has.** The ControlNet half
//! is `crates/controlnet` — whose SDXL implementation was imported from *this*
//! release and is parity-gated at 140 comparisons — the backbone is
//! `crates/sdxlunet` (165 comparisons, cosine 0.9999999999) and the face embedding is
//! `crates/arcface` (cosine 1.0000000).
//!
//! ## What is here
//!
//! [`config`] derives every shape from the released checkpoint rather than
//! assuming it, and validates that each cross-attention site carries BOTH
//! `to_k_ip` and `to_v_ip`. [`model`] holds the resampler and the decoupled
//! site attention, wired into `crates/sdxlunet`'s cross-attention sites
//! through `model::attninject::CrossAttnInject`. The resampler is PuLID's
//! Perceiver block with a single-token context, built from the same shared
//! emitter, `model::rowemit`, rather than a second copy.

pub mod config;
pub mod import;
pub mod model;
