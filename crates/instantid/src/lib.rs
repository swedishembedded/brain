// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! InstantID: SDXL + a face-keypoint ControlNet + IP-Adapter-FaceID decoupled
//! cross-attention keyed by the ArcFace embedding.
//!
//! ```text
//!   photo ─► facenet (SCRFD + align + ArcFace) ─► 512-d ─► Resampler ─► 16 x 2048 ID tokens
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
//! `crates/unet` (165 comparisons, cosine 0.9999999999) and the face embedding is
//! `crates/facenet` (cosine 1.0000000).
//!
//! ## Status: shapes and import only
//!
//! What is here and gated: [`config`] derives every shape from the released
//! checkpoint rather than assuming it, and validates that each cross-attention
//! site carries BOTH `to_k_ip` and `to_v_ip`.
//!
//! What is **not** here, and not claimed: the resampler forward, the decoupled
//! attention itself, and the wiring into `crates/unet`'s attention sites. The
//! reference activations are already dumped
//! (`tools/goldens/instantid_dump_reference.py` -> `testdata/instantid/`, 21 tensors
//! covering `proj_in`, every layer's attention and feed-forward, `proj_out`,
//! `norm_out`, plus both site widths), so the forward has a ladder to climb the
//! moment it lands.
//!
//! ## The forward should REUSE PuLID's Perceiver emitter, not copy it
//!
//! `crates/pulid`'s `IdFormer` already records exactly this block: it writes
//! `cat(norm1(ctx), norm2(latents))` into ONE buffer at row offsets, runs `to_q`
//! over the latent rows and `to_kv` over the whole buffer, and splits the result
//! by row thirds — the same `PerceiverAttention` from the same IP-Adapter
//! lineage. InstantID's Resampler is that block with a single-token context
//! (`x = proj_in(arcface)`) and without PuLID's five EVA scale-mappings.
//!
//! So the next step is to hoist that emitter into a shared home and have both
//! crates call it. Writing a second Perceiver here would be exactly the
//! duplication AGENTS.md's one-implementation rule exists to prevent — and the
//! rmsnorm-seven-times precedent it cites.

pub mod config;
pub mod import;
pub mod model;
