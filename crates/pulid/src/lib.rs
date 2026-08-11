// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PuLID identity conditioning for FLUX.1.
//!
//! PuLID makes "the same person" a model operation rather than a prompt: a face
//! photo becomes 32 ID tokens, and those tokens are cross-attended into the
//! FLUX.1 image stream at 20 points along the backbone.
//!
//! ```text
//!   face image ──► facenet (SCRFD + align + ArcFace)  ──► 512-d embedding ─┐
//!              └─► clip::EvaVision (EVA-CLIP-L/336)   ──► cls (L2-normed) ─┤ id_cond [1280]
//!                                                     └─► 5 tapped hidden states
//!                                                                          │
//!                              IdFormer  ◄────────────────────────────────┘
//!                                  │  32 x 2048 ID tokens
//!                                  ▼
//!   FLUX.1 double block i ──► (i % 2 == 0) img += id_weight · PulidCa[k](id, img)
//!   FLUX.1 single block i ──► (i % 4 == 0) img += id_weight · PulidCa[k](id, img)
//! ```
//!
//! **Nothing here re-implements a model brain already has.** The ArcFace half is
//! `crates/facenet` (parity-gated at cosine 1.0000000) and the image tower is
//! `clip::model::EvaVision` (0.99999999), which already exposes exactly the
//! taps PuLID needs (`clip::EvaVisionConfig::PULID_TAPS`). This crate adds the
//! two modules that do not exist anywhere else — the `IDFormer` resampler and
//! the injected `PerceiverAttentionCA` — plus the wiring, and **no kernel**.
//!
//! ## What is gated, and what is not
//!
//! Gated against `pulid_flux_v0.9.1` reference goldens
//! (`tools/goldens/pulid_dump_reference.py`, replayed by `tests/parity.rs`):
//!
//! * the **ID embedding pipeline**, stage by stage — `id_map`, each of the 5
//!   scale mappings, all 10 resampler layers, and the projected ID tokens,
//!   from an ArcFace embedding and EVA-CLIP hidden states that brain's own
//!   parity-gated crates produce;
//! * the **injected cross-attention**, stage by stage, on real weights;
//! * **one conditioned transformer evaluation** — the reduced-depth FLUX.1
//!   backbone with the ID injected, against a golden dumped from the same
//!   truncation, which the dumper self-validates against `crates/flux1`'s own
//!   `dit_small` golden before injecting anything.
//!
//! **Not** gated, and not claimed:
//!
//! * **end-to-end generation.** `crates/flux1` has no sampler loop and no VAE
//!   glue, so "generate a face" cannot be run at all, let alone gated.
//! * **full-depth conditioning.** The fp32 FLUX.1 backbone is 47.6 GiB and does
//!   not fit one 24 GiB card, so the
//!   conditioned gate runs at reduced depth, exactly as flux1's own fp32 gate
//!   does. The 20-site full-depth schedule is gated as a *schedule*
//!   ([`PulidConfig::schedule`]), not as a forward.
//! * **the PuLID image preprocessing.** The reference builds the EVA-CLIP input
//!   with facexlib's RetinaFace alignment and a BiSeNet face parse (background
//!   whitened, face greyscaled) — two models brain does not have. The ArcFace
//!   half needs none of it: PuLID calls insightface antelopev2, which IS
//!   `crates/facenet`.
//! * backward / gradcheck, INT8, the serving contract (no capability manifest,
//!   no residency adapter, no D-Bus surface) — all follow-ups.

pub mod adapter;
pub mod config;
pub mod idcond;
pub mod import;
pub mod model;

pub use adapter::PulidAdapter;
pub use config::{PulidConfig, Site, Stream};
pub use import::{import, read, PulidWeights, Tensors};
pub use model::{joint_kernels, IdFormer, PulidCa, KERNELS};
