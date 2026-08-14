// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Building `id_cond` from a photograph — the wiring that makes PuLID an
//! operation on an image rather than on a golden.
//!
//! Until this module existed `crates/pulid` declared the face crate as a
//! dependency and never referenced it: `tests/parity.rs` fed the ArcFace half of
//! `id_cond` straight from the dumped reference, so the crate gated the
//! *resampler* while the path from a face photo to its ID vector was untested.
//!
//! ```text
//!   photo ─► arcface: SCRFD detect ─► largest face ─► 5-point align ─► IResNet
//!                                                                        │
//!                                                        RAW 512-d ──────┤
//!                                                                        ├─► id_cond [1280]
//!   photo ─► clip::EvaVision ─► cls ─► L2 normalise ───── 768-d ─────────┘
//! ```
//!
//! # The asymmetry is the whole point
//!
//! **The ArcFace half is NOT normalised and the EVA-CLIP half IS.** PuLID's
//! `pipeline_flux.py::get_id_embedding` reads insightface's
//! `face_info['embedding']` — the raw network output, not `normed_embedding` —
//! and then divides *only* `id_cond_vit` by its own norm before
//! `torch.cat([id_ante_embedding, id_cond_vit])`.
//!
//! This is easy to get wrong in exactly one direction, because brain's own
//! `arcface` **`embed` action normalises** (its output is meant to be
//! cosine-ready). Feeding that into `id_cond` would leave the first 512
//! components ~20x too small and let the EVA half dominate the conditioning —
//! with nothing to catch it, since the shape and the dtype would both be right.
//! The dumped reference says so numerically: `‖id_cond[:512]‖ = 20.11` against
//! `‖id_cond[512:]‖ = 1.0000`. [`arcface::caps::ArcFaceSession::embed_raw_chw`] exists
//! so both consumers share one implementation and each applies its own
//! convention.
//!
//! # What this does NOT do
//!
//! The EVA-CLIP half still takes whatever image the caller supplies. The
//! reference preprocesses it with facexlib's RetinaFace alignment plus a BiSeNet
//! face parse (background whitened, face greyscaled) — two models brain does not
//! have — so [`IdCond::from_image`] takes the already-prepared EVA input as a
//! separate argument rather than pretending to reproduce it. The ArcFace half
//! needs none of that: PuLID calls insightface antelopev2, which *is*
//! `crates/arcface` + `crates/scrfd`.

use crate::config::PulidConfig;

/// The ArcFace half's width, and the offset at which the EVA-CLIP half starts.
pub const ARCFACE_DIM: usize = 512;

/// A composed `id_cond`, ready for [`crate::model::IdFormer::set_inputs`].
pub struct IdCond {
    /// `ArcFace(raw) ‖ EvaClip(L2-normalised)`, length [`PulidConfig::id_cond_dim`].
    pub cond: Vec<f32>,
    /// The detected face the ArcFace half came from, when detection ran.
    pub face: Option<arcface::Face>,
}

/// Compose `id_cond` from a raw ArcFace embedding and an EVA-CLIP CLS vector.
///
/// The EVA half is L2-normalised here; the ArcFace half is passed through
/// untouched. Both conventions are the reference's — see the module docs.
pub fn compose(cfg: &PulidConfig, arcface_raw: &[f32], eva_cls: &[f32]) -> Result<Vec<f32>, String> {
    if arcface_raw.len() != ARCFACE_DIM {
        return Err(format!("pulid: ArcFace embedding is {} wide, expected {ARCFACE_DIM}", arcface_raw.len()));
    }
    let want_vit = cfg.id_cond_dim - ARCFACE_DIM;
    if eva_cls.len() != want_vit {
        return Err(format!("pulid: EVA-CLIP cls is {} wide, expected {want_vit}", eva_cls.len()));
    }
    let mut cond = Vec::with_capacity(cfg.id_cond_dim);
    cond.extend_from_slice(arcface_raw); // raw, deliberately
    cond.extend_from_slice(&model::hostmath::l2_normalize(eva_cls));
    Ok(cond)
}

impl IdCond {
    /// Run the ArcFace half on a photograph and join it to a supplied EVA-CLIP
    /// CLS vector.
    ///
    /// `chw` is source-resolution CHW RGB in `[0,1]` (brain's wire convention).
    /// The largest detected face is used, which is the reference's primary-face
    /// rule (`sorted by box area, take the last`).
    pub fn from_image(
        cfg: &PulidConfig,
        face_session: &arcface::caps::ArcFaceSession,
        chw: &[f32],
        w: u32,
        h: u32,
        eva_cls: &[f32],
    ) -> Result<IdCond, String> {
        let (raw, face) = face_session.embed_raw_chw(chw, w, h, true, true)?;
        Ok(IdCond { cond: compose(cfg, &raw, eva_cls)?, face })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_eva_half_is_normalised_and_the_arcface_half_is_not() {
        // The asymmetry this module exists to preserve. A `compose` that
        // normalised both (or neither) would still produce a 1280-vector of the
        // right dtype, which is why it is asserted rather than assumed.
        let cfg = PulidConfig::v0_9_1();
        let arc: Vec<f32> = (0..ARCFACE_DIM).map(|i| (i % 7) as f32 + 1.0).collect();
        let eva: Vec<f32> = (0..cfg.id_cond_dim - ARCFACE_DIM).map(|i| (i % 5) as f32 + 1.0).collect();
        let c = compose(&cfg, &arc, &eva).expect("compose");
        assert_eq!(c.len(), cfg.id_cond_dim);

        let n_arc: f32 = c[..ARCFACE_DIM].iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_eva: f32 = c[ARCFACE_DIM..].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n_eva - 1.0).abs() < 1e-5, "EVA half must be L2-normalised, got {n_eva}");
        assert!(n_arc > 2.0, "ArcFace half must be passed through RAW, got norm {n_arc}");
        assert_eq!(&c[..ARCFACE_DIM], &arc[..], "ArcFace half must be byte-identical to its input");
    }

    #[test]
    fn a_wrong_width_is_named_not_padded() {
        let cfg = PulidConfig::v0_9_1();
        let e = compose(&cfg, &[0.0; 128], &[0.0; 768]).unwrap_err();
        assert!(e.contains("512"), "error should name the expected width, got: {e}");
    }
}
