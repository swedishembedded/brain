// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Face recognition: SCRFD detection, 5-point similarity alignment and the
//! ArcFace IResNet-100 embedding — the visual analogue of `crates/speaker`.
//!
//! Same shape of thing as the ECAPA-TDNN speaker encoder: an **embedding model
//! consumed by a generative one**. `speaker` turns a waveform into a 1024-d
//! voice vector for Qwen3-TTS; this turns a photo into a 512-d identity vector
//! for the identity-preserving image pipeline (PuLID / InstantID). Hence the
//! same layout: `config` / `import` / `model`
//! + a parity test replaying dumped reference goldens.
//!
//! # Pipeline
//!
//! ```text
//! image (BGR u8, HWC)
//!   -> preprocess              (host layout permute + a `film_chan` dispatch)
//!   -> Scrfd::forward          9 raw head tensors
//!   -> detect::decode          anchors + distance decode + NMS -> Face
//!   -> align::norm_crop_chw    Umeyama solve (host) + `grid_sample` (device)
//!   -> preprocess              (112x112, the OTHER normalisation — std 127.5)
//!   -> ArcFace::embed_blob     raw 512-d
//!   -> hostmath::l2_normalize  for cosine
//! ```
//!
//! # Scope
//!
//! The forward port (goldens, import, stage parity) plus the ArcFace TRAINING
//! half: [`train::ArcFaceTrainer`] is the IResNet embedding backbone with an
//! additive-angular-margin head and a hand-written device backward, gated by
//! `gradcheck::check_arcface`. Training covers the **embedding backbone only** —
//! SCRFD detection and the alignment warp are preprocessing and carry no
//! recognition gradient, which is how the reference recipe trains too.
//!
//! The serving contract is met by [`caps`] (the `detect`/`embed`
//! `capability::Provider`), `crates/cli/src/resident_facenet.rs` (the residency
//! adapter, `BRAIN_FACENET_DIR`) and `examples/vision/` — see
//! `.agents/rules/serving-contract.md`.
//!
//! # Two normalisations, one letter apart
//!
//! ArcFace divides by **127.5**, SCRFD by **128.0**. They are both "map u8 to
//! roughly [-1, 1]" and they are not the same function. Each model's constant
//! lives in its own [`config::Preprocess`] and is never defaulted.

pub mod align;
pub mod caps;
pub mod config;
pub mod detect;
pub mod import;
pub mod model;
pub mod train;

pub use align::{estimate_norm, norm_crop_chw, warp_grid};
pub use config::{ArcFaceConfig, Preprocess, ScrfdConfig, ARCFACE_DST_112};
pub use detect::{decode, nms, Face};
pub use import::{import_arcface, import_dir, import_scrfd, Tensors};
pub use model::{ArcFace, ArcFaceTaps, Scrfd, ScrfdTaps, PIPELINES};
pub use train::{ArcFaceTrainConfig, ArcFaceTrainer};

use gpu_core::Gpu;

/// `cv2.dnn.blobFromImage(bgr_u8, 1/std, size, (mean,)*3, swapRB=True)`:
/// interleaved **BGR** u8 → NCHW **RGB** f32 `(x - mean) / std`.
///
/// Split by the `crates/imaging` rule:
///   * the BGR→RGB swap and HWC→CHW are **layout permutation** — host glue, via
///     `imaging::pixels::hwc_to_chw`;
///   * `(x - mean)/std` over every pixel is **per-pixel arithmetic** — a kernel,
///     dispatched through `imaging::Ctx::normalize`, which is brain's one
///     `film_chan` seam. No host normalisation loop exists in this crate.
///
/// `gpu` must have `film_chan` registered (it is in [`PIPELINES`]).
pub fn blob_from_bgr_u8(gpu: &Gpu, bgr: &[u8], h: u32, w: u32, pre: &Preprocess) -> Vec<f32> {
    assert_eq!(bgr.len(), (h * w * 3) as usize, "blob_from_bgr_u8: expected HWC u8 [{h},{w},3]");
    // swapRB: reorder the interleaved triples, then permute HWC -> CHW.
    let hwc: Vec<f32> = if pre.swap_rb {
        bgr.chunks_exact(3).flat_map(|p| [p[2] as f32, p[1] as f32, p[0] as f32]).collect()
    } else {
        bgr.iter().map(|&v| v as f32).collect()
    };
    let chw = imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize);

    let ctx = imaging::Ctx::new(gpu);
    let shape = imaging::Shape::new(1, 3, h, w);
    let x = ctx.upload("facenet.blob.in", &chw);
    let n = imaging::Normalization { mean: [pre.mean; 3], std: [pre.std; 3] };
    let y = ctx.normalize(&x, shape, &n);
    ctx.download(&y, shape.numel())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blob layout must be NCHW-RGB with the model's own mean/std. Run
    /// against the two real configs so a swapped constant shows up here.
    #[test]
    fn blob_is_nchw_rgb_with_the_models_own_normalisation() {
        let gpu = gpu_core::testgpu::dev(PIPELINES);
        // 1x2 image, pixel0 = BGR(10, 20, 30), pixel1 = BGR(40, 50, 60)
        let bgr = [10u8, 20, 30, 40, 50, 60];
        let pre = ArcFaceConfig::iresnet100().pre;
        let b = blob_from_bgr_u8(&gpu, &bgr, 1, 2, &pre);
        assert_eq!(b.len(), 6);
        // R plane first: pixel0.R = 30, pixel1.R = 60
        let f = |v: f32| (v - 127.5) / 127.5;
        for (got, want) in b.iter().zip([f(30.0), f(60.0), f(20.0), f(50.0), f(10.0), f(40.0)]) {
            assert!((got - want).abs() < 1e-5, "{b:?}");
        }
        // the detector's std differs and must not be reused
        let d = blob_from_bgr_u8(&gpu, &bgr, 1, 2, &ScrfdConfig::scrfd_10g_bnkps().pre);
        assert!((d[0] - (30.0 - 127.5) / 128.0).abs() < 1e-5, "{d:?}");
        assert!((d[0] - b[0]).abs() > 1e-5, "the two normalisations must differ");
    }
}
