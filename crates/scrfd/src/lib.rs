// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SCRFD-10GF: the insightface `antelopev2` face **detector** - boxes, scores
//! and five landmarks per face.
//!
//! # Pipeline
//!
//! ```text
//! image (BGR u8, HWC)
//!   -> preprocess              (host layout permute + a `film_chan` dispatch)
//!   -> Scrfd::forward          9 raw head tensors
//!   -> detect::decode          anchors + distance decode + NMS -> Face
//! ```
//!
//! # Scope
//!
//! Inference only: the released graph is a forward artifact and detection
//! carries no recognition gradient, so there is no detector backward here. The
//! landmarks this emits are the input to the ArcFace 5-point alignment
//! (`crates/arcface`, which depends on this crate for exactly that step) - the
//! two are pipeline siblings, not one model: a detector is useful on its own,
//! and each has its own weights file, its own normalisation and its own served
//! model id.
//!
//! The serving contract is met by [`caps`] (the `detect`
//! `capability::Provider`), the CLI's residency adapter (`BRAIN_SCRFD_DIR`) and
//! `examples/vision/`.
//!
//! # Two normalisations, one letter apart
//!
//! SCRFD divides by **128.0**; ArcFace, in the sibling crate, by **127.5**. They
//! are both "map u8 to roughly [-1, 1]" and they are not the same function. Each
//! model's constant lives in its own [`config::Preprocess`], in its own crate,
//! and is never defaulted - which is why this crate does not import the other's
//! preprocessing and the other does not import this one's.

pub mod caps;
pub mod config;
pub mod detect;
pub mod import;
pub mod model;

pub use config::{Preprocess, ScrfdConfig};
pub use detect::{decode, nms, Face};
pub use import::{import_dir, import_scrfd};
pub use model::{Scrfd, ScrfdTaps, PIPELINES};
pub use onnx::walk::Tensors;

use gpu_core::Gpu;

/// `cv2.dnn.blobFromImage(bgr_u8, 1/std, size, (mean,)*3, swapRB=True)`:
/// interleaved **BGR** u8 → NCHW **RGB** f32 `(x - mean) / std`.
///
/// Split by the `crates/imaging` rule:
///   * the BGR→RGB swap and HWC→CHW are **layout permutation** - host glue, via
///     `imaging::pixels::hwc_to_chw`;
///   * `(x - mean)/std` over every pixel is **per-pixel arithmetic** - a kernel,
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
    let x = ctx.upload("scrfd.blob.in", &chw);
    let n = imaging::Normalization { mean: [pre.mean; 3], std: [pre.std; 3] };
    let y = ctx.normalize(&x, shape, &n);
    ctx.download(&y, shape.numel())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blob layout must be NCHW-RGB with the detector's own mean/std - 128.0,
    /// not the embedder's 127.5, which is the one constant a copy-paste gets
    /// wrong and nothing else catches.
    #[test]
    fn blob_is_nchw_rgb_with_the_detectors_own_normalisation() {
        let gpu = gpu_core::testgpu::dev(PIPELINES);
        // 1x2 image, pixel0 = BGR(10, 20, 30), pixel1 = BGR(40, 50, 60)
        let bgr = [10u8, 20, 30, 40, 50, 60];
        let pre = ScrfdConfig::scrfd_10g_bnkps().pre;
        let b = blob_from_bgr_u8(&gpu, &bgr, 1, 2, &pre);
        assert_eq!(b.len(), 6);
        // R plane first: pixel0.R = 30, pixel1.R = 60
        let f = |v: f32| (v - 127.5) / 128.0;
        for (got, want) in b.iter().zip([f(30.0), f(60.0), f(20.0), f(50.0), f(10.0), f(40.0)]) {
            assert!((got - want).abs() < 1e-5, "{b:?}");
        }
    }
}
