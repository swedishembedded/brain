// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ArcFace: the insightface `antelopev2` IResNet-100 identity **embedding**,
//! plus the 5-point similarity alignment that feeds it - the visual analogue of
//! `crates/ecapatdnn`.
//!
//! Same shape of thing as the ECAPA-TDNN speaker encoder: an **embedding model
//! consumed by a generative one**. `ecapatdnn` turns a waveform into a 1024-d
//! voice vector for Qwen3-TTS; this turns a photo into a 512-d identity vector
//! for the identity-preserving image pipeline (PuLID / InstantID). Hence the
//! same layout: `config` / `import` / `model` + a parity test replaying dumped
//! reference goldens.
//!
//! # Pipeline
//!
//! ```text
//! image (BGR u8, HWC)
//!   -> scrfd detect            the primary face's 5 landmarks (crates/scrfd)
//!   -> align::norm_crop_chw    Umeyama solve (host) + `grid_sample` (device)
//!   -> preprocess              (112x112, this model's normalisation - std 127.5)
//!   -> ArcFace::embed_blob     raw 512-d
//!   -> hostmath::l2_normalize  for cosine
//! ```
//!
//! The detection step is `crates/scrfd`, a separate crate and a separately
//! served model: the detector is useful on its own and knows nothing about
//! embeddings, while this direction of the dependency is real - the default
//! `embed` path cannot align a face it has not found. `align = false` skips it
//! entirely for an already-aligned crop.
//!
//! # Scope
//!
//! The forward port (goldens, import, stage parity) plus the TRAINING half:
//! [`train::ArcFaceTrainer`] is the IResNet embedding backbone with an
//! additive-angular-margin head and a hand-written device backward, gated by
//! `gradcheck::check_arcface`. Training covers the **embedding backbone only** -
//! detection and the alignment warp are preprocessing and carry no recognition
//! gradient, which is how the reference recipe trains too.
//!
//! The serving contract is met by [`caps`] (the `embed`
//! `capability::Provider`), the CLI's residency adapter (`BRAIN_ARCFACE_DIR`)
//! and `examples/vision/`.
//!
//! # Two normalisations, one letter apart
//!
//! ArcFace divides by **127.5**; SCRFD, in the sibling crate, by **128.0**. They
//! are both "map u8 to roughly [-1, 1]" and they are not the same function. Each
//! model's constant lives in its own [`config::Preprocess`], in its own crate,
//! and is never defaulted - which is why this crate keeps its own preprocessing
//! even though it depends on the detector's.

pub mod align;
pub mod caps;
pub mod config;
pub mod import;
pub mod model;
pub mod train;

pub use align::{estimate_norm, norm_crop_chw, warp_grid};
pub use config::{ArcFaceConfig, Preprocess, ARCFACE_DST_112};
/// The detected face the aligned `embed` path took its crop from - the
/// detector's own type, re-exported so a consumer of this crate's embedding
/// does not have to depend on the detector crate to name it.
pub use scrfd::Face;
pub use import::{import_arcface, import_dir};
pub use model::{ArcFace, ArcFaceTaps, PIPELINES};
pub use onnx::walk::Tensors;
pub use train::{ArcFaceTrainConfig, ArcFaceTrainer};

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
    let x = ctx.upload("arcface.blob.in", &chw);
    let n = imaging::Normalization { mean: [pre.mean; 3], std: [pre.std; 3] };
    let y = ctx.normalize(&x, shape, &n);
    ctx.download(&y, shape.numel())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blob layout must be NCHW-RGB with the embedder's own mean/std -
    /// 127.5, not the detector's 128.0, which is the one constant a copy-paste
    /// gets wrong and nothing else catches.
    #[test]
    fn blob_is_nchw_rgb_with_the_embedders_own_normalisation() {
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
        let d = (30.0 - 127.5) / 128.0;
        assert!((b[0] - d).abs() > 1e-5, "the two normalisations must differ");
    }
}
