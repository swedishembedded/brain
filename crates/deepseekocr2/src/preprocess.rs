// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real images in: decoded RGB pixels of any extent become the normalized
//! `[3, S, S]` tensor [`crate::encoder::sam_tokens_from_nchw`]'s SAM half
//! consumes, `S` = the SAM tower's own square (1024 for the shipped preset).
//!
//! **Global view only** - the same scope M6's real-weight tests settled on:
//! local-tile SAM inference needs a position-embedding resample
//! `crates/sam1` does not implement yet, so a document wider or taller than
//! the model's tiling threshold is downscaled into the one global view
//! rather than tiled. A caller that needs the full multi-tile "Gundam" path
//! is blocked on that SAM gap, not on this module.
//!
//! The normalization is the checkpoint's own, confirmed identical to v1's:
//! `clip.vision.image_mean = clip.vision.image_std = [0.5; 3]` (M0's ledger),
//! i.e. the plain `[0,1] -> [-1,1]` rescale `imaging::Normalization::HALF`
//! already names - not OpenAI CLIP's published constants, which this
//! checkpoint's Qwen2 tower never sees a pixel through anyway. Geometry is an
//! aspect-preserving centred fit with a mean-grey letterbox, matching every
//! DeepSeek-OCR-family preprocessor this repo has captured so far.

use gpu_core::Gpu;
use imaging::{AlignCorners, Border, Ctx, Filter, Normalization, Shape};

use crate::config::DeepseekOcr2VisionConfig;

/// Every kernel this module dispatches. A caller's `Gpu` needs this (or a
/// superset) registered before [`preprocess_image`] runs on it.
pub const PIPELINES: &[(&str, &str)] = &[
    ("resize_bilinear", kernels::RESIZE_BILINEAR),
    ("resize_bicubic", kernels::RESIZE_BICUBIC),
    ("film_chan", kernels::FILM_CHAN),
    ("pad2d", kernels::PAD2D),
    ("relu_inplace", kernels::RELU_INPLACE),
];

/// `[0,1] -> [-1,1]`, the shipped checkpoint's own `image_mean`/`image_std`.
pub const NORMALIZATION: Normalization = Normalization::HALF;

/// Half-pixel bicubic - brain's closest available filter to the reference
/// pipeline's Pillow bicubic resample; not a bit-exact match, and nothing in
/// this crate gates on it being one (no captured non-constant reference
/// exists for this checkpoint - see `tests/real_weight.rs`'s own header).
pub const FILTER: Filter = Filter::Bicubic;
pub const ALIGN: AlignCorners = AlignCorners::HalfPixel;

/// How the source rectangle lands inside the output square.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fit {
    /// Aspect-preserving, centred, mean-grey letterbox. Default.
    #[default]
    Pad,
    /// Non-aspect-preserving stretch to fill the whole square.
    Stretch,
}

/// Pure host geometry: the resized content's extent inside the square, and
/// the border around it (`Border::default()` under [`Fit::Stretch`]).
pub fn placement(fit: Fit, w: u32, h: u32, side: u32) -> (u32, u32, Border) {
    assert!(w > 0 && h > 0, "source image is {w}x{h}");
    assert!(side > 0, "target side is 0");
    if fit == Fit::Stretch {
        return (side, side, Border::default());
    }
    let scale = (side as f32 / w as f32).min(side as f32 / h as f32);
    let fw = ((w as f32 * scale).round() as u32).clamp(1, side);
    let fh = ((h as f32 * scale).round() as u32).clamp(1, side);
    // Half the remaining gap on each side, rounding a tie toward the lower
    // (top/left) edge, so an odd gap's extra pixel sits after the content.
    let (gw, gh) = (side - fw, side - fh);
    let (left, top) = (gw / 2, gh / 2);
    (fw, fh, Border { left, right: gw - left, top, bottom: gh - top })
}

/// Decoded `[h, w, 3]` RGB in `[0,1]` -> the model's `[3, S, S]` NCHW input.
///
/// `gpu` must have [`PIPELINES`] registered.
pub fn preprocess_image(gpu: &Gpu, cfg: &DeepseekOcr2VisionConfig, hwc: &[f32], w: u32, h: u32, fit: Fit) -> Vec<f32> {
    assert_eq!(cfg.sam.image_w(), cfg.sam.image_h(), "the global view's square is not square: {}x{}", cfg.sam.image_w(), cfg.sam.image_h());
    preprocess_to(gpu, hwc, w, h, cfg.sam.image_h(), fit)
}

/// [`preprocess_image`] with the output side given directly - the seam a
/// small-square unit test uses so a `1024²` device round trip is not paid
/// per assertion.
pub fn preprocess_to(gpu: &Gpu, hwc: &[f32], w: u32, h: u32, side: u32, fit: Fit) -> Vec<f32> {
    assert_eq!(hwc.len(), 3 * w as usize * h as usize, "expected [h, w, 3] = [{h}, {w}, 3] interleaved RGB in [0,1], got {} floats", hwc.len());
    let (fw, fh, border) = placement(fit, w, h, side);

    let chw = imaging::pixels::hwc_to_chw(hwc, 3, h as usize, w as usize);
    let ctx = Ctx::new(gpu);
    let src = ctx.upload("deepseekocr2.preprocess.src", &chw);
    let (resized, shape) = ctx.resize(&src, Shape::new(1, 3, h, w), fh, fw, FILTER, ALIGN);

    // A cubic kernel's negative lobes overshoot `[0,1]` at a hard edge - a
    // document scan's most common feature - so the output is clamped back to
    // the unit interval before normalizing, matching the 8-bit-saturated
    // resampler this checkpoint's own preprocessor is built on.
    let clamped = clamp01(&ctx, &resized, shape);
    let norm = ctx.normalize(&clamped, shape, &NORMALIZATION);

    // The border is filled AFTER normalizing: `pad_zero` can only fill zero,
    // and this checkpoint's mean IS zero post-affine, so the two coincide by
    // construction rather than by a special-cased pad value.
    let (out, out_shape) = if border == Border::default() { (norm, shape) } else { ctx.pad_zero(&norm, shape, border) };
    assert_eq!(out_shape, Shape::new(1, 3, side, side), "preprocess produced {out_shape:?}");
    ctx.download(&out, out_shape.numel())
}

/// `clamp(x, 0, 1)` via `1 - relu(1 - relu(x))` - `imaging` has no clamp
/// kernel of its own, and this composition of its two most general
/// elementwise primitives (affine, relu) needs no new one.
fn clamp01(ctx: &Ctx<'_>, x: &gpu_core::DeviceBuffer, s: Shape) -> gpu_core::DeviceBuffer {
    let relu = ctx.gpu.kernel_index("relu_inplace").expect("relu_inplace is not registered on this device");
    let run_relu = |b: &gpu_core::DeviceBuffer| {
        let n = s.numel();
        ctx.gpu.submit(&[], &[ctx.gpu.step(relu, &[b], &[n], n)]);
    };
    let lo = ctx.affine(x, s, &[1.0; 3], &[0.0; 3]);
    run_relu(&lo); // max(x, 0)
    let flipped = ctx.affine(&lo, s, &[-1.0; 3], &[1.0; 3]);
    run_relu(&flipped); // max(1 - max(x,0), 0)
    ctx.affine(&flipped, s, &[-1.0; 3], &[1.0; 3]) // 1 - that
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: u32 = 64;

    fn dev() -> Gpu {
        gpu_core::testgpu::dev(PIPELINES)
    }

    /// Four solid quadrants stay in their own corners, per channel - catches a
    /// transposed resize, a flipped axis or a CHW/HWC mixup at once.
    fn quadrants(side: u32, colors: [[f32; 3]; 4]) -> Vec<f32> {
        let half = side / 2;
        let mut v = Vec::with_capacity(3 * (side * side) as usize);
        for y in 0..side {
            for x in 0..side {
                let q = (y >= half) as usize * 2 + (x >= half) as usize;
                v.extend_from_slice(&colors[q]);
            }
        }
        v
    }
    fn at(out: &[f32], side: u32, c: u32, x: u32, y: u32) -> f32 {
        out[(c * side * side + y * side + x) as usize]
    }

    #[test]
    fn quadrants_survive_resize_and_normalization_in_place() {
        let colors = [[0.0, 0.2, 0.4], [1.0, 0.6, 0.8], [0.3, 1.0, 0.1], [0.7, 0.5, 0.9]];
        let src = 8 * SMALL;
        let out = preprocess_to(&dev(), &quadrants(src, colors), src, src, SMALL, Fit::Pad);
        assert_eq!(placement(Fit::Pad, src, src, SMALL).2, Border::default(), "a square source needs no border");
        let (q, e) = (SMALL / 4, SMALL / 4 + SMALL / 2);
        let (scale, shift) = NORMALIZATION.scale_shift();
        for (probe, idx) in [((q, q), 0usize), ((e, q), 1), ((q, e), 2), ((e, e), 3)] {
            for c in 0..3u32 {
                let want = colors[idx][c as usize] * scale[c as usize] + shift[c as usize];
                let got = at(&out, SMALL, c, probe.0, probe.1);
                assert!((got - want).abs() < 1e-4, "quadrant {idx} channel {c}: got {got}, want {want}");
            }
        }
    }

    #[test]
    fn a_non_square_source_is_letterboxed_with_the_normalized_mean() {
        let (w, h) = (2 * SMALL, SMALL); // 2:1
        let mut img = Vec::with_capacity(3 * (w * h) as usize);
        for _ in 0..h {
            for x in 0..w {
                let v = x as f32 / (w - 1) as f32;
                img.extend_from_slice(&[v, v, v]);
            }
        }
        let (fw, fh, border) = placement(Fit::Pad, w, h, SMALL);
        assert_eq!((fw, fh), (SMALL, SMALL / 2));
        let out = preprocess_to(&dev(), &img, w, h, SMALL, Fit::Pad);
        for c in 0..3u32 {
            assert_eq!(at(&out, SMALL, c, 0, 0), 0.0, "border pixel must be the normalized mean");
            assert_eq!(at(&out, SMALL, c, SMALL - 1, SMALL - 1), 0.0, "border pixel must be the normalized mean");
        }
        assert!(border.top > 0, "a 2:1 source under Fit::Pad must letterbox top/bottom");
    }

    #[test]
    fn a_full_range_source_reaches_both_normalized_extremes() {
        let (w, h) = (SMALL, SMALL);
        let mut img = Vec::with_capacity(3 * (w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = (x + y) as f32 / (w + h - 2) as f32;
                img.extend_from_slice(&[v; 3]);
            }
        }
        let out = preprocess_to(&dev(), &img, w, h, SMALL, Fit::Stretch);
        let (lo, hi) = out.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(lo <= -0.98 && hi >= 0.98, "range [{lo}, {hi}] does not reach a [-1,1] normalization");
        assert!(lo >= -1.001 && hi <= 1.001, "range [{lo}, {hi}] escaped [-1,1]");
    }
}
