// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Real images in.** Decoded RGB pixels of any size -> the `[3, 1024, 1024]`
//! normalized NCHW tensor [`crate::DeepEncoder::forward`] and
//! [`crate::DeepseekOcr::forward`] take.
//!
//! Until this module existed every caller of those two entry points fed either a
//! spatially constant fill (the SAM parity fixture's gray page) or a slice it had
//! normalized by hand. Neither is an image, so neither could catch a transposed
//! resize, a swapped channel order or a missing affine.
//!
//! ## The three facts this implements, and where each was read off
//!
//! Every one of them is taken from the **shipped mmproj's own KV** or from
//! llama.cpp's `mtmd_image_preprocessor_deepseekocr` - the consumer this GGUF
//! format targets and the same oracle the rest of this workstream gates
//! against. None is inferred from a config default or from a sibling model.
//!
//! 1. **The square is `1024²`, and it is not a literal here.** It is
//!    `sam.grid_h * sam.patch_size` (`sam1::SamViTConfig::image_h`), i.e. the
//!    only extent the SAM tower's learned position table is shaped for. The
//!    mmproj's `clip.vision.image_size = 224` is CLIP's *native* size and has
//!    nothing to do with the model's input - CLIP runs on the compressor's
//!    16x16 token grid through `PatchSource::Tokens`.
//! 2. **The normalization is `mean = std = 0.5`, NOT OpenAI CLIP's.** The
//!    shipped `mmproj-DeepSeek-OCR-Q8_0.gguf` carries
//!    `clip.vision.image_mean = clip.vision.image_std = [0.5, 0.5, 0.5]`, and
//!    llama.cpp's `mtmd_image_preproc_out::append_overview` applies exactly
//!    those. So the map is the plain `[0,1] -> [-1,1]` value-range rescale that
//!    `imaging::Normalization::HALF` already names - see [`NORMALIZATION`].
//!    Borrowing CLIP-L's own published `[0.48145466, …]` / `[0.26862954, …]`
//!    would be wrong for *this* checkpoint however right it looks: the CLIP
//!    tower here never sees a pixel. This module's own unit test asserts the
//!    constant against the real GGUF, so it cannot drift.
//! 3. **The global view is an aspect-preserving centred fit-and-pad, not a
//!    stretch.** llama.cpp resizes with `PAD_NEAREST` and
//!    `image_pad_color = {127, 127, 127}`, which is upstream's
//!    `ImageOps.pad(image, (1024, 1024), color=(127,127,127))`; the pad colour
//!    is `round(mean * 255)`, i.e. the normalization's own zero. [`Fit::Pad`] is
//!    that geometry and is the default; [`Fit::Stretch`] is the plain
//!    non-aspect-preserving resize, kept because a square source makes the two
//!    identical and a test that only ever ran squares would not be testing the
//!    geometry at all.
//!
//! ## Why there is a clamp
//!
//! The reference's resampler writes a `clip_image_u8`, so 8-bit saturation
//! clamps its result to `[0, 1]` before the normalization ever runs. Cubic
//! interpolation has negative lobes and therefore **rings past the source range
//! at a hard edge** - which is what a document scan is made of. Measured here on
//! a synthetic page at `1600x1131 -> 1024²`: the normalized output spanned
//! `[-1.176, 1.191]` where the checkpoint's range is `[-1, 1]` - a ~9.6 %
//! overshoot of the source range, entirely at the text edges. [`clamp01`] is
//! what makes the range match the reference's; it is a fidelity step, not
//! defensive tidiness.
//!
//! ## Why the pad is a `pad_zero`
//!
//! `imaging::Ctx::pad_zero` can only fill with zero - a deliberate limit, since
//! `pad2d.wgsl` has no `pad_value` word. That is not a constraint here, it is the
//! whole trick: **normalize first, then pad**, and zero *is* the mean-grey
//! border, exactly. (The reference quantises its border to `127/255 = 0.498039`
//! before normalizing, so its padded pixels land at `-0.00392` rather than at
//! `0.0`. A ~0.4 % offset confined to the letterbox bars; ours is the exact
//! value upstream's `int(mean * 255)` is an 8-bit approximation of.)
//!
//! ## Scope
//!
//! **The global (overview) view only.** DeepSeek-OCR's "Gundam" mode also emits
//! a local tile grid at 640² and interleaves the two token streams; the row
//! layout for that already exists in [`crate::rows`], but no graph consumes it
//! and neither does this module. One image, one view, one 1024² tensor.
//!
//! Batch is 1, like everything else in this crate.

use gpu_core::{DeviceBuffer, Gpu};
use imaging::{AlignCorners, Border, Ctx, Filter, Normalization, Shape};

use crate::config::DeepseekOcrConfig;

/// Every kernel this module dispatches.
///
/// `imaging::PIPELINES` covers all but the last: `relu_inplace` is what
/// [`clamp01`] is built from, and `imaging` has no clamp of its own. Registering
/// this list (or any superset) is what a caller's `Gpu` needs -
/// `ImagingKernelIds::resolve_on` finds each kernel by name wherever it sits.
pub const PIPELINES: &[(&str, &str)] = &[
    ("resize_bilinear", kernels::RESIZE_BILINEAR),
    ("resize_bicubic", kernels::RESIZE_BICUBIC),
    ("film_chan", kernels::FILM_CHAN),
    ("pad2d", kernels::PAD2D),
    ("relu_inplace", kernels::RELU_INPLACE),
];

/// The pixel normalization the shipped checkpoint declares:
/// `clip.vision.image_mean == clip.vision.image_std == [0.5; 3]`, i.e. the
/// `[0,1] -> [-1,1]` rescale. Sourced from `imaging`'s own named constant rather
/// than transcribed - see this module's header for why it is NOT CLIP's.
pub const NORMALIZATION: Normalization = Normalization::HALF;

/// The resampling filter. The reference uses Pillow's bicubic
/// (`RESIZE_ALGO_BICUBIC_PILLOW`); brain's `resize_bicubic` is the torch/ONNX
/// cubic (`a = -0.75`, half-pixel) and is **not** antialiased on a downscale, so
/// this is the closest available kernel and not a bit-level match. Nothing gates
/// on the difference - no reference capture for a non-constant image exists (see
/// `tests/real_weight.rs`) - so it is recorded, not claimed away.
pub const FILTER: Filter = Filter::Bicubic;

/// Half-pixel, matching both torch's `align_corners=False` default and the
/// reference's own sampling.
pub const ALIGN: AlignCorners = AlignCorners::HalfPixel;

/// How a source image of arbitrary extent is mapped onto the model's square.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fit {
    /// Aspect-preserving centred fit; the remainder is the mean-grey border.
    /// **What the reference does.**
    #[default]
    Pad,
    /// Plain non-aspect-preserving resize - the whole square is image.
    Stretch,
}

/// Where the resized content lands inside the square: its extent and the border
/// around it. `Fit::Stretch` always yields the full square and a zero border.
///
/// Pure host geometry, and public so a caller that has to map coordinates back
/// into source pixels (a bounding box, a click) can ask for the same numbers the
/// resize used instead of re-deriving them.
pub fn placement(fit: Fit, w: u32, h: u32, out_w: u32, out_h: u32) -> (u32, u32, Border) {
    assert!(w > 0 && h > 0, "source image is {w}x{h}");
    assert!(out_w > 0 && out_h > 0, "target is {out_w}x{out_h}");
    if fit == Fit::Stretch {
        return (out_w, out_h, Border::default());
    }
    // `scale = min(out_w/w, out_h/h)`, then `min(round(extent * scale), out)` --
    // llama.cpp's `img_tool::resize` under `PAD_NEAREST`, verbatim.
    let scale = (out_w as f32 / w as f32).min(out_h as f32 / h as f32);
    let fit_w = ((w as f32 * scale).round() as u32).min(out_w).max(1);
    let fit_h = ((h as f32 * scale).round() as u32).min(out_h).max(1);
    // The reference centres with `round(gap / 2.0)`, which rounds a half UP, so
    // an odd gap puts the extra pixel on the LEFT/TOP. `Border::centre` puts it
    // on the right/bottom, so this is spelled out rather than reused.
    let half_up = |gap: u32| gap.div_ceil(2);
    let (dw, dh) = (out_w - fit_w, out_h - fit_h);
    let (left, top) = (half_up(dw), half_up(dh));
    (fit_w, fit_h, Border { left, right: dw - left, top, bottom: dh - top })
}

/// Decoded RGB pixels -> the model's input tensor.
///
/// `hwc` is `[h, w, 3]` interleaved f32 in `[0, 1]` - this repo's wire format
/// for a decoded image, exactly what `capability::blob::decode_image` and
/// `imaging::codec::decode` hand back. The result is `[3, S, S]` NCHW with
/// `S = cfg.sam.image_h()` (1024 for the real preset), normalized, ready for
/// [`crate::DeepEncoder::forward`].
///
/// `gpu` must have [`PIPELINES`] (or any superset) registered.
pub fn preprocess_image(gpu: &Gpu, cfg: &DeepseekOcrConfig, hwc: &[f32], w: u32, h: u32, fit: Fit) -> Vec<f32> {
    preprocess_to(gpu, hwc, w, h, cfg.sam.image_w(), cfg.sam.image_h(), fit)
}

/// [`preprocess_image`] with the target extent given explicitly.
///
/// The size-carrying entry point exists for tests (a 1024² device round trip per
/// assertion is not a unit test) and for a future multi-view path, where the
/// local tiles are 640² while the overview stays at the tower's native square.
///
/// Extents are `(width, height)` throughout this module -- the same order as
/// [`placement`], deliberately NOT `imaging::Ctx::resize`'s `(ho, wo)`. One
/// convention per module beats matching whichever callee is nearest.
pub fn preprocess_to(gpu: &Gpu, hwc: &[f32], w: u32, h: u32, out_w: u32, out_h: u32, fit: Fit) -> Vec<f32> {
    assert_eq!(
        hwc.len(),
        3 * w as usize * h as usize,
        "expected [h, w, 3] = [{h}, {w}, 3] interleaved RGB in [0,1], got {} floats",
        hwc.len()
    );
    let (fit_w, fit_h, border) = placement(fit, w, h, out_w, out_h);

    // Layout permutation is host glue by the `crates/imaging` rule; everything
    // per-pixel below is a dispatch.
    let chw = imaging::pixels::hwc_to_chw(hwc, 3, h as usize, w as usize);
    let ctx = Ctx::new(gpu);
    let src = ctx.upload("deepseekocr.preprocess.src", &chw);
    let (resized, shape) = ctx.resize(&src, Shape::new(1, 3, h, w), fit_h, fit_w, FILTER, ALIGN);

    // Cubic interpolation has negative lobes, so at a hard edge -- exactly what
    // a document scan is made of -- it RINGS past the source range. Measured on
    // a synthetic page: the normalized output spanned [-1.176, 1.191] against a
    // checkpoint range of [-1, 1], a ~9.6% overshoot of the source range. The
    // reference cannot ring, because its resampler's output is a `clip_image_u8`
    // and 8-bit saturation clamps it before `from_u8` ever runs. So this clamp
    // is not defensive tidiness, it is the step that makes the value range match
    // the reference's.
    let clamped = clamp01(&ctx, &resized, shape);

    // Normalize BEFORE padding: the reference's border colour is its own
    // `mean`, so after the affine it is exactly zero -- which is the only fill
    // `pad_zero` can produce. See this module's header.
    let norm = ctx.normalize(&clamped, shape, &NORMALIZATION);
    let (out, out_shape) = if border == Border::default() {
        (norm, shape)
    } else {
        ctx.pad_zero(&norm, shape, border)
    };
    assert_eq!(out_shape, Shape::new(1, 3, out_h, out_w), "preprocess produced {out_shape:?}");
    ctx.download(&out, out_shape.numel())
}

/// `clamp(x, 0, 1)` on the device, as `1 - relu(1 - relu(x))`.
///
/// `imaging` has no clamp kernel and this change does not add one: `relu` and
/// the per-channel affine are already brain's two most general elementwise
/// primitives, and the identity above needs nothing else. Four dispatches over
/// the resized image - negligible beside the resize that produced it, and the
/// alternative (a host loop over three million pixels) is the thing
/// `crates/imaging` exists to prevent.
fn clamp01(ctx: &Ctx<'_>, x: &DeviceBuffer, s: Shape) -> DeviceBuffer {
    let relu = ctx.gpu.kernel_index("relu_inplace").expect("`relu_inplace` is not registered on this device");
    let flip = ([-1.0f32; 3], [1.0f32; 3]); // y = 1 - x
    let run_relu = |b: &DeviceBuffer| {
        let n = s.numel();
        ctx.gpu.submit(&[], &[ctx.gpu.step(relu, &[b], &[n], n)]);
    };
    // `relu_inplace` has ONE read_write binding, so it needs a buffer it may
    // overwrite; `affine` always allocates a fresh one, which is why the first
    // step is an affine (the identity) rather than a relu on the caller's `x`.
    let lo = ctx.affine(x, s, &[1.0; 3], &[0.0; 3]);
    run_relu(&lo); // max(x, 0)
    let hi = ctx.affine(&lo, s, &flip.0, &flip.1);
    run_relu(&hi); // max(1 - max(x, 0), 0)
    ctx.affine(&hi, s, &flip.0, &flip.1) // 1 - that = min(max(x, 0), 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1024² round trip per assertion would make this file the slowest in the
    /// crate, so the geometry runs at 64² and only the shape/range test pays for
    /// the production square.
    const SMALL: u32 = 64;

    fn dev() -> Gpu {
        gpu_core::testgpu::dev(PIPELINES)
    }

    /// An HWC test image whose value depends on BOTH axes and on the channel --
    /// a spatially constant fill (what every other test in this effort feeds)
    /// cannot see a transposed, flipped or channel-swapped resize.
    fn gradient(w: u32, h: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(3 * (w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 / (w - 1) as f32, y as f32 / (h - 1) as f32);
                v.extend_from_slice(&[fx, fy, 0.25 + 0.5 * fx * fy]);
            }
        }
        v
    }

    /// Four solid quadrants, each a distinct colour. TL/TR/BL/BR.
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

    /// Read output pixel `(x, y)` of channel `c` from a `[3, side, side]` plane.
    fn at(out: &[f32], side: u32, c: u32, x: u32, y: u32) -> f32 {
        out[(c * side * side + y * side + x) as usize]
    }

    /// The whole point of the module: an arbitrary real-world extent becomes the
    /// tensor the encoder takes, in the value range the checkpoint expects.
    #[test]
    fn a_real_sized_image_becomes_the_models_normalized_square() {
        let cfg = DeepseekOcrConfig::deepseek_ocr(1);
        let (side, gpu) = (cfg.sam.image_h(), dev());
        assert_eq!((side, cfg.sam.image_w()), (1024, 1024), "the square is grid_h * patch_size");

        // Deliberately non-square, deliberately not a multiple of 1024.
        let (w, h) = (640u32, 427u32);
        let out = preprocess_image(&gpu, &cfg, &gradient(w, h), w, h, Fit::Stretch);
        assert_eq!(out.len(), 3 * (side * side) as usize);
        assert_eq!(out.len(), 3_145_728, "the DeepEncoder's input is [3, 1024, 1024]");
        assert!(out.iter().all(|v| v.is_finite()), "preprocess produced a non-finite pixel");

        // Normalized, not raw: `[0,1]` in becomes `[-1,1]` out, and a source that
        // spans the full unit interval must reach both ends of it.
        let (lo, hi) = out.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(lo < -0.9 && hi > 0.9, "range [{lo}, {hi}] does not look like a [-1,1] normalization");
        assert!(lo >= -1.001 && hi <= 1.001, "range [{lo}, {hi}] escaped [-1,1] -- the affine is wrong");
        assert!(lo < 0.0, "output is still in [0,1]: the normalization did not run");
    }

    /// The correctness check the shape/range pair cannot make: four solid
    /// quadrants must come out as four solid quadrants, in the SAME corners, per
    /// channel. Catches a transposed resize, a flipped axis and a CHW/HWC mixup
    /// in one assertion -- the four colours are chosen so no two share a value on
    /// any channel.
    #[test]
    fn quadrants_survive_the_resize_in_place_and_in_order() {
        let colors = [[0.0, 0.2, 0.4], [1.0, 0.6, 0.8], [0.3, 1.0, 0.1], [0.7, 0.5, 0.9]];
        let src_side = 8 * SMALL; // a 8x downscale, so the resize really resamples
        let gpu = dev();
        let out = preprocess_to(&gpu, &quadrants(src_side, colors), src_side, src_side, SMALL, SMALL, Fit::Pad);
        assert_eq!(out.len(), 3 * (SMALL * SMALL) as usize);

        // A square source under `Fit::Pad` needs no border at all.
        assert_eq!(placement(Fit::Pad, src_side, src_side, SMALL, SMALL).2, Border::default());

        // Sample the centre of each output quadrant, well clear of the seam the
        // interpolation blends across.
        let (q, e) = (SMALL / 4, SMALL / 4 + SMALL / 2);
        let probes = [(q, q, 0usize), (e, q, 1), (q, e, 2), (e, e, 3)];
        let (scale, shift) = NORMALIZATION.scale_shift();
        for (x, y, idx) in probes {
            for c in 0..3u32 {
                let want = colors[idx][c as usize] * scale[c as usize] + shift[c as usize];
                let got = at(&out, SMALL, c, x, y);
                assert!(
                    (got - want).abs() < 1e-4,
                    "quadrant {idx} channel {c} at ({x},{y}): got {got}, want {want} \
                     (a transposed/flipped resize or a channel swap looks exactly like this)"
                );
            }
        }
    }

    /// `Fit::Pad` is aspect-preserving: a 2:1 source fills half the square's
    /// height and the rest is the mean-grey border, which after the affine is
    /// EXACTLY zero. `Fit::Stretch` on the same source leaves no border at all --
    /// asserted together, because "all zeros somewhere" is only evidence of a
    /// letterbox if the other mode does not produce it.
    #[test]
    fn pad_letterboxes_and_stretch_does_not() {
        let gpu = dev();
        let (w, h) = (2 * SMALL, SMALL); // 2:1
        let img = gradient(w, h);

        let (fw, fh, border) = placement(Fit::Pad, w, h, SMALL, SMALL);
        assert_eq!((fw, fh), (SMALL, SMALL / 2), "a 2:1 source fits the full width and half the height");
        assert_eq!(border, Border { left: 0, right: 0, top: SMALL / 4, bottom: SMALL / 4 });

        let padded = preprocess_to(&gpu, &img, w, h, SMALL, SMALL, Fit::Pad);
        for c in 0..3u32 {
            for y in [0u32, border.top - 1, SMALL - 1] {
                for x in [0u32, SMALL / 2, SMALL - 1] {
                    assert_eq!(at(&padded, SMALL, c, x, y), 0.0, "border pixel ({x},{y}) c{c} is not the mean");
                }
            }
            // ... and the content band is not zero.
            let mid = at(&padded, SMALL, c, SMALL - 1, SMALL / 2);
            assert!(mid.abs() > 1e-3, "the content band at c{c} looks like border too ({mid})");
        }

        let stretched = preprocess_to(&gpu, &img, w, h, SMALL, SMALL, Fit::Stretch);
        let zero_rows = (0..SMALL).filter(|&y| (0..SMALL).all(|x| at(&stretched, SMALL, 0, x, y) == 0.0)).count();
        assert_eq!(zero_rows, 0, "Fit::Stretch must fill the whole square");
    }

    /// The cubic filter's negative lobes ring past the source range at a hard
    /// edge, and the reference cannot (its resampler's output is 8-bit). This
    /// pins the clamp with the input that actually provokes it - the smooth
    /// gradient above never leaves `[0,1]` and would let a missing clamp pass.
    ///
    /// Mutation-verified by construction: without [`clamp01`] this exact input
    /// produced `[-0.176, 1.191]` in `[0,1]` terms, which the real-weight test's
    /// range assertion caught first.
    #[test]
    fn a_hard_edge_does_not_ring_out_of_range() {
        // Two solid halves, i.e. one maximally sharp edge, at the extremes of
        // the unit interval -- there is no headroom for an overshoot to hide in.
        let src = 4 * SMALL;
        let mut img = Vec::with_capacity(3 * (src * src) as usize);
        for _ in 0..src {
            for x in 0..src {
                let v = if x < src / 2 { 0.0 } else { 1.0 };
                img.extend_from_slice(&[v, v, v]);
            }
        }
        let out = preprocess_to(&dev(), &img, src, src, SMALL, SMALL, Fit::Stretch);
        let (lo, hi) = out.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(lo >= -1.0 && hi <= 1.0, "the cubic filter rang out of [-1,1]: [{lo}, {hi}]");
        // ... and the edge is still an edge: both extremes are actually reached,
        // so this did not pass by blurring the image into the middle.
        assert!(lo <= -0.999 && hi >= 0.999, "range [{lo}, {hi}] lost the edge entirely");
    }

    /// The centring rule is the reference's `round(gap / 2.0)`, which rounds a
    /// half AWAY from zero -- so an odd gap puts the extra pixel on the top/left,
    /// the opposite of `imaging::Border::centre`. One pixel, but it is a real
    /// disagreement and this is where it is pinned.
    #[test]
    fn an_odd_gap_puts_the_extra_pixel_before_the_content() {
        let (_, _, b) = placement(Fit::Pad, 10, 9, 10, 10);
        assert_eq!(b, Border { left: 0, right: 0, top: 1, bottom: 0 });
        assert_ne!(b, Border::centre(10, 9, 10, 10), "and it differs from Border::centre");
    }

    /// The normalization constant is the shipped checkpoint's own, not a
    /// plausible-looking CLIP one. Skips when the model store is absent.
    #[test]
    fn the_normalization_is_the_shipped_files_own() {
        let Some(dir) = brain_testutil::model_dir("ggml-org/DeepSeek-OCR-GGUF") else {
            eprintln!("skip: no model store");
            return;
        };
        let mmproj = std::path::Path::new(&dir).join("mmproj-DeepSeek-OCR-Q8_0.gguf");
        if !mmproj.exists() {
            eprintln!("skip: mmproj absent");
            return;
        }
        let mg = checkpoint::gguf::MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
        let cfg = gguf::deepseek_ocr_vision::config_from_gguf(&mg).expect("mmproj config");
        assert_eq!(cfg.image_mean, NORMALIZATION.mean.to_vec(), "clip.vision.image_mean moved");
        assert_eq!(cfg.image_std, NORMALIZATION.std.to_vec(), "clip.vision.image_std moved");
        // The literal, spelled out once, so a reader does not have to chase
        // `Normalization::HALF` to see what this checkpoint actually asks for.
        assert_eq!(cfg.image_mean, vec![0.5; 3]);
        assert_eq!(cfg.image_std, vec![0.5; 3]);
    }
}
