// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-path tests: every operation `imaging` exposes, dispatched for real.
//!
//! These exist because a mismatched `Params` list is *silently wrong, not a
//! crash* (`.agents/rules/kernels.md` §B). Each test pins the numeric result of
//! one dispatch against a value derived independently in the test — the same
//! discipline a gradcheck oracle uses, and the only thing that catches a field
//! swapped between `stride` and `pad`, or an `Ho`/`Wo` left stale.
//!
//! The device is the shared pooled test device (`gpu_core::testgpu`), never a
//! per-crate fixture.

use gpu_core::testgpu;
use imaging::device::{AlignCorners, Border, Ctx, Filter};
use imaging::{mask, pixels, Rect, Shape, TilePlan, TileSpec, PIPELINES};

fn ctx() -> (gpu_core::Gpu, Shape) {
    (testgpu::dev(PIPELINES), Shape::new(1, 1, 4, 4))
}

/// `0..n` as an f32 ramp.
fn ramp(n: u32) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

// ---- resampling ------------------------------------------------------------

/// The claim in `Filter::Bilinear`'s doc — that the kernel under
/// `AlignCorners::HalfPixel` is the same function as the three host bilinear
/// copies in `depth`/`cli` — re-derived here from the formula those copies use.
#[test]
fn bilinear_half_pixel_matches_the_host_formula_the_survey_calls_equivalent() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let (h0, w0, th, tw) = (3u32, 4u32, 5u32, 7u32);
    let src = ramp(h0 * w0);
    let s = Shape::new(1, 1, h0, w0);
    let x = c.upload("src", &src);
    let (y, ys) = c.resize(&x, s, th, tw, Filter::Bilinear, AlignCorners::HalfPixel);
    let got = c.download(&y, ys.numel());

    // `depth::predict::resize_map`, verbatim in structure.
    let (sx, sy) = (w0 as f32 / tw as f32, h0 as f32 / th as f32);
    for oy in 0..th {
        for ox in 0..tw {
            let fy = ((oy as f32 + 0.5) * sy - 0.5).clamp(0.0, h0 as f32 - 1.0);
            let (y0, ty) = (fy.floor() as u32, fy - fy.floor());
            let y1 = (y0 + 1).min(h0 - 1);
            let fx = ((ox as f32 + 0.5) * sx - 0.5).clamp(0.0, w0 as f32 - 1.0);
            let (x0, tx) = (fx.floor() as u32, fx - fx.floor());
            let x1 = (x0 + 1).min(w0 - 1);
            let p = |xx: u32, yy: u32| src[(yy * w0 + xx) as usize];
            let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
            let bot = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
            let want = top * (1.0 - ty) + bot * ty;
            let g = got[(oy * tw + ox) as usize];
            assert!((g - want).abs() < 1e-5, "({oy},{ox}): kernel {g} vs host {want}");
        }
    }
}

/// `align_corners` is the parameter that "both look plausible and differ by half
/// a pixel". Pin both, so a swapped word cannot pass.
#[test]
fn align_corners_selects_a_different_grid() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 1, 1, 3);
    let x = c.upload("src", &[0.0, 10.0, 20.0]);
    let (yh, sh) = c.resize(&x, s, 1, 5, Filter::Bilinear, AlignCorners::HalfPixel);
    let (yc, sc) = c.resize(&x, s, 1, 5, Filter::Bilinear, AlignCorners::Corners);
    let half = c.download(&yh, sh.numel());
    let corners = c.download(&yc, sc.numel());
    // Corners: src = o*(3-1)/(5-1) = o*0.5  ->  samples at 0, .5, 1, 1.5, 2.
    let want_corners = [0.0, 5.0, 10.0, 15.0, 20.0];
    // Half-pixel: src = (o+0.5)*3/5 - 0.5, clamped >= 0 and to the last row on
    // the high side  ->  -0.2 (clamped 0), 0.4, 1.0, 1.6, 2.2 (clamped 2).
    let want_half = [0.0, 4.0, 10.0, 16.0, 20.0];
    for i in 0..5 {
        assert!((corners[i] - want_corners[i]).abs() < 1e-4, "corners[{i}] = {}", corners[i]);
        assert!((half[i] - want_half[i]).abs() < 1e-4, "half[{i}] = {}", half[i]);
    }
    // They agree only where the two grids happen to coincide.
    assert_ne!(half, corners, "the two conventions must not produce the same image");
}

/// `Filter::Nearest` uses the kernel's asymmetric-floor rule, which is NOT the
/// half-pixel rule `letterbox_rgb` uses. Documented; pinned here so the two can
/// never be quietly conflated.
#[test]
fn nearest_is_asymmetric_floor_not_half_pixel() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 1, 1, 3);
    let x = c.upload("src", &[0.0, 10.0, 20.0]);
    let (y, ys) = c.resize(&x, s, 1, 2, Filter::Nearest, AlignCorners::HalfPixel);
    // floor(0*3/2) = 0, floor(1*3/2) = 1  ->  [0, 10]
    // half-pixel would give round(0.25)=0, round(1.75)=2 -> [0, 20]
    assert_eq!(c.download(&y, ys.numel()), vec![0.0, 10.0]);
}

/// A constant image survives bicubic exactly (the four taps sum to 1 for every
/// fraction) — the cheapest check that the polynomial is wired right.
#[test]
fn bicubic_reproduces_a_constant_and_differs_from_bilinear() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 1, 4, 4);
    let flat = c.upload("flat", &[0.375f32; 16]);
    let (y, ys) = c.resize(&flat, s, 7, 9, Filter::Bicubic, AlignCorners::HalfPixel);
    for v in c.download(&y, ys.numel()) {
        assert!((v - 0.375).abs() < 1e-5, "bicubic on a constant gave {v}");
    }
    // On a ramp the two filters must not agree, or one of them is not running.
    let x = c.upload("ramp", &ramp(16));
    let (a, sa) = c.resize(&x, s, 7, 9, Filter::Bicubic, AlignCorners::HalfPixel);
    let (b, _) = c.resize(&x, s, 7, 9, Filter::Bilinear, AlignCorners::HalfPixel);
    assert_ne!(c.download(&a, sa.numel()), c.download(&b, sa.numel()));
}

// ---- geometry --------------------------------------------------------------

#[test]
fn pad_then_crop_is_the_identity_bitwise() {
    let (gpu, s) = ctx();
    let c = Ctx::new(&gpu);
    let src = ramp(s.numel());
    let x = c.upload("src", &src);
    let b = Border { left: 2, right: 3, top: 1, bottom: 4 };
    let (padded, ps) = c.pad_zero(&x, s, b);
    assert_eq!((ps.h, ps.w), (s.h + 5, s.w + 5));
    // The border really is zero, and the content sits at (left, top).
    let p = c.download(&padded, ps.numel());
    assert_eq!(p[0], 0.0);
    assert_eq!(p[(b.top * ps.w + b.left) as usize], src[0]);
    let (back, bs) = c.crop(&padded, ps, Rect::new(b.left, b.top, s.w, s.h));
    assert_eq!(bs.numel(), s.numel());
    assert_eq!(c.download(&back, bs.numel()), src, "crop2d is pad2d's exact adjoint");
}

#[test]
fn crop_takes_the_requested_rectangle() {
    let (gpu, s) = ctx();
    let c = Ctx::new(&gpu);
    let x = c.upload("src", &ramp(s.numel()));
    let (y, ys) = c.crop(&x, s, Rect::new(1, 2, 2, 2));
    // 4x4 ramp: rows 2..4, cols 1..3  ->  [9,10, 13,14]
    assert_eq!(c.download(&y, ys.numel()), vec![9.0, 10.0, 13.0, 14.0]);
}

#[test]
fn add_region_places_a_patch_into_a_zeroed_canvas() {
    let (gpu, s) = ctx();
    let c = Ctx::new(&gpu);
    let canvas = c.upload("canvas", &vec![0f32; s.numel() as usize]);
    let patch = c.upload("patch", &[1.0, 2.0, 3.0, 4.0]);
    c.add_region(&canvas, s, &patch, Rect::new(2, 1, 2, 2));
    let got = c.download(&canvas, s.numel());
    let mut want = vec![0f32; 16];
    want[4 + 2] = 1.0;
    want[4 + 3] = 2.0;
    want[2 * 4 + 2] = 3.0;
    want[2 * 4 + 3] = 4.0;
    assert_eq!(got, want);
}

// ---- layout ----------------------------------------------------------------

#[test]
fn device_layout_matches_the_host_permutation_and_round_trips() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 3, 2, 4);
    let chw = ramp(s.numel());
    let x = c.upload("chw", &chw);
    let hwc = c.to_hwc(&x, s);
    assert_eq!(
        c.download(&hwc, s.numel()),
        pixels::chw_to_hwc(&chw, s.c as usize, s.h as usize, s.w as usize),
        "nchw_nlc must agree with pixels::chw_to_hwc"
    );
    let back = c.to_chw(&hwc, s);
    assert_eq!(c.download(&back, s.numel()), chw, "the permutation is its own inverse");
}

// ---- affine / normalisation ------------------------------------------------

#[test]
fn normalize_then_denormalize_recovers_the_input() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 3, 2, 2);
    let src: Vec<f32> = (0..12).map(|i| i as f32 / 12.0).collect();
    let x = c.upload("x", &src);
    let n = imaging::Normalization::IMAGENET;
    let y = c.normalize(&x, s, &n);
    // Channel 0 of the normalised buffer must be (x - mean0)/std0.
    let ny = c.download(&y, s.numel());
    for i in 0..4 {
        let want = (src[i] - n.mean[0]) / n.std[0];
        assert!((ny[i] - want).abs() < 1e-5, "channel 0 elem {i}: {} vs {want}", ny[i]);
    }
    let back = c.download(&c.denormalize(&y, s, &n), s.numel());
    for i in 0..12 {
        assert!((back[i] - src[i]).abs() < 1e-5, "elem {i}: {} vs {}", back[i], src[i]);
    }
}

#[test]
fn affine_applies_a_per_channel_scale_and_shift() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 2, 1, 2);
    let x = c.upload("x", &[1.0, 2.0, 3.0, 4.0]);
    let y = c.affine(&x, s, &[2.0, -1.0], &[0.5, 10.0]);
    assert_eq!(c.download(&y, s.numel()), vec![2.5, 4.5, 7.0, 6.0]);
}

// ---- mask algebra ----------------------------------------------------------

fn mask_ctx() -> (gpu_core::Gpu, Shape) {
    (testgpu::dev(PIPELINES), Shape::new(1, 1, 5, 5))
}

#[test]
fn threshold_is_a_hard_step_not_a_ramp() {
    let (gpu, _) = mask_ctx();
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 1, 1, 6);
    let x = c.upload("x", &[0.0, 0.49, 0.5, 0.51, 1.0, -3.0]);
    let m = mask::threshold(&c, &x, s, 0.5);
    // Strictly greater: 0.5 itself is below.
    assert_eq!(c.download(&m, s.numel()), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn dilate_and_erode_are_dual_on_a_single_pixel() {
    let (gpu, s) = mask_ctx();
    let c = Ctx::new(&gpu);
    let mut m = vec![0f32; 25];
    m[2 * 5 + 2] = 1.0; // centre pixel
    let x = c.upload("m", &m);

    let d = c.download(&mask::dilate(&c, &x, s, 1), s.numel());
    let on: Vec<usize> = d.iter().enumerate().filter(|(_, &v)| v > 0.5).map(|(i, _)| i).collect();
    assert_eq!(on.len(), 9, "radius-1 dilation of a point is a 3x3 block");
    assert!(on.contains(&(5 + 1)) && on.contains(&(3 * 5 + 3)));

    // Eroding that block by the same radius returns the single pixel.
    let block = c.upload("block", &d);
    let e = c.download(&mask::erode(&c, &block, s, 1), s.numel());
    let on: Vec<usize> = e.iter().enumerate().filter(|(_, &v)| v > 0.5).map(|(i, _)| i).collect();
    assert_eq!(on, vec![2 * 5 + 2]);
}

#[test]
fn erode_of_a_full_mask_keeps_the_border_by_this_conventions_definition() {
    // Documented in `mask`'s header: out-of-image taps are never selected, so a
    // full mask erodes to itself rather than losing its border ring. A caller
    // that wants SciPy's `border_value=0` must pad first.
    let (gpu, s) = mask_ctx();
    let c = Ctx::new(&gpu);
    let x = c.upload("full", &[1f32; 25]);
    assert_eq!(c.download(&mask::erode(&c, &x, s, 1), s.numel()), vec![1f32; 25]);
}

#[test]
fn feather_box_blur_averages_over_the_window() {
    let (gpu, s) = mask_ctx();
    let c = Ctx::new(&gpu);
    let mut m = vec![0f32; 25];
    m[2 * 5 + 2] = 9.0;
    let x = c.upload("m", &m);
    let b = c.download(&mask::feather(&c, &x, s, 1), s.numel());
    // A single 9.0 spread over a 3x3 box: every neighbour gets 1.0.
    assert!((b[2 * 5 + 2] - 1.0).abs() < 1e-6);
    assert!((b[5 + 1] - 1.0).abs() < 1e-6);
    assert!((b[0] - 0.0).abs() < 1e-6, "outside the window stays zero");
    // Zero-padded border: mass is conserved because the source is interior.
    let total: f32 = b.iter().sum();
    assert!((total - 9.0).abs() < 1e-4, "box blur must conserve interior mass, got {total}");
}

#[test]
fn set_operations_are_exact_on_hard_masks() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 1, 1, 4);
    let a = c.upload("a", &[0.0, 1.0, 1.0, 0.0]);
    let b = c.upload("b", &[0.0, 0.0, 1.0, 1.0]);
    assert_eq!(c.download(&mask::intersect(&c, &a, &b, s), 4), vec![0.0, 0.0, 1.0, 0.0]);
    assert_eq!(c.download(&mask::union(&c, &a, &b, s), 4), vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(c.download(&mask::difference(&c, &a, &b, s), 4), vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(c.download(&mask::invert(&c, &a, s), 4), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn composite_is_exact_at_both_ends_of_the_mask() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 1, 1, 3);
    let new = c.upload("new", &[10.0, 10.0, 10.0]);
    let old = c.upload("old", &[1.0, 1.0, 1.0]);
    let m = c.upload("m", &[0.0, 1.0, 0.5]);
    assert_eq!(c.download(&mask::composite(&c, &new, &old, &m, s), 3), vec![1.0, 10.0, 5.5]);
}

#[test]
fn downsample_area_averages_and_generalises_the_integer_ratio_case() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    // Exact ratio: 4x4 -> 2x2 is a plain 2x2 box pool.
    let s = Shape::new(1, 1, 4, 4);
    let x = c.upload("x", &ramp(16));
    let (y, ys) = mask::downsample(&c, &x, s, 2, 2);
    assert_eq!(c.download(&y, ys.numel()), vec![2.5, 4.5, 10.5, 12.5]);
    // Non-dividing ratio: 5 -> 2 is where `zimage`'s integer `w/lw` drops a
    // column. The adaptive rule covers every source pixel.
    let s5 = Shape::new(1, 1, 1, 5);
    let x5 = c.upload("x5", &[1.0, 1.0, 1.0, 0.0, 0.0]);
    let (y5, ys5) = mask::downsample(&c, &x5, s5, 1, 2);
    let got = c.download(&y5, ys5.numel());
    assert!((got[0] - 1.0).abs() < 1e-6, "cols 0..3 -> 1.0, got {}", got[0]);
    assert!((got[1] - 1.0 / 3.0).abs() < 1e-6, "cols 2..5 -> 1/3, got {}", got[1]);
}

#[test]
fn broadcast_channels_replicates_a_mask_for_compositing() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 1, 1, 2);
    let m = c.upload("m", &[0.25, 0.75]);
    let (b, bs) = mask::broadcast_channels(&c, &m, s, 3);
    assert_eq!(bs, Shape::new(1, 3, 1, 2));
    assert_eq!(c.download(&b, bs.numel()), vec![0.25, 0.75, 0.25, 0.75, 0.25, 0.75]);
}

// ---- tiling end to end -----------------------------------------------------

/// The property the tiling design rests on: crop every tile, keep every core,
/// add the cores into a zeroed canvas, and the original image comes back
/// exactly — no blend, no seam, no double-counted pixel.
#[test]
fn tiled_crop_and_recompose_reproduces_the_image_exactly() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let (w, h) = (11u32, 7u32);
    let s = Shape::new(1, 1, h, w);
    let src = ramp(w * h);
    let x = c.upload("img", &src);

    let plan = TilePlan::new(w, h, TileSpec::new(4, 2));
    assert!(plan.len() > 1 && plan.overhead() > 1.0);
    let canvas = c.upload("canvas", &vec![0f32; (w * h) as usize]);
    for t in &plan.tiles {
        let (tile, ts) = c.crop(&x, s, t.src);
        let (core, _) = c.crop(&tile, ts, t.keep);
        c.add_region(&canvas, s, &core, t.dst);
    }
    assert_eq!(c.download(&canvas, s.numel()), src);
}
