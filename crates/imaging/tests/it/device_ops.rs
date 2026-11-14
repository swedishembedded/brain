// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-path tests: every operation `imaging` exposes, dispatched for real.
//!
//! These exist because a mismatched `Params` list is *silently wrong, not a
//! crash*. Each test pins the numeric result of
//! one dispatch against a value derived independently in the test — the same
//! discipline a gradcheck oracle uses, and the only thing that catches a field
//! swapped between `stride` and `pad`, or an `Ho`/`Wo` left stale.
//!
//! The device is the shared pooled test device (`gpu_core::testgpu`), never a
//! per-crate fixture.

use gpu_core::testgpu;
use imaging::device::{AlignCorners, Border, Ctx, Fill, Filter};
use imaging::letterbox::Letterbox;
use imaging::mask::FeatherEdge;
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

    // `zipdepth::predict::resize_map`, verbatim in structure.
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
    let (padded, ps) = c.pad(&x, s, b, Fill::Constant(0.0));
    assert_eq!((ps.h, ps.w), (s.h + 5, s.w + 5));
    // The border really is zero, and the content sits at (left, top).
    let p = c.download(&padded, ps.numel());
    assert_eq!(p[0], 0.0);
    assert_eq!(p[(b.top * ps.w + b.left) as usize], src[0]);
    let (back, bs) = c.crop(&padded, ps, Rect::new(b.left, b.top, s.w, s.h));
    assert_eq!(bs.numel(), s.numel());
    assert_eq!(c.download(&back, bs.numel()), src, "crop2d is pad2d's exact adjoint");
}

/// The half-pixel nearest mode is `round((o + 0.5) * in / out - 0.5)` clamped to
/// the source: the tap whose centre is nearest the output pixel's centre.
#[test]
fn nearest_half_pixel_selects_the_nearest_centre_tap() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 1, 1, 3);
    let x = c.upload("src", &[0.0, 10.0, 20.0]);
    let (y, ys) = c.resize(&x, s, 1, 2, Filter::NearestHalfPixel, AlignCorners::HalfPixel);
    // round(0.25) = 0, round(1.75) = 2  ->  [0, 20] (the floor rule gives [0, 10]).
    assert_eq!(c.download(&y, ys.numel()), vec![0.0, 20.0]);

    // The tap rule over a sweep of ratios, against the float formula.
    for (w0, wo) in [(3u32, 2u32), (7, 3), (5, 11), (479, 640), (640, 479), (1, 5), (9, 9)] {
        let src = ramp(w0);
        let x = c.upload("src", &src);
        let (y, ys) = c.resize(&x, Shape::new(1, 1, 1, w0), 1, wo, Filter::NearestHalfPixel, AlignCorners::HalfPixel);
        let got = c.download(&y, ys.numel());
        for o in 0..wo {
            let t = ((o as f64 + 0.5) * w0 as f64 / wo as f64 - 0.5).round().clamp(0.0, w0 as f64 - 1.0);
            assert_eq!(got[o as usize], t as f32, "{w0}->{wo} at {o}");
        }
    }
}

/// `pad2d` writes the caller's fill, or replicates the nearest edge pixel.
#[test]
fn pad_fills_with_a_value_or_replicates_the_edge() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    let s = Shape::new(1, 2, 2, 2);
    let x = c.upload("src", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Border { left: 1, right: 0, top: 0, bottom: 1 };
    let (y, ys) = c.pad(&x, s, b, Fill::Constant(0.25));
    assert_eq!(
        c.download(&y, ys.numel()),
        vec![0.25, 1.0, 2.0, 0.25, 3.0, 4.0, 0.25, 0.25, 0.25, 0.25, 5.0, 6.0, 0.25, 7.0, 8.0, 0.25, 0.25, 0.25]
    );
    let (y, ys) = c.pad(&x, s, b, Fill::Edge);
    assert_eq!(
        c.download(&y, ys.numel()),
        vec![1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 7.0, 7.0, 8.0]
    );
}

/// The device letterbox against an independent per-pixel statement of the rule:
/// the content tile is the half-pixel nearest resize to `new_w x new_h`, pasted
/// at the integral pad, the rest the fill, in every channel. The sizes include
/// odd pads (640x479) and both orientations.
#[test]
fn letterbox_on_device_matches_the_per_pixel_reference() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    for (w0, h0, size) in [(640u32, 479u32, 640u32), (479, 640, 640), (100, 100, 50), (37, 11, 32), (5, 29, 32), (333, 777, 96)] {
        let src: Vec<f32> = (0..3 * w0 * h0).map(|i| ((u64::from(i) * 7919) % 251) as f32).collect();
        let s = Shape::new(1, 3, h0, w0);
        let x = c.upload("src", &src);
        let (y, lb) = c.letterbox(&x, s, size, 0.4375);
        let got = c.download(&y, 3 * size * size);
        let want = Letterbox::compute(w0, h0, size);
        assert_eq!(lb, want);
        let r = lb.content_rect();
        for ch in 0..3u32 {
            for py in 0..size {
                for px in 0..size {
                    let g = got[(ch * size * size + py * size + px) as usize];
                    let inside = px >= r.x && px < r.x + r.w && py >= r.y && py < r.y + r.h;
                    let e = if inside {
                        let tap = |d: u32, i: u32, o: u32| ((d as f64 + 0.5) * i as f64 / o as f64 - 0.5).round().clamp(0.0, i as f64 - 1.0) as u32;
                        let (sx, sy) = (tap(px - r.x, w0, lb.new_w), tap(py - r.y, h0, lb.new_h));
                        src[(ch * h0 * w0 + sy * w0 + sx) as usize]
                    } else {
                        0.4375
                    };
                    assert_eq!(g, e, "{w0}x{h0}->{size} c{ch} ({px},{py})");
                }
            }
        }
    }
}

/// The host `letterbox_rgb` (HWC) is the device `Ctx::letterbox` (CHW) bitwise:
/// one rule, two homes. Odd pads and both orientations.
#[test]
fn host_letterbox_equals_the_device_letterbox_bitwise() {
    let gpu = testgpu::dev(PIPELINES);
    let c = Ctx::new(&gpu);
    for (w0, h0, size) in [(640u32, 479u32, 640u32), (479, 640, 640), (37, 11, 32), (5, 29, 32), (333, 777, 96), (7, 3, 32), (100, 100, 50)] {
        let hwc: Vec<f32> = (0..3 * w0 * h0).map(|i| ((u64::from(i) * 7919) % 251) as f32 / 251.0).collect();
        let chw = pixels::hwc_to_chw(&hwc, 3, h0 as usize, w0 as usize);
        let fill = 114.0 / 255.0;
        let (host, host_lb) = imaging::letterbox_rgb(&hwc, w0, h0, size, fill);
        let x = c.upload("src", &chw);
        let (y, dev_lb) = c.letterbox(&x, Shape::new(1, 3, h0, w0), size, fill);
        assert_eq!(host_lb, dev_lb, "{w0}x{h0}->{size}: transform");
        assert_eq!(host, c.download(&y, 3 * size * size), "{w0}x{h0}->{size}: pixels");
    }
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

/// The host helper for callers whose consumer is a host slice is the device
/// composition (`to_chw` then `affine(2, -1)`), bit for bit: `x * 2` is exact,
/// so a fused multiply-add and a multiply then subtract round identically.
#[test]
fn host_unit_to_signed_chw_equals_the_device_composition() {
    let gpu = testgpu::private(PIPELINES);
    let c = Ctx::new(&gpu);
    let (w, h) = (5usize, 3usize);
    let hwc: Vec<f32> = (0..w * h * 3).map(|i| (i as f32 * 0.137).fract()).collect();
    let s = Shape::new(1, 3, h as u32, w as u32);
    let dev = c.affine(&c.to_chw(&c.upload("hwc", &hwc), s), s, &[2.0; 3], &[-1.0; 3]);
    assert_eq!(pixels::hwc_unit_to_chw_signed(&hwc, 3, h, w), c.download(&dev, s.numel()));
}

/// The inverse host helper equals `affine(0.5, 0.5)` then `to_hwc` bit for bit
/// on `[-1, 1]` input (`x * 0.5` is exact, so fused and unfused agree), and,
/// unlike the device composition, CLAMPS to `[0, 1]`: decoder output is
/// unbounded and its consumers are image writers.
#[test]
fn host_signed_chw_to_unit_hwc_equals_the_device_composition_and_clamps() {
    let gpu = testgpu::private(PIPELINES);
    let c = Ctx::new(&gpu);
    let (w, h) = (5usize, 3usize);
    let chw: Vec<f32> = (0..w * h * 3).map(|i| (i as f32 * 0.291).fract() * 2.0 - 1.0).collect();
    let s = Shape::new(1, 3, h as u32, w as u32);
    let dev = c.to_hwc(&c.affine(&c.upload("chw", &chw), s, &[0.5; 3], &[0.5; 3]), s);
    assert_eq!(pixels::chw_signed_to_hwc_unit(&chw, 3, h, w), c.download(&dev, s.numel()));
    assert_eq!(pixels::chw_signed_to_hwc_unit(&[-3.0, 0.0, 7.0], 1, 1, 3), vec![0.0, 0.5, 1.0]);
}

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
    let b = c.download(&mask::feather(&c, &x, s, 1, FeatherEdge::Zero), s.numel());
    // A single 9.0 spread over a 3x3 box: every neighbour gets 1.0.
    assert!((b[2 * 5 + 2] - 1.0).abs() < 1e-6);
    assert!((b[5 + 1] - 1.0).abs() < 1e-6);
    assert!((b[0] - 0.0).abs() < 1e-6, "outside the window stays zero");
    // Zero-padded border: mass is conserved because the source is interior.
    let total: f32 = b.iter().sum();
    assert!((total - 9.0).abs() < 1e-4, "box blur must conserve interior mass, got {total}");
}

/// A deterministic soft mask in `[0, 1]` (an LCG, so no RNG dependency).
fn soft_mask(n: usize) -> Vec<f32> {
    let mut st = 0x2545_f491_u32;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (st >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

/// The clamped separable box blur an inpaint pipeline applies on the host:
/// `radius` cells each side, indices clamped at the border (replicate).
fn clamped_box_blur(mask: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let win = (2 * radius + 1) as f32;
    let tap = |i: usize, d: usize, n: usize| (i + d).saturating_sub(radius).min(n - 1);
    let mut rows = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            rows[y * w + x] = (0..=2 * radius).map(|d| mask[y * w + tap(x, d, w)]).sum::<f32>() / win;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = (0..=2 * radius).map(|d| rows[tap(y, d, h) * w + x]).sum::<f32>() / win;
        }
    }
    out
}

/// `FeatherEdge::Replicate` is the clamped separable box, on an odd-sized soft
/// mask whose window overhangs every side. The two differ only in summation
/// order: the device sums `K^2` terms each scaled by `1/K^2` (one extra
/// rounding on the weight), the host two chains of `K` terms and a divide, so
/// each is within `gamma_k` of the exact value (every term is in `[0, 1]`, so
/// `S <= 1`) and they are within twice that of each other.
#[test]
fn feather_replicate_is_the_clamped_separable_box() {
    let gpu = testgpu::private(PIPELINES);
    let c = Ctx::new(&gpu);
    let (w, h) = (13usize, 7usize);
    let s = Shape::new(1, 1, h as u32, w as u32);
    let m = soft_mask(w * h);
    let x = c.upload("m", &m);
    for radius in 1..=3usize {
        let k = 2 * radius + 1;
        let got = c.download(&mask::feather(&c, &x, s, radius as u32, FeatherEdge::Replicate), s.numel());
        let want = clamped_box_blur(&m, w, h, radius);
        let bound = brain_testutil::rounding::agreement_bound(k * k + 2 * k + 2, 1.0);
        let worst = got.iter().zip(&want).map(|(a, b)| f64::from((a - b).abs())).fold(0.0, f64::max);
        assert!(worst <= bound, "radius {radius}: max abs diff {worst:e} exceeds the derived bound {bound:e}");
    }
    // The border is where the two conventions differ: a constant mask stays
    // constant under replicate and darkens under zero.
    let ones = c.upload("ones", &vec![1f32; w * h]);
    let rep = c.download(&mask::feather(&c, &ones, s, 2, FeatherEdge::Replicate), s.numel());
    let zero = c.download(&mask::feather(&c, &ones, s, 2, FeatherEdge::Zero), s.numel());
    assert!(rep.iter().all(|v| (v - 1.0).abs() < 1e-6), "replicate keeps a constant mask constant");
    assert!(zero[0] < 0.7, "zero padding darkens the corner, got {}", zero[0]);
}

/// The exact eightfold ratios the VAE latent grid uses: `avgpool2d` is the host
/// integer-window mean `sum / (sx*sy)`, within the round-off of a 64-term sum.
#[test]
fn downsample_matches_the_host_window_mean_on_exact_ratios() {
    let gpu = testgpu::private(PIPELINES);
    let c = Ctx::new(&gpu);
    let (w, h, f) = (48usize, 32usize, 8usize);
    let m = soft_mask(w * h);
    let s = Shape::new(1, 1, h as u32, w as u32);
    let x = c.upload("m", &m);
    let (y, ys) = mask::downsample(&c, &x, s, (h / f) as u32, (w / f) as u32);
    let got = c.download(&y, ys.numel());
    let bound = brain_testutil::rounding::agreement_bound(f * f + 1, 1.0);
    for ly in 0..h / f {
        for lx in 0..w / f {
            let want: f32 = (0..f * f).map(|i| m[(ly * f + i / f) * w + lx * f + i % f]).sum::<f32>() / (f * f) as f32;
            let d = f64::from((got[ly * (w / f) + lx] - want).abs());
            assert!(d <= bound, "cell ({ly},{lx}): diff {d:e} exceeds {bound:e}");
        }
    }
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
