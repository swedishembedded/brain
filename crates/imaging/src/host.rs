// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The host resampler — the ONE copy, and why it is allowed to exist.
//!
//! `AGENTS.md` says per-pixel arithmetic over a whole image belongs in a kernel,
//! and [`crate::Ctx::resize`] is that kernel. This module is not a convenience
//! twin of it. It exists because three call sites resize an image with **no
//! `Gpu` in scope at all**:
//!
//! * `cli::depth_cli::run_npu_session` and `cli::resident_depth` — the Intel-NPU
//!   paths. OpenVINO is a whole-graph compiler, not a `gpu-core` backend
//!   (`AGENTS.md`), so `--device npu` genuinely has no device handle to dispatch
//!   a `resize_bilinear` on, and building a `Gpu` just to resize would violate
//!   the one-device-per-process rule.
//! * `zipdepth::predict` resizes the incoming frame *before* the model's input
//!   buffer exists, on whichever device the predictor was built for.
//!
//! Those three sites each carried a byte-identical copy of the same two
//! functions (six functions, one implementation). They now all call the one
//! below. It is **bit-equivalent** to `resize_bilinear.wgsl` under
//! [`crate::AlignCorners::HalfPixel`]: both compute
//! `s = clamp((o + 0.5)*in/out - 0.5, 0)` with `i1 = min(i0 + 1, in - 1)`, and
//! the extra high-side clamp here is provably inert once `i1` is clamped.
//! `crates/imaging/tests/device_ops.rs` pins that equivalence.
//!
//! Anything that *does* hold a `Gpu` must call [`crate::Ctx::resize`]. A host
//! loop is invisible to `--device` and reports host numbers under a device
//! label.
//!
//! The two reference resamplers below are the other exception: a model whose
//! reference preprocessing is Pillow's or torch's antialiased resize must
//! reproduce it exactly, and Pillow's is fixed-point integer arithmetic. They
//! run once per input image, before any model buffer exists.
//!
//! * [`resize_bicubic_pil`] - Pillow's `Image.resize(BICUBIC)` on uint8 RGB
//!   (what torchvision's `resize` does to a PIL image), byte for byte.
//! * [`resize_aa_planar`] - torch's `F.interpolate(..., antialias=True,
//!   align_corners=False)` (what `transforms.Resize(antialias=True)` does to a
//!   tensor), bilinear or bicubic.
//!
//! `crates/imaging/tests/resize_reference.rs` pins both against goldens.


use crate::Rgb8;
/// Bilinear resize of an **interleaved** `[h0, w0, c]` image to `[th, tw, c]`,
/// half-pixel (`align_corners = false`) — `cv2.resize` / `F.interpolate`
/// semantics, and what ZipDepth's reference preprocessing does.
///
/// `c` is a parameter rather than two functions: the workspace had a 3-channel
/// `resize_hwc` and a 1-channel `resize_map` sitting next to each other with
/// identical bodies, three times over. The channel loop is the only difference.
///
/// Row-parallel via `backend_cpu::par` — the workspace's only rayon seam. Each
/// output row reads `src` and writes its own chunk, so the result does not
/// depend on the thread count.
pub fn resize_bilinear_hwc(src: &[f32], c: u32, w0: u32, h0: u32, tw: u32, th: u32) -> Vec<f32> {
    assert!(c > 0 && w0 > 0 && h0 > 0 && tw > 0 && th > 0, "resize_bilinear_hwc: empty extent");
    assert_eq!(
        src.len(),
        (w0 * h0 * c) as usize,
        "resize_bilinear_hwc: source is not {w0}x{h0}x{c}"
    );
    let mut out = vec![0f32; (tw * th * c) as usize];
    let sx = w0 as f32 / tw as f32;
    let sy = h0 as f32 / th as f32;
    backend_cpu::par::rows_mut(&mut out, (tw * c) as usize, |y, row| {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, h0 as f32 - 1.0);
        let (y0, ty) = (fy.floor() as u32, fy - fy.floor());
        let y1 = (y0 + 1).min(h0 - 1);
        for x in 0..tw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, w0 as f32 - 1.0);
            let (x0, tx) = (fx.floor() as u32, fx - fx.floor());
            let x1 = (x0 + 1).min(w0 - 1);
            for ch in 0..c {
                let p = |xx: u32, yy: u32| src[((yy * w0 + xx) * c + ch) as usize];
                let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
                let bot = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
                row[(x * c + ch) as usize] = top * (1.0 - ty) + bot * ty;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_size_is_a_copy() {
        let src: Vec<f32> = (0..(4 * 3 * 3)).map(|i| i as f32).collect();
        assert_eq!(resize_bilinear_hwc(&src, 3, 4, 3, 4, 3), src);
    }

    #[test]
    fn single_channel_and_three_channel_agree_per_plane() {
        // The 1-channel `resize_map` and the 3-channel `resize_hwc` the workspace
        // used to carry separately are the same function; prove it.
        let (w0, h0) = (5u32, 4u32);
        let plane: Vec<f32> = (0..(w0 * h0)).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut hwc = vec![0f32; (w0 * h0 * 3) as usize];
        for (i, &v) in plane.iter().enumerate() {
            hwc[i * 3] = v;
            hwc[i * 3 + 1] = -v;
            hwc[i * 3 + 2] = 2.0 * v;
        }
        let (tw, th) = (9u32, 7u32);
        let a = resize_bilinear_hwc(&plane, 1, w0, h0, tw, th);
        let b = resize_bilinear_hwc(&hwc, 3, w0, h0, tw, th);
        for i in 0..(tw * th) as usize {
            assert_eq!(a[i], b[i * 3], "channel 0 must be bitwise identical at {i}");
        }
    }

    #[test]
    fn half_pixel_upsample_matches_the_closed_form() {
        // 2x upsample of [0, 1]: half-pixel puts the outputs at -0.25, 0.25,
        // 0.75, 1.25 in source coordinates, clamped to [0, 1].
        let got = resize_bilinear_hwc(&[0.0, 1.0], 1, 2, 1, 4, 1);
        for (g, w) in got.iter().zip([0.0f32, 0.25, 0.75, 1.0]) {
            assert!((g - w).abs() < 1e-6, "got {got:?}");
        }
    }
}

// ---- Pillow 8bpc resampling (Resample.c) ----

/// Pillow's fixed-point precision: `32 - 8 - 2` bits of fraction.
const PRECISION_BITS: i32 = 22;

/// The bicubic filter Pillow and torch's antialiased resampler share:
/// `a = -0.5`, support 2 (torch's non-antialiased bicubic uses `a = -0.75`).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Pillow's `precompute_coeffs` for one axis: per output position, the
/// source window `[xmin, xmin + n)` and its normalized fixed-point weights.
fn pil_coeffs(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    const SUPPORT: f64 = 2.0;
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let inv = 1.0 / filterscale;
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support).floor().max(0.0)) as usize;
        let xmax = (((center + support).ceil()) as usize).min(in_size) - xmin;
        let w: Vec<f64> = (0..xmax).map(|x| cubic((x as f64 + xmin as f64 - center + 0.5) * inv)).collect();
        let ww: f64 = w.iter().sum();
        for (x, &wx) in w.iter().enumerate() {
            let k = wx / ww;
            kk[xx * ksize + x] = if k < 0.0 { (-0.5 + k * (1i64 << PRECISION_BITS) as f64) as i32 } else { (0.5 + k * (1i64 << PRECISION_BITS) as f64) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    (ksize, bounds, kk)
}

fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// One Pillow pass over interleaved RGB8: `n_lines` lines of `stride`
/// pixels, resampled along the axis `at(line, i)` addresses, into `out_len`
/// positions per line.
fn pil_pass(src: &[u8], in_size: usize, out_size: usize, n_lines: usize, at: impl Fn(usize, usize) -> usize, dst_at: impl Fn(usize, usize) -> usize, dst_len: usize) -> Vec<u8> {
    let (ksize, bounds, kk) = pil_coeffs(in_size, out_size);
    let mut out = vec![0u8; dst_len];
    for line in 0..n_lines {
        for (o, &(min, n)) in bounds.iter().enumerate() {
            for ch in 0..3 {
                let mut ss = 1i32 << (PRECISION_BITS - 1);
                for i in 0..n {
                    ss += src[at(line, min + i) * 3 + ch] as i32 * kk[o * ksize + i];
                }
                out[dst_at(line, o) * 3 + ch] = clip8(ss);
            }
        }
    }
    out
}

/// Pillow's `Image.resize((nw, nh), BICUBIC)` on interleaved RGB8, byte for
/// byte: a horizontal then a vertical fixed-point pass, antialiased when
/// shrinking, each skipped when its axis keeps its size.
pub fn resize_bicubic_pil(img: &Rgb8, nw: usize, nh: usize) -> Rgb8 {
    let (w, h) = (img.w as usize, img.h as usize);
    assert!(w > 0 && h > 0 && nw > 0 && nh > 0, "resize_bicubic_pil: empty extent");
    let horiz = if nw != w { pil_pass(&img.px, w, nw, h, |y, x| y * w + x, |y, x| y * nw + x, nw * h * 3) } else { img.px.clone() };
    let px = if nh != h { pil_pass(&horiz, h, nh, nw, |x, y| y * nw + x, |x, y| y * nw + x, nw * nh * 3) } else { horiz };
    Rgb8 { w: nw as u32, h: nh as u32, px }
}

// ---- torch antialiased resampling (UpSampleKernel.cpp, `_upsample_*_aa`) ----

/// The filter of a torch antialiased resize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaFilter {
    /// Triangle, support 1.
    Bilinear,
    /// Cubic with `a = -0.5`, support 2.
    Bicubic,
}

impl AaFilter {
    fn support(self) -> f64 {
        match self {
            AaFilter::Bilinear => 1.0,
            AaFilter::Bicubic => 2.0,
        }
    }
    fn eval(self, x: f64) -> f64 {
        match self {
            AaFilter::Bilinear => (1.0 - x.abs()).max(0.0),
            AaFilter::Bicubic => cubic(x),
        }
    }
}

/// One axis of torch's antialiased resampler: per output position, the
/// first source index and the normalized weights. A float port of Pillow's
/// windowing: centre `(i + 0.5)·scale`, window widened by the scale when
/// shrinking, clipped to the input, then normalized.
fn aa_axis_weights(in_n: usize, out_n: usize, filter: AaFilter) -> Vec<(usize, Vec<f64>)> {
    let scale = in_n as f64 / out_n as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let inv = 1.0 / filterscale;
    (0..out_n)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax = (((center + support + 0.5).floor()) as usize).min(in_n);
            let mut ws: Vec<f64> = (xmin..xmax).map(|j| filter.eval((j as f64 - center + 0.5) * inv)).collect();
            let sum: f64 = ws.iter().sum();
            for w in ws.iter_mut() {
                *w /= sum;
            }
            (xmin, ws)
        })
        .collect()
}

/// torch's `F.interpolate(x, (out_h, out_w), mode, align_corners=False,
/// antialias=True)` of a planar `[c, in_h, in_w]` tensor, to `[c, out_h,
/// out_w]`: a horizontal then a vertical pass, accumulated in f64.
pub fn resize_aa_planar(x: &[f32], c: usize, in_h: usize, in_w: usize, out_h: usize, out_w: usize, filter: AaFilter) -> Vec<f32> {
    assert_eq!(x.len(), c * in_h * in_w, "resize_aa_planar: source is not [{c}, {in_h}, {in_w}]");
    let wy = aa_axis_weights(in_h, out_h, filter);
    let wx = aa_axis_weights(in_w, out_w, filter);
    let mut mid = vec![0.0f64; c * in_h * out_w];
    for ch in 0..c {
        for y in 0..in_h {
            let row = &x[(ch * in_h + y) * in_w..][..in_w];
            for (ox, (x0, ws)) in wx.iter().enumerate() {
                mid[(ch * in_h + y) * out_w + ox] = ws.iter().enumerate().map(|(k, w)| w * row[x0 + k] as f64).sum();
            }
        }
    }
    let mut out = vec![0.0f32; c * out_h * out_w];
    for ch in 0..c {
        for (oy, (y0, ws)) in wy.iter().enumerate() {
            for ox in 0..out_w {
                let acc: f64 = ws.iter().enumerate().map(|(k, w)| w * mid[(ch * in_h + y0 + k) * out_w + ox]).sum();
                out[(ch * out_h + oy) * out_w + ox] = acc as f32;
            }
        }
    }
    out
}
