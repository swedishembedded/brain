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
