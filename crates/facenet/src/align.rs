// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 5-point similarity alignment to the canonical 112×112 ArcFace template.
//!
//! Two halves, deliberately split by where each belongs:
//!
//! * **the solve** is host math over 20 numbers and lives in
//!   `model::hostmath::similarity_transform_2d` (Umeyama), with
//!   `invert_affine_2x3` for the sampling direction. Nothing face-specific about
//!   either, so neither is duplicated here;
//! * **the warp** is per-pixel work over a whole image and therefore a KERNEL —
//!   `grid_sample.wgsl`. There is no host warp loop in this crate.
//!
//! # cv2 parity is 0.5/255, and that is expected
//!
//! `cv2.warpAffine(INTER_LINEAR)` interpolates with 5-bit fixed-point weights;
//! `grid_sample` (and torch's) is fp32 bilinear. Measured on all three golden
//! cases, the two differ by **max 0.500/255**. So this path is gated *exactly*
//! against the reference `grid_sample` output and only loosely against the cv2
//! one. Reporting a tight match to cv2 would mean the grid was wrong in a way
//! that happened to cancel.

use gpu_core::{DeviceBuffer, Gpu};

use crate::config::ARCFACE_DST_112;

/// The 4-DOF similarity transform mapping five detected landmarks onto the
/// ArcFace template, as a row-major `[2, 3]` matrix `M` with
/// `dst ≈ M · [x, y, 1]ᵀ`.
///
/// `lmk` is `[5, 2]` `(x, y)` in source-image pixels, in
/// left-eye / right-eye / nose / left-mouth / right-mouth order — the order
/// SCRFD emits and the order [`ARCFACE_DST_112`] is written in. A permuted
/// landmark set still solves, and produces a plausible, wrong crop.
///
/// The fit has an irreducible residual (5 points, 4 degrees of freedom): ~1.4 px
/// on the golden cases. That is a property of the template, not an error — a
/// caller asserting an exact landmark match would be asserting something false.
pub fn estimate_norm(lmk: &[f32]) -> Result<[f32; 6], String> {
    if lmk.len() != 10 {
        return Err(format!("estimate_norm: expected 5 (x, y) pairs = 10 values, got {}", lmk.len()));
    }
    let dst: Vec<f32> = ARCFACE_DST_112.iter().flat_map(|p| [p[0], p[1]]).collect();
    model::hostmath::similarity_transform_2d(lmk, &dst, 5)
}

/// Build the `grid_sample` grid that applies `m` as a destination-to-source warp.
///
/// Layout `[Ho, Wo, 2]` as `(gx, gy)`, normalised for `align_corners = false`:
///
/// ```text
/// (x_s, y_s) = M⁻¹ · [x_d, y_d, 1]ᵀ        (pixel CENTRES at integer coords)
/// gx = (2·x_s + 1)/W_src − 1               gy = (2·y_s + 1)/H_src − 1
/// ```
///
/// The normalisation is the exact inverse of `grid_sample.wgsl`'s
/// `align_corners = 0` unnormalise (`ix = ((gx+1)·W − 1)/2`), so `ix == x_s`.
/// The `align_corners = 1` convention is half a pixel away, looks equally
/// plausible, and no gradient check can tell the two apart — which is why the
/// mode is passed explicitly at the dispatch below rather than defaulted.
///
/// Host code, not a kernel: this is `Ho·Wo` *geometry*, evaluated once per face
/// and consumed by the kernel that does the per-pixel work.
pub fn warp_grid(m: &[f32; 6], src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> Result<Vec<f32>, String> {
    let inv = model::hostmath::invert_affine_2x3(m)?;
    let mut g = vec![0.0f32; (out_h * out_w * 2) as usize];
    for y in 0..out_h {
        for x in 0..out_w {
            let (xf, yf) = (x as f32, y as f32);
            let xs = inv[0] * xf + inv[1] * yf + inv[2];
            let ys = inv[3] * xf + inv[4] * yf + inv[5];
            let i = ((y * out_w + x) * 2) as usize;
            g[i] = (2.0 * xs + 1.0) / src_w as f32 - 1.0;
            g[i + 1] = (2.0 * ys + 1.0) / src_h as f32 - 1.0;
        }
    }
    Ok(g)
}

/// Resample `x` (`[1, C, H, W]` on device) through `grid` (`[Ho, Wo, 2]`, host)
/// with bilinear interpolation and zero padding.
///
/// `grid_sample` Params — read before dispatching, since a mismatched param
/// list is silently wrong, not a crash — are `[N, C, H, W, Ho, Wo, align_corners]`, bindings `(x, grid, y)`, one
/// invocation per OUTPUT element (`N*C*Ho*Wo`). `align_corners = 0` is PyTorch's
/// default and what this grid is built for; the padding mode is `'zeros'`, which
/// drops each out-of-range corner tap individually (it is NOT clamp-to-edge).
// The 8 positional args ARE the kernel's contract — (device, input, its NCHW
// extent, the grid, the output extent). Bundling them into a struct would put a
// second spelling of `Shape` in this crate for one call site.
#[allow(clippy::too_many_arguments)]
pub fn grid_sample(
    gpu: &Gpu,
    x: &DeviceBuffer,
    c: u32,
    h: u32,
    w: u32,
    grid: &[f32],
    out_h: u32,
    out_w: u32,
) -> DeviceBuffer {
    assert_eq!(grid.len(), (out_h * out_w * 2) as usize, "grid must be [Ho, Wo, 2]");
    let gbuf = gpu.storage(grid.len() as u64);
    gpu.write(&gbuf, bytemuck::cast_slice(grid));
    let out = gpu.storage((c * out_h * out_w) as u64);
    let total = c * out_h * out_w;
    let s = gpu.step(
        crate::model::kernel("grid_sample"),
        &[x, &gbuf, &out],
        &[1, c, h, w, out_h, out_w, 0],
        total,
    );
    gpu.submit(&[], &[s]);
    out
}

/// Warp a CHW source image to the 112×112 ArcFace crop: solve, build the grid,
/// dispatch `grid_sample`. Returns `(aligned CHW, M)`.
///
/// `src` is `[C, H, W]` in whatever value range and channel order the caller
/// wants — the warp is per-channel and order-agnostic, so it serves both a raw
/// BGR crop and an already-normalised blob.
pub fn norm_crop_chw(
    gpu: &Gpu,
    src: &[f32],
    c: u32,
    h: u32,
    w: u32,
    lmk: &[f32],
    out: u32,
) -> Result<(Vec<f32>, [f32; 6]), String> {
    assert_eq!(src.len(), (c * h * w) as usize, "src must be [C, H, W]");
    let m = estimate_norm(lmk)?;
    let grid = warp_grid(&m, w, h, out, out)?;
    let xbuf = gpu.storage(src.len() as u64);
    gpu.write(&xbuf, bytemuck::cast_slice(src));
    let ybuf = grid_sample(gpu, &xbuf, c, h, w, &grid, out, out);
    Ok((gpu.read(&ybuf, (c * out * out) as usize), m))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fitting the template to ITSELF must give the identity, and the grid must
    /// then be the `align_corners=false` identity grid.
    #[test]
    fn the_template_maps_to_itself_by_the_identity() {
        let lmk: Vec<f32> = ARCFACE_DST_112.iter().flat_map(|p| [p[0], p[1]]).collect();
        let m = estimate_norm(&lmk).unwrap();
        assert!((m[0] - 1.0).abs() < 1e-4, "{m:?}");
        assert!(m[1].abs() < 1e-4 && m[3].abs() < 1e-4, "{m:?}");
        assert!(m[2].abs() < 1e-3 && m[5].abs() < 1e-3, "{m:?}");

        let g = warp_grid(&m, 112, 112, 112, 112).unwrap();
        // pixel (0,0) centre -> gx = gy = 1/112 - 1
        assert!((g[0] - (1.0 / 112.0 - 1.0)).abs() < 1e-4, "{}", g[0]);
        // pixel (111,111) -> (2*111+1)/112 - 1 = 1 - 1/112
        let last = ((111 * 112 + 111) * 2) as usize;
        assert!((g[last] - (1.0 - 1.0 / 112.0)).abs() < 1e-4, "{}", g[last]);
    }

    #[test]
    fn a_landmark_set_of_the_wrong_length_is_an_error() {
        assert!(estimate_norm(&[0.0; 8]).is_err());
    }
}
