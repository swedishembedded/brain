// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 5-point similarity alignment to the standard FFHQ-512 template - the SAME
//! shape as `arcface::align` (Umeyama solve on the host + `grid_sample` on
//! the device), a different destination template and target size.
//!
//! `arcface::align::grid_sample` cannot be called directly from here: its
//! dispatch resolves `"grid_sample"` against ARCFACE's own kernel list, and
//! a `Step` is only meaningful to the `Gpu` handle whose kernel list built
//! it (the exact contract `flux1::inject`'s docs state, and the bug
//! `pulid::caps::Bundle::load` was fixed for) - so this crate registers and
//! dispatches its own copy of the same ~15 lines, against
//! [`crate::model::PIPELINES`].
//!
//! Template source: `facexlib.utils.face_restoration_helper.
//! FaceRestoreHelper`, `template_3points=False`, `face_size=512`,
//! `crop_ratio=(1,1)` - the exact constructor arguments the official PuLID
//! `pipeline_flux.py` uses. Verified against the real weights end to end by
//! `tests/parity.rs` (cosine 1.0000000000 against `facexlib`'s own output on
//! a real photo).

use gpu_core::{DeviceBuffer, Gpu};

/// The standard 5-landmark FFHQ-512 destination template (left eye, right
/// eye, nose, left mouth, right mouth - the order SCRFD emits), scaled for
/// `face_size=512`.
pub const FFHQ_DST_512: [[f32; 2]; 5] = [
    [192.98138, 239.94708],
    [318.90277, 240.1936],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
];

/// The 4-DOF similarity transform mapping five detected landmarks onto
/// [`FFHQ_DST_512`], as a row-major `[2, 3]` matrix `M` with `dst ≈ M ·
/// [x, y, 1]ᵀ`. `lmk` is `[5, 2]` `(x, y)` in source-image pixels, same
/// point order as the template - `arcface::align::estimate_norm`'s sibling
/// for this crate's own destination.
pub fn estimate_norm(lmk: &[f32]) -> Result<[f32; 6], String> {
    if lmk.len() != 10 {
        return Err(format!("estimate_norm: expected 5 (x, y) pairs = 10 values, got {}", lmk.len()));
    }
    let dst: Vec<f32> = FFHQ_DST_512.iter().flat_map(|p| [p[0], p[1]]).collect();
    model::hostmath::similarity_transform_2d(lmk, &dst, 5)
}

/// Build the `grid_sample` grid that applies `m` as a destination-to-source
/// warp - byte-for-byte the same construction as `arcface::align::warp_grid`
/// (shared math, `model::hostmath` only), duplicated here only because it is
/// three lines cheaper than threading a cross-crate dependency for it.
fn warp_grid(m: &[f32; 6], src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> Result<Vec<f32>, String> {
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

/// Warp a CHW source image (any value range/channel order - the warp is
/// per-channel and order-agnostic) to the 512x512 FFHQ crop: solve, build
/// the grid, dispatch `grid_sample` against THIS crate's own kernel list.
/// Returns the aligned `[3, 512, 512]` CHW buffer.
///
/// **Padding is zeros, not the reference's constant gray `(135,133,132)`
/// BGR fill** (`grid_sample.wgsl`'s only mode - `arcface::align`'s own doc
/// notes the same divergence at its own 112px template). Only affects thin
/// border strips when a detected face sits near the source image's edge;
/// unmeasured here, and the SAME gap the existing ArcFace alignment path
/// already carries at a different template/size.
pub fn norm_crop_512(gpu: &Gpu, src: &[f32], c: u32, h: u32, w: u32, lmk: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(src.len(), (c * h * w) as usize, "src must be [C, H, W]");
    const OUT: u32 = 512;
    let m = estimate_norm(lmk)?;
    let grid = warp_grid(&m, w, h, OUT, OUT)?;
    let xbuf = gpu.storage(src.len() as u64);
    gpu.write(&xbuf, bytemuck::cast_slice(src));
    let gbuf = gpu.storage(grid.len() as u64);
    gpu.write(&gbuf, bytemuck::cast_slice(&grid));
    let out: DeviceBuffer = gpu.storage((c * OUT * OUT) as u64);
    let s = gpu.step(crate::model::kernel("grid_sample"), &[&xbuf, &gbuf, &out], &[1, c, h, w, OUT, OUT, 0], c * OUT * OUT);
    gpu.submit(&[], &[s]);
    Ok(gpu.read(&out, (c * OUT * OUT) as usize))
}
