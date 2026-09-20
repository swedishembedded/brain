// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side output assembly: camera decode (`camera_utils.py` /
//! `rotation.py` parity) and per-pixel Gaussian construction
//! (`rasterization.py::prepare_splats` + `act_gs.py` parity).
//!
//! Per source pixel: means = gs_depth back-projected through the predicted
//! camera (pixel-index grid, no +0.5 — reference convention); quat wxyz
//! normalized; scales exp-clamped at 0.3; opacity sigmoid; color =
//! SH-decode(residual + RGB2SH(rgb)) = residual·C0 + rgb.

use gpu_core::Gpu;
use splat::ply::SH_C0;
use splat::types::{Camera, Splats};

use crate::model::{Head, Mirror};

/// Decode raw camera 9-vectors `[s,9]` (fov relu applied here) into cameras.
pub fn decode_cameras(raw: &[f32], s: usize, width: u32, height: u32) -> Vec<Camera> {
    let (w, h) = (width as f32, height as f32);
    (0..s)
        .map(|i| {
            let v = &raw[i * 9..i * 9 + 9];
            let (t, q) = (&v[0..3], &v[3..7]);
            let fov_v = v[7].max(0.0);
            let fov_u = v[8].max(0.0);
            // scalar-last xyzw quaternion -> w2c rotation
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-12);
            let (x, y, z, r) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
            let rm = [
                1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * r), 2.0 * (x * z + y * r),
                2.0 * (x * y + z * r), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * r),
                2.0 * (x * z - y * r), 2.0 * (y * z + x * r), 1.0 - 2.0 * (x * x + y * y),
            ];
            // c2w = rigid inverse of [R|t]
            let rt = [rm[0], rm[3], rm[6], rm[1], rm[4], rm[7], rm[2], rm[5], rm[8]];
            let ct = [
                -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]),
                -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]),
                -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]),
            ];
            let c2w = [
                rt[0], rt[1], rt[2], ct[0],
                rt[3], rt[4], rt[5], ct[1],
                rt[6], rt[7], rt[8], ct[2],
                0.0, 0.0, 0.0, 1.0,
            ];
            Camera {
                c2w,
                fx: 0.5 * w / (0.5 * fov_u).tan().max(1e-6),
                fy: 0.5 * h / (0.5 * fov_v).tan().max(1e-6),
                cx: 0.5 * w,
                cy: 0.5 * h,
                width,
                height,
            }
        })
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Assembly filters. `min_opacity` drops near-transparent gaussians early;
/// `max_depth` clips runaway sky depths (0 = off).
pub struct AssembleOpts {
    pub min_opacity: f32,
    pub max_depth: f32,
    /// Keep a pixel only where the GS head's own validity mask, after a
    /// sigmoid, is above this. The reference thresholds at 0.5; `0` disables.
    pub gs_mask_threshold: f32,
    /// Relative depth tolerance for silhouette rejection. A pixel whose depth
    /// differs from a 4-neighbour by more than this fraction sits ON a
    /// discontinuity, where the predicted depth is a blend of two surfaces
    /// and the gaussian lands between them. Reference `0.03`; `0` disables.
    pub edge_depth_rtol: f32,
}

impl Default for AssembleOpts {
    fn default() -> Self {
        AssembleOpts { min_opacity: 0.01, max_depth: 0.0, gs_mask_threshold: 0.5, edge_depth_rtol: 0.03 }
    }
}

/// Flag every pixel that sits on a depth discontinuity.
///
/// `rtol` is relative, as upstream's `depth_edge(..., rtol=...)` is: what
/// matters at a silhouette is the RATIO of the two depths, not their
/// difference, so the same test works at any distance. Both sides of a step
/// are flagged, because the blend that makes the pixel unreliable straddles
/// it.
pub fn depth_edges(z: &[f32], w: usize, h: usize, rtol: f32) -> Vec<bool> {
    let mut out = vec![false; w * h];
    if rtol <= 0.0 {
        return out;
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a = z[i];
            if !(a.is_finite() && a > 0.0) {
                out[i] = true;
                continue;
            }
            let mut nb = [None; 4];
            if x > 0 { nb[0] = Some(z[i - 1]); }
            if x + 1 < w { nb[1] = Some(z[i + 1]); }
            if y > 0 { nb[2] = Some(z[i - w]); }
            if y + 1 < h { nb[3] = Some(z[i + w]); }
            for b in nb.into_iter().flatten() {
                if !(b.is_finite() && b > 0.0) || (a - b).abs() / a.min(b) > rtol {
                    out[i] = true;
                    break;
                }
            }
        }
    }
    out
}

/// Read the GS head outputs for all frames and build the host scene.
/// `frames_chw` = the raw [0,1] input frames (color source). Also returns the
/// per-gaussian sigmoid merge weights for `splat::prune` - the same values
/// the scene's opacities carry, since the merge weight IS the opacity of the
/// assembled scene.
pub fn assemble(
    gpu: &Gpu,
    model: &Mirror,
    frames_chw: &[f32],
    s: usize,
    width: u32,
    height: u32,
    opts: &AssembleOpts,
) -> (Splats, Vec<Camera>, Vec<f32>) {
    let cams = decode_cameras(&model.cam_pred_raw(), s, width, height);
    let hw = (width * height) as usize;
    let mut out = Splats::default();
    let mut weights = Vec::new();
    for (fi, cam) in cams.iter().enumerate() {
        // The GS depth head emits THREE channels - depth, confidence, and a
        // validity mask - laid out like `Head::Depth`'s. Only the first was
        // ever read, so pixels the model itself reports as not-geometry became
        // gaussians anyway.
        let gsd = gpu.read(model.head_out(Head::GsDepth, fi), 3 * hw);
        let gsp = gpu.read(model.head_out(Head::GsParams, fi), 12 * hw);
        let rgb = &frames_chw[fi * 3 * hw..(fi + 1) * 3 * hw];
        let m = &cam.c2w;
        let depth: Vec<f32> = gsd[..hw].iter().map(|v| v.exp()).collect();
        let edge = depth_edges(&depth, width as usize, height as usize, opts.edge_depth_rtol);
        for py in 0..height as usize {
            for px in 0..width as usize {
                let i = py * width as usize + px;
                // Channel 11, the learned merge weight, IS the opacity of the
                // assembled scene - not channel 7, the raw opacity head. The
                // reference's assembly substitutes it (see
                // `splat::prune::voxel_merge`, which computes the same thing
                // as a weighted mean when it merges), and the difference is
                // not cosmetic: measured on a six-photograph reconstruction
                // rendered from the camera the model recovered for it, channel
                // 7 gives 12.5 dB carrying 7% of the photograph's detail, and
                // channel 11 gives 22.5 dB carrying 47%. Channel 7 renders as
                // a milky smear.
                let op = sigmoid(gsp[11 * hw + i]);
                if op < opts.min_opacity {
                    continue;
                }
                if opts.gs_mask_threshold > 0.0 && sigmoid(gsd[2 * hw + i]) < opts.gs_mask_threshold {
                    continue;
                }
                if edge[i] {
                    continue;
                }
                let z = depth[i];
                if opts.max_depth > 0.0 && z > opts.max_depth {
                    continue;
                }
                let xc = (px as f32 - cam.cx) * z / cam.fx;
                let yc = (py as f32 - cam.cy) * z / cam.fy;
                out.means.extend_from_slice(&[
                    m[0] * xc + m[1] * yc + m[2] * z + m[3],
                    m[4] * xc + m[5] * yc + m[6] * z + m[7],
                    m[8] * xc + m[9] * yc + m[10] * z + m[11],
                ]);
                // Normalized here, not left to the renderer. The module doc
                // has always said "quat wxyz normalized" and the code did not
                // do it; upstream normalizes before encoding, and a
                // non-unit quaternion scales the covariance it rotates.
                let q = [gsp[i], gsp[hw + i], gsp[2 * hw + i], gsp[3 * hw + i]];
                let qn = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-8);
                out.quats.extend_from_slice(&[q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn]);
                out.scales.extend_from_slice(&[
                    gsp[4 * hw + i].exp().min(0.3),
                    gsp[5 * hw + i].exp().min(0.3),
                    gsp[6 * hw + i].exp().min(0.3),
                ]);
                out.opacities.push(op);
                out.colors.extend_from_slice(&[
                    gsp[8 * hw + i] * SH_C0 + rgb[i],
                    gsp[9 * hw + i] * SH_C0 + rgb[hw + i],
                    gsp[10 * hw + i] * SH_C0 + rgb[2 * hw + i],
                ]);
                weights.push(op);
            }
        }
    }
    (out, cams, weights)
}

/// Depth/normal/confidence maps for one frame, activations applied.
pub struct FrameMaps {
    pub depth: Vec<f32>,   // exp
    pub conf: Vec<f32>,    // 1+exp
    pub mask: Vec<f32>,    // sigmoid
    pub normals: Vec<f32>, // unit, [3,H,W]
}

pub fn frame_maps(gpu: &Gpu, model: &Mirror, fi: usize, width: u32, height: u32) -> FrameMaps {
    let hw = (width * height) as usize;
    let d = gpu.read(model.head_out(Head::Depth, fi), 3 * hw);
    let n = gpu.read(model.head_out(Head::Normals, fi), 4 * hw);
    let mut normals = vec![0.0f32; 3 * hw];
    for i in 0..hw {
        let v = [n[i], n[hw + i], n[2 * hw + i]];
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
        normals[i] = v[0] / len;
        normals[hw + i] = v[1] / len;
        normals[2 * hw + i] = v[2] / len;
    }
    FrameMaps {
        depth: d[..hw].iter().map(|&v| v.exp()).collect(),
        conf: d[hw..2 * hw].iter().map(|&v| 1.0 + v.exp()).collect(),
        mask: d[2 * hw..3 * hw].iter().map(|&v| sigmoid(v)).collect(),
        normals,
    }
}
