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
}

impl Default for AssembleOpts {
    fn default() -> Self {
        AssembleOpts { min_opacity: 0.01, max_depth: 0.0 }
    }
}

/// Read the GS head outputs for all frames and build the host scene +
/// per-frame maps. `frames_chw` = the raw [0,1] input frames (color source).
pub fn assemble(
    gpu: &Gpu,
    model: &Mirror,
    frames_chw: &[f32],
    s: usize,
    width: u32,
    height: u32,
    opts: &AssembleOpts,
) -> (Splats, Vec<Camera>) {
    let cams = decode_cameras(&model.cam_pred_raw(), s, width, height);
    let hw = (width * height) as usize;
    let mut out = Splats::default();
    for (fi, cam) in cams.iter().enumerate() {
        let gsd = gpu.read(model.head_out(Head::GsDepth, fi), 3 * hw);
        let gsp = gpu.read(model.head_out(Head::GsParams, fi), 12 * hw);
        let rgb = &frames_chw[fi * 3 * hw..(fi + 1) * 3 * hw];
        let m = &cam.c2w;
        for py in 0..height as usize {
            for px in 0..width as usize {
                let i = py * width as usize + px;
                let op = sigmoid(gsp[7 * hw + i]);
                if op < opts.min_opacity {
                    continue;
                }
                let z = gsd[i].exp();
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
                out.quats.extend_from_slice(&[
                    gsp[i],
                    gsp[hw + i],
                    gsp[2 * hw + i],
                    gsp[3 * hw + i],
                ]);
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
            }
        }
    }
    (out, cams)
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
