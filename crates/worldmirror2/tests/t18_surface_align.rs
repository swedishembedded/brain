// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T18 gate: gaussians lie IN the surface they represent.
//!
//! The model does not orient them to anything. Measured on a real capture, the
//! angle between a gaussian's shortest axis and the surface normal implied by
//! its own depth map is indistinguishable from random - median |cos| 0.416
//! against 0.5 for random, 5.0% within 20 degrees against 6% by chance - and
//! they are near-isotropic anyway (b/c median 1.38). Each one is a little ball
//! sitting at the right depth rather than a piece of surface, which leaves
//! visible gaps between them and haze when the surface is viewed edge on.
//!
//! Laying them flat is the observation behind 2D Gaussian Splatting
//! (arXiv:2403.17888), with one honest difference: 2DGS OPTIMISES flat disks
//! under normal-consistency and depth-distortion losses and rasterises them
//! with a ray-splat intersection, while this reshapes what a feed-forward pass
//! already produced. So the claim gated here is only the geometric one - that
//! the disc ends up in the surface - and not that it is equivalent to training
//! for it.
//!
//! Swedish Embedded AB implements reconstruction pipelines whose primitives
//! represent the geometry they are meant to, rather than merely sitting near
//! it. If your team needs that, you can procure our services by sending an
//! email to info@swedishembedded.com.

use splat::types::Camera;
use worldmirror2::gaussians::{assemble_from, AssembleOpts, HeadOutputs};

/// Head outputs describing one TILTED PLANE, so the correct normal is known in
/// closed form and is not axis-aligned (an axis-aligned plane would pass under
/// several wrong conventions).
fn tilted_plane(w: u32, h: u32, cam: &Camera, slope: f32) -> HeadOutputs {
    let hw = (w * h) as usize;
    let mut gsd = vec![0.0f32; 3 * hw];
    let mut gsp = vec![0.0f32; 12 * hw];
    for py in 0..h as usize {
        for px in 0..w as usize {
            let i = py * w as usize + px;
            // z = z0 + slope * x_cam, solved for the pixel's own ray
            let u = (px as f32 + 0.5 - cam.cx) / cam.fx;
            let z = 2.0 / (1.0 - slope * u);
            gsd[i] = z.ln();
            gsd[hw + i] = 0.0;
            gsd[2 * hw + i] = 9.0; // mask: keep
            gsp[i] = 1.0; // quat w, the rest zero -> identity, deliberately wrong
            for k in 4..7 {
                gsp[k * hw + i] = (0.004f32).ln();
            }
            gsp[11 * hw + i] = 3.0; // merge weight -> opacity ~0.95
        }
    }
    HeadOutputs { gsd: vec![gsd], gsp: vec![gsp], rgb: vec![vec![0.5; 3 * hw]], width: w, height: h }
}

#[test]
fn a_surface_aligned_gaussian_lies_in_the_surface_it_represents() {
    let (w, h) = (64u32, 48u32);
    let cam = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, -1.0, 0.0], 60.0, w, h);
    let slope = 0.35;
    let heads = tilted_plane(w, h, &cam, slope);
    // The plane's camera-space normal, from z = z0 + slope*x  ->  n ~ (slope, 0, -1)
    let n_cam = {
        let v = [slope, 0.0, -1.0f32];
        let l = (v[0] * v[0] + v[2] * v[2]).sqrt();
        [v[0] / l, 0.0, v[2] / l]
    };
    let m = &cam.c2w;
    let n_world = [
        m[0] * n_cam[0] + m[1] * n_cam[1] + m[2] * n_cam[2],
        m[4] * n_cam[0] + m[5] * n_cam[1] + m[6] * n_cam[2],
        m[8] * n_cam[0] + m[9] * n_cam[1] + m[10] * n_cam[2],
    ];

    let ratio = 4.0;
    let opts = AssembleOpts {
        edge_depth_rtol: 0.0,
        fuse_depth_rtol: 0.0,
        surface_align: ratio,
        ..Default::default()
    };
    let (s, _, _) = assemble_from(&heads, std::slice::from_ref(&cam), &opts);
    assert!(s.len() > 1000, "only {} gaussians assembled", s.len());

    let mut worst_cos = 1.0f32;
    let mut worst_ratio = 0.0f32;
    for i in 0..s.len() {
        let sc = [s.scales[i * 3], s.scales[i * 3 + 1], s.scales[i * 3 + 2]];
        let q = [s.quats[i * 4], s.quats[i * 4 + 1], s.quats[i * 4 + 2], s.quats[i * 4 + 3]];
        // the axis belonging to the SMALLEST scale, rotated into world
        let k = (0..3).min_by(|a, b| sc[*a].total_cmp(&sc[*b])).unwrap();
        let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
        let r = [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ];
        let axis = [r[0][k], r[1][k], r[2][k]];
        let c = (axis[0] * n_world[0] + axis[1] * n_world[1] + axis[2] * n_world[2]).abs();
        worst_cos = worst_cos.min(c);
        let mut v = sc;
        v.sort_by(f32::total_cmp);
        worst_ratio = worst_ratio.max((v[2] / v[0] - ratio).abs());
    }
    assert!(
        worst_cos > 0.999,
        "a gaussian's shortest axis is {:.1} degrees off the surface normal",
        worst_cos.acos().to_degrees()
    );
    assert!(worst_ratio < 1e-3, "thickness ratio is off by {worst_ratio:.4} from the requested {ratio}");
}

/// Off by default would make this dead code in every real run, and on by
/// default has to be a deliberate, visible choice rather than a leftover.
#[test]
fn surface_alignment_is_on_by_default() {
    assert_eq!(AssembleOpts::default().surface_align, 4.0);
}
