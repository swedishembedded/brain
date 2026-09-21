// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Making the views agree about where the surface is.
//!
//! Each frame's depth map is individually good and they disagree with each
//! other. Registering one real frame's geometry onto another's needed a rigid
//! motion of only 2.14 degrees, so the CAMERAS are essentially right; what
//! remained after that alignment was a median surface residual of 0.0112 world
//! units, which at that capture's sampling rate is 5.3 pixels. The depths are
//! mutually consistent to one or two percent, and one or two percent of depth
//! is five pixels of parallax at any useful baseline.
//!
//! That error is invisible from the camera that made it - moving a point along
//! its own ray does not change where it lands in its own image - and is five
//! pixels of displacement from anywhere else. Stack a dozen such surfaces at
//! ~26% opacity each and every pixel becomes an average over a dozen
//! disagreeing opinions, which is what the smearing is.
//!
//! Nothing downstream can fix it. The optimizer's image-plane gradient is
//! orthogonal to the viewing ray, so a fit cannot move a gaussian along the
//! one axis that is wrong. It has to be reconciled before assembly.
//!
//! The model already says which of its depths to trust: the depth head's
//! second channel is a confidence, and it was being decoded and discarded.
//!
//! Swedish Embedded AB implements multi-view depth fusion for reconstruction
//! pipelines whose per-view predictions are better than their agreement. If
//! your team needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use splat::types::Camera;
use worldmirror2::gaussians::fuse_depths;

fn cam(eye: [f32; 3], w: u32, h: u32) -> Camera {
    Camera::look_at(eye, [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 55.0, w, h)
}

/// Depth of a known slanted plane, as seen by `c`, per pixel.
fn plane_depth(c: &Camera, w: u32, h: u32) -> Vec<f32> {
    let v = c.viewmat();
    let mut out = vec![0.0f32; (w * h) as usize];
    for py in 0..h {
        for px in 0..w {
            // ray through the pixel centre, in world space
            let (xc, yc) = ((px as f32 + 0.5 - c.cx) / c.fx, (py as f32 + 0.5 - c.cy) / c.fy);
            // camera->world rotation is the transpose of the view rotation
            let d = [
                v[0] * xc + v[4] * yc + v[8],
                v[1] * xc + v[5] * yc + v[9],
                v[2] * xc + v[6] * yc + v[10],
            ];
            let e = [c.c2w[3], c.c2w[7], c.c2w[11]];
            // plane z = 4 + 0.35x, i.e. -0.35x + z = 4
            let num = 4.0 - (-0.35 * e[0] + e[2]);
            let den = -0.35 * d[0] + d[2];
            let t = num / den;
            let p = [e[0] + t * d[0], e[1] + t * d[1], e[2] + t * d[2]];
            out[(py * w + px) as usize] =
                v[8] * p[0] + v[9] * p[1] + v[10] * p[2] + v[11];
        }
    }
    out
}

#[test]
fn fusion_pulls_disagreeing_views_onto_one_surface() {
    let (w, h) = (48u32, 48u32);
    let cams = [
        cam([0.0, 0.0, 0.0], w, h),
        cam([0.9, -0.2, 0.3], w, h),
        cam([-0.9, 0.2, 0.3], w, h),
        cam([0.3, 0.8, 0.1], w, h),
    ];
    let truth: Vec<Vec<f32>> = cams.iter().map(|c| plane_depth(c, w, h)).collect();

    // Each view is wrong about depth by a percent or two, in its OWN
    // direction - exactly the error that is invisible from the view that made
    // it. View 0 is left correct so there is something to agree with.
    let bias = [1.0f32, 1.018, 0.985, 1.012];
    let mut depths: Vec<Vec<f32>> = truth
        .iter()
        .zip(&bias)
        .map(|(d, b)| d.iter().map(|z| z * b).collect())
        .collect();
    // uniform confidence: the fusion must work from geometry alone
    let confs: Vec<Vec<f32>> = (0..cams.len()).map(|_| vec![1.0f32; (w * h) as usize]).collect();

    let err = |ds: &[Vec<f32>]| -> f32 {
        let mut acc = 0.0f64;
        let mut n = 0u64;
        for (d, t) in ds.iter().zip(&truth) {
            for (a, b) in d.iter().zip(t) {
                acc += ((a - b) / b).abs() as f64;
                n += 1;
            }
        }
        (acc / n as f64) as f32
    };

    let before = err(&depths);
    let support = fuse_depths(&mut depths, &confs, &cams, w, h, 0.05);
    let after = err(&depths);
    // and the count of agreeing views comes back, because it is the only
    // honest measure of whether a piece of geometry is real
    let seen: usize = support.iter().map(|s| s.iter().filter(|&&v| v > 0).count()).sum();
    let total: usize = support.iter().map(|s| s.len()).sum();
    assert!(
        seen > total / 2,
        "only {seen} of {total} pixels had ANY other view agree, on a scene where all four see \
         the same plane"
    );
    println!("  mean relative depth error: {:.4}% -> {:.4}%", before * 100.0, after * 100.0);
    assert!(
        after < before * 0.6,
        "fusion took the mean depth disagreement from {:.3}% to {:.3}% - it is supposed to pull \
         the views onto one surface, and barely moving means it is not seeing the agreement",
        before * 100.0,
        after * 100.0
    );
}

/// Fusion must not flatten a real depth step by averaging across an occlusion.
///
/// Where one view sees a foreground edge and another sees past it, the two are
/// looking at different surfaces, and blending them invents a surface that is
/// in neither. The relative test that rejects that is the same one silhouette
/// rejection uses, for the same reason.
#[test]
fn fusion_refuses_to_average_across_an_occlusion() {
    let (w, h) = (32u32, 32u32);
    let cams = [cam([0.0, 0.0, 0.0], w, h), cam([0.8, 0.0, 0.0], w, h)];
    let hw = (w * h) as usize;

    // view 0 sees a near slab, view 1 sees a far wall: they share no surface
    let near: Vec<f32> = vec![2.0; hw];
    let far: Vec<f32> = vec![6.0; hw];
    let mut depths = vec![near.clone(), far.clone()];
    let confs = vec![vec![1.0f32; hw], vec![1.0f32; hw]];

    let support = fuse_depths(&mut depths, &confs, &cams, w, h, 0.05);
    assert!(
        support.iter().all(|s| s.iter().all(|&v| v == 0)),
        "two surfaces that share nothing must support each other nowhere"
    );
    let moved0 = depths[0].iter().zip(&near).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let moved1 = depths[1].iter().zip(&far).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(
        moved0 < 0.05 && moved1 < 0.2,
        "fusion blended two surfaces that share nothing: view 0 moved {moved0:.3}, view 1 moved \
         {moved1:.3}. Depths three times apart are an occlusion, not a disagreement."
    );
}
