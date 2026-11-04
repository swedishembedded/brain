// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The filtering the reference runs between the model's dense per-pixel
//! predictions and the scene it writes.
//!
//! Brain turned nearly every predicted pixel straight into a gaussian. The
//! reference does considerably more first, and each step removes a specific
//! class of splat that otherwise survives into the cloud and shows up as
//! blur from any viewpoint but the one that made it:
//!
//! * the model's own validity mask drops pixels it says are not real geometry;
//! * a depth-discontinuity test drops silhouette pixels, whose depth is
//!   genuinely ambiguous between foreground and background and which land
//!   somewhere in between;
//! * weighted voxel fusion collapses the near-duplicate surfaces that
//!   overlapping views each predict at slightly different depths;
//! * the largest 2% of gaussians by scale are discarded outright.
//!
//! Swedish Embedded AB implements feed-forward reconstruction pipelines that
//! match their reference implementations end to end, not only at the tensor
//! boundary. If your team needs that parity, you can procure our services by
//! sending an email to info@swedishembedded.com.

use worldmirror2::gaussians::depth_edges;

/// A silhouette is a step in the depth map, and the pixels ON it are the ones
/// whose predicted depth is a blend of the two surfaces. They must go.
#[test]
fn a_depth_step_is_detected_and_a_smooth_gradient_is_not() {
    let (w, h) = (16usize, 8usize);
    // left half at z=1, right half at z=2: a 100% relative step down the middle
    let step: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 1.0 } else { 2.0 }).collect();
    let e = depth_edges(&step, w, h, 0.03);
    let flagged: Vec<usize> = (0..w).filter(|&x| e[(h / 2) * w + x]).collect();
    assert!(!flagged.is_empty(), "a 2:1 depth step was not detected at all");
    assert!(
        flagged.iter().all(|&x| (w / 2 - 1..=w / 2).contains(&x)),
        "the step flagged columns {flagged:?}, expected only the two either side of the seam"
    );

    // a gentle ramp is real geometry, not a silhouette, and must survive
    let ramp: Vec<f32> = (0..w * h).map(|i| 1.0 + 0.004 * (i % w) as f32).collect();
    let e = depth_edges(&ramp, w, h, 0.03);
    assert_eq!(e.iter().filter(|&&v| v).count(), 0, "a 0.4%-per-pixel ramp was flagged as an edge");
}

/// The test is only meaningful if the threshold is actually the knob: the same
/// gradient has to be rejected once it is steep enough.
#[test]
fn the_edge_threshold_is_what_decides() {
    let (w, h) = (16usize, 4usize);
    let ramp: Vec<f32> = (0..w * h).map(|i| 1.0 + 0.05 * (i % w) as f32).collect();
    assert_eq!(depth_edges(&ramp, w, h, 0.20).iter().filter(|&&v| v).count(), 0);
    assert!(depth_edges(&ramp, w, h, 0.01).iter().any(|&v| v), "a 5%-per-pixel ramp survived a 1% tolerance");
}

/// Upstream's `save_gs_ply` computes the 98th percentile of each gaussian's
/// LARGEST axis and keeps only those at or below it. A handful of enormous
/// translucent splats is enough to make a sharp scene look like it was blurred.
#[test]
fn the_largest_gaussians_by_scale_are_dropped() {
    let mut s = splat::types::Splats::default();
    for i in 0..100 {
        s.means.extend_from_slice(&[i as f32, 0.0, 3.0]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        // one axis grows with i, so the biggest indices are the biggest splats
        s.scales.extend_from_slice(&[0.01, 0.01 + i as f32 * 0.001, 0.01]);
        s.opacities.push(0.9);
        s.colors.extend_from_slice(&[0.5, 0.5, 0.5]);
    }
    let out = splat::prune::drop_largest_scales(&s, 0.98);
    assert_eq!(out.len(), 98, "expected the top 2% of 100 gaussians to go, kept {}", out.len());
    let biggest = (0..out.len())
        .map(|i| out.scales[i * 3..i * 3 + 3].iter().fold(0.0f32, |m, &v| m.max(v)))
        .fold(0.0f32, f32::max);
    assert!(biggest <= 0.01 + 97.0 * 0.001 + 1e-6, "a gaussian above the 98th percentile survived");
    // every field stays in step - a filter that drops a mean but keeps its colour
    // corrupts the whole cloud silently
    assert_eq!(out.means.len(), out.len() * 3);
    assert_eq!(out.quats.len(), out.len() * 4);
    assert_eq!(out.colors.len(), out.len() * 3);
    assert_eq!(out.opacities.len(), out.len());
}

/// Dropping nothing is a valid request and must not corrupt the scene.
#[test]
fn a_quantile_of_one_keeps_everything() {
    let mut s = splat::types::Splats::default();
    for i in 0..10 {
        s.means.extend_from_slice(&[i as f32, 0.0, 3.0]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[0.01 * (i + 1) as f32; 3]);
        s.opacities.push(0.9);
        s.colors.extend_from_slice(&[0.5, 0.5, 0.5]);
    }
    assert_eq!(splat::prune::drop_largest_scales(&s, 1.0).len(), 10);
}
