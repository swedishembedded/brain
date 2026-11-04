// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Voxel-merge prune parity with the reference `prune_gs` semantics:
//! floor(mean/voxel) grouping, weighted mean/scale/color averages,
//! opacity = Σw²/Σw, quats normalized → weighted sum → renormalized.

use splat::prune::voxel_merge;
use splat::types::Splats;

fn splats3() -> (Splats, Vec<f32>) {
    // g0 and g1 share voxel (0,0,0) at size 0.1; g2 is alone in (1,0,0).
    let mut s = Splats::default();
    s.means.extend_from_slice(&[0.02, 0.03, 0.01, 0.08, 0.01, 0.09, 0.15, 0.05, 0.05]);
    // g0 unit quat, g1 unnormalized (2,0,0,0) — must be normalized pre-merge
    s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    s.scales.extend_from_slice(&[0.01, 0.02, 0.03, 0.03, 0.02, 0.01, 0.05, 0.05, 0.05]);
    s.opacities.extend_from_slice(&[0.8, 0.4, 0.9]);
    s.colors.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.4, 0.6]);
    let weights = vec![0.75, 0.25, 0.5];
    (s, weights)
}

#[test]
fn merges_shared_voxel_weighted() {
    let (s, w) = splats3();
    let out = voxel_merge(&s, &w, 0.1, 0);
    assert_eq!(out.len(), 2);
    // voxel keys sort as (0,0,0) then (1,0,0)
    let wsum = 1.0f32;
    for d in 0..3 {
        let want = (0.75 * s.means[d] + 0.25 * s.means[3 + d]) / wsum;
        assert!((out.means[d] - want).abs() < 1e-6, "mean[{d}]");
        let want = (0.75 * s.scales[d] + 0.25 * s.scales[3 + d]) / wsum;
        assert!((out.scales[d] - want).abs() < 1e-6, "scale[{d}]");
        let want = (0.75 * s.colors[d] + 0.25 * s.colors[3 + d]) / wsum;
        assert!((out.colors[d] - want).abs() < 1e-6, "color[{d}]");
    }
    // opacity: Σw²/Σw = (0.5625 + 0.0625) / 1.0
    assert!((out.opacities[0] - 0.625).abs() < 1e-6);
    // quats: both normalize to (1,0,0,0) → merged (1,0,0,0)
    assert!((out.quats[0] - 1.0).abs() < 1e-6);
    // singleton voxel passes through exactly (up to quat normalization)
    for d in 0..3 {
        assert!((out.means[3 + d] - s.means[6 + d]).abs() < 1e-6);
        assert!((out.colors[3 + d] - s.colors[6 + d]).abs() < 1e-6);
    }
}

#[test]
fn singleton_opacity_is_weight_scaled() {
    // Reference semantics: even a lone gaussian gets opacity w²·1/w = w — NOT
    // its own opacity. The merged opacity depends only on the weights.
    let (s, w) = splats3();
    let out = voxel_merge(&s, &w, 0.1, 0);
    assert!((out.opacities[1] - 0.5).abs() < 1e-6, "got {}", out.opacities[1]);
}

#[test]
fn negative_coords_group_by_floor() {
    let mut s = Splats::default();
    // floor(-0.001/0.1) = -1 vs floor(0.001/0.1) = 0 — distinct voxels
    s.means.extend_from_slice(&[-0.001, 0.0, 0.0, 0.001, 0.0, 0.0]);
    s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    s.scales.extend_from_slice(&[0.01; 6]);
    s.opacities.extend_from_slice(&[0.5, 0.5]);
    s.colors.extend_from_slice(&[0.5; 6]);
    let out = voxel_merge(&s, &[1.0, 1.0], 0.1, 0);
    assert_eq!(out.len(), 2);
}

#[test]
fn max_points_keeps_heaviest() {
    let (s, w) = splats3();
    // voxel (0,0,0) has Σw = 1.0, voxel (1,0,0) has Σw = 0.5
    let out = voxel_merge(&s, &w, 0.1, 1);
    assert_eq!(out.len(), 1);
    assert!(out.means[0] < 0.1, "kept the heavier (0,0,0) voxel");
}
