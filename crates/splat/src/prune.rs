// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Voxel-merge pruning of per-pixel gaussian clouds (reference `prune_gs`).
//!
//! Multi-view WorldMirror scenes place one gaussian per source pixel, so
//! overlapping views produce many near-duplicate splats. The reference
//! collapses them onto a voxel grid: `floor(mean / voxel)` groups, then
//! per-voxel weighted averages of means/scales/colors, `Σw²/Σw` opacity
//! (the learned weight channel replaces the raw opacity), and quaternions
//! normalized → weighted-summed → renormalized.

use crate::types::Splats;
use std::collections::HashMap;

/// Merge `splats` (with per-gaussian sigmoid weights) onto a `voxel`-sized
/// grid. `max_points > 0` additionally keeps only the heaviest (largest Σw)
/// voxels. `sh_rest` is dropped — the reference merges DC only. Output order
/// is sorted by voxel coordinate (deterministic).
pub fn voxel_merge(splats: &Splats, weights: &[f32], voxel: f32, max_points: usize) -> Splats {
    let n = splats.len();
    assert_eq!(weights.len(), n, "one weight per gaussian");
    // group: voxel key -> accumulator index
    let mut slots: HashMap<[i64; 3], usize> = HashMap::new();
    let mut keys: Vec<[i64; 3]> = Vec::new();
    let mut acc: Vec<[f64; 15]> = Vec::new(); // wsum,w2sum, 3 mean, 4 quat, 3 scale, 3 color
    for (i, &wi) in weights.iter().enumerate() {
        let key = [
            (splats.means[i * 3] / voxel).floor() as i64,
            (splats.means[i * 3 + 1] / voxel).floor() as i64,
            (splats.means[i * 3 + 2] / voxel).floor() as i64,
        ];
        let slot = *slots.entry(key).or_insert_with(|| {
            keys.push(key);
            acc.push([0.0; 15]);
            acc.len() - 1
        });
        let a = &mut acc[slot];
        let w = wi as f64;
        a[0] += w;
        a[1] += w * w;
        for d in 0..3 {
            a[2 + d] += w * splats.means[i * 3 + d] as f64;
            a[9 + d] += w * splats.scales[i * 3 + d] as f64;
            a[12 + d] += w * splats.colors[i * 3 + d] as f64;
        }
        let q = &splats.quats[i * 4..i * 4 + 4];
        let qn = (q.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sqrt() + 1e-8;
        for d in 0..4 {
            a[5 + d] += w * q[d] as f64 / qn;
        }
    }
    // deterministic order; optional heaviest-first cap
    let mut order: Vec<usize> = (0..keys.len()).collect();
    if max_points > 0 && keys.len() > max_points {
        order.sort_by(|&a2, &b2| acc[b2][0].total_cmp(&acc[a2][0]));
        order.truncate(max_points);
    }
    order.sort_by_key(|&i| keys[i]);
    let mut out = Splats::default();
    for &i in &order {
        let a = &acc[i];
        let ws = a[0].max(1e-8);
        for d in 0..3 {
            out.means.push((a[2 + d] / ws) as f32);
        }
        let qn = (a[5..9].iter().map(|v| v * v).sum::<f64>()).sqrt().max(1e-8);
        for d in 0..4 {
            out.quats.push((a[5 + d] / qn) as f32);
        }
        for d in 0..3 {
            out.scales.push((a[9 + d] / ws) as f32);
        }
        out.opacities.push((a[1] / ws) as f32);
        for d in 0..3 {
            out.colors.push((a[12 + d] / ws) as f32);
        }
    }
    out
}

/// Drop the gaussians whose LARGEST axis is above the `q` quantile of that
/// statistic, matching the reference's `save_gs_ply` (`q = 0.98`).
///
/// A feed-forward head predicts a scale per pixel, and the tail of that
/// distribution is enormous translucent splats sitting over the whole scene.
/// There are few of them - two in a hundred - and they are enough to make a
/// sharp reconstruction look as though it was rendered through frosted glass,
/// because each one covers thousands of pixels at low opacity.
///
/// `q >= 1.0` keeps everything.
pub fn drop_largest_scales(s: &Splats, q: f32) -> Splats {
    let n = s.len();
    if n == 0 || q >= 1.0 {
        return s.clone();
    }
    let biggest = |i: usize| s.scales[i * 3..i * 3 + 3].iter().fold(0.0f32, |m, &v| m.max(v));
    let mut sorted: Vec<f32> = (0..n).map(biggest).collect();
    sorted.sort_by(f32::total_cmp);
    // `torch.quantile`'s linear interpolation, so the count matches upstream
    let pos = q.clamp(0.0, 1.0) as f64 * (n - 1) as f64;
    let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
    let hi = (lo + 1).min(n - 1);
    let thresh = sorted[lo] as f64 + (sorted[hi] - sorted[lo]) as f64 * frac;

    let mut out = Splats::default();
    for i in 0..n {
        if biggest(i) as f64 > thresh {
            continue;
        }
        out.means.extend_from_slice(&s.means[i * 3..i * 3 + 3]);
        out.quats.extend_from_slice(&s.quats[i * 4..i * 4 + 4]);
        out.scales.extend_from_slice(&s.scales[i * 3..i * 3 + 3]);
        out.opacities.push(s.opacities[i]);
        out.colors.extend_from_slice(&s.colors[i * 3..i * 3 + 3]);
    }
    out
}
