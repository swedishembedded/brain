// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A starting scene from a sparse, coloured point cloud - the initialization
//! 3D Gaussian Splatting defines (Kerbl et al. 2023, §5): one isotropic
//! gaussian per point, sized by the mean distance to its three nearest
//! neighbours, at low opacity so the optimizer decides what is solid.
//!
//! Neighbours are found through a uniform hash grid sized from the cloud's
//! own spacing, so the cost is linear in the number of points rather than
//! quadratic.

use crate::types::{Camera, Splats};
use std::collections::HashMap;

/// Starting scene from `xyz` `[N*3]` and `rgb` `[N*3]` (0..1).
pub fn from_points(xyz: &[f32], rgb: &[f32], opacity: f32) -> Splats {
    let n = xyz.len() / 3;
    assert_eq!(rgb.len(), n * 3);
    let scales = knn_spacing(xyz, 3);
    let mut s = Splats::default();
    for i in 0..n {
        s.means.extend_from_slice(&xyz[i * 3..i * 3 + 3]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let r = scales[i];
        s.scales.extend_from_slice(&[r, r, r]);
        s.opacities.push(opacity);
        s.colors.extend_from_slice(&rgb[i * 3..i * 3 + 3]);
    }
    s
}

/// Raise every gaussian's axes to at least `px` pixels of the camera that
/// sees it most finely.
///
/// Nearest-neighbour spacing is a WORLD distance, and where structure from
/// motion found dense texture or distant surface it is sub-pixel. Under the
/// Mip filter such a gaussian's opacity is compensated almost to nothing, so
/// it renders nothing, receives no gradient, and density control correctly
/// reads it as dead: measured on a 16-photo capture, 3,605 of 8,792 starting
/// gaussians contributed nothing to any view and the median contributed
/// 0.001 px - so only the visible remainder could ever be refined and the
/// scene grew far below its schedule.
pub fn floor_to_pixels(s: &mut Splats, cams: &[Camera], px: f32) {
    let floor = crate::mip::smoothing_sigma(s, cams, px);
    for (i, f) in floor.iter().enumerate() {
        for v in &mut s.scales[i * 3..i * 3 + 3] {
            *v = v.max(*f);
        }
    }
}

/// Mean distance from every point to its `k` nearest neighbours.
pub fn knn_spacing(xyz: &[f32], k: usize) -> Vec<f32> {
    let n = xyz.len() / 3;
    if n < 2 {
        return vec![1e-2; n];
    }
    let p = |i: usize| [xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]];
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for i in 0..n {
        let v = p(i);
        for a in 0..3 {
            lo[a] = lo[a].min(v[a]);
            hi[a] = hi[a].max(v[a]);
        }
    }
    // a cell holding about k points on average, for a cloud spread over a
    // 2D surface (which is what a reconstruction's points sample)
    let area = {
        let e: Vec<f32> = (0..3).map(|a| (hi[a] - lo[a]).max(1e-9)).collect();
        (e[0] * e[1] + e[1] * e[2] + e[0] * e[2]).max(1e-18)
    };
    let cell = (area * (k + 1) as f32 / n as f32).sqrt().max(1e-9);
    let key = |v: [f32; 3]| -> (i64, i64, i64) {
        (((v[0] - lo[0]) / cell) as i64, ((v[1] - lo[1]) / cell) as i64, ((v[2] - lo[2]) / cell) as i64)
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for i in 0..n {
        grid.entry(key(p(i))).or_default().push(i);
    }
    backend_cpu::par::map_f32(n, |i| {
        let v = p(i);
        let c = key(v);
        let mut best: Vec<f32> = Vec::with_capacity(k + 1);
        let mut ring = 1i64;
        loop {
            best.clear();
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    for dz in -ring..=ring {
                        if let Some(list) = grid.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) {
                            for &j in list {
                                if j != i {
                                    let q = p(j);
                                    best.push(((v[0] - q[0]).powi(2) + (v[1] - q[1]).powi(2) + (v[2] - q[2]).powi(2)).sqrt());
                                }
                            }
                        }
                    }
                }
            }
            best.sort_by(f32::total_cmp);
            // the k-th neighbour must lie inside the searched ring to be exact
            if best.len() >= k && best[k - 1] <= ring as f32 * cell {
                break;
            }
            if ring > 64 || best.len() >= n - 1 {
                break;
            }
            ring *= 2;
        }
        let m = best.len().min(k).max(1);
        (best.iter().take(m).sum::<f32>() / m as f32).max(1e-7)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a regular grid every point's three nearest neighbours are at the
    /// grid spacing - except at the four corners, which have only two.
    #[test]
    fn spacing_of_a_grid_is_its_pitch() {
        let mut xyz = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                xyz.extend_from_slice(&[i as f32 * 0.1, j as f32 * 0.1, 2.0]);
            }
        }
        for (i, s) in knn_spacing(&xyz, 3).into_iter().enumerate() {
            let (a, b) = (i / 20, i % 20);
            let corner = (a == 0 || a == 19) && (b == 0 || b == 19);
            let want = if corner { (0.2 + 0.1 * 2f32.sqrt()) / 3.0 } else { 0.1 };
            assert!((s - want).abs() < 1e-5, "point {i}: spacing {s} against {want}");
        }
    }
}
