// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fused cloud as a splat fit's starting scene: one thin gaussian per
//! point, lying in the surface.
//!
//! A sparse start gives the fit isotropic blobs and no geometry, and it
//! answers held-out views with floaters. A dense, oriented start is the
//! surface already: each gaussian's two tangent axes span its pixel footprint
//! ([`SplatInit::footprint_scale`] times the fused radius, so neighbours
//! overlap into a closed sheet) and its third axis, the normal, is a small
//! fraction of that ([`SplatInit::thickness`]) - the flattened, surface-aligned
//! shape 2D/planar gaussian methods converge to (Huang et al., "2D Gaussian
//! Splatting for Geometrically Accurate Radiance Fields", SIGGRAPH 2024).
//!
//! Axis convention: `scales[k]` belongs to column `k` of the rotation of the
//! quaternion ([`splat::geometry::axis`], `splat_project.wgsl`), so the
//! rotation is built with columns (tangent, bitangent, normal) and the normal
//! gets the smallest scale.
//!
//! This lives here rather than in `splat` because it is the one consumer of
//! [`Fused`] and `splat` must not depend on the stereo that feeds it; only the
//! [`Splats`] type and the quaternion convention are taken from there.

use sfm::linalg::{cross, normalize};
use splat::types::Splats;

use crate::fuse::Fused;

/// How fused points become gaussians.
#[derive(Clone, Debug, PartialEq)]
pub struct SplatInit {
    /// Tangent standard deviation as a multiple of the point's radius.
    pub footprint_scale: f32,
    /// Normal standard deviation as a fraction of the tangent one.
    pub thickness: f32,
    /// Starting opacity of every gaussian.
    pub opacity: f32,
}

impl Default for SplatInit {
    fn default() -> Self {
        SplatInit { footprint_scale: 1.2, thickness: 0.15, opacity: 0.3 }
    }
}

/// One surface-aligned gaussian per fused point.
pub fn to_splats(f: &Fused, cfg: &SplatInit) -> Splats {
    let n = f.len();
    let mut s = Splats {
        means: f.xyz.clone(),
        quats: Vec::with_capacity(4 * n),
        scales: Vec::with_capacity(3 * n),
        opacities: vec![cfg.opacity; n],
        colors: f.rgb.clone(),
        sh_rest: None,
    };
    for i in 0..n {
        let nrm = normalize([f.normal[3 * i] as f64, f.normal[3 * i + 1] as f64, f.normal[3 * i + 2] as f64]);
        let helper = if nrm[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let t1 = normalize(cross(helper, nrm));
        let t2 = cross(nrm, t1);
        // row-major, columns (t1, t2, n): right-handed since t1 x t2 = n
        let r = [t1[0], t2[0], nrm[0], t1[1], t2[1], nrm[1], t1[2], t2[2], nrm[2]];
        let q = normalize4(splat::orient::quat_of(&r));
        s.quats.extend(q.map(|v| v as f32));
        let tangent = f.radius[i] * cfg.footprint_scale;
        s.scales.extend([tangent, tangent, tangent * cfg.thickness]);
    }
    s
}

fn normalize4(q: [f64; 4]) -> [f64; 4] {
    let l = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / l)
}
