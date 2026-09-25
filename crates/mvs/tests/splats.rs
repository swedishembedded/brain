// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fused cloud as starting gaussians: each lies in the surface - its
//! shortest axis IS the normal, in the renderer's own axis convention - and
//! is as wide as its pixel footprint.

use data::rng::Lcg;
use splat::geometry::axis;

#[test]
fn every_gaussian_is_thin_along_its_normal_and_as_wide_as_its_footprint() {
    let mut g = Lcg::new(5);
    let n = 500;
    let normal: Vec<f32> = (0..n)
        .flat_map(|_| {
            let v = [g.signed(), g.signed(), g.signed()];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-3);
            v.map(|c| c / l)
        })
        .collect();
    let fused = mvs::Fused {
        xyz: g.vec(3 * n),
        normal: normal.clone(),
        rgb: g.vec_unit(3 * n),
        conf: g.vec_unit(n),
        radius: (0..n).map(|_| 0.001 + 0.01 * g.unit()).collect(),
        support: vec![3; n],
    };
    let cfg = mvs::SplatInit { footprint_scale: 1.25, thickness: 0.1, opacity: 0.2 };
    let s = mvs::to_splats(&fused, &cfg);
    assert_eq!(s.len(), n);
    assert_eq!(s.means, fused.xyz);
    assert_eq!(s.colors, fused.rgb);
    assert!(s.opacities.iter().all(|o| *o == 0.2));
    for i in 0..n {
        let q: [f32; 4] = std::array::from_fn(|k| s.quats[4 * i + k]);
        let sc = &s.scales[3 * i..3 * i + 3];
        let shortest = (0..3).min_by(|a, b| sc[*a].total_cmp(&sc[*b])).unwrap();
        let a = axis(q, shortest);
        let nrm = &normal[3 * i..3 * i + 3];
        let cos = a[0] * nrm[0] + a[1] * nrm[1] + a[2] * nrm[2];
        assert!(cos.abs() > 0.9999, "gaussian {i}: shortest axis {a:?} is not the normal {nrm:?}");
        let tangent = 1.25 * fused.radius[i];
        let mut sorted = sc.to_vec();
        sorted.sort_by(f32::total_cmp);
        assert!((sorted[1] - tangent).abs() < 1e-6 && (sorted[2] - tangent).abs() < 1e-6, "gaussian {i}: tangent scales {sc:?}");
        assert!((sorted[0] - 0.1 * tangent).abs() < 1e-7, "gaussian {i}: normal scale {}", sorted[0]);
    }
}
