// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scenes the recon tests photograph, built the way reconstruction builds
//! them: points with normals made into surface-aligned gaussians by
//! `mvs::to_splats`.

use data::rng::Lcg;
use splat::types::Splats;

/// A floor at y = 1 and a ball of radius 0.55 at (0, 0.45, 4.2) resting on
/// it, tiled with small opaque gaussians of random colour: texture
/// everywhere for stereo to match and for a view to be judged by.
pub fn floor_and_ball() -> Splats {
    let mut r = Lcg::new(0xde75e);
    let mut f = mvs::Fused::default();
    let mut add = |p: [f32; 3], n: [f32; 3], r: &mut Lcg| {
        f.xyz.extend_from_slice(&p);
        f.normal.extend_from_slice(&n);
        f.rgb.extend_from_slice(&[r.unit(), r.unit(), r.unit()]);
        f.conf.push(1.0);
        f.radius.push(0.03);
        f.support.push(3);
    };
    // the floor faces up: -y
    for iz in 0..90 {
        for ix in 0..90 {
            add([-2.2 + ix as f32 * 0.05, 1.0, 2.0 + iz as f32 * 0.05], [0.0, -1.0, 0.0], &mut r);
        }
    }
    let (c, rad) = ([0.0f32, 0.45, 4.2], 0.55f32);
    for k in 0..5000 {
        let z = 1.0 - 2.0 * (k as f32 + 0.5) / 5000.0;
        let phi = k as f32 * 2.399_963;
        let n = [(1.0 - z * z).sqrt() * phi.cos(), (1.0 - z * z).sqrt() * phi.sin(), z];
        add([c[0] + rad * n[0], c[1] + rad * n[1], c[2] + rad * n[2]], n, &mut r);
    }
    mvs::to_splats(&f, &mvs::SplatInit { footprint_scale: 1.0, thickness: 0.1, opacity: 0.98 })
}

/// The ball's centre and radius.
pub const BALL: ([f32; 3], f32) = ([0.0, 0.45, 4.2], 0.55);
