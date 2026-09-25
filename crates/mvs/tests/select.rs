// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Source views and range bounds from sparse tracks: moderate triangulation
//! angles first, a view that shares nothing never, and bounds that contain
//! the scene.

use camera::Intrinsics;
use splat::types::Camera;

/// A camera at `x` on the X axis looking at `(target_x, 0, 5)`, or straight
/// away from the scene.
fn cam(x: f32, target_x: f32, away: bool) -> Camera {
    let f = if away { [0.0, 0.0, -1.0] } else {
        let d = [target_x - x, 0.0, 5.0f32];
        let l = (d[0] * d[0] + d[2] * d[2]).sqrt();
        [d[0] / l, 0.0, d[2] / l]
    };
    // right = f x up with up = -Y; down = f x right
    let r = [f[2], 0.0, -f[0]];
    let dn = [0.0, 1.0, 0.0];
    let c2w = [r[0], dn[0], f[0], x, r[1], dn[1], f[1], 0.0, r[2], dn[2], f[2], 0.0, 0.0, 0.0, 0.0, 1.0];
    Camera::with_intrinsics(c2w, &Intrinsics::pinhole(200.0, 320, 240))
}

/// Baseline on the X axis that sees the scene centre at `deg` from the
/// reference at the origin.
fn baseline(deg: f64) -> f32 {
    (5.0 * deg.to_radians().tan()) as f32
}

#[test]
fn sources_prefer_moderate_angles_and_bounds_contain_the_scene() {
    let cams = vec![
        cam(0.0, 0.0, false),
        cam(baseline(0.3), 0.0, false),
        cam(baseline(12.0), 0.0, false),
        cam(baseline(30.0), 0.0, false),
        cam(baseline(75.0), 0.0, false),
        cam(1.0, 0.0, true),
    ];
    // a grid on the plane z = 5 (with a little relief), seen by all but the
    // camera facing away
    let mut tracks = Vec::new();
    for i in 0..15 {
        for j in 0..15 {
            let (x, y) = (-1.5 + 0.2 * i as f64, -1.1 + 0.15 * j as f64);
            tracks.push(mvs::Track { xyz: [x, y, 5.0 + 0.1 * ((i + j) % 3) as f64], views: vec![0, 1, 2, 3, 4] });
        }
    }
    let cfg = mvs::SelectCfg { max_sources: 3, ..Default::default() };
    let with_obs = mvs::select::select_sources(&cams, &tracks, &cfg);
    assert_eq!(with_obs[0], vec![2, 3, 4], "reference 0: 12 and 30 degrees first, then 75, never 0.3 or the view facing away");
    assert!(with_obs[5].is_empty(), "the view facing away shares no track with anything");
    assert!(!with_obs.iter().any(|s| s.contains(&5)));

    // without observation lists, visibility is taken from projection alone
    let bare: Vec<mvs::Track> = tracks.iter().map(|t| mvs::Track { xyz: t.xyz, views: Vec::new() }).collect();
    let from_projection = mvs::select::select_sources(&cams, &bare, &cfg);
    assert_eq!(from_projection[0], with_obs[0]);
    assert!(from_projection[5].is_empty());

    let bounds = mvs::select::range_bounds(&cams, &tracks, &cfg);
    let (near, far) = bounds[0].expect("the reference sees the grid");
    let ranges: Vec<f64> = tracks.iter().map(|t| sfm::linalg::norm(t.xyz)).collect();
    let (lo, hi) = (ranges.iter().copied().fold(f64::INFINITY, f64::min), ranges.iter().copied().fold(0.0, f64::max));
    assert!((near as f64) < lo && (far as f64) > hi, "bounds [{near}, {far}] must contain the ranges [{lo}, {hi}]");
    assert!((near as f64) > 0.5 * lo && (far as f64) < 2.0 * hi, "bounds [{near}, {far}] are too loose for [{lo}, {hi}]");
    assert!(bounds[5].is_none(), "the view facing away sees no track");
}
