// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense stereo against exact geometry: ray-cast captures of textured
//! analytic shapes through a pinhole and through a Kannala-Brandt fisheye,
//! scored pixel by pixel against the true range and normal.

mod common;

use common::{fisheye_capture, pinhole_capture, run, score, truth, Scene};
use splat::types::Camera;

fn check(name: &str, scene: &Scene, cams: &[Camera]) {
    let st = run(scene, cams);
    println!("{name}:\n{}", st.timings);
    let mut worst = None::<(usize, common::Score)>;
    for (v, map) in st.depth.iter().enumerate() {
        let tru = truth(scene, &st.cams[v]);
        let s = score(map, &tru);
        println!("  view {v}: {s:?}");
        assert!(s.coverage > 0.6, "{name} view {v}: only {:.1} % of the surface measured", 100.0 * s.coverage);
        assert!(s.within_half_percent > 0.8, "{name} view {v}: {:.1} % within 0.5 % in range", 100.0 * s.within_half_percent);
        assert!(s.outliers < 0.02, "{name} view {v}: {:.2} % outliers", 100.0 * s.outliers);
        assert!(s.median_normal_deg < 3.0, "{name} view {v}: median normal error {:.2} deg", s.median_normal_deg);
        if worst.as_ref().is_none_or(|w| s.within_half_percent < w.1.within_half_percent) {
            worst = Some((v, s));
        }
    }
    println!("  worst view: {worst:?}");
}

/// Through a pinhole: most measured pixels within 0.5 % of the true range,
/// normals within a few degrees, almost no outliers after filtering.
#[test]
fn pinhole_ranges_and_normals_match_the_scene() {
    let (scene, cams) = pinhole_capture();
    check("pinhole", &scene, &cams);
}

/// The same bar through a Kannala-Brandt fisheye, matched on the original
/// distorted images: every warp goes through the real lens.
#[test]
fn fisheye_ranges_and_normals_match_the_scene() {
    let (scene, cams) = fisheye_capture();
    check("fisheye", &scene, &cams);
}
