// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fusion against exact geometry: every view's filtered map merged into one
//! cloud whose points lie on the true surface, carry its normal and colour,
//! and appear once however many views saw them.

mod common;

use common::{pinhole_capture, run, surface_distance, surface_normal};
use sfm::linalg::{dot, norm, sub};

#[test]
fn fused_points_lie_on_the_surface_once_each() {
    let (scene, cams) = pinhole_capture();
    let st = run(&scene, &cams);
    let gpu = gpu_core::testgpu::dev(mvs::PIPELINES);
    let cfg = mvs::FuseCfg::default();
    let f = mvs::fuse(&gpu, &mvs::Kernels::at(0), &st.cams, &st.depth, &st.rgb, &cfg).expect("fuse");
    let n = f.len();
    let measured: Vec<usize> = st.depth.iter().map(|d| d.range.iter().filter(|r| **r > 0.0).count()).collect();
    let most = *measured.iter().max().unwrap();
    let total: usize = measured.iter().sum();
    println!("{n} fused points from {total} measurements ({most} in the fullest view)");
    // once each: six views of one surface fuse to about one view's worth of
    // points, not six
    assert!(n > most / 2 && n < most * 3 / 2, "{n} points for a surface one view measures with {most}");

    let eyes: Vec<[f64; 3]> = cams.iter().map(|c| [c.c2w[3] as f64, c.c2w[7] as f64, c.c2w[11] as f64]).collect();
    let (mut on_surface, mut normal_ok, mut colour_err, mut radius_ok) = (0usize, 0usize, 0.0f64, 0usize);
    for i in 0..n {
        let x = [f.xyz[3 * i] as f64, f.xyz[3 * i + 1] as f64, f.xyz[3 * i + 2] as f64];
        let range = eyes.iter().map(|e| norm(sub(x, *e))).fold(f64::INFINITY, f64::min);
        on_surface += (surface_distance(&scene.shapes, x) < 0.005 * range) as usize;
        let nt = surface_normal(&scene.shapes, x);
        let nf = [f.normal[3 * i] as f64, f.normal[3 * i + 1] as f64, f.normal[3 * i + 2] as f64];
        normal_ok += (dot(nt, nf).abs() > 5f64.to_radians().cos()) as usize;
        let c = scene.tex.rgb(x);
        colour_err += (0..3).map(|k| (c[k] - f.rgb[3 * i + k] as f64).abs()).sum::<f64>() / 3.0;
        // one pixel of a 300 px focal length at the point's range
        let footprint = range / 300.0;
        let radius = f.radius[i] as f64;
        radius_ok += (radius > 0.7 * footprint && radius < 1.5 * footprint) as usize;
        assert!(f.support[i] >= cfg.min_support, "point {i} has support {}", f.support[i]);
        assert!((0.0..=1.0).contains(&f.conf[i]), "point {i} has confidence {}", f.conf[i]);
    }
    let frac = |k: usize| k as f64 / n as f64;
    println!(
        "on surface {:.4}, normal within 5 deg {:.4}, mean colour error {:.4}, radius = footprint {:.4}",
        frac(on_surface),
        frac(normal_ok),
        colour_err / n as f64,
        frac(radius_ok)
    );
    assert!(frac(on_surface) > 0.97, "only {:.2} % of the points are on the surface", 100.0 * frac(on_surface));
    assert!(frac(normal_ok) > 0.85, "only {:.2} % of the normals are within 5 degrees", 100.0 * frac(normal_ok));
    let colour_err = colour_err / n as f64;
    assert!(colour_err < 0.05, "mean colour error {colour_err:.3}");
    assert!(frac(radius_ok) > 0.95, "radius is not the pixel footprint for {:.2} % of the points", 100.0 * (1.0 - frac(radius_ok)));
}
