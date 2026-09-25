// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a fit's time goes, at the scale that matters: a TRAINED scene, its
//! cameras and its photographs, fitted for a few iterations of the
//! sparse-start objective with density control off - the steady state a long
//! run spends most of its hours in.
//!
//! ```text
//! fit_profile <scene.ply> <cameras.json> <photos dir> [iters]
//! ```
//!
//! Prints the per-kernel DEVICE times (`gpu_core::profile::profile_live`),
//! and with `BRAIN_SPLAT_PROFILE=1` the fit's own per-stage wall clock, which
//! is the only account that includes host work and transfers.

use gpu_core::profile::profile_live;
use splat::opt::{fit_full, Densify, FitCfg, TargetView};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!("usage: fit_profile <scene.ply> <cameras.json> <photos dir> [iters]");
        std::process::exit(2);
    }
    let scene = splat::ply::read(&a[0]).unwrap_or_else(|e| panic!("{}: {e}", a[0]));
    let cams = splat::types::cameras_from_json(&std::fs::read_to_string(&a[1]).expect("cameras.json")).expect("cameras");
    let iters: usize = a.get(3).map_or(12, |v| v.parse().expect("iters"));
    let mut photos: Vec<std::path::PathBuf> = std::fs::read_dir(&a[2])
        .expect("photos dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| ["jpg", "jpeg", "png", "ppm"].contains(&x.to_string_lossy().to_lowercase().as_str())))
        .collect();
    photos.sort();
    assert_eq!(photos.len(), cams.len(), "one photograph per camera, in sorted order");
    let targets: Vec<TargetView> = photos
        .iter()
        .zip(&cams)
        .map(|(p, c)| {
            let img = imaging::load(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let rgb = imaging::host::resize_bilinear_hwc(&img.to_hwc_unit(), 3, img.w, img.h, c.width, c.height);
            TargetView::new(*c, rgb)
        })
        .collect();

    let cfg = FitCfg {
        iters,
        densify_every: 0,
        strategy: Densify::Heuristic,
        geometry_after: 0.0,
        coarse: 0.0,
        log_every: 0,
        ..FitCfg::from_sparse_points(iters, scene.len())
    };
    let g = gpu_core::Gpu::new(splat::PIPELINES);
    println!(
        "{} gaussians x {} views at {}x{}, {iters} iterations of {} views each",
        scene.len(),
        targets.len(),
        cams[0].width,
        cams[0].height,
        cfg.batch
    );
    let prof = profile_live(&g, "fit", 1, || {
        fit_full(&g, splat::Kernels::at(0), &scene, &targets, &cfg, &mut |_, _| true);
    });
    println!("wall clock per iteration: {:.1} ms", 1e3 * prof.total_secs / iters as f64);
    prof.print_top(None, 20);
}
