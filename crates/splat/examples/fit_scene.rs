// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fit an existing scene against the photographs it came from.
//!
//! The geometry budgets are set here rather than left at their defaults: a
//! scene from a feed-forward model is already metric, so a gaussian may settle
//! onto its surface but not leave it. Fitting such a scene WITHOUT them moved
//! 94% of gaussians more than three radii off the surface the model had
//! placed them on, which reads as stray splats and lost sharpness however good
//! the initial geometry was.
//!
//! Usage: fit_scene <in.ply> <cameras.json> <images-dir> <out.ply>
//!                  [iters] [lr] [holdout,csv]

use splat::opt::{fit, FitCfg, TargetView};
use splat::types::Camera;

fn read_cameras(path: &str) -> Vec<Camera> {
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    j.as_array().unwrap().iter().map(|c| Camera {
        c2w: c["c2w"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<f32>>().try_into().unwrap(),
        fx: c["fx"].as_f64().unwrap() as f32, fy: c["fy"].as_f64().unwrap() as f32,
        cx: c["cx"].as_f64().unwrap() as f32, cy: c["cy"].as_f64().unwrap() as f32,
        width: c["width"].as_u64().unwrap() as u32, height: c["height"].as_u64().unwrap() as u32,
    }).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("usage: fit_scene <in.ply> <cameras.json> <images-dir> <out.ply> [iters] [lr] [holdout,csv]");
        std::process::exit(2);
    }
    let iters: usize = a.get(4).map_or(200, |v| v.parse().unwrap());
    let lr: f32 = a.get(5).map_or(5e-3, |v| v.parse().unwrap());
    let hold: Vec<usize> = a.get(6).map_or(vec![], |v| {
        v.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect()
    });

    let scene = splat::ply::read(&a[0]).unwrap();
    let cams = read_cameras(&a[1]);
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&a[2]).unwrap()
        .filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ppm" || x == "png")).collect();
    paths.sort();
    assert_eq!(paths.len(), cams.len(), "one image per camera");

    let mut targets = Vec::new();
    for (i, cam) in cams.iter().enumerate() {
        if hold.contains(&i) {
            continue;
        }
        let img = imaging::load(&paths[i]).unwrap();
        let rgb: Vec<f32> = img.px.iter().map(|&v| v as f32 / 255.0).collect();
        targets.push(TargetView::new(*cam, rgb));
    }
    eprintln!("fitting {} gaussians against {} of {} views", scene.len(), targets.len(), cams.len());

    let g = gpu_core::Gpu::new(splat::PIPELINES);
    let cfg = FitCfg {
        iters,
        lr,
        log_every: 20,
        position_budget: 1.0,
        scale_budget: 0.5,
        rotation_budget: 0.5,
        ..Default::default()
    };
    let (fitted, mse) = fit(&g, splat::Kernels::at(0), &scene, &targets, &cfg, &mut |_, _| true);
    splat::ply::write(&a[3], &fitted).unwrap();
    println!("{}: {} gaussians, final mse {mse:.6}", a[3], fitted.len());
}
