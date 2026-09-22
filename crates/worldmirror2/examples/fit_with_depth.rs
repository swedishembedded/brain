// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fit a dumped forward pass with its own fused depth as the prior.
//!
//! The end-to-end path the pieces were built for: assemble a scene from a
//! dumped pass, reconcile the per-view depths against each other, and hand the
//! result to the fit as an anchor along the one axis an RGB loss cannot see.
//!
//! Usage: fit_with_depth <heads-dir> <images-dir> <out.ply> [iters] [lr] [depth_weight] [holdout,csv] [voxel]

use splat::opt::{fit, FitCfg, TargetView};
use splat::types::Camera;
use worldmirror2::gaussians::{assemble_from, fuse_depths, AssembleOpts, HeadOutputs};

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
    let (dir, imgs, out) = (&a[0], &a[1], &a[2]);
    let iters: usize = a.get(3).map_or(300, |v| v.parse().unwrap());
    let lr: f32 = a.get(4).map_or(2e-4, |v| v.parse().unwrap());
    let dw: f32 = a.get(5).map_or(0.0, |v| v.parse().unwrap());
    let hold: Vec<usize> = a.get(6).map_or(vec![], |v| {
        v.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect()
    });

    let heads = HeadOutputs::load(dir).expect("heads");
    let cams = read_cameras(&format!("{dir}/cameras.json"));
    let (w, h) = (heads.width, heads.height);
    let hw = (w * h) as usize;
    // Every threshold stays at the reference's own value. `edge_depth_rtol`
    // especially: a pixel straddling a silhouette gets a depth blended from
    // the two surfaces either side and unprojects to neither, so disabling it
    // hangs a combed fringe off every edge in the scene - which a fit cannot
    // remove, because the fringe is where the depth prior says the surface is.
    let opts = AssembleOpts::default();
    let (scene, cams, weights) = assemble_from(&heads, &cams, &opts);
    eprintln!("assembled {} gaussians (surface-align {})", scene.len(), opts.surface_align);
    let voxel: f32 = a.get(7).map_or(0.0011, |v| v.parse().unwrap());
    let scene = splat::prune::voxel_merge(&scene, &weights, voxel, 0);
    eprintln!("voxel fused -> {}", scene.len());

    let mut depth: Vec<Vec<f32>> = heads.gsd.iter().map(|g| g[..hw].iter().map(|v| v.exp()).collect()).collect();
    let conf: Vec<Vec<f32>> = heads.gsd.iter().map(|g| g[hw..2 * hw].iter().map(|v| 1.0 + v.exp()).collect()).collect();
    let support = fuse_depths(&mut depth, &conf, &cams, w, h, opts.fuse_depth_rtol);

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(imgs).unwrap()
        .filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ppm" || x == "png")).collect();
    paths.sort();

    let mut targets = Vec::new();
    for (i, cam) in cams.iter().enumerate() {
        if hold.contains(&i) {
            continue;
        }
        let img = imaging::load(&paths[i]).unwrap();
        let rgb: Vec<f32> = img.px.iter().map(|&v| v as f32 / 255.0).collect();
        let t = TargetView::new(*cam, rgb);
        targets.push(if dw > 0.0 {
            t.with_depth(depth[i].clone(), Some(support[i].iter().map(|&c| c as f32).collect()))
        } else {
            t
        });
    }
    eprintln!("fitting {} gaussians against {} views (depth_weight {dw})", scene.len(), targets.len());

    let g = gpu_core::Gpu::new(splat::PIPELINES);
    // The geometry here came from the model's own metric depth, so it is the
    // case the budgets exist for: a gaussian may settle onto its surface but
    // not leave it, and `lr` is free to be an appearance rate again.
    let cfg = FitCfg {
        iters,
        lr,
        log_every: 20,
        depth_weight: dw,
        position_budget: 1.0,
        scale_budget: 0.5,
        rotation_budget: 0.5,
        ..Default::default()
    };
    let (fitted, mse) = fit(&g, splat::Kernels::at(0), &scene, &targets, &cfg, &mut |_, _| true);
    splat::ply::write(out, &fitted).unwrap();
    println!("{out}: {} gaussians, final mse {mse:.6}", fitted.len());
}
