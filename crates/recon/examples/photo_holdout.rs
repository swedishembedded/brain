// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Held-out evaluation of photographs -> splat scene on a REAL capture: every
//! photograph is solved for by structure from motion, every `k`-th registered
//! one is kept out of the fit, and the result is rendered from those
//! photographs' own cameras and compared with them, inside the region the
//! undistortion covers. A capture's training views say how well the fit
//! memorized them; only views it never saw say whether the scene is right.
//!
//! ```text
//! photo_holdout <photos dir> <out dir> [iters] [width] [--every k] [--views n] <fit options>
//! ```
//!
//! Writes `heldout_<i>.png` (photograph | render) for every held-out view.
//!
//! Swedish Embedded AB implements photogrammetry and 3D reconstruction
//! pipelines and the evaluation that keeps them honest. If your team needs
//! that, you can procure our services by sending an email to
//! info@swedishembedded.com.

mod common;

use common::{fit_cfg, fitted_opts, montage, psnr_masked, render, Flags, FIT_USAGE};
use gpu_core::Gpu;
use splat::opt::fit_full;
use splat::Kernels;

fn main() {
    let flags = Flags::from_env();
    let Some(dir) = flags.positional.first().cloned().filter(|_| flags.get("help").is_none()) else {
        eprintln!("usage: photo_holdout <photos dir> <out dir> [iters] [width] [--every k] [--views n] {FIT_USAGE}");
        std::process::exit(2);
    };
    let out: String = flags.arg(1, "photo_holdout".to_string());
    let iters: usize = flags.arg(2, 3000);
    let width: u32 = flags.arg(3, 768);
    let every: usize = flags.parse("every").unwrap_or(8);
    std::fs::create_dir_all(&out).expect("out dir");

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()).is_some_and(|x| ["jpg", "jpeg", "png"].contains(&x.to_ascii_lowercase().as_str())))
        .collect();
    paths.sort();
    let photos: Vec<imaging::Rgb8> = paths.iter().map(|p| imaging::load(p).unwrap_or_else(|e| panic!("{e}"))).collect();
    let sfm_cfg = sfm::incremental::SfmCfg { verbose: true, ..Default::default() };
    let set = recon::photogrammetry::training_set(&photos, width, 0.5, &sfm_cfg).expect("structure from motion");
    println!(
        "structure from motion: {}/{} registered, {} points, rms {:.2} px, focal {:.1}",
        set.targets.len(),
        photos.len(),
        set.sfm.points.len(),
        set.sfm.rms_px,
        set.sfm.intrinsics.f
    );

    // hold out every k-th registered view, starting half a stride in so the
    // held-out views are not the capture's first and last
    let held_out = |i: usize| i % every == every / 2;
    let (train, held): (Vec<_>, Vec<_>) = set.targets.iter().cloned().enumerate().partition(|(i, _)| !held_out(*i));
    // `--views n` fits only the first n training views: a fit that cannot
    // reproduce even one photograph has a problem no amount of views explains
    let keep: usize = flags.parse("views").unwrap_or(usize::MAX);
    let train: Vec<_> = train.into_iter().map(|(_, t)| t).take(keep).collect();
    let held: Vec<_> = held.into_iter().map(|(_, t)| t).collect();
    println!("fitting {} views, holding out {}", train.len(), held.len());

    let cfg = fit_cfg(&flags, iters, train.len());
    let g = Gpu::new(splat::PIPELINES);
    let t = std::time::Instant::now();
    let res = fit_full(&g, Kernels::at(0), &set.init, &train, &cfg, &mut |_, _| true);
    let secs = t.elapsed().as_secs_f64();
    let o = fitted_opts(&cfg);
    let score = |views: &[splat::opt::TargetView]| -> Vec<(f64, Vec<f32>)> {
        views
            .iter()
            .map(|v| {
                let img = render(&g, &res.scene, &v.cam, &o);
                (psnr_masked(&img, &v.rgb, v.mask.as_deref()), img)
            })
            .collect()
    };
    let mean = |s: &[(f64, Vec<f32>)]| s.iter().map(|x| x.0).sum::<f64>() / s.len().max(1) as f64;
    // the fit may have refined the training cameras; held-out ones are the
    // structure-from-motion cameras either way
    let refit: Vec<_> = train.iter().zip(&res.cams).map(|(t, c)| splat::opt::TargetView { cam: *c, ..t.clone() }).collect();
    let on_train = score(&refit);
    let on_held = score(&held);
    println!(
        "{} gaussians in {secs:.0} s; training views {:.2} dB; HELD-OUT {:.2} dB ({})",
        res.scene.len(),
        mean(&on_train),
        mean(&on_held),
        on_held.iter().map(|x| format!("{:.1}", x.0)).collect::<Vec<_>>().join(" ")
    );
    println!("  per training view: {}", on_train.iter().map(|x| format!("{:.1}", x.0)).collect::<Vec<_>>().join(" "));
    for (i, (v, (_, img))) in refit.iter().zip(&on_train).enumerate().step_by(4) {
        imaging::save(format!("{out}/train_{i}.png"), &montage(&[&v.rgb, img], v.cam.width, v.cam.height)).expect("png");
    }
    for (i, (v, (_, img))) in held.iter().zip(&on_held).enumerate() {
        imaging::save(format!("{out}/heldout_{i}.png"), &montage(&[&v.rgb, img], v.cam.width, v.cam.height)).expect("png");
    }
    splat::ply::write(&format!("{out}/scene.ply"), &res.scene).expect("ply");
}
