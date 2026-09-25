// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Held-out evaluation of photographs -> splat scene on a REAL capture: every
//! photograph is solved for by structure from motion, every `k`-th registered
//! one is kept out of the fit, and the result is rendered from those
//! photographs' own cameras - through their own lens, at their own pixels -
//! and compared with them. A capture's training views say how well the fit
//! memorized them; only views it never saw say whether the scene is right.
//!
//! ```text
//! photo_holdout <photos dir> <out dir> [iters] [halvings] [--every k] [--views n]
//!               [--sfm-cache dir] <fit options>
//! ```
//!
//! `halvings` is how many exact halvings below the photographs' own
//! resolution the fit runs at (0 = native). `--sfm-cache` keeps structure
//! from motion's cameras and points in a directory and reads them from there
//! on the next run, so experiments on the fit do not re-solve the capture.
//!
//! Writes `heldout_<i>.png` (photograph | render) for every held-out view,
//! `train_<i>.png` for some training views, and `scene.ply`.
//!
//! Swedish Embedded AB implements photogrammetry and 3D reconstruction
//! pipelines and the evaluation that keeps them honest. If your team needs
//! that, you can procure our services by sending an email to
//! info@swedishembedded.com.

mod common;

use common::{fit_cfg, fitted_opts, montage, Flags, FIT_USAGE};
use gpu_core::Gpu;
use recon::eval::{holdout, score, stability, Viewer};
use splat::opt::{fit_full, TargetView};
use splat::types::{Camera, Splats};
use splat::Kernels;

/// Cameras (full resolution, with their lens), the source index of each and
/// the starting scene - from the cache when it has them, else from structure
/// from motion, which then fills it.
fn cameras(photos: &[imaging::Rgb8], cache: Option<&str>) -> (Vec<Camera>, Vec<usize>, Splats) {
    if let Some(dir) = cache {
        let (cj, pj, sj) = (format!("{dir}/cameras.json"), format!("{dir}/points.ply"), format!("{dir}/source.txt"));
        if let (Ok(c), Ok(s)) = (std::fs::read_to_string(&cj), std::fs::read_to_string(&sj)) {
            let cams = splat::types::cameras_from_json(&c).expect("cached cameras");
            let source: Vec<usize> = s.split_whitespace().map(|v| v.parse().expect("cached source index")).collect();
            let init = splat::ply::read(&pj).expect("cached points");
            println!("structure from motion: {} cameras and {} points from {dir}", cams.len(), init.len());
            return (cams, source, init);
        }
    }
    let t = std::time::Instant::now();
    let sfm_cfg = sfm::incremental::SfmCfg { verbose: true, ..Default::default() };
    let set = recon::photogrammetry::training_set(photos, 0, 0.1, &sfm_cfg).expect("structure from motion");
    println!(
        "structure from motion: {}/{} registered, {} points, rms {:.3} px, lens {:?} in {:.0} s",
        set.targets.len(),
        photos.len(),
        set.sfm.points.len(),
        set.sfm.rms_px,
        set.sfm.lens,
        t.elapsed().as_secs_f64()
    );
    let cams: Vec<Camera> = set.targets.iter().map(|t| t.cam).collect();
    if let Some(dir) = cache {
        std::fs::create_dir_all(dir).expect("cache dir");
        std::fs::write(format!("{dir}/cameras.json"), splat::types::cameras_to_json(&cams)).expect("cache cameras");
        std::fs::write(format!("{dir}/source.txt"), set.source.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")).expect("cache sources");
        splat::ply::write(&format!("{dir}/points.ply"), &set.init).expect("cache points");
    }
    (cams, set.source, set.init)
}

fn main() {
    let flags = Flags::from_env();
    let Some(dir) = flags.positional.first().cloned().filter(|_| flags.get("help").is_none()) else {
        eprintln!(
            "usage: photo_holdout <photos dir> <out dir> [iters] [halvings] [--every k] [--views n] [--sfm-cache dir] \
             [--dense on|off] [--stereo-halving n] {FIT_USAGE}"
        );
        std::process::exit(2);
    };
    let out: String = flags.arg(1, "photo_holdout".to_string());
    let iters: usize = flags.arg(2, 3000);
    let halvings: u32 = flags.arg(3, 2);
    let every: usize = flags.parse("every").unwrap_or(8);
    std::fs::create_dir_all(&out).expect("out dir");

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()).is_some_and(|x| ["jpg", "jpeg", "png"].contains(&x.to_ascii_lowercase().as_str())))
        .collect();
    paths.sort();
    let photos: Vec<imaging::Rgb8> = paths.iter().map(|p| imaging::load(p).unwrap_or_else(|e| panic!("{e}"))).collect();
    let (cams, source, init) = cameras(&photos, flags.get("sfm-cache"));
    let targets: Vec<TargetView> = cams
        .iter()
        .zip(&source)
        .map(|(c, &i)| {
            let mut t = recon::photogrammetry::target(&photos[i], *c, 0);
            for _ in 0..halvings {
                t = t.half();
            }
            t
        })
        .collect();

    let held_out = holdout(targets.len(), every);
    // `--views n` fits only the first n training views: a fit that cannot
    // reproduce even one photograph has a problem no amount of views explains
    let keep: usize = flags.parse("views").unwrap_or(usize::MAX);
    let train_ix: Vec<usize> = (0..targets.len()).filter(|&i| !held_out[i]).take(keep).collect();
    let mut train: Vec<TargetView> = train_ix.iter().map(|&i| targets[i].clone()).collect();
    let held: Vec<TargetView> = (0..targets.len()).filter(|&i| held_out[i]).map(|i| targets[i].clone()).collect();
    let (w, h) = (train[0].cam.width, train[0].cam.height);
    println!("fitting {} views at {w}x{h}, holding out {}", train.len(), held.len());

    let pipes: Vec<(&str, &str)> = splat::PIPELINES.iter().chain(mvs::PIPELINES).copied().collect();
    let g = Gpu::new(&pipes);
    let dense = match flags.get("dense") {
        None | Some("on") => true,
        Some("off") => false,
        Some(o) => panic!("--dense {o}: on or off"),
    };
    let init = if dense {
        // stereo on the training photographs only: a held-out view must not
        // shape the scene it is scored on
        let t = std::time::Instant::now();
        let full: Vec<Camera> = train_ix.iter().map(|&i| cams[i]).collect();
        let rgb: Vec<&imaging::Rgb8> = train_ix.iter().map(|&i| &photos[source[i]]).collect();
        let tracks: Vec<mvs::Track> =
            init.means.chunks_exact(3).map(|m| mvs::Track { xyz: [m[0] as f64, m[1] as f64, m[2] as f64], views: Vec::new() }).collect();
        let dcfg = recon::photogrammetry::DenseCfg {
            stereo: mvs::StereoCfg { halving: flags.parse("stereo-halving").unwrap_or(halvings), ..Default::default() },
            ..Default::default()
        };
        let mk = mvs::Kernels::at(splat::PIPELINES.len());
        let (dense_init, report) = recon::photogrammetry::dense(&g, &mk, &full, &rgb, &tracks, &mut train, &dcfg).expect("multi-view stereo");
        let cover = report.coverage.iter().sum::<f64>() / report.coverage.len() as f64;
        println!("multi-view stereo: {} points, mean coverage {:.1}% in {:.0} s", report.points, 100.0 * cover, t.elapsed().as_secs_f64());
        splat::ply::write(&format!("{out}/dense_init.ply"), &dense_init).expect("ply");
        dense_init
    } else {
        init
    };

    let cfg = fit_cfg(&flags, iters, train.len(), dense);
    let t = std::time::Instant::now();
    let res = fit_full(&g, Kernels::at(0), &init, &train, &cfg, &mut |_, _| true);
    let secs = t.elapsed().as_secs_f64();

    // The fit may have refined the cameras; its gauge holds the first one
    // and the scale, so the held-out cameras stay in its frame, and they
    // take their sensor's refined calibration.
    let refit: Vec<TargetView> = train.iter().zip(&res.cams).map(|(t, c)| TargetView { cam: *c, ..t.clone() }).collect();
    let k = res.cams[0].intrinsics();
    let held: Vec<TargetView> = held
        .iter()
        .map(|t| TargetView { cam: Camera { shutter: t.cam.shutter, ..Camera::with_intrinsics(t.cam.c2w, &k.resized(t.cam.width, t.cam.height)) }, ..t.clone() })
        .collect();
    let mut viewer = Viewer::new(&g, &res.scene, &res.filter3d, fitted_opts(&cfg), w, h).with_env(res.env.as_ref());
    let (on_train, train_img) = score(&mut viewer, &refit);
    let (on_held, held_img) = score(&mut viewer, &held);
    let path = recon::eval::path(&refit.iter().map(|t| t.cam).collect::<Vec<_>>(), 8);
    let flicker = stability(&mut viewer, &path);
    println!(
        "{} gaussians in {secs:.0} s; training views {:.2} dB / SSIM {:.4}; HELD-OUT {:.2} dB / SSIM {:.4} ({}); path flicker {flicker:.5}",
        res.scene.len(),
        on_train.mean_psnr(),
        on_train.mean_ssim(),
        on_held.mean_psnr(),
        on_held.mean_ssim(),
        on_held.views.iter().map(|x| format!("{:.1}", x.psnr)).collect::<Vec<_>>().join(" "),
    );
    println!("  per training view: {}", on_train.views.iter().map(|x| format!("{:.1}", x.psnr)).collect::<Vec<_>>().join(" "));
    for (i, (v, img)) in refit.iter().zip(&train_img).enumerate().step_by(4) {
        imaging::save(format!("{out}/train_{i}.png"), &montage(&[&v.rgb, img], v.cam.width, v.cam.height)).expect("png");
    }
    for (i, (v, img)) in held.iter().zip(&held_img).enumerate() {
        imaging::save(format!("{out}/heldout_{i}.png"), &montage(&[&v.rgb, img], v.cam.width, v.cam.height)).expect("png");
    }
    splat::ply::write(&format!("{out}/scene.ply"), &res.scene).expect("ply");
    std::fs::write(format!("{out}/cameras.json"), splat::types::cameras_to_json(&res.cams)).expect("cameras");
}
