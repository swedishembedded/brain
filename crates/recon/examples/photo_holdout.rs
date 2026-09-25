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
//!               [--split DESIGN[+DESIGN...]] [--path-steps n]
//!               [--sfm-cache dir] <fit options>
//! ```
//!
//! `--split` chooses the held-out views by their angle about the capture's
//! orbit (`recon::validate::Split`; azimuth and elevation in degrees, as the
//! run prints them per camera): `every:K` every k-th view in capture order,
//! `wedge:AZ:WIDTH` an azimuth wedge (a gap in the orbit), `band:LOW:HIGH` an
//! elevation band (extrapolation in elevation), `region:AZ:WIDTH:LOW:HIGH`
//! both (one camera of one ring, its ring above kept: interpolation); `+`
//! joins designs. Every held-out view is reported with its
//! angular deviation from the training views, its surface statistics and
//! its error binned by per-pixel observational support
//! (`recon::validate::Diagnosis`), and its diagnostic images are written
//! (`heldout_<i>_diag.png`: render | median range | range spread | entropy
//! | normal | support angle | free-space violations). The training views
//! are diagnosed against each other. A camera path through the held-out
//! views and their trained neighbours is rendered and diagnosed frame by
//! frame (`path/frame_<n>.png`, `path.csv`).
//!
//! `halvings` is how many exact halvings below the photographs' own
//! resolution the fit runs at (0 = native). `--sfm-cache` keeps structure
//! from motion's cameras and points in a directory and reads them from there
//! on the next run, so experiments on the fit do not re-solve the capture.
//!
//! Photographs are loaded as measurements (`imaging::load_photo`): each
//! target carries its EXIF exposure and encoding, and a linear, deep or
//! bracketed capture is fitted scene-linear when the camera model (`--isp`)
//! is on. Held-out views are scored raw (the capture's average camera at the
//! photograph's known exposure) and appearance-fitted (exposure and white
//! balance fitted on the left half of each, scored on the right half).
//!
//! Writes `heldout_<i>.png` (photograph | raw render) for every held-out
//! view, `train_<i>.png` for some training views, and `scene.ply`.
//!
//! Swedish Embedded AB implements photogrammetry and 3D reconstruction
//! pipelines and the evaluation that keeps them honest. If your team needs
//! that, you can procure our services by sending an email to
//! info@swedishembedded.com.

mod common;

use common::{fit_cfg, fitted_opts, montage, Flags, FIT_USAGE};
use gpu_core::Gpu;
use recon::eval::{score_held_out, score_training, stability, Viewer};
use recon::validate::{deviation, split, Diagnosis, Orbit, Split, SurfaceStats};
use recon::photogrammetry::{photo_target, Photometry};
use splat::isp::ColorSpace;
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
    let recorded: Vec<imaging::Photo> = paths.iter().map(|p| imaging::load_photo(p).unwrap_or_else(|e| panic!("{e}"))).collect();
    let photos: Vec<imaging::Rgb8> = recorded.iter().map(imaging::Photo::rgb8).collect();
    let (cams, source, init) = cameras(&photos, flags.get("sfm-cache"));
    let camera_model = flags.get("isp").is_some_and(|v| v != "off");
    let mut ph = Photometry::of(&recorded.iter().collect::<Vec<_>>());
    if !camera_model && ph.color_space == ColorSpace::SceneLinear {
        println!("photometry: the capture asks for a scene-linear fit, which needs the camera model; fitting display-referred");
        ph.color_space = ColorSpace::Display;
    }
    println!(
        "photometry: {:?}, exposures {} (EXIF, log2 stops relative to the capture)",
        ph.color_space,
        source.iter().map(|&i| format!("{:+.2}", ph.exposure(&recorded[i]))).collect::<Vec<_>>().join(" ")
    );
    let targets: Vec<TargetView> = cams
        .iter()
        .zip(&source)
        .map(|(c, &i)| {
            let mut t = photo_target(&recorded[i], *c, 0, &ph);
            for _ in 0..halvings {
                t = t.half_in(ph.color_space);
            }
            t
        })
        .collect();

    let design = Split::Every(every);
    let held_out: Vec<bool> = match flags.get("split") {
        None => split(&cams, design),
        Some(v) => v.split('+').map(|d| split(&cams, parse_split(d))).fold(vec![false; cams.len()], |a, b| a.iter().zip(&b).map(|(x, y)| *x || *y).collect()),
    };
    let orbit = Orbit::of(&cams);
    for (i, c) in cams.iter().enumerate() {
        let (az, el) = orbit.angles(c);
        println!("  camera {i:2}: azimuth {az:6.1}, elevation {el:5.1}{}", if held_out[i] { "  HELD OUT" } else { "" });
    }
    // `--views n` fits only the first n training views: a fit that cannot
    // reproduce even one photograph has a problem no amount of views explains
    let keep: usize = flags.parse("views").unwrap_or(usize::MAX);
    let train_ix: Vec<usize> = (0..targets.len()).filter(|&i| !held_out[i]).take(keep).collect();
    let mut train: Vec<TargetView> = train_ix.iter().map(|&i| targets[i].clone()).collect();
    let held: Vec<TargetView> = (0..targets.len()).filter(|&i| held_out[i]).map(|i| targets[i].clone()).collect();
    let (w, h) = (train[0].cam.width, train[0].cam.height);
    println!("fitting {} views at {w}x{h}, holding out {}", train.len(), held.len());

    let g = Gpu::new(&recon::photogrammetry::pipelines());
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

    let mut cfg = fit_cfg(&flags, iters, train.len(), dense);
    if let Some(i) = &mut cfg.isp {
        i.color_space = ph.color_space;
    }
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
    let isp = res.isp.as_ref();
    let (on_train, train_img) = score_training(&mut viewer, &refit, isp);
    let (on_held, held_img) = score_held_out(&mut viewer, &held, isp);
    let path = recon::eval::path(&refit.iter().map(|t| t.cam).collect::<Vec<_>>(), 8);
    let flicker = stability(&mut viewer, &path);
    let per = |s: &recon::eval::Scores| s.views.iter().map(|x| format!("{:.1}", x.psnr)).collect::<Vec<_>>().join(" ");
    println!(
        "{} gaussians in {secs:.0} s; training views {:.2} dB / SSIM {:.4}; HELD-OUT raw {:.2} dB / SSIM {:.4} ({}), \
         appearance-fitted {:.2} dB / SSIM {:.4} ({}); path flicker {flicker:.5}",
        res.scene.len(),
        on_train.mean_psnr(),
        on_train.mean_ssim(),
        on_held.raw.mean_psnr(),
        on_held.raw.mean_ssim(),
        per(&on_held.raw),
        on_held.fitted.mean_psnr(),
        on_held.fitted.mean_ssim(),
        per(&on_held.fitted),
    );
    println!("  per training view: {}", per(&on_train));
    for (i, (v, img)) in refit.iter().zip(&train_img).enumerate().step_by(4) {
        imaging::save(format!("{out}/train_{i}.png"), &montage(&[&v.rgb, img], v.cam.width, v.cam.height)).expect("png");
    }
    for (i, (v, img)) in held.iter().zip(&held_img).enumerate() {
        imaging::save(format!("{out}/heldout_{i}.png"), &montage(&[&v.rgb, img], v.cam.width, v.cam.height)).expect("png");
    }
    diagnose(&mut viewer, &out, &refit, &held, &cams, &held_out, &orbit, flags.parse("path-steps").unwrap_or(6));

    // viewers show stored colour: a scene-linear scene is exported encoded
    let scene = if ph.color_space == ColorSpace::SceneLinear { splat::isp::bake_display(&res.scene) } else { res.scene.clone() };
    splat::ply::write(&format!("{out}/scene.ply"), &scene).expect("ply");
    std::fs::write(format!("{out}/cameras.json"), splat::types::cameras_to_json(&res.cams)).expect("cameras");
}

fn parse_split(v: &str) -> Split {
    let parts: Vec<&str> = v.split(':').collect();
    let num = |i: usize| -> f64 { parts.get(i).and_then(|x| x.parse().ok()).unwrap_or_else(|| panic!("--split {v}: expected a number at field {i}")) };
    match parts[0] {
        "every" => Split::Every(num(1) as usize),
        "wedge" => Split::Wedge { centre: num(1), width: num(2) },
        "band" => Split::Band { low: num(1), high: num(2) },
        "region" => Split::Region { centre: num(1), width: num(2), low: num(3), high: num(4) },
        o => panic!("--split {o}: every:K, wedge:AZ:WIDTH, band:LOW:HIGH or region:AZ:WIDTH:LOW:HIGH"),
    }
}

fn stats_line(s: &SurfaceStats) -> String {
    format!(
        "covered {:.0}%, spread {:.4}, entropy {:.3}, share {:.3}, views {:.1}, unsupported {:.1}%, <5 {:.0}% <15 {:.0}% <30 {:.0}% <60 {:.0}% >=60 {:.0}%, free-space {:.2}%",
        100.0 * s.covered,
        s.spread,
        s.entropy,
        s.share,
        s.views,
        100.0 * s.support[0],
        100.0 * s.support[1],
        100.0 * s.support[2],
        100.0 * s.support[3],
        100.0 * s.support[4],
        100.0 * s.support[5],
        100.0 * s.free_space
    )
}

fn diag_montage(d: &Diagnosis) -> imaging::Rgb8 {
    let cols: Vec<Vec<f32>> = d.images().into_iter().map(|(_, im)| im.px.iter().map(|&v| v as f32 / 255.0).collect()).collect();
    let refs: Vec<&[f32]> = cols.iter().map(|c| c.as_slice()).collect();
    montage(&refs, d.width, d.height)
}

/// Diagnose the training views against each other, every held-out view
/// against the training views, and a camera path through the held-out views.
#[allow(clippy::too_many_arguments)]
fn diagnose(viewer: &mut Viewer, out: &str, train: &[TargetView], held: &[TargetView], all: &[Camera], held_out: &[bool], orbit: &Orbit, steps: usize) {
    let train_cams: Vec<Camera> = train.iter().map(|t| t.cam).collect();
    let observed = viewer.observe(&train_cams);
    let mut acc = SurfaceStats::default();
    for (k, t) in train.iter().enumerate() {
        let s = viewer.diagnose(&t.cam, Some(&observed), Some(k)).stats(t.mask.as_deref());
        acc.spread += s.spread / train.len() as f64;
        acc.entropy += s.entropy / train.len() as f64;
        acc.support[0] += s.support[0] / train.len() as f64;
        acc.free_space += s.free_space / train.len() as f64;
    }
    println!(
        "training views against each other: spread {:.4}, entropy {:.3}, unsupported {:.1}%, free-space {:.2}%",
        acc.spread,
        acc.entropy,
        100.0 * acc.support[0],
        100.0 * acc.free_space
    );
    for (i, v) in held.iter().enumerate() {
        let d = viewer.diagnose(&v.cam, Some(&observed), None);
        let dev = deviation(&v.cam, &train_cams, orbit.centre);
        println!("held-out view {i} (deviation {dev:.1} deg): {}", stats_line(&d.stats(v.mask.as_deref())));
        for b in d.error_by_support(&v.rgb, v.mask.as_deref()) {
            if b.pixels > 0 {
                let band = match b.upper {
                    u if u == -2.0 => "uncovered".to_string(),
                    u if u == -1.0 => "unsupported".to_string(),
                    u if u.is_infinite() => ">= 60 deg".to_string(),
                    u => format!("< {u} deg"),
                };
                println!("    {band:>12}: {:6} px, mean |error| {:.4}, {:.2} dB", b.pixels, b.mean_abs, b.psnr);
            }
        }
        imaging::save(format!("{out}/heldout_{i}_diag.png"), &diag_montage(&d)).expect("png");
    }
    // a path through the held-out views and the trained views beside them,
    // in azimuth order about the first held-out view
    let Some(first) = held_out.iter().position(|&h| h) else { return };
    let (az0, el0) = orbit.angles(&all[first]);
    let rel = |c: &Camera| {
        let (az, el) = orbit.angles(c);
        ((az - az0 + 540.0).rem_euclid(360.0) - 180.0, el)
    };
    let span = all.iter().zip(held_out).filter(|(_, &h)| h).map(|(c, _)| rel(c).0.abs()).fold(0.0f64, f64::max) + 60.0;
    let mut way: Vec<(f64, Camera)> = all.iter().map(|c| (rel(c), *c)).filter(|((a, e), _)| a.abs() <= span && (e - el0).abs() <= 12.0).map(|((a, _), c)| (a, c)).collect();
    way.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (w, h) = (train[0].cam.width, train[0].cam.height);
    let way: Vec<Camera> = way.into_iter().map(|(_, c)| c.resized(w, h)).collect();
    let path = recon::eval::path(&way, steps);
    std::fs::create_dir_all(format!("{out}/path")).expect("path dir");
    let mut csv = String::from("frame,azimuth,elevation,deviation,covered,spread,entropy,share,views,unsupported,s5,s15,s30,s60,s60plus,free_space\n");
    for (n, c) in path.iter().enumerate() {
        let d = viewer.diagnose(c, Some(&observed), None);
        let s = d.stats(None);
        let (az, el) = orbit.angles(c);
        csv.push_str(&format!(
            "{n},{az:.2},{el:.2},{:.2},{:.4},{:.5},{:.4},{:.4},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.5}\n",
            deviation(c, &train_cams, orbit.centre),
            s.covered,
            s.spread,
            s.entropy,
            s.share,
            s.views,
            s.support[0],
            s.support[1],
            s.support[2],
            s.support[3],
            s.support[4],
            s.support[5],
            s.free_space
        ));
        imaging::save(format!("{out}/path/frame_{n:03}.png"), &diag_montage(&d)).expect("png");
    }
    std::fs::write(format!("{out}/path.csv"), csv).expect("path.csv");
    println!("path: {} frames through the held-out views -> {out}/path/, {out}/path.csv", path.len());
}
