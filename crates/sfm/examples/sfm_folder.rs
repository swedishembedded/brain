// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structure from motion on a folder of photographs, reporting what the
//! calibration came out as: registered views, points, every candidate lens
//! model's reprojection RMS and information criterion, the chosen model and
//! each sensor's intrinsics (also as `cameras.json` fields).
//!
//! ```text
//! cargo run --release --offline -p brain-sfm --example sfm_folder -- <dir> \
//!     [--lens auto|pinhole|radial|brown|fisheye] [--focal <px>]
//! ```
//!
//! Photographs of different sizes are taken to be different sensors; one
//! `--focal` (pixels, e.g. from EXIF) applies to every photograph.
//!
//! Swedish Embedded AB implements camera calibration and multi-view
//! reconstruction for its clients. If your team needs photographs turned into
//! calibrated cameras and geometry, you can procure our services by sending
//! an email to info@swedishembedded.com.

use sfm::incremental::{reconstruct, Photo, SfmCfg};
use sfm::lens::{describe, LensChoice};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir = None;
    let mut lens = LensChoice::Auto;
    let mut focal = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--lens" => {
                lens = match args.next().as_deref() {
                    Some("auto") => LensChoice::Auto,
                    Some("pinhole") => LensChoice::Pinhole,
                    Some("radial") => LensChoice::Radial,
                    Some("brown") => LensChoice::Brown,
                    Some("fisheye") => LensChoice::Fisheye,
                    other => usage(&format!("unknown lens model {other:?}")),
                }
            }
            "--focal" => focal = Some(args.next().and_then(|v| v.parse::<f64>().ok()).unwrap_or_else(|| usage("--focal takes a number of pixels"))),
            _ if dir.is_none() => dir = Some(a),
            other => usage(&format!("unexpected argument {other}")),
        }
    }
    let dir = dir.unwrap_or_else(|| usage("a folder of photographs is required"));
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| usage(&format!("{dir}: {e}")))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()).is_some_and(|x| ["jpg", "jpeg", "png"].contains(&x.to_ascii_lowercase().as_str())))
        .collect();
    paths.sort();
    let images: Vec<imaging::Rgb8> = paths.iter().map(|p| imaging::load(p).unwrap_or_else(|e| usage(&e))).collect();
    // one sensor per distinct image size, numbered in order of appearance
    let mut sizes: Vec<(u32, u32)> = Vec::new();
    let photos: Vec<Photo> = images
        .iter()
        .map(|im| {
            let sensor = sizes.iter().position(|s| *s == (im.w, im.h)).unwrap_or_else(|| {
                sizes.push((im.w, im.h));
                sizes.len() - 1
            });
            Photo { width: im.w, height: im.h, rgb: &im.px, sensor, focal_px: focal }
        })
        .collect();
    println!("{} photographs from {dir}, {} sensor(s)", photos.len(), sizes.len());
    let cfg = SfmCfg { lens, verbose: true, ..SfmCfg::default() };
    let t0 = std::time::Instant::now();
    let rec = reconstruct(&photos, &cfg).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let secs = t0.elapsed().as_secs_f64();

    println!();
    println!("registered {}/{} photographs, {} points, rms {:.4} px, {secs:.1} s", rec.poses.iter().flatten().count(), photos.len(), rec.points.len(), rec.rms_px);
    for (i, p) in rec.poses.iter().enumerate() {
        if p.is_none() {
            println!("  not registered: {}", paths[i].display());
        }
    }
    println!("lens models (fitted to the incremental reconstruction's observations):");
    for f in &rec.lens_fits {
        let mark = if f.choice == rec.lens { "  <- chosen" } else { "" };
        println!("  {:<8} rms {:.4} px  {:2} parameters  BIC {:.1}{mark}", format!("{:?}", f.choice), f.rms_px, f.params, f.bic);
    }
    for (s, k) in rec.intrinsics.iter().enumerate() {
        println!("sensor {s} ({}x{}): {}", k.width, k.height, describe(k));
        println!("  cameras.json: {}", k.to_json());
    }
    radial_profile(&rec);
}

/// RMS reprojection error by distance from the image centre, in five rings
/// out to the corner - a lens model that fits leaves it flat, one that does
/// not leaves it climbing toward the edge - and by keypoint scale, which is
/// what a detector's localization error grows with.
fn radial_profile(rec: &sfm::incremental::Reconstruction) {
    const RINGS: usize = 5;
    const OCTAVES: usize = 5;
    let mut sum = [[0.0f64; RINGS]; 2];
    let mut count = [0usize; RINGS];
    let mut by_scale = [0.0f64; OCTAVES];
    let mut scale_count = [0usize; OCTAVES];
    for p in &rec.points {
        for &(i, kp) in &p.obs {
            let (Some(pose), k) = (rec.poses[i], &rec.intrinsics[rec.sensor[i]]) else { continue };
            let Some(uv) = sfm::camera::project(k, &pose, p.xyz) else { continue };
            let f = rec.keypoints[i][kp];
            let (dx, dy) = (f.x as f64 - k.width as f64 / 2.0, f.y as f64 - k.height as f64 / 2.0);
            let corner = (k.width as f64).hypot(k.height as f64) / 2.0;
            let ring = ((dx.hypot(dy) / corner * RINGS as f64) as usize).min(RINGS - 1);
            let e2 = [(uv[0] - f.x as f64).powi(2), (uv[1] - f.y as f64).powi(2)];
            sum[0][ring] += e2[0];
            sum[1][ring] += e2[1];
            count[ring] += 1;
            let octave = ((f.sigma as f64 / 1.6).log2().floor().max(0.0) as usize).min(OCTAVES - 1);
            by_scale[octave] += e2[0] + e2[1];
            scale_count[octave] += 1;
        }
    }
    println!("reprojection rms by keypoint scale (sigma, pixels):");
    for o in 0..OCTAVES {
        let lo = 1.6 * (1u32 << o) as f64;
        let hi = if o + 1 == OCTAVES { "   up".to_string() } else { format!("{:5.1}", 2.0 * lo) };
        println!("  {lo:5.1}-{hi}: {:7} observations, rms {:.4} px", scale_count[o], (by_scale[o] / scale_count[o].max(1) as f64).sqrt());
    }
    println!("reprojection rms by distance from the centre (fraction of the half-diagonal):");
    for r in 0..RINGS {
        let n = count[r].max(1) as f64;
        println!(
            "  {:.1}-{:.1}: {:7} observations, rms {:.4} px (x {:.4}, y {:.4})",
            r as f64 / RINGS as f64,
            (r + 1) as f64 / RINGS as f64,
            count[r],
            ((sum[0][r] + sum[1][r]) / n).sqrt(),
            (sum[0][r] / n).sqrt(),
            (sum[1][r] / n).sqrt()
        );
    }
}

fn usage(msg: &str) -> ! {
    eprintln!("{msg}\nusage: sfm_folder <dir> [--lens auto|pinhole|radial|brown|fisheye] [--focal <px>]");
    std::process::exit(2);
}
