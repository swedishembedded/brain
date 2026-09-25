// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structure from motion on a folder of photographs, reporting what the
//! calibration came out as: registered views, points, every candidate lens
//! model's reprojection RMS and information criterion, the chosen model and
//! each sensor's intrinsics (also as `cameras.json` fields), which
//! initializer was kept, where the time went, and the gauge - metres
//! east-north-up when the photographs' satellite fixes are good enough, and
//! the direction of gravity.
//!
//! ```text
//! cargo run --release --offline -p brain-sfm --example sfm_folder -- <dir> \
//!     [--lens auto|pinhole|radial|brown|fisheye] [--focal <px>] \
//!     [--pairs all|<k>] [--init auto|global|incremental] [--no-gps]
//! ```
//!
//! Photographs of different sizes are taken to be different sensors; one
//! `--focal` (pixels, e.g. from EXIF) applies to every photograph.
//! `--pairs <k>` matches each photograph with its `k` most similar by
//! retrieval (plus capture-order neighbours) however few photographs there
//! are; `--pairs all` matches every pair; the default decides by count.
//!
//! Swedish Embedded AB implements camera calibration and multi-view
//! reconstruction for its clients. If your team needs photographs turned into
//! calibrated cameras and geometry, you can procure our services by sending
//! an email to info@swedishembedded.com.

use sfm::georef::Wgs84;
use sfm::incremental::{reconstruct, Gauge, Initializer, Photo, SfmCfg};
use sfm::lens::{describe, LensChoice};
use sfm::retrieval::PairSelection;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir = None;
    let mut lens = LensChoice::Auto;
    let mut focal = None;
    let mut pairs = PairSelection::default();
    let mut initializer = None;
    let mut use_gps = true;
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
            "--pairs" => match args.next().as_deref() {
                Some("all") => pairs.exhaustive_up_to = usize::MAX,
                Some(k) => {
                    pairs.top_k = k.parse().unwrap_or_else(|_| usage("--pairs takes `all` or a number of pairs per photograph"));
                    pairs.exhaustive_up_to = 0;
                }
                None => usage("--pairs takes `all` or a number"),
            },
            "--init" => {
                initializer = match args.next().as_deref() {
                    Some("auto") => None,
                    Some("global") => Some(Initializer::Global),
                    Some("incremental") => Some(Initializer::Incremental),
                    other => usage(&format!("unknown initializer {other:?}")),
                }
            }
            "--no-gps" => use_gps = false,
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
    // satellite fixes from EXIF
    let fixes: Vec<Option<Wgs84>> = paths
        .iter()
        .map(|p| {
            let gps = if use_gps { imaging::load_photo(p).ok().and_then(|ph| ph.exif.gps) } else { None };
            gps.map(|g| Wgs84 { latitude_deg: g.latitude_deg, longitude_deg: g.longitude_deg, altitude_m: g.altitude_m })
        })
        .collect();
    // one sensor per distinct image size, numbered in order of appearance
    let mut sizes: Vec<(u32, u32)> = Vec::new();
    let photos: Vec<Photo> = images
        .iter()
        .zip(&fixes)
        .map(|(im, gps)| {
            let sensor = sizes.iter().position(|s| *s == (im.w, im.h)).unwrap_or_else(|| {
                sizes.push((im.w, im.h));
                sizes.len() - 1
            });
            Photo { width: im.w, height: im.h, rgb: &im.px, sensor, focal_px: focal, gps: *gps }
        })
        .collect();
    println!("{} photographs from {dir}, {} sensor(s), {} with satellite fixes", photos.len(), sizes.len(), fixes.iter().flatten().count());
    let cfg = SfmCfg { lens, pairs, initializer, verbose: true, ..SfmCfg::default() };
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
    let r = &rec.report;
    println!(
        "{:?} initializer; {} pairs matched, {} verified; features {:.1} s, matching {:.1} s, focal sweep {:.1} s, solve {:.1} s",
        r.initializer, r.pairs_matched, r.pairs_verified, r.features_s, r.matching_s, r.focal_s, r.solve_s
    );
    match &rec.gauge {
        Gauge::Seed { anchor, scale } => println!("gauge: photograph {anchor}'s frame, unit = its distance to photograph {scale}"),
        Gauge::Enu { origin, rms_m, fixes } => println!("gauge: metres east-north-up of {origin:?}, {fixes} fixes agree to {rms_m:.2} m"),
    }
    if let Some(e) = &r.georef_refused {
        println!("satellite fixes not used: {e}");
    }
    match rec.up {
        Some(u) => println!("up (against gravity): [{:.4}, {:.4}, {:.4}]", u[0], u[1], u[2]),
        None => println!("up: unknown"),
    }
    println!("lens models (fitted to the reconstruction's observations):");
    for f in &rec.lens_fits {
        let mark = if f.choice == rec.lens { "  <- chosen" } else { "" };
        println!("  {:<8} rms {:.4} px  {:2} parameters  BIC {:.1}{mark}", format!("{:?}", f.choice), f.rms_px, f.params, f.bic);
    }
    for (s, k) in rec.intrinsics.iter().enumerate() {
        println!("sensor {s} ({}x{}): {}", k.width, k.height, describe(k));
        println!("  cameras.json: {}", k.to_json());
    }
    radial_profile(&rec);
    if let Some(up) = rec.up {
        match dominant_plane(&rec) {
            Some((normal, share)) => {
                let tilt = sfm::linalg::dot(normal, up).abs().clamp(-1.0, 1.0).acos().to_degrees();
                println!("up is {tilt:.2} deg from the normal of the dominant plane of points ({:.0}% of them)", 100.0 * share);
            }
            None => println!("no dominant plane among the points to check up against"),
        }
    }
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

/// The normal of the plane the most points lie on (RANSAC over point
/// triples, inliers within 1% of the cloud's median radius, refined by the
/// inliers' principal axes) and the share of points on it; a capture on a
/// table or the ground has one, and gravity should be along its normal.
fn dominant_plane(rec: &sfm::incremental::Reconstruction) -> Option<(sfm::linalg::V3, f64)> {
    use sfm::linalg::{add, cross, dot, eigh, norm, normalize, scale, sub, V3};
    let p: Vec<V3> = rec.points.iter().map(|p| p.xyz).collect();
    if p.len() < 100 {
        return None;
    }
    let c = scale(p.iter().fold([0.0; 3], |a, b| add(a, *b)), 1.0 / p.len() as f64);
    let mut r: Vec<f64> = p.iter().map(|x| norm(sub(*x, c))).collect();
    let mid = r.len() / 2;
    let thresh = 0.01 * *r.select_nth_unstable_by(mid, f64::total_cmp).1;
    let mut rng = data::rng::Lcg::new(9);
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..2000 {
        let [a, b, d] = [0; 3].map(|_| rng.next_u32() as usize % p.len());
        let n = cross(sub(p[b], p[a]), sub(p[d], p[a]));
        if norm(n) < 1e-12 {
            continue;
        }
        let n = normalize(n);
        let inl: Vec<usize> = (0..p.len()).filter(|&i| dot(sub(p[i], p[a]), n).abs() < thresh).collect();
        if inl.len() > best.len() {
            best = inl;
        }
    }
    let m = scale(best.iter().fold([0.0; 3], |a, &i| add(a, p[i])), 1.0 / best.len() as f64);
    let mut cov = [0.0f64; 9];
    for &i in &best {
        let d = sub(p[i], m);
        for a in 0..3 {
            for b in 0..3 {
                cov[a * 3 + b] += d[a] * d[b];
            }
        }
    }
    let (_, vecs) = eigh(&cov, 3);
    Some(([vecs[0][0], vecs[0][1], vecs[0][2]], best.len() as f64 / p.len() as f64))
}

fn usage(msg: &str) -> ! {
    eprintln!("{msg}\nusage: sfm_folder <dir> [--lens auto|pinhole|radial|brown|fisheye] [--focal <px>] [--pairs all|<k>] [--init auto|global|incremental] [--no-gps]");
    std::process::exit(2);
}
