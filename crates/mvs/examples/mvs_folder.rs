// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense multi-view stereo on a folder of photographs: structure from motion
//! (cached), then every view's range and normal map on the GPU, fusion, and
//! the fused cloud as a starting splat scene.
//!
//! ```text
//! cargo run --release --offline -p brain-mvs --example mvs_folder -- <photos dir> <out dir> \
//!     [--sfm-cache <dir>] [--halving 1] [--levels 3] [--sources 8] [--photo-iters 6] \
//!     [--geom-iters 3] [--sigma-color 0.05] [--min-support 3] [--viz-every 4] [--profile 1]
//! ```
//!
//! Prints per-view coverage (the fraction of pixels with a filtered range),
//! per-stage timings and the fused point count, and writes to `<out dir>`:
//! `view_<i>.png` for every `--viz-every`-th view (photograph | inverse range
//! | normal | confidence, at the stereo's resolution) and `fused.ply`, the
//! surface-aligned gaussians of `mvs::to_splats`. The structure-from-motion
//! result is cached in `--sfm-cache` (default `<out dir>/sfm`). `--profile 1`
//! adds the device time of every kernel.
//!
//! Swedish Embedded AB implements photogrammetry pipelines - photographs to
//! calibrated cameras, dense geometry and radiance fields - for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use splat::types::{cameras_from_json, cameras_to_json, Camera};

/// Structure from motion for `photos`, read from `cache` when it holds a
/// result and written there when it does not: `cameras.json` (full
/// resolution, one per registered photograph), `source.txt` (the photograph
/// index of each camera) and `tracks.json` (every sparse point with the
/// cameras that observe it). A cache with no `tracks.json` - the one
/// `recon`'s `photo_holdout --sfm-cache` writes - falls back to `points.ply`,
/// whose points carry no observations.
fn cameras_and_tracks(photos: &[imaging::Rgb8], cache: &std::path::Path) -> (Vec<Camera>, Vec<usize>, Vec<mvs::select::Track>) {
    let cams_path = cache.join("cameras.json");
    let source_path = cache.join("source.txt");
    if cams_path.exists() && source_path.exists() {
        let cams = cameras_from_json(&std::fs::read_to_string(&cams_path).expect("cameras.json")).expect("cameras.json");
        let source: Vec<usize> = std::fs::read_to_string(&source_path)
            .expect("source.txt")
            .split_whitespace()
            .map(|s| s.parse().expect("source.txt holds photograph indices"))
            .collect();
        assert_eq!(cams.len(), source.len(), "cameras.json and source.txt disagree on the camera count");
        let tracks: Vec<mvs::select::Track> = match std::fs::read_to_string(cache.join("tracks.json")) {
            Ok(raw) => {
                let v: serde_json::Value = serde_json::from_str(&raw).expect("tracks.json");
                v.as_array()
                    .expect("tracks.json is an array")
                    .iter()
                    .map(|t| mvs::select::Track {
                        xyz: std::array::from_fn(|k| t["xyz"][k].as_f64().expect("track xyz")),
                        views: t["views"].as_array().expect("track views").iter().map(|x| x.as_u64().expect("view index") as usize).collect(),
                    })
                    .collect()
            }
            Err(_) => {
                let pts = splat::ply::read(&cache.join("points.ply").display().to_string()).expect("points.ply");
                pts.means.chunks_exact(3).map(|m| mvs::select::Track { xyz: [m[0] as f64, m[1] as f64, m[2] as f64], views: Vec::new() }).collect()
            }
        };
        println!("structure from motion: {} cameras, {} tracks, from {}", cams.len(), tracks.len(), cache.display());
        return (cams, source, tracks);
    }
    let views: Vec<sfm::incremental::Photo> =
        photos.iter().map(|p| sfm::incremental::Photo { width: p.w, height: p.h, rgb: &p.px, sensor: 0, focal_px: None, gps: None }).collect();
    let t = std::time::Instant::now();
    let rec = sfm::incremental::reconstruct(&views, &sfm::incremental::SfmCfg::default()).unwrap_or_else(|e| {
        eprintln!("structure from motion: {e}");
        std::process::exit(1);
    });
    let mut cams = Vec::new();
    let mut source = Vec::new();
    let mut view_of = vec![None; photos.len()];
    for (i, pose) in rec.poses.iter().enumerate() {
        let Some(pose) = pose else { continue };
        view_of[i] = Some(cams.len());
        cams.push(Camera::with_intrinsics(pose.c2w().map(|v| v as f32), &rec.intrinsics[rec.sensor[i]]));
        source.push(i);
    }
    let tracks = mvs::select::tracks_from_sfm(&rec, &view_of);
    println!(
        "structure from motion: {}/{} registered, {} tracks, rms {:.2} px, {:.0} s",
        cams.len(),
        photos.len(),
        tracks.len(),
        rec.rms_px,
        t.elapsed().as_secs_f64()
    );
    std::fs::create_dir_all(cache).expect("cache dir");
    std::fs::write(&cams_path, cameras_to_json(&cams)).expect("cameras.json");
    std::fs::write(&source_path, source.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n")).expect("source.txt");
    let tj: Vec<serde_json::Value> = tracks.iter().map(|t| serde_json::json!({"xyz": t.xyz, "views": t.views})).collect();
    std::fs::write(cache.join("tracks.json"), serde_json::to_string(&tj).expect("json")).expect("tracks.json");
    (cams, source, tracks)
}

/// `--name value` flags after the two positional arguments.
struct Args {
    dir: String,
    out: std::path::PathBuf,
    flags: std::collections::HashMap<String, String>,
}

impl Args {
    fn parse() -> Args {
        let mut it = std::env::args().skip(1);
        let mut pos = Vec::new();
        let mut flags = std::collections::HashMap::new();
        while let Some(a) = it.next() {
            match a.strip_prefix("--") {
                Some(name) => {
                    let v = it.next().unwrap_or_else(|| usage(&format!("--{name} needs a value")));
                    flags.insert(name.to_string(), v);
                }
                None => pos.push(a),
            }
        }
        if pos.len() != 2 {
            usage("a photographs directory and an output directory are required");
        }
        Args { dir: pos[0].clone(), out: pos[1].clone().into(), flags }
    }

    fn get<T: std::str::FromStr>(&self, name: &str, default: T) -> T {
        self.flags.get(name).map(|v| v.parse().unwrap_or_else(|_| usage(&format!("--{name}: cannot parse {v:?}")))).unwrap_or(default)
    }
}

fn usage(msg: &str) -> ! {
    eprintln!(
        "{msg}\nusage: mvs_folder <photos dir> <out dir> [--sfm-cache <dir>] [--halving 1] [--levels 3] \
         [--sources 8] [--photo-iters 6] [--geom-iters 3] [--sigma-color 0.05] [--min-support 3] [--viz-every 4] [--profile 1]"
    );
    std::process::exit(2);
}

/// Photograph | inverse range | normal | confidence, 2 x 2.
fn montage(rgb: &[f32], d: &mvs::DepthMap) -> imaging::Rgb8 {
    let (w, h) = (d.width as usize, d.height as usize);
    let photo: Vec<u8> = rgb.iter().map(|v| (v * 255.0).round() as u8).collect();
    let inv: Vec<f32> = d.range.iter().map(|r| if *r > 0.0 { 1.0 / r } else { f32::NAN }).collect();
    let measured: Vec<f32> = inv.iter().copied().filter(|v| v.is_finite()).collect();
    let bounds = imaging::viz::Bounds::from_percentiles(&measured, 0.02, 0.98);
    let mut depth = imaging::viz::colorize(&inv, bounds, imaging::viz::Colormap::Turbo);
    let mut normal = vec![0u8; w * h * 3];
    let mut conf = vec![0u8; w * h * 3];
    for i in 0..w * h {
        if d.range[i] <= 0.0 {
            depth[3 * i..3 * i + 3].fill(0);
            continue;
        }
        for c in 0..3 {
            // camera frame, facing the camera: x right, y down, z toward the viewer
            let n = if c == 2 { -d.normal[3 * i + c] } else { d.normal[3 * i + c] };
            normal[3 * i + c] = ((n * 0.5 + 0.5) * 255.0).round() as u8;
        }
        conf[3 * i..3 * i + 3].fill((d.conf[i] * 255.0).round() as u8);
    }
    let mut px = vec![0u8; 4 * w * h * 3];
    for (t, tile) in [photo, depth, normal, conf].iter().enumerate() {
        let (ox, oy) = ((t % 2) * w, (t / 2) * h);
        for y in 0..h {
            let dst = ((oy + y) * 2 * w + ox) * 3;
            px[dst..dst + 3 * w].copy_from_slice(&tile[y * w * 3..(y + 1) * w * 3]);
        }
    }
    imaging::Rgb8 { w: 2 * w as u32, h: 2 * h as u32, px }
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out).unwrap_or_else(|e| usage(&format!("{}: {e}", args.out.display())));
    let cache: std::path::PathBuf = args.get("sfm-cache", args.out.join("sfm").display().to_string()).into();
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&args.dir)
        .unwrap_or_else(|e| usage(&format!("{}: {e}", args.dir)))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()).is_some_and(|x| ["jpg", "jpeg", "png"].contains(&x.to_ascii_lowercase().as_str())))
        .collect();
    paths.sort();
    let t = std::time::Instant::now();
    let photos: Vec<imaging::Rgb8> = paths.iter().map(|p| imaging::load(p).unwrap_or_else(|e| usage(&e))).collect();
    println!("{} photographs decoded in {:.1} s", photos.len(), t.elapsed().as_secs_f64());
    let (cams, source, tracks) = cameras_and_tracks(&photos, &cache);

    let cfg = mvs::StereoCfg {
        halving: args.get("halving", 1),
        levels: args.get("levels", 3),
        photo_iters: args.get("photo-iters", 6),
        geom_iters: args.get("geom-iters", 3),
        sigma_color: args.get("sigma-color", mvs::StereoCfg::default().sigma_color),
        min_var: args.get("min-var", mvs::StereoCfg::default().min_var),
        select: mvs::SelectCfg { max_sources: args.get("sources", 8), ..Default::default() },
        ..Default::default()
    };
    let views: Vec<mvs::View> = cams.iter().zip(&source).map(|(c, &s)| mvs::View { cam: *c, rgb: &photos[s].px }).collect();
    let gpu = gpu_core::Gpu::new(mvs::PIPELINES);
    let ks = mvs::Kernels::at(0);
    let profile = args.get("profile", 0) != 0 && gpu.set_kernel_timing(true);
    let t = std::time::Instant::now();
    let st = mvs::depth_maps(&gpu, &ks, &views, &tracks, &cfg).unwrap_or_else(|e| {
        eprintln!("stereo: {e}");
        std::process::exit(1);
    });
    let stereo_secs = t.elapsed().as_secs_f64();
    println!("stereo at {}x{} ({:.1} s):\n{}", st.cams[0].width, st.cams[0].height, stereo_secs, st.timings);
    let mut total = 0.0;
    for (v, d) in st.depth.iter().enumerate() {
        let c = d.coverage();
        total += c;
        let conf = d.conf.iter().filter(|c| **c > 0.0).sum::<f32>() / d.conf.iter().filter(|c| **c > 0.0).count().max(1) as f32;
        println!(
            "  view {v:2} ({}): coverage {:5.1} %, mean confidence {conf:.2}, sources {:?}",
            paths[source[v]].file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            100.0 * c,
            st.sources[v]
        );
    }
    println!("  mean coverage {:.1} %", 100.0 * total / st.depth.len().max(1) as f64);

    let every: usize = args.get("viz-every", 4);
    for v in (0..st.depth.len()).step_by(every.max(1)) {
        let path = args.out.join(format!("view_{v:02}.png"));
        imaging::save(&path, &montage(&st.rgb[v], &st.depth[v])).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }

    let t = std::time::Instant::now();
    let fuse_cfg = mvs::FuseCfg { min_support: args.get("min-support", 3), ..Default::default() };
    let fused = mvs::fuse(&gpu, &ks, &st.cams, &st.depth, &st.rgb, &fuse_cfg).unwrap_or_else(|e| {
        eprintln!("fusion: {e}");
        std::process::exit(1);
    });
    let fuse_secs = t.elapsed().as_secs_f64();
    let mean_support = fused.support.iter().map(|s| *s as f64).sum::<f64>() / fused.len().max(1) as f64;
    println!("fusion: {} points, mean support {mean_support:.1} views, {fuse_secs:.2} s", fused.len());
    let splats = mvs::to_splats(&fused, &mvs::SplatInit::default());
    let ply = args.out.join("fused.ply");
    splat::ply::write(&ply.display().to_string(), &splats).unwrap_or_else(|e| panic!("{}: {e}", ply.display()));
    if let Some(times) = profile.then(|| gpu.kernel_times()).flatten() {
        println!("device time per kernel:");
        for (name, ms, calls) in times {
            println!("  {name:<16} {:9.1} ms  {calls:6} dispatches", ms);
        }
    }
    println!("wrote {} and {} view visualisations to {}", ply.display(), st.depth.len().div_ceil(every.max(1)), args.out.display());
}
