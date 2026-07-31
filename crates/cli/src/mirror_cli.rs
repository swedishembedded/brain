// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain mirror …` — WorldMirror-2 multi-view 3D reconstruction.
//!
//!   brain mirror import <safetensors|hf_dir> --out mirror.weights
//!   brain mirror infer  --weights F --images <dir|a.ppm,b.ppm,…> [--out DIR]
//!         [--ply scene.ply] [--maps] [--min-opacity X] [--max-depth X]
//!         [--prune VOXEL]   (voxel-merge duplicates, try 0.002 for multi-view)
//!   brain mirror demo   --weights F --images <…> [viewer flags] [--prune VOXEL]
//!
//! Inputs are P6 PPM images; any aspect ratio (the DINOv2 pos-embed is
//! bicubic-interpolated for non-native grids, reference semantics).

use gpu_core::Gpu;
use mirror::config::MirrorConfig;
use mirror::gaussians::{assemble, frame_maps, AssembleOpts};
use mirror::model::Mirror;
use mirror::preprocess;
use splat::types::Splats;

use crate::args::Args;

pub fn run_mirror(argv: &[String]) {
    match argv.first().map(|s| s.as_str()) {
        Some("import") => import(&argv[1..]),
        Some("infer") => infer(&argv[1..]),
        Some("demo") => demo(&argv[1..]),
        Some("export-npu") => export_npu(&argv[1..]),
        other => {
            eprintln!("usage: brain mirror <import|infer|demo|export-npu> ...  (got {other:?})");
            std::process::exit(2);
        }
    }
}

fn collect_images(spec: &str) -> Vec<String> {
    let p = std::path::Path::new(spec);
    if p.is_dir() {
        let mut v: Vec<String> = std::fs::read_dir(p)
            .expect("readable dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".ppm"))
            .collect();
        v.sort();
        v
    } else {
        spec.split(',').map(|s| s.trim().to_string()).collect()
    }
}

/// Load + preprocess frames; returns (raw [0,1] CHW concat, frame count, grid).
fn load_frames(spec: &str, cfg: &MirrorConfig) -> (Vec<f32>, usize, usize, usize) {
    let paths = collect_images(spec);
    if paths.is_empty() {
        eprintln!("no .ppm images found in {spec}");
        std::process::exit(2);
    }
    let mut all = Vec::new();
    let mut grid = None;
    for path in &paths {
        let img = imaging::load(path).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        let (iw, ih) = (img.w as usize, img.h as usize);
        let target = preprocess::adaptive_target(iw, ih, cfg.img, cfg.patch);
        let (nw, nh) = preprocess::resize_dims(iw, ih, target, cfg.patch);
        let resized = preprocess::resize_bicubic(&img, nw, nh);
        let (cw, ch) = (nw.min(target), nh.min(target));
        let (x0, y0) = ((nw - cw) / 2, (nh - ch) / 2);
        for c in 0..3 {
            for y in 0..ch {
                for x in 0..cw {
                    all.push(resized.px[((y0 + y) * nw + x0 + x) * 3 + c] as f32 / 255.0);
                }
            }
        }
        let g = (cw / cfg.patch, ch / cfg.patch);
        assert!(grid.is_none() || grid == Some(g), "mixed image sizes");
        grid = Some(g);
    }
    let (wp, hp) = grid.unwrap();
    (all, paths.len(), hp, wp)
}

/// Run the model, assemble the scene, then hand everything to `k` (the model
/// borrows the Gpu, so the whole flow lives in one scope).
#[allow(clippy::type_complexity)]
fn with_scene<R>(
    weights: &str,
    images: &str,
    min_op: f32,
    max_depth: f32,
    prune_voxel: f32,
    k: impl FnOnce(&Gpu, &Mirror, &Splats, &[splat::types::Camera], usize, u32, u32) -> R,
) -> R {
    let cfg = MirrorConfig::default();
    let (frames, s, hp, wp) = load_frames(images, &cfg);
    let (w, h) = ((wp * cfg.patch) as u32, (hp * cfg.patch) as u32);
    eprintln!("loading {weights} …");
    let init = mirror::import::load_weights(weights, &cfg).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    // model + splat pipelines share one Gpu (the demo renders the result)
    let pipes: Vec<(&str, &str)> =
        mirror::model::PIPELINES.iter().chain(splat::PIPELINES.iter()).copied().collect();
    let gpu = Gpu::new(&pipes);
    let mut model = Mirror::new(&gpu, cfg, &init, 0);
    drop(init);
    eprintln!("running WorldMirror-2 on {s} frame(s) at {w}x{h} …");
    let t0 = std::time::Instant::now();
    model.forward(&frames, s, hp, wp);
    let opts = AssembleOpts { min_opacity: min_op, max_depth };
    let (mut splats, cams, weights) = assemble(&gpu, &model, &frames, s, w, h, &opts);
    eprintln!(
        "forward + assembly: {:.1}s, {} gaussians",
        t0.elapsed().as_secs_f32(),
        splats.len()
    );
    if prune_voxel > 0.0 {
        let before = splats.len();
        splats = splat::prune::voxel_merge(&splats, &weights, prune_voxel, 0);
        eprintln!("voxel prune ({prune_voxel}): {before} -> {} gaussians", splats.len());
    }
    k(&gpu, &model, &splats, &cams, s, w, h)
}

/// Grayscale (depth, min-max normalized) and normal-map PPMs for inspection.
fn write_maps(gpu: &Gpu, model: &Mirror, fi: usize, w: u32, h: u32, out_dir: &str) {
    let m = frame_maps(gpu, model, fi, w, h);
    let hw = (w * h) as usize;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &d in &m.depth {
        lo = lo.min(d);
        hi = hi.max(d);
    }
    let span = (hi - lo).max(1e-9);
    let mut rgb = Vec::with_capacity(hw * 3);
    for &d in &m.depth {
        let v = (((d - lo) / span) * 255.0) as u8;
        rgb.extend_from_slice(&[v, v, v]);
    }
    crate::splat_cli::write_ppm_rgb(&format!("{out_dir}/depth_{fi:02}.ppm"), &rgb, w as usize, h as usize);
    let mut nrgb = Vec::with_capacity(hw * 3);
    for i in 0..hw {
        for c in 0..3 {
            nrgb.push(((m.normals[c * hw + i] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8);
        }
    }
    crate::splat_cli::write_ppm_rgb(&format!("{out_dir}/normal_{fi:02}.ppm"), &nrgb, w as usize, h as usize);
}

fn write_cameras_json(path: &str, cams: &[splat::types::Camera]) {
    let arr: Vec<serde_json::Value> = cams
        .iter()
        .map(|c| {
            serde_json::json!({
                "c2w": c.c2w.to_vec(),
                "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
                "width": c.width, "height": c.height,
            })
        })
        .collect();
    std::fs::write(path, serde_json::to_string_pretty(&arr).unwrap())
        .unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
}

/// Export model stages as fp32 ONNX for OpenVINO (NPU/CPU). `--stage dino`
/// (per-frame encoder) or `--stage trunk` (fixed-S alternating-attention
/// trunk → 4 taps). Weights external (model.onnx + model.onnx.data).
fn export_npu(argv: &[String]) {
    let mut a = Args::new(argv);
    let weights = a.str_or("--weights", "out/mirror.weights");
    let stage = a.str_or("--stage", "dino");
    let out = a.str_or("--out", &format!("out/mirror-{stage}.onnx"));
    let s = a.u32_or("--frames", 1) as usize;
    let hp = a.u32_or("--hp", 37) as usize;
    let wp = a.u32_or("--wp", 37) as usize;
    // debug knobs for bisecting device parity: fewer levels, taps anywhere
    let cfg = MirrorConfig::default();
    let levels = a.u32_or("--levels", cfg.depth as u32) as usize;
    let tap_spec = a.take_str("--tap-levels");
    a.finish();
    eprintln!("loading {weights} …");
    let init = mirror::import::load_weights(&weights, &cfg).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if stage == "heads" {
        // one graph per DPT head (the gs head carries the rgb-merge branch)
        for (name, out_ch, gs) in
            [("depth_head", 3i64, false), ("pts_head", 4, false), ("norm_head", 4, false), ("gs_head", 3, true)]
        {
            let mut g = onnx::builder::GraphBuilder::new(&format!("mirror_{name}"));
            npu::mirror_topology::build_dpt_head_graph(&init, &mut g, &cfg, name, out_ch, hp, wp, gs);
            let path = format!("out/mirror-{name}.onnx");
            g.finish_external(&path, 1 << 20).unwrap_or_else(|e| {
                eprintln!("ONNX write failed: {e}");
                std::process::exit(1);
            });
            println!("wrote {path} (+ external weight data)");
        }
        return;
    }
    let mut g = onnx::builder::GraphBuilder::new(&format!("mirror_{stage}"));
    match stage.as_str() {
        "dino" => npu::mirror_topology::build_dinov2_graph(&init, &mut g, cfg.depth),
        "trunk" => {
            let taps: Vec<usize> = match &tap_spec {
                Some(spec) => spec.split(',').map(|v| v.trim().parse().expect("--tap-levels N,N,…")).collect(),
                None => cfg.tap_levels.to_vec(),
            };
            npu::mirror_topology::build_trunk_graph(&init, &mut g, s, hp, wp, levels, &taps)
        }
        other => {
            eprintln!("unknown --stage {other} (dino|trunk|heads)");
            std::process::exit(2);
        }
    }
    g.finish_external(&out, 1 << 20).unwrap_or_else(|e| {
        eprintln!("ONNX write failed: {e}");
        std::process::exit(1);
    });
    println!("wrote {out} (+ external weight data)");
    println!("verify: python3 tools/mirror_check_onnx.py {out}   (OpenVINO CPU/NPU)");
}

fn infer(argv: &[String]) {
    let mut a = Args::new(argv);
    let weights = a.str_or("--weights", "out/mirror.weights");
    let images = a.take_str("--images").unwrap_or_else(|| {
        eprintln!("--images <dir|a.ppm,b.ppm,…> is required");
        std::process::exit(2);
    });
    let out_dir = a.str_or("--out", "out/mirror");
    let ply = a.take_str("--ply");
    let maps = a.take_flag("--maps");
    let min_op = a.f32_or("--min-opacity", 0.01);
    let max_depth = a.f32_or("--max-depth", 0.0);
    let prune = a.f32_or("--prune", 0.0);
    a.finish();

    std::fs::create_dir_all(&out_dir).ok();
    let ply_path = ply.unwrap_or_else(|| format!("{out_dir}/scene.ply"));
    with_scene(&weights, &images, min_op, max_depth, prune, |gpu, model, splats, cams, s, w, h| {
        splat::ply::write(&ply_path, splats).unwrap_or_else(|e| {
            eprintln!("PLY write failed: {e}");
            std::process::exit(1);
        });
        write_cameras_json(&format!("{out_dir}/cameras.json"), cams);
        println!("wrote {ply_path} ({} gaussians) + {out_dir}/cameras.json", splats.len());
        if maps {
            for fi in 0..s {
                write_maps(gpu, model, fi, w, h, &out_dir);
            }
        }
        println!("view: brain splat view {ply_path}");
    });
}

fn demo(argv: &[String]) {
    let mut a = Args::new(argv);
    let weights = a.str_or("--weights", "out/mirror.weights");
    let images = a.take_str("--images").unwrap_or_else(|| {
        eprintln!("--images <dir|a.ppm,b.ppm,…> is required");
        std::process::exit(2);
    });
    let width = a.u32_or("--width", 1280);
    let height = a.u32_or("--height", 720);
    let fov = a.f32_or("--fov", 60.0);
    let frames_cap = a.opt_u32("--frames").map(|n| n as u64);
    let min_op = a.f32_or("--min-opacity", 0.01);
    let max_depth = a.f32_or("--max-depth", 0.0);
    let prune = a.f32_or("--prune", 0.0);
    a.finish();

    let (splats, init_cam) = with_scene(&weights, &images, min_op, max_depth, prune, |_gpu, _model, splats, cams, _s, _w, _h| {
        let init_cam = cams.first().map(|c| splat::types::Camera {
            width,
            height,
            fx: c.fx * width as f32 / c.width as f32,
            fy: c.fy * height as f32 / c.height as f32,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            ..*c
        });
        (splats.clone(), init_cam)
    });
    crate::splat_cli::run_viewer(
        &splats,
        "brain mirror — WorldMirror-2",
        width,
        height,
        fov,
        [0.02, 0.02, 0.03],
        frames_cap,
        init_cam,
    );
}

fn import(argv: &[String]) {
    let mut a = Args::new(argv);
    let out = a.str_or("--out", "out/mirror.weights");
    let src = a.positional().unwrap_or_else(|| {
        eprintln!("usage: brain mirror import <model.safetensors|hf_dir> --out mirror.weights");
        std::process::exit(2);
    });
    a.finish();

    let cfg = MirrorConfig::default();
    println!("importing {src} ({} tensors expected) …", cfg.param_list().len());
    match mirror::import::convert(&src, &out, &cfg) {
        Ok(n) => println!("wrote {out}: {n} tensors, all consumed, shapes verified"),
        Err(e) => {
            eprintln!("import failed: {e}");
            std::process::exit(1);
        }
    }
}
