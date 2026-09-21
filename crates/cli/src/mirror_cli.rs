// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain worldmirror2 …` - WorldMirror-2 multi-view 3D reconstruction.
//!
//!   brain worldmirror2 import <safetensors|hf_dir> --out mirror.safetensors
//!   brain worldmirror2 infer  --weights F --images <dir|a.ppm,b.ppm,…> [--out DIR]
//!         [--ply scene.ply] [--maps] [--min-opacity X] [--max-depth X]
//!         [--prune VOXEL]   (voxel-merge duplicates, try 0.002 for multi-view)
//!         [--keep-camera-frame]  (write in frame 0's frame, not an upright one)
//!   brain worldmirror2 demo   --weights F --images <…> [viewer flags] [--prune VOXEL]
//!
//! Inputs are P6 PPM images; any aspect ratio (the DINOv2 pos-embed is
//! bicubic-interpolated for non-native grids, reference semantics).

use gpu_core::Gpu;
use worldmirror2::config::MirrorConfig;
use worldmirror2::gaussians::{assemble, frame_maps, AssembleOpts};
use worldmirror2::model::Mirror;
use worldmirror2::preprocess;
use splat::types::Splats;

use crate::args::Args;

pub fn run_mirror(argv: &[String]) {
    match argv.first().map(|s| s.as_str()) {
        Some("import") => import(&argv[1..]),
        Some("infer") => infer(&argv[1..]),
        Some("demo") => demo(&argv[1..]),
        Some("export-npu") => export_npu(&argv[1..]),
        other => {
            eprintln!("usage: brain worldmirror2 <import|infer|demo|export-npu> ...  (got {other:?})");
            std::process::exit(2);
        }
    }
}

/// Extensions `imaging::load` decodes. It dispatches on the BYTES, not the
/// name, so this list only has to keep non-images out of a directory listing.
pub const IMAGE_EXTS: [&str; 7] = ["ppm", "png", "jpg", "jpeg", "bmp", "tif", "tiff"];

/// Extensions treated as a video to decode frames from.
const VIDEO_EXTS: [&str; 6] = ["mp4", "mov", "mkv", "webm", "avi", "m4v"];

/// Read known cameras from the `cameras.json` shape `infer` itself writes, so
/// a reconstruction's own output can be fed straight back as a prior.
fn read_camera_priors(path: &str, s: usize) -> Vec<worldmirror2::priors::CameraPrior> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let j: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let arr = j.as_array().unwrap_or_else(|| {
        eprintln!("{path}: expected a JSON array of cameras");
        std::process::exit(1);
    });
    if arr.len() != s {
        eprintln!("{path} holds {} camera(s) but the run has {s} frame(s); a pose prior is per frame", arr.len());
        std::process::exit(2);
    }
    arr.iter()
        .map(|c| {
            let m: Vec<f32> = c["c2w"].as_array().expect("c2w").iter().map(|v| v.as_f64().unwrap() as f32).collect();
            worldmirror2::priors::CameraPrior {
                c2w: m.try_into().expect("16 c2w entries"),
                fx: c["fx"].as_f64().unwrap_or(0.0) as f32,
                fy: c["fy"].as_f64().unwrap_or(0.0) as f32,
                cx: c["cx"].as_f64().unwrap_or(0.0) as f32,
                cy: c["cy"].as_f64().unwrap_or(0.0) as f32,
            }
        })
        .collect()
}

/// Does `spec` name a video file (rather than a directory or an image list)?
fn is_video(spec: &str) -> bool {
    let p = std::path::Path::new(spec);
    p.is_file() && VIDEO_EXTS.contains(&ext_of(p).as_str())
}

fn ext_of(p: &std::path::Path) -> String {
    p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

pub fn collect_images(spec: &str) -> Vec<String> {
    let p = std::path::Path::new(spec);
    if p.is_dir() {
        let mut v: Vec<String> = std::fs::read_dir(p)
            .expect("readable dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| IMAGE_EXTS.contains(&ext_of(p).as_str()))
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        // Sorted, so a capture's own frame order is the order the model sees.
        v.sort();
        v
    } else {
        spec.split(',').map(|s| s.trim().to_string()).collect()
    }
}

/// Take every `stride`-th frame, then thin to at most `max` of those.
///
/// The cap SPANS the sequence rather than truncating it: a 360-degree orbit
/// capped to a prefix would become a 60-degree one, which is a far worse
/// reconstruction than the same budget spread over the whole path. Never
/// returns an empty set for a non-empty input.
fn select_frames(paths: &[String], stride: usize, max: usize) -> Vec<String> {
    let stride = stride.max(1);
    let strided: Vec<String> = paths.iter().step_by(stride).cloned().collect();
    let strided = if strided.is_empty() { paths[..paths.len().min(1)].to_vec() } else { strided };
    if max == 0 || strided.len() <= max {
        return strided;
    }
    (0..max)
        .map(|i| strided[i * (strided.len() - 1) / (max - 1).max(1)].clone())
        .collect()
}

/// Black out every pixel the mask leaves unlit, in place.
///
/// The mask is resized to the frame with nearest-neighbour sampling: it is a
/// keep/drop decision per pixel, and interpolating it would invent
/// half-transparent border pixels the model would then try to explain.
///
/// This exists for TURNTABLE captures. The model assumes one rigid scene and a
/// moving camera; a turntable gives it a rotating object in front of a
/// stationary background, which is not that, and the static part is evidence
/// for a camera that never moved. Mask the background away and what is left -
/// an object turning in front of nothing - is exactly equivalent to a camera
/// orbiting a still object, which is what the model was trained on. One mask
/// covers the whole capture, because the camera does not move.
fn apply_mask(img: &mut imaging::Rgb8, mask: &imaging::Rgb8) {
    let (mw, mh) = (mask.w as usize, mask.h as usize);
    let (w, h) = (img.w as usize, img.h as usize);
    for y in 0..h {
        let my = y * mh / h;
        for x in 0..w {
            let mx = x * mw / w;
            if mask.px[(my * mw + mx) * 3] < 128 {
                let i = (y * w + x) * 3;
                img.px[i..i + 3].fill(0);
            }
        }
    }
}

/// How the caller narrowed a long capture down to the frames the model sees.
#[derive(Clone, Copy)]
pub struct FrameSel {
    pub stride: usize,
    pub max: usize,
    /// Frames per second to resample a VIDEO to before selecting. 0 = native.
    pub fps: f64,
}

impl Default for FrameSel {
    fn default() -> FrameSel {
        // Unbounded by default: a directory of photographs is already the set
        // the user chose. A video gets a cap, because 60 seconds at 30fps is
        // 1800 frames through a trunk whose global attention is quadratic in
        // frame count.
        FrameSel { stride: 1, max: 0, fps: 0.0 }
    }
}

/// Every frame the model will see, from a directory, a comma-separated list,
/// or a video file.
fn gather(spec: &str, sel: &FrameSel, mask: Option<&str>) -> Vec<imaging::Rgb8> {
    let mask = mask.map(|m| {
        imaging::load(m).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        })
    });
    let mut frames = gather_unmasked(spec, sel);
    if let Some(m) = &mask {
        for f in frames.iter_mut() {
            apply_mask(f, m);
        }
    }
    frames
}

fn gather_unmasked(spec: &str, sel: &FrameSel) -> Vec<imaging::Rgb8> {
    let p = std::path::Path::new(spec);
    if p.is_file() && VIDEO_EXTS.contains(&ext_of(p).as_str()) {
        if !imaging::video::ffmpeg_available() {
            eprintln!("reading {spec} needs the ffmpeg CLI on PATH (Debian/Ubuntu: apt-get install ffmpeg)");
            std::process::exit(1);
        }
        // Selection happens INSIDE ffmpeg: `spread` asks for N frames spaced
        // over the whole clip, so the frames nobody wants are dropped before
        // they are converted and nothing is staged on disk. Decoding the clip
        // and thinning afterwards cost a full decode and a temp file per
        // frame to keep a couple of dozen.
        let opts = imaging::video::VideoDecodeOpts {
            fps: if sel.fps > 0.0 { Some(sel.fps) } else { None },
            max_frames: 0,
            spread: if sel.fps > 0.0 { 0 } else { sel.max as u32 },
        };
        let frames = imaging::video::decode_frames_rgb8(p, &opts).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        // `--stride` (and an explicit --fps with a cap) still thin what came back
        let n = frames.len();
        let idx = select_frames(&(0..n).map(|i| format!("{i:08}")).collect::<Vec<_>>(), sel.stride, sel.max);
        let keep: Vec<usize> = idx.iter().map(|k| k.parse().unwrap()).collect();
        eprintln!("{spec}: {n} frame(s) decoded, using {}", keep.len());
        keep.into_iter().map(|i| frames[i].clone()).collect()
    } else {
        let paths = select_frames(&collect_images(spec), sel.stride, sel.max);
        if paths.is_empty() {
            eprintln!(
                "no images found in {spec} (looked for {}; a video needs one of {})",
                IMAGE_EXTS.join("/"), VIDEO_EXTS.join("/")
            );
            std::process::exit(2);
        }
        paths
            .iter()
            .map(|path| {
                imaging::load(path).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                })
            })
            .collect()
    }
}

/// Load + preprocess frames; returns (raw [0,1] CHW concat, frame count, grid).
fn load_frames(spec: &str, cfg: &MirrorConfig, sel: &FrameSel, mask: Option<&str>, target: usize) -> (Vec<f32>, usize, usize, usize) {
    let images = gather(spec, sel, mask);
    let mut all = Vec::new();
    let mut grid = None;
    for img in &images {
        let (iw, ih) = (img.w as usize, img.h as usize);
        // `cfg.img` is the checkpoint's NATIVE grid, the size its position
        // embedding was trained at - not a ceiling on inference. The model
        // interpolates that embedding to whatever grid it is handed
        // (`Mirror::new`, torch-bicubic parity), which is what WorldMirror-2's
        // normalized RoPE was added to support. Using the native size as the
        // preprocessing target held every reconstruction to 518 and roughly a
        // third of the reference's sample count.
        let target = preprocess::adaptive_target(iw, ih, target, cfg.patch);
        let (nw, nh) = preprocess::resize_dims(iw, ih, target, cfg.patch);
        let resized = preprocess::resize_bicubic(img, nw, nh);
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
    (all, images.len(), hp, wp)
}

/// Run the model, assemble the scene, then hand everything to `k` (the model
/// OWNS the Gpu now - `model.gpu()` is how `k` and this function reach it -
/// so the whole flow still lives in one scope, but only because `k` runs
/// before `model` is dropped, not because of a borrow).
#[allow(clippy::type_complexity)]
fn with_scene<R>(
    weights: &str,
    images: &str,
    min_op: f32,
    max_depth: f32,
    prune_voxel: f32,
    poses: Option<&str>,
    gs_mask: f32,
    edge_rtol: f32,
    fuse_rtol: f32,
    scale_q: f32,
    sel: &FrameSel,
    mask: Option<&str>,
    target: usize,
    k: impl FnOnce(&Gpu, &Mirror, &Splats, &[splat::types::Camera], usize, u32, u32) -> R,
) -> R {
    let cfg = MirrorConfig::default();
    let (frames, s, hp, wp) = load_frames(images, &cfg, sel, mask, target);

    let (w, h) = ((wp * cfg.patch) as u32, (hp * cfg.patch) as u32);
    eprintln!("loading {weights} …");
    let init = worldmirror2::import::load_weights(weights, &cfg).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    // model + splat pipelines share one Gpu (the demo renders the result)
    let pipes: Vec<(&str, &str)> =
        worldmirror2::model::PIPELINES.iter().chain(splat::PIPELINES.iter()).copied().collect();
    let gpu = Gpu::new(&pipes);
    // What a shape actually costs in ONE binding. Global attention is
    // query-chunked, so this is not the token count squared - an earlier
    // version assumed it was and refused frame/resolution combinations that
    // run comfortably.
    let need = worldmirror2::model::largest_binding_bytes(&cfg, s, hp, wp);
    let limit = gpu.max_storage_binding_bytes();
    if limit > 0 && need > limit {
        eprintln!(
            "{s} frame(s) at {}x{} needs {:.2} GiB in one storage binding against this device's \
             {:.2} GiB limit. Lower --target-size or use fewer frames (--max-frames/--stride).",
            wp * cfg.patch, hp * cfg.patch,
            need as f64 / (1u64 << 30) as f64, limit as f64 / (1u64 << 30) as f64
        );
        std::process::exit(2);
    }
    let mut model = Mirror::new(gpu, cfg, &init, 0);
    drop(init);
    eprintln!("running WorldMirror-2 on {s} frame(s) at {w}x{h} …");
    let t0 = std::time::Instant::now();
    let priors = poses.map(|p| read_camera_priors(p, s));
    // The same known cameras become the back-projection cameras, not just a
    // hint to the trunk - see `gaussians::assemble`.
    let known: Option<Vec<splat::types::Camera>> = priors.as_ref().map(|ps| {
        ps.iter()
            .map(|p| splat::types::Camera {
                c2w: p.c2w, fx: p.fx, fy: p.fy, cx: p.cx, cy: p.cy, width: w, height: h,
            })
            .collect()
    });
    if priors.is_some() {
        eprintln!("conditioning on {s} known camera(s) from the pose prior");
    }
    model.forward_with_priors(&frames, s, hp, wp, priors.as_deref());
    let opts = AssembleOpts {
        min_opacity: min_op,
        max_depth,
        gs_mask_threshold: gs_mask,
        edge_depth_rtol: edge_rtol,
        fuse_depth_rtol: fuse_rtol,
    };
    let (mut splats, cams, weights) = assemble(model.gpu(), &model, &frames, s, w, h, &opts, known.as_deref());
    eprintln!(
        "forward + assembly: {:.1}s, {} gaussians",
        t0.elapsed().as_secs_f32(),
        splats.len()
    );
    if prune_voxel > 0.0 {
        let before = splats.len();
        splats = splat::prune::voxel_merge(&splats, &weights, prune_voxel, 0);
        eprintln!("voxel fusion ({prune_voxel}): {before} -> {} gaussians", splats.len());
    }
    if scale_q > 0.0 && scale_q < 1.0 {
        let before = splats.len();
        splats = splat::prune::drop_largest_scales(&splats, scale_q);
        eprintln!("largest-scale rejection (q={scale_q}): {before} -> {} gaussians", splats.len());
    }
    k(model.gpu(), &model, &splats, &cams, s, w, h)
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

/// Re-express a scene and its cameras in the frame the cameras describe: the
/// vertical is the axis they sweep about, the origin is where they all look.
fn upright(
    splats: &splat::types::Splats,
    cams: &[splat::types::Camera],
) -> (splat::types::Splats, Vec<splat::types::Camera>) {
    let mats: Vec<[f64; 16]> = cams
        .iter()
        .map(|c| std::array::from_fn(|i| c.c2w[i] as f64))
        .collect();
    let (r, centre) = splat::orient::frame_from_cameras(&mats, -1.0);
    let moved = cams
        .iter()
        .zip(&mats)
        .map(|(c, m)| {
            let t = splat::orient::transform_c2w(m, &r, &centre);
            splat::types::Camera { c2w: std::array::from_fn(|i| t[i] as f32), ..*c }
        })
        .collect();
    (splat::orient::apply(splats, &r, &centre), moved)
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
    let weights = a.str_or("--weights", "out/mirror.safetensors");
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
    let init = worldmirror2::import::load_weights(&weights, &cfg).unwrap_or_else(|e| {
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
    println!("verify: python3 tools/goldens/worldmirror2_check_onnx.py {out}   (OpenVINO CPU/NPU)");
}

fn infer(argv: &[String]) {
    let mut a = Args::new(argv);
    let weights = a.str_or("--weights", "out/mirror.safetensors");
    let images = a.take_str("--images").unwrap_or_else(|| {
        eprintln!("--images <dir|a.ppm,b.ppm,…> is required");
        std::process::exit(2);
    });
    let out_dir = crate::args::strip_out_name_prefix(&a.str_or("--out", "out/mirror"), "scene").to_string();
    let ply = a.take_str("--ply");
    let maps = a.take_flag("--maps");
    let min_op = a.f32_or("--min-opacity", 0.01);
    let max_depth = a.f32_or("--max-depth", 0.0);
    // Reference defaults. Fusion collapses the near-duplicate surfaces that
    // overlapping views each predict at slightly different depths, and without
    // it the scene is those duplicates stacked - which is what a viewer sees
    // as translucent superposition.
    let prune = a.f32_or("--prune", 0.002);
    let gs_mask = a.f32_or("--gs-mask-threshold", 0.5);
    let edge_rtol = a.f32_or("--edge-depth-threshold", 0.03);
    // Settle the frames' disagreement about where the surface is before any of
    // it becomes geometry. Pairs differing by more than this are an occlusion,
    // not a disagreement, and averaging them invents a surface in neither.
    let fuse_rtol = a.f32_or("--fuse-depth", 0.05);
    let scale_q = a.f32_or("--max-scale-quantile", 0.98);
    // The reference's inference default, independent of the checkpoint's
    // native grid. Roughly 3.4x the samples of 518 on a square image.
    let target = a.usize_or("--target-size", 952);
    // Known cameras, in the `cameras.json` shape `infer` writes. WorldMirror
    // is an any-prior model: supplying these fills trunk rows that are
    // otherwise zero.
    let poses = a.take_str("--poses");
    // Frame selection, for a capture longer than a handful of stills. A video
    // gets a default cap because the trunk's global attention is quadratic in
    // frame count; an explicit directory of photographs is left alone.
    let sel = FrameSel {
        stride: a.usize_or("--stride", 1),
        max: a.usize_or("--max-frames", if is_video(&images) { 48 } else { 0 }),
        fps: a.f32_or("--fps", 0.0) as f64,
    };
    let mask = a.take_str("--mask");
    // The model anchors the world to the FIRST frame - its c2w comes back as
    // the identity - so a scene written in that frame opens tipped by however
    // the camera happened to be held, 63 degrees on a real capture, with the
    // subject off to one side of a viewer's default view. That angle is an
    // artifact of which photograph came first, not information about the
    // scene, so land the result in the frame the cameras describe instead.
    let keep_frame = a.take_flag("--keep-camera-frame");

    a.finish();

    std::fs::create_dir_all(&out_dir).ok();
    let ply_path = ply.unwrap_or_else(|| format!("{out_dir}/scene.ply"));
    with_scene(&weights, &images, min_op, max_depth, prune, poses.as_deref(), gs_mask, edge_rtol, fuse_rtol, scale_q, &sel, mask.as_deref(), target, |gpu, model, splats, cams, s, w, h| {
        let reframed = (!keep_frame && cams.len() >= 3).then(|| upright(splats, cams));
        let (splats, cams) = match &reframed {
            Some((sp, cm)) => (sp, &cm[..]),
            None => (splats, cams),
        };
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
    let weights = a.str_or("--weights", "out/mirror.safetensors");
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
    // Reference defaults. Fusion collapses the near-duplicate surfaces that
    // overlapping views each predict at slightly different depths, and without
    // it the scene is those duplicates stacked - which is what a viewer sees
    // as translucent superposition.
    let prune = a.f32_or("--prune", 0.002);
    let gs_mask = a.f32_or("--gs-mask-threshold", 0.5);
    let edge_rtol = a.f32_or("--edge-depth-threshold", 0.03);
    let fuse_rtol = a.f32_or("--fuse-depth", 0.05);
    // Settle the frames' disagreement about where the surface is before any of
    // it becomes geometry. Pairs differing by more than this are an occlusion,
    // not a disagreement, and averaging them invents a surface in neither.
    let fuse_rtol = a.f32_or("--fuse-depth", 0.05);
    let scale_q = a.f32_or("--max-scale-quantile", 0.98);
    // The reference's inference default, independent of the checkpoint's
    // native grid. Roughly 3.4x the samples of 518 on a square image.
    let target = a.usize_or("--target-size", 952);
    // Known cameras, in the `cameras.json` shape `infer` writes. WorldMirror
    // is an any-prior model: supplying these fills trunk rows that are
    // otherwise zero.
    let poses = a.take_str("--poses");
    // Frame selection, for a capture longer than a handful of stills. A video
    // gets a default cap because the trunk's global attention is quadratic in
    // frame count; an explicit directory of photographs is left alone.
    let sel = FrameSel {
        stride: a.usize_or("--stride", 1),
        max: a.usize_or("--max-frames", if is_video(&images) { 48 } else { 0 }),
        fps: a.f32_or("--fps", 0.0) as f64,
    };
    let mask = a.take_str("--mask");

    a.finish();

    let (splats, init_cam) = with_scene(&weights, &images, min_op, max_depth, prune, poses.as_deref(), gs_mask, edge_rtol, fuse_rtol, scale_q, &sel, mask.as_deref(), target, |_gpu, _model, splats, cams, _s, _w, _h| {
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
        "brain worldmirror2 - WorldMirror-2",
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
    let out = a.str_or("--out", "out/mirror.safetensors");
    let src = a.positional().unwrap_or_else(|| {
        eprintln!("usage: brain worldmirror2 import <model.safetensors|hf_dir> --out mirror.safetensors");
        std::process::exit(2);
    });
    a.finish();

    let cfg = MirrorConfig::default();
    println!("importing {src} ({} tensors expected) …", cfg.param_list().len());
    match worldmirror2::import::convert(&src, &out, &cfg) {
        Ok(n) => println!("wrote {out}: {n} tensors, all consumed, shapes verified"),
        Err(e) => {
            eprintln!("import failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    /// `infer --out` names the DIRECTORY `scene.ply`/`cameras.json`/depth
    /// maps are written under, one per `reconstruct`'s declared output
    /// blobs - but the generic capability-manifest `name=path` form
    /// (documented by `brain caps worldmirror2`, and what `brain do`/D-Bus
    /// actually send) was taken literally, creating a directory literally
    /// named e.g. `scene=out/mirror` with no error. That site is now wired
    /// through `crate::args::strip_out_name_prefix` (shared, tested there)
    /// against `reconstruct`'s primary declared output blob name, `scene`;
    /// this just pins that the wiring did not regress.
    #[test]
    fn infer_out_accepts_the_documented_name_equals_path_form() {
        assert_eq!(crate::args::strip_out_name_prefix("scene=out/mirror", "scene"), "out/mirror");
    }

    /// A directory of photographs is the ordinary way to reach this model, and
    /// for a long time only `.ppm` counted - so a folder straight off a camera
    /// or a video extraction produced "no images found", and every caller
    /// converted by hand first. `imaging::load` decodes by sniffing the bytes,
    /// not the extension, so the narrow filter bought nothing.
    #[test]
    fn a_directory_of_ordinary_photographs_is_accepted() {
        let d = std::env::temp_dir().join(format!("brain-mirror-collect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for n in ["b.JPG", "a.jpeg", "d.ppm", "c.png", "notes.txt", "scene.ply"] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let got: Vec<String> = super::collect_images(d.to_str().unwrap())
            .iter()
            .map(|p| std::path::Path::new(p).file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        // sorted, so the order a capture wrote them in is the order the model
        // sees them - a sequence's frame order is its baseline structure.
        assert_eq!(got, vec!["a.jpeg", "b.JPG", "c.png", "d.ppm"], "wrong set or wrong order");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A mask is a keep/drop decision per pixel, so it is sampled, never
    /// interpolated - a resized mask that blends would create half-lit border
    /// pixels, and the model would dutifully reconstruct the blend.
    #[test]
    fn a_mask_blacks_out_what_it_does_not_cover_at_any_size() {
        // 2x2 mask: keep the left column, drop the right
        let mask = imaging::Rgb8::new(2, 2, vec![255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0]).unwrap();
        // a 4x4 frame of solid white
        let mut img = imaging::Rgb8::new(4, 4, vec![200u8; 4 * 4 * 3]).unwrap();
        super::apply_mask(&mut img, &mask);
        for y in 0..4 {
            for x in 0..4 {
                let v = img.px[(y * 4 + x) * 3];
                let want = if x < 2 { 200 } else { 0 };
                assert_eq!(v, want, "pixel ({x},{y}) is {v}, expected {want} after a 2x2 mask on a 4x4 frame");
            }
        }
    }

    /// Frame selection has to happen BEFORE the model sees the sequence: a
    /// 60-second orbit at 30fps is 1800 frames, and the trunk's global
    /// attention is quadratic in frame count.
    #[test]
    fn a_frame_sequence_can_be_subsampled_and_capped() {
        let frames: Vec<String> = (0..20).map(|i| format!("f{i:03}.png")).collect();
        assert_eq!(super::select_frames(&frames, 1, 0).len(), 20);
        assert_eq!(super::select_frames(&frames, 4, 0), vec!["f000.png", "f004.png", "f008.png", "f012.png", "f016.png"]);
        // a cap spreads its picks across the WHOLE sequence rather than taking
        // a prefix, or a 360-degree orbit would become a 60-degree one.
        let capped = super::select_frames(&frames, 1, 5);
        assert_eq!(capped.len(), 5);
        assert_eq!(capped.first().unwrap(), "f000.png");
        assert!(capped.last().unwrap().as_str() >= "f015.png", "cap took a prefix instead of spanning: {capped:?}");
        // stride and cap compose, and neither can produce an empty set
        assert_eq!(super::select_frames(&frames, 100, 0).len(), 1);
        assert_eq!(super::select_frames(&frames, 1, 99).len(), 20);
    }
}
