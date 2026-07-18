// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain depth` — monocular depth inference and the depth demo.
//!
//! Today: the single-image path. `brain depth --image foo.ppm --weights foo.pth`
//! loads a ZipDepth checkpoint, runs it, and shows a side-by-side RGB | colorized
//! depth window (Esc quits, `[`/`]` cycle colormaps without re-inference). With
//! `--headless` it writes the composite as a PPM and prints a hash instead — the
//! smoke-test path that needs no display.
//!
//! `brain depth --camera` adds the realtime V4L2 webcam path (Linux, YUYV): a
//! capture thread fills a single-slot latest-frame buffer, the main loop takes the
//! latest frame, runs the same [`depth::Predictor`], EMA-smooths the depth window,
//! and shows it — Esc quits, `[`/`]` cycle colormaps live.

use depth::viz::{colorize, composite_side_by_side, Bounds, Colormap};
use depth::{import, Predictor, ZipConfig};
use gpu_core::Gpu;
use wm_display::sink::{FrameSink, Hud};

use crate::image_io;

pub fn run_depth(args: &[String]) {
    // --camera anywhere selects the live path; otherwise it's the image path.
    if args.iter().any(|a| a == "--camera") {
        run_camera(args);
        return;
    }
    match args.first().map(|s| s.as_str()) {
        Some("--image") | Some("image") => run_image(args),
        Some("calib") => run_calib(&args[1..]),
        Some("--help") | Some("-h") | None => print!("{HELP}"),
        Some(other) => {
            eprintln!("brain depth: unknown option '{other}'\n");
            print!("{HELP}");
            std::process::exit(2);
        }
    }
}

const HELP: &str = "\
brain depth — monocular depth (ZipDepth)

USAGE:
  brain depth --image <img>  --weights <ckpt.pth> [options]   # single image
  brain depth --camera       --weights <ckpt.pth> [options]   # realtime webcam

OPTIONS:
  --image <path>       input image (binary PPM 'P6', or a detection-dataset dir)
  --weights <path>     ZipDepth .pth checkpoint (imported 1:1 by name)
  --variant base|npu   which checkpoint layout (default base = unfold upsampler)
  --view side|depth    side-by-side RGB|depth (default) or depth only
  --colormap turbo|gray|grayinv   initial colormap (default turbo, cycle with [ ])
  --scale <n>          window pixel scale (default 2)
  --infer engine|npu   engine = brain's CPU/GPU forward (default); npu = export to
                       ONNX and run on the Intel NPU (needs --variant npu)
  --headless           no window: write the composite PPM to --out and print a hash
  --out <path>         PPM output path (default out/depth.ppm)
";

struct Opts {
    image: String,
    weights: String,
    variant: String,
    view: String,
    colormap: Colormap,
    scale: u32,
    headless: bool,
    out: String,
    /// "engine" (brain's own CPU/GPU forward) or "npu" (export -> OpenVINO NPU).
    infer: String,
}

fn parse(args: &[String]) -> Opts {
    let mut o = Opts {
        image: String::new(),
        weights: String::new(),
        variant: "base".into(),
        view: "side".into(),
        colormap: Colormap::Turbo,
        scale: 2,
        headless: false,
        out: "out/depth.ppm".into(),
        infer: "engine".into(),
    };
    let mut i = 0;
    let next = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| {
            eprintln!("brain depth: missing value after '{}'", args[*i - 1]);
            std::process::exit(2);
        })
    };
    while i < args.len() {
        match args[i].as_str() {
            "--image" | "image" => o.image = next(&mut i),
            "--weights" => o.weights = next(&mut i),
            "--variant" => o.variant = next(&mut i),
            "--view" => o.view = next(&mut i),
            "--colormap" => {
                o.colormap = match next(&mut i).as_str() {
                    "gray" => Colormap::Gray,
                    "grayinv" => Colormap::GrayInv,
                    _ => Colormap::Turbo,
                }
            }
            "--scale" => o.scale = next(&mut i).parse().unwrap_or(2),
            "--headless" => o.headless = true,
            "--out" => o.out = next(&mut i),
            // `--infer npu` runs the exported ONNX on the Intel NPU via OpenVINO;
            // the default runs brain's own engine (honouring the global --device).
            "--infer" => o.infer = next(&mut i),
            other => {
                eprintln!("brain depth: unknown option '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if o.image.is_empty() || o.weights.is_empty() {
        eprintln!("brain depth --image needs both --image and --weights\n");
        print!("{HELP}");
        std::process::exit(2);
    }
    o
}

fn run_image(args: &[String]) {
    let o = parse(args);
    let cfg = match o.variant.as_str() {
        "npu" => ZipConfig { upsample_unfold: false, ..ZipConfig::base() },
        _ => ZipConfig::base(),
    };

    let (hwc, w, h) = image_io::load_image(&o.image).unwrap_or_else(|e| {
        eprintln!("brain depth: {e}");
        std::process::exit(1);
    });

    // Gpu::new honours the process backend (`--device cpu|vulkan` / BRAIN_DEVICE),
    // which main.rs::select_backend already parsed out of argv — so the same demo
    // runs on the CPU JIT or a real GPU with no code change here.
    let gpu = Gpu::new(depth::net::PIPELINES);
    let ps = import::load_into(&gpu, &o.weights, &cfg).unwrap_or_else(|e| {
        eprintln!("brain depth: loading {}: {e}", o.weights);
        std::process::exit(1);
    });
    let t0 = std::time::Instant::now();
    let depth = if o.infer == "npu" {
        predict_npu(&o.weights, &cfg, &hwc, w, h)
    } else {
        let predictor = Predictor::new(&gpu, cfg.clone(), ps);
        predictor.predict(&hwc, w, h)
    };
    let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
    eprintln!("depth: {w}x{h}, inference {infer_ms:.1} ms ({})", o.infer);

    // Build the initial canvas. Colormap can change later without re-inference.
    let rgb8: Vec<u8> = hwc.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    let bounds = Bounds::from_percentiles(&depth, 0.02, 0.98);
    let mut colormap = o.colormap;
    let compose = |map: Colormap| -> (Vec<u8>, u32, u32) {
        let dcol = colorize(&depth, bounds, map);
        if o.view == "depth" {
            (dcol, w, h)
        } else {
            composite_side_by_side(&rgb8, w, h, &dcol, w, h)
        }
    };

    let (canvas, cw, ch) = compose(colormap);

    if o.headless {
        let dir = std::path::Path::new(&o.out).parent();
        if let Some(d) = dir {
            let _ = std::fs::create_dir_all(d);
        }
        let ppm = events::ppm::encode_p6(&canvas, cw, ch);
        std::fs::write(&o.out, &ppm).unwrap_or_else(|e| {
            eprintln!("brain depth: writing {}: {e}", o.out);
            std::process::exit(1);
        });
        let hash = fnv1a(&canvas);
        println!("wrote {} ({cw}x{ch})  rollout_hash={hash:016x}", o.out);
        return;
    }

    // Windowed: show the canvas, cycle colormaps on [ / ], quit on Esc.
    let mut win = match wm_display::window::SdlWindow::new("brain depth", cw, ch, o.scale) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("brain depth: no display ({e}). Re-run with --headless for a PPM.");
            std::process::exit(1);
        }
    };
    let hud = Hud { model: "zipdepth".into(), quality: colormap as u32, ..Default::default() };
    let mut current = canvas;
    win.frame(&current, cw, ch, &hud);
    loop {
        let input = win.pump();
        if input.quit {
            break;
        }
        let mut changed = false;
        for u in &input.ux {
            use wm_display::keymap::UxKey;
            if matches!(u, UxKey::QualityUp | UxKey::QualityDown) {
                colormap = colormap.next();
                changed = true;
            }
        }
        if changed {
            let (c, _, _) = compose(colormap);
            current = c;
            let hud = Hud { model: "zipdepth".into(), quality: colormap as u32, ..Default::default() };
            win.frame(&current, cw, ch, &hud);
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}

/// FNV-1a over the canvas bytes — a stable content hash for the smoke test.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Live camera path
// ---------------------------------------------------------------------------

const CAM_HELP: &str = "\
brain depth --camera — realtime webcam depth (Linux/V4L2, YUYV)

USAGE:
  brain depth --camera --weights <ckpt.pth> [options]

OPTIONS:
  --camera             use the webcam instead of an image
  --device-path <dev>  V4L2 device (default /dev/video0)
  --res WxH            requested capture size (default 640x480; driver may adjust)
  --weights <path>     ZipDepth .pth checkpoint
  --variant base|npu   checkpoint layout (default base)
  --colormap turbo|gray|grayinv   initial colormap (cycle with [ ])
  --scale <n>          window pixel scale (default 1)
  --view side|depth    side-by-side (default) or depth only
Esc quits. Forces YUYV — an MJPEG-only camera is rejected (no JPEG decoder).
";

#[cfg(target_os = "linux")]
fn run_camera(args: &[String]) {
    use capture::{yuyv_to_rgb, Device, Frame, FrameSlot};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // Parse the camera-specific options (reuse the image parser's fields loosely).
    let mut weights = String::new();
    let mut variant = "base".to_string();
    let mut dev_path = "/dev/video0".to_string();
    let (mut req_w, mut req_h) = (640u32, 480u32);
    let mut colormap = Colormap::Turbo;
    let mut scale = 1u32;
    let mut view = "side".to_string();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || {
            i += 1;
            args.get(i).cloned().unwrap_or_default()
        };
        match a {
            "--camera" => {}
            "--weights" => weights = val(),
            "--variant" => variant = val(),
            "--device-path" => dev_path = val(),
            "--res" => {
                let s = val();
                if let Some((w, h)) = s.split_once('x') {
                    req_w = w.parse().unwrap_or(640);
                    req_h = h.parse().unwrap_or(480);
                }
            }
            "--colormap" => {
                colormap = match val().as_str() {
                    "gray" => Colormap::Gray,
                    "grayinv" => Colormap::GrayInv,
                    _ => Colormap::Turbo,
                }
            }
            "--scale" => scale = val().parse().unwrap_or(1),
            "--view" => view = val(),
            "--help" | "-h" => {
                print!("{CAM_HELP}");
                return;
            }
            _ => {}
        }
        i += 1;
    }
    if weights.is_empty() {
        eprintln!("brain depth --camera needs --weights\n{CAM_HELP}");
        std::process::exit(2);
    }
    let cfg = match variant.as_str() {
        "npu" => ZipConfig { upsample_unfold: false, ..ZipConfig::base() },
        _ => ZipConfig::base(),
    };

    // Open the camera and negotiate YUYV. The driver reports the size it accepted.
    let mut dev = Device::open(&dev_path, req_w, req_h, 4).unwrap_or_else(|e| {
        eprintln!("brain depth: opening {dev_path}: {e}");
        eprintln!("(the camera must expose YUYV; many UVC cams are MJPEG-only.)");
        std::process::exit(1);
    });
    let (cw, ch) = (dev.width, dev.height);
    eprintln!("camera {dev_path}: {cw}x{ch} YUYV");

    // Capture thread: block in DQBUF, convert to RGB, overwrite the slot.
    let slot = FrameSlot::new();
    let producer = slot.clone();
    let running = Arc::new(AtomicBool::new(true));
    let stop = running.clone();
    let cap_thread = std::thread::spawn(move || {
        let mut seq = 0u64;
        while stop.load(Ordering::Relaxed) {
            let r = dev.next_frame(|yuyv, w, h| yuyv_to_rgb(yuyv, w, h));
            match r {
                Ok(rgb) => {
                    seq += 1;
                    producer.push(Frame { rgb, w: cw, h: ch, seq });
                }
                Err(e) => {
                    eprintln!("capture error: {e}");
                    break;
                }
            }
        }
    });

    // Load the model once.
    let gpu = Gpu::new(depth::net::PIPELINES);
    let ps = import::load_into(&gpu, &weights, &cfg).unwrap_or_else(|e| {
        eprintln!("brain depth: loading {weights}: {e}");
        running.store(false, Ordering::Relaxed);
        std::process::exit(1);
    });
    let predictor = Predictor::new(&gpu, cfg, ps);

    // Window sized for the composite (side) or the frame (depth).
    let out_w = if view == "depth" { cw } else { cw * 2 };
    let mut win = match wm_display::window::SdlWindow::new("brain depth", out_w, ch, scale) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("brain depth: no display ({e})");
            running.store(false, Ordering::Relaxed);
            let _ = cap_thread.join();
            std::process::exit(1);
        }
    };

    let mut bounds: Option<Bounds> = None;
    let mut fps = 0.0f32;
    let mut last = std::time::Instant::now();
    loop {
        let input = win.pump();
        if input.quit {
            break;
        }
        for u in &input.ux {
            use wm_display::keymap::UxKey;
            if matches!(u, UxKey::QualityUp | UxKey::QualityDown) {
                colormap = colormap.next();
            }
        }
        // Take the latest frame; a tick with none just idles.
        let Some(frame) = slot.take() else {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        };
        let hwc: Vec<f32> = frame.rgb.iter().map(|&b| b as f32 / 255.0).collect();
        let t0 = std::time::Instant::now();
        let depth = predictor.predict(&hwc, frame.w, frame.h);
        let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // EMA the depth window so the colors do not breathe frame-to-frame.
        let target = Bounds::from_percentiles(&depth, 0.02, 0.98);
        bounds = Some(match bounds {
            Some(b) => b.ema(target, 0.1),
            None => target,
        });
        let dcol = colorize(&depth, bounds.unwrap(), colormap);
        let (canvas, ww, hh) = if view == "depth" {
            (dcol, frame.w, frame.h)
        } else {
            composite_side_by_side(&frame.rgb, frame.w, frame.h, &dcol, frame.w, frame.h)
        };

        let now = std::time::Instant::now();
        let dt = now.duration_since(last).as_secs_f32();
        last = now;
        fps = 0.9 * fps + 0.1 * (1.0 / dt.max(1e-3));
        let st = slot.stats();
        let hud = Hud {
            model: format!("zipdepth  {fps:.0} fps  {infer_ms:.0} ms  drop {}", st.dropped),
            fps,
            quality: colormap as u32,
            ..Default::default()
        };
        win.frame(&canvas, ww, hh, &hud);
    }
    running.store(false, Ordering::Relaxed);
    let _ = cap_thread.join();
}

#[cfg(not(target_os = "linux"))]
fn run_camera(_args: &[String]) {
    eprintln!("brain depth --camera is Linux/V4L2 only");
    std::process::exit(1);
}

// ---------------------------------------------------------------------------
// `brain depth calib --report` — per-layer activation outlier ratios (the INT8
// decision data, measured with NO NPU).
// ---------------------------------------------------------------------------

fn run_calib(args: &[String]) {
    let mut weights = String::new();
    let mut images_dir = String::new();
    let mut variant = "base".to_string();
    let mut max_n = 100usize;
    let mut top = 20usize;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].clone();
        let mut val = |i: &mut usize| {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match a.as_str() {
            "--report" => {}
            "--weights" => weights = val(&mut i),
            "--images" => images_dir = val(&mut i),
            "--variant" => variant = val(&mut i),
            "--max" => max_n = val(&mut i).parse().unwrap_or(100),
            "--top" => top = val(&mut i).parse().unwrap_or(20),
            _ => {}
        }
        i += 1;
    }
    if weights.is_empty() || images_dir.is_empty() {
        eprintln!("usage: brain depth calib --report --weights <pth> --images <dir-of-ppm> [--max N] [--variant base|npu]");
        std::process::exit(2);
    }
    let cfg = match variant.as_str() {
        "npu" => ZipConfig { upsample_unfold: false, ..ZipConfig::base() },
        _ => ZipConfig::base(),
    };

    // Load calibration images (any PPM in the dir), letterbox each to the model
    // input as CHW.
    let imgs = load_calib_chw(&images_dir, cfg.input, max_n);
    if imgs.is_empty() {
        eprintln!("brain depth: no PPM images found under {images_dir}");
        std::process::exit(1);
    }
    eprintln!("calibrating on {} images at {}x{}...", imgs.len(), cfg.input, cfg.input);

    let gpu = Gpu::new(depth::net::PIPELINES);
    let ps = import::load_into(&gpu, &weights, &cfg).unwrap_or_else(|e| {
        eprintln!("brain depth: loading {weights}: {e}");
        std::process::exit(1);
    });
    let stats = depth::collect_activation_stats(&gpu, &cfg, &ps, &imgs);
    let report = stats.report();

    // Encoder vs decoder summary — the QuartDepth question.
    let mean = |it: &[&depth::LayerReport]| -> f32 {
        if it.is_empty() {
            0.0
        } else {
            it.iter().map(|r| r.outlier_ratio).sum::<f32>() / it.len() as f32
        }
    };
    let enc: Vec<&depth::LayerReport> = report.iter().filter(|r| r.is_encoder()).collect();
    let dec: Vec<&depth::LayerReport> = report.iter().filter(|r| !r.is_encoder()).collect();

    println!("\nper-layer activation outlier_ratio = absmax / p99.99 (higher = more INT8-hostile)\n");
    println!("{:<48} {:>10} {:>10} {:>8}", "layer", "absmax", "p99.99", "ratio");
    for r in report.iter().take(top) {
        println!("{:<48} {:>10.4} {:>10.4} {:>8.2}", r.name, r.absmax, r.p9999, r.outlier_ratio);
    }
    println!("\nENCODER mean outlier_ratio = {:.2}  ({} layers)", mean(&enc), enc.len());
    println!("DECODER mean outlier_ratio = {:.2}  ({} layers)", mean(&dec), dec.len());
    let verdict = if mean(&dec) > 1.5 * mean(&enc).max(1e-3) {
        "decoder tail DOMINATES -> QuartDepth holds; an FP decoder is likely worth it"
    } else {
        "decoder and encoder tails comparable -> a uniform INT8 policy may suffice"
    };
    println!("VERDICT: {verdict}");
}

/// Load up to `max_n` PPM images from `dir`, letterboxed to `[3,size,size]` CHW.
fn load_calib_chw(dir: &str, size: u32, max_n: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut paths: Vec<std::path::PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for p in paths {
        if out.len() >= max_n {
            break;
        }
        let Ok(bytes) = std::fs::read(&p) else { continue };
        if !bytes.starts_with(b"P6") {
            continue;
        }
        let Ok((px, w, h)) = events::ppm::decode_p6(&bytes) else { continue };
        let hwc: Vec<f32> = px.iter().map(|&b| b as f32 / 255.0).collect();
        out.push(letterbox_chw(&hwc, w, h, size));
    }
    out
}

/// Aspect-preserving letterbox of an HWC `[0,1]` image into a `[3,size,size]` CHW
/// tensor, grey pad — the same transform the predictor uses at inference.
fn letterbox_chw(hwc: &[f32], w0: u32, h0: u32, size: u32) -> Vec<f32> {
    let scale = (size as f32 / w0 as f32).min(size as f32 / h0 as f32);
    let new_w = (w0 as f32 * scale).round() as usize;
    let new_h = (h0 as f32 * scale).round() as usize;
    let pad_x = ((size as f32 - new_w as f32) * 0.5) as usize;
    let pad_y = ((size as f32 - new_h as f32) * 0.5) as usize;
    let sz = size as usize;
    let inv = 1.0 / scale;
    let mut chw = vec![0.5f32; 3 * sz * sz];
    for yi in 0..new_h {
        let sy = (((yi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, h0 as f32 - 1.0)) as usize;
        for xi in 0..new_w {
            let sx = (((xi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, w0 as f32 - 1.0)) as usize;
            let sbase = (sy * w0 as usize + sx) * 3;
            for c in 0..3 {
                chw[c * sz * sz + (yi + pad_y) * sz + (xi + pad_x)] = hwc[sbase + c];
            }
        }
    }
    chw
}

// ---------------------------------------------------------------------------
// NPU inference: export ZipDepth to ONNX, compile on the Intel NPU, run.
// ---------------------------------------------------------------------------

/// Predict depth on the NPU: letterbox the frame, run the exported ONNX via
/// OpenVINO, unwarp back to the frame grid. Mirrors depth::Predictor's geometry.
fn predict_npu(weights: &str, cfg: &ZipConfig, hwc: &[f32], w0: u32, h0: u32) -> Vec<f32> {
    use npu::openvino::{NpuConfig, NpuDevice, NpuSession};
    assert!(!cfg.upsample_unfold, "--infer npu needs --variant npu (the blend upsampler)");
    let init = depth::import::load(weights, cfg).unwrap_or_else(|e| {
        eprintln!("brain depth: loading {weights}: {e}");
        std::process::exit(1);
    });
    let mut g = onnx::GraphBuilder::new("zipdepth");
    npu::build_depth_graph(cfg, &init, &mut g);
    let mut sess = NpuSession::load_bytes(&g.finish(), &NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() })
        .unwrap_or_else(|e| {
            eprintln!("brain depth: NPU compile failed: {e}");
            std::process::exit(1);
        });
    eprintln!("depth: running on {}", sess.device());

    // Letterbox into a [3,sz,sz] CHW input (grey pad), matching the predictor.
    let sz = cfg.input;
    let scale = (sz as f32 / w0 as f32).min(sz as f32 / h0 as f32);
    let new_w = (w0 as f32 * scale).round() as usize;
    let new_h = (h0 as f32 * scale).round() as usize;
    let pad_x = ((sz as f32 - new_w as f32) * 0.5) as usize;
    let pad_y = ((sz as f32 - new_h as f32) * 0.5) as usize;
    let szu = sz as usize;
    let inv = 1.0 / scale;
    let mut chw = vec![0.5f32; 3 * szu * szu];
    for yi in 0..new_h {
        let sy = (((yi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, h0 as f32 - 1.0)) as usize;
        for xi in 0..new_w {
            let sx = (((xi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, w0 as f32 - 1.0)) as usize;
            let sbase = (sy * w0 as usize + sx) * 3;
            for c in 0..3 {
                chw[c * szu * szu + (yi + pad_y) * szu + (xi + pad_x)] = hwc[sbase + c];
            }
        }
    }
    let out = sess.run(&chw, [1, 3, szu, szu]).unwrap_or_else(|e| {
        eprintln!("brain depth: NPU inference failed: {e}");
        std::process::exit(1);
    });
    let depth_sq = &out.tensors[0].2;

    // Unwarp the sz×sz depth onto the frame grid (bilinear), same as the predictor.
    let sample = |fx: f32, fy: f32| -> f32 {
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let cx = |v: f32| (v as i32).clamp(0, sz as i32 - 1) as usize;
        let cy = |v: f32| (v as i32).clamp(0, sz as i32 - 1) as usize;
        let at = |xx: usize, yy: usize| depth_sq[yy * szu + xx];
        let (ax, bx) = (cx(x0), cx(x0 + 1.0));
        let (ay, by) = (cy(y0), cy(y0 + 1.0));
        let top = at(ax, ay) * (1.0 - tx) + at(bx, ay) * tx;
        let bot = at(ax, by) * (1.0 - tx) + at(bx, by) * tx;
        top * (1.0 - ty) + bot * ty
    };
    let mut depth = vec![0f32; (w0 * h0) as usize];
    for y in 0..h0 {
        let fy = (y as f32 + 0.5) * scale + pad_y as f32 - 0.5;
        for x in 0..w0 {
            let fx = (x as f32 + 0.5) * scale + pad_x as f32 - 0.5;
            depth[(y * w0 + x) as usize] = sample(fx, fy);
        }
    }
    depth
}
