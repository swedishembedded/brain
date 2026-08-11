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

/// What the demo shows. `v` cycles these live; `--view` picks the initial one.
/// Each view renders at its NATURAL size (side is 2w wide, the rest are w wide), so
/// the stereogram keeps the camera's aspect — the window resizes to match.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    /// RGB | colorized depth, side by side (2w × h).
    Side,
    /// Colorized depth alone (w × h).
    Depth,
    /// A random-dot Magic-Eye autostereogram of the depth (w × h).
    Stereo,
    /// A TEXTURED autostereogram: the camera image itself is the pattern, so you see
    /// the photo's textures with depth (w × h).
    StereoTex,
    /// A cross-eye stereo PAIR of the real image (left | right), offset by depth
    /// (2w × h). Free-view CROSS-EYED to see the actual scene in 3D.
    StereoDual,
    /// The camera image with depth fog: far objects fade into an eerie haze (w × h).
    Fog,
    /// The camera image with depth-of-field blur: far objects softer (w × h).
    Blur,
}

impl ViewMode {
    fn parse(s: &str) -> ViewMode {
        match s {
            "stereo" | "magiceye" | "magic" | "dots" => ViewMode::Stereo,
            "stereo-image" | "stereo-tex" | "textured" | "photo" => ViewMode::StereoTex,
            "stereo-dual" | "dual" | "crosseye" | "cross-eye" | "cross" => ViewMode::StereoDual,
            "fog" | "haze" => ViewMode::Fog,
            "blur" | "dof" => ViewMode::Blur,
            "depth" => ViewMode::Depth,
            _ => ViewMode::Side,
        }
    }
    fn cycle(self) -> ViewMode {
        // Grouped for demoing: plain image effects, then the depth map, then stereo.
        match self {
            ViewMode::Side => ViewMode::Fog,
            ViewMode::Fog => ViewMode::Blur,
            ViewMode::Blur => ViewMode::Depth,
            ViewMode::Depth => ViewMode::Stereo,
            ViewMode::Stereo => ViewMode::StereoTex,
            ViewMode::StereoTex => ViewMode::StereoDual,
            ViewMode::StereoDual => ViewMode::Side,
        }
    }
    fn label(self) -> &'static str {
        match self {
            ViewMode::Side => "SIDE",
            ViewMode::Depth => "DEPTH",
            ViewMode::Stereo => "STEREO",
            ViewMode::StereoTex => "STEREO-IMG",
            ViewMode::StereoDual => "STEREO-DUAL",
            ViewMode::Fog => "FOG",
            ViewMode::Blur => "BLUR",
        }
    }
    /// The window/canvas size this view renders at, for a `w × h` frame.
    fn canvas(self, w: u32, h: u32) -> (u32, u32) {
        match self {
            ViewMode::Side | ViewMode::StereoDual => (2 * w, h),
            _ => (w, h),
        }
    }
}

/// Render one view at its natural size (see [`ViewMode::canvas`]).
#[allow(clippy::too_many_arguments)]
fn render_view(
    rgb8: &[u8],
    depth: &[f32],
    w: u32,
    h: u32,
    mode: ViewMode,
    bounds: Bounds,
    colormap: Colormap,
    stereo: &depth::StereoOpts,
) -> Vec<u8> {
    match mode {
        ViewMode::Side => {
            let dcol = colorize(depth, bounds, colormap);
            composite_side_by_side(rgb8, w, h, &dcol, w, h).0
        }
        ViewMode::Depth => colorize(depth, bounds, colormap),
        ViewMode::Stereo => depth::autostereogram(depth, w, h, bounds, stereo),
        ViewMode::StereoTex => depth::autostereogram_textured(depth, w, h, bounds, stereo, rgb8),
        ViewMode::StereoDual => {
            // Peak disparity ~ 1/25 of the width — enough relief to fuse, small
            // enough that disocclusion holes stay tiny.
            let max_disp = (w / 25).clamp(8, 40);
            depth::stereo_pair(rgb8, depth, w, h, bounds, max_disp, stereo.near_is_high)
        }
        ViewMode::Fog => depth::fog(rgb8, depth, w, h, bounds, [210, 216, 226], 3.5, stereo.near_is_high),
        ViewMode::Blur => {
            let max_radius = (w / 40).clamp(3, 20);
            depth::depth_blur(rgb8, depth, w, h, bounds, max_radius, stereo.near_is_high)
        }
    }
}

pub fn run_depth(args: &[String]) {
    // --camera anywhere selects the live path; otherwise it's the image path.
    if args.iter().any(|a| a == "--camera") {
        run_camera(args);
        return;
    }
    match args.first().map(|s| s.as_str()) {
        Some("--image") | Some("image") => run_image(args),
        Some("calib") => run_calib(&args[1..]),
        Some("train") => run_train(&args[1..]),
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
  brain depth train --out <w.safetensors> [--steps N --batch B --lr X --size WxH
                     --seed S --wd X --weights <init.pth>]    # train / fine-tune
                    (synthetic RGB->depth pairs; loss printed per step)

OPTIONS:
  --image <path>       input image (binary PPM 'P6', or a detection-dataset dir)
  --weights <path>     ZipDepth .pth checkpoint (imported 1:1 by name)
  --variant base|npu   which checkpoint layout (default base = unfold upsampler)
  --view MODE          side (RGB|depth, default) | fog | blur | depth | stereo
                       (random-dot Magic-Eye) | stereo-image (textured Magic-Eye) |
                       stereo-dual (cross-eye L|R pair). fog/blur are depth effects
                       on the image; free-view stereo* straight-on, dual cross-eyed.
  --colormap turbo|gray|grayinv   initial colormap (default turbo, cycle with [ ])
                       In-window keys: [ ] colormap, v cycle view, Esc quit
  --scale <n>          window pixel scale (default 2)
  --stripes <n>        stereogram pattern repeats (default 5; fewer = wider slices)
  --infer engine|npu   engine = brain's CPU/GPU forward (default); npu = export to
                       ONNX and run on the Intel NPU (needs --variant npu)
  --headless           no window: write the composite PPM to --out and print a hash
  --out <path>         PPM output path (default out/depth.ppm)
  --bench <n>          after the first (cold) inference, run n more and report
                       per-frame ms (min/median/mean) — steady-state timing
  --input <n>          model input (shorter side; default = the checkpoint's
                       native 384). The net is fully convolutional, so smaller
                       inputs are valid and faster — work scales with n²
                       (--input 256 is ~2.3x quicker, mildly softer depth)
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
    /// Number of horizontal pattern repeats in the stereograms (default 5).
    stripes: u32,
    /// Warm re-inference count for steady-state timing (0 = off).
    bench: u32,
    /// Model input (shorter side, rounded to x32 by the predictor); 0 = the
    /// checkpoint's native 384. Smaller = faster: work scales with its square.
    input: u32,
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
        stripes: 5,
        bench: 0,
        input: 0,
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
            "--stripes" => o.stripes = next(&mut i).parse().unwrap_or(5),
            "--headless" => o.headless = true,
            "--out" => o.out = next(&mut i),
            "--bench" => o.bench = next(&mut i).parse().unwrap_or(0),
            "--input" => o.input = next(&mut i).parse().unwrap_or(0),
            // `--infer npu` runs the exported ONNX on the Intel NPU via OpenVINO;
            // the default runs brain's own engine (honouring the global --device).
            "--infer" => {
                // cpu/gpu are engine aliases; npu runs on the NPU.
                o.infer = match next(&mut i).as_str() {
                    "npu" => "npu".to_string(),
                    _ => "engine".to_string(),
                };
            }
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


/// Pick the ZipConfig by inspecting the checkpoint's own tensor names (see
/// [`depth::cfg_for_checkpoint`]); an unreadable file falls back to the base
/// variant so the strict importer reports the real error.
fn cfg_for_checkpoint(weights: &str) -> ZipConfig {
    depth::cfg_for_checkpoint(weights).unwrap_or_else(|_| ZipConfig::base())
}

fn run_image(args: &[String]) {
    let o = parse(args);
    let mut cfg = cfg_for_checkpoint(&o.weights);
    if o.input > 0 {
        // Fully convolutional: any x32 input works; the predictor rounds.
        cfg.input = o.input;
    }

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
        let depth = predictor.predict(&hwc, w, h);
        if o.bench > 0 {
            // The first predict above paid the one-time costs (model build, BN
            // packing, pipeline warm-up); these repeats measure the steady state
            // a camera stream sees.
            let mut ms: Vec<f32> = (0..o.bench)
                .map(|_| {
                    let t = std::time::Instant::now();
                    let _ = predictor.predict(&hwc, w, h);
                    t.elapsed().as_secs_f32() * 1000.0
                })
                .collect();
            ms.sort_by(f32::total_cmp);
            let mean = ms.iter().sum::<f32>() / ms.len() as f32;
            eprintln!(
                "bench: {} warm frames — min {:.1} ms, median {:.1} ms, mean {:.1} ms",
                ms.len(),
                ms[0],
                ms[ms.len() / 2],
                mean
            );
        }
        depth
    };
    let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
    eprintln!("depth: {w}x{h}, inference {infer_ms:.1} ms ({})", o.infer);

    let rgb8: Vec<u8> = hwc.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    let bounds = Bounds::from_percentiles(&depth, 0.02, 0.98);
    let mut colormap = o.colormap;
    let mut mode = ViewMode::parse(&o.view);
    // The stereogram's eye separation is sized to the frame width and stripe count,
    // so it keeps the camera aspect and shows `--stripes` repeats.
    let stereo = depth::StereoOpts::with_stripes(w, o.stripes);
    let render = |mode: ViewMode, map: Colormap| render_view(&rgb8, &depth, w, h, mode, bounds, map, &stereo);

    if o.headless {
        let (cw, ch) = mode.canvas(w, h);
        let canvas = render(mode, colormap);
        // `imaging::save_ppm` creates the parent directory itself.
        let img = imaging::Rgb8::new(cw, ch, canvas.clone()).expect("canvas is cw*ch*3");
        imaging::save_ppm(&o.out, &img).unwrap_or_else(|e| {
            eprintln!("brain depth: {e}");
            std::process::exit(1);
        });
        let hash = fnv1a(&canvas);
        println!("wrote {} ({cw}x{ch}) [{}]  rollout_hash={hash:016x}", o.out, mode.label());
        return;
    }

    // Windowed: [ / ] cycle colormaps, v cycles the view (resizing the window to the
    // view's natural aspect), Esc quits.
    let (mut cw, mut ch) = mode.canvas(w, h);
    let mut win = match wm_display::window::SdlWindow::new("brain depth", cw, ch, o.scale) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("brain depth: no display ({e}). Re-run with --headless for a PPM.");
            std::process::exit(1);
        }
    };
    let hud = Hud { model: format!("zipdepth {}", mode.label()), quality: colormap as u32, ..Default::default() };
    win.frame(&render(mode, colormap), cw, ch, &hud);
    loop {
        let input = win.pump();
        if input.quit {
            break;
        }
        let mut changed = false;
        for u in &input.ux {
            use wm_display::keymap::UxKey;
            match u {
                UxKey::QualityUp | UxKey::QualityDown => {
                    colormap = colormap.next();
                    changed = true;
                }
                UxKey::CycleView => {
                    mode = mode.cycle();
                    let (nw, nh) = mode.canvas(w, h);
                    if (nw, nh) != (cw, ch) {
                        (cw, ch) = (nw, nh);
                        win = wm_display::window::SdlWindow::new("brain depth", cw, ch, o.scale).expect("recreate window");
                    }
                    changed = true;
                }
                _ => {}
            }
        }
        if changed {
            let hud = Hud { model: format!("zipdepth {}", mode.label()), quality: colormap as u32, ..Default::default() };
            win.frame(&render(mode, colormap), cw, ch, &hud);
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
  --view MODE          side (default) | fog | blur | depth | stereo | stereo-image |
                       stereo-dual. `v` cycles them live.
  --input <n>          model input (shorter side, default 384). Smaller = faster,
                       quadratically: --input 256 is ~2.3x quicker per frame.
  --infer engine|npu   engine = brain CPU/GPU (default); npu = Intel NPU
In-window keys: v cycle view (side/depth/stereo), [ ] colormap, Esc quit.
Forces YUYV — an MJPEG-only camera is rejected (no JPEG decoder).
";

#[cfg(target_os = "linux")]
fn run_camera(args: &[String]) {
    use capture::{Device, Frame, FrameSlot};
    use imaging::color::yuyv_to_rgb;
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
    let mut infer = "engine".to_string();
    let mut stripes = 5u32;
    let mut input = 0u32;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || {
            i += 1;
            args.get(i).cloned().unwrap_or_default()
        };
        match a {
            "--camera" => {}
            "--infer" => infer = if val() == "npu" { "npu".to_string() } else { "engine".to_string() },
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
            "--stripes" => stripes = val().parse().unwrap_or(5),
            "--view" => view = val(),
            "--input" => input = val().parse().unwrap_or(0),
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
    let _ = &variant; // variant is auto-detected from the checkpoint now.
    let mut cfg = cfg_for_checkpoint(&weights);
    if input > 0 {
        // Fully convolutional: a smaller input trades depth sharpness for
        // frame rate quadratically (--input 256 ≈ 2.3x faster than 384).
        cfg.input = input;
    }

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
            let r = dev.next_frame(yuyv_to_rgb);
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

    // Build the inference backend once: brain's engine, or a compiled NPU session
    // sized for this camera's target (aspect-preserving, ×32).
    let use_npu = infer == "npu";
    let (cam_th, cam_tw) = depth::predict::target_size(cw, ch, cfg.input);
    let gpu = Gpu::new(depth::net::PIPELINES);
    let mut npu_sess = if use_npu { Some(build_npu_session(&weights, &cfg, cam_th, cam_tw)) } else { None };
    let predictor = if use_npu {
        None
    } else {
        let ps = import::load_into(&gpu, &weights, &cfg).unwrap_or_else(|e| {
            eprintln!("brain depth: loading {weights}: {e}");
            running.store(false, Ordering::Relaxed);
            std::process::exit(1);
        });
        Some(Predictor::new(&gpu, cfg.clone(), ps))
    };

    // Each view renders at its natural size; the window resizes to match when `v`
    // cycles. Sized from the actual frame dims (cw/ch = the camera's negotiated res).
    let stereo = depth::StereoOpts::with_stripes(cw, stripes);
    let mut mode = ViewMode::parse(&view);
    let (mut win_w, mut win_h) = mode.canvas(cw, ch);
    let mut win = match wm_display::window::SdlWindow::new("brain depth", win_w, win_h, scale) {
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
    // The one-frame pipeline: the frame whose inference is in flight (engine
    // path only), kept so its RGB renders against its own depth.
    let mut pipe_frame: Option<capture::Frame> = None;
    loop {
        let input = win.pump();
        if input.quit {
            break;
        }
        for u in &input.ux {
            use wm_display::keymap::UxKey;
            match u {
                UxKey::QualityUp | UxKey::QualityDown => colormap = colormap.next(),
                UxKey::CycleView => {
                    mode = mode.cycle();
                    let (nw, nh) = mode.canvas(cw, ch);
                    if (nw, nh) != (win_w, win_h) {
                        (win_w, win_h) = (nw, nh);
                        win = wm_display::window::SdlWindow::new("brain depth", win_w, win_h, scale).expect("recreate window");
                    }
                }
                _ => {}
            }
        }
        // Take the latest frame; a tick with none just idles.
        let Some(frame) = slot.take() else {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        };
        let hwc: Vec<f32> = frame.rgb.iter().map(|&b| b as f32 / 255.0).collect();
        let t0 = std::time::Instant::now();
        // Engine path is PIPELINED: start this frame on the device, and while
        // it computes, collect + render the PREVIOUS frame (its RGB is kept
        // alongside so the composite panes stay aligned). Effective throughput
        // becomes max(device, host) instead of their sum, at one frame of
        // display latency. The first iteration primes the pipe; the NPU path
        // stays synchronous (its session API blocks anyway).
        let (frame, depth) = match (&mut npu_sess, &predictor) {
            (Some(sess), _) => {
                let d = run_npu_session(sess, &hwc, frame.w, frame.h, cam_th, cam_tw);
                (frame, d)
            }
            (None, Some(p)) => {
                let prev_depth = if p.in_flight() { Some(p.finish()) } else { None };
                p.begin(&hwc, frame.w, frame.h);
                let prev = pipe_frame.replace(frame);
                match (prev, prev_depth) {
                    (Some(pf), Some(d)) => (pf, d),
                    _ => continue, // priming: nothing to show yet
                }
            }
            _ => unreachable!(),
        };
        let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // EMA the depth window so the colors do not breathe frame-to-frame.
        let target = Bounds::from_percentiles(&depth, 0.02, 0.98);
        bounds = Some(match bounds {
            Some(b) => b.ema(target, 0.1),
            None => target,
        });
        let mut canvas = render_view(&frame.rgb, &depth, frame.w, frame.h, mode, bounds.unwrap(), colormap, &stereo);
        let (ww, hh) = mode.canvas(frame.w, frame.h);

        let now = std::time::Instant::now();
        let dt = now.duration_since(last).as_secs_f32();
        last = now;
        fps = 0.9 * fps + 0.1 * (1.0 / dt.max(1e-3));
        let st = slot.stats();
        let backend = if use_npu { "NPU" } else { "ENGINE" };
        // In-frame HUD so it reads on the image, not just the window title.
        let line = format!("ZIPDEPTH {backend} {} {fps:.0}FPS {infer_ms:.0}MS DROP:{}", mode.label(), st.dropped);
        depth::viz::draw_text(&mut canvas, ww, hh, 6, 6, &line, 2, [0, 255, 0]);
        let hud = Hud {
            model: format!("zipdepth {backend} {}  {fps:.0} fps  {infer_ms:.0} ms  drop {}", mode.label(), st.dropped),
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
// ---------------------------------------------------------------------------
// Training path
// ---------------------------------------------------------------------------

/// `brain depth train` — the end-to-end loop on synthetic RGB->inverse-depth
/// pairs: forward -> masked L1 -> backward -> AdamW (see `depth::train`).
/// Placeholder-grade data, real loop; `--weights <ckpt.pth>` seeds from a
/// released checkpoint (fine-tune), otherwise fresh `init_weights`.
fn run_train(args: &[String]) {
    let mut steps = 50u32;
    let mut batch = 2u32;
    let (mut w, mut h) = (64u32, 64u32);
    let mut lr = 1e-3f32;
    let mut wd = 0.0f32;
    let mut seed = 7u64;
    let mut out = String::new();
    let mut weights = String::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || {
            i += 1;
            args.get(i).cloned().unwrap_or_default()
        };
        match a {
            "--steps" => steps = val().parse().unwrap_or(50),
            "--batch" => batch = val().parse().unwrap_or(2),
            "--lr" => lr = val().parse().unwrap_or(1e-3),
            "--wd" => wd = val().parse().unwrap_or(0.0),
            "--seed" => seed = val().parse().unwrap_or(7),
            "--out" => out = val(),
            "--weights" => weights = val(),
            "--size" => {
                let s = val();
                if let Some((a, b)) = s.split_once('x') {
                    w = a.parse().unwrap_or(64);
                    h = b.parse().unwrap_or(64);
                }
            }
            other => {
                eprintln!("brain depth train: unknown option '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if out.is_empty() {
        eprintln!("brain depth train: --out <file.safetensors> is required");
        std::process::exit(2);
    }
    if w % 32 != 0 || h % 32 != 0 {
        eprintln!("brain depth train: --size must be multiples of 32 (got {w}x{h})");
        std::process::exit(2);
    }

    // Fine-tuning a released checkpoint must build the MATCHING variant; fresh
    // training defaults to the base (unfold-upsampler) layout.
    let cfg = if weights.is_empty() { ZipConfig::base() } else { cfg_for_checkpoint(&weights) };
    let init = if weights.is_empty() {
        depth::init_weights(&cfg, seed)
    } else {
        match depth::load_checkpoint(&weights, &cfg) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("brain depth train: loading {weights}: {e}");
                std::process::exit(1);
            }
        }
    };

    let upsample_unfold = cfg.upsample_unfold; // captured before cfg moves into train_loop below
    let gpu = Gpu::new(depth::net::PIPELINES);
    let t = depth::train::TrainCfg { steps, batch, h, w, lr, wd, seed, fixed_batch: false };
    println!(
        "training zipdepth on synthetic pairs: {w}x{h} batch={batch} steps={steps} lr={lr} ({})",
        if weights.is_empty() { "fresh init" } else { &weights }
    );
    let (ps, res) = depth::train::train_loop(&gpu, cfg, &t, &init, |step, loss| {
        if step == 0 || (step + 1) % 10 == 0 || step + 1 == steps {
            println!("step {:>5}/{steps}  loss {loss:.4}", step + 1);
        }
    });

    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = ps
        .params
        .iter()
        .map(|(name, _)| {
            let n = ps.numel(name);
            (name.clone(), vec![n as u64], gpu.read(ps.w(name), n))
        })
        .collect();
    // "brain/depth" matches the reserved-vendor fallback id
    // -- the same id crates/cli/src/resident_depth.rs::DepthResident::from_env
    // synthesizes for an env-loaded checkpoint -- so a checkpoint saved here
    // is auto-discoverable by crates/cli/src/model_dir.rs without requiring
    // BRAIN_DEPTH_WEIGHTS to be set. The "variant" field is informational
    // only (DepthResident::activate auto-detects the real variant from the
    // checkpoint's own tensor shapes via depth::cfg_for_checkpoint, never
    // reads this back) -- previously hardcoded "base" regardless of which
    // cfg was actually used; derive it from cfg instead so the metadata is
    // at least honest about what got trained.
    let variant = if upsample_unfold { "base" } else { "npu" };
    checkpoint::save_carded(
        &out,
        serde_json::json!({ "model": "zipdepth", "variant": variant }),
        &tensors,
        &checkpoint::st::ModelCard::new("brain/depth", "depth"),
    );
    println!("done: loss {:.4} -> {:.4}; saved {out}", res.first_loss, res.last_loss);
}

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
        let val = |i: &mut usize| {
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
    let _ = &variant; // variant is auto-detected from the checkpoint now.
    let cfg = cfg_for_checkpoint(&weights);

    // Load calibration images (any PPM in the dir), preprocessed exactly as the
    // predictor does — aspect-preserving, no pad, so each carries its own size.
    let imgs = load_calib_images(&images_dir, cfg.input, max_n);
    if imgs.is_empty() {
        eprintln!("brain depth: no PPM images found under {images_dir}");
        std::process::exit(1);
    }
    {
        let mut shapes: Vec<(u32, u32)> = imgs.iter().map(|i| (i.h, i.w)).collect();
        shapes.sort_unstable();
        shapes.dedup();
        let shown: Vec<String> = shapes.iter().take(4).map(|(h, w)| format!("{h}x{w}")).collect();
        let more = if shapes.len() > 4 { format!(" (+{} more)", shapes.len() - 4) } else { String::new() };
        eprintln!("calibrating on {} images, {} distinct input shapes: {}{more}...", imgs.len(), shapes.len(), shown.join(", "));
    }

    let gpu = Gpu::new(depth::net::PIPELINES);
    let ps = import::load_into(&gpu, &weights, &cfg).unwrap_or_else(|e| {
        eprintln!("brain depth: loading {weights}: {e}");
        std::process::exit(1);
    });
    let stats = depth::collect_activation_stats_sized(&gpu, &cfg, &ps, &imgs);
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
/// Load calibration images and preprocess each one **exactly as the predictor
/// does**: `depth::predict::preprocess_chw` — aspect-preserving bilinear resize
/// so the shorter side is `input`, both dims rounded to a multiple of 32, no pad.
///
/// This used to letterbox to a padded square with a 0.5 grey fill, so
/// calibration fitted INT8 scales to a resampler, a geometry and a border the
/// model never sees at inference (and which `depth::predict`'s own module docs
/// record as having been REMOVED because it visibly degrades the depth). Each
/// image therefore now carries its own `(h, w)`.
fn load_calib_images(dir: &str, input: u32, max_n: usize) -> Vec<depth::CalibImage> {
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
        let (chw, th, tw) = depth::predict::preprocess_chw(&hwc, w, h, input);
        out.push(depth::CalibImage { chw, h: th, w: tw });
    }
    out
}


// ---------------------------------------------------------------------------
// NPU inference: export ZipDepth to ONNX, compile on the Intel NPU, run.
// ---------------------------------------------------------------------------

/// Export ZipDepth to ONNX at target size (th×tw) and compile it on the NPU.
fn build_npu_session(weights: &str, cfg: &ZipConfig, th: u32, tw: u32) -> npu::openvino::NpuSession {
    use npu::openvino::{NpuConfig, NpuDevice, NpuSession};
    assert!(!cfg.upsample_unfold, "--infer npu needs the blend (npu) checkpoint");
    let init = depth::import::load(weights, cfg).unwrap_or_else(|e| {
        eprintln!("brain depth: loading {weights}: {e}");
        std::process::exit(1);
    });
    let mut g = onnx::GraphBuilder::new("zipdepth");
    npu::build_depth_graph_hw(cfg, &init, th, tw, &mut g);
    let sess = NpuSession::load_bytes(&g.finish(), &NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() })
        .unwrap_or_else(|e| {
            eprintln!("brain depth: NPU compile failed: {e}");
            std::process::exit(1);
        });
    eprintln!("depth: compiled ZipDepth {tw}x{th} for {}", sess.device());
    sess
}

/// One-shot NPU predict for the image path: compile at the aspect-preserving target
/// size, run, resize the depth back to the frame grid.
fn predict_npu(weights: &str, cfg: &ZipConfig, hwc: &[f32], w0: u32, h0: u32) -> Vec<f32> {
    let (th, tw) = depth::predict::target_size(w0, h0, cfg.input);
    let mut sess = build_npu_session(weights, cfg, th, tw);
    run_npu_session(&mut sess, hwc, w0, h0, th, tw)
}

/// Run one frame on an NPU session compiled for (th×tw): aspect-preserving resize
/// to (th,tw), infer, resize the depth back to the frame grid. Matches the engine
/// predictor's preprocessing (which matches the reference).
fn run_npu_session(sess: &mut npu::openvino::NpuSession, hwc: &[f32], w0: u32, h0: u32, th: u32, tw: u32) -> Vec<f32> {
    let resized = imaging::resize_bilinear_hwc(hwc, 3, w0, h0, tw, th);
    let hw = (th * tw) as usize;
    let mut chw = vec![0f32; 3 * hw];
    for y in 0..th as usize {
        for x in 0..tw as usize {
            for c in 0..3 {
                chw[c * hw + y * tw as usize + x] = resized[(y * tw as usize + x) * 3 + c];
            }
        }
    }
    let out = sess.run(&chw, [1, 3, th as usize, tw as usize]).unwrap_or_else(|e| {
        eprintln!("brain depth: NPU inference failed: {e}");
        std::process::exit(1);
    });
    imaging::resize_bilinear_hwc(&out.tensors[0].2, 1, tw, th, w0, h0)
}
