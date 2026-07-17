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
//! The camera path (`--camera`) lands with `crates/capture`; this file is the CPU/
//! GPU image path that the camera loop will reuse.

use depth::viz::{colorize, composite_side_by_side, Bounds, Colormap};
use depth::{import, Predictor, ZipConfig};
use gpu_core::Gpu;
use wm_display::sink::{FrameSink, Hud};

use crate::image_io;

pub fn run_depth(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("--image") | Some("image") => run_image(args),
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
  brain depth --image <img> --weights <ckpt.pth> [options]

OPTIONS:
  --image <path>       input image (binary PPM 'P6', or a detection-dataset dir)
  --weights <path>     ZipDepth .pth checkpoint (imported 1:1 by name)
  --variant base|npu   which checkpoint layout (default base = unfold upsampler)
  --view side|depth    side-by-side RGB|depth (default) or depth only
  --colormap turbo|gray|grayinv   initial colormap (default turbo, cycle with [ ])
  --scale <n>          window pixel scale (default 2)
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
    let predictor = Predictor::new(&gpu, cfg, ps);

    let t0 = std::time::Instant::now();
    let depth = predictor.predict(&hwc, w, h);
    let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
    eprintln!("depth: {w}x{h}, inference {infer_ms:.1} ms");

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
