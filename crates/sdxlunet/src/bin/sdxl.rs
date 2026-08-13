// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `sdxl` — generate an image from a prompt with the SDXL stack.
//!
//! Usage:
//!   sdxl --root <sdxl-base-1.0> --prompt "..." [--out out.ppm]
//!        [--steps 30] [--guidance 5.0] [--seed 0] [--size 1024]
//!        [--negative "..."]

use sdxlunet::pipeline::{GenerateOptions, Sdxl};

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = arg(&a, "--root")
        .or_else(|| std::env::var("BRAIN_SDXL").ok())
        .unwrap_or_else(|| {
            eprintln!("sdxl: --root <checkpoint> (or BRAIN_SDXL) is required");
            std::process::exit(2);
        });
    let prompt = arg(&a, "--prompt").unwrap_or_else(|| {
        eprintln!("sdxl: --prompt \"...\" is required");
        std::process::exit(2);
    });
    let side: u32 = arg(&a, "--size").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let o = GenerateOptions {
        steps: arg(&a, "--steps").and_then(|s| s.parse().ok()).unwrap_or(30),
        guidance: arg(&a, "--guidance").and_then(|s| s.parse().ok()).unwrap_or(5.0),
        seed: arg(&a, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0),
        height: side,
        width: side,
        negative: arg(&a, "--negative").unwrap_or_default(),
    };
    let out = arg(&a, "--out").unwrap_or_else(|| "out/sdxl.ppm".into());

    eprintln!("sdxl: loading {root} at {side}x{side} ...");
    let t0 = std::time::Instant::now();
    let mut p = match Sdxl::load(&root, o.height, o.width) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("sdxl: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("sdxl: loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let t1 = std::time::Instant::now();
    let img = match p.generate(&prompt, &o) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sdxl: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "sdxl: {} steps in {:.1}s ({:.2}s/step)",
        o.steps,
        t1.elapsed().as_secs_f32(),
        t1.elapsed().as_secs_f32() / o.steps as f32
    );

    if let Some(d) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let px: Vec<u8> = img.iter().map(|v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8).collect();
    let rgb = imaging::Rgb8 { w: o.width, h: o.height, px };
    match imaging::save_ppm(&out, &rgb) {
        Ok(()) => eprintln!("sdxl: wrote {out}"),
        Err(e) => {
            eprintln!("sdxl: writing {out}: {e}");
            std::process::exit(1);
        }
    }
}
