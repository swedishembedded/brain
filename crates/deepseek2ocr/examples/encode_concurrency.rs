// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in keeping a
// GPU's vision encoders busy from several streams then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Whether encoding pages from more than one thread, each with its own tile encoder on the same card, raises the views a
//! second: `--threads K` builds K tile encoders, each thread runs `--views` forwards, and the aggregate rate is printed.
//!
//! ```text
//! cargo run --release -p brain-deepseek2ocr --example encode_concurrency -- --dir <model dir> --threads 2 [--views 40]
//! ```

use deepseek2ocr::encoder::DeepEncoder;
use deepseek2ocr::serve::TILE_SIDE;
use gpu_core::Gpu;
use sam1::resample::GridSource;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone()).unwrap_or_else(|| default.to_string());
    let dir = arg("--dir", "");
    let threads: usize = arg("--threads", "1").parse().expect("--threads");
    let views: usize = arg("--views", "40").parse().expect("--views");
    assert!(!dir.is_empty(), "--dir <model dir>");
    let files = deepseek2ocr::import::Files::locate(&dir).expect("locate");
    let cfg = deepseek2ocr::import::config(&files, 1).expect("config");
    let vision = deepseek2ocr::import::encoder_weights_for(&files, &cfg).expect("vision weights");
    let tile_grid = TILE_SIDE / cfg.sam.patch_size;
    let image: Vec<f32> = (0..3 * (TILE_SIDE * TILE_SIDE) as usize).map(|i| (((i * 2654435761usize) >> 9) % 2001) as f32 / 1000.0 - 1.0).collect();
    let barrier = std::sync::Barrier::new(threads + 1);
    let mut started = None;
    std::thread::scope(|s| {
        for _ in 0..threads {
            let (vision, cfg, image, barrier) = (&vision, &cfg, &image, &barrier);
            s.spawn(move || {
                let w = GridSource::new(vision, (cfg.sam.grid_h, cfg.sam.grid_w), (tile_grid, tile_grid), cfg.sam.d_model);
                let e: DeepEncoder = DeepEncoder::new(&|k: &'static [(&'static str, &'static str)]| Gpu::new(k), cfg.at_sam_grid(tile_grid, tile_grid), &w, 0, false);
                let _ = e.forward(image);
                barrier.wait();
                for _ in 0..views {
                    let _ = e.forward(image);
                }
            });
        }
        barrier.wait();
        started = Some(Instant::now());
    });
    let secs = started.expect("started").elapsed().as_secs_f64();
    println!("{{\"threads\":{threads},\"views_per_thread\":{views},\"views_per_s\":{:.1},\"backend\":\"{}\"}}", (threads * views) as f64 / secs, gpu_core::backend_name());
}
