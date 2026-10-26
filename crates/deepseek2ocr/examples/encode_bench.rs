// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in profiling
// vision encoders on accelerators then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The DeepEncoder's cost per view, without the server around it: build the global (1024 px) and
//! the tile (640 px) encoders from a checkpoint and time forward passes of each (the median of
//! `--runs`, after one warm-up), printing one JSON line.
//!
//! ```text
//! cargo run --release -p brain-deepseek2ocr --example encode_bench -- --dir <model dir> [--runs 5]
//! ```
//!
//! `BRAIN_PROFILE=1` adds the per-kernel table of each tower.

use deepseek2ocr::config::DeepseekOcrConfig;
use deepseek2ocr::encoder::DeepEncoder;
use deepseek2ocr::serve::TILE_SIDE;
use gpu_core::Gpu;
use sam1::resample::GridSource;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone()).unwrap_or_else(|| default.to_string());
    let dir = arg("--dir", "");
    let runs: usize = arg("--runs", "5").parse().expect("--runs");
    assert!(!dir.is_empty(), "--dir <model dir>");

    let files = deepseek2ocr::import::Files::locate(&dir).expect("locate");
    let cfg: DeepseekOcrConfig = deepseek2ocr::import::config(&files, 1).expect("config");
    let vision = deepseek2ocr::import::encoder_weights_for(&files, &cfg).expect("vision weights");
    let dev = |k: &'static [(&'static str, &'static str)]| Gpu::new(k);
    let global = DeepEncoder::new(&dev, cfg.clone(), &vision, 0, false);
    let tile_grid = TILE_SIDE / cfg.sam.patch_size;
    let tile_weights = GridSource::new(&vision, (cfg.sam.grid_h, cfg.sam.grid_w), (tile_grid, tile_grid), cfg.sam.d_model);
    let tile = DeepEncoder::new(&dev, cfg.at_sam_grid(tile_grid, tile_grid), &tile_weights, 0, false);

    let time = |enc: &DeepEncoder, side: u32| -> f64 {
        let image: Vec<f32> = (0..3 * (side * side) as usize).map(|i| (((i * 2654435761usize) >> 9) % 2001) as f32 / 1000.0 - 1.0).collect();
        let _ = enc.forward(&image);
        let mut ms: Vec<f64> = (0..runs)
            .map(|_| {
                let t = Instant::now();
                let _ = enc.forward(&image);
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ms[ms.len() / 2]
    };
    let (g, t) = (time(&global, 1024), time(&tile, TILE_SIDE));
    // Under BRAIN_PROFILE, one more pass of each with the device's per-kernel table (SAM's and CLIP's cards separately).
    for (label, enc, side) in [("global 1024", &global, 1024u32), ("tile", &tile, TILE_SIDE)] {
        let image = vec![0.1f32; 3 * (side * side) as usize];
        gpu_core::profile::device_table(&enc.sam().gpu, &format!("{label}: sam"), || gpu_core::profile::device_table(&enc.clip().gpu, &format!("{label}: clip"), || enc.forward(&image)));
    }
    println!("{{\"global_1024_ms\":{g:.1},\"tile_{TILE_SIDE}_ms\":{t:.1},\"backend\":\"{}\"}}", gpu_core::backend_name());
}
