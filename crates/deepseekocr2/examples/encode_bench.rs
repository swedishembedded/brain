// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in profiling
// vision encoders on accelerators then you can procure our services by
// sending an email to info@swedishembedded.com.

//! DeepSeek-OCR-2's encoder cost per view, without the server around it: SAM at the global (1024 px)
//! and the tile (768 px) grids, and the resampler over 1, 6 and 16 views of one kind, each the median of
//! `--runs` after one warm-up, as one JSON line.
//!
//! ```text
//! cargo run --release -p brain-deepseekocr2 --example encode_bench -- --dir <model dir> [--runs 5]
//! ```
//!
//! `BRAIN_PROFILE=1` adds the per-kernel table of each stage.

use deepseekocr2::encoder::{sam_tokens_from_nchw, Resampler};
use deepseekocr2::serve::TILE_SIDE;
use gpu_core::Gpu;
use sam1::resample::GridSource;
use sam1::SamEncoder;
use std::time::Instant;

/// The median of `runs` timings of `f`, after one warm-up.
fn median(runs: usize, mut f: impl FnMut() -> f64) -> f64 {
    let _ = f();
    let mut ms: Vec<f64> = (0..runs).map(|_| f()).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ms[ms.len() / 2]
}

/// One SAM forward of a `side` px view (median ms) and the token grid it produced.
fn time_sam(cfg: &deepseekocr2::config::DeepseekOcr2VisionConfig, sam: &SamEncoder, side: u32, n_query: u32, runs: usize) -> (f64, Vec<f32>) {
    let image: Vec<f32> = (0..3 * (side * side) as usize).map(|i| (((i * 2654435761usize) >> 9) % 2001) as f32 / 1000.0 - 1.0).collect();
    let mut tokens = Vec::new();
    let ms = median(runs, || {
        let t = Instant::now();
        sam.write_image(&image);
        sam.run();
        let nchw = sam.gpu.read(sam.output(), sam.out_len());
        tokens = sam_tokens_from_nchw(&nchw, cfg.sam.compress_out as usize, n_query as usize);
        t.elapsed().as_secs_f64() * 1e3
    });
    (ms, tokens)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone()).unwrap_or_else(|| default.to_string());
    let dir = arg("--dir", "");
    let runs: usize = arg("--runs", "5").parse().expect("--runs");
    assert!(!dir.is_empty(), "--dir <model dir>");

    let files = deepseekocr2::import::Files::locate(&dir).expect("locate");
    let (cfg, _) = deepseekocr2::import::configs(&files, 1).expect("configs");
    let vision = deepseekocr2::import::vision_reader(&files).expect("vision weights");
    let sam_global = SamEncoder::new_inference(Gpu::new(sam1::PIPELINES), cfg.sam.clone(), &vision, 0);
    let tile_grid = TILE_SIDE / cfg.sam.patch_size;
    let tile_weights = GridSource::new(&vision, (cfg.sam.grid_h, cfg.sam.grid_w), (tile_grid, tile_grid), cfg.sam.d_model);
    let sam_tile = SamEncoder::new_inference(Gpu::new(sam1::PIPELINES), cfg.sam.at_grid(tile_grid, tile_grid), &tile_weights, 0);
    let resampler = Resampler::new_on(Gpu::new(deepseekocr2::encoder::PIPELINES), cfg.clone(), &vision, false);

    let (sam_global_ms, global_tokens) = time_sam(&cfg, &sam_global, 1024, cfg.encoder.n_query_global, runs);
    let (sam_tile_ms, tile_tokens) = time_sam(&cfg, &sam_tile, TILE_SIDE, cfg.encoder.n_query_local, runs);
    let resample = |tokens: &[f32], local: bool, views: usize| {
        let refs: Vec<&[f32]> = (0..views).map(|_| tokens).collect();
        median(runs, || {
            let t = Instant::now();
            let _ = resampler.encode_views(&refs, local);
            t.elapsed().as_secs_f64() * 1e3
        })
    };
    // Under BRAIN_PROFILE, the per-kernel table of one pass over six tiles.
    let refs: Vec<&[f32]> = (0..6).map(|_| tile_tokens.as_slice()).collect();
    gpu_core::profile::device_table(resampler.gpu(), "resampler x6 tiles", || resampler.encode_views(&refs, true));
    println!(
        "{{\"sam_global_1024_ms\":{sam_global_ms:.1},\"sam_tile_{TILE_SIDE}_ms\":{sam_tile_ms:.1},\"resampler_global_x1_ms\":{:.1},\"resampler_tile_x1_ms\":{:.1},\"resampler_tile_x6_ms\":{:.1},\"resampler_tile_x16_ms\":{:.1},\"backend\":\"{}\"}}",
        resample(&global_tokens, false, 1),
        resample(&tile_tokens, true, 1),
        resample(&tile_tokens, true, 6),
        resample(&tile_tokens, true, 16),
        gpu_core::backend_name()
    );
}
