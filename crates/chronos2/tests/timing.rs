// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end `forecast_quantiles` timing for Chronos-2 (excludes model load).
//! Env-gated on `CHRONOS2_WEIGHTS`; device chosen by `BRAIN_DEVICE`. Context 512
//! (→ 32 patches +REG +2 out = S=35, matching the exported NPU core graph),
//! horizon 24.

use chronos2::Chronos2;
use std::time::Instant;

#[test]
fn forecast_timing() {
    let Ok(weights) = std::env::var("CHRONOS2_WEIGHTS") else {
        eprintln!("CHRONOS2_WEIGHTS unset; skipping timing");
        return;
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let ctx: usize = std::env::var("CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let horizon: usize = std::env::var("H").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let dev = std::env::var("BRAIN_DEVICE").unwrap_or_else(|_| "gpu".into());

    let model = Chronos2::load_on(gpu_core::testgpu::dev(chronos2::model::PIPELINES), &weights).expect("load chronos2.weights");
    let context: Vec<f32> = (0..ctx).map(|i| 100.0 + 10.0 * (i as f32 * 0.03).sin()).collect();

    let _ = model.forecast_quantiles(&context, horizon); // warm
    let n = 3;
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = model.forecast_quantiles(&context, horizon);
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    eprintln!("chronos2 forecast ctx={ctx} h={horizon} [{dev}]: {:.3} s/forecast ({:.1} ms)", per, per * 1e3);
}
