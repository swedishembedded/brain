// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CPU micro-benchmark for the host KV-cache forecast — the production inference
//! path (`forecast_cached`). Gated on `BRAIN_KRONOS_TOKENIZER` +
//! `BRAIN_KRONOS_DECODER`; `#[ignore]`d so it only runs when asked:
//!
//!   BRAIN_DEVICE=cpu BRAIN_KRONOS_TOKENIZER=… BRAIN_KRONOS_DECODER=… \
//!     cargo test -p brain-kronos --release --test bench_cpu -- --ignored --nocapture
//!
//! Prints ms/forecast and ms/token so matvec/AVX optimizations are measurable.
use std::time::Instant;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

fn synth_bars(t: usize, feat: usize) -> Vec<f32> {
    let mut bars = Vec::with_capacity(t * feat);
    for i in 0..t {
        let p = 100.0 + (i as f32 * 0.1).sin() * 5.0;
        let ohlcv = [p, p * 1.005, p * 0.995, p, 1000.0 + i as f32];
        for f in 0..feat {
            bars.push(if f < 5 { ohlcv[f] } else { 1.0 });
        }
    }
    bars
}

#[test]
#[ignore]
fn bench_forecast_cached() {
    let (Some(tok), Some(dec)) = (env("BRAIN_KRONOS_TOKENIZER"), env("BRAIN_KRONOS_DECODER")) else {
        eprintln!("SKIP: set BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER");
        return;
    };
    let model = kronos::import::load_model(&tok, &dec).expect("load kronos");
    let feat = model.feat();
    let (t, h) = (120usize, 5usize);
    let bars = synth_bars(t, feat);
    let ctx_stamp = vec![0u32; t * 5];
    let fut_stamp = vec![0u32; h * 5];
    let opts = kronos::generate::GenOpts::default(); // argmax → deterministic

    // warmup (page in weights, JIT any lazy paths)
    let warm = model.forecast_cached(&bars, &ctx_stamp, &fut_stamp, h, &opts);
    assert_eq!(warm.len(), h * feat);

    // Min-of-N: the minimum is the least-contended sample, robust to the box's
    // background load (mean/two-point derivations went negative under contention).
    let time = |h: usize, iters: usize| -> f64 {
        let fut = vec![0u32; h * 5];
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = model.forecast_cached(&bars, &ctx_stamp, &fut, h, &opts);
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        best
    };

    // Wide H spread so the decode slope is robust: decode_step = (t_H21 − t_H1)/20.
    let ms_h5 = time(h, 6);
    let ms_h1 = time(1, 6);
    let ms_h21 = time(21, 4);
    let decode_step = (ms_h21 - ms_h1) / 20.0;
    let prefill = ms_h1 - decode_step;
    eprintln!(
        "BENCH kronos forecast_cached ctx={t} feat={feat} (min-of-N): \
         H={h} {ms_h5:.1} ms/forecast | prefill(T={t}) {prefill:.1} ms ({:.3} ms/ctx-token) | \
         decode {decode_step:.2} ms/step | prefill {:.0}% of an H={h} forecast",
        prefill / t as f64,
        100.0 * prefill / ms_h5
    );
    // Determinism check for the parity harness: argmax must be stable.
    let last = model.forecast_cached(&bars, &ctx_stamp, &fut_stamp, h, &opts);
    assert_eq!(last, warm, "argmax forecast must be deterministic across runs");
}
