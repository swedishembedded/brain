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

/// Shared-prefill sampling: N sampled forecasts forking one prefill must be
/// BIT-IDENTICAL to N independent rollouts (seed i), and faster. Small + quick.
#[test]
#[ignore]
fn shared_prefill_parity_and_speed() {
    let (Some(tok), Some(dec)) = (env("BRAIN_KRONOS_TOKENIZER"), env("BRAIN_KRONOS_DECODER")) else {
        eprintln!("SKIP: set BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER");
        return;
    };
    let model = kronos::import::load_model(&tok, &dec).expect("load kronos");
    let feat = model.feat();
    let (t, h, nsamp) = (96usize, 5usize, 4usize);
    let bars = synth_bars(t, feat);
    let (ctx_stamp, fut_stamp) = (vec![0u32; t * 5], vec![0u32; h * 5]);
    let opts = kronos::generate::GenOpts { argmax: false, temperature: 1.0, top_k: 0, top_p: 1.0, seed: 42 };

    // Parity: shared[i] == independent rollout with seed 42+i, bit-for-bit.
    let shared = model.forecast_cached_samples(&bars, &ctx_stamp, &fut_stamp, h, nsamp, &opts);
    assert_eq!(shared.len(), nsamp);
    for (i, s) in shared.iter().enumerate() {
        let oi = kronos::generate::GenOpts { seed: 42 + i as u64, ..opts.clone() };
        let indep = model.forecast_cached(&bars, &ctx_stamp, &fut_stamp, h, &oi);
        assert_eq!(*s, indep, "shared-prefill sample {i} != independent rollout");
    }

    // Speed (small): N-sample shared vs N independent, min-of-3.
    let shared_ms = {
        let mut b = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = model.forecast_cached_samples(&bars, &ctx_stamp, &fut_stamp, h, nsamp, &opts);
            b = b.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        b
    };
    let indep_ms = {
        let mut b = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            for i in 0..nsamp {
                let oi = kronos::generate::GenOpts { seed: 42 + i as u64, ..opts.clone() };
                let _ = model.forecast_cached(&bars, &ctx_stamp, &fut_stamp, h, &oi);
            }
            b = b.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        b
    };
    eprintln!(
        "BENCH shared-prefill nsamp={nsamp} ctx={t} H={h}: shared {shared_ms:.1} ms vs independent {indep_ms:.1} ms = {:.2}x",
        indep_ms / shared_ms
    );
}

/// Cross-sectional batch: forecasting N names via the rayon-over-names batch must
/// be BIT-IDENTICAL to N serial `forecast_cached` calls, and much faster.
#[test]
#[ignore]
fn crosssection_batch_parity_and_speed() {
    let (Some(tok), Some(dec)) = (env("BRAIN_KRONOS_TOKENIZER"), env("BRAIN_KRONOS_DECODER")) else {
        eprintln!("SKIP: set BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER");
        return;
    };
    let model = kronos::import::load_model(&tok, &dec).expect("load kronos");
    let feat = model.feat();
    let (t, h, n) = (96usize, 5usize, 32usize);
    let opts = kronos::generate::GenOpts::default(); // argmax → deterministic
    let bars_list: Vec<Vec<f32>> = (0..n)
        .map(|k| {
            let mut b = synth_bars(t, feat);
            for v in b.iter_mut() {
                *v += k as f32 * 0.017; // distinct series per name
            }
            b
        })
        .collect();
    let ctx_stamps: Vec<Vec<u32>> = (0..n).map(|_| vec![0u32; t * 5]).collect();
    let fut_stamps: Vec<Vec<u32>> = (0..n).map(|_| vec![0u32; h * 5]).collect();

    let batch = model.forecast_cached_batch(&bars_list, &ctx_stamps, &fut_stamps, h, &opts);
    assert_eq!(batch.len(), n);
    for i in 0..n {
        let single = model.forecast_cached(&bars_list[i], &ctx_stamps[i], &fut_stamps[i], h, &opts);
        assert_eq!(batch[i], single, "crosssection batch[{i}] != serial forecast_cached");
    }

    let batch_ms = {
        let mut m = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = model.forecast_cached_batch(&bars_list, &ctx_stamps, &fut_stamps, h, &opts);
            m = m.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        m
    };
    let serial_ms = {
        let mut m = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            for i in 0..n {
                let _ = model.forecast_cached(&bars_list[i], &ctx_stamps[i], &fut_stamps[i], h, &opts);
            }
            m = m.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        m
    };
    eprintln!(
        "BENCH crosssection N={n} ctx={t} H={h}: batch(rayon) {batch_ms:.0} ms vs serial {serial_ms:.0} ms = {:.2}x",
        serial_ms / batch_ms
    );
}

/// Training-side throughput: one fine-tune step (forward+backward+adamw) at batch
/// sizes b=1,2,4,8, reporting ms/window so the batching win shows in real numbers.
/// Self-contained (random weights), so no env vars — but `#[ignore]`d to keep it
/// off the default test run:
///
///   BRAIN_DEVICE=cpu cargo test -p brain-kronos --release --test bench_cpu \
///     finetune_step_batch_scaling -- --ignored --nocapture
#[test]
#[ignore]
fn finetune_step_batch_scaling() {
    use kronos::config::KronosConfig;
    use kronos::train::{param_list_c, KronosTrain, TokenBatch, CAL};
    use std::collections::HashMap;

    // A mid-size decoder: big enough that the matmuls (not just per-submit
    // overhead) matter, small enough to bench in seconds.
    let cfg = KronosConfig {
        d_model: 128,
        n_layers: 4,
        n_heads: 8,
        ff_dim: 256,
        s1_bits: 6,
        s2_bits: 6,
        learn_te: true,
        dep_n_heads: 4,
        max_context: 128,
    };
    let t = 64u32;
    let d = cfg.d_model;

    // Deterministic small random weights (reference names: fusion is the fused
    // [d,2d] `fusion_proj.weight`, split inside the constructor).
    let mut seed = 0x1234_5678u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.06
            })
            .collect()
    };
    let mut init: HashMap<String, Vec<f32>> = param_list_c(&cfg)
        .into_iter()
        .filter(|(n, _)| n != "embedding.fusion_l" && n != "embedding.fusion_r")
        .map(|(n, _)| (n, Vec::new()))
        .collect();
    for (name, v) in init.iter_mut() {
        let numel = param_list_c(&cfg).into_iter().find(|(n, _)| n == name).map(|(_, s)| s).unwrap();
        let is_norm = name.ends_with("norm.weight") || name.ends_with("norm1.weight") || name.ends_with("norm2.weight");
        *v = if is_norm { vec![1.0; numel] } else { rnd(numel) };
    }
    init.insert("embedding.fusion_proj.weight".into(), rnd(d * 2 * d));

    let s1v = 1u32 << cfg.s1_bits;
    let s2v = 1u32 << cfg.s2_bits;
    let mk_window = |k: u32| -> TokenBatch {
        let g = |off: u32, card: u32| -> Vec<u32> { (0..t).map(|i| (i * 2 + off + k * 7) % card).collect() };
        TokenBatch {
            s1: g(0, s1v),
            s2: g(1, s2v),
            cal: std::array::from_fn(|c| g(c as u32 + 2, CAL[c].1 as u32)),
            sampled_s1: g(3, s1v),
            s1_targets: g(4, s1v),
            s2_targets: g(5, s2v),
        }
    };

    eprintln!("BENCH finetune-step batch scaling (d_model={d}, layers={}, t={t}):", cfg.n_layers);
    let mut base_per_window = 0.0f32;
    for &b in &[1u32, 2, 4, 8] {
        let m = KronosTrain::new_batch(cfg.clone(), t, b, &init);
        let windows: Vec<TokenBatch> = (0..b).map(mk_window).collect();
        m.set_many(&windows);
        // warm up (JIT + allocs), then min-of-N timed steps.
        for _ in 0..2 {
            m.zero_grads();
            let _ = m.forward();
            m.backward();
            m.adamw_step(1, 1e-4, 0.0, Some(3.0));
        }
        m.poll_wait();
        let mut best = f32::INFINITY;
        for s in 0..5u32 {
            let t0 = Instant::now();
            m.zero_grads();
            let _ = m.forward();
            m.backward();
            m.adamw_step(s + 2, 1e-4, 0.0, Some(3.0));
            m.poll_wait();
            best = best.min(t0.elapsed().as_secs_f32() * 1e3);
        }
        let per_window = best / b as f32;
        if b == 1 {
            base_per_window = per_window;
        }
        eprintln!(
            "  b={b}: {best:.1} ms/step  |  {per_window:.2} ms/window  |  {:.2}x throughput vs b=1",
            base_per_window / per_window
        );
    }
}
