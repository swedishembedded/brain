// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What one LoRA training step costs on the device, at a real klein
//! configuration and a real training resolution - the number a caller has to
//! budget a run against, and the one an optimisation pass is measured from.
//!
//! Ignored by default (it wants a whole card to itself and several minutes).
//! Re-measure with:
//!
//! ```text
//! BRAIN_DEV_GPU=1 BRAIN_GPU_INDEX=<i> cargo test -p brain-flux2 --release \
//!     --test dev_step_time -- --ignored --nocapture
//! ```
//!
//! Knobs: `BRAIN_FLUX2_TRAIN_VARIANT` (klein-4b | klein-9b, default klein-4b),
//! `BRAIN_FLUX2_TRAIN_SIZE` (square px, multiple of 16, default 512),
//! `BRAIN_FLUX2_TRAIN_RANK` (default 16), `BRAIN_FLUX2_TRAIN_ITERS` (timed
//! steps after the warm-up, default 3).
//!
//! Method: the first step is a **warm-up** and never enters the statistics
//! (it pays pipeline creation and first-touch page faults); best-of-N is
//! reported alongside the mean, and N is printed. Nothing samples `nvidia-smi`
//! while the clock is running - polling it has been measured to inflate a
//! FLUX.2 forward on this hardware by a large multiple.
//!
//! The weights are a cheap constant fill, not a checkpoint: a step's cost is
//! set by its shapes and its dispatch sequence, and this test is about the
//! shapes. Correctness lives in `dev_grad.rs` / `device_train.rs`.

use flux2::devtrain::DeviceTrainer;
use flux2::grad::{DoubleW, SingleW, StreamW};
use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{make_flow_batch, Cfg, ModelWeights};
use flux2::Flux2Config;

fn envs(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}
fn envn(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// A constant fill of the right shape. Values do not affect the clock; the
/// alternative (a per-element RNG over billions of parameters) would dominate
/// the measurement's own setup.
fn fill(n: usize, v: f32) -> Vec<f32> {
    vec![v; n]
}

fn weights(c: &Cfg) -> ModelWeights<f32> {
    let (d, hd, mlp, cin) = (c.hidden, c.head_dim(), c.mlp, c.in_channels);
    let stream = || StreamW {
        wq: fill(d * d, 0.01),
        wk: fill(d * d, 0.011),
        wv: fill(d * d, 0.012),
        nq: fill(hd, 1.0),
        nk: fill(hd, 1.0),
        wo: fill(d * d, 0.013),
        w1: fill(mlp * d, 0.014),
        w3: fill(mlp * d, 0.015),
        w2: fill(d * mlp, 0.016),
    };
    ModelWeights {
        img_in: fill(d * cin, 0.02),
        txt_in: fill(d * c.context_in_dim, 0.02),
        time_a: fill(d * 256, 0.02),
        time_b: fill(d * d, 0.02),
        mod_img: fill(6 * d * d, 0.001),
        mod_txt: fill(6 * d * d, 0.001),
        mod_single: fill(3 * d * d, 0.001),
        final_adaln: fill(2 * d * d, 0.001),
        final_w: fill(cin * d, 0.02),
        dbl: (0..c.depth_double).map(|_| DoubleW { img: stream(), txt: stream() }).collect(),
        sgl: (0..c.depth_single)
            .map(|_| SingleW {
                wq: fill(d * d, 0.01),
                wk: fill(d * d, 0.011),
                wv: fill(d * d, 0.012),
                nq: fill(hd, 1.0),
                nk: fill(hd, 1.0),
                w1: fill(mlp * d, 0.014),
                w3: fill(mlp * d, 0.015),
                wo_a: fill(d * d, 0.013),
                wo_b: fill(d * mlp, 0.016),
            })
            .collect(),
    }
}

#[test]
#[ignore = "wants a whole GPU and several minutes; re-measure explicitly"]
fn device_lora_step_time() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_DEV_GPU=1 (needs a GPU) for the device step-time measurement");
        return;
    }
    let variant = envs("BRAIN_FLUX2_TRAIN_VARIANT", "klein-4b");
    let size = envn("BRAIN_FLUX2_TRAIN_SIZE", 512);
    let rank = envn("BRAIN_FLUX2_TRAIN_RANK", 16);
    let iters = envn("BRAIN_FLUX2_TRAIN_ITERS", 3);
    assert!(size.is_multiple_of(16), "size must be a multiple of 16");
    let fc = Flux2Config::from_name(&variant).expect("variant");
    let c = Cfg::from_flux2(&fc, size / 16, size / 16);
    eprintln!("flux2 device step time: {variant} at {size}px - {} joint tokens ({} txt + {} img), hidden {}, rank {rank}", c.n(), c.txt_len, c.n_img(), c.hidden);

    let t0 = std::time::Instant::now();
    let base = weights(&c);
    eprintln!("  host weights built in {:.1}s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let tr = DeviceTrainer::new(c.clone(), rank, &base);
    eprintln!("  device upload {:.1}s, base+adapter resident {:.2} GiB", t0.elapsed().as_secs_f64(), tr.weight_bytes() as f64 / (1u64 << 30) as f64);
    drop(base);

    let mut ad = LoraAdapter::new(&c, LoraCfg::new(rank));
    let x0 = fill(c.n_img() * c.in_channels, 0.3);
    let ctx = fill(c.txt_len * c.context_in_dim, 0.1);
    let noise = fill(x0.len(), -0.2);
    let batch = make_flow_batch(&c, &x0, &ctx, 0.5, &noise);

    // Warm-up: never enters the statistics.
    let t0 = std::time::Instant::now();
    let l0 = tr.step(&mut ad, &batch, 1e-4);
    eprintln!("  warm-up step {:.2}s (loss {l0:.5}, discarded from statistics)", t0.elapsed().as_secs_f64());

    tr.gpu().set_kernel_timing(true);
    tr.gpu().reset_kernel_times();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        tr.step(&mut ad, &batch, 1e-4);
        times.push(t.elapsed().as_secs_f64());
    }
    let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times.iter().sum::<f64>() / iters as f64;
    eprintln!("\nSTEP TIME {variant} @{size}px rank {rank}: best-of-{iters} {best:.2}s, mean {mean:.2}s");
    eprintln!("  projected 1500-step run: {:.2} h (at best-of-{iters})", best * 1500.0 / 3600.0);

    if let Some(mut k) = tr.gpu().kernel_times() {
        let total: f64 = k.iter().map(|(_, ms, _)| *ms).sum();
        k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("\n  GPU kernel time over {iters} steps: {:.1} ms total ({:.0}% of {:.0} ms wall)", total, 100.0 * total / (mean * iters as f64 * 1000.0), mean * iters as f64 * 1000.0);
        eprintln!("  {:<26} {:>10} {:>8} {:>9}", "kernel", "ms/step", "share", "calls/step");
        for (name, ms, n) in k.iter().take(18) {
            eprintln!("  {:<26} {:>10.1} {:>7.1}% {:>9}", name, ms / iters as f64, 100.0 * ms / total, *n as usize / iters);
        }
    } else {
        eprintln!("  (no GPU timestamp queries on this backend)");
    }
    assert!(best.is_finite() && best > 0.0, "a step must take measurable time");
}
