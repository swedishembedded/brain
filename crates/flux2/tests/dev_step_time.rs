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
//! steps after the warm-up, default 3), `BRAIN_FLUX2_TRAIN_CARDS` (GPUs the
//! block stack is spread over, default 1).
//!
//! `BRAIN_FLUX2_TRAIN_CARDS` exists because without it this harness cannot
//! measure the variant it names in its own default list: klein-9B's fp32
//! frozen base is larger than one 24 GiB card, so `klein-9b` here could only
//! ever out-of-memory. `finetune::run` builds its trainer with
//! `DeviceTrainer::new_multi(cards, ..)`; so does this, and for the same
//! reason - a harness that constructed the trainer differently from the run
//! it is meant to predict would be measuring a configuration nobody trains
//! in.
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

use flux2::devtrain::{step_flops, DeviceTrainer};
use flux2::grad::{DoubleW, SingleW, StreamW};
use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{make_flow_batch, Cfg, ModelWeights};
use flux2::Flux2Config;

/// Roofs MEASURED on this box (2x Tesla P40), not datasheet numbers. Pascal
/// has no fast fp16, so there is no half-precision rung to aim at.
const ROOF_FP32_GFLOPS: f64 = 10_517.0;
const ROOF_DRAM_GBS: f64 = 287.5;

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
    let cards = envn("BRAIN_FLUX2_TRAIN_CARDS", 1).max(1);
    assert!(size.is_multiple_of(16), "size must be a multiple of 16");
    let fc = Flux2Config::from_name(&variant).expect("variant");
    let c = Cfg::from_flux2(&fc, size / 16, size / 16);
    eprintln!("flux2 device step time: {variant} at {size}px - {} joint tokens ({} txt + {} img), hidden {}, rank {rank}", c.n(), c.txt_len, c.n_img(), c.hidden);

    let t0 = std::time::Instant::now();
    let base = weights(&c);
    eprintln!("  host weights built in {:.1}s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let tr = DeviceTrainer::new_multi(cards, c.clone(), rank, &base);
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let per: Vec<String> = tr.weight_bytes_per_card().iter().map(|b| format!("{:.2}", gib(*b))).collect();
    eprintln!(
        "  device upload {:.1}s over {} card(s), base+adapter resident {:.2} GiB ({} GiB)",
        t0.elapsed().as_secs_f64(),
        tr.cards(),
        gib(tr.weight_bytes()),
        per.join(" + ")
    );
    drop(base);

    // Measure the configuration a REAL run uses, not the gate's. The parity
    // gate keeps the frozen QK-RMSNorm gain gradient on so it can check it;
    // `finetune::run` turns it off because nothing consumes it. A harness
    // that timed the gate's configuration would report a step cost no
    // training run ever pays.
    tr.set_qk_grads(false);
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
    tr.gpu().reset_ops_counters();
    tr.reset_timing();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        tr.step(&mut ad, &batch, 1e-4);
        times.push(t.elapsed().as_secs_f64());
    }
    let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = times.iter().sum::<f64>() / iters as f64;
    let n = iters as f64;

    eprintln!("\nSTEP TIME {variant} @{size}px rank {rank}: best-of-{iters} {best:.2}s, mean {mean:.2}s");
    eprintln!("  (measured in the trainer's configuration: frozen QK gain gradient OFF)");
    eprintln!("  projected 1500-step run: {:.2} h (at best-of-{iters})", best * 1500.0 / 3600.0);

    // ---- roofline: the analytic model, then what the run actually asked for ----
    let f = step_flops(&c, rank);
    let dev = f.device_total() as f64;
    eprintln!("\n=== ROOFLINE ===");
    eprintln!("  analytic model, one step (2*M*K*N per GEMM):");
    for (name, v) in [
        ("base linears  forward", f.linear_fwd),
        ("base linears  recompute", f.linear_recompute),
        ("base linears  backward (dx only)", f.linear_bwd),
        ("attention     forward", f.attn_fwd),
        ("attention     recompute", f.attn_recompute),
        ("attention     backward", f.attn_bwd),
        ("adapter       forward", f.lora_fwd),
        ("adapter       recompute", f.lora_recompute),
        ("adapter       backward", f.lora_bwd),
        ("embedders + head", f.wrapper),
    ] {
        eprintln!("    {name:34} {:>9.1} GFLOP  {:>5.1}%", v as f64 / 1e9, 100.0 * v as f64 / dev);
    }
    eprintln!("    {:34} {:>9.1} GFLOP", "DEVICE TOTAL", dev / 1e9);
    eprintln!("    {:34} {:>9.1} MFLOP (host, m=1)", "conditioning front", f.host_cond as f64 / 1e6);
    eprintln!("    recompute is {:.1}% of the device total", 100.0 * f.recompute_share());

    let cost = tr.gpu().ops_counters();
    let meas_flops = cost.total.flops as f64 / n;
    let meas_bytes = cost.total.bytes as f64 / n;
    eprintln!("\n  measured dispatch tally (gpu_core::cost, coverage {:.1}%):", 100.0 * cost.coverage());
    eprintln!("    {:34} {:>9.1} GFLOP  ({:+.1}% vs model)", "per step", meas_flops / 1e9, 100.0 * (meas_flops - dev) / dev);
    eprintln!("    {:34} {:>9.1} GB", "bytes moved per step", meas_bytes / 1e9);
    if !cost.uncovered.is_empty() {
        eprintln!("    UNCOVERED kernels (not in the totals): {:?}", cost.uncovered.keys().collect::<Vec<_>>());
    }

    let compute_floor = dev / (ROOF_FP32_GFLOPS * 1e9);
    let memory_floor = meas_bytes / (ROOF_DRAM_GBS * 1e9);
    let ridge = ROOF_FP32_GFLOPS / ROOF_DRAM_GBS;
    eprintln!("\n  floors on this hardware (fp32 {ROOF_FP32_GFLOPS:.0} GFLOP/s, DRAM {ROOF_DRAM_GBS:.1} GB/s, ridge {ridge:.1} FLOP/B):");
    eprintln!("    compute-bound floor {compute_floor:.2} s/step   memory-bound floor {memory_floor:.2} s/step");
    eprintln!("    step arithmetic intensity {:.1} FLOP/B -> {}-BOUND", dev / meas_bytes.max(1.0), if compute_floor > memory_floor { "COMPUTE" } else { "MEMORY" });
    let floor = compute_floor.max(memory_floor);
    eprintln!("    ACHIEVED {:.1}% of the floor ({best:.2} s vs {floor:.2} s, {:.2}x off)", 100.0 * floor / best, best / floor);
    eprintln!("    floor for a 1500-step run: {:.2} h", floor * 1500.0 / 3600.0);

    // ---- where the wall clock goes: host phases first, then kernels ----
    let tm = tr.timing();
    eprintln!("\n=== WALL CLOCK, per step ===");
    let mut acct = 0.0;
    for (name, v) in tm.rows() {
        eprintln!("  {name:26} {v:>7.3} s  {:>5.1}%", 100.0 * v / mean);
        acct += v;
    }
    eprintln!("  {:26} {:>7.3} s  {:>5.1}%", "(optimiser + unaccounted)", mean - acct, 100.0 * (mean - acct) / mean);

    if let Some(mut k) = tr.gpu().kernel_times() {
        let total: f64 = k.iter().map(|(_, ms, _)| *ms).sum();
        k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("\n=== GPU KERNELS, ranked by share of step wall clock ===");
        eprintln!("  GPU kernel time is {:.1}% of wall clock ({:.2} s of {mean:.2} s)", 100.0 * total / (mean * n * 1000.0), total / (n * 1000.0));
        eprintln!("  {:<24} {:>8} {:>7} {:>10} {:>9} {:>8} {:>7}", "kernel", "ms/step", "share", "GFLOP/s", "GB/s", "FLOP/B", "%roof");
        for (name, ms, calls) in k.iter().take(16) {
            let per = ms / n;
            let (fl, by) = cost.by_kernel.get(name).map(|c| (c.cost.flops as f64 / n, c.cost.bytes as f64 / n)).unwrap_or((0.0, 0.0));
            let secs = per / 1000.0;
            let gflops = fl / secs.max(1e-9) / 1e9;
            let gbs = by / secs.max(1e-9) / 1e9;
            let ai = fl / by.max(1.0);
            // A kernel below the ridge point is memory-bound and its ceiling is
            // DRAM, not the FMA rate; above it, the other way round.
            let pct = if ai >= ridge { 100.0 * gflops / ROOF_FP32_GFLOPS } else { 100.0 * gbs / ROOF_DRAM_GBS };
            eprintln!("  {name:<24} {per:>8.1} {:>6.1}% {gflops:>10.0} {gbs:>9.1} {ai:>8.1} {pct:>6.1}%", 100.0 * ms / total, );
            let _ = calls;
        }
        eprintln!("  (%roof is against DRAM below the ridge point and against fp32 FMA above it)");
    } else {
        eprintln!("  (no GPU timestamp queries on this backend)");
    }
    assert!(best.is_finite() && best > 0.0, "a step must take measurable time");
}
