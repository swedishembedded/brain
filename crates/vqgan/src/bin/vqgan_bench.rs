// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training-step profiler: where a FORWARD **and a BACKWARD** actually spend
//! their time, per kernel kind.
//!
//! Every profiler in this tree measured a forward. `docs/kernel-checklist.md`
//! §E asks for a per-kernel-kind table before anyone optimises, and for the
//! backward there was no way to get one — so the training datapath had never
//! been looked at, only assumed to look like the forward. It does not: the
//! reverse of a conv is TWO dispatches with different shapes (`conv2d_dx` reads
//! the weights transposed, `conv2d_dw` reduces over the batch), and the
//! per-channel reductions have no cooperative twin at all
//! (`vae::blocks::BWD_KERNELS`' documented §C.2 gap).
//!
//! The VQ autoencoder is the subject because its backward IS the shared block
//! backward set — `conv2d_dx/dw`, `gn_*`, `silu_bwd`, `upsample2_dx`, the
//! `attn_bwd_*_bidir` quartet — which `crates/vae`, `crates/unet` and
//! `crates/restore` all train through. A finding here is a finding for all of
//! them.
//!
//! Method (identical to `unet_bench`, deliberately):
//!   * every timed region is `poll_wait()`-bracketed — a bare `submit` only
//!     appends to `pending`, so an unbracketed loop times the HOST and reports
//!     it as device throughput (`docs/lessons.md` #6);
//!   * best-of-N, not mean: the minimum is the least contaminated sample;
//!   * groups are CONTIGUOUS runs of one kernel in submit order, so the sum of
//!     the parts is comparable to the whole, and both are printed.
//!
//! Usage:
//!   vqgan_bench [size] [reps]        # default 256, 3

use std::collections::HashMap;
use std::time::Instant;

use gpu_core::{Gpu, Step};
use vqgan::config::VqganConfig;
use vqgan::train::VqganTrainer;

/// Tesla P40 fp32 peak, printed as a denominator so a rate above the physical
/// roof is visibly impossible rather than quietly believed.
const PEAK_TFLOPS: f64 = 11.76;

fn best_of(gpu: &Gpu, steps: &[Step], reps: usize) -> f64 {
    gpu.submit(&[], steps);
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        gpu.submit(&[], steps);
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

/// Contiguous runs of one kernel, in graph order.
fn groups(steps: &[Step]) -> Vec<(usize, usize, usize)> {
    let mut g: Vec<(usize, usize, usize)> = Vec::new();
    for (i, s) in steps.iter().enumerate() {
        let k = s.meta().map(|m| m.kernel).unwrap_or(usize::MAX);
        match g.last_mut() {
            Some((gk, _, len)) if *gk == k => *len += 1,
            _ => g.push((k, i, 1)),
        }
    }
    g
}

fn report(gpu: &Gpu, label: &str, steps: &[Step], reps: usize) -> f64 {
    let total = best_of(gpu, steps, reps);
    let gs = groups(steps);
    let mut per: HashMap<usize, (f64, usize)> = HashMap::new();
    for (k, start, len) in &gs {
        let t = best_of(gpu, &steps[*start..*start + *len], reps);
        let e = per.entry(*k).or_insert((0.0, 0));
        e.0 += t;
        e.1 += *len;
    }
    let mut rows: Vec<(String, f64, usize)> = per
        .into_iter()
        .map(|(k, (t, n))| (gpu.kernel_name(k).unwrap_or("?").to_string(), t, n))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let summed: f64 = rows.iter().map(|r| r.1).sum();

    println!("\n=== {label}: {} dispatches, {:.2} ms ===", steps.len(), total * 1e3);
    println!("{:<26} {:>9} {:>7} {:>7}  {:>9}", "kernel", "ms", "n", "%", "ms/call");
    println!("{}", "-".repeat(64));
    for (name, t, n) in rows.iter().take(14) {
        println!(
            "{:<26} {:>9.2} {:>7} {:>6.1}% {:>9.3}",
            name,
            t * 1e3,
            n,
            100.0 * t / summed,
            t * 1e3 / *n as f64
        );
    }
    println!("{}", "-".repeat(64));
    println!("{:<26} {:>9.2}  (whole {:.2} ms, {} drains)", "sum of groups", summed * 1e3, total * 1e3, gs.len());
    total
}

/// Shape-correct random weights. The profile depends only on the graph, so this
/// stands in for the 377 MB checkpoint and makes the bench runnable anywhere —
/// the same reason `unet_bench` is weight-free.
fn init_weights(cfg: &VqganConfig, seed: u64) -> vae::blocks::Tensors {
    let mut rng = data::rng::Rng::new(seed);
    let mut t = vae::blocks::Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        let u = |r: &mut data::rng::Rng| 2.0 * r.next_f32() - 1.0;
        let d: Vec<f32> = match shape.len() {
            1 if name.ends_with(".weight") => (0..n).map(|_| 1.0 + 0.1 * u(&mut rng)).collect(),
            1 => (0..n).map(|_| 0.1 * u(&mut rng)).collect(),
            2 => (0..n).map(|_| 0.6 * u(&mut rng)).collect(),
            _ => {
                let s = 1.0 / ((n / shape[0]) as f32).sqrt();
                (0..n).map(|_| s * u(&mut rng)).collect()
            }
        };
        t.insert(name, (shape, d));
    }
    t
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let size: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let reps: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let cfg = VqganConfig::codeformer();
    let scale = cfg.downscale();
    assert!(size.is_multiple_of(scale), "size {size} must be a multiple of the {scale}x downscale");

    // Weight-free: the cost depends only on shape, so random weights profile the
    // same graph as the 377 MB checkpoint and the run takes seconds.
    let gpu = Gpu::new(vqgan::TRAIN_PIPELINES);
    let tensors = init_weights(&cfg, 7);
    eprintln!("vqgan_bench: {size}x{size}, latent {}x{}, {reps} reps", size / scale, size / scale);
    let t0 = Instant::now();
    let tr = VqganTrainer::new(cfg, &tensors, size, size, gpu.share());
    eprintln!("built in {:.1}s\n", t0.elapsed().as_secs_f32());

    let img: Vec<f32> = (0..(3 * size * size) as usize).map(|i| (i % 251) as f32 / 251.0).collect();
    tr.set_batch(&img, &img);
    tr.latch_assignment();

    let fwd = report(&gpu, "FORWARD", tr.fwd_steps(), reps);
    let bwd = report(&gpu, "BACKWARD", tr.bwd_steps(), reps);

    println!("\nforward {:.2} ms + backward {:.2} ms = {:.2} ms/step", fwd * 1e3, bwd * 1e3, (fwd + bwd) * 1e3);
    println!("backward/forward = {:.2}x   ({} clears before each backward)", bwd / fwd, tr.bwd_clears().len());
    println!("P40 fp32 peak {PEAK_TFLOPS} TFLOP/s; a rate above it means the host was timed, not the device.");
}
