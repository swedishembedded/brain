// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL UNet profiler: where a forward's time actually goes, per kernel kind.
//!
//! `docs/kernel-checklist.md` §E: profile per kernel-kind BEFORE touching
//! anything, and publish the table — on this engine every confident hypothesis
//! about what is slow has been wrong and the profile has been right.
//!
//! Method, and why each part is there:
//!
//! * **Weight-free.** The UNet's cost depends only on shape, so this drives
//!   randomly-initialised weights instead of the 10 GB checkpoint. Seconds to
//!   run, and the profile is the same.
//! * **Every timed region is bracketed by `poll_wait()`.** `submit` with an
//!   empty clear list only appends to `pending`; a loop of bare submits measures
//!   host-side bind-group construction and reports it as device bandwidth. That
//!   mistake produced 377 GB/s on a ~346 GB/s card (`docs/lessons.md` #6).
//! * **Best-of-N**, not mean: the minimum is the least contaminated sample.
//! * **Groups are contiguous runs of one kernel**, submitted in graph order, so
//!   the sum of the parts is comparable to the whole. Per-group drains add one
//!   queue round-trip each — `total` is measured separately and both are
//!   printed, so the drain overhead is visible rather than assumed away.
//!
//! Usage:
//!   unet_bench [h w] [reps]     # default 96 96 (a 768x768 image), 3 reps

use std::collections::HashMap;
use std::time::Instant;

use gpu_core::Gpu;
use unet::config::UNetConfig;
use unet::init::init_weights;
use unet::model::{Unet, KERNELS};

/// Tesla P40 fp32 peak (11.76 TFLOP/s) and its memory roof (~346 GB/s). Printed
/// as a denominator so a number above the roof is visibly impossible rather than
/// quietly believed.
const PEAK_TFLOPS: f64 = 11.76;

fn best_of(gpu: &Gpu, steps: &[gpu_core::Step], reps: usize) -> f64 {
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(|s| s == "gemm").unwrap_or(false) {
        let p = |i: usize, d: u32| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        gemm_ab(p(2, 77), p(3, 2048), p(4, 2560), 5);
        return;
    }
    let lh: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(96);
    let lw: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(96);
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let cfg = UNetConfig::sdxl_base();
    let w = init_weights(&cfg, 7);
    let gpu = Gpu::new(&KERNELS);
    eprintln!("unet_bench: SDXL latent {lh}x{lw} (image {}x{}), {reps} reps", lh * 8, lw * 8);
    let t0 = Instant::now();
    let m = Unet::new(gpu.share(), cfg.clone(), &w, lh, lw, 77, false);
    eprintln!("built in {:.1}s, {} dispatches\n", t0.elapsed().as_secs_f32(), m.steps().len());

    let steps = m.steps();
    let total = best_of(&gpu, steps, reps);

    // Contiguous runs of one kernel, so the sum is comparable to the whole and
    // graph order is preserved.
    let mut groups: Vec<(usize, usize, usize)> = Vec::new(); // (kernel, start, len)
    for (i, s) in steps.iter().enumerate() {
        let k = s.meta().map(|m| m.kernel).unwrap_or(usize::MAX);
        match groups.last_mut() {
            Some((gk, _, len)) if *gk == k => *len += 1,
            _ => groups.push((k, i, 1)),
        }
    }

    let mut per_kind: HashMap<usize, (f64, usize)> = HashMap::new();
    for (k, start, len) in &groups {
        let t = best_of(&gpu, &steps[*start..*start + *len], reps);
        let e = per_kind.entry(*k).or_insert((0.0, 0));
        e.0 += t;
        e.1 += *len;
    }

    let mut rows: Vec<(String, f64, usize)> = per_kind
        .into_iter()
        .map(|(k, (t, n))| (gpu.kernel_name(k).unwrap_or("?").to_string(), t, n))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let summed: f64 = rows.iter().map(|r| r.1).sum();
    println!("{:<26} {:>9} {:>8} {:>7}  {:>9}", "kernel", "ms", "n", "%", "ms/call");
    println!("{}", "-".repeat(66));
    for (name, t, n) in &rows {
        println!(
            "{:<26} {:>9.2} {:>8} {:>6.1}% {:>9.3}",
            name,
            t * 1e3,
            n,
            100.0 * t / summed,
            t * 1e3 / *n as f64
        );
    }
    println!("{}", "-".repeat(66));
    println!("{:<26} {:>9.2} {:>8}", "sum of groups", summed * 1e3, steps.len());
    println!("{:<26} {:>9.2}", "whole graph (one submit)", total * 1e3);
    println!(
        "\nper-group drain overhead: {:.1}% ({} drains)",
        100.0 * (summed - total) / total,
        groups.len()
    );
    println!("one forward: {:.2} ms  ->  {:.1} forwards/s", total * 1e3, 1.0 / total);
    println!("P40 fp32 peak {PEAK_TFLOPS} TFLOP/s; a rate above it means the host was timed, not the device.");

    // The shapes behind the top kernel. A per-kind total says WHAT is slow; the
    // shape histogram says WHY, and whether a faster sibling could have taken it.
    if let Some((top, _, _)) = rows.first() {
        let ki = gpu.kernel_index(top);
        let mut hist: HashMap<Vec<u32>, (usize, f64)> = HashMap::new();
        for (k, start, len) in &groups {
            if Some(*k) != ki {
                continue;
            }
            let t = best_of(&gpu, &steps[*start..*start + *len], 1) / *len as f64;
            for s in &steps[*start..*start + *len] {
                if let Some(p) = s.meta().and_then(|m| m.params.clone()) {
                    let e = hist.entry(p).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += t;
                }
            }
        }
        let mut hs: Vec<_> = hist.into_iter().collect();
        hs.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        println!("\n`{top}` by shape (params as recorded):");
        println!("{:<28} {:>6} {:>10} {:>12}", "params", "n", "ms", "GFLOP/s");
        for (p, (n, t)) in hs.iter().take(10) {
            // matmul params are [m, k, n]: 2*m*k*n FLOP.
            let gf = if p.len() >= 3 {
                2.0 * p[0] as f64 * p[1] as f64 * p[2] as f64 * *n as f64 / t / 1e9
            } else {
                f64::NAN
            };
            println!("{:<28} {:>6} {:>10.2} {:>12.1}", format!("{p:?}"), n, t * 1e3, gf);
        }
    }
}

/// A/B the GEMM kernels at one shape, for CORRECTNESS and speed.
///
/// `unet_bench gemm [m k n]` — defaults to the shape the profile says dominates
/// the SDXL forward, the cross-attention `kv` projection at `[77, 2048, 2560]`.
///
/// Correctness first: the naive kernel is the reference every fast sibling must
/// reproduce. A faster kernel that disagrees is not a faster kernel.
pub fn gemm_ab(m: u32, k: u32, n: u32, reps: usize) {
    let gpu = Gpu::new(&KERNELS);
    let mut rng = data::rng::Rng::new(11);
    let x: Vec<f32> = (0..(m * k) as usize).map(|_| rng.next_f32()).collect();
    let w: Vec<f32> = (0..(n * k) as usize).map(|_| rng.next_f32()).collect();
    let xb = gpu.storage((m * k) as u64);
    let wb = gpu.storage((n * k) as u64);
    gpu.write_f32(&xb, &x);
    gpu.write_f32(&wb, &w);

    let tiles = m.div_ceil(128) * n.div_ceil(128) * 256;
    let cands: [(&str, u32); 4] =
        [("matmul", m * n), ("matmul_reg2", tiles), ("matmul_reg3", tiles), ("matmul_gemv", n * 64)];
    println!("A/B at [m {m}, k {k}, n {n}]  ({} MFLOP)", 2.0 * m as f64 * k as f64 * n as f64 / 1e6);
    println!("{:<16} {:>10} {:>12} {:>14}", "kernel", "ms", "GFLOP/s", "max|Δ| vs naive");
    let mut reference: Option<Vec<f32>> = None;
    for (name, threads) in cands {
        let Some(ki) = gpu.kernel_index(name) else {
            println!("{name:<16}   (not registered on this device)");
            continue;
        };
        let out = gpu.storage((m * n) as u64);
        let st = vec![gpu.step(ki, &[&xb, &wb, &out], &[m, k, n], threads)];
        let t = best_of(&gpu, &st, reps);
        let got = gpu.read(&out, (m * n) as usize);
        let d = match &reference {
            None => {
                reference = Some(got);
                0.0
            }
            Some(r) => r.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max),
        };
        println!(
            "{name:<16} {:>10.2} {:>12.1} {:>14.3e}",
            t * 1e3,
            2.0 * m as f64 * k as f64 * n as f64 / t / 1e9,
            d
        );
    }
}
