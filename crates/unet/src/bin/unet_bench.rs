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

    // Measure this device's own roofline first: every utilisation number below
    // divides by it, so a hardcoded P40 peak would make the table a statement
    // about one card rather than about whatever ran (`gpu_core::roof`).
    let roofs = gpu_core::roof::ensure(&gpu);
    match roofs {
        Some(r) => println!(
            "measured roofline: {:.0} GFLOP/s, {:.1} GB/s, ridge {:.1} FLOP/byte",
            r.gflops, r.gbs, r.ridge()
        ),
        None => println!("roofline unmeasured — utilisation columns print '-' rather than a guess"),
    }

    let steps = m.steps();
    // The §F.1 table, the grouping, the drain accounting and the coverage
    // honesty all live in `gpu_core::profile` — this bench used to carry its
    // own copy of them, as three others did.
    let p = gpu_core::profile::profile(&gpu, "FORWARD", steps, reps);
    p.print(roofs);
    if let Some(r) = roofs {
        for (row, bound, pct) in p.defects(r, 5.0) {
            println!(
                "  DEFECT  {:<24} {:>5.1}% of its {} roof (floor {:.0}%) — {:.1}% of this pass",
                row.name, pct, bound.as_str(), bound.defect_pct(),
                100.0 * row.secs / p.summed_secs,
            );
        }
    }
    let total = p.total_secs;
    let rows = &p.rows;
    println!("one forward: {:.2} ms  ->  {:.1} forwards/s", total * 1e3, 1.0 / total);

    // The shapes behind the top kernel. A per-kind total says WHAT is slow; the
    // shape histogram says WHY, and whether a faster sibling could have taken it.
    if let Some(top) = rows.first().map(|r| r.name.clone()) {
        let ki = gpu.kernel_index(&top);
        let mut hist: HashMap<Vec<u32>, (usize, f64)> = HashMap::new();
        for (k, start, len) in &gpu_core::profile::groups(steps) {
            if Some(*k) != ki {
                continue;
            }
            let t = gpu_core::profile::best_of(&gpu, &steps[*start..*start + *len], 1) / *len as f64;
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
        println!("{:<28} {:>6} {:>10} {:>12} {:>8}", "params", "n", "ms", "GFLOP/s", "%roof");
        for (pr, (n, t)) in hs.iter().take(10) {
            // FLOPs come from `gpu_core::cost`, not from a local matmul model:
            // this bench used to hardcode `2*p[0]*p[1]*p[2]`, which is silently
            // wrong for every kernel that is not a plain GEMM.
            let (gf, roofpct) = match gpu_core::cost::kernel_cost(&top, Some(pr), 0) {
                Some(c) => {
                    let work = c.flops.max(c.int_ops) * *n as u64;
                    let g = work as f64 / t / 1e9;
                    let pct = roofs
                        .and_then(|r| r.utilisation(work, c.bytes * *n as u64, *t))
                        .map(|u| format!("{u:.1}%"))
                        .unwrap_or_else(|| "-".into());
                    (format!("{g:.1}"), pct)
                }
                None => ("-".to_string(), "-".to_string()),
            };
            println!("{:<28} {:>6} {:>10.2} {:>12} {:>8}", format!("{pr:?}"), n, t * 1e3, gf, roofpct);
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
