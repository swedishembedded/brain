// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RMSNorm backward w.r.t. the GAIN (`rmsnorm_dw`): is the cross-row
//! accumulation a coalescing defect like `rmsnorm_dx` was, or only a
//! parallelism question?
//!
//! This measures rather than assumes, because the two backward halves have
//! genuinely different reduction shapes and the answer for one does not carry
//! to the other:
//!
//! * `rmsnorm_dx` reduces WITHIN a row (`sum(x^2)`, `sum(dy*w*x)`), one row per
//!   thread - so a warp's 32 loads are `d` floats apart and each 32-byte sector
//!   fetched serves ONE useful float. That is a coalescing defect, and
//!   `rmsnorm_dx_rows` fixes it by giving the row a whole workgroup.
//! * `rmsnorm_dw` reduces ACROSS rows (`dW[c] = sum_n dY[n,c]*x[n,c]*inv[n]`),
//!   one CHANNEL per thread. Thread `c` and thread `c+1` read adjacent
//!   addresses at every step of the loop, so this kernel is ALREADY fully
//!   coalesced - the same fix would buy nothing. What it can be short of is
//!   parallelism: it launches exactly `d` threads regardless of how many rows
//!   they each walk.
//!
//! So the number to look at here is achieved bandwidth against the device's
//! own roof at the row counts training really runs, NOT a speedup against a
//! rewritten sibling. A kernel already near the roof has no defect to fix; one
//! far below it at large `rows` and small `d` is short of occupancy, which is a
//! different (two-stage / row-split) fix than the one the `_rows` family makes.
//!
//! ```text
//! DISPLAY= BRAIN_DEVICE=gpu cargo test --release -p brain-gpu-core \
//!     --test bench_rmsnorm_dw -- --ignored --nocapture
//! ```
//!
//! `PEAK_GBPS` is a placeholder roof; on another card set it to that card's own
//! figure and read the achieved column, not the percentage.

use gpu_core::Gpu;

const PEAK_GBPS: f64 = 346.0;

/// `(rows, d_model)` - the same family `bench_rmsnorm_dx` sweeps, so the two
/// halves of one backward can be read side by side, plus the narrow per-head
/// QK-norm rows where `d` is smallest and the occupancy question sharpest.
const SHAPES: &[(u32, u32)] = &[
    (512, 896),
    (2048, 896),
    (512, 2048),
    (2048, 2048),
    (512, 5120),
    (2048, 5120),
    (2048, 128),
    (16384, 128),
];

fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + s * 13) % 197) as f32 / 197.0) - 0.5).collect()
}

/// Min-of-`reps` wall clock for one dispatch (warm-up submitted first).
fn time(gpu: &Gpu, kind: usize, bufs: &[&gpu_core::DeviceBuffer], p: &[u32], threads: u32, reps: usize) -> f64 {
    let s = gpu.step(kind, bufs, p, threads);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        // 4 back-to-back dispatches so launch overhead is amortised, as in
        // `bench_rmsnorm_dx`/`bench_layernorm`; the reported time is per
        // dispatch.
        let steps: Vec<_> = (0..4).map(|_| gpu.step(kind, bufs, p, threads)).collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() / 4.0);
    }
    best
}

#[test]
#[ignore]
fn bench_rmsnorm_dw() {
    let ks = &[("rmsnorm_dx_rows", kernels::RMSNORM_DX_ROWS), ("rmsnorm_dw", kernels::RMSNORM_DW)];
    let (dxr, dw) = (0usize, 1);
    let g = Gpu::new_wgpu(ks);
    let reps = 8;

    println!(
        "\nrmsnorm_dw (gain grad, one thread per CHANNEL, reduces across rows)\n{:<14} {:>10} {:>10} {:>8} {:>12} {:>12}",
        "rows x d", "dw ms", "dw GB/s", "% roof", "threads", "dx_rows ms"
    );
    println!("{}", "-".repeat(74));
    for &(rows, d) in SHAPES {
        let n = (rows * d) as usize;
        let xb = g.storage_init("x", &fill(n, 1));
        let wb = g.storage_init("w", &fill(d as usize, 2));
        let dyb = g.storage_init("dy", &fill(n, 3));
        let invb = g.storage_init("inv", &fill(rows as usize, 4));
        let dwb = g.storage(d as u64);
        let dxb = g.storage(n as u64);
        let p = [d, rows];

        // Bytes the minimal implementation must move: `dy` and `x` read once
        // each, `inv` once per row, `dw` read-modify-written once per channel.
        let bytes = 2.0 * n as f64 * 4.0 + rows as f64 * 4.0 + 2.0 * d as f64 * 4.0;

        let dw_bufs: Vec<&gpu_core::DeviceBuffer> = vec![&dyb, &xb, &invb, &dwb];
        let dx_bufs: Vec<&gpu_core::DeviceBuffer> = vec![&xb, &wb, &dyb, &dxb];
        let tdw = time(&g, dw, &dw_bufs, &p, d, reps);
        let tdx = time(&g, dxr, &dx_bufs, &p, rows * 64, reps);
        let gbps = bytes / tdw / 1e9;
        println!(
            "{:>6} x {:<5} {:>10.3} {:>10.0} {:>7.0}% {:>12} {:>12.3}",
            rows,
            d,
            tdw * 1e3,
            gbps,
            100.0 * gbps / PEAK_GBPS,
            d,
            tdx * 1e3
        );
    }
    println!("\n(GB/s vs {PEAK_GBPS} GB/s placeholder peak; `threads` is the whole grid this kernel launches)");
}
