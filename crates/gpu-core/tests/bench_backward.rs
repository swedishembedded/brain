// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tiled backward GEMMs (matmul_dx_reg / matmul_dw_reg) vs the naive versions:
//! parity + throughput on the P40. Training's dominant cost is the backward
//! GEMMs; the naive ones run at a tiny fraction of peak, like the old forward did.
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-gpu-core --test bench_backward -- --ignored --nocapture
//! ```

use gpu_core::Gpu;

const TOL: f32 = 5e-4;

struct Shape { label: &'static str, m: usize, k: usize, n: usize }
const SHAPES: &[Shape] = &[
    Shape { label: "gpt-small mlp 512x384->1536", m: 512, k: 384, n: 1536 },
    Shape { label: "qwen ffn      256x1024->3072", m: 256, k: 1024, n: 3072 },
    Shape { label: "square 2048", m: 2048, k: 2048, n: 2048 },
];

fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 31 + s * 17) % 89) as f32 / 89.0) - 0.5).collect()
}
fn reg_threads(rows: usize, cols: usize) -> u32 {
    (rows.div_ceil(128) * cols.div_ceil(128) * 256) as u32
}
fn diff(a: &[f32], b: &[f32]) -> f32 {
    let md = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    md / a.iter().fold(1e-6f32, |m, &v| m.max(v.abs()))
}
fn time(gpu: &Gpu, kind: usize, bufs: &[&gpu_core::DeviceBuffer], p: &[u32], threads: u32, reps: usize) -> f64 {
    let s = gpu.step(kind, bufs, p, threads); gpu.submit(&[], &[s]); gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let s = gpu.step(kind, bufs, p, threads); gpu.submit(&[], &[s]); gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn bench_backward() {
    let ks = &[
        ("matmul_dx", kernels::MATMUL_DX),
        ("matmul_dx_reg", kernels::MATMUL_DX_REG),
        ("matmul_dw", kernels::MATMUL_DW),
        ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ];
    let (dx, dxr, dw, dwr) = (0usize, 1usize, 2usize, 3usize);
    let g = Gpu::new_wgpu(ks);
    let reps = 5;
    println!("\n{:<30} {:>8} {:>12} {:>12} {:>10} {:>10}", "shape", "GFLOP", "naive GF/s", "reg GF/s", "speedup", "parity");
    println!("{}", "-".repeat(86));

    for s in SHAPES {
        let (m, k, n) = (s.m, s.k, s.n);
        let dy = fill(m * n, 1);
        let w = fill(n * k, 2);
        let x = fill(m * k, 3);
        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;

        // ---- dX = dY·W ----
        let dyb = g.storage_init("dy", &dy);
        let wb = g.storage_init("w", &w);
        let dxa = g.storage((m * k) as u64); let dxb = g.storage((m * k) as u64);
        let pdx = [m as u32, k as u32, n as u32, 0];
        let t_dxn = time(&g, dx, &[&dyb, &wb, &dxa], &pdx, (m * k) as u32, reps);
        let t_dxr = time(&g, dxr, &[&dyb, &wb, &dxb], &pdx, reg_threads(m, k), reps);
        let p_dx = diff(&g.read(&dxa, m * k), &g.read(&dxb, m * k));

        // ---- dW += dY^T·X ----
        let xb = g.storage_init("x", &x);
        let dwa = g.storage((n * k) as u64); let dwb = g.storage((n * k) as u64);
        let pdw = [m as u32, k as u32, n as u32];
        let t_dwn = time(&g, dw, &[&dyb, &xb, &dwa], &pdw, (n * k) as u32, reps);
        let t_dwr = time(&g, dwr, &[&dyb, &xb, &dwb], &pdw, reg_threads(n, k), reps);
        let p_dw = diff(&g.read(&dwa, n * k), &g.read(&dwb, n * k));

        println!("dX {:<27} {:>8.2} {:>12.0} {:>12.0} {:>9.1}x {:>10.1e}", s.label, gflop, gflop / t_dxn, gflop / t_dxr, t_dxn / t_dxr, p_dx);
        println!("dW {:<27} {:>8.2} {:>12.0} {:>12.0} {:>9.1}x {:>10.1e}", s.label, gflop, gflop / t_dwn, gflop / t_dwr, t_dwn / t_dwr, p_dw);
        assert!(p_dx < TOL, "{}: dx_reg diverges (rel {p_dx:.2e})", s.label);
        assert!(p_dw < TOL, "{}: dw_reg diverges (rel {p_dw:.2e})", s.label);
    }
}
