// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `add_rmsnorm_fwd` and `rmsnorm_dx_add` against f64 transcriptions of the dispatch pairs they
//! replace (`add2` + `rmsnorm_rows`, and `rmsnorm_dx_rows` + `add2`), at the real model width
//! 5120 and a small width, outputs poisoned first.
//!
//! Swedish Embedded AB implements memory-bound training kernels for pre-norm transformers for
//! its clients. If your team needs expertise in folding the residual adds around a normalisation
//! into the normalisation's own pass, you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::Gpu;

const EPS: f64 = 1e-6;

fn vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| scale * r.signed()).collect()
}

fn case(gpu: &Gpu, d: usize, rows: usize) {
    let n = d * rows;
    let (a, b, w) = (vals(n, 1, 2.0), vals(n, 2, 2.0), vals(d, 3, 1.0));
    let (dy, res) = (vals(n, 4, 1.0), vals(n, 5, 1.0));
    let up = |s: &str, v: &[f32]| gpu.storage_init(s, v);
    let poison = |len: usize| gpu.storage_init("out", &vec![f32::NAN; len]);
    let (ba, bb, bw) = (up("a", &a), up("b", &b), up("w", &w));
    let (sum, out) = (poison(n), poison(n));
    let step = gpu_core::provider::cuda::add_rmsnorm_fwd_step(gpu, &[&ba, &bb, &bw, &sum, &out], d as u32, rows as u32, EPS as f32).expect("add_rmsnorm_fwd must be offered");
    gpu.submit(&[], &[step]);
    let (bx, bdy, bres) = (up("x", &a), up("dy", &dy), up("res", &res));
    let dx = poison(n);
    let step = gpu_core::provider::cuda::rmsnorm_dx_add_step(gpu, &[&bx, &bw, &bdy, &bres, &dx], d as u32, rows as u32, EPS as f32).expect("rmsnorm_dx_add must be offered");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let (got_sum, got_out, got_dx) = (gpu.read(&sum, n), gpu.read(&out, n), gpu.read(&dx, n));
    let (mut e_sum, mut e_out, mut e_dx) = (0f64, 0f64, 0f64);
    for r in 0..rows {
        let row = r * d..(r + 1) * d;
        let s: Vec<f64> = row.clone().map(|i| a[i] as f64 + b[i] as f64).collect();
        let inv = 1.0 / (s.iter().map(|v| v * v).sum::<f64>() / d as f64 + EPS).sqrt();
        for (j, i) in row.clone().enumerate() {
            e_sum = e_sum.max((got_sum[i] as f64 - s[j]).abs());
            e_out = e_out.max((got_out[i] as f64 - s[j] * inv * w[j] as f64).abs());
        }
        let x: Vec<f64> = row.clone().map(|i| a[i] as f64).collect();
        let rinv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / d as f64 + EPS).sqrt();
        let dot: f64 = row.clone().enumerate().map(|(j, i)| dy[i] as f64 * w[j] as f64 * x[j]).sum();
        let coef = rinv.powi(3) * dot / d as f64;
        for (j, i) in row.enumerate() {
            let want = res[i] as f64 + rinv * w[j] as f64 * dy[i] as f64 - coef * x[j];
            e_dx = e_dx.max((got_dx[i] as f64 - want).abs());
        }
    }
    println!("  d {d} rows {rows}: sum {e_sum:.2e}, out {e_out:.2e}, dx {e_dx:.2e}");
    assert!(e_sum <= 1e-6 && e_out <= 2e-5 && e_dx <= 2e-5, "d {d} rows {rows}: errors {e_sum:.2e} {e_out:.2e} {e_dx:.2e}");
}

fn the_fused_rmsnorm_kernels_match_the_f64_reference() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if gpu.kind() != "cuda" || !gpu_core::native_kernels_enabled() {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (d, rows) in [(5120usize, 7usize), (64, 3), (20, 1)] {
        case(&gpu, d, rows);
    }
    // A width that is not a multiple of four is declined, not mis-computed.
    let t = gpu.storage(6);
    assert!(gpu_core::provider::cuda::add_rmsnorm_fwd_step(&gpu, &[&t, &t, &t, &t, &t], 6, 1, 1e-6).is_none());
}

gpu_core::card_tests!(the_fused_rmsnorm_kernels_match_the_f64_reference,);
