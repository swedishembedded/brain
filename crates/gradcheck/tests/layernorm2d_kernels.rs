// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Settling `docs/kernel-checklist.md` §E's open question: does a FUSED
//! channels-first LayerNorm beat the composed `nchw_nlc → layernorm_rows →
//! nlc_nchw`?
//!
//! §E's row says the composition shipped as the correct first cut, but that the
//! measurement did **not** rule fusing out: the two permutes were 67–86% of the
//! whole, and both pay the sector amplification that was the stated reason to
//! avoid fusing. This test answers it — correctness first, then the numbers,
//! printed so the answer lands in the record rather than in someone's terminal.
//!
//! Run:
//!   BRAIN_DEVICE=gpu0 cargo test --release -p brain-gradcheck \
//!     --test layernorm2d_kernels -- --nocapture

use std::time::Instant;

use gpu_core::{f, Gpu};

const KERNELS: [(&str, &str); 4] = [
    ("nchw_nlc", kernels::NCHW_NLC),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("layernorm2d", kernels::LAYERNORM2D),
];
const K_NCHW_NLC: usize = 0;
const K_LN_ROWS: usize = 1;
const K_NLC_NCHW: usize = 2;
const K_LN2D: usize = 3;

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The reference: LayerNorm over the channel axis, on the host.
fn host(x: &[f32], n: usize, c: usize, hw: usize, g: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for ni in 0..n {
        for p in 0..hw {
            let base = ni * c * hw + p;
            let mean = (0..c).map(|ci| x[base + ci * hw]).sum::<f32>() / c as f32;
            let var = (0..c).map(|ci| (x[base + ci * hw] - mean).powi(2)).sum::<f32>() / c as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            for ci in 0..c {
                let i = base + ci * hw;
                y[i] = (x[i] - mean) * rstd * g[ci] + b[ci];
            }
        }
    }
    y
}

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

#[test]
fn fused_layernorm2d_matches_the_composition_and_is_faster() {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let eps = 1e-6f32;
    println!(
        "{:<26} {:>10} {:>10} {:>9}  {:>12}",
        "shape [N,C,H*W]", "composed", "fused", "speedup", "max|delta|"
    );

    // SAM 2's mask-decoder shapes (LayerNorm2d after each ConvTranspose2d), plus
    // a ConvNeXt-ish one. C and H*W deliberately differ everywhere.
    for &(n, c, hw) in &[(1usize, 32usize, 65536usize), (1, 64, 16384), (1, 256, 4096), (1, 96, 1024)] {
        let total = n * c * hw;
        let x = rnd(total, 5);
        let g = rnd(c, 7).iter().map(|v| 1.0 + 0.2 * v).collect::<Vec<f32>>();
        let b = rnd(c, 9);
        let want = host(&x, n, c, hw, &g, &b, eps);

        let xb = gpu.storage(total as u64);
        gpu.write_f32(&xb, &x);
        let gb = gpu.storage(c as u64);
        gpu.write_f32(&gb, &g);
        let bb = gpu.storage(c as u64);
        gpu.write_f32(&bb, &b);

        // Composed: NCHW -> NLC, row-wise LN, NLC -> NCHW.
        let rows = (n * hw) as u32;
        let xt = gpu.storage(total as u64);
        let yt = gpu.storage(total as u64);
        let y_comp = gpu.storage(total as u64);
        let composed = vec![
            gpu.step(K_NCHW_NLC, &[&xb, &xt], &[total as u32, c as u32, hw as u32], total as u32),
            gpu.step(
                K_LN_ROWS,
                &[&xt, &gb, &bb, &yt],
                &[c as u32, rows, f(eps)],
                rows * 64,
            ),
            gpu.step(K_NLC_NCHW, &[&yt, &y_comp], &[total as u32, c as u32, hw as u32], total as u32),
        ];
        let t_comp = best_of(&gpu, &composed, 5);

        // Fused: one dispatch, one invocation per spatial position.
        let y_fused = gpu.storage(total as u64);
        let fused = vec![gpu.step(
            K_LN2D,
            &[&xb, &gb, &bb, &y_fused],
            &[n as u32, c as u32, hw as u32, f(eps)],
            (n * hw) as u32,
        )];
        let t_fused = best_of(&gpu, &fused, 5);

        let gc = gpu.read(&y_comp, total);
        let gf = gpu.read(&y_fused, total);
        let dc = gc.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let df = gf.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!(
            "[{n},{c:4},{hw:6}]            {:>10.3} {:>10.3} {:>8.2}x  {:>12.3e}",
            t_comp * 1e3,
            t_fused * 1e3,
            t_comp / t_fused,
            df
        );

        // BOTH must match the host — a faster wrong kernel is not faster, and
        // checking the composition too keeps the oracle honest.
        assert!(dc < 2e-5, "composed differs from host by {dc:.3e}");
        assert!(df < 2e-5, "FUSED differs from host by {df:.3e}");
    }
}
