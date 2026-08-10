// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Asserting correctness gates for the split-K GEMM family (audit F13).
//!
//! Before this file, the split-K weight-gradient kernels
//! (`matmul_dw_reg_splitk` + `dw_splitk_reduce`) and the split-K FORWARD GEMM
//! (`matmul_reg3_splitk`, on the served decode path) had NO asserting test
//! anywhere: their only "oracle" was a `println!` of max|delta| in
//! `vqgan_bench` — a comparison that cannot fail — and none of the serving
//! tests forces `splitk_slices` to fire at test shapes. This file dispatches
//! each kernel directly (the `router_bwd_expert_cap.rs` pattern) against an
//! f64 host oracle, at shapes that genuinely split the contraction (slices >
//! 1, including a ragged tail slice that does not divide evenly).
//!
//! All four kernels are cooperative wg256/barrier kernels, gated on
//! `workgroup_reductions` in production (`vae::blocks::grad`,
//! `qwen::serve::splitk_slices`) — skipped on a backend without it (the CPU
//! JIT), where they can never be selected.

use data::rng::Lcg;
use gpu_core::Gpu;

const PIPES: &[(&str, &str)] = &[
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("matmul_dw_reg_tn", kernels::MATMUL_DW_REG_TN),
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("dw_splitk_reduce", kernels::DW_SPLITK_REDUCE),
    ("matmul_reg3_splitk", kernels::MATMUL_REG3_SPLITK),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn skip(g: &Gpu) -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    if !g.caps().workgroup_reductions {
        eprintln!("skipping: backend has no workgroup_reductions; production never selects the split-K kernels here");
        return true;
    }
    false
}

/// f64 oracle for the whole `matmul_dw*` family's math:
/// `dw[n_, k_] += sum_m dY[m_, n_] * X[m_, k_]`, on top of `prior`.
fn host_dw(m: usize, k: usize, n: usize, dy: &[f32], x: &[f32], prior: &[f32]) -> Vec<f64> {
    let mut out: Vec<f64> = prior.iter().map(|&v| v as f64).collect();
    for n_ in 0..n {
        for k_ in 0..k {
            let mut acc = 0.0f64;
            for m_ in 0..m {
                acc += dy[m_ * n + n_] as f64 * x[m_ * k + k_] as f64;
            }
            out[n_ * k + k_] += acc;
        }
    }
    out
}

/// f64 oracle for the forward family: `out[m_, n_] = sum_k x[m_, k_] * w[n_, k_]`.
fn host_fwd(m: usize, k: usize, n: usize, x: &[f32], w: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for m_ in 0..m {
        for n_ in 0..n {
            let mut acc = 0.0f64;
            for k_ in 0..k {
                acc += x[m_ * k + k_] as f64 * w[n_ * k + k_] as f64;
            }
            out[m_ * n + n_] = acc;
        }
    }
    out
}

fn assert_close(name: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut max_abs = 0.0f64;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g as f64 - w).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
    }
    assert!(
        max_abs < tol,
        "{name}: max|delta| = {max_abs:.3e} at [{at}] (got {}, want {}) exceeds {tol:.1e}",
        got[at],
        want[at]
    );
}

/// `matmul_dw_reg` (unsplit) and `matmul_dw_reg_tn` (dY pre-transposed) must
/// both ACCUMULATE the exact weight gradient onto a non-zero prior — the
/// parameter-gradient contract `blocks::grad` relies on.
#[test]
fn dw_reg_and_tn_accumulate_the_exact_weight_gradient() {
    let g = gpu_core::testgpu::dev(PIPES);
    if skip(&g) {
        return;
    }
    let (m, k, n) = (300usize, 48usize, 32usize); // tiles = 1, ragged m
    let mut rng = Lcg::new(0xD00D);
    let dy = rng.vec_scaled(m * n, 0.5);
    let x = rng.vec_scaled(m * k, 0.5);
    let prior = rng.vec_scaled(n * k, 0.5);
    let want = host_dw(m, k, n, &dy, &x, &prior);
    let tiles = n.div_ceil(128) * k.div_ceil(128);

    // dY as [m, n] for the base kernel.
    let a = g.storage_init("dy", &dy);
    // dY^T as [n, m] for the _tn variant — same math, coalesced A loads.
    let dyt: Vec<f32> = (0..n * m).map(|i| dy[(i % m) * n + i / m]).collect();
    let at = g.storage_init("dyt", &dyt);
    let b = g.storage_init("x", &x);

    for (name, buf) in [("matmul_dw_reg", &a), ("matmul_dw_reg_tn", &at)] {
        let out = g.storage_init("dw", &prior);
        g.submit(&[], &[g.step(idx(&g, name), &[buf, &b, &out], &[m as u32, k as u32, n as u32], (tiles * 256) as u32)]);
        g.poll_wait();
        let got = g.read(&out, n * k);
        assert_close(name, &got, &want, 1e-3);
    }
}

/// The split-K weight gradient: `matmul_dw_reg_splitk` writes per-slice
/// partials (ASSIGN) and `dw_splitk_reduce` folds them (ACCUMULATE, acc=1)
/// — together they must reproduce `matmul_dw_reg`'s accumulate exactly.
/// `m` deliberately does not divide by the slice count, so the tail slice is
/// ragged (the boundary a wrong slice bound silently corrupts).
#[test]
fn dw_splitk_plus_reduce_matches_the_oracle_with_a_ragged_tail_slice() {
    let g = gpu_core::testgpu::dev(PIPES);
    if skip(&g) {
        return;
    }
    let (m, k, n) = (500usize, 48usize, 32usize);
    let slices = 4usize; // ceil(500/4)=125 -> snapped to 128; slice 3 covers 384..500
    let mut rng = Lcg::new(0xF00D);
    let dy = rng.vec_scaled(m * n, 0.5);
    let x = rng.vec_scaled(m * k, 0.5);
    let prior = rng.vec_scaled(n * k, 0.5);
    let want = host_dw(m, k, n, &dy, &x, &prior);

    let a = g.storage_init("dy", &dy);
    let b = g.storage_init("x", &x);
    let rc = n * k;
    let part = g.storage((slices * rc) as u64);
    let dw = g.storage_init("dw", &prior);
    let tiles = n.div_ceil(128) * k.div_ceil(128);
    g.submit(
        &[],
        &[
            g.step(idx(&g, "matmul_dw_reg_splitk"), &[&a, &b, &part], &[m as u32, k as u32, n as u32, slices as u32], (slices * tiles * 256) as u32),
            g.step(idx(&g, "dw_splitk_reduce"), &[&part, &dw], &[rc as u32, slices as u32, 1], rc.div_ceil(64) as u32 * 64),
        ],
    );
    g.poll_wait();
    let got = g.read(&dw, rc);
    assert_close("matmul_dw_reg_splitk + dw_splitk_reduce", &got, &want, 1e-3);
}

/// The split-K FORWARD GEMM (`matmul_reg3_splitk` + `dw_splitk_reduce` with
/// acc=0, exactly as `qwen::serve::mm_into` composes them) against the f64
/// oracle — with a ragged k tail, and a pre-filled output to prove the
/// acc=0 reduce ASSIGNS rather than accumulating stale data.
#[test]
fn fwd_splitk_plus_reduce_matches_the_oracle_and_assigns() {
    let g = gpu_core::testgpu::dev(PIPES);
    if skip(&g) {
        return;
    }
    let (m, k, n) = (256usize, 200usize, 96usize);
    let slices = 3usize; // kper = ceil(200/3)=67 -> snapped to 72; slice 2 covers 144..200
    let mut rng = Lcg::new(0xBEEF);
    let x = rng.vec_scaled(m * k, 0.5);
    let w = rng.vec_scaled(n * k, 0.5);
    let want = host_fwd(m, k, n, &x, &w);

    let xb = g.storage_init("x", &x);
    let wb = g.storage_init("w", &w);
    let mn = m * n;
    let part = g.storage((slices * mn) as u64);
    // Poison the output: acc=0 must overwrite every element.
    let out = g.storage_init("out", &vec![123.0f32; mn]);
    let tiles = m.div_ceil(128) * n.div_ceil(128);
    g.submit(
        &[],
        &[
            g.step(idx(&g, "matmul_reg3_splitk"), &[&xb, &wb, &part], &[m as u32, k as u32, n as u32, slices as u32], (slices * tiles * 256) as u32),
            g.step(idx(&g, "dw_splitk_reduce"), &[&part, &out], &[mn as u32, slices as u32, 0], mn.div_ceil(64) as u32 * 64),
        ],
    );
    g.poll_wait();
    let got = g.read(&out, mn);
    assert_close("matmul_reg3_splitk + dw_splitk_reduce(acc=0)", &got, &want, 1e-3);
    assert!(got.iter().all(|&v| v != 123.0), "acc=0 reduce must ASSIGN over the poisoned output, not skip elements");
}
