// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bound GPU kernels that run at the card's roofline for its
// clients. If your team needs expertise in normalization kernels and their numerical gates then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A model that registered only the per-element `rmsnorm` and `layernorm` kernels still dispatches the
//! workgroup-per-row twins: the shared builders resolve them, no model opts in. The rows kernels agree with an f64
//! host normalization on a wide row and a narrow one.

use model::block::{self, LayerNormIds, KernelIds};

/// What a model registers when it never heard of the `_rows` twins.
const REFERENCE_ONLY: &[(&str, &str)] = &[("rmsnorm", kernels::RMSNORM), ("layernorm", kernels::LAYERNORM)];

fn rms_ids() -> KernelIds {
    let u = block::UNREGISTERED;
    KernelIds {
        rmsnorm: 0, rms_inv: u, rmsnorm_dx: u, rmsnorm_dx_rows: u, rmsnorm_dw: u, rope: u, rope_bwd: u, gqa_scores: u,
        gqa_apply: u, attn_softmax: u, gqa_dscores: u, gqa_dv: u, gqa_dq: u, gqa_dk: u, silu_mul: u, silu_da: u,
        silu_db: u, rmsnorm_rows: u,
    }
}

fn input(rows: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x = (0..rows * d).map(|i| 3.0 * (i as f32 * 0.7 + 0.1).sin() + 0.5).collect();
    let w = (0..d).map(|i| 0.5 + 0.25 * (i as f32 * 0.31).cos()).collect();
    let b = (0..d).map(|i| 0.1 * (i as f32 * 0.17).sin()).collect();
    (x, w, b)
}

/// Worst error relative to the largest output, against the f64 normalization.
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0f64, |m, (&a, &b)| m.max((f64::from(a) - b).abs())) / scale
}

fn rms_f64(x: &[f32], w: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * d);
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let ms = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + f64::from(eps)).sqrt();
        out.extend(row.iter().zip(w).map(|(&v, &g)| f64::from(v) * inv * f64::from(g)));
    }
    out
}

fn ln_f64(x: &[f32], g: &[f32], b: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * d);
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        out.extend(row.iter().zip(g.iter().zip(b)).map(|(&v, (&g, &b))| (f64::from(v) - mean) * inv * f64::from(g) + f64::from(b)));
    }
    out
}

/// Narrow (a per-head QK norm) and wide (a residual norm of a large model, past the `d_model is small` regime).
const SHAPES: [(usize, usize); 3] = [(32, 64), (3, 4096), (1, 8192)];
const EPS: f32 = 1e-5;

#[test]
fn rmsnorm_selects_the_rows_twin_without_registration_and_matches_f64() {
    let gpu = gpu_core::testgpu::dev(REFERENCE_ONLY);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("device cannot run a workgroup reduction, so the per-element kernel is its only RMSNorm");
        return;
    }
    let twin = gpu.kernel_index("rmsnorm_rows").expect("the Gpu does not carry rmsnorm_rows for a model that registered rmsnorm");
    let (kind, grid) = block::rms_variant(&gpu, 0, None, 1, 4096);
    assert_eq!((kind, grid), (twin, gpu_core::Dispatch::Workgroups(1)));
    for (rows, d) in SHAPES {
        let (x, w, _) = input(rows, d);
        let (xb, wb, ob) = (gpu.storage_init("x", &x), gpu.storage_init("w", &w), gpu.storage((rows * d) as u64));
        gpu.submit(&[], &[block::rmsnorm_fwd(&gpu, &rms_ids(), &xb, &wb, &ob, d as u32, rows as u32, EPS)]);
        let e = rel_err(&gpu.read(&ob, rows * d), &rms_f64(&x, &w, rows, d, EPS));
        assert!(e < 2e-5, "rmsnorm_rows {rows}x{d}: relative error {e:e} against the f64 host");
    }
}

#[test]
fn layernorm_selects_the_rows_twin_without_registration_and_matches_f64() {
    let gpu = gpu_core::testgpu::dev(REFERENCE_ONLY);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("device cannot run a workgroup reduction, so the per-element kernel is its only LayerNorm");
        return;
    }
    let ids = LayerNormIds::resolve_fwd(&gpu, 0);
    let twin = gpu.kernel_index("layernorm_rows").expect("the Gpu does not carry layernorm_rows for a model that registered layernorm");
    assert_eq!(block::ln_variant(&gpu, 0, ids.layernorm_rows, 1, 4096), (twin, gpu_core::Dispatch::Workgroups(1)));
    for (rows, d) in SHAPES {
        let (x, w, b) = input(rows, d);
        let (xb, wb, bb, ob) = (gpu.storage_init("x", &x), gpu.storage_init("g", &w), gpu.storage_init("b", &b), gpu.storage((rows * d) as u64));
        gpu.submit(&[], &[block::layernorm_fwd(&gpu, &ids, &xb, &wb, &bb, &ob, d as u32, rows as u32, EPS)]);
        let e = rel_err(&gpu.read(&ob, rows * d), &ln_f64(&x, &w, &b, rows, d, EPS));
        assert!(e < 2e-5, "layernorm_rows {rows}x{d}: relative error {e:e} against the f64 host");
    }
}
