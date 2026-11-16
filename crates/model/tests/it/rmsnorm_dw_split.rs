// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two-stage RMSNorm gain gradient (`rmsnorm_dw_part` then `rmsnorm_dw_sum`, what
//! `block::rmsnorm_dw_steps` selects for a training batch's rows) computes what the one-thread-per-channel
//! `rmsnorm_dw` computes, accumulating into the gradient the same way.

use model::block::{rmsnorm_dw_steps, KernelIds, RMSNORM_DW_SPLIT_ROWS, UNREGISTERED};

const KERNELS: &[(&str, &str)] = &[("rmsnorm_dw", kernels::RMSNORM_DW), ("rmsnorm_dw_part", kernels::RMSNORM_DW_PART), ("rmsnorm_dw_sum", kernels::RMSNORM_DW_SUM)];

#[test]
fn the_two_stage_gain_gradient_matches_the_one_stage_kernel() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (dim, rows) = (96u32, RMSNORM_DW_SPLIT_ROWS + 37);
    let mut rng = data::rng::Lcg::random();
    let mut fill = |n: u32| (0..n).map(|_| rng.signed()).collect::<Vec<f32>>();
    let (dy, x, inv, start) = (fill(dim * rows), fill(dim * rows), fill(rows), fill(dim));
    let up = |v: &[f32]| {
        let b = gpu.storage(v.len() as u64);
        gpu.write_f32(&b, v);
        b
    };
    let (dyb, xb, invb) = (up(&dy), up(&x), up(&inv));
    let (split, single) = (up(&start), up(&start));
    let ids = KernelIds { rmsnorm_dw: 0, rmsnorm: UNREGISTERED, rms_inv: UNREGISTERED, rmsnorm_dx: UNREGISTERED, rope: UNREGISTERED, rope_bwd: UNREGISTERED, gqa_scores: UNREGISTERED, gqa_apply: UNREGISTERED, attn_softmax: UNREGISTERED, gqa_dscores: UNREGISTERED, gqa_dv: UNREGISTERED, gqa_dq: UNREGISTERED, gqa_dk: UNREGISTERED, silu_mul: UNREGISTERED, silu_da: UNREGISTERED, silu_db: UNREGISTERED, rmsnorm_rows: UNREGISTERED, rmsnorm_dx_rows: UNREGISTERED };
    let steps = rmsnorm_dw_steps(&gpu, ids.rmsnorm_dw, &xb, &dyb, &invb, &split, dim, rows);
    assert_eq!(steps.len(), 2, "a training batch's rows take the two-stage form");
    gpu.submit(&[], &steps);
    gpu.submit(&[], &[gpu.step(0, &[&dyb, &xb, &invb, &single], &[dim, rows], dim)]);
    let (got, want) = (gpu.read(&split, dim as usize), gpu.read(&single, dim as usize));
    for c in 0..dim as usize {
        assert!((got[c] - want[c]).abs() <= 1e-4 * (1.0 + want[c].abs()), "channel {c}: {} vs {}", got[c], want[c]);
    }
    assert_eq!(rmsnorm_dw_steps(&gpu, ids.rmsnorm_dw, &xb, &dyb, &invb, &split, dim, 16).len(), 1, "a few rows keep the one-stage kernel");
}
