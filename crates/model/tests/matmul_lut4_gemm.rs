// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device parity for the two M8.5 codebook 4-bit GEMV kernels
//! (`matmul_q4_gemv_nf4`/`matmul_q4_gemv_f4e2m1`) against a host oracle that
//! LUT-dequantizes the weight exactly (`model::lut4::dequantize_weight_lut4`)
//! and matmuls it against the UNQUANTIZED fp32 activation. Unlike
//! `matmul_q4_gemm.rs`'s own tolerance (which absorbs BOTH weight- and
//! activation-quantization noise against a doubly-approximate oracle), the
//! weight side here is EXACT (the oracle uses the identical codebook the
//! kernel does, dequantized on the host), so the only remaining error source
//! is the on-device int8 ACTIVATION quantization - a much tighter bound,
//! and the right one for isolating a bug in the kernel's own codebook lookup
//! from ordinary quantization noise.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::lut4::{dequantize_weight_lut4, quantize_weight_lut4, F4E2M1_LUT, NF4_LUT};

const KERNELS: &[(&str, &str)] = &[
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_q4_gemv_nf4", kernels::MATMUL_Q4_GEMV_NF4),
    ("matmul_q4_gemv_f4e2m1", kernels::MATMUL_Q4_GEMV_F4E2M1),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn host_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w[j * k + i];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    d / (na.sqrt() * nb.sqrt())
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (g as f64 - w as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt()
}

/// Same on-device int8 activation quant every W4A8 kernel in this tree uses
/// (`max_abs_row` -> `quant_pack`), unchanged - see `matmul_q4_gemm.rs`'s own
/// `quant_x` (this is a byte-for-byte copy, kept local since these two test
/// files must not depend on each other).
fn quant_x(g: &Gpu, k_maxr: usize, k_qp: usize, x: &gpu_core::DeviceBuffer, m: u32, k: u32) -> (gpu_core::DeviceBuffer, gpu_core::DeviceBuffer) {
    let sx = g.storage(m as u64);
    let xq = g.storage((m * k / 4) as u64);
    let steps = [g.step(k_maxr, &[x, &sx], &[m, k], m), g.step(k_qp, &[x, &sx, &xq], &[m, k], m * k / 4)];
    g.submit(&[], &steps);
    (xq, sx)
}

fn check_lut4(lut: &[f32; 16], kernel_name: &str, seed: u64) {
    let g = gpu_core::testgpu::dev(KERNELS);
    let (k_maxr, k_qp, k_lut) = (idx(&g, "max_abs_row"), idx(&g, "quant_pack"), idx(&g, kernel_name));

    // Same k=64 stride-mismatch-revealing shape `matmul_q4_gemm.rs` uses:
    // x has k/4=16 words/row, w has k/8=8; two 32-element weight-scale groups.
    let (m, k, n) = (3usize, 64usize, 7usize);
    let mut rng = Lcg::new(seed);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);

    let x = g.storage_init("x", &x_h);
    let (xq, sx) = quant_x(&g, k_maxr, k_qp, &x, m as u32, k as u32);

    let (wq, sw) = quantize_weight_lut4(&w_h, n, k, lut);
    let wqb = g.storage(wq.len() as u64);
    g.write(&wqb, &wq);
    let swb = g.storage_init("sw", &sw);

    let out = g.storage((m * n) as u64);
    let steps = [g.step(k_lut, &[&xq, &wqb, &sx, &swb, &out], &[m as u32, k as u32, n as u32], n as u32 * 64)];
    g.submit(&[], &steps);
    let got = g.read(&out, m * n);

    // The oracle: EXACT LUT dequant of the weight (same codebook the kernel
    // uses), matmul'd against the UNQUANTIZED fp32 activation. Any error left
    // is purely the on-device int8 activation quant, so the bound is much
    // tighter than `matmul_q4_gemm.rs`'s own (which also absorbs weight
    // quantization noise).
    let deq_w = dequantize_weight_lut4(&wq, &sw, n, k, lut);
    let want = host_matmul(&x_h, &deq_w, m, k, n);
    let cos = cosine(&got, &want);
    let rel = rel_l2(&got, &want);
    eprintln!("{kernel_name} ({m}x{k}->{n}): cosine={cos:.6} rel_l2={rel:.4}");
    assert!(cos >= 0.999, "{kernel_name} cosine {cos:.6} < 0.999 (weight side is exact -- only int8 activation noise should remain)");
    assert!(rel < 0.05, "{kernel_name} rel_l2 {rel:.4} >= 0.05");
}

#[test]
fn matmul_q4_gemv_nf4_matches_the_exact_lut_dequant_oracle() {
    check_lut4(&NF4_LUT, "matmul_q4_gemv_nf4", 9001);
}

#[test]
fn matmul_q4_gemv_f4e2m1_matches_the_exact_lut_dequant_oracle() {
    check_lut4(&F4E2M1_LUT, "matmul_q4_gemv_f4e2m1", 9002);
}

/// `m > 32` (the kernel's own `REQUIRES m <= 32`, `matmul_q4_gemv_nf4.wgsl`'s
/// header) is not this kernel's regime -- not exercised here, matching
/// `matmul_q4_gemm.rs`'s own scope (that file's `_gemv` tests stay `m <= 32`
/// too, leaving the tiled `_dyn`-shaped register-tiled sibling as a
/// deliberately deferred follow-up, same as this file's own).
#[test]
fn nf4_and_f4e2m1_diverge_on_the_same_weight_bits_different_codebooks_different_answers() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let (k_maxr, k_qp) = (idx(&g, "max_abs_row"), idx(&g, "quant_pack"));
    let (k_nf4, k_f4) = (idx(&g, "matmul_q4_gemv_nf4"), idx(&g, "matmul_q4_gemv_f4e2m1"));

    let (m, k, n) = (2usize, 64usize, 3usize);
    let mut rng = Lcg::new(9003);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);

    let x = g.storage_init("x", &x_h);
    let (xq, sx) = quant_x(&g, k_maxr, k_qp, &x, m as u32, k as u32);

    // Quantize the SAME weight bits under each codebook separately (the
    // nearest-code search picks different codes per codebook, so `wq` itself
    // legitimately differs -- this test is about the KERNELS disagreeing on
    // the same logical input, not about identical packed bytes).
    let (wq_nf4, sw_nf4) = quantize_weight_lut4(&w_h, n, k, &NF4_LUT);
    let (wq_f4, sw_f4) = quantize_weight_lut4(&w_h, n, k, &F4E2M1_LUT);
    let wqb_nf4 = g.storage(wq_nf4.len() as u64);
    g.write(&wqb_nf4, &wq_nf4);
    let swb_nf4 = g.storage_init("sw_nf4", &sw_nf4);
    let wqb_f4 = g.storage(wq_f4.len() as u64);
    g.write(&wqb_f4, &wq_f4);
    let swb_f4 = g.storage_init("sw_f4", &sw_f4);

    let out_nf4 = g.storage((m * n) as u64);
    let out_f4 = g.storage((m * n) as u64);
    let steps = [
        g.step(k_nf4, &[&xq, &wqb_nf4, &sx, &swb_nf4, &out_nf4], &[m as u32, k as u32, n as u32], n as u32 * 64),
        g.step(k_f4, &[&xq, &wqb_f4, &sx, &swb_f4, &out_f4], &[m as u32, k as u32, n as u32], n as u32 * 64),
    ];
    g.submit(&[], &steps);
    let got_nf4 = g.read(&out_nf4, m * n);
    let got_f4 = g.read(&out_f4, m * n);

    let mut any_diff = false;
    for (a, b) in got_nf4.iter().zip(&got_f4) {
        if (a - b).abs() > 1e-4 {
            any_diff = true;
        }
    }
    assert!(any_diff, "two different fixed codebooks over the same weight bits should not produce identical output");
}
