// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! W4A8 q4 GEMM pipeline: on-device int8 activation quant (max_abs_row ->
//! quant_pack, UNCHANGED from the int8 tier) + host int4 weight quant
//! (`model::int4::quantize_weight_q4`) + `matmul_q4_dyn` / `matmul_q4_gemv` /
//! `moe_linear_gated_q4`, each checked against a plain fp32 host oracle.
//!
//! **Tolerance note**: 4-bit weight quantization is much coarser than int8's
//! (7 levels per sign vs 127). A tight per-element bound is the wrong gate
//! here -- the right one is a whole-tensor similarity, the same choice
//! `crates/zimage/tests/int8_matmul.rs` makes for int8. Measured on these
//! tiny synthetic shapes: cosine consistently >= 0.999 and relative-L2 well
//! under a tenth (both printed below) -- LOWER than "4-bit is famously lossy"
//! might suggest, because per-CHANNEL scaling (not per-tensor) keeps a single
//! outlier row from crushing the rest of that row's resolution, exactly as
//! `model::int8::quantize_weight`'s own doc explains for int8. A test that
//! reads "cosine 0.999" and worries the kernel is TOO accurate for 4-bit is
//! mistaking a per-channel-scaled synthetic shape's easy case for the general
//! one; do not tighten this gate off one observed run.
//!
//! `k = 16` is chosen deliberately in every shape below: x (int8-packed) has
//! `k/4 = 4` u32 words per row, w (int4-packed) has `k/8 = 2` -- HALF as
//! many -- so a stride mistake between the two operands is not hideable by a
//! coincidentally-equal word count the way a larger, rounder `k` might hide it.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::int4::quantize_weight_q4;

const KERNELS: &[(&str, &str)] = &[
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
    ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
    ("moe_linear_gated_q4", kernels::MOE_LINEAR_GATED_Q4),
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

/// Quantize activations on-device via the EXISTING int8 path -- q4 is W4A8,
/// so this half of the pipeline is byte-for-byte what every int8 model
/// already dispatches, not new code.
fn quant_x(g: &Gpu, k_maxr: usize, k_qp: usize, x: &gpu_core::DeviceBuffer, m: u32, k: u32) -> (gpu_core::DeviceBuffer, gpu_core::DeviceBuffer) {
    let sx = g.storage(m as u64);
    let xq = g.storage((m * k / 4) as u64);
    let steps = [g.step(k_maxr, &[x, &sx], &[m, k], m), g.step(k_qp, &[x, &sx, &xq], &[m, k], m * k / 4)];
    g.submit(&[], &steps);
    (xq, sx)
}

#[test]
fn matmul_q4_dyn_matches_fp32_oracle() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let (k_maxr, k_qp, k_dyn) = (idx(&g, "max_abs_row"), idx(&g, "quant_pack"), idx(&g, "matmul_q4_dyn"));

    // k=16: x has k/4=4 words/row, w has k/8=2 -- a stride mismatch is not hideable here.
    let (m, k, n) = (4usize, 16usize, 5usize);
    let mut rng = Lcg::new(7001);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);

    let x = g.storage_init("x", &x_h);
    let (xq, sx) = quant_x(&g, k_maxr, k_qp, &x, m as u32, k as u32);

    let (wq, sw) = quantize_weight_q4(&w_h, n, k);
    let wqb = g.storage(wq.len() as u64);
    g.write(&wqb, &wq);
    let swb = g.storage_init("sw", &sw);

    let out = g.storage((m * n) as u64);
    let steps = [g.step(k_dyn, &[&xq, &wqb, &sx, &swb, &out], &[m as u32, k as u32, n as u32], (m * n) as u32)];
    g.submit(&[], &steps);
    let got = g.read(&out, m * n);

    let want = host_matmul(&x_h, &w_h, m, k, n);
    let cos = cosine(&got, &want);
    let rel = rel_l2(&got, &want);
    eprintln!("matmul_q4_dyn ({m}x{k}->{n}): cosine={cos:.6} rel_l2={rel:.4} got[..4]={:?} want[..4]={:?}", &got[..4.min(got.len())], &want[..4.min(want.len())]);
    assert!(cos >= 0.99, "matmul_q4_dyn cosine {cos:.6} < 0.99");
    assert!(rel < 0.15, "matmul_q4_dyn rel_l2 {rel:.4} >= 0.15");
}

#[test]
fn matmul_q4_gemv_matches_fp32_oracle_and_matches_dyn() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let (k_maxr, k_qp, k_dyn, k_gemv) = (idx(&g, "max_abs_row"), idx(&g, "quant_pack"), idx(&g, "matmul_q4_dyn"), idx(&g, "matmul_q4_gemv"));

    // Decode-regime shape: small M (<= 32, the gemv kernel's REQUIRES), same
    // k=16 stride-mismatch shape as the dyn test above.
    let (m, k, n) = (3usize, 16usize, 7usize);
    let mut rng = Lcg::new(7002);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);

    let x = g.storage_init("x", &x_h);
    let (xq, sx) = quant_x(&g, k_maxr, k_qp, &x, m as u32, k as u32);

    let (wq, sw) = quantize_weight_q4(&w_h, n, k);
    let wqb = g.storage(wq.len() as u64);
    g.write(&wqb, &wq);
    let swb = g.storage_init("sw", &sw);

    let out_gemv = g.storage((m * n) as u64);
    let out_dyn = g.storage((m * n) as u64);
    let steps = [
        g.step(k_gemv, &[&xq, &wqb, &sx, &swb, &out_gemv], &[m as u32, k as u32, n as u32], n as u32 * 64),
        g.step(k_dyn, &[&xq, &wqb, &sx, &swb, &out_dyn], &[m as u32, k as u32, n as u32], (m * n) as u32),
    ];
    g.submit(&[], &steps);
    let got_gemv = g.read(&out_gemv, m * n);
    let got_dyn = g.read(&out_dyn, m * n);

    let want = host_matmul(&x_h, &w_h, m, k, n);
    let cos = cosine(&got_gemv, &want);
    let rel = rel_l2(&got_gemv, &want);
    eprintln!("matmul_q4_gemv ({m}x{k}->{n}): cosine={cos:.6} rel_l2={rel:.4}");
    assert!(cos >= 0.99, "matmul_q4_gemv cosine {cos:.6} < 0.99");
    assert!(rel < 0.15, "matmul_q4_gemv rel_l2 {rel:.4} >= 0.15");

    // gemv and dyn quantize identically (same xq/wq/sx/sw) -- they must agree
    // with EACH OTHER much more tightly than either agrees with the fp32
    // oracle: any drift here is a kernel bug, not a quantization artifact.
    for (i, (&a, &b)) in got_gemv.iter().zip(&got_dyn).enumerate() {
        assert!((a - b).abs() < 1e-3, "gemv vs dyn disagree at {i}: gemv={a} dyn={b}");
    }
}

#[test]
fn moe_linear_gated_q4_matches_fp32_oracle_and_zeroes_ungated_rows() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let (k_maxr, k_qp, k_moe) = (idx(&g, "max_abs_row"), idx(&g, "quant_pack"), idx(&g, "moe_linear_gated_q4"));

    let (m, k, n, n_experts, e_idx) = (6usize, 16usize, 5usize, 3u32, 1u32);
    let mut rng = Lcg::new(7003);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);
    // Alternate gate on/off per row for expert `e_idx`; other experts' columns
    // are irrelevant to this kernel (it only ever reads column `e_idx`).
    let mut gate_h = vec![0f32; m * n_experts as usize];
    for r in 0..m {
        gate_h[r * n_experts as usize + e_idx as usize] = if r % 2 == 0 { 1.0 } else { 0.0 };
    }

    let x = g.storage_init("x", &x_h);
    let (xq, sx) = quant_x(&g, k_maxr, k_qp, &x, m as u32, k as u32);

    let (wq, sw) = quantize_weight_q4(&w_h, n, k);
    let wqb = g.storage(wq.len() as u64);
    g.write(&wqb, &wq);
    let swb = g.storage_init("sw", &sw);
    let gate = g.storage_init("gate", &gate_h);

    let out = g.storage((m * n) as u64);
    let steps = [g.step(
        k_moe,
        &[&xq, &wqb, &sx, &swb, &gate, &out],
        &[m as u32, k as u32, n as u32, n_experts, e_idx],
        (m * n) as u32,
    )];
    g.submit(&[], &steps);
    let got = g.read(&out, m * n);

    let want_dense = host_matmul(&x_h, &w_h, m, k, n);
    let mut cos_num = 0.0f64;
    let mut cos_na = 0.0f64;
    let mut cos_nb = 0.0f64;
    for r in 0..m {
        let routed = r % 2 == 0;
        for c in 0..n {
            let got_v = got[r * n + c];
            if !routed {
                assert_eq!(got_v, 0.0, "row {r} is not routed to expert {e_idx} but out[{r},{c}]={got_v} != 0");
                continue;
            }
            let want_v = want_dense[r * n + c];
            cos_num += got_v as f64 * want_v as f64;
            cos_na += got_v as f64 * got_v as f64;
            cos_nb += want_v as f64 * want_v as f64;
        }
    }
    let cos = cos_num / (cos_na.sqrt() * cos_nb.sqrt());
    eprintln!("moe_linear_gated_q4 ({m}x{k}->{n}, e_idx={e_idx}): routed-row cosine={cos:.6}");
    assert!(cos >= 0.99, "moe_linear_gated_q4 routed-row cosine {cos:.6} < 0.99");
}
