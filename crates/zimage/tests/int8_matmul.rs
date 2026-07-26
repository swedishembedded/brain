// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Int8 DP4A GEMM pipeline parity (GPU): on-device dynamic activation quant
//! (max_abs → quant_pack) + matmul_i8_dyn vs an fp32 matmul, at a DiT-sized
//! shape. Per-tensor int8 is lossy, so the gate is cosine ≥ 0.999, not exact.
//! Needs a GPU (DP4A + workgroup barriers don't run on the CPU JIT).

use gpu_core::Gpu;
use zimage::int8::quantize_weight;

const KERNELS: [(&str, &str); 4] = [
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("matmul", kernels::MATMUL),
];
const K_MAXR: usize = 0;
const K_QP: usize = 1;
const K_MM8: usize = 2;
const K_MM: usize = 3;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
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

#[test]
fn int8_gemm_matches_fp32() {
    if std::env::var("BRAIN_INT8_TEST").as_deref() != Ok("1") {
        eprintln!("SKIP: set BRAIN_INT8_TEST=1 (needs a GPU) for the int8 DP4A parity test");
        return;
    }
    let (m, k, n) = (320usize, 3840usize, 3840usize);
    let x = fill(m * k, 1);
    let w = fill(n * k, 2);
    let gpu = Gpu::new_wgpu(&KERNELS);

    // fp32 reference.
    let xb = gpu.storage_init("x", &x);
    let wb = gpu.storage_init("w", &w);
    let refb = gpu.storage((m * n) as u64);
    let s = gpu.step(K_MM, &[&xb, &wb, &refb], &[m as u32, k as u32, n as u32], (m * n) as u32);
    gpu.submit(&[], &[s]);
    let want = gpu.read(&refb, m * n);

    // int8: per-channel weight quant on host, activation quant on device.
    let (wq, sw) = quantize_weight(&w, n, k);
    let wqb = gpu.storage(wq.len() as u64);
    gpu.write(&wqb, &wq);
    let swb = gpu.storage_init("sw", &sw); // per-channel scales [N]
    let sx = gpu.storage(m as u64); // per-token scales [M]
    let xq = gpu.storage((m * k / 4) as u64);
    let out8 = gpu.storage((m * n) as u64);
    let steps = [
        gpu.step(K_MAXR, &[&xb, &sx], &[m as u32, k as u32], m as u32),
        gpu.step(K_QP, &[&xb, &sx, &xq], &[m as u32, k as u32], (m * k / 4) as u32),
        gpu.step(
            K_MM8,
            &[&xq, &wqb, &sx, &swb, &out8],
            &[m as u32, (k / 4) as u32, n as u32],
            (m as u32).div_ceil(128) * (n as u32).div_ceil(128) * 256,
        ),
    ];
    gpu.submit(&[], &steps);
    let got = gpu.read(&out8, m * n);

    let cos = cosine(&got, &want);
    let rel = {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&g, &w) in got.iter().zip(&want) {
            num += (g as f64 - w as f64).powi(2);
            den += (w as f64).powi(2);
        }
        (num / den).sqrt()
    };
    eprintln!("int8 DP4A GEMM parity ({m}×{k}→{n}): cosine={cos:.6}  rel_l2={rel:.4}");
    assert!(cos >= 0.999, "int8 cosine {cos:.6} < 0.999");
}
