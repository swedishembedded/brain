// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `qkv_split` kernel (`cu/qkv_split.cu`): the rows of the ONE product a fused q|k|v weight makes,
//! taken apart in a single launch.
//!
//! Swedish Embedded AB implements solutions for latency-bound LLM decode on GPUs for its clients. If your team needs
//! expertise in cutting a decode step's launches between the weight streams of an attention layer then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The reference is the host: each output element is the product's element at a computable place (the query heads
//! interleave `hd` values and `hd` gates), and the kernel copies, so every output must equal it bit for bit. The
//! de-interleaved form must also equal what the two `concat_split` launches it replaces make on the same device.

use gpu_core::provider::library::{qkv_split_rows, QkvParts, QkvRow};
use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[("concat_split", kernels::CONCAT_SPLIT)];
const CONCAT_SPLIT: usize = 0;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn the_rows_of_the_product_come_apart_bit_for_bit() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    // The 27B attention layer (24 heads of 256, 4 key heads) at decode widths, and a small odd-row case.
    for (rows, heads, kv_heads, hd) in [(1u32, 24u32, 4u32, 256u32), (16, 24, 4, 256), (128, 24, 4, 256), (5, 2, 1, 64)] {
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let width = (2 * q_dim + 2 * kv_dim) as usize;
        let mut rng = data::rng::Lcg::new(u64::from(rows) * 31 + u64::from(hd));
        let product: Vec<f32> = rng.vec_scaled(rows as usize * width, 1.0);
        let pb = gpu.storage_init("product", &product);
        let row = QkvRow { rows, q_dim, head_dim: hd, kv_dim };
        let ctx = format!("rows={rows} heads={heads} hd={hd}");
        let (r, q, kv) = (rows as usize, q_dim as usize, kv_dim as usize);
        let at = |row: usize, col: usize| product[row * width + col];

        let mut steps = Vec::new();
        let Some(QkvParts::Interleaved { q_full, k, v }) = qkv_split_rows(&gpu, &mut steps, &pb, &row, false) else { panic!("qkv_split offered: {ctx}") };
        let Some(QkvParts::Split { q_value, q_gate, k: k2, v: v2 }) = qkv_split_rows(&gpu, &mut steps, &pb, &row, true) else { panic!("qkv_split offered: {ctx}") };
        gpu.submit(&[], &steps);
        gpu.poll_wait();

        let want_q_full: Vec<f32> = (0..r).flat_map(|i| (0..2 * q).map(move |c| (i, c))).map(|(i, c)| at(i, c)).collect();
        let want_k: Vec<f32> = (0..r).flat_map(|i| (0..kv).map(move |c| (i, c))).map(|(i, c)| at(i, 2 * q + c)).collect();
        let want_v: Vec<f32> = (0..r).flat_map(|i| (0..kv).map(move |c| (i, c))).map(|(i, c)| at(i, 2 * q + kv + c)).collect();
        let hd = hd as usize;
        let head_part = |gate: usize| -> Vec<f32> { (0..r).flat_map(|i| (0..q).map(move |c| (i, c))).map(|(i, c)| at(i, (c / hd) * 2 * hd + gate * hd + c % hd)).collect() };
        assert_eq!(bits(&gpu.read(&q_full, r * 2 * q)), bits(&want_q_full), "q_full: {ctx}");
        assert_eq!(bits(&gpu.read(&k, r * kv)), bits(&want_k), "k: {ctx}");
        assert_eq!(bits(&gpu.read(&v, r * kv)), bits(&want_v), "v: {ctx}");
        assert_eq!(bits(&gpu.read(&q_value, r * q)), bits(&head_part(0)), "query values: {ctx}");
        assert_eq!(bits(&gpu.read(&q_gate, r * q)), bits(&head_part(1)), "gates: {ctx}");
        assert_eq!(bits(&gpu.read(&k2, r * kv)), bits(&want_k), "k (split): {ctx}");
        assert_eq!(bits(&gpu.read(&v2, r * kv)), bits(&want_v), "v (split): {ctx}");

        // The two launches it replaces, on the same device.
        let (ref_value, ref_gate) = (gpu.storage((r * q) as u64), gpu.storage((r * q) as u64));
        let n = u64::from(rows * q_dim);
        let both = [
            gpu.step(CONCAT_SPLIT, &[&q_full, &ref_value], &[rows * heads, 2, 1, 0, 1, hd as u32], n as u32),
            gpu.step(CONCAT_SPLIT, &[&q_full, &ref_gate], &[rows * heads, 2, 1, 1, 1, hd as u32], n as u32),
        ];
        gpu.submit(&[], &both);
        gpu.poll_wait();
        assert_eq!(bits(&gpu.read(&q_value, r * q)), bits(&gpu.read(&ref_value, r * q)), "against concat_split, values: {ctx}");
        assert_eq!(bits(&gpu.read(&q_gate, r * q)), bits(&gpu.read(&ref_gate, r * q)), "against concat_split, gates: {ctx}");
    }
}

#[test]
fn shapes_the_kernel_does_not_take_are_declined() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    let pb = gpu.storage(4096);
    let mut steps = Vec::new();
    for row in [QkvRow { rows: 0, q_dim: 64, head_dim: 64, kv_dim: 64 }, QkvRow { rows: 1, q_dim: 64, head_dim: 64, kv_dim: 6 }, QkvRow { rows: 1, q_dim: 96, head_dim: 64, kv_dim: 64 }] {
        assert!(qkv_split_rows(&gpu, &mut steps, &pb, &row, true).is_none(), "{row:?}");
    }
    assert!(steps.is_empty(), "a declined shape pushes nothing");
}

/// Back-to-back launches at the served decode shapes (knowledge 410): 2,000 launches a shape, every result the first's, and
/// `dmesg | grep -ci xid` unchanged (`#[ignore]`: a stress run).
///
/// ```text
/// BRAIN_DEVICE=gpu1 BRAIN_BACKEND=cuda make test/one P=brain-gpu-core T=qkv_split_native ARGS="--ignored --nocapture back_to_back"
/// ```
#[test]
#[ignore]
fn the_split_survives_back_to_back_launches() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for rows in [1u32, 16, 128] {
        let row = QkvRow { rows, q_dim: 24 * 256, head_dim: 256, kv_dim: 4 * 256 };
        let width = (2 * row.q_dim + 2 * row.kv_dim) as usize;
        let product = gpu.storage_init("product", &data::rng::Lcg::new(5).vec_scaled(rows as usize * width, 1.0));
        for deinterleave in [false, true] {
            let mut steps = Vec::new();
            let first = qkv_split_rows(&gpu, &mut steps, &product, &row, deinterleave).expect("offered");
            gpu.submit(&[], &steps);
            gpu.poll_wait();
            let read = |parts: &QkvParts| match parts {
                QkvParts::Interleaved { q_full, k, v } => [(q_full, 2 * row.q_dim), (k, row.kv_dim), (v, row.kv_dim)].map(|(b, w)| bits(&gpu.read(b, (rows * w) as usize))).to_vec(),
                QkvParts::Split { q_value, q_gate, k, v } => [(q_value, row.q_dim), (q_gate, row.q_dim), (k, row.kv_dim), (v, row.kv_dim)].map(|(b, w)| bits(&gpu.read(b, (rows * w) as usize))).to_vec(),
            };
            let want = read(&first);
            for round in 0..10 {
                let mut steps = Vec::new();
                let mut last = None;
                for _ in 0..200 {
                    last = qkv_split_rows(&gpu, &mut steps, &product, &row, deinterleave);
                }
                gpu.submit(&[], &steps);
                gpu.poll_wait();
                assert_eq!(read(&last.expect("offered")), want, "rows={rows} deinterleave={deinterleave} round {round}");
            }
        }
    }
}
