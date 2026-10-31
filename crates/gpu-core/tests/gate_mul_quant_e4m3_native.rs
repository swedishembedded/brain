// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `gate_mul_quant_e4m3_128` kernel (`cu/gate_mul_quant_e4m3_128.cu`): the sigmoid output gate of a gated
//! attention layer feeding an FP8 `o_proj`, quantised to E4M3 in ONE pass, the gate read in place from the query projection's
//! interleaved [query | gate] heads.
//!
//! Swedish Embedded AB implements solutions for FP8 serving of LLMs on GPUs for its clients. If your team needs expertise in
//! cutting the memory passes between a gated attention and its FP8 output projection then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The reference is what the layer ran before: `concat_split` copies the gate halves out of the projection, `quant_epilogue`
//! (mode 2) makes `ctx * sigmoid(gate)`, and `quant_e4m3_128` quantises that product. Every step is elementwise or an exact max,
//! so the product, the bytes and the scales must be BIT-identical; the form without the stored product must write the same
//! bytes and leave the buffer alone.

use gpu_core::{Fused, Gpu};

const KERNELS: &[(&str, &str)] = &[("concat_split", kernels::CONCAT_SPLIT)];
const CONCAT_SPLIT: usize = 0;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn values(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n)
        .map(|i| {
            let u = (r.next_u32() % 20001) as f32 / 10000.0 - 1.0;
            let mag = 10f32.powf((r.next_u32() % 5) as f32 - 2.0);
            if i % 89 == 0 { 0.0 } else { u * mag * spread }
        })
        .collect()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn check(gpu: &Gpu, rows: u32, heads: u32, hd: u32, ctx_v: &[f32], q_full_v: &[f32], what: &str) {
    let k = heads * hd;
    let n = u64::from(rows * k);
    let (ctx, q_full) = (gpu.storage_init("ctx", ctx_v), gpu.storage_init("q_full", q_full_v));

    // Reference: the gate halves copied out, the epilogue's product, then the standalone quantiser.
    let gate = gpu.storage(n);
    let y_r = gpu.storage(n);
    let (xq, sx) = (gpu.storage(1), gpu.storage(1));
    let split = gpu.step(CONCAT_SPLIT, &[&q_full, &gate], &[rows * heads, 2, 1, 1, 1, hd], rows * heads * hd);
    let epilogue = gpu.fused_step(Fused::QuantEpilogue, &[&ctx, &gate, &y_r, &xq, &sx], &[k, rows, 2, 1]).expect("quant_epilogue offered");
    gpu.submit(&[], &[split, epilogue]);
    let mut steps = Vec::new();
    let x = gpu_core::provider::library::Operand { buf: &y_r, range: (0, n) };
    let (q_r, s_r) = gpu_core::provider::library::quantize_e4m3_rows(gpu, &mut steps, &x, rows, k).expect("quant_e4m3_128 offered");
    gpu.submit(&[], &steps);

    // Native: one pass, with and without the stored product.
    let y_n = gpu.storage(n);
    let mut steps = Vec::new();
    let (q_n, s_n) = gpu_core::provider::library::gate_mul_quantize_e4m3_rows(gpu, &mut steps, &ctx, &q_full, Some(&y_n), rows, k, hd).expect("gate_mul_quant_e4m3_128 offered");
    gpu.submit(&[], &steps);
    let untouched = gpu.storage_init("untouched", &vec![1.5; n as usize]);
    let mut steps = Vec::new();
    let (q_s, s_s) = gpu_core::provider::library::gate_mul_quantize_e4m3_rows(gpu, &mut steps, &ctx, &q_full, None, rows, k, hd).expect("offered");
    gpu.submit(&[], &steps);
    gpu.poll_wait();

    let why = format!("rows={rows} heads={heads} hd={hd} ({what})");
    let (words, groups) = ((n / 4) as usize, (n / 128) as usize);
    assert_eq!(bits(gpu.read(&y_n, n as usize)), bits(gpu.read(&y_r, n as usize)), "product differs: {why}");
    assert_eq!(bits(gpu.read(&s_n, groups)), bits(gpu.read(&s_r, groups)), "scales differ: {why}");
    assert_eq!(bits(gpu.read(&q_n, words)), bits(gpu.read(&q_r, words)), "E4M3 bytes differ: {why}");
    assert_eq!(bits(gpu.read(&s_s, groups)), bits(gpu.read(&s_r, groups)), "scales differ without the product: {why}");
    assert_eq!(bits(gpu.read(&q_s, words)), bits(gpu.read(&q_r, words)), "E4M3 bytes differ without the product: {why}");
    assert!(gpu.read(&untouched, n as usize).iter().all(|v| *v == 1.5), "{why}");
}

#[test]
fn the_gated_product_and_its_e4m3_rows_equal_the_chain_they_replace() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    // Qwen3.8-27B's layer (24 heads of 256), a single row, a row count that is no multiple of a block's 8 groups, and a narrower
    // model (heads of 128).
    for (rows, heads, hd) in [(1u32, 24u32, 256u32), (3, 24, 256), (130, 24, 256), (17, 8, 128)] {
        let (n, nq) = ((rows * heads * hd) as usize, (rows * 2 * heads * hd) as usize);
        for (seed, spread) in [(11u64, 1.0f32), (12, 40.0)] {
            check(&gpu, rows, heads, hd, &values(n, seed, spread), &values(nq, seed + 100, spread), "mixed magnitudes with zeros");
        }
        check(&gpu, rows, heads, hd, &vec![0.0; n], &values(nq, 5, 1.0), "an all-zero attention output gives all-zero blocks");
    }
}
