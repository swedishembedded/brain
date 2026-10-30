// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `silu_mul_quant_e4m3_128` kernel (`cu/silu_mul_quant_e4m3_128.cu`): the SwiGLU product feeding
//! an FP8 `down` projection and its E4M3 rows in ONE pass.
//!
//! Swedish Embedded AB implements solutions for FP8 serving of LLMs on GPUs for its clients. If your team needs expertise
//! in cutting the memory passes between two FP8 tensor-core GEMMs then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! The reference is the pair the kernel replaces on the same device and inputs: `quant_epilogue` (mode 1, product only)
//! then `quant_e4m3_128` on the product. Every step is elementwise or an exact max, so the product, the bytes and the
//! scales must be BIT-identical.

use gpu_core::{Fused, Gpu};

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

fn check(gpu: &Gpu, rows: u32, k: u32, gate: &[f32], up: &[f32], what: &str) {
    let n = (rows * k) as u64;
    let (gb, ub) = (gpu.storage_init("gate", gate), gpu.storage_init("up", up));

    // Reference: the epilogue's product, then the standalone quantiser.
    let y_r = gpu.storage(n);
    let (xq, sx) = (gpu.storage(1), gpu.storage(1));
    let step = gpu.fused_step(Fused::QuantEpilogue, &[&gb, &ub, &y_r, &xq, &sx], &[k, rows, 1, 1]).expect("quant_epilogue offered");
    gpu.submit(&[], &[step]);
    let mut steps = Vec::new();
    let x = gpu_core::provider::library::Operand { buf: &y_r, range: (0, n) };
    let (q_r, s_r) = gpu_core::provider::library::quantize_e4m3_rows(gpu, &mut steps, &x, rows, k).expect("quant_e4m3_128 offered");
    gpu.submit(&[], &steps);

    // Native: one pass.
    let y_n = gpu.storage(n);
    let mut steps = Vec::new();
    let (q_n, s_n) = gpu_core::provider::library::silu_mul_quantize_e4m3_rows(gpu, &mut steps, &gb, &ub, &y_n, rows, k).expect("silu_mul_quant_e4m3_128 offered");
    gpu.submit(&[], &steps);
    gpu.poll_wait();

    let ctx = format!("rows={rows} k={k} ({what})");
    let words = (n / 4) as usize;
    let groups = (n / 128) as usize;
    assert_eq!(bits(gpu.read(&y_n, n as usize)), bits(gpu.read(&y_r, n as usize)), "product differs: {ctx}");
    assert_eq!(bits(gpu.read(&s_n, groups)), bits(gpu.read(&s_r, groups)), "scales differ: {ctx}");
    assert_eq!(bits(gpu.read(&q_n, words)), bits(gpu.read(&q_r, words)), "E4M3 bytes differ: {ctx}");
}

#[test]
fn the_product_and_its_e4m3_rows_equal_the_pair_of_kernels_they_replace() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    // The MLP width of the 27B model, a row count that is no multiple of a block's 8 groups, and a single row.
    for (rows, k) in [(1u32, 128u32), (3, 17408), (768, 17408), (130, 5120)] {
        let n = (rows * k) as usize;
        for (seed, spread) in [(11u64, 1.0f32), (12, 40.0)] {
            check(&gpu, rows, k, &values(n, seed, spread), &values(n, seed + 100, spread), "mixed magnitudes with zeros");
        }
        check(&gpu, rows, k, &vec![0.0; n], &values(n, 5, 1.0), "all-zero gate gives all-zero blocks");
    }
}
