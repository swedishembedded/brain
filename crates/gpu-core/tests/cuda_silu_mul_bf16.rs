// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `silu_mul_bf16`: the fp32 product equals an f64 `silu(gate) * up` to fp32 rounding and the packed bf16
//! operand is exactly the round-to-nearest-even bf16 of the fp32 product the same launch wrote (so the
//! library GEMM reads what a separate `cast_f32_bf16` would have produced).
//!
//! Swedish Embedded AB implements mixed-precision training stacks for gated-MLP language models for
//! its clients. If your team needs expertise in producing a GEMM's narrow operand in the pass that
//! computes it, you can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::provider::library::{GemmPrecision, LibraryGemmProvider};
use gpu_core::Gpu;

fn rne_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0;
    }
    ((b as u64 + 0x7fff + ((b >> 16) & 1) as u64) >> 16) as u16
}

#[test]
fn the_fused_product_and_its_bf16_rounding_agree_with_the_separate_passes() {
    let Ok(gpu) = Gpu::try_new_cuda(&[]) else {
        eprintln!("cuda_silu_mul_bf16: no CUDA device on this box - nothing to run");
        return;
    };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !provider.is_active() {
        eprintln!("cuda_silu_mul_bf16: this card runs the fp32 path - nothing to run");
        return;
    }
    let mut r = data::rng::Lcg::new(data::rng::random_seed());
    for count in [4096usize, 8, 17408 * 3] {
        let gate: Vec<f32> = (0..count).map(|_| 6.0 * r.signed()).collect();
        let up: Vec<f32> = (0..count).map(|_| 2.0 * r.signed()).collect();
        let (bg, bu) = (gpu.storage_init("gate", &gate), gpu.storage_init("up", &up));
        let h = gpu.storage_init("h", &vec![f32::NAN; count]);
        let mut steps = Vec::new();
        let bf = provider.silu_mul_bf16(&gpu, &mut steps, &bg, &bu, &h, count as u64).expect("the fused product must be offered");
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let got = gpu.read(&h, count);
        let packed: Vec<u32> = gpu.read(&bf, count / 2).iter().map(|w| w.to_bits()).collect();
        let mut worst = 0f64;
        for i in 0..count {
            let g = gate[i] as f64;
            let want = g / (1.0 + (-g).exp()) * up[i] as f64;
            worst = worst.max((got[i] as f64 - want).abs());
            let word = packed[i / 2];
            let half = if i % 2 == 0 { word & 0xffff } else { word >> 16 } as u16;
            assert_eq!(half, rne_bf16(got[i]), "count {count} element {i}: bf16 operand is not the rounding of the fp32 product");
        }
        println!("  count {count}: fp32 product max abs error {worst:.2e}");
        assert!(worst <= 2e-6 * 8.0, "count {count}: error {worst:.3e}");
    }
}
