// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native `add_rmsnorm_fwd_bf16`: a residual add and the RMSNorm after it, the normalised rows written
//! as the packed bf16 operand of the tensor-core product that reads them. The reference is the pair it replaces on the
//! same device: `add_rmsnorm_fwd` and `cast_f32_bf16` of its output. Same sum, same bf16 words, bit for bit.
//!
//! Swedish Embedded AB implements solutions for keeping the memory passes between a served language model's products
//! to the ones the arithmetic needs for its clients. If your team needs expertise in GPU inference servers then you can
//! procure our services by sending an email to info@swedishembedded.com.

use gpu_core::provider::cuda::{add_rmsnorm_fwd_bf16_step, add_rmsnorm_fwd_step};
use gpu_core::provider::library::{cast_bf16, Operand};
use gpu_core::Gpu;

fn values(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|i| if i % 97 == 0 { 0.0 } else { ((r.next_u32() % 20001) as f32 / 10000.0 - 1.0) * spread }).collect()
}

#[test]
fn the_bf16_norm_writes_the_cast_of_the_fp32_norm_bit_for_bit() {
    let Ok(gpu) = Gpu::try_new_cuda(&[]) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    for (rows, d) in [(1u32, 2560u32), (37, 2560), (5, 64), (3, 4096)] {
        let n = (rows * d) as usize;
        let (a, b) = (gpu.storage_init("a", &values(n, 1, 30.0)), gpu.storage_init("b", &values(n, 2, 3.0)));
        let w = gpu.storage_init("w", &values(d as usize, 3, 2.0));
        let (sum, out, sum2) = (gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(n as u64));
        let bf = gpu.storage(n as u64 / 2);
        let mut steps = vec![add_rmsnorm_fwd_step(&gpu, &[&a, &b, &w, &sum, &out], d, rows, 1e-6).expect("add_rmsnorm_fwd offered")];
        let want = cast_bf16(&gpu, &mut steps, &Operand { buf: &out, range: (0, 0) }, n as u64).expect("cast offered");
        steps.push(add_rmsnorm_fwd_bf16_step(&gpu, &[&a, &b, &w, &sum2, &bf], d, rows, 1e-6).expect("add_rmsnorm_fwd_bf16 offered"));
        gpu.submit(&[], &steps);
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
        assert_eq!(bits(gpu.read(&bf, n / 2)), bits(gpu.read(&want, n / 2)), "rows {rows} d {d}: bf16 words differ");
        assert_eq!(bits(gpu.read(&sum2, n)), bits(gpu.read(&sum, n)), "rows {rows} d {d}: sums differ");
    }
}
