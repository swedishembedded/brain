// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for `paged_flash_decode_gqa_combine_bf16`: the split-key decode attention's merge writing the context only
//! as the packed bf16 operand of the output projection. The reference is the merge it stands in for followed by
//! `cast_f32_bf16`, on the same device from the same partial records: the same bf16 words.
//!
//! Swedish Embedded AB implements solutions for latency-bound LLM decode on GPUs for its clients. If your team needs
//! expertise in removing the small launches between a decode step's weight streams then you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::provider::cuda::flash_decode_combine_bf16_step;
use gpu_core::provider::library::{cast_bf16, Operand};

const KERNELS: &[(&str, &str)] = &[("paged_flash_decode_gqa_combine", kernels::PAGED_FLASH_DECODE_GQA_COMBINE)];
const COMBINE: usize = 0;

#[test]
fn the_bf16_merge_writes_the_cast_of_the_fp32_merge() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() || !gpu_core::native_kernels_enabled() {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    let mut r = data::rng::Lcg::new(data::rng::random_seed());
    for (batch, heads, hd, splits) in [(1u32, 32u32, 128u32, 4u32), (16, 32, 128, 9), (3, 8, 256, 1)] {
        // Records of `hd` weighted values, the split's max logit and its weight sum.
        let records: Vec<f32> = (0..batch * heads * splits)
            .flat_map(|_| {
                let mut rec: Vec<f32> = (0..hd).map(|_| 4.0 * r.signed()).collect();
                rec.push(8.0 * r.signed());
                rec.push(1.0 + r.signed().abs());
                rec
            })
            .collect();
        let part = gpu.storage_init("part", &records);
        let n = (batch * heads * hd) as usize;
        let ctx = gpu.storage(n as u64);
        let params = [batch, heads, hd, splits];
        let mut steps = vec![gpu.step(COMBINE, &[&part, &ctx], &params, batch * heads * 256)];
        let want = cast_bf16(&gpu, &mut steps, &Operand { buf: &ctx, range: (0, 0) }, n as u64).expect("cast offered");
        let got = gpu.storage(n as u64 / 2);
        steps.push(flash_decode_combine_bf16_step(&gpu, &part, &got, params).expect("bf16 merge offered"));
        gpu.submit(&[], &steps);
        let bits = |b: &gpu_core::DeviceBuffer| gpu.read(b, n / 2).into_iter().map(f32::to_bits).collect::<Vec<u32>>();
        assert_eq!(bits(&got), bits(&want), "{params:?}: bf16 context differs");
    }
}
