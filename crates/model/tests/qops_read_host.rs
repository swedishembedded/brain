// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training and inference over
// quantized checkpoints for its clients. If your team needs expertise in
// reading and verifying low-precision weights without expanding a whole
// tensor, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `Ops::read_host` gives back the values a weight stores, whatever tier holds it, decoding a
//! bounded tile of rows at a time: the result equals the tier's own host decode of the same bytes,
//! and does not depend on the tile size.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::fp8::{dequantize_weight_fp8, quantize_weight_fp8, Fp8Format};
use model::int8::{dequantize_weight, quantize_weight};
use model::lut4::{dequantize_weight_lut4, quantize_weight_lut4, NF4_LUT};
use model::ops::{training_kernel_list, Ops, Weight, FP8_BLOCK};

const N: usize = 40;
const K: usize = 64;

/// Each tier with the host decode of the values its upload stores.
fn cases(raw: &[f32]) -> Vec<(&'static str, Dtype, Vec<f32>)> {
    let (q, s) = quantize_weight(raw, N, K);
    let (nq, ns) = quantize_weight_lut4(raw, N, K, &NF4_LUT);
    let (fq, fs) = quantize_weight_fp8(raw, N, K, Fp8Format::E4M3, FP8_BLOCK as usize);
    vec![
        ("f32", Dtype::F32, raw.to_vec()),
        ("int8", Dtype::I8, dequantize_weight(&q, &s, N, K)),
        ("nf4", Dtype::NF4, dequantize_weight_lut4(&nq, &ns, N, K, &NF4_LUT)),
        ("fp8 e4m3", Dtype::F8E4M3, dequantize_weight_fp8(&fq, &fs, N, K, Fp8Format::E4M3, FP8_BLOCK as usize)),
    ]
}

#[test]
fn read_host_returns_the_stored_values_of_every_tier_whatever_the_tile() {
    let raw = Lcg::new(data::rng::random_seed()).vec_scaled(N * K, 0.5);
    for budget in [u64::MAX / 8, 4 * K as u64 * 3] {
        let mut ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
        ops.set_tile_budget_bytes(budget);
        for (name, dtype, want) in cases(&raw) {
            let w = Weight::upload(&ops, &raw, N, K, dtype);
            if w.dtype() != dtype {
                continue; // the device does not run this tier natively
            }
            let got = ops.read_host(&w);
            assert_eq!(got, want, "{name} at a {budget} byte tile");
        }
    }
}
