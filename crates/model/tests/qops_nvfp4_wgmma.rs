// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements prompt-processing throughput for 4-bit language-model releases for its clients. If your
// team needs expertise in running NVFP4 checkpoints on Hopper tensor cores without a converted copy then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: an NVFP4 weight multiplied against 17 or more activation rows of f32 runs on the FP8 tensor cores (the activations quantised to E4M3
//! with one scale per row, the weight's codes and block scales decoded to E4M3 in shared memory) and agrees with the host product of the exact
//! decoded weight to the operand precision: both operands carry a 2^-4 relative rounding error a term (E4M3's four significant bits), so the
//! sum is held to `OPERAND_REL` of the root of its sum of squared terms. Row windows that do not start at 0, ragged row counts and a ragged
//! last band of weight rows are covered, and every output element is written (the output starts as NaN).

use checkpoint::packed::{QuantEncoding, TensorDescriptor, TensorSink, Tile};
use data::rng::Lcg;
use gpu_core::Gpu;
use model::ops::{training_kernel_list, DeviceWeightSink, Ops, Weight};

/// The root of a sum of N independent terms of relative error 2^-4 each deviates by 2^-4 of the root of their squares; five such roots are the bound.
const OPERAND_REL: f64 = 0.2;

fn decode_weight(ops: &Ops, w: &Weight) -> Vec<f32> {
    let g = ops.gpu();
    let out = g.storage((w.n() * w.k()) as u64);
    let mut steps = Vec::new();
    ops.decode_tile(&mut steps, w, 0, w.n(), &out);
    g.submit(&[], &steps);
    g.read(&out, (w.n() * w.k()) as usize)
}

fn nvfp4_weight(ops: &Ops, n: usize, k: usize) -> Weight {
    let enc = QuantEncoding::nvfp4();
    let desc = TensorDescriptor::new("w", vec![n as u64, k as u64], enc);
    let mut rng = Lcg::new(data::rng::random_seed());
    let payload: Vec<u8> = (0..desc.payload_bytes()).map(|_| rng.next_u32() as u8).collect();
    let scales = [0.0071f32.to_le_bytes().to_vec(), (0..desc.scale_bytes(1)).map(|_| 0x30 + (rng.next_u32() % 0x20) as u8).collect::<Vec<u8>>()];
    let tile = Tile { elems: 0..desc.numel(), payload: &payload, scales: vec![&scales[0], &scales[1]] };
    let mut sink = DeviceWeightSink::new(ops);
    sink.begin(&desc).unwrap();
    sink.write_tile(&desc, &tile).unwrap();
    sink.finish(&desc).unwrap();
    sink.take_weight().unwrap()
}

#[test]
fn an_nvfp4_weight_against_many_rows_runs_on_the_fp8_tensor_cores_and_agrees_to_the_operand_precision() {
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    if ops.gpu().kind() != "cuda" || !ops.gpu().has_fused(gpu_core::Fused::MatmulNvfp4Wgmma) {
        return brain_testutil::skip_unavailable("the NVFP4 tensor-core product needs a CUDA device with warpgroup tensor cores");
    }
    for (n, k) in [(300usize, 512usize), (13_000, 256)] {
        let w = nvfp4_weight(&ops, n, k);
        let wd = decode_weight(&ops, &w);
        let g = ops.gpu();
        let mut rng = Lcg::new(data::rng::random_seed());
        for (m, first, count) in [(17usize, 0usize, n), (33, 8, n - 16), (130, 0, n), (300, 2, n - 3)] {
            let x = rng.vec_scaled(m * k, 1.0);
            let xb = g.storage_init("x", &x);
            let y = g.storage_init("y", &vec![f32::NAN; m * count]);
            let mut steps = Vec::new();
            let act = ops.act_f32(&xb, 0, m as u32, k as u32);
            ops.matmul_rows(&mut steps, &w, (first as u32, count as u32), &act, &y, 0);
            let launches = steps.len();
            g.submit(&[], &steps);
            let got = g.read(&y, m * count);
            assert!(launches <= 3, "the row quantisation and the product, not a tile decode a band of weight rows: {launches} steps");
            for mi in (0..m).step_by(7) {
                for c in (0..count).step_by(5) {
                    let terms: Vec<f64> = (0..k).map(|i| f64::from(x[mi * k + i]) * f64::from(wd[(first + c) * k + i])).collect();
                    let (sum, scale) = (terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt());
                    let a = f64::from(got[mi * count + c]);
                    assert!(a.is_finite(), "n={n} k={k} m={m} first={first}: output ({mi},{c}) not written");
                    assert!((a - sum).abs() <= OPERAND_REL * scale + 1e-6, "n={n} k={k} m={m} first={first}: output ({mi},{c}) is {a}, host product {sum} (scale {scale})");
                }
            }
        }
    }
}
