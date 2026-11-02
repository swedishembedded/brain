// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements prompt-processing throughput for 4-bit language-model releases for its clients. If your
// team needs expertise in running NVFP4 checkpoints on Hopper tensor cores without a converted copy then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: an NVFP4 weight multiplied against 2 or more activation rows of f32 runs on the FP8 tensor cores (the activations quantised to E4M3
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
        for (m, first, count) in [(2usize, 0usize, n), (9, 3, n - 7), (17, 0, n), (24, 8, n - 16), (33, 8, n - 16), (130, 0, n), (300, 2, n - 3)] {
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

/// Names of the kernels one product over `m` rows of `w` launched, by the device's own kernel timing.
fn kernels_run(ops: &Ops, w: &Weight, m: usize) -> Vec<String> {
    let (g, k, n) = (ops.gpu(), w.k() as usize, w.n());
    let mut rng = Lcg::new(data::rng::random_seed());
    let xb = g.storage_init("x", &rng.vec_scaled(m * k, 1.0));
    let y = g.storage_init("y", &vec![f32::NAN; m * n as usize]);
    let act = ops.act_f32(&xb, 0, m as u32, k as u32);
    let mut steps = Vec::new();
    ops.matmul_rows(&mut steps, w, (0, n), &act, &y, 0);
    g.set_kernel_timing(true);
    g.reset_kernel_times();
    g.submit(&[], &steps);
    g.poll_wait();
    let names = g.kernel_times().expect("the device times its kernels").into_iter().filter(|e| e.2 > 0).map(|e| e.0).collect();
    g.set_kernel_timing(false);
    names
}

#[test]
fn linear_activations_choose_the_bf16_or_the_e4m3_build_of_an_nvfp4_product() {
    use model::policy::LinearActivations;
    let mut ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    if ops.gpu().kind() != "cuda" || !ops.gpu().has_fused(gpu_core::Fused::MatmulNvfp4MmaE8) || !ops.gpu().has_fused(gpu_core::Fused::MatmulNvfp4Mma8) {
        return brain_testutil::skip_unavailable("the NVFP4 tensor-core products need a CUDA device");
    }
    let w = nvfp4_weight(&ops, 512, 512);
    let ran = |ops: &Ops, m| kernels_run(ops, &w, m).join(" ");
    assert_eq!(ops.linear_activations(), LinearActivations::Auto);
    for (m, bf16_build, e4m3_build) in [(4usize, "native:matmul_nvfp4_mma8", "native:matmul_nvfp4_mmae8"), (12, "native:matmul_nvfp4_mma16", "native:matmul_nvfp4_mmae16")] {
        ops.set_linear_activations(LinearActivations::Bf16);
        let bf16 = ran(&ops, m);
        assert!(bf16.contains(bf16_build) && !bf16.contains("mmae") && !bf16.contains("wgmma"), "bf16 at {m} rows ran: {bf16}");
        ops.set_linear_activations(LinearActivations::E4m3);
        let e4m3 = ran(&ops, m);
        assert!(e4m3.contains(e4m3_build) && !e4m3.contains(bf16_build), "e4m3 at {m} rows ran: {e4m3}");
        ops.set_linear_activations(LinearActivations::Auto);
        assert_eq!(ran(&ops, m), bf16, "auto is the bf16 path at {m} rows: precision below the competitors' is opt-in");
    }
    if ops.gpu().has_fused(gpu_core::Fused::MatmulNvfp4Wgmma) {
        ops.set_linear_activations(LinearActivations::E4m3);
        assert!(ran(&ops, 130).contains("native:matmul_nvfp4_wgmma"), "e4m3 at 130 rows takes the warpgroup GEMM");
        ops.set_linear_activations(LinearActivations::Bf16);
        let bf16 = ran(&ops, 130);
        assert!(!bf16.contains("wgmma") && !bf16.contains("mmae"), "bf16 at 130 rows (past the mma builds) ran: {bf16}");
    }
}
