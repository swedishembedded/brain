// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native candidates of the FP8 linear (`matmul_fp8_dq_stream_*` for decode batches up to 128 rows,
//! `matmul_fp8_dq_wgmma` for the Hopper rounds): E4M3 weight bytes times their 128 x 128 block scale become bf16 in
//! registers (or shared memory), the activations are bf16 rows, the accumulation is one f32 sum across K.
//!
//! Swedish Embedded AB implements tensor-core inference kernels for FP8 checkpoints for its clients. If your team needs
//! expertise in serving FP8 releases natively on Hopper without converting them, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! The claim is the precision admission rule of `gpu_core::native_tune`: a candidate's error against the f64 reference of
//! the STORED representation (fp32 activations as produced, weights `decode(byte) * block scale`) stays within eight times
//! the error of rounding its own inputs to their storage resolution (bf16 rows, bf16 weights). `native_check_fp8_shape`
//! applies it to a synthetic problem of any shape, so this gate covers what the autotuner's grid does not: ragged `n`
//! (a last band of fewer than 128 weight rows), row counts at every tile edge, `K` of one scale block and of many. The random
//! weight bytes cover every finite E4M3 code, the subnormals (`0x01..=0x07`) included, which the decode must get exactly.
//!
//! ```text
//! BRAIN_DEVICE=gpu1 BRAIN_BACKEND=cuda cargo test --release --offline -p brain-gpu-core --test fp8_linear_native -- --nocapture
//! ```

use gpu_core::{Gpu, NativeVerdict};

const KERNELS: &[(&str, &str)] = &[("matmul_fp8_gemv", kernels::MATMUL_FP8_GEMV)];

fn is_cuda_hopper(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::WGMMA_MIN_CC)
}

/// Every candidate that serves a shape passes the admission rule on it. Returns the candidates that served some shape.
fn admitted_everywhere(gpu: &Gpu, shapes: &[[u32; 3]]) -> std::collections::BTreeSet<String> {
    let mut seen = std::collections::BTreeSet::new();
    for &params in shapes {
        let verdicts = gpu_core::native_check_fp8_shape(gpu, params);
        assert!(!verdicts.is_empty(), "no FP8 candidate serves m={} kg={} n={}", params[0], params[1], params[2]);
        for (name, v) in &verdicts {
            seen.insert(name.clone());
            assert!(!matches!(v, NativeVerdict::Rejected(_)), "{name} at m={} kg={} n={}: {v:?}", params[0], params[1], params[2]);
        }
    }
    seen
}

#[test]
fn the_streaming_candidates_pass_the_admission_rule_on_ragged_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    let mut shapes = Vec::new();
    // Rows at every tile boundary of 8, 16, 24, 32, 48, 64, widths with a ragged last band and K from one scale block to many.
    for m in [1u32, 2, 5, 8, 9, 16, 17, 24, 25, 32, 33, 48, 49, 64] {
        for (kg, n) in [(32u32, 17u32), (32, 128), (64, 9), (160, 64), (1280, 130), (1280, 5120), (4352, 64)] {
            shapes.push([m, kg, n]);
        }
    }
    // The wide streaming kernels.
    for m in [65u32, 96, 97, 128] {
        for (kg, n) in [(32u32, 17u32), (1280, 130), (1280, 260)] {
            shapes.push([m, kg, n]);
        }
    }
    let served = admitted_everywhere(&gpu, &shapes);
    for kernel in ["matmul_fp8_dq_stream_r1", "matmul_fp8_dq_stream_r2", "matmul_fp8_dq_stream_r4", "matmul_fp8_dq_stream_q2", "matmul_fp8_dq_stream_q4"] {
        assert!(served.contains(kernel), "{kernel} served none of the shapes: {served:?}");
    }
}

#[test]
fn the_wgmma_gemms_pass_the_admission_rule_on_ragged_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    let mut shapes = Vec::new();
    for m in [33u32, 127, 128, 129, 255, 300, 1000] {
        for (kg, n) in [(32u32, 17u32), (32, 127), (160, 128), (160, 129), (1280, 260), (4352, 130)] {
            shapes.push([m, kg, n]);
        }
    }
    let served = admitted_everywhere(&gpu, &shapes);
    for kernel in ["matmul_fp8_dq_wgmma", "matmul_fp8_wgmma"] {
        assert!(served.contains(kernel), "{kernel} served none of the shapes: {served:?}");
    }
}
