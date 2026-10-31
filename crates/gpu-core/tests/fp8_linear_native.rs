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
    for kernel in [
        "matmul_fp8_gemv_f16_a1w4d3b3",
        "matmul_fp8_gemv_f16_a1w4d4b5",
        "matmul_fp8_gemv_f16_a1w2d6b6",
        "matmul_fp8_gemv_f16_a2w4d4b3",
        "matmul_fp8_gemv_f16x_t8w8d2b1",
        "matmul_fp8_gemv_f16x_t4w4d3b3",
        "matmul_fp8_wgmma_swap_n64b64",
        "matmul_fp8_wgmma_swap_n64b64s2",
        "matmul_fp8_wgmma_swap_n64b64s4",
        "matmul_fp8_wgmma_swap_n32b64",
        "matmul_fp8_wgmma_swap_n32b64s2",
        "matmul_fp8_wgmma_swap_n32b64s4",
        "matmul_fp8_wgmma_swap_n64b128s2",
        "matmul_fp8_wgmma_swap_tma_bn80",
        "matmul_fp8_wgmma_swap_tma_bn96",
        "matmul_fp8_wgmma_swap_tma_bn136",
        "matmul_fp8_wgmma_swap_tma_bn136c1",
        "matmul_fp8_wgmma_swap_tma_bn144c1",
    ] {
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
    for m in [33u32, 127, 128, 129, 255, 300, 1000, 2300] {
        for (kg, n) in [(32u32, 17u32), (32, 127), (160, 128), (160, 129), (1280, 260), (4352, 130)] {
            shapes.push([m, kg, n]);
        }
    }
    let served = admitted_everywhere(&gpu, &shapes);
    for kernel in ["matmul_fp8_dq_wgmma", "matmul_fp8_wgmma", "matmul_fp8_wgmma_tma", "matmul_fp8_wgmma_tma_n256"] {
        assert!(served.contains(kernel), "{kernel} served none of the shapes: {served:?}");
    }
}

/// The tensor-memory-accelerator decode candidates at the widths and depths of the served 27B (tile counts that wrap the persistent grid more than
/// once, a long K that keeps the ring turning over, a last tile that ends inside a scale block), at the row counts a batch reaches.
#[test]
fn the_tma_decode_candidates_pass_the_admission_rule_on_the_served_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    let mut shapes = Vec::new();
    for m in [17u32, 33, 48, 64] {
        for (kg, n) in [(1280u32, 34816u32), (1280, 17408), (1280, 10240), (1280, 12288), (4352, 5120), (1536, 5120), (1280, 6144), (1280, 1000)] {
            shapes.push([m, kg, n]);
        }
    }
    let served = admitted_everywhere(&gpu, &shapes);
    for kernel in ["matmul_fp8_wgmma_swap_tma_bn80", "matmul_fp8_wgmma_swap_tma_bn96", "matmul_fp8_wgmma_swap_tma_bn136", "matmul_fp8_wgmma_swap_tma_bn136c1", "matmul_fp8_wgmma_swap_tma_bn144c1"] {
        assert!(served.contains(kernel), "{kernel} served none of the shapes: {served:?}");
    }
}

/// The bf16-output twin of the large-tile GEMM stores exactly their base's f32 rows rounded to bf16 - ragged rows, ragged `n`
/// (a last band of fewer than 128 or 256 weight rows) and K of one scale block and of many: they differ in the store alone.
#[test]
fn the_bf16_output_twins_store_their_bases_rows_rounded_to_bf16() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    let mut seen = std::collections::BTreeSet::new();
    for m in [33u32, 127, 129, 300, 768, 1000] {
        for (kg, n) in [(32u32, 18u32), (32, 128), (160, 130), (1280, 264), (1280, 5120), (4352, 1032)] {
            for (twin, differ) in gpu_core::native_check_fp8_bf16_out(&gpu, [m, kg, n]) {
                assert_eq!(differ, 0, "{twin} at m={m} kg={kg} n={n}: {differ} elements differ from the base rounded to bf16");
                seen.insert(twin);
            }
        }
    }
    assert!(seen.contains("matmul_fp8_wgmma_tma_n256_bf16o"), "the twin served none of the shapes: {seen:?}");
}

/// Achieved weight bandwidth of every FP8 candidate at the production decode shapes (`#[ignore]`: a measurement).
///
/// ```text
/// make test/one P=gpu-core T=fp8_linear_native ARGS="--ignored --nocapture fp8_decode_bandwidth"
/// ```
#[test]
#[ignore]
fn fp8_decode_bandwidth() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    println!("read roof (GB/s, launch floor us): {:?}", gpu_core::native_read_roof(&gpu));
    for (label, k, n) in [("gate/up 5120x34816", 5120u32, 34816u32), ("gate 5120x17408", 5120, 17408), ("down 17408x5120", 17408, 5120), ("qkv 5120x10240", 5120, 10240), ("out 6144x5120", 6144, 5120)] {
        for m in [1u32, 4, 16] {
            for (name, v) in gpu_core::native_time_fp8_shape(&gpu, [m, k / 4, n]) {
                println!("{label:22} m={m:<3} {name:28} {v:?}");
            }
        }
    }
}

/// Achieved weight bandwidth (rows 32-64) and tensor-core rate (prefill rows) of every FP8 candidate at the production
/// shapes (`#[ignore]`: a measurement).
///
/// ```text
/// make test/one P=brain-gpu-core T=fp8_linear_native ARGS="--ignored --nocapture fp8_batch_and_prefill_rate"
/// ```
#[test]
#[ignore]
fn fp8_batch_and_prefill_rate() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    let shapes = [("gate/up 5120x34816", 5120u32, 34816u32), ("gate 5120x17408", 5120, 17408), ("down 17408x5120", 17408, 5120), ("qkv 5120x10240", 5120, 10240), ("out 6144x5120", 6144, 5120)];
    for rows in [&[32u32, 48, 64][..], &[512, 768, 2048, 8192][..]] {
        for (label, k, n) in shapes {
            for &m in rows {
                for (name, v) in gpu_core::native_time_fp8_shape(&gpu, [m, k / 4, n]) {
                    println!("{label:22} m={m:<5} {name:28} {v:?}");
                }
            }
        }
    }
}

/// Tensor-core rate (TFLOPS) of the persistent TMA candidates at the production prefill shapes (`#[ignore]`: a measurement).
///
/// ```text
/// make test/one P=brain-gpu-core T=fp8_linear_native ARGS="--ignored --nocapture fp8_prefill_tflops"
/// ```
#[test]
#[ignore]
fn fp8_prefill_tflops() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    let shapes = [("gate/up 5120x34816", 5120u32, 34816u32), ("down 17408x5120", 17408, 5120), ("qkv 5120x10240", 5120, 10240), ("qkv 5120x12288", 5120, 12288), ("out 6144x5120", 6144, 5120)];
    for m in [768u32, 2048, 8192, 31000] {
        for (label, k, n) in shapes {
            let flops = 2.0 * f64::from(m) * f64::from(k) * f64::from(n);
            for (name, v) in gpu_core::native_time_fp8_candidates(&gpu, [m, k / 4, n], "tma") {
                match v {
                    gpu_core::NativeVerdict::Timed { us, .. } => println!("{label:20} m={m:<6} {name:28} {us:9.1} us {:7.1} TFLOPS", flops / us / 1e6),
                    gpu_core::NativeVerdict::Rejected(why) => println!("{label:20} m={m:<6} {name:28} REJECTED {why}"),
                }
            }
        }
    }
}

/// The bf16-output twins against their bases at the mixed-iteration rows (`#[ignore]`: a measurement).
///
/// ```text
/// make test/one P=brain-gpu-core T=fp8_linear_native ARGS="--ignored --nocapture fp8_bf16_out_rate"
/// ```
#[test]
#[ignore]
fn fp8_bf16_out_rate() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    let shapes = [("gate/up 5120x17408", 5120u32, 17408u32), ("down 17408x5120", 17408, 5120), ("qkv 5120x10240", 5120, 10240), ("out 6144x5120", 6144, 5120)];
    for m in [640u32, 768, 2048, 8192] {
        for (label, k, n) in shapes {
            for (twin, base, bf16) in gpu_core::native_time_fp8_bf16_out(&gpu, [m, k / 4, n]) {
                println!("{label:20} m={m:<5} {twin:34} f32 out {base:8.1} us   bf16 out {bf16:8.1} us   {:+.1} %", 100.0 * (bf16 / base - 1.0));
            }
        }
    }
}

/// One candidate at one shape, for `ncu -k regex:<kernel> -c 1` (`#[ignore]`: a profiling entry). Edit the two constants.
#[test]
#[ignore]
fn fp8_one_candidate() {
    const CANDIDATE: &str = "matmul_fp8_wgmma_tma_n256";
    const SHAPE: [u32; 3] = [8192, 5120 / 4, 34816];
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda_hopper(&gpu) {
        brain_testutil::skip_unavailable("the FP8 candidates are registered for a CUDA device with warpgroup tensor cores");
        return;
    }
    for (name, v) in gpu_core::native_time_fp8_candidates(&gpu, SHAPE, CANDIDATE) {
        println!("{name:28} {v:?}");
    }
}
