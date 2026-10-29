// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the dequantize-in-register candidates of the int8 linear (`matmul_i8_dq_stream_*` for the decode and
//! many-row slots, `matmul_i8_dq_wgmma` for the Hopper prefill slot): int8 weight codes times their group scale become bf16 in
//! registers, the activations are bf16 rows, the accumulation is one f32 sum across K.
//!
//! Swedish Embedded AB implements tensor-core inference kernels for quantised language models. If your team needs
//! expertise in running int8 checkpoints on bf16 tensor cores without losing the stored scales, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The claim is not bit identity with another kernel (the arithmetic differs by design) but the precision admission rule
//! of `gpu_core::native_tune`: a candidate's error against the f64 reference of the STORED representation (fp32
//! activations as produced, weights `code * scale`) stays within eight times the error of rounding its own inputs to
//! their storage resolution. `native_check_shape` applies that rule to a synthetic problem of any shape, so this
//! gate covers what the autotuner's grid does not: ragged `n`, row counts at every tile edge, `K` of one group and of
//! many stages. The second test times every candidate (`#[ignore]`: a measurement; run it under `ncu` to see why).
//!
//! ```text
//! BRAIN_DEVICE=gpu1 BRAIN_BACKEND=cuda cargo test --release --offline -p brain-gpu-core --test i8_dequant_native -- --nocapture
//! ```

use gpu_core::{Gpu, NativeVerdict};

const KERNELS: &[(&str, &str)] = &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV), ("matmul_i8_dyn", kernels::MATMUL_I8_DYN)];

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::MMA_S8_MIN_CC)
}

fn is_dequant(name: &str) -> bool {
    name.starts_with("matmul_i8_dq_")
}

/// Every dequantize candidate that serves a shape passes the admission rule on it, and the packed candidates of the
/// slot still pass the same rule on the same problem. Returns the dequantize candidates that served some shape.
fn admitted_everywhere(gpu: &Gpu, shapes: &[[u32; 3]]) -> std::collections::BTreeSet<String> {
    let mut seen = std::collections::BTreeSet::new();
    for &params in shapes {
        let verdicts = gpu_core::native_check_shape(gpu, params);
        for (name, v) in &verdicts {
            if is_dequant(name) {
                seen.insert(name.clone());
            }
            assert!(!matches!(v, NativeVerdict::Rejected(_)), "{name} at m={} kg={} n={}: {v:?}", params[0], params[1], params[2]);
        }
        // The packed prefill GEMMs stage 64 elements of K, so a K of one scale group is theirs to decline.
        if params[0] <= backend_api::select::DECODE_REGIME_MAX_ROWS || params[1] % 16 == 0 {
            assert!(verdicts.iter().any(|(n, _)| !is_dequant(n)), "no packed candidate served {params:?}");
        }
    }
    seen
}

#[test]
fn the_dequantize_candidates_of_the_decode_slots_pass_the_admission_rule_on_ragged_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("the dequantize-in-register candidates need a CUDA device with bf16 tensor cores");
        return;
    }
    let mut shapes = Vec::new();
    // Rows at every tile boundary of 8, 16, 24, 32, widths with ragged weight-row tails and K from one stage to many.
    for m in [5u32, 8, 9, 16, 17, 24, 25, 32] {
        for (kg, n) in [(8u32, 17u32), (32, 9), (40, 64), (328, 33), (1280, 130), (4352, 64)] {
            shapes.push([m, kg, n]);
        }
    }
    // The many-row regime of a large decode batch.
    for m in [33u32, 48, 49, 64] {
        for (kg, n) in [(8u32, 17u32), (328, 65), (1280, 130)] {
            shapes.push([m, kg, n]);
        }
    }
    let served = admitted_everywhere(&gpu, &shapes);
    for family in ["matmul_i8_dq_stream_r1", "matmul_i8_dq_stream_q2"] {
        assert!(served.contains(family), "{family} served none of the shapes: {served:?}");
    }
}

#[test]
fn the_dequantize_gemm_of_the_prefill_slot_passes_the_admission_rule_on_ragged_shapes() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("the dequantize-in-register candidates need a CUDA device with bf16 tensor cores");
        return;
    }
    let mut shapes = Vec::new();
    for m in [33u32, 127, 128, 129, 255, 300, 1000] {
        for (kg, n) in [(8u32, 17u32), (40, 127), (328, 128), (328, 129), (1280, 260)] {
            shapes.push([m, kg, n]);
        }
    }
    let served = admitted_everywhere(&gpu, &shapes);
    assert!(served.contains("matmul_i8_dq_stream_q2"), "the wide streaming candidate served none of the shapes: {served:?}");
    if gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::WGMMA_MIN_CC) {
        assert!(served.contains("matmul_i8_dq_wgmma"), "the Hopper prefill GEMM served none of the shapes: {served:?}");
    }
}

/// Device time (`(rows, K, N)`: the 27B's gate/up and down projections) of every candidate that serves a shape, weights cycled past the L2 for the streaming shapes, as the
/// autotuner times them: the figure `ncu` is pointed at (`--kernel-name regex:dq`). `#[ignore]`: a measurement.
///
/// ```text
/// BRAIN_DEVICE=gpu1 BRAIN_BACKEND=cuda cargo test --release --offline -p brain-gpu-core --test i8_dequant_native -- --ignored --nocapture
/// ```
fn device_time(shapes: &[(u32, u32, u32)]) {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("the dequantize-in-register candidates need a CUDA device with bf16 tensor cores");
        return;
    }
    assert!(gpu.set_kernel_timing(true), "this device cannot time kernels");
    let names = ["matmul_i8_dq_stream_r1", "matmul_i8_dq_stream_r2", "matmul_i8_dq_stream_r4", "matmul_i8_dq_stream_q2", "matmul_i8_dq_stream_q4", "matmul_i8_dq_wgmma"];
    for &(m, k, n) in shapes {
        let (kg, groups) = (k / 4, (k / 32) as usize);
        let params = [m, kg, n];
        let x = gpu.storage(u64::from(m) * u64::from(kg) * 2);
        gpu.write(&x, &vec![0x3f80_3f80u32; (m * kg * 2) as usize]);
        let weight_bytes = u64::from(n) * u64::from(k);
        let copies = (192u64 << 20).div_ceil(weight_bytes).clamp(1, 16) as usize;
        let ws: Vec<_> = (0..copies)
            .map(|_| {
                let w = gpu.storage(u64::from(n) * u64::from(kg));
                let s = gpu.storage(u64::from(n) * groups as u64);
                gpu.write(&w, &vec![0x0102_0304u32; (n * kg) as usize]);
                gpu.write_f32(&s, &vec![0.01f32; n as usize * groups]);
                (w, s)
            })
            .collect();
        let out = gpu.storage(u64::from(m) * u64::from(n));
        for name in names {
            gpu_core::force_candidate(if m <= backend_api::select::DECODE_REGIME_MAX_ROWS { "matmul_i8_gemv" } else { "matmul_i8_prefill" }, name);
            let steps: Vec<_> = (0..copies.max(6))
                .filter_map(|i| {
                    let (w, s) = &ws[i % copies];
                    gpu.int8_dequant_step(&[&x, w, s, &out], &[(0, 0), (0, 0), (0, 0), (0, 0)], &params).filter(|(_, reported)| reported.ends_with(name)).map(|(st, _)| st)
                })
                .collect();
            if steps.is_empty() {
                continue;
            }
            let mut best = f64::MAX;
            for _ in 0..4 {
                gpu.reset_kernel_times();
                gpu.submit(&[], &steps);
                gpu.poll_wait();
                let ms: f64 = gpu.kernel_times().expect("kernel timing").iter().map(|r| r.1).sum::<f64>() / steps.len() as f64;
                best = best.min(ms);
            }
            let gbs = weight_bytes as f64 * 1.125 / (best * 1e-3) / 1e9;
            let tflops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k) / (best * 1e-3) / 1e12;
            println!("m={m:<5} K={k:<6} N={n:<6} {name:<24} {:>8.1} us  {gbs:>7.0} GB/s  {tflops:>6.1} TFLOPS", best * 1e3);
        }
    }
    gpu_core::clear_forced_candidates();
}

#[test]
#[ignore]
fn dequantize_candidates_device_time_at_decode_batches() {
    device_time(&[(8, 5120, 17408), (16, 5120, 17408), (32, 5120, 17408), (32, 17408, 5120), (48, 5120, 17408), (64, 17408, 5120), (96, 5120, 17408), (128, 17408, 5120)]);
}

#[test]
#[ignore]
fn dequantize_candidates_device_time_at_prefill_rounds() {
    device_time(&[(1024, 5120, 17408), (4096, 5120, 5120), (4096, 17408, 5120)]);
}
