// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed weight bandwidth of the int8 projections of a batched decode
//! step (9 to 32 rows of x) at the Qwen3.8-27B projection shapes: the tiled
//! tensor-core GEMM a batch above the GEMV's crossover takes (`GEMM`) against
//! the candidate the device's tuned table selects for the same dispatch (`SELECTED`).
//!
//! Swedish Embedded AB implements measured memory-bandwidth contracts for GPU
//! inference stacks. If your team needs expertise in making batched decode
//! stream quantised weights at the speed of the memory system, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a correctness gate. Run on a CUDA box:
//!
//! ```text
//! cargo test --release --offline -p brain-gpu-core --test i8_decode_batch_bench -- --ignored --nocapture
//! ```
//!
//! The two paths are timed
//! trial by trial in turn, so a neighbour on the card hits both alike, and
//! every row is the best of `TRIALS` (contention only ever slows a launch).
//! Weights are cycled through copies past the L2 so nothing is read from cache.

use backend_api::BindKind;
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV)];
const FOOTPRINT_BYTES: u64 = 768 << 20;
const TRIALS: usize = 15;
const GEMM_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `(label, K, N)` - the Qwen3.8-27B decode projections as the GGUF lists them.
const SHAPES: &[(&str, u32, u32)] = &[
    ("gdn/attn k-proj 5120x1024", 5120, 1024),
    ("ffn narrow 5120x6144", 5120, 6144),
    ("attn out 6144x5120", 6144, 5120),
    ("gdn qkv 5120x10240", 5120, 10240),
    ("attn q 5120x12288", 5120, 12288),
    ("ffn down 17408x5120", 17408, 5120),
    ("ffn gate/up 5120x17408", 5120, 17408),
];

fn gemm_kernel_name(m: u32) -> &'static str {
    match m {
        0..=16 => "matmul_i8_mma_m16",
        17..=32 => "matmul_i8_mma_m32",
        _ => "matmul_i8_mma_m128",
    }
}

#[test]
#[ignore]
fn weight_bandwidth_of_a_batched_decode_step() {
    let gpu = Gpu::new(KERNELS);
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    let rows = [9u32, 12, 16, 20, 24, 32];
    println!("{:<26} {:>3} {:<24} {:>8} {:>8} | {:<24} {:>8} {:>8} | {:>6}", "shape", "m", "GEMM", "best us", "GB/s", "SELECTED", "best us", "GB/s", "speedup");
    for m in rows {
        for &(label, k, n) in SHAPES {
            let kg = k / 4;
            let wbytes = u64::from(n) * u64::from(kg) * 4 + u64::from(n) * u64::from(kg / 8) * 4;
            let copies = FOOTPRINT_BYTES.div_ceil(wbytes).max(1) as usize;
            let xq = gpu.storage(u64::from(m * kg));
            let sx = gpu.storage(u64::from(m * kg / 8));
            let out = gpu.storage(u64::from(m * n));
            gpu.write(&xq, &vec![0x01020304u32; (m * kg) as usize]);
            gpu.write_f32(&sx, &vec![1.0; (m * kg / 8) as usize]);
            let weights: Vec<_> = (0..copies)
                .map(|i| {
                    let w = gpu.storage(u64::from(n) * u64::from(kg));
                    let s = gpu.storage(u64::from(n) * u64::from(kg / 8));
                    if i == 0 {
                        gpu.write(&w, &vec![0x7f01fe03u32; (u64::from(n) * u64::from(kg)).min(1 << 22) as usize]);
                        gpu.write_f32(&s, &vec![0.01; (u64::from(n) * u64::from(kg / 8)).min(1 << 20) as usize]);
                    }
                    (w, s)
                })
                .collect();
            let spec = kernels_cuda::get(gemm_kernel_name(m)).expect("registered");
            let id = gpu.native_kernel(spec, GEMM_BINDINGS).expect("the backend compiles the GEMM");
            let blocks = spec.blocks_for(m, n);
            let gemm: Vec<_> = weights
                .iter()
                .map(|(w, s)| gpu.step_native(id, &[&xq, w, &sx, s, &out], &[m, kg, n], blocks).expect("native step"))
                .collect();
            let gemv: Vec<_> = weights.iter().map(|(w, s)| gpu.dispatch(0, &[&xq, w, &sx, s, &out], &[m, kg, n], Dispatch::Workgroups(n))).collect();
            let time = |steps: &[gpu_core::Step]| {
                gpu.reset_kernel_times();
                gpu.submit(&[], steps);
                gpu.poll_wait();
                gpu.kernel_times().expect("device timing").iter().map(|r| r.1).sum::<f64>() / steps.len() as f64
            };
            time(&gemm);
            time(&gemv);
            let (mut best_gemm, mut best_gemv) = (f64::MAX, f64::MAX);
            for _ in 0..TRIALS {
                best_gemm = best_gemm.min(time(&gemm));
                best_gemv = best_gemv.min(time(&gemv));
            }
            let gbs = |ms: f64| wbytes as f64 / (ms * 1e-3) / 1e9;
            println!(
                "{label:<26} {m:>3} {:<24} {:>8.1} {:>8.0} | {:<24} {:>8.1} {:>8.0} | {:>5.2}x",
                spec.name,
                best_gemm * 1e3,
                gbs(best_gemm),
                gpu.native_kernel_for(0, &[m, kg, n]).unwrap_or("wgsl"),
                best_gemv * 1e3,
                gbs(best_gemv),
                best_gemm / best_gemv
            );
        }
    }
    gpu.set_kernel_timing(false);
}

/// Print the device's native kernel selection table: what the autotuner chose per shape class and how close each
/// choice runs to the measured roofline. `--ignored --nocapture selection_report`.
#[test]
#[ignore]
fn selection_report() {
    let gpu = Gpu::new(KERNELS);
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("native selection is the CUDA backend's");
        return;
    }
    print!("{}", gpu_core::native_selection_report());
}
