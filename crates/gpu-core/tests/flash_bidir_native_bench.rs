// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed throughput of bidirectional flash attention at the head width
//! and sequence lengths FLUX.2 klein's DiT dispatches: the WGSL
//! `flash_attn_bidir_reg2` (generated tier) and the native `flash_bidir_f32`
//! it is redirected to on a CUDA device, each as a fraction of the device's
//! own fp32 fused-multiply-add peak measured in the same run.
//!
//! Swedish Embedded AB implements measured compute contracts for GPU inference
//! stacks. If your team needs expertise in knowing how close your attention
//! kernels run to the hardware's ceiling, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! A measurement, not a gate (that is `flash_bidir_native.rs`):
//!
//! ```text
//! cargo test --release -p brain-gpu-core --test flash_bidir_native_bench -- --device gpu1 --backend cuda
//! ```
//!
//! On a SHARED card every row is the MINIMUM over `TRIALS` device-timed
//! launches (a neighbour's time slice only ever lengthens one), the median
//! beside it, and the peak is an FMA chain this file launches itself,
//! interleaved trial by trial with the kernels so both see the same
//! neighbour. Each chain feeds its result back as the next operand, so no
//! compiler can fold it.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
    ("flash_attn_bidir_reg2_ref", kernels::FLASH_ATTN_BIDIR_REG2),
];
const K_NATIVE: usize = 0;
const K_WGSL: usize = 1;
const TRIALS: usize = 9;
const HD: u32 = 128;

/// fp32 FMA peak: eight independent loop-carried chains per thread; params
/// `[iters, b (f32 bits)]`, one float out per thread.
static PEAK: kernels_cuda::CudaKernel = kernels_cuda::CudaKernel {
    name: "bench_fma_peak",
    op: backend_api::select::Op::MatMul,
    weight: backend_api::select::Dtype::F32,
    by_name: true,
    source: backend_api::ImplSource::Tuned,
    min_cc: kernels_cuda::BASELINE_MIN_CC,
    entry: "brain_bench_fma_peak",
    what: "fp32 FMA peak probe",
    reported: "native:bench_fma_peak",
    block_dim: 256,
    tile: (1, 256),
    shared_bytes: 0,
    src: r#"
extern "C" __global__ void brain_bench_fma_peak(const unsigned int* params, float* out) {
    const unsigned int iters = params[0];
    const float b = __int_as_float((int)params[1]);
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    float a0 = (float)t, a1 = a0 + 1.f, a2 = a0 + 2.f, a3 = a0 + 3.f, a4 = a0 + 4.f, a5 = a0 + 5.f, a6 = a0 + 6.f, a7 = a0 + 7.f;
    for (unsigned int i = 0; i < iters; ++i) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            a0 = fmaf(a0, b, a1); a1 = fmaf(a1, b, a2); a2 = fmaf(a2, b, a3); a3 = fmaf(a3, b, a4);
            a4 = fmaf(a4, b, a5); a5 = fmaf(a5, b, a6); a6 = fmaf(a6, b, a7); a7 = fmaf(a7, b, a0);
        }
    }
    out[t] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
"#,
};
const PEAK_BINDINGS: &[backend_api::BindKind] = &[backend_api::BindKind::Uniform, backend_api::BindKind::StorageReadWrite];
const PEAK_BLOCKS: u32 = 2048;
const PEAK_ITERS: u32 = 64;

/// `(label, bsz, heads, T)`.
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("klein t2i 640x512, 1792 tokens", 1, 24, 1792),
    ("klein edit 640x512 + ref, 3072", 1, 24, 3072),
    ("klein t2i 1024x1024, 4608", 1, 24, 4608),
];

fn device_ms(gpu: &Gpu, step: &gpu_core::Step) -> f64 {
    gpu.reset_kernel_times();
    gpu.submit(&[], std::slice::from_ref(step));
    gpu.poll_wait();
    gpu.kernel_times().expect("kernel timing").iter().map(|r| r.1).sum()
}

fn min_median(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[0], v[v.len() / 2])
}

fn attention_throughput_against_the_fma_peak() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() || !gpu.set_kernel_timing(true) {
        brain_testutil::skip_unavailable("needs a CUDA device with kernel timing");
        return;
    }
    let peak = gpu.native_kernel(&PEAK, PEAK_BINDINGS).expect("the backend compiles the peak probe");
    let peak_out = gpu.storage(u64::from(PEAK_BLOCKS) * 256);
    let peak_step = gpu.step_native(peak, &[&peak_out], &[PEAK_ITERS, 0.999f32.to_bits()], PEAK_BLOCKS).expect("peak step");
    let peak_flop = 2.0 * 8.0 * 8.0 * f64::from(PEAK_ITERS) * f64::from(PEAK_BLOCKS) * 256.0;
    for _ in 0..20 {
        device_ms(&gpu, &peak_step);
    }
    println!("flash attention vs fp32 FMA peak, min/median of {TRIALS} device-timed launches");
    println!("{:<34} {:<16} {:>9} {:>9} {:>9} {:>10} {:>7}", "shape", "kernel", "min ms", "med ms", "TFLOP/s", "peak", "% peak");
    for &(label, bsz, heads, t) in SHAPES {
        let d = heads * HD;
        let p = [bsz, heads, t, HD, 3 * d, 0, d, 2 * d, d];
        let qkv = gpu.storage(u64::from(bsz * t) * u64::from(3 * d));
        let fill: Vec<f32> = (0..(bsz * t * 3 * d)).map(|i| ((i % 251) as f32 - 125.0) / 250.0).collect();
        gpu.write_f32(&qkv, &fill);
        let out = gpu.storage(u64::from(bsz * t) * u64::from(d));
        let wg = bsz * heads * t.div_ceil(128);
        let flop = 4.0 * f64::from(bsz * heads) * f64::from(t) * f64::from(t) * f64::from(HD);
        for (name, kind) in [("native", K_NATIVE), ("wgsl reg2", K_WGSL)] {
            let step = gpu.dispatch(kind, &[&qkv, &out], &p, Dispatch::Workgroups(wg));
            let (mut k, mut pk) = (Vec::new(), Vec::new());
            for _ in 0..TRIALS {
                pk.push(device_ms(&gpu, &peak_step));
                k.push(device_ms(&gpu, &step));
            }
            let ((kmin, kmed), (pmin, _)) = (min_median(k), min_median(pk));
            // FLOP per millisecond / 1e9 is TFLOP/s.
            let (rate, prate) = (flop / kmin / 1e9, peak_flop / pmin / 1e9);
            let kernel = gpu.native_kernel_for(kind, &p).unwrap_or(name);
            println!("{label:<34} {kernel:<16} {kmin:>9.3} {kmed:>9.3} {rate:>9.2} {prate:>10.2} {:>6.1}%", 100.0 * rate / prate);
        }
    }
    gpu.set_kernel_timing(false);
}

gpu_core::card_tests!(attention_throughput_against_the_fma_peak);
