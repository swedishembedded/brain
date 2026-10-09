// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed throughput of the int8 prefill/DiT GEMM (`matmul_i8_dyn`, the
//! native `matmul_i8_dp4a` on a CUDA device) at the shapes FLUX.2 klein's DiT
//! and its Qwen3 text encoder dispatch, as a fraction of the device's own
//! packed-int8 peak measured in the same run.
//!
//! Swedish Embedded AB implements measured compute contracts for GPU inference
//! stacks. If your team needs expertise in knowing how close your int8 kernels
//! run to the hardware's ceiling, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a correctness gate (that is
//! `i8_dyn_native.rs`). Run on a CUDA box:
//!
//! ```text
//! BRAIN_BACKEND=cuda cargo test --release --offline -p brain-gpu-core --test i8_dyn_native_bench -- --ignored --nocapture
//! ```
//!
//! What makes the numbers mean something on a SHARED card:
//!
//! * Time is the kernel's own, from the backend's per-launch device events
//!   (`Gpu::kernel_times`), never the host clock.
//! * Another context time-sliced onto the card stretches any launch it
//!   interrupts, and only ever makes a launch slower. So every row is the
//!   MINIMUM over `TRIALS` launches, with the median beside it.
//! * The peak is a packed-dot chain this file launches itself, interleaved
//!   trial by trial with the GEMMs so both see the same neighbour. The
//!   fraction of peak is the ratio of the two minima, which a uniform slowdown
//!   cancels out of; the absolute rates are printed too, and are only as good
//!   as the card's quietest moment.
//! * The peak chain feeds each result back as the next operand, so no
//!   compiler can hoist the dot out of the loop and report additions as int8
//!   throughput.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("matmul_i8_dyn", kernels::MATMUL_I8_DYN)];
const TRIALS: usize = 15;

/// Packed-int8 peak: eight independent loop-carried `__dp4a` chains per
/// thread; params `[iters, b]`, one float out per thread.
static PEAK: kernels_cuda::CudaKernel = kernels_cuda::CudaKernel {
    name: "bench_dp4a_peak",
    op: backend_api::select::Op::MatMul,
    weight: backend_api::select::Dtype::I8,
    by_name: true,
    source: backend_api::ImplSource::Tuned,
    min_cc: kernels_cuda::DP4A_MIN_CC,
    entry: "brain_bench_dp4a_peak",
    what: "packed-int8 peak probe",
    reported: "native:bench_dp4a_peak",
    block_dim: 256,
    tile: (1, 256),
    shared_bytes: 0,
    src: r#"
extern "C" __global__ void brain_bench_dp4a_peak(const unsigned int* params, float* out) {
    const unsigned int iters = params[0];
    const int b = (int)params[1];
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    int a0 = (int)t, a1 = a0 ^ 1, a2 = a0 ^ 2, a3 = a0 ^ 3, a4 = a0 ^ 4, a5 = a0 ^ 5, a6 = a0 ^ 6, a7 = a0 ^ 7;
    for (unsigned int i = 0; i < iters; ++i) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            a0 = __dp4a(a0, b, a0); a1 = __dp4a(a1, b, a1); a2 = __dp4a(a2, b, a2); a3 = __dp4a(a3, b, a3);
            a4 = __dp4a(a4, b, a4); a5 = __dp4a(a5, b, a5); a6 = __dp4a(a6, b, a6); a7 = __dp4a(a7, b, a7);
        }
    }
    out[t] = (float)(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
}
"#,
};
const PEAK_BINDINGS: &[backend_api::BindKind] = &[backend_api::BindKind::Uniform, backend_api::BindKind::StorageReadWrite];
const PEAK_BLOCKS: u32 = 4096;
const PEAK_ITERS: u32 = 160;

/// `(label, m, K, N)`.
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("dit joint qkv/o 1792x3072x3072", 1792, 3072, 3072),
    ("dit joint mlp in 1792x3072x9216", 1792, 3072, 9216),
    ("dit joint mlp out 1792x9216x3072", 1792, 9216, 3072),
    ("dit img mlp in 1280x3072x9216", 1280, 3072, 9216),
    ("dit txt qkv 512x3072x3072", 512, 3072, 3072),
    ("te q 512x2560x4096", 512, 2560, 4096),
    ("te kv 512x2560x1024", 512, 2560, 1024),
    ("te gate/up 512x2560x9728", 512, 2560, 9728),
    ("te down 512x9728x2560", 512, 9728, 2560),
];

/// Device milliseconds of everything submitted in `steps`, one trial.
fn device_ms(gpu: &Gpu, steps: &[gpu_core::Step]) -> f64 {
    gpu.reset_kernel_times();
    gpu.submit(&[], steps);
    gpu.poll_wait();
    gpu.kernel_times().expect("kernel timing").iter().map(|r| r.1).sum()
}

fn min_median(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[0], v[v.len() / 2])
}

#[test]
#[ignore]
fn int8_gemm_throughput_against_the_measured_packed_dot_peak() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let Some(cc) = gpu.caps().arch.compute_capability.filter(|_| gpu.kind() == "cuda") else {
        brain_testutil::skip_unavailable("needs a CUDA device");
        return;
    };
    if cc < kernels_cuda::DP4A_MIN_CC || !gpu.set_kernel_timing(true) {
        brain_testutil::skip_unavailable("needs a packed int8 dot and kernel timing");
        return;
    }
    let peak = gpu.native_kernel(&PEAK, PEAK_BINDINGS).expect("the backend compiles the peak probe");
    let peak_out = gpu.storage(u64::from(PEAK_BLOCKS) * 256);
    let peak_step = gpu.step_native(peak, &[&peak_out], &[PEAK_ITERS, 0x0102_0304], PEAK_BLOCKS).expect("peak step");
    let peak_ops = 2.0 * 4.0 * 8.0 * 4.0 * f64::from(PEAK_ITERS) * f64::from(PEAK_BLOCKS) * 256.0;
    // Ramp the clocks before anything is timed.
    for _ in 0..20 {
        device_ms(&gpu, std::slice::from_ref(&peak_step));
    }

    println!("int8 GEMM vs packed-dot peak, min/median of {TRIALS} device-timed launches ({})", gpu.native_kernel_for(0, &[1792, 768, 3072]).unwrap_or("wgsl tier"));
    println!("{:<36} {:>9} {:>9} {:>10} {:>10} {:>8}", "shape", "min ms", "med ms", "TOP/s", "peak TOP/s", "% peak");
    for &(label, m, k, n) in SHAPES {
        let kg = k / 4;
        let x = gpu.storage(u64::from(m) * u64::from(kg));
        let w = gpu.storage(u64::from(n) * u64::from(kg));
        let sx = gpu.storage(u64::from(m));
        let sw = gpu.storage(u64::from(n) * u64::from(kg / 8));
        let out = gpu.storage(u64::from(m) * u64::from(n));
        let step = gpu.dispatch(0, &[&x, &w, &sx, &sw, &out], &[m, kg, n], Dispatch::Workgroups(m.div_ceil(128) * n.div_ceil(128)));
        let (mut g, mut p) = (Vec::new(), Vec::new());
        for _ in 0..TRIALS {
            p.push(device_ms(&gpu, std::slice::from_ref(&peak_step)));
            g.push(device_ms(&gpu, std::slice::from_ref(&step)));
        }
        let ((gmin, gmed), (pmin, _)) = (min_median(g), min_median(p));
        let ops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
        let (rate, peak_rate) = (ops / gmin / 1e9, peak_ops / pmin / 1e9);
        println!("{label:<36} {gmin:>9.3} {gmed:>9.3} {rate:>10.2} {peak_rate:>10.2} {:>7.1}%", 100.0 * rate / peak_rate);
    }
    gpu.set_kernel_timing(false);
}
