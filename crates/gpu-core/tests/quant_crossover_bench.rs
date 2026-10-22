// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed GEMV against GEMM for quantized weights at decode row counts
//! (the int8 tier and Q4_K), at the real Qwen3.8-27B projection shapes: where
//! a batched decode step should hand its rows from the skinny-M GEMV to the
//! tiled GEMM on this device.
//!
//! Swedish Embedded AB implements measured kernel-selection policies for GPU
//! inference stacks. If your team needs expertise in batching decode on the
//! hardware it already owns, you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a gate. Run on a CUDA box:
//!
//! ```text
//! cargo test --release -p brain-gpu-core --test quant_crossover_bench -- --ignored --nocapture
//! ```
//!
//! Times are the kernels' own device events, best of several trials, each
//! trial cycling through enough weight copies to defeat the L2 (decode reads
//! every weight from memory).

use gpu_core::{DeviceBuffer, Dispatch, Gpu};

/// Distinct weight bytes cycled through per trial, well past any L2.
const FOOTPRINT_BYTES: u64 = 768 << 20;
const TRIALS: usize = 5;
const MIN_CALLS: usize = 16;

/// `(label, K, N)`.
const SHAPES: &[(&str, u32, u32)] = &[("gdn qkv 5120x10240", 5120, 10240), ("ffn gate/up 5120x17408", 5120, 17408), ("ffn down 17408x5120", 17408, 5120)];

/// A weight format: its two kernels, and how one weight copy is laid out.
struct Format {
    label: &'static str,
    gemv: usize,
    gemm: usize,
    /// u32 words of each weight buffer for `(n, k)`, in binding order.
    planes: fn(u32, u32) -> Vec<u64>,
    /// The kernels' `params` for `(m, k, n)`.
    params: fn(u32, u32, u32) -> [u32; 3],
    /// Whether the activation group sums are bound (between the planes and the output).
    group_sums: bool,
}

fn kernels() -> &'static [(&'static str, &'static str)] {
    let kq = |base, src| kernels::template::interned(base, src, &[("CODE_BITS", 4)]).expect("the K-quant template");
    Box::leak(Box::new([
        ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
        ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
        kq("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV),
        kq("matmul_kq_dyn", kernels::MATMUL_KQ_DYN),
    ]))
}

const FORMATS: &[Format] = &[
    Format { label: "i8", gemv: 0, gemm: 1, planes: |n, k| vec![u64::from(n) * u64::from(k / 4), u64::from(n) * u64::from(k / 32)], params: |m, k, n| [m, k / 4, n], group_sums: false },
    Format {
        label: "q4k",
        gemv: 2,
        gemm: 3,
        planes: |n, k| vec![u64::from(n) * u64::from(k / 8), u64::from(n) * u64::from(k / 64), u64::from(n) * u64::from(k / 256)],
        params: |m, k, n| [m, k, n],
        group_sums: true,
    },
];

/// Best per-call device time (ms) of `kernel` over `TRIALS`, and the bytes one call's weights take.
fn time(gpu: &Gpu, f: &Format, kernel: usize, m: u32, k: u32, n: u32) -> (f64, u64) {
    let planes = (f.planes)(n, k);
    let bytes: u64 = planes.iter().sum::<u64>() * 4;
    let copies = FOOTPRINT_BYTES.div_ceil(bytes).max(1) as usize;
    let xq = gpu.storage(u64::from(m * k / 4));
    gpu.write(&xq, &vec![0x01020304u32; (m * k / 4) as usize]);
    let sx = gpu.storage(u64::from(m * k / 32));
    gpu.write_f32(&sx, &vec![1.0; (m * k / 32) as usize]);
    let xgs = gpu.storage(u64::from(m * k / 32));
    let out = gpu.storage(u64::from(m * n));
    let weights: Vec<Vec<DeviceBuffer>> = (0..copies).map(|_| planes.iter().map(|&w| gpu.storage(w)).collect()).collect();
    let workgroups = if kernel == f.gemv { n } else { m.div_ceil(128) * n.div_ceil(128) };
    let steps: Vec<_> = (0..copies.max(MIN_CALLS))
        .map(|i| {
            let w = &weights[i % copies];
            let mut bufs: Vec<&DeviceBuffer> = vec![&xq, &w[0], &sx];
            bufs.extend(w[1..].iter());
            if f.group_sums {
                bufs.push(&xgs);
            }
            bufs.push(&out);
            gpu.dispatch(kernel, &bufs, &(f.params)(m, k, n), Dispatch::Workgroups(workgroups))
        })
        .collect();
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let mut best = f64::MAX;
    for _ in 0..TRIALS {
        gpu.reset_kernel_times();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let rows = gpu.kernel_times().expect("device timing on this backend");
        let (_, ms, calls) = rows.iter().max_by(|a, b| a.1.total_cmp(&b.1)).expect("the kernel launched");
        best = best.min(ms / *calls as f64);
    }
    (best, bytes)
}

#[test]
#[ignore]
fn gemv_against_gemm_at_decode_rows() {
    let gpu = Gpu::new(kernels());
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    println!("{:<5} {:<24} {:>3} {:>10} {:>10} {:>8}", "fmt", "shape", "m", "gemv us", "gemm us", "winner");
    for f in FORMATS {
        for &(label, k, n) in SHAPES {
            for m in [1u32, 4, 8, 12, 16, 24, 32] {
                let (gemv, _) = time(&gpu, f, f.gemv, m, k, n);
                let (gemm, _) = time(&gpu, f, f.gemm, m, k, n);
                println!("{:<5} {label:<24} {m:>3} {:>10.1} {:>10.1} {:>8}", f.label, gemv * 1e3, gemm * 1e3, if gemv <= gemm { "gemv" } else { "gemm" });
            }
        }
    }
}
