// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed cost of the native fused decode kernels against the WGSL chains
//! they replace, at the real Qwen3.8-27B decode widths.
//!
//! Swedish Embedded AB implements measured latency contracts for GPU inference
//! stacks. If your team needs expertise in knowing where a decode token's time
//! goes between the weight streams, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a correctness gate (that is
//! `add_rms_quant_native.rs` and `quant_epilogue_native.rs`). Run on a CUDA box:
//!
//! ```text
//! cargo test --release --offline -p brain-gpu-core --test fused_native_bench -- --ignored --nocapture
//! ```
//!
//! Time is the kernels' own, from the backend's per-launch device events
//! (`Gpu::kernel_times`), never the host clock: the card is shared with other
//! work. A launch is timed alone, so the figure includes the event pair's own
//! floor (about a microsecond and a half) - compare chain against fused, not
//! either against zero. Buffers are reused, so the data is L2-resident: this is
//! the latency of the kernel, which is what these small kernels are bound by,
//! not a bandwidth.

use gpu_core::{Dispatch, Fused, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("add2", kernels::ADD2),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("max_abs_rows", kernels::MAX_ABS_ROWS),
    ("quant_pack", kernels::QUANT_PACK),
    ("silu_mul", kernels::SILU_MUL),
    ("sigmoid", kernels::SIGMOID),
    ("mul", kernels::MUL),
];
const ADD2: usize = 0;
const RMS: usize = 1;
const MAX_ABS: usize = 2;
const QUANT: usize = 3;
const SILU_MUL: usize = 4;
const SIGMOID: usize = 5;
const MUL: usize = 6;

const REPS: usize = 50;

/// Per-call device microseconds, summed over every kernel the steps launch,
/// from the best of a few trials.
fn time_us(gpu: &Gpu, steps: &dyn Fn() -> Vec<gpu_core::Step>) -> f64 {
    let run = || {
        let all: Vec<_> = (0..REPS).flat_map(|_| steps()).collect();
        gpu.reset_kernel_times();
        gpu.submit(&[], &all);
        gpu.poll_wait();
        let rows = gpu.kernel_times().expect("device timing on this backend");
        rows.iter().map(|(_, ms, _)| ms).sum::<f64>() * 1e3 / REPS as f64
    };
    run();
    (0..5).map(|_| run()).fold(f64::MAX, f64::min)
}

fn filled(gpu: &Gpu, n: usize) -> backend_api::DeviceBuffer {
    let v: Vec<f32> = (0..n).map(|i| ((i * 37 % 1001) as f32 - 500.0) * 0.003).collect();
    gpu.storage_init("data", &v)
}

#[test]
#[ignore]
fn fused_kernels_against_the_chains_they_replace() {
    let gpu = Gpu::new(KERNELS);
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    println!("{:<44} {:>10} {:>10}", "case", "chain us", "fused us");

    // add2 + rmsnorm_rows + max_abs_rows + quant_pack, one 5120-wide row.
    for (rows, d) in [(1u32, 5120u32), (4, 5120)] {
        let n = (rows * d) as usize;
        let (a, b, w) = (filled(&gpu, n), filled(&gpu, n), filled(&gpu, d as usize));
        let (sum, xn, sx) = (gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(rows as u64));
        let xq = gpu.storage((n / 4) as u64);
        let eps = 1e-6f32.to_bits();
        let chain = time_us(&gpu, &|| {
            vec![
                gpu.dispatch(ADD2, &[&a, &b, &sum], &[rows * d], Dispatch::Threads(rows * d)),
                gpu.dispatch(RMS, &[&sum, &w, &xn], &[d, rows, eps], Dispatch::Workgroups(rows)),
                gpu.dispatch(MAX_ABS, &[&xn, &sx], &[rows, d], Dispatch::Workgroups(rows)),
                gpu.dispatch(QUANT, &[&xn, &sx, &xq], &[rows, d], Dispatch::Threads(rows * d / 4)),
            ]
        });
        let fused = time_us(&gpu, &|| {
            vec![gpu.fused_step(Fused::AddRmsQuant, &[&a, &b, &w, &sum, &xn, &xq, &sx], &[d, rows, eps, 1]).expect("add_rms_quant offered")]
        });
        println!("{:<44} {chain:>10.1} {fused:>10.1}", format!("add_rms_quant rows={rows} d={d}"));
    }

    // The three epilogues at the widths a layer hands them.
    for (mode, k, label) in [(1u32, 17408u32, "silu_mul -> quant (MLP hidden)"), (0, 6144, "quant (GDN gated)"), (2, 6144, "sigmoid*ctx -> quant (attention)")] {
        let rows = 1u32;
        let n = (rows * k) as usize;
        let (a, b) = (filled(&gpu, n), filled(&gpu, n));
        let (y, gate, sx) = (gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(rows as u64));
        let xq = gpu.storage((n / 4) as u64);
        let chain = time_us(&gpu, &|| {
            let mut s = Vec::new();
            let produced = match mode {
                0 => &a,
                1 => {
                    s.push(gpu.dispatch(SILU_MUL, &[&a, &b, &y], &[rows * k], Dispatch::Threads(rows * k)));
                    &y
                }
                _ => {
                    s.push(gpu.dispatch(SIGMOID, &[&b, &gate], &[rows * k], Dispatch::Threads(rows * k)));
                    s.push(gpu.dispatch(MUL, &[&a, &gate, &y], &[rows * k], Dispatch::Threads(rows * k)));
                    &y
                }
            };
            s.push(gpu.dispatch(MAX_ABS, &[produced, &sx], &[rows, k], Dispatch::Workgroups(rows)));
            s.push(gpu.dispatch(QUANT, &[produced, &sx, &xq], &[rows, k], Dispatch::Threads(rows * k / 4)));
            s
        });
        let fused = time_us(&gpu, &|| vec![gpu.fused_step(Fused::QuantEpilogue, &[&a, &b, &y, &xq, &sx], &[k, rows, mode, 0]).expect("quant_epilogue offered")]);
        println!("{label:<44} {chain:>10.1} {fused:>10.1}");
    }
}
