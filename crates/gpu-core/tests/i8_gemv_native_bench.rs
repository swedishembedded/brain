// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed achieved bandwidth of the int8 decode GEMV at the real
//! Qwen3.8-27B projection shapes, for whichever tier the handle dispatches -
//! the native CUDA kernel, and the WGSL `matmul_i8_gemv_reg` ladder in the
//! `_on_the_generated_tier` test, which withholds native kernels for its
//! process (`gpu_core::set_native_kernels`) - run each alone (`--exact`).
//!
//! Swedish Embedded AB implements measured memory-bandwidth contracts for GPU
//! inference stacks. If your team needs expertise in knowing how close your
//! decode kernels run to the memory system's ceiling, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a correctness gate (that is
//! `i8_gemv_native.rs`). Run on a CUDA box:
//!
//! ```text
//! cargo test --release --offline -p brain-gpu-core --test i8_gemv_native_bench -- --ignored --nocapture
//! cargo test ... achieved_bandwidth_on_the_generated_tier -- --exact --ignored (the A/B)
//! ```
//!
//! What makes the number mean something:
//!
//! * Time is the kernel's own, from the backend's per-launch device events
//!   (`Gpu::kernel_times`), never the host clock - the card is shared with
//!   other work and a host clock would time that work too.
//! * Decode reads every weight once per token from HBM, never from cache, so
//!   each trial cycles through enough distinct copies of the weights to
//!   exceed the L2 several times over. A single resident copy of a small
//!   projection would be read from L2 and report a bandwidth no token ever
//!   sees.
//! * Bytes are what the kernel must move: the packed weights and their group
//!   scales (the activation row is a few KiB and sits in L1/L2).
//! * Each shape is repeated `TRIALS` times and the best and median trial are
//!   both printed, since contention only ever makes a trial slower.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV)];

/// Distinct weight bytes cycled through per trial, well past any L2.
const FOOTPRINT_BYTES: u64 = 768 << 20;
const TRIALS: usize = 7;
/// Dispatches per trial, at least one full cycle of the weight copies.
const MIN_CALLS: usize = 24;

/// `(label, K, N)` - the Qwen3.8-27B decode projections as the GGUF lists them.
const SHAPES: &[(&str, u32, u32)] = &[
    ("gdn/attn k-proj 5120x1024", 5120, 1024),
    ("ffn narrow 5120x6144", 5120, 6144),
    ("attn out 6144x5120", 6144, 5120),
    ("gdn qkv 5120x10240", 5120, 10240),
    ("attn q 5120x12288", 5120, 12288),
    ("ffn down 17408x5120", 17408, 5120),
    ("ffn gate/up 5120x17408", 5120, 17408),
    ("lm head 5120x248320", 5120, 248320),
];

fn main_rows(gpu: &Gpu, m: u32, k: u32, n: u32) -> (String, Vec<f64>, u64) {
    let kg = k / 4;
    let wbytes = u64::from(n) * u64::from(kg) * 4 + u64::from(n) * u64::from(kg / 8) * 4;
    let copies = FOOTPRINT_BYTES.div_ceil(wbytes).max(1) as usize;
    let xq = gpu.storage(u64::from(m * kg));
    let sx = gpu.storage(u64::from(m));
    let out = gpu.storage(u64::from(m * n));
    gpu.write(&xq, &vec![0x01020304u32; (m * kg) as usize]);
    gpu.write_f32(&sx, &vec![1.0; m as usize]);
    // Weight bytes are arbitrary; only their placement matters.
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
    let calls = copies.max(MIN_CALLS);
    let steps: Vec<_> = (0..calls)
        .map(|i| {
            let (w, s) = &weights[i % copies];
            gpu.dispatch(0, &[&xq, w, &sx, s, &out], &[m, kg, n], Dispatch::Workgroups(n))
        })
        .collect();
    // Warm: compile, clocks, TLBs.
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let mut per_call_ms = Vec::new();
    let mut kernel = String::new();
    for _ in 0..TRIALS {
        gpu.reset_kernel_times();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let rows = gpu.kernel_times().expect("device timing on this backend");
        let (name, ms, n_calls) = rows
            .iter()
            .filter(|(name, _, _)| name.contains("gemv"))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("the gemv launched");
        kernel = name.clone();
        per_call_ms.push(ms / *n_calls as f64);
    }
    per_call_ms.sort_by(f64::total_cmp);
    (kernel, per_call_ms, wbytes)
}

#[test]
#[ignore]
fn achieved_bandwidth_at_the_27b_decode_shapes() {
    bandwidth();
}

/// The same table with native kernels withheld for this process - the WGSL
/// tier's side of the A/B. Run it on its own (`--exact`): the switch is
/// process-wide.
#[test]
#[ignore]
fn achieved_bandwidth_on_the_generated_tier() {
    gpu_core::set_native_kernels(false);
    bandwidth();
}

fn bandwidth() {
    let gpu = Gpu::new(KERNELS);
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    println!("{:<28} {:>3} {:<34} {:>9} {:>9} {:>9}", "shape", "m", "kernel", "best us", "med us", "GB/s(best)");
    // `BRAIN_BENCH_M=1,2` restricts the row counts (default: 1, 2, 4, 8).
    let rows: Vec<u32> = std::env::var("BRAIN_BENCH_M")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);
    for m in rows {
        for &(label, k, n) in SHAPES {
            // The head at batched m is a 1.2 GB weight read per call; one
            // batched row count is enough to show it.
            if n > 100_000 && m > 1 {
                continue;
            }
            let (kernel, t, bytes) = main_rows(&gpu, m, k, n);
            let (best, med) = (t[0], t[t.len() / 2]);
            println!(
                "{label:<28} {m:>3} {kernel:<34} {:>9.1} {:>9.1} {:>9.0}  (median {:.0})",
                best * 1e3,
                med * 1e3,
                bytes as f64 / (best * 1e-3) / 1e9,
                bytes as f64 / (med * 1e-3) / 1e9,
            );
        }
    }
}
