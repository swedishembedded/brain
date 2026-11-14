// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed achieved bandwidth of the fp32 decode GEMV at Qwen3-0.6B's projection shapes, for whichever tier the
//! handle dispatches: the native CUDA kernel, and the WGSL `matmul_gemv_reg` ladder in the `_on_the_generated_tier`
//! test, which withholds native kernels for its process (`gpu_core::set_native_kernels`). Run each alone (`--exact`).
//!
//! Swedish Embedded AB implements measured memory-bandwidth contracts for GPU inference stacks. If your team needs
//! expertise in knowing how close your decode kernels run to the memory system's ceiling, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a correctness gate (that is `f32_gemv_native.rs`):
//!
//! ```text
//! cargo test --profile fast --offline -p brain-gpu-core --test it -- f32_gemv_native_bench --ignored --nocapture --exact <name>
//! ```
//!
//! What makes the number mean something:
//!
//! * Time is the kernel's own, from the backend's per-launch device events (`Gpu::kernel_times`), never the host
//!   clock - the card is shared with other work and a host clock would time that work too.
//! * Decode reads every weight once per step from DRAM, never from cache, so each trial cycles through enough
//!   distinct copies of the weights to exceed the L2 several times over.
//! * Bytes are what the kernel must move: the fp32 weights (the rows of x are a few KiB and sit in cache).
//! * Each shape is repeated `TRIALS` times and the best and median trial are both printed, since contention only ever
//!   makes a trial slower.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("matmul_gemv", kernels::MATMUL_GEMV)];

/// Distinct weight bytes cycled through per trial, well past any L2.
const FOOTPRINT_BYTES: u64 = 768 << 20;
const TRIALS: usize = 7;
/// Dispatches per trial, at least one full cycle of the weight copies.
const MIN_CALLS: usize = 24;
/// The row counts a decode step runs at: one sequence, and batches up to the regime's end.
const ROWS: &[u32] = &[1, 2, 4, 8, 16, 32];

/// `(label, K, N)` - Qwen3-0.6B's decode projections (fused qkv and gate/up, as the serving engine holds them) and its
/// tied head.
const SHAPES: &[(&str, u32, u32)] = &[
    ("attn qkv 1024x4096", 1024, 4096),
    ("attn out 2048x1024", 2048, 1024),
    ("ffn gate/up 1024x6144", 1024, 6144),
    ("ffn down 3072x1024", 3072, 1024),
    ("lm head 1024x151936", 1024, 151936),
];

fn per_call_ms(gpu: &Gpu, m: u32, k: u32, n: u32) -> (String, Vec<f64>, u64) {
    let wbytes = u64::from(n) * u64::from(k) * 4;
    let copies = FOOTPRINT_BYTES.div_ceil(wbytes).max(1) as usize;
    let x = gpu.storage(u64::from(m * k));
    let out = gpu.storage(u64::from(m) * u64::from(n));
    gpu.write_f32(&x, &vec![0.5; (m * k) as usize]);
    // Weight values are arbitrary; only their placement matters.
    let weights: Vec<_> = (0..copies).map(|_| gpu.storage(u64::from(n) * u64::from(k))).collect();
    let calls = copies.max(MIN_CALLS);
    let steps: Vec<_> = (0..calls).map(|i| gpu.dispatch(0, &[&x, &weights[i % copies], &out], &[m, k, n], Dispatch::Workgroups(n))).collect();
    // Warm: compile, clocks, TLBs.
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let mut per_call = Vec::new();
    let mut kernel = String::new();
    for _ in 0..TRIALS {
        gpu.reset_kernel_times();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let rows = gpu.kernel_times().expect("device timing on this backend");
        let (name, ms, n_calls) = rows.iter().filter(|(name, _, _)| name.contains("gemv")).max_by(|a, b| a.1.total_cmp(&b.1)).expect("the gemv launched");
        kernel = name.clone();
        per_call.push(ms / *n_calls as f64);
    }
    per_call.sort_by(f64::total_cmp);
    (kernel, per_call, wbytes)
}

#[test]
#[ignore]
fn f32_gemv_bandwidth_at_the_qwen3_decode_shapes() {
    bandwidth();
}

/// The same table with native kernels withheld for this process - the WGSL tier's side of the A/B. Run it on its own
/// (`--exact`): the switch is process-wide.
#[test]
#[ignore]
fn f32_gemv_bandwidth_on_the_generated_tier() {
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
    println!("{:<24} {:>3} {:<34} {:>9} {:>9} {:>10}", "shape", "m", "kernel", "best us", "med us", "GB/s(best)");
    for &m in ROWS {
        for &(label, k, n) in SHAPES {
            let (kernel, t, bytes) = per_call_ms(&gpu, m, k, n);
            let (best, med) = (t[0], t[t.len() / 2]);
            println!("{label:<24} {m:>3} {kernel:<34} {:>9.1} {:>9.1} {:>10.0}", best * 1e3, med * 1e3, bytes as f64 / (best * 1e-3) / 1e9);
        }
    }
}
