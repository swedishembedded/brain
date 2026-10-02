// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native int8 GEMV owns device resources - a loaded module per handle,
//! uniform and staging blocks per dispatch structure - and none of them may
//! outlive the handle that created them.
//!
//! Swedish Embedded AB implements long-running GPU inference services that do
//! not leak. If your team needs expertise in keeping device memory flat across
//! the lifetime of a serving process, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! Its own file, so its process is the only one churning the card while the
//! free-memory reading is taken: the checks in `i8_gemv_native.rs` allocate
//! tens of megabytes concurrently. The card may still be shared with other
//! PROCESSES, which is why the bound below is a trend over many handles, not
//! a byte count: each cycle allocates [`CYCLE_BYTES`], so a leaked buffer per
//! cycle is orders of magnitude over the slack, while another job's churn is
//! not.

use gpu_core::{Dispatch, Gpu};

static KERNELS: &[(&str, &str)] = &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV)];

const CYCLES: usize = 24;
const M: u32 = 3;
const KG: u32 = 1280;
const N: u32 = 4096;
/// Device bytes one cycle's buffers occupy.
const CYCLE_BYTES: u64 = 4 * (N as u64 * KG as u64 + N as u64 * (KG as u64 / 8)) + 4 * (M as u64 * KG as u64);
/// Free memory may drift by this much from other processes without it meaning
/// anything; a per-cycle leak of even one weight buffer is `CYCLES` times
/// larger than the slack.
const SLACK_BYTES: u64 = 4 * CYCLE_BYTES;

fn one_cycle(gpu: &Gpu) {
    let xq = gpu.storage(u64::from(M * KG));
    let wq = gpu.storage(u64::from(N * KG));
    let sx = gpu.storage(u64::from(M));
    let sw = gpu.storage(u64::from(N * KG / 8));
    let out = gpu.storage(u64::from(M * N));
    let params = [M, KG, N];
    assert_eq!(gpu.native_kernel_for(0, &params), Some("matmul_i8_gemv"), "the cycle must exercise the native kernel");
    gpu.submit(&[], &[gpu.dispatch(0, &[&xq, &wq, &sx, &sw, &out], &params, Dispatch::Workgroups(N))]);
    gpu.poll_wait();
}

#[test]
fn handles_that_dispatch_the_native_kernel_return_every_byte_they_took() {
    // The anchor keeps the device context alive across the loop and is the
    // one handle the free-memory reading is taken from.
    let anchor = Gpu::new(KERNELS);
    if anchor.kind() != "cuda" || anchor.native_kernel_for(0, &[M, KG, N]).is_none() {
        brain_testutil::skip_unavailable("the native int8 GEMV needs a CUDA device with int8 dot support");
        return;
    }
    // One warm cycle so one-time driver state is not counted as a leak.
    one_cycle(&anchor);
    let before = anchor.max_buffer_bytes();
    let own_before = brain_testutil::own_gpu_memory_mib();
    for _ in 0..CYCLES {
        let gpu = Gpu::new(KERNELS);
        one_cycle(&gpu);
        // A sibling handle registers the same catalogue again: it must reuse
        // the module, and its own blocks must go with it.
        one_cycle(&gpu.share());
    }
    anchor.poll_wait();
    // The card is shared: another process loading a model moves device-wide
    // free memory by gigabytes. Where the driver says what THIS process holds,
    // that is the figure a leak shows up in.
    if let (Some(b), Some(a)) = (own_before, brain_testutil::own_gpu_memory_mib()) {
        assert!(
            a <= b + (SLACK_BYTES >> 20) + 1,
            "the driver attributes {a} MiB to this process after {CYCLES} build/dispatch/drop cycles against {b} MiB before - something a handle owns outlived it"
        );
        return;
    }
    let after = anchor.max_buffer_bytes();
    assert!(
        after + SLACK_BYTES >= before,
        "free device memory fell from {before} to {after} bytes over {CYCLES} build/dispatch/drop cycles \
         of {CYCLE_BYTES} bytes each - something a handle owns outlived it"
    );
}
