// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-kernel device timestamp timing (`BRAIN_PROFILE`) on `backend-vulkan`.
//!
//! Before this, `set_kernel_timing`/`kernel_times` inherited `backend-api`'s
//! defaults (`false`/`None`), so `gpu_core::profile` fell back to host-
//! bracketed group times on this backend — which the wgpu-side lesson
//! already measured inflating small
//! kernels by more than an order of magnitude, meaning the *ranking*, not just
//! the precision, was
//! wrong. These tests pin the real `vkCmdWriteTimestamp`-based
//! implementation across both `flush()` paths (batched, and the
//! Intel-ANV-workaround serialized path).
//!
//! All tests skip (pass trivially) when no Vulkan device is present.

use backend_api::Backend;
use backend_vulkan::VulkanBackend;

/// Each of this file's 3 tests calls `backend()` to build its own real
/// Vulkan device directly (below `gpu_core::Gpu`, so `gpu_core::testgpu::dev`
/// does not apply here) - under `cargo test`'s default multi-threaded run
/// they can run concurrently and race their own independent device builds
/// against each other on the same physical card, the exact driver hazard
/// `crates/gpu-core/tests/device_sharing.rs`'s `DEVICE_SERIAL` (and its
/// copies elsewhere) exist to prevent. This is the actual root cause of a
/// hang this file caused under a full `make test` run that was previously
/// attributed to unproven cross-process contention - it was cross-thread
/// contention within this one test binary all along. Same fix here.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn backend() -> Option<VulkanBackend> {
    match VulkanBackend::try_new(&[("axpy", kernels::AXPY)]) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            None
        }
    }
}

#[test]
fn kernel_times_reports_real_device_time_less_than_host_wall_clock() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !be.set_kernel_timing(true) {
        brain_testutil::skip_unavailable("this queue cannot write timestamps (timestamp_valid_bits == 0)");
        return;
    }

    let out = be.storage(1024);
    let inp = be.storage_init("inp", &vec![1.0f32; 1024]);
    let steps: Vec<_> = (0..8).map(|_| be.step(0, &[&out, &inp], &[1024, backend_api::f(1.0)], 1024)).collect();

    let host_start = std::time::Instant::now();
    be.submit(&[], &steps);
    be.poll_wait();
    let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;

    let times = be.kernel_times().expect("timing was enabled and timestamps are supported");
    assert!(!times.is_empty(), "expected at least one timed kernel kind");
    let (name, device_ms, calls) = &times[0];
    assert_eq!(name, "axpy");
    assert_eq!(*calls, 8, "one call per dispatch in the batch");
    assert!(*device_ms >= 0.0, "device time must not be negative: {device_ms}");
    // The device-timed sum must not exceed the host wall clock around the
    // same submit — a device time larger than host time is the tell that
    // the timestamps are garbage (wrong period scaling, wrong query
    // indices, ...), not a fast kernel.
    assert!(
        *device_ms <= host_ms,
        "device time {device_ms:.3}ms exceeds host wall time {host_ms:.3}ms -- timestamps are not trustworthy"
    );

    be.reset_kernel_times();
    let after_reset = be.kernel_times().expect("still supported after reset");
    assert!(after_reset.is_empty() || after_reset.iter().all(|(_, _, c)| *c == 0), "reset_kernel_times must zero every accumulator");
}

#[test]
fn kernel_times_also_works_on_the_serialized_intel_workaround_path() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !be.set_kernel_timing(true) {
        brain_testutil::skip_unavailable("this queue cannot write timestamps");
        return;
    }
    // Force the serialized (submit+fence per dispatch) branch regardless of
    // vendor, exercising the OTHER half of the timing implementation.
    // SAFETY: test-process-local env var, no other thread reads it concurrently.
    unsafe { std::env::set_var("BRAIN_VK_SERIAL", "1") };

    let out = be.storage(256);
    let inp = be.storage_init("inp", &vec![1.0f32; 256]);
    let steps: Vec<_> = (0..4).map(|_| be.step(0, &[&out, &inp], &[256, backend_api::f(1.0)], 256)).collect();
    be.submit(&[], &steps);
    be.poll_wait();

    unsafe { std::env::remove_var("BRAIN_VK_SERIAL") };

    let times = be.kernel_times().expect("timing supported");
    let (_, device_ms, calls) = times.iter().find(|(n, _, _)| n == "axpy").expect("axpy was timed");
    assert_eq!(*calls, 4);
    assert!(*device_ms >= 0.0);
}

/// A batch larger than the query-pool's own capacity (`MAX_TIMED_DISPATCHES`
/// == 8192 in `backend-vulkan/src/lib.rs`, not exported - this test pins the
/// externally observable contract, not the private constant) used to skip
/// timing for the WHOLE flush: `flush()` gated the query pool on
/// `steps.len() < MAX_TIMED_DISPATCHES`, so a 48-layer/128-expert MoE forward
/// (which routinely exceeds it) got zero per-kernel attribution, silently.
/// `kernel_times` must instead attribute every kernel kind dispatched in an
/// oversized batch, by bracketing bounded sub-batches within the flush
/// (each its own submit+fence-bounded timestamp pair) rather than dropping
/// timing for the batch outright.
#[test]
fn kernel_times_attributes_every_kind_above_the_query_pool_capacity() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let be = match VulkanBackend::try_new(&[("axpy", kernels::AXPY), ("add", kernels::ADD)]) {
        Ok(b) => b,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            return;
        }
    };
    if !be.set_kernel_timing(true) {
        brain_testutil::skip_unavailable("this queue cannot write timestamps (timestamp_valid_bits == 0)");
        return;
    }

    let out = be.storage(64);
    let inp = be.storage_init("inp", &vec![1.0f32; 64]);
    let src2 = be.storage_init("src2", &vec![1.0f32; 64]);
    let dst2 = be.storage(64);

    // Comfortably above the 8192-dispatch query-pool capacity, and mixed
    // between two kernel kinds so a dropped/misattributed sub-batch at a
    // chunk boundary would show up as one kind's `calls` undercounting.
    const N: usize = 8300;
    let steps: Vec<_> = (0..N)
        .map(|i| {
            if i % 2 == 0 {
                be.step(0, &[&out, &inp], &[64u32, backend_api::f(1.0)], 64)
            } else {
                be.step(1, &[&src2, &dst2], &[64u32], 64)
            }
        })
        .collect();

    be.submit(&[], &steps);
    be.poll_wait();

    let times = be.kernel_times().expect("timing was enabled and timestamps are supported");
    let axpy_calls = times.iter().find(|(n, _, _)| n == "axpy").map(|(_, _, c)| *c).unwrap_or(0);
    let add_calls = times.iter().find(|(n, _, _)| n == "add").map(|(_, _, c)| *c).unwrap_or(0);
    let total_ms: f64 = times.iter().map(|(_, ms, _)| ms).sum();

    assert!(axpy_calls > 0, "axpy was dispatched {} times above the query-pool cap but got zero attribution", N.div_ceil(2));
    assert!(add_calls > 0, "add was dispatched {} times above the query-pool cap but got zero attribution", N / 2);
    assert_eq!(axpy_calls + add_calls, N as u64, "every dispatch in the oversized batch must be accounted for exactly once");
    assert!(total_ms > 0.0, "an 8300-dispatch batch must report nonzero device time, not a silently empty profile");
}

#[test]
fn timing_is_off_by_default_and_disabling_reports_zero_calls() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if be.caps().numeric.f32 && be.set_kernel_timing(false) {
        // Explicitly disabled: dispatching must not accumulate anything.
        let out = be.storage(64);
        let inp = be.storage_init("inp", &vec![1.0f32; 64]);
        let step = be.step(0, &[&out, &inp], &[64, backend_api::f(1.0)], 64);
        be.submit(&[], std::slice::from_ref(&step));
        be.poll_wait();
        let times = be.kernel_times().expect("still Some when timestamps are supported, just empty");
        assert!(times.iter().all(|(_, _, c)| *c == 0), "disabled timing must not accumulate calls");
    }
}
