// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-kernel device timestamp timing (`BRAIN_PROFILE`) on `backend-vulkan`.
//!
//! Before this, `set_kernel_timing`/`kernel_times` inherited `backend-api`'s
//! defaults (`false`/`None`), so `gpu_core::profile` fell back to host-
//! bracketed group times on this backend — which the wgpu-side lesson
//! (`docs/kernel-checklist.md` §F.1) already measured inflating small
//! kernels up to 29x, meaning the *ranking*, not just the precision, was
//! wrong. These tests pin the real `vkCmdWriteTimestamp`-based
//! implementation across both `flush()` paths (batched, and the
//! Intel-ANV-workaround serialized path).
//!
//! All tests skip (pass trivially) when no Vulkan device is present.

use backend_api::Backend;
use backend_vulkan::VulkanBackend;

fn backend() -> Option<VulkanBackend> {
    match VulkanBackend::try_new(&[("axpy", kernels::AXPY)]) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping (no Vulkan device): {e}");
            None
        }
    }
}

#[test]
fn kernel_times_reports_real_device_time_less_than_host_wall_clock() {
    let Some(be) = backend() else { return };
    if !be.set_kernel_timing(true) {
        eprintln!("skipping: this queue cannot write timestamps (timestamp_valid_bits == 0)");
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
    let Some(be) = backend() else { return };
    if !be.set_kernel_timing(true) {
        eprintln!("skipping: this queue cannot write timestamps");
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

#[test]
fn timing_is_off_by_default_and_disabling_reports_zero_calls() {
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
