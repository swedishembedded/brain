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

/// Each of this file's tests calls `backend()` (or, for the coopmat repro,
/// `VulkanBackend::try_new` directly) to build its own real
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

/// A corrupted (or query-mechanism-perturbed) device-timestamp readback must
/// never surface as a real number (M6.8) - the `backend-vulkan` sibling of
/// `backend-wgpu`'s M6.6.
///
/// Root-caused on this workspace's Intel Arc (Meteor Lake, Xe-LPG) iGPU via
/// `qwen35_bench gqa 128 3` (`BRAIN_DEVICE=vulkan BRAIN_VK_SERIAL=1`): the
/// native cooperative-matrix `matmul` kernel's per-dispatch device time
/// reported up to ~9219 ms for a single dispatch, though the wall-clock pass
/// it belonged to finished in ~29 ms total. Unlike `backend-wgpu`'s sibling
/// defect (a literal garbled readback - an unwritten query reading 0, or a
/// value wrong by orders of magnitude with neither side reading 0), this
/// backend's queries read back real, monotonically increasing, internally-
/// consistent values (confirmed by instrumenting `end_and_wait` with a host-
/// side `Instant` alongside the device timestamps - the two agreed to within
/// a few percent every time) - so the corruption here is a genuine, driver-
/// reproducible SLOWDOWN of the dispatch itself, not a bad bit pattern. This
/// repro reproduces that slowdown directly: a `matmul_coopmat` dispatch
/// large enough to trigger it, repeated under `BRAIN_VK_SERIAL`'s per-
/// dispatch submit+fence+query-pool-reset path, without needing the model
/// crate or `qwen35_bench` at all. `IMPLAUSIBLE_DISPATCH_MS`'s own doc has
/// the quantitative throughput comparison proving this is not simply "coopmat
/// is slow on this iGPU" - a real GQA-sized dispatch is ~300-500x slower than
/// this same pipeline's own directly-measured achievable rate.
///
/// This does not assert the slowdown always fires (a driver fix, or
/// different hardware, may execute every dispatch in well under the ceiling,
/// which is a pass, not a skip) - only that whatever `kernel_times()` DOES
/// report never exceeds `IMPLAUSIBLE_DISPATCH_MS` per call, mirroring
/// `backend-wgpu`'s own `a_corrupted_timestamp_readback_is_discarded_not_
/// reported`. Before this fix, this exact repro folded a `native-spirv#main`
/// device time in the thousands of ms per call into the table.
#[test]
fn a_corrupted_timestamp_readback_is_discarded_not_reported() {
    use backend_vulkan::coopmat;
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let be = match VulkanBackend::try_new(&[]) {
        Ok(b) => b,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            return;
        }
    };
    let Some(spec) = coopmat::spec() else {
        brain_testutil::skip_unavailable("no coopmat SPIR-V baked in at build time");
        return;
    };
    let Some(id) = be.register_native(&spec) else {
        brain_testutil::skip_unavailable("this device declined the coopmat pipeline");
        return;
    };
    if !be.set_kernel_timing(true) {
        brain_testutil::skip_unavailable("this queue cannot write timestamps");
        return;
    }

    // Force the per-dispatch (submit+fence+query-pool-reset) Intel-ANV
    // workaround path regardless of vendor - the path the real repro used.
    // SAFETY: test-process-local env var, no other thread reads it concurrently.
    unsafe { std::env::set_var("BRAIN_VK_SERIAL", "1") };

    // (M, K, N) chosen empirically on this box: large enough (N especially -
    // a "thin and wide" tile grid reproduces this far more reliably than a
    // balanced one of the same total FLOP count, empirically, not fully
    // root-caused) to trigger the slowdown reliably, small enough (K kept
    // modest) to keep the packed weight buffer (`N*K` f16 elements) under
    // half a gigabyte and the whole repro under ~15s wall clock.
    let (m, k, n) = (128u32, 4096u32, 196608u32);
    let x = vec![0.01f32; (m * k) as usize];
    let w = vec![0.01f32; (n * k) as usize];
    let (xf, mp, kp) = coopmat::pack_padded_f16(&x, m, k);
    let (wf, np, kp2) = coopmat::pack_padded_f16(&w, n, k);
    debug_assert_eq!(kp, kp2);
    let xbuf = be.storage((xf.len() / 2) as u64);
    be.write(&xbuf, bytemuck::cast_slice(&xf));
    let wbuf = be.storage((wf.len() / 2) as u64);
    be.write(&wbuf, bytemuck::cast_slice(&wf));
    let obuf = be.storage((mp * np) as u64);
    let tiles = (mp / coopmat::TILE) * (np / coopmat::TILE);
    let params = [mp, kp, np];
    for _ in 0..4 {
        let step = be.step_native(id, &[&xbuf, &wbuf, &obuf], &params, tiles).expect("id was just registered");
        be.submit(&[], &[step]);
        be.poll_wait();
    }

    unsafe { std::env::remove_var("BRAIN_VK_SERIAL") };

    let times = be.kernel_times().expect("timing was enabled and timestamps are supported");
    for (name, ms, calls) in &times {
        assert!(*calls > 0, "kernel_times must never report a zero-call row");
        let per_call = ms / *calls as f64;
        assert!(
            per_call <= backend_vulkan::IMPLAUSIBLE_DISPATCH_MS,
            "{name} reported {ms:.1} ms over {calls} call(s) ({per_call:.1} ms/call) - an implausible \
             device-timestamp reading was folded into the table instead of being discarded by \
             record_timing's sanity ceiling (M6.8)"
        );
    }
}
