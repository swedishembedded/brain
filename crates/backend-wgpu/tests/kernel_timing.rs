// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `BRAIN_PROFILE` timestamp path must MEASURE a flush without BECOMING
//! one.
//!
//! Swedish Embedded AB implements GPU profiling instrumentation for inference
//! engines for its clients. If your team needs expertise in device timestamp
//! queries and profiler-induced serialization then you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! A timed flush resolves its query sets into a buffer and then has to map that
//! buffer to read the ticks - and mapping means blocking until the submission
//! completes. Read back inside the flush, that is a full device round trip per
//! flush: the instrument imposes exactly the per-flush drain that a caller
//! overlapping host work with device work exists to remove, so such a caller
//! measures as no faster than the serial one it replaced. The readback is
//! therefore DEFERRED, and these tests pin the two properties the resulting
//! numbers depend on:
//!
//! * deferral loses nothing - every dispatch of every unresolved flush is in
//!   the table by the time anyone reads it;
//! * the flush itself returns while the card is still working, which is the
//!   property that makes overlap measurable at all.

use backend_api::Backend;
use backend_wgpu::WgpuBackend;

/// Each test builds its own real device, and concurrent independent device
/// builds on one physical card are the driver hazard
/// `crates/gpu-core/tests/device_sharing.rs` exists to prevent. Same fix here.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn backend() -> Option<WgpuBackend> {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
        return None;
    }
    Some(WgpuBackend::new(&[("axpy", kernels::AXPY)]))
}

/// Turn timing on, or say why the rest of the test cannot run. A device whose
/// adapter never offered timestamp queries has no `GpuProfile` at all.
fn timed(be: &WgpuBackend) -> bool {
    if be.set_kernel_timing(true) {
        return true;
    }
    brain_testutil::skip_unavailable("this adapter cannot write kernel timestamps");
    false
}

/// Several separate flushes, resolved only when the table is asked for. Every
/// dispatch has to be in it. A deferral that dropped the batches, or that
/// resolved only the newest, would show up here as a short call count - which
/// is precisely the failure a profiler cannot self-report.
#[test]
fn deferring_the_timestamp_readback_loses_no_dispatch() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !timed(&be) {
        return;
    }
    be.reset_kernel_times();

    const FLUSHES: usize = 5;
    const PER_FLUSH: usize = 8;
    let out = Backend::storage(&be, 1024);
    let inp = Backend::storage_init(&be, "inp", &vec![1.0f32; 1024]);
    for _ in 0..FLUSHES {
        let steps: Vec<_> = (0..PER_FLUSH).map(|_| Backend::step(&be, 0, &[&out, &inp], &[1024, backend_api::f(1.0)], 1024)).collect();
        Backend::submit(&be, &[], &steps);
        // Queue it and move on, exactly as a pipelined caller does. Nothing is
        // read back here.
        Backend::flush(&be);
    }

    let times = be.kernel_times().expect("timing is on and this adapter supports timestamps");
    let (name, device_ms, calls) = times.iter().find(|(n, _, _)| n == "axpy").expect("axpy was timed");
    assert_eq!(name, "axpy");
    assert_eq!(*calls as usize, FLUSHES * PER_FLUSH, "the deferred batches did not all reach the table: {calls} of {} dispatches", FLUSHES * PER_FLUSH);
    assert!(*device_ms > 0.0, "a timed dispatch that reports exactly zero device time means the ticks were never read");

    // A reset drops what has not been folded in yet, because those batches
    // measured work from before the reset.
    be.reset_kernel_times();
    let after = be.kernel_times().expect("still supported after reset");
    assert!(after.iter().all(|(_, _, c)| *c == 0), "reset_kernel_times must zero every accumulator");
}

/// The flush returns while the card is still busy. This is the property the
/// whole deferral exists for, so it is asserted rather than assumed: with the
/// readback inside the flush, the two spans below swap and this fails.
///
/// Sized so the device work is far longer than the host encode: the same
/// buffers dispatched many times, so the pass is bandwidth-bound and the
/// encode is a few hundred `set_bind_group`/`dispatch` calls.
#[test]
fn a_timed_flush_returns_before_the_device_has_finished() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !timed(&be) {
        return;
    }
    const N: u64 = 4 << 20;
    const DISPATCHES: usize = 256;
    let out = Backend::storage(&be, N);
    let inp = Backend::storage_init(&be, "inp", &vec![1.0f32; N as usize]);
    let steps: Vec<_> = (0..DISPATCHES).map(|_| Backend::step(&be, 0, &[&out, &inp], &[N as u32, backend_api::f(1.0)], N as u32)).collect();
    Backend::submit(&be, &[], &steps);

    let t_flush = std::time::Instant::now();
    Backend::flush(&be);
    let flush_ms = t_flush.elapsed().as_secs_f64() * 1e3;
    let t_wait = std::time::Instant::now();
    Backend::poll_wait(&be);
    let wait_ms = t_wait.elapsed().as_secs_f64() * 1e3;

    println!("timed flush: {flush_ms:.2} ms to queue {DISPATCHES} dispatches, {wait_ms:.2} ms waiting for them");
    assert!(
        flush_ms * 4.0 < wait_ms,
        "the timed flush spent {flush_ms:.2} ms against {wait_ms:.2} ms of device work: it is draining the queue itself, so a caller that overlaps host work with device work cannot be measured by this profiler"
    );
}

/// Dropping a SHARED handle must not wait for the device either.
///
/// This one is not obvious and it cost a full measurement round to find. Every
/// handle's `Drop` reports its profile, and folding the deferred tick batches
/// in means blocking until the queue is idle - so a `Drop` that resolves turns
/// every share release into a full device stall. That matters because shares
/// are released on hot paths: `ltxv`'s resident weight window drops the share
/// an evicted block held, once per streamed block, in the middle of a forward.
///
/// The failure is invisible to every correctness gate (nothing computes a
/// different number) AND to the caller's own stage timers, because a
/// destructor runs outside every span the caller has. It reads as the
/// overlapping simply not working, which is the most expensive kind of wrong
/// answer - so the property is pinned here rather than left to the next
/// measurement to rediscover.
#[test]
fn dropping_a_shared_handle_does_not_wait_for_the_device() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !timed(&be) {
        return;
    }
    const N: u64 = 4 << 20;
    const DISPATCHES: usize = 256;
    let share = be.share_device();
    let out = Backend::storage(&share, N);
    let inp = Backend::storage_init(&share, "inp", &vec![1.0f32; N as usize]);
    let steps: Vec<_> = (0..DISPATCHES).map(|_| Backend::step(&share, 0, &[&out, &inp], &[N as u32, backend_api::f(1.0)], N as u32)).collect();
    Backend::submit(&share, &[], &steps);
    Backend::flush(&share);

    let t_drop = std::time::Instant::now();
    drop(share);
    let drop_ms = t_drop.elapsed().as_secs_f64() * 1e3;
    // The work has to still be outstanding, or the drop had nothing to wait
    // for and this test proves nothing. The device wait is measured AFTER the
    // drop, on the parent handle, and it is the same queue.
    let t_wait = std::time::Instant::now();
    Backend::poll_wait(&be);
    let wait_ms = t_wait.elapsed().as_secs_f64() * 1e3;

    println!("shared-handle drop: {drop_ms:.2} ms, with {wait_ms:.2} ms of device work still outstanding after it");
    assert!(
        wait_ms > 5.0,
        "the device finished before the drop was even reached ({wait_ms:.2} ms left), so this run cannot tell a waiting destructor from a cheap one"
    );
    assert!(
        drop_ms * 4.0 < wait_ms,
        "dropping a shared handle spent {drop_ms:.2} ms while {wait_ms:.2} ms of device work was still queued: the destructor is draining the queue, which stalls every caller that releases a share inside a hot loop"
    );
}
