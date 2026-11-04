// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Asynchronous submission (M6.2): timeline semaphores, persistent/reused
//! command buffers, N submissions in flight, and no host wait unless data is
//! actually needed.
//!
//! Before this milestone, `VulkanBackend::flush` was ALWAYS a blocking
//! submit + fence wait - `crates/vulkan/src/context.rs`'s `queue_lock` doc
//! said so explicitly ("every submit here is already synchronous
//! submit+fence-wait, never pipelined"), and `Backend::flush`'s own trait
//! doc ("send recorded work WITHOUT waiting for completion... synchronise at
//! the next `read`") was aspirational on this backend, not actually true.
//! These tests pin the corrected contract:
//!
//!   1. `Backend::flush()` alone does not wait for the submission it just
//!      made to complete - the batch stays outstanding until something that
//!      actually needs the result (`read`/`poll_wait`/`write`) drains it.
//!   2. Up to `RING_SIZE` submissions can be outstanding at once per handle,
//!      never more - the ring bounds it, it does not grow unboundedly.
//!   3. `poll_wait`/`read` retire every outstanding submission.
//!   4. Chaining many more flushes than the ring is deep, each depending on
//!      the previous one's output, with no read in between, still computes
//!      the right answer - the correctness gate for reusing a slot's command
//!      buffer, and for relying on the timeline semaphore (not a full drain)
//!      to order dependent work across separate submissions.
//!
//! Skips (trivially) when no Vulkan device is present, matching this crate's
//! other test files.

use backend_vulkan::VulkanBackend;

/// Same cross-thread device-construction hazard as this crate's other test
/// files (`perf_contract.rs`, `kernel_timing.rs`) - each test here builds its
/// own real Vulkan device directly.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn backend() -> Option<VulkanBackend> {
    match VulkanBackend::try_new(&[("add2", kernels::ADD2)]) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            None
        }
    }
}

#[test]
fn flush_alone_does_not_wait_for_the_submission_to_complete() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !be.async_capable() {
        brain_testutil::skip_unavailable("device has no timeline semaphore (defensive fallback only)");
        return;
    }
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);

    assert_eq!(be.async_inflight_count(), 0, "nothing submitted yet");
    let step = be.step(0, &[&a, &b, &out], &[4], 4);
    be.submit(&[], &[step]);
    <VulkanBackend as backend_api::Backend>::flush(&be);
    assert_eq!(
        be.async_inflight_count(),
        1,
        "Backend::flush() must send the batch to the device WITHOUT waiting \
         for it to complete - its own trait doc says so, and this backend's \
         `queue_lock` used to make that false. An inflight count of 0 here \
         means flush() drained (waited), not merely submitted."
    );

    // The deferred batch still produces the right answer once actually read.
    assert_eq!(be.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(be.async_inflight_count(), 0, "read() must retire everything it waited on");
}

#[test]
fn at_most_ring_capacity_submissions_are_ever_in_flight() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !be.async_capable() {
        brain_testutil::skip_unavailable("device has no timeline semaphore (defensive fallback only)");
        return;
    }
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);
    let cap = be.async_ring_capacity();
    assert!(cap >= 2, "N-submissions-in-flight is not exercised by a ring of size {cap}");

    // Fill the ring exactly: `cap` flushes, no read in between, must all
    // stay outstanding - the actual "N submissions in flight" pipelining.
    for _ in 0..cap {
        let step = be.step(0, &[&a, &b, &out], &[4], 4);
        be.submit(&[], &[step]);
        <VulkanBackend as backend_api::Backend>::flush(&be);
    }
    assert_eq!(
        be.async_inflight_count(),
        cap,
        "exactly {cap} flushes with no intervening read/poll_wait must leave \
         all {cap} ring slots outstanding"
    );

    // One more flush wraps the ring: it must wait for and retire the OLDEST
    // slot before reusing it, so the count never exceeds capacity.
    let step = be.step(0, &[&a, &b, &out], &[4], 4);
    be.submit(&[], &[step]);
    <VulkanBackend as backend_api::Backend>::flush(&be);
    assert_eq!(
        be.async_inflight_count(),
        cap,
        "the (cap+1)th flush must retire the slot it reused, not grow past \
         the ring's own capacity"
    );

    be.poll_wait();
    assert_eq!(be.async_inflight_count(), 0, "poll_wait must retire every outstanding submission");
}

#[test]
fn dependent_batches_stay_correct_across_many_ring_wraparounds() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(be) = backend() else { return };
    if !be.async_capable() {
        brain_testutil::skip_unavailable("device has no timeline semaphore (defensive fallback only)");
        return;
    }
    let cap = be.async_ring_capacity();

    // Ping-pong accumulation: `next = prev + delta`, `cap * 3 + 1` times (several
    // full ring wraparounds), with NO read/poll_wait in between - only
    // `Backend::flush()`, so every iteration stays on the asynchronous path
    // and each one's dispatch genuinely depends on the previous submission's
    // write to a DIFFERENT command buffer (a cross-submission RAW hazard,
    // not just the within-one-batch hazard `perf_contract.rs` already
    // covers). If cross-submission ordering on the shared timeline / queue
    // were wrong, or a ring slot's command buffer were reused while its
    // previous submission was still executing, this either computes the
    // wrong sum, hangs, or - this exact hardware/backend combination has a
    // real history of exactly this failure mode from unsynchronised GPU
    // access - segfaults or reports `DEVICE_LOST`.
    let mut cur = be.storage_init("acc0", &[0.0, 0.0, 0.0, 0.0]);
    let mut nxt = be.storage_init("acc1", &[0.0, 0.0, 0.0, 0.0]);
    let delta = be.storage_init("delta", &[1.0, 1.0, 1.0, 1.0]);
    let iters = cap * 3 + 1;
    for _ in 0..iters {
        let step = be.step(0, &[&cur, &delta, &nxt], &[4], 4);
        be.submit(&[], &[step]);
        <VulkanBackend as backend_api::Backend>::flush(&be);
        std::mem::swap(&mut cur, &mut nxt);
    }
    let got = be.read(&cur, 4);
    let want = vec![iters as f32; 4];
    assert_eq!(got, want, "{iters} chained cross-submission additions produced the wrong sum");
}
