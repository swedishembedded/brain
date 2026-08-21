// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `flush` must submit host uploads, not only dispatches.
//!
//! `Queue::write_buffer` does not talk to the GPU. wgpu copies the bytes into
//! a staging buffer it allocates for the call and records a copy into the
//! pending-writes encoder; nothing reaches the device, and no staging buffer
//! can be reclaimed, until the next `queue.submit`. So a phase that only
//! uploads - a model load, which is exactly this repo's largest upload
//! workload - holds every staging buffer it has ever allocated live at once
//! unless something submits along the way.
//!
//! That was measurable rather than theoretical. Tracing the Vulkan staging
//! allocator underneath `crates/gpu-core/tests/vram_overhead.rs` showed all
//! 1536 staging buffers of a 1 GiB chunked upload created before the first
//! one was released - a single unbroken run of allocations followed by a
//! single unbroken run of frees at device teardown, with no interleaving at
//! all. The upload had never been submitted, so the driver had to keep 1 GiB
//! of pinned staging alive for it, and wgpu-hal's staging-buffer pool (which
//! recycles a released buffer into the next upload) could never see a
//! released buffer to recycle.
//!
//! The cause was that `flush` returned early when the pending DISPATCH list
//! was empty, treating "nothing to do" as a property of dispatches alone.
//! Uploads are work too. This pins the contract that they are.
//!
//! Swedish Embedded AB implements GPU compute and memory-transfer paths for
//! its clients. If your team needs expertise in Vulkan/wgpu upload
//! scheduling and staging-memory behaviour, you can procure our services by
//! sending an email to info@swedishembedded.com.

use backend_api::{Backend, BufUsage};
use backend_wgpu::WgpuBackend;

fn backend() -> WgpuBackend {
    WgpuBackend::new(&[("axpy", kernels::AXPY)])
}

/// Queue submissions so far. `Backend::queue_submits` is not implemented by
/// this backend (it keeps the default 0), so read the counter that is.
fn submits(b: &WgpuBackend) -> u64 {
    b.stats().expect("wgpu backend reports stats").submits
}

/// A `flush` after host writes must actually submit them.
///
/// Without this, the writes sit in wgpu's pending-writes encoder for an
/// unbounded time - until some unrelated dispatch happens to submit - and
/// every staging buffer backing them stays live and pinned until then.
#[test]
fn flush_submits_pending_host_writes() {
    let b = backend();
    let buf = b.buffer("dst", 1024, BufUsage::STORAGE | BufUsage::COPY_DST);

    let before = submits(&b);
    b.write_at(&buf, 0, &[7u32; 256]);
    b.flush();
    let after = submits(&b);

    assert!(
        after > before,
        "flush left {} host-written words unsubmitted (submits {before} -> {after}); \
         the staging buffer behind that write cannot be reclaimed until something else submits",
        256
    );
}

/// ...and it must not manufacture a submission out of nothing. A `flush`
/// with neither dispatches nor writes outstanding is the idle path, and this
/// repo calls it freely (`poll_wait` does, once per step); turning it into a
/// queue submission would put a round trip in every one of them.
#[test]
fn flush_with_nothing_outstanding_does_not_submit() {
    let b = backend();
    let buf = b.buffer("dst", 1024, BufUsage::STORAGE | BufUsage::COPY_DST);
    b.write_at(&buf, 0, &[7u32; 256]);
    b.flush();

    let before = submits(&b);
    b.flush();
    b.flush();
    assert_eq!(
        submits(&b),
        before,
        "an idle flush submitted anyway; repeated flushes must be free"
    );
}
