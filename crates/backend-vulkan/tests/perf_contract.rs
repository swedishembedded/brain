// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-frame performance contract of the Vulkan backend.
//!
//! Inference issues hundreds of dispatches per frame. If building a dispatch
//! touches the GPU queue (a submit + fence wait), the frame serializes into
//! hundreds of host<->GPU round trips and an integrated GPU runs ~100x slower
//! than the same kernels batched — measured 9.3 s/frame for ZipDepth on Intel
//! Arc (MTL) against ~0.1 s of actual GPU work. These tests pin the contract
//! that makes batching real:
//!
//!   1. `step()` / `step_sliced()` are HOST-side work only — no queue submits.
//!   2. A steady-state frame loop (build steps -> submit -> read) performs a
//!      BOUNDED number of submits per frame (the flush + the readback), not
//!      O(dispatches).
//!   3. Transient uniform buffers are recycled across flushes — a camera loop
//!      must not grow GPU allocations per frame.
//!
//! All tests skip (pass trivially) when no Vulkan device is present, like
//! `gpu-core`'s `vulkan_dispatch_storage_and_readback`.

use backend_vulkan::VulkanBackend;

fn backend() -> Option<VulkanBackend> {
    match VulkanBackend::try_new(&[("add2", kernels::ADD2)]) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping (no Vulkan device): {e}");
            None
        }
    }
}

#[test]
fn step_creation_performs_no_queue_submits() {
    let Some(be) = backend() else { return };
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);

    let base = be.queue_submits();
    let steps: Vec<_> = (0..16).map(|_| be.step(0, &[&a, &b, &out], &[4], 4)).collect();
    assert_eq!(
        be.queue_submits() - base,
        0,
        "building a dispatch must not submit to the GPU queue (uniform writes \
         must go through mapped host-visible memory, not zero/upload commands)"
    );

    // The batch still computes the right thing.
    be.submit(&[], &steps);
    assert_eq!(be.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn frame_loop_submits_are_bounded_per_frame() {
    let Some(be) = backend() else { return };
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);

    // Warm frame: first-touch allocations (uniform pool, descriptor pool).
    let steps: Vec<_> = (0..32).map(|_| be.step(0, &[&a, &b, &out], &[4], 4)).collect();
    be.submit(&[], &steps);
    let _ = be.read(&out, 4);

    // Steady-state frame: 32 dispatches must cost O(1) submits (the batched
    // flush + the readback), NOT O(dispatches).
    let base = be.queue_submits();
    let steps: Vec<_> = (0..32).map(|_| be.step(0, &[&a, &b, &out], &[4], 4)).collect();
    be.submit(&[], &steps);
    let r = be.read(&out, 4);
    assert_eq!(r, vec![11.0, 22.0, 33.0, 44.0]);
    let submits = be.queue_submits() - base;
    assert!(
        submits <= 4,
        "a 32-dispatch frame performed {submits} queue submits — the batch is \
         not actually batching (expected <= 4: flush + readback copy)"
    );
}

#[test]
fn transient_uniforms_are_recycled_across_flushes() {
    let Some(be) = backend() else { return };
    let a = be.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let b = be.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
    let out = be.storage(4);

    // Three frames of 16 transient-uniform dispatches each. Without recycling
    // the live transient-uniform count grows by 16 per frame (a 30 fps camera
    // leaks ~7k buffers + descriptor sets per second); with it, the pool peaks
    // at one frame's worth.
    for _ in 0..3 {
        let steps: Vec<_> = (0..16).map(|_| be.step(0, &[&a, &b, &out], &[4], 4)).collect();
        be.submit(&[], &steps);
        assert_eq!(be.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
    }
    let live = be.transient_uniform_count();
    assert!(
        live <= 16,
        "{live} transient uniforms live after 3 flushed frames of 16 dispatches \
         — transients are not being recycled across flushes"
    );
}
