// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The five properties `gpu_core::scratch`'s aliasing argument rests on.
//!
//! Swedish Embedded AB implements device-memory reuse for inference engines for
//! its clients. If your team needs expertise in GPU allocator behaviour and the
//! aliasing rules that make buffer reuse safe then you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! Buffer identity is `DeviceBuffer::alloc_id` - two handles alias iff their
//! ids match - so every assertion here is on ids, not on contents. Contents
//! are what `crates/ltxv/tests/scratch_pool.rs` gates, at the level of a real
//! forward's output bits.
//!
//! One rule these tests have to obey to mean anything: an `alloc_id` is only
//! an identity while SOME handle to the allocation is alive. Read it off a
//! temporary and the allocator is free to hand the same address to the next
//! request, which reads as "these two aliased" when nothing aliased at all -
//! this file failed that way on its first run. So every id compared below is
//! taken from a buffer that is still held: by the test, or (which is the point
//! of the facility) by the arena.

use gpu_core::Gpu;

const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];

/// These tests run in parallel with each other and each opens a scope, which
/// would be a problem if they shared one arena. They do not: the arena is a
/// field of the `Gpu` HANDLE, and `testgpu::dev` hands every caller its own
/// handle on the shared device (`WeakGpu::upgrade` builds a fresh one), so no
/// serialisation is needed here.
fn dev() -> Option<Gpu> {
    (std::env::var("MOE_SKIP_GPU_TESTS").is_err()).then(|| gpu_core::testgpu::dev(&KERNELS))
}

/// Within one scope every request is a DIFFERENT allocation. A cursor that
/// failed to advance would hand two live operands of the same dispatch one
/// buffer, which is the worst thing this facility could do and is invisible in
/// any timing.
#[test]
fn one_scope_never_hands_out_the_same_buffer_twice() {
    let Some(gpu) = dev() else { return };
    let _s = gpu.scratch_scope();
    let bufs: Vec<_> = (0..8).map(|_| gpu.storage(1024)).collect();
    let mut ids: Vec<_> = bufs.iter().map(|b| b.alloc_id()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), bufs.len(), "a scope handed the same allocation to two live requests");
}

/// The next scope replays the same allocations, in the same order. This is the
/// whole win: no create, no destroy.
#[test]
fn a_released_buffer_comes_back_in_the_next_scope() {
    let Some(gpu) = dev() else { return };
    // These ids survive their scope because the ARENA still holds each buffer,
    // which is exactly the state under test.
    let first: Vec<_> = {
        let _s = gpu.scratch_scope();
        (0..6).map(|_| gpu.storage(4096)).collect::<Vec<_>>()
    }
    .iter()
    .map(|b| b.alloc_id())
    .collect();
    let second: Vec<_> = {
        let _s = gpu.scratch_scope();
        (0..6).map(|_| gpu.storage(4096)).collect::<Vec<_>>()
    }
    .iter()
    .map(|b| b.alloc_id())
    .collect();
    assert_eq!(first, second, "the arena re-allocated instead of replaying - the pooling is not happening at all");
    gpu.scratch_release();
}

/// A buffer the caller KEPT is never handed out again. This is the property
/// that makes a chained activation safe with no special case, and the one the
/// `is_unique` check implements. It is also the only gate that holds it:
/// deleting that check leaves `crates/ltxv/tests/scratch_pool.rs` green,
/// because the model whose activation it protects happens not to read that
/// buffer any more by the time the slot is rewritten. Mutation-verified here.
#[test]
fn a_buffer_still_held_by_the_caller_is_never_recycled() {
    let Some(gpu) = dev() else { return };
    // Scope 1: keep the third buffer alive past the scope, exactly as a block
    // forward keeps the activation it produced.
    let (ids, kept) = {
        let _s = gpu.scratch_scope();
        let bufs: Vec<_> = (0..5).map(|_| gpu.storage(2048)).collect();
        let ids: Vec<_> = bufs.iter().map(|b| b.alloc_id()).collect();
        let kept = bufs[2].clone();
        (ids, kept)
    };

    let next_bufs: Vec<_> = {
        let _s = gpu.scratch_scope();
        (0..5).map(|_| gpu.storage(2048)).collect::<Vec<_>>()
    };
    let next: Vec<_> = next_bufs.iter().map(|b| b.alloc_id()).collect();

    assert_ne!(next[2], kept.alloc_id(), "the arena recycled a buffer the caller still holds - a live operand can now be overwritten by a later dispatch");
    for i in [0, 1, 3, 4] {
        assert_eq!(next[i], ids[i], "slot {i} was released and must have been replayed, not re-allocated");
    }
    drop(kept);
    gpu.scratch_release();
}

/// Outside a scope, allocation is exactly what it always was. A model that
/// never opts in must not be able to observe the facility at all.
#[test]
fn outside_a_scope_every_allocation_is_fresh() {
    let Some(gpu) = dev() else { return };
    {
        let _s = gpu.scratch_scope();
        let _ = gpu.storage(512);
    }
    let a = gpu.storage(512);
    let b = gpu.storage(512);
    assert_ne!(a.alloc_id(), b.alloc_id(), "two allocations outside a scope aliased each other");
    gpu.scratch_release();
}

/// A slot asked for a different size is re-allocated rather than bound short.
/// A caller whose sequence is not in fact identical degrades to plain
/// allocation; it does not silently get a buffer that cannot hold its data.
///
/// The first scope's buffer is RELEASED before the second scope runs, so the
/// uniqueness guard has nothing to say here and the size check is the only
/// thing that can refuse the slot. Written the other way - keeping the small
/// buffer alive - the uniqueness guard refuses first and the test passes with
/// the size check deleted, which is how it was first written and what the
/// mutation run caught.
///
/// The assertion is on the arena's HELD WORDS rather than on buffer identity:
/// installing a replacement drops the slot's previous buffer, after which its
/// `alloc_id` is a dangling address the allocator may hand straight back.
#[test]
fn a_changed_size_is_re_allocated_not_reused() {
    let Some(gpu) = dev() else { return };
    {
        let _s = gpu.scratch_scope();
        let _small = gpu.storage(256);
    }
    assert_eq!(gpu.scratch_held(), (1, 256), "the arena should be holding exactly the one 256-word slot the first scope asked for");
    {
        let _s = gpu.scratch_scope();
        let _big = gpu.storage(65536);
    }
    assert_eq!(gpu.scratch_held(), (1, 65536), "a 256-word slot was handed back for a 65536-word request - the arena bound a buffer too small for its dispatch");
    gpu.scratch_release();
}
