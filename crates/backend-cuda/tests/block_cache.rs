// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! While a repeated pass asks for it, a freed device block is kept for the next
//! allocation of its size instead of going back to the driver - and every byte
//! of it is still returned when the pass ends or its owner goes.
//!
//! Swedish Embedded AB implements low-latency GPU inference services for its
//! clients, where a driver free on the hot path stalls the whole pipeline. If
//! your team needs expertise in device-memory reuse that stays leak-free and
//! bounded, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this exists
//!
//! `cuMemFree` waits for the device (it is an implicit synchronisation point),
//! so a prefill round that drops a dozen multi-megabyte activations per layer
//! spends its host time blocked on frees, in lock step with the card instead
//! of ahead of it. Sampled on the real 27B (`qwen35_gguf_prefill_profile`),
//! every host sample landed in `cuMemFree`. Reusing a freed block by size
//! removes the free and the matching allocation from the loop entirely.
//!
//! # What is asserted
//!
//! - **opt-in**: nothing is held until [`Context::hold_freed_blocks`] says so, so
//!   an idle handle frees exactly as it always did (`leaks.rs` holds that for
//!   every other owner), and turning it off returns what was held;
//! - **reuse**: a block freed and reallocated at its size stays allocated from
//!   the driver's point of view (the live counters do not dip) and is held by
//!   the cache in between;
//! - **bounded**: what the cache holds never exceeds [`Context::cache_cap_bytes`];
//!   an allocation that does not fit under it goes straight back to the driver;
//! - **returned**: dropping the context (and every block it handed out) returns
//!   the counters to baseline - the cache is not a leak, it is a bounded loan;
//! - **what is held**: any block up to half the cap - tiny per-step uniforms
//!   included, whose driver round trips add up - while a block larger than the
//!   cache could ever hold (a model weight) goes straight back to the driver;
//! - **zeroed storage**: `Backend::storage` still hands back zeros from a
//!   recycled block, because model code adds into fresh accumulators.
//!
//! The counters are process-global, so every test holds [`serial`]. Skipped
//! when there is no CUDA device.

use std::sync::{Mutex, MutexGuard};

use backend_api::Backend as _;
use backend_cuda::exec::Context;
use backend_cuda::{live_resources, CudaBackend};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const MIB: usize = 1 << 20;

/// A context with holding switched on - the state a prefill round runs in.
fn ctx() -> Option<Context> {
    match Context::open(0) {
        Ok(c) => {
            c.hold_freed_blocks(true);
            Some(c)
        }
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device: {e}"));
            None
        }
    }
}

#[test]
fn nothing_is_held_until_asked_and_turning_it_off_returns_what_was_held() {
    let _s = serial();
    let Some(ctx) = ctx() else { return };
    let base = live_resources();
    ctx.hold_freed_blocks(false);
    drop(ctx.alloc(4 * MIB).expect("alloc"));
    assert_eq!(ctx.cached_bytes(), 0, "a context that was not asked to hold anything holds nothing");
    assert_eq!(live_resources(), base, "an unasked-for free must reach the driver at once");

    ctx.hold_freed_blocks(true);
    drop(ctx.alloc(4 * MIB).expect("alloc"));
    drop(ctx.alloc(7 * MIB).expect("alloc"));
    assert_eq!(ctx.cached_bytes(), 11 * MIB as u64);
    assert_eq!(live_resources().device_allocs, base.device_allocs + 2, "held blocks are still allocated");
    ctx.hold_freed_blocks(false);
    assert_eq!(ctx.cached_bytes(), 0);
    assert_eq!(live_resources(), base, "turning holding off must return every held block");
}

#[test]
fn a_freed_block_is_held_for_the_next_allocation_of_its_size() {
    let _s = serial();
    let Some(ctx) = ctx() else { return };
    assert_eq!(ctx.cached_bytes(), 0, "a fresh context holds nothing");

    let first = ctx.alloc(4 * MIB).expect("alloc");
    let ptr = first.device_ptr();
    let allocs = live_resources().device_allocs;
    drop(first);
    assert_eq!(live_resources().device_allocs, allocs, "the freed block must stay allocated from the driver, held by the cache");
    assert_eq!(ctx.cached_bytes(), 4 * MIB as u64);

    let again = ctx.alloc(4 * MIB).expect("alloc");
    assert_eq!(again.device_ptr(), ptr, "the held block is the one reissued");
    assert_eq!(ctx.cached_bytes(), 0);
    assert_eq!(live_resources().device_allocs, allocs, "reissuing must not touch the driver");
    drop(again);
}

#[test]
fn a_block_too_large_for_the_cache_goes_straight_back() {
    let _s = serial();
    let Some(ctx) = ctx() else { return };
    let before = live_resources().device_allocs;
    // A step's uniform is a few words and is held like any other block...
    drop(ctx.alloc(256).expect("alloc"));
    assert_eq!(ctx.cached_bytes(), 256, "a tiny block is held too: freeing it costs the host a driver call");
    ctx.trim_cache();
    // ...but a block larger than half the cache could ever hold (a model weight)
    // is returned at once.
    let huge = (ctx.cache_cap_bytes() as usize).saturating_add(MIB);
    if let Ok(block) = ctx.alloc(huge) {
        drop(block);
        assert_eq!(ctx.cached_bytes(), 0, "a block larger than the cache could ever hold goes straight back");
    }
    assert_eq!(live_resources().device_allocs, before);
}

#[test]
fn the_cache_never_holds_more_than_its_cap() {
    let _s = serial();
    let Some(ctx) = ctx() else { return };
    let cap = ctx.cache_cap_bytes();
    let block = (cap / 4).clamp(MIB as u64, 64 * MIB as u64) as usize;
    let n = (cap as usize / block) + 3;
    let held: Vec<_> = (0..n).map(|_| ctx.alloc(block).expect("alloc")).collect();
    let live = live_resources();
    drop(held);
    assert!(ctx.cached_bytes() <= cap, "cache holds {} bytes over a cap of {cap}", ctx.cached_bytes());
    assert!(ctx.cached_bytes() > 0, "the cache should keep what fits");
    let kept = (ctx.cached_bytes() / block as u64) as usize;
    assert_eq!(
        live_resources().device_allocs,
        live.device_allocs - (n - kept) as u64,
        "everything that did not fit under the cap must have gone back to the driver"
    );
}

#[test]
fn dropping_the_context_and_its_blocks_returns_everything() {
    let _s = serial();
    let Some(probe) = ctx() else { return };
    let base = live_resources();
    {
        let ctx = Context::open(0).expect("second handle");
        ctx.hold_freed_blocks(true);
        let blocks: Vec<_> = [2, 3, 5, 8].iter().map(|m| ctx.alloc(m * MIB).expect("alloc")).collect();
        drop(blocks);
        assert!(ctx.cached_bytes() > 0, "the blocks should be cached, or this proves nothing");
        // The context goes while the cache is still full.
    }
    // The second handle's stream and primary retain are back; so are its cached blocks.
    assert_eq!(live_resources(), base, "a context dropped with a full cache left device memory behind");
    drop(probe);
}

#[test]
fn a_block_outliving_its_context_is_still_freed() {
    let _s = serial();
    let Some(probe) = ctx() else { return };
    let base = live_resources();
    let straggler = {
        let ctx = Context::open(0).expect("second handle");
        ctx.hold_freed_blocks(true);
        ctx.alloc(6 * MIB).expect("alloc")
    };
    // The context is gone; the block must still be valid to drop, and free.
    drop(straggler);
    assert_eq!(live_resources(), base, "a block dropped after its context leaked");
    drop(probe);
}

#[test]
fn storage_from_a_recycled_block_is_zero() {
    let _s = serial();
    let Ok(b) = CudaBackend::try_new(&[]) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    b.hold_freed_blocks(true);
    let n = (2 * MIB / 4) as u64;
    let dirty = b.storage_init("dirty", &vec![7.0f32; n as usize]);
    drop(dirty);
    let clean = b.storage(n);
    assert!(b.read(&clean, n as usize).iter().all(|&x| x == 0.0), "recycled storage was not zeroed");
}
