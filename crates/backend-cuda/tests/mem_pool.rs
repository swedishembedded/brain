// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Held blocks live in a stream-ordered pool: bounded, returned, and cheap.
//!
//! Swedish Embedded AB implements low-latency GPU inference services for its
//! clients, where a free that waits for the device stalls every request behind
//! it. If your team needs expertise in stream-ordered device-memory pools that
//! stay bounded and leak-free, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # What is asserted
//!
//! - **pooled**: with holding on, a free-and-reallocate loop makes no
//!   `cuMemAlloc` call; every block comes from the pool, of any size, not only
//!   an exact repeat;
//! - **bounded**: after the stream synchronises, the pool holds at most the
//!   retention cap in idle bytes, however much was churned through it;
//! - **returned**: turning holding off, dropping the context, and dropping a
//!   block that outlives its context all bring every counter, the pool count
//!   included, back to baseline - and the driver's own accounting of this
//!   process with it;
//! - **cheaper**: the host cost of a prefill-shaped alloc/free round is lower
//!   pooled than going to the driver each time (a loose ratio, because the
//!   figures move with the machine; `BRAIN_PRINT_POOL_BENCH=1` prints them).
//!
//! Counters are process-global, so every test holds [`serial`]. Skipped when
//! there is no CUDA device or the device has no memory pools.

use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use backend_cuda::exec::Context;
use backend_cuda::{call_totals, live_resources};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const MIB: usize = 1 << 20;

/// A context holding blocks in its pool, or `None` (skip) where there is no
/// device or no pool.
fn pooled() -> Option<Context> {
    let c = match Context::open(0) {
        Ok(c) => c,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device: {e}"));
            return None;
        }
    };
    c.hold_freed_blocks(true);
    if !c.holds_in_pool() {
        brain_testutil::skip_unavailable("this device has no stream-ordered memory pool");
        return None;
    }
    Some(c)
}

#[test]
fn held_blocks_come_from_the_pool_and_cost_no_driver_allocation() {
    let _s = serial();
    let Some(ctx) = pooled() else { return };
    let base_calls = call_totals();
    let base = live_resources();
    // Different sizes each round: the exact-size cache could not serve these.
    for round in 0..8usize {
        let blocks: Vec<_> = (1..=6).map(|k| ctx.alloc((k + round % 3) * MIB).expect("alloc")).collect();
        let live = live_resources();
        assert_eq!(live.pool_allocs - base.pool_allocs, 6, "outstanding pooled blocks are counted");
        assert_eq!(live.device_allocs, base.device_allocs, "a pooled block is not a cuMemAlloc block");
        drop(blocks);
    }
    let calls = call_totals();
    assert_eq!(calls.device_alloc_calls, base_calls.device_alloc_calls, "the loop reached cuMemAlloc");
    assert_eq!(calls.pool_alloc_calls - base_calls.pool_alloc_calls, 48);
    assert_eq!(live_resources().pool_allocs, base.pool_allocs, "every pooled block was freed");
    assert!(ctx.cached_bytes() > 0, "freed blocks are retained for reuse");
}

#[test]
fn the_pool_never_retains_more_than_its_cap_once_the_stream_has_run() {
    let _s = serial();
    let Some(ctx) = pooled() else { return };
    let cap = ctx.cache_cap_bytes();
    let block = (cap / 4).clamp(MIB as u64, 64 * MIB as u64) as usize;
    let n = (cap as usize / block) * 3 + 3;
    // Hold three caps' worth at once, then free it all.
    let held: Vec<_> = (0..n).map(|_| ctx.alloc(block).expect("alloc")).collect();
    drop(held);
    ctx.sync().expect("sync");
    assert!(
        ctx.pool_reserved_bytes() <= cap,
        "the pool retains {} bytes over a cap of {cap} after the stream synchronised",
        ctx.pool_reserved_bytes()
    );
}

#[test]
fn everything_is_returned_when_holding_ends_or_the_context_goes() {
    let _s = serial();
    let Some(probe) = pooled() else { return };
    let base = live_resources();
    let own_before = brain_testutil::own_gpu_memory_mib();

    // Holding off returns what the pool kept.
    {
        let ctx = Context::open(0).expect("second handle");
        ctx.hold_freed_blocks(true);
        drop((0..4).map(|k| ctx.alloc((k + 8) * MIB).expect("alloc")).collect::<Vec<_>>());
        assert!(ctx.pool_reserved_bytes() > 0);
        ctx.hold_freed_blocks(false);
        assert_eq!(ctx.pool_reserved_bytes(), 0, "turning holding off must return the pool's idle bytes");
        assert_eq!(ctx.cached_bytes(), 0);
    }
    assert_eq!(live_resources(), base, "a context dropped after holding left something behind");

    // A block that outlives its context frees cleanly, and takes the pool and
    // the stream it needs down with it.
    let straggler = {
        let ctx = Context::open(0).expect("second handle");
        ctx.hold_freed_blocks(true);
        ctx.alloc(6 * MIB).expect("alloc")
    };
    assert_eq!(live_resources().pool_allocs, base.pool_allocs + 1);
    drop(straggler);
    assert_eq!(live_resources(), base, "a pooled block dropped after its context leaked");

    if let (Some(before), Some(now)) = (own_before, brain_testutil::own_gpu_memory_mib()) {
        assert!(now <= before + 48, "the driver attributes {now} MiB to this process against {before} MiB before");
    }
    drop(probe);
}

/// The host cost of one prefill-shaped round: a dozen multi-megabyte
/// activations and a few hundred small blocks, all allocated and freed. Driver
/// `cuMemFree` waits for the device; the pool's free is enqueued.
fn round_cost(ctx: &Context, rounds: usize) -> f64 {
    let sizes: Vec<usize> = (0..12).map(|k| (2 + k) * MIB).chain((0..256).map(|k| 256 + 64 * (k % 8))).collect();
    // Warm: first touch of every size, so both modes measure steady state.
    for _ in 0..2 {
        drop(sizes.iter().map(|&s| ctx.alloc(s).expect("alloc")).collect::<Vec<_>>());
    }
    ctx.sync().expect("sync");
    let t = Instant::now();
    for _ in 0..rounds {
        drop(sizes.iter().map(|&s| ctx.alloc(s).expect("alloc")).collect::<Vec<_>>());
    }
    let host = t.elapsed().as_secs_f64() / rounds as f64;
    ctx.sync().expect("sync");
    host
}

#[test]
fn a_pooled_round_costs_the_host_less_than_going_to_the_driver() {
    let _s = serial();
    let Some(ctx) = pooled() else { return };
    let rounds = 30;
    let pooled_cost = round_cost(&ctx, rounds);
    ctx.hold_freed_blocks(false);
    let direct_cost = round_cost(&ctx, rounds);
    ctx.use_stream_ordered_pool(false);
    ctx.hold_freed_blocks(true);
    let cache_cost = round_cost(&ctx, rounds);
    ctx.hold_freed_blocks(false);
    if std::env::var("BRAIN_PRINT_POOL_BENCH").is_ok() {
        eprintln!(
            "host cost of one prefill-shaped alloc/free round (268 blocks): driver {:.1} us, exact-size cache {:.1} us, pool {:.1} us",
            direct_cost * 1e6,
            cache_cost * 1e6,
            pooled_cost * 1e6
        );
    }
    assert!(
        pooled_cost < direct_cost,
        "pooled round {:.1} us is not cheaper than the driver's {:.1} us",
        pooled_cost * 1e6,
        direct_cost * 1e6
    );
}
