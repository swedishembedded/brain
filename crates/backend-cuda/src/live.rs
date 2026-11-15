// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Live-resource accounting: how many driver objects a handle family (a
//! handle opened on its own and the handles shared from it) and the whole
//! process currently own through `backend-cuda`, by kind; and the host-call
//! totals and device-byte tallies, scoped the same way.
//!
//! Swedish Embedded AB implements long-running GPU services for its clients,
//! where a leaked device allocation is an outage rather than a nuisance. If
//! your team needs expertise in proving an accelerator runtime returns every
//! byte and handle it takes, you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! # Why a counter and not just `cuMemGetInfo`
//!
//! Free memory moves whenever any other process on the card allocates, so it
//! can only ever be a tolerance check. These counters are exact: each is
//! incremented when the driver confirms a create and decremented when the
//! driver confirms the matching free, so a nonzero difference against a
//! baseline taken before a workload is a leak, by kind, with no tolerance. A
//! free the driver REFUSES is deliberately not decremented - the object is
//! still live, and hiding that would defeat the point.
//!
//! # Why per handle as well as per process
//!
//! A model's leak check reads its own handle's tally
//! (`backend_api::Backend::live_tally`): the process-wide figure moves with
//! every other test or model the process runs, so a check on it fails beside a
//! sibling that is still allocating. The process-wide figure stays for tools
//! that mean the whole process, and for the leak tests that run in a process of
//! their own because they also ask the driver.
//!
//! Cost: two relaxed atomic adds per create and per free (the handle's and the
//! process's). Nothing is on a dispatch path.

use std::sync::atomic::{AtomicU64, Ordering};

pub use backend_api::{CallTotals, LiveResources};

// The process-wide `CallTotals`: every handle's calls together. A figure that has to be one model's own is read from
// its handle instead (`backend_api::Backend::call_totals`, kept by `Calls`); this sum moves whenever any other handle
// of the process works.
static TOTAL_DEVICE_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static TOTAL_HOST_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static TOTAL_GRAPH_REPLAYS: AtomicU64 = AtomicU64::new(0);
static TOTAL_POOL_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static TOTAL_SUBMIT_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);
static TOTAL_HOST_WAITS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ZERO_CALLS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ZEROED_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_UNINIT_CALLS: AtomicU64 = AtomicU64::new(0);
static TOTAL_UNINIT_BYTES: AtomicU64 = AtomicU64::new(0);

/// The process-wide [`CallTotals`]: every handle's calls together, so it moves whenever any other handle of the process
/// works. One model's own figure is its handle's (`backend_api::Backend::call_totals`).
pub fn call_totals() -> CallTotals {
    CallTotals {
        device_alloc_calls: TOTAL_DEVICE_ALLOC_CALLS.load(Ordering::Relaxed),
        host_launches: TOTAL_HOST_LAUNCHES.load(Ordering::Relaxed),
        graph_replays: TOTAL_GRAPH_REPLAYS.load(Ordering::Relaxed),
        pool_alloc_calls: TOTAL_POOL_ALLOC_CALLS.load(Ordering::Relaxed),
        submit_nanos: TOTAL_SUBMIT_NANOS.load(Ordering::Relaxed),
        wait_nanos: TOTAL_WAIT_NANOS.load(Ordering::Relaxed),
        host_waits: TOTAL_HOST_WAITS.load(Ordering::Relaxed),
        zero_calls: TOTAL_ZERO_CALLS.load(Ordering::Relaxed),
        zeroed_bytes: TOTAL_ZEROED_BYTES.load(Ordering::Relaxed),
        uninit_calls: TOTAL_UNINIT_CALLS.load(Ordering::Relaxed),
        uninit_bytes: TOTAL_UNINIT_BYTES.load(Ordering::Relaxed),
    }
}

/// One handle's [`CallTotals`], scoped as [`InUse`] is: a handle opened on its own starts its own, a handle `share`
/// gives counts into its parent's. Every count also lands in the process-wide totals, so the two can never disagree
/// about what was counted.
#[derive(Debug, Default)]
pub(crate) struct Calls {
    device_alloc_calls: AtomicU64,
    host_launches: AtomicU64,
    graph_replays: AtomicU64,
    pool_alloc_calls: AtomicU64,
    submit_nanos: AtomicU64,
    wait_nanos: AtomicU64,
    host_waits: AtomicU64,
    zero_calls: AtomicU64,
    zeroed_bytes: AtomicU64,
    uninit_calls: AtomicU64,
    uninit_bytes: AtomicU64,
}

fn bump(own: &AtomicU64, total: &AtomicU64, n: u64) {
    own.fetch_add(n, Ordering::Relaxed);
    total.fetch_add(n, Ordering::Relaxed);
}

impl Calls {
    pub(crate) fn read(&self) -> CallTotals {
        CallTotals {
            device_alloc_calls: self.device_alloc_calls.load(Ordering::Relaxed),
            host_launches: self.host_launches.load(Ordering::Relaxed),
            graph_replays: self.graph_replays.load(Ordering::Relaxed),
            pool_alloc_calls: self.pool_alloc_calls.load(Ordering::Relaxed),
            submit_nanos: self.submit_nanos.load(Ordering::Relaxed),
            wait_nanos: self.wait_nanos.load(Ordering::Relaxed),
            host_waits: self.host_waits.load(Ordering::Relaxed),
            zero_calls: self.zero_calls.load(Ordering::Relaxed),
            zeroed_bytes: self.zeroed_bytes.load(Ordering::Relaxed),
            uninit_calls: self.uninit_calls.load(Ordering::Relaxed),
            uninit_bytes: self.uninit_bytes.load(Ordering::Relaxed),
        }
    }

    /// A `cuMemAlloc` the driver granted.
    pub(crate) fn device_allocated(&self) {
        bump(&self.device_alloc_calls, &TOTAL_DEVICE_ALLOC_CALLS, 1);
    }
    /// A block taken from a stream-ordered pool.
    pub(crate) fn pool_allocated(&self) {
        bump(&self.pool_alloc_calls, &TOTAL_POOL_ALLOC_CALLS, 1);
    }
    pub(crate) fn host_launched(&self, n: u64) {
        bump(&self.host_launches, &TOTAL_HOST_LAUNCHES, n);
    }
    pub(crate) fn graph_replayed(&self) {
        bump(&self.graph_replays, &TOTAL_GRAPH_REPLAYS, 1);
    }
    pub(crate) fn host_submitted(&self, nanos: u64) {
        bump(&self.submit_nanos, &TOTAL_SUBMIT_NANOS, nanos);
    }
    pub(crate) fn host_waited(&self, nanos: u64) {
        bump(&self.wait_nanos, &TOTAL_WAIT_NANOS, nanos);
        bump(&self.host_waits, &TOTAL_HOST_WAITS, 1);
    }
    pub(crate) fn left_uninitialised(&self, bytes: usize) {
        bump(&self.uninit_calls, &TOTAL_UNINIT_CALLS, 1);
        bump(&self.uninit_bytes, &TOTAL_UNINIT_BYTES, bytes as u64);
    }
    pub(crate) fn zeroed(&self, bytes: usize) {
        bump(&self.zero_calls, &TOTAL_ZERO_CALLS, 1);
        bump(&self.zeroed_bytes, &TOTAL_ZEROED_BYTES, bytes as u64);
        let bucket = (bytes.max(1).ilog2() as usize).min(ZERO_BUCKETS - 1);
        ZERO_CALLS_BY_SIZE[bucket].fetch_add(1, Ordering::Relaxed);
        ZERO_BYTES_BY_SIZE[bucket].fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

const ZERO_BUCKETS: usize = 48;
static ZERO_CALLS_BY_SIZE: [AtomicU64; ZERO_BUCKETS] = [const { AtomicU64::new(0) }; ZERO_BUCKETS];
static ZERO_BYTES_BY_SIZE: [AtomicU64; ZERO_BUCKETS] = [const { AtomicU64::new(0) }; ZERO_BUCKETS];

/// The clears counted in [`CallTotals`], by allocation size: element `i` is
/// `(calls, bytes)` for allocations of `2^i ..< 2^(i+1)` bytes. It says which
/// sizes dominate the cleared bytes, so a caller that clears storage it then
/// overwrites completely can be found.
pub fn zeroed_by_size() -> Vec<(u64, u64)> {
    (0..ZERO_BUCKETS)
        .map(|i| (ZERO_CALLS_BY_SIZE[i].load(Ordering::Relaxed), ZERO_BYTES_BY_SIZE[i].load(Ordering::Relaxed)))
        .collect()
}

/// Bytes in the device blocks (from the driver, the block cache or a pool) that owners of one handle's allocations
/// hold, and the most they have been since the peak was last reset: the accounting behind
/// `backend_api::Backend::memory_usage`.
///
/// One per handle, not one per process: a handle opened on its own (`CudaBackend::try_new*`, `new_like`) starts its
/// own, and a handle `share` gives charges its parent's, as the wgpu backend's device tally does. A block is
/// uncharged from the tally it was charged to, whichever handle drops it, so the figure for one model is its own
/// while other models allocate on the same card in the same process.
#[derive(Debug, Default)]
pub(crate) struct InUse {
    live: AtomicU64,
    peak: AtomicU64,
}

impl InUse {
    /// A block handed to an owner.
    pub(crate) fn alloc(&self, bytes: usize) {
        let now = self.live.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
        self.peak.fetch_max(now, Ordering::Relaxed);
    }

    /// A block given back by its owner (to the cache, a pool or the driver).
    pub(crate) fn free(&self, bytes: usize) {
        self.live.fetch_sub(bytes as u64, Ordering::Relaxed);
    }

    /// `(bytes in blocks an owner holds, the most they have been since [`Self::reset_peak`])`.
    pub(crate) fn read(&self) -> (u64, u64) {
        let now = self.live.load(Ordering::Relaxed);
        (now, self.peak.load(Ordering::Relaxed).max(now))
    }

    /// Start a new peak from the current figure.
    pub(crate) fn reset_peak(&self) {
        self.peak.store(self.live.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}


/// One set of [`LiveResources`] counters, as atomics.
struct Counters {
    device_allocs: AtomicU64,
    device_bytes: AtomicU64,
    pinned_allocs: AtomicU64,
    pinned_bytes: AtomicU64,
    events: AtomicU64,
    streams: AtomicU64,
    graphs: AtomicU64,
    graph_execs: AtomicU64,
    modules: AtomicU64,
    primary_retains: AtomicU64,
    mem_pools: AtomicU64,
    pool_allocs: AtomicU64,
    pool_bytes: AtomicU64,
    managed_allocs: AtomicU64,
    managed_bytes: AtomicU64,
    system_allocs: AtomicU64,
    system_bytes: AtomicU64,
    sparse_allocs: AtomicU64,
    sparse_backed_bytes: AtomicU64,
}

impl Counters {
    const fn new() -> Counters {
        Counters {
            device_allocs: AtomicU64::new(0),
            device_bytes: AtomicU64::new(0),
            pinned_allocs: AtomicU64::new(0),
            pinned_bytes: AtomicU64::new(0),
            events: AtomicU64::new(0),
            streams: AtomicU64::new(0),
            graphs: AtomicU64::new(0),
            graph_execs: AtomicU64::new(0),
            modules: AtomicU64::new(0),
            primary_retains: AtomicU64::new(0),
            mem_pools: AtomicU64::new(0),
            pool_allocs: AtomicU64::new(0),
            pool_bytes: AtomicU64::new(0),
            managed_allocs: AtomicU64::new(0),
            managed_bytes: AtomicU64::new(0),
            system_allocs: AtomicU64::new(0),
            system_bytes: AtomicU64::new(0),
            sparse_allocs: AtomicU64::new(0),
            sparse_backed_bytes: AtomicU64::new(0),
        }
    }

    fn read(&self) -> LiveResources {
        let at = |n: &AtomicU64| n.load(Ordering::Relaxed);
        LiveResources {
            device_allocs: at(&self.device_allocs),
            device_bytes: at(&self.device_bytes),
            pinned_allocs: at(&self.pinned_allocs),
            pinned_bytes: at(&self.pinned_bytes),
            events: at(&self.events),
            streams: at(&self.streams),
            graphs: at(&self.graphs),
            graph_execs: at(&self.graph_execs),
            modules: at(&self.modules),
            primary_retains: at(&self.primary_retains),
            mem_pools: at(&self.mem_pools),
            pool_allocs: at(&self.pool_allocs),
            pool_bytes: at(&self.pool_bytes),
            managed_allocs: at(&self.managed_allocs),
            managed_bytes: at(&self.managed_bytes),
            system_allocs: at(&self.system_allocs),
            system_bytes: at(&self.system_bytes),
            sparse_allocs: at(&self.sparse_allocs),
            sparse_backed_bytes: at(&self.sparse_backed_bytes),
        }
    }
}

// The process-wide `LiveResources`: every handle's objects together, for whole-process tools and for the leak tests
// that run in a process of their own. One model's objects are its handle's (`Live`).
static PROCESS: Counters = Counters::new();

/// What this process holds through `backend-cuda`, every handle together. It moves whenever any other handle of the
/// process creates or frees an object; what one model holds is its handle's tally
/// (`backend_api::Backend::live_tally`).
pub fn live_resources() -> LiveResources {
    PROCESS.read()
}

/// One handle's [`LiveResources`], scoped as [`InUse`] is: a handle opened on its own starts its own, a handle `share`
/// gives counts into its parent's. Every object remembers the tally it was counted into (through the context share it
/// holds) and is counted down from that one, whichever handle drops it. Every count also lands in the process-wide
/// figure, so the two can never disagree about what was counted.
pub(crate) struct Live(Counters);

impl Default for Live {
    fn default() -> Live {
        Live(Counters::new())
    }
}

impl std::fmt::Debug for Live {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.read().fmt(f)
    }
}

impl Live {
    pub(crate) fn read(&self) -> LiveResources {
        self.0.read()
    }

    fn up(&self, field: fn(&Counters) -> &AtomicU64, n: u64) {
        field(&self.0).fetch_add(n, Ordering::Relaxed);
        field(&PROCESS).fetch_add(n, Ordering::Relaxed);
    }

    fn down(&self, field: fn(&Counters) -> &AtomicU64, n: u64) {
        field(&self.0).fetch_sub(n, Ordering::Relaxed);
        field(&PROCESS).fetch_sub(n, Ordering::Relaxed);
    }

    pub(crate) fn device_alloc(&self, bytes: usize) {
        self.up(|c| &c.device_allocs, 1);
        self.up(|c| &c.device_bytes, bytes as u64);
    }
    pub(crate) fn device_free(&self, bytes: usize) {
        self.down(|c| &c.device_allocs, 1);
        self.down(|c| &c.device_bytes, bytes as u64);
    }
    pub(crate) fn pinned_alloc(&self, bytes: usize) {
        self.up(|c| &c.pinned_allocs, 1);
        self.up(|c| &c.pinned_bytes, bytes as u64);
    }
    pub(crate) fn pinned_free(&self, bytes: usize) {
        self.down(|c| &c.pinned_allocs, 1);
        self.down(|c| &c.pinned_bytes, bytes as u64);
    }
    pub(crate) fn event_created(&self) {
        self.up(|c| &c.events, 1);
    }
    pub(crate) fn event_destroyed(&self) {
        self.down(|c| &c.events, 1);
    }
    pub(crate) fn stream_created(&self) {
        self.up(|c| &c.streams, 1);
    }
    pub(crate) fn stream_destroyed(&self) {
        self.down(|c| &c.streams, 1);
    }
    pub(crate) fn graph_created(&self) {
        self.up(|c| &c.graphs, 1);
    }
    pub(crate) fn graph_destroyed(&self) {
        self.down(|c| &c.graphs, 1);
    }
    pub(crate) fn graph_exec_created(&self) {
        self.up(|c| &c.graph_execs, 1);
    }
    pub(crate) fn graph_exec_destroyed(&self) {
        self.down(|c| &c.graph_execs, 1);
    }
    pub(crate) fn module_loaded(&self) {
        self.up(|c| &c.modules, 1);
    }
    pub(crate) fn module_unloaded(&self) {
        self.down(|c| &c.modules, 1);
    }
    pub(crate) fn primary_retained(&self) {
        self.up(|c| &c.primary_retains, 1);
    }
    pub(crate) fn primary_released(&self) {
        self.down(|c| &c.primary_retains, 1);
    }
    pub(crate) fn mem_pool_created(&self) {
        self.up(|c| &c.mem_pools, 1);
    }
    pub(crate) fn mem_pool_destroyed(&self) {
        self.down(|c| &c.mem_pools, 1);
    }
    pub(crate) fn pool_alloc(&self, bytes: usize) {
        self.up(|c| &c.pool_allocs, 1);
        self.up(|c| &c.pool_bytes, bytes as u64);
    }
    pub(crate) fn pool_free(&self, bytes: usize) {
        self.down(|c| &c.pool_allocs, 1);
        self.down(|c| &c.pool_bytes, bytes as u64);
    }
    pub(crate) fn managed_alloc(&self, bytes: usize) {
        self.up(|c| &c.managed_allocs, 1);
        self.up(|c| &c.managed_bytes, bytes as u64);
    }
    pub(crate) fn managed_free(&self, bytes: usize) {
        self.down(|c| &c.managed_allocs, 1);
        self.down(|c| &c.managed_bytes, bytes as u64);
    }
    pub(crate) fn sparse_alloc(&self) {
        self.up(|c| &c.sparse_allocs, 1);
    }
    pub(crate) fn sparse_free(&self) {
        self.down(|c| &c.sparse_allocs, 1);
    }
    pub(crate) fn sparse_backed(&self, bytes: usize) {
        self.up(|c| &c.sparse_backed_bytes, bytes as u64);
    }
    pub(crate) fn sparse_unbacked(&self, bytes: usize) {
        self.down(|c| &c.sparse_backed_bytes, bytes as u64);
    }
    pub(crate) fn system_alloc(&self, bytes: usize) {
        self.up(|c| &c.system_allocs, 1);
        self.up(|c| &c.system_bytes, bytes as u64);
    }
    pub(crate) fn system_free(&self, bytes: usize) {
        self.down(|c| &c.system_allocs, 1);
        self.down(|c| &c.system_bytes, bytes as u64);
    }
}
