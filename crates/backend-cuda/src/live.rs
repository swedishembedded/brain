// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Live-resource accounting: how many driver objects this process currently
//! owns through `backend-cuda`, by kind.
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
//! baseline taken before a workload is a leak in this process, by kind, with no
//! tolerance. A free the driver REFUSES is deliberately not decremented - the
//! object is still live, and hiding that would defeat the point.
//!
//! Cost: one relaxed atomic add per create and per free. Nothing is on a
//! dispatch path.

use std::sync::atomic::{AtomicU64, Ordering};

/// A point-in-time reading of every kind of driver object this crate owns.
///
/// Compare two readings with `==` (or field by field) to prove a workload
/// returned everything it took. Concurrent workloads in the same process make
/// the absolute values move; take both readings while the process is otherwise
/// idle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveResources {
    /// Outstanding `cuMemAlloc` blocks.
    pub device_allocs: u64,
    /// Bytes in those blocks, as requested (a zero-byte request counts as 1).
    pub device_bytes: u64,
    /// Outstanding `cuMemAllocHost` blocks.
    pub pinned_allocs: u64,
    /// Bytes in those blocks, as requested.
    pub pinned_bytes: u64,
    /// Outstanding `CUevent`s.
    pub events: u64,
    /// Outstanding `CUstream`s.
    pub streams: u64,
    /// Outstanding `CUgraph`s (recorded, not instantiated).
    pub graphs: u64,
    /// Outstanding `CUgraphExec`s.
    pub graph_execs: u64,
    /// Outstanding loaded `CUmodule`s.
    pub modules: u64,
    /// Outstanding `cuDevicePrimaryCtxRetain` retains not yet released.
    pub primary_retains: u64,
    /// Outstanding `CUmemoryPool`s (stream-ordered allocation pools).
    pub mem_pools: u64,
    /// Outstanding blocks handed out from a pool and not yet freed. A freed
    /// block stays in its pool, bounded by the pool's release threshold, and
    /// goes back to the driver when the pool is trimmed or destroyed.
    pub pool_allocs: u64,
    /// Bytes in those blocks, as requested (a zero-byte request counts as 1).
    pub pool_bytes: u64,
}

/// Monotonic, process-wide totals of the calls whose per-step repetition is a
/// performance defect - the counterpart of [`LiveResources`], which proves
/// balance but cannot see an allocate-and-free pair per step.
///
/// A steady-state decode token that allocates, launches one kernel at a time or
/// fails to replay shows here by differencing two readings around it; none of
/// these ever decreases.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallTotals {
    /// `cuMemAlloc` calls made, whether or not freed since.
    pub device_alloc_calls: u64,
    /// Individual kernel launch calls the host made into the driver
    /// (a graph capture counts the launches it records).
    pub host_launches: u64,
    /// Captured graphs replayed.
    pub graph_replays: u64,
    /// Blocks taken from a stream-ordered pool (`cuMemAllocFromPoolAsync`),
    /// whether or not freed since: the pooled counterpart of
    /// [`Self::device_alloc_calls`].
    pub pool_alloc_calls: u64,
}

static TOTAL_DEVICE_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static TOTAL_HOST_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static TOTAL_GRAPH_REPLAYS: AtomicU64 = AtomicU64::new(0);
static TOTAL_POOL_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

/// See [`CallTotals`].
pub fn call_totals() -> CallTotals {
    CallTotals {
        device_alloc_calls: TOTAL_DEVICE_ALLOC_CALLS.load(Ordering::Relaxed),
        host_launches: TOTAL_HOST_LAUNCHES.load(Ordering::Relaxed),
        graph_replays: TOTAL_GRAPH_REPLAYS.load(Ordering::Relaxed),
        pool_alloc_calls: TOTAL_POOL_ALLOC_CALLS.load(Ordering::Relaxed),
    }
}

pub(crate) fn host_launched(n: u64) {
    TOTAL_HOST_LAUNCHES.fetch_add(n, Ordering::Relaxed);
}
pub(crate) fn graph_replayed() {
    TOTAL_GRAPH_REPLAYS.fetch_add(1, Ordering::Relaxed);
}

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
}

static COUNTERS: Counters = Counters {
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
};

/// What this process currently holds through `backend-cuda`. See
/// [`LiveResources`].
pub fn live_resources() -> LiveResources {
    let c = &COUNTERS;
    LiveResources {
        device_allocs: c.device_allocs.load(Ordering::Relaxed),
        device_bytes: c.device_bytes.load(Ordering::Relaxed),
        pinned_allocs: c.pinned_allocs.load(Ordering::Relaxed),
        pinned_bytes: c.pinned_bytes.load(Ordering::Relaxed),
        events: c.events.load(Ordering::Relaxed),
        streams: c.streams.load(Ordering::Relaxed),
        graphs: c.graphs.load(Ordering::Relaxed),
        graph_execs: c.graph_execs.load(Ordering::Relaxed),
        modules: c.modules.load(Ordering::Relaxed),
        primary_retains: c.primary_retains.load(Ordering::Relaxed),
        mem_pools: c.mem_pools.load(Ordering::Relaxed),
        pool_allocs: c.pool_allocs.load(Ordering::Relaxed),
        pool_bytes: c.pool_bytes.load(Ordering::Relaxed),
    }
}

fn up(n: &AtomicU64) {
    n.fetch_add(1, Ordering::Relaxed);
}

fn down(n: &AtomicU64) {
    n.fetch_sub(1, Ordering::Relaxed);
}

pub(crate) fn device_alloc(bytes: usize) {
    TOTAL_DEVICE_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
    up(&COUNTERS.device_allocs);
    COUNTERS.device_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
}
pub(crate) fn device_free(bytes: usize) {
    down(&COUNTERS.device_allocs);
    COUNTERS.device_bytes.fetch_sub(bytes as u64, Ordering::Relaxed);
}
pub(crate) fn pinned_alloc(bytes: usize) {
    up(&COUNTERS.pinned_allocs);
    COUNTERS.pinned_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
}
pub(crate) fn pinned_free(bytes: usize) {
    down(&COUNTERS.pinned_allocs);
    COUNTERS.pinned_bytes.fetch_sub(bytes as u64, Ordering::Relaxed);
}
pub(crate) fn event_created() {
    up(&COUNTERS.events);
}
pub(crate) fn event_destroyed() {
    down(&COUNTERS.events);
}
pub(crate) fn stream_created() {
    up(&COUNTERS.streams);
}
pub(crate) fn stream_destroyed() {
    down(&COUNTERS.streams);
}
pub(crate) fn graph_created() {
    up(&COUNTERS.graphs);
}
pub(crate) fn graph_destroyed() {
    down(&COUNTERS.graphs);
}
pub(crate) fn graph_exec_created() {
    up(&COUNTERS.graph_execs);
}
pub(crate) fn graph_exec_destroyed() {
    down(&COUNTERS.graph_execs);
}
pub(crate) fn module_loaded() {
    up(&COUNTERS.modules);
}
pub(crate) fn module_unloaded() {
    down(&COUNTERS.modules);
}
pub(crate) fn primary_retained() {
    up(&COUNTERS.primary_retains);
}
pub(crate) fn primary_released() {
    down(&COUNTERS.primary_retains);
}
pub(crate) fn mem_pool_created() {
    up(&COUNTERS.mem_pools);
}
pub(crate) fn mem_pool_destroyed() {
    down(&COUNTERS.mem_pools);
}
pub(crate) fn pool_alloc(bytes: usize) {
    TOTAL_POOL_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
    up(&COUNTERS.pool_allocs);
    COUNTERS.pool_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
}
pub(crate) fn pool_free(bytes: usize) {
    down(&COUNTERS.pool_allocs);
    COUNTERS.pool_bytes.fetch_sub(bytes as u64, Ordering::Relaxed);
}
