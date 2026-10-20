// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Memory accounting shared by every concurrent load: a [`MemoryBudget`] caps
//! the host (and pinned) staging bytes in flight and records the high-water
//! mark; a [`BoundedWorkspace`] is one load's handle on it.
//!
//! The contract is "staging is a configured number, independent of the largest
//! tensor": every scratch buffer a load needs for one tile is reserved as ONE
//! [`Lease`] before it is allocated. A lease waits until the bytes are free, so
//! loads running in parallel share the cap instead of each assuming the whole
//! of it. One lease per step means a load never holds part of its need while
//! waiting for the rest, so the loads cannot deadlock each other. Leases are
//! released on drop, i.e. on success, error, panic and cancellation alike.
//!
//! Process-local: this coordinates threads of one process, not separate
//! processes. File-backed pages of memory-mapped checkpoints are not heap and
//! are accounted separately ([`resident_file_bytes`]).

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Which capped pool a lease draws on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    /// Pageable host memory used to stage tiles.
    HostStaging,
    /// Page-locked host memory used for device transfers.
    Pinned,
}

/// A request that can never be satisfied or a budget that was misused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// The request alone exceeds the pool's capacity, so waiting cannot help.
    ExceedsCapacity { pool: Pool, requested: u64, capacity: u64 },
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let BudgetError::ExceedsCapacity { pool, requested, capacity } = self;
        write!(f, "{pool:?} staging request of {requested} bytes exceeds the {capacity}-byte budget; raise the budget or shrink the tile")
    }
}

impl std::error::Error for BudgetError {}

#[derive(Default)]
struct PoolState {
    capacity: u64,
    in_use: u64,
    high_water: u64,
}

struct Shared {
    pools: Mutex<[PoolState; 2]>,
    freed: Condvar,
}

fn idx(pool: Pool) -> usize {
    match pool {
        Pool::HostStaging => 0,
        Pool::Pinned => 1,
    }
}

/// A point-in-time view of one pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolStats {
    pub capacity: u64,
    pub in_use: u64,
    pub high_water: u64,
}

/// Shared, cloneable staging budget.
#[derive(Clone)]
pub struct MemoryBudget {
    shared: Arc<Shared>,
}

impl MemoryBudget {
    pub fn new(host_staging_bytes: u64, pinned_bytes: u64) -> MemoryBudget {
        let pools = [PoolState { capacity: host_staging_bytes, ..Default::default() }, PoolState { capacity: pinned_bytes, ..Default::default() }];
        MemoryBudget { shared: Arc::new(Shared { pools: Mutex::new(pools), freed: Condvar::new() }) }
    }

    /// Reserve `bytes` from `pool`, waiting until they are free.
    pub fn lease(&self, pool: Pool, bytes: u64) -> Result<Lease, BudgetError> {
        let mut pools = self.shared.pools.lock().unwrap_or_else(|e| e.into_inner());
        let capacity = pools[idx(pool)].capacity;
        if bytes > capacity {
            return Err(BudgetError::ExceedsCapacity { pool, requested: bytes, capacity });
        }
        while pools[idx(pool)].in_use + bytes > capacity {
            pools = self.shared.freed.wait(pools).unwrap_or_else(|e| e.into_inner());
        }
        let p = &mut pools[idx(pool)];
        p.in_use += bytes;
        p.high_water = p.high_water.max(p.in_use);
        Ok(Lease { shared: self.shared.clone(), pool, bytes, local: None })
    }

    pub fn stats(&self, pool: Pool) -> PoolStats {
        let pools = self.shared.pools.lock().unwrap_or_else(|e| e.into_inner());
        let p = &pools[idx(pool)];
        PoolStats { capacity: p.capacity, in_use: p.in_use, high_water: p.high_water }
    }

    /// Forget the recorded high-water marks (current usage is kept), to measure one phase.
    pub fn reset_high_water(&self) {
        let mut pools = self.shared.pools.lock().unwrap_or_else(|e| e.into_inner());
        for p in pools.iter_mut() {
            p.high_water = p.in_use;
        }
    }
}

/// Reserved staging bytes; released on drop.
pub struct Lease {
    shared: Arc<Shared>,
    pool: Pool,
    bytes: u64,
    local: Option<Arc<LocalPeak>>,
}

impl Lease {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        {
            let mut pools = self.shared.pools.lock().unwrap_or_else(|e| e.into_inner());
            pools[idx(self.pool)].in_use -= self.bytes;
        }
        if let Some(l) = &self.local {
            l.in_use.fetch_sub(self.bytes, Ordering::Relaxed);
        }
        self.shared.freed.notify_all();
    }
}

#[derive(Default)]
struct LocalPeak {
    in_use: AtomicU64,
    peak: AtomicU64,
}

/// One load's handle on a shared [`MemoryBudget`]: leases through it also feed
/// this load's own staging peak, so a report can say what THIS load used while
/// the budget's high-water covers every concurrent load.
#[derive(Clone)]
pub struct BoundedWorkspace {
    budget: MemoryBudget,
    local: Arc<LocalPeak>,
}

impl BoundedWorkspace {
    pub fn new(budget: MemoryBudget) -> BoundedWorkspace {
        BoundedWorkspace { budget, local: Arc::default() }
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// [`MemoryBudget::lease`], attributed to this workspace.
    pub fn lease(&self, pool: Pool, bytes: u64) -> Result<Lease, BudgetError> {
        let mut lease = self.budget.lease(pool, bytes)?;
        let now = self.local.in_use.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.local.peak.fetch_max(now, Ordering::Relaxed);
        lease.local = Some(self.local.clone());
        Ok(lease)
    }

    /// Most bytes this workspace held at once.
    pub fn peak(&self) -> u64 {
        self.local.peak.load(Ordering::Relaxed)
    }
}

/// Bytes of file-backed pages currently resident in this process (`RssFile`),
/// the cost of memory-mapped checkpoints that heap accounting cannot see.
/// `None` where the platform does not report it.
pub fn resident_file_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("RssFile:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_waits_for_free_bytes_and_high_water_never_exceeds_capacity() {
        let b = MemoryBudget::new(100, 0);
        let first = b.lease(Pool::HostStaging, 70).unwrap();
        let b2 = b.clone();
        let t = std::thread::spawn(move || {
            let l = b2.lease(Pool::HostStaging, 60).unwrap();
            l.bytes()
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(b.stats(Pool::HostStaging).in_use, 70, "second lease must still be waiting");
        drop(first);
        assert_eq!(t.join().unwrap(), 60);
        let s = b.stats(Pool::HostStaging);
        assert_eq!((s.in_use, s.high_water), (0, 70));
    }

    #[test]
    fn a_request_larger_than_the_pool_fails_instead_of_waiting_forever() {
        let b = MemoryBudget::new(10, 0);
        assert!(matches!(b.lease(Pool::HostStaging, 11), Err(BudgetError::ExceedsCapacity { .. })));
        assert!(b.lease(Pool::Pinned, 1).is_err());
    }

    #[test]
    fn workspace_peak_is_per_load_while_the_budget_sees_all_loads() {
        let b = MemoryBudget::new(1000, 0);
        let (w1, w2) = (BoundedWorkspace::new(b.clone()), BoundedWorkspace::new(b.clone()));
        let (l1, l2) = (w1.lease(Pool::HostStaging, 300).unwrap(), w2.lease(Pool::HostStaging, 200).unwrap());
        drop((l1, l2));
        assert_eq!((w1.peak(), w2.peak()), (300, 200));
        assert_eq!(b.stats(Pool::HostStaging).high_water, 500);
    }
}
