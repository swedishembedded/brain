// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The process-wide memory authority.
//!
//! Two problems this crate exists to solve, neither of which
//! `crates/residency`'s per-device integer budgets can express on their own:
//!
//! 1. **Unified memory.** On a box with an integrated GPU (or NPU), the
//!    "device VRAM" and "system RAM" are the SAME physical bytes. Budgeting
//!    them as two independent pools (today's behaviour, before this crate)
//!    double-counts: placing 14 GiB on the GPU does not shrink what the CPU
//!    pool believes it has free, when in reality it must.
//! 2. **A live ceiling, not a snapshot.** `/proc/meminfo`'s `MemAvailable`
//!    (and a cgroup's `memory.max`) change while a process runs — another
//!    process can take RAM mid-generation. A budget probed once at startup
//!    and never refreshed cannot see that.
//!
//! [`Topology`] declares which [`Device`]s share a physical [`PoolId`].
//! [`PoolProbe`] is the injectable live view of a pool (a real one reads
//! `/proc/meminfo` + cgroup v2; every test uses [`FixedProbe`] and never
//! touches a machine). [`MemoryAuthority`] answers "may I allocate N bytes on
//! device D right now" as a [`Grant`] (RAII — the free mechanism, since
//! `gpu_core::DeviceBuffer` has no size and no drop hook by design: it is
//! `Arc<Erased>`, clones alias, and a size field would lie for a sub-range
//! binding) or a [`Denied`] that distinguishes "try again later" from "this
//! can never fit, don't bother planning eviction for it".
//!
//! This is a leaf crate (`std` only, no GPU, no model code) precisely so both
//! `crates/residency` (cross-instance placement) and `crates/weightset`
//! (within-instance weight residency, checked mid-generation) can depend on
//! it without depending on each other.
//!
//! # Wiring status — read before trusting the name (audit F12)
//!
//! Despite the title above, [`MemoryAuthority`]/[`Grant`]/[`Topology`] have
//! **no production consumer yet**: the LIVE accounting every placement
//! decision actually runs through is `residency::Budgets` (whose
//! `set_pool` carries the unified-memory fix), and the only production uses
//! of this crate are [`PoolId`]/[`HOST_POOL`] (as `Budgets`' pool key type)
//! and one [`HostProbe::available`] call. The authority half is the
//! *intended* future single owner — wiring it into residency/weightset (or
//! folding `Budgets`' pool layer into it) is the open follow-up; until that
//! lands, changes to `Budgets` are what change behaviour, not changes here.
//! The `request` transaction is race-free (check+charge under one lock), so
//! wiring it later does not inherit an over-commit bug.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A device that can hold a hot model instance. The canonical home for this
/// type — `residency::Device` is `pub use memauth::Device;`, so every
/// existing `residency::Device` path compiles unchanged.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Device {
    /// A GPU by canonical index — the physical card `gpu<i>` names in the
    /// device registry (`gpu_core::devices`, PCI-bus order).
    Gpu(u32),
    /// RAM-resident, CPU-executed (e.g. the CPU encoder path).
    Cpu,
    /// An Intel NPU by index (whole-graph OpenVINO device).
    Npu(u32),
}

/// A physically distinct memory pool. Two [`Device`]s backed by the same
/// silicon (an integrated GPU and the CPU, on a unified-memory box) share a
/// `PoolId` — that is the entire double-counting fix. Opaque and orderable
/// only so it can be a stable map key / sorted for display; the numeric
/// value carries no meaning of its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct PoolId(pub u32);

/// The pool every [`Device`] belongs to until [`Topology::declare`] says
/// otherwise. Correct, with zero setup, for a discrete-GPU box (CPU RAM and
/// GPU VRAM really are separate); a unified-memory box's caller declares the
/// iGPU into this pool alongside `Device::Cpu`.
pub const HOST_POOL: PoolId = PoolId(0);

/// Which physical pool each [`Device`] draws from. Declared by the caller
/// from real hardware topology (on the brain side, `backend_api::DeviceClass`
/// — `IntegratedGpu`/`Npu` into [`HOST_POOL`], each `DiscreteGpu` its own
/// pool) once at startup. `memauth` never inspects a backend itself.
#[derive(Clone, Debug, Default)]
pub struct Topology {
    of: HashMap<Device, PoolId>,
}

impl Topology {
    pub fn new() -> Topology {
        Topology::default()
    }

    /// Declare that `d`'s memory comes from pool `p`.
    pub fn declare(&mut self, d: Device, p: PoolId) -> &mut Self {
        self.of.insert(d, p);
        self
    }

    /// `d`'s pool: whatever was declared, else a stable per-device default —
    /// `Cpu` defaults to [`HOST_POOL`]; every `Gpu`/`Npu` index defaults to
    /// its OWN pool (so two undeclared discrete GPUs never collide with each
    /// other or with the host), reproducing today's "every device is its own
    /// budget" behaviour with zero declarations.
    pub fn pool_of(&self, d: Device) -> PoolId {
        *self.of.get(&d).unwrap_or(&default_pool(d))
    }
}

fn default_pool(d: Device) -> PoolId {
    match d {
        Device::Cpu => HOST_POOL,
        // Offset ranges keep every undeclared device's default pool unique
        // and disjoint from HOST_POOL (0) and from each other's index space.
        Device::Gpu(i) => PoolId(1_000_000 + i),
        Device::Npu(i) => PoolId(2_000_000 + i),
    }
}

/// A live view of one memory pool, injectable so every test uses a fake and
/// no test reads a real machine.
pub trait PoolProbe: Send + Sync {
    /// Physically present bytes in `pool`. Assumed constant for the
    /// process's lifetime (no hot-plugged memory).
    fn total(&self, pool: PoolId) -> u64;
    /// Bytes free in `pool` right now, from this probe's own external view —
    /// independent of what [`MemoryAuthority`] itself has charged.
    /// [`MemoryAuthority::headroom`] takes the min of this and its own
    /// ledger, so a probe that has no real external signal for a pool
    /// (nothing else competes for a discrete GPU brain owns exclusively) can
    /// simply return `total` — the ledger alone then governs correctly.
    fn available(&self, pool: PoolId) -> u64;
}

/// The test double: every [`MemoryAuthority`] test builds one of these.
/// [`FixedProbe::set_available`] is how a test simulates another process
/// taking (or a demotion releasing) RAM mid-run.
#[derive(Debug, Default)]
pub struct FixedProbe {
    totals: Mutex<HashMap<PoolId, u64>>,
    avail: Mutex<HashMap<PoolId, u64>>,
}

impl FixedProbe {
    pub fn new() -> FixedProbe {
        FixedProbe::default()
    }
    /// Declare `pool`'s total capacity and current availability.
    pub fn set(&self, pool: PoolId, total: u64, available: u64) -> &Self {
        self.totals.lock().unwrap().insert(pool, total);
        self.avail.lock().unwrap().insert(pool, available);
        self
    }
    /// Change only `pool`'s live availability — simulates the OS/cgroup view
    /// shifting without touching physical capacity.
    pub fn set_available(&self, pool: PoolId, available: u64) {
        self.avail.lock().unwrap().insert(pool, available);
    }
}

impl PoolProbe for FixedProbe {
    fn total(&self, pool: PoolId) -> u64 {
        *self.totals.lock().unwrap().get(&pool).unwrap_or(&0)
    }
    fn available(&self, pool: PoolId) -> u64 {
        *self.avail.lock().unwrap().get(&pool).unwrap_or(&0)
    }
}

/// The real probe: `/proc/meminfo` (`MemAvailable`, replacing
/// `crates/cli/src/run_cli.rs::query_ram_bytes` — its doc-comment rationale
/// for preferring `MemAvailable` over `MemTotal` on a unified-memory box
/// moves here verbatim) intersected with a cgroup v2 limit, for exactly one
/// declared "host" pool; every other pool falls back to a fixed total set by
/// the caller (a discrete GPU's reported VRAM), with `available == total` —
/// correct per [`PoolProbe::available`]'s doc, since nothing outside brain
/// competes for a discrete card's memory.
///
/// Prefers `MemAvailable` over `MemTotal`: on a unified-memory box (an
/// integrated GPU/NPU sharing system RAM, or just a box with something else
/// already resident) `MemTotal` includes bytes this process can never
/// actually get, so a budget sized from it books a placement that "fits"
/// against physical memory that is already committed elsewhere — a swap
/// cliff, not a valid placement. `MemAvailable` (present on any kernel since
/// 3.14) already accounts for reclaimable cache, which `MemTotal - used`
/// does not.
#[cfg(not(target_arch = "wasm32"))]
pub struct HostProbe {
    host_pool: PoolId,
    other_totals: HashMap<PoolId, u64>,
    meminfo_path: String,
    cgroup_procself_path: String,
    cgroup_root: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl HostProbe {
    pub fn new(host_pool: PoolId) -> HostProbe {
        HostProbe {
            host_pool,
            other_totals: HashMap::new(),
            meminfo_path: "/proc/meminfo".to_string(),
            cgroup_procself_path: "/proc/self/cgroup".to_string(),
            cgroup_root: "/sys/fs/cgroup".to_string(),
        }
    }

    /// Declare a non-host pool's fixed total (a discrete GPU's VRAM).
    pub fn with_pool_total(mut self, pool: PoolId, total: u64) -> HostProbe {
        self.other_totals.insert(pool, total);
        self
    }

    /// Test-only: point the meminfo/cgroup reads at fixture files instead of
    /// the real `/proc`/`/sys`, so the parsing logic is exercised without a
    /// real machine.
    #[cfg(test)]
    fn with_roots(mut self, meminfo_path: &str, cgroup_procself_path: &str, cgroup_root: &str) -> HostProbe {
        self.meminfo_path = meminfo_path.to_string();
        self.cgroup_procself_path = cgroup_procself_path.to_string();
        self.cgroup_root = cgroup_root.to_string();
        self
    }

    fn meminfo_field(&self, name: &str) -> Option<u64> {
        let text = std::fs::read_to_string(&self.meminfo_path).ok()?;
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<u64>().ok())
            .map(|kb| kb << 10)
    }

    fn meminfo(&self) -> (u64, u64) {
        let total = self.meminfo_field("MemTotal:").unwrap_or(0);
        let avail = self.meminfo_field("MemAvailable:").unwrap_or(total);
        (total, avail)
    }

    /// `min(memory.max, memory.high) - memory.current` for this process's
    /// cgroup v2 unified hierarchy, if one is in effect. `None` (no
    /// additional limit) if `/proc/self/cgroup` or the `memory.*` files are
    /// absent (no cgroup, or cgroup v1 — v1's `memory.limit_in_bytes` is not
    /// read here; a v1-only box falls back to meminfo alone) or declare
    /// `"max"` (unlimited) for both.
    fn cgroup_headroom(&self) -> Option<u64> {
        let procself = std::fs::read_to_string(&self.cgroup_procself_path).ok()?;
        // cgroup v2: a single line "0::<path>".
        let rel = procself.lines().find_map(|l| l.strip_prefix("0::"))?;
        let base = format!("{}{}", self.cgroup_root, rel);
        let read_limit = |file: &str| -> Option<u64> {
            let s = std::fs::read_to_string(format!("{base}/{file}")).ok()?;
            let s = s.trim();
            if s == "max" {
                None
            } else {
                s.parse::<u64>().ok()
            }
        };
        let max = read_limit("memory.max");
        let high = read_limit("memory.high");
        let limit = match (max, high) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }?;
        let current: u64 = std::fs::read_to_string(format!("{base}/memory.current")).ok()?.trim().parse().ok()?;
        Some(limit.saturating_sub(current))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl PoolProbe for HostProbe {
    fn total(&self, pool: PoolId) -> u64 {
        if pool == self.host_pool {
            self.meminfo().0
        } else {
            *self.other_totals.get(&pool).unwrap_or(&0)
        }
    }
    fn available(&self, pool: PoolId) -> u64 {
        if pool == self.host_pool {
            let (_, mem_avail) = self.meminfo();
            match self.cgroup_headroom() {
                Some(c) => mem_avail.min(c),
                None => mem_avail,
            }
        } else {
            *self.other_totals.get(&pool).unwrap_or(&0)
        }
    }
}

/// Why a [`MemoryAuthority::request`] was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Denied {
    /// Would fit an empty pool, but not right now — transient: retry once
    /// something else is demoted/dropped, or the probe reports more.
    WouldExceedPool { pool: PoolId, want: u64, headroom: u64 },
    /// Cannot fit even with EVERY OTHER resident evicted — permanent. A
    /// caller sees this and fails cleanly instead of planning eviction that
    /// could never succeed (or, worse, evicting everything and still OOMing).
    NeverFits { pool: PoolId, want: u64, usable: u64 },
}

struct Inner {
    topo: Topology,
    probe: Arc<dyn PoolProbe>,
    reserved: HashMap<PoolId, u64>,
    charged: Mutex<HashMap<PoolId, u64>>,
    memo: Mutex<HashMap<PoolId, (Instant, u64, u64)>>,
    memo_ttl: Duration,
}

impl Inner {
    fn release(&self, pool: PoolId, bytes: u64) {
        let mut charged = self.charged.lock().unwrap();
        let e = charged.entry(pool).or_insert(0);
        *e = e.saturating_sub(bytes);
    }
}

/// An RAII claim on `bytes` in a pool. Dropping it releases the charge — the
/// free mechanism, since nothing adds a drop hook to `gpu_core::DeviceBuffer`
/// (see the module doc). Bytes and their `Grant` are meant to live together
/// in one owned value: you cannot free the bytes without releasing the
/// grant, because there is nothing else that does the release.
pub struct Grant {
    pool: PoolId,
    bytes: u64,
    auth: Arc<Inner>,
}

impl Grant {
    pub fn pool(&self) -> PoolId {
        self.pool
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Grant {
    fn drop(&mut self) {
        self.auth.release(self.pool, self.bytes);
    }
}

impl std::fmt::Debug for Grant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grant").field("pool", &self.pool).field("bytes", &self.bytes).finish()
    }
}

/// One pool's live report — the shape `crates/stats`/braintop render so a
/// unified box shows ONE pool instead of two budgets that sum to more than
/// the machine.
#[derive(Clone, Copy, Debug)]
pub struct PoolReport {
    pub pool: PoolId,
    pub total: u64,
    pub available: u64,
    pub charged: u64,
    pub reserved: u64,
}

/// The process-wide memory authority. Cheap to clone (an `Arc` internally);
/// share one instance across the residency dispatcher and every within-
/// instance weight window.
#[derive(Clone)]
pub struct MemoryAuthority(Arc<Inner>);

impl MemoryAuthority {
    pub fn new(topo: Topology, probe: Arc<dyn PoolProbe>, reserved: HashMap<PoolId, u64>) -> MemoryAuthority {
        MemoryAuthority(Arc::new(Inner {
            topo,
            probe,
            reserved,
            charged: Mutex::new(HashMap::new()),
            memo: Mutex::new(HashMap::new()),
            memo_ttl: Duration::from_millis(250),
        }))
    }

    fn probed(&self, pool: PoolId) -> (u64, u64) {
        let mut memo = self.0.memo.lock().unwrap();
        if let Some((stamp, total, avail)) = memo.get(&pool) {
            if stamp.elapsed() < self.0.memo_ttl {
                return (*total, *avail);
            }
        }
        let total = self.0.probe.total(pool);
        let avail = self.0.probe.available(pool);
        memo.insert(pool, (Instant::now(), total, avail));
        (total, avail)
    }

    /// Force the next probe read to be live, bypassing the memo window.
    pub fn refresh_now(&self) {
        self.0.memo.lock().unwrap().clear();
    }

    /// Bytes `device`'s pool could grant right now — the tighter of the live
    /// probe's view and this authority's own charge ledger, minus the
    /// pool's reserved headroom. Taking the MIN is what makes this correct
    /// in both directions: it shrinks the instant another process takes
    /// memory (the probe drops), and it never double-counts this process's
    /// own already-charged bytes when the probe is stale (a device
    /// allocation does not always move `MemAvailable` promptly).
    pub fn headroom(&self, device: Device) -> u64 {
        let pool = self.0.topo.pool_of(device);
        let (total, avail) = self.probed(pool);
        let charged = *self.0.charged.lock().unwrap().get(&pool).unwrap_or(&0);
        let reserved = *self.0.reserved.get(&pool).unwrap_or(&0);
        avail.min(total.saturating_sub(charged)).saturating_sub(reserved)
    }

    /// The most `device`'s pool could EVER grant, even fully empty (reserved
    /// headroom aside) — used to tell "no room right now" apart from "can
    /// never fit".
    pub fn usable(&self, device: Device) -> u64 {
        let pool = self.0.topo.pool_of(device);
        let (total, _) = self.probed(pool);
        total.saturating_sub(*self.0.reserved.get(&pool).unwrap_or(&0))
    }

    /// Ask for `bytes` on `device`. `tag` is a short label for logging/
    /// observability only (not currently surfaced — reserved for the perf
    /// artifact wiring in a later phase).
    ///
    /// The check and the charge happen under ONE `charged` lock: the earlier
    /// shape computed `headroom()` (locking and releasing the ledger), then
    /// re-locked to add the charge — two concurrent requests could both pass
    /// the check and jointly exceed the pool, the exact over-commit this
    /// authority exists to make impossible.
    pub fn request(&self, device: Device, bytes: u64, _tag: &'static str) -> Result<Grant, Denied> {
        let pool = self.0.topo.pool_of(device);
        // Probe BEFORE taking the ledger lock (probed() takes the memo lock;
        // never nest the two).
        let (total, avail) = self.probed(pool);
        let reserved = *self.0.reserved.get(&pool).unwrap_or(&0);
        let usable = total.saturating_sub(reserved);
        if bytes > usable {
            return Err(Denied::NeverFits { pool, want: bytes, usable });
        }
        let mut charged = self.0.charged.lock().unwrap();
        let cur = *charged.get(&pool).unwrap_or(&0);
        // Same formula as `headroom()`, against the ledger value this lock
        // now protects through the charge below.
        let headroom = avail.min(total.saturating_sub(cur)).saturating_sub(reserved);
        if bytes > headroom {
            return Err(Denied::WouldExceedPool { pool, want: bytes, headroom });
        }
        *charged.entry(pool).or_insert(0) += bytes;
        Ok(Grant { pool, bytes, auth: self.0.clone() })
    }

    /// [`Self::request`] without the reason on failure, for a caller that
    /// only needs "did it work".
    pub fn try_request(&self, device: Device, bytes: u64) -> Option<Grant> {
        self.request(device, bytes, "").ok()
    }

    /// One report per pool this authority has ever charged against or been
    /// asked about (via `headroom`/`usable`/`request`) — good enough for
    /// `crates/stats`; a pool nobody has touched yet simply never appears
    /// (it would report zeroes anyway, which is not more informative).
    pub fn snapshot(&self) -> Vec<PoolReport> {
        let pools: std::collections::BTreeSet<PoolId> = {
            let memo = self.0.memo.lock().unwrap();
            let charged = self.0.charged.lock().unwrap();
            memo.keys().chain(charged.keys()).copied().collect()
        };
        pools
            .into_iter()
            .map(|pool| {
                let (total, avail) = self.probed(pool);
                let charged = *self.0.charged.lock().unwrap().get(&pool).unwrap_or(&0);
                let reserved = *self.0.reserved.get(&pool).unwrap_or(&0);
                PoolReport { pool, total, available: avail, charged, reserved }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn auth(topo: Topology, probe: &Arc<FixedProbe>, reserved: &[(PoolId, u64)]) -> MemoryAuthority {
        MemoryAuthority::new(topo, probe.clone(), reserved.iter().copied().collect())
    }

    #[test]
    fn unified_pool_is_not_double_counted() {
        // Gpu(0) and Cpu share ONE 30 GB pool -- the unified-memory fix.
        let probe = Arc::new(FixedProbe::new());
        probe.set(HOST_POOL, 30 * GB, 30 * GB);
        let mut topo = Topology::new();
        topo.declare(Device::Gpu(0), HOST_POOL);
        topo.declare(Device::Cpu, HOST_POOL);
        let a = auth(topo, &probe, &[]);

        assert_eq!(a.headroom(Device::Cpu), 30 * GB);
        let g = a.request(Device::Gpu(0), 14 * GB, "dit").expect("fits");
        assert_eq!(a.headroom(Device::Cpu), 16 * GB, "a GPU-side charge must reduce the CPU's headroom on a unified box");
        assert_eq!(a.headroom(Device::Gpu(0)), 16 * GB);
        drop(g);
        assert_eq!(a.headroom(Device::Cpu), 30 * GB, "releasing the grant must restore the shared pool");
    }

    #[test]
    fn discrete_pools_stay_independent_with_no_declaration() {
        // No topology declared: Gpu(0) and Cpu default to DIFFERENT pools --
        // today's discrete-GPU behaviour, unchanged.
        let probe = Arc::new(FixedProbe::new());
        probe.set(HOST_POOL, 20 * GB, 20 * GB);
        probe.set(default_pool(Device::Gpu(0)), 24 * GB, 24 * GB);
        let a = auth(Topology::new(), &probe, &[]);
        let _g = a.request(Device::Gpu(0), 14 * GB, "dit").expect("fits its own pool");
        assert_eq!(a.headroom(Device::Cpu), 20 * GB, "an undeclared GPU charge must not touch the CPU pool");
    }

    #[test]
    fn governor_shrinks_the_pool_when_another_process_takes_ram() {
        let probe = Arc::new(FixedProbe::new());
        probe.set(HOST_POOL, 20 * GB, 20 * GB);
        let a = auth(Topology::new(), &probe, &[]);
        assert!(a.try_request(Device::Cpu, 8 * GB).is_some());

        // Another process (outside this authority's charge ledger) takes RAM.
        probe.set_available(HOST_POOL, 4 * GB);
        a.refresh_now();
        assert_eq!(
            a.request(Device::Cpu, 8 * GB, "x").unwrap_err(),
            Denied::WouldExceedPool { pool: HOST_POOL, want: 8 * GB, headroom: 4 * GB },
            "a request that fit a moment ago must be refused once the live probe shrinks"
        );
    }

    #[test]
    fn never_fits_is_denied_before_any_allocation() {
        let probe = Arc::new(FixedProbe::new());
        probe.set(HOST_POOL, 30 * GB, 30 * GB);
        let a = auth(Topology::new(), &probe, &[]);
        assert_eq!(
            a.request(Device::Cpu, 40 * GB, "huge").unwrap_err(),
            Denied::NeverFits { pool: HOST_POOL, want: 40 * GB, usable: 30 * GB }
        );
        // Nothing was charged -- a failed permanent request costs nothing.
        assert_eq!(a.headroom(Device::Cpu), 30 * GB);
    }

    #[test]
    fn reserved_headroom_bounds_usable_not_just_available() {
        let probe = Arc::new(FixedProbe::new());
        probe.set(HOST_POOL, 30 * GB, 30 * GB);
        let a = auth(Topology::new(), &probe, &[(HOST_POOL, 4 * GB)]);
        assert_eq!(a.usable(Device::Cpu), 26 * GB);
        assert_eq!(a.headroom(Device::Cpu), 26 * GB);
        assert_eq!(a.request(Device::Cpu, 27 * GB, "x").unwrap_err(), Denied::NeverFits { pool: HOST_POOL, want: 27 * GB, usable: 26 * GB });
    }

    #[test]
    fn grant_release_is_symmetric_under_random_order() {
        let probe = Arc::new(FixedProbe::new());
        probe.set(HOST_POOL, 100 * GB, 100 * GB);
        let a = auth(Topology::new(), &probe, &[]);
        let mut grants = Vec::new();
        for i in 0..10 {
            grants.push(a.try_request(Device::Cpu, (i + 1) * GB).unwrap());
        }
        // Drop in reverse-ish (not insertion) order.
        grants.swap(0, 9);
        grants.swap(2, 7);
        drop(grants);
        assert_eq!(a.headroom(Device::Cpu), 100 * GB, "every grant released must leave the ledger exactly as it started");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cgroup_limit_beats_meminfo() {
        let dir = std::env::temp_dir().join(format!("memauth-cgroup-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let meminfo = dir.join("meminfo");
        std::fs::write(&meminfo, "MemTotal:       31831044 kB\nMemAvailable:   20971520 kB\n").unwrap();
        let procself = dir.join("cgroup");
        std::fs::write(&procself, "0::/user.slice/test\n").unwrap();
        let cgdir = dir.join("sys-fs-cgroup/user.slice/test");
        std::fs::create_dir_all(&cgdir).unwrap();
        std::fs::write(cgdir.join("memory.max"), "7523905536\n").unwrap(); // ~7 GiB, this box's real limit
        std::fs::write(cgdir.join("memory.high"), "max\n").unwrap();
        std::fs::write(cgdir.join("memory.current"), "3000000000\n").unwrap(); // ~2.8 GiB used

        let probe = HostProbe::new(HOST_POOL).with_roots(
            meminfo.to_str().unwrap(),
            procself.to_str().unwrap(),
            dir.join("sys-fs-cgroup").to_str().unwrap(),
        );
        // MemAvailable says 20 GiB free; the cgroup says only ~4.2 GiB
        // (7.0 GiB limit - 2.8 GiB current) -- the tighter number must win.
        let avail = probe.available(HOST_POOL);
        let expected = 7_523_905_536u64.saturating_sub(3_000_000_000);
        assert_eq!(avail, expected, "cgroup headroom must beat MemAvailable when it is tighter");
        assert!(avail < 20u64 << 30, "must be far below the raw MemAvailable reading");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn no_cgroup_falls_back_to_meminfo_alone() {
        let dir = std::env::temp_dir().join(format!("memauth-nocgroup-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let meminfo = dir.join("meminfo");
        std::fs::write(&meminfo, "MemTotal:       16000000 kB\nMemAvailable:   9000000 kB\n").unwrap();
        // No /proc/self/cgroup file at all -- e.g. cgroup v1, or none.
        let probe = HostProbe::new(HOST_POOL).with_roots(meminfo.to_str().unwrap(), dir.join("nonexistent-cgroup").to_str().unwrap(), dir.join("sys-fs-cgroup").to_str().unwrap());
        assert_eq!(probe.available(HOST_POOL), 9_000_000 * 1024);
        std::fs::remove_dir_all(&dir).ok();
    }
}
