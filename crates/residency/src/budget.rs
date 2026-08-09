// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-device memory budgets with a reserved headroom.
//!
//! The manager accounts every resident instance's bytes against its device's
//! [`Budget`]; `--reserve-gb` keeps a slice permanently free (so a GPU is never
//! packed to the brim, leaving room for a job's transient activations and for the
//! OS/driver). `free()` is what new instances may use.

use std::collections::HashMap;

use memauth::PoolId;

use crate::Device;

/// One device's budget: total capacity, a reserved headroom kept free, and the
/// bytes currently accounted to resident instances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub total: u64,
    pub reserved: u64,
    pub used: u64,
}

impl Budget {
    pub fn new(total: u64, reserved: u64) -> Budget {
        Budget { total, reserved: reserved.min(total), used: 0 }
    }
    /// Bytes available to new instances (capacity minus reserved headroom minus used).
    pub fn free(&self) -> u64 {
        self.total.saturating_sub(self.reserved).saturating_sub(self.used)
    }
    /// The usable budget (capacity minus reserve), ignoring current use — the most a
    /// single instance could ever occupy here.
    pub fn usable(&self) -> u64 {
        self.total.saturating_sub(self.reserved)
    }
    pub fn fits(&self, bytes: u64) -> bool {
        bytes <= self.free()
    }
    pub fn alloc(&mut self, bytes: u64) {
        self.used += bytes;
    }
    pub fn release(&mut self, bytes: u64) {
        self.used = self.used.saturating_sub(bytes);
    }
}

/// The full memory picture: one [`Budget`] per device (GPUs + the CPU/RAM pool),
/// plus an optional layer of [`Budget`]s per physical [`PoolId`] a group of
/// devices shares (a unified-memory box's integrated GPU and CPU RAM are the
/// SAME bytes — see `memauth`'s module doc). With no pool declared,
/// [`Self::free_on`]/[`Self::usable_on`]/[`Self::fits_on`] are numerically
/// identical to the plain per-device [`Budget`] methods, so every caller and
/// test written before pools existed is unaffected.
#[derive(Clone, Debug, Default)]
pub struct Budgets {
    devices: HashMap<Device, Budget>,
    pools: HashMap<PoolId, Budget>,
    pool_of: HashMap<Device, PoolId>,
}

impl Budgets {
    pub fn new() -> Budgets {
        Budgets { devices: HashMap::new(), pools: HashMap::new(), pool_of: HashMap::new() }
    }
    /// Set a device's total capacity and reserved headroom.
    pub fn set(&mut self, device: Device, total: u64, reserved: u64) -> &mut Self {
        self.devices.insert(device, Budget::new(total, reserved));
        self
    }
    pub fn get(&self, device: Device) -> Option<&Budget> {
        self.devices.get(&device)
    }
    pub fn get_mut(&mut self, device: Device) -> Option<&mut Budget> {
        self.devices.get_mut(&device)
    }
    pub fn devices(&self) -> impl Iterator<Item = Device> + '_ {
        self.devices.keys().copied()
    }
    /// Declare that `members` physically share `total` bytes (`reserved` kept
    /// free across all of them) — the unified-memory fix. Each member's own
    /// per-device [`Budget`] (set via [`Self::set`]) is unaffected; the pool
    /// is an ADDITIONAL ceiling `free_on`/`usable_on`/`fits_on` also enforce.
    pub fn set_pool(&mut self, id: PoolId, members: &[Device], total: u64, reserved: u64) -> &mut Self {
        self.pools.insert(id, Budget::new(total, reserved));
        for &d in members {
            self.pool_of.insert(d, id);
        }
        self
    }
    /// The NPUs, sorted by index (deterministic placement order).
    pub fn npus(&self) -> Vec<Device> {
        let mut g: Vec<Device> = self.devices.keys().copied().filter(|d| matches!(d, Device::Npu(_))).collect();
        g.sort_by_key(|d| if let Device::Npu(i) = d { *i } else { u32::MAX });
        g
    }
    /// The GPUs, sorted by index (deterministic placement order).
    pub fn gpus(&self) -> Vec<Device> {
        let mut g: Vec<Device> = self.devices.keys().copied().filter(|d| matches!(d, Device::Gpu(_))).collect();
        g.sort_by_key(|d| match d {
            Device::Gpu(i) => *i,
            Device::Cpu | Device::Npu(_) => u32::MAX,
        });

        g
    }
    pub fn alloc(&mut self, device: Device, bytes: u64) {
        if let Some(b) = self.devices.get_mut(&device) {
            b.alloc(bytes);
        }
        if let Some(p) = self.pool_of.get(&device).copied() {
            if let Some(b) = self.pools.get_mut(&p) {
                b.alloc(bytes);
            }
        }
    }
    pub fn release(&mut self, device: Device, bytes: u64) {
        if let Some(b) = self.devices.get_mut(&device) {
            b.release(bytes);
        }
        if let Some(p) = self.pool_of.get(&device).copied() {
            if let Some(b) = self.pools.get_mut(&p) {
                b.release(bytes);
            }
        }
    }
    /// [`Budget::free`] for `device`, clamped by its pool's free bytes (if any
    /// pool was declared for it). Identical to the plain device-only figure
    /// when no pool is declared.
    pub fn free_on(&self, device: Device) -> u64 {
        let dev_free = self.devices.get(&device).map(|b| b.free()).unwrap_or(0);
        match self.pool_of.get(&device).and_then(|p| self.pools.get(p)) {
            Some(pool) => dev_free.min(pool.free()),
            None => dev_free,
        }
    }
    /// [`Budget::usable`] for `device`, clamped by its pool's usable bytes.
    pub fn usable_on(&self, device: Device) -> u64 {
        let dev_usable = self.devices.get(&device).map(|b| b.usable()).unwrap_or(0);
        match self.pool_of.get(&device).and_then(|p| self.pools.get(p)) {
            Some(pool) => dev_usable.min(pool.usable()),
            None => dev_usable,
        }
    }
    pub fn fits_on(&self, device: Device, bytes: u64) -> bool {
        bytes <= self.free_on(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    #[test]
    fn reserve_headroom_is_never_available() {
        let mut b = Budget::new(24 * GB, 4 * GB);
        assert_eq!(b.usable(), 20 * GB);
        assert_eq!(b.free(), 20 * GB);
        b.alloc(13 * GB);
        assert_eq!(b.free(), 7 * GB);
        assert!(b.fits(7 * GB));
        assert!(!b.fits(8 * GB)); // the 4 GB reserve stays free
        b.release(13 * GB);
        assert_eq!(b.free(), 20 * GB);
    }

    #[test]
    fn budgets_track_per_device_and_sort_gpus() {
        let mut bs = Budgets::new();
        bs.set(Device::Gpu(1), 24 * GB, 2 * GB).set(Device::Gpu(0), 24 * GB, 2 * GB).set(Device::Cpu, 128 * GB, 8 * GB);
        assert_eq!(bs.gpus(), vec![Device::Gpu(0), Device::Gpu(1)]);
        bs.alloc(Device::Gpu(0), 10 * GB);
        assert_eq!(bs.get(Device::Gpu(0)).unwrap().free(), 12 * GB);
        assert_eq!(bs.get(Device::Gpu(1)).unwrap().free(), 22 * GB);
        assert_eq!(bs.get(Device::Cpu).unwrap().free(), 120 * GB);
    }

    /// With no pool declared, `_on` must be numerically identical to the
    /// plain per-device methods — every caller/test written before pools
    /// existed stays correct unchanged.
    #[test]
    fn no_pool_declared_is_unchanged() {
        let mut bs = Budgets::new();
        bs.set(Device::Gpu(0), 24 * GB, 2 * GB).set(Device::Cpu, 128 * GB, 8 * GB);
        bs.alloc(Device::Gpu(0), 10 * GB);
        assert_eq!(bs.free_on(Device::Gpu(0)), bs.get(Device::Gpu(0)).unwrap().free());
        assert_eq!(bs.usable_on(Device::Cpu), bs.get(Device::Cpu).unwrap().usable());
        // Gpu(0): 24 GB total, 2 GB reserved, 10 GB allocated -> 12 GB free.
        assert!(bs.fits_on(Device::Gpu(0), 12 * GB));
        assert!(!bs.fits_on(Device::Gpu(0), 13 * GB));
    }

    /// The unified-memory fix at the `Budgets` layer: an integrated GPU and
    /// the CPU declared into one 30 GB pool. A charge on the GPU must reduce
    /// what the CPU sees as free, and neither ever exceeds the pool's own
    /// ceiling even though each also has its own (larger, individually
    /// meaningless) per-device budget.
    #[test]
    fn pool_layer_clamps_free_and_usable_across_declared_members() {
        use memauth::HOST_POOL;
        let mut bs = Budgets::new();
        // Each device's OWN budget claims 24 GB -- individually larger than
        // the 20 GB the pool says they physically share, so the clamp is
        // observable rather than coincidentally equal.
        bs.set(Device::Gpu(0), 24 * GB, 0).set(Device::Cpu, 24 * GB, 0);
        bs.set_pool(HOST_POOL, &[Device::Gpu(0), Device::Cpu], 20 * GB, 0);

        assert_eq!(bs.usable_on(Device::Gpu(0)), 20 * GB, "the pool's usable ceiling must clamp the larger 24 GB device budget");
        assert_eq!(bs.free_on(Device::Cpu), 20 * GB);

        bs.alloc(Device::Gpu(0), 8 * GB);
        // Device-local free would be 24-8=16 GB, but the pool (20-8=12 GB) is
        // now the tighter constraint, so THAT is what free_on must report.
        assert_eq!(bs.free_on(Device::Gpu(0)), 12 * GB, "the pool ceiling, not the device's own larger number, must win");
        assert_eq!(bs.free_on(Device::Cpu), 12 * GB, "the SAME 8 GB charge must also reduce the CPU's pool-clamped free (20-8)");

        bs.release(Device::Gpu(0), 8 * GB);
        assert_eq!(bs.free_on(Device::Cpu), 20 * GB, "releasing must restore the shared pool for the other member too");

        // A device with no pool declared (a genuinely separate GPU) is untouched.
        bs.set(Device::Gpu(1), 24 * GB, 0);
        bs.alloc(Device::Gpu(0), 10 * GB);
        assert_eq!(bs.free_on(Device::Gpu(1)), 24 * GB, "a pool charge on gpu0 must not affect an undeclared gpu1");
    }
}
