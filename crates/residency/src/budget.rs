// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-device memory budgets with a reserved headroom.
//!
//! The manager accounts every resident instance's bytes against its device's
//! [`Budget`]; `--reserve-gb` keeps a slice permanently free (so a GPU is never
//! packed to the brim, leaving room for a job's transient activations and for the
//! OS/driver). `free()` is what new instances may use.

use std::collections::HashMap;

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

/// The full memory picture: one [`Budget`] per device (GPUs + the CPU/RAM pool).
#[derive(Clone, Debug, Default)]
pub struct Budgets {
    devices: HashMap<Device, Budget>,
}

impl Budgets {
    pub fn new() -> Budgets {
        Budgets { devices: HashMap::new() }
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
    }
    pub fn release(&mut self, device: Device, bytes: u64) {
        if let Some(b) = self.devices.get_mut(&device) {
            b.release(bytes);
        }
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
}
