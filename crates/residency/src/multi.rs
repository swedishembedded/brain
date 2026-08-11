// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multi-device placement: an instance that spans SEVERAL devices at once
//! (e.g. an int8 MoE model layer-sharded across two GPUs, neither of which
//! alone has room for the whole thing).
//!
//! This is the exception, not the rule — [`crate::Device`]/[`crate::MemCost`]/
//! [`crate::place::pick_device`] already handle the overwhelming majority of
//! resident models correctly (one instance, one device). This module is
//! purely ADDITIVE: nothing here changes the single-device path's behaviour,
//! and a model that never implements [`MultiDeviceResidentModel`] is entirely
//! unaffected.
//!
//! # Why this exists (and why it didn't before)
//!
//! `manager.rs`'s own module doc used to document the only workaround
//! available: "a model that spans multiple devices... reports the extra
//! footprint via `ResidentModel::estimate`'s `ram`/secondary accounting and
//! pins its own cards; the manager still tracks the primary-device budget."
//! That LIES to whichever budget doesn't see the real bytes — either the
//! secondary device's VRAM is invisible to the manager (double-booking risk:
//! another model could be placed on top of memory that is actually spoken
//! for), or it gets folded into `ram` (wrong pool, wrong eviction pressure).
//! This is exactly the cataloged "gates that lie" pattern.
//! [`MultiDeviceCost`] is the honest alternative: every
//! device an instance touches is named, with its own real byte count,
//! checked against its own real budget.
//!
//! # What this does NOT do yet
//!
//! [`pick_devices`] is a pure function, exactly like [`crate::place::
//! pick_device`]. `crate::manager::ResidencyManager::claim_multi` is the
//! synchronous integration: it reserves on every named device, evicts
//! single-device LRU victims per-device to make room when needed, and
//! unwinds every device's reservation on a failed `activate_multi` — see
//! that method's own doc, and `manager.rs`'s own module doc for the ONE
//! deliberate scope limit that integration still has (multi-device
//! residents are not themselves auto-evicted). `crate::executor::Executor`
//! layers the ASYNC dispatcher/lane integration on top (`register_multi`,
//! busy-tracking across every device a group spans, home-lane dispatch) —
//! see `executor.rs`'s own module doc. What is genuinely NOT wired yet:
//! `crates/stats`' `ModelStat`/`Instance` schema is single-device by
//! construction, so a multi-device resident does not show up in
//! `braintop`/the stats JSON: rendering it would need a real schema change
//! (something like a `devices: Vec<(Device, u64)>`-shaped field), not a bug
//! fix, and that is real, separable follow-up work, not done here. The
//! first (and so far only) real `MultiDeviceResidentModel`: the int8
//! dual-GPU Thinker,
//! `crates/omni/src/int8_thinker_resident.rs`.

use std::collections::HashSet;

use capability::Invocation;

use crate::budget::Budgets;
use crate::{Device, Instance, InstanceKey, ResidentModel};

/// An instance's memory footprint SPANNING one or more devices — the
/// multi-device generalisation of [`crate::MemCost`]. Most instances need
/// only one device ([`crate::MemCost`] stays the right type for those);
/// this is for the rare instance that genuinely occupies real bytes on
/// SEVERAL devices simultaneously.
#[derive(Clone, Debug, Default)]
pub struct MultiDeviceCost {
    /// `(device, bytes)` pairs — every device this instance would occupy,
    /// and exactly how many of ITS bytes (never double-counted against a
    /// different device, never folded into a different pool). At most one
    /// entry per device (enforced by [`MultiDeviceCost::new`]'s dedup panic
    /// — two entries for the same device is a caller bug, not a valid
    /// "spans it twice" state).
    per_device: Vec<(Device, u64)>,
    /// Host RAM this instance holds regardless of accelerator placement
    /// (staging buffers, host-side scratch) — same meaning as
    /// `MemCost::ram`, kept separate from `per_device` because RAM is not
    /// itself a placement decision the way a GPU/NPU slot is.
    ram: u64,
}

impl MultiDeviceCost {
    /// Panics on a duplicate device in `per_device` — a caller bug (a real
    /// instance occupies EACH device exactly once; reporting the same
    /// device twice would double-book or silently drop one figure
    /// depending on iteration order, neither of which should ever
    /// silently "work").
    pub fn new(per_device: Vec<(Device, u64)>, ram: u64) -> MultiDeviceCost {
        let mut seen = HashSet::with_capacity(per_device.len());
        for &(d, _) in &per_device {
            assert!(seen.insert(d), "MultiDeviceCost::new: device {d:?} named twice");
        }
        MultiDeviceCost { per_device, ram }
    }

    /// The devices this instance would occupy, in the order given to `new`.
    pub fn devices(&self) -> impl Iterator<Item = Device> + '_ {
        self.per_device.iter().map(|&(d, _)| d)
    }

    /// Bytes this instance occupies on `device` specifically — 0 if it does
    /// not touch that device at all (matches `MemCost::on`'s "0 means not
    /// placed there" convention).
    pub fn on(&self, device: Device) -> u64 {
        self.per_device.iter().find(|&&(d, _)| d == device).map(|&(_, b)| b).unwrap_or(0)
    }

    pub fn ram(&self) -> u64 {
        self.ram
    }

    /// Total accelerator bytes across every named device — for reporting
    /// ("this instance costs N bytes total"), never for budgeting against
    /// any ONE device's capacity (that would be exactly the "folds into the
    /// wrong pool" lie this type exists to avoid).
    pub fn total_accelerator_bytes(&self) -> u64 {
        self.per_device.iter().map(|&(_, b)| b).sum()
    }
}

/// A resident model that spans MULTIPLE devices at once — the exception,
/// not the rule (see this module's doc). A model implements this trait IN
/// ADDITION to [`ResidentModel`] (never instead of it — every existing
/// single-device consumer of `ResidentModel` stays exactly as it is,
/// completely unaware this trait exists), and a multi-device-aware caller
/// checks for it (e.g. via a downcast or a registry-level flag) before
/// falling back to the ordinary single-device `estimate`/`activate` path.
pub trait MultiDeviceResidentModel: ResidentModel {
    /// The multi-device footprint for `key` — parallel to
    /// [`ResidentModel::estimate`], but naming every device this instance
    /// would occupy instead of assuming one.
    ///
    /// **The `Executor` dispatcher-thread contract** (does not apply to a
    /// caller that only ever drives this through `ResidencyManager` directly,
    /// e.g. a synchronous test — but every registered-with-`Executor` model
    /// must honour it): once registered via `Executor::register_multi`, this
    /// is called **on the dispatcher thread, on every scheduling round, for
    /// every queued group of this model** (`ResidencyManager::placeable_multi`
    /// calls it to decide whether a group can even be considered this round).
    /// It therefore MUST be cheap — memoize the result rather than
    /// recomputing it (e.g. re-opening a checkpoint) on every call — and MUST
    /// NOT panic: a panic on the dispatcher thread kills the dispatcher, and
    /// every OTHER model on the server starts returning "executor worker
    /// gone", not just this one. Report an unavailable model (a bad
    /// checkpoint path, a config that can't be read) by returning a cost
    /// naming **zero devices** — `ResidencyManager::claim_multi` turns that
    /// into a clean, per-job [`crate::manager::ClaimError::Activate`] instead.
    fn estimate_multi(&self, key: &InstanceKey) -> MultiDeviceCost;

    /// Build the instance across EXACTLY the devices `estimate_multi`
    /// named (same set, any order) — parallel to
    /// [`ResidentModel::activate`]'s single-device contract.
    fn activate_multi(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String>;
}

/// Ergonomic default: an [`InstanceKey`] chosen the same way
/// [`ResidentModel::instance_key`] does, for a caller that only has the
/// action name + invocation and wants the multi-device cost for whatever
/// `model` would build. Exists so call sites don't need to duplicate
/// `model.instance_key(action, inv)` before calling `estimate_multi`.
pub fn estimate_for(model: &dyn MultiDeviceResidentModel, action: &str, inv: &Invocation) -> MultiDeviceCost {
    model.estimate_multi(&model.instance_key(action, inv))
}

/// Choose the SET of devices for a multi-device instance of `cost`: every
/// named device must independently fit its own portion (no borrowing free
/// space from a device the instance doesn't actually occupy, no double
/// counting). Devices in `exclude` are skipped, exactly like
/// [`crate::place::pick_device`]. Returns `None` if ANY named device lacks
/// room right now — an all-or-nothing placement (a multi-device instance
/// half-placed is not a valid state; the caller's eviction path, once it
/// exists, is what makes room on the missing device(s), mirroring how
/// [`crate::place::plan_eviction`] already works for the single-device
/// case).
///
/// Deliberately does NOT fall back to "pick a different device than `cost`
/// named" the way [`crate::place::pick_device`] searches across candidates
/// — a multi-device cost's device set is the CALLER's placement decision
/// (e.g. "shard layers 0..N/2 on gpu0, N/2..N on gpu1"), not something this
/// function is free to renegotiate; this function only checks that the
/// caller's chosen set actually has room, and reports the total.
pub fn pick_devices(cost: &MultiDeviceCost, budgets: &Budgets, exclude: &HashSet<Device>) -> Option<Vec<Device>> {
    let devices: Vec<Device> = cost.devices().collect();
    if devices.is_empty() {
        return None; // a multi-device cost naming zero devices is not placeable as "multi"
    }
    for &d in &devices {
        if exclude.contains(&d) {
            return None;
        }
        let need = cost.on(d);
        match budgets.get(d) {
            Some(b) if b.fits(need) => {}
            _ => return None,
        }
    }
    Some(devices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Budgets;

    const GB: u64 = 1 << 30;

    fn budgets_2gpu() -> Budgets {
        let mut b = Budgets::new();
        b.set(Device::Gpu(0), 24 * GB, 2 * GB);
        b.set(Device::Gpu(1), 24 * GB, 2 * GB);
        b.set(Device::Cpu, 128 * GB, 8 * GB);
        b
    }

    #[test]
    fn on_reports_exactly_the_named_devices_never_more() {
        let cost = MultiDeviceCost::new(vec![(Device::Gpu(0), 15 * GB), (Device::Gpu(1), 15 * GB)], 512 << 20);
        assert_eq!(cost.on(Device::Gpu(0)), 15 * GB);
        assert_eq!(cost.on(Device::Gpu(1)), 15 * GB);
        assert_eq!(cost.on(Device::Cpu), 0, "CPU was never named");
        assert_eq!(cost.on(Device::Gpu(2)), 0, "gpu2 was never named");
        assert_eq!(cost.total_accelerator_bytes(), 30 * GB);
        assert_eq!(cost.ram(), 512 << 20);
    }

    #[test]
    #[should_panic(expected = "named twice")]
    fn duplicate_device_panics_rather_than_silently_dropping_or_doubling() {
        let _ = MultiDeviceCost::new(vec![(Device::Gpu(0), 1 * GB), (Device::Gpu(0), 2 * GB)], 0);
    }

    #[test]
    fn pick_devices_requires_every_named_device_to_independently_fit() {
        let budgets = budgets_2gpu();
        // 15 GB + 15 GB: each card has 22 GB usable (24 - 2 reserve), so
        // both fit independently -- this is the whole point of the type,
        // neither card alone holds 30 GB but the SPLIT genuinely fits.
        let cost = MultiDeviceCost::new(vec![(Device::Gpu(0), 15 * GB), (Device::Gpu(1), 15 * GB)], 0);
        let placed = pick_devices(&cost, &budgets, &crate::place::no_exclude()).expect("both halves fit");
        assert_eq!(placed, vec![Device::Gpu(0), Device::Gpu(1)]);
    }

    #[test]
    fn pick_devices_is_all_or_nothing() {
        let mut budgets = budgets_2gpu();
        // Fill gpu1 so only 1 GB remains free there.
        budgets.alloc(Device::Gpu(1), 21 * GB);
        let cost = MultiDeviceCost::new(vec![(Device::Gpu(0), 15 * GB), (Device::Gpu(1), 15 * GB)], 0);
        assert!(
            pick_devices(&cost, &budgets, &crate::place::no_exclude()).is_none(),
            "gpu0 having room must not paper over gpu1 not having room"
        );
    }

    #[test]
    fn pick_devices_never_double_counts_across_devices() {
        // A cost that (incorrectly, if this function were buggy) could only
        // be satisfied by summing two GPUs' free space into one placement
        // decision must still fail: 20 GB on gpu0 alone does not fit gpu0's
        // 22 GB *usable* budget once gpu0 already holds 10 GB.
        let mut budgets = budgets_2gpu();
        budgets.alloc(Device::Gpu(0), 10 * GB);
        let cost = MultiDeviceCost::new(vec![(Device::Gpu(0), 20 * GB)], 0);
        assert!(pick_devices(&cost, &budgets, &crate::place::no_exclude()).is_none());
    }

    #[test]
    fn pick_devices_respects_exclude() {
        let budgets = budgets_2gpu();
        let cost = MultiDeviceCost::new(vec![(Device::Gpu(0), 1 * GB), (Device::Gpu(1), 1 * GB)], 0);
        let mut exclude = HashSet::new();
        exclude.insert(Device::Gpu(1));
        assert!(pick_devices(&cost, &budgets, &exclude).is_none(), "gpu1 excluded must block the whole placement");
    }

    #[test]
    fn empty_device_set_is_not_placeable() {
        let budgets = budgets_2gpu();
        let cost = MultiDeviceCost::new(vec![], 1 * GB);
        assert!(pick_devices(&cost, &budgets, &crate::place::no_exclude()).is_none());
    }
}
