// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The resident-instance table: which model instances are currently Hot (or Warm),
//! on which device, at what cost, and when each was last used — the state the
//! manager evicts against (least-recently-used first).

use std::collections::HashMap;

use crate::{Device, InstanceKey, MemCost, Tier};

/// One resident instance's bookkeeping.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub cost: MemCost,
    pub device: Device,
    /// Monotonic last-use stamp (larger = more recent).
    pub last_use: u64,
    /// How many times this instance has been used — the popularity signal a
    /// cost-aware eviction policy scores against.
    pub uses: u64,
    /// True while a job is actively running on this instance — it must not be evicted.
    pub pinned: bool,
    /// `Hot` (on `device`, ready to run) or `Warm` (demoted: the `Instance`
    /// itself is still alive, `device` is where it will be promoted back
    /// to next, `cost` is the *Warm* footprint from
    /// `ResidentModel::estimate_at`, charged against `Device::Cpu` — see
    /// `ResidencyManager::evict`/`claim`'s tier handling). `Cold` is not
    /// produced by anything in this crate yet.
    pub tier: Tier,
}

/// Access-ordered table of resident instances. Recency is tracked with a monotonic
/// counter (no linked list); eviction sorts the candidates by `last_use`.
#[derive(Default)]
pub struct Residents {
    map: HashMap<InstanceKey, Entry>,
    tick: u64,
}

impl Residents {
    pub fn new() -> Residents {
        Residents::default()
    }

    /// The current logical time (the last tick issued) — what eviction policies
    /// measure age against.
    pub fn now(&self) -> u64 {
        self.tick
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Record a newly-resident instance (or overwrite an existing one) as most-recent.
    pub fn insert(&mut self, key: InstanceKey, cost: MemCost, device: Device) {
        let last_use = self.next_tick();
        self.map.insert(key, Entry { cost, device, last_use, uses: 1, pinned: false, tier: Tier::Hot });
    }

    /// Move an already-resident entry to a new tier/device/cost in place —
    /// `ResidencyManager` calls this after a successful `demote`/`promote`,
    /// which changes where and how much an instance costs without it ever
    /// leaving residency (unlike `remove`, which drops it entirely). Not a
    /// touch: this is a placement change, not a use.
    pub fn retier(&mut self, key: &InstanceKey, cost: MemCost, device: Device, tier: Tier) {
        if let Some(e) = self.map.get_mut(key) {
            e.cost = cost;
            e.device = device;
            e.tier = tier;
        }
    }

    /// Mark `key` as just-used (most-recent). No-op if absent.
    pub fn touch(&mut self, key: &InstanceKey) {
        let t = self.next_tick();
        if let Some(e) = self.map.get_mut(key) {
            e.last_use = t;
            e.uses += 1;
        }
    }

    /// Pin/unpin an instance (pinned while a job runs on it).
    pub fn set_pinned(&mut self, key: &InstanceKey, pinned: bool) {
        if let Some(e) = self.map.get_mut(key) {
            e.pinned = pinned;
        }
    }

    pub fn remove(&mut self, key: &InstanceKey) -> Option<Entry> {
        self.map.remove(key)
    }

    pub fn get(&self, key: &InstanceKey) -> Option<&Entry> {
        self.map.get(key)
    }

    pub fn contains(&self, key: &InstanceKey) -> bool {
        self.map.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Evictable instances on `device` (unpinned), least-recently-used **first**.
    pub fn lru_on(&self, device: Device) -> Vec<(InstanceKey, Entry)> {
        let mut v: Vec<(InstanceKey, Entry)> = self
            .map
            .iter()
            .filter(|(_, e)| e.device == device && !e.pinned)
            .map(|(k, e)| (k.clone(), *e))
            .collect();
        v.sort_by_key(|(_, e)| e.last_use);
        v
    }

    /// All resident instances (for reporting).
    pub fn iter(&self) -> impl Iterator<Item = (&InstanceKey, &Entry)> {
        self.map.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;
    fn k(m: &str) -> InstanceKey {
        InstanceKey::new(m, "default")
    }

    #[test]
    fn lru_order_reflects_touches_and_skips_pinned() {
        let mut r = Residents::new();
        r.insert(k("a"), MemCost::new(GB, 0), Device::Gpu(0));
        r.insert(k("b"), MemCost::new(2 * GB, 0), Device::Gpu(0));
        r.insert(k("c"), MemCost::new(3 * GB, 0), Device::Gpu(0));
        // a is oldest so far; touch it so b becomes the LRU.
        r.touch(&k("a"));
        let order: Vec<String> = r.lru_on(Device::Gpu(0)).into_iter().map(|(k, _)| k.model).collect();
        assert_eq!(order, vec!["b", "c", "a"]);
        // pin b → it drops out of the eviction candidates.
        r.set_pinned(&k("b"), true);
        let order: Vec<String> = r.lru_on(Device::Gpu(0)).into_iter().map(|(k, _)| k.model).collect();
        assert_eq!(order, vec!["c", "a"]);
    }

    #[test]
    fn insert_defaults_to_hot_and_retier_moves_tier_device_and_cost_in_place() {
        let mut r = Residents::new();
        r.insert(k("a"), MemCost::new(GB, 0), Device::Gpu(0));
        assert_eq!(r.get(&k("a")).unwrap().tier, Tier::Hot);

        let warm_cost = MemCost::new(0, GB / 2);
        r.retier(&k("a"), warm_cost, Device::Cpu, Tier::Warm);
        let e = r.get(&k("a")).unwrap();
        assert_eq!(e.tier, Tier::Warm);
        assert_eq!(e.device, Device::Cpu);
        assert_eq!(e.cost, warm_cost);

        // retier on an absent key is a no-op, not a panic.
        r.retier(&k("nope"), warm_cost, Device::Cpu, Tier::Warm);
    }

    #[test]
    fn per_device_isolation() {
        let mut r = Residents::new();
        r.insert(k("x"), MemCost::new(GB, 0), Device::Gpu(0));
        r.insert(k("y"), MemCost::new(GB, 0), Device::Gpu(1));
        assert_eq!(r.lru_on(Device::Gpu(0)).len(), 1);
        assert_eq!(r.lru_on(Device::Gpu(1)).len(), 1);
        assert_eq!(r.lru_on(Device::Cpu).len(), 0);
    }
}
