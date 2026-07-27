// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Placement + eviction decisions — the core of "use all the memory well".
//!
//! [`pick_device`] chooses where a new instance goes: the device with the most free
//! budget that already fits it (balancing models across GPUs). If nothing fits
//! outright, [`plan_eviction`] picks the least-recently-used victims on the best
//! candidate device to make room — never touching pinned (actively-running) or the
//! keep-set. Both are pure functions of the budget + resident tables, so they are
//! fully unit-tested without a GPU.

use std::collections::HashSet;

use crate::budget::Budgets;
use crate::lru::Residents;
use crate::{Device, InstanceKey, MemCost};

/// An empty exclusion set (the common single-lane case).
pub fn no_exclude() -> HashSet<Device> {
    HashSet::new()
}

/// Choose a device for a new instance of `cost` that fits **right now** (respecting
/// each device's reserved headroom). Prefers the GPU with the most free bytes
/// (spreads load); falls back to the CPU/RAM pool for a CPU-resident model. Devices
/// in `exclude` are skipped (used by the parallel scheduler to avoid a device a lane
/// is already running on). Returns `None` if none has room without eviction.
pub fn pick_device(cost: &MemCost, budgets: &Budgets, exclude: &HashSet<Device>) -> Option<Device> {
    // GPUs first, most-free wins (ties broken by lower index for determinism).
    let mut best: Option<(Device, u64)> = None;
    for d in budgets.gpus() {
        if cost.vram == 0 || exclude.contains(&d) {
            continue;
        }
        if let Some(b) = budgets.get(d) {
            if b.fits(cost.vram) {
                let free = b.free();
                if best.is_none_or(|(_, f)| free > f) {
                    best = Some((d, free));
                }
            }
        }
    }
    if let Some((d, _)) = best {
        return Some(d);
    }
    // CPU/RAM-resident model.
    if cost.ram > 0 && !exclude.contains(&Device::Cpu) {
        if let Some(b) = budgets.get(Device::Cpu) {
            if b.fits(cost.ram) {
                return Some(Device::Cpu);
            }
        }
    }
    None
}

/// The victims to evict from a device to fit `needed` bytes, and where to place the
/// new instance. Considers each GPU whose **usable** budget could ever hold `needed`
/// (a model bigger than a card minus reserve can never fit — that's a hard error the
/// caller surfaces), and picks the device that reaches `needed` free by evicting the
/// fewest bytes of LRU instances. `keep` names instances that must not be evicted
/// (the target itself and anything the caller is protecting).
pub struct EvictionPlan {
    pub device: Device,
    pub victims: Vec<InstanceKey>,
    pub freed: u64,
}

/// Plan eviction for a GPU-resident instance of `cost`, avoiding `exclude` devices.
/// Returns `None` only if no eligible GPU's usable budget can ever hold it.
pub fn plan_eviction(cost: &MemCost, budgets: &Budgets, residents: &Residents, keep: &[InstanceKey], exclude: &HashSet<Device>) -> Option<EvictionPlan> {
    let need = cost.vram;
    let mut best: Option<EvictionPlan> = None;
    for d in budgets.gpus() {
        if exclude.contains(&d) {
            continue;
        }
        let b = match budgets.get(d) {
            Some(b) => b,
            None => continue,
        };
        if b.usable() < need {
            continue; // can never fit here, even empty
        }
        if b.fits(need) {
            // Already fits with no eviction.
            return Some(EvictionPlan { device: d, victims: Vec::new(), freed: 0 });
        }
        // Evict LRU-first until free() would cover `need`.
        let mut deficit = need - b.free();
        let mut victims = Vec::new();
        let mut freed = 0u64;
        for (key, entry) in residents.lru_on(d) {
            if keep.contains(&key) {
                continue;
            }
            victims.push(key);
            freed += entry.cost.vram;
            if entry.cost.vram >= deficit {
                deficit = 0;
                break;
            }
            deficit -= entry.cost.vram;
        }
        if deficit == 0 {
            // This device can be made to fit; prefer the plan evicting the least.
            let plan = EvictionPlan { device: d, victims, freed };
            if best.as_ref().is_none_or(|p| plan.freed < p.freed) {
                best = Some(plan);
            }
        }
    }
    best.or_else(|| {
        // No GPU can be made to fit by eviction, but if some GPU is big enough when
        // empty we still report it (should be covered above); otherwise None.
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;
    fn ik(m: &str) -> InstanceKey {
        InstanceKey::new(m, "default")
    }
    fn vram(g: u64) -> MemCost {
        MemCost::new(g * GB, 0)
    }

    fn two_gpus() -> Budgets {
        let mut b = Budgets::new();
        b.set(Device::Gpu(0), 24 * GB, 2 * GB).set(Device::Gpu(1), 24 * GB, 2 * GB).set(Device::Cpu, 128 * GB, 8 * GB);
        b
    }

    #[test]
    fn picks_the_emptier_gpu() {
        let mut b = two_gpus();
        b.alloc(Device::Gpu(0), 15 * GB); // gpu0 has 7 free, gpu1 has 22
        assert_eq!(pick_device(&vram(6), &b, &no_exclude()), Some(Device::Gpu(1)));
    }

    #[test]
    fn none_when_full_then_eviction_frees_lru() {
        let mut b = two_gpus();
        // Fill both GPUs so a 13 GB model fits nowhere.
        b.alloc(Device::Gpu(0), 20 * GB);
        b.alloc(Device::Gpu(1), 20 * GB);
        assert_eq!(pick_device(&vram(13), &b, &no_exclude()), None);

        // Residents on gpu1: old=8GB (LRU), new=12GB.
        let mut r = Residents::new();
        r.insert(ik("old"), vram(8), Device::Gpu(1));
        r.insert(ik("new"), vram(12), Device::Gpu(1));
        r.touch(&ik("new"));
        // Need 13 GB; gpu1 free=2, deficit=11 → evicting `old`(8) alone isn't enough,
        // so both are evicted (LRU order: old then new).
        let plan = plan_eviction(&vram(13), &b, &r, &[], &no_exclude()).expect("a plan");
        assert_eq!(plan.device, Device::Gpu(1));
        assert_eq!(plan.victims, vec![ik("old"), ik("new")]);
        assert_eq!(plan.freed, 20 * GB);
    }

    #[test]
    fn evicting_one_lru_suffices() {
        let mut b = two_gpus();
        b.alloc(Device::Gpu(0), 20 * GB);
        b.alloc(Device::Gpu(1), 20 * GB);
        let mut r = Residents::new();
        r.insert(ik("big_old"), vram(14), Device::Gpu(0));
        r.insert(ik("small_new"), vram(6), Device::Gpu(0));
        r.touch(&ik("small_new"));
        // Need 13 on a card with free=2, deficit=11; big_old(14) is LRU and covers it.
        let plan = plan_eviction(&vram(13), &b, &r, &[], &no_exclude()).expect("plan");
        assert_eq!(plan.device, Device::Gpu(0));
        assert_eq!(plan.victims, vec![ik("big_old")]);
    }

    #[test]
    fn too_big_for_any_card_is_none() {
        let b = two_gpus(); // usable = 22 GB each
        let r = Residents::new();
        assert!(plan_eviction(&vram(23), &b, &r, &[], &no_exclude()).is_none());
    }

    #[test]
    fn keep_set_is_never_evicted() {
        let mut b = two_gpus();
        b.alloc(Device::Gpu(0), 22 * GB);
        b.alloc(Device::Gpu(1), 22 * GB);
        let mut r = Residents::new();
        r.insert(ik("keepme"), vram(22), Device::Gpu(0));
        // Only resident is protected → can't free enough on gpu0; gpu1 empty-usable=22<23? no, need 5.
        let plan = plan_eviction(&vram(5), &b, &r, &[ik("keepme")], &no_exclude());
        // gpu1 has nothing to evict and free=0 (22 alloc, 2 reserve) → deficit stays; gpu0 keepme protected.
        assert!(plan.is_none() || plan.unwrap().victims.iter().all(|v| v != &ik("keepme")));
    }
}
