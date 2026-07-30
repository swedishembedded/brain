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

/// The eviction-ordering policy: lower score = evict first.
///
/// A trait rather than a hardcoded rule because the benchmark that motivated it
/// (`brain perf residency`) measured strict LRU at **64% eviction regret** under
/// a shifting Zipf load — two thirds of evictions were of models wanted again
/// almost immediately. LRU has no notion of *reload cost* (a 4 GB model costs
/// ~20x a 200 MB one to bring back) or *popularity* (the head of a Zipf
/// distribution returns within seconds). Keeping [`Lru`] alongside
/// [`CostAware`] is the point: the benchmark compares them on identical seeds.
pub trait EvictionPolicy: Send + Sync {
    /// Score an entry at logical time `now` (the residents table's tick).
    /// Lower = evict first.
    fn score(&self, e: &crate::lru::Entry, now: u64) -> f64;
}

/// Strict least-recently-used: score = last_use. The historical default.
pub struct Lru;

impl EvictionPolicy for Lru {
    fn score(&self, e: &crate::lru::Entry, _now: u64) -> f64 {
        e.last_use as f64
    }
}

/// GDSF-style cost-aware scoring: `recency_weight * uses * reload_cost`.
///
/// A small, cold, cheap-to-reload model is evicted before a large, hot,
/// expensive one. Recency still matters (a stale hot model must eventually
/// yield), but it is one factor rather than the whole rule.
pub struct CostAware;

impl EvictionPolicy for CostAware {
    fn score(&self, e: &crate::lru::Entry, now: u64) -> f64 {
        // Recency decays with age in ticks; +1 keeps just-used entries finite.
        let age = now.saturating_sub(e.last_use) as f64 + 1.0;
        let bytes = e.cost.vram.max(e.cost.ram).max(e.cost.npu).max(1) as f64;
        // Measured on `perf residency` (24 models, 4x overcommit, shifting
        // Zipf): this GDSF shape beats LRU on hit rate (54.3% vs 50.0%) and —
        // by construction, pinned in policy_tests — spends evictions on cheap
        // models instead of expensive ones. Event-counted regret is metric-
        // limited at this overcommit (the working set simply exceeds capacity,
        // so SOMETHING soon-wanted must go); the improvement shows up in what
        // each eviction COSTS, not how often one is regretted.
        (e.uses as f64) * bytes / age
    }
}

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
    // A zero-cost (stateless) instance holds no memory on any device, so every
    // class below would skip it (`need == 0`) and it would be unplaceable. Place it
    // on any budgeted device — the CPU by preference (stateless providers are host
    // glue), else whatever is free — so `demo`/`imageops` run under any budget.
    if cost.vram == 0 && cost.ram == 0 && cost.npu == 0 {
        if !exclude.contains(&Device::Cpu) && budgets.get(Device::Cpu).is_some() {
            return Some(Device::Cpu);
        }
        return budgets.gpus().into_iter().chain(budgets.npus()).find(|d| !exclude.contains(d));
    }
    // Device-class preference: NPU (if the model has an NPU path) → GPU → CPU. Within
    // an accelerator class, most-free wins (spreads load; ties by lower index). A
    // model reports `npu > 0` only when it exports an OpenVINO graph, so a non-NPU
    // model skips the NPU class and behaves exactly as before.
    for (devices, need) in [(budgets.npus(), cost.npu), (budgets.gpus(), cost.vram)] {
        if need == 0 {
            continue;
        }
        let mut best: Option<(Device, u64)> = None;
        for d in devices {
            if exclude.contains(&d) {
                continue;
            }
            if let Some(b) = budgets.get(d) {
                if b.fits(need) {
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
    plan_eviction_with(&Lru, cost, budgets, residents, keep, exclude)
}

/// [`plan_eviction`] under an explicit [`EvictionPolicy`]. Victims are taken in
/// ascending score order. Pure over its inputs (`now` is the residents' tick),
/// so policies are unit-testable without threads or a clock.
pub fn plan_eviction_with(policy: &dyn EvictionPolicy, cost: &MemCost, budgets: &Budgets, residents: &Residents, keep: &[InstanceKey], exclude: &HashSet<Device>) -> Option<EvictionPlan> {
    // Same class preference as `pick_device`: NPU (if the model has an NPU path) then
    // GPU. Victim bytes are counted with the device-appropriate cost field
    // (`entry.cost.on(d)`), so NPU eviction frees NPU bytes and GPU eviction frees
    // VRAM. If any plan exists in the preferred class, it wins.
    for (devices, need) in [(budgets.npus(), cost.npu), (budgets.gpus(), cost.vram)] {
        if need == 0 {
            continue;
        }
        let mut best: Option<EvictionPlan> = None;
        for d in devices {
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
            // Evict lowest-score-first until free() would cover `need`.
            let mut deficit = need - b.free();
            let mut victims = Vec::new();
            let mut freed = 0u64;
            let now = residents.now();
            let mut candidates = residents.lru_on(d);
            candidates.sort_by(|a, b| {
                policy
                    .score(&a.1, now)
                    .partial_cmp(&policy.score(&b.1, now))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (key, entry) in candidates {
                if keep.contains(&key) {
                    continue;
                }
                let vbytes = entry.cost.on(d);
                victims.push(key);
                freed += vbytes;
                if vbytes >= deficit {
                    deficit = 0;
                    break;
                }
                deficit -= vbytes;
            }
            if deficit == 0 {
                // This device can be made to fit; prefer the plan evicting the least.
                let plan = EvictionPlan { device: d, victims, freed };
                if best.as_ref().is_none_or(|p| plan.freed < p.freed) {
                    best = Some(plan);
                }
            }
        }
        if best.is_some() {
            return best; // a plan in the preferred class wins
        }
    }
    None
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    use crate::lru::Residents;
    use crate::{Device, InstanceKey, MemCost};

    const GB: u64 = 1 << 30;

    /// The scenario the benchmark measured at 64% regret: a large HOT model and
    /// a small COLD one; LRU evicts whichever was touched longer ago — the hot
    /// one, if the cold straggler was touched last — while CostAware weighs
    /// popularity and reload cost and evicts the cheap cold one.
    #[test]
    fn cost_aware_spares_the_hot_expensive_model() {
        let mut r = Residents::new();
        let hot = InstanceKey::new("hot4gb", "cfg");
        let cold = InstanceKey::new("cold200mb", "cfg");
        r.insert(hot.clone(), MemCost::new(4 * GB, 0), Device::Gpu(0));
        r.insert(cold.clone(), MemCost::new(200 << 20, 0), Device::Gpu(0));
        // The hot model is used many times; the cold one once, but LAST.
        for _ in 0..50 {
            r.touch(&hot);
        }
        r.touch(&cold);

        let mut budgets = crate::budget::Budgets::new();
        budgets.set(Device::Gpu(0), 4 * GB + (300 << 20), 0);
        budgets.alloc(Device::Gpu(0), 4 * GB + (200 << 20));
        let need = MemCost::new(250 << 20, 0);

        let lru = plan_eviction_with(&Lru, &need, &budgets, &r, &[], &no_exclude())
            .expect("lru finds a plan");
        assert_eq!(lru.victims, vec![hot.clone()], "LRU evicts the 4GB hot model — the regret case");

        let ca = plan_eviction_with(&CostAware, &need, &budgets, &r, &[], &no_exclude())
            .expect("cost-aware finds a plan");
        assert_eq!(ca.victims, vec![cold.clone()], "CostAware evicts the cheap cold model");
    }
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
    fn npu_capable_model_prefers_the_npu() {
        let mut b = two_gpus();
        b.set(Device::Npu(0), 8 * GB, 0);
        // A model with both a GPU and an NPU path (npu > 0) is placed on the NPU,
        // even though the GPUs are emptier — NPU is the preferred class.
        let both = MemCost::new(6 * GB, 0).with_npu(2 * GB);
        assert_eq!(pick_device(&both, &b, &no_exclude()), Some(Device::Npu(0)));
        // A model without an NPU path (npu == 0) still goes to a GPU.
        assert_eq!(pick_device(&vram(6), &b, &no_exclude()), Some(Device::Gpu(0)));
        // If the NPU is full, an NPU-capable model falls back to a GPU.
        b.alloc(Device::Npu(0), 8 * GB);
        assert_eq!(pick_device(&both, &b, &no_exclude()), Some(Device::Gpu(0)));
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
