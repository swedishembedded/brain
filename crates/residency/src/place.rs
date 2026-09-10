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
/// (`brain perf residency`) measured strict LRU at an eviction-regret rate of
/// nearly two thirds under a shifting Zipf load: most evictions were of models
/// wanted again almost immediately. LRU has no notion of *reload cost* (a 4 GB
/// model costs an order of magnitude more than a 200 MB one to bring back) or
/// *popularity* (the head of a Zipf
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
        // Measured on `perf residency` (24 models, memory overcommitted four
        // times over, shifting Zipf): this GDSF shape beats LRU on hit rate
        // and - by construction, pinned in policy_tests - spends evictions on
        // cheap models instead of expensive ones. Event-counted regret is metric-
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

/// Host bytes the CPU tier must hold to run a model of `cost` **there**.
///
/// A weight-holding model reports its footprint as `vram` (or `npu`); running
/// it on the host means holding those same weights in RAM, alongside whatever
/// host bytes it declared outright. They are the SAME bytes, not two copies -
/// a model does not stage its own weights beside itself - so this is a `max`,
/// not a sum, which is the convention [`pick_device`]'s CPU branch has always
/// used.
///
/// This is the one definition of "what does the host tier cost", shared by
/// [`pick_device`], [`spill_to_host`], [`could_ever_fit`] and
/// [`plan_eviction_with`], so the four cannot disagree about whether a model
/// fits in RAM.
pub fn host_need(cost: &MemCost) -> u64 {
    cost.ram.max(cost.vram).max(cost.npu)
}

/// The host tier as a **last resort**: `Some(Device::Cpu)` when RAM can hold
/// `cost` right now.
///
/// Deliberately separate from [`pick_device`] rather than folded into it,
/// because the two answer different questions. `pick_device` answers "where
/// does this go if it goes somewhere fast", and its `None` is what tells
/// `ResidencyManager::claim` to EVICT rather than quietly park a GPU model in
/// RAM at a hundredth of the speed. This function answers the question after
/// that one: eviction was tried and could not free enough (or there was
/// nothing to evict), so the choice is now between the host tier and failing
/// the request outright. The operator's requirement is explicit that it must
/// be the host tier - "it may still run slower if all of a sudden vram is not
/// available but it should never fail".
pub fn spill_to_host(cost: &MemCost, budgets: &Budgets, exclude: &HashSet<Device>) -> Option<Device> {
    let need = host_need(cost);
    let ok = need > 0
        && !exclude.contains(&Device::Cpu)
        && budgets.get(Device::Cpu).is_some()
        && budgets.fits_on(Device::Cpu, need);
    ok.then_some(Device::Cpu)
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
            if budgets.get(d).is_some() && budgets.fits_on(d, need) {
                let free = budgets.free_on(d);
                if best.is_none_or(|(_, f)| free > f) {
                    best = Some((d, free));
                }
            }
        }
        if let Some((d, _)) = best {
            return Some(d);
        }
    }
    // CPU fallback: an explicit RAM-resident model, OR a GPU/NPU model on a host with
    // no accelerator of that class (a GPU-less/NPU-less box) — a weight-holding model's
    // host RAM footprint is the same bytes it would take on an accelerator, so spill it
    // to the CPU. (On a host that HAS the accelerator but it's full, we return None so
    // the caller's eviction path frees the accelerator instead of spilling to slow CPU.)
    let cpu_need = cost
        .ram
        .max(if budgets.gpus().is_empty() { cost.vram } else { 0 })
        .max(if budgets.npus().is_empty() { cost.npu } else { 0 });
    if cpu_need > 0 && !exclude.contains(&Device::Cpu) && budgets.get(Device::Cpu).is_some() && budgets.fits_on(Device::Cpu, cpu_need) {
        return Some(Device::Cpu);
    }
    // Zero-cost (stateless) instance: fits anywhere by definition. Prefer the
    // CPU so it never ties up an accelerator lane; fall back to any free
    // device. (Without this branch a stateless model — demo, imageops — was
    // UNPLACEABLE: every class loop skips on need == 0 and the CPU branch
    // requires ram > 0, so its jobs sat in the queue forever, silently.)
    if cost.npu == 0 && cost.vram == 0 && cost.ram == 0 {
        if !exclude.contains(&Device::Cpu) && budgets.get(Device::Cpu).is_some() {
            return Some(Device::Cpu);
        }
        for d in budgets.devices() {
            if !exclude.contains(&d) {
                return Some(d);
            }
        }
    }
    None
}

/// True if SOME device's usable budget could ever hold `cost`, evaluated
/// against each device's usable ceiling alone (no current occupancy). This is
/// the permanent/"never fits" check — `ResidencyManager::claim` calls it
/// BEFORE planning any eviction to tell `ClaimError::TooLarge` (this returns
/// `false`: no eviction, however aggressive, could ever succeed) apart from
/// `ClaimError::NoCapacity` (some device COULD hold it, just not with what is
/// evictable right now). Mirrors [`pick_device`]'s class-preference and CPU-
/// fallback structure exactly, substituting `usable_on` for `fits_on` so it
/// answers "ever", not "right now".
pub fn could_ever_fit(cost: &MemCost, budgets: &Budgets) -> bool {
    if cost.vram == 0 && cost.ram == 0 && cost.npu == 0 {
        return true; // stateless: placeable anywhere a budget exists at all.
    }
    for (devices, need) in [(budgets.npus(), cost.npu), (budgets.gpus(), cost.vram)] {
        if need == 0 {
            continue;
        }
        if devices.iter().any(|&d| budgets.usable_on(d) >= need) {
            return true;
        }
    }
    // The host tier, unconditionally. This asks "could this model EVER run
    // here", not "should it run here now", and brain can always run a model on
    // the CPU backend - slowly. Gating this on `gpus().is_empty()` (which it
    // used to do, mirroring `pick_device`'s deliberate refusal to spill while
    // a card exists to be evicted for) conflated the two questions and made a
    // model that is merely bigger than any single card `ClaimError::TooLarge`
    // - which the executor treats as PERMANENT and uses to fail every queued
    // job for that model, forever, on a machine with 45 GiB of idle RAM.
    // `pick_device` still refuses (eviction stays preferred); what changes is
    // that "no card will ever hold it" is no longer reported as "it can never
    // run".
    host_need(cost) > 0 && budgets.usable_on(Device::Cpu) >= host_need(cost)
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
    // Same class preference as `pick_device`: NPU (if the model has an NPU path),
    // then GPU, then - last - the HOST tier. Victim bytes are counted with the
    // device-appropriate cost field (`entry.cost.on(d)`), so NPU eviction frees NPU
    // bytes, GPU eviction frees VRAM and host eviction frees RAM. If any plan
    // exists in the preferred class, it wins, so adding the host class cannot
    // divert a model away from a card it could have had: it is only reached when
    // no accelerator plan exists at all.
    //
    // The host class is not cosmetic. Without it `Device::Cpu` was not an
    // eviction victim class ANYWHERE, so (a) a CPU-resident model that did not
    // fit right now got `NoCapacity` forever - on a CPU-only box the residency
    // system had no eviction at all, it filled up once and livelocked - and (b)
    // every warm-demotion copy `evict_entry` parks on `Device::Cpu` was
    // unreclaimable for the daemon's whole life, so the CPU `used` counter only
    // ever went up and the demoted `Instance`s leaked.
    let host: Vec<Device> = budgets.get(Device::Cpu).map(|_| vec![Device::Cpu]).unwrap_or_default();
    for (devices, need) in [(budgets.npus(), cost.npu), (budgets.gpus(), cost.vram), (host, host_need(cost))] {
        if need == 0 {
            continue;
        }
        let mut best: Option<EvictionPlan> = None;
        for d in devices {
            if exclude.contains(&d) {
                continue;
            }
            if budgets.get(d).is_none() {
                continue;
            }
            if budgets.usable_on(d) < need {
                continue; // can never fit here, even empty (device- or pool-limited)
            }
            if budgets.fits_on(d, need) {
                // Already fits with no eviction.
                return Some(EvictionPlan { device: d, victims: Vec::new(), freed: 0 });
            }
            // Evict lowest-score-first until free_on() would cover `need`.
            let mut deficit = need - budgets.free_on(d);
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
                let vbytes = entry.cost.resident_on(d);
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

    /// The scenario behind the benchmark's regret finding: a large HOT model and
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
    fn vram_model_spills_to_cpu_on_a_gpu_less_host() {
        // A weight-holding model reports its footprint as `vram` (est_vram sets
        // MemCost::new(bytes, 0)). On a CPU-only host (no GPU/NPU) it MUST fall back to
        // the CPU using that footprint — otherwise every request to it is unplaceable
        // and 429s forever (the bug the Claude Code e2e caught).
        let mut cpu_only = Budgets::new();
        cpu_only.set(Device::Cpu, 32 * GB, 2 * GB);
        assert_eq!(pick_device(&vram(3), &cpu_only, &no_exclude()), Some(Device::Cpu));
        // Too big for the CPU budget -> None (the caller queues / evicts).
        assert_eq!(pick_device(&vram(64), &cpu_only, &no_exclude()), None);
        // But on a host WITH a GPU that happens to be full, a vram model does NOT spill
        // to slow CPU — it returns None so the caller's eviction frees the GPU.
        // Filled to their usable ceiling (24 GB card, 2 GB reserve).
        let mut b = two_gpus();
        b.alloc(Device::Gpu(0), 22 * GB);
        b.alloc(Device::Gpu(1), 22 * GB);
        assert_eq!(pick_device(&vram(6), &b, &no_exclude()), None);
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

    /// A model bigger than any card gets no GPU plan, however aggressive the
    /// eviction - and lands on the host tier instead of nowhere. With no host
    /// tier budgeted at all there is genuinely nowhere, and it says so.
    #[test]
    fn too_big_for_any_card_falls_to_the_host_tier_and_only_then_gives_up() {
        let b = two_gpus(); // usable = 22 GB each, plus a 128 GB host tier
        let r = Residents::new();
        let plan = plan_eviction(&vram(23), &b, &r, &[], &no_exclude()).expect("the host tier can hold it");
        assert_eq!(plan.device, Device::Cpu, "no card can ever hold 23 GB here");
        assert!(plan.victims.is_empty(), "the host tier had room; nothing needed evicting");

        let mut cards_only = Budgets::new();
        cards_only.set(Device::Gpu(0), 24 * GB, 2 * GB).set(Device::Gpu(1), 24 * GB, 2 * GB);
        assert!(plan_eviction(&vram(23), &cards_only, &r, &[], &no_exclude()).is_none(), "no host tier: nowhere at all");
    }

    /// The host tier is a real eviction victim class, not just a spill target.
    /// Without this a warm-demotion copy parked on `Device::Cpu` could never be
    /// reclaimed (the CPU `used` counter only ever went up) and a CPU-resident
    /// model that did not fit right now got `NoCapacity` forever - on a
    /// CPU-only box the residency system had no eviction at all.
    #[test]
    fn the_host_tier_evicts_its_own_residents() {
        let mut cpu_only = Budgets::new();
        cpu_only.set(Device::Cpu, 32 * GB, 0);
        cpu_only.alloc(Device::Cpu, 30 * GB);
        let mut r = Residents::new();
        r.insert(ik("warm_leftover"), MemCost::new(0, 30 * GB), Device::Cpu);

        let plan = plan_eviction(&MemCost::new(0, 20 * GB), &cpu_only, &r, &[], &no_exclude()).expect("a host-tier plan");
        assert_eq!(plan.device, Device::Cpu);
        assert_eq!(plan.victims, vec![ik("warm_leftover")], "the host tier's own resident is the victim");
    }

    /// ...and a GPU plan still WINS over the host tier whenever one exists, so
    /// adding the host class cannot quietly divert a model off a card it could
    /// have had by evicting something.
    #[test]
    fn an_evictable_card_beats_the_host_tier() {
        let mut b = two_gpus();
        b.alloc(Device::Gpu(0), 22 * GB);
        b.alloc(Device::Gpu(1), 22 * GB);
        let mut r = Residents::new();
        r.insert(ik("old"), vram(22), Device::Gpu(1));
        let plan = plan_eviction(&vram(20), &b, &r, &[], &no_exclude()).expect("a plan");
        assert_eq!(plan.device, Device::Gpu(1), "evicting a card beats running on the CPU");
        assert_eq!(plan.victims, vec![ik("old")]);
    }

    #[test]
    fn could_ever_fit_distinguishes_permanent_from_transient() {
        let mut b = two_gpus(); // usable = 22 GB each, 24 GB total each.
        b.set(Device::Cpu, 24 * GB, 0); // a host tier smaller than the pair of cards
        // Fits an EMPTY card -> could ever fit, even though nothing is free
        // right now in this test (budgets start empty here, so it's also
        // immediately placeable -- the "ever" question is what matters).
        assert!(could_ever_fit(&vram(22), &b));
        // Bigger than the largest card's usable budget AND bigger than the
        // host tier: this one genuinely can never run here.
        assert!(!could_ever_fit(&vram(25), &b));
        // Stateless (all-zero cost) is always "ever fits".
        assert!(could_ever_fit(&MemCost::new(0, 0), &b));
    }

    /// `could_ever_fit` answers "could this model EVER run here", which decides
    /// `TooLarge` (permanent, fails every queued job for the model) against
    /// `NoCapacity` (transient, wait and retry). It used to mirror
    /// `pick_device`'s deliberate refusal to spill to the CPU while a card
    /// exists - so a model merely bigger than any single card was reported as
    /// unable to ever run, and its jobs were failed permanently, on a box with
    /// tens of GiB of idle RAM and a working CPU backend. The two questions
    /// are different: `pick_device` still refuses (eviction stays preferred).
    #[test]
    fn could_ever_fit_offers_the_host_tier_when_no_card_could_ever_hold_it() {
        let mut b = two_gpus();
        b.set(Device::Cpu, 512 * GB, 0);
        assert!(could_ever_fit(&vram(23), &b), "23 GB fits no card, but the CPU backend can run it");
        // `pick_device` is unchanged: it still says "no", which is what makes
        // the caller evict a card rather than park the model in RAM.
        let mut full = two_gpus();
        full.alloc(Device::Gpu(0), 22 * GB);
        full.alloc(Device::Gpu(1), 22 * GB);
        assert_eq!(pick_device(&vram(6), &full, &no_exclude()), None);
    }

    #[test]
    fn could_ever_fit_allows_cpu_spill_on_a_gpu_less_host() {
        let mut cpu_only = Budgets::new();
        cpu_only.set(Device::Cpu, 32 * GB, 2 * GB);
        assert!(could_ever_fit(&vram(20), &cpu_only), "a weight-holding model must be able to spill to CPU with no GPU present");
        assert!(!could_ever_fit(&vram(40), &cpu_only));
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
