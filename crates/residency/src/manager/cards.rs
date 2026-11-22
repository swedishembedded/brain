// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for giving one training job several
// of a server's cards while the server keeps serving for its clients. If your
// team needs expertise in scheduling multi-GPU work beside inference then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The claim of an instance that runs on **any N cards with room** ([`ResidentModel::cards`]): a workload that runs
//! as ranks, one per card, each card charged the same footprint ([`ResidentModel::estimate`]).
//!
//! It differs from [`ResidencyManager::claim_multi`] in who chooses the devices: a multi-device model names its own
//! device set in its estimate, while a cards claim names a count and the manager picks the cards. The bookkeeping is
//! the multi-device one (a [`MultiEntry`] charged on every card, pinned while it runs), so release, failure and the
//! executor's lanes treat it as a multi-device group.
//!
//! Every claim is planned before anything is mutated ([`ResidencyManager::plan_cards`]):
//!
//! * a count no set of cards could ever hold (fewer cards whose budget holds one share than the count) is refused as
//!   [`ClaimError::TooLarge`], so the job fails at once instead of queueing forever;
//! * cards that hold a share now are taken first, the ones with the most room first;
//! * the rest come one card at a time from the card where evicting **idle** residents makes room with the fewest
//!   victims: single-device residents on that card, residents on several cards that span it (one victim frees all
//!   of its cards, which the next card counts), or both. A pinned resident, one a job runs on, is never a victim;
//! * when that is still short, nothing is evicted and the claim waits ([`ClaimError::NoCapacity`]).

use std::cmp::Reverse;
use std::collections::HashSet;

use capability::Invocation;

use super::{device_order, synth_cost_for, ClaimError, ClaimedMulti, MultiEntry, ResidencyManager};
use crate::multi::MultiDeviceCost;
use crate::place::plan_eviction_with;
use crate::{Device, InstanceKey, ResidentModel};

/// Where a cards claim goes: the cards, and the idle residents evicted to make room on them.
pub(super) struct CardsPlan {
    pub(super) devices: Vec<Device>,
    pub(super) victims: Vec<InstanceKey>,
    /// The idle residents on several cards evicted to make room.
    pub(super) spanning: Vec<InstanceKey>,
    /// What the claim charges, card by card (and on the host, for a process of its own).
    pub(super) cost: MultiDeviceCost,
}

impl ResidencyManager {
    /// Whether `model`'s instance `key` runs on several cards at once ([`ResidentModel::cards`]).
    pub fn spans_cards(&self, key: &InstanceKey, model: &str) -> bool {
        self.models.get(model).is_some_and(|m| m.cards(key) > 1)
    }

    /// Where the instance `key` of `m` would run on cards outside `exclude`, and what must give way for it, without
    /// changing anything. See the module doc for the order of preference and the refusals.
    pub(super) fn plan_cards(&self, m: &dyn ResidentModel, key: &InstanceKey, exclude: &HashSet<Device>) -> Result<CardsPlan, ClaimError> {
        let count = m.cards(key);
        let cost = self.hot_cost(m, key);
        let (class, kind) = if cost.npu > 0 { (self.budgets.npus(), "NPU") } else { (self.budgets.gpus(), "card") };
        let share = |d: Device| cost.resident_on(d);
        let need = class.first().map_or(cost.vram.max(cost.npu), |&d| share(d));
        if need == 0 {
            return Err(ClaimError::Activate(format!("{key} runs on {count} cards but declares no bytes on a card")));
        }
        let mib = need >> 20;
        let ever = class.iter().filter(|&&d| self.budgets.ceiling_on(d) >= need).count();
        if ever < count {
            return Err(ClaimError::TooLarge(format!(
                "{key} runs on {count} cards with {mib} MiB on each, and this server has {ever} {kind}(s) whose budget can hold that share even with nothing resident"
            )));
        }
        let usable = class.iter().filter(|&&d| self.budgets.usable_on(d) >= need).count();
        if usable < count {
            return Err(ClaimError::Starved(format!(
                "{key} runs on {count} cards with {mib} MiB on each, and only {usable} can hold that share with the memory this server can use right now ({}); another process holds the memory, and the request is retried when it returns",
                self.usable_summary()
            )));
        }
        // One process holds the run, however many cards it spans: its own host memory is charged once.
        let process_host = self.process_host_charge(&cost);
        self.refuse_host_short(key, &cost, None, 0)?;

        let mut free: Vec<Device> = class.iter().copied().filter(|&d| !exclude.contains(&d) && self.budgets.fits_on(d, share(d))).collect();
        free.sort_by_key(|&d| (Reverse(self.budgets.free_on(d)), device_order(d)));
        let mut devices: Vec<Device> = free.into_iter().take(count).collect();
        let mut victims: Vec<InstanceKey> = Vec::new();
        let mut spanning: Vec<InstanceKey> = Vec::new();
        // The rest, one card at a time, each from the card the fewest new victims free; a resident on several cards
        // that gives way frees every one of them, which the next round counts.
        while devices.len() < count {
            let best = class
                .iter()
                .copied()
                .filter(|&d| !exclude.contains(&d) && !devices.contains(&d) && self.budgets.usable_on(d) >= share(d))
                .filter_map(|d| self.free_card(d, share(d), &spanning).map(|freeing| (d, freeing)))
                .min_by_key(|(d, freeing)| (freeing.victims.len() + freeing.spanning.len(), freeing.spanning.len(), device_order(*d)));
            let Some((d, freeing)) = best else {
                return Err(ClaimError::NoCapacity(format!(
                    "{key} runs on {count} cards with {mib} MiB on each; {} have room now or can be freed by evicting idle instances",
                    devices.len()
                )));
            };
            devices.push(d);
            victims.extend(freeing.victims);
            spanning.extend(freeing.spanning);
        }
        // In card order, whichever had the most room: rank r of the run goes to the r-th card, so the same set of
        // cards always gets the same ranks and a repeat reduces in the same order (the same bytes).
        devices.sort_by_key(|&d| device_order(d));
        let per_card: Vec<(Device, u64)> = devices.iter().map(|&d| (d, share(d))).collect();
        Ok(CardsPlan { devices, victims, spanning, cost: MultiDeviceCost::new(per_card, process_host) })
    }

    /// Whether [`Self::claim_cards`] would place `key` now, or fail it for good (a refusal is let through, so the
    /// claim answers the job instead of leaving it queued); `false` only while it must wait for room.
    pub fn placeable_cards(&self, key: &InstanceKey, model: &str, exclude: &HashSet<Device>) -> bool {
        let Some(m) = self.models.get(model) else { return false };
        !matches!(self.plan_cards(&**m, key, exclude), Err(ClaimError::NoCapacity(_)))
    }

    /// Place and pin the instance of `(model, action, inv)` on [`ResidentModel::cards`] cards outside `exclude`,
    /// evicting the idle residents its plan names, and charge every card. The build is deferred to the caller
    /// ([`ClaimedMulti::Cards`]), which reports [`Self::adopt_multi`] or [`Self::build_failed_multi`]; the run ends
    /// with [`Self::release_multi`] (which drops a transient instance and returns every card's charge) or
    /// [`Self::unwind_failed_multi`].
    pub fn claim_cards(&mut self, model: &str, action: &str, inv: &Invocation, replica: usize, exclude: &HashSet<Device>) -> Result<(ClaimedMulti, Vec<Device>, InstanceKey), ClaimError> {
        let m = self.models.get(model).ok_or_else(|| ClaimError::Activate(format!("no model '{model}'")))?.clone();
        let key = m.instance_key(action, inv).replica(replica);
        if self.residents.get(&key).is_some() || self.multi_residents.contains_key(&key) {
            return Err(ClaimError::NoCapacity(format!("{key}: a claim for this key is already in flight")));
        }
        let plan = self.plan_cards(&*m, &key, exclude)?;
        // The whole claim is satisfiable: only now is anything destroyed or charged.
        for victim in &plan.victims {
            self.evict_entry(victim);
        }
        for victim in &plan.spanning {
            self.evict_multi(victim);
        }
        for &d in &plan.devices {
            self.budgets.alloc(d, plan.cost.on(d));
        }
        self.charge_multi_host(&plan.cost, &plan.devices);
        self.multi_claims += 1;
        self.multi_residents.insert(key.clone(), MultiEntry { cost: plan.cost, devices: plan.devices.clone(), pinned: true, last_claimed: self.multi_claims });
        self.event(format!("promote {key} -> {:?} (building, {} cards)", plan.devices, plan.devices.len()));
        Ok((ClaimedMulti::Cards(m), plan.devices, key))
    }
}

/// What freeing one card for a cards claim takes: idle single-device residents on it, and idle residents on several
/// cards (each also freeing its other cards).
struct Freeing {
    victims: Vec<InstanceKey>,
    spanning: Vec<InstanceKey>,
}

impl ResidencyManager {
    /// How card `d` holds `need` bytes once `spanning` (idle residents on several cards already chosen to give way) are
    /// gone: with nothing more, with idle single-device residents on `d` alone, with idle residents on several cards
    /// that span `d` (the smallest share on `d` first), or with both; the way with the fewest victims, and the fewest
    /// residents on several cards among those. `None` when even every idle resident on `d` is not enough. Reads only.
    fn free_card(&self, d: Device, need: u64, spanning: &[InstanceKey]) -> Option<Freeing> {
        let freed = |keys: &[InstanceKey]| keys.iter().filter_map(|k| self.multi_residents.get(k)).map(|e| e.cost.on(d)).sum::<u64>();
        let free = self.budgets.free_on(d) + freed(spanning);
        if free >= need {
            return Some(Freeing { victims: Vec::new(), spanning: Vec::new() });
        }
        // Idle single-device residents on `d` alone, for what is still short after `extra` more bytes are freed.
        let every: HashSet<Device> = self.budgets.devices().collect();
        let mut only = every;
        only.remove(&d);
        let singles = |extra: u64| -> Option<Vec<InstanceKey>> {
            let still = need.saturating_sub(free - self.budgets.free_on(d) + extra);
            if self.budgets.fits_on(d, still) {
                return Some(Vec::new());
            }
            plan_eviction_with(&*self.eviction, &synth_cost_for(d, still), &self.budgets, &self.residents, &[], &only).map(|plan| plan.victims)
        };
        let mut candidates: Vec<(&InstanceKey, u64)> =
            self.multi_residents.iter().filter(|(k, e)| !e.pinned && e.devices.contains(&d) && !spanning.contains(k)).map(|(k, e)| (k, e.cost.on(d))).collect();
        candidates.sort_by(|(a, x), (b, y)| x.cmp(y).then_with(|| a.to_string().cmp(&b.to_string())));
        let mut options: Vec<Freeing> = Vec::new();
        if let Some(victims) = singles(0) {
            options.push(Freeing { victims, spanning: Vec::new() });
        }
        let (mut taken, mut extra) = (Vec::new(), 0u64);
        for (k, bytes) in candidates {
            taken.push(k.clone());
            extra += bytes;
            if let Some(victims) = singles(extra) {
                options.push(Freeing { victims, spanning: taken.clone() });
            }
        }
        options.into_iter().min_by_key(|f| (f.victims.len() + f.spanning.len(), f.spanning.len()))
    }
}
