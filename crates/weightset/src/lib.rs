// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Within-instance weight residency: a fixed-size window of device slots over
//! a model's weight **groups** (a transformer block, or an equivalent unit
//! that is always used together — coarser than a tensor on purpose, since
//! per-tensor tiering would multiply bookkeeping for no hit-rate gain).
//!
//! `residency::EvictionPolicy` (a different trait, in a different crate)
//! answers "which of these should I throw away" from a *past* access
//! pattern — right for an unpredictable request stream across independent
//! model instances. This crate answers a different question: a denoise loop
//! or a decode loop visits its groups in an order that is **known exactly in
//! advance** (the schedule is built before the first step ever runs), so the
//! right policy plans over the known future instead of scoring the past.
//! [`ResidencyPlan`] is that second, different trait for that second,
//! different problem.
//!
//! [`WeightSet`] owns no device memory itself — it is pure host-side
//! bookkeeping (which [`GroupId`] occupies which [`SlotId`]) so it is fully
//! unit-testable without a GPU. The caller (a model's device engine) owns a
//! fixed pool of `budget` device buffers sized for one group's shape, and
//! asks `advance(cursor)` at each step of its own schedule which slot to
//! bind and upload into on a miss.

use std::collections::HashSet;

/// One weight group (e.g. one transformer block) in a model's weight set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GroupId(pub u32);

/// One device slot in the fixed pool — stable across `advance` calls until
/// the group it holds is evicted.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SlotId(pub u32);

/// The known future: one entry per (pass, position-in-pass), flattened. A
/// denoise loop is `steps` passes over `n_blocks` groups in a fixed order; an
/// autoregressive decode loop has the identical shape (`tokens` passes over
/// `n_layers`).
#[derive(Clone, Debug)]
pub struct Schedule {
    pub order: Vec<GroupId>,
}

impl Schedule {
    /// `n_groups` groups visited in order, `passes` times.
    pub fn cyclic(n_groups: u32, passes: u32) -> Schedule {
        let mut order = Vec::with_capacity(n_groups as usize * passes as usize);
        for _ in 0..passes {
            order.extend((0..n_groups).map(GroupId));
        }
        Schedule { order }
    }
}

/// Decides which groups stay resident and which to evict, given the
/// schedule and a fixed slot budget.
pub trait ResidencyPlan {
    /// Groups to have resident *before* the first `advance` call, up to
    /// `budget` of them. Fixes initial slot assignment 0..len.
    fn pin(&self, n_groups: u32, sched: &Schedule, budget: u32) -> Vec<GroupId>;

    /// Which of `resident` (groups currently holding an *unpinned* slot) to
    /// evict to make room for the miss at `sched.order[cursor]`. Never
    /// called with an empty `resident` — `WeightSet::advance` only reaches
    /// this when the window is full and at least one slot is unpinned.
    fn victim(&self, resident: &[GroupId], sched: &Schedule, cursor: usize) -> GroupId;

    /// True for a plan whose correctness *requires* every group to fit in
    /// the window at once (e.g. [`AllResident`]) — checked at
    /// [`WeightSet::build`] time so an undersized budget is a clean `Err`,
    /// never a panic on the first miss.
    fn requires_full_residency(&self) -> bool {
        false
    }
}

/// Optimal for a fully-known schedule: pin the longest prefix the budget
/// allows (minus a small rotating reserve so the unpinned tail has
/// somewhere to load into), and evict by furthest-next-use (Bélády) — exact,
/// not a heuristic, because the future is the schedule.
pub struct CyclicScan {
    /// Rotating slots reserved for the unpinned tail (prefetch depth). At
    /// least 1 whenever the window is narrower than the model, or nothing
    /// unpinned could ever load. Ignored when `budget >= n_groups` — then
    /// everything is pinned and the reserve would only waste slots.
    pub lookahead: u32,
}

impl ResidencyPlan for CyclicScan {
    fn pin(&self, n_groups: u32, _sched: &Schedule, budget: u32) -> Vec<GroupId> {
        let reserve = if budget >= n_groups { 0 } else { self.lookahead.max(1).min(budget) };
        let pin_n = budget.saturating_sub(reserve).min(n_groups);
        (0..pin_n).map(GroupId).collect()
    }
    fn victim(&self, resident: &[GroupId], sched: &Schedule, cursor: usize) -> GroupId {
        resident
            .iter()
            .copied()
            .max_by_key(|&g| next_use(sched, cursor, g))
            .expect("WeightSet only calls victim() with a non-empty resident set")
    }
}

/// Index of the next occurrence of `g` at or after `cursor`, or `usize::MAX`
/// if it is never touched again (the correct group to evict first).
fn next_use(sched: &Schedule, cursor: usize, g: GroupId) -> usize {
    sched.order[cursor..].iter().position(|&x| x == g).map(|i| cursor + i).unwrap_or(usize::MAX)
}

/// The control arm: every group pinned, nothing ever evicted. Only valid
/// when `budget >= n_groups` — [`WeightSet::build`] rejects a smaller
/// budget via [`ResidencyPlan::requires_full_residency`] rather than this
/// panicking on the first miss.
pub struct AllResident;

impl ResidencyPlan for AllResident {
    fn pin(&self, n_groups: u32, _sched: &Schedule, budget: u32) -> Vec<GroupId> {
        (0..n_groups.min(budget)).map(GroupId).collect()
    }
    fn victim(&self, _resident: &[GroupId], _sched: &Schedule, _cursor: usize) -> GroupId {
        unreachable!("AllResident guarantees no miss ever occurs when budget >= n_groups")
    }
    fn requires_full_residency(&self) -> bool {
        true
    }
}

/// The naive baseline: nothing pinned ahead of time, evict by recency of
/// use — exactly what a cache with no knowledge of the future does. Kept
/// deliberately (not deleted) so a benchmark can show it losing on
/// identical seeds instead of a doc merely asserting it.
pub struct Lru;

impl ResidencyPlan for Lru {
    fn pin(&self, _n_groups: u32, _sched: &Schedule, _budget: u32) -> Vec<GroupId> {
        Vec::new() // no foreknowledge assumed, so nothing is pre-pinned
    }
    fn victim(&self, resident: &[GroupId], sched: &Schedule, cursor: usize) -> GroupId {
        resident
            .iter()
            .copied()
            .min_by_key(|&g| last_use(sched, cursor, g))
            .expect("WeightSet only calls victim() with a non-empty resident set")
    }
}

/// Index of the most recent occurrence of `g` strictly before `cursor`, or
/// `-1` if never seen (evict an unseen-so-far group first).
fn last_use(sched: &Schedule, cursor: usize, g: GroupId) -> isize {
    sched.order[..cursor].iter().rposition(|&x| x == g).map(|i| i as isize).unwrap_or(-1)
}

/// Why [`WeightSet::build`] refused to build a window.
#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    /// A window with no slots can hold nothing — always a caller bug, never
    /// a legitimate "tiny budget" request.
    ZeroSlots,
    /// The plan pinned more groups than the budget allows (a well-behaved
    /// built-in plan never does this; a custom plan might).
    PinExceedsBudget { pinned: u32, budget: u32 },
    /// The plan requires every group resident at once ([`AllResident`]) but
    /// the budget is smaller than the model.
    WouldNotFit { n_groups: u32, budget: u32 },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::ZeroSlots => write!(f, "weightset: zero slots"),
            BuildError::PinExceedsBudget { pinned, budget } => {
                write!(f, "weightset: plan pinned {pinned} groups but the budget is only {budget} slots")
            }
            BuildError::WouldNotFit { n_groups, budget } => {
                write!(f, "weightset: {n_groups} groups do not fit in a {budget}-slot window and this plan requires full residency")
            }
        }
    }
}

/// A fixed-size window of device slots over a model's weight groups,
/// scheduled by a [`ResidencyPlan`]. `budget` never changes after
/// [`build`](Self::build) — no suballocator, no fragmentation, no
/// steady-state OOM path; the only allocation-failure point is `build`
/// itself, where the caller has already secured room for `budget` slots.
pub struct WeightSet {
    slots: Vec<Option<GroupId>>, // len == budget; None = free
    pinned: HashSet<GroupId>,
    plan: Box<dyn ResidencyPlan>,
    schedule: Schedule,
    reloads: u64,
}

impl WeightSet {
    pub fn build(n_groups: u32, budget: u32, schedule: Schedule, plan: Box<dyn ResidencyPlan>) -> Result<WeightSet, BuildError> {
        if budget == 0 {
            return Err(BuildError::ZeroSlots);
        }
        if plan.requires_full_residency() && budget < n_groups {
            return Err(BuildError::WouldNotFit { n_groups, budget });
        }
        let pin = plan.pin(n_groups, &schedule, budget);
        if pin.len() as u32 > budget {
            return Err(BuildError::PinExceedsBudget { pinned: pin.len() as u32, budget });
        }
        let mut slots: Vec<Option<GroupId>> = vec![None; budget as usize];
        for (i, g) in pin.iter().enumerate() {
            slots[i] = Some(*g);
        }
        Ok(WeightSet { slots, pinned: pin.into_iter().collect(), plan, schedule, reloads: 0 })
    }

    pub fn n_slots(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Count of `advance` calls that were a miss (required a load). The
    /// churn measure the whole design exists to bound: an optimal plan
    /// reloads exactly the unpinned tail once per pass, never more.
    pub fn reloads(&self) -> u64 {
        self.reloads
    }

    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    fn slot_of(&self, g: GroupId) -> Option<usize> {
        self.slots.iter().position(|s| *s == Some(g))
    }

    /// Ensure the group at schedule position `cursor` occupies a slot,
    /// evicting per the plan if the window is full. Returns the slot it now
    /// occupies (stable until evicted) and whether this call was a miss —
    /// a hit means the caller's device buffer for that slot is untouched
    /// and does not need re-uploading.
    pub fn advance(&mut self, cursor: usize) -> (SlotId, bool) {
        let g = self.schedule.order[cursor];
        if let Some(i) = self.slot_of(g) {
            return (SlotId(i as u32), false);
        }
        let free = self.slots.iter().position(|s| s.is_none());
        let idx = match free {
            Some(i) => i,
            None => {
                let resident: Vec<GroupId> = self.slots.iter().filter_map(|s| *s).filter(|g| !self.pinned.contains(g)).collect();
                let victim = self.plan.victim(&resident, &self.schedule, cursor);
                self.slot_of(victim).expect("victim must currently be resident")
            }
        };
        self.slots[idx] = Some(g);
        self.reloads += 1;
        (SlotId(idx as u32), true)
    }
}

/// `reloads / (required_per_pass * passes)` — `1.0` is optimal (every
/// promotion was necessary), `>1.0` means avoidable reloads happened.
/// `required_per_pass` is the caller's chosen baseline (typically
/// `n_groups - target_pin` for whatever pin count an optimal plan would
/// achieve at this budget) — a fixed yardstick every plan's `reloads()` is
/// measured against, not each plan's own (possibly much smaller) pin count.
pub fn churn_overhead(reloads: u64, required_per_pass: u64, passes: u32) -> f64 {
    reloads as f64 / (required_per_pass * passes as u64) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_slots_is_a_clean_error() {
        let sched = Schedule::cyclic(4, 1);
        match WeightSet::build(4, 0, sched, Box::new(Lru)) {
            Err(e) => assert_eq!(e, BuildError::ZeroSlots),
            Ok(_) => panic!("zero slots must be a clean Err, not a built WeightSet"),
        }
    }

    #[test]
    fn all_resident_with_an_undersized_budget_is_a_clean_error_not_a_panic() {
        let sched = Schedule::cyclic(8, 2);
        match WeightSet::build(8, 5, sched, Box::new(AllResident)) {
            Err(e) => assert_eq!(e, BuildError::WouldNotFit { n_groups: 8, budget: 5 }),
            Ok(_) => panic!("an undersized AllResident budget must be a clean Err, not a built WeightSet"),
        }
    }

    /// `W >= N`: everything is pinned at build time and `advance` never
    /// misses, on the very first pass included — the exact behaviour of
    /// today's fully-resident engines (upload everything once, forward
    /// passes touch host memory zero times).
    #[test]
    fn window_at_least_as_wide_as_the_model_never_reloads() {
        let sched = Schedule::cyclic(8, 3);
        let mut ws = WeightSet::build(8, 8, sched, Box::new(CyclicScan { lookahead: 2 })).unwrap();
        for cursor in 0..ws.schedule().order.len() {
            let (_, miss) = ws.advance(cursor);
            assert!(!miss, "cursor {cursor}: W>=N must never miss, not even on the first pass");
        }
        assert_eq!(ws.reloads(), 0);

        // A wider budget than the model needs behaves identically.
        let sched2 = Schedule::cyclic(8, 3);
        let mut ws2 = WeightSet::build(8, 20, sched2, Box::new(CyclicScan { lookahead: 2 })).unwrap();
        for cursor in 0..ws2.schedule().order.len() {
            let (_, miss) = ws2.advance(cursor);
            assert!(!miss);
        }
        assert_eq!(ws2.reloads(), 0);
    }

    /// `CyclicScan` at a real window (10 slots, 1 reserved for rotation —
    /// so 9 groups pinned, 23 rotate through the last slot) reloads exactly
    /// the unpinned tail once per pass: churn_overhead == 1.0, exactly, not
    /// approximately. A regression in the eviction policy changes this
    /// number; the test names it instead of a doc merely claiming it.
    #[test]
    fn cyclic_scan_reloads_exactly_the_unpinned_tail_once_per_pass() {
        let n_groups = 32u32;
        let budget = 10u32;
        let passes = 4u32;
        let sched = Schedule::cyclic(n_groups, passes);
        let total = sched.order.len();
        let mut ws = WeightSet::build(n_groups, budget, sched, Box::new(CyclicScan { lookahead: 1 })).unwrap();
        for cursor in 0..total {
            ws.advance(cursor);
        }
        let pinned = budget - 1; // lookahead=1 -> 9 pinned, 1 rotating slot
        let required_per_pass = (n_groups - pinned) as u64; // 23
        assert_eq!(ws.reloads(), required_per_pass * passes as u64, "every unpinned-tail group loads exactly once per pass, no more");
        assert_eq!(churn_overhead(ws.reloads(), required_per_pass, passes), 1.0);
    }

    /// The same schedule and budget, but `Lru`: pinning nothing means the
    /// would-be-pinned prefix gets reloaded every pass too, not just the
    /// tail — a full miss on every touch, every pass, exactly `n_groups`
    /// reloads/pass (the "100% miss on a scan longer than the cache"
    /// pathology), strictly worse than `CyclicScan`'s exact 1.0 above.
    #[test]
    fn lru_reloads_the_whole_model_every_pass_including_the_prefix_that_should_stay_pinned() {
        let n_groups = 32u32;
        let budget = 10u32;
        let passes = 4u32;
        let sched = Schedule::cyclic(n_groups, passes);
        let total = sched.order.len();
        let mut ws = WeightSet::build(n_groups, budget, sched, Box::new(Lru)).unwrap();
        for cursor in 0..total {
            ws.advance(cursor);
        }
        // No group is ever touched twice within a `budget`-sized window (the
        // scan is strictly monotonic and n_groups > budget), so LRU misses
        // on literally every touch, every pass: n_groups reloads/pass.
        assert_eq!(ws.reloads(), n_groups as u64 * passes as u64);

        let cyclic_required_per_pass = (n_groups - (budget - 1)) as u64; // CyclicScan's baseline, 23
        let ratio = churn_overhead(ws.reloads(), cyclic_required_per_pass, passes);
        assert_eq!(ratio, (n_groups as f64 * passes as f64) / (cyclic_required_per_pass as f64 * passes as f64));
        assert!(ratio > 1.0, "Lru must be strictly worse than CyclicScan's optimal 1.0");
    }

    /// A slot is stable (the caller's device buffer for it is untouched)
    /// across repeated `advance` calls for a group that's already resident.
    #[test]
    fn advance_returns_a_stable_slot_until_the_group_is_evicted() {
        let sched = Schedule::cyclic(3, 1);
        let mut ws = WeightSet::build(3, 3, sched, Box::new(AllResident)).unwrap();
        let (s0, miss0) = ws.advance(0);
        assert!(!miss0);
        let (s0_again, miss0_again) = ws.advance(0);
        assert_eq!(s0, s0_again);
        assert!(!miss0_again);
    }

    /// Bélády picks the resident-unpinned group with the furthest next use
    /// -- verified directly against a hand-built schedule where the correct
    /// choice is unambiguous.
    #[test]
    fn belady_victim_evicts_the_group_used_furthest_in_the_future() {
        // Order: 0,1,2,3,1,0 -- at cursor=4 (about to touch group 1, already
        // resident) that's moot; check at cursor=3 (about to touch group 3)
        // with {0,1,2} resident and unpinned: group 2 is never touched again
        // (index MAX), group 1 is next touched at index 4, group 0 at index
        // 5 -- so group 2 must be the victim.
        let sched = Schedule { order: vec![GroupId(0), GroupId(1), GroupId(2), GroupId(3), GroupId(1), GroupId(0)] };
        let plan = CyclicScan { lookahead: 1 };
        let resident = [GroupId(0), GroupId(1), GroupId(2)];
        assert_eq!(plan.victim(&resident, &sched, 3), GroupId(2));
    }
}
