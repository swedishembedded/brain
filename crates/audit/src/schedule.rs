// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Re-checking what the reader already learned, at a cost that does not grow
//! with how much it has learned.
//!
//! The natural anchor suite for a continual learner is every earlier
//! episode's frozen probes, decoded for both arms on every decision. That is
//! correct and it is `O(N)` per episode, `O(N^2)` over a run: at 48 probes an
//! episode, decision 1,000 costs 96,000 decodes. Correct for a twelve-cycle
//! study, impossible for a stream.
//!
//! So the budget is fixed and the CLAIM changes shape. A reader on an
//! unbounded stream cannot honestly say "nothing was forgotten", because it
//! cannot afford to have looked. It can say:
//!
//! > any regression larger than `d` on any earlier episode is detected within
//! > `L` episodes
//!
//! and both numbers fall out of the schedule rather than being asserted.
//! [`Schedule::detection_latency`] is `L`; [`block_drop_bar`] is where `d`
//! comes from.
//!
//! ## Two halves, one job each
//!
//! **Rotation guarantees coverage.** Strict round-robin over the bank,
//! `rotating` blocks a tick, so every episode is revisited within
//! `ceil(N / rotating)` ticks and `L` is exact rather than expected. Nothing
//! is allowed to influence this half, which is the point: a priority rule
//! that could delay an episode would turn a guarantee into a hope.
//!
//! **The canary spends attention where it is most likely to pay.** A separate,
//! smaller draw, chosen by priority - an episode whose last score sat near
//! the pass/fail boundary is the one most likely to tip - and resampled only
//! every `canary_refresh` ticks so it is stable enough to compare against
//! itself. It makes NO coverage claim, which is exactly why priority is
//! allowed to decide it.
//!
//! Keeping the two apart is what lets the reader be opportunistic without
//! weakening the only guarantee it offers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::stream::EpisodeId;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Probe decodes per tick, for ONE arm. The gate decodes two arms, so the
    /// real spend is twice this.
    pub budget: usize,
    /// Probes sampled from each episode that is scheduled. Fewer means more
    /// episodes per tick and a shorter latency, paid for by a looser
    /// per-block bar - see [`block_drop_bar`].
    pub probes_per_block: usize,
    /// How many of a tick's blocks go to the canary rather than to rotation.
    /// Must leave at least one for rotation, or coverage never completes.
    pub canary_blocks: usize,
    /// Ticks between canary resamples.
    pub canary_refresh: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        AuditConfig { budget: 192, probes_per_block: 16, canary_blocks: 3, canary_refresh: 32 }
    }
}

/// What to decode this tick.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Episodes revisited by rotation. These are what the coverage guarantee
    /// is made of.
    pub rotating: Vec<EpisodeId>,
    /// Episodes revisited because they look most likely to have tipped.
    pub canary: Vec<EpisodeId>,
    /// Probes to draw from each scheduled episode.
    pub probes_per_block: usize,
}

impl Plan {
    /// Probe decodes this plan costs, for one arm.
    pub fn decodes(&self) -> usize {
        (self.rotating.len() + self.canary.len()) * self.probes_per_block
    }

    /// Every episode this plan touches, rotation and canary together.
    pub fn blocks(&self) -> Vec<&EpisodeId> {
        self.rotating.iter().chain(self.canary.iter()).collect()
    }
}

/// The bank of earlier episodes, and whose turn it is.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Schedule {
    cfg: AuditConfig,
    order: Vec<EpisodeId>,
    priority: BTreeMap<EpisodeId, f64>,
    cursor: usize,
    canary: Vec<EpisodeId>,
    canary_drawn: u64,
    now: u64,
}

impl Schedule {
    pub fn new(cfg: AuditConfig) -> Schedule {
        Schedule { cfg, order: Vec::new(), priority: BTreeMap::new(), cursor: 0, canary: Vec::new(), canary_drawn: 0, now: 0 }
    }

    /// Add an episode to the bank. It joins the rotation and is revisited
    /// within one full sweep like any other.
    pub fn admit(&mut self, id: EpisodeId) {
        if !self.order.contains(&id) {
            self.order.push(id);
        }
    }

    /// How likely this episode looks to have tipped, in `[0, 1]`. The canary
    /// draw prefers the high ones. A caller sets it from the last score: an
    /// episode sitting near the pass/fail boundary is the one worth watching.
    pub fn set_priority(&mut self, id: &EpisodeId, priority: f64) {
        self.priority.insert(id.clone(), priority.clamp(0.0, 1.0));
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Blocks a tick spends on rotation. At least one whenever the budget
    /// affords any block at all, or coverage would never complete.
    pub fn rotating_per_tick(&self) -> usize {
        let blocks = self.blocks_per_tick();
        // Rotation keeps at least one block whenever the budget affords any
        // block at all. A canary sized to consume the whole tick would
        // otherwise stop coverage completing, silently and forever.
        blocks.saturating_sub(self.cfg.canary_blocks).max(blocks.min(1))
    }

    /// Blocks a tick spends on the canary. Whatever the budget has left after
    /// rotation's guaranteed share, which is why an oversized
    /// `canary_blocks` costs the canary rather than the guarantee.
    pub fn canary_per_tick(&self) -> usize {
        self.blocks_per_tick().saturating_sub(self.rotating_per_tick()).min(self.cfg.canary_blocks)
    }

    /// Blocks the budget affords at this sample size.
    fn blocks_per_tick(&self) -> usize {
        self.cfg.budget / self.cfg.probes_per_block.max(1)
    }

    /// Ticks to revisit every episode in the bank. This is the `L` the
    /// reader reports instead of a retention guarantee.
    pub fn detection_latency(&self) -> u64 {
        let per = self.rotating_per_tick();
        if self.order.is_empty() || per == 0 {
            return 0;
        }
        self.order.len().div_ceil(per) as u64
    }

    /// What to decode now. Advances the rotation.
    pub fn plan(&mut self) -> Plan {
        let n = self.order.len();
        if n == 0 {
            self.now += 1;
            return Plan { rotating: Vec::new(), canary: Vec::new(), probes_per_block: self.cfg.probes_per_block };
        }

        // Rotation: strict round-robin, nothing consulted but the cursor.
        // An episode is never taken twice in one tick, so a bank smaller
        // than a tick is simply covered whole.
        let per = self.rotating_per_tick().min(n);
        let mut rotating = Vec::with_capacity(per);
        for _ in 0..per {
            rotating.push(self.order[self.cursor].clone());
            self.cursor = (self.cursor + 1) % n;
        }

        // Canary: redrawn only on the refresh boundary, so it is stable
        // enough between draws to be compared against itself.
        let due = self.canary.is_empty() || self.now.saturating_sub(self.canary_drawn) >= self.cfg.canary_refresh;
        if due {
            let mut by_priority: Vec<&EpisodeId> = self.order.iter().collect();
            // Descending priority, ties broken by id so two equal scores do
            // not depend on iteration order.
            by_priority.sort_by(|a, b| {
                let pa = self.priority.get(*a).copied().unwrap_or(0.0);
                let pb = self.priority.get(*b).copied().unwrap_or(0.0);
                pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.cmp(b))
            });
            self.canary = by_priority.into_iter().take(self.canary_per_tick().min(n)).cloned().collect();
            self.canary_drawn = self.now;
        }

        // The canary is drawn from the whole bank, so it can name an episode
        // rotation also reached this tick. Decoding it twice would spend
        // budget on one block and report it as two.
        let canary: Vec<EpisodeId> = self.canary.iter().filter(|c| !rotating.contains(c)).cloned().collect();

        self.now += 1;
        Plan { rotating, canary, probes_per_block: self.cfg.probes_per_block }
    }
}

/// How far a single earlier episode's block may fall before a candidate is
/// refused, given that the block is now a SAMPLE of that episode's probes
/// rather than all of them.
///
/// `promote::document::MAX_BLOCK_DROP` was derived for a complete block of
/// `MIN_HELD_OUT_PROBES` probes, where the sampling standard error is at most
/// `0.5/sqrt(48) = 0.072` and a 0.20 bar sits about 2.8 of them out. Sample
/// fewer and that error grows, so a bar that stayed at 0.20 would start
/// rejecting candidates for sampling noise - the classic way a tightened
/// audit makes a system look worse than it is.
///
/// So the bar is the pre-registered one OR two standard errors of the
/// realised sample, whichever is looser. It can only ever loosen: a thinner
/// sample buys a shorter detection latency and pays for it in the smallest
/// regression it can still see, which is the trade stated rather than hidden.
pub fn block_drop_bar(probes_per_block: usize) -> f64 {
    if probes_per_block == 0 {
        return f64::INFINITY;
    }
    // A pass/fail block mean has standard error at most `0.5/sqrt(m)`; two of
    // them is `1/sqrt(m)`.
    let two_se = 1.0 / (probes_per_block as f64).sqrt();
    promote::document::MAX_BLOCK_DROP.max(two_se)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn ep(i: usize) -> EpisodeId {
        EpisodeId::of(&format!("episode {i}"))
    }

    fn bank(cfg: AuditConfig, n: usize) -> Schedule {
        let mut s = Schedule::new(cfg);
        for i in 0..n {
            s.admit(ep(i));
        }
        s
    }

    /// The guarantee, asserted rather than described. Run exactly the
    /// reported number of ticks and every episode in the bank must have been
    /// revisited by ROTATION - the canary does not count, because it makes no
    /// coverage claim.
    #[test]
    fn rotation_covers_the_whole_bank_within_the_reported_latency() {
        let cfg = AuditConfig { budget: 96, probes_per_block: 16, canary_blocks: 2, canary_refresh: 8 };
        let mut s = bank(cfg, 47);
        let latency = s.detection_latency();

        let mut seen: BTreeSet<EpisodeId> = BTreeSet::new();
        for _ in 0..latency {
            seen.extend(s.plan().rotating);
        }
        assert_eq!(seen.len(), 47, "every episode must be revisited within the reported latency of {latency} ticks");
    }

    /// And the latency must be TIGHT, or the reader is reporting a number
    /// that overstates how fast it would notice.
    #[test]
    fn the_reported_latency_is_the_real_one_and_not_a_loose_bound() {
        let cfg = AuditConfig { budget: 96, probes_per_block: 16, canary_blocks: 2, canary_refresh: 8 };
        let mut s = bank(cfg, 47);
        let latency = s.detection_latency();

        let mut seen: BTreeSet<EpisodeId> = BTreeSet::new();
        for _ in 0..latency - 1 {
            seen.extend(s.plan().rotating);
        }
        assert!(seen.len() < 47, "if one tick fewer already covered the bank, the reported latency is too pessimistic");
    }

    /// The budget is the point. A plan that overspent it would put the reader
    /// back on the quadratic path it exists to leave.
    #[test]
    fn no_tick_ever_schedules_more_decodes_than_the_budget() {
        let cfg = AuditConfig { budget: 100, probes_per_block: 16, canary_blocks: 2, canary_refresh: 4 };
        let mut s = bank(cfg, 200);
        for _ in 0..50 {
            let p = s.plan();
            assert!(p.decodes() <= cfg.budget, "a tick spent {} of a {} budget", p.decodes(), cfg.budget);
        }
    }

    /// The canary is compared against itself over time, so it has to hold
    /// still between refreshes - and it has to actually move on one, or the
    /// refresh interval is decoration.
    #[test]
    fn the_canary_holds_still_between_refreshes_and_moves_on_one() {
        let cfg = AuditConfig { budget: 96, probes_per_block: 16, canary_blocks: 2, canary_refresh: 4 };
        let mut s = bank(cfg, 60);
        for i in 0..60 {
            s.set_priority(&ep(i), i as f64 / 60.0);
        }

        let first = s.plan().canary;
        for _ in 1..4 {
            assert_eq!(s.plan().canary, first, "the canary must be stable within its refresh interval");
        }
        // Priorities now point somewhere else entirely.
        for i in 0..60 {
            s.set_priority(&ep(i), 1.0 - i as f64 / 60.0);
        }
        assert_ne!(s.plan().canary, first, "a refresh must actually redraw");
    }

    /// Priority buys attention out of the canary and never out of rotation,
    /// which is what keeps the coverage guarantee a guarantee.
    #[test]
    fn priority_decides_the_canary_and_cannot_touch_rotation() {
        let cfg = AuditConfig { budget: 96, probes_per_block: 16, canary_blocks: 2, canary_refresh: 64 };
        let mut flat = bank(cfg, 40);
        let mut skewed = bank(cfg, 40);
        skewed.set_priority(&ep(37), 1.0);
        skewed.set_priority(&ep(38), 0.9);

        let a = flat.plan();
        let b = skewed.plan();
        assert_eq!(a.rotating, b.rotating, "rotation must be unaffected by priority");
        assert!(b.canary.contains(&ep(37)) && b.canary.contains(&ep(38)), "the canary must follow priority, got {:?}", b.canary);
    }

    /// A newly learned episode has to enter the rotation, or the most recent
    /// thing the reader knows is the one it never re-checks.
    #[test]
    fn an_episode_admitted_later_is_covered_within_one_further_sweep() {
        let cfg = AuditConfig { budget: 64, probes_per_block: 16, canary_blocks: 1, canary_refresh: 8 };
        let mut s = bank(cfg, 20);
        for _ in 0..s.detection_latency() {
            s.plan();
        }
        s.admit(ep(999));
        let mut seen = BTreeSet::new();
        for _ in 0..s.detection_latency() {
            seen.extend(s.plan().rotating);
        }
        assert!(seen.contains(&ep(999)), "a late arrival must be revisited like any other");
    }

    /// The trade, made explicit. A thinner sample shortens the latency and
    /// pays for it in the smallest regression it can still see.
    #[test]
    fn a_thinner_block_loosens_the_regression_bar_and_a_full_one_does_not() {
        let full = block_drop_bar(promote::document::MIN_HELD_OUT_PROBES);
        assert!(
            (full - promote::document::MAX_BLOCK_DROP).abs() < 1e-12,
            "a complete block must keep the pre-registered bar, got {full}"
        );

        let thin = block_drop_bar(12);
        assert!(thin > full, "a 12 probe block has a larger standard error and must loosen the bar, got {thin} against {full}");
        assert!((thin - 1.0 / 12f64.sqrt()).abs() < 1e-12, "the loosened bar must be two standard errors of the realised sample");

        assert!(block_drop_bar(4096) >= full, "the bar may loosen but must never tighten below what was pre-registered");
    }

    /// A bank smaller than one tick is fully covered every tick, and must say
    /// so rather than reporting a latency of zero.
    #[test]
    fn a_bank_smaller_than_one_tick_reports_a_latency_of_one() {
        let cfg = AuditConfig { budget: 160, probes_per_block: 16, canary_blocks: 2, canary_refresh: 8 };
        let mut s = bank(cfg, 3);
        assert_eq!(s.detection_latency(), 1);
        assert_eq!(s.plan().rotating.len(), 3, "it cannot rotate over more episodes than exist");
    }

    /// Rotation must always get at least one block, whatever the canary asks
    /// for, or coverage silently never completes.
    #[test]
    fn a_canary_that_would_consume_the_whole_budget_cannot_starve_rotation() {
        let cfg = AuditConfig { budget: 32, probes_per_block: 16, canary_blocks: 9, canary_refresh: 8 };
        let mut s = bank(cfg, 50);
        assert!(s.rotating_per_tick() >= 1, "rotation must keep at least one block");
        let p = s.plan();
        assert!(!p.rotating.is_empty());
        assert!(p.decodes() <= cfg.budget, "and the budget still holds");
    }
}
