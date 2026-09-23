// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! When the reader stops learning, deciding WHY before deciding what to do.
//!
//! A falling promote rate has two causes that look identical from outside
//! and have opposite fixes:
//!
//! - **Interference.** The capacity to hold all of this exists, but training
//!   episode by episode keeps trading one piece of it for another. The fix
//!   is more rehearsal and a gentler rate. Adding capacity would not help and
//!   would cost memory for nothing.
//! - **Saturation.** The capacity does not exist. The fix is another adapter.
//!   More rehearsal would make it strictly worse, by spending an already
//!   exhausted budget on old material.
//!
//! Guessing between them is how a design ends up with a page of tuned
//! constants. They can be TOLD apart, by one measurement: train a single
//! adapter jointly on everything learned so far and score it on the same
//! probes the sequential run is scored on. Joint training is the best the
//! shape can do, so it bounds what any schedule could have achieved.
//!
//! | joint | sequential | diagnosis | action |
//! |---|---|---|---|
//! | passes | fails | the capacity is there and the schedule lost it | raise rehearsal |
//! | fails | fails | the capacity is not there | add an adapter |
//! | passes | passes | nothing is wrong | nothing |
//! | fails | passes | the oracle is not an upper bound | nothing, and say so |
//!
//! **That fourth row is not in the design this implements, and it matters.**
//! The oracle is supposed to bound the sequential run from above, so an
//! oracle scoring BELOW it means the measurement is untrustworthy - an
//! under-trained oracle, a mismatched probe set, a seed with too much
//! variance. Both remedies are wrong under a broken measurement, so the only
//! safe action is none, reported as such rather than silently folded into
//! "healthy".
//!
//! ## The oracle is expensive, so it is not asked often
//!
//! It costs a full training run over the whole history. It is asked only
//! when the promote rate over a recent WINDOW has fallen through a floor,
//! never on a schedule and never on a healthy run. The window rather than
//! the whole run is deliberate: a reader that promoted well for a thousand
//! episodes and has promoted nothing for the last fifty has a problem now,
//! and a lifetime average would hide it for a long time.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrowthConfig {
    /// Promote rate over the window below which the oracle is worth its cost.
    pub promote_rate_floor: f64,
    /// Episodes the promote rate is measured over. The oracle is not asked
    /// until this many have been recorded, since a rate over three episodes
    /// is noise.
    pub window: usize,
    /// Score at or above which an arm counts as passing.
    pub pass_at: f64,
    /// Episodes after a growth before another may fire, so one diagnosis
    /// cannot add capacity every episode while the new adapter is still
    /// earning its place.
    pub cooldown: u64,
    /// What a rehearsal cap is multiplied by when interference is diagnosed.
    pub rehearsal_step: f64,
}

impl Default for GrowthConfig {
    fn default() -> Self {
        GrowthConfig { promote_rate_floor: 0.25, window: 64, pass_at: 0.60, cooldown: 128, rehearsal_step: 1.5 }
    }
}

/// What the two arms say together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diagnosis {
    /// Both arms pass. Nothing is wrong.
    Healthy,
    /// Joint training holds it; the sequential schedule does not.
    Interference,
    /// Joint training cannot hold it either.
    Saturation,
    /// The oracle scored below the run it is supposed to bound.
    Inconclusive,
}

/// What to do about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Nothing,
    RaiseRehearsal { from: usize, to: usize },
    Grow,
    /// A diagnosis that would have acted, but must not. Carries why, so a
    /// ledger row says "capacity, but inside the cooldown" rather than
    /// nothing at all.
    Hold { diagnosis: Diagnosis, reason: &'static str },
}

/// Promote history, and whether the oracle is worth asking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Growth {
    cfg: GrowthConfig,
    recent: VecDeque<bool>,
    last_grew: Option<u64>,
    now: u64,
}

impl Growth {
    pub fn new(cfg: GrowthConfig) -> Growth {
        Growth { cfg, recent: VecDeque::with_capacity(cfg.window), last_grew: None, now: 0 }
    }

    /// Record one episode's outcome and advance the episode counter.
    pub fn record(&mut self, promoted: bool) {
        if self.recent.len() == self.cfg.window {
            self.recent.pop_front();
        }
        self.recent.push_back(promoted);
        self.now += 1;
    }

    pub fn now(&self) -> u64 {
        self.now
    }

    /// Promote rate over the window, once a full window has been seen.
    /// `None` before that, because a rate over a part-window is not one.
    pub fn promote_rate(&self) -> Option<f64> {
        if self.cfg.window == 0 || self.recent.len() < self.cfg.window {
            return None;
        }
        Some(self.recent.iter().filter(|p| **p).count() as f64 / self.recent.len() as f64)
    }

    /// Whether the oracle is worth a full training run over the history.
    pub fn oracle_due(&self) -> bool {
        self.promote_rate().is_some_and(|r| r < self.cfg.promote_rate_floor)
    }

    /// Read the two arms. A pure function of the scores, so it can be
    /// reasoned about and tested without a model anywhere near it.
    pub fn diagnose(&self, joint: f64, sequential: f64) -> Diagnosis {
        let (joint_ok, seq_ok) = (joint >= self.cfg.pass_at, sequential >= self.cfg.pass_at);
        match (joint_ok, seq_ok) {
            (true, true) => Diagnosis::Healthy,
            (true, false) => Diagnosis::Interference,
            (false, false) if joint >= sequential => Diagnosis::Saturation,
            // Both failed AND the oracle came in below the run it bounds:
            // the same broken-measurement signal as the row below, and it
            // has to win, or a bad oracle reads as a capacity problem and
            // grows the pool for nothing.
            (false, false) => Diagnosis::Inconclusive,
            (false, true) => Diagnosis::Inconclusive,
        }
    }

    /// Turn a diagnosis into an action, applying the growth cooldown.
    pub fn act(&mut self, diagnosis: Diagnosis, rehearsal_cap: usize) -> Action {
        match diagnosis {
            Diagnosis::Healthy => Action::Nothing,
            Diagnosis::Inconclusive => Action::Hold { diagnosis, reason: "the oracle scored below the run it is supposed to bound, so neither remedy is safe" },
            // Deliberately NOT under the growth cooldown. Raising rehearsal
            // costs no memory, and putting the cheap remedy behind the
            // expensive one's timer would leave a diagnosable problem
            // untreated for the length of a cooldown it has nothing to do
            // with.
            Diagnosis::Interference => {
                let to = ((rehearsal_cap as f64) * self.cfg.rehearsal_step).round() as usize;
                Action::RaiseRehearsal { from: rehearsal_cap, to: to.max(rehearsal_cap + 1) }
            }
            Diagnosis::Saturation => {
                let cooling = self.last_grew.is_some_and(|t| self.now.saturating_sub(t) < self.cfg.cooldown);
                if cooling {
                    return Action::Hold { diagnosis, reason: "a newly added adapter has not yet had its cooldown to earn its place" };
                }
                self.last_grew = Some(self.now);
                Action::Grow
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GrowthConfig {
        GrowthConfig { promote_rate_floor: 0.25, window: 10, pass_at: 0.60, cooldown: 20, rehearsal_step: 2.0 }
    }

    fn after(outcomes: &[bool]) -> Growth {
        let mut g = Growth::new(cfg());
        for &p in outcomes {
            g.record(p);
        }
        g
    }

    /// The oracle costs a full training run over the whole history, so a run
    /// that is working must never pay for it.
    #[test]
    fn a_healthy_run_never_asks_the_oracle_and_a_stalled_one_does() {
        let healthy = after(&[true; 10]);
        assert_eq!(healthy.promote_rate(), Some(1.0));
        assert!(!healthy.oracle_due(), "a run promoting everything has nothing to diagnose");

        let stalled = after(&[false; 10]);
        assert_eq!(stalled.promote_rate(), Some(0.0));
        assert!(stalled.oracle_due());
    }

    /// A rate over a part-window is not a rate. Asking the oracle on the
    /// strength of the first three episodes of a run would pay its cost for
    /// noise.
    #[test]
    fn the_oracle_is_not_asked_before_a_full_window_has_been_seen() {
        let young = after(&[false; 9]);
        assert_eq!(young.promote_rate(), None, "nine of a ten episode window is not a rate");
        assert!(!young.oracle_due());
        assert!(after(&[false; 10]).oracle_due(), "and the tenth makes it one");
    }

    /// A lifetime average would hide a run that has just stopped working
    /// behind a thousand episodes of history that did.
    #[test]
    fn the_rate_is_measured_over_the_window_not_the_whole_run() {
        let mut g = after(&[true; 200]);
        assert!(!g.oracle_due());
        for _ in 0..10 {
            g.record(false);
        }
        assert_eq!(g.promote_rate(), Some(0.0), "the window must have rolled over entirely");
        assert!(g.oracle_due(), "a recent stall must be visible despite a long healthy history");
    }

    /// The two rows with opposite fixes. Getting these the wrong way round
    /// spends memory on an interference problem or rehearses an exhausted
    /// adapter harder.
    #[test]
    fn interference_raises_rehearsal_and_saturation_grows() {
        let mut g = after(&[false; 10]);

        // Joint training holds it, the sequential schedule does not.
        assert_eq!(g.diagnose(0.85, 0.30), Diagnosis::Interference);
        assert_eq!(g.act(Diagnosis::Interference, 100), Action::RaiseRehearsal { from: 100, to: 200 });

        // Neither holds it.
        assert_eq!(g.diagnose(0.30, 0.25), Diagnosis::Saturation);
        assert_eq!(g.act(Diagnosis::Saturation, 100), Action::Grow);
    }

    /// The stay-silent row.
    #[test]
    fn both_arms_passing_is_healthy_and_does_nothing() {
        let mut g = after(&[false; 10]);
        assert_eq!(g.diagnose(0.90, 0.80), Diagnosis::Healthy);
        assert_eq!(g.act(Diagnosis::Healthy, 100), Action::Nothing);
    }

    /// The row the design did not have. An oracle scoring below the run it
    /// bounds means the measurement is broken, and BOTH remedies are wrong
    /// under a broken measurement.
    #[test]
    fn an_oracle_below_the_run_it_bounds_is_inconclusive_and_nothing_is_done() {
        let mut g = after(&[false; 10]);
        assert_eq!(g.diagnose(0.20, 0.75), Diagnosis::Inconclusive);
        match g.act(Diagnosis::Inconclusive, 100) {
            Action::Hold { diagnosis, reason } => {
                assert_eq!(diagnosis, Diagnosis::Inconclusive);
                assert!(!reason.is_empty(), "a hold must say why, or a ledger row cannot explain itself");
            }
            other => panic!("expected a Hold, got {other:?}"),
        }
    }

    /// The case the four row table does not reach: both arms fail AND the
    /// oracle is below the run. A plain "both failed" reading would call
    /// that saturation and grow the pool, but the ordering says the
    /// measurement is broken, and growing on a broken measurement spends
    /// memory to fix something that was never diagnosed.
    #[test]
    fn both_arms_failing_with_the_oracle_underneath_is_inconclusive_not_saturation() {
        let g = after(&[false; 10]);
        assert_eq!(g.diagnose(0.25, 0.40), Diagnosis::Inconclusive);
        // ...and the ordinary saturation reading is unaffected.
        assert_eq!(g.diagnose(0.40, 0.25), Diagnosis::Saturation);
        assert_eq!(g.diagnose(0.30, 0.30), Diagnosis::Saturation, "an exact tie is a valid bound, not a broken one");
    }

    /// A new adapter has to be given a chance to earn its place before
    /// another is added on the strength of the same stalled window.
    #[test]
    fn growth_does_not_fire_again_inside_its_cooldown() {
        let mut g = after(&[false; 10]);
        assert_eq!(g.act(Diagnosis::Saturation, 100), Action::Grow);

        for _ in 0..19 {
            g.record(false);
            match g.act(Diagnosis::Saturation, 100) {
                Action::Hold { diagnosis: Diagnosis::Saturation, .. } => {}
                other => panic!("expected a Hold inside the cooldown, got {other:?}"),
            }
        }
        g.record(false);
        assert_eq!(g.act(Diagnosis::Saturation, 100), Action::Grow, "past the cooldown it must be able to grow again");
    }

    /// Interference is not under the growth cooldown: raising rehearsal
    /// costs no memory, and holding it back would leave the one cheap remedy
    /// blocked by the expensive one's timer.
    #[test]
    fn the_cooldown_governs_growth_only_and_not_rehearsal() {
        let mut g = after(&[false; 10]);
        assert_eq!(g.act(Diagnosis::Saturation, 100), Action::Grow);
        assert_eq!(g.act(Diagnosis::Interference, 64), Action::RaiseRehearsal { from: 64, to: 128 });
    }
}
