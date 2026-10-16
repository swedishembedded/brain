// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a GRPO run did, for the people deciding whether to trust it: the
//! per-step counters ([`GrpoStats`]), the shared log a caller keeps after the
//! objective has been consumed by the training loop ([`CycleLog`]), and the
//! typed reasons a row or a run was refused.
//!
//! Swedish Embedded AB implements solutions for making reinforcement-learning
//! runs auditable - how many groups taught nothing, how far the sampler was
//! from the trainer, what was skipped and why - for its clients. If your team
//! needs expertise in operating RL fine-tuning you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::fmt;

/// Consecutive optimiser steps with a non-finite loss or gradient after which
/// the run is aborted instead of skipping again.
pub const MAX_CONSECUTIVE_NONFINITE: u32 = 3;

/// The run was aborted because the loss or gradient was non-finite on
/// [`MAX_CONSECUTIVE_NONFINITE`] optimiser steps in a row. Carried as the
/// source of the `std::io::Error` the training loop returns, so a caller can
/// `downcast_ref` it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonFiniteSteps {
    pub consecutive: u32,
}

impl fmt::Display for NonFiniteSteps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the loss or gradient was not finite on {} optimiser steps in a row; the weights were left as they were", self.consecutive)
    }
}

impl std::error::Error for NonFiniteSteps {}

/// Why a queued row was not trained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowRefusal {
    /// Its behaviour policy is `age` optimiser steps older than the current
    /// one, beyond the `bound` allowed.
    Stale { age: u64, bound: u32 },
}

/// Counters of one objective's life.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GrpoStats {
    /// Groups (or RFT attempt series) drawn, and how many of them held no
    /// non-zero advantage (every reward equal, or nothing verified).
    pub groups_sampled: usize,
    pub groups_zero_variance: usize,
    /// Completions sampled (RFT attempts included).
    pub completions_sampled: usize,
    /// Groups drawn beyond what a step nominally needs (dynamic sampling).
    pub resamples: usize,
    /// Optimiser steps asked for, those that held fewer rows than their
    /// `grad_accum` slots, and the slots left empty in total.
    pub steps: usize,
    pub steps_short: usize,
    pub rows_short: usize,
    /// Rows trained.
    pub rows_trained: usize,
    /// Steps the loop was allowed to apply, and steps skipped for a
    /// non-finite loss or gradient.
    pub steps_applied: usize,
    pub steps_skipped_nonfinite: usize,
    /// Completions cut to fit the row, stop tokens trained as targets, and
    /// rows whose sampler truncated (top-k/top-p) the behaviour policy.
    pub completions_truncated: usize,
    pub eos_targets: usize,
    pub truncated_sampling_rows: usize,
    /// Rows refused as too old, and the last reason.
    pub rows_refused: usize,
    pub last_refusal: Option<RowRefusal>,
    /// Largest `|ratio - 1|` over the tokens of rows trained under the policy
    /// that was current when they were packed (0 up to float noise when the
    /// behaviour policy is the trained one).
    pub max_onpolicy_ratio_dev: f32,
    /// Largest `|trainer_lp - sampler_lp|` over untruncated own-model rollouts:
    /// how far the sampler's reported probability was from the trainer's.
    pub max_sampler_gap: f32,
    /// k3 KL to the reference, summed over supervised tokens and their count.
    pub kl_sum: f64,
    pub kl_tokens: usize,
}

impl GrpoStats {
    pub(crate) fn record_group(&mut self, attempts: usize, zero_variance: bool, truncated: usize, eos_targets: usize, truncated_sampling_rows: usize, sampler_gap: f32) {
        self.groups_sampled += 1;
        self.groups_zero_variance += zero_variance as usize;
        self.completions_sampled += attempts;
        self.completions_truncated += truncated;
        self.eos_targets += eos_targets;
        self.truncated_sampling_rows += truncated_sampling_rows;
        self.max_sampler_gap = self.max_sampler_gap.max(sampler_gap);
    }

    pub(crate) fn record_step(&mut self, rows: usize, slots: usize) {
        self.steps += 1;
        self.rows_trained += rows;
        if rows < slots {
            self.steps_short += 1;
            self.rows_short += slots - rows;
        }
    }

    pub(crate) fn record_row(&mut self, ratio_dev: f32, kl: Option<(f32, usize)>) {
        self.max_onpolicy_ratio_dev = self.max_onpolicy_ratio_dev.max(ratio_dev);
        if let Some((sum, tokens)) = kl {
            self.kl_sum += f64::from(sum);
            self.kl_tokens += tokens;
        }
    }

    pub(crate) fn refuse(&mut self, why: RowRefusal) {
        self.rows_refused += 1;
        self.last_refusal = Some(why);
    }

    /// Mean k3 KL to the reference over every trained token, when one was measured.
    pub fn mean_kl(&self) -> Option<f32> {
        (self.kl_tokens > 0).then(|| (self.kl_sum / self.kl_tokens as f64) as f32)
    }

    /// Fraction of sampled groups that carried no signal.
    pub fn zero_variance_frac(&self) -> f32 {
        if self.groups_sampled == 0 {
            0.0
        } else {
            self.groups_zero_variance as f32 / self.groups_sampled as f32
        }
    }

    /// The counters as `(name, value)` metrics for the periodic log line.
    pub(crate) fn metrics(&self) -> Vec<(&'static str, f32)> {
        let mut m = vec![
            ("grpo_zero_variance_frac", self.zero_variance_frac()),
            ("grpo_resamples", self.resamples as f32),
            ("grpo_rows_trained", self.rows_trained as f32),
            ("grpo_rows_short", self.rows_short as f32),
            ("grpo_steps_skipped_nonfinite", self.steps_skipped_nonfinite as f32),
            ("grpo_rows_refused", self.rows_refused as f32),
            ("grpo_completions_truncated", self.completions_truncated as f32),
            ("grpo_sampler_gap", self.max_sampler_gap),
        ];
        if let Some(kl) = self.mean_kl() {
            m.push(("grpo_kl_ref", kl));
        }
        m
    }
}

/// Observability hook for the structural assertion that every completion span
/// written into a training set is a member of the multiset the policy actually
/// sampled that cycle: every completion span a group drew (as verified) and
/// every span the objective trained, in order, plus the counters. A plain
/// `Rc<RefCell<..>>` handle rather than a getter on the objective, because the
/// objective is typically consumed by value (`model::fit_with`) - attach a log
/// with [`super::grpo::Grpo::with_log`] before handing the objective off, keep
/// your own clone, and read it back afterwards.
#[derive(Clone, Default)]
pub struct CycleLog(std::rc::Rc<std::cell::RefCell<CycleLogInner>>);

#[derive(Default)]
struct CycleLogInner {
    sampled: Vec<Vec<u32>>,
    rewards: Vec<f32>,
    trained: Vec<Vec<u32>>,
    stats: GrpoStats,
}

impl CycleLog {
    pub fn new() -> CycleLog {
        CycleLog::default()
    }

    /// Every completion span sampled (kept or dropped) since construction, in
    /// sample order. A completion longer than the row holds appears cut to
    /// the span that was verified and trained.
    pub fn sampled(&self) -> Vec<Vec<u32>> {
        self.0.borrow().sampled.clone()
    }

    /// The verified reward of every completion in [`CycleLog::sampled`],
    /// parallel to it and in the same order: the honest cycle-wide reward series.
    pub fn rewards(&self) -> Vec<f32> {
        self.0.borrow().rewards.clone()
    }

    /// Every completion span actually trained on, in training order.
    pub fn trained(&self) -> Vec<Vec<u32>> {
        self.0.borrow().trained.clone()
    }

    /// The objective's counters as of its last optimiser step.
    pub fn stats(&self) -> GrpoStats {
        self.0.borrow().stats.clone()
    }

    pub(crate) fn push_sampled(&self, tokens: Vec<u32>) {
        self.0.borrow_mut().sampled.push(tokens);
    }

    pub(crate) fn push_reward(&self, reward: f32) {
        self.0.borrow_mut().rewards.push(reward);
    }

    pub(crate) fn push_trained(&self, tokens: Vec<u32>) {
        self.0.borrow_mut().trained.push(tokens);
    }

    pub(crate) fn set_stats(&self, stats: &GrpoStats) {
        self.0.borrow_mut().stats = stats.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_log_clones_share_the_same_underlying_log() {
        // A caller keeps a clone before handing the original into something
        // that consumes it by value, and still sees every push made through
        // the moved clone afterward.
        let log = CycleLog::new();
        let handle = log.clone();
        log.push_sampled(vec![1, 2, 3]);
        log.push_sampled(vec![4, 5]);
        log.push_trained(vec![1, 2, 3]);
        drop(log);
        assert_eq!(handle.sampled(), vec![vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(handle.trained(), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn a_step_short_of_rows_is_counted_with_the_slots_it_left_empty() {
        let mut s = GrpoStats::default();
        s.record_step(4, 4);
        s.record_step(1, 4);
        s.record_step(0, 4);
        assert_eq!((s.steps, s.steps_short, s.rows_short, s.rows_trained), (3, 2, 7, 5));
    }
}
