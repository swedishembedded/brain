// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What earlier episodes still score, and the backward transfer over it.
//!
//! A continual reader's claim is not "it learned episode 12"; it is "it
//! learned episode 12 and episode 3 still works". The second half is a
//! matrix: `R[k][j]` is how episode `j`'s own frozen probes score after
//! episode `k` was read. Backward transfer is the mean, over every earlier
//! episode that was looked at again, of what it scores now minus what it
//! scored when it was learned.
//!
//! ## The matrix is SPARSE, and that is the point
//!
//! Scoring every earlier episode after every new one is quadratic, and a
//! reader that did it would spend its entire budget auditing. The audit
//! schedule samples instead, which is why this crate reports a detection
//! latency rather than a retention guarantee. So the matrix here records
//! only what was actually measured, and [`Retention::bwt`] returns `None`
//! when nothing has been re-observed yet.
//!
//! **A BWT of zero and a BWT of nothing are different answers.** An absent
//! measurement defaulted to `0.0` reads as perfect retention - the single
//! most flattering number available - which is exactly the substitution the
//! acceptance block exists to prevent.
//!
//! ## The diagonal is what an episode scored when it was learned
//!
//! `R[j][j]` is recorded from the arm that was actually promoted, at the
//! moment it was promoted. Comparing a later observation against anything
//! else - the base model's zero-shot score, a mean over the corpus - would
//! measure something other than forgetting.
//!
//! Swedish Embedded AB builds continual-learning systems that can say what
//! they lost as precisely as what they gained. If your team needs a model
//! that keeps learning from your own material without quietly giving back
//! last week's capability, you can procure our services by sending an email
//! to info@swedishembedded.com.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::stream::EpisodeId;

/// One earlier episode's history: what it scored when it was learned, and
/// the last thing it scored when it was looked at again.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    /// `R[j][j]`: the score its own probes got from the arm that was
    /// promoted for it.
    pub learned: f64,
    /// The most recent re-observation, and the episode after which it was
    /// taken. `None` until the schedule has brought this episode round
    /// again.
    pub latest: Option<f64>,
    pub latest_after: u64,
    /// How many times it has been re-observed. Reported because one
    /// observation of one episode is not the same evidence as forty, and a
    /// bare BWT cannot tell them apart.
    pub observations: u64,
}

/// The retention matrix, accumulated as the reader goes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Retention {
    traces: BTreeMap<EpisodeId, Trace>,
}

impl Retention {
    /// Record `R[j][j]`: episode `id` was learned, and its own probes scored
    /// `score` under the arm that was promoted.
    ///
    /// Re-learning the same id overwrites the diagonal and forgets any
    /// re-observations of the old one: they were about a different adapter,
    /// and comparing across the two would report an artefact of the
    /// replacement as forgetting.
    pub fn learned(&mut self, id: &EpisodeId, score: f64) {
        self.traces.insert(id.clone(), Trace { learned: score, latest: None, latest_after: 0, observations: 0 });
    }

    /// Record `R[k][j]`: after episode `k`, episode `j` scored `score`.
    ///
    /// Silently ignores an episode with no diagonal. An audit block can
    /// outlive the promotion it belonged to (the pool retires adapters), and
    /// a re-observation with nothing to compare against is not a row of this
    /// matrix.
    pub fn observed(&mut self, after: u64, id: &EpisodeId, score: f64) {
        let Some(t) = self.traces.get_mut(id) else { return };
        t.latest = Some(score);
        t.latest_after = after;
        t.observations += 1;
    }

    /// Backward transfer: the mean of `latest - learned` over every episode
    /// that has been re-observed at least once.
    ///
    /// `None` when none has been. Negative means earlier episodes are worse
    /// than when they were learned, which is forgetting.
    pub fn bwt(&self) -> Option<f64> {
        let deltas: Vec<f64> = self.traces.values().filter_map(|t| t.latest.map(|l| l - t.learned)).collect();
        if deltas.is_empty() {
            return None;
        }
        Some(deltas.iter().sum::<f64>() / deltas.len() as f64)
    }

    /// The single worst drop any one episode has taken, which is what the
    /// per-block bar is about. A healthy mean can hide one episode that
    /// collapsed, and the acceptance block asks about both.
    ///
    /// Reported as a POSITIVE magnitude: `0.0` when nothing dropped.
    pub fn worst_drop(&self) -> Option<f64> {
        let drops: Vec<f64> = self.traces.values().filter_map(|t| t.latest.map(|l| t.learned - l)).collect();
        drops.into_iter().fold(None, |acc: Option<f64>, d| Some(acc.map_or(d, |a: f64| a.max(d)))).map(|d| d.max(0.0))
    }

    /// How many episodes carry a diagonal, and how many of those have been
    /// looked at again. The second number is the weight any BWT here
    /// carries.
    pub fn coverage(&self) -> (usize, usize) {
        (self.traces.len(), self.traces.values().filter(|t| t.latest.is_some()).count())
    }

    pub fn trace(&self, id: &EpisodeId) -> Option<&Trace> {
        self.traces.get(id)
    }

    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }
}

/// Backward transfer over a DENSE matrix, where `r[i][j]` is the score on
/// task `j` after training task `i`.
///
/// The classical definition, for a caller that has the whole square - a
/// fixed-task-sequence study rather than a sampling reader. The reader's own
/// sparse path is [`Retention::bwt`]; this is the same quantity computed
/// from a different shape, and lives beside it so the two cannot drift.
pub fn bwt_dense(r: &[Vec<f64>]) -> f64 {
    let t = r.len();
    if t < 2 {
        return 0.0;
    }
    let last = &r[t - 1];
    let mut total = 0.0;
    for (j, row) in r.iter().enumerate().take(t - 1) {
        total += last[j] - row[j];
    }
    total / (t - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> EpisodeId {
        EpisodeId::of(s)
    }

    /// The distinction the whole module exists for: nothing re-observed is
    /// not the same answer as nothing lost.
    #[test]
    fn a_matrix_with_no_re_observation_reports_no_bwt_rather_than_zero() {
        let mut r = Retention::default();
        r.learned(&id("a"), 0.8);
        r.learned(&id("b"), 0.6);
        assert_eq!(r.bwt(), None, "two diagonals and no second look is not a backward transfer of zero");
        assert_eq!(r.coverage(), (2, 0));

        r.observed(5, &id("a"), 0.8);
        assert_eq!(r.bwt(), Some(0.0), "an episode re-observed at what it learned HAS a backward transfer, and it is zero");
        assert_eq!(r.coverage(), (2, 1));
    }

    /// Forgetting is negative, and it is measured against what the episode
    /// scored when it was learned rather than against any other baseline.
    #[test]
    fn bwt_is_negative_when_an_earlier_episode_decayed() {
        let mut r = Retention::default();
        r.learned(&id("a"), 1.0);
        r.learned(&id("b"), 0.5);
        r.observed(9, &id("a"), 0.6);
        r.observed(9, &id("b"), 0.5);
        assert!((r.bwt().expect("observed") - -0.2).abs() < 1e-12, "{:?}", r.bwt());
        assert!((r.worst_drop().expect("observed") - 0.4).abs() < 1e-12, "the worst single drop, not the mean");
    }

    /// A pooled mean can look healthy while one episode collapsed, which is
    /// what the per-block bar is for.
    #[test]
    fn a_healthy_mean_does_not_hide_the_worst_block() {
        let mut r = Retention::default();
        r.learned(&id("a"), 1.0);
        r.learned(&id("b"), 0.0);
        r.observed(4, &id("a"), 0.0);
        r.observed(4, &id("b"), 1.0);
        assert!(r.bwt().expect("observed").abs() < 1e-12, "the mean is zero");
        assert!((r.worst_drop().expect("observed") - 1.0).abs() < 1e-12, "and one episode lost everything");
    }

    /// An observation of an episode that was never learned here is not a row
    /// of this matrix, and must not invent a diagonal of zero for it.
    #[test]
    fn an_observation_without_a_diagonal_is_ignored_rather_than_compared_against_zero() {
        let mut r = Retention::default();
        r.observed(3, &id("ghost"), 0.9);
        assert_eq!(r.bwt(), None);
        assert_eq!(r.coverage(), (0, 0));
    }

    /// Re-learning an id replaces the adapter, so the history of the old one
    /// is not retention evidence about the new one.
    #[test]
    fn relearning_an_episode_restarts_its_history() {
        let mut r = Retention::default();
        r.learned(&id("a"), 0.2);
        r.observed(2, &id("a"), 0.9);
        r.learned(&id("a"), 0.7);
        assert_eq!(r.bwt(), None, "the new diagonal has not been re-observed");
        assert_eq!(r.trace(&id("a")).expect("a").learned, 0.7);
    }

    /// The dense form is the same quantity from a different shape.
    #[test]
    fn the_dense_form_matches_the_classical_definition() {
        let r = vec![vec![0.9, 0.0], vec![0.7, 0.8]];
        assert!((bwt_dense(&r) - -0.2).abs() < 1e-12, "{}", bwt_dense(&r));
        assert_eq!(bwt_dense(&[vec![0.9]]), 0.0, "one task has no earlier task to transfer to");
        assert_eq!(bwt_dense(&[]), 0.0);
    }
}
