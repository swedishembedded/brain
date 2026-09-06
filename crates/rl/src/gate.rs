// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Programmatic promote/reject: replaces "a human reads a printed
//! leaderboard" with a single [`gate`] call producing a [`GateReport`]
//! (self-improve roadmap P16).
//!
//! Promotion requires **all four**:
//! 1. An exact one-sided paired binomial sign test, `p <= alpha`, conditioned
//!    on discordant pairs only ([`bench::metrics::sign_test`], hoisted from
//!    `wan`'s finetune A/B gate rather than re-derived here).
//! 2. A minimum effect size (mean candidate-minus-incumbent score), so a
//!    significant-but-trivial win does not promote.
//! 3. The anchor suite not regressing past a fixed budget.
//! 4. A non-degeneracy check: candidate completion entropy over held-out
//!    tasks must not have collapsed relative to the incumbent's, catching
//!    mode collapse / reward hacking that a pure win-rate would miss.
//!
//! [`gate`] itself is a pure function over already-scored pairs - it does
//! not decode anything. The caller is responsible for producing
//! [`GateInput`]'s scores by decoding BOTH arms greedily, on the same
//! held-out tasks in the same order, from disk-reloaded checkpoints (never
//! the freshly-trained in-memory instance) - what is actually served must be
//! what is scored.

/// Why a candidate was rejected. Each variant carries the numbers that
/// decided it, so a caller can log/report without recomputing anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cause {
    /// The paired sign test did not reach significance at `alpha`.
    NotSignificant { p_value: f64, alpha: f64 },
    /// The test was significant, but the win margin is too small to matter.
    EffectTooSmall { effect_size: f64, min_effect_size: f64 },
    /// The anchor suite regressed past the allowed budget.
    AnchorRegressed { delta: f64, budget: f64 },
    /// Candidate completion entropy collapsed relative to the incumbent's -
    /// the mode-collapse / reward-hacking catch.
    Degenerate { entropy_ratio: f64, min_entropy_ratio: f64 },
}

/// The programmatic promote/reject decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    Promote,
    Reject(Cause),
}

/// Paired, already-scored inputs to [`gate`]. Every score pair must come from
/// the SAME held-out tasks in the SAME order for both arms, both decoded
/// greedily from disk-reloaded checkpoints (see this module's doc comment).
pub struct GateInput<'a> {
    /// Per-task score, candidate arm (e.g. 1.0/0.0 pass-fail, or a continuous
    /// margin) - same tasks, same order as `incumbent_scores`.
    pub candidate_scores: &'a [f64],
    /// Per-task score, incumbent arm.
    pub incumbent_scores: &'a [f64],
    /// Anchor-suite headline score, candidate (e.g. `bench::Metrics::score`).
    pub anchor_candidate: f64,
    /// Anchor-suite headline score, incumbent.
    pub anchor_incumbent: f64,
    /// Mean completion entropy over held-out tasks, candidate.
    pub entropy_candidate: f64,
    /// Mean completion entropy over held-out tasks, incumbent.
    pub entropy_incumbent: f64,
}

/// Thresholds [`gate`] checks `GateInput` against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateConfig {
    /// Sign-test significance level. Promotion requires `p_value <= alpha`.
    pub alpha: f64,
    /// Minimum `mean(candidate) - mean(incumbent)` to count as a real win.
    pub min_effect_size: f64,
    /// Maximum allowed anchor-suite regression (`anchor_incumbent -
    /// anchor_candidate`); higher-is-better anchor scores assumed.
    pub anchor_budget: f64,
    /// Minimum `entropy_candidate / entropy_incumbent` before flagging
    /// degeneracy.
    pub min_entropy_ratio: f64,
}

impl Default for GateConfig {
    /// `alpha = 0.05` (the spec's fixed significance level); the other three
    /// are deliberately conservative starting points a caller is expected to
    /// tune per regime, not tuned science.
    fn default() -> Self {
        GateConfig { alpha: 0.05, min_effect_size: 0.02, anchor_budget: 0.02, min_entropy_ratio: 0.5 }
    }
}

/// The full result of a gate run: the decision plus every number that fed it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateReport {
    pub decision: Decision,
    /// Discordant-pair count the sign test was computed over.
    pub n_discordant: usize,
    /// Of those, how many favored the candidate.
    pub k_wins: usize,
    pub p_value: f64,
    pub effect_size: f64,
    /// `anchor_incumbent - anchor_candidate`; positive means regression.
    pub anchor_delta: f64,
    pub entropy_ratio: f64,
}

/// Score `input` against `cfg` and return the promote/reject decision plus
/// every intermediate number. Checks run in a fixed order (significance,
/// effect size, anchor, degeneracy) and the first one to fail is reported as
/// the [`Cause`] - all four still get computed either way, so `GateReport`
/// always carries the full picture regardless of which one rejected.
pub fn gate(input: &GateInput, cfg: &GateConfig) -> GateReport {
    let sign = bench::metrics::sign_test(input.candidate_scores, input.incumbent_scores);
    let effect_size = mean(input.candidate_scores) - mean(input.incumbent_scores);
    let anchor_delta = input.anchor_incumbent - input.anchor_candidate;
    let entropy_ratio = if input.entropy_incumbent.abs() > f64::EPSILON { input.entropy_candidate / input.entropy_incumbent } else { 1.0 };

    let decision = if sign.p_value > cfg.alpha {
        Decision::Reject(Cause::NotSignificant { p_value: sign.p_value, alpha: cfg.alpha })
    } else if effect_size < cfg.min_effect_size {
        Decision::Reject(Cause::EffectTooSmall { effect_size, min_effect_size: cfg.min_effect_size })
    } else if anchor_delta > cfg.anchor_budget {
        Decision::Reject(Cause::AnchorRegressed { delta: anchor_delta, budget: cfg.anchor_budget })
    } else if entropy_ratio < cfg.min_entropy_ratio {
        Decision::Reject(Cause::Degenerate { entropy_ratio, min_entropy_ratio: cfg.min_entropy_ratio })
    } else {
        Decision::Promote
    };

    GateReport { decision, n_discordant: sign.n, k_wins: sign.k, p_value: sign.p_value, effect_size, anchor_delta, entropy_ratio }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 20 tasks, candidate wins 18/2 ties-none -> comfortably significant,
    /// clears every other bar too -> promote.
    fn winning_pairs() -> (Vec<f64>, Vec<f64>) {
        let mut candidate = vec![1.0; 18];
        candidate.extend([0.0, 0.0]);
        let mut incumbent = vec![0.0; 18];
        incumbent.extend([1.0, 1.0]);
        (candidate, incumbent)
    }

    #[test]
    fn promotes_when_all_four_bars_clear() {
        let (candidate, incumbent) = winning_pairs();
        let input = GateInput {
            candidate_scores: &candidate,
            incumbent_scores: &incumbent,
            anchor_candidate: 0.90,
            anchor_incumbent: 0.90,
            entropy_candidate: 2.0,
            entropy_incumbent: 2.0,
        };
        let report = gate(&input, &GateConfig::default());
        assert_eq!(report.decision, Decision::Promote);
        assert_eq!(report.n_discordant, 20);
        assert_eq!(report.k_wins, 18);
    }

    #[test]
    fn rejects_not_significant_on_a_coin_flip_split() {
        // 10 discordant pairs split 5/5 -> p = 1.0, nowhere near alpha.
        let candidate = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let incumbent = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let input = GateInput {
            candidate_scores: &candidate,
            incumbent_scores: &incumbent,
            anchor_candidate: 0.9,
            anchor_incumbent: 0.9,
            entropy_candidate: 2.0,
            entropy_incumbent: 2.0,
        };
        let report = gate(&input, &GateConfig::default());
        match report.decision {
            Decision::Reject(Cause::NotSignificant { p_value, alpha }) => {
                assert!(p_value > alpha);
            }
            other => panic!("expected NotSignificant, got {other:?}"),
        }
    }

    #[test]
    fn rejects_effect_too_small_despite_significance() {
        // Same win/loss pattern as `winning_pairs` (significant), but the
        // actual magnitude of every win is tiny -> significant, trivial.
        let mut candidate = vec![0.51; 18];
        candidate.extend([0.50, 0.50]);
        let mut incumbent = vec![0.50; 18];
        incumbent.extend([0.51, 0.51]);
        let input = GateInput {
            candidate_scores: &candidate,
            incumbent_scores: &incumbent,
            anchor_candidate: 0.9,
            anchor_incumbent: 0.9,
            entropy_candidate: 2.0,
            entropy_incumbent: 2.0,
        };
        let cfg = GateConfig { min_effect_size: 0.05, ..GateConfig::default() };
        let report = gate(&input, &cfg);
        match report.decision {
            Decision::Reject(Cause::EffectTooSmall { effect_size, min_effect_size }) => {
                assert!(effect_size < min_effect_size);
            }
            other => panic!("expected EffectTooSmall, got {other:?}"),
        }
    }

    #[test]
    fn rejects_anchor_regressed_even_with_a_strong_primary_win() {
        let (candidate, incumbent) = winning_pairs();
        let input = GateInput {
            candidate_scores: &candidate,
            incumbent_scores: &incumbent,
            anchor_candidate: 0.60, // well below incumbent - catastrophic forgetting
            anchor_incumbent: 0.90,
            entropy_candidate: 2.0,
            entropy_incumbent: 2.0,
        };
        let report = gate(&input, &GateConfig::default());
        match report.decision {
            Decision::Reject(Cause::AnchorRegressed { delta, budget }) => {
                assert!(delta > budget);
                assert!((delta - 0.30).abs() < 1e-9);
            }
            other => panic!("expected AnchorRegressed, got {other:?}"),
        }
    }

    #[test]
    fn rejects_degenerate_when_entropy_collapses() {
        let (candidate, incumbent) = winning_pairs();
        let input = GateInput {
            candidate_scores: &candidate,
            incumbent_scores: &incumbent,
            anchor_candidate: 0.90,
            anchor_incumbent: 0.90,
            entropy_candidate: 0.05, // near-zero: candidate always emits the same completion
            entropy_incumbent: 2.0,
        };
        let report = gate(&input, &GateConfig::default());
        match report.decision {
            Decision::Reject(Cause::Degenerate { entropy_ratio, min_entropy_ratio }) => {
                assert!(entropy_ratio < min_entropy_ratio);
            }
            other => panic!("expected Degenerate, got {other:?}"),
        }
    }
}
