// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for deciding, before a reinforcement-learning run spends GPU hours, whether the
// serving engine and the trainer agree on the policy, for its clients. If your team needs expertise in the numerical
// validation of on-policy training against an inference engine then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The numerics gate of a policy-gradient run: before any update, does the trainer's own forward pass reproduce the
//! log-probabilities the serving engine recorded for the tokens it sampled?
//!
//! A rollout carries the behaviour log-probability `old` of each sampled token (what the engine said), and the
//! trainer's teacher-forced forward gives `new` for the same token at the same weights. At step 0 the two are one
//! policy, so `ratio = exp(new - old)` should be one: whatever it is instead is the serve/train gap, and it eats the
//! clip band before the first update. The gate measures it over a set of rollouts and compares it with bounds fixed in
//! advance:
//!
//! * **gap**: `new - old` per token: its mean absolute value, its largest magnitude and its mean (a bias);
//! * **mismatch KL**: the k3 estimator `r - 1 - ln r`, `r = exp(new - old)`, of `KL(engine || trainer)` at the
//!   sampled tokens (non-negative, zero iff the two agree);
//! * **ratio**: its mean, smallest and largest value and the share of tokens outside the clip band;
//! * **effective sample size**: `(sum w)^2 / (n * sum w^2)` of the importance weights `w = ratio`, per token and per
//!   sequence (`w = exp(sum_t gap_t)`, computed against the largest to stay finite): one means the batch is as good as
//!   on-policy, a small value means a few tokens or sequences carry the update;
//! * **gradient health**: the finiteness and norm of the policy-gradient step's gradient on each batch drawn, and the
//!   largest against the median (the spike that the step guard exists for).
//!
//! The statistics are plain sums, so [`NumericsAccumulator`] is fed one sequence or batch at a time and a report is
//! taken whenever asked.

use super::grpo_term::ClipBand;

/// What a gate run must satisfy; every bound that is `None` is reported and not enforced.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NumericsBounds {
    /// The mean of `|new - old|` over the tokens.
    pub max_mean_abs_gap: Option<f32>,
    /// The largest `|new - old|` of any token.
    pub max_abs_gap: Option<f32>,
    /// The mean k3 mismatch KL over the tokens.
    pub max_mismatch_kl: Option<f32>,
    /// The smallest per-sequence effective-sample-size fraction.
    pub min_sequence_ess_fraction: Option<f32>,
    /// The largest gradient norm of any batch.
    pub max_grad_norm: Option<f32>,
}

impl NumericsBounds {
    /// Refuse a bound that has no meaning.
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [("max_mean_abs_gap", self.max_mean_abs_gap), ("max_abs_gap", self.max_abs_gap), ("max_mismatch_kl", self.max_mismatch_kl), ("max_grad_norm", self.max_grad_norm)] {
            if let Some(v) = v.filter(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(format!("numerics bound {name} is {v}; it must be finite and positive (leave it unset to only report)"));
            }
        }
        if let Some(v) = self.min_sequence_ess_fraction.filter(|v| !(v.is_finite() && *v > 0.0 && *v <= 1.0)) {
            return Err(format!("numerics bound min_sequence_ess_fraction is {v}; it must lie in (0, 1]"));
        }
        Ok(())
    }

    /// Whether any bound is set.
    pub fn is_enforced(&self) -> bool {
        *self != NumericsBounds::default()
    }
}

/// What the gate measured, and which bounds it broke.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct NumericsReport {
    pub sequences: usize,
    pub tokens: usize,
    /// `new - old` over the tokens: mean absolute value, largest magnitude and mean.
    pub gap_mean_abs: f32,
    pub gap_max_abs: f32,
    pub gap_mean: f32,
    /// The k3 estimate of `KL(engine || trainer)` at the sampled tokens.
    pub mismatch_kl: f32,
    /// `exp(new - old)`: mean, smallest, largest, and the share of tokens outside the clip band.
    pub ratio_mean: f32,
    pub ratio_min: f32,
    pub ratio_max: f32,
    pub ratio_outside_band_fraction: f32,
    /// Effective sample size as a fraction of the sample, over token weights and over sequence weights.
    pub ess_fraction_token: f32,
    pub ess_fraction_sequence: f32,
    /// Gradient norms of the batches drawn: how many, how many were not finite, the largest, the median, and the
    /// largest over the median (zero when the median is).
    pub grad_batches: usize,
    pub grad_nonfinite: usize,
    /// Where the largest gap sat: the sequence (in the order the run measured them), the token's offset among the
    /// sequence's supervised tokens, and the recorded and the trainer's log-probability there.
    pub worst_gap: Option<WorstGap>,
    pub grad_norm_max: f32,
    pub grad_norm_median: f32,
    pub grad_norm_spike_ratio: f32,
    /// One line per bound the run broke, with its value and the bound. Empty when it passed.
    pub violations: Vec<String>,
}

impl NumericsReport {
    /// Whether every enforced bound held.
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
}

/// The token with the largest `|new - old|` of a run.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct WorstGap {
    pub sequence: usize,
    pub offset: usize,
    pub recorded: f32,
    pub trainer: f32,
}

/// The sums a [`NumericsReport`] is made of.
#[derive(Clone, Debug, Default)]
pub struct NumericsAccumulator {
    tokens: usize,
    abs_gap_sum: f64,
    gap_max_abs: f32,
    gap_sum: f64,
    kl_sum: f64,
    ratio_sum: f64,
    ratio_sq_sum: f64,
    ratio_min: Option<f32>,
    ratio_max: f32,
    outside: usize,
    /// Log of each sequence's importance weight: the sum of its token gaps.
    sequence_log_weights: Vec<f64>,
    worst: Option<WorstGap>,
    grad_norms: Vec<f32>,
    grad_nonfinite: usize,
}

impl NumericsAccumulator {
    /// Add one sequence's supervised tokens: `old[i]` the engine's log-probability of token `i`, `new[i]` the
    /// trainer's. `band` is the clip band the ratios are counted against.
    pub fn add_sequence(&mut self, old: &[f32], new: &[f32], band: ClipBand) {
        assert_eq!(old.len(), new.len(), "one trainer log-probability per recorded one");
        if old.is_empty() {
            return;
        }
        let mut log_weight = 0.0f64;
        for (offset, (&o, &n)) in old.iter().zip(new).enumerate() {
            let gap = f64::from(n) - f64::from(o);
            if self.worst.is_none_or(|w| gap.abs() > f64::from((w.trainer - w.recorded).abs())) {
                self.worst = Some(WorstGap { sequence: self.sequence_log_weights.len(), offset, recorded: o, trainer: n });
            }
            log_weight += gap;
            self.tokens += 1;
            self.abs_gap_sum += gap.abs();
            self.gap_sum += gap;
            self.gap_max_abs = self.gap_max_abs.max(gap.abs() as f32);
            let ratio = gap.exp();
            self.kl_sum += ratio - 1.0 - gap;
            self.ratio_sum += ratio;
            self.ratio_sq_sum += ratio * ratio;
            self.ratio_min = Some(self.ratio_min.map_or(ratio as f32, |m| m.min(ratio as f32)));
            self.ratio_max = self.ratio_max.max(ratio as f32);
            self.outside += usize::from(ratio < 1.0 - f64::from(band.low) || ratio > 1.0 + f64::from(band.high));
        }
        self.sequence_log_weights.push(log_weight);
    }

    /// Add the gradient norm of one batch (`None`: not finite).
    pub fn add_grad_norm(&mut self, norm: Option<f32>) {
        match norm.filter(|n| n.is_finite()) {
            Some(n) => self.grad_norms.push(n),
            None => self.grad_nonfinite += 1,
        }
    }

    /// The report so far, judged against `bounds`.
    pub fn report(&self, bounds: &NumericsBounds) -> NumericsReport {
        let n = self.tokens.max(1) as f64;
        let mut r = NumericsReport { sequences: self.sequence_log_weights.len(), tokens: self.tokens, ..Default::default() };
        if self.tokens > 0 {
            r.gap_mean_abs = (self.abs_gap_sum / n) as f32;
            r.gap_max_abs = self.gap_max_abs;
            r.gap_mean = (self.gap_sum / n) as f32;
            r.mismatch_kl = (self.kl_sum / n) as f32;
            r.ratio_mean = (self.ratio_sum / n) as f32;
            r.ratio_min = self.ratio_min.unwrap_or(1.0);
            r.ratio_max = self.ratio_max;
            r.ratio_outside_band_fraction = self.outside as f32 / self.tokens as f32;
            r.ess_fraction_token = if self.ratio_sq_sum > 0.0 { (self.ratio_sum * self.ratio_sum / (n * self.ratio_sq_sum)) as f32 } else { 0.0 };
            r.ess_fraction_sequence = ess_fraction_of_log_weights(&self.sequence_log_weights);
        }
        r.worst_gap = self.worst;
        r.grad_batches = self.grad_norms.len() + self.grad_nonfinite;
        r.grad_nonfinite = self.grad_nonfinite;
        if !self.grad_norms.is_empty() {
            let mut sorted = self.grad_norms.clone();
            sorted.sort_by(f32::total_cmp);
            r.grad_norm_max = sorted[sorted.len() - 1];
            r.grad_norm_median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0;
            r.grad_norm_spike_ratio = if r.grad_norm_median > 0.0 { r.grad_norm_max / r.grad_norm_median } else { 0.0 };
        }
        r.violations = violations(&r, bounds);
        r
    }
}

/// `(sum w)^2 / (n sum w^2)` of `w = exp(log_w)`, computed against the largest weight so it cannot overflow; `0` for no
/// weights.
fn ess_fraction_of_log_weights(log_w: &[f64]) -> f32 {
    let Some(max) = log_w.iter().copied().reduce(f64::max) else { return 0.0 };
    let (sum, sq): (f64, f64) = log_w.iter().map(|l| (l - max).exp()).fold((0.0, 0.0), |(s, q), w| (s + w, q + w * w));
    (sum * sum / (log_w.len() as f64 * sq)) as f32
}

fn violations(r: &NumericsReport, bounds: &NumericsBounds) -> Vec<String> {
    let mut out = Vec::new();
    let mut over = |name: &str, value: f32, bound: Option<f32>, upper: bool| {
        if let Some(bound) = bound {
            let broken = value.is_nan() || if upper { value > bound } else { value < bound };
            if broken {
                out.push(format!("{name} {value:.6} is {} the bound {bound}", if upper { "over" } else { "under" }));
            }
        }
    };
    over("mean |log-prob gap|", r.gap_mean_abs, bounds.max_mean_abs_gap, true);
    over("max |log-prob gap|", r.gap_max_abs, bounds.max_abs_gap, true);
    over("mismatch KL", r.mismatch_kl, bounds.max_mismatch_kl, true);
    over("sequence ESS fraction", r.ess_fraction_sequence, bounds.min_sequence_ess_fraction, false);
    over("gradient norm", r.grad_norm_max, bounds.max_grad_norm, true);
    if r.grad_nonfinite > 0 {
        out.push(format!("{} of {} batches had a non-finite gradient", r.grad_nonfinite, r.grad_batches));
    }
    if r.tokens == 0 {
        out.push("no supervised token was measured".to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const BAND: ClipBand = ClipBand { low: 0.2, high: 0.2 };

    #[test]
    fn identical_log_probabilities_are_one_policy() {
        let mut acc = NumericsAccumulator::default();
        acc.add_sequence(&[-1.0, -2.0, -0.5], &[-1.0, -2.0, -0.5], BAND);
        acc.add_sequence(&[-3.0], &[-3.0], BAND);
        acc.add_grad_norm(Some(0.3));
        let r = acc.report(&NumericsBounds { max_mean_abs_gap: Some(1e-6), max_mismatch_kl: Some(1e-6), min_sequence_ess_fraction: Some(0.99), max_grad_norm: Some(1.0), ..Default::default() });
        assert!(r.passed(), "{:?}", r.violations);
        assert_eq!((r.sequences, r.tokens), (2, 4));
        assert_eq!((r.gap_mean_abs, r.gap_max_abs, r.mismatch_kl, r.ratio_mean, r.ratio_outside_band_fraction), (0.0, 0.0, 0.0, 1.0, 0.0));
        assert!((r.ess_fraction_token - 1.0).abs() < 1e-6 && (r.ess_fraction_sequence - 1.0).abs() < 1e-6);
    }

    /// Gaps of +0.1 and -0.3: the statistics by hand.
    #[test]
    fn the_gap_the_mismatch_kl_and_the_ratios_are_the_definitions() {
        let mut acc = NumericsAccumulator::default();
        acc.add_sequence(&[-1.0, -2.0], &[-0.9, -2.3], BAND);
        let r = acc.report(&NumericsBounds::default());
        assert!((r.gap_mean_abs - 0.2).abs() < 1e-6 && (r.gap_max_abs - 0.3).abs() < 1e-6 && (r.gap_mean + 0.1).abs() < 1e-6, "{r:?}");
        let (a, b) = (0.1f64.exp(), (-0.3f64).exp());
        let kl = ((a - 1.0 - 0.1) + (b - 1.0 + 0.3)) / 2.0;
        assert!((f64::from(r.mismatch_kl) - kl).abs() < 1e-6, "{} against {kl}", r.mismatch_kl);
        assert_eq!(r.worst_gap, Some(WorstGap { sequence: 0, offset: 1, recorded: -2.0, trainer: -2.3 }));
        assert!((f64::from(r.ratio_mean) - (a + b) / 2.0).abs() < 1e-6);
        assert!((r.ratio_min - b as f32).abs() < 1e-6 && (r.ratio_max - a as f32).abs() < 1e-6);
        assert_eq!(r.ratio_outside_band_fraction, 0.5, "0.74 is under 0.8, 1.105 is inside");
        let ess = (a + b).powi(2) / (2.0 * (a * a + b * b));
        assert!((f64::from(r.ess_fraction_token) - ess).abs() < 1e-6);
    }

    #[test]
    fn one_dominant_sequence_collapses_the_effective_sample_size() {
        let mut acc = NumericsAccumulator::default();
        acc.add_sequence(&[-1.0; 4], &[-1.0 + 5.0 / 4.0; 4], BAND);
        for _ in 0..9 {
            acc.add_sequence(&[-1.0; 4], &[-1.0; 4], BAND);
        }
        let r = acc.report(&NumericsBounds { min_sequence_ess_fraction: Some(0.5), ..Default::default() });
        assert!(r.ess_fraction_sequence < 0.15, "{}", r.ess_fraction_sequence);
        assert!(r.violations.iter().any(|v| v.contains("sequence ESS fraction") && v.contains("under")), "{:?}", r.violations);
        // A sequence weight of e^5 does not overflow the computation, nor does e^500.
        let mut huge = NumericsAccumulator::default();
        huge.add_sequence(&[-1.0; 4], &[-1.0 + 125.0; 4], BAND);
        huge.add_sequence(&[-1.0; 4], &[-1.0; 4], BAND);
        assert!(huge.report(&NumericsBounds::default()).ess_fraction_sequence.is_finite());
    }

    #[test]
    fn gradient_health_reads_the_spike_against_the_median_and_non_finite_batches() {
        let mut acc = NumericsAccumulator::default();
        acc.add_sequence(&[-1.0], &[-1.0], BAND);
        for norm in [0.3, 0.2, 0.4, 21.1, 0.3] {
            acc.add_grad_norm(Some(norm));
        }
        let r = acc.report(&NumericsBounds::default());
        assert_eq!((r.grad_batches, r.grad_nonfinite, r.grad_norm_max, r.grad_norm_median), (5, 0, 21.1, 0.3));
        assert!((r.grad_norm_spike_ratio - 70.333).abs() < 0.01, "{}", r.grad_norm_spike_ratio);
        assert!(acc.report(&NumericsBounds { max_grad_norm: Some(1.0), ..Default::default() }).violations.iter().any(|v| v.contains("gradient norm")));
        acc.add_grad_norm(None);
        let r = acc.report(&NumericsBounds::default());
        assert!(!r.passed() && r.violations[0].contains("non-finite"), "a non-finite gradient fails the gate even with no bound: {:?}", r.violations);
    }

    #[test]
    fn a_run_that_measured_nothing_fails_and_a_nan_gap_breaks_every_upper_bound() {
        assert!(!NumericsAccumulator::default().report(&NumericsBounds::default()).passed());
        let mut acc = NumericsAccumulator::default();
        acc.add_sequence(&[-1.0], &[f32::NAN], BAND);
        let r = acc.report(&NumericsBounds { max_mean_abs_gap: Some(10.0), max_mismatch_kl: Some(10.0), ..Default::default() });
        assert_eq!(r.violations.len(), 2, "NaN is not within a bound: {:?}", r.violations);
    }

    #[test]
    fn bounds_are_validated_by_name() {
        assert!(NumericsBounds::default().validate().is_ok() && !NumericsBounds::default().is_enforced());
        for bad in [NumericsBounds { max_mean_abs_gap: Some(0.0), ..Default::default() }, NumericsBounds { max_grad_norm: Some(f32::NAN), ..Default::default() }, NumericsBounds { min_sequence_ess_fraction: Some(1.5), ..Default::default() }] {
            assert!(bad.validate().unwrap_err().contains("numerics bound"), "{bad:?}");
        }
    }
}
