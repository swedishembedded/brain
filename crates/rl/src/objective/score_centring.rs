// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for stable off-policy reinforcement learning of language models from
// asynchronously sampled rollouts for its clients. If your team needs expertise in policy-gradient estimators that stay
// unbiased when the sampler lags the learner then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The score-centred policy-gradient term of one sampled token ("Score Centering Stabilizes Off-policy RL",
//! <https://huggingface.co/papers/2609.20807>, appendix A.2; the estimator TRL's `AsyncGRPOTrainer` trains with).
//!
//! A token `a` drawn by the sampler `mu` with advantage `A` has the importance ratio `r = pi(a) / mu(a)`, masked to
//! `f(r) = r 1[low <= r <= high]`. The plain masked estimator `-A f(r_a) grad log pi(a)` is biased once the mask cuts:
//! its expectation under a constant advantage is not zero. Score centring subtracts the expectation of the score
//! under the sampler, `E_{v~mu}[f(r_v) grad log pi(v)] = sum_v pi(v) 1[low <= r_v <= high] grad log pi(v)`, estimated
//! over the sampler's top-K ids `v_j` (probabilities `q_j`, trainer probabilities `p_j`) plus one bucket for the rest
//! of the vocabulary:
//!
//! * `head_j = p_j` when `low q_j <= p_j <= high q_j`, else 0;
//! * the tail ratio `max(1 - sum p, 0) / max(1 - sum q, tiny)` decides whether the remaining mass is in the band
//!   (`tail = 1`; a top-K whose mass is the whole distribution to float32 rounding has no tail) and, since `sum_{v not in top} pi(v) grad log pi(v) = -sum_j p_j grad log pi(v_j)` (the scores of a
//!   distribution average to zero), the bucket's share is `-tail p_j` on each listed id;
//! * `residual_j = head_j - tail p_j`, and the token's loss is `-A (f(r_a) log pi(a) - sum_j residual_j log pi(v_j))`
//!   with `f(r_a)` and the residuals held constant.
//!
//! As a weighted cross-entropy (`sum weight * (-log pi)`) that is the weight `A f(r_a)` at the sampled token and the
//! extra targets `-A residual_j` at the `v_j` ([`model::head_ce::SparseTargets`]), one backward of the LM head.
//! With the sampler's whole distribution as its top-K the tail is empty and the estimate's expectation is the exact
//! centred gradient, which is zero for an advantage that is the same for every token.

/// The importance band `[low, high]` of the ratio `pi / mu` outside which a token, and a listed id, carry no score
/// (TRL's `vllm_importance_sampling_clip_min` / `_max`).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CentringBand {
    pub low: f32,
    pub high: f32,
}

impl CentringBand {
    /// TRL's defaults: `[0.5, 5.0]`.
    pub const DEFAULT: CentringBand = CentringBand { low: 0.5, high: 5.0 };

    /// Refuse a band with no meaning: `0 <= low < 1 < high`, both finite.
    pub fn validate(self) -> Result<(), String> {
        if !(self.low.is_finite() && self.high.is_finite() && self.low >= 0.0 && self.low < 1.0 && self.high > 1.0) {
            return Err(format!("the score-centring band [{}, {}] must satisfy 0 <= low < 1 < high, both finite", self.low, self.high));
        }
        Ok(())
    }

    fn holds(self, ratio: f64) -> bool {
        ratio >= f64::from(self.low) && ratio <= f64::from(self.high)
    }
}

/// The sampler's unlisted mass, per listed id, at or below which its top-K is the whole distribution: float32 rounding
/// of each listed probability.
const LISTED_WHOLE: f64 = f32::EPSILON as f64;

/// What one token's score-centred term is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CentredTerm {
    /// The cross-entropy weight at the sampled token, `A f(r_a)`.
    pub weight: f32,
    /// The loss value `-A (f(r_a) log pi(a) - sum_j residual_j log pi(v_j))`.
    pub loss: f32,
    /// The token's ratio left the band (`f(r_a) = 0`).
    pub masked: bool,
    /// `sum_j q_j`: how much of the sampler's distribution its top-K lists.
    pub sampler_mass: f32,
    /// The remaining mass's ratio was inside the band, so its bucket was centred.
    pub tail: bool,
}

/// The term of one token: `old_lp` / `new_lp` the sampler's and the trainer's log-probability of the sampled token,
/// `sampler_top` / `trainer_top` the two log-probabilities at each of the sampler's K listed ids (NaN in
/// `sampler_top`: an unused slot). The extra-target coefficients `-A residual_j` are written to `coefs` (0 at an
/// unused slot).
///
/// # Panics
/// The three slices differ in length.
pub fn centred_term(old_lp: f32, new_lp: f32, advantage: f32, band: CentringBand, sampler_top: &[f32], trainer_top: &[f32], coefs: &mut [f32]) -> CentredTerm {
    assert!(sampler_top.len() == trainer_top.len() && coefs.len() == sampler_top.len(), "centred_term: {} sampler, {} trainer and {} coefficient slots", sampler_top.len(), trainer_top.len(), coefs.len());
    let (low, high) = (f64::from(band.low), f64::from(band.high));
    let ratio = (f64::from(new_lp) - f64::from(old_lp)).exp();
    let masked = !band.holds(ratio);
    let f = if masked { 0.0 } else { ratio };
    let advantage = f64::from(advantage);
    let (mut q_sum, mut p_sum) = (0.0f64, 0.0f64);
    let q: Vec<f64> = sampler_top.iter().map(|&l| if l.is_nan() { 0.0 } else { f64::from(l).exp() }).collect();
    // A slot the sampler did not list has no trainer probability either (TRL: `p * (q > 0)`).
    let p: Vec<f64> = trainer_top.iter().zip(&q).map(|(&l, &q)| if q > 0.0 { f64::from(l).exp() } else { 0.0 }).collect();
    for (q, p) in q.iter().zip(&p) {
        q_sum += q;
        p_sum += p;
    }
    // Only a zero denominator is guarded: flooring small positive masses could move their ratio into the band. A listed
    // mass within float32 rounding of the whole (the sampler listed its distribution) leaves no bucket at all: its
    // ratio would be one rounding error over another, and the exact estimator has no tail term then.
    let unlisted = 1.0 - q_sum;
    let tail_ratio = (1.0 - p_sum).max(0.0) / unlisted.max(f64::from(f32::MIN_POSITIVE));
    let tail = unlisted > LISTED_WHOLE * q.len().max(1) as f64 && band.holds(tail_ratio);
    let mut correction = 0.0f64;
    for j in 0..q.len() {
        let head = if p[j] >= low * q[j] && p[j] <= high * q[j] { p[j] } else { 0.0 };
        let residual = head - if tail { p[j] } else { 0.0 };
        coefs[j] = if q[j] > 0.0 { (-advantage * residual) as f32 } else { 0.0 };
        if q[j] > 0.0 {
            correction += residual * f64::from(trainer_top[j]);
        }
    }
    CentredTerm { weight: (advantage * f) as f32, loss: (-advantage * (f * f64::from(new_lp) - correction)) as f32, masked, sampler_mass: q_sum as f32, tail }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inside the band the token keeps its ratio and every listed id its probability; at the edges (inclusive) too;
    /// past them the token is masked and an id drops out of the head.
    #[test]
    fn the_band_masks_the_token_and_the_listed_ids_at_its_edges() {
        let band = CentringBand { low: 0.5, high: 2.0 };
        let ln = |x: f64| x.ln() as f32;
        // Two listed ids holding 0.9 of the sampler; the trainer has moved the first by 2.0 (edge) and the second by 0.4.
        let sampler = [ln(0.5), ln(0.4)];
        let trainer = [ln(1.0 - 1e-7), ln(0.16)];
        for (r, masked) in [(1.0f64, false), (0.5001, false), (1.9999, false), (0.49, true), (2.01, true)] {
            let mut c = [0f32; 2];
            let t = centred_term(ln(0.5), ln(0.5 * r), 1.0, band, &sampler, &trainer, &mut c);
            assert_eq!(t.masked, masked, "ratio {r}");
            assert!((f64::from(t.weight) - if masked { 0.0 } else { r }).abs() < 1e-5, "ratio {r}: weight {}", t.weight);
            // id 0 at ratio ~2.0 is in the head, id 1 at 0.4 is not; the tail (0 / 0.1) is out of the band.
            assert!(!t.tail && (f64::from(c[0]) + 1.0).abs() < 1e-5 && c[1] == 0.0, "{c:?}");
            assert!((t.sampler_mass - 0.9).abs() < 1e-6);
        }
    }

    /// The bucket of the unlisted mass is centred when its ratio is in the band: each listed id then carries
    /// `head - p`, and an unused slot carries nothing.
    #[test]
    fn the_tail_bucket_is_centred_when_its_ratio_is_in_the_band() {
        let ln = |x: f64| x.ln() as f32;
        let sampler = [ln(0.6), f32::NAN];
        let trainer = [ln(0.5), -1.0];
        let mut c = [7f32; 2];
        // Tail ratio (1 - 0.5) / (1 - 0.6) = 1.25: in the band.
        let t = centred_term(ln(0.6), ln(0.5), 2.0, CentringBand::DEFAULT, &sampler, &trainer, &mut c);
        assert!(t.tail);
        assert!(c[0].abs() < 1e-6, "head 0.5 less tail 0.5: {c:?}");
        assert_eq!(c[1], 0.0, "an unused slot");
        let narrow = CentringBand { low: 0.9, high: 1.1 };
        let t = centred_term(ln(0.6), ln(0.5), 2.0, narrow, &sampler, &trainer, &mut c);
        assert!(!t.tail && c[0] == 0.0, "ratio 0.83 is out of [0.9, 1.1]: no head, no tail: {c:?}");
    }

    /// A sampler that listed its whole distribution leaves no bucket, whichever way float32 rounding moves the listed
    /// mass around one: every listed id keeps its head and nothing is subtracted for an unlisted rest (before, a ratio of
    /// two rounding errors decided it, and TRL's float32 arithmetic decided it the other way).
    #[test]
    fn a_whole_listed_distribution_has_no_tail_whatever_the_rounding() {
        let ln = |x: f64| x.ln() as f32;
        for (q, p) in [([0.5, 0.3, 0.2], [0.6, 0.25, 0.15]), ([0.7, 0.2, 0.1], [0.65, 0.25, 0.1]), ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [0.3, 0.3, 0.4])] {
            let (sampler, trainer) = (q.map(ln), p.map(ln));
            let mut c = [0f32; 3];
            let t = centred_term(sampler[0], trainer[0], 2.0, CentringBand::DEFAULT, &sampler, &trainer, &mut c);
            assert!(!t.tail, "{q:?}: a listed whole has no tail");
            for j in 0..3 {
                assert!((f64::from(c[j]) + 2.0 * p[j]).abs() < 1e-6, "{q:?}: slot {j} keeps its head: {c:?}");
            }
        }
    }

    #[test]
    fn a_band_without_meaning_is_refused() {
        for (low, high) in [(1.0, 2.0), (0.5, 1.0), (-0.1, 2.0), (0.5, f32::INFINITY)] {
            assert!(CentringBand { low, high }.validate().is_err(), "[{low}, {high}]");
        }
        assert!(CentringBand::DEFAULT.validate().is_ok());
    }
}
