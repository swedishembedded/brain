// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-token term of GRPO's clipped surrogate and its analytic gradient
//! weight.
//!
//! A model's ordinary cross-entropy backward writes, per position,
//! `d_logits = (softmax(z) - onehot(target)) / count`, and
//! [`model::training::TokenObjectives::set_loss_weights`] scales that row-wise
//! before the head's backward reads it. GRPO's clipped surrogate for one token
//! is `min(r*A, clip(r, 1-eps_low, 1+eps_high)*A)` with
//! `r = exp(new_lp - old_lp)`, the importance ratio between the current
//! policy's log-probability and the behaviour policy's, and `A` the token's
//! group-broadcast advantage. Differentiating it with respect to the logits
//! gives exactly `weight * d_logits` for `weight = A*r` on an unclipped token
//! and `0` on a clipped one: `r` is not stop-gradient here, its own derivative
//! `dr/dz = r * dlogp/dz` is what produces the `A*r` factor, so the ordinary
//! cross-entropy backward differentiates the real clipped objective by using
//! this weight. The optional k3 KL-to-reference term (`e^u - u - 1`,
//! `u = ref_lp - new_lp`) adds `kl_beta*(e^u - 1)` to the same weight by an
//! identical derivation.
//!
//! Swedish Embedded AB implements solutions for reward-driven training that
//! reduces to one gradient-checked weighted cross-entropy kernel, for its
//! clients. If your team needs expertise in reinforcement learning from
//! verifiable rewards you can procure our services by sending an email to
//! info@swedishembedded.com.

/// The largest exponent evaluated exactly: `exp(20)` is about 4.9e8, far
/// outside any trust region and still finite in f32. Past it the exponential
/// continues along its tangent at the cap ([`bounded_exp`]), so a policy that
/// has drifted (or an `old_lp` that was never the behaviour policy's) cannot
/// make the loss or the weight overflow, and the weight stays the derivative
/// of the loss that is reported.
pub const MAX_LOG_RATIO: f32 = 20.0;

/// `exp(x)` and its slope, continued linearly beyond [`MAX_LOG_RATIO`]:
/// `exp(c) * (1 + x - c)` there, whose value and slope match the exponential at
/// the cap. The objective built on it is C1, so the loss and the analytic
/// gradient weight agree everywhere, including past the cap. Evaluated in f64
/// so the extension does not round at the cap.
fn bounded_exp(x: f64) -> (f64, f64) {
    let cap = f64::from(MAX_LOG_RATIO);
    if x <= cap {
        let e = x.exp();
        (e, e)
    } else {
        let at_cap = cap.exp();
        (at_cap * (1.0 + x - cap), at_cap)
    }
}

/// The clip band of the importance ratio: `[1 - low, 1 + high]`. The upper
/// edge may be wider than the lower so that a low-probability token that turns
/// out well can still gain probability mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipBand {
    pub low: f32,
    pub high: f32,
}

impl ClipBand {
    /// The band `[1 - eps, 1 + eps]`.
    pub fn symmetric(eps: f32) -> ClipBand {
        ClipBand { low: eps, high: eps }
    }
}

/// One token's [`model::Batch::LmWeighted`] weight, plus the host-computed loss
/// term it is the analytic gradient of (`d(loss)/dz = weight *
/// (softmax(z) - onehot(target))`) - the pair the gradcheck gate's
/// `loss()`/`backward()` must agree on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenTerm {
    pub weight: f32,
    pub loss: f32,
    /// The policy term sat outside the clip band on the side its advantage pushes toward, so it contributes no
    /// gradient (`false` for a token without an advantage).
    pub clipped: bool,
}

/// [`token_term_banded`] with a symmetric band `[1 - clip_eps, 1 + clip_eps]`.
pub fn token_term(old_lp: f32, new_lp: f32, advantage: f32, clip_eps: f32, ref_lp: Option<f32>, kl_beta: f32) -> TokenTerm {
    token_term_banded(old_lp, new_lp, advantage, ClipBand::symmetric(clip_eps), ref_lp, kl_beta)
}

/// `advantage == 0.0` means this token contributes nothing to the policy term
/// (a dropped zero-variance group, or a completion whose reward landed exactly
/// on its group's mean) - the KL term (when `kl_beta > 0.0`) still applies
/// independently, since it regularises every token toward the reference
/// regardless of that token's own reward.
///
/// Clip rule, derived from `min(r*A, clip(r)*A)`: the raw (differentiable)
/// term wins whenever `r*A <= clip(r)*A`; the derivative is then exactly
/// `A*r`. When the clipped term is strictly smaller, the surrogate is locally
/// constant in `r` (the clamp has saturated), so the weight is exactly `0.0`.
///
/// KL penalty direction: `kl_beta * k3` is added to the loss with the sample
/// held fixed, so its expected gradient under `pi` samples is the gradient of
/// `KL(ref || pi)` (see [`k3_kl`]), even though the reported value estimates
/// `KL(pi || ref)`. Descending `KL(pi || ref)` itself would need the penalty
/// folded into the reward (`-kl_beta * (new_lp - ref_lp)` per token) so the
/// policy gradient differentiates through the sampling distribution.
///
/// The ratio `exp(new_lp - old_lp)` is evaluated by [`bounded_exp`], and so is the
/// k3 exponent: past [`MAX_LOG_RATIO`] both continue along their tangent, and `weight`
/// is the derivative of the `loss` reported with it.
pub fn token_term_banded(old_lp: f32, new_lp: f32, advantage: f32, band: ClipBand, ref_lp: Option<f32>, kl_beta: f32) -> TokenTerm {
    token_term_weighted(old_lp, new_lp, advantage, band, ref_lp, kl_beta, 1.0)
}

/// [`token_term_banded`] with the policy term scaled by `policy_weight`, a constant the parameters do not move (the
/// truncated importance-sampling weight of a token: `min(pi_trainer / pi_rollout, cap)`, or `0.0` for a token the
/// mismatch mask drops). The scale multiplies the clipped surrogate and its gradient weight alike, so the weight stays
/// the derivative of the reported loss; the KL term is not scaled, since it regularises the policy at every sampled
/// token whatever engine sampled it.
pub fn token_term_weighted(old_lp: f32, new_lp: f32, advantage: f32, band: ClipBand, ref_lp: Option<f32>, kl_beta: f32, policy_weight: f32) -> TokenTerm {
    let policy_weight = f64::from(policy_weight);
    let (ratio, ratio_slope) = bounded_exp(f64::from(new_lp) - f64::from(old_lp));
    let advantage = f64::from(advantage);
    let mut weight = 0.0f64;
    let mut loss = 0.0f64;
    let mut clipped_out = false;

    if advantage != 0.0 {
        let unclipped = ratio * advantage;
        let clipped = ratio.clamp(1.0 - f64::from(band.low), 1.0 + f64::from(band.high)) * advantage;
        let surrogate = unclipped.min(clipped);
        loss += -surrogate * policy_weight;
        if unclipped <= clipped {
            weight += advantage * ratio_slope * policy_weight;
        } else {
            clipped_out = true;
        }
    }

    if kl_beta > 0.0 {
        let ref_lp = ref_lp.expect("token_term: kl_beta > 0.0 requires a reference logprob");
        let (k3, slope) = k3_with_slope(ref_lp, new_lp);
        weight += f64::from(kl_beta) * (slope - 1.0);
        loss += f64::from(kl_beta) * k3;
    }

    TokenTerm { weight: weight as f32, loss: loss as f32, clipped: clipped_out }
}

/// The k3 estimator of `KL(policy || reference)` at one sampled token,
/// `e^u - u - 1` with `u = ref_lp - new_lp`; non-negative, zero iff the two
/// log-probabilities agree. Its *value* estimates `KL(pi || ref)` for samples
/// drawn from `pi`, but descending it with the sample held fixed does not
/// descend that KL: `grad k3 = (1 - ref/pi) grad ln pi`, so
/// `E_pi[grad k3] = sum_a (pi_a - ref_a) grad ln pi_a = grad KL(ref || pi)`
/// (using `sum_a grad pi_a = 0`). The penalty therefore pulls `pi` toward
/// covering `ref` (forward KL), not toward seeking one of its modes. Its gradient weight is `e^u - 1`: with
/// `d(new_lp)/dz = onehot - softmax` the KL gradient is `(e^u - 1)` more weight
/// on the ordinary cross-entropy direction. The exponential is continued
/// linearly past [`MAX_LOG_RATIO`] ([`bounded_exp`]) so a reference far more
/// confident than the policy cannot overflow, and the weight stays this
/// function's derivative.
pub fn k3_kl(ref_lp: f32, new_lp: f32) -> f32 {
    k3_with_slope(ref_lp, new_lp).0 as f32
}

/// The k3 value and the derivative of `e^u` in `u` (the gradient weight is that minus one).
fn k3_with_slope(ref_lp: f32, new_lp: f32) -> (f64, f64) {
    let u = f64::from(ref_lp) - f64::from(new_lp);
    let (eu, slope) = bounded_exp(u);
    (eu - u - 1.0, slope)
}

/// True if `ratio` sits within `margin` of either clip boundary (`1 -
/// clip_eps` or `1 + clip_eps`) - [`token_term`]'s clip indicator is
/// piecewise-constant there, so an arbitrarily small parameter perturbation
/// can flip which term the `min` picks. A finite-difference gate must keep its
/// sampled base-point ratios clear of this margin rather than loosen tolerance
/// to paper over the resulting mismatch.
pub fn near_clip_boundary(ratio: f32, clip_eps: f32, margin: f32) -> bool {
    (ratio - (1.0 - clip_eps)).abs() < margin || (ratio - (1.0 + clip_eps)).abs() < margin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_term_unclipped_positive_advantage_is_advantage_times_ratio() {
        // ratio = exp(0) = 1, inside [0.8, 1.2] - not clipped.
        let t = token_term(0.0, 0.0, 2.0, 0.2, None, 0.0);
        assert!((t.weight - 2.0).abs() < 1e-6, "{t:?}");
        assert!((t.loss - -2.0).abs() < 1e-6, "{t:?}");
    }

    #[test]
    fn token_term_clips_a_large_ratio_with_positive_advantage_to_zero_weight() {
        // ratio = 1.5 > 1 + eps(0.2) = 1.2, advantage > 0: the classic PPO
        // "ran too far in the rewarded direction" clip.
        let t = token_term(0.0, 1.5_f32.ln(), 1.0, 0.2, None, 0.0);
        assert_eq!(t.weight, 0.0, "{t:?}");
        assert!((t.loss - -1.2).abs() < 1e-5, "{t:?}");
    }

    #[test]
    fn token_term_does_not_clip_a_large_ratio_with_negative_advantage() {
        // Same ratio, negative advantage: pushing further into an already
        // over-large ratio is exactly what a negative advantage should keep
        // penalising.
        let t = token_term(0.0, 1.5_f32.ln(), -1.0, 0.2, None, 0.0);
        assert!((t.weight - -1.5).abs() < 1e-5, "{t:?}");
        assert!((t.loss - 1.5).abs() < 1e-5, "{t:?}");
    }

    #[test]
    fn token_term_clips_a_small_ratio_with_negative_advantage_to_zero_weight() {
        // ratio = 0.5 < 1 - eps(0.2) = 0.8, advantage < 0.
        let t = token_term(0.0, 0.5_f32.ln(), -1.0, 0.2, None, 0.0);
        assert_eq!(t.weight, 0.0, "{t:?}");
        // surrogate = min(-0.5, -0.8) = -0.8, loss = -surrogate = 0.8.
        assert!((t.loss - 0.8).abs() < 1e-5, "{t:?}");
    }

    /// The decoupled band: a ratio of 1.25 is clipped by the symmetric 0.2 band
    /// and still inside the 0.2/0.28 band, which keeps pushing the token.
    #[test]
    fn a_wider_upper_edge_keeps_a_rewarded_token_moving_past_the_symmetric_clip() {
        let (old, new) = (0.0, 1.25_f32.ln());
        assert_eq!(token_term(old, new, 1.0, 0.2, None, 0.0).weight, 0.0);
        let wide = token_term_banded(old, new, 1.0, ClipBand { low: 0.2, high: 0.28 }, None, 0.0);
        assert!((wide.weight - 1.25).abs() < 1e-5, "{wide:?}");
        // Past the upper edge it clips again.
        assert_eq!(token_term_banded(old, 1.3_f32.ln(), 1.0, ClipBand { low: 0.2, high: 0.28 }, None, 0.0).weight, 0.0);
    }

    /// R7: a log-ratio of 100 must not overflow the weight; the exponent is capped.
    #[test]
    fn an_enormous_log_ratio_gives_a_finite_weight() {
        let t = token_term(0.0, 100.0, -1.0, 0.2, None, 0.0);
        assert!(t.weight.is_finite() && t.loss.is_finite(), "{t:?}");
        assert!((t.weight + MAX_LOG_RATIO.exp()).abs() / MAX_LOG_RATIO.exp() < 1e-5, "{t:?}");
        let kl = token_term(0.0, -100.0, 0.0, 0.2, Some(0.0), 0.5);
        assert!(kl.weight.is_finite() && kl.loss.is_finite(), "{kl:?}");
    }

    #[test]
    fn token_term_kl_applies_even_at_zero_advantage() {
        // advantage == 0.0 must still carry the KL regulariser - it does not
        // depend on the sample's own reward.
        let ref_lp: f32 = -0.5;
        let new_lp: f32 = -0.2;
        let u = ref_lp - new_lp;
        let eu = u.exp();
        let t = token_term(-0.3, new_lp, 0.0, 0.2, Some(ref_lp), 0.5);
        assert!((t.weight - 0.5 * (eu - 1.0)).abs() < 1e-6, "{t:?}");
        assert!((t.loss - 0.5 * (eu - u - 1.0)).abs() < 1e-6, "{t:?}");
    }

    /// R9: the k3 estimator against its definition, and its two defining
    /// properties - zero when the policies agree, never negative.
    #[test]
    fn k3_matches_its_closed_form_and_is_a_non_negative_divergence() {
        assert_eq!(k3_kl(-1.3, -1.3), 0.0);
        let (ref_lp, new_lp): (f32, f32) = (-2.0, -0.7);
        let u = ref_lp - new_lp;
        assert!((k3_kl(ref_lp, new_lp) - (u.exp() - u - 1.0)).abs() < 1e-6);
        for (r, n) in [(-0.1, -3.0), (-3.0, -0.1), (-1.0, -1.01), (-6.0, -0.01)] {
            assert!(k3_kl(r, n) >= 0.0, "{r} {n}");
        }
    }

    #[test]
    #[should_panic(expected = "kl_beta > 0.0 requires")]
    fn token_term_panics_without_a_reference_logprob_when_kl_is_enabled() {
        let _ = token_term(0.0, 0.0, 1.0, 0.2, None, 0.5);
    }

    #[test]
    fn near_clip_boundary_flags_only_the_margin() {
        assert!(near_clip_boundary(1.19, 0.2, 0.05));
        assert!(near_clip_boundary(0.81, 0.2, 0.05));
        assert!(!near_clip_boundary(1.6, 0.2, 0.05));
        assert!(!near_clip_boundary(0.5, 0.2, 0.05));
    }

    // ---- T005: the loss and the gradient weight are one function ----------------------------

    /// FP64 scalar reference of the documented objective: the importance ratio and the k3
    /// exponent are exact up to [`MAX_LOG_RATIO`] and continue linearly (value and slope
    /// match at the cap) beyond it, so the objective is C1 and bounded in slope.
    fn reference(old_lp: f64, new_lp: f64, advantage: f64, band: ClipBand, ref_lp: Option<f64>, kl_beta: f64) -> (f64, f64) {
        let cap = f64::from(MAX_LOG_RATIO);
        let ext = |x: f64| if x <= cap { (x.exp(), x.exp()) } else { (cap.exp() * (1.0 + x - cap), cap.exp()) };
        let (mut weight, mut loss) = (0.0, 0.0);
        if advantage != 0.0 {
            let (r, slope) = ext(new_lp - old_lp);
            let (unclipped, clipped) = (r * advantage, r.clamp(1.0 - f64::from(band.low), 1.0 + f64::from(band.high)) * advantage);
            loss -= unclipped.min(clipped);
            if unclipped <= clipped {
                weight += advantage * slope;
            }
        }
        if let (true, Some(ref_lp)) = (kl_beta > 0.0, ref_lp) {
            let u = ref_lp - new_lp;
            let (eu, slope) = ext(u);
            weight += kl_beta * (slope - 1.0);
            loss += kl_beta * (eu - u - 1.0);
        }
        (weight, loss)
    }

    /// The importance-sampling weight scales the policy term only: the surrogate, its gradient weight and the clip
    /// flag follow the unweighted ones, a zero weight leaves the KL term alone, and the weight stays the derivative
    /// of the weighted loss.
    #[test]
    fn a_policy_weight_scales_the_surrogate_and_leaves_the_kl_term_alone() {
        let band = ClipBand::symmetric(0.2);
        let (old_lp, new_lp, advantage, ref_lp, beta) = (-2.0f32, -1.9f32, 1.5f32, Some(-2.2f32), 0.04f32);
        let plain = token_term_weighted(old_lp, new_lp, advantage, band, None, 0.0, 1.0);
        let half = token_term_weighted(old_lp, new_lp, advantage, band, None, 0.0, 0.5);
        assert!((half.weight - 0.5 * plain.weight).abs() < 1e-6 && (half.loss - 0.5 * plain.loss).abs() < 1e-6, "{plain:?} {half:?}");
        let kl_only = token_term_weighted(old_lp, new_lp, advantage, band, ref_lp, beta, 0.0);
        let kl_ref = token_term_banded(old_lp, new_lp, 0.0, band, ref_lp, beta);
        assert!((kl_only.weight - kl_ref.weight).abs() < 1e-6 && (kl_only.loss - kl_ref.loss).abs() < 1e-6, "{kl_only:?} {kl_ref:?}");
        let h = 1e-2f32;
        let (lo, hi) = (token_term_weighted(old_lp, new_lp - h, advantage, band, ref_lp, beta, 0.7), token_term_weighted(old_lp, new_lp + h, advantage, band, ref_lp, beta, 0.7));
        let t = token_term_weighted(old_lp, new_lp, advantage, band, ref_lp, beta, 0.7);
        let fd = -f64::from(hi.loss - lo.loss) / f64::from(2.0 * h);
        assert!((fd - f64::from(t.weight)).abs() <= 1e-2 * fd.abs() + 1e-3, "finite difference {fd}, weight {}", t.weight);
    }

    /// Log-probability gaps that exercise every regime: ordinary, past the cap on both sides,
    /// f32-subnormal ratios, and one step either side of each clip boundary.
    fn gaps() -> Vec<f32> {
        let b = ClipBand::symmetric(0.2);
        let edge = |r: f32| r.ln();
        vec![0.0, 0.3, -0.7, 5.0, -5.0, 19.9, 20.0, 20.1, 25.0, 60.0, 400.0, -60.0, -100.0, -400.0, edge(1.0 + b.high) - 1e-2, edge(1.0 + b.high) + 1e-2, edge(1.0 - b.low) - 1e-2, edge(1.0 - b.low) + 1e-2]
    }

    #[test]
    fn the_term_matches_the_fp64_reference_at_normal_extreme_and_boundary_inputs() {
        let band = ClipBand::symmetric(0.2);
        for gap in gaps() {
            for advantage in [1.5f32, -1.5, 0.0] {
                for (ref_lp, beta) in [(None, 0.0f32), (Some(-1.0f32), 0.04)] {
                    let (old_lp, new_lp) = (-2.0f32, -2.0 + gap);
                    let t = token_term_banded(old_lp, new_lp, advantage, band, ref_lp, beta);
                    let (w, l) = reference(f64::from(old_lp), f64::from(new_lp), f64::from(advantage), band, ref_lp.map(f64::from), f64::from(beta));
                    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
                    assert!(t.weight.is_finite() && t.loss.is_finite(), "gap {gap} A {advantage}: {t:?}");
                    assert!(rel(f64::from(t.weight), w) < 1e-4 && rel(f64::from(t.loss), l) < 1e-4, "gap {gap} A {advantage} kl {ref_lp:?}: got {t:?}, reference weight {w} loss {l}");
                }
            }
        }
    }

    /// `d(loss)/d(new_lp) = -weight`: the weight is the gradient of the loss the forward
    /// reports, beyond the cap as much as inside it. Central differences on the f32 term,
    /// stepping only within one clip regime.
    #[test]
    fn the_weight_is_the_derivative_of_the_loss_including_beyond_the_cap() {
        let band = ClipBand::symmetric(0.2);
        for gap in gaps() {
            for advantage in [1.5f32, -1.5] {
                for (ref_lp, beta) in [(None, 0.0f32), (Some(-1.0f32), 0.04)] {
                    let (old_lp, new_lp) = (-2.0f32, -2.0 + gap);
                    let t = token_term_banded(old_lp, new_lp, advantage, band, ref_lp, beta);
                    let h = 1e-2f32;
                    let lo = token_term_banded(old_lp, new_lp - h, advantage, band, ref_lp, beta);
                    let hi = token_term_banded(old_lp, new_lp + h, advantage, band, ref_lp, beta);
                    if lo.clipped != hi.clipped || lo.clipped != t.clipped {
                        continue; // the step straddles a clip kink, where no derivative exists
                    }
                    let fd = -f64::from(hi.loss - lo.loss) / f64::from(2.0 * h);
                    let tol = 1e-2 * fd.abs().max(f64::from(t.weight).abs()) + 1e-3;
                    assert!((fd - f64::from(t.weight)).abs() <= tol, "gap {gap} A {advantage} kl {ref_lp:?}: finite difference {fd}, weight {}", t.weight);
                }
            }
        }
    }

    /// Which divergence the k3 penalty descends, by exact enumeration. With samples drawn from `pi` and held
    /// fixed, `E_pi[grad_z kl_beta * k3] = kl_beta * (pi - ref)` for logits `z` (`pi = softmax z`), which is the
    /// gradient of `KL(ref || pi)`, not of `KL(pi || ref)`, `kl_beta * pi_j * (ln(pi_j / ref_j) - KL(pi || ref))`.
    /// A token's logit gradient is `weight * (softmax - onehot)` ([`TokenTerm`]).
    #[test]
    fn k3_penalty_expected_gradient_is_that_of_forward_kl_ref_to_policy() {
        let beta = 0.7f32;
        let pi = [0.5f64, 0.3, 0.15, 0.05];
        let reference = [0.1f64, 0.2, 0.3, 0.4];
        // The kernel sees f32 log-probabilities; the analytic side uses the same rounded values.
        let pi_lp: Vec<f32> = pi.iter().map(|p| p.ln() as f32).collect();
        let ref_lp: Vec<f32> = reference.iter().map(|p| p.ln() as f32).collect();
        let pi: Vec<f64> = pi_lp.iter().map(|l| f64::from(*l).exp()).collect();
        let reference: Vec<f64> = ref_lp.iter().map(|l| f64::from(*l).exp()).collect();

        let mut expected = vec![0.0f64; pi.len()];
        for a in 0..pi.len() {
            let t = token_term(0.0, pi_lp[a], 0.0, 0.2, Some(ref_lp[a]), beta);
            for (j, g) in expected.iter_mut().enumerate() {
                *g += pi[a] * f64::from(t.weight) * (pi[j] - f64::from(a == j));
            }
        }

        let forward: Vec<f64> = (0..pi.len()).map(|j| f64::from(beta) * (pi[j] - reference[j])).collect();
        let kl_reverse: f64 = pi.iter().zip(&reference).map(|(p, r)| p * (p / r).ln()).sum();
        let reverse: Vec<f64> = (0..pi.len()).map(|j| f64::from(beta) * pi[j] * ((pi[j] / reference[j]).ln() - kl_reverse)).collect();

        // Error budget: each weight is rounded once to f32 (relative 2^-24) and |weight| <= beta * max(ref/pi) < 10 here;
        // the sum has 4 terms of magnitude <= 1, so 4 * 10 * beta * 2^-24 < 2e-6, plus the f32 weight's own exp error.
        let bound = 1e-5;
        for j in 0..pi.len() {
            assert!((expected[j] - forward[j]).abs() <= bound, "z{j}: expected {} vs grad KL(ref||pi) {}", expected[j], forward[j]);
        }
        let gap = expected.iter().zip(&reverse).map(|(e, r)| (e - r).abs()).fold(0.0, f64::max);
        assert!(gap > 0.05, "the penalty must differ from grad KL(pi||ref); gap {gap}");
    }
}
