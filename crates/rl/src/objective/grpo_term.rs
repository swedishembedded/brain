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

/// The largest exponent of the importance ratio: `exp(20)` is about 4.9e8, far
/// outside any trust region and still finite in f32, so a policy that has
/// drifted (or an `old_lp` that was never the behaviour policy's) cannot make
/// the weight overflow to infinity.
pub const MAX_LOG_RATIO: f32 = 20.0;

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
/// The ratio is computed from the log-probability difference, capped at
/// [`MAX_LOG_RATIO`] before it is exponentiated.
pub fn token_term_banded(old_lp: f32, new_lp: f32, advantage: f32, band: ClipBand, ref_lp: Option<f32>, kl_beta: f32) -> TokenTerm {
    let ratio = (new_lp - old_lp).min(MAX_LOG_RATIO).exp();
    let mut weight = 0.0f32;
    let mut loss = 0.0f32;

    if advantage != 0.0 {
        let unclipped = ratio * advantage;
        let clipped = ratio.clamp(1.0 - band.low, 1.0 + band.high) * advantage;
        let surrogate = unclipped.min(clipped);
        loss += -surrogate;
        if unclipped <= clipped {
            weight += advantage * ratio;
        }
    }

    if kl_beta > 0.0 {
        let ref_lp = ref_lp.expect("token_term: kl_beta > 0.0 requires a reference logprob");
        let (eu, u) = k3_exponent(ref_lp, new_lp);
        weight += kl_beta * (eu - 1.0);
        loss += kl_beta * (eu - u - 1.0);
    }

    TokenTerm { weight, loss }
}

/// The k3 estimator of `KL(policy || reference)` at one sampled token,
/// `e^u - u - 1` with `u = ref_lp - new_lp`; non-negative, zero iff the two
/// log-probabilities agree. Its gradient weight is `e^u - 1`: with
/// `d(new_lp)/dz = onehot - softmax` the KL gradient is `(e^u - 1)` more weight
/// on the ordinary cross-entropy direction.
pub fn k3_kl(ref_lp: f32, new_lp: f32) -> f32 {
    let (eu, u) = k3_exponent(ref_lp, new_lp);
    eu - u - 1.0
}

/// `(e^u, u)` for `u = ref_lp - new_lp`, with `u` capped at [`MAX_LOG_RATIO`]
/// so that a reference far more confident than the policy cannot overflow.
fn k3_exponent(ref_lp: f32, new_lp: f32) -> (f32, f32) {
    let u = (ref_lp - new_lp).min(MAX_LOG_RATIO);
    (u.exp(), u)
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
}
