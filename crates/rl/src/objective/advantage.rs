// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Group-relative advantages for GRPO: the reward of each completion in a
//! group, turned into the scalar every one of its tokens is pushed with.
//!
//! Two normalisations are offered because they answer different questions.
//! [`AdvantageNorm::ZScore`] divides by the group's standard deviation, which
//! makes every group count equally - and makes a two-member group's advantages
//! exactly +1 and -1 whatever the reward gap was, so a tiny preference weighs as
//! much as a decisive one. [`AdvantageNorm::MeanCentered`] only subtracts the
//! group mean, so the size of the reward gap survives into the gradient.
//!
//! Swedish Embedded AB implements solutions for reward-driven fine-tuning that
//! learns from the strength of the signal, not only its sign, for its clients.
//! If your team needs expertise in reinforcement learning for language models
//! you can procure our services by sending an email to info@swedishembedded.com.

/// A group whose rewards differ by less than this carries no comparison.
const NO_VARIANCE: f32 = 1e-6;

/// How a group's rewards are turned into advantages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdvantageNorm {
    /// `(r - mean) / (std + 1e-4)`: unit-scale advantages, every informative
    /// group weighs the same. The legacy behaviour.
    ZScore,
    /// `r - mean`: the advantage keeps the reward's own scale, so a group
    /// that separated its members by 0.1 pushes ten times less than one that
    /// separated them by 1.0.
    #[default]
    MeanCentered,
}

/// Advantages of one group's `rewards` under `norm`.
///
/// A group of one is the rejection-sampling/STaR degenerate case: there is
/// nothing to compare the reward with, so its advantage is `1.0` (train on the
/// kept completion with uniform weight). A group of two or more whose rewards
/// are all equal teaches nothing and gets `0.0` for every member under either
/// normalisation.
pub fn group_advantages_with(norm: AdvantageNorm, rewards: &[f32]) -> Vec<f32> {
    if rewards.len() <= 1 {
        return vec![1.0; rewards.len()];
    }
    let g = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / g;
    let var = rewards.iter().map(|&r| (r - mean) * (r - mean)).sum::<f32>() / g;
    let std = var.sqrt();
    if std < NO_VARIANCE {
        return vec![0.0; rewards.len()];
    }
    match norm {
        AdvantageNorm::ZScore => rewards.iter().map(|&r| (r - mean) / (std + 1e-4)).collect(),
        AdvantageNorm::MeanCentered => rewards.iter().map(|&r| r - mean).collect(),
    }
}

/// [`group_advantages_with`] under [`AdvantageNorm::ZScore`], the form the
/// gradient checks and older callers use.
pub fn group_advantages(rewards: &[f32]) -> Vec<f32> {
    group_advantages_with(AdvantageNorm::ZScore, rewards)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 3e-3, "{a:?} vs {b:?}");
        }
    }

    /// R2: a two-member group is the case the normalisations disagree on. The
    /// z-score is +-1 for a reward gap of 0.1 and of 1.0 alike; mean-centring
    /// keeps the gap.
    #[test]
    fn a_pair_keeps_its_reward_gap_only_when_mean_centred() {
        close(&group_advantages_with(AdvantageNorm::ZScore, &[0.6, 0.5]), &[1.0, -1.0]);
        close(&group_advantages_with(AdvantageNorm::ZScore, &[1.0, 0.0]), &[1.0, -1.0]);
        close(&group_advantages_with(AdvantageNorm::MeanCentered, &[0.6, 0.5]), &[0.05, -0.05]);
        close(&group_advantages_with(AdvantageNorm::MeanCentered, &[1.0, 0.0]), &[0.5, -0.5]);
    }

    /// R2: a larger group, hand-computed. Mean 0.5; population std of
    /// [1, 0, 0, 1] is 0.5, so the z-scores are +-1 and the centred values +-0.5.
    #[test]
    fn a_larger_group_matches_the_hand_computed_values() {
        close(&group_advantages_with(AdvantageNorm::ZScore, &[1.0, 0.0, 0.0, 1.0]), &[1.0, -1.0, -1.0, 1.0]);
        close(&group_advantages_with(AdvantageNorm::MeanCentered, &[1.0, 0.0, 0.0, 1.0]), &[0.5, -0.5, -0.5, 0.5]);
        // Unequal: rewards [3, 0, 0] have mean 1, centred [2, -1, -1].
        close(&group_advantages_with(AdvantageNorm::MeanCentered, &[3.0, 0.0, 0.0]), &[2.0, -1.0, -1.0]);
    }

    #[test]
    fn centred_advantages_sum_to_zero_and_a_flat_group_is_dropped_under_both() {
        for norm in [AdvantageNorm::ZScore, AdvantageNorm::MeanCentered] {
            let adv = group_advantages_with(norm, &[0.2, 0.9, 0.4, 0.7, 0.1]);
            assert!(adv.iter().sum::<f32>().abs() < 1e-4, "{norm:?}: {adv:?}");
            assert_eq!(group_advantages_with(norm, &[0.7, 0.7, 0.7]), vec![0.0; 3]);
            assert_eq!(group_advantages_with(norm, &[1.0]), vec![1.0], "a group of one is the STaR case");
            assert!(group_advantages_with(norm, &[]).is_empty());
        }
    }

    #[test]
    fn the_default_keeps_the_reward_scale() {
        assert_eq!(AdvantageNorm::default(), AdvantageNorm::MeanCentered);
    }
}
