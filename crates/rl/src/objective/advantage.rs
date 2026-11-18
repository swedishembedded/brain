// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Group-relative advantages: the reward of each completion in a group, turned
//! into the scalar every one of its tokens is pushed with.
//!
//! Three normalisations are offered because they answer different questions.
//! [`AdvantageNorm::ZScore`] divides by the group's standard deviation, which
//! makes every group count equally - and makes a two-member group's advantages
//! exactly +1 and -1 whatever the reward gap was, so a tiny preference weighs as
//! much as a decisive one. [`AdvantageNorm::MeanCentered`] only subtracts the
//! group mean, so the size of the reward gap survives into the gradient.
//! [`AdvantageNorm::LeaveOneOut`] (RLOO) subtracts the mean of the *other*
//! members, a baseline that does not depend on the member's own reward, so the
//! estimator is unbiased; at a fixed group size `G` it is mean-centring scaled
//! by `G / (G - 1)`, and groups of different sizes are weighed differently.
//!
//! The arithmetic is in `f64`: rewards arrive as `f64` on the wire, and an
//! advantage is narrowed only where a token is weighted with it.
//!
//! Swedish Embedded AB implements solutions for reward-driven fine-tuning that
//! learns from the strength of the signal, not only its sign, for its clients.
//! If your team needs expertise in reinforcement learning for language models
//! you can procure our services by sending an email to info@swedishembedded.com.

/// A group whose rewards differ by less than this carries no comparison.
const NO_VARIANCE: f64 = 1e-6;
/// Added to the standard deviation before dividing by it, so a nearly flat group does not blow its advantages up.
const STD_EPSILON: f64 = 1e-4;

/// How a group's rewards are turned into advantages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdvantageNorm {
    /// `(r - mean) / (std + 1e-4)` with the population standard deviation: unit-scale advantages, every informative
    /// group weighs the same (GRPO as published).
    ZScore,
    /// `r - mean`: the advantage keeps the reward's own scale, so a group
    /// that separated its members by 0.1 pushes ten times less than one that
    /// separated them by 1.0.
    #[default]
    MeanCentered,
    /// `r_i - (sum_{j != i} r_j) / (G - 1)`: each member against the mean of the others (RLOO).
    LeaveOneOut,
}

impl AdvantageNorm {
    /// Every normalisation, in the order the wire lists them.
    pub const ALL: [AdvantageNorm; 3] = [AdvantageNorm::MeanCentered, AdvantageNorm::ZScore, AdvantageNorm::LeaveOneOut];

    /// The normalisation's name on the wire (`advantage_norm`).
    pub fn name(self) -> &'static str {
        match self {
            AdvantageNorm::MeanCentered => "mean-centred",
            AdvantageNorm::ZScore => "z-score",
            AdvantageNorm::LeaveOneOut => "leave-one-out",
        }
    }

    /// The normalisation a wire name names.
    ///
    /// # Errors
    /// The name is none of [`Self::name`]'s; the error lists them.
    pub fn parse(name: &str) -> Result<AdvantageNorm, String> {
        AdvantageNorm::ALL.into_iter().find(|n| n.name() == name).ok_or_else(|| format!("advantage_norm {name:?} is none of {}", AdvantageNorm::ALL.map(AdvantageNorm::name).join(", ")))
    }
}

/// Advantages of one group's `rewards` under `norm`.
///
/// A group of one is the rejection-sampling/STaR degenerate case: there is
/// nothing to compare the reward with, so its advantage is `1.0` (train on the
/// kept completion with uniform weight). A caller that holds a group of one to be
/// an error (rollouts collected outside brain) refuses it before asking. A group
/// of two or more whose rewards are all equal teaches nothing and gets `0.0` for
/// every member under every normalisation.
pub fn group_advantages_with(norm: AdvantageNorm, rewards: &[f64]) -> Vec<f64> {
    if rewards.len() <= 1 {
        return vec![1.0; rewards.len()];
    }
    let g = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / g;
    let var = rewards.iter().map(|&r| (r - mean) * (r - mean)).sum::<f64>() / g;
    let std = var.sqrt();
    if std < NO_VARIANCE {
        return vec![0.0; rewards.len()];
    }
    match norm {
        AdvantageNorm::ZScore => rewards.iter().map(|&r| (r - mean) / (std + STD_EPSILON)).collect(),
        AdvantageNorm::MeanCentered => rewards.iter().map(|&r| r - mean).collect(),
        // r_i - (S - r_i) / (G - 1), with S the group's sum: the others' mean without a second pass per member.
        AdvantageNorm::LeaveOneOut => {
            let sum = mean * g;
            rewards.iter().map(|&r| r - (sum - r) / (g - 1.0)).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fp64 oracle: each definition written out member by member, independently of the implementation's
    /// closed forms (population standard deviation; the leave-one-out baseline summed over the others).
    fn oracle(norm: AdvantageNorm, rewards: &[f64]) -> Vec<f64> {
        let g = rewards.len();
        let mean = rewards.iter().sum::<f64>() / g as f64;
        let std = (rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / g as f64).sqrt();
        (0..g)
            .map(|i| match norm {
                AdvantageNorm::MeanCentered => rewards[i] - mean,
                AdvantageNorm::ZScore => (rewards[i] - mean) / (std + 1e-4),
                AdvantageNorm::LeaveOneOut => rewards[i] - (0..g).filter(|&j| j != i).map(|j| rewards[j]).sum::<f64>() / (g - 1) as f64,
            })
            .collect()
    }

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
        }
    }

    /// L1: every normalisation equals its fp64 oracle on groups of several sizes and reward scales.
    #[test]
    fn every_normalisation_matches_its_fp64_oracle() {
        let groups: [&[f64]; 5] = [&[1.0, 0.0], &[0.6, 0.5], &[1.0, 0.0, 0.0, 1.0], &[3.0, 0.0, 0.0], &[0.2, 0.9, 0.4, 0.7, 0.1, -2.5, 1e3]];
        for norm in AdvantageNorm::ALL {
            for rewards in groups {
                close(&group_advantages_with(norm, rewards), &oracle(norm, rewards), 1e-12 * rewards.iter().fold(1.0f64, |m, r| m.max(r.abs())));
            }
        }
    }

    /// L1, hand-computed: a pair is the case z-score and mean-centring disagree on (+-1 whatever the gap), and
    /// leave-one-out is the whole gap, twice the centred value.
    #[test]
    fn a_pair_by_hand() {
        close(&group_advantages_with(AdvantageNorm::ZScore, &[0.6, 0.5]), &[0.05 / (0.05 + 1e-4), -0.05 / (0.05 + 1e-4)], 1e-12);
        close(&group_advantages_with(AdvantageNorm::MeanCentered, &[0.6, 0.5]), &[0.05, -0.05], 1e-12);
        close(&group_advantages_with(AdvantageNorm::LeaveOneOut, &[0.6, 0.5]), &[0.1, -0.1], 1e-12);
        // [3, 0, 0]: mean 1, centred [2, -1, -1]; the others' means are 0, 1.5, 1.5.
        close(&group_advantages_with(AdvantageNorm::LeaveOneOut, &[3.0, 0.0, 0.0]), &[3.0, -1.5, -1.5], 1e-12);
    }

    /// L1: at a fixed group size `G`, leave-one-out is mean-centring scaled by `G / (G - 1)`; across groups of
    /// different sizes the factor differs, so the two weigh a dataset's groups differently.
    #[test]
    fn leave_one_out_is_scaled_mean_centring_at_a_fixed_size_and_not_across_sizes() {
        let mut factors = Vec::new();
        for rewards in [&[1.0, 0.0][..], &[1.0, 0.0, 0.5, 0.25][..], &[0.9, 0.1, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6][..]] {
            let g = rewards.len() as f64;
            let centred = group_advantages_with(AdvantageNorm::MeanCentered, rewards);
            let loo = group_advantages_with(AdvantageNorm::LeaveOneOut, rewards);
            close(&loo, &centred.iter().map(|a| a * g / (g - 1.0)).collect::<Vec<_>>(), 1e-12);
            factors.push(loo[0] / centred[0]);
        }
        close(&factors, &[2.0, 4.0 / 3.0, 8.0 / 7.0], 1e-12);
        assert!(factors.windows(2).all(|w| (w[0] - w[1]).abs() > 0.1), "the factor depends on the group size: {factors:?}");
    }

    #[test]
    fn centred_advantages_sum_to_zero_and_a_flat_group_is_dropped_under_every_norm() {
        for norm in AdvantageNorm::ALL {
            let adv = group_advantages_with(norm, &[0.2, 0.9, 0.4, 0.7, 0.1]);
            assert!(adv.iter().sum::<f64>().abs() < 1e-12, "{norm:?}: {adv:?}");
            assert_eq!(group_advantages_with(norm, &[0.7, 0.7, 0.7]), vec![0.0; 3]);
            assert_eq!(group_advantages_with(norm, &[1.0]), vec![1.0], "a group of one is the STaR case");
            assert!(group_advantages_with(norm, &[]).is_empty());
        }
    }

    #[test]
    fn the_wire_names_round_trip_and_the_default_keeps_the_reward_scale() {
        for norm in AdvantageNorm::ALL {
            assert_eq!(AdvantageNorm::parse(norm.name()), Ok(norm));
        }
        assert!(AdvantageNorm::parse("standardised").unwrap_err().contains("mean-centred, z-score, leave-one-out"));
        assert_eq!(AdvantageNorm::default(), AdvantageNorm::MeanCentered);
    }
}
