// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The critic: `V(s)`, and the advantage estimator that needs it.
//!
//! [`crate::policy::normalize`] centres a batch of returns and calls the
//! result an advantage. That is a legitimate baseline - it depends on the
//! batch, never on the action taken, so the estimator stays unbiased - and for
//! a one-step decision it is the whole story.
//!
//! It is not enough for a sequential task, and the reason is worth stating
//! because it is easy to talk yourself out of. A batch-mean baseline is a
//! single number for every state in the batch. In an episode where the agent
//! is winning at step 1 and losing at step 8, both steps are compared against
//! the same constant, so the advantage is dominated by *which episode this was*
//! rather than by *whether this action was good*. A state-dependent `V(s)`
//! removes exactly that component. Published PPO uses one, with generalized
//! advantage estimation on top; an implementation without one is
//! REINFORCE-with-baseline wearing PPO's clipped ratio, and should not be
//! called PPO.
//!
//! **The critic here is host-side and small on purpose.** In the configuration
//! this is built for the encoder is FROZEN, so the state representation is a
//! fixed feature vector - and fitting a value function on fixed features is a
//! small regression, not a second deep network. It costs one readback per
//! rollout step and no device work at all.
//!
//! Swedish Embedded AB implements reinforcement learning for on-device control
//! for its clients. If your team needs a policy trained against a real reward
//! rather than a labelled dataset, you can procure our services by sending an
//! email to info@swedishembedded.com.

use data::rng::Rng;

/// A two-layer value function over a fixed state representation.
///
/// Deliberately not [`crate::routing::ConfidenceNet`], which it otherwise
/// resembles: that one ends in a logistic because a confidence is a
/// probability. A value is a return and must be **unbounded**, so this one ends
/// in a linear unit. Squashing a value head is a silent way to cap what the
/// critic can ever predict.
#[derive(Clone, Debug)]
pub struct Critic {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
    d_in: usize,
    hidden: usize,
}

impl Critic {
    pub fn new(d_in: usize, hidden: usize, seed: u64) -> Critic {
        let hidden = hidden.max(1);
        let mut rng = Rng::new(seed);
        let s = (2.0 / d_in.max(1) as f32).sqrt();
        Critic {
            w1: (0..hidden * d_in).map(|_| rng.next_gaussian() as f32 * s).collect(),
            b1: vec![0.0; hidden],
            w2: vec![0.0; hidden],
            b2: 0.0,
            d_in,
            hidden,
        }
    }

    fn hidden_act(&self, x: &[f32]) -> Vec<f32> {
        (0..self.hidden)
            .map(|j| {
                let row = &self.w1[j * self.d_in..(j + 1) * self.d_in];
                (row.iter().zip(x).map(|(&a, &b)| a * b).sum::<f32>() + self.b1[j]).max(0.0)
            })
            .collect()
    }

    pub fn predict(&self, x: &[f32]) -> f32 {
        let a = self.hidden_act(x);
        a.iter().zip(&self.w2).map(|(&v, &w)| v * w).sum::<f32>() + self.b2
    }

    /// Fit to observed returns by mean squared error plus L2.
    ///
    /// Returns the final mean squared error, which is the one number that says
    /// whether the critic is worth trusting: a critic that cannot predict the
    /// return is adding variance rather than removing it.
    pub fn fit(&mut self, x: &[Vec<f32>], targets: &[f32], epochs: usize, lr: f32, l2: f32) -> f32 {
        assert_eq!(x.len(), targets.len(), "one target per state");
        if x.is_empty() {
            return 0.0;
        }
        let scale = 1.0 / x.len() as f32;
        let mut mse = 0.0f32;
        for _ in 0..epochs {
            let mut g_w1 = vec![0.0f32; self.w1.len()];
            let mut g_b1 = vec![0.0f32; self.hidden];
            let mut g_w2 = vec![0.0f32; self.hidden];
            let mut g_b2 = 0.0f32;
            mse = 0.0;
            for (xi, &y) in x.iter().zip(targets) {
                let a = self.hidden_act(xi);
                let v: f32 = a.iter().zip(&self.w2).map(|(&h, &w)| h * w).sum::<f32>() + self.b2;
                let e = v - y;
                mse += e * e * scale;
                let dv = 2.0 * e * scale;
                g_b2 += dv;
                for j in 0..self.hidden {
                    g_w2[j] += dv * a[j];
                    if a[j] > 0.0 {
                        let da = dv * self.w2[j];
                        g_b1[j] += da;
                        for i in 0..self.d_in {
                            g_w1[j * self.d_in + i] += da * xi[i];
                        }
                    }
                }
            }
            for (w, g) in self.w1.iter_mut().zip(&g_w1) {
                *w -= lr * (g + l2 * *w);
            }
            for (w, g) in self.b1.iter_mut().zip(&g_b1) {
                *w -= lr * g;
            }
            for (w, g) in self.w2.iter_mut().zip(&g_w2) {
                *w -= lr * (g + l2 * *w);
            }
            self.b2 -= lr * g_b2;
        }
        mse
    }
}

/// Generalized advantage estimation over one episode.
///
/// `A_t = sum_k (gamma*lambda)^k * delta_{t+k}` where
/// `delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`.
///
/// `lambda` trades bias against variance: `0` is the one-step TD error (low
/// variance, biased by however wrong the critic is) and `1` is the full
/// discounted return minus the baseline (unbiased, high variance). 0.95 is the
/// usual middle.
///
/// `values` must be `V(s_t)` for every step; the episode is assumed to END at
/// the last one, so there is no bootstrap past it. That is right for a game
/// that terminates on a win or a death and WRONG for one truncated by a step
/// limit - a truncated episode should bootstrap from `V(s_last)`, and treating
/// it as terminal tells the critic the return really was zero from there.
pub fn gae(rewards: &[f32], values: &[f32], gamma: f32, lambda: f32, truncated_value: Option<f32>) -> Vec<f32> {
    assert_eq!(rewards.len(), values.len(), "one value per reward");
    let n = rewards.len();
    let mut adv = vec![0.0f32; n];
    let mut acc = 0.0f32;
    for t in (0..n).rev() {
        let next_v = if t + 1 < n {
            values[t + 1]
        } else {
            // The last step: either the episode really ended (0) or it was cut
            // off and the rest of the return has to be estimated.
            truncated_value.unwrap_or(0.0)
        };
        let delta = rewards[t] + gamma * next_v - values[t];
        acc = delta + gamma * lambda * acc;
        adv[t] = acc;
    }
    adv
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A critic has to be able to predict a return that depends on the state,
    /// or it is adding noise where it claims to remove it.
    #[test]
    fn the_critic_learns_a_state_dependent_return() {
        let mut rng = Rng::new(4);
        let d = 6;
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for _ in 0..300 {
            let row: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32).collect();
            // A return that is a real function of the state, on the scale a
            // game like this produces.
            let target = 2.0 * row[0] - 1.5 * row[1] + 0.5;
            x.push(row);
            y.push(target);
        }
        let mut c = Critic::new(d, 16, 1);
        let before = c.fit(&x, &y, 1, 0.0, 0.0);
        let after = c.fit(&x, &y, 600, 0.05, 1e-5);
        assert!(after < before * 0.2, "critic barely fit: {before:.3} -> {after:.3}");
        assert!(after < 0.15, "critic did not fit a linear target: mse {after:.3}");
    }

    /// A value must be able to go NEGATIVE and above one - the arena's returns
    /// run from about -2.5 to +2.5. A squashed head would silently clamp them.
    #[test]
    fn a_value_is_unbounded() {
        let mut c = Critic::new(2, 8, 2);
        let x = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let y = vec![-2.5f32, 2.5];
        c.fit(&x, &y, 3000, 0.05, 0.0);
        assert!(c.predict(&x[0]) < -1.5, "cannot represent a large negative: {}", c.predict(&x[0]));
        assert!(c.predict(&x[1]) > 1.5, "cannot represent a large positive: {}", c.predict(&x[1]));
    }

    /// With a perfect critic every advantage is zero: there is nothing to
    /// learn from a policy whose value is exactly predicted.
    #[test]
    fn a_perfect_critic_leaves_no_advantage() {
        // Rewards 0,0,1 with gamma 1: returns are 1,1,1.
        let rewards = [0.0f32, 0.0, 1.0];
        let values = [1.0f32, 1.0, 1.0];
        let a = gae(&rewards, &values, 1.0, 1.0, None);
        for (t, &v) in a.iter().enumerate() {
            assert!(v.abs() < 1e-5, "step {t} had advantage {v} under a perfect critic");
        }
    }

    /// lambda = 1 must reproduce return-minus-baseline exactly, which is the
    /// identity that says the estimator is the one it claims to be.
    #[test]
    fn lambda_one_is_the_discounted_return_minus_the_baseline() {
        let rewards = [0.5f32, -0.25, 2.0];
        let values = [0.1f32, 0.2, 0.3];
        let gamma = 0.9;
        let got = gae(&rewards, &values, gamma, 1.0, None);
        let ret = crate::policy::returns_to_go(&rewards, gamma);
        for t in 0..rewards.len() {
            let want = ret[t] - values[t];
            assert!((got[t] - want).abs() < 1e-5, "step {t}: gae {} vs return-baseline {want}", got[t]);
        }
    }

    /// lambda = 0 must be the one-step TD error, the other end of the dial.
    #[test]
    fn lambda_zero_is_the_one_step_td_error() {
        let rewards = [0.5f32, -0.25, 2.0];
        let values = [0.1f32, 0.2, 0.3];
        let gamma = 0.9;
        let got = gae(&rewards, &values, gamma, 0.0, None);
        for t in 0..rewards.len() {
            let next = if t + 1 < 3 { values[t + 1] } else { 0.0 };
            let want = rewards[t] + gamma * next - values[t];
            assert!((got[t] - want).abs() < 1e-6, "step {t}: {} vs {want}", got[t]);
        }
    }

    /// A truncated episode must bootstrap rather than be told its future was
    /// worth nothing - the difference between "you died" and "we stopped
    /// watching".
    #[test]
    fn truncation_bootstraps_instead_of_assuming_the_end() {
        let rewards = [0.0f32, 0.0];
        let values = [0.0f32, 0.0];
        let terminal = gae(&rewards, &values, 0.99, 0.95, None);
        let cut = gae(&rewards, &values, 0.99, 0.95, Some(3.0));
        assert!(terminal[1].abs() < 1e-6, "a terminal step should see no future: {}", terminal[1]);
        assert!(cut[1] > 2.9, "a truncated step should inherit the estimate: {}", cut[1]);
        assert!(cut[0] > terminal[0], "the bootstrap did not propagate backward");
    }
}
