// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training that gives the same model on every backend for its clients. If your team
// needs expertise in cross-backend numerical parity of training then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Comparing two training runs of one model on one batch that differ only in where they ran (two backends, two
//! reduction orders), held to the binary32 round-off bound of [`crate::rounding`].
//!
//! A run is recorded as a [`Trace`]: the loss before each optimiser step and after the last, and the gradient of every
//! trained tensor at each step. One forward and backward whose longest chain of rounded operations is `k` puts the
//! two runs within [`crate::rounding::agreement_bound`] of each other, relative to the magnitudes involved. Each step
//! adds one more forward and backward through weights that differ by the previous steps' rounding, so the bound at
//! step `s` (counting from 0) is `s + 1` times the first step's. That holds while the optimiser step is a Lipschitz
//! function of the gradient: an AdamW whose epsilon is far above the gradient's round-off, so a reordered sum cannot
//! flip the direction an element near zero gradient steps in.

/// What one run measured: the loss before each step and after the last, and each step's gradient of every trained
/// tensor, by name.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub losses: Vec<f32>,
    pub grads: Vec<Vec<(String, Vec<f32>)>>,
}

/// The relative bound of one forward and backward whose longest chain of rounded binary32 operations is `k`.
pub fn step_bound(k: usize) -> f64 {
    crate::rounding::agreement_bound(k, 1.0)
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let diff: f64 = got.iter().zip(want).map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = want.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>().sqrt();
    diff / norm
}

/// Assert that `got` trained as `want` did: every loss within `(s + 1) * bound` of the reference's (relative, at
/// least one nat of scale), every non-zero gradient within `(s + 1) * bound` in relative L2, a gradient the reference
/// has none of absent too, at least half the trained tensors' gradients compared, and the reference's loss falling
/// (so the comparison is of training, not of a model that did not move). `what` names the run in a failure.
pub fn assert_same_training(want: &Trace, got: &Trace, bound: f64, what: &str) {
    assert_eq!(got.losses.len(), want.losses.len(), "{what}: the same number of losses");
    for (step, (g, w)) in got.losses.iter().zip(&want.losses).enumerate() {
        let allowed = (step + 1) as f64 * bound * f64::from(w.abs()).max(1.0);
        let off = (f64::from(*g) - f64::from(*w)).abs();
        assert!(off <= allowed, "{what}: loss {step} is {g} against the reference's {w}, off by {off:e} past the round-off bound {allowed:e}");
    }
    assert_eq!(got.grads.len(), want.grads.len(), "{what}: the same number of steps");
    let mut compared = 0;
    for (step, (g, w)) in got.grads.iter().zip(&want.grads).enumerate() {
        assert_eq!(g.iter().map(|t| &t.0).collect::<Vec<_>>(), w.iter().map(|t| &t.0).collect::<Vec<_>>(), "{what}: step {step} trains the same tensors");
        for ((name, g), (_, w)) in g.iter().zip(w) {
            if w.iter().all(|x| *x == 0.0) {
                assert!(g.iter().all(|x| *x == 0.0), "{what}: step {step} {name} has a gradient where the reference has none");
                continue;
            }
            compared += 1;
            let (off, allowed) = (rel_l2(g, w), (step + 1) as f64 * bound);
            assert!(off <= allowed, "{what}: step {step} {name}: the gradient is off the reference's by {off:e} (relative L2), past the round-off bound {allowed:e}");
        }
    }
    let tensors: usize = want.grads.iter().map(Vec::len).sum();
    assert!(compared * 2 >= tensors, "{what}: gradients reached the trained tensors ({compared} of {tensors} compared)");
    assert!(want.losses.last() < want.losses.first(), "{what}: the steps descend ({:?}), so the comparison is of training", want.losses);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(losses: &[f32], grad: &[f32]) -> Trace {
        Trace { losses: losses.to_vec(), grads: vec![vec![("w".into(), grad.to_vec()), ("zero".into(), vec![0.0; 2])]; losses.len() - 1] }
    }

    #[test]
    fn runs_within_the_bound_agree_and_one_past_it_does_not() {
        let want = trace(&[3.0, 2.5, 2.0], &[0.5, -0.25]);
        assert_same_training(&want, &trace(&[3.0 + 1e-6, 2.5, 2.0], &[0.5, -0.25 + 1e-7]), step_bound(64), "close");
        let far = std::panic::catch_unwind(|| assert_same_training(&want, &trace(&[3.0, 2.5, 2.01], &[0.5, -0.25]), step_bound(64), "far"));
        assert!(far.is_err(), "a loss off by 1e-2 is past a 64-operation bound");
        let stray = std::panic::catch_unwind(|| {
            let mut got = want.clone();
            got.grads[0][1].1[0] = 1e-9;
            assert_same_training(&want, &got, step_bound(64), "stray");
        });
        assert!(stray.is_err(), "a gradient where the reference has none is a difference");
    }
}
