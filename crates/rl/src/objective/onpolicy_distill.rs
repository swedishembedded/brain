// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distilling a large teacher model into a small student on the student's
// own samples, for its clients. If your team needs expertise in on-policy distillation of language models across
// vocabularies then you can procure our services by sending an email to info@swedishembedded.com.

//! On-policy distillation from a teacher model: the student samples, a teacher scores the samples, and the student is
//! trained toward the reverse KL `KL(student || teacher)` on its own samples. The teacher is any model, scored outside
//! the training process (the `score` action of a served model gives exactly the log-probabilities both estimators need);
//! the student is trained through the one shared policy-gradient path ([`super::pg_batch`], [`super::external`]), so
//! every model with [`model::training::TokenObjectives`] trains this way and one without is refused with
//! [`model::training::Unsupported`] when the objective prepares it.
//!
//! Which estimator a pair of models may use is decided by comparing their tokenizers
//! ([`data::tokenizer_identity::TokenizerIdentity`]), never by a setting: [`DistillEstimator::for_tokenizers`] names
//! the one to use, and [`DistillEstimator::admit`] refuses token-level distillation for tokenizers that are not
//! identical with a typed [`DistillRefusal`].
//!
//! ## Identical tokenizers: token-level reverse KL
//!
//! For a sampled token `y_t` in context `c_t` (the prompt and the student's tokens before it), the teacher's
//! log-probability `log q(y_t | c_t)` of the same id is the student's signal. Each supervised token carries the
//! per-token reward
//!
//! `A_t = log q(y_t | c_t) - log p_old(y_t | c_t)`  ([`token_advantage`])
//!
//! treated as a constant (stop-gradient) and trained through the clipped policy-gradient step, whose gradient at the
//! behaviour policy is `-A_t * grad log p(y_t | c_t)`. Its expectation over the student's own samples is
//! `sum_t E[(log p - log q)(y_t) * grad log p(y_t)] = sum_t grad KL(p(. | c_t) || q(. | c_t))` with the context held
//! fixed: the per-token reverse KL, the standard on-policy distillation estimator (a per-token reward with no discount
//! over later tokens). We chose it over differentiating a k1 or k3 estimate of the KL directly: a k1 loss
//! `log p - log q` has zero expected gradient on-policy (`E_p[grad log p] = 0`), and a differentiated k3 loss
//! `(q/p - 1) - log(q/p)` has the expected gradient of the forward KL `KL(q || p)`, not the reverse one. The reward
//! form is exact in expectation for the per-token objective, adds no loss term beside the one the policy gradient
//! already gradchecks, and its clip, bound and cap guards ([`super::external::PolicyGradientConfig::advantage_bound`],
//! `sequence_advantage_cap`) apply unchanged. Dropping the later tokens' terms (the full sequence-level gradient also
//! credits `y_t` with how the KL of every later position changes) biases it toward the per-position divergence and
//! makes its variance far lower; it is what published on-policy distillation trains with.
//!
//! When the teacher is the student (`q = p_old`) every `A_t` is zero, so the term and its gradient vanish.
//!
//! ## Different tokenizers: the sequence-level estimator (higher variance)
//!
//! A teacher with another tokenizer has no log-probability for the student's token ids, but it has one for the
//! student's *text*. For a sampled completion `x` of a prompt, each model renders the conversation with its own chat
//! template, tokenizes the completion its own way and sums its log-probabilities: `log q(x)` and `log p(x)`. Then
//!
//! `grad KL(p || q) = E_{x ~ p}[(log p(x) - log q(x)) * grad log p(x)]`
//!
//! (the second term of the product rule, `E_p[grad log p(x)]`, is zero), so with the signal
//! `r = log q(x) - log p(x)` ([`sequence_log_ratio`]) REINFORCE with `-r * grad log p(x)` is unbiased for the
//! sequence-level reverse KL. Each completion's signal is taken against the leave-one-out baseline of the other
//! completions of its prompt ([`sequence_advantages`]): `A_i = r_i - mean_{j != i} r_j`. The baseline does not depend on
//! `x_i`, so it leaves the expectation unchanged, and it removes the offset the completions of one prompt share (a
//! teacher that disagrees with the student scores every sampled completion nats below the student's own
//! log-probability), which is most of the variance; the group mean, which includes `r_i` itself, would scale the
//! expectation by `(G - 1) / G`. Where the signals share little offset (a teacher close to the student) the baseline's
//! own noise can outweigh what it removes; the estimator stays unbiased either way. Every token of the completion
//! carries the same `A_i`.
//!
//! This path is **higher variance** than the token-level one: one scalar per completion, whose size grows with the
//! completion's length, credits every token alike, where the token-level reward says which tokens the teacher
//! disagrees with. It needs more samples per prompt (a group of at least two, more is better) for the same progress,
//! and it is the only exact option when the tokenizers differ.

use std::fmt;

use data::tokenizer_identity::TokenizerIdentity;

use super::advantage::{group_advantages_with, AdvantageNorm};

/// The two on-policy distillation estimators. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistillEstimator {
    /// Token-level reverse KL on the student's sampled ids, from the teacher's log-probabilities of the same ids. Needs
    /// identical tokenizers.
    TokenLevel,
    /// The sequence-level REINFORCE estimator on the completion's text, scored by each model in its own tokenization:
    /// any pair of models, higher variance.
    SequenceLevel,
}

impl DistillEstimator {
    /// The estimator a run distilling from a teacher whose tokenizer compared to the student's as `identity` uses: the
    /// token-level one when the tokenizers are identical, the sequence-level one otherwise.
    pub fn for_tokenizers(identity: &TokenizerIdentity) -> DistillEstimator {
        if identity.is_identical() {
            DistillEstimator::TokenLevel
        } else {
            DistillEstimator::SequenceLevel
        }
    }

    /// The estimator's name on the wire and in a run's record.
    pub fn name(self) -> &'static str {
        match self {
            DistillEstimator::TokenLevel => "token_level",
            DistillEstimator::SequenceLevel => "sequence_level",
        }
    }

    /// Whether this estimator is exact for a teacher whose tokenizer compared to the student's as `teacher_tokenizer`
    /// (`None`: no teacher tokenizer was compared). The sequence-level estimator is exact for any teacher.
    ///
    /// # Errors
    /// Token-level distillation with no tokenizer comparison, or with tokenizers that differ: the teacher's
    /// log-probabilities would be of other tokens than the student's ids.
    pub fn admit(self, teacher_tokenizer: Option<&TokenizerIdentity>) -> Result<(), DistillRefusal> {
        match (self, teacher_tokenizer) {
            (DistillEstimator::SequenceLevel, _) => Ok(()),
            (DistillEstimator::TokenLevel, None) => Err(DistillRefusal::TeacherTokenizerUnchecked),
            (DistillEstimator::TokenLevel, Some(identity)) if identity.is_identical() => Ok(()),
            (DistillEstimator::TokenLevel, Some(identity)) => Err(DistillRefusal::TokenizersDiffer(*identity)),
        }
    }
}

/// Why token-level distillation is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistillRefusal {
    /// The run carries the teacher's per-token log-probabilities and names no teacher whose tokenizer was compared
    /// with the student's.
    TeacherTokenizerUnchecked,
    /// The teacher's tokenizer differs from the student's in the parts the identity names.
    TokenizersDiffer(TokenizerIdentity),
}

impl fmt::Display for DistillRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistillRefusal::TeacherTokenizerUnchecked => write!(f, "token-level distillation (\"teacher_logprobs\") needs the teacher's tokenizer compared with the student's, and the run names no teacher; name the teacher checkpoint so its tokenizer is checked"),
            DistillRefusal::TokenizersDiffer(identity) => write!(f, "token-level distillation (\"teacher_logprobs\") needs identical tokenizers and the teacher's differs from the student's: {identity}; its log-probabilities are of other tokens than the student's ids. Score each completion's text with the teacher instead (\"teacher_logprob\", the sequence-level estimator)"),
        }
    }
}

impl std::error::Error for DistillRefusal {}

/// The token-level per-token reward of a sampled token: the teacher's log-probability of it less the behaviour
/// policy's, a constant to the gradient. Zero when the teacher is the student.
pub fn token_advantage(teacher_logprob: f32, student_logprob: f32) -> f32 {
    teacher_logprob - student_logprob
}

/// The sequence-level signal of one completion: the teacher's log-probability of its text (in the teacher's own
/// tokenization and chat template) less the student's (in the student's).
pub fn sequence_log_ratio(teacher_logprob: f64, student_logprob: f64) -> f64 {
    teacher_logprob - student_logprob
}

/// The advantages of the completions of one prompt from their [`sequence_log_ratio`]s: each against the mean of the
/// others (leave-one-out), which leaves the estimator's expectation unchanged. A group of one has no baseline; the
/// caller refuses it before asking (its advantage would be the raw signal).
pub fn sequence_advantages(log_ratios: &[f64]) -> Vec<f64> {
    group_advantages_with(AdvantageNorm::LeaveOneOut, log_ratios)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(teacher_vocab: serde_json::Value) -> TokenizerIdentity {
        let doc = |vocab: serde_json::Value| serde_json::json!({"model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"]]}}).to_string();
        TokenizerIdentity::of_json(doc(serde_json::json!({"a": 0, "b": 1, "ab": 2})).as_bytes(), doc(teacher_vocab).as_bytes()).unwrap()
    }

    #[test]
    fn the_tokenizer_check_picks_the_estimator_and_token_level_is_refused_for_a_different_tokenizer() {
        let same = identity(serde_json::json!({"a": 0, "b": 1, "ab": 2}));
        let other = identity(serde_json::json!({"a": 0, "b": 1, "ab": 3}));
        assert_eq!(DistillEstimator::for_tokenizers(&same), DistillEstimator::TokenLevel);
        assert_eq!(DistillEstimator::for_tokenizers(&other), DistillEstimator::SequenceLevel);
        assert_eq!(DistillEstimator::TokenLevel.admit(Some(&same)), Ok(()));
        let refusal = DistillEstimator::TokenLevel.admit(Some(&other)).expect_err("different tokenizers");
        assert_eq!(refusal, DistillRefusal::TokenizersDiffer(other));
        assert!(refusal.to_string().contains("vocab"), "the refusal names what differs: {refusal}");
        assert_eq!(DistillEstimator::TokenLevel.admit(None), Err(DistillRefusal::TeacherTokenizerUnchecked));
        for teacher in [None, Some(&same), Some(&other)] {
            assert_eq!(DistillEstimator::SequenceLevel.admit(teacher), Ok(()), "the sequence-level estimator holds for any teacher");
        }
    }

    /// Completions in the toy.
    const N: usize = 5;

    /// A toy student `p = softmax(theta)` and teacher `q` over five whole completions (the texts a prompt can be
    /// answered with), small enough that every group of `G` samples can be enumerated: the expectation and covariance
    /// of the estimator are then exact sums, not Monte Carlo. The teacher puts most of its mass on a completion the
    /// student rarely samples, so the completions the student does sample all score well below its own
    /// log-probability: the shared offset a distillation signal carries, which the baseline is there to remove.
    struct Toy {
        p: [f64; N],
        q: [f64; N],
    }

    impl Toy {
        fn new() -> Toy {
            let theta = [0.4f64, -0.3, 1.1, 0.0, -3.0];
            let z: f64 = theta.iter().map(|t| t.exp()).sum();
            Toy { p: theta.map(|t| t.exp() / z), q: [0.05, 0.04, 0.12, 0.04, 0.75] }
        }

        /// `grad_theta log p(x) = e_x - p`.
        fn score(&self, x: usize) -> [f64; N] {
            std::array::from_fn(|k| f64::from(u8::from(k == x)) - self.p[k])
        }

        /// The exact gradient of `KL(p || q)` in `theta`.
        fn kl_gradient(&self) -> [f64; N] {
            let mut g = [0.0; N];
            for x in 0..N {
                let w = self.p[x] * (self.p[x].ln() - self.q[x].ln());
                for (k, s) in self.score(x).iter().enumerate() {
                    g[k] += w * s;
                }
            }
            g
        }

        /// The exact mean and total variance (trace of the covariance) of the group estimator
        /// `-(1/G) sum_i A_i grad log p(x_i)` over every ordered group of `g` samples, with `advantages` turning the
        /// group's signals `r_i = log q(x_i) - log p(x_i)` into the `A_i`.
        fn estimator(&self, g: u32, advantages: impl Fn(&[f64]) -> Vec<f64>) -> ([f64; N], f64) {
            let (mut mean, mut second) = ([0.0f64; N], 0.0f64);
            for code in 0..N.pow(g) {
                let xs: Vec<usize> = (0..g).map(|i| code / N.pow(i) % N).collect();
                let prob: f64 = xs.iter().map(|&x| self.p[x]).product();
                let signals: Vec<f64> = xs.iter().map(|&x| sequence_log_ratio(self.q[x].ln(), self.p[x].ln())).collect();
                let a = advantages(&signals);
                let mut est = [0.0f64; N];
                for (i, &x) in xs.iter().enumerate() {
                    for (k, s) in self.score(x).iter().enumerate() {
                        est[k] -= a[i] * s / f64::from(g);
                    }
                }
                for k in 0..N {
                    mean[k] += prob * est[k];
                }
                second += prob * est.iter().map(|e| e * e).sum::<f64>();
            }
            (mean, second - mean.iter().map(|m| m * m).sum::<f64>())
        }
    }

    #[test]
    fn the_sequence_level_estimator_is_the_reverse_kl_gradient_in_expectation() {
        let toy = Toy::new();
        let exact = toy.kl_gradient();
        assert!(exact.iter().map(|g| g.abs()).sum::<f64>() > 0.1, "the toy teacher must disagree with the student: {exact:?}");
        let (raw, _) = toy.estimator(3, |r| r.to_vec());
        for k in 0..N {
            assert!((raw[k] - exact[k]).abs() < 1e-12, "without a baseline: {raw:?} vs {exact:?}");
        }
    }

    #[test]
    fn the_leave_one_out_baseline_keeps_the_expectation_and_lowers_the_variance() {
        let toy = Toy::new();
        let exact = toy.kl_gradient();
        for g in [2u32, 3, 4] {
            let (raw_mean, raw_var) = toy.estimator(g, |r| r.to_vec());
            let (loo_mean, loo_var) = toy.estimator(g, sequence_advantages);
            for k in 0..N {
                assert!((loo_mean[k] - exact[k]).abs() < 1e-12, "G {g}: {loo_mean:?} vs {exact:?}");
                assert!((raw_mean[k] - exact[k]).abs() < 1e-12, "G {g}: {raw_mean:?} vs {exact:?}");
            }
            assert!(loo_var < 0.5 * raw_var, "G {g}: variance with the baseline {loo_var}, without {raw_var}");
        }
    }

    #[test]
    fn a_teacher_that_is_the_student_gives_no_signal() {
        for lp in [-0.01f32, -2.5, -17.0] {
            assert_eq!(token_advantage(lp, lp), 0.0);
        }
        assert_eq!(sequence_advantages(&[sequence_log_ratio(-12.5, -12.5), sequence_log_ratio(-3.0, -3.0)]), vec![0.0, 0.0]);
    }
}
