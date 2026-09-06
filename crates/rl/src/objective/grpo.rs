// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GRPO: group-relative, clipped-importance-ratio policy-gradient training,
//! with an optional k3 KL-to-a-frozen-reference term - and plain
//! rejection-sampling/STaR training as the SAME weight formula's degenerate
//! (single-completion) case, not a parallel implementation next to it.
//!
//! Swedish Embedded AB builds the training-regime machinery that turns a
//! verifiable-reward environment into a trained model - GRPO, rejection
//! sampling, and everything in between reduce to one gradient-checked
//! weighted cross-entropy kernel here. If your team needs RL-from-
//! verifiable-rewards on top of a from-scratch training stack, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! ## The reduction to a per-token `Batch::LmWeighted` weight
//!
//! A model's ordinary cross-entropy backward already writes, per token
//! position, `d_logits = (softmax(z) - onehot(target)) / count` (`count`
//! being that forward's own non-IGNORE target count), and
//! [`model::Model::set_loss_weights`] scales that row-wise before the head's
//! backward reads it. GRPO's clipped surrogate for one token is
//! `min(r*A, clip(r, 1-eps, 1+eps)*A)` where `r = exp(new_lp - old_lp)` is
//! the importance ratio between the CURRENT policy's logprob (`new_lp`,
//! read back from [`model::Model::batch_token_logprobs`] immediately after
//! an ordinary forward) and the SAMPLING policy's logprob captured at
//! rollout time (`old_lp`, [`model::rollout::Completion::logprobs`]) and `A`
//! is the token's (group-broadcast) advantage. Differentiating that
//! surrogate w.r.t. the logits and dividing by `count` gives EXACTLY
//! `weight * d_logits` for `weight = A*r` on an unclipped token and `0` on a
//! clipped one - `r` is not stopped-gradient here, its own derivative
//! `dr/dz = r * dlogp/dz` is what produces the `A*r` factor, so ordinary
//! cross-entropy backward differentiates the real clipped objective exactly
//! by using this `weight`, no new kernel required. The optional k3
//! KL-to-reference term (`e^u - u - 1`, `u = ref_lp - new_lp`) adds
//! `kl_beta*(e^u - 1)` to that same weight by an identical derivation (see
//! [`token_term`]'s own doc comment for both derivations spelled out
//! term-by-term).
//!
//! [`Grpo::micro_step`] therefore differs from a plain weighted-CE step only
//! in WHERE the weight comes from: one ordinary forward (to read `new_lp`),
//! a host-side computation of `weight` per token, [`model::Model::
//! set_loss_weights`], then one ordinary backward. This is also why the
//! blanket `gradcheck::CheckModel for M: Model` cannot gate this file:
//! `Model::forward()`'s own return value is the model's plain (default- or
//! stale-weighted) loss, computed BEFORE the true per-token weight is even
//! known - not the scalar `backward()` ends up differentiating. The gate
//! (`crates/rl/tests/grpo_gradcheck.rs`) supplies its own local `CheckModel`
//! whose `loss()` recomputes the true clipped surrogate host-side, using
//! this exact [`token_term`] function, so finite differences test the real
//! objective.
//!
//! ## RFT/STaR as the degenerate case
//!
//! [`group_advantages`] special-cases a group of size <= 1: GRPO's own
//! z-score advantage would divide a single reward by a zero (or near-zero)
//! group standard deviation and collapse to `0` (or blow up), which is
//! nonsensical for a group that was never a comparison in the first place.
//! Plain rejection-sampling/STaR training - draw completions, keep only a
//! verified-correct one, deduplicate to one kept completion per prompt, and
//! train on it with uniform weight - is realized by [`Grpo::micro_step`]
//! itself running the SAME `group_size == 1` path through the SAME
//! [`token_term`] weight formula with `advantage = 1.0`: since STaR trains
//! on the completion immediately after sampling it (no intervening
//! optimizer step), `new_lp == old_lp` at that point, so `r = 1` and
//! `weight = 1.0 * 1 = 1.0` - ordinary uniform-weight SFT on the kept
//! completion, falling out of the identical code path rather than a second
//! implementation next to it.

use data::rng::Rng;
use model::rollout::{Completion, ModelRollout, Rollout};
use model::{Batch, Model, Objective, IGNORE};

use crate::env::{Environment, Verifier};

/// Per-group advantage `A_i = (r_i - mean_r) / (std_r + 1e-4)`.
///
/// A group of size `<= 1` cannot be z-scored meaningfully (there is nothing
/// to compare the one reward against) - this is the RFT/STaR degenerate
/// case (see the module doc comment), realized as uniform advantage `1.0`
/// rather than GRPO's z-score. A group of size `>= 2` with (near-)zero
/// reward variance (an all-right or all-wrong group) is dropped - `0.0` for
/// every member - because it carries no useful comparison either; this is
/// what makes GRPO cheap: a group that teaches nothing costs nothing.
pub fn group_advantages(rewards: &[f32]) -> Vec<f32> {
    if rewards.len() <= 1 {
        return vec![1.0; rewards.len()];
    }
    let g = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / g;
    let var = rewards.iter().map(|&r| (r - mean) * (r - mean)).sum::<f32>() / g;
    let std = var.sqrt();
    if std < 1e-6 {
        return vec![0.0; rewards.len()];
    }
    rewards.iter().map(|&r| (r - mean) / (std + 1e-4)).collect()
}

/// One token's [`model::Batch::LmWeighted`] weight, plus the host-computed
/// loss term it is the analytic gradient of (`d(loss)/dz = weight *
/// (softmax(z) - onehot(target))` - see the module doc comment for the full
/// derivation) - the pair the gradcheck gate's `loss()`/`backward()` must
/// agree on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenTerm {
    pub weight: f32,
    pub loss: f32,
}

/// `advantage == 0.0` means this token contributes nothing to the policy
/// term (a dropped zero-variance group, or a completion whose reward
/// happened to land exactly on its group's mean) - the KL term (when
/// `kl_beta > 0.0`) still applies independently, since it regularizes
/// EVERY token toward the reference regardless of that token's own reward.
///
/// Clip rule, derived (not looked up) from `min(r*A, clip(r,1-eps,1+eps)*A)`
/// (see the module doc comment): the raw (differentiable) term wins
/// whenever `r*A <= clip(r,1-eps,1+eps)*A`; the derivative is then exactly
/// `A*r`. When the clipped term is strictly smaller, the surrogate is
/// locally CONSTANT in `r` (the clamp has saturated), so the weight is
/// exactly `0.0`, not "clamped to a small value" but genuinely zero
/// gradient contribution.
pub fn token_term(old_lp: f32, new_lp: f32, advantage: f32, clip_eps: f32, ref_lp: Option<f32>, kl_beta: f32) -> TokenTerm {
    let ratio = (new_lp - old_lp).exp();
    let mut weight = 0.0f32;
    let mut loss = 0.0f32;

    if advantage != 0.0 {
        let unclipped = ratio * advantage;
        let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage;
        let surrogate = unclipped.min(clipped);
        loss += -surrogate;
        if unclipped <= clipped {
            weight += advantage * ratio;
        }
    }

    if kl_beta > 0.0 {
        let ref_lp = ref_lp.expect("token_term: kl_beta > 0.0 requires a reference logprob");
        // k3 estimator: KL = e^u - u - 1, u = ref_lp - new_lp(theta).
        // d(KL)/d(new_lp) = -(e^u - 1); d(new_lp)/dz = onehot - softmax, so
        // d(KL)/dz = (e^u - 1) * (softmax - onehot) - exactly `(e^u - 1)`
        // more weight on the ordinary CE gradient direction.
        let u = ref_lp - new_lp;
        let eu = u.exp();
        weight += kl_beta * (eu - 1.0);
        loss += kl_beta * (eu - u - 1.0);
    }

    TokenTerm { weight, loss }
}

/// True if `ratio` sits within `margin` of either clip boundary (`1 -
/// clip_eps` or `1 + clip_eps`) - [`token_term`]'s clip indicator is
/// piecewise-constant there, so an arbitrarily small parameter perturbation
/// can flip which term the `min` picks. A finite-difference gate must keep
/// its sampled base-point ratios clear of this margin (see
/// `crates/rl/tests/grpo_gradcheck.rs`'s own doc comment) rather than
/// loosen tolerance to paper over the resulting mismatch.
pub fn near_clip_boundary(ratio: f32, clip_eps: f32, margin: f32) -> bool {
    (ratio - (1.0 - clip_eps)).abs() < margin || (ratio - (1.0 + clip_eps)).abs() < margin
}

/// Pack one sampled `completion` (of `prompt`) into row `row` of the
/// `rows x seq_len` batch tensors, at the alignment [`Completion::logprobs`]
/// already uses: position `prompt.len() - 1 + i` predicts `completion.
/// tokens[i]` - the same "targets[t] predicts position t+1" convention
/// every other [`model::Batch::Lm`] user follows, restricted to the
/// completion's own span (the prompt itself is never scored). A completion
/// that overruns `seq_len` is silently truncated at the tail - the
/// un-fitting tokens are dropped, not scored, rather than panicking on an
/// over-long rollout.
#[allow(clippy::too_many_arguments)]
fn pack_row(
    tokens: &mut [u32],
    targets: &mut [u32],
    old_lp: &mut [f32],
    ref_lp: &mut [f32],
    advantage: &mut [f32],
    row: usize,
    seq_len: usize,
    prompt: &[u32],
    completion: &Completion,
    adv: f32,
    ref_logprobs: Option<&[f32]>,
) {
    let base = row * seq_len;
    let plen = prompt.len();
    if plen == 0 {
        // Nothing to condition the first completion token on in this packed
        // representation - not a real scenario (every task prompt is
        // non-empty), but the underflow below would otherwise be silent.
        return;
    }
    for (i, &t) in prompt.iter().chain(completion.tokens.iter()).take(seq_len).enumerate() {
        tokens[base + i] = t;
    }
    for i in 0..completion.tokens.len() {
        let tok_idx = plen + i;
        if tok_idx >= seq_len {
            break;
        }
        let t_pos = tok_idx - 1;
        targets[base + t_pos] = completion.tokens[i];
        old_lp[base + t_pos] = completion.logprobs[i];
        advantage[base + t_pos] = adv;
        if let Some(rl) = ref_logprobs {
            ref_lp[base + t_pos] = rl[i];
        }
    }
}

/// One already-sampled, already-scored training row, queued for a future
/// [`Grpo::micro_step`] call - see that struct's own doc comment on why
/// sampling and training happen on different calls rather than in one.
struct PackedRow {
    tokens: Vec<u32>,
    targets: Vec<u32>,
    old_lp: Vec<f32>,
    ref_lp: Vec<f32>,
    advantage: Vec<f32>,
    /// The raw completion span this row was packed from - kept alongside
    /// the packed arrays purely for [`CycleLog`]'s bookkeeping (self-improve
    /// roadmap P18's "no label ever enters training" structural assertion),
    /// not read by [`Grpo::micro_step`]'s own training math.
    completion: Vec<u32>,
}

/// Observability hook for self-improve roadmap P18's structural "every
/// completion span written into a training set is a member of the multiset
/// the policy actually sampled that cycle" assertion: every completion
/// [`Grpo::refill`] draws (kept or dropped) and every completion span
/// [`Grpo::micro_step`] actually pops off the training queue, recorded in
/// order. A plain `Rc<RefCell<..>>` handle rather than a getter on `Grpo`
/// itself, because `Grpo` is typically consumed by value (`model::fit_with`
/// takes its [`model::Objective`] by value and drops it at the end) - attach
/// a log via [`Grpo::with_log`] before handing the objective off, keep your
/// own clone, and read it back afterward.
#[derive(Clone, Default)]
pub struct CycleLog(std::rc::Rc<std::cell::RefCell<CycleLogInner>>);

#[derive(Default)]
struct CycleLogInner {
    sampled: Vec<Vec<u32>>,
    trained: Vec<Vec<u32>>,
}

impl CycleLog {
    pub fn new() -> CycleLog {
        CycleLog::default()
    }

    /// Every completion [`Grpo::refill`] sampled (kept or dropped) since
    /// construction, in sample order.
    pub fn sampled(&self) -> Vec<Vec<u32>> {
        self.0.borrow().sampled.clone()
    }

    /// Every completion span [`Grpo::micro_step`] actually popped off the
    /// training queue and trained on, in training order.
    pub fn trained(&self) -> Vec<Vec<u32>> {
        self.0.borrow().trained.clone()
    }

    fn push_sampled(&self, tokens: Vec<u32>) {
        self.0.borrow_mut().sampled.push(tokens);
    }

    fn push_trained(&self, tokens: Vec<u32>) {
        self.0.borrow_mut().trained.push(tokens);
    }
}

/// GRPO's (and RFT/STaR's) hyperparameters. `seq_len` is the packed row's
/// length - MUST match the `t` the model was constructed with; `b` is
/// implicitly `1` (see [`Grpo`]'s own doc comment on why one micro-step
/// trains exactly one row, and why that is not a shortcut but a real
/// constraint of every current [`model::Model::logits_all`] impl this
/// crate's `Rollout` sits on). Drive a whole group's worth of rows through
/// ONE optimizer step by setting [`model::FitOpts::grad_accum`] to a
/// multiple of `group_size`.
#[derive(Clone, Copy, Debug)]
pub struct GrpoConfig {
    /// Completions sampled per prompt. `1` selects the RFT/STaR degenerate
    /// path (see the module doc comment); `>= 2` is ordinary GRPO.
    pub group_size: usize,
    /// PPO-style clip band half-width (`eps` in `clip(r, 1-eps, 1+eps)`).
    pub clip_eps: f32,
    /// k3 KL-to-reference coefficient. `0.0` disables the KL term entirely
    /// (no reference logprobs are read or required).
    pub kl_beta: f32,
    /// Packed row length (`t`) - prompt length + `rollout.max_new`, at
    /// least.
    pub seq_len: usize,
    /// Sampling policy + stopping conditions for every rollout.
    pub rollout: model::rollout::RolloutParams,
    /// `group_size == 1` (RFT/STaR) only: how many completions to sample
    /// per prompt before giving up on finding a verified-correct one. Not
    /// read when `group_size != 1`.
    pub max_attempts: usize,
}

/// One GRPO (or, with `cfg.group_size == 1`, RFT/STaR) training objective
/// over environment `E`'s tasks, scored by verifier `V`. Generic over any
/// `M: model::Model` that has adopted weighted-loss support
/// ([`model::Model::enable_weighted_loss`]/[`model::Model::
/// set_loss_weights`]/[`model::Model::batch_token_logprobs`]) - today
/// `qwen3::Qwen` and `gpt2::Gpt`.
///
/// **Why `micro_step` trains exactly one row, not a whole packed batch of
/// `group_size` (or more) rows at once.** [`model::rollout::Rollout`]
/// samples through [`model::Model::logits_all`], and every current impl of
/// it (`qwen3::Qwen::logits_all`, `gpt2::Gpt::logits_all`) asserts `self.b
/// == 1` - a single-sequence-at-a-time, re-prefill-per-sample oracle by
/// design (see `model::rollout`'s own doc comment). A `Model` is
/// constructed once, at a fixed `(b, t)`, so the SAME instance this
/// objective trains cannot also be the one a group's `group_size`
/// completions are rolled out from unless `b == 1` throughout - meaning one
/// micro-step's forward/backward is naturally one row, not a `group_size`-
/// row packed batch. [`Grpo`] embraces that: [`Objective::micro_step`]
/// pops one already-sampled, already-scored [`PackedRow`] off an internal
/// queue and trains it; when the queue is empty it samples and scores a
/// FRESH group (or, for RFT/STaR, rejection-samples one completion) and
/// refills. The training loop's own grad-accumulation
/// (`model::train::fit_with`'s existing `opts.grad_accum` loop) is what
/// turns a run of `group_size` such micro-steps back into one optimizer
/// step over the whole group - no second accumulation mechanism is
/// invented here.
/// The frozen-reference logprob source the k3 KL term reads: `(prompt's
/// task, sampled completion) -> per-completion-token reference logprob`.
type RefLogprobFn = Box<dyn Fn(&crate::env::Task, &Completion) -> Vec<f32>>;

pub struct Grpo<E: Environment, V: Verifier> {
    env: E,
    verifier: V,
    cfg: GrpoConfig,
    /// Precomputed, frozen reference logprobs for one (prompt, completion)
    /// pair, called ONCE per sampled completion (at rollout time, not at
    /// every training step) - the module doc comment's "no co-resident
    /// reference model needed, only its constant logprobs". `None` is only
    /// valid when `cfg.kl_beta <= 0.0`.
    ref_logprobs: Option<RefLogprobFn>,
    pending: std::collections::VecDeque<PackedRow>,
    last_mean_reward: f32,
    last_kept_frac: f32,
    log: CycleLog,
}

impl<E: Environment, V: Verifier> Grpo<E, V> {
    pub fn new(env: E, verifier: V, cfg: GrpoConfig) -> Grpo<E, V> {
        assert!(cfg.group_size >= 1, "Grpo: group_size must be >= 1");
        Grpo {
            env,
            verifier,
            cfg,
            ref_logprobs: None,
            pending: std::collections::VecDeque::new(),
            last_mean_reward: 0.0,
            last_kept_frac: 0.0,
            log: CycleLog::default(),
        }
    }

    /// Attach the frozen-reference logprob source the k3 KL term reads
    /// (`cfg.kl_beta > 0.0` requires this). Called once per sampled
    /// completion, immediately after it is drawn - see this struct's own
    /// field doc comment.
    pub fn with_reference<F>(mut self, f: F) -> Grpo<E, V>
    where
        F: Fn(&crate::env::Task, &Completion) -> Vec<f32> + 'static,
    {
        self.ref_logprobs = Some(Box::new(f));
        self
    }

    /// Attach an external [`CycleLog`] so a caller can observe what this
    /// objective actually sampled/trained on after it has been consumed
    /// (typically by `model::fit_with`) - see [`CycleLog`]'s own doc
    /// comment.
    pub fn with_log(mut self, log: CycleLog) -> Grpo<E, V> {
        self.log = log;
        self
    }

    fn pack(&self, prompt: &[u32], completion: &Completion, adv: f32, ref_logprobs: Option<&[f32]>) -> PackedRow {
        let mut row = PackedRow {
            tokens: vec![0u32; self.cfg.seq_len],
            targets: vec![IGNORE; self.cfg.seq_len],
            old_lp: vec![0f32; self.cfg.seq_len],
            ref_lp: vec![0f32; self.cfg.seq_len],
            advantage: vec![0f32; self.cfg.seq_len],
            completion: completion.tokens.clone(),
        };
        pack_row(&mut row.tokens, &mut row.targets, &mut row.old_lp, &mut row.ref_lp, &mut row.advantage, 0, self.cfg.seq_len, prompt, completion, adv, ref_logprobs);
        row
    }

    /// Sample (and score) one fresh group, queuing every KEPT row (nonzero
    /// advantage) for future [`Objective::micro_step`] calls to train. A
    /// group that drops entirely (all-right/all-wrong GRPO group, or a
    /// STaR prompt with no verified-correct completion within
    /// `max_attempts`) queues nothing - the caller's next `micro_step` call
    /// then either refills again or (if `pending` stays empty) reports a
    /// no-op loss, matching [`model::Batch::LmWeighted`]'s own "a weight of
    /// 0.0 ... masked-out and reward=0 collapse to the same thing".
    fn refill<M: Model>(&mut self, model: &M, rng: &mut Rng) {
        let task = self.env.tasks(rng.next_u64()).into_iter().next().expect("Grpo: Environment::tasks produced no task");
        let mut roll = ModelRollout::new(model);

        if self.cfg.group_size == 1 {
            // RFT/STaR: bounded rejection sampling for one verified correct,
            // deduplicated completion - see module doc comment.
            let mut rewards = Vec::with_capacity(self.cfg.max_attempts.max(1));
            for _ in 0..self.cfg.max_attempts.max(1) {
                let c = roll.sample_n(&task.prompt, 1, &self.cfg.rollout, rng).pop().expect("sample_n(1) returns exactly one completion");
                self.log.push_sampled(c.tokens.clone());
                let reward = self.verifier.verify(&task, &[], &c.tokens).value;
                rewards.push(reward);
                if reward > 0.0 {
                    let rl = self.ref_logprobs.as_ref().map(|f| f(&task, &c));
                    self.pending.push_back(self.pack(&task.prompt, &c, 1.0, rl.as_deref()));
                    self.last_mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
                    self.last_kept_frac = 1.0;
                    return;
                }
            }
            self.last_mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
            self.last_kept_frac = 0.0;
        } else {
            let group = roll.sample_n(&task.prompt, self.cfg.group_size, &self.cfg.rollout, rng);
            for c in &group {
                self.log.push_sampled(c.tokens.clone());
            }
            let rewards: Vec<f32> = group.iter().map(|c| self.verifier.verify(&task, &[], &c.tokens).value).collect();
            self.last_mean_reward = rewards.iter().sum::<f32>() / rewards.len().max(1) as f32;
            let advs = group_advantages(&rewards);
            self.last_kept_frac = if advs.iter().any(|&a| a != 0.0) { 1.0 } else { 0.0 };
            for (c, &a) in group.iter().zip(&advs) {
                if a != 0.0 {
                    let rl = self.ref_logprobs.as_ref().map(|f| f(&task, c));
                    self.pending.push_back(self.pack(&task.prompt, c, a, rl.as_deref()));
                }
            }
        }
    }
}

impl<M: Model, E: Environment, V: Verifier> Objective<M> for Grpo<E, V> {
    fn regime(&self) -> &'static str {
        "grpo"
    }

    fn prepare(&mut self, model: &mut M) {
        model.enable_weighted_loss();
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        assert!(
            self.cfg.kl_beta <= 0.0 || self.ref_logprobs.is_some(),
            "Grpo: kl_beta > 0.0 requires with_reference() to have been called"
        );
        if self.pending.is_empty() {
            self.refill(model, rng);
        }
        let Some(row) = self.pending.pop_front() else {
            // Every completion this round was dropped (STaR found nothing
            // correct, or GRPO's whole group was zero-variance) - a
            // legitimate no-signal micro-step, not an error.
            return 0.0;
        };
        self.log.push_trained(row.completion.clone());

        // One ordinary forward (reads the CURRENT policy's per-token
        // logprobs) ...
        model.set_batch(Batch::Lm { tokens: &row.tokens, targets: &row.targets });
        let _ = model.forward();
        let new_lp = model.batch_token_logprobs().expect("Grpo: model must implement Model::batch_token_logprobs");

        // ... a host-side weight/loss computation from THIS module's own
        // token_term (the exact function the gate's CheckModel harness also
        // calls) ...
        let n = row.tokens.len();
        let mut weights = vec![0f32; n];
        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        for i in 0..n {
            if row.targets[i] == IGNORE {
                continue;
            }
            count += 1;
            let rl = if self.cfg.kl_beta > 0.0 { Some(row.ref_lp[i]) } else { None };
            let term = token_term(row.old_lp[i], new_lp[i], row.advantage[i], self.cfg.clip_eps, rl, self.cfg.kl_beta);
            weights[i] = term.weight;
            total_loss += term.loss;
        }

        // ... then one ordinary backward, weighted by that computation.
        model.set_loss_weights(&weights);
        model.backward();

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        vec![("grpo_mean_reward", self.last_mean_reward), ("grpo_kept_group_frac", self.last_kept_frac)]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cycle_log_clones_share_the_same_underlying_log() {
        // The whole point of `CycleLog` (self-improve roadmap P18): a
        // caller keeps a clone before handing the original into something
        // that consumes it by value (`model::fit_with`'s own `Objective`
        // parameter), and still sees every push made through the moved
        // clone afterward.
        let log = super::CycleLog::new();
        let handle = log.clone();
        log.push_sampled(vec![1, 2, 3]);
        log.push_sampled(vec![4, 5]);
        log.push_trained(vec![1, 2, 3]);
        drop(log);

        assert_eq!(handle.sampled(), vec![vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(handle.trained(), vec![vec![1, 2, 3]]);
    }

    use super::*;

    #[test]
    fn group_advantages_z_scores_a_real_group() {
        let a = group_advantages(&[1.0, 0.0]);
        assert_eq!(a.len(), 2);
        assert!(a[0] > 0.9 && a[0] < 1.1, "{a:?}");
        assert!(a[1] < -0.9 && a[1] > -1.1, "{a:?}");
    }

    #[test]
    fn group_advantages_drops_a_zero_variance_group() {
        assert_eq!(group_advantages(&[1.0, 1.0, 1.0]), vec![0.0, 0.0, 0.0]);
        assert_eq!(group_advantages(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn group_advantages_size_one_is_uniform_not_zscored() {
        // The RFT/STaR degenerate case - see module doc comment. GRPO's own
        // z-score would divide by ~0 std and collapse to 0, which is wrong
        // here: a single already-verified-correct completion must train.
        assert_eq!(group_advantages(&[1.0]), vec![1.0]);
        assert_eq!(group_advantages(&[]), Vec::<f32>::new());
    }

    #[test]
    fn token_term_unclipped_positive_advantage_is_advantage_times_ratio() {
        // ratio = exp(0) = 1, inside [0.8, 1.2] - not clipped.
        let t = token_term(0.0, 0.0, 2.0, 0.2, None, 0.0);
        assert!((t.weight - 2.0).abs() < 1e-6, "{t:?}");
        assert!((t.loss - -2.0).abs() < 1e-6, "{t:?}");
    }

    #[test]
    fn token_term_clips_a_large_ratio_with_positive_advantage_to_zero_weight() {
        // ratio = exp(ln 1.5) = 1.5 > 1 + eps(0.2) = 1.2, advantage > 0: the
        // classic PPO "ran too far in the rewarded direction" clip.
        let t = token_term(0.0, 1.5_f32.ln(), 1.0, 0.2, None, 0.0);
        assert_eq!(t.weight, 0.0, "{t:?}");
        assert!((t.loss - -1.2).abs() < 1e-5, "{t:?}");
    }

    #[test]
    fn token_term_does_not_clip_a_large_ratio_with_negative_advantage() {
        // Same ratio, negative advantage: pushing further into an already
        // over-large ratio is exactly what a negative advantage should keep
        // penalizing - not clipped (see module doc comment's derivation).
        let t = token_term(0.0, 1.5_f32.ln(), -1.0, 0.2, None, 0.0);
        assert!((t.weight - -1.5).abs() < 1e-5, "{t:?}");
        assert!((t.loss - 1.5).abs() < 1e-5, "{t:?}");
    }

    #[test]
    fn token_term_clips_a_small_ratio_with_negative_advantage_to_zero_weight() {
        // ratio = exp(ln 0.5) = 0.5 < 1 - eps(0.2) = 0.8, advantage < 0.
        let t = token_term(0.0, 0.5_f32.ln(), -1.0, 0.2, None, 0.0);
        assert_eq!(t.weight, 0.0, "{t:?}");
        // surrogate = min(-0.5, -0.8) = -0.8, loss = -surrogate = 0.8.
        assert!((t.loss - 0.8).abs() < 1e-5, "{t:?}");
    }

    #[test]
    fn token_term_kl_applies_even_at_zero_advantage() {
        // advantage == 0.0 (e.g. a completion whose reward landed exactly on
        // its group's mean) must still carry the KL regularizer - it does
        // not depend on the sample's own reward.
        let ref_lp: f32 = -0.5;
        let new_lp: f32 = -0.2;
        let u = ref_lp - new_lp;
        let eu = u.exp();
        let t = token_term(-0.3, new_lp, 0.0, 0.2, Some(ref_lp), 0.5);
        assert!((t.weight - 0.5 * (eu - 1.0)).abs() < 1e-6, "{t:?}");
        assert!((t.loss - 0.5 * (eu - u - 1.0)).abs() < 1e-6, "{t:?}");
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

    fn completion(tokens: Vec<u32>, logprobs: Vec<f32>) -> Completion {
        Completion { tokens, logprobs, stop: model::rollout::StopReason::MaxNew }
    }

    #[test]
    fn pack_row_aligns_targets_and_old_logprobs_to_the_completion_span() {
        let seq_len = 8;
        let mut tokens = vec![0u32; seq_len];
        let mut targets = vec![IGNORE; seq_len];
        let mut old_lp = vec![0f32; seq_len];
        let mut ref_lp = vec![0f32; seq_len];
        let mut advantage = vec![0f32; seq_len];
        let prompt = [1u32, 2, 3];
        let c = completion(vec![4, 5, 6], vec![-0.1, -0.2, -0.3]);

        pack_row(&mut tokens, &mut targets, &mut old_lp, &mut ref_lp, &mut advantage, 0, seq_len, &prompt, &c, 1.5, Some(&[-0.4, -0.5, -0.6]));

        assert_eq!(tokens, vec![1, 2, 3, 4, 5, 6, 0, 0]);
        // Position 2 (prompt.len()-1) predicts completion[0]=4, position 3
        // predicts completion[1]=5, position 4 predicts completion[2]=6.
        assert_eq!(targets, vec![IGNORE, IGNORE, 4, 5, 6, IGNORE, IGNORE, IGNORE]);
        assert_eq!(old_lp[2..5], [-0.1, -0.2, -0.3]);
        assert_eq!(ref_lp[2..5], [-0.4, -0.5, -0.6]);
        assert_eq!(advantage[2..5], [1.5, 1.5, 1.5]);
        assert_eq!(advantage[0], 0.0);
    }

    #[test]
    fn pack_row_truncates_a_completion_that_overruns_seq_len() {
        let seq_len = 4;
        let mut tokens = vec![0u32; seq_len];
        let mut targets = vec![IGNORE; seq_len];
        let mut old_lp = vec![0f32; seq_len];
        let mut ref_lp = vec![0f32; seq_len];
        let mut advantage = vec![0f32; seq_len];
        let prompt = [1u32, 2];
        let c = completion(vec![3, 4, 5, 6], vec![-0.1, -0.2, -0.3, -0.4]);

        pack_row(&mut tokens, &mut targets, &mut old_lp, &mut ref_lp, &mut advantage, 0, seq_len, &prompt, &c, 1.0, None);

        // Only position 1 (predicts completion[0]=3) and position 2
        // (predicts completion[1]=4) fit in seq_len=4; completion[2..] is
        // dropped, not scored.
        assert_eq!(tokens, vec![1, 2, 3, 4]);
        assert_eq!(targets, vec![IGNORE, 3, 4, IGNORE]);
    }
}
