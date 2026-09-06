// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DPO: direct preference optimization over a chosen/rejected pair packed
//! as the two rows of ONE `b=2` batch - one forward, one backward per pair,
//! never four.
//!
//! Swedish Embedded AB builds the training-regime machinery that turns a
//! verifier-derived preference pair into a trained model without a second
//! (four-forward/four-backward) implementation next to ordinary weighted
//! cross-entropy. If your team needs preference optimization on top of a
//! from-scratch training stack, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! ## The reduction to a per-token `Batch::LmWeighted` weight
//!
//! DPO's loss for one pair is `L = -log sigma(u)`, `u = beta * ((logpi_c -
//! logref_c) - (logpi_r - logref_r))`, where `logpi_c`/`logpi_r` are the
//! CURRENT policy's total log-probability of the chosen/rejected completion
//! (summed over that completion's own token positions) and `logref_c`/
//! `logref_r` are the same sums under a FROZEN reference policy, precomputed
//! once per pair. A model's per-token `log p_theta(target)` is exactly the
//! negation of the per-position CE loss its own loss kernel already writes
//! ([`model::Model::batch_token_logprobs`]), so `logpi_c = -sum(CE_t for t
//! in chosen row)` and likewise for the rejected row.
//!
//! `dL/d(logpi_c)` equals `-sigma(-u) * beta` and `dL/d(logpi_r)` equals
//! `+sigma(-u) * beta` (differentiate `-log sigma(u)` through the sign each
//! term enters `u` with). Since `logpi_c = -sum CE_t`, `dL/d(CE_t) = beta *
//! sigma(-u)` for every active position `t` of the chosen row, and
//! `dL/d(CE_t) = -beta * sigma(-u)` for every active position of the
//! rejected row: the SAME scalar for every token in a row, because `u`
//! depends on each row only through its total log-probability.
//!
//! `CE_GRAD_STATS` writes the unweighted per-token gradient as `(softmax(z)
//! minus onehot(y)) / C` (`C` = the whole forward's active-position count,
//! both rows), and [`model::lossw::WeightedCe`] scales that per row by a
//! weight `w_i` before backward reads it: so to realize `dL/d(CE_t) =
//! +-beta * sigma(-u)` exactly, `w_i / C` must equal `+-beta * sigma(-u)`,
//! i.e. `w_i` equals `+-beta * sigma(-u) * C`, chosen rows getting `+` and
//! rejected rows getting `-`. [`pair_term`] computes exactly this pair
//! (`weight_chosen`,
//! `weight_rejected`) plus the scalar loss it is the analytic gradient of,
//! from one ordinary forward's freshly read-back per-token logprobs - so
//! [`Dpo::micro_step`] differs from a plain weighted-CE step only in WHERE
//! the weight comes from: one forward (to read `new_lp` for both rows), a
//! host-side `pair_term` call, [`model::Model::set_loss_weights`], one
//! backward. No second (four-pass) implementation is needed.
//!
//! ## Why pairs are supplied, not sampled in-line
//!
//! Every current weighted-loss-capable [`model::Model::logits_all`] impl
//! asserts a single-sequence (`b == 1`) shape, because [`model::rollout`]
//! re-prefills per sample - but a DPO training step needs a `b == 2`
//! forward (the whole point of packing the pair into one batch). A `Model`
//! is constructed once at a fixed `(b, t)`, so the SAME instance that
//! trains a DPO pair cannot also be the `b == 1` oracle a completion group
//! is rolled out from - unlike GRPO (P12), which sidesteps this by training
//! one row (`b == 1`) at a time, DPO cannot: it structurally needs both
//! rows in the same forward. [`Dpo`] therefore takes its verifier-derived
//! pairs from an externally supplied source (a plain closure, or
//! [`Dpo::from_pairs`]'s fixed/cycling dataset) rather than owning a live
//! [`model::rollout::Rollout`] against its own training model - the
//! sampling and verification (whoever performs it: a `b == 1` snapshot of
//! the same architecture, or an offline collection pass) happen upstream of
//! this file, which is exactly where [`pairs_from_group`] plugs in: it
//! turns one already-sampled, already-[`crate::env::Verifier`]-scored P12-
//! style group into the (at most one) contrasting pair DPO trains on -
//! never by asking a model to judge which completion is better.

use std::collections::VecDeque;

use data::rng::Rng;
use model::rollout::Completion;
use model::{Batch, Model, Objective, IGNORE};

/// One verifier-derived preference pair for a single prompt: `chosen` is a
/// verified-correct completion, `rejected` a verified-incorrect one from
/// the same sampled group (see [`pairs_from_group`]) - never a judge
/// model's output.
#[derive(Clone, Debug, PartialEq)]
pub struct DpoPair {
    pub prompt: Vec<u32>,
    pub chosen: Vec<u32>,
    pub rejected: Vec<u32>,
}

/// Build the (at most one) DPO pair from one P12-style sampled group:
/// `completions` (with their per-completion `rewards`, from a
/// [`crate::env::Verifier`]) drawn for the same `prompt`. Pairs the FIRST
/// verified-correct (`reward > 0.0`) completion with the FIRST
/// verified-incorrect (`reward <= 0.0`) one - STaR-style dedup (at most one
/// pair, not a combinatorial cross product of every correct against every
/// incorrect completion) - and returns empty when the group has no
/// contrast (all-correct or all-incorrect groups carry no preference
/// signal, the same "a group that teaches nothing costs nothing"
/// discipline [`crate::objective::grpo::group_advantages`] applies to
/// zero-variance groups).
pub fn pairs_from_group(prompt: &[u32], completions: &[Completion], rewards: &[f32]) -> Vec<DpoPair> {
    assert_eq!(completions.len(), rewards.len(), "pairs_from_group: one reward per completion");
    let chosen = completions.iter().zip(rewards.iter()).find(|&(_, &r)| r > 0.0).map(|(c, _)| c);
    let rejected = completions.iter().zip(rewards.iter()).find(|&(_, &r)| r <= 0.0).map(|(c, _)| c);
    match (chosen, rejected) {
        (Some(c), Some(r)) => vec![DpoPair { prompt: prompt.to_vec(), chosen: c.tokens.clone(), rejected: r.tokens.clone() }],
        _ => Vec::new(),
    }
}

/// DPO's hyperparameters. `seq_len` is ONE row's packed length - the model
/// this objective trains MUST be constructed with `b = 2`, `t = seq_len`
/// (see the module doc comment on why both rows share one forward).
#[derive(Clone, Copy, Debug)]
pub struct DpoConfig {
    /// The DPO temperature `beta` scaling the (reference-normalized)
    /// log-probability margin `u`.
    pub beta: f32,
    pub seq_len: usize,
}

/// One pair's [`model::Batch::LmWeighted`] weights (applied uniformly
/// across each row's own active positions - see the module doc comment for
/// why the weight does not vary by token within a row), plus the
/// host-computed loss they are the analytic gradient of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairTerm {
    pub weight_chosen: f32,
    pub weight_rejected: f32,
    pub loss: f32,
}

/// Numerically stable logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Numerically stable `ln(1 + e^x)`.
fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// The DPO weight/loss pair for one packed pair, derived (not looked up)
/// from `L = -log sigma(u)` - see the module doc comment for the full
/// derivation. `count` is the SAME whole-forward active-position count
/// (both rows) the model's own weighted-CE division uses.
pub fn pair_term(beta: f32, sum_new_chosen: f32, sum_ref_chosen: f32, sum_new_rejected: f32, sum_ref_rejected: f32, count: f32) -> PairTerm {
    let u = beta * ((sum_new_chosen - sum_ref_chosen) - (sum_new_rejected - sum_ref_rejected));
    let loss = softplus(-u);
    let w = beta * sigmoid(-u) * count;
    PairTerm { weight_chosen: w, weight_rejected: -w, loss }
}

/// Sum `new_lp`/`ref_lp` (and count active positions) over row `row`'s
/// `seq_len`-wide span of a packed `2 * seq_len` batch - the same reduction
/// [`Dpo::micro_step`] and its gradcheck gate both perform on a freshly
/// forwarded model's per-token logprobs.
pub fn row_sum(new_lp: &[f32], ref_lp: &[f32], targets: &[u32], row: usize, seq_len: usize) -> (f32, f32, usize) {
    let base = row * seq_len;
    let mut sum_new = 0.0f32;
    let mut sum_ref = 0.0f32;
    let mut count = 0usize;
    for i in 0..seq_len {
        if targets[base + i] != IGNORE {
            sum_new += new_lp[base + i];
            sum_ref += ref_lp[base + i];
            count += 1;
        }
    }
    (sum_new, sum_ref, count)
}

/// Pack one row (`prompt` followed by `completion`) into row `row` of a
/// `2 * seq_len`-wide batch, at the same "targets[t] predicts position
/// t+1" alignment [`crate::objective::grpo::pack_row`] uses: position
/// `prompt.len() - 1 + i` predicts `completion[i]`. A completion that
/// overruns `seq_len` is silently truncated at the tail.
fn pack_row(tokens: &mut [u32], targets: &mut [u32], ref_lp: &mut [f32], row: usize, seq_len: usize, prompt: &[u32], completion: &[u32], ref_logprobs: &[f32]) {
    let base = row * seq_len;
    let plen = prompt.len();
    if plen == 0 {
        // Not a real scenario (every task prompt is non-empty) - guards the
        // `tok_idx - 1` underflow below rather than relying on callers.
        return;
    }
    for (i, &t) in prompt.iter().chain(completion.iter()).take(seq_len).enumerate() {
        tokens[base + i] = t;
    }
    for i in 0..completion.len() {
        let tok_idx = plen + i;
        if tok_idx >= seq_len {
            break;
        }
        let t_pos = tok_idx - 1;
        targets[base + t_pos] = completion[i];
        ref_lp[base + t_pos] = ref_logprobs[i];
    }
}

/// One packed, reference-scored pair queued for a future [`Dpo::micro_step`]
/// call.
struct PackedPair {
    tokens: Vec<u32>,
    targets: Vec<u32>,
    ref_lp: Vec<f32>,
}

fn pack_pair(cfg: &DpoConfig, pair: &DpoPair, chosen_ref: &[f32], rejected_ref: &[f32]) -> PackedPair {
    let seq_len = cfg.seq_len;
    let mut tokens = vec![0u32; 2 * seq_len];
    let mut targets = vec![IGNORE; 2 * seq_len];
    let mut ref_lp = vec![0f32; 2 * seq_len];
    pack_row(&mut tokens, &mut targets, &mut ref_lp, 0, seq_len, &pair.prompt, &pair.chosen, chosen_ref);
    pack_row(&mut tokens, &mut targets, &mut ref_lp, 1, seq_len, &pair.prompt, &pair.rejected, rejected_ref);
    PackedPair { tokens, targets, ref_lp }
}

/// Where [`Dpo`] draws its next verifier-derived pair from - a plain
/// closure so the pairing mechanism (live rollout + [`pairs_from_group`],
/// or a fixed offline dataset via [`Dpo::from_pairs`]) stays a caller
/// concern, not this file's (see the module doc comment on why DPO cannot
/// own its own live rollout the way GRPO does). `None` means "nothing
/// available this round" - a legitimate no-op micro-step.
type PairSource = Box<dyn FnMut(&mut Rng) -> Option<DpoPair>>;

/// The frozen-reference logprob source: per-token `log pi_ref(token)` for
/// the chosen and rejected completions of one pair, called ONCE per pair
/// (when it is queued) and cached in the resulting [`PackedPair`] - never
/// recomputed at every training step.
type RefLogprobFn = Box<dyn Fn(&DpoPair) -> (Vec<f32>, Vec<f32>)>;

/// DPO training objective: packs each verifier-derived [`DpoPair`] as the
/// two rows of one `b = 2` batch and runs exactly one forward and one
/// backward per pair. Generic over any `M: model::Model` that has adopted
/// weighted-loss support ([`model::Model::enable_weighted_loss`]/
/// [`model::Model::set_loss_weights`]/[`model::Model::batch_token_logprobs`])
/// - today `qwen3::Qwen` and `gpt2::Gpt`.
pub struct Dpo {
    cfg: DpoConfig,
    source: PairSource,
    ref_logprobs: RefLogprobFn,
    pending: VecDeque<PackedPair>,
    last_margin: f32,
}

impl Dpo {
    /// `source` supplies the next verifier-derived pair (or `None` for a
    /// no-signal round); `ref_logprobs` computes the frozen reference's
    /// per-token logprobs for a pair's chosen/rejected completions, called
    /// once per pair and cached - see this struct's own field doc comments.
    pub fn new<S, R>(cfg: DpoConfig, source: S, ref_logprobs: R) -> Dpo
    where
        S: FnMut(&mut Rng) -> Option<DpoPair> + 'static,
        R: Fn(&DpoPair) -> (Vec<f32>, Vec<f32>) + 'static,
    {
        Dpo { cfg, source: Box::new(source), ref_logprobs: Box::new(ref_logprobs), pending: VecDeque::new(), last_margin: 0.0 }
    }

    /// Convenience over [`Self::new`]: a fixed, cycling dataset of
    /// already-verifier-derived pairs - the common case for a small
    /// synthetic pair set or a pre-collected offline preference dataset,
    /// rather than a live pairing source.
    pub fn from_pairs<R>(cfg: DpoConfig, pairs: Vec<DpoPair>, ref_logprobs: R) -> Dpo
    where
        R: Fn(&DpoPair) -> (Vec<f32>, Vec<f32>) + 'static,
    {
        assert!(!pairs.is_empty(), "Dpo::from_pairs: need at least one pair");
        let mut idx = 0usize;
        Dpo::new(
            cfg,
            move |_: &mut Rng| {
                let p = pairs[idx].clone();
                idx = (idx + 1) % pairs.len();
                Some(p)
            },
            ref_logprobs,
        )
    }
}

impl<M: Model> Objective<M> for Dpo {
    fn regime(&self) -> &'static str {
        "dpo"
    }

    fn prepare(&mut self, model: &mut M) {
        model.enable_weighted_loss();
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        if self.pending.is_empty() {
            let Some(pair) = (self.source)(rng) else {
                // No pair available this round - a legitimate no-op
                // micro-step, matching GRPO's own "every completion this
                // round was dropped" convention.
                return 0.0;
            };
            let (chosen_ref, rejected_ref) = (self.ref_logprobs)(&pair);
            self.pending.push_back(pack_pair(&self.cfg, &pair, &chosen_ref, &rejected_ref));
        }
        let packed = self.pending.pop_front().expect("just ensured pending is non-empty");

        // One ordinary forward (reads the CURRENT policy's per-token
        // logprobs for BOTH rows) ...
        model.set_batch(Batch::Lm { tokens: &packed.tokens, targets: &packed.targets });
        let _ = model.forward();
        let new_lp = model.batch_token_logprobs().expect("Dpo: model must implement Model::batch_token_logprobs");

        // ... a host-side weight/loss computation from this module's own
        // pair_term (the exact function the gate's CheckModel harness also
        // calls) ...
        let seq_len = self.cfg.seq_len;
        let (sum_new_chosen, sum_ref_chosen, count_chosen) = row_sum(&new_lp, &packed.ref_lp, &packed.targets, 0, seq_len);
        let (sum_new_rejected, sum_ref_rejected, count_rejected) = row_sum(&new_lp, &packed.ref_lp, &packed.targets, 1, seq_len);
        let count = (count_chosen + count_rejected).max(1) as f32;

        let term = pair_term(self.cfg.beta, sum_new_chosen, sum_ref_chosen, sum_new_rejected, sum_ref_rejected, count);
        self.last_margin = (sum_new_chosen - sum_ref_chosen) - (sum_new_rejected - sum_ref_rejected);

        let mut weights = vec![0f32; 2 * seq_len];
        for i in 0..seq_len {
            if packed.targets[i] != IGNORE {
                weights[i] = term.weight_chosen;
            }
            if packed.targets[seq_len + i] != IGNORE {
                weights[seq_len + i] = term.weight_rejected;
            }
        }

        // ... then one ordinary backward, weighted by that computation.
        model.set_loss_weights(&weights);
        model.backward();

        term.loss
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        vec![("dpo_margin", self.last_margin)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion(tokens: &[u32]) -> Completion {
        Completion { tokens: tokens.to_vec(), logprobs: vec![0.0; tokens.len()], stop: model::rollout::StopReason::MaxNew }
    }

    #[test]
    fn pairs_from_group_pairs_first_correct_with_first_incorrect() {
        let prompt = [1, 2, 3];
        let completions = [completion(&[4, 5]), completion(&[6, 7]), completion(&[8, 9])];
        let rewards = [0.0, 1.0, 0.0];
        let pairs = pairs_from_group(&prompt, &completions, &rewards);
        assert_eq!(pairs.len(), 1, "exactly one dedup'd pair, not a cross product");
        assert_eq!(pairs[0].chosen, vec![6, 7]);
        assert_eq!(pairs[0].rejected, vec![4, 5]);
    }

    #[test]
    fn pairs_from_group_drops_a_group_with_no_contrast() {
        let prompt = [1, 2, 3];
        let all_correct = [completion(&[4, 5]), completion(&[6, 7])];
        assert!(pairs_from_group(&prompt, &all_correct, &[1.0, 1.0]).is_empty());
        let all_incorrect = [completion(&[4, 5]), completion(&[6, 7])];
        assert!(pairs_from_group(&prompt, &all_incorrect, &[0.0, 0.0]).is_empty());
    }

    #[test]
    fn pair_term_rewards_chosen_and_penalizes_rejected_when_chosen_is_already_preferred() {
        // Chosen already scores higher than reference, rejected lower: u > 0,
        // sigma(-u) < 0.5, but still positive - both directions still push
        // (DPO never fully stops pushing at finite margin).
        let t = pair_term(1.0, 2.0, 0.0, -2.0, 0.0, 10.0);
        assert!(t.weight_chosen > 0.0, "{t:?}");
        assert!(t.weight_rejected < 0.0, "{t:?}");
        assert!((t.weight_chosen + t.weight_rejected).abs() < 1e-6, "chosen/rejected weights must be exact opposites: {t:?}");
        assert!(t.loss > 0.0 && t.loss < 0.7, "{t:?}");
    }

    #[test]
    fn pair_term_pushes_harder_when_rejected_is_currently_preferred() {
        // u < 0 here (rejected currently scores higher than chosen relative
        // to reference) - sigma(-u) > 0.5, so the pull should be stronger
        // than the already-preferred case above.
        let already_preferred = pair_term(1.0, 2.0, 0.0, -2.0, 0.0, 10.0);
        let currently_wrong = pair_term(1.0, -2.0, 0.0, 2.0, 0.0, 10.0);
        assert!(currently_wrong.weight_chosen > already_preferred.weight_chosen, "{currently_wrong:?} vs {already_preferred:?}");
        assert!(currently_wrong.loss > already_preferred.loss, "{currently_wrong:?} vs {already_preferred:?}");
    }

    #[test]
    fn row_sum_reduces_only_active_positions_of_the_given_row() {
        let seq_len = 4;
        let new_lp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ref_lp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let targets = [IGNORE, 9, 9, IGNORE, 9, 9, 9, IGNORE];
        let (sum_new, sum_ref, count) = row_sum(&new_lp, &ref_lp, &targets, 0, seq_len);
        assert_eq!(count, 2);
        assert!((sum_new - 5.0).abs() < 1e-6); // 2.0 + 3.0
        assert!((sum_ref - 0.5).abs() < 1e-6); // 0.2 + 0.3

        let (sum_new, sum_ref, count) = row_sum(&new_lp, &ref_lp, &targets, 1, seq_len);
        assert_eq!(count, 3);
        assert!((sum_new - 18.0).abs() < 1e-6); // 5.0 + 6.0 + 7.0
        assert!((sum_ref - 1.8).abs() < 1e-6); // 0.5 + 0.6 + 0.7
    }
}
