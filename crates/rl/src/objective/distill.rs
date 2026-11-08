// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DistillTopK: top-K knowledge distillation as literally `K` weighted-CE
//! passes over ONE model - zero new kernels, zero new gradient path beyond
//! the same `Batch::LmWeighted`/`scale_row.wgsl` primitive DPO and GRPO
//! already gradcheck.
//!
//! Swedish Embedded AB builds the training-regime machinery that turns a
//! teacher's top-K token distribution into a trained student without a
//! second, dense-softmax-gradient kernel next to ordinary cross-entropy. If
//! your team needs knowledge distillation on top of a from-scratch training
//! stack, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! ## The reduction to `K` per-token `Batch::LmWeighted` weights
//!
//! Forward KL distillation trains the student's softmax `p = softmax(z)`
//! toward a (frozen) teacher distribution `q` by minimizing `KL(q‖p) = Σ_v
//! q_v·(ln q_v − ln p_v)`. Differentiating w.r.t. the student's logits `z`
//! (the `ln q_v` term is a constant w.r.t. `z`) gives the standard
//! softmax-cross-entropy-with-a-general-target-distribution result `d(KL)/dz
//! = p − q` - the same shape as ordinary one-hot cross-entropy's `p −
//! e_y`, just with a dense target instead of a one-hot one.
//!
//! Top-K distillation approximates the (often vocab-sized, expensive to
//! transmit) teacher distribution `q` by its `K` largest entries,
//! renormalized so they sum to 1.0: `q_K`, a sparse vector with `q_K[v_k] =
//! q_k` for the `K` kept ids and `0` elsewhere. Because `Σ_k q_k = 1` (the
//! renormalization), plain linearity gives:
//!
//! `p − q_K = p·(Σ_k q_k) − Σ_k q_k·e_{v_k} = Σ_k q_k·(p − e_{v_k})`
//!
//! Each `(p − e_{v_k})` term is EXACTLY `CE_GRAD_STATS`'s own unweighted
//! per-token gradient for one-hot target `v_k` (`softmax(z) −
//! onehot(v_k)`), and [`model::lossw::WeightedCe`] already scales that by
//! any per-row scalar before backward reads it. So realizing `d(KL)/dz =
//! p − q_K` needs no new kernel at all: run `K` ordinary
//! [`model::Batch::LmWeighted`] passes over the SAME tokens, pass `k` using
//! target `v_k` and weight `w_k = q_k·count_k / active` (`count_k` is the
//! model's own `/count` weighted-CE divisor for that pass, cancelled exactly
//! the way `rl::objective::dpo::pair_term`'s `w = β·σ(−u)·count` does, and
//! `active` the number of teacher-scored positions, which makes the loss a
//! mean), and let [`model::Model::backward`]'s documented
//! ACCUMULATE-not-overwrite contract sum the `K` passes' gradients: `Σ_k
//! (q_k/active)·(p − e_{v_k}) = (p − q_K)/active` per position. Unlike DPO's and
//! GRPO's weight formulas, this one needs no readback of the model's own
//! current predictions at all (`w_k` is pure teacher data) - simpler than
//! either, not just a third instance of the same trick.
//!
//! ## One normalisation convention
//!
//! An example's loss is the MEAN over its `active` teacher-scored positions of
//! `KL(q_K‖p)`, and the gradient written is the gradient of exactly that mean.
//! The examples of an optimiser step are averaged by the shared loop
//! (`grad_accum`), so an example counts once whatever its length. (An earlier
//! revision reported the mean but wrote `active` times its gradient.)
//!
//! The realized scalar loss composes the same way: with that weight, summing
//! the `K` passes' [`model::Model::forward`] returns `(1/active)·Σ_t Σ_k
//! q_k·(−ln p_{v_k}) = mean KL(q_K‖p) + mean H(q_K)` (`H` the teacher's own
//! entropy, a constant that does not depend on the student's parameters).
//! [`neg_entropy`] computes `−H(q_K) = Σ_k q_k·ln(q_k)` directly from the
//! teacher's own (data-only) `TopK`, and [`mean_kl`] adds its mean to the
//! summed pass losses to report the actual mean `KL(q_K‖p)` rather than the
//! teacher-entropy-shifted cross-entropy - this constant addition changes
//! nothing about the gradient, only what the reported metric means.
//!
//! ## Why `K = vocab` is not a different approximation
//!
//! Nothing above assumes `K < vocab`: at `K = vocab` (every id present, `q_K
//! = q`), the identity `p − q_K = Σ_k q_k·(p − e_{v_k})` is the ordinary
//! full-KL gradient by the exact same derivation, term for term - there is
//! no separate "dense" code path. `crates/rl/tests/distill_full_kl.rs` is
//! this phase's SEPARATE gate proving exactly that: `K = vocab` reproduces
//! the textbook `KL(q‖p)`, computed a second, independent way, to fp32
//! tolerance.

use std::collections::VecDeque;

use data::rng::Rng;
use super::stream::{self, IndexedStream};
use model::{Batch, Model, ModelConfig, Objective, StepVerdict, IGNORE};
use model::training::Unsupported;

/// One position's top-K teacher distribution: `ids[k]`/`probs[k]` pairs,
/// `probs` already renormalized so they sum to 1.0 (the keystone identity's
/// `Σ_k q_k = 1` requirement) - producing that renormalization from a raw
/// teacher distribution is the caller's job (e.g. a teacher model's own
/// top-K logits, softmax-renormalized over just those K), not this file's.
/// `K` may vary by position (e.g. a short completion tail naturally carries
/// fewer teacher candidates in some capture pipelines) - [`pass_arrays`]
/// treats a position with fewer than `k+1` entries as inactive for pass `k`,
/// not an error. `Default` (empty `ids`/`probs`) means "no teacher signal at
/// this position" - a prompt token that is never distilled.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TopK {
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
}

impl TopK {
    /// The `k` largest entries of one position's teacher `logits`, softmaxed
    /// over the whole row (so the probabilities are the teacher's) and then
    /// renormalized over the kept entries. Ties keep the lower id. `k` is
    /// clamped to the row width.
    ///
    /// # Panics
    /// If `k` is zero or `logits` is empty or holds a non-finite value: such a distribution
    /// carries no information and would train nothing.
    pub fn from_logits(logits: &[f32], k: usize) -> TopK {
        assert!(k >= 1 && !logits.is_empty(), "TopK::from_logits: need k >= 1 and a non-empty row (k {k}, {} logits)", logits.len());
        assert!(logits.iter().all(|l| l.is_finite()), "TopK::from_logits: the teacher's logits must be finite");
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b)));
        order.truncate(k);
        let kept: Vec<f32> = order.iter().map(|&i| (logits[i as usize] - max).exp()).collect();
        let sum: f32 = kept.iter().sum();
        TopK { probs: kept.iter().map(|p| p / sum).collect(), ids: order }
    }
}

impl TopK {
    /// Whether this is a usable teacher distribution: as many probabilities as ids, no id twice,
    /// every probability finite and non-negative, and (unless empty, which marks a position
    /// without teacher signal) summing to one - the keystone identity needs `sum_k q_k = 1`.
    pub fn validate(&self) -> Result<(), String> {
        if self.ids.len() != self.probs.len() {
            return Err(format!("{} ids but {} probabilities", self.ids.len(), self.probs.len()));
        }
        let mut seen = std::collections::HashSet::new();
        if let Some(id) = self.ids.iter().find(|&&id| !seen.insert(id)) {
            return Err(format!("token id {id} listed twice"));
        }
        if self.probs.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err("a probability is negative or not finite".to_string());
        }
        let sum: f32 = self.probs.iter().sum();
        if !self.ids.is_empty() && (sum - 1.0).abs() > 1e-3 {
            return Err(format!("probabilities sum to {sum}, not 1 (renormalise over the kept ids)"));
        }
        Ok(())
    }
}

/// One example DistillTopK trains on: `prompt` tokens (never scored, never
/// carry a `TopK`) followed by `completion` tokens, with `teacher[i]` the
/// top-K teacher distribution for the position that predicts
/// `completion[i]` (`teacher.len()` must equal `completion.len()`).
#[derive(Clone, Debug, PartialEq)]
pub struct DistillExample {
    pub prompt: Vec<u32>,
    pub completion: Vec<u32>,
    pub teacher: Vec<TopK>,
}

/// DistillTopK's only hyperparameter: the packed row length (`t`) the model
/// this objective trains MUST be constructed with, at `b = 1` (see the
/// module doc comment on why, unlike DPO/GRPO, no `b > 1` packing is ever
/// needed here - the weight formula never reads the model's own
/// predictions).
#[derive(Clone, Copy, Debug)]
pub struct DistillConfig {
    pub seq_len: usize,
}

impl DistillConfig {
    /// Refuse a row too short to hold a prompt token and a distilled one.
    pub fn validate(&self) -> Result<(), String> {
        if self.seq_len < 2 {
            return Err(format!("seq_len {} leaves no position that predicts a token", self.seq_len));
        }
        Ok(())
    }
}

impl model::training_plan::PlanCheck for DistillConfig {
    fn plan_name(&self) -> &'static str {
        "distillation"
    }

    fn check(&self, opts: &model::FitOpts) -> Result<(), String> {
        self.validate()?;
        if self.seq_len != opts.block_size as usize {
            return Err(format!("rows are packed at seq_len {} but the model is built for the block size {}", self.seq_len, opts.block_size));
        }
        Ok(())
    }
}

/// Pass `k`'s weight for one position, scaled by `count` (the model's own `/count` weighted-CE
/// divisor of that pass, cancelled exactly - see the module doc comment's derivation) and divided
/// by `active`, the number of teacher-scored positions, so the loss is their mean. `prob` is pure
/// teacher data, no current-policy readback needed (unlike `rl::objective::dpo::pair_term`/
/// `rl::objective::grpo::token_term`, both of which DO need it).
pub fn topk_weight(prob: f32, count: f32, active: f32) -> f32 {
    prob * count / active
}

/// `Σ_k q_k·ln(q_k)` (`0` contributes `0`, matching the standard `0·ln 0 :=
/// 0` entropy convention) - the negative of the teacher's own entropy
/// `H(q_K)`, added to the summed weighted-CE pass losses to report the true
/// `KL(q_K‖p)` rather than the teacher-entropy-shifted cross-entropy (see
/// the module doc comment's derivation). Depends only on the teacher's own
/// (data-only) distribution, never on the student.
pub fn neg_entropy(topk: &TopK) -> f32 {
    topk.probs.iter().map(|&q| if q > 0.0 { q * q.ln() } else { 0.0 }).sum()
}

/// The positions of `teacher` that carry a distribution (at least one), the divisor of the mean (never zero).
fn active_positions(teacher: &[TopK]) -> f32 {
    teacher.iter().filter(|t| !t.ids.is_empty()).count().max(1) as f32
}

/// The mean `KL(q_K‖p)` over the teacher-scored positions of one example, from the sum of its `K`
/// passes' forward losses (`pass_losses`, already divided by the active positions through the
/// weights): the passes give the mean cross-entropy, the teacher's own mean entropy completes it.
pub fn mean_kl(pass_losses: f32, teacher: &[TopK]) -> f32 {
    pass_losses + teacher.iter().map(neg_entropy).sum::<f32>() / active_positions(teacher)
}

/// Build pass `k`'s [`model::Batch::LmWeighted`] `targets`/`weights` arrays
/// (length `teacher.len()`) plus that pass's own active-position `count`,
/// from one packed row's per-position top-K teacher distributions. A
/// position with fewer than `k + 1` teacher entries (either genuinely
/// inactive, or this pass's `k` exceeds that position's own `K`) gets
/// `IGNORE`/`0.0` - "masked-out and reward=0 collapse to the same thing",
/// the same convention `rl::objective::grpo`'s own weighted-CE positions
/// already follow. `count` is THIS pass's own active-position count (not a
/// single shared count across all `K` passes): since [`topk_weight`] scales
/// by exactly the count the model's own `/count` division will use for THIS
/// SAME pass, ragged per-position `K` needs no special-casing - each pass
/// independently cancels its own divisor.
pub fn pass_arrays(teacher: &[TopK], k: usize) -> (Vec<u32>, Vec<f32>, f32) {
    let count = teacher.iter().filter(|t| k < t.ids.len()).count().max(1) as f32;
    let active = active_positions(teacher);
    let mut targets = vec![IGNORE; teacher.len()];
    let mut weights = vec![0.0f32; teacher.len()];
    for (i, t) in teacher.iter().enumerate() {
        if let (Some(&id), Some(&p)) = (t.ids.get(k), t.probs.get(k)) {
            targets[i] = id;
            weights[i] = topk_weight(p, count, active);
        }
    }
    (targets, weights, count)
}

/// One packed, teacher-scored example queued for a future
/// [`DistillTopK::micro_step`] call.
struct PackedExample {
    tokens: Vec<u32>,
    /// Length `seq_len`; `TopK::default()` at every position not predicting
    /// a completion token (the prompt span, and anything past `seq_len`).
    teacher: Vec<TopK>,
}

/// Pack one `example` (`prompt` followed by `completion`) into a
/// `seq_len`-wide row, at the same "targets[t] predicts position t+1"
/// alignment `rl::objective::grpo::pack_row`/`rl::objective::dpo::pack_row`
/// both use: position `prompt.len() - 1 + i` predicts `completion[i]`, so
/// `teacher[i]`'s distribution is placed at that same position. An example that
/// does not fit the row, has no prompt, carries a malformed teacher
/// distribution, or has no teacher signal at all is refused, never truncated or
/// skipped.
fn pack_row(seq_len: usize, vocab: Option<u32>, prompt: &[u32], completion: &[u32], teacher: &[TopK]) -> Result<PackedExample, String> {
    if completion.len() != teacher.len() {
        return Err(format!("{} completion tokens but {} teacher distributions", completion.len(), teacher.len()));
    }
    if prompt.is_empty() {
        return Err("the example has no prompt".to_string());
    }
    if prompt.len() + completion.len() > seq_len {
        return Err(format!("{} prompt + {} completion tokens do not fit a row of {seq_len}", prompt.len(), completion.len()));
    }
    for (i, t) in teacher.iter().enumerate() {
        t.validate().map_err(|why| format!("teacher distribution of completion token {i}: {why}"))?;
    }
    if teacher.iter().all(|t| t.ids.is_empty()) {
        return Err("no completion position carries a teacher distribution".to_string());
    }
    if let Some(vocab) = vocab {
        let ids = prompt.iter().chain(completion).copied().chain(teacher.iter().flat_map(|t| t.ids.iter().copied()));
        if let Some(id) = ids.into_iter().find(|&id| id >= vocab) {
            return Err(format!("token id {id} is outside the model's vocabulary of {vocab}"));
        }
    }
    let mut tokens = vec![0u32; seq_len];
    let mut packed_teacher = vec![TopK::default(); seq_len];
    let plen = prompt.len();
    for (i, &t) in prompt.iter().chain(completion.iter()).enumerate() {
        tokens[i] = t;
    }
    for (i, t) in teacher.iter().enumerate() {
        packed_teacher[plen + i - 1] = t.clone();
    }
    Ok(PackedExample { tokens, teacher: packed_teacher })
}

/// Top-K distillation training objective: packs each teacher-scored
/// [`DistillExample`] into one row and runs `K` weighted-CE forward/backward
/// passes per micro-step (`K` = that row's own largest per-position teacher
/// list). Generic over any `M: model::Model` that has adopted weighted-loss
/// support ([`model::training::TokenObjectives`]; its log-probability readout is NOT used - see the module
/// doc comment on why this weight formula needs no current-policy readback)
/// - today `qwen3::Qwen` and `gpt2::Gpt`.
pub struct DistillTopK {
    cfg: DistillConfig,
    source: IndexedStream<DistillExample>,
    pending: VecDeque<PackedExample>,
    last_loss: f32,
    /// Teacher-scored passes run since the last [`Objective::step_verdict`]: a step with none changes nothing.
    passes: u32,
    /// An example the source supplied that cannot be packed faithfully; aborts the run at the step's verdict.
    refusal: Option<String>,
    /// The model's vocabulary, known once [`Objective::check_data`] ran: ids past it are refused.
    vocab: Option<u32>,
}

impl DistillTopK {
    /// Make sure an example is queued, drawing and packing one from the source when none is.
    /// `false`: nothing to train (the source had none, or the one it supplied was refused and
    /// the refusal recorded for the step's verdict).
    fn enqueue(&mut self) -> bool {
        if self.pending.is_empty() {
            let Some(example) = self.source.draw() else { return false };
            match pack_row(self.cfg.seq_len, self.vocab, &example.prompt, &example.completion, &example.teacher) {
                Ok(packed) => self.pending.push_back(packed),
                Err(why) => {
                    self.refusal = Some(why);
                    return false;
                }
            }
        }
        true
    }

    /// `source` supplies the example at the index it is given (or `None` for a no-signal round): a
    /// function of the index alone, so a resumed run - which asks for the index it stopped at again -
    /// continues the stream exactly.
    pub fn new<S>(cfg: DistillConfig, source: S) -> DistillTopK
    where
        S: FnMut(u64) -> Option<DistillExample> + 'static,
    {
        DistillTopK { cfg, source: IndexedStream::new(source), pending: VecDeque::new(), last_loss: 0.0, passes: 0, refusal: None, vocab: None }
    }

    /// Convenience over [`Self::new`]: a fixed, cycling dataset of
    /// already teacher-scored examples - the common case for a small
    /// synthetic distillation set or a pre-collected offline teacher-logit
    /// dump, rather than a live scoring source.
    pub fn from_examples(cfg: DistillConfig, examples: Vec<DistillExample>) -> DistillTopK {
        assert!(!examples.is_empty(), "DistillTopK::from_examples: need at least one example");
        DistillTopK::new(cfg, stream::cycle(examples))
    }
}

impl DistillTopK {
    /// Literally K weighted-CE passes (see the module doc comment for the derivation), one
    /// ordinary forward per k and, when `train`, one backward, letting `Model::backward`'s
    /// accumulate-not-overwrite contract sum them. Returns the mean KL over the active
    /// positions.
    fn run_passes<M: Model>(&mut self, model: &M, packed: &PackedExample, train: bool) -> f32 {
        let k_max = packed.teacher.iter().map(|t| t.ids.len()).max().unwrap_or(0);
        let mut total = 0.0f32;
        for k in 0..k_max {
            let (targets, weights, _count) = pass_arrays(&packed.teacher, k);
            model.set_batch(Batch::LmWeighted { tokens: &packed.tokens, targets: &targets, weights: &weights });
            total += model.forward();
            if train {
                model.backward();
                self.passes += 1;
            }
        }
        mean_kl(total, &packed.teacher)
    }
}

impl<M: Model> Objective<M> for DistillTopK {
    fn regime(&self) -> &'static str {
        "distill_topk"
    }

    fn prepare(&mut self, model: &mut M) -> Result<(), Unsupported> {
        model::training::require_token_objectives(model)?.enable_weighted_loss();
        Ok(())
    }

    fn check_data(&mut self, model: &M) -> Result<(), std::io::Error> {
        self.vocab = Some(model.config().vocab());
        Ok(())
    }

    fn micro_step(&mut self, model: &M, _rng: &mut Rng) -> f32 {
        // No example this round is a legitimate no-op micro-step, matching GRPO's/DPO's own convention.
        if !self.enqueue() {
            return 0.0;
        }
        let packed = self.pending.pop_front().expect("just ensured pending is non-empty");
        self.last_loss = self.run_passes(model, &packed, true);
        self.last_loss
    }

    /// The loss of the example the next training step takes, forward only; the example stays
    /// queued, so probing moves no cursor.
    fn loss_probe(&mut self, model: &M, _rng: &mut Rng) -> f32 {
        if !self.enqueue() {
            return 0.0;
        }
        let packed = self.pending.pop_front().expect("just ensured pending is non-empty");
        let loss = self.run_passes(model, &packed, false);
        self.pending.push_front(packed);
        loss
    }

    /// The examples trained so far: where a resumed run continues the stream.
    fn cursor(&self) -> Option<serde_json::Value> {
        Some(self.source.position(self.pending.len()))
    }

    fn restore_cursor(&mut self, cursor: &serde_json::Value) -> Result<(), std::io::Error> {
        self.pending.clear();
        self.source.seek(cursor)
    }

    /// An optimiser step with no teacher-scored position holds no gradient: applying it
    /// would still decay the weights, advance the moments and the bias-correction step.
    fn step_verdict(&mut self) -> StepVerdict {
        if let Some(why) = self.refusal.take() {
            return StepVerdict::Abort(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("distillation example refused: {why}")));
        }
        match std::mem::take(&mut self.passes) {
            0 => StepVerdict::Skip("no teacher-scored position this step".into()),
            _ => StepVerdict::Apply,
        }
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        vec![("distill_kl", self.last_loss)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keystone identity itself, on plain vectors, independent of any
    /// model or kernel: `p - q_K == Σ_k q_k·(p - e_{v_k})` for ANY `p`,
    /// whenever `Σ_k q_k = 1` - pure linearity, the fact everything else in
    /// this file leans on.
    #[test]
    fn keystone_identity_p_minus_qk_equals_weighted_onehot_sum() {
        let p = [0.1f32, 0.2, 0.3, 0.4];
        let topk = TopK { ids: vec![0, 2], probs: vec![0.75, 0.25] }; // sums to 1

        let mut lhs = p.to_vec();
        for (&id, &prob) in topk.ids.iter().zip(&topk.probs) {
            lhs[id as usize] -= prob;
        }

        let mut rhs = vec![0.0f32; 4];
        for (&id, &prob) in topk.ids.iter().zip(&topk.probs) {
            for (v, &pv) in p.iter().enumerate() {
                let e = if v as u32 == id { 1.0 } else { 0.0 };
                rhs[v] += prob * (pv - e);
            }
        }

        for (l, r) in lhs.iter().zip(&rhs) {
            assert!((l - r).abs() < 1e-6, "{lhs:?} vs {rhs:?}");
        }
    }

    #[test]
    fn pass_arrays_scales_weight_by_this_passs_own_active_count() {
        let teacher = vec![
            TopK { ids: vec![5, 7], probs: vec![0.6, 0.4] },
            TopK { ids: vec![9], probs: vec![1.0] }, // only one entry - inactive at k=1
            TopK::default(),                         // fully inactive (e.g. a prompt position)
        ];

        let (t0, w0, c0) = pass_arrays(&teacher, 0);
        assert_eq!(t0, vec![5, 9, IGNORE]);
        assert_eq!(c0, 2.0, "positions 0 and 1 are active at k=0");
        assert!((w0[0] - 0.6 * 2.0 / 2.0).abs() < 1e-6, "{w0:?}");
        assert!((w0[1] - 1.0 * 2.0 / 2.0).abs() < 1e-6, "{w0:?}");
        assert_eq!(w0[2], 0.0);

        let (t1, w1, c1) = pass_arrays(&teacher, 1);
        assert_eq!(t1, vec![7, IGNORE, IGNORE]);
        assert_eq!(c1, 1.0, "only position 0 has a k=1 entry");
        assert!((w1[0] - 0.4 * 1.0 / 2.0).abs() < 1e-6, "{w1:?}");
    }

    #[test]
    fn from_logits_keeps_the_largest_entries_renormalized_in_descending_order() {
        let logits = [0.0f32, 2.0, 1.0, 2.0, -3.0];
        let topk = TopK::from_logits(&logits, 3);
        assert_eq!(topk.ids, vec![1, 3, 2], "ties keep the lower id first");
        assert!((topk.probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let (e2, e1) = (2.0f32.exp(), 1.0f32.exp());
        assert!((topk.probs[0] - e2 / (2.0 * e2 + e1)).abs() < 1e-6, "{:?}", topk.probs);
        assert!((topk.probs[2] - e1 / (2.0 * e2 + e1)).abs() < 1e-6, "{:?}", topk.probs);
        assert_eq!(TopK::from_logits(&logits, 99).ids.len(), logits.len(), "k is clamped to the row");
    }

    #[test]
    fn neg_entropy_of_a_one_hot_topk_is_zero() {
        let onehot = TopK { ids: vec![3], probs: vec![1.0] };
        assert!(neg_entropy(&onehot).abs() < 1e-6, "1*ln(1) == 0");
    }

    #[test]
    fn neg_entropy_matches_the_closed_form_for_a_uniform_pair() {
        let uniform = TopK { ids: vec![0, 1], probs: vec![0.5, 0.5] };
        let expected = 2.0 * 0.5 * 0.5f32.ln();
        assert!((neg_entropy(&uniform) - expected).abs() < 1e-6);
    }

    #[test]
    fn pack_row_aligns_teacher_to_the_completion_span() {
        let seq_len = 6;
        let prompt = [1u32, 2, 3];
        let completion = [4u32, 5];
        let teacher = vec![TopK { ids: vec![4, 9], probs: vec![0.9, 0.1] }, TopK { ids: vec![5], probs: vec![1.0] }];

        let packed = pack_row(seq_len, None, &prompt, &completion, &teacher).expect("fits");

        assert_eq!(packed.tokens, vec![1, 2, 3, 4, 5, 0]);
        // Position 2 (prompt.len()-1) predicts completion[0]=4, position 3
        // predicts completion[1]=5 - every other position stays inactive.
        for i in [0usize, 1, 4, 5] {
            assert!(packed.teacher[i].ids.is_empty(), "position {i} should be inactive: {:?}", packed.teacher[i]);
        }
        assert_eq!(packed.teacher[2].ids, vec![4, 9]);
        assert_eq!(packed.teacher[3].ids, vec![5]);
    }

    #[test]
    fn pack_row_refuses_a_completion_that_overruns_seq_len_instead_of_truncating_it() {
        let teacher: Vec<TopK> = (0..4).map(|i| TopK { ids: vec![10 + i as u32], probs: vec![1.0] }).collect();
        let Err(err) = pack_row(4, None, &[1, 2], &[3, 4, 5, 6], &teacher) else { panic!("2 + 4 tokens do not fit a row of 4") };
        assert!(err.contains("do not fit"), "{err}");
        assert!(pack_row(6, None, &[1, 2], &[3, 4, 5, 6], &teacher).is_ok(), "exactly full is allowed");
    }

    #[test]
    fn pack_row_refuses_a_zero_information_example() {
        let none = vec![TopK::default(); 2];
        assert!(pack_row(8, None, &[1, 2], &[3, 4], &none).is_err());
        assert!(pack_row(8, None, &[], &[3, 4], &[TopK { ids: vec![1], probs: vec![1.0] }, TopK { ids: vec![1], probs: vec![1.0] }]).is_err(), "no prompt");
    }

    #[test]
    fn a_teacher_distribution_must_be_a_distribution() {
        let ok = TopK { ids: vec![1, 2], probs: vec![0.25, 0.75] };
        assert!(ok.validate().is_ok());
        let bad = [
            TopK { ids: vec![1, 1], probs: vec![0.5, 0.5] },
            TopK { ids: vec![1, 2], probs: vec![0.5] },
            TopK { ids: vec![1, 2], probs: vec![f32::NAN, 1.0] },
            TopK { ids: vec![1, 2], probs: vec![-0.5, 1.5] },
            TopK { ids: vec![1, 2], probs: vec![0.2, 0.2] },
        ];
        for t in bad {
            assert!(t.validate().is_err(), "{t:?}");
        }
    }

    #[test]
    #[should_panic(expected = "need k >= 1")]
    fn top_k_of_zero_is_refused() {
        let _ = TopK::from_logits(&[0.1, 0.2], 0);
    }

    #[test]
    fn the_plan_refuses_a_row_the_model_is_not_built_for() {
        use model::training_plan::ValidatedTrainingPlan;
        let opts = model::FitOpts { block_size: 8, batch_size: 1, ..Default::default() };
        assert!(ValidatedTrainingPlan::build(opts.clone(), &[&DistillConfig { seq_len: 8 }]).is_ok());
        for seq_len in [0, 1, 9] {
            assert!(ValidatedTrainingPlan::build(opts.clone(), &[&DistillConfig { seq_len }]).is_err(), "seq_len {seq_len}");
        }
    }
}
