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
//! target `v_k` and weight `w_k = q_k·count` (the same "multiply by the
//! model's own `/count` weighted-CE divisor to cancel it exactly"
//! `rl::objective::dpo::pair_term`'s `w = β·σ(−u)·count` already uses), and
//! let [`model::Model::backward`]'s documented ACCUMULATE-not-overwrite
//! contract sum the `K` passes' gradients: `Σ_k (q_k·count)·(p −
//! e_{v_k})/count = Σ_k q_k·(p − e_{v_k}) = p − q_K`. Unlike DPO's and
//! GRPO's weight formulas, this one needs no readback of the model's own
//! current predictions at all (`w_k` is pure teacher data) - simpler than
//! either, not just a third instance of the same trick.
//!
//! The realized scalar loss composes the same way: with that weight, one
//! pass's [`model::Model::forward`] returns `Σ_v(active) q_k·CE_k = Σ
//! q_k·(−ln p_{v_k})`; summing all `K` passes gives `Σ_k q_k·(−ln p_{v_k}) =
//! KL(q_K‖p) + H(q_K)` (`H` the teacher's own entropy, a constant that does
//! not depend on the student's parameters). [`neg_entropy`] computes `−H(q_K)
//! = Σ_k q_k·ln(q_k)` directly from the teacher's own (data-only) `TopK`, so
//! adding it to the summed pass losses reports the actual `KL(q_K‖p)`
//! rather than the teacher-entropy-shifted cross-entropy - this constant
//! addition changes nothing about the gradient (entropy is a fixed offset
//! w.r.t. the student's parameters), only what the reported metric means.
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
use model::{Batch, Model, Objective, IGNORE};

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

/// Pass `k`'s weight for one position, scaled by `count` to cancel the
/// model's own `/count` weighted-CE division exactly (see the module doc
/// comment's derivation) - `prob` is pure teacher data, no current-policy
/// readback needed (unlike `rl::objective::dpo::pair_term`/
/// `rl::objective::grpo::token_term`, both of which DO need it).
pub fn topk_weight(prob: f32, count: f32) -> f32 {
    prob * count
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
    let mut targets = vec![IGNORE; teacher.len()];
    let mut weights = vec![0.0f32; teacher.len()];
    for (i, t) in teacher.iter().enumerate() {
        if let (Some(&id), Some(&p)) = (t.ids.get(k), t.probs.get(k)) {
            targets[i] = id;
            weights[i] = topk_weight(p, count);
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
/// `teacher[i]`'s distribution is placed at that same position. A
/// completion that overruns `seq_len` is silently truncated at the tail,
/// same as its siblings.
fn pack_row(seq_len: usize, prompt: &[u32], completion: &[u32], teacher: &[TopK]) -> PackedExample {
    assert_eq!(completion.len(), teacher.len(), "pack_row: one TopK per completion token");
    let mut tokens = vec![0u32; seq_len];
    let mut packed_teacher = vec![TopK::default(); seq_len];
    let plen = prompt.len();
    if plen == 0 {
        // Not a real scenario (every task prompt is non-empty) - guards the
        // `tok_idx - 1` underflow below rather than relying on callers.
        return PackedExample { tokens, teacher: packed_teacher };
    }
    for (i, &t) in prompt.iter().chain(completion.iter()).take(seq_len).enumerate() {
        tokens[i] = t;
    }
    for (i, t) in teacher.iter().enumerate() {
        let tok_idx = plen + i;
        if tok_idx >= seq_len {
            break;
        }
        let t_pos = tok_idx - 1;
        packed_teacher[t_pos] = t.clone();
    }
    PackedExample { tokens, teacher: packed_teacher }
}

/// Where [`DistillTopK`] draws its next teacher-scored example from - a
/// plain closure, the same "sampling/scoring happens upstream" shape
/// `rl::objective::dpo::Dpo`'s `PairSource` uses (here: whatever produced
/// the teacher's top-K token distributions, e.g. an offline teacher-model
/// pass, is a caller concern, not this file's). `None` means "nothing
/// available this round" - a legitimate no-op micro-step.
type ExampleSource = Box<dyn FnMut(&mut Rng) -> Option<DistillExample>>;

/// Top-K distillation training objective: packs each teacher-scored
/// [`DistillExample`] into one row and runs `K` weighted-CE forward/backward
/// passes per micro-step (`K` = that row's own largest per-position teacher
/// list). Generic over any `M: model::Model` that has adopted weighted-loss
/// support ([`model::Model::enable_weighted_loss`]/
/// [`model::Model::batch_token_logprobs`] is NOT required - see the module
/// doc comment on why this weight formula needs no current-policy readback)
/// - today `qwen3::Qwen` and `gpt2::Gpt`.
pub struct DistillTopK {
    cfg: DistillConfig,
    source: ExampleSource,
    pending: VecDeque<PackedExample>,
    last_loss: f32,
}

impl DistillTopK {
    /// `source` supplies the next teacher-scored example (or `None` for a
    /// no-signal round) - see this struct's own field doc comment.
    pub fn new<S>(cfg: DistillConfig, source: S) -> DistillTopK
    where
        S: FnMut(&mut Rng) -> Option<DistillExample> + 'static,
    {
        DistillTopK { cfg, source: Box::new(source), pending: VecDeque::new(), last_loss: 0.0 }
    }

    /// Convenience over [`Self::new`]: a fixed, cycling dataset of
    /// already teacher-scored examples - the common case for a small
    /// synthetic distillation set or a pre-collected offline teacher-logit
    /// dump, rather than a live scoring source.
    pub fn from_examples(cfg: DistillConfig, examples: Vec<DistillExample>) -> DistillTopK {
        assert!(!examples.is_empty(), "DistillTopK::from_examples: need at least one example");
        let mut idx = 0usize;
        DistillTopK::new(cfg, move |_: &mut Rng| {
            let e = examples[idx].clone();
            idx = (idx + 1) % examples.len();
            Some(e)
        })
    }
}

impl<M: Model> Objective<M> for DistillTopK {
    fn regime(&self) -> &'static str {
        "distill_topk"
    }

    fn prepare(&mut self, model: &mut M) {
        model.enable_weighted_loss();
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        if self.pending.is_empty() {
            let Some(example) = (self.source)(rng) else {
                // No example available this round - a legitimate no-op
                // micro-step, matching GRPO's/DPO's own convention.
                return 0.0;
            };
            self.pending.push_back(pack_row(self.cfg.seq_len, &example.prompt, &example.completion, &example.teacher));
        }
        let packed = self.pending.pop_front().expect("just ensured pending is non-empty");

        // Literally K weighted-CE passes (see the module doc comment for
        // the derivation) - one ordinary forward + backward per k, letting
        // `Model::backward`'s accumulate-not-overwrite contract sum them.
        let k_max = packed.teacher.iter().map(|t| t.ids.len()).max().unwrap_or(0);
        let mut total = 0.0f32;
        for k in 0..k_max {
            let (targets, weights, _count) = pass_arrays(&packed.teacher, k);
            model.set_batch(Batch::LmWeighted { tokens: &packed.tokens, targets: &targets, weights: &weights });
            total += model.forward();
            model.backward();
        }

        let entropy: f32 = packed.teacher.iter().map(neg_entropy).sum();
        let active = packed.teacher.iter().filter(|t| !t.ids.is_empty()).count().max(1) as f32;
        self.last_loss = (total + entropy) / active;
        self.last_loss
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
        assert!((w0[0] - 0.6 * 2.0).abs() < 1e-6, "{w0:?}");
        assert!((w0[1] - 1.0 * 2.0).abs() < 1e-6, "{w0:?}");
        assert_eq!(w0[2], 0.0);

        let (t1, w1, c1) = pass_arrays(&teacher, 1);
        assert_eq!(t1, vec![7, IGNORE, IGNORE]);
        assert_eq!(c1, 1.0, "only position 0 has a k=1 entry");
        assert!((w1[0] - 0.4 * 1.0).abs() < 1e-6, "{w1:?}");
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

        let packed = pack_row(seq_len, &prompt, &completion, &teacher);

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
    fn pack_row_truncates_a_completion_that_overruns_seq_len() {
        let seq_len = 4;
        let prompt = [1u32, 2];
        let completion = [3u32, 4, 5, 6];
        let teacher: Vec<TopK> = (0..4).map(|i| TopK { ids: vec![10 + i as u32], probs: vec![1.0] }).collect();

        let packed = pack_row(seq_len, &prompt, &completion, &teacher);

        // Only position 1 (predicts completion[0]=3) and position 2
        // (predicts completion[1]=4) fit in seq_len=4; completion[2..] is
        // dropped, not scored.
        assert_eq!(packed.tokens, vec![1, 2, 3, 4]);
        assert_eq!(packed.teacher[1].ids, vec![10]);
        assert_eq!(packed.teacher[2].ids, vec![11]);
        assert!(packed.teacher[0].ids.is_empty());
        assert!(packed.teacher[3].ids.is_empty());
    }
}
