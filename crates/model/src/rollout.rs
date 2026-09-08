// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Architecture-agnostic rollout: drawing `n` sampled completions for one
//! prompt, with each completion's own per-token log-probabilities under the
//! sampling policy captured at sample time (GRPO's `pi_old`, never re-derived
//! by a later pass over the model).
//!
//! Two implementations of the one [`Rollout`] trait: [`ModelRollout`], the
//! always-correct O(T^2) path over any [`Model::logits_all`] (works for
//! every token-head model with zero per-model code, and doubles as the
//! oracle [`PagedRollout`] is tested against), and [`PagedRollout`], the fast
//! path over an existing [`crate::serve::PagedDecoder`] serving engine -
//! every one of which already shares one prompt's KV across N samples via
//! its own [`crate::paged::PrefixCache`] (see [`PagedRollout`]'s own doc
//! comment for why that, and not [`crate::paged::BlockTable::fork`], is the
//! sharing primitive this builds on).
//!
//! EOS and top-p/top-k sampling are not reinvented here: both rollouts
//! sample via [`crate::serve::sample_from_topk_with_logprob`], the same
//! primitive [`crate::serve::Scheduler`]'s production decode loop uses.
//!
//! Swedish Embedded AB builds training engines where a new sampling-based
//! objective (GRPO, RFT/STaR, best-of-n) composes from one rollout
//! implementation per serving backend instead of a bespoke generation loop
//! per objective. If your team needs expertise in RL-from-verifiable-rewards
//! on top of a from-scratch training stack, you can procure our services by
//! sending an email to info@swedishembedded.com.

use data::rng::Rng;

use crate::paged::BlockTable;
use crate::serve::{self, PagedDecoder, SampleParams};
use crate::{Model, ModelConfig};

/// Why a completion stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The sampled token matched [`RolloutParams::eos`] - excluded from
    /// [`Completion::tokens`]/[`Completion::logprobs`], matching
    /// `crate::serve::Scheduler`'s existing `accept_token` convention (EOS is
    /// a control signal, not output).
    Eos,
    /// [`RolloutParams::max_new`] tokens were produced without hitting EOS.
    MaxNew,
}

/// One sampled completion of a prompt. `logprobs[i]` is `log
/// pi(tokens[i] | prompt, tokens[..i])` under the sampling policy
/// [`RolloutParams::sample`] actually drew from - captured for free at
/// sample time (see [`crate::serve::sample_from_topk_with_logprob`]), never
/// re-derived by a second forward pass. `tokens.len() == logprobs.len()`
/// always.
#[derive(Clone, Debug)]
pub struct Completion {
    pub tokens: Vec<u32>,
    pub logprobs: Vec<f32>,
    pub stop: StopReason,
}

/// Sampling policy + stopping conditions shared by every completion a
/// [`Rollout::sample_n`] call draws for one prompt.
#[derive(Clone, Copy, Debug)]
pub struct RolloutParams {
    /// Stop (with [`StopReason::MaxNew`]) once a completion reaches this many
    /// tokens without hitting EOS first.
    pub max_new: usize,
    /// Temperature / top-k / top-p - see [`SampleParams`]. `SampleParams::greedy()`
    /// reproduces deterministic argmax decoding.
    pub sample: SampleParams,
    /// A sampled token matching this id ends the completion immediately
    /// (see [`StopReason::Eos`]). `None` disables EOS (always run to `max_new`).
    pub eos: Option<u32>,
}

/// The seam objectives (GRPO, RFT/STaR, best-of-n) sample completions
/// through: draw `n` i.i.d. completions of `prompt` under `params`, advancing
/// `rng` once per token actually sampled (so `n == 1` costs exactly the same
/// draws a hand-written single-sample loop would - this is what lets
/// [`crate::train::generate`] wrap [`ModelRollout::sample_n`] and stay
/// byte-identical to its pre-rollout implementation).
pub trait Rollout {
    fn sample_n(&mut self, prompt: &[u32], n: usize, params: &RolloutParams, rng: &mut Rng) -> Vec<Completion>;
}

/// Build the sorted-descending `(token id, logit)` candidate list
/// [`serve::sample_from_topk_with_logprob`] expects, from one position's full
/// vocabulary logits. Ties break toward the LOWER token id (stable secondary
/// key) so greedy decoding here matches the leftmost-wins convention of a
/// plain host `argmax` fold exactly - the property [`ModelRollout`]'s
/// byte-identical-refactor gate depends on. Deliberately not
/// [`PagedDecoder::admit_topk`]'s default body: that one has no such
/// tie-break contract to keep, since a paged decoder's own kernels resolve
/// ties before the host ever sees a full vocab vector.
fn candidates_from_logits(logits: &[f32]) -> Vec<(u32, f32)> {
    let mut candidates: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    candidates
}

/// One sampling step's outcome for a single position: accept `next` into
/// `tokens`/`logprobs` unless it matches `eos`, and report whether the
/// completion is now done. Shared by [`ModelRollout`] and [`PagedRollout`] so
/// the two never drift on EOS/`max_new` semantics.
fn accept(tokens: &mut Vec<u32>, logprobs: &mut Vec<f32>, next: u32, logprob: f32, eos: Option<u32>, max_new: usize) -> Option<StopReason> {
    if Some(next) == eos {
        return Some(StopReason::Eos);
    }
    tokens.push(next);
    logprobs.push(logprob);
    if tokens.len() >= max_new {
        return Some(StopReason::MaxNew);
    }
    None
}

/// The mean per-token entropy (in nats) of the model's own next-token
/// distribution at each position of `completion`, re-derived one
/// [`Model::logits_all`] call per position - the same O(T^2),
/// architecture-agnostic shape [`ModelRollout::sample_one`] samples from, so
/// any [`Model`] with a token-classification head works with zero
/// per-architecture code. `completion` is read back AFTER generation (it need
/// not have been produced by this same rollout machinery - `qwen3::caps`'s
/// promote/reject gate calls this over a completion
/// `crate::sample::generate_kv_stream_with_head` already produced), so this
/// recomputes the distribution rather than reusing a [`Completion::logprobs`]
/// that was never captured.
///
/// A low mean entropy means the model was confident (near-argmax) at almost
/// every step; `promote::gate` reads it as one more signal about whether a
/// candidate adapter overfit its probe set, alongside the verified pass rate.
///
/// An empty `completion` has no position to average and returns `0.0` without
/// calling `logits_all` at all.
pub fn mean_completion_entropy<M: Model>(m: &M, prompt: &[u32], completion: &[u32]) -> f64 {
    if completion.is_empty() {
        return 0.0;
    }
    let block = m.config().block_size() as usize;
    let vocab = m.config().vocab() as usize;
    let mut ctx: Vec<u32> = prompt.to_vec();
    let mut total = 0.0f64;
    for &next in completion {
        let window: &[u32] = if ctx.len() > block { &ctx[ctx.len() - block..] } else { &ctx };
        let logits = m.logits_all(window).expect("mean_completion_entropy: model has no token-classification head");
        let last = &logits[logits.len() - vocab..];
        total += softmax_entropy(last);
        ctx.push(next);
    }
    total / completion.len() as f64
}

/// `-Σ p·ln(p)` over the softmax of `logits`, max-subtracted for numerical
/// stability. Zero-probability entries (only reachable after max-subtraction
/// underflows a logit to exactly `0.0` in the exponentiated domain) are
/// skipped rather than producing a `NaN` from `0·ln(0)`.
fn softmax_entropy(logits: &[f32]) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&l| (l as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().fold(0.0, |h, &e| {
        let p = e / sum;
        if p > 0.0 {
            h - p * p.ln()
        } else {
            h
        }
    })
}

/// The always-correct, architecture-agnostic rollout: one sample at a time,
/// O(T^2) re-prefill via [`Model::logits_all`] exactly like
/// `crate::train::generate` did before this module existed. Works for every
/// token-head [`Model`] with zero per-model code; the oracle
/// [`PagedRollout`]'s fast path is tested against.
pub struct ModelRollout<'a, M: Model> {
    m: &'a M,
}

impl<'a, M: Model> ModelRollout<'a, M> {
    pub fn new(m: &'a M) -> ModelRollout<'a, M> {
        ModelRollout { m }
    }

    /// One completion, consuming `rng` exactly the way the pre-rollout
    /// `generate`'s inline loop did (no extra draws) - what makes `n == 1`
    /// byte-identical to that implementation.
    fn sample_one(&self, prompt: &[u32], params: &RolloutParams, rng: &mut Rng) -> Completion {
        let block = self.m.config().block_size() as usize;
        let vocab = self.m.config().vocab() as usize;
        let mut ctx: Vec<u32> = prompt.to_vec();
        let mut tokens = Vec::with_capacity(params.max_new);
        let mut logprobs = Vec::with_capacity(params.max_new);
        let mut stop = StopReason::MaxNew;
        for _ in 0..params.max_new {
            let window: &[u32] = if ctx.len() > block { &ctx[ctx.len() - block..] } else { &ctx };
            let logits = self.m.logits_all(window).expect("ModelRollout: model has no token-classification head");
            let last = &logits[logits.len() - vocab..];
            let candidates = candidates_from_logits(last);
            let (next, logprob) = serve::sample_from_topk_with_logprob(&candidates, params.sample, rng);
            match accept(&mut tokens, &mut logprobs, next, logprob, params.eos, params.max_new) {
                Some(s) => {
                    stop = s;
                    break;
                }
                None => ctx.push(next),
            }
        }
        Completion { tokens, logprobs, stop }
    }
}

impl<'a, M: Model> Rollout for ModelRollout<'a, M> {
    fn sample_n(&mut self, prompt: &[u32], n: usize, params: &RolloutParams, rng: &mut Rng) -> Vec<Completion> {
        assert!(n > 0, "sample_n: n must be > 0");
        if n == 1 {
            // No sub-stream split: `rng` is consumed directly, exactly as the
            // pre-rollout single-sample `generate` loop did.
            return vec![self.sample_one(prompt, params, rng)];
        }
        (0..n)
            .map(|_| {
                let mut sub = Rng::new(rng.next_u64());
                self.sample_one(prompt, params, &mut sub)
            })
            .collect()
    }
}

/// The fast path over an existing [`PagedDecoder`] serving engine: each
/// sample prefills the SAME prompt through [`PagedDecoder::prefill`], whose
/// own [`crate::paged::PrefixCache`] already shares one prompt's KV blocks
/// across N samples - the second call onward is served (almost) entirely
/// from cache instead of re-running the model, without this rollout
/// reimplementing that reuse. Any decoder already implementing
/// [`PagedDecoder`] (every `qwen3::serve::Engine` mode today) gets this for
/// free.
///
/// [`BlockTable::fork`] is deliberately NOT used to share the prompt's KV
/// directly: it forks a table's blocks byte-for-byte, including whatever
/// partially-filled tail block the prompt's own last, still-appendable block
/// is - unlike [`crate::paged::PrefixCache::lookup`]/`adopt_prefix`, which
/// only ever share CLOSED, immutable full blocks. The very next token
/// appended by any two forked samples would then race to write the SAME
/// physical slot ([`BlockTable::append`] only allocates a fresh block once
/// the current one is exactly full); privatizing it first
/// ([`BlockTable::unshare_tail`]) needs a device-side byte copy of that
/// block's live KV, which [`PagedDecoder`]'s trait surface has no operation
/// for. `PrefixCache`'s block-boundary-only sharing sidesteps the hazard
/// entirely, which is why it is the primitive this rollout builds on.
pub struct PagedRollout<D: PagedDecoder> {
    dec: D,
}

impl<D: PagedDecoder> PagedRollout<D> {
    pub fn new(dec: D) -> PagedRollout<D> {
        PagedRollout { dec }
    }

    /// Hand the decoder back - the caller's own accounting (`prefix_stats`,
    /// `kv_pool_bytes`, ...) all live on `D` itself.
    pub fn into_decoder(self) -> D {
        self.dec
    }
}

struct Seq {
    table: BlockTable,
    next_input: u32,
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
    stop: Option<StopReason>,
    rng: Rng,
}

impl<D: PagedDecoder> Rollout for PagedRollout<D> {
    fn sample_n(&mut self, prompt: &[u32], n: usize, params: &RolloutParams, rng: &mut Rng) -> Vec<Completion> {
        assert!(n > 0, "sample_n: n must be > 0");
        // Mirrors the capacity/vocab checks `serve::Scheduler` runs at
        // admission (`Engine::prefill`'s own doc comment: callers bypassing
        // the scheduler must check these themselves, or a too-long/bad-token
        // prompt silently corrupts the block table / reads out of bounds).
        let need = prompt.len() + params.max_new;
        assert!(need <= self.dec.max_seq_len(), "PagedRollout: prompt+max_new ({need}) exceeds engine capacity ({})", self.dec.max_seq_len());
        let vocab = self.dec.vocab() as u32;
        assert!(prompt.iter().all(|&t| t < vocab), "PagedRollout: prompt token outside vocabulary ({vocab})");

        // Each sample prefills the prompt independently; `PagedDecoder::prefill`'s
        // own prefix cache serves every call after the first from cache (see
        // this type's own doc comment for why that beats a raw `fork`).
        let mut seqs: Vec<Seq> = (0..n)
            .map(|_| {
                let mut table = BlockTable::new();
                let hidden = self.dec.prefill(&mut table, prompt);
                let mut seq_rng = Rng::new(rng.next_u64());
                let mut tokens = Vec::with_capacity(params.max_new);
                let mut logprobs = Vec::with_capacity(params.max_new);
                let (first, logprob) = if params.sample.is_greedy() {
                    (self.dec.admit_greedy(&hidden), 0.0)
                } else {
                    let k = self.dec.topk_capacity().max(1);
                    let candidates = self.dec.admit_topk(&hidden, k);
                    serve::sample_from_topk_with_logprob(&candidates, params.sample, &mut seq_rng)
                };
                let stop = accept(&mut tokens, &mut logprobs, first, logprob, params.eos, params.max_new);
                Seq { table, next_input: first, tokens, logprobs, stop, rng: seq_rng }
            })
            .collect();

        // Batched decode over whatever is still active, exactly the shape
        // `serve::Scheduler::step_inner` drives - one round-trip per token
        // (no on-device window; correctness over throughput for phase P10).
        loop {
            let active: Vec<usize> = (0..seqs.len()).filter(|&i| seqs[i].stop.is_none()).collect();
            if active.is_empty() {
                break;
            }
            let inputs: Vec<u32> = active.iter().map(|&i| seqs[i].next_input).collect();
            let mut refs: Vec<&mut BlockTable> = Vec::with_capacity(active.len());
            for (idx, s) in seqs.iter_mut().enumerate() {
                if active.contains(&idx) {
                    refs.push(&mut s.table);
                }
            }
            if params.sample.is_greedy() {
                let nexts = self.dec.forward_batched_greedy(&mut refs, &inputs);
                for (bi, &si) in active.iter().enumerate() {
                    let s = &mut seqs[si];
                    s.stop = accept(&mut s.tokens, &mut s.logprobs, nexts[bi], 0.0, params.eos, params.max_new);
                    s.next_input = nexts[bi];
                }
            } else {
                let k = self.dec.topk_capacity().max(1);
                let cands = self.dec.forward_batched_topk(&mut refs, &inputs, k);
                for (bi, &si) in active.iter().enumerate() {
                    let s = &mut seqs[si];
                    let (next, logprob) = serve::sample_from_topk_with_logprob(&cands[bi], params.sample, &mut s.rng);
                    s.stop = accept(&mut s.tokens, &mut s.logprobs, next, logprob, params.eos, params.max_new);
                    s.next_input = next;
                }
            }
        }

        seqs.into_iter()
            .map(|mut s| {
                self.dec.release_table(&mut s.table);
                Completion { tokens: s.tokens, logprobs: s.logprobs, stop: s.stop.unwrap_or(StopReason::MaxNew) }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use super::*;
    use crate::{Batch, ModelConfig};

    #[derive(Clone)]
    struct ToyCfg {
        vocab: u32,
        block_size: u32,
    }
    impl ModelConfig for ToyCfg {
        fn param_list(&self) -> Vec<(String, usize)> {
            vec![]
        }
        fn to_json(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        fn from_json(_v: &serde_json::Value) -> Self {
            unimplemented!("not exercised by these tests")
        }
        fn vocab(&self) -> u32 {
            self.vocab
        }
        fn block_size(&self) -> u32 {
            self.block_size
        }
        fn finalize_for_dataset(self, _v: u32, _b: u32) -> Self {
            self
        }
    }

    /// A [`Model`] whose `logits_all` returns one fixed row per call, drawn in
    /// order from `rows` (repeated for every position in the window - only the
    /// LAST row is ever read) - just enough surface for
    /// [`mean_completion_entropy`] to exercise, with every other method
    /// `unimplemented!` so an accidental call (this function reading more than
    /// `config`/`logits_all`) fails loudly rather than returning nonsense.
    struct ToyModel {
        cfg: ToyCfg,
        rows: Vec<Vec<f32>>,
        calls: Cell<usize>,
    }
    impl Model for ToyModel {
        type Config = ToyCfg;
        fn new(_cfg: ToyCfg, _b: u32, _t: u32, _init: &HashMap<String, Vec<f32>>) -> Self {
            unimplemented!()
        }
        fn init_weights(_cfg: &ToyCfg, _seed: u64) -> HashMap<String, Vec<f32>> {
            unimplemented!()
        }
        fn config(&self) -> &ToyCfg {
            &self.cfg
        }
        fn set_batch(&self, _b: Batch) {
            unimplemented!()
        }
        fn forward(&self) -> f32 {
            unimplemented!()
        }
        fn backward(&self) {
            unimplemented!()
        }
        fn zero_grads(&self) {
            unimplemented!()
        }
        fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _clip: Option<f32>, _extra_scale: f32) {
            unimplemented!()
        }
        fn poll_wait(&self) {}
        fn param_names(&self) -> Vec<String> {
            unimplemented!()
        }
        fn read_weight(&self, _name: &str) -> Vec<f32> {
            unimplemented!()
        }
        fn write_weight(&self, _name: &str, _data: &[f32]) {
            unimplemented!()
        }
        fn read_grad(&self, _name: &str) -> Vec<f32> {
            unimplemented!()
        }
        fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
            let i = self.calls.get();
            self.calls.set(i + 1);
            let row = &self.rows[i];
            Some(row.repeat(tokens.len()))
        }
        fn save(&self, _path: &str) {
            unimplemented!()
        }
        fn config_json(&self) -> serde_json::Value {
            unimplemented!()
        }
    }

    #[test]
    fn a_uniform_distribution_has_entropy_ln_vocab() {
        let vocab = 4;
        let m = ToyModel { cfg: ToyCfg { vocab, block_size: 64 }, rows: vec![vec![0.0; vocab as usize]], calls: Cell::new(0) };
        let h = mean_completion_entropy(&m, &[1, 2], &[3]);
        assert!((h - (vocab as f64).ln()).abs() < 1e-9, "uniform softmax entropy must be ln(vocab), got {h}");
    }

    #[test]
    fn a_one_hot_distribution_has_near_zero_entropy() {
        let m = ToyModel { cfg: ToyCfg { vocab: 4, block_size: 64 }, rows: vec![vec![50.0, 0.0, 0.0, 0.0]], calls: Cell::new(0) };
        let h = mean_completion_entropy(&m, &[1], &[0]);
        assert!(h < 1e-9, "a near-one-hot softmax must have ~0 entropy, got {h}");
    }

    #[test]
    fn an_empty_completion_is_zero_and_never_calls_the_model() {
        let m = ToyModel { cfg: ToyCfg { vocab: 4, block_size: 64 }, rows: vec![], calls: Cell::new(0) };
        assert_eq!(mean_completion_entropy(&m, &[1, 2, 3], &[]), 0.0);
    }

    #[test]
    fn multiple_positions_average_their_own_distinct_entropy() {
        let vocab = 4;
        // Position 0: uniform (entropy = ln 4). Position 1: one-hot (entropy ~ 0).
        let m = ToyModel {
            cfg: ToyCfg { vocab, block_size: 64 },
            rows: vec![vec![0.0; vocab as usize], vec![50.0, 0.0, 0.0, 0.0]],
            calls: Cell::new(0),
        };
        let h = mean_completion_entropy(&m, &[1], &[0, 1]);
        let want = ((vocab as f64).ln() + 0.0) / 2.0;
        assert!((h - want).abs() < 1e-6, "must be the unweighted mean of each position's own entropy, got {h} want {want}");
    }

    #[test]
    fn candidates_from_logits_sorts_descending_with_lowest_index_tie_break() {
        let logits = [1.0f32, 5.0, 5.0, 2.0];
        let c = candidates_from_logits(&logits);
        assert_eq!(c[0], (1, 5.0)); // tie between idx 1 and 2 -> lower index first
        assert_eq!(c[1], (2, 5.0));
        assert_eq!(c[2], (3, 2.0));
        assert_eq!(c[3], (0, 1.0));
    }

    #[test]
    fn accept_excludes_eos_and_reports_max_new() {
        let (mut tokens, mut logprobs) = (Vec::new(), Vec::new());
        assert_eq!(accept(&mut tokens, &mut logprobs, 9, -0.1, Some(9), 5), Some(StopReason::Eos));
        assert!(tokens.is_empty(), "EOS must not be appended");

        let (mut tokens, mut logprobs) = (Vec::new(), Vec::new());
        assert_eq!(accept(&mut tokens, &mut logprobs, 3, -0.2, Some(9), 1), Some(StopReason::MaxNew));
        assert_eq!(tokens, vec![3]);
        assert_eq!(logprobs, vec![-0.2]);
    }
}
