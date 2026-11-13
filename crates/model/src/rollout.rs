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
//! oracle [`SchedulerRollout`] is tested against), and [`SchedulerRollout`],
//! the serving path: the group is submitted to the same generic
//! [`crate::serve::Scheduler`] over an existing [`crate::serve::PagedDecoder`]
//! engine that answers requests, so a group's samples share their prompt's
//! prefill, batch their decode steps and run under whichever adapter the
//! trainer last published, exactly as a served request does.
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

use std::collections::HashMap;

use data::rng::Rng;

use crate::adapter::bank::{AdapterError, BankedEngine, VerifiedAdapter};
use crate::serve::{self, AdapterId, PagedDecoder, Request, SampleParams, Scheduler};
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
    /// `Some(log pi(eos | prompt, tokens))` exactly when [`Self::stop`] is
    /// [`StopReason::Eos`]: the stop token is not part of [`Self::tokens`], but
    /// an objective that wants the policy to learn to stop trains on it as one
    /// more sampled token and needs the probability it was drawn with.
    pub eos_logprob: Option<f32>,
    /// The version of the policy that sampled this completion, as the rollout
    /// numbers its own policies: the count of [`SyncRollout::sync`] calls the
    /// engine had taken when the sequence was submitted. A sequence finishes on
    /// the policy it started on, so every token of the completion carries this
    /// one version; a trainer masks a completion whose version is too far
    /// behind its own. `0` is a rollout that does not version (the model's own
    /// [`ModelRollout`]) or a sampler never synced.
    pub policy_version: u64,
    /// Prompt tokens the engine's prefix cache served instead of computing
    /// them: how much of this sample's prefill the group shared. `0` for a
    /// rollout without a cache.
    pub cached_tokens: usize,
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
/// draws a bespoke single-sample loop would - this is what lets
/// [`crate::train::generate`] wrap [`ModelRollout::sample_n`] and stay
/// byte-identical to its pre-rollout implementation).
pub trait Rollout {
    fn sample_n(&mut self, prompt: &[u32], n: usize, params: &RolloutParams, rng: &mut Rng) -> Vec<Completion>;
}

/// Builds the serving-path rollout of one cycle's starting weights: a trainer that runs
/// several cycles (each from a checkpoint) holds the recipe, and each cycle calls
/// [`Self::build`] with its own starting checkpoint and the longest row it trains. The
/// architecture-specific part (which engine, its geometry and the adapter shape it refits to)
/// is the closure's.
#[derive(Clone)]
pub struct RolloutFactory(std::rc::Rc<dyn Fn(&std::path::Path, u32) -> std::io::Result<Box<dyn SyncRollout>>>);

impl RolloutFactory {
    /// A factory that calls `build` with a cycle's starting checkpoint and row length.
    pub fn new(build: impl Fn(&std::path::Path, u32) -> std::io::Result<Box<dyn SyncRollout>> + 'static) -> RolloutFactory {
        RolloutFactory(std::rc::Rc::new(build))
    }

    /// The rollout whose base is the checkpoint at `base`, for rows of up to `row_len` tokens.
    pub fn build(&self, base: &std::path::Path, row_len: u32) -> std::io::Result<Box<dyn SyncRollout>> {
        (self.0)(base, row_len)
    }
}

impl std::fmt::Debug for RolloutFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RolloutFactory")
    }
}

/// A [`Rollout`] that samples from its own copy of a policy (a serving engine,
/// possibly on another device) rather than from the model being trained, and
/// so has to be told what the trained model now holds. `trained` maps every
/// trainable tensor of the model to its current value
/// ([`crate::Model::optimized_params`], or every parameter when the whole
/// model trains); what the rollout does with them (replace, or fold LoRA
/// adapters into its base) is its own business.
pub trait SyncRollout: Rollout {
    fn sync(&mut self, trained: &HashMap<String, Vec<f32>>) -> Result<(), String>;

    /// The sampling distribution this engine actually draws from, when it can say: what
    /// [`Completion::logprobs`] is the log-likelihood under. An objective whose importance
    /// ratio assumes a distribution compares it against this and refuses a mismatch instead
    /// of training on a likelihood that is not the one sampled from. `None` (the default):
    /// the engine does not declare it, so the request in [`RolloutParams`] is taken on trust.
    fn sampling(&self) -> Option<SampleParams> {
        None
    }
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
/// completion is now done. Shared by [`ModelRollout`] and the tests of [`SchedulerRollout`]
/// so the two never drift on EOS/`max_new` semantics.
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

/// `Some(logprob)` when `stop` ended the completion on the stop token.
fn eos_logprob(stop: Option<StopReason>, logprob: f32) -> Option<f32> {
    (stop == Some(StopReason::Eos)).then_some(logprob)
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
        // An empty prompt is zero tokens, and a forward pass over zero
        // tokens reaches the device as a matmul with zero rows, which aborts
        // the process from somewhere that says nothing about the caller.
        // Named here, where the caller is still visible.
        assert!(!prompt.is_empty(), "ModelRollout: an empty prompt is not a question - there is nothing to continue from");
        let block = self.m.config().block_size() as usize;
        let vocab = self.m.config().vocab() as usize;
        let mut ctx: Vec<u32> = prompt.to_vec();
        let mut tokens = Vec::with_capacity(params.max_new);
        let mut logprobs = Vec::with_capacity(params.max_new);
        let mut stop = StopReason::MaxNew;
        let mut eos_lp = None;
        for _ in 0..params.max_new {
            let window: &[u32] = if ctx.len() > block { &ctx[ctx.len() - block..] } else { &ctx };
            let logits = self.m.logits_all(window).expect("ModelRollout: model has no token-classification head");
            let last = &logits[logits.len() - vocab..];
            let candidates = candidates_from_logits(last);
            let (next, logprob) = serve::sample_from_topk_with_logprob(&candidates, params.sample, rng);
            match accept(&mut tokens, &mut logprobs, next, logprob, params.eos, params.max_new) {
                Some(s) => {
                    stop = s;
                    eos_lp = eos_logprob(Some(s), logprob);
                    break;
                }
                None => ctx.push(next),
            }
        }
        Completion { tokens, logprobs, stop, eos_logprob: eos_lp, policy_version: 0, cached_tokens: 0 }
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

/// The serving path: a group of samples is `n` requests to the same
/// [`Scheduler`] that answers served traffic, over any [`PagedDecoder`]
/// engine. Nothing here reimplements admission, prefill, sampling or decode:
///
/// * the group's members share their prompt's prefill through the scheduler
///   (the first prefills, the rest adopt its closed KV blocks from the prefix
///   cache), and run in the same batched decode steps;
/// * each member samples with its own seeded RNG stream, so a seed reproduces
///   whatever else is in the batch;
/// * the log-probability of every token is the one the sequence was drawn
///   with, captured at sample time, never re-derived;
/// * the sequences run under [`Self::set_adapter`]'s adapter (the base
///   model without one), whose KV namespace keeps a refit policy from ever
///   adopting another policy's cached prompt.
///
/// A request the scheduler refuses (too long for the engine, a token outside
/// the vocabulary, an adapter the engine does not serve) is a panic naming the
/// reason: there is no completion to return for it and training on a stand-in
/// would be silent corruption.
pub struct SchedulerRollout<D: PagedDecoder> {
    sched: Scheduler<D>,
    adapter: Option<AdapterId>,
    policy_version: u64,
}

impl<D: PagedDecoder> SchedulerRollout<D> {
    /// A rollout over `dec` that runs up to `max_running` of a group's
    /// sequences at once (the engine's own batch limit applies as well).
    pub fn new(dec: D, max_running: usize) -> SchedulerRollout<D> {
        SchedulerRollout { sched: Scheduler::new(dec, max_running), adapter: None, policy_version: 0 }
    }

    /// Sample under `adapter` (`None`: the base model) from now on, and stamp
    /// the completions with `version`. A refit publishes its adapter to the
    /// engine, then calls this with the next version.
    pub fn set_adapter(&mut self, adapter: Option<AdapterId>, version: u64) {
        self.adapter = adapter;
        self.policy_version = version;
    }

    /// The version completions are stamped with now.
    pub fn policy_version(&self) -> u64 {
        self.policy_version
    }

    /// The scheduler, borrowed: its accounting (`prefix_stats`, `kv_pool_bytes`, ...).
    pub fn scheduler(&self) -> &Scheduler<D> {
        &self.sched
    }

    /// The decoder, borrowed.
    pub fn decoder(&self) -> &D {
        self.sched.decoder()
    }

    /// The decoder, mutably: how a refit loads and unloads adapters.
    pub fn decoder_mut(&mut self) -> &mut D {
        self.sched.decoder_mut()
    }

    /// Hand the decoder back.
    pub fn into_decoder(self) -> D {
        self.sched.into_decoder()
    }
}

impl<D: PagedDecoder + BankedEngine> SchedulerRollout<D> {
    /// Refit the policy to `adapter` without rebuilding the engine: register it
    /// with the engine's adapter bank, sample under it from now on as the next
    /// policy version, and unload the one it replaces (which purges that
    /// adapter's cached prompts, so a prefix cached under the old policy is
    /// never read under the new). Nothing runs between groups, so the old
    /// adapter has no sequence left and goes at once. Returns the new version.
    ///
    /// # Errors
    ///
    /// The engine has no bank, or refuses the adapter (rank above the bank's
    /// cap, a site the bank does not hold). The previous policy keeps serving.
    pub fn publish_adapter(&mut self, adapter: VerifiedAdapter) -> Result<u64, AdapterError> {
        let id = self.sched.decoder_mut().load_adapter(adapter)?;
        let previous = self.adapter.replace(id);
        self.policy_version += 1;
        if let Some(previous) = previous.filter(|p| *p != id) {
            self.sched.decoder_mut().unload_adapter(previous)?;
        }
        Ok(self.policy_version)
    }
}

impl<D: PagedDecoder> Rollout for SchedulerRollout<D> {
    fn sample_n(&mut self, prompt: &[u32], n: usize, params: &RolloutParams, rng: &mut Rng) -> Vec<Completion> {
        assert!(n > 0, "sample_n: n must be > 0");
        assert!(!prompt.is_empty(), "SchedulerRollout: an empty prompt is not a question - there is nothing to continue from");
        let stop: Vec<u32> = params.eos.into_iter().collect();
        let request = || Request { prompt: prompt.to_vec(), max_new: params.max_new, stop: stop.clone() };
        if let Some(why) = self.sched.refusal(&request(), self.adapter) {
            panic!("SchedulerRollout: the engine refuses this prompt: {why}");
        }
        // Group members are submitted together, in order: the scheduler
        // prefills the first and lets the rest adopt its prompt blocks.
        let ids: Vec<u64> = (0..n)
            .map(|_| match self.adapter {
                Some(adapter) => self.sched.submit_for_adapter(request(), adapter, params.sample, rng.next_u64()),
                None => self.sched.submit_sampled(request(), params.sample, rng.next_u64()),
            })
            .collect();

        let mut done: HashMap<u64, Completion> = HashMap::with_capacity(n);
        while done.len() < n {
            let report = self.sched.step_report();
            if let Some((id, why)) = report.rejected.first() {
                panic!("SchedulerRollout: request {id} was refused after it was submitted: {why}");
            }
            let mut logprobs: HashMap<u64, Vec<f32>> = report.completed_logprobs.into_iter().collect();
            let mut meta: HashMap<u64, serve::Completion> = report.completed_meta.into_iter().collect();
            for (id, tokens) in report.completed {
                let logprobs = logprobs.remove(&id).expect("every completed request reports its log-probabilities");
                let meta = meta.remove(&id).expect("every completed request reports how it was served");
                let (stop, eos_logprob) = match meta.stop_token {
                    Some(_) => (StopReason::Eos, meta.stop_logprob),
                    None => (StopReason::MaxNew, None),
                };
                done.insert(id, Completion { tokens, logprobs, stop, eos_logprob, policy_version: self.policy_version, cached_tokens: meta.cached_tokens });
            }
            assert!(!report.finished.is_empty() || self.sched.pending(), "SchedulerRollout: the scheduler went idle with {} of {n} samples unfinished", n - done.len());
        }
        ids.into_iter().map(|id| done.remove(&id).expect("every submitted request completed")).collect()
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
        fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _adam: crate::Adam, _clip: Option<f32>, _extra_scale: f32) {
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

    /// A completion that ended on the stop token reports the probability that
    /// token was drawn with (the stop token itself stays out of `tokens`), and
    /// one that ran to `max_new` reports none.
    #[test]
    fn a_stopped_completion_carries_the_stop_tokens_logprob() {
        let vocab = 4u32;
        // Sampling is greedy over candidate rows, so token 1 is drawn then token 3 (the stop token).
        let rows = vec![vec![0.0, 5.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 9.0], vec![0.0; 4]];
        let m = ToyModel { cfg: ToyCfg { vocab, block_size: 8 }, rows, calls: Cell::new(0) };
        let params = RolloutParams { max_new: 4, sample: SampleParams { temp: 1.0, top_k: 0, top_p: 1.0, ..SampleParams::greedy() }, eos: Some(3) };
        // seed: fixed so the draw (token 1, then the stop token) reproduces
        let mut rng = data::rng::Rng::new(1);
        let c = ModelRollout::new(&m).sample_n(&[0], 1, &params, &mut rng).pop().unwrap();
        assert_eq!(c.stop, StopReason::Eos);
        assert_eq!(c.tokens, vec![1], "the stop token is not part of the tokens");
        let lp = c.eos_logprob.expect("a stopped completion has the stop token's logprob");
        let want = 9.0 - (3.0 * 0.0f32.exp() + 9.0f32.exp()).ln();
        assert!((lp - want).abs() < 1e-4, "{lp} vs {want}");

        let m = ToyModel { cfg: ToyCfg { vocab, block_size: 8 }, rows: vec![vec![0.0, 5.0, 0.0, 0.0]; 2], calls: Cell::new(0) };
        let params = RolloutParams { max_new: 2, sample: SampleParams::greedy(), eos: Some(3) };
        let c = ModelRollout::new(&m).sample_n(&[0], 1, &params, &mut rng).pop().unwrap();
        assert_eq!((c.stop, c.eos_logprob), (StopReason::MaxNew, None));
    }
}
