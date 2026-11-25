// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that every model of a training stack trains online while it
// serves, the same way, for its clients. If your team needs expertise in asynchronous policy-gradient training across
// model architectures then you can procure our services by sending an email to info@swedishembedded.com.

//! The asynchronous policy-gradient half of the conformance suite (A49): what the streaming job
//! ([`crate::objective::stream_pg`]) relies on, checked on a registered model at toy size on the CPU. Every
//! architecture whose trainer declares the policy gradient runs it; what one cannot do it must refuse, by name, and the
//! refusal is part of what is checked.
//!
//! * **online policy gradient:** groups fed to a [`RolloutFeed`] while the run goes on train (each step's update is
//!   applied, the adapter moves), and a run resumed from its saved state trains no group twice: the groups it had
//!   committed are duplicates when they are offered again, the rest train once.
//! * **score centring:** the score-centred estimator's weights, installed as the label weight and the extra targets of
//!   one backward, give the gradient of the sum of label-equivalent weighted backwards (an extra target `(v, c)` adds
//!   what the label `v` at weight `c` adds); a model whose head takes no extra targets refuses the estimator.
//! * **support replay:** a position trained over the support its cut sampler drew it from reads the trainer's
//!   log-probability renormalised over that support (one forward with the support's ids as extra targets, equal to a
//!   softmax over the support's logits of the model's own `logits_all`), and the weights the estimator installs for it
//!   are one backward equal to the sum of their label equivalents; a model whose head takes no extra targets refuses it.
//! * **policy publication:** the adapter the run hands to its publisher every step is a file the architecture's own
//!   serving bank takes (the check is the caller's, [`Publication::check`]: the bank lives above this crate).
//!
//! [`run_async`] answers, per capability, whether it held or how the model refused it ([`AsyncReport`]); the caller
//! asserts the answer it expects, so a declaration that drifts from what the model does fails either way.

use std::path::Path;
use std::sync::{Arc, Mutex};

use model::conformance::{build, build_exact, Subject, BATCH, RANK, SEQ};
use model::training::require_token_objectives;
use model::{AdapterPublish, AdapterSnapshots, Batch, FitControl, FitOpts, Model, ModelConfig, StepReport, IGNORE};
use serde_json::Value;

use crate::objective::external::PolicyGradientConfig;
use crate::objective::grpo::LossAgg;
use crate::objective::pg_batch::{weigh, PgBatch, SamplerSupport, SamplerTop};
use crate::objective::pg_loss::{Estimator, PgLoss};
use crate::objective::stream_pg::{RolloutFeed, StreamExample, StreamGroup, StreamSettings, StreamedPolicyGradient};

/// What an architecture declares about the asynchronous path. `score_centring` is the policy gradient's and the
/// head's extra targets' together ([`model::conformance::Declares::sparse_targets`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AsyncDeclares {
    /// The trainer declares the policy-gradient objective (`ChatTrainer::POLICY_GRADIENT`).
    pub online_policy_gradient: bool,
    /// The architecture serves adapters (a serving bank takes the versions a stream publishes).
    pub policy_publication: bool,
}

/// How the adapters a run publishes are written and checked: the card they carry and the architecture's serving bank's
/// verdict on a file.
pub struct Publication<'a> {
    pub card_id: &'a str,
    pub base_id: &'a str,
    /// The card family the bank serves (`qwen`, `qwen35`, `ricursive`, ...).
    pub family: &'a str,
    /// Whether the serving bank takes the file; the reason when it does not.
    pub check: &'a dyn Fn(&Path) -> Result<(), String>,
}

/// One capability's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Holds,
    /// Not offered, and the reason the model or its declaration gives.
    Refused(String),
}

/// Every capability's answer for one architecture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncReport {
    pub online_policy_gradient: Verdict,
    pub score_centring: Verdict,
    /// Support replay ([`support_replay`]): held exactly where score centring is (both are the head's extra targets).
    pub support_replay: Verdict,
    pub policy_publication: Verdict,
}

/// The policy tokens of one group: two rows of [`SEQ`] tokens with four supervised.
const GROUP_TOKENS: usize = 8;
const GROUPS: usize = 4;

/// A configuration a stream trains with: every token weighed alike, no reference (the checks are of the stream).
fn config(estimator: Estimator) -> PolicyGradientConfig {
    let loss = PgLoss { loss_agg: LossAgg::TokenSum, kl_beta: 0.0, log_kl: false, estimator, ..PolicyGradientConfig::default().loss };
    PolicyGradientConfig { loss, ..PolicyGradientConfig::default() }
}

/// The toy task's groups: two rollouts each, rewarded apart (advantages +1 and -1), behaviour log-probabilities of a
/// near-uniform sampler over `vocab`.
fn groups(vocab: u32) -> Vec<StreamGroup> {
    (0..GROUPS)
        .map(|g| {
            let examples = [1.0f32, -1.0]
                .iter()
                .enumerate()
                .map(|(i, &advantage)| {
                    let ids: Vec<u32> = (0..SEQ).map(|t| (g as u32 * 7 + i as u32 * 3 + t * 5 + 1) % vocab).collect();
                    let mask: Vec<bool> = (0..SEQ as usize).map(|t| t >= 2).collect();
                    StreamExample { ids, mask, behavior: vec![-(vocab as f32).ln(); SEQ as usize], advantage, policy_version: None, top: None, support: None }
                })
                .collect();
            StreamGroup { key: format!("g{g}"), examples }
        })
        .collect()
}

fn opts(seed: u64) -> FitOpts {
    FitOpts { steps: 1000, batch_size: BATCH, block_size: SEQ, lr: 1e-2, min_lr: 1e-2, warmup: 1, decay_iters: 1000, grad_accum: 1, eval_interval: 0, initial_probe_batches: 0, seed, ..FitOpts::default() }
}

/// The groups' keys a report's `applied` frames name, in order.
fn applied(events: &[Value]) -> Vec<String> {
    events.iter().filter(|e| e["type"] == "applied" && e.get("skipped").is_none()).flat_map(|e| e["groups"].as_array().cloned().unwrap_or_default()).filter_map(|g| g.as_str().map(str::to_string)).collect()
}

/// A run over a feed the groups are offered to by another thread once the run takes them (a resumed run first restores
/// what it consumed), finished after; stopped after `stop_after` steps when given. The feed's reports and the model.
#[allow(clippy::too_many_arguments)]
fn streamed<M: Model>(subject: &Subject<M>, seed: u64, state: &Path, stop_after: Option<u32>, publication: Option<&Publication<'_>>, dir: &Path, published: &Mutex<Vec<Result<(), String>>>) -> Result<(Vec<Value>, Vec<String>, M), String> {
    let (m, _) = build(subject, Some(RANK), BATCH, seed);
    let vocab = ModelConfig::vocab(m.config());
    let settings = StreamSettings { step_tokens: GROUP_TOKENS, queue_tokens: GROUPS * GROUP_TOKENS, max_staleness: None, block: SEQ as usize, top_k: None, support: None };
    let feed = RolloutFeed::new(settings, || true, Default::default())?;
    if !state.exists() {
        feed.open();
    }
    let duplicates: Arc<Mutex<Vec<String>>> = Arc::default();
    let offer = {
        let (feed, duplicates) = (Arc::clone(&feed), Arc::clone(&duplicates));
        std::thread::spawn(move || -> Result<(), String> {
            if !feed.wait_open(std::time::Duration::from_secs(600)) {
                return Err("the run never took records".to_string());
            }
            let admission = feed.offer(groups(vocab)).map_err(|e| format!("{e:?}"))?;
            duplicates.lock().unwrap_or_else(|e| e.into_inner()).extend(admission.duplicates);
            feed.finish();
            Ok(())
        })
    };
    let mut objective = StreamedPolicyGradient::new(Arc::clone(&feed), config(Estimator::ClippedSurrogate), BATCH as usize)?;
    let mut on_step = |s: &StepReport| stop_after.is_none_or(|n| s.step < n);
    let mut publish = |_step: u32, path: &Path| -> std::io::Result<()> {
        let verdict = publication.map_or(Ok(()), |p| (p.check)(path));
        published.lock().unwrap_or_else(|e| e.into_inner()).push(verdict);
        Ok(())
    };
    let snapshots = publication.map(|p| AdapterSnapshots { dir, card_id: p.card_id, base_id: p.base_id, family: p.family, lora: model::adapter::device::LoraCfg::new(RANK, RANK as f32, Vec::new()) });
    let control = FitControl { on_step: Some(&mut on_step), state: Some(state), state_every: 0, identity: serde_json::json!({"conformance": subject.name}), publish: snapshots.map(|snapshots| AdapterPublish { every: 1, initial: false, snapshots, provenance: None, publish: &mut publish }), ..FitControl::default() };
    let fitted = model::fit_controlled(m, &mut objective, &opts(seed), None, control).map_err(|e| format!("{}: the streamed run: {e}", subject.name));
    // A stopped run leaves the offer waiting for a finish nobody sends: the feed is told so it ends.
    feed.finish();
    let offered = offer.join().map_err(|_| "the offering thread panicked".to_string())?;
    let (_, m) = fitted?;
    offered?;
    let duplicates = duplicates.lock().unwrap_or_else(|e| e.into_inner()).clone();
    Ok((feed.drain_events(), duplicates, m))
}

/// The online policy gradient on `subject`: groups train, the adapter moves, a resumed run trains no group twice; the
/// published versions' verdicts collected in `published`.
fn online<M: Model>(subject: &Subject<M>, seed: u64, publication: Option<&Publication<'_>>, dir: &Path, published: &Mutex<Vec<Result<(), String>>>) -> Result<(), String> {
    let name = subject.name;
    let (fresh, _) = build(subject, Some(RANK), BATCH, seed);
    let adapters: Vec<String> = fresh.param_names().into_iter().filter(|n| model::adapter::device::is_adapter_param(n)).collect();
    if adapters.is_empty() {
        return Err(format!("{name}: no adapter tensors to train"));
    }
    let before: Vec<Vec<f32>> = adapters.iter().map(|n| fresh.read_weight(n)).collect();
    drop(fresh);

    // A straight run: every group trains once, in order, and the adapter moves.
    let straight = dir.join("straight.state");
    let (events, _, trained) = streamed(subject, seed, &straight, None, publication, dir, published)?;
    let keys = applied(&events);
    let all: Vec<String> = (0..GROUPS).map(|g| format!("g{g}")).collect();
    if keys != all {
        return Err(format!("{name}: the streamed run applied {keys:?}; every group once, in order, was {all:?}"));
    }
    if adapters.iter().zip(&before).all(|(n, w)| &trained.read_weight(n) == w) {
        return Err(format!("{name}: four applied steps left every adapter tensor where it started"));
    }

    // Stopped after two steps (its state written), then resumed: the committed groups are duplicates, the rest train once.
    let state = dir.join("resumed.state");
    let (first, _, _) = streamed(subject, seed, &state, Some(2), None, dir, published)?;
    let (second, duplicates, _) = streamed(subject, seed, &state, None, None, dir, published)?;
    let (before_stop, after) = (applied(&first), applied(&second));
    let mut both = before_stop.clone();
    both.extend(after.iter().cloned());
    both.sort();
    if both != all || duplicates != before_stop {
        return Err(format!("{name}: a resumed stream applied {before_stop:?} then {after:?} (duplicates {duplicates:?}); every group exactly once, the committed ones duplicates"));
    }
    Ok(())
}

/// The gradients of every trainable tensor after one backward of the batch set on `m`.
fn gradients<M: Model>(m: &M) -> Vec<(String, Vec<f32>)> {
    m.zero_grads();
    m.forward();
    m.backward();
    m.poll_wait();
    m.param_names().into_iter().map(|n| {
        let g = m.read_grad(&n);
        (n, g)
    }).collect()
}

/// The score-centred estimator's weights, as one backward with extra targets, against the sum of their label
/// equivalents: an identity exact in real arithmetic, so both models are built at the exact fp32 arithmetic
/// ([`build_exact`]) on every device.
fn score_centring<M: Model>(subject: &Subject<M>, seed: u64) -> Result<Verdict, String> {
    const K: usize = 2;
    let name = subject.name;
    let (mut m, _) = build_exact(subject, Some(RANK), BATCH, seed);
    model::conformance_grad::randomize_adapters(&m, seed);
    let objectives = require_token_objectives(&mut m).map_err(|e| e.to_string())?;
    objectives.enable_weighted_loss();
    if let Err(e) = objectives.enable_sparse_targets(K as u32) {
        return Ok(Verdict::Refused(format!("its LM head takes no extra targets: {e}")));
    }
    let vocab = ModelConfig::vocab(m.config());
    let (tokens, targets) = model::conformance::batch(vocab);
    let n = tokens.len();
    let supervised = |i: usize| targets[i] != IGNORE;
    // The sampler's top-2 at each supervised position: the label and its neighbour, at a sampler that liked them.
    let ids: Vec<u32> = (0..n).flat_map(|i| if supervised(i) { [targets[i], (targets[i] + 1) % vocab] } else { [IGNORE, IGNORE] }).collect();
    let sampler: Vec<f32> = (0..n).flat_map(|i| if supervised(i) { [0.3f32.ln(), 0.2f32.ln()] } else { [f32::NAN, f32::NAN] }).collect();
    let old_lp: Vec<f32> = (0..n).map(|i| if supervised(i) { 0.3f32.ln() } else { 0.0 }).collect();
    let advantage: Vec<f32> = (0..n).map(|i| if supervised(i) { if i % 2 == 0 { 1.0 } else { -0.5 } } else { f32::NAN }).collect();
    // A band the toy policy's token and listed ids sit inside (their ratios to the sampler are about 0.1 over a
    // vocabulary of a few dozen) and the unlisted mass's ratio (about 1.9) outside, so no listed id's coefficient is
    // cancelled by the tail's.
    let loss = PgLoss { centring_low: 1e-6, centring_high: 1.5, ..config(Estimator::ScoreCentred).loss };
    let spec = loss.spec();
    let batch = PgBatch { targets: &targets, old_lp: &old_lp, ref_lp: None, advantage: &advantage, is_weight: None, top: Some(SamplerTop { k: K, ids: &ids, logprobs: &sampler }), support: None };
    let denominator = 7.0f32;

    m.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
    let o = m.token_objectives().expect("checked above");
    o.set_sparse_target_ids(&ids)?;
    m.forward();
    let o = m.token_objectives().expect("checked above");
    let new_lp = o.batch_token_logprobs();
    let new_top = o.batch_sparse_target_logprobs().ok_or_else(|| format!("{name}: no extra-target log-probabilities after a forward"))?;
    let (weights, terms) = weigh(&new_lp, Some(&new_top), &batch, &spec);
    let extra = weights.extra.clone().ok_or_else(|| format!("{name}: the score-centred estimator wrote no extra-target coefficients"))?;
    if !terms.finite || extra.iter().all(|&c| c == 0.0) {
        return Err(format!("{name}: the score-centred weights are not finite, or centre nothing"));
    }
    let with_extra = |m: &M| -> Result<Vec<(String, Vec<f32>)>, String> {
        m.set_batch(Batch::LmWeighted { tokens: &tokens, targets: &targets, weights: &weights.label });
        let o = m.token_objectives().expect("checked above");
        o.set_loss_denominator(denominator);
        o.set_sparse_target_ids(&ids)?;
        o.set_sparse_target_weights(&extra)?;
        Ok(gradients(m))
    };
    let got = with_extra(&m)?;
    // The label equivalents on a model that takes no extra targets: the labels at their weights, then each slot's ids as
    // labels at its coefficients.
    let (mut plain, _) = build_exact(subject, Some(RANK), BATCH, seed);
    model::conformance_grad::randomize_adapters(&plain, seed);
    require_token_objectives(&mut plain).map_err(|e| e.to_string())?.enable_weighted_loss();
    let weighted = |labels: &[u32], w: &[f32]| {
        plain.set_batch(Batch::LmWeighted { tokens: &tokens, targets: labels, weights: w });
        plain.token_objectives().expect("checked above").set_loss_denominator(denominator);
        gradients(&plain)
    };
    let mut want = weighted(&targets, &weights.label);
    for j in 0..K {
        let labels: Vec<u32> = (0..n).map(|i| ids[i * K + j]).collect();
        let coefs: Vec<f32> = (0..n).map(|i| extra[i * K + j]).collect();
        for ((_, total), (_, g)) in want.iter_mut().zip(weighted(&labels, &coefs)) {
            total.iter_mut().zip(&g).for_each(|(t, g)| *t += g);
        }
    }
    let largest = want.iter().flat_map(|(_, w)| w.iter()).fold(0.0f32, |a, &b| a.max(b.abs())).max(1e-12);
    for ((tensor, g), (_, w)) in got.iter().zip(&want) {
        let off = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if off > 1e-4 * largest {
            return Err(format!("{name}: {tensor}: the score-centred backward is {off:e} from the sum of its label equivalents (largest gradient {largest:e})"));
        }
    }
    Ok(Verdict::Holds)
}

/// The extra-target weights of `batch` (forwarded on `m` with `ids` as its slots), installed as one backward, against the
/// sum of their label equivalents on a second model built the same way: an extra target `(v, c)` adds what the label `v`
/// at weight `c` adds.
#[allow(clippy::too_many_arguments)]
fn one_backward_is_its_label_equivalents<M: Model>(subject: &Subject<M>, seed: u64, m: &M, tokens: &[u32], targets: &[u32], ids: &[u32], k: usize, weights: &crate::objective::pg_batch::TokenWeights, what: &str) -> Result<(), String> {
    let name = subject.name;
    let extra = weights.extra.clone().ok_or_else(|| format!("{name}: {what} wrote no extra-target coefficients"))?;
    let denominator = 7.0f32;
    m.set_batch(Batch::LmWeighted { tokens, targets, weights: &weights.label });
    let o = m.token_objectives().expect("checked above");
    o.set_loss_denominator(denominator);
    o.set_sparse_target_ids(ids)?;
    o.set_sparse_target_weights(&extra)?;
    let got = gradients(m);
    let (mut plain, _) = build_exact(subject, Some(RANK), BATCH, seed);
    model::conformance_grad::randomize_adapters(&plain, seed);
    require_token_objectives(&mut plain).map_err(|e| e.to_string())?.enable_weighted_loss();
    let weighted = |labels: &[u32], w: &[f32]| {
        plain.set_batch(Batch::LmWeighted { tokens, targets: labels, weights: w });
        plain.token_objectives().expect("checked above").set_loss_denominator(denominator);
        gradients(&plain)
    };
    let n = targets.len();
    let mut want = weighted(targets, &weights.label);
    for j in 0..k {
        let labels: Vec<u32> = (0..n).map(|i| ids[i * k + j]).collect();
        let coefs: Vec<f32> = (0..n).map(|i| extra[i * k + j]).collect();
        for ((_, total), (_, g)) in want.iter_mut().zip(weighted(&labels, &coefs)) {
            total.iter_mut().zip(&g).for_each(|(t, g)| *t += g);
        }
    }
    let largest = want.iter().flat_map(|(_, w)| w.iter()).fold(0.0f32, |a, &b| a.max(b.abs())).max(1e-12);
    for ((tensor, g), (_, w)) in got.iter().zip(&want) {
        let off = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if off > 1e-4 * largest {
            return Err(format!("{name}: {tensor}: the {what} backward is {off:e} from the sum of its label equivalents (largest gradient {largest:e})"));
        }
    }
    Ok(())
}

/// Support replay: the trainer's log-probability renormalised over a support of three ids (the label and two others)
/// against an f64 softmax over the same ids of the model's own `logits_all`, and the clipped surrogate's weights over the
/// support as one backward against their label equivalents. Built at the exact fp32 arithmetic ([`build_exact`]).
fn support_replay<M: Model>(subject: &Subject<M>, seed: u64) -> Result<Verdict, String> {
    const K: usize = 3;
    let name = subject.name;
    let (mut m, _) = build_exact(subject, Some(RANK), BATCH, seed);
    model::conformance_grad::randomize_adapters(&m, seed);
    let objectives = require_token_objectives(&mut m).map_err(|e| e.to_string())?;
    objectives.enable_weighted_loss();
    if let Err(e) = objectives.enable_sparse_targets(K as u32) {
        return Ok(Verdict::Refused(format!("its LM head takes no extra targets: {e}")));
    }
    let vocab = ModelConfig::vocab(m.config());
    let (tokens, targets) = model::conformance::batch(vocab);
    let n = tokens.len();
    let supervised = |i: usize| targets[i] != IGNORE;
    let ids: Vec<u32> = (0..n).flat_map(|i| if supervised(i) { [targets[i], (targets[i] + 1) % vocab, (targets[i] + 5) % vocab] } else { [IGNORE; K] }).collect();
    m.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
    m.token_objectives().expect("checked above").set_sparse_target_ids(&ids)?;
    m.forward();
    let o = m.token_objectives().expect("checked above");
    let (new_lp, new_slots) = (o.batch_token_logprobs(), o.batch_sparse_target_logprobs().ok_or_else(|| format!("{name}: no extra-target log-probabilities after a forward"))?);
    // The renormalised figure against the model's own logits, row by row, read on the same model built for one row (the
    // same weights and adapters: `logits_all` scores a single sequence).
    let seq = n / BATCH as usize;
    let (one, _) = build_exact(subject, Some(RANK), 1, seed);
    model::conformance_grad::randomize_adapters(&one, seed);
    for r in 0..BATCH as usize {
        let logits = one.logits_all(&tokens[r * seq..(r + 1) * seq]).ok_or_else(|| format!("{name}: no logits_all to check the support against"))?;
        for t in (0..seq).filter(|&t| supervised(r * seq + t)) {
            let i = r * seq + t;
            let row = &logits[t * vocab as usize..(t + 1) * vocab as usize];
            let at: Vec<f64> = ids[i * K..(i + 1) * K].iter().map(|&id| f64::from(row[id as usize])).collect();
            let max = at.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let want = at[0] - (max + at.iter().map(|z| (z - max).exp()).sum::<f64>().ln());
            let got = model::head_ce::renormalise_over_support(new_lp[i], &new_slots[i * K..(i + 1) * K], &ids[i * K..(i + 1) * K]).ok_or_else(|| format!("{name}: position {i} has no support"))?.label;
            if (got - want).abs() > 1e-3 * want.abs().max(1.0) {
                return Err(format!("{name}: position {i}: the trainer's log-probability over the support is {got}, the softmax of its logits over the support {want}"));
            }
        }
    }
    // The sampler's figure within the support, a little off the trainer's so the ratios are not one.
    let sampler: Vec<f32> = (0..n).map(|i| if supervised(i) { model::head_ce::renormalise_over_support(new_lp[i], &new_slots[i * K..(i + 1) * K], &ids[i * K..(i + 1) * K]).expect("a support").label as f32 - 0.05 * (i % 3) as f32 } else { f32::NAN }).collect();
    let old_lp: Vec<f32> = (0..n).map(|i| if supervised(i) { new_lp[i] - 0.1 } else { 0.0 }).collect();
    let advantage: Vec<f32> = (0..n).map(|i| if supervised(i) { if i % 2 == 0 { 1.0 } else { -0.5 } } else { f32::NAN }).collect();
    let spec = config(Estimator::ClippedSurrogate).loss.spec();
    let batch = PgBatch { targets: &targets, old_lp: &old_lp, ref_lp: None, advantage: &advantage, is_weight: None, top: None, support: Some(SamplerSupport { k: K, ids: &ids, logprobs: &sampler }) };
    let (weights, terms) = weigh(&new_lp, Some(&new_slots), &batch, &spec);
    if !terms.finite || weights.extra.as_ref().is_none_or(|e| e.iter().all(|&c| c == 0.0)) {
        return Err(format!("{name}: the weights over the supports are not finite, or renormalise nothing"));
    }
    one_backward_is_its_label_equivalents(subject, seed, &m, &tokens, &targets, &ids, K, &weights, "support-normalised")?;
    Ok(Verdict::Holds)
}

/// Run A49 for `subject` as `declares` says it should behave, publishing through `publication` when the architecture
/// serves adapters; `dir` holds the run's state and published files. `seed` draws the weights.
pub fn run_async<M: Model>(subject: &Subject<M>, declares: AsyncDeclares, publication: Option<&Publication<'_>>, dir: &Path, seed: u64) -> Result<AsyncReport, String> {
    let name = subject.name;
    if !declares.online_policy_gradient {
        let why = Verdict::Refused("its trainer declares no policy gradient (ChatTrainer::POLICY_GRADIENT)".to_string());
        return Ok(AsyncReport { online_policy_gradient: why.clone(), score_centring: why.clone(), support_replay: why.clone(), policy_publication: why });
    }
    if !subject.declares.token_objectives || !subject.declares.adapters {
        return Err(format!("{name}: declares the policy gradient but no token objectives and adapters to train it with"));
    }
    match (declares.policy_publication, publication.is_some()) {
        (true, false) => return Err(format!("{name}: declares adapter serving but the suite was given no serving bank to publish to")),
        (false, true) => return Err(format!("{name}: declares no adapter serving but the suite was given a serving bank")),
        _ => {}
    }
    let published = Mutex::new(Vec::new());
    online(subject, seed, publication, dir, &published)?;
    let policy_publication = match publication {
        None => Verdict::Refused("it serves no adapters: no bank takes a published version".to_string()),
        Some(_) => {
            let verdicts = published.into_inner().unwrap_or_else(|e| e.into_inner());
            if verdicts.len() != GROUPS {
                return Err(format!("{name}: {} versions were published over {GROUPS} steps", verdicts.len()));
            }
            if let Some(Err(why)) = verdicts.into_iter().find(Result::is_err) {
                return Err(format!("{name}: the serving bank refused a published version: {why}"));
            }
            Verdict::Holds
        }
    };
    let score_centring = score_centring(subject, seed)?;
    if (score_centring == Verdict::Holds) != subject.declares.sparse_targets {
        return Err(format!("{name}: declares sparse_targets={} and the score-centred check answered {score_centring:?}", subject.declares.sparse_targets));
    }
    let support_replay = support_replay(subject, seed)?;
    if (support_replay == Verdict::Holds) != subject.declares.sparse_targets {
        return Err(format!("{name}: declares sparse_targets={} and the support-replay check answered {support_replay:?}", subject.declares.sparse_targets));
    }
    Ok(AsyncReport { online_policy_gradient: Verdict::Holds, score_centring, support_replay, policy_publication })
}
