// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training agents on whole sessions of their own sampled tokens, for its
// clients. If your team needs expertise in policy-gradient training on long multi-call agent episodes then you can
// procure our services by sending an email to info@swedishembedded.com.

#![cfg(feature = "qwen3")]

//! How a streamed policy-gradient step weighs the episodes of a group, pinned on a toy `qwen3` decoder on the CPU.
//!
//! An agent episode reaches a stream as records of its model calls: one record per call (the request as it was sent,
//! the sampled answer supervised), every record of the episode carrying the episode's advantage. A step's gradient is
//! the sum over its policy tokens of each token's term, divided by the step's count of them (`token-sum`) or by its
//! rows times a constant (`constant-length`): either way each call record is one row and each of its answer tokens
//! weighs alike, so an episode weighs by its calls and their answer tokens, not as one episode.

use std::sync::Arc;

use data::rng::Rng;
use model::{Model, Objective, StepVerdict, StepWork, Supply};
use qwen3::{Qwen, QwenConfig};
use rl::objective::external::PolicyGradientConfig;
use rl::objective::grpo::LossAgg;
use rl::objective::pg_loss::PgLoss;
use rl::objective::stream_pg::{RolloutFeed, StreamExample, StreamGroup, StreamSettings, StreamedPolicyGradient};

/// Rows of one micro-batch.
const ROWS: u32 = 2;

/// A tiny decoder with LoRA adapters moved off their zero start, so every adapter tensor carries gradient.
fn toy(block: u32, seed: u64) -> Qwen {
    let cfg = QwenConfig { lora: Some(qwen3::config::LoraCfg::new(2, 4.0, qwen3::finetune::default_lora_targets())), block_size: block, max_position_embeddings: block, ..QwenConfig::tiny() };
    let init = <Qwen as Model>::init_weights(&cfg, seed);
    let m = Qwen::new(cfg, ROWS, block, &init);
    let mut rng = Rng::new(seed ^ 0x5e55);
    for name in Model::param_names(&m).into_iter().filter(|n| n.contains(".lora_")) {
        let w: Vec<f32> = (0..Model::read_weight(&m, &name).len()).map(|_| 0.6 * (rng.next_f32() - 0.5)).collect();
        Model::write_weight(&m, &name, &w);
    }
    m
}

/// A stream configuration of aggregation `agg`: the clipped surrogate, no KL term and no reference read.
fn config(agg: LossAgg) -> PolicyGradientConfig {
    PolicyGradientConfig { loss: PgLoss { loss_agg: agg, kl_beta: 0.0, log_kl: false, ..PgLoss::ONLINE }, ..PolicyGradientConfig::default() }
}

/// The adapters' gradient of one streamed step that takes exactly `groups`, read after the step's micro-batches.
fn step_gradient(m: &mut Qwen, groups: Vec<StreamGroup>, agg: LossAgg, block: usize) -> Vec<Vec<f32>> {
    measured_step(m, groups, agg, block).0
}

/// [`step_gradient`] and what the step's work was: the tokens it forwarded and the seconds it spent.
fn measured_step(m: &mut Qwen, groups: Vec<StreamGroup>, agg: LossAgg, block: usize) -> (Vec<Vec<f32>>, StepWork) {
    let tokens: usize = groups.iter().map(StreamGroup::policy_tokens).sum();
    let settings = StreamSettings { step_tokens: tokens, queue_tokens: tokens, max_staleness: None, block, top_k: None, support: None };
    let feed = RolloutFeed::new(settings, || true, Default::default()).unwrap();
    feed.open();
    let mut objective = StreamedPolicyGradient::new(Arc::clone(&feed), config(agg), ROWS as usize).unwrap();
    Objective::<Qwen>::prepare(&mut objective, m).unwrap();
    Objective::<Qwen>::begin(&mut objective, 7, 0);
    feed.offer(groups).unwrap();
    assert!(matches!(Objective::<Qwen>::supply(&mut objective), Supply::Ready));
    m.zero_grads();
    Objective::<Qwen>::micro_steps(&mut objective, m, &mut Rng::new(1), 1);
    m.poll_wait();
    assert!(matches!(Objective::<Qwen>::step_verdict(&mut objective), StepVerdict::Apply), "the step is finite and unguarded");
    let work = Objective::<Qwen>::step_work(&mut objective).expect("a step reports its work");
    (Model::param_names(m).into_iter().filter(|n| n.contains(".lora_")).map(|n| m.read_grad(&n)).collect(), work)
}

/// One call record: `prompt` as context, then `answer` supervised with behaviour log-probabilities `behavior`.
fn call(prompt: &[u32], answer: &[u32], behavior: f32, advantage: f32) -> StreamExample {
    let ids: Vec<u32> = prompt.iter().chain(answer).copied().collect();
    let mask: Vec<bool> = (0..ids.len()).map(|t| t >= prompt.len()).collect();
    let behavior = mask.iter().map(|&m| if m { behavior } else { f32::NAN }).collect();
    StreamExample { ids, mask, behavior, advantage, policy_version: None, top: None, support: None }
}

/// The largest elementwise gap between `a` and `b`, relative to the largest magnitude in `b`.
fn relative_gap(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let largest = b.iter().flatten().fold(0.0f32, |m, &g| m.max(g.abs())).max(1e-20);
    a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) / largest
}

/// C0, the per-call weighting as it is: an episode of 60 calls against one of 2, every call the same row, in one
/// group. With only the long episode's advantage set, the step's gradient is exactly 30 times what it is with only the
/// short one's: the long episode outweighs the short one by its count of calls (and so of policy tokens), under the
/// token sum and under the constant length alike. One episode is not one sample.
#[test]
fn a_streamed_step_weighs_an_episode_by_its_call_records_so_sixty_calls_outweigh_two_thirty_to_one() {
    const BLOCK: usize = 12;
    let (prompt, answer) = ([1u32, 4, 9, 2, 7], [3u32, 8, 5, 11]);
    let episode = |calls: usize, advantage: f32| (0..calls).map(|_| call(&prompt, &answer, -2.5, advantage)).collect::<Vec<_>>();
    for agg in [LossAgg::TokenSum, LossAgg::ConstantLength(16)] {
        let group = |long: f32, short: f32| vec![StreamGroup { key: "g".to_string(), examples: [episode(60, long), episode(2, short)].concat() }];
        let mut m = toy(BLOCK as u32, 11);
        let long = step_gradient(&mut m, group(1.0, 0.0), agg, BLOCK);
        let short = step_gradient(&mut m, group(0.0, 1.0), agg, BLOCK);
        assert!(short.iter().flatten().any(|g| g.abs() > 1e-6), "{agg}: the short episode moves the adapters");
        let thirty: Vec<Vec<f32>> = short.iter().map(|g| g.iter().map(|v| 30.0 * v).collect()).collect();
        let gap = relative_gap(&long, &thirty);
        assert!(gap < 1e-4, "{agg}: the 60-call episode's gradient is {gap:e} (relative) from 30 times the 2-call episode's");
    }
}

/// One agent session at the token level: a context, then calls of an answer the policy sampled and the observation the
/// next request appends. Append-only by construction: call `k`'s request is the context and every earlier answer and
/// observation.
struct Session {
    context: Vec<u32>,
    calls: Vec<(Vec<u32>, Vec<u32>)>,
}

impl Session {
    /// A session of `calls` calls whose answers are `answer` tokens and observations `observation` tokens, drawn
    /// deterministically from `seed` over a vocabulary of `vocab`.
    fn new(context: usize, calls: usize, answer: usize, observation: usize, vocab: u32, seed: u64) -> Session {
        let mut rng = Rng::new(seed);
        let mut draw = |n: usize| (0..n).map(|_| 1 + (rng.next_u64() % u64::from(vocab - 1)) as u32).collect::<Vec<u32>>();
        Session { context: draw(context), calls: (0..calls).map(|_| (draw(answer), draw(observation))).collect() }
    }

    /// The behaviour log-probability the sampler gave answer token `t` of call `k`: a value per token, the same in
    /// whichever sample holds it.
    fn behavior(k: usize, t: usize) -> f32 {
        -1.2 - 0.15 * ((3 * k + t) % 7) as f32
    }

    /// One record per call: the request (context, earlier answers and observations), the answer supervised.
    fn per_call(&self, advantage: f32) -> Vec<StreamExample> {
        let mut request = self.context.clone();
        let mut out = Vec::new();
        for (k, (answer, observation)) in self.calls.iter().enumerate() {
            let mut e = call(&request, answer, 0.0, advantage);
            (0..answer.len()).for_each(|t| e.behavior[request.len() + t] = Session::behavior(k, t));
            out.push(e);
            request.extend(answer);
            request.extend(observation);
        }
        out
    }

    /// The session as one sample: the last call's request and answer, every answer supervised where it was sampled.
    fn merged(&self, advantage: f32) -> StreamExample {
        let (mut ids, mut mask, mut behavior) = (self.context.clone(), vec![false; self.context.len()], vec![f32::NAN; self.context.len()]);
        for (k, (answer, observation)) in self.calls.iter().enumerate() {
            ids.extend(answer);
            mask.extend(std::iter::repeat_n(true, answer.len()));
            behavior.extend((0..answer.len()).map(|t| Session::behavior(k, t)));
            if k + 1 < self.calls.len() {
                ids.extend(observation);
                mask.extend(std::iter::repeat_n(false, observation.len()));
                behavior.extend(std::iter::repeat_n(f32::NAN, observation.len()));
            }
        }
        StreamExample { ids, mask, behavior, advantage, policy_version: None, top: None, support: None }
    }
}

/// A group of two sessions of 4 and 2 calls, rewarded apart, as one record per call or as one merged sample each.
fn sessions(vocab: u32) -> [(Session, f32); 2] {
    [(Session::new(5, 4, 3, 2, vocab, 21), 0.8), (Session::new(5, 2, 4, 3, vocab, 22), -0.8)]
}

fn group(examples: Vec<StreamExample>) -> Vec<StreamGroup> {
    vec![StreamGroup { key: "g".to_string(), examples }]
}

/// C2, the equivalence oracle: under the token sum, a session merged into one sample trains exactly what its per-call
/// records train. Every answer token has the same context (the session is append-only), the same behaviour
/// log-probability and advantage, and the step divides by the same count of policy tokens, so the gradients agree to
/// floating-point summation order while the merged step forwards each shared prefix once.
#[test]
fn a_merged_session_trains_its_per_call_records_gradient_under_the_token_sum() {
    const BLOCK: usize = 32;
    let vocab = QwenConfig::tiny().vocab;
    let sessions = sessions(vocab);
    let mut m = toy(BLOCK as u32, 5);
    let per_call = step_gradient(&mut m, group(sessions.iter().flat_map(|(s, a)| s.per_call(*a)).collect()), LossAgg::TokenSum, BLOCK);
    let merged = step_gradient(&mut m, group(sessions.iter().map(|(s, a)| s.merged(*a)).collect()), LossAgg::TokenSum, BLOCK);
    assert!(merged.iter().flatten().any(|g| g.abs() > 1e-6), "the step moves the adapters");
    let gap = relative_gap(&merged, &per_call);
    assert!(gap < 1e-4, "the merged sessions' gradient is {gap:e} (relative) from their per-call records'");
}

/// The per-sequence mean a stream can now train: each sample's token terms divided by its own count of policy tokens,
/// averaged over the step's samples, however many share a micro-batch. On merged sessions that weighs every session
/// alike, whatever its count of calls: the step's gradient is the mean of each session's own per-token mean, which
/// is each session alone in a step of its own.
#[test]
fn a_per_sequence_mean_over_merged_sessions_weighs_every_session_alike() {
    const BLOCK: usize = 32;
    let vocab = QwenConfig::tiny().vocab;
    let sessions = sessions(vocab);
    let mut m = toy(BLOCK as u32, 5);
    let both = step_gradient(&mut m, group(sessions.iter().map(|(s, a)| s.merged(*a)).collect()), LossAgg::SeqMean, BLOCK);
    let alone: Vec<Vec<Vec<f32>>> = sessions.iter().map(|(s, a)| step_gradient(&mut m, group(vec![s.merged(*a)]), LossAgg::SeqMean, BLOCK)).collect();
    let mean: Vec<Vec<f32>> = alone[0].iter().zip(&alone[1]).map(|(a, b)| a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)).collect()).collect();
    assert!(both.iter().flatten().any(|g| g.abs() > 1e-6), "the step moves the adapters");
    let gap = relative_gap(&both, &mean);
    assert!(gap < 1e-4, "a step over two merged sessions is {gap:e} (relative) from the mean of each session alone");
    // A single session alone under the per-sequence mean is its per-token mean: the token sum of a step of one sample.
    let token_sum = step_gradient(&mut m, group(vec![sessions[0].0.merged(sessions[0].1)]), LossAgg::TokenSum, BLOCK);
    assert!(relative_gap(&alone[0], &token_sum) < 1e-4, "one sample's sequence mean is its token mean");
}

/// C2, the cost: a group of four agent sessions of 12 calls each (a 48-token context, 12-token answers, 20-token
/// observations), trained as 48 call records and as 4 merged samples, gives the same gradient while the merged step
/// forwards the shared context once. Measured on the CPU at toy size; the token counts are exact, the seconds are
/// printed for the record (`--nocapture`).
#[test]
fn a_merged_group_forwards_each_sessions_context_once() {
    const BLOCK: usize = 416;
    let vocab = QwenConfig::tiny().vocab;
    let sessions: Vec<(Session, f32)> = (0..4).map(|i| (Session::new(48, 12, 12, 20, vocab, 40 + i), if i % 2 == 0 { 0.7 } else { -0.7 })).collect();
    let mut m = toy(BLOCK as u32, 9);
    let (per_call, per_call_work) = measured_step(&mut m, group(sessions.iter().flat_map(|(s, a)| s.per_call(*a)).collect()), LossAgg::TokenSum, BLOCK);
    let (merged, merged_work) = measured_step(&mut m, group(sessions.iter().map(|(s, a)| s.merged(*a)).collect()), LossAgg::TokenSum, BLOCK);
    let gap = relative_gap(&merged, &per_call);
    let secs = |w: &StepWork| w.forward_secs + w.backward_secs;
    eprintln!(
        "per-call: {} sequence tokens, {} forwarded, {:.3} s forward+backward; merged: {} sequence tokens, {} forwarded, {:.3} s; supervised {} vs {}; gradient gap {gap:e}",
        per_call_work.useful_tokens, per_call_work.computed_tokens, secs(&per_call_work), merged_work.useful_tokens, merged_work.computed_tokens, secs(&merged_work), per_call_work.supervised_positions, merged_work.supervised_positions
    );
    assert!(gap < 1e-4, "the merged group's gradient is {gap:e} (relative) from its per-call records'");
    assert_eq!(per_call_work.supervised_positions, merged_work.supervised_positions, "the same policy tokens train");
    // Per call, call k re-sends the context and the k earlier calls: 48 records of 48 + 32k + 12 tokens. Merged, each
    // session is its last request and answer once: 4 x (48 + 12 x 12 + 11 x 20) = 1648.
    assert_eq!(merged_work.useful_tokens, 4 * (48 + 12 * 12 + 11 * 20));
    assert_eq!(per_call_work.useful_tokens, 4 * (0..12u64).map(|k| 48 + 32 * k + 12).sum::<u64>());
}
