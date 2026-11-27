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
use model::{Model, Objective, StepVerdict, Supply};
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
    let tokens: usize = groups.iter().map(StreamGroup::policy_tokens).sum();
    let settings = StreamSettings { step_tokens: tokens, queue_tokens: tokens, max_staleness: None, block, top_k: None };
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
    Model::param_names(m).into_iter().filter(|n| n.contains(".lora_")).map(|n| m.read_grad(&n)).collect()
}

/// One call record: `prompt` as context, then `answer` supervised with behaviour log-probabilities `behavior`.
fn call(prompt: &[u32], answer: &[u32], behavior: f32, advantage: f32) -> StreamExample {
    let ids: Vec<u32> = prompt.iter().chain(answer).copied().collect();
    let mask: Vec<bool> = (0..ids.len()).map(|t| t >= prompt.len()).collect();
    let behavior = mask.iter().map(|&m| if m { behavior } else { f32::NAN }).collect();
    StreamExample { ids, mask, behavior, advantage, policy_version: None, top: None }
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
