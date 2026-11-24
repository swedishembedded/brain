// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reward-driven fine-tuning of looped mixture-of-experts models for its clients. If
// your team needs expertise in GRPO with routing replay over an expert decoder then you can procure our services by
// sending an email to info@swedishembedded.com.

//! GRPO on a tiny `inkling2::Inkling2Train` (a looped core with messages between passes, four experts of which two
//! route, a LoRA over every linear and merged site): the trained distribution is the sampled one, and a completion's
//! recorded MoE routing is replayed in every forward that scores or trains its row (`GrpoOptions::routing_replay`).

mod grpo_common;

use data::rng::Rng;
use grpo_common::{always_variable, config_loss, Env, Judge, PROMPT, T};
use inkling2::Inkling2Train;
use model::rollout::{Completion, Rollout, RolloutParams, StopReason, SyncRollout};
use model::routing::{RoutedCall, RoutingRecord};
use model::{Model, Objective};
use rl::objective::grpo::{Grpo, GrpoOptions};
use rl::objective::pg_loss::PgLoss;

fn seed() -> u64 {
    let s = data::rng::random_seed();
    println!("seed {s}");
    s
}

fn lora_model(seed: u64) -> Inkling2Train {
    let tc = inkling2::train::tiny_config().with_lora(model::adapter::device::LoraCfg::new(2, 4.0, inkling2::train::targets(false))).unwrap();
    let init = <Inkling2Train as Model>::init_weights(&tc, seed);
    let m = Inkling2Train::new(tc, 1, T as u32, &init);
    model::conformance_grad::randomize_adapters(&m, seed);
    m
}

#[test]
fn the_trained_distribution_is_the_sampled_one_on_inkling2() {
    let s = seed();
    for temp in [0.7f32, 1.5] {
        grpo_common::assert_trained_distribution_is_sampled(&mut lora_model(s), temp, s);
    }
}

/// A rollout that is not the model: it returns the scripted completions, in order, for every group.
struct Scripted {
    group: Vec<Completion>,
}

impl Rollout for Scripted {
    fn sample_n(&mut self, _prompt: &[u32], n: usize, _params: &RolloutParams, _rng: &mut Rng) -> Vec<Completion> {
        (0..n).map(|i| self.group[i % self.group.len()].clone()).collect()
    }
}

impl SyncRollout for Scripted {
    fn sync(&mut self, _trained: &std::collections::HashMap<String, Vec<f32>>) -> Result<(), String> {
        Ok(())
    }
}

fn first_wins(c: &[u32], _variable: bool) -> f32 {
    (c.first() == Some(&4)) as u32 as f32
}

/// The routing the trainer itself takes on `prompt + tokens` (laid over a row of `T`), as a completion's record;
/// `rotate` moves every selection to other experts (a sampler that routed differently).
fn routing_of(m: &Inkling2Train, tokens: &[u32], rotate: bool) -> RoutingRecord {
    let row: Vec<u32> = PROMPT.iter().chain(tokens).copied().chain(std::iter::repeat(0)).take(T).collect();
    model::training::TokenObjectives::set_routing_replay(m, None).unwrap();
    m.set_batch(model::Batch::Lm { tokens: &row, targets: &[model::IGNORE; T] });
    m.forward();
    let taken = m.taken_routing().unwrap();
    let (record, _) = taken.rows[0].clone().unwrap();
    let positions = PROMPT.len() + tokens.len();
    let (experts, k) = (m.config().experts, m.config().top_k as usize);
    // Rotating every id by one keeps a row's ids distinct and in range.
    let calls = record.calls.iter().map(|c| RoutedCall { top_k: c.top_k, experts: c.experts[..positions * k].iter().map(|&e| if rotate { (e + 1) % experts } else { e }).collect() }).collect();
    RoutingRecord { policy_version: 0, positions: positions as u32, recorded: vec![true; positions], calls }
}

fn completion(tokens: &[u32], routing: RoutingRecord) -> Completion {
    Completion { tokens: tokens.to_vec(), logprobs: vec![-1.0; tokens.len()], stop: StopReason::MaxNew, eos_logprob: None, policy_version: 0, cached_tokens: 0, routing: Some(routing) }
}

fn run(m: &mut Inkling2Train, group: Vec<Completion>, s: u64) -> rl::objective::grpo::GrpoStats {
    let (judge, _) = Judge::new(first_wins);
    let options = GrpoOptions { routing_replay: true, ..GrpoOptions::default() };
    let mut obj = Grpo::try_new(Env { variable: always_variable }, judge, config_loss(2, 1.0, options, PgLoss { log_kl: false, ..PgLoss::ONLINE })).expect("a valid GRPO configuration").with_rollout(Box::new(Scripted { group }), 1);
    Objective::<Inkling2Train>::prepare(&mut obj, m).unwrap();
    obj.micro_steps(m, &mut Rng::new(s), 2);
    obj.stats().clone()
}

/// Every trained row replays its completion's record: replaying the routing the trainer takes itself scores exactly
/// what it scores without replay, and a record whose experts differ is applied (the replayed scores move).
#[test]
fn grpo_replays_each_completions_recorded_routing_on_inkling2() {
    let s = seed();
    let tokens: [&[u32]; 2] = [&[4, 5, 6], &[6, 6, 5, 4]];
    let mut m = lora_model(s);
    let own: Vec<Completion> = tokens.iter().map(|t| completion(t, routing_of(&m, t, false))).collect();
    let st = run(&mut m, own, s);
    assert!(st.rows_trained > 0, "nothing was trained");
    assert_eq!(st.rows_routing_replayed, st.rows_trained, "every trained row carried a record");
    let (gap, replayed) = (st.serve_gap.expect("measured"), st.serve_gap_replayed.expect("measured with replay"));
    assert_eq!(gap.gap_mean, replayed.gap_mean, "replaying the trainer's own routing changes nothing");

    let mut m = lora_model(s);
    let other: Vec<Completion> = tokens.iter().map(|t| completion(t, routing_of(&m, t, true))).collect();
    let st = run(&mut m, other, s);
    assert_eq!(st.rows_routing_replayed, st.rows_trained);
    let (gap, replayed) = (st.serve_gap.expect("measured"), st.serve_gap_replayed.expect("measured with replay"));
    assert_ne!(gap.gap_mean, replayed.gap_mean, "a record with other experts is applied");
}
