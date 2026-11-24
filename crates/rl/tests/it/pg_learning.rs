// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that reinforcement learning on a language model learns what its
// reward asks for, for its clients. If your team needs expertise in verified policy-gradient training then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Learning effectiveness of GRPO end to end ([`rl::objective::grpo::Grpo`] driven by [`model::fit_controlled`]: the
//! model's own sampling, the verifier, group advantages, the clipped objective, the weighted backward and AdamW), on a
//! tiny `qwen3::Qwen` from random weights, on the CPU in seconds.
//!
//! - A contextual bandit with one state and two actions (an even or an odd next token, so the untrained policy draws
//!   either about half the time): rewarding one action raises its probability, rewarding the other lowers it, and
//!   rewarding both alike moves nothing (every group is flat, so no step is taken).
//! - The same with two states that reward opposite actions: the policy learns each state's own action.
//! - A three-token sequence scored by how many positions match a target (optimum 1.0): the mean reward rises.
//!
//! Every claim is a threshold over several seeds, never one run's value. The same checks reject a run whose updates
//! are disabled (a learning rate of zero) or whose reward sign is reversed, and a learning rate of zero leaves every
//! parameter bit for bit.

use data::rng::Rng;
use model::rollout::{ModelRollout, Rollout, RolloutParams};
use model::serve::SampleParams;
use model::{fit_controlled, FitControl, FitOpts, Model};
use qwen3::{Qwen, QwenConfig};
use rl::env::{Environment, Reward, Step, Task, Verifier};
use rl::objective::grpo::{Grpo, GrpoConfig, GrpoOptions};
use rl::objective::pg_loss::PgLoss;

/// The prompt is `[1, 2, state]`, the completion `MAX_NEW` tokens: a row holds both.
const PROMPT_LEN: usize = 3;

/// Tasks cycle through `states` by the draw's seed.
struct Bandit {
    states: &'static [u32],
}

impl Environment for Bandit {
    fn name(&self) -> &str {
        "bandit"
    }
    fn tasks(&self, seed: u64) -> Vec<Task> {
        let state = self.states[(seed % self.states.len() as u64) as usize];
        vec![Task { id: format!("s{state}-{seed}"), prompt: vec![1, 2, state], answer: serde_json::json!(state) }]
    }
}

/// The reward of a completion in a state.
struct Judge(fn(u32, &[u32]) -> f32);

impl Verifier for Judge {
    fn verify(&self, task: &Task, _t: &[Step], completion: &[u32]) -> Reward {
        Reward { value: (self.0)(task.answer.as_u64().expect("the state") as u32, completion), parts: Default::default() }
    }
}

fn model(seed: u64, t: usize) -> Qwen {
    let cfg = QwenConfig::tiny();
    Qwen::new(cfg.clone(), 1, t as u32, &qwen3::init_weights(&cfg, seed))
}

/// `P(next token is even | [1, 2, state])` under the model's own softmax.
fn p_even(m: &Qwen, state: u32) -> f64 {
    let prompt = [1, 2, state];
    let logits = Model::logits_all(m, &prompt).expect("a decoder");
    let vocab = logits.len() / prompt.len();
    let row = &logits[(prompt.len() - 1) * vocab..];
    let top = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let w: Vec<f64> = row.iter().map(|&l| f64::from(l - top).exp()).collect();
    w.iter().step_by(2).sum::<f64>() / w.iter().sum::<f64>()
}

/// A GRPO run of `steps` optimiser steps of one group of `group` completions each.
fn train(m: Qwen, states: &'static [u32], reward: fn(u32, &[u32]) -> f32, max_new: usize, steps: u32, lr: f32, seed: u64) -> Qwen {
    let group = 8;
    let cfg = GrpoConfig {
        loss: PgLoss::ONLINE,
        group_size: group,
        seq_len: PROMPT_LEN + max_new,
        rollout: RolloutParams { max_new, sample: SampleParams { temp: 1.0, top_k: 0, top_p: 1.0, ..SampleParams::greedy() }, eos: None },
        max_attempts: 1,
        options: GrpoOptions::default(),
    };
    let obj = Grpo::try_new(Bandit { states }, Judge(reward), cfg).expect("a valid GRPO configuration");
    let opts = FitOpts { steps, batch_size: 1, block_size: (PROMPT_LEN + max_new) as u32, lr, min_lr: lr, warmup: 0, decay_iters: steps, weight_decay: 0.1, grad_clip: 1.0, grad_accum: group as u32, eval_interval: 0, eval_batches: 0, seed, checkpoint_secs: 0, ..FitOpts::default() };
    fit_controlled(m, obj, &opts, None, FitControl::default()).expect("the run").1
}

// seed: a fixed set of runs, so each learning threshold is a claim over the same five seeds every time
const SEEDS: [u64; 5] = [11, 23, 37, 41, 59];
const STEPS: u32 = 15;
const LR: f32 = 1e-2;

fn even_wins(_: u32, c: &[u32]) -> f32 {
    f32::from(c[0].is_multiple_of(2))
}
fn odd_wins(_: u32, c: &[u32]) -> f32 {
    f32::from(c[0] % 2 == 1)
}
fn all_win(_: u32, _: &[u32]) -> f32 {
    1.0
}

/// `P(even)` in `state` before and after a run, per seed.
fn bandit_moves(reward: fn(u32, &[u32]) -> f32, lr: f32) -> Vec<(f64, f64)> {
    SEEDS
        .iter()
        .map(|&seed| {
            let m = model(seed, PROMPT_LEN + 1);
            let before = p_even(&m, 3);
            let m = train(m, &[3], reward, 1, STEPS, lr, seed);
            (before, p_even(&m, 3))
        })
        .collect()
}

/// The bandit's learning claim: in every seed the rewarded action gains at least 0.1 of probability, and on average
/// across seeds at least 0.25 (`sign` +1: even is rewarded, -1: odd is).
fn check_learns(moves: &[(f64, f64)], sign: f64) -> Result<(), String> {
    let gains: Vec<f64> = moves.iter().map(|(b, a)| sign * (a - b)).collect();
    let mean = gains.iter().sum::<f64>() / gains.len() as f64;
    if gains.iter().any(|g| *g < 0.1) || mean < 0.25 {
        return Err(format!("the rewarded action's probability gains {gains:?} across seeds (mean {mean:.3}); every seed must gain 0.1 and the mean 0.25"));
    }
    Ok(())
}

/// Learning: one state, two actions near even odds. Rewarding even raises P(even); rewarding odd lowers it.
#[test]
fn the_bandit_learns_the_rewarded_action_and_unlearns_the_other() {
    let up = bandit_moves(even_wins, LR);
    println!("even rewarded: P(even) before -> after per seed {up:?}");
    assert!(up.iter().all(|(b, _)| (0.3..0.7).contains(b)), "the untrained policy draws either action about half the time: {up:?}");
    check_learns(&up, 1.0).unwrap();
    let down = bandit_moves(odd_wins, LR);
    println!("odd rewarded: P(even) before -> after per seed {down:?}");
    check_learns(&down, -1.0).unwrap();
}

/// Learning: when every action earns the same, every group is flat, no step is taken and the policy does not move.
#[test]
fn the_bandit_does_not_move_when_every_action_earns_the_same() {
    for (before, after) in bandit_moves(all_win, LR) {
        assert_eq!(before, after, "a run without a reward difference moved the policy");
    }
}

/// Mutation kills: the learning check rejects a run whose updates are disabled (learning rate zero), and a run whose
/// reward sign is reversed (it unlearns the action the check expects to rise).
#[test]
fn the_learning_check_rejects_disabled_updates_and_a_reversed_reward() {
    let frozen = bandit_moves(even_wins, 0.0);
    assert!(frozen.iter().all(|(b, a)| b == a), "a learning rate of zero moved the policy: {frozen:?}");
    let disabled = check_learns(&frozen, 1.0).expect_err("updates disabled");
    let reversed = check_learns(&bandit_moves(odd_wins, LR), 1.0).expect_err("reward sign reversed");
    println!("killed, updates disabled: {disabled}\nkilled, reward sign reversed: {reversed}");
}

/// Learning: two states that reward opposite actions. Each state's rewarded action gains at least 0.25, in every seed.
#[test]
fn a_two_state_bandit_learns_each_states_own_action() {
    fn by_state(state: u32, c: &[u32]) -> f32 {
        f32::from(c[0].is_multiple_of(2) == (state == 3))
    }
    let mut gains = Vec::new();
    for &seed in &SEEDS {
        let m = model(seed, PROMPT_LEN + 1);
        let before = (p_even(&m, 3), p_even(&m, 4));
        let m = train(m, &[3, 4], by_state, 1, 4 * STEPS, LR, seed);
        let after = (p_even(&m, 3), p_even(&m, 4));
        gains.push((after.0 - before.0, before.1 - after.1));
    }
    println!("two states: (gain of even in state 3, gain of odd in state 4) per seed {gains:?}");
    assert!(gains.iter().all(|(a, b)| *a > 0.25 && *b > 0.25), "each state learns its own action by 0.25 in every seed: {gains:?}");
}

/// The mean reward of `n` completions the model samples at temperature 1.
fn mean_reward(m: &Qwen, reward: fn(u32, &[u32]) -> f32, max_new: usize, n: usize, seed: u64) -> f64 {
    let params = RolloutParams { max_new, sample: SampleParams { temp: 1.0, top_k: 0, top_p: 1.0, ..SampleParams::greedy() }, eos: None };
    let samples = ModelRollout::new(m).sample_n(&[1, 2, 3], n, &params, &mut Rng::new(seed));
    samples.iter().map(|c| f64::from(reward(3, &c.tokens))).sum::<f64>() / n as f64
}

/// Learning: a three-token completion scored by the fraction of positions that match `[4, 5, 6]` (optimum 1.0, about
/// 0.04 for the untrained policy). Across seeds the sampled mean reward rises by at least 0.2 in every seed.
#[test]
fn grpo_raises_the_reward_of_a_short_sequence_task() {
    fn matches(_: u32, c: &[u32]) -> f32 {
        c.iter().zip([4u32, 5, 6]).filter(|(a, b)| **a == *b).count() as f32 / 3.0
    }
    let mut rises = Vec::new();
    for &seed in &SEEDS[..3] {
        let m = model(seed, PROMPT_LEN + 3);
        let before = mean_reward(&m, matches, 3, 256, seed);
        let m = train(m, &[3], matches, 3, 40, LR, seed);
        let after = mean_reward(&m, matches, 3, 256, seed);
        rises.push((before, after));
    }
    println!("sequence task: mean reward before -> after per seed {rises:?}");
    assert!(rises.iter().all(|(b, a)| a - b > 0.2), "the reward rises in every seed: {rises:?}");
}

/// A learning rate of zero (with weight decay on) leaves every parameter bit for bit, though steps are applied.
#[test]
fn a_learning_rate_of_zero_leaves_every_parameter_unchanged() {
    let m = model(5, PROMPT_LEN + 1);
    let before: Vec<Vec<u32>> = m.param_names().iter().map(|n| m.read_weight(n).iter().map(|v| v.to_bits()).collect()).collect();
    let m = train(m, &[3], even_wins, 1, 3, 0.0, 5);
    let after: Vec<Vec<u32>> = m.param_names().iter().map(|n| m.read_weight(n).iter().map(|v| v.to_bits()).collect()).collect();
    assert!(before == after, "a learning rate of zero changed a parameter");
}
