// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reward-driven fine-tuning that
// can be trusted - on-policy ratios, steps that never run on empty batches,
// stop tokens that are learned - for its clients. If your team needs expertise
// in reinforcement learning for language models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec-level checks of the GRPO objective's corrections, on a tiny real
//! `qwen3::Qwen` with a rank-2 LoRA (CPU backend): the trained distribution is
//! the sampled one (R1), no optimiser step runs on an empty batch (R3), token
//! level normalisation is length invariant (R4), truncated completions are
//! rewarded on the span that is trained (R5), the stop token is a target and
//! is learned (R6), non-finite values never reach the weights (R7), stale rows
//! are refused (R8), the KL to the adapter-disabled reference is the analytic
//! one (R9), and a loss probe leaves the objective untouched.
//!
//! Seeds are drawn at run time and printed.

use data::rng::Rng;
use model::rollout::{Completion, Rollout, RolloutParams, StopReason, SyncRollout};
use model::{Model, Objective, StepVerdict, IGNORE};
use qwen3::{LoraCfg, Qwen, QwenConfig};
mod grpo_common;

use grpo_common::{always_variable, config, parity_reward, Env, Judge, T};
use rl::env::{Environment, Verifier};
use rl::objective::grpo::{CycleLog, Grpo, GrpoConfig, GrpoOptions, LossAgg, NonFiniteSteps, RatioBasis, RowRefusal};

const EOS: u32 = 7;

fn seed() -> u64 {
    let s = data::rng::random_seed();
    println!("seed {s}");
    s
}

/// A tiny decoder with a rank-2 LoRA whose `B` is non-zero, so the adapter
/// really differs from the base.
fn lora_model(seed: u64) -> (Qwen, QwenConfig, std::collections::HashMap<String, Vec<f32>>) {
    let cfg = QwenConfig { lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, seed);
    let m = Qwen::new(cfg.clone(), 1, T as u32, &init);
    model::conformance_grad::randomize_adapters(&m, seed);
    (m, cfg, init)
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

fn completion(tokens: &[u32]) -> Completion {
    Completion { tokens: tokens.to_vec(), logprobs: vec![-1.0; tokens.len()], stop: StopReason::MaxNew, eos_logprob: None }
}

fn stopped(tokens: &[u32]) -> Completion {
    Completion { stop: StopReason::Eos, eos_logprob: Some(-1.0), ..completion(tokens) }
}

/// Reward 1 for the first scripted completion (starts with 4), 0 for the others.
fn first_wins(c: &[u32], _variable: bool) -> f32 {
    (c.first() == Some(&4)) as u32 as f32
}

fn verdict<E: Environment, V: Verifier>(obj: &mut Grpo<E, V>) -> StepVerdict {
    Objective::<Qwen>::step_verdict(obj)
}

fn scripted<E: Environment, V: Verifier>(env: E, judge: V, cfg: GrpoConfig, group: Vec<Completion>) -> Grpo<E, V> {
    Grpo::new(env, judge, cfg).with_rollout(Box::new(Scripted { group }), 1)
}

/// R1: whatever the sampling temperature, the first update of a fresh row sees
/// an importance ratio of one at every token, and the probability the sampler
/// reported is the one the trainer assigns. (With the legacy objective the
/// trainer scored the untempered softmax and the ratio was 1.12 off at T=1.5.)
#[test]
fn the_trained_distribution_is_the_sampled_one_at_every_temperature() {
    let s = seed();
    for temp in [0.7f32, 1.0, 1.5] {
        let (mut m, _, _) = lora_model(s);
        grpo_common::assert_trained_distribution_is_sampled(&mut m, temp, s);
    }
}

/// The legacy basis exists to be measured against: at T = 1.5 the trainer scored
/// the untempered softmax, so a fresh row's first update starts off-policy.
#[test]
fn the_legacy_ratio_basis_starts_off_policy_at_a_non_unit_temperature() {
    let s = seed();
    let (mut m, _, _) = lora_model(s);
    let (judge, _) = Judge::new(parity_reward);
    let options = GrpoOptions { ratio_basis: RatioBasis::Sampler, ..GrpoOptions::default() };
    let mut obj = Grpo::new(Env { variable: always_variable }, judge, config(4, 1.5, options));
    obj.prepare(&mut m).unwrap();
    obj.micro_steps(&m, &mut Rng::new(s), 4);
    let st = obj.stats();
    assert!(st.rows_trained > 0);
    assert!(st.max_onpolicy_ratio_dev > 1e-2, "the legacy basis should start off policy, ratio deviation {}", st.max_onpolicy_ratio_dev);
}

/// R3: with most groups carrying no signal, a step holds exactly `grad_accum`
/// informative rows or is logged as short, and an optimiser step is never
/// allowed on zero rows.
#[test]
fn no_optimiser_step_runs_on_an_empty_batch() {
    let s = seed();
    let (mut m, _, _) = lora_model(s);
    grpo_common::assert_no_step_on_an_empty_batch(&mut m, s);
}

/// A run over an environment that never separates its completions takes no
/// optimiser step at all, even with weight decay that would move the weights.
#[test]
fn a_run_without_signal_leaves_the_weights_alone() {
    let s = seed();
    let (m, _, _) = lora_model(s);
    let before: Vec<Vec<f32>> = m.param_names().iter().map(|n| m.read_weight(n)).collect();
    let (judge, _) = Judge::new(parity_reward);
    let log = CycleLog::new();
    let obj = Grpo::new(Env { variable: |_| false }, judge, config(2, 1.0, GrpoOptions::default())).with_log(log.clone());
    let opts = model::FitOpts { steps: 3, grad_accum: 2, lr: 1e-2, min_lr: 1e-3, weight_decay: 0.5, eval_interval: 0, seed: s, ..model::FitOpts::default() };
    let (_, m) = model::fit_controlled(m, obj, &opts, None, Default::default()).expect("fit");
    let after: Vec<Vec<f32>> = m.param_names().iter().map(|n| m.read_weight(n)).collect();
    assert_eq!(before, after, "the weights moved without any signal");
    let st = log.stats();
    assert_eq!((st.steps_applied, st.rows_trained), (0, 0));
    assert!(st.resamples > 0, "the empty steps should have tried to resample");
}

/// R4: a token weighs the same in a long row as in a short one under
/// `TokenSum`; under `SeqMean` each row's mean counts once whatever its length.
#[test]
fn token_level_normalisation_is_length_invariant() {
    let s = seed();
    let group = vec![completion(&[4]), completion(&[5, 5, 5, 5])];
    for (agg, want) in [(LossAgg::TokenSum, 0.6f32), (LossAgg::SeqMean, 0.0)] {
        let (mut m, _, _) = lora_model(s);
        let (judge, _) = Judge::new(first_wins);
        let options = GrpoOptions { loss_agg: agg, ..GrpoOptions::default() };
        let mut obj = scripted(Env { variable: always_variable }, judge, config(2, 1.0, options), group.clone());
        obj.prepare(&mut m).unwrap();
        // Advantages are +0.5 (1 token) and -0.5 (4 tokens); on policy the per-token loss is -A.
        // TokenSum: denominator 5/2 -> (-0.5*1 + 0.5*4) / 2.5 = 0.6. SeqMean: -0.5 + 0.5 = 0.
        let loss = obj.micro_steps(&m, &mut Rng::new(s), 2);
        assert!((loss - want).abs() < 1e-3, "{agg:?}: step loss {loss}, expected {want}");
    }
}

/// R5: a completion longer than the row holds is cut before it is rewarded, so
/// what is trained is exactly what was scored.
#[test]
fn a_truncated_completion_is_rewarded_on_the_trained_span() {
    let s = seed();
    let (mut m, _, _) = lora_model(s);
    let (judge, seen) = Judge::new(first_wins);
    let log = CycleLog::new();
    // The row holds T - 3 = 5 completion tokens; the first completion has 7.
    let group = vec![completion(&[4, 4, 4, 4, 4, 4, 4]), completion(&[5, 5])];
    let mut obj = scripted(Env { variable: always_variable }, judge, config(2, 1.0, GrpoOptions::default()), group).with_log(log.clone());
    obj.prepare(&mut m).unwrap();
    obj.micro_steps(&m, &mut Rng::new(s), 2);
    assert!(seen.borrow().iter().all(|c| c.len() <= 5), "the verifier saw {:?}", seen.borrow());
    assert!(seen.borrow().contains(&vec![4; 5]), "the long completion was scored on its first 5 tokens");
    assert_eq!(obj.stats().completions_truncated, 1);
    assert!(log.trained().contains(&vec![4; 5]), "the trained span is the scored span: {:?}", log.trained());
}

/// R6: a sampled stop token is a training target, and the policy learns to stop.
#[test]
fn the_policy_learns_to_stop() {
    let s = seed();
    let (mut m, _, _) = lora_model(s);
    let (judge, _) = Judge::new(first_wins);
    let mut cfg = config(2, 1.0, GrpoOptions::default());
    cfg.rollout.eos = Some(EOS);
    // Stopping after [4, 5] earns the reward; running on to [5, 5, 5] does not.
    let group = vec![stopped(&[4, 5]), completion(&[5, 5, 5])];
    let mut obj = scripted(Env { variable: always_variable }, judge, cfg, group);
    obj.prepare(&mut m).unwrap();
    let stop_lp = |m: &Qwen| {
        let tokens = [1u32, 2, 3, 4, 5];
        let targets = [IGNORE, IGNORE, IGNORE, IGNORE, EOS];
        *model::logprobs::token_logprobs(m, &tokens, &targets).expect("logits").last().unwrap()
    };
    let before = stop_lp(&m);
    let mut rng = Rng::new(s);
    for step in 1..=12 {
        m.zero_grads();
        obj.micro_steps(&m, &mut rng, 2);
        assert!(matches!(verdict(&mut obj), StepVerdict::Apply));
        m.adamw_step(step, 3e-2, 0.0, Default::default(), Some(1.0), 0.5);
        m.poll_wait();
    }
    assert_eq!(obj.stats().eos_targets, 12, "one stop-token target per step");
    let after = stop_lp(&m);
    assert!(after > before + 0.05, "log P(stop | prompt, 4, 5) did not rise: {before} -> {after}");
}

/// R7: a non-finite log-probability never reaches the weights: the step is
/// skipped, and the run aborts with a typed error after three in a row.
#[test]
fn non_finite_values_skip_the_step_and_three_in_a_row_abort() {
    let s = seed();
    let (mut m, _, _) = lora_model(s);
    let (judge, _) = Judge::new(first_wins);
    let group = vec![completion(&[4, 4]), completion(&[5, 5])];
    let mut obj = scripted(Env { variable: always_variable }, judge, config(2, 1.0, GrpoOptions::default()), group);
    obj.prepare(&mut m).unwrap();
    let b = m.param_names().into_iter().find(|n| n.ends_with(".lora_b")).expect("a LoRA tensor");
    let mut poisoned = m.read_weight(&b);
    poisoned[0] = f32::NAN;
    m.write_weight(&b, &poisoned);
    let mut rng = Rng::new(s);
    for expected in 1..=2 {
        obj.micro_steps(&m, &mut rng, 2);
        match verdict(&mut obj) {
            StepVerdict::Skip(why) => assert!(why.contains("non-finite"), "{why}"),
            other => panic!("step {expected}: {other:?}"),
        }
    }
    obj.micro_steps(&m, &mut rng, 2);
    let StepVerdict::Abort(e) = verdict(&mut obj) else { panic!("the third bad step must abort") };
    let typed = e.get_ref().and_then(|source| source.downcast_ref::<NonFiniteSteps>()).expect("a typed error");
    assert_eq!(typed.consecutive, 3);
    assert_eq!(obj.stats().steps_applied, 0);
}

/// R8: a row older than the staleness bound is refused with a typed reason,
/// and one within it is trained.
#[test]
fn rows_older_than_the_staleness_bound_are_refused() {
    let s = seed();
    for (bound, refused) in [(0u32, 2usize), (1, 0)] {
        let (mut m, _, _) = lora_model(s);
        let (judge, _) = Judge::new(first_wins);
        // Four rows per group (rewards 1, 0, 1, 0), two consumed by each step.
        let group = vec![completion(&[4, 4]), completion(&[5, 5]), completion(&[4, 6]), completion(&[6, 6])];
        let options = GrpoOptions { staleness_bound: bound, ..GrpoOptions::default() };
        let mut obj = scripted(Env { variable: always_variable }, judge, config(4, 1.0, options), group);
        obj.prepare(&mut m).unwrap();
        let mut rng = Rng::new(s);
        obj.micro_steps(&m, &mut rng, 2);
        assert!(matches!(verdict(&mut obj), StepVerdict::Apply));
        obj.micro_steps(&m, &mut rng, 2);
        assert_eq!(obj.stats().rows_refused, refused, "bound {bound}");
        if refused > 0 {
            assert_eq!(obj.stats().last_refusal, Some(RowRefusal::Stale { age: 1, bound: 0 }));
        }
    }
}

/// Reference log-probabilities of a padded row for its targets.
type Reference<'a> = &'a dyn Fn(&[u32], &[u32]) -> Vec<f32>;

/// The mean k3 KL of the two trained rows' supervised tokens ([4, 5] and
/// [6, 6, 6] after the prompt) between `m` and `reference`, from independent forwards.
fn analytic_kl(m: &Qwen, reference: Reference) -> f32 {
    let (mut sum, mut n) = (0.0f32, 0usize);
    for toks in [&[1u32, 2, 3, 4, 5][..], &[1, 2, 3, 6, 6, 6][..]] {
        let mut padded = toks.to_vec();
        padded.resize(T, 0);
        // Position i predicts toks[i + 1]; the prompt's own positions are not trained.
        let targets: Vec<u32> = (0..T).map(|i| if i >= 2 && i + 1 < toks.len() { toks[i + 1] } else { IGNORE }).collect();
        let lp = model::logprobs::token_logprobs(m, &padded, &targets).expect("logits");
        let ref_lp = reference(&padded, &targets);
        for i in (0..T).filter(|&i| targets[i] != IGNORE) {
            let u = ref_lp[i] - lp[i];
            sum += u.exp() - u - 1.0;
            n += 1;
        }
    }
    sum / n as f32
}

/// R9: the logged KL is the analytic k3 mean against the reference. With a
/// caller-supplied constant reference it is computed from an independent
/// forward of the same rows; with none, the reference is the model with its
/// adapters disabled, i.e. a model built without them.
#[test]
fn the_logged_kl_is_the_analytic_one() {
    let s = seed();
    let group = vec![completion(&[4, 5]), completion(&[6, 6, 6])];

    // (a) a constant reference supplied by the caller.
    let (mut m, _, _) = lora_model(s);
    let (judge, _) = Judge::new(first_wins);
    let mut obj = scripted(Env { variable: always_variable }, judge, config(2, 1.0, GrpoOptions::default()), group.clone()).with_reference(|_, c| vec![-2.5; c.tokens.len()]);
    obj.prepare(&mut m).unwrap();
    obj.micro_steps(&m, &mut Rng::new(s), 2);
    let got = obj.stats().mean_kl().expect("a reference was supplied");
    let want = analytic_kl(&m, &|toks, _| vec![-2.5; toks.len()]);
    assert!((got - want).abs() < 1e-4, "logged {got}, analytic {want}");

    // (b) no reference: adapters disabled == a model built without them.
    let (mut m, cfg, init) = lora_model(s);
    let base_init: std::collections::HashMap<_, _> = init.into_iter().filter(|(k, _)| !k.contains(".lora_")).collect();
    let base = Qwen::new(QwenConfig { lora: None, ..cfg }, 1, T as u32, &base_init);
    let (judge, _) = Judge::new(first_wins);
    let mut obj = scripted(Env { variable: always_variable }, judge, config(2, 1.0, GrpoOptions::default()), group);
    obj.prepare(&mut m).unwrap();
    obj.micro_steps(&m, &mut Rng::new(s), 2);
    let got = obj.stats().mean_kl().expect("adapters can be disabled");
    let want = analytic_kl(&m, &|toks, targets| model::logprobs::token_logprobs(&base, toks, targets).expect("logits"));
    assert!(want > 1e-6, "the adapter should differ from the base");
    assert!((got - want).abs() < 1e-4, "logged {got}, analytic {want}");
}

/// A loss probe samples and scores a group without queuing, logging or counting it.
#[test]
fn a_loss_probe_leaves_the_objective_untouched() {
    let s = seed();
    let (mut m, _, _) = lora_model(s);
    let (judge, _) = Judge::new(parity_reward);
    let log = CycleLog::new();
    let mut obj = Grpo::new(Env { variable: always_variable }, judge, config(4, 1.0, GrpoOptions::default())).with_log(log.clone());
    obj.prepare(&mut m).unwrap();
    let loss = obj.loss_probe(&m, &mut Rng::new(s));
    assert!(loss.is_finite());
    assert_eq!(*obj.stats(), Default::default());
    assert!(log.sampled().is_empty() && log.trained().is_empty());
}
