// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The policy-gradient objective scores an example when a step first draws it ([`rl::objective::external`]), on a tiny
//! `qwen35` LoRA model: an example no step draws is never scored, one drawn again is read back rather than scored again,
//! and what is measured is what a pass over the whole dataset before the first step would have measured -- the
//! behaviour log-probabilities are the start-of-run policy's, however far the adapter has moved when the example is
//! first drawn.

use std::path::Path;

use data::chat::{EncodedSample, EncodedSplit};
use data::rng::Rng;
use model::training;
use model::{Batch, FitOpts, Model, Objective, StepVerdict};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};
use rl::objective::external::{BehaviourSource, ExternalPolicyGradient, PolicyGradientConfig, ReferencePolicy};

const T: usize = 8;
const ANSWER_FROM: usize = 3;
const STUDENT_LEN: usize = 6;
/// The tokens a teacher context adds in front of the student's.
const PRIVILEGE: [u32; 2] = [20, 21];

fn student_ids(e: usize) -> Vec<u32> {
    (0..STUDENT_LEN).map(|i| 1 + ((3 * e + i) % 25) as u32).collect()
}

fn write_dataset(dir: &Path, examples: usize, teacher: bool) {
    let mut split = EncodedSplit::default();
    for e in 0..examples {
        let ids = student_ids(e);
        let mask: Vec<bool> = (0..ids.len()).map(|i| i >= ANSWER_FROM).collect();
        let teacher = teacher.then(|| {
            let with: Vec<u32> = PRIVILEGE.iter().copied().chain(ids.iter().copied()).collect();
            let lead = PRIVILEGE.len() + ANSWER_FROM;
            let mask = (0..with.len()).map(|i| i >= lead).collect::<Vec<bool>>();
            (with, mask)
        });
        let advantage = if e % 2 == 0 { 1.0 } else { -1.0 };
        split.push_rollout(&EncodedSample { ids, mask, behavior: vec![f32::NAN; STUDENT_LEN], teacher, ..Default::default() }, &[], Some(advantage));
    }
    data::chat::write_split(dir, "train", &split).unwrap();
    data::chat::write_split(dir, "val", &split).unwrap();
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(29)).unwrap();
}

fn is_zero_started(name: &str) -> bool {
    name.ends_with(".lora_b")
}

/// A model whose adapter moves the distribution off its base; `fresh` zeroes the up projections so the adapter is the base.
fn lora_model(seed: u64, fresh: bool) -> Qwen35 {
    let cfg = Qwen35Config { lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let init = qwen35::init::init_weights(&cfg, seed);
    let m = Qwen35::new_train_on(gpu_core::Gpu::new(pipelines()), cfg, 1, T as u32, &init);
    let mut rng = Rng::new(seed);
    for name in m.param_names().into_iter().filter(|n| n.contains(".lora_")) {
        let n = m.read_weight(&name).len();
        let w: Vec<f32> = if fresh && is_zero_started(&name) { vec![0.0; n] } else { (0..n).map(|_| 1.5 * (rng.next_f32() - 0.5)).collect() };
        m.write_weight(&name, &w);
    }
    m
}

/// Stand in for optimiser steps: every adapter tensor moves.
fn move_adapter(m: &Qwen35, seed: u64) {
    let mut rng = Rng::new(seed);
    for name in m.param_names().into_iter().filter(|n| n.contains(".lora_")) {
        let w: Vec<f32> = m.read_weight(&name).iter().map(|x| x + 0.4 * (rng.next_f32() - 0.5)).collect();
        m.write_weight(&name, &w);
    }
    m.poll_wait();
}

fn adapter_weights(m: &Qwen35) -> Vec<Vec<f32>> {
    m.param_names().iter().filter(|n| n.contains(".lora_")).map(|n| m.read_weight(n)).collect()
}

/// The model's log-probability, now, of the token each row position predicts, for `ids` supervised from `from`.
fn logprobs_now(m: &Qwen35, ids: &[u32], from: usize) -> Vec<f32> {
    let (mut tokens, mut targets) = (vec![0u32; T], vec![model::IGNORE; T]);
    tokens[..ids.len()].copy_from_slice(ids);
    let (first, last) = (from - 1, ids.len() - 1);
    targets[first..last].copy_from_slice(&ids[from..]);
    Model::set_batch(m, Batch::Lm { tokens: &tokens, targets: &targets });
    let _ = m.forward();
    let lp = training::objectives(m).batch_token_logprobs();
    lp[first..last].to_vec()
}

/// The same under the base: the up projections zeroed for the call.
fn base_logprobs(m: &Qwen35, ids: &[u32], from: usize) -> Vec<f32> {
    rl::objective::pg_batch::with_adapters_disabled(m, || logprobs_now(m, ids, from)).expect("a LoRA model")
}

fn teacher_logprobs(m: &Qwen35, e: usize, policy: impl Fn(&Qwen35, &[u32], usize) -> Vec<f32>) -> Vec<f32> {
    let with: Vec<u32> = PRIVILEGE.iter().copied().chain(student_ids(e)).collect();
    policy(m, &with, PRIVILEGE.len() + ANSWER_FROM)
}

fn config() -> PolicyGradientConfig {
    PolicyGradientConfig { kl_beta: 0.02, log_kl: true, clip_eps: 0.5, behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }
}

fn opts() -> FitOpts {
    FitOpts { steps: 1, batch_size: 1, block_size: T as u32, lr: 1e-2, grad_accum: 1, seed: 17, ..Default::default() }
}

/// An objective opened on `examples` examples, prepared on `m` and positioned at draw `draws_done`.
fn started(examples: usize, teacher: bool, cfg: PolicyGradientConfig, m: &mut Qwen35, draws_done: u64) -> (ExternalPolicyGradient, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), examples, teacher);
    let mut obj = ExternalPolicyGradient::open(dir.path(), &opts(), cfg).unwrap();
    Objective::<Qwen35>::prepare(&mut obj, m).unwrap();
    // As the loop does before its first step: a resumed run has applied one update per draw so far.
    Objective::<Qwen35>::updates_applied(&mut obj, draws_done as u32);
    Objective::<Qwen35>::begin(&mut obj, 17, draws_done);
    (obj, dir)
}

fn step(obj: &mut ExternalPolicyGradient, m: &Qwen35) {
    m.zero_grads();
    let _ = Objective::<Qwen35>::micro_steps(obj, m, &mut Rng::new(1), 1);
    m.poll_wait();
    assert!(matches!(Objective::<Qwen35>::step_verdict(obj), StepVerdict::Apply));
    // As the loop does once the update is applied.
    let applied = obj.stats().steps.len() as u32;
    Objective::<Qwen35>::updates_applied(obj, applied);
}

fn drawn(obj: &ExternalPolicyGradient, examples: usize) -> Vec<usize> {
    (0..examples).filter(|&e| obj.scores_of(e).is_some()).collect()
}

#[test]
fn an_example_no_step_draws_is_never_scored_and_one_drawn_again_is_scored_once() {
    let mut m = lora_model(data::rng::random_seed(), false);
    let examples = 6;
    let (mut obj, _dir) = started(examples, true, config(), &mut m, 0);
    assert_eq!(obj.stats().scored_behavior + obj.stats().scored_reference + obj.stats().scored_teacher, 0, "nothing is scored before a step asks");

    for _ in 0..3 {
        step(&mut obj, &m);
    }
    let first_three = drawn(&obj, examples);
    assert_eq!(first_three.len(), 3, "an epoch draws each example once, so three steps scored three examples");
    let s = obj.stats();
    assert_eq!((s.scored_behavior, s.scored_reference, s.scored_teacher), (3, 3, 3));
    assert_eq!(s.steps.iter().filter(|st| st.scoring_seconds > 0.0).count(), 3, "every step drew a new example and timed its scoring");

    for _ in 0..3 {
        step(&mut obj, &m);
    }
    assert_eq!(drawn(&obj, examples).len(), examples);
    let disk = obj.score_cache_bytes().1;
    for _ in 0..examples {
        step(&mut obj, &m);
    }
    let s = obj.stats();
    assert_eq!((s.scored_behavior, s.scored_reference, s.scored_teacher), (examples, examples, examples), "a second epoch scores nothing");
    assert_eq!(obj.score_cache_bytes().1, disk, "and writes nothing");
    assert!(s.steps[2 * examples - examples..].iter().all(|st| st.scoring_seconds == 0.0));
}

#[test]
fn the_memory_the_scores_hold_does_not_grow_with_what_was_scored() {
    let mut m = lora_model(data::rng::random_seed(), false);
    let (mut obj, _dir) = started(6, true, config(), &mut m, 0);
    let before = obj.score_cache_bytes();
    assert_eq!(before.1, 0);
    for _ in 0..6 {
        step(&mut obj, &m);
    }
    let after = obj.score_cache_bytes();
    assert_eq!(after.0, before.0, "the scores of every example live on disk; memory keeps a slot per example");
    assert!(after.1 > 0);
}

/// An adapter that has moved since the run began, scored when an example is first drawn.
fn behaviour_is_the_start_policys_after_the_adapter_moved(fresh: bool) {
    let seed = data::rng::random_seed();
    println!("seed {seed}");
    let mut m = lora_model(seed, fresh);
    let examples = 5;
    let start: Vec<Vec<f32>> = (0..examples).map(|e| logprobs_now(&m, &student_ids(e), ANSWER_FROM)).collect();
    let (mut obj, _dir) = started(examples, true, config(), &mut m, 0);

    for round in 0..examples {
        if round > 0 {
            move_adapter(&m, 100 + round as u64);
        }
        let moved = adapter_weights(&m);
        step(&mut obj, &m);
        assert_eq!(adapter_weights(&m), moved, "scoring puts the live adapter back exactly as it was");
    }
    assert_eq!(drawn(&obj, examples).len(), examples);
    for (e, start_e) in start.iter().enumerate() {
        let scores = obj.scores_of(e).unwrap();
        let behavior = scores.behavior.expect("no log-probabilities were recorded");
        let supervised = &behavior[ANSWER_FROM - 1..STUDENT_LEN - 1];
        assert_eq!(supervised, &start_e[..], "example {e}: the behaviour log-probabilities are the start policy's, bit for bit");
        let now = logprobs_now(&m, &student_ids(e), ANSWER_FROM);
        assert!(now.iter().zip(supervised).any(|(a, b)| (a - b).abs() > 1e-3), "example {e}: the adapter moved, so the test tells the two policies apart");

        // The reference and the teacher are the base (the configuration's default); the advantage is the base teacher's
        // log-probability against the start-policy behaviour.
        let reference = scores.reference.unwrap();
        assert_eq!(&reference[ANSWER_FROM - 1..STUDENT_LEN - 1], &base_logprobs(&m, &student_ids(e), ANSWER_FROM)[..], "example {e}: reference");
        let adv = scores.teacher.unwrap();
        for (i, t) in (ANSWER_FROM - 1..STUDENT_LEN - 1).enumerate() {
            let want = teacher_logprobs(&m, e, base_logprobs)[i] - start_e[i];
            assert_eq!(adv.value[t].to_bits(), want.to_bits(), "example {e} position {t}: teacher advantage");
        }
    }
}

#[test]
fn a_continued_adapter_is_scored_under_a_frozen_copy_of_the_start_adapter() {
    behaviour_is_the_start_policys_after_the_adapter_moved(false);
}

#[test]
fn a_fresh_adapter_is_scored_as_the_base_after_it_moved() {
    behaviour_is_the_start_policys_after_the_adapter_moved(true);
}

#[test]
fn a_resumed_run_scores_under_the_adapter_it_started_from_not_the_one_it_restored() {
    let mut m = lora_model(data::rng::random_seed(), false);
    let start = logprobs_now(&m, &student_ids(0), ANSWER_FROM);
    // `prepare` runs before the weights of the checkpoint are restored; the run resumes at draw 4.
    let (mut obj, _dir) = started(1, false, config(), &mut m, 4);
    move_adapter(&m, 7);
    step(&mut obj, &m);
    let behavior = obj.scores_of(0).unwrap().behavior.unwrap();
    assert_eq!(&behavior[ANSWER_FROM - 1..STUDENT_LEN - 1], &start[..]);
}

#[test]
fn start_of_run_reference_and_teacher_share_the_behaviour_policy() {
    let mut m = lora_model(data::rng::random_seed(), false);
    let examples = 2;
    let start: Vec<Vec<f32>> = (0..examples).map(|e| logprobs_now(&m, &student_ids(e), ANSWER_FROM)).collect();
    let start_teacher: Vec<Vec<f32>> = (0..examples).map(|e| teacher_logprobs(&m, e, logprobs_now)).collect();
    let cfg = PolicyGradientConfig { reference: ReferencePolicy::StartOfRun, teacher: ReferencePolicy::StartOfRun, ..config() };
    let (mut obj, _dir) = started(examples, true, cfg, &mut m, 0);
    step(&mut obj, &m);
    move_adapter(&m, 9);
    step(&mut obj, &m);
    for e in 0..examples {
        let scores = obj.scores_of(e).unwrap();
        assert_eq!(scores.reference.as_ref(), scores.behavior.as_ref(), "example {e}: one pass serves both");
        let adv = scores.teacher.unwrap();
        for (i, t) in (ANSWER_FROM - 1..STUDENT_LEN - 1).enumerate() {
            assert_eq!(adv.value[t].to_bits(), (start_teacher[e][i] - start[e][i]).to_bits(), "example {e} position {t}");
        }
    }
}
