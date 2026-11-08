// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

#![cfg(feature = "study")]

//! `brain::ChatFineTune::policy_gradient` on a `qwen35` base (the 27B's architecture at toy width, CPU backend):
//! records of rollouts, each carrying the advantage of its rollout, train a LoRA that makes the actions of
//! well-rewarded rollouts likelier and those of badly rewarded ones less likely; a run can continue an adapter and
//! name its reference; records and settings that cannot be honoured are refused before a device is claimed.

use std::path::{Path, PathBuf};

use brain::{BehaviourSource, ChatFineTune, FineTuneStatus, PolicyGradientConfig, ReferencePolicy};

use crate::chat_fixture::common::{cpu, qwen35_model_dir};

/// (prompt, answer, advantage): two actions of well-rewarded rollouts and two of badly rewarded ones.
const ROLLOUTS: [(&str, &str, f64); 4] = [("cab", "bad", 1.0), ("fed", "dad", 1.0), ("ace", "cab", -1.0), ("bag", "egg", -1.0)];

fn record(prompt: &str, answer: &str, advantage: Option<f64>, behavior: Option<Vec<f64>>) -> String {
    let mut assistant = serde_json::json!({"role": "assistant", "content": answer, "train": true});
    if let Some(b) = behavior {
        assistant["behavior_logprobs"] = serde_json::json!(b);
    }
    let mut record = serde_json::json!({"messages": [{"role": "user", "content": prompt, "train": false}, assistant], "tools": []});
    if let Some(a) = advantage {
        record["advantage"] = serde_json::json!(a);
    }
    record.to_string()
}

fn write_lines(path: &Path, lines: &[String]) -> PathBuf {
    std::fs::write(path, lines.join("\n")).unwrap();
    path.to_path_buf()
}

fn rollouts(dir: &Path) -> PathBuf {
    write_lines(&dir.join("rollouts.jsonl"), &ROLLOUTS.iter().map(|(p, a, adv)| record(p, a, Some(*adv), None)).collect::<Vec<_>>())
}

/// The same actions as plain records, for scoring.
fn plain(dir: &Path) -> PathBuf {
    write_lines(&dir.join("plain.jsonl"), &ROLLOUTS.iter().map(|(p, a, _)| record(p, a, None, None)).collect::<Vec<_>>())
}

fn base(model: &Path) -> String {
    model.join("model.safetensors").to_str().unwrap().to_string()
}

fn policy_gradient(model: &Path, dataset: &Path, out: &Path, config: PolicyGradientConfig) -> ChatFineTune {
    ChatFineTune::from_pretrained(base(model))
        .arch("qwen35")
        .dataset(dataset)
        .out_dir(out)
        .rank(2)
        .steps(10)
        .lr(2e-2)
        .grad_accum(4)
        // seed: runs that are compared with each other need the same stream.
        .seed(7)
        .device(cpu())
        .policy_gradient(config)
}

/// The summed log-probability of the rewarded and of the penalised actions under `adapter`.
fn action_logprobs(model: &Path, plain: &Path, adapter: Option<&Path>) -> (f32, f32) {
    let readout = brain::score_tokens_qwen35(&base(model), adapter, plain, Default::default(), &cpu()).expect("scoring");
    let sum = |records: std::ops::Range<usize>| readout.records[records].iter().flat_map(|r| r.positions.iter().map(|p| p.logprob)).sum::<f32>();
    (sum(0..2), sum(2..4))
}

#[test]
fn rollouts_train_an_adapter_that_raises_rewarded_actions_and_lowers_penalised_ones() {
    let model = qwen35_model_dir("pg-direction");
    let (dataset, plain) = (rollouts(&model), plain(&model));
    let (good_before, bad_before) = action_logprobs(&model, &plain, None);

    let outcome = policy_gradient(&model, &dataset, &model.join("run"), PolicyGradientConfig { behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }).run().expect("the policy-gradient run");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let adapter = outcome.adapter.clone().expect("an adapter");
    let (good_after, bad_after) = action_logprobs(&model, &plain, Some(&adapter));
    println!("rewarded actions {good_before:.4} -> {good_after:.4}, penalised actions {bad_before:.4} -> {bad_after:.4}");
    assert!(good_after > good_before + 1e-3, "{good_before} -> {good_after}");
    assert!(bad_after < bad_before - 1e-3, "{bad_before} -> {bad_after}");

    // How it was trained is on the outcome, the training record and the adapter's card.
    let pg = outcome.policy_gradient.as_ref().expect("a policy-gradient run reports its statistics");
    assert_eq!((pg.stats.steps.len(), pg.stats.sequences_with_advantage, pg.stats.sequences_imitation), (10, 4, 0));
    assert_eq!((pg.stats.tokens_recorded_behavior, pg.stats.tokens_start_of_run_behavior), (0, 16), "four answers of four tokens, the end-of-turn marker included");
    assert!((pg.stats.steps[0].ratio_mean - 1.0).abs() < 1e-4, "the first step is on-policy: {:?}", pg.stats.steps[0]);
    assert!(pg.stats.steps.iter().all(|s| s.loss.is_finite() && s.clip_fraction.is_finite() && s.kl.is_some()));
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("a card");
    let training = card.training.expect("the card records its training");
    assert_eq!(training.regime, "policy_gradient_lora");
    let settings = &training.hyperparams["policy_gradient"];
    assert_eq!((settings["clip_eps"].as_f64(), settings["kl_beta"].as_f64(), settings["reference"].as_str()), (Some(0.2f32 as f64), Some(0.02f32 as f64), Some("base")));
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.unwrap()).unwrap()).unwrap();
    assert_eq!(record["policy_gradient"]["steps"].as_array().map(Vec::len), Some(10));
}

#[test]
fn recorded_behaviour_log_probs_are_the_behaviour_policy_and_the_rest_is_the_start_of_the_run() {
    let model = qwen35_model_dir("pg-behaviour");
    // Two actions come with the sampler's log-probabilities (four tokens: the answer and its end-of-turn marker).
    let lines: Vec<String> = ROLLOUTS.iter().enumerate().map(|(i, (p, a, adv))| record(p, a, Some(*adv), (i < 2).then(|| vec![-3.0; 4]))).collect();
    let dataset = write_lines(&model.join("rollouts.jsonl"), &lines);
    let outcome = policy_gradient(&model, &dataset, &model.join("run"), PolicyGradientConfig { behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }).run().expect("the run");
    let pg = outcome.policy_gradient.expect("statistics");
    assert_eq!((pg.stats.tokens_recorded_behavior, pg.stats.tokens_start_of_run_behavior), (8, 8));
    // A recorded log-probability of -3 against a policy near uniform makes the ratios large: some tokens are clipped.
    assert!(pg.stats.steps.iter().any(|s| s.clip_fraction > 0.0), "{:?}", pg.stats.steps.iter().map(|s| s.clip_fraction).collect::<Vec<_>>());
}

#[test]
fn a_run_continues_an_adapter_and_its_reference_is_the_base_or_that_adapter() {
    let model = qwen35_model_dir("pg-continue");
    let dataset = rollouts(&model);
    let first = policy_gradient(&model, &dataset, &model.join("first"), PolicyGradientConfig { behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }).run().expect("the first run");
    let parent = first.adapter.expect("the first adapter");

    let from_base = policy_gradient(&model, &dataset, &model.join("second"), PolicyGradientConfig { reference: ReferencePolicy::Base, behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }).continue_from(&parent).run().expect("continuing against the base");
    assert!(from_base.trained_from.is_some(), "the parent is recorded");
    let kl_to_base = from_base.policy_gradient.unwrap().stats.steps[0].kl.expect("a measured KL");
    assert!(kl_to_base > 1e-6, "the continued adapter has left the base: {kl_to_base}");

    let from_start = policy_gradient(&model, &dataset, &model.join("third"), PolicyGradientConfig { reference: ReferencePolicy::StartOfRun, behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }).continue_from(&parent).run().expect("continuing against its own start");
    let kl_to_start = from_start.policy_gradient.unwrap().stats.steps[0].kl.expect("a measured KL");
    assert!(kl_to_start.abs() < 1e-6, "the first step sits on the policy it started from: {kl_to_start}");
}

#[test]
fn records_and_settings_a_policy_gradient_run_cannot_honour_are_refused_before_training() {
    let model = qwen35_model_dir("pg-refused");
    let dataset = rollouts(&model);
    let out = model.join("run");
    let refused = |run: ChatFineTune| run.run().expect_err("refused").to_string();

    // An advantage is never silently ignored by supervised training.
    let sft = ChatFineTune::from_pretrained(base(&model)).arch("qwen35").dataset(&dataset).out_dir(&out).steps(2).device(cpu());
    let why = refused(sft);
    assert!(why.contains("advantage") && why.contains("policy_gradient"), "{why}");

    // And the objective that reads one needs some.
    let plain = plain(&model);
    let why = refused(policy_gradient(&model, &plain, &out, Default::default()));
    assert!(why.contains("no record") && why.contains("advantage"), "{why}");

    let why = refused(policy_gradient(&model, &dataset, &out, PolicyGradientConfig { clip_eps: 1.5, behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }));
    assert!(why.contains("clip_eps"), "{why}");
    let why = refused(policy_gradient(&model, &dataset, &out, Default::default()).eval_every(2));
    assert!(why.contains("eval_every"), "{why}");
    let why = refused(policy_gradient(&model, &dataset, &out, Default::default()).held_out(&plain));
    assert!(why.contains("held-out"), "{why}");

    // A sampler's log-probabilities that do not number the answer's tokens name the record and both counts.
    let bad = write_lines(&model.join("bad.jsonl"), &[record("cab", "bad", Some(1.0), None), record("fed", "dad", Some(-1.0), Some(vec![-1.0; 3]))]);
    let why = refused(policy_gradient(&model, &bad, &out, Default::default()));
    assert!(why.contains("bad.jsonl") && why.contains("record 2") && why.contains("behavior_logprobs") && why.contains('3') && why.contains('4'), "{why}");
    assert!(!out.join("adapter.safetensors").exists(), "nothing was trained");
}

/// Self-distillation: records with a privileged teacher context and no advantage train through the same run; the
/// teacher's per-token advantages are on the outcome, and the run reports its guards.
#[test]
fn records_with_a_teacher_context_train_on_the_teachers_per_token_advantage() {
    let model = qwen35_model_dir("pg-self-distill");
    let lines: Vec<String> = ROLLOUTS
        .iter()
        .map(|(p, a, _)| {
            let mut r: serde_json::Value = serde_json::from_str(&record(p, a, None, None)).unwrap();
            r["teacher_context"] = serde_json::json!({"at": 1, "messages": [{"role": "user", "content": "hint: the answer is short"}]});
            r.to_string()
        })
        .collect();
    let dataset = write_lines(&model.join("self-distill.jsonl"), &lines);
    let config = PolicyGradientConfig { teacher: ReferencePolicy::StartOfRun, advantage_bound: Some(2.0), sequence_advantage_cap: Some(3.0), behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() };
    let outcome = policy_gradient(&model, &dataset, &model.join("run"), config).run().expect("the self-distillation run");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let pg = outcome.policy_gradient.expect("statistics");
    assert_eq!(pg.stats.sequences_with_teacher, 4);
    assert!(pg.stats.steps.iter().all(|s| s.loss.is_finite() && s.token_advantage_max <= 2.0 + 1e-6));
    assert!(pg.stats.steps[0].token_advantage_max > 0.0, "a hint in the context must change the teacher's view: {:?}", pg.stats.steps[0]);
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.unwrap()).unwrap()).unwrap();
    assert_eq!(record["policy_gradient"]["teacher"], "start");
}

#[test]
fn every_step_reports_what_the_objective_measured_while_the_run_is_still_going() {
    let model = qwen35_model_dir("pg-progress");
    let dataset = rollouts(&model);
    let mut frames: Vec<brain::FineTuneProgress> = Vec::new();
    let outcome = policy_gradient(&model, &dataset, &model.join("run"), PolicyGradientConfig { behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() })
        .run_with(&brain::CancelToken::default(), |p| frames.push(p.clone()))
        .expect("the run");
    let stats = outcome.policy_gradient.expect("statistics").stats;
    assert_eq!(frames.len(), 10, "one frame per optimiser step");
    for (frame, step) in frames.iter().zip(&stats.steps) {
        let metric = |name: &str| frame.metrics.iter().find(|(n, _)| n == name).map(|(_, v)| *v).unwrap_or_else(|| panic!("step {}: no metric {name}: {:?}", frame.step, frame.metrics));
        // The frame is the step's own statistics, not the run's final ones.
        assert_eq!(metric("pg_loss"), step.loss);
        assert_eq!(metric("pg_clip_fraction"), step.clip_fraction);
        assert_eq!(metric("pg_ratio_mean"), step.ratio_mean);
        assert_eq!(metric("pg_mean_advantage"), step.mean_advantage);
        assert_eq!(Some(metric("pg_kl")), step.kl);
        assert_eq!(metric("pg_scoring_seconds"), step.scoring_seconds);
    }
}
