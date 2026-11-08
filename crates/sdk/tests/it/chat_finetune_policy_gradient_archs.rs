// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

#![cfg(all(feature = "study", feature = "text"))]

//! `brain::ChatFineTune::policy_gradient` on every trainable architecture of the CPU harness (`qwen3`, `qwen35`; toy
//! width): the objective is written once over the shared training contract, so the same spec holds for each. Records
//! of rollouts carrying an advantage make the actions of well-rewarded rollouts likelier and those of badly rewarded
//! ones less likely, the behaviour policy is scored lazily when the records carry none, and self-distillation records
//! train through the same run.

use std::path::{Path, PathBuf};

use brain::{BehaviourSource, ChatFineTune, FineTuneStatus, PolicyGradientConfig};

use crate::chat_fixture::{chat_model_dir, cpu, qwen35_model_dir, TempDir};

/// (prompt, answer, advantage): two actions of well-rewarded rollouts and two of badly rewarded ones.
const ROLLOUTS: [(&str, &str, f64); 4] = [("cab", "bad", 1.0), ("fed", "dad", 1.0), ("ace", "cab", -1.0), ("bag", "egg", -1.0)];

fn record(prompt: &str, answer: &str, reasoning: Option<&str>, advantage: Option<f64>) -> String {
    let mut assistant = serde_json::json!({"role": "assistant", "content": answer, "train": true});
    if let Some(r) = reasoning {
        assistant["reasoning_content"] = serde_json::json!(r);
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

fn base(model: &Path) -> String {
    model.join("model.safetensors").to_str().unwrap().to_string()
}

/// The trainable architectures of the harness.
const ARCHS: [&str; 2] = ["qwen3", "qwen35"];

fn tiny_model(arch: &str, tag: &str) -> TempDir {
    match arch {
        "qwen3" => chat_model_dir(tag),
        _ => qwen35_model_dir(tag),
    }
}

fn policy_gradient(arch: &str, model: &Path, dataset: &Path, out: &Path) -> ChatFineTune {
    ChatFineTune::from_pretrained(base(model))
        .arch(arch)
        .dataset(dataset)
        .out_dir(out)
        .rank(2)
        .steps(10)
        // The synthetic qwen3 fixture is near-uniform with tiny activations, so its adapter needs a larger step than qwen35 to move.
        .lr(if arch == "qwen3" { 0.03 } else { 2e-2 })
        .grad_accum(4)
        // seed: runs that are compared with each other need the same stream.
        .seed(7)
        .device(cpu())
        .policy_gradient(PolicyGradientConfig { behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() })
}

/// The mean log-probability of the actions of `ROLLOUTS` under `adapter` (per answer token for qwen3, summed for
/// qwen35: the direction is what is compared).
fn action_logprob(arch: &str, model: &Path, adapter: Option<&Path>) -> f32 {
    let plain = write_lines(&model.join("plain.jsonl"), &ROLLOUTS.iter().map(|(p, a, _)| record(p, a, None, None)).collect::<Vec<_>>());
    match arch {
        "qwen3" => -brain::score_chat(&base(model), adapter, &plain).expect("scoring").loss.expect("a loss"),
        _ => brain::score_tokens_qwen35(&base(model), adapter, &plain, Default::default(), &cpu()).expect("scoring").records.iter().flat_map(|r| r.positions.iter().map(|p| p.logprob)).sum::<f32>(),
    }
}

/// A positive advantage raises the log-probability of the trained action and a negative one lowers it, on every
/// architecture: the same actions, rewarded in one run and penalised in the other.
#[test]
fn a_positive_advantage_raises_and_a_negative_one_lowers_the_trained_action_on_every_architecture() {
    for arch in ARCHS {
        let model = tiny_model(arch, "pg-direction");
        let before = action_logprob(arch, &model, None);
        for advantage in [1.0f64, -1.0] {
            let dataset = write_lines(&model.join("rollouts.jsonl"), &ROLLOUTS.iter().map(|(p, a, _)| record(p, a, None, Some(advantage))).collect::<Vec<_>>());
            let out = model.join(format!("run{advantage}"));
            let outcome = policy_gradient(arch, &model, &dataset, &out).run().unwrap_or_else(|e| panic!("{arch}: the policy-gradient run: {e}"));
            assert_eq!(outcome.status, FineTuneStatus::Completed, "{arch}");
            let adapter = outcome.adapter.clone().expect("an adapter");
            let after = action_logprob(arch, &model, Some(&adapter));
            println!("{arch}: advantage {advantage}: action log-probability {before:.4} -> {after:.4}");
            assert!(advantage * f64::from(after - before) > 1e-3, "{arch}: advantage {advantage}: {before} -> {after}");

            let pg = outcome.policy_gradient.as_ref().expect("a policy-gradient run reports its statistics");
            assert_eq!((pg.stats.steps.len(), pg.stats.sequences_with_advantage, pg.stats.sequences_imitation), (10, 4, 0), "{arch}");
            assert_eq!((pg.stats.tokens_recorded_behavior, pg.stats.tokens_start_of_run_behavior), (0, 16), "{arch}: the behaviour policy is the start of the run, scored lazily");
            assert!((pg.stats.steps[0].ratio_mean - 1.0).abs() < 1e-4, "{arch}: the first step is on-policy: {:?}", pg.stats.steps[0]);
            assert!(pg.stats.steps.iter().all(|s| s.loss.is_finite() && s.clip_fraction.is_finite() && s.kl.is_some()), "{arch}");
            let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("a card");
            assert_eq!(card.training.expect("the card records its training").regime, "policy_gradient_lora", "{arch}");
        }
    }
}

#[test]
fn self_distillation_records_train_through_the_same_run_on_every_architecture() {
    for arch in ARCHS {
        let model = tiny_model(arch, "pg-teacher");
        let lines: Vec<String> = ROLLOUTS
            .iter()
            .map(|(p, a, _)| {
                let mut r: serde_json::Value = serde_json::from_str(&record(p, a, None, None)).unwrap();
                r["teacher_context"] = serde_json::json!({"at": 1, "messages": [{"role": "user", "content": "hint: the answer is short"}]});
                r.to_string()
            })
            .collect();
        let dataset = write_lines(&model.join("self-distill.jsonl"), &lines);
        let config = PolicyGradientConfig { teacher: brain::ReferencePolicy::StartOfRun, behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() };
        let outcome = policy_gradient(arch, &model, &dataset, &model.join("run")).policy_gradient(config).run().unwrap_or_else(|e| panic!("{arch}: {e}"));
        let pg = outcome.policy_gradient.expect("statistics");
        assert_eq!(pg.stats.sequences_with_teacher, 4, "{arch}");
        assert!(pg.stats.steps.iter().all(|s| s.loss.is_finite()), "{arch}");
    }
}
