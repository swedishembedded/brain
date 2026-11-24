// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for stable off-policy reinforcement learning of language models from
// asynchronously sampled rollouts for its clients. If your team needs expertise in policy-gradient estimators that stay
// unbiased when the sampler lags the learner then you can procure our services by sending an email to
// info@swedishembedded.com.

#![cfg(all(feature = "study", feature = "text"))]

//! `brain::ChatFineTune::policy_gradient` with the score-centred estimator, end to end on the CPU, on a tiny `qwen3`
//! and a tiny `qwen35` base: records of rollouts carrying the sampler's behaviour log-probabilities and its top-K on
//! every answer train a LoRA that raises the rewarded actions and lowers the penalised ones, every step reports a
//! finite loss, the sampler's listed mass and the share of masked tokens, and the settings are on the adapter's card.
//! Records it cannot centre are refused before training.

use std::path::{Path, PathBuf};

use brain::{ChatFineTune, Estimator, FineTuneStatus, PgLoss, PolicyGradientConfig};

use crate::chat_fixture::{chat_model_dir, cpu, qwen35_model_dir, TempDir};

/// (prompt, answer, reward): one group, mean 0, so the mean-centred advantages are the rewards.
const ROLLOUTS: [(&str, &str, f64); 4] = [("cab", "bad", 1.0), ("fed", "dad", 1.0), ("ace", "cab", -1.0), ("bag", "egg", -1.0)];
/// Every answer renders to four tokens (the three letters' token and the end-of-turn marker included).
const ANSWER_TOKENS: usize = 4;
const TOP: [u32; 3] = [100, 200, 300];
const ARCHS: [&str; 2] = ["qwen3", "qwen35"];

fn plain_record(prompt: &str, answer: &str) -> serde_json::Value {
    serde_json::json!({"messages": [{"role": "user", "content": prompt, "train": false}, {"role": "assistant", "content": answer, "train": true}], "tools": []})
}

/// A rollout record whose sampler drew each token with log-probability `behaviour[t]` and listed [`TOP`] at the same
/// probability each.
fn record(prompt: &str, answer: &str, reward: f64, behaviour: &[f32], sampler: Option<serde_json::Value>, top: bool) -> String {
    let mut r = plain_record(prompt, answer);
    r["messages"][1]["behavior_logprobs"] = serde_json::json!(behaviour);
    if top {
        r["messages"][1]["behavior_top_ids"] = serde_json::json!(behaviour.iter().map(|_| TOP).collect::<Vec<_>>());
        r["messages"][1]["behavior_top_logprobs"] = serde_json::json!(behaviour.iter().map(|&lp| [lp; 3]).collect::<Vec<_>>());
    }
    r["reward"] = serde_json::json!(reward);
    r["group"] = serde_json::json!("toy");
    if let Some(s) = sampler {
        r["sampler"] = s;
    }
    r.to_string()
}

fn write_lines(path: &Path, lines: &[String]) -> PathBuf {
    std::fs::write(path, lines.join("\n")).unwrap();
    path.to_path_buf()
}

fn base(model: &Path) -> String {
    model.join("model.safetensors").to_str().unwrap().to_string()
}

fn tiny_model(arch: &str, tag: &str) -> TempDir {
    match arch {
        "qwen3" => chat_model_dir(tag),
        _ => qwen35_model_dir(tag),
    }
}

/// The answers' token log-probabilities under `adapter`, per record (`qwen35` per token; `qwen3`'s scorer reports a
/// mean per token, taken for each of them: near enough on its near-uniform fixture to keep every ratio in the band).
fn answer_logprobs(arch: &str, model: &Path, adapter: Option<&Path>, records: &[(&str, &str)]) -> Vec<Vec<f32>> {
    records
        .iter()
        .enumerate()
        .map(|(i, (p, a))| {
            let plain = write_lines(&model.join(format!("plain-{i}.jsonl")), &[plain_record(p, a).to_string()]);
            match arch {
                "qwen3" => vec![-brain::score_chat(&base(model), adapter, &plain).expect("scoring").loss.expect("a loss"); ANSWER_TOKENS],
                _ => {
                    // The scorer reads the whole conversation; the answer is its last tokens.
                    let all: Vec<f32> = brain::score_tokens_qwen35(&base(model), adapter, &plain, Default::default(), &cpu()).expect("scoring").records[0].positions.iter().map(|p| p.logprob).collect();
                    all[all.len() - ANSWER_TOKENS..].to_vec()
                }
            }
        })
        .collect()
}

fn score_centred() -> PolicyGradientConfig {
    PolicyGradientConfig { loss: PgLoss { estimator: Estimator::ScoreCentred, ..PgLoss::RECORDED }, ..Default::default() }
}

fn run(arch: &str, model: &Path, dataset: &Path, out: &Path, config: PolicyGradientConfig) -> ChatFineTune {
    ChatFineTune::from_pretrained(base(model))
        .arch(arch)
        .dataset(dataset)
        .out_dir(out)
        .rank(2)
        .steps(10)
        // The synthetic qwen3 fixture is near-uniform with tiny activations, so its adapter needs a larger step to move.
        .lr(if arch == "qwen3" { 0.03 } else { 2e-2 })
        .grad_accum(4)
        // seed: runs that are compared with each other need the same stream.
        .seed(7)
        .device(cpu())
        .policy_gradient(config)
}

/// The rollouts as records sampled by the base itself (the behaviour is the base's own log-probabilities).
fn on_policy_records(arch: &str, model: &Path, sampler: Option<serde_json::Value>, top: bool) -> PathBuf {
    let pairs: Vec<(&str, &str)> = ROLLOUTS.iter().map(|(p, a, _)| (*p, *a)).collect();
    let behaviour = answer_logprobs(arch, model, None, &pairs);
    let lines: Vec<String> = ROLLOUTS.iter().zip(&behaviour).map(|((p, a, r), b)| record(p, a, *r, b, sampler.clone(), top)).collect();
    write_lines(&model.join("rollouts.jsonl"), &lines)
}

#[test]
fn score_centred_records_train_an_adapter_on_every_architecture() {
    for arch in ARCHS {
        let model = tiny_model(arch, "sc-train");
        let dataset = on_policy_records(arch, &model, Some(serde_json::json!({"temperature": 1.0})), true);
        let sum = |lps: Vec<Vec<f32>>, sign: f64| ROLLOUTS.iter().zip(lps).filter(|((_, _, r), _)| r * sign > 0.0).flat_map(|(_, l)| l).sum::<f32>();
        let pairs: Vec<(&str, &str)> = ROLLOUTS.iter().map(|(p, a, _)| (*p, *a)).collect();
        let before = [1.0f64, -1.0].map(|sign| sum(answer_logprobs(arch, &model, None, &pairs), sign));
        let outcome = run(arch, &model, &dataset, &model.join("run"), score_centred()).run().unwrap_or_else(|e| panic!("{arch}: the score-centred run: {e}"));
        assert_eq!(outcome.status, FineTuneStatus::Completed, "{arch}");
        let adapter = outcome.adapter.clone().expect("an adapter");
        let after = [1.0f64, -1.0].map(|sign| sum(answer_logprobs(arch, &model, Some(&adapter), &pairs), sign));
        for ((sign, b), a) in [1.0f64, -1.0].into_iter().zip(before).zip(after) {
            println!("{arch}: advantage {sign}: action log-probability {b:.4} -> {a:.4}");
            assert!(sign * f64::from(a - b) > 0.0, "{arch}: advantage {sign}: {b} -> {a}");
        }
        let pg = outcome.policy_gradient.as_ref().expect("statistics");
        assert_eq!((pg.stats.steps.len(), pg.stats.tokens_recorded_behavior), (10, 16), "{arch}");
        for s in &pg.stats.steps {
            assert!(s.loss.is_finite() && s.grad_norm.is_finite(), "{arch}: {s:?}");
            // Three listed ids at the sampled token's probability, a tiny share of a 151936-token vocabulary.
            assert!(s.sampler_mass > 0.0 && s.sampler_mass < 1e-2, "{arch}: sampler mass {}", s.sampler_mass);
            assert!((0.0..1.0).contains(&s.centred_masked_fraction), "{arch}: masked fraction {}", s.centred_masked_fraction);
        }
        assert_eq!(pg.stats.steps[0].centred_masked_fraction, 0.0, "{arch}: the first step is on-policy");
        let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("a card");
        let settings = &card.training.expect("the card records its training").hyperparams["policy_gradient"];
        assert_eq!(settings["estimator"].as_str(), Some("score-centred"), "{arch}: {settings}");
    }
}

#[test]
fn records_the_score_centred_estimator_cannot_centre_are_refused_before_training() {
    let arch = "qwen35";
    let model = tiny_model(arch, "sc-refused");
    let refused = |dataset: &Path, config: PolicyGradientConfig| run(arch, &model, dataset, &model.join("run"), config).run().expect_err("refused").to_string();

    let bare = on_policy_records(arch, &model, None, false);
    let why = refused(&bare, score_centred());
    assert!(why.contains("behavior_top_ids"), "{why}");

    let cut = on_policy_records(arch, &model, Some(serde_json::json!({"top_k": 20})), true);
    let why = refused(&cut, score_centred());
    assert!(why.contains("truncated") && why.contains("top_k"), "{why}");

    let fine = on_policy_records(arch, &model, None, true);
    let why = refused(&fine, PolicyGradientConfig { importance: brain::ImportanceCorrection::Truncate { cap: 2.0 }, ..score_centred() });
    assert!(why.contains("score-centred") && why.contains("importance"), "{why}");
}
