// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `score` action through the real executor on a resident Qwen3 with an adapter bank: the log-likelihood of given
//! answers is the model's under the adapter the model id names (as a server that folded that adapter reports it), the
//! base and the adapters answer concurrently with generation, and the action is declared exactly where the decoder
//! can score.

use std::sync::Arc;

use brain_testutil::adapters::{scratch_dir, write_byte_tokenizer, write_lora_adapter, Linear};
use capability::{Blob, Invocation, Media};
use catalog::adapter_registry::{AdapterBankSettings, AdapterTargets, LiveAdapters};
use catalog::adapter_release::verify_adapter;
use catalog::resident_llm::{QwenResident, QwenServeConfig};
use checkpoint::st::ModelCard;
use qwen3::config::QwenConfig;
use residency::budget::Budgets;
use residency::{Device, Executor, Policy, ResidentModel};
use serde_json::{json, Value};

const BASE: &str = "Qwen/Tiny";
const ONE: &str = "Qwen/Tiny:lab:one:v1";
const RANK: u32 = 4;

fn cfg() -> QwenConfig {
    QwenConfig { vocab: 128, ..QwenConfig::tiny() }
}

fn linears(cfg: &QwenConfig) -> Vec<Linear> {
    let (d, hq, hkv) = (cfg.d_model as usize, cfg.q_dim() as usize, cfg.kv_dim() as usize);
    (0..cfg.n_layers).flat_map(|l| [("wq", hq, d), ("wk", hkv, d), ("wv", hkv, d), ("wo", d, hq)].map(|(leaf, out, inn)| Linear { base: format!("blocks.{l}.attn.{leaf}.weight"), out, inn })).collect()
}

fn write_checkpoint(path: &std::path::Path, cfg: &QwenConfig, weights: &std::collections::HashMap<String, Vec<f32>>) {
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], weights[&name].clone())).collect();
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
}

fn resident(tokenizer: &str, id: &str, path: &std::path::Path, bank: Option<AdapterBankSettings>) -> Arc<QwenResident> {
    let config = QwenServeConfig { ctx: Some(1024), kv_int8: false, max_batch: 4, max_prefill_cap: 64, adapter_bank: bank, ..QwenServeConfig::default() };
    Arc::new(QwenResident::from_card_configured(path.to_str().unwrap(), &ModelCard::new(id, "qwen"), Some(tokenizer), config))
}

fn executor(models: Vec<Arc<dyn ResidentModel>>) -> Executor {
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

fn records() -> Blob {
    let lines = [
        json!({"messages": [{"role": "user", "content": "the sky is", "train": false}, {"role": "assistant", "content": "blue", "train": true}]}),
        json!({"messages": [{"role": "user", "content": "two plus two", "train": false}, {"role": "assistant", "content": "four", "train": true}, {"role": "user", "content": "and three", "train": false}, {"role": "assistant", "content": "five", "train": true}]}),
    ];
    Blob::new(Media::Text, lines.iter().map(Value::to_string).collect::<Vec<_>>().join("\n").into_bytes())
}

fn score(exec: &Executor, model: &str) -> Result<capability::Outcome, String> {
    exec.run_blocking(model, "score", Invocation::new().blob("records", records()).set("per_token", json!(true)), |_| {})
}

fn total(outcome: &capability::Outcome) -> f64 {
    outcome.outputs["logprob"].as_f64().expect("a total")
}

/// Scoring under an adapter id is what a server that folded the adapter into its weights reports, differs from the
/// base's, and every figure is a log-probability. Base and adapter ask at once, with a generation running beside them.
#[test]
fn scoring_under_an_adapter_is_what_the_folded_model_reports_and_runs_beside_generation() {
    let dir = scratch_dir("scored-adapter");
    let tokenizer = write_byte_tokenizer(&dir.join("tok")).to_string_lossy().into_owned();
    let cfg = cfg();
    let weights = qwen3::init_weights(&cfg, 7);
    let base = dir.join("base.safetensors");
    write_checkpoint(&base, &cfg, &weights);
    let adapter = dir.join("one.safetensors");
    write_lora_adapter(&adapter, ONE, BASE, RANK, &linears(&cfg), 11);
    let mut folded_weights = weights.clone();
    qwen3::lora::fold_adapter_into(&mut folded_weights, adapter.to_str().unwrap()).expect("the adapter folds");
    let folded = dir.join("folded.safetensors");
    write_checkpoint(&folded, &cfg, &folded_weights);

    let bank = AdapterBankSettings { slots: 2, rank_cap: RANK, targets: AdapterTargets::ATTENTION };
    let served = resident(&tokenizer, BASE, &base, Some(bank));
    let exec = executor(vec![served.clone()]);
    let live = LiveAdapters::new(served.clone(), exec.clone());
    live.serve(verify_adapter(&adapter, &served.served_base()).expect("verifies")).expect("offered");
    let plain = executor(vec![resident(&tokenizer, BASE, &folded, None)]);

    let want = score(&plain, BASE).expect("the folded server scores");
    let (base_score, adapter_score) = std::thread::scope(|s| {
        // A generation on the same instance while both scores are asked.
        let generating = s.spawn(|| exec.run_blocking(BASE, "generate", Invocation::new().set("prompt", json!("the sky is")).set("chat", json!(false)).set("max_new", json!(24)).set("temp", json!(0.0)).set("top_k", json!(1)), |_| {}));
        let a = s.spawn(|| score(&exec, BASE));
        let b = s.spawn(|| score(&exec, ONE));
        generating.join().unwrap().expect("the generation completes");
        (a.join().unwrap().expect("base scores"), b.join().unwrap().expect("adapter scores"))
    });

    assert!((total(&adapter_score) - total(&want)).abs() < 1e-2 * total(&want).abs().max(1.0), "adapter {} vs folded {}", total(&adapter_score), total(&want));
    assert!((total(&adapter_score) - total(&base_score)).abs() > 1e-3, "the adapter moves the score: {} vs {}", total(&adapter_score), total(&base_score));
    assert!(adapter_score.outputs["adapter"]["id"] == ONE && base_score.outputs.get("adapter").is_none(), "the reply names the policy that scored");
    let record = &adapter_score.outputs["records"][1];
    assert_eq!(record["answers"].as_array().map(Vec::len), Some(2), "two supervised answers in the second record");
    let tokens: usize = record["answers"].as_array().unwrap().iter().map(|a| a["token_logprobs"].as_array().unwrap().len()).sum();
    assert_eq!(record["tokens"].as_u64().map(|t| t as usize), Some(tokens));
    assert!(adapter_score.outputs["records"].as_array().unwrap().iter().all(|r| r["answers"].as_array().unwrap().iter().all(|a| a["token_logprobs"].as_array().unwrap().iter().all(|l| l.as_f64().unwrap() <= 0.0))), "log-probabilities");
    for e in [exec, plain] {
        e.shutdown();
    }
}

/// The action is declared by a manifest exactly when its decoder can score a continuation, which is also what
/// `policy_trace` needs: a family that cannot score does not advertise it, and so is refused by capability.
#[test]
fn a_manifest_declares_score_exactly_when_it_declares_the_policy_trace() {
    let dir = scratch_dir("scored-manifest");
    let tokenizer = write_byte_tokenizer(&dir.join("tok")).to_string_lossy().into_owned();
    let cfg = cfg();
    let base = dir.join("base.safetensors");
    write_checkpoint(&base, &cfg, &qwen3::init_weights(&cfg, 7));
    let qwen = resident(&tokenizer, BASE, &base, None).manifest();
    let declares = |m: &capability::Manifest, action: &str, param: &str| m.actions.iter().any(|a| a.name == action && (param.is_empty() || a.params.iter().any(|p| p.name == param)));
    assert!(declares(&qwen, "score", "per_token") && declares(&qwen, "generate", "policy_trace"));
    assert_eq!(qwen.actions.iter().find(|a| a.name == "score").map(|a| a.name.clone()), Some(llmserve::SCORE_ACTION.to_string()));
    for (name, manifest) in [("edgerock", catalog::resident_edgerock::manifest()), ("huginn", catalog::resident_huginn::manifest()), ("ouro", catalog::resident_ouro::manifest())] {
        assert!(!declares(&manifest, "score", ""), "{name}: its decoder has no teacher-forced scoring pass, so it does not advertise score");
    }
}
