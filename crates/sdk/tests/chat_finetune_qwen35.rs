// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improvement loops that train a large
// hybrid-attention model on its own verified work and serve the result for its
// clients. If your team needs expertise in LoRA fine-tuning and serving of
// Qwen3.8-class models then you can procure our services by sending an email
// to info@swedishembedded.com.

// Fine-tuning is the `study` surface.
#![cfg(feature = "study")]

//! `brain::ChatFineTune` on a `qwen35` base (the 27B's architecture, at toy
//! width, CPU backend): the chat dataset trains a LoRA over every targetable
//! projection, the adapter file carries the card a server binds it by, and a
//! cancelled run resumes to the uninterrupted adapter.

use std::path::{Path, PathBuf};

use brain::{ChatFineTune, ChatFullFineTune, FineTuneStatus, PreferenceFineTune};

#[path = "chat_fixture/common.rs"]
mod common;
use common::{cpu, qwen35_model_dir as qwen35_chat_model_dir};

fn write_dataset(dir: &Path) -> PathBuf {
    let path = dir.join("train.jsonl");
    let rows: Vec<String> = [("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")]
        .iter()
        .map(|(u, a)| {
            serde_json::json!({
                "messages": [{"role": "user", "content": u, "train": false}, {"role": "assistant", "content": a, "train": true}],
                "tools": []
            })
            .to_string()
        })
        .collect();
    std::fs::write(&path, rows.join("\n")).unwrap();
    path
}

// seed: runs that are compared with each other need the same stream.
fn fine_tune(model: &Path, out: &Path) -> ChatFineTune {
    ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("qwen35")
        .dataset(write_dataset(model))
        .out_dir(out)
        .rank(2)
        .steps(6)
        .lr(1e-2)
        .seed(7)
        .device(cpu())
}

#[test]
fn a_qwen35_fine_tune_exports_a_servable_adapter_with_its_digests() {
    let model = qwen35_chat_model_dir("q35-export");
    let outcome = fine_tune(&model, &model.join("run")).adapter_id("local/tiny:lab:facts:v1").run().expect("the fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    // No native tensor-core device here (CPU): the automatic choice is the fp32 reference, and it is recorded.
    assert_eq!(outcome.base_tier, "f32");
    assert!(outcome.base_tier_reason.contains("native"), "{}", outcome.base_tier_reason);
    assert_eq!(outcome.steps_completed, 6);
    let (first, last) = (outcome.initial_loss.unwrap(), outcome.final_loss.unwrap());
    assert!(first.is_finite() && last.is_finite() && last < first, "the loss falls on its own training data: {first} -> {last}");

    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("the adapter carries a card");
    assert_eq!(card.id, "local/tiny:lab:facts:v1");
    let spec = card.adapter.expect("an adapter card");
    assert_eq!((spec.kind.as_str(), spec.rank), ("lora", Some(2)));
    // All twelve targetable projections of the 27B architecture are adapted.
    assert_eq!(spec.targets.as_ref().map(Vec::len), Some(12), "{:?}", spec.targets);
    let training = card.training.expect("the card records its training run");
    assert_eq!((training.regime.as_str(), training.seed), ("sft_lora", 7));
    let base_digest = brain_modelstore::fetch::file_digest(&model.join("model.safetensors")).unwrap();
    assert_eq!(training.base_digest.as_deref(), Some(base_digest.as_str()), "the base is named by content");
    assert_eq!(outcome.adapter_digest.as_deref(), Some(brain_modelstore::fetch::file_digest(&adapter).unwrap().as_str()));
    assert!(outcome.record.as_ref().is_some_and(|p| p.exists()), "the training record is written");
}

/// A Hugging Face checkpoint of the 27B names its architecture in its
/// config; the caller leaves it out and the qwen35 trainer is the one that runs.
#[test]
fn a_base_whose_config_names_qwen35_trains_without_the_architecture_being_named() {
    let model = qwen35_chat_model_dir("q35-detect");
    std::fs::write(model.join("config.json"), r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#).unwrap();
    let run = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .adapter_id("local/tiny:lab:facts:v1")
        .rank(2)
        .steps(2)
        .lr(1e-2)
        .seed(7)
        .device(cpu());
    let outcome = run.run().expect("the fine-tune runs on the qwen35 trainer");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let card = checkpoint::st::read_card(outcome.adapter.unwrap().to_str().unwrap()).unwrap().unwrap();
    assert_eq!(card.family, "qwen35");
}

#[test]
fn a_cancelled_qwen35_run_resumes_to_the_uninterrupted_adapter() {
    let model = qwen35_chat_model_dir("q35-resume");
    let straight = fine_tune(&model, &model.join("straight")).run().unwrap();

    let out = model.join("interrupted");
    let cancel = brain::CancelToken::armed();
    let stopped = fine_tune(&model, &out)
        .run_with(&cancel, |p| {
            if p.step == 3 {
                cancel.cancel();
            }
        })
        .unwrap();
    assert_eq!((stopped.status, stopped.steps_completed), (FineTuneStatus::Cancelled, 3));
    assert!(stopped.resume_state.as_ref().is_some_and(|p| p.exists()));

    let resumed = fine_tune(&model, &out).run().unwrap();
    assert_eq!((resumed.status, resumed.resumed_at, resumed.steps_completed), (FineTuneStatus::Completed, Some(3), 6));
    assert_eq!(resumed.adapter_digest, straight.adapter_digest, "the same adapter file, byte for byte");
}

/// The monitoring pass, the held-out score and the per-evaluation adapters a 27B self-distillation run
/// selects on: scored on the training process's own model (a second copy of the base does not fit
/// beside it), over the positions training supervises.
#[test]
fn a_qwen35_fine_tune_scores_held_out_records_and_keeps_every_evaluation() {
    let model = qwen35_chat_model_dir("q35-heldout");
    let records = write_dataset(&model);
    let evaluations = model.join("evaluations");
    let outcome = fine_tune(&model, &model.join("run"))
        .held_out(&records)
        .held_out_text(&records)
        .monitor(&records)
        .eval_every(2)
        .keep_evaluations(&evaluations)
        .run()
        .expect("the fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let (base, tuned) = (outcome.base_score.expect("the base is scored on the held-out set"), outcome.tuned_score.expect("the tuned model is too"));
    let (base_loss, tuned_loss) = (base.loss.unwrap(), tuned.loss.unwrap());
    assert!(base_loss.is_finite() && tuned_loss < base_loss, "records it trained on score better after training: {base_loss} -> {tuned_loss}");
    assert_eq!((base.records, tuned.records), (4, 4));
    assert!(base.positions > 0 && base.positions == tuned.positions, "the same supervised positions: {base:?} {tuned:?}");
    assert!(outcome.base_text_score.is_some() && outcome.tuned_text_score.is_some(), "the second held-out set is scored apart");
    // The untouched base on the monitoring set is what the loop measured before step 0.
    assert!((outcome.initial_loss.unwrap() - base_loss).abs() < 1e-3 * base_loss.abs().max(1.0), "{:?} vs {base_loss}", outcome.initial_loss);

    for step in [2, 4, 6] {
        let file = evaluations.join(format!("step-{step}.safetensors"));
        let card = checkpoint::st::read_card(file.to_str().unwrap()).unwrap().unwrap_or_else(|| panic!("{} has no card", file.display()));
        assert_eq!(card.family, "qwen35", "step {step}");
    }
    assert_eq!(outcome.curve.iter().map(|p| p.step).collect::<Vec<_>>(), vec![2, 4, 6]);
}

#[test]
fn what_qwen35_cannot_hold_is_refused_before_anything_is_read() {
    let model = qwen35_chat_model_dir("q35-refuse");
    let err = fine_tune(&model, &model.join("run")).int8_base(true).run().unwrap_err().to_string();
    assert!(err.contains("int8"), "{err}");
}

/// Continuing an adapter trains ITS factors at its own shape and records it
/// as the new adapter's parent, on the outcome and on the card: the chain a
/// multi-generation campaign (adapter on adapter) is audited by.
#[test]
fn continuing_a_qwen35_adapter_records_its_parent() {
    let model = qwen35_chat_model_dir("q35-continue");
    let first = fine_tune(&model, &model.join("first")).run().unwrap();
    let parent = first.adapter.clone().unwrap();

    let second = fine_tune(&model, &model.join("second")).continue_from(&parent).run().unwrap();
    assert_eq!(second.status, FineTuneStatus::Completed);
    assert_eq!((second.rank, second.alpha), (first.rank, first.alpha));
    assert_eq!(second.trained_from, first.adapter_digest);
    assert_ne!(second.adapter_digest, first.adapter_digest, "it trained on top of the parent");
    let card = checkpoint::st::read_card(second.adapter.as_ref().unwrap().to_str().unwrap()).unwrap().unwrap();
    assert_eq!(card.training.unwrap().trained_from, first.adapter_digest);
}

/// The full-parameter and preference fine-tuners are written once over the
/// model seam: a second architecture runs them with nothing but its trainer.
#[test]
fn a_qwen35_full_fine_tune_trains_every_weight_and_writes_a_base() {
    let model = qwen35_chat_model_dir("q35-full");
    let out = model.join("run");
    let outcome = ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("qwen35")
        .dataset(write_dataset(&model))
        .out_dir(&out)
        .steps(6)
        .lr(1e-2)
        .seed(7)
        .device(cpu())
        .run()
        .expect("the full fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!(outcome.steps_completed, 6);
    assert!(outcome.final_loss.zip(outcome.initial_loss).is_some_and(|(last, first)| last < first), "{outcome:?}");
    let checkpoint = outcome.checkpoint.expect("a completed run writes its checkpoint directory");
    assert!(checkpoint.join("tokenizer.json").is_file());
    assert!(!out.join("tuned.brain.safetensors").exists(), "the training format is not left behind");
    // The written weights are a qwen35 base: they open, and every tensor of the
    // architecture is there.
    let base = qwen35::base_source::open_base(&checkpoint.join("model.safetensors")).expect("the checkpoint opens as a qwen35 base");
    assert_eq!(base.cfg.param_list().len(), qwen35::config::Qwen35Config::tiny().param_list().len());
}

#[test]
fn a_qwen35_preference_fine_tune_exports_a_dpo_adapter() {
    let model = qwen35_chat_model_dir("q35-dpo");
    let pairs = model.join("pairs.jsonl");
    let lines: Vec<String> = [("cab", "bad", "wig"), ("fed", "deaf", "vow"), ("ace", "face", "jug"), ("bag", "gab", "mop")]
        .iter()
        .map(|(u, c, r)| {
            serde_json::json!({
                "prompt": [{"role": "user", "content": u}],
                "chosen": {"role": "assistant", "content": c},
                "rejected": {"role": "assistant", "content": r},
                "metadata": {"source": "test"}
            })
            .to_string()
        })
        .collect();
    std::fs::write(&pairs, lines.join("\n")).unwrap();
    let outcome = PreferenceFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("qwen35")
        .dataset(&pairs)
        .out_dir(model.join("run"))
        .rank(2)
        .steps(6)
        .lr(1e-1)
        .seed(7)
        .device(cpu())
        .run()
        .expect("the preference fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let initial = outcome.initial_loss.expect("measured");
    assert!((initial - std::f32::consts::LN_2).abs() < 1e-3, "the margin starts at zero against the starting model: {initial}");
    assert!(outcome.final_loss.is_some_and(|last| last < initial), "{outcome:?}");
    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("the adapter carries a card");
    assert_eq!(card.training.expect("its training run is recorded").regime, "dpo");
    let score = outcome.train_score.expect("a run on the resident model scores its training pairs");
    assert_eq!((score.pairs, score.skipped), (4, 0));
    assert!(score.mean_margin.is_some_and(|m| m > 0.0), "the trained policy prefers chosen over rejected more than its start did: {score:?}");
}

/// Every step reports the seconds its four phases took and the tokens it
/// accounted for, its record carries them, and the record's totals are the
/// sum of them: what an optimization of the step is measured against, without
/// adding a profiler to the run.
#[test]
fn a_run_records_every_step_s_phases_and_token_accounting() {
    let model = qwen35_chat_model_dir("q35-work");
    let outcome = fine_tune(&model, &model.join("run")).run().expect("the fine-tune runs");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!(outcome.step_work.len(), outcome.steps_completed as usize, "one entry per completed step");

    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.as_ref().expect("the record is written")).unwrap()).unwrap();
    // Additive: the fields the record always carried are still there.
    assert_eq!(record["block"], serde_json::json!(outcome.block));
    assert_eq!(record["steps_completed"], serde_json::json!(outcome.steps_completed));
    let per_step = record["step_work"].as_array().expect("the record carries every step's work");
    assert_eq!(per_step.len(), outcome.steps_completed as usize);
    for (i, s) in per_step.iter().enumerate() {
        assert_eq!(s["step"], serde_json::json!(i as u32 + 1), "the steps of a straight run are consecutive from 1");
        let (useful, computed, supervised) = (s["useful_tokens"].as_u64().expect("useful tokens"), s["computed_tokens"].as_u64().expect("computed tokens"), s["supervised_positions"].as_u64().expect("supervised positions"));
        assert!(useful > 0 && supervised > 0 && supervised <= useful, "step {}: {s}", i + 1);
        // A one-record micro-step is computed at its own length rounded up to a
        // whole GDN chunk (64), not at the block the run was built for.
        assert!(computed <= useful + 64 && computed + 64 >= useful, "step {}: computed {computed} tokens for {useful} useful ones", i + 1);
        assert_eq!(s["padded_tokens"].as_u64().unwrap(), computed.saturating_sub(useful), "padding is what was computed and not useful");
        let step_secs = s["step_s"].as_f64().expect("step seconds");
        assert!(step_secs > 0.0, "step {}: a step takes time", i + 1);
        let phases: f64 = ["prepare_s", "forward_s", "backward_s", "optimizer_s"].iter().map(|k| s[k].as_f64().unwrap_or(f64::NAN)).sum();
        assert!(phases.is_finite() && (phases - step_secs).abs() < 1e-6, "step {}: the phases partition the step", i + 1);
    }

    // The totals are the sum of the steps, and the rates follow from them.
    let total = &record["work"];
    let sum_f = |k: &str| per_step.iter().map(|s| s[k].as_f64().unwrap()).sum::<f64>();
    let sum_u = |k: &str| per_step.iter().map(|s| s[k].as_u64().unwrap()).sum::<u64>();
    for k in ["prepare_s", "forward_s", "backward_s", "optimizer_s", "step_s"] {
        assert!((total[k].as_f64().unwrap() - sum_f(k)).abs() < 1e-6, "{k}: {} vs {}", total[k], sum_f(k));
    }
    for k in ["useful_tokens", "computed_tokens", "padded_tokens", "supervised_positions"] {
        assert_eq!(total[k].as_u64().unwrap(), sum_u(k), "{k}: totals are the sum of the steps");
    }
    assert_eq!(total["steps"].as_u64(), Some(u64::from(outcome.steps_completed)));
    let (useful, computed, step_secs) = (total["useful_tokens"].as_u64().unwrap() as f64, total["computed_tokens"].as_u64().unwrap() as f64, total["step_s"].as_f64().unwrap());
    assert!((total["useful_tokens_per_s"].as_f64().unwrap() - useful / step_secs).abs() < 1e-6);
    assert!((total["computed_tokens_per_s"].as_f64().unwrap() - computed / step_secs).abs() < 1e-6);
}
