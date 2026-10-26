// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Fine-tuning is the `study` surface; serving the result is `text`.
#![cfg(all(feature = "study", feature = "text"))]

//! `brain::ChatFineTune` end to end on a synthetic chat model, on the CPU
//! backend: train, export, score, serve the adapter; continue an adapter;
//! cancel and resume exactly.
//!
//! The fixture model is `chat_fixture`'s synthetic chat checkpoint.

use std::path::{Path, PathBuf};

use brain::{ChatFineTune, FineTuneStatus};

mod chat_fixture;
use chat_fixture::{chat_model_dir, chat_model_dir_of, cpu};

/// One JSONL chat record: a user turn and a supervised assistant answer.
fn record(user: &str, assistant: &str) -> String {
    serde_json::json!({
        "messages": [
            {"role": "user", "content": user, "train": false},
            {"role": "assistant", "content": assistant, "train": true}
        ],
        "tools": []
    })
    .to_string()
}

fn write_dataset(dir: &Path, name: &str, records: &[(&str, &str)]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, records.iter().map(|(u, a)| record(u, a)).collect::<Vec<_>>().join("\n")).unwrap();
    path
}

/// A fine-tune of the fixture model with no seed of its own: its weights, a
/// small dataset and a held-out set, on the CPU.
fn unseeded_fine_tune(model: &Path, out: &Path) -> ChatFineTune {
    let train = write_dataset(model, "train.jsonl", &[("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")]);
    let held_out = write_dataset(model, "held_out.jsonl", &[("cad", "bead")]);
    ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(train).held_out(held_out).out_dir(out).rank(2).steps(6).lr(1e-2).device(cpu())
}

/// [`unseeded_fine_tune`] at a fixed seed.
fn fine_tune(model: &Path, out: &Path) -> ChatFineTune {
    // seed: runs that must be compared with each other need the same stream.
    unseeded_fine_tune(model, out).seed(7)
}

/// Train, export the adapter and its record, score base and tuned on the
/// held-out set, and serve the adapter: the served identity names the
/// exported adapter by the digest the outcome reports.
#[test]
fn a_fine_tune_exports_an_adapter_that_scores_and_serves() {
    let model = chat_model_dir("export");
    let out = model.join("run");
    let mut steps_seen = Vec::new();
    let outcome = fine_tune(&model, &out).run_with(&brain::CancelToken::armed(), |p| steps_seen.push(p.step)).expect("the fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!(steps_seen, (1..=6).collect::<Vec<_>>(), "one progress report per step");
    assert_eq!(outcome.steps_completed, 6);
    assert!(outcome.initial_loss.is_some_and(f32::is_finite) && outcome.final_loss.is_some_and(f32::is_finite), "{outcome:?}");
    assert_eq!(outcome.resumed_at, None);
    assert_eq!(outcome.trained_from, None, "a fresh adapter has no parent");
    assert!(outcome.resume_state.is_none());

    let adapter = outcome.adapter.clone().expect("a completed run exports its adapter");
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("the adapter carries a card");
    let training = card.training.expect("the card records its training run");
    assert_eq!(training.regime, "sft_lora");
    assert_eq!(training.seed, 7);
    assert_eq!(training.trained_from, None);
    assert_eq!(training.environment, "cpu", "these tests run on the CPU backend");
    // The base is named by content, so a server can refuse to fold this
    // adapter into any other base.
    let base_digest = brain_modelstore::fetch::file_digest(&model.join("model.safetensors")).unwrap();
    assert_eq!(training.base_digest.as_deref(), Some(base_digest.as_str()));
    // The outcome reports the same digest, so a caller need not re-hash the base.
    assert_eq!(outcome.base_digest, training.base_digest);
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.as_ref().expect("a training record")).unwrap()).unwrap();
    assert_eq!(record["adapter_digest"].as_str(), outcome.adapter_digest.as_deref());

    let (base, tuned) = (outcome.base_score.expect("scored before"), outcome.tuned_score.expect("scored after"));
    assert!(base.loss.is_some_and(f32::is_finite) && tuned.loss.is_some_and(f32::is_finite), "{base:?} {tuned:?}");
    assert!(base.positions > 0 && base.positions == tuned.positions, "{base:?} {tuned:?}");
    let scored = brain::score_chat(model.join("model.safetensors").to_str().unwrap(), Some(&adapter), &model.join("held_out.jsonl")).unwrap();
    assert_eq!(scored.loss, tuned.loss, "the public scorer is the one the run used");

    let chat = brain::ChatPipeline::from(
        brain::TextGenerationPipeline::builder(model.join("model.safetensors").to_str().unwrap())
            .tokenizer(model.join("tokenizer.json").to_str().unwrap())
            .adapter(adapter.to_str().unwrap())
            .capacity(256)
            .device(cpu())
            .load()
            .expect("the exported adapter serves"),
    );
    assert_eq!(chat.identity().adapter.as_ref().map(|a| a.digest.as_str()), outcome.adapter_digest.as_deref());
}

/// One `base_format` names the frozen base of every architecture: the run
/// trains over it, reports it, and a format the trainer cannot hold is refused
/// by name before anything is read.
#[test]
fn a_fine_tune_trains_over_the_base_format_it_is_given() {
    let model = chat_model_dir_of("formats", qwen3::QwenConfig::tiny_i8());
    for format in ["bf16", "nf4", "fp8"] {
        let out = model.join(format!("run-{format}"));
        let outcome = fine_tune(&model, &out).base_format(format).run().unwrap_or_else(|e| panic!("{format}: {e}"));
        assert_eq!(outcome.status, FineTuneStatus::Completed, "{format}");
        assert_eq!(outcome.base_tier, format, "the outcome reports the format the base was held in");
        assert!(outcome.initial_loss.is_some_and(f32::is_finite) && outcome.final_loss.is_some_and(f32::is_finite), "{format}: {outcome:?}");
    }
    let err = fine_tune(&model, &model.join("run-refused")).base_format("fp3").run().unwrap_err().to_string();
    assert!(err.contains("fp3") && err.contains("qwen3"), "{err}");
}

/// Continuing an adapter trains ITS factors, at its own rank, and records
/// it as the new adapter's parent - on the outcome and on the card.
#[test]
fn continuing_an_adapter_records_its_parent() {
    let model = chat_model_dir("continue");
    let first = fine_tune(&model, &model.join("first")).run().unwrap();
    let parent = first.adapter.clone().unwrap();

    let err = fine_tune(&model, &model.join("second")).rank(5).continue_from(&parent).run().unwrap_err();
    assert!(err.to_string().contains("rank"), "an adapter is continued at its own rank: {err}");

    let second = fine_tune(&model, &model.join("second")).continue_from(&parent).run().unwrap();
    assert_eq!(second.status, FineTuneStatus::Completed);
    assert_eq!((second.rank, second.alpha), (first.rank, first.alpha), "the continued adapter's shape");
    assert_eq!(second.trained_from, first.adapter_digest);
    let card = checkpoint::st::read_card(second.adapter.as_ref().unwrap().to_str().unwrap()).unwrap().unwrap();
    assert_eq!(card.training.unwrap().trained_from, first.adapter_digest);
    assert_ne!(second.adapter_digest, first.adapter_digest);
}

/// A cancelled run stops at the next step boundary, exports nothing and
/// leaves its state; a state from different options is refused; the same
/// fine-tune run again resumes it and ends at the bit-identical adapter an
/// uninterrupted run produces.
#[test]
fn a_cancelled_run_resumes_to_the_uninterrupted_result() {
    let model = chat_model_dir("resume");
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
    assert_eq!(stopped.status, FineTuneStatus::Cancelled);
    assert_eq!(stopped.steps_completed, 3);
    assert!(stopped.adapter.is_none() && stopped.tuned_score.is_none(), "{stopped:?}");
    assert!(stopped.resume_state.as_ref().is_some_and(|p| p.exists()));

    let err = fine_tune(&model, &out).lr(5e-3).run().unwrap_err();
    assert!(err.to_string().contains("different training options"), "{err}");

    let resumed = fine_tune(&model, &out).run().unwrap();
    assert_eq!(resumed.status, FineTuneStatus::Completed);
    assert_eq!(resumed.resumed_at, Some(3));
    assert_eq!(resumed.steps_completed, 6);
    assert!(!stopped.resume_state.unwrap().exists(), "an exported run leaves no state behind");

    let tensors = |p: &Option<PathBuf>| checkpoint::st::load_safetensors(p.as_ref().unwrap().to_str().unwrap()).unwrap().tensors;
    let (a, b) = (tensors(&straight.adapter), tensors(&resumed.adapter));
    assert_eq!(a.len(), b.len());
    for (name, values) in &a {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(values), bits(&b[name]), "{name} differs between the uninterrupted and the resumed run");
    }
    assert_eq!(resumed.adapter_digest, straight.adapter_digest, "the same adapter file, byte for byte");
}

/// A fine-tune given no seed draws one and records it - on the outcome, in
/// the training record and on the adapter's card - and a run given that seed
/// writes the same adapter. A cancelled unseeded run resumes when run again
/// unseeded: the state holds the seed it was drawn with.
#[test]
fn an_unseeded_fine_tune_records_the_seed_it_drew() {
    let model = chat_model_dir("drawn_seed");
    let first = unseeded_fine_tune(&model, &model.join("first")).run().unwrap();
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(first.record.as_ref().unwrap()).unwrap()).unwrap();
    assert_eq!(record["seed"].as_u64(), Some(first.seed));
    let card = checkpoint::st::read_card(first.adapter.as_ref().unwrap().to_str().unwrap()).unwrap().unwrap();
    assert_eq!(card.training.unwrap().seed, first.seed);

    let again = unseeded_fine_tune(&model, &model.join("again")).seed(first.seed).run().unwrap();
    assert_eq!(again.adapter_digest, first.adapter_digest, "the recorded seed reproduces the adapter");
    let other = unseeded_fine_tune(&model, &model.join("other")).run().unwrap();
    assert_ne!(other.seed, first.seed, "each unseeded run draws its own seed");

    let out = model.join("interrupted");
    let cancel = brain::CancelToken::armed();
    let stopped = unseeded_fine_tune(&model, &out)
        .run_with(&cancel, |p| {
            if p.step == 3 {
                cancel.cancel();
            }
        })
        .unwrap();
    assert_eq!(stopped.status, FineTuneStatus::Cancelled);
    let resumed = unseeded_fine_tune(&model, &out).run().unwrap();
    assert_eq!((resumed.status, resumed.resumed_at, resumed.seed), (FineTuneStatus::Completed, Some(3), stopped.seed));
}

/// A fine-tune that evaluates a monitoring set as it trains records the
/// curve, exports the adapter of the evaluation with the lowest monitoring
/// loss rather than the last step's, and says so in the outcome, the
/// training record and the adapter's card.
#[test]
fn a_selecting_fine_tune_exports_the_best_evaluation_and_records_the_curve() {
    let model = chat_model_dir("select");
    let monitor = write_dataset(&model, "monitor.jsonl", &[("dab", "bad"), ("fade", "deaf")]);
    let out = model.join("run");
    let outcome = fine_tune(&model, &out).monitor(&monitor).eval_every(2).keep_best(true).run().expect("the fine-tune runs");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert_eq!(outcome.curve.iter().map(|p| p.step).collect::<Vec<_>>(), [2, 4, 6], "an evaluation every two steps, the last step included");
    assert!(outcome.curve.iter().all(|p| p.train_loss.is_finite() && p.monitor_loss.is_finite()), "{:?}", outcome.curve);
    assert_eq!(outcome.monitor_records, 2);
    let best = outcome.curve.iter().min_by(|a, b| a.monitor_loss.total_cmp(&b.monitor_loss)).unwrap();
    assert_eq!((outcome.selected_step, outcome.selection), (best.step, brain::Selection::BestMonitorLoss));
    assert!(!outcome.stopped_early, "keeping the best alone never stops a run early");

    // The exported adapter is the one the best monitoring loss was measured
    // on: scored on the same records it gives that loss back.
    let adapter = outcome.adapter.clone().unwrap();
    let scored = brain::score_chat(model.join("model.safetensors").to_str().unwrap(), Some(&adapter), &monitor).unwrap();
    assert!((scored.loss.unwrap() - best.monitor_loss).abs() < 0.05, "scored {:?} but the best evaluation was {}", scored.loss, best.monitor_loss);

    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(outcome.record.as_ref().unwrap()).unwrap()).unwrap();
    assert_eq!(record["selected_step"].as_u64(), Some(u64::from(best.step)));
    assert_eq!(record["selection"].as_str(), Some("best_monitor_loss"));
    assert_eq!(record["curve"].as_array().map(Vec::len), Some(3));
    assert_eq!((record["eval_every"].as_u64(), record["patience"].as_u64(), record["monitor_records"].as_u64()), (Some(2), Some(0), Some(2)));
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().unwrap();
    let hyper = card.training.unwrap().hyperparams;
    assert_eq!((hyper["selected_step"].as_u64(), hyper["selection"].as_str()), (Some(u64::from(best.step)), Some("best_monitor_loss")));

    // A run that does not select exports its last step and says so.
    let plain = fine_tune(&model, &model.join("plain")).run().unwrap();
    assert_eq!((plain.selected_step, plain.selection, plain.curve.len()), (6, brain::Selection::LastStep, 0));
}

/// Exact resume covers selection: a selecting run cancelled mid-way and run
/// again exports the same adapter the uninterrupted one does, with the
/// whole run's curve.
#[test]
fn a_cancelled_selecting_run_resumes_to_the_uninterrupted_result() {
    let model = chat_model_dir("select-resume");
    let monitor = write_dataset(&model, "monitor.jsonl", &[("dab", "bad"), ("fade", "deaf")]);
    let selecting = |out: &Path| fine_tune(&model, out).monitor(&monitor).eval_every(1).patience(3);
    let straight = selecting(&model.join("straight")).run().unwrap();

    let out = model.join("interrupted");
    let cancel = brain::CancelToken::armed();
    let stopped = selecting(&out)
        .run_with(&cancel, |p| {
            if p.step == 3 {
                cancel.cancel();
            }
        })
        .unwrap();
    assert_eq!(stopped.status, FineTuneStatus::Cancelled);
    let resumed = selecting(&out).run().unwrap();
    assert_eq!(resumed.resumed_at, Some(3));
    assert_eq!(resumed.curve, straight.curve, "the curve is the whole run's");
    assert_eq!((resumed.selected_step, resumed.selection), (straight.selected_step, straight.selection));
    assert_eq!(resumed.adapter_digest, straight.adapter_digest, "the same adapter file, byte for byte");
}

/// A base that is a `transformers` directory (the way every DeepSeek
/// checkpoint is downloaded) fine-tunes as a brain checkpoint does: it is read
/// as it is, scored before and after, named by a digest of its files, and the
/// adapter is exported.
#[test]
fn a_transformers_directory_base_fine_tunes_and_scores() {
    let model = chat_model_dir("hfdir");
    let store = model.join("models");
    let repo = store.join("acme").join("toy");
    let cfg = qwen3::QwenConfig { vocab: 151_936, block_size: 128, max_position_embeddings: 128, ..qwen3::QwenConfig::tiny() };
    let reader = checkpoint::weightio::WeightReader::open(model.join("model.safetensors").to_str().unwrap()).unwrap();
    qwen3::export::export_hf(&reader, &cfg, &repo, &qwen3::export::HfExport { dtype: qwen3::export::HfDtype::F32, tokenizer_dir: Some(&model), ..Default::default() }).unwrap();
    std::fs::copy(model.join("tokenizer_config.json"), repo.join("tokenizer_config.json")).unwrap();

    let train = write_dataset(&model, "train.jsonl", &[("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")]);
    let held_out = write_dataset(&model, "held_out.jsonl", &[("cad", "bead")]);
    let out = model.join("run");
    let outcome = ChatFineTune::from_pretrained(repo.to_str().unwrap()).dataset(train).held_out(held_out).out_dir(&out).rank(2).steps(4).lr(1e-2).seed(data::rng::random_seed()).device(cpu()).run().expect("a transformers directory base fine-tunes");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert!(outcome.base_score.is_some() && outcome.tuned_score.is_some(), "scored before and after");
    let digest = outcome.base_digest.expect("the base is named");
    assert!(digest.starts_with("sha256:"), "{digest}");
    assert!(outcome.adapter.is_some_and(|a| a.is_file()));
}

/// `score_tokens` is the always-correct oracle of the trainer's per-token
/// log-probabilities: it equals `batch_token_logprobs` of a forward on the
/// same checkpoint, at the unit logit scale and at a halved one (a sampler at
/// temperature 2), masked positions score zero, and an adapter changes the
/// scores while no adapter is the frozen base.
#[test]
fn score_tokens_matches_the_trainers_logprobs_and_applies_the_logit_scale() {
    let model = chat_model_dir("score-tokens");
    let weights = model.join("model.safetensors");
    let weights_str = weights.to_str().unwrap();
    let tokens: Vec<u32> = vec![3, 7, 1, 12, 5, 9];
    let targets: Vec<u32> = vec![7, 1, brain::IGNORE_TARGET, 5, 9, 2];

    for scale in [1.0f32, 0.5] {
        let got = brain::score_tokens(weights_str, None, &tokens, &targets, scale, &cpu()).expect("scored");
        assert_eq!(got[2], 0.0, "a masked position scores zero");

        let mut trainer = qwen3::Qwen::load(weights_str, 1, tokens.len() as u32);
        trainer.set_logit_scale(scale);
        <qwen3::Qwen as model::Model>::set_batch(&trainer, model::Batch::Lm { tokens: &tokens, targets: &targets });
        <qwen3::Qwen as model::Model>::forward(&trainer);
        let want = trainer.batch_token_logprobs();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-3, "position {i} at scale {scale}: score_tokens {g} vs trainer {w}");
        }
    }

    let out = model.join("run");
    let outcome = fine_tune(&model, &out).run().expect("the fine-tune runs");
    let adapter = outcome.adapter.expect("an exported adapter");
    let base = brain::score_tokens(weights_str, None, &tokens, &targets, 1.0, &cpu()).unwrap();
    let tuned = brain::score_tokens(weights_str, Some(&adapter), &tokens, &targets, 1.0, &cpu()).unwrap();
    let moved = base.iter().zip(&tuned).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(moved > 1e-4, "the adapter must change the scores, moved only {moved}");

    let refused = brain::score_tokens(weights_str, None, &tokens, &targets[..3], 1.0, &cpu()).unwrap_err().to_string();
    assert!(refused.contains("as many targets as tokens"), "{refused}");
}
