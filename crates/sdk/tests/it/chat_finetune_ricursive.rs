// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning very large dense
// decoders on the memory a team actually has, from full-weight training to
// adapters over a 4-bit base. If your team needs expertise in training
// hybrid-attention language models then you can procure our services by
// sending an email to info@swedishembedded.com.

// Fine-tuning is the `study` surface.
#![cfg(feature = "study")]

//! The generic chat fine-tuners on a Ripple base (toy width, CPU backend): the supervised LoRA fine-tune
//! (`ChatFineTune`), the full-parameter fine-tune and the preference fine-tune all run the same `model::finetune`
//! entry points the other architectures do.
//!

use std::path::{Path, PathBuf};

use brain::{ChatFineTune, ChatFullFineTune, FineTuneStatus, PreferenceFineTune};

#[path = "../chat_fixture/common.rs"]
mod common;
use common::{cpu, TempDir};

const TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}{{ message.content }}<|im_end|>{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}";

/// The fixture tokenizer with its special tokens just above the letters instead of at the 151k mark, so the toy model's
/// vocabulary is 32 rather than 151,936 (the CPU backend's backward through a head that wide costs seconds a step).
fn write_small_tokenizer(dir: &Path) {
    let byte_encoder = data::bpe::bytes_to_unicode();
    let vocab: serde_json::Map<String, serde_json::Value> = (b'a'..=b'w').enumerate().map(|(id, b)| (byte_encoder[b as usize].to_string(), serde_json::json!(id as u32))).collect();
    let added = serde_json::json!([{"content": "<|endoftext|>", "id": 28}, {"content": "<|im_start|>", "id": 29}, {"content": "<|im_end|>", "id": 30}]);
    std::fs::write(dir.join("tokenizer.json"), serde_json::json!({ "model": { "vocab": vocab, "merges": [] }, "added_tokens": added }).to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({ "chat_template": TEMPLATE }).to_string()).unwrap();
}

/// A tiny Ripple checkpoint (brain format) over the small vocabulary.
fn ripple_model_dir(tag: &str) -> TempDir {
    let dir = TempDir::new(tag);
    let mut cfg = ricursive::TrainConfig::tiny();
    cfg.block_size = 128;
    let init = cfg.tiny_weights(11); // seed: any weights serve, the test reads the training, not the model
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, numel)| (name.clone(), vec![numel as u64], init[&name].clone())).collect();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
    write_small_tokenizer(&dir);
    dir
}

fn write_dataset(dir: &Path) -> PathBuf {
    common::write_chat_dataset(dir, &[("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")])
}

/// Spec: a supervised LoRA fine-tune trains an adapter over the frozen base, exports it with its card, and the
/// adapter holds adapter tensors only, at the rank asked for.
#[test]
fn a_ricursive_lora_fine_tune_exports_an_adapter_over_the_frozen_base() {
    let model = ripple_model_dir("ripple-sft");
    let outcome = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .rank(2)
        .steps(6)
        .lr(5e-2)
        .seed(7) // seed: runs compared with each other need the same stream
        .device(cpu())
        .run()
        .expect("the LoRA fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert!(outcome.final_loss.zip(outcome.initial_loss).is_some_and(|(last, first)| last < first), "{outcome:?}");
    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("the adapter carries a card");
    assert_eq!(card.family, "ricursive");
    assert_eq!(card.training.expect("the card records its run").regime, "sft_lora");
    let st = checkpoint::st::load_safetensors(adapter.to_str().unwrap()).expect("the adapter opens");
    assert!(!st.tensors.is_empty() && st.tensors.keys().all(|n| model::adapter::device::is_adapter_param(n)), "an adapter file holds adapter tensors only");
    let read = model::lora::device_adapter::read_adapter(adapter.to_str().unwrap()).expect("the adapter validates");
    assert_eq!(read.rank, 2);
    assert_eq!(read.sites.len(), 3 * 8, "q, k, v, o, the gate and the feed-forward trio of each of the three layers");
}

/// Spec: a variant of the adapter this architecture does not train is refused by name, not trained as the plain one.
#[test]
fn a_ricursive_lora_variant_it_does_not_train_is_refused_by_name() {
    let model = ripple_model_dir("ripple-variant");
    let err = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .rank_stabilized(true)
        .steps(1)
        .device(cpu())
        .run()
        .expect_err("a rank-stabilised adapter is not wired");
    assert!(err.to_string().contains("ricursive"), "{err}");
}

/// Spec: every weight trains in a full run and the result is a base the architecture opens again.
#[test]
fn a_ricursive_full_fine_tune_trains_every_weight_and_writes_a_base() {
    let model = ripple_model_dir("ripple-full");
    let outcome = ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .steps(3)
        .lr(1e-2)
        .seed(7) // seed: runs compared with each other need the same stream
        .device(cpu())
        .run()
        .expect("the full fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert!(outcome.final_loss.zip(outcome.initial_loss).is_some_and(|(last, first)| last < first), "{outcome:?}");
    let checkpoint = outcome.checkpoint.expect("a completed run writes its checkpoint directory");
    assert!(checkpoint.join("tokenizer.json").is_file());
    let trained = checkpoint::weightio::WeightReader::open(checkpoint.join("model.safetensors").to_str().unwrap()).expect("the checkpoint opens");
    let cfg = ricursive::TrainConfig::from_json(&trained.config()).expect("the header carries the training configuration");
    assert_eq!(trained.names().count(), cfg.param_list().len(), "every tensor of the architecture is written");
}

/// Spec: a preference run trains adapters over the frozen base, and the margin starts at zero against the model it
/// starts from.
#[test]
fn a_ricursive_preference_fine_tune_trains_adapters_over_the_frozen_base() {
    let model = ripple_model_dir("ripple-dpo");
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
        .arch("ricursive")
        .dataset(&pairs)
        .out_dir(model.join("run"))
        .rank(2)
        .steps(3)
        .lr(1e-1)
        .seed(7) // seed: runs compared with each other need the same stream
        .device(cpu())
        .run()
        .expect("the preference fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let initial = outcome.initial_loss.expect("measured");
    assert!((initial - std::f32::consts::LN_2).abs() < 1e-3, "the margin starts at zero against the starting model: {initial}");
    assert!(outcome.final_loss.is_some_and(|last| last < initial), "{outcome:?}");
    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    let st = checkpoint::st::load_safetensors(adapter.to_str().unwrap()).expect("the adapter opens");
    assert!(st.tensors.keys().all(|n| model::adapter::device::is_adapter_param(n)), "an adapter file holds adapter tensors only");
}
