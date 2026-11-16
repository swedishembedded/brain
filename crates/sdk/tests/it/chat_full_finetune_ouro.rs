// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning recurrent-depth
// language models through one generic training stack for its clients. If your
// team needs expertise in looped decoders then you can procure our services by
// sending an email to info@swedishembedded.com.

#![cfg(feature = "study")]

//! Ouro through the generic full-parameter chat fine-tuner, on the CPU backend: a brain checkpoint with no
//! `config.json` is told by its card, every weight trains on role-masked chat records, the trained checkpoint is
//! written under the model store's base file name, and an adapter run is refused by the capability it lacks.

use brain::{ChatFineTune, ChatFullFineTune, FineTuneStatus};

use crate::chat_fixture::{cpu, TempDir};

/// A byte-level vocabulary of `a`..`w` and the three chat tokens, 26 ids: a small head keeps the CPU run short.
fn write_small_tokenizer(dir: &std::path::Path) {
    let byte_encoder = data::bpe::bytes_to_unicode();
    let vocab: serde_json::Map<String, serde_json::Value> = (b'a'..=b'w').enumerate().map(|(id, b)| (byte_encoder[b as usize].to_string(), serde_json::json!(id as u32))).collect();
    let added = serde_json::json!([{"content": "<|endoftext|>", "id": 23}, {"content": "<|im_start|>", "id": 24}, {"content": "<|im_end|>", "id": 25}]);
    std::fs::write(dir.join("tokenizer.json"), serde_json::json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": added}).to_string()).unwrap();
    let template = "{% for message in messages %}<|im_start|>{{ message.role }}{{ message.content }}<|im_end|>{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}";
    std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({"chat_template": template, "eos_token": "<|im_end|>"}).to_string()).unwrap();
}

/// A tiny Ouro checkpoint at the fixture tokenizer's vocabulary, as the model saves one (card family `ouro`).
fn ouro_model_dir(tag: &str) -> TempDir {
    let dir = TempDir::new(tag);
    let cfg = ouro::OuroConfig { vocab: 26, block_size: 128, ..ouro::OuroConfig::tiny() };
    // seed: any weights serve; the test reads a training, never the model's quality.
    let mut init = ouro::init::init_weights(&cfg, 11);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, numel)| { let w = init.remove(&name).unwrap(); (name, vec![numel as u64], w) }).collect();
    let card = checkpoint::st::ModelCard::new("brain/ouro", "ouro");
    checkpoint::st::save_safetensors(dir.join("model.brain.safetensors").to_str().unwrap(), &tensors, &cfg.to_json(), Some(&card)).unwrap();
    write_small_tokenizer(&dir);
    dir
}

#[test]
fn an_ouro_full_fine_tune_trains_every_weight_and_writes_a_base_the_store_opens() {
    let model = ouro_model_dir("ouro-full");
    let records: Vec<String> = [("cab", "bad"), ("fed", "dad"), ("ace", "cab")]
        .iter()
        .map(|(u, a)| serde_json::json!({"messages": [{"role": "user", "content": u, "train": false}, {"role": "assistant", "content": a, "train": true}], "tools": []}).to_string())
        .collect();
    let dataset = model.join("chat.jsonl");
    std::fs::write(&dataset, records.join("\n")).unwrap();
    // No architecture named: the checkpoint's card says it is ouro.
    let outcome = ChatFullFineTune::from_pretrained(model.to_str().unwrap()).dataset(&dataset).out_dir(model.join("run")).steps(6).lr(5e-2).seed(7).device(cpu()).run().expect("the full fine-tune runs");
    assert_eq!((outcome.status, outcome.steps_completed), (FineTuneStatus::Completed, 6));
    assert!(outcome.final_loss.zip(outcome.initial_loss).is_some_and(|(last, first)| last < first), "{outcome:?}");
    let checkpoint = outcome.checkpoint.expect("a completed run writes its checkpoint directory");
    assert_eq!(outcome.weights.as_deref(), Some(checkpoint.join("model.brain.safetensors").as_path()));
    assert!(checkpoint.join("tokenizer_config.json").is_file());
    let card = checkpoint::st::read_card(checkpoint.join("model.brain.safetensors").to_str().unwrap()).unwrap().expect("a card");
    assert_eq!(card.family, "ouro");
    // The trained checkpoint is a base the store resolves, and trains again.
    let again = ChatFullFineTune::from_pretrained(checkpoint.to_str().unwrap()).dataset(&dataset).out_dir(model.join("again")).steps(1).seed(7).device(cpu()).run().expect("the trained checkpoint is a base");
    assert_eq!(again.status, FineTuneStatus::Completed);

    let refused = ChatFineTune::from_pretrained(model.to_str().unwrap()).arch("ouro").dataset(&dataset).out_dir(model.join("lora")).steps(1).device(cpu()).run().expect_err("no LoRA trainer").to_string();
    assert!(refused.contains("ouro"), "{refused}");
}
