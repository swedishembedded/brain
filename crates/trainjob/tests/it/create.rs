// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `init_model` through the executor: a model from nothing leaves as a manifest of artifacts and as a model of the
//! model store, the same request writes the same files, and what is not a declared capability, a bounded document,
//! a digest or an attached blob is refused before any work.

use serde_json::{json, Value};
use trainjob::create::{spec, ACTION};

use crate::support::{assert_reservation_returned, data_digest, rig_qwen3, run_action, work_is_clean, Frames, Rig};

/// A ChatML-like template with tools in the system turn and `<tool_call>` JSON calls.
pub const TEMPLATE: &str = r#"{%- if tools %}<|im_start|>system
# Tools
{% for t in tools %}{{ t | tojson }}
{% endfor %}<|im_end|>
{% endif %}
{%- for m in messages %}
{%- if m.role == 'assistant' %}<|im_start|>assistant
{{ m.content }}{% for c in m.tool_calls %}<tool_call>{"name": "{{ c.function.name }}", "arguments": {{ c.function.arguments | tojson }}}</tool_call>{% endfor %}<|im_end|>
{% else %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

/// A corpus of varied short sentences, one `{"text": ...}` per line.
pub fn corpus(docs: usize) -> Vec<u8> {
    let words = ["the", "cat", "sat", "on", "mat", "river", "stone", "window", "garden", "quickly", "under", "yellow", "machine", "letter", "answer", "morning", "winter", "bright", "little", "large"];
    (0..docs).map(|d| json!({"text": format!("{} {} {} {}.", words[d % words.len()], words[(d * 7) % words.len()], (d * 37) % 1000, words[(d * 3 + 1) % words.len()])}).to_string()).collect::<Vec<_>>().join("\n").into_bytes()
}

pub fn ouro_config() -> Value {
    json!({"architectures": ["OuroForCausalLM"], "model_type": "ouro", "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64, "max_position_embeddings": 512, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "total_ut_steps": 2, "tie_word_embeddings": false})
}

pub fn tokenizer() -> Value {
    json!({"vocab_size": 256 + 5 + 100, "special_tokens": ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"], "eos_token": "<|im_end|>"})
}

/// The params of an `init_model` request for a tiny ouro over the corpus `digest`.
pub fn init_params(digest: &str, install_as: &str) -> Value {
    json!({"arch": "ouro", "config": ouro_config().to_string(), "tokenizer": tokenizer().to_string(), "chat_template": TEMPLATE, "corpus_digest": digest, "seed": 7, "install_as": install_as})
}

fn files(rig: &Rig, report: &Value) -> Vec<(String, String)> {
    report["files"].as_array().unwrap().iter().map(|f| (f["name"].as_str().unwrap().to_string(), f["digest"].as_str().unwrap().to_string())).inspect(|(_, d)| assert!(rig.store.root().join(d).is_file())).collect()
}

#[test]
fn a_model_from_nothing_is_published_installed_and_the_same_request_writes_the_same_files() {
    let rig = rig_qwen3("init", 16 << 30);
    let digest = data_digest(&rig, &corpus(400)).hex();
    let frames = Frames::default();
    let out = run_action(&rig, ACTION, &spec(), init_params(&digest, "scratch-1"), &[], &frames).expect("the model is created").outputs;
    assert_eq!((out["status"].as_str(), out["objective"].as_str(), out["arch"].as_str(), out["seed"].as_u64(), out["vocab_size"].as_u64()), (Some("completed"), Some("init"), Some("ouro"), Some(7), Some(361)), "{out}");
    assert_eq!((out["installed_as"].as_str(), out["tool_calls"].as_bool()), (Some("local/scratch-1"), Some(true)));
    assert_eq!(out["config"]["vocab_size"], 361);
    let published = files(&rig, &out);
    let names: Vec<&str> = published.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["generation_config.json", "model.brain.safetensors", "tokenizer.json", "tokenizer_config.json"]);
    let installed = rig.dir.join("models").join("local").join("scratch-1");
    assert!(installed.join("model.brain.safetensors").is_file());
    assert!(frames.events().iter().any(|e| e["type"] == "stage"));
    // The same request (the corpus attached this time) writes the same bytes; a taken name is refused.
    let mut again = init_params(&digest, "scratch-2");
    again.as_object_mut().unwrap().remove("corpus_digest");
    let second = run_action(&rig, ACTION, &spec(), again, &[("corpus", corpus(400))], &Frames::default()).expect("again").outputs;
    assert_eq!(files(&rig, &second), published, "a seed, a corpus and settings determine the files");
    assert!(run_action(&rig, ACTION, &spec(), init_params(&digest, "scratch-1"), &[], &Frames::default()).unwrap_err().contains("already in the model store"));
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

/// A tokenizer spec with `split_digits` makes every digit its own token: the installed model's tokenizer encodes a
/// number one token per digit, and declares the split for any loader.
#[test]
fn a_tokenizer_with_digits_split_reads_numbers_one_digit_a_token() {
    let rig = rig_qwen3("init-digits", 16 << 30);
    let digest = data_digest(&rig, &corpus(400)).hex();
    let mut params = init_params(&digest, "digits-1");
    let mut tok = tokenizer();
    tok["split_digits"] = json!(true);
    // Digits learn no merges, so this small corpus holds fewer.
    tok["vocab_size"] = json!(256 + 5 + 80);
    params["tokenizer"] = json!(tok.to_string());
    run_action(&rig, ACTION, &spec(), params, &[], &Frames::default()).expect("the model is created");
    let path = rig.dir.join("models").join("local").join("digits-1").join("tokenizer.json");
    let written: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(written["pre_tokenizer"]["pretokenizers"][0], json!({"type": "Digits", "individual_digits": true}), "{}", written["pre_tokenizer"]);
    let bpe = data::qwen_tokenizer::QwenBpe::from_file(path.to_str().unwrap()).unwrap();
    assert_eq!(data::tokenizer::Tokenizer::encode(&bpe, "370").len(), 3, "one token per digit");
    assert_reservation_returned(&rig);
}

/// The sampling a request declares is the installed model's default: written into its `generation_config.json`, which
/// serving reads for a request that names none.
#[test]
fn the_declared_sampling_is_the_models_default() {
    let rig = rig_qwen3("init-sampling", 16 << 30);
    let digest = data_digest(&rig, &corpus(400)).hex();
    let mut params = init_params(&digest, "greedy-1");
    params["sampling"] = json!(json!({"do_sample": false, "top_k": 1}).to_string());
    run_action(&rig, ACTION, &spec(), params, &[], &Frames::default()).expect("the model is created");
    let dir = rig.dir.join("models").join("local").join("greedy-1");
    let declared = data::generation::SamplingDefaults::from_generation_config(&dir).unwrap();
    assert_eq!((declared.temp, declared.top_k), (Some(0.0), Some(1)), "do_sample false is greedy");
    let written: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("generation_config.json")).unwrap()).unwrap();
    assert!(written["eos_token_id"].is_array(), "the end-of-sequence ids stay: {written}");
    assert_reservation_returned(&rig);
}

#[test]
fn init_model_refuses_what_it_cannot_create_before_any_work() {
    let rig = rig_qwen3("init-door", 16 << 30);
    let digest = data_digest(&rig, &corpus(400)).hex();
    let frames = Frames::default();
    let ask = |patch: Value, blobs: &[(&str, Vec<u8>)]| {
        let mut p = init_params(&digest, "door");
        for (k, v) in patch.as_object().unwrap() {
            if v.is_null() {
                p.as_object_mut().unwrap().remove(k);
            } else {
                p[k] = v.clone();
            }
        }
        run_action(&rig, ACTION, &spec(), p, blobs, &frames).unwrap_err()
    };
    for (patch, needle) in [
        (json!({"arch": "ricursive"}), "does not declare a from-scratch initialization"),
        (json!({"arch": "nonesuch"}), "nonesuch"),
        (json!({"corpus_digest": "/etc/passwd"}), "64 lowercase hexadecimal"),
        (json!({"corpus_digest": trainjob::store::digest_of(b"absent").hex()}), "not in the artifact store"),
        (json!({"corpus_digest": null}), "exactly once"),
        (json!({"install_as": "../escape"}), "install_as"),
        (json!({"config": "[1, 2"}), "config"),
        (json!({"config": format!("{{\"pad\": \"{}\"}}", "x".repeat(70_000))}), "at most 65536"),
        (json!({"tokenizer": json!({"vocab_size": 300, "special_tokens": ["<e>"], "eos_token": "<e>", "extra": 1}).to_string()}), "unknown field"),
        (json!({"tokenizer": json!({"vocab_size": 300, "special_tokens": ["<e>"], "eos_token": "<f>"}).to_string()}), "not one of the special tokens"),
        (json!({"config": json!({"model_type": "ouro", "hidden_size": 65536, "num_hidden_layers": 64, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 65536, "intermediate_size": 65536, "max_position_embeddings": 2048, "rope_theta": 1e4, "rms_norm_eps": 1e-6, "total_ut_steps": 4}).to_string()}), "parameters"),
        (json!({"seed": -1}), "seed"),
        (json!({"tokenizer_bytes": 0}), "tokenizer_bytes"),
        (json!({"sampling": json!({"temperature": 7000}).to_string()}), "temperature"),
        (json!({"sampling": json!({"beam": 4}).to_string()}), "not a sampling setting"),
        (json!({"sampling": json!({"do_sample": "no"}).to_string()}), "do_sample"),
        (json!({"path": "/etc/passwd"}), "unknown param"),
        (json!({"out": "elsewhere"}), "unknown param"),
    ] {
        let e = ask(patch.clone(), &[]);
        assert!(e.contains(needle), "{patch}: {e}");
    }
    assert!(ask(json!({}), &[("corpus", corpus(10))]).contains("exactly once"), "a digest and a blob both");
    let e = ask(json!({"corpus_digest": null}), &[("dataset", corpus(10))]);
    assert!(e.contains("dataset"), "a blob the action does not take is refused by name: {e}");
    assert!(frames.events().is_empty() && work_is_clean(&rig.work_root));
    assert!(!rig.dir.join("models").join("local").join("door").exists());
    assert!(rig.exec.residency().placements.is_empty());
}
