// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from
// random weights through one architecture-generic pipeline for its clients. If
// your team needs expertise in comparing small architectures trained the same
// way then you can procure our services by sending an email to
// info@swedishembedded.com.

#![cfg(feature = "study")]

//! The from-scratch sequence over every architecture that declares it, on the CPU backend: a vocabulary and chat
//! template with a random initialization ([`brain::ModelInit`]), raw-text pretraining with held-out evaluation and an
//! exact resume ([`brain::Pretrain`]), and a chat full fine-tune on tool-call records ([`brain::ChatFullFineTune`]).
//! The same calls run on each family; only the tiny configuration differs.

use std::path::{Path, PathBuf};

use brain::{ChatFullFineTune, FineTuneStatus, ModelInit, Pretrain};

use crate::chat_fixture::TempDir;

/// The chat template a from-scratch model in these tests is made with: ChatML-like turns, the tool schemas in the
/// system turn and each call as `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
pub const TEMPLATE: &str = r#"{%- if tools %}<|im_start|>system
{% if messages[0].role == 'system' %}{{ messages[0].content }}
{% endif %}# Tools
{% for t in tools %}{{ t | tojson }}
{% endfor %}<|im_end|>
{% elif messages[0].role == 'system' %}<|im_start|>system
{{ messages[0].content }}<|im_end|>
{% endif %}
{%- for m in messages %}
{%- if m.role == 'system' %}
{%- elif m.role == 'assistant' %}<|im_start|>assistant
{{ m.content }}{% for c in m.tool_calls %}<tool_call>{"name": "{{ c.function.name }}", "arguments": {{ c.function.arguments | tojson }}}</tool_call>{% endfor %}<|im_end|>
{% else %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

pub const SPECIALS: [&str; 5] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"];

/// A tiny configuration of `arch` in its `config.json` spelling, for every architecture that declares the
/// capability. A new one fails [`every_from_scratch_architecture_runs_the_whole_sequence`] until it is added here.
pub fn tiny_config(arch: &str) -> Option<serde_json::Value> {
    match arch {
        "ouro" => Some(serde_json::json!({
            "architectures": ["OuroForCausalLM"], "model_type": "ouro", "hidden_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64,
            "max_position_embeddings": 512, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "total_ut_steps": 2,
            "tie_word_embeddings": false
        })),
        "qwen3" => Some(serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "intermediate_size": 64,
            "max_position_embeddings": 512, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": false
        })),
        _ => None,
    }
}

/// A corpus of short sentences, one `{"text": ...}` document per line; `offset` makes a disjoint held-out set.
pub fn write_corpus(path: &Path, docs: usize, offset: usize) {
    let words = [
        "the", "cat", "sat", "on", "a", "mat", "and", "then", "read", "file", "tool", "number", "is", "seven", "box", "green", "river", "stone",
        "window", "garden", "quickly", "slowly", "under", "over", "between", "yellow", "purple", "orange", "machine", "letter", "answer", "question",
        "morning", "evening", "summer", "winter", "bright", "quiet", "little", "large",
    ];
    let lines: Vec<String> = (0..docs)
        .map(|d| {
            let n = 6 + (d + offset) % 7;
            let text: Vec<String> = (0..n).map(|i| if i == 2 { format!("{}", (d * 37 + offset) % 1000) } else { words[(d * 7 + i * 11 + offset) % words.len()].to_string() }).collect();
            serde_json::json!({"text": format!("{}.", text.join(" "))}).to_string()
        })
        .collect();
    std::fs::write(path, lines.join("\n")).unwrap();
}

/// Tool-call chat records: the assistant reads a file through a tool and answers with what it held.
pub fn write_tool_records(path: &Path) {
    let tools = serde_json::json!([{"type": "function", "function": {"name": "read_file", "description": "read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]);
    let lines: Vec<String> = ["a", "b", "c"]
        .iter()
        .enumerate()
        .map(|(i, f)| {
            serde_json::json!({"tools": tools, "messages": [
                {"role": "system", "content": "use tools", "train": false},
                {"role": "user", "content": format!("what is in {f}.txt"), "train": false},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": format!("{{\"path\": \"{f}.txt\"}}")}}], "train": true},
                {"role": "tool", "content": format!("{i}"), "tool_call_id": "call_0", "train": false},
                {"role": "assistant", "content": format!("the number is {i}"), "train": true}
            ]})
            .to_string()
        })
        .collect();
    std::fs::write(path, lines.join("\n")).unwrap();
}

/// The from-scratch model of `arch` the sequence starts from.
pub fn init(arch: &str, corpus: &Path, out: &Path, seed: u64) -> brain::Result<brain::ModelInitOutcome> {
    ModelInit::new(arch)
        .config(tiny_config(arch).expect("a tiny config"))
        .corpus(corpus)
        .vocab_size(256 + SPECIALS.len() + 120)
        .special_tokens(SPECIALS)
        .eos_token("<|im_end|>")
        .chat_template(TEMPLATE)
        .seed(seed)
        .out_dir(out)
        .run()
}

fn pretrain(base: &Path, dir: &TempDir, out: &str, steps: u32) -> Pretrain {
    Pretrain::from_pretrained(base.to_str().unwrap())
        .corpus(dir.join("corpus.jsonl"))
        .held_out(dir.join("held.jsonl"))
        .steps(steps)
        .batch(2)
        .block(32)
        .lr(5e-3)
        .eval_every(3)
        .eval_batches(2)
        // seed: a fixed draw order, so a resumed run and a straight one are comparable.
        .seed(3)
        .device(crate::chat_fixture::cpu())
        .out_dir(dir.join(out))
}

fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap())).collect();
    out.sort();
    out
}

#[test]
fn every_from_scratch_architecture_runs_the_whole_sequence() {
    let archs = brain::from_scratch_architectures();
    assert!(archs.contains(&"ouro") && archs.contains(&"qwen3"), "{archs:?}");
    for arch in archs {
        assert!(tiny_config(arch).is_some(), "{arch} declares a from-scratch initialization: give it a tiny config here");
        let dir = TempDir::new(&format!("from-scratch-{arch}"));
        write_corpus(&dir.join("corpus.jsonl"), 300, 0);
        write_corpus(&dir.join("held.jsonl"), 40, 5);
        write_tool_records(&dir.join("tools.jsonl"));

        // A vocabulary, a chat format and random weights, the same bytes for the same seed.
        let made = init(arch, &dir.join("corpus.jsonl"), &dir.join("base"), 7).unwrap_or_else(|e| panic!("{arch}: {e}"));
        let again = init(arch, &dir.join("corpus.jsonl"), &dir.join("again"), 7).unwrap();
        assert_eq!(files(&made.checkpoint), files(&again.checkpoint), "{arch}: the same seed and corpus write the same checkpoint");
        assert!(made.tool_calls, "{arch}: the template serves tool calls");
        assert_eq!((made.vocab_size, made.seed), (256 + SPECIALS.len() + 120, 7));
        assert!(files(&made.checkpoint).iter().any(|(n, _)| n == "tokenizer.json") && made.params > 0);

        // Pretraining with evaluations, stopped after a periodic save and continued, ends where a straight run ends.
        let straight = pretrain(&made.checkpoint, &dir, "straight", 6).run().unwrap_or_else(|e| panic!("{arch}: {e}"));
        assert_eq!((straight.status, straight.steps_completed, straight.resumed_at), (FineTuneStatus::Completed, 6, None));
        assert_eq!(straight.evaluations.iter().map(|e| e.0).collect::<Vec<_>>(), [3, 6], "{arch}: evaluated every 3 steps");
        assert!(straight.evaluations.iter().all(|e| e.2.is_finite()) && straight.final_loss.is_some_and(f32::is_finite));
        let state = dir.join("pretrain.state");
        let cancel = brain::CancelToken::armed();
        let first = pretrain(&made.checkpoint, &dir, "resumed", 6).resume_state(&state).checkpoint_every(2).run_with(&cancel, |p| {
            if p.step == 3 {
                cancel.cancel();
            }
        });
        assert_eq!(first.unwrap().status, FineTuneStatus::Cancelled);
        assert!(state.exists(), "{arch}: a stopped run leaves its state");
        let resumed = pretrain(&made.checkpoint, &dir, "resumed", 6).resume_state(&state).checkpoint_every(2).run().unwrap();
        assert_eq!((resumed.status, resumed.resumed_at), (FineTuneStatus::Completed, Some(3)));
        assert_eq!(resumed.weights_digest, straight.weights_digest, "{arch}: the resumed run trains the same weights as the straight one");
        assert_eq!(resumed.evaluations, straight.evaluations);
        assert!(!state.exists(), "{arch}: an exported run removes its state");
        // A state of another run is refused, never continued.
        let other = pretrain(&made.checkpoint, &dir, "other", 6).resume_state(&state).checkpoint_every(1).run_with(&cancel, |_| {});
        assert_eq!(other.unwrap().status, FineTuneStatus::Cancelled);
        let refused = pretrain(&made.checkpoint, &dir, "other", 6).lr(1e-3).resume_state(&state).run().unwrap_err().to_string();
        assert!(refused.contains("cannot be continued"), "{arch}: {refused}");

        // The pretrained checkpoint takes a chat full fine-tune on tool-call records.
        let pretrained: PathBuf = straight.checkpoint.clone().unwrap();
        let tuned = ChatFullFineTune::from_pretrained(pretrained.to_str().unwrap())
            .dataset(dir.join("tools.jsonl"))
            .steps(3)
            .lr(5e-3)
            .seed(3)
            .thinking(false)
            .device(crate::chat_fixture::cpu())
            .out_dir(dir.join("chat"))
            .run()
            .unwrap_or_else(|e| panic!("{arch}: {e}"));
        assert_eq!((tuned.status, tuned.steps_completed), (FineTuneStatus::Completed, 3), "{arch}");
        assert!(tuned.checkpoint.as_deref().is_some_and(|c| c.join("tokenizer_config.json").exists()));
    }
}

#[test]
fn a_from_scratch_model_is_refused_by_what_it_cannot_be_before_anything_is_written() {
    let dir = TempDir::new("from-scratch-refusals");
    write_corpus(&dir.join("corpus.jsonl"), 300, 0);
    let corpus = dir.join("corpus.jsonl");
    let base = || ModelInit::new("ouro").config(tiny_config("ouro").unwrap()).corpus(&corpus).vocab_size(256 + 5 + 120).special_tokens(SPECIALS).eos_token("<|im_end|>").chat_template(TEMPLATE).seed(1);
    let refused = |init: ModelInit, out: &str, needle: &str| {
        let e = init.out_dir(dir.join(out)).run().unwrap_err().to_string();
        assert!(e.contains(needle), "{needle}: {e}");
        assert!(!dir.join(out).exists(), "{out}: nothing is left behind");
    };
    refused(ModelInit::new("ricursive").config(serde_json::json!({})).corpus(&corpus).vocab_size(400).special_tokens(["<e>"]).eos_token("<e>"), "a", "does not declare a from-scratch initialization");
    refused(base().eos_token("<|nope|>"), "b", "not one of the special tokens");
    refused(base().special_tokens(["<a>", "<a>"]).eos_token("<a>"), "c", "declared twice");
    refused(base().vocab_size(100), "d", "vocab_size");
    refused(base().vocab_size(256 + 5 + 100_000), "e", "merges");
    refused(base().config(serde_json::json!({"hidden_size": 8})), "f", "missing keys");
    refused(base().config(serde_json::json!(5)), "g", "JSON object");
    let mut wrong_vocab = tiny_config("ouro").unwrap();
    wrong_vocab["vocab_size"] = serde_json::json!(999);
    refused(base().config(wrong_vocab), "h", "is not the vocabulary's");
    // A template that never closes an assistant turn with the eos token would train a model that never stops.
    refused(base().chat_template("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"), "i", "eos token");
    // A template whose tool-call turn the serving parser cannot read back is refused.
    let unreadable = TEMPLATE.replace(r#"<tool_call>{"name": "{{ c.function.name }}", "arguments": {{ c.function.arguments | tojson }}}</tool_call>"#, "CALL {{ c.function.name }}");
    refused(base().chat_template(unreadable), "j", "serving parser");
    refused(base().chat_template("{% for"), "k", "chat template");
}
