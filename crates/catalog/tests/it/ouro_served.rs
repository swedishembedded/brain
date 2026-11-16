// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving small models trained
// from scratch with the chat format they were trained in for its clients. If
// your team needs expertise in keeping training and serving renderings
// identical then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A from-scratch Ouro checkpoint served with its own chat template: a request's messages and tools (the OpenAI
//! shape, assistant `tool_calls` with JSON-text arguments, `tool` results) render to exactly the tokens the chat
//! fine-tune trained the same conversation on; a reply's tool call is read back as the call; and the resident serves
//! the request on the CPU within the model's positions.

use std::path::Path;

use capability::Invocation;
use catalog::resident_ouro::{chat_format, OuroResident};
use checkpoint::st::ModelCard;
use data::tokenizer::Tokenizer;
use model::finetune::ChatTrainer;
use residency::{Device, InstanceKey, ResidentModel};
use serde_json::{json, Value};

/// A ChatML-like template with tools in the system turn and `<tool_call>` JSON calls: not Qwen's, a template of its own.
const TEMPLATE: &str = r#"{%- if tools %}<|im_start|>system
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

const SPECIALS: [&str; 5] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"];

/// A from-scratch model directory: a byte-level vocabulary, the template, an initialised tiny Ouro.
fn model_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ouro-served-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let text: String = (0..400).map(|i| format!("read the file {i}.txt and tell me what number it holds, the answer is {}.\n", i * 7)).collect();
    let bpe = data::bytebpe::ByteBpe::train_reserving(&text, 120, &SPECIALS);
    std::fs::write(dir.join("tokenizer.json"), bpe.to_hf_tokenizer_json(&SPECIALS).unwrap().to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), json!({"chat_template": TEMPLATE, "eos_token": "<|im_end|>"}).to_string()).unwrap();
    let vocab = 256 + bpe.merge_count() + SPECIALS.len();
    let cfg = ouro::OuroConfig { vocab: vocab as u32, block_size: 256, ..ouro::OuroConfig::tiny() };
    // seed: any weights serve; the test reads renderings and a served request, never the model's quality.
    ouro::finetune::OuroTrainer::init_checkpoint(&cfg.to_json(), 3, &dir).unwrap();
    dir
}

fn tools() -> Value {
    json!([{"type": "function", "function": {"name": "read_file", "description": "read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}])
}

/// A conversation in the OpenAI request shape: an earlier tool call and its result, then the answer.
fn conversation() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "use the tools"}),
        json!({"role": "user", "content": "what is in 3.txt"}),
        json!({"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"3.txt\"}"}}]}),
        json!({"role": "tool", "content": "21", "tool_call_id": "call_0"}),
        json!({"role": "assistant", "content": "it holds 21"}),
    ]
}

fn ask(messages: &[Value]) -> Invocation {
    Invocation::new().set("messages", json!(Value::from(messages.to_vec()).to_string())).set("tools", json!(tools().to_string()))
}

fn tokenizer(dir: &Path) -> data::qwen_tokenizer::QwenBpe {
    data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap()
}

#[test]
fn the_served_prompt_is_the_training_rendering_of_the_same_conversation() {
    let dir = model_dir("render");
    let tok = tokenizer(&dir);
    // Training: the record as the chat fine-tune reads it, the final answer supervised.
    let mut record = conversation();
    for (i, m) in record.iter_mut().enumerate() {
        m["train"] = json!(i == 4);
    }
    let line = json!({"messages": record, "tools": tools()}).to_string();
    let sample = data::chat::ChatSample::from_json_line_with(&line, &data::chat::TextOnly::TRAINER).unwrap();
    let tmpl = data::chat_template::ChatTemplate::from_model_dir(&dir).unwrap();
    // As the trainer takes a record whose context holds tool turns: its answer as the model is asked for it.
    let sample = sample.final_turn_as_asked(&tmpl, true).unwrap().expect("one supervised answer");
    let (ids, mask) = sample.encode_with(&tok, &tmpl, data::chat::RenderOpts::default()).unwrap();
    let first = mask.iter().position(|m| *m).expect("the answer is supervised");
    // Serving: the request of everything before the answer, rendered with the generation prompt.
    let format = chat_format(Some(&dir), &tok);
    let served = qwen3::chat::render_prompt_as(&format, &ask(&conversation()[..4])).unwrap();
    assert_eq!(tok.encode(&served.text), ids[..first], "the served prompt is the training context, token for token");
    assert_eq!(tok.decode(&ids[first..]), "it holds 21<|im_end|>\n", "what is supervised is the answer and its end of turn");

    // A reply holding a call is read back as the call, the way the template wrote it.
    let reply = r#"<tool_call>{"name": "read_file", "arguments": {"path": "4.txt"}}</tool_call>"#;
    let outcome = qwen3::chat::parse_reply_as(&format, &ask(&conversation()[..2]), reply).unwrap();
    let calls: Value = serde_json::from_str(outcome.outputs["tool_calls"].as_str().unwrap()).unwrap();
    assert_eq!((calls[0]["name"].as_str(), calls[0]["arguments"].as_str()), (Some("read_file"), Some(r#"{"path": "4.txt"}"#)), "{calls}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_resident_serves_a_tool_request_through_its_own_template_within_its_positions() {
    let dir = model_dir("serve");
    let tok = tokenizer(&dir);
    let resident = OuroResident::from_card(dir.join("model.brain.safetensors").to_str().unwrap(), &ModelCard::new("local/scratch", "ouro"), None).unwrap().with_sizing(4096, 16, 2);
    assert_eq!(resident.manifest().max_context_tokens, Some(256), "the context is the model's positions, not the pool's");
    let format = chat_format(Some(&dir), &tok);
    let messages = &conversation()[..4];
    let expected = qwen3::chat::render_prompt_as(&format, &ask(messages)).unwrap();
    gpu_core::devices::with_host_tier(|| {
        let mut instance = resident.activate(&InstanceKey::new("local/scratch", "default"), Device::Cpu).expect("the resident activates");
        let inv = ask(messages).set("max_new", json!(3)).set("temperature", json!(0.0));
        let out = instance.run("generate", &inv, &mut |_| {}).expect("served").outputs;
        assert_eq!(out["prompt_tokens"].as_u64(), Some(tok.encode(&expected.text).len() as u64), "{out}");
    });
    std::fs::remove_dir_all(&dir).ok();
}
