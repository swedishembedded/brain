// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on the exact prompts they are served for
// its clients. If your team needs expertise in train-serve consistency of chat models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a training record is trained in the context the model is served in. For every supervised answer of an
//! agent record (a system prompt, a user turn, a reasoned tool call, its result, a reasoned final answer), the ids
//! the trainer renders before the answer are, token for token, the ids the served `generate` action renders for the
//! same messages and tools: through each of the trainer's renderings (each answer as asked, the packed dialogue's
//! first answer, and a plain record rendered whole). Knowledge 531 is the defect this guards against: a served
//! prompt carrying an instruction the trainer never rendered, so no recorded context matched.
//!
//! Runs on the real Qwen3-4B tokenizer and chat template from the model store (skipped by name when absent). The
//! comparison is shown to fail on two mutants of the trainer's rendering: the tool schemas left out, and an
//! instruction in the system turn the served prompt does not carry.

use capability::Invocation;
use data::chat::{ChatSample, TextOnly};
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::chat::{parse_request_as, ChatFormat};
use serde_json::{json, Value};

const TOOLS: &str = r#"[{"type":"function","function":{"name":"read_file","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]"#;

/// The agent record: which messages are supervised is the `train` flag; the served request is the same messages
/// without it.
fn record() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are a careful software agent.", "train": false}),
        json!({"role": "user", "content": "Time: 2026-11-21T08:00:00Z\n\nWhat does case.md say about refunds?", "train": false}),
        json!({"role": "assistant", "content": "", "reasoning_content": "I should read the file.", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"case.md\"}"}}], "train": true}),
        json!({"role": "tool", "tool_call_id": "call_0", "content": "# Case\nRefunds within 30 days.", "train": false}),
        json!({"role": "assistant", "content": "Refunds are accepted within 30 days.", "reasoning_content": "The file states the rule.", "train": true}),
    ]
}

/// The record as a dataset line. The tool schemas are spliced in as written: a `serde_json::Value` would sort their
/// keys, and the template renders them in the order the line holds them.
fn sample(messages: &[Value], tools: bool) -> ChatSample {
    let messages = Value::from(messages.to_vec()).to_string();
    let line = if tools { format!(r#"{{"messages":{messages},"tools":{TOOLS}}}"#) } else { format!(r#"{{"messages":{messages}}}"#) };
    ChatSample::from_json_line_with(&line, &TextOnly::TRAINER).expect("a valid record")
}

/// The ids the served `generate` action renders for `messages` (their `train` flags dropped) and the tools.
fn served(tok: &QwenBpe, format: &ChatFormat, messages: &[Value]) -> Vec<u32> {
    let turns: Vec<Value> = messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            m.as_object_mut().unwrap().remove("train");
            m
        })
        .collect();
    let generate = qwen3::caps::manifest().actions.into_iter().find(|a| a.name == "generate").expect("generate").for_serving();
    let request = Invocation::new().set("messages", json!(Value::from(turns).to_string())).set("tools", json!(TOOLS));
    let admitted = generate.validate_served(request).expect("the request is admitted");
    parse_request_as(tok, format, &admitted).expect("the request renders").ids
}

/// The trainer's context: the ids before the first supervised one.
fn context(ids: &[u32], mask: &[bool]) -> Vec<u32> {
    ids[..mask.iter().position(|&m| m).expect("something is supervised")].to_vec()
}

/// `Err` with the first differing position when `trained` is not `served`.
fn same(tok: &QwenBpe, what: &str, trained: &[u32], served: &[u32]) -> Result<(), String> {
    if trained == served {
        return Ok(());
    }
    let at = trained.iter().zip(served).position(|(a, b)| a != b).unwrap_or(trained.len().min(served.len()));
    let show = |v: &[u32]| tok.decode(&v[at.min(v.len())..(at + 12).min(v.len())]);
    Err(format!("{what}: the trainer's context ({} ids) and the served prompt ({} ids) differ at {at}: trainer {:?} / served {:?}", trained.len(), served.len(), show(trained), show(served)))
}

/// The trainer's context of each supervised answer of `s`, through `answers_as_asked` with reasoning `thinking`.
fn asked_contexts(s: &ChatSample, tok: &QwenBpe, tmpl: &ChatTemplate, thinking: bool) -> Vec<Vec<u32>> {
    s.answers_as_asked(tmpl, thinking).unwrap().expect("assistant answers").iter().map(|a| a.encode(tok, tmpl).map(|(ids, mask)| context(&ids, &mask)).unwrap()).collect()
}

#[test]
fn every_answer_is_trained_in_the_context_it_is_served_in() {
    let Some(dir) = brain_testutil::model_dir("Qwen/Qwen3-4B").map(std::path::PathBuf::from).filter(|d| d.join("tokenizer_config.json").is_file()) else {
        brain_testutil::skip("Qwen/Qwen3-4B not in the model store (brain pull Qwen/Qwen3-4B)");
        return;
    };
    let tok = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).expect("Qwen3-4B tokenizer");
    let tmpl = ChatTemplate::from_model_dir(&dir).expect("Qwen3-4B chat template");
    let format = ChatFormat::for_checkpoint(Some(&dir), &tok);
    let msgs = record();
    let answers: Vec<usize> = msgs.iter().enumerate().filter(|(_, m)| m["train"] == json!(true)).map(|(i, _)| i).collect();
    let served_at: Vec<Vec<u32>> = answers.iter().map(|&at| served(&tok, &format, &msgs[..at])).collect();

    // Each answer as asked: its own context.
    let full = sample(&msgs, true);
    for (k, ctx) in asked_contexts(&full, &tok, &tmpl, true).iter().enumerate() {
        same(&tok, &format!("answer {k} as asked"), ctx, &served_at[k]).unwrap_or_else(|why| panic!("{why}"));
    }
    // The packed dialogue: its first answer's context (a later answer sees the earlier one as trained, by design).
    let packed = full.answers_packed(&tmpl, true).unwrap().expect("assistant answers");
    let (ids, mask) = packed.encode(&tok, &tmpl).unwrap();
    same(&tok, "the packed dialogue's first answer", &context(&ids, &mask), &served_at[0]).unwrap_or_else(|why| panic!("{why}"));
    // A plain record rendered whole: its ids begin with the served prompt of its context.
    let plain = [msgs[0].clone(), msgs[1].clone(), json!({"role": "assistant", "content": "I will read case.md.", "train": true})];
    let (ids, _) = sample(&plain, true).encode(&tok, &tmpl).unwrap();
    let want = served(&tok, &format, &plain[..2]);
    same(&tok, "a plain record rendered whole", &ids[..want.len().min(ids.len())], &want).unwrap_or_else(|why| panic!("{why}"));

    // Mutants of the trainer's rendering.
    let why = same(&tok, "tools left out", &asked_contexts(&sample(&msgs, false), &tok, &tmpl, true)[0], &served_at[0]).expect_err("a context without the tools must differ");
    eprintln!("mutant \"tool schemas left out\" killed: {why}");
    // Knowledge 531's shape: one side writes an instruction into the system turn that the other does not.
    let mut instructed = msgs.clone();
    instructed[0]["content"] = json!("Reasoning effort: high.\n\nYou are a careful software agent.");
    let why = same(&tok, "an extra system instruction", &asked_contexts(&sample(&instructed, true), &tok, &tmpl, true)[0], &served_at[0]).expect_err("a context with an instruction the served prompt lacks must differ");
    eprintln!("mutant \"system instruction on one side only\" killed: {why}");
}
