// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on the
// exact prompts they were served for its clients. If your team needs expertise
// in serving and on-policy training pipelines then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The prompt a served Qwen3 checkpoint is given is its own chat template's
//! rendering of the request, token for token: an agent turn as an agent runtime
//! sends it (system prompt, a user turn with the harness's time line, tool
//! schemas in the order the request wrote them, an earlier reasoned tool call
//! and its result), admitted through the served `generate` action with every
//! default it declares, renders and encodes to the ids the checkpoint's Jinja
//! template gives. A served prompt that differed (an instruction the template
//! never writes, re-sorted schemas) would make every recorded log-probability
//! belong to a context no trainer reproduces.

use std::collections::BTreeMap;

use capability::Invocation;
use data::chat_template::{parse_json_ordered, ChatTemplate, Value};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::chat::{parse_request_as, ChatFormat};
use serde_json::json;

const TOOLS: &str = r#"[{"type":"function","function":{"name":"shell","description":"Run a shell command in the workspace.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"the command"}},"required":["command"]}}},{"type":"function","function":{"name":"read_file","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]"#;

fn agent_turns() -> serde_json::Value {
    json!([
        {"role": "system", "content": "You are a careful software agent."},
        {"role": "user", "content": "Time: 2026-11-19T08:00:00Z\n\nFix the failing test in case.md."},
        {"role": "assistant", "content": "", "reasoning_content": "I should look at the file first.", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"case.md\"}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "# Case\nThe refund rule is wrong."},
    ])
}

#[test]
fn a_served_qwen3_prompt_is_the_checkpoints_own_template_rendering() {
    let Some(dir) = brain_testutil::model_dir("Qwen/Qwen3-4B").map(std::path::PathBuf::from).filter(|d| d.join("tokenizer_config.json").is_file()) else {
        brain_testutil::skip("Qwen/Qwen3-4B not in the model store (brain pull Qwen/Qwen3-4B)");
        return;
    };
    let tok = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).expect("Qwen3-4B tokenizer");
    let template = ChatTemplate::from_model_dir(&dir).expect("Qwen3-4B chat template");

    // The request as a serving route admits it: the served generate action fills every default it declares.
    let generate = qwen3::caps::manifest().actions.into_iter().find(|a| a.name == "generate").expect("generate").for_serving();
    let request = Invocation::new().set("messages", json!(agent_turns().to_string())).set("tools", json!(TOOLS));
    let admitted = generate.validate_served(request).expect("the request is admitted");
    let format = ChatFormat::for_checkpoint(Some(&dir), &tok);
    assert!(matches!(format, ChatFormat::Qwen), "Qwen3-4B renders ChatML through the Qwen renderer");
    let served = parse_request_as(&tok, &format, &admitted).expect("the request renders").ids;

    // The checkpoint's own template, given the same turns and the tools as written.
    let turns = template.conversation(agent_turns().as_array().unwrap()).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(true));
    let text = template.render(turns, Some(parse_json_ordered(TOOLS).unwrap()), true, &extra).expect("the template renders");
    let own = tok.encode(&text);

    let first = served.iter().zip(&own).position(|(a, b)| a != b);
    assert_eq!(served, own, "first difference at {first:?}: served {:?} / template {:?}", first.map(|i| tok.decode(&served[i..(i + 12).min(served.len())])), first.map(|i| tok.decode(&own[i..(i + 12).min(own.len())])));
}

/// A template that reads `reasoning_effort` (Qwen3.8's) gets the request's value, and its own default when the
/// request leaves it out: the served prompt equals the checkpoint template's rendering either way.
#[test]
fn reasoning_effort_reaches_a_template_that_reads_it() {
    let Some(dir) = brain_testutil::model_dir("Qwen/Qwen3.8-27B-FP8").map(std::path::PathBuf::from).filter(|d| d.join("chat_template.jinja").is_file()) else {
        brain_testutil::skip("Qwen/Qwen3.8-27B-FP8 not in the model store (brain pull Qwen/Qwen3.8-27B-FP8)");
        return;
    };
    let template = ChatTemplate::from_model_dir(&dir).expect("Qwen3.8 chat template");
    assert!(template.source().contains("reasoning_effort"), "the Qwen3.8 template reads reasoning_effort");
    let turns = json!([{"role": "system", "content": "You are a careful software agent."}, {"role": "user", "content": "Time: 2026-11-19T08:00:00Z\n\nFix it."}]);
    let request = Invocation::new().set("messages", json!(turns.to_string())).set("tools", json!(TOOLS)).set("template_flavor", json!("qwen3.8"));
    for effort in [None, Some("low"), Some("medium"), Some("xhigh")] {
        let inv = match effort {
            Some(e) => request.clone().set("reasoning_effort", json!(e)),
            None => request.clone(),
        };
        let served = qwen3::chat::render_prompt(&inv).expect("the request renders").text;
        let mut extra = BTreeMap::new();
        extra.insert("enable_thinking".to_string(), Value::from(true));
        if let Some(e) = effort {
            extra.insert("reasoning_effort".to_string(), Value::from(e));
        }
        let own = template.render(template.conversation(turns.as_array().unwrap()).unwrap(), Some(parse_json_ordered(TOOLS).unwrap()), true, &extra).expect("the template renders");
        assert_eq!(served, own, "reasoning_effort {effort:?}");
    }
}
