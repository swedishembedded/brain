// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cross-validates the GENERIC Jinja-driven renderer (`data::chat_template`,
//! executes the checkpoint's own `chat_template` string) against
//! `data::qwen_chat::render` (a hand-transcribed, Qwen3-specific Rust port of
//! that exact same Jinja source) for identical inputs. Agreement on real
//! multi-turn/tool-call cases is strong evidence BOTH are correct -- the
//! generic engine gets to lean on `qwen_chat`'s already-scrutinized port as
//! an independent oracle, and any future model family (GLM, etc.) then only
//! needs `chat_template`, no new hand-port.
//!
//! Gated on `QWEN3_DIR` (skips loudly if unset) -- needs the real Qwen3
//! `tokenizer_config.json` to source the actual `chat_template` string from.

use std::collections::BTreeMap;
use std::path::PathBuf;

use data::chat_template::{parse_json_ordered, ChatTemplate};
use data::qwen_chat::{self, ChatMessage as QcMessage, TemplateOpts, ToolCallMsg};
use minijinja::Value;

fn qwen3_dir() -> Option<PathBuf> {
    std::env::var("QWEN3_DIR").ok().map(PathBuf::from)
}

fn load_template() -> Option<ChatTemplate> {
    let dir = qwen3_dir()?;
    let cfg_text = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text).ok()?;
    let src = cfg["chat_template"].as_str()?.to_string();
    Some(ChatTemplate::compile(&src).expect("compile real Qwen3 chat_template"))
}

#[test]
fn matches_qwen_chat_on_a_tool_call_conversation() {
    let Some(tmpl) = load_template() else {
        eprintln!("QWEN3_DIR unset; skipping");
        return;
    };

    // -------- qwen_chat (hand-ported) side --------
    let qc_msgs = vec![
        QcMessage::system("You are a helpful assistant."),
        QcMessage::user("What is 2+2, then check the weather in Paris."),
        QcMessage::assistant("Let me check that.").with_tool_calls(vec![ToolCallMsg {
            id: "c1".into(),
            name: "get_weather".into(),
            arguments: r#"{"location": "Paris"}"#.into(),
        }]),
        QcMessage::tool("18C, sunny"),
        QcMessage::assistant("2+2 is 4, and it's 18C and sunny in Paris."),
    ];
    let expected = qwen_chat::render(&qc_msgs, &[], TemplateOpts { add_generation_prompt: false, enable_thinking: true }).expect("qwen_chat render");

    // -------- generic Jinja engine side (same conversation, as JSON) --------
    let messages_json = r#"[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"What is 2+2, then check the weather in Paris."},
        {"role":"assistant","content":"Let me check that.","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"location\": \"Paris\"}"}}
        ]},
        {"role":"tool","content":"18C, sunny"},
        {"role":"assistant","content":"2+2 is 4, and it's 18C and sunny in Paris."}
    ]"#;
    let messages = parse_json_ordered(messages_json).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(true));
    let got = tmpl.render(messages, None, false, &extra).expect("generic render");

    assert_eq!(got, expected, "generic Jinja engine diverges from the hand-ported qwen_chat::render");
}

#[test]
fn matches_qwen_chat_with_a_tools_schema_and_generation_prompt() {
    let Some(tmpl) = load_template() else {
        eprintln!("QWEN3_DIR unset; skipping");
        return;
    };

    let tools_src = vec![r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}"#.to_string()];

    let qc_msgs = vec![QcMessage::system("sys prompt"), QcMessage::user("hi")];
    let expected =
        qwen_chat::render(&qc_msgs, &tools_src, TemplateOpts { add_generation_prompt: true, enable_thinking: true }).expect("qwen_chat render");

    let messages = parse_json_ordered(r#"[{"role":"system","content":"sys prompt"},{"role":"user","content":"hi"}]"#).unwrap();
    let tools = parse_json_ordered(&format!("[{}]", tools_src[0])).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(true));
    let got = tmpl.render(messages, Some(tools), true, &extra).expect("generic render");

    assert_eq!(got, expected, "generic Jinja engine diverges from qwen_chat::render on the tools-schema branch");
}

#[test]
fn matches_qwen_chat_with_enable_thinking_false_generation_prompt() {
    let Some(tmpl) = load_template() else {
        eprintln!("QWEN3_DIR unset; skipping");
        return;
    };
    let qc_msgs = vec![QcMessage::user("hi")];
    let expected = qwen_chat::render_for_generation(&qc_msgs, &[], false).expect("qwen_chat render");

    let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(false));
    let got = tmpl.render(messages, None, true, &extra).expect("generic render");

    assert_eq!(got, expected);
}
