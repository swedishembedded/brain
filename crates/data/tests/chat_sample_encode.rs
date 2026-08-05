// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Golden test: `ChatSample::encode`'s ids and mask, against a real Qwen3
//! tokenizer AND the real `chat_template` from `tokenizer_config.json`.
//! Gated on `QWEN3_DIR` — skips loudly rather than failing when unset,
//! matching `crates/qwen/tests/integration_qwen3.rs`'s convention.
//!
//! The independent oracle for the expected TEXT is `qwen_chat::render` (a
//! hand-transcribed, already-scrutinized Qwen3-specific port), built from
//! the SAME conversation via `qwen_chat`'s own types -- not by calling
//! anything `ChatSample::encode` itself calls, so a bug in the generic
//! engine's rendering would show up as a text mismatch here, not be
//! invisible because both sides share the same code
//! (`chat_template_cross_check.rs` covers that agreement directly; this
//! test covers the ChatSample -> tokens -> mask integration on top of it).

use std::path::PathBuf;

use data::chat::{ChatMessage, ChatSample, ToolCall};
use data::chat_template::ChatTemplate;
use data::qwen_chat::{self, ChatMessage as QcMessage, TemplateOpts};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

fn qwen3_dir() -> Option<PathBuf> {
    std::env::var("QWEN3_DIR").ok().map(PathBuf::from)
}

fn load_template(dir: &std::path::Path) -> ChatTemplate {
    let cfg_text = std::fs::read_to_string(dir.join("tokenizer_config.json")).expect("read tokenizer_config.json");
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text).expect("parse tokenizer_config.json");
    let src = cfg["chat_template"].as_str().expect("chat_template field");
    ChatTemplate::compile(src).expect("compile chat_template")
}

#[test]
fn encode_matches_the_qwen_chat_oracle_text_with_correct_mask_boundaries() {
    let Some(dir) = qwen3_dir() else {
        eprintln!("QWEN3_DIR unset; skipping (needs a real Qwen3 tokenizer.json + tokenizer_config.json)");
        return;
    };
    let tok = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).expect("load tokenizer");
    let tmpl = load_template(&dir);

    // A single decision (one user turn, one trainable assistant answer) --
    // the common case, and one that does NOT hit the prefix-stability
    // hazard `render_with_message_boundaries` documents (see the other test
    // in this file for that hazard, and why it fails loudly instead of
    // silently mismasking).
    let sample = ChatSample {
        messages: vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("What is 2+2?"),
            ChatMessage::assistant("2+2 is 4.", true),
        ],
        tools: Vec::new(),
    };

    let (ids, mask) = sample.encode(&tok, &tmpl).expect("encode");
    assert_eq!(ids.len(), mask.len());

    let qc_msgs = vec![QcMessage::system("You are a helpful assistant."), QcMessage::user("What is 2+2?"), QcMessage::assistant("2+2 is 4.")];
    let expected_text =
        qwen_chat::render(&qc_msgs, &[], TemplateOpts { add_generation_prompt: false, enable_thinking: true }).expect("qwen_chat render");
    let expected_text_with_eot = format!("{expected_text}<|endoftext|>");

    let decoded = tok.decode(&ids);
    assert_eq!(decoded, expected_text_with_eot, "encode() token ids do not decode back to the oracle rendering");

    let trainable_text: String =
        ids.iter().zip(&mask).filter(|(_, &m)| m).map(|(&id, _)| tok.decode(&[id])).collect::<Vec<_>>().join("");
    assert!(trainable_text.contains("2+2 is 4."), "trainable span missing the assistant turn: {trainable_text:?}");
    assert!(!trainable_text.contains("What is 2+2"), "the user turn leaked into the trainable span: {trainable_text:?}");
    assert!(!trainable_text.contains("You are a helpful"), "the system turn leaked into the trainable span: {trainable_text:?}");
    assert!(mask.iter().any(|&m| m), "nothing was marked trainable");
    assert!(mask.iter().any(|&m| !m), "nothing was masked as context");
}

#[test]
fn encode_fails_loudly_rather_than_mismask_a_tool_call_then_final_answer_conversation() {
    // KNOWN LIMITATION (docs/guides/training.md "Known gaps"): a tool-call
    // turn followed by a tool result and a final answer is TWO assistant
    // turns after the last real user turn -- Qwen3's own template inserts
    // an empty <think></think> block only for the LITERALLY LAST assistant
    // turn, so a truncated prefix (where the tool-call turn looks last)
    // renders it differently than the true conversation does.
    // ChatSample::encode must propagate that as an error, not silently
    // produce a wrong mask boundary.
    let Some(dir) = qwen3_dir() else {
        eprintln!("QWEN3_DIR unset; skipping (needs a real Qwen3 tokenizer.json + tokenizer_config.json)");
        return;
    };
    let tok = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).expect("load tokenizer");
    let tmpl = load_template(&dir);

    let sample = ChatSample {
        messages: vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("What is 2+2, then check the weather in Paris."),
            ChatMessage::assistant_tool_calls(
                "Let me check that.",
                vec![ToolCall { id: Some("c1".into()), name: "get_weather".into(), arguments: r#"{"location": "Paris"}"#.into() }],
                true,
            ),
            ChatMessage::tool_result("18C, sunny"),
            ChatMessage::assistant("2+2 is 4, and it's 18C and sunny in Paris.", true),
        ],
        tools: Vec::new(),
    };

    let err = sample.encode(&tok, &tmpl).unwrap_err();
    assert!(err.to_string().contains("not prefix-stable"), "got: {err}");
}
