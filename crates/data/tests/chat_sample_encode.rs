// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Golden test: `ChatSample::encode`'s ids and mask boundaries, against a real
//! Qwen3 tokenizer. Gated on `QWEN3_DIR` (a real Qwen3 checkpoint dir with
//! `tokenizer.json`) — skips loudly rather than failing when unset, matching
//! `crates/qwen/tests/integration_qwen3.rs`'s convention.
//!
//! "Golden" here means: derive each message's expected span by calling the
//! SAME `QwenBpe::frame_message` + `encode` `ChatSample::encode` itself calls,
//! then assert the concatenation and per-message mask value exactly — an off-
//! by-one mask boundary is the classic silent SFT bug finite differences
//! cannot see (docs/lessons.md).

use std::path::PathBuf;

use data::chat::{ChatMessage, ChatSample, ToolCall};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

fn qwen3_dir() -> Option<PathBuf> {
    std::env::var("QWEN3_DIR").ok().map(PathBuf::from)
}

#[test]
fn encode_matches_per_message_reference_spans_and_mask_boundaries() {
    let Some(dir) = qwen3_dir() else {
        eprintln!("QWEN3_DIR unset; skipping (needs a real Qwen3 tokenizer.json)");
        return;
    };
    let tok = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).expect("load tokenizer");

    let sample = ChatSample {
        messages: vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("What is 2+2, then check the weather in Paris."),
            ChatMessage::assistant_tool_calls(
                "Let me check that.",
                vec![ToolCall { name: "get_weather".into(), arguments: serde_json::json!({"location": "Paris"}) }],
                true,
            ),
            ChatMessage::tool_result("18C, sunny"),
            ChatMessage::assistant("2+2 is 4, and it's 18C and sunny in Paris.", true),
        ],
    };

    let (ids, mask) = sample.encode(&tok);
    assert_eq!(ids.len(), mask.len());

    // Reference: hand-written literal framed text per message -- deliberately
    // NOT built by calling ChatMessage::rendered_content or
    // QwenBpe::frame_message, so a bug in either of those implementations
    // (wrong tool_call formatting, wrong <|im_start|>/<|im_end|> framing)
    // would show up as a mismatch here instead of being invisible because
    // both sides share the same buggy code.
    let expected_texts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        "<|im_start|>user\nWhat is 2+2, then check the weather in Paris.<|im_end|>\n",
        "<|im_start|>assistant\nLet me check that.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"Paris\"}}\n</tool_call><|im_end|>\n",
        "<|im_start|>user\n<tool_response>\n18C, sunny\n</tool_response><|im_end|>\n",
        "<|im_start|>assistant\n2+2 is 4, and it's 18C and sunny in Paris.<|im_end|>\n",
    ];
    let expected_train = [false, false, true, false, true];

    let mut expected_ids: Vec<u32> = Vec::new();
    let mut expected_mask: Vec<bool> = Vec::new();
    for (text, &train) in expected_texts.iter().zip(&expected_train) {
        let tid = tok.encode(text);
        expected_ids.extend_from_slice(&tid);
        expected_mask.extend(std::iter::repeat_n(train, tid.len()));
    }
    expected_ids.push(data::chat::ENDOFTEXT);
    expected_mask.push(false);

    assert_eq!(ids, expected_ids, "encode() ids diverge from the hand-written reference rendering");
    assert_eq!(mask, expected_mask, "encode() mask diverges from the hand-written reference rendering");

    // The masked span is a strict, non-empty subset -- catches a mask that's
    // accidentally all-true or all-false (either passes a shape check).
    assert!(mask.iter().any(|&m| m), "nothing was marked trainable");
    assert!(mask.iter().any(|&m| !m), "nothing was masked as context");
}
