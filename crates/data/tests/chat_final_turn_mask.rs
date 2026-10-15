// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for exact loss masking when
// fine-tuning chat models on tool-using conversations for its clients. If your
// team needs expertise in edge AI training infrastructure then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The loss mask of a tool conversation whose only supervised turn is its last
//! one, under the real Qwen3 chat template (the model store's
//! `tokenizer_config.json`; no GPU). Qwen3's template renders an assistant turn
//! differently as the last message than in the middle of a conversation, so the
//! whole-conversation boundary probe refuses such a record; the final-turn mode
//! needs one boundary only. Each test skips loudly when the checkpoint is not
//! fetched.

use data::chat::{ChatMessage, ChatSample, ToolCall};
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

const THINK_EMPTY: &str = "<think>\n\n</think>\n\n";

fn qwen3() -> Option<(QwenBpe, ChatTemplate)> {
    let dir = std::path::PathBuf::from(brain_testutil::model_dir("Qwen/Qwen3-0.6B")?);
    if !dir.join("tokenizer_config.json").exists() || !dir.join("tokenizer.json").exists() {
        brain_testutil::skip("Qwen/Qwen3-0.6B tokenizer files are not in the model store (`brain fetch Qwen/Qwen3-0.6B`)");
        return None;
    }
    let tok = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).expect("load tokenizer");
    Some((tok, ChatTemplate::from_model_dir(&dir).expect("load template")))
}

fn tool_conversation(final_train: bool) -> ChatSample {
    ChatSample {
        messages: vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("What is the weather in Paris?"),
            ChatMessage::assistant_tool_calls(
                "Let me check that.",
                vec![ToolCall { id: Some("c1".into()), name: "get_weather".into(), arguments: r#"{"location": "Paris"}"#.into() }],
                false,
            ),
            ChatMessage::tool_result("18C, sunny"),
            ChatMessage::assistant("It is 18C and sunny in Paris.", final_train),
        ],
        tools: Vec::new(),
        rendered: None,
    }
}

/// The text of the supervised tokens of `sample` encoded for `tmpl`.
fn supervised_text(tok: &QwenBpe, tmpl: &ChatTemplate, sample: &ChatSample) -> String {
    let (ids, mask) = sample.encode(tok, tmpl).expect("encode");
    assert_eq!(ids.len(), mask.len());
    let kept: Vec<u32> = ids.iter().zip(&mask).filter(|(_, &m)| m).map(|(&id, _)| id).collect();
    tok.decode(&kept)
}

#[test]
fn final_turn_mask_of_a_tool_conversation_is_exact_under_qwen3_template() {
    let Some((tok, tmpl)) = qwen3() else { return };
    let record = tool_conversation(true);

    // The whole-conversation probe refuses this record: the tool-call turn
    // looks like the last message when the conversation is cut after it.
    let refused = record.encode(&tok, &tmpl).unwrap_err();
    assert!(refused.to_string().contains("not prefix-stable"), "got: {refused}");

    let sample = record.final_turn_as_asked(&tmpl, true).expect("render").expect("a record supervising only its last turn");
    assert_eq!(supervised_text(&tok, &tmpl, &sample), format!("{THINK_EMPTY}It is 18C and sunny in Paris.<|im_end|>\n"));

    // Everything before the supervised span is the context the model is asked from.
    let (ids, mask) = sample.encode(&tok, &tmpl).unwrap();
    let context: Vec<u32> = ids.iter().zip(&mask).filter(|(_, &m)| !m).map(|(&id, _)| id).collect();
    let context = tok.decode(&context);
    assert!(context.starts_with("<|im_start|>system\nYou are a helpful assistant."), "got: {context:?}");
    assert!(context.contains("<tool_call>") && context.contains("<tool_response>\n18C, sunny"), "got: {context:?}");
    assert!(context.ends_with("<|im_start|>assistant\n"), "got: {context:?}");
}

#[test]
fn a_record_supervising_two_assistant_turns_is_still_refused() {
    let Some((tok, tmpl)) = qwen3() else { return };
    let mut record = tool_conversation(true);
    record.messages[2].train = true;

    assert!(record.final_turn_as_asked(&tmpl, true).expect("render").is_none(), "only a record supervising its last message alone has a final-turn mask");
    let refused = record.encode(&tok, &tmpl).unwrap_err();
    assert!(refused.to_string().contains("not prefix-stable"), "got: {refused}");
}

#[test]
fn no_think_prefix_is_context_not_target() {
    let Some((tok, tmpl)) = qwen3() else { return };
    let sample = tool_conversation(true).final_turn_as_asked(&tmpl, false).expect("render").expect("a record supervising only its last turn");

    assert_eq!(supervised_text(&tok, &tmpl, &sample), "It is 18C and sunny in Paris.<|im_end|>\n");

    // The prompt is byte-for-byte what inference sends for the context.
    let (ids, mask) = sample.encode(&tok, &tmpl).unwrap();
    let context: Vec<u32> = ids.iter().zip(&mask).filter(|(_, &m)| !m).map(|(&id, _)| id).collect();
    assert!(tok.decode(&context).ends_with(&format!("<|im_start|>assistant\n{THINK_EMPTY}")));
}
