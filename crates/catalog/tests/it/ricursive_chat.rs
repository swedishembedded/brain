// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that a served model
// tokenizes and prompts exactly like its published reference for its clients.
// If your team needs expertise in tokenizer and chat-template parity then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Ripple-65B's tokenizer and chat template through brain's shared machinery, against the references that
//! define them: the `tokenizers` library for the tokenizer, `transformers`' `apply_chat_template` for the
//! template (`tools/goldens/ripple_chat_dump.py` writes the golden from the published files).
//!
//! The tokenizer is the release's own `tokenizer.json` read by `data::qwen_tokenizer::QwenBpe`; the template is
//! the release's own `chat_template.jinja` compiled by `data::chat_template::ChatTemplate` and rendered by
//! `qwen3::chat::render_prompt_as`, the path every served model with its own template takes. Nothing is
//! transcribed by hand. Skips by name when the golden or the files are absent (`testdata/ricursive/
//! ripple-65b-partial/`).

use capability::Invocation;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::chat::{render_prompt_as, ChatFormat};
use serde_json::{json, Value};

const DIR: &str = "ricursive/ripple-65b-partial";

fn golden() -> Option<(std::path::PathBuf, Value)> {
    let dir = brain_testutil::testdata(DIR);
    let dir = std::path::PathBuf::from(dir);
    let Ok(text) = std::fs::read_to_string(dir.join("chat_golden.json")) else {
        brain_testutil::skip("no Ripple chat golden (python -I tools/goldens/ripple_chat_dump.py MODEL_DIR testdata/ricursive/ripple-65b-partial/chat_golden.json)");
        return None;
    };
    if !dir.join("tokenizer.json").exists() || !dir.join("chat_template.jinja").exists() {
        brain_testutil::skip("no Ripple tokenizer.json or chat_template.jinja under testdata/ricursive/ripple-65b-partial");
        return None;
    }
    Some((dir, serde_json::from_str(&text).expect("golden JSON")))
}

#[test]
fn the_release_tokenizer_encodes_and_decodes_like_the_tokenizers_library() {
    let Some((dir, golden)) = golden() else { return };
    let tok = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).expect("Ripple's tokenizer.json");
    let cases = golden["tokenizer"].as_array().unwrap();
    assert!(cases.len() >= 15, "the sample covers code, unicode, whitespace runs and special tokens");
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.encode(text), want, "encode {text:?}");
        assert_eq!(tok.decode(&want), case["decoded"].as_str().unwrap(), "decode of the ids of {text:?}");
    }
    // The ids the release declares: bos `<|endoftext|>`, eos `<|im_end|>`, pad `<|pad|>`.
    for (name, id) in [("bos_token_id", "<|endoftext|>"), ("pad_token_id", "<|pad|>")] {
        assert_eq!(tok.special_id(id), golden[name].as_u64().map(|v| v as u32), "{name}");
    }
    assert_eq!(tok.special_id("<|im_end|>"), Some(100265));
}

/// A request for `case` as the OpenAI surface hands it to a model: messages and tools as the client's JSON text.
fn request(case: &Value) -> Invocation {
    let mut inv = Invocation::new().set("messages", case["messages_json"].clone());
    if !case["tools_json"].is_null() {
        inv = inv.set("tools", case["tools_json"].clone());
    }
    if case["kwargs_json"] != "{}" {
        inv = inv.set("chat_template_kwargs", case["kwargs_json"].clone());
    }
    inv
}

#[test]
fn rendered_prompts_are_byte_identical_to_apply_chat_template() {
    let Some((dir, golden)) = golden() else { return };
    let format = ChatFormat::own_template(&dir).expect("Ripple's chat_template.jinja");
    let cases = golden["templates"].as_array().unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let want = case["rendered"].as_str().unwrap();
        let rendered = render_prompt_as(&format, &request(case)).unwrap_or_else(|e| panic!("{name}: {e}"));
        if case["add_generation_prompt"].as_bool().unwrap() {
            assert_eq!(rendered.text, want, "{name}");
        } else {
            // A served request always asks for the generation prompt; the reference without it is the same
            // text minus that one marker, which the template renders last.
            assert_eq!(rendered.text.strip_suffix("<|im_start|>assistant\n"), Some(want), "{name}");
        }
        assert!(!rendered.thinking_open, "{name}: the template does not open a think block for the model");
    }
    assert!(cases.len() >= 12);
}

/// What the model generates for an assistant turn (everything after the generation prompt, before `<|im_end|>`).
const REPLIES: [&str; 7] = [
    "<think>\nAdd them.\n</think>\n4",
    "<think>\nNeed a tool.\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Zürich\", \"unit\": \"celsius\"}}\n</tool_call>",
    "<think>\nNeed a tool.\n</think>\nChecking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Zürich\"}}\n</tool_call>",
    "<think>\nBoth.\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Zürich\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>",
    "Plain answer without thinking.",
    "<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"ls -la\"}}\n</tool_call>",
    "<think>\nmulti\nline\n</think>\nline one\nline two",
];

/// The OpenAI message a server would send back for `outcome`.
fn assistant_message(outcome: &capability::Outcome) -> Value {
    let text = |key: &str| outcome.outputs.get(key).and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let mut message = json!({"role": "assistant", "content": text("text")});
    if !text("reasoning_content").is_empty() {
        message["reasoning_content"] = json!(text("reasoning_content"));
    }
    if let Some(calls) = outcome.outputs.get("tool_calls").and_then(|v| v.as_str()) {
        let calls: Vec<Value> = serde_json::from_str(calls).unwrap();
        message["tool_calls"] = calls.iter().map(|c| json!({"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}})).collect();
    }
    message
}

#[test]
fn a_reply_parsed_and_sent_back_unchanged_renders_to_the_tokens_the_model_generated() {
    let Some((dir, golden)) = golden() else { return };
    let format = ChatFormat::own_template(&dir).expect("Ripple's chat_template.jinja");
    let tools = golden["templates"].as_array().unwrap().iter().find(|c| c["name"] == "tools").unwrap()["tools_json"].clone();
    let user = json!([{"role": "user", "content": "go"}]);
    let ask = Invocation::new().set("messages", json!(user.to_string())).set("tools", tools.clone());
    let prompt = render_prompt_as(&format, &ask).unwrap();
    for reply in REPLIES {
        let outcome = qwen3::chat::parse_reply_as(&format, &ask, reply).unwrap_or_else(|e| panic!("{reply:?}: {e}"));
        assert!(outcome.outputs.get("scan_anomalies").is_none(), "{reply:?}: {:?}", outcome.outputs.get("scan_anomalies"));
        let again = json!([user[0], assistant_message(&outcome)]);
        let sent_back = Invocation::new().set("messages", json!(again.to_string())).set("tools", tools.clone());
        let rendered = render_prompt_as(&format, &sent_back).unwrap().text;
        // The history renders the turn as the model wrote it, closed by `<|im_end|>`; the generation prompt follows.
        let want = format!("{}{reply}<|im_end|>\n<|im_start|>assistant\n", prompt.text);
        let at = rendered.chars().zip(want.chars()).take_while(|(a, b)| a == b).count();
        let tail = |s: &str| s.chars().skip(at.saturating_sub(20)).collect::<String>();
        assert_eq!(rendered, want, "{reply:?}: first difference near {:?} (rendered) vs {:?} (generated)", tail(&rendered), tail(&want));
    }
}

#[test]
fn replies_split_into_reasoning_content_and_calls_as_the_reference_parser_does() {
    let Some((dir, _)) = golden() else { return };
    let format = ChatFormat::own_template(&dir).unwrap();
    let ask = Invocation::new().set("messages", json!(json!([{"role": "user", "content": "go"}]).to_string()));
    let parse = |reply: &str| qwen3::chat::parse_reply_as(&format, &ask, reply).unwrap();
    let text = |o: &capability::Outcome, key: &str| o.outputs.get(key).and_then(|v| v.as_str()).unwrap_or_default().to_string();

    let o = parse(REPLIES[0]);
    assert_eq!((text(&o, "reasoning_content"), text(&o, "text")), ("Add them.".to_string(), "4".to_string()));
    assert_eq!(o.outputs["finish_reason"], "stop");

    let o = parse(REPLIES[3]);
    assert_eq!((text(&o, "reasoning_content"), text(&o, "text")), ("Both.".to_string(), String::new()));
    assert_eq!(o.outputs["finish_reason"], "tool_calls");
    let calls: Vec<Value> = serde_json::from_str(&text(&o, "tool_calls")).unwrap();
    assert_eq!(calls.iter().map(|c| c["name"].as_str().unwrap()).collect::<Vec<_>>(), ["get_weather", "execute_bash"]);
    assert_eq!(calls[0]["arguments"], "{\"city\": \"Zürich\"}");

    // Truncated inside the reasoning: everything so far is reasoning, there is no answer.
    let o = parse("<think>\nstill thinking");
    assert_eq!((text(&o, "reasoning_content"), text(&o, "text")), ("still thinking".to_string(), String::new()));
}
