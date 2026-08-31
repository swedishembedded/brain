// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for on-device LLM inference --
// training, quantizing and serving models at the edge -- for its clients. If
// your team needs expertise in edge AI inference infrastructure then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Cross-validates the GENERIC Jinja-driven renderer (`data::chat_template`,
//! executes the checkpoint's own `chat_template` string) against
//! `data::qwen_chat::render` (hand-transcribed, per-generation Rust ports of
//! that same Jinja source) for identical inputs. Agreement on real
//! multi-turn/tool-call cases is strong evidence BOTH are correct -- the
//! generic engine gets to lean on `qwen_chat`'s already-scrutinized port as
//! an independent oracle, and any future model family (GLM, etc.) then only
//! needs `chat_template`, no new hand-port.
//!
//! Two template generations are cross-checked, matched to their hand-port
//! [`data::qwen_chat::TemplateFlavor`]:
//!
//! | template on disk                                                       | flavor                    | tests below                   |
//! |------------------------------------------------------------------------|---------------------------|-------------------------------|
//! | Qwen3-era (Qwen3-0.6B vintage)                                         | `TemplateFlavor::Qwen3`   | `matches_qwen_chat_*`         |
//! | Qwen3.8 (`reasoning_instructions` / `preserve_thinking` machinery)      | `TemplateFlavor::Qwen38`  | `matches_qwen_chat_qwen38_*`  |
//!
//! The generation is sniffed from the template source (the Qwen3.8 template
//! defines `reasoning_instructions`; the Qwen3-era one predates it), because
//! the directory resolution below may legitimately hold either. Every test
//! skips loudly when the checkpoint isn't fetched at all -- these guards used
//! to be gated on a private env var nothing set in CI, so the train/serve
//! prompt-skew surface silently skipped by default -- and when the directory
//! holds the OTHER generation's template, so a swapped dir can never silently
//! validate the wrong port.

use std::collections::BTreeMap;
use std::path::PathBuf;

use data::chat_template::{parse_json_ordered, ChatTemplate};
use data::qwen_chat::{self, ChatMessage as QcMessage, Role, TemplateFlavor, TemplateOpts, ToolCallMsg};
use minijinja::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Flavor {
    /// Qwen3-era chat template.
    Qwen3,
    /// Qwen3.8 chat template (`reasoning_instructions` / `preserve_thinking`).
    Qwen38,
}

fn qwen3_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("QWEN3_DIR") {
        return Some(PathBuf::from(d));
    }
    // The served Qwen3 chat model this cross-check guards (the same
    // tokenizer_config.json qwen3::chat renders from).
    let dir = PathBuf::from(brain_testutil::model_dir("Qwen/Qwen3-0.6B")?);
    dir.join("tokenizer_config.json").exists().then_some(dir)
}

/// Compile the template found at [`qwen3_dir`] and sniff which generation it
/// is: the Qwen3.8 template carries the `reasoning_instructions` machinery
/// (`reasoning_effort` -> system directive, `preserve_thinking`), which the
/// Qwen3-era template predates.
fn load_template() -> Option<(ChatTemplate, Flavor)> {
    let dir = qwen3_dir()?;
    let cfg_text = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text).ok()?;
    let src = cfg["chat_template"].as_str()?.to_string();
    let flavor = if src.contains("reasoning_instructions") {
        Flavor::Qwen38
    } else {
        Flavor::Qwen3
    };
    Some((
        ChatTemplate::compile(&src).expect("compile real Qwen chat_template"),
        flavor,
    ))
}

/// Skip loudly unless the loaded template is the generation these tests
/// cross-validate: a wrong-generation directory must never silently validate
/// the wrong hand-port.
macro_rules! want_flavor {
    ($flavor:expr, $want:expr, $what:expr) => {
        if $flavor != $want {
            brain_testutil::skip(concat!(
                $what,
                " template on disk, but these cross-checks need the ",
                stringify!($want),
                " one; point QWEN3_DIR at the matching tokenizer_config.json"
            ));
            return;
        }
    };
}

// ========================================================================
// Qwen3-era generation (`TemplateFlavor::Qwen3`)
// ========================================================================

#[test]
fn matches_qwen_chat_on_a_tool_call_conversation() {
    let Some((tmpl, flavor)) = load_template() else {
        brain_testutil::skip("Qwen3 tokenizer_config.json not found (set QWEN3_DIR or fetch Qwen/Qwen3-0.6B into the model store)");
        return;
    };
    want_flavor!(flavor, Flavor::Qwen3, "Qwen3.8");

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
    let expected = qwen_chat::render(&qc_msgs, &[], TemplateOpts { add_generation_prompt: false, enable_thinking: true , ..Default::default() }).expect("qwen_chat render");

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
    let Some((tmpl, flavor)) = load_template() else {
        brain_testutil::skip("Qwen3 tokenizer_config.json not found (set QWEN3_DIR or fetch Qwen/Qwen3-0.6B into the model store)");
        return;
    };
    want_flavor!(flavor, Flavor::Qwen3, "Qwen3.8");

    let tools_src = vec![r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}"#.to_string()];

    let qc_msgs = vec![QcMessage::system("sys prompt"), QcMessage::user("hi")];
    let expected =
        qwen_chat::render(&qc_msgs, &tools_src, TemplateOpts { add_generation_prompt: true, enable_thinking: true , ..Default::default() }).expect("qwen_chat render");

    let messages = parse_json_ordered(r#"[{"role":"system","content":"sys prompt"},{"role":"user","content":"hi"}]"#).unwrap();
    let tools = parse_json_ordered(&format!("[{}]", tools_src[0])).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(true));
    let got = tmpl.render(messages, Some(tools), true, &extra).expect("generic render");

    assert_eq!(got, expected, "generic Jinja engine diverges from qwen_chat::render on the tools-schema branch");
}

#[test]
fn matches_qwen_chat_with_enable_thinking_false_generation_prompt() {
    let Some((tmpl, flavor)) = load_template() else {
        brain_testutil::skip("Qwen3 tokenizer_config.json not found (set QWEN3_DIR or fetch Qwen/Qwen3-0.6B into the model store)");
        return;
    };
    want_flavor!(flavor, Flavor::Qwen3, "Qwen3.8");
    let qc_msgs = vec![QcMessage::user("hi")];
    let expected = qwen_chat::render_for_generation(&qc_msgs, &[], qwen_chat::TemplateOpts { enable_thinking: false, ..Default::default() }).expect("qwen_chat render");

    let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(false));
    let got = tmpl.render(messages, None, true, &extra).expect("generic render");

    assert_eq!(got, expected);
}

// ========================================================================
// Qwen3.8 generation (`TemplateFlavor::Qwen38`)
// ========================================================================

#[test]
fn matches_qwen_chat_qwen38_on_a_tool_call_conversation() {
    let Some((tmpl, flavor)) = load_template() else {
        brain_testutil::skip("Qwen3.8 tokenizer_config.json not found (set QWEN3_DIR to the qwen3.8 resources dir)");
        return;
    };
    want_flavor!(flavor, Flavor::Qwen38, "Qwen3");

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
    let expected = qwen_chat::render(&qc_msgs, &[], TemplateOpts {
        add_generation_prompt: false,
        enable_thinking: true,
        flavor: TemplateFlavor::Qwen38,
        ..Default::default()
    })
    .expect("qwen_chat render");

    // -------- generic Jinja engine side (same conversation, as JSON) --------
    // The Qwen3.8 template iterates `tool_call.arguments|items`, so the
    // generic side feeds arguments as a JSON OBJECT; the hand-port takes the
    // JSON-string form and parses it internally.
    let messages_json = r#"[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"What is 2+2, then check the weather in Paris."},
        {"role":"assistant","content":"Let me check that.","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"get_weather","arguments":{"location":"Paris"}}}
        ]},
        {"role":"tool","content":"18C, sunny"},
        {"role":"assistant","content":"2+2 is 4, and it's 18C and sunny in Paris."}
    ]"#;
    let messages = parse_json_ordered(messages_json).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(true));
    let got = tmpl.render(messages, None, false, &extra).expect("generic render");

    assert_eq!(got, expected, "generic Jinja engine diverges from the hand-ported qwen_chat::render (Qwen38)");
}

#[test]
fn matches_qwen_chat_qwen38_with_a_tools_schema_and_generation_prompt() {
    let Some((tmpl, flavor)) = load_template() else {
        brain_testutil::skip("Qwen3.8 tokenizer_config.json not found (set QWEN3_DIR to the qwen3.8 resources dir)");
        return;
    };
    want_flavor!(flavor, Flavor::Qwen38, "Qwen3");

    let tools_src = vec![r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}"#.to_string()];

    let qc_msgs = vec![QcMessage::system("sys prompt"), QcMessage::user("hi")];
    let expected = qwen_chat::render(&qc_msgs, &tools_src, TemplateOpts {
        add_generation_prompt: true,
        enable_thinking: false,
        flavor: TemplateFlavor::Qwen38,
        ..Default::default()
    })
    .expect("qwen_chat render");

    let messages = parse_json_ordered(r#"[{"role":"system","content":"sys prompt"},{"role":"user","content":"hi"}]"#).unwrap();
    let tools = parse_json_ordered(&format!("[{}]", tools_src[0])).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(false));
    let got = tmpl.render(messages, Some(tools), true, &extra).expect("generic render");

    assert_eq!(got, expected, "generic Jinja engine diverges from qwen_chat::render on the tools-schema branch (Qwen38)");
}

#[test]
fn matches_qwen_chat_qwen38_with_preserve_thinking_false() {
    let Some((tmpl, flavor)) = load_template() else {
        brain_testutil::skip("Qwen3.8 tokenizer_config.json not found (set QWEN3_DIR to the qwen3.8 resources dir)");
        return;
    };
    want_flavor!(flavor, Flavor::Qwen38, "Qwen3");

    // -------- qwen_chat (hand-ported) side --------
    // Reasoning on the FIRST assistant turn sits before the last real user
    // query, so `preserve_thinking=false` must strip it (the trailing turn,
    // after that query, keeps its -- empty -- think block either way).
    let qc_msgs = vec![
        QcMessage::user("q1"),
        QcMessage {
            role: Role::Assistant,
            content: "a1".into(),
            reasoning_content: Some("thinking one".into()),
            ..Default::default()
        },
        QcMessage::user("q2"),
        QcMessage::assistant("a2"),
    ];
    let expected = qwen_chat::render(&qc_msgs, &[], TemplateOpts {
        add_generation_prompt: false,
        enable_thinking: true,
        flavor: TemplateFlavor::Qwen38,
        preserve_thinking: Some(false),
        ..Default::default()
    })
    .expect("qwen_chat render");

    // -------- generic Jinja engine side (same conversation, as JSON) --------
    let messages_json = r#"[
        {"role":"user","content":"q1"},
        {"role":"assistant","content":"a1","reasoning_content":"thinking one"},
        {"role":"user","content":"q2"},
        {"role":"assistant","content":"a2"}
    ]"#;
    let messages = parse_json_ordered(messages_json).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(true));
    extra.insert("preserve_thinking".to_string(), Value::from(false));
    let got = tmpl.render(messages, None, false, &extra).expect("generic render");

    assert_eq!(got, expected, "generic Jinja engine diverges from qwen_chat::render on preserve_thinking=false (Qwen38)");
}

#[test]
fn matches_qwen_chat_qwen38_with_enable_thinking_false_generation_prompt() {
    let Some((tmpl, flavor)) = load_template() else {
        brain_testutil::skip("Qwen3.8 tokenizer_config.json not found (set QWEN3_DIR to the qwen3.8 resources dir)");
        return;
    };
    want_flavor!(flavor, Flavor::Qwen38, "Qwen3");
    let qc_msgs = vec![QcMessage::user("hi")];
    let expected = qwen_chat::render(&qc_msgs, &[], TemplateOpts {
        add_generation_prompt: true,
        enable_thinking: false,
        flavor: TemplateFlavor::Qwen38,
        ..Default::default()
    })
    .expect("qwen_chat render");

    let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
    let mut extra = BTreeMap::new();
    extra.insert("enable_thinking".to_string(), Value::from(false));
    let got = tmpl.render(messages, None, true, &extra).expect("generic render");

    assert_eq!(got, expected);
}
