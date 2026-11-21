// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Inkling 2 Mini's own `chat_template.jinja`, rendered by `data::chat_template`, against
//! `transformers`' `apply_chat_template` on the same conversations
//! (`tools/goldens/inkling2_chat_dump.py`).
//!
//! The conversations cover what the template does with each shape: a system turn (the
//! thinking-effort line follows it, or leads when there is none), thinking history, a tool
//! declaration with a call and its result (sorted-key compact JSON), parts lists with images,
//! and shapes it refuses. A refusal in the reference is a refusal here, with the template's own
//! message. `reasoning_effort` goes through the typed [`ReasoningEffort`], named or numeric.

use std::collections::BTreeMap;

use data::chat_template::{parse_json_ordered, ChatTemplate, ReasoningEffort, Value, ValueKind};

fn effort_of(v: &Value) -> Result<ReasoningEffort, String> {
    match v.kind() {
        ValueKind::String => Ok(ReasoningEffort::Named(v.as_str().unwrap().to_string())),
        ValueKind::Number => ReasoningEffort::level(f64::try_from(v.clone()).unwrap()).map_err(|e| e.to_string()),
        _ => Err(format!("{v} is neither a name nor a number")),
    }
}

#[test]
fn inkling2_chat_template_renders_as_transformers_does() {
    let golden = brain_testutil::testdata_path("inkling2/chat.json");
    let Some(dir) = brain_testutil::model_dir("thinkingmachines/Inkling-2-Mini") else {
        brain_testutil::skip("Inkling-2-Mini: no models directory");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !golden.exists() || !dir.join("chat_template.jinja").exists() {
        brain_testutil::skip(&format!("Inkling-2-Mini: needs {} and the checkpoint's template", golden.display()));
        return;
    }
    // Parsed straight into template values: the messages must keep the golden's key order, as a
    // served request does, because the template serializes call arguments in the order given.
    let g = parse_json_ordered(&std::fs::read_to_string(&golden).unwrap()).unwrap();
    let template = ChatTemplate::from_model_dir(&dir).unwrap();
    let field = |case: &Value, key: &str| case.get_attr(key).unwrap();
    let mut compared = 0;
    for case in field(&g, "cases").try_iter().unwrap() {
        let name = field(&case, "name").as_str().unwrap().to_string();
        // The reference refuses an empty list in Python before the template runs.
        if name == "empty" {
            continue;
        }
        let tools = Some(field(&case, "tools")).filter(|t| !t.is_none());
        let generation_prompt = field(&case, "add_generation_prompt").is_true();
        let effort = field(&case, "reasoning_effort");
        let label = format!("{name} gen={generation_prompt} effort={effort}");
        let typed = if effort.is_none() { Ok(None) } else { effort_of(&effort).map(Some) };
        let rendered = typed.and_then(|e| {
            let mut extra = BTreeMap::new();
            if let Some(e) = e {
                e.apply(&mut extra);
            }
            template.render(field(&case, "messages"), tools, generation_prompt, &extra).map_err(|e| e.to_string())
        });
        match (field(&case, "text").as_str(), field(&case, "error").as_str()) {
            (Some(want), _) => assert_eq!(rendered.unwrap_or_else(|e| panic!("{label}: {e}")), want, "{label}"),
            (None, Some(refusal)) => {
                let message = refusal.split_once(": ").unwrap().1;
                let err = rendered.expect_err(&format!("{label}: the reference refuses this"));
                assert!(err.contains(message), "{label}: {err:?} does not carry {message:?}");
            }
            _ => unreachable!("a golden case has a render or an error"),
        }
        compared += 1;
    }
    assert!(compared > 60, "compared only {compared} cases");
}

/// What the template writes for an assistant turn reads back as the same reasoning, text and calls,
/// through the reader the template's own markers select.
#[test]
fn a_rendered_assistant_turn_scans_back_to_what_it_was() {
    use data::qwen_chat::{ChatScanner, TemplateFlavor};

    let Some(dir) = brain_testutil::model_dir("thinkingmachines/Inkling-2-Mini") else {
        brain_testutil::skip("Inkling-2-Mini: no models directory");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("chat_template.jinja").exists() {
        brain_testutil::skip("Inkling-2-Mini: needs the checkpoint's template");
        return;
    }
    let template = ChatTemplate::from_model_dir(&dir).unwrap();
    assert_eq!(template.reply_flavor(), Some(TemplateFlavor::TypedMessages), "the template's markers select the typed-message reader");

    let turns = [
        r#"{"role":"assistant","reasoning_content":"Need the tool.","content":"Let me check.","tool_calls":[{"id":"a","type":"function","function":{"name":"get_weather","arguments":{"units":"c","city":"Malmö","days":[1,2.5,null,true],"nested":{"b":1,"a":"å"}}}},{"id":"b","type":"function","function":{"name":"no_args"}}]}"#,
        r#"{"role":"assistant","content":"Just text with <|not_a_marker|> and a \"quote\"."}"#,
        r#"{"role":"assistant","reasoning_content":"only thoughts","content":""}"#,
    ];
    let user = r#"{"role":"user","content":"go"}"#;
    for turn in turns {
        let render = |conversation: &str| template.render(parse_json_ordered(&format!("[{conversation}]")).unwrap(), None, false, &BTreeMap::new()).unwrap();
        let prompt = render(user);
        let full = render(&format!("{user},{turn}"));
        let reply = full.strip_prefix(prompt.as_str()).expect("the assistant turn extends the prompt");

        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::TypedMessages);
        let mut events = Vec::new();
        scanner.push(reply, &mut events);
        scanner.finish(&mut events);
        assert_eq!(scanner.anomalies(), [], "{turn}");

        let want: serde_json::Value = serde_json::from_str(turn).unwrap();
        assert_eq!(scanner.reasoning(), want["reasoning_content"].as_str().unwrap_or(""), "{turn}");
        assert_eq!(scanner.content(), want["content"].as_str().unwrap_or(""), "{turn}");
        let calls = want["tool_calls"].as_array().cloned().unwrap_or_default();
        assert_eq!(scanner.tool_calls().len(), calls.len(), "{turn}");
        for (got, want) in scanner.tool_calls().iter().zip(calls) {
            assert_eq!(got.name, want["function"]["name"].as_str().unwrap());
            let args: serde_json::Value = serde_json::from_str(&got.arguments).unwrap();
            let sent = want["function"].get("arguments").cloned().unwrap_or(serde_json::json!({}));
            assert_eq!(args, sent, "{turn}");
        }
    }
}
