// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for teaching models what their
// tools will answer, with exact and weighted loss masks, for its clients. If
// your team needs expertise in chat fine-tuning data pipelines then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A record that supervises messages other than assistant answers (a tool's
//! observations) trains exactly those messages' contents, placed where one
//! rendering of the conversation through the last of them puts them
//! ([`ChatSample::observations_in_place`]); and a message's `loss_weight`
//! reaches exactly its supervised tokens, whichever way the record is encoded.
//!
//! The template frames consecutive tool results inside one turn, as Qwen3's
//! does, so where a result's framing starts and ends depends on its
//! neighbours: no per-message boundary of the whole conversation exists, and
//! only the content itself can be located.

use data::chat::{ChatMessage, ChatSample, EncodedSplit, RenderOpts, ToolCall};
use data::chat_rollout::RecordRule;
use data::chat_template::ChatTemplate;
use data::tokenizer::{CharTokenizer, Tokenizer};

const TEMPLATE: &str = "{% for m in messages %}\
{% if m.role == 'tool' %}\
{% if loop.first or messages[loop.index0 - 1].role != 'tool' %}<|im_start|>user{% endif %}\
\n<tool_response>\n{{ m.content }}\n</tool_response>\
{% if loop.last or messages[loop.index0 + 1].role != 'tool' %}<|im_end|>\n{% endif %}\
{% else %}<|im_start|>{{ m.role }}\n\
{% if m.role == 'assistant' and loop.last %}<think>\n\n</think>\n\n{% endif %}\
{{ m.content }}{% for c in m.tool_calls or [] %}<tool_call>{{ c.function.name }}</tool_call>{% endfor %}<|im_end|>\n\
{% endif %}{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_corpus(&(32u8..127).map(char::from).chain(['\n']).collect::<String>())
}

fn call(id: &str, name: &str) -> Vec<ToolCall> {
    vec![ToolCall { id: Some(id.into()), name: name.into(), arguments: "{}".into() }]
}

fn result(id: &str, content: &str, train: bool) -> ChatMessage {
    ChatMessage { tool_call_id: Some(id.into()), train, ..ChatMessage::tool_result(content) }
}

/// Two calls, the first answered by two results; `observed` marks which results are supervised.
fn trajectory(observed: [bool; 3]) -> ChatSample {
    ChatSample {
        messages: vec![
            ChatMessage::system("sys"),
            ChatMessage::user("fix it"),
            ChatMessage::assistant_tool_calls("look", [call("a", "read"), call("b", "stat")].concat(), false),
            result("a", "file text", observed[0]),
            result("b", "size 9", observed[1]),
            ChatMessage::assistant_tool_calls("edit", call("c", "write"), false),
            result("c", "ok", observed[2]),
            ChatMessage::assistant("done", false),
        ],
        ..Default::default()
    }
}

/// Each run of consecutive supervised tokens, decoded, with its loss weight
/// (every token of a run carries the same one).
fn weighted_runs(sample: &ChatSample, tmpl: &ChatTemplate) -> Vec<(String, f32)> {
    let tok = tokenizer();
    let encoded = sample.encode_rollout(&tok, tmpl, RenderOpts::default()).expect("encode");
    let weights = encoded.loss_weights.clone().unwrap_or_else(|| vec![1.0; encoded.ids.len()]);
    assert_eq!(weights.len(), encoded.ids.len());
    let mut runs: Vec<(Vec<u32>, f32)> = Vec::new();
    let mut previous = false;
    for ((&id, &m), &w) in encoded.ids.iter().zip(&encoded.mask).zip(&weights) {
        if m && !previous {
            runs.push((Vec::new(), w));
        }
        if m {
            let run = runs.last_mut().expect("a run is open");
            assert_eq!(run.1, w, "one weight per supervised run");
            run.0.push(id);
        } else {
            assert_eq!(w, 1.0, "context carries no weight");
        }
        previous = m;
    }
    runs.into_iter().map(|(ids, w)| (tok.decode(&ids), w)).collect()
}

/// `messages` rendered through the template as a JSON conversation, without a generation prompt: the oracle.
fn rendered(tmpl: &ChatTemplate, messages: &[ChatMessage]) -> String {
    let turns: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let calls: Vec<serde_json::Value> = m.tool_calls.iter().map(|c| serde_json::json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}})).collect();
            let mut turn = serde_json::json!({"role": m.role, "content": m.content, "tool_calls": calls, "tool_call_id": m.tool_call_id});
            if let Some(reasoning) = &m.reasoning_content {
                turn["reasoning_content"] = serde_json::json!(reasoning);
            }
            turn
        })
        .collect();
    tmpl.render(tmpl.conversation(&turns).unwrap(), None, false, &Default::default()).unwrap()
}

fn runs(sample: &ChatSample, tmpl: &ChatTemplate) -> Vec<String> {
    weighted_runs(sample, tmpl).into_iter().map(|(text, _)| text).collect()
}

#[test]
fn supervised_tool_results_encode_as_exactly_their_contents_in_place() {
    let tmpl = ChatTemplate::compile(TEMPLATE).unwrap();
    let record = trajectory([true, false, true]);
    for thinking in [true, false] {
        let observed = record.observations_in_place(&tmpl, thinking).expect("render").expect("observations are supervised");
        assert_eq!(runs(&observed, &tmpl), ["file text", "ok"], "thinking={thinking}");

        // The context is the conversation rendered once through the last supervised result, cut after its content:
        // the earlier assistant turns as middle turns (no reasoning block), the other result as context.
        let (ids, _) = observed.encode(&tokenizer(), &tmpl).unwrap();
        let decoded = tokenizer().decode(&ids);
        let full = rendered(&tmpl, &record.messages[..=6]);
        assert_eq!(decoded, &full[..full.rfind("ok").unwrap() + "ok".len()]);
        assert!(!decoded.contains("<think>") && !decoded.contains("done"), "{decoded:?}");
        assert!(decoded.contains("size 9"), "an unsupervised result stays context");
    }
    assert!(trajectory([false; 3]).observations_in_place(&tmpl, true).unwrap().is_none(), "nothing supervised");
    let mut answers = trajectory([false; 3]);
    answers.messages[7].train = true;
    assert!(answers.observations_in_place(&tmpl, true).unwrap().is_none(), "assistant answers take their own paths");
}

#[test]
fn a_record_mixing_supervised_answers_and_observations_or_carrying_a_rollout_is_refused() {
    let tmpl = ChatTemplate::compile(TEMPLATE).unwrap();
    let mut mixed = trajectory([true, false, false]);
    mixed.messages[5].train = true;
    let err = mixed.observations_in_place(&tmpl, true).unwrap_err().to_string();
    assert!(err.contains("messages[3]") && err.contains("messages[5]") && err.contains("not both"), "{err}");

    let user = r#"{"role":"user","content":"q","train":false}"#;
    let call = r#"{"role":"assistant","content":"","tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{}"}}],"train":false}"#;
    let observed = r#"{"role":"tool","content":"text","tool_call_id":"a","train":true}"#;
    let rollout = ChatSample::from_json_line_with(&format!(r#"{{"reward": 1.0, "group": "g", "messages": [{user},{call},{observed}]}}"#), &data::chat::TextOnly::TRAINER).unwrap();
    let err = rollout.observations_in_place(&tmpl, true).unwrap_err().to_string();
    assert!(err.contains("rollout"), "{err}");
}

#[test]
fn a_template_that_does_not_place_the_content_verbatim_is_refused() {
    let tmpl = ChatTemplate::compile(&TEMPLATE.replace("{{ m.content }}\n</tool_response>", "{{ m.content | upper }}\n</tool_response>")).unwrap();
    let err = trajectory([true, false, false]).observations_in_place(&tmpl, true).unwrap_err().to_string();
    assert!(err.contains("verbatim"), "{err}");
}

/// A record line with `messages`, parsed.
fn parse(messages: &str) -> Result<ChatSample, data::chat_rollout::RecordError> {
    ChatSample::from_json_line_with(&format!(r#"{{"messages": [{messages}]}}"#), &data::chat::TextOnly::TRAINER)
}

#[test]
fn loss_weight_reaches_exactly_its_messages_supervised_tokens() {
    let tmpl = ChatTemplate::compile(TEMPLATE).unwrap();
    let user = r#"{"role":"user","content":"q","train":false}"#;
    let call = |id: &str, train: bool, weight: &str| format!(r#"{{"role":"assistant","content":"c{id}","tool_calls":[{{"id":"{id}","type":"function","function":{{"name":"read","arguments":"{{}}"}}}}],"train":{train}{weight}}}"#);
    let tool = |id: &str, text: &str, train: bool, weight: &str| format!(r#"{{"role":"tool","content":"{text}","tool_call_id":"{id}","train":{train}{weight}}}"#);
    let w = r#","loss_weight":0.3"#;

    // Observations: the weighted result's content carries 0.3, the other its default.
    let observations = parse(&[user.to_string(), call("a", false, ""), tool("a", "first", true, w), call("b", false, ""), tool("b", "second", true, "")].join(",")).unwrap();
    let observed = observations.observations_in_place(&tmpl, true).unwrap().unwrap();
    assert_eq!(weighted_runs(&observed, &tmpl), [("first".to_string(), 0.3), ("second".to_string(), 1.0)]);

    // Answers, packed and one at a time: each answer's rendering carries its own weight.
    let answers = parse(&[user.to_string(), call("a", true, w), tool("a", "first", false, ""), call("b", true, r#","loss_weight":2.5"#)].join(",")).unwrap();
    let packed = answers.answers_packed(&tmpl, true).unwrap().unwrap();
    assert_eq!(weighted_runs(&packed, &tmpl).iter().map(|(_, w)| *w).collect::<Vec<_>>(), [0.3, 2.5]);
    let each: Vec<f32> = answers.answers_as_asked(&tmpl, true).unwrap().unwrap().iter().flat_map(|s| weighted_runs(s, &tmpl)).map(|(_, w)| w).collect();
    assert_eq!(each, [0.3, 2.5]);

    // The whole record, encoded at its message boundaries.
    let whole = parse(&[user.to_string(), r#"{"role":"assistant","content":"plain","train":true,"loss_weight":0.3}"#.to_string()].join(",")).unwrap();
    let runs = weighted_runs(&whole, &tmpl);
    assert_eq!(runs.len(), 1);
    assert!(runs[0].0.contains("plain") && runs[0].1 == 0.3, "{runs:?}");

    // No weight, or a weight of one: no weights at all, and a prepared split writes no weight file.
    for unit in ["", r#","loss_weight":1.0"#] {
        let plain = parse(&[user.to_string(), format!(r#"{{"role":"assistant","content":"plain","train":true{unit}}}"#)].join(",")).unwrap();
        assert_eq!(plain.encode_rollout(&tokenizer(), &tmpl, RenderOpts::default()).unwrap().loss_weights, None);
    }
    let dir = tempfile::tempdir().unwrap();
    let split = |samples: &[&ChatSample]| {
        let mut split = EncodedSplit::default();
        for s in samples {
            split.push_rollout(&s.encode_rollout(&tokenizer(), &tmpl, RenderOpts::default()).unwrap(), &[], None);
        }
        split
    };
    let unweighted = parse(&[user.to_string(), r#"{"role":"assistant","content":"plain","train":true}"#.to_string()].join(",")).unwrap();
    data::chat::write_split(dir.path(), "train", &split(&[&unweighted])).unwrap();
    assert!(!dir.path().join("train.weight.bin").exists());
    data::chat::write_split(dir.path(), "train", &split(&[&unweighted, &whole])).unwrap();
    let written = data::binio::read_f32_bin(&dir.path().join("train.weight.bin")).unwrap();
    let encoded = [&unweighted, &whole].map(|s| s.encode_rollout(&tokenizer(), &tmpl, RenderOpts::default()).unwrap());
    let mask: Vec<bool> = encoded.iter().flat_map(|e| e.mask.clone()).collect();
    assert_eq!(written.len(), mask.len());
    let first = encoded[0].ids.len();
    assert!(written.iter().enumerate().all(|(i, &w)| w == if i >= first && mask[i] { 0.3 } else { 1.0 }), "{written:?}");
    data::chat::write_split(dir.path(), "train", &split(&[&unweighted])).unwrap();
    assert!(!dir.path().join("train.weight.bin").exists(), "a stale weight file is removed");
}

#[test]
fn loss_weight_off_a_supervised_message_or_out_of_range_is_refused() {
    let user = |extra: &str| format!(r#"{{"role":"user","content":"q","train":false{extra}}}"#);
    let answer = |extra: &str| format!(r#"{{"role":"assistant","content":"a","train":true{extra}}}"#);
    for (messages, field) in [
        ([user(r#","loss_weight":0.5"#), answer("")], "messages[0].loss_weight"),
        ([user(""), answer(r#","loss_weight":0"#)], "messages[1].loss_weight"),
        ([user(""), answer(r#","loss_weight":-1.0"#)], "messages[1].loss_weight"),
    ] {
        let err = parse(&messages.join(",")).expect_err(field);
        let fault = err.fault().cloned().unwrap_or_else(|| panic!("{field}: {err}"));
        assert_eq!((fault.rule, fault.field.as_str()), (RecordRule::LossWeight, field), "{err}");
    }
    let rollout = format!(r#"{{"reward": 1.0, "group": "g", "messages": [{},{}]}}"#, user(""), answer(r#","loss_weight":0.5"#));
    let err = ChatSample::from_json_line_with(&rollout, &data::chat::TextOnly::TRAINER).expect_err("a rollout's weights are its advantages");
    assert_eq!(err.fault().map(|f| f.rule), Some(RecordRule::LossWeight), "{err}");
}
