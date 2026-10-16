// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for exact loss masking when
// fine-tuning chat models on long tool-using conversations for its clients. If
// your team needs expertise in edge AI training infrastructure then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A conversation with several supervised assistant turns trained as ONE
//! sequence ([`ChatSample::answers_packed`]) supervises exactly the text that
//! training each turn alone ([`ChatSample::answers_as_asked`]) supervises, and
//! a turn that is not marked supervised stays context. The template here
//! renders the last assistant turn with an empty reasoning block and an
//! earlier one without, as Qwen3's does, which is what makes a whole-
//! conversation boundary probe refuse such records.

use data::chat::{ChatMessage, ChatSample, ToolCall};
use data::chat_template::ChatTemplate;
use data::tokenizer::{CharTokenizer, Tokenizer};

const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n\
{% if m.role == 'assistant' and loop.last %}<think>\n\n</think>\n\n{% endif %}\
{{ m.content }}{% for c in m.tool_calls or [] %}<tool_call>{{ c.function.name }}</tool_call>{% endfor %}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}";

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_corpus(&(32u8..127).map(char::from).chain(['\n']).collect::<String>())
}

fn call(name: &str) -> Vec<ToolCall> {
    vec![ToolCall { id: Some("c".into()), name: name.into(), arguments: "{}".into() }]
}

/// Three actions: two tool calls and a final answer, with a tool result after each call.
fn trajectory(train: [bool; 3]) -> ChatSample {
    ChatSample {
        messages: vec![
            ChatMessage::system("sys"),
            ChatMessage::user("fix it"),
            ChatMessage::assistant_tool_calls("look", call("read"), train[0]),
            ChatMessage::tool_result("file text"),
            ChatMessage::assistant_tool_calls("edit", call("write"), train[1]),
            ChatMessage::tool_result("ok"),
            ChatMessage::assistant("done", train[2]),
        ],
        tools: Vec::new(),
        rendered: None,
    }
}

/// The text of each run of consecutive supervised tokens, in order.
fn supervised_runs(sample: &ChatSample, tmpl: &ChatTemplate) -> Vec<String> {
    let tok = tokenizer();
    let (ids, mask) = sample.encode(&tok, tmpl).expect("encode");
    assert_eq!(ids.len(), mask.len());
    let mut runs: Vec<Vec<u32>> = Vec::new();
    let mut previous = false;
    for (&id, &m) in ids.iter().zip(&mask) {
        if m && !previous {
            runs.push(Vec::new());
        }
        if m {
            runs.last_mut().expect("a run is open").push(id);
        }
        previous = m;
    }
    runs.iter().map(|r| tok.decode(r)).collect()
}

fn token_count(sample: &ChatSample, tmpl: &ChatTemplate) -> usize {
    sample.encode(&tokenizer(), tmpl).expect("encode").0.len()
}

/// What training the turns one at a time supervises and costs.
fn unpacked(record: &ChatSample, tmpl: &ChatTemplate, thinking: bool) -> (Vec<String>, usize) {
    let each = record.answers_as_asked(tmpl, thinking).expect("render").expect("assistant answers");
    (each.iter().flat_map(|s| supervised_runs(s, tmpl)).collect(), each.iter().map(|s| token_count(s, tmpl)).sum())
}

#[test]
fn packed_supervises_the_union_of_the_unpacked_answers() {
    let tmpl = ChatTemplate::compile(TEMPLATE).unwrap();
    for thinking in [true, false] {
        let record = trajectory([true; 3]);
        let (expected, unpacked_tokens) = unpacked(&record, &tmpl, thinking);
        assert_eq!(expected.len(), 3);

        let packed = record.answers_packed(&tmpl, thinking).expect("render").expect("assistant answers");
        assert_eq!(supervised_runs(&packed, &tmpl), expected, "thinking={thinking}");
        assert!(token_count(&packed, &tmpl) < unpacked_tokens, "one sequence is shorter than three growing prefixes");
    }
}

#[test]
fn an_action_not_marked_supervised_stays_context() {
    let tmpl = ChatTemplate::compile(TEMPLATE).unwrap();
    let record = trajectory([true, false, true]);
    let (expected, _) = unpacked(&record, &tmpl, true);
    assert_eq!(expected.len(), 2);

    let packed = record.answers_packed(&tmpl, true).expect("render").expect("assistant answers");
    let runs = supervised_runs(&packed, &tmpl);
    assert_eq!(runs, expected);
    assert!(runs.iter().all(|r| !r.contains("edit")), "the unsupervised action is not a target: {runs:?}");

    let (ids, _) = packed.encode(&tokenizer(), &tmpl).unwrap();
    assert!(tokenizer().decode(&ids).contains("edit"), "it is still in the context of the later action");
}

#[test]
fn a_record_that_supervises_a_prompt_turn_or_nothing_is_not_packed() {
    let tmpl = ChatTemplate::compile(TEMPLATE).unwrap();
    assert!(trajectory([false; 3]).answers_packed(&tmpl, true).unwrap().is_none());
    let mut record = trajectory([true; 3]);
    record.messages[1].train = true;
    assert!(record.answers_packed(&tmpl, true).unwrap().is_none());
}
