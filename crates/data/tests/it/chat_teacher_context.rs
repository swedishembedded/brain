// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for dense per-token credit on a language model's own rollouts, for its
// clients. If your team needs expertise in on-policy self-distillation with privileged information then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A chat record of a rollout may carry a `"teacher_context"`: extra messages the teacher is shown and the student
//! never saw. A prepared dataset keeps, beside the student's tokens, the teacher's rendering of the same conversation
//! (`<split>.tch.*`), whose supervised tokens are exactly the student's; a record whose two renderings do not agree on
//! the assistant tokens is refused.

use data::binio;
use data::chat::{prepare_chat_samples, ChatSample, RenderOpts};
use data::chat_rollout;
use data::chat_template::ChatTemplate;
use data::tokenizer::CharTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// An assistant answer that renders the number of messages before it: the two renderings of one conversation then
/// disagree on the assistant tokens.
const CONTEXT_DEPENDENT: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}{% if m.role == 'assistant' %}{{ loop.index }}{% endif %}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_corpus(&(32u8..127).map(char::from).chain(['\n']).collect::<String>())
}

fn template(source: &str) -> ChatTemplate {
    ChatTemplate::compile(source).unwrap()
}

fn record(extra: &str, teacher: &str) -> String {
    format!(
        r#"{{{extra} "teacher_context": {teacher}, "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true}}]}}"#
    )
}

const HINT: &str = r#"{"at": 1, "messages": [{"role": "user", "content": "the hidden test wants a"}]}"#;

fn read(line: &str) -> Result<Vec<ChatSample>, String> {
    ChatSample::from_json_line_with(line, &data::chat::TextOnly::TRAINER).and_then(|r| r.answers_as_asked(&template(TEMPLATE), true).map_err(|e| e.to_string()).map(|o| o.expect("assistant answers")))
}

fn prepare(samples: &[ChatSample], source: &str, dir: &std::path::Path) -> Result<(), String> {
    prepare_chat_samples(samples, samples, &tokenizer(), &template(source), RenderOpts::default(), 128, dir).map(drop).map_err(|e| e.to_string())
}

fn supervised(ids: &[u32], mask: &[bool]) -> Vec<u32> {
    ids.iter().zip(mask).filter(|(_, m)| **m).map(|(i, _)| *i).collect()
}

#[test]
fn a_prepared_dataset_keeps_the_teachers_rendering_whose_supervised_tokens_are_the_students() {
    let tmp = tempfile::tempdir().unwrap();
    let plain = r#"{"messages": [{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true}]}"#;
    let mut samples = read(&record("", HINT)).unwrap();
    samples.extend(read(plain).unwrap());
    let out = tmp.path().join("prepared");
    prepare(&samples, TEMPLATE, &out).unwrap();

    let starts = binio::read_u64_bin(&out.join("train.ex.bin")).unwrap();
    let ids = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap();
    let mask = binio::read_mask_bin(&out.join("train.mask.bin")).unwrap();
    let rollout = chat_rollout::read(&out, "train", starts.len(), ids.len()).unwrap().expect("sidecars");
    assert_eq!(rollout.advantage[0], 0.0, "a teacher-only record trains on its per-token advantage alone");
    assert!(rollout.advantage[1].is_nan(), "a record without rollout data stays an imitation example");

    let teacher = chat_rollout::read_teacher(&out, "train", starts.len()).unwrap().expect("teacher sidecar");
    assert_eq!(teacher.starts.len(), 2);
    let (t_ids, t_mask) = teacher.example(0).expect("the first example has a teacher");
    assert!(teacher.example(1).is_none(), "an example without a teacher context has none");
    let first = starts[1] as usize;
    assert_eq!(supervised(t_ids, t_mask), supervised(&ids[..first], &mask[..first]));
    let hint_tokens = "<|im_start|>user\nthe hidden test wants a<|im_end|>\n".len();
    assert_eq!(t_ids.len(), first + hint_tokens, "the teacher differs from the student by the inserted messages only");
}

#[test]
fn a_teacher_context_may_come_with_a_scalar_advantage_and_behaviour_log_probs() {
    let tmp = tempfile::tempdir().unwrap();
    let line = format!(
        r#"{{"advantage": 0.75, "teacher_context": {HINT}, "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true,"behavior_logprobs":{}}}]}}"#,
        serde_json::to_string(&vec![-0.5f32; 1 + "<|im_end|>".len()]).unwrap()
    );
    let out = tmp.path().join("prepared");
    prepare(&read(&line).unwrap(), TEMPLATE, &out).unwrap();
    let ids = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap();
    let rollout = chat_rollout::read(&out, "train", 1, ids.len()).unwrap().unwrap();
    assert_eq!(rollout.advantage, [0.75]);
    assert_eq!(rollout.behavior.iter().filter(|b| !b.is_nan()).count(), 1 + "<|im_end|>".len());
}

#[test]
fn a_record_that_misstates_its_teacher_context_is_refused_naming_the_field() {
    let refused = |teacher: &str| read(&record("", teacher)).expect_err("refused");
    let why = refused(r#"{"messages": [{"role": "user", "content": "x"}]}"#);
    assert!(why.contains("at"), "a context without a position: {why}");
    let why = refused(r#"{"at": 1, "messages": [{"role": "user", "content": "x"}], "extra": 1}"#);
    assert!(why.contains("extra"), "unknown field: {why}");
    let why = refused(r#"{"at": 1, "messages": []}"#);
    assert!(why.contains("teacher_context") && why.contains("messages"), "no messages: {why}");
    let why = refused(r#"{"at": 1, "messages": [{"role": "assistant", "content": "x"}]}"#);
    assert!(why.contains("teacher_context") && why.contains("assistant"), "an inserted assistant turn: {why}");
    let why = refused(r#"{"at": 2, "messages": [{"role": "user", "content": "x"}]}"#);
    assert!(why.contains("teacher_context") && why.contains("at"), "inserted after the supervised answer: {why}");
    let why = refused(r#"{"at": 9, "messages": [{"role": "user", "content": "x"}]}"#);
    assert!(why.contains("teacher_context") && why.contains("at"), "past the end: {why}");
    let why = refused(r#"{"at": 1, "messages": [{"role": "user", "content": ""}]}"#);
    assert!(why.contains("teacher_context") && why.contains("content"), "empty text: {why}");
}

#[test]
fn a_record_whose_assistant_tokens_differ_between_the_two_renderings_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let tmpl = template(CONTEXT_DEPENDENT);
    let record = ChatSample::from_json_line_with(&record("", HINT), &data::chat::TextOnly::TRAINER).unwrap();
    let samples = record.answers_as_asked(&tmpl, true).unwrap().unwrap();
    let why = prepare(&samples, CONTEXT_DEPENDENT, &tmp.path().join("prepared")).expect_err("refused");
    assert!(why.contains("teacher") && why.contains("assistant tokens"), "{why}");
}
