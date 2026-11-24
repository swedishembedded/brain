// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distilling a large teacher model into a small student on the student's
// own samples, for its clients. If your team needs expertise in on-policy distillation of language models then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A rollout record may carry a teacher model's scores of its sampled answers, for on-policy distillation: per token
//! (`"teacher_logprobs"`, one per token of the answer, for a teacher with the student's tokenizer) or per answer
//! (`"teacher_logprob"`, the teacher's log-probability of the answer's text in its own tokenization, for any teacher).
//! A prepared dataset keeps the per-token scores aligned with the token stream (`<split>.tlp.bin`) and, for the
//! per-answer form, the record's log-ratio `sum teacher_logprob - sum behavior_logprobs` with its rollout's outcome.
//! A record that states them in a way the estimators cannot use is refused by rule and field.

use data::binio;
use data::chat::{prepare_chat_samples, ChatSample, RenderOpts};
use data::chat_rollout::{self, RecordRule};
use data::chat_template::ChatTemplate;
use data::tokenizer::CharTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// The tokens of the answer `a` as the template renders it after its prompt: the text and the end-of-turn marker.
const ANSWER_TOKENS: usize = 1 + "<|im_end|>".len();

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_corpus(&(32u8..127).map(char::from).chain(['\n']).collect::<String>())
}

fn template() -> ChatTemplate {
    ChatTemplate::compile(TEMPLATE).unwrap()
}

fn list(values: &[f32]) -> String {
    serde_json::to_string(values).unwrap()
}

/// A record whose one user turn is answered by `a`, with `outer` fields on the record and `answer` fields on the
/// answer.
fn record(outer: &str, answer: &str) -> String {
    format!(r#"{{{outer}"messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true{answer}}}]}}"#)
}

fn read(line: &str) -> Result<Vec<ChatSample>, data::chat_rollout::RecordError> {
    let record = ChatSample::from_json_line_with(line, &data::chat::TextOnly::TRAINER)?;
    Ok(record.answers_as_asked(&template(), true).expect("renders").expect("assistant answers"))
}

fn prepare(samples: &[ChatSample], dir: &std::path::Path) -> Result<(), String> {
    prepare_chat_samples(samples, samples, &tokenizer(), &template(), RenderOpts::default(), 128, dir).map(drop).map_err(|e| e.to_string())
}

fn refusal(line: &str) -> (RecordRule, String) {
    let err = read(line).expect_err("refused");
    let fault = err.fault().unwrap_or_else(|| panic!("a contract refusal names its rule: {err}"));
    (fault.rule, fault.field.clone())
}

#[test]
fn per_token_teacher_scores_are_kept_aligned_with_the_answers_tokens() {
    let tmp = tempfile::tempdir().unwrap();
    let teacher: Vec<f32> = (0..ANSWER_TOKENS).map(|i| -0.1 * (i + 1) as f32).collect();
    let line = record("", &format!(r#","teacher_logprobs":{}"#, list(&teacher)));
    let out = tmp.path().join("prepared");
    prepare(&read(&line).expect("a teacher's per-token scores alone make a rollout"), &out).unwrap();

    let ids = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap();
    let mask = binio::read_mask_bin(&out.join("train.mask.bin")).unwrap();
    let rollout = chat_rollout::read(&out, "train", 1, ids.len()).unwrap().expect("a rollout sidecar");
    assert_eq!(rollout.examples, [Some(chat_rollout::ExampleRollout { graded: None, ..Default::default() })], "a rollout without a reward of its own");
    let placed = chat_rollout::read_teacher_logprobs(&out, "train", ids.len()).unwrap().expect("the per-token sidecar");
    let supervised: Vec<f32> = placed.iter().zip(&mask).filter(|(_, m)| **m).map(|(t, _)| *t).collect();
    assert_eq!(supervised, teacher, "one teacher log-probability per supervised token, in order");
    assert!(placed.iter().zip(&mask).all(|(t, m)| *m != t.is_nan()), "NaN exactly where nothing is supervised");
}

#[test]
fn per_answer_teacher_scores_become_the_records_log_ratio_beside_its_outcome() {
    let tmp = tempfile::tempdir().unwrap();
    let behavior = vec![-0.25f32; ANSWER_TOKENS];
    let line = record(r#""reward": 1.0, "group": "g", "#, &format!(r#","behavior_logprobs":{},"teacher_logprob":-9.5"#, list(&behavior)));
    let other = record(r#""reward": 0.0, "group": "g", "#, &format!(r#","behavior_logprobs":{},"teacher_logprob":-1.0"#, list(&behavior)));
    let mut samples = read(&line).unwrap();
    samples.extend(read(&other).unwrap());
    let out = tmp.path().join("prepared");
    prepare(&samples, &out).unwrap();
    let ids = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap();
    let rollout = chat_rollout::read(&out, "train", 2, ids.len()).unwrap().unwrap();
    let ratios: Vec<Option<f64>> = rollout.examples.iter().map(|e| e.and_then(|e| e.graded).and_then(|g| g.teacher_log_ratio)).collect();
    let student = -0.25 * ANSWER_TOKENS as f64;
    assert_eq!(ratios, [Some(-9.5 - student), Some(-1.0 - student)], "the teacher's log-probability of the text less the student's");
    assert!(chat_rollout::read_teacher_logprobs(&out, "train", ids.len()).unwrap().is_none(), "no per-token scores were recorded");
}

#[test]
fn a_per_token_score_count_that_is_not_the_answers_is_refused() {
    let line = record("", &format!(r#","teacher_logprobs":{}"#, list(&[-0.1; ANSWER_TOKENS - 1])));
    let sample = read(&line).unwrap().remove(0);
    let err = sample.encode_rollout(&tokenizer(), &template(), RenderOpts::default()).expect_err("one value short");
    let fault = err.fault().expect("names its rule");
    assert_eq!((fault.rule, fault.field.as_str()), (RecordRule::TeacherLogprobCount, "messages[1].teacher_logprobs"));
    assert_eq!(RecordRule::TeacherLogprobCount.name(), "teacher_logprob_count");
}

#[test]
fn teacher_scores_the_estimators_cannot_use_are_refused_by_rule_and_field() {
    let behavior = format!(r#","behavior_logprobs":{}"#, list(&[-0.25; ANSWER_TOKENS]));
    let tokens = format!(r#","teacher_logprobs":{}"#, list(&[-0.1; ANSWER_TOKENS]));
    let graded = r#""reward": 1.0, "group": "g", "#;
    let cases = [
        // Both forms on one answer: one record trains one estimator.
        (record(graded, &format!("{behavior}{tokens},\"teacher_logprob\":-3.0")), "messages[1].teacher_logprob"),
        // The per-answer form needs the student's log-probability of the same answer.
        (record(graded, ",\"teacher_logprob\":-3.0"), "messages[1].teacher_logprob"),
        // ... and a group whose other completions are its baseline.
        (record("", &format!("{behavior},\"teacher_logprob\":-3.0")), "messages[1].teacher_logprob"),
        // ... and the record is the whole rollout: the log-ratio of a rollout of several calls is not one record's.
        (record(r#""reward": 1.0, "group": "g", "rollout": "r", "#, &format!("{behavior},\"teacher_logprob\":-3.0")), "rollout"),
        // A per-answer score is a log-probability.
        (record(graded, &format!("{behavior},\"teacher_logprob\":0.5")), "messages[1].teacher_logprob"),
        // Scores of a privileged-context teacher and of a teacher model do not mix.
        (record(r#""teacher_context": {"at": 1, "messages": [{"role": "user", "content": "hint"}]}, "#, &tokens), "messages[1].teacher_logprobs"),
    ];
    for (line, field) in cases {
        assert_eq!(refusal(&line), (RecordRule::TeacherScores, field.to_string()), "{line}");
    }
    // Every supervised answer of a record carries the teacher's score or none does.
    let two = format!(
        r#"{{"messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true{tokens}}},{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true}}]}}"#
    );
    assert_eq!(refusal(&two), (RecordRule::TeacherScores, "messages[3]".to_string()));
    assert_eq!(RecordRule::TeacherScores.name(), "teacher_scores");
    // On a message that is not a supervised answer there is nothing to score.
    let context = r#"{"messages": [{"role":"user","content":"q","train":false,"teacher_logprob":-1.0},{"role":"assistant","content":"a","train":true}]}"#;
    let err = read(context).expect_err("refused");
    assert!(err.to_string().contains("teacher_logprob"), "{err}");
}
