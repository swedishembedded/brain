// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A chat record may carry the outcome of the rollout it came from: `"advantage"` on the record, and
//! `"behavior_logprobs"` on an assistant message (the log-probabilities the collector sampled that answer's tokens
//! with). A prepared dataset keeps them as sidecars aligned with its token stream, and a record without them is
//! prepared exactly as before.

use data::binio;
use data::chat::{prepare_chat_samples, ChatSample, RenderOpts};
use data::chat_rollout;
use data::chat_template::ChatTemplate;
use data::tokenizer::CharTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_corpus(&(32u8..127).map(char::from).chain(['\n']).collect::<String>())
}

fn template() -> ChatTemplate {
    ChatTemplate::compile(TEMPLATE).unwrap()
}

/// `"x<|im_end|>\n"`: what a one-letter answer supervises under this template (the answer and the closing marker).
const ANSWER_TOKENS: usize = 1 + "<|im_end|>\n".len();

fn write_records(dir: &std::path::Path, lines: &[String]) -> std::path::PathBuf {
    let path = dir.join("records.jsonl");
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

fn record(advantage: Option<&str>, behavior: Option<Vec<f32>>) -> String {
    let adv = advantage.map(|a| format!(r#""advantage": {a},"#)).unwrap_or_default();
    let blp = behavior.map(|b| format!(r#", "behavior_logprobs": {}"#, serde_json::to_string(&b).unwrap())).unwrap_or_default();
    format!(
        r#"{{{adv} "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true{blp}}}]}}"#
    )
}

/// What the trainer's reader does with a record that carries rollout data: one example per supervised answer.
fn examples(path: &std::path::Path) -> Result<Vec<ChatSample>, String> {
    let tmpl = template();
    let mut out = Vec::new();
    for record in ChatSample::from_jsonl(path).map_err(|e| e.to_string())? {
        out.extend(record.answers_as_asked(&tmpl, true).map_err(|e| e.to_string())?.expect("assistant answers"));
    }
    Ok(out)
}

fn prepare(samples: &[ChatSample], dir: &std::path::Path) -> Result<(), String> {
    prepare_chat_samples(samples, samples, &tokenizer(), &template(), RenderOpts::default(), 128, dir).map(drop).map_err(|e| e.to_string())
}

#[test]
fn a_prepared_dataset_keeps_the_advantage_of_each_example_and_the_behaviour_log_probs_of_each_supervised_token() {
    let tmp = tempfile::tempdir().unwrap();
    let lps: Vec<f32> = (0..ANSWER_TOKENS).map(|i| -0.1 * (i + 1) as f32).collect();
    let path = write_records(tmp.path(), &[record(Some("-1.5"), Some(lps.clone())), record(None, None), record(Some("0.5"), None)]);
    let samples = examples(&path).unwrap();
    let out = tmp.path().join("prepared");
    prepare(&samples, &out).unwrap();

    let starts = binio::read_u64_bin(&out.join("train.ex.bin")).unwrap();
    let ids = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap();
    let mask = binio::read_mask_bin(&out.join("train.mask.bin")).unwrap();
    let rollout = chat_rollout::read(&out, "train", starts.len(), ids.len()).unwrap().expect("sidecars");
    assert_eq!(rollout.advantage.len(), 3);
    assert_eq!(rollout.advantage[0], -1.5);
    assert!(rollout.advantage[1].is_nan(), "a record without an advantage is an imitation example");
    assert_eq!(rollout.advantage[2], 0.5);

    assert_eq!(rollout.behavior.len(), ids.len());
    let first_end = starts[1] as usize;
    let supervised: Vec<usize> = (0..first_end).filter(|&i| mask[i]).collect();
    assert_eq!(supervised.len(), ANSWER_TOKENS);
    for (k, &i) in supervised.iter().enumerate() {
        assert_eq!(rollout.behavior[i], lps[k], "token {i}");
    }
    let known = rollout.behavior.iter().filter(|b| !b.is_nan()).count();
    assert_eq!(known, ANSWER_TOKENS, "only the recorded answer has behaviour log-probs; the rest is NaN");
}

#[test]
fn records_without_rollout_data_prepare_with_no_sidecar_and_a_stale_one_is_removed() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_records(tmp.path(), &[record(None, None)]);
    let out = tmp.path().join("prepared");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("train.adv.bin"), [0u8; 4]).unwrap();
    prepare(&examples(&path).unwrap(), &out).unwrap();
    assert!(chat_rollout::read(&out, "train", 1, 1).unwrap().is_none());
    assert!(!out.join("train.adv.bin").exists() && !out.join("train.blp.bin").exists());
}

#[test]
fn a_record_that_misstates_its_rollout_data_is_refused_naming_the_field() {
    let tmp = tempfile::tempdir().unwrap();
    let refused = |line: String| {
        let path = write_records(tmp.path(), &[record(None, None), line]);
        examples(&path).expect_err("refused")
    };
    let why = refused(record(Some("\"high\""), None));
    assert!(why.contains(":2:") && why.contains("advantage"), "{why}");

    let positive = vec![0.1; ANSWER_TOKENS];
    let why = refused(record(Some("1.0"), Some(positive)));
    assert!(why.contains(":2:") && why.contains("messages[1]") && why.contains("behavior_logprobs"), "{why}");

    let why = refused(record(None, Some(vec![-0.1; ANSWER_TOKENS])));
    assert!(why.contains(":2:") && why.contains("behavior_logprobs") && why.contains("advantage"), "log-probs without an advantage: {why}");

    let on_context = r#"{"advantage": 1.0, "messages": [{"role":"user","content":"q","train":false,"behavior_logprobs":[-1.0]},{"role":"assistant","content":"a","train":true}]}"#.to_string();
    let why = refused(on_context);
    assert!(why.contains("messages[0]") && why.contains("behavior_logprobs"), "{why}");

    let empty = refused(record(Some("1.0"), Some(Vec::new())));
    assert!(empty.contains("messages[1]") && empty.contains("behavior_logprobs"), "{empty}");
}

#[test]
fn behaviour_log_probs_that_do_not_match_the_answers_tokens_are_refused_with_both_counts() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_records(tmp.path(), &[record(Some("1.0"), Some(vec![-0.5; ANSWER_TOKENS - 1]))]);
    let samples = examples(&path).unwrap();
    let why = prepare(&samples, &tmp.path().join("prepared")).expect_err("refused");
    assert!(why.contains("behavior_logprobs") && why.contains(&format!("{}", ANSWER_TOKENS - 1)) && why.contains(&format!("{ANSWER_TOKENS}")), "{why}");
}

#[test]
fn a_record_with_behaviour_log_probs_is_not_encoded_as_a_whole_conversation() {
    // The whole-conversation mask also supervises the turn's header, which the collector did not sample: the
    // log-probs are placed only by the answer-as-asked rendering, so encoding the record itself refuses.
    let tmp = tempfile::tempdir().unwrap();
    let path = write_records(tmp.path(), &[record(Some("1.0"), Some(vec![-0.5; ANSWER_TOKENS]))]);
    let whole = ChatSample::from_jsonl(&path).unwrap().remove(0);
    let why = whole.encode(&tokenizer(), &template()).expect_err("refused").to_string();
    assert!(why.contains("behavior_logprobs"), "{why}");
}

#[test]
fn a_packed_record_places_each_answers_log_probs_on_that_answers_tokens() {
    let tmp = tempfile::tempdir().unwrap();
    let (first, second): (Vec<f32>, Vec<f32>) = (vec![-0.2; ANSWER_TOKENS], vec![-0.7; ANSWER_TOKENS]);
    let line = format!(
        r#"{{"advantage": 2.0, "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true,"behavior_logprobs":{}}},{{"role":"user","content":"r","train":false}},{{"role":"assistant","content":"b","train":true,"behavior_logprobs":{}}}]}}"#,
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap()
    );
    let path = write_records(tmp.path(), &[line]);
    let record = ChatSample::from_jsonl(&path).unwrap().remove(0);
    let packed = record.answers_packed(&template(), true).unwrap().expect("assistant answers");
    let encoded = packed.encode_rollout(&tokenizer(), &template(), RenderOpts::default()).unwrap();
    let placed: Vec<f32> = encoded.behavior.iter().copied().filter(|b| !b.is_nan()).collect();
    assert_eq!(placed, [first, second].concat());
    for (mask, lp) in encoded.mask.iter().zip(&encoded.behavior) {
        assert_eq!(*mask, !lp.is_nan(), "a log-probability sits on exactly the supervised tokens");
    }
}
