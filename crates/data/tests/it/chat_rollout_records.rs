// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A chat record may carry the outcome of the rollout it came from (training-record contract v2): `"reward"` and
//! `"group"` on the record, optionally `"rollout"`, and `"behavior_logprobs"` on an assistant message (the
//! log-probabilities the collector sampled that answer's tokens with). A prepared dataset keeps them as sidecars
//! aligned with its token stream, and a record without them is prepared exactly as before.

use data::binio;
use data::chat::{prepare_chat_samples, ChatSample, RenderOpts};
use data::chat_rollout;
use data::chat_template::ChatTemplate;
use data::tokenizer::{CharTokenizer, Tokenizer};

const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_corpus(&(32u8..127).map(char::from).chain(['\n']).collect::<String>())
}

fn template() -> ChatTemplate {
    ChatTemplate::compile(TEMPLATE).unwrap()
}

/// `"x<|im_end|>"`: what a one-letter answer supervises under this template (the answer and the closing marker; the
/// newline the template writes after it is context, since the sampler stops at the marker).
const ANSWER_TOKENS: usize = 1 + "<|im_end|>".len();

fn write_records(dir: &std::path::Path, lines: &[String]) -> std::path::PathBuf {
    let path = dir.join("records.jsonl");
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

/// A one-answer record; `reward` is the JSON of the reward, written with group `g`.
fn record(reward: Option<&str>, behavior: Option<Vec<f32>>) -> String {
    let adv = reward.map(|r| format!(r#""reward": {r}, "group": "g","#)).unwrap_or_default();
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
fn a_prepared_dataset_keeps_the_outcome_of_each_example_and_the_behaviour_log_probs_of_each_supervised_token() {
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
    let graded = |member, reward| Some(chat_rollout::ExampleRollout { graded: Some(chat_rollout::Graded { group: 0, member, reward, teacher_log_ratio: None }) });
    assert_eq!(rollout.groups, ["g"]);
    assert_eq!(rollout.examples, [graded(0, -1.5), None, graded(1, 0.5)], "a record without a reward is an imitation example; each record is a rollout of its own");

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
    std::fs::write(out.join("train.rollouts.json"), "{}").unwrap();
    prepare(&examples(&path).unwrap(), &out).unwrap();
    assert!(chat_rollout::read(&out, "train", 1, 1).unwrap().is_none());
    assert!(!out.join("train.rollouts.json").exists() && !out.join("train.blp.bin").exists());
}

#[test]
fn a_record_that_misstates_its_rollout_data_is_refused_naming_the_field() {
    let tmp = tempfile::tempdir().unwrap();
    let refused = |line: String| {
        let path = write_records(tmp.path(), &[record(None, None), line]);
        examples(&path).expect_err("refused")
    };
    let why = refused(record(Some("\"high\""), None));
    assert!(why.contains(":2:") && why.contains("reward"), "{why}");

    let positive = vec![0.1; ANSWER_TOKENS];
    let why = refused(record(Some("1.0"), Some(positive)));
    assert!(why.contains(":2:") && why.contains("messages[1]") && why.contains("behavior_logprobs"), "{why}");

    let why = refused(record(None, Some(vec![-0.1; ANSWER_TOKENS])));
    assert!(why.contains(":2:") && why.contains("behavior_logprobs") && why.contains("reward"), "log-probs without a reward: {why}");

    let on_context = r#"{"reward": 1.0, "group": "g", "messages": [{"role":"user","content":"q","train":false,"behavior_logprobs":[-1.0]},{"role":"assistant","content":"a","train":true}]}"#.to_string();
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
        r#"{{"reward": 2.0, "group": "g", "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true,"behavior_logprobs":{}}},{{"role":"user","content":"r","train":false}},{{"role":"assistant","content":"b","train":true,"behavior_logprobs":{}}}]}}"#,
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

/// A record marked `"truncated": true` (the rollout never reached an outcome) is kept as such in a sidecar beside its
/// outcome, and a record that is not marked costs no sidecar at all.
#[test]
fn a_truncated_rollout_is_flagged_in_its_own_sidecar() {
    let tmp = tempfile::tempdir().unwrap();
    let truncated = record(Some("1.0"), None).replacen("{", "{\"truncated\": true,", 1);
    let path = write_records(tmp.path(), &[record(Some("0.5"), None), truncated]);
    let out = tmp.path().join("prepared");
    prepare(&examples(&path).unwrap(), &out).unwrap();
    let starts = binio::read_u64_bin(&out.join("train.ex.bin")).unwrap();
    let ids = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap();
    let rollout = chat_rollout::read(&out, "train", starts.len(), ids.len()).unwrap().expect("sidecars");
    assert_eq!(rollout.truncated, vec![false, true]);

    let plain = write_records(tmp.path(), &[record(Some("0.5"), None)]);
    prepare(&examples(&plain).unwrap(), &out).unwrap();
    assert!(!out.join("train.trn.bin").exists(), "a stale flag sidecar is removed");
}

#[test]
fn truncated_belongs_to_a_rollout() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_records(tmp.path(), &[record(None, None).replacen("{", "{\"truncated\": true,", 1)]);
    let why = examples(&path).expect_err("refused");
    assert!(why.contains("truncated") && why.contains("reward"), "{why}");
}

/// Observation tokens (a tool's output, the user's turn) are context, not action: only the assistant answer marked
/// to train is supervised, so a policy gradient moves nothing the environment wrote.
#[test]
fn the_tokens_of_an_observation_are_never_supervised() {
    let tmp = tempfile::tempdir().unwrap();
    let line = r#"{"reward": 1.0, "group": "g", "messages": [{"role":"user","content":"q","train":false},{"role":"assistant","content":"x","train":false,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"a long observation the environment wrote","train":false},{"role":"assistant","content":"y","train":true}]}"#;
    let path = write_records(tmp.path(), &[line.to_string()]);
    let samples = examples(&path).unwrap();
    let out = tmp.path().join("prepared");
    prepare(&samples, &out).unwrap();
    let ids = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap();
    let mask = binio::read_mask_bin(&out.join("train.mask.bin")).unwrap();
    let supervised: String = ids.iter().zip(&mask).filter(|(_, m)| **m).map(|(i, _)| tokenizer().decode(&[*i])).collect();
    assert_eq!(supervised, "y<|im_end|>", "only the final answer is action; the observation and the earlier turns are context");
}

/// `"token_ids"` on a supervised answer must belong to its message: under a template whose replies brain does not read
/// back (no think block, no tool calls) the ids the template renders for the text must be exactly those, and a record
/// whose text renders to others is refused at the first position that differs (`chat_rollout_sampled` covers the reply
/// read back).
#[test]
fn recorded_token_ids_must_be_the_tokens_the_template_renders_where_its_replies_are_not_read_back() {
    let tmp = tempfile::tempdir().unwrap();
    let rendered: Vec<u32> = tokenizer().encode("a<|im_end|>");
    let line = |ids: &[u32], lps: usize| {
        format!(
            r#"{{"reward": 1.0, "group": "g", "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true,"behavior_logprobs":{},"token_ids":{}}}]}}"#,
            serde_json::to_string(&vec![-0.5f32; lps]).unwrap(),
            serde_json::to_string(ids).unwrap()
        )
    };
    let ok = write_records(tmp.path(), &[line(&rendered, ANSWER_TOKENS)]);
    prepare(&examples(&ok).unwrap(), &tmp.path().join("prepared")).expect("ids that match render");

    let mut wrong = rendered.clone();
    wrong[2] = wrong[2].wrapping_add(1);
    let bad = write_records(tmp.path(), &[line(&wrong, ANSWER_TOKENS)]);
    let why = prepare(&examples(&bad).unwrap(), &tmp.path().join("prepared")).expect_err("refused");
    assert!(why.contains("token_ids") && why.contains("first differ at position 2") && why.contains("do not read back"), "{why}");

    // A sequence that ends early or runs on is the first difference too.
    let short = write_records(tmp.path(), &[line(&rendered[..rendered.len() - 1], ANSWER_TOKENS)]);
    let why = prepare(&examples(&short).unwrap(), &tmp.path().join("prepared")).expect_err("refused");
    assert!(why.contains(&format!("position {}", rendered.len() - 1)) && why.contains("nothing (the sequence ended)"), "{why}");
}

#[test]
fn token_ids_belong_to_a_supervised_answer_of_a_rollout() {
    let tmp = tempfile::tempdir().unwrap();
    let refused = |line: &str| {
        let path = write_records(tmp.path(), &[line.to_string()]);
        examples(&path).expect_err("refused")
    };
    let why = refused(r#"{"messages": [{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true,"token_ids":[1]}]}"#);
    assert!(why.contains("token_ids") && why.contains("reward"), "{why}");
    let why = refused(r#"{"reward": 1.0, "group": "g", "messages": [{"role":"user","content":"q","train":false,"token_ids":[1]},{"role":"assistant","content":"a","train":true}]}"#);
    assert!(why.contains("messages[0]") && why.contains("token_ids"), "{why}");
    let why = refused(r#"{"reward": 1.0, "group": "g", "messages": [{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true,"token_ids":[]}]}"#);
    assert!(why.contains("messages[1]") && why.contains("empty"), "{why}");
}

/// Contract v2 refuses version 1's precomputed advantage by name, and a reward or a group without the other.
#[test]
fn a_v1_advantage_and_a_half_outcome_are_refused_by_rule_and_field() {
    let refusal = |line: &str| {
        let error = ChatSample::from_json_line_with(line, &data::chat::TextOnly::TRAINER).expect_err("refused");
        let fault = error.fault().cloned().expect("a contract rule");
        (fault.rule, fault.field, error.to_string())
    };
    let answer = r#""messages": [{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true}]"#;
    let (rule, field, why) = refusal(&format!(r#"{{"advantage": 1.0, {answer}}}"#));
    assert_eq!((rule, field.as_str()), (chat_rollout::RecordRule::AdvantageRemoved, "advantage"));
    assert!(why.contains("reward") && why.contains("group") && why.contains("v2"), "the refusal names the replacement: {why}");
    assert_eq!(refusal(&format!(r#"{{"reward": 1.0, {answer}}}"#)).0, chat_rollout::RecordRule::MissingGroup);
    assert_eq!(refusal(&format!(r#"{{"group": "g", {answer}}}"#)).0, chat_rollout::RecordRule::MissingReward);
    assert_eq!(refusal(&format!(r#"{{"rollout": "r", {answer}}}"#)).0, chat_rollout::RecordRule::MissingGroup);
    let why = ChatSample::from_json_line_with(&format!(r#"{{"reward": 1.0, "group": "", {answer}}}"#), &data::chat::TextOnly::TRAINER).expect_err("refused").to_string();
    assert!(why.contains("group") && why.contains("empty"), "{why}");
    let why = ChatSample::from_json_line_with(&format!(r#"{{"reward": 1.0, "group": "g", "tokenizer_digest": "md5:x", {answer}}}"#), &data::chat::TextOnly::TRAINER).expect_err("refused").to_string();
    assert!(why.contains("tokenizer_digest"), "{why}");
}

/// The rules between records: the records of one rollout agree, a group holds two rollouts that reached an outcome
/// (a truncated one does not count), and one tokenizer. Each refusal is on the record that breaks it.
#[test]
fn the_rules_between_records_refuse_the_record_that_breaks_them() {
    let digest = |d: char| format!("sha256:{}", d.to_string().repeat(64));
    let line = |extra: &str| format!(r#"{{{extra}"messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true}}]}}"#);
    let lines = [
        line(r#""reward": 1, "group": "a", "rollout": "r0", "#),
        line(r#""reward": 1, "group": "a", "rollout": "r0", "#),
        line(r#""reward": 0, "group": "a", "rollout": "r0", "#),
        line(r#""reward": 0, "group": "a", "rollout": "r1", "#),
        line(r#""reward": 1, "group": "b", "#),
        line(r#""reward": 0, "group": "c", "#),
        line(r#""reward": 1, "group": "c", "truncated": true, "#),
        line(&format!(r#""reward": 1, "group": "d", "tokenizer_digest": "{}", "#, digest('a'))),
        line(&format!(r#""reward": 0, "group": "d", "tokenizer_digest": "{}", "#, digest('b'))),
        line(""),
    ];
    let records: Vec<ChatSample> = lines.iter().map(|l| ChatSample::from_json_line_with(l, &data::chat::TextOnly::TRAINER).unwrap()).collect();
    let rollouts: Vec<Option<&chat_rollout::Rollout>> = records.iter().map(|r| r.rollout.as_ref()).collect();
    let refused = |truncated| -> Vec<(usize, chat_rollout::RecordRule, String)> {
        chat_rollout::check_records(&rollouts, truncated, None)
            .into_iter()
            .map(|(at, e)| {
                let f = e.fault().cloned().unwrap();
                (at, f.rule, f.field)
            })
            .collect()
    };
    use chat_rollout::RecordRule::*;
    assert_eq!(
        refused(chat_rollout::TruncatedRollouts::Mask),
        [(2, RolloutMismatch, "reward".to_string()), (4, GroupOfOne, "group".to_string()), (5, GroupOfOne, "group".to_string()), (8, MixedTokenizerDigest, "tokenizer_digest".to_string())],
        "a repeated record of a rollout is the same outcome; c's truncated rollout does not count"
    );
    assert_eq!(
        refused(chat_rollout::TruncatedRollouts::Keep),
        [(2, RolloutMismatch, "reward".to_string()), (4, GroupOfOne, "group".to_string()), (8, MixedTokenizerDigest, "tokenizer_digest".to_string())],
        "a run that keeps truncated rollouts counts c's in its group"
    );
    // Against the trained base's own tokenizer: the record naming another is refused by its own rule, not as a mix.
    let base = digest('a');
    let against_base: Vec<(usize, chat_rollout::RecordRule)> = chat_rollout::check_records(&rollouts, chat_rollout::TruncatedRollouts::Mask, Some(&base)).into_iter().map(|(at, e)| (at, e.fault().unwrap().rule)).collect();
    assert!(against_base.contains(&(8, TokenizerMismatch)) && !against_base.iter().any(|(at, _)| *at == 7), "{against_base:?}");
    let message = chat_rollout::check_records(&rollouts, chat_rollout::TruncatedRollouts::Mask, Some(&base)).into_iter().find(|(at, _)| *at == 8).unwrap().1.to_string();
    assert!(message.contains(&digest('a')) && message.contains(&digest('b')), "both digests are named: {message}");
}

/// The wire names of the truncated-rollout handling round-trip, and only a kept handling counts a truncated rollout.
#[test]
fn a_truncated_rollout_counts_only_where_the_run_keeps_it() {
    use chat_rollout::TruncatedRollouts;
    for t in TruncatedRollouts::ALL {
        assert_eq!(TruncatedRollouts::parse(t.name()), Ok(t));
    }
    assert!(TruncatedRollouts::parse("drop").unwrap_err().contains("mask"));
    assert_eq!(TruncatedRollouts::default(), TruncatedRollouts::Mask);
    assert!(TruncatedRollouts::Mask.counts(false) && !TruncatedRollouts::Mask.counts(true));
    assert!(TruncatedRollouts::Keep.counts(false) && TruncatedRollouts::Keep.counts(true));
}

/// A baseline-only record (`"baseline_only": true`, with a reward and group and nothing trained) is accepted, takes
/// part in its group's rules (it makes a lone trained rollout's group a pair), is kept in the prepared split's sidecar
/// as a baseline and never as an example; one that trains something, carries an answer's per-token fields, a teacher
/// context or `truncated`, or has no reward is refused by rule and field.
#[test]
fn a_baseline_only_rollout_counts_in_its_group_and_is_never_an_example() {
    let user = r#"{"role":"user","content":"q","train":false}"#;
    let baseline = |extra: &str, answer: &str| format!(r#"{{"reward": 0, "group": "g", "rollout": "base", "baseline_only": true{extra}, "messages": [{user},{answer}]}}"#);
    let quiet = r#"{"role":"assistant","content":"b","train":false}"#;
    let parsed = ChatSample::from_json_line_with(&baseline("", quiet), &data::chat::TextOnly::TRAINER).expect("a baseline");
    assert!(parsed.rollout.as_ref().is_some_and(|r| r.baseline_only && r.outcome.as_ref().is_some_and(|o| o.reward == 0.0)));

    use chat_rollout::RecordRule::*;
    for (line, rule, field) in [
        (baseline("", r#"{"role":"assistant","content":"b","train":true}"#), BaselineTrains, "messages[1].train"),
        (baseline(r#", "truncated": true"#, quiet), BaselineTrains, "truncated"),
        (baseline(r#", "teacher_context": {"at": 1, "messages": [{"role": "system", "content": "hint"}]}"#, quiet), BaselineTrains, "teacher_context"),
        (format!(r#"{{"baseline_only": true, "messages": [{user},{quiet}]}}"#), MissingReward, "baseline_only"),
    ] {
        let err = ChatSample::from_json_line_with(&line, &data::chat::TextOnly::TRAINER).expect_err(field);
        let fault = err.fault().cloned().unwrap_or_else(|| panic!("{field}: {err}"));
        assert_eq!((fault.rule, fault.field.as_str()), (rule, field), "{err}");
    }

    // One trained rollout and a baseline of its group: a pair, not a group of one.
    let trained = record(Some("1"), None);
    let records: Vec<ChatSample> = [trained.clone(), baseline("", quiet)].iter().map(|l| ChatSample::from_json_line_with(l, &data::chat::TextOnly::TRAINER).unwrap()).collect();
    let rollouts: Vec<Option<&chat_rollout::Rollout>> = records.iter().map(|r| r.rollout.as_ref()).collect();
    assert!(chat_rollout::check_records(&rollouts, chat_rollout::TruncatedRollouts::Mask, None).is_empty());
    let alone = ChatSample::from_json_line_with(&trained, &data::chat::TextOnly::TRAINER).unwrap();
    assert_eq!(chat_rollout::check_records(&[alone.rollout.as_ref()], chat_rollout::TruncatedRollouts::Mask, None).len(), 1, "without the baseline the group holds one");

    let dir = tempfile::tempdir().unwrap();
    let mut samples = alone.answers_as_asked(&template(), true).unwrap().unwrap();
    samples.push(records[1].clone());
    prepare(&samples, dir.path()).unwrap();
    let starts = binio::read_u64_bin(&dir.path().join("train.ex.bin")).unwrap();
    let tokens = binio::read_tokens_u32(&dir.path().join("train")).unwrap();
    let split = chat_rollout::read(dir.path(), "train", starts.len(), tokens.len()).unwrap().expect("rollouts");
    assert_eq!((starts.len(), split.examples.len(), split.baselines.len()), (1, 1, 1), "the baseline is no example");
    let (trained, base) = (split.examples[0].unwrap().graded.unwrap(), split.baselines[0]);
    assert_eq!((trained.group, base.group, base.reward), (0, 0, 0.0), "one group");
    assert_ne!(trained.member, base.member, "two rollouts");
}
