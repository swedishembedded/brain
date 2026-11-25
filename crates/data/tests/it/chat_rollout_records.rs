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
    let graded = |member, reward| Some(chat_rollout::ExampleRollout { graded: Some(chat_rollout::Graded { group: 0, member, reward, teacher_log_ratio: None }), ..Default::default() });
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

/// What a collector that asks its sampler for the top-`k` of every step records on an answer of `tokens` tokens: ids
/// distinct within a step, whole-row log-probabilities of a distribution whose top-`k` mass is below one.
fn top_of(tokens: usize, k: usize) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let ids = (0..tokens).map(|t| (0..k).map(|j| (100 * t + j) as u32).collect()).collect();
    let lps = (0..tokens).map(|_| (0..k).map(|j| (0.4 / (j + 1) as f32).ln()).collect()).collect();
    (ids, lps)
}

/// A rollout record of group `g` with one supervised answer carrying `behavior_logprobs` and the given top-k fields
/// (raw JSON, `None` to omit); `extra` is spliced into the record's own keys.
fn top_record(extra: &str, behavior: usize, ids: Option<serde_json::Value>, lps: Option<serde_json::Value>) -> String {
    let field = |name: &str, v: Option<serde_json::Value>| v.map(|v| format!(r#","{name}":{v}"#)).unwrap_or_default();
    let blp = serde_json::to_string(&vec![-0.5f32; behavior]).unwrap();
    format!(
        r#"{{"reward": 1.0, "group": "g", {extra} "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true,"behavior_logprobs":{blp}{}{}}}]}}"#,
        field("behavior_top_ids", ids),
        field("behavior_top_logprobs", lps)
    )
}

fn valid_top(tokens: usize, k: usize) -> (Option<serde_json::Value>, Option<serde_json::Value>) {
    let (ids, lps) = top_of(tokens, k);
    (Some(serde_json::json!(ids)), Some(serde_json::json!(lps)))
}

fn refusal(line: &str) -> (chat_rollout::RecordRule, String, String) {
    let error = ChatSample::from_json_line_with(line, &data::chat::TextOnly::TRAINER).expect_err("refused");
    let fault = error.fault().cloned().unwrap_or_else(|| panic!("a contract rule: {error}"));
    (fault.rule, fault.field, error.to_string())
}

/// The sampler's top-k of each token lands on exactly the tokens its `behavior_logprobs` land on, in the prepared
/// split's `btk`/`btl` sidecars (`[tokens * k]`, `NO_TOP_ID` and NaN where none was recorded), whatever else the split
/// holds; a split without any has no sidecar and a stale one is removed.
#[test]
fn the_top_k_of_each_token_is_kept_where_its_behaviour_log_prob_is() {
    use chat_rollout::{read_behavior_top, NO_TOP_ID};
    let tmp = tempfile::tempdir().unwrap();
    let k = 3;
    let (ids, lps) = valid_top(ANSWER_TOKENS, k);
    let lines = [record(None, None), top_record("", ANSWER_TOKENS, ids, lps), record(Some("0.5"), Some(vec![-0.3; ANSWER_TOKENS]))];
    let path = write_records(tmp.path(), &lines);
    let out = tmp.path().join("prepared");
    prepare(&examples(&path).unwrap(), &out).unwrap();

    let tokens = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap().len();
    let starts = binio::read_u64_bin(&out.join("train.ex.bin")).unwrap();
    let rollout = chat_rollout::read(&out, "train", starts.len(), tokens).unwrap().expect("sidecars");
    let top = read_behavior_top(&out, "train", tokens).unwrap().expect("top-k sidecars");
    assert_eq!((top.k, top.ids.len(), top.logprobs.len()), (k, tokens * k, tokens * k));
    let (want_ids, want_lps) = top_of(ANSWER_TOKENS, k);
    let mut seen = 0;
    for t in 0..tokens {
        let (row_ids, row_lps) = (&top.ids[t * k..(t + 1) * k], &top.logprobs[t * k..(t + 1) * k]);
        if t >= starts[1] as usize && t < starts[2] as usize && !rollout.behavior[t].is_nan() {
            assert_eq!((row_ids, row_lps), (want_ids[seen].as_slice(), want_lps[seen].as_slice()), "token {t}");
            seen += 1;
        } else {
            assert!(row_ids.iter().all(|&i| i == NO_TOP_ID) && row_lps.iter().all(|l| l.is_nan()), "token {t} has none");
        }
    }
    assert_eq!(seen, ANSWER_TOKENS, "every token with a behaviour log-probability of the record that carries a top-k has its own");
    assert!(read_behavior_top(&out, "train", tokens + 1).is_err(), "checked against the tokens the split holds");

    let plain = write_records(tmp.path(), &[record(Some("0.5"), Some(vec![-0.3; ANSWER_TOKENS]))]);
    prepare(&examples(&plain).unwrap(), &out).unwrap();
    let tokens = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap().len();
    assert!(read_behavior_top(&out, "train", tokens).unwrap().is_none());
    assert!(!out.join("train.btk.bin").exists() && !out.join("train.btl.bin").exists(), "stale sidecars are removed");
}

#[test]
fn a_packed_record_places_each_answers_top_k_on_that_answers_tokens() {
    let tmp = tempfile::tempdir().unwrap();
    let (a, b) = (top_of(ANSWER_TOKENS, 2), top_of(ANSWER_TOKENS, 2));
    let line = format!(
        r#"{{"reward": 2.0, "group": "g", "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true,"behavior_logprobs":{blp},"behavior_top_ids":{},"behavior_top_logprobs":{}}},{{"role":"user","content":"r","train":false}},{{"role":"assistant","content":"b","train":true,"behavior_logprobs":{blp},"behavior_top_ids":{},"behavior_top_logprobs":{}}}]}}"#,
        serde_json::json!(a.0),
        serde_json::json!(a.1),
        serde_json::json!(b.0),
        serde_json::json!(b.1),
        blp = serde_json::to_string(&vec![-0.5f32; ANSWER_TOKENS]).unwrap()
    );
    let record = ChatSample::from_jsonl(&write_records(tmp.path(), &[line])).unwrap().remove(0);
    let packed = record.answers_packed(&template(), true).unwrap().expect("assistant answers");
    let encoded = packed.encode_rollout(&tokenizer(), &template(), RenderOpts::default()).unwrap();
    let top = encoded.behavior_top.expect("top-k placed");
    assert_eq!(top.k, 2);
    let placed: Vec<u32> = (0..encoded.ids.len()).filter(|&t| !encoded.behavior[t].is_nan()).flat_map(|t| top.ids[t * 2..t * 2 + 2].to_vec()).collect();
    assert_eq!(placed, [a.0, b.0].concat().concat());
}

#[test]
fn a_record_that_misstates_its_top_k_is_refused_by_rule_and_field() {
    use chat_rollout::RecordRule::{BaselineTrains, BehaviorTop, BehaviorTopCount, MissingReward};
    let n = ANSWER_TOKENS;
    type Edit<'a> = &'a dyn Fn(&mut Vec<Vec<u32>>, &mut Vec<Vec<f32>>);
    let with = |f: Edit| {
        let (mut ids, mut lps) = top_of(n, 3);
        f(&mut ids, &mut lps);
        top_record("", n, Some(serde_json::json!(ids)), Some(serde_json::json!(lps)))
    };
    let (ids, lps) = valid_top(n, 3);
    ChatSample::from_json_line_with(&top_record("", n, ids.clone(), lps.clone()), &data::chat::TextOnly::TRAINER).expect("a valid top-k");

    let cases: Vec<(&str, String, chat_rollout::RecordRule, &str)> = vec![
        ("ids count", with(&|ids, _| drop(ids.pop())), BehaviorTopCount, "messages[1].behavior_top_ids"),
        ("logprobs count", with(&|_, lps| lps.push(vec![-1.0; 3])), BehaviorTopCount, "messages[1].behavior_top_logprobs"),
        ("ragged k in a record", with(&|ids, lps| { ids[4].pop(); lps[4].pop(); }), BehaviorTop, "messages[1].behavior_top_ids"),
        ("ids and logprobs of a token differ", with(&|_, lps| {
            lps[2].pop();
        }), BehaviorTop, "messages[1].behavior_top_logprobs"),
        ("duplicate id", with(&|ids, _| ids[3][1] = ids[3][0]), BehaviorTop, "messages[1].behavior_top_ids"),
        ("positive logprob", with(&|_, lps| lps[1][0] = 0.5), BehaviorTop, "messages[1].behavior_top_logprobs"),
        ("mass above one", with(&|_, lps| lps[0] = vec![-0.1; 3]), BehaviorTop, "messages[1].behavior_top_logprobs"),
        ("empty k", with(&|ids, lps| { ids.iter_mut().for_each(Vec::clear); lps.iter_mut().for_each(Vec::clear); }), BehaviorTop, "messages[1].behavior_top_ids"),
        ("ids without logprobs", top_record("", n, ids.clone(), None), BehaviorTop, "messages[1].behavior_top_logprobs"),
        ("logprobs without ids", top_record("", n, None, lps.clone()), BehaviorTop, "messages[1].behavior_top_ids"),
        (
            "without behavior_logprobs",
            format!(r#"{{"reward": 1.0, "group": "g", "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true,"behavior_top_ids":{},"behavior_top_logprobs":{}}}]}}"#, ids.clone().unwrap(), lps.clone().unwrap()),
            BehaviorTop,
            "messages[1].behavior_top_ids",
        ),
        ("on a baseline", top_record(r#""baseline_only": true,"#, n, ids.clone(), lps.clone()), BaselineTrains, "messages[1].train"),
        ("without a reward", top_record("", n, ids.clone(), lps.clone()).replace(r#""reward": 1.0, "group": "g","#, ""), MissingReward, "messages[1].behavior_logprobs"),
    ];
    for (what, line, rule, field) in cases {
        let (got_rule, got_field, why) = refusal(&line);
        assert_eq!((got_rule, got_field.as_str()), (rule, field), "{what}: {why}");
    }
    // Ragged K across the messages of one record.
    let (a, b) = (top_of(n, 2), top_of(n, 3));
    let blp = serde_json::to_string(&vec![-0.5f32; n]).unwrap();
    let line = format!(
        r#"{{"reward": 1.0, "group": "g", "messages": [{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"a","train":true,"behavior_logprobs":{blp},"behavior_top_ids":{},"behavior_top_logprobs":{}}},{{"role":"user","content":"r","train":false}},{{"role":"assistant","content":"b","train":true,"behavior_logprobs":{blp},"behavior_top_ids":{},"behavior_top_logprobs":{}}}]}}"#,
        serde_json::json!(a.0),
        serde_json::json!(a.1),
        serde_json::json!(b.0),
        serde_json::json!(b.1)
    );
    let (rule, field, _) = refusal(&line);
    assert_eq!((rule, field.as_str()), (BehaviorTop, "messages[3].behavior_top_ids"));
    assert_eq!(chat_rollout::RecordRule::BehaviorTop.name(), "behavior_top");
    assert_eq!(chat_rollout::RecordRule::BehaviorTopCount.name(), "behavior_top_count");
}

#[test]
fn a_split_that_mixes_top_k_widths_is_refused_naming_the_example() {
    let tmp = tempfile::tempdir().unwrap();
    let line = |k| {
        let (ids, lps) = valid_top(ANSWER_TOKENS, k);
        top_record("", ANSWER_TOKENS, ids, lps)
    };
    let path = write_records(tmp.path(), &[line(2), line(3)]);
    let why = prepare(&examples(&path).unwrap(), &tmp.path().join("prepared")).expect_err("refused");
    assert!(why.contains("behavior_top") && why.contains("example 1") && why.contains('2') && why.contains('3'), "{why}");
}

/// `policy_version` (the oldest policy version behind the record) and `sampler` (the controls the decoder applied)
/// are rollout fields: validated, and kept per example in the prepared split.
#[test]
fn policy_version_and_sampler_are_validated_and_kept_per_example() {
    let tmp = tempfile::tempdir().unwrap();
    let extra = r#""policy_version": 7, "sampler": {"temperature": 0.8, "top_p": 0.95},"#;
    let path = write_records(tmp.path(), &[top_record(extra, ANSWER_TOKENS, None, None), top_record("", ANSWER_TOKENS, None, None), record(None, None)]);
    let out = tmp.path().join("prepared");
    prepare(&examples(&path).unwrap(), &out).unwrap();
    let starts = binio::read_u64_bin(&out.join("train.ex.bin")).unwrap();
    let tokens = binio::read_u32_bin(&out.join("train.u32.bin")).unwrap().len();
    let split = chat_rollout::read(&out, "train", starts.len(), tokens).unwrap().expect("sidecars");
    let stamped = split.examples[0].unwrap();
    assert_eq!(stamped.policy_version, Some(7));
    let sampler = stamped.sampler.expect("sampler");
    assert_eq!((sampler.temperature, sampler.top_k, sampler.top_p, sampler.min_p), (Some(0.8), None, Some(0.95), None));
    assert!(sampler.truncates());
    assert_eq!((split.examples[1].unwrap().policy_version, split.examples[1].unwrap().sampler), (None, None));

    use chat_rollout::SamplerDecl;
    let decl = |temperature, top_k, top_p, min_p| SamplerDecl { temperature, top_k, top_p, min_p };
    assert!(!decl(Some(1.0), None, None, None).truncates() && !decl(None, None, Some(1.0), Some(0.0)).truncates());
    assert!(decl(None, Some(40), None, None).truncates() && decl(None, None, None, Some(0.05)).truncates());

    // An older sidecar without the fields reads as before.
    let old = r#"{"groups":["g"],"examples":[{"graded":{"group":0,"member":0,"reward":1.0}}]}"#;
    let older = tmp.path().join("older");
    std::fs::create_dir_all(&older).unwrap();
    std::fs::write(older.join("train.rollouts.json"), old).unwrap();
    let again = chat_rollout::read(&older, "train", 1, 0).unwrap_or_else(|e| panic!("{e}"));
    let example = again.expect("sidecar").examples[0].unwrap();
    assert_eq!((example.policy_version, example.sampler), (None, None));

    for (extra, needle) in [
        (r#""policy_version": -1,"#, "policy_version"),
        (r#""policy_version": 1.5,"#, "policy_version"),
        (r#""policy_version": "3","#, "policy_version"),
        (r#""sampler": {"temperature": 0},"#, "temperature"),
        (r#""sampler": {"top_k": 0},"#, "top_k"),
        (r#""sampler": {"top_p": 0},"#, "top_p"),
        (r#""sampler": {"top_p": 1.5},"#, "top_p"),
        (r#""sampler": {"min_p": 1},"#, "min_p"),
        (r#""sampler": {"seed": 1},"#, "seed"),
    ] {
        let why = ChatSample::from_json_line_with(&top_record(extra, ANSWER_TOKENS, None, None), &data::chat::TextOnly::TRAINER).expect_err(extra).to_string();
        assert!(why.contains(needle) && why.contains("sampler") == extra.contains("sampler"), "{extra}: {why}");
    }
    let alone = record(None, None).replacen("{", r#"{"policy_version": 3,"#, 1);
    let (rule, field, _) = refusal(&alone);
    assert_eq!((rule, field.as_str()), (chat_rollout::RecordRule::MissingReward, "policy_version"));
}

/// The contract this build reports (`CONTRACT_DIGEST`, served as `server.training_records`) is the digest of its own
/// example set, and the digest is of what the set says, not of where it lies: a copy elsewhere digests the same, a
/// changed record does not.
#[test]
fn the_reported_contract_digest_is_the_digest_of_the_example_set() {
    use data::chat_rollout::{contract_digest, CONTRACT, CONTRACT_DIGEST};
    let set = std::path::PathBuf::from(brain_testutil::repo_path("crates/data/tests/fixtures/training-records/v2"));
    assert_eq!(contract_digest(&set).unwrap(), CONTRACT_DIGEST, "the example set changed: update CONTRACT_DIGEST (and the version, when an accepted record is now refused)");
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(set.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["contract"], CONTRACT);

    let copy = brain_testutil::Scratch::dir("contract-digest-copy");
    for entry in walk(&set) {
        let to = copy.join(entry.strip_prefix(&set).unwrap());
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(&entry, &to).unwrap();
    }
    std::fs::write(copy.join("README.md"), "prose is not the contract").unwrap();
    assert_eq!(contract_digest(&copy).unwrap(), CONTRACT_DIGEST, "a copy digests the same; its README is prose");
    std::fs::write(copy.join("chat.jsonl"), "{}\n").unwrap();
    assert_ne!(contract_digest(&copy).unwrap(), CONTRACT_DIGEST);
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(dir).unwrap().flat_map(|e| {
        let path = e.unwrap().path();
        if path.is_dir() { walk(&path) } else { vec![path] }
    }).collect()
}
