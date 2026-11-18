// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the records an orchestrator builds, for
// its clients. If your team needs expertise in training-data contracts between agent runtimes and trainers then you
// can procure our services by sending an email to info@swedishembedded.com.

#![cfg(feature = "study")]

//! The training-record contract, version 2, held to its fixtures (`crates/data/tests/fixtures/training-records/v2`):
//! `brain::dataset_report::check_dataset`, against the fixtures' own tokenizer and chat template, accepts every record
//! of the valid files and refuses each invalid one with the rule and the field its `manifest.json` names. A producer
//! copies the same files as its own test, so a change to what brain accepts is a change to these files and to the
//! contract's version.

use std::path::PathBuf;

use brain::dataset_report::{check_dataset, DatasetCheck, DatasetKind};

fn fixtures() -> PathBuf {
    PathBuf::from(brain_testutil::repo_path("crates/data/tests/fixtures/training-records/v2"))
}

#[test]
fn every_fixture_is_accepted_or_refused_by_the_rule_and_field_its_manifest_names() {
    let root = fixtures();
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["contract"], "training-records-v2");
    let model = root.join(manifest["tokenizer"].as_str().unwrap());
    let files = manifest["files"].as_array().unwrap();
    assert!(files.len() >= 11, "{manifest}");
    for file in files {
        let path = file["path"].as_str().unwrap();
        let kind = match file["kind"].as_str().unwrap() {
            "chat" => DatasetKind::Chat,
            "preference" => DatasetKind::Preference,
            "feedback" => DatasetKind::Feedback,
            other => panic!("{path}: unknown kind {other}"),
        };
        let check = DatasetCheck { kind, thinking: true, pack_answers: false, max_block: 512, list_records: None, truncated: Default::default() };
        let report = check_dataset(&model, root.join(path), &check, &mut || true).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!((report.records as u64, report.valid_records as u64), (file["records"].as_u64().unwrap(), file["valid_records"].as_u64().unwrap()), "{path}: {:?}", report.invalid);
        let refused: Vec<serde_json::Value> = report.invalid.iter().map(|r| serde_json::json!({"line": r.line, "rule": r.rule.map(|rule| rule.name()), "field": r.field})).collect();
        assert_eq!(serde_json::Value::Array(refused), file["refusals"], "{path}: {:?}", report.invalid);
        for refusal in &report.invalid {
            let field = refusal.field.as_deref().unwrap_or_default();
            assert!(refusal.error.contains(field), "{path}: the message names the field too: {}", refusal.error);
        }
    }
}

/// A rollout file's tokens are the fixture tokenizer's: every supervised answer of `chat_rollout.jsonl` carries
/// `token_ids` that round-trip and one `behavior_logprobs` value per token (the answer and its end-of-turn marker).
#[test]
fn the_rollout_fixture_supervises_the_tokens_its_records_carry() {
    let root = fixtures();
    let check = DatasetCheck { kind: DatasetKind::Chat, thinking: true, pack_answers: false, max_block: 512, list_records: None, truncated: Default::default() };
    let report = check_dataset(root.join("model"), root.join("chat_rollout.jsonl"), &check, &mut || true).unwrap();
    // Seven records; the ones with two and three answers are two and three examples.
    assert_eq!((report.valid_records, report.examples), (7, 10));
    let text = std::fs::read_to_string(root.join("chat_rollout.jsonl")).unwrap();
    for line in text.lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        for message in record["messages"].as_array().unwrap() {
            if let (Some(ids), Some(lps)) = (message["token_ids"].as_array(), message["behavior_logprobs"].as_array()) {
                assert_eq!(ids.len(), lps.len(), "{line}");
                assert_eq!(ids.len(), message["content"].as_str().unwrap().len() + 1, "one id per letter and the end-of-turn marker: {line}");
            }
        }
    }
}
