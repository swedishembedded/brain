// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data pipelines that check a dataset against the
// model that will train on it before any accelerator is claimed, for its clients. If your
// team needs expertise in preparing and validating fine-tuning data, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The `check_dataset` action through the executor on the toy `qwen3` base: the report, the per-record lengths a
//! caller pages through, and the refusal of a record the trainer would refuse, all without a card.

use serde_json::{json, Value};
use trainjob::dryrun::{spec, ACTION};

use crate::support::{assert_reservation_returned, data_digest, plain, record, rig_qwen3, run_action, work_is_clean, Frames, BASE};

fn check(rig: &crate::support::Rig, params: Value, dataset: Vec<u8>) -> Result<Value, String> {
    run_action(rig, ACTION, &spec(), params, &[("dataset", dataset)], &Frames::default()).map(|o| o.outputs)
}

#[test]
fn a_dry_run_reports_the_dataset_and_lists_each_records_length_a_page_at_a_time() {
    let rig = rig_qwen3("dryrun", 16 << 30);
    let long = record("cab cab cab cab cab cab cab cab", "bad bad bad bad bad bad bad bad", None);
    let dataset = [String::from_utf8(plain()).unwrap(), long].join("\n").into_bytes();

    let whole = check(&rig, json!({"base": BASE, "per_record": true}), dataset.clone()).expect("the report");
    assert_eq!((whole["records"].as_u64(), whole["valid_records"].as_u64(), whole["invalid_total"].as_u64()), (Some(5), Some(5), Some(0)));
    let lengths: Vec<(u64, u64)> = whole["per_record"].as_array().unwrap().iter().map(|p| (p[0].as_u64().unwrap(), p[1].as_u64().unwrap())).collect();
    assert_eq!(lengths.iter().map(|(line, _)| *line).collect::<Vec<_>>(), [1, 2, 3, 4, 5]);
    assert!(lengths[4].1 > lengths[0].1, "the longer record is longer in tokens: {lengths:?}");
    assert_eq!(whole["per_record_next"], Value::Null);
    assert_eq!(whole["tokens"]["max"].as_u64(), lengths.iter().map(|(_, t)| *t).max(), "no teacher here: the longest example is the longest record");

    // Pages cover the file without gaps or overlap, and the last one says there is no next.
    let mut paged = Vec::new();
    let mut offset = 0;
    loop {
        let page = check(&rig, json!({"base": BASE, "per_record": true, "offset": offset, "limit": 2}), dataset.clone()).expect("a page");
        paged.extend(page["per_record"].as_array().unwrap().iter().map(|p| (p[0].as_u64().unwrap(), p[1].as_u64().unwrap())));
        match page["per_record_next"].as_u64() {
            Some(next) => offset = next,
            None => break,
        }
    }
    assert_eq!(paged, lengths);

    // Without `per_record` the report carries no list: a large dataset's report stays small.
    let plain_report = check(&rig, json!({"base": BASE}), dataset).expect("the report");
    assert!(plain_report.get("per_record").is_none());
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

#[test]
fn a_dry_run_names_the_records_the_trainer_refuses_and_the_examples_longer_than_the_row() {
    let rig = rig_qwen3("dryrun-refused", 16 << 30);
    let digest = data_digest(&rig, &[plain(), b"\nnot json\n".to_vec()].concat());
    let report = run_action(&rig, ACTION, &spec(), json!({"base": BASE, "dataset_digest": digest.hex(), "max_block": 64, "per_record": true}), &[], &Frames::default()).expect("the report").outputs;
    assert_eq!((report["valid_records"].as_u64(), report["invalid_total"].as_u64()), (Some(4), Some(1)));
    assert_eq!(report["invalid"][0]["line"], 5);
    assert_eq!(report["per_record"].as_array().unwrap().len(), 4, "a refused record has no length");
    assert!(work_is_clean(&rig.work_root));
}

#[test]
fn a_timeline_dataset_is_checked_line_by_line_without_a_base() {
    let rig = rig_qwen3("dryrun-timeline", 16 << 30);
    let (subjects, _) = horizon::synthetic::population(5, 1);
    let mut text: String = subjects.iter().map(|s| serde_json::to_string(s).unwrap() + "\n").collect();
    text.push_str("{\"not\": \"a subject\"}\n");
    let report = check(&rig, json!({"kind": "timeline"}), text.into_bytes()).expect("the report");
    assert_eq!((report["records"].as_u64(), report["valid_records"].as_u64(), report["invalid_total"].as_u64()), (Some(6), Some(5), Some(1)));
    assert_eq!(report["invalid"][0]["line"], 6);
    let no_base = check(&rig, json!({}), plain()).unwrap_err();
    assert!(no_base.contains("`base` is needed"), "{no_base}");
}
