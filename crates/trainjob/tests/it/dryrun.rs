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

use crate::support::{assert_reservation_returned, data_digest, plain, record, rig_qwen3, rig_qwen3_on, run_action, work_is_clean, Frames, BASE};

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
    // Each entry splits its tokens into the prompt and the trained answer: `[line, tokens, prompt, trained, examples]`.
    for entry in whole["per_record"].as_array().unwrap() {
        let n: Vec<u64> = entry.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!((n.len(), n[2] + n[3], n[3] > 0, n[4]), (5, n[1], true, 1), "{entry}");
    }
    assert_eq!(whole["per_record_refused"], json!([]));
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

/// `brain serve --training --device gpu3` refused `check_dataset` with "brain/train[host] (256 MiB) exceeds every
/// device's budget": the check is host work, and naming only cards left it nowhere to go. It runs on the host
/// whatever the server was given: one card, several, or the CPU alone.
#[test]
fn a_dry_run_runs_on_the_host_whatever_devices_the_server_was_given() {
    for (devices, gpus, cpu_compute) in [("gpu3", &[3][..], false), ("gpu0,gpu1", &[0, 1][..], false), ("cpu", &[][..], true)] {
        let rig = rig_qwen3_on(&format!("dryrun-devices-{}", gpus.len()), 16 << 30, gpus, cpu_compute);
        let report = check(&rig, json!({"base": BASE}), plain()).unwrap_or_else(|e| panic!("--device {devices}: {e}"));
        assert_eq!(report["valid_records"].as_u64(), Some(4), "--device {devices}: {report}");
        assert_reservation_returned(&rig);
    }
}

#[test]
fn a_dry_run_names_the_records_the_trainer_refuses_and_the_examples_longer_than_the_row() {
    let rig = rig_qwen3("dryrun-refused", 16 << 30);
    let digest = data_digest(&rig, &[plain(), b"\nnot json\n".to_vec()].concat());
    let report = run_action(&rig, ACTION, &spec(), json!({"base": BASE, "dataset_digest": digest.hex(), "max_block": 64, "per_record": true}), &[], &Frames::default()).expect("the report").outputs;
    assert_eq!((report["valid_records"].as_u64(), report["invalid_total"].as_u64()), (Some(4), Some(1)));
    assert_eq!(report["invalid"][0]["line"], 5);
    assert_eq!((&report["invalid"][0]["rule"], &report["invalid"][0]["field"]), (&Value::Null, &Value::Null), "not JSON breaks no rollout rule");
    assert_eq!(report["per_record"].as_array().unwrap().len(), 4, "a refused record has no length");
    let refused = &report["per_record_refused"][0];
    assert_eq!((refused["line"].as_u64(), refused["rule"].is_null(), refused["error"].as_str().map(str::is_empty)), (Some(5), true, Some(false)), "the page lists the refusal in place of a length: {report}");
    assert!(work_is_clean(&rig.work_root));

    // A refusal of the training-record contract names its rule and field on the wire.
    let mut short = serde_json::from_str::<Value>(&record("cab", "bad", Some(1.0))).unwrap();
    short["messages"][1]["behavior_logprobs"] = json!([-0.5, -0.5]);
    let report = check(&rig, json!({"base": BASE}), short.to_string().into_bytes()).expect("the report");
    assert_eq!((&report["invalid"][0]["rule"], &report["invalid"][0]["field"]), (&json!("behavior_logprob_count"), &json!("messages[1].behavior_logprobs")), "{report}");
}

/// The dry run reads the rules between rollout records as a run with its `truncated` setting would: a group whose
/// other rollout is truncated is a group of one when truncated rollouts are masked (the default) and a pair when they
/// are kept.
#[test]
fn a_dry_run_counts_a_truncated_rollout_in_its_group_only_where_the_run_would_keep_it() {
    let rig = rig_qwen3("dryrun-truncated", 16 << 30);
    let mut cut = serde_json::from_str::<Value>(&record("cab", "bad", Some(0.0))).unwrap();
    cut["truncated"] = json!(true);
    let dataset = [record("cab", "abc", Some(1.0)), cut.to_string()].join("\n").into_bytes();
    let masked = check(&rig, json!({"base": BASE}), dataset.clone()).expect("the report");
    assert_eq!((masked["invalid_total"].as_u64(), &masked["invalid"][0]["rule"]), (Some(1), &json!("group_of_one")), "{masked}");
    let kept = check(&rig, json!({"base": BASE, "truncated": "keep"}), dataset).expect("the report");
    assert_eq!(kept["invalid_total"].as_u64(), Some(0), "{kept}");
    assert_reservation_returned(&rig);
}

/// A feedback dataset is read as the feedback trainer reads it: each judged completion is one example, and a record
/// that is not one is refused by line.
#[test]
fn a_feedback_dataset_is_checked_as_the_trainer_reads_it() {
    let rig = rig_qwen3("dryrun-feedback", 16 << 30);
    let dataset = [crate::support::feedback(), b"\n{\"prompt\": [], \"completion\": {\"role\": \"assistant\", \"content\": \"x\"}, \"desirable\": true}".to_vec()].concat();
    let report = check(&rig, json!({"base": BASE, "kind": "feedback"}), dataset).expect("the report");
    assert_eq!((report["records"].as_u64(), report["valid_records"].as_u64(), report["examples"].as_u64(), report["invalid_total"].as_u64()), (Some(5), Some(4), Some(4), Some(1)), "{report}");
    assert_eq!(report["invalid"][0]["line"], 5);
    assert_reservation_returned(&rig);
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

/// A token that fires mid-check stops the timeline dataset check at the next line.
#[test]
fn a_timeline_check_stops_at_the_next_line() {
    use capability::{Invocation, Outcome, Progress};
    use capability_mock::cancellation::{certify, Bound};
    let (subjects, _) = horizon::synthetic::population(5, 1);
    let text: String = subjects.iter().map(|s| serde_json::to_string(s).unwrap() + "\n").collect();
    let run = |inv: &Invocation, progress: &mut dyn FnMut(Progress)| {
        inv.cancel.check()?;
        let report = trainjob::dryrun::timeline_report(&text, 64, &mut inv.keep_going(progress, 0, "record"))?;
        Ok(Outcome::new().set("records", report["records"].clone()))
    };
    certify("check_dataset timeline", &Invocation::new, &run, Bound::ONE_STEP);
}

/// The page is bounded by the action's spec, whatever the dataset: a window larger than the cap, or a negative one, is
/// refused as a typed error before anything is read, and a page never holds more than its `limit` records.
#[test]
fn a_page_larger_than_the_cap_is_refused_and_a_page_never_exceeds_its_limit() {
    let rig = rig_qwen3("dryrun-bound", 16 << 30);
    for limit in [json!(brain::dataset_report::MAX_RECORD_WINDOW + 1), json!(0), json!(-1)] {
        let refused = check(&rig, json!({"base": BASE, "per_record": true, "limit": limit}), plain()).unwrap_err();
        assert!(refused.contains("limit"), "{limit}: {refused}");
    }
    let mixed = [plain(), b"\nnot json\n".to_vec(), plain()].concat();
    let page = check(&rig, json!({"base": BASE, "per_record": true, "limit": 3}), mixed).expect("a page");
    assert_eq!(page["per_record"].as_array().unwrap().len() + page["per_record_refused"].as_array().unwrap().len(), 3);
    assert_eq!(page["per_record_next"], 3);
}
