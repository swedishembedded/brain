// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reproducing the held-out
// records a fine-tune reported its score on, for its clients. If your team
// needs expertise in reproducible evaluation splits of language-model
// fine-tunes then you can procure our services by sending an email to
// info@swedishembedded.com.

// The emitter is the `study` surface.
#![cfg(feature = "study")]

//! SPEC: `brain::emit_held_out_records` - the held-out split of a chat dataset,
//! as the ids and masks the trainer encodes, on a toy `qwen35` model (CPU).
//!
//! * Whole groups only, decided by the file alone: the same dataset emits the
//!   same bytes twice, and no group is on both sides of the split.
//! * Every emitted line is a record the model can score: ids and mask of equal
//!   length, with the tokens the trainer's own `ChatSample::encode` gives that
//!   record.
//! * A dataset whose records are all one group is refused, not divided.

use std::path::{Path, PathBuf};

use brain::emit_held_out_records;

#[path = "chat_fixture/common.rs"]
mod common;
use common::qwen35_model_dir;

/// A chat dataset whose records name a group and carry an experience, which is
/// what makes them examinable: `groups` groups of `per_group` records each.
fn grouped_dataset(dir: &Path, groups: usize, per_group: usize) -> PathBuf {
    let path = dir.join("records.jsonl");
    let mut rows = Vec::new();
    for g in 0..groups {
        for n in 0..per_group {
            rows.push(
                serde_json::json!({
                    "messages": [
                        {"role": "user", "content": format!("question {g}-{n}"), "train": false},
                        {"role": "assistant", "content": format!("answer {g}-{n}"), "train": true}
                    ],
                    "tools": [],
                    "metadata": { "experiences": [format!("blake3:{g:064x}")], "group": format!("g{g}"), "view": "sft-step", "objective": "sft" }
                })
                .to_string(),
            );
        }
    }
    std::fs::write(&path, rows.join("\n")).unwrap();
    path
}

/// One emitted record's ids and mask.
fn emitted(path: &Path) -> Vec<(Vec<u32>, Vec<bool>)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).expect("an emitted line is JSON");
            let ints = |key: &str| v[key].as_array().unwrap_or_else(|| panic!("no {key} array")).clone();
            let ids: Vec<u32> = ints("ids").iter().map(|x| x.as_u64().expect("an id") as u32).collect();
            let mask: Vec<bool> = ints("mask").iter().map(|x| x.as_bool().expect("a mask entry")).collect();
            (ids, mask)
        })
        .collect()
}

/// SPEC: the split is whole groups, deterministic, and every line is the
/// trainer's own encoding of a held-out record.
#[test]
fn the_held_out_split_is_whole_groups_encoded_as_the_trainer_encodes_them() {
    let model = qwen35_model_dir("emit-held-out");
    // 12 groups of one record: a tenth is one group, grown to eight within a
    // quarter, so the rule holds out the quarter's worth from the file's end.
    let dataset = grouped_dataset(&model, 12, 1);
    let out = model.join("held-out.jsonl");
    let once = emit_held_out_records(model.join("model.safetensors").to_str().unwrap(), &dataset, &out, false, None).expect("the dataset emits");
    assert_eq!(once.held_out, 3, "a quarter of twelve one-record groups");
    assert_eq!(once.emitted, 3);
    let lines = emitted(&out);
    assert_eq!(lines.len(), 3);
    for (ids, mask) in &lines {
        assert_eq!(ids.len(), mask.len());
        let supervised = mask.iter().enumerate().skip(1).filter(|(_, &m)| m).count();
        assert!(supervised > 0, "an emitted record supervises nothing: {ids:?} {mask:?}");
    }
    // The manifest names the split, and the held-out and fitted groups are
    // disjoint - the property the whole-group rule exists for.
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&once.manifest).expect("the manifest is written")).unwrap();
    let groups: Vec<String> = manifest["held_out"]["group_names"].as_array().expect("group names").iter().map(|g| g.as_str().unwrap().to_string()).collect();
    assert_eq!(groups.len(), 3);
    assert_eq!(manifest["held_out"]["train_groups"], 9);
    assert_eq!(manifest["held_out"]["records"], 3);
    assert_eq!(manifest["emitted"]["positions"].as_u64().unwrap() as usize, once.positions);
    assert!(once.positions == lines.iter().map(|(_, m)| m.iter().skip(1).filter(|&&s| s).count()).sum::<usize>());

    // The same dataset emits the same bytes: nothing in the rule reads a seed,
    // the clock or the machine.
    let again = emit_held_out_records(model.join("model.safetensors").to_str().unwrap(), &dataset, &model.join("again.jsonl"), false, None).expect("the dataset emits twice");
    assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(&again.out).unwrap(), "the emitted bytes are the same");
    assert_eq!((again.held_out, again.positions), (once.held_out, once.positions));
}

/// SPEC: a dataset whose records are one group is refused, never divided.
#[test]
fn one_group_is_refused_rather_than_divided() {
    let model = qwen35_model_dir("emit-one-group");
    let dataset = grouped_dataset(&model, 1, 8);
    let err = emit_held_out_records(model.join("model.safetensors").to_str().unwrap(), &dataset, &model.join("held-out.jsonl"), false, None).unwrap_err().to_string();
    assert!(err.contains("no whole group"), "{err}");
    assert!(!model.join("held-out.jsonl").exists(), "nothing is written when the split is refused");
}
