// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data-parallel preference optimization and
// distillation of chat models across the cards of a server for its clients. If
// your team needs expertise in scaling alignment fine-tunes out, you can procure
// our services by sending an email to info@swedishembedded.com.

#![cfg(feature = "study")]

//! Preference optimization (a loss per pair) and distillation (a KL per token to the teacher) as data-parallel ranks
//! through the SDK, on the CPU backend: two in-process ranks of one pair or record each train what one rank
//! accumulating the same two a step trains, so the ranks really train disjoint shards and their gradients are
//! reduced; the ranks end byte-identical, and the report says how the run was spread.

use std::path::{Path, PathBuf};

use brain::{DataParallel, DistillFineTune, DistillTeacher, FineTuneStatus, PreferenceFineTune};

use crate::chat_fixture::{self, common::cpu};

const PAIRS: &[(&str, &str, &str)] = &[("cab", "bad", "wig"), ("fed", "deaf", "vow"), ("ace", "face", "jug"), ("bag", "gab", "mop")];
const RECORDS: &[(&str, &str)] = &[("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")];
const STEPS: u32 = 3;
const LR: f32 = 5e-2;
/// How far, as a fraction of one step at the peak rate, the ranks may end from one rank accumulating their records:
/// the summation order of the reduced gradient differs, nothing else (`model::conformance_zero`'s bound).
const STEP_FRACTION: f32 = 1e-2;

fn write_pairs(dir: &Path) -> PathBuf {
    let path = dir.join("pairs.jsonl");
    let line = |(u, c, r): &(&str, &str, &str)| serde_json::json!({"prompt": [{"role": "user", "content": u}], "chosen": {"role": "assistant", "content": c}, "rejected": {"role": "assistant", "content": r}}).to_string();
    std::fs::write(&path, PAIRS.iter().map(line).collect::<Vec<_>>().join("\n")).unwrap();
    path
}

fn write_records(dir: &Path) -> PathBuf {
    let path = dir.join("records.jsonl");
    let line = |(u, a): &(&str, &str)| serde_json::json!({"messages": [{"role": "user", "content": u, "train": false}, {"role": "assistant", "content": a, "train": true}], "tools": []}).to_string();
    std::fs::write(&path, RECORDS.iter().map(line).collect::<Vec<_>>().join("\n")).unwrap();
    path
}

type Tensors = std::collections::HashMap<String, Vec<f32>>;

fn tensors(adapter: &Path) -> Tensors {
    checkpoint::st::load_safetensors(adapter.to_str().unwrap()).unwrap().tensors.into_iter().collect()
}

/// The largest elementwise difference between two adapters, and its tensor.
fn worst(a: &Tensors, b: &Tensors) -> (f32, String) {
    assert_eq!(a.len(), b.len(), "the adapters hold the same tensors");
    let mut out = (0f32, String::new());
    for (name, x) in a {
        let diff = x.iter().zip(&b[name]).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        if diff > out.0 {
            out = (diff, name.clone());
        }
    }
    out
}

/// What a rank ran to: its adapter file and its report's `parallel` object.
struct RankRun {
    adapter: PathBuf,
    parallel: serde_json::Value,
}

/// `train(rank, out)` on each of two in-process ranks, each on its own thread.
fn two_ranks(dir: &Path, train: impl Fn(DataParallel, &Path) -> RankRun + Sync) -> Vec<RankRun> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = DataParallel::in_process(2)
            .expect("a two-rank group")
            .into_iter()
            .map(|rank| {
                let train = &train;
                scope.spawn(move || {
                    let out = dir.join(format!("rank{}", rank.rank()));
                    train(rank, &out)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank's thread")).collect()
    })
}

/// The equivalence and the report every objective that trains as ranks is held to.
fn check(objective: &str, ranks: &[RankRun], accumulating: &Path, one_record: &Path) {
    let bytes = |p: &Path| std::fs::read(p).unwrap();
    assert_eq!(bytes(&ranks[0].adapter), bytes(&ranks[1].adapter), "{objective}: the ranks' adapters are byte-identical");
    let (diff, tensor) = worst(&tensors(&ranks[0].adapter), &tensors(accumulating));
    assert!(diff / LR <= STEP_FRACTION, "{objective}: two ranks of one record each end {diff:.2e} ({tensor}) from one rank accumulating the same two a step, {:.2e} of a step", diff / LR);
    let (apart, _) = worst(&tensors(&ranks[0].adapter), &tensors(one_record));
    assert!(apart / LR > 10.0 * STEP_FRACTION, "{objective}: the ranks trained more than one record a step (they differ from a one-record run by {apart:.2e} only)");
    let cpu_rank = |rank: usize| serde_json::json!({"rank": rank, "device": "cpu", "name": null});
    for rank in ranks {
        assert_eq!(rank.parallel, serde_json::json!({"world_size": 2, "ranks": [0, 2], "devices": [cpu_rank(0), cpu_rank(1)], "collective": "host", "transport": "in_process", "network": null, "sharding": "replicated"}), "{objective}");
    }
}

#[test]
fn preference_ranks_train_what_one_rank_accumulating_their_pairs_trains() {
    let model = chat_fixture::chat_model_dir("pref-data-parallel");
    let pairs = write_pairs(&model);
    // seed: runs that must be compared with each other need the same stream.
    let tune = |out: &Path| PreferenceFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&pairs).out_dir(out).rank(2).steps(STEPS).lr(LR).seed(7).device(cpu());
    let ranks = two_ranks(&model, |rank, out| {
        let outcome = tune(out).data_parallel(rank).run().expect("the rank trains");
        assert_eq!((outcome.status, outcome.steps_completed), (FineTuneStatus::Completed, STEPS));
        RankRun { adapter: outcome.adapter.expect("an adapter"), parallel: serde_json::to_value(&outcome.parallel).unwrap() }
    });
    let accumulating = tune(&model.join("accumulating")).grad_accum(2).run().expect("one rank accumulating two pairs");
    let one_record = tune(&model.join("one")).run().expect("one rank of one pair a step");
    check("preference", &ranks, &accumulating.adapter.unwrap(), &one_record.adapter.unwrap());
    let unseeded = PreferenceFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&pairs).out_dir(model.join("unseeded")).steps(1).device(cpu());
    let err = unseeded.data_parallel(DataParallel::in_process(1).unwrap().remove(0)).run().expect_err("a rank needs the shared seed");
    assert!(err.to_string().contains("seed"), "{err}");
}

#[test]
fn distillation_ranks_train_what_one_rank_accumulating_their_records_trains() {
    let model = chat_fixture::common::TempDir::new("distill-data-parallel");
    // seed: a fixture checkpoint.
    brain::toy::write_toy_chat_checkpoint(&model, 5).unwrap();
    let records = write_records(&model);
    // seed: runs that must be compared with each other need the same stream.
    let tune = |out: &Path| DistillFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&records).teacher(DistillTeacher::Base).top_k(4).out_dir(out).rank(4).steps(STEPS).lr(LR).seed(7).device(cpu());
    // A fresh adapter is the base, so a base teacher gives no gradient: the student starts from a trained adapter.
    // seed: the starting adapter is a fixture.
    let start = brain::ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&records).out_dir(model.join("start")).rank(4).steps(10).lr(2e-2).seed(3).device(cpu()).run().expect("a starting adapter").adapter.unwrap();
    let ranks = two_ranks(&model, |rank, out| {
        let outcome = tune(out).continue_from(&start).data_parallel(rank).run().expect("the rank trains");
        assert_eq!((outcome.status, outcome.steps_completed), (FineTuneStatus::Completed, STEPS));
        RankRun { adapter: outcome.adapter.expect("an adapter"), parallel: serde_json::to_value(&outcome.parallel).unwrap() }
    });
    let accumulating = tune(&model.join("accumulating")).continue_from(&start).grad_accum(2).run().expect("one rank accumulating two records");
    let one_record = tune(&model.join("one")).continue_from(&start).run().expect("one rank of one record a step");
    check("distill", &ranks, &accumulating.adapter.unwrap(), &one_record.adapter.unwrap());
}
