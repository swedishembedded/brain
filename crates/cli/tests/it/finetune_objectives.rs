// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for aligning open-weight language models with a team's own judgements on
// the memory it actually has, for its clients. If your team needs expertise in preference optimization, KTO or
// distillation of language models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! `brain finetune --objective preference|feedback|distill` end to end through the compiled binary on the CPU
//! backend, over the toy chat checkpoint: each objective runs the SDK pipeline the training job of the HTTP API runs,
//! writes an adapter and the training record of its own regime, and a flag that objective does not read is a usage
//! error before anything is loaded.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The toy chat checkpoint (`brain::toy`) in a fresh scratch directory; returns the directory and the base file.
fn scratch(name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("brain-cli-finetune-objectives-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // seed: a fixture checkpoint.
    brain::toy::write_toy_chat_checkpoint(&dir, 5).unwrap();
    let base = dir.join("model.safetensors");
    (dir, base)
}

fn write(dir: &Path, name: &str, lines: &[serde_json::Value]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, lines.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n")).unwrap();
    path
}

fn pair(user: &str, chosen: &str, rejected: &str) -> serde_json::Value {
    serde_json::json!({"prompt": [{"role": "user", "content": user}], "chosen": {"role": "assistant", "content": chosen}, "rejected": {"role": "assistant", "content": rejected}, "metadata": {"source": "test"}})
}

fn judged(user: &str, answer: &str, desirable: bool) -> serde_json::Value {
    serde_json::json!({"prompt": [{"role": "user", "content": user}], "completion": {"role": "assistant", "content": answer}, "desirable": desirable})
}

fn chat(user: &str, answer: &str) -> serde_json::Value {
    serde_json::json!({"messages": [{"role": "user", "content": user, "train": false}, {"role": "assistant", "content": answer, "train": true}], "tools": []})
}

/// `brain finetune BASE ARGS... --device cpu --lora 4 --steps 4 --seed 7`.
fn finetune(base: &Path, args: &[&str]) -> Output {
    Command::new(brain_testutil::package_bin("brain"))
        .arg("finetune")
        .arg(base)
        .args(args)
        .args(["--device", "cpu", "--lora", "4", "--steps", "4", "--lr", "0.05", "--seed", "7"])
        .output()
        .expect("run brain finetune")
}

fn stdout(out: &Output) -> String {
    assert!(out.status.success(), "brain finetune failed:\nstdout: {}\nstderr: {}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The training record beside the adapter a run wrote under `out`.
fn record(out: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(out.join("training.json")).expect("a training record")).unwrap()
}

fn adapter(out: &Path) -> PathBuf {
    let path = out.join("adapter.safetensors");
    assert!(path.is_file(), "no adapter at {}", path.display());
    path
}

#[test]
fn a_preference_run_trains_the_named_pair_loss_and_scores_the_held_out_pairs() {
    let (dir, base) = scratch("preference");
    let train = write(&dir, "pairs.jsonl", &[pair("cab", "bad", "wig"), pair("fed", "deaf", "vow"), pair("ace", "face", "jug"), pair("bag", "gab", "mop")]);
    let held_out = write(&dir, "held.jsonl", &[pair("cad", "bead", "kiwi")]);
    let out = dir.join("pref");
    let said = stdout(&finetune(&base, &["--objective", "preference", "--loss", "simpo", "--gamma", "0.7", "--beta", "2", "--dataset", train.to_str().unwrap(), "--held-out", held_out.to_str().unwrap(), "--out", out.to_str().unwrap()]));
    assert!(said.contains("status: Completed, 4 steps") && said.contains("(simpo)") && said.contains("held-out: 1 pairs"), "{said}");
    adapter(&out);
    let record = record(&out);
    assert_eq!((record["regime"].as_str(), record["loss"].as_str()), (Some("dpo"), Some("simpo")), "{record}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_feedback_run_weighs_the_two_judgements_and_reports_their_log_ratios() {
    let (dir, base) = scratch("feedback");
    let lines: Vec<_> = [("cab", "bad", true), ("cab", "wig", false), ("fed", "deaf", true), ("fed", "vow", false), ("ace", "face", true), ("ace", "jug", false)].iter().map(|(u, a, d)| judged(u, a, *d)).collect();
    let train = write(&dir, "judged.jsonl", &lines);
    let out = dir.join("kto");
    let said = stdout(&finetune(&base, &["--objective", "feedback", "--desirable-weight", "1.5", "--undesirable-weight", "0.5", "--dataset", train.to_str().unwrap(), "--out", out.to_str().unwrap()]));
    assert!(said.contains("desirable") && said.contains("(3 examples)"), "{said}");
    adapter(&out);
    let record = record(&out);
    assert_eq!(record["regime"].as_str(), Some("feedback"), "{record}");
    assert_eq!((record["desirable_weight"].as_f64(), record["undesirable_weight"].as_f64()), (Some(1.5), Some(0.5)), "{record}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_distillation_is_taught_by_the_base_the_start_adapter_or_an_adapter_file() {
    let (dir, base) = scratch("distill");
    let records: Vec<_> = [("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")].iter().map(|(u, a)| chat(u, a)).collect();
    let data = write(&dir, "records.jsonl", &records);
    let data = data.to_str().unwrap();

    let first = dir.join("first");
    let said = stdout(&finetune(&base, &["--objective", "distill", "--top-k", "4", "--dataset", data, "--out", first.to_str().unwrap()]));
    assert!(said.contains("teacher: base (top 4 of 4 records)"), "{said}");
    let record_of = record(&first);
    assert_eq!((record_of["regime"].as_str(), record_of["teacher"].as_str(), record_of["top_k"].as_u64()), (Some("distill"), Some("base"), Some(4)), "{record_of}");
    let trained = adapter(&first);

    let by_file = dir.join("by-file");
    let said = stdout(&finetune(&base, &["--objective", "distill", "--teacher", &format!("adapter:{}", trained.display()), "--dataset", data, "--out", by_file.to_str().unwrap()]));
    assert!(said.contains("teacher: adapter"), "{said}");
    assert!(record(&by_file)["teacher_adapter"].as_str().is_some_and(|d| d.starts_with("sha256:")));

    let by_start = dir.join("by-start");
    let said = stdout(&finetune(&base, &["--objective", "distill", "--teacher", "start", "--continue", trained.to_str().unwrap(), "--dataset", data, "--out", by_start.to_str().unwrap()]));
    assert!(said.contains("teacher: start"), "{said}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_setting_the_objective_does_not_read_is_a_usage_error_before_anything_loads() {
    let missing = Path::new("/nonexistent/base");
    for (args, named) in [(&["--objective", "feedback", "--loss", "ipo"][..], "--objective preference"), (&["--objective", "preference", "--teacher", "base"], "--objective distill"), (&["--objective", "distill", "--weight-decay", "0.1"], "--weight-decay")] {
        let out = finetune(missing, args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains(named), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
}
