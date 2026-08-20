// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end test of `brain qwen3 finetune --lora ...`: the "simple command
//! to fully retrain and overwrite the lora checkpoint" from the training
//! benchmark's Definition of Done. Runs the actual compiled
//! `brain` binary against a real Qwen3-0.6B checkpoint symlinked into a
//! scratch model store, and asserts the named adapter lands exactly where
//! `brain_modelstore::Store` expects to find it -- proving the CLI, the
//! store layout, and the training core (`qwen3::finetune`) are
//! actually wired together, not just individually correct.
//!
//! Gated on `QWEN3_DIR` (a real Qwen3-0.6B checkpoint dir with
//! `qwen.brain.safetensors` + `tokenizer.json` + `tokenizer_config.json`,
//! same convention as `crates/data/tests/chat_sample_encode.rs` /
//! `chat_template_cross_check.rs`) -- skips loudly rather than failing when
//! unset, since normal CI has no multi-GB checkpoint on disk:
//!
//! ```text
//! QWEN3_DIR=/path/to/qwen3-0.6b \
//!   cargo test --release -p brain-cli --test qwen_lora_finetune -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

fn qwen3_dir() -> Option<PathBuf> {
    let d = std::env::var("QWEN3_DIR").ok().map(PathBuf::from)?;
    d.join("qwen.brain.safetensors").is_file().then_some(d)
}

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("brain");
    p
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-cli-qwen-lora-finetune-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(unix)]
fn link(src: &Path, dst: &Path) {
    std::os::unix::fs::symlink(src, dst).unwrap_or_else(|e| panic!("symlink {} -> {}: {e}", src.display(), dst.display()));
}

fn write_dataset(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    // Minimal but real generic-messages-v2-shaped conversations: every
    // message carries an explicit `train` flag (chat.rs's WireMessage has no
    // default for it -- an omitted boundary is a hard error, not an implicit
    // default).
    // Several short pairs, not two -- the packed+rendered total must clear
    // `--block` (see crates/model/src/train.rs's `too_short` check: a split
    // with fewer tokens than block_size has no valid sampling window at all).
    let pairs = [("2+2?", "4"), ("3+3?", "6"), ("5+5?", "10"), ("7+1?", "8"), ("9+9?", "18"), ("6+2?", "8"), ("4+4?", "8"), ("8+1?", "9")];
    let lines: Vec<String> = pairs
        .iter()
        .map(|(q, a)| {
            format!(
                r#"{{"messages":[{{"role":"system","content":"You are terse.","train":false}},{{"role":"user","content":"{q}","train":false}},{{"role":"assistant","content":"{a}","train":true}}],"tools":[]}}"#
            )
        })
        .collect();
    std::fs::write(dir.join("train.jsonl"), lines.join("\n") + "\n").unwrap();
    std::fs::write(dir.join("validation.jsonl"), lines[..2].join("\n") + "\n").unwrap();
}

#[test]
#[ignore]
fn finetune_lora_writes_a_named_adapter_the_store_can_resolve() {
    let Some(qwen_dir) = qwen3_dir() else {
        brain_testutil::skip("QWEN3_DIR unset (or missing qwen.brain.safetensors)");
        return;
    };

    let scratch = tmp("run");
    let store_dir = scratch.join("models");
    let repo_dir = store_dir.join("Qwen").join("Qwen3-0.6B");
    std::fs::create_dir_all(&repo_dir).unwrap();
    link(&qwen_dir.join("qwen.brain.safetensors"), &repo_dir.join("model.brain.safetensors"));
    link(&qwen_dir.join("tokenizer.json"), &repo_dir.join("tokenizer.json"));
    link(&qwen_dir.join("tokenizer_config.json"), &repo_dir.join("tokenizer_config.json"));

    let dataset_dir = scratch.join("dataset");
    write_dataset(&dataset_dir);

    let out = Command::new(bin())
        .args([
            "qwen3",
            "finetune",
            "--lora",
            "4",
            "--weights",
            "Qwen/Qwen3-0.6B",
            "--adapter",
            "test-owner/test-adapter",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--models-dir",
            store_dir.to_str().unwrap(),
            "--steps",
            "2",
            "--batch",
            "1",
            "--block",
            "48",
            "--dataset-id",
            "test-dataset",
        ])
        .env("BRAIN_DEVICE", "cpu")
        .output()
        .expect("run brain qwen3 finetune --lora");
    assert!(
        out.status.success(),
        "brain qwen3 finetune --lora failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let adapter_path = repo_dir.join("adapters").join("test-owner").join("test-adapter").join("latest").join("adapter.brain.safetensors");
    assert!(adapter_path.is_file(), "adapter file not written to the expected store path: {}", adapter_path.display());

    // Not just a file on disk in the right place -- the store itself must
    // resolve the full named ref, exactly what the serving path relies on.
    let store = brain_modelstore::Store::new(&store_dir);
    let r = brain_modelref::ModelRef::parse("Qwen/Qwen3-0.6B:test-owner:test-adapter:latest").unwrap();
    let found = store.local(&r).expect("store must resolve the freshly trained adapter");
    assert_eq!(found.adapter.as_deref(), Some(adapter_path.as_path()));
    let card = found.card.expect("save_adapter must write a ModelCard");
    assert_eq!(card.id, "Qwen/Qwen3-0.6B:test-owner:test-adapter:latest");
    let adapter_meta = card.adapter.expect("card.adapter must be set");
    assert_eq!(adapter_meta.rank, Some(4));
    assert_eq!(adapter_meta.dataset_id.as_deref(), Some("test-dataset"));

    // Retraining the SAME --adapter must OVERWRITE the tag, not fail or
    // create a second copy -- the "fully retrain and overwrite" requirement.
    let out2 = Command::new(bin())
        .args([
            "qwen3",
            "finetune",
            "--lora",
            "4",
            "--weights",
            "Qwen/Qwen3-0.6B",
            "--adapter",
            "test-owner/test-adapter",
            "--dataset",
            dataset_dir.to_str().unwrap(),
            "--models-dir",
            store_dir.to_str().unwrap(),
            "--steps",
            "2",
            "--batch",
            "1",
            "--block",
            "48",
        ])
        .env("BRAIN_DEVICE", "cpu")
        .output()
        .expect("run brain qwen3 finetune --lora a second time");
    assert!(out2.status.success(), "retrain-and-overwrite failed: {}", String::from_utf8_lossy(&out2.stderr));
    assert!(adapter_path.is_file(), "adapter file must still be there after overwrite");
}
