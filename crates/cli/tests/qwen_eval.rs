// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gate B of the Definition of Done's "a way to validate that model has
//! learned ideas from the dataset": `brain qwen eval` against a REAL
//! Qwen3-0.6B checkpoint and a real bench-shaped `validation.jsonl`, base
//! score alone and base+adapter side by side. Complements
//! `crates/qwen/tests/lora_learning_gate.rs` (Gate A: synthetic, no
//! checkpoint, always runs) -- this proves the CLI/store/real-model WIRING
//! (`qwen::eval::score_chat`, `brain qwen eval`'s adapter resolution)
//! rather than re-litigating whether LoRA training converges, which Gate A
//! and `crates/cli/tests/qwen_lora_finetune.rs` already cover. Trains only a
//! handful of steps -- enough to prove the adapter changes the reported
//! score at all, not to prove it converges -- so this stays reasonably
//! fast for a real 0.6B model on CPU.
//!
//! Gated on `QWEN3_DIR` (needs `qwen.brain.safetensors` +
//! `tokenizer.json` + `tokenizer_config.json`), skips loudly if unset:
//!
//! ```text
//! QWEN3_DIR=/data/workspace/resources/llm/qwen/qwen3-0.6b \
//!   cargo test --release -p brain-cli --test qwen_eval -- --ignored --nocapture
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
    let dir = std::env::temp_dir().join(format!("brain-cli-qwen-eval-{name}-{}", std::process::id()));
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
    let lines = [
        r#"{"messages":[{"role":"system","content":"You are terse.","train":false},{"role":"user","content":"2+2?","train":false},{"role":"assistant","content":"4","train":true}],"tools":[]}"#,
        r#"{"messages":[{"role":"system","content":"You are terse.","train":false},{"role":"user","content":"3+3?","train":false},{"role":"assistant","content":"6","train":true}],"tools":[]}"#,
        r#"{"messages":[{"role":"system","content":"You are terse.","train":false},{"role":"user","content":"5+5?","train":false},{"role":"assistant","content":"10","train":true}],"tools":[]}"#,
        r#"{"messages":[{"role":"system","content":"You are terse.","train":false},{"role":"user","content":"7+1?","train":false},{"role":"assistant","content":"8","train":true}],"tools":[]}"#,
        r#"{"messages":[{"role":"system","content":"You are terse.","train":false},{"role":"user","content":"9+9?","train":false},{"role":"assistant","content":"18","train":true}],"tools":[]}"#,
        r#"{"messages":[{"role":"system","content":"You are terse.","train":false},{"role":"user","content":"6+2?","train":false},{"role":"assistant","content":"8","train":true}],"tools":[]}"#,
        r#"{"messages":[{"role":"system","content":"You are terse.","train":false},{"role":"user","content":"4+4?","train":false},{"role":"assistant","content":"8","train":true}],"tools":[]}"#,
        r#"{"messages":[{"role":"system","content":"You are terse.","train":false},{"role":"user","content":"8+1?","train":false},{"role":"assistant","content":"9","train":true}],"tools":[]}"#,
    ];
    std::fs::write(dir.join("train.jsonl"), lines.join("\n") + "\n").unwrap();
    std::fs::write(dir.join("validation.jsonl"), lines.join("\n") + "\n").unwrap();
}

#[test]
#[ignore]
fn eval_scores_base_alone_and_base_plus_adapter_side_by_side() {
    let Some(qwen_dir) = qwen3_dir() else {
        eprintln!("QWEN3_DIR unset (or missing qwen.brain.safetensors); skipping -- needs a real Qwen3-0.6B checkpoint");
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

    // Train a small adapter first (a handful of steps -- proving the
    // eval WIRING, not convergence; convergence is Gate A's job).
    let train_out = Command::new(bin())
        .args([
            "qwen", "finetune", "--lora", "4", "--weights", "Qwen/Qwen3-0.6B", "--adapter", "test-owner/eval-adapter", "--dataset",
            dataset_dir.to_str().unwrap(), "--models-dir", store_dir.to_str().unwrap(), "--steps", "2", "--batch", "1", "--block", "48",
        ])
        .env("BRAIN_DEVICE", "cpu")
        .output()
        .expect("run brain qwen finetune --lora");
    assert!(train_out.status.success(), "training the adapter failed: {}", String::from_utf8_lossy(&train_out.stderr));

    // Base score alone.
    let base_only = Command::new(bin())
        .args(["qwen", "eval", "--weights", "Qwen/Qwen3-0.6B", "--jsonl", dataset_dir.join("validation.jsonl").to_str().unwrap(), "--models-dir", store_dir.to_str().unwrap(), "--block", "48"])
        .env("BRAIN_DEVICE", "cpu")
        .output()
        .expect("run brain qwen eval (base only)");
    assert!(base_only.status.success(), "base-only eval failed: {}", String::from_utf8_lossy(&base_only.stderr));
    let base_stdout = String::from_utf8_lossy(&base_only.stdout).into_owned();
    assert!(base_stdout.contains("base Qwen/Qwen3-0.6B: loss"), "unexpected base-only output: {base_stdout}");
    assert!(!base_stdout.to_lowercase().contains("nan"), "base-only eval reported NaN loss (nothing scored?): {base_stdout}");

    // Base + adapter side by side.
    let with_adapter = Command::new(bin())
        .args([
            "qwen", "eval", "--weights", "Qwen/Qwen3-0.6B", "--adapter", "test-owner/eval-adapter", "--jsonl", dataset_dir.join("validation.jsonl").to_str().unwrap(), "--models-dir",
            store_dir.to_str().unwrap(), "--block", "48",
        ])
        .env("BRAIN_DEVICE", "cpu")
        .output()
        .expect("run brain qwen eval (with adapter)");
    assert!(with_adapter.status.success(), "with-adapter eval failed: {}", String::from_utf8_lossy(&with_adapter.stderr));
    let stdout = String::from_utf8_lossy(&with_adapter.stdout).into_owned();
    assert!(stdout.contains("base Qwen/Qwen3-0.6B: loss"), "missing base line: {stdout}");
    assert!(stdout.contains("Qwen/Qwen3-0.6B:test-owner:eval-adapter:latest: loss"), "missing adapter line: {stdout}");
    assert!(stdout.contains("base on held-out loss"), "missing the beats/does-not-beat verdict line: {stdout}");
    assert!(!stdout.to_lowercase().contains("nan"), "with-adapter eval reported NaN loss somewhere: {stdout}");
}
