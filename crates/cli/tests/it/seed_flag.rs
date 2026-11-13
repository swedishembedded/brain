// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reproducible, auditable
// randomness in ML pipelines for its clients. If your team needs expertise in
// seeding discipline for training and generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Process-level contract of every seeded `brain` verb: a `--seed` that is not
//! a seed is refused (exit 2, the flag named) and never replaced by a default,
//! and an absent one draws a random seed that is printed to stderr so the run
//! can be reproduced. Each case stops at a usage error right after the seed is
//! resolved (no weights, no device), so it checks the flag handling and nothing
//! heavier.

use std::process::{Command, Stdio};

/// Run `brain <verb...>`; returns (exit code, stderr).
fn brain(verb: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_brain"))
        .args(verb)
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("run brain {verb:?}: {e}"));
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// Verbs that resolve `--seed` before doing any work, so a malformed one stops
/// them at once. The flag is the verb's own, never a shared default.
const SEEDED_VERBS: &[&[&str]] = &[
    &["perf", "run", "latency"],
    &["bench"],
    &["bench", "eval"],
    &["bench", "scale"],
    &["data", "gen", "calculator"],
    &["data", "gen", "pong"],
    &["federated", "train-expert"],
    &["forecast", "predict"],
    &["forecast", "compare"],
    &["glmdsa", "train", "--data", "d"],
    &["glmdsa", "generate", "--weights", "w"],
    &["glmdsa", "eval", "--weights", "w", "--data", "d"],
    &["gpt2", "train", "--data", "d"],
    &["gpt2", "generate", "--weights", "w"],
    &["gpt2", "eval", "--weights", "w", "--data", "d"],
    &["lfm2", "finetune", "--data", "d", "--save", "s"],
    &["lfm2", "eval", "--data", "d"],
    &["qwen3", "train", "--data", "d", "--save", "s"],
    &["qwen3", "finetune", "--data", "d", "--save", "s"],
    &["qwen3", "generate", "--weights", "w"],
    &["qwen3", "lora_train", "--weights", "w"],
    &["qwen3", "toolcall_gen", "--out", "o"],
    &["qwen3", "toolcall_eval", "--weights", "w"],
    &["qwen35", "infer"],
    &["qwen35", "finetune"],
    &["qwen35moe", "generate", "--weights", "w"],
    &["serve"],
    &["flux2", "text2image", "--prompt", "p"],
    &["flux2", "lora_train", "--data", "d", "--save", "s"],
    &["wan", "t2v", "--prompt", "p"],
    &["wan", "lora_train", "--data", "d", "--save", "s"],
    &["ltxv", "t2v", "--prompt", "p"],
    &["qwen3tts", "finetune"],
    &["yolov8", "train"],
    &["zipdepth", "train", "--save", "s"],
    &["toypid", "train", "--save", "s"],
    &["diamond", "finetune", "--data", "d", "--save", "s"],
    &["diamond", "rollout", "--model", "fake"],
    &["diamond", "bench", "--model", "fake"],
];

/// Verbs that stop at a usage error with no further arguments, without
/// loading a model, opening a device or writing anything.
const USAGE_STOPPED_VERBS: &[&[&str]] = &[
    &["perf", "run", "latency"],
    &["federated", "train-expert"],
];

#[test]
fn a_malformed_seed_is_refused_naming_the_flag() {
    let mut problems = Vec::new();
    for verb in SEEDED_VERBS {
        for bad in ["x", "-1", "1.5"] {
            let args: Vec<&str> = verb.iter().copied().chain(["--seed", bad]).collect();
            let (code, err) = brain(&args);
            if code != 2 || !err.contains("--seed") || err.contains("using random seed") {
                problems.push(format!("brain {args:?}: exit {code}: {}", err.lines().next().unwrap_or("")));
            }
        }
    }
    assert!(problems.is_empty(), "a malformed --seed must exit 2 naming the flag:\n{}", problems.join("\n"));
}

#[test]
fn an_absent_seed_is_drawn_and_printed_so_the_run_reproduces() {
    let notice = |err: &str| err.lines().find(|l| l.contains("using random seed")).map(str::to_owned);
    let mut problems = Vec::new();
    for verb in USAGE_STOPPED_VERBS {
        let (a, b) = (notice(&brain(verb).1), notice(&brain(verb).1));
        match (a, b) {
            (Some(a), Some(b)) if a != b && a.contains("pass --seed ") => {}
            (a, b) => problems.push(format!("brain {verb:?}: first {a:?}, second {b:?}")),
        }
    }
    assert!(problems.is_empty(), "an absent --seed must print a fresh seed and how to reproduce it:\n{}", problems.join("\n"));
}
