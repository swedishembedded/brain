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
    &["glmdsa", "train"],
    &["glmdsa", "infer", "--weights", "w"],
    &["glmdsa", "eval"],
    &["gpt2", "train"],
    &["gpt2", "infer", "--weights", "w"],
    &["gpt2", "eval"],
    &["lfm2", "finetune"],
    &["lfm2", "eval"],
    &["qwen3", "train"],
    &["qwen3", "infer"],
    &["qwen3", "finetune", "--lora", "8"],
    &["qwen3", "toolcall", "gen"],
    &["qwen3", "toolcall", "eval"],
    &["qwen35", "infer"],
    &["qwen35", "finetune"],
    &["qwen35moe", "infer", "--weights", "w"],
    &["serve"],
    &["flux2", "generate"],
    &["flux2", "finetune"],
    &["wan", "t2v"],
    &["wan", "finetune"],
    &["ltxv", "t2v"],
    &["qwen3tts", "finetune"],
    &["yolov8", "train"],
    &["zipdepth", "train"],
    &["toypid", "train"],
    &["diamond", "finetune", "--weights", "w", "--data", "d", "--out", "o"],
    &["diamond", "play"],
    &["diamond", "bench"],
];

/// Verbs that stop at a usage error with no further arguments, without
/// loading a model, opening a device or writing anything.
const USAGE_STOPPED_VERBS: &[&[&str]] = &[
    &["perf", "run", "latency"],
    &["federated", "train-expert"],
    &["glmdsa", "infer", "--weights", "w"],
    &["glmdsa", "eval"],
    &["gpt2", "infer", "--weights", "w"],
    &["gpt2", "eval"],
    &["lfm2", "eval"],
    &["qwen3", "infer"],
    &["qwen3", "train"],
    &["qwen3", "toolcall", "gen"],
    &["qwen3", "toolcall", "eval"],
    &["qwen35moe", "infer", "--weights", "w"],
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
