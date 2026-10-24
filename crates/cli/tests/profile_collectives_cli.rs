// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distributed training runs that account for
// their own communication cost for its clients. If your team needs expertise
// in distributed-training profiling or command-line validation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `--profile-collectives` on the two verbs that take a job plan.
//!
//! Each verb must consume the flag itself (an unparsed flag is refused by the
//! leftovers check) and must refuse it when the run is not distributed. Both
//! checks stop before a model or a dataset is opened, so no fixture is needed.

use std::process::{Command, Output};

fn brain(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_brain")).args(args).output().expect("run the brain binary")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A profile of a one-process run is not the point: it is a named usage
/// error, before anything is trained.
#[test]
fn gpt2_train_refuses_a_profile_without_a_job_plan() {
    let out = brain(&["gpt2", "train", "data/calculator", "--profile-collectives", "collectives.json"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("--profile-collectives"), "the refusal must name the flag: {err}");
    assert!(err.contains("--job-plan"), "and the missing companion: {err}");
    assert!(!err.contains("unrecognised"), "the flag must not fall through to the leftovers check: {err}");
}

/// The same refusal on the LoRA fine-tune verb.
#[test]
fn qwen3_finetune_refuses_a_profile_without_a_job_plan() {
    let out = brain(&["qwen3", "finetune", "--lora", "4", "--profile-collectives", "collectives.json"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("--profile-collectives"), "the refusal must name the flag: {err}");
    assert!(err.contains("--job-plan"), "and the missing companion: {err}");
    assert!(!err.contains("ignoring unknown flag"), "the flag must not fall through to the unknown-flag notice: {err}");
}

/// With the job pair present the flag is accepted: the run stops at the
/// missing plan file, not at the flag. That proves each verb takes the flag
/// by name rather than leaving it to a leftovers or unknown-flag path.
#[test]
fn both_verbs_accept_the_flag_with_a_job_plan() {
    let gpt = brain(&[
        "gpt2",
        "train",
        "data/calculator",
        "--job-plan",
        "/nonexistent/brain-plan.json",
        "--job-rank",
        "0",
        "--profile-collectives",
        "collectives.json",
    ]);
    assert_eq!(gpt.status.code(), Some(1), "stderr: {}", stderr(&gpt));
    assert!(stderr(&gpt).contains("/nonexistent/brain-plan.json"), "the plan read must be what failed: {}", stderr(&gpt));
    assert!(!stderr(&gpt).contains("--profile-collectives"), "the flag was not accepted: {}", stderr(&gpt));

    let qwen = brain(&[
        "qwen3",
        "finetune",
        "--lora",
        "4",
        "--job-plan",
        "/nonexistent/brain-plan.json",
        "--job-rank",
        "0",
        "--profile-collectives",
        "collectives.json",
    ]);
    assert_eq!(qwen.status.code(), Some(1), "stderr: {}", stderr(&qwen));
    assert!(stderr(&qwen).contains("/nonexistent/brain-plan.json"), "the plan read must be what failed: {}", stderr(&qwen));
    assert!(!stderr(&qwen).contains("--profile-collectives"), "the flag was not accepted: {}", stderr(&qwen));
}
