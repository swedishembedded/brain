// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distributed training runs that account for
// their own communication cost for its clients. If your team needs expertise
// in distributed-training profiling or command-line validation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `--profile-collectives` on the served training actions that join a
//! distributed job (`model::job_params`): it is refused, naming the job
//! params it needs, when the run is not distributed, and accepted beside them,
//! so the run stops at the missing plan file. Both checks stop before a model
//! or a dataset is read, so the checkpoint and dataset named are placeholders.

use std::process::{Command, Output};

fn brain(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_brain")).args(args).output().expect("run the brain binary")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Placeholder files for the LoRA fine-tune's checkpoint, tokenizer and data,
/// named outright so nothing resolves from the model store.
fn qwen3_lora(dir: &std::path::Path) -> Vec<String> {
    std::fs::create_dir_all(dir).unwrap();
    let file = |name: &str| {
        let path = dir.join(name);
        std::fs::write(&path, b"{}").unwrap();
        path.to_string_lossy().into_owned()
    };
    vec!["qwen3".into(), "lora_train".into(), "--weights".into(), file("w.safetensors"), "--tokenizer".into(), file("tokenizer.json"), "--dataset".into(), file("train.jsonl")]
}

const JOB: [&str; 6] = ["--job-plan", "/nonexistent/brain-plan.json", "--job-rank", "0", "--job-secret", "/nonexistent/brain-job.secret"];

fn scratch(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("brain-profile-collectives-{tag}-{}", std::process::id()))
}

/// A profile of a one-process run is not the point: it is refused, naming the
/// job params it needs, before anything is trained.
#[test]
fn a_profile_without_a_job_plan_is_refused() {
    let dir = scratch("refused");
    let mut qwen = qwen3_lora(&dir);
    qwen.extend(["--profile-collectives".into(), "collectives.json".into()]);
    let save = dir.join("gpt.safetensors").to_string_lossy().into_owned();
    let gpt: Vec<String> = ["gpt2", "train", "--data", "data/calculator", "--save", &save, "--profile-collectives", "collectives.json"].map(String::from).to_vec();
    for argv in [gpt, qwen] {
        let out = brain(&argv.iter().map(String::as_str).collect::<Vec<_>>());
        let err = stderr(&out);
        assert_eq!(out.status.code(), Some(1), "{argv:?}: {err}");
        assert!(err.contains("profile_collectives") && err.contains("job_plan"), "{argv:?}: the refusal must name the param and its companion: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// With the job's params present the profile is accepted: the run stops at
/// the missing plan file, not at the profile.
#[test]
fn a_profile_with_a_job_plan_is_accepted() {
    let dir = scratch("accepted");
    let mut qwen = qwen3_lora(&dir);
    let save = dir.join("gpt.safetensors").to_string_lossy().into_owned();
    let mut gpt: Vec<String> = ["gpt2", "train", "--data", "data/calculator", "--save", &save].map(String::from).to_vec();
    for argv in [&mut gpt, &mut qwen] {
        argv.extend(JOB.iter().map(|s| s.to_string()));
        argv.extend(["--profile-collectives".into(), "collectives.json".into()]);
    }
    for argv in [gpt, qwen] {
        let out = brain(&argv.iter().map(String::as_str).collect::<Vec<_>>());
        let err = stderr(&out);
        assert_eq!(out.status.code(), Some(1), "{argv:?}: {err}");
        assert!(err.contains("/nonexistent/brain-plan.json"), "{argv:?}: the plan read must be what failed: {err}");
        assert!(!err.contains("profile_collectives"), "{argv:?}: the profile was not accepted: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
