// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data-parallel training that a launcher can
// start as one process per rank over a plain network for its clients. If your
// team needs expertise in distributed training drivers or replica-consistency
// verification, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A real distributed GPT training job: two OS processes of this test binary
//! train one plan over loopback, each on its own shard of the batch stream,
//! and must end with bit-identical replicas.
//!
//! The children are this binary re-executed (`--ignored --exact rank_process`),
//! so the job handshake, the cross-process gradient reduction and the batch
//! sharding are exercised across process boundaries, not across threads.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use gpt2::model::Gpt;
use gpt2::train::TrainOpts;
use gpt2::GptConfig;
use model::{GradSync, JobContext, JobPlan, Model};

/// The name libtest filters to when this binary is re-executed as a rank.
const CHILD: &str = "rank_process";

/// Both ranks train the same model on different batches, so the job is small
/// enough to run in a test and large enough for the reduction to matter.
const STEPS: u32 = 4;
const BATCH: u32 = 2;
const BLOCK: u32 = 16;
const D_MODEL: u32 = 32;

fn test_config() -> GptConfig {
    GptConfig { vocab: 0, block_size: BLOCK, n_layers: 1, d_model: D_MODEL, n_heads: 2, d_ff: 64 }
}

fn test_opts() -> TrainOpts {
    TrainOpts {
        steps: STEPS,
        batch_size: BATCH,
        block_size: BLOCK,
        lr: 1e-2,
        warmup: 0,
        decay_iters: STEPS,
        eval_interval: 0,
        checkpoint_secs: 0,
        mask_before: Some('='),
        mask_per_line: true,
        seed: 7, // seed: every rank must start from the same weights
        ..Default::default()
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-gpt2-distributed-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A port nothing is listening on, for the coordinator to bind.
fn free_address() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    drop(listener);
    address
}

fn spawn_rank(dir: &Path, rank: usize, plan: &Path, data: &Path, out: &Path, profile: Option<&Path>) -> Child {
    let log = dir.join(format!("rank-{rank}.log"));
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--ignored", "--exact", CHILD, "--test-threads=1", "--nocapture", "--"])
        .arg(format!("--rank={rank}"))
        .arg(format!("--plan={}", plan.display()))
        .arg(format!("--data={}", data.display()))
        .arg(format!("--out={}", out.display()));
    if let Some(profile) = profile {
        command.arg(format!("--profile={}", profile.display()));
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(std::fs::File::create(&log).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&log).unwrap()))
        .spawn()
        .unwrap()
}

/// Wait for `child` no longer than `limit`; kill it and fail with its log when
/// it does not finish.
fn wait_bounded(dir: &Path, rank: usize, mut child: Child, limit: Duration) -> i32 {
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait().unwrap() {
            Some(status) => return status.code().unwrap_or(-1),
            None if Instant::now() >= deadline => {
                child.kill().ok();
                let log = std::fs::read_to_string(dir.join(format!("rank-{rank}.log"))).unwrap_or_default();
                panic!("rank {rank} did not finish in {limit:?}; log:\n{log}");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

#[test]
fn two_ranks_train_bit_identical_replicas_over_loopback() {
    let dir = scratch("weights");
    let data = dir.join("calculator");
    data::prepare::prepare(data::prepare::Dataset::Calculator, &data, 4000, 1).unwrap();

    let mut plan = JobPlan::new("gpt2-two-rank", 2, free_address());
    plan.handshake_timeout_ms = 120_000;
    plan.op_timeout_ms = 120_000;
    let plan_path = dir.join("plan.json");
    plan.write(&plan_path).unwrap();

    let weights: Vec<PathBuf> = (0..2).map(|rank| dir.join(format!("rank-{rank}.safetensors"))).collect();
    let children: Vec<Child> = (0..2).map(|rank| spawn_rank(&dir, rank, &plan_path, &data, &weights[rank], None)).collect();
    for (rank, child) in children.into_iter().enumerate() {
        assert_eq!(wait_bounded(&dir, rank, child, Duration::from_secs(300)), 0, "rank {rank} failed; see {}/rank-{rank}.log", dir.display());
    }

    let replicas: Vec<Gpt> = weights.iter().map(|path| Gpt::load(path.to_str().unwrap(), BATCH, BLOCK)).collect();
    let names = replicas[0].param_names();
    assert!(!names.is_empty(), "a trained GPT has parameters to compare");
    for name in names {
        let a = replicas[0].read_weight(&name);
        let b = replicas[1].read_weight(&name);
        assert_eq!(a.len(), b.len(), "{name} has a different shape on each rank");
        assert_eq!(a, b, "{name} differs between the ranks' replicas");
    }
    // No `--profile-collectives` was asked for, so no rank wrote a profile.
    let profiles: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|name| name.contains(".rank") && name.ends_with(".json"))
        .collect();
    assert!(profiles.is_empty(), "a run without the profile flag wrote {profiles:?}");
    std::fs::remove_dir_all(&dir).ok();
}

/// With a profile asked for, every rank measures its own collective and writes
/// one document; the documents agree on the run's identity and report real
/// traffic rather than an empty accounting.
#[test]
fn every_rank_reports_its_collective_cost_in_its_own_document() {
    let dir = scratch("profile");
    let data = dir.join("calculator");
    data::prepare::prepare(data::prepare::Dataset::Calculator, &data, 4000, 1).unwrap();

    let mut plan = JobPlan::new("gpt2-two-rank-profile", 2, free_address());
    plan.handshake_timeout_ms = 120_000;
    plan.op_timeout_ms = 120_000;
    let plan_path = dir.join("plan.json");
    plan.write(&plan_path).unwrap();

    let weights: Vec<PathBuf> = (0..2).map(|rank| dir.join(format!("rank-{rank}.safetensors"))).collect();
    // The flag's value; each rank appends `.rank<rank>.json` to it.
    let base = dir.join("collectives.json");
    let children: Vec<Child> = (0..2).map(|rank| spawn_rank(&dir, rank, &plan_path, &data, &weights[rank], Some(&base))).collect();
    for (rank, child) in children.into_iter().enumerate() {
        assert_eq!(wait_bounded(&dir, rank, child, Duration::from_secs(300)), 0, "rank {rank} failed; see {}/rank-{rank}.log", dir.display());
    }

    let mut documents = Vec::new();
    for rank in 0..2 {
        let path = model::CollectiveProfile::path_for(&base, rank);
        assert!(path.is_file(), "rank {rank} wrote no profile at {}", path.display());
        let text = std::fs::read_to_string(&path).unwrap();
        let document: serde_json::Value = serde_json::from_str(&text).expect("a profile is one JSON document");
        assert_eq!(document["rank"], rank, "{text}");
        assert_eq!(document["world"], 2, "{text}");
        assert_eq!(document["plan"], plan.identity(), "the document must name the plan every rank agreed on");
        let count = document["count"].as_u64().expect("a document has an op count");
        let bytes = document["bytes"].as_u64().expect("a document has a byte count");
        assert!(count > 0, "a training run reduces gradients every step: {text}");
        assert!(bytes > 0, "the reduction moves payload: {text}");
        assert!(document["ops"]["all_reduce"]["count"].as_u64().expect("an all_reduce count") > 0, "{text}");
        let share = document["share_of_run"].as_f64().expect("a share of the run");
        assert!((0.0..=1.0).contains(&share), "collective time is part of the run it measured: {text}");
        assert!(document["run_ms"].as_f64().expect("a run wall time") > 0.0, "{text}");
        documents.push(document);
    }
    assert_eq!(documents[0]["world"], documents[1]["world"], "every rank must report the same world");
    assert_eq!(documents[0]["plan"], documents[1]["plan"], "every rank must report the same plan identity");
    std::fs::remove_dir_all(&dir).ok();
}

/// The rank process. It only runs when the test above re-executes this binary
/// with `--ignored --exact rank_process -- <args>`.
#[test]
#[ignore = "runs as a child process of the distributed training test"]
fn rank_process() {
    let args: Vec<String> = std::env::args().collect();
    let Some(separator) = args.iter().position(|arg| arg == "--") else { return };
    let after = &args[separator + 1..];
    let flag = |name: &str| after.iter().find_map(|arg| arg.strip_prefix(name).map(str::to_string));
    let Some(rank) = flag("--rank=").and_then(|value| value.parse::<usize>().ok()) else { return };
    let Some(plan) = flag("--plan=") else { return };
    let Some(data) = flag("--data=") else { return };
    let Some(out) = flag("--out=") else { return };
    let profile = flag("--profile=");

    let context = match JobContext::from_plan_file(Path::new(&plan), rank) {
        Ok(context) => context,
        Err(e) => {
            eprintln!("rank {rank}: {e}");
            std::process::exit(2);
        }
    };
    let collective = match context.collective() {
        Ok(collective) => collective,
        Err(e) => {
            eprintln!("rank {rank}: {e}");
            std::process::exit(3);
        }
    };
    let timed = profile.as_ref().map(|_| model::TimedCollective::new(&collective));
    let group: &dyn model::Collective = match &timed {
        Some(timed) => timed,
        None => &collective,
    };
    let sync = GradSync { collective: group, rank, checkpoint: None };
    let started = Instant::now();
    let trained = gpt2::train::train_distributed(Path::new(&data), test_config(), &test_opts(), None, sync);
    let run = started.elapsed();
    if let (Some(base), Some(timed)) = (&profile, &timed) {
        let document = model::CollectiveProfile::new(rank, context.plan().world_size, context.plan().identity(), run, timed.snapshot());
        if let Err(e) = document.write(Path::new(base)) {
            eprintln!("rank {rank}: write the collective profile: {e}");
            std::process::exit(3);
        }
    }
    match trained {
        // Each rank writes its own replica so the test can compare what every
        // rank actually ended with, not only what the writer published.
        Ok((_report, replica)) => replica.save(&out),
        Err(e) => {
            eprintln!("rank {rank}: {e}");
            std::process::exit(3);
        }
    }
}
