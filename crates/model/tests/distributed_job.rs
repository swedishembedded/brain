// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-process training that a batch system
// can launch without brain knowing the batch system for its clients. If your
// team needs expertise in distributed job startup or cross-process collective
// testing, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A distributed job as a launcher starts it: one real OS process per rank,
//! the same plan file, a loopback coordinator, and no shared memory.
//!
//! The children are this test binary re-executed (`--ignored --exact
//! rank_process`), so the transport, the plan handshake, the abort path and the
//! socket timeouts are exercised across process boundaries rather than across
//! threads.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use model::collective::{Collective, Payload};
use model::{JobContext, JobPlan};

/// The name libtest filters to when this binary is re-executed as a rank.
const CHILD: &str = "rank_process";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-job-{}-{name}", std::process::id()));
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

fn plan_file(dir: &Path, plan: &JobPlan) -> PathBuf {
    let path = dir.join(format!("plan-{}-{}.json", plan.world_size, plan.dp));
    plan.write(&path).unwrap();
    path
}

fn spawn_rank(dir: &Path, rank: usize, plan: &Path, exit_after_join: bool) -> Child {
    let log = dir.join(format!("rank-{rank}.log"));
    let output = dir.join(format!("rank-{rank}.out"));
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--ignored", "--exact", CHILD, "--test-threads=1", "--nocapture", "--"])
        .arg(format!("--rank={rank}"))
        .arg(format!("--plan={}", plan.display()))
        .arg(format!("--out={}", output.display()));
    if exit_after_join {
        command.arg("--exit-after-join");
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
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

fn read_result(dir: &Path, rank: usize) -> Vec<f32> {
    let text = std::fs::read_to_string(dir.join(format!("rank-{rank}.out"))).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn separate_processes_reduce_over_loopback() {
    let dir = scratch("converge");
    let mut plan = JobPlan::new("three-way", 3, free_address());
    plan.handshake_timeout_ms = 15_000;
    plan.op_timeout_ms = 15_000;
    let path = plan_file(&dir, &plan);

    let children: Vec<Child> = (1..3).map(|rank| spawn_rank(&dir, rank, &path, false)).collect();
    let context = JobContext::from_plan_file(&path, 0).unwrap();
    let collective = context.collective().expect("rank 0 joins");
    // Local payload is rank 0's own contribution: [0, 0].
    let reduced = pollster::block_on(collective.all_reduce(0, Payload::f32(vec![0.0, 0.0]))).expect("all_reduce");
    // rank r contributes [r, 2r], so the sum is [0+1+2, 0+2+4].
    assert_eq!(reduced.data, vec![3.0, 6.0]);

    for (index, child) in children.into_iter().enumerate() {
        let rank = index + 1;
        assert_eq!(wait_bounded(&dir, rank, child, Duration::from_secs(20)), 0, "rank {rank} failed");
        assert_eq!(read_result(&dir, rank), vec![3.0, 6.0], "rank {rank} saw a different reduction");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_rank_that_dies_does_not_hang_its_peers() {
    let dir = scratch("dead-rank");
    let mut plan = JobPlan::new("two-way", 2, free_address());
    plan.handshake_timeout_ms = 10_000;
    plan.op_timeout_ms = 2_000;
    let path = plan_file(&dir, &plan);

    // Rank 1 joins and then exits without contributing.
    let child = spawn_rank(&dir, 1, &path, true);
    let context = JobContext::from_plan_file(&path, 0).unwrap();
    let collective = context.collective().expect("rank 0 joins");
    let started = Instant::now();
    let result = pollster::block_on(collective.all_reduce(0, Payload::f32(vec![1.0, 2.0])));
    let elapsed = started.elapsed();
    let error = result.expect_err("a rank that left must surface as an error");
    assert!(error.to_string().contains("transport") || error.to_string().contains("abort") || error.to_string().contains("rank"), "unexpected error: {error}");
    assert!(elapsed < Duration::from_secs(10), "the remaining rank waited {elapsed:?}, not the plan's bound");
    wait_bounded(&dir, 1, child, Duration::from_secs(10));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_rank_started_with_a_different_plan_is_refused() {
    let dir = scratch("plan-mismatch");
    let mut mine = JobPlan::new("the-job", 2, free_address());
    mine.handshake_timeout_ms = 10_000;
    mine.op_timeout_ms = 2_000;
    let path = plan_file(&dir, &mine);
    // The worker is started with a three-rank plan for the same job.
    let mut theirs = mine.clone();
    theirs.world_size = 3;
    theirs.dp = 3;
    let wrong_path = dir.join("wrong-plan.json");
    theirs.write(&wrong_path).unwrap();

    let child = spawn_rank(&dir, 1, &wrong_path, false);
    let context = JobContext::from_plan_file(&path, 0).unwrap();
    let error = context.collective().expect_err("the plans do not agree");
    assert!(error.to_string().contains("different plan"), "unexpected error: {error}");
    assert_ne!(wait_bounded(&dir, 1, child, Duration::from_secs(10)), 0, "the worker joined a job it was not part of");
    std::fs::remove_dir_all(&dir).ok();
}

/// The rank process. It only runs when the tests above re-execute this binary
/// with `--ignored --exact rank_process -- <args>`.
#[test]
#[ignore = "runs as a child process of the distributed job tests"]
fn rank_process() {
    let args: Vec<String> = std::env::args().collect();
    let Some(separator) = args.iter().position(|arg| arg == "--") else { return };
    let after = &args[separator + 1..];
    let flag = |name: &str| after.iter().find_map(|arg| arg.strip_prefix(name).map(str::to_string));
    let Some(rank) = flag("--rank=").and_then(|value| value.parse::<usize>().ok()) else { return };
    let Some(plan) = flag("--plan=") else { return };
    let Some(out) = flag("--out=") else { return };
    let exit_after_join = after.iter().any(|arg| arg == "--exit-after-join");

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
    if exit_after_join {
        // Leave the group the way a killed rank would: no contribution, no
        // result, only a closed socket.
        return;
    }
    let local = vec![rank as f32, 2.0 * rank as f32];
    match pollster::block_on(collective.all_reduce(rank, Payload::f32(local))) {
        Ok(reduced) => {
            std::fs::write(Path::new(&out), serde_json::to_string(&reduced.data).unwrap()).unwrap();
        }
        Err(e) => {
            eprintln!("rank {rank}: {e}");
            std::process::exit(3);
        }
    }
}
