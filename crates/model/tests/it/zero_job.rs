// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sharded data-parallel training across the cards and nodes of a cluster for its
// clients. If your team needs expertise in ZeRO-style sharding over NCCL, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The sharded optimizer over a job's device collective: one OS process per rank, one card per process, the
//! reduce-scatter, all-reduce and all-gather on the cards over NCCL. Every rank ends with the same parameters bit for
//! bit, a repeat of the job gives the same bits, and the parameters are the replicated AdamW update computed on the
//! host within a stated tolerance. Needs the CUDA backend, as many selected cards as ranks and NCCL; skips by name
//! without them.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use gpu_core::devgroup::RankGroup;
use gpu_core::{CollectiveDtype, Gpu};
use model::zero::{self, Sharding, StateTier, StepOutcome, ZeroOptimizer};
use model::{CollectivePath, JobContext, JobPlan, JobSecret};
use serde_json::{json, Value};

use crate::zero_shard::{args, grad, params, values, SIZES, UNIT};

const CHILD: &str = "zero_job::rank_process";
const STEPS: u32 = 3;
const SEED: u32 = 3;

fn cards_for(world: usize) -> bool {
    if brain_testutil::gpu_unavailable(gpu_core::testgpu::missing_gpu()) {
        return false;
    }
    if gpu_core::backend_name() != "cuda" {
        brain_testutil::skip_unavailable(&format!("needs the CUDA backend selected (BRAIN_BACKEND=cuda), not '{}'", gpu_core::backend_name()));
        return false;
    }
    if let Err(e) = gpu_core::testgpu::stage_cards(None, world) {
        brain_testutil::skip_unavailable(&format!("{world} ranks need {world} cards: {e}"));
        return false;
    }
    if let Err(e) = RankGroup::supported(&Gpu::new_on(gpu_core::testgpu::cards()[0], &[])) {
        brain_testutil::skip_unavailable(&format!("needs NCCL's per-process communicator: {e}"));
        return false;
    }
    true
}

fn free_address() -> String {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string()
}

/// Run `world` ranks of one sharded job and return each rank's result document.
fn run(world: usize, sharding: Sharding, tier: StateTier, dtype: &str) -> Vec<Value> {
    let dir = std::env::temp_dir().join(format!("brain-zero-job-{}-{world}-{sharding:?}-{tier:?}-{dtype}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut plan = JobPlan::new("zero", world, free_address());
    plan.collectives = CollectivePath::Device;
    plan.sharding = sharding;
    plan.handshake_timeout_ms = 60_000;
    plan.op_timeout_ms = 60_000;
    let plan_path = dir.join("plan.json");
    plan.write(&plan_path).unwrap();
    let secret = dir.join("job.secret");
    JobSecret::random().write(&secret).unwrap();
    let children: Vec<Child> = (0..world)
        .map(|rank| {
            let log = std::fs::File::create(dir.join(format!("rank-{rank}.log"))).unwrap();
            Command::new(std::env::current_exe().unwrap())
                .args(["--ignored", "--exact", CHILD, "--test-threads=1", "--nocapture", "--"])
                .arg(format!("--rank={rank}"))
                .arg(format!("--dtype={dtype}"))
                .arg(format!("--tier={}", if tier == StateTier::Host { "host" } else { "device" }))
                .arg(format!("--plan={}", plan_path.display()))
                .arg(format!("--secret={}", secret.display()))
                .arg(format!("--out={}", dir.join(format!("rank-{rank}.json")).display()))
                .stdin(Stdio::null())
                .stdout(Stdio::from(log.try_clone().unwrap()))
                .stderr(Stdio::from(log))
                .spawn()
                .unwrap()
        })
        .collect();
    let log = |rank: usize| std::fs::read_to_string(dir.join(format!("rank-{rank}.log"))).unwrap_or_default();
    for (rank, mut child) in children.into_iter().enumerate() {
        let deadline = Instant::now() + Duration::from_secs(180);
        let code = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status.code().unwrap_or(-1);
            }
            if Instant::now() >= deadline {
                child.kill().ok();
                child.wait().ok();
                panic!("rank {rank} did not finish; log:\n{}", log(rank));
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(code, 0, "rank {rank} failed; log:\n{}", log(rank));
    }
    let docs = (0..world).map(|r| serde_json::from_str(&std::fs::read_to_string(dir.join(format!("rank-{r}.json"))).unwrap()).unwrap()).collect();
    std::fs::remove_dir_all(&dir).ok();
    docs
}

/// The replicated AdamW run on the host: every rank's gradient weighted by its share (`1 + rank`), the global clip,
/// then the update of every element.
fn reference(world: usize) -> Vec<Vec<f32>> {
    let mut w: Vec<Vec<f32>> = SIZES.iter().enumerate().map(|(i, &n)| values(SEED * 31 + i as u32, n)).collect();
    let mut m: Vec<Vec<f32>> = SIZES.iter().map(|&n| vec![0.0; n as usize]).collect();
    let mut v = m.clone();
    let total: f32 = (0..world).map(|r| 1.0 + r as f32).sum();
    for t in 1..=STEPS {
        let g: Vec<Vec<f32>> = SIZES
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let mut sum = vec![0f32; n as usize];
                for r in 0..world {
                    for (s, x) in sum.iter_mut().zip(grad(t, r, i, n)) {
                        *s += x * (1.0 + r as f32) / total;
                    }
                }
                sum
            })
            .collect();
        let a = args(t);
        let sq: f64 = g.iter().flatten().map(|&x| f64::from(x) * f64::from(x)).sum();
        let mult = optim::grad_multiplier(sq, a.clip, a.extra_scale);
        let bc = a.adam.bias_corrections(t);
        for i in 0..SIZES.len() {
            for e in 0..g[i].len() {
                a.adam.update(bc, a.lr, a.wd, g[i][e] * mult, &mut w[i][e], &mut m[i][e], &mut v[i][e]);
            }
        }
    }
    w
}

fn floats(v: &Value) -> Vec<Vec<f32>> {
    v["values"].as_array().unwrap().iter().map(|t| t.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect()
}

fn check(world: usize, sharding: Sharding) {
    if !cards_for(world) {
        return;
    }
    let lr = args(1).lr;
    for tier in [StateTier::Device, StateTier::Host] {
        let first = run(world, sharding, tier, "f32");
        for (rank, doc) in first.iter().enumerate() {
            assert_eq!(doc["provider"], json!("nccl"), "rank {rank} reduced on {}", doc["provider"]);
            assert_eq!(doc["values"], first[0]["values"], "{tier:?}: rank {rank}'s parameters differ from rank 0's");
        }
        assert_eq!(run(world, sharding, tier, "f32")[0]["values"], first[0]["values"], "{tier:?}: a repeat of the job is not bit-identical");
        let worst = floats(&first[0]).iter().zip(reference(world)).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect::<Vec<_>>()).fold(0f32, f32::max);
        assert!(worst <= 1e-2 * lr, "{sharding:?} {tier:?} x{world}: {worst:e} from the host reference, over 1% of a step");
        let half = run(world, sharding, tier, "bf16");
        for doc in &half {
            assert_eq!(doc["values"], half[0]["values"], "{tier:?}: bf16 parameters differ between ranks");
        }
        eprintln!("{sharding:?} {tier:?} x{world} over NCCL: worst {worst:e} from the host reference; ranks and repeat bit-identical");
    }
}

#[test]
fn two_ranks_shard_gradients_over_nccl() {
    check(2, Sharding::Gradients);
}

#[test]
fn four_ranks_shard_gradients_over_nccl() {
    check(4, Sharding::Gradients);
}

#[test]
fn four_ranks_shard_the_optimizer_over_nccl() {
    check(4, Sharding::Optimizer);
}

#[test]
#[ignore = "runs as a child process of the sharded job tests"]
fn rank_process() {
    let args_: Vec<String> = std::env::args().collect();
    let Some(separator) = args_.iter().position(|arg| arg == "--") else { return };
    let after = &args_[separator + 1..];
    let flag = |name: &str| after.iter().find_map(|arg| arg.strip_prefix(name).map(str::to_string));
    let (Some(rank), Some(dtype), Some(tier), Some(plan), Some(secret), Some(out)) = (flag("--rank=").and_then(|v| v.parse::<usize>().ok()), flag("--dtype="), flag("--tier="), flag("--plan="), flag("--secret="), flag("--out=")) else {
        return;
    };
    let fail = |e: String| -> ! {
        eprintln!("rank {rank}: {e}");
        std::process::exit(1)
    };
    let dtype = if dtype == "bf16" { CollectiveDtype::Bf16 } else { CollectiveDtype::F32 };
    let ctx = JobContext::from_plan_file(Path::new(&plan), rank, Path::new(&secret)).unwrap_or_else(|e| fail(e));
    let job = ctx.job_collective().unwrap_or_else(|e| fail(e.to_string()));
    let tier = if tier == "host" { StateTier::Host } else { StateTier::Device };
    // A handle without the sharded step's kernels: the optimizer registers them on a handle of its own.
    // Rank r runs on the r-th selected card (`BRAIN_DEVICE` / `--device`), not on gpu r.
    let gpu = Gpu::new_on(gpu_core::testgpu::cards()[rank], &[]);
    let p = params(&gpu, SEED, dtype);
    let handles = p.tensors.clone();
    let mut opt = ZeroOptimizer::new(p, ctx.plan().sharding, tier, &job, rank, UNIT).unwrap_or_else(|e| fail(e));
    for t in 1..=STEPS {
        for (i, h) in handles.iter().enumerate() {
            zero::write_elems(&gpu, &h.grad, 0, &grad(t, rank, i, h.numel), dtype);
        }
        match opt.step(&job, 1.0 + rank as f32, true, &args(t)).unwrap_or_else(|e| fail(e.to_string())) {
            StepOutcome::Applied { .. } => {}
            skipped => fail(format!("{skipped:?}")),
        }
    }
    let values: Vec<Vec<f32>> = handles.iter().map(|h| zero::read_elems(&gpu, &h.value, 0, h.numel, dtype)).collect();
    let doc = json!({ "provider": job.device_provider(), "values": values });
    std::fs::write(PathBuf::from(out), serde_json::to_string(&doc).unwrap()).unwrap();
}
