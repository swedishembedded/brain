// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-node training runtimes whose gradient
// traffic stays on the accelerator interconnect for its clients. If your team
// needs expertise in cross-process device collectives and their failure
// semantics, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Device collectives across processes: one OS process per rank, one card per
//! process, the rendezvous id carried by the job's sealed channel.
//!
//! The ranks are this test binary re-executed (`--ignored --exact
//! nccl_job::rank_process`), as `distributed_job.rs` does for the host star.
//! Every rank writes what its card ended up holding; the parent compares it
//! with a reference it computes on the host. The tests need the CUDA backend
//! selected, as many selected cards as ranks, and NCCL; without them they
//! skip by name.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use gpu_core::devgroup::RankGroup;
use gpu_core::{CollectiveDtype, DeviceBuffer, Gpu};
use model::collective::Collective;
use model::devcollective::rendezvous;
use model::{Axis, CollectivePath, Grid, JobContext, JobPlan, JobSecret};
use serde_json::{json, Value};

const CHILD: &str = "nccl_job::rank_process";
/// Elements per rank in the op checks; not a multiple of anything NCCL likes.
const N: usize = 1001;
/// Gradient tensor sizes for the bucketed reduction under a 1 MiB cap: small ones packed into buckets
/// (more than one), and one at the cap (262,144 fp32) reduced in place.
const GRADS: [usize; 7] = [1001, 200_000, 262_144, 7, 100_000, 3, 70_000];
/// bf16 gradient sizes, odd ones among them, so packed members must be aligned to whole words; one at the
/// 1 MiB cap (524,288 bf16) stays in place.
const GRADS_BF16: [usize; 5] = [3, 1001, 524_288, 5, 70_001];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-nccl-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn free_address() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().to_string()
}

/// Whether this process can run `world` ranks on cards: the CUDA backend, that
/// many selected cards, and NCCL with the per-process communicator. Skips by
/// name when not.
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
    let probe = Gpu::new_on(gpu_core::testgpu::cards()[0], &[]);
    if let Err(e) = RankGroup::supported(&probe) {
        brain_testutil::skip_unavailable(&format!("needs NCCL's per-process communicator: {e}"));
        return false;
    }
    true
}

struct Job {
    dir: PathBuf,
    plan: PathBuf,
    secret: PathBuf,
}

impl Job {
    fn new(name: &str, world: usize, op_timeout_ms: u64) -> Job {
        let dir = scratch(name);
        let mut plan = JobPlan::new(name, world, free_address());
        plan.handshake_timeout_ms = 60_000;
        plan.op_timeout_ms = op_timeout_ms;
        plan.collectives = CollectivePath::Device;
        // A 1 MiB bucket cap, so the gradient check below has packed buckets and a tensor reduced in place.
        plan.bucket_mib = 1;
        if world.is_multiple_of(2) {
            plan.tp = 2;
            plan.dp = world / 2;
        }
        let path = dir.join("plan.json");
        plan.write(&path).unwrap();
        let secret = dir.join("job.secret");
        JobSecret::random().write(&secret).unwrap();
        Job { dir, plan: path, secret }
    }

    fn spawn(&self, rank: usize, case: &str, secret: &Path) -> Child {
        let log = std::fs::File::create(self.dir.join(format!("rank-{rank}-{case}.log"))).unwrap();
        Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", CHILD, "--test-threads=1", "--nocapture", "--"])
            .arg(format!("--rank={rank}"))
            .arg(format!("--case={case}"))
            .arg(format!("--plan={}", self.plan.display()))
            .arg(format!("--secret={}", secret.display()))
            .arg(format!("--out={}", self.dir.join(format!("rank-{rank}.json")).display()))
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone().unwrap()))
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap()
    }

    fn wait(&self, rank: usize, case: &str, mut child: Child, limit: Duration) -> i32 {
        let deadline = Instant::now() + limit;
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                return status.code().unwrap_or(-1);
            }
            if Instant::now() >= deadline {
                child.kill().ok();
                child.wait().ok();
                panic!("rank {rank} did not finish in {limit:?}; log:\n{}", self.log(rank, case));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn log(&self, rank: usize, case: &str) -> String {
        std::fs::read_to_string(self.dir.join(format!("rank-{rank}-{case}.log"))).unwrap_or_default()
    }

    fn result(&self, rank: usize) -> Value {
        serde_json::from_str(&std::fs::read_to_string(self.dir.join(format!("rank-{rank}.json"))).unwrap()).unwrap()
    }

    /// Run every rank of `case` and return each rank's result document.
    fn run(&self, world: usize, case: &str) -> Vec<Value> {
        let children: Vec<Child> = (0..world).map(|r| self.spawn(r, case, &self.secret)).collect();
        for (rank, child) in children.into_iter().enumerate() {
            let code = self.wait(rank, case, child, Duration::from_secs(180));
            assert_eq!(code, 0, "rank {rank} failed; log:\n{}", self.log(rank, case));
        }
        (0..world).map(|r| self.result(r)).collect()
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

/// Rank `rank`'s exact-sum input: small multiples of a quarter, so every
/// summation order gives the same fp32 (and bf16) value.
fn exact_input(rank: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + rank * 13) % 17) as f32 * 0.25 - 2.0).collect()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn sum_of(ranks: impl IntoIterator<Item = usize>, n: usize) -> Vec<f32> {
    let mut sum = vec![0f32; n];
    for r in ranks {
        for (s, x) in sum.iter_mut().zip(exact_input(r, n)) {
            *s += x;
        }
    }
    sum
}

fn check_ops(world: usize) {
    if !cards_for(world) {
        return;
    }
    let job = Job::new(&format!("ops{world}"), world, 30_000);
    let results = job.run(world, "ops");
    let grid = Grid::new(2, 1, world / 2);
    let total = sum_of(0..world, N);
    for (rank, got) in results.iter().enumerate() {
        assert_eq!(got["provider"], json!("nccl"), "rank {rank}");
        let grads: Vec<f32> = GRADS.iter().flat_map(|&n| sum_of(0..world, n)).map(|x| x * 0.5).collect();
        assert_eq!(floats(&got["grad"]), grads, "rank {rank}: bucketed device gradient reduction (scale 0.5)");
        let halves: Vec<f32> = GRADS_BF16.iter().flat_map(|&n| sum_of(0..world, n)).collect();
        assert_eq!(floats(&got["grad_bf16"]), halves, "rank {rank}: bucketed bf16 gradient reduction");
        assert_eq!(floats(&got["all_reduce"]), total, "rank {rank}: all_reduce f32");
        assert_eq!(floats(&got["all_reduce_bf16"]), total, "rank {rank}: all_reduce bf16");
        let gathered: Vec<f32> = (0..world).flat_map(|r| exact_input(r, N)).collect();
        assert_eq!(floats(&got["all_gather"]), gathered, "rank {rank}: all_gather");
        let scattered = sum_of(0..world, N * world)[rank * N..(rank + 1) * N].to_vec();
        assert_eq!(floats(&got["reduce_scatter"]), scattered, "rank {rank}: reduce_scatter");
        assert_eq!(floats(&got["broadcast"]), exact_input(world - 1, N), "rank {rank}: broadcast from the last rank");
        for (axis, key, group) in [(Axis::Tp, "tp", grid.tp_group(rank)), (Axis::Dp, "dp", grid.dp_group(rank))] {
            let sub = &got[key];
            assert_eq!(sub["world"], json!(group.len()), "rank {rank}: {axis:?} group size");
            assert_eq!(sub["rank"], json!(grid.split_key(rank, axis).1), "rank {rank}: {axis:?} rank");
            assert_eq!(floats(&sub["sum"]), sum_of(group, N), "rank {rank}: {axis:?} group all_reduce");
        }
    }
    // Inexact sums: every rank holds the same bytes, and a repeat gives them again.
    let reference = &results[0]["inexact"];
    for (rank, got) in results.iter().enumerate() {
        assert_eq!(&got["inexact"], reference, "rank {rank} received different bytes than rank 0");
        assert_eq!(got["inexact"], got["inexact_again"], "rank {rank}: a repeated reduction gave other bytes");
    }
}

#[test]
fn two_ranks_reduce_gather_scatter_broadcast_and_split_on_their_cards() {
    check_ops(2);
}

#[test]
fn four_ranks_reduce_gather_scatter_broadcast_and_split_on_their_cards() {
    check_ops(4);
}

/// A rank that leaves after the group formed must surface on the survivor as
/// an error within the plan's op timeout, never as a hang.
#[test]
fn a_rank_that_dies_is_an_error_within_the_op_timeout() {
    if !cards_for(2) {
        return;
    }
    let job = Job::new("dies", 2, 5_000);
    let survivor = job.spawn(0, "survive", &job.secret);
    let quitter = job.spawn(1, "quit", &job.secret);
    assert_eq!(job.wait(1, "quit", quitter, Duration::from_secs(120)), 0, "{}", job.log(1, "quit"));
    assert_eq!(job.wait(0, "survive", survivor, Duration::from_secs(120)), 0, "{}", job.log(0, "survive"));
    let got = job.result(0);
    assert!(got["error"].as_str().is_some_and(|e| !e.is_empty()), "the survivor saw no error: {got}");
    let waited = got["waited_ms"].as_f64().unwrap();
    eprintln!("the survivor's error after {waited:.0} ms: {}", got["error"]);
    assert!(waited < 5_000.0 + 10_000.0, "the survivor waited {waited} ms, not the plan's bound");
}

/// A process with the plan and the address but without the job secret never
/// receives the rendezvous id: it is refused at the sealed handshake, and the
/// real ranks form the device group and reduce as if it had not been there.
#[test]
fn a_rank_without_the_secret_cannot_join_the_device_group() {
    if !cards_for(2) {
        return;
    }
    let job = Job::new("intruder", 2, 30_000);
    let wrong = job.dir.join("wrong.secret");
    JobSecret::random().write(&wrong).unwrap();
    let rank0 = job.spawn(0, "ops", &job.secret);
    let intruder = job.spawn(1, "intrude", &wrong);
    assert_ne!(job.wait(1, "intrude", intruder, Duration::from_secs(120)), 0, "a rank without the secret joined: {}", job.log(1, "intrude"));
    let rank1 = job.spawn(1, "ops", &job.secret);
    assert_eq!(job.wait(1, "ops", rank1, Duration::from_secs(180)), 0, "{}", job.log(1, "ops"));
    assert_eq!(job.wait(0, "ops", rank0, Duration::from_secs(180)), 0, "{}", job.log(0, "ops"));
    assert_eq!(floats(&job.result(0)["all_reduce"]), sum_of(0..2, N));
}

/// Gradient payloads the measurement reduces, as `(name, fp32 elements)`: the
/// Qwen3-0.6B LoRA (rank 4) step the ledger measured over the host star, a
/// 27B hybrid-attention LoRA (rank 16 on every linear of a 64-layer,
/// d_model 5120 model), and a full Qwen3-0.6B gradient.
const PAYLOADS: [(&str, usize); 3] = [("qwen3-0.6b-lora4", 2_523_136), ("27b-lora16", 108_863_488), ("qwen3-0.6b-full", 596_049_920)];

/// Time one data-parallel gradient reduction per payload, on the device (the
/// job's device path) and through the host star (read the gradient back,
/// all-reduce the fp32 payload over the sealed transport, write it back), and
/// print one line per payload. A measurement, not a check: run by hand on a
/// node with the cards, e.g. `it nccl_job::measure --ignored --nocapture`.
fn measure(world: usize) {
    if !cards_for(world) {
        return;
    }
    let job = Job::new(&format!("measure{world}"), world, 600_000);
    let results = job.run(world, "measure");
    for (name, elems) in PAYLOADS {
        let row: Vec<Value> = results.iter().map(|r| r[name].clone()).collect();
        println!("{}", json!({ "world": world, "payload": name, "bytes": elems * 4, "ranks": row }));
    }
}

#[test]
#[ignore = "measurement: run by hand on a node with two cards"]
fn measure_two_ranks() {
    measure(2);
}

#[test]
#[ignore = "measurement: run by hand on a node with four cards"]
fn measure_four_ranks() {
    measure(4);
}

/// The rank's side of [`measure`]: median milliseconds per reduction on each path.
fn measure_rank(rank: usize, ctx: &JobContext, gpu: &Gpu) -> Value {
    let device = ctx.job_collective().unwrap_or_else(|e| fail(rank, e));
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut doc = json!({});
    for (name, elems) in PAYLOADS {
        let buf = gpu.storage(elems as u64);
        gpu.write_f32(&buf, &data::rng::Lcg::new(rank as u64).vec(elems));
        let bufs = [(&buf, elems as u64)];
        let mut on_device = Vec::new();
        for i in 0..6 {
            gpu.flush();
            let started = Instant::now();
            assert!(pollster::block_on(device.all_reduce_device(rank, gpu, &bufs, CollectiveDtype::F32, 1.0 / ctx.plan().world_size as f32)).unwrap_or_else(|e| fail(rank, e)));
            if i > 0 {
                on_device.push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        let mut through_host = Vec::new();
        for i in 0..if elems > 100_000_000 { 3 } else { 6 } {
            gpu.flush();
            let started = Instant::now();
            // In pieces no larger than a large model's largest tensor, as the trainer's per-tensor host path
            // sends them (one message also has a wire limit).
            let local = gpu.read(&buf, elems);
            let mut summed = Vec::with_capacity(elems);
            for piece in local.chunks(1 << 27) {
                let reduced = pollster::block_on(device.all_reduce(rank, model::collective::Payload::f32(piece.to_vec()))).unwrap_or_else(|e| fail(rank, e));
                summed.extend(reduced.data);
            }
            gpu.write_f32(&buf, &summed);
            gpu.flush();
            if i > 0 || elems > 100_000_000 {
                through_host.push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        doc[name] = json!({ "device_ms": median(on_device), "host_star_ms": median(through_host) });
    }
    doc
}

// ---- the rank process ----------------------------------------------------------------------------------------

fn bf16_words(values: &[f32]) -> Vec<u32> {
    values.chunks(2).map(|p| (p[0].to_bits() >> 16) | (p.get(1).map_or(0, |x| x.to_bits() >> 16) << 16)).collect()
}

fn from_bf16_words(words: &[f32], n: usize) -> Vec<f32> {
    words.iter().flat_map(|w| [w.to_bits() << 16, w.to_bits() & 0xffff_0000]).take(n).map(f32::from_bits).collect()
}

fn fail(rank: usize, what: impl std::fmt::Display) -> ! {
    eprintln!("rank {rank}: {what}");
    std::process::exit(3)
}

/// The rank's checks of every op, written as one JSON document.
fn ops(rank: usize, ctx: &JobContext, gpu: &Gpu) -> Value {
    let world = ctx.plan().world_size;
    let job = ctx.job_collective().unwrap_or_else(|e| fail(rank, e));
    let host: &dyn Collective = &job;
    let up = |data: &[f32]| {
        let buf = gpu.storage(data.len() as u64);
        gpu.write_f32(&buf, data);
        buf
    };

    // The trainer's path: the job's collective reduces device gradients itself.
    // The trainer's shape: many tensors, small ones packed into buckets, one at the cap reduced in place.
    let grads: Vec<DeviceBuffer> = GRADS.iter().map(|&n| up(&exact_input(rank, n))).collect();
    let bufs: Vec<(&DeviceBuffer, u64)> = grads.iter().zip(GRADS).map(|(b, n)| (b, n as u64)).collect();
    let on_device = pollster::block_on(host.all_reduce_device(rank, gpu, &bufs, CollectiveDtype::F32, 0.5)).unwrap_or_else(|e| fail(rank, e));
    assert!(on_device, "the plan requires the device path");
    // bf16 gradients of odd lengths: packed members start on whole words.
    let halves: Vec<DeviceBuffer> = GRADS_BF16
        .iter()
        .map(|&n| {
            let buf = gpu.storage(n.div_ceil(2) as u64);
            gpu.write(&buf, &bf16_words(&exact_input(rank, n)));
            buf
        })
        .collect();
    let half_bufs: Vec<(&DeviceBuffer, u64)> = halves.iter().zip(GRADS_BF16).map(|(b, n)| (b, n as u64)).collect();
    assert!(pollster::block_on(host.all_reduce_device(rank, gpu, &half_bufs, CollectiveDtype::Bf16, 1.0)).unwrap_or_else(|e| fail(rank, e)));
    let provider = job.device_provider();

    let group = rendezvous(host, rank, RankGroup::supported(gpu), || RankGroup::id(gpu), |id| RankGroup::join(gpu, id, world, rank, Duration::from_secs(60)))
        .unwrap_or_else(|e| fail(rank, e))
        .unwrap_or_else(|e| fail(rank, e));
    let timeout = Duration::from_secs(30);
    let done = |r: Result<(), String>| r.and_then(|()| group.wait(timeout)).unwrap_or_else(|e| fail(rank, e));

    let reduced = up(&exact_input(rank, N));
    done(group.all_reduce_sum(&[(&reduced, N as u64)], CollectiveDtype::F32, 1.0));

    let half = gpu.storage(N.div_ceil(2) as u64);
    gpu.write(&half, &bf16_words(&exact_input(rank, N)));
    done(group.all_reduce_sum(&[(&half, N as u64)], CollectiveDtype::Bf16, 1.0));

    let gathered = gpu.storage((N * world) as u64);
    done(group.all_gather(&up(&exact_input(rank, N)), &gathered, N as u64, CollectiveDtype::F32));

    let scattered = gpu.storage(N as u64);
    done(group.reduce_scatter(&up(&exact_input(rank, N * world)), &scattered, N as u64, CollectiveDtype::F32));

    let cast = up(&exact_input(rank, N));
    done(group.broadcast(&cast, N as u64, CollectiveDtype::F32, world - 1));

    let mut doc = json!({
        "provider": provider,
        "grad": grads.iter().zip(GRADS).flat_map(|(b, n)| gpu.read(b, n)).collect::<Vec<f32>>(),
        "grad_bf16": halves.iter().zip(GRADS_BF16).flat_map(|(b, n)| from_bf16_words(&gpu.read(b, n.div_ceil(2)), n)).collect::<Vec<f32>>(),
        "all_reduce": gpu.read(&reduced, N),
        "all_reduce_bf16": from_bf16_words(&gpu.read(&half, N.div_ceil(2)), N),
        "all_gather": gpu.read(&gathered, N * world),
        "reduce_scatter": gpu.read(&scattered, N),
        "broadcast": gpu.read(&cast, N),
    });

    let grid = ctx.plan().grid();
    for (axis, key) in [(Axis::Tp, "tp"), (Axis::Dp, "dp")] {
        let (color, k) = grid.split_key(rank, axis);
        let sub = group.split(Some(color), k, timeout).unwrap_or_else(|e| fail(rank, e)).unwrap_or_else(|| fail(rank, "a colored rank got no group"));
        let buf = up(&exact_input(rank, N));
        sub.all_reduce_sum(&[(&buf, N as u64)], CollectiveDtype::F32, 1.0).and_then(|()| sub.wait(timeout)).unwrap_or_else(|e| fail(rank, e));
        doc[key] = json!({ "world": sub.world(), "rank": sub.rank(), "sum": gpu.read(&buf, N) });
    }

    // Inexact sums, twice: the bytes every rank holds, as bit patterns.
    let noise = data::rng::Lcg::new(0x5eed + rank as u64).vec(1 << 20);
    let mut again = Vec::new();
    for _ in 0..2 {
        let buf = up(&noise);
        done(group.all_reduce_sum(&[(&buf, noise.len() as u64)], CollectiveDtype::F32, 1.0));
        let bits: Vec<u32> = gpu.read(&buf, noise.len()).into_iter().map(f32::to_bits).collect();
        again.push(format!("{:016x}", bits.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))));
    }
    doc["inexact"] = json!(again[0]);
    doc["inexact_again"] = json!(again[1]);
    doc
}

#[test]
#[ignore = "runs as a child process of the NCCL job tests"]
fn rank_process() {
    let args: Vec<String> = std::env::args().collect();
    let Some(separator) = args.iter().position(|arg| arg == "--") else { return };
    let after = &args[separator + 1..];
    let flag = |name: &str| after.iter().find_map(|arg| arg.strip_prefix(name).map(str::to_string));
    let (Some(rank), Some(case), Some(plan), Some(secret), Some(out)) = (flag("--rank=").and_then(|v| v.parse::<usize>().ok()), flag("--case="), flag("--plan="), flag("--secret="), flag("--out=")) else {
        return;
    };
    let ctx = JobContext::from_plan_file(Path::new(&plan), rank, Path::new(&secret)).unwrap_or_else(|e| fail(rank, e));
    let gpu = Gpu::new_on(gpu_core::testgpu::card(rank as u32), &[]);
    let doc = match case.as_str() {
        "ops" | "intrude" => ops(rank, &ctx, &gpu),
        "measure" => measure_rank(rank, &ctx, &gpu),
        "quit" | "survive" => {
            let job = ctx.job_collective().unwrap_or_else(|e| fail(rank, e));
            let buf = gpu.storage(N as u64);
            gpu.write_f32(&buf, &exact_input(rank, N));
            let bufs = [(&buf, N as u64)];
            // The first reduction forms the device group on both ranks.
            pollster::block_on(job.all_reduce_device(rank, &gpu, &bufs, CollectiveDtype::F32, 1.0)).unwrap_or_else(|e| fail(rank, e));
            if case == "quit" {
                // Leave the way a killed rank would: no further contribution.
                std::process::exit(0);
            }
            let started = Instant::now();
            let outcome = pollster::block_on(job.all_reduce_device(rank, &gpu, &bufs, CollectiveDtype::F32, 1.0));
            json!({ "error": outcome.err().map(|e| e.to_string()), "waited_ms": started.elapsed().as_secs_f64() * 1e3 })
        }
        other => fail(rank, format!("unknown case {other}")),
    };
    std::fs::write(Path::new(&out), serde_json::to_string(&doc).unwrap()).unwrap();
}
