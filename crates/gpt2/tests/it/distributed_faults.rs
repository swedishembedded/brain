// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distributed training jobs that
// survive a lost rank without corrupting their state for its clients. If your
// team needs expertise in collective failure handling and checkpoint recovery
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A distributed GPT job that loses a rank, at every point a step can lose one, and is restarted.
//!
//! Two OS processes (this test binary re-executed) train one plan over loopback, committing their exact-resume
//! state every step. In each case one rank fails at step [`FAULT_STEP`]: it is killed during the gradient
//! all-reduce, killed after the ranks agreed to apply the step (during the optimizer step), killed while the step's
//! state is being committed, or stalled inside the all-reduce. Then:
//!
//! - every surviving rank exits with the restart status, having applied no step its peers did not agree to;
//! - the newest committed generation holds every rank's state of the same step, with identical weights;
//! - the job started again with the same arguments resumes from that generation and ends with the weights of the
//!   uninterrupted run, bit for bit (the host transport sums in a fixed rank order, so the run is repeatable).

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use gpt2::model::Gpt;
use gpt2::train::TrainOpts;
use gpt2::GptConfig;
use model::collective::{Collective, CollectiveFuture, Payload};
use model::jobfault::{StepPhase, RESTART_STATUS};
use model::{CheckpointIdentity, GradSync, JobContext, JobPlan, JobSecret, Model};

const CHILD: &str = "distributed_faults::rank_process";
const STEPS: u32 = 6;
/// The step every fault happens in.
const FAULT_STEP: u32 = 4;
const BATCH: u32 = 2;
const BLOCK: u32 = 16;

fn config() -> GptConfig {
    GptConfig { vocab: 0, block_size: BLOCK, n_layers: 1, d_model: 32, n_heads: 2, d_ff: 64 }
}

fn opts() -> TrainOpts {
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
        seed: 11, // seed: every rank and every restart must start from the same weights and data order
        ..Default::default()
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-gpt2-faults-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn free_address() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().to_string()
}

/// One job's files: plan, secret, data, and where each rank writes.
struct Job {
    dir: PathBuf,
    data: PathBuf,
    /// Reduce on the cards over NCCL (one card per rank) instead of through the host transport.
    device: bool,
    /// The plan's op timeout.
    op_timeout_ms: u64,
}

impl Job {
    fn new(name: &str, data: &Path, device: bool) -> Job {
        let dir = scratch(name);
        let mut plan = JobPlan::new(format!("gpt2-faults-{name}"), 2, free_address());
        plan.handshake_timeout_ms = 120_000;
        // A stalled peer is noticed within this; short so the test is. On the cards a rank's first state save also
        // compiles its kernels (seconds) while its peer waits in the vote, so the device run allows more.
        plan.op_timeout_ms = if device { 30_000 } else { 4_000 };
        plan.write(&dir.join("plan.json")).unwrap();
        JobSecret::random().write(&dir.join("job.secret")).unwrap();
        if device {
            plan.collectives = model::CollectivePath::Device;
            plan.write(&dir.join("plan.json")).unwrap();
        }
        Job { dir, data: data.to_path_buf(), device, op_timeout_ms: plan.op_timeout_ms }
    }

    fn spawn(&self, rank: usize, attempt: u32, fault: Option<&str>) -> Child {
        let log = self.dir.join(format!("rank-{rank}.attempt-{attempt}.log"));
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--ignored", "--exact", CHILD, "--test-threads=1", "--nocapture", "--"])
            .arg(format!("--rank={rank}"))
            .arg(format!("--dir={}", self.dir.display()))
            .arg(format!("--data={}", self.data.display()));
        if let Some(fault) = fault {
            command.arg(format!("--fault={fault}"));
        }
        if self.device {
            command.arg("--device-collective");
        }
        // One file for both streams: two handles opened on one path would write over each other.
        let file = std::fs::File::create(&log).unwrap();
        command.stdin(Stdio::null()).stdout(Stdio::from(file.try_clone().unwrap())).stderr(Stdio::from(file)).spawn().unwrap()
    }

    fn log(&self, rank: usize, attempt: u32) -> String {
        std::fs::read_to_string(self.dir.join(format!("rank-{rank}.attempt-{attempt}.log"))).unwrap_or_default()
    }

    /// Run both ranks (`faults[rank]`) to their end; a rank still running once the other ended is killed after a
    /// grace period, as a launcher kills the rest of a step. Returns each rank's exit status (`None`: killed).
    fn run(&self, attempt: u32, faults: [Option<&str>; 2]) -> [Option<i32>; 2] {
        let mut children: Vec<Child> = (0..2).map(|rank| self.spawn(rank, attempt, faults[rank])).collect();
        let mut status = [None, None];
        let deadline = Instant::now() + Duration::from_secs(300);
        let mut first_end: Option<Instant> = None;
        loop {
            for (rank, child) in children.iter_mut().enumerate() {
                if status[rank].is_none() {
                    if let Some(s) = child.try_wait().unwrap() {
                        status[rank] = Some(s.code().unwrap_or(-1));
                        first_end.get_or_insert_with(Instant::now);
                    }
                }
            }
            if status.iter().all(Option::is_some) {
                return status;
            }
            // A survivor notices a dead peer within the plan's op timeout: it gets that long and more before the
            // launcher kills it.
            let grace = Duration::from_millis(self.op_timeout_ms) + Duration::from_secs(30);
            let grace_over = first_end.is_some_and(|t| t.elapsed() > grace);
            if grace_over || Instant::now() > deadline {
                for (rank, child) in children.iter_mut().enumerate() {
                    if status[rank].is_none() {
                        child.kill().ok();
                        child.wait().ok();
                    }
                }
                assert!(!(Instant::now() > deadline), "attempt {attempt} did not end; logs:\n{}\n{}", self.log(0, attempt), self.log(1, attempt));
                return status;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn replica(&self, rank: usize) -> PathBuf {
        self.dir.join(format!("replica-{rank}.safetensors"))
    }

    fn phases(&self, rank: usize) -> String {
        std::fs::read_to_string(self.dir.join(format!("phases-{rank}.log"))).unwrap_or_default()
    }
}

/// A model's trainable tensors by name.
type Weights = Vec<(String, Vec<f32>)>;

fn weights(path: &Path) -> Weights {
    let gpt = Gpt::load(path.to_str().unwrap(), BATCH, BLOCK);
    gpt.param_names().into_iter().map(|n| (n.clone(), gpt.read_weight(&n))).collect()
}

/// The newest generation the whole job committed, and every rank's weights in its own committed file of it.
fn committed_generation(job: &Job) -> (u32, Vec<Weights>) {
    let gens = job.dir.join("state.safetensors.job");
    let step = model::job::CheckpointCommit::latest(&gens).unwrap().expect("a generation the job committed").step;
    let per_rank = (0..2)
        .map(|rank| {
            let own = gens.join(format!("rank{rank}"));
            let manifest = model::job::committed(&own).unwrap().into_iter().find(|c| c.step == step).expect("every rank's manifest of the committed generation");
            manifest.verify(&own).expect("every rank's file of it verifies");
            let st = checkpoint::st::load_safetensors(own.join(format!("gen{step:08}.state")).to_str().unwrap()).unwrap();
            assert_eq!(st.config()["step"], step, "rank {rank}'s committed state is of the committed step");
            let mut w: Weights = st.tensors.iter().filter(|(n, _)| n.starts_with("w:")).map(|(n, v)| (n.clone(), v.clone())).collect();
            w.sort_by(|a, b| a.0.cmp(&b.0));
            w
        })
        .collect();
    (step, per_rank)
}

fn dataset() -> PathBuf {
    let data = scratch("data").join("calculator");
    data::prepare::prepare(data::prepare::Dataset::Calculator, &data, 4000, 1).unwrap();
    data
}

/// Lose a rank to `fault`, check what the failure left, restart the job and compare it with `reference`.
fn lose_and_restart(name: &str, data: &Path, device: bool, faulty_rank: usize, fault: &str, reference: &[(String, Vec<f32>)], survivor_may_apply: bool) {
    let job = Job::new(name, data, device);
    let mut faults = [None, None];
    faults[faulty_rank] = Some(fault);
    let status = job.run(0, faults);
    let survivor = 1 - faulty_rank;
    assert_eq!(status[survivor], Some(RESTART_STATUS), "{name}: the surviving rank exits with the restart status; log:\n{}", job.log(survivor, 0));
    assert_ne!(status[faulty_rank], Some(0), "{name}: the faulty rank did not finish");
    // No step the ranks did not all agree to was applied: before the agreement the survivor stops.
    let applied = format!("applied {FAULT_STEP}\n");
    if !survivor_may_apply {
        assert!(!job.phases(survivor).contains(&applied), "{name}: the survivor applied step {FAULT_STEP} although a peer was lost before agreeing to it");
    }
    let (step, states) = committed_generation(&job);
    assert_eq!(step, FAULT_STEP - 1, "{name}: the newest commit is the last step every rank made durable");
    assert_eq!(states[0], states[1], "{name}: every rank's committed state holds the same weights");

    // Restart: the same command on every rank, no fault.
    let status = job.run(1, [None, None]);
    assert_eq!(status, [Some(0), Some(0)], "{name}: the restarted job completes; logs:\n{}\n{}", job.log(0, 1), job.log(1, 1));
    assert!(job.log(0, 1).contains(&format!("rebuilding from committed generation {}", FAULT_STEP - 1)), "{name}: {}", job.log(0, 1));
    for rank in 0..2 {
        assert_eq!(weights(&job.replica(rank)), reference, "{name}: rank {rank} ends with the uninterrupted run's weights");
    }
    std::fs::remove_dir_all(&job.dir).ok();
}

/// Every fault, then a restart, against the uninterrupted run; `device` reduces over NCCL on the cards.
fn every_fault(device: bool) {
    let data = dataset();
    let reference_job = Job::new("reference", &data, device);
    assert_eq!(reference_job.run(0, [None, None]), [Some(0), Some(0)], "the uninterrupted run; logs:\n{}\n{}", reference_job.log(0, 0), reference_job.log(1, 0));
    let reference = weights(&reference_job.replica(0));
    assert_eq!(weights(&reference_job.replica(1)), reference, "the replicas agree");
    assert!(reference_job.phases(0).contains(&format!("committed {STEPS}\n")), "every step was committed");

    lose_and_restart("killed-in-all-reduce", &data, device, 1, "kill-in-reduce", &reference, false);
    lose_and_restart("killed-in-optimizer-step", &data, device, 1, "kill-after-agreement", &reference, true);
    lose_and_restart("killed-in-commit", &data, device, 1, "kill-after-state-saved", &reference, true);
    lose_and_restart("stalled-in-all-reduce", &data, device, 1, "stall-in-reduce", &reference, false);
    lose_and_restart("coordinator-killed-in-all-reduce", &data, device, 0, "kill-in-reduce", &reference, false);
    std::fs::remove_dir_all(reference_job.dir).ok();
    std::fs::remove_dir_all(data.parent().unwrap()).ok();
}

#[test]
fn a_job_that_loses_a_rank_anywhere_in_a_step_resumes_from_its_last_commit_and_ends_where_it_would_have() {
    every_fault(false);
}

/// The same over NCCL, the gradients reduced on two cards: a rank lost mid-reduction aborts the communicator on the
/// survivor (its async error, or the op deadline for a stalled peer). Needs the CUDA backend selected, two cards
/// and NCCL; skips by name without them.
#[test]
fn a_job_reducing_on_the_cards_survives_the_same_faults() {
    if brain_testutil::gpu_unavailable(gpu_core::testgpu::missing_gpu()) {
        return;
    }
    if gpu_core::backend_name() != "cuda" {
        brain_testutil::skip_unavailable(&format!("needs the CUDA backend selected (BRAIN_BACKEND=cuda), not '{}'", gpu_core::backend_name()));
        return;
    }
    if let Err(e) = gpu_core::testgpu::stage_cards(None, 2) {
        brain_testutil::skip_unavailable(&format!("two ranks need two cards: {e}"));
        return;
    }
    let probe = gpu_core::Gpu::new_on(gpu_core::testgpu::cards()[0], &[]);
    if let Err(e) = gpu_core::devgroup::RankGroup::supported(&probe) {
        brain_testutil::skip_unavailable(&format!("needs NCCL's per-process communicator: {e}"));
        return;
    }
    every_fault(true);
}

/// The job's collective as this rank sees it, with a fault armed at the first all-reduce of [`FAULT_STEP`] (the
/// gradient reduction: the step before it has committed).
struct Faulty<'a> {
    inner: &'a dyn Collective,
    /// The last step this rank committed.
    committed: &'a AtomicU32,
    fault: Option<String>,
}

impl Faulty<'_> {
    fn trip(&self) {
        if self.committed.load(Ordering::SeqCst) + 1 != FAULT_STEP {
            return;
        }
        match self.fault.as_deref() {
            Some("kill-in-reduce") => std::process::abort(),
            Some("stall-in-reduce") => std::thread::sleep(Duration::from_secs(3600)),
            _ => {}
        }
    }
}

impl Collective for Faulty<'_> {
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn all_reduce<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.trip();
        self.inner.all_reduce(rank, local)
    }
    fn all_gather<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.inner.all_gather(rank, local)
    }
    fn reduce_scatter<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.inner.reduce_scatter(rank, local)
    }
    fn broadcast<'a>(&'a self, rank: usize, local: Payload, root: usize) -> CollectiveFuture<'a, Payload> {
        self.inner.broadcast(rank, local, root)
    }
    fn all_reduce_device<'a>(&'a self, rank: usize, gpu: &'a gpu_core::Gpu, bufs: &'a [(&'a gpu_core::DeviceBuffer, u64)], dtype: gpu_core::CollectiveDtype, scale: f32) -> CollectiveFuture<'a, bool> {
        self.trip();
        self.inner.all_reduce_device(rank, gpu, bufs, dtype, scale)
    }
}

/// The rank process: one rank of the job in `--dir`, with an optional `--fault`.
#[test]
#[ignore = "runs as a child process of the fault-injection test"]
fn rank_process() {
    let args: Vec<String> = std::env::args().collect();
    let Some(separator) = args.iter().position(|arg| arg == "--") else { return };
    let after = &args[separator + 1..];
    let flag = |name: &str| after.iter().find_map(|arg| arg.strip_prefix(name).map(str::to_string));
    let (Some(rank), Some(dir), Some(data)) = (flag("--rank=").and_then(|v| v.parse::<usize>().ok()), flag("--dir="), flag("--data=")) else { return };
    let fault = flag("--fault=");
    let dir = PathBuf::from(dir);
    let context = JobContext::from_plan_file(&dir.join("plan.json"), rank, &dir.join("job.secret")).unwrap_or_else(|e| {
        eprintln!("rank {rank}: {e}");
        std::process::exit(2)
    });
    let device = after.iter().any(|a| a == "--device-collective");
    // The host transport, or the job's collective that reduces on the card over NCCL.
    let (host, job);
    let collective: &dyn Collective = if device {
        job = context.job_collective().unwrap_or_else(|e| {
            eprintln!("rank {rank}: {e}");
            std::process::exit(2)
        });
        &job
    } else {
        host = context.collective().unwrap_or_else(|e| {
            eprintln!("rank {rank}: {e}");
            std::process::exit(2)
        });
        &host
    };
    let committed = AtomicU32::new(0);
    let faulty = Faulty { inner: collective, committed: &committed, fault: fault.clone() };
    let phases = std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("phases-{rank}.log"))).unwrap());
    let observe = |phase: StepPhase, step: u32| {
        use std::io::Write;
        let name = match phase {
            StepPhase::Reduced => "reduced",
            StepPhase::Applied => "applied",
            StepPhase::StateSaved => "state-saved",
            StepPhase::Committed => "committed",
        };
        writeln!(phases.lock().unwrap(), "{name} {step}").unwrap();
        if step == FAULT_STEP {
            match (fault.as_deref(), phase) {
                (Some("kill-after-agreement"), StepPhase::Reduced) | (Some("kill-after-state-saved"), StepPhase::StateSaved) => std::process::abort(),
                _ => {}
            }
        }
        if phase == StepPhase::Committed {
            committed.store(step, Ordering::SeqCst);
        }
    };
    let identity = context.plan().identity();
    let sync = GradSync { collective: &faulty, rank, checkpoint: Some(CheckpointIdentity { job: &context.plan().job, plan: &identity }), observer: Some(&observe) };
    let out = dir.join(format!("replica-{rank}.safetensors"));
    let state = dir.join("state.safetensors");
    let train = || gpt2::train::train_distributed_resumable(Path::new(&data), config(), &opts(), None, sync, &state, 1);
    let trained = if device { gpu_core::devices::with_gpu(gpu_core::testgpu::cards()[rank].index, train).unwrap_or_else(|e| Err(std::io::Error::other(e))) } else { train() };
    match trained {
        Ok((_report, replica)) => replica.save(out.to_str().expect("a UTF-8 path")),
        Err(e) => {
            eprintln!("rank {rank}: {e}");
            let status = if model::jobfault::job_failure(&e).is_some() { RESTART_STATUS } else { 3 };
            std::process::exit(status);
        }
    }
}
