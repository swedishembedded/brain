// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distributed training jobs that survive a lost rank without corrupting
// their state for its clients. If your team needs expertise in fault injection for collective failure handling and
// checkpoint recovery then you can procure our services by sending an email to info@swedishembedded.com.

//! The fault-injection harness a model's own test runs a distributed training job under ([`crate::jobfault`]).
//!
//! The ranks are OS processes: the test binary re-executes itself, one process per rank, each running the test's
//! rank function (an ignored test named to [`FaultJob::new`]) over the job's real transport, the host transport over
//! loopback or NCCL on the cards. One rank carries a [`Fault`] armed for one step: it is killed or stalled inside the
//! `n`th call of a collective operation of that step, or killed at a phase of the step ([`FaultyCollective::observe`]).
//! The parent ([`FaultJob::run`]) waits for every rank, kills those still running a grace period after the first ended
//! (a launcher ends the rest of a step that way), and reads what each rank left: its exit status, its log, the phases
//! it reported, and the newest generation the whole job committed ([`FaultJob::committed`]).

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::collective::{Collective, CollectiveFuture, Payload};
use crate::jobfault::{StepPhase, RESTART_STATUS};
use crate::{JobContext, JobPlan, JobSecret};

/// How long a fault-free attempt of a test job may run before the harness fails it.
const ATTEMPT_DEADLINE: Duration = Duration::from_secs(600);

/// Where and how a rank fails, in the step its [`FaultyCollective`] arms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// `kill-in-<op>[@n]`: the process aborts on entering the `n`th call (default 1) of the operation.
    KillIn { op: String, nth: u32 },
    /// `stall-in-<op>[@n]`: the call never returns (the rank hangs as a peer stuck on its card would).
    StallIn { op: String, nth: u32 },
    /// `kill-after-agreement`: aborts once every rank agreed the step's reduction completed, before it is applied.
    KillAfterAgreement,
    /// `kill-after-state-saved`: aborts once this rank's state of the step is durable, before the commit.
    KillAfterStateSaved,
}

impl Fault {
    /// Parse a fault's spelling (the variants' documentation). Operations: `reduce` (an all-reduce), `all-gather`,
    /// `reduce-scatter`.
    pub fn parse(spec: &str) -> Result<Fault, String> {
        let in_op = |rest: &str| -> Result<(String, u32), String> {
            let (op, nth) = match rest.split_once('@') {
                Some((op, n)) => (op, n.parse::<u32>().map_err(|e| format!("fault {spec:?}: call number {n:?}: {e}"))?),
                None => (rest, 1),
            };
            if !["reduce", "all-gather", "reduce-scatter"].contains(&op) || nth == 0 {
                return Err(format!("fault {spec:?}: no operation {op:?} call {nth} (reduce, all-gather or reduce-scatter, from 1)"));
            }
            Ok((op.to_string(), nth))
        };
        match spec {
            "kill-after-agreement" => Ok(Fault::KillAfterAgreement),
            "kill-after-state-saved" => Ok(Fault::KillAfterStateSaved),
            _ => {
                if let Some(rest) = spec.strip_prefix("kill-in-") {
                    let (op, nth) = in_op(rest)?;
                    Ok(Fault::KillIn { op, nth })
                } else if let Some(rest) = spec.strip_prefix("stall-in-") {
                    let (op, nth) = in_op(rest)?;
                    Ok(Fault::StallIn { op, nth })
                } else {
                    Err(format!("unknown fault {spec:?}"))
                }
            }
        }
    }
}

/// The job's collective as a rank of a fault test sees it: every operation goes to `inner`, and the rank's [`Fault`]
/// fires in the step after the one it last committed reaches `fault_step - 1`. A call is counted once per operation:
/// a device call the transport declines (it has no device path) and the payload call that replaces it are one call.
pub struct FaultyCollective {
    inner: Arc<dyn Collective>,
    fault: Option<Fault>,
    fault_step: u32,
    /// The last step this rank committed.
    committed: AtomicU32,
    /// Calls of each operation in the armed step.
    calls: Mutex<HashMap<&'static str, u32>>,
    /// A device call of this operation was declined: the payload call that follows is the same operation.
    declined: Arc<Mutex<Option<&'static str>>>,
    /// The phases this rank completed, one `<phase> <step>` line each.
    phases: Mutex<std::fs::File>,
}

impl FaultyCollective {
    /// `inner` with `fault` armed for step `fault_step`, the rank's phases appended to `phases`.
    pub fn new(inner: Arc<dyn Collective>, fault: Option<Fault>, fault_step: u32, phases: &Path) -> std::io::Result<FaultyCollective> {
        let phases = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(phases)?);
        Ok(FaultyCollective { inner, fault, fault_step, committed: AtomicU32::new(0), calls: Mutex::new(HashMap::new()), declined: Arc::new(Mutex::new(None)), phases })
    }

    fn armed(&self) -> bool {
        self.committed.load(Ordering::SeqCst) + 1 == self.fault_step
    }

    /// One call of `op` begins (`device`: its device form); fire the fault if it is this call.
    fn enter(&self, op: &'static str, device: bool) {
        if !self.armed() {
            return;
        }
        let fallback = !device && self.declined.lock().expect("declined").take_if(|d| *d == op).is_some();
        let call = {
            let mut calls = self.calls.lock().expect("calls");
            let n = calls.entry(op).or_insert(0);
            if !fallback {
                *n += 1;
            }
            *n
        };
        match &self.fault {
            Some(Fault::KillIn { op: at, nth }) if at == op && *nth == call => std::process::abort(),
            Some(Fault::StallIn { op: at, nth }) if at == op && *nth == call => std::thread::sleep(Duration::from_secs(3600)),
            _ => {}
        }
    }

    /// A device call of `op`: when the transport declines it, the payload call that follows is not a new call.
    fn device<'a>(&'a self, op: &'static str, call: CollectiveFuture<'a, bool>) -> CollectiveFuture<'a, bool> {
        self.enter(op, true);
        let declined = Arc::clone(&self.declined);
        Box::pin(async move {
            let done = call.await;
            if matches!(done, Ok(false)) {
                *declined.lock().expect("declined") = Some(op);
            }
            done
        })
    }

    /// The fit loop's report that `phase` of `step` completed on this rank ([`crate::GradSync::observer`]): logged,
    /// a phase fault fired, and a commit arms the next step.
    pub fn observe(&self, phase: StepPhase, step: u32) {
        let name = match phase {
            StepPhase::Reduced => "reduced",
            StepPhase::Applied => "applied",
            StepPhase::StateSaved => "state-saved",
            StepPhase::Committed => "committed",
        };
        writeln!(self.phases.lock().expect("phases"), "{name} {step}").expect("the phase log");
        if step == self.fault_step {
            match (&self.fault, phase) {
                (Some(Fault::KillAfterAgreement), StepPhase::Reduced) | (Some(Fault::KillAfterStateSaved), StepPhase::StateSaved) => std::process::abort(),
                _ => {}
            }
        }
        if phase == StepPhase::Committed {
            self.committed.store(step, Ordering::SeqCst);
            self.calls.lock().expect("calls").clear();
        }
    }
}

impl Collective for FaultyCollective {
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn all_reduce<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.enter("reduce", false);
        self.inner.all_reduce(rank, local)
    }
    fn all_gather<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.enter("all-gather", false);
        self.inner.all_gather(rank, local)
    }
    fn reduce_scatter<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.enter("reduce-scatter", false);
        self.inner.reduce_scatter(rank, local)
    }
    fn broadcast<'a>(&'a self, rank: usize, local: Payload, root: usize) -> CollectiveFuture<'a, Payload> {
        self.inner.broadcast(rank, local, root)
    }
    fn all_reduce_device<'a>(&'a self, rank: usize, gpu: &'a gpu_core::Gpu, bufs: &'a [(&'a gpu_core::DeviceBuffer, u64)], dtype: gpu_core::CollectiveDtype, scale: f32) -> CollectiveFuture<'a, bool> {
        self.device("reduce", self.inner.all_reduce_device(rank, gpu, bufs, dtype, scale))
    }
    fn reduce_scatter_device<'a>(&'a self, rank: usize, gpu: &'a gpu_core::Gpu, send: &'a gpu_core::DeviceBuffer, recv: &'a gpu_core::DeviceBuffer, count: u64, dtype: gpu_core::CollectiveDtype, scale: f32) -> CollectiveFuture<'a, bool> {
        self.device("reduce-scatter", self.inner.reduce_scatter_device(rank, gpu, send, recv, count, dtype, scale))
    }
    fn all_gather_device<'a>(&'a self, rank: usize, gpu: &'a gpu_core::Gpu, send: &'a gpu_core::DeviceBuffer, recv: &'a gpu_core::DeviceBuffer, count: u64, dtype: gpu_core::CollectiveDtype) -> CollectiveFuture<'a, bool> {
        self.device("all-gather", self.inner.all_gather_device(rank, gpu, send, recv, count, dtype))
    }
    fn settle_device<'a>(&'a self, rank: usize, gpu: &'a gpu_core::Gpu) -> CollectiveFuture<'a, ()> {
        self.inner.settle_device(rank, gpu)
    }
}

/// One fault test's job: its plan and secret in `dir`, and how to start its ranks.
pub struct FaultJob {
    pub dir: PathBuf,
    /// The test that runs one rank (`module::name`, an ignored test of the same binary).
    child: String,
    /// The rank function's own arguments.
    args: Vec<String>,
    world: usize,
    /// The plan's op timeout: a survivor notices a lost peer within it.
    pub op_timeout_ms: u64,
}

impl FaultJob {
    /// A job of `world` ranks named `name` in `dir` (emptied first), its ranks run by the test `child` with `args`;
    /// `device` reduces over NCCL on the cards (one per rank) instead of through the host transport.
    pub fn new(dir: PathBuf, name: &str, world: usize, device: bool, child: &str, args: Vec<String>) -> std::io::Result<FaultJob> {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        let coordinator = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.to_string();
        let mut plan = JobPlan::new(name.to_string(), world, coordinator);
        plan.handshake_timeout_ms = 120_000;
        // A stalled peer is noticed within this; short so the test is. On the cards a rank's first state save also
        // compiles its kernels (seconds) while its peer waits in the vote, so the device run allows more.
        plan.op_timeout_ms = if device { 30_000 } else { 4_000 };
        if device {
            plan.collectives = crate::CollectivePath::Device;
        }
        plan.write(&dir.join("plan.json")).map_err(std::io::Error::other)?;
        JobSecret::random().write(&dir.join("job.secret")).map_err(std::io::Error::other)?;
        Ok(FaultJob { dir, child: child.to_string(), args, world, op_timeout_ms: plan.op_timeout_ms })
    }

    fn spawn(&self, rank: usize, attempt: u32, fault: Option<&str>) -> std::io::Result<Child> {
        let mut command = Command::new(std::env::current_exe()?);
        command.args(["--ignored", "--exact", &self.child, "--test-threads=1", "--nocapture", "--"]).arg(format!("--rank={rank}")).arg(format!("--dir={}", self.dir.display())).args(&self.args);
        if let Some(fault) = fault {
            command.arg(format!("--fault={fault}"));
        }
        // One file for both streams: two handles opened on one path would write over each other.
        let file = std::fs::File::create(self.dir.join(format!("rank-{rank}.attempt-{attempt}.log")))?;
        command.stdin(Stdio::null()).stdout(Stdio::from(file.try_clone()?)).stderr(Stdio::from(file)).spawn()
    }

    /// Run every rank (`faults[rank]`) to its end; a rank still running once another ended is killed after the op
    /// timeout and a grace period. Each rank's exit status (`None`: killed).
    ///
    /// # Panics
    /// When the attempt is still running after the harness's deadline (its logs in the message).
    pub fn run(&self, attempt: u32, faults: &[Option<&str>]) -> Vec<Option<i32>> {
        let mut children: Vec<Child> = (0..self.world).map(|rank| self.spawn(rank, attempt, faults.get(rank).copied().flatten()).expect("a rank process")).collect();
        let mut status = vec![None; self.world];
        let deadline = Instant::now() + ATTEMPT_DEADLINE;
        let grace = Duration::from_millis(self.op_timeout_ms) + Duration::from_secs(30);
        let mut first_end: Option<Instant> = None;
        loop {
            for (rank, child) in children.iter_mut().enumerate() {
                if status[rank].is_none() {
                    if let Some(s) = child.try_wait().expect("a rank's status") {
                        status[rank] = Some(s.code().unwrap_or(-1));
                        first_end.get_or_insert_with(Instant::now);
                    }
                }
            }
            if status.iter().all(Option::is_some) {
                return status;
            }
            let late = Instant::now() > deadline;
            if late || first_end.is_some_and(|t| t.elapsed() > grace) {
                for (rank, child) in children.iter_mut().enumerate() {
                    if status[rank].is_none() {
                        child.kill().ok();
                        child.wait().ok();
                    }
                }
                let logs: Vec<String> = (0..self.world).map(|r| self.log(r, attempt)).collect();
                assert!(!late, "attempt {attempt} did not end; logs:\n{}", logs.join("\n"));
                return status;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Rank `rank`'s output of `attempt`.
    pub fn log(&self, rank: usize, attempt: u32) -> String {
        std::fs::read_to_string(self.dir.join(format!("rank-{rank}.attempt-{attempt}.log"))).unwrap_or_default()
    }

    /// Every phase rank `rank` reported, over every attempt.
    pub fn phases(&self, rank: usize) -> String {
        std::fs::read_to_string(phases_path(&self.dir, rank)).unwrap_or_default()
    }

    /// The newest generation of the state `state` (a file name in the job's directory) the whole job committed, and
    /// every rank's state file of it, each verified against that rank's own manifest.
    ///
    /// # Panics
    /// When the job committed no generation, or a rank's file of it is missing or does not verify.
    pub fn committed(&self, state: &str) -> (u32, Vec<PathBuf>) {
        let gens = self.dir.join(format!("{state}.job"));
        let step = crate::job::CheckpointCommit::latest(&gens).expect("the job's manifests").expect("a generation the job committed").step;
        let files = (0..self.world)
            .map(|rank| {
                let own = gens.join(format!("rank{rank}"));
                let manifest = crate::job::committed(&own).expect("the rank's manifests").into_iter().find(|c| c.step == step).expect("every rank's manifest of the committed generation");
                manifest.verify(&own).expect("every rank's file of it verifies");
                own.join(format!("gen{step:08}.state"))
            })
            .collect();
        (step, files)
    }
}

fn phases_path(dir: &Path, rank: usize) -> PathBuf {
    dir.join(format!("phases-{rank}.log"))
}

/// What a rank process was started with ([`FaultJob`]): its rank, the job's directory, its fault, and the rank
/// function's own `--name=value` arguments.
pub struct RankArgs {
    pub rank: usize,
    pub dir: PathBuf,
    pub fault: Option<Fault>,
    after: Vec<String>,
}

impl RankArgs {
    /// The arguments after `--` of this process; `None` when it was not started as a rank (the ignored test run by
    /// hand), so the rank function returns at once.
    pub fn from_env() -> Option<RankArgs> {
        let args: Vec<String> = std::env::args().collect();
        let after = args[args.iter().position(|arg| arg == "--")? + 1..].to_vec();
        let flag = |name: &str| after.iter().find_map(|arg| arg.strip_prefix(name).map(str::to_string));
        let rank = flag("--rank=")?.parse().ok()?;
        let dir = PathBuf::from(flag("--dir=")?);
        let fault = flag("--fault=").map(|f| Fault::parse(&f).expect("a fault"));
        Some(RankArgs { rank, dir, fault, after })
    }

    /// The value of the rank function's argument `--<name>=value`.
    pub fn flag(&self, name: &str) -> Option<String> {
        let prefix = format!("--{name}=");
        self.after.iter().find_map(|arg| arg.strip_prefix(&prefix).map(str::to_string))
    }

    /// Whether the bare argument `--<name>` was given.
    pub fn has(&self, name: &str) -> bool {
        self.after.iter().any(|arg| *arg == format!("--{name}"))
    }

    /// Join the job: its context, and its collective (NCCL on the cards when `device`, the host transport otherwise)
    /// wrapped with this rank's fault for step `fault_step`. Exits with status 2 when the job cannot be joined.
    pub fn join(&self, device: bool, fault_step: u32) -> (JobContext, FaultyCollective) {
        let rank = self.rank;
        let fail = |e: &dyn std::fmt::Display| -> ! {
            eprintln!("rank {rank}: {e}");
            std::process::exit(2)
        };
        let context = JobContext::from_plan_file(&self.dir.join("plan.json"), rank, &self.dir.join("job.secret")).unwrap_or_else(|e| fail(&e));
        let inner: Arc<dyn Collective> = if device { Arc::new(context.job_collective().unwrap_or_else(|e| fail(&e))) } else { Arc::new(context.collective().unwrap_or_else(|e| fail(&e))) };
        let faulty = FaultyCollective::new(inner, self.fault.clone(), fault_step, &phases_path(&self.dir, rank)).unwrap_or_else(|e| fail(&e));
        (context, faulty)
    }

    /// End the rank process on the run's outcome: 0, the restart status for the job's failure, 3 for anything else.
    pub fn exit(&self, outcome: std::io::Result<()>) -> ! {
        match outcome {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("rank {}: {e}", self.rank);
                std::process::exit(if crate::jobfault::job_failure(&e).is_some() { RESTART_STATUS } else { 3 })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fault_names_an_operation_call_or_a_phase() {
        assert_eq!(Fault::parse("kill-in-reduce-scatter@2"), Ok(Fault::KillIn { op: "reduce-scatter".into(), nth: 2 }));
        assert_eq!(Fault::parse("stall-in-all-gather"), Ok(Fault::StallIn { op: "all-gather".into(), nth: 1 }));
        assert_eq!(Fault::parse("kill-after-agreement"), Ok(Fault::KillAfterAgreement));
        for bad in ["kill-in-broadcast", "kill-in-reduce@0", "kill-in-reduce@x", "explode"] {
            assert!(Fault::parse(bad).is_err(), "{bad}");
        }
    }
}
