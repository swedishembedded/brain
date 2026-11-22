// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distributed training jobs that
// survive a lost rank without corrupting their state for its clients. If your
// team needs expertise in collective failure handling, transactional steps and
// checkpoint recovery then you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a distributed training job does when a rank fails: stop before any partial step is committed, say why, and
//! leave a state the job restarts from exactly.
//!
//! **The communicator's state** ([`CommState`], tracked by [`CommHealth`]): `active` while operations complete in
//! their usual time; `suspect` when one took far longer than the ones before it (a peer that is slow or about to
//! fail); `aborting` once an operation failed or passed its deadline (the transport's own op timeout, NCCL's async
//! error on the device path), at which point the process stops taking part; `rebuilding` while a restarted job
//! resumes from its last committed generation, until its first operation completes. Rebuilding is a job-level
//! restart: a lost rank's process is gone, so every rank starts again from the same commit (shrinking the group or
//! replacing a rank in place is not done).
//!
//! **The transactional step.** Every rank reduces its gradients, then votes over the job's transport that its
//! reduction completed ([`agree`]); only a rank that sees every vote applies the optimizer step. A rank that dies
//! anywhere in the step leaves its peers unable to finish the vote, so none applies it. What survives a failure is
//! what was committed: each rank writes its exact-resume state as a numbered generation, the ranks vote that every
//! state is durable, and only then does rank 0 write the generation's commit ([`Generations`]). A crash at any point
//! before that leaves the previous generation the newest, every rank's file of it still on disk.
//!
//! **A parameter-sharded step** (ZeRO-3, [`crate::zero::Sharding::Parameters`]) is not voted: each unit is updated on
//! its owners as soon as its reduce-scatter completes, inside the backward, and holding every reduced slice until a
//! vote would cost the whole gradient's slice per rank. Its guarantee is the commit's instead: a rank whose step
//! failed part way never votes that its state is durable, so no rank commits that step or any later one, and the
//! restart reloads the last committed generation; the failed step's partial updates are discarded with the process.
//! The model reports the failure ([`crate::Model::collective_failure`]) and the fit loop ends on it as here.
//!
//! **The exit.** A failure of the job's collective is a [`JobFailure`]; a process that ends on one exits with
//! [`RESTART_STATUS`], which tells a launcher to restart every rank with the same plan, secret and arguments. The run
//! resumes from the newest generation every rank committed and continues as the uninterrupted run would have.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::collective::{Collective, CollectiveResult, Payload};
use crate::job::{committed, CheckpointCommit};
use crate::CheckpointIdentity;

/// The exit status of a rank whose job lost a peer or its collective: restart every rank from the last commit.
/// (`EX_TEMPFAIL`: a temporary failure, retrying is right.)
pub const RESTART_STATUS: i32 = 75;

/// Where in a training step the job's collective is used, or a step's state becomes durable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepPhase {
    /// The gradients are reduced and every rank agreed they were: the step may be applied.
    Reduced,
    /// The optimizer step is applied on this rank.
    Applied,
    /// This rank's exact-resume state of the step is durable.
    StateSaved,
    /// The step's generation is committed (every rank's state named in rank 0's manifest).
    Committed,
}

/// The communicator's state, as the module documentation describes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommState {
    Active,
    Suspect,
    Aborting,
    Rebuilding,
}

impl CommState {
    pub fn name(self) -> &'static str {
        match self {
            CommState::Active => "active",
            CommState::Suspect => "suspect",
            CommState::Aborting => "aborting",
            CommState::Rebuilding => "rebuilding",
        }
    }
}

/// The job's collective failed: this rank stopped inside `step`, and nothing of the step is committed. A replicated or
/// optimizer-sharded rank applied none of it; a parameter-sharded rank may hold some of the step's units updated,
/// which the restart from the last commit discards.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobFailure {
    /// The step (1-based) whose collective failed.
    pub step: u32,
    /// What the step was doing.
    pub phase: &'static str,
    /// The communicator's states up to the failure.
    pub states: Vec<CommState>,
    /// The transport's own account.
    pub cause: String,
}

impl fmt::Display for JobFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path: Vec<&str> = self.states.iter().map(|s| s.name()).collect();
        write!(f, "the job's collective failed in step {} ({}): {} [communicator {}]; nothing of the step is committed, restart the job from its last commit", self.step, self.phase, self.cause, path.join(" -> "))
    }
}

impl std::error::Error for JobFailure {}

/// The [`JobFailure`] `err` carries, if any: the one error a launcher restarts the job for.
pub fn job_failure(err: &io::Error) -> Option<&JobFailure> {
    let mut source: Option<&(dyn std::error::Error + 'static)> = err.get_ref().map(|e| e as &(dyn std::error::Error + 'static));
    while let Some(e) = source {
        if let Some(failure) = e.downcast_ref::<JobFailure>() {
            return Some(failure);
        }
        source = e.source();
    }
    None
}

/// A collective operation slower than this many times the usual one marks the communicator suspect.
const SUSPECT_FACTOR: f64 = 10.0;

/// And never one under this: a short operation's time varies more than its usual time says.
const SUSPECT_FLOOR: Duration = Duration::from_secs(1);

/// The communicator's state through a run, driven by every collective operation's outcome and time.
#[derive(Debug)]
pub struct CommHealth {
    states: Vec<CommState>,
    /// The usual time of an operation, a moving average.
    usual: Option<f64>,
}

impl Default for CommHealth {
    fn default() -> CommHealth {
        CommHealth { states: vec![CommState::Active], usual: None }
    }
}

impl CommHealth {
    /// A run that resumes generation `step` starts rebuilding, until its first operation completes.
    pub fn rebuilding(step: u32) -> CommHealth {
        eprintln!("comm: rebuilding from committed generation {step}");
        CommHealth { states: vec![CommState::Rebuilding], usual: None }
    }

    pub fn state(&self) -> CommState {
        *self.states.last().expect("a state")
    }

    /// Every state the communicator went through, in order (repeats collapsed).
    pub fn history(&self) -> &[CommState] {
        &self.states
    }

    fn enter(&mut self, state: CommState) {
        if self.state() != state {
            self.states.push(state);
        }
    }

    /// Run one collective operation of `step`'s `phase`: its outcome and time move the state; a failure is a
    /// [`JobFailure`].
    pub fn guarded<T>(&mut self, step: u32, phase: &'static str, op: impl FnOnce() -> CollectiveResult<T>) -> io::Result<T> {
        let started = Instant::now();
        let outcome = op();
        let took = started.elapsed();
        match outcome {
            Ok(value) => {
                let secs = took.as_secs_f64();
                let slow = self.usual.is_some_and(|usual| took > SUSPECT_FLOOR && secs > usual * SUSPECT_FACTOR);
                if slow {
                    eprintln!("comm: step {step} {phase} took {took:?}, far over the usual {:.3?}: suspect", Duration::from_secs_f64(self.usual.unwrap_or_default()));
                    self.enter(CommState::Suspect);
                } else {
                    self.enter(CommState::Active);
                }
                self.usual = Some(self.usual.map_or(secs, |u| 0.9 * u + 0.1 * secs));
                Ok(value)
            }
            Err(e) => Err(self.failed(step, phase, e.to_string())),
        }
    }

    /// The job's collective failed in `step`'s `phase` for `cause` (reported by work that ran the collective itself,
    /// such as a parameter-sharded model's step): the state moves to aborting, and the failure is a [`JobFailure`].
    pub fn failed(&mut self, step: u32, phase: &'static str, cause: String) -> io::Error {
        self.enter(CommState::Aborting);
        io::Error::other(JobFailure { step, phase, states: self.states.clone(), cause })
    }
}

/// Every rank of `collective` states that `phase` of `step` completed; `Ok` only when every rank did. A rank that
/// is gone or stalled fails the vote on every live rank (the transport's deadline), so none goes on.
pub fn agree(collective: &dyn Collective, rank: usize, health: &mut CommHealth, step: u32, phase: &'static str) -> io::Result<()> {
    let votes = health.guarded(step, phase, || pollster::block_on(collective.all_reduce(rank, Payload::f32(vec![1.0]))))?;
    let world = collective.world_size();
    if votes.data.first().copied() != Some(world as f32) {
        let states = health.states.clone();
        health.enter(CommState::Aborting);
        return Err(io::Error::other(JobFailure { step, phase, states, cause: format!("{:?} of {world} ranks agreed", votes.data.first()) }));
    }
    Ok(())
}

/// A distributed run's exact-resume states as numbered generations, in a directory beside each rank's state path
/// (`<state>.job/`): every rank keeps its own files of generation `k` (the state after `k` steps) under `rank<r>/`,
/// with its own manifest of them, and rank 0 keeps the job's manifest of every generation the whole job made durable.
/// A rank never reads another rank's files, so the state paths may be on node-local disks.
pub struct Generations {
    dir: PathBuf,
    rank: usize,
    world: usize,
    job: String,
    plan: String,
}

impl Generations {
    /// The generations of the run whose state path (on this rank) is `state`, for `rank` of `world` in the job
    /// `identity` names.
    pub fn new(state: &Path, rank: usize, world: usize, identity: CheckpointIdentity<'_>) -> io::Result<Generations> {
        let mut name = state.file_name().ok_or_else(|| io::Error::other(format!("{} names no file", state.display())))?.to_os_string();
        name.push(".job");
        let dir = state.with_file_name(name);
        let generations = Generations { dir, rank, world, job: identity.job.to_string(), plan: identity.plan.to_string() };
        std::fs::create_dir_all(generations.rank_dir())?;
        Ok(generations)
    }

    fn rank_dir(&self) -> PathBuf {
        self.dir.join(format!("rank{}", self.rank))
    }

    /// This rank's state file of generation `step`.
    pub fn path(&self, step: u32) -> PathBuf {
        self.rank_dir().join(format!("gen{step:08}.state"))
    }

    fn role(&self) -> String {
        format!("state.rank{}", self.rank)
    }

    /// The newest generation the whole job committed (rank 0's manifest, sent to every rank over `collective`), and
    /// this rank's file of it verified against this rank's own manifest; `None` for a run that has none, which
    /// starts from the beginning. Every rank calls it.
    ///
    /// # Errors
    /// Generations committed by another job, plan or world (a run is never started over them), a rank missing its
    /// file of the committed generation, or a file that no longer matches its manifest.
    pub fn latest(&self, collective: &dyn Collective) -> io::Result<Option<u32>> {
        // Rank 0's answer: the newest generation, `-1` for none, or `-2` when the directory holds another run's.
        let local = if self.rank == 0 {
            let commits = committed(&self.dir).unwrap_or_default();
            match commits.iter().find(|c| c.compatible_with(&self.job, &self.plan, self.world)) {
                Some(commit) => commit.step as f32,
                None if commits.is_empty() => -1.0,
                None => -2.0,
            }
        } else {
            0.0
        };
        let agreed = pollster::block_on(collective.broadcast(self.rank, Payload::f32(vec![local]), 0)).map_err(io::Error::other)?;
        let step = match agreed.data.first().copied() {
            Some(s) if s >= 0.0 => s as u32,
            Some(-2.0) => {
                return Err(io::Error::other(format!(
                    "{} holds generations of another job, plan or world (the newest names {:?}); refusing to start over them",
                    self.dir.display(),
                    committed(&self.dir).ok().and_then(|c| c.first().map(|c| (c.job.clone(), c.world_size)))
                )))
            }
            _ => return Ok(None),
        };
        let mine = committed(&self.rank_dir())?.into_iter().find(|c| c.step == step && c.compatible_with(&self.job, &self.plan, self.world));
        let mine = mine.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("rank {} has no manifest of committed generation {step} in {}", self.rank, self.rank_dir().display())))?;
        mine.verify(&self.rank_dir()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("generation {step}: {e}")))?;
        Ok(Some(step))
    }

    /// Commit generation `step` once this rank's state of it is written: each rank writes its own manifest of its
    /// file, every rank votes it is durable, rank 0 writes the job's manifest, every rank votes it saw that done,
    /// and each drops its older generations.
    pub fn commit(&self, collective: &dyn Collective, health: &mut CommHealth, step: u32) -> io::Result<()> {
        let role = self.role();
        let path = self.path(step);
        CheckpointCommit::build(&self.job, &self.plan, step, self.world, &self.rank_dir(), &[(role.as_str(), path.as_path())])?.write(&self.rank_dir())?;
        agree(collective, self.rank, health, step, "agreement that every rank's state is durable")?;
        if self.rank == 0 {
            // The job's manifest names rank 0's own file: the vote above is what says every other rank's is durable.
            let relative = Path::new(&format!("rank{}", self.rank)).join(path.file_name().expect("a file name"));
            CheckpointCommit::build(&self.job, &self.plan, step, self.world, &self.dir, &[(role.as_str(), self.dir.join(relative).as_path())])?.write(&self.dir)?;
        }
        agree(collective, self.rank, health, step, "agreement that the generation is committed")?;
        self.prune(step);
        Ok(())
    }

    /// Remove this rank's generations and manifests older than `step` (rank 0 also the job's).
    fn prune(&self, step: u32) {
        let old = |name: &str| -> bool {
            let number = |prefix: &str| name.strip_prefix(prefix).and_then(|rest| rest.get(..8)).and_then(|digits| digits.parse::<u32>().ok());
            number("gen").or_else(|| number("commit-")).is_some_and(|g| g < step)
        };
        let mut dirs = vec![self.rank_dir()];
        if self.rank == 0 {
            dirs.push(self.dir.clone());
        }
        for dir in dirs {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                if old(&entry.file_name().to_string_lossy()) {
                    let path = entry.path();
                    let _ = if path.is_dir() { std::fs::remove_dir_all(&path) } else { std::fs::remove_file(&path) };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collective::CollectiveError;

    #[test]
    fn the_communicator_goes_suspect_on_a_slow_op_and_aborting_on_a_failed_one() {
        let mut health = CommHealth { states: vec![CommState::Active], usual: Some(0.0001) };
        health.guarded(1, "reduce", || {
            std::thread::sleep(SUSPECT_FLOOR + Duration::from_millis(50));
            Ok(())
        })
        .unwrap();
        assert_eq!(health.state(), CommState::Suspect);
        let err = health.guarded(2, "reduce", || Err::<(), _>(CollectiveError::Transport("peer 1 closed the connection".into()))).unwrap_err();
        let failure = job_failure(&err).expect("a job failure");
        assert_eq!((failure.step, failure.phase), (2, "reduce"));
        assert_eq!(failure.states, [CommState::Active, CommState::Suspect, CommState::Aborting]);
        assert!(err.to_string().contains("nothing of the step is committed"), "{err}");
        let mut rebuilt = CommHealth::rebuilding(4);
        rebuilt.guarded(5, "reduce", || Ok(())).unwrap();
        assert_eq!(rebuilt.history(), [CommState::Rebuilding, CommState::Active]);
        assert!(job_failure(&io::Error::other("unrelated")).is_none());
    }

    #[test]
    fn a_generation_is_the_newest_one_the_whole_job_committed_and_older_ones_are_dropped() {
        let dir = std::env::temp_dir().join(format!("brain-generations-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let identity = CheckpointIdentity { job: "j", plan: "p" };
        // Each rank its own state path, as on node-local disks.
        let ranks: Vec<Generations> = (0..2).map(|r| Generations::new(&dir.join(format!("node{r}/state")), r, 2, identity).unwrap()).collect();
        for g in &ranks {
            std::fs::create_dir_all(g.rank_dir()).unwrap();
        }
        let coll = crate::HostCollective::new(2);
        let together = |f: &(dyn Fn(&Generations) -> io::Result<Option<u32>> + Sync)| -> Vec<io::Result<Option<u32>>> {
            std::thread::scope(|scope| {
                let handles: Vec<_> = ranks.iter().map(|g| scope.spawn(move || f(g))).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        let latest = |coll: &std::sync::Arc<crate::HostCollective>| together(&|g: &Generations| g.latest(&**coll));
        assert!(latest(&coll).iter().all(|r| matches!(r, Ok(None))), "nothing committed yet");
        // Generations 2 and 4 written and committed in turn, then generation 6 written but never committed.
        for step in [2u32, 4] {
            let outcomes = together(&|g: &Generations| {
                std::fs::write(g.path(step), format!("rank {} step {step}", g.rank))?;
                g.commit(&*coll, &mut CommHealth::default(), step).map(|()| None)
            });
            assert!(outcomes.iter().all(|r| r.is_ok()), "{outcomes:?}");
        }
        for g in &ranks {
            std::fs::write(g.path(6), "uncommitted").unwrap();
        }
        assert!(latest(&coll).iter().all(|r| matches!(r, Ok(Some(4)))), "6 was written but never committed");
        assert!(!ranks[0].path(2).exists() && !ranks[1].path(2).exists() && ranks[1].path(4).exists(), "older generations are dropped");
        // A file changed after its commit is refused, never resumed from.
        std::fs::write(ranks[1].path(4), "tampered").unwrap();
        let outcomes = latest(&coll);
        assert!(outcomes[0].is_ok() && outcomes[1].is_err(), "{outcomes:?}");
        // Another world's generations are refused, not started over.
        let other = Generations::new(&dir.join("node0/state"), 0, 1, identity).unwrap();
        let err = other.latest(&*crate::HostCollective::new(1)).unwrap_err();
        assert!(err.to_string().contains("another job, plan or world"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
