// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distributed training jobs that launchers can
// start on any number of machines for its clients. If your team needs expertise
// in multi-node training runtimes or cluster job contracts, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One distributed training job: the plan every rank is given, and the group
//! it joins.
//!
//! A launcher (a scheduler, a batch system, or a test) starts one process per
//! rank with the same [`JobPlan`] and a different rank. The plan is the
//! agreement the processes make before any tensor moves: it names the job, the
//! world size, the parallelism grid and the coordinator. Every rank sends the
//! plan's exact bytes when it joins, so a process started with a different
//! world size, grid or job cannot silently take part in someone else's
//! collective - the coordinator refuses it and says why.
//!
//! The plan is deliberately a value the launcher writes, not something brain
//! reads from a scheduler: brain stays runnable under plain `srun`, a batch
//! script or a single process, and neither Slurm nor any other scheduler is a
//! dependency of this crate.
//!
//! Transport failures are bounded: the handshake and every collective op carry
//! a timeout from the plan, so a rank that dies cannot block its peers forever.
//! A peer that fails validation receives an explicit abort with the reason
//! instead of waiting for a result that will never arrive.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use sha2::{Digest as _, Sha256};

use crate::collective::CollectiveError;
use crate::grid::Grid;
use crate::netcollective::{GroupHello, NetworkCollective};

/// How long a rank waits for its peers to join, when the plan does not say.
pub const DEFAULT_HANDSHAKE_TIMEOUT_MS: u64 = 120_000;

/// How long one collective op may block on a peer, when the plan does not say.
pub const DEFAULT_OP_TIMEOUT_MS: u64 = 600_000;

/// A distributed training job as its launcher describes it to every rank.
///
/// One plan is written once per job and handed to every process unchanged; the
/// rank is the only per-process part and travels separately ([`JobContext`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobPlan {
    /// The job's durable identity, as the launcher names it.
    pub job: String,
    /// How many ranks the job has.
    pub world_size: usize,
    /// Tensor-parallel group size.
    pub tp: usize,
    /// Pipeline-parallel group size.
    pub pp: usize,
    /// Data-parallel group size; `tp * pp * dp` must equal `world_size`.
    pub dp: usize,
    /// Where rank 0 binds and every other rank connects (`host:port`).
    pub coordinator: String,
    /// How long the group waits for every rank to join.
    pub handshake_timeout_ms: u64,
    /// How long one collective op may wait on a peer.
    pub op_timeout_ms: u64,
}

impl JobPlan {
    /// The plan for a job.
    pub fn new(job: impl Into<String>, world_size: usize, coordinator: impl Into<String>) -> JobPlan {
        JobPlan {
            job: job.into(),
            world_size,
            tp: 1,
            pp: 1,
            dp: world_size.max(1),
            coordinator: coordinator.into(),
            handshake_timeout_ms: DEFAULT_HANDSHAKE_TIMEOUT_MS,
            op_timeout_ms: DEFAULT_OP_TIMEOUT_MS,
        }
    }

    /// The plan as every rank must receive it: the bytes compared at join.
    pub fn identity(&self) -> String {
        serde_json::to_string(self).expect("a job plan is plain data")
    }

    /// The rank-to-coordinate grid the plan describes.
    pub fn grid(&self) -> Grid {
        Grid::new(self.tp, self.pp, self.dp)
    }

    /// Check what every rank relies on before starting: a non-empty job,
    /// a grid whose dimensions multiply to the world size, a coordinator that
    /// can be dialled, and non-zero timeouts.
    pub fn validate(&self) -> Result<(), String> {
        if self.job.trim().is_empty() {
            return Err("a job plan names no job".into());
        }
        if self.world_size == 0 {
            return Err(format!("job {:?} has no ranks", self.job));
        }
        if self.tp == 0 || self.pp == 0 || self.dp == 0 {
            return Err(format!("job {:?} has a zero-sized parallelism dimension", self.job));
        }
        let product = self.tp.saturating_mul(self.pp).saturating_mul(self.dp);
        if product != self.world_size {
            return Err(format!(
                "job {:?} is tp={} x pp={} x dp={} ({product} ranks) but declares world_size {}",
                self.job, self.tp, self.pp, self.dp, self.world_size
            ));
        }
        if !self.coordinator.contains(':') {
            return Err(format!("job {:?} coordinator {:?} is not host:port", self.job, self.coordinator));
        }
        if self.handshake_timeout_ms == 0 || self.op_timeout_ms == 0 {
            return Err(format!("job {:?} asks for a zero timeout", self.job));
        }
        Ok(())
    }

    /// Read a plan a launcher wrote.
    pub fn read(path: &Path) -> Result<JobPlan, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let plan: JobPlan = serde_json::from_slice(&bytes).map_err(|e| format!("parsing {}: {e}", path.display()))?;
        plan.validate()?;
        Ok(plan)
    }

    /// Write a plan so a launcher can hand the same bytes to every rank: a
    /// temporary file beside it, then a rename. A reader sees the whole file or
    /// none of it.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let mut body = serde_json::to_vec_pretty(self).map_err(|e| format!("serialising the job plan: {e}"))?;
        body.push(b'\n');
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, &body).map_err(|e| format!("writing {}: {e}", temporary.display()))?;
        std::fs::rename(&temporary, path).map_err(|e| format!("renaming {}: {e}", temporary.display()))
    }

    fn handshake_timeout(&self) -> Duration {
        Duration::from_millis(self.handshake_timeout_ms)
    }

    fn op_timeout(&self) -> Duration {
        Duration::from_millis(self.op_timeout_ms)
    }
}

/// This process's part of a job: the plan every rank shares and the rank it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobContext {
    plan: JobPlan,
    rank: usize,
}

impl JobContext {
    /// The context of `rank` in `plan`, refusing a rank the plan does not have.
    pub fn new(plan: JobPlan, rank: usize) -> Result<JobContext, String> {
        plan.validate()?;
        if rank >= plan.world_size {
            return Err(format!("rank {rank} is outside job {:?}'s {} ranks", plan.job, plan.world_size));
        }
        Ok(JobContext { plan, rank })
    }

    /// The context a launcher asked for, read from a plan file.
    pub fn from_plan_file(path: &Path, rank: usize) -> Result<JobContext, String> {
        JobContext::new(JobPlan::read(path)?, rank)
    }

    /// The plan every rank was given.
    pub fn plan(&self) -> &JobPlan {
        &self.plan
    }

    /// This process's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The grid coordinate this rank owns.
    pub fn coord(&self) -> crate::grid::Coord {
        self.plan.grid().coord(self.rank)
    }

    /// Join the job's collective, or say why the group could not be formed.
    ///
    /// Rank 0 binds the coordinator address and waits for every other rank;
    /// every other rank dials it. Both sides exchange the plan's identity, so a
    /// rank started with a different plan is refused with the mismatch rather
    /// than joining the wrong collective.
    pub fn collective(&self) -> Result<NetworkCollective, CollectiveError> {
        let hello = GroupHello { plan: self.plan.identity(), world: self.plan.world_size, rank: self.rank };
        if self.rank == 0 {
            let listener = std::net::TcpListener::bind(&self.plan.coordinator)
                .map_err(|e| CollectiveError::Transport(format!("job {:?} rank 0 cannot bind {}: {e}", self.plan.job, self.plan.coordinator)))?;
            NetworkCollective::coordinator_checked(&listener, &hello, self.plan.handshake_timeout(), self.plan.op_timeout())
        } else {
            NetworkCollective::worker_checked(&hello, &self.plan.coordinator, self.plan.handshake_timeout(), self.plan.op_timeout())
        }
    }
}

// ---- the committed checkpoint of a distributed job -------------------------

/// One file a commit names: its role, its path relative to the checkpoint
/// directory, and the digest of its bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitFile {
    /// What the file is for ("weights", "state", "config", ...).
    pub role: String,
    /// The file's path inside the checkpoint directory.
    pub path: String,
    /// Its exact length in bytes.
    pub len: u64,
    /// `sha256:<hex>` of its bytes.
    pub sha256: String,
}

/// One committed checkpoint generation of a distributed job.
///
/// The manifest is written once, atomically, by rank 0 after the files it
/// names are durable: a step is committed when and only when its manifest
/// exists. A crash before the rename leaves the previous generation the newest
/// one, and a reader that finds a manifest verifies every file against it
/// rather than trusting a directory listing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCommit {
    /// The job this checkpoint belongs to.
    pub job: String,
    /// The plan identity every rank ran under.
    pub plan: String,
    /// The global optimizer step committed.
    pub step: u32,
    /// How many ranks committed it.
    pub world_size: usize,
    /// Every file the commit is made of.
    pub files: Vec<CommitFile>,
}

impl CheckpointCommit {
    /// Measure `files` (role, path relative to `dir`) and build the manifest.
    pub fn build(job: &str, plan: &str, step: u32, world_size: usize, dir: &Path, files: &[(&str, &Path)]) -> io::Result<CheckpointCommit> {
        if files.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "a checkpoint commit names no files"));
        }
        let mut entries = Vec::with_capacity(files.len());
        for (role, path) in files {
            let relative = path.strip_prefix(dir).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, format!("{} is not inside {}", path.display(), dir.display())))?;
            let relative = relative.to_string_lossy().into_owned();
            if relative.split('/').any(|part| part == "..") {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{} leaves the checkpoint directory", path.display())));
            }
            let (len, sha256) = file_digest(path)?;
            entries.push(CommitFile { role: (*role).to_string(), path: relative, len, sha256 });
        }
        entries.sort_by_key(|entry| entry.role.clone());
        Ok(CheckpointCommit { job: job.to_string(), plan: plan.to_string(), step, world_size, files: entries })
    }

    /// Write the manifest as `commit-<step>.json`, durable before its name
    /// exists: a temporary file, fsync, then a rename.
    pub fn write(&self, dir: &Path) -> io::Result<PathBuf> {
        let path = dir.join(format!("commit-{:08}.json", self.step));
        let temporary = dir.join(format!("commit-{:08}.json.tmp", self.step));
        let mut body = serde_json::to_vec_pretty(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        body.push(b'\n');
        {
            use std::io::Write as _;
            let mut file = std::fs::File::create(&temporary)?;
            file.write_all(&body)?;
            file.sync_all()?;
        }
        std::fs::rename(&temporary, &path)?;
        if let Ok(root) = std::fs::File::open(dir) {
            root.sync_all().ok();
        }
        Ok(path)
    }

    /// The commit a manifest file names, verified against its own content.
    pub fn read(path: &Path) -> Result<CheckpointCommit, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let commit: CheckpointCommit = serde_json::from_slice(&bytes).map_err(|e| format!("parsing {}: {e}", path.display()))?;
        if commit.files.is_empty() {
            return Err(format!("{} names no files", path.display()));
        }
        Ok(commit)
    }

    /// The newest committed generation in `dir`, if any. A temporary manifest
    /// left by a crashed commit is not a generation.
    pub fn latest(dir: &Path) -> io::Result<Option<CheckpointCommit>> {
        Ok(committed(dir)?.into_iter().next())
    }

    /// Whether this commit belongs to a rank that runs under `plan` with
    /// `world_size`: the same job, plan bytes and world.
    pub fn compatible_with(&self, job: &str, plan: &str, world_size: usize) -> bool {
        self.job == job && self.plan == plan && self.world_size == world_size
    }

    /// Check every named file exists with the length and digest the manifest
    /// records. A commit that fails this must not be resumed from.
    pub fn verify(&self, dir: &Path) -> Result<(), String> {
        for file in &self.files {
            let path = dir.join(&file.path);
            let (len, sha256) = file_digest(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            if len != file.len {
                return Err(format!("{} is {len} bytes, the commit records {}", path.display(), file.len));
            }
            if sha256 != file.sha256 {
                return Err(format!("{} hashes to {sha256}, the commit records {}", path.display(), file.sha256));
            }
        }
        Ok(())
    }
}

/// Every committed generation in `dir`, newest step first.
pub fn committed(dir: &Path) -> io::Result<Vec<CheckpointCommit>> {
    let mut commits = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(step) = name.strip_prefix("commit-").and_then(|rest| rest.strip_suffix(".json")) else { continue };
        if step.parse::<u32>().is_err() {
            continue;
        }
        if let Ok(commit) = CheckpointCommit::read(&entry.path()) {
            commits.push(commit);
        }
    }
    commits.sort_by_key(|commit| std::cmp::Reverse(commit.step));
    Ok(commits)
}

/// The length and `sha256:<hex>` of a file, streamed.
fn file_digest(path: &Path) -> io::Result<(u64, String)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut len = 0u64;
    loop {
        let n = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        hasher.update(&buffer[..n]);
        len += n as u64;
    }
    Ok((len, format!("sha256:{:x}", hasher.finalize())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plan_must_describe_a_grid_that_covers_its_world() {
        let mut plan = JobPlan::new("run", 4, "127.0.0.1:7000");
        assert_eq!(plan.grid().world_size(), 4);
        plan.dp = 3;
        let err = plan.validate().expect_err("3 x 1 x 1 is not four ranks");
        assert!(err.contains("world_size 4"), "{err}");
    }

    #[test]
    fn a_rank_outside_the_world_is_refused() {
        let plan = JobPlan::new("run", 2, "127.0.0.1:7000");
        let err = JobContext::new(plan, 2).expect_err("rank 2 of a two-rank job");
        assert!(err.contains("outside"), "{err}");
    }

    #[test]
    fn a_plan_round_trips_through_its_file() {
        let dir = std::env::temp_dir().join(format!("brain-job-plan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plan.json");
        let plan = JobPlan::new("run", 2, "127.0.0.1:7000");
        plan.write(&path).unwrap();
        assert_eq!(JobPlan::read(&path).unwrap(), plan);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-commit-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_commit_is_the_generation_a_reader_verifies() {
        let dir = scratch("commit");
        let weights = dir.join("weights.safetensors");
        std::fs::write(&weights, b"weights-v1").unwrap();
        let plan = JobPlan::new("run", 2, "127.0.0.1:7000");
        let commit = CheckpointCommit::build("run", &plan.identity(), 4, 2, &dir, &[("weights", &weights)]).unwrap();
        commit.write(&dir).unwrap();
        assert_eq!(CheckpointCommit::latest(&dir).unwrap().as_ref(), Some(&commit));
        commit.verify(&dir).unwrap();
        assert!(commit.compatible_with("run", &plan.identity(), 2));
        assert!(!commit.compatible_with("other", &plan.identity(), 2));
        assert!(!commit.compatible_with("run", &plan.identity(), 3));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_crashed_commit_leaves_the_previous_generation_and_a_tampered_file_is_refused() {
        let dir = scratch("crash");
        let weights = dir.join("weights.safetensors");
        std::fs::write(&weights, b"weights-v1").unwrap();
        let plan = JobPlan::new("run", 1, "127.0.0.1:7000");
        let first = CheckpointCommit::build("run", &plan.identity(), 4, 1, &dir, &[("weights", &weights)]).unwrap();
        first.write(&dir).unwrap();

        // A commit that died before its rename is not a generation.
        std::fs::write(dir.join("commit-00000005.json.tmp"), b"{not json").unwrap();
        assert_eq!(CheckpointCommit::latest(&dir).unwrap().unwrap().step, 4);

        // A newer generation whose file no longer matches is not resumable.
        std::fs::write(&weights, b"weights-v2").unwrap();
        let second = CheckpointCommit::build("run", &plan.identity(), 5, 1, &dir, &[("weights", &weights)]).unwrap();
        second.write(&dir).unwrap();
        assert_eq!(CheckpointCommit::latest(&dir).unwrap().unwrap().step, 5);
        std::fs::write(&weights, b"weights-v3").unwrap();
        assert!(second.verify(&dir).is_err());
        // The older generation's file is gone as well, so neither is resumable.
        assert!(first.verify(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
