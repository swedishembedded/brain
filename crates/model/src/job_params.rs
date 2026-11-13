// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The parameters that make a training action one rank of a distributed
//! job, declared once for every trainer that can run that way: the plan
//! every rank shares, this rank's place in it, the secret that proves it
//! belongs, and where to write what the run's own collective cost.
//!
//! [`JobRequest::run`] joins the job, hands the trainer the group to reduce
//! gradients over, and writes the collective profile when one was asked for.
//!
//! Swedish Embedded AB implements distributed training runs across its
//! clients' own machines. If your team needs expertise in data-parallel
//! training and its cost accounting then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use capability::{Invocation, ParamSpec, ParamType};

use crate::{CheckpointIdentity, CollectiveProfile, GradSync, JobContext, TimedCollective};

/// Most ranks a plan may place this process at.
const MAX_RANK: f64 = 65_535.0;

/// The distributed-job parameters a trainer takes; all host-side files.
pub fn specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("job_plan", ParamType::Str, "the plan file every rank of a distributed job shares; with job_rank and job_secret, this run is one rank of it").host_resolved(),
        ParamSpec::new("job_rank", ParamType::Int, "this rank's place in the job plan").min(0.0).max(MAX_RANK),
        ParamSpec::new("job_secret", ParamType::Str, "the owner-only secret file every rank shares (openssl rand -hex 32)").host_resolved(),
        ParamSpec::new("profile_collectives", ParamType::Str, "a distributed run writes what its own collectives cost, one JSON document per rank beside this path").host_resolved(),
    ]
}

/// One rank of a distributed job, as an invocation of [`specs`] names it.
#[derive(Clone, Debug, PartialEq)]
pub struct JobRequest {
    pub plan: PathBuf,
    pub rank: usize,
    pub secret: PathBuf,
    pub profile: Option<PathBuf>,
}

impl JobRequest {
    /// The job `inv` names, or `None` for a single-process run. The plan,
    /// rank and secret go together; a profile needs a job to measure.
    pub fn from_invocation(inv: &Invocation) -> Result<Option<JobRequest>, String> {
        let profile = inv.get_str("profile_collectives").map(PathBuf::from);
        match (inv.get_str("job_plan"), inv.get_i64("job_rank"), inv.get_str("job_secret")) {
            (None, None, None) if profile.is_some() => Err("profile_collectives needs a distributed run: give job_plan, job_rank and job_secret, the collective it would measure".into()),
            (None, None, None) => Ok(None),
            (Some(plan), Some(rank), Some(secret)) => Ok(Some(JobRequest { plan: plan.into(), rank: rank.max(0) as usize, secret: secret.into(), profile })),
            _ => Err("job_plan, job_rank and job_secret go together: the plan every rank shares, this rank's place in it, and the secret that proves it belongs".into()),
        }
    }

    /// The seed every rank trains with when none was given: the digest of
    /// the plan file, which every rank holds byte for byte, so all ranks
    /// start from the same weights and batch order.
    pub fn plan_seed(&self) -> Result<u64, String> {
        let bytes = std::fs::read(&self.plan).map_err(|e| format!("{}: {e}", self.plan.display()))?;
        Ok(plan_seed(&bytes))
    }

    /// Join the job and run `train` as this rank, with the group to reduce
    /// gradients over; when a profile was asked for, the group is timed and
    /// the profile written once the run returns.
    pub fn run<R>(&self, train: impl FnOnce(GradSync<'_>) -> Result<R, String>) -> Result<R, String> {
        let context = JobContext::from_plan_file(&self.plan, self.rank, &self.secret)?;
        let collective = context.collective().map_err(|e| e.to_string())?;
        let timed = self.profile.as_ref().map(|_| TimedCollective::new(&collective));
        let group: &dyn crate::Collective = match &timed {
            Some(timed) => timed,
            None => &collective,
        };
        let identity = context.plan().identity();
        let sync = GradSync { collective: group, rank: context.rank(), checkpoint: Some(CheckpointIdentity { job: context.plan().job.as_str(), plan: &identity }) };
        let started = std::time::Instant::now();
        let result = train(sync);
        if let (Some(base), Some(timed)) = (&self.profile, &timed) {
            write_profile(base, &context, started.elapsed(), timed)?;
        }
        result
    }
}

fn write_profile(base: &Path, context: &JobContext, run: std::time::Duration, timed: &TimedCollective<'_>) -> Result<(), String> {
    let document = CollectiveProfile::new(context.rank(), context.plan().world_size, context.plan().identity(), run, timed.snapshot());
    document.write(base).map(|_| ()).map_err(|e| format!("writing the collective profile beside {}: {e}", base.display()))
}

/// The first eight bytes of the SHA-256 of `plan`, as a seed.
fn plan_seed(plan: &[u8]) -> u64 {
    use sha2::{Digest as _, Sha256};
    let digest = Sha256::digest(plan);
    u64::from_le_bytes(digest[..8].try_into().expect("a SHA-256 digest holds eight bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_job_is_all_three_or_none_and_a_profile_needs_one() {
        assert_eq!(JobRequest::from_invocation(&Invocation::new()), Ok(None));
        let job = JobRequest::from_invocation(&Invocation::new().set("job_plan", json!("p.json")).set("job_rank", json!(1)).set("job_secret", json!("s"))).unwrap().unwrap();
        assert_eq!((job.plan, job.rank, job.secret, job.profile), ("p.json".into(), 1, "s".into(), None));
        assert!(JobRequest::from_invocation(&Invocation::new().set("job_plan", json!("p.json"))).unwrap_err().contains("go together"));
        assert!(JobRequest::from_invocation(&Invocation::new().set("profile_collectives", json!("x"))).unwrap_err().contains("distributed run"));
    }

    /// The same plan gives every rank the same seed, a different plan a
    /// different one.
    #[test]
    fn the_plan_yields_one_seed_and_a_different_plan_a_different_one() {
        assert_eq!(plan_seed(br#"{"job":"a"}"#), plan_seed(br#"{"job":"a"}"#));
        assert_ne!(plan_seed(br#"{"job":"a"}"#), plan_seed(br#"{"job":"b"}"#));
    }
}
