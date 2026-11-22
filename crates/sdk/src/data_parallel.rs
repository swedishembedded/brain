// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data-parallel fine-tuning across the cards of
// one server and across machines for its clients. If your team needs expertise
// in scaling training out without changing the training code, you can procure
// our services by sending an email to info@swedishembedded.com.

//! One rank of a data-parallel fine-tune, for any trainer the SDK runs.
//!
//! A [`DataParallel`] is handed to a trainer ([`crate::ChatFineTune::data_parallel`]); every rank runs the same
//! trainer with the same settings and seed, each on its own card, trains on its own shard of the records, and every
//! step's gradients are reduced over the group before the optimizer step - on the cards over NCCL when every rank
//! can, through the host otherwise, saying why. The ranks are either processes a launcher started from one job plan
//! ([`DataParallel::join_job`]) or threads of one process, one per card ([`DataParallel::in_process`]).

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use model::{Collective, CollectivePath, HostCollective, JobCollective, JobContext};

use crate::{Error, Result};

/// How long in-process ranks wait for each other to form the device group, and for one reduction.
const IN_PROCESS_JOIN: Duration = Duration::from_secs(120);
const IN_PROCESS_OP: Duration = Duration::from_secs(600);

/// One rank of a data-parallel group. See the module documentation.
#[derive(Clone)]
pub struct DataParallel {
    group: Arc<dyn Collective>,
    rank: usize,
    /// The job and plan identity a checkpoint commit is filed under, for a job a launcher started.
    job: Option<(String, String)>,
    /// What the group shards across its ranks: the plan's for a launched job, [`Self::sharded`]'s otherwise.
    sharding: model::zero::Sharding,
    /// What the ranks of an in-process group share ([`Self::in_process`]).
    local: Option<Arc<LocalGroup>>,
}

/// What the ranks that are threads of this process share: each one's collective (so a rank that fails can break the
/// group on the host and on every card at once) and the step they agreed to stop after.
struct LocalGroup {
    /// The members that joined, and the reason the group was broken, once it was: a member that joins after that is
    /// broken as it joins.
    members: Mutex<Members>,
    stop_after: Mutex<Option<u32>>,
}

/// The members of a [`LocalGroup`] that joined, and why the group was broken, once it was.
///
/// Held weakly: each rank's collective ends with the rank, on its own thread. Ending a device group (NCCL's
/// finalize) is itself collective, so the ranks of one process must end theirs at once; a group that outlived its
/// ranks here would end them one after another on one thread, and the first would wait forever for the others.
#[derive(Default)]
struct Members {
    joined: Vec<std::sync::Weak<dyn Collective>>,
    broken: Option<String>,
    /// The rank whose failure broke the group, when a rank did.
    broken_by: Option<usize>,
}

impl LocalGroup {
    fn new(members: &[Arc<dyn Collective>]) -> Arc<LocalGroup> {
        Arc::new(LocalGroup { members: Mutex::new(Members { joined: members.iter().map(Arc::downgrade).collect(), broken: None, broken_by: None }), stop_after: Mutex::new(None) })
    }

    /// Break the group because `rank` failed (`why` names it).
    fn abort(&self, rank: usize, why: &str) {
        let why = format!("rank {rank} failed: {why}");
        let why = why.as_str();
        let mut members = self.members.lock().unwrap_or_else(|e| e.into_inner());
        if members.broken.is_none() {
            (members.broken, members.broken_by) = (Some(why.to_string()), Some(rank));
        }
        for member in members.joined.iter().filter_map(std::sync::Weak::upgrade) {
            member.abort(why);
        }
    }

    fn first_failed(&self) -> Option<usize> {
        self.members.lock().unwrap_or_else(|e| e.into_inner()).broken_by
    }

    /// Count `member` in; `Err` with the reason when the group was broken before it joined (it is then broken too).
    fn admit(&self, member: &Arc<dyn Collective>) -> Result<()> {
        let mut members = self.members.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(why) = &members.broken {
            member.abort(why);
            return Err(Error::Backend(format!("joining the job: the group was aborted: {why}")));
        }
        members.joined.push(Arc::downgrade(member));
        Ok(())
    }
}

/// The ranks of a distributed job that run as threads of this process: a contiguous range of the plan's ranks, each
/// on its own card ([`DataParallel::job_ranks`]). Each one joins the job on its own thread ([`JobRanks::join`]); they
/// share a cancellation and a failure as the ranks of [`DataParallel::in_process`] do, and reach the job's other
/// ranks, in other processes or on other machines, over the job's sealed transport.
pub struct JobRanks {
    plan: crate::JobPlan,
    secret: crate::JobSecret,
    ranks: std::ops::Range<usize>,
    local: Arc<LocalGroup>,
}

impl std::fmt::Debug for JobRanks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobRanks").field("job", &self.plan.job).field("world", &self.plan.world_size).field("ranks", &self.ranks).finish()
    }
}

impl JobRanks {
    /// The ranks this process runs, in order.
    pub fn ranks(&self) -> std::ops::Range<usize> {
        self.ranks.clone()
    }

    /// Join the job as `rank` (one of [`Self::ranks`]): blocks until every rank of the job joined, within the plan's
    /// handshake timeout. The rank that is the plan's rank 0 binds the coordinator address. Call it on the thread
    /// that trains the rank, every rank of the range at once.
    pub fn join(&self, rank: usize) -> Result<DataParallel> {
        if !self.ranks.contains(&rank) {
            return Err(Error::MissingArgument(format!("rank {rank} is not one of this process's ranks {:?} of job {:?}", self.ranks, self.plan.job)));
        }
        let context = JobContext::new(self.plan.clone(), self.secret.clone(), rank).map_err(Error::MissingArgument)?;
        let group: Arc<dyn Collective> = Arc::new(context.job_collective().map_err(|e| Error::Backend(format!("joining the job: {e}")))?);
        self.local.admit(&group)?;
        Ok(DataParallel { group, rank, job: Some((self.plan.job.clone(), self.plan.identity())), sharding: self.plan.sharding, local: Some(Arc::clone(&self.local)) })
    }

    /// Break the ranks of this process, joined or still joining, because `rank` failed (before it could join, say):
    /// the ones that joined fail at once, and so does a later join.
    pub fn abort(&self, rank: usize, why: &str) {
        self.local.abort(rank, why);
    }

    /// The rank whose failure broke this process's ranks first, if one did: the cause, where the others' failures
    /// are its consequences.
    pub fn first_failed(&self) -> Option<usize> {
        self.local.first_failed()
    }
}

/// Breaks the in-process group of one rank, as [`DataParallel::abort`] does ([`DataParallel::breaker`]).
#[derive(Clone)]
pub struct GroupBreaker {
    local: Option<Arc<LocalGroup>>,
    rank: usize,
}

impl std::fmt::Debug for GroupBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupBreaker").field("rank", &self.rank).finish()
    }
}

impl GroupBreaker {
    /// [`DataParallel::abort`] for the rank this breaker was made from.
    pub fn abort(&self, why: &str) {
        if let Some(local) = &self.local {
            local.abort(self.rank, why);
        }
    }

    /// The rank whose failure broke the group first, if one did ([`JobRanks::first_failed`]).
    pub fn first_failed(&self) -> Option<usize> {
        self.local.as_ref().and_then(|local| local.first_failed())
    }
}

/// Holds a rank's group until the rank's outcome is known, and breaks the group first when the rank failed
/// ([`DataParallel::guard`]).
///
/// A device group ends collectively: a rank's communicator, finalized while its peers wait for it in a reduction,
/// waits for them until its deadline. So a rank that fails must abort the group before its last reference to it goes,
/// and the trainer that holds the rank drops it on the way out of a failure, an error or a panic alike. The guard is
/// that last reference: taken before the rank is handed to the trainer, finished with the rank's result, and on a
/// drop without one (a panic unwinding) it aborts the group.
#[must_use = "a guard dropped at once holds nothing"]
pub struct RankGuard {
    rank: DataParallel,
    finished: bool,
}

impl std::fmt::Debug for RankGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankGuard").field("rank", &self.rank.rank).field("finished", &self.finished).finish()
    }
}

impl RankGuard {
    /// The rank's outcome: an error breaks the group with its message ([`DataParallel::abort`]) before the guard lets
    /// the group go.
    pub fn finish<T, E: std::fmt::Display>(mut self, result: &std::result::Result<T, E>) {
        if let Err(e) = result {
            self.rank.abort(&e.to_string());
        }
        self.finished = true;
    }
}

impl Drop for RankGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.rank.abort("the rank stopped without an outcome (it panicked)");
        }
    }
}

impl std::fmt::Debug for DataParallel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataParallel").field("rank", &self.rank).field("world", &self.group.world_size()).field("job", &self.job.as_ref().map(|(job, _)| job)).field("sharding", &self.sharding).finish()
    }
}

impl DataParallel {
    /// Join the distributed job a launcher started: the plan file every rank shares, this process's rank, and the
    /// owner-only secret file that proves it belongs. Blocks until every rank has joined, within the plan's handshake
    /// timeout.
    pub fn join_job(plan: &Path, rank: usize, secret: &Path) -> Result<DataParallel> {
        let context = JobContext::from_plan_file(plan, rank, secret).map_err(Error::MissingArgument)?;
        let group = context.job_collective().map_err(|e| Error::Backend(format!("joining the job: {e}")))?;
        let job = Some((context.plan().job.clone(), context.plan().identity()));
        Ok(DataParallel { group: Arc::new(group), rank, job, sharding: context.plan().sharding, local: None })
    }

    /// The ranks `ranks` of the job `plan`, run as threads of this process, one per card, beside the job's other ranks
    /// in other processes or on other machines (each started with the same plan and secret and its own range): what a
    /// server runs of a job an orchestrator spread over several servers. The ranks join one by one
    /// ([`JobRanks::join`]). Refused when the range is empty or outside the plan, or the plan does not hold.
    pub fn job_ranks(plan: crate::JobPlan, secret: crate::JobSecret, ranks: std::ops::Range<usize>) -> Result<JobRanks> {
        plan.validate().map_err(Error::InvalidRequest)?;
        if ranks.is_empty() || ranks.end > plan.world_size {
            return Err(Error::InvalidRequest(format!("ranks {ranks:?} are not a range of job {:?}'s {} ranks", plan.job, plan.world_size)));
        }
        Ok(JobRanks { plan, secret, ranks, local: LocalGroup::new(&[]) })
    }

    /// `world` ranks that are threads of this process, one per card: hand rank `r` to the trainer that runs on card
    /// `r`. Gradients are reduced on the cards when every one of them can, through host memory otherwise.
    pub fn in_process(world: usize) -> Result<Vec<DataParallel>> {
        if world == 0 {
            return Err(Error::MissingArgument("a data-parallel group needs at least one rank".into()));
        }
        let host = HostCollective::new(world);
        let members: Vec<Arc<dyn Collective>> = (0..world)
            .map(|_| Arc::new(JobCollective::new(Arc::clone(&host), CollectivePath::Auto, IN_PROCESS_JOIN, IN_PROCESS_OP).with_bucket_bytes(u64::from(model::job::DEFAULT_BUCKET_MIB) << 20)) as Arc<dyn Collective>)
            .collect();
        let local = LocalGroup::new(&members);
        Ok(members.into_iter().enumerate().map(|(rank, group)| DataParallel { group, rank, job: None, sharding: model::zero::Sharding::Replicated, local: Some(Arc::clone(&local)) }).collect())
    }

    /// Break an in-process group because this rank failed and will not take part again: every other rank waiting in
    /// a reduction, through host memory or on the cards (the device group is aborted, which ends its operations on
    /// every card), and every later one, fails at once with `why` instead of waiting for it. Nothing for a rank of a
    /// launched job, whose launcher sees the process end.
    pub fn abort(&self, why: &str) {
        self.breaker().abort(why);
    }

    /// A [`RankGuard`] over this rank: take it before handing the rank to its trainer, finish it with the result.
    pub fn guard(&self) -> RankGuard {
        RankGuard { rank: self.clone(), finished: false }
    }

    /// What breaks this rank's in-process group ([`Self::abort`]) without holding the rank: a runner keeps it while
    /// the rank itself moves to the thread that trains it and ends there, as its collective must.
    pub fn breaker(&self) -> GroupBreaker {
        GroupBreaker { local: self.local.clone(), rank: self.rank }
    }

    /// Whether this rank goes on after completing step `completed`, given whether its run was `cancelled`. The ranks
    /// of a group must stop after the same step, or the ones that went on wait in the next reduction for the ones that
    /// stopped. In-process ranks share a cancellation that each sees at its own moment, so the first rank to see it
    /// names a step every rank still reaches: the steps run in lockstep (a step completes only once every rank has
    /// entered its reduction), so no rank has completed more than `completed + 1`, and every rank stops after that
    /// one. A rank of a launched job stops when it is cancelled, as before.
    pub(crate) fn keep_going(&self, completed: u32, cancelled: bool) -> bool {
        let Some(local) = &self.local else { return !cancelled };
        let mut stop_after = local.stop_after.lock().unwrap_or_else(|e| e.into_inner());
        if stop_after.is_none() && cancelled {
            *stop_after = Some(completed + 1);
        }
        stop_after.is_none_or(|last| completed < last)
    }

    /// Shard `sharding` across the group ([`model::zero`]): every rank of the group must ask for the same. A job a
    /// launcher started shards as its plan says; naming another stage for one of its ranks is refused.
    pub fn sharded(mut self, sharding: model::zero::Sharding) -> Result<DataParallel> {
        if self.job.is_some() && sharding != self.sharding {
            return Err(Error::MissingArgument(format!("this rank's job plan shards {:?}; a rank cannot shard {sharding:?} on its own", self.sharding)));
        }
        self.sharding = sharding;
        Ok(self)
    }

    /// The group as a parameter-sharded model holds it ([`model::zero::ShardGroup`]).
    pub(crate) fn shard_group(&self) -> model::zero::ShardGroup {
        model::zero::ShardGroup { collective: std::sync::Arc::clone(&self.group), rank: self.rank }
    }

    /// What the group shards across its ranks.
    pub fn sharding(&self) -> model::zero::Sharding {
        self.sharding
    }

    /// This rank's place in the group.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// How many ranks the group has.
    pub fn world(&self) -> usize {
        self.group.world_size()
    }

    /// Rank 0's `words` on every rank of the group (what the other ranks pass is ignored): how a result one rank
    /// computed for all of them is shared. Every rank calls it at the same point.
    pub(crate) fn share_from_root(&self, words: &[u64]) -> Result<Vec<u64>> {
        model::broadcast_words(&*self.group, self.rank, words, 0).map_err(|e| Error::Backend(format!("sharing rank 0's result: {e}")))
    }

    /// The gradient group a trainer's fit loop reduces over.
    pub(crate) fn grad_sync(&self) -> model::GradSync<'_> {
        let checkpoint = self.job.as_ref().map(|(job, plan)| model::CheckpointIdentity { job, plan });
        model::GradSync { collective: &*self.group, rank: self.rank, checkpoint, observer: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ending a device group is collective, so each rank's must end with the rank, on its own thread, never with a
    /// group that outlives the ranks and ends them one after another.
    #[test]
    fn a_ranks_collective_ends_with_the_rank_not_with_its_peers() {
        let ranks = DataParallel::in_process(2).unwrap();
        let breakers: Vec<GroupBreaker> = ranks.iter().map(DataParallel::breaker).collect();
        let groups: Vec<std::sync::Weak<dyn Collective>> = ranks.iter().map(|r| Arc::downgrade(&r.group)).collect();
        let mut ranks = ranks.into_iter();
        drop(ranks.next());
        assert!(groups[0].upgrade().is_none(), "rank 0's collective ended with rank 0 although rank 1 and the breakers live");
        assert!(groups[1].upgrade().is_some());
        breakers[0].abort("gone");
        drop(ranks);
        assert!(groups[1].upgrade().is_none());
    }
}
