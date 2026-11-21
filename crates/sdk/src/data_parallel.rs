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
use std::sync::Arc;
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
        Ok(DataParallel { group: Arc::new(group), rank, job, sharding: context.plan().sharding })
    }

    /// `world` ranks that are threads of this process, one per card: hand rank `r` to the trainer that runs on card
    /// `r`. Gradients are reduced on the cards when every one of them can, through host memory otherwise.
    pub fn in_process(world: usize) -> Result<Vec<DataParallel>> {
        if world == 0 {
            return Err(Error::MissingArgument("a data-parallel group needs at least one rank".into()));
        }
        let host = HostCollective::new(world);
        Ok((0..world)
            .map(|rank| {
                let group = JobCollective::new(Arc::clone(&host), CollectivePath::Auto, IN_PROCESS_JOIN, IN_PROCESS_OP).with_bucket_bytes(u64::from(model::job::DEFAULT_BUCKET_MIB) << 20);
                DataParallel { group: Arc::new(group), rank, job: None, sharding: model::zero::Sharding::Replicated }
            })
            .collect())
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

    /// The gradient group a trainer's fit loop reduces over.
    pub(crate) fn grad_sync(&self) -> model::GradSync<'_> {
        let checkpoint = self.job.as_ref().map(|(job, plan)| model::CheckpointIdentity { job, plan });
        model::GradSync { collective: &*self.group, rank: self.rank, checkpoint, observer: None }
    }
}
