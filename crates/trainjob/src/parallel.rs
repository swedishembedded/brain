// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that spread one served
// fine-tune over the cards of a server and the servers of a fleet, for its
// clients. If your team needs expertise in data-parallel and sharded training
// as a service, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How a served run is spread over cards and servers, one implementation for every action that trains as ranks
//! (`train`'s supervised and policy-gradient objectives, `full_train`):
//!
//! * the request fields `cards` and `job` ([`cards_param`], [`job_param`]) read into a [`Spread`], with the
//!   `job` document checked ([`JobRef`]) and what a spread run needs (a `seed`, no `resume_key` across servers);
//! * the check against the server ([`Spread::admit`]): no more cards than it has, and a `job` only with the
//!   secret it holds;
//! * the ranks themselves ([`train_as_ranks`]): this server's ranks of the run, one per claimed card, made in this
//!   process ([`brain::DataParallel::in_process`]) or joined to a job spread over several servers
//!   ([`brain::DataParallel::job_ranks`]), each on its own thread and card ([`run_ranks`]);
//! * the answer of a server whose ranks do not include rank 0 ([`Spread::answer_for_ranks`]): rank 0's server
//!   publishes the run.

use std::sync::mpsc::Sender;
use std::sync::Mutex;

use capability::{ParamSpec, ParamType, Progress};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::job::{trainer_failure, JobAccess, JobEnv};

/// The most cards one run trains on at once.
pub const MAX_CARDS: u32 = 64;
/// The longest `job` document accepted, in bytes.
pub const MAX_JOB_DOCUMENT_BYTES: usize = 4096;
/// The most ranks a job spread over several servers has.
pub const MAX_JOB_WORLD: usize = 4096;
/// The longest a job may wait for its ranks to join, or one of its collectives for a peer: a day.
pub const MAX_JOB_TIMEOUT_MS: u64 = 24 * 3600 * 1000;
/// The longest job name and coordinator address accepted.
const MAX_JOB_NAME_LEN: usize = 128;
const MAX_COORDINATOR_LEN: usize = 255;

/// `cards`: how many of the server's cards a run trains on; `holds` says what each rank holds, `objectives` which
/// of the action's objectives take more than one.
pub fn cards_param(holds: &str, objectives: &str) -> ParamSpec {
    ParamSpec::new("cards", ParamType::Int, &format!("how many of the server's cards the run trains on at once: one data-parallel rank per card, each holding {holds}, training its own share of the records with the same seed, with every step's gradients reduced over the cards ({objectives}; at most the cards the server has)"))
        .default(json!(1))
        .min(1.0)
        .max(MAX_CARDS as f64)
        .step(1.0)
}

/// `job`: this run as the server's part of one data-parallel job spread over several servers; `plan_sharding`
/// says what the plan may shard, `needs` what else the request must hold.
pub fn job_param(plan_sharding: &str, needs: &str) -> ParamSpec {
    ParamSpec::new("job", ParamType::Str, &format!("JSON document that makes this run part of one data-parallel job spread over several servers: {{\"plan\": PLAN, \"first_rank\": N, \"secret_sha256\": \"<hex>\"}} where PLAN is the job plan every server is given (job, world_size, tp 1, pp 1, dp = world_size, coordinator = the address the server holding rank 0 was started to bind as a job coordinator, handshake_timeout_ms, op_timeout_ms, optional collectives and bucket_mib; {plan_sharding}), this server runs ranks first_rank .. first_rank + cards - 1, and secret_sha256 is the SHA-256 of the job secret this server was started with, never the secret; needs seed, {needs}no resume_key"))
}

/// This server's part of a job spread over several servers, as the `job` document names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobRef {
    /// The plan every server of the job is given.
    pub plan: brain::JobPlan,
    /// This server's first rank; it runs `cards` ranks from it.
    pub first_rank: usize,
    /// The SHA-256 of the job secret, which this server must hold.
    pub secret_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JobDoc {
    plan: brain::JobPlan,
    first_rank: usize,
    secret_sha256: String,
}

impl JobRef {
    /// Parses and checks `action`'s `job` document against the run's `cards`: a bounded document with no unknown
    /// field, a plan that holds and is plain data parallelism (replicated unless `plan_may_shard`) with bounded
    /// timeouts, a rank range inside it, and a well-formed secret digest.
    pub fn parse(action: &str, text: &str, cards: u32, plan_may_shard: bool) -> Result<JobRef, String> {
        let refused = |why: String| Err(format!("{action}: job: {why}"));
        if text.len() > MAX_JOB_DOCUMENT_BYTES {
            return refused(format!("the document is {} bytes; at most {MAX_JOB_DOCUMENT_BYTES} are accepted", text.len()));
        }
        let doc: JobDoc = serde_json::from_str(text).map_err(|e| format!("{action}: job: {e}"))?;
        let plan = doc.plan;
        plan.validate().map_err(|why| format!("{action}: job: {why}"))?;
        if plan.job.len() > MAX_JOB_NAME_LEN || !plan.job.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c)) {
            return refused(format!("plan.job must be 1 to {MAX_JOB_NAME_LEN} characters of letters, digits and . _ -"));
        }
        if plan.coordinator.len() > MAX_COORDINATOR_LEN {
            return refused(format!("plan.coordinator is longer than {MAX_COORDINATOR_LEN} characters"));
        }
        if plan.world_size > MAX_JOB_WORLD {
            return refused(format!("plan.world_size {} is more than {MAX_JOB_WORLD} ranks", plan.world_size));
        }
        if plan.tp != 1 || plan.pp != 1 {
            return refused("a served job trains data-parallel only: tp and pp must be 1".to_string());
        }
        if !plan_may_shard && !plan.sharding.is_replicated() {
            return refused("this job trains replicas: plan.sharding must be replicated".to_string());
        }
        if plan.handshake_timeout_ms > MAX_JOB_TIMEOUT_MS || plan.op_timeout_ms > MAX_JOB_TIMEOUT_MS {
            return refused(format!("a timeout above {MAX_JOB_TIMEOUT_MS} ms"));
        }
        let end = doc.first_rank.checked_add(cards as usize).filter(|&end| end <= plan.world_size);
        if end.is_none() {
            return refused(format!("ranks {} .. {} (first_rank and cards {cards}) are outside the plan's {} ranks", doc.first_rank, doc.first_rank.saturating_add(cards as usize), plan.world_size));
        }
        if doc.secret_sha256.len() != 64 || !doc.secret_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return refused("secret_sha256 is the SHA-256 of the job secret, 64 hexadecimal digits".to_string());
        }
        Ok(JobRef { plan, first_rank: doc.first_rank, secret_sha256: doc.secret_sha256 })
    }

    /// The ranks this server runs.
    pub fn ranks(&self, cards: u32) -> std::ops::Range<usize> {
        self.first_rank..self.first_rank + cards as usize
    }
}

/// How one run is spread: over `cards` of this server, and, with a `job`, as this server's part of a job over
/// several servers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spread {
    pub cards: u32,
    pub job: Option<JobRef>,
}

impl Spread {
    /// Reads `action`'s `cards` and `job` fields: the job document checked ([`JobRef::parse`]), and a job holding
    /// a `seed` (every server's ranks must draw the same initialisation and batch order) and no `resume_key` (a job
    /// spread over several servers restarts whole).
    pub fn parse(action: &str, cards: u32, job: Option<&str>, seed: Option<u64>, resume_key: Option<&str>, plan_may_shard: bool) -> Result<Spread, String> {
        let job = job.map(|text| JobRef::parse(action, text, cards, plan_may_shard)).transpose()?;
        if job.is_some() {
            if seed.is_none() {
                return Err(format!("{action}: job needs seed: every server's ranks must draw the same initialisation and batch order"));
            }
            if resume_key.is_some() {
                return Err(format!("{action}: job does not resume (resume_key): a job spread over several servers restarts whole"));
            }
        }
        Ok(Spread { cards, job })
    }

    /// Whether the run trains as ranks: on several cards, or as part of a job.
    pub fn ranked(&self) -> bool {
        self.cards > 1 || self.job.is_some()
    }

    /// The ranks of the whole group, every server's.
    pub fn world(&self) -> usize {
        self.job.as_ref().map_or(self.cards as usize, |job| job.plan.world_size)
    }

    /// Whether a server with `devices` cards that took part in jobs as `access` says can run this: no more cards
    /// than it has, and for a job, the secret the job names.
    pub fn admit(&self, action: &str, devices: usize, access: Option<&JobAccess>) -> Result<(), String> {
        if self.cards as usize > devices {
            return Err(format!("{action}: cards {}: this server trains on at most {devices} card(s) at once", self.cards));
        }
        match &self.job {
            Some(job) => JobAccess::admit(action, access, job),
            None => Ok(()),
        }
    }

    /// What a server whose ranks do not include the job's rank 0 answers for `objective`: its ranks, the run being
    /// rank 0's server's to publish; `None` for the server that holds rank 0, or a run that is no job.
    pub fn answer_for_ranks(&self, objective: &str) -> Option<Value> {
        let job = self.job.as_ref().filter(|job| job.first_rank != 0)?;
        let last = job.first_rank + self.cards as usize - 1;
        Some(json!({"status": "completed", "objective": objective, "job": job.plan.job, "ranks": [job.first_rank, last]}))
    }
}

/// Runs `body(i, rank, steps)` as this server's ranks of `spread`, rank `i` on `devices[i]` (one per claimed card),
/// every rank sharding `sharding`, and returns every rank's result in order. Without a job the ranks are made in this
/// process; with one, each joins the job on its own thread. A rank's group is broken before a failed rank lets it go
/// ([`brain::RankGuard`]); [`run_ranks`] says what a failing rank does to the others. The first rank's `body` is
/// handed the sender of the steps it reports.
pub fn train_as_ranks<T: Send>(
    spread: &Spread,
    action: &str,
    env: &JobEnv,
    devices: &[residency::Device],
    sharding: brain::Sharding,
    body: impl Fn(usize, brain::DataParallel, Option<Sender<Progress>>) -> Result<T, String> + Sync,
    progress: &mut dyn FnMut(Progress),
) -> Result<Vec<T>, String> {
    let guarded = |i: usize, rank: brain::DataParallel, frames| {
        let guard = rank.guard();
        let result = rank.sharded(sharding).map_err(trainer_failure).and_then(|rank| body(i, rank, frames));
        guard.finish(&result);
        result
    };
    match &spread.job {
        Some(job) => {
            let access = env.job_access.as_ref();
            JobAccess::admit(action, access, job)?;
            let secret = access.map(|a| a.secret.clone()).ok_or_else(|| format!("{action}: job: this server holds no job secret"))?;
            let ranks = brain::DataParallel::job_ranks(job.plan.clone(), secret, job.ranks(spread.cards)).map_err(trainer_failure)?;
            let first = job.first_rank;
            let body = |i: usize, frames| guarded(i, ranks.join(first + i).map_err(trainer_failure)?, frames);
            run_ranks(devices, &|i, why| ranks.abort(first + i, why), &|| ranks.first_failed().map(|r| r - first), body, progress)
        }
        None => {
            let ranks = brain::DataParallel::in_process(devices.len()).map_err(trainer_failure)?;
            let breakers: Vec<brain::GroupBreaker> = ranks.iter().map(brain::DataParallel::breaker).collect();
            // Each rank moves to the thread that trains it and ends there: ending a device group is collective, so
            // the ranks must end theirs at once, not one after another here.
            let slots: Vec<Mutex<Option<brain::DataParallel>>> = ranks.into_iter().map(|rank| Mutex::new(Some(rank))).collect();
            let body = |i: usize, frames| {
                let rank = slots[i].lock().unwrap_or_else(|e| e.into_inner()).take().ok_or_else(|| format!("{action}: a rank ran twice"))?;
                guarded(i, rank, frames)
            };
            run_ranks(devices, &|i, why| breakers[i].abort(why), &|| breakers.first().and_then(brain::GroupBreaker::first_failed), body, progress)
        }
    }
}

/// Runs `body(i, ..)` for this process's rank `i` of a group, one per card of `devices`, each on its own thread
/// scoped to `devices[i]`, and returns every rank's result in order once all are done. Rank `0`'s `body` is handed
/// the sender of the steps it reports, which reach `progress` on this thread as they arrive.
///
/// A rank that fails (an error, a panic or a cancellation) breaks the group at once with `abort(i, why)`
/// ([`brain::DataParallel::abort`]: the host transport and the device group on every card), so the others fail in
/// their next or current reduction instead of waiting for it until the group's op timeout; the run then fails with
/// the reason of the rank that failed first (`first_failed`, the rank that broke the group, else the first result
/// in), not the "aborted" its failure caused in the others.
pub fn run_ranks<T: Send>(
    devices: &[residency::Device],
    abort: &(dyn Fn(usize, &str) + Sync),
    first_failed: &(dyn Fn() -> Option<usize> + Sync),
    body: impl Fn(usize, Option<Sender<Progress>>) -> Result<T, String> + Sync,
    progress: &mut dyn FnMut(Progress),
) -> Result<Vec<T>, String> {
    let first_failure: Mutex<Option<String>> = Mutex::new(None);
    let (frames, steps) = std::sync::mpsc::channel::<Progress>();
    let results: Vec<Result<T, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = devices
            .iter()
            .copied()
            .enumerate()
            .map(|(i, device)| {
                let frames = (i == 0).then(|| frames.clone());
                let (body, first_failure) = (&body, &first_failure);
                scope.spawn(move || {
                    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| residency::on_device(device, || body(i, frames))));
                    let result = match run {
                        Ok(placed) => placed.and_then(|done| done),
                        Err(panic) => {
                            let why = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default();
                            eprintln!("brain train: a rank on {device:?} panicked: {why}");
                            Err("train: the device failed during the run (see the server log)".to_string())
                        }
                    };
                    if let Err(why) = &result {
                        // Recorded before the group is broken, so no failure the break causes can come first.
                        first_failure.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(|| why.clone());
                        abort(i, why);
                    }
                    result
                })
            })
            .collect();
        drop(frames);
        // Rank 0's steps reach the caller on this thread, as they arrive; the channel closes when rank 0 ends.
        for frame in steps.iter() {
            progress(frame);
        }
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("train: a rank's thread ended abnormally".to_string()))).collect()
    });
    // The rank that broke the group failed first; the others' failures are what the break did to them. A rank can
    // break it (its guard, `brain::RankGuard`) before its own result is in.
    if let Some(Err(why)) = first_failed().and_then(|i| results.get(i)) {
        return Err(why.clone());
    }
    if let Some(why) = first_failure.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(why);
    }
    results.into_iter().collect()
}
