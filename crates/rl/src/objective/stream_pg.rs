// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training a served policy online, on the rollout groups an orchestrator
// produces against the same server while it trains, for its clients. If your team needs expertise in asynchronous
// policy-gradient training with bounded staleness and backpressure then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Policy-gradient training on rollout groups that arrive while the run goes on.
//!
//! An orchestrator samples rollouts from the served policy, grades them, and hands each complete group to the running
//! job ([`RolloutFeed::offer`]). The group's advantages are computed when it arrives, by the code the recorded-rollout
//! objective computes them with ([`super::external`]), and it waits in a queue bounded in policy tokens (the tokens a
//! step trains). [`StreamedPolicyGradient`] is the objective over that queue: before each optimiser step it waits
//! ([`model::Objective::supply`]) until [`StreamSettings::step_tokens`] policy tokens are queued, takes whole groups in
//! arrival order until the step holds that many, and trains them as one step of the shared policy-gradient batch
//! ([`super::pg_batch`]): the gradient of a step is the sum over its policy tokens divided by the step's own count of
//! them (`token-sum`) or by its sequences times a constant (`constant-length`), however many micro-batches hold them.
//!
//! What a step trains is a function of the records and their order alone: the same records in the same order give the
//! same steps however many offers carried them.
//!
//! * **Staleness.** A record's `policy_version` is the oldest served version among the requests that produced it; a
//!   group whose oldest record is more than [`StreamSettings::max_staleness`] versions behind the version the job last
//!   published ([`RolloutFeed::set_version`]) is dropped when a step would take it, and counted.
//! * **Backpressure.** An offer that would take the queue past [`StreamSettings::queue_tokens`] is refused whole,
//!   naming the allowance, unless the queue holds less than one step: a client is never refused while the learner is
//!   waiting for it.
//! * **Receipts.** Every group key is accepted once per run. A step's groups are reported applied once the optimiser
//!   update holding their gradient is taken (or skipped, when a guard skipped it), and committed once a training state
//!   that holds that update is written ([`model::Objective::state_saved`]); the consumed keys ride in that state, so a
//!   resumed run refuses them as duplicates and accepts the rest again.
//!
//! The behaviour log-probabilities are the sampler's own (`behavior_logprobs` on every policy token): a stream's
//! records were drawn by served versions this run cannot replay, so the start-of-run proxy has no meaning here and is
//! refused, as are what this objective does not yet train (importance correction, teacher contexts, images, the
//! numerics gate, a per-sequence mean). The estimator is the configuration's: the clipped surrogate, or the
//! score-centred one over the sampler's top-K, whose K the stream fixes before the run ([`StreamSettings::top_k`]).

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use data::chat::EncodedSplit;
use data::chat_rollout::{SplitRollouts, NO_TOP_ID};
use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;
use model::training::{self, Capability, Unsupported};
use model::{Batch, Model, ModelConfig, Objective, StepVerdict, StepWork, Supply, IGNORE};
use serde_json::{json, Value};

use super::external::{advantages, BehaviourSource, ImportanceCorrection, PolicyGradientConfig, PolicyGradientStats, ReferencePolicy, StepAccumulator};
use super::grpo::LossAgg;
use super::grpo_stats::{NonFiniteSteps, MAX_CONSECUTIVE_NONFINITE};
use super::pg_batch::{adapters_are_zero, adapters_can_be_disabled, backward_weighted_lp, install_sampler_ids, with_adapters_disabled, AdapterState, PgBatch, SamplerTop};
use super::pg_loss::{gradient_scan, GuardVerdict, StepGuard};

/// The widest sampler top-K a stream centres over.
pub const MAX_TOP_K: usize = 256;

/// How often a learner waiting for records checks whether it was stopped.
const STOP_POLL: Duration = Duration::from_millis(100);

/// What a stream is bounded by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamSettings {
    /// Policy tokens (supervised tokens of rollouts) one optimiser step trains: a step takes whole groups until it holds
    /// at least this many.
    pub step_tokens: usize,
    /// Policy tokens the queue holds at most, at least [`Self::step_tokens`].
    pub queue_tokens: usize,
    /// How many published versions a record's `policy_version` may be behind the job's newest; `None`: unbounded.
    pub max_staleness: Option<u64>,
    /// The longest record, in tokens: the training row.
    pub block: usize,
    /// The sampler's top-K every record carries, which the score-centred estimator centres over; `None` for the
    /// clipped surrogate. Fixed before the run, since the model's head is sized to it.
    pub top_k: Option<usize>,
}

impl StreamSettings {
    pub fn validate(&self) -> Result<(), String> {
        if self.step_tokens == 0 {
            return Err("step_tokens must be at least 1".to_string());
        }
        if self.queue_tokens < self.step_tokens {
            return Err(format!("queue_tokens {} is less than step_tokens {}; the queue must hold at least one step", self.queue_tokens, self.step_tokens));
        }
        if self.block < 2 {
            return Err("the training row must hold at least two tokens".to_string());
        }
        if self.top_k.is_some_and(|k| !(1..=MAX_TOP_K).contains(&k)) {
            return Err(format!("top_k must be 1 to {MAX_TOP_K}"));
        }
        Ok(())
    }
}

/// The settings a stream cannot train with, refused before the run starts; `top_k` is the stream's sampler top-K.
pub fn check_streamable(cfg: &PolicyGradientConfig, top_k: Option<usize>) -> Result<(), String> {
    let refuse = |what: &str| Err(format!("a streamed policy gradient {what}"));
    if cfg.behaviour != BehaviourSource::Recorded {
        return refuse("takes the behaviour log-probabilities its records carry (behaviour recorded): the versions that sampled them are not the run's start");
    }
    if cfg.importance != ImportanceCorrection::None {
        return refuse("takes no importance correction yet (importance none)");
    }
    if cfg.numerics.is_some() {
        return refuse("is a training run, not a numerics measurement");
    }
    if cfg.kv_roundtrip.is_some() {
        return refuse("does not replay a KV round trip");
    }
    if cfg.loss.loss_agg == LossAgg::SeqMean {
        return refuse("normalises each step over its own tokens; name loss_agg token-sum (or constant-length): a per-sequence mean needs one sequence per micro-batch and a fixed count of them per step");
    }
    match (cfg.loss.spec().estimator.needs_sampler_top(), top_k) {
        (true, None) => return refuse("with estimator score-centred centres over the sampler's top-K: name its K (stream.top_k), which every record carries"),
        (false, Some(_)) => return refuse("reads a sampler top-K only with estimator score-centred; drop stream.top_k"),
        _ => {}
    }
    Ok(())
}

/// One example of a queued group, laid out as a training row reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamExample {
    pub ids: Vec<u32>,
    pub mask: Vec<bool>,
    /// Per token; finite on every supervised token.
    pub behavior: Vec<f32>,
    pub advantage: f32,
    pub policy_version: Option<u64>,
    /// The sampler's top-K at each token (`[tokens * k]` ids, `NO_TOP_ID` where none, and log-probabilities), for the
    /// score-centred estimator.
    pub top: Option<(Vec<u32>, Vec<f32>)>,
}

impl StreamExample {
    /// The tokens the example trains: its supervised tokens after the first.
    fn policy_tokens(&self) -> usize {
        self.mask.iter().skip(1).filter(|&&m| m).count()
    }
}

/// One complete group of rollouts: its key and its examples, each with its advantage.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamGroup {
    pub key: String,
    pub examples: Vec<StreamExample>,
}

impl StreamGroup {
    pub fn policy_tokens(&self) -> usize {
        self.examples.iter().map(StreamExample::policy_tokens).sum()
    }

    /// The oldest policy version among its records, when they name one.
    pub fn policy_version(&self) -> Option<u64> {
        self.examples.iter().filter_map(|e| e.policy_version).min()
    }

    /// Whether every advantage of the group is zero (its rewards were all equal): it trains nothing but is counted.
    pub fn zero_advantage(&self) -> bool {
        self.examples.iter().all(|e| e.advantage == 0.0)
    }
}

/// The groups of an encoded split of rollout records: each complete group with every example's advantage, computed over
/// the group by the recorded-rollout objective's own arithmetic (`cfg`'s normalisation, floor and truncation rule).
///
/// # Errors
/// The record contract's group rules, and what a stream does not train: a record without a reward (imitation), a
/// teacher context or a teacher's scores, images, a supervised token without a recorded behaviour log-probability, a
/// record longer than the row. The message names the example.
pub fn groups_of_split(split: &EncodedSplit, cfg: &PolicyGradientConfig, settings: &StreamSettings) -> Result<Vec<StreamGroup>, String> {
    let block = settings.block;
    let rollouts = SplitRollouts::of_split(split)?;
    let advantage = advantages(&rollouts, cfg.loss.advantage_norm, cfg.loss.advantage_floor, cfg.truncated)?;
    let mut groups: Vec<StreamGroup> = rollouts.groups.iter().map(|key| StreamGroup { key: key.clone(), examples: Vec::new() }).collect();
    let ends: Vec<usize> = (0..split.starts.len()).map(|e| split.starts.get(e + 1).copied().unwrap_or(split.ids.len())).collect();
    // The score-centred estimator: every policy token carries the sampler's top-K, of the stream's K, from a sampler
    // that drew from the whole row (the recorded-rollout objective's own check).
    let top = match settings.top_k {
        None => None,
        Some(k) => {
            let top = split.behavior_top.as_ref().ok_or_else(|| "estimator \"score-centred\" centres over the sampler's top-K, and the records carry none (behavior_top_ids / behavior_top_logprobs on every supervised answer)".to_string())?;
            if top.k != k {
                return Err(format!("behavior_top_ids: the records carry a top-{} and the stream centres over a top-{k} (stream.top_k)", top.k));
            }
            let eligible: Vec<usize> = (0..split.starts.len()).filter(|&e| cfg.truncated.counts(split.truncated[e])).collect();
            super::external::check_centrable(top, &rollouts.examples, &advantage, &eligible, &split.starts, &ends, &split.mask).map_err(|why| format!("record {why}"))?;
            Some(top)
        }
    };
    for (e, example) in rollouts.examples.iter().enumerate() {
        let (start, end) = (split.starts[e], ends[e]);
        let refuse = |why: &str| Err(format!("record example {e}: {why}"));
        let Some(graded) = example.and_then(|r| r.graded) else {
            return refuse("carries no \"reward\" and \"group\"; a streamed policy gradient trains rollouts only");
        };
        if split.teacher[e].is_some() || split.teacher_logprobs[start..end].iter().any(|v| !v.is_nan()) || graded.teacher_log_ratio.is_some() {
            return refuse("carries a teacher context or a teacher's scores, which a streamed policy gradient does not train yet");
        }
        if !split.media[e].is_empty() {
            return refuse("carries images, which a streamed policy gradient does not train yet");
        }
        if end - start > block {
            return refuse(&format!("is {} tokens, longer than the training row of {block} (max_block)", end - start));
        }
        if !cfg.truncated.counts(split.truncated[e]) {
            continue;
        }
        if let Some(t) = (start + 1..end).find(|&t| split.mask[t] && !split.behavior[t].is_finite()) {
            return refuse(&format!("supervised token {} has no recorded behaviour log-probability (\"behavior_logprobs\")", t - start));
        }
        groups[graded.group as usize].examples.push(StreamExample {
            ids: split.ids[start..end].to_vec(),
            mask: split.mask[start..end].to_vec(),
            behavior: split.behavior[start..end].to_vec(),
            advantage: advantage[e],
            policy_version: example.and_then(|r| r.policy_version),
            top: top.map(|t| (t.ids[start * t.k..end * t.k].to_vec(), t.logprobs[start * t.k..end * t.k].to_vec())),
        });
    }
    // A group whose every rollout was truncated and masked trains nothing; it is still a group the client sent.
    Ok(groups)
}

/// What an offer did.
#[derive(Clone, Debug, PartialEq)]
pub struct Admission {
    /// Keys newly queued, in order.
    pub accepted: Vec<String>,
    /// Keys this run already took (queued, trained or dropped): nothing was queued for them.
    pub duplicates: Vec<String>,
    /// Of the accepted, the groups whose rewards were all equal: zero advantage, counted, trained as nothing.
    pub zero_advantage: usize,
    pub queued_groups: usize,
    pub queued_tokens: usize,
    /// Policy tokens an offer may still add whole (see [`RolloutFeed::offer`]).
    pub allowance: usize,
}

/// Why an offer was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum FeedRefusal {
    /// The queue is full: nothing was queued.
    Full { queued_tokens: usize, allowance: usize, offered_tokens: usize },
    /// The run was told to finish, or stopped: it takes no more records.
    Closed,
    /// The learner is not ready yet (the run is loading, or restoring a resumed run's state): retry.
    Starting,
}

/// A group waiting in the queue, and when it arrived.
struct Queued {
    group: StreamGroup,
    tokens: usize,
    arrived: Instant,
}

struct FeedState {
    queue: VecDeque<Queued>,
    queued_tokens: usize,
    /// Every key this run took: queued, trained, dropped, and those a resumed run's state holds.
    known: HashSet<String>,
    finished: bool,
    version: u64,
    events: Vec<Value>,
    /// Keys dropped as stale and not yet handed to the learner, which counts them consumed.
    dropped: Vec<String>,
    /// The learner has restored what a resumed run already consumed: offers are answered from here on.
    open: bool,
}

/// The queue between the surface that receives a job's rollout groups and the learner that trains them. Shared between
/// the two threads; every method takes the lock briefly, and only [`StreamedPolicyGradient`]'s supply waits on it.
pub struct RolloutFeed {
    settings: StreamSettings,
    state: Mutex<FeedState>,
    wake: Condvar,
    keep_going: Box<dyn Fn() -> bool + Send + Sync>,
    /// The learner's device, lent to the server between steps (see [`RolloutFeed::new`]).
    lease: capability::lease::DeviceLease,
}

/// What a step took from the queue.
struct Taken {
    groups: Vec<Queued>,
    dropped_stale: Vec<String>,
    waited_secs: f64,
    queue_groups: usize,
    queue_tokens: usize,
}

impl RolloutFeed {
    /// A feed bounded by `settings`; the learner waiting on it stops once `keep_going` says no (a cancelled job).
    ///
    /// Between two steps the learner lends its device through `lease` while it takes the next step's groups: the
    /// records it waits for are drawn from the policy the same server serves, often on the same device, so a learner
    /// that kept the device while it waited would wait for ever; and one that has records still gives a request
    /// waiting for the device its turn (the server lets running serving finish before the device comes back).
    pub fn new(settings: StreamSettings, keep_going: impl Fn() -> bool + Send + Sync + 'static, lease: capability::lease::DeviceLease) -> Result<Arc<RolloutFeed>, String> {
        settings.validate()?;
        let state = FeedState { queue: VecDeque::new(), queued_tokens: 0, known: HashSet::new(), finished: false, version: 0, events: Vec::new(), dropped: Vec::new(), open: false };
        Ok(Arc::new(RolloutFeed { settings, state: Mutex::new(state), wake: Condvar::new(), keep_going: Box::new(keep_going), lease }))
    }

    pub fn settings(&self) -> &StreamSettings {
        &self.settings
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FeedState> {
        // Plain data changed by whole assignments: a panic elsewhere leaves it consistent.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn allowance(&self, queued: usize) -> usize {
        self.settings.queue_tokens.saturating_sub(queued)
    }

    /// Queue the groups whose keys this run has not taken. The offer is taken whole or not at all: refused when the
    /// queue already holds at least one step's policy tokens and the new groups would take it past the queue's bound.
    /// Answered once the learner is ready ([`Self::wait_open`]); an offer made before is a caller's error.
    pub fn offer(&self, groups: Vec<StreamGroup>) -> Result<Admission, FeedRefusal> {
        let mut state = self.lock();
        if state.finished {
            return Err(FeedRefusal::Closed);
        }
        if !state.open {
            return Err(FeedRefusal::Starting);
        }
        let (fresh, duplicates): (Vec<StreamGroup>, Vec<StreamGroup>) = groups.into_iter().partition(|g| !state.known.contains(&g.key));
        let offered_tokens: usize = fresh.iter().map(StreamGroup::policy_tokens).sum();
        let queued = state.queued_tokens;
        if !fresh.is_empty() && queued >= self.settings.step_tokens && queued + offered_tokens > self.settings.queue_tokens {
            return Err(FeedRefusal::Full { queued_tokens: queued, allowance: self.allowance(queued), offered_tokens });
        }
        let zero_advantage = fresh.iter().filter(|g| g.zero_advantage()).count();
        let accepted: Vec<String> = fresh.iter().map(|g| g.key.clone()).collect();
        let now = Instant::now();
        for group in fresh {
            state.known.insert(group.key.clone());
            let tokens = group.policy_tokens();
            state.queued_tokens += tokens;
            state.queue.push_back(Queued { group, tokens, arrived: now });
        }
        let admission = Admission { accepted, duplicates: duplicates.into_iter().map(|g| g.key).collect(), zero_advantage, queued_groups: state.queue.len(), queued_tokens: state.queued_tokens, allowance: self.allowance(state.queued_tokens) };
        drop(state);
        self.wake.notify_all();
        Ok(admission)
    }

    /// Whether the learner is ready for offers (see [`Self::wait_open`]).
    pub fn is_open(&self) -> bool {
        self.lock().open
    }

    /// Wait at most `timeout` for the learner to be ready: until then a resumed run does not know which keys it took
    /// before, so an offer could not be told a duplicate. True once it is.
    pub fn wait_open(&self, timeout: Duration) -> bool {
        let state = self.lock();
        let (state, _) = self.wake.wait_timeout_while(state, timeout, |s| !s.open).unwrap_or_else(|e| e.into_inner());
        state.open
    }

    /// Answer offers from now on: the learner calls it when the run is positioned; a run with nothing to restore is
    /// ready from the start.
    pub fn open(&self) {
        self.lock().open = true;
        self.wake.notify_all();
    }

    /// No more records: the learner trains what is queued and the run ends.
    pub fn finish(&self) -> Admission {
        let mut state = self.lock();
        state.finished = true;
        let admission = Admission { accepted: Vec::new(), duplicates: Vec::new(), zero_advantage: 0, queued_groups: state.queue.len(), queued_tokens: state.queued_tokens, allowance: 0 };
        drop(state);
        self.wake.notify_all();
        admission
    }

    /// The version the job published last: what a record's staleness is measured against.
    pub fn set_version(&self, version: u64) {
        self.lock().version = version;
    }

    pub fn version(&self) -> u64 {
        self.lock().version
    }

    /// What the learner reported since the last call, oldest first: `applied`, `committed`, `dropped` frames.
    pub fn drain_events(&self) -> Vec<Value> {
        std::mem::take(&mut self.lock().events)
    }

    fn push_event(&self, event: Value) {
        self.lock().events.push(event);
    }

    /// Keys a resumed run already trained: offered again, they are duplicates.
    fn restore_known(&self, keys: &[String]) {
        self.lock().known.extend(keys.iter().cloned());
    }

    /// Wait until a step's worth of policy tokens is queued (or the feed finished with some queued), and take it:
    /// whole groups in arrival order, those too stale to train dropped on the way (reported, and kept for the learner
    /// to count consumed, [`Self::take_dropped`]). A step whose every group was dropped is no step: the wait goes on.
    /// [`Supply::Finished`] when the feed finished with nothing left, [`Supply::Stopped`] when the job was stopped.
    fn take_step(&self) -> Result<Taken, Supply> {
        let began = Instant::now();
        let mut state = self.lock();
        let stale = |state: &FeedState, group: &StreamGroup| self.settings.max_staleness.is_some_and(|bound| group.policy_version().is_some_and(|v| state.version.saturating_sub(v) > bound));
        let mut dropped_stale = Vec::new();
        loop {
            if !(self.keep_going)() {
                return Err(Supply::Stopped);
            }
            // Stale groups at the head leave before they count toward a step.
            while state.queue.front().is_some_and(|head| stale(&state, &head.group)) {
                let gone = state.queue.pop_front().expect("the head is there");
                state.queued_tokens -= gone.tokens;
                dropped_stale.push(gone.group.key);
            }
            let ready = state.queued_tokens >= self.settings.step_tokens || (state.finished && !state.queue.is_empty());
            if ready {
                let (mut groups, mut tokens) = (Vec::new(), 0);
                while tokens < self.settings.step_tokens {
                    let Some(next) = state.queue.pop_front() else { break };
                    state.queued_tokens -= next.tokens;
                    if stale(&state, &next.group) {
                        dropped_stale.push(next.group.key);
                        continue;
                    }
                    tokens += next.tokens;
                    groups.push(next);
                }
                if !groups.is_empty() {
                    Self::report_dropped(&mut state, &dropped_stale);
                    return Ok(Taken { groups, dropped_stale, waited_secs: began.elapsed().as_secs_f64(), queue_groups: state.queue.len(), queue_tokens: state.queued_tokens });
                }
                continue;
            }
            if state.finished {
                Self::report_dropped(&mut state, &dropped_stale);
                return Err(Supply::Finished);
            }
            state = self.wake.wait_timeout(state, STOP_POLL).unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    fn report_dropped(state: &mut FeedState, keys: &[String]) {
        if !keys.is_empty() {
            state.events.push(json!({"type": "dropped", "reason": "stale", "groups": keys}));
            state.dropped.extend(keys.iter().cloned());
        }
    }

    /// The keys dropped as stale since the last call: consumed, never trained.
    fn take_dropped(&self) -> Vec<String> {
        std::mem::take(&mut self.lock().dropped)
    }
}

/// What the stream measured in the step being trained, beyond the shared policy-gradient record.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StreamStep {
    staleness_mean: f32,
    staleness_max: f32,
    policy_tokens: usize,
    wait_secs: f32,
    queue_groups: usize,
    queue_tokens: usize,
    time_in_queue_secs: f32,
    dropped_stale: usize,
}

/// The policy the run started from, which the reference is measured under when it names it.
enum StartPolicy {
    Base,
    Frozen(AdapterState),
}

/// Policy-gradient training over a [`RolloutFeed`]. See the module docs.
pub struct StreamedPolicyGradient {
    cfg: PolicyGradientConfig,
    feed: Arc<RolloutFeed>,
    batch_size: usize,
    start: Option<StartPolicy>,
    wants_reference: bool,
    weights_moved: bool,
    /// The groups the next micro-steps train, taken by `supply`.
    step: Vec<Queued>,
    step_extra: StreamStep,
    /// Keys of the step just trained, until the loop says whether its update was taken.
    trained: Vec<String>,
    applied: Option<u32>,
    /// Keys trained or dropped since the last state was written, and every key the run trained before it.
    uncommitted: Vec<String>,
    committed: Vec<String>,
    stats: PolicyGradientStats,
    work: Option<StepWork>,
    verdict: Option<StepVerdict>,
    guard: StepGuard,
    consecutive_nonfinite: u32,
}

impl StreamedPolicyGradient {
    /// The objective over `feed`, training micro-batches of `batch_size` rows.
    pub fn new(feed: Arc<RolloutFeed>, cfg: PolicyGradientConfig, batch_size: usize) -> Result<StreamedPolicyGradient, String> {
        cfg.validate()?;
        check_streamable(&cfg, feed.settings.top_k)?;
        let stats = PolicyGradientStats { behaviour: cfg.behaviour, ..Default::default() };
        Ok(StreamedPolicyGradient { cfg, feed, batch_size: batch_size.max(1), start: None, wants_reference: cfg.loss.wants_reference(), weights_moved: false, step: Vec::new(), step_extra: StreamStep::default(), trained: Vec::new(), applied: None, uncommitted: Vec::new(), committed: Vec::new(), stats, work: None, verdict: None, guard: StepGuard::default(), consecutive_nonfinite: 0 })
    }

    pub fn stats(&self) -> &PolicyGradientStats {
        &self.stats
    }

    pub fn into_stats(self) -> PolicyGradientStats {
        self.stats
    }

    /// The keys trained (or dropped) since the last state was written: what a restart would lose.
    pub fn uncommitted(&self) -> &[String] {
        &self.uncommitted
    }

    /// The row's tokens and targets, behaviour and advantage per position, for `examples` laid out `batch_size` a batch.
    fn lay_out(&self, examples: &[&StreamExample], block: usize) -> Vec<Laid> {
        let cfg = BatchConfig { batch_size: self.batch_size, block_size: block, ..BatchConfig::default() };
        examples
            .chunks(self.batch_size)
            .map(|rows| {
                let mut ids = Vec::new();
                let mut mask = Vec::new();
                let mut starts = Vec::new();
                for e in rows {
                    starts.push(ids.len());
                    ids.extend_from_slice(&e.ids);
                    mask.extend_from_slice(&e.mask);
                }
                let dataset = TokenDataset::new_examples(ids, mask, &starts, 0, &cfg).expect("an offered record fits the row");
                let picks: Vec<Option<usize>> = (0..self.batch_size).map(|b| (b < rows.len()).then_some(b)).collect();
                let (tokens, y) = dataset.example_batch(&cfg, &picks).expect("the dataset indexes its examples");
                let targets: Vec<u32> = y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect();
                let (mut old_lp, mut advantage) = (vec![0.0; targets.len()], vec![f32::NAN; targets.len()]);
                let k = self.feed.settings.top_k;
                let mut top = k.map(|k| (vec![IGNORE; targets.len() * k], vec![f32::NAN; targets.len() * k]));
                for (b, e) in rows.iter().enumerate() {
                    for t in 0..block.min(e.ids.len() - 1) {
                        let at = b * block + t;
                        if targets[at] == IGNORE {
                            continue;
                        }
                        // Row position t predicts token t + 1: its behaviour log-probability and its sampler's top-K.
                        old_lp[at] = e.behavior[t + 1];
                        advantage[at] = e.advantage;
                        if let (Some(k), Some((ids, lps)), Some((own_ids, own_lps))) = (k, top.as_mut(), e.top.as_ref()) {
                            for j in 0..k {
                                let id = own_ids[(t + 1) * k + j];
                                if id != NO_TOP_ID {
                                    ids[at * k + j] = id;
                                    lps[at * k + j] = own_lps[(t + 1) * k + j];
                                }
                            }
                        }
                    }
                }
                Laid { tokens, targets, old_lp, advantage, ref_lp: None, rows: rows.len(), top }
            })
            .collect()
    }

    /// The reference policy's log-probability of every target of `laid`, one forward each, under the policy the
    /// configuration names.
    fn score_reference<M: Model>(&self, model: &M, laid: &mut [Laid]) -> Result<(), Unsupported> {
        let read = |laid: &mut [Laid]| {
            for l in laid.iter_mut() {
                model.set_batch(Batch::Lm { tokens: &l.tokens, targets: &l.targets });
                let _ = model.forward();
                l.ref_lp = Some(training::objectives(model).batch_token_logprobs());
            }
        };
        let under_base = self.cfg.reference == ReferencePolicy::Base || matches!(self.start, Some(StartPolicy::Base));
        if under_base {
            return with_adapters_disabled(model, || read(laid)).ok_or_else(|| Unsupported::of::<M>(Capability::ChatLoraFineTune));
        }
        match &self.start {
            Some(StartPolicy::Frozen(state)) if self.weights_moved => state.with_applied(model, || read(laid)),
            _ => read(laid),
        }
        Ok(())
    }

    fn rule(&mut self, finite: bool) -> StepVerdict {
        if !finite {
            self.guard.reset();
            self.consecutive_nonfinite += 1;
            self.stats.steps_skipped_nonfinite += 1;
            return if self.consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE {
                StepVerdict::Abort(std::io::Error::new(std::io::ErrorKind::InvalidData, NonFiniteSteps { consecutive: self.consecutive_nonfinite }))
            } else {
                StepVerdict::Skip(format!("non-finite loss or gradient ({} in a row)", self.consecutive_nonfinite))
            };
        }
        self.consecutive_nonfinite = 0;
        let last = self.stats.steps.last_mut().expect("the step was recorded");
        let (why, verdict) = match self.guard.judge(&self.cfg.loss, last.kl, Some(last.grad_norm)) {
            GuardVerdict::Pass => return StepVerdict::Apply,
            GuardVerdict::Skip(why) => (why.clone(), StepVerdict::Skip(why)),
            GuardVerdict::Abort { why, error } => (why, StepVerdict::Abort(std::io::Error::new(std::io::ErrorKind::InvalidData, error))),
        };
        last.skipped = Some(why);
        self.stats.steps_skipped_guard += 1;
        verdict
    }
}

/// One micro-batch of a step.
struct Laid {
    tokens: Vec<u32>,
    targets: Vec<u32>,
    old_lp: Vec<f32>,
    advantage: Vec<f32>,
    ref_lp: Option<Vec<f32>>,
    rows: usize,
    /// The sampler's top-K at each position, `[rows * block * k]`, under the score-centred estimator.
    top: Option<(Vec<u32>, Vec<f32>)>,
}

impl Laid {
    fn batch(&self, k: Option<usize>) -> PgBatch<'_> {
        let top = self.top.as_ref().zip(k).map(|((ids, logprobs), k)| SamplerTop { k, ids, logprobs });
        PgBatch { targets: &self.targets, old_lp: &self.old_lp, ref_lp: self.ref_lp.as_deref(), advantage: &self.advantage, is_weight: None, top }
    }
}

impl<M: Model> Objective<M> for StreamedPolicyGradient {
    fn regime(&self) -> &'static str {
        "policy_gradient_stream"
    }

    fn prepare(&mut self, model: &mut M) -> Result<(), Unsupported> {
        let objectives = training::require_token_objectives(model)?;
        objectives.enable_weighted_loss();
        objectives.set_logit_scale(1.0 / self.cfg.temperature);
        if let Some(k) = self.feed.settings.top_k {
            objectives.enable_sparse_targets(k as u32).map_err(|_| Unsupported::of::<M>(Capability::SparseTargets))?;
        }
        let start = if adapters_are_zero(model) { StartPolicy::Base } else { StartPolicy::Frozen(AdapterState::capture(model)) };
        let base_is_read = self.wants_reference && (matches!(start, StartPolicy::Base) || self.cfg.reference == ReferencePolicy::Base);
        if base_is_read && !adapters_can_be_disabled(model) {
            return Err(Unsupported::of::<M>(Capability::ChatLoraFineTune));
        }
        self.start = Some(start);
        Ok(())
    }

    /// The run is positioned (a resumed one has restored its consumed keys): the feed answers offers from now on.
    fn begin(&mut self, _seed: u64, _draws_done: u64) {
        self.feed.open();
    }

    fn supply(&mut self) -> Supply {
        let taken = self.feed.lease.lent(|| self.feed.take_step());
        self.uncommitted.extend(self.feed.take_dropped());
        match taken {
            Ok(taken) => {
                let now = Instant::now();
                let version = self.feed.version();
                let behind: Vec<f32> = taken.groups.iter().filter_map(|q| q.group.policy_version()).map(|v| version.saturating_sub(v) as f32).collect();
                let in_queue: f32 = taken.groups.iter().map(|q| now.duration_since(q.arrived).as_secs_f32()).sum();
                self.step_extra = StreamStep {
                    staleness_mean: if behind.is_empty() { 0.0 } else { behind.iter().sum::<f32>() / behind.len() as f32 },
                    staleness_max: behind.iter().copied().fold(0.0, f32::max),
                    policy_tokens: taken.groups.iter().map(|q| q.tokens).sum(),
                    wait_secs: taken.waited_secs as f32,
                    queue_groups: taken.queue_groups,
                    queue_tokens: taken.queue_tokens,
                    time_in_queue_secs: if taken.groups.is_empty() { 0.0 } else { in_queue / taken.groups.len() as f32 },
                    dropped_stale: taken.dropped_stale.len(),
                };
                self.step = taken.groups;
                Supply::Ready
            }
            Err(end) => end,
        }
    }

    /// The step's groups, all of them, whatever `k` the loop asks for: the step is cut by policy tokens.
    fn micro_steps(&mut self, model: &M, _rng: &mut Rng, _k: u32) -> f32 {
        let clock = Instant::now();
        let step = std::mem::take(&mut self.step);
        let block = self.feed.settings.block;
        let examples: Vec<&StreamExample> = step.iter().flat_map(|q| q.group.examples.iter()).collect();
        let mut laid = self.lay_out(&examples, block);
        if self.wants_reference && !laid.is_empty() {
            self.score_reference(model, &mut laid).expect("Objective::prepare checked the reference can be read");
        }
        let supervised: usize = laid.iter().map(|l| l.targets.iter().filter(|&&t| t != IGNORE).count()).sum();
        let mut work = StepWork { useful_tokens: examples.iter().map(|e| e.ids.len() as u64).sum(), supervised_positions: supervised as u64, prepare_secs: clock.elapsed().as_secs_f64(), ..StepWork::default() };
        // One divisor for the whole step, however many micro-batches hold it; the loop divides by nothing more.
        let rows: usize = laid.iter().map(|l| l.rows).sum();
        let denominator = match self.cfg.loss.loss_agg {
            LossAgg::ConstantLength(n) => Some(rows.max(1) as f32 * n as f32),
            agg => agg.denominator(rows, supervised, 1),
        };
        let spec = self.cfg.loss.spec();
        let mut acc = StepAccumulator::default();
        let (mut loss, mut finite) = (0.0, true);
        // A sampler id the model has no row for is a record this run cannot train: the run stops naming it.
        let vocab = model.config().vocab();
        if let Some(id) = laid.iter().filter_map(|l| l.top.as_ref()).flat_map(|(ids, _)| ids.iter().copied()).filter(|&id| id != IGNORE).find(|&id| id >= vocab) {
            self.verdict = Some(StepVerdict::Abort(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("behavior_top_ids: a record of this step names token id {id} but the model's vocabulary is {vocab}"))));
            return 0.0;
        }
        for l in &laid {
            model.set_batch(Batch::Lm { tokens: &l.tokens, targets: &l.targets });
            let batch = l.batch(self.feed.settings.top_k);
            install_sampler_ids(model, &batch, &spec).expect("prepare enabled the extra targets and every sampler id is in the vocabulary");
            let pass = Instant::now();
            let _ = model.forward();
            work.forward_secs += pass.elapsed().as_secs_f64();
            work.computed_tokens += model.computed_tokens().unwrap_or(l.tokens.len() as u64);
            let pass = Instant::now();
            let (terms, _) = backward_weighted_lp(model, &batch, &spec, denominator);
            work.backward_secs += pass.elapsed().as_secs_f64();
            finite &= terms.finite;
            loss += terms.normalised_loss(denominator);
            acc.terms.push(terms);
        }
        for e in &examples {
            acc.sequences += 1;
            acc.advantaged_sequences += 1;
            acc.advantage_sum += f64::from(e.advantage);
        }
        let (grads_finite, grad_sq) = gradient_scan(model);
        finite &= grads_finite;
        let record = acc.close(self.stats.steps.len() + 1, grad_sq.sqrt() as f32, 0.0);
        self.stats.steps.push(record);
        self.stats.sequences_with_advantage += examples.len();
        self.stats.tokens_recorded_behavior += self.step_extra.policy_tokens;
        self.trained = step.into_iter().map(|q| q.group.key).collect();
        self.verdict = Some(self.rule(finite));
        self.work = Some(work);
        loss
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        self.micro_steps(model, rng, 1)
    }

    fn step_work(&mut self) -> Option<StepWork> {
        self.work.take()
    }

    fn step_verdict(&mut self) -> StepVerdict {
        self.verdict.take().unwrap_or(StepVerdict::Apply)
    }

    /// The step just trained is reported applied (its update taken) or skipped; the first call, before any step, is
    /// the count a resumed run continues from.
    fn updates_applied(&mut self, applied: u32) {
        self.weights_moved = applied > 0;
        let before = self.applied.replace(applied);
        if before.is_none() || self.trained.is_empty() {
            return;
        }
        let groups = std::mem::take(&mut self.trained);
        let step = self.stats.steps.len();
        let event = match before {
            Some(b) if applied > b => json!({"type": "applied", "step": step, "groups": groups}),
            _ => json!({"type": "applied", "step": step, "groups": groups, "skipped": self.stats.steps.last().and_then(|s| s.skipped.clone()).unwrap_or_else(|| "the update gate skipped the step".to_string())}),
        };
        self.uncommitted.extend(groups);
        self.feed.push_event(event);
    }

    /// Keys consumed so far (committed once this state is written) and the step history.
    fn cursor(&self) -> Option<Value> {
        let consumed: Vec<&String> = self.committed.iter().chain(&self.uncommitted).collect();
        Some(json!({"consumed": consumed, "steps": self.stats.steps, "steps_skipped_nonfinite": self.stats.steps_skipped_nonfinite, "steps_skipped_guard": self.stats.steps_skipped_guard}))
    }

    fn restore_cursor(&mut self, cursor: &Value) -> Result<(), std::io::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct History {
            consumed: Vec<String>,
            steps: Vec<super::external::PolicyGradientStep>,
            steps_skipped_nonfinite: usize,
            steps_skipped_guard: usize,
        }
        let history: History = serde_json::from_value(cursor.clone()).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("the streamed policy-gradient state: {e}")))?;
        self.feed.restore_known(&history.consumed);
        self.committed = history.consumed;
        self.stats.steps = history.steps;
        self.stats.steps_skipped_nonfinite = history.steps_skipped_nonfinite;
        self.stats.steps_skipped_guard = history.steps_skipped_guard;
        Ok(())
    }

    fn state_saved(&mut self, completed: u32) {
        if self.uncommitted.is_empty() {
            return;
        }
        let groups = std::mem::take(&mut self.uncommitted);
        self.feed.push_event(json!({"type": "committed", "step": completed, "groups": groups}));
        self.committed.extend(groups);
    }

    /// A stream holds nothing to probe before it is fed.
    fn loss_probe(&mut self, _model: &M, _rng: &mut Rng) -> f32 {
        f32::NAN
    }

    fn step_log(&self) -> Option<String> {
        let last = self.stats.steps.last()?;
        let s = &self.step_extra;
        let kl = last.kl.map(|kl| format!("kl {kl:.4}  ")).unwrap_or_default();
        Some(format!("{kl}clip {:.3}  ratio {:.3}  tokens {}  stale mean {:.2} max {:.0}  waited {:.2}s  queue {} groups {} tokens{}", last.clip_fraction, last.ratio_mean, s.policy_tokens, s.staleness_mean, s.staleness_max, s.wait_secs, s.queue_groups, s.queue_tokens, if last.skipped.is_some() { "  SKIPPED" } else { "" }))
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        let Some(last) = self.stats.steps.last() else { return Vec::new() };
        let s = &self.step_extra;
        let mut m = vec![
            ("pg_loss", last.loss),
            ("pg_clip_fraction", last.clip_fraction),
            ("pg_clip_fraction_high", last.clip_fraction_high),
            ("pg_clip_fraction_low", last.clip_fraction_low),
            ("pg_clip_fraction_dual", last.clip_fraction_dual),
            ("pg_entropy_estimate", last.entropy_estimate),
            ("pg_response_tokens", last.mean_response_tokens),
            ("pg_grad_norm", last.grad_norm),
            ("pg_ratio_mean", last.ratio_mean),
            ("pg_ratio_min", last.ratio_min),
            ("pg_ratio_max", last.ratio_max),
            ("pg_mean_advantage", last.mean_advantage),
            ("pg_staleness_mean", s.staleness_mean),
            ("pg_staleness_max", s.staleness_max),
            ("pg_policy_tokens", s.policy_tokens as f32),
            ("pg_learner_wait_seconds", s.wait_secs),
            ("pg_queue_groups", s.queue_groups as f32),
            ("pg_queue_tokens", s.queue_tokens as f32),
            ("pg_time_in_queue_seconds", s.time_in_queue_secs),
            ("pg_dropped_stale", s.dropped_stale as f32),
        ];
        if let Some(kl) = last.kl {
            m.push(("pg_kl", kl));
        }
        if self.feed.settings.top_k.is_some() {
            m.extend([("pg_sampler_mass", last.sampler_mass), ("pg_centred_masked_fraction", last.centred_masked_fraction)]);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(tokens: usize, version: Option<u64>) -> StreamExample {
        StreamExample { ids: vec![1; tokens], mask: vec![true; tokens], behavior: vec![-1.0; tokens], advantage: 1.0, policy_version: version, top: None }
    }

    fn group(key: &str, tokens: usize, version: Option<u64>) -> StreamGroup {
        StreamGroup { key: key.to_string(), examples: vec![example(tokens + 1, version)] }
    }

    fn feed(step: usize, queue: usize, staleness: Option<u64>) -> Arc<RolloutFeed> {
        let feed = RolloutFeed::new(StreamSettings { step_tokens: step, queue_tokens: queue, max_staleness: staleness, block: 64, top_k: None }, || true, Default::default()).unwrap();
        assert_eq!(feed.offer(vec![group("early", 1, None)]), Err(FeedRefusal::Starting), "nothing is taken before the learner is ready");
        feed.open();
        feed
    }

    /// The queue refuses an offer only when it holds a step already and the offer would pass its bound; a key is taken
    /// once.
    #[test]
    fn an_offer_is_refused_only_past_a_full_step_and_a_key_is_taken_once() {
        let f = feed(10, 20, None);
        // An offer larger than the whole queue is still taken while the learner waits for a step.
        let a = f.offer(vec![group("a", 25, None)]).unwrap();
        assert_eq!((a.accepted.clone(), a.queued_tokens), (vec!["a".to_string()], 25));
        assert!(matches!(f.offer(vec![group("b", 1, None)]), Err(FeedRefusal::Full { queued_tokens: 25, allowance: 0, offered_tokens: 1 })));
        let again = f.offer(vec![group("a", 25, None)]).unwrap();
        assert_eq!((again.accepted.len(), again.duplicates.clone()), (0, vec!["a".to_string()]), "a re-sent group is a duplicate, not a refusal");
        let Ok(taken) = f.take_step() else { panic!("a step is queued") };
        assert_eq!(taken.groups.len(), 1);
        assert_eq!(f.offer(vec![group("b", 5, None)]).unwrap().accepted, vec!["b".to_string()]);
        f.finish();
        assert_eq!(f.take_step().ok().map(|t| t.groups.len()), Some(1), "a finished feed trains what it holds");
        assert!(matches!(f.take_step(), Err(Supply::Finished)));
        assert_eq!(f.offer(vec![group("c", 5, None)]), Err(FeedRefusal::Closed));
    }

    /// A step takes whole groups in arrival order until it holds its tokens; a group too far behind the published
    /// version is dropped and reported, never trained.
    #[test]
    fn a_step_takes_whole_groups_in_order_and_drops_the_stale() {
        let f = feed(6, 100, Some(1));
        f.set_version(3);
        f.offer(vec![group("old", 4, Some(1)), group("x", 4, Some(2)), group("y", 4, None), group("z", 4, Some(3))]).unwrap();
        let Ok(taken) = f.take_step() else { panic!("a step") };
        assert_eq!(taken.groups.iter().map(|q| q.group.key.as_str()).collect::<Vec<_>>(), ["x", "y"]);
        assert_eq!(taken.dropped_stale, vec!["old".to_string()]);
        assert_eq!(taken.queue_tokens, 4);
        assert_eq!(f.drain_events(), vec![json!({"type": "dropped", "reason": "stale", "groups": ["old"]})]);
    }
}
