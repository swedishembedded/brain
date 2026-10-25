// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The ONE resident glue between a residency [`Instance`] and the generic
//! continuous-batching scheduler, `model::serve::Scheduler<D: PagedDecoder>`.
//! A crate of its own because both the catalog's residents and a model crate's
//! multi-device resident (qwen35's released GGUF) build on it.
//!
//! Every text LLM brain serves - Qwen3, Qwen3.8 (fp32 and the quantized
//! released checkpoint), Qwen3.5-MoE - hands its decoder to a `Scheduler` and
//! drives it through [`run_batch_scheduled`]. The scheduler owns admission,
//! per-sequence sampling streams (a request's seed reproduces its tokens
//! whatever else shares the batch), stop ids and preemption; a decode
//! iteration carries one row per running sequence in a single batched
//! forward. What differs between model families is only how a request becomes
//! a prompt, and which adapter (if any) it runs under; that is the
//! [`Frontend`] seam, so no family needs its own copy of the drive loop.
//!
//! Scope of "continuous": sequences join and leave the batch between steps
//! and between prefills, for everything handed to ONE `run_batch` call (the
//! dispatcher groups queued jobs into such a call). Work that arrives while
//! a call is already running waits for the next call.

use capability::{ActionResult, Invocation, Progress};
use data::qwen_tokenizer::QwenBpe;
use model::serve::{PagedDecoder, RejectReason, Request, Scheduler};
pub use qwen3::chat::ParsedRequest;
use qwen3::chat::SeqState;
use residency::{AdapterIdentity, Instance, JobFeed};

/// How one model family turns a request into a prompt and names its failures.
pub trait Frontend {
    /// The tokenizer prompts are built with and generated ids decode through.
    fn tok(&self) -> &QwenBpe;
    /// The ids that end a generation.
    fn stop(&self) -> &[u32];
    /// A short family label for logs and scheduler rejections.
    fn label(&self) -> &str;
    /// Render and tokenize `inv`.
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String>;
    /// The adapter `inv` runs under; `None` is the base model.
    fn route(&self, _inv: &Invocation) -> Result<Option<(model::serve::AdapterId, AdapterIdentity)>, String> {
        Ok(None)
    }
    /// Whether what [`Self::route`] answers from is out of date, so a request
    /// routed now could run on a release that has since been replaced. An open
    /// group that is idle (nothing running, so nothing a refresh could change
    /// under it) hands control back to its instance instead of admitting
    /// further jobs, and the instance refreshes before it calls again.
    fn stale(&self) -> bool {
        false
    }
    /// `sha256:<hex>` of the base checkpoint this front end serves, once it is
    /// known (hashing a multi-gigabyte file takes a while and does not hold
    /// requests up). `None` until then, or when the front end does not know it.
    fn base_digest(&self) -> Option<String> {
        None
    }
    /// The caller-facing message for a request the scheduler refused.
    fn rejection(&self, _inv: &Invocation, reason: &RejectReason) -> String {
        format!("{}: {reason}", self.label())
    }
}

/// A shared front end is a front end: an instance that is rebuilt (a demoted one promoted again) keeps its tokenizer
/// by sharing it.
impl<F: Frontend + ?Sized> Frontend for std::sync::Arc<F> {
    fn tok(&self) -> &QwenBpe {
        (**self).tok()
    }
    fn stop(&self) -> &[u32] {
        (**self).stop()
    }
    fn label(&self) -> &str {
        (**self).label()
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        (**self).parse(inv)
    }
    fn route(&self, inv: &Invocation) -> Result<Option<(model::serve::AdapterId, AdapterIdentity)>, String> {
        (**self).route(inv)
    }
    fn stale(&self) -> bool {
        (**self).stale()
    }
    fn base_digest(&self) -> Option<String> {
        (**self).base_digest()
    }
    fn rejection(&self, inv: &Invocation, reason: &RejectReason) -> String {
        (**self).rejection(inv, reason)
    }
}

/// One request in a [`Group`]: its parsed streaming state and the scheduler id
/// it runs under.
struct Job {
    invocation: Invocation,
    /// `None` once the job has its result (or never had a scheduled request).
    seq: Option<SeqState>,
    id: Option<u64>,
    /// The release the request's adapter was routed to; a reply names it.
    identity: Option<AdapterIdentity>,
    /// What the request asked to get back about the tokens it was decoded
    /// from, and the prompt ids to return when it asked for them.
    capture: Capture,
    /// The base checkpoint's digest when the request was admitted, if known.
    base_digest: Option<String>,
}

/// The token-level facts a request asks to have returned (`logprobs`,
/// `return_token_ids`): off unless set, so a request pays only for what it
/// reads.
#[derive(Default)]
struct Capture {
    logprobs: bool,
    token_ids: bool,
    prompt: Vec<u32>,
}

impl Capture {
    fn of(inv: &Invocation, prompt: &[u32]) -> Capture {
        let logprobs = inv.get_bool("logprobs").unwrap_or(false);
        let token_ids = inv.get_bool("return_token_ids").unwrap_or(false);
        Capture { logprobs, token_ids, prompt: if token_ids { prompt.to_vec() } else { Vec::new() } }
    }

    /// `outcome` with the requested ids and log-probabilities added.
    fn attach(&self, outcome: capability::Outcome, tokens: &[u32], logprobs: &[f32]) -> capability::Outcome {
        let mut outcome = outcome;
        if self.token_ids {
            outcome = outcome.set("prompt_token_ids", serde_json::json!(self.prompt)).set("completion_token_ids", serde_json::json!(tokens));
        }
        if self.logprobs {
            outcome = outcome.set("completion_logprobs", serde_json::json!(logprobs));
            if !self.token_ids {
                outcome = outcome.set("completion_token_ids", serde_json::json!(tokens));
            }
        }
        outcome
    }
}

/// The requests of one `run_batch` / `run_open` call on a persistent
/// scheduler: they are admitted into it (at the start, or later between
/// steps), streamed and finished here. A job is addressed by its index in
/// arrival order.
struct Group<'a, D: PagedDecoder> {
    sched: &'a mut Scheduler<D>,
    front: &'a dyn Frontend,
    jobs: Vec<Job>,
    /// Indices of jobs still running in the scheduler.
    open: std::collections::HashSet<usize>,
}

/// How [`run_open_scheduled`] ended.
#[derive(Debug, PartialEq, Eq)]
pub enum OpenGroup {
    /// The feed closed with nothing running.
    Closed,
    /// The group is idle and the frontend is stale; `numbered` jobs have been
    /// numbered so far.
    Refresh { numbered: usize },
}

impl<'a, D: PagedDecoder> Group<'a, D> {
    /// Resume numbering after `n` jobs of an earlier call of the same open
    /// group, all of which are finished: placeholders keep indices aligned.
    fn skip_numbered(&mut self, n: usize) {
        self.jobs.extend((0..n).map(|_| Job { invocation: Invocation::new(), seq: None, id: None, identity: None, capture: Capture::default(), base_digest: None }));
    }

    fn new(sched: &'a mut Scheduler<D>, front: &'a dyn Frontend) -> Self {
        Group { sched, front, jobs: Vec::new(), open: std::collections::HashSet::new() }
    }

    /// Admit one request. A request that cannot even be parsed or routed has
    /// its error back at once; otherwise it is queued in the scheduler and
    /// `None` is returned.
    fn admit(&mut self, inv: &Invocation, progress: &mut dyn FnMut(usize, Progress)) -> (usize, Option<ActionResult>) {
        let bi = self.jobs.len();
        let mut job = Job { invocation: inv.clone(), seq: None, id: None, identity: None, capture: Capture::default(), base_digest: self.front.base_digest() };
        // A name the bank no longer holds is the caller's "no model", never
        // the base's answer under another name.
        let adapter = match self.front.route(inv) {
            Ok(Some((adapter, identity))) => {
                job.identity = Some(identity);
                Some(adapter)
            }
            Ok(None) => None,
            Err(e) => {
                self.jobs.push(job);
                return (bi, Some(Err(e)));
            }
        };
        match self.front.parse(inv) {
            Ok(req) => {
                let sample = req.sampler;
                let seed = req.seed;
                let max_new = req.max_new;
                job.capture = Capture::of(inv, &req.ids);
                let seq = SeqState::new(&req, inv.cancel.clone());
                let request = Request { prompt: req.ids, max_new, stop: self.front.stop().to_vec() };
                let id = match adapter {
                    Some(adapter) => self.sched.submit_for_adapter(request, adapter, sample, seed),
                    None => self.sched.submit_sampled(request, sample, seed),
                };
                progress(bi, Progress::step(0, max_new as u32, "generating"));
                job.seq = Some(seq);
                job.id = Some(id);
                self.jobs.push(job);
                self.open.insert(bi);
                (bi, None)
            }
            Err(e) => {
                self.jobs.push(job);
                (bi, Some(Err(e)))
            }
        }
    }

    /// Why a request can never be served by this group's scheduler, in the words its caller
    /// gets: a route or parse failure (an unknown adapter, a malformed message, a template
    /// error), or a request the scheduler would refuse at admission (too long for the
    /// engine, a token outside the vocabulary, an adapter it does not hold). `None` when it
    /// may be queued.
    fn refusal(&mut self, inv: &Invocation) -> Option<String> {
        let adapter = match self.front.route(inv) {
            Ok(Some((adapter, _))) => Some(adapter),
            Ok(None) => None,
            Err(e) => return Some(e),
        };
        let req = match self.front.parse(inv) {
            Ok(req) => req,
            Err(e) => return Some(e),
        };
        let request = Request { prompt: req.ids, max_new: req.max_new, stop: Vec::new() };
        self.sched.refusal(&request, adapter).map(|reason| self.front.rejection(inv, &reason))
    }

    /// Jobs still running.
    fn live(&self) -> usize {
        self.open.len()
    }

    /// What served the reply: the adapter that answered and, when known, the
    /// base checkpoint's digest, the backend it ran on and the engine version.
    fn named(outcome: capability::Outcome, job: &Job) -> capability::Outcome {
        let mut outcome = outcome.set("backend", serde_json::json!(gpu_core::backend_name())).set("engine", serde_json::json!(concat!("brain ", env!("CARGO_PKG_VERSION"))));
        if let Some(digest) = &job.base_digest {
            outcome = outcome.set("base_digest", serde_json::json!(digest));
        }
        match &job.identity {
            Some(identity) => outcome.set("adapter", serde_json::json!({ "id": identity.id, "digest": identity.digest })),
            None => outcome,
        }
    }

    /// One iteration: stream each open job's new suffix and check its stop
    /// strings and cancellation (a triggered job is cancelled and finished at
    /// once - `Scheduler::cancel` returns its tokens so far and reclaims its
    /// blocks), then ONE scheduler step (admissions, then a single batched
    /// decode over every running sequence). Returns the jobs that finished.
    /// A request refused at admission never completes, so its error is
    /// delivered from the step report instead of leaving the group waiting.
    fn pump(&mut self, progress: &mut dyn FnMut(usize, Progress)) -> Vec<(usize, ActionResult)> {
        let tok = self.front.tok();
        let mut finished: Vec<(usize, ActionResult)> = Vec::new();
        let mut order: Vec<usize> = self.open.iter().copied().collect();
        order.sort_unstable();
        for bi in order {
            let id = self.jobs[bi].id.expect("an open job is scheduled");
            let Some(all_tokens) = self.sched.tokens_of(id) else { continue };
            let seq = self.jobs[bi].seq.as_mut().expect("an open job has its stream");
            if seq.advance(tok, all_tokens, &mut |p| progress(bi, p)) {
                let (toks, logprobs) = self.sched.cancel_detailed(id).unwrap_or_default();
                let seq = self.jobs[bi].seq.take().expect("checked above");
                let outcome = Self::named(seq.finish(tok, &toks, &mut |p| progress(bi, p)), &self.jobs[bi]);
                let outcome = self.jobs[bi].capture.attach(outcome, &toks, &logprobs);
                self.open.remove(&bi);
                finished.push((bi, Ok(outcome)));
            }
        }
        if self.open.is_empty() {
            return finished;
        }
        let report = self.sched.step_report();
        // Gated: a decode loop calls this once per token, so an unconditional
        // `format!` would cost real throughput even with nothing printed.
        if residency::log::verbosity() >= 3 {
            residency::log::debug(&format!(
                "{} batch step: {} running, {} waiting, admitted={}, produced={}, finished={}, rejected={}, demoted={}, promoted={}, recomputed={}",
                self.front.label(),
                self.sched.running_len(),
                self.sched.waiting_len(),
                report.admitted.len(),
                report.produced.len(),
                report.finished.len(),
                report.rejected.len(),
                report.demoted.len(),
                report.promoted.len(),
                report.recomputed.len(),
            ));
        }
        let job_of = |jobs: &[Job], id: u64| jobs.iter().position(|j| j.id == Some(id)).expect("a scheduler id belongs to this group");
        for (id, reason) in report.rejected {
            let bi = job_of(&self.jobs, id);
            self.jobs[bi].seq = None;
            self.open.remove(&bi);
            finished.push((bi, Err(self.front.rejection(&self.jobs[bi].invocation, &reason))));
        }
        let mut logprobs_of: std::collections::HashMap<u64, Vec<f32>> = report.completed_logprobs.into_iter().collect();
        for (id, toks) in report.completed {
            let bi = job_of(&self.jobs, id);
            let logprobs = logprobs_of.remove(&id).unwrap_or_default();
            if let Some(seq) = self.jobs[bi].seq.take() {
                let outcome = Self::named(seq.finish(tok, &toks, &mut |p| progress(bi, p)), &self.jobs[bi]);
                let outcome = self.jobs[bi].capture.attach(outcome, &toks, &logprobs);
                finished.push((bi, Ok(outcome)));
            }
            self.open.remove(&bi);
        }
        finished
    }
}

/// Drive every invocation in `invs` to completion on the SAME persistent
/// `sched` (built once at activation, so the KV pool and prefix cache are
/// shared and reused across calls). One result per invocation, in order. See
/// [`Group::pump`] for what an iteration does.
pub fn run_batch_scheduled<D: PagedDecoder>(
    sched: &mut Scheduler<D>,
    front: &dyn Frontend,
    invs: &[Invocation],
    progress: &mut dyn FnMut(usize, Progress),
) -> Vec<ActionResult> {
    residency::log::info(&format!("{} batch: {} request(s) admitted to the scheduler", front.label(), invs.len()));
    let mut group = Group::new(sched, front);
    let mut results: Vec<Option<ActionResult>> = vec![None; invs.len()];
    for inv in invs {
        let (bi, immediate) = group.admit(inv, progress);
        results[bi] = immediate;
    }
    while group.live() > 0 {
        for (bi, result) in group.pump(progress) {
            results[bi] = Some(result);
        }
    }
    results.into_iter().map(|r| r.expect("every batch index resolved")).collect()
}

/// [`run_batch_scheduled`] as an OPEN group: besides `invs`, jobs the feed
/// delivers while the group runs are admitted into the scheduler between
/// steps (up to `capacity` running at once), and every job's result is handed
/// to the feed the moment it is final. Returns when nothing is running and the
/// feed has closed - continuous batching across everything queued for the
/// instance, not only what was queued when the call began.
///
/// Job indices continue from `numbered` (0 for a new group): the feed numbers
/// the jobs of one group across calls. Returns [`OpenGroup::Closed`] when the
/// feed closed, or [`OpenGroup::Refresh`] when the group went idle while the
/// frontend was [stale](Frontend::stale): no further job was taken, and the
/// caller refreshes the frontend's routes and calls again with `invs` empty
/// and the `numbered` it was given.
pub fn run_open_scheduled<D: PagedDecoder>(
    sched: &mut Scheduler<D>,
    front: &dyn Frontend,
    invs: &[Invocation],
    capacity: usize,
    numbered: usize,
    progress: &mut dyn FnMut(usize, Progress),
    feed: &mut dyn JobFeed,
) -> OpenGroup {
    residency::log::info(&format!("{} open batch: {} request(s) admitted to the scheduler", front.label(), invs.len()));
    let mut group = Group::new(sched, front);
    group.skip_numbered(numbered);
    for inv in invs {
        let (bi, immediate) = group.admit(inv, progress);
        if let Some(result) = immediate {
            feed.finish(bi, result);
        }
    }
    loop {
        if group.live() == 0 && front.stale() {
            return OpenGroup::Refresh { numbered: group.jobs.len() };
        }
        // Whatever waits for a slot is first asked whether it can ever run: a request that
        // cannot is answered now, not after the running ones finish.
        feed.screen(&mut |inv| group.refusal(inv));
        for inv in feed.take(capacity.saturating_sub(group.live())) {
            let (bi, immediate) = group.admit(&inv, progress);
            if let Some(result) = immediate {
                feed.finish(bi, result);
            }
        }
        if group.live() == 0 {
            if feed.close() {
                return OpenGroup::Closed;
            }
            continue;
        }
        for (bi, result) in group.pump(progress) {
            feed.finish(bi, result);
        }
    }
}

/// The prefix-cache metrics every scheduler-backed resident reports through
/// `Executor::stats().metrics`.
pub fn scheduler_metrics<D: PagedDecoder>(sched: &Scheduler<D>) -> Vec<(String, serde_json::Value)> {
    let (hit, looked, cached) = sched.prefix_stats();
    let rate = if looked > 0 { hit as f64 / looked as f64 } else { 0.0 };
    vec![
        ("kv_prefix_hit_rate".to_string(), serde_json::json!(rate)),
        ("kv_prefix_hit_tokens".to_string(), serde_json::json!(hit)),
        ("kv_prefix_lookup_tokens".to_string(), serde_json::json!(looked)),
        ("kv_prefix_cached_blocks".to_string(), serde_json::json!(cached)),
    ]
}

/// A residency [`Instance`] that serves one text family through the generic
/// scheduler: the instance every non-adapter LLM resident builds.
pub struct ScheduledInstance<D: PagedDecoder, F: Frontend> {
    pub front: F,
    pub sched: Scheduler<D>,
    /// How many requests may run in the scheduler at once (an open group
    /// admits late jobs up to this).
    capacity: usize,
    /// The last `run_batch` call's size, generated tokens and wall time.
    last: LastBatch,
}

/// What [`Instance::metrics`] reports about the last `run_batch` call: the
/// aggregate rate batching moves.
#[derive(Clone, Copy, Default)]
struct LastBatch {
    requests: usize,
    tokens: usize,
    seconds: f64,
}

impl<D: PagedDecoder, F: Frontend> ScheduledInstance<D, F> {
    /// `capacity` is the number of requests the scheduler runs at once (its
    /// `max_running`), the width of an open group.
    pub fn new(front: F, sched: Scheduler<D>, capacity: usize) -> Self {
        ScheduledInstance { front, sched, capacity, last: LastBatch::default() }
    }
}

impl<D: PagedDecoder + Send, F: Frontend + Send> Instance for ScheduledInstance<D, F> {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    fn run_batch(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        let started = std::time::Instant::now();
        let results = run_batch_scheduled(&mut self.sched, &self.front, invs, progress);
        let tokens = results.iter().filter_map(|r| r.as_ref().ok()).filter_map(|o| o.outputs.get("completion_tokens").and_then(|v| v.as_u64())).sum::<u64>() as usize;
        self.last = LastBatch { requests: invs.len(), tokens, seconds: started.elapsed().as_secs_f64() };
        results
    }

    fn run_open(&mut self, _action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn JobFeed) -> bool {
        let started = std::time::Instant::now();
        let capacity = self.capacity;
        let mut counting = CountingFeed { inner: feed, requests: invs.len(), tokens: 0 };
        run_open_scheduled(&mut self.sched, &self.front, invs, capacity, 0, progress, &mut counting);
        self.last = LastBatch { requests: counting.requests, tokens: counting.tokens, seconds: started.elapsed().as_secs_f64() };
        true
    }

    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        let mut m = scheduler_metrics(&self.sched);
        let l = self.last;
        let rate = if l.seconds > 0.0 { l.tokens as f64 / l.seconds } else { 0.0 };
        m.push(("last_batch_requests".to_string(), serde_json::json!(l.requests)));
        m.push(("last_batch_generated_tokens".to_string(), serde_json::json!(l.tokens)));
        m.push(("last_batch_seconds".to_string(), serde_json::json!(l.seconds)));
        m.push(("last_batch_tok_per_s".to_string(), serde_json::json!(rate)));
        m
    }
}

/// A [`JobFeed`] that also counts the group's jobs and generated tokens, for
/// the instance's last-batch metrics.
struct CountingFeed<'a> {
    inner: &'a mut dyn JobFeed,
    requests: usize,
    tokens: usize,
}

impl JobFeed for CountingFeed<'_> {
    fn take(&mut self, max: usize) -> Vec<Invocation> {
        let invs = self.inner.take(max);
        self.requests += invs.len();
        invs
    }
    fn finish(&mut self, index: usize, result: ActionResult) {
        if let Ok(o) = &result {
            self.tokens += o.outputs.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        }
        self.inner.finish(index, result);
    }
    fn close(&mut self) -> bool {
        self.inner.close()
    }
    fn screen(&mut self, check: &mut dyn FnMut(&Invocation) -> Option<String>) {
        self.inner.screen(check)
    }
}

/// Request parsing for a family with no adapters: a function from tokenizer and
/// invocation to a parsed request (a family's chat template flavour is chosen
/// inside it).
///
/// A boxed closure rather than a plain `fn` pointer because a model whose chat
/// format is a *compiled* artifact (a checkpoint's own Jinja template) has to
/// keep that artifact between requests; a bare function could only recompile
/// it per call.
pub type ParseFn = std::sync::Arc<dyn Fn(&QwenBpe, &Invocation) -> Result<ParsedRequest, String> + Send + Sync>;

/// The [`Frontend`] of a base-model-only family.
pub struct TextFrontend {
    pub tok: QwenBpe,
    pub stop: Vec<u32>,
    pub label: &'static str,
    pub parse: ParseFn,
}

impl Frontend for TextFrontend {
    fn tok(&self) -> &QwenBpe {
        &self.tok
    }
    fn stop(&self) -> &[u32] {
        &self.stop
    }
    fn label(&self) -> &str {
        self.label
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        (self.parse)(&self.tok, inv)
    }
}
