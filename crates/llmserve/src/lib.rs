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
use model::serve::{PagedDecoder, RejectReason, Request, SampleParams, Scheduler};
use qwen3::chat::{ParsedRequest, SeqState};
use residency::{AdapterIdentity, Instance};

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
    /// The caller-facing message for a request the scheduler refused.
    fn rejection(&self, _inv: &Invocation, reason: &RejectReason) -> String {
        format!("{}: {reason}", self.label())
    }
}

/// Drive every invocation in `invs` to completion on the SAME persistent
/// `sched` (built once at activation, so the KV pool and prefix cache are
/// shared and reused across calls).
///
/// Each iteration streams every open sequence's new suffix and checks its
/// stop strings and cancellation (a triggered sequence is cancelled at once;
/// `Scheduler::cancel` returns its tokens and reclaims its blocks), then runs
/// ONE scheduler step: admissions, then a single batched decode over all
/// running sequences. A request refused at admission never completes, so its
/// error is delivered from the step report instead of leaving the loop
/// waiting on it.
pub fn run_batch_scheduled<D: PagedDecoder>(
    sched: &mut Scheduler<D>,
    front: &dyn Frontend,
    invs: &[Invocation],
    progress: &mut dyn FnMut(usize, Progress),
) -> Vec<ActionResult> {
    let tok = front.tok();
    residency::log::info(&format!("{} batch: {} request(s) admitted to the scheduler", front.label(), invs.len()));
    let mut results: Vec<Option<ActionResult>> = vec![None; invs.len()];
    let mut seq_for_bi: Vec<Option<SeqState>> = Vec::with_capacity(invs.len());
    let mut id_for_bi: Vec<Option<u64>> = Vec::with_capacity(invs.len());
    // The release each request's adapter was routed to; a reply names it.
    let mut identity_for_bi: Vec<Option<AdapterIdentity>> = vec![None; invs.len()];
    let named = |outcome: capability::Outcome, identity: &Option<AdapterIdentity>| match identity {
        Some(identity) => outcome.set("adapter", serde_json::json!({ "id": identity.id, "digest": identity.digest })),
        None => outcome,
    };

    for (bi, inv) in invs.iter().enumerate() {
        // A name the bank no longer holds is the caller's "no model", never
        // the base's answer under another name.
        let (adapter, identity) = match front.route(inv) {
            Ok(Some((adapter, identity))) => (Some(adapter), Some(identity)),
            Ok(None) => (None, None),
            Err(e) => {
                results[bi] = Some(Err(e));
                seq_for_bi.push(None);
                id_for_bi.push(None);
                continue;
            }
        };
        identity_for_bi[bi] = identity;
        match front.parse(inv) {
            Ok(req) => {
                let sample = SampleParams { temp: req.temp, top_k: req.top_k, top_p: req.top_p };
                let seed = req.seed;
                let max_new = req.max_new;
                let seq = SeqState::new(&req, inv.cancel.clone());
                let request = Request { prompt: req.ids, max_new, stop: front.stop().to_vec() };
                let id = match adapter {
                    Some(adapter) => sched.submit_for_adapter(request, adapter, sample, seed),
                    None => sched.submit_sampled(request, sample, seed),
                };
                progress(bi, Progress::step(0, max_new as u32, "generating"));
                seq_for_bi.push(Some(seq));
                id_for_bi.push(Some(id));
            }
            Err(e) => {
                results[bi] = Some(Err(e));
                seq_for_bi.push(None);
                id_for_bi.push(None);
            }
        }
    }

    let mut remaining: std::collections::HashSet<usize> = (0..invs.len()).filter(|&bi| id_for_bi[bi].is_some()).collect();
    while !remaining.is_empty() {
        let mut just_finished = Vec::new();
        for &bi in &remaining {
            let id = id_for_bi[bi].unwrap();
            let Some(all_tokens) = sched.tokens_of(id) else { continue };
            let seq = seq_for_bi[bi].as_mut().unwrap();
            if seq.advance(tok, all_tokens, &mut |p| progress(bi, p)) {
                let toks = sched.cancel(id).unwrap_or_default();
                let seq = seq_for_bi[bi].take().unwrap();
                results[bi] = Some(Ok(named(seq.finish(tok, &toks, &mut |p| progress(bi, p)), &identity_for_bi[bi])));
                just_finished.push(bi);
            }
        }
        for bi in just_finished {
            remaining.remove(&bi);
        }
        if remaining.is_empty() {
            break;
        }
        let report = sched.step_report();
        // Gated: a decode loop calls this once per token, so an unconditional
        // `format!` would cost real throughput even with nothing printed.
        if residency::log::verbosity() >= 3 {
            residency::log::debug(&format!(
                "{} batch step: {} running, {} waiting, admitted={}, produced={}, finished={}, rejected={}, demoted={}, promoted={}",
                front.label(),
                sched.running_len(),
                sched.waiting_len(),
                report.admitted.len(),
                report.produced.len(),
                report.finished.len(),
                report.rejected.len(),
                report.demoted.len(),
                report.promoted.len(),
            ));
        }
        for (id, reason) in report.rejected {
            let bi = id_for_bi.iter().position(|x| *x == Some(id)).expect("rejected id must belong to this batch");
            seq_for_bi[bi] = None;
            results[bi] = Some(Err(front.rejection(&invs[bi], &reason)));
            remaining.remove(&bi);
        }
        for (id, toks) in report.completed {
            let bi = id_for_bi.iter().position(|x| *x == Some(id)).expect("completed id must belong to this batch");
            if let Some(seq) = seq_for_bi[bi].take() {
                results[bi] = Some(Ok(named(seq.finish(tok, &toks, &mut |p| progress(bi, p)), &identity_for_bi[bi])));
            }
            remaining.remove(&bi);
        }
    }
    results.into_iter().map(|r| r.expect("every batch index resolved")).collect()
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
    pub fn new(front: F, sched: Scheduler<D>) -> Self {
        ScheduledInstance { front, sched, last: LastBatch::default() }
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

/// Request parsing for a family with no adapters: a plain function from
/// tokenizer and invocation to a parsed request (a family's chat template
/// flavour is chosen inside it).
pub type ParseFn = fn(&QwenBpe, &Invocation) -> Result<ParsedRequest, String>;

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
