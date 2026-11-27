// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training a served policy online, on the rollout groups an orchestrator
// produces against the same server while it trains, for its clients. If your team needs expertise in asynchronous
// policy-gradient training with bounded staleness and backpressure then you can procure our services by sending an
// email to info@swedishembedded.com.

//! A policy-gradient job fed while it runs: `brain/train` with `stream` instead of `data`.
//!
//! The job is created like any training job (`POST /v1/jobs`, D-Bus `Subscribe`) and then takes input from its
//! client through the job's inbox ([`capability::inbox`]): `records` (a JSONL body of training-record contract v2
//! rollout records forming complete groups) and `finish`. The records go through [`brain::RolloutStream`] (the batch
//! path's checks, the group advantages on arrival, the token-bounded queue, deduplication by group key; see
//! [`rl::objective::stream_pg`]), and the fine-tune trains a step whenever `step_tokens` policy tokens are queued.
//! Every `publish_every` steps the adapter is offered to the serving bank under `publish_adapter` as its next version
//! ([`crate::bank`]), and staleness is measured against that version. `finish` trains what is queued, publishes the
//! final adapter (to the bank and the artifact store, as a batch run's) and ends the job.
//!
//! **Receipts.** The job's events carry, beside each `step`: `applied {step, groups}` once a step's update is taken
//! (`skipped` names why when a guard skipped it), `committed {step, groups}` once a training state holding that update
//! is written (`checkpoint_every`, a stop, and the job's completion), `dropped {reason, groups}` for groups too stale to
//! train, and `published {version, digest, step, lineage}` once a version is routable. `lineage` is the version's
//! run-level identity, `{run, epoch, update, base_digest, adapter_digest}`, which its card records and every reply it
//! decodes names (`brain.adapter.lineage`): the run is the resume key (or `job:<hex>` drawn for a run without one,
//! named in the `stream` frame, beside the base's `tokenizer_digest`), the update the step. The consumed keys ride in the training
//! state, so a job that resumes the run (`resume_key`) answers every committed key as a duplicate; a client re-sends
//! every group it has no `committed` receipt for.
//!
//! **Epochs.** The first frame (`stream`) and every answer carry the run's epoch, one more for each job that continued
//! it ([`crate::runs::Record::epoch`]). A job that ended (a server restart included) leaves its id unknown to the
//! surface, so the client's next call is refused; a client that names the epoch it believes in (`?epoch=N`) is refused
//! with the current one when another job has taken the run over since.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use capability::inbox::{InputRefusal, InputRefusalKind, InputReply, JobInput};
use capability::{Blob, Invocation, Progress};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::job::{self, Inputs, JobEnv, RunFiles, Staged, Trained, Workdir};
use crate::request::{Objective, Params, Request, Source, SourceDoc};
use crate::store::{ArtifactStore, Digest};

/// The most optimiser steps a stream runs, and what one without `max_steps` runs: open-ended, at its peak rate.
pub const MAX_STREAM_STEPS: u32 = brain::OPEN_ENDED_STREAM_STEPS;
/// The most policy tokens one step trains.
pub const MAX_STEP_TOKENS: u64 = 1 << 24;
/// The most policy tokens a queue holds.
pub const MAX_QUEUE_TOKENS: u64 = 1 << 26;
/// The most published versions a stream keeps served under ids of their own: each holds a slot of the bank's.
pub const MAX_RETAINED_VERSIONS: u32 = 16;
/// The longest `stream` document accepted, in bytes.
pub const MAX_STREAM_DOCUMENT_BYTES: usize = 16 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamDoc {
    publish_adapter: String,
    publish_every: u32,
    step_tokens: u64,
    queue_tokens: u64,
    max_staleness: Option<u64>,
    max_steps: Option<u32>,
    top_k: Option<u64>,
    support: Option<u64>,
    start_adapter: Option<SourceDoc>,
    retain_versions: Option<u32>,
}

/// The `stream` document of a request.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamRefs {
    /// The adapter id every published version is served under.
    pub publish_adapter: String,
    pub publish_every: u32,
    pub step_tokens: usize,
    pub queue_tokens: usize,
    pub max_staleness: Option<u64>,
    pub max_steps: Option<u32>,
    /// The sampler top-K every record carries, for the score-centred estimator.
    pub top_k: Option<usize>,
    /// The widest sampler support a record may carry per token (`behavior_support_*`).
    pub support: Option<usize>,
    pub start_adapter: Option<Source>,
    /// How many of the newest published versions are also served under an id of their own (`<id>-v<N>`), so a
    /// client can stay on the version it started on, each also kept in the artifact store (named in its published
    /// frame's `artifact_digest`) so a later run can start from it; 0 serves and keeps none.
    pub retain_versions: usize,
}

impl StreamRefs {
    /// Parses and bounds the document; unknown fields are refused.
    pub fn parse(text: &str) -> Result<StreamRefs, String> {
        let refuse = |why: String| Err(format!("stream: {why}"));
        if text.len() > MAX_STREAM_DOCUMENT_BYTES {
            return refuse(format!("the document is {} bytes; at most {MAX_STREAM_DOCUMENT_BYTES} are accepted", text.len()));
        }
        let doc: StreamDoc = serde_json::from_str(text).map_err(|e| format!("stream: {e}"))?;
        if !brain_modelref::ModelRef::parse(&doc.publish_adapter).is_ok_and(|r| r.adapter().is_some()) {
            return refuse(format!("publish_adapter {:?} does not name an adapter: <vendor>/<repo>:<owner>:<name>:<tag>, as the server offers it", doc.publish_adapter));
        }
        if doc.publish_adapter.len() > crate::request::MAX_ADAPTER_ID_LEN {
            return refuse(format!("publish_adapter is longer than {} characters", crate::request::MAX_ADAPTER_ID_LEN));
        }
        if !(1..=MAX_STREAM_STEPS).contains(&doc.publish_every) {
            return refuse(format!("publish_every must be 1 to {MAX_STREAM_STEPS} steps"));
        }
        if !(1..=MAX_STEP_TOKENS).contains(&doc.step_tokens) {
            return refuse(format!("step_tokens must be 1 to {MAX_STEP_TOKENS}"));
        }
        if !(doc.step_tokens..=MAX_QUEUE_TOKENS).contains(&doc.queue_tokens) {
            return refuse(format!("queue_tokens must be step_tokens ({}) to {MAX_QUEUE_TOKENS}: the queue holds at least one step", doc.step_tokens));
        }
        if doc.max_steps.is_some_and(|n| !(1..=MAX_STREAM_STEPS).contains(&n)) {
            return refuse(format!("max_steps must be 1 to {MAX_STREAM_STEPS}"));
        }
        if doc.top_k.is_some_and(|k| !(1..=brain::MAX_STREAM_TOP_K as u64).contains(&k)) {
            return refuse(format!("top_k must be 1 to {}", brain::MAX_STREAM_TOP_K));
        }
        if doc.support.is_some_and(|n| !(1..=brain::MAX_STREAM_SUPPORT as u64).contains(&n)) {
            return refuse(format!("support must be 1 to {}", brain::MAX_STREAM_SUPPORT));
        }
        if doc.retain_versions.is_some_and(|n| n > MAX_RETAINED_VERSIONS) {
            return refuse(format!("retain_versions must be 0 to {MAX_RETAINED_VERSIONS}"));
        }
        let start_adapter = doc.start_adapter.map(|s| s.resolve("start_adapter").map_err(|e| format!("stream: {}", e.trim_start_matches("data.")))).transpose()?;
        if let Some(Source::Blob(name)) = &start_adapter {
            if name != "adapter" {
                return refuse(format!("start_adapter names the input blob {name:?}; the starting adapter is the input \"adapter\""));
            }
        }
        Ok(StreamRefs { publish_adapter: doc.publish_adapter, publish_every: doc.publish_every, step_tokens: doc.step_tokens as usize, queue_tokens: doc.queue_tokens as usize, max_staleness: doc.max_staleness, max_steps: doc.max_steps, top_k: doc.top_k.map(|k| k as usize), support: doc.support.map(|n| n as usize), start_adapter, retain_versions: doc.retain_versions.unwrap_or(0) as usize })
    }

    pub fn digests(&self) -> Vec<Digest> {
        match &self.start_adapter {
            Some(Source::Digest(d)) => vec![*d],
            _ => Vec::new(),
        }
    }

    /// The call carries the starting adapter's blob when the document names it, and no other.
    pub(crate) fn check_attached(&self, inv: &Invocation) -> Result<(), String> {
        let named = matches!(self.start_adapter, Some(Source::Blob(_)));
        match (named, inv.blobs.keys().next()) {
            (true, None) => Err("stream: the input blob \"adapter\" is named but not attached".to_string()),
            (false, Some(extra)) => Err(format!("stream: the input blob \"{extra}\" is attached but the document does not use it")),
            (true, Some(_)) if inv.blobs.len() > 1 || !inv.blobs.contains_key("adapter") => Err("stream: a stream job takes the input blob \"adapter\" alone".to_string()),
            _ => Ok(()),
        }
    }
}

/// The settings of the request a stream cannot honour, refused before a job exists.
pub(crate) fn check_request(params: &Params, refs: &StreamRefs) -> Result<(), String> {
    let refuse = |why: &str| Err(format!("train: stream: {why}"));
    if params.objective != Objective::PolicyGradient {
        return refuse(&format!("a stream trains the policy gradient; the objective is {}", params.objective.name()));
    }
    let foreign = [
        (params.cards != 1, "cards"),
        (params.job.is_some(), "job"),
        (params.grad_accum != 1, "grad_accum (a step is cut by step_tokens)"),
        (params.eval_every > 0, "eval_every"),
        (params.keep_evaluations, "keep_evaluations"),
        (params.replay_share.is_some(), "replay_share"),
        (params.distill_teacher.is_some(), "distill_teacher"),
    ];
    if let Some((_, name)) = foreign.iter().find(|(named, _)| *named) {
        return refuse(&format!("{name} is not taken by a stream job"));
    }
    if params.adapter_id.is_some() {
        return refuse("the adapter is written under stream.publish_adapter; drop adapter_id");
    }
    let config = crate::job::policy_gradient_config(params)?;
    brain::check_streamable(&config, refs.top_k).map_err(|why| format!("train: stream: {why}"))
}

/// What a stream job needs of the server beyond the request: a bank to publish to and a surface that delivers input.
pub(crate) fn admit(env: &JobEnv, inv: &Invocation) -> Result<(), String> {
    if env.bank.get().is_none() {
        return Err("train: stream: this server offers no adapter bank to publish the policy to (serve a base with --adapter-slots, --serve-adapters or another adapter flag)".to_string());
    }
    if !inv.inbox.is_armed() {
        return Err("train: stream: a stream job takes its records from the surface that started it (HTTP POST /v1/jobs, D-Bus Subscribe), and this call came without one".to_string());
    }
    // Records sent before the job starts wait for it.
    inv.inbox.will_read();
    Ok(())
}

/// The starting adapter, when the document names one, in the job's directory.
pub(crate) fn stage(refs: &StreamRefs, blobs: &BTreeMap<String, Blob>, store: Option<&ArtifactStore>, work: &Workdir) -> Result<Staged, String> {
    let start_adapter = match &refs.start_adapter {
        Some(source) => {
            let dest = work.path().join("start-adapter");
            job::bring(source, blobs, store, work.staging(), &dest)?;
            Some(dest)
        }
        None => None,
    };
    Ok(Staged { start_adapter, ..Staged::default() })
}

/// A streamed run's adapter and report, and the keys it trained since its last saved state.
pub(crate) struct Streamed {
    pub(crate) trained: Trained,
    pub(crate) uncommitted: Vec<String>,
}

/// The groups of an `applied`, `dropped` or `committed` frame.
fn groups_of(frame: &Value) -> Vec<String> {
    frame["groups"].as_array().into_iter().flatten().filter_map(|g| g.as_str().map(str::to_string)).collect()
}

/// Trains `request`'s stream, reporting every step and receipt as a frame, while a thread answers the job's input.
#[allow(clippy::too_many_arguments)]
pub(crate) fn train(env: &JobEnv, request: &Request, inputs: &Inputs<'_>, files: &RunFiles, inv: &Invocation, epoch: u64, began: Instant, progress: &mut dyn FnMut(Progress)) -> Result<Streamed, String> {
    let refs = request.stream.as_ref().expect("a stream request");
    let bank = env.bank.get().cloned().ok_or("train: stream: this server offers no adapter bank to publish the policy to")?;
    let receiver = inv.inbox.take().ok_or("train: stream: the job's input was taken already, or this call came without one")?;
    let steps = refs.max_steps.unwrap_or(MAX_STREAM_STEPS);
    // A run that continues one whose versions are served already serves from the start; any other offers the policy
    // it starts from as its first version, which its first rollouts are drawn from.
    let served_already = bank.identity(&refs.publish_adapter).is_some();
    let run = run_id(request);
    let mut ft = job::fine_tune_settings(request, inputs.base_dir, None, files)?.adapter_id(refs.publish_adapter.as_str()).publish_every(refs.publish_every).publish_initial(!served_already).steps(steps).run_lineage(run.as_str(), epoch);
    if let Some(path) = &inputs.staged.start_adapter {
        ft = ft.continue_from(path);
    }
    let cancel = inv.cancel.clone();
    let stream = Arc::new(ft.open_rollout_stream(refs.step_tokens, refs.queue_tokens, refs.max_staleness, refs.top_k, refs.support, move || !cancel.is_cancelled(), inv.lease.clone()).map_err(job::trainer_failure)?);
    let ft = ft.rollout_stream(&stream);
    let published_at_start = bank.identity(&refs.publish_adapter).map_or(0, |identity| identity.version);
    stream.set_version(published_at_start);
    progress(Progress::event(0, steps, json!({"type": "stream", "run": run, "epoch": epoch, "tokenizer_digest": stream.tokenizer_digest(), "publish_adapter": refs.publish_adapter, "version": published_at_start, "step_tokens": refs.step_tokens, "queue_tokens": refs.queue_tokens, "max_staleness": refs.max_staleness})));

    let done = Arc::new(AtomicBool::new(false));
    let listener = {
        let (stream, done) = (Arc::clone(&stream), Arc::clone(&done));
        std::thread::Builder::new().name("brain-train-input".into()).spawn(move || serve_inputs(&receiver, &stream, epoch, &done)).map_err(|e| format!("train: stream: cannot start the input thread: {e}"))?
    };
    // Frames the publisher made, emitted with the step they belong to; the keys trained and not yet in a saved state.
    let published: RefCell<Vec<Value>> = RefCell::new(Vec::new());
    let last_published: RefCell<Option<u32>> = RefCell::new(None);
    let mut open: Vec<String> = Vec::new();
    let emit = |frames: Vec<Value>, open: &mut Vec<String>, progress: &mut dyn FnMut(Progress)| {
        for frame in frames {
            match frame["type"].as_str() {
                Some("applied" | "dropped") => open.extend(groups_of(&frame)),
                Some("committed") => {
                    let committed = groups_of(&frame);
                    open.retain(|g| !committed.contains(g));
                }
                _ => {}
            }
            progress(Progress::event(0, steps, frame));
        }
    };
    let publish_id = refs.publish_adapter.clone();
    let mut publish = |step: u32, path: &std::path::Path| -> std::io::Result<()> {
        // A retained version is kept in the artifact store too, so a client can continue a later run from it.
        let artifact = match (&env.store, refs.retain_versions > 0) {
            (Some(store), true) => Some(store.publish(path).map_err(|e| std::io::Error::other(job::store_failure(&e)))?.0.hex()),
            _ => None,
        };
        let (identity, version_id) = bank.publish_retaining(&publish_id, path, refs.retain_versions).map_err(std::io::Error::other)?;
        std::fs::remove_file(path).ok();
        // The bank read the identity from the card this run wrote: it must be this run's, at this step.
        let lineage = identity.lineage.as_ref().filter(|l| l.run == run && l.epoch == epoch && l.update == step).ok_or_else(|| std::io::Error::other(format!("publishing adapter {publish_id}: the served release names {:?}, not run {run:?} epoch {epoch} update {step}", identity.lineage)))?;
        stream.set_version(identity.version);
        let mut frame = json!({"type": "published", "version": identity.version, "digest": identity.digest, "step": step, "lineage": lineage});
        if let Some(version_id) = version_id {
            frame["version_id"] = json!(version_id);
        }
        if let Some(artifact) = artifact {
            frame["artifact_digest"] = json!(artifact);
        }
        published.borrow_mut().push(frame);
        *last_published.borrow_mut() = Some(step);
        Ok(())
    };
    let outcome = ft.run_publishing(
        &inv.cancel,
        |step| {
            emit(stream.drain_events(), &mut open, progress);
            emit(std::mem::take(&mut *published.borrow_mut()), &mut open, progress);
            progress(Progress::event(step.step, step.steps, job::step_frame(step, began.elapsed().as_secs_f64())));
        },
        &mut publish,
    );
    done.store(true, Ordering::SeqCst);
    let _ = listener.join();
    // What the run reported after its last step: the state it wrote when it stopped, the groups dropped at the end.
    emit(stream.drain_events(), &mut open, progress);
    let outcome = outcome.map_err(job::trainer_failure)?;
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err("cancelled".to_string());
    }
    let adapter = outcome.adapter.clone().ok_or("train: the run completed with no adapter")?;
    if outcome.steps_completed > 0 && *last_published.borrow() != Some(outcome.steps_completed) {
        publish(outcome.steps_completed, &adapter).map_err(|e| format!("train: {e}"))?;
        // The exported adapter also goes to the artifact store: it was read, not consumed.
    }
    emit(std::mem::take(&mut *published.borrow_mut()), &mut open, progress);
    let trained = Trained { adapter, record: outcome.record.clone(), adapter_sha256: outcome.adapter_digest.clone().unwrap_or_default(), evaluations: Vec::new(), resumed_at: outcome.resumed_at, summary: outcome.report() };
    Ok(Streamed { trained, uncommitted: open })
}

/// The run every version a stream publishes names (`TrainingProvenance::run`): its resume key, which every job that
/// continues the run shares, or for a run without one an id drawn once (`job:<32 hex>`; a key holds no `:`).
fn run_id(request: &Request) -> String {
    match &request.params.resume_key {
        Some(key) => key.clone(),
        None => format!("job:{:016x}{:016x}", brain::random_seed(), brain::random_seed()),
    }
}

/// Answers the job's input until the job is done or its surface has gone.
fn serve_inputs(receiver: &std::sync::mpsc::Receiver<JobInput>, stream: &brain::RolloutStream, epoch: u64, done: &AtomicBool) {
    while !done.load(Ordering::SeqCst) {
        // Input waits in the channel until the run is ready: a resumed run first restores the keys it took.
        if !stream.wait_ready(Duration::from_millis(100)) {
            continue;
        }
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(input) => {
                let reply = answer(stream, epoch, &input.kind, &input.query, &input.body);
                input.answer(reply);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// The job's answer to one input: `records` or `finish`, refused when it names another epoch than the run's.
pub fn answer(stream: &brain::RolloutStream, epoch: u64, kind: &str, query: &Value, body: &[u8]) -> InputReply {
    let refuse = |kind: InputRefusalKind, message: String| Err(InputRefusal::new(kind, message).with_detail(json!({"epoch": epoch})));
    match query.get("epoch") {
        None | Some(Value::Null) => {}
        Some(asked) if asked.as_u64() == Some(epoch) => {}
        Some(asked) => return refuse(InputRefusalKind::Conflict, format!("the run is at epoch {epoch}, and the request names epoch {asked}: a job continued the run from its last saved state since; re-send every group without a committed receipt")),
    }
    let settings = stream.settings();
    match kind {
        "records" => match stream.ingest(body) {
            Ok(admission) => Ok(json!({
                "accepted": admission.accepted,
                "duplicates": admission.duplicates,
                "zero_advantage": admission.zero_advantage,
                "queued_groups": admission.queued_groups,
                "queued_tokens": admission.queued_tokens,
                "allowance": admission.allowance,
                "epoch": epoch,
                "version": stream.version(),
            })),
            Err(brain::StreamRefusal::Invalid(why)) => refuse(InputRefusalKind::Invalid, format!("records: {why}")),
            Err(brain::StreamRefusal::Full { queued_tokens, allowance, offered_tokens }) => Err(InputRefusal::new(
                InputRefusalKind::Busy,
                format!("the queue holds {queued_tokens} of its {} policy tokens and these records add {offered_tokens}; send at most {allowance} more, or retry once the learner has taken a step", settings.queue_tokens),
            )
            .with_detail(json!({"allowance": allowance, "queued_tokens": queued_tokens, "offered_tokens": offered_tokens, "epoch": epoch}))),
            Err(brain::StreamRefusal::Closed) => refuse(InputRefusalKind::Conflict, "the job was told to finish; it takes no more records".to_string()),
            Err(brain::StreamRefusal::Starting) => refuse(InputRefusalKind::Unavailable, "the job is starting; retry".to_string()),
        },
        "finish" => {
            let admission = stream.finish();
            Ok(json!({"finishing": true, "queued_groups": admission.queued_groups, "queued_tokens": admission.queued_tokens, "epoch": epoch}))
        }
        other => refuse(InputRefusalKind::Invalid, format!("a stream job takes records and finish, not {other:?}")),
    }
}
