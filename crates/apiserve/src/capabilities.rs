// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Routes that reach every model class brain serves, not just the ones a chat,
//! embeddings or image dialect has a shape for.
//!
//! - `GET /v1/capabilities` lists every model's every action with its contract:
//!   params, blob inputs and blob outputs. It is what a client is generated from.
//! - `GET /v1/devices` lists the devices the server schedules work on: each one's memory (total, reserved,
//!   in use, free) and, where the host can say, what it is.
//! - `POST /v1/run` runs one action and answers with its outputs and blobs.
//! - `POST /v1/jobs` starts one in the background, for work too long to hold a
//!   request open (video, music, 3D); `GET /v1/jobs/{id}` reports its progress,
//!   `GET /v1/jobs/{id}/result` returns it when it is done and
//!   `DELETE /v1/jobs/{id}` cancels it.
//! - `GET /v1/jobs/{id}/events` follows one as a server-sent event stream: each step
//!   tick (`progress`) and each structured frame the action emits (`frame`, such as a
//!   training step's loss and statistics) as it happens, then one terminal `done`
//!   event carrying the job's final status. The last [`MAX_EVENTS_PER_JOB`] frames are
//!   kept, so a client that connects late, or reconnects with `Last-Event-ID`, replays
//!   what it missed.
//!
//! A call is `{"model", "action", "params", "blobs": {name: {"media", "data",
//! "meta"}}}` with `data` base64. It is validated against the action's own
//! [`ActionSpec`], as a caller sees it (never offered, and never accepting, a
//! param the host resolves such as a weights path), before anything is submitted, and then goes through the same
//! [`bridge`] every dialect route does, so the surface's hooks, admission and
//! cancel-on-disconnect apply to it unchanged.
//!
//! A job belongs to the caller that started it: it is looked up by the
//! [`Authenticator::scope`](crate::Authenticator::scope) of the caller and its
//! id, so another caller gets "no such job" for an id that exists. Jobs are held
//! in memory, bounded in number per caller and in the bytes of finished results
//! kept, and a finished one is forgotten after [`RESULT_TTL`]. A process restart
//! forgets them all.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use capability::{ActionSpec, Blob, Invocation, Media, Outcome};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{oneshot, watch};
use uuid::Uuid;

use crate::auth::CallerExt;
use crate::bridge::{self, EventStream, StreamMsg};
use crate::error::ApiError;
use crate::state::AppState;

/// The largest request body these routes accept. Media travels inline as
/// base64, so this is larger than the dialects' 8 MiB, and still a ceiling.
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// Jobs one caller may hold at once, running or finished and not yet forgotten.
pub const MAX_JOBS_PER_SCOPE: usize = 32;
/// Jobs the surface holds at once across every caller.
pub const MAX_JOBS: usize = 1024;
/// How long a finished job is kept before it is forgotten.
pub const RESULT_TTL: Duration = Duration::from_secs(60 * 60);
/// The most bytes of finished results kept; the oldest are forgotten first.
pub const MAX_RETAINED_BYTES: usize = 1024 * 1024 * 1024;
/// The frames of one job kept for [`events`]; the oldest are forgotten first.
pub const MAX_EVENTS_PER_JOB: usize = 4096;
/// Event streams open on one job at once.
pub const MAX_EVENT_STREAMS_PER_JOB: usize = 8;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/capabilities", get(list))
        .route("/v1/devices", get(devices))
        .route("/v1/run", post(run))
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/:id", get(job_status).delete(cancel_job))
        .route("/v1/jobs/:id/result", get(job_result))
        .route("/v1/jobs/:id/events", get(job_events))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
}

// ------------------------------------------------------------------ the call

/// `deny_unknown_fields`: a field this route does not define is a client bug (or a probe), and ignoring it would
/// hide that the caller believes it has set something.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CallBody {
    model: String,
    action: String,
    #[serde(default)]
    params: Value,
    #[serde(default)]
    blobs: BTreeMap<String, BlobWire>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BlobWire {
    media: String,
    data: String,
    #[serde(default)]
    meta: Value,
}

/// Reads and validates a call against the action's own spec, so nothing invalid
/// reaches the hooks or the executor.
fn parse_call(state: &AppState, body: &Bytes) -> Result<(String, String, Invocation), ApiError> {
    let provider = state.provider;
    let call: CallBody = serde_json::from_slice(body).map_err(|e| ApiError::invalid_request(provider, format!("invalid JSON body: {e}")))?;

    let manifests = state.exec.manifests();
    let manifest = manifests.iter().find(|m| m.model == call.model).ok_or_else(|| ApiError::model_not_found(provider, &call.model))?;
    let spec: &ActionSpec = manifest
        .actions
        .iter()
        .find(|a| a.name == call.action)
        .ok_or_else(|| ApiError::not_found(provider, format!("model '{}' has no action '{}'", call.model, call.action)))?;

    let mut inv = Invocation::new();
    inv.params = match call.params {
        Value::Null => json!({}),
        Value::Object(map) => Value::Object(map),
        _ => return Err(ApiError::invalid_request(provider, "'params' must be an object")),
    };
    for (name, wire) in call.blobs {
        let media = Media::parse(&wire.media).ok_or_else(|| ApiError::invalid_request(provider, format!("blob '{name}': unknown media '{}'", wire.media)))?;
        let bytes = crate::b64::decode(&wire.data).map_err(|e| ApiError::invalid_request(provider, format!("blob '{name}': {e}")))?;
        let meta = if wire.meta.is_null() { json!({}) } else { wire.meta };
        inv.blobs.insert(name, Blob::new(media, bytes).with_meta(meta));
    }
    // A param the host resolves (a weights path, a dataset folder) is refused
    // as unknown, so a remote caller can never name a file on this machine.
    let inv = spec.validate_served(inv).map_err(|e| ApiError::invalid_request(provider, e))?;
    // What the spec cannot say - that a base exists, that a dataset is in the store - the model is asked at the door,
    // so a call that can never run is a 400 here and not a job that is accepted and then fails.
    state.exec.check_job(&call.model, &call.action, &inv).map_err(|e| ApiError::invalid_request(provider, e))?;
    Ok((call.model, call.action, inv))
}

fn blob_json(blob: &Blob) -> Value {
    json!({
        "media": blob.media.name(),
        "data": events::base64::encode(&blob.bytes),
        "meta": blob.meta,
        "bytes": blob.bytes.len(),
    })
}

fn outcome_json(outcome: &Outcome) -> Value {
    let blobs: BTreeMap<&String, Value> = outcome.blobs.iter().map(|(name, blob)| (name, blob_json(blob))).collect();
    json!({ "outputs": outcome.outputs, "blobs": blobs })
}

/// The bytes a finished outcome occupies, for the retention budget.
fn outcome_bytes(outcome: &Outcome) -> usize {
    outcome.blobs.values().map(|b| b.bytes.len()).sum::<usize>() + outcome.outputs.to_string().len()
}

// ------------------------------------------------------------------ catalogue and run

async fn list(State(state): State<AppState>) -> Json<Value> {
    let manifests = state.exec.manifests();
    let data: Vec<Value> = manifests
        .iter()
        .filter(|manifest| state.lists(&manifest.model))
        .flat_map(|manifest| {
            // What a caller may see: nothing the host resolves for itself.
            manifest.clone().for_serving().actions.into_iter().map(move |action| {
                let params: Vec<Value> = action
                    .params
                    .iter()
                    .map(|p| {
                        let mut entry = json!({"name": p.name, "type": p.ty.name(), "required": p.required, "help": p.help});
                        if let capability::ParamType::Enum(values) = &p.ty {
                            entry["values"] = json!(values);
                        }
                        if let Some(default) = &p.default {
                            entry["default"] = default.clone();
                        }
                        if let Some(min) = p.min {
                            entry["min"] = json!(min);
                        }
                        if let Some(max) = p.max {
                            entry["max"] = json!(max);
                        }
                        entry
                    })
                    .collect();
                json!({
                    "model": manifest.model,
                    "action": action.name,
                    "summary": action.summary,
                    "streaming": action.streaming,
                    "params": params,
                    "inputs": action.inputs.iter().map(|b| json!({"name": b.name, "media": b.media.name(), "required": b.required})).collect::<Vec<_>>(),
                    "outputs": action.outputs.iter().map(|b| json!({"name": b.name, "media": b.media.name()})).collect::<Vec<_>>(),
                })
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

/// `GET /v1/devices`: the devices the executor schedules on, each with its memory as the scheduler accounts it,
/// merged with what the host's [`crate::DeviceInventory`] says each is. Nothing here names a path or an address of
/// the machine: an id is `cpu`, `gpu<i>` or `npu<i>`, the names `--device` takes.
async fn devices(State(state): State<AppState>) -> Json<Value> {
    let report = state.exec.residency();
    let mut budgets = report.budgets;
    budgets.sort_by_key(|b| match b.device {
        residency::Device::Cpu => (0, 0),
        residency::Device::Gpu(i) => (1, i),
        residency::Device::Npu(i) => (2, i),
    });
    let data: Vec<Value> = budgets
        .iter()
        .map(|b| {
            let (id, kind) = match b.device {
                residency::Device::Cpu => ("cpu".to_string(), "cpu"),
                residency::Device::Gpu(i) => (format!("gpu{i}"), "gpu"),
                residency::Device::Npu(i) => (format!("npu{i}"), "npu"),
            };
            let mut entry = json!({
                "id": id,
                "kind": kind,
                "memory": {
                    "total_bytes": b.total,
                    "reserved_bytes": b.reserved,
                    "used_bytes": b.used,
                    "free_bytes": b.total.saturating_sub(b.reserved).saturating_sub(b.used),
                },
            });
            if let (Some(inventory), Some(fields)) = (&state.devices, entry.as_object_mut()) {
                fields.extend(inventory.device(b.device));
            }
            entry
        })
        .collect();
    let host = state.devices.as_ref().map(|inventory| Value::Object(inventory.host())).unwrap_or_else(|| json!({}));
    Json(json!({ "object": "list", "host": host, "data": data }))
}

async fn run(State(state): State<AppState>, caller: CallerExt, body: Bytes) -> Response {
    let state = state.scoped(caller);
    let (model, action, inv) = match parse_call(&state, &body) {
        Ok(call) => call,
        Err(e) => return e.into_response(),
    };
    match bridge::submit(&state, &model, &action, inv).await {
        Ok(outcome) => Json(outcome_json(&outcome)).into_response(),
        Err(e) => e.into_response(),
    }
}

// ------------------------------------------------------------------ jobs

enum Phase {
    Running { step: u32, total: u32 },
    Succeeded { outcome: Arc<Outcome>, bytes: usize },
    Failed(ApiError),
    Cancelled,
}

struct Entry {
    scope: String,
    finished: Option<Instant>,
    phase: Phase,
    cancel: Option<oneshot::Sender<()>>,
    log: EventLog,
    /// Carries the sequence number of the newest frame, so a follower waits for the next one.
    wake: watch::Sender<u64>,
    /// Event streams open on this job.
    followers: Arc<AtomicUsize>,
}

/// One frame of a job's event stream: what happened, as the SSE `event:` name and its JSON.
#[derive(Clone, Debug, PartialEq)]
struct Frame {
    seq: u64,
    kind: &'static str,
    data: Value,
}

/// The newest [`MAX_EVENTS_PER_JOB`] frames of one job, numbered from 1.
#[derive(Default)]
struct EventLog {
    last: u64,
    frames: VecDeque<Frame>,
}

impl EventLog {
    fn push(&mut self, kind: &'static str, data: Value) -> u64 {
        self.last += 1;
        if self.frames.len() == MAX_EVENTS_PER_JOB {
            self.frames.pop_front();
        }
        self.frames.push_back(Frame { seq: self.last, kind, data });
        self.last
    }

    fn after(&self, seq: u64) -> Vec<Frame> {
        self.frames.iter().filter(|f| f.seq > seq).cloned().collect()
    }
}

/// The surface's background jobs. See the module doc for what it promises.
#[derive(Default)]
pub struct AsyncJobs {
    entries: Mutex<HashMap<Uuid, Entry>>,
}

impl AsyncJobs {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Uuid, Entry>> {
        // Plain data mutated by infallible assignment: a poisoned lock is still
        // consistent, and taking the surface down over it would be worse.
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Forgets finished jobs past their TTL, then the oldest finished ones
    /// while the retained results exceed [`MAX_RETAINED_BYTES`].
    fn reap(entries: &mut HashMap<Uuid, Entry>, now: Instant) {
        entries.retain(|_, e| e.finished.is_none_or(|at| now.duration_since(at) < RESULT_TTL));
        let mut retained: Vec<(Instant, Uuid, usize)> = entries
            .iter()
            .filter_map(|(id, e)| match (&e.phase, e.finished) {
                (Phase::Succeeded { bytes, .. }, Some(at)) => Some((at, *id, *bytes)),
                _ => None,
            })
            .collect();
        let mut total: usize = retained.iter().map(|(_, _, b)| b).sum();
        retained.sort();
        for (_, id, bytes) in retained {
            if total <= MAX_RETAINED_BYTES {
                break;
            }
            entries.remove(&id);
            total -= bytes;
        }
    }

    /// Makes room for one more job of `scope`, or says why there is none.
    fn open(&self, scope: &str, cancel: oneshot::Sender<()>) -> Result<Uuid, &'static str> {
        let mut entries = self.lock();
        Self::reap(&mut entries, Instant::now());
        if entries.len() >= MAX_JOBS {
            return Err("the surface is holding as many jobs as it can");
        }
        if entries.values().filter(|e| e.scope == scope).count() >= MAX_JOBS_PER_SCOPE {
            return Err("too many jobs: fetch or cancel some before starting another");
        }
        let id = Uuid::new_v4();
        entries.insert(
            id,
            Entry {
                scope: scope.to_string(),
                finished: None,
                phase: Phase::Running { step: 0, total: 0 },
                cancel: Some(cancel),
                log: EventLog::default(),
                wake: watch::channel(0).0,
                followers: Arc::default(),
            },
        );
        Ok(id)
    }

    fn discard(&self, id: Uuid) {
        self.lock().remove(&id);
    }

    fn progress(&self, id: Uuid, step: u32, total: u32) {
        if let Some(entry) = self.lock().get_mut(&id).filter(|e| matches!(e.phase, Phase::Running { .. })) {
            entry.phase = Phase::Running { step, total };
            entry.record("progress", json!({"step": step, "total": total}));
        }
    }

    /// A structured frame the action emitted (see [`Progress::event`](capability::Progress::event)).
    fn frame(&self, id: Uuid, data: Value) {
        if let Some(entry) = self.lock().get_mut(&id).filter(|e| matches!(e.phase, Phase::Running { .. })) {
            entry.record("frame", data);
        }
    }

    /// Moves a running job to its final phase. A job that already finished
    /// (cancelled by its owner while the model was still winding down) keeps
    /// the phase it has.
    fn finish(&self, id: Uuid, phase: Phase) {
        if let Some(entry) = self.lock().get_mut(&id) {
            if matches!(entry.phase, Phase::Running { .. }) {
                entry.phase = phase;
                entry.finished = Some(Instant::now());
                entry.cancel = None;
                entry.record_end();
            }
        }
    }

    /// Cancel a running job, or forget a finished one and its result: a caller
    /// that has read what it needs frees its slot at once instead of waiting
    /// out the result's time to live.
    fn cancel(&self, scope: &str, id: Uuid) -> Option<&'static str> {
        let mut entries = self.lock();
        let entry = entries.get_mut(&id).filter(|e| e.scope == scope)?;
        if matches!(entry.phase, Phase::Running { .. }) {
            if let Some(cancel) = entry.cancel.take() {
                let _ = cancel.send(());
            }
            entry.phase = Phase::Cancelled;
            entry.finished = Some(Instant::now());
            entry.record_end();
            return Some(phase_name(&entry.phase));
        }
        entries.remove(&id);
        Some("deleted")
    }

    /// The frames of job `id` after `seq`, for `scope`; with a receiver that wakes when there are more, and a
    /// guard counting this follower. `None` when the job does not exist for `scope` or has too many followers.
    fn follow(&self, scope: &str, id: Uuid, seq: u64) -> Option<Result<Follow, ()>> {
        let entries = self.lock();
        let entry = entries.get(&id).filter(|e| e.scope == scope)?;
        if entry.followers.fetch_add(1, Ordering::SeqCst) >= MAX_EVENT_STREAMS_PER_JOB {
            entry.followers.fetch_sub(1, Ordering::SeqCst);
            return Some(Err(()));
        }
        Some(Ok(Follow { scope: scope.to_string(), id, seq, wake: entry.wake.subscribe(), pending: VecDeque::new(), ended: false, count: Arc::clone(&entry.followers) }))
    }

    /// The frames of job `id` after `seq`, and whether the job has ended (so no more will come); `None` once the
    /// job is forgotten.
    fn frames_after(&self, scope: &str, id: Uuid, seq: u64) -> Option<(Vec<Frame>, bool)> {
        self.lock().get(&id).filter(|e| e.scope == scope).map(|e| (e.log.after(seq), !matches!(e.phase, Phase::Running { .. })))
    }

    /// What `scope` is allowed to see of job `id`.
    fn view(&self, scope: &str, id: Uuid) -> Option<View> {
        let entries = self.lock();
        let entry = entries.get(&id).filter(|e| e.scope == scope)?;
        Some(match &entry.phase {
            Phase::Running { step, total } => View::Running { step: *step, total: *total },
            Phase::Succeeded { outcome, .. } => View::Succeeded(Arc::clone(outcome)),
            Phase::Failed(error) => View::Failed(error.clone()),
            Phase::Cancelled => View::Cancelled,
        })
    }
}

impl Entry {
    fn record(&mut self, kind: &'static str, data: Value) {
        let seq = self.log.push(kind, data);
        self.wake.send_replace(seq);
    }

    /// The terminal frame, from the phase the job has just entered.
    fn record_end(&mut self) {
        let mut data = json!({"status": phase_name(&self.phase)});
        if let Phase::Failed(error) = &self.phase {
            data["error"] = json!({"message": error.message, "code": error.kind.status().as_u16()});
        }
        self.record("done", data);
    }
}

fn phase_name(phase: &Phase) -> &'static str {
    match phase {
        Phase::Running { .. } => "running",
        Phase::Succeeded { .. } => "succeeded",
        Phase::Failed(_) => "failed",
        Phase::Cancelled => "cancelled",
    }
}

enum View {
    Running { step: u32, total: u32 },
    Succeeded(Arc<Outcome>),
    Failed(ApiError),
    Cancelled,
}

fn no_such_job(state: &AppState) -> Response {
    ApiError::not_found(state.provider, "no such job").into_response()
}

fn parse_id(raw: &str) -> Option<Uuid> {
    Uuid::parse_str(raw).ok()
}

async fn create_job(State(state): State<AppState>, caller: CallerExt, body: Bytes) -> Response {
    let state = state.scoped(caller);
    let (model, action, inv) = match parse_call(&state, &body) {
        Ok(call) => call,
        Err(e) => return e.into_response(),
    };
    let scope = state.scope();
    let (cancel_tx, cancel_rx) = oneshot::channel();
    let id = match state.async_jobs.open(&scope, cancel_tx) {
        Ok(id) => id,
        Err(why) => return ApiError::rate_limited(state.provider, why).into_response(),
    };
    // Begun here, not in the background: a caller the hooks refuse learns it from
    // this response, and the model is never queued for them.
    let stream = match bridge::stream_progress(&state, &model, &action, inv).await {
        Ok(stream) => stream,
        Err(e) => {
            state.async_jobs.discard(id);
            return e.into_response();
        }
    };
    tokio::spawn(drive(Arc::clone(&state.async_jobs), state.provider, id, stream, cancel_rx));

    let mut response = (StatusCode::ACCEPTED, Json(json!({"id": id.to_string(), "status": "running"}))).into_response();
    if let Ok(location) = HeaderValue::from_str(&format!("/v1/jobs/{id}")) {
        response.headers_mut().insert(header::LOCATION, location);
    }
    response
}

/// Follows one job's stream to its end, recording progress and the result.
/// Dropping the stream, on cancel, cancels the running action.
async fn drive(jobs: Arc<AsyncJobs>, provider: crate::Provider, id: Uuid, mut stream: EventStream, mut cancel: oneshot::Receiver<()>) {
    loop {
        tokio::select! {
            // A cancel, or the job's record having gone: either way nobody can
            // collect the result any more.
            _ = &mut cancel => {
                jobs.finish(id, Phase::Cancelled);
                return;
            }
            msg = stream.next() => match msg {
                Some(StreamMsg::Progress(step, total)) => jobs.progress(id, step, total),
                Some(StreamMsg::Event(frame)) => jobs.frame(id, frame),
                Some(StreamMsg::Delta(_) | StreamMsg::Fetching(_)) => {}
                Some(StreamMsg::Done(outcome)) => {
                    let bytes = outcome_bytes(&outcome);
                    jobs.finish(id, Phase::Succeeded { outcome: Arc::new(outcome), bytes });
                    return;
                }
                Some(StreamMsg::Err(error)) => {
                    jobs.finish(id, Phase::Failed(error));
                    return;
                }
                None => {
                    jobs.finish(id, Phase::Failed(ApiError::internal(provider, "the job ended without a result")));
                    return;
                }
            }
        }
    }
}

async fn job_status(State(state): State<AppState>, caller: CallerExt, Path(raw): Path<String>) -> Response {
    let state = state.scoped(caller);
    let Some(id) = parse_id(&raw) else { return no_such_job(&state) };
    match state.async_jobs.view(&state.scope(), id) {
        None => no_such_job(&state),
        Some(View::Running { step, total }) => Json(json!({"id": id.to_string(), "status": "running", "progress": {"step": step, "total": total}})).into_response(),
        Some(View::Succeeded(_)) => Json(json!({"id": id.to_string(), "status": "succeeded"})).into_response(),
        Some(View::Failed(error)) => Json(json!({"id": id.to_string(), "status": "failed", "error": {"message": error.message, "code": error.kind.status().as_u16()}})).into_response(),
        Some(View::Cancelled) => Json(json!({"id": id.to_string(), "status": "cancelled"})).into_response(),
    }
}

async fn job_result(State(state): State<AppState>, caller: CallerExt, Path(raw): Path<String>) -> Response {
    let state = state.scoped(caller);
    let Some(id) = parse_id(&raw) else { return no_such_job(&state) };
    match state.async_jobs.view(&state.scope(), id) {
        None => no_such_job(&state),
        Some(View::Succeeded(outcome)) => Json(outcome_json(&outcome)).into_response(),
        Some(View::Running { .. }) => (StatusCode::CONFLICT, Json(json!({"id": id.to_string(), "status": "running", "error": {"message": "the job has not finished"}}))).into_response(),
        Some(View::Failed(error)) => error.into_response(),
        Some(View::Cancelled) => (StatusCode::GONE, Json(json!({"id": id.to_string(), "status": "cancelled", "error": {"message": "the job was cancelled"}}))).into_response(),
    }
}

/// One client following a job's event stream: where it has got to, and what it has still to send.
struct Follow {
    scope: String,
    id: Uuid,
    seq: u64,
    wake: watch::Receiver<u64>,
    pending: VecDeque<Frame>,
    /// The terminal frame has been sent, or the job is gone.
    ended: bool,
    count: Arc<AtomicUsize>,
}

impl Drop for Follow {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Follow {
    /// The next frame, waiting for the job to make one; `None` when the stream is over.
    async fn next(&mut self, jobs: &AsyncJobs) -> Option<Frame> {
        loop {
            if let Some(frame) = self.pending.pop_front() {
                self.seq = frame.seq;
                self.ended = frame.kind == "done";
                return Some(frame);
            }
            if self.ended {
                return None;
            }
            // Mark the current value seen BEFORE reading, so a frame recorded in between wakes the wait below.
            self.wake.borrow_and_update();
            let (frames, finished) = jobs.frames_after(&self.scope, self.id, self.seq)?;
            self.pending = frames.into();
            if self.pending.is_empty() {
                if finished {
                    // The client already has the terminal frame.
                    return None;
                }
                if self.wake.changed().await.is_err() {
                    // The job was forgotten while it was followed.
                    return None;
                }
            }
        }
    }
}

/// `GET /v1/jobs/{id}/events`: the job's frames as server-sent events, from the start or from `Last-Event-ID`.
async fn job_events(State(state): State<AppState>, caller: CallerExt, Path(raw): Path<String>, headers: axum::http::HeaderMap) -> Response {
    let state = state.scoped(caller);
    let Some(id) = parse_id(&raw) else { return no_such_job(&state) };
    // A malformed Last-Event-ID is a client bug, not a reason to replay from the start.
    let seq = match headers.get("last-event-id") {
        None => 0,
        Some(value) => match value.to_str().ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            Some(seq) => seq,
            None => return ApiError::invalid_request(state.provider, "Last-Event-ID must be the number of the last event received").into_response(),
        },
    };
    let jobs = Arc::clone(&state.async_jobs);
    let follow = match jobs.follow(&state.scope(), id, seq) {
        None => return no_such_job(&state),
        Some(Err(())) => return ApiError::rate_limited(state.provider, "too many event streams are open on this job").into_response(),
        Some(Ok(follow)) => follow,
    };
    let events = futures::stream::unfold((follow, jobs), |(mut follow, jobs)| async move {
        let frame = follow.next(&jobs).await?;
        let event = Event::default().id(frame.seq.to_string()).event(frame.kind).json_data(&frame.data).unwrap_or_else(|_| Event::default().event("error"));
        Some((Ok::<_, std::convert::Infallible>(event), (follow, jobs)))
    });
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

async fn cancel_job(State(state): State<AppState>, caller: CallerExt, Path(raw): Path<String>) -> Response {
    let state = state.scoped(caller);
    let Some(id) = parse_id(&raw) else { return no_such_job(&state) };
    match state.async_jobs.cancel(&state.scope(), id) {
        None => no_such_job(&state),
        Some(status) => Json(json!({"id": id.to_string(), "status": status})).into_response(),
    }
}
