// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Routes that reach every model class brain serves, not just the ones a chat,
//! embeddings or image dialect has a shape for.
//!
//! - `GET /v1/capabilities` lists every model's every action with its contract:
//!   params, blob inputs and blob outputs. It is what a client is generated from.
//! - `POST /v1/run` runs one action and answers with its outputs and blobs.
//! - `POST /v1/jobs` starts one in the background, for work too long to hold a
//!   request open (video, music, 3D); `GET /v1/jobs/{id}` reports its progress,
//!   `GET /v1/jobs/{id}/result` returns it when it is done and
//!   `DELETE /v1/jobs/{id}` cancels it.
//!
//! A call is `{"model", "action", "params", "blobs": {name: {"media", "data",
//! "meta"}}}` with `data` base64. It is validated against the action's own
//! [`ActionSpec`] before anything is submitted, and then goes through the same
//! [`bridge`] every dialect route does, so the surface's hooks, admission and
//! cancel-on-disconnect apply to it unchanged.
//!
//! A job belongs to the caller that started it: it is looked up by the
//! [`Authenticator::scope`](crate::Authenticator::scope) of the caller and its
//! id, so another caller gets "no such job" for an id that exists. Jobs are held
//! in memory, bounded in number per caller and in the bytes of finished results
//! kept, and a finished one is forgotten after [`RESULT_TTL`]. A process restart
//! forgets them all.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use capability::{ActionSpec, Blob, Invocation, Media, Outcome};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::oneshot;
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

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/capabilities", get(list))
        .route("/v1/run", post(run))
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/:id", get(job_status).delete(cancel_job))
        .route("/v1/jobs/:id/result", get(job_result))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
}

// ------------------------------------------------------------------ the call

#[derive(Deserialize)]
struct CallBody {
    model: String,
    action: String,
    #[serde(default)]
    params: Value,
    #[serde(default)]
    blobs: BTreeMap<String, BlobWire>,
}

#[derive(Deserialize)]
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
    let inv = spec.validate(inv).map_err(|e| ApiError::invalid_request(provider, e))?;
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
            manifest.actions.iter().map(move |action| {
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
        entries.insert(id, Entry { scope: scope.to_string(), finished: None, phase: Phase::Running { step: 0, total: 0 }, cancel: Some(cancel) });
        Ok(id)
    }

    fn discard(&self, id: Uuid) {
        self.lock().remove(&id);
    }

    fn progress(&self, id: Uuid, step: u32, total: u32) {
        if let Some(Entry { phase: phase @ Phase::Running { .. }, .. }) = self.lock().get_mut(&id) {
            *phase = Phase::Running { step, total };
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
            }
        }
    }

    fn cancel(&self, scope: &str, id: Uuid) -> Option<&'static str> {
        let mut entries = self.lock();
        let entry = entries.get_mut(&id).filter(|e| e.scope == scope)?;
        if matches!(entry.phase, Phase::Running { .. }) {
            if let Some(cancel) = entry.cancel.take() {
                let _ = cancel.send(());
            }
            entry.phase = Phase::Cancelled;
            entry.finished = Some(Instant::now());
        }
        Some(phase_name(&entry.phase))
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

fn scope_of(state: &AppState) -> String {
    state.caller.as_ref().map(|p| state.authenticator.scope(p)).unwrap_or_default()
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
    let scope = scope_of(&state);
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
                Some(StreamMsg::Delta(_) | StreamMsg::Event(_) | StreamMsg::Fetching(_)) => {}
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
    match state.async_jobs.view(&scope_of(&state), id) {
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
    match state.async_jobs.view(&scope_of(&state), id) {
        None => no_such_job(&state),
        Some(View::Succeeded(outcome)) => Json(outcome_json(&outcome)).into_response(),
        Some(View::Running { .. }) => (StatusCode::CONFLICT, Json(json!({"id": id.to_string(), "status": "running", "error": {"message": "the job has not finished"}}))).into_response(),
        Some(View::Failed(error)) => error.into_response(),
        Some(View::Cancelled) => (StatusCode::GONE, Json(json!({"id": id.to_string(), "status": "cancelled", "error": {"message": "the job was cancelled"}}))).into_response(),
    }
}

async fn cancel_job(State(state): State<AppState>, caller: CallerExt, Path(raw): Path<String>) -> Response {
    let state = state.scoped(caller);
    let Some(id) = parse_id(&raw) else { return no_such_job(&state) };
    match state.async_jobs.cancel(&scope_of(&state), id) {
        None => no_such_job(&state),
        Some(status) => Json(json!({"id": id.to_string(), "status": status})).into_response(),
    }
}
