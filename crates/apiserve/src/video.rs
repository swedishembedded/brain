// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Video generation and editing on the OpenRouter `/videos` contract.
//!
//! - `POST /videos` validates a request against the chosen action's
//!   [`VideoContract`] and starts it as a background job on the same executor
//!   path every other route uses (hooks, admission, cancellation unchanged).
//!   An `Idempotency-Key` header names the job: the same key from the same
//!   caller returns the job it already started instead of rendering twice.
//! - `GET /videos/{id}` reports `pending`, `in_progress` (with
//!   `progress {step,total}`), `completed` (with `unsigned_urls` and `usage`),
//!   `failed`, `cancelled` or `expired`.
//! - `GET /videos/{id}/content?index=0` streams the MP4, honouring `Range`.
//! - `GET /videos/models` lists every video model with what it accepts.
//! - `DELETE /videos/{id}` (an extension) cancels a running job, or forgets a
//!   finished one and its file.
//!
//! A finished outcome is encoded ONCE into an AV1/Opus MP4 in this server's
//! spool directory and its frames are dropped: a clip's raw frames are four
//! bytes per channel (a 5 s 720p clip is over a gigabyte), its file a few
//! megabytes. The spool is private to this process, bounded in bytes (the
//! oldest finished clip is expired first) and in age ([`RESULT_TTL`]), and a
//! file is removed with its job. Jobs live in memory: a restart forgets them,
//! and a client that needs durability keeps its own record and re-submits
//! with the same `Idempotency-Key`.
//!
//! Inputs travel inline: an edit's `input_video` is a `data:` URL of an MP4 or
//! WebM (AV1), decoded at the boundary, and `frame_images` and
//! `input_references` reach the action that declares them
//! ([`crate::video_inputs`]). A remote URL is refused; the server never
//! fetches on a caller's behalf.
//!
//! Swedish Embedded AB implements solutions for serving generative video
//! behind standard APIs for its clients. If your team needs expertise in
//! media-serving infrastructure then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use capability::video::VideoRequest;
use capability::{ActionSpec, Invocation, Manifest, Media, Outcome};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::auth::CallerExt;
use crate::bridge::{self, EventStream, StreamMsg};
use crate::error::ApiError;
use crate::state::AppState;
use crate::video_inputs::{self, JobInputs, Wanted};

/// The largest request body: an edit carries its clip inline as base64.
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// Video jobs one caller may hold at once, running or finished.
pub const MAX_JOBS_PER_SCOPE: usize = 32;
/// Video jobs the surface holds at once across every caller.
pub const MAX_JOBS: usize = 1024;
/// How long a finished job and its file are kept.
pub const RESULT_TTL: Duration = Duration::from_secs(60 * 60);
/// The most bytes of finished clips the spool keeps by default.
pub const DEFAULT_SPOOL_BYTES: u64 = 20 * 1024 * 1024 * 1024;
/// The chunk the content route streams the file in.
const CHUNK: usize = 256 * 1024;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/videos", post(create))
        .route("/v1/videos", post(create))
        .route("/videos/models", get(models))
        .route("/v1/videos/models", get(models))
        .route("/videos/:id", get(status).delete(remove))
        .route("/v1/videos/:id", get(status).delete(remove))
        .route("/videos/:id/content", get(content))
        .route("/v1/videos/:id/content", get(content))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
}

// ------------------------------------------------------------------ request

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateBody {
    model: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    resolution: Option<String>,
    #[serde(default)]
    aspect_ratio: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    generate_audio: Option<bool>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    frame_images: Vec<video_inputs::FrameImage>,
    #[serde(default)]
    input_references: Vec<video_inputs::Reference>,
    #[serde(default)]
    input_video: Option<VideoUrl>,
    #[serde(default)]
    callback_url: Option<String>,
    #[serde(default)]
    provider: Option<ProviderBody>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VideoUrl {
    #[serde(rename = "type")]
    kind: String,
    video_url: UrlBody,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UrlBody {
    url: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderBody {
    #[serde(default)]
    options: HashMap<String, Value>,
}

/// `provider.options.brain`: which action, and its own extra params.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct BrainOptions {
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    params: Map<String, Value>,
}

/// The action a model renders a `/videos` request with: the one named, which
/// must declare a [`VideoContract`] and take the request's frame images and
/// references, else the model's first generation action that takes them.
fn video_action<'m>(manifest: &'m Manifest, named: Option<&str>, wanted: &Wanted) -> Result<&'m ActionSpec, String> {
    let model = &manifest.model;
    match named {
        Some(name) => {
            let spec = manifest.actions.iter().find(|a| a.name == name && a.video.is_some()).ok_or_else(|| format!("{model} has no video action '{name}'"))?;
            match wanted.unmet(spec.video.as_ref().expect("filtered on video")) {
                Some(why) => Err(format!("{model} {name} {why}")),
                None => Ok(spec),
            }
        }
        None => {
            let generation: Vec<&ActionSpec> = manifest.actions.iter().filter(|a| a.video.as_ref().is_some_and(|c| c.clip_input.is_none())).collect();
            let first = generation.first().ok_or_else(|| format!("{model} has no text-to-video action; name an edit with provider.options.brain.action"))?;
            generation.iter().find(|a| wanted.unmet(a.video.as_ref().expect("filtered on video")).is_none()).copied().ok_or_else(|| {
                let why = wanted.unmet(first.video.as_ref().expect("filtered on video")).unwrap_or_default();
                format!("{model} {why}")
            })
        }
    }
}

/// Whether `model` serves video at all.
pub fn is_video_model(manifests: &[Manifest], model: &str) -> bool {
    manifests.iter().any(|m| m.model == model && m.actions.iter().any(|a| a.video.is_some()))
}

struct Planned {
    model: String,
    action: String,
    inv: Invocation,
    /// The server's copies of the request's frame images and references,
    /// kept until the job ends.
    inputs: Option<JobInputs>,
}

/// Turn a request into a validated invocation, or a 400 naming what is wrong.
async fn plan(state: &AppState, body: &Bytes) -> Result<Planned, ApiError> {
    let provider = state.provider;
    let bad = |m: String| ApiError::invalid_request(provider, m);
    let req: CreateBody = serde_json::from_slice(body).map_err(|e| bad(format!("invalid JSON body: {e}")))?;
    if req.callback_url.is_some() {
        return Err(bad("callback_url is not supported: poll the job".into()));
    }
    let media_in = Wanted::new(req.frame_images, req.input_references).map_err(bad)?;
    let manifests = state.exec.manifests();
    let model = if is_video_model(&manifests, &req.model) {
        req.model.clone()
    } else {
        let model = req.model.clone();
        bridge::ensure_and_recheck(state, provider, &model, |id| is_video_model(&state.exec.manifests(), id).then_some(())).await?;
        model
    };
    let manifests = state.exec.manifests();
    let manifest = manifests.iter().find(|m| m.model == model).ok_or_else(|| ApiError::model_not_found(provider, &model))?;

    let mut brain = BrainOptions::default();
    if let Some(p) = req.provider {
        for (slug, value) in p.options {
            if slug != "brain" {
                return Err(bad(format!("provider.options.{slug}: this server only takes provider.options.brain")));
            }
            brain = serde_json::from_value(value).map_err(|e| bad(format!("provider.options.brain: {e}")))?;
        }
    }
    let spec = video_action(manifest, brain.action.as_deref(), &media_in).map_err(bad)?;
    let contract = spec.video.clone().expect("video_action only returns actions with a contract");

    let wanted = VideoRequest {
        duration: req.duration,
        resolution: req.resolution,
        aspect_ratio: req.aspect_ratio,
        size: req.size,
        generate_audio: req.generate_audio,
        seed: req.seed,
    };
    let mut params = contract.map(&wanted).map_err(bad)?;
    for (k, v) in brain.params {
        if params.contains_key(&k) {
            return Err(bad(format!("provider.options.brain.params.{k} repeats a field the request already sets")));
        }
        params.insert(k, v);
    }
    if let Some(prompt) = req.prompt {
        if !spec.params.iter().any(|p| p.name == "prompt") {
            return Err(bad(format!("{model} {} takes no prompt", spec.name)));
        }
        params.insert("prompt".into(), json!(prompt));
    }
    let mut inv = Invocation::new();
    inv.params = Value::Object(params);

    match (&contract.clip_input, req.input_video) {
        (Some(name), Some(video)) => {
            if video.kind != "video_url" {
                return Err(bad(format!("input_video.type must be \"video_url\", not \"{}\"", video.kind)));
            }
            let blob = clip_from_data_url(&video.video_url.url).await.map_err(bad)?;
            inv.blobs.insert(name.clone(), blob);
        }
        (Some(_), None) => return Err(bad(format!("{model} {} edits a clip: send it as input_video", spec.name))),
        (None, Some(_)) => return Err(bad(format!("{model} {} takes no input_video", spec.name))),
        (None, None) => {}
    }
    // `validate_served`'s two passes with the host's answers between them: a
    // host path param the caller named is refused as unknown first, then the
    // server fills the ones the contract gives it from the request's own
    // inputs, then the full spec checks the result.
    let mut inv = spec.clone().for_serving().validate(inv).map_err(bad)?;
    let inputs = if media_in.is_empty() {
        None
    } else {
        let root = private_temp_dir().map_err(|e| ApiError::internal(provider, e))?;
        let (inputs, host_params) = tokio::task::spawn_blocking(move || media_in.materialize(&root, &contract)).await.map_err(|e| ApiError::internal(provider, e.to_string()))?.map_err(bad)?;
        if let Value::Object(params) = &mut inv.params {
            params.extend(host_params);
        }
        Some(inputs)
    };
    let inv = spec.validate(inv).map_err(bad)?;
    Ok(Planned { model, action: spec.name.clone(), inv, inputs })
}

/// An inline `data:video/...;base64,` clip, decoded into the video blob wire
/// format. Decoding needs a file, so the bytes pass through a private
/// temporary one that is removed before this returns.
async fn clip_from_data_url(url: &str) -> Result<capability::Blob, String> {
    let ext = if url.starts_with("data:video/mp4;") {
        "mp4"
    } else if url.starts_with("data:video/webm;") {
        "webm"
    } else if url.starts_with("data:") {
        return Err("input_video must be a data:video/mp4 or data:video/webm URL".into());
    } else {
        return Err("input_video must be an inline data: URL; remote URLs are not fetched".into());
    };
    let payload = capability::blob::data_url_payload(url).ok_or("input_video is not a base64 data: URL")?;
    let bytes = crate::b64::decode(payload).map_err(|e| format!("input_video: {e}"))?;
    let ext = ext.to_string();
    tokio::task::spawn_blocking(move || {
        // The detail of a scratch-file failure names this machine's paths:
        // logged here, never sent.
        let dir = private_temp_dir().map_err(|e| {
            eprintln!("brain apiserve: input_video scratch directory: {e}");
            "input_video could not be staged".to_string()
        })?;
        let path = dir.join(format!("{}.{ext}", Uuid::new_v4()));
        let result = std::fs::write(&path, &bytes).map_err(|e| {
            eprintln!("brain apiserve: input_video scratch file {}: {e}", path.display());
            "input_video could not be staged".to_string()
        });
        let result = result.and_then(|()| {
            imaging::video::clip_from_untrusted_file(&path, capability::video::MAX_EDIT_CLIP_FRAMES, capability::video::MAX_EDIT_CLIP_PIXELS).map_err(|e| format!("input_video: {e}"))
        });
        let _ = std::fs::remove_file(&path);
        result
    })
    .await
    .map_err(|e| format!("input_video: {e}"))?
}

// ------------------------------------------------------------------ jobs

#[derive(Clone)]
struct Clip {
    path: PathBuf,
    bytes: u64,
    sha256: String,
    usage: Value,
}

enum Phase {
    Running { step: u32, total: u32 },
    Encoding,
    Completed(Clip),
    Failed(ApiError),
    Cancelled,
    Expired,
}

struct Entry {
    scope: String,
    /// The request's `Idempotency-Key` and the digest of its body.
    key: Option<(String, [u8; 32])>,
    finished: Option<Instant>,
    phase: Phase,
    cancel: Option<oneshot::Sender<()>>,
}

/// What a caller may see of one job.
enum View {
    Pending,
    Running { step: u32, total: u32 },
    Completed(Clip),
    Failed(ApiError),
    Cancelled,
    Expired,
}

/// The surface's video jobs and the spool their files live in.
pub struct VideoJobs {
    entries: Mutex<HashMap<Uuid, Entry>>,
    spool: PathBuf,
    spool_bytes: u64,
}

impl Default for VideoJobs {
    fn default() -> VideoJobs {
        VideoJobs::new(std::env::temp_dir().join(format!("brain-videos-{}", std::process::id())), DEFAULT_SPOOL_BYTES)
    }
}

impl VideoJobs {
    /// Jobs whose clips are kept under `spool` (created private on first use,
    /// removed when the jobs are dropped), at most `spool_bytes` of them.
    pub fn new(spool: PathBuf, spool_bytes: u64) -> VideoJobs {
        VideoJobs { entries: Mutex::default(), spool, spool_bytes }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Uuid, Entry>> {
        // Plain data mutated by assignment: a poisoned lock is still consistent.
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn spool_dir(&self) -> Result<&FsPath, String> {
        if !self.spool.is_dir() {
            create_private_dir(&self.spool)?;
        }
        Ok(&self.spool)
    }

    /// Forget finished jobs past [`RESULT_TTL`] and remove their files; then
    /// expire the oldest finished clips while the spool is over budget.
    fn reap(&self, entries: &mut HashMap<Uuid, Entry>, now: Instant) {
        entries.retain(|_, e| {
            let keep = e.finished.is_none_or(|at| now.duration_since(at) < RESULT_TTL);
            if !keep {
                if let Phase::Completed(clip) = &e.phase {
                    let _ = std::fs::remove_file(&clip.path);
                }
            }
            keep
        });
        let mut done: Vec<(Instant, Uuid, u64)> = entries
            .iter()
            .filter_map(|(id, e)| match (&e.phase, e.finished) {
                (Phase::Completed(clip), Some(at)) => Some((at, *id, clip.bytes)),
                _ => None,
            })
            .collect();
        let mut total: u64 = done.iter().map(|(.., b)| b).sum();
        done.sort();
        for (_, id, bytes) in done {
            if total <= self.spool_bytes {
                break;
            }
            if let Some(e) = entries.get_mut(&id) {
                if let Phase::Completed(clip) = &e.phase {
                    let _ = std::fs::remove_file(&clip.path);
                }
                e.phase = Phase::Expired;
            }
            total -= bytes;
        }
    }

    /// The existing job for `(scope, key)`, or a new one. `Err` with the
    /// reason when there is no room, or the key names a different request.
    fn open(&self, scope: &str, key: Option<(String, [u8; 32])>, cancel: oneshot::Sender<()>) -> Result<(Uuid, bool), (StatusCode, &'static str)> {
        let mut entries = self.lock();
        self.reap(&mut entries, Instant::now());
        if let Some((k, digest)) = &key {
            if let Some((id, e)) = entries.iter().find(|(_, e)| e.scope == scope && e.key.as_ref().is_some_and(|(ek, _)| ek == k)) {
                return match &e.key {
                    Some((_, d)) if d == digest => Ok((*id, false)),
                    _ => Err((StatusCode::UNPROCESSABLE_ENTITY, "this Idempotency-Key was used for a different request")),
                };
            }
        }
        if entries.len() >= MAX_JOBS {
            return Err((StatusCode::TOO_MANY_REQUESTS, "the server is holding as many video jobs as it can"));
        }
        if entries.values().filter(|e| e.scope == scope).count() >= MAX_JOBS_PER_SCOPE {
            return Err((StatusCode::TOO_MANY_REQUESTS, "too many video jobs: fetch or delete some before starting another"));
        }
        let id = Uuid::new_v4();
        entries.insert(id, Entry { scope: scope.to_string(), key, finished: None, phase: Phase::Running { step: 0, total: 0 }, cancel: Some(cancel) });
        Ok((id, true))
    }

    fn discard(&self, id: Uuid) {
        self.lock().remove(&id);
    }

    fn progress(&self, id: Uuid, step: u32, total: u32) {
        if let Some(Entry { phase: phase @ Phase::Running { .. }, .. }) = self.lock().get_mut(&id) {
            *phase = Phase::Running { step, total };
        }
    }

    /// Move a running job on; a job already ended (cancelled while the model
    /// wound down) keeps its phase, and a clip made for it is removed.
    fn set(&self, id: Uuid, phase: Phase) {
        let mut entries = self.lock();
        match entries.get_mut(&id) {
            Some(e) if matches!(e.phase, Phase::Running { .. } | Phase::Encoding) => {
                let terminal = !matches!(phase, Phase::Encoding);
                e.phase = phase;
                if terminal {
                    e.finished = Some(Instant::now());
                    e.cancel = None;
                }
            }
            _ => {
                if let Phase::Completed(clip) = phase {
                    let _ = std::fs::remove_file(&clip.path);
                }
            }
        }
        if !entries.is_empty() {
            self.reap(&mut entries, Instant::now());
        }
    }

    fn view(&self, scope: &str, id: Uuid) -> Option<View> {
        let entries = self.lock();
        let e = entries.get(&id).filter(|e| e.scope == scope)?;
        Some(match &e.phase {
            Phase::Running { total: 0, .. } => View::Pending,
            Phase::Running { step, total } => View::Running { step: *step, total: *total },
            Phase::Encoding => View::Running { step: 1, total: 1 },
            Phase::Completed(clip) => View::Completed(clip.clone()),
            Phase::Failed(error) => View::Failed(error.clone()),
            Phase::Cancelled => View::Cancelled,
            Phase::Expired => View::Expired,
        })
    }

    /// Cancel a running job, or forget a finished one and its file.
    fn remove(&self, scope: &str, id: Uuid) -> Option<&'static str> {
        let mut entries = self.lock();
        let e = entries.get_mut(&id).filter(|e| e.scope == scope)?;
        if matches!(e.phase, Phase::Running { .. } | Phase::Encoding) {
            if let Some(cancel) = e.cancel.take() {
                let _ = cancel.send(());
            }
            e.phase = Phase::Cancelled;
            e.finished = Some(Instant::now());
            return Some("cancelled");
        }
        if let Some(Entry { phase: Phase::Completed(clip), .. }) = entries.remove(&id) {
            let _ = std::fs::remove_file(&clip.path);
        }
        Some("deleted")
    }
}

impl Drop for VideoJobs {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.spool);
    }
}

fn create_private_dir(dir: &FsPath) -> Result<(), String> {
    crate::surface::private_dir(dir).map_err(|e| format!("the video spool {}: {e}", dir.display()))
}

fn private_temp_dir() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("brain-video-inputs-{}", std::process::id()));
    create_private_dir(&dir)?;
    Ok(dir)
}

/// What a finished clip measured: the facts a caller bills or checks by.
fn usage_of(outcome: &Outcome) -> Value {
    let Some(video) = outcome.blobs.get("video").filter(|b| b.media == Media::Video) else { return json!({}) };
    let frames = video.meta.get("frames").and_then(Value::as_u64).unwrap_or(0);
    let fps = video.meta.get("fps").and_then(Value::as_f64).unwrap_or(16.0);
    json!({
        "seconds": if fps > 0.0 { frames as f64 / fps } else { 0.0 },
        "frames": frames,
        "fps": fps,
        "width": video.meta.get("w"),
        "height": video.meta.get("h"),
        "audio": outcome.blobs.contains_key("audio"),
    })
}

/// Encode a finished outcome into `path`, hashing the file it wrote.
fn encode(outcome: &Outcome, path: &FsPath) -> Result<Clip, String> {
    let video = outcome.blobs.get("video").filter(|b| b.media == Media::Video).ok_or("the action returned no video")?;
    imaging::video::encode_clip(video, outcome.blobs.get("audio"), path, Default::default())?;
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        bytes += n as u64;
    }
    let sha256 = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Ok(Clip { path: path.to_path_buf(), bytes, sha256, usage: usage_of(outcome) })
}

/// Follow one job to its end: progress, then the encode, then the result.
/// Follow one job to its end. `_inputs` (the server's copies of the request's
/// frame images and references) lives exactly as long as the job.
async fn drive(jobs: Arc<VideoJobs>, provider: crate::Provider, id: Uuid, mut stream: EventStream, mut cancel: oneshot::Receiver<()>, _inputs: Option<JobInputs>) {
    let outcome = loop {
        tokio::select! {
            _ = &mut cancel => return,
            msg = stream.next() => match msg {
                Some(StreamMsg::Progress(step, total)) => jobs.progress(id, step, total),
                Some(StreamMsg::Delta(_) | StreamMsg::Event(_) | StreamMsg::Fetching(_)) => {}
                Some(StreamMsg::Done(outcome)) => break outcome,
                Some(StreamMsg::Err(error)) => return jobs.set(id, Phase::Failed(error)),
                None => return jobs.set(id, Phase::Failed(ApiError::internal(provider, "the job ended without a result"))),
            }
        }
    };
    drop(stream);
    jobs.set(id, Phase::Encoding);
    let path = match jobs.spool_dir() {
        Ok(dir) => dir.join(format!("{id}.mp4")),
        Err(e) => return jobs.set(id, Phase::Failed(ApiError::internal(provider, e))),
    };
    let encoded = tokio::task::spawn_blocking(move || encode(&outcome, &path).inspect_err(|_| drop(std::fs::remove_file(&path)))).await;
    match encoded {
        Ok(Ok(clip)) => jobs.set(id, Phase::Completed(clip)),
        Ok(Err(e)) => jobs.set(id, Phase::Failed(ApiError::internal(provider, format!("encoding the clip: {e}")))),
        Err(e) => jobs.set(id, Phase::Failed(ApiError::internal(provider, format!("encoding the clip: {e}")))),
    }
}

// ------------------------------------------------------------------ handlers

fn scope_of(state: &AppState) -> String {
    state.caller.as_ref().map(|p| state.authenticator.scope(p)).unwrap_or_default()
}

fn no_such_job(state: &AppState) -> Response {
    ApiError::not_found(state.provider, "no such video job").into_response()
}

fn polling_url(id: Uuid) -> String {
    format!("/v1/videos/{id}")
}

async fn create(State(state): State<AppState>, caller: CallerExt, headers: HeaderMap, body: Bytes) -> Response {
    let state = state.scoped(caller);
    let provider = state.provider;
    let key = match headers.get("idempotency-key").map(|v| v.to_str().map(str::trim)) {
        None => None,
        Some(Ok(k)) if !k.is_empty() && k.len() <= 255 => Some((k.to_string(), Sha256::digest(&body).into())),
        Some(_) => return ApiError::invalid_request(provider, "Idempotency-Key must be 1 to 255 visible characters").into_response(),
    };
    let scope = scope_of(&state);
    // A repeated key answers from the job it names, before any validation work.
    if let Some((k, digest)) = &key {
        let found = {
            let entries = state.videos.lock();
            entries.iter().find(|(_, e)| e.scope == scope && e.key.as_ref().is_some_and(|(ek, _)| ek == k)).map(|(id, e)| (*id, e.key.as_ref().is_some_and(|(_, d)| d == digest)))
        };
        match found {
            Some((id, true)) => return status_response(&state, id),
            Some((_, false)) => return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": {"message": "this Idempotency-Key was used for a different request"}}))).into_response(),
            None => {}
        }
    }
    let planned = match plan(&state, &body).await {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let (cancel_tx, cancel_rx) = oneshot::channel();
    let id = match state.videos.open(&scope, key, cancel_tx) {
        Ok((id, true)) => id,
        Ok((id, false)) => return status_response(&state, id),
        Err((code, why)) => return (code, Json(json!({"error": {"message": why}}))).into_response(),
    };
    let stream = match bridge::stream_progress(&state, &planned.model, &planned.action, planned.inv).await {
        Ok(stream) => stream,
        Err(e) => {
            state.videos.discard(id);
            return e.into_response();
        }
    };
    tokio::spawn(drive(Arc::clone(&state.videos), provider, id, stream, cancel_rx, planned.inputs));
    let mut response = (StatusCode::ACCEPTED, Json(json!({"id": id.to_string(), "polling_url": polling_url(id), "status": "pending"}))).into_response();
    if let Ok(location) = HeaderValue::from_str(&polling_url(id)) {
        response.headers_mut().insert(header::LOCATION, location);
    }
    response
}

fn status_response(state: &AppState, id: Uuid) -> Response {
    let Some(view) = state.videos.view(&scope_of(state), id) else { return no_such_job(state) };
    let mut body = json!({"id": id.to_string(), "polling_url": polling_url(id)});
    match view {
        View::Pending => body["status"] = json!("pending"),
        View::Running { step, total } => {
            body["status"] = json!("in_progress");
            body["progress"] = json!({"step": step, "total": total});
        }
        View::Completed(clip) => {
            body["status"] = json!("completed");
            body["unsigned_urls"] = json!([format!("/v1/videos/{id}/content?index=0")]);
            body["usage"] = clip.usage;
            body["content"] = json!([{"media_type": "video/mp4", "bytes": clip.bytes, "sha256": clip.sha256}]);
        }
        View::Failed(error) => {
            body["status"] = json!("failed");
            body["error"] = json!(error.message);
        }
        View::Cancelled => body["status"] = json!("cancelled"),
        View::Expired => body["status"] = json!("expired"),
    }
    Json(body).into_response()
}

async fn status(State(state): State<AppState>, caller: CallerExt, Path(raw): Path<String>) -> Response {
    let state = state.scoped(caller);
    match Uuid::parse_str(&raw) {
        Ok(id) => status_response(&state, id),
        Err(_) => no_such_job(&state),
    }
}

async fn remove(State(state): State<AppState>, caller: CallerExt, Path(raw): Path<String>) -> Response {
    let state = state.scoped(caller);
    let Ok(id) = Uuid::parse_str(&raw) else { return no_such_job(&state) };
    match state.videos.remove(&scope_of(&state), id) {
        Some(outcome) => Json(json!({"id": id.to_string(), "status": outcome})).into_response(),
        None => no_such_job(&state),
    }
}

#[derive(Deserialize)]
struct ContentQuery {
    #[serde(default)]
    index: Option<u32>,
}

/// `bytes=a-b`, `bytes=a-` or `bytes=-n` against a file of `len` bytes, as an
/// inclusive range; `None` for anything else (the whole file is sent).
fn parse_range(value: &str, len: u64) -> Option<Result<(u64, u64), ()>> {
    let spec = value.strip_prefix("bytes=")?;
    if spec.contains(',') || len == 0 {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    let range = match (a.trim(), b.trim()) {
        ("", n) => {
            let n: u64 = n.parse().ok()?;
            (len.saturating_sub(n), len - 1)
        }
        (a, "") => (a.parse().ok()?, len - 1),
        (a, b) => (a.parse().ok()?, b.parse::<u64>().ok()?.min(len - 1)),
    };
    Some(if range.0 <= range.1 && range.0 < len { Ok(range) } else { Err(()) })
}

async fn content(State(state): State<AppState>, caller: CallerExt, Path(raw): Path<String>, Query(q): Query<ContentQuery>, headers: HeaderMap) -> Response {
    let state = state.scoped(caller);
    let Ok(id) = Uuid::parse_str(&raw) else { return no_such_job(&state) };
    let clip = match state.videos.view(&scope_of(&state), id) {
        None => return no_such_job(&state),
        Some(View::Completed(clip)) => clip,
        Some(_) => return (StatusCode::CONFLICT, Json(json!({"error": {"message": "the video is not available: poll its status"}}))).into_response(),
    };
    if q.index.unwrap_or(0) != 0 {
        return ApiError::not_found(state.provider, "a job makes one video: index 0").into_response();
    }
    let range = headers.get(header::RANGE).and_then(|v| v.to_str().ok()).and_then(|v| parse_range(v, clip.bytes));
    let (start, end, code) = match range {
        None => (0, clip.bytes.saturating_sub(1), StatusCode::OK),
        Some(Ok((a, b))) => (a, b, StatusCode::PARTIAL_CONTENT),
        Some(Err(())) => {
            let mut r = StatusCode::RANGE_NOT_SATISFIABLE.into_response();
            if let Ok(v) = HeaderValue::from_str(&format!("bytes */{}", clip.bytes)) {
                r.headers_mut().insert(header::CONTENT_RANGE, v);
            }
            return r;
        }
    };
    let length = if clip.bytes == 0 { 0 } else { end - start + 1 };
    // Read on a blocking thread, a bounded number of chunks ahead of the
    // client: memory stays flat whatever the clip's size.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    let path = clip.path.clone();
    tokio::task::spawn_blocking(move || {
        let send = |item| tx.blocking_send(item).is_ok();
        let mut file = match std::fs::File::open(&path).and_then(|mut f| f.seek(SeekFrom::Start(start)).map(|_| f)) {
            Ok(f) => f,
            Err(e) => {
                send(Err(e));
                return;
            }
        };
        let mut left = length;
        let mut buf = vec![0u8; CHUNK];
        while left > 0 {
            let want = buf.len().min(left as usize);
            match file.read(&mut buf[..want]) {
                Ok(0) => break,
                Ok(n) => {
                    left -= n as u64;
                    if !send(Ok(Bytes::copy_from_slice(&buf[..n]))) {
                        break;
                    }
                }
                Err(e) => {
                    send(Err(e));
                    return;
                }
            }
        }
    });
    let mut response = Response::new(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)));
    *response.status_mut() = code;
    let h = response.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("video/mp4"));
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    h.insert(header::CONTENT_LENGTH, HeaderValue::from(length));
    if code == StatusCode::PARTIAL_CONTENT {
        if let Ok(v) = HeaderValue::from_str(&format!("bytes {start}-{end}/{}", clip.bytes)) {
            h.insert(header::CONTENT_RANGE, v);
        }
    }
    response
}

/// `GET /videos/models`: one entry per video model, in the OpenRouter video
/// model listing's field names, from its generation action's contract; the
/// edit actions it also offers are listed under `actions` (an extension).
async fn models(State(state): State<AppState>) -> Json<Value> {
    let manifests = state.exec.manifests();
    let mut data = Vec::new();
    for m in manifests.iter().filter(|m| state.lists(&m.model)) {
        let sellable: Vec<&ActionSpec> = m.actions.iter().filter(|a| a.video.as_ref().is_some_and(|c| c.sellable)).collect();
        let Some(generate) = sellable.iter().find(|a| a.video.as_ref().is_some_and(|c| c.clip_input.is_none())).copied() else { continue };
        let contract = generate.video.as_ref().expect("filtered on video");
        let mut entry = json!({
            "id": m.model,
            "canonical_slug": m.model,
            "name": m.model,
            "description": generate.summary,
            "pricing_skus": {},
            "allowed_passthrough_parameters": ["brain"],
        });
        if let (Some(obj), Value::Object(listing)) = (entry.as_object_mut(), contract.listing()) {
            obj.extend(listing);
        }
        // Frame images and references the model takes through ANY of its
        // generation actions: a request carrying them is routed to that one.
        let generation = || sellable.iter().filter_map(|a| a.video.as_ref()).filter(|c| c.clip_input.is_none());
        let mut frames: Vec<&str> = generation().flat_map(|c| c.frame_types()).collect();
        frames.sort_by_key(|t| capability::video::FRAME_TYPES.iter().position(|f| f == t));
        frames.dedup();
        let mut references: Vec<&str> = Vec::new();
        for kind in generation().flat_map(|c| c.reference_types()) {
            if !references.contains(&kind) {
                references.push(kind);
            }
        }
        entry["supported_frame_images"] = json!(frames);
        entry["supported_reference_types"] = json!(references);
        entry["actions"] = sellable
            .iter()
            .map(|a| {
                let served = (*a).clone().for_serving();
                json!({
                    "action": a.name,
                    "summary": a.summary,
                    "edits": a.video.as_ref().and_then(|c| c.clip_input.clone()).is_some(),
                    "params": served.params.iter().map(|p| json!({"name": p.name, "type": p.ty.name(), "required": p.required, "default": p.default, "min": p.min, "max": p.max})).collect::<Vec<_>>(),
                })
            })
            .collect();
        data.push(entry);
    }
    data.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Json(json!({ "data": data }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_follow_rfc_9110_for_one_part() {
        assert_eq!(parse_range("bytes=0-9", 100), Some(Ok((0, 9))));
        assert_eq!(parse_range("bytes=90-", 100), Some(Ok((90, 99))));
        assert_eq!(parse_range("bytes=-10", 100), Some(Ok((90, 99))));
        assert_eq!(parse_range("bytes=95-200", 100), Some(Ok((95, 99))));
        assert_eq!(parse_range("bytes=100-", 100), Some(Err(())));
        assert_eq!(parse_range("bytes=0-1,5-6", 100), None);
        assert_eq!(parse_range("items=0-1", 100), None);
    }
}
