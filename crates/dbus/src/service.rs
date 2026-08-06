// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `com.swedishembedded.Brain1.Manager` D-Bus interface.
//!
//! Each method only **validates and translates**: it turns D-Bus args (+ input fds)
//! into a [`capability::Invocation`], submits a [`residency::Job`] to the shared
//! [`Executor`], and returns — no model runs on the zbus dispatch task. `Run` awaits
//! the executor's reply and returns the result fds; `Subscribe` returns a stream fd
//! immediately and lets the executor fan events into it. The executor schedules,
//! batches, and manages residency across every front-end uniformly.

use std::collections::HashMap;
use std::io::Read;
use std::os::fd::{AsFd, OwnedFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use capability::{Blob, CancelToken, Invocation, Media, Outcome, Progress};
use residency::{Executor, Job, ModelSupplier, Supply};
use serde_json::{json, Value};
use zbus::fdo;
use zbus::object_server::SignalEmitter;
use zbus::zvariant::OwnedFd as ZOwnedFd;

use crate::fd::{bytes_to_fd, read_fd_to_vec};
use crate::stream::StreamTx;

/// Armed cancel tokens for in-flight jobs, keyed by job id — `residency::jobs::
/// JobRegistry`, the SAME shared type `crates/apiserve`'s `AppState` uses
/// (keyed by `Uuid` there instead). An entry lives from submission until the
/// job's reply fires, so `Cancel` can find any running job.
type JobRegistry = residency::jobs::JobRegistry<u64>;

/// A legacy short name (e.g. `"mock"`) is a deprecation, not a second id: every
/// D-Bus entry point that takes a `model` string resolves it to its canonical
/// `brain/<name>` form here, before it ever reaches a [`Job`] or a manifest
/// lookup -- `Manifest.model` itself is never a legacy name (see
/// `modelref::alias`'s module docs).
fn resolve_model_alias(model: String) -> String {
    brain_modelref::alias::canonical(&model).map(str::to_string).unwrap_or(model)
}

pub struct Manager {
    executor: Executor,
    version: String,
    jobs: JobRegistry,
    next_job: AtomicU64,
    /// Classifies/fetches a `model` string that isn't already resident
    /// (transparent auto-fetch). `None` -- the default -- means an unresolved
    /// model is a plain `"no model '…'"` reply with zero I/O, exactly today's
    /// behavior. Set only via [`Manager::with_supplier`] (`crate::serve`'s `ServeOpts::supplier`).
    supplier: Option<Arc<dyn ModelSupplier>>,
    /// Edge concurrency ceiling (`residency::admission::EDGE_CONCURRENCY`) --
    /// the SAME gate `crates/apiserve`'s HTTP surfaces apply, so a saturated
    /// server sheds identically over either transport. A `Run`/`Subscribe`
    /// that can't acquire a permit immediately is shed, not queued.
    edge_permits: Arc<tokio::sync::Semaphore>,
    /// Bounded wait for a request to be ADMITTED (claimed onto a lane) before
    /// it is shed -- `residency::admission::admit_deadline_from_env()`,
    /// mirroring `apiserve::AppState::admit_deadline`.
    admit_deadline: std::time::Duration,
}

impl Manager {
    pub fn new(executor: Executor) -> Manager {
        Manager {
            executor,
            version: env!("CARGO_PKG_VERSION").to_string(),
            jobs: JobRegistry::new(),
            next_job: AtomicU64::new(0),
            supplier: None,
            edge_permits: Arc::new(tokio::sync::Semaphore::new(residency::admission::EDGE_CONCURRENCY)),
            admit_deadline: residency::admission::admit_deadline_from_env(),
        }
    }

    /// Attach a model supplier (builder-style) so an unresolved model
    /// auto-fetches instead of every dispatch entry point replying `"no model
    /// '…'"`. `None` restores today's no-auto-fetch behavior.
    pub fn with_supplier(mut self, supplier: Option<Arc<dyn ModelSupplier>>) -> Manager {
        self.supplier = supplier;
        self
    }

    /// Override the admission deadline (builder-style) -- tests use this for a
    /// short, deterministic deadline instead of the env-derived default.
    pub fn with_admit_deadline(mut self, deadline: std::time::Duration) -> Manager {
        self.admit_deadline = deadline;
        self
    }

    /// `model` isn't in `self.executor.manifests()` -- try to make it resident
    /// via `self.supplier` before the caller dispatches. Blocks (this is called
    /// from an async zbus method, so it runs on a `spawn_blocking` task) for the
    /// duration of a cold fetch. Every non-success outcome (no supplier, a
    /// classify of `Unknown`, or the fetch itself failing) maps to the SAME
    /// `"no model '{model}'"` `fdo::Error` a genuinely unknown model already
    /// gets elsewhere in this file -- the raw fetch-failure reason (which could
    /// carry a hub URL or a filesystem path) is logged server-side, never
    /// reflected to the caller (`docs/api-security-audit.md`'s error-hygiene
    /// requirement).
    async fn ensure_resident(&self, model: &str) -> fdo::Result<()> {
        if self.executor.manifests().iter().any(|m| m.model == model) {
            return Ok(());
        }
        let Some(supplier) = self.supplier.clone() else {
            return Err(fdo::Error::Failed(format!("no model '{model}'")));
        };
        match supplier.classify(model) {
            Supply::Fetchable => {}
            Supply::Resident | Supply::Unknown(_) => return Err(fdo::Error::Failed(format!("no model '{model}'"))),
        }
        let exec = self.executor.clone();
        let model_owned = model.to_string();
        let outcome = tokio::task::spawn_blocking(move || exec.ensure_model(&model_owned, supplier.as_ref(), &mut |_, _, _| {})).await;
        match outcome {
            Ok(Ok(())) => Ok(()),
            Ok(Err(reason)) => {
                eprintln!("brain dbus: auto-fetch {model}: {reason}");
                Err(fdo::Error::Failed(format!("no model '{model}'")))
            }
            Err(join_err) => {
                eprintln!("brain dbus: auto-fetch {model}: blocking task failed: {join_err}");
                Err(fdo::Error::Failed(format!("no model '{model}'")))
            }
        }
    }

    /// Arm `inv` with a fresh [`CancelToken`] and register it under a new job id,
    /// returning both -- the caller must remove the entry (via [`finish_job`])
    /// when the reply fires, and may use the token to cancel on an admission
    /// timeout before the job ever starts.
    fn register_job(&self, inv: &mut Invocation) -> (u64, CancelToken) {
        let token = CancelToken::armed();
        inv.cancel = token.clone();
        let job = self.next_job.fetch_add(1, Ordering::Relaxed) + 1;
        self.jobs.insert(job, token.clone());
        (job, token)
    }

    /// Assemble an [`Invocation`] from params JSON, input fds, and per-fd metadata.
    fn build_inv(&self, params: &str, in_fds: HashMap<String, ZOwnedFd>, in_meta: &str) -> Result<Invocation, String> {
        let params: Value = if params.trim().is_empty() { json!({}) } else { serde_json::from_str(params).map_err(|e| format!("params JSON: {e}"))? };
        let meta: Value = if in_meta.trim().is_empty() { json!({}) } else { serde_json::from_str(in_meta).map_err(|e| format!("in_meta JSON: {e}"))? };
        let mut inv = Invocation { params, blobs: Default::default(), cancel: Default::default() };
        for (name, zfd) in in_fds {
            let ofd: std::os::fd::OwnedFd = zfd.into();
            let bytes = read_fd_to_vec(ofd.as_fd()).map_err(|e| format!("reading input fd '{name}': {e}"))?;
            let m = meta.get(&name);
            let media = m.and_then(|v| v.get("media")).and_then(|v| v.as_str()).and_then(Media::parse).unwrap_or(Media::Bytes);
            let bmeta = m.cloned().unwrap_or(Value::Null);
            inv.blobs.insert(name, Blob { media, bytes, meta: bmeta });
        }
        Ok(inv)
    }
}

/// The server-side streaming loop for [`Manager::stream_transcribe`]. Reads f32 LE
/// PCM from `pcm` until EOF. When the model advertises `transcribe_stream`
/// (`session` is `Some`), every window is one step of a **live session** — the
/// model carries encoder/decoder state across windows (frame-synchronous, no
/// per-window re-encode) and each `segment` frame is the newly emitted text;
/// EOF sends a final `eos` step that flushes the tail. Otherwise each window is an
/// independent `transcribe` job (the offline fallback, e.g. qwen-asr). Runs on its
/// own thread (blocking reads + a blocking wait per window), so a stream whose
/// model keeps up (RTF < 1) stays near-real-time and the OS pipe buffer absorbs
/// the compute gap.
fn stream_reader(pcm: OwnedFd, executor: Executor, model: String, window_samples: usize, prompt_id: i64, session: Option<String>, stream: Arc<Mutex<StreamTx>>) {
    let mut file = std::fs::File::from(pcm);
    let mut carry: Vec<u8> = Vec::new();
    let mut samples: Vec<f32> = Vec::new();
    let mut buf = [0u8; 16384];
    let mut index = 0u32;
    let mut full = String::new();
    loop {
        let n = match file.read(&mut buf) {
            Ok(0) => break, // writer closed → EOF
            Ok(n) => n,
            Err(_) => break,
        };
        carry.extend_from_slice(&buf[..n]);
        let whole = carry.len() / 4 * 4;
        samples.extend(carry[..whole].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        carry.drain(..whole);
        while samples.len() >= window_samples {
            let window: Vec<f32> = samples.drain(..window_samples).collect();
            emit_window(&executor, &model, &window, prompt_id, index, session.as_deref(), false, &stream, &mut full);
            index += 1;
        }
        // stop early if the subscriber went away
        if stream.lock().map(|s| s.disconnected).unwrap_or(true) {
            return;
        }
    }
    // flush the tail: a session always gets a final eos step (even with no
    // samples left — it drains the model's internal lookahead); the offline
    // fallback only submits a non-empty trailing window.
    if session.is_some() || !samples.is_empty() {
        emit_window(&executor, &model, &samples, prompt_id, index, session.as_deref(), true, &stream, &mut full);
        index += 1;
    }
    if let Ok(mut s) = stream.lock() {
        s.segment(index, "", true);
        s.done(&json!({ "text": full.trim(), "segments": index }));
    }
}

/// Submit one window — a `transcribe_stream` session step when `session` is set,
/// an independent `transcribe` job otherwise — block for its result, and emit a
/// `segment` frame (or an `error`). Appends the segment text to the running
/// transcript: session segments are exact deltas of one growing transcription and
/// concatenate verbatim; independent windows are joined with a space.
#[allow(clippy::too_many_arguments)]
fn emit_window(executor: &Executor, model: &str, window: &[f32], prompt_id: i64, index: u32, session: Option<&str>, eos: bool, stream: &Arc<Mutex<StreamTx>>, full: &mut String) {
    let bytes: Vec<u8> = window.iter().flat_map(|f| f.to_le_bytes()).collect();
    let blob = Blob::new(Media::Audio, bytes).with_meta(json!({"sample_rate": 16000}));
    let mut inv = Invocation::new().set("prompt_id", json!(prompt_id)).blob("audio", blob);
    let action = match session {
        Some(id) => {
            inv = inv.set("stream", json!(id)).set("eos", json!(eos));
            "transcribe_stream"
        }
        None => "transcribe",
    };
    let (tx, rx) = std::sync::mpsc::channel();
    executor.submit(Job::new(model.to_string(), action, inv).reply(move |r| {
        let _ = tx.send(r);
    }));
    match rx.recv() {
        Ok(Ok(outcome)) => {
            let raw = outcome.outputs.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let text = if session.is_some() { raw.to_string() } else { raw.trim().to_string() };
            if !text.is_empty() {
                if session.is_none() && !full.is_empty() {
                    full.push(' ');
                }
                full.push_str(&text);
            }
            if let Ok(mut s) = stream.lock() {
                s.segment(index, &text, false);
            }
        }
        Ok(Err(e)) => {
            if let Ok(mut s) = stream.lock() {
                s.error(&e);
            }
        }
        Err(_) => {} // executor gone
    }
}

/// Drop a finished job's cancel token from the registry.
fn finish_job(jobs: &JobRegistry, job: u64) {
    jobs.remove(&job);
}

/// Convert an [`Outcome`]'s blobs to fds (memfd, or best-effort dmabuf) + the
/// `out_meta` JSON describing each. Errors propagate as a D-Bus failure.
fn outcome_to_fds(outcome: &Outcome, want_dmabuf: bool) -> Result<(HashMap<String, ZOwnedFd>, Value), String> {
    let mut out_fds = HashMap::new();
    let mut meta = serde_json::Map::new();
    for (name, blob) in &outcome.blobs {
        let (fd, transport) = bytes_to_fd(name, &blob.bytes, want_dmabuf).map_err(|e| format!("blob {name}: {e}"))?;
        meta.insert(name.clone(), json!({"media": blob.media.name(), "transport": transport, "bytes": blob.bytes.len(), "meta": blob.meta}));
        out_fds.insert(name.clone(), fd.into());
    }
    Ok((out_fds, Value::Object(meta)))
}

/// Submit `Job::new(model, action, inv)`, forwarding its progress/blobs/result
/// into `stream` -- the tail shared by `subscribe`'s already-resident fast path
/// and its post-fetch continuation, so there is exactly one place that turns a
/// running job's callbacks into stream frames. A free function, not a method
/// on `Manager`: the `#[zbus::interface]` macro on `impl Manager` below treats
/// every item in that block as a candidate D-Bus method, so a private helper
/// must live outside it.
/// `admit_deadline`/`token` implement the SAME admission race `run` does
/// (`apiserve::bridge::submit`'s shape): if the job hasn't started on a lane
/// within the deadline, cancel it and tell the subscriber via an `error`
/// frame rather than leaving them waiting silently. `permit` is the edge
/// concurrency slot `subscribe` acquired -- held for the job's whole life,
/// released only when its reply fires (whichever way it resolves).
#[allow(clippy::too_many_arguments)]
fn submit_subscribed_job(
    exec: Executor,
    jobs: JobRegistry,
    stream: Arc<Mutex<crate::stream::StreamTx>>,
    job: u64,
    model: String,
    action: String,
    inv: Invocation,
    admit_deadline: std::time::Duration,
    token: CancelToken,
    permit: tokio::sync::OwnedSemaphorePermit,
) {
    let (sp, sr) = (stream.clone(), stream.clone());
    let (admit_tx, admit_rx) = tokio::sync::oneshot::channel::<()>();
    let mut admit_tx = Some(admit_tx);
    exec.submit(
        Job::new(model, action, inv)
            .on_admit(move || {
                if let Some(tx) = admit_tx.take() {
                    let _ = tx.send(());
                }
            })
            .on_progress(move |p: Progress| {
                if let Ok(mut s) = sp.lock() {
                    s.progress(p.step, p.total, &p.message, None, p.delta.as_deref(), p.event.as_ref());
                }
            })
            .reply(move |r| {
                let _permit = permit; // held until the reply fires, then dropped
                finish_job(&jobs, job);
                if let Ok(mut s) = sr.lock() {
                    match r {
                        Ok(outcome) => {
                            for (name, blob) in &outcome.blobs {
                                match bytes_to_fd(name, &blob.bytes, false) {
                                    Ok((fd, _)) => s.blob(name, blob.media.name(), &blob.meta, fd.as_fd()),
                                    Err(e) => s.error(&format!("blob {name}: {e}")),
                                }
                            }
                            s.done(&outcome.outputs);
                        }
                        Err(e) => s.error(&e),
                    }
                }
            }),
    );

    // Race the admit deadline in the background: `subscribe` already returned
    // the stream fd, so a timeout here can't turn into an error return the way
    // `run`'s does -- it cancels the (already-submitted) job and reports the
    // timeout as an `error` frame on the SAME stream instead. If admission
    // wins the race first, this task is a no-op and exits immediately.
    tokio::spawn(async move {
        if tokio::time::timeout(admit_deadline, admit_rx).await.is_err() {
            token.cancel();
            if let Ok(mut s) = stream.lock() {
                s.error("request could not be admitted within the deadline");
            }
        }
    });
}

#[zbus::interface(name = "com.swedishembedded.Brain1.Manager")]
impl Manager {
    /// JSON array of every model's manifest (discovery).
    async fn manifests(&self) -> String {
        serde_json::to_string(&Value::Array(self.executor.manifests().iter().map(|m| m.to_json()).collect())).unwrap_or_else(|_| "[]".into())
    }

    /// The served model names.
    async fn list_models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.executor.manifests().iter().map(|m| m.model.clone()).collect();
        v.sort();
        v
    }

    /// Run one action (see the crate/example docs for the fd protocol). `transport`
    /// requests `"memfd"` (default) or `"dmabuf"` (best-effort).
    #[zbus(out_args("result", "out_fds", "out_meta"))]
    async fn run(
        &self,
        model: String,
        action: String,
        params: String,
        in_fds: HashMap<String, ZOwnedFd>,
        in_meta: String,
        transport: String,
    ) -> fdo::Result<(String, HashMap<String, ZOwnedFd>, String)> {
        // Edge concurrency ceiling: shed immediately (never queue for a permit)
        // when the server is already saturated -- the same "rejected at the
        // edge" signal `apiserve`'s `GlobalConcurrencyLimitLayer`/`LoadShedLayer`
        // give over HTTP, distinct from the admit-deadline case below ("accepted,
        // but couldn't start a lane in time").
        let _permit = self.edge_permits.clone().try_acquire_owned().map_err(|_| fdo::Error::Failed("server saturated: request rejected at the edge".into()))?;
        let model = resolve_model_alias(model);
        self.ensure_resident(&model).await?;
        let mut inv = self.build_inv(&params, in_fds, &in_meta).map_err(fdo::Error::Failed)?;
        // Armed so ActiveJobs counts it; a Run has no client-visible job id, so its
        // token is only ever dropped here (the reply), never cancelled by Cancel.
        let (job, token) = self.register_job(&mut inv);
        let jobs = self.jobs.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let (admit_tx, admit_rx) = tokio::sync::oneshot::channel::<()>();
        self.executor.submit(
            Job::new(model, action, inv)
                .on_admit(move || {
                    let _ = admit_tx.send(());
                })
                // Run is one-shot; no mid-run streaming (default no-op on_progress).
                .reply(move |r| {
                    finish_job(&jobs, job);
                    let _ = tx.send(r);
                }),
        );

        // Admission race: work started on a lane vs. the deadline elapsing --
        // mirrors `apiserve::bridge::submit` exactly, so both transports shed a
        // request that couldn't start in time identically.
        tokio::select! {
            _ = admit_rx => {}
            _ = tokio::time::sleep(self.admit_deadline) => {
                token.cancel();
                finish_job(&self.jobs, job);
                return Err(fdo::Error::Failed("request could not be admitted within the deadline".into()));
            }
        }

        let outcome = rx.await.map_err(|_| fdo::Error::Failed("executor dropped the reply".into()))?.map_err(fdo::Error::Failed)?;
        let want_dmabuf = transport.eq_ignore_ascii_case("dmabuf");
        let (out_fds, out_meta) = outcome_to_fds(&outcome, want_dmabuf).map_err(fdo::Error::Failed)?;
        Ok((serde_json::to_string(&outcome.outputs).unwrap_or_else(|_| "{}".into()), out_fds, serde_json::to_string(&out_meta).unwrap_or_else(|_| "{}".into())))
    }

    /// Start a streaming run. Returns a job id + a SEQPACKET fd delivering framed
    /// progress/blob/done/error events.
    ///
    /// For a model that isn't already resident and classifies `Fetchable`, this
    /// returns the stream fd IMMEDIATELY (before any network I/O) and runs the
    /// fetch in the background, forwarding its progress as `phase: "fetching"`
    /// progress frames over the SAME stream, then submits the real job once the
    /// model is ready -- mirroring `crate::bridge::stream_with_autofetch` on the
    /// HTTP side. An `Unknown`/no-supplier model is still the plain `"no model
    /// '…'"` error with zero I/O, no stream opened at all.
    #[zbus(out_args("job", "event_fd"))]
    async fn subscribe(
        &self,
        model: String,
        action: String,
        params: String,
        in_fds: HashMap<String, ZOwnedFd>,
        in_meta: String,
    ) -> fdo::Result<(u64, ZOwnedFd)> {
        // Edge concurrency ceiling -- see `run`'s comment. Held for the whole
        // subscription (not just this method call), so it's threaded into
        // `submit_subscribed_job` and dropped only when the job's reply fires.
        let permit = self.edge_permits.clone().try_acquire_owned().map_err(|_| fdo::Error::Failed("server saturated: request rejected at the edge".into()))?;
        let model = resolve_model_alias(model);
        let mut inv = self.build_inv(&params, in_fds, &in_meta).map_err(fdo::Error::Failed)?;
        // Arm a cancel token under the returned job id: `Cancel(job)` flips it and
        // the running action aborts at its next poll (see docs/serving-contract.md).
        let (job, token) = self.register_job(&mut inv);
        let jobs = self.jobs.clone();
        let (stream, client) = crate::stream::pair().map_err(|e| fdo::Error::Failed(e.to_string()))?;
        let stream = Arc::new(Mutex::new(stream));
        let admit_deadline = self.admit_deadline;

        if self.executor.manifests().iter().any(|m| m.model == model) {
            submit_subscribed_job(self.executor.clone(), jobs, stream, job, model, action, inv, admit_deadline, token, permit);
            return Ok((job, client.into()));
        }

        let Some(supplier) = self.supplier.clone() else {
            finish_job(&jobs, job);
            return Err(fdo::Error::Failed(format!("no model '{model}'")));
        };
        match supplier.classify(&model) {
            Supply::Fetchable => {}
            Supply::Resident | Supply::Unknown(_) => {
                finish_job(&jobs, job);
                return Err(fdo::Error::Failed(format!("no model '{model}'")));
            }
        }

        let exec = self.executor.clone();
        let model_owned = model.clone();
        let stream_bg = stream.clone();
        tokio::spawn(async move {
            let fetch_stream = stream_bg.clone();
            let fetch_model = model_owned.clone();
            let fetch_exec = exec.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                fetch_exec.ensure_model(&fetch_model, supplier.as_ref(), &mut |name, got, total| {
                    if let Ok(mut s) = fetch_stream.lock() {
                        s.progress(got, total, &format!("fetching {fetch_model}: {name}"), Some("fetching"), None, None);
                    }
                })
            })
            .await;
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(reason)) => {
                    eprintln!("brain dbus: auto-fetch {model_owned}: {reason}");
                    if let Ok(mut s) = stream_bg.lock() {
                        s.error(&format!("no model '{model_owned}'"));
                    }
                    finish_job(&jobs, job);
                    return;
                }
                Err(join_err) => {
                    eprintln!("brain dbus: auto-fetch {model_owned}: blocking task failed: {join_err}");
                    if let Ok(mut s) = stream_bg.lock() {
                        s.error(&format!("no model '{model_owned}'"));
                    }
                    finish_job(&jobs, job);
                    return;
                }
            }
            submit_subscribed_job(exec, jobs, stream_bg, job, model_owned, action, inv, admit_deadline, token, permit);
        });

        Ok((job, client.into()))
    }

    /// **Live streaming transcription.** The client writes raw mono f32 LE PCM at
    /// 16 kHz to the returned end of `pcm` (a pipe) continuously; the server reads
    /// it, slices it into `window_ms` windows and submits them as [`Job`]s to the
    /// shared executor (so concurrent streams **batch** and are scheduled
    /// uniformly), streaming `segment` frames back over the returned SEQPACKET
    /// event fd — ending with a `done` frame carrying the full transcript when the
    /// input reaches EOF (client closes its write end).
    ///
    /// A model advertising `transcribe_stream` (nemotron) is driven as **one live
    /// session**: state carries across windows (frame-synchronous, no per-window
    /// re-encode) and each `segment` is the newly emitted text; EOF flushes the
    /// model's tail. Other models (qwen-asr) fall back to independent per-window
    /// `transcribe` jobs.
    ///
    /// `params` (JSON): `{"window_ms":1000,"sample_rate":16000,"prompt_id":0}`.
    /// `model` is `"nemotron"` (the streaming model) or `"qwen-asr"`.
    #[zbus(out_args("job", "event_fd"))]
    async fn stream_transcribe(&self, model: String, params: String, pcm: ZOwnedFd) -> fdo::Result<(u64, ZOwnedFd)> {
        let model = resolve_model_alias(model);
        self.ensure_resident(&model).await?;
        let p: Value = if params.trim().is_empty() { json!({}) } else { serde_json::from_str(&params).map_err(|e| fdo::Error::Failed(format!("params JSON: {e}")))? };
        let sample_rate = p.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(16000);
        if sample_rate != 16000 {
            return Err(fdo::Error::Failed(format!("stream_transcribe: sample_rate must be 16000, got {sample_rate}")));
        }
        let window_ms = p.get("window_ms").and_then(|v| v.as_u64()).unwrap_or(1000).max(50);
        let window_samples = (sample_rate * window_ms / 1000) as usize;
        let prompt_id = p.get("prompt_id").and_then(|v| v.as_i64()).unwrap_or(0);

        // A model advertising `transcribe_stream` gets a live session (state carried
        // across windows — frame-synchronous); anything else falls back to
        // independent per-window `transcribe` jobs.
        let session = self
            .executor
            .manifests()
            .iter()
            .any(|m| m.model == model && m.actions.iter().any(|a| a.name == "transcribe_stream"))
            .then(|| {
                static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                format!("dbus-{}-{}", std::process::id(), SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            });

        let (stream, client) = crate::stream::pair().map_err(|e| fdo::Error::Failed(e.to_string()))?;
        let stream = Arc::new(Mutex::new(stream));
        let executor = self.executor.clone();
        let pcm_ofd: OwnedFd = pcm.into();
        std::thread::Builder::new()
            .name("brain-asr-stream".into())
            .spawn(move || stream_reader(pcm_ofd, executor, model, window_samples, prompt_id, session, stream))
            .map_err(|e| fdo::Error::Failed(format!("spawn stream reader: {e}")))?;

        // Correlation id only: the stream's windows are internal executor jobs, so
        // this id is not in the cancel registry — end a stream by closing the pipe.
        let job = self.next_job.fetch_add(1, Ordering::Relaxed) + 1;
        Ok((job, client.into()))
    }

    /// Scheduler + residency counters as JSON (builds/evictions/batches/max_batch/…)
    /// — proof that batching and eviction happen, and the numbers to profile with.
    async fn stats(&self) -> String {
        let s = self.executor.stats();
        json!({
            "builds": s.builds, "evictions": s.evictions, "batches": s.batches,
            "jobs": s.jobs, "max_batch": s.max_batch, "resident": s.resident, "queue_peak": s.queue_peak,
        })
        .to_string()
    }

    /// The full **self-describing stats snapshot** as JSON — the hierarchical tree
    /// braintop renders (accelerators / models with per-instance residency /
    /// executor counters / requests / connections, plus open `extra` maps). Built
    /// from the live executor via `brain-stats`. One-shot pull; `StatsStream`
    /// pushes the same document live at >=2 Hz.
    async fn stats_snapshot(&self) -> String {
        brain_stats::snapshot_from_executor(&self.executor).to_json_string()
    }

    /// Live stats: the same JSON document as `StatsSnapshot`, emitted on a timer by
    /// the background task ([`run_stats_stream`]) so braintop can subscribe instead
    /// of polling. Declared here so it belongs to this interface/object path.
    #[zbus(signal)]
    async fn stats_stream(emitter: &SignalEmitter<'_>, snapshot: String) -> zbus::Result<()>;

    /// Cooperatively cancel a job (from `Subscribe`) by id: flips its token so the
    /// running action aborts at its next poll (`Err("cancelled")` arrives as an
    /// `error` frame). Returns `true` if the job was found still in flight,
    /// `false` for an unknown or already-finished id.
    async fn cancel(&self, job: u64) -> bool {
        self.jobs.get(&job).map(|t| t.cancel()).is_some()
    }

    #[zbus(property)]
    async fn version(&self) -> String {
        self.version.clone()
    }

    /// Jobs currently in flight (submitted via `Run`/`Subscribe`, reply not yet fired).
    #[zbus(property)]
    async fn active_jobs(&self) -> u32 {
        self.jobs.len() as u32
    }

    #[zbus(property)]
    async fn models(&self) -> Vec<String> {
        self.list_models().await
    }
}

/// How often the `StatsStream` signal fires. 500 ms == 2 Hz — the floor a live
/// braintop view needs. Kept a const so the cadence is one obvious knob.
pub const STATS_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

/// Background task that pushes the stats snapshot: every [`STATS_INTERVAL`] it
/// builds a fresh snapshot from `executor` and emits it as the `StatsStream`
/// signal on the `Manager` served at `path`. Reuses the existing interface/object
/// path (braintop subscribes there). Returns when the connection is gone **or**
/// `shutdown` fires.
///
/// The `shutdown` race is load-bearing, not cosmetic: this task holds a clone of
/// `conn` for as long as it runs, and `Connection::graceful_shutdown()` awaits a
/// drop event that fires only once every clone is gone. Without an explicit exit
/// path, a healthy connection (every `stats_stream` emit keeps succeeding) never
/// releases its clone, and shutdown deadlocks forever — the tick loop looks
/// exactly like the reason to keep going, from the inside.
pub async fn run_stats_stream(conn: zbus::Connection, executor: Executor, path: &'static str, shutdown: brain_shutdown::Shutdown) {
    let mut tick = tokio::time::interval(STATS_INTERVAL);
    loop {
        tokio::select! {
            _ = shutdown.wait() => return,
            _ = tick.tick() => {}
        }
        // Fetch the served interface fresh each tick: if it is not yet registered
        // (startup race) skip this tick rather than give up the whole stream.
        let iface = match conn.object_server().interface::<_, Manager>(path).await {
            Ok(i) => i,
            Err(_) => continue,
        };
        let json = brain_stats::snapshot_from_executor(&executor).to_json_string();
        if Manager::stats_stream(iface.signal_emitter(), json).await.is_err() {
            return; // connection dropped — stop emitting
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use capability::Manifest;
    use residency::budget::Budgets;
    use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel, Supply};

    use super::{resolve_model_alias, Manager};

    const GB: u64 = 1 << 30;

    #[test]
    fn resolve_model_alias_maps_a_legacy_short_name_to_its_canonical_form() {
        assert_eq!(resolve_model_alias("mock".to_string()), "brain/mock");
    }

    #[test]
    fn resolve_model_alias_leaves_a_canonical_or_unknown_name_untouched() {
        assert_eq!(resolve_model_alias("brain/mock".to_string()), "brain/mock");
        assert_eq!(resolve_model_alias("Qwen/Qwen3-0.6B".to_string()), "Qwen/Qwen3-0.6B");
    }

    /// The `StatsSnapshot`/`StatsStream` payload is built by `brain-stats` from the
    /// executor. Verify the wiring end-to-end (executor -> stats -> JSON): the
    /// accelerators enumerate from the device budgets and the document round-trips.
    #[test]
    fn stats_snapshot_json_is_well_formed_from_the_executor() {
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 2 * GB).set(Device::Cpu, 16 * GB, 0);
        let exec = Executor::start(vec![], budgets, Policy::default());
        let json = brain_stats::snapshot_from_executor(&exec).to_json_string();
        let snap = brain_stats::StatsSnapshot::from_json_str(&json).expect("valid snapshot JSON");
        // Two budgeted devices → two accelerator rows, data-driven from the executor.
        assert_eq!(snap.accelerators.len(), 2);
        assert!(snap.accelerators.iter().any(|a| a.id == "cpu"));
        assert!(snap.accelerators.iter().any(|a| a.id == "gpu0" && a.mem_total == 24 * GB));
        assert!(snap.models.is_empty());
    }

    /// A stub [`residency::ModelSupplier`] mirroring the one in
    /// `crates/apiserve/tests/api.rs`: `"vendor/fetchable"` classifies as
    /// fetchable and "fetching" it registers a minimal resident under that exact
    /// name; every other model is `Unknown`. Counts `ensure` calls.
    struct StubSupplier {
        ensure_calls: AtomicUsize,
    }
    struct StubResident(String);
    struct StubInst;
    impl ResidentModel for StubResident {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.0, "stub", vec![])
        }
        fn instance_key(&self, _action: &str, _inv: &capability::Invocation) -> InstanceKey {
            InstanceKey::new(self.0.clone(), "default")
        }
        fn estimate(&self, _key: &InstanceKey) -> MemCost {
            MemCost::default()
        }
        fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
            Ok(Box::new(StubInst))
        }
    }
    impl Instance for StubInst {
        fn run(&mut self, _action: &str, _inv: &capability::Invocation, _progress: &mut dyn FnMut(capability::Progress)) -> capability::ActionResult {
            Err("not implemented in this stub".into())
        }
    }
    impl residency::ModelSupplier for StubSupplier {
        fn classify(&self, model: &str) -> Supply {
            if model == "vendor/fetchable" { Supply::Fetchable } else { Supply::Unknown(format!("{model}: not in the stub catalog")) }
        }
        fn ensure(&self, model: &str, exec: &Executor, _progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
            self.ensure_calls.fetch_add(1, Ordering::SeqCst);
            exec.register(std::sync::Arc::new(StubResident(model.to_string())));
            Ok(())
        }
    }

    fn empty_exec() -> Executor {
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, GB, 0);
        Executor::start(vec![], budgets, Policy::default())
    }

    #[tokio::test]
    async fn ensure_resident_fetches_an_unresident_fetchable_model_exactly_once() {
        let supplier = std::sync::Arc::new(StubSupplier { ensure_calls: AtomicUsize::new(0) });
        let mgr = Manager::new(empty_exec()).with_supplier(Some(supplier.clone() as std::sync::Arc<dyn residency::ModelSupplier>));
        mgr.ensure_resident("vendor/fetchable").await.expect("must resolve after fetch");
        assert_eq!(supplier.ensure_calls.load(Ordering::SeqCst), 1);
        assert!(mgr.executor.manifests().iter().any(|m| m.model == "vendor/fetchable"));

        // A second call for the SAME model finds it already resident -- no
        // second `ensure` call.
        mgr.ensure_resident("vendor/fetchable").await.unwrap();
        assert_eq!(supplier.ensure_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn ensure_resident_never_calls_ensure_for_an_unknown_model() {
        let supplier = std::sync::Arc::new(StubSupplier { ensure_calls: AtomicUsize::new(0) });
        let mgr = Manager::new(empty_exec()).with_supplier(Some(supplier.clone() as std::sync::Arc<dyn residency::ModelSupplier>));
        let err = mgr.ensure_resident("brain/reserved-or-nonsense").await.unwrap_err();
        assert!(err.to_string().contains("no model"), "{err}");
        assert_eq!(supplier.ensure_calls.load(Ordering::SeqCst), 0, "classify-only Unknown must never call ensure");
    }

    #[tokio::test]
    async fn ensure_resident_with_no_supplier_is_the_plain_no_model_error() {
        let mgr = Manager::new(empty_exec());
        let err = mgr.ensure_resident("vendor/fetchable").await.unwrap_err();
        assert!(err.to_string().contains("no model"), "{err}");
    }

    #[tokio::test]
    async fn ensure_resident_does_not_leak_the_internal_fetch_error_reason() {
        struct AlwaysFails;
        impl residency::ModelSupplier for AlwaysFails {
            fn classify(&self, _model: &str) -> Supply {
                Supply::Fetchable
            }
            fn ensure(&self, _model: &str, _exec: &Executor, _progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
                Err("hub error: /data/workspace/secret-internal-path unreachable".to_string())
            }
        }
        let mgr = Manager::new(empty_exec()).with_supplier(Some(std::sync::Arc::new(AlwaysFails) as std::sync::Arc<dyn residency::ModelSupplier>));
        let err = mgr.ensure_resident("vendor/will-fail").await.unwrap_err().to_string();
        assert!(!err.contains("secret-internal-path"), "internal fetch error leaked: {err}");
    }
}
