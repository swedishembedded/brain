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
use residency::{Executor, Job};
use serde_json::{json, Value};
use zbus::fdo;
use zbus::zvariant::OwnedFd as ZOwnedFd;

use crate::fd::{bytes_to_fd, read_fd_to_vec};
use crate::stream::StreamTx;

/// Armed cancel tokens for in-flight jobs, keyed by job id. An entry lives from
/// submission until the job's reply fires, so `Cancel` can find any running job.
type JobRegistry = Arc<Mutex<HashMap<u64, CancelToken>>>;

pub struct Manager {
    executor: Executor,
    version: String,
    jobs: JobRegistry,
    next_job: AtomicU64,
}

impl Manager {
    pub fn new(executor: Executor) -> Manager {
        Manager { executor, version: env!("CARGO_PKG_VERSION").to_string(), jobs: Arc::new(Mutex::new(HashMap::new())), next_job: AtomicU64::new(0) }
    }

    /// Arm `inv` with a fresh [`CancelToken`] and register it under a new job id.
    /// The caller must remove the entry (via [`finish_job`]) when the reply fires.
    fn register_job(&self, inv: &mut Invocation) -> u64 {
        let token = CancelToken::armed();
        inv.cancel = token.clone();
        let job = self.next_job.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut m) = self.jobs.lock() {
            m.insert(job, token);
        }
        job
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
    executor.submit(Job {
        model: model.to_string(),
        action: action.into(),
        inv,
        on_progress: Box::new(|_| {}),
        reply: Box::new(move |r| {
            let _ = tx.send(r);
        }),
    });
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
    if let Ok(mut m) = jobs.lock() {
        m.remove(&job);
    }
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
        let mut inv = self.build_inv(&params, in_fds, &in_meta).map_err(fdo::Error::Failed)?;
        // Armed so ActiveJobs counts it; a Run has no client-visible job id, so its
        // token is only ever dropped here (the reply), never cancelled by Cancel.
        let job = self.register_job(&mut inv);
        let jobs = self.jobs.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.executor.submit(Job {
            model,
            action,
            inv,
            on_progress: Box::new(|_| {}), // Run is one-shot; no streaming
            reply: Box::new(move |r| {
                finish_job(&jobs, job);
                let _ = tx.send(r);
            }),
        });
        let outcome = rx.await.map_err(|_| fdo::Error::Failed("executor dropped the reply".into()))?.map_err(fdo::Error::Failed)?;
        let want_dmabuf = transport.eq_ignore_ascii_case("dmabuf");
        let (out_fds, out_meta) = outcome_to_fds(&outcome, want_dmabuf).map_err(fdo::Error::Failed)?;
        Ok((serde_json::to_string(&outcome.outputs).unwrap_or_else(|_| "{}".into()), out_fds, serde_json::to_string(&out_meta).unwrap_or_else(|_| "{}".into())))
    }

    /// Start a streaming run. Returns a job id + a SEQPACKET fd delivering framed
    /// progress/blob/done/error events.
    #[zbus(out_args("job", "event_fd"))]
    async fn subscribe(
        &self,
        model: String,
        action: String,
        params: String,
        in_fds: HashMap<String, ZOwnedFd>,
        in_meta: String,
    ) -> fdo::Result<(u64, ZOwnedFd)> {
        let mut inv = self.build_inv(&params, in_fds, &in_meta).map_err(fdo::Error::Failed)?;
        // Arm a cancel token under the returned job id: `Cancel(job)` flips it and
        // the running action aborts at its next poll (see docs/serving-contract.md).
        let job = self.register_job(&mut inv);
        let jobs = self.jobs.clone();
        let (stream, client) = crate::stream::pair().map_err(|e| fdo::Error::Failed(e.to_string()))?;
        let stream = Arc::new(Mutex::new(stream));
        let (sp, sr) = (stream.clone(), stream.clone());
        self.executor.submit(Job {
            model,
            action,
            inv,
            on_progress: Box::new(move |p: Progress| {
                if let Ok(mut s) = sp.lock() {
                    s.progress(p.step, p.total, &p.message);
                }
            }),
            reply: Box::new(move |r| {
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

    /// Cooperatively cancel a job (from `Subscribe`) by id: flips its token so the
    /// running action aborts at its next poll (`Err("cancelled")` arrives as an
    /// `error` frame). Returns `true` if the job was found still in flight,
    /// `false` for an unknown or already-finished id.
    async fn cancel(&self, job: u64) -> bool {
        match self.jobs.lock() {
            Ok(m) => m.get(&job).map(|t| t.cancel()).is_some(),
            Err(_) => false,
        }
    }

    #[zbus(property)]
    async fn version(&self) -> String {
        self.version.clone()
    }

    /// Jobs currently in flight (submitted via `Run`/`Subscribe`, reply not yet fired).
    #[zbus(property)]
    async fn active_jobs(&self) -> u32 {
        self.jobs.lock().map(|m| m.len()).unwrap_or(0) as u32
    }

    #[zbus(property)]
    async fn models(&self) -> Vec<String> {
        self.list_models().await
    }
}
