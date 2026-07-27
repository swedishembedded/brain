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
use std::os::fd::AsFd;
use std::sync::{Arc, Mutex};

use capability::{Blob, Invocation, Media, Outcome, Progress};
use residency::{Executor, Job};
use serde_json::{json, Value};
use zbus::fdo;
use zbus::zvariant::OwnedFd as ZOwnedFd;

use crate::fd::{bytes_to_fd, read_fd_to_vec};

pub struct Manager {
    executor: Executor,
    version: String,
}

impl Manager {
    pub fn new(executor: Executor) -> Manager {
        Manager { executor, version: env!("CARGO_PKG_VERSION").to_string() }
    }

    /// Assemble an [`Invocation`] from params JSON, input fds, and per-fd metadata.
    fn build_inv(&self, params: &str, in_fds: HashMap<String, ZOwnedFd>, in_meta: &str) -> Result<Invocation, String> {
        let params: Value = if params.trim().is_empty() { json!({}) } else { serde_json::from_str(params).map_err(|e| format!("params JSON: {e}"))? };
        let meta: Value = if in_meta.trim().is_empty() { json!({}) } else { serde_json::from_str(in_meta).map_err(|e| format!("in_meta JSON: {e}"))? };
        let mut inv = Invocation { params, blobs: Default::default() };
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
        let inv = self.build_inv(&params, in_fds, &in_meta).map_err(fdo::Error::Failed)?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.executor.submit(Job {
            model,
            action,
            inv,
            on_progress: Box::new(|_| {}), // Run is one-shot; no streaming
            reply: Box::new(move |r| {
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
        let inv = self.build_inv(&params, in_fds, &in_meta).map_err(fdo::Error::Failed)?;
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
        // A monotonically-unique-enough id for the client to correlate (the fd is the
        // real handle). The executor owns lifecycle; Cancel is phase 2.
        let job = self.executor.stats().jobs.wrapping_add(1);
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

    async fn cancel(&self, _job: u64) -> bool {
        false // phase 2
    }

    #[zbus(property)]
    async fn version(&self) -> String {
        self.version.clone()
    }

    #[zbus(property)]
    async fn active_jobs(&self) -> u32 {
        self.executor.stats().resident as u32
    }

    #[zbus(property)]
    async fn models(&self) -> Vec<String> {
        self.list_models().await
    }
}
