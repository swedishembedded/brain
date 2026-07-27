// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `com.swedishembedded.Brain1.Manager` D-Bus interface.
//!
//! Each method only **validates and translates**: it turns D-Bus args (+ input fds)
//! into a [`capability::Invocation`], hands a [`Cmd`] to the worker, and returns
//! promptly — no inference runs on the zbus dispatch task. `Run` awaits the worker's
//! one-shot reply and returns the result fds; `Subscribe` returns a stream fd
//! immediately and lets the worker fan events into it.

use std::collections::HashMap;
use std::os::fd::AsFd;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use capability::{Blob, Invocation, Media};
use serde_json::{json, Value};
use zbus::fdo;
use zbus::zvariant::OwnedFd as ZOwnedFd;

use crate::fd::read_fd_to_vec;
use crate::worker::{Cmd, Handle, Transport};

pub struct Manager {
    worker: Arc<Handle>,
    version: String,
}

impl Manager {
    pub fn new(worker: Arc<Handle>) -> Manager {
        Manager { worker, version: env!("CARGO_PKG_VERSION").to_string() }
    }

    /// Assemble an [`Invocation`] from params JSON, input fds, and per-fd metadata.
    /// Each input fd is mmap-read into a `Blob`; `in_meta[name].media` selects the
    /// `Media` (default `bytes`) and the whole `in_meta[name]` object is kept as the
    /// blob's `meta` (so `{w,h}` etc. reach the action).
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

#[zbus::interface(name = "com.swedishembedded.Brain1.Manager")]
impl Manager {
    /// JSON array of every model's manifest (models, actions, params) — discovery.
    async fn manifests(&self) -> String {
        self.worker.manifests_json.clone()
    }

    /// The list of served model names.
    async fn list_models(&self) -> Vec<String> {
        self.worker.models.clone()
    }

    /// Run one action. Input blobs arrive as fds (`in_fds` keyed by blob name) with
    /// `in_meta` describing each (`{name:{media,w,h,…}}`). Outputs come back as fds
    /// in `out_fds`; `out_meta` records each output's media, transport, and meta.
    /// `transport` requests `"memfd"` (default) or `"dmabuf"` (best-effort).
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
        let (rtx, rrx) = tokio::sync::oneshot::channel();
        self.worker
            .tx
            .send(Cmd::Run { model, action, inv, transport: Transport::parse(&transport), reply: rtx })
            .map_err(|_| fdo::Error::Failed("worker thread is gone".into()))?;
        let reply = rrx
            .await
            .map_err(|_| fdo::Error::Failed("worker dropped the reply".into()))?
            .map_err(fdo::Error::Failed)?;
        let out_fds: HashMap<String, ZOwnedFd> = reply.out_fds.into_iter().map(|(n, fd)| (n, fd.into())).collect();
        Ok((
            serde_json::to_string(&reply.result).unwrap_or_else(|_| "{}".into()),
            out_fds,
            serde_json::to_string(&reply.out_meta).unwrap_or_else(|_| "{}".into()),
        ))
    }

    /// Start a streaming run. Returns a job id and a `SOCK_SEQPACKET` fd that
    /// delivers framed `progress`/`blob`/`done`/`error` events (see [`crate::stream`]).
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
        let job = self.worker.next_job();
        self.worker
            .tx
            .send(Cmd::Subscribe { model, action, inv, stream })
            .map_err(|_| fdo::Error::Failed("worker thread is gone".into()))?;
        Ok((job, client.into()))
    }

    /// Cooperative cancel (phase 2 — currently a no-op returning false).
    async fn cancel(&self, _job: u64) -> bool {
        false
    }

    #[zbus(property)]
    async fn version(&self) -> String {
        self.version.clone()
    }

    #[zbus(property)]
    async fn active_jobs(&self) -> u32 {
        self.worker.active.load(Ordering::Relaxed)
    }

    #[zbus(property)]
    async fn models(&self) -> Vec<String> {
        self.worker.models.clone()
    }
}
