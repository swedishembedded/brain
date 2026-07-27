// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The inference worker: a dedicated thread that owns the [`capability::Registry`]
//! and runs the **blocking** action execution off the async/D-Bus threads.
//!
//! The D-Bus interface only validates a request, turns it into a [`Cmd`], and sends
//! it here; this thread runs it and replies (one-shot for `Run`, a stream for
//! `Subscribe`). One worker thread ⇒ jobs serialize — which is exactly right for a
//! single-GPU engine (two inferences cannot run at once anyway) and gives the
//! ordered, lock-free state the D-Bus guidance calls for.

use std::os::fd::AsFd;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use capability::{Invocation, Progress, Registry};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use crate::fd::bytes_to_fd;
use crate::stream::StreamTx;

/// Requested result-FD transport.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transport {
    Memfd,
    Dmabuf,
}

impl Transport {
    pub fn parse(s: &str) -> Transport {
        if s.eq_ignore_ascii_case("dmabuf") { Transport::Dmabuf } else { Transport::Memfd }
    }
    fn want_dmabuf(self) -> bool {
        self == Transport::Dmabuf
    }
}

/// A completed one-shot run: scalar result JSON + one fd per output blob + the
/// per-blob metadata (media, actual transport, blob meta).
pub struct RunReply {
    pub result: Value,
    pub out_fds: Vec<(String, std::os::fd::OwnedFd)>,
    pub out_meta: Value,
}

/// Commands the D-Bus interface sends to the worker.
pub enum Cmd {
    Run {
        model: String,
        action: String,
        inv: Invocation,
        transport: Transport,
        reply: oneshot::Sender<Result<RunReply, String>>,
    },
    Subscribe {
        model: String,
        action: String,
        inv: Invocation,
        stream: StreamTx,
    },
}

/// Handle the D-Bus layer holds: the command channel + cached discovery data.
pub struct Handle {
    pub tx: mpsc::UnboundedSender<Cmd>,
    pub manifests_json: String,
    pub models: Vec<String>,
    pub active: Arc<AtomicU32>,
    next_job: AtomicU64,
}

impl Handle {
    pub fn next_job(&self) -> u64 {
        self.next_job.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Spawn the worker over `registry`. Returns the [`Handle`] and the thread's join
/// handle (the thread ends when every `tx` sender is dropped).
pub fn spawn(registry: Registry) -> (Arc<Handle>, std::thread::JoinHandle<()>) {
    let manifests = registry.manifests();
    let manifests_json = serde_json::to_string(&Value::Array(manifests.iter().map(|m| m.to_json()).collect()))
        .unwrap_or_else(|_| "[]".into());
    let models: Vec<String> = manifests.iter().map(|m| m.model.clone()).collect();
    let active = Arc::new(AtomicU32::new(0));
    let (tx, mut rx) = mpsc::unbounded_channel::<Cmd>();

    let act = active.clone();
    let join = std::thread::Builder::new()
        .name("brain-dbus-worker".into())
        .spawn(move || {
            while let Some(cmd) = rx.blocking_recv() {
                match cmd {
                    Cmd::Run { model, action, inv, transport, reply } => {
                        act.fetch_add(1, Ordering::Relaxed);
                        let out = run_once(&registry, &model, &action, inv, transport);
                        act.fetch_sub(1, Ordering::Relaxed);
                        let _ = reply.send(out);
                    }
                    Cmd::Subscribe { model, action, inv, mut stream } => {
                        act.fetch_add(1, Ordering::Relaxed);
                        run_streaming(&registry, &model, &action, inv, &mut stream);
                        act.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }
        })
        .expect("spawn brain-dbus worker thread");

    (Arc::new(Handle { tx, manifests_json, models, active, next_job: AtomicU64::new(0) }), join)
}

/// Run an action to completion; package each output blob as an fd.
fn run_once(registry: &Registry, model: &str, action: &str, inv: Invocation, transport: Transport) -> Result<RunReply, String> {
    let outcome = registry.run(model, action, inv, &mut |_p| {})?;
    let mut out_fds = Vec::new();
    let mut meta = serde_json::Map::new();
    for (name, blob) in outcome.blobs {
        let (fd, actual) = bytes_to_fd(&name, &blob.bytes, transport.want_dmabuf()).map_err(|e| format!("blob {name}: {e}"))?;
        meta.insert(name.clone(), json!({"media": blob.media.name(), "transport": actual, "bytes": blob.bytes.len(), "meta": blob.meta}));
        out_fds.push((name, fd));
    }
    Ok(RunReply { result: outcome.outputs, out_fds, out_meta: Value::Object(meta) })
}

/// Run an action, streaming progress to `stream`, then the output blobs (as
/// out-of-band memfds) and a terminal `done`/`error` frame.
fn run_streaming(registry: &Registry, model: &str, action: &str, inv: Invocation, stream: &mut StreamTx) {
    let res = {
        let mut cb = |p: Progress| stream.progress(p.step, p.total, &p.message);
        registry.run(model, action, inv, &mut cb)
    };
    match res {
        Ok(outcome) => {
            for (name, blob) in &outcome.blobs {
                // Streaming always uses memfd (the SCM_RIGHTS out-of-band fd).
                match bytes_to_fd(name, &blob.bytes, false) {
                    Ok((fd, _)) => stream.blob(name, blob.media.name(), &blob.meta, fd.as_fd()),
                    Err(e) => stream.error(&format!("blob {name}: {e}")),
                }
            }
            stream.done(&outcome.outputs);
        }
        Err(e) => stream.error(&e),
    }
}
