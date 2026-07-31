// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The async→sync seam between the HTTP handlers (Tokio tasks) and the shared,
//! synchronous [`residency::Executor`]. Mirrors `crates/dbus/src/service.rs`:
//! a handler only builds a [`capability::Invocation`], submits a [`residency::Job`],
//! and waits for the reply/progress — the model never runs on the HTTP task.
//!
//! Two shapes:
//! - [`submit`] — one-shot: arm a cancel token, submit, await the single reply
//!   (non-streaming chat).
//! - [`stream`] — streaming: the job's `on_progress`/`reply` callbacks push
//!   [`StreamMsg`]s onto an unbounded channel, returned as a [`futures::Stream`].
//!   The closures only ever touch the channel (`Send`), NEVER the model/GPU. A
//!   [`CancelGuard`] rides inside the returned stream so a dropped SSE response
//!   (client disconnect) cancels the running generation and clears the registry.

use std::pin::Pin;
use std::task::{Context, Poll};

use capability::{Invocation, Outcome};
use futures::Stream;
use residency::Job;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::error::ApiError;
use crate::state::{AppState, JobRegistry};
use crate::surface::Provider;

/// Read the contract's `generate` [`Outcome`] into `(text, prompt_tokens,
/// completion_tokens, finish_reason)`. `text` is the `text` blob (full assistant
/// text); the counts + `finish_reason` come from the `outputs` object. Missing
/// fields degrade gracefully (empty text, zero counts, `"stop"`).
pub fn read_outcome(o: &Outcome) -> (String, i64, i64, String) {
    let text = o.blobs.get("text").map(|b| String::from_utf8_lossy(&b.bytes).into_owned()).unwrap_or_default();
    let prompt = o.outputs.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let completion = o.outputs.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let finish = o.outputs.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("stop").to_string();
    (text, prompt, completion, finish)
}

/// Map an executor reply error string to a provider-shaped [`ApiError`]. An unknown
/// model/action surfaces as `model_not_found` (model resolution already happens in the
/// handler, so this is a belt-and-braces fallback). Every *other* reply error is a
/// server-side runtime/activation failure (backend/device errors, and potentially
/// on-disk model paths in the message): its raw text is NEVER reflected to the client.
/// The detail is logged for the operator; the client gets a generic message.
pub fn map_reply_err(provider: Provider, model: &str, e: &str) -> ApiError {
    if e.starts_with("no model") || e.contains("no action") {
        ApiError::model_not_found(provider, model)
    } else {
        // Do not leak internal error strings (paths, panic text, backend internals)
        // to callers; log server-side, return a generic message.
        eprintln!("apiserve: model '{model}' request failed: {e}");
        ApiError::invalid_request(provider, "the model failed to process the request")
    }
}

/// Submit one `generate` job and block (async) for its single reply. Arms a fresh
/// cancel token, registers it, then runs the **admission race**: a bounded wait
/// (`state.admit_deadline`) for the job to be ADMITTED (claimed onto a lane — its
/// `on_admit` signal fires). If it is not admitted in time the job is cancelled
/// (dropped from the queue so it never runs wastefully), unregistered, and the
/// caller gets a 429. Once admitted, the wait for the (possibly long) reply is
/// UNBOUNDED — only the wait-to-start is deadlined.
pub async fn submit(state: &AppState, model: &str, action: &str, mut inv: Invocation) -> Result<Outcome, ApiError> {
    let provider = state.provider;
    let (id, token) = state.register();
    inv.cancel = token.clone();
    let (tx, rx) = oneshot::channel();
    let (admit_tx, admit_rx) = oneshot::channel::<()>();
    let job = Job::new(model.to_string(), action.to_string(), inv)
        .on_progress(|_| {})
        .on_admit(move || {
            let _ = admit_tx.send(());
        })
        .reply(move |r| {
            let _ = tx.send(r);
        });
    state.exec.submit(job);

    // Admission race: work started on a lane vs. the deadline elapsing.
    tokio::select! {
        _ = admit_rx => {}
        _ = tokio::time::sleep(state.admit_deadline) => {
            token.cancel();
            state.finish(&id);
            return Err(ApiError::overloaded(provider, "request could not be admitted within the deadline"));
        }
    }

    // Admitted — await the reply (unbounded; a running job may take long).
    let res = rx.await;
    state.finish(&id);
    match res {
        Ok(Ok(outcome)) => Ok(outcome),
        Ok(Err(e)) => Err(map_reply_err(provider, model, &e)),
        Err(_) => Err(ApiError::overloaded(provider, "executor dropped the reply")),
    }
}

/// One item on a streaming generation: an incremental text piece, a coarse
/// `(step, total)` progress tick (used by image generation's denoise loop; chat
/// streaming ignores it), the terminal outcome (full text/image + counts), or an
/// error.
pub enum StreamMsg {
    Delta(String),
    Progress(u32, u32),
    Done(Outcome),
    Err(ApiError),
}

/// Dropped when the returned [`EventStream`] is dropped (SSE finished OR the client
/// disconnected): cancels the job's token and clears its registry entry, so a
/// disconnect aborts the running generation (cancel-on-disconnect).
pub struct CancelGuard {
    token: capability::CancelToken,
    jobs: JobRegistry,
    id: Uuid,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.token.cancel();
        if let Ok(mut m) = self.jobs.lock() {
            m.remove(&self.id);
        }
    }
}

/// The receiver side of a streaming generation plus its [`CancelGuard`]. Yields
/// [`StreamMsg`]s until the reply's `Done`/`Err`; dropping it cancels the job.
pub struct EventStream {
    rx: mpsc::UnboundedReceiver<StreamMsg>,
    _guard: CancelGuard,
}

impl Stream for EventStream {
    type Item = StreamMsg;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<StreamMsg>> {
        self.rx.poll_recv(cx)
    }
}

/// Submit one `generate` job whose progress/reply fan into a channel, running the
/// same **admission race** as [`submit`] BEFORE returning the stream: a bounded wait
/// (`state.admit_deadline`) for the job to be claimed onto a lane. If it is not
/// admitted in time the job is cancelled + unregistered and a 429 [`ApiError`] is
/// returned — so the caller returns a plain 429 body, NOT an SSE stream that then
/// errors. Once admitted, the returned [`EventStream`] yields a [`StreamMsg::Delta`]
/// per generated token (progress updates whose `delta` is set — coarse step-only
/// progress is ignored), then one terminal [`StreamMsg::Done`]/[`StreamMsg::Err`].
/// The `on_progress`/`reply` closures only touch the (`Send`) channel — never the
/// model.
pub async fn stream(state: &AppState, model: &str, action: &str, inv: Invocation) -> Result<EventStream, ApiError> {
    stream_inner(state, model, action, inv, false).await
}

/// Like [`stream`], but also forwards coarse `(step, total)` progress ticks as
/// [`StreamMsg::Progress`] (a denoise loop reports these with no `delta`). Used by
/// image generation, whose "progress" is denoise steps, not token deltas.
pub async fn stream_progress(state: &AppState, model: &str, action: &str, inv: Invocation) -> Result<EventStream, ApiError> {
    stream_inner(state, model, action, inv, true).await
}

/// Shared implementation of [`stream`]/[`stream_progress`]. `forward_steps` decides
/// whether a `delta`-less progress update becomes a [`StreamMsg::Progress`] tick
/// (image denoise) or is dropped (chat, which streams only token deltas).
async fn stream_inner(state: &AppState, model: &str, action: &str, mut inv: Invocation, forward_steps: bool) -> Result<EventStream, ApiError> {
    let provider = state.provider;
    let model_owned = model.to_string();
    let (id, token) = state.register();
    inv.cancel = token.clone();
    let (tx, rx) = mpsc::unbounded_channel::<StreamMsg>();
    let (admit_tx, admit_rx) = oneshot::channel::<()>();
    let tx_progress = tx.clone();
    let job = Job::new(model.to_string(), action.to_string(), inv)
        .on_progress(move |p| {
            if let Some(piece) = p.delta {
                let _ = tx_progress.send(StreamMsg::Delta(piece));
            } else if forward_steps {
                let _ = tx_progress.send(StreamMsg::Progress(p.step, p.total));
            }
        })
        .on_admit(move || {
            let _ = admit_tx.send(());
        })
        .reply(move |r| {
            let msg = match r {
                Ok(outcome) => StreamMsg::Done(outcome),
                Err(e) => StreamMsg::Err(map_reply_err(provider, &model_owned, &e)),
            };
            let _ = tx.send(msg);
        });
    state.exec.submit(job);

    // Admission race BEFORE returning the SSE body: a shed request yields a plain
    // 429, never an event-stream that immediately errors.
    tokio::select! {
        _ = admit_rx => {}
        _ = tokio::time::sleep(state.admit_deadline) => {
            token.cancel();
            state.finish(&id);
            return Err(ApiError::overloaded(provider, "request could not be admitted within the deadline"));
        }
    }

    Ok(EventStream { rx, _guard: CancelGuard { token, jobs: state.jobs.clone(), id } })
}
