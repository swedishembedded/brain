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
use residency::{Job, Supply};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::error::ApiError;
use crate::state::{AppState, JobRegistry};
use crate::surface::Provider;

/// Auto-fetch entry point: `model` didn't resolve against `resolve` (the caller's
/// `catalog::resolve_*` predicate, already tried once). If `state.supplier` can
/// classify `model` as fetchable, blocks (off the async runtime, via
/// `spawn_blocking` — a cold fetch can take minutes) until it's on disk and
/// registered, then retries `resolve`. Returns a `model_not_found` for every other
/// outcome (no supplier configured, `Unknown`, or the fetch itself failing) —
/// deliberately the SAME error as an unknown model, so a failed/refused fetch
/// tells a client nothing more than "this model isn't available" (see
/// `docs/api-security-audit.md`'s error-hygiene section: hub URLs, filesystem
/// paths, and other fetch-internal detail are logged server-side, never reflected
/// to the caller).
pub async fn ensure_and_recheck<T>(state: &AppState, provider: Provider, model: &str, resolve: impl Fn(&str) -> Option<T>) -> Result<T, ApiError> {
    let Some(supplier) = state.supplier.clone() else {
        return Err(ApiError::model_not_found(provider, model));
    };
    match supplier.classify(model) {
        Supply::Fetchable => {}
        Supply::Resident | Supply::Unknown(_) => return Err(ApiError::model_not_found(provider, model)),
    }
    let exec = state.exec.clone();
    let model_owned = model.to_string();
    let outcome = tokio::task::spawn_blocking(move || exec.ensure_model(&model_owned, supplier.as_ref(), &mut |_, _, _| {})).await;
    match outcome {
        Ok(Ok(())) => resolve(model).ok_or_else(|| ApiError::model_not_found(provider, model)),
        Ok(Err(reason)) => {
            eprintln!("apiserve: auto-fetch {model}: {reason}");
            Err(ApiError::model_not_found(provider, model))
        }
        Err(join_err) => {
            eprintln!("apiserve: auto-fetch {model}: blocking task failed: {join_err}");
            Err(ApiError::model_not_found(provider, model))
        }
    }
}

/// Read the contract's `generate` [`Outcome`] into `(text, prompt_tokens,
/// completion_tokens, finish_reason)`. `text` is the `text` blob (full assistant
/// text); the counts + `finish_reason` come from the `outputs` object. Missing
/// fields degrade gracefully (empty text, zero counts, `"stop"`). Kept as the
/// smaller reader every existing caller (image generation's non-chat callers never
/// use it; Anthropic — no tool-calling support yet — still does) uses unchanged;
/// [`read_chat_outcome`] is an additive sibling for callers that also need
/// reasoning/tool_calls, not a replacement.
pub fn read_outcome(o: &Outcome) -> (String, i64, i64, String) {
    let text = o.blobs.get("text").map(|b| String::from_utf8_lossy(&b.bytes).into_owned()).unwrap_or_default();
    let prompt = o.outputs.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let completion = o.outputs.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let finish = o.outputs.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("stop").to_string();
    (text, prompt, completion, finish)
}

/// A generation outcome's full chat shape, read from the contract `Outcome`:
/// everything [`read_outcome`] reads, plus `reasoning` (the model's
/// `<think>…</think>` text, empty when it didn't reason) and `tool_calls` (one
/// JSON object per call — `{"id","name","arguments"}`, `arguments` a raw JSON-text
/// string, never re-parsed — from `outputs.tool_calls`'s JSON-array-string;
/// empty when absent). `text` is VISIBLE content only — the resident layer
/// (`crates/cli/src/resident_llm.rs::QwenInstance::run`,
/// `resident_mock.rs::generate_tool_call`) guarantees `<think>`/`<tool_call>`
/// markup never reaches the `text` blob.
pub struct ChatOutcome {
    pub text: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub finish: String,
    pub reasoning: String,
    pub tool_calls: Vec<serde_json::Value>,
}

/// Read a [`ChatOutcome`] from the contract `Outcome`. See [`read_outcome`] for the
/// smaller (pre-tool-calling) shape this extends.
pub fn read_chat_outcome(o: &Outcome) -> ChatOutcome {
    let (text, prompt_tokens, completion_tokens, finish) = read_outcome(o);
    let reasoning = o.outputs.get("reasoning_content").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let tool_calls = o
        .outputs
        .get("tool_calls")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    ChatOutcome { text, prompt_tokens, completion_tokens, finish, reasoning, tool_calls }
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

/// One `StreamMsg::Fetching` tick: a human-readable phase description plus the
/// raw byte counts (mirrors `residency::Executor::ensure_model`'s progress
/// callback shape: `name`, `got`, `total`). Rendered as an SSE COMMENT line
/// (`: …`, RFC-legal, ignored by every conformant parser) so a cold auto-fetch
/// keeps the connection alive and visibly progressing through a multi-minute
/// download instead of holding the response back in silence — without adding
/// a new `data:` event shape that would break the vendored-OpenAPI/Anthropic
/// conformance tests.
pub struct FetchProgress {
    pub message: String,
    pub got: u64,
    pub total: Option<u64>,
}

impl FetchProgress {
    /// Render as SSE COMMENT-line text (the caller wraps it in
    /// `axum::response::sse::Event::comment`) -- e.g. `BRAIN fetching Qwen/
    /// Qwen3-0.6B: model.safetensors 34% (512/1503 KiB)` with a known total,
    /// `BRAIN fetching …: model.safetensors (512 KiB)` without one (some hub
    /// responses omit `Content-Length`).
    pub fn comment_text(&self) -> String {
        match self.total {
            Some(total) if total > 0 => {
                let pct = (self.got.min(total).saturating_mul(100) / total).min(100);
                format!("BRAIN {} {pct}% ({}/{} KiB)", self.message, self.got / 1024, total / 1024)
            }
            _ => format!("BRAIN {} ({} KiB)", self.message, self.got / 1024),
        }
    }
}

/// One item on a streaming generation: an incremental text piece, a structured
/// out-of-band event (reasoning/tool-call progress — see [`capability::Progress::event`]
/// and `resident_llm.rs::emit_chat_events`'s neutral `{"kind":...}` shapes; chat
/// streaming renders these into `delta.reasoning_content`/`delta.tool_calls`,
/// NEVER into `delta.content` — that structural separation is what keeps raw
/// `<think>`/`<tool_call>` markup from ever leaking into plain content deltas), a
/// coarse `(step, total)` progress tick (used by image generation's denoise loop;
/// chat streaming ignores it), a cold-auto-fetch progress tick (see
/// [`FetchProgress`] — only ever produced by [`stream_with_autofetch`]), the
/// terminal outcome (full text/image + counts), or an error.
pub enum StreamMsg {
    Delta(String),
    Event(serde_json::Value),
    Progress(u32, u32),
    Fetching(FetchProgress),
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
            } else if let Some(ev) = p.event {
                let _ = tx_progress.send(StreamMsg::Event(ev));
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

/// Like [`stream`]/[`stream_progress`], but for a `model` that is NOT already
/// resident: returns the [`EventStream`] IMMEDIATELY (before any network I/O),
/// then — in a background task — classifies and (if `Fetchable`) fetches the
/// model with its progress forwarded as [`StreamMsg::Fetching`] ticks, and only
/// once it's ready submits the real job under the normal admission race,
/// continuing to forward `Delta`/`Progress`/`Done`/`Err` into the SAME stream.
/// The caller (e.g. `openai::stream_chat_with_autofetch`) is expected to have
/// already confirmed `state.supplier.is_some()` and classified `Fetchable`
/// with zero I/O (`ModelSupplier::classify`) BEFORE calling this — an `Unknown`
/// or no-supplier model must still be a plain 404 with no SSE body opened at
/// all, exactly like [`crate::bridge::ensure_and_recheck`]'s non-streaming
/// sibling.
///
/// A 429 (admission failed after the fetch completed) becomes a terminal
/// [`StreamMsg::Err`] here, not an HTTP 429 status — the SSE body's headers
/// are already committed by the time admission is even attempted.
pub fn stream_with_autofetch(state: &AppState, supplier: std::sync::Arc<dyn residency::ModelSupplier>, model: &str, action: &str, inv: Invocation, forward_steps: bool) -> EventStream {
    let provider = state.provider;
    let (id, token) = state.register();
    let (tx, rx) = mpsc::unbounded_channel::<StreamMsg>();
    let exec = state.exec.clone();
    let jobs = state.jobs.clone();
    let admit_deadline = state.admit_deadline;
    let model_owned = model.to_string();
    let action_owned = action.to_string();
    let mut inv = inv;
    inv.cancel = token.clone();
    let guard_token = token.clone();

    tokio::spawn(async move {
        let tx_fetch = tx.clone();
        let fetch_model = model_owned.clone();
        let fetch_exec = exec.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            fetch_exec.ensure_model(&fetch_model, supplier.as_ref(), &mut |name, got, total| {
                let _ = tx_fetch.send(StreamMsg::Fetching(FetchProgress {
                    message: format!("fetching {fetch_model}: {name}").replace(['\n', '\r'], " "),
                    got: got as u64,
                    total: if total == 0 { None } else { Some(total as u64) },
                }));
            })
        })
        .await;
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => {
                eprintln!("apiserve: auto-fetch {model_owned}: {reason}");
                let _ = tx.send(StreamMsg::Err(ApiError::model_not_found(provider, &model_owned)));
                if let Ok(mut m) = jobs.lock() { m.remove(&id); }
                return;
            }
            Err(join_err) => {
                eprintln!("apiserve: auto-fetch {model_owned}: blocking task failed: {join_err}");
                let _ = tx.send(StreamMsg::Err(ApiError::model_not_found(provider, &model_owned)));
                if let Ok(mut m) = jobs.lock() { m.remove(&id); }
                return;
            }
        }

        // Model is ready -- submit the real job, forwarding into the SAME
        // channel the fetch ticks just used, so the client sees one
        // continuous stream (fetch progress, then generation deltas). A
        // dedicated clone for the admission-timeout branch below: the other
        // two are moved into the job's closures and won't be available if
        // admission times out before either ever fires.
        let tx_timeout = tx.clone();
        let (admit_tx, admit_rx) = oneshot::channel::<()>();
        let tx_progress = tx.clone();
        let tx_reply = tx;
        let job = Job::new(model_owned.clone(), action_owned, inv)
            .on_progress(move |p| {
                if let Some(piece) = p.delta {
                    let _ = tx_progress.send(StreamMsg::Delta(piece));
                } else if let Some(ev) = p.event {
                    let _ = tx_progress.send(StreamMsg::Event(ev));
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
                let _ = tx_reply.send(msg);
            });
        exec.submit(job);

        tokio::select! {
            _ = admit_rx => {}
            _ = tokio::time::sleep(admit_deadline) => {
                token.cancel();
                if let Ok(mut m) = jobs.lock() {
                    m.remove(&id);
                }
                let _ = tx_timeout.send(StreamMsg::Err(ApiError::overloaded(provider, "request could not be admitted within the deadline")));
            }
        }
    });

    EventStream { rx, _guard: CancelGuard { token: guard_token, jobs: state.jobs.clone(), id } }
}
