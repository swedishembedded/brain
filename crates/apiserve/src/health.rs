// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference fleets that route
// around a busy or failed server for their clients. If your team needs
// expertise in multi-node LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `GET /health` (+ `/v1/health`): that the server answers, how much work it takes at once, how much waits, and what
//! it has served. A router in front of several servers (`crate::route`) reads it to keep an upstream in rotation, to
//! weigh its load against its capacity, and to report each upstream's recent token rate and prefix-cache hit rate.
//! Behind the key like every route; it names no model, path or device, and carries no request content.

use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::state::AppState;

/// The health routes (registered on every provider surface, outside the edge limit).
pub fn routes() -> Router<AppState> {
    Router::new().route("/health", get(health)).route("/v1/health", get(health))
}

/// What this process has served since it started, over every surface: finished generations and their tokens.
struct Served {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    cached_prompt_tokens: AtomicU64,
}

static SERVED: Served = Served { requests: AtomicU64::new(0), prompt_tokens: AtomicU64::new(0), completion_tokens: AtomicU64::new(0), cached_prompt_tokens: AtomicU64::new(0) };

/// Count one finished generation by the token counts its outcome reports (none counted when it reports none).
pub(crate) fn count_served(outcome: &capability::Outcome) {
    let tokens = |name: &str| outcome.outputs.get(name).and_then(serde_json::Value::as_u64).unwrap_or(0);
    SERVED.requests.fetch_add(1, Ordering::Relaxed);
    SERVED.prompt_tokens.fetch_add(tokens("prompt_tokens"), Ordering::Relaxed);
    SERVED.completion_tokens.fetch_add(tokens("completion_tokens"), Ordering::Relaxed);
    SERVED.cached_prompt_tokens.fetch_add(tokens("cached_tokens"), Ordering::Relaxed);
}

/// `{"status": "ok", "decode_slots": N, "edge_concurrency": M, "queue_depth": Q, "served": {...}}`: the sequences the
/// served models decode at once, the requests the surface admits at once before it sheds with 503, the jobs waiting
/// for a lane now, and the cumulative counts of finished generations, prompt, completion and prefix-cached prompt
/// tokens (a reader takes rates from two readings).
async fn health(State(state): State<AppState>) -> Json<Value> {
    let read = |c: &AtomicU64| c.load(Ordering::Relaxed);
    Json(json!({
        "status": "ok",
        "decode_slots": state.exec.decode_slots(),
        "edge_concurrency": state.edge_concurrency,
        "queue_depth": state.exec.stats().queue_depth,
        "served": {
            "requests": read(&SERVED.requests),
            "prompt_tokens": read(&SERVED.prompt_tokens),
            "completion_tokens": read(&SERVED.completion_tokens),
            "cached_prompt_tokens": read(&SERVED.cached_prompt_tokens),
        },
    }))
}
