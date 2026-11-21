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

/// This process's identity for its lifetime: a reader that sees it change knows the server restarted, so whatever
/// it cached (its KV, its weights) may differ from before.
fn instance() -> &'static str {
    static INSTANCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    INSTANCE.get_or_init(|| format!("{:016x}", rand::random::<u64>()))
}

/// `{"status", "instance", "decode_slots", "free_slots", "kv_free_tokens", "kv_total_tokens", "edge_concurrency",
/// "queue_depth", "served": {...}}`: the process's identity, the sequences the served models decode at once and how
/// many of those are free now, the KV tokens a new request can claim (free blocks and blocks only the prefix cache
/// holds, an upper bound) and the pool's total, summed over the paged text models (null when none is resident), the requests the surface admits at once before it sheds with 503, the jobs waiting for a lane now,
/// and the cumulative counts of finished generations, prompt, completion and prefix-cached prompt tokens (a reader
/// takes rates from two readings). Model counters are as of the dispatcher's last pass.
async fn health(State(state): State<AppState>) -> Json<Value> {
    let read = |c: &AtomicU64| c.load(Ordering::Relaxed);
    let stats = state.exec.stats();
    let sum = |name: &str| -> Option<u64> {
        let values: Vec<u64> = stats.metrics.values().filter_map(|m| m.iter().find(|(n, _)| n == name)).filter_map(|(_, v)| v.as_u64()).collect();
        (!values.is_empty()).then(|| values.iter().sum())
    };
    let slots = state.exec.decode_slots();
    let running = sum("running_sequences").unwrap_or(0) as usize;
    Json(json!({
        "status": "ok",
        "instance": instance(),
        "decode_slots": slots,
        "free_slots": slots.saturating_sub(running),
        "kv_free_tokens": sum("kv_free_tokens"),
        "kv_total_tokens": sum("kv_total_tokens"),
        "edge_concurrency": state.edge_concurrency,
        "queue_depth": stats.queue_depth,
        "served": {
            "requests": read(&SERVED.requests),
            "prompt_tokens": read(&SERVED.prompt_tokens),
            "completion_tokens": read(&SERVED.completion_tokens),
            "cached_prompt_tokens": read(&SERVED.cached_prompt_tokens),
        },
    }))
}
