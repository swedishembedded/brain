// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference fleets that route
// around a busy or failed server for their clients. If your team needs
// expertise in multi-node LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `GET /health` (+ `/v1/health`): that the server answers, and how much work it takes at once. A router in front of
//! several servers (`crate::route`) reads it to keep an upstream in rotation and to weigh its load against its
//! capacity. Behind the key like every route; it names no model, path or device.

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::state::AppState;

/// The health routes (registered on every provider surface, outside the edge limit).
pub fn routes() -> Router<AppState> {
    Router::new().route("/health", get(health)).route("/v1/health", get(health))
}

/// `{"status": "ok", "decode_slots": N, "edge_concurrency": M}`: the sequences the served models decode at once, and
/// the requests the surface admits at once before it sheds with 503.
async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({"status": "ok", "decode_slots": state.exec.decode_slots(), "edge_concurrency": state.edge_concurrency}))
}
