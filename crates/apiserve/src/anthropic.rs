// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Anthropic Messages surface. P4 registers the routes and returns a
//! spec-valid `not_implemented` error; the request→[`capability::Invocation`] and
//! [`capability::Outcome`]→response mappings are stubbed with real signatures to
//! be filled in P5+.

use axum::extract::State;
use axum::routing::post;
use axum::Router;
use capability::{Invocation, Outcome};
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::state::AppState;

/// Anthropic-specific routes (merged onto the shared `/models` router).
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
}

/// `POST /v1/messages` — 501 until P5 wires generation onto the executor.
async fn messages(State(state): State<AppState>) -> ApiError {
    ApiError::not_implemented(state.provider, "POST /v1/messages is not implemented yet")
}

/// `POST /v1/messages/count_tokens` — 501 for now.
async fn count_tokens(State(state): State<AppState>) -> ApiError {
    ApiError::not_implemented(state.provider, "POST /v1/messages/count_tokens is not implemented yet")
}

/// Map an Anthropic Messages request body to `(model, action, invocation)` for the
/// executor. Filled in P5. Returns the model id, the capability action name, and
/// the built invocation.
pub fn to_invocation(_body: &Value) -> Result<(String, String, Invocation), ApiError> {
    todo!("P5: map Anthropic Messages request -> capability::Invocation")
}

/// Map an executor [`Outcome`] back to an Anthropic Messages response body. Filled
/// in P5.
pub fn from_outcome(_o: &Outcome, _model: &str) -> Value {
    let _ = json!({});
    todo!("P5: map capability::Outcome -> Anthropic Messages response")
}
