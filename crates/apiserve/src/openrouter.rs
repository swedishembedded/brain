// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The OpenRouter-compatible surface. Same request grammar as OpenAI (chat /
//! embeddings / images) with OpenRouter's richer model cards and error envelope.
//! P4 registers routes and returns a spec-valid `not_implemented` error; mappings
//! are stubbed for P5+.

use axum::extract::State;
use axum::routing::post;
use axum::Router;
use capability::{Invocation, Outcome};
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::state::AppState;

/// OpenRouter chat/embeddings/image routes (merged onto the shared `/models` router).
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/images/generations", post(images_generations))
        .route("/images/generations", post(images_generations))
}

/// `POST /chat/completions` — 501 until P5.
async fn chat_completions(State(state): State<AppState>) -> ApiError {
    ApiError::not_implemented(state.provider, "POST /chat/completions is not implemented yet")
}

/// `POST /embeddings` — 501 until P5.
async fn embeddings(State(state): State<AppState>) -> ApiError {
    ApiError::not_implemented(state.provider, "POST /embeddings is not implemented yet")
}

/// `POST /images/generations` — 501 until P5.
async fn images_generations(State(state): State<AppState>) -> ApiError {
    ApiError::not_implemented(state.provider, "POST /images/generations is not implemented yet")
}

/// Map an OpenRouter request body to `(model, action, invocation)`. Filled in P5.
pub fn to_invocation(_body: &Value) -> Result<(String, String, Invocation), ApiError> {
    todo!("P5: map OpenRouter request -> capability::Invocation")
}

/// Map an executor [`Outcome`] back to an OpenRouter response body. Filled in P5.
pub fn from_outcome(_o: &Outcome, _model: &str) -> Value {
    let _ = json!({});
    todo!("P5: map capability::Outcome -> OpenRouter response")
}
