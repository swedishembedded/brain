// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The OpenRouter-compatible surface. Same request grammar and chat handler as
//! OpenAI (it reuses [`crate::openai::handle_chat`]) with `native = true`, which
//! adds OpenRouter's `native_finish_reason` (mirroring `finish_reason`) and the
//! `system_fingerprint` its `ChatResult` requires. Embeddings and image generation
//! reuse the shared OpenAI handlers verbatim (identical request grammar).

use axum::body::Bytes;
use axum::extract::State;
use axum::response::Response;
use axum::routing::post;
use axum::Router;

use crate::openai;
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

/// `POST /chat/completions` — the OpenAI chat handler with OpenRouter's response
/// extras (`native_finish_reason`, `system_fingerprint`).
async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    openai::handle_chat(state, body, true).await
}

/// `POST /embeddings` — the shared OpenAI embeddings handler (OpenRouter uses the
/// identical `CreateEmbeddingRequest`/`CreateEmbeddingResponse` grammar).
async fn embeddings(State(state): State<AppState>, body: Bytes) -> Response {
    openai::handle_embeddings(state, body).await
}

/// `POST /images/generations` — the shared OpenAI image handler (OpenRouter uses the
/// identical `CreateImageRequest`/`ImagesResponse` grammar).
async fn images_generations(State(state): State<AppState>, body: Bytes) -> Response {
    openai::handle_images(state, body).await
}
