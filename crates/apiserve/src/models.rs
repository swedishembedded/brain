// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `GET /models` (+ `/v1/models`) and `GET /models/{id}` — the model catalog in
//! each provider's dialect, built from `exec.manifests()` filtered by
//! [`catalog::exposed`]. No generation here; discovery only.

use axum::extract::{Path, State};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::catalog::{self, CapSet};
use crate::error::ApiError;
use crate::state::AppState;
use crate::surface::Provider;

/// Fixed creation instant advertised for every brain model (Unix seconds) — brain
/// serves live weights, not dated snapshots, so a stable value keeps responses
/// deterministic. 2026-01-01T00:00:00Z.
pub const CREATED_UNIX: i64 = 1_767_225_600;
/// The same instant as RFC3339, for Anthropic's `created_at`.
pub const CREATED_RFC3339: &str = "2026-01-01T00:00:00Z";

/// The `/models` + `/models/{id}` routes (registered on every provider surface).
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/models", get(list))
        .route("/v1/models", get(list))
        .route("/models/:id", get(get_one))
        .route("/v1/models/:id", get(get_one))
}

/// `GET /models` — the provider-shaped list of exposed models.
async fn list(State(state): State<AppState>) -> Json<Value> {
    let models = catalog::exposed(&state.exec, state.provider);
    let cards: Vec<Value> = models.iter().map(|(name, caps, ctx)| card(state.provider, name, *caps, *ctx)).collect();
    Json(envelope(state.provider, cards))
}

/// `GET /models/{id}` — one model card, or 404 if it is not exposed here.
async fn get_one(State(state): State<AppState>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let found = catalog::exposed(&state.exec, state.provider).into_iter().find(|(n, _, _)| *n == id);
    match found {
        Some((name, caps, ctx)) => Ok(Json(card(state.provider, &name, caps, ctx))),
        None => Err(ApiError::model_not_found(state.provider, &id)),
    }
}

/// Wrap the model cards in the provider's list envelope.
pub fn envelope(provider: Provider, cards: Vec<Value>) -> Value {
    match provider {
        Provider::OpenAI => json!({ "object": "list", "data": cards }),
        Provider::OpenRouter => json!({ "data": cards, "total_count": cards.len(), "links": { "next": Value::Null } }),
        Provider::Anthropic => json!({
            "data": cards,
            "has_more": false,
            "first_id": cards.first().and_then(|c| c.get("id").cloned()).unwrap_or(Value::Null),
            "last_id": cards.last().and_then(|c| c.get("id").cloned()).unwrap_or(Value::Null),
        }),
    }
}

/// Fallback advertised context length for a model that hasn't reported its
/// real capacity (non-chat models, or a resident kind that doesn't populate
/// `Manifest::max_context_tokens` yet). Deliberately conservative rather than
/// a large made-up number.
const FALLBACK_CONTEXT_LENGTH: u64 = 4096;

/// One model card in the provider's dialect. `max_context_tokens` is the
/// model's REAL current serving capacity from `Manifest::max_context_tokens`
/// (`None` for non-chat models or kinds that don't report it yet) — this is
/// what makes the advertised number trustworthy: a client that reads it and
/// builds a prompt within it should never be rejected by the actual engine.
pub fn card(provider: Provider, name: &str, _caps: CapSet, max_context_tokens: Option<u64>) -> Value {
    match provider {
        // The real OpenAI `Model` object has no context-length field at all
        // (see tests/specs/openai.json's `Model` schema) - this is an
        // additive, schema-safe (no `additionalProperties: false`) extension
        // for clients willing to read it, not a spec requirement.
        Provider::OpenAI => {
            let mut v = json!({
                "id": name,
                "object": "model",
                "created": CREATED_UNIX,
                "owned_by": "brain",
            });
            if let Some(ctx) = max_context_tokens {
                v["context_length"] = json!(ctx);
            }
            v
        }
        Provider::Anthropic => json!({
            "type": "model",
            "id": name,
            "display_name": name,
            "created_at": CREATED_RFC3339,
        }),
        Provider::OpenRouter => openrouter_card(name, max_context_tokens),
    }
}

/// OpenRouter's richer model card (validates against its `Model` schema).
/// `context_length` is a real, documented field of this dialect (unlike
/// OpenAI's) - use the model's actual capacity when known, the conservative
/// fallback otherwise.
fn openrouter_card(name: &str, max_context_tokens: Option<u64>) -> Value {
    let ctx = max_context_tokens.unwrap_or(FALLBACK_CONTEXT_LENGTH);
    json!({
        "id": name,
        "canonical_slug": name,
        "name": name,
        "created": CREATED_UNIX,
        "description": format!("brain-served model '{name}'"),
        "context_length": ctx,
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tokenizer": "Other",
            "instruct_type": Value::Null,
        },
        "pricing": { "prompt": "0", "completion": "0", "request": "0", "image": "0" },
        "top_provider": { "context_length": ctx, "max_completion_tokens": Value::Null, "is_moderated": false },
        "per_request_limits": Value::Null,
        "supported_parameters": ["temperature", "top_p", "max_tokens"],
        "default_parameters": Value::Null,
        "supported_voices": Value::Null,
        "links": { "details": format!("/api/v1/models/{name}/endpoints") },
    })
}
