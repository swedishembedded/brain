// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `GET /models` (+ `/v1/models`) and `GET /models/{id}` — the model catalog in
//! each provider's dialect, built from `exec.manifests()` filtered by
//! [`catalog::exposed`]. No generation here; discovery only.

use axum::extract::{Path, State};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use std::collections::HashSet;

use residency::{AdapterIdentity, Tier};

use crate::catalog::{self, CapSet, ModelInfo};
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
        .route("/models/{id}", get(get_one))
        .route("/v1/models/{id}", get(get_one))
}

/// `GET /models` — the provider-shaped list of exposed models.
async fn list(State(state): State<AppState>) -> Json<Value> {
    let models = catalog::exposed(&state.exec, state.provider);
    let resident = resident_models(&state);
    let cards: Vec<Value> = models.iter().filter(|info| state.lists(&info.name)).map(|info| card_of(&state, info, &resident)).collect();
    Json(envelope(state.provider, cards))
}

/// `GET /models/{id}` — one model card, or 404 if it is not exposed here.
async fn get_one(State(state): State<AppState>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let found = catalog::exposed(&state.exec, state.provider).into_iter().find(|info| info.name == id && state.lists(&info.name));
    match found {
        Some(info) => Ok(Json(card_of(&state, &info, &resident_models(&state)))),
        None => Err(ApiError::model_not_found(state.provider, &id)),
    }
}

/// `info`'s card with the adapter currently attached to it, read at request time
/// from the executor (a hot swap changes it without re-registering the model), and
/// on the OpenAI and OpenRouter dialects what serving it takes here: the sequences
/// it decodes at once (`decode_slots`, absent when unknown) and whether it is built
/// on a device now (`resident`). A router in front of several
/// servers places a model's requests by these; `/health` sums every model.
fn card_of(state: &AppState, info: &ModelInfo, resident: &HashSet<String>) -> Value {
    let mut card = card(state.provider, info, state.exec.served_adapter(&info.name).as_ref());
    if state.provider != Provider::Anthropic {
        if let Some(slots) = state.exec.model_decode_slots(&info.name) {
            card["decode_slots"] = json!(slots);
        }
        card["resident"] = json!(resident.contains(&info.name));
    }
    card
}

/// The models with an instance built on a device now (one dispatcher round trip).
fn resident_models(state: &AppState) -> HashSet<String> {
    let report = state.exec.residency();
    let hot = report.placements.iter().filter(|p| p.tier == Tier::Hot).map(|p| &p.key.model);
    let hot_multi = report.multi_placements.iter().filter(|p| p.tier == Tier::Hot).map(|p| &p.key.model);
    hot.chain(hot_multi).cloned().collect()
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
///
/// OpenAI and OpenRouter cards also carry the additive fields `capabilities`,
/// `max_output_tokens`, `adapter`, `support` and `zero_data_retention` (see
/// [`add_discovery`]); every dialect's card names the sampling parameters the
/// model honours (`supported_parameters`, see [`add_sampling`]).
pub fn card(provider: Provider, info: &ModelInfo, adapter: Option<&AdapterIdentity>) -> Value {
    let name = info.name.as_str();
    let mut card = match provider {
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
            if let Some(ctx) = info.max_context_tokens {
                v["context_length"] = json!(ctx);
            }
            v
        }
        Provider::Anthropic => {
            let mut card = json!({
                "type": "model",
                "id": name,
                "display_name": name,
                "created_at": CREATED_RFC3339,
            });
            add_sampling(&mut card, provider, info);
            return card;
        }
        Provider::OpenRouter => openrouter_card(info),
    };
    add_discovery(&mut card, provider, info, adapter);
    card
}

/// The fields an agent runtime reads to discover what a served model can do and
/// which fine-tune is in front of it. Each is optional and absent when unknown:
/// an unmeasured limit is never `0`, and an adapter is named only by the
/// identity (`id`, `digest`, `version`) its release was verified under, never by where its
/// file lives.
fn add_discovery(card: &mut Value, provider: Provider, info: &ModelInfo, adapter: Option<&AdapterIdentity>) {
    add_sampling(card, provider, info);
    let capabilities = info.caps.names();
    if !capabilities.is_empty() {
        card["capabilities"] = json!(capabilities);
    }
    if info.caps.chat {
        // OpenRouter's spelling, which agent runtimes read for every dialect.
        card["architecture"]["input_modalities"] = json!(info.caps.input_modalities());
    }
    if info.caps.vision() {
        // What one request may carry (a request past either is refused with 400),
        // so a client can leave older pictures out of a long conversation instead.
        card["max_images_per_request"] = json!(info.caps.max_images);
        card["max_image_pixels_per_request"] = json!(capability::blob::MAX_REQUEST_PIXELS);
    }
    if let Some(max) = info.max_output_tokens {
        card["max_output_tokens"] = json!(max);
    }
    if !info.speech_modes.is_empty() {
        // What `/audio/speech` takes for this model: the containers it answers in, how long a reference voice may
        // be, and per speech action whether it imitates a recording (and needs its transcript), takes
        // `instructions`, preset voices or a seed. A field no mode takes is refused, so a client can tell first.
        card["speech"] = json!({
            "response_formats": crate::speech::RESPONSE_FORMATS,
            "max_reference_seconds": crate::speech::MAX_REFERENCE_SECONDS,
            "modes": info.speech_modes,
        });
    }
    if let Some(adapter) = adapter {
        card["adapter"] = adapter.to_json();
    }
    // Whether the model's architecture is certified for production serving,
    // so a caller can tell before relying on it. Said on every card: a model
    // nobody certified is `preview`, never left unsaid.
    card["support"] = json!(info.support.name());
    // The server keeps nothing of a request: no prompt, answer or upload is
    // written anywhere or logged. Said on every card, so a client that routes
    // by data retention can tell it from a server that does not say.
    card["zero_data_retention"] = json!(true);
}

/// `supported_parameters` (OpenRouter's field, which agent runtimes read for every dialect before they send a
/// sampling parameter such as `seed` or `top_k`): the request fields the model's `generate` honours, derived from the
/// tables the request is read with ([`crate::sampling::honoured`]). Absent on a model that generates no text (empty
/// on OpenRouter, whose card always carries the field).
fn add_sampling(card: &mut Value, provider: Provider, info: &ModelInfo) {
    if !info.sampling_parameters.is_empty() {
        card["supported_parameters"] = json!(supported_parameters(provider, info));
    }
}

/// The model's honoured request fields as `provider`'s card may list them: all of them, except on OpenRouter, whose
/// schema enumerates the values the field may hold ([`OPENROUTER_PARAMETERS`]).
fn supported_parameters(provider: Provider, info: &ModelInfo) -> Vec<&'static str> {
    info.sampling_parameters.iter().copied().filter(|p| provider != Provider::OpenRouter || OPENROUTER_PARAMETERS.contains(p)).collect()
}

/// The values OpenRouter's `supported_parameters` may hold (its `ModelsListResponse` schema enumerates them): an
/// OpenRouter card names only these of the fields the model honours; brain's own extensions (`thinking_budget`,
/// `policy_trace`, `return_token_ids`, ...) are listed on the other dialects' cards.
const OPENROUTER_PARAMETERS: [&str; 26] = [
    "temperature", "top_p", "top_k", "min_p", "top_a", "frequency_penalty", "presence_penalty", "repetition_penalty", "max_tokens", "max_completion_tokens", "logit_bias", "logprobs", "top_logprobs", "prediction", "seed", "response_format", "structured_outputs", "stop", "tools", "tool_choice", "parallel_tool_calls", "include_reasoning", "reasoning", "reasoning_effort", "web_search_options", "verbosity",
];

/// What a model takes in and gives out, in OpenRouter's modality names: text
/// for chat and embeddings, audio in and `transcription` out for a transcription model, `speech` out for a
/// text-to-speech one (OpenRouter's own names for the two, distinct from `text` and from the `audio` a chat model
/// writes, so a client picks the right endpoint from the card), image output for an image generator, video output
/// for a video generator. A model that does several lists the union.
fn modalities(caps: CapSet) -> (Vec<&'static str>, Vec<&'static str>) {
    let mut input = Vec::new();
    let mut output = Vec::new();
    if caps.chat || caps.embeddings || caps.image || caps.speech || caps.video {
        input.push("text");
    }
    if caps.vision() {
        input.push("image");
    }
    if caps.transcription || caps.audio_input || caps.speech_reference {
        input.push("audio");
    }
    if caps.chat {
        output.push("text");
    }
    if caps.transcription {
        output.push("transcription");
    }
    if caps.speech {
        output.push("speech");
    }
    if caps.embeddings {
        output.push("embeddings");
    }
    if caps.image {
        output.push("image");
    }
    if caps.video {
        output.push("video");
    }
    (input, output)
}

/// OpenRouter's richer model card (validates against its `Model` schema).
/// `context_length` is a real, documented field of this dialect (unlike
/// OpenAI's) - use the model's actual capacity when known, the conservative
/// fallback otherwise. `top_provider.max_completion_tokens` is its documented
/// home for the output limit.
fn openrouter_card(info: &ModelInfo) -> Value {
    let name = info.name.as_str();
    let ctx = info.max_context_tokens.unwrap_or(FALLBACK_CONTEXT_LENGTH);
    let (input, output) = modalities(info.caps);
    json!({
        "id": name,
        "canonical_slug": name,
        "name": name,
        "created": CREATED_UNIX,
        "description": format!("brain-served model '{name}'"),
        "context_length": ctx,
        "architecture": {
            "modality": format!("{}->{}", input.join("+"), output.join("+")),
            "input_modalities": input,
            "output_modalities": output,
            "tokenizer": "Other",
            "instruct_type": Value::Null,
        },
        "pricing": { "prompt": "0", "completion": "0", "request": "0", "image": "0" },
        "top_provider": { "context_length": ctx, "max_completion_tokens": info.max_output_tokens, "is_moderated": false },
        "per_request_limits": Value::Null,
        "supported_parameters": supported_parameters(Provider::OpenRouter, info),
        "default_parameters": Value::Null,
        "supported_voices": Value::Null,
        "links": { "details": format!("/api/v1/models/{name}/endpoints") },
    })
}
