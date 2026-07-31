// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The OpenAI-compatible surface: `POST /chat/completions` (non-streaming and SSE
//! token streaming), plus the still-stubbed embeddings/image routes. Chat dispatches
//! to the shared executor's `generate` action via [`crate::bridge`]. The OpenRouter
//! surface reuses [`handle_chat`] with `native = true` (adds `native_finish_reason`
//! + `system_fingerprint`).

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use capability::Invocation;
use serde_json::{json, Value};
use std::convert::Infallible;
use uuid::Uuid;

use crate::bridge::{self, StreamMsg};
use crate::catalog;
use crate::error::ApiError;
use crate::models::CREATED_UNIX;
use crate::state::AppState;
use crate::surface::Provider;

/// OpenAI chat/embeddings/image routes (merged onto the shared `/models` router).
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/images/generations", post(images_generations))
        .route("/images/generations", post(images_generations))
}

/// `POST /chat/completions` — real chat (non-stream + SSE) on the OpenAI dialect.
async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    handle_chat(state, body, false).await
}

/// `POST /v1/embeddings` — real embeddings on the OpenAI dialect (shared with
/// OpenRouter via [`handle_embeddings`]).
async fn embeddings(State(state): State<AppState>, body: Bytes) -> Response {
    handle_embeddings(state, body).await
}

/// `POST /v1/images/generations` — 501 until a later phase.
async fn images_generations(State(state): State<AppState>) -> ApiError {
    ApiError::not_implemented(state.provider, "POST /images/generations is not implemented yet")
}

/// How the embedding vector is serialized in the response.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EncodingFormat {
    /// A JSON array of floats (OpenAI default).
    Float,
    /// A base64 string of the little-endian f32 bytes.
    Base64,
}

/// A parsed `CreateEmbeddingRequest`: the resolved model, the input strings (one
/// per requested embedding), the wire format, and an optional truncation width.
struct EmbeddingRequest {
    model: String,
    inputs: Vec<String>,
    encoding_format: EncodingFormat,
    dimensions: Option<usize>,
}

/// Parse + validate a `CreateEmbeddingRequest`. `input` accepts a single string or
/// an array of strings; a token-array input (array of ints, or array of int arrays)
/// is a 400 with a clear message — brain has no tokenizer at this layer to decode
/// it. `model` is required. `encoding_format` is `float` (default) or `base64`;
/// `dimensions`, if given, must be a positive integer.
fn parse_embedding_request(provider: Provider, body: &Value) -> Result<EmbeddingRequest, ApiError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::invalid_request(provider, "'model' is required"))?
        .to_string();

    let inputs = match body.get("input") {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::String(_)) => return Err(ApiError::invalid_request(provider, "'input' string must not be empty")),
        Some(Value::Array(a)) if a.is_empty() => return Err(ApiError::invalid_request(provider, "'input' array must not be empty")),
        Some(Value::Array(a)) => {
            if a.iter().all(|x| x.is_string()) {
                let strs: Vec<String> = a.iter().map(|x| x.as_str().unwrap_or_default().to_string()).collect();
                if strs.iter().any(|s| s.is_empty()) {
                    return Err(ApiError::invalid_request(provider, "'input' strings must not be empty"));
                }
                strs
            } else if a.iter().all(|x| x.is_number() || x.is_array()) {
                // Array of tokens (or array of token arrays): brain has no tokenizer
                // at the API layer to turn ids back into text.
                return Err(ApiError::invalid_request(provider, "token-array 'input' is not supported; pass text as a string or array of strings"));
            } else {
                return Err(ApiError::invalid_request(provider, "'input' must be a string or an array of strings"));
            }
        }
        Some(_) => return Err(ApiError::invalid_request(provider, "'input' must be a string or an array of strings")),
        None => return Err(ApiError::invalid_request(provider, "'input' is required")),
    };

    let encoding_format = match body.get("encoding_format") {
        None | Some(Value::Null) => EncodingFormat::Float,
        Some(Value::String(s)) if s == "float" => EncodingFormat::Float,
        Some(Value::String(s)) if s == "base64" => EncodingFormat::Base64,
        Some(_) => return Err(ApiError::invalid_request(provider, "'encoding_format' must be \"float\" or \"base64\"")),
    };

    let dimensions = match body.get("dimensions") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let n = v.as_i64().filter(|&n| n >= 1).ok_or_else(|| ApiError::invalid_request(provider, "'dimensions' must be a positive integer"))?;
            Some(n as usize)
        }
    };

    Ok(EmbeddingRequest { model, inputs, encoding_format, dimensions })
}

/// Read the mean-pooled embedding vector from an `embed` [`capability::Outcome`]:
/// the `mean` output is a JSON array of f32 (length = the model's `d_model`).
fn read_mean(o: &capability::Outcome) -> Option<Vec<f32>> {
    let arr = o.outputs.get("mean")?.as_array()?;
    Some(arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
}

/// The embeddings handler shared by the OpenAI and OpenRouter surfaces. Both speak
/// the identical `CreateEmbeddingRequest`/`CreateEmbeddingResponse` grammar under
/// Bearer auth, so the provider only shapes the error bodies. Each input string is
/// dispatched as one `embed` job through [`bridge::submit`] (so admission → 429 is
/// enforced exactly as for chat); the response collects one `data` entry per input,
/// indexed in request order.
pub async fn handle_embeddings(state: AppState, body: Bytes) -> Response {
    let provider = state.provider;
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return ApiError::invalid_request(provider, format!("invalid JSON body: {e}")).into_response(),
    };
    let req = match parse_embedding_request(provider, &body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    // Resolve the model against the embeddings-capable manifests before dispatching.
    if !catalog::resolve_embed(&state.exec, &req.model) {
        return ApiError::model_not_found(provider, &req.model).into_response();
    }

    let mut data: Vec<Value> = Vec::with_capacity(req.inputs.len());
    let mut prompt_tokens = 0i64;
    for (index, text) in req.inputs.iter().enumerate() {
        let inv = Invocation::new().set("text", json!(text));
        let outcome = match bridge::submit(&state, &req.model, "embed", inv).await {
            Ok(o) => o,
            Err(e) => return e.into_response(),
        };
        let mut vector = match read_mean(&outcome) {
            Some(v) if !v.is_empty() => v,
            _ => return ApiError::invalid_request(provider, "model did not return a 'mean' embedding vector").into_response(),
        };
        // `dimensions`: truncate to the first N dims (brain does NOT re-project);
        // asking for more dims than the model produces is a 400.
        if let Some(dims) = req.dimensions {
            if dims > vector.len() {
                return ApiError::invalid_request(provider, format!("'dimensions' ({dims}) exceeds the model's embedding length ({})", vector.len())).into_response();
            }
            vector.truncate(dims);
        }
        // Usage: the encoder reports the token count as the `tokens` output; if a
        // model omits it, fall back to a chars/4 heuristic (documented, coarse).
        prompt_tokens += outcome
            .outputs
            .get("tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or_else(|| (text.chars().count().div_ceil(4)).max(1) as i64);

        let embedding = match req.encoding_format {
            EncodingFormat::Float => json!(vector),
            EncodingFormat::Base64 => json!(events::bytes::encode_f32(&vector)),
        };
        data.push(json!({ "object": "embedding", "index": index, "embedding": embedding }));
    }

    let resp = json!({
        "object": "list",
        "data": data,
        "model": req.model,
        // OpenAI's embedding usage has only prompt_tokens + total_tokens (no
        // completion side); the two are equal for an encoder.
        "usage": { "prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens },
    });
    Json(resp).into_response()
}

/// The chat handler shared by the OpenAI and OpenRouter surfaces. `native` adds
/// OpenRouter's `native_finish_reason` (mirroring `finish_reason`) and the
/// `system_fingerprint` its `ChatResult` requires.
pub async fn handle_chat(state: AppState, body: Bytes, native: bool) -> Response {
    let provider = state.provider;
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return ApiError::invalid_request(provider, format!("invalid JSON body: {e}")).into_response(),
    };
    let (model, inv, stream) = match to_invocation(provider, &body) {
        Ok(x) => x,
        Err(e) => return e.into_response(),
    };
    // Resolve the model against the chat-capable manifests before dispatching.
    if !catalog::resolve_chat(&state.exec, &model) {
        return ApiError::model_not_found(provider, &model).into_response();
    }
    if stream {
        let want_usage = body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(|v| v.as_bool()).unwrap_or(false);
        stream_chat(state, model, inv, native, want_usage).await
    } else {
        match bridge::submit(&state, &model, "generate", inv).await {
            Ok(outcome) => {
                let (text, prompt, completion, finish) = bridge::read_outcome(&outcome);
                Json(non_stream_body(&model, &text, prompt, completion, &finish, native)).into_response()
            }
            Err(e) => e.into_response(),
        }
    }
}

/// Parse + validate an OpenAI chat request into `(model, invocation, stream)`.
/// Enforces `model`/`messages` present and rejects `n > 1`; builds the contract
/// `generate` invocation.
pub fn to_invocation(provider: Provider, body: &Value) -> Result<(String, Invocation, bool), ApiError> {
    let model = body.get("model").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).ok_or_else(|| ApiError::invalid_request(provider, "'model' is required"))?;
    let messages = body.get("messages").and_then(|v| v.as_array()).filter(|a| !a.is_empty()).ok_or_else(|| ApiError::invalid_request(provider, "'messages' must be a non-empty array"))?;
    if body.get("n").and_then(|v| v.as_i64()).unwrap_or(1) > 1 {
        return Err(ApiError::invalid_request(provider, "'n' > 1 is not supported"));
    }
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // Flatten each message to the contract shape {role, content(text)}.
    let msgs: Vec<Value> = messages.iter().map(flatten_message).collect();
    let mut inv = Invocation::new()
        .set("messages", json!(serde_json::to_string(&msgs).unwrap_or_else(|_| "[]".into())))
        .set("max_new", json!(max_new(body)))
        .set("temp", json!(body.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0)))
        .set("top_p", json!(body.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0)))
        .set("top_k", json!(body.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0)))
        .set("seed", json!(body.get("seed").and_then(|v| v.as_i64()).unwrap_or(0)));
    if let Some(stop) = normalize_stop(body.get("stop")) {
        inv = inv.set("stop", json!(stop));
    }
    Ok((model.to_string(), inv, stream))
}

/// `max_tokens` (or the newer `max_completion_tokens`), defaulting to 1024.
fn max_new(body: &Value) -> i64 {
    body.get("max_completion_tokens").and_then(|v| v.as_i64()).or_else(|| body.get("max_tokens").and_then(|v| v.as_i64())).unwrap_or(1024)
}

/// Normalize OpenAI `stop` (string | array | null) to a JSON-array string, or `None`.
fn normalize_stop(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) => Some(serde_json::to_string(&vec![s.clone()]).unwrap_or_default()),
        Some(Value::Array(a)) if !a.is_empty() => Some(serde_json::to_string(a).unwrap_or_default()),
        _ => None,
    }
}

/// One OpenAI message → the contract `{role, content}` (content flattened to text).
fn flatten_message(m: &Value) -> Value {
    let role = match m.get("role").and_then(|v| v.as_str()).unwrap_or("user") {
        "system" | "developer" => "system",
        "assistant" => "assistant",
        // tool results and anything else fold into the user turn.
        _ => "user",
    };
    json!({ "role": role, "content": content_text(m.get("content")) })
}

/// Flatten OpenAI message content (a string, or an array of typed parts) to text.
fn content_text(c: Option<&Value>) -> String {
    match c {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Map the contract `finish_reason` to OpenAI's enum. There is no `stop_sequence`
/// in OpenAI; a stop-sequence hit is reported as `stop`.
fn finish_openai(fr: &str) -> &'static str {
    match fr {
        "length" => "length",
        _ => "stop",
    }
}

/// The non-streaming `chat.completion` body.
fn non_stream_body(model: &str, text: &str, prompt: i64, completion: i64, finish: &str, native: bool) -> Value {
    let fr = finish_openai(finish);
    let mut choice = json!({
        "index": 0,
        "message": { "role": "assistant", "content": text, "refusal": Value::Null },
        "finish_reason": fr,
        "logprobs": Value::Null,
    });
    if native {
        choice["native_finish_reason"] = json!(fr);
    }
    let mut body = json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4().simple()),
        "object": "chat.completion",
        "created": CREATED_UNIX,
        "model": model,
        "choices": [choice],
        "usage": { "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion },
    });
    if native {
        // OpenRouter's ChatResult requires system_fingerprint (nullable).
        body["system_fingerprint"] = Value::Null;
    }
    body
}

/// One streaming `chat.completion.chunk`.
fn chunk(id: &str, model: &str, delta: Value, finish: Option<&str>, native: bool) -> Value {
    let mut choice = json!({ "index": 0, "delta": delta, "finish_reason": finish });
    if native {
        choice["native_finish_reason"] = json!(finish);
    }
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": CREATED_UNIX,
        "model": model,
        "choices": [choice],
    })
}

/// The SSE `chat.completion.chunk` stream: a role chunk, one content chunk per token
/// delta, a terminal chunk carrying `finish_reason`, an optional usage-only chunk
/// (when `stream_options.include_usage`), then `data: [DONE]`. Runs the admission
/// race FIRST: if the job cannot start on a lane within `state.admit_deadline`, this
/// returns a plain 429 body (with `Retry-After`) instead of an event-stream.
async fn stream_chat(state: AppState, model: String, inv: Invocation, native: bool, want_usage: bool) -> Response {
    use futures::StreamExt;
    // Admit BEFORE returning the SSE body — a shed request is a plain 429, not an
    // event-stream that immediately errors.
    let mut src = match bridge::stream(&state, &model, "generate", inv).await {
        Ok(src) => src,
        Err(e) => return e.into_response(),
    };
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let events = async_stream::stream! {
        // First chunk: announce the assistant role.
        yield Ok::<Event, Infallible>(Event::default().data(chunk(&id, &model, json!({ "role": "assistant", "content": "" }), None, native).to_string()));

        let mut finish = String::from("stop");
        let (mut prompt, mut completion) = (0i64, 0i64);
        while let Some(msg) = src.next().await {
            match msg {
                StreamMsg::Delta(piece) => {
                    yield Ok(Event::default().data(chunk(&id, &model, json!({ "content": piece }), None, native).to_string()));
                }
                StreamMsg::Done(outcome) => {
                    let (_t, p, c, fr) = bridge::read_outcome(&outcome);
                    prompt = p;
                    completion = c;
                    finish = fr;
                }
                StreamMsg::Err(e) => {
                    // Surface the error as an OpenAI error frame, then terminate.
                    yield Ok(Event::default().data(e.body().to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }
        // Terminal chunk: empty delta + finish_reason.
        let fr = finish_openai(&finish);
        yield Ok(Event::default().data(chunk(&id, &model, json!({}), Some(fr), native).to_string()));

        if want_usage {
            let usage_chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": CREATED_UNIX,
                "model": model,
                "choices": [],
                "usage": { "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion },
            });
            yield Ok(Event::default().data(usage_chunk.to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(events.boxed()).into_response()
}
