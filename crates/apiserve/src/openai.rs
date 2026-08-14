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

use residency::Supply;

use crate::bridge::{self, StreamMsg};
use crate::catalog;
use crate::error::ApiError;
use crate::models::CREATED_UNIX;
use crate::png;
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

/// `POST /v1/images/generations` — real image generation (non-stream + SSE),
/// shared with OpenRouter via [`handle_images`].
async fn images_generations(State(state): State<AppState>, body: Bytes) -> Response {
    handle_images(state, body).await
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
/// enforced exactly as for chat), and every input's job is submitted CONCURRENTLY
/// so the executor's batch-by-model dispatch can group them into one `run_batch`;
/// the response collects one `data` entry per input, indexed in request order.
pub async fn handle_embeddings(state: AppState, body: Bytes) -> Response {
    let provider = state.provider;
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return ApiError::invalid_request(provider, format!("invalid JSON body: {e}")).into_response(),
    };
    let mut req = match parse_embedding_request(provider, &body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    // Resolve the model against the embeddings-capable manifests before dispatching. On
    // the OpenRouter surface this also strips a `"<provider>/"` prefix and walks a
    // `models` fallback array; OpenAI is exact-match only. The resolved local id
    // replaces the requested one for dispatch and the echoed response `model`.
    let manifests = state.exec.manifests(); // one catalog snapshot per request (see catalog::resolve_chat)
    match catalog::resolve_model(provider, &body, |id| catalog::resolve_embed(&manifests, id).then_some(())) {
        Some((id, ())) => req.model = id,
        None => match bridge::ensure_and_recheck(&state, provider, &req.model, |id| catalog::resolve_embed(&state.exec.manifests(), id).then_some(())).await {
            Ok(()) => {}
            Err(e) => return e.into_response(),
        },
    }

    // Submit ALL inputs concurrently (order preserved by position): the
    // executor batches by model, so N jobs in flight together can be grouped
    // into ONE `Instance::run_batch` call -- awaiting them one at a time in a
    // for-loop serialized the single most batchable workload on the API into N
    // sequential submit->admit->run round-trips and the dispatcher never saw a
    // batch. Admission (429) is still enforced per job by `bridge::submit`;
    // `try_join_all` returns the FIRST failure and drops (cancels) the
    // remaining futures, keeping the request's all-or-nothing semantics.
    let outcomes = match futures::future::try_join_all(
        req.inputs.iter().map(|text| bridge::submit(&state, &req.model, "embed", Invocation::new().set("text", json!(text)))),
    )
    .await
    {
        Ok(o) => o,
        Err(e) => return e.into_response(),
    };

    let mut data: Vec<Value> = Vec::with_capacity(req.inputs.len());
    let mut prompt_tokens = 0i64;
    for (index, (text, outcome)) in req.inputs.iter().zip(&outcomes).enumerate() {
        let mut vector = match read_mean(outcome) {
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
    let (requested, inv, stream) = match to_invocation(provider, &body) {
        Ok(x) => x,
        Err(e) => return e.into_response(),
    };
    let want_usage = || body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(|v| v.as_bool()).unwrap_or(false);

    // Resolve the model against the chat-capable manifests before dispatching. On the
    // OpenRouter surface this also strips a `"<provider>/"` prefix and walks a `models`
    // fallback array (see [`catalog::resolve_model`]); OpenAI is exact-match only.
    let manifests = state.exec.manifests(); // one catalog snapshot per request (see catalog::resolve_chat)
    match catalog::resolve_model(provider, &body, |id| catalog::resolve_chat(&manifests, id).then_some(())) {
        Some((model, ())) => {
            if stream {
                stream_chat(state, model, inv, native, want_usage()).await
            } else {
                match bridge::submit(&state, &model, "generate", inv).await {
                    Ok(outcome) => Json(non_stream_body(&model, &bridge::read_chat_outcome(&outcome), native)).into_response(),
                    Err(e) => e.into_response(),
                }
            }
        }
        // Not already resident. Non-streaming blocks on the fetch (the plan's
        // accepted trade-off); streaming opens the SSE body immediately and
        // reports fetch progress as it happens (`stream_chat_with_autofetch`)
        // -- but only once classify() has ALREADY confirmed Fetchable with
        // zero I/O, so an Unknown/no-supplier model still never opens a
        // stream that would just immediately error.
        None if stream => match state.supplier.clone() {
            Some(supplier) if matches!(supplier.classify(&requested), Supply::Fetchable) => {
                stream_chat_with_autofetch(state, supplier, requested, inv, native, want_usage())
            }
            _ => ApiError::model_not_found(provider, &requested).into_response(),
        },
        None => match bridge::ensure_and_recheck(&state, provider, &requested, |id| catalog::resolve_chat(&state.exec.manifests(), id).then_some(())).await {
            Ok(()) => match bridge::submit(&state, &requested, "generate", inv).await {
                Ok(outcome) => Json(non_stream_body(&requested, &bridge::read_chat_outcome(&outcome), native)).into_response(),
                Err(e) => e.into_response(),
            },
            Err(e) => e.into_response(),
        },
    }
}

/// Bounds enforced on `tools`/`tool_choice` before they ever reach the resident
/// model: a request
/// body is otherwise unbounded attacker-controlled JSON, and `tools` feeds
/// straight into prompt construction (the `<tools>` block's byte length is
/// unbounded input to the tokenizer/model if left unchecked).
const MAX_TOOLS: usize = 128;
const MAX_TOOLS_BYTES: usize = 256 * 1024;
const MAX_TOOL_NAME_LEN: usize = 64;

/// Parse + validate an OpenAI chat request into `(model, invocation, stream)`.
/// Enforces `model`/`messages` present, rejects `n > 1`, and enforces every
/// `tools`/`tool_choice`/`tool_call_id` INPUT BOUND (see [`validate_tools`]/
/// [`validate_tool_choice`]); builds the contract `generate` invocation, setting
/// `tools`/`tool_choice`/`enable_thinking` only when the request supplies them
/// (the same optional-param pattern as `stop`).
pub fn to_invocation(provider: Provider, body: &Value) -> Result<(String, Invocation, bool), ApiError> {
    let model = body.get("model").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).ok_or_else(|| ApiError::invalid_request(provider, "'model' is required"))?;
    let messages = body.get("messages").and_then(|v| v.as_array()).filter(|a| !a.is_empty()).ok_or_else(|| ApiError::invalid_request(provider, "'messages' must be a non-empty array"))?;
    if body.get("n").and_then(|v| v.as_i64()).unwrap_or(1) > 1 {
        return Err(ApiError::invalid_request(provider, "'n' > 1 is not supported"));
    }
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // Every `role:"tool"` message must carry the `tool_call_id` it answers — a
    // client that omits it has sent a malformed request that would otherwise
    // silently misattribute the tool result; reject it up front.
    for (i, m) in messages.iter().enumerate() {
        if m.get("role").and_then(|v| v.as_str()) == Some("tool") {
            let has_id = m.get("tool_call_id").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).is_some();
            if !has_id {
                return Err(ApiError::invalid_request(provider, format!("messages[{i}]: a 'tool' message requires a non-empty 'tool_call_id'")));
            }
        }
    }

    // Flatten each message to the contract shape, preserving tool_calls/
    // tool_call_id/reasoning_content so a multi-turn tool-calling conversation
    // round-trips (see [`flatten_message`]).
    let msgs: Vec<Value> = messages.iter().map(flatten_message).collect();
    let mut inv = Invocation::new()
        .set("messages", json!(serde_json::to_string(&msgs).unwrap_or_else(|_| "[]".into())))
        .set("max_new", json!(max_new(body)))
        .set("temp", json!(body.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0)))
        .set("top_p", json!(body.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0)))
        // 40 is the standard top-k default (matches e.g. Google AI Studio): wide
        // enough to keep text natural while filtering out the improbable tail.
        // `top_k=1` degenerates to greedy; `0` or negative disables the filter
        // entirely (`sample_logits` only applies it when `top_k > 0`).
        .set("top_k", json!(body.get("top_k").and_then(|v| v.as_i64()).unwrap_or(40)))
        .set("seed", json!(body.get("seed").and_then(|v| v.as_i64()).unwrap_or(0)));
    if let Some(stop) = normalize_stop(body.get("stop")) {
        inv = inv.set("stop", json!(stop));
    }

    if let Some(tools) = body.get("tools").filter(|v| !v.is_null()) {
        validate_tools(provider, tools)?;
        inv = inv.set("tools", json!(serde_json::to_string(tools).unwrap_or_else(|_| "[]".into())));
    }
    if let Some(tc) = body.get("tool_choice").filter(|v| !v.is_null()) {
        validate_tool_choice(provider, tc)?;
        inv = inv.set("tool_choice", json!(serde_json::to_string(tc).unwrap_or_default()));
    }
    // `enable_thinking`: nested under `chat_template_kwargs` (the vLLM/SGLang
    // OpenAI-compatible extension point real Qwen3 tool-calling clients use) or,
    // tolerated, top-level.
    let enable_thinking = body
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        .or_else(|| body.get("enable_thinking").and_then(|v| v.as_bool()));
    if let Some(et) = enable_thinking {
        inv = inv.set("enable_thinking", json!(et));
    }

    // image_url/input_audio content parts' REAL bytes -- flatten_message's
    // own "content" (message_content) now preserves a lightweight, payload-
    // stripped typed marker for the chat template's own placeholder
    // placement (see that function's doc), but the actual pixel/audio bytes
    // still only ever flow through here, into a blob -- see crate::media's
    // module doc. Attaching the blobs unconditionally is harmless for a
    // model whose generate action doesn't declare an "image"/"audio" input
    // (same as any unused blob passed over D-Bus).
    let media = crate::media::extract_openai(messages).map_err(|e| ApiError::invalid_request(provider, e))?;
    if let Some(img) = media.image {
        inv = inv.blob("image", img);
    }
    if let Some(a) = media.audio {
        inv = inv.blob("audio", a);
    }

    Ok((model.to_string(), inv, stream))
}

/// INPUT BOUNDS on the `tools` array: must be an array, at most [`MAX_TOOLS`]
/// entries, at most [`MAX_TOOLS_BYTES`] serialized bytes, and every element's
/// `function.name` a non-empty string of at most [`MAX_TOOL_NAME_LEN`]
/// characters. Enforced BEFORE the array ever reaches the resident model /
/// prompt renderer.
fn validate_tools(provider: Provider, tools: &Value) -> Result<(), ApiError> {
    let arr = tools.as_array().ok_or_else(|| ApiError::invalid_request(provider, "'tools' must be an array"))?;
    if arr.len() > MAX_TOOLS {
        return Err(ApiError::invalid_request(provider, format!("'tools' must not have more than {MAX_TOOLS} entries")));
    }
    let bytes = serde_json::to_vec(tools).map(|v| v.len()).unwrap_or(usize::MAX);
    if bytes > MAX_TOOLS_BYTES {
        return Err(ApiError::invalid_request(provider, format!("'tools' payload must not exceed {MAX_TOOLS_BYTES} bytes")));
    }
    for (i, t) in arr.iter().enumerate() {
        let name = t.get("function").and_then(|f| f.get("name"));
        let ok = matches!(name, Some(Value::String(s)) if !s.is_empty() && s.chars().count() <= MAX_TOOL_NAME_LEN);
        if !ok {
            return Err(ApiError::invalid_request(
                provider,
                format!("'tools[{i}].function.name' must be a non-empty string of at most {MAX_TOOL_NAME_LEN} characters"),
            ));
        }
    }
    Ok(())
}

/// INPUT BOUNDS on `tool_choice`: `"auto"` | `"none"` | `"required"`, or
/// `{"type":"function","function":{"name":<non-empty string>}}` — any other shape
/// is a 400 rather than being forwarded verbatim to the resident.
fn validate_tool_choice(provider: Provider, tc: &Value) -> Result<(), ApiError> {
    let ok = match tc {
        Value::String(s) => s == "auto" || s == "none" || s == "required",
        Value::Object(o) => {
            o.get("type").and_then(|v| v.as_str()) == Some("function")
                && matches!(o.get("function").and_then(|f| f.get("name")), Some(Value::String(s)) if !s.is_empty())
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(ApiError::invalid_request(
            provider,
            "'tool_choice' must be \"auto\", \"none\", \"required\", or {\"type\":\"function\",\"function\":{\"name\":...}}",
        ))
    }
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

/// One OpenAI message → the contract `{role, content, reasoning_content?,
/// tool_calls?, tool_call_id?}` (`content` - see [`message_content`]; a
/// `tool_calls` element's `function.{name,arguments}` flattens to
/// `{id,name,arguments}` - the shape `crates/cli/src/resident_llm.rs::
/// parse_chat_messages` reads). `role:"tool"` is now its OWN contract role
/// (no longer folded into `user`) so the resident's chat template renders it
/// as a `<tool_response>` turn.
fn flatten_message(m: &Value) -> Value {
    let role = match m.get("role").and_then(|v| v.as_str()).unwrap_or("user") {
        "system" | "developer" => "system",
        "assistant" => "assistant",
        "tool" => "tool",
        // anything else folds into the user turn.
        _ => "user",
    };
    let mut out = json!({ "role": role, "content": message_content(m.get("content")) });
    if let Some(rc) = m.get("reasoning_content").and_then(|v| v.as_str()) {
        out["reasoning_content"] = json!(rc);
    }
    if let Some(id) = m.get("tool_call_id").and_then(|v| v.as_str()) {
        out["tool_call_id"] = json!(id);
    }
    if let Some(calls) = m.get("tool_calls").and_then(|v| v.as_array()).filter(|a| !a.is_empty()) {
        let flat: Vec<Value> = calls
            .iter()
            .map(|c| {
                let id = c.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                let name = c.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()).unwrap_or_default();
                let arguments = match c.get("function").and_then(|f| f.get("arguments")) {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => serde_json::to_string(other).unwrap_or_default(),
                    None => "{}".to_string(),
                };
                json!({ "id": id, "name": name, "arguments": arguments })
            })
            .collect();
        out["tool_calls"] = json!(flat);
    }
    out
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

/// The contract `content` for one message: [`content_text`]'s flattened
/// STRING when the original `content` is already a string, or an array with
/// ONLY `"text"` parts (byte-identical to this function's old always-flatten
/// behavior - every other model's own `messages` parser, and every
/// text-only client, sees no change at all); a NORMALIZED typed-part ARRAY
/// when `content` has any non-`"text"` part (`image_url`/`input_audio`/...).
///
/// **Real bug this closes**: this used to be [`content_text`] unconditionally
/// - ANY multimodal content part (`image_url`, `input_audio`, ...) was
/// silently dropped before the chat-templated `messages` JSON was ever built,
/// so `omni::caps::render_chat_prompt`'s real Jinja template (which DOES
/// detect typed parts and place `<|vision_start|><|image_pad|><|vision_end|>`
/// / `<|audio_start|><|audio_pad|><|audio_end|>` at each part's own position
/// - see that function's doc) never actually saw a typed array in
/// production, only ever flattened text; verified by tracing this exact
/// code path (not a standalone probe) against a real captured sven request.
/// Preserving the array here is what lets `crate::mm::build_multimodal_prompt`
/// expand each medium's real embeddings IN PLACE at its own placeholder,
/// instead of a whole-block splice heuristic.
///
/// `input_audio` parts also get TWO additional keys, `"audio"` and
/// `"audio_url"` (additive - the original `"type"`/`"input_audio"` keys stay
/// untouched, so any other consumer of the raw shape is unaffected): read
/// directly off the real checkpoint's `chat_template.json`
/// (`/tmp/.X11-unix/brain/omni/Qwen3-Omni-30B-A3B-Instruct/chat_template.json`
/// at the time this was written), the template's own audio detection is
/// `content.type == 'audio' or 'audio' in content or 'audio_url' in
/// content` - none of which match OpenAI's real
/// `{"type":"input_audio","input_audio":{...}}` shape as sent by a real
/// OpenAI-compatible client (sven included) as-is, so without this the
/// audio part would reach the template as a typed array yet still render as
/// NOTHING (silently falls through every `{%- elif %}` branch). `image_url`
/// parts need no such normalization: OpenAI's own `{"type":"image_url",
/// "image_url":{...}}` shape already satisfies `'image_url' in content`
/// unmodified.
fn message_content(c: Option<&Value>) -> Value {
    match c {
        Some(Value::Array(parts)) if parts.iter().any(|p| p.get("type").and_then(|t| t.as_str()) != Some("text")) => {
            Value::Array(parts.iter().map(normalize_content_part_for_template).collect())
        }
        other => json!(content_text(other)),
    }
}

/// One content part, normalized for the chat template's own typed-part
/// detection (see [`message_content`]'s doc for why `input_audio` needs an
/// extra key and `image_url`/`text` don't) AND stripped of its actual
/// pixel/audio PAYLOAD bytes: the template only ever checks part-KEY
/// membership and emits a FIXED placeholder literal (`<|vision_start|>
/// <|image_pad|><|vision_end|>` etc.) - never the url/data VALUE itself (read
/// directly off the real template text) - so carrying the real (often
/// multi-hundred-KB base64) payload through the `messages` JSON string
/// alongside the ALREADY-separately-extracted blob (`crate::media::
/// extract_openai`, still the real bytes' only consumer) would only double
/// the memory this request holds for no benefit.
fn normalize_content_part_for_template(p: &Value) -> Value {
    match p.get("type").and_then(|t| t.as_str()) {
        Some("image_url") => json!({ "type": "image_url", "image_url": {} }),
        Some("input_audio") => {
            // Keep "format" (small, harmless) if present; always drop "data"
            // (the real base64 payload).
            let format = p.get("input_audio").and_then(|a| a.get("format")).cloned();
            json!({ "type": "input_audio", "input_audio": { "format": format.unwrap_or(Value::Null) }, "audio": true, "audio_url": true })
        }
        _ => p.clone(),
    }
}

/// Map the contract `finish_reason` to OpenAI's enum. There is no `stop_sequence`
/// in OpenAI; a stop-sequence hit is reported as `stop`. `tool_calls` passes
/// through unchanged (both the contract and OpenAI use that exact name).
fn finish_openai(fr: &str) -> &'static str {
    match fr {
        "length" => "length",
        "tool_calls" => "tool_calls",
        _ => "stop",
    }
}

/// The non-streaming `chat.completion` body. `co.tool_calls` non-empty emits
/// `message.tool_calls` ([`openai_tool_calls`]: `{id, type:"function",
/// function:{name, arguments}}`, `arguments` a JSON STRING) and sets
/// `message.content` to JSON `null` (never an empty string —
/// `ChatCompletionResponseMessage.content` is nullable specifically for a
/// tool-call-only turn).
fn non_stream_body(model: &str, co: &bridge::ChatOutcome, native: bool) -> Value {
    let fr = finish_openai(&co.finish);
    let content = if co.tool_calls.is_empty() { json!(co.text) } else { Value::Null };
    let mut message = json!({ "role": "assistant", "content": content, "refusal": Value::Null });
    if !co.reasoning.is_empty() {
        message["reasoning_content"] = json!(co.reasoning);
    }
    if !co.tool_calls.is_empty() {
        message["tool_calls"] = json!(openai_tool_calls(&co.tool_calls));
    }
    let mut choice = json!({
        "index": 0,
        "message": message,
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
        "usage": { "prompt_tokens": co.prompt_tokens, "completion_tokens": co.completion_tokens, "total_tokens": co.prompt_tokens + co.completion_tokens },
    });
    if native {
        // OpenRouter's ChatResult requires system_fingerprint (nullable).
        body["system_fingerprint"] = Value::Null;
    }
    body
}

/// The internal `{id,name,arguments}` tool-call shape ([`bridge::ChatOutcome`]) →
/// OpenAI's `ChatCompletionMessageToolCall`: `{id, type:"function",
/// function:{name, arguments}}`. `arguments` is re-emitted verbatim as the
/// JSON-text string the resident produced — never re-parsed (the server relays
/// model output; it does not parse-and-execute tool calls).
fn openai_tool_calls(calls: &[Value]) -> Vec<Value> {
    calls
        .iter()
        .map(|c| {
            let id = c.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            let name = c.get("name").and_then(|v| v.as_str()).unwrap_or_default();
            let arguments = c.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
            json!({ "id": id, "type": "function", "function": { "name": name, "arguments": arguments } })
        })
        .collect()
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

/// A [`StreamMsg::Event`]'s neutral `{"kind":...}` payload (see
/// `resident_llm.rs::emit_chat_events`/`resident_mock.rs::generate_tool_call` for
/// the emitting side) → an OpenAI streaming `delta` fragment: `reasoning` becomes
/// `delta.reasoning_content`; `tool_call_start` becomes the first `delta.tool_calls`
/// chunk for that index (`id`+`type`+`function.name`+empty `function.arguments`);
/// `tool_call_args` becomes a later chunk carrying only `index`+`function.arguments`.
/// `tool_call_end` (OpenAI's wire format has no explicit per-call terminator — the
/// terminal `finish_reason:"tool_calls"` chunk covers it) and any unrecognized
/// `kind` (forward-compatible: ignored, not an error) yield no chunk. Deliberately
/// the ONLY path that builds `delta.reasoning_content`/`delta.tool_calls` — never
/// touches `delta.content` — so raw `<think>`/`<tool_call>` markup can never reach
/// a plain content delta (see [`StreamMsg`]'s doc comment).
fn event_delta(v: &Value) -> Option<Value> {
    match v.get("kind").and_then(|k| k.as_str())? {
        "reasoning" => Some(json!({ "reasoning_content": v.get("text").and_then(|t| t.as_str()).unwrap_or("") })),
        "tool_call_start" => Some(json!({
            "tool_calls": [{
                "index": v.get("index").and_then(|i| i.as_u64()).unwrap_or(0),
                "id": v.get("id").and_then(|i| i.as_str()).unwrap_or(""),
                "type": "function",
                "function": { "name": v.get("name").and_then(|n| n.as_str()).unwrap_or(""), "arguments": "" },
            }]
        })),
        "tool_call_args" => Some(json!({
            "tool_calls": [{
                "index": v.get("index").and_then(|i| i.as_u64()).unwrap_or(0),
                "function": { "arguments": v.get("text").and_then(|t| t.as_str()).unwrap_or("") },
            }]
        })),
        _ => None,
    }
}

// ============================================================ image generation

/// The whitelist of accepted `size` values → `(width, height)` in pixels. A request
/// size outside this set is a 400 (an arbitrary WxH could blow up a model's VRAM or
/// simply not be a supported latent grid). Covers the common OpenAI square/portrait/
/// landscape sizes; the default is `1024x1024`.
const IMAGE_SIZES: &[(&str, u32, u32)] = &[
    ("256x256", 256, 256),
    ("512x512", 512, 512),
    ("1024x1024", 1024, 1024),
    ("1024x1536", 1024, 1536),
    ("1536x1024", 1536, 1024),
    ("1024x1792", 1024, 1792),
    ("1792x1024", 1792, 1024),
];

/// The `size` values OpenAI's streaming image events accept in their `size` enum.
/// A requested size outside this set is reported as `"auto"` in the stream frames
/// (the response body itself never echoes a constrained `size`).
const STREAM_SIZES: &[&str] = &["1024x1024", "1024x1536", "1536x1024"];

/// A parsed OpenAI `CreateImageRequest`: the resolved model + prompt, how many
/// images, the resolved pixel size (+ its label), a base seed, and whether to stream.
struct ImageRequest {
    model: String,
    prompt: String,
    n: u32,
    width: u32,
    height: u32,
    size_label: String,
    seed: i64,
    stream: bool,
    /// Non-standard, like `seed`: `"int8"` (default) or `"fp32"` — see
    /// `s3dit::caps`'s `precision` param (`Opts::hifi`). `None` when the
    /// caller omitted it, so the resident model's own default applies
    /// unchanged rather than this endpoint silently pinning one.
    precision: Option<String>,
}

/// Current Unix time (seconds) — the `created`/`created_at` the image responses carry.
fn now_unix() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(CREATED_UNIX)
}

/// Parse + validate a `CreateImageRequest`. `prompt` and `model` are required;
/// `n` defaults to 1 and must be 1..=10; `size` (default `1024x1024`) must be in
/// [`IMAGE_SIZES`]; `response_format` is `b64_json` (default) or `url` — brain has no
/// object store, so a `url` request is still answered with `b64_json` (documented);
/// any other value is a 400. `quality`/`style`/`background` etc. are accepted and
/// ignored. An optional `seed` (non-standard, honoured when present) seeds
/// generation; an optional `precision` (non-standard, `"int8"`|`"fp32"`) selects
/// the DiT precision on a model that supports it (zimage) — omitted, the
/// resident model's own default applies.
fn parse_image_request(provider: Provider, body: &Value) -> Result<ImageRequest, ApiError> {
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::invalid_request(provider, "'prompt' is required"))?
        .to_string();
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::invalid_request(provider, "'model' is required"))?
        .to_string();

    let n = match body.get("n") {
        None | Some(Value::Null) => 1,
        Some(v) => {
            let n = v.as_i64().filter(|&n| (1..=10).contains(&n)).ok_or_else(|| ApiError::invalid_request(provider, "'n' must be an integer between 1 and 10"))?;
            n as u32
        }
    };

    let size_label = match body.get("size") {
        None | Some(Value::Null) => "1024x1024".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => return Err(ApiError::invalid_request(provider, "'size' must be a string like \"1024x1024\"")),
    };
    let (width, height) = IMAGE_SIZES
        .iter()
        .find(|(label, ..)| *label == size_label)
        .map(|(_, w, h)| (*w, *h))
        .ok_or_else(|| {
            let allowed = IMAGE_SIZES.iter().map(|(l, ..)| *l).collect::<Vec<_>>().join(", ");
            ApiError::invalid_request(provider, format!("unsupported 'size' {size_label:?}; allowed: {allowed}"))
        })?;

    // response_format: accept both, but always answer with b64_json (no object store
    // for a hosted URL). An unknown value is a 400.
    match body.get("response_format") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s == "b64_json" || s == "url" => {}
        Some(_) => return Err(ApiError::invalid_request(provider, "'response_format' must be \"b64_json\" or \"url\"")),
    }

    // precision: same shape as response_format above -- an unrecognized value is
    // a 400, not a silent fall-through to whatever the model defaults to.
    let precision = match body.get("precision") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s == "int8" || s == "fp32" => Some(s.clone()),
        Some(_) => return Err(ApiError::invalid_request(provider, "'precision' must be \"int8\" or \"fp32\"")),
    };

    let seed = body.get("seed").and_then(|v| v.as_i64()).unwrap_or(0);
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    Ok(ImageRequest { model, prompt, n, width, height, size_label, seed, stream, precision })
}

/// The image-action [`Invocation`] for the `i`-th requested image: prompt + size +
/// a per-image seed (so `n>1` yields distinct images from a seed-driven model).
fn image_invocation(req: &ImageRequest, i: u32) -> Invocation {
    let mut inv = Invocation::new()
        .set("prompt", json!(req.prompt))
        .set("width", json!(req.width))
        .set("height", json!(req.height))
        .set("seed", json!(req.seed.wrapping_add(i as i64)));
    if let Some(p) = &req.precision {
        inv = inv.set("precision", json!(p));
    }
    inv
}

/// Read the generated image from an [`capability::Outcome`] and return it as
/// base64-of-PNG (OpenAI's `b64_json`). The blob is either already a PNG (base64 it
/// as-is) or brain's raw HWC-f32 image wire format (`{w,h,c}` meta, f32-LE in
/// `[0,1]`), which is quantised to 8-bit RGB and PNG-encoded via [`crate::png`].
fn image_b64_from_outcome(o: &capability::Outcome) -> Result<String, String> {
    let blob = o.blobs.get("image").ok_or("model returned no 'image' output blob")?;
    if blob.bytes.starts_with(&png::SIGNATURE) {
        return Ok(events::base64::encode(&blob.bytes));
    }
    // Raw HWC f32 in [0,1] (capability::blob::image_blob): quantise to RGB8.
    let dim = |k: &str| blob.meta.get(k).and_then(|v| v.as_u64());
    let (w, h) = (dim("w").ok_or("image blob missing 'w'")? as u32, dim("h").ok_or("image blob missing 'h'")? as u32);
    let px = w as usize * h as usize;
    if px == 0 || blob.bytes.len() % 4 != 0 {
        return Err("image blob is not a whole number of f32 samples".into());
    }
    let samples: Vec<f32> = blob.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
    if !samples.len().is_multiple_of(px) {
        return Err(format!("image blob ({} samples) is not a whole number of {w}×{h} planes", samples.len()));
    }
    let c = samples.len() / px;
    let q = |f: f32| (f.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut rgb = Vec::with_capacity(px * 3);
    for i in 0..px {
        let s = &samples[i * c..i * c + c];
        match c {
            1 => rgb.extend_from_slice(&[q(s[0]), q(s[0]), q(s[0])]), // grayscale → RGB
            _ => rgb.extend_from_slice(&[q(s[0]), q(s[1]), q(s[2])]), // RGB (drop alpha if c==4)
        }
    }
    Ok(events::base64::encode(&png::encode_rgb8(&rgb, w, h)))
}

/// The image handler shared by the OpenAI and OpenRouter surfaces (identical
/// `CreateImageRequest`/`ImagesResponse` grammar; the provider only shapes errors).
/// Non-streaming: dispatch the resolved text-to-image action once per requested image
/// through [`bridge::submit`] (so admission → 429 is enforced as for chat), collect
/// one `b64_json` PNG per image. Streaming (`stream:true`): map the denoise-step
/// progress to OpenAI image streaming events (see [`stream_images`]).
pub async fn handle_images(state: AppState, body: Bytes) -> Response {
    let provider = state.provider;
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return ApiError::invalid_request(provider, format!("invalid JSON body: {e}")).into_response(),
    };
    let mut req = match parse_image_request(provider, &body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    // Resolve the model → the text-to-image action to dispatch (404 if unknown/non-image).
    // On the OpenRouter surface this also strips a `"<provider>/"` prefix and walks a
    // `models` fallback array; OpenAI is exact-match only. The resolved local id
    // replaces the requested one for dispatch.
    let manifests = state.exec.manifests(); // one catalog snapshot per request (see catalog::resolve_chat)
    let action = match catalog::resolve_model(provider, &body, |id| catalog::resolve_image(&manifests, id)) {
        Some((id, action)) => {
            req.model = id;
            action
        }
        None => {
            let requested = req.model.clone();
            match bridge::ensure_and_recheck(&state, provider, &requested, |id| catalog::resolve_image(&state.exec.manifests(), id)).await {
                Ok(action) => action,
                Err(e) => return e.into_response(),
            }
        }
    };

    if req.stream {
        return stream_images(state, req, action).await;
    }

    let mut data: Vec<Value> = Vec::with_capacity(req.n as usize);
    for i in 0..req.n {
        let outcome = match bridge::submit(&state, &req.model, &action, image_invocation(&req, i)).await {
            Ok(o) => o,
            Err(e) => return e.into_response(),
        };
        match image_b64_from_outcome(&outcome) {
            Ok(b64) => data.push(json!({ "b64_json": b64 })),
            Err(e) => return ApiError::invalid_request(provider, e).into_response(),
        }
    }
    Json(json!({ "created": now_unix(), "data": data })).into_response()
}

/// The `size` value the streaming events may carry: the request size if it is in the
/// stream schema's enum, else `"auto"` (arbitrary sizes are not a valid stream enum).
fn stream_size(label: &str) -> &'static str {
    if STREAM_SIZES.contains(&label) {
        // Return a 'static copy from the whitelist (the event schema constrains this).
        STREAM_SIZES.iter().copied().find(|s| *s == label).unwrap()
    } else {
        "auto"
    }
}

/// One `image_generation.partial_image` event (progress tick, no true pixels — brain
/// does not expose intermediate denoise latents, so `b64_json` is empty).
fn partial_event(size: &str, index: u32) -> Value {
    json!({
        "type": "image_generation.partial_image",
        "b64_json": "",
        "created_at": now_unix(),
        "size": size,
        "quality": "auto",
        "background": "auto",
        "output_format": "png",
        "partial_image_index": index,
    })
}

/// The terminal `image_generation.completed` event carrying the final PNG `b64_json`.
fn completed_event(size: &str, b64: &str) -> Value {
    json!({
        "type": "image_generation.completed",
        "b64_json": b64,
        "created_at": now_unix(),
        "size": size,
        "quality": "auto",
        "background": "auto",
        "output_format": "png",
        // brain does not meter image tokens; report a zeroed usage (schema-required).
        "usage": {
            "total_tokens": 0, "input_tokens": 0, "output_tokens": 0,
            "input_tokens_details": { "text_tokens": 0, "image_tokens": 0 },
        },
    })
}

/// The SSE image stream: run the admission race FIRST (a shed request is a plain 429,
/// not an event-stream), then map each denoise-step progress tick to an
/// `image_generation.partial_image` event and the final image to a terminal
/// `image_generation.completed` event. Only the first image (`n` is effectively 1 for
/// the streaming surface) is streamed. Partial events carry no pixels (brain exposes
/// no intermediate latents); the completed event carries the real PNG.
async fn stream_images(state: AppState, req: ImageRequest, action: String) -> Response {
    use futures::StreamExt;
    let provider = state.provider;
    let mut src = match bridge::stream_progress(&state, &req.model, &action, image_invocation(&req, 0)).await {
        Ok(src) => src,
        Err(e) => return e.into_response(),
    };
    let size = stream_size(&req.size_label).to_string();
    let events = async_stream::stream! {
        let mut idx = 0u32;
        while let Some(msg) = src.next().await {
            match msg {
                StreamMsg::Progress(..) => {
                    yield Ok::<Event, Infallible>(Event::default().data(partial_event(&size, idx).to_string()));
                    idx += 1;
                }
                StreamMsg::Fetching(p) => {
                    yield Ok(Event::default().comment(p.comment_text()));
                }
                StreamMsg::Delta(_) | StreamMsg::Event(_) => {} // image generation has no text/chat-event deltas
                StreamMsg::Done(outcome) => match image_b64_from_outcome(&outcome) {
                    Ok(b64) => yield Ok(Event::default().data(completed_event(&size, &b64).to_string())),
                    Err(e) => yield Ok(Event::default().data(ApiError::invalid_request(provider, e).body().to_string())),
                },
                StreamMsg::Err(e) => {
                    yield Ok(Event::default().data(e.body().to_string()));
                    return;
                }
            }
        }
    };
    Sse::new(events.boxed()).into_response()
}

/// The SSE `chat.completion.chunk` stream: a role chunk, one content chunk per token
/// delta, a terminal chunk carrying `finish_reason`, an optional usage-only chunk
/// (when `stream_options.include_usage`), then `data: [DONE]`. Runs the admission
/// race FIRST: if the job cannot start on a lane within `state.admit_deadline`, this
/// returns a plain 429 body (with `Retry-After`) instead of an event-stream.
async fn stream_chat(state: AppState, model: String, inv: Invocation, native: bool, want_usage: bool) -> Response {
    // Admit BEFORE returning the SSE body — a shed request is a plain 429, not an
    // event-stream that immediately errors.
    let src = match bridge::stream(&state, &model, "generate", inv).await {
        Ok(src) => src,
        Err(e) => return e.into_response(),
    };
    render_chat_stream(src, model, native, want_usage)
}

/// Like [`stream_chat`], but for a `model` that ISN'T already resident and
/// classifies `Fetchable`: opens the SSE body immediately and interleaves
/// [`StreamMsg::Fetching`] progress (as SSE comment lines) ahead of the usual
/// chunks — see [`bridge::stream_with_autofetch`]. Never called for a model
/// that's already resident or that classifies `Unknown`/has no supplier —
/// those stay a plain, zero-I/O 404 (see `handle_chat`).
fn stream_chat_with_autofetch(state: AppState, supplier: std::sync::Arc<dyn residency::ModelSupplier>, model: String, inv: Invocation, native: bool, want_usage: bool) -> Response {
    let src = bridge::stream_with_autofetch(&state, supplier, &model, "generate", inv, false);
    render_chat_stream(src, model, native, want_usage)
}

fn render_chat_stream(mut src: bridge::EventStream, model: String, native: bool, want_usage: bool) -> Response {
    use futures::StreamExt;
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let events = async_stream::stream! {
        // The role chunk announces an assistant message, and it is emitted
        // LAZILY - only once this request has actually produced something to
        // put in one.
        //
        // Emitting it up front is what let a failed request read as a
        // successful empty one: a request that died inside the model (a lane
        // panic - a real 30B load OOM'd a card exactly this way) still sent a
        // syntactically valid `{"role":"assistant","content":""}` chunk before
        // the error, and a client that accumulates `choices` and stops at
        // `[DONE]` reported success with empty text. With nothing emitted
        // before the failure, such a stream now carries NO assistant-message
        // frame at all - only the named `error` event - so there is no empty
        // success left to mistake it for.
        let mut announced = false;
        macro_rules! announce {
            () => {
                if !announced {
                    announced = true;
                    yield Ok::<Event, Infallible>(Event::default().data(chunk(&id, &model, json!({ "role": "assistant", "content": "" }), None, native).to_string()));
                }
            };
        }

        let mut finish = String::from("stop");
        let (mut prompt, mut completion) = (0i64, 0i64);
        // Whether any real token delta was streamed. A model that only reports
        // coarse `Progress::step` ticks (no `delta`) - `brain/qwen3omnimoe` is one -
        // produces its whole answer in the terminal `Outcome` and would
        // otherwise stream a syntactically valid but EMPTY assistant message.
        // See the `Done` arm below for the one-shot fallback chunk.
        let mut saw_delta = false;
        // The terminal outcome's full text, emitted as a single content chunk
        // only when nothing was streamed incrementally.
        let mut final_text = String::new();
        while let Some(msg) = src.next().await {
            match msg {
                StreamMsg::Delta(piece) => {
                    saw_delta = true;
                    announce!();
                    yield Ok(Event::default().data(chunk(&id, &model, json!({ "content": piece }), None, native).to_string()));
                }
                StreamMsg::Event(v) => {
                    // reasoning_content / tool_calls deltas only — never delta.content
                    // (see `event_delta`'s doc comment).
                    if let Some(delta) = event_delta(&v) {
                        announce!();
                        yield Ok(Event::default().data(chunk(&id, &model, delta, None, native).to_string()));
                    }
                }
                StreamMsg::Progress(..) => {} // chat streams token deltas, not coarse steps
                StreamMsg::Fetching(p) => {
                    yield Ok(Event::default().comment(p.comment_text()));
                }
                StreamMsg::Done(outcome) => {
                    announce!();
                    let co = bridge::read_chat_outcome(&outcome);
                    prompt = co.prompt_tokens;
                    completion = co.completion_tokens;
                    finish = co.finish;
                    final_text = co.text;
                }
                StreamMsg::Err(e) => {
                    // Surface the error as a NAMED `error` SSE event (mirrors the
                    // Anthropic surface, see `anthropic.rs`'s `StreamMsg::Err` arm),
                    // then terminate. The payload itself stays a bare `{"error":
                    // ...}` object - real OpenAI does the same for a mid-stream
                    // failure, and there is no `finish_reason` value in the actual
                    // OpenAI schema that means "error" (`stop`/`length`/
                    // `tool_calls`/`content_filter`/`function_call` only - see
                    // `tests/specs/openai.json`'s `CreateChatCompletionStreamResponse`),
                    // so forcing this into a normal chunk shape would just be a
                    // different lie. `event("error")` is the addition: it was
                    // previously unnamed, so a client that dispatches on SSE event
                    // name (rather than inspecting every `data:` payload) had no
                    // signal to key off at all.
                    yield Ok(Event::default().event("error").data(e.body().to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }
        // Fallback for a resident that never emitted a token delta: stream the
        // completed outcome's text as ONE content chunk so the assistant message
        // isn't empty. Invisible to residents that DO stream (`saw_delta`).
        if !saw_delta && !final_text.is_empty() {
            yield Ok(Event::default().data(chunk(&id, &model, json!({ "content": final_text }), None, native).to_string()));
        }

        // A stream that reached its terminal outcome always carries an
        // assistant message, even an empty one - that IS a successful empty
        // answer, and is what distinguishes it from the failure path above.
        if !announced {
            yield Ok(Event::default().data(chunk(&id, &model, json!({ "role": "assistant", "content": "" }), None, native).to_string()));
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
