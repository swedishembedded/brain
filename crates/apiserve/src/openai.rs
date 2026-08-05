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
/// enforced exactly as for chat); the response collects one `data` entry per input,
/// indexed in request order.
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
    match catalog::resolve_model(provider, &body, |id| catalog::resolve_embed(&state.exec, id).then_some(())) {
        Some((id, ())) => req.model = id,
        None => match bridge::ensure_and_recheck(&state, provider, &req.model, |id| catalog::resolve_embed(&state.exec, id).then_some(())).await {
            Ok(()) => {}
            Err(e) => return e.into_response(),
        },
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
    let (requested, inv, stream) = match to_invocation(provider, &body) {
        Ok(x) => x,
        Err(e) => return e.into_response(),
    };
    let want_usage = || body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(|v| v.as_bool()).unwrap_or(false);

    // Resolve the model against the chat-capable manifests before dispatching. On the
    // OpenRouter surface this also strips a `"<provider>/"` prefix and walks a `models`
    // fallback array (see [`catalog::resolve_model`]); OpenAI is exact-match only.
    match catalog::resolve_model(provider, &body, |id| catalog::resolve_chat(&state.exec, id).then_some(())) {
        Some((model, ())) => {
            if stream {
                stream_chat(state, model, inv, native, want_usage()).await
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
        None => match bridge::ensure_and_recheck(&state, provider, &requested, |id| catalog::resolve_chat(&state.exec, id).then_some(())).await {
            Ok(()) => match bridge::submit(&state, &requested, "generate", inv).await {
                Ok(outcome) => {
                    let (text, prompt, completion, finish) = bridge::read_outcome(&outcome);
                    Json(non_stream_body(&requested, &text, prompt, completion, &finish, native)).into_response()
                }
                Err(e) => e.into_response(),
            },
            Err(e) => e.into_response(),
        },
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
/// ignored. An optional `seed` (non-standard, honoured when present) seeds generation.
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

    let seed = body.get("seed").and_then(|v| v.as_i64()).unwrap_or(0);
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    Ok(ImageRequest { model, prompt, n, width, height, size_label, seed, stream })
}

/// The image-action [`Invocation`] for the `i`-th requested image: prompt + size +
/// a per-image seed (so `n>1` yields distinct images from a seed-driven model).
fn image_invocation(req: &ImageRequest, i: u32) -> Invocation {
    Invocation::new()
        .set("prompt", json!(req.prompt))
        .set("width", json!(req.width))
        .set("height", json!(req.height))
        .set("seed", json!(req.seed.wrapping_add(i as i64)))
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
    let action = match catalog::resolve_model(provider, &body, |id| catalog::resolve_image(&state.exec, id)) {
        Some((id, action)) => {
            req.model = id;
            action
        }
        None => {
            let requested = req.model.clone();
            match bridge::ensure_and_recheck(&state, provider, &requested, |id| catalog::resolve_image(&state.exec, id)).await {
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
                StreamMsg::Delta(_) => {} // image generation has no text deltas
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
        // First chunk: announce the assistant role.
        yield Ok::<Event, Infallible>(Event::default().data(chunk(&id, &model, json!({ "role": "assistant", "content": "" }), None, native).to_string()));

        let mut finish = String::from("stop");
        let (mut prompt, mut completion) = (0i64, 0i64);
        while let Some(msg) = src.next().await {
            match msg {
                StreamMsg::Delta(piece) => {
                    yield Ok(Event::default().data(chunk(&id, &model, json!({ "content": piece }), None, native).to_string()));
                }
                StreamMsg::Progress(..) => {} // chat streams token deltas, not coarse steps
                StreamMsg::Fetching(p) => {
                    yield Ok(Event::default().comment(p.comment_text()));
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
