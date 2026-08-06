// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Anthropic Messages surface: `POST /v1/messages` (non-streaming + SSE event
//! streaming) and `POST /v1/messages/count_tokens`. Chat dispatches to the shared
//! executor's `generate` action via [`crate::bridge`]; the streaming event order
//! follows Anthropic's `message_start → content_block_start → content_block_delta*
//! → content_block_stop → message_delta → message_stop` sequence.

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
use crate::state::AppState;
use crate::surface::Provider;

const PROVIDER: Provider = Provider::Anthropic;

/// Anthropic-specific routes (merged onto the shared `/models` router).
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
}

/// `POST /v1/messages` — real chat (non-stream + SSE) on the Anthropic dialect.
async fn messages(State(state): State<AppState>, body: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return ApiError::invalid_request(PROVIDER, format!("invalid JSON body: {e}")).into_response(),
    };
    let (model, inv, stream) = match to_invocation(&body) {
        Ok(x) => x,
        Err(e) => return e.into_response(),
    };
    // A legacy short name (e.g. "mock") is a deprecation, not a second id: it
    // resolves to its canonical `brain/<name>` form here, before the catalog
    // lookup and before it is echoed back into any response body (see
    // `modelref::alias`'s module docs). OpenAI/OpenRouter get the same
    // treatment inside `catalog::candidates`; Anthropic has no candidate list
    // (exact-match only), so it resolves directly.
    let model = brain_modelref::alias::canonical(&model).map(str::to_string).unwrap_or(model);
    if catalog::resolve_chat(&state.exec, &model) {
        if stream {
            let est_input = heuristic_tokens(&request_text(&body));
            stream_messages(state, model, inv, est_input).await
        } else {
            match bridge::submit(&state, &model, "generate", inv).await {
                Ok(outcome) => {
                    let (text, prompt, completion, finish) = bridge::read_outcome(&outcome);
                    Json(from_outcome(&model, &text, prompt, completion, &finish)).into_response()
                }
                Err(e) => e.into_response(),
            }
        }
    } else if stream {
        // Not already resident. Cheap, zero-I/O classify BEFORE opening any SSE
        // body: an Unknown/no-supplier model stays a plain 404, matching the
        // non-streaming path below and never opening a stream that would just
        // immediately error.
        match state.supplier.clone() {
            Some(supplier) if matches!(supplier.classify(&model), residency::Supply::Fetchable) => {
                let est_input = heuristic_tokens(&request_text(&body));
                stream_messages_with_autofetch(state, supplier, model, inv, est_input)
            }
            _ => ApiError::model_not_found(PROVIDER, &model).into_response(),
        }
    } else {
        match bridge::ensure_and_recheck(&state, PROVIDER, &model, |id| catalog::resolve_chat(&state.exec, id).then_some(())).await {
            Ok(()) => match bridge::submit(&state, &model, "generate", inv).await {
                Ok(outcome) => {
                    let (text, prompt, completion, finish) = bridge::read_outcome(&outcome);
                    Json(from_outcome(&model, &text, prompt, completion, &finish)).into_response()
                }
                Err(e) => e.into_response(),
            },
            Err(e) => e.into_response(),
        }
    }
}

/// `POST /v1/messages/count_tokens` — an APPROXIMATE input-token count.
///
/// NOTE: this is a heuristic (total content chars / 4), NOT a real tokenizer count.
/// Replace with the served model's actual tokenizer once chat tokenization is wired.
async fn count_tokens(State(_state): State<AppState>, body: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return ApiError::invalid_request(PROVIDER, format!("invalid JSON body: {e}")).into_response(),
    };
    Json(json!({ "input_tokens": heuristic_tokens(&request_text(&body)) })).into_response()
}

/// Reject a request that uses tool-calling features this surface does not support
/// yet: a top-level `tools` array, or any message content block of type
/// `tool_use`/`tool_result`. An explicit 400 (not a silent drop) — full Anthropic
/// tool-calling (the block-index streaming restructure it needs) is a documented
/// follow-up, out of scope here.
fn reject_unsupported_tools(body: &Value) -> Result<(), ApiError> {
    if body.get("tools").map(|v| !v.is_null()).unwrap_or(false) {
        return Err(ApiError::invalid_request(PROVIDER, "'tools' is not supported on this surface yet"));
    }
    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for m in msgs {
            if let Some(blocks) = m.get("content").and_then(|v| v.as_array()) {
                for b in blocks {
                    if matches!(b.get("type").and_then(|v| v.as_str()), Some("tool_use") | Some("tool_result")) {
                        return Err(ApiError::invalid_request(PROVIDER, "'tool_use'/'tool_result' content blocks are not supported on this surface yet"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Parse + validate an Anthropic Messages request into `(model, invocation, stream)`.
/// Enforces `model`/`messages`/`max_tokens` present, rejects unsupported
/// tool-calling ([`reject_unsupported_tools`]); builds the contract `generate`
/// invocation (Anthropic's top-level `system` maps to the `system` param).
pub fn to_invocation(body: &Value) -> Result<(String, Invocation, bool), ApiError> {
    reject_unsupported_tools(body)?;
    let model = body.get("model").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).ok_or_else(|| ApiError::invalid_request(PROVIDER, "'model' is required"))?;
    let messages = body.get("messages").and_then(|v| v.as_array()).filter(|a| !a.is_empty()).ok_or_else(|| ApiError::invalid_request(PROVIDER, "'messages' must be a non-empty array"))?;
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_i64()).ok_or_else(|| ApiError::invalid_request(PROVIDER, "'max_tokens' is required"))?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let msgs: Vec<Value> = messages.iter().map(flatten_message).collect();
    let mut inv = Invocation::new()
        .set("messages", json!(serde_json::to_string(&msgs).unwrap_or_else(|_| "[]".into())))
        .set("max_new", json!(max_tokens))
        .set("temp", json!(body.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0)))
        .set("top_p", json!(body.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0)))
        // See openai.rs's to_invocation: 40 is the standard top-k default.
        .set("top_k", json!(body.get("top_k").and_then(|v| v.as_i64()).unwrap_or(40)))
        .set("seed", json!(body.get("seed").and_then(|v| v.as_i64()).unwrap_or(0)));
    let system = system_text(body.get("system"));
    if !system.is_empty() {
        inv = inv.set("system", json!(system));
    }
    if let Some(stops) = body.get("stop_sequences").and_then(|v| v.as_array()).filter(|a| !a.is_empty()) {
        inv = inv.set("stop", json!(serde_json::to_string(stops).unwrap_or_default()));
    }
    Ok((model.to_string(), inv, stream))
}

/// One Anthropic message → the contract `{role, content}` (blocks flattened to text).
fn flatten_message(m: &Value) -> Value {
    let role = match m.get("role").and_then(|v| v.as_str()).unwrap_or("user") {
        "assistant" => "assistant",
        _ => "user",
    };
    json!({ "role": role, "content": content_text(m.get("content")) })
}

/// Flatten Anthropic content (string, or an array of blocks) to its text.
fn content_text(c: Option<&Value>) -> String {
    match c {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Flatten the top-level `system` (string, or an array of text blocks) to text.
fn system_text(s: Option<&Value>) -> String {
    match s {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(_)) => content_text(s),
        _ => String::new(),
    }
}

/// All request text (every message's content + system) — the input to the
/// approximate token count.
fn request_text(body: &Value) -> String {
    let mut acc = system_text(body.get("system"));
    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for m in msgs {
            acc.push('\n');
            acc.push_str(&content_text(m.get("content")));
        }
    }
    acc
}

/// Approximate token count: total chars / 4 (min 1 for non-empty text). NOTE: a
/// placeholder for a real tokenizer.
fn heuristic_tokens(text: &str) -> i64 {
    let n = text.chars().count();
    if n == 0 {
        0
    } else {
        (n as i64 / 4).max(1)
    }
}

/// Map the contract `finish_reason` to Anthropic's `stop_reason` enum.
fn stop_reason(fr: &str) -> &'static str {
    match fr {
        "length" => "max_tokens",
        "stop_sequence" => "stop_sequence",
        _ => "end_turn",
    }
}

/// The non-streaming `Message` response body.
pub fn from_outcome(model: &str, text: &str, prompt: i64, completion: i64, finish: &str) -> Value {
    json!({
        "id": format!("msg_{}", Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "content": [ { "type": "text", "text": text } ],
        "model": model,
        "stop_reason": stop_reason(finish),
        "stop_sequence": Value::Null,
        "usage": { "input_tokens": prompt, "output_tokens": completion },
    })
}

/// The SSE Messages event stream in Anthropic's fixed order. `est_input` is the
/// approximate input-token count surfaced in `message_start` (the real prompt-token
/// count is not known until generation completes).
async fn stream_messages(state: AppState, model: String, inv: Invocation, est_input: i64) -> Response {
    // Admit BEFORE returning the SSE body — a shed request is a plain 429, not an
    // event-stream that immediately errors.
    let src = match bridge::stream(&state, &model, "generate", inv).await {
        Ok(src) => src,
        Err(e) => return e.into_response(),
    };
    render_messages_stream(src, model, est_input)
}

/// Like [`stream_messages`], but for a `model` that ISN'T already resident and
/// classifies `Fetchable`: opens the SSE body immediately and interleaves
/// [`StreamMsg::Fetching`] progress (as SSE comment lines) ahead of the usual
/// events — see [`bridge::stream_with_autofetch`]. Never called for a model
/// that's already resident or classifies `Unknown`/has no supplier.
fn stream_messages_with_autofetch(state: AppState, supplier: std::sync::Arc<dyn residency::ModelSupplier>, model: String, inv: Invocation, est_input: i64) -> Response {
    let src = bridge::stream_with_autofetch(&state, supplier, &model, "generate", inv, false);
    render_messages_stream(src, model, est_input)
}

fn render_messages_stream(mut src: bridge::EventStream, model: String, est_input: i64) -> Response {
    use futures::StreamExt;
    let id = format!("msg_{}", Uuid::new_v4().simple());
    let events = async_stream::stream! {
        // message_start: an empty-content Message carrying the input-token usage.
        let start = json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": { "input_tokens": est_input, "output_tokens": 0 },
            },
        });
        yield Ok::<Event, Infallible>(Event::default().event("message_start").data(start.to_string()));

        // content_block_start: the single text block.
        yield Ok(Event::default().event("content_block_start").data(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "text", "text": "" },
        }).to_string()));

        let mut finish = String::from("stop");
        let mut completion = 0i64;
        while let Some(msg) = src.next().await {
            match msg {
                StreamMsg::Delta(piece) => {
                    yield Ok(Event::default().event("content_block_delta").data(json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": { "type": "text_delta", "text": piece },
                    }).to_string()));
                }
                StreamMsg::Progress(..) => {} // chat streams token deltas, not coarse steps
                // Reasoning/tool-call events: no Anthropic surface shape yet (full
                // tool-calling is a documented follow-up — see
                // `reject_unsupported_tools`); dropped rather than misrendered.
                StreamMsg::Event(_) => {}
                StreamMsg::Fetching(p) => {
                    yield Ok(Event::default().comment(p.comment_text()));
                }
                StreamMsg::Done(outcome) => {
                    let (_t, _p, c, fr) = bridge::read_outcome(&outcome);
                    completion = c;
                    finish = fr;
                }
                StreamMsg::Err(e) => {
                    yield Ok(Event::default().event("error").data(json!({
                        "type": "error",
                        "error": e.body().get("error").cloned().unwrap_or(Value::Null),
                    }).to_string()));
                    return;
                }
            }
        }

        // content_block_stop → message_delta (stop_reason + cumulative output) → message_stop.
        yield Ok(Event::default().event("content_block_stop").data(json!({
            "type": "content_block_stop", "index": 0,
        }).to_string()));
        yield Ok(Event::default().event("message_delta").data(json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason(&finish), "stop_sequence": Value::Null },
            "usage": { "output_tokens": completion },
        }).to_string()));
        yield Ok(Event::default().event("message_stop").data(json!({ "type": "message_stop" }).to_string()));
    };
    Sse::new(events.boxed()).into_response()
}
