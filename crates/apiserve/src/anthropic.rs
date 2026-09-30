// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Anthropic Messages surface: `POST /v1/messages` (non-streaming + SSE event
//! streaming) and `POST /v1/messages/count_tokens`. Chat dispatches to the shared
//! executor's `generate` action via [`crate::bridge`]; the streaming event order
//! follows Anthropic's `message_start → content_block_start → content_block_delta*
//! → content_block_stop → message_delta → message_stop` sequence. A model's
//! reasoning is a `thinking` block ahead of the answer's `text` block, streamed as
//! `thinking_delta`s; the request's `thinking` config turns it on or off.

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
    if catalog::resolve_chat(&state.exec.manifests(), &model) {
        if stream {
            let est_input = heuristic_tokens(&request_text(&body));
            stream_messages(state, model, inv, est_input).await
        } else {
            match bridge::submit(&state, &model, "generate", inv).await {
                Ok(outcome) => Json(from_outcome(&model, &bridge::read_chat_outcome(&outcome))).into_response(),
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
        match bridge::ensure_and_recheck(&state, PROVIDER, &model, |id| catalog::resolve_chat(&state.exec.manifests(), id).then_some(())).await {
            Ok(()) => match bridge::submit(&state, &model, "generate", inv).await {
                Ok(outcome) => Json(from_outcome(&model, &bridge::read_chat_outcome(&outcome))).into_response(),
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

/// The smallest `thinking.budget_tokens` Anthropic accepts.
const MIN_THINKING_BUDGET: i64 = 1024;

/// Anthropic's `thinking` config as the contract's `enable_thinking`:
/// `{"type": "enabled", "budget_tokens": N}` (Anthropic's own bounds,
/// `1024 <= N < max_tokens`) turns reasoning on, `{"type": "disabled"}` off,
/// and an absent config leaves the model's own default. The budget is the
/// target Anthropic defines it as: reasoning counts against `max_tokens`, which
/// is what bounds it.
fn thinking(body: &Value, max_tokens: i64) -> Result<Option<bool>, ApiError> {
    let Some(cfg) = body.get("thinking").filter(|v| !v.is_null()) else { return Ok(None) };
    let bad = |msg: String| Err(ApiError::invalid_request(PROVIDER, msg));
    match cfg.get("type").and_then(Value::as_str) {
        Some("disabled") => Ok(Some(false)),
        Some("enabled") => match cfg.get("budget_tokens").and_then(Value::as_i64) {
            Some(n) if (MIN_THINKING_BUDGET..max_tokens).contains(&n) => Ok(Some(true)),
            Some(n) => bad(format!("'thinking.budget_tokens' must be at least {MIN_THINKING_BUDGET} and less than 'max_tokens' ({max_tokens}), got {n}")),
            None => bad("'thinking.budget_tokens' is required when thinking is enabled".to_string()),
        },
        _ => bad("'thinking' must be {\"type\": \"enabled\", \"budget_tokens\": N} or {\"type\": \"disabled\"}".to_string()),
    }
}

/// Parse + validate an Anthropic Messages request into `(model, invocation, stream)`.
/// Enforces `model`/`messages`/`max_tokens` present, rejects unsupported
/// tool-calling ([`reject_unsupported_tools`]); builds the contract `generate`
/// invocation (Anthropic's top-level `system` maps to the `system` param, and
/// `thinking` to `enable_thinking`).
pub fn to_invocation(body: &Value) -> Result<(String, Invocation, bool), ApiError> {
    reject_unsupported_tools(body)?;
    let model = body.get("model").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).ok_or_else(|| ApiError::invalid_request(PROVIDER, "'model' is required"))?;
    let messages = body.get("messages").and_then(|v| v.as_array()).filter(|a| !a.is_empty()).ok_or_else(|| ApiError::invalid_request(PROVIDER, "'messages' must be a non-empty array"))?;
    let max_tokens = crate::sampling::max_tokens(PROVIDER, "max_tokens", body.get("max_tokens"))?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let msgs: Vec<Value> = messages.iter().map(flatten_message).collect();
    let inv = Invocation::new()
        .set("messages", json!(serde_json::to_string(&msgs).unwrap_or_else(|_| "[]".into())))
        .set("max_new", json!(max_tokens));
    let mut inv = crate::sampling::apply(PROVIDER, body, inv)?;
    if let Some(on) = thinking(body, max_tokens)? {
        inv = inv.set("enable_thinking", json!(on));
    }
    let system = system_text(body.get("system"));
    if !system.is_empty() {
        inv = inv.set("system", json!(system));
    }
    if let Some(stops) = body.get("stop_sequences").and_then(|v| v.as_array()).filter(|a| !a.is_empty()) {
        inv = inv.set("stop", json!(serde_json::to_string(stops).unwrap_or_default()));
    }

    // image content blocks, previously silently dropped by flatten_message/
    // content_text (which only ever kept "text" blocks) -- see
    // crate::media's module doc.
    let media = crate::media::extract_anthropic(messages).map_err(|e| ApiError::invalid_request(PROVIDER, e))?;
    if let Some(img) = media.image {
        inv = inv.blob("image", img);
    }

    Ok((model.to_string(), inv, stream))
}

/// One Anthropic message → the contract `{role, content, reasoning_content?}`:
/// text blocks flattened to `content`, an assistant turn's `thinking` blocks to
/// its `reasoning_content` (the chat template keeps or drops it, as the model
/// was trained).
fn flatten_message(m: &Value) -> Value {
    let role = match m.get("role").and_then(|v| v.as_str()).unwrap_or("user") {
        "assistant" => "assistant",
        _ => "user",
    };
    let mut out = json!({ "role": role, "content": content_text(m.get("content")) });
    let reasoning: String = m
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
        .filter_map(|b| b.get("thinking").and_then(Value::as_str))
        .collect();
    if role == "assistant" && !reasoning.is_empty() {
        out["reasoning_content"] = json!(reasoning);
    }
    out
}

/// Flatten Anthropic content (string, or an array of blocks) to the text of its
/// `text` blocks.
fn content_text(c: Option<&Value>) -> String {
    match c {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str).is_none_or(|t| t == "text"))
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

/// A `thinking` content block. brain has no signing key, and Anthropic's
/// signature only lets its own service verify a block it issued, so the
/// signature is empty; the block round-trips through a client's history as
/// any other.
fn thinking_block(thinking: &str) -> Value {
    json!({ "type": "thinking", "thinking": thinking, "signature": "" })
}

/// The non-streaming `Message` response body: the reasoning (when there is
/// any) as a `thinking` block, then the answer as a `text` block.
pub fn from_outcome(model: &str, co: &bridge::ChatOutcome) -> Value {
    let mut content = Vec::new();
    if !co.reasoning.is_empty() {
        content.push(thinking_block(&co.reasoning));
    }
    content.push(json!({ "type": "text", "text": co.text }));
    json!({
        "id": format!("msg_{}", Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason(&co.finish),
        "stop_sequence": Value::Null,
        "usage": { "input_tokens": co.prompt_tokens, "output_tokens": co.completion_tokens },
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

        // Blocks open as their content arrives: the reasoning's `thinking`
        // block first when the model reasons, then the answer's `text` block,
        // which every completed message carries even when empty.
        let mut blocks = Blocks::default();
        let mut finish = String::from("stop");
        let mut completion = 0i64;
        // Whether any real token delta was streamed. A resident that only
        // reports coarse `Progress::step` ticks (no `delta`) - `brain/qwen3omnimoe` is
        // one - carries its whole answer in the terminal `Outcome`, and would
        // otherwise stream a well-formed but EMPTY text block. See the one-shot
        // fallback after the loop.
        let mut saw_delta = false;
        let mut final_text = String::new();
        while let Some(msg) = src.next().await {
            match msg {
                StreamMsg::Delta(piece) => {
                    saw_delta = true;
                    for ev in blocks.delta(BlockKind::Text, &piece) {
                        yield Ok(ev);
                    }
                }
                StreamMsg::Progress(..) => {} // chat streams token deltas, not coarse steps
                StreamMsg::Event(v) => {
                    // Tool-call events have no shape here: tools are refused up
                    // front (`reject_unsupported_tools`).
                    if v.get("kind").and_then(Value::as_str) == Some("reasoning") {
                        let piece = v.get("text").and_then(Value::as_str).unwrap_or_default();
                        for ev in blocks.delta(BlockKind::Thinking, piece) {
                            yield Ok(ev);
                        }
                    }
                }
                StreamMsg::Fetching(p) => {
                    yield Ok(Event::default().comment(p.comment_text()));
                }
                StreamMsg::Done(outcome) => {
                    let (t, _p, c, fr) = bridge::read_outcome(&outcome);
                    completion = c;
                    finish = fr;
                    final_text = t;
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

        // Fallback for a resident that never emitted a token delta: emit the
        // completed outcome's text as ONE `text_delta` so the text block isn't
        // empty. Invisible to residents that DO stream (`saw_delta`).
        let tail = if saw_delta { String::new() } else { final_text };
        for ev in blocks.finish(&tail) {
            yield Ok(ev);
        }

        // message_delta (stop_reason + cumulative output) → message_stop.
        yield Ok(Event::default().event("message_delta").data(json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason(&finish), "stop_sequence": Value::Null },
            "usage": { "output_tokens": completion },
        }).to_string()));
        yield Ok(Event::default().event("message_stop").data(json!({ "type": "message_stop" }).to_string()));
    };
    Sse::new(events.boxed()).into_response()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
}

/// The content blocks of one streamed message: which one is open, and the
/// index the next one takes. A delta of a kind other than the open block's
/// closes that block and opens its own.
#[derive(Default)]
struct Blocks {
    open: Option<BlockKind>,
    next: u32,
    saw_text: bool,
}

impl Blocks {
    /// The events that stream `piece` as `kind`: the open block closed and a
    /// new one started when the kind changes, then the delta itself.
    fn delta(&mut self, kind: BlockKind, piece: &str) -> Vec<Event> {
        let mut out = Vec::new();
        if self.open != Some(kind) {
            out.extend(self.close());
            out.push(self.start(kind));
        }
        if !piece.is_empty() {
            let index = self.next - 1;
            let delta = match kind {
                BlockKind::Thinking => json!({ "type": "thinking_delta", "thinking": piece }),
                BlockKind::Text => json!({ "type": "text_delta", "text": piece }),
            };
            out.push(sse("content_block_delta", json!({ "type": "content_block_delta", "index": index, "delta": delta })));
        }
        out
    }

    /// Close the message's blocks: `tail` streamed as text (the one-shot
    /// fallback), a text block opened if the message has none, and the open
    /// block closed.
    fn finish(&mut self, tail: &str) -> Vec<Event> {
        let mut out = Vec::new();
        if !tail.is_empty() || !self.saw_text {
            out.extend(self.delta(BlockKind::Text, tail));
        }
        out.extend(self.close());
        out
    }

    fn start(&mut self, kind: BlockKind) -> Event {
        let block = match kind {
            BlockKind::Thinking => thinking_block(""),
            BlockKind::Text => json!({ "type": "text", "text": "" }),
        };
        self.saw_text |= kind == BlockKind::Text;
        self.open = Some(kind);
        self.next += 1;
        sse("content_block_start", json!({ "type": "content_block_start", "index": self.next - 1, "content_block": block }))
    }

    /// Close the open block; a thinking block carries its (empty) signature
    /// as the last delta, as Anthropic streams it.
    fn close(&mut self) -> Vec<Event> {
        let Some(kind) = self.open.take() else { return Vec::new() };
        let index = self.next - 1;
        let mut out = Vec::new();
        if kind == BlockKind::Thinking {
            out.push(sse("content_block_delta", json!({ "type": "content_block_delta", "index": index, "delta": { "type": "signature_delta", "signature": "" } })));
        }
        out.push(sse("content_block_stop", json!({ "type": "content_block_stop", "index": index })));
        out
    }
}

/// One named SSE event carrying `data` as JSON.
fn sse(name: &str, data: Value) -> Event {
    Event::default().event(name).data(data.to_string())
}
