// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Anthropic Messages surface: `POST /v1/messages` (non-streaming + SSE event
//! streaming) and `POST /v1/messages/count_tokens`. Chat dispatches to the shared
//! executor's `generate` action via [`crate::bridge`]; the streaming event order
//! follows Anthropic's `message_start → content_block_start → content_block_delta*
//! → content_block_stop → message_delta → message_stop` sequence. A model's
//! reasoning is a `thinking` block ahead of the answer's `text` block, streamed as
//! `thinking_delta`s; the request's `thinking` config turns it on or off.
//!
//! Tool calling maps onto the contract the OpenAI surface uses: a client tool
//! (`name`, `description`, `input_schema`) is handed to the model as a function
//! tool, an assistant turn's `tool_use` blocks become its `tool_calls`, a user
//! turn's `tool_result` blocks become `tool` turns, and the model's calls come
//! back as `tool_use` blocks (streamed as `input_json_delta`s) with
//! `stop_reason: "tool_use"`.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use capability::Invocation;
use serde_json::{json, Value};
use std::convert::Infallible;
use uuid::Uuid;

use crate::auth::CallerExt;
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
async fn messages(State(state): State<AppState>, caller: CallerExt, raw: Bytes) -> Response {
    let state = state.scoped(caller);
    let body: Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(e) => return ApiError::invalid_request(PROVIDER, format!("invalid JSON body: {e}")).into_response(),
    };
    // Media is decoded off the async workers, a few requests at a time
    // (`crate::preprocess`); a text-only request is parsed in place.
    let (body, parsed) = if crate::media::carries_media(&body) {
        match crate::preprocess::off_runtime(PROVIDER, move || {
            let parsed = to_invocation(&body);
            (body, parsed)
        })
        .await
        {
            Ok(done) => done,
            Err(e) => return e.into_response(),
        }
    } else {
        let parsed = to_invocation(&body);
        (body, parsed)
    };
    let (model, inv, stream) = match parsed {
        Ok((model, inv, stream)) => (model, with_tools_as_written(&raw, inv), stream),
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
                Ok(outcome) => message_response(&model, &outcome),
                Err(e) => e.into_response(),
            }
        }
    } else if stream {
        // Not already resident. Cheap, zero-I/O classify BEFORE opening any SSE
        // body: an Unknown/no-supplier model stays a plain 404, matching the
        // non-streaming path below and never opening a stream that would just
        // immediately error.
        match state.supplier_for_caller() {
            Some(supplier) if matches!(supplier.classify(&model), residency::Supply::Fetchable) => {
                let est_input = heuristic_tokens(&request_text(&body));
                stream_messages_with_autofetch(state, supplier, model, inv, est_input)
            }
            _ => ApiError::model_not_found(PROVIDER, &model).into_response(),
        }
    } else {
        match bridge::ensure_and_recheck(&state, PROVIDER, &model, |id| catalog::resolve_chat(&state.exec.manifests(), id).then_some(())).await {
            Ok(()) => match bridge::submit(&state, &model, "generate", inv).await {
                Ok(outcome) => message_response(&model, &outcome),
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

/// The non-streaming reply to a finished generation: the `Message`, or the
/// error [`from_outcome`] refuses it with.
fn message_response(model: &str, outcome: &capability::Outcome) -> Response {
    match from_outcome(model, &bridge::read_chat_outcome(outcome)) {
        Ok(message) => Json(message).into_response(),
        Err(e) => e.into_response(),
    }
}

/// The request's client tools in the contract's function form, `None` when it
/// names none. A tool needs a `name` and an `input_schema` object; a server
/// tool (one with a `type` other than `custom`, such as Anthropic's hosted web
/// search) runs on Anthropic's side and has no counterpart here, so it is
/// refused by name. The result is held to the same bounds as the OpenAI
/// surface's `tools`.
fn tools(body: &Value) -> Result<Option<Value>, ApiError> {
    let Some(tools) = body.get("tools").filter(|v| !v.is_null()) else { return Ok(None) };
    let list = tools.as_array().ok_or_else(|| ApiError::invalid_request(PROVIDER, "'tools' must be an array"))?;
    let mut functions = Vec::with_capacity(list.len());
    for (i, tool) in list.iter().enumerate() {
        if let Some(kind) = tool.get("type").and_then(Value::as_str).filter(|k| *k != "custom") {
            return Err(ApiError::invalid_request(PROVIDER, format!("'tools[{i}]': the server tool '{kind}' is not served here; give client tools (name, description, input_schema)")));
        }
        let schema = tool.get("input_schema").filter(|v| v.is_object()).ok_or_else(|| ApiError::invalid_request(PROVIDER, format!("'tools[{i}].input_schema' must be a JSON schema object")))?;
        let mut function = json!({ "name": tool.get("name").cloned().unwrap_or(Value::Null), "parameters": schema });
        if let Some(description) = tool.get("description").filter(|v| v.is_string()) {
            function["description"] = description.clone();
        }
        functions.push(json!({ "type": "function", "function": function }));
    }
    let functions = Value::Array(functions);
    crate::openai::validate_tools(PROVIDER, &functions)?;
    Ok(Some(functions))
}

/// `inv` with its `tools` built from the text the client wrote: a chat template
/// prints a tool schema member by member, and the parsed request keeps object
/// members sorted, so the model would see a schema's `properties` ahead of its
/// `type`, unlike the client and the model's training prompts. Each tool keeps
/// [`tools`]'s function form; only its members' text comes from the request.
fn with_tools_as_written(raw: &[u8], inv: Invocation) -> Invocation {
    use serde_json::value::RawValue;
    #[derive(serde::Deserialize)]
    struct Tool<'a> {
        #[serde(borrow)]
        name: &'a RawValue,
        #[serde(borrow, default)]
        description: Option<&'a RawValue>,
        #[serde(borrow)]
        input_schema: &'a RawValue,
    }
    #[derive(serde::Deserialize)]
    struct Written<'a> {
        #[serde(borrow, default)]
        tools: Option<Vec<Tool<'a>>>,
    }
    let Some(tools) = serde_json::from_slice::<Written>(raw).ok().and_then(|w| w.tools) else { return inv };
    if inv.get_str("tools").is_none() {
        return inv;
    }
    let functions: Vec<String> = tools
        .iter()
        .map(|t| {
            let description = t.description.map(|d| format!(",\"description\":{}", d.get())).unwrap_or_default();
            format!("{{\"type\":\"function\",\"function\":{{\"name\":{}{description},\"parameters\":{}}}}}", t.name.get(), t.input_schema.get())
        })
        .collect();
    inv.set("tools", json!(format!("[{}]", functions.join(","))))
}

/// Anthropic's `tool_choice` in the contract's OpenAI form: `auto`, `any`
/// (some tool must be called: `required`), `tool` (that one), `none`.
/// `disable_parallel_tool_use: true` is refused: the served models have no
/// control that keeps them to one call, and accepting it would promise one.
fn tool_choice(body: &Value) -> Result<Option<Value>, ApiError> {
    let Some(choice) = body.get("tool_choice").filter(|v| !v.is_null()) else { return Ok(None) };
    if choice.get("disable_parallel_tool_use").and_then(Value::as_bool) == Some(true) {
        return Err(ApiError::invalid_request(PROVIDER, "'tool_choice.disable_parallel_tool_use' is not supported: the model may call several tools in one turn"));
    }
    let mapped = match choice.get("type").and_then(Value::as_str) {
        Some("auto") => json!("auto"),
        Some("any") => json!("required"),
        Some("none") => json!("none"),
        Some("tool") => json!({ "type": "function", "function": { "name": choice.get("name").cloned().unwrap_or(Value::Null) } }),
        _ => return Err(ApiError::invalid_request(PROVIDER, "'tool_choice' must be {\"type\": \"auto\"}, {\"type\": \"any\"}, {\"type\": \"none\"} or {\"type\": \"tool\", \"name\": ...}")),
    };
    crate::openai::validate_tool_choice(PROVIDER, &mapped)?;
    Ok(Some(mapped))
}

/// The smallest `thinking.budget_tokens` Anthropic accepts.
const MIN_THINKING_BUDGET: i64 = 1024;

/// Anthropic's `thinking` config as the contract's `enable_thinking`:
/// `{"type": "enabled", "budget_tokens": N}` (Anthropic's own bounds,
/// `1024 <= N < max_tokens`) turns reasoning on, `{"type": "disabled"}` off,
/// and an absent config leaves the model's own default. The budget is the
/// target Anthropic defines it as - reasoning counts against `max_tokens` - and
/// `to_invocation` also hands it on as the think block's token budget.
fn thinking(body: &Value, max_tokens: i64) -> Result<Option<bool>, ApiError> {
    let Some(cfg) = body.get("thinking").filter(|v| !v.is_null()) else { return Ok(None) };
    let bad = |msg: String| Err(ApiError::invalid_request(PROVIDER, msg));
    match cfg.get("type").and_then(Value::as_str) {
        Some("disabled") => Ok(Some(false)),
        // The model decides whether and how much to reason: its own default.
        Some("adaptive") => Ok(None),
        Some("enabled") => match cfg.get("budget_tokens").and_then(Value::as_i64) {
            Some(n) if (MIN_THINKING_BUDGET..max_tokens).contains(&n) => Ok(Some(true)),
            Some(n) => bad(format!("'thinking.budget_tokens' must be at least {MIN_THINKING_BUDGET} and less than 'max_tokens' ({max_tokens}), got {n}")),
            None => bad("'thinking.budget_tokens' is required when thinking is enabled".to_string()),
        },
        _ => bad("'thinking' must be {\"type\": \"enabled\", \"budget_tokens\": N}, {\"type\": \"adaptive\"} or {\"type\": \"disabled\"}".to_string()),
    }
}

/// Parse + validate an Anthropic Messages request into `(model, invocation, stream)`.
/// Enforces `model`/`messages`/`max_tokens` present; builds the contract
/// `generate` invocation (Anthropic's top-level `system` maps to the `system`
/// param, `thinking` to `enable_thinking`, and `tools`/`tool_choice` to their
/// contract forms - see [`tools`], [`tool_choice`], [`flatten_messages`]).
pub fn to_invocation(body: &Value) -> Result<(String, Invocation, bool), ApiError> {
    let model = body.get("model").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).ok_or_else(|| ApiError::invalid_request(PROVIDER, "'model' is required"))?;
    let messages = body.get("messages").and_then(|v| v.as_array()).filter(|a| !a.is_empty()).ok_or_else(|| ApiError::invalid_request(PROVIDER, "'messages' must be a non-empty array"))?;
    let max_tokens = crate::sampling::max_tokens(PROVIDER, "max_tokens", body.get("max_tokens"))?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let msgs = flatten_messages(messages)?;
    let inv = Invocation::new()
        .set("messages", json!(serde_json::to_string(&msgs).unwrap_or_else(|_| "[]".into())))
        .set("max_new", json!(max_tokens));
    let mut inv = crate::sampling::apply(PROVIDER, body, inv)?;
    if let Some(on) = thinking(body, max_tokens)? {
        inv = inv.set("enable_thinking", json!(on));
        // The validated `budget_tokens` is the think block's token budget.
        if let Some(n) = body.get("thinking").and_then(|t| t.get("budget_tokens")).and_then(Value::as_i64).filter(|_| on) {
            inv = inv.set("thinking_budget", json!(n));
        }
    }
    let system = system_text(body.get("system"));
    if !system.is_empty() {
        inv = inv.set("system", json!(system));
    }
    if let Some(stops) = body.get("stop_sequences").and_then(|v| v.as_array()).filter(|a| !a.is_empty()) {
        inv = inv.set("stop", json!(serde_json::to_string(stops).unwrap_or_default()));
    }
    if let Some(tools) = tools(body)? {
        inv = inv.set("tools", json!(tools.to_string()));
    }
    if let Some(choice) = tool_choice(body)? {
        inv = inv.set("tool_choice", json!(choice.to_string()));
    }

    // image content blocks, previously silently dropped by flatten_message/
    // content_text (which only ever kept "text" blocks) -- see
    // crate::media's module doc.
    let mut media = crate::media::extract_anthropic(messages).map_err(|e| ApiError::invalid_request(PROVIDER, e))?;
    inv = media.attach_images(inv);

    Ok((model.to_string(), inv, stream))
}

/// The conversation in the contract's message shape. An assistant turn keeps
/// its text as `content`, its `thinking` blocks as `reasoning_content` (the chat
/// template keeps or drops it, as the model was trained) and its `tool_use`
/// blocks as `tool_calls`. A user turn's `tool_result` blocks each become a
/// `tool` turn answering its `tool_use_id`, in place among the turn's other
/// blocks, so images keep the order [`crate::media::extract_anthropic`] reads
/// them in. A `system` turn stays a system turn.
fn flatten_messages(messages: &[Value]) -> Result<Vec<Value>, ApiError> {
    let mut out = Vec::with_capacity(messages.len());
    for (i, m) in messages.iter().enumerate() {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = m.get("content");
        let blocks: Vec<Value> = match content {
            Some(Value::Array(blocks)) => blocks.clone(),
            Some(Value::String(text)) => vec![json!({ "type": "text", "text": text })],
            _ => Vec::new(),
        };
        match role {
            "assistant" => out.push(assistant_message(i, &blocks)?),
            "system" => out.push(json!({ "role": "system", "content": message_content(&blocks) })),
            _ => {
                let mut pending: Vec<Value> = Vec::new();
                let mut answered = false;
                for (j, block) in blocks.iter().enumerate() {
                    if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                        pending.push(block.clone());
                        continue;
                    }
                    if !pending.is_empty() {
                        out.push(json!({ "role": "user", "content": message_content(&pending) }));
                        pending.clear();
                    }
                    out.push(tool_message(i, j, block)?);
                    answered = true;
                }
                if !pending.is_empty() || !answered {
                    out.push(json!({ "role": "user", "content": message_content(&pending) }));
                }
            }
        }
    }
    Ok(out)
}

/// An assistant turn: its text, reasoning and tool calls (`input` as the JSON
/// text the contract's `arguments` carries).
fn assistant_message(i: usize, blocks: &[Value]) -> Result<Value, ApiError> {
    let of_type = |kind: &'static str| blocks.iter().filter(move |b| b.get("type").and_then(Value::as_str) == Some(kind));
    let mut out = json!({ "role": "assistant", "content": message_content(blocks) });
    let reasoning: String = of_type("thinking").filter_map(|b| b.get("thinking").and_then(Value::as_str)).collect();
    if !reasoning.is_empty() {
        out["reasoning_content"] = json!(reasoning);
    }
    let mut calls = Vec::new();
    for b in of_type("tool_use") {
        let id = b.get("id").and_then(Value::as_str).filter(|s| !s.is_empty());
        let name = b.get("name").and_then(Value::as_str).filter(|s| !s.is_empty());
        let (Some(id), Some(name)) = (id, name) else {
            return Err(ApiError::invalid_request(PROVIDER, format!("messages[{i}]: a 'tool_use' block needs a non-empty 'id' and 'name'")));
        };
        let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
        calls.push(json!({ "id": id, "name": name, "arguments": input.to_string() }));
    }
    if !calls.is_empty() {
        out["tool_calls"] = json!(calls);
    }
    Ok(out)
}

/// A `tool_result` block as the `tool` turn answering its `tool_use_id`. Its
/// content is a string or blocks (text and images); a result the client marked
/// `is_error` says so first, since the chat template has no field for it.
fn tool_message(i: usize, j: usize, block: &Value) -> Result<Value, ApiError> {
    let id = block.get("tool_use_id").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or_else(|| ApiError::invalid_request(PROVIDER, format!("messages[{i}].content[{j}]: a 'tool_result' block needs a non-empty 'tool_use_id'")))?;
    let mut blocks: Vec<Value> = match block.get("content") {
        Some(Value::Array(blocks)) => blocks.clone(),
        Some(Value::String(text)) => vec![json!({ "type": "text", "text": text })],
        _ => Vec::new(),
    };
    if block.get("is_error").and_then(Value::as_bool) == Some(true) {
        blocks.insert(0, json!({ "type": "text", "text": "Error: " }));
    }
    Ok(json!({ "role": "tool", "tool_call_id": id, "content": message_content(&blocks) }))
}

/// The contract `content` of `blocks`: their text, or - when one is an `image` - their typed parts in order (`text`
/// parts and payload-free `image_url` markers, the shape the OpenAI surface hands over), so a model that reads images
/// sees each where the client put it. The image's pixels travel as a blob ([`crate::media`]).
fn message_content(blocks: &[Value]) -> Value {
    let kind = |b: &Value| b.get("type").and_then(Value::as_str).map(str::to_string);
    if !blocks.iter().any(|b| kind(b).as_deref() == Some("image")) {
        return json!(blocks.iter().filter(|b| kind(b).is_none_or(|k| k == "text")).filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<String>());
    }
    Value::Array(
        blocks
            .iter()
            .filter_map(|b| match kind(b).as_deref() {
                Some("image") => Some(json!({ "type": "image_url", "image_url": {} })),
                None | Some("text") => b.get("text").and_then(Value::as_str).map(|t| json!({ "type": "text", "text": t })),
                _ => None,
            })
            .collect(),
    )
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

/// All request text (every message's content, its tool calls and results,
/// the system prompt and the tool definitions): the input to the approximate
/// token count.
fn request_text(body: &Value) -> String {
    let mut acc = system_text(body.get("system"));
    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for m in msgs {
            acc.push('\n');
            acc.push_str(&content_text(m.get("content")));
            for block in m.get("content").and_then(Value::as_array).into_iter().flatten() {
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => acc.push_str(&block.get("input").map(Value::to_string).unwrap_or_default()),
                    Some("tool_result") => acc.push_str(&content_text(block.get("content"))),
                    _ => {}
                }
            }
        }
    }
    if let Some(tools) = body.get("tools").filter(|v| !v.is_null()) {
        acc.push_str(&tools.to_string());
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
        "tool_calls" => "tool_use",
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

/// A contract tool call (`{id, name, arguments}`, `arguments` the JSON text
/// the model wrote) as a `tool_use` block. Anthropic's `input` is an object, so
/// arguments that are not one are refused rather than passed on as something
/// the client cannot run.
fn tool_use_block(call: &Value) -> Result<Value, ApiError> {
    let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
    let arguments = call.get("arguments").and_then(Value::as_str).unwrap_or("{}");
    let input = serde_json::from_str::<Value>(arguments).ok().filter(Value::is_object).ok_or_else(|| {
        ApiError::failure(PROVIDER, capability::FailureClass::BackendFailure, format!("the model called '{name}' with arguments that are not a JSON object"))
    })?;
    Ok(json!({ "type": "tool_use", "id": call.get("id").and_then(Value::as_str).unwrap_or_default(), "name": name, "input": input }))
}

/// The non-streaming `Message` response body: the reasoning (when there is
/// any) as a `thinking` block, then the answer as a `text` block (left out of a
/// turn that only calls tools), then one `tool_use` block per call.
pub fn from_outcome(model: &str, co: &bridge::ChatOutcome) -> Result<Value, ApiError> {
    let mut content = Vec::new();
    if !co.reasoning.is_empty() {
        content.push(thinking_block(&co.reasoning));
    }
    if !co.text.is_empty() || co.tool_calls.is_empty() {
        content.push(json!({ "type": "text", "text": co.text }));
    }
    for call in &co.tool_calls {
        content.push(tool_use_block(call)?);
    }
    Ok(json!({
        "id": format!("msg_{}", Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason(&co.finish),
        "stop_sequence": Value::Null,
        "usage": { "input_tokens": co.prompt_tokens, "output_tokens": co.completion_tokens },
    }))
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
    match bridge::stream_with_autofetch(&state, supplier, &model, "generate", inv, false) {
        Ok(src) => render_messages_stream(src, model, est_input),
        Err(e) => e.into_response(),
    }
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
        let mut final_calls: Vec<Value> = Vec::new();
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
                    let text = || v.get("text").and_then(Value::as_str).unwrap_or_default();
                    let events = match v.get("kind").and_then(Value::as_str) {
                        Some("reasoning") => blocks.delta(BlockKind::Thinking, text()),
                        Some("tool_call_start") => blocks.tool_start(
                            v.get("id").and_then(Value::as_str).unwrap_or_default(),
                            v.get("name").and_then(Value::as_str).unwrap_or_default(),
                        ),
                        Some("tool_call_args") => blocks.tool_args(text()),
                        Some("tool_call_end") => blocks.close(),
                        // Forward-compatible: an unknown kind adds no block.
                        _ => Vec::new(),
                    };
                    for ev in events {
                        yield Ok(ev);
                    }
                }
                StreamMsg::Fetching(p) => {
                    yield Ok(Event::default().comment(p.comment_text()));
                }
                StreamMsg::Done(outcome) => {
                    let co = bridge::read_chat_outcome(&outcome);
                    completion = co.completion_tokens;
                    finish = co.finish;
                    final_text = co.text;
                    final_calls = co.tool_calls;
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
        // Likewise its tool calls, each as one whole block, when none streamed.
        let calls = if blocks.saw_tool { Vec::new() } else { final_calls };
        match blocks.finish(&tail, &calls) {
            Ok(events) => {
                for ev in events {
                    yield Ok(ev);
                }
            }
            Err(e) => {
                yield Ok(Event::default().event("error").data(json!({
                    "type": "error",
                    "error": e.body().get("error").cloned().unwrap_or(Value::Null),
                }).to_string()));
                return;
            }
        }

        // message_delta (stop_reason + cumulative output) → message_stop.
        yield Ok(Event::default().event("message_delta").data(json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason(&finish), "stop_sequence": Value::Null },
            "usage": { "output_tokens": completion },
        }).to_string()));
        yield Ok(Event::default().event("message_stop").data(json!({ "type": "message_stop" }).to_string()));
    };
    Sse::new(events.boxed()).keep_alive(KeepAlive::default()).into_response()
}

/// A block that streams prose deltas.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
}

/// The block a streamed message has open.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Open {
    Prose(BlockKind),
    Tool,
}

/// The content blocks of one streamed message: which one is open, and the
/// index the next one takes. A delta of a kind other than the open block's
/// closes that block and opens its own; every tool call is a block of its own.
/// A model writes its calls one after another, so a call's arguments stream
/// into the block its start opened.
#[derive(Default)]
struct Blocks {
    open: Option<Open>,
    next: u32,
    saw_text: bool,
    saw_tool: bool,
}

impl Blocks {
    /// The events that stream `piece` as `kind`: the open block closed and a
    /// new one started when the kind changes, then the delta itself.
    fn delta(&mut self, kind: BlockKind, piece: &str) -> Vec<Event> {
        let mut out = Vec::new();
        if self.open != Some(Open::Prose(kind)) {
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

    /// Open a `tool_use` block for the call `id` to `name`, closing the open one.
    fn tool_start(&mut self, id: &str, name: &str) -> Vec<Event> {
        let mut out = self.close();
        self.saw_tool = true;
        self.open = Some(Open::Tool);
        self.next += 1;
        out.push(sse("content_block_start", json!({ "type": "content_block_start", "index": self.next - 1, "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} } })));
        out
    }

    /// A fragment of the open call's arguments, as the model wrote them; the
    /// client joins the fragments into the call's `input`. Nothing when no
    /// call is open.
    fn tool_args(&mut self, piece: &str) -> Vec<Event> {
        if self.open != Some(Open::Tool) || piece.is_empty() {
            return Vec::new();
        }
        vec![sse("content_block_delta", json!({ "type": "content_block_delta", "index": self.next - 1, "delta": { "type": "input_json_delta", "partial_json": piece } }))]
    }

    /// Close the message's blocks: `tail` streamed as text and `calls` as whole
    /// tool blocks (the one-shot fallback for a model that streamed neither),
    /// a text block opened if the message has neither text nor a call, and
    /// the open block closed. A call whose arguments are not a JSON object is
    /// refused, as [`from_outcome`] refuses it.
    fn finish(&mut self, tail: &str, calls: &[Value]) -> Result<Vec<Event>, ApiError> {
        let mut out = Vec::new();
        if !tail.is_empty() || (!self.saw_text && !self.saw_tool && calls.is_empty()) {
            out.extend(self.delta(BlockKind::Text, tail));
        }
        for call in calls {
            let block = tool_use_block(call)?;
            out.extend(self.tool_start(block["id"].as_str().unwrap_or_default(), block["name"].as_str().unwrap_or_default()));
            out.extend(self.tool_args(&block["input"].to_string()));
        }
        out.extend(self.close());
        Ok(out)
    }

    fn start(&mut self, kind: BlockKind) -> Event {
        let block = match kind {
            BlockKind::Thinking => thinking_block(""),
            BlockKind::Text => json!({ "type": "text", "text": "" }),
        };
        self.saw_text |= kind == BlockKind::Text;
        self.open = Some(Open::Prose(kind));
        self.next += 1;
        sse("content_block_start", json!({ "type": "content_block_start", "index": self.next - 1, "content_block": block }))
    }

    /// Close the open block; a thinking block carries its (empty) signature
    /// as the last delta, as Anthropic streams it.
    fn close(&mut self) -> Vec<Event> {
        let Some(open) = self.open.take() else { return Vec::new() };
        let index = self.next - 1;
        let mut out = Vec::new();
        if open == Open::Prose(BlockKind::Thinking) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(inv: &Invocation, key: &str) -> Value {
        serde_json::from_str(&inv.get_str(key).unwrap_or_else(|| panic!("{key} is set"))).unwrap()
    }

    /// A Claude Code turn: client tools, adaptive thinking, a mid-conversation
    /// `system` turn, a `tool_use` the assistant made and the `tool_result`
    /// answering it all reach the model in the contract's shape.
    #[test]
    fn a_tool_using_conversation_maps_onto_the_contract() {
        let body = json!({
            "model": "m", "max_tokens": 64,
            "thinking": {"type": "adaptive"},
            "tools": [{"name": "Bash", "description": "run a command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}],
            "tool_choice": {"type": "any"},
            "messages": [
                {"role": "user", "content": "list the files"},
                {"role": "system", "content": [{"type": "text", "text": "env: linux"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "use ls", "signature": ""},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a.txt"}]},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "denied", "is_error": true},
                    {"type": "text", "text": "and now?"},
                ]},
            ],
        });
        let (_, inv, _) = to_invocation(&body).unwrap();
        assert!(inv.get_bool("enable_thinking").is_none(), "adaptive thinking leaves the model's default");
        let tools = contract(&inv, "tools");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "Bash");
        assert_eq!(tools[0]["function"]["parameters"]["properties"]["command"]["type"], "string");
        assert_eq!(inv.get_str("tool_choice").as_deref(), Some("\"required\""));

        let messages = contract(&inv, "messages");
        let roles: Vec<&str> = messages.as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "system", "assistant", "tool", "tool", "user"]);
        let assistant = &messages[2];
        assert_eq!(assistant["reasoning_content"], "use ls");
        assert_eq!(assistant["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(serde_json::from_str::<Value>(assistant["tool_calls"][0]["arguments"].as_str().unwrap()).unwrap(), json!({"command": "ls"}));
        assert_eq!((messages[3]["tool_call_id"].as_str(), messages[3]["content"].as_str()), (Some("toolu_1"), Some("a.txt")));
        assert_eq!(messages[4]["content"], "Error: denied", "an error result says so");
        assert_eq!(messages[5]["content"], "and now?");
    }

    /// The written order of a tool's schema members reaches the model.
    #[test]
    fn tools_reach_the_model_as_the_client_wrote_them() {
        let raw = br#"{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"Read","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]}"#;
        let (_, inv, _) = to_invocation(&serde_json::from_slice(raw).unwrap()).unwrap();
        let tools = with_tools_as_written(raw, inv).get_str("tools").unwrap();
        assert_eq!(tools, r#"[{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]"#);
    }

    /// What the contract cannot honour is refused by name, never dropped.
    #[test]
    fn unservable_tool_requests_are_refused() {
        let base = || json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}]});
        let refused = |patch: Value| {
            let mut body = base();
            body.as_object_mut().unwrap().extend(patch.as_object().unwrap().clone());
            to_invocation(&body).err().map(|e| e.message).unwrap_or_default()
        };
        assert!(refused(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})).contains("web_search_20250305"));
        assert!(refused(json!({"tools": [{"name": "x"}]})).contains("input_schema"));
        assert!(refused(json!({"tool_choice": {"type": "auto", "disable_parallel_tool_use": true}})).contains("disable_parallel_tool_use"));
        assert!(refused(json!({"messages": [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]})).contains("tool_use_id"));
        assert_eq!(refused(json!({"tool_choice": {"type": "tool", "name": "Read"}})), "");
    }

    /// A model's call becomes a `tool_use` block with an object `input`, and
    /// arguments that are no object are an error, not a block a client cannot run.
    #[test]
    fn a_tool_call_outcome_is_a_tool_use_block() {
        let call = json!({"id": "call_0", "name": "Read", "arguments": "{\"path\":\"a\"}"});
        assert_eq!(tool_use_block(&call).unwrap(), json!({"type": "tool_use", "id": "call_0", "name": "Read", "input": {"path": "a"}}));
        assert!(tool_use_block(&json!({"id": "c", "name": "Read", "arguments": "[1]"})).is_err());
        assert_eq!(stop_reason("tool_calls"), "tool_use");
    }
}
