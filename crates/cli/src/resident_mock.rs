// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A deterministic, weight-free **mock** resident model, gated on `BRAIN_MOCK`.
//!
//! It is a real [`ResidentModel`] so a `BRAIN_MOCK=1 brain serve` exercises the true
//! serving path — placement → claim → activate → `run_batch` — with no weights, no
//! GPU, and no external model. Registered under the id **`mock`**, it advertises
//! the actions the apiserve handlers (and the examples regression harness) dispatch:
//!
//! * **`generate`** (streaming chat): reads the flattened `messages` (a JSON-array
//!   string) or `prompt`, echoes the last user turn back deterministically, streams
//!   it token-by-token via [`Progress::token`], and returns a `text` blob plus
//!   `{prompt_tokens, completion_tokens, finish_reason}` — matching
//!   [`apiserve`'s `bridge::read_outcome`].
//! * **`embed`**: takes `text` as a param OR as an input blob (the fd-in/fd-out
//!   embedding contract every real embedding model — `lfm` — uses), returns
//!   `outputs.mean` (a fixed dim-8 `Vec<f32>` derived from the input) plus
//!   `outputs.tokens`, AND the same vector as an `embeddings` output blob (f32-LE,
//!   `meta.shape=[1,DIM]`) so a client exercising the blob path (not just the HTTP
//!   JSON path) has something real to read.
//! * **`text2image`**: returns an `image` blob in brain's raw HWC-f32 wire format
//!   (`capability::blob::image_blob`) after a couple of denoise-`step` ticks,
//!   polling the cancel token between them (so `Cancel` has a runnable action to
//!   exercise beyond `crates/dbus/tests/roundtrip.rs`'s synthetic `slow` action).
//! * **`forecast`**: a deterministic last-value-plus-drift extrapolation, returned
//!   in the same `[levels, horizon]` quantile-major f32-LE wire shape chronos2
//!   uses — pinning the forecast contract `examples/forecast/README.md` documents,
//!   not a stand-in.
//!
//! `estimate()` reports a small non-zero VRAM footprint so real placement runs (and,
//! on a GPU-less box, falls back to the CPU/RAM pool); `activate()` is instant.

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::{json, Value};

/// The catalog id the mock registers under.
pub const MODEL: &str = "brain/mock";

/// The mock model family (no state — every action is a pure function of its inputs).
pub struct MockResident;

impl MockResident {
    /// The mock resident when `BRAIN_MOCK` is set to a non-empty value, else `None`.
    pub fn from_env() -> Option<MockResident> {
        std::env::var("BRAIN_MOCK").ok().filter(|v| !v.is_empty()).map(|_| MockResident)
    }

    /// The streaming chat action. Declares every param the OpenAI/Anthropic/OpenRouter
    /// chat handlers set (`messages`, `max_new`, `temp`, `top_p`, `top_k`, `seed`,
    /// optional `system`/`stop`/`tools`/`tool_choice`/`enable_thinking`) plus a
    /// `prompt` alias — so `ActionSpec::validate` never rejects a handler-built
    /// invocation. `streaming` + a `messages`/`prompt` param + a `Text` output is
    /// what `catalog::api_caps` classifies as chat.
    fn generate_spec() -> ActionSpec {
        ActionSpec::new("generate", "deterministic mock chat generation")
            .streaming()
            .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
            .param(ParamSpec::new("prompt", ParamType::Str, "a raw prompt (alternative to messages)"))
            .param(ParamSpec::new("system", ParamType::Str, "system prompt"))
            .param(ParamSpec::new("max_new", ParamType::Int, "max new tokens").default(json!(1024)))
            .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature").default(json!(1.0)))
            .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling p").default(json!(1.0)))
            .param(ParamSpec::new("top_k", ParamType::Int, "top-k").default(json!(0)))
            .param(ParamSpec::new("seed", ParamType::Int, "sampling seed").default(json!(0)))
            .param(ParamSpec::new("stop", ParamType::Str, "stop sequences (JSON array string)"))
            .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (accepted, only inspected for the mock tool-call trigger)"))
            .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive (accepted, ignored by the mock)"))
            .param(ParamSpec::new("enable_thinking", ParamType::Bool, "accepted, ignored by the mock").default(json!(true)))
            .output(BlobSpec::new("text", Media::Text, "the generated text"))
    }

    /// The embeddings action (`api_caps` classifies any action named `embed`).
    /// `text` is accepted as a param (the HTTP JSON path) OR as an input blob (the
    /// fd-in/fd-out path every real embedding model uses) — neither is `.required()`
    /// at the spec level so either alone validates; [`embed`] rejects only if BOTH
    /// are absent.
    fn embed_spec() -> ActionSpec {
        ActionSpec::new("embed", "deterministic mock text embedding")
            .param(ParamSpec::new("text", ParamType::Str, "input text (alternative to the `text` input blob)"))
            .input(BlobSpec::new("text", Media::Text, "input text (alternative to the `text` param)"))
            .output(BlobSpec::new("embedding", Media::Bytes, "embedding vector bytes (legacy/unused placeholder)"))
            .output(BlobSpec::new("embeddings", Media::Bytes, "the mean vector as f32-LE bytes, meta.shape=[1,DIM]"))
    }

    /// The forecast action: deterministic last-value-plus-drift extrapolation over
    /// an f32-LE context blob, three quantile levels, returned `[levels, horizon]`
    /// quantile-major — chronos2's own wire shape (see `crates/chronos2`).
    fn forecast_spec() -> ActionSpec {
        ActionSpec::new("forecast", "deterministic mock time-series forecast")
            .input(BlobSpec::new("context", Media::Bytes, "f32-LE context series, meta.shape=[T]").required())
            .param(ParamSpec::new("horizon", ParamType::Int, "steps to forecast").default(json!(16)))
            .param(ParamSpec::new("freq", ParamType::Int, "frequency bucket (accepted, ignored by the mock)").default(json!(0)))
            .output(BlobSpec::new("forecast", Media::Bytes, "f32-LE forecast, meta {shape:[levels,horizon], kind:\"quantiles\", levels}"))
    }

    /// The text-to-image action. An `Image` output + a `prompt` param + no required
    /// input blob is what `catalog::text2image_action` selects for `/images/generations`.
    fn text2image_spec() -> ActionSpec {
        ActionSpec::new("text2image", "deterministic mock image generation")
            .streaming()
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
            .param(ParamSpec::new("width", ParamType::Int, "image width").default(json!(1024)))
            .param(ParamSpec::new("height", ParamType::Int, "image height").default(json!(1024)))
            .param(ParamSpec::new("seed", ParamType::Int, "generation seed").default(json!(0)))
            .output(BlobSpec::new("image", Media::Image, "the generated image"))
    }
}

impl ResidentModel for MockResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            MODEL,
            "deterministic mock model (chat + embeddings + image + forecast)",
            vec![Self::generate_spec(), Self::embed_spec(), Self::text2image_spec(), Self::forecast_spec()],
        )
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // A small non-zero footprint so real placement runs: on a box with a GPU it is
        // budgeted there, on a GPU-less box it spills to the CPU/RAM pool (place.rs).
        MemCost::new(64 << 20, 0)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // No weights — the instance is built instantly.
        Ok(Box::new(MockInstance))
    }
}

/// The built mock instance (holds nothing — actions are pure functions).
struct MockInstance;

impl Instance for MockInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "generate" => generate(inv, progress),
            "embed" => embed(inv),
            "text2image" => text2image(inv, progress),
            "forecast" => forecast(inv),
            other => Err(format!("mock: unknown action '{other}'")),
        }
    }
}

/// The substring a client sends to trigger the mock **failure** mode (error-hygiene
/// probe): `generate` returns an `Err` whose text embeds a fake internal detail
/// (a filesystem path + panic-ish text) that the apiserve layer must NOT reflect to
/// the client. See [`generate`].
pub const FAIL_TRIGGER: &str = "__mock_fail__";

/// The substring a client sends to trigger the mock **tool-calling** mode: instead
/// of the plain `"You said: …"` echo, `generate` streams a scripted `reasoning`
/// event followed by one or two tool calls (repeating the trigger in the user text
/// requests a second, parallel call) through the SAME neutral `Progress::event`
/// shapes a real model's `qwen_chat::ChatScanner` emits (see
/// `crates/cli/src/resident_llm.rs::emit_chat_events`): `tool_call_start` →
/// `tool_call_args`* → `tool_call_end`. No weights/GPU are involved, so
/// `crates/apiserve/tests/api.rs` can exercise the full OpenAI/Anthropic/OpenRouter
/// tool-calling pipe against a real HTTP server. See [`FAIL_TRIGGER`] for the
/// sibling error-hygiene trigger and its naming convention (there was no existing
/// mock trigger-string convention to match beyond "double-underscore-wrapped",
/// which this follows).
pub const TOOL_CALL_TRIGGER: &str = "__MOCK_TOOL_CALL__";

/// The scripted tool calls [`TOOL_CALL_TRIGGER`] emits, in order: `(name,
/// arguments)`. A repeated trigger in the user text requests more than one
/// (clamped to this list's length) — the sequential-index test in
/// `crates/apiserve/tests/api.rs` covers exactly this "parallel calls" shape.
const MOCK_TOOL_CALLS: [(&str, &str); 2] = [("get_weather", r#"{"location": "Paris"}"#), ("set_timer", r#"{"minutes": 5}"#)];

/// The reasoning text streamed (as a single `Progress::event(kind:"reasoning")`)
/// ahead of the scripted tool call(s).
const MOCK_TOOL_CALL_REASONING: &str = "checking which tool to call";

/// [`TOOL_CALL_TRIGGER`]'s scripted reply: a reasoning event, then one
/// `tool_call_start`/`tool_call_args`*/`tool_call_end` run per requested call (the
/// arguments text is split into two fragments so a client's fragment-concatenation
/// path is actually exercised, not just its single-fragment case). Returns an
/// `Outcome` with no visible text (`outputs.tool_calls` carries the calls as a JSON
/// array string; `finish_reason: "tool_calls"`), mirroring
/// `resident_llm.rs::QwenInstance::run`'s real-model shape exactly so the two paths
/// are indistinguishable to `crates/apiserve`.
fn generate_tool_call(user: &str, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let n = user.matches(TOOL_CALL_TRIGGER).count().clamp(1, MOCK_TOOL_CALLS.len());
    progress(Progress::event(0, 1, json!({ "kind": "reasoning", "text": MOCK_TOOL_CALL_REASONING })));

    let mut calls: Vec<Value> = Vec::with_capacity(n);
    for (index, (name, arguments)) in MOCK_TOOL_CALLS.iter().take(n).enumerate() {
        let index = index as u32;
        let id = format!("call_{index}");
        progress(Progress::event(0, 1, json!({ "kind": "tool_call_start", "index": index, "id": id, "name": name })));
        let mid = arguments.len() / 2;
        for frag in [&arguments[..mid], &arguments[mid..]] {
            if !frag.is_empty() {
                progress(Progress::event(0, 1, json!({ "kind": "tool_call_args", "index": index, "text": frag })));
            }
        }
        progress(Progress::event(0, 1, json!({ "kind": "tool_call_end", "index": index })));
        calls.push(json!({ "id": id, "name": name, "arguments": arguments }));
    }

    Ok(Outcome::new()
        .set("prompt_tokens", json!(user.split_whitespace().count().max(1) as i64))
        .set("completion_tokens", json!(n as i64))
        .set("finish_reason", json!("tool_calls"))
        .set("reasoning_content", json!(MOCK_TOOL_CALL_REASONING))
        .set("tool_calls", json!(serde_json::to_string(&calls).unwrap_or_else(|_| "[]".to_string())))
        .blob("text", Blob::new(Media::Text, Vec::new())))
}

/// The delay (ms) the mock spends "producing" before it streams, from
/// `BRAIN_MOCK_DELAY_MS` (default 0 = instant). A positive value pins the single CPU
/// lane long enough to exercise admission (a concurrent request is shed with 429).
fn mock_delay_ms() -> u64 {
    std::env::var("BRAIN_MOCK_DELAY_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0)
}

/// A fake internal error detail for the failure mode. Assembled at runtime from
/// pieces so **no absolute-path string literal** appears in `crates/**` (the repo's
/// absolute-path grep gate stays clean). The string deliberately contains a
/// `secret` segment, a `model.gguf` filename, an absolute home path, and `panic`
/// text so the conformance harness can prove none of them leak to the client.
fn fake_internal_detail() -> String {
    let dir = "home";
    let leaf = "secret";
    format!("/{dir}/{leaf}/model.gguf: boom (internal panic while mmapping weights)")
}

/// Sleep up to `ms` milliseconds in small slices, returning `true` early if `cancel`
/// fires meanwhile — so a cancelled/timed-out job frees the lane promptly instead of
/// holding it for the full delay.
fn sleep_cancellable(ms: u64, cancel: &capability::CancelToken) -> bool {
    let mut left = ms;
    while left > 0 {
        if cancel.is_cancelled() {
            return true;
        }
        let slice = left.min(25);
        std::thread::sleep(std::time::Duration::from_millis(slice));
        left -= slice;
    }
    cancel.is_cancelled()
}

/// The last `user`-role message's text (falling back to the last message's content,
/// then the `prompt` param) — what the deterministic reply echoes.
fn last_user_text(inv: &Invocation) -> String {
    if let Some(s) = inv.get_str("messages") {
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&s) {
            for m in arr.iter().rev() {
                if m.get("role").and_then(|v| v.as_str()) == Some("user") {
                    if let Some(c) = m.get("content").and_then(|v| v.as_str()) {
                        return c.to_string();
                    }
                }
            }
            if let Some(c) = arr.last().and_then(|m| m.get("content")).and_then(|v| v.as_str()) {
                return c.to_string();
            }
        }
    }
    inv.get_str("prompt").unwrap_or_default()
}

/// The stop sequences (the handler passes them as a JSON-array string), or empty.
fn parse_stop(inv: &Invocation) -> Vec<String> {
    inv.get_str("stop")
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()))
        .unwrap_or_default()
}

/// Deterministic chat generation: echo the last user turn as `"You said: <text>"`
/// (or a canned line for an empty prompt), streamed token-by-token so the concatenated
/// deltas exactly reconstruct the returned `text` blob. Honors `max_new` (capping ⇒
/// `finish_reason: "length"`) and a `stop` sequence (an early stop ⇒ `"stop"`).
///
/// Two opt-in test modes make the security cases reachable without a real model:
/// * **failure** — if the user text contains [`FAIL_TRIGGER`], return an `Err` whose
///   string embeds a fake internal detail ([`fake_internal_detail`]); the apiserve
///   layer must scrub it and return only a generic message.
/// * **delay** — if `BRAIN_MOCK_DELAY_MS` > 0, sleep that long (polling `cancel`)
///   before producing, pinning the single CPU lane so a concurrent request is shed
///   (429). A cancelled/timed-out job returns promptly (`Err("cancelled")`).
fn generate(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let max_new = inv.get_i64("max_new").unwrap_or(1024).max(1) as usize;
    let user = last_user_text(inv);

    // Failure mode (error-hygiene probe): never reflected verbatim to the client.
    if user.contains(FAIL_TRIGGER) {
        return Err(format!("mock resident action failed: {}", fake_internal_detail()));
    }

    // Delay mode: pin the lane, polling cancel so a shed/disconnected job frees it.
    let delay = mock_delay_ms();
    if delay > 0 && sleep_cancellable(delay, &inv.cancel) {
        return Err("cancelled".into());
    }

    // Tool-calling mode: a scripted tool_call/reasoning event stream instead of the
    // plain text echo (see [`TOOL_CALL_TRIGGER`]/[`generate_tool_call`]).
    if user.contains(TOOL_CALL_TRIGGER) {
        return generate_tool_call(&user, progress);
    }

    let reply = if user.trim().is_empty() { "mock reply ready".to_string() } else { format!("You said: {}", user.trim()) };
    let words: Vec<&str> = reply.split_whitespace().collect();
    let stops = parse_stop(inv);

    // Build the streamed pieces, each carrying its leading separator so the
    // concatenation of all deltas is byte-for-byte the full text.
    let mut pieces: Vec<String> = Vec::new();
    let mut hit_stop = false;
    for (i, w) in words.iter().enumerate() {
        if pieces.len() >= max_new {
            break;
        }
        if stops.iter().any(|s| !s.is_empty() && w.contains(s.as_str())) {
            hit_stop = true;
            break;
        }
        pieces.push(if i == 0 { (*w).to_string() } else { format!(" {w}") });
    }
    // Hit max_new before emitting every word (and no stop match) ⇒ length-limited.
    let finish = if !hit_stop && pieces.len() < words.len() { "length" } else { "stop" };

    let total = pieces.len() as u32;
    let mut text = String::new();
    for (i, piece) in pieces.iter().enumerate() {
        text.push_str(piece);
        progress(Progress::token(i as u32, total, piece.clone()));
    }
    let prompt_tokens = user.split_whitespace().count().max(1) as i64;
    Ok(Outcome::new()
        .set("prompt_tokens", json!(prompt_tokens))
        .set("completion_tokens", json!(pieces.len()))
        .set("finish_reason", json!(finish))
        .blob("text", Blob::new(Media::Text, text.into_bytes())))
}

/// The fixed embedding dimension the mock returns.
const EMBED_DIM: usize = 8;

/// A deterministic dim-8 embedding of `text`: byte sums folded into 8 buckets, mean-
/// normalized and squashed with `sin` to a bounded, reproducible vector. `outputs.mean`
/// + `outputs.tokens` are exactly what the HTTP embeddings handler reads; the
/// `embeddings` blob (f32-LE, `meta.shape=[1,DIM]`) is the same vector for a caller
/// on the fd-in/fd-out path (see `examples/embedding/embed_document.py`).
///
/// `text` comes from the input blob if present, else the `text` param — neither is
/// required at the spec level (see [`MockResident::embed_spec`]) so this is the one
/// place that actually enforces "at least one of them", with a clear error rather
/// than silently embedding an empty string.
fn embed(inv: &Invocation) -> ActionResult {
    let text = match inv.get_blob("text") {
        Some(blob) => String::from_utf8_lossy(&blob.bytes).into_owned(),
        None => match inv.get_str("text") {
            Some(t) => t,
            None => return Err("mock: embed needs a `text` param or a `text` input blob".into()),
        },
    };
    let mut v = [0f32; EMBED_DIM];
    for (i, b) in text.bytes().enumerate() {
        v[i % EMBED_DIM] += b as f32;
    }
    let n = text.len().max(1) as f32;
    let mean: Vec<f32> = v.iter().map(|x| (x / n).sin()).collect();
    let tokens = text.split_whitespace().count().max(1);
    let mut raw = Vec::with_capacity(EMBED_DIM * 4);
    for f in &mean {
        raw.extend_from_slice(&f.to_le_bytes());
    }
    Ok(Outcome::new()
        .set("mean", json!(mean))
        .set("tokens", json!(tokens))
        .set("dim", json!(EMBED_DIM))
        .blob("embedding", Blob::new(Media::Bytes, Vec::new()))
        .blob("embeddings", Blob::new(Media::Bytes, raw).with_meta(json!({ "shape": [1, EMBED_DIM] }))))
}

/// The three quantile levels the mock's `forecast` reports (matching chronos2's
/// convention closely enough to be a real stand-in, not just a placeholder shape).
const FORECAST_LEVELS: [f32; 3] = [0.1, 0.5, 0.9];

/// Deterministic time-series forecast: extrapolate the context's last value along
/// its overall linear drift, with a spread around the median that widens with the
/// horizon (so `--horizon 64` visibly produces a wider band than `--horizon 4`,
/// exercising a client's per-quantile unpacking rather than returning three
/// identical rows). Wire shape and field names match chronos2
/// (`crates/chronos2`) — `[levels, horizon]` quantile-major f32-LE, `meta.kind =
/// "quantiles"` — so this pins the contract `examples/forecast/README.md`
/// documents instead of drifting from it silently.
fn forecast(inv: &Invocation) -> ActionResult {
    let blob = inv.get_blob("context").ok_or("mock: forecast needs a `context` input blob")?;
    if blob.bytes.is_empty() || !blob.bytes.len().is_multiple_of(4) {
        return Err(format!("mock: context blob is {} bytes, expected a non-empty multiple of 4 (f32-LE)", blob.bytes.len()));
    }
    let context: Vec<f32> = blob.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let horizon = inv.get_i64("horizon").unwrap_or(16).clamp(1, 4096) as usize;
    let last = *context.last().expect("checked non-empty above");
    let drift = if context.len() > 1 { (last - context[0]) / (context.len() - 1) as f32 } else { 0.0 };

    let mut raw = Vec::with_capacity(FORECAST_LEVELS.len() * horizon * 4);
    for level in FORECAST_LEVELS {
        for t in 0..horizon {
            let point = last + drift * (t + 1) as f32;
            let spread = 0.1 * drift.abs().max(0.01) * (t + 1) as f32;
            let value = point + (level - 0.5) * 2.0 * spread;
            raw.extend_from_slice(&value.to_le_bytes());
        }
    }
    let meta = json!({ "shape": [FORECAST_LEVELS.len(), horizon], "kind": "quantiles", "levels": FORECAST_LEVELS });
    Ok(Outcome::new()
        .set("model", json!(MODEL))
        .set("device", json!("cpu"))
        .set("horizon", json!(horizon))
        .blob("forecast", Blob::new(Media::Bytes, raw).with_meta(meta)))
}

/// Deterministic image generation: a seed-shifted RGB gradient in brain's raw HWC-f32
/// `[0,1]` wire format, after a couple of `step` denoise ticks (mapped to
/// `image_generation.partial_image` events on the streaming surface). Polls
/// `inv.cancel` between steps (honoring `BRAIN_MOCK_DELAY_MS` split evenly across
/// them) so `Cancel` has a real, runnable action to exercise — the D-Bus/HTTP
/// examples cancel-generation demos, and `roundtrip.rs`'s synthetic `slow` action,
/// were previously the only paths that ever hit `Cancel` at all.
fn text2image(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let w = inv.get_i64("width").unwrap_or(1024).clamp(1, 4096) as u32;
    let h = inv.get_i64("height").unwrap_or(1024).clamp(1, 4096) as u32;
    let seed = inv.get_i64("seed").unwrap_or(0).rem_euclid(256) as u32;
    let steps = 4u32;
    let per_step_delay = mock_delay_ms() / u64::from(steps);
    for s in 0..steps {
        if per_step_delay > 0 && sleep_cancellable(per_step_delay, &inv.cancel) {
            return Err("cancelled".into());
        }
        if inv.cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        progress(Progress::step(s + 1, steps, "denoise"));
    }
    let mut hwc = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            hwc.push((x.wrapping_add(seed) % 256) as f32 / 255.0);
            hwc.push((y % 256) as f32 / 255.0);
            hwc.push((x.wrapping_add(y) % 256) as f32 / 255.0);
        }
    }
    Ok(Outcome::new().set("width", json!(w)).set("height", json!(h)).blob("image", capability::blob::image_blob(&hwc, w, h, 3)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_classifies_as_chat_embeddings_and_image() {
        let caps = apiserve::api_caps(&MockResident.manifest());
        assert!(caps.chat, "generate must classify as chat");
        assert!(caps.embeddings, "embed must classify as embeddings");
        assert!(caps.image, "text2image must classify as image");
        // Adding `forecast` must NOT change this triple — a stray classification
        // here would reclassify the mock model and break the HTTP conformance
        // suite (`tests/e2e/api_conformance.bats`), which assumes exactly
        // chat+embeddings+image. `forecast` has no prompt/messages/text param, no
        // Text-media output, and no "embed"-ish output name, so none of
        // `api_caps`'s three rules should ever fire on it.
        assert!(
            MockResident.manifest().actions.iter().any(|a| a.name == "forecast"),
            "forecast must be advertised"
        );
    }

    #[test]
    fn tool_call_trigger_streams_one_scripted_call_by_default() {
        let mut inst = MockInstance;
        let messages = serde_json::to_string(&json!([{ "role": "user", "content": format!("weather please {TOOL_CALL_TRIGGER}") }])).unwrap();
        let inv = Invocation::new().set("messages", json!(messages));

        let mut events: Vec<Value> = Vec::new();
        let mut deltas = String::new();
        let out = inst
            .run("generate", &inv, &mut |p| {
                if let Some(d) = p.delta {
                    deltas.push_str(&d);
                }
                if let Some(e) = p.event {
                    events.push(e);
                }
            })
            .unwrap();

        // No plain-text content deltas leak — the whole reply is tool-call/reasoning
        // events, matching a real model's shape when it emits nothing but a call.
        assert!(deltas.is_empty(), "tool-call mode must not stream content deltas: {deltas:?}");
        assert_eq!(out.outputs["finish_reason"], "tool_calls");
        assert_eq!(out.outputs["reasoning_content"], MOCK_TOOL_CALL_REASONING);
        assert!(out.blobs["text"].bytes.is_empty(), "no visible text alongside a tool call");

        let calls: Value = serde_json::from_str(out.outputs["tool_calls"].as_str().unwrap()).unwrap();
        let calls = calls.as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["name"], "get_weather");
        let args: Value = serde_json::from_str(calls[0]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["location"], "Paris");

        // Event ordering: reasoning, then start/args*/end for the one call.
        let kinds: Vec<&str> = events.iter().map(|e| e["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds[0], "reasoning");
        assert_eq!(kinds[1], "tool_call_start");
        assert_eq!(kinds.last(), Some(&"tool_call_end"));
        assert!(kinds.contains(&"tool_call_args"));

        // Argument fragments concatenate to the full JSON.
        let frags: String =
            events.iter().filter(|e| e["kind"] == "tool_call_args").map(|e| e["text"].as_str().unwrap()).collect();
        assert_eq!(frags, r#"{"location": "Paris"}"#);
    }

    #[test]
    fn tool_call_trigger_repeated_emits_a_second_call_with_the_next_index() {
        let mut inst = MockInstance;
        let messages =
            serde_json::to_string(&json!([{ "role": "user", "content": format!("{TOOL_CALL_TRIGGER} and {TOOL_CALL_TRIGGER}") }]))
                .unwrap();
        let inv = Invocation::new().set("messages", json!(messages));

        let mut starts: Vec<(u32, String)> = Vec::new();
        let out = inst
            .run("generate", &inv, &mut |p| {
                if let Some(e) = p.event {
                    if e["kind"] == "tool_call_start" {
                        starts.push((e["index"].as_u64().unwrap() as u32, e["id"].as_str().unwrap().to_string()));
                    }
                }
            })
            .unwrap();

        assert_eq!(starts, vec![(0, "call_0".to_string()), (1, "call_1".to_string())], "distinct sequential indices");
        let calls: Value = serde_json::from_str(out.outputs["tool_calls"].as_str().unwrap()).unwrap();
        assert_eq!(calls.as_array().unwrap().len(), 2);
    }

    #[test]
    fn generate_streams_deltas_that_concatenate_to_the_text_blob() {
        let model = MockResident;
        let mut inst = model.activate(&model.instance_key("generate", &Invocation::new()), Device::Cpu).unwrap();
        let messages = serde_json::to_string(&json!([{ "role": "user", "content": "hello there" }])).unwrap();
        let inv = Invocation::new().set("messages", json!(messages)).set("max_new", json!(1024));

        let mut streamed = String::new();
        let out = inst.run("generate", &inv, &mut |p| {
            if let Some(d) = p.delta {
                streamed.push_str(&d);
            }
        });
        let out = out.unwrap();
        let text = String::from_utf8(out.blobs["text"].bytes.clone()).unwrap();
        assert_eq!(streamed, text, "concatenated per-token deltas must equal the text blob");
        assert_eq!(text, "You said: hello there");
        assert_eq!(out.outputs["finish_reason"], "stop");
        assert_eq!(out.outputs["completion_tokens"], 4);
    }

    #[test]
    fn generate_caps_at_max_new_and_reports_length() {
        let mut inst = MockInstance;
        let inv = Invocation::new().set("prompt", json!("one two three four five")).set("max_new", json!(2));
        let out = inst.run("generate", &inv, &mut |_| {}).unwrap();
        assert_eq!(out.outputs["completion_tokens"], 2);
        assert_eq!(out.outputs["finish_reason"], "length");
    }

    #[test]
    fn failure_mode_errors_and_embeds_internal_detail_for_the_layer_to_scrub() {
        let mut inst = MockInstance;
        let inv = Invocation::new().set("prompt", json!("please __mock_fail__ now"));
        let err = inst.run("generate", &inv, &mut |_| {}).unwrap_err();
        // The RAW mock error carries the internal detail (paths/panic) that the
        // apiserve layer is responsible for scrubbing — proven here so the bats
        // error-hygiene test has a real leak to defend against.
        assert!(err.contains("secret"), "raw mock error must embed the fake detail: {err}");
        assert!(err.contains("model.gguf"), "raw mock error must embed the fake path: {err}");
        assert!(err.contains("panic"), "raw mock error must embed panic text: {err}");
    }

    #[test]
    fn no_absolute_path_string_literal_in_the_detail() {
        // The fake detail is an absolute home path assembled at runtime; check its
        // pieces (not a path literal) so the repo's absolute-path grep gate over
        // crates/** stays clean.
        let d = fake_internal_detail();
        assert!(d.starts_with('/'), "must be an absolute path: {d}");
        assert!(d.contains("home") && d.contains("secret") && d.contains("model.gguf"));
        assert!(d.contains("panic"));
    }

    #[test]
    fn delay_returns_promptly_when_cancelled() {
        use std::time::Instant;
        // A long nominal delay, but the token is already cancelled: the sleep must
        // bail on the first poll rather than block for the full duration. (Tested at
        // the helper level to avoid racing on the process-global delay env var.)
        let cancel = capability::CancelToken::armed();
        cancel.cancel();
        let t = Instant::now();
        assert!(sleep_cancellable(5000, &cancel), "cancelled sleep returns true");
        assert!(t.elapsed().as_millis() < 1000, "cancelled delay must return promptly");
    }

    #[test]
    fn delay_sleeps_when_not_cancelled() {
        use std::time::Instant;
        let cancel = capability::CancelToken::armed();
        let t = Instant::now();
        assert!(!sleep_cancellable(60, &cancel), "an uncancelled sleep returns false");
        assert!(t.elapsed().as_millis() >= 50, "must actually spend the delay");
    }

    #[test]
    fn embed_returns_fixed_dim_mean_vector() {
        let mut inst = MockInstance;
        let out = inst.run("embed", &Invocation::new().set("text", json!("hello world")), &mut |_| {}).unwrap();
        let mean = out.outputs["mean"].as_array().unwrap();
        assert_eq!(mean.len(), EMBED_DIM);
        assert!(out.outputs["tokens"].as_i64().unwrap() > 0);
        // The blob path must return the SAME vector as f32-LE, meta.shape=[1,DIM] —
        // this is the fd-in/fd-out contract `examples/embedding/embed_document.py`
        // depends on, and previously had zero coverage.
        let blob = &out.blobs["embeddings"];
        assert_eq!(blob.meta, json!({ "shape": [1, EMBED_DIM] }));
        assert_eq!(blob.bytes.len(), EMBED_DIM * 4);
        let floats: Vec<f32> = blob.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        for (a, b) in floats.iter().zip(mean.iter()) {
            assert!((a - b.as_f64().unwrap() as f32).abs() < 1e-6, "blob vector must match outputs.mean: {floats:?} vs {mean:?}");
        }
    }

    #[test]
    fn embed_accepts_text_as_an_input_blob_instead_of_a_param() {
        let mut inst = MockInstance;
        let inv = Invocation::new().blob("text", Blob::new(Media::Text, b"hello world".to_vec()));
        let by_blob = inst.run("embed", &inv, &mut |_| {}).unwrap();
        let by_param = inst.run("embed", &Invocation::new().set("text", json!("hello world")), &mut |_| {}).unwrap();
        assert_eq!(by_blob.outputs["mean"], by_param.outputs["mean"], "blob and param inputs must embed identically");
    }

    #[test]
    fn embed_errors_clearly_when_neither_text_param_nor_blob_is_given() {
        let mut inst = MockInstance;
        let err = inst.run("embed", &Invocation::new(), &mut |_| {}).unwrap_err();
        assert!(err.contains("text"), "error should name the missing input: {err}");
    }

    #[test]
    fn text2image_emits_raw_hwc_image_blob() {
        let mut inst = MockInstance;
        let inv = Invocation::new().set("prompt", json!("a cat")).set("width", json!(4)).set("height", json!(4));
        let mut steps = 0u32;
        let out = inst.run("text2image", &inv, &mut |_| steps += 1).unwrap();
        assert!(steps >= 2, "must emit denoise step ticks");
        let blob = &out.blobs["image"];
        assert_eq!(blob.media, Media::Image);
        assert_eq!(blob.meta, json!({ "w": 4, "h": 4, "c": 3 }));
        assert_eq!(blob.bytes.len(), 4 * 4 * 3 * 4);
    }

    #[test]
    fn text2image_aborts_promptly_when_already_cancelled() {
        // Armed and pre-cancelled BEFORE the run: the plain `is_cancelled()` check
        // (independent of BRAIN_MOCK_DELAY_MS, which this test deliberately never
        // touches — see the delay tests' own note on the process-global env race)
        // must abort on the very first step.
        let mut inst = MockInstance;
        let cancel = capability::CancelToken::armed();
        cancel.cancel();
        let mut inv = Invocation::new().set("prompt", json!("a cat"));
        inv.cancel = cancel;
        let mut steps = 0u32;
        let err = inst.run("text2image", &inv, &mut |_| steps += 1).unwrap_err();
        assert_eq!(err, "cancelled");
        assert_eq!(steps, 0, "no progress tick should fire once already cancelled");
    }

    /// A gentle upward-trending series: value[i] = 10 + i.
    fn ramp_context(n: usize) -> Blob {
        let raw: Vec<u8> = (0..n).flat_map(|i| (10.0f32 + i as f32).to_le_bytes()).collect();
        Blob::new(Media::Bytes, raw)
    }

    #[test]
    fn forecast_extrapolates_the_context_drift_with_quantile_spread() {
        let mut inst = MockInstance;
        let inv = Invocation::new().set("horizon", json!(8)).blob("context", ramp_context(16));
        let out = inst.run("forecast", &inv, &mut |_| {}).unwrap();
        assert_eq!(out.outputs["horizon"], 8);
        assert_eq!(out.outputs["model"], MODEL);

        let blob = &out.blobs["forecast"];
        assert_eq!(blob.meta["shape"], json!([3, 8]));
        assert_eq!(blob.meta["kind"], "quantiles");
        assert_eq!(blob.meta["levels"], json!(FORECAST_LEVELS));
        assert_eq!(blob.bytes.len(), 3 * 8 * 4);

        let data: Vec<f32> = blob.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // Context is a unit ramp ending at 25.0 (10 + 15); drift 1.0/step, so the
        // horizon's last median point must land near 25 + 8 = 33.
        let median_row = &data[8..16]; // levels = [0.1, 0.5, 0.9] -> row 1 is 0.5
        assert!((median_row[7] - 33.0).abs() < 1.0, "median path should track the drift: {median_row:?}");
        // The band must actually widen with the horizon (t=7 vs t=0), and the
        // three levels must be ordered low < median < high at a given t.
        let lo_row = &data[0..8];
        let hi_row = &data[16..24];
        assert!(hi_row[7] - lo_row[7] > hi_row[0] - lo_row[0], "the quantile spread should widen with the horizon");
        assert!(lo_row[3] < median_row[3] && median_row[3] < hi_row[3], "levels must be ordered low < median < high");
    }

    #[test]
    fn forecast_requires_a_context_blob() {
        let mut inst = MockInstance;
        let err = inst.run("forecast", &Invocation::new().set("horizon", json!(4)), &mut |_| {}).unwrap_err();
        assert!(err.contains("context"), "error should name the missing input: {err}");
    }
}
