// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared chat-serving logic: prompt rendering (chat template, tool schemas),
//! per-token streaming through the `<think>`/`<tool_call>` scanner, stop
//! strings, and cancellation. Used by every caller that drives Qwen's chat
//! contract over a real tokenizer — `crates/cli/src/resident_llm.rs`'s
//! `QwenInstance` (the HTTP/D-Bus serving path) and [`crate::caps`]'s
//! `GenerateAction` (the `brain do qwen generate` / event-API path) — so
//! `brain do` and HTTP cannot diverge on chat rendering, tool calls, stop
//! strings, or cancellation. Pulled out of `resident_llm.rs` rather than
//! grown a second time in `caps.rs`, which is exactly the duplication this
//! module retires.
//!
//! What does NOT move here: how a caller obtains its `Qwen`/`Engine` model
//! instance (residency-managed and persistent for `resident_llm.rs`;
//! loaded-on-demand and `Hot`-cached for `caps.rs`) — those are genuinely
//! different, valid lifecycles for genuinely different use cases (concurrent
//! serving vs. a one-shot CLI invocation), and unifying them is not what
//! "retire the duplicate" means here. Only the request-shape/streaming logic
//! that must behave identically regardless of caller is shared.

use capability::{Blob, CancelToken, Invocation, Media, Outcome, Progress};
use data::qwen_chat::{self, ChatEvent, ChatMessage, ChatScanner, Role, ToolCallMsg};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use serde_json::json;

/// Parse the `messages` param into [`ChatMessage`]s, prepending a leading `system`
/// turn when one is supplied. Back-compatible with the plain `{"role","content"}`
/// shape (the whole surface before this milestone); additionally reads, when
/// present, `reasoning_content` (string), `tool_call_id` (string, meaningful on
/// `role:"tool"`), and `tool_calls` (an array of `{id,name,arguments}` — the flat
/// shape `crates/apiserve`'s message-flattening emits — or, tolerated for direct
/// callers, the nested OpenAI `{id, function:{name,arguments}}` shape; `arguments`
/// is read as-is if it's already a JSON string, else re-serialized from whatever
/// JSON value is there).
pub fn parse_chat_messages(raw: &str, system: Option<&str>) -> Result<Vec<ChatMessage>, String> {
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("qwen: messages JSON: {e}"))?;
    let arr = v.as_array().ok_or("qwen: messages must be a JSON array")?;
    let mut out = Vec::with_capacity(arr.len() + 1);
    if let Some(s) = system.filter(|s| !s.is_empty()) {
        out.push(ChatMessage::system(s));
    }
    for m in arr {
        let role_str = m.get("role").and_then(|r| r.as_str()).ok_or("qwen: message.role missing")?;
        let role = match role_str {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => return Err(format!("qwen: unknown message.role {other:?}")),
        };
        // capability::content_text (not a bare `.as_str()`): `content` is a
        // plain string for a text-only message but an OpenAI-style typed-part
        // ARRAY for a genuinely multimodal one (`crates/apiserve/src/
        // openai.rs`'s `to_invocation` now preserves that shape instead of
        // always flattening it, for omni's chat-template splice -- see
        // `omni::caps::render_chat_prompt`'s doc). Qwen3 has no image/audio
        // input of its own, but a client sending one anyway should still get
        // this message's TEXT, not silently lose it (`.as_str()` alone
        // would return `None` for an array and drop the whole turn).
        let content = capability::content_text(m.get("content"));
        let reasoning_content = m.get("reasoning_content").and_then(|c| c.as_str()).map(str::to_string);
        let tool_call_id = m.get("tool_call_id").and_then(|c| c.as_str()).map(str::to_string);
        let tool_calls = match m.get("tool_calls").and_then(|c| c.as_array()) {
            Some(calls) => calls.iter().map(parse_tool_call_msg).collect(),
            None => Vec::new(),
        };
        out.push(ChatMessage { role, content, reasoning_content, tool_calls, tool_call_id });
    }
    Ok(out)
}

/// One `tool_calls[]` element (flat `{id,name,arguments}` or nested OpenAI
/// `{id, function:{name,arguments}}`) into a [`ToolCallMsg`]. `arguments` passes
/// through unchanged if it's already a JSON string (the exact-bytes contract
/// [`ToolCallMsg::arguments`] documents); any other JSON value is re-serialized to
/// text via `serde_json` — acceptable here since this is deserialize-then-reserialize
/// bookkeeping, not the byte-exact rendering path (see `qwen_chat`'s module docs).
fn parse_tool_call_msg(c: &serde_json::Value) -> ToolCallMsg {
    let id = c.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let name = c
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| c.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()))
        .unwrap_or_default()
        .to_string();
    let args = c.get("arguments").or_else(|| c.get("function").and_then(|f| f.get("arguments")));
    let arguments = match args {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => "{}".to_string(),
    };
    ToolCallMsg { id, name, arguments }
}

/// Parse the `tools` param (a JSON array of OpenAI-shaped tool-schema objects) into
/// one raw JSON text per tool. Re-serializes each element via plain `serde_json`
/// (NOT `qwen_chat`'s Python-`json.dumps`-exact `json_py`, which is crate-private to
/// `brain-data`) — exact byte-for-byte matching only matters inside the prompt
/// renderer itself, which re-normalizes every tool's text through its own
/// `json_py::dumps` when it emits the `<tools>` block (see `qwen_chat::render`).
pub fn parse_tools(raw: Option<&str>) -> Result<Vec<String>, String> {
    let Some(raw) = raw.filter(|s| !s.is_empty()) else { return Ok(Vec::new()) };
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("qwen: tools JSON: {e}"))?;
    let arr = v.as_array().ok_or("qwen: tools must be a JSON array")?;
    arr.iter().map(|t| serde_json::to_string(t).map_err(|e| format!("qwen: tools JSON: {e}"))).collect()
}

/// The OpenAI-compatible `tool_choice` constraint, parsed from the raw JSON
/// text the invocation carries (apiserve serializes the request value; absent
/// or empty means [`ToolChoice::Auto`]).
///
/// Semantics follow vLLM/SGLang's constrained-decoding contract: `none` hides
/// the tool schemas from the prompt entirely (no tool call is possible);
/// `required` / `named` demand a (specific) tool call in the response. Upstream
/// delivers that demand with grammar-constrained decoding at sampling time -
/// which brain does not yet implement - so the forced-call variants are
/// enforced post-hoc by [`SeqState::finish`] (an unmet demand is reported as
/// `finish_reason: "tool_choice_unmet"` rather than silently returning prose).
#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    /// Model decides (the default; tool schemas rendered normally).
    Auto,
    /// Tool schemas withheld from the prompt; the model cannot call tools.
    None,
    /// The response must contain at least one tool call.
    Required,
    /// The response must contain a call to exactly this function.
    Named(String),
}

/// Parse the raw `tool_choice` invocation text (a JSON-encoded string or
/// object, exactly the shapes OpenAI's API and vLLM accept) into the typed
/// constraint. Errors carry the same message apiserve's own
/// `validate_tool_choice` uses, so a value that slips past one is rejected by
/// the other with identical wording.
pub fn parse_tool_choice(raw: Option<&str>) -> Result<ToolChoice, String> {
    const EXPECTED: &str =
        "qwen: tool_choice must be \"auto\", \"none\", \"required\", or {\"type\":\"function\",\"function\":{\"name\":...}}";
    let Some(raw) = raw.filter(|s| !s.is_empty()) else { return Ok(ToolChoice::Auto) };
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|_| EXPECTED.to_string())?;
    match v {
        serde_json::Value::String(s) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" => Ok(ToolChoice::Required),
            _ => Err(EXPECTED.to_string()),
        },
        serde_json::Value::Object(o) => {
            if o.get("type").and_then(|t| t.as_str()) != Some("function") {
                return Err(EXPECTED.to_string());
            }
            match o.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) {
                Some(name) if !name.is_empty() => Ok(ToolChoice::Named(name.to_string())),
                _ => Err(EXPECTED.to_string()),
            }
        }
        _ => Err(EXPECTED.to_string()),
    }
}

/// Extract each tool schema's `function.name` (best-effort: a schema without a
/// name contributes nothing and is never matchable by [`ToolChoice::Named`]).
fn tool_schema_names(tools: &[String]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|t| serde_json::from_str::<serde_json::Value>(t).ok())
        .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).map(|s| s.to_string()))
        .collect()
}

/// Forward a batch of [`ChatEvent`]s from the [`ChatScanner`] as [`Progress`]
/// ticks: visible [`ChatEvent::Content`] becomes [`Progress::token`] (exactly what
/// a no-tools request already streamed); reasoning/tool-call events become
/// [`Progress::event`] with a neutral, consistently-shaped JSON payload —
/// `{"kind":"reasoning","text":...}`, `{"kind":"tool_call_start","index":N,
/// "id":"call_N","name":...}`, `{"kind":"tool_call_args","index":N,"text":...}`,
/// `{"kind":"tool_call_end","index":N}`. The `id` mints as `call_<index>`,
/// matching [`ChatScanner::tool_calls`]'s own id convention (see
/// `qwen_chat::ChatScanner::close_call`) so a streamed id and the final
/// non-streaming `tool_calls` id always agree.
pub fn emit_chat_events(events: &[ChatEvent], progress: &mut dyn FnMut(Progress), step: u32, total: u32) {
    for ev in events {
        match ev {
            ChatEvent::Content(text) => {
                if !text.is_empty() {
                    progress(Progress::token(step, total, text.clone()));
                }
            }
            ChatEvent::Reasoning(text) => {
                progress(Progress::event(step, total, json!({ "kind": "reasoning", "text": text })));
            }
            ChatEvent::ToolCallStart { index, name } => {
                progress(Progress::event(
                    step,
                    total,
                    json!({ "kind": "tool_call_start", "index": index, "id": format!("call_{index}"), "name": name }),
                ));
            }
            ChatEvent::ToolCallArgs { index, fragment } => {
                progress(Progress::event(step, total, json!({ "kind": "tool_call_args", "index": index, "text": fragment })));
            }
            ChatEvent::ToolCallEnd { index } => {
                progress(Progress::event(step, total, json!({ "kind": "tool_call_end", "index": index })));
            }
        }
    }
}

/// Parse the `stop` param (a JSON array of strings) into non-empty stop strings.
pub fn parse_stops(raw: Option<&str>) -> Result<Vec<String>, String> {
    let Some(raw) = raw.filter(|s| !s.is_empty()) else { return Ok(Vec::new()) };
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("qwen: stop JSON: {e}"))?;
    let arr = v.as_array().ok_or("qwen: stop must be a JSON array")?;
    Ok(arr.iter().filter_map(|s| s.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect())
}

/// Split the freshly-decoded `full` text into the fragment safe to emit now and
/// the new "printed" prefix, holding back a trailing replacement char (an
/// incomplete multi-byte UTF-8 sequence awaiting the next token). `full` always
/// extends `printed`, so concatenating every emitted fragment — plus a final
/// flush of any held-back tail — reproduces `full` byte-for-byte.
pub fn stream_delta(printed: &str, full: &str) -> (String, String) {
    let safe = full.strip_suffix('\u{FFFD}').unwrap_or(full);
    let delta = safe.get(printed.len()..).unwrap_or("").to_string();
    (delta, safe.to_string())
}

/// If `text` ends with any stop string, the byte index where the earliest such
/// match begins (so `text[..idx]` is the truncated output); else `None`.
pub fn find_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter(|s| text.ends_with(s.as_str())).map(|s| text.len() - s.len()).min()
}

/// Read the four shared sampling params (with the spec defaults).
pub fn sampling_params(inv: &Invocation) -> (usize, f32, usize, u64) {
    let max_new = inv.get_i64("max_new").unwrap_or(128).max(0) as usize;
    let temp = inv.get_f64("temp").unwrap_or(0.8) as f32;
    let top_k = inv.get_i64("top_k").unwrap_or(40).max(0) as usize;
    let seed = inv.get_i64("seed").unwrap_or(0).max(0) as u64;
    (max_new, temp, top_k, seed)
}

/// Wrap generated text as a text-output [`Outcome`] (`text` value + `text` blob).
pub fn text_outcome(text: String) -> Outcome {
    Outcome::new().set("text", json!(text)).blob("text", Blob::new(Media::Text, text.into_bytes()))
}

/// The four sampling/prompt params every request needs, parsed once and
/// shared by every caller — prompt rendering and param parsing don't depend
/// on which model instance/engine ends up consuming them.
#[derive(Debug)]
pub struct ParsedRequest {
    pub ids: Vec<u32>,
    pub max_new: usize,
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    pub stops: Vec<String>,
    /// The parsed `tool_choice` constraint - enforced at prompt level for
    /// [`ToolChoice::None`] (schemas withheld) and post-generation for the
    /// forced-call variants (see [`ToolChoice`]'s doc).
    pub tool_choice: ToolChoice,
}

/// Build a [`ParsedRequest`] from an [`Invocation`]: `messages` (chat template,
/// tool-calling and reasoning-aware) wins; else the legacy single-`prompt`
/// (+ `chat`) path, wrapped as a one-turn user message so it goes through the
/// SAME renderer (`qwen_chat::render_for_generation`'s byte-exact Jinja
/// renderer — see that module's doc comment on why it coexists with the
/// older plain-string `QwenBpe::apply_chat_template`).
pub fn parse_request(tok: &QwenBpe, inv: &Invocation) -> Result<ParsedRequest, String> {
    let (max_new, temp, top_k, seed) = sampling_params(inv);
    let top_p = inv.get_f64("top_p").unwrap_or(1.0) as f32;
    let enable_thinking = inv.get_bool("enable_thinking").unwrap_or(true);
    // Resolve reasoning_effort: upstream Qwen3.8 defaults to "xhigh" when
    // thinking is enabled and no value is provided.
    let reasoning_effort = if enable_thinking {
        Some(inv.get_str("reasoning_effort").unwrap_or_else(|| "xhigh".into()))
    } else {
        None // thinking disabled: reasoning_effort is ignored
    };
    // Qwen3.8's `preserve_thinking` chat-template kwarg: only takes effect on
    // a Qwen38-flavor render (see data::qwen_chat); under this module's
    // Qwen3-era flavor the upstream template has no such kwarg and the value
    // is inert, but it is forwarded so a flavor switch needs no parser change.
    let preserve_thinking = inv.get_bool("preserve_thinking");
    let tool_choice = parse_tool_choice(inv.get_str("tool_choice").as_deref())?;
    let mut tools = parse_tools(inv.get_str("tools").as_deref())?;
    // Prompt-level tool_choice enforcement:
    //  - `none` withholds the tool schemas entirely (the model cannot emit a
    //    well-formed `<tool_call>` for a function it was never shown).
    //  - `named` must name a function the tools array actually offers - the
    //    same validation OpenAI/vLLM apply server-side. Without it, a typo'd
    //    name would silently degrade into an unmet post-hoc demand.
    match &tool_choice {
        ToolChoice::None => tools.clear(),
        ToolChoice::Named(name) => {
            if !tool_schema_names(&tools).iter().any(|n| n == name) {
                return Err(format!("qwen: tool_choice names function '{name}' which is not present in tools"));
            }
        }
        _ => {}
    }
    let text = match inv.get_str("messages").filter(|s| !s.is_empty()) {
        Some(raw) => {
            let msgs = parse_chat_messages(&raw, inv.get_str("system").as_deref())?;
            qwen_chat::render_for_generation(&msgs, &tools, qwen_chat::TemplateOpts {
                enable_thinking,
                reasoning_effort: reasoning_effort.clone(),
                preserve_thinking,
                ..Default::default()
            })?
        }
        None => {
            let prompt = inv.get_str("prompt").unwrap_or_default();
            if inv.get_bool("chat").unwrap_or(true) {
                let msgs = [ChatMessage::user(prompt)];
                qwen_chat::render_for_generation(&msgs, &tools, qwen_chat::TemplateOpts {
                    enable_thinking,
                    reasoning_effort: reasoning_effort.clone(),
                    ..Default::default()
                })?
            } else {
                prompt
            }
        }
    };
    let ids = tok.encode(&text);
    if ids.is_empty() {
        return Err("qwen: empty prompt".to_string());
    }
    let stops = parse_stops(inv.get_str("stop").as_deref())?;
    Ok(ParsedRequest { ids, max_new, temp, top_p, top_k, seed, stops, tool_choice })
}

/// Per-sequence streaming/finalisation state, common to every caller: the
/// running de-tokenised text, the chat-markup scanner, and why the sequence
/// stopped.
pub struct SeqState {
    stops: Vec<String>,
    cancel: CancelToken,
    printed: String,
    scan: ChatScanner,
    stop_at: Option<usize>,
    cancelled: bool,
    max_new: usize,
    prompt_tokens: usize,
    tool_choice: ToolChoice,
}

impl SeqState {
    pub fn new(req: &ParsedRequest, cancel: CancelToken) -> SeqState {
        // `thinking_open=false`: the model must emit its own literal `<think>`
        // tag to think (no prefilled-open-think prefix on this path).
        SeqState {
            stops: req.stops.clone(),
            cancel,
            printed: String::new(),
            scan: ChatScanner::new(false),
            stop_at: None,
            cancelled: false,
            max_new: req.max_new,
            prompt_tokens: req.ids.len(),
            tool_choice: req.tool_choice.clone(),
        }
    }

    /// Stream the delta between `all_tokens` (everything generated so far) and
    /// what has already been printed, honouring stop-strings and
    /// cancellation. Returns `true` once this sequence should stop.
    pub fn advance(&mut self, tok: &QwenBpe, all_tokens: &[u32], progress: &mut dyn FnMut(Progress)) -> bool {
        let full = tok.decode(all_tokens);
        let (delta, new_printed) = stream_delta(&self.printed, &full);
        self.printed = new_printed;
        if !delta.is_empty() {
            let mut evs = Vec::new();
            self.scan.push(&delta, &mut evs);
            emit_chat_events(&evs, progress, all_tokens.len() as u32, self.max_new as u32);
        }
        if let Some(idx) = find_stop(&self.printed, &self.stops) {
            self.stop_at = Some(idx);
            return true;
        }
        if self.cancel.is_cancelled() {
            self.cancelled = true;
            return true;
        }
        false
    }

    /// Final text + finish reason, and the outcome those become. A
    /// stop-string truncates the visible RAW text (existing behavior, kept
    /// as-is — a stop-string cutting mid-generation takes precedence over any
    /// in-flight tool call, which is left unclosed and unreported). Otherwise
    /// flush any held-back multi-byte tail through the scanner, then
    /// [`ChatScanner::finish`] closes out a still-open tool call (truncated
    /// by `max_new`) rather than silently dropping it.
    pub fn finish(mut self, tok: &QwenBpe, all_tokens: &[u32], progress: &mut dyn FnMut(Progress)) -> Outcome {
        // Post-hoc `required`/`named` enforcement: did the model actually
        // produce the demanded tool call? Evaluated AFTER the scanner is
        // flushed below, so a call closing only in the final held-back tail
        // still counts. A stop-string cut takes precedence (existing
        // behavior); a cancelled sequence is the caller's doing, not the
        // model failing the constraint.
        let demanded = matches!(self.tool_choice, ToolChoice::Required | ToolChoice::Named(_));
        let (text, finish) = if let Some(idx) = self.stop_at {
            (self.printed[..idx].to_string(), "stop_sequence")
        } else {
            let full = tok.decode(all_tokens);
            if full.len() > self.printed.len() {
                let tail = full[self.printed.len()..].to_string();
                let mut evs = Vec::new();
                self.scan.push(&tail, &mut evs);
                emit_chat_events(&evs, progress, all_tokens.len() as u32, self.max_new as u32);
            }
            let mut evs = Vec::new();
            self.scan.finish(&mut evs);
            emit_chat_events(&evs, progress, all_tokens.len() as u32, self.max_new as u32);
            let calls_present = !self.scan.tool_calls().is_empty();
            let met = match &self.tool_choice {
                ToolChoice::Auto | ToolChoice::None => true,
                ToolChoice::Required => calls_present,
                ToolChoice::Named(name) => self.scan.tool_calls().iter().any(|c| &c.name == name),
            };
            // Precedence: an unmet demand outranks a mismatched tool call;
            // a cancellation is the caller's doing and never reports
            // tool_choice_unmet. tool_calls > cancelled > length > stop is
            // the pre-existing order, kept.
            let reason = if self.cancelled {
                if calls_present {
                    "tool_calls"
                } else {
                    "stop"
                }
            } else if demanded && !met {
                "tool_choice_unmet"
            } else if calls_present {
                "tool_calls"
            } else if all_tokens.len() >= self.max_new {
                "length"
            } else {
                "stop" // eos
            };
            (self.scan.content().to_string(), reason)
        };
        progress(Progress::step(self.max_new as u32, self.max_new as u32, "done"));
        let mut out = text_outcome(text);
        out = out
            .set("prompt_tokens", json!(self.prompt_tokens as i64))
            .set("completion_tokens", json!(all_tokens.len() as i64))
            .set("finish_reason", json!(finish))
            .set("reasoning_content", json!(self.scan.reasoning()));
        if !self.scan.tool_calls().is_empty() {
            let calls: Vec<serde_json::Value> =
                self.scan.tool_calls().iter().map(|c| json!({ "id": c.id, "name": c.name, "arguments": c.arguments })).collect();
            out = out.set("tool_calls", json!(serde_json::to_string(&calls).unwrap_or_else(|_| "[]".to_string())));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_parse_with_and_without_system() {
        let raw = r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]"#;
        let m = parse_chat_messages(raw, None).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!((m[0].role, m[0].content.as_str()), (Role::User, "hi"));
        assert_eq!((m[1].role, m[1].content.as_str()), (Role::Assistant, "yo"));
        let m = parse_chat_messages(raw, Some("be terse")).unwrap();
        assert_eq!((m[0].role, m[0].content.as_str()), (Role::System, "be terse"));
        assert_eq!(m.len(), 3);
        // an empty system turn is dropped; malformed JSON is a clean error.
        assert_eq!(parse_chat_messages(raw, Some("")).unwrap().len(), 2);
        assert!(parse_chat_messages("not json", None).is_err());
        assert!(parse_chat_messages("{}", None).is_err());
    }

    /// Back-compat + the new fields: `reasoning_content`, `tool_calls` (both the
    /// flat `{id,name,arguments}` shape and the nested OpenAI `{id,function:{...}}`
    /// shape), and `tool_call_id` all round-trip onto the parsed [`ChatMessage`].
    #[test]
    fn messages_parse_reads_reasoning_and_tool_calls() {
        let raw = r#"[
            {"role":"user","content":"weather?"},
            {"role":"assistant","content":"","reasoning_content":"thinking",
             "tool_calls":[{"id":"call_0","name":"get_weather","arguments":"{\"c\":\"Paris\"}"}]},
            {"role":"tool","content":"22C","tool_call_id":"call_0"},
            {"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"name":"f","arguments":{"x":1}}}]}
        ]"#;
        let m = parse_chat_messages(raw, None).unwrap();
        assert_eq!(m.len(), 4);
        assert_eq!(m[1].reasoning_content.as_deref(), Some("thinking"));
        assert_eq!(m[1].tool_calls.len(), 1);
        assert_eq!(m[1].tool_calls[0].name, "get_weather");
        assert_eq!(m[1].tool_calls[0].arguments, r#"{"c":"Paris"}"#);
        assert_eq!(m[2].role, Role::Tool);
        assert_eq!(m[2].tool_call_id.as_deref(), Some("call_0"));
        // nested OpenAI shape with a JSON object `arguments` re-serializes to text.
        assert_eq!(m[3].tool_calls[0].name, "f");
        let args: serde_json::Value = serde_json::from_str(&m[3].tool_calls[0].arguments).unwrap();
        assert_eq!(args["x"], 1);
    }

    #[test]
    fn tools_parse_json_array_to_per_tool_text() {
        assert_eq!(parse_tools(None).unwrap(), Vec::<String>::new());
        assert_eq!(parse_tools(Some("")).unwrap(), Vec::<String>::new());
        let raw = r#"[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"b"}}]"#;
        let tools = parse_tools(Some(raw)).unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools[0].contains("\"a\""));
        assert!(tools[1].contains("\"b\""));
        assert!(parse_tools(Some("not json")).is_err());
        assert!(parse_tools(Some("{}")).is_err());
    }

    #[test]
    fn stops_parse_and_match() {
        assert_eq!(parse_stops(None).unwrap(), Vec::<String>::new());
        assert_eq!(parse_stops(Some(r#"["\n\n","END"]"#)).unwrap(), vec!["\n\n".to_string(), "END".to_string()]);
        assert!(parse_stops(Some("nope")).is_err());
        let stops = vec!["END".to_string(), "STOP".to_string()];
        assert_eq!(find_stop("all done END", &stops), Some(9));
        assert_eq!(find_stop("mid END dle", &stops), None); // only a trailing match
        assert_eq!(find_stop("nothing here", &stops), None);
        // earliest boundary wins when stops overlap at the tail.
        let overlap = vec!["done".to_string(), "all done".to_string()];
        assert_eq!(find_stop("all done", &overlap), Some(0));
    }

    /// The delta bookkeeping used by streaming callers: concatenated per-token
    /// deltas (plus a final flush of a held-back tail) must reproduce the
    /// full decoded text — even when a multi-byte char is split across two
    /// tokens (transient U+FFFD).
    #[test]
    fn stream_deltas_reconstruct_full_text() {
        // Simulated per-step `decode(ids_out)` outputs, incl. a split euro sign.
        let steps = ["Hi", "Hi\u{FFFD}", "Hi€", "Hi€!"];
        let mut printed = String::new();
        let mut concat = String::new();
        for full in steps {
            let (delta, np) = stream_delta(&printed, full);
            concat.push_str(&delta);
            printed = np;
        }
        let full = *steps.last().unwrap();
        if full.len() > printed.len() {
            concat.push_str(&full[printed.len()..]); // final flush
        }
        assert_eq!(concat, "Hi€!");
        // No replacement char ever escapes into an emitted fragment.
        assert!(!concat.contains('\u{FFFD}'));
    }

    // ---- tool_choice (M2.2) ----

    /// Every `tool_choice` shape OpenAI's API and vLLM accept, absent/empty ->
    /// [`ToolChoice::Auto`], and the malformed-shape error paths.
    #[test]
    fn tool_choice_parses_every_shape() {
        assert_eq!(parse_tool_choice(None), Ok(ToolChoice::Auto));
        assert_eq!(parse_tool_choice(Some("")), Ok(ToolChoice::Auto));
        assert_eq!(parse_tool_choice(Some("\"auto\"")), Ok(ToolChoice::Auto));
        assert_eq!(parse_tool_choice(Some("\"none\"")), Ok(ToolChoice::None));
        assert_eq!(parse_tool_choice(Some("\"required\"")), Ok(ToolChoice::Required));
        assert_eq!(
            parse_tool_choice(Some(r#"{"type":"function","function":{"name":"get_weather"}}"#)),
            Ok(ToolChoice::Named("get_weather".into()))
        );
        // Unwrapped text (the literal word, not its JSON encoding) is
        // rejected: apiserve always JSON-serializes the request value.
        assert!(parse_tool_choice(Some("required")).is_err());
        assert!(parse_tool_choice(Some("\"sometimes\"")).is_err());
        assert!(parse_tool_choice(Some(r#"{"type":"tool","function":{"name":"f"}}"#)).is_err());
        assert!(parse_tool_choice(Some(r#"{"type":"function","function":{}}"#)).is_err());
        assert!(parse_tool_choice(Some(r#"{"type":"function","function":{"name":""}}"#)).is_err());
        assert!(parse_tool_choice(Some(r#"{"type":"function","function":{"name":42}}"#)).is_err());
        assert!(parse_tool_choice(Some("[]")).is_err());
    }

    /// A real Qwen3 tokenizer for the chat-path tests below; skipped loudly
    /// when `QWEN3_DIR` is unset.
    fn real_tok() -> Option<QwenBpe> {
        match std::env::var("QWEN3_DIR") {
            Ok(dir) => Some(QwenBpe::from_file(&format!("{dir}/tokenizer.json")).expect("load Qwen3 tokenizer")),
            Err(_) => {
                brain_testutil::skip("QWEN3_DIR unset (needs a real Qwen3 tokenizer.json)");
                None
            }
        }
    }

    const WEATHER_TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather","description":"get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},{"type":"function","function":{"name":"get_time","description":"get time","parameters":{}}}]"#;

    /// Parse a chat request with the weather tools and the given serialized
    /// `tool_choice` invocation value.
    fn req_with_tool_choice(tok: &QwenBpe, tc: &str) -> Result<ParsedRequest, String> {
        parse_request(
            tok,
            &Invocation::new()
                .set("messages", json!(r#"[{"role": "user", "content": "weather?"}]"#))
                .set("tools", json!(WEATHER_TOOLS))
                .set("max_new", json!(64))
                .set("tool_choice", json!(tc)),
        )
    }

    /// Server-side named-function validation: `tool_choice` naming a function
    /// the tools array doesn't offer is rejected before any prompt is rendered
    /// (a typo would otherwise degrade into a guaranteed-unmet post-hoc
    /// demand).
    #[test]
    fn tool_choice_named_function_must_exist_in_tools() {
        let Some(tok) = real_tok() else { return };
        let err = req_with_tool_choice(&tok, r#"{"type":"function","function":{"name":"no_such_tool"}}"#).unwrap_err();
        assert!(err.contains("no_such_tool"), "got: {err}");
        // ... and one that DOES exist parses fine.
        let req = req_with_tool_choice(&tok, r#"{"type":"function","function":{"name":"get_time"}}"#).unwrap();
        assert_eq!(req.tool_choice, ToolChoice::Named("get_time".into()));
    }

    /// Prompt-level enforcement: `none` withholds the tool schemas entirely
    /// (the model cannot emit a well-formed call for a function it was never
    /// shown) while `auto` renders the `<tools>` block as usual.
    #[test]
    fn tool_choice_none_withholds_tool_schemas_from_the_prompt() {
        let Some(tok) = real_tok() else { return };
        let auto = req_with_tool_choice(&tok, "\"auto\"").unwrap();
        let none = req_with_tool_choice(&tok, "\"none\"").unwrap();
        let auto_text = tok.decode(&auto.ids);
        let none_text = tok.decode(&none.ids);
        assert!(auto_text.contains("<tools>"), "auto must render the tool block:\n{auto_text}");
        assert!(
            !none_text.contains("<tools>") && !none_text.contains("get_weather"),
            "none must withhold the schemas entirely:\n{none_text}"
        );
    }

    /// Post-hoc enforcement in [`SeqState::finish`]: `required`/`named` demand
    /// a tool call in the generated text; prose (or a call to a different
    /// function) is reported as `finish_reason: "tool_choice_unmet"` rather
    /// than silently returning prose - upstream delivers the same demand with
    /// grammar-constrained decoding, which brain does not implement yet.
    #[test]
    fn tool_choice_required_and_named_are_enforced_after_generation() {
        let Some(tok) = real_tok() else { return };
        let call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let prose = "The weather in Paris is lovely today.";
        let finish = |req: &ParsedRequest, gen: &str| {
            let seq = SeqState::new(req, CancelToken::default());
            seq.finish(&tok, &tok.encode(gen), &mut |_| {})
        };
        let reason = |out: &Outcome| out.outputs.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("").to_string();

        // auto: prose is a perfectly good answer.
        assert_eq!(reason(&finish(&req_with_tool_choice(&tok, "\"auto\"").unwrap(), prose)), "stop");
        // demanded, prose generated: the constraint is unmet.
        assert_eq!(reason(&finish(&req_with_tool_choice(&tok, "\"required\"").unwrap(), prose)), "tool_choice_unmet");
        assert_eq!(
            reason(&finish(&req_with_tool_choice(&tok, r#"{"type":"function","function":{"name":"get_weather"}}"#).unwrap(), prose)),
            "tool_choice_unmet"
        );
        // demanded and (specifically) met: a tool call, and the right one.
        for tc in ["\"required\"", r#"{"type":"function","function":{"name":"get_weather"}}"#] {
            let out = finish(&req_with_tool_choice(&tok, tc).unwrap(), call);
            assert_eq!(reason(&out), "tool_calls");
            assert!(out.outputs.get("tool_calls").is_some());
        }
        // named() met by a call to a DIFFERENT offered function: unmet.
        assert_eq!(
            reason(&finish(&req_with_tool_choice(&tok, r#"{"type":"function","function":{"name":"get_time"}}"#).unwrap(), call)),
            "tool_choice_unmet"
        );
    }

    /// A cancelled sequence is the caller's doing, not the model failing the
    /// constraint: cancellation still wins over an unmet `tool_choice`
    /// demand.
    #[test]
    fn cancellation_wins_over_unmet_tool_choice() {
        let Some(tok) = real_tok() else { return };
        let req = req_with_tool_choice(&tok, "\"required\"").unwrap();
        let cancel = CancelToken::armed();
        let mut seq = SeqState::new(&req, cancel.clone());
        cancel.cancel();
        assert!(seq.advance(&tok, &tok.encode("prose"), &mut |_| {}));
        let out = seq.finish(&tok, &tok.encode("prose"), &mut |_| {});
        assert_eq!(out.outputs.get("finish_reason").and_then(|v| v.as_str()), Some("stop"));
    }

    /// `preserve_thinking` threads through to the renderer, but it is the
    /// QWEN3.8 template's kwarg: under the Qwen3-era flavor this module
    /// renders, the upstream template has no such kwarg and history framing
    /// stays positional (a reasoned turn after the last user query keeps its
    /// think block, earlier turns lose it) regardless of the kwarg. If this
    /// test ever fails, parse_request started rendering a different flavor -
    /// which also flips the tool-call wire format (JSON vs the 3.8 XML form)
    /// and needs the scanner to follow.
    #[test]
    fn preserve_thinking_is_inert_under_the_qwen3_flavor() {
        let Some(tok) = real_tok() else { return };
        let msgs = r#"[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "thinking one"},
            {"role": "user", "content": "q2"}
        ]"#;
        let render = |pt: Option<bool>| {
            let mut inv = Invocation::new().set("messages", json!(msgs));
            if let Some(pt) = pt {
                inv = inv.set("preserve_thinking", json!(pt));
            }
            tok.decode(&parse_request(&tok, &inv).unwrap().ids)
        };
        for pt in [None, Some(true), Some(false)] {
            let text = render(pt);
            // Earlier turn's reasoning dropped, its content kept.
            assert!(!text.contains("thinking one"), "pt={pt:?}:\n{text}");
            assert!(text.contains("a1"), "pt={pt:?}:\n{text}");
        }
    }
}
