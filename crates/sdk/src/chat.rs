// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements in-process chat inference - tool-calling
// language models linked straight into a product - for its clients. If your
// team needs expertise in embedding and serving local language models, you
// can procure our services by sending an email to info@swedishembedded.com.

//! [`ChatPipeline`]: multi-turn chat with tool calling, over a Qwen3 model
//! (optionally with a LoRA adapter, attachable and detachable on the resident
//! base), in-process.
//!
//! ```no_run
//! use brain::{ChatMessage, ChatPipeline, ChatRequest};
//!
//! let chat = ChatPipeline::from_pretrained("unsloth/Qwen3-4B-GGUF")?;
//! let reply = chat.generate(&ChatRequest::new(vec![ChatMessage::user("Explain DMA in one sentence.")]))?;
//! println!("{}", reply.text);
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! Tools, streaming and cancellation, on a local checkpoint with an adapter
//! (every loader knob is [`crate::TextGenerationPipelineBuilder`]'s; a chat
//! pipeline is the model that builder loaded):
//!
//! ```no_run
//! use brain::chat::{ChatDelta, ToolSchema};
//! use brain::{CancelToken, ChatMessage, ChatPipeline, ChatRequest, TextGenerationPipeline};
//!
//! let chat = ChatPipeline::from(
//!     TextGenerationPipeline::builder("/models/qwen3-4b.safetensors")
//!         .tokenizer("/models/qwen3-4b/tokenizer.json")
//!         .adapter("/adapters/support.safetensors")
//!         .capacity(16384)
//!         .load()?,
//! );
//! let request = ChatRequest::new(vec![ChatMessage::system("You are terse."), ChatMessage::user("Weather in Paris?")])
//!     .tools(vec![ToolSchema::new("get_weather", "Current weather", serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}))])
//!     .thinking(false)
//!     .max_tokens(512);
//! let cancel = CancelToken::armed(); // clone it to another thread to stop the turn
//! let reply = chat.generate_stream(&request, &cancel, |delta| {
//!     if let ChatDelta::Text(text) = delta {
//!         print!("{text}");
//!     }
//! })?;
//! for call in &reply.tool_calls {
//!     println!("{}({})", call.name, call.arguments);
//! }
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! **One implementation.** A [`ChatRequest`] becomes the same invocation the
//! served OpenAI-compatible chat endpoint builds from its wire request, and
//! runs through the same functions: `qwen3::chat::parse_request` (chat
//! template, tool schemas, `tool_choice`, sampling, stop strings),
//! `qwen3::chat::SeqState` (the `<think>`/`<tool_call>` scanner, stop
//! strings, cancellation, finish reason) and
//! `qwen3::sample::generate_kv_stream_on_device` (chunked prefill, KV-cached
//! decode, the LM head applied on the device). [`ChatRequest::render_prompt`] and [`ChatRequest::parse_reply`]
//! are those same functions with no model behind them.
//!
//! **One generation at a time.** The model holds its KV cache across a
//! generation, so a [`ChatPipeline`] is `Send` but not `Sync`: move it to
//! the thread that runs generations, or share it behind a `Mutex`.

use std::path::PathBuf;

use serde_json::json;

use crate::text::{Engine, Sampling, TextGenerationPipeline};
use crate::{CancelToken, Error, Result};

/// Prompt tokens per prefill chunk when [`ChatPipeline::prefill_chunk`] is
/// never called. Cancellation is polled between chunks, so this bounds how
/// long a fired [`CancelToken`] can wait on a long prompt; each chunk costs
/// one extra device readback, which is noise at this size.
const DEFAULT_PREFILL_CHUNK: usize = 512;

/// Who a [`ChatMessage`] is from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    fn wire(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One turn of a conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    role: Role,
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<ToolCall>,
    tool_call_id: Option<String>,
}

impl ChatMessage {
    fn new(role: Role, content: impl Into<String>) -> ChatMessage {
        ChatMessage { role, content: content.into(), reasoning: None, tool_calls: Vec::new(), tool_call_id: None }
    }

    /// Instructions for the whole conversation.
    pub fn system(content: impl Into<String>) -> ChatMessage {
        ChatMessage::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> ChatMessage {
        ChatMessage::new(Role::User, content)
    }

    /// An earlier assistant turn. Add its tool calls with
    /// [`ChatMessage::with_tool_calls`] and its reasoning with
    /// [`ChatMessage::with_reasoning`].
    pub fn assistant(content: impl Into<String>) -> ChatMessage {
        ChatMessage::new(Role::Assistant, content)
    }

    /// A tool's result, answering the assistant's call `tool_call_id`.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> ChatMessage {
        ChatMessage { tool_call_id: Some(tool_call_id.into()), ..ChatMessage::new(Role::Tool, content) }
    }

    /// The tool calls an assistant turn made. Only an assistant turn makes
    /// calls: on any other turn the request is refused when it is rendered.
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> ChatMessage {
        self.tool_calls = calls;
        self
    }

    /// The reasoning an assistant turn produced before its answer. Only an
    /// assistant turn reasons: on any other turn the request is refused when
    /// it is rendered.
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> ChatMessage {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// The OpenAI message shape the chat parser reads - the one the served
    /// endpoint hands it.
    fn wire(&self) -> Result<serde_json::Value> {
        if self.role != Role::Assistant && (!self.tool_calls.is_empty() || self.reasoning.is_some()) {
            return Err(Error::Backend(format!("chat: a {} turn carries tool calls or reasoning; only an assistant turn has either", self.role.wire())));
        }
        let mut message = json!({ "role": self.role.wire(), "content": self.content });
        if let Some(reasoning) = &self.reasoning {
            message["reasoning_content"] = json!(reasoning);
        }
        if !self.tool_calls.is_empty() {
            message["tool_calls"] = self.tool_calls.iter().map(|c| json!({ "id": c.id, "name": c.name, "arguments": c.arguments })).collect();
        }
        if let Some(id) = &self.tool_call_id {
            message["tool_call_id"] = json!(id);
        }
        Ok(message)
    }
}

/// A call to a tool: made by the model in a [`ChatResponse`], or recorded
/// on an earlier assistant turn with [`ChatMessage::with_tool_calls`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// Pairs the call with its [`ChatMessage::tool`] result. A call the model
    /// makes is numbered `call_<index>` within its reply.
    pub id: String,
    pub name: String,
    /// The arguments as JSON text, exactly as the model wrote them - parse
    /// them with the tool's own schema; they are not validated here.
    pub arguments: String,
}

/// A tool the model may call.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments object.
    pub parameters: serde_json::Value,
}

impl ToolSchema {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> ToolSchema {
        ToolSchema { name: name.into(), description: description.into(), parameters }
    }

    fn wire(&self) -> serde_json::Value {
        json!({ "type": "function", "function": { "name": self.name, "description": self.description, "parameters": self.parameters } })
    }
}

/// Whether, and which, tool the model must call.
///
/// `None` withholds the schemas from the prompt entirely. `Required` and
/// `Named` are enforced after generation: a reply that does not make the
/// demanded call finishes with [`FinishReason::ToolChoiceUnmet`] rather than
/// passing for an answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides.
    #[default]
    Auto,
    /// No tool may be called.
    None,
    /// Some tool must be called.
    Required,
    /// This tool must be called; it must be one of the request's tools.
    Named(String),
}

impl ToolChoice {
    fn wire(&self) -> serde_json::Value {
        match self {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::None => json!("none"),
            ToolChoice::Required => json!("required"),
            ToolChoice::Named(name) => json!({ "type": "function", "function": { "name": name } }),
        }
    }
}

/// One chat turn to generate: the conversation so far, the tools on offer,
/// and the sampling knobs. Every knob left unset keeps the served chat
/// endpoint's own default (temperature 0.8, top-k 40, top-p 1.0, 128 new
/// tokens, thinking on, a fresh random seed per call).
#[derive(Clone, Debug)]
pub struct ChatRequest {
    messages: Vec<ChatMessage>,
    tools: Vec<ToolSchema>,
    tool_choice: ToolChoice,
    sampling: Sampling,
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest { messages, tools: Vec::new(), tool_choice: ToolChoice::Auto, sampling: Sampling::default() }
    }

    /// The tools the model may call.
    pub fn tools(mut self, tools: Vec<ToolSchema>) -> Self {
        self.tools = tools;
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// The most tokens to generate. An upper bound, and so is the context:
    /// a budget larger than what the prompt leaves of the pipeline's
    /// capacity generates until the context is full. Either way a reply cut
    /// short says [`FinishReason::Length`].
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.sampling.max_new_tokens = Some(n);
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.sampling.temperature = Some(t);
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.sampling.top_k = Some(k);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.sampling.top_p = Some(p);
        self
    }

    /// Reproducible decoding. Left unset, every call gets a real random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.sampling.seed = Some(seed);
        self
    }

    /// Add a stop string: generation ends the moment the decoded text ends
    /// with it, and the stop string itself is cut from the reply. May be
    /// called more than once.
    pub fn stop(mut self, s: impl Into<String>) -> Self {
        self.sampling.stop.push(s.into());
        self
    }

    /// Whether the model deliberates in a `<think>` block before answering.
    /// On by default; its reasoning is reported apart from the answer in
    /// [`ChatResponse::reasoning`], and it spends the same token budget.
    pub fn thinking(mut self, on: bool) -> Self {
        self.sampling.thinking = Some(on);
        self
    }

    /// The prompt this request renders to - the text the model is fed,
    /// before tokenization. Needs no model.
    pub fn render_prompt(&self) -> Result<String> {
        let inv = self.to_invocation()?;
        Ok(qwen3::chat::render_prompt(&inv).map_err(Error::Backend)?.text)
    }

    /// Parse `raw`, a completion generated for this request elsewhere (a
    /// recorded reply, a replayed transcript), exactly as a generation's own
    /// output is parsed: reasoning split out, tool calls extracted, the
    /// `tool_choice` demand checked. Needs no model.
    ///
    /// Nothing was counted, so [`ChatResponse::usage`] is all `None`, and
    /// the reason is never [`FinishReason::Length`] - whether a budget ran
    /// out is not knowable from the text.
    pub fn parse_reply(&self, raw: &str) -> Result<ChatResponse> {
        let inv = self.to_invocation()?;
        response_from_outcome(qwen3::chat::parse_reply(&inv, raw).map_err(Error::Backend)?)
    }

    /// The invocation the served chat endpoint builds from the same request
    /// on the wire.
    fn to_invocation(&self) -> Result<capability::Invocation> {
        if self.messages.is_empty() {
            return Err(Error::MissingArgument("chat: a request needs at least one message (e.g. ChatMessage::user(..))".to_string()));
        }
        let messages = self.messages.iter().map(ChatMessage::wire).collect::<Result<Vec<_>>>()?;
        let tools: Vec<serde_json::Value> = self.tools.iter().map(ToolSchema::wire).collect();
        let inv = capability::Invocation::new()
            .set("messages", json!(serde_json::Value::Array(messages).to_string()))
            .set("tools", json!(serde_json::Value::Array(tools).to_string()))
            .set("tool_choice", json!(self.tool_choice.wire().to_string()));
        self.sampling.apply(inv)
    }
}

/// Why a generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The model ended its turn.
    Stop,
    /// A [`ChatRequest::stop`] string ended it.
    StopSequence,
    /// The token budget or the context ran out; the reply is truncated.
    Length,
    /// The model called tools; see [`ChatResponse::tool_calls`].
    ToolCalls,
    /// A [`ToolChoice::Required`]/[`ToolChoice::Named`] demand was not met.
    ToolChoiceUnmet,
    /// The caller's [`CancelToken`] fired; the reply is partial.
    Cancelled,
}

impl FinishReason {
    /// The OpenAI-compatible wire name (`"stop"`, `"length"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::StopSequence => "stop_sequence",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ToolChoiceUnmet => "tool_choice_unmet",
            FinishReason::Cancelled => "cancelled",
        }
    }

    fn from_wire(s: &str) -> Result<FinishReason> {
        [FinishReason::Stop, FinishReason::StopSequence, FinishReason::Length, FinishReason::ToolCalls, FinishReason::ToolChoiceUnmet, FinishReason::Cancelled]
            .into_iter()
            .find(|r| r.as_str() == s)
            .ok_or_else(|| Error::Backend(format!("chat: unknown finish reason {s:?}")))
    }
}

/// Token accounting for one reply. A count that was not measured is `None`,
/// never `0`: a generation measures both, a [`ChatRequest::parse_reply`]
/// neither.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatUsage {
    /// Tokens the rendered prompt consumed.
    pub prompt_tokens: Option<u32>,
    /// Tokens generated, reasoning and tool calls included.
    pub completion_tokens: Option<u32>,
}

/// One generated (or parsed) assistant turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatResponse {
    /// The visible answer: no reasoning, no tool-call markup.
    pub text: String,
    /// What the model reasoned before answering; empty when it did not.
    pub reasoning: String,
    /// The tools the model called, in order.
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    pub usage: ChatUsage,
}

/// A piece of a reply, delivered while it is generated.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChatDelta {
    /// Visible answer text. All of a reply's `Text` deltas concatenated are
    /// exactly its [`ChatResponse::text`].
    Text(String),
    /// Reasoning text, as [`ChatResponse::reasoning`] accumulates it.
    Reasoning(String),
}

/// What a pipeline loaded, by content - enough to tell two deployments of
/// "the same model" apart when their weights differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelIdentity {
    pub base: WeightsIdentity,
    /// Present exactly when an adapter is applied to the base.
    pub adapter: Option<WeightsIdentity>,
}

/// One loaded weights file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightsIdentity {
    /// The id the file's own model card declares (a trained adapter's card
    /// id, a converted checkpoint's model id), when it carries a card.
    pub id: Option<String>,
    /// The file that was loaded (for a hub id, the resolved file).
    pub path: PathBuf,
    /// `sha256:<lowercase hex>` of the file's bytes - the digest the model
    /// store verifies downloads against.
    pub digest: String,
}

impl WeightsIdentity {
    /// Hashes `path`, streaming: a whole-file read, paid once at load.
    pub(crate) fn of_file(path: &str, id: Option<String>) -> Result<WeightsIdentity> {
        let digest = brain_modelstore::fetch::file_digest(std::path::Path::new(path)).map_err(|e| Error::Backend(format!("{path}: hashing the loaded weights: {e}")))?;
        Ok(WeightsIdentity { id, path: PathBuf::from(path), digest })
    }

    /// [`Self::of_file`] for a checkpoint that may be a directory: a
    /// directory's digest covers each weight file (safetensors or
    /// `pytorch_model*.bin`) by name and content, in name order, so it names
    /// exactly the bytes a load of it reads.
    pub(crate) fn of_path(path: &str, id: Option<String>) -> Result<WeightsIdentity> {
        let dir = std::path::Path::new(path);
        if !dir.is_dir() {
            return Self::of_file(path, id);
        }
        let mut files: Vec<String> = std::fs::read_dir(dir)
            .map_err(|e| Error::Backend(format!("{path}: {e}")))?
            .flatten()
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .filter(|n| n.ends_with(".safetensors") || n.ends_with(".bin") || n.ends_with(".gguf"))
            .collect();
        files.sort();
        let mut listing = String::new();
        for f in &files {
            let d = brain_modelstore::fetch::file_digest(&dir.join(f)).map_err(|e| Error::Backend(format!("{path}/{f}: hashing the loaded weights: {e}")))?;
            listing.push_str(&format!("{f} {d}\n"));
        }
        let digest = format!("sha256:{}", brain_modelstore::fetch::bytes_digest(listing.as_bytes()));
        Ok(WeightsIdentity { id, path: PathBuf::from(path), digest })
    }
}

/// Multi-turn chat with tool calling. See this module's doc.
pub struct ChatPipeline {
    engine: Engine,
    prefill_chunk: usize,
}

impl std::fmt::Debug for ChatPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatPipeline").field("capacity", &self.engine.capacity).field("prefill_chunk", &self.prefill_chunk).field("identity", &self.engine.identity).finish()
    }
}

/// The model a text pipeline loaded, as a chat pipeline - how every
/// [`crate::TextGenerationPipelineBuilder`] knob (tokenizer, adapter,
/// capacity, device, download policy) reaches chat.
impl From<TextGenerationPipeline> for ChatPipeline {
    fn from(pipe: TextGenerationPipeline) -> ChatPipeline {
        ChatPipeline { engine: pipe.into_engine(), prefill_chunk: DEFAULT_PREFILL_CHUNK }
    }
}

impl ChatPipeline {
    /// Load `weights_path` - a local checkpoint or a hub id, exactly as
    /// [`TextGenerationPipeline::from_pretrained`] takes it.
    pub fn from_pretrained(weights_path: impl AsRef<str>) -> Result<ChatPipeline> {
        Ok(TextGenerationPipeline::from_pretrained(weights_path)?.into())
    }

    /// Prefill the prompt in chunks of this many tokens (at least one;
    /// default 512). A fired [`CancelToken`] is noticed between chunks, so
    /// smaller chunks stop a long prompt sooner and cost one device readback
    /// each.
    pub fn prefill_chunk(mut self, tokens: usize) -> Self {
        self.prefill_chunk = tokens.max(1);
        self
    }

    /// The base weights, and the adapter when one is applied, by content.
    pub fn identity(&self) -> &ModelIdentity {
        &self.engine.identity
    }

    /// Apply the LoRA adapter at `path` (as written by [`crate::ChatFineTune`])
    /// to the resident base from the next turn on, replacing any adapter
    /// already applied. The adapter's low-rank correction runs beside the
    /// resident base, so switching costs reading the adapter alone (plus,
    /// once, restoring the linears of an adapter folded in at load), and
    /// [`Self::detach_adapter`] returns to exactly the base. The correction
    /// is exact at every precision.
    ///
    /// Refused - the pipeline keeps serving what it served - for a file that
    /// is not a LoRA adapter, or one whose linears do not fit this base.
    pub fn attach_adapter(&mut self, path: impl AsRef<str>) -> Result<()> {
        self.engine.attach_adapter(path.as_ref())
    }

    /// Remove the applied adapter: turns run on exactly the base again.
    /// `false` when none was applied. An adapter folded in at load
    /// ([`crate::TextGenerationPipelineBuilder::adapter`] on an fp32 base) is
    /// taken out by restoring its linears from the base checkpoint, which is
    /// the one way this can fail.
    pub fn detach_adapter(&mut self) -> Result<bool> {
        self.engine.detach_adapter()
    }

    /// Generate a reply to a prompt in which a run of positions is embedding
    /// `rows` instead of tokens: `before` (the chat template up to where the
    /// user's words go), one position per row of `rows` (row-major, the model's
    /// embedding width), then `after`. Greedy, ending at the model's
    /// end-of-turn token or after `max_new` tokens; `on_text` sees each new
    /// piece as it is decoded and `cancel` stops it between tokens. This is how
    /// speech, projected into the model's input space, is answered. It
    /// returns the reply's text only: no reasoning or tool-call parsing.
    pub fn generate_with_rows(&self, before: &str, rows: &[f32], after: &str, max_new: usize, cancel: &CancelToken, on_text: impl FnMut(&str)) -> Result<String> {
        let mut on_text = on_text;
        self.engine.generate_rows(before, rows, after, max_new, cancel, self.prefill_chunk, &mut on_text)
    }

    /// Generate the next assistant turn.
    pub fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.generate_stream(request, &CancelToken::default(), |_| {})
    }

    /// [`ChatPipeline::generate`], handing each piece of the reply to
    /// `on_delta` as it is produced and stopping when `cancel` fires.
    ///
    /// Cancellation is cooperative: it is noticed between prefill chunks and
    /// after each generated token, never inside one device call. A cancelled
    /// turn returns what was generated so far with
    /// [`FinishReason::Cancelled`] - not an error. Pass a
    /// [`CancelToken::armed`] token (a `default()` one can never fire).
    ///
    /// Tool calls arrive complete in the response, not as deltas.
    pub fn generate_stream(&self, request: &ChatRequest, cancel: &CancelToken, mut on_delta: impl FnMut(ChatDelta)) -> Result<ChatResponse> {
        let inv = request.to_invocation()?;
        let mut parsed = qwen3::chat::parse_request_as(&self.engine.tok, &self.engine.format, &inv).map_err(Error::Backend)?;
        let capacity = self.engine.capacity as usize;
        if parsed.ids.len() >= capacity {
            return Err(Error::Backend(format!(
                "chat: the rendered prompt ({} tokens) fills this pipeline's built capacity ({capacity} tokens), leaving nothing to generate -- rebuild with a larger .capacity(..) or send a shorter conversation",
                parsed.ids.len()
            )));
        }
        parsed.max_new = parsed.max_new.min(capacity - parsed.ids.len());
        let outcome = self.engine.run(&parsed, cancel, self.prefill_chunk, &mut |progress| {
            if let Some(text) = progress.delta {
                on_delta(ChatDelta::Text(text));
            } else if let Some(event) = progress.event {
                // `qwen3::chat::emit_chat_events`' event vocabulary; tool-call
                // events are not streamed (the response carries the calls).
                if event.get("kind").and_then(|k| k.as_str()) == Some("reasoning") {
                    if let Some(text) = event.get("text").and_then(|t| t.as_str()) {
                        on_delta(ChatDelta::Reasoning(text.to_string()));
                    }
                }
            }
        });
        response_from_outcome(outcome)
    }
}

/// `qwen3::chat::SeqState`'s outcome as a [`ChatResponse`].
fn response_from_outcome(o: capability::Outcome) -> Result<ChatResponse> {
    let out = &o.outputs;
    let text = out.get("text").and_then(|v| v.as_str()).ok_or_else(|| Error::Backend("chat: outcome carries no text".to_string()))?.to_string();
    let reasoning = out.get("reasoning_content").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let finish_reason = FinishReason::from_wire(out.get("finish_reason").and_then(|v| v.as_str()).ok_or_else(|| Error::Backend("chat: outcome carries no finish_reason".to_string()))?)?;
    let tool_calls = match out.get("tool_calls").and_then(|v| v.as_str()) {
        None => Vec::new(),
        Some(raw) => {
            let calls: Vec<serde_json::Value> = serde_json::from_str(raw).map_err(|e| Error::Backend(format!("chat: outcome tool_calls: {e}")))?;
            calls
                .iter()
                .map(|c| {
                    let field = |key: &str| c.get(key).and_then(|v| v.as_str()).map(str::to_string).ok_or_else(|| Error::Backend(format!("chat: an outcome tool call has no {key}")));
                    Ok(ToolCall { id: field("id")?, name: field("name")?, arguments: field("arguments")? })
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    // Absent is unmeasured; present but not a token count is a broken outcome.
    let count = |key: &str| -> Result<Option<u32>> {
        match out.get(key) {
            None => Ok(None),
            Some(v) => v.as_u64().and_then(|n| u32::try_from(n).ok()).map(Some).ok_or_else(|| Error::Backend(format!("chat: outcome {key} is not a token count: {v}"))),
        }
    };
    let usage = ChatUsage { prompt_tokens: count("prompt_tokens")?, completion_tokens: count("completion_tokens")? };
    Ok(ChatResponse { text, reasoning, tool_calls, finish_reason, usage })
}
