// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chat / tool-call fine-tuning data pipeline.
//!
//! Turns structured chat examples (system/user prompt + the assistant response
//! we want the model to learn) into brain's token-dataset layout with
//! **token-level supervision masking**: the prompt is masked (IGNORE targets)
//! and only the assistant span is trained — the correct recipe for function-call
//! and reasoning fine-tuning, where training on the (given) prompt would teach
//! the model to hallucinate user turns.
//!
//! Output (consumed unchanged by `model::fit` / `brain qwen finetune`), per
//! split ([`write_split`]):
//!   * `train.u32.bin` / `val.u32.bin` — `u32` token ids.
//!   * `train.mask.bin` / `val.mask.bin` — `u8` per-token mask (1 = trainable).
//!   * `train.ex.bin` / `val.ex.bin` — `u64` start offset of every example, so
//!     a trainer can give each example a row of its own.
//!   * `meta.json` — `{ vocab_size, token_width: 32 }`.
//!
//! Every conversation is rendered through the checkpoint's own chat template
//! and encoded message by message, so framing, the BOS the template writes
//! and the end-of-turn token after each assistant turn are the checkpoint's
//! own. Examples are delimited out of band: the stream holds nothing but the
//! conversations' tokens, whatever the vocabulary.

use std::io;
use std::path::Path;

use crate::binio;
use crate::chat_template::{ChatTemplate, TemplateError};
use crate::tokenizer::Tokenizer;

/// The in-band separator (Qwen's `<|endoftext|>`) that closed every example of
/// a dataset written before examples were indexed out of band. Only a trainer
/// reading such a dataset needs it; nothing writes it any more.
pub const LEGACY_EXAMPLE_SEPARATOR: u32 = 151643;

/// One supervised chat example: a prompt (system optional + user) and the
/// assistant response the model must learn to produce.
#[derive(Clone, Debug)]
pub struct ChatExample {
    pub system: Option<String>,
    pub user: String,
    /// The assistant turn content to train on (e.g. a `<tool_call>…</tool_call>`
    /// block, or a `<think>…</think>` + answer). The chat template closes the
    /// turn with the checkpoint's own end-of-turn token.
    pub assistant: String,
}

impl ChatExample {
    pub fn new(user: impl Into<String>, assistant: impl Into<String>) -> ChatExample {
        ChatExample { system: None, user: user.into(), assistant: assistant.into() }
    }
    pub fn with_system(system: impl Into<String>, user: impl Into<String>, assistant: impl Into<String>) -> ChatExample {
        ChatExample { system: Some(system.into()), user: user.into(), assistant: assistant.into() }
    }

    /// This example as a conversation: the optional system turn and the
    /// user turn as context, the assistant turn trained.
    pub fn to_sample(&self) -> ChatSample {
        let mut messages: Vec<ChatMessage> = self.system.iter().map(ChatMessage::system).collect();
        messages.push(ChatMessage::user(self.user.clone()));
        messages.push(ChatMessage::assistant(self.assistant.clone(), true));
        ChatSample { messages, tools: Vec::new(), rendered: None }
    }
}

/// A tool call within an assistant turn. `arguments` is the RAW JSON text (a
/// JSON-encoded string, the OpenAI wire convention) -- never re-parsed into
/// an object and reserialized, which would risk losing key order or
/// diverging from what the model actually emitted. The chat template itself
/// takes this same "already a string" path
/// (`{%- if tool_call.arguments is string %}{{- tool_call.arguments }}`),
/// matching `data::qwen_chat::ToolCallMsg`'s convention exactly.
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

/// One message in a multi-turn chat sample. Unlike [`ChatExample`] (one
/// fixed system/user/assistant boundary), a [`ChatSample`] can carry many
/// trainable assistant turns interleaved with tool results in one packed
/// conversation -- `train` is per-message, not implied by position, which is
/// what lets a whole trajectory be one sample instead of N nested-prefix
/// samples repeating the same context (see bench's `extract_packed_sample`).
///
/// `role` can be `"system"`, `"user"`, `"assistant"`, or `"tool"` --
/// rendering (via [`ChatSample::encode`]) hands every message straight to
/// the checkpoint's own chat template, which knows how to merge consecutive
/// `"tool"` turns and where think-blocks go; this struct does not
/// pre-render or pre-fold anything.
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// Only meaningful when `role == "tool"`: which tool call this result
    /// answers.
    pub tool_call_id: Option<String>,
    /// Whether this message's tokens are supervised (assistant decision
    /// turns the model should learn) or masked context (system/user/tool-
    /// result turns, and any assistant turn a producer excluded from
    /// training).
    pub train: bool,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> ChatMessage {
        ChatMessage { role: "system".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, train: false }
    }
    pub fn user(content: impl Into<String>) -> ChatMessage {
        ChatMessage { role: "user".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, train: false }
    }
    pub fn assistant(content: impl Into<String>, train: bool) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, train }
    }
    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>, train: bool) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: content.into(), tool_calls, tool_call_id: None, train }
    }
    /// A tool result. Never trainable; the template merges consecutive
    /// `"tool"`-role messages into one wrapping turn on its own.
    pub fn tool_result(content: impl Into<String>) -> ChatMessage {
        ChatMessage { role: "tool".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, train: false }
    }

    /// This message as the JSON-shaped value a chat template reads
    /// (`message.role`, `message.content`, `message.tool_calls`, …).
    fn to_template_value(&self) -> minijinja::Value {
        #[derive(serde::Serialize)]
        struct TplFunction<'a> {
            name: &'a str,
            arguments: &'a str,
        }
        #[derive(serde::Serialize)]
        struct TplToolCall<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            id: Option<&'a str>,
            r#type: &'a str,
            function: TplFunction<'a>,
        }
        #[derive(serde::Serialize)]
        struct TplMessage<'a> {
            role: &'a str,
            content: &'a str,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            tool_calls: Vec<TplToolCall<'a>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tool_call_id: Option<&'a str>,
        }
        minijinja::Value::from_serialize(TplMessage {
            role: &self.role,
            content: &self.content,
            tool_calls: self
                .tool_calls
                .iter()
                .map(|tc| TplToolCall { id: tc.id.as_deref(), r#type: "function", function: TplFunction { name: &tc.name, arguments: &tc.arguments } })
                .collect(),
            tool_call_id: self.tool_call_id.as_deref(),
        })
    }
}

/// A packed multi-turn chat sample: a full conversation, encoded and masked
/// message-by-message. See [`ChatMessage`] for why `train` lives per-message.
#[derive(Clone, Debug, Default)]
pub struct ChatSample {
    pub messages: Vec<ChatMessage>,
    /// The tool JSON-schema array, if any (drives the template's tools
    /// preamble). Order-preserving `minijinja::Value`s, never
    /// `serde_json::Value` -- see `chat_template`'s module doc.
    pub tools: Vec<minijinja::Value>,
    /// Set on a sample made by [`ChatSample::answers_as_asked`]: the text to
    /// encode, which `messages` then no longer describe.
    pub rendered: Option<Rendered>,
}

/// One answer as a model is asked for it: the prompt its template renders
/// for the conversation so far, and what the template renders for the
/// answer after it. Only the completion is supervised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendered {
    pub prompt: String,
    pub completion: String,
}

impl ChatSample {
    /// This sample for a model asked not to reason: each supervised answer
    /// that has no reasoning block of its own starts with `block` (the
    /// template's [`ChatTemplate::no_think_block`]), so the model trains on the
    /// state its no-think prompt leaves it in. Encode it with
    /// [`RenderOpts::keep_reasoning`], or a reasoning model's template drops
    /// the block. Turns that are not supervised, that call a tool or that
    /// carry their own `</think>` are left as they are.
    #[must_use]
    pub fn answering_without_thinking(&self, block: &str) -> ChatSample {
        let messages = self
            .messages
            .iter()
            .map(|m| {
                let answer = m.role == "assistant" && m.train && m.tool_calls.is_empty() && !m.content.contains(THINK_END);
                if answer {
                    ChatMessage { content: format!("{block}{}", m.content), ..m.clone() }
                } else {
                    m.clone()
                }
            })
            .collect();
        ChatSample { messages, tools: self.tools.clone(), rendered: None }
    }

    /// This sample as one example per supervised answer, each the way the
    /// model is asked for it: the prompt its template renders for the
    /// conversation so far (with reasoning off when `thinking` is `false`),
    /// then the answer as the template renders it after that prompt, so a
    /// dialogue trains each reply on the prompt it will be given.
    ///
    /// A template renders an earlier turn of a dialogue without its reasoning
    /// and an answer that is last with it, so one rendering of the whole
    /// conversation cannot show every answer the way it is asked for; and its
    /// message boundaries are refused as not prefix-stable. `None` when a
    /// supervised message is not an assistant answer (a tool call, a
    /// supervised prompt turn): such a sample is encoded as a whole.
    ///
    /// # Errors
    /// The template renders an answer that does not follow its prompt.
    pub fn answers_as_asked(&self, tmpl: &ChatTemplate, thinking: bool) -> Result<Option<Vec<ChatSample>>, TemplateError> {
        let answers = |m: &ChatMessage| m.role == "assistant" && m.tool_calls.is_empty();
        if self.messages.iter().any(|m| m.train && !answers(m)) {
            return Ok(None);
        }
        let values: Vec<minijinja::Value> = self.messages.iter().map(ChatMessage::to_template_value).collect();
        let tools = (!self.tools.is_empty()).then(|| minijinja::Value::from(self.tools.clone()));
        let none = std::collections::BTreeMap::new();
        let mut out = Vec::new();
        for at in (0..self.messages.len()).filter(|&at| self.messages[at].train) {
            let asked = minijinja::Value::from(values[..at].to_vec());
            let prompt = if thinking { tmpl.render(asked, tools.clone(), true, &none)? } else { tmpl.render_no_think(asked, tools.clone(), &none)? };
            let full = tmpl.render(minijinja::Value::from(values[..=at].to_vec()), tools.clone(), false, &none)?;
            let Some(completion) = full.strip_prefix(prompt.as_str()) else {
                return Err(TemplateError(format!(
                    "the answer at message {at} does not follow the prompt the model is asked from: the template renders the conversation so far differently with and without it"
                )));
            };
            out.push(ChatSample { messages: Vec::new(), tools: self.tools.clone(), rendered: Some(Rendered { prompt, completion: completion.to_string() }) });
        }
        Ok(Some(out))
    }

    /// [`ChatSample::encode_with`] at the default [`RenderOpts`].
    pub fn encode(&self, tok: &dyn Tokenizer, tmpl: &ChatTemplate) -> Result<(Vec<u32>, Vec<bool>), TemplateError> {
        self.encode_with(tok, tmpl, RenderOpts::default())
    }

    /// Encode to `(ids, mask)` by rendering the WHOLE conversation through
    /// `tmpl` once (via [`ChatTemplate::render_with_message_boundaries`]),
    /// then encoding each message's own byte range of that single rendered
    /// text and concatenating -- so tool-call/tool-response framing,
    /// think-block placement, the BOS the template writes and the end-of-turn
    /// token closing each assistant turn all come from the checkpoint's OWN
    /// template, not a hand-rolled approximation. A message's whole range is
    /// masked by its `train` flag, so a trained assistant turn trains its
    /// end-of-turn token too: that token is how the model learns to stop.
    /// Fails (does not silently mismask) if the template's rendering of some
    /// message is not prefix-stable -- see `render_with_message_boundaries`'s
    /// doc for exactly when that happens.
    pub fn encode_with(&self, tok: &dyn Tokenizer, tmpl: &ChatTemplate, opts: RenderOpts) -> Result<(Vec<u32>, Vec<bool>), TemplateError> {
        if let Some(Rendered { prompt, completion }) = &self.rendered {
            let (mut ids, answer) = (tok.encode(prompt), tok.encode(completion));
            let mut mask = vec![false; ids.len()];
            mask.extend(std::iter::repeat_n(true, answer.len()));
            ids.extend(answer);
            return Ok((ids, mask));
        }
        let messages: Vec<ChatMessage> = if opts.keep_reasoning {
            self.messages
                .iter()
                .map(|m| if m.role == "assistant" && m.train { ChatMessage { content: m.content.replace(THINK_END, THINK_END_SENTINEL), ..m.clone() } } else { m.clone() })
                .collect()
        } else {
            self.messages.clone()
        };
        let values: Vec<minijinja::Value> = messages.iter().map(ChatMessage::to_template_value).collect();
        let tools = (!self.tools.is_empty()).then(|| minijinja::Value::from(self.tools.clone()));
        let (full, ranges) = tmpl.render_with_message_boundaries(&values, tools)?;

        let mut ids = Vec::new();
        let mut mask = Vec::new();
        for (m, range) in self.messages.iter().zip(ranges) {
            let text = &full[range];
            let tid = if opts.keep_reasoning { tok.encode(&text.replace(THINK_END_SENTINEL, THINK_END)) } else { tok.encode(text) };
            mask.extend(std::iter::repeat_n(m.train, tid.len()));
            ids.extend(tid);
        }
        Ok((ids, mask))
    }

    /// Parse bench's `generic-messages-v2` JSONL export: one packed sample
    /// per line, `{"messages":[{"role","content","train",...}], "tools":[...]}`.
    /// Deserializes DIRECTLY into the strict, `deny_unknown_fields` wire
    /// structs below (never through an intermediate `serde_json::Value` --
    /// that would both permit `.unwrap_or(default)`-style silent coercion
    /// AND lose `tools`' JSON key order, since the workspace's `serde_json`
    /// does not build with `preserve_order`; `WireRecord.tools` is typed
    /// `minijinja::Value` specifically so serde deserializes it order-
    /// preservingly on its own, independent of that). A missing/mistyped/
    /// unexpected field is a hard parse error naming the exact line and
    /// field, not a silent default that only produces a wrong answer
    /// downstream when some particular record happens to hit the gap.
    /// `messages[].train` is REQUIRED on every message -- a record with no
    /// explicit supervision boundary is rejected rather than silently
    /// treated as all-context or all-trained (either would be a silent
    /// no-op or a silent prompt-leak into the loss). bench independently
    /// enforces the SAME schema before writing
    /// (`benchlib/datasets/formats/schema.py`, `validate_record`), so a
    /// malformed export fails the bench build, not just the brain read.
    pub fn from_jsonl(path: &Path) -> io::Result<Vec<ChatSample>> {
        let text = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record: WireRecord = serde_json::from_str(line).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("{}:{}: {e}", path.display(), lineno + 1))
            })?;
            out.push(sample_from_wire(record).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("{}:{}: {e}", path.display(), lineno + 1))
            })?);
        }
        Ok(out)
    }
}

// ===================== strict wire schema =====================
//
// bench's generic-messages-v2 export is the one true wire contract this
// parses; the shape lives in bench's benchlib/datasets/formats/messages.py
// (render_generic_messages_v2) and benchlib/datasets/segment.py
// (extract_packed_sample). Deserializing into typed, `deny_unknown_fields`
// structs (rather than indexing a raw serde_json::Value with `.unwrap_or(..)`
// fallbacks) means a missing/mistyped/unexpected field is a hard parse error
// naming exactly which field and line, not a silent default that only shows
// up as a wrong answer downstream when some particular record happens to hit
// the gap -- matching the precedent in checkpoint::st::ModelCard (required
// fields are plain, non-Option types; only genuinely optional fields are
// `Option<T>`).

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRecord {
    messages: Vec<WireMessage>,
    /// `minijinja::Value`, not `serde_json::Value` -- see `from_jsonl`'s doc.
    #[serde(default)]
    tools: Vec<minijinja::Value>,
    #[allow(dead_code)]
    #[serde(default)]
    metadata: serde_json::Value,
}

#[derive(Clone, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WireRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireMessage {
    pub(crate) role: WireRole,
    /// Required: bench's exporter always writes this key (possibly `""`),
    /// never omits it -- an absent `content` is a real shape violation, not
    /// something to paper over with a default.
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) tool_calls: Vec<WireToolCall>,
    /// Only meaningful on `WireRole::Tool`.
    #[serde(default)]
    pub(crate) tool_call_id: Option<String>,
    /// No default: a message with no explicit supervision boundary is
    /// rejected rather than silently treated as all-context or all-trained.
    pub(crate) train: bool,
}

#[derive(Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireToolCall {
    #[serde(default)]
    id: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    r#type: Option<String>,
    pub(crate) function: WireFunction,
}

#[derive(Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireFunction {
    pub(crate) name: String,
    /// OpenAI wire format requires a JSON-ENCODED STRING here, not a nested
    /// object -- bench's `_stringify_tool_call_arguments` (messages.py)
    /// writes it that way. Typed as `String` so serde itself rejects a
    /// record that regresses to the old (pre-fix) object shape, instead of
    /// silently accepting it and later double-encoding it into the rendered
    /// <tool_call> block.
    pub(crate) arguments: String,
}

/// `deny_unknown_fields` typed parsing (above) is STRUCTURAL validation only
/// -- the right fields, the right types. It cannot catch a well-typed field
/// that is still nonsense: `arguments` that types-check as a string but
/// isn't actually valid JSON, or a tool response whose `tool_call_id` names
/// no tool call anyone ever made. Both are semantically malformed input a
/// permissive parser would happily pass through to training as garbled
/// `<tool_call>` text or a tool response in the wrong place -- the "only
/// shows up when a particular field takes some particular value" failure
/// mode. See AGENTS.md "Validate everything crossing into brain from
/// outside" -- this is that rule's brain-side half for bench's export;
/// `benchlib/datasets/formats/schema.py`'s `validate_record` (structural)
/// plus `build.py`'s automatic tool-call-resolution check (semantic) are the
/// bench-side half.
fn sample_from_wire(record: WireRecord) -> Result<ChatSample, String> {
    if record.messages.is_empty() {
        return Err("\"messages\" is empty".to_string());
    }
    Ok(ChatSample { messages: messages_from_wire(record.messages)?, tools: record.tools, rendered: None })
}

/// One conversation's wire messages as [`ChatMessage`]s, with the semantic
/// checks [`sample_from_wire`] documents: tool-call arguments that are valid
/// JSON, and every tool result answering a call made earlier in the same
/// conversation. `messages[i]` in an error is the index into `wire`.
pub(crate) fn messages_from_wire(wire: Vec<WireMessage>) -> Result<Vec<ChatMessage>, String> {
    let mut messages = Vec::with_capacity(wire.len());
    let mut known_tool_call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (i, m) in wire.into_iter().enumerate() {
        let role = match m.role {
            WireRole::System => "system",
            WireRole::User => "user",
            WireRole::Assistant => "assistant",
            WireRole::Tool => "tool",
        };

        let mut tool_calls = Vec::with_capacity(m.tool_calls.len());
        for (j, tc) in m.tool_calls.into_iter().enumerate() {
            serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                .map_err(|e| format!("messages[{i}].tool_calls[{j}]: function.arguments is not valid JSON: {e}"))?;
            if let Some(id) = &tc.id {
                known_tool_call_ids.insert(id.clone());
            }
            tool_calls.push(ToolCall { id: tc.id, name: tc.function.name, arguments: tc.function.arguments });
        }

        if matches!(m.role, WireRole::Tool) {
            let id = m
                .tool_call_id
                .as_deref()
                .ok_or_else(|| format!("messages[{i}]: a \"tool\"-role message must carry \"tool_call_id\""))?;
            if !known_tool_call_ids.contains(id) {
                return Err(format!(
                    "messages[{i}]: tool_call_id {id:?} does not match any tool_calls id from an \
                     earlier assistant message in this sample -- a tool response with no matching \
                     call is malformed input, not something to silently train on"
                ));
            }
        }

        messages.push(ChatMessage { role: role.to_string(), content: m.content, tool_calls, tool_call_id: m.tool_call_id, train: m.train });
    }
    Ok(messages)
}

/// How a sample renders for training.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderOpts {
    /// Train the reasoning of trained assistant turns. A reasoning model's
    /// template drops `<think>...</think>` from the assistant turns it
    /// renders as history - DeepSeek-R1's keeps only what follows the last
    /// `</think>` - which is right at inference and wrong for reasoning SFT,
    /// where the reasoning is what is being taught. With this set, a trained
    /// assistant turn's `</think>` is hidden from the template and restored
    /// in its rendered text, whatever the template's own rule.
    pub keep_reasoning: bool,
}

const THINK_END: &str = "</think>";
/// Stands in for `</think>` while the template renders: private-use code
/// points no tokenizer maps and no template matches.
const THINK_END_SENTINEL: &str = "\u{F8FF}brain-think-end\u{F8FF}";

/// A split's token stream, supervision mask and the start offset of each
/// example in it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EncodedSplit {
    pub ids: Vec<u32>,
    pub mask: Vec<bool>,
    pub starts: Vec<usize>,
}

impl EncodedSplit {
    /// Append one example.
    pub fn push(&mut self, ids: &[u32], mask: &[bool]) {
        assert_eq!(ids.len(), mask.len(), "EncodedSplit::push: ids and mask differ in length");
        if ids.is_empty() {
            return;
        }
        self.starts.push(self.ids.len());
        self.ids.extend_from_slice(ids);
        self.mask.extend_from_slice(mask);
    }

    /// The longest example, in tokens.
    pub fn longest_example(&self) -> usize {
        self.starts.iter().enumerate().map(|(i, &a)| self.starts.get(i + 1).copied().unwrap_or(self.ids.len()) - a).max().unwrap_or(0)
    }
}

/// Encode a set of packed samples into one split.
pub fn encode_sample_split(samples: &[ChatSample], tok: &dyn Tokenizer, tmpl: &ChatTemplate, opts: RenderOpts) -> Result<EncodedSplit, TemplateError> {
    let mut out = EncodedSplit::default();
    for s in samples {
        let (i, m) = s.encode_with(tok, tmpl, opts)?;
        out.push(&i, &m);
    }
    Ok(out)
}

/// Write one split - `<split>.u32.bin`, `<split>.mask.bin` and the example
/// index `<split>.ex.bin` - into `dir`: the one layout every chat dataset
/// writer produces.
pub fn write_split(dir: &Path, split: &str, data: &EncodedSplit) -> io::Result<()> {
    binio::write_u32_bin(&dir.join(format!("{split}.u32.bin")), &data.ids)?;
    binio::write_mask_bin(&dir.join(format!("{split}.mask.bin")), &data.mask)?;
    let starts: Vec<u64> = data.starts.iter().map(|&s| s as u64).collect();
    binio::write_u64_bin(&dir.join(format!("{split}.ex.bin")), &starts)
}

/// Write a train/val split of packed [`ChatSample`]s to `dir` (see the
/// module doc for the layout). `tmpl` is the checkpoint's OWN chat template
/// (compiled once by the caller; see `chat_template`).
pub fn prepare_chat_samples(
    train: &[ChatSample],
    val: &[ChatSample],
    tok: &dyn Tokenizer,
    tmpl: &ChatTemplate,
    opts: RenderOpts,
    vocab: usize,
    dir: &Path,
) -> Result<Prepared, PrepareError> {
    std::fs::create_dir_all(dir).map_err(PrepareError::Io)?;
    let train = encode_sample_split(train, tok, tmpl, opts).map_err(PrepareError::Template)?;
    let val = encode_sample_split(val, tok, tmpl, opts).map_err(PrepareError::Template)?;
    write_split(dir, "train", &train).map_err(PrepareError::Io)?;
    write_split(dir, "val", &val).map_err(PrepareError::Io)?;
    std::fs::write(dir.join("meta.json"), binio::Meta::vocab_only(vocab)).map_err(PrepareError::Io)?;
    Ok(Prepared { longest_example: train.longest_example().max(val.longest_example()) })
}

/// What [`prepare_chat_samples`] measured while writing, so a caller can size
/// a training row to the data instead of guessing. A row shorter than
/// `longest_example` cannot hold its example, and one much longer than it is
/// mostly padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prepared {
    pub longest_example: usize,
}

#[derive(Debug)]
pub enum PrepareError {
    Io(io::Error),
    Template(TemplateError),
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrepareError::Io(e) => write!(f, "{e}"),
            PrepareError::Template(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PrepareError {}

/// Write a train/val split of single-turn [`ChatExample`]s to `dir`, through
/// the checkpoint's own chat template - [`prepare_chat_samples`] over
/// [`ChatExample::to_sample`]. `vocab` is the MODEL's vocabulary (from its
/// `config.json`, matching its `lm_head`), not the tokenizer's derived size.
pub fn prepare_chat(train: &[ChatExample], val: &[ChatExample], tok: &dyn Tokenizer, tmpl: &ChatTemplate, vocab: usize, dir: &Path) -> Result<Prepared, PrepareError> {
    let train: Vec<ChatSample> = train.iter().map(ChatExample::to_sample).collect();
    let val: Vec<ChatSample> = val.iter().map(ChatExample::to_sample).collect();
    prepare_chat_samples(&train, &val, tok, tmpl, RenderOpts::default(), vocab, dir)
}

#[cfg(test)]
mod tests {
    use super::*;


    fn answering(content: &str, train: bool) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, train }
    }

    /// A model asked not to reason is trained on what it is asked from: each
    /// supervised answer follows the no-think block. An answer not supervised,
    /// one that already closes a reasoning block, and a turn that calls a tool
    /// are not touched.
    #[test]
    fn a_supervised_answer_follows_the_no_think_block() {
        let call = ChatMessage { tool_calls: vec![ToolCall { id: Some("c".into()), name: "f".into(), arguments: "{}".into() }], ..answering("", true) };
        let sample = ChatSample {
            messages: vec![ChatMessage::user("q"), answering("first", true), answering("context", false), answering("<think>x</think>done", true), call.clone()],
            tools: Vec::new(),
            rendered: None,
        };
        let out = sample.answering_without_thinking("<think>\n\n</think>\n\n");
        let said: Vec<&str> = out.messages.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(said, ["q", "<think>\n\n</think>\n\nfirst", "context", "<think>x</think>done", ""]);
        assert_eq!(out.messages[4].tool_calls.len(), 1, "the call is kept");
    }

    /// A dialogue of several supervised answers trains for no-think mode under
    /// Qwen3's own template: it is not refused, and each answer is trained as
    /// the model is asked for it - after the prompt a no-think request renders
    /// for the conversation so far, whose earlier answers carry no reasoning
    /// block - with only the answer supervised.
    #[test]
    fn each_answer_of_a_dialogue_is_trained_as_the_model_is_asked_for_it() {
        let tmpl = ChatTemplate::compile(include_str!("../testdata/qwen3_chat_template.jinja")).unwrap();
        let turn = |role: &str, text: &str, train: bool| ChatMessage { role: role.into(), content: text.into(), tool_calls: Vec::new(), tool_call_id: None, train };
        let dialogue = ChatSample {
            messages: vec![turn("system", "s", false), turn("user", "q1", false), turn("assistant", "a1", true), turn("user", "q2", false), turn("assistant", "a2", true)],
            ..ChatSample::default()
        };
        let chars: String = (32u8..127).map(char::from).chain(['\n']).collect();
        let tok = crate::tokenizer::CharTokenizer::from_corpus(&chars);
        let samples = dialogue.answers_as_asked(&tmpl, false).unwrap().expect("plain answers");
        assert_eq!(samples.len(), 2);
        let supervised = |sample: &ChatSample| {
            let (ids, mask) = sample.encode(&tok, &tmpl).expect("encodes");
            let said: String = ids.iter().zip(&mask).filter(|(_, m)| **m).map(|(i, _)| tok.decode(&[*i])).collect();
            let shown: String = ids.iter().zip(&mask).filter(|(_, m)| !**m).map(|(i, _)| tok.decode(&[*i])).collect();
            (shown, said)
        };
        let (shown, said) = supervised(&samples[0]);
        assert!(shown.ends_with("<think>\n\n</think>\n\n"), "{shown:?}");
        assert_eq!(said, "a1<|im_end|>\n");
        let (shown, said) = supervised(&samples[1]);
        assert!(shown.contains("a1") && !shown.contains("<think>\n\n</think>\n\na1"), "an earlier answer has no reasoning block: {shown:?}");
        assert!(shown.ends_with("<think>\n\n</think>\n\n"), "{shown:?}");
        assert_eq!(said, "a2<|im_end|>\n");
    }

    #[test]
    fn from_jsonl_parses_a_packed_multi_turn_sample() {
        let samples = ChatSample::from_jsonl(std::path::Path::new("testdata/chat_sample_packed.jsonl")).expect("parses");
        assert_eq!(samples.len(), 1);
        let s = &samples[0];
        assert_eq!(s.messages.len(), 5);
        assert_eq!(s.messages[0].role, "system");
        assert!(!s.messages[0].train);
        assert_eq!(s.messages[2].role, "assistant");
        assert!(s.messages[2].train);
        assert_eq!(s.messages[2].tool_calls.len(), 1);
        assert_eq!(s.messages[2].tool_calls[0].name, "get_weather");
        // role: "tool" passes through as-is -- the chat template merges
        // consecutive tool-role turns and wraps them, not this parser.
        assert_eq!(s.messages[3].role, "tool");
        assert_eq!(s.messages[3].content, "18C, sunny");
        assert!(!s.messages[3].train);
        assert!(s.messages[4].train);
    }

    #[test]
    fn from_jsonl_rejects_a_message_with_no_train_field() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_train.jsonl");
        std::fs::write(&path, r#"{"messages":[{"role":"user","content":"hi","train":false},{"role":"assistant","content":"hey"}]}"#).unwrap();
        let err = ChatSample::from_jsonl(&path).unwrap_err();
        assert!(err.to_string().contains("train"), "error should mention the missing train field: {err}");
    }

    #[test]
    fn from_jsonl_rejects_empty_messages() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.jsonl");
        std::fs::write(&path, r#"{"messages":[]}"#).unwrap();
        assert!(ChatSample::from_jsonl(&path).is_err());
    }

    /// This is the exact bug class the strict wire schema exists to catch:
    /// `arguments` as a raw nested object (the shape before bench's export
    /// fix) instead of a JSON-encoded string. A permissive `Value`-indexing
    /// parser accepts this silently and only breaks later, when
    /// `rendered_content` double-encodes it into garbled `<tool_call>` text
    /// -- a problem that "only appears when a data field is set to some
    /// particular value" instead of failing at parse time.
    #[test]
    fn from_jsonl_rejects_tool_call_arguments_as_a_raw_object() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-badargs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_args.jsonl");
        std::fs::write(
            &path,
            r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"location":"Paris"}}}],"train":true}]}"#,
        )
        .unwrap();
        let err = ChatSample::from_jsonl(&path).unwrap_err();
        assert!(
            err.to_string().contains("expected a string"),
            "error should flag the type mismatch (object where a JSON-encoded string was required), got: {err}"
        );
    }

    /// SEMANTIC malformation, not structural: `arguments` type-checks as a
    /// string (satisfying `deny_unknown_fields` + the required-String type),
    /// but the string itself is not valid JSON syntax. A parser that only
    /// checks shape, never content, would pass this straight through as
    /// literal garbled text in a rendered `<tool_call>` block.
    #[test]
    fn from_jsonl_rejects_syntactically_invalid_json_in_tool_call_arguments() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-badjson-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_json.jsonl");
        std::fs::write(
            &path,
            r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{not valid json"}}],"train":true}]}"#,
        )
        .unwrap();
        let err = ChatSample::from_jsonl(&path).unwrap_err();
        assert!(err.to_string().contains("not valid JSON"), "got: {err}");
    }

    /// SEMANTIC malformation: every field individually type-checks (role is
    /// a real role, content is a string, train is a bool), but the
    /// CONVERSATION is nonsense -- a tool response that answers no tool call
    /// anyone ever made. Exactly "tool responses in the right places" from
    /// AGENTS.md's validate-at-the-boundary rule.
    #[test]
    fn from_jsonl_rejects_a_tool_message_with_no_tool_call_id() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-notoolid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_tool_id.jsonl");
        std::fs::write(&path, r#"{"messages":[{"role":"tool","content":"18C, sunny","train":false}]}"#).unwrap();
        let err = ChatSample::from_jsonl(&path).unwrap_err();
        assert!(err.to_string().contains("tool_call_id"), "got: {err}");
    }

    #[test]
    fn from_jsonl_rejects_a_tool_message_whose_tool_call_id_matches_no_prior_call() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-orphantoolid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("orphan_tool_id.jsonl");
        std::fs::write(
            &path,
            r#"{"messages":[{"role":"user","content":"hi","train":false},{"role":"tool","content":"18C, sunny","tool_call_id":"does-not-exist","train":false}]}"#,
        )
        .unwrap();
        let err = ChatSample::from_jsonl(&path).unwrap_err();
        assert!(err.to_string().contains("does not match any tool_calls id"), "got: {err}");
    }

    #[test]
    fn from_jsonl_rejects_a_non_bool_train_value() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-badtrain-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_train.jsonl");
        std::fs::write(&path, r#"{"messages":[{"role":"user","content":"hi","train":"yes"}]}"#).unwrap();
        assert!(ChatSample::from_jsonl(&path).is_err());
    }

    #[test]
    fn from_jsonl_rejects_an_unrecognized_role() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-badrole-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_role.jsonl");
        std::fs::write(&path, r#"{"messages":[{"role":"developer","content":"hi","train":false}]}"#).unwrap();
        assert!(ChatSample::from_jsonl(&path).is_err());
    }

    #[test]
    fn from_jsonl_rejects_a_missing_content_field() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-nocontent-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_content.jsonl");
        std::fs::write(&path, r#"{"messages":[{"role":"user","train":false}]}"#).unwrap();
        assert!(ChatSample::from_jsonl(&path).is_err());
    }

    #[test]
    fn from_jsonl_rejects_an_unexpected_top_level_field() {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-extrafield-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("extra_field.jsonl");
        std::fs::write(
            &path,
            r#"{"messages":[{"role":"user","content":"hi","train":false}],"unexpected_new_field":123}"#,
        )
        .unwrap();
        assert!(ChatSample::from_jsonl(&path).is_err());
    }

    #[test]
    fn to_template_value_omits_empty_tool_calls_and_tool_call_id() {
        // Fields the template checks with `is defined`/truthiness must be
        // genuinely ABSENT when empty, not present-but-empty -- an empty
        // `tool_calls: []` vs. a missing key can take different branches in
        // a real template (`{%- if message.tool_calls %}`).
        let m = ChatMessage::user("hi");
        let v = m.to_template_value();
        assert_eq!(v.get_attr("tool_calls").unwrap().kind(), minijinja::value::ValueKind::Undefined);
        assert_eq!(v.get_attr("tool_call_id").unwrap().kind(), minijinja::value::ValueKind::Undefined);
    }
}
