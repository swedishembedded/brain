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
    /// Only meaningful when `role == "assistant"`: the reasoning the model
    /// produced before this turn. It reaches the template as
    /// `reasoning_content`, the field a reasoning model's template renders as
    /// its think block (the same field inference sends back), so a trained
    /// turn shows the reasoning where the model is asked to produce it. A
    /// turn with none keeps the shape it always had.
    pub reasoning_content: Option<String>,
    /// Whether this message's tokens are supervised (assistant decision
    /// turns the model should learn) or masked context (system/user/tool-
    /// result turns, and any assistant turn a producer excluded from
    /// training).
    pub train: bool,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> ChatMessage {
        ChatMessage { role: "system".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, reasoning_content: None, train: false }
    }
    pub fn user(content: impl Into<String>) -> ChatMessage {
        ChatMessage { role: "user".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, reasoning_content: None, train: false }
    }
    pub fn assistant(content: impl Into<String>, train: bool) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, reasoning_content: None, train }
    }
    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>, train: bool) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: content.into(), tool_calls, tool_call_id: None, reasoning_content: None, train }
    }
    /// A tool result. Never trainable; the template merges consecutive
    /// `"tool"`-role messages into one wrapping turn on its own.
    pub fn tool_result(content: impl Into<String>) -> ChatMessage {
        ChatMessage { role: "tool".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, reasoning_content: None, train: false }
    }

    /// This message as the JSON-shaped value a chat template reads
    /// (`message.role`, `message.content`, `message.tool_calls`, …). A call's
    /// arguments are the raw JSON text, or, for a template that iterates them
    /// ([`ChatTemplate::reads_argument_objects`]), that text parsed with its key
    /// order kept; text that does not parse stays text.
    fn to_template_value(&self, argument_objects: bool) -> minijinja::Value {
        #[derive(serde::Serialize)]
        struct TplFunction<'a> {
            name: &'a str,
            arguments: minijinja::Value,
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
            #[serde(skip_serializing_if = "Option::is_none")]
            reasoning_content: Option<&'a str>,
        }
        minijinja::Value::from_serialize(TplMessage {
            role: &self.role,
            content: &self.content,
            tool_calls: self
                .tool_calls
                .iter()
                .map(|tc| TplToolCall { id: tc.id.as_deref(), r#type: "function", function: TplFunction { name: &tc.name, arguments: template_arguments(&tc.arguments, argument_objects) } })
                .collect(),
            tool_call_id: self.tool_call_id.as_deref(),
            reasoning_content: self.reasoning_content.as_deref(),
        })
    }
}

/// A call's arguments as a template reads them: the parsed object when the
/// template iterates them and the text is JSON, else the text itself.
fn template_arguments(raw: &str, as_object: bool) -> minijinja::Value {
    if as_object {
        if let Ok(parsed) = serde_json::from_str::<minijinja::Value>(raw) {
            return parsed;
        }
    }
    minijinja::Value::from(raw)
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

/// Text already rendered through a chat template, in order, each stretch
/// either context or supervised. A sample made by
/// [`ChatSample::answers_as_asked`] holds one answer as a model is asked for
/// it (the prompt its template renders for the conversation so far, then the
/// answer after it); one made by [`ChatSample::answers_packed`] holds several.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendered {
    pub segments: Vec<RenderedSegment>,
}

/// One stretch of a [`Rendered`] text and whether it is supervised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedSegment {
    pub text: String,
    pub supervised: bool,
}

impl Rendered {
    fn answer(prompt: &str, completion: &str) -> Self {
        Self { segments: vec![RenderedSegment { text: prompt.to_string(), supervised: false }, RenderedSegment { text: completion.to_string(), supervised: true }] }
    }
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

    /// Whether a supervised assistant turn calls a tool. Such a turn is not
    /// prefix-stable under a template that renders only the last assistant
    /// turn of a conversation differently, so it can only be trained through
    /// [`ChatSample::answers_as_asked`].
    #[must_use]
    pub fn supervises_tool_calls(&self) -> bool {
        self.messages.iter().any(|m| m.train && !m.tool_calls.is_empty())
    }

    /// This sample as one example per supervised answer, each the way the
    /// model is asked for it: the prompt its template renders for the
    /// conversation so far (with reasoning off when `thinking` is `false`),
    /// then the answer as the template renders it after that prompt, so a
    /// dialogue trains each reply on the prompt it will be given. An answer
    /// may call tools: a turn that does is the same kind of example, and a
    /// conversation of several calls and their results trains turn by turn.
    ///
    /// A template renders an earlier turn of a dialogue without its reasoning
    /// and an answer that is last with it, so one rendering of the whole
    /// conversation cannot show every answer the way it is asked for; and its
    /// message boundaries are refused as not prefix-stable. `None` when a
    /// supervised message is not an assistant turn (a supervised prompt
    /// turn): such a sample is encoded as a whole.
    ///
    /// # Errors
    /// The template renders an answer that does not follow its prompt.
    pub fn answers_as_asked(&self, tmpl: &ChatTemplate, thinking: bool) -> Result<Option<Vec<ChatSample>>, TemplateError> {
        if self.messages.iter().any(|m| m.train && m.role != "assistant") {
            return Ok(None);
        }
        let mut out = Vec::new();
        for at in (0..self.messages.len()).filter(|&at| self.messages[at].train) {
            out.push(self.until(at + 1).rendered_final_turn(tmpl, thinking)?);
        }
        Ok(Some(out))
    }

    /// The first `len` messages as a sample of their own.
    fn until(&self, len: usize) -> ChatSample {
        ChatSample { messages: self.messages[..len].to_vec(), tools: self.tools.clone(), rendered: None }
    }

    /// This sample's last message as one example the way the model is asked
    /// for it ([`ChatTemplate::render_with_supervised_suffix`]): the prompt is
    /// context and only what the template renders for that message is
    /// supervised.
    fn rendered_final_turn(&self, tmpl: &ChatTemplate, thinking: bool) -> Result<ChatSample, TemplateError> {
        let values: Vec<minijinja::Value> = self.messages.iter().map(|m| m.to_template_value(tmpl.reads_argument_objects())).collect();
        let tools = (!self.tools.is_empty()).then(|| minijinja::Value::from(self.tools.clone()));
        let (full, answer) = tmpl.render_with_supervised_suffix(&values, tools, thinking)?;
        let (prompt, completion) = full.split_at(answer.start);
        Ok(ChatSample { messages: Vec::new(), tools: self.tools.clone(), rendered: Some(Rendered::answer(prompt, completion)) })
    }

    /// This sample as ONE example in which every supervised answer is
    /// trained the way [`ChatSample::answers_as_asked`] trains it alone, and
    /// everything else is context: the conversation is rendered once instead
    /// of once per answer, so a dialogue of `n` answers costs one sequence
    /// where `answers_as_asked` costs `n` growing prefixes.
    ///
    /// The supervised text of each answer is byte-for-byte what its
    /// `answers_as_asked` example supervises (the template's rendering of the
    /// answer as the last turn, end-of-turn token included). The context
    /// before the first answer is its prompt; the context between two answers
    /// is the next answer's prompt from the point where the template's
    /// rendering of the earlier answer as a middle turn ends. An earlier
    /// answer is therefore shown to the later ones in the form it was
    /// trained in, which a template that renders a last turn differently
    /// (Qwen3's reasoning block) shows a model asked at a later turn without
    /// that block; the targets are exact, the history is the one difference.
    ///
    /// `None` when nothing is supervised or a supervised message is not an
    /// assistant turn.
    ///
    /// # Errors
    /// The template has no recognisable end-of-turn marker, an answer holds
    /// that marker anywhere but at its end, or an earlier answer is not
    /// rendered with an end-of-turn marker in the next answer's prompt: no
    /// boundary is guessed.
    pub fn answers_packed(&self, tmpl: &ChatTemplate, thinking: bool) -> Result<Option<ChatSample>, TemplateError> {
        if !self.messages.iter().any(|m| m.train) || self.messages.iter().any(|m| m.train && m.role != "assistant") {
            return Ok(None);
        }
        let end = tmpl.assistant_turn_end()?;
        let mut segments: Vec<RenderedSegment> = Vec::new();
        let mut previous_prompt = String::new();
        for at in (0..self.messages.len()).filter(|&at| self.messages[at].train) {
            let values: Vec<minijinja::Value> = self.messages[..=at].iter().map(|m| m.to_template_value(tmpl.reads_argument_objects())).collect();
            let tools = (!self.tools.is_empty()).then(|| minijinja::Value::from(self.tools.clone()));
            let (full, answer) = tmpl.render_with_supervised_suffix(&values, tools, thinking)?;
            let (prompt, completion) = full.split_at(answer.start);
            if completion.matches(end.as_str()).count() != 1 || !completion.ends_with(end.as_str()) {
                return Err(TemplateError(format!("message {at} is not a single answer closed by the template's end-of-turn marker {end:?}")));
            }
            let context = if segments.is_empty() {
                prompt
            } else {
                // The earlier answer starts where the two prompts part and,
                // rendered as a middle turn, ends at its first end-of-turn marker.
                let shared = previous_prompt.bytes().zip(prompt.bytes()).take_while(|(a, b)| a == b).count();
                let shared = (0..=shared).rev().find(|&n| prompt.is_char_boundary(n)).unwrap_or(0);
                let after = prompt[shared..]
                    .find(end.as_str())
                    .ok_or_else(|| TemplateError(format!("the prompt of message {at} does not render the earlier answer closed by the end-of-turn marker {end:?}")))?;
                &prompt[shared + after + end.len()..]
            };
            segments.push(RenderedSegment { text: context.to_string(), supervised: false });
            segments.push(RenderedSegment { text: completion.to_string(), supervised: true });
            previous_prompt = prompt.to_string();
        }
        Ok(Some(ChatSample { messages: Vec::new(), tools: self.tools.clone(), rendered: Some(Rendered { segments }) }))
    }

    /// Whether the last message is the only supervised one and it is an
    /// assistant turn: the shape [`ChatSample::final_turn_as_asked`] masks
    /// exactly whatever the context holds.
    #[must_use]
    pub fn supervises_only_final_turn(&self) -> bool {
        self.messages.last().is_some_and(|m| m.train && m.role == "assistant") && self.messages.iter().filter(|m| m.train).count() == 1
    }

    /// Whether the context before the last message holds a tool call or a tool
    /// result, which a template may render differently there than when it is
    /// the whole conversation.
    #[must_use]
    pub fn context_holds_tool_turns(&self) -> bool {
        self.messages.split_last().is_some_and(|(_, context)| context.iter().any(|m| m.role == "tool" || !m.tool_calls.is_empty()))
    }

    /// This sample as a single example when only its last message is
    /// supervised (see [`ChatSample::supervises_only_final_turn`]), masked at
    /// the one boundary between the prompt its context renders and the answer
    /// ([`ChatTemplate::render_with_supervised_suffix`]); `None` for any other
    /// record, which keeps the whole-conversation mask (and its refusals) or
    /// [`ChatSample::answers_as_asked`].
    ///
    /// # Errors
    /// The template renders the answer so that it does not follow its prompt.
    pub fn final_turn_as_asked(&self, tmpl: &ChatTemplate, thinking: bool) -> Result<Option<ChatSample>, TemplateError> {
        if !self.supervises_only_final_turn() {
            return Ok(None);
        }
        self.rendered_final_turn(tmpl, thinking).map(Some)
    }

    /// The conversation obeys [`check_tool_pairing`].
    pub fn validate(&self) -> Result<(), String> {
        check_tool_pairing(&self.messages)
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
    /// template, not an ad-hoc approximation. A message's whole range is
    /// masked by its `train` flag, so a trained assistant turn trains its
    /// end-of-turn token too: that token is how the model learns to stop.
    /// Refuses a conversation that breaks [`check_tool_pairing`], so no
    /// training path can encode corrupt tool pairing. Fails (does not silently mismask) if the template's rendering of some
    /// message is not prefix-stable -- see `render_with_message_boundaries`'s
    /// doc for exactly when that happens.
    pub fn encode_with(&self, tok: &dyn Tokenizer, tmpl: &ChatTemplate, opts: RenderOpts) -> Result<(Vec<u32>, Vec<bool>), TemplateError> {
        if let Some(Rendered { segments }) = &self.rendered {
            let (mut ids, mut mask) = (Vec::new(), Vec::new());
            for RenderedSegment { text, supervised } in segments {
                let encoded = tok.encode(text);
                mask.extend(std::iter::repeat_n(*supervised, encoded.len()));
                ids.extend(encoded);
            }
            return Ok((ids, mask));
        }
        self.validate().map_err(TemplateError)?;
        let messages: Vec<ChatMessage> = if opts.keep_reasoning {
            self.messages
                .iter()
                .map(|m| if m.role == "assistant" && m.train { ChatMessage { content: m.content.replace(THINK_END, THINK_END_SENTINEL), ..m.clone() } } else { m.clone() })
                .collect()
        } else {
            self.messages.clone()
        };
        let values: Vec<minijinja::Value> = messages.iter().map(|m| m.to_template_value(tmpl.reads_argument_objects())).collect();
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
    /// Only meaningful on `WireRole::Assistant`: the reasoning before the turn.
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
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
/// JSON, a `tool_call_id` on every tool result, and every tool result answering a call made earlier in the same
/// conversation ([`check_tool_pairing`]). `messages[i]` in an error is the
/// index into `wire`.
pub(crate) fn messages_from_wire(wire: Vec<WireMessage>) -> Result<Vec<ChatMessage>, String> {
    let mut messages = Vec::with_capacity(wire.len());

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
            tool_calls.push(ToolCall { id: tc.id, name: tc.function.name, arguments: tc.function.arguments });
        }

        if m.reasoning_content.is_some() && !matches!(m.role, WireRole::Assistant) {
            return Err(format!("messages[{i}]: only an \"assistant\" message carries \"reasoning_content\""));
        }
        if matches!(m.role, WireRole::Tool) && m.tool_call_id.is_none() {
            return Err(format!("messages[{i}]: a \"tool\"-role message must carry \"tool_call_id\""));
        }

        messages.push(ChatMessage { role: role.to_string(), content: m.content, tool_calls, tool_call_id: m.tool_call_id, reasoning_content: m.reasoning_content, train: m.train });
    }
    check_tool_pairing(&messages)?;
    Ok(messages)
}

/// The pairing rules every conversation that reaches training obeys, whatever
/// produced it (a wire record, a recorded trajectory, an SDK caller):
/// * every assistant tool call id is unique within the conversation;
/// * a tool message naming a `tool_call_id` answers exactly one EARLIER call
///   that no earlier tool message answered;
/// * a tool message without an id answers the earliest call still unanswered
///   (the order the chat template renders consecutive results in); there must
///   be one.
///
/// A repeated id is the failure this exists for: results pair to calls by id,
/// so a trajectory whose calls are all named alike gives every call every
/// result - duplicated, reordered and future results in the context, which
/// nothing downstream can detect. Errors name the message index (`messages[i]`).
pub fn check_tool_pairing(messages: &[ChatMessage]) -> Result<(), String> {
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    let mut unanswered: Vec<Option<&str>> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        for (j, tc) in m.tool_calls.iter().enumerate() {
            if let Some(id) = tc.id.as_deref() {
                if !seen.insert(id) {
                    return Err(format!(
                        "messages[{i}].tool_calls[{j}]: tool call id {id:?} is used more than once in this conversation -- \
                         results cannot be paired to calls by id, so the record is refused"
                    ));
                }
            }
            unanswered.push(tc.id.as_deref());
        }
        if m.role != "tool" {
            continue;
        }
        let at = match m.tool_call_id.as_deref() {
            Some(id) => unanswered.iter().position(|c| *c == Some(id)).ok_or_else(|| {
                let why = if seen.contains(id) {
                    "that call was already answered"
                } else {
                    "it does not match any tool_calls id from an earlier assistant message in this sample"
                };
                format!("messages[{i}]: tool_call_id {id:?}: {why} -- a tool response must answer one earlier, unanswered call")
            })?,
            None if unanswered.is_empty() => return Err(format!("messages[{i}]: a tool message without an id has no unanswered call to answer")),
            None => 0,
        };
        unanswered.remove(at);
    }
    Ok(())
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
        ChatMessage { role: "assistant".into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None, reasoning_content: None, train }
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
        let tmpl = ChatTemplate::compile(&std::fs::read_to_string(brain_testutil::testdata("qwen3/chat_template.jinja")).unwrap()).unwrap();
        let turn = |role: &str, text: &str, train: bool| ChatMessage { role: role.into(), content: text.into(), tool_calls: Vec::new(), tool_call_id: None, reasoning_content: None, train };
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

    /// A conversation in which the model calls a tool twice before it answers
    /// trains under Qwen3's own template, in either thinking mode, with every
    /// assistant turn supervised: one example per turn, whose prompt and
    /// completion are what the serving renderer gives for the same prefix with
    /// its generation prompt - the completion being the turn as the template
    /// writes it after that prompt.
    #[test]
    fn every_turn_of_a_tool_conversation_is_trained_as_serving_renders_it() {
        use crate::qwen_chat::{self, ChatMessage as Served, TemplateOpts, ToolCallMsg};
        let tmpl = ChatTemplate::compile(&std::fs::read_to_string(brain_testutil::testdata("qwen3/chat_template.jinja")).unwrap()).unwrap();
        let schema = r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}"#;
        let call = |id: &str, city: &str| ToolCall { id: Some(id.into()), name: "get_weather".into(), arguments: format!(r#"{{"city": "{city}"}}"#) };
        let served_call = |id: &str, city: &str| ToolCallMsg { id: id.into(), name: "get_weather".into(), arguments: format!(r#"{{"city": "{city}"}}"#) };
        let conversation = ChatSample {
            messages: vec![
                ChatMessage::user("Weather in Paris and Rome?"),
                ChatMessage::assistant_tool_calls("", vec![call("c1", "Paris")], true),
                ChatMessage::tool_result("18C"),
                ChatMessage::assistant_tool_calls("", vec![call("c2", "Rome")], true),
                ChatMessage::tool_result("22C"),
                ChatMessage::assistant("Paris 18C, Rome 22C.", true),
            ],
            tools: vec![crate::chat_template::parse_json_ordered(schema).unwrap()],
            rendered: None,
        };
        let served = [
            Served::user("Weather in Paris and Rome?"),
            Served::assistant("").with_tool_calls(vec![served_call("c1", "Paris")]),
            Served::tool("18C"),
            Served::assistant("").with_tool_calls(vec![served_call("c2", "Rome")]),
            Served::tool("22C"),
            Served::assistant("Paris 18C, Rome 22C."),
        ];
        for thinking in [false, true] {
            let opts = TemplateOpts { enable_thinking: thinking, ..TemplateOpts::default() };
            let samples = conversation.answers_as_asked(&tmpl, thinking).unwrap().expect("every turn is an assistant turn");
            assert_eq!(samples.len(), 3);
            for (sample, at) in samples.iter().zip([1, 3, 5]) {
                let prompt = qwen_chat::render_for_generation(&served[..at], &[schema.to_string()], opts.clone()).unwrap();
                let full = qwen_chat::render(&served[..=at], &[schema.to_string()], TemplateOpts { add_generation_prompt: false, ..opts.clone() }).unwrap();
                let rendered = sample.rendered.as_ref().expect("a rendered example");
                assert_eq!(rendered.segments[0].text, prompt, "thinking={thinking}, turn {at}");
                assert_eq!(rendered.segments[1].text, full.strip_prefix(prompt.as_str()).unwrap(), "thinking={thinking}, turn {at}");
            }
            assert!(samples[0].rendered.as_ref().unwrap().segments[1].text.contains("<tool_call>"));
        }
    }

    /// A turn's reasoning travels as `reasoning_content`, the field serving
    /// sends back and the template renders as the turn's think block, so a
    /// trained turn is the turn as serving renders it, reasoning included, in
    /// either thinking mode; the same turn without it keeps its empty block.
    #[test]
    fn a_turns_reasoning_is_trained_as_serving_renders_it() {
        use crate::qwen_chat::{self, ChatMessage as Served, TemplateOpts, ToolCallMsg};
        let tmpl = ChatTemplate::compile(&std::fs::read_to_string(brain_testutil::testdata("qwen3/chat_template.jinja")).unwrap()).unwrap();
        let schema = r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}"#;
        let call = ToolCall { id: Some("c1".into()), name: "get_weather".into(), arguments: r#"{"city": "Paris"}"#.into() };
        let reasoned = |m: ChatMessage, reasoning: &str| ChatMessage { reasoning_content: Some(reasoning.into()), ..m };
        let conversation = ChatSample {
            messages: vec![
                ChatMessage::user("Weather in Paris?"),
                reasoned(ChatMessage::assistant_tool_calls("", vec![call], true), "I should ask the tool."),
                ChatMessage::tool_result("18C"),
                reasoned(ChatMessage::assistant("Paris is 18C.", true), "The tool said 18C."),
            ],
            tools: vec![crate::chat_template::parse_json_ordered(schema).unwrap()],
            rendered: None,
        };
        let served = [
            Served::user("Weather in Paris?"),
            Served { reasoning_content: Some("I should ask the tool.".into()), ..Served::assistant("") }.with_tool_calls(vec![ToolCallMsg { id: "c1".into(), name: "get_weather".into(), arguments: r#"{"city": "Paris"}"#.into() }]),
            Served::tool("18C"),
            Served { reasoning_content: Some("The tool said 18C.".into()), ..Served::assistant("Paris is 18C.") },
        ];
        let opts = TemplateOpts { enable_thinking: true, ..TemplateOpts::default() };
        let samples = conversation.answers_as_asked(&tmpl, true).unwrap().expect("every turn is an assistant turn");
        assert_eq!(samples.len(), 2);
        for (sample, at) in samples.iter().zip([1, 3]) {
            let prompt = qwen_chat::render_for_generation(&served[..at], &[schema.to_string()], opts.clone()).unwrap();
            let full = qwen_chat::render(&served[..=at], &[schema.to_string()], TemplateOpts { add_generation_prompt: false, ..opts.clone() }).unwrap();
            let rendered = sample.rendered.as_ref().expect("a rendered example");
            assert_eq!(rendered.segments[0].text, prompt, "turn {at}");
            assert_eq!(rendered.segments[1].text, full.strip_prefix(prompt.as_str()).unwrap(), "turn {at}");
        }
        assert!(samples[0].rendered.as_ref().unwrap().segments[1].text.contains("I should ask the tool.\n</think>"));
        assert!(samples[1].rendered.as_ref().unwrap().segments[1].text.contains("The tool said 18C.\n</think>\n\nParis is 18C."));
    }

    #[test]
    fn reasoning_content_is_read_from_the_wire_on_an_assistant_turn_only() {
        let line = |role: &str| format!(r#"{{"messages":[{{"role":"{role}","content":"x","reasoning_content":"r","train":false}}]}}"#);
        let dir = std::env::temp_dir().join(format!("brain-reasoning-wire-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("d.jsonl");
        std::fs::write(&path, line("assistant")).unwrap();
        let samples = ChatSample::from_jsonl(&path).unwrap();
        assert_eq!(samples[0].messages[0].reasoning_content.as_deref(), Some("r"));
        std::fs::write(&path, line("user")).unwrap();
        let err = ChatSample::from_jsonl(&path).err().expect("refused").to_string();
        assert!(err.contains("reasoning_content"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
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

    fn jsonl_error(name: &str, body: &str) -> String {
        let dir = std::env::temp_dir().join(format!("brain-chat-jsonl-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("record.jsonl");
        std::fs::write(&path, body).unwrap();
        ChatSample::from_jsonl(&path).unwrap_err().to_string()
    }

    const CALL_0: &str = r#"{"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function","function":{"name":"sh","arguments":"{}"}}],"train":true}"#;
    const RESULT_0: &str = r#"{"role":"tool","content":"out","tool_call_id":"call_0","train":false}"#;

    /// The corruption this rule exists for: a recorded agent run in which every
    /// call is named `call_0`. Pairing results to calls by id then gives each
    /// call every result; the record must be refused, naming the line and id.
    #[test]
    fn from_jsonl_rejects_a_tool_call_id_used_twice_in_a_conversation() {
        let body = format!(r#"{{"messages":[{CALL_0},{RESULT_0},{CALL_0},{RESULT_0}]}}"#);
        let err = jsonl_error("dupcall", &body);
        assert!(err.contains(":1:") && err.contains("call_0") && err.contains("more than once"), "names line and id: {err}");
    }

    #[test]
    fn from_jsonl_rejects_a_tool_call_answered_twice() {
        let body = format!(r#"{{"messages":[{CALL_0},{RESULT_0},{RESULT_0}]}}"#);
        let err = jsonl_error("answeredtwice", &body);
        assert!(err.contains("call_0") && err.contains("already answered"), "got: {err}");
    }

    /// A result placed before the call it names is a future result.
    #[test]
    fn from_jsonl_rejects_a_tool_result_that_precedes_its_call() {
        let body = format!(r#"{{"messages":[{RESULT_0},{CALL_0}]}}"#);
        let err = jsonl_error("future", &body);
        assert!(err.contains("call_0") && err.contains("does not match"), "got: {err}");
    }

    #[test]
    fn a_tool_message_without_an_id_answers_the_next_unanswered_call_in_order() {
        let call = |id: &str| ToolCall { id: Some(id.into()), name: "f".into(), arguments: "{}".into() };
        let sample = |messages| ChatSample { messages, tools: Vec::new(), rendered: None };
        let calls = ChatMessage::assistant_tool_calls("", vec![call("a"), call("b")], true);
        sample(vec![ChatMessage::user("q"), calls.clone(), ChatMessage::tool_result("1"), ChatMessage::tool_result("2")]).validate().expect("in order");
        let err = sample(vec![ChatMessage::user("q"), calls, ChatMessage::tool_result("1"), ChatMessage::tool_result("2"), ChatMessage::tool_result("3")]).validate().unwrap_err();
        assert!(err.contains("no unanswered call"), "got: {err}");
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

    /// Qwen3.8's template iterates a call's arguments (`arguments|items`), so a
    /// record's JSON text must reach it parsed and in its key order; Qwen3's
    /// passes the text through, and still gets it verbatim.
    #[test]
    fn tool_call_arguments_reach_the_template_in_the_shape_it_reads() {
        let call = ChatMessage::assistant_tool_calls("", vec![ToolCall { id: Some("c".into()), name: "f".into(), arguments: r#"{"b": 2, "a": "x"}"#.into() }], true);
        let render = |template: &str| {
            let tmpl = ChatTemplate::compile(template).unwrap();
            let messages = minijinja::Value::from(vec![call.to_template_value(tmpl.reads_argument_objects())]);
            tmpl.render(messages, None, false, &std::collections::BTreeMap::new()).unwrap()
        };
        let iterating = "{%- for m in messages %}{%- for tc in m.tool_calls %}{%- for k, v in tc.function.arguments|items %}{{ k }}={{ v }};{%- endfor %}{%- endfor %}{%- endfor %}";
        assert_eq!(render(iterating), "b=2;a=x;");
        let passing = "{%- for m in messages %}{%- for tc in m.tool_calls %}{{ tc.function.arguments }}{%- endfor %}{%- endfor %}";
        assert_eq!(render(passing), r#"{"b": 2, "a": "x"}"#);
    }

    #[test]
    fn to_template_value_omits_empty_tool_calls_and_tool_call_id() {
        // Fields the template checks with `is defined`/truthiness must be
        // genuinely ABSENT when empty, not present-but-empty -- an empty
        // `tool_calls: []` vs. a missing key can take different branches in
        // a real template (`{%- if message.tool_calls %}`).
        let m = ChatMessage::user("hi");
        let v = m.to_template_value(false);
        assert_eq!(v.get_attr("tool_calls").unwrap().kind(), minijinja::value::ValueKind::Undefined);
        assert_eq!(v.get_attr("tool_call_id").unwrap().kind(), minijinja::value::ValueKind::Undefined);
    }
}
