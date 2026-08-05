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
//! Output (consumed unchanged by `model::fit` / `brain qwen finetune`):
//!   * `train.u32.bin` / `val.u32.bin` — `u32` token ids (Qwen's 151936 vocab).
//!   * `train.mask.bin` / `val.mask.bin` — `u8` per-token mask (1 = trainable).
//!   * `meta.json` — `{ vocab_size, token_width: 32 }`.
//!
//! Prompt and response are encoded **separately** and concatenated — exactly how
//! inference sees them (the prompt is encoded alone, the model then generates) —
//! so there is no train/inference tokenization skew across the boundary. Each
//! example is terminated by `<|endoftext|>` so windows can be sampled aligned to
//! example starts.

use std::io;
use std::path::Path;

use crate::binio;
use crate::chat_template::{ChatTemplate, TemplateError};
use crate::qwen_tokenizer::QwenBpe;
use crate::tokenizer::Tokenizer;

/// `<|endoftext|>` — the document/example separator (also Qwen's pad/eos base).
pub const ENDOFTEXT: u32 = 151643;

/// One supervised chat example: a prompt (system optional + user) and the
/// assistant response the model must learn to produce.
#[derive(Clone, Debug)]
pub struct ChatExample {
    pub system: Option<String>,
    pub user: String,
    /// The assistant turn content to train on (e.g. a `<tool_call>…</tool_call>`
    /// block, or a `<think>…</think>` + answer). `<|im_end|>` is appended for you.
    pub assistant: String,
}

impl ChatExample {
    pub fn new(user: impl Into<String>, assistant: impl Into<String>) -> ChatExample {
        ChatExample { system: None, user: user.into(), assistant: assistant.into() }
    }
    pub fn with_system(system: impl Into<String>, user: impl Into<String>, assistant: impl Into<String>) -> ChatExample {
        ChatExample { system: Some(system.into()), user: user.into(), assistant: assistant.into() }
    }

    /// The prompt string through `<|im_start|>assistant\n` (what inference sees).
    pub fn prompt_str(&self, tok: &QwenBpe) -> String {
        let mut msgs: Vec<(&str, &str)> = Vec::new();
        if let Some(s) = &self.system {
            msgs.push(("system", s));
        }
        msgs.push(("user", &self.user));
        tok.apply_chat_template(&msgs, true)
    }

    /// Encode to `(ids, mask)`: prompt ids masked (false), response ids trained
    /// (true), then a masked `<|endoftext|>` separator.
    pub fn encode(&self, tok: &QwenBpe) -> (Vec<u32>, Vec<bool>) {
        let prompt = tok.encode(&self.prompt_str(tok));
        let resp = tok.encode(&format!("{}<|im_end|>\n", self.assistant));
        let mut ids = Vec::with_capacity(prompt.len() + resp.len() + 1);
        let mut mask = Vec::with_capacity(ids.capacity());
        ids.extend_from_slice(&prompt);
        mask.extend(std::iter::repeat_n(false, prompt.len()));
        ids.extend_from_slice(&resp);
        mask.extend(std::iter::repeat_n(true, resp.len()));
        ids.push(ENDOFTEXT);
        mask.push(false);
        (ids, mask)
    }
}

/// Encode a set of examples into one `(ids, mask)` stream.
pub fn encode_split(examples: &[ChatExample], tok: &QwenBpe) -> (Vec<u32>, Vec<bool>) {
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for ex in examples {
        let (i, m) = ex.encode(tok);
        ids.extend_from_slice(&i);
        mask.extend_from_slice(&m);
    }
    (ids, mask)
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
}

impl ChatSample {
    /// Encode to `(ids, mask)` by rendering the WHOLE conversation through
    /// `tmpl` once (via [`ChatTemplate::render_with_message_boundaries`]),
    /// then encoding each message's own byte range of that single rendered
    /// text and concatenating -- so tool-call/tool-response framing and
    /// think-block placement all come from the checkpoint's OWN template,
    /// not a hand-rolled approximation. Terminated by a masked
    /// `<|endoftext|>`. Fails (does not silently mismask) if the template's
    /// rendering of some message is not prefix-stable -- see
    /// `render_with_message_boundaries`'s doc for exactly when that happens.
    pub fn encode(&self, tok: &QwenBpe, tmpl: &ChatTemplate) -> Result<(Vec<u32>, Vec<bool>), TemplateError> {
        let values: Vec<minijinja::Value> = self.messages.iter().map(ChatMessage::to_template_value).collect();
        let tools = (!self.tools.is_empty()).then(|| minijinja::Value::from(self.tools.clone()));
        let (full, ranges) = tmpl.render_with_message_boundaries(&values, tools)?;

        let mut ids = Vec::new();
        let mut mask = Vec::new();
        for (m, range) in self.messages.iter().zip(ranges) {
            let tid = tok.encode(&full[range]);
            mask.extend(std::iter::repeat_n(m.train, tid.len()));
            ids.extend(tid);
        }
        ids.push(ENDOFTEXT);
        mask.push(false);
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

#[derive(serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum WireRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: WireRole,
    /// Required: bench's exporter always writes this key (possibly `""`),
    /// never omits it -- an absent `content` is a real shape violation, not
    /// something to paper over with a default.
    content: String,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
    /// Only meaningful on `WireRole::Tool`.
    #[serde(default)]
    tool_call_id: Option<String>,
    /// No default: a message with no explicit supervision boundary is
    /// rejected rather than silently treated as all-context or all-trained.
    train: bool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireToolCall {
    #[serde(default)]
    id: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    r#type: Option<String>,
    function: WireFunction,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFunction {
    name: String,
    /// OpenAI wire format requires a JSON-ENCODED STRING here, not a nested
    /// object -- bench's `_stringify_tool_call_arguments` (messages.py)
    /// writes it that way. Typed as `String` so serde itself rejects a
    /// record that regresses to the old (pre-fix) object shape, instead of
    /// silently accepting it and later double-encoding it into the rendered
    /// <tool_call> block.
    arguments: String,
}

fn sample_from_wire(record: WireRecord) -> Result<ChatSample, String> {
    if record.messages.is_empty() {
        return Err("\"messages\" is empty".to_string());
    }
    let messages = record
        .messages
        .into_iter()
        .map(|m| {
            let role = match m.role {
                WireRole::System => "system",
                WireRole::User => "user",
                WireRole::Assistant => "assistant",
                WireRole::Tool => "tool",
            };
            let tool_calls =
                m.tool_calls.into_iter().map(|tc| ToolCall { id: tc.id, name: tc.function.name, arguments: tc.function.arguments }).collect();
            ChatMessage { role: role.to_string(), content: m.content, tool_calls, tool_call_id: m.tool_call_id, train: m.train }
        })
        .collect();
    Ok(ChatSample { messages, tools: record.tools })
}

/// Encode a set of packed samples into one `(ids, mask)` stream.
pub fn encode_sample_split(samples: &[ChatSample], tok: &QwenBpe, tmpl: &ChatTemplate) -> Result<(Vec<u32>, Vec<bool>), TemplateError> {
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for s in samples {
        let (i, m) = s.encode(tok, tmpl)?;
        ids.extend_from_slice(&i);
        mask.extend_from_slice(&m);
    }
    Ok((ids, mask))
}

/// Write a train/val split of packed [`ChatSample`]s to `dir`, in the same
/// on-disk layout as [`prepare_chat`]. `tmpl` is the checkpoint's OWN
/// chat template (compiled once by the caller; see `chat_template`).
pub fn prepare_chat_samples(
    train: &[ChatSample],
    val: &[ChatSample],
    tok: &QwenBpe,
    tmpl: &ChatTemplate,
    vocab: usize,
    dir: &Path,
) -> Result<(), PrepareError> {
    std::fs::create_dir_all(dir).map_err(PrepareError::Io)?;
    let (train_ids, train_mask) = encode_sample_split(train, tok, tmpl).map_err(PrepareError::Template)?;
    let (val_ids, val_mask) = encode_sample_split(val, tok, tmpl).map_err(PrepareError::Template)?;
    binio::write_u32_bin(&dir.join("train.u32.bin"), &train_ids).map_err(PrepareError::Io)?;
    binio::write_mask_bin(&dir.join("train.mask.bin"), &train_mask).map_err(PrepareError::Io)?;
    binio::write_u32_bin(&dir.join("val.u32.bin"), &val_ids).map_err(PrepareError::Io)?;
    binio::write_mask_bin(&dir.join("val.mask.bin"), &val_mask).map_err(PrepareError::Io)?;
    std::fs::write(dir.join("meta.json"), binio::Meta::vocab_only(vocab)).map_err(PrepareError::Io)?;
    Ok(())
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

/// Write a train/val split to `dir` as brain's masked token-dataset layout.
/// `vocab` is the MODEL's vocab (from its `config.json`), which must match the
/// checkpoint's `lm_head` — not the tokenizer's derived size.
pub fn prepare_chat(
    train: &[ChatExample],
    val: &[ChatExample],
    tok: &QwenBpe,
    vocab: usize,
    dir: &Path,
) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let (train_ids, train_mask) = encode_split(train, tok);
    let (val_ids, val_mask) = encode_split(val, tok);
    binio::write_u32_bin(&dir.join("train.u32.bin"), &train_ids)?;
    binio::write_mask_bin(&dir.join("train.mask.bin"), &train_mask)?;
    binio::write_u32_bin(&dir.join("val.u32.bin"), &val_ids)?;
    binio::write_mask_bin(&dir.join("val.mask.bin"), &val_mask)?;
    std::fs::write(dir.join("meta.json"), binio::Meta::vocab_only(vocab))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
