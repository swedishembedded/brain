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

/// A tool call within an assistant turn: `{"name": ..., "arguments": {...}}`,
/// rendered exactly as `data::toolcall`'s single-turn path does (the
/// established brain-side convention, shared with bench's `qwen-hermes`
/// export format).
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// One message in a multi-turn chat sample. Unlike [`ChatExample`] (one
/// fixed system/user/assistant boundary), a [`ChatSample`] can carry many
/// trainable assistant turns interleaved with tool results in one packed
/// conversation -- `train` is per-message, not implied by position, which is
/// what lets a whole trajectory be one sample instead of N nested-prefix
/// samples repeating the same context (see bench's `extract_packed_sample`).
///
/// `role` is always `"system"`, `"user"`, or `"assistant"` by construction —
/// [`ChatSample::from_jsonl`] folds a `"tool"`-role message into a
/// `"user"`-role one wrapped in `<tool_response>` tags, the Hermes/Qwen
/// tool-calling convention (no chat-template support for a bare `"tool"`
/// role turn).
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// Whether this message's tokens are supervised (assistant decision
    /// turns the model should learn) or masked context (system/user/tool-
    /// result turns, and any assistant turn a producer excluded from
    /// training).
    pub train: bool,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> ChatMessage {
        ChatMessage { role: "system".into(), content: content.into(), tool_calls: Vec::new(), train: false }
    }
    pub fn user(content: impl Into<String>) -> ChatMessage {
        ChatMessage { role: "user".into(), content: content.into(), tool_calls: Vec::new(), train: false }
    }
    pub fn assistant(content: impl Into<String>, train: bool) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: content.into(), tool_calls: Vec::new(), train }
    }
    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>, train: bool) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: content.into(), tool_calls, train }
    }
    /// A tool result, rendered as a `user`-role turn wrapped in
    /// `<tool_response>` tags (never trainable).
    pub fn tool_result(content: impl AsRef<str>) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: format!("<tool_response>\n{}\n</tool_response>", content.as_ref()),
            tool_calls: Vec::new(),
            train: false,
        }
    }

    /// This message's content, with any tool calls inlined as
    /// `<tool_call>{...}</tool_call>` blocks -- matches
    /// `data::toolcall`'s single-call rendering exactly.
    fn rendered_content(&self) -> String {
        if self.tool_calls.is_empty() {
            return self.content.clone();
        }
        let mut blocks: Vec<String> = Vec::new();
        if !self.content.is_empty() {
            blocks.push(self.content.clone());
        }
        for tc in &self.tool_calls {
            blocks.push(format!("<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>", tc.name, tc.arguments));
        }
        blocks.join("\n")
    }
}

/// A packed multi-turn chat sample: a full conversation, encoded and masked
/// message-by-message. See [`ChatMessage`] for why `train` lives per-message.
#[derive(Clone, Debug, Default)]
pub struct ChatSample {
    pub messages: Vec<ChatMessage>,
}

impl ChatSample {
    /// Encode to `(ids, mask)`. Each message is framed via
    /// `QwenBpe::frame_message` -- the SAME function `apply_chat_template`
    /// (what inference renders) folds over -- so concatenating every
    /// message's framed+encoded span is byte-identical to encoding the whole
    /// conversation through the batch template at once; only the per-message
    /// mask value differs. Terminated by a masked `<|endoftext|>`.
    pub fn encode(&self, tok: &QwenBpe) -> (Vec<u32>, Vec<bool>) {
        let mut ids = Vec::new();
        let mut mask = Vec::new();
        for m in &self.messages {
            let framed = tok.frame_message(&m.role, &m.rendered_content());
            let tid = tok.encode(&framed);
            mask.extend(std::iter::repeat_n(m.train, tid.len()));
            ids.extend(tid);
        }
        ids.push(ENDOFTEXT);
        mask.push(false);
        (ids, mask)
    }

    /// Parse bench's `generic-messages-v2` JSONL export: one packed sample
    /// per line, `{"messages":[{"role","content","train",...}], ...}`.
    /// `messages[].train` is REQUIRED on every message -- a record with no
    /// explicit supervision boundary is rejected rather than silently
    /// treated as all-context or all-trained (either would be a silent
    /// no-op or a silent prompt-leak into the loss).
    pub fn from_jsonl(path: &Path) -> io::Result<Vec<ChatSample>> {
        let text = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("{}:{}: invalid JSON: {e}", path.display(), lineno + 1))
            })?;
            out.push(sample_from_json(&v).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("{}:{}: {e}", path.display(), lineno + 1))
            })?);
        }
        Ok(out)
    }
}

fn sample_from_json(v: &serde_json::Value) -> Result<ChatSample, String> {
    let msgs = v["messages"].as_array().ok_or("missing \"messages\" array")?;
    let mut messages = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let role = m["role"].as_str().ok_or_else(|| format!("messages[{i}]: missing \"role\""))?;
        let content = m["content"].as_str().unwrap_or("").to_string();
        let train = m
            .get("train")
            .ok_or_else(|| format!("messages[{i}]: missing \"train\" -- every message must state its supervision boundary explicitly"))?
            .as_bool()
            .ok_or_else(|| format!("messages[{i}]: \"train\" must be a bool"))?;

        if role == "tool" {
            messages.push(ChatMessage::tool_result(content));
            continue;
        }

        let tool_calls: Vec<ToolCall> = m["tool_calls"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .map(|(j, tc)| {
                        let f = &tc["function"];
                        let name = f["name"].as_str().ok_or_else(|| format!("messages[{i}].tool_calls[{j}]: missing function.name"))?;
                        Ok(ToolCall { name: name.to_string(), arguments: f["arguments"].clone() })
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .transpose()?
            .unwrap_or_default();

        messages.push(ChatMessage { role: role.to_string(), content, tool_calls, train });
    }
    if messages.is_empty() {
        return Err("\"messages\" is empty".to_string());
    }
    Ok(ChatSample { messages })
}

/// Encode a set of packed samples into one `(ids, mask)` stream.
pub fn encode_sample_split(samples: &[ChatSample], tok: &QwenBpe) -> (Vec<u32>, Vec<bool>) {
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for s in samples {
        let (i, m) = s.encode(tok);
        ids.extend_from_slice(&i);
        mask.extend_from_slice(&m);
    }
    (ids, mask)
}

/// Write a train/val split of packed [`ChatSample`]s to `dir`, in the same
/// on-disk layout as [`prepare_chat`].
pub fn prepare_chat_samples(train: &[ChatSample], val: &[ChatSample], tok: &QwenBpe, vocab: usize, dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let (train_ids, train_mask) = encode_sample_split(train, tok);
    let (val_ids, val_mask) = encode_sample_split(val, tok);
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
        // role: "tool" folds into a user-role <tool_response> turn, never trainable.
        assert_eq!(s.messages[3].role, "user");
        assert!(s.messages[3].content.contains("<tool_response>"));
        assert!(s.messages[3].content.contains("18C, sunny"));
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

    #[test]
    fn rendered_content_inlines_tool_calls_matching_the_toolcall_convention() {
        let m = ChatMessage::assistant_tool_calls(
            "checking",
            vec![ToolCall { name: "get_weather".into(), arguments: serde_json::json!({"location": "Paris"}) }],
            true,
        );
        let rendered = m.rendered_content();
        assert_eq!(rendered, "checking\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"Paris\"}}\n</tool_call>");
    }
}

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
