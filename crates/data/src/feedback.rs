// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements feedback datasets and feedback fine-tuning of chat models for its clients. If your
// team needs expertise in turning a product's thumbs-up and thumbs-down into training data, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `generic-feedback-v1`: one judged completion per JSONL line - a prompt conversation, one assistant turn and
//! whether that turn is one to learn from (`desirable`) or to unlearn. No second completion is needed: the trainer
//! (KTO) weighs each completion against the policy's divergence from its reference.
//!
//! ```json
//! {"prompt": [{"role": "user", "content": "hi"}],
//!  "completion": {"role": "assistant", "content": "hello"},
//!  "desirable": true,
//!  "tools": [],
//!  "metadata": {"source": "thumbs"}}
//! ```
//!
//! Messages take the `generic-preference-v1` shape ([`crate::preference`]): `generic-messages-v2` without `train`,
//! the prompt never supervised and the completion always. `tools` and `metadata` are optional as there. Parsing is
//! strict: an unknown or mistyped field, a missing `desirable`, an empty prompt or one that ends in an assistant turn,
//! and a completion that is not an assistant turn are refused by line and field.

use std::io;
use std::path::Path;

use crate::chat::{messages_from_wire, ChatSample, RenderOpts, WireMessage, WireRole};
use crate::chat_template::{ChatTemplate, TemplateError};
use crate::preference::{EncodedTurn, WireTurn};
use crate::qwen_tokenizer::QwenBpe;

/// One judged completion: the prompt followed by the completion, as a [`ChatSample`] whose only supervised message
/// is that final turn, and the judgement.
#[derive(Clone, Debug)]
pub struct FeedbackSample {
    pub sample: ChatSample,
    pub desirable: bool,
}

impl FeedbackSample {
    /// Messages before the completion.
    pub fn prompt_len(&self) -> usize {
        self.sample.messages.len() - 1
    }

    /// This example's prompt followed by `other`'s completion: the mismatched pair the trainer estimates the policy's
    /// divergence from its reference on (never trained, only scored).
    pub fn with_completion_of(&self, other: &FeedbackSample) -> ChatSample {
        let mut messages = self.sample.messages[..self.prompt_len()].to_vec();
        messages.extend(other.sample.messages.last().cloned());
        ChatSample { messages, tools: self.sample.tools.clone(), rendered: None, media: Vec::new(), rollout: None }
    }

    /// Render the example through the checkpoint's own chat template, exactly as [`ChatSample::encode_with`] renders a
    /// training record.
    pub fn encode_with(&self, tok: &QwenBpe, tmpl: &ChatTemplate, opts: RenderOpts) -> Result<EncodedTurn, TemplateError> {
        let (ids, mask) = self.sample.encode_with(tok, tmpl, opts)?;
        Ok(EncodedTurn { ids, mask })
    }

    /// Parse a `generic-feedback-v1` JSONL file (see the module doc). Errors name the file, the line and the field.
    pub fn from_jsonl(path: &Path) -> io::Result<Vec<FeedbackSample>> {
        let text = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            out.push(Self::from_json_line(line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}:{}: {e}", path.display(), lineno + 1)))?);
        }
        Ok(out)
    }

    /// One example of a dataset file, as [`Self::from_jsonl`] reads each line of it.
    pub fn from_json_line(line: &str) -> Result<FeedbackSample, String> {
        let record: WireFeedback = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let desirable = record.desirable.as_bool().ok_or_else(|| format!("\"desirable\" must be true or false, got {}", record.desirable))?;
        let Some(last) = record.prompt.last() else {
            return Err("\"prompt\" is empty".to_string());
        };
        if matches!(last.role, WireRole::Assistant) {
            return Err("\"prompt\" ends in an assistant turn; the judged assistant turn is \"completion\"".to_string());
        }
        if !matches!(record.completion.role, WireRole::Assistant) {
            return Err("\"completion\" must be an assistant turn (\"role\": \"assistant\")".to_string());
        }
        if record.completion.tool_call_id.is_some() {
            return Err("\"completion\": \"tool_call_id\" belongs on a tool result, not on an assistant turn".to_string());
        }
        let prompt_len = record.prompt.len();
        let mut wire: Vec<WireMessage> = record.prompt.into_iter().map(|t| t.into_message(false)).collect();
        wire.push(record.completion.into_message(true));
        let messages = messages_from_wire(wire).map_err(|e| format!("\"prompt\" + \"completion\" (messages[{prompt_len}] is \"completion\"): {e}"))?;
        Ok(FeedbackSample { sample: ChatSample { messages, tools: record.tools, rendered: None, media: Vec::new(), rollout: None }, desirable })
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFeedback {
    prompt: Vec<WireTurn>,
    completion: WireTurn,
    /// Read as any JSON value so a mistyped one is refused naming the field (serde's own message does not).
    desirable: serde_json::Value,
    /// `minijinja::Value` for the reason [`ChatSample::from_jsonl`] gives: the schema's key order must survive.
    #[serde(default)]
    tools: Vec<minijinja::Value>,
    /// Carried for the producer; typed as an object so a mistyped one is still refused.
    #[allow(dead_code)]
    #[serde(default)]
    metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{"prompt":[{"role":"system","content":"be kind"},{"role":"user","content":"hi"}],"completion":{"role":"assistant","content":"hello"},"desirable":true,"metadata":{"source":"thumbs"}}"#;

    #[test]
    fn an_example_is_one_conversation_supervised_only_on_its_completion() {
        let e = FeedbackSample::from_json_line(GOOD).expect("parses");
        assert!(e.desirable);
        assert_eq!(e.prompt_len(), 2);
        assert_eq!(e.sample.messages.iter().map(|m| m.train).collect::<Vec<_>>(), [false, false, true]);
        let other = FeedbackSample::from_json_line(&GOOD.replace("hello", "go away").replace("true", "false")).unwrap();
        let mismatched = e.with_completion_of(&other);
        assert_eq!(mismatched.messages.len(), 3);
        assert_eq!((mismatched.messages[1].content.as_str(), mismatched.messages[2].content.as_str()), ("hi", "go away"), "this prompt, the other's completion");
        assert!(mismatched.messages[2].train);
    }

    #[test]
    fn malformed_examples_are_refused_by_field() {
        for (what, line, needle) in [
            ("unknown field", GOOD.replace("\"desirable\":true", "\"desirable\":true,\"score\":1"), "score"),
            ("no judgement", GOOD.replace(",\"desirable\":true", ""), "desirable"),
            ("judgement not a bool", GOOD.replace("\"desirable\":true", "\"desirable\":1"), "desirable"),
            ("empty prompt", r#"{"prompt":[],"completion":{"role":"assistant","content":"a"},"desirable":true}"#.to_string(), "empty"),
            ("prompt ends in assistant", r#"{"prompt":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}],"completion":{"role":"assistant","content":"a"},"desirable":false}"#.to_string(), "ends in an assistant turn"),
            ("completion not assistant", r#"{"prompt":[{"role":"user","content":"x"}],"completion":{"role":"user","content":"a"},"desirable":true}"#.to_string(), "\"completion\" must be an assistant turn"),
            ("train on a message", r#"{"prompt":[{"role":"user","content":"x","train":false}],"completion":{"role":"assistant","content":"a"},"desirable":true}"#.to_string(), "train"),
        ] {
            let err = FeedbackSample::from_json_line(&line).expect_err(what);
            assert!(err.contains(needle), "{what}: {err}");
        }
    }
}
