// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements feedback datasets and feedback fine-tuning of chat models for its clients. If your
// team needs expertise in turning a product's thumbs-up and thumbs-down into training data, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `generic-feedback-v1`: one judged completion per JSONL line - a prompt conversation, the completion that follows
//! it and whether that completion is one to learn from (`desirable`) or to unlearn. No second completion is needed: the trainer
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
//! `completion` is a `generic-preference-v1` candidate ([`crate::preference`]): one assistant turn, or a continuation
//! (a non-empty array of turns that starts and ends with an assistant turn, such as a tool call, its result and the
//! answer). Messages are `generic-messages-v2` without `train`: the prompt is never supervised, the completion's
//! assistant turns always are and its other turns are context. `tools` and `metadata` are optional as there. Parsing
//! is strict: an unknown or mistyped field, a missing `desirable`, an empty prompt or one that ends in an assistant
//! turn, and a completion that is not an assistant turn or a continuation of one are refused by line and field.

use std::io;
use std::path::Path;

use crate::chat::{ChatSample, RenderOpts, WireRole};
use crate::chat_template::{ChatTemplate, TemplateError};
use crate::preference::{Candidate, EncodedTurn, WireTurn};
use crate::qwen_tokenizer::QwenBpe;

/// One judged completion: the prompt followed by the completion, as a [`ChatSample`] whose supervised messages are
/// the completion's assistant turns, and the judgement.
#[derive(Clone, Debug)]
pub struct FeedbackSample {
    pub sample: ChatSample,
    pub desirable: bool,
    /// Messages before the completion.
    pub prompt_len: usize,
}

impl FeedbackSample {
    /// This example's prompt followed by `other`'s whole completion: the mismatched pair the trainer estimates the
    /// policy's divergence from its reference on (never trained, only scored).
    pub fn with_completion_of(&self, other: &FeedbackSample) -> ChatSample {
        let mut messages = self.sample.messages[..self.prompt_len].to_vec();
        messages.extend_from_slice(&other.sample.messages[other.prompt_len..]);
        ChatSample { messages, tools: self.sample.tools.clone(), rendered: None, media: Vec::new(), rollout: None }
    }

    /// Render the example through the checkpoint's own chat template as a preference candidate is rendered
    /// ([`ChatSample::encode_candidate`]): a continuation trains each answer as asked.
    pub fn encode_with(&self, tok: &QwenBpe, tmpl: &ChatTemplate, opts: RenderOpts) -> Result<EncodedTurn, TemplateError> {
        let (ids, mask) = self.sample.encode_candidate(tok, tmpl, opts)?;
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
            return Err("\"prompt\" ends in an assistant turn; the judged completion is \"completion\"".to_string());
        }
        let messages = Candidate::from_wire("completion", record.completion)?.complete("completion", &record.prompt)?;
        let prompt_len = record.prompt.len();
        Ok(FeedbackSample { sample: ChatSample { messages, tools: record.tools, rendered: None, media: Vec::new(), rollout: None }, desirable, prompt_len })
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFeedback {
    prompt: Vec<WireTurn>,
    /// A [`Candidate`]: read as any JSON value so a refusal names the field.
    completion: serde_json::Value,
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
        assert_eq!(e.prompt_len, 2);
        assert_eq!(e.sample.messages.iter().map(|m| m.train).collect::<Vec<_>>(), [false, false, true]);
        let other = FeedbackSample::from_json_line(&GOOD.replace("hello", "go away").replace("true", "false")).unwrap();
        let mismatched = e.with_completion_of(&other);
        assert_eq!(mismatched.messages.len(), 3);
        assert_eq!((mismatched.messages[1].content.as_str(), mismatched.messages[2].content.as_str()), ("hi", "go away"), "this prompt, the other's completion");
        assert!(mismatched.messages[2].train);
    }

    /// A continuation completion is supervised on its assistant turns, and a mismatched pair carries the other
    /// example's whole continuation after this one's prompt.
    #[test]
    fn a_continuation_completion_is_carried_whole_into_a_mismatched_pair() {
        let call = r#"{"id":"c1","type":"function","function":{"name":"weather","arguments":"{}"}}"#;
        let line = format!(
            r#"{{"prompt":[{{"role":"user","content":"weather?"}}],"completion":[{{"role":"assistant","content":"","tool_calls":[{call}]}},{{"role":"tool","content":"18C","tool_call_id":"c1"}},{{"role":"assistant","content":"18C"}}],"desirable":true}}"#
        );
        let tooled = FeedbackSample::from_json_line(&line).expect("parses");
        assert_eq!(tooled.prompt_len, 1);
        assert_eq!(tooled.sample.messages.iter().map(|m| m.train).collect::<Vec<_>>(), [false, true, false, true]);

        let plain = FeedbackSample::from_json_line(GOOD).unwrap();
        let mismatched = plain.with_completion_of(&tooled);
        assert_eq!(mismatched.messages.iter().map(|m| (m.role.as_str(), m.train)).collect::<Vec<_>>(), [("system", false), ("user", false), ("assistant", true), ("tool", false), ("assistant", true)]);
        assert_eq!((mismatched.messages[1].content.as_str(), mismatched.messages[4].content.as_str()), ("hi", "18C"));
        let back = tooled.with_completion_of(&plain);
        assert_eq!(back.messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(), ["weather?", "hello"]);

        let open = line.replace(r#",{"role":"assistant","content":"18C"}]"#, "]");
        let err = FeedbackSample::from_json_line(&open).unwrap_err();
        assert!(err.contains("\"completion\"") && err.contains("ends in a non-assistant turn"), "{err}");
        let err = FeedbackSample::from_json_line(r#"{"prompt":[{"role":"user","content":"x"}],"completion":[],"desirable":true}"#).unwrap_err();
        assert!(err.contains("\"completion\" is an empty continuation"), "{err}");
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
