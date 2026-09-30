// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements preference datasets and preference
// fine-tuning of chat models for its clients. If your team needs expertise in
// preference optimization data pipelines, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `generic-preference-v1`: one preference pair per JSONL line - a prompt
//! conversation and two candidate assistant turns, the one to prefer
//! (`chosen`) and the one to prefer it over (`rejected`).
//!
//! ```json
//! {"prompt": [{"role": "user", "content": "hi"}],
//!  "chosen": {"role": "assistant", "content": "hello"},
//!  "rejected": {"role": "assistant", "content": "go away"},
//!  "tools": [],
//!  "metadata": {"source": "review-queue"}}
//! ```
//!
//! Messages have the `generic-messages-v2` shape ([`crate::chat`]) without
//! `train`: which tokens are supervised follows from where a message is. The
//! prompt is never supervised; `chosen` and `rejected` always are. Tool calls
//! (in `chosen`, `rejected` or an earlier assistant turn of the prompt) and
//! tool results take exactly the `generic-messages-v2` form, and get the same
//! semantic checks. `tools` (optional) is the tool schema array the chat
//! template's preamble renders; `metadata` (optional) is an object carried for
//! the producer and never read.
//!
//! Parsing is strict: an unknown or mistyped field, a prompt that is empty or
//! already ends in an assistant turn, a candidate that is not an assistant
//! turn, and a pair whose two candidates are the same turn are all refused by
//! line and field.

use std::io;
use std::path::Path;

use crate::chat::{messages_from_wire, ChatSample, WireMessage, WireRole, WireToolCall};
use crate::chat_template::{ChatTemplate, TemplateError};
use crate::qwen_tokenizer::QwenBpe;

/// One preference pair: the same prompt followed by the preferred and by the
/// dispreferred assistant turn, each as a packed [`ChatSample`] whose only
/// supervised message is that final turn.
#[derive(Clone, Debug)]
pub struct PreferenceSample {
    pub chosen: ChatSample,
    pub rejected: ChatSample,
}

/// One candidate rendered for training: token ids and the per-token
/// supervision mask ([`ChatSample::encode`]'s output).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTurn {
    pub ids: Vec<u32>,
    pub mask: Vec<bool>,
}

impl EncodedTurn {
    /// The supervised tokens, in order.
    pub fn supervised(&self) -> Vec<u32> {
        self.ids.iter().zip(&self.mask).filter(|(_, m)| **m).map(|(t, _)| *t).collect()
    }
}

impl PreferenceSample {
    /// Messages before the candidate turn.
    pub fn prompt_len(&self) -> usize {
        self.chosen.messages.len() - 1
    }

    /// Render both candidates through the checkpoint's own chat template,
    /// exactly as [`ChatSample::encode`] renders a training record.
    pub fn encode(&self, tok: &QwenBpe, tmpl: &ChatTemplate) -> Result<(EncodedTurn, EncodedTurn), TemplateError> {
        let (ids, mask) = self.chosen.encode(tok, tmpl)?;
        let chosen = EncodedTurn { ids, mask };
        let (ids, mask) = self.rejected.encode(tok, tmpl)?;
        Ok((chosen, EncodedTurn { ids, mask }))
    }

    /// Parse a `generic-preference-v1` JSONL file (see the module doc).
    /// Errors name the file, the line and the offending field.
    pub fn from_jsonl(path: &Path) -> io::Result<Vec<PreferenceSample>> {
        let text = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidData, format!("{}:{}: {e}", path.display(), lineno + 1));
            let record: WirePreference = serde_json::from_str(line).map_err(|e| invalid(e.to_string()))?;
            out.push(pair_from_wire(record).map_err(invalid)?);
        }
        Ok(out)
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePreference {
    prompt: Vec<WireTurn>,
    chosen: WireTurn,
    rejected: WireTurn,
    /// `minijinja::Value`, not `serde_json::Value`, for the reason
    /// [`ChatSample::from_jsonl`] gives: the schema's key order must survive.
    #[serde(default)]
    tools: Vec<minijinja::Value>,
    /// Carried for the producer; typed as an object so a mistyped one is
    /// still refused.
    #[allow(dead_code)]
    #[serde(default)]
    metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

/// A `generic-messages-v2` message without `train`.
#[derive(Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTurn {
    role: WireRole,
    content: String,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

impl WireTurn {
    fn into_message(self, train: bool) -> WireMessage {
        WireMessage { role: self.role, content: self.content, tool_calls: self.tool_calls, tool_call_id: self.tool_call_id, train }
    }
}

fn pair_from_wire(record: WirePreference) -> Result<PreferenceSample, String> {
    let Some(last) = record.prompt.last() else {
        return Err("\"prompt\" is empty".to_string());
    };
    if matches!(last.role, WireRole::Assistant) {
        return Err("\"prompt\" ends in an assistant turn; the candidate assistant turns are \"chosen\" and \"rejected\"".to_string());
    }
    for (field, turn) in [("chosen", &record.chosen), ("rejected", &record.rejected)] {
        if !matches!(turn.role, WireRole::Assistant) {
            return Err(format!("\"{field}\" must be an assistant turn (\"role\": \"assistant\")"));
        }
        if turn.tool_call_id.is_some() {
            return Err(format!("\"{field}\": \"tool_call_id\" belongs on a tool result, not on an assistant turn"));
        }
    }
    let same_call = |a: &WireToolCall, b: &WireToolCall| a.function.name == b.function.name && a.function.arguments == b.function.arguments;
    if record.chosen.content == record.rejected.content
        && record.chosen.tool_calls.len() == record.rejected.tool_calls.len()
        && record.chosen.tool_calls.iter().zip(&record.rejected.tool_calls).all(|(a, b)| same_call(a, b))
    {
        return Err("\"chosen\" and \"rejected\" are the same turn, so the pair states no preference".to_string());
    }

    let prompt_len = record.prompt.len();
    // Each candidate is validated as the conversation it completes, so a tool
    // call in the prompt, a tool result answering it and the candidate turn
    // get exactly the checks a `generic-messages-v2` record gets.
    let complete = |field: &str, prompt: &[WireTurn], turn: WireTurn| -> Result<ChatSample, String> {
        let mut wire: Vec<WireMessage> = prompt.iter().cloned().map(|t| t.into_message(false)).collect();
        wire.push(turn.into_message(true));
        let messages = messages_from_wire(wire).map_err(|e| format!("\"prompt\" + \"{field}\" (messages[{prompt_len}] is \"{field}\"): {e}"))?;
        Ok(ChatSample { messages, tools: record.tools.clone() })
    };
    let chosen = complete("chosen", &record.prompt, record.chosen)?;
    let rejected = complete("rejected", &record.prompt, record.rejected)?;
    Ok(PreferenceSample { chosen, rejected })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-preference-jsonl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    fn parse(name: &str, body: &str) -> io::Result<Vec<PreferenceSample>> {
        PreferenceSample::from_jsonl(&write(name, body))
    }

    #[test]
    fn a_pair_parses_into_two_conversations_supervised_only_on_the_candidate() {
        let pairs = parse(
            "ok.jsonl",
            r#"{"prompt":[{"role":"system","content":"be kind"},{"role":"user","content":"hi"}],"chosen":{"role":"assistant","content":"hello"},"rejected":{"role":"assistant","content":"go away"},"metadata":{"source":"review"}}"#,
        )
        .expect("parses");
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        assert_eq!(p.prompt_len(), 2);
        for (sample, answer) in [(&p.chosen, "hello"), (&p.rejected, "go away")] {
            assert_eq!(sample.messages.len(), 3);
            assert_eq!(sample.messages.iter().map(|m| m.train).collect::<Vec<_>>(), vec![false, false, true]);
            assert_eq!(sample.messages[2].role, "assistant");
            assert_eq!(sample.messages[2].content, answer);
        }
    }

    /// Tool calls take the `generic-messages-v2` shape and get its semantic
    /// checks: a result must answer a call made earlier in the conversation.
    #[test]
    fn tool_calls_take_the_chat_shape_and_its_checks() {
        let call = r#"{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}"#;
        let ok = format!(
            r#"{{"prompt":[{{"role":"user","content":"weather?"}},{{"role":"assistant","content":"","tool_calls":[{call}]}},{{"role":"tool","content":"18C","tool_call_id":"c1"}}],"chosen":{{"role":"assistant","content":"18C and sunny"}},"rejected":{{"role":"assistant","content":"","tool_calls":[{call}]}},"tools":[{{"type":"function"}}]}}"#
        );
        let pairs = parse("tools.jsonl", &ok).expect("parses");
        assert_eq!(pairs[0].rejected.messages[3].tool_calls[0].name, "get_weather");
        assert_eq!(pairs[0].chosen.tools.len(), 1);

        let orphan = r#"{"prompt":[{"role":"user","content":"x"},{"role":"tool","content":"18C","tool_call_id":"nope"}],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"b"}}"#;
        let err = parse("orphan.jsonl", orphan).unwrap_err().to_string();
        assert!(err.contains("does not match any tool_calls id") && err.contains(":1:"), "{err}");
    }

    #[test]
    fn malformed_pairs_are_refused_by_line_and_field() {
        let cases: &[(&str, &str, &str)] = &[
            ("unknown field", r#"{"prompt":[{"role":"user","content":"x"}],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"b"},"score":1}"#, "score"),
            ("train on a message", r#"{"prompt":[{"role":"user","content":"x","train":false}],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"b"}}"#, "train"),
            ("missing rejected", r#"{"prompt":[{"role":"user","content":"x"}],"chosen":{"role":"assistant","content":"a"}}"#, "rejected"),
            ("empty prompt", r#"{"prompt":[],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"b"}}"#, "empty"),
            ("prompt ends in assistant", r#"{"prompt":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"b"}}"#, "ends in an assistant turn"),
            ("candidate not assistant", r#"{"prompt":[{"role":"user","content":"x"}],"chosen":{"role":"user","content":"a"},"rejected":{"role":"assistant","content":"b"}}"#, "\"chosen\" must be an assistant turn"),
            ("identical candidates", r#"{"prompt":[{"role":"user","content":"x"}],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"a"}}"#, "same turn"),
            ("metadata not an object", r#"{"prompt":[{"role":"user","content":"x"}],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"b"},"metadata":3}"#, "expected a map"),
        ];
        for (what, line, needle) in cases {
            let body = format!("{}\n{line}\n", r#"{"prompt":[{"role":"user","content":"x"}],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"b"}}"#);
            let err = parse("bad.jsonl", &body).expect_err(what).to_string();
            assert!(err.contains(":2:"), "{what}: the error must name line 2: {err}");
            assert!(err.contains(needle), "{what}: the error must name {needle:?}: {err}");
        }
    }
}
