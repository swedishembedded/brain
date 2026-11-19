// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements preference datasets and preference
// fine-tuning of chat models for its clients. If your team needs expertise in
// preference optimization data pipelines, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `generic-preference-v1`: one preference pair per JSONL line - a prompt
//! conversation and two candidates that complete it, the one to prefer
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
//! A candidate is one assistant turn, or a continuation: a non-empty array of
//! turns that starts and ends with an assistant turn, such as a tool call, its
//! result and the answer that reads it.
//!
//! ```json
//! {"prompt": [{"role": "user", "content": "weather?"}],
//!  "chosen": [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
//!              "function": {"name": "weather", "arguments": "{}"}}]},
//!             {"role": "tool", "content": "18C", "tool_call_id": "c1"},
//!             {"role": "assistant", "content": "18C"}],
//!  "rejected": {"role": "assistant", "content": "no idea"}}
//! ```
//!
//! Messages have the `generic-messages-v2` shape ([`crate::chat`]) without
//! `train`: which tokens are supervised follows from where a message is. The
//! prompt is never supervised; every assistant turn of `chosen` and `rejected`
//! is, and every other turn of a continuation is context. Tool calls (in a
//! candidate or an earlier assistant turn of the prompt) and tool results take
//! exactly the `generic-messages-v2` form, and get the same semantic checks
//! over the prompt and the candidate together. `tools` (optional) is the tool schema array the chat
//! template's preamble renders; `metadata` (optional) is an object carried for
//! the producer and never read.
//!
//! Parsing is strict: an unknown or mistyped field, a prompt that is empty or
//! already ends in an assistant turn, a candidate that is not an assistant
//! turn, an empty continuation or one that starts or ends with another turn,
//! and a pair whose two candidates are the same turn or continuation are all
//! refused by line and field.

use std::io;
use std::path::Path;

use crate::chat::{messages_from_wire, ChatMessage, ChatSample, RenderOpts, WireMessage, WireRole, WireToolCall};
use crate::chat_template::{ChatTemplate, TemplateError};
use crate::qwen_tokenizer::QwenBpe;

/// One preference pair: the same prompt followed by the preferred and by the
/// dispreferred candidate, each as a [`ChatSample`] whose supervised messages
/// are the candidate's assistant turns.
#[derive(Clone, Debug)]
pub struct PreferenceSample {
    pub chosen: ChatSample,
    pub rejected: ChatSample,
    /// Messages before the candidate, the same in both samples.
    pub prompt_len: usize,
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
    /// Render both candidates through the checkpoint's own chat template,
    /// exactly as [`ChatSample::encode`] renders a training record.
    pub fn encode(&self, tok: &QwenBpe, tmpl: &ChatTemplate) -> Result<(EncodedTurn, EncodedTurn), TemplateError> {
        self.encode_with(tok, tmpl, RenderOpts::default())
    }

    /// [`Self::encode`] with `opts`, as [`ChatSample::encode_with`] renders a
    /// training record: with `keep_reasoning`, a single-answer candidate's
    /// closed think block stays in its rendered turn instead of being dropped
    /// by a template that strips reasoning from history. A continuation trains
    /// each answer as asked ([`ChatSample::encode_candidate`]).
    pub fn encode_with(&self, tok: &QwenBpe, tmpl: &ChatTemplate, opts: RenderOpts) -> Result<(EncodedTurn, EncodedTurn), TemplateError> {
        let (ids, mask) = self.chosen.encode_candidate(tok, tmpl, opts)?;
        let chosen = EncodedTurn { ids, mask };
        let (ids, mask) = self.rejected.encode_candidate(tok, tmpl, opts)?;
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
            out.push(Self::from_json_line(line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}:{}: {e}", path.display(), lineno + 1)))?);
        }
        Ok(out)
    }

    /// One pair of a dataset file, as [`Self::from_jsonl`] reads each line of it.
    pub fn from_json_line(line: &str) -> Result<PreferenceSample, String> {
        let record: WirePreference = serde_json::from_str(line).map_err(|e| e.to_string())?;
        pair_from_wire(record)
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePreference {
    prompt: Vec<WireTurn>,
    /// A [`Candidate`]: read as any JSON value so a refusal names the field.
    chosen: serde_json::Value,
    rejected: serde_json::Value,
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
pub(crate) struct WireTurn {
    pub(crate) role: WireRole,
    content: String,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
    #[serde(default)]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

impl WireTurn {
    pub(crate) fn into_message(self, train: bool) -> WireMessage {
        WireMessage { role: self.role, content: self.content.into(), tool_calls: self.tool_calls, tool_call_id: self.tool_call_id, reasoning_content: self.reasoning_content, train, behavior_logprobs: None, token_ids: None, prompt_token_ids: None }
    }
}

/// A candidate (or a judged completion) as the wire carries it: one assistant
/// turn, or a continuation of the prompt whose first and last turns are
/// assistant turns.
pub(crate) struct Candidate {
    turns: Vec<WireTurn>,
    continuation: bool,
}

impl Candidate {
    /// Read `field`'s value, refusing by `field` a shape that is not a
    /// candidate.
    pub(crate) fn from_wire(field: &str, value: serde_json::Value) -> Result<Candidate, String> {
        let continuation = value.is_array();
        let turns = if continuation { serde_json::from_value(value) } else { serde_json::from_value(value).map(|t| vec![t]) }.map_err(|e| format!("\"{field}\": {e}"))?;
        let candidate = Candidate { turns, continuation };
        let (Some(first), Some(last)) = (candidate.turns.first(), candidate.turns.last()) else {
            return Err(format!("\"{field}\" is an empty continuation; it holds at least one assistant turn"));
        };
        if !matches!(first.role, WireRole::Assistant) {
            let at = if continuation { "[0]" } else { "" };
            return Err(format!("\"{field}\"{at} must be an assistant turn (\"role\": \"assistant\")"));
        }
        if !matches!(last.role, WireRole::Assistant) {
            return Err(format!("\"{field}\" ends in a non-assistant turn; a continuation ends with the assistant turn that follows its last tool result or user turn"));
        }
        for (i, turn) in candidate.turns.iter().enumerate() {
            if matches!(turn.role, WireRole::Assistant) && turn.tool_call_id.is_some() {
                let at = if continuation { format!("[{i}]") } else { String::new() };
                return Err(format!("\"{field}\"{at}: \"tool_call_id\" belongs on a tool result, not on an assistant turn"));
            }
        }
        Ok(candidate)
    }

    /// The same turns in order: roles, contents, and tool calls by name and
    /// arguments (ids aside).
    fn same_as(&self, other: &Candidate) -> bool {
        let same_call = |a: &WireToolCall, b: &WireToolCall| a.function.name == b.function.name && a.function.arguments == b.function.arguments;
        let same_turn = |a: &WireTurn, b: &WireTurn| {
            std::mem::discriminant(&a.role) == std::mem::discriminant(&b.role) && a.content == b.content && a.tool_calls.len() == b.tool_calls.len() && a.tool_calls.iter().zip(&b.tool_calls).all(|(x, y)| same_call(x, y))
        };
        self.turns.len() == other.turns.len() && self.turns.iter().zip(&other.turns).all(|(a, b)| same_turn(a, b))
    }

    /// The prompt followed by this candidate as one conversation, validated as
    /// a `generic-messages-v2` record is: the prompt is context, the
    /// candidate's assistant turns are supervised and its other turns context.
    pub(crate) fn complete(self, field: &str, prompt: &[WireTurn]) -> Result<Vec<ChatMessage>, String> {
        let prompt_len = prompt.len();
        let at = if self.continuation { format!("messages[{prompt_len}..{}] are", prompt_len + self.turns.len()) } else { format!("messages[{prompt_len}] is") };
        let mut wire: Vec<WireMessage> = prompt.iter().cloned().map(|t| t.into_message(false)).collect();
        wire.extend(self.turns.into_iter().map(|t| {
            let supervised = matches!(t.role, WireRole::Assistant);
            t.into_message(supervised)
        }));
        messages_from_wire(wire).map_err(|e| format!("\"prompt\" + \"{field}\" ({at} \"{field}\"): {e}"))
    }
}

fn pair_from_wire(record: WirePreference) -> Result<PreferenceSample, String> {
    let Some(last) = record.prompt.last() else {
        return Err("\"prompt\" is empty".to_string());
    };
    if matches!(last.role, WireRole::Assistant) {
        return Err("\"prompt\" ends in an assistant turn; the candidates are \"chosen\" and \"rejected\"".to_string());
    }
    let chosen = Candidate::from_wire("chosen", record.chosen)?;
    let rejected = Candidate::from_wire("rejected", record.rejected)?;
    if chosen.same_as(&rejected) {
        let what = if chosen.turns.len() == 1 { "turn" } else { "continuation" };
        return Err(format!("\"chosen\" and \"rejected\" are the same {what}, so the pair states no preference"));
    }
    // Each candidate is validated as the conversation it completes, so a tool
    // call in the prompt or the candidate and the tool result answering it get
    // exactly the checks a `generic-messages-v2` record gets.
    let sample = |messages| ChatSample { messages, tools: record.tools.clone(), rendered: None, media: Vec::new(), rollout: None };
    let chosen = sample(chosen.complete("chosen", &record.prompt)?);
    let rejected = sample(rejected.complete("rejected", &record.prompt)?);
    Ok(PreferenceSample { chosen, rejected, prompt_len: record.prompt.len() })
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
        assert_eq!(p.prompt_len, 2);
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
        let again = call.replace("c1", "c2");
        let ok = format!(
            r#"{{"prompt":[{{"role":"user","content":"weather?"}},{{"role":"assistant","content":"","tool_calls":[{call}]}},{{"role":"tool","content":"18C","tool_call_id":"c1"}}],"chosen":{{"role":"assistant","content":"18C and sunny"}},"rejected":{{"role":"assistant","content":"","tool_calls":[{again}]}},"tools":[{{"type":"function"}}]}}"#
        );
        let pairs = parse("tools.jsonl", &ok).expect("parses");
        assert_eq!(pairs[0].rejected.messages[3].tool_calls[0].name, "get_weather");
        assert_eq!(pairs[0].chosen.tools.len(), 1);

        let orphan = r#"{"prompt":[{"role":"user","content":"x"},{"role":"tool","content":"18C","tool_call_id":"nope"}],"chosen":{"role":"assistant","content":"a"},"rejected":{"role":"assistant","content":"b"}}"#;
        let err = parse("orphan.jsonl", orphan).unwrap_err().to_string();
        assert!(err.contains("does not match any tool_calls id") && err.contains(":1:"), "{err}");

        // A candidate that re-uses the prompt's call id breaks the pairing
        // rule: the conversation it completes names one id twice.
        let reused = ok.replace("c2", "c1");
        let err = parse("reused.jsonl", &reused).unwrap_err().to_string();
        assert!(err.contains("c1") && err.contains("more than once") && err.contains(":1:"), "{err}");
    }

    const CALL: &str = r#"{"id":"c1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}"#;

    /// A continuation candidate: a tool call, its result and the answer that
    /// reads it, supervised on its assistant turns only.
    #[test]
    fn a_continuation_supervises_its_assistant_turns_and_reads_its_tool_results_as_context() {
        let line = format!(
            r#"{{"prompt":[{{"role":"system","content":"s"}},{{"role":"user","content":"weather?"}}],"chosen":[{{"role":"assistant","content":"","tool_calls":[{CALL}]}},{{"role":"tool","content":"18C","tool_call_id":"c1"}},{{"role":"assistant","content":"18C"}}],"rejected":{{"role":"assistant","content":"no idea"}}}}"#
        );
        let p = PreferenceSample::from_json_line(&line).expect("parses");
        assert_eq!(p.prompt_len, 2);
        assert_eq!(p.chosen.messages.iter().map(|m| m.train).collect::<Vec<_>>(), [false, false, true, false, true]);
        assert_eq!(p.chosen.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(), ["system", "user", "assistant", "tool", "assistant"]);
        assert_eq!(p.rejected.messages.iter().map(|m| m.train).collect::<Vec<_>>(), [false, false, true]);

        // The pairing rule holds over the prompt and the continuation: a result
        // answering no call of either is refused.
        let orphan = line.replace(r#""tool_call_id":"c1""#, r#""tool_call_id":"c9""#);
        let err = PreferenceSample::from_json_line(&orphan).unwrap_err();
        assert!(err.contains("chosen") && err.contains("does not match any tool_calls id"), "{err}");
    }

    #[test]
    fn malformed_continuations_are_refused_naming_the_field() {
        let open = format!(r#"[{{"role":"assistant","content":"","tool_calls":[{CALL}]}},{{"role":"tool","content":"18C","tool_call_id":"c1"}}]"#);
        let full = format!(r#"[{{"role":"assistant","content":"","tool_calls":[{CALL}]}},{{"role":"tool","content":"18C","tool_call_id":"c1"}},{{"role":"assistant","content":"18C"}}]"#);
        let pair = |chosen: &str, rejected: &str| format!(r#"{{"prompt":[{{"role":"user","content":"x"}}],"chosen":{chosen},"rejected":{rejected}}}"#);
        let single = r#"{"role":"assistant","content":"b"}"#;
        for (what, line, needles) in [
            ("ends in a tool turn", pair(single, &open), &["\"rejected\"", "ends in a non-assistant turn"][..]),
            ("empty", pair("[]", single), &["\"chosen\"", "empty continuation"][..]),
            ("starts with a tool turn", pair(r#"[{"role":"tool","content":"t","tool_call_id":"c1"},{"role":"assistant","content":"a"}]"#, single), &["\"chosen\"[0]", "must be an assistant turn"][..]),
            ("tool_call_id on an assistant turn", pair(r#"[{"role":"assistant","content":"a"},{"role":"user","content":"u"},{"role":"assistant","content":"b","tool_call_id":"c1"}]"#, single), &["\"chosen\"[2]", "tool_call_id"][..]),
            ("the same continuation", pair(&full, &full.replace("c1", "c2")), &["same continuation"][..]),
            ("an unknown field in a turn", pair(&full.replace(r#""content":"18C"}]"#, r#""content":"18C","train":true}]"#), single), &["\"chosen\"", "train"][..]),
        ] {
            let err = PreferenceSample::from_json_line(&line).expect_err(what);
            for needle in needles {
                assert!(err.contains(needle), "{what}: the error must name {needle:?}: {err}");
            }
        }
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
