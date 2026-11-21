// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reading a typed-message reply: the wire form of a template whose assistant turn is a run of
//! messages, each opened by a content-kind control token and closed by an end-of-message token.
//!
//! ```text
//! [<|message_model|>NAME]<|content_thinking|>reasoning<|end_message|>
//! [<|message_model|>NAME]<|content_text|>answer<|end_message|>
//! [<|message_model|>NAME]<|content_invoke_tool_json|>{"name":"f","args":{...}}<|end_message|>
//! <|content_model_end_sampling|>
//! ```
//!
//! The reply is the decoded text of the generated ids, control tokens spelled out. The grammar is
//! the one the checkpoint's template writes for history (`ChatTemplate::reply_flavor` selects this
//! reader from the markers the template contains) and its `response_template` reads back: the
//! optional `<|message_model|>` header and its name are accepted but not required, reasoning is the
//! thinking messages joined, content the text, tool-error and tool-text messages joined, and a call
//! is a JSON object with a `name` and an `args` object.
//!
//! [`MessageScanner`] is the push parser; [`crate::qwen_chat::ChatScanner`] wraps it so one scanner
//! type serves every wire form. Chunking never changes the result: a marker split over two pushes
//! is withheld until it resolves, down to a byte per push.
//!
//! Swedish Embedded AB implements tool-calling and reasoning serving for its clients. If your team
//! needs generated replies parsed into reasoning, text and tool calls without losing a byte, you can
//! procure our services by sending an email to info@swedishembedded.com.

use serde::Deserialize;

use crate::qwen_chat::ScanAnomaly;

/// Opens an assistant message; its name (a function name, or a participant) follows up to the
/// content-kind token.
pub const MESSAGE_HEADER: &str = "<|message_model|>";
const CONTENT_OPEN: &str = "<|content_";
/// Closes one message.
pub const END_MESSAGE: &str = "<|end_message|>";
/// Closes the reply: the token sampling stops on.
pub const END_SAMPLING: &str = "<|content_model_end_sampling|>";
const THINKING: &str = "<|content_thinking|>";
const TEXT: [&str; 3] = ["<|content_text|>", "<|content_tool_error|>", "<|content_invoke_tool_text|>"];
const TOOL_JSON: &str = "<|content_invoke_tool_json|>";

/// What the scanner found, in generated order.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    /// Reasoning text, and where it starts in the generated text.
    Reasoning { text: String, at: usize },
    Content(String),
    /// A complete call; `arguments` is the JSON text of `args` as the model wrote it.
    Call { name: String, arguments: String },
    Anomaly(ScanAnomaly),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Thinking,
    Text,
    ToolJson,
}

#[derive(Debug)]
enum State {
    /// Between messages: awaiting a header or a content-kind token.
    Boundary,
    Body(Kind),
}

#[derive(Debug)]
pub struct MessageScanner {
    state: State,
    /// Text not yet resolved: a possible marker prefix, a call body, or a header being read.
    pending: String,
    /// Where `pending` starts in the generated text.
    consumed: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CallBody<'a> {
    name: String,
    #[serde(borrow)]
    args: &'a serde_json::value::RawValue,
}

impl MessageScanner {
    /// `thinking_open`: the prompt already opened a thinking message.
    pub fn new(thinking_open: bool) -> MessageScanner {
        MessageScanner { state: if thinking_open { State::Body(Kind::Thinking) } else { State::Boundary }, pending: String::new(), consumed: 0 }
    }

    pub fn push(&mut self, delta: &str, out: &mut Vec<Piece>) {
        self.pending.push_str(delta);
        while self.step(out) {}
    }

    /// End of generation: everything withheld is resolved as far as it can be, and anything that
    /// ended mid-message is reported.
    pub fn finish(&mut self, out: &mut Vec<Piece>) {
        match self.state {
            State::Boundary if !self.pending.is_empty() => {
                out.push(Piece::Anomaly(ScanAnomaly::UnframedText));
                let text = self.take(self.pending.len());
                out.push(Piece::Content(text));
            }
            State::Boundary => {}
            State::Body(Kind::Thinking) => {
                let at = self.consumed;
                let text = self.take(self.pending.len());
                if !text.is_empty() {
                    out.push(Piece::Reasoning { text, at });
                }
                out.push(Piece::Anomaly(ScanAnomaly::EndedInsideThink));
            }
            State::Body(Kind::Text) => {
                let text = self.take(self.pending.len());
                if !text.is_empty() {
                    out.push(Piece::Content(text));
                }
            }
            State::Body(Kind::ToolJson) => {
                out.push(Piece::Anomaly(ScanAnomaly::TruncatedToolCall));
                let body = self.take(self.pending.len());
                Self::emit_call(body, out, false);
            }
        }
        self.state = State::Boundary;
    }

    /// Consume `len` bytes of `pending`.
    fn take(&mut self, len: usize) -> String {
        self.consumed += len;
        self.pending.drain(..len).collect()
    }

    /// One resolution step; `false` when `pending` needs more input.
    fn step(&mut self, out: &mut Vec<Piece>) -> bool {
        match self.state {
            State::Boundary => self.step_boundary(out),
            State::Body(kind) => self.step_body(kind, out),
        }
    }

    fn step_boundary(&mut self, out: &mut Vec<Piece>) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        for marker in [END_SAMPLING, END_MESSAGE] {
            if self.pending.starts_with(marker) {
                self.take(marker.len());
                return true;
            }
            if marker.starts_with(&self.pending) {
                return false;
            }
        }
        let header = if self.pending.starts_with(MESSAGE_HEADER) {
            let after = &self.pending[MESSAGE_HEADER.len()..];
            // The name runs to the `<` of the content-kind token.
            let Some(name_end) = after.find('<') else { return false };
            MESSAGE_HEADER.len() + name_end
        } else if MESSAGE_HEADER.starts_with(&self.pending) {
            return false;
        } else {
            0
        };
        let rest = &self.pending[header..];
        if !rest.starts_with(CONTENT_OPEN) {
            if CONTENT_OPEN.starts_with(rest) {
                return false;
            }
            return self.unframed(out);
        }
        let Some(close) = rest.find("|>") else { return false };
        let token_len = close + 2;
        let token = &rest[..token_len];
        let kind = if token == THINKING {
            Kind::Thinking
        } else if TEXT.contains(&token) {
            Kind::Text
        } else if token == TOOL_JSON {
            Kind::ToolJson
        } else {
            // A content kind the reply grammar does not read (an image, XML): kept as content.
            out.push(Piece::Anomaly(ScanAnomaly::UnexpectedContentKind));
            Kind::Text
        };
        self.take(header + token_len);
        self.state = State::Body(kind);
        true
    }

    /// Text that is not a message: kept as content, flagged.
    fn unframed(&mut self, out: &mut Vec<Piece>) -> bool {
        out.push(Piece::Anomaly(ScanAnomaly::UnframedText));
        self.state = State::Body(Kind::Text);
        true
    }

    fn step_body(&mut self, kind: Kind, out: &mut Vec<Piece>) -> bool {
        let end = [END_MESSAGE, END_SAMPLING].iter().filter_map(|m| self.pending.find(m).map(|at| (at, *m))).min_by_key(|(at, _)| *at);
        let Some((at, marker)) = end else {
            // No marker yet: a call is buffered whole, other text streams except a possible marker prefix.
            if kind != Kind::ToolJson {
                let safe = self.pending.len() - marker_prefix_len(&self.pending);
                if safe > 0 {
                    self.deliver(kind, safe, out);
                }
            }
            return false;
        };
        self.deliver(kind, at, out);
        self.take(marker.len());
        self.state = State::Boundary;
        true
    }

    /// The first `len` bytes of `pending` as a piece of `kind`; a call body is parsed whole.
    fn deliver(&mut self, kind: Kind, len: usize, out: &mut Vec<Piece>) {
        let start = self.consumed;
        let text = self.take(len);
        match kind {
            Kind::Thinking if !text.is_empty() => out.push(Piece::Reasoning { text, at: start }),
            Kind::Text if !text.is_empty() => out.push(Piece::Content(text)),
            Kind::ToolJson => Self::emit_call(text, out, true),
            _ => {}
        }
    }

    fn emit_call(body: String, out: &mut Vec<Piece>, complete: bool) {
        match serde_json::from_str::<CallBody>(&body) {
            Ok(call) if !call.name.is_empty() && call.args.get().starts_with('{') => {
                out.push(Piece::Call { name: call.name, arguments: call.args.get().to_string() });
            }
            _ => {
                // Not a call: the bytes stay visible, as an unreadable tool-call payload is.
                if complete {
                    out.push(Piece::Anomaly(ScanAnomaly::InvalidToolCallHeader));
                }
                out.push(Piece::Content(body));
            }
        }
    }
}

/// Whether a template writes its replies in this form: it has the assistant header and a reasoning or
/// JSON tool-call content kind.
pub fn is_written_by(template_source: &str) -> bool {
    template_source.contains(MESSAGE_HEADER) && (template_source.contains(THINKING) || template_source.contains(TOOL_JSON))
}

/// The length of the longest suffix of `text` that is a proper prefix of an end marker.
fn marker_prefix_len(text: &str) -> usize {
    let longest = END_SAMPLING.len().max(END_MESSAGE.len()) - 1;
    let from = text.len().saturating_sub(longest);
    text.char_indices()
        .map(|(at, _)| at)
        .filter(|at| *at >= from)
        .find(|&at| [END_MESSAGE, END_SAMPLING].iter().any(|m| m.len() > text.len() - at && m.starts_with(&text[at..])))
        .map_or(0, |at| text.len() - at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_chat::{ChatEvent, ChatScanner, TemplateFlavor};

    const REPLY: &str = concat!(
        "<|message_model|><|content_thinking|>Need the tool.<|end_message|>",
        "<|message_model|><|content_text|>Let me check.<|end_message|>",
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"units\":\"c\",\"city\":\"Malmö\"}}<|end_message|>",
        "<|message_model|>no_args<|content_invoke_tool_json|>{\"name\":\"no_args\",\"args\":{}}<|end_message|>",
        "<|content_model_end_sampling|>",
    );

    type Report = (String, String, Vec<(String, String)>, Vec<ScanAnomaly>, Vec<ChatEvent>);

    fn scan(text: &str, chunk: usize) -> Report {
        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::TypedMessages);
        let mut events = Vec::new();
        let bytes = text.as_bytes();
        let mut at = 0;
        while at < bytes.len() {
            // Chunk on char boundaries: a token is a whole string, never half a character.
            let mut end = (at + chunk).min(bytes.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            scanner.push(&text[at..end], &mut events);
            at = end;
        }
        scanner.finish(&mut events);
        let calls = scanner.tool_calls().iter().map(|c| (c.name.clone(), c.arguments.clone())).collect();
        (scanner.content().to_string(), scanner.reasoning().to_string(), calls, scanner.anomalies().to_vec(), coalesced(events))
    }

    /// The events with adjacent text events joined and call ids blanked: what chunking must not change.
    fn coalesced(events: Vec<ChatEvent>) -> Vec<ChatEvent> {
        let mut out: Vec<ChatEvent> = Vec::new();
        for event in events {
            match (out.last_mut(), event) {
                (Some(ChatEvent::Reasoning(a)), ChatEvent::Reasoning(b)) | (Some(ChatEvent::Content(a)), ChatEvent::Content(b)) => a.push_str(&b),
                (_, ChatEvent::ToolCallStart { index, name, .. }) => out.push(ChatEvent::ToolCallStart { index, id: String::new(), name }),
                (_, event) => out.push(event),
            }
        }
        out
    }

    /// Reasoning, text and calls come out of a reply as the template wrote them, and no chunking of
    /// the stream, down to a byte per push, changes the events.
    #[test]
    fn a_reply_reads_as_reasoning_text_and_calls_however_it_is_chunked() {
        let whole = scan(REPLY, REPLY.len());
        assert_eq!(whole.0, "Let me check.");
        assert_eq!(whole.1, "Need the tool.");
        assert_eq!(whole.2, vec![("get_weather".to_string(), r#"{"units":"c","city":"Malmö"}"#.to_string()), ("no_args".to_string(), "{}".to_string())]);
        assert_eq!(whole.3, vec![]);
        for chunk in [1, 2, 3, 7, 16, 40] {
            assert_eq!(scan(REPLY, chunk), whole, "chunk {chunk}");
        }
    }

    #[test]
    fn the_header_is_optional_and_a_name_before_the_kind_is_accepted() {
        let bare = scan("<|content_thinking|>t<|end_message|><|content_text|>a<|end_message|>", 5);
        assert_eq!((bare.0.as_str(), bare.1.as_str(), bare.3.is_empty()), ("a", "t", true));
        let named = scan("<|message_model|>analyst<|content_text|>a<|end_message|><|content_model_end_sampling|>", 1);
        assert_eq!((named.0.as_str(), named.3.is_empty()), ("a", true));
    }

    #[test]
    fn messages_of_a_kind_join_and_tool_errors_read_as_content() {
        let r = scan("<|content_text|>a<|end_message|><|content_tool_error|>b<|end_message|><|content_text|>c<|end_message|>", 4);
        assert_eq!(r.0, "abc");
    }

    #[test]
    fn what_is_not_a_clean_reply_is_kept_and_flagged() {
        let unframed = scan("just talking<|end_message|>", 3);
        assert_eq!((unframed.0.as_str(), unframed.3.clone()), ("just talking", vec![ScanAnomaly::UnframedText]));
        let image = scan("<|content_image|>x<|end_message|>", 3);
        assert_eq!((image.0.as_str(), image.3.clone()), ("x", vec![ScanAnomaly::UnexpectedContentKind]));
        let bad_call = scan("<|content_invoke_tool_json|>{\"name\":\"f\"}<|end_message|>", 6);
        assert_eq!((bad_call.0.as_str(), bad_call.2.len(), bad_call.3.clone()), ("{\"name\":\"f\"}", 0, vec![ScanAnomaly::InvalidToolCallHeader]));
        let cut_think = scan("<|content_thinking|>half a thou", 4);
        assert_eq!((cut_think.1.as_str(), cut_think.3.clone()), ("half a thou", vec![ScanAnomaly::EndedInsideThink]));
        let cut_call = scan("<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1", 5);
        assert_eq!((cut_call.2.len(), cut_call.3.clone()), (0, vec![ScanAnomaly::TruncatedToolCall]));
        assert_eq!(cut_call.0, "{\"name\":\"f\",\"args\":{\"a\":1", "the cut payload stays visible");
        let whole_but_unclosed = scan("<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}", 5);
        assert_eq!((whole_but_unclosed.2.len(), whole_but_unclosed.3.clone()), (1, vec![ScanAnomaly::TruncatedToolCall]));
    }

    /// Text that merely starts like an end marker is released once it stops being one.
    #[test]
    fn a_lookalike_of_a_marker_is_text() {
        let r = scan("<|content_text|>a <|end_x and <|end_message", 1);
        assert_eq!(r.0, "a <|end_x and <|end_message");
    }

    /// Reasoning byte spans point at the reasoning in the generated text, tags excluded.
    #[test]
    fn reasoning_spans_cover_the_thinking_bodies() {
        let text = "<|content_thinking|>ab<|end_message|><|content_text|>x<|end_message|><|content_thinking|>cd<|end_message|>";
        let mut scanner = ChatScanner::with_flavor(false, TemplateFlavor::TypedMessages);
        let mut events = Vec::new();
        scanner.push(text, &mut events);
        scanner.finish(&mut events);
        let spans: Vec<&str> = scanner.reasoning_spans().iter().map(|r| &text[r.clone()]).collect();
        assert_eq!(spans, ["ab", "cd"]);
    }
}
