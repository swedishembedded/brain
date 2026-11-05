// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving agent workloads on every
// card of a multi-GPU host for its clients: conversations that resend a growing
// history each turn keep their prefix-cache hits while all cards stay busy. If
// your team needs expertise in multi-replica LLM serving then you can procure
// our services by sending an email to info@swedishembedded.com.

//! A text request as a prefix chain, for choosing the replica whose prefix cache holds its history.
//!
//! An agent resends its conversation every turn: the same tools and system prompt, the same messages, one more at the
//! end. The chain has one link for everything that shapes the start of the rendered prompt (tools, system prompt, template
//! switches, the adapter) and one per chat message after it, so turn `n + 1` of a conversation shares `n + 1` links with
//! turn `n`. The request text is hashed once, in one pass, on the dispatcher thread: nothing is rendered or tokenized, which
//! the replica does when it admits the request.

use capability::Invocation;
use residency::affinity::PrefixChain;

/// The request parameters that shape the start of the rendered prompt, before the first message.
const PROMPT_SHAPING: [&str; 8] = ["chat", "tools", "tool_choice", "system", "enable_thinking", "reasoning_effort", "preserve_thinking", "template_flavor"];

/// A raw prompt (no message boundaries) is cut into links of this many bytes.
const RAW_PROMPT_LINK_BYTES: usize = 1024;

/// `inv`'s prompt as a [`PrefixChain`]. `namespace` names the KV namespace the request runs in (the adapter it asks for,
/// empty for the base), because a prefix cached under one adapter is not another's.
pub fn request_prefix_chain(inv: &Invocation, namespace: &str) -> PrefixChain {
    let mut root = String::from(namespace);
    for key in PROMPT_SHAPING {
        if let Some(value) = inv.params.get(key) {
            root.push('\u{1f}');
            root.push_str(key);
            root.push('=');
            match value.as_str() {
                Some(text) => root.push_str(text),
                None => root.push_str(&value.to_string()),
            }
        }
    }
    let mut chain = PrefixChain::rooted(root.as_bytes());
    match inv.get_str("messages").filter(|m| !m.is_empty()) {
        Some(messages) => json_array_elements(&messages).into_iter().for_each(|message| chain.push(message.as_bytes())),
        None => chain.push_chunked(inv.get_str("prompt").unwrap_or_default().as_bytes(), RAW_PROMPT_LINK_BYTES),
    }
    chain
}

/// The top-level elements of the JSON array `text`, as slices of it, found in one scan without building a value. Text
/// that is not an array, or is cut short, yields what the scan found: a routing hint never refuses a request (the replica
/// that admits it reports the parse error).
fn json_array_elements(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut elements = Vec::new();
    let (mut depth, mut in_string, mut escaped, mut start) = (0usize, false, false, None::<usize>);
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                start.get_or_insert(i);
            }
            b'[' | b'{' => {
                depth += 1;
                if depth > 1 {
                    start.get_or_insert(i);
                }
            }
            b']' | b'}' => {
                if depth == 1 {
                    elements.extend(start.take().map(|s| text[s..i].trim_end()));
                    break;
                }
                depth = depth.saturating_sub(1);
            }
            b',' if depth == 1 => elements.extend(start.take().map(|s| text[s..i].trim_end())),
            b if depth >= 1 && !b.is_ascii_whitespace() => {
                start.get_or_insert(i);
            }
            _ => {}
        }
    }
    elements
}

#[cfg(test)]
mod tests {
    use super::*;
    use residency::affinity::PrefixDirectory;
    use serde_json::json;

    fn chat(system: &str, messages: &[(&str, &str)]) -> Invocation {
        let messages: Vec<_> = messages.iter().map(|(role, content)| json!({"role": role, "content": content})).collect();
        Invocation::new().set("system", json!(system)).set("messages", json!(serde_json::Value::Array(messages).to_string()))
    }

    fn matched(held: &PrefixChain, asked: &PrefixChain) -> u64 {
        let mut dir = PrefixDirectory::new(4);
        dir.record(held);
        dir.matched(asked)
    }

    #[test]
    fn the_next_turn_of_a_conversation_shares_every_message_of_the_last() {
        let system = "You are an agent. ".repeat(100);
        let turn1 = chat(&system, &[("user", "list the files")]);
        let turn2 = chat(&system, &[("user", "list the files"), ("assistant", "a.rs, b.rs, \"quoted, text\" [1,2]"), ("user", "open a.rs")]);
        let held = request_prefix_chain(&turn1, "");
        let asked = request_prefix_chain(&turn2, "");
        assert_eq!(matched(&held, &asked), matched(&held, &held), "everything turn one held is shared by turn two");
        assert!(matched(&held, &asked) > system.len() as u64);
        let other = chat(&system, &[("user", "something else")]);
        let root = request_prefix_chain(&chat(&system, &[]), "");
        assert_eq!(matched(&held, &request_prefix_chain(&other, "")), matched(&root, &root), "a different first message shares only the system prompt");
    }

    #[test]
    fn what_shapes_the_prompt_start_or_the_kv_namespace_separates_conversations() {
        let a = chat("system one", &[("user", "hi")]);
        let b = chat("system two", &[("user", "hi")]);
        assert_eq!(matched(&request_prefix_chain(&a, ""), &request_prefix_chain(&b, "")), 0, "another system prompt shares nothing");
        assert_eq!(matched(&request_prefix_chain(&a, ""), &request_prefix_chain(&a, "adapter-x")), 0, "another adapter's KV is not this one's");
        let with_tools = a.clone().set("tools", json!("[{\"type\":\"function\"}]"));
        assert_eq!(matched(&request_prefix_chain(&a, ""), &request_prefix_chain(&with_tools, "")), 0, "tools are the start of the rendered prompt");
    }

    #[test]
    fn a_raw_prompt_is_chained_in_chunks_that_extend_with_it() {
        let long = "x".repeat(5000);
        let held = request_prefix_chain(&Invocation::new().set("chat", json!(false)).set("prompt", json!(long.clone())), "");
        let asked = request_prefix_chain(&Invocation::new().set("chat", json!(false)).set("prompt", json!(format!("{long} and more"))), "");
        let shared = matched(&held, &asked);
        assert!((4096..4096 + 64).contains(&shared), "the four whole chunks of the shared text and the root, not the partial fifth: {shared}");
    }

    #[test]
    fn the_array_scan_splits_only_at_the_top_level_and_survives_hostile_text() {
        assert_eq!(json_array_elements(r#" [ {"a":[1,2]} , "x,]" , 3 ] "#), vec![r#"{"a":[1,2]}"#, r#""x,]""#, "3"]);
        assert!(json_array_elements("[]").is_empty() && json_array_elements("").is_empty());
        let _ = json_array_elements(r#"[{"a": "unterminated"#);
        let _ = json_array_elements("]]]}}}[[[{{{");
        let _ = json_array_elements(r#"["\"#);
    }
}
