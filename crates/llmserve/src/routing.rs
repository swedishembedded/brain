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

use std::borrow::Cow;

use capability::Invocation;
use residency::affinity::{text_request_chain, PrefixChain, TextPrompt, PROMPT_SHAPING};

/// `inv`'s prompt as a [`PrefixChain`] ([`text_request_chain`]). `namespace` names the KV namespace the request runs in
/// (the adapter it asks for, empty for the base), because a prefix cached under one adapter is not another's.
pub fn request_prefix_chain(inv: &Invocation, namespace: &str) -> PrefixChain {
    let shaping = PROMPT_SHAPING.iter().filter_map(|&key| {
        inv.params.get(key).map(|value| match value.as_str() {
            Some(text) => (key, Cow::Borrowed(text)),
            None => (key, Cow::Owned(value.to_string())),
        })
    });
    let messages = inv.get_str("messages").filter(|m| !m.is_empty());
    let raw = inv.get_str("prompt").unwrap_or_default();
    let prompt = match messages.as_deref() {
        Some(messages) => TextPrompt::Messages(messages),
        None => TextPrompt::Raw(&raw),
    };
    text_request_chain(namespace, shaping, prompt)
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
}
