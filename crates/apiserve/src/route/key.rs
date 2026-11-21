// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for putting one inference endpoint
// in front of every card of a cluster for its clients. If your team needs
// expertise in multi-node LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! What a request is routed by: its prompt as the served model's tokens.
//!
//! A prefix cache shares KV by token blocks, so the router keys a prompt the same way: each message's role and content
//! (or a raw prompt) encoded with the model's own tokenizer, cut into [`TOKEN_LINK`]-token links
//! ([`token_request_chain`]). Two requests then share a prefix exactly when their token streams do, whatever JSON
//! spelling the client used, and the prefill a hit saves is counted in tokens. The namespace of the chain is the model
//! id, the adapter digest the upstream serves under it and the prompt-shaping fields, so affinity never crosses an
//! adapter release; a base-weight change restarts the upstream, which changes the instance id its `/health` reports and
//! drops the router's record of it.
//!
//! The tokenizer comes from a [`TokenizerSource`] the embedder supplies (`brain serve --upstream` reads the router's
//! model store). A model without one is keyed by its request text instead ([`text_request_chain`]), reported per
//! request in the router's metrics: the index stays advisory either way, a stale entry costing one prefill, never a
//! wrong answer.
//!
//! Encodings are memoized per message ([`TokenCache`], bounded by tokens held), so a conversation that resends its
//! history every turn pays the tokenizer only for its new messages.

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::Hasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use residency::affinity::{text_request_chain, token_request_chain, PrefixChain, RequestSize, TextPrompt, PROMPT_SHAPING};
use serde_json::Value;

/// A model's tokenizer, as the router needs it: text in, token ids out.
pub trait PromptTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
}

/// Where the router finds a served model's tokenizer. Asked once per base model id (the id up to its first `:`, which
/// names an adapter); `None` keys that model's requests by text.
pub trait TokenizerSource: Send + Sync {
    fn tokenizer(&self, base_model: &str) -> Option<Arc<dyn PromptTokenizer>>;
}

/// The most tokens the memo holds, over all models.
const MAX_CACHED_TOKENS: usize = 8 << 20;

/// The output a request without `max_tokens` is assumed to add to the KV.
const DEFAULT_OUTPUT_TOKENS: u64 = 256;

/// Memoized encodings, oldest dropped first once they hold [`MAX_CACHED_TOKENS`].
#[derive(Default)]
struct TokenCache {
    encodings: HashMap<u64, Arc<[u32]>>,
    order: VecDeque<u64>,
    tokens: usize,
}

impl TokenCache {
    fn get(&self, key: u64) -> Option<Arc<[u32]>> {
        self.encodings.get(&key).cloned()
    }
    fn put(&mut self, key: u64, ids: Arc<[u32]>) {
        if ids.len() > MAX_CACHED_TOKENS || self.encodings.contains_key(&key) {
            return;
        }
        self.tokens += ids.len();
        self.encodings.insert(key, ids);
        self.order.push_back(key);
        while self.tokens > MAX_CACHED_TOKENS {
            let Some(old) = self.order.pop_front() else { break };
            if let Some(gone) = self.encodings.remove(&old) {
                self.tokens -= gone.len();
            }
        }
    }
}

/// The router's tokenizers, their memo and what they cost.
#[derive(Default)]
pub(crate) struct Tokens {
    source: Option<Arc<dyn TokenizerSource>>,
    tokenizers: Mutex<HashMap<String, Option<Arc<dyn PromptTokenizer>>>>,
    cache: Mutex<TokenCache>,
    pub(crate) token_keyed: AtomicU64,
    pub(crate) text_keyed: AtomicU64,
    pub(crate) encoded_tokens: AtomicU64,
    pub(crate) memo_tokens: AtomicU64,
    pub(crate) encode_nanos: AtomicU64,
}

impl Tokens {
    pub(crate) fn new(source: Option<Arc<dyn TokenizerSource>>) -> Tokens {
        Tokens { source, ..Tokens::default() }
    }

    /// The tokenizer of `model`'s base, loaded on first use. `model` is one the fleet lists.
    fn tokenizer(&self, model: &str) -> Option<Arc<dyn PromptTokenizer>> {
        let base = model.split(':').next().unwrap_or(model);
        if let Some(found) = self.tokenizers.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(base) {
            return found.clone();
        }
        let loaded = self.source.as_ref().and_then(|s| s.tokenizer(base));
        if loaded.is_none() {
            eprintln!("brain route: no tokenizer for {base}: its requests are routed by their text, not their tokens");
        }
        self.tokenizers.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(base.to_string(), loaded.clone());
        loaded
    }

    /// `text` encoded by `tok`, from the memo when it was encoded before.
    fn encode(&self, tok: &dyn PromptTokenizer, model: &str, text: &str) -> Arc<[u32]> {
        let mut h = DefaultHasher::new();
        h.write(model.as_bytes());
        h.write_u8(0x1f);
        h.write(text.as_bytes());
        let key = h.finish();
        if let Some(ids) = self.cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(key) {
            self.memo_tokens.fetch_add(ids.len() as u64, Ordering::Relaxed);
            return ids;
        }
        let started = std::time::Instant::now();
        let ids: Arc<[u32]> = tok.encode(text).into();
        self.encode_nanos.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.encoded_tokens.fetch_add(ids.len() as u64, Ordering::Relaxed);
        self.cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).put(key, ids.clone());
        ids
    }
}

/// How a forwarded route keys its requests.
#[derive(Clone, Copy)]
pub(crate) enum Keyed {
    /// A chat: the conversation's messages.
    Chat,
    /// A completion: the raw prompt.
    Completion,
    /// Nothing worth routing by: load alone decides.
    Load,
}

/// The parts of a request body the key is built from, owned, so the key can be built off the async workers.
pub(crate) struct Parts {
    pub model: String,
    pub keyed: Keyed,
    /// `(field, value as the client wrote it)` for each prompt-shaping field present.
    pub shaping: Vec<(&'static str, String)>,
    /// The `messages` array, or the `prompt`, as the client wrote it.
    pub prompt: Option<String>,
    pub max_tokens: Option<u64>,
}

impl Parts {
    /// The parts of `head` the key needs.
    pub(crate) fn of(head: &super::Head, model: String, keyed: Keyed) -> Parts {
        let shaping = PROMPT_SHAPING.iter().filter_map(|&key| head.get(key).map(|v| (key, v.get().to_string()))).collect();
        let field = match keyed {
            Keyed::Chat => Some("messages"),
            Keyed::Completion => Some("prompt"),
            Keyed::Load => None,
        };
        let prompt = field.and_then(|f| head.get(f)).map(|v| v.get().to_string());
        let max_tokens = ["max_completion_tokens", "max_tokens"].iter().find_map(|f| head.get(f)).and_then(|v| serde_json::from_str::<u64>(v.get()).ok());
        Parts { model, keyed, shaping, prompt, max_tokens }
    }
}

/// A request's prompt, as the router compares it with what each upstream holds.
pub(crate) enum PromptKey {
    /// Nothing to compare: load decides.
    Load,
    /// Keyed by the model's tokens: the shaping text and its token count, and the prompt's tokens.
    Tokens { root: String, root_tokens: u64, prompt: Vec<u32> },
    /// Keyed by text, the model having no tokenizer here.
    Text { shaping: Vec<(&'static str, String)>, messages: Option<String>, raw: Option<String> },
}

/// A request's routing key.
pub(crate) struct RequestKey {
    pub prompt: PromptKey,
    pub output: u64,
}

impl RequestKey {
    /// Build the key of `parts` (tokenizing it when its model has a tokenizer). CPU work: run it off the async workers.
    pub(crate) fn build(tokens: &Tokens, parts: Parts) -> RequestKey {
        let output = parts.max_tokens.unwrap_or(DEFAULT_OUTPUT_TOKENS).min(1 << 20);
        let Some(prompt_text) = parts.prompt.filter(|_| !matches!(parts.keyed, Keyed::Load)) else {
            return RequestKey { prompt: PromptKey::Load, output };
        };
        let root: String = parts.shaping.iter().map(|(k, v)| format!("\u{1f}{k}={v}")).collect();
        let Some(tok) = tokens.tokenizer(&parts.model) else {
            tokens.text_keyed.fetch_add(1, Ordering::Relaxed);
            let (messages, raw) = match parts.keyed {
                Keyed::Chat => (Some(prompt_text), None),
                _ => (None, Some(serde_json::from_str::<String>(&prompt_text).unwrap_or(prompt_text))),
            };
            return RequestKey { prompt: PromptKey::Text { shaping: parts.shaping, messages, raw }, output };
        };
        tokens.token_keyed.fetch_add(1, Ordering::Relaxed);
        let root_tokens = if root.is_empty() { 0 } else { tokens.encode(tok.as_ref(), &parts.model, &root).len() as u64 };
        let mut prompt = Vec::new();
        match parts.keyed {
            Keyed::Chat => {
                // Each message as role and content, so a client's JSON spelling (key order, escapes, spacing) does not
                // change its identity.
                let messages: Vec<Value> = serde_json::from_str(&prompt_text).unwrap_or_default();
                for message in &messages {
                    let text = message_text(message);
                    prompt.extend_from_slice(&tokens.encode(tok.as_ref(), &parts.model, &text));
                }
            }
            _ => {
                let raw = serde_json::from_str::<String>(&prompt_text).unwrap_or(prompt_text);
                prompt.extend_from_slice(&tokens.encode(tok.as_ref(), &parts.model, &raw));
            }
        }
        RequestKey { prompt: PromptKey::Tokens { root, root_tokens, prompt }, output }
    }

    /// The chain under `namespace` (the model and the adapter digest an upstream serves it under).
    pub(crate) fn chain(&self, namespace: &str) -> PrefixChain {
        match &self.prompt {
            PromptKey::Load => PrefixChain::default(),
            PromptKey::Tokens { root, root_tokens, prompt } => token_request_chain(format!("{namespace}{root}").as_bytes(), *root_tokens, prompt),
            PromptKey::Text { shaping, messages, raw } => {
                let shaping = shaping.iter().map(|(k, v)| (*k, Cow::Borrowed(v.as_str())));
                let prompt = match (messages, raw) {
                    (Some(m), _) => TextPrompt::Messages(m),
                    (None, Some(r)) => TextPrompt::Raw(r),
                    (None, None) => TextPrompt::Raw(""),
                };
                text_request_chain(namespace, shaping, prompt)
            }
        }
    }

    /// The request's size in the units of its chain.
    pub(crate) fn size(&self, chain: &PrefixChain) -> RequestSize {
        RequestSize { prompt: chain.units(), output: if matches!(self.prompt, PromptKey::Tokens { .. }) { self.output } else { 0 } }
    }

    /// Whether the chain counts tokens, so an upstream's free KV (in tokens) compares with it.
    pub(crate) fn in_tokens(&self) -> bool {
        matches!(self.prompt, PromptKey::Tokens { .. })
    }
}

/// A chat message as the text it is identified by: its role, then its content (a string as written, anything else as
/// canonical JSON), then any tool calls or tool-call id it carries.
fn message_text(message: &Value) -> String {
    let mut text = String::new();
    text.push_str(message.get("role").and_then(Value::as_str).unwrap_or(""));
    text.push('\n');
    match message.get("content") {
        Some(Value::String(s)) => text.push_str(s),
        Some(other) => text.push_str(&other.to_string()),
        None => {}
    }
    for field in ["tool_calls", "tool_call_id", "name"] {
        if let Some(v) = message.get(field) {
            text.push('\u{1f}');
            text.push_str(&v.to_string());
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splits on whitespace into one token per word (its length), and counts calls.
    struct Words(AtomicU64);
    impl PromptTokenizer for Words {
        fn encode(&self, text: &str) -> Vec<u32> {
            self.0.fetch_add(1, Ordering::SeqCst);
            text.split_whitespace().map(|w| w.len() as u32).collect()
        }
    }
    struct Source(Arc<Words>);
    impl TokenizerSource for Source {
        fn tokenizer(&self, _base: &str) -> Option<Arc<dyn PromptTokenizer>> {
            Some(self.0.clone())
        }
    }

    fn parts(messages: &str) -> Parts {
        Parts { model: "m:adapter".into(), keyed: Keyed::Chat, shaping: vec![], prompt: Some(messages.into()), max_tokens: Some(32) }
    }

    #[test]
    fn a_chat_is_keyed_by_its_tokens_whatever_its_json_spelling_and_history_is_encoded_once() {
        let words = Arc::new(Words(AtomicU64::new(0)));
        let tokens = Tokens::new(Some(Arc::new(Source(words.clone()))));
        let one = RequestKey::build(&tokens, parts(r#"[{"role":"user","content":"a b c d"}]"#));
        let spaced = RequestKey::build(&tokens, parts(r#"[ { "content" : "a b c d", "role" : "user" } ]"#));
        assert_eq!(one.chain("ns"), spaced.chain("ns"), "the same message in another JSON spelling");
        assert_eq!(words.0.load(Ordering::SeqCst), 1, "the second spelling came from the memo");
        let two = RequestKey::build(&tokens, parts(r#"[{"role":"user","content":"a b c d"},{"role":"assistant","content":"e f"}]"#));
        assert_eq!(words.0.load(Ordering::SeqCst), 2, "turn two encodes only its new message");
        let (c1, c2) = (one.chain("ns"), two.chain("ns"));
        assert!(c2.units() > c1.units());
        assert_ne!(one.chain("ns"), one.chain("ns-other-adapter"), "another namespace shares nothing");
        assert_eq!(one.size(&c1).output, 32);
        assert_eq!(tokens.token_keyed.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn without_a_tokenizer_a_request_is_keyed_by_text_and_counted() {
        let tokens = Tokens::new(None);
        let key = RequestKey::build(&tokens, parts(r#"[{"role":"user","content":"hi"}]"#));
        assert!(!key.in_tokens());
        assert!(key.chain("ns").units() > 0);
        assert_eq!(tokens.text_keyed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_memo_is_bounded_by_the_tokens_it_holds() {
        let mut cache = TokenCache::default();
        let big: Arc<[u32]> = vec![0u32; MAX_CACHED_TOKENS / 2 + 1].into();
        cache.put(1, big.clone());
        cache.put(2, big.clone());
        assert!(cache.tokens <= MAX_CACHED_TOKENS && cache.get(1).is_none() && cache.get(2).is_some());
    }
}
