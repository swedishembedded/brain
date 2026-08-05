// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic chat-template rendering: execute a checkpoint's OWN
//! `tokenizer_config.json` `chat_template` string (Jinja2, exactly what
//! HuggingFace's `AutoTokenizer.apply_chat_template` runs) via `minijinja`,
//! instead of hand-porting each model family's template control flow into
//! Rust one model at a time. [`crate::qwen_chat`] is Qwen3-specific hardcoded
//! Rust (its own doc: "reproduced exactly... following the Jinja template's
//! control flow line for line") — correct for Qwen3, but a new model family
//! (GLM, a future import, a fine-tuned checkpoint with a customized template)
//! needs the SAME transcription work done again by hand. This module instead
//! interprets whatever `chat_template` string a checkpoint actually ships,
//! so it scales across model families with no per-model Rust code.
//!
//! # Why `minijinja::Value`, never `serde_json::Value`, for template data
//!
//! A `tojson` filter call inside a real chat template must reproduce the
//! SOURCE JSON's key order byte-for-byte (Python's `json.dumps` preserves
//! insertion order; a tool schema round-tripped through an order-losing map
//! would silently reorder its own keys). The workspace's shared `serde_json`
//! dependency does NOT build with `preserve_order` (`crate::qwen_chat`'s
//! `json_py` submodule exists specifically because of this), and turning
//! that on would be a workspace-wide behavior change with unknown blast
//! radius on everything else that serializes a `serde_json::Value`. Instead,
//! [`parse_json_ordered`] deserializes raw JSON text DIRECTLY into
//! `minijinja::Value` (`serde_json::Deserializer` only parses JSON syntax;
//! `minijinja::Value`'s own `Deserialize` impl builds the tree using ITS OWN
//! map type, governed by minijinja's own, independent `preserve_order`
//! feature) — a `serde_json::Value` intermediate is never constructed, so
//! this is fully isolated from the rest of the workspace.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

use minijinja::{Environment, Value};

#[derive(Debug)]
pub struct TemplateError(String);

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TemplateError {}

/// Parse raw JSON text into a `minijinja::Value`, preserving source key
/// order (see the module doc). Use this for anything a template might
/// `tojson` back out (tool schemas, echoed tool-call arguments) — never
/// round-trip such data through `serde_json::Value`.
pub fn parse_json_ordered(raw: &str) -> Result<Value, TemplateError> {
    serde_json::from_str::<Value>(raw).map_err(|e| TemplateError(format!("invalid JSON: {e}")))
}

/// Write `v` the way Python's `json.dumps(v, ensure_ascii=False)` would:
/// `", "` between elements, `": "` after keys, map key order preserved
/// (`minijinja::Value`'s own map already IS ordered — see the module doc),
/// non-ASCII passed through literally. Mirrors `crate::qwen_chat::json_py`'s
/// write side, but over `minijinja::Value` instead of a hand-rolled parse
/// tree, since minijinja already gives us an order-preserving parsed value.
///
/// Known limitation: a genuine float that happens to be a whole number
/// (`16.0`) prints as `16`, not `16.0` — `minijinja::Value` does not retain
/// a number's original lexical text the way `json_py::Node::Num` does, and
/// tool-schema JSON in practice is overwhelmingly strings/objects/arrays, so
/// this has not been worth chasing further.
fn write_pyjson(v: &Value, out: &mut String) {
    use minijinja::value::ValueKind;
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            if let Some(i) = v.as_i64() {
                out.push_str(&i.to_string());
            } else if let Ok(f) = f64::try_from(v.clone()) {
                out.push_str(&f.to_string());
            } else {
                out.push_str(&v.to_string());
            }
        }
        ValueKind::String => write_pystr(v.as_str().unwrap_or_default(), out),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (i, item) in v.try_iter().into_iter().flatten().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_pyjson(&item, out);
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, key) in v.try_iter().into_iter().flatten().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let key_str = key.as_str().map(str::to_string).unwrap_or_else(|| key.to_string());
                write_pystr(&key_str, out);
                out.push_str(": ");
                let val = v.get_item(&key).unwrap_or(Value::from(()));
                write_pyjson(&val, out);
            }
            out.push('}');
        }
        _ => write_pystr(&v.to_string(), out),
    }
}

/// Python `json.dumps` string escaping: `\`, `"`, and the six named control
/// escapes get their short form; every other control char (< 0x20) gets
/// `\u00xx`; everything else — including all non-ASCII — is written
/// through literally (`ensure_ascii=False`).
fn write_pystr(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A compiled chat template, ready to render. Cheap to keep around (one per
/// distinct `chat_template` string in use); NOT cheap to reconstruct per
/// call — `compile` parses the Jinja source.
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    /// Compile `jinja_src` (the verbatim `chat_template` string from a
    /// checkpoint's `tokenizer_config.json`). Registers the globals real HF
    /// templates reference: `raise_exception` (a template's own validation
    /// errors surface as a [`TemplateError`], not a panic), `strftime_now`
    /// and Python-method emulation (`.split`/`.strip`/`.startswith`/…) via
    /// `minijinja_contrib`'s `pycompat` layer — templates written against
    /// Python's Jinja2 call these as if `messages`/strings were Python
    /// objects, and minijinja does not do that by default.
    pub fn compile(jinja_src: &str) -> Result<ChatTemplate, TemplateError> {
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        // minijinja's BUILT-IN `tojson` writes compact JSON (no spaces) --
        // real HF chat templates run under `transformers`, which registers a
        // `tojson` matching Python's `json.dumps(x, ensure_ascii=False)`
        // defaults: ", " between elements, ": " after keys. A tool schema
        // round-tripped through the wrong separator convention would not
        // byte-match what the checkpoint's OWN template actually produces
        // for real inference/training, so this OVERRIDES the built-in filter.
        env.add_filter("tojson", |v: Value| -> String {
            let mut out = String::new();
            write_pyjson(&v, &mut out);
            out
        });
        env.add_template_owned("chat", jinja_src.to_string()).map_err(|e| TemplateError(format!("{e:#}")))?;
        Ok(ChatTemplate { env })
    }

    /// Compile the `chat_template` field out of `<model_dir>/tokenizer_config.json`
    /// -- for a base model fetched through brain's normal store/plan path,
    /// this file is ALREADY on disk (`modelstore::plan`'s `plan_base`
    /// downloads it alongside `tokenizer.json` whenever the upstream repo
    /// ships one), so this needs no separate import-time wiring. Returns a
    /// `TemplateError` naming which part is missing (file absent, no
    /// `chat_template` key, unparseable Jinja) rather than panicking --
    /// a checkpoint with no template is a real, expected case (e.g. a base,
    /// non-instruction-tuned model), not a bug to crash on.
    pub fn from_model_dir(dir: &std::path::Path) -> Result<ChatTemplate, TemplateError> {
        let path = dir.join("tokenizer_config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| TemplateError(format!("{}: {e}", path.display())))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| TemplateError(format!("{}: invalid JSON: {e}", path.display())))?;
        let src = cfg
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TemplateError(format!("{}: no \"chat_template\" string field", path.display())))?;
        Self::compile(src)
    }

    /// Render `messages` (a JSON array of `{role, content, tool_calls?,
    /// tool_call_id?, ...}` objects — build via [`parse_json_ordered`] or
    /// `Value::from_serialize`) through the template. `tools` is the JSON
    /// schema array (or `None` for no tool preamble). `extra` carries any
    /// further template kwargs a specific model's template reads
    /// (`enable_thinking`, `bos_token`, …) — passed through verbatim, so
    /// this stays generic across templates with different kwarg needs.
    pub fn render(&self, messages: Value, tools: Option<Value>, add_generation_prompt: bool, extra: &BTreeMap<String, Value>) -> Result<String, TemplateError> {
        let tmpl = self.env.get_template("chat").map_err(|e| TemplateError(format!("{e:#}")))?;
        let mut ctx: BTreeMap<String, Value> = extra.clone();
        ctx.insert("messages".to_string(), messages);
        ctx.insert("tools".to_string(), tools.unwrap_or_else(|| Value::from(Vec::<Value>::new())));
        ctx.insert("add_generation_prompt".to_string(), Value::from(add_generation_prompt));
        tmpl.render(ctx).map_err(|e| TemplateError(format!("{e:#}")))
    }

    /// Render the full conversation once, AND determine each message's byte
    /// range within it — for SFT loss masking, which needs to know exactly
    /// which tokens came from which message.
    ///
    /// The boundary for message `i` is found by rendering `messages[0..=i]`
    /// in isolation (`add_generation_prompt: false`) and checking it is a
    /// genuine PREFIX of the full render. This is NOT guessable in general:
    /// a template's rendering of message `i` can depend on whether more
    /// messages follow it — Qwen3's own `chat_template` inserts an empty
    /// `<think>\n\n</think>\n\n` block for an assistant turn only when it is
    /// LITERALLY the last message (`loop.last`), so a truncated prefix can
    /// render that same message differently than the full conversation does
    /// (concretely: a tool-call turn followed by a tool result and a final
    /// answer -- an ordinary packed-conversation shape, two assistant turns
    /// after the last real user turn -- hits this, since the tool-call turn
    /// LOOKS last in a 3-message truncation but isn't in the true 5-message
    /// conversation).
    ///
    /// KNOWN LIMITATION, not silently absorbed: a smarter probe (render the
    /// full-length array with everything after `i` reduced to role-only, so
    /// `loop.last`/message count stay correct) was tried and reverted -- it
    /// can pull a FOLLOWING message's shared opening boilerplate into `i`'s
    /// span whenever the role-only and real renderings of that following
    /// message share a literal prefix (the common case: `<|im_start|>{role}`
    /// is always emitted before any content-dependent branching), which is a
    /// smaller but still real silent-mismask risk. Failing loudly on the
    /// hazard, as this does, is the safe choice until a provably exact fix
    /// exists — see docs/guides/training.md "Known gaps".
    pub fn render_with_message_boundaries(&self, messages: &[Value], tools: Option<Value>) -> Result<(String, Vec<Range<usize>>), TemplateError> {
        let full = self.render(Value::from(messages.to_vec()), tools.clone(), false, &BTreeMap::new())?;
        let mut boundaries = Vec::with_capacity(messages.len());
        let mut prev_len = 0usize;
        for i in 0..messages.len() {
            let prefix_msgs: Vec<Value> = messages[..=i].to_vec();
            let prefix = self.render(Value::from(prefix_msgs), tools.clone(), false, &BTreeMap::new())?;
            if !full.as_bytes().starts_with(prefix.as_bytes()) {
                return Err(TemplateError(format!(
                    "message {i} is not prefix-stable under this template: rendering it in isolation \
                     from the rest of the conversation produced different text than rendering it in \
                     full context (the template's output for this message depends on what comes AFTER \
                     it). Refusing to guess a loss-mask boundary for it."
                )));
            }
            boundaries.push(prev_len..prefix.len());
            prev_len = prefix.len();
        }
        if prev_len != full.len() {
            return Err(TemplateError(format!(
                "message boundaries sum to {prev_len} bytes but the full render is {} bytes",
                full.len()
            )));
        }
        Ok((full, boundaries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_env() -> ChatTemplate {
        ChatTemplate::compile(
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
        )
        .unwrap()
    }

    #[test]
    fn renders_a_minimal_template() {
        let t = tiny_env();
        let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        let out = t.render(messages, None, true, &BTreeMap::new()).unwrap();
        assert_eq!(out, "<|user|>hi<|assistant|>");
    }

    #[test]
    fn parse_json_ordered_preserves_key_order_through_tojson() {
        let t = ChatTemplate::compile("{{ obj | tojson }}").unwrap();
        // Deliberately reversed from natural alphabetical order.
        let obj = parse_json_ordered(r#"{"zebra":1,"apple":2}"#).unwrap();
        let mut ctx = BTreeMap::new();
        ctx.insert("obj".to_string(), obj);
        let out = t.render(Value::from(Vec::<Value>::new()), None, false, &ctx).unwrap();
        assert_eq!(out, r#"{"zebra": 1, "apple": 2}"#);
    }

    #[test]
    fn render_with_message_boundaries_splits_a_simple_conversation_exactly() {
        let t = tiny_env();
        let messages: Vec<Value> = vec![
            parse_json_ordered(r#"{"role":"user","content":"hi"}"#).unwrap(),
            parse_json_ordered(r#"{"role":"assistant","content":"hey"}"#).unwrap(),
        ];
        let (full, ranges) = t.render_with_message_boundaries(&messages, None).unwrap();
        assert_eq!(full, "<|user|>hi<|assistant|>hey");
        assert_eq!(&full[ranges[0].clone()], "<|user|>hi");
        assert_eq!(&full[ranges[1].clone()], "<|assistant|>hey");
    }

    #[test]
    fn from_model_dir_reads_chat_template_out_of_tokenizer_config_json() {
        let dir = std::env::temp_dir().join(format!("brain-chat-template-fromdir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"chat_template": "<|{{ messages[0].role }}|>{{ messages[0].content }}"}"#).unwrap();
        let t = ChatTemplate::from_model_dir(&dir).expect("load");
        let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        assert_eq!(t.render(messages, None, false, &BTreeMap::new()).unwrap(), "<|user|>hi");
    }

    #[test]
    fn from_model_dir_errors_clearly_when_chat_template_is_absent() {
        let dir = std::env::temp_dir().join(format!("brain-chat-template-nofield-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"some_other_field": true}"#).unwrap();
        let Err(err) = ChatTemplate::from_model_dir(&dir) else { panic!("expected an error") };
        assert!(err.to_string().contains("chat_template"), "got: {err}");
    }

    #[test]
    fn render_with_message_boundaries_rejects_a_non_prefix_stable_template() {
        // A template whose rendering of a message depends on whether it's
        // literally the last one -- exactly Qwen3's <think> empty-block
        // hazard, minimized, and exactly the common "tool call, then a final
        // answer" packed-conversation shape (two assistant turns after the
        // real content, only the true-last one special-cased). A truncated
        // 1-message prefix makes message 0 LOOK last when it isn't. Must
        // error, not silently mis-measure -- see this method's doc for why a
        // smarter probe was tried and reverted rather than shipped half-right.
        let t = ChatTemplate::compile("{% for m in messages %}{{ m.content }}{% if loop.last %}[LAST]{% endif %}{% endfor %}").unwrap();
        let messages: Vec<Value> = vec![
            parse_json_ordered(r#"{"role":"assistant","content":"a"}"#).unwrap(),
            parse_json_ordered(r#"{"role":"assistant","content":"b"}"#).unwrap(),
        ];
        let err = t.render_with_message_boundaries(&messages, None).unwrap_err();
        assert!(err.to_string().contains("not prefix-stable"), "got: {err}");
    }

    #[test]
    fn raise_exception_surfaces_as_a_template_error_not_a_panic() {
        let t = ChatTemplate::compile("{{ raise_exception('nope') }}").unwrap();
        let err = t.render(Value::from(Vec::<Value>::new()), None, false, &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn pycompat_string_methods_work() {
        // pycompat renders a bool the Python way ("True"/"False"), matching
        // what a real Jinja2/Python environment would produce for the same
        // template -- distinct from minijinja's own native "true"/"false".
        let t = ChatTemplate::compile("{{ 'hello world'.startswith('hello') }}").unwrap();
        let out = t.render(Value::from(Vec::<Value>::new()), None, false, &BTreeMap::new()).unwrap();
        assert_eq!(out, "True");
    }
}
