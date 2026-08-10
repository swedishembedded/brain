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
    /// Context defaults injected under every render, overridable per call via
    /// `extra` — today `bos_token`/`eos_token` parsed out of the same
    /// `tokenizer_config.json` the template came from ([`Self::from_model_dir`]).
    /// Before this existed those fields were parsed and DISCARDED, and a
    /// template's `{{ bos_token }}` rendered as the empty string silently —
    /// a prompt that does not byte-match HF's for the same checkpoint.
    defaults: BTreeMap<String, Value>,
}

impl ChatTemplate {
    /// Compile `jinja_src` (the verbatim `chat_template` string from a
    /// checkpoint's `tokenizer_config.json`). Matches how `transformers`
    /// compiles the same string, because "renders byte-identically to HF" is
    /// this module's whole contract:
    ///
    /// * `trim_blocks` + `lstrip_blocks` are ON (`transformers` passes both
    ///   to its Jinja2 environment). Invisible for a template that `-`-marks
    ///   every tag (Qwen3 does), silently-wrong extra newlines for any
    ///   template with bare block tags — exactly the "new model family" case
    ///   this module exists for.
    /// * undefined is CHAINABLE (`transformers` uses `ChainableUndefined`):
    ///   `a.b.c` over an undefined renders empty instead of erroring.
    ///
    /// Registers the globals real HF templates reference: `raise_exception`
    /// (a template's own validation errors surface as a [`TemplateError`],
    /// not a panic), `strftime_now` (Llama/Mistral date headers; UTC — see
    /// [`strftime_utc_now`] for the documented local-time divergence) and
    /// Python-method emulation (`.split`/`.strip`/`.startswith`/…) via
    /// `minijinja_contrib`'s `pycompat` layer — templates written against
    /// Python's Jinja2 call these as if `messages`/strings were Python
    /// objects, and minijinja does not do that by default.
    pub fn compile(jinja_src: &str) -> Result<ChatTemplate, TemplateError> {
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Chainable);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| -> String { strftime_utc_now(&fmt) });
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
        Ok(ChatTemplate { env, defaults: BTreeMap::new() })
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
        let mut t = Self::compile(src)?;
        // The SAME file declares the special tokens many templates reference
        // (`{{ bos_token }}` heads Llama/Mistral prompts) — inject them as
        // render defaults instead of parsing and discarding them. HF encodes
        // them either as a bare string or as an AddedToken object with a
        // `content` field; both are read, absent stays absent (chainable
        // undefined then matches transformers' own behaviour).
        for key in ["bos_token", "eos_token"] {
            let tok = cfg.get(key).and_then(|v| v.as_str().or_else(|| v.get("content").and_then(|c| c.as_str())));
            if let Some(s) = tok {
                t.defaults.insert(key.to_string(), Value::from(s.to_string()));
            }
        }
        Ok(t)
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
        let mut ctx: BTreeMap<String, Value> = self.defaults.clone();
        ctx.extend(extra.clone());
        ctx.insert("messages".to_string(), messages);
        // Absent tools are `none`, exactly as `apply_chat_template` passes
        // them — the old `[]` stand-in flipped every `tools is not none`
        // branch into emitting a tool preamble for a tool-less call.
        ctx.insert("tools".to_string(), tools.unwrap_or_else(|| Value::from(())));
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
    /// COST (deliberate): O(n²) in rendered bytes per sample — message `i`'s
    /// boundary re-renders `messages[..=i]`, so a 40-turn trajectory costs
    /// ~20× the single-render work, per sample, over the whole training set.
    /// The correctness reasoning below is why; cache/incrementalise only if
    /// multi-turn packing becomes the norm and this shows up in a profile.
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

/// `strftime_now(format)` for templates (Llama/Mistral date headers), over
/// **UTC** from `SystemTime` — pure std, no tz database. HF's Python
/// `strftime_now` uses the machine's LOCAL time; near midnight the rendered
/// date can differ by one day, a documented divergence preferred over adding
/// a timezone dependency for a prompt header. Supports the strftime codes
/// real chat templates use; an unrecognised `%x` passes through verbatim.
fn strftime_utc_now(fmt: &str) -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    strftime_utc(fmt, secs)
}

/// [`strftime_utc_now`]'s pure core, on an explicit unix timestamp (testable).
fn strftime_utc(fmt: &str, unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86400);
    let sod = unix_secs.rem_euclid(86400);
    let (hh, mm, ss) = ((sod / 3600) as u32, ((sod / 60) % 60) as u32, (sod % 60) as u32);
    // Howard Hinnant's civil_from_days: days since 1970-01-01 -> (y, m, d).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    let weekday = ((days % 7 + 4) % 7 + 7) % 7; // 0 = Sunday (1970-01-01 was a Thursday)
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    const CUM: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let yday = CUM[(m - 1) as usize] + d + u32::from(leap && m > 2);
    const MON: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const DAY: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

    let mut out = String::with_capacity(fmt.len() + 8);
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:2}")),
            Some('H') => out.push_str(&format!("{hh:02}")),
            Some('M') => out.push_str(&format!("{mm:02}")),
            Some('S') => out.push_str(&format!("{ss:02}")),
            Some('I') => out.push_str(&format!("{:02}", if hh % 12 == 0 { 12 } else { hh % 12 })),
            Some('p') => out.push_str(if hh < 12 { "AM" } else { "PM" }),
            Some('j') => out.push_str(&format!("{yday:03}")),
            Some('B') => out.push_str(MON[(m - 1) as usize]),
            Some('b') | Some('h') => out.push_str(&MON[(m - 1) as usize][..3]),
            Some('A') => out.push_str(DAY[weekday as usize]),
            Some('a') => out.push_str(&DAY[weekday as usize][..3]),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
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

    /// SPEC (audit F29): transformers compiles chat templates with
    /// `trim_blocks=True, lstrip_blocks=True`. A template with BARE block
    /// tags (no `-` markers) must render without the extra newlines/indent
    /// minijinja's defaults leave behind — invisible on Qwen3 (all 89 tags
    /// are `-`-marked), silently different prompts on any template that
    /// isn't.
    #[test]
    fn bare_block_tags_render_like_hf_trim_and_lstrip() {
        let t = ChatTemplate::compile("{% for m in messages %}\n  {% if true %}\n<{{ m.content }}>\n  {% endif %}\n{% endfor %}").unwrap();
        let messages = parse_json_ordered(r#"[{"role":"user","content":"a"},{"role":"user","content":"b"}]"#).unwrap();
        let out = t.render(messages, None, false, &BTreeMap::new()).unwrap();
        // trim_blocks eats the newline after each block tag; lstrip_blocks
        // eats the indentation before one. Only the literal lines remain.
        assert_eq!(out, "<a>\n<b>\n");
    }

    /// SPEC (audit F29): absent tools must be `none` (what HF passes), not
    /// `[]` — an empty list flips every `tools is not none` branch into
    /// emitting a tool preamble for a tool-less call.
    #[test]
    fn absent_tools_are_none_not_an_empty_list() {
        let t = ChatTemplate::compile("{% if tools is not none %}TOOLS{% else %}NO-TOOLS{% endif %}").unwrap();
        let msgs = || Value::from(Vec::<Value>::new());
        assert_eq!(t.render(msgs(), None, false, &BTreeMap::new()).unwrap(), "NO-TOOLS");
        let tools = parse_json_ordered(r#"[{"name":"f"}]"#).unwrap();
        assert_eq!(t.render(msgs(), Some(tools), false, &BTreeMap::new()).unwrap(), "TOOLS");
    }

    /// SPEC (audit F29): the doc always claimed `strftime_now` was
    /// registered; it wasn't, and Llama/Mistral date-header templates failed
    /// loudly. Registered now (UTC).
    #[test]
    fn strftime_now_is_registered_and_renders_a_date() {
        let t = ChatTemplate::compile("{{ strftime_now('%d %b %Y') }}").unwrap();
        let out = t.render(Value::from(Vec::<Value>::new()), None, false, &BTreeMap::new()).unwrap();
        // e.g. "10 Aug 2026" — shape-check, not clock-pinning.
        assert_eq!(out.len(), 11, "{out:?}");
        assert!(out[7..].chars().all(|c| c.is_ascii_digit()), "{out:?}");
    }

    /// The pure strftime core against known timestamps.
    #[test]
    fn strftime_utc_formats_known_timestamps_exactly() {
        // 2026-08-10 (a Monday) 15:04:05 UTC.
        let ts = 1_786_374_245;
        assert_eq!(strftime_utc("%Y-%m-%d %H:%M:%S", ts), "2026-08-10 15:04:05");
        assert_eq!(strftime_utc("%d %b %Y", ts), "10 Aug 2026");
        assert_eq!(strftime_utc("%A, %B %d", ts), "Monday, August 10");
        assert_eq!(strftime_utc("%I:%M %p", ts), "03:04 PM");
        assert_eq!(strftime_utc("%j", ts), "222");
        assert_eq!(strftime_utc("100%% %q", ts), "100% %q");
        // Epoch: Thursday 1970-01-01, and midnight is 12 AM.
        assert_eq!(strftime_utc("%a %Y-%m-%d %I %p", 0), "Thu 1970-01-01 12 AM");
        // Leap-year day-of-year past February.
        assert_eq!(strftime_utc("%Y-%m-%d %j", 951_868_800), "2000-03-01 061");
    }

    /// SPEC (audit F29): `bos_token`/`eos_token` from the SAME
    /// tokenizer_config.json are injected as render defaults (string or
    /// AddedToken-object form), overridable via `extra`; a template
    /// referencing them no longer renders empty strings silently.
    #[test]
    fn from_model_dir_injects_bos_and_eos_tokens() {
        let dir = std::env::temp_dir().join(format!("brain-chat-template-bos-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template": "{{ bos_token }}{{ messages[0].content }}{{ eos_token }}", "bos_token": "<s>", "eos_token": {"content": "</s>"}}"#,
        )
        .unwrap();
        let t = ChatTemplate::from_model_dir(&dir).expect("load");
        let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        assert_eq!(t.render(messages.clone(), None, false, &BTreeMap::new()).unwrap(), "<s>hi</s>");
        // extra overrides a default.
        let mut extra = BTreeMap::new();
        extra.insert("bos_token".to_string(), Value::from("<B>"));
        assert_eq!(t.render(messages, None, false, &extra).unwrap(), "<B>hi</s>");
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
