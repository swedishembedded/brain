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

use minijinja::Environment;
/// The value type templates render and [`parse_json_ordered`] produces.
pub use minijinja::Value;
/// What kind of value a [`Value`] is (a map, a string, ...).
pub use minijinja::value::ValueKind;

/// Why a conversation could not be rendered or encoded. When the reason is a rule of the training-record contract
/// ([`crate::chat_rollout::RecordFault`]: a log-probability count, sampled ids that do not round-trip), it is carried
/// typed beside the message, so a dataset check reports the rule and the field and not only the text.
#[derive(Debug)]
pub struct TemplateError {
    message: String,
    fault: Option<crate::chat_rollout::RecordFault>,
}

impl TemplateError {
    pub(crate) fn new(message: impl Into<String>) -> TemplateError {
        TemplateError { message: message.into(), fault: None }
    }

    /// A refusal of the training-record contract.
    pub(crate) fn of_record(fault: crate::chat_rollout::RecordFault, message: String) -> TemplateError {
        TemplateError { message, fault: Some(fault) }
    }

    /// The contract rule the conversation broke, when that is why it was refused.
    pub fn fault(&self) -> Option<&crate::chat_rollout::RecordFault> {
        self.fault.as_ref()
    }
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TemplateError {}

impl From<TemplateError> for crate::chat_rollout::RecordError {
    fn from(e: TemplateError) -> crate::chat_rollout::RecordError {
        match e.fault {
            Some(fault) => crate::chat_rollout::RecordError::of(fault, e.message),
            None => e.message.into(),
        }
    }
}

/// Parse raw JSON text into a `minijinja::Value`, preserving source key
/// order (see the module doc). Use this for anything a template might
/// `tojson` back out (tool schemas, echoed tool-call arguments) — never
/// round-trip such data through `serde_json::Value`.
pub fn parse_json_ordered(raw: &str) -> Result<Value, TemplateError> {
    serde_json::from_str::<Value>(raw).map_err(|e| TemplateError::new(format!("invalid JSON: {e}")))
}

/// The knobs of `transformers`' `tojson` filter, which is `json.dumps(x, ensure_ascii=False,
/// indent=indent, separators=separators, sort_keys=sort_keys)`: what a template can ask of the
/// serialization it writes into a prompt (tool schemas, call arguments).
#[derive(Clone, Debug)]
struct PyJson {
    item_separator: String,
    key_separator: String,
    indent: Option<usize>,
    sort_keys: bool,
    ensure_ascii: bool,
}

impl PyJson {
    /// Python's defaults: `", "` and `": "`, or `","` and `": "` once an indent puts items on lines.
    fn new(indent: Option<usize>, separators: Option<(String, String)>, sort_keys: bool, ensure_ascii: bool) -> PyJson {
        let (item_separator, key_separator) = separators.unwrap_or_else(|| (if indent.is_some() { "," } else { ", " }.to_string(), ": ".to_string()));
        PyJson { item_separator, key_separator, indent, sort_keys, ensure_ascii }
    }

    /// `v` as Python would write it, map key order preserved (`minijinja::Value`'s own map IS
    /// ordered, see the module doc) unless `sort_keys`.
    fn dumps(&self, v: &Value) -> String {
        let mut out = String::new();
        self.write(v, 0, &mut out);
        out
    }

    fn newline(&self, level: usize, out: &mut String) {
        if let Some(indent) = self.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', indent * level));
        }
    }

    fn write(&self, v: &Value, level: usize, out: &mut String) {
        use minijinja::value::ValueKind;
        match v.kind() {
            ValueKind::Undefined | ValueKind::None => out.push_str("null"),
            ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
            ValueKind::Number => match v.as_i64() {
                Some(i) if v.is_integer() => out.push_str(&i.to_string()),
                _ if v.is_integer() => out.push_str(&v.to_string()),
                _ => out.push_str(&py_float_json(f64::try_from(v.clone()).unwrap_or(f64::NAN))),
            },
            ValueKind::String => self.write_str(v.as_str().unwrap_or_default(), out),
            ValueKind::Seq | ValueKind::Iterable => {
                let items: Vec<Value> = v.try_iter().into_iter().flatten().collect();
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_separator);
                    }
                    self.newline(level + 1, out);
                    self.write(item, level + 1, out);
                }
                self.newline(level, out);
                out.push(']');
            }
            ValueKind::Map => {
                let mut keys: Vec<(String, Value)> = v.try_iter().into_iter().flatten().map(|k| (k.as_str().map(str::to_string).unwrap_or_else(|| k.to_string()), k)).collect();
                if keys.is_empty() {
                    out.push_str("{}");
                    return;
                }
                if self.sort_keys {
                    keys.sort_by(|a, b| a.0.cmp(&b.0));
                }
                out.push('{');
                for (i, (name, key)) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_separator);
                    }
                    self.newline(level + 1, out);
                    self.write_str(name, out);
                    out.push_str(&self.key_separator);
                    self.write(&v.get_item(key).unwrap_or(Value::from(())), level + 1, out);
                }
                self.newline(level, out);
                out.push('}');
            }
            _ => self.write_str(&v.to_string(), out),
        }
    }

    /// Python `json.dumps` string escaping: `\`, `"`, and the six named control escapes get their
    /// short form; every other control char (< 0x20) gets `\u00xx`; non-ASCII is written through
    /// literally, or as `\uXXXX` (surrogate pairs above the BMP) when `ensure_ascii`.
    fn write_str(&self, s: &str, out: &mut String) {
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
                c if self.ensure_ascii && !c.is_ascii() => {
                    let mut units = [0u16; 2];
                    for unit in c.encode_utf16(&mut units) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

/// A float as `json.dumps` writes it: Python's shortest `repr` (`1.0`, `2.5`, `1e-05`, `1e+16`), with
/// `NaN` and `Infinity` for the values JSON proper cannot hold.
fn py_float_json(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let sign = if f.is_sign_negative() { "-" } else { "" };
    if f == 0.0 {
        return format!("{sign}0.0");
    }
    // `{:e}` yields the shortest digits that round-trip, as `d.ddde<exp>`.
    let scientific = format!("{:e}", f.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("`{:e}` of a finite float has an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes an integer exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let body = if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let whole = exponent as usize + 1;
            let padded = format!("{digits:0<whole$}");
            let (int, frac) = padded.split_at(whole);
            format!("{int}.{}", if frac.is_empty() { "0" } else { frac })
        } else {
            format!("0.{}{digits}", "0".repeat((-exponent - 1) as usize))
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let mantissa = if rest.is_empty() { first.to_string() } else { format!("{first}.{rest}") };
        format!("{mantissa}e{}{:02}", if exponent < 0 { '-' } else { '+' }, exponent.abs())
    };
    format!("{sign}{body}")
}

/// The `tojson` filter with `transformers`' signature: `indent`, `separators` (a pair of strings),
/// `sort_keys` and `ensure_ascii` keywords.
fn tojson_filter(v: Value, kwargs: minijinja::value::Kwargs) -> Result<String, minijinja::Error> {
    use minijinja::{Error, ErrorKind};
    let indent: Option<usize> = kwargs.get("indent")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let separators = match separators {
        None => None,
        Some(pair) => match <[String; 2]>::try_from(pair) {
            Ok([item, key]) => Some((item, key)),
            Err(_) => return Err(Error::new(ErrorKind::InvalidOperation, "tojson separators must be a pair of strings")),
        },
    };
    Ok(PyJson::new(indent, separators, sort_keys.unwrap_or(false), ensure_ascii.unwrap_or(false)).dumps(&v))
}

/// How hard a model is asked to reason, for a template that reads a `reasoning_effort` variable: a
/// name the template maps itself (`low`, `high`, ...) or a numeric level.
///
/// The template owns the names and refuses one it does not know, so a [`Self::Named`] value passes
/// through unchecked; the numeric level is validated here, at the boundary, against the range the
/// templates that take one declare.
#[derive(Clone, Debug, PartialEq)]
pub enum ReasoningEffort {
    Named(String),
    Level(f64),
}

impl ReasoningEffort {
    /// The template variable this value is passed as.
    pub const KWARG: &'static str = "reasoning_effort";
    /// The largest numeric level a template accepts.
    pub const MAX_LEVEL: f64 = 0.99;

    /// A numeric level in `0.0..=MAX_LEVEL`.
    ///
    /// # Errors
    /// The level is not finite or lies outside the range.
    pub fn level(level: f64) -> Result<ReasoningEffort, TemplateError> {
        if level.is_finite() && (0.0..=Self::MAX_LEVEL).contains(&level) {
            Ok(ReasoningEffort::Level(level))
        } else {
            Err(TemplateError::new(format!("reasoning_effort must be in [0.0, {}], got {level}", Self::MAX_LEVEL)))
        }
    }

    /// A request parameter's text: a number is a [`Self::level`], anything else a name.
    ///
    /// # Errors
    /// The text is empty, or a number outside the level range.
    pub fn parse(text: &str) -> Result<ReasoningEffort, TemplateError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(TemplateError::new("reasoning_effort is empty".to_string()));
        }
        match text.parse::<f64>() {
            Ok(level) => Self::level(level),
            Err(_) => Ok(ReasoningEffort::Named(text.to_string())),
        }
    }

    /// Set this as the `reasoning_effort` variable of a render's `extra` knobs.
    pub fn apply(self, extra: &mut BTreeMap<String, Value>) {
        extra.insert(Self::KWARG.to_string(), self.into());
    }
}

impl From<ReasoningEffort> for Value {
    fn from(effort: ReasoningEffort) -> Value {
        match effort {
            ReasoningEffort::Named(name) => Value::from(name),
            ReasoningEffort::Level(level) => Value::from(level),
        }
    }
}

/// A compiled chat template, ready to render. Cheap to keep around (one per
/// distinct `chat_template` string in use); NOT cheap to reconstruct per
/// call — `compile` parses the Jinja source.
/// Rewrite the `transformers`-only `generation` tags to the no-op block they
/// render as (see [`ChatTemplate::compile`]).
fn without_generation_tags(src: &str) -> String {
    let mut out = src.to_string();
    for (tag, block) in [("generation", "if true"), ("endgeneration", "endif")] {
        for lp in ["{%", "{%-"] {
            for rp in ["%}", "-%}"] {
                let from = format!("{lp} {tag} {rp}");
                let to = format!("{lp} {block} {rp}");
                out = out.replace(&from, &to);
                let from = format!("{lp}{tag}{rp}");
                let to = format!("{lp}{block}{rp}");
                out = out.replace(&from, &to);
            }
        }
    }
    out
}

pub struct ChatTemplate {
    env: Environment<'static>,
    /// Context defaults injected under every render, overridable per call via
    /// `extra` — today `bos_token`/`eos_token` parsed out of the same
    /// `tokenizer_config.json` the template came from ([`Self::from_model_dir`]).
    /// Before this existed those fields were parsed and DISCARDED, and a
    /// template's `{{ bos_token }}` rendered as the empty string silently —
    /// a prompt that does not byte-match HF's for the same checkpoint.
    defaults: BTreeMap<String, Value>,
    /// The template's own Jinja source, as compiled.
    source: String,
    /// [`Self::reads_argument_objects`], decided on first use (it may render a probe).
    argument_objects: std::sync::OnceLock<bool>,
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
    /// * `{% generation %}`/`{% endgeneration %}` are rewritten to a no-op
    ///   block: they are a `transformers` Jinja *extension* that marks the
    ///   assistant span of the rendered text for the returned loss mask and
    ///   emits nothing. minijinja has no such tag, so without the rewrite a
    ///   checkpoint shipping one (locally trained templates do) fails to
    ///   compile and the model silently loses its chat format. Rewriting to
    ///   `if`/`endif` keeps the tag's own whitespace control.
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
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .expect("the default delimiters with trim and lstrip are a valid syntax"),
        );
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
        env.add_filter("tojson", tojson_filter);
        env.add_template_owned("chat", without_generation_tags(jinja_src)).map_err(|e| TemplateError::new(format!("{e:#}")))?;
        Ok(ChatTemplate { env, defaults: BTreeMap::new(), source: jinja_src.to_string(), argument_objects: std::sync::OnceLock::new() })
    }

    /// Compile the `chat_template` field out of `<model_dir>/tokenizer_config.json`
    /// -- for a base model fetched through brain's normal store/plan path,
    /// this file is ALREADY on disk (`modelstore::plan`'s `plan_base`
    /// downloads it alongside `tokenizer.json` whenever the upstream repo
    /// ships one), so this needs no separate import-time wiring.
    ///
    /// **Prefers a standalone template file** to `tokenizer_config.json`'s
    /// inline `chat_template` (and uses one when the config has none or is
    /// itself absent): `chat_template.jinja` (a bare Jinja file, no JSON
    /// wrapper) first, then `chat_template.json` (`{"chat_template": "..."}`).
    /// Real, current `transformers` resolution: recently-published
    /// checkpoints moved the template OUT of `tokenizer_config.json` into one
    /// of these two standalone files - a checkpoint that ships only
    /// `chat_template.json` is common now, not an edge case (confirmed
    /// against a real Qwen3-Omni-30B-A3B-Instruct checkpoint on disk, whose
    /// `tokenizer_config.json` has no `chat_template` key at all).
    ///
    /// Returns a `TemplateError` naming every location checked (rather than
    /// panicking) when none of them has a template - a checkpoint with no
    /// template is a real, expected case (e.g. a base, non-instruction-tuned
    /// model), not a bug to crash on.
    pub fn from_model_dir(dir: &std::path::Path) -> Result<ChatTemplate, TemplateError> {
        let tcfg_path = dir.join("tokenizer_config.json");
        // Best-effort: bos_token/eos_token defaults below still apply even
        // when this file exists but has no chat_template key, and a missing
        // file is not fatal on its own -- a standalone chat_template.json/
        // .jinja may still supply the template.
        let tcfg: Option<serde_json::Value> = std::fs::read_to_string(&tcfg_path).ok().and_then(|text| serde_json::from_str(&text).ok());

        let inline = tcfg.as_ref().and_then(|c| c.get("chat_template")).and_then(|v| v.as_str()).map(str::to_string);
        let json_path = dir.join("chat_template.json");
        let jinja_path = dir.join("chat_template.jinja");
        // The standalone files win over the config's inline string, as in `transformers`: a release that ships
        // `chat_template.jinja` may leave an older, simpler template in `tokenizer_config.json` (Ripple's has no
        // tools), and the file is the one its authors render with.
        let from_jinja = std::fs::read_to_string(&jinja_path).ok();
        let from_json = || {
            std::fs::read_to_string(&json_path)
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|v| v.get("chat_template").and_then(|v| v.as_str()).map(str::to_string))
        };
        let src = from_jinja.or_else(from_json).or(inline).ok_or_else(|| {
            TemplateError::new(format!(
                "no chat template found: checked \"chat_template\" in {} and {}, and the bare file {}",
                tcfg_path.display(),
                json_path.display(),
                jinja_path.display()
            ))
        })?;
        let mut t = Self::compile(&src)?;
        // The SAME tokenizer_config.json declares the special tokens many
        // templates reference (`{{ bos_token }}` heads Llama/Mistral prompts)
        // - inject them as render defaults instead of parsing and discarding
        // them. HF encodes them either as a bare string or as an AddedToken
        // object with a `content` field; both are read, absent stays absent
        // (chainable undefined then matches transformers' own behaviour).
        if let Some(cfg) = &tcfg {
            for key in ["bos_token", "eos_token"] {
                let tok = cfg.get(key).and_then(|v| v.as_str().or_else(|| v.get("content").and_then(|c| c.as_str())));
                if let Some(s) = tok {
                    t.defaults.insert(key.to_string(), Value::from(s.to_string()));
                }
            }
        }
        Ok(t)
    }

    /// Compile the `tokenizer.chat_template` field out of a GGUF's embedded
    /// tokenizer KV - the sibling of [`Self::from_model_dir`] for a `.gguf`
    /// checkpoint that carries no `tokenizer_config.json` at all (a `.gguf`
    /// file is commonly the ONLY artifact on disk; see `brain-data`'s own
    /// `checkpoint::gguf::GgufTokenizer::chat_template` doc for the KV key).
    ///
    /// Errors the same shape as `from_model_dir` when no template is present
    /// (a base, non-instruction-tuned GGUF is a real, expected case, not a
    /// bug). `from_model_dir` sources its `bos_token`/`eos_token` render
    /// defaults from `tokenizer_config.json`'s own string fields; a GGUF has
    /// no such sibling file, so the equivalent here is `gt`'s own `bos`/`eos`
    /// token IDS mapped through `gt.tokens` to their literal text - the same
    /// information, read from the one place a GGUF actually carries it.
    /// Best-effort exactly like `from_model_dir`: an out-of-range or absent
    /// id simply leaves that default unset rather than erroring.
    pub fn from_gguf(gt: &checkpoint::gguf::GgufTokenizer) -> Result<ChatTemplate, TemplateError> {
        let src = gt.chat_template.as_ref().ok_or_else(|| {
            TemplateError::new(format!(
                "no chat template found: GgufTokenizer has no \"tokenizer.chat_template\" key (model = {:?})",
                gt.model
            ))
        })?;
        let mut t = Self::compile(src)?;
        for (key, id) in [("bos_token", gt.bos), ("eos_token", gt.eos)] {
            if let Some(text) = id.and_then(|i| gt.tokens.get(i as usize)) {
                t.defaults.insert(key.to_string(), Value::from(text.clone()));
            }
        }
        Ok(t)
    }

    /// Whether the template reads a tool call's `arguments` as an object
    /// (`arguments|items`, as Qwen3.8's does) rather than passing the JSON text
    /// through (Qwen3's `arguments is string` branch). The messages handed to
    /// it must then carry the arguments parsed, as `transformers` does for a
    /// served request, or rendering fails on the first tool call.
    ///
    /// A template that serializes the arguments itself (`arguments | tojson`) with no branch for text
    /// (`arguments is string`) reads objects too: handed the JSON text it would encode it a second time. So does one
    /// that refuses the text outright, whatever it names the value (Inkling 2's `args is string` raises): a probe
    /// call renders with the arguments as an object and fails with them as text.
    #[must_use]
    pub fn reads_argument_objects(&self) -> bool {
        *self.argument_objects.get_or_init(|| {
            let src = &self.source;
            let iterates = src.contains("arguments|items") || src.contains("arguments | items");
            let serializes = (src.contains("arguments|tojson") || src.contains("arguments | tojson")) && !src.contains("arguments is string");
            iterates || serializes || self.refuses_argument_text()
        })
    }

    /// Whether a call whose arguments are an object renders and the same call with them as JSON text does not.
    fn refuses_argument_text(&self) -> bool {
        let renders = |arguments: serde_json::Value| {
            let call = serde_json::json!({"id": "call0", "type": "function", "function": {"name": "probe", "arguments": arguments}});
            let turns = serde_json::json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [call]}]);
            parse_json_ordered(&turns.to_string()).is_ok_and(|messages| self.render(messages, None, false, &BTreeMap::new()).is_ok())
        };
        renders(serde_json::json!({"a": 1})) && !renders(serde_json::json!("{\"a\":1}"))
    }

    /// Whether this template writes an assistant turn with one framing newline between its parts, the layout
    /// `<think>\n{reasoning}\n</think>\n{content}` and `{content}\n<tool_call>`, read off a probe conversation.
    /// A reply generated in that layout is scanned back with the framing newlines removed
    /// ([`crate::qwen_chat::TemplateFlavor::JsonFramed`]), or rendering the parsed message again would add them a
    /// second time on every turn.
    #[must_use]
    pub fn frames_replies_with_newlines(&self) -> bool {
        let render = |assistant: &str| {
            let probe = parse_json_ordered(&format!(r#"[{{"role":"user","content":"q"}},{assistant}]"#)).ok()?;
            self.render(probe, None, false, &BTreeMap::new()).ok()
        };
        let reasoned = render(r#"{"role":"assistant","reasoning_content":"PROBE-REASONING","content":"PROBE-CONTENT"}"#);
        let calling = render(r#"{"role":"assistant","content":"PROBE-CONTENT","tool_calls":[{"id":"c","type":"function","function":{"name":"probe_tool","arguments":{"a":1}}}]}"#);
        let framed = |text: &Option<String>, needle: &str| text.as_deref().is_some_and(|t| t.contains(needle));
        framed(&reasoned, "<think>\nPROBE-REASONING\n</think>\nPROBE-CONTENT") && framed(&calling, "PROBE-CONTENT\n<tool_call>\n")
    }

    /// The reply format this template writes an assistant turn in, when brain reads that format back
    /// ([`crate::qwen_chat::ChatScanner`]): typed messages ([`crate::message_scan`]), a `<function=` call (Qwen3.8), else a JSON `<tool_call>` or a think block
    /// (Qwen3, or [`crate::qwen_chat::TemplateFlavor::JsonFramed`] when [`Self::frames_replies_with_newlines`]).
    /// `None` for a template that writes neither.
    #[must_use]
    pub fn reply_flavor(&self) -> Option<crate::qwen_chat::TemplateFlavor> {
        use crate::qwen_chat::TemplateFlavor;
        let src = &self.source;
        if crate::message_scan::is_written_by(src) {
            Some(TemplateFlavor::TypedMessages)
        } else if src.contains("<function=") {
            Some(TemplateFlavor::Qwen38)
        } else if src.contains("<tool_call>") || src.contains("<think>") {
            Some(if self.frames_replies_with_newlines() { TemplateFlavor::JsonFramed } else { TemplateFlavor::Qwen3 })
        } else {
            None
        }
    }

    /// The tokens that end a generation under this template beyond the checkpoint's own eos. A template that
    /// writes typed messages ends the whole reply with its sampling eos ([`crate::message_scan::END_SAMPLING`]), and
    /// only that stops a generation: its message boundary ([`crate::message_scan::END_MESSAGE`]) closes every message
    /// of a reply, the thinking message before the answer and each tool call included, so stopping there would cut a
    /// reply after its first message. Empty for a template whose turn the checkpoint's eos ends.
    #[must_use]
    pub fn end_of_turn_markers(&self) -> Vec<&'static str> {
        match self.reply_flavor() {
            Some(crate::qwen_chat::TemplateFlavor::TypedMessages) => vec![crate::message_scan::END_SAMPLING],
            _ => Vec::new(),
        }
    }

    /// Whether `generated` (the text a sampler generated for an answer) is the same answer as `rendered` (this
    /// template's rendering of the answer's parsed message), read back by the template's reply format
    /// ([`Self::reply_flavor`]): the same reasoning and the same content up to the whitespace around them, and the same
    /// calls (names, and arguments compared as JSON values, so `{"a": 1}` and `{"a":1}` agree), with no byte of either
    /// discarded by the reader (a call's trailing text). Each may end with the
    /// end-of-turn marker. `thinking_open`: the prompt before the answer opened the think block. `false` for a template
    /// whose replies brain does not read: there only [`Tokenizer`](crate::tokenizer::Tokenizer)-exact ids are admitted.
    #[must_use]
    pub fn reads_as_same_reply(&self, rendered: &str, generated: &str, thinking_open: bool) -> bool {
        let Some(flavor) = self.reply_flavor() else { return false };
        let Ok(end) = self.assistant_turn_end() else { return false };
        let end = end.trim_end();
        let read = |text: &str| {
            let text = text.strip_suffix(end).unwrap_or(text);
            let mut scanner = crate::qwen_chat::ChatScanner::with_flavor(thinking_open, flavor);
            let mut events = Vec::new();
            scanner.push(text, &mut events);
            scanner.finish(&mut events);
            let calls: Vec<(String, Result<serde_json::Value, String>)> = scanner
                .tool_calls()
                .iter()
                .map(|c| (c.name.clone(), serde_json::from_str::<serde_json::Value>(&c.arguments).map_err(|_| c.arguments.clone())))
                .collect();
            // Every anomaly but a discarded call tail keeps its bytes as reasoning or content, where they compare.
            let kept_every_byte = !scanner.anomalies().contains(&crate::qwen_chat::ScanAnomaly::DiscardedToolCallTail);
            (scanner.reasoning().trim().to_string(), scanner.content().trim().to_string(), calls, kept_every_byte)
        };
        let (a, b) = (read(rendered), read(generated));
        a.3 && b.3 && (a.0, a.1, a.2) == (b.0, b.1, b.2)
    }

    /// The conversation of a served request as this template reads it: `turns` are the request's messages as
    /// JSON (a tool call's `arguments` the JSON text a client sent), and the result is the ordered value to
    /// render, with each call's arguments parsed (key order kept) when [`Self::reads_argument_objects`].
    /// Arguments that are not JSON stay text, as `transformers` leaves them.
    ///
    /// # Errors
    /// The messages cannot be turned into template values.
    pub fn conversation(&self, turns: &[serde_json::Value]) -> Result<Value, TemplateError> {
        let turns: Vec<serde_json::Value> = turns.iter().map(with_openai_calls).collect();
        if !self.reads_argument_objects() {
            return parse_json_ordered(&serde_json::Value::Array(turns).to_string());
        }
        Ok(Value::from(turns.iter().map(|turn| with_argument_objects(turn, false)).collect::<Vec<_>>()))
    }

    /// Render `messages` (a JSON array of `{role, content, tool_calls?,
    /// tool_call_id?, ...}` objects — build via [`parse_json_ordered`] or
    /// `Value::from(Serde(..))`) through the template. `tools` is the JSON
    /// schema array (or `None` for no tool preamble). `extra` carries any
    /// further template kwargs a specific model's template reads
    /// (`enable_thinking`, `bos_token`, …) — passed through verbatim, so
    /// this stays generic across templates with different kwarg needs.
    /// The Jinja source this template was compiled from.
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn render(&self, messages: Value, tools: Option<Value>, add_generation_prompt: bool, extra: &BTreeMap<String, Value>) -> Result<String, TemplateError> {
        let tmpl = self.env.get_template("chat").map_err(|e| TemplateError::new(format!("{e:#}")))?;
        let mut ctx: BTreeMap<String, Value> = self.defaults.clone();
        ctx.extend(extra.clone());
        ctx.insert("messages".to_string(), messages);
        // Absent tools are `none`, exactly as `apply_chat_template` passes
        // them — the old `[]` stand-in flipped every `tools is not none`
        // branch into emitting a tool preamble for a tool-less call.
        ctx.insert("tools".to_string(), tools.unwrap_or_else(|| Value::from(())));
        ctx.insert("add_generation_prompt".to_string(), Value::from(add_generation_prompt));
        tmpl.render(ctx).map_err(|e| TemplateError::new(format!("{e:#}")))
    }

    /// The generation prompt for `messages` asking the model not to reason.
    ///
    /// Models reach no-think mode three ways, and this is the one place that
    /// knows them: a template with a switch is asked to turn it off
    /// (`enable_thinking`, Qwen3's; `thinking`, DeepSeek-V3.1's); a template
    /// that opens a reasoning block and has no switch (DeepSeek-R1-Distill's
    /// ends its generation prompt inside `<think>`) has the block closed
    /// empty, which is how that model is meant to be run without reasoning;
    /// and a template that never reasons renders as it always does.
    pub fn render_no_think(&self, messages: Value, tools: Option<Value>, extra: &BTreeMap<String, Value>) -> Result<String, TemplateError> {
        let mut extra = extra.clone();
        extra.extend(no_think_switches());
        let mut text = self.render(messages, tools, true, &extra)?;
        if text.trim_end_matches('\n').ends_with("<think>") {
            text.push_str("\n</think>\n\n");
        }
        Ok(text)
    }

    /// What a no-think prompt puts after the assistant token, from the first
    /// reasoning tag on (`<think>\n\n</think>\n\n`, or `</think>`): the block
    /// a model asked not to reason is trained to follow. `None` when the
    /// template's no-think prompt has no such tag - a model that never reasons.
    #[must_use]
    pub fn no_think_block(&self) -> Option<String> {
        let probe = || parse_json_ordered(r#"[{"role":"user","content":"x"}]"#).ok();
        // Plain is rendered with the switches off too: a template that adds
        // reasoning instructions to the system turn while thinking is on
        // (Qwen3.8's) would otherwise differ from the asked prompt before the
        // assistant turn, and no block would be found.
        let plain = self.render(probe()?, None, false, &no_think_switches()).ok()?;
        let asked = self.render_no_think(probe()?, None, &BTreeMap::new()).ok()?;
        let tail = asked.strip_prefix(plain.as_str())?;
        let at = ["<think>", "</think>"].iter().filter_map(|tag| tail.find(tag)).min()?;
        Some(tail[at..].to_string())
    }

    /// The text this template writes after an assistant turn's content to end
    /// the turn (Qwen3's `<|im_end|>\n`), read off a probe conversation.
    ///
    /// # Errors
    /// The template does not render a plain assistant turn followed by anything.
    pub fn assistant_turn_end(&self) -> Result<String, TemplateError> {
        const CONTENT: &str = "probe-answer";
        let probe = parse_json_ordered(&format!(r#"[{{"role":"user","content":"probe-question"}},{{"role":"assistant","content":"{CONTENT}"}}]"#))
            .map_err(|e| TemplateError::new(format!("end-of-turn probe: {e}")))?;
        let text = self.render(probe, None, false, &BTreeMap::new())?;
        match text.rfind(CONTENT).map(|at| &text[at + CONTENT.len()..]) {
            Some(end) if !end.is_empty() => Ok(end.to_string()),
            _ => Err(TemplateError::new("the template writes nothing after an assistant turn, so turns cannot be told apart".to_string())),
        }
    }

    /// Render the whole conversation `messages` as a training record, without a generation prompt, under the
    /// reasoning switches inference uses (`thinking` false turns them off, as [`Self::render_no_think`] does).
    pub fn render_conversation(&self, messages: Value, tools: Option<Value>, thinking: bool) -> Result<String, TemplateError> {
        let switches = if thinking { BTreeMap::new() } else { no_think_switches() };
        self.render(messages, tools, false, &switches)
    }

    /// Render a conversation whose LAST message is the one supervised, and
    /// find the single byte boundary between what a model is asked from and
    /// what it is trained to generate.
    ///
    /// The context is `messages[..len-1]` rendered with the generation
    /// prompt, under the same reasoning flags inference uses (`thinking`
    /// false is [`Self::render_no_think`], whose pre-filled reasoning block is
    /// therefore context). The full render must start with exactly those
    /// bytes; the supervised span is what follows, i.e. the final message as
    /// the template renders it as the last turn plus its end-of-turn token.
    ///
    /// Unlike [`Self::render_with_message_boundaries`] this needs no boundary
    /// for any earlier message, so a template that renders a turn differently
    /// as the last message than in the middle of a conversation (Qwen3's
    /// reasoning block) does not make a tool-call turn in the context
    /// unmaskable. Returns the full text and the supervised byte range.
    ///
    /// # Errors
    /// `messages` is empty, or the full render does not start with the
    /// context's prompt (the template renders the context differently with
    /// and without the final message): no boundary is guessed.
    pub fn render_with_supervised_suffix(&self, messages: &[Value], tools: Option<Value>, thinking: bool) -> Result<(String, Range<usize>), TemplateError> {
        let Some((_, context)) = messages.split_last() else {
            return Err(TemplateError::new("a supervised suffix needs at least one message".to_string()));
        };
        let context = Value::from(context.to_vec());
        let prompt = if thinking { self.render(context, tools.clone(), true, &BTreeMap::new())? } else { self.render_no_think(context, tools.clone(), &BTreeMap::new())? };
        // The answer is rendered with the switches the prompt was asked with:
        // a template whose system turn depends on them (Qwen3.8's reasoning
        // instructions) would not otherwise extend the prompt.
        let switches = if thinking { BTreeMap::new() } else { no_think_switches() };
        let full = self.render(Value::from(messages.to_vec()), tools, false, &switches)?;
        if !full.as_bytes().starts_with(prompt.as_bytes()) {
            return Err(TemplateError::new(format!(
                "the final message does not follow the prompt the model is asked from: the template renders the conversation so far ({} bytes with the generation prompt) differently with and without it",
                prompt.len()
            )));
        }
        let supervised = prompt.len()..full.len();
        Ok((full, supervised))
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
    /// boundary re-renders `messages[..=i]`, so a 40-turn trajectory costs an
    /// order of magnitude more than the single-render work, per sample, over the
    /// whole training set.
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
    /// exists. A conversation that hits this hazard is trained through
    /// [`crate::chat::ChatSample::answers_as_asked`] instead, which renders
    /// each supervised turn after its own prefix and so needs no boundary.
    pub fn render_with_message_boundaries(&self, messages: &[Value], tools: Option<Value>) -> Result<(String, Vec<Range<usize>>), TemplateError> {
        let full = self.render(Value::from(messages.to_vec()), tools.clone(), false, &BTreeMap::new())?;
        let mut boundaries = Vec::with_capacity(messages.len());
        let mut prev_len = 0usize;
        for i in 0..messages.len() {
            let prefix_msgs: Vec<Value> = messages[..=i].to_vec();
            let prefix = match self.render(Value::from(prefix_msgs), tools.clone(), false, &BTreeMap::new()) {
                Ok(prefix) => prefix,
                // A template that needs a user turn (Qwen3.8's: "No user query
                // found") cannot render the system turn that precedes it
                // alone. Its text is attributed to the first message that
                // renders with it; no message that can be supervised is ever
                // folded into another.
                Err(_) if is_leading_system(messages, i) => {
                    boundaries.push(prev_len..prev_len);
                    continue;
                }
                Err(e) => return Err(e),
            };
            if !full.as_bytes().starts_with(prefix.as_bytes()) {
                return Err(TemplateError::new(format!(
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
            return Err(TemplateError::new(format!(
                "message boundaries sum to {prev_len} bytes but the full render is {} bytes",
                full.len()
            )));
        }
        Ok((full, boundaries))
    }
}

/// Whether message `i` is a system or developer turn with only such turns
/// before it: the instructions that open a conversation, never supervised.
fn is_leading_system(messages: &[Value], i: usize) -> bool {
    messages[..=i].iter().all(|m| matches!(m.get_attr("role").ok().as_ref().and_then(Value::as_str), Some("system" | "developer")))
}

/// The context variables that turn reasoning off in the templates that have a
/// switch: `enable_thinking` (Qwen3, Qwen3.8) and `thinking` (DeepSeek-V3.1).
/// `node` as a template value, with the `arguments` of every tool call under it parsed as JSON where they are
/// JSON text. `in_call` marks `node` as a tool call (or its `function`), whose `arguments` field is the one parsed.
/// `turn` with every tool call in the OpenAI shape `transformers` renders (`{id, type, function: {name, arguments}}`),
/// the shape [`ChatMessage`](crate::chat::ChatMessage) hands a template too. The serving contract carries a call flat
/// (`{id, name, arguments}`); a template that reads `tool_call.function` (Inkling 2's, Mistral's) finds nothing in
/// that, while one written for both (Qwen's `if tool_call.function`) reads either. A call already in the OpenAI shape
/// is kept as sent.
fn with_openai_calls(turn: &serde_json::Value) -> serde_json::Value {
    let Some(calls) = turn.get("tool_calls").and_then(serde_json::Value::as_array) else { return turn.clone() };
    let lifted: Vec<serde_json::Value> = calls
        .iter()
        .map(|call| match call.as_object() {
            Some(flat) if !flat.contains_key("function") && flat.contains_key("name") => {
                let mut function = serde_json::Map::new();
                function.insert("name".into(), flat["name"].clone());
                function.insert("arguments".into(), flat.get("arguments").cloned().unwrap_or_else(|| serde_json::Value::from("{}")));
                let mut lifted = serde_json::Map::new();
                if let Some(id) = flat.get("id") {
                    lifted.insert("id".into(), id.clone());
                }
                lifted.insert("type".into(), serde_json::Value::from("function"));
                lifted.insert("function".into(), serde_json::Value::Object(function));
                serde_json::Value::Object(lifted)
            }
            _ => call.clone(),
        })
        .collect();
    let mut turn = turn.clone();
    turn["tool_calls"] = serde_json::Value::Array(lifted);
    turn
}

fn with_argument_objects(node: &serde_json::Value, in_call: bool) -> Value {
    let Some(object) = node.as_object() else { return Value::from(minijinja::value::Serde(node)) };
    let fields: BTreeMap<String, Value> = object
        .iter()
        .map(|(key, value)| {
            let converted = match (key.as_str(), value) {
                ("tool_calls", serde_json::Value::Array(calls)) => Value::from(calls.iter().map(|call| with_argument_objects(call, true)).collect::<Vec<_>>()),
                ("function", _) if in_call => with_argument_objects(value, true),
                ("arguments", serde_json::Value::String(text)) if in_call => parse_json_ordered(text).unwrap_or_else(|_| Value::from(text.as_str())),
                _ => Value::from(minijinja::value::Serde(value)),
            };
            (key.clone(), converted)
        })
        .collect();
    Value::from(minijinja::value::Serde(&fields))
}

fn no_think_switches() -> BTreeMap<String, Value> {
    BTreeMap::from([("enable_thinking".to_string(), Value::from(false)), ("thinking".to_string(), Value::from(false))])
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

    /// `transformers`' `{% generation %}` extension marks the assistant span
    /// and emits nothing; a template shipping it must render exactly like the
    /// same template with the markers removed - same text, same newlines.
    #[test]
    fn generation_tags_render_as_nothing() {
        let plain = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<A>{% endif %}";
        let marked = "{% for m in messages %}<{{ m.role }}>{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}{% if add_generation_prompt %}<A>{% endif %}";
        let msgs = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        let a = ChatTemplate::compile(plain).unwrap().render(msgs.clone(), None, true, &BTreeMap::new()).unwrap();
        let b = ChatTemplate::compile(marked).unwrap().render(msgs, None, true, &BTreeMap::new()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, "<user>hi<A>");
    }

    fn tiny_env() -> ChatTemplate {
        ChatTemplate::compile(
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
        )
        .unwrap()
    }

    /// A template in the shape of Qwen3.8's: with thinking on (the default) the
    /// system turn gains an instruction, and the assistant turn opens a reasoning
    /// block; with it off the prompt closes an empty block instead.
    fn switching_env() -> ChatTemplate {
        ChatTemplate::compile(
            "{%- set on = enable_thinking is undefined or enable_thinking is true -%}\
             {%- for m in messages %}{%- if loop.first and on %}<|system|>THINK-HARD<|end|>\n{%- endif %}\
             <|{{ m.role }}|>{{ '<think>\\n\\n</think>\\n\\n' if m.role == 'assistant' }}{{ m.content }}<|end|>\n{%- endfor %}\
             {%- if add_generation_prompt %}<|assistant|>{{ '<think>\\n' if on else '<think>\\n\\n</think>\\n\\n' }}{%- endif %}",
        )
        .unwrap()
    }

    /// The block a model asked not to reason is trained to follow is found even
    /// when the template's thinking-on render differs earlier in the prompt.
    #[test]
    fn the_no_think_block_is_found_when_thinking_changes_the_system_turn() {
        assert_eq!(switching_env().no_think_block().as_deref(), Some("<think>\n\n</think>\n\n"));
    }

    /// An answer trained for no-think mode follows the prompt a no-think request
    /// renders, though the thinking-on render of the same conversation differs.
    #[test]
    fn a_no_think_answer_follows_the_no_think_prompt() {
        let tmpl = switching_env();
        let user = parse_json_ordered(r#"{"role":"user","content":"hi"}"#).unwrap();
        let answer = parse_json_ordered(r#"{"role":"assistant","content":"yo"}"#).unwrap();
        let (full, supervised) = tmpl.render_with_supervised_suffix(&[user.clone(), answer], None, false).unwrap();
        let prompt = tmpl.render_no_think(Value::from(vec![user]), None, &BTreeMap::new()).unwrap();
        assert!(!prompt.contains("THINK-HARD"), "{prompt}");
        assert_eq!(&full[..supervised.start], prompt);
        assert_eq!(&full[supervised], "yo<|end|>");
    }

    /// A served request's tool calls arrive in the contract shape (`{id, name, arguments}`, the arguments the JSON
    /// text the client sent). A template that reads the OpenAI shape (`tool_call.function.name`) and refuses
    /// arguments as text (it iterates them itself; Inkling 2's) is handed `{id, type, function: {name, arguments}}`
    /// with the arguments parsed, key order kept, as `transformers` is by an OpenAI server; one that takes the text
    /// (Qwen3's `arguments is string` branch) still gets the client's bytes.
    #[test]
    fn a_served_tool_call_reaches_the_template_in_the_shape_it_reads() {
        let turns = [
            serde_json::json!({"role": "user", "content": "q"}),
            serde_json::json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get", "arguments": "{\"b\":2,\"a\":1}"}]}),
        ];
        let strict = ChatTemplate::compile(
            "{%- for m in messages -%}{%- for tc in m.tool_calls or [] -%}{%- set fn = tc.function -%}\
             {%- if fn.name is not defined or fn.name is not string -%}{{ raise_exception('tool call function name must be a string') }}{%- endif -%}\
             {%- set args = fn.arguments -%}{%- if args is string -%}{{ raise_exception('arguments must be a parsed object') }}{%- endif -%}\
             [{{ tc.id }}:{{ fn.name }}({% for k, v in args | dictsort %}{{ k }}={{ v }};{% endfor %})]{%- endfor -%}\
             <{{ m.role }}>{{ m.content }}{%- endfor -%}",
        )
        .unwrap();
        assert!(strict.reads_argument_objects());
        let rendered = strict.render(strict.conversation(&turns).unwrap(), None, false, &BTreeMap::new()).unwrap();
        assert_eq!(rendered, "<user>q[c1:get(a=1;b=2;)]<assistant>");

        let textual = ChatTemplate::compile(
            "{%- for m in messages -%}{%- for tc in m.tool_calls or [] -%}{%- if tc.function %}{%- set tc = tc.function %}{%- endif -%}\
             {{ tc.name }}:{%- if tc.arguments is string %}{{ tc.arguments }}{%- else %}{{ tc.arguments | tojson }}{%- endif -%}{%- endfor -%}\
             <{{ m.role }}>{%- endfor -%}",
        )
        .unwrap();
        assert!(!textual.reads_argument_objects());
        let rendered = textual.render(textual.conversation(&turns).unwrap(), None, false, &BTreeMap::new()).unwrap();
        assert_eq!(rendered, "<user>get:{\"b\":2,\"a\":1}<assistant>");
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

    /// `tojson(...)` keywords as Python's `json.dumps` reads them, and floats as its `repr` writes them.
    #[test]
    fn tojson_takes_the_transformers_keywords_and_writes_python_floats() {
        let render = |expr: &str, json: &str| {
            let t = ChatTemplate::compile(&format!("{{{{ obj | tojson{expr} }}}}")).unwrap();
            let ctx = BTreeMap::from([("obj".to_string(), parse_json_ordered(json).unwrap())]);
            t.render(Value::from(Vec::<Value>::new()), None, false, &ctx).unwrap()
        };
        let obj = r#"{"b": [1, 2.0, {}], "a": "\u00e5\ud83d\ude00", "c": []}"#;
        assert_eq!(render("(sort_keys=true, separators=(',', ':'))", obj), r#"{"a":"å😀","b":[1,2.0,{}],"c":[]}"#);
        assert_eq!(render("(indent=2)", obj), "{\n  \"b\": [\n    1,\n    2.0,\n    {}\n  ],\n  \"a\": \"å😀\",\n  \"c\": []\n}");
        assert_eq!(render("(ensure_ascii=true)", r#""å😀""#), r#""\u00e5\ud83d\ude00""#);
        for (written, text) in [(1e-5, "1e-05"), (0.0001, "0.0001"), (1e16, "1e+16"), (1.5e300, "1.5e+300"), (123456789012345.0, "123456789012345.0"), (-0.0, "-0.0"), (0.1, "0.1"), (1e15, "1000000000000000.0")] {
            assert_eq!(py_float_json(written), text);
        }
        let t = ChatTemplate::compile("{{ obj | tojson(sort_key=true) }}").unwrap();
        let ctx = BTreeMap::from([("obj".to_string(), Value::from(1))]);
        assert!(t.render(Value::from(Vec::<Value>::new()), None, false, &ctx).is_err(), "a misspelled keyword is refused, not ignored");
    }

    #[test]
    fn reasoning_effort_is_a_name_or_a_level_in_range() {
        assert_eq!(ReasoningEffort::parse(" high ").unwrap(), ReasoningEffort::Named("high".to_string()));
        assert_eq!(ReasoningEffort::parse("0.25").unwrap(), ReasoningEffort::Level(0.25));
        for bad in ["", "1", "-0.1", "0.995", "NaN", "inf"] {
            assert!(ReasoningEffort::parse(bad).is_err(), "{bad:?}");
        }
        let mut extra = BTreeMap::new();
        ReasoningEffort::Level(0.5).apply(&mut extra);
        assert_eq!(extra["reasoning_effort"].as_str(), None);
        assert_eq!(f64::try_from(extra["reasoning_effort"].clone()).unwrap(), 0.5);
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

    /// Qwen3.8's template refuses a conversation without a user turn, so the
    /// system turn that opens a record cannot be rendered alone to find its
    /// boundary; its text rides with the first message that renders.
    #[test]
    fn a_leading_system_turn_a_template_cannot_render_alone_joins_the_next_message() {
        let t = ChatTemplate::compile(
            "{% set ns = namespace(user=false) %}{% for m in messages %}{% if m.role == 'user' %}{% set ns.user = true %}{% endif %}{% endfor %}\
             {% if not ns.user %}{{ raise_exception('No user query found in messages.') }}{% endif %}\
             {% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}",
        )
        .unwrap();
        let messages: Vec<Value> = vec![
            parse_json_ordered(r#"{"role":"system","content":"rules"}"#).unwrap(),
            parse_json_ordered(r#"{"role":"user","content":"hi"}"#).unwrap(),
            parse_json_ordered(r#"{"role":"assistant","content":"hey"}"#).unwrap(),
        ];
        let (full, ranges) = t.render_with_message_boundaries(&messages, None).unwrap();
        assert_eq!(full, "<|system|>rules<|user|>hi<|assistant|>hey");
        assert_eq!(&full[ranges[0].clone()], "");
        assert_eq!(&full[ranges[1].clone()], "<|system|>rules<|user|>hi");
        assert_eq!(&full[ranges[2].clone()], "<|assistant|>hey");

        // A conversation that never has a user turn is still refused.
        let err = t.render_with_message_boundaries(&messages[..1], None).unwrap_err();
        assert!(err.to_string().contains("No user query"), "got: {err}");
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

    /// SPEC: a checkpoint whose `tokenizer_config.json` has NO inline
    /// `chat_template` field (real, current `transformers` convention for
    /// recently-published checkpoints - confirmed against a real
    /// Qwen3-Omni-30B-A3B-Instruct checkpoint on disk) still loads its
    /// template from a standalone `chat_template.json` (`{"chat_template":
    /// "..."}`) beside it, and bos/eos defaults from `tokenizer_config.json`
    /// still apply (the two files are independent, both read).
    #[test]
    fn from_model_dir_falls_back_to_standalone_chat_template_json() {
        let dir = std::env::temp_dir().join(format!("brain-chat-template-standalone-json-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A trimmed shape of the real checkpoint's tokenizer_config.json:
        // present, has bos/eos, but genuinely no "chat_template" key.
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"bos_token": "<s>", "eos_token": {"content": "</s>"}, "unrelated": true}"#).unwrap();
        std::fs::write(dir.join("chat_template.json"), r#"{"chat_template": "{{ bos_token }}<|{{ messages[0].role }}|>{{ messages[0].content }}{{ eos_token }}"}"#).unwrap();
        let t = ChatTemplate::from_model_dir(&dir).expect("load");
        let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        assert_eq!(t.render(messages, None, false, &BTreeMap::new()).unwrap(), "<s><|user|>hi</s>");
    }

    /// SPEC: a checkpoint shipping a BARE `chat_template.jinja` (no JSON
    /// wrapper at all - the other real `transformers` convention) loads from
    /// that file when neither `tokenizer_config.json` nor `chat_template.json`
    /// has the template.
    #[test]
    fn from_model_dir_falls_back_to_bare_chat_template_jinja() {
        let dir = std::env::temp_dir().join(format!("brain-chat-template-standalone-jinja-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"unrelated": true}"#).unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "<|{{ messages[0].role }}|>{{ messages[0].content }}").unwrap();
        let t = ChatTemplate::from_model_dir(&dir).expect("load");
        let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        assert_eq!(t.render(messages, None, false, &BTreeMap::new()).unwrap(), "<|user|>hi");
    }

    /// SPEC: the standalone files outrank `tokenizer_config.json`'s inline string, `chat_template.jinja` first,
    /// as in `transformers`. Ripple ships a tools-aware `chat_template.jinja` beside a `tokenizer_config.json`
    /// whose inline template is a plain ChatML loop with no tools and no reasoning.
    #[test]
    fn from_model_dir_prefers_the_standalone_template_files_to_the_inline_one() {
        let dir = std::env::temp_dir().join(format!("brain-chat-template-precedence-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"chat_template": "inline"}"#).unwrap();
        let render = || {
            let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
            ChatTemplate::from_model_dir(&dir).expect("load").render(messages, None, false, &BTreeMap::new()).unwrap()
        };
        assert_eq!(render(), "inline");
        std::fs::write(dir.join("chat_template.json"), r#"{"chat_template": "json"}"#).unwrap();
        assert_eq!(render(), "json");
        std::fs::write(dir.join("chat_template.jinja"), "jinja").unwrap();
        assert_eq!(render(), "jinja");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SPEC: with NO `tokenizer_config.json` at all, a standalone
    /// `chat_template.json` alone is still enough -- the file is read
    /// best-effort (bos/eos defaults simply stay unset), not required.
    #[test]
    fn from_model_dir_works_with_no_tokenizer_config_json_at_all() {
        let dir = std::env::temp_dir().join(format!("brain-chat-template-no-tcfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("chat_template.json"), r#"{"chat_template": "<|{{ messages[0].role }}|>{{ messages[0].content }}"}"#).unwrap();
        let t = ChatTemplate::from_model_dir(&dir).expect("load");
        let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        assert_eq!(t.render(messages, None, false, &BTreeMap::new()).unwrap(), "<|user|>hi");
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

    /// Build a real, minimal GGUF carrying a `tokenizer.chat_template` (plus
    /// bos/eos ids and a tiny vocab) via `checkpoint::gguf_write::write`, then
    /// read it back through `MmapGguf::open` - real KV parsing end to end,
    /// not a hand-built `GgufTokenizer`, mirroring `from_model_dir`'s own
    /// tests building real fixture files rather than mocking `ChatTemplate`.
    fn gguf_with_chat_template(path: &str, template: &str) {
        use checkpoint::gguf::GgufValue;
        let kv = vec![
            ("tokenizer.ggml.model".to_string(), GgufValue::String("gpt2".to_string())),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(vec![GgufValue::String("<s>".to_string()), GgufValue::String("hi".to_string()), GgufValue::String("</s>".to_string())]),
            ),
            ("tokenizer.ggml.bos_token_id".to_string(), GgufValue::U32(0)),
            ("tokenizer.ggml.eos_token_id".to_string(), GgufValue::U32(2)),
            ("tokenizer.chat_template".to_string(), GgufValue::String(template.to_string())),
        ];
        checkpoint::gguf_write::write(
            path,
            &kv,
            &[checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![1], ty: 0, data: 1.0f32.to_le_bytes().to_vec() }],
            32,
        )
        .unwrap();
    }

    /// SPEC: `ChatTemplate::from_gguf` compiles the real `tokenizer.chat_template`
    /// KV string and injects `bos_token`/`eos_token` render defaults sourced
    /// from `gt.tokens[gt.bos]`/`gt.tokens[gt.eos]` - the GGUF-native
    /// equivalent of `from_model_dir_injects_bos_and_eos_tokens`.
    #[test]
    fn from_gguf_compiles_template_and_injects_bos_eos_from_the_vocab() {
        let path = std::env::temp_dir().join(format!("brain-chat-template-gguf-{}.gguf", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        gguf_with_chat_template(&path, "{{ bos_token }}{{ messages[0].content }}{{ eos_token }}");

        let mg = checkpoint::gguf::MmapGguf::open(&path).unwrap();
        let gt = mg.tokenizer().expect("tokenizer present");
        let t = ChatTemplate::from_gguf(&gt).expect("load");
        let messages = parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        assert_eq!(t.render(messages.clone(), None, false, &BTreeMap::new()).unwrap(), "<s>hi</s>");
        // extra overrides a default, exactly as it does for from_model_dir.
        let mut extra = BTreeMap::new();
        extra.insert("bos_token".to_string(), Value::from("<B>"));
        assert_eq!(t.render(messages, None, false, &extra).unwrap(), "<B>hi</s>");

        std::fs::remove_file(&path).ok();
    }

    /// SPEC: a GGUF with no `tokenizer.chat_template` key errors clearly
    /// (mirrors `from_model_dir_errors_clearly_when_chat_template_is_absent`)
    /// rather than panicking or silently compiling an empty template.
    #[test]
    fn from_gguf_errors_clearly_when_chat_template_is_absent() {
        let path = std::env::temp_dir().join(format!("brain-chat-template-gguf-absent-{}.gguf", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        checkpoint::gguf_write::write(
            &path,
            &[("tokenizer.ggml.model".to_string(), checkpoint::gguf::GgufValue::String("gpt2".to_string()))],
            &[checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![1], ty: 0, data: 1.0f32.to_le_bytes().to_vec() }],
            32,
        )
        .unwrap();

        let mg = checkpoint::gguf::MmapGguf::open(&path).unwrap();
        let gt = mg.tokenizer().expect("tokenizer present");
        let Err(err) = ChatTemplate::from_gguf(&gt) else { panic!("expected an error") };
        assert!(err.to_string().contains("chat_template"), "got: {err}");

        std::fs::remove_file(&path).ok();
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

    const R1_STYLE: &str = "{% for m in messages %}{% if m.role == 'user' %}<U>{{ m.content }}{% else %}<A>{{ m.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}<A><think>\n{% endif %}";
    const QWEN3_STYLE: &str = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}";
    const V31_STYLE: &str = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<A>{% if thinking is defined and thinking is true %}<think>{% else %}</think>{% endif %}{% endif %}";
    const PLAIN: &str = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";

    /// Every way the models in use are put in no-think mode, by one call: a
    /// template with a switch (`enable_thinking`, `thinking`) is asked to turn
    /// it off; one that opens a reasoning block and has no switch is closed
    /// empty, which is how its model is meant to be run without reasoning; a
    /// template that never reasons is left as it is.
    #[test]
    fn a_template_renders_a_prompt_that_asks_for_no_reasoning() {
        let messages = || parse_json_ordered(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        let off = |src: &str| ChatTemplate::compile(src).unwrap().render_no_think(messages(), None, &BTreeMap::new()).unwrap();
        assert_eq!(off(R1_STYLE), "<user>hi<A><think>\n\n</think>\n\n".replace("<user>", "<U>"));
        assert_eq!(off(QWEN3_STYLE), "<user>hi<assistant><think>\n\n</think>\n\n");
        assert_eq!(off(V31_STYLE), "<user>hi<A></think>");
        assert_eq!(off(PLAIN), "<user>hi<assistant>");
    }

    /// The block a no-think prompt puts after the assistant token, which a
    /// model asked without reasoning is trained to follow: `None` for a model
    /// that never reasons.
    #[test]
    fn the_no_think_block_is_what_a_no_think_prompt_adds_after_the_assistant_token() {
        let block = |src: &str| ChatTemplate::compile(src).unwrap().no_think_block();
        assert_eq!(block(R1_STYLE).as_deref(), Some("<think>\n\n</think>\n\n"));
        assert_eq!(block(QWEN3_STYLE).as_deref(), Some("<think>\n\n</think>\n\n"));
        assert_eq!(block(V31_STYLE).as_deref(), Some("</think>"));
        assert_eq!(block(PLAIN), None);
    }
}
