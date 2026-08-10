// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generalized model-capability interface — the single shape every brain model
//! exposes its actions through, so the **CLI** (`brain do …`) and the **event
//! API** (`ActionRequest`/`ActionResult`) dispatch them *generically*.
//!
//! The whole design in four types:
//! * [`Provider`] — a loaded model, advertising a [`Manifest`] of the actions it
//!   supports and handing back an [`Action`] by name.
//! * [`ActionSpec`] — a self-describing schema for one action (typed
//!   [`ParamSpec`]s plus binary [`BlobSpec`] inputs/outputs), serializable so a
//!   host can discover
//!   and drive it without hard-coding anything.
//! * [`Invocation`] → [`Outcome`] — one call: typed params (`serde_json`) + named
//!   binary blobs in, scalar outputs + named blobs out.
//! * [`Registry`] — the shared dispatcher the CLI and the runtime both use:
//!   register providers, list manifests, validate + run an action by
//!   `(model, action)`.
//!
//! Adding a capability to brain is implementing [`Action`] and listing it in a
//! [`Provider`] — no new CLI subcommand, no new `events::Event` variant. That is
//! the point: one generalized interface, every model, every action.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

pub mod blob;

// ===================== descriptors (the self-describing schema) =====================

/// The type of a scalar action parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamType {
    Str,
    Int,
    Float,
    Bool,
    /// One of a fixed set of string values.
    Enum(Vec<String>),
}

impl ParamType {
    pub fn name(&self) -> &'static str {
        match self {
            ParamType::Str => "str",
            ParamType::Int => "int",
            ParamType::Float => "float",
            ParamType::Bool => "bool",
            ParamType::Enum(_) => "enum",
        }
    }
}

/// A typed, optionally-defaulted scalar parameter of an action.
#[derive(Clone, Debug)]
pub struct ParamSpec {
    pub name: String,
    pub ty: ParamType,
    pub required: bool,
    pub default: Option<Value>,
    pub help: String,
}

impl ParamSpec {
    pub fn new(name: &str, ty: ParamType, help: &str) -> ParamSpec {
        ParamSpec { name: name.into(), ty, required: false, default: None, help: help.into() }
    }
    pub fn required(mut self) -> ParamSpec {
        self.required = true;
        self
    }
    pub fn default(mut self, v: Value) -> ParamSpec {
        self.default = Some(v);
        self
    }
    fn to_json(&self) -> Value {
        json!({
            "name": self.name, "type": self.ty.name(),
            "required": self.required, "default": self.default,
            "help": self.help,
            "values": match &self.ty { ParamType::Enum(v) => json!(v), _ => Value::Null },
        })
    }
}

/// The kind of a binary input/output — enough for a host to pick a codec.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Media {
    Image,
    Mask,
    Audio,
    Text,
    Bytes,
}

impl Media {
    pub fn name(&self) -> &'static str {
        match self {
            Media::Image => "image",
            Media::Mask => "mask",
            Media::Audio => "audio",
            Media::Text => "text",
            Media::Bytes => "bytes",
        }
    }
    pub fn parse(s: &str) -> Option<Media> {
        Some(match s {
            "image" => Media::Image,
            "mask" => Media::Mask,
            "audio" => Media::Audio,
            "text" => Media::Text,
            "bytes" => Media::Bytes,
            _ => return None,
        })
    }
}

/// A named binary input or output (an image, a mask, audio, …).
#[derive(Clone, Debug)]
pub struct BlobSpec {
    pub name: String,
    pub media: Media,
    pub required: bool,
    pub help: String,
}

impl BlobSpec {
    pub fn new(name: &str, media: Media, help: &str) -> BlobSpec {
        BlobSpec { name: name.into(), media, required: false, help: help.into() }
    }
    pub fn required(mut self) -> BlobSpec {
        self.required = true;
        self
    }
    fn to_json(&self) -> Value {
        json!({ "name": self.name, "media": self.media.name(), "required": self.required, "help": self.help })
    }
}

/// The complete, self-describing schema of one action.
#[derive(Clone, Debug)]
pub struct ActionSpec {
    pub name: String,
    pub summary: String,
    pub params: Vec<ParamSpec>,
    pub inputs: Vec<BlobSpec>,
    pub outputs: Vec<BlobSpec>,
    /// Whether the action emits progress updates while running.
    pub streaming: bool,
}

impl ActionSpec {
    pub fn new(name: &str, summary: &str) -> ActionSpec {
        ActionSpec { name: name.into(), summary: summary.into(), params: Vec::new(), inputs: Vec::new(), outputs: Vec::new(), streaming: false }
    }
    pub fn param(mut self, p: ParamSpec) -> ActionSpec {
        self.params.push(p);
        self
    }
    pub fn input(mut self, b: BlobSpec) -> ActionSpec {
        self.inputs.push(b);
        self
    }
    pub fn output(mut self, b: BlobSpec) -> ActionSpec {
        self.outputs.push(b);
        self
    }
    pub fn streaming(mut self) -> ActionSpec {
        self.streaming = true;
        self
    }
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name, "summary": self.summary, "streaming": self.streaming,
            "params": self.params.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
            "inputs": self.inputs.iter().map(|b| b.to_json()).collect::<Vec<_>>(),
            "outputs": self.outputs.iter().map(|b| b.to_json()).collect::<Vec<_>>(),
        })
    }

    /// Validate + normalize an invocation against this spec: reject unknown or
    /// missing-required params/blobs and bad enum/type values, fill defaults.
    /// Returns the normalized [`Invocation`] ready for [`Action::run`].
    pub fn validate(&self, inv: Invocation) -> Result<Invocation, String> {
        let mut params = inv.params.as_object().cloned().unwrap_or_default();
        // reject unknown params
        for k in params.keys() {
            if !self.params.iter().any(|p| &p.name == k) {
                return Err(format!("unknown param '{k}' for action '{}'", self.name));
            }
        }
        // type-check + defaults + required
        for p in &self.params {
            match params.get(&p.name) {
                Some(v) => check_type(&p.name, &p.ty, v)?,
                None => {
                    if let Some(d) = &p.default {
                        params.insert(p.name.clone(), d.clone());
                    } else if p.required {
                        return Err(format!("missing required param '{}' for action '{}'", p.name, self.name));
                    }
                }
            }
        }
        // required blobs present, correct media
        for b in &self.inputs {
            match inv.blobs.get(&b.name) {
                Some(blob) => {
                    if blob.media != b.media {
                        return Err(format!("input '{}' expects {} but got {}", b.name, b.media.name(), blob.media.name()));
                    }
                }
                None if b.required => return Err(format!("missing required input '{}' for action '{}'", b.name, self.name)),
                None => {}
            }
        }
        for k in inv.blobs.keys() {
            if !self.inputs.iter().any(|b| &b.name == k) {
                return Err(format!("unknown input '{k}' for action '{}'", self.name));
            }
        }
        Ok(Invocation { params: Value::Object(params), blobs: inv.blobs, cancel: inv.cancel })
    }
}

fn check_type(name: &str, ty: &ParamType, v: &Value) -> Result<(), String> {
    let ok = match ty {
        ParamType::Str => v.is_string(),
        ParamType::Int => v.is_i64() || v.is_u64(),
        ParamType::Float => v.is_number(),
        ParamType::Bool => v.is_boolean(),
        ParamType::Enum(vals) => v.as_str().map(|s| vals.iter().any(|x| x == s)).unwrap_or(false),
    };
    if ok {
        Ok(())
    } else {
        Err(format!("param '{name}' must be {} (got {v})", ty.name()))
    }
}

/// A model's advertised set of actions.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub model: String,
    pub summary: String,
    pub actions: Vec<ActionSpec>,
    /// Maximum tokens (prompt + completion) this instance can actually serve
    /// right now, when known. Populated by chat-capable resident models from
    /// their REAL configured engine capacity (e.g. the paged KV-cache sizing
    /// derived from `BRAIN_QWEN_CTX`) — deliberately NOT the checkpoint's
    /// architectural `max_position_embeddings`, which the engine may not
    /// actually be able to hold (this is what advertising the wrong number
    /// looks like: a client builds a prompt the model claims to support, the
    /// server admits it, then rejects it after already paying the connection
    /// setup cost). `None` for non-chat models, or when not yet known.
    pub max_context_tokens: Option<u64>,
}

impl Manifest {
    pub fn new(model: &str, summary: &str, actions: Vec<ActionSpec>) -> Manifest {
        Manifest { model: model.into(), summary: summary.into(), actions, max_context_tokens: None }
    }
    /// Builder setter for [`Self::max_context_tokens`]. Kept as a separate
    /// setter (rather than a `new()` parameter) so the other ~35 existing
    /// `Manifest::new()` call sites across non-chat model kinds are untouched.
    pub fn with_max_context_tokens(mut self, tokens: u64) -> Manifest {
        self.max_context_tokens = Some(tokens);
        self
    }
    pub fn to_json(&self) -> Value {
        let mut v = json!({ "model": self.model, "summary": self.summary, "actions": self.actions.iter().map(|a| a.to_json()).collect::<Vec<_>>() });
        if let Some(t) = self.max_context_tokens {
            v["max_context_tokens"] = json!(t);
        }
        v
    }
}

// ===================== runtime values =====================

/// A binary payload (image bytes, mask, audio, …) plus free-form metadata
/// (e.g. `{"w":512,"h":512}` for a raw image, or `{"format":"png"}`).
#[derive(Clone, Debug)]
pub struct Blob {
    pub media: Media,
    pub bytes: Vec<u8>,
    pub meta: Value,
}

impl Blob {
    pub fn new(media: Media, bytes: Vec<u8>) -> Blob {
        Blob { media, bytes, meta: Value::Null }
    }
    /// Re-tag the media kind. `blob::image_blob` encodes brain's ONE image wire
    /// format and tags it [`Media::Image`]; a mask uses the identical format and
    /// differs only in the tag, so it is a re-tag rather than a second encoder.
    pub fn with_media(mut self, media: Media) -> Blob {
        self.media = media;
        self
    }
    pub fn with_meta(mut self, meta: Value) -> Blob {
        self.meta = meta;
        self
    }
}

/// A cooperative cancellation flag riding in an [`Invocation`]. A `Default` token
/// is unarmed and never cancelled, so existing construction sites are unaffected.
/// A front-end that wants to abort arms one ([`CancelToken::armed`]), puts it in
/// the invocation, keeps a clone, and later calls [`CancelToken::cancel`].
/// Long-running actions poll [`CancelToken::is_cancelled`] between steps and
/// return `Err("cancelled".into())`.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Option<Arc<AtomicBool>>);

impl CancelToken {
    /// An armed token: any clone can request — and observe — cancellation.
    pub fn armed() -> CancelToken {
        CancelToken(Some(Arc::new(AtomicBool::new(false))))
    }
    /// Request cancellation (a no-op on an unarmed `Default` token).
    pub fn cancel(&self) {
        if let Some(f) = &self.0 {
            f.store(true, Ordering::Relaxed);
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.as_ref().map(|f| f.load(Ordering::Relaxed)).unwrap_or(false)
    }
}

/// The inputs to one action call.
#[derive(Clone, Debug, Default)]
pub struct Invocation {
    pub params: Value,
    pub blobs: BTreeMap<String, Blob>,
    /// Cooperative cancellation; `Default` (unarmed) is never cancelled.
    pub cancel: CancelToken,
}

impl Invocation {
    pub fn new() -> Invocation {
        Invocation { params: json!({}), blobs: BTreeMap::new(), cancel: CancelToken::default() }
    }
    pub fn set(mut self, key: &str, v: Value) -> Invocation {
        self.params.as_object_mut().expect("params must be an object").insert(key.into(), v);
        self
    }
    pub fn blob(mut self, key: &str, b: Blob) -> Invocation {
        self.blobs.insert(key.into(), b);
        self
    }
    pub fn get_str(&self, key: &str) -> Option<String> {
        self.params.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    }
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(|v| v.as_f64())
    }
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.params.get(key).and_then(|v| v.as_i64())
    }
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.params.get(key).and_then(|v| v.as_bool())
    }
    pub fn get_blob(&self, key: &str) -> Option<&Blob> {
        self.blobs.get(key)
    }
}

/// The last user turn from a flattened `messages` JSON array
/// (`[{"role":...,"content":...}, ...]`), falling back to the last message's
/// content, then the bare `prompt` param.
///
/// This is the ONLY path real OpenAI/Anthropic HTTP traffic exercises (both
/// chat handlers always populate `messages`, never a bare `prompt`; `prompt`
/// is a convenience for D-Bus/direct callers). It lives here ONCE — it used
/// to be copy-pasted character-for-character into omni, qwenvl and the mock
/// resident, each copy annotated "kept in sync deliberately"; per the
/// hoist-and-migrate policy a shared extraction cannot silently disagree
/// between models the way three hand-synced copies can.
pub fn last_user_text(inv: &Invocation) -> String {
    if let Some(s) = inv.get_str("messages") {
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&s) {
            for m in arr.iter().rev() {
                if m.get("role").and_then(|v| v.as_str()) == Some("user") {
                    if let Some(c) = m.get("content").and_then(|v| v.as_str()) {
                        return c.to_string();
                    }
                }
            }
            if let Some(c) = arr.last().and_then(|m| m.get("content")).and_then(|v| v.as_str()) {
                return c.to_string();
            }
        }
    }
    inv.get_str("prompt").unwrap_or_default()
}

/// The result of one action call.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub outputs: Value,
    pub blobs: BTreeMap<String, Blob>,
}

impl Outcome {
    pub fn new() -> Outcome {
        Outcome { outputs: json!({}), blobs: BTreeMap::new() }
    }
    pub fn set(mut self, key: &str, v: Value) -> Outcome {
        self.outputs.as_object_mut().expect("outputs must be an object").insert(key.into(), v);
        self
    }
    pub fn blob(mut self, key: &str, b: Blob) -> Outcome {
        self.blobs.insert(key.into(), b);
        self
    }
}

/// A progress update emitted while a streaming action runs. `delta` carries a
/// per-token text fragment for streaming generation (`None` for plain step
/// progress); a front-end appends deltas to reconstruct the running output.
/// `event` carries a structured, out-of-band progress payload — e.g. a
/// tool-call event surfaced mid-generation — for updates that don't fit the
/// plain-text `delta` shape; `None` for ordinary step/token progress.
#[derive(Clone, Debug)]
pub struct Progress {
    pub step: u32,
    pub total: u32,
    pub message: String,
    pub delta: Option<String>,
    pub event: Option<serde_json::Value>,
}

impl Progress {
    /// A plain step update (no token payload).
    pub fn step(step: u32, total: u32, message: impl Into<String>) -> Progress {
        Progress { step, total, message: message.into(), delta: None, event: None }
    }
    /// A streaming token: `text` is the new fragment carried in `delta`.
    pub fn token(step: u32, total: u32, text: impl Into<String>) -> Progress {
        Progress { step, total, message: "token".into(), delta: Some(text.into()), event: None }
    }
    /// A structured out-of-band progress payload (e.g. a tool-call event
    /// surfaced during streaming generation), carried in `event`.
    pub fn event(step: u32, total: u32, v: serde_json::Value) -> Progress {
        Progress { step, total, message: "event".into(), delta: None, event: Some(v) }
    }
}

pub type ActionResult = Result<Outcome, String>;

// ===================== the traits a model implements =====================

/// One executable capability.
pub trait Action: Send + Sync {
    fn spec(&self) -> ActionSpec;
    /// Run the action. `progress` is invoked for streaming updates (ignore it for
    /// one-shot actions). The invocation is already validated against [`Action::spec`].
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult;
}

/// A loaded model advertising a set of actions.
pub trait Provider: Send + Sync {
    fn manifest(&self) -> Manifest;
    fn action(&self, name: &str) -> Option<Arc<dyn Action>>;
}

// ===================== the shared dispatcher =====================

/// Holds the loaded providers and dispatches actions by `(model, action)` — the
/// one path the CLI and the event runtime both go through.
#[derive(Default)]
pub struct Registry {
    providers: Vec<Arc<dyn Provider>>,
}

impl Registry {
    pub fn new() -> Registry {
        Registry { providers: Vec::new() }
    }
    pub fn register(&mut self, p: Arc<dyn Provider>) {
        self.providers.push(p);
    }
    /// Every registered model's manifest (for discovery).
    pub fn manifests(&self) -> Vec<Manifest> {
        self.providers.iter().map(|p| p.manifest()).collect()
    }
    pub fn provider(&self, model: &str) -> Option<&Arc<dyn Provider>> {
        self.providers.iter().find(|p| p.manifest().model == model)
    }
    /// Resolve an action by `(model, action)`.
    pub fn find(&self, model: &str, action: &str) -> Option<Arc<dyn Action>> {
        self.provider(model)?.action(action)
    }
    /// Validate the invocation against the action's spec and run it. This is the
    /// single entry point the CLI and the runtime call.
    pub fn run(&self, model: &str, action: &str, inv: Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let act = self.find(model, action).ok_or_else(|| format!("no action '{action}' on model '{model}'"))?;
        let inv = act.spec().validate(inv)?;
        act.run(&inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    impl Action for Echo {
        fn spec(&self) -> ActionSpec {
            ActionSpec::new("echo", "echo a string N times")
                .param(ParamSpec::new("text", ParamType::Str, "the text").required())
                .param(ParamSpec::new("times", ParamType::Int, "repeat count").default(json!(1)))
                .param(ParamSpec::new("mode", ParamType::Enum(vec!["upper".into(), "lower".into()]), "case").default(json!("lower")))
                .output(BlobSpec::new("result", Media::Text, "the echoed text"))
        }
        fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
            let text = inv.get_str("text").unwrap();
            let n = inv.get_i64("times").unwrap() as usize;
            let up = inv.get_str("mode").unwrap() == "upper";
            let s = if up { text.to_uppercase() } else { text.to_lowercase() };
            progress(Progress::step(1, 1, "echoing"));
            let out = s.repeat(n);
            Ok(Outcome::new().set("len", json!(out.len())).blob("result", Blob::new(Media::Text, out.into_bytes())))
        }
    }
    struct EchoModel;
    impl Provider for EchoModel {
        fn manifest(&self) -> Manifest {
            Manifest::new("echo-model", "a trivial demo model", vec![Echo.spec()])
        }
        fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
            (name == "echo").then(|| Arc::new(Echo) as Arc<dyn Action>)
        }
    }

    #[test]
    fn manifest_serializes() {
        let j = EchoModel.manifest().to_json();
        assert_eq!(j["model"], "echo-model");
        assert_eq!(j["actions"][0]["name"], "echo");
        assert_eq!(j["actions"][0]["params"][2]["values"][0], "upper");
    }

    #[test]
    fn validate_fills_defaults_and_checks() {
        let spec = Echo.spec();
        // defaults filled
        let inv = spec.validate(Invocation::new().set("text", json!("Hi"))).unwrap();
        assert_eq!(inv.get_i64("times"), Some(1));
        assert_eq!(inv.get_str("mode").as_deref(), Some("lower"));
        // missing required
        assert!(spec.validate(Invocation::new()).is_err());
        // unknown param
        assert!(spec.validate(Invocation::new().set("text", json!("x")).set("bogus", json!(1))).is_err());
        // bad enum
        assert!(spec.validate(Invocation::new().set("text", json!("x")).set("mode", json!("weird"))).is_err());
        // wrong type
        assert!(spec.validate(Invocation::new().set("text", json!(5))).is_err());
    }

    /// A "long-running" action that polls the invocation's cancel token each step.
    struct Poller;
    impl Action for Poller {
        fn spec(&self) -> ActionSpec {
            ActionSpec::new("poll", "polls the cancel token per step").streaming()
        }
        fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
            for step in 0..100 {
                if inv.cancel.is_cancelled() {
                    return Err("cancelled".into());
                }
                progress(Progress::step(step, 100, "step"));
            }
            Ok(Outcome::new())
        }
    }

    #[test]
    fn cancel_token_default_is_never_cancelled() {
        let t = CancelToken::default();
        assert!(!t.is_cancelled());
        t.cancel(); // a no-op on an unarmed token
        assert!(!t.is_cancelled());
        // an invocation's default token is likewise inert
        assert!(!Invocation::new().cancel.is_cancelled());
    }

    #[test]
    fn armed_token_cancels_across_clones_and_validate() {
        let t = CancelToken::armed();
        let handle = t.clone();
        assert!(!t.is_cancelled());
        handle.cancel();
        assert!(t.is_cancelled());
        // validate() carries the token through to the normalized invocation
        let mut inv = Invocation::new();
        inv.cancel = t;
        let inv = Poller.spec().validate(inv).unwrap();
        assert!(inv.cancel.is_cancelled());
    }

    #[test]
    fn polling_action_aborts_when_cancelled() {
        // uncancelled: runs to completion
        let inv = Poller.spec().validate(Invocation::new()).unwrap();
        assert!(Poller.run(&inv, &mut |_| {}).is_ok());
        // cancelled mid-run (from the progress callback, as a client would over a stream)
        let mut inv = Invocation::new();
        inv.cancel = CancelToken::armed();
        let handle = inv.cancel.clone();
        let inv = Poller.spec().validate(inv).unwrap();
        let mut steps = 0u32;
        let err = Poller
            .run(&inv, &mut |p| {
                steps += 1;
                if p.step == 3 {
                    handle.cancel();
                }
            })
            .unwrap_err();
        assert_eq!(err, "cancelled");
        assert!(steps < 100, "action must abort early, ran {steps} steps");
    }

    #[test]
    fn registry_dispatches_and_runs() {
        let mut reg = Registry::new();
        reg.register(Arc::new(EchoModel));
        assert_eq!(reg.manifests().len(), 1);
        let mut steps = 0;
        let out = reg
            .run("echo-model", "echo", Invocation::new().set("text", json!("Ab")).set("times", json!(3)).set("mode", json!("upper")), &mut |_p| steps += 1)
            .unwrap();
        assert_eq!(steps, 1);
        assert_eq!(out.outputs["len"], 6);
        assert_eq!(String::from_utf8(out.blobs["result"].bytes.clone()).unwrap(), "ABABAB");
        // unknown model / action
        assert!(reg.run("nope", "echo", Invocation::new(), &mut |_| {}).is_err());
        assert!(reg.run("echo-model", "nope", Invocation::new(), &mut |_| {}).is_err());
    }
}
