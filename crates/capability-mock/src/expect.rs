// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A declarative, expectation-based mock for [`capability::Provider`] - the
//! FFF/CMock/pytest-`unittest.mock` shape applied to brain's capability
//! contract: a test says exactly what a `(model, action)` call should
//! return (or fail with), and [`Mock::verify`] reports which expectations
//! were never met.
//!
//! This exists alongside [`crate::MockProvider`], not instead of it:
//! `MockProvider` answers "give me SOMETHING shaped like this model's real
//! output, with no test-authored content" (mirroring a manifest, inferring
//! shape); [`Mock`] answers "run this exact scripted scenario and prove the
//! caller's logic handles it correctly" - the OCR node returns THIS
//! markdown, the extraction node returns THIS JSON, the third page fails
//! with THIS error. A fully-mocked run through `MockProvider` alone can
//! only ever prove a downstream deterministic gate rejects placeholder
//! text (`MockProvider`'s text output is an echo of the prompt); it can
//! never prove the gate accepts a REAL record, because nothing can script
//! what a real record looks like. [`Mock`] is what makes that provable,
//! weight-free and fast.
//!
//! # Example
//!
//! ```
//! use capability_mock::{Mock, MockBlob};
//! use capability::{ActionSpec, BlobSpec, Manifest, Media, Registry};
//! use serde_json::json;
//!
//! let manifest = Manifest::new(
//!     "vendor/ocr",
//!     "test",
//!     vec![ActionSpec::new("generate", "").output(BlobSpec::new("text", Media::Text, ""))],
//! );
//!
//! let mock = Mock::new();
//! mock.on("vendor/ocr", "generate")
//!     .returns_blob("text", MockBlob::text("# Invoice 4711"));
//!
//! let registry = mock.registry_from(vec![manifest]).unwrap();
//! let out = registry.run("vendor/ocr", "generate", capability::Invocation::new(), &mut |_| {}).unwrap();
//! assert_eq!(out.blobs["text"].bytes, b"# Invoice 4711");
//!
//! mock.verify().unwrap();
//! ```
//!
//! # What this does NOT do
//!
//! No async (matches [`capability::Action::run`]'s synchronous contract), no
//! real inference, no RNG, no wall-clock delay, no cross-`(model, action)`
//! call ordering (each rule's `at_call` ordinal counts calls to that ONE
//! action only - whale-shaped callers dispatch independent graph nodes
//! concurrently, so a global sequence would be nondeterministic by
//! construction), and no scenario-document (YAML/JSON) loader yet - this is
//! the Rust builder layer only.

use std::panic::Location;
use std::sync::{Arc, Mutex};

use capability::blob::{image_blob, video_blob};
use capability::{Action, ActionResult, ActionSpec, Blob, Invocation, Manifest, Media, Outcome, Progress, Provider, Registry};
use serde_json::{json, Value};

use crate::synth::{gradient_hwc, run_steps, sine_pcm, video_frame_hwc};

/// Content a [`Mock`] rule hands back, in place of a real model's output.
/// Every variant terminates in the same encoders brain's real capabilities
/// and [`crate::MockProvider`] already use - one wire format, not a second
/// one invented for tests.
#[derive(Clone, Debug)]
pub enum MockBlob {
    /// UTF-8 text, verbatim.
    Text(String),
    /// A JSON value, serialized and tagged [`Media::Text`] (brain's
    /// convention for a structured-output action: JSON travels as a text
    /// blob, decoded downstream).
    Json(Value),
    /// Raw bytes under a caller-chosen [`Media`] tag, for anything the other
    /// variants don't cover.
    Bytes { media: Media, bytes: Vec<u8> },
    /// A deterministic gradient image, [`capability::blob::image_blob`]-encoded.
    Image { w: u32, h: u32, seed: u32 },
    /// The same gradient, tagged [`Media::Mask`].
    Mask { w: u32, h: u32, seed: u32 },
    /// N frames of a moving-gradient clip, [`capability::blob::video_blob`]-encoded.
    Video { frames: u32, w: u32, h: u32, fps: f32, seed: u32 },
    /// A sine tone as raw f32-LE PCM.
    Audio { seconds: f32, sample_rate: u32, channels: u32, seed: u32 },
}

impl MockBlob {
    pub fn text(s: impl Into<String>) -> MockBlob {
        MockBlob::Text(s.into())
    }
    pub fn json(v: Value) -> MockBlob {
        MockBlob::Json(v)
    }
    pub fn bytes(media: Media, bytes: Vec<u8>) -> MockBlob {
        MockBlob::Bytes { media, bytes }
    }
    /// A deterministic image at `seed` `0` - pass the same `w`/`h` twice with
    /// different [`Self::image_seeded`] seeds to get two distinguishable images.
    pub fn image(w: u32, h: u32) -> MockBlob {
        MockBlob::Image { w, h, seed: 0 }
    }
    pub fn image_seeded(w: u32, h: u32, seed: u32) -> MockBlob {
        MockBlob::Image { w, h, seed }
    }
    pub fn mask(w: u32, h: u32) -> MockBlob {
        MockBlob::Mask { w, h, seed: 0 }
    }
    pub fn video(frames: u32, w: u32, h: u32, fps: f32) -> MockBlob {
        MockBlob::Video { frames, w, h, fps, seed: 0 }
    }
    pub fn audio(seconds: f32, sample_rate: u32, channels: u32) -> MockBlob {
        MockBlob::Audio { seconds, sample_rate, channels, seed: 0 }
    }

    /// The [`Media`] this blob is tagged with on the wire - what
    /// [`Mock::bind`] checks against the target action's declared output.
    pub fn media(&self) -> Media {
        match self {
            MockBlob::Text(_) | MockBlob::Json(_) => Media::Text,
            MockBlob::Bytes { media, .. } => *media,
            MockBlob::Image { .. } => Media::Image,
            MockBlob::Mask { .. } => Media::Mask,
            MockBlob::Video { .. } => Media::Video,
            MockBlob::Audio { .. } => Media::Audio,
        }
    }

    fn build(&self) -> Result<Blob, String> {
        Ok(match self {
            MockBlob::Text(s) => Blob::new(Media::Text, s.clone().into_bytes()),
            MockBlob::Json(v) => Blob::new(Media::Text, serde_json::to_vec(v).map_err(|e| format!("mock: encoding json blob: {e}"))?),
            MockBlob::Bytes { media, bytes } => Blob::new(*media, bytes.clone()),
            MockBlob::Image { w, h, seed } => image_blob(&gradient_hwc(*seed, *w, *h), *w, *h, 3),
            MockBlob::Mask { w, h, seed } => image_blob(&gradient_hwc(*seed, *w, *h), *w, *h, 3).with_media(Media::Mask),
            MockBlob::Video { frames, w, h, fps, seed } => {
                let frames: Vec<(Vec<f32>, u32, u32)> = (0..(*frames).max(1)).map(|f| (video_frame_hwc(*seed, *w, *h, f), *w, *h)).collect();
                let mut b = video_blob(&frames)?;
                let mut meta = b.meta.clone();
                meta["fps"] = json!(fps);
                b = b.with_meta(meta);
                b
            }
            MockBlob::Audio { seconds, sample_rate, channels, seed } => {
                Blob::new(Media::Audio, sine_pcm(*seed, *seconds, *sample_rate, *channels)).with_meta(json!({ "sample_rate": sample_rate, "channels": channels }))
            }
        })
    }
}

/// One condition a call must satisfy for a [`Rule`] to apply. All predicates
/// on a rule are AND-ed together.
#[derive(Clone, Debug)]
enum Predicate {
    TextContains(String),
    ParamEq(String, Value),
    ParamContains(String, String),
    BlobPresent(String),
    BlobMediaIs(String, Media),
    /// 1-based ordinal of this call among every call to the SAME
    /// `(model, action)` pair - never a global sequence, since concurrent
    /// dispatch across different actions is a real, expected caller shape.
    AtCall(u32),
}

impl Predicate {
    fn matches(&self, inv: &Invocation, call_index: u32) -> bool {
        match self {
            Predicate::TextContains(s) => capability::last_user_text(inv).contains(s.as_str()),
            Predicate::ParamEq(name, expected) => inv.params.get(name).is_some_and(|got| got == expected),
            Predicate::ParamContains(name, s) => inv.get_str(name).is_some_and(|got| got.contains(s.as_str())),
            Predicate::BlobPresent(name) => inv.get_blob(name).is_some(),
            Predicate::BlobMediaIs(name, media) => inv.get_blob(name).is_some_and(|b| b.media == *media),
            Predicate::AtCall(n) => call_index == *n,
        }
    }
    fn describe(&self) -> String {
        match self {
            Predicate::TextContains(s) => format!("text contains {s:?}"),
            Predicate::ParamEq(name, v) => format!("param '{name}' == {v}"),
            Predicate::ParamContains(name, s) => format!("param '{name}' contains {s:?}"),
            Predicate::BlobPresent(name) => format!("blob '{name}' present"),
            Predicate::BlobMediaIs(name, media) => format!("blob '{name}' is {}", media.name()),
            Predicate::AtCall(n) => format!("is call #{n}"),
        }
    }
}

#[derive(Clone, Debug)]
enum Response {
    Blob { output: Option<String>, blob: MockBlob },
    Fail(String),
}

/// How many times a rule is allowed/required to be consumed, checked by
/// [`Mock::verify`].
#[derive(Clone, Copy, Debug)]
enum Times {
    /// No expectation - matches as many times as its predicates allow.
    Any,
    /// Must be consumed exactly this many times by the time [`Mock::verify`] runs.
    Exactly(u32),
}

#[derive(Clone, Debug)]
struct Rule {
    predicates: Vec<Predicate>,
    response: Response,
    expected: Times,
    consumed: u32,
    origin: &'static Location<'static>,
}

/// One call to a `(model, action)` this [`Mock`] has already served, recorded
/// regardless of whether a rule matched it - what a test inspects via
/// [`Mock::calls`] to assert not just what came back, but what the node
/// under test actually asked for. Blob bytes are never retained (a digest
/// prefix would cost more than it proves here); only the shape.
#[derive(Clone, Debug)]
pub struct RecordedCall {
    pub params: Value,
    pub blobs: Vec<(String, Media, usize)>,
}

#[derive(Default)]
struct MockState {
    rules: std::collections::BTreeMap<(String, String), Vec<Rule>>,
    calls: std::collections::BTreeMap<(String, String), Vec<RecordedCall>>,
}

/// A declarative, shared mock: register expectations with [`Mock::on`], turn
/// it into a real [`Registry`] with [`Mock::registry_from`], run the code
/// under test against that registry, then call [`Mock::verify`]. Cheaply
/// `Clone`- every clone shares the same underlying call log and rule table,
/// so a `Mock` can be captured by closures/threads freely.
#[derive(Clone, Default)]
pub struct Mock(Arc<Mutex<MockState>>);

impl Mock {
    pub fn new() -> Mock {
        Mock::default()
    }

    /// Begin declaring a rule for calls to `model:action`. Nothing is
    /// registered until a terminal method ([`RuleBuilder::returns`],
    /// [`RuleBuilder::returns_blob`], [`RuleBuilder::fails`] or
    /// [`RuleBuilder::never`]) is called.
    #[track_caller]
    pub fn on(&self, model: &str, action: &str) -> RuleBuilder {
        RuleBuilder {
            mock: self.clone(),
            model: model.to_string(),
            action: action.to_string(),
            predicates: Vec::new(),
            expected: Times::Any,
            origin: Location::caller(),
        }
    }

    /// Checks every registered rule against `manifests`: the `(model,
    /// action)` must exist, an explicit `returns_blob` output name must be a
    /// declared [`capability::BlobSpec`] of the same [`Media`], and a
    /// param-matching predicate must name a declared param. Called
    /// automatically by [`Self::registry_from`]; exposed separately so a
    /// test can assert a malformed scenario is refused before anything runs.
    pub fn bind(&self, manifests: &[Manifest]) -> Result<(), String> {
        let state = self.0.lock().expect("mock state poisoned");
        let mut errors = Vec::new();
        for ((model, action), rules) in &state.rules {
            let Some(m) = manifests.iter().find(|m| &m.model == model) else {
                errors.push(format!("mock rule declared for unknown model '{model}'"));
                continue;
            };
            let Some(spec) = m.actions.iter().find(|a| &a.name == action) else {
                errors.push(format!("mock rule declared for unknown action '{model}:{action}'"));
                continue;
            };
            for rule in rules {
                if let Response::Blob { output: Some(name), blob } = &rule.response {
                    match spec.outputs.iter().find(|o| &o.name == name) {
                        None => errors.push(format!("mock rule for '{model}:{action}' (declared at {}) returns undeclared output '{name}'", rule.origin)),
                        Some(o) if o.media != blob.media() => errors.push(format!(
                            "mock rule for '{model}:{action}' (declared at {}) returns '{name}' as {} but the action declares it as {}",
                            rule.origin,
                            blob.media().name(),
                            o.media.name()
                        )),
                        _ => {}
                    }
                }
                for p in &rule.predicates {
                    match p {
                        Predicate::ParamEq(name, _) | Predicate::ParamContains(name, _) => {
                            if !spec.params.iter().any(|ps| &ps.name == name) {
                                errors.push(format!("mock rule for '{model}:{action}' (declared at {}) matches on undeclared param '{name}'", rule.origin));
                            }
                        }
                        Predicate::BlobPresent(name) | Predicate::BlobMediaIs(name, _) => {
                            if !spec.inputs.iter().any(|b| &b.name == name) {
                                errors.push(format!("mock rule for '{model}:{action}' (declared at {}) matches on undeclared blob input '{name}'", rule.origin));
                            }
                        }
                        Predicate::TextContains(_) | Predicate::AtCall(_) => {}
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    }

    /// [`Self::bind`], then a real [`Registry`] with one provider per
    /// manifest, each dispatching through this `Mock`'s shared rule table -
    /// the registry a caller hands to whatever code under test expects a
    /// `capability::Registry`.
    pub fn registry_from(&self, manifests: Vec<Manifest>) -> Result<Registry, String> {
        self.bind(&manifests)?;
        let mut registry = Registry::new();
        for m in manifests {
            registry.register(Arc::new(MockRegistryProvider { manifest: m, state: self.0.clone() }));
        }
        Ok(registry)
    }

    /// Every call this `Mock` has served to `model:action` so far, in order -
    /// recorded whether or not a rule matched it.
    pub fn calls(&self, model: &str, action: &str) -> Vec<RecordedCall> {
        self.0.lock().expect("mock state poisoned").calls.get(&(model.to_string(), action.to_string())).cloned().unwrap_or_default()
    }

    /// Every rule declared with [`RuleBuilder::times`] or
    /// [`RuleBuilder::never`] whose consumed count does not match what was
    /// expected - the CMock/FFF "did every expectation actually fire" step.
    pub fn verify(&self) -> Result<(), MockReport> {
        let state = self.0.lock().expect("mock state poisoned");
        let mut problems = Vec::new();
        for ((model, action), rules) in &state.rules {
            for rule in rules {
                if let Times::Exactly(n) = rule.expected {
                    if rule.consumed != n {
                        problems.push(format!("'{model}:{action}' rule declared at {} expected exactly {n} call(s), got {}", rule.origin, rule.consumed));
                    }
                }
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(MockReport(problems))
        }
    }
}

/// [`Mock::verify`]'s failure: every unmet expectation, one per line.
#[derive(Debug)]
pub struct MockReport(Vec<String>);

impl std::fmt::Display for MockReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "mock expectations not met:")?;
        for p in &self.0 {
            writeln!(f, "  - {p}")?;
        }
        Ok(())
    }
}

impl std::error::Error for MockReport {}

/// Accumulates one rule's predicates and expected call count; a terminal
/// method ([`Self::returns`], [`Self::returns_blob`], [`Self::fails`],
/// [`Self::never`]) registers it on the [`Mock`] that created this builder.
pub struct RuleBuilder {
    mock: Mock,
    model: String,
    action: String,
    predicates: Vec<Predicate>,
    expected: Times,
    origin: &'static Location<'static>,
}

impl RuleBuilder {
    pub fn when_text_contains(mut self, s: &str) -> Self {
        self.predicates.push(Predicate::TextContains(s.to_string()));
        self
    }
    pub fn when_param_eq(mut self, name: &str, value: Value) -> Self {
        self.predicates.push(Predicate::ParamEq(name.to_string(), value));
        self
    }
    pub fn when_param_contains(mut self, name: &str, s: &str) -> Self {
        self.predicates.push(Predicate::ParamContains(name.to_string(), s.to_string()));
        self
    }
    pub fn when_blob(mut self, name: &str) -> Self {
        self.predicates.push(Predicate::BlobPresent(name.to_string()));
        self
    }
    pub fn when_blob_media_is(mut self, name: &str, media: Media) -> Self {
        self.predicates.push(Predicate::BlobMediaIs(name.to_string(), media));
        self
    }
    /// Matches only the Nth call (1-based) to this `(model, action)` pair -
    /// never a global ordinal across different actions.
    pub fn at_call(mut self, n: u32) -> Self {
        self.predicates.push(Predicate::AtCall(n));
        self
    }
    /// This rule must be consumed exactly `n` times by [`Mock::verify`].
    pub fn times(mut self, n: u32) -> Self {
        self.expected = Times::Exactly(n);
        self
    }

    fn finish(self, response: Response) {
        let mut state = self.mock.0.lock().expect("mock state poisoned");
        state.rules.entry((self.model, self.action)).or_default().push(Rule {
            predicates: self.predicates,
            response,
            expected: self.expected,
            consumed: 0,
            origin: self.origin,
        });
    }

    /// Return `blob` from the action's own first declared output whose media
    /// matches (falling back to its first output at all) - the common case
    /// where an action has exactly one output worth naming.
    pub fn returns(self, blob: MockBlob) {
        self.finish(Response::Blob { output: None, blob });
    }
    /// Return `blob` from the explicitly named output - required when an
    /// action declares more than one output of the same media, or to make a
    /// scenario self-documenting.
    pub fn returns_blob(self, output: &str, blob: MockBlob) {
        self.finish(Response::Blob { output: Some(output.to_string()), blob });
    }
    /// Fail this call with `msg`, exactly as a real action returning `Err`
    /// would - for proving a caller handles a failed upstream call, not just
    /// a successful one.
    pub fn fails(self, msg: impl Into<String>) {
        self.finish(Response::Fail(msg.into()));
    }
    /// This exact call must never happen. Implemented as a rule that always
    /// matches and always fails loudly if it ever does, with
    /// `expected: Exactly(0)` - so [`Mock::verify`] reports it as a real
    /// violation rather than the call silently falling through to another
    /// rule or a generic "unmatched call" error.
    pub fn never(mut self) {
        let predicates = self.predicates.iter().map(Predicate::describe).collect::<Vec<_>>().join(", ");
        let where_ = if predicates.is_empty() { "any call".to_string() } else { predicates };
        let msg = format!("mock: '{}:{}' was called matching [{where_}], declared with .never() at {}", self.model, self.action, self.origin);
        self.expected = Times::Exactly(0);
        self.finish(Response::Fail(msg));
    }
}

/// The [`Provider`] [`Mock::registry_from`] builds - one per manifest,
/// dispatching every action through the shared [`MockState`].
struct MockRegistryProvider {
    manifest: Manifest,
    state: Arc<Mutex<MockState>>,
}

impl Provider for MockRegistryProvider {
    fn manifest(&self) -> Manifest {
        self.manifest.clone()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let spec = self.manifest.actions.iter().find(|a| a.name == name)?.clone();
        Some(Arc::new(MockRegistryAction { model: self.manifest.model.clone(), spec, state: self.state.clone() }))
    }
}

struct MockRegistryAction {
    model: String,
    spec: ActionSpec,
    state: Arc<Mutex<MockState>>,
}

impl Action for MockRegistryAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        run_steps(&inv.cancel, progress)?;
        let key = (self.model.clone(), self.spec.name.clone());

        let (chosen, call_index, near_misses) = {
            let mut state = self.state.lock().expect("mock state poisoned");
            let call_index = state.calls.entry(key.clone()).or_default().len() as u32 + 1;

            let mut chosen_index = None;
            let mut near_misses: Vec<String> = Vec::new();
            if let Some(rules) = state.rules.get_mut(&key) {
                for (i, rule) in rules.iter().enumerate() {
                    // `n == 0` (as `.never()` sets) must NOT be treated as
                    // "already exhausted, skip it" - a never-expected rule
                    // has to stay eligible to match so it can fire its Fail
                    // response and be caught, rather than silently falling
                    // through to the next rule (or "no rule matched") as if
                    // it had never been declared at all.
                    if let Times::Exactly(n) = rule.expected {
                        if n > 0 && rule.consumed >= n {
                            near_misses.push(format!("rule declared at {} already consumed its {n} expected call(s)", rule.origin));
                            continue;
                        }
                    }
                    let mismatched: Vec<String> = rule.predicates.iter().filter(|p| !p.matches(inv, call_index)).map(Predicate::describe).collect();
                    if mismatched.is_empty() {
                        chosen_index = Some(i);
                        break;
                    }
                    near_misses.push(format!("rule declared at {} did not match: {}", rule.origin, mismatched.join(", ")));
                }
                if let Some(i) = chosen_index {
                    rules[i].consumed += 1;
                }
            }
            let chosen = chosen_index.and_then(|i| state.rules.get(&key).and_then(|rules| rules.get(i)).map(|r| r.response.clone()));

            state.calls.get_mut(&key).expect("just inserted above").push(RecordedCall {
                params: inv.params.clone(),
                blobs: inv.blobs.iter().map(|(n, b)| (n.clone(), b.media, b.bytes.len())).collect(),
            });

            (chosen, call_index, near_misses)
        };

        let Some(response) = chosen else {
            return Err(if near_misses.is_empty() {
                format!("mock: no rule registered for '{}/{}' (call #{call_index})", self.model, self.spec.name)
            } else {
                let blobs = inv.blobs.iter().map(|(n, b)| format!("{n}:{}", b.media.name())).collect::<Vec<_>>().join(", ");
                format!(
                    "mock: no rule matched call #{call_index} to '{}/{}' - params {}, blobs [{blobs}]; rules tried: {}",
                    self.model,
                    self.spec.name,
                    inv.params,
                    near_misses.join(" | ")
                )
            });
        };

        match response {
            Response::Fail(msg) => Err(msg),
            Response::Blob { output, blob } => {
                let media = blob.media();
                let name = output.unwrap_or_else(|| {
                    self.spec
                        .outputs
                        .iter()
                        .find(|o| o.media == media)
                        .or_else(|| self.spec.outputs.first())
                        .map(|o| o.name.clone())
                        .unwrap_or_else(|| media.name().to_string())
                });
                let built = blob.build()?;
                Ok(Outcome::new().set("mock", json!(true)).blob(&name, built))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::blob::{decode_image, decode_plane};
    use capability::{BlobSpec, ParamSpec, ParamType};

    fn ocr_manifest() -> Manifest {
        Manifest::new(
            "vendor/ocr",
            "test",
            vec![ActionSpec::new("generate", "").param(ParamSpec::new("prompt", ParamType::Str, "")).output(BlobSpec::new("text", Media::Text, ""))],
        )
    }

    #[test]
    fn a_scripted_text_response_reaches_the_caller_verbatim() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").returns_blob("text", MockBlob::text("# Invoice 4711"));
        let registry = mock.registry_from(vec![ocr_manifest()]).unwrap();
        let out = registry.run("vendor/ocr", "generate", Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(out.blobs["text"].bytes, b"# Invoice 4711");
        mock.verify().unwrap();
    }

    #[test]
    fn returns_resolves_the_output_name_from_the_spec_when_none_is_given() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").returns(MockBlob::text("auto-named"));
        let registry = mock.registry_from(vec![ocr_manifest()]).unwrap();
        let out = registry.run("vendor/ocr", "generate", Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(out.blobs["text"].bytes, b"auto-named");
    }

    #[test]
    fn when_text_contains_selects_between_two_rules() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").when_text_contains("invoice").returns_blob("text", MockBlob::text("invoice path"));
        mock.on("vendor/ocr", "generate").when_text_contains("receipt").returns_blob("text", MockBlob::text("receipt path"));
        let registry = mock.registry_from(vec![ocr_manifest()]).unwrap();

        let a = registry.run("vendor/ocr", "generate", Invocation::new().set("prompt", json!("an invoice")), &mut |_| {}).unwrap();
        assert_eq!(a.blobs["text"].bytes, b"invoice path");
        let b = registry.run("vendor/ocr", "generate", Invocation::new().set("prompt", json!("a receipt")), &mut |_| {}).unwrap();
        assert_eq!(b.blobs["text"].bytes, b"receipt path");
    }

    #[test]
    fn at_call_scripts_a_sequence_by_ordinal_and_fails_a_specific_call() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").at_call(1).returns_blob("text", MockBlob::text("page one"));
        mock.on("vendor/ocr", "generate").at_call(2).fails("page unreadable");
        mock.on("vendor/ocr", "generate").at_call(3).returns_blob("text", MockBlob::text("page three"));
        let registry = mock.registry_from(vec![ocr_manifest()]).unwrap();

        let r1 = registry.run("vendor/ocr", "generate", Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(r1.blobs["text"].bytes, b"page one");
        let e2 = registry.run("vendor/ocr", "generate", Invocation::new(), &mut |_| {}).unwrap_err();
        assert_eq!(e2, "page unreadable");
        let r3 = registry.run("vendor/ocr", "generate", Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(r3.blobs["text"].bytes, b"page three");
    }

    #[test]
    fn an_unmatched_call_is_refused_with_the_near_miss_reasons() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").when_text_contains("invoice").returns_blob("text", MockBlob::text("x"));
        let registry = mock.registry_from(vec![ocr_manifest()]).unwrap();
        let err = registry.run("vendor/ocr", "generate", Invocation::new().set("prompt", json!("a receipt")), &mut |_| {}).unwrap_err();
        assert!(err.contains("no rule matched"), "{err}");
        assert!(err.contains("text contains \"invoice\""), "{err}");
    }

    #[test]
    fn times_and_verify_report_an_unmet_expectation() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").times(2).returns_blob("text", MockBlob::text("x"));
        let _ = mock.registry_from(vec![ocr_manifest()]).unwrap();
        // Never actually called -> verify must report the shortfall.
        let report = mock.verify().unwrap_err();
        assert!(report.to_string().contains("expected exactly 2"), "{report}");
    }

    #[test]
    fn never_is_violated_the_moment_a_matching_call_happens() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").when_text_contains("forbidden").never();
        let registry = mock.registry_from(vec![ocr_manifest()]).unwrap();
        let err = registry.run("vendor/ocr", "generate", Invocation::new().set("prompt", json!("a forbidden request")), &mut |_| {}).unwrap_err();
        assert!(err.contains("declared with .never()"), "{err}");
    }

    #[test]
    fn calls_records_params_and_blob_shapes_even_when_unmatched() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").returns_blob("text", MockBlob::text("x"));
        let registry = mock.registry_from(vec![ocr_manifest()]).unwrap();
        registry.run("vendor/ocr", "generate", Invocation::new().set("prompt", json!("hello")), &mut |_| {}).unwrap();
        let calls = mock.calls("vendor/ocr", "generate");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].params["prompt"], json!("hello"));
    }

    #[test]
    fn bind_refuses_an_undeclared_output_name_before_anything_runs() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").returns_blob("nope", MockBlob::text("x"));
        let err = mock.bind(&[ocr_manifest()]).unwrap_err();
        assert!(err.contains("undeclared output 'nope'"), "{err}");
    }

    #[test]
    fn bind_refuses_a_media_mismatch_between_the_rule_and_the_declared_output() {
        let mock = Mock::new();
        mock.on("vendor/ocr", "generate").returns_blob("text", MockBlob::image(4, 4));
        let err = mock.bind(&[ocr_manifest()]).unwrap_err();
        assert!(err.contains("declares it as text"), "{err}");
    }

    #[test]
    fn bind_refuses_an_unknown_model_or_action() {
        let mock = Mock::new();
        mock.on("vendor/nope", "generate").returns_blob("text", MockBlob::text("x"));
        assert!(mock.bind(&[ocr_manifest()]).unwrap_err().contains("unknown model"));

        let mock2 = Mock::new();
        mock2.on("vendor/ocr", "nope").returns_blob("text", MockBlob::text("x"));
        assert!(mock2.bind(&[ocr_manifest()]).unwrap_err().contains("unknown action"));
    }

    #[test]
    fn an_image_blob_round_trips_through_the_real_decoder() {
        let manifest = Manifest::new("vendor/vision", "test", vec![ActionSpec::new("detect", "").output(BlobSpec::new("image", Media::Image, ""))]);
        let mock = Mock::new();
        mock.on("vendor/vision", "detect").returns_blob("image", MockBlob::image_seeded(4, 3, 7));
        let registry = mock.registry_from(vec![manifest]).unwrap();
        let out = registry.run("vendor/vision", "detect", Invocation::new(), &mut |_| {}).unwrap();
        let (hwc, w, h) = decode_image(&Invocation::new().blob("image", out.blobs["image"].clone()), "image").unwrap();
        assert_eq!((w, h), (4, 3));
        assert_eq!(hwc.len(), 4 * 3 * 3);
    }

    #[test]
    fn a_mask_blob_is_tagged_correctly() {
        let manifest = Manifest::new("vendor/seg", "test", vec![ActionSpec::new("segment", "").output(BlobSpec::new("mask", Media::Mask, ""))]);
        let mock = Mock::new();
        mock.on("vendor/seg", "segment").returns_blob("mask", MockBlob::mask(2, 2));
        let registry = mock.registry_from(vec![manifest]).unwrap();
        let out = registry.run("vendor/seg", "segment", Invocation::new(), &mut |_| {}).unwrap();
        let blob = out.blobs["mask"].clone();
        assert_eq!(blob.media, Media::Mask);
        let (plane, w, h) = decode_plane(&Invocation::new().blob("mask", blob), "mask").unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(plane.len(), 4);
    }

    #[test]
    fn a_json_blob_round_trips() {
        let manifest = Manifest::new("vendor/extract", "test", vec![ActionSpec::new("generate", "").output(BlobSpec::new("text", Media::Text, ""))]);
        let mock = Mock::new();
        mock.on("vendor/extract", "generate").returns_blob("text", MockBlob::json(json!({"total_cents": 129900})));
        let registry = mock.registry_from(vec![manifest]).unwrap();
        let out = registry.run("vendor/extract", "generate", Invocation::new(), &mut |_| {}).unwrap();
        let parsed: Value = serde_json::from_slice(&out.blobs["text"].bytes).unwrap();
        assert_eq!(parsed["total_cents"], json!(129900));
    }

    #[test]
    fn cancellation_is_honored_before_any_rule_is_consulted() {
        let manifest = Manifest::new("vendor/x", "test", vec![ActionSpec::new("go", "").output(BlobSpec::new("text", Media::Text, ""))]);
        let mock = Mock::new();
        mock.on("vendor/x", "go").returns_blob("text", MockBlob::text("never reached"));
        let registry = mock.registry_from(vec![manifest]).unwrap();
        let cancel = capability::CancelToken::armed();
        cancel.cancel();
        let mut inv = Invocation::new();
        inv.cancel = cancel;
        let err = registry.run("vendor/x", "go", inv, &mut |_| {}).unwrap_err();
        assert_eq!(err, "cancelled");
        assert!(mock.calls("vendor/x", "go").is_empty(), "a cancelled call must never be recorded");
    }
}
