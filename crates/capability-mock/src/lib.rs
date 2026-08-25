// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A deterministic, weight-free synthetic-content [`capability::Provider`].
//!
//! [`MockProvider`] advertises a [`capability::Manifest`] exactly like a real
//! model - built by hand ([`MockProvider::new`] + [`MockProvider::action`]) or
//! mirrored 1:1 from a real one ([`MockProvider::from_manifest`]) - and every
//! action it hands back generates deterministic synthetic content (a gradient
//! image, a moving-gradient video, a sine-tone PCM clip, a short derived
//! string, or a counter byte pattern) with **no GPU, no weights, no RNG crate
//! and no file I/O**. It exists so an in-process consumer of
//! `capability::Provider` - a workflow UI, an integration test, a CI box with
//! no accelerator - can exercise the full generalized-capability contract
//! (typed params, blob in/out, streaming progress, cancellation) against
//! synthetic content instead of downloading and running real model weights.
//!
//! Swedish Embedded AB builds exactly this kind of weight-free integration
//! seam for teams that need to develop and test against a model API before
//! the real model is available or affordable to run in every environment. If
//! your team needs a synthetic stand-in for a heavyweight inference backend,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Cancellation
//!
//! Every mock action's `run()` ticks a handful of synthetic "steps"
//! ([`capability::Progress::step`]) and polls [`capability::CancelToken`]
//! between them, returning `Err("cancelled")` promptly on a cancelled token - 
//! the same discipline every real model action is required to follow (see
//! `capability::Action::run`'s doc), demonstrated here rather than merely
//! claimed.
//!
//! # Determinism
//!
//! Every generator is pure arithmetic over a `u32` seed folded from
//! `(model_id, action name, an optional "seed" param)` - no RNG crate, the
//! same wrapping-arithmetic style `crates/cli/src/resident_mock.rs`'s
//! `text2image` mock uses, just extended to fold in the model/action identity
//! so two different mock providers never coincidentally produce the same
//! pixels.

use std::f32::consts::TAU;
use std::sync::Arc;

use capability::blob::{image_blob, video_blob};
use capability::{Action, ActionResult, ActionSpec, Blob, CancelToken, Invocation, Manifest, Media, Outcome, Progress, Provider};
use serde_json::json;

/// The shape of synthetic content one mock action produces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MockOutput {
    /// An interleaved HWC f32 `[0,1]` RGB gradient, [`capability::blob::image_blob`]-encoded.
    Image { w: u32, h: u32 },
    /// The same gradient, re-tagged [`Media::Mask`].
    Mask { w: u32, h: u32 },
    /// A sine tone as raw f32-LE PCM bytes, `meta.sample_rate`/`meta.channels` set.
    Audio { seconds: f32, sample_rate: u32, channels: u32 },
    /// N frames of a moving-gradient pattern, [`capability::blob::video_blob`]-encoded
    /// with `meta.fps` set.
    Video { frames: u32, w: u32, h: u32, fps: f32 },
    /// A short deterministic string derived from the invocation's prompt/messages,
    /// falling back to a fixed model+action-keyed string.
    Text,
    /// `n` deterministic bytes (a repeating counter pattern).
    Bytes(usize),
}

impl MockOutput {
    /// The [`Media`] kind this output is tagged with on the wire.
    fn media(&self) -> Media {
        match self {
            MockOutput::Image { .. } => Media::Image,
            MockOutput::Mask { .. } => Media::Mask,
            MockOutput::Audio { .. } => Media::Audio,
            MockOutput::Video { .. } => Media::Video,
            MockOutput::Text => Media::Text,
            MockOutput::Bytes(_) => Media::Bytes,
        }
    }
    /// The blob name a hand-built [`ActionSpec`] would plausibly use for this
    /// output kind - the fallback when the actual spec declares no outputs.
    fn default_name(&self) -> &'static str {
        match self {
            MockOutput::Image { .. } => "image",
            MockOutput::Mask { .. } => "mask",
            MockOutput::Audio { .. } => "audio",
            MockOutput::Video { .. } => "video",
            MockOutput::Text => "text",
            MockOutput::Bytes(_) => "bytes",
        }
    }

    /// Infer a plausible [`MockOutput`] from an [`ActionSpec`]'s declared
    /// FIRST output's [`Media`] kind, reading `width`/`height`/`frames`/`fps`/
    /// `seconds`/`duration`/`sample_rate`/`channels` param DEFAULTS when the
    /// spec declares them, else a small fixed size - used by
    /// [`MockProvider::from_manifest`] so a mirrored action produces content
    /// shaped like what the real one would.
    fn infer(spec: &ActionSpec) -> MockOutput {
        let int_default = |name: &str, fallback: u32| -> u32 {
            spec.params.iter().find(|p| p.name == name).and_then(|p| p.default.as_ref()).and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(fallback)
        };
        let float_default = |name: &str, fallback: f32| -> f32 {
            spec.params.iter().find(|p| p.name == name).and_then(|p| p.default.as_ref()).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(fallback)
        };
        match spec.outputs.first().map(|o| o.media) {
            Some(Media::Image) => MockOutput::Image { w: int_default("width", 64), h: int_default("height", 64) },
            Some(Media::Mask) => MockOutput::Mask { w: int_default("width", 64), h: int_default("height", 64) },
            Some(Media::Video) => MockOutput::Video {
                frames: int_default("frames", 4),
                w: int_default("width", 64),
                h: int_default("height", 64),
                fps: float_default("fps", 8.0),
            },
            Some(Media::Audio) => MockOutput::Audio {
                seconds: float_default("seconds", float_default("duration", 1.0)),
                sample_rate: int_default("sample_rate", 16000),
                channels: int_default("channels", 1),
            },
            Some(Media::Bytes) => MockOutput::Bytes(256),
            Some(Media::Text) | None => MockOutput::Text,
        }
    }
}

/// A deterministic, weight-free [`Provider`]. Build with [`MockProvider::new`]
/// and [`MockProvider::action`], or mirror a real model's manifest wholesale
/// with [`MockProvider::from_manifest`].
///
/// **Naming note**: this inherent [`Self::action`] builder and
/// [`Provider::action`] (dispatch-by-name, returning a runnable
/// [`capability::Action`]) share the name `action` by design, matching the
/// two roles "add an action" / "look up an action by name" play elsewhere in
/// this crate's own `Provider` impls. Method-call syntax always resolves to
/// the INHERENT one; a caller that wants to dispatch by name (as `capability::
/// Registry::run` does internally) must call the trait method explicitly - 
/// `Provider::action(&provider, name)` - exactly as this crate's own tests do.
pub struct MockProvider {
    model_id: String,
    summary: String,
    max_context_tokens: Option<u64>,
    actions: Vec<(ActionSpec, MockOutput)>,
}

impl MockProvider {
    /// A provider with no actions yet - add them with [`Self::action`].
    pub fn new(model_id: &str, summary: &str) -> MockProvider {
        MockProvider { model_id: model_id.into(), summary: summary.into(), max_context_tokens: None, actions: Vec::new() }
    }

    /// Builder-style: add one action, generating `output`-shaped synthetic
    /// content when it runs.
    pub fn action(mut self, spec: ActionSpec, output: MockOutput) -> MockProvider {
        self.actions.push((spec, output));
        self
    }

    /// Mirror an existing real [`Manifest`]'s action list 1:1 - same names,
    /// params, inputs and outputs - inferring each action's [`MockOutput`]
    /// from its declared output [`Media`] kind (see [`MockOutput::infer`]).
    /// Lets a caller stand up a weight-free stand-in for a specific real
    /// model (matching `brain caps`'s advertised shape exactly) without
    /// hand-describing every action again.
    pub fn from_manifest(m: Manifest) -> MockProvider {
        let mut p = MockProvider::new(&m.model, &m.summary);
        p.max_context_tokens = m.max_context_tokens;
        for spec in m.actions {
            let output = MockOutput::infer(&spec);
            p.actions.push((spec, output));
        }
        p
    }
}

impl Provider for MockProvider {
    fn manifest(&self) -> Manifest {
        let m = Manifest::new(&self.model_id, &self.summary, self.actions.iter().map(|(s, _)| s.clone()).collect());
        match self.max_context_tokens {
            Some(t) => m.with_max_context_tokens(t),
            None => m,
        }
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let (spec, output) = self.actions.iter().find(|(s, _)| s.name == name)?;
        Some(Arc::new(MockAction { model_id: self.model_id.clone(), spec: spec.clone(), output: *output }))
    }
}

/// One runnable mock action, bound to the provider's model id (folded into
/// the deterministic seed) and its declared [`ActionSpec`] (read for the
/// output blob's name and any `width`/`height`/... params a caller passed).
struct MockAction {
    model_id: String,
    spec: ActionSpec,
    output: MockOutput,
}

/// How many synthetic "steps" every mock action ticks through before
/// returning, polling `cancel` between each - enough to make cancellation
/// observably interruptible (see the module doc's Cancellation section)
/// without slowing tests down.
const STEPS: u32 = 3;

/// Poll `cancel` between `STEPS` synthetic progress ticks, returning
/// `Err("cancelled")` the moment it fires (checked BEFORE the first tick too,
/// so an already-cancelled invocation never emits progress at all).
fn run_steps(cancel: &CancelToken, progress: &mut dyn FnMut(Progress)) -> Result<(), String> {
    for s in 0..STEPS {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        progress(Progress::step(s + 1, STEPS, "mock generating"));
    }
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    Ok(())
}

/// Fold `(model_id, action, an optional numeric extra)` into one `u32` seed - 
/// plain wrapping arithmetic (FNV-1a over the bytes, then an avalanche mix),
/// no RNG crate, matching `resident_mock.rs::text2image`'s "no randomness,
/// just wrapping arithmetic" style but extended to the identity strings so
/// distinct mock providers/actions never coincide.
fn fold_seed(model_id: &str, action: &str, extra: i64) -> u32 {
    let mut h: u32 = 0x811C9DC5; // FNV-1a offset basis
    for b in model_id.bytes().chain(std::iter::once(b'/')).chain(action.bytes()) {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193); // FNV prime
    }
    h ^= extra as u32;
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h
}

/// The invocation's `seed` param, or `0` - folded into [`fold_seed`]'s extra
/// term so a caller-supplied seed actually changes the output.
fn seed_param(inv: &Invocation) -> i64 {
    inv.get_i64("seed").unwrap_or(0)
}

/// A deterministic interleaved-HWC f32 `[0,1]` RGB gradient - the same shape
/// `resident_mock.rs::text2image` produces (three independent per-channel
/// ramps offset by `seed`), so a mock image/mask "looks like" the real mock
/// resident's own output, not an unrelated pattern.
fn gradient_hwc(seed: u32, w: u32, h: u32) -> Vec<f32> {
    let mut hwc = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        for x in 0..w {
            hwc.push((x.wrapping_add(seed) % 256) as f32 / 255.0);
            hwc.push((y.wrapping_add(seed >> 8) % 256) as f32 / 255.0);
            hwc.push((x.wrapping_add(y).wrapping_add(seed) % 256) as f32 / 255.0);
        }
    }
    hwc
}

/// One frame of a moving-gradient clip: the same "bright block sweeps across
/// a per-axis ramp" idea as `crates/imaging/src/video.rs`'s private test
/// helper `moving_block`, reimplemented directly as an f32 HWC `[0,1]` plane
/// (no u8 round trip, no `Rgb8`) so [`video_blob`] can encode it straight - 
/// the point of the fixture is the same: frames must NOT all be identical.
fn video_frame_hwc(seed: u32, w: u32, h: u32, frame: u32) -> Vec<f32> {
    let mut hwc = Vec::with_capacity((w as usize) * (h as usize) * 3);
    let period = w.max(1);
    for y in 0..h {
        for x in 0..w {
            let on = (x.wrapping_add(frame).wrapping_add(seed)) % period == 0;
            hwc.push(if on { 1.0 } else { (x % 256) as f32 / 255.0 });
            hwc.push((y.wrapping_add(seed >> 8) % 256) as f32 / 255.0);
            hwc.push((frame.wrapping_mul(20).wrapping_add(seed) % 256) as f32 / 255.0);
        }
    }
    hwc
}

/// A sine tone as raw interleaved f32-LE PCM: frequency `220 + seed % 440` Hz
/// (always audibly non-silent and seed-distinguishable), `channels` identical
/// copies per frame (a real stereo/mono split has no meaning for a synthetic
/// tone) - no WAV container, matching `audio::asr_caps`'s raw-PCM-plus-meta
/// convention for an untagged-format audio blob.
fn sine_pcm(seed: u32, seconds: f32, sample_rate: u32, channels: u32) -> Vec<u8> {
    let freq = 220.0 + (seed % 440) as f32;
    let n_frames = ((seconds.max(0.0)) * sample_rate as f32).round() as usize;
    let mut bytes = Vec::with_capacity(n_frames * channels as usize * 4);
    for i in 0..n_frames {
        let t = i as f32 / sample_rate.max(1) as f32;
        let sample = (TAU * freq * t).sin() * 0.5;
        for _ in 0..channels {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }
    bytes
}

/// A short deterministic string: the invocation's last user turn / `prompt`
/// (via [`capability::last_user_text`], the same extraction every real chat
/// model uses), prefixed with the model/action identity - or, when the
/// invocation carries no such param, a fixed string keyed by model+action.
fn mock_text(model_id: &str, action: &str, inv: &Invocation) -> String {
    let prompt = capability::last_user_text(inv);
    if prompt.trim().is_empty() {
        format!("[{model_id}/{action}] deterministic mock output")
    } else {
        format!("[{model_id}/{action}] {}", prompt.trim())
    }
}

/// `n` deterministic bytes: a repeating `(i + seed) % 256` counter pattern.
fn counter_bytes(seed: u32, n: usize) -> Vec<u8> {
    (0..n).map(|i| ((i as u32).wrapping_add(seed) % 256) as u8).collect()
}

impl Action for MockAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        run_steps(&inv.cancel, progress)?;

        let seed = fold_seed(&self.model_id, &self.spec.name, seed_param(inv));
        let name = self.spec.outputs.first().map(|o| o.name.as_str()).unwrap_or_else(|| self.output.default_name());
        if let Some(o) = self.spec.outputs.first() {
            debug_assert_eq!(o.media, self.output.media(), "a mock action's MockOutput must match its spec's declared output media");
        }

        let blob = match self.output {
            MockOutput::Image { w, h } => image_blob(&gradient_hwc(seed, w, h), w, h, 3),
            MockOutput::Mask { w, h } => image_blob(&gradient_hwc(seed, w, h), w, h, 3).with_media(Media::Mask),
            MockOutput::Audio { seconds, sample_rate, channels } => {
                Blob::new(Media::Audio, sine_pcm(seed, seconds, sample_rate, channels)).with_meta(json!({ "sample_rate": sample_rate, "channels": channels }))
            }
            MockOutput::Video { frames, w, h, fps } => {
                let frames: Vec<(Vec<f32>, u32, u32)> = (0..frames.max(1)).map(|f| (video_frame_hwc(seed, w, h, f), w, h)).collect();
                let mut b = video_blob(&frames)?;
                let mut meta = b.meta.clone();
                meta["fps"] = json!(fps);
                b = b.with_meta(meta);
                b
            }
            MockOutput::Text => Blob::new(Media::Text, mock_text(&self.model_id, &self.spec.name, inv).into_bytes()),
            MockOutput::Bytes(n) => Blob::new(Media::Bytes, counter_bytes(seed, n)),
        };

        Ok(Outcome::new().set("mock", json!(true)).blob(name, blob))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::blob::{decode_hwc, decode_image, decode_plane, decode_video};
    use capability::{BlobSpec, ParamSpec, ParamType};

    fn spec_for(output: MockOutput) -> ActionSpec {
        let s = ActionSpec::new("go", "a mock action");
        match output {
            MockOutput::Image { .. } => s.output(BlobSpec::new("image", Media::Image, "")),
            MockOutput::Mask { .. } => s.output(BlobSpec::new("mask", Media::Mask, "")),
            MockOutput::Audio { .. } => s.output(BlobSpec::new("audio", Media::Audio, "")),
            MockOutput::Video { .. } => s.output(BlobSpec::new("video", Media::Video, "")),
            MockOutput::Text => s.output(BlobSpec::new("text", Media::Text, "")),
            MockOutput::Bytes(_) => s.output(BlobSpec::new("bytes", Media::Bytes, "")),
        }
    }

    #[test]
    fn image_round_trips_through_the_real_decoder() {
        let p = MockProvider::new("mock/vision", "test").action(spec_for(MockOutput::Image { w: 4, h: 3 }), MockOutput::Image { w: 4, h: 3 });
        let out = Provider::action(&p, "go").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        let (hwc, w, h) = decode_image(&Invocation::new().blob("image", out.blobs["image"].clone()), "image").unwrap();
        assert_eq!((w, h), (4, 3));
        assert_eq!(hwc.len(), 4 * 3 * 3);
        assert!(hwc.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn mask_is_the_same_wire_format_retagged() {
        let p = MockProvider::new("mock/vision", "test").action(spec_for(MockOutput::Mask { w: 2, h: 2 }), MockOutput::Mask { w: 2, h: 2 });
        let out = Provider::action(&p, "go").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        let blob = out.blobs["mask"].clone();
        assert_eq!(blob.media, Media::Mask);
        let (plane, w, h) = decode_plane(&Invocation::new().blob("mask", blob), "mask").unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(plane.len(), 4);
    }

    #[test]
    fn video_round_trips_and_frames_are_not_all_identical() {
        let p =
            MockProvider::new("mock/video", "test").action(spec_for(MockOutput::Video { frames: 3, w: 2, h: 2, fps: 12.0 }), MockOutput::Video { frames: 3, w: 2, h: 2, fps: 12.0 });
        let out = Provider::action(&p, "go").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        let blob = out.blobs["video"].clone();
        assert_eq!(blob.meta["fps"], json!(12.0));
        let frames = decode_video(&Invocation::new().blob("video", blob), "video").unwrap();
        assert_eq!(frames.len(), 3);
        assert_ne!(frames[0].0, frames[1].0, "moving-gradient frames must differ");
        assert_ne!(frames[1].0, frames[2].0, "moving-gradient frames must differ");
    }

    #[test]
    fn audio_is_raw_pcm_with_sample_rate_and_channel_meta() {
        let p = MockProvider::new("mock/audio", "test").action(
            spec_for(MockOutput::Audio { seconds: 0.1, sample_rate: 8000, channels: 2 }),
            MockOutput::Audio { seconds: 0.1, sample_rate: 8000, channels: 2 },
        );
        let out = Provider::action(&p, "go").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        let blob = &out.blobs["audio"];
        assert_eq!(blob.media, Media::Audio);
        assert_eq!(blob.meta, json!({"sample_rate": 8000, "channels": 2}));
        // 0.1s @ 8000Hz * 2 channels * 4 bytes/f32
        assert_eq!(blob.bytes.len(), 800 * 2 * 4);
        let samples: Vec<f32> = blob.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert!(samples.iter().any(|&s| s.abs() > 1e-6), "tone must not be silent");
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn text_echoes_the_prompt_and_falls_back_when_absent() {
        let p = MockProvider::new("mock/chat", "test").action(spec_for(MockOutput::Text), MockOutput::Text);
        let with_prompt = Provider::action(&p, "go").unwrap().run(&Invocation::new().set("prompt", json!("hello there")), &mut |_| {}).unwrap();
        let text = String::from_utf8(with_prompt.blobs["text"].bytes.clone()).unwrap();
        assert!(text.contains("hello there"), "{text}");

        let without_prompt = Provider::action(&p, "go").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        let text2 = String::from_utf8(without_prompt.blobs["text"].bytes.clone()).unwrap();
        assert!(text2.contains("mock/chat") && text2.contains("go"), "{text2}");
        assert_ne!(text, text2);
    }

    #[test]
    fn bytes_are_deterministic_and_the_declared_length() {
        let p = MockProvider::new("mock/blob", "test").action(spec_for(MockOutput::Bytes(16)), MockOutput::Bytes(16));
        let a = Provider::action(&p, "go").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        let b = Provider::action(&p, "go").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        assert_eq!(a.blobs["bytes"].bytes.len(), 16);
        assert_eq!(a.blobs["bytes"].bytes, b.blobs["bytes"].bytes, "same inputs must produce the same bytes");
    }

    #[test]
    fn different_seeds_produce_different_output() {
        let spec = spec_for(MockOutput::Image { w: 4, h: 4 }).param(ParamSpec::new("seed", ParamType::Int, "seed").default(json!(0)));
        let p = MockProvider::new("mock/vision", "test").action(spec, MockOutput::Image { w: 4, h: 4 });
        let a = Provider::action(&p, "go").unwrap().run(&Invocation::new().set("seed", json!(1)), &mut |_| {}).unwrap();
        let b = Provider::action(&p, "go").unwrap().run(&Invocation::new().set("seed", json!(2)), &mut |_| {}).unwrap();
        assert_ne!(a.blobs["image"].bytes, b.blobs["image"].bytes);
    }

    #[test]
    fn from_manifest_mirrors_actions_and_infers_output_shape() {
        let real = Manifest::new(
            "real/model",
            "a real model",
            vec![
                ActionSpec::new("text2image", "generate an image")
                    .param(ParamSpec::new("width", ParamType::Int, "w").default(json!(8)))
                    .param(ParamSpec::new("height", ParamType::Int, "h").default(json!(6)))
                    .output(BlobSpec::new("image", Media::Image, "the image")),
                ActionSpec::new("generate", "chat").streaming().output(BlobSpec::new("text", Media::Text, "the reply")),
            ],
        );
        let mock = MockProvider::from_manifest(real.clone());
        let mirrored = mock.manifest();
        assert_eq!(mirrored.model, "real/model");
        assert_eq!(mirrored.actions.len(), 2);
        assert_eq!(mirrored.actions[0].name, "text2image");
        assert!(mirrored.actions[1].streaming);

        let out = Provider::action(&mock, "text2image").unwrap().run(&Invocation::new(), &mut |_| {}).unwrap();
        let (_, w, h, c) = decode_hwc(&Invocation::new().blob("image", out.blobs["image"].clone()), "image").unwrap();
        assert_eq!((w, h, c), (8, 6, 3), "inferred image dims must come from the real spec's width/height defaults");
    }

    #[test]
    fn cancellation_aborts_promptly_and_before_any_content_is_built() {
        let p = MockProvider::new("mock/vision", "test").action(spec_for(MockOutput::Image { w: 1000, h: 1000 }), MockOutput::Image { w: 1000, h: 1000 });
        let cancel = CancelToken::armed();
        cancel.cancel();
        let mut inv = Invocation::new();
        inv.cancel = cancel;
        let mut steps = 0u32;
        let err = Provider::action(&p, "go").unwrap().run(&inv, &mut |_| steps += 1).unwrap_err();
        assert_eq!(err, "cancelled");
        assert_eq!(steps, 0);
    }

    #[test]
    fn cancellation_mid_stream_aborts_before_all_steps_complete() {
        let p = MockProvider::new("mock/vision", "test").action(spec_for(MockOutput::Text), MockOutput::Text);
        let cancel = CancelToken::armed();
        let mut inv = Invocation::new();
        inv.cancel = cancel.clone();
        let mut steps = 0u32;
        let err = Provider::action(&p, "go")
            .unwrap()
            .run(&inv, &mut |_p| {
                steps += 1;
                if steps == 1 {
                    cancel.cancel();
                }
            })
            .unwrap_err();
        assert_eq!(err, "cancelled");
        assert!(steps < STEPS, "must abort before every step ticks: {steps}");
    }

    #[test]
    fn unknown_action_is_none() {
        let p = MockProvider::new("mock/x", "test");
        assert!(Provider::action(&p, "nope").is_none());
    }
}
