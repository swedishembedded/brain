// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS's capabilities behind the generalized [`capability`] interface -
//! what makes `brain caps tts` / `brain do tts synth …` (and the perf suite's
//! `CapabilityTarget`) work with no tts-specific plumbing in the CLI.
//!
//! Three actions, each a thin wrapper over the same pipeline the dedicated
//! `tts_cli.rs` runs:
//!   - `synth` - speaker-free text-to-speech ([`crate::pipeline::synth`]).
//!   - `clone` - voice cloning from a reference wav, x-vector-only or ICL
//!     when `ref_text` is given ([`crate::pipeline::clone`]).
//!   - `design` - instruct-style VoiceDesign and/or CustomVoice preset
//!     speakers ([`crate::pipeline::design`]).
//! All one-shot: the wav is the single artifact. Like the pipeline functions
//! themselves (and the `TtsResident` adapter), the weights load per call -
//! the load-once seam (`serve::TtsEngine`) is OpenVINO/NPU-only, so there is
//! nothing resident to cache here without duplicating the pipeline internals.
//! Before this, `clone`/`design` were reachable ONLY from `tts_cli.rs`,
//! invisible to anything (D-Bus/HTTP/`brain do`) that discovers capabilities
//! generically - the dedicated CLI was always more capable than the manifest
//! let on.

use std::path::Path;
use std::sync::Arc;

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use serde_json::json;

use crate::{GenOpts, TtsPaths};

/// The model id used on the CLI (`brain do tts …`) and the event API.
pub const MODEL: &str = "brain/qwen3tts";

/// Qwen3-TTS output sample rate (see [`crate::pipeline`] / `brain tts synth`).
const SAMPLE_RATE: u32 = 24_000;

/// The `weights_dir`/`ckpt`/`text` params + the common `GenOpts` sampling
/// knobs, shared by every action's spec so the three can't silently drift.
fn common_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("weights_dir", ParamType::Str, "dir holding talker.safetensors, mtp.safetensors, codec.safetensors (from `brain tts import`)").required().host_env("BRAIN_QWEN3TTS_WEIGHTS"))
        .param(ParamSpec::new("ckpt", ParamType::Str, "HF checkpoint dir (config.json + tokenizer)").required().host_env("BRAIN_QWEN3TTS_CKPT"))
        .param(ParamSpec::new("text", ParamType::Str, "the text to speak").required())
        .param(ParamSpec::new("lang", ParamType::Str, "synthesis language").default(json!("english")))
        .param(ParamSpec::new("max_frames", ParamType::Int, "max codec frames (length cap)").default(json!(256)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (0 = greedy, degenerate for this model)").default(json!(0.9)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k sampling cutoff").default(json!(50)))
        .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling cutoff (0 = disabled)").default(json!(0.0)))
        .param(ParamSpec::new("repetition_penalty", ParamType::Float, "repetition penalty (1.0 = disabled)").default(json!(1.0)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (reproducible run)").default(json!(0)))
}

fn gen_opts_from(inv: &Invocation) -> GenOpts {
    let mut opts = GenOpts::default();
    if let Some(f) = inv.get_i64("max_frames") {
        opts.max_frames = f.max(1) as usize;
    }
    if let Some(t) = inv.get_f64("temp") {
        opts.temperature = t as f32;
    }
    if let Some(k) = inv.get_i64("top_k") {
        opts.top_k = k.max(0) as usize;
    }
    if let Some(p) = inv.get_f64("top_p") {
        opts.top_p = p as f32;
    }
    if let Some(r) = inv.get_f64("repetition_penalty") {
        opts.repetition_penalty = r as f32;
    }
    if let Some(s) = inv.get_i64("seed") {
        opts.seed = s.max(0) as u64;
    }
    opts
}

fn audio_outcome(wav: Vec<f32>) -> ActionResult {
    let bytes = audio::wav::encode(&wav, SAMPLE_RATE);
    Ok(Outcome::new()
        .set("samples", json!(wav.len()))
        .set("sample_rate", json!(SAMPLE_RATE))
        .set("seconds", json!(wav.len() as f64 / SAMPLE_RATE as f64))
        .blob("audio", Blob::new(Media::Audio, bytes).with_meta(json!({"sample_rate": SAMPLE_RATE, "format": "wav", "channels": 1}))))
}

fn paths_from(weights_dir: &str, ckpt: String) -> TtsPaths {
    TtsPaths {
        talker: format!("{weights_dir}/talker.safetensors"),
        mtp: format!("{weights_dir}/mtp.safetensors"),
        codec: format!("{weights_dir}/codec.safetensors"),
        speaker: format!("{weights_dir}/speaker.safetensors"),
        ckpt_dir: ckpt,
    }
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let synth = common_params(ActionSpec::new("synth", "speaker-free text-to-speech (Talker + MTP + codec, 24 kHz wav)"))
        .output(BlobSpec::new("audio", Media::Audio, "the synthesized speech as a 24 kHz mono WAV"));
    let clone = common_params(ActionSpec::new("clone", "voice cloning from a reference wav (x-vector-only, or in-context when ref_text is given)"))
        .param(ParamSpec::new("ref", ParamType::Str, "reference audio wav path").required())
        .param(ParamSpec::new("ref_text", ParamType::Str, "transcript of the reference wav - given, runs in-context (ICL) cloning; omitted, x-vector-only").default(json!("")))
        .output(BlobSpec::new("audio", Media::Audio, "the cloned-voice speech as a 24 kHz mono WAV"));
    let design = common_params(ActionSpec::new("design", "VoiceDesign (instruct) and/or CustomVoice preset speakers - needs a CustomVoice/VoiceDesign checkpoint"))
        .param(ParamSpec::new("instruct", ParamType::Str, "natural-language voice/emotion/prosody description").default(json!("")))
        .param(ParamSpec::new("speaker", ParamType::Str, "CustomVoice preset speaker name").default(json!("")))
        .output(BlobSpec::new("audio", Media::Audio, "the designed-voice speech as a 24 kHz mono WAV"));
    Manifest::new(MODEL, "Qwen3-TTS voice synthesis - text to 24 kHz speech (Talker + MTP + codec).", vec![synth, clone, design])
}

/// The env-configured RESIDENT surface's `speak` action (`cli::resident_tts::
/// TtsResident` - weights resolved from `BRAIN_QWEN3TTS_WEIGHTS`/`BRAIN_QWEN3TTS_CKPT`
/// at registration, so no path params; declares the audio output the
/// instance actually emits). Lives HERE, next to [`manifest`]'s `synth`
/// spec, so the two surfaces' specs cannot silently diverge - the resident
/// used to build its own private Manifest in `crates/cli`.
pub fn speak_spec() -> ActionSpec {
    ActionSpec::new("speak", "synthesize speech from text (Qwen3-TTS, 24 kHz f32 PCM)")
        .param(ParamSpec::new("text", ParamType::Str, "the text to speak").required())
        .param(ParamSpec::new("lang", ParamType::Str, "synthesis language").default(json!("english")))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature").default(json!(0.9)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k sampling cutoff").default(json!(50)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (reproducible run)").default(json!(0)))
        .param(ParamSpec::new("max_frames", ParamType::Int, "max codec frames (length cap)").default(json!(256)))
        .output(BlobSpec::new("audio", Media::Audio, "the synthesized speech: raw mono f32 little-endian PCM at 24 kHz"))
}

/// The resident's `design` action - VoiceDesign/CustomVoice, per-call
/// `instruct`/`speaker` (unlike `speak`, whose voice is fixed at instance
/// configuration via `BRAIN_QWEN3TTS_REF`). Needs a CustomVoice/VoiceDesign
/// checkpoint; lives here for the same reason as [`speak_spec`].
pub fn design_spec() -> ActionSpec {
    ActionSpec::new("design", "VoiceDesign (instruct) and/or CustomVoice preset speakers (Qwen3-TTS, 24 kHz f32 PCM)")
        .param(ParamSpec::new("text", ParamType::Str, "the text to speak").required())
        .param(ParamSpec::new("lang", ParamType::Str, "synthesis language").default(json!("english")))
        .param(ParamSpec::new("instruct", ParamType::Str, "natural-language voice/emotion/prosody description").default(json!("")))
        .param(ParamSpec::new("speaker", ParamType::Str, "CustomVoice preset speaker name").default(json!("")))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature").default(json!(0.9)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k sampling cutoff").default(json!(50)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (reproducible run)").default(json!(0)))
        .param(ParamSpec::new("max_frames", ParamType::Int, "max codec frames (length cap)").default(json!(256)))
        .output(BlobSpec::new("audio", Media::Audio, "the designed-voice speech: raw mono f32 little-endian PCM at 24 kHz"))
}

/// The resident (D-Bus/HTTP) manifest - [`speak_spec`]/[`design_spec`] under
/// the same [`MODEL`] id the catalog's [`manifest`] uses.
pub fn resident_manifest() -> Manifest {
    Manifest::new(MODEL, "text-to-speech (Qwen3-TTS Talker + MTP + codec)", vec![speak_spec(), design_spec()])
}

/// The executable TTS model behind the manifest. Stateless: `pipeline::synth`
/// owns loading (per call), so construction is free and there is no hot cache.
#[derive(Default)]
pub struct TtsProvider;

impl TtsProvider {
    pub fn new() -> TtsProvider {
        TtsProvider
    }
}

impl Provider for TtsProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "synth" => Some(Arc::new(SynthAction)),
            "clone" => Some(Arc::new(CloneAction)),
            "design" => Some(Arc::new(DesignAction)),
            _ => None,
        }
    }
}

/// Common param extraction + weights-existence check every action shares.
/// `need_speaker` additionally requires `speaker.safetensors` (only `clone`
/// does - CustomVoice/VoiceDesign checkpoints don't ship a speaker encoder).
fn common_run(inv: &Invocation, action: &str, need_speaker: bool) -> Result<(TtsPaths, String, String, GenOpts), String> {
    let weights_dir = inv.get_str("weights_dir").ok_or_else(|| format!("tts {action}: missing required param 'weights_dir'"))?;
    let ckpt = inv.get_str("ckpt").ok_or_else(|| format!("tts {action}: missing required param 'ckpt'"))?;
    let text = inv.get_str("text").unwrap_or_default();
    if text.trim().is_empty() {
        return Err(format!("tts {action}: 'text' must be non-empty"));
    }
    let lang = inv.get_str("lang").unwrap_or_else(|| "english".to_string());
    let paths = paths_from(&weights_dir, ckpt);
    // Fail cleanly (not a panic in the loaders) when the checkpoints the
    // pipeline loads are absent.
    let mut need = vec![&paths.talker, &paths.mtp, &paths.codec];
    if need_speaker {
        need.push(&paths.speaker);
    }
    for p in need {
        if !Path::new(p).exists() {
            return Err(format!("tts {action}: weights not found at '{p}' (run `brain tts import`)"));
        }
    }
    Ok((paths, text, lang, gen_opts_from(inv)))
}

struct SynthAction;

impl Action for SynthAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "synth").expect("known action")
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (paths, text, lang, opts) = common_run(inv, "synth", false)?;
        let wav = crate::pipeline::synth(&paths, &opts, &text, &lang)?;
        audio_outcome(wav)
    }
}

struct CloneAction;

impl Action for CloneAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "clone").expect("known action")
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (paths, text, lang, opts) = common_run(inv, "clone", true)?;
        let refw = inv.get_str("ref").ok_or("tts clone: missing required param 'ref'")?;
        if !Path::new(&refw).exists() {
            return Err(format!("tts clone: reference wav not found at '{refw}'"));
        }
        let ref_text = inv.get_str("ref_text").unwrap_or_default();
        let wav = crate::pipeline::clone(&paths, &opts, &text, &refw, &ref_text, &lang, None)?;
        audio_outcome(wav)
    }
}

struct DesignAction;

impl Action for DesignAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "design").expect("known action")
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (paths, text, lang, opts) = common_run(inv, "design", false)?;
        let instruct = inv.get_str("instruct").unwrap_or_default();
        let speaker = inv.get_str("speaker").filter(|s| !s.is_empty());
        let wav = crate::pipeline::design(&paths, &opts, &text, &lang, &instruct, speaker.as_deref())?;
        audio_outcome(wav)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::Registry;

    #[test]
    fn manifest_declares_synth_clone_and_design() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 3);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["synth", "clone", "design"], "clone/design must be reachable generically, not only from tts_cli.rs");
        for a in &m.actions {
            assert!(!a.streaming, "{}: one-shot (the wav is the single artifact)", a.name);
            for req in ["weights_dir", "ckpt", "text"] {
                assert!(a.params.iter().any(|p| p.name == req && p.required), "{}: {req} must be required", a.name);
            }
            assert_eq!(a.params.iter().find(|p| p.name == "max_frames").unwrap().default, Some(json!(256)));
            assert_eq!(a.outputs[0].media, Media::Audio);
        }
        let synth = &m.actions[0];
        let inv = synth
            .validate(Invocation::new().set("weights_dir", json!("d")).set("ckpt", json!("c")).set("text", json!("hi")))
            .unwrap();
        assert_eq!(inv.get_f64("temp"), Some(0.9));
        assert!(synth.validate(Invocation::new().set("weights_dir", json!("d")).set("ckpt", json!("c"))).is_err());
        assert_eq!(manifest().to_json()["actions"][0]["name"], "synth");

        let clone = &m.actions[1];
        assert!(clone.params.iter().any(|p| p.name == "ref" && p.required), "clone: 'ref' must be required");
        assert!(clone.params.iter().any(|p| p.name == "ref_text" && !p.required));

        let design = &m.actions[2];
        assert!(design.params.iter().any(|p| p.name == "instruct" && !p.required));
        assert!(design.params.iter().any(|p| p.name == "speaker" && !p.required));
    }

    /// The imported TTS checkpoints are not on every box: missing weights must
    /// surface as a clean `ActionResult` error, not a panic in a loader, on
    /// EVERY action - not just `synth`, which is all the old test covered.
    #[test]
    fn missing_weights_is_a_clean_error_on_every_action() {
        let mut reg = Registry::new();
        reg.register(Arc::new(TtsProvider::new()));
        let base = || {
            Invocation::new()
                .set("weights_dir", json!("/nonexistent/tts"))
                .set("ckpt", json!("/nonexistent/ckpt"))
                .set("text", json!("hello"))
        };
        for action in ["synth", "design"] {
            let err = reg.run(MODEL, action, base(), &mut |_| {}).unwrap_err();
            assert!(err.contains("not found"), "{action} got: {err}");
        }
        let err = reg.run(MODEL, "clone", base().set("ref", json!("/nonexistent/ref.wav")), &mut |_| {}).unwrap_err();
        assert!(err.contains("not found"), "clone got: {err}");
    }

    /// `clone` needs a `speaker.safetensors` (the ECAPA x-vector encoder);
    /// `synth`/`design` don't, matching CustomVoice/VoiceDesign checkpoints
    /// that ship no speaker encoder at all. Uses a weights_dir with
    /// talker/mtp/codec present but speaker.safetensors absent, so the two
    /// `need_speaker` values are only distinguishable by that one file.
    #[test]
    fn only_clone_requires_the_speaker_encoder() {
        let dir = std::env::temp_dir().join(format!("qwen3tts-caps-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["talker.safetensors", "mtp.safetensors", "codec.safetensors"] {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        let inv = Invocation::new().set("weights_dir", json!(dir.to_str().unwrap())).set("ckpt", json!("c")).set("text", json!("hi"));
        assert!(common_run(&inv, "synth", false).is_ok(), "synth must not require speaker.safetensors");
        match common_run(&inv, "clone", true) {
            Err(e) => assert!(e.contains("speaker.safetensors"), "clone got: {e}"),
            Ok(_) => panic!("clone must require speaker.safetensors"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// End-to-end: a real generation through the SAME `capability::Registry`
    /// path D-Bus/HTTP/`brain do` use (not `pipeline::design` called
    /// directly, which the dedicated `tts_cli.rs` already exercises) -
    /// proves `caps.rs`'s new `clone`/`design` wiring, not just its
    /// argument-validation error paths. Gated on `BRAIN_QWEN3TTS_WEIGHTS`/
    /// `BRAIN_QWEN3TTS_CKPT` (the same env vars the CLI reads) pointing at a
    /// real imported checkpoint - skips cleanly when unset, same convention
    /// as every other real-checkpoint test in this crate.
    #[test]
    fn design_action_produces_real_audio_through_the_registry() {
        let (Ok(weights_dir), Ok(ckpt)) = (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT")) else {
            brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT not set");
            return;
        };
        if !Path::new(&format!("{weights_dir}/talker.safetensors")).exists() {
            brain_testutil::skip("weights not found at BRAIN_QWEN3TTS_WEIGHTS");
            return;
        }
        let mut reg = Registry::new();
        reg.register(Arc::new(TtsProvider::new()));
        let inv = Invocation::new()
            .set("weights_dir", json!(weights_dir))
            .set("ckpt", json!(ckpt))
            .set("text", json!("Reached through the generic action surface now."))
            .set("max_frames", json!(24));
        let outcome = reg.run(MODEL, "design", inv, &mut |_| {}).expect("design action should succeed");
        let samples = outcome.outputs["samples"].as_u64().expect("samples field");
        assert!(samples > 0, "design produced no audio");
        let audio = outcome.blobs.get("audio").expect("audio blob");
        assert!(!audio.bytes.is_empty(), "audio blob is empty");
        eprintln!("design-action e2e: {samples} samples, {} audio bytes", audio.bytes.len());
    }
}
