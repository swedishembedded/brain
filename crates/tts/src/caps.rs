// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS's capabilities behind the generalized [`capability`] interface —
//! what makes `brain caps tts` / `brain do tts synth …` (and the perf suite's
//! `CapabilityTarget`) work with no tts-specific plumbing in the CLI.
//!
//! One action, `synth`: speaker-free text-to-speech through the exact pipeline
//! `brain tts synth` runs ([`crate::pipeline::synth`]: Talker KV decode + MTP
//! residual fill + codec decode, 24 kHz). One-shot: the wav is the single
//! artifact. Like `pipeline::synth` itself (and the `TtsResident` adapter),
//! the weights load per call — the load-once seam (`serve::TtsEngine`) is
//! OpenVINO/NPU-only, so there is nothing resident to cache here without
//! duplicating the pipeline internals.

use std::path::Path;
use std::sync::Arc;

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use serde_json::json;

use crate::{GenOpts, TtsPaths};

/// The model id used on the CLI (`brain do tts …`) and the event API.
pub const MODEL: &str = "brain/tts";

/// Qwen3-TTS output sample rate (see [`crate::pipeline`] / `brain tts synth`).
const SAMPLE_RATE: u32 = 24_000;

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let synth = ActionSpec::new("synth", "speaker-free text-to-speech (Talker + MTP + codec, 24 kHz wav)")
        .param(ParamSpec::new("weights_dir", ParamType::Str, "dir holding talker.safetensors, mtp.safetensors, codec.safetensors (from `brain tts import`)").required())
        .param(ParamSpec::new("ckpt", ParamType::Str, "HF checkpoint dir (config.json + tokenizer)").required())
        .param(ParamSpec::new("text", ParamType::Str, "the text to speak").required())
        .param(ParamSpec::new("lang", ParamType::Str, "synthesis language").default(json!("english")))
        .param(ParamSpec::new("max_frames", ParamType::Int, "max codec frames (length cap)").default(json!(256)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (0 = greedy, degenerate for this model)").default(json!(0.9)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k sampling cutoff").default(json!(50)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (reproducible run)").default(json!(0)))
        .output(BlobSpec::new("audio", Media::Audio, "the synthesized speech as a 24 kHz mono WAV"));
    Manifest::new(MODEL, "Qwen3-TTS voice synthesis — text to 24 kHz speech (Talker + MTP + codec).", vec![synth])
}

/// The env-configured RESIDENT surface's `speak` action (`cli::resident_tts::
/// TtsResident` — weights resolved from `BRAIN_TTS_WEIGHTS`/`BRAIN_TTS_CKPT`
/// at registration, so no path params; declares the audio output the
/// instance actually emits). Lives HERE, next to [`manifest`]'s `synth`
/// spec, so the two surfaces' specs cannot silently diverge — the resident
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

/// The resident (D-Bus/HTTP) manifest — [`speak_spec`] under the same
/// [`MODEL`] id the catalog's [`manifest`] uses.
pub fn resident_manifest() -> Manifest {
    Manifest::new(MODEL, "text-to-speech (Qwen3-TTS Talker + MTP + codec)", vec![speak_spec()])
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
        (name == "synth").then(|| Arc::new(SynthAction) as Arc<dyn Action>)
    }
}

struct SynthAction;

impl Action for SynthAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "synth").expect("known action")
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights_dir = inv.get_str("weights_dir").ok_or("tts synth: missing required param 'weights_dir'")?;
        let ckpt = inv.get_str("ckpt").ok_or("tts synth: missing required param 'ckpt'")?;
        let text = inv.get_str("text").unwrap_or_default();
        if text.trim().is_empty() {
            return Err("tts synth: 'text' must be non-empty".to_string());
        }
        let lang = inv.get_str("lang").unwrap_or_else(|| "english".to_string());

        // The same layout `brain tts`'s paths() helper builds from --weights-dir.
        let paths = TtsPaths {
            talker: format!("{weights_dir}/talker.safetensors"),
            mtp: format!("{weights_dir}/mtp.safetensors"),
            codec: format!("{weights_dir}/codec.safetensors"),
            speaker: format!("{weights_dir}/speaker.safetensors"),
            ckpt_dir: ckpt,
        };
        // Fail cleanly (not a panic in the loaders) when the checkpoints the
        // synth path loads are absent. `speaker.safetensors` is not needed here.
        for p in [&paths.talker, &paths.mtp, &paths.codec] {
            if !Path::new(p).exists() {
                return Err(format!("tts synth: weights not found at '{p}' (run `brain tts import`)"));
            }
        }

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
        if let Some(s) = inv.get_i64("seed") {
            opts.seed = s.max(0) as u64;
        }

        let wav = crate::pipeline::synth(&paths, &opts, &text, &lang)?;
        let bytes = audio::wav::encode(&wav, SAMPLE_RATE);
        Ok(Outcome::new()
            .set("samples", json!(wav.len()))
            .set("sample_rate", json!(SAMPLE_RATE))
            .set("seconds", json!(wav.len() as f64 / SAMPLE_RATE as f64))
            .blob("audio", Blob::new(Media::Audio, bytes).with_meta(json!({"sample_rate": SAMPLE_RATE, "format": "wav", "channels": 1}))))
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::Registry;

    #[test]
    fn manifest_declares_synth() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1);
        let a = &m.actions[0];
        assert_eq!(a.name, "synth");
        assert!(!a.streaming, "synth is one-shot (the wav is the single artifact)");
        for req in ["weights_dir", "ckpt", "text"] {
            assert!(a.params.iter().any(|p| p.name == req && p.required), "{req} must be required");
        }
        assert_eq!(a.params.iter().find(|p| p.name == "max_frames").unwrap().default, Some(json!(256)));
        assert_eq!(a.outputs[0].media, Media::Audio);
        // validation without weights: defaults fill, missing text rejected.
        let inv = a
            .validate(Invocation::new().set("weights_dir", json!("d")).set("ckpt", json!("c")).set("text", json!("hi")))
            .unwrap();
        assert_eq!(inv.get_f64("temp"), Some(0.9));
        assert!(a.validate(Invocation::new().set("weights_dir", json!("d")).set("ckpt", json!("c"))).is_err());
        assert_eq!(manifest().to_json()["actions"][0]["name"], "synth");
    }

    /// The imported TTS checkpoints are not on every box: missing weights must
    /// surface as a clean `ActionResult` error, not a panic in a loader.
    #[test]
    fn missing_weights_is_a_clean_error() {
        let mut reg = Registry::new();
        reg.register(Arc::new(TtsProvider::new()));
        let inv = Invocation::new()
            .set("weights_dir", json!("/nonexistent/tts"))
            .set("ckpt", json!("/nonexistent/ckpt"))
            .set("text", json!("hello"));
        let err = reg.run(MODEL, "synth", inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("not found"), "got: {err}");
    }
}
