// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax Music 3's capabilities behind the generalized [`capability`]
//! interface - what makes `brain caps brain/minimaxmusic3` / `brain do
//! minimaxmusic3 generate …` and the D-Bus surface work with no
//! model-specific plumbing in the CLI or the runtime.
//!
//! The manifest is **static** (no weights needed) so capability discovery
//! is free; only [`MinimaxMusic3Provider`]'s own action execution loads
//! anything. One action, `generate`, running the exact pipeline
//! `crate::generate::generate` implements - the SAME function the
//! residency adapter (`crates/cli/src/resident_minimaxmusic3.rs`) calls,
//! so there is one implementation of param decoding + generation +
//! outcome shaping, not two that could drift (the `wan::caps`/`flux2::
//! caps` pattern).
//!
//! Unlike `wan::caps::WanProvider` (which holds a **hot** resident DiT
//! across calls), nothing here is held warm: every real generation
//! reloads all five components fresh (see `crate::generate`'s own doc for
//! why - this model's whole checkpoint does not fit in RAM even once on
//! the machine this port was built on, so a "keep it loaded" cache would
//! be actively wrong, not just unhelpful). This matches `qwen3tts::caps::
//! TtsProvider`'s stateless shape, not `wan`'s.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

use crate::generate::{generate, GenOpts, Paths};

/// The model id used on the CLI (`brain do brain/minimaxmusic3 …`) and the
/// event API.
pub const MODEL: &str = "brain/minimaxmusic3";

/// The one action every surface (direct, resident, D-Bus) declares -
/// generation params only, no weight paths (those come from the
/// `BRAIN_MINIMAXMUSIC3_*` env vars `crate::generate::Paths::from_env`
/// reads, matching `crates/arch`'s own `weights_env` registration for
/// this architecture).
pub fn generate_spec() -> ActionSpec {
    let d = GenOpts::default();
    ActionSpec::new("generate", "generate a song from lyrics and a caption (Qwen3-8B Global LLM AR sampling, CFG-guided flow-matching DiT denoise, DAC-style vocoder, 44.1 kHz stereo)")
        .streaming()
        .param(ParamSpec::new("lyrics", ParamType::Str, "song lyrics, with [verse]/[chorus]/etc structural tags").required())
        .param(ParamSpec::new("caption", ParamType::Str, "structured music description: genre, BPM, vocal timbre, instrumentation, arrangement").required())
        .param(ParamSpec::new("duration_seconds", ParamType::Float, "target song length in seconds (the AR stage may stop earlier)").default(json!(d.duration_seconds)))
        .param(ParamSpec::new("num_inference_steps", ParamType::Int, "Euler steps per denoise chunk").default(json!(d.num_inference_steps as i64)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (reproducible run)").default(json!(0)))
        .output(BlobSpec::new("audio", Media::Audio, "the generated song: a complete 44.1 kHz stereo WAV").required())
}

/// The full, static capability manifest - safe to build with no weights
/// loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "MiniMax Music 3 - lyrics + caption-conditioned music generation (Qwen3-8B Global LLM + RVQ depth decoder + flow-matching DiT + DAC-style vocoder).",
        vec![generate_spec()],
    )
}

/// [`manifest`] again, under the name the resident adapter
/// (`crates/cli/src/resident_minimaxmusic3.rs`) reaches for - identical
/// today (one action, no path params either way), kept as its own
/// function so the two surfaces have a named seam if they ever need to
/// diverge, matching `qwen3tts::caps::resident_manifest`'s own precedent.
pub fn resident_manifest() -> Manifest {
    manifest()
}

/// Decode + validate `generate`'s params from an invocation.
fn opts_from(inv: &Invocation) -> GenOpts {
    let d = GenOpts::default();
    GenOpts {
        duration_seconds: inv.get_f64("duration_seconds").map(|v| v as f32).unwrap_or(d.duration_seconds).max(0.1),
        num_inference_steps: inv.get_i64("num_inference_steps").unwrap_or(d.num_inference_steps as i64).max(1) as usize,
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
    }
}

/// Run one `generate` call and wrap the result as an audio-output
/// [`Outcome`] - ONE implementation, shared by [`MinimaxMusic3Provider`]
/// and the residency adapter. `paths` is resolved by the caller (direct
/// vs. resident differ only in error framing, not in how paths resolve -
/// both ultimately read [`Paths::from_env`]).
pub fn generate_action(paths: &Paths, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let lyrics = inv.get_str("lyrics").ok_or("minimaxmusic3 generate: missing required param 'lyrics'")?;
    let caption = inv.get_str("caption").ok_or("minimaxmusic3 generate: missing required param 'caption'")?;
    if caption.trim().is_empty() {
        return Err("minimaxmusic3 generate: 'caption' must be non-empty".to_string());
    }
    let opts = opts_from(inv);
    let song = generate(paths, &opts, &lyrics, &caption)?;

    // Pack a COMPLETE WAV byte stream (stereo, via audio::wav::encode_multi)
    // rather than headerless PCM: `caps_cli::save_blob`'s audio arm only
    // knows how to write a header for MONO raw PCM, and `meta.format ==
    // "wav"` is this repo's own established convention for a
    // pre-encoded stream (qwen3tts's `synth` action does the same).
    let bytes = audio::wav::encode_multi(&[&song.left, &song.right], song.sample_rate);
    Ok(Outcome::new()
        .set("samples", json!(song.left.len()))
        .set("sample_rate", json!(song.sample_rate))
        .set("seconds", json!(song.left.len() as f32 / song.sample_rate as f32))
        .blob("audio", Blob::new(Media::Audio, bytes).with_meta(json!({"format": "wav", "sample_rate": song.sample_rate, "channels": 2}))))
}

/// The executable MiniMax Music 3 model behind the manifest. Stateless -
/// see this module's own doc for why nothing is held warm across calls.
#[derive(Default)]
pub struct MinimaxMusic3Provider;

impl MinimaxMusic3Provider {
    pub fn new() -> MinimaxMusic3Provider {
        MinimaxMusic3Provider
    }
}

impl Provider for MinimaxMusic3Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "generate").then(|| Arc::new(GenerateAction) as Arc<dyn Action>)
    }
}

struct GenerateAction;

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        generate_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let paths = Paths::from_env().map_err(|e| format!("minimaxmusic3 generate: {e}"))?;
        generate_action(&paths, inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_declares_the_full_surface() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["generate"]);
        let gen = &m.actions[0];
        assert!(gen.streaming, "a multi-minute generation with no progress reporting is a hang, not a feature");
        let required: Vec<&str> = gen.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["lyrics", "caption"]);
        assert_eq!(gen.outputs.len(), 1);
        assert_eq!(gen.outputs[0].media, Media::Audio);
    }

    #[test]
    fn resident_manifest_matches_the_catalog_manifest() {
        assert_eq!(resident_manifest().model, manifest().model);
        assert_eq!(resident_manifest().actions.len(), manifest().actions.len());
    }

    #[test]
    fn provider_exposes_generate_and_nothing_else() {
        let p = MinimaxMusic3Provider::new();
        assert!(p.action("generate").is_some());
        assert!(p.action("nonexistent").is_none());
    }

    #[test]
    fn generate_action_rejects_a_missing_required_param() {
        let paths = Paths { lm: String::new(), depth: String::new(), condition: String::new(), dit: String::new(), vocoder: String::new(), tokenizer: String::new() };
        let inv = Invocation::new().set("caption", json!("test"));
        let mut progress = |_: Progress| {};
        let err = generate_action(&paths, &inv, &mut progress).unwrap_err();
        assert!(err.contains("lyrics"), "unexpected error: {err}");
    }
}
