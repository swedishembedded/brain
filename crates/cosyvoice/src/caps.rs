// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CosyVoice's capabilities behind the generalized [`capability`] interface -
//! what makes `brain caps brain/cosyvoice` / `brain do cosyvoice synth …` /
//! `brain cosyvoice synth …`, and the D-Bus/HTTP surfaces built on top of the
//! same `Provider`/`ResidentModel` pair, work with no CosyVoice-specific
//! plumbing in the CLI or the transports.
//!
//! The manifest is **static** (no weights needed) so capability discovery is
//! free; only [`SynthAction`]'s own execution loads anything. One action,
//! `synth`, running the exact pipeline [`crate::pipeline::generate`]
//! implements - the SAME function the residency adapter
//! (`crates/cli/src/resident_cosyvoice.rs`) calls through [`synth_action`],
//! so there is one implementation of param decoding + generation + outcome
//! shaping, not two that could drift.
//!
//! Nothing here is held warm across calls: every real `synth` call reloads
//! all five checkpoints fresh inside `crate::pipeline::generate`'s own
//! sequential-stage scopes, exactly matching that function's own documented
//! RAM discipline. See `crates/cli/src/resident_cosyvoice.rs`'s module doc
//! for the resulting tension with the residency scheduler's own "reserved
//! while Hot" cost model, and the judgment call made about it.
//!
//! # `variant`: `cosyvoice2` (implemented) vs `cosyvoice3` (not yet)
//!
//! `crate::pipeline::generate` only wires CosyVoice 2's flow decoder (UNet
//! CFM), vocoder (non-causal HiFT) and LM branch. CosyVoice 3's own flow
//! decoder (`crate::cv3_flow`), vocoder (`crate::cv3_hift`) and
//! `SpecialTokenSource::SpeechEmbedding` LM branch are individually
//! forward-parity-proven against their real checkpoints, but composing them
//! into a second, streaming-aware `generate()` is a deliberate, separately
//! tracked follow-up, not attempted here. `variant="cosyvoice3"` is accepted
//! by this action's schema (so a client can discover the option and get a
//! clear, typed rejection instead of an "unknown param") but always fails
//! with an explicit error naming the gap - it never silently falls back to
//! CosyVoice 2's weights, and never panics.
//!
//! # Reference audio: a blob, not a server-side path
//!
//! `crate::pipeline::generate` takes a filesystem path for the reference
//! clip (`ref_wav_path`), because it is driven directly from
//! `crates/cosyvoice/examples/synth.rs` today. A served action cannot take a
//! path - a D-Bus/HTTP caller's reference clip lives on THEIR disk, not
//! this process's - so [`synth_action`] instead takes it as an `Media::Audio`
//! input blob (matching `qwen3omnimoe`'s speech-input convention) and
//! bridges it to `crate::pipeline::generate`'s path-based signature with a
//! short-lived scratch WAV file ([`ScratchWav`]), deleted once `generate`
//! returns (success or failure alike).
//!
//! One real, honestly-recorded consequence of reusing the generic CLI blob
//! loader: `crates/cli/src/caps_cli.rs`'s `--in ref_audio=clip.wav` decodes
//! ANY input WAV through `audio::asr_caps::audio_blob_from_wav`, which
//! downmixes to mono and resamples to a fixed 16 kHz before this action ever
//! sees the bytes - a convention shared by every ASR-style audio input in
//! this workspace, not something this action introduces. `crate::pipeline`
//! separately resamples the clip it is given to BOTH 16 kHz (CAM++/
//! S3Tokenizer) and 24 kHz (the prompt mel), so feeding it an
//! already-16-kHz-capped clip means the 24 kHz prompt mel is built from
//! audio with no content above 8 kHz - a real fidelity ceiling on the plain
//! CLI path, not a correctness bug (the pipeline still runs and produces
//! valid speech). A direct D-Bus/HTTP caller that builds its own `Blob`
//! (rather than going through `brain do ... --in`) is NOT subject to this
//! cap: [`decode_ref_audio`] honours a WAV container's own sample rate, or a
//! raw-PCM blob's own `meta.sample_rate`, either way preserving full-rate
//! audio when the caller supplies it directly.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

use crate::pipeline::{generate, CosyVoicePaths, GenOpts};

/// The model id used on the CLI (`brain do brain/cosyvoice …`, `brain
/// cosyvoice synth …`), over D-Bus/HTTP, and in the residency manifest.
pub const MODEL: &str = "brain/cosyvoice";

/// The two generations `synth`'s `variant` param accepts. See this module's
/// own doc for why only the first is actually runnable today.
pub const VARIANTS: [&str; 2] = ["cosyvoice2", "cosyvoice3"];

/// `Ok(())` for a runnable variant, else the exact, honest reason it cannot
/// run yet (never a panic, never a silent fallback to the other variant's
/// weights) - see this module's own doc for the CosyVoice 3 scope.
fn check_variant(variant: &str) -> Result<(), String> {
    match variant {
        "cosyvoice2" => Ok(()),
        "cosyvoice3" => Err(
            "cosyvoice synth: variant 'cosyvoice3' is not implemented yet - crate::pipeline only composes CosyVoice 2's flow decoder/vocoder/LM branch today; use variant='cosyvoice2'"
                .to_string(),
        ),
        other => Err(format!("cosyvoice synth: unknown variant '{other}' (expected one of {VARIANTS:?})")),
    }
}

fn synth_spec() -> ActionSpec {
    let d = GenOpts::default();
    ActionSpec::new(
        "synth",
        "zero-shot voice cloning: target text + a reference audio clip and its transcript -> a real 24 kHz WAV (CosyVoice 2's non-streaming pipeline; cosyvoice3 is accepted as a value but not implemented yet)",
    )
    .streaming()
    .param(ParamSpec::new("text", ParamType::Str, "the target text to synthesize").required())
    .param(ParamSpec::new("ref_text", ParamType::Str, "the reference clip's own transcript").required())
    .param(ParamSpec::new("variant", ParamType::Str, "cosyvoice2 (implemented) or cosyvoice3 (accepted, not yet implemented)").default(json!("cosyvoice2")))
    .param(ParamSpec::new("seed", ParamType::Int, "RNG seed for the LM sampler and HiFT NSF noise (reproducible on this port, not bit-exact vs the python reference)").default(json!(0)))
    .param(
        ParamSpec::new("n_timesteps", ParamType::Int, "Euler steps the flow decoder's CFM solver takes")
            .default(json!(d.n_timesteps as i64))
            .min(1.0)
            .max(50.0),
    )
    .input(BlobSpec::new("ref_audio", Media::Audio, "reference audio clip for zero-shot cloning: a WAV container at its own sample rate, or raw mono f32 little-endian PCM (meta.sample_rate, defaulting to 16 kHz when absent)").required())
    .output(BlobSpec::new("audio", Media::Audio, "the synthesized speech: a complete 24 kHz mono WAV"))
}

/// The full, static capability manifest - safe to build with no weights
/// loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "CosyVoice 2/3 - LLM-based streaming zero-shot TTS (Qwen2.5-0.5B speech-token LM, causal flow-matching mel decoder, ISTFT/NSF HiFT vocoder). Only cosyvoice2 is servable today.",
        vec![synth_spec()],
    )
}

/// [`manifest`] again, under the name the resident adapter
/// (`crates/cli/src/resident_cosyvoice.rs`) reaches for - identical today,
/// kept as its own function so the two surfaces have a named seam if they
/// ever need to diverge, matching `minimaxmusic3::caps::resident_manifest`'s
/// own precedent.
pub fn resident_manifest() -> Manifest {
    manifest()
}

/// A named-unique scratch WAV file, removed on drop regardless of how the
/// scope holding it exits (an early `?` included) - the bridge from a
/// caller-supplied [`Blob`] to `crate::pipeline::generate`'s path-based
/// signature. Never written under a repo path and never a fixture: this is
/// a transient per-call file, not test/golden data.
struct ScratchWav(std::path::PathBuf);

impl ScratchWav {
    fn write(samples: &[f32], sample_rate: u32) -> Result<ScratchWav, String> {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let path = std::env::temp_dir().join(format!("brain-cosyvoice-ref-{}-{nanos}.wav", std::process::id()));
        audio::wav::write(&path, samples, sample_rate).map_err(|e| format!("cosyvoice synth: writing the reference clip to a scratch wav: {e}"))?;
        Ok(ScratchWav(path))
    }
    fn path_str(&self) -> &str {
        self.0.to_str().expect("std::env::temp_dir() joined with an ASCII filename is always valid UTF-8")
    }
}

impl Drop for ScratchWav {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Decode the `ref_audio` input blob to `(samples, sample_rate)`. A WAV
/// container (any sample rate, `audio::asr_caps::is_wav`) is parsed
/// directly; anything else is treated as raw mono f32 little-endian PCM at
/// `meta.sample_rate`, defaulting to 16 kHz when the meta is absent (matching
/// this workspace's other raw-PCM audio blob conventions, `audio::asr_caps`
/// and `qwen3omnimoe::caps`). See this module's own doc for why a WAV
/// container's own rate is honoured rather than assumed.
fn decode_ref_audio(blob: &Blob) -> Result<(Vec<f32>, u32), String> {
    if audio::asr_caps::is_wav(&blob.bytes) {
        let wav = audio::wav::parse(&blob.bytes).map_err(|e| format!("cosyvoice synth: parsing 'ref_audio' as a wav container: {e}"))?;
        return Ok((wav.samples, wav.sample_rate));
    }
    if !blob.bytes.len().is_multiple_of(4) {
        return Err(format!("cosyvoice synth: 'ref_audio' blob length {} is not a multiple of 4 (expected f32 little-endian PCM or a WAV container)", blob.bytes.len()));
    }
    let sample_rate = blob.meta.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(16000) as u32;
    let samples: Vec<f32> = blob.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((samples, sample_rate))
}

/// Run one `synth` call and wrap the result as an audio-output [`Outcome`] -
/// ONE implementation, shared by [`CosyVoiceProvider`] and the residency
/// adapter. `paths` is resolved by the caller (direct vs. resident differ
/// only in error framing, not in how paths resolve - both ultimately read
/// [`CosyVoicePaths::from_env`]).
pub fn synth_action(paths: &CosyVoicePaths, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let text = inv.get_str("text").ok_or("cosyvoice synth: missing required param 'text'")?;
    let ref_text = inv.get_str("ref_text").ok_or("cosyvoice synth: missing required param 'ref_text'")?;
    let variant = inv.get_str("variant").unwrap_or_else(|| "cosyvoice2".to_string());
    check_variant(&variant)?;

    let blob = inv.get_blob("ref_audio").ok_or("cosyvoice synth: missing required input 'ref_audio'")?;
    let (samples, sample_rate) = decode_ref_audio(blob)?;

    let d = GenOpts::default();
    let opts = GenOpts {
        seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
        n_timesteps: inv.get_i64("n_timesteps").unwrap_or(d.n_timesteps as i64).max(1) as usize,
        ..d
    };

    let scratch = ScratchWav::write(&samples, sample_rate)?;
    let out = generate(paths, &opts, &text, scratch.path_str(), &ref_text)?;
    drop(scratch); // explicit: the scratch file must not outlive this call either way

    let bytes = audio::wav::encode(&out.samples, out.sample_rate);
    Ok(Outcome::new()
        .set("samples", json!(out.samples.len()))
        .set("sample_rate", json!(out.sample_rate))
        .set("seconds", json!(out.samples.len() as f32 / out.sample_rate as f32))
        .set("variant", json!(variant))
        .blob("audio", Blob::new(Media::Audio, bytes).with_meta(json!({"format": "wav", "sample_rate": out.sample_rate, "channels": 1}))))
}

/// The executable CosyVoice model behind the manifest. Stateless - see this
/// module's own doc for why nothing is held warm across calls.
#[derive(Default)]
pub struct CosyVoiceProvider;

impl CosyVoiceProvider {
    pub fn new() -> CosyVoiceProvider {
        CosyVoiceProvider
    }
}

impl Provider for CosyVoiceProvider {
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
        synth_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let paths = CosyVoicePaths::from_env().map_err(|e| format!("cosyvoice synth: {e}"))?;
        synth_action(&paths, inv, progress)
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
        assert_eq!(names, ["synth"]);
        let synth = &m.actions[0];
        assert!(synth.streaming, "a multi-stage, multi-second generation with no progress reporting is a hang, not a feature");
        let required: Vec<&str> = synth.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        assert_eq!(required, ["text", "ref_text"]);
        assert!(synth.inputs.iter().any(|b| b.name == "ref_audio" && b.media == Media::Audio && b.required));
        assert_eq!(synth.outputs.len(), 1);
        assert_eq!(synth.outputs[0].media, Media::Audio);
    }

    #[test]
    fn resident_manifest_matches_the_catalog_manifest() {
        assert_eq!(resident_manifest().model, manifest().model);
        assert_eq!(resident_manifest().actions.len(), manifest().actions.len());
    }

    #[test]
    fn provider_exposes_synth_and_nothing_else() {
        let p = CosyVoiceProvider::new();
        assert!(p.action("synth").is_some());
        assert!(p.action("nonexistent").is_none());
    }

    #[test]
    fn check_variant_accepts_cosyvoice2_and_names_cosyvoice3s_gap_without_panicking() {
        assert!(check_variant("cosyvoice2").is_ok());
        let err = check_variant("cosyvoice3").unwrap_err();
        assert!(err.contains("not implemented"), "unexpected error: {err}");
        let err = check_variant("bogus").unwrap_err();
        assert!(err.contains("unknown variant"), "unexpected error: {err}");
    }

    #[test]
    fn synth_action_rejects_a_missing_required_param() {
        let paths = CosyVoicePaths { llm: String::new(), flow: String::new(), hift: String::new(), s3tokenizer: String::new(), campplus: String::new(), tokenizer: String::new() };
        let inv = Invocation::new().set("ref_text", json!("hello"));
        let mut progress = |_: Progress| {};
        let err = synth_action(&paths, &inv, &mut progress).unwrap_err();
        assert!(err.contains("text"), "unexpected error: {err}");
    }

    #[test]
    fn synth_action_rejects_a_missing_ref_audio_blob() {
        let paths = CosyVoicePaths { llm: String::new(), flow: String::new(), hift: String::new(), s3tokenizer: String::new(), campplus: String::new(), tokenizer: String::new() };
        let inv = Invocation::new().set("text", json!("hi")).set("ref_text", json!("hello"));
        let mut progress = |_: Progress| {};
        let err = synth_action(&paths, &inv, &mut progress).unwrap_err();
        assert!(err.contains("ref_audio"), "unexpected error: {err}");
    }

    #[test]
    fn synth_action_rejects_cosyvoice3_before_touching_any_weights() {
        let paths = CosyVoicePaths { llm: String::new(), flow: String::new(), hift: String::new(), s3tokenizer: String::new(), campplus: String::new(), tokenizer: String::new() };
        let samples = vec![0.0f32; 1600];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let inv = Invocation::new()
            .set("text", json!("hi"))
            .set("ref_text", json!("hello"))
            .set("variant", json!("cosyvoice3"))
            .blob("ref_audio", Blob::new(Media::Audio, bytes).with_meta(json!({"sample_rate": 16000})));
        let mut progress = |_: Progress| {};
        let err = synth_action(&paths, &inv, &mut progress).unwrap_err();
        assert!(err.contains("not implemented"), "unexpected error: {err}");
    }

    #[test]
    fn decode_ref_audio_reads_a_wav_containers_own_sample_rate() {
        let samples = vec![0.1f32, -0.2, 0.3, -0.4];
        let wav_bytes = audio::wav::encode(&samples, 24000);
        let blob = Blob::new(Media::Audio, wav_bytes);
        let (decoded, sr) = decode_ref_audio(&blob).expect("wav decodes");
        assert_eq!(sr, 24000);
        assert_eq!(decoded.len(), samples.len());
        // `audio::wav::encode` is int16 PCM, so this round-trips only up to
        // quantization noise, not bit-exactly.
        for (a, b) in decoded.iter().zip(&samples) {
            assert!((a - b).abs() < 1e-3, "decoded {a} too far from encoded {b}");
        }
    }

    #[test]
    fn decode_ref_audio_honours_raw_pcm_meta_sample_rate_and_defaults_to_16k() {
        let samples = vec![0.5f32, -0.5];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let with_meta = Blob::new(Media::Audio, bytes.clone()).with_meta(json!({"sample_rate": 44100}));
        let (decoded, sr) = decode_ref_audio(&with_meta).expect("pcm decodes");
        assert_eq!(sr, 44100);
        assert_eq!(decoded, samples);

        let no_meta = Blob::new(Media::Audio, bytes);
        let (_, sr) = decode_ref_audio(&no_meta).expect("pcm decodes");
        assert_eq!(sr, 16000, "an absent meta.sample_rate must default to 16 kHz, matching this workspace's other raw-PCM audio blob conventions");
    }

    #[test]
    fn decode_ref_audio_rejects_a_misaligned_raw_payload() {
        let blob = Blob::new(Media::Audio, vec![0u8; 5]);
        assert!(decode_ref_audio(&blob).is_err());
    }

    #[test]
    fn scratch_wav_is_removed_on_drop() {
        let scratch = ScratchWav::write(&[0.0, 0.1, -0.1], 16000).expect("scratch wav writes");
        let path = scratch.0.clone();
        assert!(path.exists());
        drop(scratch);
        assert!(!path.exists(), "the scratch wav must not outlive the call that created it");
    }
}
