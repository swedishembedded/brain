// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared `capability` contract for speech-to-text models (Nemotron, Qwen3-ASR).
//!
//! The audio-in / text-out shape is identical across ASR models, so the schema, the
//! audio-blob decoding and the result packing live here **once** (per the repo's
//! one-implementation rule) and each model crate adds only its own manifest name and
//! `Provider`. Audio is **raw mono f32 little-endian PCM at 16 kHz** (meta
//! `{"sample_rate":16000}`) — the convention the D-Bus fd transport and the Python
//! example use.

use capability::{ActionSpec, Blob, BlobSpec, Media, Outcome, ParamSpec, ParamType};

/// The `transcribe` action schema: one required `audio` blob in, `text` out, a
/// `prompt_id` param, marked streaming (progress frames while running).
pub fn transcribe_spec() -> ActionSpec {
    ActionSpec::new("transcribe", "transcribe 16 kHz mono speech to text")
        .param(ParamSpec::new("prompt_id", ParamType::Int, "language-prompt id (0 = en / default)").default(serde_json::json!(0)))
        .param(ParamSpec::new("sample_rate", ParamType::Int, "input PCM sample rate; must be 16000").default(serde_json::json!(16000)))
        .input(BlobSpec::new("audio", Media::Audio, "raw mono f32 little-endian PCM at 16 kHz").required())
        .output(BlobSpec::new("text", Media::Text, "the transcription"))
        .streaming()
}

/// The `transcribe_stream` action schema: one window of a live session. `stream`
/// names the session (created on first use, per serving instance); `eos` flushes
/// and closes it. Each call returns the session's *newly emitted* text/tokens —
/// concatenating every segment reproduces the offline transcription. The audio
/// blob is optional so a final `eos`-only call can flush a closed microphone.
pub fn transcribe_stream_spec() -> ActionSpec {
    ActionSpec::new("transcribe_stream", "frame-synchronous streaming transcription; one window of a live session")
        .param(ParamSpec::new("stream", ParamType::Str, "session id; state persists across calls until eos"))
        .param(ParamSpec::new("eos", ParamType::Bool, "flush and close the session after this window").default(serde_json::json!(false)))
        .param(ParamSpec::new("prompt_id", ParamType::Int, "language-prompt id (0 = en / default); fixed at session creation").default(serde_json::json!(0)))
        .param(ParamSpec::new("sample_rate", ParamType::Int, "input PCM sample rate; must be 16000").default(serde_json::json!(16000)))
        .input(BlobSpec::new("audio", Media::Audio, "raw mono f32 little-endian PCM at 16 kHz (may be absent on the final eos call)"))
        .output(BlobSpec::new("text", Media::Text, "text newly emitted by this window"))
        .streaming()
}

/// The sample rate every brain ASR front end is fixed at, and the rate the
/// `audio` blob wire format is defined in.
pub const ASR_SAMPLE_RATE: u32 = 16000;

/// Does `bytes` start with a canonical RIFF/WAVE header? Used by the callers
/// that accept EITHER a container file or an already-raw PCM payload (the CLI's
/// `--in audio=...`) and must tell the two apart without guessing.
pub fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

/// Decode a whole WAV **file** into brain's `audio` blob wire format: mono f32
/// little-endian PCM at 16 kHz with `meta = {"sample_rate": 16000}`.
///
/// The single implementation of "WAV file → brain audio blob", shared by every
/// surface that accepts a container file from a client: the HTTP `input_audio`
/// content part (`apiserve::media`) and the CLI's `--in audio=clip.wav`
/// (`brain do`). Multi-channel input is downmixed to mono by [`crate::wav::parse`]
/// itself; any source rate is linearly resampled to 16 kHz.
pub fn audio_blob_from_wav(bytes: &[u8]) -> Result<Blob, String> {
    let wav = crate::wav::parse(bytes).map_err(|e| e.to_string())?;
    let samples = crate::resample_linear(&wav.samples, wav.sample_rate, ASR_SAMPLE_RATE);
    Ok(audio_blob_from_samples(&samples))
}

/// Pack already-16 kHz mono f32 samples into the `audio` blob wire format.
pub fn audio_blob_from_samples(samples: &[f32]) -> Blob {
    let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    Blob::new(Media::Audio, pcm).with_meta(serde_json::json!({ "sample_rate": ASR_SAMPLE_RATE }))
}

/// Decode an `audio` [`Blob`] to a 16 kHz mono f32 waveform. Rejects a non-16 kHz
/// `sample_rate` (the ASR front ends are fixed at 16 kHz) and a byte length that is
/// not a whole number of f32 samples.
pub fn wav_from_blob(blob: &Blob) -> Result<Vec<f32>, String> {
    if !blob.bytes.len().is_multiple_of(4) {
        return Err(format!("audio blob length {} is not a multiple of 4 (expected f32 LE PCM)", blob.bytes.len()));
    }
    if let Some(sr) = blob.meta.get("sample_rate").and_then(|v| v.as_u64()) {
        if sr != 16000 {
            return Err(format!("transcribe: sample_rate must be 16000, got {sr}"));
        }
    }
    Ok(blob.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Pack a transcription into an [`Outcome`]: `text` blob + `text`/`tokens`/
/// `num_tokens` scalars. Identical wire shape for every ASR model.
pub fn transcription_outcome(text: String, tokens: &[u32]) -> Outcome {
    Outcome::new()
        .set("text", serde_json::json!(text))
        .set("tokens", serde_json::json!(tokens))
        .set("num_tokens", serde_json::json!(tokens.len()))
        .blob("text", Blob::new(Media::Text, text.into_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_and_blob_roundtrip() {
        let s = transcribe_spec();
        assert_eq!(s.name, "transcribe");
        assert!(s.streaming);
        assert!(s.inputs.iter().any(|b| b.name == "audio" && b.media == Media::Audio && b.required));
        let samples = [0.0f32, 1.0, -0.5, 0.25];
        let bytes: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
        let blob = Blob::new(Media::Audio, bytes).with_meta(serde_json::json!({"sample_rate": 16000}));
        assert_eq!(wav_from_blob(&blob).unwrap(), samples);
        assert!(wav_from_blob(&Blob::new(Media::Audio, vec![0u8; 4]).with_meta(serde_json::json!({"sample_rate": 44100}))).is_err());
        assert!(wav_from_blob(&Blob::new(Media::Audio, vec![0u8; 5])).is_err());
    }
}
