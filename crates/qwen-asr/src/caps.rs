// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability` surface for Qwen3-ASR — speech-to-text behind the shared ASR
//! contract ([`audio::asr_caps`]).
//!
//! Qwen3-ASR is an **offline** model: a Whisper-style audio tower feeds a spliced
//! Qwen3-1.7B decoder, and the decoder is assembled for a *fixed* audio-placeholder
//! run. A served instance is therefore built once **for a fixed window** — the
//! encoder is probed at load for the exact audio-token count of a full window (the
//! chunked packing makes it non-analytic), the decoder is assembled for that count,
//! and each clip is padded/truncated to the window before transcription. The chat
//! prompt is fixed apart from the number of audio placeholders, so its prefix (up to
//! `<|audio_bos|>`) and suffix (from `<|audio_eos|>`) are constants; `build_input_ids`
//! splices `n_audio` placeholder ids between them.

use std::sync::{Arc, Mutex};

use audio::asr_caps::{transcribe_spec, transcription_outcome, wav_from_blob};
use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Progress, Provider};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

use crate::config::QwenAsrConfig;
use crate::model::Qwen3Asr;

/// Model name in the manifest.
pub const MODEL: &str = "qwen-asr";

/// Contiguous audio-placeholder token id (`config.audio_token_id`).
pub const AUDIO_TOKEN_ID: u32 = 151676;
/// Row where the audio-placeholder run begins in the fixed prompt.
pub const AUDIO_ROW0: u32 = 10;
/// End-of-stream ids that stop greedy decoding (`<|endoftext|>`, `<|im_end|>`).
pub const EOS: [u32; 2] = [151643, 151645];
/// The marker the model emits between its `language <Lang>` preamble and the actual
/// transcription; text after the last occurrence is the transcription.
pub const ASR_TEXT_MARKER: &str = "<asr_text>";

/// Prompt token ids BEFORE the audio-placeholder run — the Qwen3-ASR chat template
/// up to and including `<|audio_bos|>` (151669). Extracted from the HF processor's
/// output; audio-length-independent.
pub const PROMPT_PREFIX: [u32; 10] = [151644, 8948, 198, 22574, 151645, 198, 151644, 872, 198, 151669];
/// Prompt token ids AFTER the audio-placeholder run — `<|audio_eos|>` (151670) then
/// `<|im_end|>\n<|im_start|>assistant\n`.
pub const PROMPT_SUFFIX: [u32; 6] = [151670, 151645, 198, 151644, 77091, 198];

/// Assemble the decoder prompt ids for `n_audio` audio placeholders:
/// `PREFIX ++ [AUDIO_TOKEN_ID; n_audio] ++ SUFFIX`.
pub fn build_input_ids(n_audio: u32) -> Vec<u32> {
    let mut ids = Vec::with_capacity(PROMPT_PREFIX.len() + n_audio as usize + PROMPT_SUFFIX.len());
    ids.extend_from_slice(&PROMPT_PREFIX);
    ids.extend(std::iter::repeat(AUDIO_TOKEN_ID).take(n_audio as usize));
    ids.extend_from_slice(&PROMPT_SUFFIX);
    ids
}

/// Pad (with silence) or truncate a waveform to exactly `window_samples` — so the
/// front end reports a full window of valid frames and the encoder yields the fixed
/// `n_audio` the decoder was assembled for.
pub fn pad_to_window(wav: &[f32], window_samples: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; window_samples];
    let n = wav.len().min(window_samples);
    out[..n].copy_from_slice(&wav[..n]);
    out
}

/// The manifest (one `transcribe` action; schema shared via [`audio::asr_caps`]).
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "Qwen3-ASR 1.7B — offline speech-to-text (fixed audio window).", vec![transcribe_spec()])
}

/// A loaded, fixed-window Qwen3-ASR model behind the `capability` interface.
pub struct QwenAsrProvider {
    inner: Arc<QwenAsrInner>,
}

/// Shared inner state (behind a `Mutex` because the decoder mutates device KV
/// buffers via interior mutability and must serialize).
pub struct QwenAsrInner {
    pub model: Mutex<Qwen3Asr>,
    pub tok: QwenBpe,
    pub input_ids: Vec<u32>,
    pub window_samples: usize,
    pub max_new: usize,
}

impl QwenAsrProvider {
    /// Load a Qwen3-ASR checkpoint for a fixed `window_secs` audio window.
    pub fn load(dir: &str, cfg: QwenAsrConfig, window_secs: f32, max_new: usize) -> Result<QwenAsrProvider, String> {
        let window_samples = (window_secs * 16_000.0) as usize;
        let (model, n_audio) = Qwen3Asr::from_hf_windowed(dir, cfg, window_samples, AUDIO_ROW0, max_new as u32)?;
        let tok = QwenBpe::from_dir(dir)?;
        let input_ids = build_input_ids(n_audio);
        Ok(QwenAsrProvider {
            inner: Arc::new(QwenAsrInner { model: Mutex::new(model), tok, input_ids, window_samples, max_new }),
        })
    }

    /// Transcribe one waveform (padded/truncated to the window) → text + token ids.
    pub fn transcribe(&self, wav: &[f32]) -> Result<(String, Vec<u32>), String> {
        self.inner.transcribe(wav)
    }

    /// [`transcribe`](Self::transcribe) with the audio-encoder HEAD run by a closure
    /// — the seam the NPU resident uses to run the audio-tower ONNX head on the
    /// Intel NPU while the conv stem + Qwen decoder stay on the device backend.
    pub fn transcribe_with_head<F>(&self, wav: &[f32], head: F) -> Result<(String, Vec<u32>), String>
    where
        F: FnOnce(&[f32], u32, &[(u32, u32)]) -> (Vec<f32>, Vec<f32>),
    {
        self.inner.transcribe_with_head(wav, head)
    }
}

impl QwenAsrInner {
    /// The shared transcribe path (also used by the resident adapter).
    pub fn transcribe(&self, wav: &[f32]) -> Result<(String, Vec<u32>), String> {
        let padded = pad_to_window(wav, self.window_samples);
        let (mel, valid, _n) = audio::asr_frontend::qwen_logmel(&padded, self.window_samples);
        let model = self.model.lock().map_err(|_| "qwen-asr: model lock poisoned")?;
        let embeds = model.encode_audio(&mel, valid as u32);
        self.decode(&model, &embeds)
    }

    /// [`transcribe`](Self::transcribe) with the audio-encoder head supplied by a
    /// closure (the NPU path); everything else — window padding, frontend, decoder,
    /// detok — is identical.
    pub fn transcribe_with_head<F>(&self, wav: &[f32], head: F) -> Result<(String, Vec<u32>), String>
    where
        F: FnOnce(&[f32], u32, &[(u32, u32)]) -> (Vec<f32>, Vec<f32>),
    {
        let padded = pad_to_window(wav, self.window_samples);
        let (mel, valid, _n) = audio::asr_frontend::qwen_logmel(&padded, self.window_samples);
        let model = self.model.lock().map_err(|_| "qwen-asr: model lock poisoned")?;
        let embeds = model.encode_audio_with_head(&mel, valid as u32, head);
        self.decode(&model, &embeds)
    }

    /// Splice the audio embeds into the Qwen decoder, greedy-decode, and detok —
    /// the tail shared by [`transcribe`](Self::transcribe) and
    /// [`transcribe_with_head`](Self::transcribe_with_head).
    fn decode(&self, model: &crate::model::Qwen3Asr, embeds: &[f32]) -> Result<(String, Vec<u32>), String> {
        let out = model.transcribe(&self.input_ids, embeds, &EOS, self.max_new);
        // strip trailing EOS before detok
        let trimmed: Vec<u32> = out.iter().copied().take_while(|t| !EOS.contains(t)).collect();
        // The model emits a `language <Lang><asr_text>` preamble before the words
        // (HF's processor strips this with return_format="transcription_only"); keep
        // only what follows the last `<asr_text>` marker.
        let decoded = self.tok.decode(&trimmed);
        let text = decoded.rsplit(ASR_TEXT_MARKER).next().unwrap_or(&decoded).trim().to_string();
        Ok((text, trimmed))
    }
}

impl Provider for QwenAsrProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "transcribe").then(|| Arc::new(TranscribeAction { inner: self.inner.clone() }) as Arc<dyn Action>)
    }
}

struct TranscribeAction {
    inner: Arc<QwenAsrInner>,
}

impl Action for TranscribeAction {
    fn spec(&self) -> ActionSpec {
        transcribe_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let blob = inv.get_blob("audio").ok_or("qwen-asr transcribe: missing 'audio' input")?;
        let wav = wav_from_blob(blob)?;
        progress(Progress::step(0, 1, "transcribing"));
        let (text, tokens) = self.inner.transcribe(&wav)?;
        progress(Progress::step(1, 1, text.clone()));
        Ok(transcription_outcome(text, &tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_and_window() {
        // input_ids = prefix + n placeholders + suffix
        let ids = build_input_ids(76);
        assert_eq!(ids.len(), PROMPT_PREFIX.len() + 76 + PROMPT_SUFFIX.len());
        assert_eq!(&ids[..PROMPT_PREFIX.len()], &PROMPT_PREFIX);
        assert_eq!(ids.iter().filter(|&&t| t == AUDIO_TOKEN_ID).count(), 76);
        assert_eq!(&ids[ids.len() - PROMPT_SUFFIX.len()..], &PROMPT_SUFFIX);
        // padding/truncation to the window
        assert_eq!(pad_to_window(&[1.0, 2.0], 4), vec![1.0, 2.0, 0.0, 0.0]);
        assert_eq!(pad_to_window(&[1.0, 2.0, 3.0, 4.0, 5.0], 3), vec![1.0, 2.0, 3.0]);
        assert_eq!(manifest().model, "qwen-asr");
    }

    #[test]
    fn strips_language_preamble() {
        // The `language <Lang><asr_text>` preamble is dropped; only the words remain.
        let decoded = "language English<asr_text>Mr. Quilter.";
        let text = decoded.rsplit(ASR_TEXT_MARKER).next().unwrap_or(decoded).trim();
        assert_eq!(text, "Mr. Quilter.");
        // no marker → unchanged
        let plain = "just words";
        assert_eq!(plain.rsplit(ASR_TEXT_MARKER).next().unwrap(), "just words");
    }
}
