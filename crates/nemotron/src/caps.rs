// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability` surface for Nemotron 3.5 ASR — the shared pieces every serving
//! path reuses so transcription is exposed the one brain way (`brain do`, the
//! event API, D-Bus, the residency scheduler).
//!
//! The heavy model + tokenizer are held by the caller (the resident adapter builds
//! them once — see `cli::resident_asr`); this module owns the *contract*: the
//! [`ActionSpec`], the audio-blob decoding, and a thin [`Provider`] for the direct
//! `brain do` path. Audio arrives as **raw mono f32 little-endian PCM at 16 kHz**
//! (meta `{"sample_rate":16000}`) — the same convention the D-Bus fd transport and
//! the Python example use.

use std::sync::Arc;

use audio::asr_caps::{transcribe_spec, transcription_outcome, wav_from_blob};
use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Progress, Provider};

use crate::config::NemotronConfig;
use crate::model::NemotronAsr;
use crate::tokenizer::Detokenizer;

/// The model name advertised in the manifest.
pub const MODEL: &str = "nemotron";

/// The single-utterance manifest (one `transcribe` action; schema shared via
/// [`audio::asr_caps::transcribe_spec`]).
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "NVIDIA Nemotron 3.5 ASR Streaming 0.6B — speech-to-text.", vec![transcribe_spec()])
}

// ---------------------------------------------------------------- direct Provider

/// A loaded Nemotron model behind the `capability` interface for the direct
/// `brain do nemotron transcribe` / `Registry` path (the resident adapter has its
/// own build-once, batched instance and does not go through this).
pub struct NemotronProvider {
    model: Arc<NemotronAsr>,
    detok: Arc<Detokenizer>,
}

impl NemotronProvider {
    /// Load an HF checkpoint dir (weights + `tokenizer.json`) and build the model on
    /// a device-resolved `Gpu` (`BRAIN_DEVICE`).
    pub fn load(dir: &str, cfg: NemotronConfig) -> Result<NemotronProvider, String> {
        let model = Arc::new(NemotronAsr::from_hf(dir, cfg)?);
        let detok = Arc::new(Detokenizer::from_hf(dir)?);
        Ok(NemotronProvider { model, detok })
    }
}

impl Provider for NemotronProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "transcribe").then(|| Arc::new(TranscribeAction { model: self.model.clone(), detok: self.detok.clone() }) as Arc<dyn Action>)
    }
}

struct TranscribeAction {
    model: Arc<NemotronAsr>,
    detok: Arc<Detokenizer>,
}

impl Action for TranscribeAction {
    fn spec(&self) -> ActionSpec {
        transcribe_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let blob = inv.get_blob("audio").ok_or("nemotron transcribe: missing 'audio' input")?;
        let wav = wav_from_blob(blob)?;
        let prompt_id = inv.get_i64("prompt_id").unwrap_or(0).max(0) as usize;
        progress(Progress { step: 0, total: 1, message: "transcribing".into() });
        let tokens = self.model.transcribe(&wav, prompt_id);
        let text = self.detok.decode(&tokens);
        progress(Progress { step: 1, total: 1, message: text.clone() });
        Ok(transcription_outcome(text, &tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_advertises_transcribe() {
        let m = manifest();
        assert_eq!(m.model, "nemotron");
        assert_eq!(m.actions.len(), 1);
        assert_eq!(m.actions[0].name, "transcribe");
    }
}
