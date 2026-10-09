// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`TranscribePipeline`]: brain's speech-to-text surface, one-shot, over
//! qwen3-asr (Qwen3-ASR-1.7B, offline, a fixed audio decode window) or
//! nemotronasr (Nemotron-3.5-ASR, a transducer with no fixed window).
//! Resolved through `crates/loader`'s resolver over the requested model's own directory, against
//! `qwen3asr::spec::Qwen3AsrSpec` and `nemotronasr::spec::NemotronAsrSpec`;
//! one public type that dispatches on the resolved architecture, like
//! [`crate::TtsPipeline`].
//!
//! Nemotron is a *streaming* model, and only its one-shot path is exposed
//! here: a streaming transcriber needs a genuinely different call shape
//! (feed chunks, get segments back incrementally) than this pipeline's
//! `transcribe(wav) -> Transcript`, tracked as a separate extension. The
//! one-shot path matters because an independent second recognizer is what a
//! speech round trip is judged with.
//!
//! ```no_run
//! let pipe = brain::TranscribePipeline::from_pretrained("Qwen/Qwen3-ASR-1.7B")?;
//! let out = pipe.transcribe_wav(&std::fs::read("clip.wav")?)?;
//! println!("{}", out.text);
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

pub use qwen3asr::AudioFeatures;

use crate::{Device, Error, Result};

/// One transcription. `truncated` is `Some((available_secs, window_secs))`
/// when the input exceeded this pipeline's fixed decode window - everything
/// past `window_secs` was dropped (the window is a construction-time graph
/// size), surfaced explicitly rather than silently transcribing a prefix
/// (the served path treats this the same way, for the same reason: a
/// caller who trusted a silently-partial transcript once is the audited bug
/// class this field exists to prevent).
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub tokens: Vec<u32>,
    pub truncated: Option<(f32, f32)>,
}

/// `brain`'s speech-to-text pipeline. See this module's doc for scope.
pub struct TranscribePipeline {
    backend: Backend,
}

enum Backend {
    Qwen3Asr(qwen3asr::caps::QwenAsrProvider),
    Nemotron(Box<Nemotron>),
}

/// The rate recognition takes: 16 kHz mono.
const ASR_SAMPLE_RATE: u32 = 16_000;

/// The Nemotron model with its detokenizer, boxed in [`Backend`] because it is
/// far larger than the other variant.
struct Nemotron {
    model: nemotronasr::model::NemotronAsr,
    detokenizer: nemotronasr::tokenizer::Detokenizer,
}

/// Language prompt index for English, the only prompt this pipeline selects.
const NEMOTRON_ENGLISH_PROMPT: usize = 0;

impl std::fmt::Debug for TranscribePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("TranscribePipeline");
        match &self.backend {
            Backend::Qwen3Asr(p) => d.field("backend", &"qwen3asr").field("window_samples", &p.window_samples()),
            Backend::Nemotron(_) => d.field("backend", &"nemotronasr"),
        };
        d.finish()
    }
}

impl TranscribePipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<TranscribePipeline> {
        TranscribePipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> TranscribePipelineBuilder {
        TranscribePipelineBuilder {
            model_id: model_id.as_ref().to_string(),
            device: Device::default(),
            window_secs: DEFAULT_WINDOW_SECS,
            max_new_tokens: DEFAULT_MAX_NEW,
            download_policy: loader::DownloadPolicy::default(),
        }
    }

    /// Transcribe already-16 kHz mono f32 PCM.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        match &self.backend {
            Backend::Qwen3Asr(provider) => {
                let truncated = qwen3asr::caps::window_truncation(provider.window_samples(), samples);
                let (text, tokens) = provider.transcribe(samples).map_err(Error::Backend)?;
                Ok(Transcript { text, tokens, truncated })
            }
            Backend::Nemotron(n) => {
                let (model, detokenizer) = (&n.model, &n.detokenizer);
                let blank = model.config().blank_token_id;
                let tokens: Vec<u32> = model.transcribe(samples, NEMOTRON_ENGLISH_PROMPT).into_iter().filter(|&t| t != blank).collect();
                Ok(Transcript { text: detokenizer.decode(&tokens), tokens, truncated: None })
            }
        }
    }

    /// Transcribe a synthesized or loaded [`crate::Audio`] clip at whatever
    /// rate it has, resampled to 16 kHz by the SAME `audio::resample_linear`
    /// every surface in this workspace uses. This is the step a speech round
    /// trip needs between [`crate::TtsPipeline`] (24 kHz) and recognition.
    pub fn transcribe_audio(&self, clip: &crate::Audio) -> Result<Transcript> {
        self.transcribe(&audio::resample_linear(clip.samples(), clip.sample_rate(), ASR_SAMPLE_RATE))
    }

    /// The audio encoder's features of `clip` at any rate and length: what a
    /// caller needs to splice speech into a language model of its own rather
    /// than turn it into text. Qwen3-ASR only; the Nemotron backend is a
    /// transducer with no features of that kind and returns
    /// [`Error::MissingArgument`].
    pub fn features(&self, clip: &crate::Audio) -> Result<AudioFeatures> {
        let Backend::Qwen3Asr(provider) = &self.backend else {
            return Err(Error::MissingArgument("nemotronasr: has no spliceable audio features; use a Qwen3-ASR pipeline".to_string()));
        };
        provider.features(&audio::resample_linear(clip.samples(), clip.sample_rate(), ASR_SAMPLE_RATE)).map_err(Error::Backend)
    }

    /// [`Self::features`] for several clips with the audio encoder built once,
    /// on the ambient device. Building it uploads the whole audio tower, which
    /// costs more than encoding a short clip.
    pub fn features_many(&self, clips: &[&crate::Audio]) -> Result<Vec<AudioFeatures>> {
        let Backend::Qwen3Asr(provider) = &self.backend else {
            return Err(Error::MissingArgument("nemotronasr: has no spliceable audio features; use a Qwen3-ASR pipeline".to_string()));
        };
        let resampled: Vec<Vec<f32>> = clips.iter().map(|c| audio::resample_linear(c.samples(), c.sample_rate(), ASR_SAMPLE_RATE)).collect();
        let views: Vec<&[f32]> = resampled.iter().map(Vec::as_slice).collect();
        provider.features_many(&views).map_err(Error::Backend)
    }

    /// Decode a WAV file's bytes (any channel count/sample rate - downmixed
    /// to mono and resampled to 16 kHz by the SAME shared decode every
    /// other surface in this workspace uses,
    /// `audio::asr_caps::audio_blob_from_wav`) and transcribe it.
    pub fn transcribe_wav(&self, wav_bytes: &[u8]) -> Result<Transcript> {
        let blob = audio::asr_caps::audio_blob_from_wav(wav_bytes).map_err(Error::Backend)?;
        let samples = audio::asr_caps::wav_from_blob(&blob).map_err(Error::Backend)?;
        self.transcribe(&samples)
    }
}

const DEFAULT_WINDOW_SECS: f32 = 30.0;
const DEFAULT_MAX_NEW: usize = 200;

/// Builds a [`TranscribePipeline`]. `.device(...)`/`.window_secs(...)`/
/// `.max_new_tokens(...)` are the only knobs this milestone exposes -
/// `window_secs`/`max_new_tokens` default to the SAME values
/// `resident_asr.rs`'s own `BRAIN_QWEN3ASR_WINDOW`/`BRAIN_QWEN3ASR_MAXNEW`
/// fall back to, so an embedder who never touches them gets the served
/// path's own default behavior.
pub struct TranscribePipelineBuilder {
    model_id: String,
    device: Device,
    window_secs: f32,
    max_new_tokens: usize,
    download_policy: loader::DownloadPolicy,
}

impl TranscribePipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// The fixed decode window, in seconds - audio beyond this is dropped
    /// (loudly - see [`Transcript::truncated`]), never silently kept.
    pub fn window_secs(mut self, secs: f32) -> Self {
        self.window_secs = secs;
        self
    }

    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    /// How [`TranscribePipelineBuilder::load`] may use the network to
    /// resolve `model_id`. Defaults to [`loader::DownloadPolicy::IfMissing`]
    /// -- see that type's own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve `model_id` and build a real [`TranscribePipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `crates/loader`'s resolver is tried FIRST against
    ///    `qwen3asr::spec::Qwen3AsrSpec`'s one `"weights"` role, then
    ///    against `nemotronasr::spec::NemotronAsrSpec`'s, and only on
    ///    `Missing` everywhere does `model_id` get parsed and, under
    ///    [`TranscribePipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`]), fetched - then resolution
    ///    is retried once. See [`crate::resolve_policy::resolve_either_for_reference`]: the
    ///    model asked for, not the order the architectures are tried in,
    ///    decides which one loads.
    /// 3. The resolved architecture's own loader runs on that directory:
    ///    `qwen3asr::caps::QwenAsrProvider::load` builds the audio encoder and
    ///    Qwen3 decoder at the fixed window this builder chose (`window_secs`
    ///    and `max_new_tokens` apply to it alone);
    ///    `nemotronasr::model::NemotronAsr::from_hf` builds the transducer.
    pub fn load(self) -> Result<TranscribePipeline> {
        let TranscribePipelineBuilder { model_id, device, window_secs, max_new_tokens, download_policy } = self;

        crate::device::apply(&device)?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let resolved = crate::resolve_policy::resolve_either_for_reference(
            "qwen3asr",
            &qwen3asr::spec::Qwen3AsrSpec,
            "nemotronasr",
            &nemotronasr::spec::NemotronAsrSpec,
            &model_id,
            &overrides,
            download_policy,
        )?;
        let weights_dir = |assembly: &capability::Assembly, arch: &str| -> Result<String> {
            let dir = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("{arch}: resolved assembly {:?} has no weights role", assembly.id)))?;
            Ok(dir.to_string_lossy().into_owned())
        };

        let backend = match resolved {
            crate::resolve_policy::Resolved2::A(assembly) => {
                let dir = weights_dir(&assembly, "qwen3asr")?;
                let cfg = qwen3asr::config::QwenAsrConfig::qwen3_asr_1_7b();
                Backend::Qwen3Asr(qwen3asr::caps::QwenAsrProvider::load(&dir, cfg, window_secs, max_new_tokens).map_err(Error::Backend)?)
            }
            crate::resolve_policy::Resolved2::B(assembly) => {
                let dir = weights_dir(&assembly, "nemotronasr")?;
                let cfg = nemotronasr::NemotronConfig::nemotron_3_5_asr_0_6b();
                let model = nemotronasr::model::NemotronAsr::from_hf(&dir, cfg).map_err(|e| Error::Backend(format!("nemotronasr: {e}")))?;
                let detokenizer = nemotronasr::tokenizer::Detokenizer::from_hf(&dir).map_err(|e| Error::Backend(format!("nemotronasr: {e}")))?;
                Backend::Nemotron(Box::new(Nemotron { model, detokenizer }))
            }
        };

        Ok(TranscribePipeline { backend })
    }
}
