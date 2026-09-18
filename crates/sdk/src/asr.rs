// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`TranscribePipeline`]: brain's speech-to-text surface, over qwen3-asr
//! (Qwen3-ASR-1.7B, offline, a fixed audio decode window). Resolved through
//! `crates/loader`'s resolver against `qwen3asr::spec::Qwen3AsrSpec`'s one
//! `"weights"` role.
//!
//! nemotronasr (the *streaming* ASR model in this workspace, true batched
//! forward across concurrent windows) is deliberately not covered here: a
//! streaming transcriber needs a genuinely different call shape (feed
//! chunks, get segments back incrementally) than this pipeline's one-shot
//! `transcribe(wav) -> Transcript` - tracked as a real, separate extension,
//! not forced into this type by pretending the two models work alike.
//!
//! ```no_run
//! let pipe = brain::TranscribePipeline::from_pretrained("Qwen/Qwen3-ASR-1.7B")?;
//! let out = pipe.transcribe_wav(&std::fs::read("clip.wav")?)?;
//! println!("{}", out.text);
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

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
    provider: qwen3asr::caps::QwenAsrProvider,
}

impl std::fmt::Debug for TranscribePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscribePipeline").field("window_samples", &self.provider.window_samples()).finish()
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
        let truncated = qwen3asr::caps::window_truncation(self.provider.window_samples(), samples);
        let (text, tokens) = self.provider.transcribe(samples).map_err(Error::Backend)?;
        Ok(Transcript { text, tokens, truncated })
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
    ///    `qwen3asr::spec::Qwen3AsrSpec`'s one `"weights"` role. Only on
    ///    `Missing` does `model_id` get parsed and, under
    ///    [`TranscribePipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`]), fetched - then resolution
    ///    is retried once. See [`crate::resolve_policy::resolve_with_policy`],
    ///    shared by every pipeline builder that resolves this way -
    ///    qwen3asr has a working recipe, so unlike some siblings this
    ///    reorder closes no live bug, only unifies every pipeline builder
    ///    onto one resolve strategy.
    /// 3. `qwen3asr::caps::QwenAsrProvider::load` reads the checkpoint's own
    ///    tokenizer from the SAME directory (`QwenBpe::from_dir`) and builds
    ///    the audio encoder + Qwen3 decoder at the fixed window this builder
    ///    chose.
    pub fn load(self) -> Result<TranscribePipeline> {
        let TranscribePipelineBuilder { model_id, device, window_secs, max_new_tokens, download_policy } = self;

        crate::device::apply(&device)?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = crate::resolve_policy::resolve_with_policy("qwen3asr", &qwen3asr::spec::Qwen3AsrSpec, &model_id, &overrides, download_policy)?;
        let dir = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("qwen3asr: resolved assembly {:?} has no weights role", assembly.id)))?;

        let cfg = qwen3asr::config::QwenAsrConfig::qwen3_asr_1_7b();
        let provider = qwen3asr::caps::QwenAsrProvider::load(&dir.to_string_lossy(), cfg, window_secs, max_new_tokens).map_err(Error::Backend)?;

        Ok(TranscribePipeline { provider })
    }
}
