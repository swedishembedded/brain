// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`MusicPipeline`]: brain's lyrics+caption-to-song surface, over MiniMax
//! Music 3. Resolved through `crates/loader`'s resolver against
//! `minimaxmusic3::spec::MinimaxMusic3Spec`'s six roles (`language_model`,
//! `depth_decoder`, `condition_encoder`, `transformer`, `vocoder`,
//! `tokenizer`) - the same resolver `brain do brain/minimaxmusic3 generate`
//! uses.
//!
//! A different call shape than [`crate::TtsPipeline`] on purpose: MiniMax
//! Music 3 takes STRUCTURED lyrics (`[verse]`/`[chorus]`/... tags) plus a
//! free-text music-description caption, not a speaker/voice to render text
//! in, and produces up to five minutes of 44.1 kHz STEREO audio - genuinely
//! not the same domain object [`crate::Audio`] (mono, TTS-shaped) already
//! represents, so this module introduces [`Song`] rather than stretching
//! `Audio` to carry a second, TTS-unrelated meaning for its own `channels`.
//!
//! ```no_run
//! let pipe = brain::MusicPipeline::from_pretrained("MiniMaxAI/MiniMax-Music3")?;
//! let song = pipe.generate("[verse]\nSailing under a brain-lit sky", "warm acoustic folk, 90 BPM, female vocals")?;
//! song.save("out.wav")?;
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use crate::{Device, Error, Result};

/// A finished song: separate left/right 44.1 kHz channels. The music-bucket
/// analog of [`crate::Image`]/[`crate::Video`]/[`crate::Audio`] - a
/// normalized domain type over the backend's own
/// `minimaxmusic3::generate::GeneratedSong`, not a second representation a
/// caller has to convert out of. Genuinely stereo (unlike [`crate::Audio`]),
/// since that is what the model actually produces - see this module's own
/// doc for why that is a different type rather than a `channels` field
/// bolted onto `Audio`.
#[derive(Clone, Debug, PartialEq)]
pub struct Song {
    left: Vec<f32>,
    right: Vec<f32>,
    sample_rate: u32,
}

impl Song {
    pub fn left(&self) -> &[f32] {
        &self.left
    }

    pub fn right(&self) -> &[f32] {
        &self.right
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn seconds(&self) -> f64 {
        self.left.len() as f64 / self.sample_rate as f64
    }

    /// Write a stereo 16-bit PCM WAV file (`audio::wav::write_multi`, the
    /// same multi-channel codec [`crate::Audio::save`]'s mono
    /// `audio::wav::write` is the one-channel special case of).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        audio::wav::write_multi(path, &[&self.left, &self.right], self.sample_rate).map_err(|e| Error::Backend(e.to_string()))
    }
}

/// [`MusicPipeline::generate_with`]'s knobs, mirroring `minimaxmusic3::caps`'s
/// own `generate` action params. Every field defaults to `None`
/// (`minimaxmusic3::generate::GenOpts::default`'s own values), the same
/// "explicit unset stays unset" contract [`crate::TtsOptions`]/
/// [`crate::VideoOptions`] already established.
#[derive(Clone, Debug, Default)]
pub struct MusicOptions {
    duration_seconds: Option<f32>,
    num_inference_steps: Option<usize>,
    seed: Option<u64>,
}

impl MusicOptions {
    pub fn new() -> MusicOptions {
        MusicOptions::default()
    }

    /// Target song length; the AR stage may stop earlier (its own
    /// `AUDIO_END_TOKEN_ID`, not a fixed length).
    pub fn duration_seconds(mut self, seconds: f32) -> Self {
        self.duration_seconds = Some(seconds);
        self
    }

    /// Euler steps per flow-matching denoise chunk.
    pub fn num_inference_steps(mut self, n: usize) -> Self {
        self.num_inference_steps = Some(n);
        self
    }

    /// Reproducible run. Omitted draws a fresh random seed per call.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn to_gen_opts(&self) -> minimaxmusic3::generate::GenOpts {
        let d = minimaxmusic3::generate::GenOpts::default();
        minimaxmusic3::generate::GenOpts {
            duration_seconds: self.duration_seconds.unwrap_or(d.duration_seconds),
            num_inference_steps: self.num_inference_steps.unwrap_or(d.num_inference_steps),
            seed: self.seed.unwrap_or(d.seed),
            ..d
        }
    }
}

/// `brain`'s lyrics+caption-to-song pipeline. One architecture today
/// (MiniMax Music 3).
pub struct MusicPipeline {
    paths: minimaxmusic3::generate::Paths,
}

/// Hand-written, not derived: `minimaxmusic3::generate::Paths` carries no
/// GPU/device handle itself (every `generate()` call builds and drops its
/// own, stage by stage - see `minimaxmusic3::caps`'s own module doc), but is
/// still just plain file paths with no `Debug` impl of its own convenient to
/// derive through; a short summary is worth having for the same reason
/// [`crate::ImagePipeline`]'s own hand-written `Debug` is.
impl std::fmt::Debug for MusicPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MusicPipeline").field("lm", &self.paths.lm).finish()
    }
}

impl MusicPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<MusicPipeline> {
        MusicPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> MusicPipelineBuilder {
        MusicPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), download_policy: loader::DownloadPolicy::default() }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`minimaxmusic3::caps::manifest`) - reflected, not
    /// re-described, same reason [`crate::VideoPipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        minimaxmusic3::caps::manifest()
    }

    /// Generate a song from `lyrics` (structural tags like `[verse]`/
    /// `[chorus]` encouraged) and `caption` (a free-text music description:
    /// genre, BPM, vocal timbre, instrumentation, arrangement), at
    /// [`MusicOptions`]'s defaults.
    pub fn generate(&self, lyrics: &str, caption: &str) -> Result<Song> {
        self.generate_with(lyrics, caption, MusicOptions::default())
    }

    /// [`MusicPipeline::generate`] plus [`MusicOptions`].
    pub fn generate_with(&self, lyrics: &str, caption: &str, opts: MusicOptions) -> Result<Song> {
        let gen_opts = opts.to_gen_opts();
        let song = minimaxmusic3::generate::generate(&self.paths, &gen_opts, lyrics, caption, &mut |_, _, _| {}).map_err(Error::Backend)?;
        Ok(Song { left: song.left, right: song.right, sample_rate: song.sample_rate })
    }
}

/// Builds a [`MusicPipeline`]. `.device(...)` is the only knob this
/// milestone exposes.
pub struct MusicPipelineBuilder {
    model_id: String,
    device: Device,
    download_policy: loader::DownloadPolicy,
}

impl MusicPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// How [`MusicPipelineBuilder::load`] may use the network to resolve
    /// `model_id`. Defaults to [`loader::DownloadPolicy::IfMissing`] -- see
    /// that type's own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve `model_id` and build a real [`MusicPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `crates/loader`'s resolver is tried FIRST against
    ///    `minimaxmusic3::spec::MinimaxMusic3Spec`'s six roles, before ever
    ///    consulting `Store::local`/`plan` - the same reordering
    ///    `crate::tts::TtsPipelineBuilder::load`/`crate::video::VideoPipelineBuilder::load`
    ///    already apply, for the same real reason:
    ///    `MinimaxMusic3Spec::classify` reads raw file content (an HF
    ///    `language_model/config.json`, four brain-native components' own
    ///    safetensors tensor names), a strictly wider net than
    ///    `Store::local`'s narrow "a compound `brain.manifest.json`, or a
    ///    bare `model.brain.safetensors`" shapes - and a real MiniMax Music 3
    ///    release's own multi-directory layout satisfies neither.
    /// 3. Only on `Missing` does `model_id` get parsed and, under
    ///    [`MusicPipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`]), fetched - then resolution
    ///    is retried once. See [`crate::resolve_policy::resolve_with_policy`],
    ///    shared by every pipeline builder that resolves this way.
    /// 4. [`minimaxmusic3::generate::Paths::from_assembly`] builds the
    ///    concrete per-role directories; nothing loads until the first
    ///    `.generate()` (`minimaxmusic3::caps`'s own module doc - this
    ///    provider is stateless, four of the five components' host weights
    ///    warm in `minimaxmusic3::weightcache` on first use instead).
    pub fn load(self) -> Result<MusicPipeline> {
        let MusicPipelineBuilder { model_id, device, download_policy } = self;

        crate::device::apply(&device)?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = crate::resolve_policy::resolve_with_policy("minimaxmusic3", &minimaxmusic3::spec::MinimaxMusic3Spec, &model_id, &overrides, download_policy)?;

        let paths = minimaxmusic3::generate::Paths::from_assembly(&assembly).map_err(Error::Backend)?;

        Ok(MusicPipeline { paths })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`MusicOptions`] left entirely unset keeps `GenOpts::default`'s own
    /// values - this facade layers on top of the crate's own defaults, it
    /// does not invent its own.
    #[test]
    fn unset_music_options_keep_gen_opts_own_defaults() {
        let o = MusicOptions::new().to_gen_opts();
        let want = minimaxmusic3::generate::GenOpts::default();
        assert_eq!(o.duration_seconds, want.duration_seconds);
        assert_eq!(o.num_inference_steps, want.num_inference_steps);
        assert_eq!(o.seed, want.seed);
    }

    /// Every field that IS set overrides its `GenOpts` counterpart, and
    /// nothing else moves.
    #[test]
    fn set_music_options_override_only_their_own_field() {
        let o = MusicOptions::new().duration_seconds(30.0).num_inference_steps(12).seed(7).to_gen_opts();
        assert_eq!(o.duration_seconds, 30.0);
        assert_eq!(o.num_inference_steps, 12);
        assert_eq!(o.seed, 7);
    }
}
