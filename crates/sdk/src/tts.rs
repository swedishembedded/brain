// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`TtsPipeline`]: brain's text-to-speech surface, over Qwen3-TTS. Resolved
//! through `crates/loader`'s resolver against `qwen3tts::spec::
//! Qwen3TtsSpec`'s two roles (`weights_dir`, `ckpt`) - the same resolver
//! `brain do tts synth`/`clone`/`design` uses.
//!
//! ONE pipeline type, not three: `speak`/`clone_voice`/`design` are three
//! voice-selection call shapes over the SAME capability
//! (`qwen3tts::caps::manifest`'s own doc frames them identically - "each a
//! thin wrapper over the same pipeline"), not three different capabilities
//! the way restoration and upscaling are - so this does not fork into
//! sibling pipeline types the way [`crate::RestorePipeline`]/
//! [`crate::UpscalePipeline`] did for CodeFormer/RRDBNet.
//!
//! Unlike every other pipeline in this crate, [`TtsPipelineBuilder::load`]
//! builds no resident GPU state at all: Qwen3-TTS's own design is stateless
//! per call (`qwen3tts::caps`'s own module doc - "the weights load per
//! call... there is nothing resident to cache"), so this pipeline is just a
//! resolved, existence-checked [`qwen3tts::pipeline::TtsPaths`] handle, and
//! every call pays the load cost `qwen3tts::pipeline::{synth,clone,design}`
//! already pay on the CLI/D-Bus path.
//!
//! ```no_run
//! let pipe = brain::TtsPipeline::from_pretrained("Qwen/Qwen3-TTS-0.6B-Base")?;
//! let clip = pipe.speak("Hello from brain.")?;
//! clip.save("hello.wav")?;
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use capability::CancelToken;

use crate::{Device, Error, Result};

/// Qwen3-TTS's fixed codec output rate - the same `SAMPLE_RATE` constant
/// every existing caller (`qwen3tts::caps::audio_outcome`,
/// `crates/cli/src/tts_cli.rs::write_wav`) hardcodes, since
/// `qwen3tts::pipeline::{synth,clone,design}` return raw samples with no
/// rate attached.
const SAMPLE_RATE: u32 = 24_000;

/// A synthesized clip: interleaved-nothing (mono) `f32` PCM samples plus the
/// rate they were generated at. The audio-bucket analog of [`crate::Image`]/
/// [`crate::Mask`]/[`crate::DepthMap`] - a normalized domain type over a
/// backend's raw output, not a second representation a caller has to
/// convert out of.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl Audio {
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// Write a mono 16-bit PCM WAV file (`audio::wav::write`), the same
    /// codec every CLI/D-Bus caller in this workspace writes.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        audio::wav::write(path, &self.samples, self.sample_rate).map_err(|e| Error::Backend(e.to_string()))
    }
}

/// [`TtsPipeline::speak_with`]/`clone_voice_with`/`design_with`'s shared
/// knobs, mirroring `qwen3tts::caps`'s own `checkpoint_and_sampling_params`.
///
/// `temperature`/`top_k`/`top_p`/`repetition_penalty`/`seed` default to
/// `None` (genuinely unset, not merely "the reference default") on purpose -
/// [`qwen3tts::pipeline::GenOpts::default`]'s own doc explains why: an unset
/// knob resolves from the checkpoint's `generation_config.json`, then the
/// reference recipe, and a `TtsOptions::default()` that baked in a concrete
/// value here would silently shadow that chain the exact way a stale
/// `repetition_penalty = 1.0` once did on the wire path. `lang`/`max_frames`
/// are NOT part of that chain (`qwen3tts::caps::checkpoint_and_sampling_params`
/// gives both a concrete `ActionSpec` default too), so they default to
/// concrete values here as well.
#[derive(Clone, Debug)]
pub struct TtsOptions {
    lang: String,
    max_frames: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
}

impl Default for TtsOptions {
    fn default() -> TtsOptions {
        TtsOptions { lang: "english".to_string(), max_frames: 256, temperature: None, top_k: None, top_p: None, repetition_penalty: None, seed: None }
    }
}

impl TtsOptions {
    pub fn new() -> TtsOptions {
        TtsOptions::default()
    }

    pub fn lang(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }

    /// Hard cap on generated codec frames (length cap). Default `256`.
    pub fn max_frames(mut self, n: usize) -> Self {
        self.max_frames = n.max(1);
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn repetition_penalty(mut self, r: f32) -> Self {
        self.repetition_penalty = Some(r);
        self
    }

    /// Reproducible run. Omitted (the default) draws a fresh random seed per
    /// call, the same `data::rng::random_seed()` fallback
    /// `qwen3tts::caps::gen_opts_from` uses.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn to_gen_opts(&self) -> qwen3tts::pipeline::GenOpts {
        let mut opts = qwen3tts::pipeline::GenOpts { max_frames: self.max_frames, ..qwen3tts::pipeline::GenOpts::default() };
        opts.sampling.temperature = self.temperature;
        opts.sampling.top_k = self.top_k;
        opts.sampling.top_p = self.top_p;
        opts.sampling.repetition_penalty = self.repetition_penalty;
        opts.seed = self.seed.unwrap_or_else(data::rng::random_seed);
        opts
    }
}

/// `brain`'s text-to-speech pipeline. One architecture today (Qwen3-TTS).
pub struct TtsPipeline {
    paths: qwen3tts::pipeline::TtsPaths,
}

/// Hand-written, not derived: `qwen3tts::pipeline::TtsPaths` itself carries
/// no `Debug` impl (it is a plain path bundle, not a live device handle the
/// way most other pipelines' wrapped types are), so this reflects the one
/// field a caller actually wants to see.
impl std::fmt::Debug for TtsPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsPipeline").field("ckpt_dir", &self.paths.ckpt_dir).finish()
    }
}

impl TtsPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<TtsPipeline> {
        TtsPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> TtsPipelineBuilder {
        TtsPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default() }
    }

    /// The real, static `capability::Manifest` this session's actions
    /// declare (`qwen3tts::caps::manifest`) - reflected, not re-described,
    /// same reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        qwen3tts::caps::manifest()
    }

    /// Speaker-free text-to-speech, at [`TtsOptions`]'s defaults.
    pub fn speak(&self, text: &str) -> Result<Audio> {
        self.speak_with(text, TtsOptions::default())
    }

    /// [`TtsPipeline::speak`] plus [`TtsOptions`].
    pub fn speak_with(&self, text: &str, opts: TtsOptions) -> Result<Audio> {
        let gen_opts = opts.to_gen_opts();
        let samples = qwen3tts::pipeline::synth(&self.paths, &gen_opts, text, &opts.lang, &CancelToken::default()).map_err(Error::Backend)?;
        Ok(Audio { samples, sample_rate: SAMPLE_RATE })
    }

    /// Voice cloning from a reference wav at `reference`'s path -
    /// x-vector-only timbre matching, or in-context (ICL) cloning when
    /// `ref_text` (the reference clip's own transcript) is given. Needs a
    /// checkpoint whose `weights_dir` includes `speaker.safetensors` (every
    /// released checkpoint except CustomVoice/VoiceDesign-only ones).
    pub fn clone_voice(&self, text: &str, reference: impl AsRef<Path>, ref_text: Option<&str>) -> Result<Audio> {
        self.clone_voice_with(text, reference, ref_text, TtsOptions::default())
    }

    /// [`TtsPipeline::clone_voice`] plus [`TtsOptions`].
    pub fn clone_voice_with(&self, text: &str, reference: impl AsRef<Path>, ref_text: Option<&str>, opts: TtsOptions) -> Result<Audio> {
        let gen_opts = opts.to_gen_opts();
        let refw = reference.as_ref().to_string_lossy();
        let samples =
            qwen3tts::pipeline::clone(&self.paths, &gen_opts, text, &refw, ref_text.unwrap_or(""), &opts.lang, None, &CancelToken::default()).map_err(Error::Backend)?;
        Ok(Audio { samples, sample_rate: SAMPLE_RATE })
    }

    /// VoiceDesign (`instruct`, a natural-language voice/emotion/prosody
    /// description) and/or CustomVoice preset `speaker` selection. Needs a
    /// CustomVoice/VoiceDesign checkpoint - the released 0.6B Base model has
    /// no instruct control.
    pub fn design(&self, text: &str, instruct: &str, speaker: Option<&str>) -> Result<Audio> {
        self.design_with(text, instruct, speaker, TtsOptions::default())
    }

    /// [`TtsPipeline::design`] plus [`TtsOptions`].
    pub fn design_with(&self, text: &str, instruct: &str, speaker: Option<&str>, opts: TtsOptions) -> Result<Audio> {
        let gen_opts = opts.to_gen_opts();
        let samples = qwen3tts::pipeline::design(&self.paths, &gen_opts, text, &opts.lang, instruct, speaker, &CancelToken::default()).map_err(Error::Backend)?;
        Ok(Audio { samples, sample_rate: SAMPLE_RATE })
    }
}

/// Builds a [`TtsPipeline`]. `.device(...)` is the only knob this milestone
/// exposes.
pub struct TtsPipelineBuilder {
    model_id: String,
    device: Device,
}

impl TtsPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Resolve `model_id` and build a real [`TtsPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`])
    ///    - every call still builds its own `Gpu` lazily (see this module's
    ///    doc), but that construction reads the SAME ambient placement.
    /// 2. `model_id` is parsed and, under `DownloadPolicy::IfMissing`,
    ///    fetched only when nothing local already resolves it (mirrors every
    ///    other pipeline in this crate).
    /// 3. `crates/loader`'s resolver looks for `qwen3tts::spec::
    ///    Qwen3TtsSpec`'s two roles: `weights_dir` (talker/mtp/codec/speaker,
    ///    from `brain tts import`) and `ckpt` (the HF checkpoint dir for
    ///    `config.json`/the tokenizer).
    /// 4. [`qwen3tts::pipeline::TtsPaths::from_assembly`] builds the concrete
    ///    per-file paths, and this builder checks the three files every call
    ///    needs (`talker`/`mtp`/`codec` - `speaker` is `clone`-only, checked
    ///    lazily by [`qwen3tts::pipeline::clone`] itself) exist before
    ///    returning, so a broken checkpoint fails at `load()` rather than on
    ///    the first `.speak()`.
    pub fn load(self) -> Result<TtsPipeline> {
        let TtsPipelineBuilder { model_id, device } = self;

        crate::device::apply(&device)?;

        let reference = brain_modelref::ModelRef::parse(&model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;

        let root = loader::model_dir::resolve(None).ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
        let store = brain_modelstore::Store::new(root);
        let hub = brain_modelstore::HfHub::new();

        if store.local(&reference).is_none() {
            let plan = brain_modelstore::plan(&reference, &store, &hub)?;
            loader::supply::execute_plan(&store, &hub, &plan, &model_id, &mut |_name, _got, _total| {}).map_err(Error::Download)?;
        }

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let outcome = loader::resolve_structured("qwen3tts", &qwen3tts::spec::Qwen3TtsSpec, &overrides).map_err(Error::Backend)?;
        let assembly = match outcome {
            brain_modelstore::resolve::Resolution::Resolved(a) => *a,
            brain_modelstore::resolve::Resolution::Ambiguous(a) => return Err(Error::Ambiguous(a)),
            brain_modelstore::resolve::Resolution::Missing(m) => return Err(Error::Missing(m)),
        };

        let paths = qwen3tts::pipeline::TtsPaths::from_assembly(&assembly).map_err(Error::Backend)?;
        for (role, p) in [("talker", &paths.talker), ("mtp", &paths.mtp), ("codec", &paths.codec)] {
            if !Path::new(p).exists() {
                return Err(Error::Backend(format!("qwen3tts: resolved assembly {:?} is missing {role} at {p}", assembly.id)));
            }
        }

        Ok(TtsPipeline { paths })
    }
}
