// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`TtsPipeline`]: brain's text-to-speech surface, over Qwen3-TTS and
//! CosyVoice. Resolved through `crates/loader`'s resolver against
//! `qwen3tts::spec::Qwen3TtsSpec`'s two roles (`weights_dir`, `ckpt`) tried
//! first, then `cosyvoice::spec::CosyVoiceSpec`'s four (`llm`, `flow`,
//! `hift`, `tokenizer`) - the same resolver `brain do tts synth`/`clone`/
//! `design` and `brain do cosyvoice synth` use.
//!
//! ONE pipeline type, not two: like [`crate::ImagePipeline`] (flux2 vs.
//! s3dit), this dispatches on the resolved architecture internally
//! (`resolve_arch`) rather than forking into sibling types - two
//! architectures performing the same capability stay one public type, never
//! a model-specific type next to a generic one. `speak`/`clone_voice`/`design` are three
//! voice-selection call shapes over Qwen3-TTS's ONE capability
//! (`qwen3tts::caps::manifest`'s own doc frames them identically - "each a
//! thin wrapper over the same pipeline"); CosyVoice's own `synth` action
//! always requires a reference clip AND its transcript
//! (`cosyvoice::caps::manifest`'s own doc: "zero-shot voice cloning: target
//! text + a reference audio clip and its transcript"), so it maps onto
//! [`TtsPipeline::clone_voice`] alone, at REDUCED fidelity - unlike
//! Qwen3-TTS's `clone_voice`, `ref_text` is not optional on this backend
//! (see that method's own doc). `speak`/`design` return
//! [`crate::Error::MissingArgument`] on a CosyVoice-resolved pipeline -
//! knowable before any backend call, since CosyVoice has no speaker-free or
//! instruct-controlled action at all, the same class of caller-programming
//! error that variant already covers (M6 in the roadmap).
//!
//! Unlike every other pipeline in this crate, [`TtsPipelineBuilder::load`]
//! builds no resident GPU state at all: neither backend holds anything warm
//! across calls (`qwen3tts::caps`'s own module doc - "the weights load per
//! call... there is nothing resident to cache"; `cosyvoice::caps`'s own doc -
//! "every real `synth` call reloads all five checkpoints fresh"), so this
//! pipeline is just a resolved, existence-checked path handle, and every
//! call pays the same load cost the CLI/D-Bus path already pays.
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
///
/// `variant`/`n_timesteps` are CosyVoice-only (ignored on a Qwen3-TTS-backed
/// pipeline, the same "the other backend's knobs are silently inert" shape
/// [`crate::ImageGenerationOptions`] already has for flux2-vs-s3dit-only
/// fields): `variant` selects `cosyvoice2`/`cosyvoice3`
/// (`cosyvoice::pipeline::Variant`, default `cosyvoice2`) and `n_timesteps`
/// is the flow decoder's Euler-step count, defaulting to the SELECTED
/// variant's own default (`cosyvoice::pipeline::GenOpts::for_variant`) when
/// unset - Qwen3-TTS has neither concept (it is autoregressive, not a
/// diffusion flow decoder). `seed` IS shared: both backends' `GenOpts` carry
/// one.
#[derive(Clone, Debug)]
pub struct TtsOptions {
    lang: String,
    max_frames: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
    variant: Option<String>,
    n_timesteps: Option<usize>,
}

impl Default for TtsOptions {
    fn default() -> TtsOptions {
        TtsOptions { lang: "english".to_string(), max_frames: 256, temperature: None, top_k: None, top_p: None, repetition_penalty: None, seed: None, variant: None, n_timesteps: None }
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

    /// CosyVoice only: `"cosyvoice2"` or `"cosyvoice3"` (default
    /// `cosyvoice2`). Ignored on a Qwen3-TTS-backed pipeline.
    pub fn variant(mut self, variant: impl Into<String>) -> Self {
        self.variant = Some(variant.into());
        self
    }

    /// CosyVoice only: Euler steps the flow decoder's CFM solver takes.
    /// Omitted (the default) uses the selected variant's own default
    /// (`cosyvoice::pipeline::Variant::default_timesteps`). Ignored on a
    /// Qwen3-TTS-backed pipeline.
    pub fn n_timesteps(mut self, n: usize) -> Self {
        self.n_timesteps = Some(n.max(1));
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

    /// `Err` only when `variant` is set to a string neither backend
    /// recognizes - the same vocabulary `cosyvoice::pipeline::Variant::parse`
    /// (and `brain do cosyvoice synth --variant`) already reject.
    fn to_cosyvoice_gen_opts(&self) -> Result<cosyvoice::pipeline::GenOpts> {
        let variant = match &self.variant {
            Some(v) => cosyvoice::pipeline::Variant::parse(v).map_err(Error::Backend)?,
            None => cosyvoice::pipeline::Variant::default(),
        };
        let mut opts = cosyvoice::pipeline::GenOpts::for_variant(variant);
        opts.seed = self.seed.unwrap_or_else(data::rng::random_seed);
        if let Some(n) = self.n_timesteps {
            opts.n_timesteps = n;
        }
        Ok(opts)
    }
}

/// Which architecture [`TtsPipelineBuilder::load`] resolved. Like
/// [`crate::ImagePipeline`]'s internal `Backend`, this exists only to carry
/// each architecture's own resolved path bundle through construction -
/// every dispatch past this point happens once, inside each
/// [`TtsPipeline`] method.
enum Backend {
    Qwen3Tts(qwen3tts::pipeline::TtsPaths),
    CosyVoice(cosyvoice::pipeline::CosyVoicePaths),
}

/// `brain`'s text-to-speech pipeline. Two architectures today (Qwen3-TTS,
/// CosyVoice) - see this module's own doc for how the three call shapes
/// split across them.
pub struct TtsPipeline {
    backend: Backend,
}

/// Hand-written, not derived: neither `qwen3tts::pipeline::TtsPaths` nor
/// `cosyvoice::pipeline::CosyVoicePaths` carries a `Debug` impl (both are
/// plain path bundles, not live device handles the way most other
/// pipelines' wrapped types are), so this reflects the one field per backend
/// a caller actually wants to see.
impl std::fmt::Debug for TtsPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backend {
            Backend::Qwen3Tts(paths) => f.debug_struct("TtsPipeline").field("arch", &"qwen3tts").field("ckpt_dir", &paths.ckpt_dir).finish(),
            Backend::CosyVoice(paths) => f.debug_struct("TtsPipeline").field("arch", &"cosyvoice").field("llm_dir", &paths.llm).finish(),
        }
    }
}

/// The `clone_voice`-on-CosyVoice error a caller gets for the one input this
/// backend cannot proceed without - factored out since both
/// [`TtsPipeline::speak_with`]/`design_with` (unconditionally, CosyVoice has
/// neither action) and [`TtsPipeline::clone_voice_with`] (only when
/// `ref_text` is `None`) reach a variant of it.
fn missing_argument(what: &str) -> Error {
    Error::MissingArgument(format!("cosyvoice: {what}"))
}

impl TtsPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<TtsPipeline> {
        TtsPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> TtsPipelineBuilder {
        TtsPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default() }
    }

    /// The real, static `capability::Manifest` the resolved backend's own
    /// actions declare (`qwen3tts::caps::manifest` or
    /// `cosyvoice::caps::manifest`) - reflected, not re-described, same
    /// reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        match &self.backend {
            Backend::Qwen3Tts(_) => qwen3tts::caps::manifest(),
            Backend::CosyVoice(_) => cosyvoice::caps::manifest(),
        }
    }

    /// Speaker-free text-to-speech, at [`TtsOptions`]'s defaults. Qwen3-TTS
    /// only - see [`TtsPipeline::speak_with`].
    pub fn speak(&self, text: &str) -> Result<Audio> {
        self.speak_with(text, TtsOptions::default())
    }

    /// [`TtsPipeline::speak`] plus [`TtsOptions`]. Qwen3-TTS only:
    /// CosyVoice's one `synth` action always requires a reference clip and
    /// its transcript (see this module's own doc), so a CosyVoice-resolved
    /// pipeline returns [`Error::MissingArgument`] here rather than
    /// attempting a call CosyVoice cannot serve - knowable before any
    /// backend call, the same class of error
    /// [`crate::CreatureBuilder::build`]'s own required-field checks use.
    pub fn speak_with(&self, text: &str, opts: TtsOptions) -> Result<Audio> {
        let Backend::Qwen3Tts(paths) = &self.backend else {
            return Err(missing_argument("has no speaker-free synth; call clone_voice with a reference clip and its transcript instead"));
        };
        let gen_opts = opts.to_gen_opts();
        let samples = qwen3tts::pipeline::synth(paths, &gen_opts, text, &opts.lang, &CancelToken::default()).map_err(Error::Backend)?;
        Ok(Audio { samples, sample_rate: SAMPLE_RATE })
    }

    /// Voice cloning from a reference wav at `reference`'s path.
    ///
    /// On a Qwen3-TTS-resolved pipeline: x-vector-only timbre matching, or
    /// in-context (ICL) cloning when `ref_text` (the reference clip's own
    /// transcript) is given. Needs a checkpoint whose `weights_dir` includes
    /// `speaker.safetensors` (every released checkpoint except
    /// CustomVoice/VoiceDesign-only ones).
    ///
    /// On a CosyVoice-resolved pipeline: `ref_text` is REQUIRED, not
    /// optional - CosyVoice has no x-vector-only mode
    /// (`cosyvoice::caps::manifest`'s own doc: "zero-shot voice cloning:
    /// target text + a reference audio clip and its transcript"). `None`
    /// here returns [`Error::MissingArgument`].
    pub fn clone_voice(&self, text: &str, reference: impl AsRef<Path>, ref_text: Option<&str>) -> Result<Audio> {
        self.clone_voice_with(text, reference, ref_text, TtsOptions::default())
    }

    /// [`TtsPipeline::clone_voice`] plus [`TtsOptions`].
    pub fn clone_voice_with(&self, text: &str, reference: impl AsRef<Path>, ref_text: Option<&str>, opts: TtsOptions) -> Result<Audio> {
        let refw = reference.as_ref().to_string_lossy();
        match &self.backend {
            Backend::Qwen3Tts(paths) => {
                let gen_opts = opts.to_gen_opts();
                let samples =
                    qwen3tts::pipeline::clone(paths, &gen_opts, text, &refw, ref_text.unwrap_or(""), &opts.lang, None, &CancelToken::default()).map_err(Error::Backend)?;
                Ok(Audio { samples, sample_rate: SAMPLE_RATE })
            }
            Backend::CosyVoice(paths) => {
                let ref_text = ref_text.ok_or_else(|| missing_argument("clone_voice requires ref_text (its zero-shot cloning has no x-vector-only mode)"))?;
                let gen_opts = opts.to_cosyvoice_gen_opts()?;
                let out = cosyvoice::pipeline::generate(paths, &gen_opts, text, &refw, ref_text).map_err(Error::Backend)?;
                Ok(Audio { samples: out.samples, sample_rate: out.sample_rate })
            }
        }
    }

    /// VoiceDesign (`instruct`, a natural-language voice/emotion/prosody
    /// description) and/or CustomVoice preset `speaker` selection. Qwen3-TTS
    /// only - see [`TtsPipeline::design_with`]. Needs a CustomVoice/
    /// VoiceDesign checkpoint - the released 0.6B Base model has no instruct
    /// control.
    pub fn design(&self, text: &str, instruct: &str, speaker: Option<&str>) -> Result<Audio> {
        self.design_with(text, instruct, speaker, TtsOptions::default())
    }

    /// [`TtsPipeline::design`] plus [`TtsOptions`]. Qwen3-TTS only:
    /// CosyVoice has no VoiceDesign/CustomVoice action at all, so a
    /// CosyVoice-resolved pipeline returns [`Error::MissingArgument`] here,
    /// the same reasoning [`TtsPipeline::speak_with`] documents.
    pub fn design_with(&self, text: &str, instruct: &str, speaker: Option<&str>, opts: TtsOptions) -> Result<Audio> {
        let Backend::Qwen3Tts(paths) = &self.backend else {
            return Err(missing_argument("has no VoiceDesign/CustomVoice action"));
        };
        let gen_opts = opts.to_gen_opts();
        let samples = qwen3tts::pipeline::design(paths, &gen_opts, text, &opts.lang, instruct, speaker, &CancelToken::default()).map_err(Error::Backend)?;
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
    /// 2. [`resolve_arch`] tries `crates/loader`'s resolver against
    ///    `qwen3tts::spec::Qwen3TtsSpec`'s two roles (`weights_dir`, `ckpt`)
    ///    first, then `cosyvoice::spec::CosyVoiceSpec`'s four (`llm`, `flow`,
    ///    `hift`, `tokenizer`) only when qwen3tts did not resolve - the same
    ///    "try each known architecture, first hit wins" shape
    ///    [`crate::ForecastPipeline`]'s own `resolve_arch` uses for
    ///    kronos-then-timesfm3. Tried BEFORE any download attempt (see the
    ///    real gap this order fixes, documented inline below) - a real,
    ///    already-local checkpoint never touches `Store::local`/`plan` at
    ///    all.
    /// 3. Only when step 2 reports the model genuinely missing everywhere:
    ///    `model_id` is fetched under `DownloadPolicy::IfMissing` (skipped
    ///    if `Store::local` already recognizes it), then step 2 retries
    ///    once.
    /// 4. Each architecture's own `*Paths::from_assembly` builds the concrete
    ///    per-file paths, and this builder checks every file each backend's
    ///    calls need (qwen3tts: `talker`/`mtp`/`codec`, `speaker` is
    ///    `clone`-only, checked lazily by `qwen3tts::pipeline::clone` itself;
    ///    cosyvoice: `llm.pt`/`flow.pt`/`hift.pt` under their resolved
    ///    directories - `s3tokenizer`/`campplus` are not yet resolver-migrated
    ///    (see `CosyVoicePaths::from_assembly`'s own doc) and stay a
    ///    call-time check the same way qwen3tts's `speaker` is) exist before
    ///    returning, so a broken checkpoint fails at `load()` rather than on
    ///    the first call.
    pub fn load(self) -> Result<TtsPipeline> {
        let TtsPipelineBuilder { model_id, device } = self;

        crate::device::apply(&device)?;

        let reference = brain_modelref::ModelRef::parse(&model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;
        let overrides: BTreeMap<String, String> = BTreeMap::new();

        // Try what's already resolvable on disk FIRST, before ever
        // consulting `Store::local`/`plan` - deliberately the REVERSE order
        // every other pipeline in this crate uses (`ImagePipeline`/
        // `ForecastPipeline`/`VideoPipeline` all check `Store::local` then
        // unconditionally fetch-if-missing before resolving). This is not a
        // stylistic choice: `Qwen3TtsSpec`/`CosyVoiceSpec::classify` read raw
        // file content, a strictly WIDER net than `Store::local`'s narrower
        // "a compound `brain.manifest.json`, or a bare
        // `model.brain.safetensors`" shapes - and CosyVoice has neither.
        // Real gap this milestone found reaching this crate's first genuine
        // end-to-end resolution of a real local cosyvoice fixture:
        // `brain_modelstore` has no `FilesRecipe` entry for cosyvoice (no
        // conversion step either, unlike qwen3tts's `brain tts import`), so
        // `Store::local` never recognizes even a real, already-downloaded
        // `FunAudioLLM/CosyVoice2-0.5B` repo (`hf download ... --local-dir
        // $BRAIN_MODELS_DIR/FunAudioLLM/CosyVoice2-0.5B`, the same layout
        // every other architecture in this workspace uses) and `plan()`
        // falls through to `TransformersRecipe`'s catch-all, which reads
        // `config.json` for an `architectures` field cosyvoice's repo does
        // not have and fails outright - never a `brain::Error::Download`
        // reachable from real content, no matter how the resolver would
        // have classified it. A cosyvoice `FilesRecipe` (teaching
        // `brain_modelstore` to actually FETCH one) is real future work,
        // out of scope here - this only fixes RESOLVING what is already
        // local, not automatic acquisition, for cosyvoice specifically.
        match resolve_arch(&overrides) {
            Ok(arch) => return build_from(arch),
            Err(Error::Missing(_)) => {}
            Err(e) => return Err(e),
        }

        let root = loader::model_dir::resolve(None).ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
        let store = brain_modelstore::Store::new(root);
        let hub = brain_modelstore::HfHub::new();

        if store.local(&reference).is_none() {
            let plan = brain_modelstore::plan(&reference, &store, &hub)?;
            loader::supply::execute_plan(&store, &hub, &plan, &model_id, &mut |_name, _got, _total| {}).map_err(Error::Download)?;
        }

        build_from(resolve_arch(&overrides)?)
    }
}

/// The second half of [`TtsPipelineBuilder::load`]: given a resolved
/// architecture, check every file its own calls need exists, then build the
/// pipeline. Factored out so [`TtsPipelineBuilder::load`] can call it from
/// its early "already resolvable" return and its fetch-then-resolve
/// fallback without duplicating either branch.
fn build_from(arch: ResolvedArch) -> Result<TtsPipeline> {
    match arch {
        ResolvedArch::Qwen3Tts(assembly) => {
            let paths = qwen3tts::pipeline::TtsPaths::from_assembly(&assembly).map_err(Error::Backend)?;
            for (role, p) in [("talker", &paths.talker), ("mtp", &paths.mtp), ("codec", &paths.codec)] {
                if !Path::new(p).exists() {
                    return Err(Error::Backend(format!("qwen3tts: resolved assembly {:?} is missing {role} at {p}", assembly.id)));
                }
            }
            Ok(TtsPipeline { backend: Backend::Qwen3Tts(paths) })
        }
        ResolvedArch::CosyVoice(assembly) => {
            let paths = cosyvoice::pipeline::CosyVoicePaths::from_assembly(&assembly).map_err(Error::Backend)?;
            for (role, dir, file) in [("llm", &paths.llm, "llm.pt"), ("flow", &paths.flow, "flow.pt"), ("hift", &paths.hift, "hift.pt")] {
                if !Path::new(dir).join(file).exists() {
                    return Err(Error::Backend(format!("cosyvoice: resolved assembly {:?} is missing {role}'s {file} under {dir}", assembly.id)));
                }
            }
            Ok(TtsPipeline { backend: Backend::CosyVoice(paths) })
        }
    }
}

/// Like [`crate::ImagePipeline`]'s own `ResolvedArch`, this exists only to
/// carry the resolved [`capability::Assembly`] through construction; every
/// method past [`TtsPipelineBuilder::load`] is uniform across both (see this
/// module's own doc).
enum ResolvedArch {
    Qwen3Tts(capability::Assembly),
    CosyVoice(capability::Assembly),
}

/// Resolve `overrides` against BOTH known TTS architectures with a real
/// resolver - qwen3tts first (it is the more complete of the two: the only
/// one with a genuinely shrinkable full-graph forward pass, see Phase 4.1 in
/// the roadmap for why it was picked first), then cosyvoice only when
/// qwen3tts did not resolve. minimaxmusic3 (this bucket's third served
/// architecture) is not tried here at all - see this module's own doc.
fn resolve_arch(overrides: &BTreeMap<String, String>) -> Result<ResolvedArch> {
    use brain_modelstore::resolve::Resolution;

    let qwen3tts_outcome = loader::resolve_structured("qwen3tts", &qwen3tts::spec::Qwen3TtsSpec, overrides).map_err(Error::Backend)?;
    if matches!(qwen3tts_outcome, Resolution::Resolved(_)) {
        let Resolution::Resolved(a) = qwen3tts_outcome else { unreachable!("just matched") };
        return Ok(ResolvedArch::Qwen3Tts(*a));
    }

    let cosyvoice_outcome = loader::resolve_structured("cosyvoice", &cosyvoice::spec::CosyVoiceSpec, overrides).map_err(Error::Backend)?;
    if matches!(cosyvoice_outcome, Resolution::Resolved(_)) {
        let Resolution::Resolved(a) = cosyvoice_outcome else { unreachable!("just matched") };
        return Ok(ResolvedArch::CosyVoice(*a));
    }

    // Both `Resolved` cases already returned above; only `Ambiguous`/
    // `Missing` combinations can reach here.
    match (qwen3tts_outcome, cosyvoice_outcome) {
        (Resolution::Ambiguous(a), _) => Err(Error::Ambiguous(a)),
        (_, Resolution::Ambiguous(a)) => Err(Error::Ambiguous(a)),
        (q, _) => match q {
            Resolution::Missing(m) => Err(Error::Missing(m)),
            Resolution::Resolved(_) => unreachable!("Resolved handled above"),
            Resolution::Ambiguous(_) => unreachable!("Ambiguous handled above"),
        },
    }
}
