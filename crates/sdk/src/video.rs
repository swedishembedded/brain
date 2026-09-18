// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`VideoPipeline`]: brain's text-to-video surface, over Wan2.1. Resolved
//! through `crates/loader`'s resolver against `wan::spec::WanSpec`'s four
//! roles (`dit`, `vae`, `text_encoder`, `tokenizer`) - the same resolver
//! `brain do brain/wan t2v` uses.
//!
//! Only Wan's `t2v` action is covered: `lora_train` is a training entry
//! point, the same "no training call yet" gap every other pipeline in this
//! crate has (rule 9, open, tracked per-pipeline in the roadmap) - not a
//! narrower scope than this milestone could have covered, a real gap this
//! milestone did not close either. I2V is not covered because
//! `wan::caps`'s own module doc says the underlying crate does not support
//! it yet (no 36-channel input path, no CLIP vision tower).
//!
//! Unlike [`crate::ImagePipeline`], [`VideoPipelineBuilder::load`] never asks
//! for a variant: `wan::spec::WanSpec::assemble` already derives the full
//! variant name (`t2v-1.3B`/`t2v-14B`) from the resolved `dit`'s own tensor
//! shapes - a resolved checkpoint always names one, never an ambiguity - so
//! `model_id` alone picks it, the same way it picks flux2 vs s3dit.
//!
//! [`VideoPipeline`] holds the denoise transformer resident across calls
//! (`wan::pipeline::HotDit`, behind a `Mutex` for the same `&self`-only
//! public-method reason [`crate::SegmentPipeline`]'s own `Mutex<sam2::caps::
//! Session>` does) - a cold call pays a slow load plus several GB of upload;
//! a second call at the same variant/resolution pays neither.
//!
//! ```no_run
//! let pipe = brain::VideoPipeline::from_pretrained("Wan-AI/Wan2.1-T2V-1.3B")?;
//! let clip = pipe.generate("a whale submarine")?;
//! clip.save("out.mp4")?;
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use capability::CancelToken;

use crate::{Device, Error, Result};

/// A generated clip: RGB8 frames plus the rate they were generated at.
/// The video-bucket analog of [`crate::Image`]/[`crate::Mask`]/
/// [`crate::DepthMap`]/[`crate::Audio`] - a normalized domain type over the
/// backend's own [`wan::pipeline::Video`], not a second representation a
/// caller has to convert out of.
#[derive(Clone, Debug, PartialEq)]
pub struct Video {
    width: u32,
    height: u32,
    fps: usize,
    /// Each element is one frame's `width * height * 3` interleaved RGB8
    /// bytes, in playback order.
    frames: Vec<Vec<u8>>,
}

impl Video {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn fps(&self) -> usize {
        self.fps
    }

    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Each frame's `width * height * 3` interleaved RGB8 bytes, in playback
    /// order.
    pub fn frames(&self) -> &[Vec<u8>] {
        &self.frames
    }

    /// Encode and write the clip (`imaging::video::encode_frames` - `ffmpeg`
    /// when available, else numbered PPM frames beside `path`; see that
    /// function's own doc for the no-`ffmpeg` fallback contract). The
    /// extension picks the container the same way [`crate::Image::save`]
    /// picks an image codec.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let rgb8: Vec<imaging::Rgb8> =
            self.frames.iter().map(|px| imaging::Rgb8::new(self.width, self.height, px.clone())).collect::<std::result::Result<_, String>>().map_err(Error::Backend)?;
        imaging::video::encode_frames(&rgb8, path.as_ref(), self.fps as f64, &imaging::video::VideoEncodeOpts::default()).map_err(Error::Backend)?;
        Ok(())
    }
}

/// [`VideoPipeline::generate_with`]'s knobs, mirroring `wan::caps`'s own
/// `t2v` action params. Every field defaults to `None` (the variant's own
/// upstream default, via [`wan::pipeline::GenOpts::from_config`]), the same
/// "explicit unset stays unset, resolved from the checkpoint" contract
/// [`crate::TtsOptions`] already established.
#[derive(Clone, Debug, Default)]
pub struct VideoOptions {
    negative_prompt: Option<String>,
    frames: Option<usize>,
    size: Option<(usize, usize)>,
    steps: Option<usize>,
    shift: Option<f32>,
    guidance: Option<f32>,
    seed: Option<u64>,
    fps: Option<usize>,
}

impl VideoOptions {
    pub fn new() -> VideoOptions {
        VideoOptions::default()
    }

    /// What to avoid. Omitted uses the variant's own sample negative prompt;
    /// `""` means genuinely none.
    pub fn negative_prompt(mut self, s: impl Into<String>) -> Self {
        self.negative_prompt = Some(s.into());
        self
    }

    /// Video frames; must be `1 + 4k` (the causal VAE gives the first frame
    /// its own latent frame).
    pub fn frames(mut self, n: usize) -> Self {
        self.frames = Some(n);
        self
    }

    /// Output size, px (a multiple of the VAE stride x patch size - 16 for
    /// every released variant).
    pub fn size(mut self, width: usize, height: usize) -> Self {
        self.size = Some((width, height));
        self
    }

    pub fn steps(mut self, n: usize) -> Self {
        self.steps = Some(n);
        self
    }

    /// Flow-matching sigma shift.
    pub fn shift(mut self, s: f32) -> Self {
        self.shift = Some(s);
        self
    }

    /// Classifier-free guidance. `<= 1.0` skips the unconditional forward
    /// entirely (exact, not an approximation) and halves the cost.
    pub fn guidance(mut self, g: f32) -> Self {
        self.guidance = Some(g);
        self
    }

    /// Reproducible run. Omitted draws a fresh random seed per call.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn fps(mut self, fps: usize) -> Self {
        self.fps = Some(fps);
        self
    }

    fn to_gen_opts(&self, cfg: &wan::config::WanConfig) -> wan::pipeline::GenOpts {
        let d = wan::pipeline::GenOpts::from_config(cfg);
        wan::pipeline::GenOpts {
            frames: self.frames.unwrap_or(d.frames),
            width: self.size.map(|(w, _)| w).unwrap_or(d.width),
            height: self.size.map(|(_, h)| h).unwrap_or(d.height),
            steps: self.steps.unwrap_or(d.steps),
            shift: self.shift.unwrap_or(d.shift),
            guidance: self.guidance.unwrap_or(d.guidance),
            seed: self.seed.unwrap_or_else(data::rng::random_seed),
            negative_prompt: self.negative_prompt.clone(),
            fps: self.fps.unwrap_or(d.fps),
            ..d
        }
    }
}

/// `brain`'s text-to-video pipeline. One architecture today (Wan2.1 T2V).
pub struct VideoPipeline {
    cfg: wan::config::WanConfig,
    paths: wan::pipeline::Paths,
    hot: Mutex<Option<wan::pipeline::HotDit>>,
}

/// Hand-written, not derived: the resident `HotDit` behind the `Mutex` has
/// no `Debug` impl (a live device handle), the same reason
/// [`crate::UpscalePipeline`]'s own `Debug` is hand-written.
impl std::fmt::Debug for VideoPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoPipeline").field("variant", &self.cfg.name).finish()
    }
}

impl VideoPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<VideoPipeline> {
        VideoPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> VideoPipelineBuilder {
        VideoPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), download_policy: loader::DownloadPolicy::default() }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`wan::caps::manifest`) - reflected, not re-described, same
    /// reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        wan::caps::manifest()
    }

    /// Generate a clip from `prompt`, at [`VideoOptions`]'s (the variant's
    /// own upstream) defaults.
    pub fn generate(&self, prompt: &str) -> Result<Video> {
        self.generate_with(prompt, VideoOptions::default())
    }

    /// [`VideoPipeline::generate`] plus [`VideoOptions`].
    pub fn generate_with(&self, prompt: &str, opts: VideoOptions) -> Result<Video> {
        let gen_opts = opts.to_gen_opts(&self.cfg);
        let mut hot = self.hot.lock().map_err(|_| Error::Backend("wan: hot DiT lock poisoned".to_string()))?;
        let (video, _timings) =
            wan::pipeline::generate_hot(&self.cfg, &self.paths, prompt, &gen_opts, &CancelToken::default(), &mut hot, |_, _, _| {}).map_err(Error::Backend)?;
        Ok(Video { width: video.width, height: video.height, fps: video.fps, frames: video.frames })
    }
}

/// Builds a [`VideoPipeline`]. `.device(...)` is the only knob this
/// milestone exposes.
pub struct VideoPipelineBuilder {
    model_id: String,
    device: Device,
    download_policy: loader::DownloadPolicy,
}

impl VideoPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// How [`VideoPipelineBuilder::load`] may use the network to resolve
    /// `model_id`. Defaults to [`loader::DownloadPolicy::IfMissing`] -- see
    /// that type's own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve `model_id` and build a real [`VideoPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `crates/loader`'s resolver is tried FIRST against
    ///    `wan::spec::WanSpec`'s four roles (`dit`, `vae`, `text_encoder`,
    ///    `tokenizer`), before ever consulting `Store::local`/`plan` -
    ///    deliberately the reverse of the naive "check `Store::local`, then
    ///    fetch-if-missing, then resolve" order, for the same real reason
    ///    `crate::tts::TtsPipelineBuilder::load`/`crate::text::resolve_hub_weights`
    ///    do it: `WanSpec::classify` reads raw file content (a `.pth`'s own
    ///    tensor names/shapes, a GGUF's KV metadata, a safetensors file's own
    ///    shapes), a strictly wider net than `Store::local`'s narrow "a
    ///    compound `brain.manifest.json`, or a bare `model.brain.safetensors`"
    ///    shapes - and a real Wan release's own native filenames
    ///    (`Wan2.1_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, a diffusers-
    ///    named DiT safetensors shard) satisfy NEITHER, so `Store::local`
    ///    would otherwise never recognize even an already-downloaded release.
    /// 3. Only on `Missing` does `model_id` get parsed and, under
    ///    [`VideoPipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`]), fetched (mirrors every
    ///    other pipeline in this crate) - then resolution is retried once.
    ///    See [`crate::resolve_policy::resolve_with_policy`], shared by
    ///    every pipeline builder that resolves this way.
    /// 4. The resolved variant is derived from the `dit`'s own tensor
    ///    shapes and [`wan::pipeline::Paths::from_assembly`] builds the
    ///    concrete per-file paths; the DiT itself loads lazily on the first
    ///    `.generate()` (`wan::caps`'s own module doc - "only `WanProvider`
    ///    (execution) loads anything").
    pub fn load(self) -> Result<VideoPipeline> {
        let VideoPipelineBuilder { model_id, device, download_policy } = self;

        crate::device::apply(&device)?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = crate::resolve_policy::resolve_with_policy("wan", &wan::spec::WanSpec, &model_id, &overrides, download_policy)?;

        let variant = assembly.variant.as_deref().ok_or_else(|| Error::Backend(format!("wan: resolved assembly {:?} names no variant", assembly.id)))?;
        let cfg = wan::caps::config_from_name(variant).map_err(Error::Backend)?;
        let paths = wan::pipeline::Paths::from_assembly(&assembly).map_err(Error::Backend)?;

        Ok(VideoPipeline { cfg, paths, hot: Mutex::new(None) })
    }
}
