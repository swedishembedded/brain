// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`VisionLanguagePipeline`]: brain's vision-language surface, over
//! Qwen3-VL (1-8 images + text in, text out) - the Multimodal/VLM/OCR
//! domain's first SDK pipeline (see `crates/sdk/Cargo.toml`'s own
//! `multimodal` feature comment for why qwen3vl was picked first among this
//! bucket's served architectures).
//!
//! Resolved through `crates/loader`'s resolver against `qwen3vl::spec::
//! Qwen3VlSpec`'s one `weights` role - the same resolver `brain do qwen3vl
//! generate` uses. Reuses `qwen3vl::caps::Resident::generate` directly - the
//! SAME function the served action and the residency adapter both call, by
//! that function's own doc ("so a residency-scheduled instance and the
//! direct provider execute byte-for-byte the same code") - by building a
//! plain `capability::Invocation` the same way
//! [`crate::TextGenerationPipeline`] already does, not a second
//! implementation of chat-templating/image-preprocessing/DeepStack.
//!
//! Video input and tool-calling are real, already-implemented capabilities
//! of `Resident::generate` this pipeline does not expose - see
//! [`VisionLanguagePipeline::ask_multi_with`]'s own doc.
//!
//! ```no_run
//! let pipe = brain::VisionLanguagePipeline::from_pretrained("Qwen/Qwen3-VL-4B-Instruct")?;
//! let image = brain::Image::open("whale.png")?;
//! let out = pipe.ask(&image, "What is in this image?")?;
//! println!("{}", out.text);
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

use capability::{Blob, CancelToken, Invocation};
use serde_json::json;

use crate::{Device, Error, GeneratedText, Image, Result};

/// [`VisionLanguagePipeline::ask_with`]/`ask_multi_with`'s knobs, layered
/// over `qwen3vl::caps::Resident::generate`'s own request defaults (greedy,
/// 64 new tokens) - every field left unset here keeps whatever that
/// function already does, the same convention `TextGenerationOptions`
/// follows.
#[derive(Clone, Debug, Default)]
pub struct VlmOptions {
    max_new_tokens: Option<u32>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    seed: Option<u64>,
}

impl VlmOptions {
    pub fn new() -> VlmOptions {
        VlmOptions::default()
    }

    pub fn max_new_tokens(mut self, n: u32) -> Self {
        self.max_new_tokens = Some(n);
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Reproducible decoding. Left unset, every call gets a real random seed
    /// (`Resident::generate`'s own `data::rng::random_seed()` fallback).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn into_invocation(self, prompt: &str) -> Invocation {
        let mut inv = Invocation::new().set("prompt", json!(prompt));
        if let Some(v) = self.max_new_tokens {
            inv = inv.set("max_new", json!(v));
        }
        if let Some(v) = self.temperature {
            inv = inv.set("temp", json!(v));
        }
        if let Some(v) = self.top_k {
            inv = inv.set("top_k", json!(v));
        }
        if let Some(v) = self.top_p {
            inv = inv.set("top_p", json!(v));
        }
        if let Some(v) = self.seed {
            inv = inv.set("seed", json!(v));
        }
        inv
    }
}

/// `brain`'s vision-language pipeline: 1-8 images + text in, text out.
pub struct VisionLanguagePipeline {
    resident: qwen3vl::caps::Resident,
}

/// Hand-written, not derived: `qwen3vl::caps::Resident` carries no `Debug`
/// impl (it holds a live GPU-resident model), so a derive here would not
/// compile.
impl std::fmt::Debug for VisionLanguagePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionLanguagePipeline").finish_non_exhaustive()
    }
}

/// [`crate::Image`] (interleaved RGB8, [`crate::Image::pixels`]) normalized
/// to the raw HWC `f32` `[0,1]` blob shape `qwen3vl::caps::decode_images`
/// (via `capability::blob::decode_image`) reads - `capability::blob::
/// image_blob` is the shared encode-side counterpart of that decode
/// function, the same pairing every other image-blob producer in this
/// workspace uses.
fn image_blob(image: &Image) -> Blob {
    let hwc: Vec<f32> = image.pixels().iter().map(|&b| b as f32 / 255.0).collect();
    capability::blob::image_blob(&hwc, image.width(), image.height(), 3)
}

/// The wire name of image `i` (0-based): `image`, `image1`, `image2`, ... -
/// mirrors `qwen3vl::caps`'s own (private) `image_key`, the ONE numbered-
/// blob-key convention `decode_images` reads, cited here rather than
/// re-derived as a guess.
fn image_key(i: usize) -> String {
    if i == 0 { "image".to_string() } else { format!("image{i}") }
}

/// [`VisionLanguagePipeline::ask_multi_with`]'s image-count gate, factored
/// out so it is testable with no real pipeline in hand (the same
/// "factored out for testability" shape `crate::pipeline::check_s3dit_size`/
/// `adapter_source_path` already use): `images` must be non-empty (this
/// model has no text-only mode) and at most `qwen3vl::caps::MAX_IMAGES`.
fn check_image_count(n: usize) -> Result<()> {
    if n == 0 {
        return Err(Error::MissingArgument("qwen3vl: at least one image is required (this model has no text-only mode)".to_string()));
    }
    if n > qwen3vl::caps::MAX_IMAGES {
        return Err(Error::MissingArgument(format!("qwen3vl: {n} images given, this pipeline supports at most {}", qwen3vl::caps::MAX_IMAGES)));
    }
    Ok(())
}

impl VisionLanguagePipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<VisionLanguagePipeline> {
        VisionLanguagePipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> VisionLanguagePipelineBuilder {
        VisionLanguagePipelineBuilder {
            model_id: model_id.as_ref().to_string(),
            device: Device::default(),
            max_pixels: qwen3vl::caps::DEFAULT_SERVE_MAX_PIXELS,
            precision: qwen3vl::caps::Precision::default(),
            download_policy: loader::DownloadPolicy::default(),
        }
    }

    /// The real, static `capability::Manifest` this session's actions
    /// declare (`qwen3vl::caps::manifest`) - reflected, not re-described,
    /// same reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        qwen3vl::caps::manifest()
    }

    /// Ask about ONE image, at [`VlmOptions`]'s defaults.
    pub fn ask(&self, image: &Image, prompt: &str) -> Result<GeneratedText> {
        self.ask_with(image, prompt, VlmOptions::default())
    }

    /// [`VisionLanguagePipeline::ask`] plus [`VlmOptions`].
    pub fn ask_with(&self, image: &Image, prompt: &str, opts: VlmOptions) -> Result<GeneratedText> {
        self.ask_multi_with(std::slice::from_ref(image), prompt, opts)
    }

    /// Ask about 1-8 images together, at [`VlmOptions`]'s defaults - the SAME
    /// capability [`VisionLanguagePipeline::ask`] is a one-image special
    /// case of, not a different one (`qwen3vl::caps`'s own doc: numbered
    /// blob keys `image`, `image1`, ... contiguous from `image`).
    pub fn ask_multi(&self, images: &[Image], prompt: &str) -> Result<GeneratedText> {
        self.ask_multi_with(images, prompt, VlmOptions::default())
    }

    /// [`VisionLanguagePipeline::ask_multi`] plus [`VlmOptions`].
    ///
    /// `images` must be non-empty (this model has no text-only mode:
    /// `qwen3vl::caps`'s own `decode_media` requires exactly one of
    /// image/video, and this facade does not expose video) and at most
    /// `qwen3vl::caps::MAX_IMAGES` (8) - both checked here, before any
    /// backend call, so a caller-programming error is
    /// [`Error::MissingArgument`] rather than an indistinguishable
    /// [`Error::Backend`] from `capability::blob::decode_image`'s own
    /// missing-key message.
    ///
    /// Video input and tool-calling are real, already-implemented
    /// capabilities of `qwen3vl::caps::Resident::generate` this method does
    /// not reach - it always passes `video_frames: None`,
    /// `tool_choice: ToolChoice::Auto`, `tools: &[]`. Tracked future
    /// extensions (the same class of narrowing `TtsPipeline`'s own
    /// `lora_train`/progress gaps already document), not silently
    /// unsupported.
    pub fn ask_multi_with(&self, images: &[Image], prompt: &str, opts: VlmOptions) -> Result<GeneratedText> {
        check_image_count(images.len())?;

        let mut inv = opts.into_invocation(prompt);
        for (i, image) in images.iter().enumerate() {
            inv = inv.blob(&image_key(i), image_blob(image));
        }
        inv.cancel = CancelToken::default();

        let outcome = self.resident.generate(&inv, None, qwen3::chat::ToolChoice::Auto, &[], &mut |_| {}).map_err(Error::Backend)?;
        crate::text::generated_text_from_outcome(outcome)
    }
}

/// Builds a [`VisionLanguagePipeline`]. `.device(...)`/`.max_pixels(...)`/
/// `.precision(...)` are the only knobs this milestone exposes.
pub struct VisionLanguagePipelineBuilder {
    model_id: String,
    device: Device,
    max_pixels: u32,
    precision: qwen3vl::caps::Precision,
    download_policy: loader::DownloadPolicy,
}

impl VisionLanguagePipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Resident capacity: max input image area in pixels, applied to EACH
    /// image independently (`qwen3vl::caps::generate_spec`'s own doc) -
    /// default `qwen3vl::caps::DEFAULT_SERVE_MAX_PIXELS` (~1024x1024).
    pub fn max_pixels(mut self, max_pixels: u32) -> Self {
        self.max_pixels = max_pixels.max(1);
        self
    }

    /// Decoder storage tier: `"fp32"` (default, exact) or `"int8"` (LOSSY -
    /// see `qwen3vl::caps::Precision`'s own doc). `Err` for any other
    /// spelling, the same vocabulary `qwen3vl::caps::Precision::from_name`
    /// (and `brain do qwen3vl generate --precision`) already reject.
    pub fn precision(mut self, precision: impl AsRef<str>) -> Result<Self> {
        self.precision = qwen3vl::caps::Precision::from_name(precision.as_ref()).map_err(Error::Backend)?;
        Ok(self)
    }

    /// How [`VisionLanguagePipelineBuilder::load`] may use the network to
    /// resolve `model_id`. Defaults to [`loader::DownloadPolicy::IfMissing`]
    /// -- see that type's own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve `model_id` and build a real [`VisionLanguagePipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `model_id` is parsed. Resolution is tried FIRST, before ever
    ///    consulting `Store::local`/`plan` - the same fix
    ///    `crate::tts::TtsPipelineBuilder::load`/`crate::text::
    ///    resolve_hub_weights` both needed: a real, already-downloaded
    ///    qwen3vl checkpoint directory (`config.json` + `model.safetensors`
    ///    shards + `tokenizer.json`, the standard HF layout) is neither a
    ///    compound `brain.manifest.json` nor a bare `model.brain.safetensors`,
    ///    so `Store::local` would never recognize it and `plan()` would fall
    ///    through to `TransformersRecipe`'s catch-all - which, unlike a
    ///    plain qwen3 decoder, would misread a multimodal `config.json` and
    ///    fail or (worse) attempt the wrong conversion.
    /// 3. Only when nothing resolves: `model_id` is fetched under
    ///    [`VisionLanguagePipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`], skipped if `Store::local`
    ///    already recognizes it some other way), then resolution retries
    ///    once. See [`crate::resolve_policy::resolve_with_policy`], shared
    ///    by every pipeline builder that resolves this way.
    /// 4. `qwen3vl::caps::Resident::load_on` builds the resident model at
    ///    this builder's `max_pixels`/`precision`, letting device placement
    ///    fall to whatever step 1 already narrowed the ambient selection to
    ///    (`gpu: None`) - the same pattern every other resolver-backed
    ///    pipeline in this crate uses.
    pub fn load(self) -> Result<VisionLanguagePipeline> {
        let VisionLanguagePipelineBuilder { model_id, device, max_pixels, precision, download_policy } = self;

        crate::device::apply(&device)?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = crate::resolve_policy::resolve_with_policy("qwen3vl", &qwen3vl::spec::Qwen3VlSpec, &model_id, &overrides, download_policy)?;

        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("qwen3vl: resolved assembly {:?} has no weights role", assembly.id)))?;
        let resident = qwen3vl::caps::Resident::load_on(&weights.to_string_lossy(), max_pixels, precision, None).map_err(Error::Backend)?;

        Ok(VisionLanguagePipeline { resident })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_image_count_refuses_zero_images() {
        assert!(matches!(check_image_count(0), Err(Error::MissingArgument(_))));
    }

    #[test]
    fn check_image_count_accepts_the_full_range() {
        for n in 1..=qwen3vl::caps::MAX_IMAGES {
            assert!(check_image_count(n).is_ok(), "{n} images should be accepted");
        }
    }

    #[test]
    fn check_image_count_refuses_past_the_cap() {
        assert!(matches!(check_image_count(qwen3vl::caps::MAX_IMAGES + 1), Err(Error::MissingArgument(_))));
    }

    #[test]
    fn image_key_matches_qwen3vls_own_numbered_convention() {
        assert_eq!(image_key(0), "image");
        assert_eq!(image_key(1), "image1");
        assert_eq!(image_key(7), "image7");
    }
}
