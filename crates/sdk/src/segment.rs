// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`SegmentPipeline`]: brain's promptable-segmentation surface, over SAM
//! 2.1. Resolved through `crates/loader`'s resolver against
//! `sam2::spec::Sam2Spec`'s one `"weights"` role - the same resolver
//! `brain do sam2 segment`/`brain sam2 track` use, never a parallel path.
//!
//! A NEW public domain type, [`Mask`], joining [`crate::Detection`] as the
//! vision/detection bucket's second non-`Image` result shape: a
//! segmentation is a per-pixel probability grid, not a box list and not
//! another image, so it does not fit either `image(+opts) -> Image` or
//! `image(+opts) -> Vec<Detection>`.
//!
//! # Prompts, not a zero-argument call
//!
//! Unlike every other pipeline in this crate, `.segment()` takes a
//! [`Prompt`] - SAM 2.1 is promptable-only, with no "segment everything"
//! mode, so there is no meaningful zero-argument default the way
//! `UpscalePipeline::upscale`'s options are all optional.
//!
//! # Encode once, prompt many
//!
//! The trunk encoder is ~99% of the cost and depends only on the image; the
//! mask decoder is tiny and depends only on the prompt
//! (`sam2::caps::Session`'s own module doc). Calling `.segment()` several
//! times with the SAME [`Image`] on the SAME [`SegmentPipeline`] reuses the
//! cached encoding automatically - the pipeline holds one `Session`, not a
//! fresh one per call - so multiple prompts on one photo cost one trunk pass
//! total, with no separate "encode" call a caller has to remember to make.
//!
//! ```no_run
//! let pipe = brain::SegmentPipeline::from_pretrained("facebook/sam2.1-hiera-tiny")?;
//! let photo = brain::Image::open("photo.png")?;
//! let mask = pipe.segment(&photo, &brain::Prompt::new().point(320.0, 240.0, true))?;
//! println!("confidence {:.2}, {} px", mask.confidence, mask.area);
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::{Device, Error, Image, Result};

/// A promptable-segmentation query, in SOURCE-image pixel coordinates - the
/// same coordinate space [`Image::width`]/[`Image::height`] report, never
/// the model's internal square frame (`sam2::caps::Session` rescales
/// internally). Mirrors `sam2::caps::parse_prompt`'s own box-then-points
/// ordering, but as a typed builder rather than `"x1,y1,x2,y2"` strings.
#[derive(Clone, Debug, Default)]
pub struct Prompt {
    bbox: Option<(f32, f32, f32, f32)>,
    points: Vec<(f32, f32, bool)>,
}

impl Prompt {
    pub fn new() -> Prompt {
        Prompt::default()
    }

    /// A box prompt, `(x1, y1, x2, y2)`. At most one - a second call
    /// replaces the first, matching the wire schema's single `box` param.
    pub fn bbox(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        self.bbox = Some((x1, y1, x2, y2));
        self
    }

    /// A click point. `foreground = true` marks the object; `false` marks
    /// what to exclude.
    pub fn point(mut self, x: f32, y: f32, foreground: bool) -> Self {
        self.points.push((x, y, foreground));
        self
    }

    fn is_empty(&self) -> bool {
        self.bbox.is_none() && self.points.is_empty()
    }

    /// `(coords, labels)` in `sam2::caps::run_prompt`'s own order: the box
    /// (labels `2.0`/`3.0`) before every click point (`1.0` foreground,
    /// `0.0` background) - the reference concatenates a box before its
    /// click points, and the prompt tokens are positional.
    fn to_coords_labels(&self) -> (Vec<(f32, f32)>, Vec<f32>) {
        let mut coords = Vec::with_capacity(self.points.len() + 2);
        let mut labels = Vec::with_capacity(self.points.len() + 2);
        if let Some((x1, y1, x2, y2)) = self.bbox {
            coords.push((x1, y1));
            coords.push((x2, y2));
            labels.push(2.0);
            labels.push(3.0);
        }
        for &(x, y, fg) in &self.points {
            coords.push((x, y));
            labels.push(if fg { 1.0 } else { 0.0 });
        }
        (coords, labels)
    }
}

/// One segmentation result: the winning mask's per-pixel probability at
/// SOURCE-image resolution, plus SAM 2.1's own quality estimate for it.
/// `probabilities[y * width + x] > 0.5` is the hard mask
/// (`sam2::caps::Session`'s own doc: `prob > 0.5` is exactly `logit > 0`).
#[derive(Clone, Debug)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub probabilities: Vec<f32>,
    /// SAM 2.1's own IoU estimate for this mask, `[0, 1]`.
    pub confidence: f32,
    /// Pixel count where `probabilities[i] > 0.5`.
    pub area: usize,
}

/// [`SegmentPipeline::segment_with`]'s one knob, mirroring `sam2::caps::
/// segment_spec`'s own `multimask` param.
#[derive(Clone, Copy, Debug)]
pub struct SegmentOptions {
    multimask: bool,
}

impl Default for SegmentOptions {
    fn default() -> SegmentOptions {
        SegmentOptions { multimask: true }
    }
}

impl SegmentOptions {
    pub fn new() -> SegmentOptions {
        SegmentOptions::default()
    }

    /// Score the 3-way ambiguity head and keep the highest-IoU mask
    /// (default `true`) instead of the single-mask head.
    pub fn multimask(mut self, v: bool) -> Self {
        self.multimask = v;
        self
    }
}

/// `brain`'s promptable-segmentation pipeline. One architecture today (SAM
/// 2.1).
pub struct SegmentPipeline {
    // `Mutex`, not `&mut self` methods: every other pipeline in this crate
    // takes `&self`, and `sam2::caps::Session`'s own encode-cache is exactly
    // the kind of interior mutability `rrdbnet::caps::Session` already wraps
    // in a `Mutex` for the same reason (a rebuild-on-change cache behind a
    // shared reference).
    session: Mutex<sam2::caps::Session>,
}

impl std::fmt::Debug for SegmentPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentPipeline").finish()
    }
}

impl SegmentPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<SegmentPipeline> {
        SegmentPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> SegmentPipelineBuilder {
        SegmentPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), download_policy: loader::DownloadPolicy::default() }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`sam2::caps::manifest`) - reflected, not re-described, same
    /// reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        sam2::caps::manifest()
    }

    /// Segment `image` against `prompt` at [`SegmentOptions`]'s defaults.
    pub fn segment(&self, image: &Image, prompt: &Prompt) -> Result<Mask> {
        self.segment_with(image, prompt, SegmentOptions::default())
    }

    /// [`SegmentPipeline::segment`] plus [`SegmentOptions`]. Calling this
    /// again with the SAME `image` (by content) skips the trunk encoder -
    /// see this module's own doc.
    pub fn segment_with(&self, image: &Image, prompt: &Prompt, opts: SegmentOptions) -> Result<Mask> {
        if prompt.is_empty() {
            return Err(Error::MissingArgument("Prompt needs at least one point or a bbox - segment(image, &Prompt::new().point(x, y, true))".to_string()));
        }
        let (w, h) = (image.width(), image.height());
        let hwc = image.to_hwc_unit();
        let (coords, labels) = prompt.to_coords_labels();

        let mut session = self.session.lock().map_err(|_| Error::Backend("sam2: session lock poisoned".to_string()))?;
        let out = session.segment_typed(&hwc, w, h, &coords, &labels, opts.multimask).map_err(Error::Backend)?;
        Ok(Mask { width: out.width, height: out.height, probabilities: out.mask, confidence: out.iou, area: out.area })
    }
}

/// Builds a [`SegmentPipeline`]. `.device(...)` is the only knob this
/// milestone exposes.
pub struct SegmentPipelineBuilder {
    model_id: String,
    device: Device,
    download_policy: loader::DownloadPolicy,
}

impl SegmentPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// How [`SegmentPipelineBuilder::load`] may use the network to resolve
    /// `model_id`. Defaults to [`loader::DownloadPolicy::IfMissing`] -- see
    /// that type's own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve `model_id` and build a real [`SegmentPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `crates/loader`'s resolver is tried FIRST against
    ///    `sam2::spec::Sam2Spec`'s one `"weights"` role, and derives
    ///    `tiny`/`large` from the checkpoint's OWN trunk width (`sam2::spec`'s
    ///    own doc) rather than trusting a separate claim. Only on `Missing`
    ///    does `model_id` get parsed and, under
    ///    [`SegmentPipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`]), fetched - then resolution
    ///    is retried once. See [`crate::resolve_policy::resolve_with_policy`],
    ///    shared by every pipeline builder that resolves this way - SAM 2.1
    ///    has a working recipe, so unlike some siblings this reorder closes
    ///    no live bug, only unifies every pipeline builder onto one resolve
    ///    strategy.
    /// 3. `sam2::caps::load` re-reads the SAME checkpoint at the resolved
    ///    variant's config and builds the model on a fresh [`gpu_core::Gpu`].
    pub fn load(self) -> Result<SegmentPipeline> {
        let SegmentPipelineBuilder { model_id, device, download_policy } = self;

        crate::device::apply(&device)?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = crate::resolve_policy::resolve_with_policy("sam2", &sam2::spec::Sam2Spec, &model_id, &overrides, download_policy)?;
        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("sam2: resolved assembly {:?} has no weights role", assembly.id)))?;
        let variant = assembly.variant.as_deref().ok_or_else(|| Error::Backend(format!("sam2: resolved assembly {:?} has no variant", assembly.id)))?;

        let cfg = sam2::caps::variant_config(variant).map_err(Error::Backend)?;
        let gpu = gpu_core::Gpu::new(sam2::PIPELINES);
        let model = sam2::caps::load(&weights.to_string_lossy(), cfg, gpu).map_err(Error::Backend)?;

        Ok(SegmentPipeline { session: Mutex::new(sam2::caps::Session::new(model)) })
    }
}
