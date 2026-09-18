// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`DetectionPipeline`]: brain's object-detection surface, over YOLOv8.
//! Resolved through `crates/loader`'s resolver against
//! `yolov8::spec::YoloSpec`'s one `"weights"` role - the same resolver
//! `brain do yolov8 detect` uses, never a parallel path.
//!
//! A NEW public domain type, [`Detection`], rather than joining
//! [`crate::Image`]-shaped pipelines: detection returns boxes over an image,
//! not another image, so it does not fit `image(+opts) -> Image` the way
//! [`crate::ImagePipeline`]/[`crate::UpscalePipeline`]/[`crate::RestorePipeline`]
//! all do - the vision/detection domain bucket's first pipeline is a
//! different shape, not a fourth backend of an existing one. Named for the
//! `brain_arch::Domain` it resolves under (`vision`, the same feature
//! [`crate::EmbeddingPipeline`] already joined for the same reason - CLIP's
//! own `arch!` row is `Vision` too, and there is no `Detection` domain
//! variant).
//!
//! ```no_run
//! let pipe = brain::DetectionPipeline::from_pretrained("Ultralytics/YOLOv8")?;
//! let photo = brain::Image::open("street.jpg")?;
//! for d in pipe.detect(&photo)? {
//!     println!("class {} at ({:.0},{:.0})-({:.0},{:.0}), conf {:.2}", d.class, d.x1, d.y1, d.x2, d.y2, d.confidence);
//! }
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

use crate::{Device, Error, Image, Result};

/// One detected object, in ORIGINAL image pixel coordinates (already mapped
/// back through the model's internal letterbox - see `yolov8::Yolo::detect`'s
/// own doc). `class` is an index into the checkpoint's own label set; this
/// crate carries no label-name table (the released COCO weights use the
/// standard 80-class order, but a fine-tuned checkpoint may not), so mapping
/// an index to a name is the caller's job.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    /// Class confidence, `[0, 1]`.
    pub confidence: f32,
    pub class: u32,
}

impl Detection {
    fn from_raw(d: yolov8::Detection) -> Detection {
        Detection { x1: d[0], y1: d[1], x2: d[2], y2: d[3], confidence: d[4], class: d[5] as u32 }
    }
}

/// [`DetectionPipeline::detect_with`]'s knobs, mirroring `yolov8::caps`'s own
/// `conf`/`iou` action params.
#[derive(Clone, Copy, Debug)]
pub struct DetectOptions {
    confidence: f32,
    iou: f32,
}

impl Default for DetectOptions {
    fn default() -> DetectOptions {
        DetectOptions { confidence: 0.25, iou: 0.45 }
    }
}

impl DetectOptions {
    pub fn new() -> DetectOptions {
        DetectOptions::default()
    }

    /// Minimum class confidence to keep a detection. Default `0.25`.
    pub fn confidence(mut self, v: f32) -> Self {
        self.confidence = v;
        self
    }

    /// NMS IoU threshold. Default `0.45`.
    pub fn iou(mut self, v: f32) -> Self {
        self.iou = v;
        self
    }
}

/// `brain`'s object-detection pipeline. One architecture today (YOLOv8).
pub struct DetectionPipeline {
    model: yolov8::Yolo,
}

/// Hand-written, not derived: `yolov8::Yolo` holds a live GPU device handle
/// (no `Debug` impl), the same reason [`crate::UpscalePipeline`]'s own
/// `Debug` is hand-written.
impl std::fmt::Debug for DetectionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetectionPipeline").finish()
    }
}

impl DetectionPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<DetectionPipeline> {
        DetectionPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> DetectionPipelineBuilder {
        DetectionPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), download_policy: loader::DownloadPolicy::default() }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`yolov8::caps::manifest`) - reflected, not re-described,
    /// same reason [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        yolov8::caps::manifest()
    }

    /// Detect objects in `image` at [`DetectOptions`]'s defaults.
    pub fn detect(&self, image: &Image) -> Result<Vec<Detection>> {
        self.detect_with(image, DetectOptions::default())
    }

    /// [`DetectionPipeline::detect`] plus [`DetectOptions`].
    pub fn detect_with(&self, image: &Image, opts: DetectOptions) -> Result<Vec<Detection>> {
        let (w, h) = (image.width(), image.height());
        let hwc = image.to_hwc_unit();
        let raw = self.model.detect(&hwc, w, h, opts.confidence, opts.iou);
        Ok(raw.into_iter().map(Detection::from_raw).collect())
    }
}

/// Builds a [`DetectionPipeline`]. `.device(...)` is the only knob this
/// milestone exposes.
pub struct DetectionPipelineBuilder {
    model_id: String,
    device: Device,
    download_policy: loader::DownloadPolicy,
}

impl DetectionPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// How [`DetectionPipelineBuilder::load`] may use the network to resolve
    /// `model_id`. Defaults to [`loader::DownloadPolicy::IfMissing`] -- see
    /// that type's own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve `model_id` and build a real [`DetectionPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `crates/loader`'s resolver is tried FIRST against
    ///    `yolov8::spec::YoloSpec`'s one `"weights"` role: a `.safetensors`
    ///    archive whose `brain.config` header derives a real YOLOv8 shape
    ///    that the file's own tensors actually match. Only on `Missing` does
    ///    `model_id` get parsed and, under
    ///    [`DetectionPipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`]), fetched - then resolution
    ///    is retried once. See [`crate::resolve_policy::resolve_with_policy`],
    ///    shared by every pipeline builder that resolves this way -
    ///    YOLOv8 has a working `brain_modelstore::recipe::YoloRecipe`, so
    ///    unlike some siblings this reorder closes no live bug, only unifies
    ///    every pipeline builder onto one resolve strategy.
    /// 3. `yolov8::Yolo::load` re-reads the SAME checkpoint and builds a
    ///    single-image (`batch = 1`) detector.
    pub fn load(self) -> Result<DetectionPipeline> {
        let DetectionPipelineBuilder { model_id, device, download_policy } = self;

        crate::device::apply(&device)?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = crate::resolve_policy::resolve_with_policy("yolov8", &yolov8::spec::YoloSpec, &model_id, &overrides, download_policy)?;
        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("yolov8: resolved assembly {:?} has no weights role", assembly.id)))?;

        let model = yolov8::Yolo::load(&weights.to_string_lossy(), 1);
        Ok(DetectionPipeline { model })
    }
}
