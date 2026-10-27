// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`DetectionPipeline`]: brain's object-detection surface, over YOLOv8 and
//! YOLOv5. Resolved through `crates/loader`'s resolver against each
//! architecture's one `"weights"` role (`yolov8::spec::YoloSpec`,
//! `yolov5::spec::Yolov5Spec`) - the same resolver `brain do yolov8 detect`
//! uses, never a parallel path. The architecture is read from the checkpoint,
//! not chosen by the caller: both are the one [`DetectionPipeline`].
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
use std::sync::Mutex;

use crate::resolve_policy::Resolved2;
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
    fn from_raw(d: imaging::nms::Detection) -> Detection {
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

/// The resolved architecture's resident model.
enum Detector {
    // Boxed: each is kilobytes of resident device state, and the enum should stay one pointer wide.
    V8(Box<yolov8::Yolo>),
    V5(Box<yolov5::Yolov5>),
    /// YOLOv5 compiled onto a Rockchip NPU. A run writes the model's NPU
    /// buffers, so concurrent detections take turns.
    V5Npu(Box<Mutex<yolov5::npu::NpuDetector>>),
}

/// `brain`'s object-detection pipeline over YOLOv8 and YOLOv5 checkpoints.
pub struct DetectionPipeline {
    model: Detector,
}

/// Explicit, not derived: both models hold a live GPU device handle
/// (no `Debug` impl), the same reason [`crate::UpscalePipeline`]'s own
/// `Debug` is explicit.
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
        DetectionPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), download_policy: loader::DownloadPolicy::default(), models_dir: None }
    }

    /// The real, static `capability::Manifest` this session's action
    /// declares (`yolov8::caps::manifest` / `yolov5::caps::manifest`, by the
    /// resolved architecture) - reflected, not re-described, same reason
    /// [`crate::UpscalePipeline::capabilities`] is.
    pub fn capabilities(&self) -> capability::Manifest {
        match self.model {
            Detector::V8(_) => yolov8::caps::manifest(),
            Detector::V5(_) | Detector::V5Npu(_) => yolov5::caps::manifest(),
        }
    }

    /// The name of class `class`, for a model trained on COCO-80 (the
    /// published YOLOv8 and YOLOv5 weights are); `None` for any other class
    /// count, whose names the checkpoint does not carry.
    pub fn class_name(&self, class: u32) -> Option<&'static str> {
        let classes = match &self.model {
            Detector::V8(m) => m.cfg.nc,
            Detector::V5(m) => m.config().nc,
            Detector::V5Npu(m) => m.lock().map(|d| d.config().nc).unwrap_or(0),
        };
        (classes as usize == imaging::coco::NAMES.len()).then(|| imaging::coco::name(class as usize)).flatten()
    }

    /// Detect objects in `image` at [`DetectOptions`]'s defaults.
    pub fn detect(&self, image: &Image) -> Result<Vec<Detection>> {
        self.detect_with(image, DetectOptions::default())
    }

    /// [`DetectionPipeline::detect`] plus [`DetectOptions`].
    pub fn detect_with(&self, image: &Image, opts: DetectOptions) -> Result<Vec<Detection>> {
        let (w, h) = (image.width(), image.height());
        let raw = match &self.model {
            Detector::V8(m) => m.detect(&image.to_hwc_unit(), w, h, opts.confidence, opts.iou),
            Detector::V5(m) => m.detect(&image.to_hwc_unit(), w, h, opts.confidence, opts.iou).map_err(|e| Error::Backend(e.to_string()))?,
            Detector::V5Npu(m) => {
                let mut npu = m.lock().map_err(|_| Error::Backend("a previous NPU detection panicked".into()))?;
                npu.detect(image.pixels(), w, h, opts.confidence, opts.iou, yolov5::decode::MAX_DETECTIONS).map_err(|e| Error::Backend(e.to_string()))?.0
            }
        };
        Ok(raw.into_iter().map(Detection::from_raw).collect())
    }
}

/// Builds a [`DetectionPipeline`]. `.device(...)` is the only knob this
/// milestone exposes.
pub struct DetectionPipelineBuilder {
    model_id: String,
    device: Device,
    download_policy: loader::DownloadPolicy,
    models_dir: Option<String>,
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

    /// The models directory `model_id` (`<vendor>/<repo>`) resolves in,
    /// instead of the default (`BRAIN_MODELS_DIR`, else the user's data dir).
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// Resolve `model_id` and build a real [`DetectionPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `crates/loader`'s resolver is tried FIRST over what `model_id`'s own
    ///    repo directory holds, against `yolov8::spec::YoloSpec` and then
    ///    `yolov5::spec::Yolov5Spec`, each requiring one `"weights"`
    ///    `.safetensors` archive whose `brain.config` header derives a real
    ///    network that the file's own tensors actually match - so the
    ///    checkpoint, not the caller, decides the architecture. Only when
    ///    neither resolves does `model_id` get fetched, under
    ///    [`DetectionPipelineBuilder::download_policy`] (default
    ///    [`loader::DownloadPolicy::IfMissing`]), and resolution is retried
    ///    once. Only YOLOv8 has a model-store recipe: a YOLOv5 checkpoint
    ///    (AGPL-derived weights) is never fetched, it is supplied locally.
    /// 3. The matching crate re-reads the SAME checkpoint and builds a
    ///    single-image detector: a YOLOv5 checkpoint on a Rockchip NPU when
    ///    the device selection includes one (brain's own NPU compiler; the
    ///    checkpoint needs its `<model>.npu-calibration.json`, see
    ///    `yolov5::npu`), otherwise on the GPU/CPU backends.
    pub fn load(self) -> Result<DetectionPipeline> {
        let DetectionPipelineBuilder { model_id, device, download_policy, models_dir } = self;

        let set = crate::device::apply(&device)?;
        let rockchip_npu = !set.npus.is_empty() && gpu_core::devices::npu_kind() == Some(gpu_core::devices::NpuKind::Rockchip);

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let resolved = crate::resolve_policy::resolve_either_for_reference("yolov8", &yolov8::spec::YoloSpec, "yolov5", &yolov5::spec::Yolov5Spec, &model_id, &overrides, download_policy, models_dir.as_deref())?;
        let (arch, assembly) = match resolved {
            Resolved2::A(a) => ("yolov8", a),
            Resolved2::B(b) => ("yolov5", b),
        };
        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("{arch}: resolved assembly {:?} has no weights role", assembly.id)))?;

        let model = match arch {
            "yolov5" if rockchip_npu => Detector::V5Npu(Box::new(Mutex::new(yolov5::npu::NpuDetector::load(weights).map_err(|e| Error::Backend(e.to_string()))?))),
            "yolov5" => Detector::V5(Box::new(yolov5::Yolov5::load(weights, 0).map_err(|e| Error::Backend(e.to_string()))?)),
            _ => Detector::V8(Box::new(yolov8::Yolo::load(&weights.to_string_lossy(), 1))),
        };
        Ok(DetectionPipeline { model })
    }
}
