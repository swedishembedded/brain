// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapters that put brain's real models behind the residency
//! [`Executor`], and [`build_executor`] which every serving path uses.
//!
//! Heavy, weight-holding models (z-image) get a bespoke adapter whose `activate`
//! builds the model on the assigned GPU and whose `Instance` owns it (so eviction
//! actually reclaims VRAM). Stateless providers (imageops, demo) ride the generic
//! [`residency::bridge::ProviderResident`]. The result is one `Executor` that
//! schedules, batches, and swaps every model uniformly, shared by `brain serve
//! --dbus` and (later) the CLI/event paths.

use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Media, Outcome, Progress};
use residency::bridge::ProviderResident;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use zimage::pipeline::{HotPipeline, Image, Paths};

/// Build the shared executor with every model registered, sized to the given per-GPU
/// budgets. `gpus` is `(index, total_bytes)` per card; `reserved` bytes are kept free
/// on each. The RAM pool bounds CPU-resident models. Falls back gracefully if a heavy
/// model's weights are not configured (it is simply not registered).
pub fn build_executor(gpus: &[(u32, u64)], npus: &[(u32, u64)], reserved: u64, ram_total: u64, models_dir: Option<&std::path::Path>, policy: Policy) -> Executor {
    let mut budgets = residency::budget::Budgets::new();
    for &(i, total) in gpus {
        budgets.set(Device::Gpu(i), total, reserved);
    }
    // NPUs get their own budget + lane; a model advertising an NPU path (MemCost.npu
    // > 0) is then auto-placed there in preference to CPU/GPU (see place::pick_device).
    for &(i, total) in npus {
        budgets.set(Device::Npu(i), total, 0);
    }
    budgets.set(Device::Cpu, ram_total, 0);

    let mut models: Vec<Arc<dyn ResidentModel>> = Vec::new();
    // z-image if its weights are configured (BRAIN_ZIMAGE_*).
    match ZImageResident::from_env() {
        Ok(z) => models.push(Arc::new(z)),
        Err(e) => eprintln!("brain: z-image not served over the scheduler ({e})"),
    }
    // yolo object detection if a checkpoint is configured (BRAIN_YOLO).
    if let Some(y) = YoloResident::from_env() {
        models.push(Arc::new(y));
    } else {
        eprintln!("brain: yolo not served over the scheduler (set BRAIN_YOLO to a checkpoint)");
    }
    // Text-generation LLMs (each gated on its own weights env var).
    if let Some(g) = crate::resident_llm::GptResident::from_env() {
        models.push(Arc::new(g));
    }
    if let Some(g) = crate::resident_llm::GlmResident::from_env() {
        models.push(Arc::new(g));
    }
    if let Some(q) = crate::resident_llm::QwenResident::from_env() {
        models.push(Arc::new(q));
    }
    // LFM2.5-Encoder (BRAIN_LFM + BRAIN_LFM_TOKENIZER): fill-mask + embeddings
    // with equal-length true batching (see resident_lfm.rs).
    if let Some(l) = crate::resident_lfm::LfmResident::from_env() {
        models.push(Arc::new(l));
    } else {
        eprintln!("brain: lfm not served over the scheduler (set BRAIN_LFM + BRAIN_LFM_TOKENIZER)");
    }
    // FLUX.2 Klein (BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}): text-to-image,
    // reference-image editing, LoRA training (see resident_flux2.rs).
    if let Some(f) = crate::resident_flux2::Flux2Resident::from_env() {
        models.push(Arc::new(f));
    } else {
        eprintln!("brain: flux2-klein not served over the scheduler (set BRAIN_FLUX2_DIT/_VAE/_TE/_TOKENIZER)");
    }
    // Monocular depth (BRAIN_DEPTH_WEIGHTS).
    if let Some(d) = crate::resident_depth::DepthResident::from_env() {
        models.push(Arc::new(d));
    }
    // Imaging models, each gated on its own weights env var: SAM 2.1 promptable
    // segmentation (BRAIN_SAM2_WEIGHTS, prompt-batched per image), the
    // antelopev2 face stack (BRAIN_FACENET_DIR), the VQ autoencoder
    // (BRAIN_VQGAN_WEIGHTS), CodeFormer restoration (BRAIN_RESTORE_WEIGHTS) and
    // the CLIP encoders (BRAIN_CLIP_DIR, genuinely batched per tower).
    // The imaging models come from `crate::catalog`, which owns their manifests
    // and providers too — so a model cannot be listed by `brain caps`, runnable
    // by `brain do` and yet missing here (which is exactly how Real-ESRGAN
    // shipped unreachable). Each is still gated on its own weights env var.
    models.extend(crate::catalog::residents());
    // Time-series forecasting foundation models — each gated on its weights env
    // var. chronos2/fincast advertise an NPU footprint (auto-placed on the NPU
    // when budgeted); kronos serves on CPU/GPU (see resident_forecast.rs).
    if let Some(c) = crate::resident_forecast::Chronos2Resident::from_env() {
        models.push(Arc::new(c));
    }
    if let Some(f) = crate::resident_forecast::FincastResident::from_env() {
        models.push(Arc::new(f));
    }
    if let Some(k) = crate::resident_forecast::KronosResident::from_env() {
        models.push(Arc::new(k));
    }
    // Text-to-speech (BRAIN_TTS_WEIGHTS).
    if let Some(t) = crate::resident_tts::TtsResident::from_env() {
        models.push(Arc::new(t));
    }
    // Speech-to-text: Nemotron 3.5 ASR (BRAIN_NEMOTRON) + Qwen3-ASR (BRAIN_QWEN_ASR).
    if let Some(a) = crate::resident_asr::NemotronResident::from_env() {
        models.push(Arc::new(a));
    }
    if let Some(a) = crate::resident_asr::QwenAsrResident::from_env() {
        models.push(Arc::new(a));
    }
    // Qwen3-Omni Thinker text generation, validation-tier (BRAIN_OMNI_HF_DIR)
    // -- see crate::resident_omni's module doc for the scope this covers.
    if let Some(o) = crate::resident_omni::OmniResident::from_env() {
        models.push(Arc::new(o));
    }
    // Deterministic mock model (BRAIN_MOCK): a real ResidentModel — no weights, no
    // GPU — registered as `mock` so the HTTP conformance harness can validate the
    // whole API surface through the true serving path (placement → activate →
    // run_batch). Advertises generate (chat) + embed + text2image.
    if let Some(m) = crate::resident_mock::MockResident::from_env() {
        models.push(Arc::new(m));
    }
    // Stateless helpers (no weights) — always available. `demo` is the worked
    // example every transport smoke (busctl_smoke.sh) exercises.
    models.push(Arc::new(ProviderResident::stateless(Arc::new(crate::caps_cli::DemoModel))));
    models.push(Arc::new(ProviderResident::stateless(Arc::new(crate::imageops::ImageOps))));
    // FastVLM captioning: the provider manages its own weight residency
    // (lazy per checkpoint dir, resident thereafter), so it serves as a
    // stateless resident — invoking it with no checkpoint on disk is a clean
    // per-call error, not a registration failure.
    models.push(Arc::new(ProviderResident::stateless(Arc::new(fastvlm::caps::FastVlmProvider::new()))));
    // brain/imgpipe: the pipeline holds no weights of its own (each stage
    // resolves its own via BRAIN_* env vars, same as when called through
    // `brain do`), so it is stateless from the scheduler's point of view too.
    // Built via `catalog::provider`, not a fresh `PipelineProvider`, so this
    // resident is guaranteed to compose the SAME stage registry `brain caps`/
    // `brain do` see — the earlier bug (`ai-forever/Real-ESRGAN` unreachable
    // over D-Bus/HTTP despite a working `brain do`) was exactly two lists
    // drifting apart. `provider()` never fails for imgpipe (its ctor is
    // `always!`-shaped), so this push is unconditional.
    if let Ok(p) = crate::catalog::provider(imgpipe::caps::MODEL) {
        models.push(Arc::new(ProviderResident::stateless(p)));
    }

    // Global model directory: append every discovered file as its own catalog
    // entry (keyed by model-card id), deduped against the env-gated residents
    // above (their manifest model == id). Additive — the env-gated path stands
    // on its own when no dir is configured or the scan finds nothing.
    if let Some(dir) = models_dir {
        let existing: std::collections::BTreeSet<String> = models.iter().map(|m| m.manifest().model).collect();
        for r in crate::model_dir::discover(dir) {
            let id = r.manifest().model;
            if existing.contains(&id) {
                eprintln!("brain: model dir entry '{id}' shadowed by an env-gated resident; keeping the env one");
                continue;
            }
            models.push(r);
        }
    }

    Executor::start(models, budgets, policy)
}

// ---------------------------------------------------------------- yolo

/// COCO-80 class names (index → label), so detections carry human labels (dog = 16).
const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// YOLO detection behind the scheduler. Loads a brain-format YOLOv8 checkpoint
/// (`BRAIN_YOLO`); the resident instance holds the model on the CPU (brain's yolo
/// default) — dropping it frees the RAM. One action, `detect`.
pub struct YoloResident {
    /// Catalog id (the model-card id): the manifest/instance-key key, so two
    /// checkpoints of the same family are two distinct selectable models
    /// (mirrors `resident_llm.rs::GptResident`).
    id: String,
    path: String,
}

impl YoloResident {
    pub fn from_env() -> Option<YoloResident> {
        let path = std::env::var("BRAIN_YOLO").ok().filter(|p| !p.is_empty())?;
        // See resident_llm.rs::GptResident::from_env's comment: env-loaded,
        // no upstream vendor/repo provenance.
        Some(Self::from_card(&path, &checkpoint::st::ModelCard::new("brain/yolo", "yolo"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused -- yolo's class
    /// names are a fixed COCO-80 table, not learned from a tokenizer.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, _tokenizer: Option<&str>) -> YoloResident {
        YoloResident { id: card.id.clone(), path: path.to_string() }
    }

    fn detect_spec() -> capability::ActionSpec {
        use capability::{BlobSpec, ParamSpec, ParamType};
        capability::ActionSpec::new("detect", "detect objects in an image (YOLOv8, COCO-80 classes)")
            .param(ParamSpec::new("conf", ParamType::Float, "confidence threshold").default(json!(0.25)))
            .param(ParamSpec::new("iou", ParamType::Float, "NMS IoU threshold").default(json!(0.45)))
            .input(BlobSpec::new("image", Media::Image, "the image to run detection on").required())
    }
}

impl ResidentModel for YoloResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, "object detection (YOLOv8, COCO-80)", vec![Self::detect_spec()])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // YOLOv8n is small and runs on the CPU in brain → a modest RAM footprint.
        MemCost::new(0, 128 << 20)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // `BRAIN_YOLO_BATCH` (default 1) sets the forward batch: >1 enables a TRUE
        // batched forward (one detect over N images) when the scheduler groups jobs.
        let batch = std::env::var("BRAIN_YOLO_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1u32).max(1);
        Ok(Box::new(YoloInstance { yolo: yolo::Yolo::load(&self.path, batch), batch: batch as usize }))
    }
}

struct YoloInstance {
    yolo: yolo::Yolo,
    batch: usize,
}

fn detections_outcome(dets: &[[f32; 6]]) -> Outcome {
    let objects: Vec<Value> = dets
        .iter()
        .map(|d| {
            let cls = d[5] as usize;
            json!({"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "class": cls, "label": COCO.get(cls).copied().unwrap_or("?")})
        })
        .collect();
    Outcome::new().set("count", json!(objects.len())).set("detections", json!(objects))
}

impl Instance for YoloInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    /// TRUE batched forward: chunk the invocations to the model's batch and run one
    /// `detect_batch` per chunk (the last chunk padded to the batch, its padding
    /// results discarded). With batch 1 this is one forward per image.
    fn run_batch(&mut self, _action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        let b = self.batch;
        let mut out: Vec<ActionResult> = Vec::with_capacity(invs.len());
        for chunk in invs.chunks(b) {
            // Decode this chunk's images (errors become per-job error results).
            let mut imgs: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(chunk.len());
            let mut errs: Vec<Option<String>> = Vec::with_capacity(chunk.len());
            for inv in chunk {
                match capability::blob::decode_image(inv, "image") {
                    Ok(im) => {
                        imgs.push(im);
                        errs.push(None);
                    }
                    Err(e) => errs.push(Some(e)),
                }
            }
            if imgs.is_empty() {
                out.extend(errs.into_iter().map(|e| Err(e.unwrap_or_default())));
                continue;
            }
            // NMS thresholds from the first valid invocation (post-forward, per-image).
            let (conf, iou) = chunk
                .iter()
                .find_map(|i| i.get_blob("image").map(|_| (i.get_f64("conf").unwrap_or(0.25) as f32, i.get_f64("iou").unwrap_or(0.45) as f32)))
                .unwrap_or((0.25, 0.45));
            // Pad to the model's batch by repeating the last image; drop the padding.
            let last = imgs.last().unwrap().clone();
            while imgs.len() < b {
                imgs.push(last.clone());
            }
            let refs: Vec<(&[f32], u32, u32)> = imgs.iter().map(|(p, w, h)| (p.as_slice(), *w, *h)).collect();
            let batched = self.yolo.detect_batch(&refs, conf, iou);
            // Zip results back to the (possibly-erroring) chunk jobs, in order.
            let mut valid = batched.into_iter();
            for e in errs {
                match e {
                    Some(msg) => out.push(Err(msg)),
                    None => out.push(Ok(detections_outcome(&valid.next().unwrap_or_default()))),
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------- z-image

/// z-image behind the scheduler. A resident instance is a built [`HotPipeline`] for a
/// `(width, height, precision, adapter)` key — the DiT (and its encoder) on the GPU;
/// dropping it frees the VRAM. `text2image` runs on the resident pipeline; the
/// image-editing actions build fresh per call (they take a variable-size input) via a
/// held provider, so the full manifest still works over the bus.
pub struct ZImageResident {
    id: String,
    paths: Paths,
    provider: Arc<zimage::caps::ZImageProvider>,
}

impl ZImageResident {
    pub fn from_env() -> Result<ZImageResident, String> {
        Self::from_paths(zimage::caps::MODEL, Paths::from_env()?)
    }

    /// Built from an already-resolved [`Paths`] rather than the environment,
    /// under `id` rather than the compiled-in `zimage::caps::MODEL` -- what
    /// `crate::model_dir::resident_for_local` uses for a compound
    /// (multi-file) model found on disk or just auto-fetched, whose four
    /// component paths come from a `brain.manifest.json`'s roles and whose id
    /// is the fully-qualified ref it was fetched as (e.g.
    /// `Tongyi-MAI/Z-Image-Turbo`) -- registering under the compiled-in
    /// constant instead would silently strand the request that triggered the
    /// fetch (it named the fetched ref, not `brain/z-image`).
    pub fn from_paths(id: impl Into<String>, paths: Paths) -> Result<ZImageResident, String> {
        Ok(ZImageResident { id: id.into(), paths, provider: Arc::new(zimage::caps::ZImageProvider::load()?) })
    }
}

impl ResidentModel for ZImageResident {
    fn manifest(&self) -> Manifest {
        let mut m = zimage::caps::manifest();
        m.model = self.id.clone();
        m
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        if action == "text2image" {
            let w = inv.get_i64("width").unwrap_or(1024);
            let h = inv.get_i64("height").unwrap_or(1024);
            let prec = if inv.get_str("precision").as_deref() == Some("fp32") { "fp32" } else { "int8" };
            let adapter = inv.get_str("adapter").unwrap_or_default();
            InstanceKey::new(&self.id, format!("{w}x{h}:{prec}:{adapter}"))
        } else {
            // Editing/training actions build fresh per call — one transient instance.
            InstanceKey::new(&self.id, format!("edit:{action}"))
        }
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // int8 DiT (~13 GB) or fp32 sharded (~24 GB/card); edit builds are transient
        // and small-footprint (they build + drop within the call).
        let vram = if key.config.contains(":fp32:") {
            24u64 << 30
        } else if key.config.starts_with("edit:") {
            2u64 << 30
        } else {
            14u64 << 30
        };
        MemCost::new(vram, 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if key.config.starts_with("edit:") {
            // No persistent pipeline — the provider builds fresh per call.
            return Ok(Box::new(ZImageInstance { pipe: None, provider: self.provider.clone() }));
        }
        let (w, h, hifi, adapter) = parse_key(&key.config);
        let adapter = if adapter.is_empty() { None } else { Some(adapter.as_str()) };
        // Place the DiT on the assigned card (scoped registry selection); the
        // encoder card is z-image's own (BRAIN_ZIMAGE_ENCODER_GPU) and left as
        // configured.
        let pipe = crate::resident_llm::on_device(device, || {
            HotPipeline::build_adapted(&self.paths, w, h, 64, hifi, adapter, |_| {})
        })??;
        Ok(Box::new(ZImageInstance { pipe: Some(pipe), provider: self.provider.clone() }))
    }
}

/// A resident z-image instance: `pipe` when a text2image pipeline is built; the
/// `provider` handles the fresh-build editing/training actions.
struct ZImageInstance {
    pipe: Option<HotPipeline>,
    provider: Arc<zimage::caps::ZImageProvider>,
}

impl Instance for ZImageInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action == "text2image" {
            let pipe = self.pipe.as_ref().ok_or("z-image: text2image instance has no pipeline")?;
            let prompt = inv.get_str("prompt").unwrap_or_default();
            let seed = inv.get_i64("seed").unwrap_or(42).max(0) as u64;
            let steps = inv.get_i64("steps").unwrap_or(8).max(1) as u32;
            let img = pipe.generate(&prompt, seed, steps, &inv.cancel, |s, t, m| progress(Progress::step(s, t, m.to_string())))?;
            return Ok(emit_image(img));
        }
        // Editing / training: delegate to the provider's action (fresh build).
        use capability::Provider;
        let act = self.provider.action(action).ok_or_else(|| format!("z-image: unknown action '{action}'"))?;
        let inv = act.spec().validate(inv.clone())?;
        act.run(&inv, progress)
    }
}

/// Parse a `"WxH:precision:adapter"` instance key.
fn parse_key(config: &str) -> (u32, u32, bool, String) {
    let mut parts = config.splitn(3, ':');
    let wh = parts.next().unwrap_or("1024x1024");
    let prec = parts.next().unwrap_or("int8");
    let adapter = parts.next().unwrap_or("").to_string();
    let (w, h) = wh.split_once('x').unwrap_or(("1024", "1024"));
    (w.parse().unwrap_or(1024), h.parse().unwrap_or(1024), prec == "fp32", adapter)
}

/// Wrap a generated [`Image`] as an image-output [`Outcome`] (the shared
/// `capability::blob` wire format).
fn emit_image(img: Image) -> Outcome {
    Outcome::new()
        .set("width", json!(img.w))
        .set("height", json!(img.h))
        .blob("image", capability::blob::image_blob(&img.hwc, img.w as u32, img.h as u32, 3))
}
