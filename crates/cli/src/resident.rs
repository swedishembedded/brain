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
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel, Tier};
use serde_json::{json, Value};
use s3dit::pipeline::{HotPipeline, Image, Paths};

/// Build the shared executor with every model registered, sized to the given per-GPU
/// budgets. `gpus` is `(index, total_bytes)` per card; `reserved` bytes are kept free
/// on each. `unified_gpus` names the indices among `gpus` that physically share RAM
/// with the CPU (an integrated GPU, or the no-discrete-GPU fallback) — every NPU
/// always shares RAM too (`docs/models/*`'s Meteor-Lake NPU note) — so those,
/// plus `Device::Cpu`, are declared into ONE `memauth` pool sized to `pool_ram` (the
/// real, physical host RAM — see that parameter's own doc for why this must NEVER be
/// `cpu_ram`); a discrete GPU keeps its own independent budget, unaffected. Falls back
/// gracefully if a heavy model's weights are not configured (it is simply not
/// registered).
///
/// `cpu_ram`/`pool_ram` are deliberately two separate parameters, not one reused
/// value: `cpu_ram` is `Device::Cpu`'s OWN per-device budget (legitimately `0` when
/// `--device` excludes CPU from compute — that alone is what stops the CPU device
/// itself being chosen as a placement target), while `pool_ram` is the physical
/// capacity of the shared pool `Device::Cpu` and every unified GPU/NPU draw from
/// together. Passing the same (possibly zeroed) value for both used to starve the
/// POOL to zero bytes too whenever `--device` excluded CPU — correctly stopping CPU
/// placements, but ALSO clamping every unified GPU/NPU's `usable_on`/`free_on` to
/// `min(real_budget, 0) == 0` (`residency::budget::Budgets::usable_on`'s `pool.min`),
/// making an otherwise-perfectly-placeable model (e.g. a small model on an
/// integrated GPU with `--device gpu`) silently unplaceable forever — no error, just
/// a 10s admission timeout and a generic 429, since a claim failure never fires
/// `on_admit`. The physical RAM does not disappear just because CPU-side compute is
/// disabled, so `pool_ram` must always be the real, ungated host RAM figure.
pub fn build_executor(gpus: &[(u32, u64)], npus: &[(u32, u64)], unified_gpus: &[u32], reserved: u64, cpu_ram: u64, pool_ram: u64, models_dir: Option<&std::path::Path>, policy: Policy) -> Executor {
    let mut budgets = residency::budget::Budgets::new();
    for &(i, total) in gpus {
        budgets.set(Device::Gpu(i), total, reserved);
    }
    // NPUs get their own budget + lane; a model advertising an NPU path (MemCost.npu
    // > 0) is then auto-placed there in preference to CPU/GPU (see place::pick_device).
    for &(i, total) in npus {
        budgets.set(Device::Npu(i), total, 0);
    }
    budgets.set(Device::Cpu, cpu_ram, 0);
    // The unified-memory fix (see memauth's module doc): declare every device
    // that physically shares this RAM into ONE pool, so a charge on any of
    // them correctly reduces what all the others have free — instead of two
    // (or more) independent budgets that together claim more bytes than the
    // machine has. Sized to `pool_ram`, NEVER `cpu_ram` — see this function's
    // own doc on why those must stay two separate parameters.
    let mut shared: Vec<Device> = vec![Device::Cpu];
    shared.extend(unified_gpus.iter().map(|&i| Device::Gpu(i)));
    shared.extend(npus.iter().map(|&(i, _)| Device::Npu(i)));
    if shared.len() > 1 {
        budgets.set_pool(memauth::HOST_POOL, &shared, pool_ram, 0);
    }

    let mut models: Vec<Arc<dyn ResidentModel>> = Vec::new();
    // z-image if its weights are configured (BRAIN_ZIMAGE_*).
    match ZImageResident::from_env() {
        Ok(z) => models.push(Arc::new(z)),
        Err(e) => eprintln!("brain: z-image not served over the scheduler ({e})"),
    }
    // yolo object detection if a checkpoint is configured (BRAIN_YOLOV8).
    if let Some(y) = YoloResident::from_env() {
        models.push(Arc::new(y));
    } else {
        eprintln!("brain: yolo not served over the scheduler (set BRAIN_YOLOV8 to a checkpoint)");
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
    // Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder
    // (BRAIN_QWEN35MOE_WEIGHTS + BRAIN_QWEN35MOE_TOKENIZER) -- single-GPU,
    // fp32 weights + KV only (see resident_qwen35moe.rs's own module doc for
    // the exact scope vs QwenResident's).
    if let Some(q) = crate::resident_qwen35moe::Qwen35Resident::from_env() {
        models.push(Arc::new(q));
    }
    // Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder
    // (BRAIN_QWEN35_WEIGHTS + BRAIN_QWEN35_TOKENIZER) -- same single-GPU,
    // fp32 weights + KV scope as qwen35moe above (see resident_qwen35.rs's
    // own module doc).
    if let Some(q) = crate::resident_qwen35::Qwen35Resident::from_env() {
        models.push(Arc::new(q));
    }
    // LFM2.5-Encoder (BRAIN_LFM2 + BRAIN_LFM2_TOKENIZER): fill-mask + embeddings
    // with equal-length true batching (see resident_lfm.rs).
    if let Some(l) = crate::resident_lfm::LfmResident::from_env() {
        models.push(Arc::new(l));
    } else {
        eprintln!("brain: lfm not served over the scheduler (set BRAIN_LFM2 + BRAIN_LFM2_TOKENIZER)");
    }
    // FLUX.2 Klein (BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}): text-to-image,
    // reference-image editing, LoRA training (see resident_flux2.rs).
    if let Some(f) = crate::resident_flux2::Flux2Resident::from_env() {
        models.push(Arc::new(f));
    } else {
        eprintln!("brain: flux2-klein not served over the scheduler (set BRAIN_FLUX2_DIT/_VAE/_TE/_TOKENIZER)");
    }
    // Wan2.1 text-to-video (BRAIN_WAN_{DIT,VAE,T5,TOKENIZER}): a resident
    // transformer per (variant, frames, size) - see resident_wan.rs.
    if let Some(w) = crate::resident_wan::WanResident::from_env() {
        models.push(Arc::new(w));
    } else {
        eprintln!("brain: wan not served over the scheduler (set BRAIN_WAN_DIT/_VAE/_T5/_TOKENIZER)");
    }
    // LTX-2.5 text-to-video (BRAIN_LTXV_VAE): a smoke-test pipeline (real VAE,
    // tiny random-weight DiT, no real text encoder yet) with nothing worth
    // caching resident - see resident_ltxv.rs's module doc.
    if let Some(l) = crate::resident_ltxv::LtxvResident::from_env() {
        models.push(Arc::new(l));
    } else {
        eprintln!("brain: ltxv not served over the scheduler (set BRAIN_LTXV_VAE)");
    }
    // Monocular depth (BRAIN_ZIPDEPTH_WEIGHTS).
    if let Some(d) = crate::resident_depth::DepthResident::from_env() {
        models.push(Arc::new(d));
    }
    // Imaging models, each gated on its own weights env var: SAM 2.1 promptable
    // segmentation (BRAIN_SAM2_WEIGHTS, prompt-batched per image), the
    // antelopev2 face stack (BRAIN_SCRFD_DIR + BRAIN_ARCFACE_DIR), the VQ autoencoder
    // (BRAIN_VQGAN_WEIGHTS), CodeFormer restoration (BRAIN_CODEFORMER_WEIGHTS) and
    // the CLIP encoders (BRAIN_CLIP_DIR, genuinely batched per tower).
    // The imaging models come from `crate::catalog`, which owns their manifests
    // and providers too — so a model cannot be listed by `brain caps`, runnable
    // by `brain do` and yet missing here (which is exactly how Real-ESRGAN
    // shipped unreachable). Each is still gated on its own weights env var.
    // ... plus, from the same catalog: TTS (BRAIN_QWEN3TTS_WEIGHTS), speech-to-text
    // (BRAIN_NEMOTRONASR + BRAIN_QWEN3ASR), and the forecasting foundation models
    // (BRAIN_CHRONOS2 / BRAIN_FINCAST / BRAIN_KRONOS_* — chronos2/fincast
    // advertise an NPU footprint and auto-place there when budgeted). Folded
    // into `catalog::models()` so `brain caps`/`brain do` and this executor
    // can no longer disagree about their existence.
    models.extend(crate::catalog::residents());
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

    let exec = Executor::start(models, budgets, policy);
    // Qwen3-Omni (BRAIN_QWEN3OMNIMOE_HF_DIR): the full chat/multimodal surface, placed
    // across as many budgeted cards as its real per-layer bytes need. Like the
    // int8 Thinker below it is multi-device and therefore registered AFTER
    // `start` via `register_multi` -- see resident_omni's module doc for why a
    // plain `register` (which is what it used to take) let it spend VRAM the
    // scheduler had not budgeted.
    if let Some(o) = crate::resident_omni::OmniResident::from_env(gpus, reserved) {
        exec.register_multi(Arc::new(o));
    }
    // The int8 dual-GPU Thinker is multi-device-only, so it is registered
    // AFTER `start` via `register_multi`, never folded into `models` above
    // (see `resident_omni::int8_thinker_multi_from_env`'s own doc for why a
    // plain `register` would be structurally wrong for it).
    if let Some(t) = crate::resident_omni::int8_thinker_multi_from_env(gpus, reserved) {
        exec.register_multi(Arc::new(t));
    } else {
        eprintln!("brain: {} not served over the scheduler (set BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT)", qwen3omnimoe::int8_thinker_resident::MODEL);
    }
    exec
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
/// (`BRAIN_YOLOV8`); the resident instance holds the model on the CPU (brain's yolo
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
        let path = std::env::var("BRAIN_YOLOV8").ok().filter(|p| !p.is_empty())?;
        // See resident_llm.rs::GptResident::from_env's comment: env-loaded,
        // no upstream vendor/repo provenance.
        Some(Self::from_card(&path, &checkpoint::st::ModelCard::new("brain/yolov8", "yolo"), None))
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
        // `BRAIN_YOLOV8_BATCH` (default 1) sets the forward batch: >1 enables a TRUE
        // batched forward (one detect over N images) when the scheduler groups jobs.
        let batch = std::env::var("BRAIN_YOLOV8_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1u32).max(1);
        Ok(Box::new(YoloInstance { yolo: yolov8::Yolo::load(&self.path, batch), batch: batch as usize }))
    }
}

struct YoloInstance {
    yolo: yolov8::Yolo,
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
    provider: Arc<s3dit::caps::ZImageProvider>,
}

impl ZImageResident {
    pub fn from_env() -> Result<ZImageResident, String> {
        Self::from_paths(s3dit::caps::MODEL, Paths::from_env()?)
    }

    /// Built from an already-resolved [`Paths`] rather than the environment,
    /// under `id` rather than the compiled-in `s3dit::caps::MODEL` -- what
    /// `crate::model_dir::resident_for_local` uses for a compound
    /// (multi-file) model found on disk or just auto-fetched, whose four
    /// component paths come from a `brain.manifest.json`'s roles and whose id
    /// is the fully-qualified ref it was fetched as (e.g.
    /// `Tongyi-MAI/Z-Image-Turbo`) -- registering under the compiled-in
    /// constant instead would silently strand the request that triggered the
    /// fetch (it named the fetched ref, not `brain/s3dit`).
    pub fn from_paths(id: impl Into<String>, paths: Paths) -> Result<ZImageResident, String> {
        Ok(ZImageResident { id: id.into(), paths, provider: Arc::new(s3dit::caps::ZImageProvider::load()?) })
    }
}

impl ResidentModel for ZImageResident {
    fn manifest(&self) -> Manifest {
        let mut m = s3dit::caps::manifest();
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
        // int8 DiT (~13 GB); edit builds are transient and small-footprint
        // (they build + drop within the call). fp32 delegates to
        // `s3dit::pipeline::hifi_cost_bytes`, which picks between the
        // 2-GPU-shard estimate and the real windowed-engine estimate from
        // the SAME machine-shape decision (`gpu_core::devices::schedulable_gpu_count()`)
        // `DitEngine::build_from_source` itself makes — the number budgeted
        // here and the number the code actually allocates must be the same
        // expression, or this estimate silently outlives whichever engine
        // it was written for.
        if key.config.contains(":fp32:") {
            let (vram, ram) = s3dit::pipeline::hifi_cost_bytes(gpu_core::devices::schedulable_gpu_count());
            return MemCost::new(vram, ram);
        }
        let vram = if key.config.starts_with("edit:") { 2u64 << 30 } else { 14u64 << 30 };
        MemCost::new(vram, 0)
    }

    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        // The shape is `ZImageConfig::turbo()` because that IS the only config
        // the pipeline ever builds (`s3dit::pipeline` hardcodes turbo() at
        // every build site); deriving it per-checkpoint belongs with the model
        // crate growing a second config, not here.
        let cache_ram = || s3dit::pipeline::int8_cache_bytes_estimate(&s3dit::ZImageConfig::turbo());
        match tier {
            // A cache-retaining build holds the multi-GB host `DitI8Cache`
            // ALONGSIDE the hot pipeline — charging Hot only the VRAM left
            // those bytes invisible to every budget exactly while they
            // coexist with the device copy (the residency contract this
            // adapter exists to keep honest). Only the shape that can
            // actually retain a cache pays it: fp32/adapter/edit builds
            // keep the plain estimate.
            Tier::Hot => {
                let mut cost = self.estimate(key);
                let (_, _, hifi, adapter) = parse_key(&key.config);
                let adapter = if adapter.is_empty() { None } else { Some(adapter.as_str()) };
                if !key.config.starts_with("edit:") && retains_int8_cache(hifi, adapter) {
                    cost.ram += cache_ram();
                }
                cost
            }
            // Real, not `0`: only the plain int8 build (see
            // `retains_int8_cache`) ever actually has a Warm state (every
            // other shape's `demote` refuses, so the manager never
            // consults this for those) -- but when it does, the retained
            // `DitI8Cache` genuinely holds several GB, and claiming
            // otherwise is precisely the kind of budgeting lie this whole
            // workstream exists to avoid.
            Tier::Warm | Tier::Cold => MemCost::new(0, cache_ram()),
        }
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if key.config.starts_with("edit:") {
            // No persistent pipeline — the provider builds fresh per call.
            return Ok(Box::new(ZImageInstance { pipe: None, dit_cache: None, provider: self.provider.clone(), paths: self.paths.clone(), width: 0, height: 0, cap_len: 0 }));
        }
        let (w, h, hifi, adapter) = parse_key(&key.config);
        let adapter = if adapter.is_empty() { None } else { Some(adapter.as_str()) };
        let cap_len = 64;
        // Place the DiT on the assigned card (scoped registry selection); the
        // encoder card is z-image's own (BRAIN_S3DIT_ENCODER_GPU) and left as
        // configured.
        let (pipe, dit_cache) = if retains_int8_cache(hifi, adapter) {
            let (pipe, cache) = crate::resident_llm::on_device(device, || HotPipeline::build_adapted_with_cache(&self.paths, w, h, cap_len, |_| {}))??;
            (pipe, Some(cache))
        } else {
            let pipe = crate::resident_llm::on_device(device, || HotPipeline::build_adapted(&self.paths, w, h, cap_len, hifi, adapter, |_| {}))??;
            (pipe, None)
        };
        Ok(Box::new(ZImageInstance { pipe: Some(pipe), dit_cache, provider: self.provider.clone(), paths: self.paths.clone(), width: w, height: h, cap_len }))
    }
}

/// Whether an `activate` for `(hifi, adapter)` should retain a
/// [`s3dit::DitI8Cache`] alongside the built pipeline - real, permanent
/// extra host RAM (see [`ZImageDitI8::build_from_source_with_cache`]'s
/// doc), so opt-in only (`BRAIN_S3DIT_RETAIN_INT8_CACHE=1`) and only for
/// the one shape a cache can even be built for: plain int8, no adapter.
/// fp32 and LoRA-folded builds always return `false` regardless of the env
/// var — `demote` for those stays the manager's unmodified default
/// (`Err("unsupported")`, today's full drop-and-rebuild), not a silent lie.
fn retains_int8_cache(hifi: bool, adapter: Option<&str>) -> bool {
    !hifi && adapter.is_none() && std::env::var("BRAIN_S3DIT_RETAIN_INT8_CACHE").ok().as_deref() == Some("1")
}

/// A resident z-image instance: `pipe` when a text2image pipeline is built; the
/// `provider` handles the fresh-build editing/training actions. `dit_cache`
/// is `Some` only for an instance that opted into [`retains_int8_cache`] —
/// what makes `demote`/`promote` real instead of the default `Err`.
struct ZImageInstance {
    pipe: Option<HotPipeline>,
    dit_cache: Option<s3dit::DitI8Cache>,
    provider: Arc<s3dit::caps::ZImageProvider>,
    paths: Paths,
    width: u32,
    height: u32,
    cap_len: u32,
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

    fn metrics(&self) -> Vec<(String, Value)> {
        self.pipe.as_ref().map(|p| p.metrics()).unwrap_or_default()
    }

    /// Real only when `activate` retained a `dit_cache` (plain int8, no
    /// adapter, `BRAIN_S3DIT_RETAIN_INT8_CACHE=1`): drops the whole
    /// resident pipeline — encoder, DiT, VAE, every device buffer — while
    /// the cache (already held separately) survives, ready for `promote`.
    /// Refuses for everything else (fp32, an adapter build, or a plain
    /// int8 build that didn't opt in), so the manager falls back to its
    /// default full evict+rebuild exactly as it does for every model that
    /// never overrides this.
    fn demote(&mut self, tier: Tier) -> Result<(), String> {
        if tier == Tier::Hot {
            return Err("z-image: Hot is not a demotion target".to_string());
        }
        if self.dit_cache.is_none() {
            return Err("z-image: this instance retained no demote/promote cache".to_string());
        }
        self.pipe = None;
        Ok(())
    }

    /// The inverse: rebuild the pipeline from `dit_cache` on `device` — no
    /// DiT checkpoint read, no re-quantization (see
    /// `s3dit::ZImageDitI8::rebuild_from_cache`'s doc). Only reachable
    /// after a successful `demote`, so `dit_cache` is always `Some` here.
    fn promote(&mut self, device: Device) -> Result<(), String> {
        let cache = self.dit_cache.as_ref().ok_or("z-image: promote called on an instance with no retained cache")?;
        let pipe = crate::resident_llm::on_device(device, || HotPipeline::build_from_dit_cache(&self.paths, self.width, self.height, self.cap_len, cache, |_| {}))??;
        self.pipe = Some(pipe);
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Real end-to-end proof that demote/promote works against the actual
    /// ~31 GB Z-Image checkpoint, not a synthetic model: activate (builds
    /// fresh, retains a DitI8Cache), demote (drops the whole pipeline --
    /// GPU AND host -- keeping only the cache), promote (rebuilds from the
    /// cache: no checkpoint read, no re-quantization), then run a real
    /// generation and confirm it produces a real image. Times both
    /// activate() and promote() so "promote is faster" is a measured
    /// number, not an assertion resting on the design alone.
    #[test]
    #[ignore = "slow: real checkpoint + GPU; set BRAIN_S3DIT_* and run with --ignored"]
    fn zimage_demote_then_promote_produces_a_real_image_and_promote_is_faster() {
        std::env::set_var("BRAIN_S3DIT_RETAIN_INT8_CACHE", "1");
        // Two different failures used to share one skip: "the checkpoint paths
        // are not on this box" (a fixture that is legitimately absent) and
        // "the paths resolved but the provider would not load" (a real
        // failure). Resolve them separately so only the first is a skip.
        let paths = match s3dit::pipeline::Paths::from_env() {
            Ok(p) => p,
            Err(e) => return brain_testutil::skip(&format!("Z-Image checkpoint paths not set: {e}")),
        };
        let model = ZImageResident::from_paths(s3dit::caps::MODEL, paths)
            .expect("BRAIN_S3DIT_* all resolved, so the Z-Image provider must load");
        let key = InstanceKey::new(s3dit::caps::MODEL, "256x256:int8:");

        let t0 = std::time::Instant::now();
        let mut inst = model.activate(&key, Device::Gpu(0)).expect("activate");
        let activate_secs = t0.elapsed().as_secs_f64();

        inst.demote(Tier::Warm).expect("a cache-retaining instance must demote successfully");

        let t1 = std::time::Instant::now();
        inst.promote(Device::Gpu(0)).expect("promote must rebuild from the cache");
        let promote_secs = t1.elapsed().as_secs_f64();

        let inv = Invocation::new().set("prompt", json!("a red fox in snow, photograph")).set("seed", json!(42)).set("steps", json!(4));
        let result = inst.run("text2image", &inv, &mut |_| {}).expect("run after promote must succeed");
        assert!(result.blobs.contains_key("image"), "a promoted instance must still be able to generate a real image");

        eprintln!("activate: {activate_secs:.1}s, promote: {promote_secs:.1}s");
        assert!(promote_secs < activate_secs, "promote (cache-based) must be faster than the fresh activate() it followed -- it skips the checkpoint read AND the quantization activate() just did");
    }
}
