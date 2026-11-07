// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`build_executor`]: the one shared serving [`Executor`], with every model the
//! machine can serve registered and sized to the machine's budgets.
//!
//! The adapters themselves live in `crates/catalog` (one entry per model, so a
//! model cannot be listed, runnable and unscheduled all at once). This module
//! only assembles them: it builds the memory budgets from a [`Machine`],
//! registers the catalog's single- and multi-device residents, the no-weights
//! stateless helpers, and whatever the models directory holds, then starts the
//! executor. `brain serve` and any other embedder that wants the same
//! scheduling, batching and eviction call it.

use std::sync::Arc;

use catalog::adapter_release::ServedBase;
use catalog::adapter_registry::AdapterHost;
use catalog::resident_llm::QwenResident;
use catalog::resident_yolo::YoloResident;
use catalog::resident_zimage::ZImageResident;
use residency::bridge::ProviderResident;
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, Executor, Policy, ResidentModel};

use crate::Machine;


/// The one shared serving [`Executor`], plus the CONCRETE model handles a
/// caller needs for an operation the erased `Arc<dyn ResidentModel>` does not
/// carry.
///
/// Today that is exactly one: offering LoRA adapters beside the Qwen3 base
/// ([`QwenResident::adapter_view`], what the CLI's `continuous_train` drives
/// through `catalog::adapter_registry::LiveAdapters`). It is inherent rather
/// than a `ResidentModel` method because "serve this adapter beside yourself"
/// is not a concept the trait has - putting it there would put a decoder-LM-
/// shaped method on every detector, VAE and forecaster that implements it - so
/// the erased handle the executor holds cannot do it at all.
///
/// The handle here is the SAME `Arc` that was registered, never a second
/// `QwenResident` over the same checkpoint: an adapter view leases an entry of
/// that object's own registry, read by ITS instance, so an adapter offered on
/// a copy is a model no instance ever serves.
pub struct Serving {
    pub executor: Executor,
    /// `None` when `--checkpoint` names no checkpoint, i.e. this process
    /// serves no flag-selected Qwen3 for an adapter flag to apply to.
    pub qwen: Option<Arc<QwenResident>>,
    /// The released Qwen3.8 GGUFs the model scan registered, as concrete
    /// handles for the same reason: an adapter flag (or a watcher) offers its
    /// adapters on the allocation the executor holds. When `--checkpoint`
    /// names one of them, only that one: it is the checkpoint's host.
    pub qwen35: Vec<Arc<catalog::resident_qwen35::Qwen35GgufServed>>,
    /// The Ripple checkpoints the model scan registered, as concrete handles: `--adapter` offers its adapter as a
    /// model of its own beside the base ([`catalog::resident_ricursive::RicursiveResident::with_adapter`]).
    pub ricursive: Vec<Arc<catalog::resident_ricursive::RicursiveResident>>,
}

/// Register `qwen` (when its weights are configured) as an ordinary erased
/// resident AND keep the concrete handle - see [`Serving::qwen`] for why
/// both, and why they must be one allocation.
fn register_qwen(models: &mut Vec<Arc<dyn ResidentModel>>, qwen: Option<QwenResident>) -> Option<Arc<QwenResident>> {
    let q = Arc::new(qwen?);
    models.push(q.clone());
    Some(q)
}

/// The index in `scanned` of the base `checkpoint` (the base `--checkpoint`
/// named) is: the same store id, else the same checkpoint on disk.
fn checkpoint_base(checkpoint: &ServedBase, scanned: &[ServedBase]) -> Option<usize> {
    let same_id = |b: &ServedBase| checkpoint.id().is_some() && checkpoint.id() == b.id();
    let same_path = |b: &ServedBase| same_checkpoint(&checkpoint.path().to_string_lossy(), &b.path().to_string_lossy());
    scanned.iter().position(|b| same_id(b) || same_path(b))
}

/// Whether two weight paths name the same file.
fn same_checkpoint(a: &str, b: &str) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// The per-model serving knobs, parsed from flags. The CLI parses, this
/// crate never reads the environment for them.
#[derive(Default)]
pub struct ServeKnobs {
    /// The Qwen3 paged engine.
    pub qwen: catalog::resident_llm::QwenServeConfig,
    /// Qwen3.8: `gguf` holds the server-wide defaults of every released GGUF
    /// the model scan serves through the multi-card INT8 resident, and `fp32`
    /// sizes the brain-checkpoint resident.
    pub qwen35: catalog::resident_qwen35::Qwen35ServeConfig,
    /// Ripple: the model-agnostic policy flags it honours (`--weight-format`, `--kv-cache`, `--sparse-attention`, `--ctx`, `--max-batch`).
    pub ricursive: catalog::resident_ricursive::RicursiveServeConfig,
    /// Which decision models to serve beyond the ones the model store holds,
    /// and which one `"auto"` resolves to.
    pub decision: catalog::resident_decision::DecisionServeConfig,
    /// The DeepSeek-OCR generations: the model-agnostic policy flags they honour (`--max-batch`, `--ctx`, `--kv-cache`, `--gemm`).
    pub ocr: ocrserve::OcrServeConfig,
    /// Serve training (`brain/train`) beside the models; `None` serves none.
    pub train: Option<TrainKnobs>,
}

/// What `brain serve --training` decides about the training workload.
#[derive(Clone, Debug, Default)]
pub struct TrainKnobs {
    /// The content-addressed store datasets are named in and adapters are published to
    /// (`--artifact-store`); `None` serves a training workload that takes and returns data by descriptor only.
    pub artifact_store: Option<trainjob::ArtifactStore>,
}

/// Build the shared executor with every model registered, sized to the given per-GPU
/// budgets. `gpus` is `(index, total_bytes)` per card; `reserved` bytes are kept free
/// on each. `unified_gpus` names the indices among `gpus` that physically share RAM
/// with the CPU (an integrated GPU, or the no-discrete-GPU fallback) - every NPU
/// always shares RAM too (`docs/reference/models/*`'s Meteor-Lake NPU note) - so those,
/// plus `Device::Cpu`, are declared into ONE `memauth` pool sized to `pool_ram` (the
/// real, physical host RAM - see that parameter's own doc for why this must NEVER be
/// `cpu_ram`); a discrete GPU keeps its own independent budget, unaffected. Falls back
/// gracefully if a heavy model's weights are not configured (it is simply not
/// registered).
///
/// `cpu_ram`/`pool_ram` are deliberately two separate parameters, not one reused
/// value: `cpu_ram` is `Device::Cpu`'s OWN per-device budget (legitimately `0` when
/// `--device` excludes CPU from compute - that alone is what stops the CPU device
/// itself being chosen as a placement target), while `pool_ram` is the physical
/// capacity of the shared pool `Device::Cpu` and every unified GPU/NPU draw from
/// together. Passing the same (possibly zeroed) value for both used to starve the
/// POOL to zero bytes too whenever `--device` excluded CPU - correctly stopping CPU
/// placements, but ALSO clamping every unified GPU/NPU's `usable_on`/`free_on` to
/// `min(real_budget, 0) == 0` (`residency::budget::Budgets::usable_on`'s `pool.min`),
/// making an otherwise-perfectly-placeable model (e.g. a small model on an
/// integrated GPU with `--device gpu`) silently unplaceable forever - no error, just
/// a 10s admission timeout and a generic 429, since a claim failure never fires
/// `on_admit`. The physical RAM does not disappear just because CPU-side compute is
/// disabled, so `pool_ram` must always be the real, ungated host RAM figure.
///
/// `knobs` carries each model's serving knobs as parsed data (the CLI parses
/// flags, this code never reads env): `knobs.qwen` for the Qwen3 paged engine,
/// `knobs.qwen35.gguf` configuring every released Qwen3.8 GGUF the model scan
/// serves through the multi-card INT8 resident (placed on `gpus`, i.e. the
/// cards `--device` allows), `knobs.qwen35.fp32` sizing the brain-checkpoint
/// resident, and `knobs.decision` naming the decision models served beyond
/// those the store holds.
pub fn build_executor(machine: &Machine, reserved: u64, models_dir: Option<&std::path::Path>, policy: Policy, knobs: ServeKnobs) -> Serving {
    let ServeKnobs { qwen: qwen_cfg, qwen35, ricursive, decision, ocr, train } = knobs;
    // Process-wide, before any OCR model is built: the models read it when they load.
    ocrserve::configure(ocr);
    let (gpus, npus, unified_gpus) = (&machine.gpus, &machine.npus, &machine.unified_gpus);
    let (cpu_ram, pool_ram) = (machine.cpu_compute_ram, machine.pool_ram);
    let mut budgets = residency::budget::Budgets::new();
    // The process-wide ceiling (`--limit-vram-total`/`--limit-ram-total`) is
    // applied to EVERY budget below, so the advisory placement layer and the
    // hard enforcement in `gpu_core::Gpu`'s allocation path cannot disagree: a
    // placement is never planned against capacity the ceiling would refuse at
    // the first `storage()` call. `Limits::clamp` is the identity function
    // when no ceiling is set, which is the default.
    let limits = memauth::limits();
    for &(i, total) in gpus {
        budgets.set(Device::Gpu(i), limits.clamp(Device::Gpu(i), total), reserved);
    }
    // NPUs get their own budget + lane; a model advertising an NPU path (MemCost.npu
    // > 0) is then auto-placed there in preference to CPU/GPU (see place::pick_device).
    // An NPU's bytes ARE host RAM, so `--limit-ram-total` is what bounds it.
    for &(i, total) in npus {
        budgets.set(Device::Npu(i), limits.clamp(Device::Npu(i), total), 0);
    }
    budgets.set(Device::Cpu, limits.clamp(Device::Cpu, cpu_ram), 0);
    // The unified-memory fix (see memauth's module doc): declare every device
    // that physically shares this RAM into ONE pool, so a charge on any of
    // them correctly reduces what all the others have free - instead of two
    // (or more) independent budgets that together claim more bytes than the
    // machine has. Sized to `pool_ram`, NEVER `cpu_ram` - see this function's
    // own doc on why those must stay two separate parameters.
    let mut shared: Vec<Device> = vec![Device::Cpu];
    shared.extend(unified_gpus.iter().map(|&i| Device::Gpu(i)));
    shared.extend(npus.iter().map(|&(i, _)| Device::Npu(i)));
    if shared.len() > 1 {
        budgets.set_pool(memauth::HOST_POOL, &shared, limits.clamp(Device::Cpu, pool_ram), 0);
    }

    let mut models: Vec<Arc<dyn ResidentModel>> = Vec::new();
    // z-image (BRAIN_S3DIT_{DIT,VAE,QWEN,TOKENIZER}, else whatever the model
    // store resolves through `s3dit::spec::S3ditSpec` - see `ZImageResident::
    // from_store`'s own doc).
    match ZImageResident::from_store(models_dir) {
        Some(z) => models.push(Arc::new(z)),
        None => residency::log::not_served("brain: z-image not served over the scheduler (set BRAIN_S3DIT_DIT/_VAE/_QWEN/_TOKENIZER, or place a checkpoint under the model dir)"),
    }
    // yolo object detection if a checkpoint is configured (BRAIN_YOLOV8).
    if let Some(y) = YoloResident::from_env() {
        models.push(Arc::new(y));
    } else {
        residency::log::not_served("brain: yolo not served over the scheduler (set BRAIN_YOLOV8 to a checkpoint)");
    }
    // Text-generation LLMs (each gated on its own weights env var).
    if let Some(g) = catalog::resident_llm::GptResident::from_env() {
        models.push(Arc::new(g));
    }
    if let Some(g) = catalog::resident_llm::GlmResident::from_env() {
        models.push(Arc::new(g));
    }
    // The one resident kept BOTH ways: erased for the executor, concrete for
    // the adapter flags' `adapter_view` -- see `Serving::qwen`.
    // `auto_budget_bytes` is filled in HERE, from this call's own real
    // per-card free VRAM (never the caller's business - only this function
    // ever sees `gpus`/`reserved`), regardless of whether `qwen_cfg.ctx`
    // was ALSO given explicitly: `QwenResident::resolve_ctx` checks the
    // explicit value first, so filling this in unconditionally is harmless
    // when it is, and is what makes auto-sizing real when it is not.
    let qwen_cfg = catalog::resident_llm::QwenServeConfig { auto_budget_bytes: gpus.iter().map(|&(_, total)| total.saturating_sub(reserved)).max(), card_budget_bytes: gpus.iter().map(|&(_, total)| total.saturating_sub(reserved)).max(), ..qwen_cfg };
    let mut qwen = register_qwen(&mut models, QwenResident::from_config(qwen_cfg.clone()));
    // Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder
    // (BRAIN_QWEN35MOE_WEIGHTS + BRAIN_QWEN35MOE_TOKENIZER) -- single-GPU,
    // fp32 weights + KV only (see resident_qwen35moe.rs's own module doc for
    // the exact scope vs QwenResident's).
    if let Some(q) = catalog::resident_qwen35moe::Qwen35Resident::from_env() {
        models.push(Arc::new(q));
    }
    // Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder, a brain-format
    // checkpoint resolved through `qwen35::spec::Qwen35Fp32Spec`'s
    // `weights`/`tokenizer` roles -- same single-GPU, fp32 weights + KV scope
    // as qwen35moe above (see resident_qwen35.rs's own module doc). Released
    // GGUFs are not candidates: the model-dir scan serves each as its own
    // multi-card model. Sized from the caller's parsed `fp32` knobs (context,
    // batch ceiling).
    match loader::resolver::try_resolve(models_dir, "qwen35", &qwen35::spec::Qwen35Fp32Spec, &Default::default()) {
        Ok(assembly) => {
            if let Some(why) = qwen35.fp32_refused.as_ref().filter(|_| qwen35.fp32.refused) {
                residency::log::not_served(&format!("brain: qwen35 (brain checkpoint) not served: {why}"));
            } else if let Some(q) = catalog::resident_qwen35::Qwen35Resident::from_assembly_sized(&assembly, qwen35.fp32.sanitized()) {
                models.push(Arc::new(q));
            }
        }
        Err(e) => residency::log::not_served(&format!("brain: qwen35 not served over the scheduler ({e})")),
    }
    // LFM2.5-Encoder (BRAIN_LFM2 + BRAIN_LFM2_TOKENIZER): fill-mask + embeddings
    // with equal-length true batching (see resident_lfm.rs).
    if let Some(l) = catalog::resident_lfm::LfmResident::from_env() {
        models.push(Arc::new(l));
    } else {
        residency::log::not_served("brain: lfm not served over the scheduler (set BRAIN_LFM2 + BRAIN_LFM2_TOKENIZER)");
    }
    // FLUX.2 Klein (BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}, else every
    // independent `dit` checkpoint the model store holds, each its own
    // resident under its real vendor/repo id): text-to-image,
    // reference-image editing, LoRA training (see resident_flux2.rs).
    let flux2_residents = catalog::resident_flux2::Flux2Resident::all_from_store(models_dir);
    if flux2_residents.is_empty() {
        residency::log::not_served("brain: flux2-klein not served over the scheduler (pull a FLUX.2 Klein checkpoint into the model dir)");
    }
    for f in flux2_residents {
        models.push(Arc::new(f));
    }
    // Wan2.1 text-to-video: a resident transformer per (variant, frames,
    // size) - see resident_wan.rs. A pulled checkpoint reaches its adapter
    // through `model_dir::resident_for_compound`'s `"wan"` arm, so whether
    // Wan is missing is decided after the model-dir scan below.
    if let Some(w) = catalog::resident_wan::WanResident::from_env() {
        models.push(Arc::new(w));
    }
    // MiniMax-H3 joint video+audio reaches its adapter through
    // `model_dir::resident_for_compound`'s `"minimaxh3"` arm, from the store.
    // LTX-2.5 video+audio, from every checkpoint the model store resolves,
    // each under its own reference: see resident_ltxv.rs. One with only its
    // VAE runs the random-weight smoke-test DiT and is not offered for sale
    // on /videos.
    let ltxv_residents = catalog::resident_ltxv::LtxvResident::all_from_store(models_dir);
    if ltxv_residents.is_empty() {
        residency::log::not_served("brain: ltxv not served over the scheduler (pull an LTX-2.5 checkpoint into the model dir)");
    }
    for l in ltxv_residents {
        models.push(Arc::new(l));
    }
    // Monocular depth (BRAIN_ZIPDEPTH_WEIGHTS).
    if let Some(d) = catalog::resident_depth::DepthResident::from_env() {
        models.push(Arc::new(d));
    }
    // Imaging models, each gated on its own weights env var: SAM 2.1 promptable
    // segmentation (BRAIN_SAM2_WEIGHTS, prompt-batched per image), the
    // antelopev2 face stack (BRAIN_SCRFD_DIR + BRAIN_ARCFACE_DIR), the VQ autoencoder
    // (BRAIN_VQGAN_WEIGHTS), CodeFormer restoration (BRAIN_CODEFORMER_WEIGHTS) and
    // the CLIP encoders (BRAIN_CLIP_DIR, genuinely batched per tower).
    // The imaging models come from `crate::catalog`, which owns their manifests
    // and providers too - so a model cannot be listed by `brain caps`, runnable
    // by `brain do` and yet missing here (which is exactly how Real-ESRGAN
    // shipped unreachable). Each is still gated on its own weights env var.
    // ... plus, from the same catalog: TTS (BRAIN_QWEN3TTS_WEIGHTS), speech-to-text
    // (BRAIN_NEMOTRONASR + BRAIN_QWEN3ASR), the forecasting foundation models
    // (BRAIN_CHRONOS2 / BRAIN_FINCAST / BRAIN_KRONOS_* - chronos2/fincast
    // advertise an NPU footprint and auto-place there when budgeted), and
    // splat (`brain/splat` - render/fit, `resident_splat.rs`) which, unlike
    // every other entry here, needs no weights at all: the scene arrives as
    // request bytes, so it is always registered, gated on nothing, and
    // worldmirror2 (`brain/worldmirror2` - reconstruct, `BRAIN_WORLDMIRROR2_
    // WEIGHTS`, `resident_worldmirror2.rs`) which, like GLM/qwen3.5 above,
    // takes `weights` as a per-invocation action param on the direct path,
    // so its own `manifest` is weights-free - but whose resident (like
    // splat's) is registered declaratively here rather than by a manual push
    // below, since `resident_worldmirror2.rs` lives in this crate the same
    // reason `resident_splat.rs` does. Folded into `catalog::models()` so
    // `brain caps`/`brain do` and this executor can no longer disagree about
    // their existence.
    models.extend(catalog::residents(models_dir));
    // Decision models (decide + Laya): one resident per checkpoint under its
    // store id, plus the family model that answers inventory and `auto`
    // selection without loading anything.
    models.extend(catalog::resident_decision::serve_all(&decision, models_dir));
    // Deterministic mock model (BRAIN_MOCK): a real ResidentModel - no weights, no
    // GPU - registered as `mock` so the HTTP conformance harness can validate the
    // whole API surface through the true serving path (placement → activate →
    // run_batch). Advertises generate (chat) + embed + text2image; a
    // speech-only sibling `mock-asr` advertises transcribe, and `mock-video`
    // text-to-video and extend.
    if let Some(m) = catalog::resident_mock::MockResident::from_env() {
        models.push(Arc::new(m));
        models.push(Arc::new(catalog::resident_mock::MockAsrResident));
        models.push(Arc::new(catalog::resident_mock_video::MockVideoResident));
    }
    // Stateless helpers (no weights) - always available. `demo` is the worked
    // example every transport smoke (busctl_smoke.sh) exercises.
    models.push(Arc::new(ProviderResident::stateless(Arc::new(catalog::demo::DemoModel))));
    models.push(Arc::new(ProviderResident::stateless(Arc::new(catalog::imageops::ImageOps))));
    // FastVLM captioning: the provider manages its own weight residency
    // (lazy per checkpoint dir, resident thereafter), so it serves as a
    // stateless resident - invoking it with no checkpoint on disk is a clean
    // per-call error, not a registration failure. Served (and validated)
    // under `manifest_resident`, not the raw `manifest`: the raw spec's
    // `weights` param is CLI-only convenience (`brain fastvlm caption
    // --weights ...`) that a scheduled caller must never see or be able to
    // set - see `fastvlm::caps::manifest_resident`'s doc.
    let fastvlm_weights = match loader::resolver::try_resolve(models_dir, "fastvlm", &fastvlm::spec::FastvlmSpec, &Default::default()) {
        Ok(assembly) => assembly.roles.get("weights").map(|p| p.to_string_lossy().into_owned()),
        Err(e) => {
            residency::log::not_served(&format!("brain: fastvlm serving no default checkpoint ({e})"));
            None
        }
    };
    models.push(Arc::new(ProviderResident::stateless_with_manifest(
        Arc::new(fastvlm::caps::FastVlmProvider::new(fastvlm_weights)),
        fastvlm::caps::manifest_resident(),
    )));
    // Moondream 3: `dir` resolved through `moondream3::spec::Moondream3Spec`
    // instead of `BRAIN_MOONDREAM3_WEIGHTS` - registered directly (not through
    // `catalog::residents()`) because `Moondream3Resident::
    // from_assembly` needs the resolved `Assembly` the generic `SingleCtor`
    // shape has no room for, the same reason FLUX.2's own resident is
    // registered directly above rather than through that list.
    match loader::resolver::try_resolve(models_dir, "moondream3", &moondream3::spec::Moondream3Spec, &Default::default()) {
        Ok(assembly) => {
            if let Some(m) = catalog::resident_moondream3::Moondream3Resident::from_assembly(&assembly) {
                models.push(Arc::new(m));
            }
        }
        Err(e) => residency::log::not_served(&format!("brain: moondream3 not served over the scheduler ({e})")),
    }
    // Janus-Pro: registered directly, like Moondream 3, because its plan
    // needs the cards' budgets (see `catalog::resident_januspro`).
    match loader::resolver::try_resolve(models_dir, "januspro", &januspro::spec::JANUS_PRO, &Default::default()) {
        Ok(assembly) => {
            for m in catalog::resident_januspro::JanusProResident::family_from_assembly(&assembly, gpus, reserved) {
                models.push(Arc::new(m));
            }
        }
        Err(e) => residency::log::not_served(&format!("brain: januspro not served over the scheduler ({e})")),
    }
    // LLaVA-1.5-13B captioning: same stateless-resident shape as FastVLM
    // above - the provider manages its own weight residency lazily, per
    // checkpoint dir. Same `manifest_resident` reasoning as FastVLM.
    models.push(Arc::new(ProviderResident::stateless_with_manifest(
        Arc::new(llava::caps::LlavaProvider::new()),
        llava::caps::manifest_resident(),
    )));
    // brain/imgpipe: the pipeline holds no weights of its own (each stage
    // resolves its own through the model store), so it is stateless from the
    // scheduler's point of view too. Its stages come from
    // `catalog::stage_registry`, the SAME registry `brain caps`/`brain do`
    // compose (two lists drifting apart once left `ai-forever/Real-ESRGAN`
    // unreachable over D-Bus/HTTP despite a working `brain do`), resolved
    // against this process's `models_dir` rather than the library default,
    // so the pipeline reads the store every other resident here reads.
    models.push(Arc::new(ProviderResident::stateless(Arc::new(imgpipe::caps::PipelineProvider::new(Arc::new(catalog::stage_registry(models_dir)))))));

    // Global model directory: append every discovered file as its own catalog
    // entry (keyed by model-card id), deduped against the env-gated residents
    // above (their manifest model == id). Additive - the env-gated path stands
    // on its own when no dir is configured or the scan finds nothing. Models
    // that span several cards (released Qwen3.8 GGUFs) are held back for
    // `register_multi` once the executor runs.
    let mut multi_discovered: Vec<Arc<dyn MultiDeviceResidentModel>> = Vec::new();
    let mut qwen35_discovered: Vec<Arc<catalog::resident_qwen35::Qwen35GgufServed>> = Vec::new();
    // The released Qwen3.8 `--checkpoint` named, by served id.
    let mut qwen35_discovered_pick: Option<String> = None;
    let mut ricursive_discovered = Vec::new();
    if let Some(dir) = models_dir {
        let existing: std::collections::BTreeSet<String> = models.iter().map(|m| m.manifest().model).collect();
        let settings = crate::model_dir::ScanSettings { qwen: qwen_cfg, qwen35: qwen35.clone(), ricursive, gpus: gpus.clone(), reserved };
        let discovered = crate::model_dir::discover(dir, &settings);
        // The env-selected resident and a model-dir entry may be the same
        // checkpoint. Two residents would be two instances: two copies of the
        // weights, and base and adapter requests that never share a batch.
        // The model-dir resident keeps the checkpoint; the env id becomes an
        // alias of it, and the adapter flags attach to it.
        if let Some(env) = qwen.clone() {
            if let Some(shared) = discovered.qwen.iter().find(|d| same_checkpoint(d.weights_path(), env.weights_path())) {
                let env_id = env.manifest().model;
                models.retain(|m| m.manifest().model != env_id);
                models.push(Arc::new(residency::Alias::new(env_id, shared.clone())));
                qwen = Some(shared.clone());
            }
        }
        // `--checkpoint` may name a released Qwen3.8 the scan serves as a
        // model of its own. That release is then the checkpoint's one resident:
        // the Qwen3 resident built over the same directory is dropped, and the
        // adapter flags (verified against the host's family) attach to the
        // release, not to a Qwen3 that is not what is served.
        if let Some(env) = qwen.clone() {
            let scanned: Vec<ServedBase> = discovered.qwen35.iter().map(|q| q.served_base()).collect();
            if let Some(i) = checkpoint_base(&env.served_base(), &scanned) {
                let env_id = env.manifest().model;
                models.retain(|m| m.manifest().model != env_id);
                qwen = None;
                qwen35_discovered_pick = Some(discovered.qwen35[i].manifest().model);
            }
        }
        ricursive_discovered = discovered.ricursive.clone();
        for r in discovered.residents {
            let id = r.manifest().model;
            if existing.contains(&id) {
                eprintln!("brain: model dir entry '{id}' shadowed by an env-gated resident; keeping the env one");
                continue;
            }
            models.push(r);
        }
        for m in discovered.multi {
            let id = m.manifest().model;
            if existing.contains(&id) {
                eprintln!("brain: model dir entry '{id}' shadowed by an env-gated resident; keeping the env one");
                continue;
            }
            multi_discovered.push(m);
        }
        for q in discovered.qwen35 {
            // Only the ones that were registered (not shadowed above).
            let id = q.manifest().model;
            if multi_discovered.iter().any(|m| m.manifest().model == id) {
                qwen35_discovered.push(q);
            }
        }
        for e in &discovered.errors {
            eprintln!("brain: {} (family '{}') not registered: {} -- check its brain.manifest.json roles, or re-fetch it", e.dir.display(), e.family, e.reason);
        }
    }
    if !serves_wan(&models) {
        residency::log::not_served("brain: wan not served over the scheduler (pull a Wan2.1 checkpoint into the model dir: brain pull Wan-AI/Wan2.1-T2V-1.3B)");
    }

    // The live capacity probe: the dispatcher re-measures every card's real
    // free VRAM on a cadence, so a neighbouring process taking or releasing
    // bytes changes what this daemon believes it may use - and so a card that
    // frees up is used again without a restart. Same `gpu_core::capacity` probe
    // the one-shot placer uses, which is what stops the two halves of this
    // process disagreeing about the same card.
    let exec = Executor::start_with_probe(
        models,
        budgets,
        policy,
        Some(std::sync::Arc::new(|| {
            gpu_core::capacity::available_gpus().into_iter().map(|(i, free)| (Device::Gpu(i), free)).collect()
        })),
    );
    // Multi-device models the catalog owns (today: DeepSeek-OCR, whose vision
    // tower is on wgpu while its decoder is on the CPU backend). They are
    // registered HERE rather than folded into `models` above because only
    // `register_multi`/`claim_multi` reserve on every device such an instance
    // occupies -- the single-device path's one `budgets.alloc(device, ...)`
    // can charge only one of them, leaving the other silently unbudgeted.
    // `catalog::residents()` deliberately excludes them, so nothing is
    // registered twice.
    for m in catalog::multi_residents(models_dir, gpus, reserved) {
        exec.register_multi(m);
    }
    // Qwen3-Omni (BRAIN_QWEN3OMNIMOE_HF_DIR): the full chat/multimodal surface, placed
    // across as many budgeted cards as its real per-layer bytes need. Like the
    // int8 Thinker below it is multi-device and therefore registered AFTER
    // `start` via `register_multi` -- see resident_omni's module doc for why a
    // plain `register` (which is what it used to take) let it spend VRAM the
    // scheduler had not budgeted.
    if let Some(o) = catalog::resident_omni::OmniResident::from_env(gpus, reserved) {
        exec.register_multi(Arc::new(o));
    }
    // The int8 dual-GPU Thinker is multi-device-only, so it is registered
    // AFTER `start` via `register_multi`, never folded into `models` above
    // (see `resident_omni::int8_thinker_multi_from_env`'s own doc for why a
    // plain `register` would be structurally wrong for it).
    if let Some(t) = catalog::resident_omni::int8_thinker_multi_from_env(gpus, reserved) {
        exec.register_multi(Arc::new(t));
    } else {
        residency::log::not_served(&format!("brain: {} not served over the scheduler (set BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT)", qwen3omnimoe::int8_thinker_resident::MODEL));
    }
    // Released Qwen3.8 GGUFs found by the model scan: INT8 straight from the
    // file, layer-sharded across as many of the allowed cards as their real
    // per-layer bytes need. Multi-device only, so registered here rather than
    // folded into `models` above -- same reason as the two residents just
    // above. Each is a separate model from `brain/qwen35` (the single-GPU fp32
    // brain-checkpoint path), not a mode of it; see
    // `resident_qwen35::gguf_resident`'s own doc.
    for m in multi_discovered {
        exec.register_multi(m);
    }
    if let Some(id) = qwen35_discovered_pick {
        qwen35_discovered.retain(|q| q.manifest().model == id);
    }
    // Training is a workload of this same executor, so it takes cards from the models above and gives them back:
    // see `trainjob::resident` for the policy. One run per card that serving may use.
    if let Some(train) = train {
        let largest = gpus.iter().map(|&(_, total)| total.saturating_sub(reserved)).max();
        exec.register(trainjob::resident(trainjob::TrainConfig { models_dir: models_dir.map(std::path::Path::to_path_buf), store: train.artifact_store, work_root: trainjob::job::default_work_root(), largest_card_bytes: largest, devices: gpus.len().max(1) }));
    }
    Serving { executor: exec, qwen, qwen35: qwen35_discovered, ricursive: ricursive_discovered }
}

/// Whether any of `models` is a Wan resident, under whatever id it registered:
/// every one shares `wan::caps::manifest`'s schema (see `WanResident::manifest`).
fn serves_wan(models: &[Arc<dyn ResidentModel>]) -> bool {
    let wan = wan::caps::manifest().summary;
    models.iter().any(|m| m.manifest().summary == wan)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Wan checkpoint pulled into the model dir registers under its own
    /// reference, not `wan::caps::MODEL`: it still counts as Wan being
    /// served, so `serve` must not warn that Wan is missing.
    #[test]
    fn a_wan_served_from_the_model_dir_counts_as_wan() {
        let paths = wan::Paths { dit: "d".into(), vae: "v".into(), t5: "t".into(), tokenizer: "k".into() };
        let store: Arc<dyn ResidentModel> = Arc::new(catalog::resident_wan::WanResident::from_paths("Wan-AI/Wan2.1-T2V-1.3B".into(), paths));
        let splat: Arc<dyn ResidentModel> = Arc::new(catalog::resident_splat::SplatResident);
        assert!(serves_wan(&[splat.clone(), store]));
        assert!(!serves_wan(&[splat]));
    }

    /// `QwenResident::adapter_view` is INHERENT, not a `ResidentModel`
    /// method, so the erased `Arc<dyn ResidentModel>` the executor holds
    /// cannot call it at all - which is why [`build_executor`] hands the
    /// concrete handle back alongside the executor.
    ///
    /// It must be the SAME object, not a second `QwenResident` built over
    /// the same checkpoint: the adapter registry lives in that object, read
    /// by ITS instance, so an adapter offered on a copy is a model nothing
    /// ever serves.
    #[test]
    fn the_registered_qwen_resident_and_the_concrete_handle_are_one_object() {
        let mut models: Vec<Arc<dyn ResidentModel>> = Vec::new();
        assert!(register_qwen(&mut models, None).is_none(), "no configured qwen weights must yield no handle");
        assert!(models.is_empty(), "and must register nothing");

        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        let resident = catalog::resident_llm::QwenResident::from_card("unused.safetensors", &card, Some("unused.json"));
        let handle = register_qwen(&mut models, Some(resident)).expect("a configured qwen resident yields a concrete handle");
        assert_eq!(models.len(), 1, "the erased handle is registered exactly once");
        assert!(
            std::ptr::eq(Arc::as_ptr(&handle) as *const (), Arc::as_ptr(&models[0]) as *const ()),
            "the concrete handle and the registered erased one must be the same allocation, or an adapter is offered where no instance serves it"
        );
    }

    fn dir_with_config(root: &std::path::Path, name: &str) -> std::path::PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        dir
    }

    /// `--checkpoint` names the base `--adapter` rides on. When it names a
    /// Qwen3.8 release the model scan serves, that release is the host, not a
    /// Qwen3 resident built over the same directory.
    #[test]
    fn a_checkpoint_naming_a_scanned_release_selects_that_release() {
        use catalog::adapter_release::{ServedBase, QWEN35_FAMILY};
        let root = std::env::temp_dir().join(format!("brain-exec-ckpt-{}", std::process::id()));
        let fp8 = dir_with_config(&root, "fp8");
        let other = dir_with_config(&root, "other");
        let scanned = [ServedBase::of_family(QWEN35_FAMILY, other.join("."), Some("v/other".into())), ServedBase::of_family(QWEN35_FAMILY, fp8.join("."), Some("v/fp8".into()))];

        let by_dir = ServedBase::new(&fp8, None);
        assert_eq!(checkpoint_base(&by_dir, &scanned), Some(1), "the same directory, spelled differently");
        let by_id = ServedBase::new(root.join("elsewhere"), Some("v/other".into()));
        assert_eq!(checkpoint_base(&by_id, &scanned), Some(0), "the same store id");
        let unrelated = ServedBase::new(root.join("elsewhere"), Some("v/none".into()));
        assert_eq!(checkpoint_base(&unrelated, &scanned), None, "a checkpoint the scan does not serve selects nothing");
        std::fs::remove_dir_all(&root).ok();
    }
}
