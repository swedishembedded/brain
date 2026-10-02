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

use catalog::resident_llm::QwenResident;
use catalog::resident_yolo::YoloResident;
use catalog::resident_zimage::ZImageResident;
use residency::bridge::ProviderResident;
use residency::{Device, Executor, Policy, ResidentModel};

use crate::Machine;


/// The one shared serving [`Executor`], plus the CONCRETE model handles a
/// caller needs for an inherent method the erased `Arc<dyn ResidentModel>`
/// does not carry.
///
/// Today that is exactly one: [`QwenResident::set_adapter`], the write half
/// of the continuous-learning hot swap (the CLI's `continuous_train`). It is
/// inherent rather than a `ResidentModel` method because "point yourself at
/// a new LoRA adapter" is not a concept the trait has - putting it there
/// would put a decoder-LM-shaped method on every detector, VAE and
/// forecaster that implements it - so the erased handle the executor holds
/// cannot call it at all.
///
/// The handle here is the SAME `Arc` that was registered, never a second
/// `QwenResident` over the same checkpoint: `set_adapter` writes into that
/// object's own `RwLock`, read by ITS `activate`, so a write to a copy is a
/// hot swap that silently does nothing.
pub struct Serving {
    pub executor: Executor,
    /// `None` when `BRAIN_QWEN_WEIGHTS` names no checkpoint, i.e. this
    /// process serves no Qwen3 and there is nothing to hot-swap.
    pub qwen: Option<Arc<QwenResident>>,
}

/// Register `qwen` (when its weights are configured) as an ordinary erased
/// resident AND keep the concrete handle - see [`Serving::qwen`] for why
/// both, and why they must be one allocation.
fn register_qwen(models: &mut Vec<Arc<dyn ResidentModel>>, qwen: Option<QwenResident>) -> Option<Arc<QwenResident>> {
    let q = Arc::new(qwen?);
    models.push(q.clone());
    Some(q)
}

/// Build the shared executor with every model registered, sized to the given per-GPU
/// budgets. `gpus` is `(index, total_bytes)` per card; `reserved` bytes are kept free
/// on each. `unified_gpus` names the indices among `gpus` that physically share RAM
/// with the CPU (an integrated GPU, or the no-discrete-GPU fallback) - every NPU
/// always shares RAM too (`docs/models/*`'s Meteor-Lake NPU note) - so those,
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
pub fn build_executor(machine: &Machine, reserved: u64, models_dir: Option<&std::path::Path>, policy: Policy, qwen_cfg: catalog::resident_llm::QwenServeConfig) -> Serving {
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
        None => eprintln!("brain: z-image not served over the scheduler (set BRAIN_S3DIT_DIT/_VAE/_QWEN/_TOKENIZER, or place a checkpoint under the model dir)"),
    }
    // yolo object detection if a checkpoint is configured (BRAIN_YOLOV8).
    if let Some(y) = YoloResident::from_env() {
        models.push(Arc::new(y));
    } else {
        eprintln!("brain: yolo not served over the scheduler (set BRAIN_YOLOV8 to a checkpoint)");
    }
    // Text-generation LLMs (each gated on its own weights env var).
    if let Some(g) = catalog::resident_llm::GptResident::from_env() {
        models.push(Arc::new(g));
    }
    if let Some(g) = catalog::resident_llm::GlmResident::from_env() {
        models.push(Arc::new(g));
    }
    // The one resident kept BOTH ways: erased for the executor, concrete for
    // the hot-swap path's inherent `set_adapter` -- see `Serving::qwen`.
    // `auto_budget_bytes` is filled in HERE, from this call's own real
    // per-card free VRAM (never the caller's business - only this function
    // ever sees `gpus`/`reserved`), regardless of whether `qwen_cfg.ctx`
    // was ALSO given explicitly: `QwenResident::resolve_ctx` checks the
    // explicit value first, so filling this in unconditionally is harmless
    // when it is, and is what makes auto-sizing real when it is not.
    let qwen_cfg = catalog::resident_llm::QwenServeConfig { auto_budget_bytes: gpus.iter().map(|&(_, total)| total.saturating_sub(reserved)).max(), ..qwen_cfg };
    let qwen = register_qwen(&mut models, QwenResident::from_env(qwen_cfg));
    // Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder
    // (BRAIN_QWEN35MOE_WEIGHTS + BRAIN_QWEN35MOE_TOKENIZER) -- single-GPU,
    // fp32 weights + KV only (see resident_qwen35moe.rs's own module doc for
    // the exact scope vs QwenResident's).
    if let Some(q) = catalog::resident_qwen35moe::Qwen35Resident::from_env() {
        models.push(Arc::new(q));
    }
    // Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder, resolved through
    // `qwen35::spec::Qwen35Spec`'s `weights`/`tokenizer` roles instead of
    // `BRAIN_QWEN35_{WEIGHTS,TOKENIZER}` -- same single-GPU, fp32 weights +
    // KV scope as qwen35moe above (see resident_qwen35.rs's own module doc).
    match loader::resolver::try_resolve(models_dir, "qwen35", &qwen35::spec::Qwen35Spec, &Default::default()) {
        Ok(assembly) => {
            if let Some(q) = catalog::resident_qwen35::Qwen35Resident::from_assembly(&assembly) {
                models.push(Arc::new(q));
            }
        }
        Err(e) => eprintln!("brain: qwen35 not served over the scheduler ({e})"),
    }
    // LFM2.5-Encoder (BRAIN_LFM2 + BRAIN_LFM2_TOKENIZER): fill-mask + embeddings
    // with equal-length true batching (see resident_lfm.rs).
    if let Some(l) = catalog::resident_lfm::LfmResident::from_env() {
        models.push(Arc::new(l));
    } else {
        eprintln!("brain: lfm not served over the scheduler (set BRAIN_LFM2 + BRAIN_LFM2_TOKENIZER)");
    }
    // FLUX.2 Klein (BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}, else every
    // independent `dit` checkpoint the model store holds, each its own
    // resident under its real vendor/repo id): text-to-image,
    // reference-image editing, LoRA training (see resident_flux2.rs).
    let flux2_residents = catalog::resident_flux2::Flux2Resident::all_from_store(models_dir);
    if flux2_residents.is_empty() {
        eprintln!("brain: flux2-klein not served over the scheduler (set BRAIN_FLUX2_DIT/_VAE/_TE/_TOKENIZER, or place a checkpoint under the model dir)");
    }
    for f in flux2_residents {
        models.push(Arc::new(f));
    }
    // Wan2.1 text-to-video (BRAIN_WAN_{DIT,VAE,T5,TOKENIZER}): a resident
    // transformer per (variant, frames, size) - see resident_wan.rs.
    if let Some(w) = catalog::resident_wan::WanResident::from_env() {
        models.push(Arc::new(w));
    } else {
        eprintln!("brain: wan not served over the scheduler (set BRAIN_WAN_DIT/_VAE/_T5/_TOKENIZER)");
    }
    // LTX-2.5 text-to-video (BRAIN_LTXV_VAE): a smoke-test pipeline (real VAE,
    // tiny random-weight DiT, no real text encoder yet) with nothing worth
    // caching resident - see resident_ltxv.rs's module doc.
    if let Some(l) = catalog::resident_ltxv::LtxvResident::from_env() {
        models.push(Arc::new(l));
    } else {
        eprintln!("brain: ltxv not served over the scheduler (set BRAIN_LTXV_VAE)");
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
    // Deterministic mock model (BRAIN_MOCK): a real ResidentModel - no weights, no
    // GPU - registered as `mock` so the HTTP conformance harness can validate the
    // whole API surface through the true serving path (placement → activate →
    // run_batch). Advertises generate (chat) + embed + text2image.
    if let Some(m) = catalog::resident_mock::MockResident::from_env() {
        models.push(Arc::new(m));
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
            eprintln!("brain: fastvlm serving no default checkpoint ({e})");
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
        Err(e) => eprintln!("brain: moondream3 not served over the scheduler ({e})"),
    }
    // Janus-Pro: registered directly, like Moondream 3, because its plan
    // needs the cards' budgets (see `catalog::resident_januspro`).
    match loader::resolver::try_resolve(models_dir, "januspro", &januspro::spec::JANUS_PRO, &Default::default()) {
        Ok(assembly) => {
            for m in catalog::resident_januspro::JanusProResident::family_from_assembly(&assembly, gpus, reserved) {
                models.push(Arc::new(m));
            }
        }
        Err(e) => eprintln!("brain: januspro not served over the scheduler ({e})"),
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
    // on its own when no dir is configured or the scan finds nothing.
    if let Some(dir) = models_dir {
        let existing: std::collections::BTreeSet<String> = models.iter().map(|m| m.manifest().model).collect();
        let (discovered, errors) = crate::model_dir::discover(dir, qwen_cfg);
        for r in discovered {
            let id = r.manifest().model;
            if existing.contains(&id) {
                eprintln!("brain: model dir entry '{id}' shadowed by an env-gated resident; keeping the env one");
                continue;
            }
            models.push(r);
        }
        for e in &errors {
            eprintln!("brain: {} (family '{}') not registered: {} -- check its brain.manifest.json roles, or re-fetch it", e.dir.display(), e.family, e.reason);
        }
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
        eprintln!("brain: {} not served over the scheduler (set BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT)", qwen3omnimoe::int8_thinker_resident::MODEL);
    }
    // Qwen3.8-27B straight from its released Q8_0 GGUF, INT8 and layer-sharded
    // across as many cards as its real per-layer bytes need. Multi-device only,
    // so registered here rather than folded into `models` above -- same reason
    // as the two residents just above it. A separate model from `brain/qwen35`
    // (the single-GPU fp32 brain-checkpoint path), not a mode of it; see
    // `resident_qwen35::multi_gpu_gguf_from_env`'s own doc.
    if let Some(q) = catalog::resident_qwen35::multi_gpu_gguf_from_env(models_dir, gpus, reserved) {
        exec.register_multi(Arc::new(q));
    }
    Serving { executor: exec, qwen }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `QwenResident::set_adapter` is INHERENT, not a `ResidentModel`
    /// method, so the erased `Arc<dyn ResidentModel>` the executor holds
    /// cannot call it at all - which is why [`build_executor`] hands the
    /// concrete handle back alongside the executor.
    ///
    /// It must be the SAME object, not a second `QwenResident` built over
    /// the same checkpoint: the adapter path lives in that object's own
    /// `RwLock`, read by ITS `activate`, so a write to a copy is a write
    /// nothing ever reads - a hot swap that silently does nothing.
    #[test]
    fn the_registered_qwen_resident_and_the_concrete_handle_are_one_object() {
        let mut models: Vec<Arc<dyn ResidentModel>> = Vec::new();
        assert!(register_qwen(&mut models, None).is_none(), "no configured qwen weights must yield no handle");
        assert!(models.is_empty(), "and must register nothing");

        let card = checkpoint::st::ModelCard::new("brain/qwen3", "qwen");
        let resident = catalog::resident_llm::QwenResident::from_card("unused.safetensors", &card, Some("unused.json"), None);
        let handle = register_qwen(&mut models, Some(resident)).expect("a configured qwen resident yields a concrete handle");
        assert_eq!(models.len(), 1, "the erased handle is registered exactly once");
        assert!(
            std::ptr::eq(Arc::as_ptr(&handle) as *const (), Arc::as_ptr(&models[0]) as *const ()),
            "the concrete handle and the registered erased one must be the same allocation, or set_adapter writes where no activate reads"
        );
    }
}
