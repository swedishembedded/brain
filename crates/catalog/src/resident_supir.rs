// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SUPIR behind the residency scheduler.
//!
//! `activate` builds a [`supir::caps::Session`] rooted at the SDXL backbone
//! (`sdxl`, the same directory `resident_sdxl.rs`/`resident_controlnet.rs`
//! load) and the SUPIR delta checkpoint (`supir`), both resolved through
//! `supir::spec::SupirSpec`; the
//! restoration graph is built lazily inside it, keyed by `(pixel size,
//! control_scale)` (see `supir::caps::Session`'s own doc for why - the graph
//! bakes `control_scale` in). All of the work comes from `supir::caps`, so
//! this file holds no second copy of param decoding, the resize/dual-encode,
//! or the sampler loop.
//!
//! # LLaVA auto-captioning
//!
//! `crates/supir` links no VLM (see `supir::caps`'s own doc): the optional
//! auto-caption call goes through a `capability::Registry` this file builds
//! itself, carrying `llava::caps::LlavaProvider` under `supir::caps::LLAVA_MODEL`
//! over the LLaVA checkpoint the same store resolves (`llava::spec::LlavaSpec`) -
//! this crate depends on both, unlike `crates/supir`. The catalog's own SUPIR
//! entry builds an equivalent registry for the direct `brain do`/D-Bus-via-provider
//! path (`supir_registry` in `lib.rs`).
//!
//! # No batching
//!
//! Same reasoning as `resident_sdxl.rs`/`resident_controlnet.rs`:
//! `supir::pipeline::Restorer::restore` is a full multi-step diffusion sample
//! per call - there is no batch axis a residency-level grouping could fill.
//! `run_batch` is the serial default `Instance` already provides.

use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Progress, Registry};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use supir::caps::Session;

/// SUPIR behind the scheduler over the SDXL backbone (same layout
/// `resident_sdxl.rs` uses) and the SUPIR delta checkpoint.
pub struct SupirResident {
    sdxl_root: String,
    supir_ckpt: String,
    /// The captioner's checkpoint; `None` leaves auto-captioning to fail
    /// per call with llava's own "no checkpoint" error.
    llava_weights: Option<String>,
}

impl SupirResident {
    /// Both checkpoints `store` resolves (or pins) for `supir`, plus the
    /// optional LLaVA captioner; `None` unless both SUPIR checkpoints are
    /// there and the backbone holds a released `unet/` - registering a model
    /// whose every call would fail is worse than not serving it.
    pub fn from_store(store: &loader::served::ServedStore) -> Option<SupirResident> {
        let assembly = loader::served::served_assembly(store, "supir", &supir::spec::SupirSpec)?;
        let paths = supir::pipeline::Paths::from_assembly(&assembly).map_err(|e| residency::log::not_served(&format!("brain: supir not served ({e})"))).ok()?;
        if !std::path::Path::new(&paths.backbone_root).join("unet").exists() {
            residency::log::not_served(&format!("brain: supir not served ({} holds no unet/)", paths.backbone_root));
            return None;
        }
        let llava_weights = store.resolve("llava", &llava::spec::LlavaSpec).ok().and_then(|a| a.roles.get("weights").map(|p| p.to_string_lossy().into_owned()));
        Some(SupirResident { sdxl_root: paths.backbone_root, supir_ckpt: paths.supir_ckpt, llava_weights })
    }
}

/// The registry SUPIR's `caption` auto-fill dispatches through, over the
/// residency path - see this module's doc.
fn llava_registry(llava_weights: Option<String>) -> Registry {
    let mut reg = Registry::new();
    reg.register(Arc::new(llava::caps::LlavaProvider::new(llava_weights)));
    reg
}

impl ResidentModel for SupirResident {
    fn manifest(&self) -> Manifest {
        supir::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One session serves every requested size/control_scale: graphs are
        // built lazily inside it, so splitting the key would duplicate the
        // device handle without saving any weights.
        InstanceKey::new(supir::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // The combined trunk (1.24B, ~5.0 GB fp32 - AGENTS.md 12f) + frozen
        // SDXL backbone (~3.5B, ~14 GB) + adaptors (54.8M, ~0.22 GB) +
        // denoise_encoder (34.16M, ~0.14 GB) plus per-size activations and
        // the transient CLIP/VAE encoder-decoder builds - the same flat,
        // documented-estimate shape `resident_controlnet.rs`'s own
        // `estimate()` uses for its backbone+trunk pair.
        MemCost::new(24u64 << 30, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(SupirInstance { session: Session::new(self.sdxl_root.clone(), self.supir_ckpt.clone()), registry: Arc::new(llava_registry(self.llava_weights.clone())), device }))
    }
}

/// A resident SUPIR session; restoration graphs for each requested size live
/// inside it.
struct SupirInstance {
    session: Session,
    registry: Arc<Registry>,
    device: Device,
}

impl Instance for SupirInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let registry = self.registry.clone();
        crate::resident_llm::on_device(self.device, || self.session.run(action, inv, Some(&registry), progress))?
    }
}
