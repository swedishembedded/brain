// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SUPIR behind the residency scheduler.
//!
//! `activate` builds a [`supir::caps::Session`] rooted at the SDXL backbone
//! (`BRAIN_SDXL_DIR`, same directory `resident_sdxl.rs`/`resident_controlnet.rs`
//! already load) and the SUPIR delta checkpoint (`BRAIN_SUPIR_DIR`); the
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
//! - `brain-cli` sits at the top of the crate-graph layering and may depend on
//! both, unlike `crates/supir`. `crates/catalog`'s own SUPIR entry builds an
//! equivalent registry for the direct `brain do`/D-Bus-via-provider path (see
//! that crate's `supir_registry`), so the two callers agree without sharing code
//! neither can reach.
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

/// SUPIR behind the scheduler (`BRAIN_SDXL_DIR` - the backbone, same layout
/// `resident_sdxl.rs` uses; `BRAIN_SUPIR_DIR` - a SUPIR delta checkpoint,
/// file or directory).
pub struct SupirResident {
    sdxl_root: String,
    supir_ckpt: String,
}

impl SupirResident {
    /// `None` unless both are set and the backbone holds a released `unet/`
    /// - registering a model whose every call would fail is worse than not
    /// serving it.
    pub fn from_env() -> Option<SupirResident> {
        let sdxl_root = std::env::var("BRAIN_SDXL_DIR").ok().filter(|p| !p.is_empty())?;
        let supir_dir = std::env::var("BRAIN_SUPIR_DIR").ok().filter(|p| !p.is_empty())?;
        Self::new(sdxl_root, supir_dir)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(sdxl_root: impl Into<String>, supir_ckpt: impl Into<String>) -> Option<SupirResident> {
        let sdxl_root = sdxl_root.into();
        if !std::path::Path::new(&sdxl_root).join("unet").exists() {
            eprintln!("brain: supir not served ({sdxl_root} holds no unet/)");
            return None;
        }
        Some(SupirResident { sdxl_root, supir_ckpt: supir_ckpt.into() })
    }
}

/// The registry SUPIR's `caption` auto-fill dispatches through, over the
/// residency path - see this module's doc.
fn llava_registry() -> Registry {
    let mut reg = Registry::new();
    reg.register(Arc::new(llava::caps::LlavaProvider::new()));
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
        Ok(Box::new(SupirInstance { session: Session::new(self.sdxl_root.clone(), self.supir_ckpt.clone()), registry: Arc::new(llava_registry()), device }))
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
