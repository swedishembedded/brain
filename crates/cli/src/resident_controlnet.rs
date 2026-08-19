// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL + ControlNet behind the residency scheduler.
//!
//! `activate` builds a [`controlnet::caps::Session`] rooted at the SDXL
//! backbone and ControlNet checkpoint directories; the (Unet, ControlNet)
//! pair is built lazily inside it, keyed by the requested `(h, w)`, the same
//! shape `resident_sdxl.rs` uses. All of the work comes from
//! `controlnet::caps`, so this file holds no second copy of param decoding,
//! the conditioning-image resize, or the generation call.
//!
//! # No batching
//!
//! Same reasoning as `resident_sdxl.rs`: `controlnet::caps::Session::run`'s
//! `text2image` is a full multi-step diffusion sample per call, now with a
//! ControlNet evaluation on top at every step - there is no batch axis a
//! residency-level grouping could fill. `run_batch` is the serial default
//! `Instance` already provides.

use capability::{ActionResult, Invocation, Manifest, Progress};
use controlnet::caps::Session;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// SDXL + ControlNet behind the scheduler (`BRAIN_SDXL_DIR` - the backbone,
/// same layout `resident_sdxl.rs` uses; `BRAIN_CONTROLNET_DIR` - a released
/// diffusers SDXL `ControlNetModel` checkpoint).
pub struct ControlnetResident {
    sdxl_root: String,
    control_root: String,
}

impl ControlnetResident {
    /// `None` unless both directories are set and the backbone holds a
    /// released `unet/` - registering a model whose every call would fail is
    /// worse than not serving it.
    pub fn from_env() -> Option<ControlnetResident> {
        let sdxl_root = std::env::var("BRAIN_SDXL_DIR").ok().filter(|p| !p.is_empty())?;
        let control_root = std::env::var("BRAIN_CONTROLNET_DIR").ok().filter(|p| !p.is_empty())?;
        Self::new(sdxl_root, control_root)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(sdxl_root: impl Into<String>, control_root: impl Into<String>) -> Option<ControlnetResident> {
        let sdxl_root = sdxl_root.into();
        if !std::path::Path::new(&sdxl_root).join("unet").exists() {
            eprintln!("brain: sdxl-controlnet not served ({sdxl_root} holds no unet/)");
            return None;
        }
        Some(ControlnetResident { sdxl_root, control_root: control_root.into() })
    }
}

impl ResidentModel for ControlnetResident {
    fn manifest(&self) -> Manifest {
        controlnet::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One session serves every requested size: pipelines are built lazily
        // inside it, so splitting the key would duplicate the device handle
        // without saving any weights.
        InstanceKey::new(controlnet::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // SDXL's ~14 GB (see `resident_sdxl.rs`) plus a ControlNet copy of the
        // backbone's early blocks - 1.25B params = 5.00 GB fp32
        // (AGENTS.md 12f) - plus its conditioning embedder and per-size
        // activations for both models.
        MemCost::new(22u64 << 30, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // `controlnet::caps::Controlled::load` builds its own `Gpu` per size,
        // the same as `pipeline::Sdxl::load` - see `resident_sdxl.rs`'s
        // module docs for why every `run` call, not `activate`, is what needs
        // to be device-scoped.
        Ok(Box::new(ControlnetInstance {
            session: Session::new(self.sdxl_root.clone(), self.control_root.clone()),
            device,
        }))
    }
}

/// A resident SDXL+ControlNet session; pipelines for each requested size live
/// inside it.
struct ControlnetInstance {
    session: Session,
    device: Device,
}

impl Instance for ControlnetInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        crate::resident_llm::on_device(self.device, || self.session.run(action, inv))?
    }
}
