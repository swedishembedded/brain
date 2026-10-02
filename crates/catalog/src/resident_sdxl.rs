// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL behind the residency scheduler.
//!
//! `activate` builds a [`sdxlunet::caps::Session`] rooted at the checkpoint
//! directory; the pipeline itself is built lazily inside it, keyed by the
//! requested `(h, w)`, since `pipeline::Sdxl::load` records the UNet graph at
//! one size. All of the work comes from `sdxlunet::caps`, so this file holds
//! no second copy of param decoding or the generation call.
//!
//! # No batching - genuinely, not by omission
//!
//! `sdxlunet::caps::Session::text2image` runs a full multi-step diffusion
//! sample per call; there is no batch axis a residency-level grouping could
//! fill without recording a second graph at `b = N`, which `pipeline::Sdxl`
//! does not do. `run_batch` is the serial default `Instance` already provides.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use sdxlunet::caps::Session;

/// SDXL behind the scheduler (`BRAIN_SDXL_DIR` - a released diffusers SDXL
/// checkpoint root holding `unet/`, `vae/`, `text_encoder/`, `text_encoder_2/`,
/// `tokenizer/`, `tokenizer_2/`).
pub struct SdxlResident {
    root: String,
}

impl SdxlResident {
    /// `None` when the directory is unset or holds no `unet/` - registering a
    /// model whose every call would fail is worse than not serving it.
    pub fn from_env() -> Option<SdxlResident> {
        Self::new(std::env::var("BRAIN_SDXL_DIR").ok().filter(|p| !p.is_empty())?)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(root: impl Into<String>) -> Option<SdxlResident> {
        let root = root.into();
        if !std::path::Path::new(&root).join("unet").exists() {
            eprintln!("brain: sdxl not served ({root} holds no unet/)");
            return None;
        }
        Some(SdxlResident { root })
    }
}

impl ResidentModel for SdxlResident {
    fn manifest(&self) -> Manifest {
        sdxlunet::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One session serves every requested size: pipelines are built lazily
        // inside it, so splitting the key would duplicate the device handle
        // without saving any weights.
        InstanceKey::new(sdxlunet::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // SDXL is ~3.5 B params across UNet + both CLIP towers + VAE, ~14 GB
        // at fp32 - see `pipeline::Sdxl`'s doc for why only the UNet stays
        // resident. A flat bound covering the UNet plus one size's activations
        // and the transient encoder/VAE builds.
        MemCost::new(16u64 << 30, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // `pipeline::Sdxl::load` builds its own `Gpu` PER SIZE, lazily, inside
        // `Session::text2image` on a cache miss - unlike `resident_clip`'s
        // session, it takes no external `Gpu` at construction. So the device
        // assignment cannot be fixed once here; every `run` call is wrapped in
        // `on_device` instead, which every `Gpu::new` triggered during it
        // (however deep) binds to.
        Ok(Box::new(SdxlInstance { session: Session::new(self.root.clone()), device }))
    }
}

/// A resident SDXL session; pipelines for each requested size live inside it.
struct SdxlInstance {
    session: Session,
    device: Device,
}

impl Instance for SdxlInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        crate::resident_llm::on_device(self.device, || self.session.run(action, inv))?
    }
}
