// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PuLID-conditioned FLUX.1 behind the residency scheduler.
//!
//! `activate` builds a [`pulid::caps::Session`] rooted at the four checkpoint
//! directories `pulid::caps::PulidProvider::from_env` requires; each
//! (variant, size) bundle - FLUX.1, ArcFace, EVA-CLIP, IDFormer, the
//! `PulidAdapter` - is built lazily inside it. All of the work comes from
//! `pulid::caps`, so this file holds no second copy of param decoding, the
//! ID-conditioning composition, or the generation call.
//!
//! # No batching, same reasoning as `resident_flux1.rs`
//!
//! Every request is its own multi-step sample with an ID-conditioned DiT -
//! there is no batch axis a residency-level grouping could fill.

use capability::{ActionResult, Invocation, Manifest, Progress};
use pulid::caps::Session;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// PuLID-conditioned FLUX.1 behind the scheduler. Four directories, all
/// required: `BRAIN_FLUX1_DIR` (the backbone, same as `resident_flux1.rs`),
/// `BRAIN_PULID_DIR` (`pulid_flux_v0.9.1.safetensors` or its directory),
/// `BRAIN_ARCFACE_DIR` (same as `resident_arcface.rs`), `BRAIN_CLIP_DIR` (for
/// the EVA-CLIP-L/336 file, same as `resident_clip.rs`).
pub struct PulidResident {
    flux1_root: String,
    pulid_root: String,
    arcface_root: String,
    clip_root: String,
}

impl PulidResident {
    /// `None` unless every directory is set and the FLUX.1 root holds a
    /// released `transformer/` - registering a model whose every call would
    /// fail is worse than not serving it.
    pub fn from_env() -> Option<PulidResident> {
        let get = |k: &str| std::env::var(k).ok().filter(|p| !p.is_empty());
        let (flux1_root, pulid_root, arcface_root, clip_root) =
            (get("BRAIN_FLUX1_DIR")?, get("BRAIN_PULID_DIR")?, get("BRAIN_ARCFACE_DIR")?, get("BRAIN_CLIP_DIR")?);
        Self::new(flux1_root, pulid_root, arcface_root, clip_root)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(
        flux1_root: impl Into<String>,
        pulid_root: impl Into<String>,
        arcface_root: impl Into<String>,
        clip_root: impl Into<String>,
    ) -> Option<PulidResident> {
        let flux1_root = flux1_root.into();
        if !std::path::Path::new(&flux1_root).join("transformer").exists() {
            eprintln!("brain: flux1-pulid not served ({flux1_root} holds no transformer/)");
            return None;
        }
        Some(PulidResident { flux1_root, pulid_root: pulid_root.into(), arcface_root: arcface_root.into(), clip_root: clip_root.into() })
    }
}

impl ResidentModel for PulidResident {
    fn manifest(&self) -> Manifest {
        pulid::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(pulid::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // FLUX.1-dev's ~52 GB (`resident_flux1.rs`) plus ArcFace + EVA-CLIP-L
        // (~1 GB combined, similar order to `resident_clip.rs`'s image tower)
        // plus PuLID's own IDFormer/PulidCa (~140 M params, a few hundred MB).
        MemCost::new(54u64 << 30, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Every model `pulid::caps::Bundle::load` builds constructs its own
        // `Gpu` lazily (the same shape `resident_flux1.rs`/`resident_sdxl.rs`
        // document) - every `run` call, not just `activate`, is device-scoped.
        Ok(Box::new(PulidInstance {
            session: Session::new(self.flux1_root.clone(), self.pulid_root.clone(), self.arcface_root.clone(), self.clip_root.clone()),
            device,
        }))
    }
}

struct PulidInstance {
    session: Session,
    device: Device,
}

impl Instance for PulidInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        crate::resident_llm::on_device(self.device, || self.session.run(action, inv))?
    }
}
