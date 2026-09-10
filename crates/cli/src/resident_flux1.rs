// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.1 behind the residency scheduler.
//!
//! `activate` builds a [`flux1::caps::Session`] rooted at the checkpoint
//! directory; the pipeline itself is built lazily inside it, keyed by the
//! requested `(variant, h, w)`, since `pipeline::Flux1::load` records the
//! DiT's joint-token budget for one size and imports the variant's weights.
//! All of the work comes from `flux1::caps`, so this file holds no second
//! copy of param decoding or the generation call.
//!
//! # No batching - genuinely, not by omission
//!
//! `flux1::caps::Session::text2image` runs a full multi-step diffusion sample
//! per call; there is no batch axis a residency-level grouping could fill
//! without recording a second graph at `b = N`, which `pipeline::Flux1` does
//! not do (the same reasoning `resident_sdxl.rs` documents). `run_batch` is
//! the serial default `Instance` already provides.

use capability::{ActionResult, Invocation, Manifest, Progress};
use flux1::caps::Session;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resolver_cli::RoleEnv;

/// FLUX.1 behind the scheduler. The checkpoint root - a released diffusers
/// FLUX.1 directory holding `transformer/`, `vae/`, `text_encoder/`,
/// `text_encoder_2/`, `tokenizer/`, `tokenizer_2/` - comes from
/// `BRAIN_FLUX1_DIR` when set, and otherwise from the model-store resolver.
pub struct Flux1Resident {
    root: String,
}

impl Flux1Resident {
    /// `BRAIN_FLUX1_DIR` if the operator set it, else whatever the model-store
    /// resolver finds - the same scan, the same `flux1::spec::Flux1Spec`, and
    /// the same candidate rules `brain flux1 …` uses on the one-shot path.
    ///
    /// `None` (not served, never a daemon startup failure) when the store
    /// holds no released FLUX.1 pipeline, or holds more than one and nothing
    /// says which - see `crate::resolver_cli::served_assembly`.
    pub fn from_env() -> Option<Flux1Resident> {
        let assembly = crate::resolver_cli::served_assembly("flux1", &flux1::spec::Flux1Spec, &[RoleEnv { role: "root", var: "BRAIN_FLUX1_DIR" }])?;
        Self::new(assembly.roles.get("root")?.to_string_lossy().into_owned())
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(root: impl Into<String>) -> Option<Flux1Resident> {
        let root = root.into();
        if !std::path::Path::new(&root).join("transformer").exists() {
            eprintln!("brain: flux1 not served ({root} holds no transformer/)");
            return None;
        }
        Some(Flux1Resident { root })
    }
}

impl ResidentModel for Flux1Resident {
    fn manifest(&self) -> Manifest {
        flux1::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One session serves every requested (variant, size): pipelines are
        // built lazily inside it, so splitting the key would duplicate the
        // device handle without saving any weights.
        InstanceKey::new(flux1::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // FLUX.1-dev is ~11.9 B params (~47.6 GB fp32) - even heavier than
        // SDXL's ~14 GB (`resident_sdxl.rs`). A flat bound covering the DiT
        // plus one size's activations and the transient encoder/VAE builds.
        MemCost::new(52u64 << 30, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // `pipeline::Flux1::load` builds its own `Gpu` per (variant, size),
        // lazily, the same as `pipeline::Sdxl::load` - see `resident_sdxl.rs`'s
        // module docs for why every `run` call, not just `activate`, needs to
        // be device-scoped.
        Ok(Box::new(Flux1Instance { session: Session::new(self.root.clone()), device }))
    }
}

/// A resident FLUX.1 session; pipelines for each requested (variant, size)
/// live inside it.
struct Flux1Instance {
    session: Session,
    device: Device,
}

impl Instance for Flux1Instance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        crate::resident_llm::on_device(self.device, || self.session.run(action, inv))?
    }
}
