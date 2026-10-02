// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WorldMirror-2 multi-view 3D reconstruction (`worldmirror2::caps`) behind
//! the residency scheduler.
//!
//! # One resident instance, keyed on checkpoint identity ONLY
//!
//! [`WorldMirror2Resident::instance_key`] ignores the request entirely and
//! always returns the same key: `worldmirror2::model::Mirror` is
//! shape-ADAPTIVE (see that module's own doc), not shape-fixed - it keeps its
//! ~5GB `ParamStore` OUTSIDE its per-shape `Built` buffers, and its `forward`
//! lazily rebuilds `Built` only when the requested `(frames, hp, wp)` differs
//! from what is cached. A Wan-style key fingerprinted on the request shape
//! would therefore duplicate that ~5GB `ParamStore` once per DISTINCT shape a
//! client happens to request, for zero benefit - it would fight `Mirror`'s
//! own internal shape cache instead of using it. Do not "fix" this by adding
//! a shape fingerprint: it is not a gap, it is the design.
//!
//! The honest cost of that choice: `Built` is `Option<Built>`, so exactly ONE
//! shape's buffers are cached at a time. A workload that ALTERNATES between
//! two shapes on this one instance pays a full rebuild on every single call.
//! That is a latency cost, not a correctness bug, and is not something this
//! adapter (or `Mirror` itself) needs to fix - see
//! `crates/worldmirror2/tests/t9_caps_matches_direct.rs`'s `shape_cycle_*`
//! test, which drives exactly that alternating pattern on one instance and
//! checks every result is bit-identical to the equivalent single-shape run
//! (proving correctness survives the rebuild-churn, not that the churn is
//! free).
//!
//! # Serving contract
//!
//! `manifest()` is `worldmirror2::caps::manifest_resident()`: `weights`
//! carries [`capability::ParamSpec::host_env`]
//! (`BRAIN_WORLDMIRROR2_WEIGHTS`), so the served action never advertises it -
//! a remote caller cannot answer "where is the checkpoint on THIS machine".
//! `Instance::run` therefore calls
//! [`worldmirror2::caps::run_reconstruct`] directly (the `Session`
//! `activate` already built), never through `worldmirror2::caps::Provider`'s
//! own `Action` - that layer's `weights` resolution is for the DIRECT
//! (`brain do worldmirror2 reconstruct --weights …`) path only. Same shape as
//! `sam2::caps::Session::segment` being called straight from
//! `crate::resident_sam2::Sam2Instance`.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use worldmirror2::caps::Session;

/// WorldMirror-2 behind the scheduler (`BRAIN_WORLDMIRROR2_WEIGHTS`).
pub struct WorldMirror2Resident {
    path: String,
}

impl WorldMirror2Resident {
    /// `None` when the checkpoint is unset or absent - registering a model
    /// whose every call would fail is worse than not serving it.
    pub fn from_env() -> Option<WorldMirror2Resident> {
        let path = std::env::var("BRAIN_WORLDMIRROR2_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        if !std::path::Path::new(&path).exists() {
            eprintln!("brain: worldmirror2 not served ({path} does not exist)");
            return None;
        }
        Some(WorldMirror2Resident { path })
    }
}

impl ResidentModel for WorldMirror2Resident {
    fn manifest(&self) -> Manifest {
        worldmirror2::caps::manifest_resident()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // See the module doc: the checkpoint path is the WHOLE key, deliberately.
        InstanceKey::new(worldmirror2::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // The imported fp32 tensors go into a `ParamStore` on the device (the
        // reference checkpoint is ~5GB), so `file*12/10` (the same margin
        // `resident_sam2.rs::Sam2Resident::estimate` uses) covers the weight
        // footprint. On top of that: 24 trunk levels of frame+global
        // attention, each holding `scores`/`probs` slabs sized off
        // `s * (7+hp*wp)` (see `model.rs::Built`'s scratch), are genuinely not
        // negligible against the weights for more than a couple of frames -
        // so, following `resident_sam2.rs`'s own reasoning (a bare
        // zero-activation estimate undercounts a many-block attention
        // stack), a flat activation slack is added rather than reported as
        // zero. Sized for a modest multi-frame request (a handful of views at
        // moderate resolution); a much larger one still allocates correctly,
        // it is simply under-budgeted here.
        let file = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        let activations: u64 = 2u64 << 30; // 2 GiB
        MemCost::new(file.saturating_mul(12) / 10 + activations, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(worldmirror2::model::PIPELINES))?;
        let session = worldmirror2::caps::load(&self.path, gpu)?;
        Ok(Box::new(WorldMirror2Instance { session, device }))
    }
}

/// A resident WorldMirror-2: the built [`Session`] plus the scheduler-
/// assigned device its GPU work must land on.
struct WorldMirror2Instance {
    session: Session,
    device: Device,
}

impl Instance for WorldMirror2Instance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "reconstruct" {
            return Err(format!("worldmirror2: unknown action '{action}'"));
        }
        let session = &mut self.session;
        crate::resident_llm::on_device(self.device, || worldmirror2::caps::run_reconstruct(session, inv))?
    }
}
