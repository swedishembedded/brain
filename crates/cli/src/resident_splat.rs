// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3D Gaussian Splatting (`splat::caps`) behind the residency scheduler.
//!
//! Unlike every other resident adapter in this file's siblings, splat has NO
//! fixed checkpoint: the scene arrives as request bytes (`render`'s `scene`
//! input, `fit`'s `scene`+`video`), so there is nothing to key an
//! [`InstanceKey`] on and nothing for `activate` to import ahead of a call -
//! `render`'s own per-request cache inside `splat::caps::SplatProvider`
//! (keyed on a digest of the scene bytes + render size, see that module's
//! doc) does the work a fixed checkpoint's `activate` would otherwise do
//! once; `fit` rebuilds fresh every call (it is streaming and rarely
//! repeated with identical inputs, same as `wan`/`flux2`'s training actions).
//! So this adapter always serves one shared instance (like
//! `residency::bridge::ProviderResident`'s stateless case), but as a bespoke
//! type rather than that generic bridge: `render`'s GPU build must land on
//! the scheduler's ASSIGNED device (`resident_llm::on_device`), which the
//! stateless bridge has no hook for.

use capability::{ActionResult, Invocation, Manifest, Progress, Provider};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use splat::caps::SplatProvider;

/// Splat's render/fit behind the scheduler. Always available - it needs no
/// weights (per-request scene bytes ARE the model).
pub struct SplatResident;

impl SplatResident {
    pub fn from_env() -> Option<SplatResident> {
        Some(SplatResident)
    }
}

impl ResidentModel for SplatResident {
    fn manifest(&self) -> Manifest {
        splat::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // No weights to key on; the one shared instance's own scene-digest
        // cache (see the module doc) handles per-request rebuilds.
        InstanceKey::new(splat::caps::MODEL, "stateless")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // A conservative fixed slack for one resident render/fit graph.
        // Real footprint varies per request (scene size, resolution) and is
        // not knowable before the invocation arrives - the same limitation
        // `ProviderResident::stateless`'s zero estimate has, just not zeroed
        // out here because a splat render/fit genuinely does allocate real
        // GPU buffers, unlike imageops/demo.
        MemCost::new(512u64 << 20, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(SplatInstance { provider: SplatProvider::new(), device }))
    }
}

struct SplatInstance {
    provider: SplatProvider,
    /// The scheduler-assigned device - `render`'s lazy per-request GPU build
    /// happens inside `run`, not `activate` (the scene isn't known yet), so
    /// this is threaded through instead of scoping the device once up front.
    device: Device,
}

impl Instance for SplatInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let act = self.provider.action(action).ok_or_else(|| format!("splat: unknown action '{action}'"))?;
        crate::resident_llm::on_device(self.device, || act.run(inv, progress))?
    }
}
