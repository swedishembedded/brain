// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Bridge an existing [`capability::Provider`] into a [`ResidentModel`].
//!
//! For **stateless** providers (no weights — e.g. `imageops`, `demo`) this is the
//! whole story: the cost is ~zero, "activation" just holds the provider `Arc`, and
//! dropping frees nothing. Heavy, weight-holding models instead get a bespoke
//! adapter whose `activate` builds the model on the GPU and whose `Instance` owns it
//! (so eviction actually reclaims memory) — this bridge is not for those.
//!
//! A provider whose action spec carries a param that is service-side
//! configuration rather than a real per-request input - a checkpoint
//! directory a served instance is already resolved from (`BRAIN_*_WEIGHTS`),
//! not something a scheduled caller ever has a reason to name - needs
//! [`ProviderResident::stateless_with_manifest`] instead of
//! [`ProviderResident::stateless`]: the override manifest is both what a
//! caller is TOLD it may pass and what every invocation is actually
//! validated against, so a param the override omits is unreachable even from
//! a caller that ignores the advertised schema and crafts a raw invocation
//! by hand - the same reason `glmdsa::caps::manifest_resident` and
//! `qwen3tts::caps::resident_manifest` exist as a second, weights-free
//! manifest next to their static one.

use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Progress, Provider};

use crate::{Device, InstanceKey, Instance, MemCost, ResidentModel};

/// Wrap a `Provider` as a resident model with a fixed [`MemCost`] (default zero —
/// only correct for stateless providers).
pub struct ProviderResident {
    provider: Arc<dyn Provider>,
    cost: MemCost,
    model: String,
    /// `None` - serve (and validate against) the provider's own `manifest()`
    /// unmodified. `Some` - serve (and validate against) THIS manifest
    /// instead; see the module doc for why this is the mechanism that
    /// actually removes a param, not just hides it from display.
    manifest_override: Option<Manifest>,
}

impl ProviderResident {
    /// Stateless (no-weight) provider: zero cost, never evicts anything real.
    pub fn stateless(provider: Arc<dyn Provider>) -> ProviderResident {
        let model = provider.manifest().model;
        ProviderResident { provider, cost: MemCost::default(), model, manifest_override: None }
    }

    /// Stateless provider, served (and validated) under `manifest` instead of
    /// the provider's own `manifest()` - see the module doc.
    pub fn stateless_with_manifest(provider: Arc<dyn Provider>, manifest: Manifest) -> ProviderResident {
        let model = manifest.model.clone();
        ProviderResident { provider, cost: MemCost::default(), model, manifest_override: Some(manifest) }
    }
}

impl ResidentModel for ProviderResident {
    fn manifest(&self) -> Manifest {
        self.manifest_override.clone().unwrap_or_else(|| self.provider.manifest())
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // Stateless ⇒ a single shared instance.
        InstanceKey::new(&self.model, "stateless")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        self.cost
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(ProviderInstance { provider: self.provider.clone(), manifest_override: self.manifest_override.clone() }))
    }
}

struct ProviderInstance {
    provider: Arc<dyn Provider>,
    manifest_override: Option<Manifest>,
}

impl Instance for ProviderInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let act = self.provider.action(action).ok_or_else(|| format!("no action '{action}'"))?;
        // Validate against the SERVED spec (the override when one is set),
        // never the provider's own raw spec - a param the override strips
        // (e.g. a checkpoint path) must be genuinely unreachable through this
        // resident, not merely absent from what a well-behaved caller was
        // shown.
        let spec = match &self.manifest_override {
            Some(m) => m
                .actions
                .iter()
                .find(|a| a.name == action)
                .cloned()
                .ok_or_else(|| format!("no action '{action}' in the served manifest"))?,
            None => act.spec(),
        };
        let inv = spec.validate(inv.clone())?;
        act.run(&inv, progress)
    }
}
