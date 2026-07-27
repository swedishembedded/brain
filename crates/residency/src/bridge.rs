// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Bridge an existing [`capability::Provider`] into a [`ResidentModel`].
//!
//! For **stateless** providers (no weights — e.g. `imageops`, `demo`) this is the
//! whole story: the cost is ~zero, "activation" just holds the provider `Arc`, and
//! dropping frees nothing. Heavy, weight-holding models instead get a bespoke
//! adapter whose `activate` builds the model on the GPU and whose `Instance` owns it
//! (so eviction actually reclaims memory) — this bridge is not for those.

use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Progress, Provider};

use crate::{Device, InstanceKey, Instance, MemCost, ResidentModel};

/// Wrap a `Provider` as a resident model with a fixed [`MemCost`] (default zero —
/// only correct for stateless providers).
pub struct ProviderResident {
    provider: Arc<dyn Provider>,
    cost: MemCost,
    model: String,
}

impl ProviderResident {
    /// Stateless (no-weight) provider: zero cost, never evicts anything real.
    pub fn stateless(provider: Arc<dyn Provider>) -> ProviderResident {
        let model = provider.manifest().model;
        ProviderResident { provider, cost: MemCost::default(), model }
    }
}

impl ResidentModel for ProviderResident {
    fn manifest(&self) -> Manifest {
        self.provider.manifest()
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // Stateless ⇒ a single shared instance.
        InstanceKey::new(&self.model, "stateless")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        self.cost
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(ProviderInstance { provider: self.provider.clone() }))
    }
}

struct ProviderInstance {
    provider: Arc<dyn Provider>,
}

impl Instance for ProviderInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let act = self.provider.action(action).ok_or_else(|| format!("no action '{action}'"))?;
        // Validate against the action's spec, then run — same path as the CLI/registry.
        let inv = act.spec().validate(inv.clone())?;
        act.run(&inv, progress)
    }
}
