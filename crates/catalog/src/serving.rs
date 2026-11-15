// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The residency adapters the catalog owns, for a process that serves them.
//!
//! [`residents`] and [`multi_residents`] read the `resident` field of every
//! [`ModelEntry`](crate::ModelEntry), so a model cannot be listed by
//! `brain caps`, runnable by `brain do` and missing from what a serving
//! process schedules: all three come from the one entry. Each adapter is
//! registered with the architecture its entry's resolver spec names declared
//! on its manifest ([`residency::OfArchitecture`]), so a front end can report
//! the support level that architecture is certified at.

use std::sync::Arc;

use capability::Assembly;
use residency::multi::MultiDeviceResidentModel;
use residency::{OfArchitecture, ResidentModel};

use crate::{empty_assembly, models, resolver_spec_for, ModelEntry, ResidentCtor, ServedStore};

/// The architecture an entry's residents serve: the one its resolver spec
/// resolves weights for. An entry with no spec declares none.
fn architecture_of(entry: &ModelEntry) -> Option<&'static str> {
    entry.spec.map(|(arch, _)| arch)
}

/// `model` with `architecture` declared on its manifest, when there is one.
fn declared(architecture: Option<&'static str>, model: Arc<dyn ResidentModel>) -> Arc<dyn ResidentModel> {
    match architecture {
        Some(arch) => Arc::new(OfArchitecture::new(arch, model)),
        None => model,
    }
}

/// [`declared`] for a multi-device adapter.
fn declared_multi(architecture: Option<&'static str>, model: Arc<dyn MultiDeviceResidentModel>) -> Arc<dyn MultiDeviceResidentModel> {
    match architecture {
        Some(arch) => Arc::new(OfArchitecture::new(arch, model)),
        None => model,
    }
}

/// A catalog model's real [`Assembly`], resolved from `store`, collapsed to
/// one `Result`. `None` means the model's weights are not resolver-based at
/// all, so the caller falls back to [`empty_assembly`].
fn resolved_assembly_for(store: &ServedStore, model: &str) -> Option<Result<Assembly, String>> {
    let (arch, spec) = resolver_spec_for(model)?;
    Some(store.resolve(arch, spec).map_err(|e| e.message().to_string()))
}

/// The SINGLE-device residency adapters the catalog owns, for models whose
/// weights are configured: what a serving executor starts with.
///
/// Multi-device models are deliberately absent: they come from
/// [`multi_residents`], and registering one in both is the double
/// registration `Executor::register_multi`'s doc forbids.
///
/// `store` is the serving process's model store and role pins, handed to
/// every adapter that resolves its weights through the store.
pub fn residents(store: &ServedStore) -> Vec<Arc<dyn ResidentModel>> {
    models()
        .into_iter()
        .flat_map(|e| {
            let architecture = architecture_of(&e);
            match e.resident {
                Some(ResidentCtor::Single(f)) => f(store).into_iter().map(|r| declared(architecture, r)).collect(),
                _ => Vec::new(),
            }
        })
        .collect()
}

/// The MULTI-device residency adapters the catalog owns, registered after
/// `Executor::start` through `register_multi`: the scheduler's multi-device
/// claim path is what reserves on every device such an instance occupies.
///
/// `gpus` is the budgeted `(index, TOTAL bytes)` list and `reserved` its
/// per-card headroom, forwarded verbatim so each adapter picks its device set
/// against the same usable capacity the scheduler budgets.
pub fn multi_residents(store: &ServedStore, gpus: &[(u32, u64)], reserved: u64) -> Vec<Arc<dyn MultiDeviceResidentModel>> {
    models()
        .into_iter()
        .flat_map(|e| {
            let architecture = architecture_of(&e);
            match e.resident {
                Some(ResidentCtor::Multi(f)) => {
                    let model = (e.manifest)().model;
                    let assembly = match resolved_assembly_for(store, &model) {
                        Some(Ok(a)) => a,
                        Some(Err(err)) => {
                            residency::log::not_served(&format!("brain: {model} not served ({err})"));
                            return Vec::new();
                        }
                        None => empty_assembly(),
                    };
                    f(&assembly, gpus, reserved).into_iter().map(|r| declared_multi(architecture, r)).collect()
                }
                _ => Vec::new(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each adapter is registered under the architecture its entry's resolver spec names, so every such name must be
    /// a registered architecture: the support level a front end reports is looked up by it.
    #[test]
    fn every_declared_architecture_is_registered() {
        for entry in models() {
            if let Some(arch) = architecture_of(&entry) {
                assert!(brain_arch::by_id(arch).is_some(), "{}: resolver spec architecture {arch} is not in brain_arch::ARCHS", (entry.manifest)().model);
            }
        }
    }
}
