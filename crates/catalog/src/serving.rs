// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The residency adapters the catalog owns, for a process that serves them.
//!
//! [`residents`] and [`multi_residents`] read the `resident` field of every
//! [`ModelEntry`](crate::ModelEntry), so a model cannot be listed by
//! `brain caps`, runnable by `brain do` and missing from what a serving
//! process schedules: all three come from the one entry.

use std::path::Path;
use std::sync::Arc;

use capability::Assembly;
use residency::ResidentModel;

use crate::{empty_assembly, models, resolver_spec_for, ResidentCtor};

/// A catalog model's real [`Assembly`], resolved from `models_dir`, collapsed
/// to one `Result`. `None` means the model's weights are not resolver-based at
/// all, so the caller falls back to [`empty_assembly`].
fn resolved_assembly_for(models_dir: Option<&Path>, model: &str) -> Option<Result<Assembly, String>> {
    let (arch, spec) = resolver_spec_for(model)?;
    Some(loader::resolver::try_resolve(models_dir, arch, spec, &std::collections::BTreeMap::new()).map_err(|e| e.message().to_string()))
}

/// The SINGLE-device residency adapters the catalog owns, for models whose
/// weights are configured: what a serving executor starts with.
///
/// Multi-device models are deliberately absent: they come from
/// [`multi_residents`], and registering one in both is the double
/// registration `Executor::register_multi`'s doc forbids.
///
/// `models_dir` is the serving process's resolved models directory, handed to
/// every adapter that resolves its weights through the store.
pub fn residents(models_dir: Option<&Path>) -> Vec<Arc<dyn ResidentModel>> {
    models()
        .into_iter()
        .filter_map(|e| match e.resident {
            Some(ResidentCtor::Single(f)) => f(models_dir),
            _ => None,
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
pub fn multi_residents(models_dir: Option<&Path>, gpus: &[(u32, u64)], reserved: u64) -> Vec<Arc<dyn residency::multi::MultiDeviceResidentModel>> {
    models()
        .into_iter()
        .flat_map(|e| match e.resident {
            Some(ResidentCtor::Multi(f)) => {
                let model = (e.manifest)().model;
                let assembly = match resolved_assembly_for(models_dir, &model) {
                    Some(Ok(a)) => a,
                    Some(Err(err)) => {
                        eprintln!("brain: {model} not served ({err})");
                        return Vec::new();
                    }
                    None => empty_assembly(),
                };
                f(&assembly, gpus, reserved)
            }
            _ => Vec::new(),
        })
        .collect()
}
