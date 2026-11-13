// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`capability::Components`] over the model store: a multi-model action
//! asks for an architecture's weights and gets them resolved exactly as
//! `brain <model> <action>` resolves its own.

use std::collections::BTreeMap;

use brain_modelstore::resolve::ArchSpec;
use capability::{Assembly, Components};

/// The model store, for the architectures `specs` names.
pub struct StoreComponents {
    specs: Vec<(&'static str, &'static dyn ArchSpec)>,
}

impl StoreComponents {
    pub fn new(specs: Vec<(&'static str, &'static dyn ArchSpec)>) -> StoreComponents {
        StoreComponents { specs }
    }
}

impl Components for StoreComponents {
    fn resolve(&self, arch: &str, overrides: &BTreeMap<String, String>) -> Result<Assembly, String> {
        let (_, spec) = self.specs.iter().find(|(a, _)| *a == arch).ok_or_else(|| format!("no weights resolver for '{arch}' here"))?;
        crate::resolver::try_resolve(crate::model_dir::resolve(None).as_deref(), arch, *spec, overrides).map_err(|e| format!("{arch}: {}", e.message().trim_end()))
    }
}
