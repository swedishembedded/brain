// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FinCast's [`ArchSpec`]: which checkpoint in a model store is FinCast. The
//! single `weights` role is a brain container (its header holds the config
//! `import` wrote) or the released safetensors, which carries no config at all
//! and is recognised by holding every tensor the published model declares.
//!
//! Classification reads the file's header only, never its tensors: the content,
//! never the filename.
//!
//! Swedish Embedded AB implements model stores that find what is on a machine
//! without configuration for its clients. If your team needs expertise in
//! financial time-series foundation model deployment then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::FincastConfig;

pub struct FincastSpec;

const ROLES: &[&str] = &["weights"];

/// Whether the safetensors file at `path` is FinCast: a config of its own that
/// parses as one, or - for the released file with none - every tensor the
/// published model declares.
fn is_fincast(path: &Path) -> bool {
    let Ok(m) = checkpoint::mmap::MmapSafetensors::open(path) else { return false };
    let config = m.config();
    if !config.is_null() {
        return FincastConfig::from_json(&config).is_ok();
    }
    let have: HashSet<&str> = m.names().iter().map(String::as_str).collect();
    FincastConfig::default().param_list().iter().all(|(name, _)| have.contains(name.as_str()))
}

impl ArchSpec for FincastSpec {
    fn arch(&self) -> &'static str {
        "fincast"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.usable() && r.kind == ArtifactKind::Safetensors && is_fincast(&r.path))
            .map(|(idx, _)| (idx, "weights".to_string(), Confidence::Derived))
            .collect()
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("fincast assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/fincast".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("fincast validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::resolve::{resolve, Resolution};

    fn root(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fincast-spec-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn resolved(root: &Path) -> Resolution {
        let records = brain_modelstore::inventory::scan(root);
        let spec = FincastSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        resolve("fincast", &records, &specs, &BTreeMap::new())
    }

    /// A brain container is found by its config, and any other safetensors file
    /// is not FinCast.
    #[test]
    fn a_brain_container_resolves_and_a_foreign_file_does_not() {
        let dir = root("container");
        let cfg = FincastConfig::tiny();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(k, s)| (k, vec![s.iter().product::<usize>() as u64], vec![0.0; s.iter().product()])).collect();
        let path = dir.join("local").join("fincast.safetensors");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
        match resolved(&dir) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], path),
            other => panic!("expected Resolved, got {other:?}"),
        }

        let foreign = root("foreign");
        checkpoint::save(foreign.join("x.safetensors").to_str().unwrap(), serde_json::json!({"hidden_size": 8}), &[("w".to_string(), vec![2], vec![1.0, 2.0])]);
        assert!(matches!(resolved(&foreign), Resolution::Missing(_)), "a config that is not FinCast's is not FinCast");
    }

    /// A file with no config holds FinCast only if it declares every tensor the
    /// published model does (the header is all that is read, so one element per
    /// tensor is enough of a fixture); a partial one is not.
    #[test]
    fn a_configless_file_is_fincast_only_when_it_holds_every_declared_tensor() {
        let dir = root("configless");
        let all: Vec<(String, Vec<u64>, Vec<f32>)> = FincastConfig::default().param_list().into_iter().map(|(k, _)| (k, vec![1], vec![0.0])).collect();
        let full = dir.join("full.safetensors");
        checkpoint::save(full.to_str().unwrap(), serde_json::Value::Null, &all);
        assert!(is_fincast(&full));
        let partial = dir.join("partial.safetensors");
        checkpoint::save(partial.to_str().unwrap(), serde_json::Value::Null, &all[1..]);
        assert!(!is_fincast(&partial));
    }
}
