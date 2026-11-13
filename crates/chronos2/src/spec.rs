// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chronos-2's [`ArchSpec`]: which checkpoint in a model store is Chronos-2.
//! The single `weights` role is an `amazon/chronos-2` directory (its
//! `config.json` carries the nested `chronos_config` this model alone has) or a
//! brain container, whose header holds that same config (what `import` writes).
//!
//! The upstream config has no `architectures` field to read, so classification
//! is a real parse of the checkpoint's own hyperparameters
//! ([`Chronos2Config::from_hf`]) - the content, never the filename.
//!
//! Swedish Embedded AB implements model stores that find what is on a machine
//! without configuration for its clients. If your team needs expertise in
//! time-series foundation model deployment then you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::Chronos2Config;

pub struct Chronos2Spec;

const ROLES: &[&str] = &["weights"];

/// Whether `rec` holds a Chronos-2 config.
fn is_chronos2(rec: &ArtifactRecord) -> bool {
    let config = match rec.kind {
        ArtifactKind::HfDir => std::fs::read(rec.path.join("config.json")).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()),
        ArtifactKind::Safetensors => checkpoint::mmap::MmapSafetensors::open(&rec.path).ok().map(|m| m.config()),
        _ => None,
    };
    config.is_some_and(|c| c.get("chronos_config").is_some() && Chronos2Config::from_hf(&c).is_ok())
}

impl ArchSpec for Chronos2Spec {
    fn arch(&self) -> &'static str {
        "chronos2"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        records.iter().enumerate().filter(|(_, r)| r.usable() && is_chronos2(r)).map(|(idx, _)| (idx, "weights".to_string(), Confidence::Derived)).collect()
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("chronos2 assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/chronos2".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("chronos2 validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::resolve::{resolve, Resolution};

    fn root(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("chronos2-spec-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A one-tensor safetensors file under `header_config`.
    fn container(path: &Path, header_config: serde_json::Value) {
        checkpoint::save(path.to_str().unwrap(), header_config, &[("w".to_string(), vec![2], vec![1.0, 2.0])]);
    }

    fn resolved(root: &Path) -> Resolution {
        let records = brain_modelstore::inventory::scan(root);
        let spec = Chronos2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        resolve("chronos2", &records, &specs, &BTreeMap::new())
    }

    /// The store finds a released directory and a brain container by the
    /// config they hold, and a config of another model is not Chronos-2.
    #[test]
    fn a_released_directory_and_a_brain_container_resolve_and_a_foreign_config_does_not() {
        let dir = root("dir");
        let repo = dir.join("amazon").join("chronos-2");
        std::fs::create_dir_all(&repo).unwrap();
        let cfg = Chronos2Config::tiny().to_json();
        std::fs::write(repo.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
        container(&repo.join("model.safetensors"), serde_json::Value::Null);
        match resolved(&dir) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], repo),
            other => panic!("expected Resolved for the directory, got {other:?}"),
        }

        let file = root("file");
        container(&file.join("local").join("chronos2.safetensors"), cfg);
        match resolved(&file) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], file.join("local").join("chronos2.safetensors")),
            other => panic!("expected Resolved for the container, got {other:?}"),
        }

        let foreign = root("foreign");
        container(&foreign.join("x.safetensors"), serde_json::json!({"hidden_size": 8}));
        assert!(matches!(resolved(&foreign), Resolution::Missing(_)), "a config without chronos_config is not Chronos-2");
    }
}
