// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-store resolution: one `dir` role, a checkpoint directory whose
//! `config.json` the architecture table assigns to `minicpmv`
//! (`MiniCPMV4_6ForConditionalGeneration`).

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::family_of_config;
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct MiniCpmVSpec;

const ARCH: &str = "minicpmv";
const ROLES: &[&str] = &["dir"];

/// The files a served checkpoint needs beside its weights.
pub const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json", "preprocessor_config.json"];

impl ArchSpec for MiniCpmVSpec {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::HfDir {
                continue;
            }
            let Ok(bytes) = std::fs::read(rec.path.join("config.json")) else { continue };
            let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
            if family_of_config(&config) == Some(ARCH) {
                out.push((idx, "dir".to_string(), Confidence::Declared));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("dir").ok_or("minicpmv assemble: no dir chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/{ARCH}"), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let dir = assembly.roles.get("dir").ok_or("minicpmv validate: assembly has no dir role")?;
        for file in REQUIRED_FILES {
            if !dir.join(file).is_file() {
                return Err(format!("minicpmv validate: {} has no {file}", dir.display()));
            }
        }
        if data::chat_template::ChatTemplate::from_model_dir(dir).is_err() {
            return Err(format!("minicpmv validate: {} has no chat template", dir.display()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::resolve::{resolve, Resolution};

    #[test]
    fn the_spec_claims_a_minicpmv_checkpoint_and_nothing_else() {
        let root = std::env::temp_dir().join(format!("brain-minicpmv-spec-{}", std::process::id()));
        let (mine, other) = (root.join("openbmb/MiniCPM-V-4.6"), root.join("Qwen/Qwen3.8-27B"));
        for (dir, arch, model_type) in [(&mine, "MiniCPMV4_6ForConditionalGeneration", "minicpmv4_6"), (&other, "Qwen3_5ForConditionalGeneration", "qwen3_5")] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("config.json"), serde_json::json!({"architectures": [arch], "model_type": model_type}).to_string()).unwrap();
            std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
            std::fs::write(dir.join("preprocessor_config.json"), b"{}").unwrap();
            std::fs::write(dir.join("chat_template.jinja"), b"{{ messages }}").unwrap();
            std::fs::write(dir.join("model.safetensors"), [8u8, 0, 0, 0, 0, 0, 0, 0, b'{', b'}', b' ', b' ', b' ', b' ', b' ', b' ']).unwrap();
        }
        let records = brain_modelstore::inventory::scan(&root);
        let specs: Vec<&dyn ArchSpec> = vec![&MiniCpmVSpec];
        match resolve(ARCH, &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.roles["dir"], mine),
            other => panic!("expected Resolved, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
