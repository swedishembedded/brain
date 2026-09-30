// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-store resolution for the `MultiModalityCausalLM` family: one `dir`
//! role, a checkpoint directory whose `config.json` the architecture table
//! assigns to this spec's architecture. DeepSeek-VL and Janus-Pro declare the
//! same class; `brain_arch::by_hf_config` tells them apart by the generation
//! heads, so each spec claims only its own checkpoints.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::family_of_config;
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct MultiModalitySpec {
    arch: &'static str,
}

/// DeepSeek-VL's spec.
pub const DEEPSEEK_VL: MultiModalitySpec = MultiModalitySpec { arch: "deepseekvl" };

impl MultiModalitySpec {
    /// The spec for architecture row `arch` (`deepseekvl` or `januspro`).
    pub const fn new(arch: &'static str) -> MultiModalitySpec {
        MultiModalitySpec { arch }
    }
}

const ROLES: &[&str] = &["dir"];

impl ArchSpec for MultiModalitySpec {
    fn arch(&self) -> &'static str {
        self.arch
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
            if family_of_config(&config) == Some(self.arch) {
                out.push((idx, "dir".to_string(), Confidence::Declared));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("dir").ok_or_else(|| format!("{} assemble: no dir chosen", self.arch))?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/{}", self.arch), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let dir = assembly.roles.get("dir").ok_or_else(|| format!("{} validate: assembly has no dir role", self.arch))?;
        for file in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"] {
            if !dir.join(file).is_file() {
                return Err(format!("{} validate: {} has no {file}", self.arch, dir.display()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("brain-deepseekvl-spec-{tag}-{}", std::process::id()))
    }

    fn checkpoint(dir: &Path, janus: bool) {
        std::fs::create_dir_all(dir).unwrap();
        let mut config = serde_json::json!({"architectures": ["MultiModalityCausalLM"], "model_type": "multi_modality"});
        if janus {
            config["gen_head_config"] = serde_json::json!({});
        }
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        for f in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"] {
            std::fs::write(dir.join(f), b"{}").unwrap();
        }
        let st = dir.join("model.safetensors");
        std::fs::write(&st, [8u8, 0, 0, 0, 0, 0, 0, 0, b'{', b'}', b' ', b' ', b' ', b' ', b' ', b' ']).unwrap();
    }

    #[test]
    fn each_family_member_claims_only_its_own_checkpoint() {
        let root = tmp("claims");
        let (vl, janus) = (root.join("deepseek-ai/deepseek-vl-7b-chat"), root.join("deepseek-ai/Janus-Pro-7B"));
        checkpoint(&vl, false);
        checkpoint(&janus, true);
        let records = brain_modelstore::inventory::scan(&root);
        for (spec, want) in [(DEEPSEEK_VL, &vl), (MultiModalitySpec::new("januspro"), &janus)] {
            let specs: Vec<&dyn ArchSpec> = vec![&spec];
            match resolve(spec.arch, &records, &specs, &BTreeMap::new()) {
                Resolution::Resolved(a) => assert_eq!(&a.roles["dir"], want, "{}", spec.arch),
                other => panic!("{}: expected Resolved, got {other:?}", spec.arch),
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
