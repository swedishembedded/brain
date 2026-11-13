// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The store spec of a model whose checkpoint says which architecture it is:
//! a Hugging Face directory whose `config.json` declares it (the registry's
//! `hf` class names, `crate::family_of_config`), or a brain
//! checkpoint whose card names the family. Its `weights` role is that
//! checkpoint and, for a family that has one, its `tokenizer` role the
//! `tokenizer.json` beside it.
//!
//! One spec for every such architecture, instead of a classifier per model.
//!
//! Swedish Embedded AB implements model stores that find what is on a
//! machine without configuration for its clients. If your team needs
//! expertise in model distribution and resolution then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use crate::inventory::{ArtifactKind, ArtifactRecord};
use crate::resolve::{classify_tokenizer_role, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// Whether a family has a `tokenizer.json` role beside its checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerRole {
    /// The checkpoint carries its own vocabulary, or its directory holds it.
    None,
    /// The checkpoint cannot be served without one.
    Required,
    /// Taken when the store holds one; without it the model runs on token ids.
    Optional,
}

/// The spec of architecture `arch`; `card_family` is what a brain checkpoint
/// of it carries on its card.
pub struct FamilySpec {
    pub arch: &'static str,
    pub card_family: &'static str,
    pub tokenizer: TokenizerRole,
}

const WEIGHTS_ONLY: &[&str] = &["weights"];
const WEIGHTS_AND_TOKENIZER: &[&str] = &["weights", "tokenizer"];
const TOKENIZER: &[&str] = &["tokenizer"];

impl FamilySpec {
    fn is_weights(&self, rec: &ArtifactRecord) -> bool {
        match rec.kind {
            ArtifactKind::HfDir => std::fs::read(rec.path.join("config.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .is_some_and(|config| crate::family_of_config(&config) == Some(self.arch)),
            ArtifactKind::Safetensors => checkpoint::st::read_card(&rec.path.to_string_lossy()).ok().flatten().is_some_and(|card| card.family == self.card_family),
            _ => false,
        }
    }
}

impl ArchSpec for FamilySpec {
    fn arch(&self) -> &'static str {
        self.arch
    }

    fn roles(&self) -> &'static [&'static str] {
        match self.tokenizer {
            TokenizerRole::None => WEIGHTS_ONLY,
            TokenizerRole::Required | TokenizerRole::Optional => WEIGHTS_AND_TOKENIZER,
        }
    }

    fn optional_roles(&self) -> &'static [&'static str] {
        match self.tokenizer {
            TokenizerRole::Optional => TOKENIZER,
            TokenizerRole::None | TokenizerRole::Required => &[],
        }
    }

    fn classify(&self, records: &[ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out: Vec<(usize, String, Confidence)> = records.iter().enumerate().filter(|(_, r)| r.usable() && self.is_weights(r)).map(|(idx, _)| (idx, "weights".to_string(), Confidence::Declared)).collect();
        if self.tokenizer != TokenizerRole::None {
            let weights: Vec<&Path> = out.iter().map(|(idx, ..)| records[*idx].path.as_path()).collect();
            classify_tokenizer_role(records, inventory_root, "tokenizer", &weights, &mut out);
        }
        out
    }

    /// A pulled directory is filed under the repo it came from, so the model
    /// answers to the id a client asks for.
    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let weights = *chosen.get("weights").ok_or_else(|| format!("{} assemble: no weights chosen", self.arch))?;
        let path = &records[weights].path;
        let dir = if path.is_dir() { path.as_path() } else { path.parent().unwrap_or(path) };
        let id = crate::repo_id(dir);
        let id = if id.starts_with("local/") { format!("local/{}", self.arch) } else { id };
        Ok(AssembleOutcome::Assembled(AssembledVariant { id, variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or_else(|| format!("{} validate: assembly has no weights role", self.arch))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::Completeness;

    fn record(path: std::path::PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A directory whose config declares the architecture is its weights,
    /// one that declares another is not, and the id is the repo it is filed
    /// under.
    #[test]
    fn a_directory_is_classified_by_the_architecture_its_config_declares() {
        let root = std::env::temp_dir().join(format!("family-spec-{}", std::process::id()));
        let ours = root.join("vendor").join("huginn-0125");
        let other = root.join("vendor").join("something-else");
        for (dir, class) in [(&ours, "RavenForCausalLM"), (&other, "LlamaForCausalLM")] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("config.json"), serde_json::json!({"architectures": [class]}).to_string()).unwrap();
        }
        let spec = FamilySpec { arch: "huginn", card_family: "huginn", tokenizer: TokenizerRole::Required };
        let records = vec![record(ours.clone(), ArtifactKind::HfDir), record(other, ArtifactKind::HfDir)];
        let out = spec.classify(&records, &root);
        assert!(out.contains(&(0, "weights".to_string(), Confidence::Declared)), "{out:?}");
        assert!(!out.iter().any(|(i, role, _)| *i == 1 && role == "weights"), "{out:?}");
        let chosen = BTreeMap::from([("weights".to_string(), 0usize)]);
        let AssembleOutcome::Assembled(v) = spec.assemble(&chosen, &records, &BTreeMap::new()).unwrap() else { panic!("assembled") };
        assert_eq!(v.id, "vendor/huginn-0125");
        let _ = std::fs::remove_dir_all(&root);
    }
}
