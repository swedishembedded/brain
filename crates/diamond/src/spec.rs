// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAMOND's [`ArchSpec`]: which on-disk checkpoint satisfies the single
//! `weights` role.
//!
//! A brain-format DIAMOND checkpoint is a safetensors file whose model card
//! declares the `diamond` family - what `import` and `finetune` write. A
//! reference `agent.pt` is not one: it is the input of `import`.
//!
//! Swedish Embedded AB implements resolver-based checkpoint discovery like
//! this for clients running mixed fleets of hand-placed and fetched weights.
//! If your team needs the same discipline for its own model store, you can
//! procure our services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::import::CARD_FAMILY;

pub struct DiamondSpec;

const ROLES: &[&str] = &["weights"];

impl ArchSpec for DiamondSpec {
    fn arch(&self) -> &'static str {
        "diamond"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        records
            .iter()
            .enumerate()
            .filter(|(_, rec)| rec.usable() && rec.kind == ArtifactKind::Safetensors)
            .filter(|(_, rec)| matches!(checkpoint::st::read_card(&rec.path.to_string_lossy()), Ok(Some(card)) if card.family == CARD_FAMILY))
            .map(|(idx, _)| (idx, "weights".to_string(), Confidence::Declared))
            .collect()
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("diamond assemble: no weights chosen")?;
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/diamond".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("diamond validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The card family is the signal, not the file's name: a brain DIAMOND
    /// checkpoint is recognized and another family's checkpoint is not.
    #[test]
    fn classify_recognizes_the_card_family() {
        let dir = std::env::temp_dir().join(format!("brain-diamond-spec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, family: &str| {
            let card = checkpoint::st::ModelCard::new("brain/x", family);
            let path = dir.join(name);
            checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), Some(&card)).unwrap();
        };
        write("breakout.safetensors", CARD_FAMILY);
        write("other.safetensors", "lfm");
        let records = brain_modelstore::inventory::scan(&dir);
        let out = DiamondSpec.classify(&records, &dir);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(records[out[0].0].path.ends_with("breakout.safetensors"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
