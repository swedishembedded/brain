// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Serving what [`crate::train`] trained: a directory holding
//! `adapter.safetensors` (the decoder's LoRA) and, when the aligner trained,
//! `aligner.safetensors`. The adapter is folded into the decoder's weights as
//! they load ([`crate::model::Vlm::assemble`]), and the aligner's parameters
//! replace the checkpoint's in the loaded tower.

use std::path::Path;

use crate::model::Vlm;
use crate::train::{ADAPTER_FILE, ALIGNER_FILE};

/// A fine-tune kept in a model's store directory, at
/// `<model dir>/adapters/<owner>/<name>/<tag>/`: the layout the model store
/// gives a text adapter, so every fine-tune of a base is found the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stored {
    /// `owner:name:tag`, the suffix that makes a model id of the base's id.
    pub label: String,
    pub dir: std::path::PathBuf,
}

/// The fine-tunes stored under `model_dir`, ordered by label. A directory
/// counts when it holds a file `holds` accepts; anything else beside them is
/// left alone, not an error.
pub fn scan(model_dir: &Path, holds: impl Fn(&Path) -> bool) -> Vec<Stored> {
    let subdirs = |p: &Path| -> Vec<(String, std::path::PathBuf)> {
        let Ok(rd) = std::fs::read_dir(p) else { return Vec::new() };
        rd.flatten().filter(|e| e.path().is_dir()).filter_map(|e| Some((e.file_name().to_str()?.to_string(), e.path()))).collect()
    };
    let mut out = Vec::new();
    for (owner, o) in subdirs(&model_dir.join("adapters")) {
        for (name, n) in subdirs(&o) {
            for (tag, dir) in subdirs(&n) {
                if holds(&dir) {
                    out.push(Stored { label: format!("{owner}:{name}:{tag}"), dir });
                }
            }
        }
    }
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

/// Whether `dir` holds a decoder adapter or an aligner, i.e. [`apply`] can use it.
pub fn is_vl_fine_tune(dir: &Path) -> bool {
    dir.join(ADAPTER_FILE).is_file() || dir.join(ALIGNER_FILE).is_file()
}

/// The decoder adapter of the fine-tune in `dir`, to fold into the decoder as
/// it loads (`None` when only the aligner trained). A directory with neither
/// file is an error: serving "a fine-tune" that changes nothing would be a
/// silent no-op.
pub fn decoder_adapter(dir: &Path) -> Result<Option<std::path::PathBuf>, String> {
    if !is_vl_fine_tune(dir) {
        return Err(format!("{}: holds neither {ADAPTER_FILE} nor {ALIGNER_FILE}, so it is not a fine-tune of this model", dir.display()));
    }
    Ok(Some(dir.join(ADAPTER_FILE)).filter(|p| p.is_file()))
}

/// Replace the tower's aligner with the one the fine-tune in `dir` trained,
/// when it trained one.
pub fn apply_aligner(vlm: &Vlm, dir: &Path) -> Result<(), String> {
    let aligner = dir.join(ALIGNER_FILE);
    if !aligner.is_file() {
        return Ok(());
    }
    let path = aligner.to_str().ok_or("path is not UTF-8")?;
    let st = checkpoint::st::load_safetensors(path).map_err(|e| format!("{path}: {e}"))?;
    vlm.tower.set_aligner(&st.tensors).map_err(|e| format!("{path}: {e}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn stored_fine_tunes_are_found_by_owner_name_and_tag() {
        let root = std::env::temp_dir().join(format!("brain-vl-tuned-scan-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        for (rel, file) in [("adapters/acme/puppies/v2", super::ADAPTER_FILE), ("adapters/acme/puppies/v1", super::ALIGNER_FILE), ("adapters/acme/empty/v1", "notes.txt")] {
            std::fs::create_dir_all(root.join(rel)).unwrap();
            std::fs::write(root.join(rel).join(file), b"x").unwrap();
        }
        let found = super::scan(&root, super::is_vl_fine_tune);
        assert_eq!(found.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(), ["acme:puppies:v1", "acme:puppies:v2"], "a directory with no fine-tune file is not one");
        assert!(super::scan(&root.join("missing"), super::is_vl_fine_tune).is_empty(), "no adapters directory, no fine-tunes");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_file_names_are_the_trainer_s() {
        assert_eq!((super::ADAPTER_FILE, super::ALIGNER_FILE), ("adapter.safetensors", "aligner.safetensors"));
    }
}
