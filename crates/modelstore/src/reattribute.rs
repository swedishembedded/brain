// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Repairing a stale family attribution in a pulled repo's manifest.
//!
//! A repo pulled before a recipe-selection fix keeps the
//! `brain.manifest.json` the old selection wrote, forever: `brain pull`
//! short-circuits on a repo that already resolves locally. This module asks
//! the CURRENT selection ([`crate::recipe::select`]) what the repo's own
//! LOCAL file listing is, and compares it with the manifest on disk.
//!
//! Only recipes keyed on files (the [`crate::recipe::files_recipe_roles`]
//! table) can be decided offline, and only those are ever written: every
//! other recipe writes its manifest through a finish step that may have
//! converted tensors, which a manifest rewrite cannot reproduce. A repo whose
//! selection is ambiguous, or whose selected roles are not on disk, is left
//! alone and says why -- never silently re-attributed on a guess.
//!
//! The read-only half ([`assess`]) is what `brain models list` uses to flag
//! a stale row; the writing half ([`reattribute`]) is `brain pull`'s.
//!
//! Swedish Embedded AB implements model distribution and weight-management
//! tooling for its clients. If your team needs expertise in keeping fleets of
//! pulled models consistent across upstream recipe changes then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelref::ModelRef;

use crate::recipe::{files_recipe_roles, recipes, select};
use crate::{CompoundManifest, Store, MANIFEST_FILE};

/// What the current recipe selection says about a repo's manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attribution {
    /// No compound manifest here, or the selection is not a file-keyed
    /// recipe: nothing this module owns.
    NotApplicable,
    /// The manifest already names what the current selection gives.
    Current,
    /// The manifest differs from the current selection.
    Stale { from: String, to: String, roles: BTreeMap<String, String> },
    /// The right attribution cannot be decided offline; the manifest stays.
    Undecided(String),
}

impl Attribution {
    /// The one-line description printed for a stale manifest.
    pub fn stale_note(from: &str, to: &str) -> String {
        if from == to {
            format!("roles re-derived for {to}")
        } else {
            format!("re-attributed from {from} to {to}")
        }
    }
}

/// Every file under `dir`, as `/`-joined paths relative to it, sorted. The
/// manifest itself is not part of the repo's upstream listing.
fn local_listing(dir: &Path) -> std::io::Result<Vec<String>> {
    fn walk(dir: &Path, prefix: &str, out: &mut Vec<String>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let rel = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
            let kind = entry.file_type()?;
            if kind.is_dir() {
                walk(&entry.path(), &rel, out)?;
            } else if rel != MANIFEST_FILE {
                out.push(rel);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, "", &mut out)?;
    out.sort();
    Ok(out)
}

/// Compares `reference`'s on-disk manifest with the current recipe selection
/// over the repo's local file listing. Reads only; no network.
pub fn assess(store: &Store, reference: &ModelRef) -> Attribution {
    if reference.adapter().is_some() || reference.quant().is_some() {
        return Attribution::NotApplicable;
    }
    let dir = store.repo_dir(reference);
    let Ok(bytes) = std::fs::read(dir.join(MANIFEST_FILE)) else { return Attribution::NotApplicable };
    let Ok(manifest) = serde_json::from_slice::<CompoundManifest>(&bytes) else { return Attribution::NotApplicable };
    let Ok(listing) = local_listing(&dir) else { return Attribution::NotApplicable };
    let picked = match select(recipes(), reference, &listing) {
        Ok(p) => p,
        Err(e) => return Attribution::Undecided(format!("left as {}: {e}", manifest.family)),
    };
    let Some((family, roles_table)) = files_recipe_roles(picked.id()) else { return Attribution::NotApplicable };
    let roles: BTreeMap<String, String> = roles_table.iter().map(|(r, p)| (r.to_string(), p.to_string())).collect();
    if manifest.family == family && manifest.roles == roles {
        return Attribution::Current;
    }
    if let Some((role, rel)) = roles.iter().find(|(_, rel)| !dir.join(rel).exists()) {
        return Attribution::Undecided(format!("left as {}: {family}'s role {role:?} ({rel}) is not on disk", manifest.family));
    }
    Attribution::Stale { from: manifest.family, to: family.to_string(), roles }
}

/// [`assess`], and when the manifest is stale rewrite it atomically (a
/// sibling temp file renamed over it, so a crash leaves the old manifest or
/// the new one, never a torn file). Returns what was found.
pub fn reattribute(store: &Store, reference: &ModelRef) -> Result<Attribution, String> {
    let found = assess(store, reference);
    if let Attribution::Stale { to, roles, .. } = &found {
        let dir = store.repo_dir(reference);
        let manifest = CompoundManifest { id: format!("{}/{}", reference.vendor(), reference.repo()), family: to.clone(), roles: roles.clone() };
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| format!("{reference}: encode manifest: {e}"))?;
        let tmp = dir.join(format!("{MANIFEST_FILE}.tmp"));
        std::fs::write(&tmp, bytes).map_err(|e| format!("{reference}: write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, dir.join(MANIFEST_FILE)).map_err(|e| format!("{reference}: replace manifest: {e}"))?;
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_store::test_store;

    const TTS_FILES: &[&str] = &[
        "config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
    ];

    fn repo(store: &Store, files: &[&str], manifest: &str) -> ModelRef {
        let r = ModelRef::new("Qwen", "Qwen3-TTS-12Hz-1.7B-Base", None);
        let dir = store.repo_dir(&r);
        for f in files {
            let p = dir.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"{}").unwrap();
        }
        std::fs::write(dir.join(MANIFEST_FILE), manifest).unwrap();
        r
    }

    const STALE: &str = r#"{"id":"Qwen/Qwen3-TTS-12Hz-1.7B-Base","family":"qwen3asr","roles":{"weights":"."}}"#;

    #[test]
    fn a_stale_family_is_rewritten_to_what_the_files_select() {
        let store = test_store("reattr-stale");
        let r = repo(&store, TTS_FILES, STALE);
        let found = reattribute(&store, &r).unwrap();
        assert!(matches!(&found, Attribution::Stale { from, to, .. } if from == "qwen3asr" && to == "qwen3tts"), "{found:?}");
        let m: CompoundManifest = serde_json::from_slice(&std::fs::read(store.repo_dir(&r).join(MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(m.family, "qwen3tts");
        assert_eq!(m.roles, BTreeMap::from([("ckpt".to_string(), ".".to_string()), ("weights_dir".to_string(), ".".to_string())]));
        assert_eq!(m.id, "Qwen/Qwen3-TTS-12Hz-1.7B-Base");
        assert!(!store.repo_dir(&r).join(format!("{MANIFEST_FILE}.tmp")).exists(), "temp file left behind");
        assert_eq!(assess(&store, &r), Attribution::Current, "a second pass has nothing to do");
        assert_eq!(Attribution::stale_note("qwen3asr", "qwen3tts"), "re-attributed from qwen3asr to qwen3tts");
    }

    #[test]
    fn a_correct_manifest_is_left_byte_identical() {
        let store = test_store("reattr-current");
        let good = r#"{ "id": "x",   "family": "qwen3tts", "roles": {"weights_dir": ".", "ckpt": "."} }"#;
        let r = repo(&store, TTS_FILES, good);
        assert_eq!(reattribute(&store, &r).unwrap(), Attribution::Current);
        assert_eq!(std::fs::read_to_string(store.repo_dir(&r).join(MANIFEST_FILE)).unwrap(), good);
    }

    /// Two same-tier recipes (the tiny and large SAM2 releases) both claim a
    /// listing that carries both checkpoints, and neither signature contains
    /// the other: selection refuses, so the manifest stays and says why.
    #[test]
    fn an_offline_ambiguous_repo_is_left_alone_with_a_reason() {
        let store = test_store("reattr-ambiguous");
        let r = repo(&store, &["sam2.1_hiera_tiny.pt", "sam2.1_hiera_large.pt"], STALE);
        match reassess(&store, &r) {
            Attribution::Undecided(why) => assert!(why.contains("ambiguous") && why.contains("qwen3asr"), "{why}"),
            other => panic!("expected Undecided, got {other:?}"),
        }
        assert_eq!(std::fs::read_to_string(store.repo_dir(&r).join(MANIFEST_FILE)).unwrap(), STALE);
    }

    fn reassess(store: &Store, r: &ModelRef) -> Attribution {
        reattribute(store, r).unwrap()
    }
}
