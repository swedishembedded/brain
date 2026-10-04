// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The models-directory lookup -- moved out of `crates/serving/src/model_dir.rs`
//! verbatim. Everything else that used to live in that file (the store scan,
//! the per-family `resident_for`/`resident_for_compound` dispatch) stays in
//! `crates/cli`: those construct the ~20 CLI-local `ResidentModel` adapters
//! (`crate::resident_llm::QwenResident` and its siblings), which is exactly
//! the "residency-adapter glue" `crates/catalog`'s own module doc draws the
//! same line around. [`resolve`] alone is pure "which directory" logic, with
//! no CLI-local type in sight, so it is what any embedder needs too.
//! [`resolve_base`] is its one consumer-facing companion: which checkpoint a
//! base-model argument names, given the directory [`resolve`] answered.

use std::path::{Path, PathBuf};

/// Resolve the models directory. Precedence: the `--models-dir` flag (if the
/// caller has one to pass), then [`brain_modelstore::default_root`]
/// (`BRAIN_MODELS_DIR`, then `$XDG_DATA_HOME/brain/models`, then
/// `$HOME/.local/share/brain/models`). `None` only when the flag and all
/// three env vars are unset (no HOME) -- a caller then has no store to scan.
pub fn resolve(flag: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = flag.filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(p));
    }
    brain_modelstore::default_root()
}

/// Resolve a base-model argument to `(weights file, its directory, canonical
/// id)`. `base` is one of:
///
/// - a repo DIRECTORY (`<root>/<vendor>/<repo>`), resolved through
///   [`brain_modelstore::Store`] so the compound/quant/manifest layout rule
///   lives in one place instead of a guessed filename;
/// - a checkpoint FILE, whose sibling directory supplies the tokenizer and
///   chat template; its id is synthesized as `local/<stem>` under the
///   reserved `local` vendor, because the adapter ref grammar
///   (`vendor/repo:owner:name:tag`) needs a `vendor/repo` a bare filename
///   could never parse as;
/// - a `vendor/repo[-QUANT]` reference, looked up in `store_root` (the
///   directory [`resolve`] answered for the caller's own flag).
///
/// Filesystem existence is checked FIRST: a relative path such as
/// `out/qwen.safetensors` also parses as a syntactically valid `ModelRef`, so
/// "is this a real file" must win before "is this a ref" is considered.
pub fn resolve_base(base: &str, store_root: Option<&Path>) -> Result<(PathBuf, PathBuf, String), String> {
    let path = Path::new(base);
    if path.is_dir() {
        return resolve_repo_dir(path);
    }
    if path.is_file() {
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("base");
        return Ok((path.to_path_buf(), dir, format!("local/{stem}")));
    }
    let r = brain_modelref::ModelRef::parse(base).map_err(|e| format!("{base}: not a file, and not a valid model ref ({e})"))?;
    let root = store_root.ok_or_else(|| format!("{base}: a model ref, but no models directory resolved (name one, or set BRAIN_MODELS_DIR or HOME)"))?;
    let local = brain_modelstore::Store::new(root).local(&r).ok_or_else(|| format!("{base}: not found in the model store at {}", root.display()))?;
    Ok((checkpoint_of(&local), local.dir, r.to_string()))
}

/// What a caller opens as the weights of a store entry. A compound manifest
/// anchors `weights` at `brain.manifest.json` itself and names the checkpoint
/// in its `weights` role (the repo directory for a pulled Hugging Face
/// model); opening the anchor would read the manifest as tensors.
fn checkpoint_of(local: &brain_modelstore::LocalModel) -> PathBuf {
    local.roles.as_ref().and_then(|roles| roles.get("weights")).cloned().unwrap_or_else(|| local.weights.clone())
}

/// A repo directory -> the same `(weights, dir, id)` a `vendor/repo` ref
/// resolves to: a directory in the store IS `<root>/<vendor>/<repo>`, so
/// splitting those parts back out reuses the store's own resolution.
fn resolve_repo_dir(dir: &Path) -> Result<(PathBuf, PathBuf, String), String> {
    let unservable = || format!("{}: a directory with no servable checkpoint in it", dir.display());
    let repo = dir.file_name().and_then(|s| s.to_str()).ok_or_else(unservable)?;
    let parent = dir.parent().ok_or_else(unservable)?;
    let vendor = parent.file_name().and_then(|s| s.to_str()).ok_or_else(unservable)?;
    let root = parent.parent().ok_or_else(unservable)?;
    let r = brain_modelref::ModelRef::parse(&format!("{vendor}/{repo}")).map_err(|_| unservable())?;
    let local = brain_modelstore::Store::new(root).local(&r).ok_or_else(unservable)?;
    Ok((checkpoint_of(&local), local.dir, r.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prefers_the_flag() {
        // The explicit flag wins over env/default; empty flag falls through.
        assert!(resolve(Some("flagdir")).unwrap().ends_with("flagdir"));
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-loader-resolve-base-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A `<root>/Qwen/Qwen3-0.6B` store entry holding a carded brain checkpoint.
    fn store_with_qwen(root: &Path) -> PathBuf {
        let repo_dir = root.join("Qwen").join("Qwen3-0.6B");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let card = checkpoint::st::ModelCard::new("Qwen/Qwen3-0.6B", "qwen");
        checkpoint::st::save_safetensors(
            repo_dir.join("model.brain.safetensors").to_str().unwrap(),
            &[("weight".to_string(), vec![2], vec![1.0, 2.0])],
            &serde_json::json!({"vocab_size": 23}),
            Some(&card),
        )
        .unwrap();
        std::fs::write(repo_dir.join("tokenizer.json"), b"{}").unwrap();
        repo_dir
    }

    #[test]
    fn resolve_base_prefers_a_real_file_over_parsing_it_as_a_model_ref() {
        // "<dir>/qwen.safetensors" is ALSO a syntactically valid ModelRef -
        // the file on disk must win.
        let dir = tmp("file-path");
        let weights = dir.join("qwen.safetensors");
        std::fs::write(&weights, b"not a real checkpoint, just needs to exist").unwrap();

        let (path, base_dir, id) = resolve_base(weights.to_str().unwrap(), None).unwrap();
        assert_eq!(path, weights);
        assert_eq!(base_dir, dir);
        assert_eq!(id, "local/qwen");
        assert!(brain_modelref::ModelRef::parse(&format!("{id}:o:n:latest")).is_ok(), "id {id:?} must combine into a parseable adapter ref");
    }

    #[test]
    fn resolve_base_reports_neither_a_file_nor_a_valid_ref_by_name_not_a_panic() {
        let err = resolve_base("not-a-file-and-not-a-ref-either", None).unwrap_err();
        assert!(err.contains("not a file"), "{err}");
    }

    #[test]
    fn resolve_base_reports_a_missing_models_dir_by_name_not_a_panic() {
        let err = resolve_base("Qwen/Qwen3-0.6B", None).unwrap_err();
        assert!(err.contains("no models directory"), "{err}");
    }

    #[test]
    fn resolve_base_resolves_a_store_ref_via_the_model_store() {
        let dir = tmp("store-ref");
        let repo_dir = store_with_qwen(&dir);
        let (path, base_dir, id) = resolve_base("Qwen/Qwen3-0.6B", Some(&dir)).unwrap();
        assert_eq!(path, repo_dir.join("model.brain.safetensors"));
        assert_eq!(base_dir, repo_dir);
        assert_eq!(id, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn resolve_base_resolves_a_repo_directory_to_the_checkpoint_inside_it() {
        let dir = tmp("repo-dir");
        let repo_dir = store_with_qwen(&dir);
        let (path, base_dir, id) = resolve_base(repo_dir.to_str().unwrap(), None).unwrap();
        assert_eq!(path, repo_dir.join("model.brain.safetensors"));
        assert_eq!(base_dir, repo_dir);
        assert_eq!(id, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn resolve_base_opens_a_pulled_checkpoint_directory_not_its_manifest() {
        // A pulled Hugging Face checkpoint is a compound manifest whose
        // `weights` role is the repo directory itself. The weights a caller
        // opens are that directory, never `brain.manifest.json`.
        let dir = tmp("compound-manifest");
        let repo_dir = dir.join("deepseek-ai").join("R1-Tiny");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(repo_dir.join("model.safetensors"), b"weights").unwrap();
        std::fs::write(
            repo_dir.join("brain.manifest.json"),
            br#"{"id":"deepseek-ai/R1-Tiny","family":"qwen2","roles":{"weights":"."}}"#,
        )
        .unwrap();

        for base in [repo_dir.to_str().unwrap(), "deepseek-ai/R1-Tiny"] {
            let (path, base_dir, id) = resolve_base(base, Some(&dir)).unwrap();
            assert_eq!(path, repo_dir, "{base}");
            assert_eq!(base_dir, repo_dir, "{base}");
            assert_eq!(id, "deepseek-ai/R1-Tiny");
        }
    }

    #[test]
    fn resolve_base_reports_a_directory_with_no_checkpoint_by_name() {
        let dir = tmp("empty-repo-dir");
        let repo_dir = dir.join("Qwen").join("Nothing-Here");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let err = resolve_base(repo_dir.to_str().unwrap(), None).unwrap_err();
        assert!(err.contains("no servable checkpoint"), "{err}");
    }

    #[test]
    fn resolve_base_reports_a_ref_not_found_in_the_store_by_name_not_a_panic() {
        let dir = tmp("store-ref-missing");
        let err = resolve_base("Qwen/Qwen3-0.6B", Some(&dir)).unwrap_err();
        assert!(err.contains("not found in the model store"), "{err}");
    }
}
