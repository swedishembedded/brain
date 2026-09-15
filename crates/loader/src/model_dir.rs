// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The models-directory lookup -- moved out of `crates/cli/src/model_dir.rs`
//! verbatim. Everything else that used to live in that file (the store scan,
//! the per-family `resident_for`/`resident_for_compound` dispatch) stays in
//! `crates/cli`: those construct the ~20 CLI-local `ResidentModel` adapters
//! (`crate::resident_llm::QwenResident` and its siblings), which is exactly
//! the "residency-adapter glue" `crates/catalog`'s own module doc draws the
//! same line around. [`resolve`] alone is pure "which directory" logic, with
//! no CLI-local type in sight, so it is what any embedder needs too.

use std::path::PathBuf;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prefers_the_flag() {
        // The explicit flag wins over env/default; empty flag falls through.
        assert!(resolve(Some("flagdir")).unwrap().ends_with("flagdir"));
    }
}
