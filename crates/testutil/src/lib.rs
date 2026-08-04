// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one implementation of brain's testdata fixture path helper.
//!
//! Every crate with a parity/goldens test resolves its fixtures the same way:
//! `$BRAIN_TESTDATA` if set, else `<repo>/testdata` — computed from **this**
//! crate's own `CARGO_MANIFEST_DIR`, not the caller's. That is safe (not a bug)
//! specifically because every crate in the workspace lives at the same depth,
//! `<repo>/crates/<name>/`, so `<this-crate>/../../testdata` and
//! `<caller-crate>/../../testdata` are the identical path. Before this crate
//! existed, that four-line function was copy-pasted, byte-for-byte, into 36
//! separate files — the "one implementation" invariant in `AGENTS.md` exists
//! precisely because of drift risk like that.
//!
//! See `AGENTS.md`'s "No absolute paths in source" section and
//! `scripts/data/fetch-testdata.sh` for how the tree gets populated. A test whose
//! fixture is absent is expected to skip itself (`eprintln!` + early `return`
//! or a `let-else`), never `panic!` — this helper only builds the path, it
//! never checks existence, so the caller stays in control of the skip.

/// The path to a testdata fixture, `<testdata-root>/<rel>`. The root is
/// `$BRAIN_TESTDATA` if set (and non-empty), else `<repo>/testdata`.
pub fn testdata(rel: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/{rel}")
}

/// Same as [`testdata`], as a [`std::path::PathBuf`] — for the handful of
/// callers that immediately need one (`Path::join` rather than string
/// formatting) instead of a `String`.
pub fn testdata_path(rel: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(testdata(rel))
}

/// The on-disk store directory for a fully-qualified upstream model
/// reference (e.g. `"nvidia/nemotron-3.5-asr-streaming-0.6b"`) — the same
/// `<models-dir>/<vendor>/<repo>/` directory `brain fetch` writes and
/// [`brain_modelstore::Store`] scans, resolved via
/// [`brain_modelstore::default_root`]. Import/parity tests that need a real
/// upstream HF checkpoint (raw `model.safetensors`, `config.json`,
/// `tokenizer.json` — not yet brain-converted) resolve it here instead of
/// keeping a private copy under `testdata/`, which stays reserved for actual
/// fixtures (audio/image inputs, golden output dumps).
///
/// `None` when `reference` doesn't parse as `<vendor>/<repo>` or no models
/// directory can be resolved at all (no `$HOME`) — the caller stays in
/// control of skipping, exactly like an absent [`testdata`] fixture; this
/// helper only builds the path, it never checks existence.
pub fn model_dir(reference: &str) -> Option<String> {
    let r = brain_modelref::ModelRef::parse(reference).ok()?;
    let root = brain_modelstore::default_root()?;
    Some(brain_modelstore::Store::new(root).repo_dir(&r).to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_repo_root_testdata() {
        // SAFETY: single-threaded within this test; no other test in this crate
        // touches BRAIN_TESTDATA.
        unsafe {
            std::env::remove_var("BRAIN_TESTDATA");
        }
        let raw = testdata("asr/audio/clip.wav");
        let p = std::path::Path::new(&raw);
        assert_eq!(p.file_name().unwrap(), "clip.wav");
        // `..` components are left uncanonicalized (matching every one of the 36
        // call sites this replaces) — canonicalize the DIRECTORY (not the
        // fixture file, which need not exist) and check it lands on the real
        // repo-root `testdata/`, not `crates/testutil/testdata` or similar.
        let dir = p.parent().unwrap().parent().unwrap().parent().unwrap(); // .../testdata
        let canon = dir.canonicalize().expect("testdata/ must exist in this checkout");
        let expect = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata"))
            .canonicalize()
            .unwrap();
        assert_eq!(canon, expect);
    }

    #[test]
    fn honors_brain_testdata_override() {
        // SAFETY: see above.
        unsafe {
            std::env::set_var("BRAIN_TESTDATA", "/scratch/td");
        }
        assert_eq!(testdata("x/y.bin"), "/scratch/td/x/y.bin");
        unsafe {
            std::env::remove_var("BRAIN_TESTDATA");
        }
    }

    #[test]
    fn empty_override_falls_back_to_default() {
        // SAFETY: see above.
        unsafe {
            std::env::set_var("BRAIN_TESTDATA", "");
        }
        let p = testdata("x");
        assert!(p.ends_with("/testdata/x"), "{p}");
        unsafe {
            std::env::remove_var("BRAIN_TESTDATA");
        }
    }

    #[test]
    fn model_dir_joins_the_models_root_with_vendor_and_repo() {
        // SAFETY: single-threaded within this test; capture and restore the
        // real value so no other test in this binary is left disturbed.
        let orig = std::env::var_os("BRAIN_MODELS_DIR");
        unsafe {
            std::env::set_var("BRAIN_MODELS_DIR", "/scratch/models");
        }
        assert_eq!(
            model_dir("nvidia/nemotron-3.5-asr-streaming-0.6b").as_deref(),
            Some("/scratch/models/nvidia/nemotron-3.5-asr-streaming-0.6b")
        );
        unsafe {
            match orig {
                Some(v) => std::env::set_var("BRAIN_MODELS_DIR", v),
                None => std::env::remove_var("BRAIN_MODELS_DIR"),
            }
        }
    }

    #[test]
    fn model_dir_rejects_a_reference_with_no_vendor() {
        let orig = std::env::var_os("BRAIN_MODELS_DIR");
        unsafe {
            std::env::set_var("BRAIN_MODELS_DIR", "/scratch/models");
        }
        assert_eq!(model_dir("no-slash-here"), None);
        unsafe {
            match orig {
                Some(v) => std::env::set_var("BRAIN_MODELS_DIR", v),
                None => std::env::remove_var("BRAIN_MODELS_DIR"),
            }
        }
    }
}
