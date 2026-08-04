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
//! `scripts/fetch-testdata.sh` for how the tree gets populated. A test whose
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
}
