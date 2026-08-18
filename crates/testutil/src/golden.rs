// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which checkpoint a golden dump came from, recorded by the dumper and
//! enforced by the test.
//!
//! # The failure this exists to make impossible
//!
//! A golden dump is a set of tensors plus a claim: "this is what the reference
//! implementation produced". The claim is only meaningful together with *which
//! checkpoint* produced it, and that half was never written down. Pair a dump
//! with a different tier of the same architecture and one of three things
//! happens, none of them a parity failure:
//!
//! 1. The shapes differ, so the importer dies deep inside with
//!    "embedding.emb_s1.weight has 851968 elems, expected 524288" - a
//!    tensor-shape error where the real problem is "these goldens are not for
//!    this checkpoint".
//! 2. The shapes happen to agree and the comparison runs against the wrong
//!    reference, certifying a number that means nothing.
//! 3. The suite notices something is off, prints to stderr and returns. Cargo
//!    reports that as a PASS.
//!
//! All three have happened here. The Kronos parity suite hit (1), and before
//! the shapes diverged it had been sitting in (3).
//!
//! # The convention
//!
//! Every `tools/goldens/*_dump_reference.py` already writes a `manifest.json`
//! next to its tensors, carrying per-file shapes and sha256. That manifest
//! gains one more block, written by `tools/goldens/golden_source.py` so every
//! dumper spells it the same way:
//!
//! ```json
//! "source": {
//!   "checkpoint": "NeoQuasar/Kronos-small",
//!   "files": { "model.safetensors": "sha256:9f86d0…" },
//!   "identity": { "d_model": 512, "n_layers": 12 }
//! }
//! ```
//!
//! `identity` is the load-bearing field and it is deliberately NOT the
//! checkpoint's path or name. A path resolves on one machine (and `crates/`
//! may not contain one at all), and a name is a label a dumper can get wrong.
//! `identity` holds the architectural config that *determines every tensor
//! shape in the dump* - the same numbers the test's own importer reads out of
//! the checkpoint it is about to compare against. Two tiers of one
//! architecture cannot agree on it, which is exactly the case that used to be
//! undetectable. `files`/`checkpoint` are forensics: they say which artifact
//! to go look at once a mismatch is reported, and they are not required to
//! match (the same weights legitimately arrive under several names).
//!
//! # What a mismatch does
//!
//! It is a **missing fixture**, not a parity violation: the checkpoint on this
//! box is fine and the goldens are fine, they just do not belong together. So
//! it routes through [`crate::skip`] - a named, actionable skip normally, and a
//! hard failure under `BRAIN_REQUIRE_FIXTURES=1`. The one thing it can never be
//! is silent.
//!
//! # The ratchet
//!
//! Goldens dumped before this convention have no `source` block, and a test
//! that cannot verify its pairing must not pretend it did. So an absent block
//! prints `UNVERIFIED GOLDEN SOURCE` on every run, and setting
//! `BRAIN_REQUIRE_GOLDEN_SOURCE=1` turns it into a failure. That is the same
//! shape as the clippy ratchet: the end state is enforced by a flag that can be
//! switched on suite by suite as each dumper is re-run, rather than by a flag
//! day that takes every suite red at once.
//! `scripts/gates/check-golden-source.sh` keeps NEW dumpers from regressing.

use std::path::{Path, PathBuf};

/// The `source` block of a golden `manifest.json`, or the absence of one.
pub struct Source {
    manifest: PathBuf,
    /// The dumper that regenerates this golden, for the message a mismatch
    /// prints - the reader is being told what to re-run.
    dumper: String,
    source: Option<serde_json::Value>,
}

impl Source {
    /// Read `<dir>/manifest.json`. `dumper` is the `tools/goldens/…` script
    /// that produces this golden, quoted verbatim in every message.
    ///
    /// `None` (having already called [`crate::skip`], so it has named itself
    /// and is fatal under `BRAIN_REQUIRE_FIXTURES=1`) when the manifest is
    /// absent or unparseable: a golden whose provenance cannot be read is a
    /// fixture problem, and this helper owns that skip so the caller can write
    /// the bare `let Some(src) = Source::open(..) else { return };`.
    pub fn open(dir: &Path, dumper: &str) -> Option<Source> {
        Self::open_manifest(&dir.join("manifest.json"), dumper)
    }

    /// [`Source::open`] for a manifest that is not named `manifest.json` -
    /// Kronos writes `t_meta.json`, and a suite that already reads run
    /// parameters out of its own metadata file keeps one file, not two.
    pub fn open_manifest(manifest: &Path, dumper: &str) -> Option<Source> {
        let raw = match std::fs::read(manifest) {
            Ok(r) => r,
            Err(e) => {
                crate::skip(&format!("{}: {e} (run {dumper})", manifest.display()));
                return None;
            }
        };
        let json: serde_json::Value = match serde_json::from_slice(&raw) {
            Ok(j) => j,
            Err(e) => {
                crate::skip(&format!("{}: not valid JSON ({e}); re-run {dumper}", manifest.display()));
                return None;
            }
        };
        // Accept the block at the top level or nested under "source", so a
        // dumper whose metadata file IS the source record needs no wrapper.
        let source = json.get("source").cloned().filter(|v| v.is_object());
        Some(Source { manifest: manifest.to_path_buf(), dumper: dumper.to_string(), source })
    }

    /// The recorded identity of the checkpoint that produced this golden, or
    /// `None` when the dump predates the convention.
    pub fn identity(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.source.as_ref()?.get("identity")?.as_object()
    }

    /// The upstream reference the dumper was pointed at, purely informational.
    pub fn checkpoint(&self) -> Option<&str> {
        self.source.as_ref()?.get("checkpoint")?.as_str()
    }

    /// Demand that every named field of the recorded identity equals what the
    /// checkpoint under test reports. `true` when the pairing is proven.
    ///
    /// `false` means the golden and the checkpoint do not belong together, so
    /// this run may not compare anything - and it has already said so through
    /// [`crate::skip`], which makes it fatal under `BRAIN_REQUIRE_FIXTURES=1`.
    /// A dump that predates the convention returns `true` after printing
    /// `UNVERIFIED GOLDEN SOURCE`, because an unproven pairing is weaker
    /// evidence rather than none; `BRAIN_REQUIRE_GOLDEN_SOURCE=1` promotes that
    /// to a failure (see the ratchet in the module docs).
    ///
    /// Pass the fields that FIX THE TENSOR SHAPES (width, depth, head count,
    /// vocab), not every config key: a field the dumper does not record cannot
    /// be checked, and a field that does not change a shape does not
    /// distinguish a tier.
    ///
    /// ```no_run
    /// # use std::path::Path;
    /// # struct Cfg { d_model: usize, n_layers: usize }
    /// # let cfg = Cfg { d_model: 512, n_layers: 12 };
    /// let dir = Path::new("testdata/kronos");
    /// let Some(src) = brain_testutil::golden::Source::open(dir, "tools/goldens/kronos_dump_reference.py")
    /// else { return };
    /// if !src.require(&[("d_model", cfg.d_model as i64), ("n_layers", cfg.n_layers as i64)]) {
    ///     return;
    /// }
    /// ```
    pub fn require(&self, actual: &[(&str, i64)]) -> bool {
        let Some(identity) = self.identity() else {
            return self.unverified(actual);
        };
        let mut mismatches = Vec::new();
        let mut unrecorded = Vec::new();
        for (field, got) in actual {
            match identity.get(*field).and_then(serde_json::Value::as_i64) {
                Some(want) if want == *got => {}
                Some(want) => mismatches.push(format!("{field}: golden={want} checkpoint={got}")),
                None => unrecorded.push(*field),
            }
        }
        if !mismatches.is_empty() {
            let named = self.checkpoint().unwrap_or("an unrecorded checkpoint");
            crate::skip(&format!(
                "golden/checkpoint MISMATCH - {} was dumped from {named} ({}), \
                 but the checkpoint this run loaded is a different one. \
                 Re-dump with `{}` against THIS checkpoint, or point the test at the tier the golden came from.",
                self.manifest.display(),
                mismatches.join(", "),
                self.dumper,
            ));
            return false;
        }
        if !unrecorded.is_empty() {
            // A partial identity is a half-proof. Say so rather than counting
            // the fields that did match as a pass.
            return self.unverified_fields(&unrecorded);
        }
        true
    }

    fn unverified(&self, actual: &[(&str, i64)]) -> bool {
        let fields: Vec<&str> = actual.iter().map(|(f, _)| *f).collect();
        self.report_unverified(&format!(
            "{} carries no `source.identity` block, so nothing proves this golden was dumped \
             from the checkpoint under test ({})",
            self.manifest.display(),
            fields.join(", "),
        ))
    }

    fn unverified_fields(&self, unrecorded: &[&str]) -> bool {
        self.report_unverified(&format!(
            "{}'s `source.identity` does not record {}, so the pairing is only partly proven",
            self.manifest.display(),
            unrecorded.join(", "),
        ))
    }

    fn report_unverified(&self, what: &str) -> bool {
        let msg = format!("UNVERIFIED GOLDEN SOURCE: {what}. Re-dump with `{}`.", self.dumper);
        if std::env::var_os("BRAIN_REQUIRE_GOLDEN_SOURCE")
            .is_some_and(|v| !v.is_empty() && v != "0")
        {
            panic!("BRAIN_REQUIRE_GOLDEN_SOURCE is set: {msg}");
        }
        eprintln!("{msg}");
        // The comparison itself is still worth running - an unverified pairing
        // is weaker evidence, not no evidence, and the line above says which.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, body: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join("manifest.json");
        std::fs::write(&p, body).unwrap();
        p
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("brain-golden-source-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    const DUMPER: &str = "tools/goldens/x_dump_reference.py";
    const FIXTURES: &str = "BRAIN_REQUIRE_FIXTURES";
    const GOLDEN: &str = "BRAIN_REQUIRE_GOLDEN_SOURCE";

    #[test]
    fn a_matching_identity_proves_the_pairing() {
        let d = tmp("match");
        write(&d, r#"{"source":{"checkpoint":"V/R","identity":{"d_model":512,"n_layers":12}}}"#);
        let src = Source::open(&d, DUMPER).expect("manifest present");
        assert_eq!(src.checkpoint(), Some("V/R"));
        assert!(env(&[], || src.require(&[("d_model", 512), ("n_layers", 12)])));
    }

    /// The whole point: a golden from another tier must not be usable, and the
    /// refusal must be a named skip rather than a shape error later on.
    #[test]
    fn a_different_tier_is_refused() {
        let d = tmp("mismatch");
        write(&d, r#"{"source":{"checkpoint":"V/small","identity":{"d_model":512,"n_layers":12}}}"#);
        let src = Source::open(&d, DUMPER).expect("manifest present");
        assert!(!env(&[], || src.require(&[("d_model", 768), ("n_layers", 12)])));
    }

    /// And under the fixture flag it is not even a skip - it is red.
    #[test]
    fn a_different_tier_is_fatal_under_require_fixtures() {
        let d = tmp("mismatch-strict");
        write(&d, r#"{"source":{"identity":{"d_model":512}}}"#);
        let src = Source::open(&d, DUMPER).expect("manifest present");
        assert!(
            panics(&[(FIXTURES, "1")], || src.require(&[("d_model", 768)])),
            "a tier mismatch must fail, not skip, under {FIXTURES}"
        );
    }

    #[test]
    fn a_pre_convention_dump_runs_but_says_it_is_unverified() {
        let d = tmp("legacy");
        write(&d, r#"{"tensors":{"a":[1,2]}}"#);
        let src = Source::open(&d, DUMPER).expect("manifest present");
        assert!(src.identity().is_none());
        assert!(
            env(&[], || src.require(&[("d_model", 512)])),
            "unverified is weaker evidence, not no evidence"
        );
    }

    #[test]
    fn the_ratchet_can_be_turned_on() {
        let d = tmp("ratchet");
        write(&d, r#"{"tensors":{}}"#);
        let src = Source::open(&d, DUMPER).expect("manifest present");
        assert!(
            panics(&[(GOLDEN, "1")], || src.require(&[("d_model", 512)])),
            "{GOLDEN}=1 must refuse an unverifiable pairing"
        );
    }

    /// A field the checkpoint has and the golden never recorded is a half
    /// proof, and counting it as a whole one is how the hole reopens.
    #[test]
    fn a_partial_identity_is_reported_as_partial() {
        let d = tmp("partial");
        write(&d, r#"{"source":{"identity":{"d_model":512}}}"#);
        let src = Source::open(&d, DUMPER).expect("manifest present");
        assert!(
            panics(&[(GOLDEN, "1")], || src.require(&[("d_model", 512), ("n_layers", 12)])),
            "an unrecorded field must not count as a match"
        );
    }

    /// Run `f` with exactly `set` in the environment: both flags this module
    /// reads are cleared first, then `set` is applied, and everything is put
    /// back on the way out.
    ///
    /// Every test here shares one process, cargo runs them on parallel threads,
    /// and both flags are read from the process environment - so a test that
    /// merely set its own flag would be read by whichever other test happened
    /// to be inside `require` at that moment. Hence one lock around all of
    /// them, and a `Drop` (not a post-call restore) to put things back, since
    /// half of these tests exist to provoke a panic and a post-call restore
    /// would be skipped by the unwind exactly when the assertion held.
    fn env<T>(set: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());

        struct Restore(Vec<(&'static str, Option<std::ffi::OsString>)>);
        impl Drop for Restore {
            fn drop(&mut self) {
                for (k, v) in self.0.drain(..) {
                    match v {
                        Some(p) => std::env::set_var(k, p),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }

        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _restore = Restore(
            [FIXTURES, GOLDEN].iter().map(|k| (*k, std::env::var_os(k))).collect(),
        );
        for k in [FIXTURES, GOLDEN] {
            std::env::remove_var(k);
        }
        for (k, v) in set {
            std::env::set_var(k, v);
        }
        f()
    }

    /// `true` when `f` panicked under `set`.
    ///
    /// The panic hook is process-global too, so swapping it for a silent one
    /// (these panics are the expected result, not a failure worth a backtrace)
    /// happens INSIDE `env`'s lock, not around it.
    fn panics(set: &[(&str, &str)], f: impl FnOnce() -> bool) -> bool {
        env(set, || {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            std::panic::set_hook(prev);
            out.is_err()
        })
    }
}
