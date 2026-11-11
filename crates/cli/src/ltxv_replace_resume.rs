// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What `brain ltxv replace` may reuse from an earlier run in its work
//! directory.
//!
//! Swedish Embedded AB implements resumable generative video pipelines for
//! its clients, so that a run which fails late does not pay for the early
//! steps twice. If your team needs expertise in long-running GPU media
//! pipelines then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Every step writes its output and then a `<step>.key` file holding the
//! key it was produced under: a JSON text naming everything the output
//! depends on. A later run reuses the output only when the key it computes
//! now is byte-for-byte the recorded one, so an output is never reused for
//! different inputs or settings, and a step that died half way (no key
//! written yet) is simply done again.

use std::path::{Path, PathBuf};

pub(crate) struct Resume {
    dir: PathBuf,
    /// `--fresh`: reuse nothing.
    fresh: bool,
}

impl Resume {
    pub(crate) fn new(dir: &Path, fresh: bool) -> Resume {
        Resume { dir: dir.to_path_buf(), fresh }
    }

    fn key_path(&self, step: &str) -> PathBuf {
        self.dir.join(format!("{step}.key"))
    }

    /// Whether `step`'s output in the work directory was produced under `key`.
    pub(crate) fn holds(&self, step: &str, key: &str) -> bool {
        !self.fresh && std::fs::read_to_string(self.key_path(step)).is_ok_and(|recorded| recorded == key)
    }

    /// Record that `step`'s output, complete on disk, was produced under
    /// `key`. Call it after the output is written, never before.
    pub(crate) fn record(&self, step: &str, key: &str) -> Result<(), String> {
        std::fs::write(self.key_path(step), key).map_err(|e| format!("{}: {e}", self.key_path(step).display()))
    }
}

/// A file as a key names it: where it is, how big and when last changed. A
/// file replaced by another of the same name changes the key.
pub(crate) fn stamp(path: &str) -> serde_json::Value {
    let meta = std::fs::metadata(path).ok();
    let modified = meta.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos().to_string());
    serde_json::json!({ "path": path, "len": meta.map(|m| m.len()), "modified": modified })
}

/// FNV-1a of `bytes`, for keys that name pixels or masks without holding
/// them. Fixed by definition, so keys stay comparable across builds.
pub(crate) fn digest(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

pub(crate) fn digest_mask(mask: &[bool]) -> u64 {
    digest(mask.iter().map(|&m| m as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory for one test, removed when dropped.
    struct Work(PathBuf);

    impl Work {
        fn new(test: &str) -> Work {
            let dir = std::env::temp_dir().join(format!("brain-replace-resume-{test}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("temp dir");
            Work(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Work {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// An output is reused only under the key it was recorded with; a
    /// step with nothing recorded (it died before finishing) is redone.
    #[test]
    fn an_output_is_reused_only_under_the_key_it_was_made_with() {
        let dir = Work::new("keys");
        let resume = Resume::new(dir.path(), false);
        assert!(!resume.holds("generated", "a"), "nothing recorded yet");
        resume.record("generated", "a").unwrap();
        assert!(resume.holds("generated", "a"));
        assert!(!resume.holds("generated", "b"), "another key must not match");
        assert!(!resume.holds("actor", "a"), "another step must not match");
    }

    #[test]
    fn fresh_reuses_nothing() {
        let dir = Work::new("fresh");
        Resume::new(dir.path(), false).record("actor", "a").unwrap();
        assert!(!Resume::new(dir.path(), true).holds("actor", "a"));
    }

    #[test]
    fn a_digest_tells_masks_apart() {
        assert_eq!(digest_mask(&[true, false, true]), digest_mask(&[true, false, true]));
        assert_ne!(digest_mask(&[true, false, true]), digest_mask(&[true, true, false]));
        assert_ne!(digest_mask(&[true, false]), digest_mask(&[true, false, false]));
    }

    #[test]
    fn a_replaced_file_changes_its_stamp() {
        let dir = Work::new("stamp");
        let path = dir.path().join("f").to_string_lossy().into_owned();
        std::fs::write(&path, b"one").unwrap();
        let before = stamp(&path);
        std::fs::write(&path, b"three").unwrap();
        assert_ne!(before, stamp(&path));
    }
}
