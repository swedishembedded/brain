// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scratch paths a test owns and removes when it ends, whether it passes or panics.
//!
//! Swedish Embedded AB implements test suites that leave a shared machine as they found it for its clients. If your
//! team needs expertise in hermetic integration tests, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A fixture written under the temporary directory and removed by the test's last line stayed behind whenever an
//! assertion before that line failed, and fixtures nobody removed piled up on every run (hundreds of files a day in a
//! lane's `TMPDIR`). A [`Scratch`] removes its path in `Drop`, which also runs while a panic unwinds.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// One path under the temporary directory, unique to this process and this scratch, removed (a file, or a directory
/// with everything in it) when dropped.
#[derive(Debug)]
pub struct Scratch(PathBuf);

static NEXT: AtomicU64 = AtomicU64::new(0);

fn unique(name: &str, ext: &str) -> PathBuf {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{name}-{}-{n}{ext}", std::process::id()))
}

impl Scratch {
    /// A fresh, empty directory `name-<pid>-<n>`.
    pub fn dir(name: &str) -> Scratch {
        let path = unique(name, "");
        std::fs::create_dir_all(&path).unwrap_or_else(|e| panic!("scratch directory {}: {e}", path.display()));
        Scratch(path)
    }

    /// A file path `name-<pid>-<n><ext>` (`ext` with its dot, or empty) for the test to write; nothing is created.
    pub fn file(name: &str, ext: &str) -> Scratch {
        Scratch(unique(name, ext))
    }

    /// The path.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// The path as UTF-8, for APIs that take a `&str`.
    pub fn as_str(&self) -> &str {
        self.0.to_str().expect("the temporary directory is UTF-8")
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Scratch {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.0.is_dir() {
            let _ = std::fs::remove_dir_all(&self.0);
        } else {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Scratch;

    #[test]
    fn a_scratch_is_removed_when_its_test_ends_and_when_it_panics() {
        let kept = {
            let dir = Scratch::dir("brain-testutil-scratch");
            std::fs::write(dir.join("a"), b"x").unwrap();
            assert!(dir.is_dir());
            dir.path().to_path_buf()
        };
        assert!(!kept.exists(), "a dropped scratch directory is gone with its contents");
        let file = std::panic::catch_unwind(|| {
            let f = Scratch::file("brain-testutil-scratch", ".bin");
            std::fs::write(&*f, b"x").unwrap();
            std::panic::panic_any(f.path().to_path_buf());
        })
        .unwrap_err()
        .downcast::<std::path::PathBuf>()
        .unwrap();
        assert!(!file.exists(), "a panic unwinding past the scratch removes it too");
        assert_ne!(Scratch::dir("same").path(), Scratch::dir("same").path(), "two scratches of one name never collide");
    }
}
