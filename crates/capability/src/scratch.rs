// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for servers that stage untrusted
// input on shared machines without exposing it to other local users for its
// clients. If your team needs expertise in hardening service file handling
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Where a served call stages its bytes on this machine.
//!
//! The system temporary directory is shared with every local user. A name in
//! it that is predictable (a pid, a timestamp) lets another user create it
//! first - a directory they own and can read, or a link to somewhere else -
//! and a file created with default permissions there is readable by all.
//! Everything brain stages for a request (an uploaded clip, a reference voice,
//! a training set, a compiled graph) goes through here instead: under one
//! per-user root only this user can enter, in a directory made for that one
//! use and refused if anything was there first.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Make `dir` (and its missing parents) a directory only this user can enter:
/// created `0700` when missing, refused when it exists but is a link, belongs
/// to another user, or others may write to it.
///
/// # Errors
///
/// The directory cannot be made, or what is there is not this user's own.
pub fn private_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt};
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
        let meta = std::fs::symlink_metadata(dir)?;
        // SAFETY: geteuid has no preconditions and cannot fail.
        let me = unsafe { libc::geteuid() };
        if !meta.is_dir() || meta.uid() != me || meta.mode() & 0o077 != 0 {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, format!("{} is not a directory only this user can enter", dir.display())));
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

/// This user's private root under the system temporary directory,
/// `<tmp>/brain-<uid>`, made and checked by [`private_dir`]. A cache that
/// outlives one call (a compiled graph) lives under it.
///
/// # Errors
///
/// As [`private_dir`].
pub fn private_root() -> io::Result<PathBuf> {
    #[cfg(unix)]
    // SAFETY: geteuid has no preconditions and cannot fail.
    let root = std::env::temp_dir().join(format!("brain-{}", unsafe { libc::geteuid() }));
    #[cfg(not(unix))]
    let root = std::env::temp_dir().join("brain");
    private_dir(&root)?;
    Ok(root)
}

/// A fresh directory for one use, `<private root>/<purpose>-<pid>-<n>`,
/// created exclusively (`0700`, refused if the name exists). The caller owns
/// it and removes it when the use ends.
///
/// # Errors
///
/// As [`private_root`], or the directory cannot be made.
pub fn scratch_dir(purpose: &str) -> io::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = private_root()?;
    loop {
        let dir = root.join(format!("{purpose}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        match builder.create(&dir) {
            Ok(()) => return Ok(dir),
            // Left by an earlier process with this pid: take the next name.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn scratch_dirs_are_private_fresh_and_distinct() {
        let (a, b) = (scratch_dir("test").unwrap(), scratch_dir("test").unwrap());
        assert_ne!(a, b);
        for dir in [&a, &b] {
            assert_eq!(std::fs::metadata(dir).unwrap().permissions().mode() & 0o777, 0o700);
            std::fs::remove_dir(dir).unwrap();
        }
    }

    #[test]
    fn a_directory_others_may_enter_is_refused() {
        let dir = std::env::temp_dir().join(format!("brain-scratch-open-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(private_dir(&dir).unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        std::fs::remove_dir(&dir).unwrap();
    }
}
