// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Whole-document atomic writer, adapted from `sven-input`'s
//! `chat_document.rs` YAML save path for JSON `Trajectory` documents.
//!
//! Two write entry points, mirroring `save_chat_to` / `save_chat_to_atomic`:
//! - [`write_trajectory`] - a plain `fs::write`, no concurrency guarantees.
//! - [`write_trajectory_atomic`] - temp file + `flock`-guarded sidecar lock
//!   + inode/mtime identity check + atomic `rename`.
//!
//! One deliberate adaptation from `chat_document.rs`: `save_chat_to_atomic`
//! re-stats the target file itself at call-entry, so it only catches a
//! modification that happens *during* the save call (a narrow window).
//! Here, [`write_trajectory_atomic`] instead takes the caller's previously
//! captured [`FileFingerprint`] (from [`read_trajectory_with_fingerprint`])
//! as an explicit `expected` parameter, so it can detect a modification that
//! happened any time between the caller's read and this write - which is
//! the guarantee a "read, edit, save" workflow actually needs.

use std::fs;
use std::path::Path;

use serde::Serialize;
use thiserror::Error;

use crate::model::Trajectory;

/// Errors from the atomic trajectory writer/reader.
#[derive(Debug, Error)]
pub enum PersistError {
    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialization failure.
    #[error("serialization error: {0}")]
    Json(#[from] serde_json::Error),
    /// The file changed (identity or mtime) since the caller's fingerprint
    /// was captured; the write was refused rather than clobbering it.
    #[error("file was modified by another process since it was read")]
    Conflict,
}

/// Identity snapshot of a file used to detect concurrent modification.
///
/// On Unix this is the real `(inode, mtime)` pair. On other platforms it
/// falls back to `(file size, mtime seconds)`, same as `chat_document.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileFingerprint {
    identity: u64,
    mtime: i64,
}

fn fingerprint_of(metadata: &fs::Metadata) -> FileFingerprint {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        FileFingerprint {
            identity: metadata.ino(),
            mtime: metadata.mtime(),
        }
    }
    #[cfg(not(unix))]
    {
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        FileFingerprint {
            identity: metadata.len(),
            mtime,
        }
    }
}

fn serialize_pretty(trajectory: &Trajectory) -> Result<String, PersistError> {
    let mut buf = Vec::new();
    let mut serializer =
        serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::PrettyFormatter::new());
    trajectory.serialize(&mut serializer)?;
    Ok(String::from_utf8(buf).expect("serde_json always produces valid UTF-8"))
}

/// Write a trajectory to `path`, overwriting whatever is there. No atomicity
/// or concurrent-modification guarantees - mirrors `chat_document.rs`'s
/// `save_chat_to`.
pub fn write_trajectory(path: &Path, trajectory: &Trajectory) -> Result<(), PersistError> {
    let content = serialize_pretty(trajectory)?;
    fs::write(path, content)?;
    Ok(())
}

/// Read a trajectory from `path` along with a [`FileFingerprint`] snapshot,
/// for later use with [`write_trajectory_atomic`]'s `expected` parameter.
pub fn read_trajectory_with_fingerprint(
    path: &Path,
) -> Result<(Trajectory, FileFingerprint), PersistError> {
    let metadata = fs::metadata(path)?;
    let content = fs::read_to_string(path)?;
    let trajectory: Trajectory = serde_json::from_str(&content)?;
    Ok((trajectory, fingerprint_of(&metadata)))
}

/// Write a trajectory to `path` atomically:
///
/// 1. Take an exclusive `flock` on a sidecar lock file (Unix; on other
///    platforms writers are only atomic against readers, not each other).
/// 2. While holding the lock, compare the target's current
///    [`FileFingerprint`] against `expected`. Mismatch (or an unexpected
///    appearance/disappearance of the file) fails with
///    [`PersistError::Conflict`] instead of overwriting.
/// 3. Serialize to a **writer-unique** temp file in the same directory (so
///    `rename` stays on one filesystem, and concurrent writers can never
///    stomp on each other's temp file) and `fsync` it, so the rename can
///    never publish an empty/partial file after a crash.
/// 4. Replace the target via a single `rename()` (atomic on POSIX).
///
/// Pass `expected = None` to skip the concurrent-modification check
/// entirely (first write of a new file, or the caller doesn't care).
/// Otherwise pass the [`FileFingerprint`] returned by an earlier
/// [`read_trajectory_with_fingerprint`] call - a mismatch means someone else
/// wrote to the file after that read.
pub fn write_trajectory_atomic(
    path: &Path,
    trajectory: &Trajectory,
    expected: Option<&FileFingerprint>,
) -> Result<(), PersistError> {
    let content = serialize_pretty(trajectory)?;

    // Serialize writers first (Unix): the lock must be held across the
    // fingerprint check AND the temp-write+rename, or two writers race.
    #[cfg(unix)]
    let _guard = {
        use std::os::unix::io::AsRawFd;
        let lock_path = path.with_extension("json.lock");
        let lock_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        // SAFETY: `lock_file` stays open (and thus its fd valid) for the
        // duration of the flock/unlock pair; `LockGuard` releases it on drop
        // even if we return early via `?`.
        let ret = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        LockGuard(lock_file)
    };

    if let Some(expected) = expected {
        match fs::metadata(path) {
            Ok(current) => {
                if fingerprint_of(&current) != *expected {
                    return Err(PersistError::Conflict);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File vanished since the caller's read: also a conflict.
                return Err(PersistError::Conflict);
            }
            Err(e) => return Err(e.into()),
        }
    }

    let temp_path = unique_temp_path(path);
    let write_and_rename = || -> Result<(), PersistError> {
        {
            use std::io::Write;
            let mut temp = fs::File::create(&temp_path)?;
            temp.write_all(content.as_bytes())?;
            // Durability: the data must be on disk before the rename makes
            // it the document, or a crash could publish a truncated file.
            temp.sync_all()?;
        }
        fs::rename(&temp_path, path)?;
        Ok(())
    };
    if let Err(e) = write_and_rename() {
        let _ = fs::remove_file(&temp_path);
        return Err(e);
    }
    Ok(())
}

/// Remove a trajectory document together with its writer artifacts: the
/// `.json.lock` flock sidecar [`write_trajectory_atomic`] creates and any
/// orphaned `.json.tmp.*` temp files a crashed writer left behind. Callers
/// that delete trajectory files directly leak the sidecar - use this.
///
/// Missing files are not an error (idempotent); sidecar/temp cleanup is
/// best-effort and only the document's own removal can fail.
pub fn remove_trajectory(path: &Path) -> std::io::Result<()> {
    let _ = fs::remove_file(path.with_extension("json.lock"));
    if let (Some(parent), Some(temp_prefix)) = (
        path.parent(),
        path.with_extension("json.tmp.")
            .file_name()
            .map(|n| n.to_string_lossy().into_owned()),
    ) {
        if let Ok(entries) = fs::read_dir(parent) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(&temp_prefix)
                {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
    match fs::remove_file(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

/// A temp path beside `path` that no concurrent writer (thread or process)
/// can collide on: `<file>.json.tmp.<pid>.<seq>`. A deterministic shared
/// name (the old scheme) let one writer rename another's half-written temp
/// file - or fail because a sibling already renamed the shared temp away.
fn unique_temp_path(path: &Path) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("json.tmp.{}.{}", std::process::id(), seq))
}

#[cfg(unix)]
struct LockGuard(fs::File);

#[cfg(unix)]
impl Drop for LockGuard {
    fn drop(&mut self) {
        use std::os::unix::io::AsRawFd;
        // SAFETY: `self.0` is a valid, open file descriptor for the
        // lifetime of this guard; unlocking a lock we hold is always safe.
        unsafe {
            libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
        }
    }
}
