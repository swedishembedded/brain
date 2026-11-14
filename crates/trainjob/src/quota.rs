// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage budgets for shared training servers,
// so that one job can never fill the disk the other jobs depend on, for its
// clients. If your team needs expertise in resource governance for multi-tenant
// model training, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The storage an operator lets the training service use, as explicit numbers.
//!
//! * a per-artifact maximum: no single published file is larger;
//! * a whole-store maximum: the artifact store never holds more in total;
//! * a per-job staging maximum: what one job keeps in its private directory
//!   (staged datasets, checkpoints, temporaries) stays under it;
//! * a free-disk high-watermark: while the filesystems the service writes to
//!   have less than this free, new work is refused at admission.
//!
//! Space is reserved before it is used and every reservation is a guard that
//! gives its bytes back when it is dropped, so a failure, a cancellation or a
//! panic cannot leak a claim on the store.

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const GIB: u64 = 1 << 30;

/// The storage limits of one serving process. The defaults suit a node that dedicates a few
/// terabytes to training; an operator tightens them to the disk the service actually owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quotas {
    /// The most bytes one published artifact may hold.
    pub max_artifact_bytes: u64,
    /// The most bytes the artifact store may hold in total.
    pub max_store_bytes: u64,
    /// The most bytes one job may keep in its private directory.
    pub max_job_staging_bytes: u64,
    /// Work is refused while a filesystem it writes to has less than this many bytes free; 0 turns the check off.
    pub min_free_bytes: u64,
}

impl Default for Quotas {
    fn default() -> Quotas {
        Quotas { max_artifact_bytes: 64 * GIB, max_store_bytes: 1024 * GIB, max_job_staging_bytes: 128 * GIB, min_free_bytes: 10 * GIB }
    }
}

/// How much room a filesystem has: injectable so admission can be tested without filling a disk.
pub trait FreeSpace: Send + Sync + fmt::Debug {
    /// The bytes available to this process on the filesystem holding `path` (or its nearest existing ancestor).
    fn available(&self, path: &Path) -> std::io::Result<u64>;
}

/// The real filesystem.
#[derive(Debug, Default)]
pub struct DiskFree;

impl FreeSpace for DiskFree {
    #[cfg(unix)]
    fn available(&self, path: &Path) -> std::io::Result<u64> {
        use std::os::unix::ffi::OsStrExt;
        let existing = path.ancestors().find(|p| p.exists()).unwrap_or(Path::new("/"));
        let c_path = std::ffi::CString::new(existing.as_os_str().as_bytes()).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let mut stat = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
        // SAFETY: `c_path` is a valid NUL-terminated string and `stat` is a writable `statvfs`.
        if unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `statvfs` succeeded, so it initialised the struct.
        let stat = unsafe { stat.assume_init() };
        // The field widths differ by platform (u64 on Linux, narrower on macOS).
        #[allow(clippy::unnecessary_cast)]
        let (blocks, block_bytes) = (stat.f_bavail as u64, stat.f_frsize as u64);
        Ok(blocks.saturating_mul(block_bytes))
    }

    /// The free-space check is not implemented off Unix: the watermark never refuses there.
    #[cfg(not(unix))]
    fn available(&self, _path: &Path) -> std::io::Result<u64> {
        Ok(u64::MAX)
    }
}

/// Refuses work when any of `paths` sits on a filesystem with less than `quotas.min_free_bytes` free.
pub fn admit_free_space(probe: &dyn FreeSpace, quotas: &Quotas, paths: &[&Path]) -> Result<(), String> {
    if quotas.min_free_bytes == 0 {
        return Ok(());
    }
    for path in paths {
        let free = probe.available(path).map_err(|e| {
            eprintln!("brain train: free space of {}: {e}", path.display());
            "train: the free disk space of this server could not be read".to_string()
        })?;
        if free < quotas.min_free_bytes {
            return Err(format!(
                "train: this server is low on disk ({free} bytes free, the floor is {} bytes) and takes no new training work until space is freed",
                quotas.min_free_bytes
            ));
        }
    }
    Ok(())
}

/// What a job has kept in its private directory against its [`Quotas::max_job_staging_bytes`].
#[derive(Debug)]
pub struct Staging {
    limit: u64,
    used: AtomicU64,
}

impl Staging {
    pub fn new(limit: u64) -> Staging {
        Staging { limit, used: AtomicU64::new(0) }
    }

    /// The bytes the job may still stage.
    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.used.load(Ordering::Relaxed))
    }

    /// Claims `bytes` of the budget, or says the job is over it; a refused charge claims nothing.
    pub fn charge(&self, bytes: u64) -> Result<(), String> {
        if bytes > self.remaining() {
            return Err(self.over());
        }
        self.used.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    /// The error for a job that went over its budget.
    pub fn over(&self) -> String {
        format!("train: the job needs more than the {} bytes of staging space a job may keep", self.limit)
    }

    /// The most bytes the job may keep.
    pub fn limit(&self) -> u64 {
        self.limit
    }
}

/// The bytes of the artifact store: what is published and what running publishes have claimed.
#[derive(Debug)]
pub(crate) struct StoreLedger {
    limit: u64,
    books: Mutex<Books>,
}

#[derive(Debug, Default)]
struct Books {
    /// Bytes of published artifacts, refreshed from the directory when a publish begins so files other
    /// nodes sharing the store added are counted.
    committed: u64,
    /// Bytes claimed by publishes that have not finished.
    in_flight: u64,
}

impl StoreLedger {
    pub(crate) fn new(limit: u64) -> StoreLedger {
        StoreLedger { limit, books: Mutex::new(Books::default()) }
    }

    pub(crate) fn limit(&self) -> u64 {
        self.limit
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Books> {
        self.books.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Starts a publish into the store at `root`: counts what the directory holds now and refuses when it is already full.
    pub(crate) fn begin(&self, root: &Path) -> Result<Claim<'_>, StoreFull> {
        let mut books = self.lock();
        books.committed = published_bytes(root);
        if books.committed >= self.limit {
            return Err(StoreFull);
        }
        Ok(Claim { ledger: self, held: 0 })
    }

    /// Runs `publish` (the rename that makes an artifact appear) and, when it succeeded, counts `added` bytes as
    /// committed and returns the claim, all under the ledger's lock so no other publish counts the artifact
    /// twice or not at all.
    pub(crate) fn commit<T>(&self, claim: &mut Claim<'_>, added: u64, publish: impl FnOnce() -> std::io::Result<T>) -> std::io::Result<T> {
        let mut books = self.lock();
        let value = publish()?;
        books.committed += added;
        books.in_flight -= claim.held;
        claim.held = 0;
        Ok(value)
    }
}

/// The store has no room for any more.
#[derive(Debug)]
pub(crate) struct StoreFull;

/// Bytes of the store a running publish has claimed; given back when it is dropped.
#[derive(Debug)]
pub(crate) struct Claim<'a> {
    ledger: &'a StoreLedger,
    held: u64,
}

impl Claim<'_> {
    /// Claims `bytes` more, or fails when the store would then hold more than its maximum.
    pub(crate) fn grow(&mut self, bytes: u64) -> Result<(), StoreFull> {
        let mut books = self.ledger.lock();
        if books.committed.saturating_add(books.in_flight).saturating_add(bytes) > self.ledger.limit {
            return Err(StoreFull);
        }
        books.in_flight += bytes;
        self.held += bytes;
        Ok(())
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.ledger.lock().in_flight -= self.held;
    }
}

/// The bytes of the artifacts in `root`: the files named by a digest, not the temporaries of running publishes.
fn published_bytes(root: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(root) else { return 0 };
    entries
        .flatten()
        .filter(|e| e.file_name().to_str().is_some_and(|n| crate::store::Digest::parse(n).is_ok()))
        .filter_map(|e| e.metadata().ok().filter(|m| m.is_file()).map(|m| m.len()))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_refuses_the_charge_that_crosses_the_limit_and_keeps_what_it_has() {
        let staging = Staging::new(10);
        staging.charge(6).unwrap();
        assert!(staging.charge(5).is_err());
        assert_eq!(staging.remaining(), 4, "a refused charge claims nothing");
        staging.charge(4).unwrap();
    }

    #[derive(Debug)]
    struct Fixed(u64);
    impl FreeSpace for Fixed {
        fn available(&self, _: &Path) -> std::io::Result<u64> {
            Ok(self.0)
        }
    }

    #[test]
    fn work_is_refused_under_the_watermark_and_taken_at_it() {
        let quotas = Quotas { min_free_bytes: 100, ..Quotas::default() };
        let path = Path::new("/somewhere");
        let refusal = admit_free_space(&Fixed(99), &quotas, &[path]).unwrap_err();
        assert!(refusal.contains("low on disk"), "{refusal}");
        admit_free_space(&Fixed(100), &quotas, &[path]).unwrap();
        admit_free_space(&Fixed(0), &Quotas { min_free_bytes: 0, ..quotas }, &[path]).unwrap();
    }

    #[test]
    fn the_real_probe_reads_a_filesystem_and_a_path_that_does_not_exist_yet() {
        let dir = tempfile::tempdir().unwrap();
        assert!(DiskFree.available(dir.path()).unwrap() > 0);
        assert!(DiskFree.available(&dir.path().join("not").join("yet")).unwrap() > 0);
    }
}
