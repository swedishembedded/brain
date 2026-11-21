// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model distribution for training and serving
// clusters for its clients. If your team needs expertise in getting large
// checkpoints onto many nodes quickly and verifiably, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Node-local staging of model-store entries.
//!
//! On a cluster every rank reading a 50 GB base from the shared filesystem at
//! once is slow, and every node has fast local disk. [`stage`] copies the
//! store entry (`<vendor>/<repo>`) a path lies in to a local directory once
//! per node and answers the same path inside the copy: the first process on a
//! node takes the entry's lock and copies, every other process on the node
//! blocks on the lock and then finds the copy complete.
//!
//! Every file is checked as it is copied: its SHA-256 is computed from the
//! bytes written and compared with the source's digest (the store's validated
//! sidecar when there is one, otherwise the source read again), and a mismatch
//! fails the stage and leaves nothing behind. The copy is built in a private
//! temporary directory and renamed into place only once complete, so a crash
//! leaves either the old state or a whole copy, never a partial one.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest as _, Sha256};

/// The marker a complete staged copy carries.
const COMPLETE: &str = ".staged";

/// How many files are copied at once: enough to keep a parallel filesystem and a local NVMe busy.
const PARALLEL_FILES: usize = 8;

/// `path` inside a node-local copy of its store entry, staging the entry under `local` first when this node has none
/// yet. A path that is not inside a `<vendor>/<repo>` entry of `store_root` is answered unchanged.
pub fn stage(store_root: &Path, path: &Path, local: &Path) -> io::Result<PathBuf> {
    let Ok(rel) = path.strip_prefix(store_root) else { return Ok(path.to_path_buf()) };
    let parts: Vec<&std::ffi::OsStr> = rel.components().filter_map(|c| if let Component::Normal(p) = c { Some(p) } else { None }).collect();
    if parts.len() < 2 || rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Ok(path.to_path_buf());
    }
    let entry = Path::new(parts[0]).join(parts[1]);
    let staged = stage_entry(&store_root.join(&entry), &local.join(&entry))?;
    Ok(staged.join(rel.strip_prefix(&entry).expect("the entry is a prefix of the path")))
}

/// Copy the directory `source` to `dest` once (see the module documentation); answers `dest`.
fn stage_entry(source: &Path, dest: &Path) -> io::Result<PathBuf> {
    if dest.join(COMPLETE).is_file() {
        return Ok(dest.to_path_buf());
    }
    let parent = dest.parent().ok_or_else(|| io::Error::other(format!("{} has no parent", dest.display())))?;
    fs::create_dir_all(parent)?;
    let name = dest.file_name().ok_or_else(|| io::Error::other(format!("{} has no name", dest.display())))?.to_string_lossy().into_owned();
    let lock = File::create(parent.join(format!(".{name}.lock")))?;
    // Every other process staging this entry on this node waits here, then finds the copy complete.
    lock.lock()?;
    if dest.join(COMPLETE).is_file() {
        return Ok(dest.to_path_buf());
    }
    let temp = parent.join(format!(".{name}.staging-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp);
    let copied = copy_tree(source, &temp).and_then(|()| {
        File::create(temp.join(COMPLETE))?.sync_all()?;
        // A copy a crashed stager left complete but unrenamed is replaced, never merged.
        let _ = fs::remove_dir_all(dest);
        fs::rename(&temp, dest)
    });
    if copied.is_err() {
        let _ = fs::remove_dir_all(&temp);
    }
    copied.map(|()| dest.to_path_buf())
}

/// Every regular file under `source` (recursively), relative to it.
fn files(source: &Path, rel: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(source.join(rel))? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = rel.join(entry.file_name());
        // A symlink inside a store entry points at the same bytes; copy what it names.
        if kind.is_dir() || (kind.is_symlink() && fs::metadata(entry.path())?.is_dir()) {
            files(source, &path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

fn copy_tree(source: &Path, dest: &Path) -> io::Result<()> {
    let mut all = Vec::new();
    files(source, Path::new(""), &mut all)?;
    // The largest first, so the parallel copies finish together.
    all.sort_by_key(|p| std::cmp::Reverse(fs::metadata(source.join(p)).map(|m| m.len()).unwrap_or(0)));
    for dir in all.iter().filter_map(|p| p.parent()) {
        fs::create_dir_all(dest.join(dir))?;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..PARALLEL_FILES.min(all.len().max(1)))
            .map(|_| {
                scope.spawn(|| -> io::Result<()> {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(rel) = all.get(i) else { return Ok(()) };
                        copy_verified(&source.join(rel), &dest.join(rel))?;
                    }
                })
            })
            .collect();
        workers.into_iter().try_for_each(|w| w.join().unwrap_or_else(|_| Err(io::Error::other("a staging thread panicked"))))
    })
}

/// Copy one file, hashing what is written, and refuse it unless the digest is the source's.
fn copy_verified(source: &Path, dest: &Path) -> io::Result<()> {
    // A sidecar of the store is itself copied, and has no sidecar of its own to check against.
    let is_sidecar = source.extension().is_some_and(|e| e == "sha256");
    let expected = if is_sidecar { None } else { Some(crate::fetch::file_digest_sidecar(source)?) };
    let mut from = File::open(source)?;
    let mut to = File::create(dest)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = match from.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        hasher.update(&buffer[..n]);
        to.write_all(&buffer[..n])?;
    }
    to.sync_all()?;
    let got = format!("sha256:{}", crate::fetch::hex_lower(&hasher.finalize()));
    match expected {
        Some(expected) if expected != got => Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: copied {got}, the store holds {expected}", source.display()))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(root: &Path) {
        let repo = root.join("acme/model");
        fs::create_dir_all(repo.join("sub")).unwrap();
        fs::write(repo.join("config.json"), b"{\"x\":1}").unwrap();
        fs::write(repo.join("weights.safetensors"), vec![7u8; 3 << 20]).unwrap();
        fs::write(repo.join("sub/tokenizer.json"), b"{}").unwrap();
    }

    #[test]
    fn concurrent_ranks_stage_one_verified_copy_and_read_from_it() {
        let dir = brain_testutil::Scratch::dir("stage-concurrent");
        let (store, local) = (dir.path().join("store"), dir.path().join("local"));
        tree(&store);
        let wanted = store.join("acme/model/sub/tokenizer.json");
        let got: Vec<PathBuf> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4).map(|_| s.spawn(|| stage(&store, &wanted, &local).unwrap())).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for path in &got {
            assert_eq!(path, &local.join("acme/model/sub/tokenizer.json"));
        }
        assert_eq!(fs::read(local.join("acme/model/weights.safetensors")).unwrap(), vec![7u8; 3 << 20]);
        assert!(local.join("acme/model").join(COMPLETE).is_file());
        // Nothing of a staging attempt is left beside the copy.
        let leftovers: Vec<_> = fs::read_dir(local.join("acme")).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.contains("staging")).collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn a_path_outside_the_store_is_answered_unchanged() {
        let dir = brain_testutil::Scratch::dir("stage-outside");
        let elsewhere = dir.path().join("data/train.jsonl");
        assert_eq!(stage(&dir.path().join("store"), &elsewhere, &dir.path().join("local")).unwrap(), elsewhere);
        assert!(!dir.path().join("local").exists(), "nothing is staged for it");
    }

    #[test]
    fn a_copy_that_does_not_match_the_stores_digest_is_refused_and_removed() {
        let dir = brain_testutil::Scratch::dir("stage-mismatch");
        let (store, local) = (dir.path().join("store"), dir.path().join("local"));
        tree(&store);
        let weights = store.join("acme/model/weights.safetensors");
        // A sidecar is trusted only for a file whose last change is more than a timestamp tick (a second) older.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        // The store's recorded digest names other bytes: the sidecar a corrupted file would disagree with.
        crate::fetch::file_digest_sidecar(&weights).unwrap();
        let sidecar = fs::read_to_string(store.join("acme/model/weights.safetensors.sha256")).unwrap();
        let forged = format!("sha256:{}{}", "0".repeat(64), &sidecar[sidecar.find('\t').unwrap()..]);
        fs::write(store.join("acme/model/weights.safetensors.sha256"), forged).unwrap();
        let err = stage(&store, &store.join("acme/model/config.json"), &local).expect_err("a mismatch must fail");
        assert!(err.to_string().contains("the store holds"), "{err}");
        assert!(!local.join("acme/model").exists(), "no partial copy is left");
    }
}
