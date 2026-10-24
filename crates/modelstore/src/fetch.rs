// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming, atomic file download: the mechanics shared by every [`crate::Hub`]
//! implementation that actually touches a network. Isolated from `hub.rs` so it
//! can be exercised with an in-memory [`std::io::Read`] in tests -- no live
//! server required to prove the atomic-write and sha256 behavior.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

/// Streams `reader` to `dest`, writing to a `.part` sibling and renaming into
/// place only on full success -- a killed or failed download never leaves a
/// partial file where [`crate::Store::scan`] could find it. Never buffers more
/// than one fixed-size chunk regardless of file size (the same OOM invariant
/// weight loading follows). `progress(got, total)` is called after each chunk;
/// `total` is `None` when the caller does not know the expected size.
///
/// **No byte-level resume.** The `.part` is `File::create`d, so a retry after
/// an interrupted transfer starts that file over rather than continuing it.
/// Resume is per FILE, one level up (`crate::plan`'s already-on-disk skip).
/// That was a fair trade while every large checkpoint arrived as a shard set:
/// an interrupted pull lost at most one shard. A GGUF release repo breaks the
/// assumption, since its whole artifact is ONE multi-gigabyte file and losing
/// it means losing the entire transfer. Fixing it needs an HTTP `Range`
/// request, which [`crate::Hub`] cannot express today -- its three methods are
/// list, read-a-whole-small-file, and stream-to-disk.
pub fn stream_to_file(
    mut reader: impl Read,
    dest: &Path,
    total: Option<u64>,
    expected_sha256: Option<&str>,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> std::io::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension(match dest.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{ext}.part"),
        None => "part".to_string(),
    });
    let mut file = std::fs::File::create(&tmp)?;
    let mut hasher = expected_sha256.is_some().then(Sha256::new);
    let mut buf = [0u8; 64 * 1024];
    let mut got: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        if let Some(h) = &mut hasher {
            h.update(&buf[..n]);
        }
        got += n as u64;
        progress(got, total);
    }
    file.sync_all()?;
    drop(file);
    if let (Some(expected), Some(h)) = (expected_sha256, hasher) {
        let digest = hex_lower(&h.finalize());
        if digest != expected {
            std::fs::remove_file(&tmp).ok();
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("sha256 mismatch: expected {expected}, got {digest}"),
            ));
        }
    }
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

/// The lowercase-hex SHA-256 of a file's bytes - the digest
/// [`stream_to_file`] checks a download against, so a digest recorded for a
/// file already on disk compares directly with the one its download was
/// verified by. Streams the file through a fixed-size buffer: a
/// multi-gigabyte checkpoint is never held in memory to hash it.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

/// Lowercase hex sha256 of `bytes` - [`sha256_file`] for data already in
/// memory.
pub fn bytes_digest(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

/// `sha256:<lowercase hex>` of a file's bytes: [`sha256_file`] tagged with
/// its algorithm. The one spelling of a weights file's identity - a
/// fine-tune's adapter digest, a loaded pipeline's identity and the adapter
/// a server reports it serves are all this string, so any two of them
/// compare directly.
pub fn file_digest(path: &Path) -> std::io::Result<String> {
    Ok(format!("sha256:{}", sha256_file(path)?))
}

/// [`file_digest`], reusing a validated digest recorded in a sidecar next to
/// the file (see [`base_digest_sidecar`] for the whole-base form).
///
/// The sidecar (`<file>.sha256`) records the digest with the length and
/// modification time in nanoseconds the file had when it was hashed. It is
/// trusted only when both still match the file, so a hit returns the digest
/// those bytes gave when they were hashed; a file whose length or marker moved
/// is hashed again. A same-length rewrite that also restores the modification
/// time is indistinguishable from an unchanged file - the limit of a stat
/// marker. When the filesystem reports no modification time there is no marker
/// to prove a hit, so every call recomputes and no sidecar is written.
///
/// A missing, unreadable, corrupt or mismatched sidecar costs one re-hash, and
/// the sidecar is replaced. Writes go to a per-process temporary and are
/// renamed into place, so several ranks hashing the same base at once can only
/// leave a complete sidecar behind.
pub fn file_digest_sidecar(path: &Path) -> std::io::Result<String> {
    let meta = std::fs::metadata(path)?;
    let Some((len, mtime)) = change_marker(&meta) else {
        return file_digest(path);
    };
    let sidecar = sidecar_path(path);
    if let Some(recorded) = read_sidecar(&sidecar, len, mtime) {
        return Ok(recorded);
    }
    let digest = file_digest(path)?;
    write_sidecar(&sidecar, &digest, len, mtime);
    Ok(digest)
}

/// The content digest naming a base: a file's own ([`file_digest`]), or for a
/// `transformers` directory the digest of its `config.json` and weight files'
/// digests in name order, so a change to any of them is a different base. A
/// fine-tune records it and a server recomputes it to bind an adapter to the
/// base it was trained on, so both must come from this one naming rule;
/// [`base_digest_sidecar`] is the cached form, answering the same digest.
pub fn base_digest(path: &Path) -> std::io::Result<String> {
    base_digest_with(path, &mut |p| file_digest(p))
}

/// [`base_digest`] with each file's digest reused from its validated sidecar
/// ([`file_digest_sidecar`]). Hashing a released checkpoint reads the whole
/// file (seconds per gigabyte), which every rank of a distributed fine-tune,
/// and every save of the same base, would otherwise repeat; the sidecar makes
/// the second and later reads a stat. The digest is the same one
/// [`base_digest`] returns - only the reading is skipped, and only while the
/// file's length and modification time still match.
pub fn base_digest_sidecar(path: &Path) -> std::io::Result<String> {
    base_digest_with(path, &mut |p| file_digest_sidecar(p))
}

/// [`base_digest`], with each file's digest kept in the cache file `cache`
/// under the file's identity - canonical path, size and modification time -
/// so a base is read and hashed once, and again only when its file changes.
/// Hashing a released checkpoint reads all of it (tens of gigabytes), which a
/// server naming its base at every start would otherwise repeat. The cache is
/// best effort: one that cannot be read or written costs a re-hash, never the
/// digest.
pub fn base_digest_cached(path: &Path, cache: &Path) -> std::io::Result<String> {
    let mut known = read_digest_cache(cache);
    let before = known.len();
    let digest = base_digest_with(path, &mut |p| {
        let Some(id) = file_identity(p) else { return file_digest(p) };
        if let Some(d) = known.get(&id) {
            return Ok(d.clone());
        }
        let d = file_digest(p)?;
        known.insert(id, d.clone());
        Ok(d)
    })?;
    if known.len() != before {
        write_digest_cache(cache, &known);
    }
    Ok(digest)
}

/// What a cached digest is valid for: the file's canonical path, size and
/// modification time in nanoseconds.
type FileIdentity = (String, u64, u128);

fn file_identity(p: &Path) -> Option<FileIdentity> {
    let (len, mtime) = change_marker(&std::fs::metadata(p).ok()?)?;
    Some((std::fs::canonicalize(p).ok()?.to_str()?.to_string(), len, mtime))
}

/// A file's change marker: its length and modification time in nanoseconds.
/// `None` when the platform reports no modification time, which callers read
/// as "no marker, so never trust a cached digest".
fn change_marker(meta: &std::fs::Metadata) -> Option<(u64, u128)> {
    let mtime = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
    Some((meta.len(), mtime))
}

/// Where a file's digest sidecar lives: the file's own name plus `.sha256`.
fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".sha256");
    PathBuf::from(name)
}

/// The digest a sidecar records, if it is a well-formed record that still
/// describes a file of `len` bytes modified at `mtime` (nanoseconds). Anything
/// else - unreadable, truncated, not lowercase hex, another length or another
/// marker - is `None`, which the caller answers by hashing the file again.
fn read_sidecar(sidecar: &Path, len: u64, mtime: u128) -> Option<String> {
    let text = std::fs::read_to_string(sidecar).ok()?;
    let mut fields = text.trim_end().split('\t');
    let digest = fields.next()?;
    let recorded_len: u64 = fields.next()?.parse().ok()?;
    let recorded_mtime: u128 = fields.next()?.parse().ok()?;
    if fields.next().is_some() || recorded_len != len || recorded_mtime != mtime {
        return None;
    }
    let hex = digest.strip_prefix("sha256:")?;
    (hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
        .then(|| digest.to_string())
}

/// Record `digest` for a `len`-byte file modified at `mtime`, atomically. A
/// failure costs the next call a re-hash, never the digest.
fn write_sidecar(sidecar: &Path, digest: &str, len: u64, mtime: u128) {
    write_atomic(sidecar, &format!("{digest}\t{len}\t{mtime}\n"));
}

/// Replace `path` with `contents` through a sibling temporary that is renamed
/// over it, so a concurrent reader sees the old file or the complete new one.
/// The temporary name carries the process id and a per-call sequence, so ranks
/// and threads writing the same path never share one.
fn write_atomic(path: &Path, contents: &str) -> bool {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let mut name = path.as_os_str().to_os_string();
    name.push(format!(".{}.{}.tmp", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
    let temporary = PathBuf::from(name);
    let written = std::fs::write(&temporary, contents).is_ok() && std::fs::rename(&temporary, path).is_ok();
    if !written {
        std::fs::remove_file(&temporary).ok();
    }
    written
}

/// The cache file's entries: one `digest<TAB>size<TAB>mtime_ns<TAB>path` a line.
fn read_digest_cache(cache: &Path) -> HashMap<FileIdentity, String> {
    let Ok(text) = std::fs::read_to_string(cache) else { return HashMap::new() };
    text.lines()
        .filter_map(|line| {
            let mut f = line.splitn(4, '\t');
            let (digest, len, mtime, path) = (f.next()?, f.next()?.parse().ok()?, f.next()?.parse().ok()?, f.next()?);
            digest.starts_with("sha256:").then(|| ((path.to_string(), len, mtime), digest.to_string()))
        })
        .collect()
}

/// Replace the cache file atomically (write a sibling, rename over it), so a
/// concurrent reader sees the old or the new set, never half of one.
fn write_digest_cache(cache: &Path, known: &HashMap<FileIdentity, String>) {
    let mut lines: Vec<String> = known.iter().map(|((path, len, mtime), d)| format!("{d}\t{len}\t{mtime}\t{path}")).collect();
    lines.sort();
    let parent_ready = cache.parent().is_none_or(|d| d.as_os_str().is_empty() || std::fs::create_dir_all(d).is_ok());
    if parent_ready {
        write_atomic(cache, &(lines.join("\n") + "\n"));
    }
}

/// [`base_digest`] over a per-file digest function.
fn base_digest_with(path: &Path, file_digest: &mut dyn FnMut(&Path) -> std::io::Result<String>) -> std::io::Result<String> {
    if !path.is_dir() {
        return file_digest(path);
    }
    let mut names: Vec<_> = std::fs::read_dir(path)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n == "config.json" || n.ends_with(".safetensors") || n.ends_with(".bin") || n.ends_with(".index.json")))
        .collect();
    names.sort();
    let mut combined = String::new();
    for p in &names {
        combined.push_str(&format!("{}={}\n", p.file_name().and_then(|n| n.to_str()).unwrap_or_default(), file_digest(p)?));
    }
    Ok(format!("sha256:{}", bytes_digest(combined.as_bytes())))
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_bytes_to_dest_atomically_via_rename() {
        let dir = std::env::temp_dir().join("modelstore-fetch-test-basic");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("weights.bin");
        let data = b"hello model weights";
        let mut seen = Vec::new();
        stream_to_file(&data[..], &dest, Some(data.len() as u64), None, &mut |got, total| {
            seen.push((got, total));
        })
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), data);
        assert!(!dest.with_extension("bin.part").exists());
        assert_eq!(seen.last(), Some(&(data.len() as u64, Some(data.len() as u64))));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sha256_mismatch_is_rejected_and_leaves_no_dest_file() {
        let dir = std::env::temp_dir().join("modelstore-fetch-test-sha-mismatch");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("weights.bin");
        let data = b"payload";
        let err = stream_to_file(&data[..], &dest, None, Some("0000000000000000000000000000000000000000000000000000000000000000"), &mut |_, _| {})
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(!dest.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sha256_match_renames_into_place() {
        let dir = std::env::temp_dir().join("modelstore-fetch-test-sha-ok");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("weights.bin");
        let data = b"";
        // sha256("") -- a known vector.
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        stream_to_file(&data[..], &dest, None, Some(expected), &mut |_, _| {}).unwrap();
        assert!(dest.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file hashes to the digest of its bytes, and a file that is not
    /// there is an error, not a digest.
    #[test]
    fn a_file_hashes_to_the_digest_of_its_bytes() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-sha-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.bin");
        std::fs::write(&path, b"abc").unwrap();
        // sha256("abc") -- the FIPS 180-2 example vector.
        assert_eq!(sha256_file(&path).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert!(sha256_file(&dir.join("absent.bin")).is_err());
        // The algorithm-tagged form every identity and record carries.
        assert_eq!(file_digest(&path).unwrap(), "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert!(file_digest(&dir.join("absent.bin")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A base is named by its file's digest, or for a `transformers`
    /// directory by its config and weight files together: a change to any of
    /// them is another base, and files that do not shape the model are not
    /// part of the name.
    #[test]
    fn a_base_directory_is_named_by_its_config_and_weights() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-base-digest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        let file = dir.join("model.safetensors");
        assert_eq!(base_digest(&file).unwrap(), file_digest(&file).unwrap(), "a file is named by its own digest");

        let named = base_digest(&dir).unwrap();
        assert!(named.starts_with("sha256:"), "{named}");
        assert_ne!(named, file_digest(&file).unwrap());

        std::fs::write(dir.join("README.md"), b"notes").unwrap();
        assert_eq!(base_digest(&dir).unwrap(), named, "a README does not shape the model");
        std::fs::write(dir.join("config.json"), b"{\"layers\":1}").unwrap();
        assert_ne!(base_digest(&dir).unwrap(), named, "a config change is another base");
        assert!(base_digest(&dir.join("absent")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A base is read and hashed once: the digest is kept in the cache file
    /// under the file's identity (path, size, modification time) and served
    /// from there until the file changes. A cache that cannot be written
    /// costs a re-hash, never the digest.
    #[test]
    fn a_base_is_hashed_once_and_again_only_when_its_file_changes() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-digest-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (file, cache) = (dir.join("weights.bin"), dir.join("cache").join("digests.txt"));
        std::fs::write(&file, b"abc").unwrap();
        let real = file_digest(&file).unwrap();
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), real);

        // A cached entry is what answers: forge it and the forgery comes back.
        let hex = real.trim_start_matches("sha256:");
        let forged = "0".repeat(64);
        std::fs::write(&cache, std::fs::read_to_string(&cache).unwrap().replace(hex, &forged)).unwrap();
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), format!("sha256:{forged}"), "the file was not read again");

        // Another file at the same path is hashed again.
        std::fs::write(&file, b"abcd").unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options().write(true).open(&file).unwrap().set_modified(later).unwrap();
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), file_digest(&file).unwrap());

        // A directory base caches each of its files; an unwritable cache still answers.
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        assert_eq!(base_digest_cached(&dir, &cache).unwrap(), base_digest(&dir).unwrap());
        assert_eq!(base_digest_cached(&file, &file.join("not-a-dir").join("digests.txt")).unwrap(), file_digest(&file).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Read and set the marker a sidecar test asserts against.
    fn marker_of(path: &Path) -> std::time::SystemTime {
        std::fs::metadata(path).unwrap().modified().unwrap()
    }

    fn set_marker(path: &Path, at: std::time::SystemTime) {
        std::fs::File::options().write(true).open(path).unwrap().set_modified(at).unwrap();
    }

    /// The first call hashes the file and records the digest with the length
    /// and nanosecond marker it was hashed at; the returned digest is the
    /// content digest.
    #[test]
    fn a_sidecar_is_written_on_the_first_call_and_records_the_digest() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-sidecar-first-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("weights.bin");
        std::fs::write(&file, b"abc").unwrap();
        let real = file_digest(&file).unwrap();
        assert_eq!(file_digest_sidecar(&file).unwrap(), real);

        let sidecar = sidecar_path(&file);
        let text = std::fs::read_to_string(&sidecar).unwrap();
        let fields: Vec<&str> = text.trim_end().split('\t').collect();
        assert_eq!(fields.len(), 3, "{text:?}");
        assert_eq!(fields[0], real);
        let (len, mtime) = change_marker(&std::fs::metadata(&file).unwrap()).unwrap();
        assert_eq!(fields[1], len.to_string());
        assert_eq!(fields[2], mtime.to_string());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A recorded digest answers only while the file's length and marker are
    /// still the ones it was hashed at. The reuse is proved without timing: the
    /// bytes are replaced with content that hashes differently while the
    /// original marker is restored, and the old digest still comes back - which
    /// can only happen if the file was not read. Moving the marker (same
    /// length) makes the next call hash the new bytes.
    #[test]
    fn a_sidecar_is_reused_only_while_length_and_marker_match() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-sidecar-reuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("weights.bin");
        std::fs::write(&file, b"abc").unwrap();
        let real = file_digest(&file).unwrap();
        let original = marker_of(&file);
        assert_eq!(file_digest_sidecar(&file).unwrap(), real);

        std::fs::write(&file, b"abd").unwrap();
        set_marker(&file, original);
        assert_ne!(file_digest(&file).unwrap(), real, "the bytes differ");
        assert_eq!(file_digest_sidecar(&file).unwrap(), real, "same length and marker: the recorded digest answers");

        let later = original + std::time::Duration::from_secs(5);
        set_marker(&file, later);
        assert_eq!(file_digest_sidecar(&file).unwrap(), file_digest(&file).unwrap(), "a moved marker is hashed again");
        assert_ne!(file_digest_sidecar(&file).unwrap(), real);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A sidecar that is not a well-formed record for this exact file is
    /// ignored and replaced: garbage, a truncation, a stale length and a stale
    /// marker all cost one re-hash and never a wrong digest.
    #[test]
    fn a_corrupt_or_mismatched_sidecar_is_ignored_and_rewritten() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-sidecar-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("weights.bin");
        std::fs::write(&file, b"abc").unwrap();
        let real = file_digest(&file).unwrap();
        let sidecar = sidecar_path(&file);
        let (len, mtime) = change_marker(&std::fs::metadata(&file).unwrap()).unwrap();
        for corrupt in [
            "not a digest".to_string(),
            format!("sha256:{}", "z".repeat(64)),
            format!("sha256:{}\t{}\t{mtime}\tunexpected", "0".repeat(64), len),
            format!("{}\t{}\t{mtime}", real, len + 1),
            format!("{}\t{len}\t{}", real, mtime + 1),
        ] {
            std::fs::write(&sidecar, &corrupt).unwrap();
            assert_eq!(file_digest_sidecar(&file).unwrap(), real, "{corrupt:?} was ignored");
            let repaired = std::fs::read_to_string(&sidecar).unwrap();
            assert_eq!(repaired, format!("{real}\t{len}\t{mtime}\n"), "{corrupt:?} was replaced");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Several writers hashing the same file at once leave one complete sidecar
    /// and no temporary behind: each temporary is per-process-unique across
    /// calls, and the rename publishes only a whole record.
    #[test]
    fn concurrent_writers_leave_one_complete_sidecar() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-sidecar-concurrent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("weights.bin");
        std::fs::write(&file, b"concurrent writers").unwrap();
        let real = file_digest(&file).unwrap();
        let (len, mtime) = change_marker(&std::fs::metadata(&file).unwrap()).unwrap();

        let expected = real.as_str();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let file = &file;
                scope.spawn(move || {
                    for _ in 0..8 {
                        assert_eq!(file_digest_sidecar(file).unwrap().as_str(), expected);
                    }
                });
            }
        });

        let sidecar = sidecar_path(&file);
        assert_eq!(std::fs::read_to_string(&sidecar).unwrap(), format!("{real}\t{len}\t{mtime}\n"));
        let temporaries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(temporaries.is_empty(), "temporaries left behind: {temporaries:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A base directory reuses each file's sidecar while answering exactly the
    /// digest the uncached form gives.
    #[test]
    fn a_directory_base_reuses_each_files_sidecar() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-base-sidecar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        assert_eq!(base_digest_sidecar(&dir).unwrap(), base_digest(&dir).unwrap());
        assert!(sidecar_path(&dir.join("config.json")).is_file());
        assert!(sidecar_path(&dir.join("model.safetensors")).is_file());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hex_lower_matches_known_vector() {
        let mut h = Sha256::new();
        h.update(b"");
        assert_eq!(hex_lower(&h.finalize()), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}
