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
/// `total` is `None` when the caller does not know the expected size. `stop` is
/// asked before each chunk: `true` abandons the download with an
/// [`std::io::ErrorKind::Interrupted`] error, removing the `.part` like any
/// other failure.
///
/// **No byte-level resume.** Each transfer creates its own `.part` afresh, so a
/// retry after an interrupted transfer starts that file over rather than
/// continuing it.
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
    stop: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> std::io::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // A download that would fill the store's filesystem is refused before it
    // starts: a request for a repository larger than the disk must not take
    // the disk the server, its logs and every other model share.
    if let (Some(total), Some(parent)) = (total, dest.parent()) {
        if let Some(free) = free_bytes(parent) {
            if !room_for(total, free) {
                return Err(std::io::Error::new(std::io::ErrorKind::StorageFull, format!("a {total}-byte download leaves less than {} bytes free of {free}", DISK_RESERVE)));
            }
        }
    }
    // Each download writes its own `.part` (two fetches of one file, two
    // quantisations of one base, must not interleave into one), created
    // exclusively, and removed on every way out but success.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = dest.with_extension(format!(
        "{}part-{}-{}",
        dest.extension().and_then(|e| e.to_str()).map(|e| format!("{e}.")).unwrap_or_default(),
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let written = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        let mut hasher = expected_sha256.is_some().then(Sha256::new);
        let mut buf = [0u8; 64 * 1024];
        let mut got: u64 = 0;
        loop {
            if stop() {
                return Err(std::io::Error::new(std::io::ErrorKind::Interrupted, "the download was stopped"));
            }
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            got += n as u64;
            // A server that sends more than it declared is not sending the file.
            if total.is_some_and(|t| got > t) {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("the download runs past its declared {} bytes", total.unwrap_or(0))));
            }
            file.write_all(&buf[..n])?;
            if let Some(h) = &mut hasher {
                h.update(&buf[..n]);
            }
            progress(got, total);
        }
        // A stream that ends short of its declared size is a dropped
        // connection, not the file: renaming it into place would pass a
        // truncated checkpoint off as complete.
        if let Some(t) = total.filter(|&t| got < t) {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, format!("the download ended at {got} of its declared {t} bytes")));
        }
        file.sync_all()?;
        if let (Some(expected), Some(h)) = (expected_sha256, hasher) {
            let digest = hex_lower(&h.finalize());
            if digest != expected {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("sha256 mismatch: expected {expected}, got {digest}")));
            }
        }
        Ok(())
    })();
    if let Err(e) = written {
        std::fs::remove_file(&tmp).ok();
        return Err(e);
    }
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

/// What a download must leave free on the store's filesystem.
pub const DISK_RESERVE: u64 = 1 << 30;

/// Whether a download of `total` bytes leaves [`DISK_RESERVE`] of `free`.
fn room_for(total: u64, free: u64) -> bool {
    total.checked_add(DISK_RESERVE).is_some_and(|need| need <= free)
}

/// Bytes an unprivileged user may still write on the filesystem holding
/// `dir`; `None` when that cannot be asked.
fn free_bytes(dir: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut stat = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
        // SAFETY: `path` is a valid C string and `stat` a writable statvfs;
        // the struct is read only when the call succeeds.
        let ok = unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } == 0;
        ok.then(|| {
            // SAFETY: statvfs succeeded, so it wrote the whole struct.
            let stat = unsafe { stat.assume_init() };
            // `fsblkcnt_t` and `c_ulong` are u64 on Linux but narrower on other
            // Unix targets: the conversion is for them.
            #[allow(clippy::useless_conversion)]
            let free = u64::from(stat.f_bavail).saturating_mul(u64::from(stat.f_frsize));
            free
        })
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        None
    }
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
/// The sidecar (`<file>.sha256`) records the digest with the file's [`Stamp`]
/// (device, inode, size, modification and status-change times in nanoseconds)
/// and the time the hashing began. It is trusted only when the stamp still
/// matches and the hashing began more than one timestamp tick after the
/// file's last change ([`Stamp::settled_before`]): a rewrite inside the tick
/// the file was hashed in leaves an identical stamp, so a recent record is
/// hashed again rather than believed. A file whose stamp moved while it was
/// read gets no record, and a platform without the stamp fields never records.
///
/// A missing, unreadable, corrupt or mismatched sidecar costs one re-hash, and
/// the sidecar is replaced. Writes go to a per-process temporary and are
/// renamed into place, so several ranks hashing the same base at once can only
/// leave a complete sidecar behind.
pub fn file_digest_sidecar(path: &Path) -> std::io::Result<String> {
    let sidecar = sidecar_path(path);
    if let Some(recorded) = Stamp::of(&std::fs::metadata(path)?).and_then(|stamp| read_sidecar(&sidecar, &stamp)) {
        return Ok(recorded);
    }
    let (digest, record) = hash_stamped(path)?;
    if let Some((stamp, hashed_at)) = record {
        write_atomic(&sidecar, &format!("{digest}\t{}\t{hashed_at}\n", stamp.fields()));
    }
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
/// file's stamp still matches and its record has settled.
pub fn base_digest_sidecar(path: &Path) -> std::io::Result<String> {
    base_digest_with(path, &mut |p| file_digest_sidecar(p))
}

/// [`base_digest`], with each file's digest kept in the cache file `cache`
/// under the file's [`Stamp`], so a base is read and hashed once, and again
/// only when its file changes. Hashing a released checkpoint reads all of it
/// (tens of gigabytes), which a server naming its base at every start would
/// otherwise repeat. An entry answers only once it has settled
/// ([`Stamp::settled_before`]); a recent one is hashed again and refreshed, so
/// a same-size rewrite inside one timestamp tick is never served the old
/// digest. The cache is best effort: one that cannot be read or written costs
/// a re-hash, never the digest.
pub fn base_digest_cached(path: &Path, cache: &Path) -> std::io::Result<String> {
    let mut known = read_digest_cache(cache);
    let mut changed = false;
    let digest = base_digest_with(path, &mut |p| {
        let stamp = std::fs::metadata(p).ok().and_then(|m| Stamp::of(&m));
        if let Some(entry) = stamp.and_then(|s| known.get(&s).filter(|e| s.settled_before(e.hashed_at))) {
            return Ok(entry.digest.clone());
        }
        let (digest, record) = hash_stamped(p)?;
        let canonical = std::fs::canonicalize(p).ok().and_then(|c| c.to_str().map(str::to_string)).filter(|c| !c.contains('\n'));
        if let (Some((stamp, hashed_at)), Some(path)) = (record, canonical) {
            known.retain(|_, e| e.path != path);
            known.insert(stamp, CacheEntry { digest: digest.clone(), hashed_at, path });
            changed = true;
        }
        Ok(digest)
    })?;
    if changed {
        write_digest_cache(cache, &known);
    }
    Ok(digest)
}

/// What a recorded digest is valid for: the file's device and inode (which
/// file), its size, and its modification and status-change times in
/// nanoseconds (which version of it). A rewrite moves the change time even
/// when it restores the modification time, but only by the filesystem's
/// timestamp tick, which is why a record must also have settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Stamp {
    dev: u64,
    ino: u64,
    len: u64,
    mtime_ns: i128,
    ctime_ns: i128,
}

impl Stamp {
    #[cfg(unix)]
    fn of(meta: &std::fs::Metadata) -> Option<Stamp> {
        use std::os::unix::fs::MetadataExt;
        let ns = |secs: i64, nanos: i64| i128::from(secs) * 1_000_000_000 + i128::from(nanos);
        Some(Stamp { dev: meta.dev(), ino: meta.ino(), len: meta.size(), mtime_ns: ns(meta.mtime(), meta.mtime_nsec()), ctime_ns: ns(meta.ctime(), meta.ctime_nsec()) })
    }

    /// No inode or change time to prove a hit: never trust a record.
    #[cfg(not(unix))]
    fn of(_meta: &std::fs::Metadata) -> Option<Stamp> {
        None
    }

    /// The filesystem's timestamp tick, as far as the stamp shows it. Whole
    /// seconds in both times means a coarse filesystem (Lustre and NFS without
    /// sub-second times tick at 1 s, FAT at 2 s): two seconds. Otherwise one
    /// second, which also covers coarse kernel clocks and modest clock skew
    /// between the host and a network filesystem's timestamps.
    fn tick_ns(&self) -> i128 {
        const SECOND: i128 = 1_000_000_000;
        if self.mtime_ns % SECOND == 0 && self.ctime_ns % SECOND == 0 { 2 * SECOND } else { SECOND }
    }

    /// Whether a digest whose hashing began at `hashed_at` (nanoseconds since
    /// the epoch) can stand for the file: only when it began more than a tick
    /// after the file's last change. Inside the tick a later rewrite of the
    /// same size can leave this very stamp, so the record proves nothing.
    fn settled_before(&self, hashed_at: i128) -> bool {
        hashed_at > self.mtime_ns.max(self.ctime_ns) + self.tick_ns()
    }

    /// The stamp as tab-separated record fields.
    fn fields(&self) -> String {
        format!("{}\t{}\t{}\t{}\t{}", self.dev, self.ino, self.len, self.mtime_ns, self.ctime_ns)
    }

    /// Read the five fields [`Stamp::fields`] wrote.
    fn parse<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Option<Stamp> {
        Some(Stamp { dev: fields.next()?.parse().ok()?, ino: fields.next()?.parse().ok()?, len: fields.next()?.parse().ok()?, mtime_ns: fields.next()?.parse().ok()?, ctime_ns: fields.next()?.parse().ok()? })
    }
}

/// A digest cache entry: the digest, when its hashing began, and the
/// canonical path it was taken at (one entry per path is kept).
struct CacheEntry {
    digest: String,
    hashed_at: i128,
    path: String,
}

/// Nanoseconds since the epoch on the host clock.
fn now_ns() -> i128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as i128)
}

/// Hash `path`, answering its digest and, when the file kept one stamp from
/// before the read to after it, that stamp with the time the hashing began. A
/// file that changed while it was read gets no record to keep.
fn hash_stamped(path: &Path) -> std::io::Result<(String, Option<(Stamp, i128)>)> {
    let hashed_at = now_ns();
    let before = Stamp::of(&std::fs::metadata(path)?);
    let digest = file_digest(path)?;
    let after = std::fs::metadata(path).ok().and_then(|m| Stamp::of(&m));
    Ok((digest, before.filter(|b| after.as_ref() == Some(b)).map(|b| (b, hashed_at))))
}

/// Where a file's digest sidecar lives: the file's own name plus `.sha256`.
fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".sha256");
    PathBuf::from(name)
}

/// The digest a sidecar records, if it is a well-formed record for exactly
/// `stamp` that has settled. Anything else - unreadable, truncated, not
/// lowercase hex, another stamp or a record taken inside the file's last
/// timestamp tick - is `None`, which the caller answers by hashing again.
fn read_sidecar(sidecar: &Path, stamp: &Stamp) -> Option<String> {
    let text = std::fs::read_to_string(sidecar).ok()?;
    let mut fields = text.trim_end().split('\t');
    let digest = fields.next()?;
    let recorded = Stamp::parse(&mut fields)?;
    let hashed_at: i128 = fields.next()?.parse().ok()?;
    if fields.next().is_some() || recorded != *stamp || !stamp.settled_before(hashed_at) || !is_digest(digest) {
        return None;
    }
    Some(digest.to_string())
}

/// Whether `digest` is `sha256:` and 64 lowercase hex digits.
fn is_digest(digest: &str) -> bool {
    digest.strip_prefix("sha256:").is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
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

/// The cache file's entries: one
/// `digest<TAB>dev<TAB>ino<TAB>size<TAB>mtime_ns<TAB>ctime_ns<TAB>hashed_at_ns<TAB>path`
/// a line. A line in any other shape (an older layout included) is dropped,
/// which costs its file one re-hash.
fn read_digest_cache(cache: &Path) -> HashMap<Stamp, CacheEntry> {
    let Ok(text) = std::fs::read_to_string(cache) else { return HashMap::new() };
    text.lines()
        .filter_map(|line| {
            let mut fields = line.splitn(8, '\t');
            let digest = fields.next().filter(|d| is_digest(d))?.to_string();
            let stamp = Stamp::parse(&mut fields)?;
            let hashed_at = fields.next()?.parse().ok()?;
            let path = fields.next()?.to_string();
            Some((stamp, CacheEntry { digest, hashed_at, path }))
        })
        .collect()
}

/// Replace the cache file atomically (write a sibling, rename over it), so a
/// concurrent reader sees the old or the new set, never half of one.
fn write_digest_cache(cache: &Path, known: &HashMap<Stamp, CacheEntry>) {
    let mut lines: Vec<String> = known.iter().map(|(stamp, e)| format!("{}\t{}\t{}\t{}", e.digest, stamp.fields(), e.hashed_at, e.path)).collect();
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

    /// A download is refused unless it leaves the reserve free, however large
    /// the size a server declares.
    #[test]
    fn a_download_must_leave_the_reserve_free() {
        assert!(room_for(10, DISK_RESERVE + 10));
        assert!(!room_for(11, DISK_RESERVE + 10));
        assert!(!room_for(u64::MAX, u64::MAX));
        assert!(free_bytes(&std::env::temp_dir()).is_some_and(|free| free > 0));
    }

    /// A download that fails leaves no `.part`, and one that sends more than
    /// it declared is refused.
    #[test]
    fn a_failed_download_leaves_nothing_behind() {
        let dir = std::env::temp_dir().join(format!("brain-fetch-part-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("w.safetensors");
        let err = stream_to_file(&b"0123456789"[..], &dest, Some(4), None, &|| false, &mut |_, _| {}).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let err = stream_to_file(&b"abc"[..], &dest, Some(3), Some("00"), &|| false, &mut |_, _| {}).unwrap_err();
        assert!(err.to_string().contains("sha256"), "{err}");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "no .part and no destination remain");
        stream_to_file(&b"abc"[..], &dest, Some(3), None, &|| false, &mut |_, _| {}).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"abc");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Spec: a stream that ends before the size it declared is not the file:
    /// the download is refused and leaves neither a `.part` nor the
    /// destination, so a dropped connection cannot pass as a complete file.
    #[test]
    fn a_download_shorter_than_declared_is_refused() {
        let dir = std::env::temp_dir().join(format!("brain-fetch-short-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("w.safetensors");
        let err = stream_to_file(&b"abc"[..], &dest, Some(10), None, &|| false, &mut |_, _| {}).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof, "{err}");
        assert!(err.to_string().contains("3 of its declared 10"), "{err}");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "no .part and no destination remain");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streams_bytes_to_dest_atomically_via_rename() {
        let dir = std::env::temp_dir().join("modelstore-fetch-test-basic");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("weights.bin");
        let data = b"hello model weights";
        let mut seen = Vec::new();
        stream_to_file(&data[..], &dest, Some(data.len() as u64), None, &|| false, &mut |got, total| {
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
        let err = stream_to_file(&data[..], &dest, None, Some("0000000000000000000000000000000000000000000000000000000000000000"), &|| false, &mut |_, _| {})
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
        stream_to_file(&data[..], &dest, None, Some(expected), &|| false, &mut |_, _| {}).unwrap();
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
    /// under the file's stamp and served from there, once settled, until the
    /// file changes. A cache that cannot be written costs a re-hash, never the
    /// digest.
    #[test]
    fn a_base_is_hashed_once_and_again_only_when_its_file_changes() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-digest-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (file, cache) = (dir.join("weights.bin"), dir.join("cache").join("digests.txt"));
        std::fs::write(&file, b"abc").unwrap();
        let real = file_digest(&file).unwrap();
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), real);

        // A fresh entry (hashed inside the file's tick) is not trusted: a
        // forgery of it is hashed over.
        let forged = format!("sha256:{}", "0".repeat(64));
        std::fs::write(&cache, forge_record(&std::fs::read_to_string(&cache).unwrap(), &forged, false)).unwrap();
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), real, "an unsettled entry was believed");

        // A settled entry is what answers: forge it and the forgery comes back.
        std::fs::write(&cache, forge_record(&std::fs::read_to_string(&cache).unwrap(), &forged, true)).unwrap();
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), forged, "the file was not read again");

        // Another file at the same path is hashed again, and replaces its entry.
        std::fs::write(&file, b"abcd").unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options().write(true).open(&file).unwrap().set_modified(later).unwrap();
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), file_digest(&file).unwrap());
        assert_eq!(std::fs::read_to_string(&cache).unwrap().lines().count(), 1, "one entry per path");

        // A directory base caches each of its files; an unwritable cache still answers.
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        assert_eq!(base_digest_cached(&dir, &cache).unwrap(), base_digest(&dir).unwrap());
        assert_eq!(base_digest_cached(&file, &file.join("not-a-dir").join("digests.txt")).unwrap(), file_digest(&file).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Rewrite every record line (sidecar or cache) to carry `digest`, with its
    /// hashing time moved far after (`settled`) or onto the file's change.
    fn forge_record(text: &str, digest: &str, settled: bool) -> String {
        text.lines()
            .map(|line| {
                let mut f: Vec<String> = line.split('\t').map(str::to_string).collect();
                f[0] = digest.to_string();
                f[6] = if settled { (i128::from(i64::MAX)).to_string() } else { f[4].clone() };
                f.join("\t") + "\n"
            })
            .collect()
    }

    /// Spec: a same-size rewrite inside one timestamp tick is detected. On a
    /// filesystem with whole-second timestamps a rewrite within the second
    /// keeps size and modification time, so a key of those alone serves the
    /// old digest; the rewrite is simulated on any filesystem by restoring the
    /// modification time. Both the cache file and the sidecar answer the new
    /// bytes' digest.
    #[test]
    fn a_same_size_rewrite_within_one_second_is_detected() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-same-tick-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (file, cache) = (dir.join("model.safetensors"), dir.join("digests.txt"));
        std::fs::write(&file, b"abc").unwrap();
        let first = marker_of(&file);
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), file_digest(&file).unwrap());
        assert_eq!(file_digest_sidecar(&file).unwrap(), file_digest(&file).unwrap());

        std::fs::write(&file, b"abd").unwrap();
        set_marker(&file, first);
        let rewritten = file_digest(&file).unwrap();
        assert_eq!(base_digest_cached(&file, &cache).unwrap(), rewritten, "the cache served the old bytes' digest");
        assert_eq!(file_digest_sidecar(&file).unwrap(), rewritten, "the sidecar served the old bytes' digest");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Read and set the marker a sidecar test asserts against.
    fn marker_of(path: &Path) -> std::time::SystemTime {
        std::fs::metadata(path).unwrap().modified().unwrap()
    }

    fn set_marker(path: &Path, at: std::time::SystemTime) {
        std::fs::File::options().write(true).open(path).unwrap().set_modified(at).unwrap();
    }

    /// The stamp record a sidecar for `file` must start with.
    fn record_prefix(file: &Path, digest: &str) -> String {
        format!("{digest}\t{}\t", Stamp::of(&std::fs::metadata(file).unwrap()).unwrap().fields())
    }

    /// The first call hashes the file and records the digest with the stamp
    /// it was hashed at and when the hashing began; the returned digest is the
    /// content digest.
    #[test]
    fn a_sidecar_is_written_on_the_first_call_and_records_the_digest() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-sidecar-first-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("weights.bin");
        std::fs::write(&file, b"abc").unwrap();
        let real = file_digest(&file).unwrap();
        let begun = now_ns();
        assert_eq!(file_digest_sidecar(&file).unwrap(), real);

        let text = std::fs::read_to_string(sidecar_path(&file)).unwrap();
        assert!(text.starts_with(&record_prefix(&file, &real)), "{text:?}");
        let fields: Vec<&str> = text.trim_end().split('\t').collect();
        assert_eq!(fields.len(), 7, "{text:?}");
        assert!(fields[6].parse::<i128>().unwrap() >= begun, "{text:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A recorded digest answers only for the stamp it was taken at, and only
    /// once settled. The reuse is proved without timing: the record is forged
    /// to another digest, which comes back only while it is settled - so the
    /// file was not read. A moved stamp is hashed again.
    #[test]
    fn a_sidecar_is_reused_only_for_its_stamp_once_settled() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-sidecar-reuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("weights.bin");
        std::fs::write(&file, b"abc").unwrap();
        let real = file_digest(&file).unwrap();
        assert_eq!(file_digest_sidecar(&file).unwrap(), real);
        let sidecar = sidecar_path(&file);
        let forged = format!("sha256:{}", "0".repeat(64));

        std::fs::write(&sidecar, forge_record(&std::fs::read_to_string(&sidecar).unwrap(), &forged, false)).unwrap();
        assert_eq!(file_digest_sidecar(&file).unwrap(), real, "an unsettled record is hashed over");
        std::fs::write(&sidecar, forge_record(&std::fs::read_to_string(&sidecar).unwrap(), &forged, true)).unwrap();
        assert_eq!(file_digest_sidecar(&file).unwrap(), forged, "a settled record for this stamp answers");

        set_marker(&file, marker_of(&file) + std::time::Duration::from_secs(5));
        assert_eq!(file_digest_sidecar(&file).unwrap(), real, "a moved stamp is hashed again");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A sidecar that is not a well-formed record for this exact file is
    /// ignored and replaced: garbage, a truncation, an older layout and a
    /// stale stamp all cost one re-hash and never a wrong digest.
    #[test]
    fn a_corrupt_or_mismatched_sidecar_is_ignored_and_rewritten() {
        let dir = std::env::temp_dir().join(format!("modelstore-fetch-test-sidecar-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("weights.bin");
        std::fs::write(&file, b"abc").unwrap();
        let real = file_digest(&file).unwrap();
        let sidecar = sidecar_path(&file);
        let stamp = Stamp::of(&std::fs::metadata(&file).unwrap()).unwrap();
        let wrong = format!("sha256:{}", "0".repeat(64));
        let far = i64::MAX;
        for corrupt in [
            "not a digest".to_string(),
            format!("sha256:{}\t{}\t{far}", "z".repeat(64), stamp.fields()),
            format!("{wrong}\t{}\t{far}\tunexpected", stamp.fields()),
            format!("{wrong}\t{}\t{}", stamp.len, stamp.mtime_ns),
            format!("{wrong}\t{}\t{far}", Stamp { len: stamp.len + 1, ..stamp }.fields()),
            format!("{wrong}\t{}\t{far}", Stamp { ctime_ns: stamp.ctime_ns + 1, ..stamp }.fields()),
            format!("{wrong}\t{}\t{far}", Stamp { ino: stamp.ino + 1, ..stamp }.fields()),
        ] {
            std::fs::write(&sidecar, &corrupt).unwrap();
            assert_eq!(file_digest_sidecar(&file).unwrap(), real, "{corrupt:?} was ignored");
            let repaired = std::fs::read_to_string(&sidecar).unwrap();
            assert!(repaired.starts_with(&record_prefix(&file, &real)), "{corrupt:?} was replaced: {repaired:?}");
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
        let record = std::fs::read_to_string(&sidecar).unwrap();
        assert!(record.starts_with(&record_prefix(&file, &real)) && record.ends_with('\n') && record.lines().count() == 1, "{record:?}");
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
