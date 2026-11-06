// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that move datasets and
// trained adapters between nodes by content, never by path, for its clients.
// If your team needs expertise in content-addressed artifact exchange for
// model training clusters, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The content-addressed artifact store a serving process is configured with
//! (`brain serve --artifact-store DIR`): bulk data a training job reads and
//! writes, named by the blake3 digest of its bytes.
//!
//! A file of the store is named by its digest alone (`<64 lowercase hex>`).
//! A caller never names a path: it names a [`Digest`], which parses from
//! exactly 64 lowercase hex digits and so cannot spell a separator, a dot or
//! anything else that could leave the store directory, and the server
//! resolves it against the directory the operator chose. What it opens is
//! verified against that digest as it is copied into the job's private
//! directory, so a file that was swapped, truncated or corrupted after it was
//! published is refused, and what the job then reads is the bytes that were
//! hashed (there is no window between checking a file and using it).
//!
//! A store may live on a filesystem several nodes share, which is what moves
//! a dataset to a node that has none of the caller's files; writers publish
//! atomically (a temporary file in the store, renamed to its digest), so a
//! reader never sees a half-written artifact.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The blake3 digest of a stored artifact.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest([u8; 32]);

/// Why a string is not a digest.
#[derive(Debug, PartialEq, Eq)]
pub struct NotADigest;

impl fmt::Display for NotADigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a digest is 64 lowercase hexadecimal digits (the blake3 of the artifact)")
    }
}

impl Digest {
    /// The digest `text` spells: exactly 64 lowercase hex digits and nothing else, so
    /// no spelling of it names anything but a file of the store.
    pub fn parse(text: &str) -> Result<Digest, NotADigest> {
        let bytes = text.as_bytes();
        if bytes.len() != 64 {
            return Err(NotADigest);
        }
        let nibble = |c: u8| match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            _ => Err(NotADigest),
        };
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = nibble(bytes[2 * i])? << 4 | nibble(bytes[2 * i + 1])?;
        }
        Ok(Digest(out))
    }

    /// The lowercase hex spelling, which is also the artifact's file name.
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn of(hash: &blake3::Hash) -> Digest {
        Digest(*hash.as_bytes())
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest({})", self.hex())
    }
}

/// Why a store operation failed. The client-facing text ([`StoreError::client_message`])
/// never carries a path of this machine.
#[derive(Debug)]
pub enum StoreError {
    /// No artifact has this digest (or it is not a regular file of the store).
    Missing(Digest),
    /// The artifact exists but its bytes do not hash to its name.
    Corrupt(Digest),
    /// The artifact is larger than the caller may take.
    TooLarge { digest: Digest, limit: u64 },
    /// The store directory or a file in it could not be read or written; the cause is for the operator's log.
    Io(String),
}

impl StoreError {
    /// What a remote caller may be told.
    pub fn client_message(&self) -> String {
        match self {
            StoreError::Missing(d) => format!("artifact {d} is not in the artifact store"),
            StoreError::Corrupt(d) => format!("artifact {d} does not hash to its digest"),
            StoreError::TooLarge { digest, limit } => format!("artifact {digest} is larger than the {limit} bytes a job may read"),
            StoreError::Io(_) => "the artifact store could not be read or written".to_string(),
        }
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Io(cause) => write!(f, "artifact store: {cause}"),
            other => f.write_str(&other.client_message()),
        }
    }
}

impl std::error::Error for StoreError {}

fn io_error(what: &str, path: &Path, e: &std::io::Error) -> StoreError {
    StoreError::Io(format!("{what} {}: {e}", path.display()))
}

/// A directory of artifacts named by their blake3 digest.
#[derive(Clone, Debug)]
pub struct ArtifactStore {
    root: PathBuf,
}

/// Bytes copied per read while hashing.
const CHUNK: usize = 1 << 20;

/// Makes the names of temporary files unique within this process.
static PUBLISHED: AtomicU64 = AtomicU64::new(0);

impl ArtifactStore {
    /// The store in `dir`, which must already exist: the operator creates it, so a mistyped flag is
    /// an error at startup and not a store created where nobody looked.
    pub fn open(dir: &Path) -> Result<ArtifactStore, StoreError> {
        let root = std::fs::canonicalize(dir).map_err(|e| io_error("artifact store directory", dir, &e))?;
        if !root.is_dir() {
            return Err(StoreError::Io(format!("artifact store directory {}: not a directory", root.display())));
        }
        Ok(ArtifactStore { root })
    }

    /// The directory the store lives in.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether a regular file named `digest` is in the store now: a cheap answer for refusing a request at the
    /// door. It does not read the file, so it is not a promise: [`Self::fetch`] verifies what it copies.
    pub fn contains(&self, digest: &Digest) -> bool {
        std::fs::symlink_metadata(self.root.join(digest.hex())).is_ok_and(|m| m.is_file())
    }

    /// Copies the artifact `digest` to the new file `dest` (mode 0600), at most `limit` bytes, verifying
    /// that what was copied hashes to `digest`; `dest` is removed again when it does not. Returns the size.
    pub fn fetch(&self, digest: &Digest, dest: &Path, limit: u64) -> Result<u64, StoreError> {
        let path = self.root.join(digest.hex());
        let mut file = match open_stored(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(StoreError::Missing(*digest)),
            // A symbolic link (ELOOP) is not an artifact of the store.
            #[cfg(unix)]
            Err(e) if e.raw_os_error() == Some(libc::ELOOP) => return Err(StoreError::Missing(*digest)),
            Err(e) => return Err(io_error("opening", &path, &e)),
        };
        let meta = file.metadata().map_err(|e| io_error("reading", &path, &e))?;
        if !meta.is_file() {
            return Err(StoreError::Missing(*digest));
        }
        if meta.len() > limit {
            return Err(StoreError::TooLarge { digest: *digest, limit });
        }
        let mut out = create_new(dest, 0o600).map_err(|e| io_error("creating", dest, &e))?;
        let copied = copy_hashing(&mut file, &mut out, limit);
        let verdict = match copied {
            Ok((hash, size)) if &Digest::of(&hash) == digest => Ok(size),
            Ok((_, _)) => Err(StoreError::Corrupt(*digest)),
            Err(CopyError::TooLong) => Err(StoreError::TooLarge { digest: *digest, limit }),
            Err(CopyError::Io(e)) => Err(io_error("copying", &path, &e)),
        };
        drop(out);
        if verdict.is_err() {
            std::fs::remove_file(dest).ok();
        }
        verdict
    }

    /// Publishes the file `src` to the store and returns its digest and size. Atomic: the artifact appears
    /// under its digest complete, or not at all. Publishing what the store already holds changes nothing.
    pub fn publish(&self, src: &Path) -> Result<(Digest, u64), StoreError> {
        let mut input = File::open(src).map_err(|e| io_error("opening", src, &e))?;
        self.publish_from(&mut input)
    }

    /// [`Self::publish`] of bytes held in memory.
    pub fn publish_bytes(&self, bytes: &[u8]) -> Result<Digest, StoreError> {
        self.publish_from(&mut &bytes[..]).map(|(digest, _)| digest)
    }

    fn publish_from(&self, input: &mut dyn Read) -> Result<(Digest, u64), StoreError> {
        let temp = self.root.join(format!(".publishing-{}-{}", std::process::id(), PUBLISHED.fetch_add(1, Ordering::Relaxed)));
        let mut out = create_new(&temp, 0o640).map_err(|e| io_error("creating", &temp, &e))?;
        let finish = (|| {
            let (hash, size) = copy_hashing(input, &mut out, u64::MAX).map_err(|e| match e {
                CopyError::Io(e) => io_error("writing", &temp, &e),
                CopyError::TooLong => StoreError::Io("artifact too long".to_string()),
            })?;
            out.sync_all().map_err(|e| io_error("syncing", &temp, &e))?;
            let digest = Digest::of(&hash);
            std::fs::rename(&temp, self.root.join(digest.hex())).map_err(|e| io_error("publishing", &temp, &e))?;
            Ok((digest, size))
        })();
        if finish.is_err() {
            std::fs::remove_file(&temp).ok();
        }
        finish
    }
}

enum CopyError {
    Io(std::io::Error),
    TooLong,
}

/// Opens the stored artifact `path` for reading, refusing to be led out of the store.
fn open_stored(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    // `O_NOFOLLOW`: a link planted in the store is not followed out of it. `O_NONBLOCK`: opening a FIFO there
    // must not hang the job. The type is checked on the open handle by the caller.
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    // `FILE_FLAG_OPEN_REPARSE_POINT`: a link is opened as itself, which the caller's type check refuses.
    #[cfg(windows)]
    std::os::windows::fs::OpenOptionsExt::custom_flags(&mut options, 0x0020_0000);
    options.open(path)
}

/// Creates the new file `path`, with the permission bits `mode` on Unix. Elsewhere it takes its directory's
/// access control, which the store's owner sets.
fn create_new(path: &Path, mode: u32) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, mode);
    #[cfg(not(unix))]
    let _ = mode;
    options.open(path)
}

/// Copies `from` to `to`, hashing what passes, refusing more than `limit` bytes.
fn copy_hashing(from: &mut dyn Read, to: &mut dyn Write, limit: u64) -> Result<(blake3::Hash, u64), CopyError> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; CHUNK];
    let mut total: u64 = 0;
    loop {
        let n = match from.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(CopyError::Io(e)),
        };
        total += n as u64;
        if total > limit {
            return Err(CopyError::TooLong);
        }
        hasher.update(&buffer[..n]);
        to.write_all(&buffer[..n]).map_err(CopyError::Io)?;
    }
    Ok((hasher.finalize(), total))
}

/// The digest `bytes` would be published under.
pub fn digest_of(bytes: &[u8]) -> Digest {
    Digest::of(&blake3::hash(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mode bits of a published artifact: what other users may read.
    #[cfg(unix)]
    fn published_mode(store: &ArtifactStore, digest: &Digest) -> Option<u32> {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(store.root.join(digest.hex())).ok().map(|m| m.permissions().mode() & 0o777)
    }

    fn store() -> (tempfile::TempDir, ArtifactStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::open(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn a_digest_is_exactly_sixty_four_lowercase_hex_digits() {
        let ok = "0123456789abcdef".repeat(4);
        assert_eq!(Digest::parse(&ok).unwrap().hex(), ok);
        // Nothing that is not a digest may parse - least of all anything that could name a path.
        for bad in [
            "", "abc", &ok[..63], &format!("{ok}0"), &ok.to_uppercase(), &format!("blake3:{ok}"), &format!("../{}", &ok[3..]), &format!("{}/", &ok[1..]),
            &format!("{}/x", &ok[..60]), &ok.replace('0', "g"), &format!("{}\0", &ok[1..]), &format!(" {}", &ok[1..]), &format!("{}\n", &ok[1..]), "..", "/etc/passwd",
        ] {
            assert_eq!(Digest::parse(bad), Err(NotADigest), "{bad:?}");
        }
    }

    #[test]
    fn what_is_published_comes_back_by_its_digest_and_only_by_it() {
        let (_dir, store) = store();
        let digest = store.publish_bytes(b"{\"messages\":[]}\n").unwrap();
        assert_eq!(digest, digest_of(b"{\"messages\":[]}\n"));
        #[cfg(unix)]
        assert_eq!(published_mode(&store, &digest), Some(0o640));
        let work = tempfile::tempdir().unwrap();
        let copy = work.path().join("copy");
        assert_eq!(store.fetch(&digest, &copy, 1 << 20).unwrap(), 16);
        assert_eq!(std::fs::read(&copy).unwrap(), b"{\"messages\":[]}\n");
        // Publishing the same bytes again is the same artifact, and leaves no temporary file behind.
        assert_eq!(store.publish_bytes(b"{\"messages\":[]}\n").unwrap(), digest);
        let names: Vec<String> = std::fs::read_dir(store.root()).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        assert_eq!(names, vec![digest.hex()]);
        let absent = digest_of(b"not published");
        assert!(matches!(store.fetch(&absent, &work.path().join("x"), 1 << 20), Err(StoreError::Missing(d)) if d == absent));
        assert!(!work.path().join("x").exists());
    }

    #[test]
    fn an_artifact_that_does_not_hash_to_its_name_is_refused_and_not_left_behind() {
        let (_dir, store) = store();
        let digest = store.publish_bytes(b"the real dataset").unwrap();
        std::fs::write(store.root().join(digest.hex()), b"a swapped dataset").unwrap();
        let work = tempfile::tempdir().unwrap();
        let copy = work.path().join("copy");
        assert!(matches!(store.fetch(&digest, &copy, 1 << 20), Err(StoreError::Corrupt(d)) if d == digest));
        assert!(!copy.exists(), "unverified bytes must not remain where a job could read them");
    }

    #[cfg(unix)]
    #[test]
    fn a_link_a_directory_or_a_pipe_in_the_store_is_not_an_artifact() {
        let (dir, store) = store();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret");
        std::fs::write(&secret, b"outside the store").unwrap();
        let work = tempfile::tempdir().unwrap();

        let linked = digest_of(b"outside the store");
        std::os::unix::fs::symlink(&secret, dir.path().join(linked.hex())).unwrap();
        assert!(matches!(store.fetch(&linked, &work.path().join("a"), 1 << 20), Err(StoreError::Missing(_))), "a link out of the store is not followed even when its target hashes right");

        let directory = digest_of(b"a directory");
        std::fs::create_dir(dir.path().join(directory.hex())).unwrap();
        assert!(matches!(store.fetch(&directory, &work.path().join("b"), 1 << 20), Err(StoreError::Missing(_))));

        let pipe = digest_of(b"a pipe");
        let c_path = std::ffi::CString::new(dir.path().join(pipe.hex()).to_str().unwrap()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        assert!(matches!(store.fetch(&pipe, &work.path().join("c"), 1 << 20), Err(StoreError::Missing(_))), "opening a pipe must neither hang nor read");
    }

    #[test]
    fn an_artifact_past_the_limit_is_refused_before_a_byte_is_kept() {
        let (_dir, store) = store();
        let digest = store.publish_bytes(&[7u8; 4096]).unwrap();
        let work = tempfile::tempdir().unwrap();
        let copy = work.path().join("copy");
        assert!(matches!(store.fetch(&digest, &copy, 4095), Err(StoreError::TooLarge { limit: 4095, .. })));
        assert!(!copy.exists());
        assert_eq!(store.fetch(&digest, &copy, 4096).unwrap(), 4096);
    }

    #[test]
    fn what_a_remote_caller_is_told_names_no_path_of_this_machine() {
        let (dir, store) = store();
        let missing = digest_of(b"x");
        let error = store.fetch(&missing, &dir.path().join("no-such-dir").join("y"), 10).unwrap_err();
        let shown = error.client_message();
        assert!(!shown.contains(dir.path().to_str().unwrap()), "{shown}");
        let unwritable = store.publish(&dir.path().join("absent")).unwrap_err();
        assert!(!unwritable.client_message().contains(dir.path().to_str().unwrap()));
        assert!(unwritable.to_string().contains("absent"), "the operator's log keeps the cause: {unwritable}");
    }

    #[test]
    fn the_store_directory_must_already_exist() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ArtifactStore::open(&dir.path().join("typo")).is_err());
        let file = dir.path().join("file");
        std::fs::write(&file, b"").unwrap();
        assert!(ArtifactStore::open(&file).is_err());
    }
}
