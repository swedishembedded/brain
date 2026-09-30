// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming, atomic file download: the mechanics shared by every [`crate::Hub`]
//! implementation that actually touches a network. Isolated from `hub.rs` so it
//! can be exercised with an in-memory [`std::io::Read`] in tests -- no live
//! server required to prove the atomic-write and sha256 behavior.

use std::io::{Read, Write};
use std::path::Path;

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

    #[test]
    fn hex_lower_matches_known_vector() {
        let mut h = Sha256::new();
        h.update(b"");
        assert_eq!(hex_lower(&h.finalize()), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}
