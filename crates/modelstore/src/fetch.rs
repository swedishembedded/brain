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

    #[test]
    fn hex_lower_matches_known_vector() {
        let mut h = Sha256::new();
        h.update(b"");
        assert_eq!(hex_lower(&h.finalize()), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}
