// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A persistent cache for a fine-tune's pre-training work: the text
//! encoder's caption contexts and the VAE's image latents.
//!
//! Swedish Embedded AB implements resumable training pipelines for its
//! clients. If your team needs expertise in making a restarted fine-tune skip
//! the hour it already paid for, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Both are pure functions of their inputs, and on a real tile set they are
//! the first hour of every run: a restart after a crash, a resumed run or a
//! second run over the same data paid it again. This keeps them on disk under
//! a directory the CALLER names (`TrainOpts::cache_dir`, the CLI's
//! `--cache-dir`); no directory means no cache.
//!
//! # A hit is verified, never trusted
//!
//! A wrong hit does not fail - it trains the adapter against another prompt
//! or another image. So the digest of the key is only a FILE NAME: the full
//! key text is stored in the file and compared byte for byte on load, along
//! with the payload length. A digest collision, a truncated file or a stale
//! entry is a MISS, never a wrong hit.
//!
//! The key is everything the value depends on: for a caption, the prompt, the
//! text encoder's weights identity (every file of it: path, length and
//! modification time), the tokenizer's identity, the encoder's tier, the
//! context layout; for a latent, a digest of the exact pixels encoded, the
//! VAE weights identity, the size and the token layout - plus
//! [`ENCODE_REVISION`], which names the encode procedure itself. Changing a
//! weight file (in place or by path) or a prompt is a different key.
//!
//! Entries are written whole to a temporary file and renamed into place, so
//! a crash mid-write leaves no entry rather than a half one. Nothing is ever
//! evicted: the directory belongs to the caller.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Names the encode procedure (prompt templating, padding, tap layers,
/// latent packing). Bump it when any of those change, so entries written by
/// the old procedure stop matching.
pub const ENCODE_REVISION: u32 = 1;

/// The first line of every entry file.
const MAGIC: &str = "brain-flux2-encoding-cache v1";

/// The identity of a weight file or directory without reading its contents:
/// every regular file's canonical path, byte length and modification time,
/// in sorted order. A rewrite in place or a different file is a different
/// identity.
pub fn weights_identity(path: &Path) -> Result<String, String> {
    let root = path.canonicalize().map_err(|e| format!("encoding cache: {}: {e}", path.display()))?;
    let mut files = Vec::new();
    collect(&root, &mut files)?;
    files.sort();
    let mut id = String::new();
    for f in files {
        let m = std::fs::metadata(&f).map_err(|e| format!("encoding cache: {}: {e}", f.display()))?;
        let mtime = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        id.push_str(&format!("{}:{}:{};", f.display(), m.len(), mtime));
    }
    Ok(id)
}

fn collect(p: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if p.is_dir() {
        for e in std::fs::read_dir(p).map_err(|e| format!("encoding cache: {}: {e}", p.display()))? {
            collect(&e.map_err(|e| e.to_string())?.path(), out)?;
        }
    } else {
        out.push(p.to_path_buf());
    }
    Ok(())
}

/// A digest of `values`' exact bits - the content key of an image.
pub fn content_digest(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_bits().to_le_bytes());
    }
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The cache under one caller-chosen directory.
pub struct EncodeCache {
    dir: PathBuf,
}

impl EncodeCache {
    /// Use (creating if needed) `dir`.
    pub fn open(dir: &Path) -> Result<EncodeCache, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("encoding cache: create {}: {e}", dir.display()))?;
        Ok(EncodeCache { dir: dir.to_path_buf() })
    }

    fn file(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{}.f32", hex(&Sha256::digest(key.as_bytes()))))
    }

    /// The stored value for `key`, or `None` on any miss - absent, a different
    /// key under the same name, a wrong length or an unreadable file.
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        let mut raw = Vec::new();
        std::fs::File::open(self.file(key)).ok()?.read_to_end(&mut raw).ok()?;
        let header = format!("{MAGIC}\n{}\n{key}\n", ENCODE_REVISION);
        let rest = raw.strip_prefix(header.as_bytes())?;
        let (len, payload) = rest.split_at_checked(8)?;
        let n = u64::from_le_bytes(len.try_into().ok()?) as usize;
        if payload.len() != n * 4 {
            return None;
        }
        Some(payload.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    /// Store `value` under `key`, atomically.
    pub fn put(&self, key: &str, value: &[f32]) -> Result<(), String> {
        let path = self.file(key);
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        let write = || -> std::io::Result<()> {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(format!("{MAGIC}\n{}\n{key}\n", ENCODE_REVISION).as_bytes())?;
            f.write_all(&(value.len() as u64).to_le_bytes())?;
            for v in value {
                f.write_all(&v.to_le_bytes())?;
            }
            f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
            std::fs::rename(&tmp, &path)
        };
        write().map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("encoding cache: write {}: {e}", path.display())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("brain-flux2-enccache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_stored_value_comes_back_bit_for_bit_and_only_under_its_own_key() {
        let d = dir("roundtrip");
        let c = EncodeCache::open(&d).unwrap();
        let v = vec![1.5f32, -0.0, f32::MIN_POSITIVE, 3.25e7, f32::NAN];
        assert!(c.get("caption|a cat").is_none(), "an empty cache misses");
        c.put("caption|a cat", &v).unwrap();
        let got = c.get("caption|a cat").expect("hit");
        assert_eq!(got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert!(c.get("caption|a dog").is_none(), "another prompt misses");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The digest names the file; the key inside it decides. Planting one
    /// key's entry under another key's file name - what a digest collision
    /// would look like - must read as a miss.
    #[test]
    fn a_colliding_or_truncated_file_is_a_miss() {
        let d = dir("collide");
        let c = EncodeCache::open(&d).unwrap();
        c.put("k1", &[1.0, 2.0]).unwrap();
        std::fs::copy(c.file("k1"), c.file("k2")).unwrap();
        assert!(c.get("k2").is_none(), "a file holding another key is a miss");
        let raw = std::fs::read(c.file("k1")).unwrap();
        std::fs::write(c.file("k1"), &raw[..raw.len() - 2]).unwrap();
        assert!(c.get("k1").is_none(), "a truncated payload is a miss");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_rewritten_weight_file_has_a_different_identity() {
        let d = dir("ident");
        std::fs::create_dir_all(&d).unwrap();
        let w = d.join("w.safetensors");
        std::fs::write(&w, b"one").unwrap();
        let a = weights_identity(&w).unwrap();
        assert_eq!(a, weights_identity(&w).unwrap(), "stable while untouched");
        std::fs::write(&w, b"three").unwrap();
        assert_ne!(a, weights_identity(&w).unwrap(), "a rewritten file is a different identity");
        // A directory's identity covers every file in it.
        let a = weights_identity(&d).unwrap();
        std::fs::write(d.join("shard2.safetensors"), b"x").unwrap();
        assert_ne!(a, weights_identity(&d).unwrap(), "a new shard is a different identity");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_content_digest_sees_every_bit() {
        assert_ne!(content_digest(&[0.0]), content_digest(&[-0.0]));
        assert_eq!(content_digest(&[1.0, 2.0]), content_digest(&[1.0, 2.0]));
    }
}
