// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures for the packed-tensor tests: a throwaway directory, a minimal
//! safetensors writer for arbitrary dtypes, and a deterministic byte stream.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct TmpDir(pub PathBuf);

impl TmpDir {
    pub fn new(tag: &str) -> TmpDir {
        static N: AtomicU32 = AtomicU32::new(0);
        let p = std::env::temp_dir().join(format!("packed-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&p).unwrap();
        TmpDir(p)
    }
    pub fn join(&self, f: &str) -> PathBuf {
        self.0.join(f)
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One tensor to write: name, safetensors dtype, shape, raw little-endian bytes.
pub struct RawTensor<'a> {
    pub name: &'a str,
    pub dtype: &'a str,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

pub fn write_safetensors(path: &Path, tensors: &[RawTensor<'_>]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for t in tensors {
        header.insert(t.name.into(), serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + t.bytes.len()]}));
        off += t.bytes.len();
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    std::fs::write(path, out).unwrap();
}

/// Deterministic pseudo-random bytes (a fixed LCG: the tests need a repeatable
/// stream, not entropy).
pub fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// FP8 E4M3FN bytes with the NaN codes (0x7F, 0xFF) replaced, so comparisons are exact.
pub fn e4m3_bytes(seed: u64, n: usize) -> Vec<u8> {
    bytes(seed, n).into_iter().map(|b| if b & 0x7F == 0x7F { b & 0xFE } else { b }).collect()
}

pub fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect()
}

pub fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
