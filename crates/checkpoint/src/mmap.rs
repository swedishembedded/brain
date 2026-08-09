// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Memory-mapped safetensors: parse the header once, then materialize tensors
//! **on demand** from the mmap instead of reading the whole file up front.
//!
//! This is the residency layer's *cold tier*: a model's weights stay mapped on disk
//! (paged in by the OS only when touched) until a tensor is actually needed, and can
//! be dropped back to disk with `advise_dontneed` when the model is demoted. Tensor
//! values are byte-identical to [`crate::safetensors::parse`] (same F32/F16/BF16/int
//! decoding); the only difference is *when* the bytes are read.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::Value;

use crate::safetensors::{bf16_to_f32, f16_to_f32, StTensor};
use crate::st::{ModelCard, CONFIG_KEY};

/// Header metadata for one tensor (byte range is relative to the tensor blob).
#[derive(Clone, Debug)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A memory-mapped safetensors file with an on-demand tensor accessor.
pub struct MmapSafetensors {
    mmap: memmap2::Mmap,
    blob_start: usize,
    index: HashMap<String, TensorMeta>,
    order: Vec<String>,
    /// The file's `__metadata__` string map (holds `brain.config`, card fields).
    metadata: BTreeMap<String, String>,
}

impl MmapSafetensors {
    /// Open + mmap `path` and parse only its header (no tensor bytes are read).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<MmapSafetensors, String> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        // SAFETY: weight files are treated as immutable for the mapping's lifetime.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;
        if mmap.len() < 8 {
            return Err("safetensors: file too short".into());
        }
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let hend = 8 + hlen;
        if mmap.len() < hend {
            return Err("safetensors: truncated header".into());
        }
        let header: Value = serde_json::from_slice(&mmap[8..hend]).map_err(|e| format!("safetensors: bad header json: {e}"))?;
        let obj = header.as_object().ok_or("safetensors: header is not an object")?;
        let mut index = HashMap::new();
        let mut order = Vec::new();
        let mut metadata = BTreeMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                if let Some(m) = meta.as_object() {
                    for (k, v) in m {
                        if let Some(s) = v.as_str() {
                            metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = meta["dtype"].as_str().ok_or("safetensors: missing dtype")?.to_string();
            let shape: Vec<usize> = meta["shape"].as_array().ok_or("safetensors: missing shape")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
            let off = meta["data_offsets"].as_array().ok_or("safetensors: missing data_offsets")?;
            let start = off[0].as_u64().unwrap() as usize;
            let end = off[1].as_u64().unwrap() as usize;
            index.insert(name.clone(), TensorMeta { dtype, shape, start, end });
            order.push(name.clone());
        }
        order.sort(); // deterministic; callers remap by name
        Ok(MmapSafetensors { mmap, blob_start: hend, index, order, metadata })
    }

    /// Tensor names, sorted (deterministic).
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// A tensor's shape, if present.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|m| m.shape.as_slice())
    }

    /// A tensor's dtype string (`F32`/`F16`/`BF16`/…), if present.
    pub fn dtype(&self, name: &str) -> Option<&str> {
        self.index.get(name).map(|m| m.dtype.as_str())
    }

    /// The raw `__metadata__` string map.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// The model config parsed from `brain.config` (or `Null` if absent).
    pub fn config(&self) -> Value {
        self.metadata
            .get(CONFIG_KEY)
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null)
    }

    /// The [`ModelCard`], if one was stored.
    pub fn card(&self) -> Option<ModelCard> {
        ModelCard::from_metadata(&self.metadata)
    }

    /// Total tensor bytes on disk (the resident footprint of the cold mapping when
    /// fully paged in) — used for cost estimation.
    pub fn blob_bytes(&self) -> u64 {
        (self.mmap.len() - self.blob_start) as u64
    }

    /// Materialize one tensor to f32, decoding on access from the mmap. Byte-identical
    /// to [`crate::safetensors::parse`] for the same tensor.
    ///
    /// Panics (does not silently return an empty/wrong vector) if the tensor's
    /// dtype has no f32 decoding — `U32` is the case that matters in practice
    /// (brain's own int8-native packed-weight tensors, `crate::weightio::
    /// Dtype::U32`): those are not meaningfully convertible to f32 at all, so
    /// a caller that reaches for `tensor_f32` on one has the wrong accessor,
    /// not a value worth defaulting to `[]` for. Use [`Self::tensor_u32`].
    pub fn tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        let m = self.index.get(name)?;
        let raw = &self.mmap[self.blob_start + m.start..self.blob_start + m.end];
        Some(decode(name, &m.dtype, raw))
    }

    /// Materialize one tensor as packed `u32` words (little-endian), decoding
    /// on access from the mmap — the read side of [`crate::weightio::
    /// StWriter::write_u32`]'s int8-native packed layout (4 int8 lanes per
    /// u32, written as-is with no repacking). Panics if the tensor's declared
    /// dtype is not `"U32"` — reading a float tensor through this accessor
    /// would silently reinterpret its bits as something else entirely, which
    /// is worse than refusing.
    pub fn tensor_u32(&self, name: &str) -> Option<Vec<u32>> {
        let m = self.index.get(name)?;
        assert_eq!(m.dtype, "U32", "tensor_u32: '{name}' has dtype {}, not U32", m.dtype);
        let raw = &self.mmap[self.blob_start + m.start..self.blob_start + m.end];
        Some(raw.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    /// One tensor as an [`StTensor`] (shape + f32 data).
    pub fn tensor(&self, name: &str) -> Option<StTensor> {
        let m = self.index.get(name)?;
        Some(StTensor { name: name.to_string(), shape: m.shape.clone(), data: self.tensor_f32(name)? })
    }

    /// Materialize **all** tensors (name-sorted) — a drop-in for
    /// [`crate::safetensors::read`] that skips the extra `fs::read` copy.
    pub fn read_all(&self) -> Vec<StTensor> {
        self.order.iter().filter_map(|n| self.tensor(n)).collect()
    }

    /// Advise the kernel it may drop the mapped pages (call when demoting a cold
    /// model so its page-cache footprint does not sit resident). Best-effort. Safe
    /// here because the mapping is read-only — dropped pages simply re-fault from the
    /// file on next access.
    pub fn advise_dontneed(&self) {
        // SAFETY: read-only file mapping ⇒ MADV_DONTNEED loses no data.
        let _ = unsafe { self.mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
    }
}

fn decode(name: &str, dtype: &str, raw: &[u8]) -> Vec<f32> {
    match dtype {
        "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        "BF16" => raw.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        "I64" => raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        "I32" => raw.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32).collect(),
        "U8" => raw.iter().map(|&b| b as f32).collect(),
        // Previously fell through to `Vec::new()` -- a packed U32 tensor (or
        // any future dtype this decoder doesn't know) would silently read as
        // an empty, valid-looking vector instead of failing. Loud is correct
        // here: `tensor_f32` promises an f32 decoding exists; if it doesn't,
        // the caller has the wrong accessor (U32 -> `tensor_u32`), not a
        // value worth defaulting to `[]` for.
        other => panic!("'{name}': no f32 decoding for dtype {other} (packed dtypes need their own accessor, e.g. tensor_u32)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal safetensors buffer with an F32 and a BF16 tensor.
    fn make_file() -> (std::path::PathBuf, Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0];
        // bf16 of [0.5, -1.0, 2.0]: top 16 bits of each f32.
        let b_vals: Vec<f32> = vec![0.5, -1.0, 2.0];
        let mut blob = Vec::new();
        for &x in &a {
            blob.extend_from_slice(&x.to_le_bytes());
        }
        let a_end = blob.len();
        for &x in &b_vals {
            blob.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
        }
        let b_end = blob.len();
        let header = serde_json::json!({
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, a_end]},
            "b": {"dtype": "BF16", "shape": [3], "data_offsets": [a_end, b_end]},
        });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut file = (hbytes.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&hbytes);
        file.extend_from_slice(&blob);
        let path = std::env::temp_dir().join(format!("mmap_st_{}.safetensors", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        (path, a, b_vals)
    }

    #[test]
    fn mmap_matches_full_parse_and_is_lazy() {
        let (path, a, b) = make_file();
        let mm = MmapSafetensors::open(&path).expect("open");
        assert_eq!(mm.names(), &["a".to_string(), "b".to_string()]);
        assert_eq!(mm.shape("a"), Some([4usize].as_slice()));
        // On-demand tensor decode matches the value we wrote (bf16 rounds exactly here).
        assert_eq!(mm.tensor_f32("a").unwrap(), a);
        assert_eq!(mm.tensor_f32("b").unwrap(), b);
        // read_all == the full-read path, byte for byte.
        let full = crate::safetensors::read(path.to_str().unwrap()).unwrap();
        let all = mm.read_all();
        assert_eq!(all.len(), full.len());
        for (x, y) in all.iter().zip(&full) {
            assert_eq!((&x.name, &x.shape, &x.data), (&y.name, &y.shape, &y.data));
        }
        std::fs::remove_file(&path).ok();
    }
}
