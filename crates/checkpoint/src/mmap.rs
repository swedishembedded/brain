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

use crate::safetensors::{bf16_to_f32, decode_e4m3_bytes, f16_to_f32, StTensor};
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
            let dtype = meta["dtype"].as_str().ok_or_else(|| format!("safetensors: tensor '{name}': missing dtype"))?.to_string();
            // Every header field is user-supplied (a downloaded, possibly
            // truncated or corrupt file) — validate NOW, once, so every later
            // accessor can slice the mmap without panicking mid-import or
            // mid-activation. One loud Err naming the tensor.
            let width = try_dtype_width(&dtype).ok_or_else(|| format!("safetensors: tensor '{name}': unknown dtype '{dtype}'"))?;
            let mut shape = Vec::new();
            for v in meta["shape"].as_array().ok_or_else(|| format!("safetensors: tensor '{name}': missing shape"))? {
                shape.push(v.as_u64().ok_or_else(|| format!("safetensors: tensor '{name}': non-integer shape entry {v}"))? as usize);
            }
            let off = meta["data_offsets"].as_array().ok_or_else(|| format!("safetensors: tensor '{name}': missing data_offsets"))?;
            let start = off.first().and_then(Value::as_u64).ok_or_else(|| format!("safetensors: tensor '{name}': bad data_offsets"))? as usize;
            let end = off.get(1).and_then(Value::as_u64).ok_or_else(|| format!("safetensors: tensor '{name}': bad data_offsets"))? as usize;
            if start > end {
                return Err(format!("safetensors: tensor '{name}': data_offsets start {start} > end {end}"));
            }
            if !(end - start).is_multiple_of(width) {
                return Err(format!("safetensors: tensor '{name}': byte range {} not a multiple of {dtype}'s element width {width}", end - start));
            }
            if end > mmap.len() - hend {
                return Err(format!(
                    "safetensors: tensor '{name}': data_offsets end {end} exceeds the tensor blob ({} bytes) — truncated or corrupt file",
                    mmap.len() - hend
                ));
            }
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
        let mut out = Vec::new();
        decode_into(name, &m.dtype, raw, &mut out);
        // A whole-file streaming scan (32 blocks of a multi-GB checkpoint)
        // must not accumulate every tensor's page-cache pages in this
        // process's RSS -- drop this tensor's pages now that it is decoded;
        // a later re-read simply re-faults from disk. See advise_dontneed_tensor.
        self.advise_dontneed_tensor(name);
        Some(out)
    }

    /// Element count of `name`, without decoding — the declared dtype's byte
    /// width divides the byte range exactly (safetensors invariant).
    pub fn numel(&self, name: &str) -> Option<usize> {
        let m = self.index.get(name)?;
        Some((m.end - m.start) / dtype_width(&m.dtype))
    }

    /// Decode elements `[start_elem, start_elem+len_elem)` of `name`'s FLAT,
    /// row-major layout - for a `[rows, cols]` tensor, one row is `cols`
    /// elements starting at `row*cols`. Unlike [`Self::tensor_f32`] (whole
    /// tensor) or [`Self::with_tensor_chunks`] (every chunk from offset 0),
    /// this never touches bytes outside the requested range - the accessor a
    /// caller wants for a handful of embedding-table rows out of a
    /// multi-hundred-thousand-row vocabulary (`[vocab, d_model]`), where
    /// decoding the whole table, or even scanning from its start, would cost
    /// far more than the few rows actually needed.
    ///
    /// `None` when `name` is absent, or when `[start_elem, start_elem+len_elem)`
    /// falls even partially outside the tensor's element count - never a
    /// panic or a silently truncated read. Shares [`decode_into`] with
    /// [`Self::tensor_f32`]/[`Self::with_tensor_chunks`], so a range read is
    /// byte-identical to the corresponding slice of a whole-tensor decode for
    /// every dtype those already support.
    pub fn tensor_f32_range(&self, name: &str, start_elem: usize, len_elem: usize) -> Option<Vec<f32>> {
        let m = self.index.get(name)?;
        let width = dtype_width(&m.dtype);
        let numel = (m.end - m.start) / width;
        let end_elem = start_elem.checked_add(len_elem)?;
        if end_elem > numel {
            return None;
        }
        let start = self.blob_start + m.start + start_elem * width;
        let end = start + len_elem * width;
        let mut out = Vec::new();
        decode_into(name, &m.dtype, &self.mmap[start..end], &mut out);
        Some(out)
    }

    /// Zero-copy: `name`'s on-disk bytes reinterpreted as `u32` words,
    /// borrowed straight from the mapping — no allocation. `None` unless the
    /// dtype is already `F32` or `U32` (what the device binds) AND the byte
    /// range is 4-byte aligned relative to the mapping's base (safetensors
    /// does not guarantee this — an odd header length, or a dtype narrower
    /// than 4 bytes earlier in the file, can misalign a later tensor).
    /// `bytemuck::try_cast_slice` is the alignment check: never an `unsafe`
    /// transmute, and a real misalignment or the wrong dtype both cleanly
    /// fall through to `None` rather than panicking or misreading bytes.
    pub fn raw_words(&self, name: &str) -> Option<&[u32]> {
        let m = self.index.get(name)?;
        if !matches!(m.dtype.as_str(), "F32" | "U32") {
            return None;
        }
        let raw = &self.mmap[self.blob_start + m.start..self.blob_start + m.end];
        bytemuck::try_cast_slice::<u8, u32>(raw).ok()
    }

    /// Ordered chunks of at most `max_elems` f32, each decoded into ONE
    /// reused scratch buffer this call owns — so peak *extra* host
    /// allocation is `O(max_elems)`, never `O(tensor)` (relevant tensors run
    /// up to ~1.5 GB as f32). `max_elems == 0` decodes the whole tensor as
    /// one chunk (degrades to [`Self::tensor_f32`]'s behaviour). Returns
    /// whether the tensor was found.
    pub fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        let Some(m) = self.index.get(name) else { return false };
        let width = dtype_width(&m.dtype);
        let numel = (m.end - m.start) / width;
        let chunk = if max_elems == 0 { numel.max(1) } else { max_elems };
        let mut scratch: Vec<f32> = Vec::with_capacity(chunk.min(numel.max(1)));
        let mut off = 0usize;
        while off < numel {
            let n = chunk.min(numel - off);
            let start = self.blob_start + m.start + off * width;
            let end = start + n * width;
            decode_into(name, &m.dtype, &self.mmap[start..end], &mut scratch);
            f(off as u64, &scratch);
            off += n;
        }
        // Same rationale as tensor_f32: this tensor is now fully consumed by
        // the caller, so its pages can be dropped instead of sitting resident
        // for the rest of a whole-checkpoint streaming scan.
        self.advise_dontneed_tensor(name);
        true
    }

    /// [`Self::with_tensor_chunks`] for a packed-`U32` tensor: ordered chunks
    /// of at most `max_elems` `u32` words, each decoded into ONE reused
    /// scratch this call owns, so peak *extra* host allocation is
    /// `O(max_elems)` rather than `O(tensor)`. The bounded counterpart to
    /// [`Self::tensor_u32`], which materializes the whole thing - an int8
    /// checkpoint is mostly packed `U32`, so without this there was no
    /// bounded way to move one to a device at all when
    /// [`Self::raw_words`] declines (a misaligned byte range).
    /// `max_elems == 0` decodes the whole tensor as one chunk. Panics on a
    /// non-`U32` dtype, exactly like [`Self::tensor_u32`] - reaching for this
    /// accessor on a float tensor is a caller bug, not a value to reinterpret.
    pub fn with_tensor_u32_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[u32])) -> bool {
        let Some(m) = self.index.get(name) else { return false };
        assert_eq!(m.dtype, "U32", "with_tensor_u32_chunks: '{name}' has dtype {}, not U32", m.dtype);
        let numel = (m.end - m.start) / 4;
        let chunk = if max_elems == 0 { numel.max(1) } else { max_elems };
        let mut scratch: Vec<u32> = Vec::with_capacity(chunk.min(numel.max(1)));
        let mut off = 0usize;
        while off < numel {
            let n = chunk.min(numel - off);
            let start = self.blob_start + m.start + off * 4;
            scratch.clear();
            scratch.extend(self.mmap[start..start + n * 4].chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
            f(off as u64, &scratch);
            off += n;
        }
        // Same rationale as `with_tensor_chunks`: fully consumed, so the pages
        // need not stay resident for the rest of a whole-checkpoint scan.
        self.advise_dontneed_tensor(name);
        true
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

    /// [`Self::advise_dontneed`] for one tensor's byte range only — per-group
    /// demotion (dropping the pages a specific slice no longer needs) without
    /// discarding the whole mapping's page-cache footprint.
    pub fn advise_dontneed_tensor(&self, name: &str) {
        let Some(m) = self.index.get(name) else { return };
        let start = self.blob_start + m.start;
        let len = m.end - m.start;
        // SAFETY: read-only file mapping ⇒ MADV_DONTNEED loses no data.
        let _ = unsafe { self.mmap.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, start, len) };
    }
}

/// On-disk byte width of one element for a safetensors dtype string, or `None`
/// for a dtype this reader does not know. Only the dtypes [`decode_into`] knows
/// how to decode as f32 (`tensor_u32`'s `U32` is handled separately — packed,
/// not a per-element f32 decode). `open()` rejects unknown dtypes up front, so
/// [`dtype_width`] below is infallible for any tensor that made it into the index.
fn try_dtype_width(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" => Some(2),
        "I64" => Some(8),
        // F8_E5M2 is included here (not just F8_E4M3) so `open()`'s header
        // validation and byte-range/width checks succeed for a file that
        // merely CONTAINS an E5M2 tensor - the loud, name-it rejection lives
        // in `decode_into`, at actual decode time, matching `crate::
        // safetensors::parse`'s own split between "the file is well-formed"
        // and "this tensor's dtype is one we can decode".
        "U8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        _ => None,
    }
}

/// [`try_dtype_width`] for a dtype `open()` already validated (every indexed
/// tensor's dtype is known by construction).
fn dtype_width(dtype: &str) -> usize {
    try_dtype_width(dtype).unwrap_or_else(|| panic!("'{dtype}': unknown element width (open() validates dtypes, so this is unreachable for indexed tensors)"))
}

/// Decode `raw` (a whole tensor's bytes, or one chunk of them) as `dtype`,
/// clearing and extending `out` in place — the one implementation both the
/// whole-tensor [`MmapSafetensors::tensor_f32`] and the chunked
/// [`MmapSafetensors::with_tensor_chunks`] share, so they cannot drift.
/// Thin alias for the shared decoder, so each arm of [`decode_into`] reads as
/// "width + per-element conversion" and nothing else.
fn decode(raw: &[u8], width: usize, f: fn(&[u8]) -> f32, out: &mut Vec<f32>) {
    crate::safetensors::decode_elems_into(raw, width, f, out)
}

fn decode_into(name: &str, dtype: &str, raw: &[u8], out: &mut Vec<f32>) {
    out.clear();
    // Every fixed-width arm goes through the SHARED, host-parallel
    // `decode_elems_into` rather than a serial `extend`. Same per-element
    // function, so the bytes are unchanged; what changes is that a
    // billion-element bf16 tensor no longer converts on one core. The eager
    // whole-file parse was fanned out long ago and this, the path a streaming
    // multi-GB import actually takes, was left behind - which made streaming
    // look like a memory-for-time trade when most of the "time" was an idle
    // thread pool.
    match dtype {
        "F32" => decode(raw, 4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]), out),
        "F16" => decode(raw, 2, |b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])), out),
        "BF16" => decode(raw, 2, |b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])), out),
        "I64" => decode(raw, 8, |b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32, out),
        "I32" => decode(raw, 4, |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32, out),
        "U8" => decode(raw, 1, |b| b[0] as f32, out),
        "F8_E4M3" => out.extend(decode_e4m3_bytes(raw)),
        // Named explicitly rather than falling into the `other` panic below:
        // E5M2 is a real, if rarer, FP8 checkpoint format (more exponent
        // range, less mantissa) - silently decoding its bytes as if they were
        // E4M3 would produce plausible-looking but badly wrong values instead
        // of a loud, name-it failure. Mirrors `crate::safetensors::parse`'s
        // own F8_E5M2 handling exactly.
        "F8_E5M2" => panic!("'{name}': F8_E5M2 not supported (only F8_E4M3 is)"),
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
        // A per-call unique suffix, not just the PID: multiple tests call
        // make_file() and cargo runs them concurrently in one process, so a
        // PID-only path lets one test's std::fs::write race another's still-
        // active mmap (observed as a SIGBUS when the file underneath a live
        // mapping gets truncated/rewritten mid-test).
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("mmap_st_{}_{n}.safetensors", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        (path, a, b_vals)
    }

    /// Reading every tensor in turn (auto-dropping each one's pages as it's
    /// consumed, so host RSS never accumulates across a whole-file scan) must
    /// not corrupt a later re-read of an earlier tensor: `MADV_DONTNEED` on a
    /// read-only mapping only means "re-fault from the file if touched
    /// again", never data loss. This is the property the OOM fix on a
    /// multi-GB checkpoint depends on.
    #[test]
    fn consuming_a_tensor_evicts_its_pages_but_a_later_reread_is_still_correct() {
        let (path, a, b_vals) = make_file();
        let m = MmapSafetensors::open(&path).unwrap();

        // Force multiple chunk calls so with_tensor_chunks's own per-chunk
        // decode path is exercised, not just tensor_f32's whole-tensor path.
        let mut seen: Vec<f32> = Vec::new();
        assert!(m.with_tensor_chunks("a", 1, &mut |_, chunk| seen.extend_from_slice(chunk)));
        assert_eq!(seen, a);

        assert_eq!(m.tensor_f32("b").unwrap(), b_vals);

        // Re-read "a" after "b" was touched (and after "a"'s own pages were
        // advised away) -- must still decode byte-identically.
        let mut reread: Vec<f32> = Vec::new();
        assert!(m.with_tensor_chunks("a", 1, &mut |_, chunk| reread.extend_from_slice(chunk)));
        assert_eq!(reread, a);
        assert_eq!(m.tensor_f32("a").unwrap(), a);

        std::fs::remove_file(&path).ok();
    }

    /// `open()`/`tensor_f32`/`with_tensor_chunks` must handle F8_E4M3 exactly
    /// like `crate::safetensors::parse`'s own eager path does (they share
    /// `e4m3fn_to_f32`) - a real, blockwise-FP8 checkpoint (DeepSeek-V3-style
    /// scaling) is exactly the shape this mmap streaming path exists for, and
    /// until this test was added `open()` itself failed with "unknown dtype
    /// 'F8_E4M3'" before ever reaching a decode call at all (`try_dtype_width`
    /// never had an F8_E4M3 arm, unlike `crate::safetensors::parse`'s own
    /// `dtype` match, which got one when FP8 import landed).
    #[test]
    fn f8_e4m3_opens_and_decodes_through_both_the_whole_tensor_and_chunked_paths() {
        let raw_bytes: Vec<u8> = vec![0x00, 0x80, 0x38, 0xB8, 0x7E, 0x08]; // +0, -0, 1.0, -1.0, 448.0, 2^-6
        let want: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 448.0, 2f32.powi(-6)];
        let header = serde_json::json!({ "w": {"dtype": "F8_E4M3", "shape": [6], "data_offsets": [0, raw_bytes.len()]} });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut file = (hbytes.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&hbytes);
        file.extend_from_slice(&raw_bytes);
        let path = std::env::temp_dir().join(format!("mmap_st_f8e4m3_{}.safetensors", std::process::id()));
        std::fs::write(&path, &file).unwrap();

        let m = MmapSafetensors::open(&path).expect("open() must accept a file containing an F8_E4M3 tensor");
        assert_eq!(m.tensor_f32("w").unwrap(), want);

        let mut chunked: Vec<f32> = Vec::new();
        assert!(m.with_tensor_chunks("w", 2, &mut |_, chunk| chunked.extend_from_slice(chunk)));
        assert_eq!(chunked, want);

        std::fs::remove_file(&path).ok();
    }

    /// F8_E5M2 must still let `open()` succeed (a real checkpoint file may
    /// legitimately contain one even if this importer never reads it), but
    /// decoding it must fail loudly by name, never silently misdecode its
    /// bytes as E4M3 - mirrors `crate::safetensors::parse`'s own
    /// `f8_e5m2_errors_loudly_by_name_instead_of_silently_misdecoding` test.
    #[test]
    #[should_panic(expected = "F8_E5M2")]
    fn f8_e5m2_decode_panics_loudly_instead_of_silently_misdecoding() {
        let header = serde_json::json!({ "w": {"dtype": "F8_E5M2", "shape": [1], "data_offsets": [0, 1]} });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut file = (hbytes.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&hbytes);
        file.push(0x38);
        let path = std::env::temp_dir().join(format!("mmap_st_f8e5m2_{}.safetensors", std::process::id()));
        std::fs::write(&path, &file).unwrap();

        let m = MmapSafetensors::open(&path).expect("open() must accept a file containing an F8_E5M2 tensor");
        let _ = m.tensor_f32("w"); // must panic, not return a wrong value
    }

    /// Build a one-tensor safetensors file from raw little-endian bytes, with
    /// the header padded (via a `__metadata__` filler string, which `open()`
    /// stores separately and never treats as a tensor) so the tensor's byte
    /// offset relative to the mapping base is `blob_start % 4 == want_mod4`.
    /// This is what lets the alignment tests below construct BOTH the
    /// aligned and the misaligned case deterministically, rather than hoping
    /// serde_json's header length happens to land one way or the other.
    ///
    /// The file name carries a per-call serial number, NOT just
    /// `(dtype, want_mod4, pid)`: two of the callers below ask for the same
    /// `("BF16", 0)` fixture with DIFFERENT tensor lengths, and libtest runs
    /// them on separate threads of one process. Keying only on the arguments
    /// gave them one shared path, so whichever wrote second silently replaced
    /// the other's file mid-test - the 200k-element chunking test would then
    /// open a 4-element tensor and fail its "must span multiple chunks"
    /// assertion, roughly one run in three.
    fn make_file_at_alignment(dtype: &str, raw: &[u8], want_mod4: usize) -> std::path::PathBuf {
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for pad in 0..4 {
            let header = serde_json::json!({
                "t": {"dtype": dtype, "shape": [raw.len() / dtype_width(dtype)], "data_offsets": [0, raw.len()]},
                "__metadata__": {"pad": "x".repeat(pad)},
            });
            let hbytes = serde_json::to_vec(&header).unwrap();
            let blob_start = 8 + hbytes.len();
            if blob_start % 4 == want_mod4 {
                let mut file = (hbytes.len() as u64).to_le_bytes().to_vec();
                file.extend_from_slice(&hbytes);
                file.extend_from_slice(raw);
                let path = std::env::temp_dir().join(format!("mmap_st_align_{want_mod4}_{}_{}_{serial}.safetensors", dtype, std::process::id()));
                std::fs::write(&path, &file).unwrap();
                return path;
            }
        }
        unreachable!("pad 0..4 covers every residue mod 4");
    }

    #[test]
    fn raw_words_is_zero_copy_for_f32_and_none_for_bf16() {
        let a: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0];
        let raw: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = make_file_at_alignment("F32", &raw, 0);
        let mm = MmapSafetensors::open(&path).expect("open");
        let words = mm.raw_words("t").expect("F32, 4-byte aligned, must be zero-copyable");
        let want: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
        assert_eq!(words, want.as_slice());
        std::fs::remove_file(&path).ok();

        // BF16 never zero-copies -- the device never binds BF16 directly, so
        // raw_words must say "not zero-copyable" regardless of alignment.
        let bf_raw: Vec<u8> = [0.5f32, -1.0, 2.0].iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let bf_path = make_file_at_alignment("BF16", &bf_raw, 0);
        let bf = MmapSafetensors::open(&bf_path).expect("open");
        assert!(bf.raw_words("t").is_none(), "BF16 must never be zero-copied as u32 words");
        // ...but it still decodes correctly through the normal path.
        assert_eq!(bf.tensor_f32("t").unwrap(), vec![0.5, -1.0, 2.0]);
        std::fs::remove_file(&bf_path).ok();
    }

    #[test]
    fn misaligned_tensor_is_not_cast_but_still_decodes() {
        let a: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0];
        let raw: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Force the tensor's absolute byte offset to be 2 (mod 4) -- an F32
        // tensor that a naive `bytemuck::cast_slice` would misread, and that
        // `raw_words` must refuse rather than transmute.
        let path = make_file_at_alignment("F32", &raw, 2);
        let mm = MmapSafetensors::open(&path).expect("open");
        assert!(mm.raw_words("t").is_none(), "a misaligned F32 tensor must not be zero-copied");
        // The chunked and whole-tensor decode paths are unaffected -- they
        // read through `decode_into`, which has no alignment requirement.
        assert_eq!(mm.tensor_f32("t").unwrap(), a);
        let mut chunks = Vec::new();
        mm.with_tensor_chunks("t", 2, &mut |off, d| chunks.push((off, d.to_vec())));
        assert_eq!(chunks, vec![(0, vec![1.0, -2.5]), (2, vec![3.25, 0.0])]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn chunked_decode_bounds_host_peak_below_one_tensor() {
        // A BF16 tensor big enough that "decode the whole thing at once"
        // and "decode 4 KiB at a time" are easily distinguishable in bytes
        // retained, without needing a global-allocator harness (that lives
        // in weightio.rs and is exercised there for the end-to-end claim;
        // this test pins the mmap-level chunk boundary contract itself).
        let n = 200_000usize; // 400 KB as f32, 400 KB as BF16 raw
        let vals: Vec<f32> = (0..n).map(|i| (i % 997) as f32 * 0.125).collect();
        let raw: Vec<u8> = vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let path = make_file_at_alignment("BF16", &raw, 0);
        let mm = MmapSafetensors::open(&path).expect("open");

        let max_elems = 4096usize;
        let mut max_chunk_len = 0usize;
        let mut reassembled = Vec::with_capacity(n);
        let mut n_chunks = 0usize;
        mm.with_tensor_chunks("t", max_elems, &mut |off, d| {
            assert_eq!(off as usize, reassembled.len(), "chunks must arrive in order, contiguous");
            max_chunk_len = max_chunk_len.max(d.len());
            reassembled.extend_from_slice(d);
            n_chunks += 1;
        });
        assert!(n_chunks > 1, "the tensor must actually be split across multiple chunks");
        assert!(max_chunk_len <= max_elems, "no chunk may exceed max_elems ({max_chunk_len} > {max_elems})");
        // BF16 rounds each value's low 16 mantissa bits to zero -- expect the
        // rounded values, not the exact inputs (same rounding `tensor_f32`
        // already applies; this is what makes the equality below exact).
        assert_eq!(reassembled, mm.tensor_f32("t").unwrap(), "chunked reassembly must equal the whole-tensor decode exactly");
        std::fs::remove_file(&path).ok();
    }

    /// The common corruption in practice: a partially-downloaded multi-GB
    /// checkpoint whose header is intact but whose tensor blob is truncated.
    /// `open()` must refuse with a clean error naming the tensor — never
    /// panic at first tensor access deep inside an import or a serving
    /// lane's activate().
    #[test]
    fn truncated_blob_is_a_clean_open_error_not_a_panic() {
        let (path, _, _) = make_file();
        let bytes = std::fs::read(&path).unwrap();
        // Chop half the blob off (header stays intact).
        let truncated = &bytes[..bytes.len() - 8];
        let tpath = std::env::temp_dir().join(format!("mmap_st_trunc_{}.safetensors", std::process::id()));
        std::fs::write(&tpath, truncated).unwrap();
        let err = MmapSafetensors::open(&tpath).err().expect("truncated blob must be refused at open()");
        assert!(err.contains("truncated") || err.contains("exceeds"), "error must say the file is truncated: {err}");
        assert!(err.contains("tensor '"), "error must name the offending tensor: {err}");
        std::fs::remove_file(&tpath).ok();
        std::fs::remove_file(&path).ok();
    }

    /// Malformed header metadata (unknown dtype, inverted offsets, garbage
    /// data_offsets) — each must be a clean Err from open(), not a later panic.
    #[test]
    fn malformed_header_metadata_is_a_clean_open_error() {
        let cases = [
            (serde_json::json!({"t": {"dtype": "Q4_K", "shape": [4], "data_offsets": [0, 16]}}), "unknown dtype"),
            (serde_json::json!({"t": {"dtype": "F32", "shape": [4], "data_offsets": [16, 0]}}), "start"),
            (serde_json::json!({"t": {"dtype": "F32", "shape": [4], "data_offsets": ["x", 16]}}), "data_offsets"),
            (serde_json::json!({"t": {"dtype": "F32", "shape": [4], "data_offsets": [0]}}), "data_offsets"),
            (serde_json::json!({"t": {"dtype": "F32", "shape": ["x"], "data_offsets": [0, 16]}}), "shape"),
            (serde_json::json!({"t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 15]}}), "element width"),
        ];
        for (i, (header, want)) in cases.iter().enumerate() {
            let hbytes = serde_json::to_vec(header).unwrap();
            let mut file = (hbytes.len() as u64).to_le_bytes().to_vec();
            file.extend_from_slice(&hbytes);
            file.extend_from_slice(&[0u8; 16]);
            let path = std::env::temp_dir().join(format!("mmap_st_malformed_{i}_{}.safetensors", std::process::id()));
            std::fs::write(&path, &file).unwrap();
            let err = MmapSafetensors::open(&path).err().unwrap_or_else(|| panic!("case {i} must be refused"));
            assert!(err.contains(want), "case {i}: error {err:?} must mention {want:?}");
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn range_read_matches_the_corresponding_slice_of_a_full_decode() {
        // A BF16 "[rows, cols]"-shaped tensor (5 rows of 3), so a row read
        // exercises the same start-element/len-element arithmetic a real
        // embedding-table row lookup would use.
        let vals: Vec<f32> = (0..15).map(|i| i as f32 * 0.5 - 1.0).collect();
        let raw: Vec<u8> = vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let path = make_file_at_alignment("BF16", &raw, 0);
        let mm = MmapSafetensors::open(&path).expect("open");
        let full = mm.tensor_f32("t").unwrap();
        assert_eq!(full.len(), 15);

        // Row 2 of a [5,3] tensor: elements [6,9).
        let row2 = mm.tensor_f32_range("t", 6, 3).expect("row 2 must be readable");
        assert_eq!(row2, full[6..9]);

        // A single-element read and a read spanning the whole tensor both
        // agree with the full decode too.
        assert_eq!(mm.tensor_f32_range("t", 0, 1).unwrap(), full[0..1]);
        assert_eq!(mm.tensor_f32_range("t", 0, 15).unwrap(), full);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn range_read_out_of_bounds_returns_none_not_a_panic() {
        let a: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0];
        let raw: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = make_file_at_alignment("F32", &raw, 0);
        let mm = MmapSafetensors::open(&path).expect("open");

        // Range partially past the end.
        assert!(mm.tensor_f32_range("t", 2, 3).is_none());
        // Range starting past the end.
        assert!(mm.tensor_f32_range("t", 4, 1).is_none());
        // start_elem + len_elem overflowing usize must not panic either.
        assert!(mm.tensor_f32_range("t", usize::MAX, 1).is_none());
        // An unknown tensor name is also a clean None.
        assert!(mm.tensor_f32_range("nope", 0, 1).is_none());
        // A well-formed range still works after the out-of-bounds probes above.
        assert_eq!(mm.tensor_f32_range("t", 1, 2).unwrap(), vec![-2.5, 3.25]);

        std::fs::remove_file(&path).ok();
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
