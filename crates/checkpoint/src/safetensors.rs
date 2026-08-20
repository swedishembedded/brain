// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal safetensors reader (HuggingFace weight format), fp32 output.
//!
//! Format: `[u64 LE header_len][JSON header][raw tensor blob]`. The JSON maps
//! `tensor_name -> {dtype, shape:[...], data_offsets:[start,end]}` where the
//! offsets are *byte* ranges into the blob (plus a `__metadata__` entry we
//! skip). Tensors are dequantised to `f32` so the rest of brain's fp32-only
//! engine can consume them directly. Supports F32, F16, and BF16 (Qwen ships
//! bf16). [`parse`]/[`read`] handle one file; [`read_model_dir`] additionally
//! resolves sharded checkpoints (`model.safetensors.index.json` +
//! `model-0000k-of-000NN.safetensors`) by iterating the index `weight_map`.

use serde_json::Value;

/// A tensor read from a safetensors file: name, shape, and fp32 values.
pub struct StTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Decode an IEEE-754 binary16 (half) bit pattern to f32.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31 // signed zero
        } else {
            // subnormal: normalise
            let mut e = -1i32;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            let mant = (m & 0x3ff) << 13;
            let exp = (127 - 15 - e) as u32;
            (sign << 31) | (exp << 23) | mant
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13) // inf / nan
    } else {
        let exp = exp + (127 - 15);
        (sign << 31) | (exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Decode a bfloat16 bit pattern to f32 (bf16 is the top 16 bits of an f32).
pub(crate) fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Decode an `E4M3FN` (OCP/PyTorch `float8_e4m3fn`) byte to its OWN f32
/// value: 1 sign, 4 exponent (bias 7), 3 mantissa bits, **no infinities**
/// (the "fn" variant): only `0x7F`/`0xFF` are NaN, so `exponent=0b1111` with
/// any OTHER mantissa is a regular (the largest) finite value (max magnitude
/// `1.75 * 2^8 = 448`). This is the raw per-element value ONLY, a
/// `weight_block_size`-quantized checkpoint (DeepSeek-V3/Qwen3.5-FP8 style)
/// needs a SEPARATE blockwise scale multiply on top, which is not a
/// byte-decode concern and does not belong here (see `model::fp8`).
///
/// Only 256 distinct input bytes exist, so this is a lookup into a table
/// built once from [`e4m3fn_to_f32_scalar`] rather than re-deriving the
/// value (branches + `powi`) on every one of a real tensor's tens of
/// millions of elements - measured as the dominant cost of importing a real
/// FP8 checkpoint layer, far more than the blockwise-scale multiply or the
/// int8 requantization downstream of it.
pub fn e4m3fn_to_f32(b: u8) -> f32 {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|i| e4m3fn_to_f32_scalar(i as u8)))[b as usize]
}

/// Decode a whole `F8_E4M3` byte buffer to f32 via [`e4m3fn_to_f32`]'s LUT -
/// the one implementation [`parse`] (whole-file, eager) and
/// `crate::mmap::decode_into` (whole-tensor/chunked, on-demand) both share,
/// so they cannot drift.
///
/// Real per-layer profiling of `qwen35::stream::generate`'s streaming decode
/// path (`crates/qwen35/tests/stream_profile.rs`) found this exact loop -
/// even with the O(1) LUT above, which already cut the PER-ELEMENT cost by
/// ~90% over the branches+`powi` scalar path - to still be the single
/// largest real stage of a decode step: ~11-23s of a ~15-28s real per-layer
/// total on this box (a real 372-383 MB layer, one FP8 byte per element),
/// dwarfing the already-parallel `model::fp8::dequant_block128` block-scale
/// multiply downstream of it and the GPU forward compute after that (both
/// well under a second). The LUT made each element's OWN cost O(1); it never
/// made the WALK over tens of millions of elements per tensor use more than
/// one core. Native builds fix that here, the same way `model::int8::
/// quantize_weight`/`model::fp8::dequant_block128` already fan their own
/// host-parallel work out across `backend_cpu::par` ("rayon lives in exactly
/// one crate" - that module's own doc) rather than adding a second, competing
/// thread pool. wasm32 has no `backend_cpu` (Cranelift JIT + rayon do not
/// target it) and keeps the sequential loop - identical math, not fanned out.
pub fn decode_e4m3_bytes(raw: &[u8]) -> Vec<f32> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut out = vec![0f32; raw.len()];
        backend_cpu::par::each_mut(&mut out, |i, v| *v = e4m3fn_to_f32(raw[i]));
        out
    }
    #[cfg(target_arch = "wasm32")]
    {
        raw.iter().map(|&b| e4m3fn_to_f32(b)).collect()
    }
}

fn e4m3fn_to_f32_scalar(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (b >> 3) & 0x0F;
    let mant = (b & 0x07) as f32;
    if exp == 0x0F && b & 0x07 == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        // subnormal: mant/8 * 2^(1-bias), bias=7 -> 2^-6
        return sign * (mant / 8.0) * 2f32.powi(-6);
    }
    // normal: (1 + mant/8) * 2^(exp-bias)
    sign * (1.0 + mant / 8.0) * 2f32.powi(exp as i32 - 7)
}

/// Decode `raw` as a packed stream of little-endian elements, `width` bytes
/// each, converting element `i` with `f`. Trailing bytes shorter than one
/// element are ignored, exactly as `chunks_exact` did.
///
/// Element-wise and index-preserving, so splitting the stream across threads
/// cannot change a value - element `i` reads only its own `width` bytes and
/// writes only `out[i]`. That is what makes this a scheduling change: the
/// conversion of a large checkpoint is bit-identical to the serial form, and
/// `parse_is_schedule_invariant_across_every_dtype` pins it by bit pattern.
///
/// Worth parallelising because dtype conversion is not incidental work on a
/// real checkpoint: a multi-billion-parameter bf16 file is billions of
/// independent 2-byte decodes, and it was running on one core while the other
/// forty-seven idled.
#[cfg(not(target_arch = "wasm32"))]
fn decode_elems(raw: &[u8], width: usize, f: fn(&[u8]) -> f32) -> Vec<f32> {
    // Large enough that the per-chunk dispatch is noise, small enough that a
    // lopsided tensor still spreads over the pool.
    const CHUNK: usize = 1 << 16;
    let n = raw.len() / width;
    let mut out = vec![0f32; n];
    backend_cpu::par::chunks_mut(&mut out, CHUNK, |c, dst| {
        let base = c * CHUNK;
        for (j, v) in dst.iter_mut().enumerate() {
            let i = base + j;
            *v = f(&raw[i * width..(i + 1) * width]);
        }
    });
    out
}

/// The serial decode [`decode_elems`] documents - wasm has no threads and does
/// not build `backend-cpu`.
#[cfg(target_arch = "wasm32")]
fn decode_elems(raw: &[u8], width: usize, f: fn(&[u8]) -> f32) -> Vec<f32> {
    raw.chunks_exact(width).map(f).collect()
}

/// Parse a safetensors byte buffer into fp32 tensors (declared order preserved).
pub fn parse(bytes: &[u8]) -> Result<Vec<StTensor>, String> {
    if bytes.len() < 8 {
        return Err("safetensors: file too short".into());
    }
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hend = 8 + hlen;
    if bytes.len() < hend {
        return Err("safetensors: truncated header".into());
    }
    let header: Value = serde_json::from_slice(&bytes[8..hend])
        .map_err(|e| format!("safetensors: bad header json: {e}"))?;
    let blob = &bytes[hend..];
    let obj = header.as_object().ok_or("safetensors: header is not an object")?;

    let mut out = Vec::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta["dtype"].as_str().ok_or("safetensors: missing dtype")?;
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .ok_or("safetensors: missing shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = meta["data_offsets"].as_array().ok_or("safetensors: missing data_offsets")?;
        let start = off[0].as_u64().unwrap() as usize;
        let end = off[1].as_u64().unwrap() as usize;
        let raw = &blob[start..end];

        let data: Vec<f32> = match dtype {
            "F32" => decode_elems(raw, 4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            "F16" => decode_elems(raw, 2, |b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            "BF16" => decode_elems(raw, 2, |b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            // Integer buffers (never learnable weights — e.g. Kronos's BSQ basis
            // buffers) are read as f32 so the whole file parses; callers skip
            // them by name. Exact for the small-int values these hold.
            "I64" => decode_elems(raw, 8, |b| i64::from_le_bytes(b.try_into().unwrap()) as f32),
            "I32" => decode_elems(raw, 4, |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32),
            "U8" => decode_elems(raw, 1, |b| b[0] as f32),
            // `decode_e4m3_bytes`, not `decode_elems`+`e4m3fn_to_f32` inline -
            // `crate::mmap::decode_into`'s chunked/on-demand path shares this
            // exact function (see its own doc), so parsing E4M3 any other way
            // here would let the two decode paths drift.
            "F8_E4M3" => decode_e4m3_bytes(raw),
            // Named explicitly rather than falling into the generic `other`
            // arm below: E5M2 is a real, if rarer, FP8 checkpoint format
            // (more exponent range, less mantissa) - silently decoding its
            // bytes as if they were E4M3 would produce plausible-looking but
            // badly wrong values instead of a loud, name-it error.
            "F8_E5M2" => return Err(format!("safetensors: F8_E5M2 not supported for {name} (only F8_E4M3 is)")),
            other => return Err(format!("safetensors: unsupported dtype {other} for {name}")),
        };
        out.push(StTensor { name: name.clone(), shape, data });
    }
    Ok(out)
}

/// Read and parse a safetensors file from disk.
///
/// The file is MAPPED, not slurped. `std::fs::read` would place a second,
/// anonymous copy of every byte on the heap alongside the fp32 tensors
/// [`parse`] is building from it - so a large bf16 checkpoint peaked at
/// roughly three times its own size in resident memory (the raw copy plus its
/// fp32 expansion) when two are enough. On a machine where the checkpoints
/// being read are a large fraction of RAM, that extra copy is not merely
/// wasteful: it evicts page cache, which is where the bytes of the NEXT
/// checkpoint the process reads would otherwise still be sitting. Mapping
/// leaves those pages file-backed and reclaimable instead of duplicating them
/// into anonymous memory the kernel cannot drop.
///
/// The parsed result is unchanged - [`parse`] sees exactly the same bytes.
#[cfg(not(target_arch = "wasm32"))]
pub fn read(path: &str) -> Result<Vec<StTensor>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    // SAFETY: weight files are treated as immutable for the mapping's
    // lifetime - the same contract `crate::mmap::MmapSafetensors::open` and
    // `crate::gguf::MmapGguf::open` already rely on for every mapped
    // checkpoint in this crate.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("cannot mmap {path}: {e}"))?;
    parse(&mmap)
}

/// Read all tensors from a HuggingFace model directory, handling both the
/// single-file (`model.safetensors`) and sharded
/// (`model.safetensors.index.json` + `model-0000k-of-000NN.safetensors`)
/// layouts. Large models (e.g. GLM) ship sharded; small ones (e.g. Qwen3-0.6B)
/// ship a single file. When both are present the index wins.
///
/// Tensors are returned in a deterministic order: shards are read in the sorted
/// order they first appear in the index `weight_map`, and within each shard the
/// declared (file) order is preserved. The caller (an importer) remaps by name,
/// so the exact interleaving does not matter — only that every tensor appears
/// exactly once.
#[cfg(not(target_arch = "wasm32"))]
/// HF-transformers and diffusers checkpoints name their shard index
/// differently (`model.safetensors.index.json` vs.
/// `diffusion_pytorch_model.safetensors.index.json`); both must resolve to
/// the same sharded-read behavior.
fn index_filename(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    ["model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"]
        .into_iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())
}

pub fn read_model_dir(dir: &std::path::Path) -> Result<Vec<StTensor>, String> {
    if let Some(index) = index_filename(dir) {
        let idx_bytes =
            std::fs::read(&index).map_err(|e| format!("cannot read {}: {e}", index.display()))?;
        let idx: Value = serde_json::from_slice(&idx_bytes)
            .map_err(|e| format!("safetensors index: bad json: {e}"))?;
        let map = idx["weight_map"]
            .as_object()
            .ok_or("safetensors index: missing weight_map object")?;
        // Unique shard filenames in first-seen order, then sorted for determinism.
        let mut shards: Vec<String> = Vec::new();
        for shard in map.values() {
            let s = shard.as_str().ok_or("safetensors index: non-string shard name")?;
            if !shards.iter().any(|x| x == s) {
                shards.push(s.to_string());
            }
        }
        shards.sort();
        if shards.is_empty() {
            return Err("safetensors index: empty weight_map".into());
        }
        let mut out = Vec::new();
        for shard in &shards {
            let p = dir.join(shard);
            out.extend(read(p.to_str().ok_or("safetensors: non-utf8 shard path")?)?);
        }
        return Ok(out);
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return read(single.to_str().ok_or("safetensors: non-utf8 path")?);
    }
    Err(format!(
        "no model.safetensors or model.safetensors.index.json in {}",
        dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_and_f32_roundtrip() {
        // Build a tiny in-memory safetensors with one F32 and one BF16 tensor.
        let header = serde_json::json!({
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]},
        });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hbytes);
        buf.extend_from_slice(&1.5f32.to_le_bytes());
        buf.extend_from_slice(&(-2.0f32).to_le_bytes());
        // bf16 of 1.0 = 0x3F80, of -4.0 = 0xC080
        buf.extend_from_slice(&0x3F80u16.to_le_bytes());
        buf.extend_from_slice(&0xC080u16.to_le_bytes());

        let ts = parse(&buf).unwrap();
        let a = ts.iter().find(|t| t.name == "a").unwrap();
        let b = ts.iter().find(|t| t.name == "b").unwrap();
        assert_eq!(a.data, vec![1.5, -2.0]);
        assert_eq!(b.data, vec![1.0, -4.0]);
        assert_eq!(b.shape, vec![2]);
    }

    #[test]
    fn f8_e4m3_parses_via_the_full_reader() {
        let header = serde_json::json!({
            "w": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]},
        });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hbytes);
        buf.push(0x38); // 1.0
        buf.push(0xB8); // -1.0
        let ts = parse(&buf).unwrap();
        let w = ts.iter().find(|t| t.name == "w").unwrap();
        assert_eq!(w.data, vec![1.0, -1.0]);
    }

    #[test]
    fn f8_e5m2_errors_loudly_by_name_instead_of_silently_misdecoding() {
        let header = serde_json::json!({
            "w": {"dtype": "F8_E5M2", "shape": [1], "data_offsets": [0, 1]},
        });
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hbytes);
        buf.push(0x38);
        let err = match parse(&buf) {
            Ok(_) => panic!("F8_E5M2 must not parse"),
            Err(e) => e,
        };
        assert!(err.contains("F8_E5M2"), "error must name the unsupported dtype: {err}");
        assert!(err.contains('w'), "error must name the tensor: {err}");
    }

    #[test]
    fn f16_half_decode() {
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0xC000), -2.0); // -2.0
        assert_eq!(f16_to_f32(0x0000), 0.0); // +0
    }

    #[test]
    fn e4m3fn_decode_known_values() {
        assert_eq!(e4m3fn_to_f32(0x00), 0.0); // +0
        assert_eq!(e4m3fn_to_f32(0x80), -0.0); // -0
        assert_eq!(e4m3fn_to_f32(0x38), 1.0); // exp=0111(7,bias7->0), mant=0 -> 1.0
        assert_eq!(e4m3fn_to_f32(0xB8), -1.0);
        assert_eq!(e4m3fn_to_f32(0x7E), 448.0); // exp=1111, mant=110 -> the known e4m3fn max
        assert!(e4m3fn_to_f32(0x7F).is_nan()); // the ONE reserved NaN pattern (positive)
        assert!(e4m3fn_to_f32(0xFF).is_nan()); // and its negative twin
        // exp=1111 with mantissa != 111 is a REGULAR finite value, unlike
        // standard IEEE float - this is the "fn" (no-infinity) variant's
        // whole point, and the one easiest detail to get wrong.
        assert!(e4m3fn_to_f32(0x7D).is_finite());
        assert_eq!(e4m3fn_to_f32(0x08), 2f32.powi(-6)); // smallest normal (exp=1,mant=0)
        assert_eq!(e4m3fn_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal (exp=0,mant=1)
    }

    /// Build a one-tensor F32 safetensors byte buffer for the shard tests.
    #[cfg(not(target_arch = "wasm32"))]
    fn one_tensor_bytes(name: &str, vals: &[f32]) -> Vec<u8> {
        let hdr = serde_json::json!({
            name: {"dtype": "F32", "shape": [vals.len()], "data_offsets": [0, vals.len() * 4]},
        });
        let hb = serde_json::to_vec(&hdr).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hb);
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn read_model_dir_sharded_and_single() {
        let base = std::env::temp_dir().join(format!("brain-st-shard-{}", std::process::id()));
        // Sharded layout: two shards + an index weight_map.
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("model-00001-of-00002.safetensors"), one_tensor_bytes("a", &[1.0, 2.0]))
            .unwrap();
        std::fs::write(base.join("model-00002-of-00002.safetensors"), one_tensor_bytes("b", &[3.0]))
            .unwrap();
        let index = serde_json::json!({
            "metadata": {"total_size": 12},
            "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"},
        });
        std::fs::write(base.join("model.safetensors.index.json"), serde_json::to_vec(&index).unwrap())
            .unwrap();

        let ts = read_model_dir(&base).unwrap();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.iter().find(|t| t.name == "a").unwrap().data, vec![1.0, 2.0]);
        assert_eq!(ts.iter().find(|t| t.name == "b").unwrap().data, vec![3.0]);

        // Diffusers-style index filename (Z-Image's `transformer/` dir ships
        // this, not `model.safetensors.index.json`) must be recognized too.
        let diffusers = base.join("diffusers");
        std::fs::create_dir_all(&diffusers).unwrap();
        std::fs::write(diffusers.join("diffusion_pytorch_model-00001-of-00002.safetensors"), one_tensor_bytes("a", &[1.0, 2.0])).unwrap();
        std::fs::write(diffusers.join("diffusion_pytorch_model-00002-of-00002.safetensors"), one_tensor_bytes("b", &[3.0])).unwrap();
        let diffusers_index = serde_json::json!({
            "metadata": {"total_size": 12},
            "weight_map": {
                "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                "b": "diffusion_pytorch_model-00002-of-00002.safetensors",
            },
        });
        std::fs::write(diffusers.join("diffusion_pytorch_model.safetensors.index.json"), serde_json::to_vec(&diffusers_index).unwrap()).unwrap();
        let ts = read_model_dir(&diffusers).unwrap();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.iter().find(|t| t.name == "a").unwrap().data, vec![1.0, 2.0]);
        assert_eq!(ts.iter().find(|t| t.name == "b").unwrap().data, vec![3.0]);

        // Single-file layout in a fresh dir.
        let single = base.join("single");
        std::fs::create_dir_all(&single).unwrap();
        std::fs::write(single.join("model.safetensors"), one_tensor_bytes("c", &[9.0])).unwrap();
        let ts = read_model_dir(&single).unwrap();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].name, "c");

        std::fs::remove_dir_all(&base).ok();
    }

    /// Every dtype's decode is element-wise, so splitting the element stream
    /// across threads must be a scheduling change and nothing else. Compared
    /// by BIT PATTERN against a serial `chunks_exact` oracle: the fixture is
    /// random bytes, which legitimately decode to NaN for the float types, and
    /// `NaN != NaN` would fail a value comparison between two byte-identical
    /// results. Bits are also the stronger claim.
    ///
    /// Lengths deliberately straddle the internal chunk size and are not
    /// multiples of it, so a chunk-boundary or tail-handling mistake shows up
    /// (lesson #4: a fixture that divides evenly hides exactly this).
    #[test]
    fn parse_is_schedule_invariant_across_every_dtype() {
        fn noise(n: usize, seed: u64) -> Vec<u8> {
            let mut s = seed | 1;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 24) as u8
                })
                .collect()
        }
        /// One dtype's `(byte width, element decoder)` - the exact pair
        /// `decode_elems` takes, named so the case table is not an inline
        /// function-pointer type.
        type DtypeCase = (usize, fn(&[u8]) -> f32);
        let cases: [DtypeCase; 6] = [
            (4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            (2, |b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            (2, |b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            (8, |b| i64::from_le_bytes(b.try_into().unwrap()) as f32),
            (1, |b| b[0] as f32),
            (1, |b| e4m3fn_to_f32(b[0])),
        ];
        for (ci, (width, f)) in cases.into_iter().enumerate() {
            for elems in [0usize, 1, 1000, (1 << 16) + 7, 3 * (1 << 16) - 1] {
                let raw = noise(elems * width + width - 1, 0xC0FFEE + ci as u64 * 31 + elems as u64);
                let got = decode_elems(&raw, width, f);
                let want: Vec<f32> = raw.chunks_exact(width).map(f).collect();
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                assert_eq!(bits(&got), bits(&want), "dtype case {ci}, {elems} elements of {width} bytes");
            }
        }
    }
}
