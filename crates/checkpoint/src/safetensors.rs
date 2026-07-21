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
            "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            "F16" => raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            // Integer buffers (never learnable weights — e.g. Kronos's BSQ basis
            // buffers) are read as f32 so the whole file parses; callers skip
            // them by name. Exact for the small-int values these hold.
            "I64" => raw
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
            "I32" => raw
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                .collect(),
            "U8" => raw.iter().map(|&b| b as f32).collect(),
            other => return Err(format!("safetensors: unsupported dtype {other} for {name}")),
        };
        out.push(StTensor { name: name.clone(), shape, data });
    }
    Ok(out)
}

/// Read and parse a safetensors file from disk.
#[cfg(not(target_arch = "wasm32"))]
pub fn read(path: &str) -> Result<Vec<StTensor>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    parse(&bytes)
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
pub fn read_model_dir(dir: &std::path::Path) -> Result<Vec<StTensor>, String> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
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
    fn f16_half_decode() {
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0xC000), -2.0); // -2.0
        assert_eq!(f16_to_f32(0x0000), 0.0); // +0
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

        // Single-file layout in a fresh dir.
        let single = base.join("single");
        std::fs::create_dir_all(&single).unwrap();
        std::fs::write(single.join("model.safetensors"), one_tensor_bytes("c", &[9.0])).unwrap();
        let ts = read_model_dir(&single).unwrap();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].name, "c");

        std::fs::remove_dir_all(&base).ok();
    }
}
