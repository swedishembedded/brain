// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal safetensors reader (HuggingFace weight format), fp32 output.
//!
//! Format: `[u64 LE header_len][JSON header][raw tensor blob]`. The JSON maps
//! `tensor_name -> {dtype, shape:[...], data_offsets:[start,end]}` where the
//! offsets are *byte* ranges into the blob (plus a `__metadata__` entry we
//! skip). Tensors are dequantised to `f32` so the rest of brain's fp32-only
//! engine can consume them directly. Supports F32, F16, and BF16 (Qwen ships
//! bf16). Single-file only; sharded checkpoints (`*.index.json`) are handled by
//! the caller iterating the shard list.

use serde_json::Value;

/// A tensor read from a safetensors file: name, shape, and fp32 values.
pub struct StTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Decode an IEEE-754 binary16 (half) bit pattern to f32.
fn f16_to_f32(h: u16) -> f32 {
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
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
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
}
