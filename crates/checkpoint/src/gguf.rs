// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal GGUF v3 reader: unquantized tensors (F32 / F16 / BF16) only.
//!
//! Enough to import the FLUX.2 Klein 9B transformer, which is redistributed as
//! a BF16 GGUF carrying the BFL reference tensor names. Quantized GGUF types
//! (Q4_K, Q8_0, …) are out of scope — importing lossy weights would silently
//! break parity gates, so they error by type id instead.
//!
//! GGUF stores dims fastest-varying first (`ne[0]` = innermost); torch/brain
//! shapes are the reverse. The byte layout of the data itself is identical
//! row-major, so only the shape vector is reversed on read.

use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::safetensors::StTensor;

const T_F32: u32 = 0;
const T_F16: u32 = 1;
const T_BF16: u32 = 30;

struct Rd<R: Read>(R);

impl<R: Read> Rd<R> {
    fn u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.0.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.0.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u64::from_le_bytes(b))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        let mut b = vec![0u8; n];
        self.0.read_exact(&mut b).map_err(|e| e.to_string())?;
        String::from_utf8(b).map_err(|e| e.to_string())
    }
    fn skip(&mut self, n: u64) -> Result<(), String> {
        std::io::copy(&mut (&mut self.0).take(n), &mut std::io::sink())
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    /// Skip one metadata value of the given GGUF value type; strings and
    /// arrays are length-prefixed, scalars fixed-width.
    fn skip_value(&mut self, ty: u32) -> Result<(), String> {
        match ty {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4 | 5 | 6 => self.skip(4),
            10 | 11 | 12 => self.skip(8),
            8 => {
                let n = self.u64()?;
                self.skip(n)
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip_value(et)?;
                }
                Ok(())
            }
            other => Err(format!("gguf: unknown metadata value type {other}")),
        }
    }
}

/// Read every tensor of an unquantized GGUF file, dequantizing F16/BF16 to f32.
pub fn read(path: &str) -> Result<Vec<StTensor>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("gguf: open {path}: {e}"))?;
    let mut r = Rd(BufReader::new(f));

    let mut magic = [0u8; 4];
    r.0.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"GGUF" {
        return Err(format!("gguf: bad magic {magic:?} in {path}"));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("gguf: unsupported version {version}"));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;

    let mut alignment: u64 = 32;
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        if key == "general.alignment" && ty == 4 {
            alignment = r.u32()? as u64;
        } else {
            r.skip_value(ty)?;
        }
    }

    struct Info {
        name: String,
        shape: Vec<usize>, // torch order (already reversed)
        ty: u32,
        offset: u64,
    }
    let mut infos = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()? as usize;
        let mut ne = Vec::with_capacity(nd);
        for _ in 0..nd {
            ne.push(r.u64()? as usize);
        }
        ne.reverse();
        let ty = r.u32()?;
        let offset = r.u64()?;
        infos.push(Info { name, shape: ne, ty, offset });
    }

    let header_end = r.0.stream_position().map_err(|e| e.to_string())?;
    let data_start = header_end.div_ceil(alignment) * alignment;

    let mut f = r.0.into_inner();
    let mut out = Vec::with_capacity(infos.len());
    for info in infos {
        let numel: usize = info.shape.iter().product();
        let (bytes_per, is_f32) = match info.ty {
            T_F32 => (4usize, true),
            T_F16 | T_BF16 => (2usize, false),
            other => {
                return Err(format!(
                    "gguf: tensor {} has quantized/unsupported type {other} \
                     (only F32/F16/BF16 are importable)",
                    info.name
                ))
            }
        };
        f.seek(SeekFrom::Start(data_start + info.offset))
            .map_err(|e| e.to_string())?;
        let mut raw = vec![0u8; numel * bytes_per];
        f.read_exact(&mut raw)
            .map_err(|e| format!("gguf: reading {}: {e}", info.name))?;
        let data = if is_f32 {
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else if info.ty == T_BF16 {
            raw.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        } else {
            raw.chunks_exact(2)
                .map(|c| crate::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        };
        out.push(StTensor { name: info.name, shape: info.shape, data });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn put_str(v: &mut Vec<u8>, s: &str) {
        v.extend((s.len() as u64).to_le_bytes());
        v.extend(s.as_bytes());
    }

    /// Build a tiny two-tensor GGUF (one BF16, one F32) in memory and read it.
    #[test]
    fn synthetic_roundtrip() {
        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes());
        h.extend(2u64.to_le_bytes()); // tensors
        h.extend(1u64.to_le_bytes()); // kv
        put_str(&mut h, "general.architecture");
        h.extend(8u32.to_le_bytes());
        put_str(&mut h, "flux");
        // tensor 0: bf16 [2,3] (torch) -> gguf ne [3,2]
        put_str(&mut h, "a.weight");
        h.extend(2u32.to_le_bytes());
        h.extend(3u64.to_le_bytes());
        h.extend(2u64.to_le_bytes());
        h.extend(30u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        // tensor 1: f32 [4]
        put_str(&mut h, "b.scale");
        h.extend(1u32.to_le_bytes());
        h.extend(4u64.to_le_bytes());
        h.extend(0u32.to_le_bytes());
        h.extend(32u64.to_le_bytes()); // offset within data (aligned)
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        let vals = [1.0f32, -2.0, 0.5, 3.0, -0.25, 8.0];
        for v in vals {
            let bits = (v.to_bits() >> 16) as u16; // exact in bf16
            h.extend(bits.to_le_bytes());
        }
        h.resize(data_start + 32, 0);
        for v in [9.0f32, -1.5, 0.0, 2.5] {
            h.extend(v.to_le_bytes());
        }

        let dir = std::env::temp_dir().join(format!("gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        std::fs::File::create(&path).unwrap().write_all(&h).unwrap();

        let ts = read(path.to_str().unwrap()).unwrap();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts[0].name, "a.weight");
        assert_eq!(ts[0].shape, vec![2, 3]);
        assert_eq!(ts[0].data, vals);
        assert_eq!(ts[1].shape, vec![4]);
        assert_eq!(ts[1].data, [9.0, -1.5, 0.0, 2.5]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn quantized_types_are_rejected() {
        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes());
        h.extend(1u64.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        put_str(&mut h, "q.weight");
        h.extend(1u32.to_le_bytes());
        h.extend(32u64.to_le_bytes());
        h.extend(12u32.to_le_bytes()); // Q4_K
        h.extend(0u64.to_le_bytes());
        h.resize(h.len().div_ceil(32) * 32 + 64, 0);

        let dir = std::env::temp_dir().join(format!("gguf-test-q-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q.gguf");
        std::fs::File::create(&path).unwrap().write_all(&h).unwrap();
        let Err(err) = read(path.to_str().unwrap()) else {
            panic!("quantized gguf must be rejected");
        };
        assert!(err.contains("quantized"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
