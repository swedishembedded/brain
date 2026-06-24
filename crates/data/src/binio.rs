// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! On-disk dataset formats. Self-contained and pure-Rust; mirrors nanogpt's
//! layout except `meta.pkl` (Python pickle) becomes `meta.json`, and the
//! time-series `.npy` floats become a raw little-endian `f32` blob + JSON shape.
//!
//! Layouts written under `<dir>/`:
//!   * token datasets — `train.bin` / `val.bin` : raw little-endian `u16`
//!     token arrays (GPT-2 vocab 50257 fits in `u16`, as in nanogpt); `input.txt`
//!     (the source text); and, for char-level datasets, `meta.json`.
//!   * float datasets — `train.f32` / `val.f32` : raw little-endian `f32`, plus
//!     `meta.json` carrying `{n_features, rows}`.

use std::fs;
use std::io;
use std::path::Path;

/// Character-level vocabulary metadata (`meta.json`).
#[derive(Clone, Debug)]
pub struct Meta {
    pub vocab_size: usize,
    /// id -> char, indexed by token id.
    pub itos: Vec<char>,
}

impl Meta {
    /// char -> id lookup (built on demand).
    pub fn stoi(&self) -> std::collections::HashMap<char, u16> {
        self.itos
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u16))
            .collect()
    }

    pub fn to_json(&self) -> String {
        let itos: serde_json::Map<String, serde_json::Value> = self
            .itos
            .iter()
            .enumerate()
            .map(|(i, &c)| (i.to_string(), serde_json::Value::String(c.to_string())))
            .collect();
        serde_json::json!({
            "vocab_size": self.vocab_size,
            "itos": itos,
        })
        .to_string()
    }

    pub fn from_json(s: &str) -> Result<Meta, String> {
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let vocab_size = v["vocab_size"].as_u64().ok_or("meta: vocab_size")? as usize;
        let mut itos = vec!['\0'; vocab_size];
        let map = v["itos"].as_object().ok_or("meta: itos")?;
        for (k, val) in map {
            let id: usize = k.parse().map_err(|_| "meta: itos key")?;
            let ch = val.as_str().and_then(|s| s.chars().next()).ok_or("meta: itos val")?;
            if id < vocab_size {
                itos[id] = ch;
            }
        }
        Ok(Meta { vocab_size, itos })
    }
}

/// Write a `u16` token array as a raw little-endian `.bin` file.
pub fn write_u16_bin(path: &Path, tokens: &[u16]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(tokens.len() * 2);
    for &t in tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    fs::write(path, bytes)
}

/// Read a raw little-endian `u16` `.bin` token array.
pub fn read_u16_bin(path: &Path) -> io::Result<Vec<u16>> {
    let bytes = fs::read(path)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Write a `f32` array as a raw little-endian `.f32` file.
pub fn write_f32_bin(path: &Path, values: &[f32]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes)
}

/// Read a raw little-endian `.f32` array.
pub fn read_f32_bin(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_json_roundtrips() {
        let m = Meta {
            vocab_size: 4,
            itos: vec!['\n', '=', 'a', 'b'],
        };
        let back = Meta::from_json(&m.to_json()).unwrap();
        assert_eq!(back.vocab_size, 4);
        assert_eq!(back.itos, m.itos);
        assert_eq!(back.stoi()[&'='], 1);
    }
}
