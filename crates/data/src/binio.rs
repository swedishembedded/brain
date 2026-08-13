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
//!   * large-vocab token datasets — `train.u32.bin` / `val.u32.bin` : raw
//!     little-endian `u32` token arrays, for vocabularies that overflow `u16`
//!     (e.g. Qwen's 151936). `meta.json` carries `"token_width": 32`.
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
    pub fn stoi(&self) -> std::collections::HashMap<char, u32> {
        self.itos
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
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
        // `itos` is optional: large-vocab (BPE) datasets record only `vocab_size`
        // (no per-char table). Char datasets carry the full id->char map.
        let mut itos = Vec::new();
        if let Some(map) = v["itos"].as_object() {
            // vocab_size comes from an untrusted meta.json and sizes this
            // allocation; a per-CHAR table beyond 2^20 entries is not a real
            // char dataset, it is a corrupt or hostile file.
            if vocab_size > (1 << 20) {
                return Err(format!("meta: vocab_size {vocab_size} is implausible for a char-level itos table (max {})", 1 << 20));
            }
            itos = vec!['\0'; vocab_size];
            for (k, val) in map {
                let id: usize = k.parse().map_err(|_| "meta: itos key")?;
                let ch = val.as_str().and_then(|s| s.chars().next()).ok_or("meta: itos val")?;
                if id < vocab_size {
                    itos[id] = ch;
                }
            }
        }
        Ok(Meta { vocab_size, itos })
    }

    /// Minimal metadata for a large-vocab (BPE) dataset: just the vocab size,
    /// no char table. Serialized as `{"vocab_size":N,"token_width":32}`.
    pub fn vocab_only(vocab_size: usize) -> String {
        serde_json::json!({ "vocab_size": vocab_size, "token_width": 32 }).to_string()
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

/// Read a raw little-endian `u16` `.bin` token array. A trailing partial
/// element is an `InvalidData` error — a truncated download must fail here,
/// not silently load one token short and TRAIN on it.
pub fn read_u16_bin(path: &Path) -> io::Result<Vec<u16>> {
    let bytes = fs::read(path)?;
    reject_trailing(path, bytes.len(), 2)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Write a `u32` token array as a raw little-endian `.bin` file. Used for
/// large-vocabulary datasets (e.g. Qwen) whose ids overflow `u16`.
pub fn write_u32_bin(path: &Path, tokens: &[u32]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &t in tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    fs::write(path, bytes)
}

/// Read a raw little-endian `u32` `.bin` token array. Truncation is an
/// error — see [`read_u16_bin`].
pub fn read_u32_bin(path: &Path) -> io::Result<Vec<u32>> {
    let bytes = fs::read(path)?;
    reject_trailing(path, bytes.len(), 4)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Read a token split as `u32`, choosing the width from what exists on disk:
/// a `<stem>.u32.bin` (raw `u32`) takes precedence over `<stem>.bin` (raw
/// `u16`, upcast). `stem` is e.g. `dir.join("train")`. This lets the same
/// training loop consume both small-vocab (u16) and large-vocab (u32) datasets.
pub fn read_tokens_u32(stem: &Path) -> io::Result<Vec<u32>> {
    let u32_path = stem.with_extension("u32.bin");
    if u32_path.exists() {
        return read_u32_bin(&u32_path);
    }
    let u16_path = stem.with_extension("bin");
    Ok(read_u16_bin(&u16_path)?.into_iter().map(|t| t as u32).collect())
}

/// Write a `f32` array as a raw little-endian `.f32` file.
pub fn write_f32_bin(path: &Path, values: &[f32]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes)
}

/// Read a raw little-endian `.f32` array. Truncation is an error — see
/// [`read_u16_bin`].
pub fn read_f32_bin(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    reject_trailing(path, bytes.len(), 4)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// `chunks_exact` silently DISCARDS a trailing partial element — for a raw
/// `.bin` a partial tail means the file is truncated or the wrong width, and
/// the exact "silent truncation" class the repo's validate-everything rule
/// names. One shared refusal for every raw reader above.
fn reject_trailing(path: &Path, len: usize, width: usize) -> io::Result<()> {
    if len % width != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{}: {len} bytes is not a multiple of the {width}-byte element width — truncated or wrong-format file",
                path.display()
            ),
        ));
    }
    Ok(())
}

/// A normalized ground-truth box for object detection: class id + center-xywh in
/// `[0,1]` of image size. Mirrors `yolov8::GtBox` (minus the per-batch `img`
/// index, which the loader assigns when batching).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectBox {
    pub class: u32,
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// Write per-image detection boxes to `boxes.bin`. Layout, per image in order:
/// `[u32 num]` then `num × (u32 class, f32 cx, f32 cy, f32 w, f32 h)`, all raw
/// little-endian. `boxes[i]` is image `i`'s box list (may be empty).
pub fn write_detect_boxes(path: &Path, boxes: &[Vec<DetectBox>]) -> io::Result<()> {
    let mut bytes = Vec::new();
    for img in boxes {
        bytes.extend_from_slice(&(img.len() as u32).to_le_bytes());
        for b in img {
            bytes.extend_from_slice(&b.class.to_le_bytes());
            bytes.extend_from_slice(&b.cx.to_le_bytes());
            bytes.extend_from_slice(&b.cy.to_le_bytes());
            bytes.extend_from_slice(&b.w.to_le_bytes());
            bytes.extend_from_slice(&b.h.to_le_bytes());
        }
    }
    fs::write(path, bytes)
}

/// Read `n` images' detection boxes back from `boxes.bin` (inverse of
/// [`write_detect_boxes`]). `n` comes from `meta.json`.
pub fn read_detect_boxes(path: &Path, n: usize) -> io::Result<Vec<Vec<DetectBox>>> {
    let bytes = fs::read(path)?;
    let mut off = 0usize;
    let rd_u32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let rd_f32 = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let mut out = Vec::with_capacity(n);
    let need = |off: usize, n: usize| -> io::Result<()> {
        if off + n > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: truncated/corrupt boxes.bin — need {} bytes at offset {off}, file has {}", path.display(), n, bytes.len()),
            ));
        }
        Ok(())
    };
    for _ in 0..n {
        need(off, 4)?;
        let num = rd_u32(&bytes, off) as usize;
        off += 4;
        // The counts are file-supplied: check the whole record fits BEFORE
        // allocating `num` capacity or indexing into it — a corrupt count
        // used to drive raw indexing (panic) and an unbounded reserve.
        need(off, num.saturating_mul(20))?;
        let mut v = Vec::with_capacity(num);
        for _ in 0..num {
            let class = rd_u32(&bytes, off);
            let cx = rd_f32(&bytes, off + 4);
            let cy = rd_f32(&bytes, off + 8);
            let w = rd_f32(&bytes, off + 12);
            let h = rd_f32(&bytes, off + 16);
            off += 20;
            v.push(DetectBox { class, cx, cy, w, h });
        }
        out.push(v);
    }
    Ok(out)
}

/// Read the raw `images.f32` blob (`N×3×H×W` CHW f32, normalized `[0,1]`).
/// A thin alias over [`read_f32_bin`] documenting the detection layout.
pub fn read_detect_images(path: &Path) -> io::Result<Vec<f32>> {
    read_f32_bin(path)
}

/// Write a per-token supervision mask as raw `u8` (1 = trainable target, 0 = masked).
pub fn write_mask_bin(path: &Path, mask: &[bool]) -> io::Result<()> {
    let bytes: Vec<u8> = mask.iter().map(|&b| b as u8).collect();
    std::fs::write(path, &bytes)
}

/// Read a `u8` supervision mask written by [`write_mask_bin`].
pub fn read_mask_bin(path: &Path) -> io::Result<Vec<bool>> {
    Ok(std::fs::read(path)?.into_iter().map(|b| b != 0).collect())
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

    #[test]
    fn detect_boxes_roundtrip() {
        let dir = std::env::temp_dir().join(format!("brain_binio_boxes_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let p = dir.join("boxes.bin");
        let boxes = vec![
            vec![
                DetectBox { class: 0, cx: 0.5, cy: 0.25, w: 0.1, h: 0.2 },
                DetectBox { class: 3, cx: 0.75, cy: 0.6, w: 0.3, h: 0.4 },
            ],
            vec![], // an empty (background) image
            vec![DetectBox { class: 1, cx: 0.125, cy: 0.875, w: 0.05, h: 0.05 }],
        ];
        write_detect_boxes(&p, &boxes).unwrap();
        let back = read_detect_boxes(&p, boxes.len()).unwrap();
        assert_eq!(back, boxes);
        let _ = fs::remove_dir_all(&dir);
    }
}
