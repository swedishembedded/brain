// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures for HuggingFace-layout checkpoints: random weights written as safetensors shards in the formats real
//! releases ship (plain float, FP8 with a per-tensor `weight_scale`, ModelOpt NVFP4), by encoders that share no code
//! with brain's codecs, so a load can be checked against values the fixture decoded independently.
//!
//! Swedish Embedded AB implements serving of quantized HuggingFace releases as published for its clients. If your team
//! needs expertise in ModelOpt and compressed-tensors checkpoints then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::path::Path;

/// One stored tensor.
pub struct Raw {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// A float32 tensor.
pub fn f32_raw(name: &str, shape: &[usize], v: &[f32]) -> Raw {
    Raw { name: name.to_string(), dtype: "F32", shape: shape.to_vec(), bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The value of an OCP E4M3 code (sign, 4-bit exponent with bias 7, 3-bit mantissa).
pub fn e4m3_value(code: u8) -> f32 {
    let (e, m) = ((code >> 3) & 15, code & 7);
    let v = if e == 0 { m as f32 / 512.0 } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7) };
    if code & 128 != 0 { -v } else { v }
}

/// The E4M3 code nearest to `v`.
pub fn e4m3_code(v: f32) -> u8 {
    (0u8..127).chain(128..255).min_by(|a, b| (e4m3_value(*a) - v).abs().total_cmp(&(e4m3_value(*b) - v).abs())).expect("a non-empty code range")
}

fn fp4_code(v: f32) -> u8 {
    let best = (0..8).min_by(|a, b| (E2M1[*a] - v.abs()).abs().total_cmp(&(E2M1[*b] - v.abs()).abs())).expect("eight codes") as u8;
    best | if v < 0.0 { 8 } else { 0 }
}

fn fp4_value(c: u8) -> f32 {
    let v = E2M1[(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

/// FP8 with one scale for the tensor, as `<hf>.weight` + `<hf>.weight_scale` (+ an `input_scale` a loader must
/// ignore); returns the values the stored tensor decodes to.
pub fn fp8_per_tensor(out: &mut Vec<Raw>, hf: &str, shape: &[usize], w: &[f32]) -> Vec<f32> {
    let scale = w.iter().fold(0f32, |m, v| m.max(v.abs())) / 448.0;
    let codes: Vec<u8> = w.iter().map(|v| e4m3_code(v / scale)).collect();
    let decoded = codes.iter().map(|c| e4m3_value(*c) * scale).collect();
    out.push(Raw { name: format!("{hf}.weight"), dtype: "F8_E4M3", shape: shape.to_vec(), bytes: codes });
    out.push(f32_raw(&format!("{hf}.weight_scale"), &[], &[scale]));
    out.push(f32_raw(&format!("{hf}.input_scale"), &[], &[0.0123]));
    decoded
}

/// ModelOpt NVFP4: E2M1 codes two to a byte, an E4M3 scale per 16 and one f32 `weight_scale_2`; returns the decoded
/// values. `shape[1]` is a multiple of 16.
pub fn nvfp4(out: &mut Vec<Raw>, hf: &str, shape: &[usize], w: &[f32]) -> Vec<f32> {
    let global = w.iter().fold(0f32, |m, v| m.max(v.abs())) / (6.0 * 448.0);
    let (mut codes, mut scales, mut decoded) = (Vec::new(), Vec::new(), Vec::new());
    for block in w.chunks(16) {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let sb = e4m3_code(amax / 6.0 / global);
        let eff = e4m3_value(sb) * global;
        scales.push(sb);
        for v in block {
            let c = if eff > 0.0 { fp4_code(v / eff) } else { 0 };
            codes.push(c);
            decoded.push(fp4_value(c) * e4m3_value(sb) * global);
        }
    }
    let packed = codes.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
    out.push(Raw { name: format!("{hf}.weight"), dtype: "U8", shape: vec![shape[0], shape[1] / 2], bytes: packed });
    out.push(Raw { name: format!("{hf}.weight_scale"), dtype: "F8_E4M3", shape: vec![shape[0], shape[1] / 16], bytes: scales });
    out.push(f32_raw(&format!("{hf}.weight_scale_2"), &[], &[global]));
    out.push(f32_raw(&format!("{hf}.input_scale"), &[], &[0.5]));
    decoded
}

/// Write `tensors` as the safetensors file `path`.
pub fn write_safetensors(path: &Path, tensors: &[Raw]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for t in tensors {
        header.insert(t.name.clone(), serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + t.bytes.len()]}));
        off += t.bytes.len();
    }
    let h = serde_json::to_vec(&header).expect("a header serializes");
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Write `tensors` into `dir` as two shards plus `model.safetensors.index.json`, as a real release is laid out.
pub fn write_sharded(dir: &Path, tensors: &[Raw]) {
    let (a, b) = tensors.split_at(tensors.len() / 2);
    let (fa, fb) = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors");
    write_safetensors(&dir.join(fa), a);
    write_safetensors(&dir.join(fb), b);
    let weight_map: serde_json::Map<String, serde_json::Value> = a.iter().map(|r| (r.name.clone(), fa.into())).chain(b.iter().map(|r| (r.name.clone(), fb.into()))).collect();
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::json!({"metadata": {}, "weight_map": weight_map}).to_string()).expect("write the shard index");
}
