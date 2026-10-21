// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded serving of quantized models for its clients. If your team needs
// expertise in serving bf16 and int8 checkpoints without expanding them to full precision then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Test fixtures: tiny random models written as brain-native safetensors in the encoding a serving test asks for.

use std::collections::HashMap;
use std::path::Path;

/// How a fixture stores its linears; every other tensor stays f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stored {
    F32,
    Bf16,
    /// brain's int8-native layout: `u32` words of four codes plus the `<name>.scale` grid.
    Int8,
}

/// `v` rounded to the nearest bf16 value, as the f32 a bf16 checkpoint decodes to.
fn bf16_round(v: f32) -> f32 {
    f32::from_bits((model::half::f32_to_bf16(v) as u32) << 16)
}

/// `weights` with each linear of `linears` replaced by what a checkpoint stored as `stored` decodes to: the
/// f32 weights an engine given the whole map would serve to reproduce that checkpoint's function.
pub(crate) fn decoded(weights: &HashMap<String, Vec<f32>>, linears: &HashMap<String, (usize, usize)>, stored: Stored) -> HashMap<String, Vec<f32>> {
    weights
        .iter()
        .map(|(name, w)| {
            let v = match (stored, linears.get(name)) {
                (Stored::Bf16, Some(_)) => w.iter().map(|&x| bf16_round(x)).collect(),
                (Stored::Int8, Some(&(n, k))) => {
                    let (words, scales) = model::int8::quantize_weight(w, n, k);
                    model::int8::dequantize_weight(&words, &scales, n, k)
                }
                _ => w.clone(),
            };
            (name.clone(), v)
        })
        .collect()
}

/// Write `weights` to `path` as a brain safetensors checkpoint carrying `config`: the tensors named in `linears`
/// (name to `[n, k]`) stored as `stored`, all others as flat f32.
pub(crate) fn write(path: &Path, config: &serde_json::Value, weights: &HashMap<String, Vec<f32>>, linears: &HashMap<String, (usize, usize)>, stored: Stored) {
    // (name, dtype, shape, bytes) in file order.
    let mut tensors: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut names: Vec<&String> = weights.keys().collect();
    names.sort();
    for name in names {
        let w = &weights[name];
        match (stored, linears.get(name)) {
            (Stored::Bf16, Some(&(n, k))) => tensors.push((name.clone(), "BF16", vec![n, k], w.iter().flat_map(|&x| model::half::f32_to_bf16(x).to_le_bytes()).collect())),
            (Stored::Int8, Some(&(n, k))) => {
                let (words, scales) = model::int8::quantize_weight(w, n, k);
                tensors.push((name.clone(), "U32", vec![n, k / 4], words.iter().flat_map(|x| x.to_le_bytes()).collect()));
                tensors.push((format!("{name}{}", checkpoint::packed::INT8_SCALE_SUFFIX), "F32", vec![n, k / model::int8::GROUP], scales.iter().flat_map(|x| x.to_le_bytes()).collect()));
            }
            (_, Some(&(n, k))) => tensors.push((name.clone(), "F32", vec![n, k], w.iter().flat_map(|x| x.to_le_bytes()).collect())),
            (_, None) => tensors.push((name.clone(), "F32", vec![w.len()], w.iter().flat_map(|x| x.to_le_bytes()).collect())),
        }
    }
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dtype, shape, bytes) in &tensors {
        header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    header.insert("__metadata__".to_string(), serde_json::json!({ checkpoint::st::CONFIG_KEY: config.to_string() }));
    let mut h = serde_json::to_vec(&header).expect("a header serializes");
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (_, _, _, bytes) in &tensors {
        out.extend_from_slice(bytes);
    }
    std::fs::write(path, out).expect("the fixture checkpoint is written");
}
