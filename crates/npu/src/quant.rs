// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 quantization parameters.
//!
//! Activations are quantized **per-tensor, symmetric** (zero-point 0); the scale
//! for each quantized conv is keyed by the conv's prefix (which is both the
//! `yolov8::net::ActTap` name and the exported ONNX node prefix). Weights are
//! quantized per-output-channel directly from the folded weights in
//! [`crate::fold::quantize_weight_per_channel`] (data-independent), so they are
//! NOT stored here.

use std::collections::HashMap;

/// Calibrated INT8 quantization parameters: one activation scale per quantized
/// conv (symmetric, zero-point 0).
#[derive(Clone, Debug, Default)]
pub struct Quant {
    pub act_scales: HashMap<String, f32>,
}

impl Quant {
    pub fn new() -> Quant {
        Quant { act_scales: HashMap::new() }
    }

    /// The activation scale for conv `prefix`, if calibrated.
    pub fn act_scale(&self, prefix: &str) -> Option<f32> {
        self.act_scales.get(prefix).copied()
    }

    /// Number of calibrated conv activations.
    pub fn len(&self) -> usize {
        self.act_scales.len()
    }
    pub fn is_empty(&self) -> bool {
        self.act_scales.is_empty()
    }

    /// Serialize to a stable, inspectable JSON object `{ "act_scales": {…} }`.
    pub fn to_json(&self) -> serde_json::Value {
        let mut keys: Vec<&String> = self.act_scales.keys().collect();
        keys.sort();
        let map: serde_json::Map<String, serde_json::Value> = keys
            .into_iter()
            .map(|k| (k.clone(), serde_json::json!(self.act_scales[k])))
            .collect();
        serde_json::json!({ "act_scales": map })
    }

    /// Parse from the JSON produced by [`Quant::to_json`].
    pub fn from_json(v: &serde_json::Value) -> Quant {
        let mut act_scales = HashMap::new();
        if let Some(obj) = v.get("act_scales").and_then(|m| m.as_object()) {
            for (k, val) in obj {
                if let Some(f) = val.as_f64() {
                    act_scales.insert(k.clone(), f as f32);
                }
            }
        }
        Quant { act_scales }
    }
}

/// Symmetric INT8 scale for a per-tensor activation range `[min,max]`:
/// `scale = max(|min|,|max|) / 127`, zero-point 0.
pub fn symmetric_act_scale(min: f32, max: f32) -> f32 {
    (min.abs().max(max.abs()) / 127.0).max(1e-12)
}
