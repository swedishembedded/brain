// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a brain Qwen `.weights` checkpoint to an ONNX decoder graph (fixed
//! sequence length) for OpenVINO whole-graph compilation. Pure Rust — no NPU
//! needed to produce the file.

use std::collections::HashMap;

use onnx::builder::GraphBuilder;
use qwen::config::QwenConfig;

/// Build the fp32 ONNX decoder for `seq_len` and return `(bytes, config)`.
pub fn build_qwen_fp32_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    let c = checkpoint::load(weights_path);
    let cfg = QwenConfig::from_json(&c.header["config"]);
    let w: HashMap<String, Vec<f32>> = c.by_role("");
    let mut g = GraphBuilder::new("qwen_decoder");
    crate::qwen_topology::build_qwen_graph(&cfg, &w, seq_len, &mut g);
    Ok((g.finish(), cfg))
}

/// Bytes larger than this go to the ONNX external-data sidecar (keeps the proto
/// under protobuf's 2GB parse limit while inlining the small tensors).
const EXTERNAL_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Export the fp32 ONNX decoder to `out_path` (+ a `<out_path>.data` sidecar for
/// large weights). The pair is read back with a file-based OpenVINO loader.
pub fn export_qwen_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    let c = checkpoint::load(weights_path);
    let cfg = QwenConfig::from_json(&c.header["config"]);
    let w: HashMap<String, Vec<f32>> = c.by_role("");
    let mut g = GraphBuilder::new("qwen_decoder");
    crate::qwen_topology::build_qwen_graph(&cfg, &w, seq_len, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}
