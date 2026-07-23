// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a brain GLM `.weights` checkpoint to an ONNX decoder graph (fixed
//! sequence length) for OpenVINO whole-graph compilation. Pure Rust — no NPU
//! needed to produce the file. See `docs/models/glm/npu.md`.

use std::collections::HashMap;

use glm::config::GlmConfig;
use onnx::builder::GraphBuilder;

/// Build the fp32 ONNX GLM decoder for `seq_len` and return `(bytes, config)`.
pub fn build_glm_fp32_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, GlmConfig)> {
    let c = checkpoint::load(weights_path);
    let cfg = GlmConfig::from_json(&c.header["config"]);
    let w: HashMap<String, Vec<f32>> = c.by_role("");
    let mut g = GraphBuilder::new("glm_decoder");
    crate::glm_topology::build_glm_graph(&cfg, &w, seq_len, &mut g);
    Ok((g.finish(), cfg))
}

/// Export the fp32 ONNX GLM decoder to `out_path`.
pub fn export_glm_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    let (bytes, _cfg) = build_glm_fp32_bytes(weights_path, seq_len)?;
    std::fs::write(out_path, bytes)
}

/// Build the INT8 (per-output-channel weight-only) ONNX GLM decoder — ~4x smaller
/// than fp32; norms/RoPE/router stay fp32.
pub fn build_glm_int8_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, GlmConfig)> {
    let c = checkpoint::load(weights_path);
    let cfg = GlmConfig::from_json(&c.header["config"]);
    let w: HashMap<String, Vec<f32>> = c.by_role("");
    let mut g = GraphBuilder::new("glm_decoder_int8");
    crate::glm_topology::build_glm_graph_quant(&cfg, &w, seq_len, &mut g, crate::qwen_topology::Quant::Int8);
    Ok((g.finish(), cfg))
}

/// Export the INT8 ONNX GLM decoder to `out_path`.
pub fn export_glm_int8(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    let (bytes, _cfg) = build_glm_int8_bytes(weights_path, seq_len)?;
    std::fs::write(out_path, bytes)
}
