// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a brain codec `.weights` checkpoint to an ONNX decoder graph (fixed
//! code length) for OpenVINO whole-graph compilation. Pure Rust — no NPU needed
//! to produce the file. Mirrors [`crate::qwen_export`] for the conv-heavy codec.

use std::collections::HashMap;

use codec::CodecConfig;
use onnx::builder::GraphBuilder;

/// Build the fp32 ONNX codec decoder for `code_len` frames and return
/// `(bytes, config)`. Input `codes:[num_quantizers, code_len]` (int64,
/// codebook-major), output `waveform:[1,1,L]` (f32).
pub fn build_codec_fp32_bytes(weights_path: &str, code_len: usize) -> std::io::Result<(Vec<u8>, CodecConfig)> {
    let c = checkpoint::load(weights_path);
    let cfg = CodecConfig::from_json(&c.header["config"]);
    let w: HashMap<String, Vec<f32>> = c.by_role("");
    let mut g = GraphBuilder::new("qwen3tts_codec_decoder");
    crate::codec_topology::build_codec_graph(&cfg, &w, code_len, &mut g);
    Ok((g.finish(), cfg))
}

/// Bytes larger than this go to the ONNX external-data sidecar.
const EXTERNAL_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Export the fp32 ONNX codec decoder to `out_path` (+ a `<out_path>.data`
/// sidecar for the large conv/codebook weights).
pub fn export_codec_fp32(weights_path: &str, out_path: &str, code_len: usize) -> std::io::Result<()> {
    let c = checkpoint::load(weights_path);
    let cfg = CodecConfig::from_json(&c.header["config"]);
    let w: HashMap<String, Vec<f32>> = c.by_role("");
    let mut g = GraphBuilder::new("qwen3tts_codec_decoder");
    crate::codec_topology::build_codec_graph(&cfg, &w, code_len, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}
