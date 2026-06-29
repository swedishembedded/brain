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

/// Build the fp32 ONNX **Talker** decoder for `seq_len` and return
/// `(bytes, config)`. The Qwen3-TTS Talker is byte-for-byte a Qwen3 decoder with
/// an *untied* codec head (`tie_embeddings = false`, so a separate
/// `lm_head.weight`), exported by the same [`build_qwen_fp32_bytes`] path — the
/// `text_projection`/`text_embedding` tensors that ride along in the Talker
/// container are simply unused by the decoder graph. Provided as a named entry
/// point so callers reach for it by intent; the input is `input_ids` (codec
/// token ids) and the output is the codebook-0 `logits`.
pub fn build_talker_fp32_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    build_qwen_fp32_bytes(weights_path, seq_len)
}

/// Export the fp32 ONNX Talker decoder to `out_path` (+ sidecar).
pub fn export_talker_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_qwen_fp32(weights_path, out_path, seq_len)
}

/// Build the fp32 ONNX **Talker hidden-state** graph for `seq_len` and return
/// `(bytes, config)`. Unlike [`build_talker_fp32_bytes`] (token-id → logits),
/// this is the input-embedding-driven graph the autoregressive Talker loop needs:
/// input `inputs_embeds:[1,seq_len,d]` (f32), output `hidden:[1,seq_len,d]` (f32,
/// post-final-norm). The codebook-0 head and MTP residual fill stay on the host.
/// See [`crate::qwen_topology::build_talker_hidden_graph`].
pub fn build_talker_hidden_fp32_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    talker_hidden_bytes(weights_path, seq_len, false)
}

/// As [`build_talker_hidden_fp32_bytes`] but weight-only **INT8** (per-output-
/// channel symmetric, `DequantizeLinear` -> MatMul): ~4x smaller, so the 1.7B
/// Talker fits the NPU and compiles faster.
pub fn build_talker_hidden_int8_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    talker_hidden_bytes(weights_path, seq_len, true)
}

fn talker_hidden_bytes(weights_path: &str, seq_len: usize, quant: bool) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    // Drop the checkpoint as soon as the tensors are extracted to bound peak RAM.
    let (cfg, w) = {
        let c = checkpoint::load(weights_path);
        let cfg = QwenConfig::from_json(&c.header["config"]);
        let w: HashMap<String, Vec<f32>> = c.by_role("");
        (cfg, w)
    };
    let mut g = GraphBuilder::new("qwen_talker_hidden");
    crate::qwen_topology::build_talker_hidden_graph(&cfg, &w, seq_len, quant, &mut g);
    Ok((g.finish(), cfg))
}

/// Export the fp32 ONNX Talker hidden-state graph to `out_path` (+ a
/// `<out_path>.data` sidecar for the large decoder weights).
pub fn export_talker_hidden_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_talker_hidden(weights_path, out_path, seq_len, false)
}

/// Export the weight-only **INT8** Talker hidden-state graph to `out_path`.
pub fn export_talker_hidden_int8(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_talker_hidden(weights_path, out_path, seq_len, true)
}

fn export_talker_hidden(weights_path: &str, out_path: &str, seq_len: usize, quant: bool) -> std::io::Result<()> {
    let (cfg, w) = {
        let c = checkpoint::load(weights_path);
        let cfg = QwenConfig::from_json(&c.header["config"]);
        let w: HashMap<String, Vec<f32>> = c.by_role("");
        (cfg, w)
    };
    let mut g = GraphBuilder::new("qwen_talker_hidden");
    crate::qwen_topology::build_talker_hidden_graph(&cfg, &w, seq_len, quant, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
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
