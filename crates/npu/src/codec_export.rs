// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a brain codec `.safetensors` checkpoint to an ONNX decoder graph (fixed
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

/// Export the **front-only** codec graph (`codes:[nq,T]` -> `latent:[1,latent,T]`).
pub fn export_codec_front_fp32(weights_path: &str, out_path: &str, t: usize) -> std::io::Result<CodecConfig> {
    let c = checkpoint::load(weights_path);
    let cfg = CodecConfig::from_json(&c.header["config"]);
    let w: HashMap<String, Vec<f32>> = c.by_role("");
    let mut g = GraphBuilder::new("qwen3tts_codec_front");
    crate::codec_topology::build_codec_front_graph(&cfg, &w, t, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)?;
    Ok(cfg)
}

/// Export the **streaming-back** codec graph (`latent:[1,latent,chunk]` + per-conv
/// `bufin.*` -> `waveform` + per-conv `bufout.*`). Returns the buffer specs
/// `(prefix, channels, width)` the host needs to allocate + carry across chunks.
pub fn export_codec_back_stream_fp32(
    weights_path: &str,
    out_path: &str,
    chunk: usize,
) -> std::io::Result<(CodecConfig, Vec<(String, i64, i64)>)> {
    let c = checkpoint::load(weights_path);
    let cfg = CodecConfig::from_json(&c.header["config"]);
    let w: HashMap<String, Vec<f32>> = c.by_role("");
    let mut g = GraphBuilder::new("qwen3tts_codec_back_stream");
    let bufs = crate::codec_topology::build_codec_back_stream_graph(&cfg, &w, chunk, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)?;
    Ok((cfg, bufs))
}

#[cfg(test)]
mod tests {
    /// Structural check: the front + streaming-back graphs build and emit valid
    /// ONNX, and the back exposes the expected per-conv state buffers. Run:
    ///   BRAIN_CODEC_WEIGHTS=.../codec.safetensors \
    ///   cargo test -p brain-npu export_streaming_graphs -- --ignored --nocapture
    #[test]
    #[ignore]
    fn export_streaming_graphs() {
        let path = std::env::var("BRAIN_CODEC_WEIGHTS").expect("set BRAIN_CODEC_WEIGHTS");
        let dir = std::env::temp_dir();
        let front = dir.join("codec_front.onnx");
        let back = dir.join("codec_back_stream.onnx");
        let cfg = super::export_codec_front_fp32(&path, front.to_str().unwrap(), 32).unwrap();
        let (_, bufs) = super::export_codec_back_stream_fp32(&path, back.to_str().unwrap(), 16).unwrap();
        eprintln!("front+back exported (latent={}); {} streaming buffers:", cfg.latent_dim, bufs.len());
        for (p, c, w) in &bufs {
            eprintln!("  {p}: [1,{c},{w}]");
        }
        assert!(front.exists() && back.exists(), "ONNX files not written");
        assert!(bufs.len() >= 15, "expected ~21 conv state buffers, got {}", bufs.len());
    }
}
