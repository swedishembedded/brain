// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a Nemotron FastConformer encoder to a fixed-shape ONNX graph for
//! OpenVINO (Intel NPU / GPU / CPU). Static shapes are an NPU-plugin requirement,
//! so export one graph per (mel length, valid length, prompt) bucket and cache the
//! compiled blob — the LFM / qwen `precompile` pattern.
//!
//! The full 0.6 B-param encoder's single-protobuf ONNX exceeds protobuf's 2 GB
//! limit, so [`export`] writes an external-data sidecar (`<out>.data`) and the NPU
//! path compiles it via `read_model_from_file` (`NpuGraph::compile_path`), while
//! [`build_nemotron_bytes`] keeps the in-memory path for tiny test graphs.

use onnx::builder::GraphBuilder;

use crate::nemotron_topology::{build_nemotron_encoder, NemotronTopo};
use crate::topology::WeightSource;

/// Declare the `mel [1,1,mel_t,num_mel]` input, build the encoder, declare the
/// `pooler [T', decoder_hidden]` output, and return the ONNX bytes in-memory.
/// For tiny test graphs — big checkpoints exceed the single-protobuf limit; use
/// [`export`] + `NpuGraph::compile_path` for the real 0.6 B encoder.
pub fn build_nemotron_bytes(w: &dyn WeightSource, topo: &NemotronTopo, mel_t: u32, mel_valid: u32, prompt_id: u32) -> Vec<u8> {
    let mut g = GraphBuilder::new("nemotron_encoder");
    let t = topo.subsampled_len(mel_t);
    g.input_f32("mel", &[1, 1, mel_t as i64, topo.num_mel_bins as i64]);
    build_nemotron_encoder(&mut g, topo, w, mel_t, mel_valid, prompt_id, "mel", "pooler");
    g.output_f32("pooler", &[t as i64, topo.decoder_hidden as i64]);
    g.finish()
}

/// Export to `out` with an external-data sidecar (`<out>.data`) for the large
/// linears/embeddings — the NPU path compiles this via `NpuGraph::compile_path`.
pub fn export(w: &dyn WeightSource, topo: &NemotronTopo, mel_t: u32, mel_valid: u32, prompt_id: u32, out: &str) -> Result<(), String> {
    let mut g = GraphBuilder::new("nemotron_encoder");
    let t = topo.subsampled_len(mel_t);
    g.input_f32("mel", &[1, 1, mel_t as i64, topo.num_mel_bins as i64]);
    build_nemotron_encoder(&mut g, topo, w, mel_t, mel_valid, prompt_id, "mel", "pooler");
    g.output_f32("pooler", &[t as i64, topo.decoder_hidden as i64]);
    g.finish_external(out, 1 << 20).map_err(|e| format!("write {out}: {e}"))?;
    eprintln!("exported nemotron encoder (mel_t={mel_t}, valid={mel_valid}, T'={t}) -> {out}");
    Ok(())
}
