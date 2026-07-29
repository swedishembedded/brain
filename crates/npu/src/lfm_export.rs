// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a brain LFM2.5-Encoder checkpoint to a fixed-shape ONNX graph for
//! OpenVINO (Intel NPU / GPU / CPU). Static shapes are an NPU-plugin
//! requirement — export one graph per sequence-length bucket and cache the
//! compiled blobs (the qwen `precompile` pattern).

use lfm::config::LfmConfig;
use onnx::builder::GraphBuilder;

use crate::qwen_topology::Quant;

/// Export `<weights>` at sequence length `s` to `out` (external-data sidecar
/// `<out>.data` for the 256 MB embedding + linears). `int8` selects
/// per-output-channel weight-only INT8.
pub fn export(weights: &str, s: usize, out: &str, int8: bool) -> Result<(), String> {
    let c = checkpoint::load(weights);
    let cfg = LfmConfig::from_json(&c.header["config"]);
    let w = c.by_role("");
    let mut g = GraphBuilder::new("lfm25_encoder");
    crate::lfm_topology::build_lfm_graph_quant(&cfg, &w, s, &mut g, Quant::from_bool(int8));
    g.finish_external(out, 1 << 20).map_err(|e| format!("write {out}: {e}"))?;
    eprintln!("exported {weights} (S={s}, {}) -> {out}", if int8 { "int8-weights" } else { "fp32" });
    Ok(())
}

/// In-memory ONNX bytes (tests / small graphs; big checkpoints exceed the
/// buffer path — use [`export`] + `LfmSession::load_path` instead).
pub fn build_lfm_bytes(weights: &str, s: usize, int8: bool) -> Result<(Vec<u8>, LfmConfig), String> {
    let c = checkpoint::load(weights);
    let cfg = LfmConfig::from_json(&c.header["config"]);
    let w = c.by_role("");
    let mut g = GraphBuilder::new("lfm25_encoder");
    crate::lfm_topology::build_lfm_graph_quant(&cfg, &w, s, &mut g, Quant::from_bool(int8));
    Ok((g.finish(), cfg))
}
