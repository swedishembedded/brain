// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a brain `qwen35moe` (Qwen3.5-35B-A3B) `.safetensors` checkpoint to
//! an ONNX decoder graph (fixed sequence length) for a best-effort
//! OpenVINO/NPU compile attempt. Pure Rust — no NPU needed to produce the
//! file. See `crate::qwen35moe_topology`'s module doc for the two hard
//! sub-problems this export solves (the GDN chunk emitter and the sparse
//! MoE dispatch) and exactly where this pass stops.

use qwen35moe::config::Qwen35Config;
use onnx::builder::GraphBuilder;

/// Build the fp32 ONNX Qwen3.5 decoder for `seq_len` and return `(bytes, config)`.
pub fn build_qwen35_fp32_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, Qwen35Config)> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path)?;
    let cfg = Qwen35Config::from_json(&reader.config());
    let mut g = GraphBuilder::new("qwen35_decoder");
    crate::qwen35moe_topology::build_qwen35_graph(&cfg, &reader, seq_len, &mut g);
    Ok((g.finish(), cfg))
}

/// Export the fp32 ONNX Qwen3.5 decoder to `out_path`.
pub fn export_qwen35_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    let (bytes, _cfg) = build_qwen35_fp32_bytes(weights_path, seq_len)?;
    std::fs::write(out_path, bytes)
}

/// Export via ONNX **external data** (`GraphBuilder::finish_external`, the
/// same mechanism `lfm_export::export`/`qwen_export`/`nemotron_export` use
/// for their own largest initializers) — keeps the `.onnx` proto itself under
/// protobuf's 2GB limit by writing every initializer above `threshold` bytes
/// to a `<out_path>.data` sidecar instead of inlining it.
///
/// **Known real-scale limitation, not solved by this function**: at the real
/// `Qwen35Config::qwen35_35b_a3b()` shape, [`crate::qwen35moe_topology::Topo::expert_stack`]
/// builds each layer's `[256,d,ff]` stacked expert-weight initializer as one
/// host `Vec<f32>` (~1 GB per stack, 3 stacks/layer, 40 layers) and every
/// stack stays resident in the in-memory [`onnx::graph::Graph`] until this
/// function's single `finish_external` call serializes ALL of them at once —
/// external data solves the *protobuf 2GB proto-size* limit, not the
/// *transient host RAM* one (~120 GB, the same order of magnitude as
/// `docs/models/qwen35/status.md`'s already-documented "~140GB, does not fit
/// on this hardware" full-precision-import finding). A real fix needs the
/// graph builder itself to stream each initializer straight to the sidecar
/// file as it's produced rather than accumulating them all in `Graph::
/// initializers` first — real, larger graph-builder surgery, explicitly out
/// of scope for this best-effort pass (verified correct and exercised only at
/// `Qwen35Config::tiny()`'s scale, see `crates/npu/tests/qwen35moe_onnx.rs`).
pub fn export_qwen35_fp32_external(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path)?;
    let cfg = Qwen35Config::from_json(&reader.config());
    let mut g = GraphBuilder::new("qwen35_decoder");
    crate::qwen35moe_topology::build_qwen35_graph(&cfg, &reader, seq_len, &mut g);
    g.finish_external(out_path, 1 << 20)
}
