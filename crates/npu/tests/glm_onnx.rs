// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM ONNX export tests.
//!
//! Structural (always runs, no backend/OpenVINO): build a tiny GLM checkpoint,
//! export the ONNX decoder, and assert it decodes to a well-formed graph with
//! the right inputs/outputs and a real node count — this validates the graph
//! wiring/shapes (the MLA + dense-expert MoE builders) without hardware.
//!
//! Numerical parity vs brain's own forward requires OpenVINO + NPU hardware and
//! is gated separately (see docs/glm/NPU.md); it is not asserted here.

use std::collections::HashMap;

use glm::config::GlmConfig;

/// Write a checkpoint directly from freshly-initialised weights (no GPU/CPU
/// backend needed — `init_weights`, `checkpoint::save`, and the ONNX export are
/// all pure Rust).
fn write_tiny_ckpt(dir: &std::path::Path, cfg: &GlmConfig) -> std::path::PathBuf {
    let init: HashMap<String, Vec<f32>> = glm::init_weights(cfg, 3);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(n, numel)| (n.clone(), vec![numel as u64], init[&n].clone()))
        .collect();
    let path = dir.join("glm.weights");
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
    path
}

#[test]
fn glm_onnx_graph_is_well_formed() {
    let dir = std::env::temp_dir().join(format!("brain-glm-onnx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // tiny(): layer 0 dense, layer 1 MoE -> exercises both the dense and the
    // dense-expert MoE (TopK/GatherElements/ScatterElements) graph paths.
    let cfg = GlmConfig::tiny();
    let path = write_tiny_ckpt(&dir, &cfg);

    let seq = 8usize;
    let (bytes, _cfg) = npu::glm_export::build_glm_fp32_bytes(path.to_str().unwrap(), seq).unwrap();
    assert!(bytes.len() > 1000, "onnx export suspiciously small: {} bytes", bytes.len());

    let model = onnx::decode_model(&bytes).expect("export must decode as a valid ONNX ModelProto");
    let g = model.graph.expect("model has a graph");
    assert!(g.node.len() > 30, "expected a real multi-layer graph, got {} nodes", g.node.len());
    assert!(g.input.iter().any(|v| v.name == "input_ids"), "missing input_ids");
    assert!(g.output.iter().any(|v| v.name == "logits"), "missing logits output");
    // MoE dense-expert path emits TopK / GatherElements / ScatterElements.
    let ops: Vec<&str> = g.node.iter().map(|n| n.op_type.as_str()).collect();
    for op in ["MatMul", "Softmax", "Sigmoid", "TopK", "ScatterElements", "GatherElements"] {
        assert!(ops.contains(&op), "expected a {op} node in the GLM MoE graph");
    }

    std::fs::remove_dir_all(&dir).ok();
}
