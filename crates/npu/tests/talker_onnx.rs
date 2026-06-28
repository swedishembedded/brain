// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structural validation of the Qwen3-TTS **Talker** ONNX export.
//!
//! The Talker is a Qwen3 decoder with an *untied* head (`tie_embeddings =
//! false`), so it reuses `qwen_export`/`qwen_topology` but must emit the separate
//! `lm_head.weight` (not the tied `tok.weight`). OpenVINO is absent here, so we
//! validate the graph structurally: it builds, the proto decodes, the head reads
//! the untied weight, and IO is `input_ids -> logits`.

use std::collections::HashMap;

use qwen::{Qwen, QwenConfig};

/// A tiny *untied* Qwen decoder (the Talker's backbone shape).
fn untied_tiny() -> QwenConfig {
    let mut c = QwenConfig::tiny();
    c.tie_embeddings = false;
    c
}

#[test]
fn talker_onnx_graph_is_well_formed_untied_head() {
    let cfg = untied_tiny();
    let t = 4u32;
    let init: HashMap<String, Vec<f32>> = qwen::init_weights(&cfg, 9);
    let model = Qwen::new(cfg.clone(), 1, t, &init);

    let dir = std::env::temp_dir().join(format!("brain_talker_onnx_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wpath = dir.join("talker_tiny.weights");
    model.save(wpath.to_str().unwrap());

    let (bytes, ecfg) = npu::qwen_export::build_talker_fp32_bytes(wpath.to_str().unwrap(), t as usize)
        .expect("build talker onnx");
    std::fs::remove_dir_all(&dir).ok();
    assert!(!ecfg.tie_embeddings, "exported Talker config is untied");

    let model = onnx::decode_model(&bytes).expect("talker ONNX must decode");
    let graph = model.graph.expect("graph");

    // IO.
    assert!(graph.input.iter().any(|i| i.name == "input_ids"));
    assert!(graph.output.iter().any(|o| o.name == "logits"));

    // The untied head transposes `lm_head.weight` (not `tok.weight`) into the
    // `lm_head.w` initializer; assert that initializer exists and is sized
    // `[d, vocab]`.
    let d = cfg.d_model as i64;
    let v = cfg.vocab as i64;
    let head = graph
        .initializer
        .iter()
        .find(|t| t.name == "lm_head.w")
        .expect("untied head initializer lm_head.w");
    assert_eq!(head.dims, vec![d, v], "lm_head.w must be [d, vocab]");

    // Sanity: standard decoder ops are present.
    let mut ops: HashMap<&str, usize> = HashMap::new();
    for n in &graph.node {
        *ops.entry(n.op_type.as_str()).or_default() += 1;
    }
    assert!(ops.get("MatMul").copied().unwrap_or(0) > 0);
    assert!(ops.get("Softmax").copied().unwrap_or(0) >= cfg.n_layers as usize);
}
