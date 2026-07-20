// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structural test for the WorldMirror-2 DINOv2 ONNX export: build a
//! 2-block graph from zero weights with the real shapes, decode it, and
//! assert the IO signature and op inventory. (Numerical parity vs the
//! reference goldens is the manual `tools/mirror_check_onnx.py` step —
//! OpenVINO CPU/NPU — since the full 24-block graph carries 1.2 GB of
//! weights.)

use std::collections::HashMap;

#[test]
fn dinov2_graph_structure() {
    let pe = "visual_geometry_transformer.patch_embed";
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let c = 1024usize;
    w.insert(format!("{pe}.patch_embed.proj.weight"), vec![0.0; c * 3 * 14 * 14]);
    w.insert(format!("{pe}.patch_embed.proj.bias"), vec![0.0; c]);
    w.insert(format!("{pe}.cls_token"), vec![0.0; c]);
    w.insert(format!("{pe}.register_tokens"), vec![0.0; 4 * c]);
    w.insert(format!("{pe}.pos_embed"), vec![0.0; 1370 * c]);
    w.insert(format!("{pe}.norm.weight"), vec![0.0; c]);
    w.insert(format!("{pe}.norm.bias"), vec![0.0; c]);
    for b in 0..2 {
        let p = format!("{pe}.blocks.{b}");
        for (n, len) in [
            ("norm1.weight", c),
            ("norm1.bias", c),
            ("attn.qkv.weight", 3 * c * c),
            ("attn.qkv.bias", 3 * c),
            ("attn.proj.weight", c * c),
            ("attn.proj.bias", c),
            ("ls1.gamma", c),
            ("norm2.weight", c),
            ("norm2.bias", c),
            ("mlp.fc1.weight", 4 * c * c),
            ("mlp.fc1.bias", 4 * c),
            ("mlp.fc2.weight", c * 4 * c),
            ("mlp.fc2.bias", c),
            ("ls2.gamma", c),
        ] {
            w.insert(format!("{p}.{n}"), vec![0.0; len]);
        }
    }

    let mut g = onnx::builder::GraphBuilder::new("mirror_dino_tiny");
    npu::mirror_topology::build_dinov2_graph(&w, &mut g, 2);
    let bytes = g.finish();
    let model = onnx::decode_model(&bytes).expect("decodes");
    let graph = model.graph.expect("graph");
    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.input[0].name, "frame");
    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.output[0].name, "patch_tokens");
    let count = |op: &str| graph.node.iter().filter(|n| n.op_type == op).count();
    assert_eq!(count("Conv"), 1);
    assert_eq!(count("Softmax"), 2); // one per block
    assert_eq!(count("Erf"), 2);
    // per block: qkv+proj+fc1+fc2 (4) + scores+ctx (2) = 6; total 12
    assert_eq!(count("MatMul"), 12);
    assert_eq!(count("Slice"), 1);
    // LN per block ×2 + final = 5, each has 2 ReduceMean
    assert_eq!(count("ReduceMean"), 10);
}
