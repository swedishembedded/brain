// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structural test for the WorldMirror-2 DINOv2 ONNX export: build a
//! 2-block graph from zero weights with the real shapes, decode it, and
//! assert the IO signature and op inventory. (Numerical parity vs the
//! reference goldens is the manual `tools/goldens/mirror_check_onnx.py` step —
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

#[test]
fn trunk_graph_structure() {
    // Tiny trunk: 2 frames, 2x2 patch grid (td = 7 + 4 = 11), 2 levels,
    // taps at both levels.
    let vgt = "visual_geometry_transformer";
    let c = 1024usize;
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    w.insert(format!("{vgt}.cam_token"), vec![0.0; 2 * c]);
    w.insert(format!("{vgt}.reg_token"), vec![0.0; 2 * 4 * c]);
    w.insert(format!("{vgt}.frame_blocks.0.attn.rope.periods"), vec![1.0; 16]);
    for kind in ["frame_blocks", "global_blocks"] {
        for b in 0..2 {
            let p = format!("{vgt}.{kind}.{b}");
            for (n, len) in [
                ("norm1.weight", c),
                ("norm1.bias", c),
                ("attn.qkv.weight", 3 * c * c),
                ("attn.qkv.bias", 3 * c),
                ("attn.q_norm.weight", 64),
                ("attn.q_norm.bias", 64),
                ("attn.k_norm.weight", 64),
                ("attn.k_norm.bias", 64),
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
    }

    let mut g = onnx::builder::GraphBuilder::new("mirror_trunk_tiny");
    npu::mirror_topology::build_trunk_graph(&w, &mut g, 2, 2, 2, 2, &[0, 1]);
    let bytes = g.finish();
    let model = onnx::decode_model(&bytes).expect("decodes");
    let graph = model.graph.expect("graph");
    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.input[0].name, "patch_tokens");
    // taps are emitted as separate frame/global halves — Concat(a, f(a)) whose
    // result is a graph output miscompiles on the Intel NPU (see topology).
    let outs: Vec<&str> = graph.output.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(outs, ["tap0_frame", "tap0_global", "tap1_frame", "tap1_global"]);
    assert_eq!(graph.node.iter().filter(|n| n.op_type == "Concat" && n.output.iter().any(|o| o.starts_with("tap"))).count(), 0);
    let count = |op: &str| graph.node.iter().filter(|n| n.op_type == op).count();
    // 2 levels x (frame + global) = 4 attention blocks
    assert_eq!(count("Softmax"), 4);
    assert_eq!(count("Erf"), 4);
    // per block: qkv+proj+fc1+fc2 + scores+ctx = 6 MatMul
    assert_eq!(count("MatMul"), 24);
    // LN: per block norm1+norm2 (2 ReduceMean each) + q/k head-norms (2 each)
    assert_eq!(count("ReduceMean"), 4 * (2 * 2 + 2 * 2));
    // rope: q,k each slice hi/lo halves per block = 4 Slices per block
    assert_eq!(count("Slice"), 16);
}

#[test]
fn dpt_head_graph_structure() {
    // tiny config (same shape family as mirror's t8 test), gs head with the
    // rgb merge branch
    let cfg = worldmirror2::config::MirrorConfig {
        depth: 4,
        dim: 64,
        heads: 2,
        mlp_ratio: 2,
        patch: 14,
        img: 56,
        reg_tokens: 4,
        tap_levels: [0, 1, 2, 3],
        dpt_proj: [16, 32, 64, 64],
        dpt_feat: 16,
        cam_blocks: 2,
        cam_params: 9,
    };
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, shape) in cfg.param_list() {
        if name.starts_with("gs_head.") || name.starts_with("gs_renderer.") {
            w.insert(name, vec![0.0; shape.iter().product()]);
        }
    }
    let mut g = onnx::builder::GraphBuilder::new("mirror_gs_head_tiny");
    npu::mirror_topology::build_dpt_head_graph(&w, &mut g, &cfg, "gs_head", 3, 4, 4, true);
    let bytes = g.finish();
    let model = onnx::decode_model(&bytes).expect("decodes");
    let graph = model.graph.expect("graph");
    let ins: Vec<&str> = graph.input.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(ins, ["tap0", "tap1", "tap2", "tap3", "rgb"]);
    let outs: Vec<&str> = graph.output.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(outs, ["head_out", "gs_params"]);
    let count = |op: &str| graph.node.iter().filter(|n| n.op_type == op).count();
    assert_eq!(count("ConvTranspose"), 2); // resize_layers 0 and 1
    assert_eq!(count("Resize"), 5); // 4 fusion upsamples + full-res
    // 7 RCUs x 2 convs + 4 projects + 2 resize convs(no: 1 conv k3s2)
    // + 4 layer_rn + 4 out_conv + output_conv1 + output_conv2 (2)
    // + gs: input_merger + gs_head.0 + gs_head.2
    assert_eq!(count("Conv"), 14 + 4 + 1 + 4 + 4 + 3 + 3);
    // LN per tap = 4, each 2 ReduceMean
    assert_eq!(count("ReduceMean"), 8);
}
