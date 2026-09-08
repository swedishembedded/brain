// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR-2 ONNX export tests: the decoder (`deepseek2`) and the new
//! resampler tower (`deepseekocr2`).
//!
//! Structural only (always runs, no OpenVINO/NPU): build a TINY resampler
//! and a TINY decoder from freshly-seeded weights and assert each exported
//! graph is well-formed, mirroring `qwen35moe_onnx.rs`'s own structural-test
//! convention. This host has no NPU firmware, so a real compile/run attempt
//! is out of reach here (recorded in the ledger, per this milestone's
//! explicit "unvalidated" scope) - this test proves the GRAPH BUILDS
//! correctly, not that it runs on an accelerator.
//!
//! SAM is intentionally absent from these graphs - see
//! `npu::deepseekocr2_topology`'s module doc for why.

use std::collections::HashMap;

use deepseek2::config::DeepseekV2Config;
use deepseekocr2::config::Qwen2EncoderConfig;

/// A small, seeded pseudo-random fill - deterministic, no external RNG
/// dependency, good enough for a structural (not numerical) export check.
fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
        })
        .collect()
}

fn fill(names: &[(String, usize)], seed: u64) -> HashMap<String, Vec<f32>> {
    names.iter().enumerate().map(|(i, (n, sz))| (n.clone(), seeded(*sz, seed + i as u64))).collect()
}

/// The same tiny dims this whole campaign has shared since M2's golden
/// (`testdata/deepseekocr2/tiny/manifest-tiny.json`): hidden 24, 6 heads / 2
/// KV heads, head_dim 4, ff 17, n_query_local 5, n_query_global 8 - kept
/// mutually distinct so a transposed axis or a swapped shape cannot pass by
/// coincidence.
fn tiny_encoder_cfg() -> Qwen2EncoderConfig {
    Qwen2EncoderConfig {
        d_model: 24,
        n_layers: 2,
        n_heads: 6,
        n_kv_heads: 2,
        ffn_hidden: 17,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
        n_query_local: 5,
        n_query_global: 8,
    }
}

#[test]
fn resampler_onnx_graph_is_well_formed_for_both_views() {
    let ecfg = tiny_encoder_cfg();
    let decoder_hidden = 15usize;
    let mut w = fill(&ecfg.param_list(), 0xC0FFEE);
    // The projector is a property of the whole tower
    // (`DeepseekOcr2VisionConfig`), not the encoder stack
    // (`Qwen2EncoderConfig::param_list` deliberately excludes it) - added
    // directly here rather than pulling in a full vision config just for
    // its `sam` field, which this graph never reads.
    w.insert("vision.projector.fc.weight".to_string(), seeded(decoder_hidden * ecfg.d_model as usize, 0xFC00));
    w.insert("vision.projector.fc.bias".to_string(), seeded(decoder_hidden, 0xFC01));

    for (local, n_query, tag) in [(true, ecfg.n_query_local as usize, "local"), (false, ecfg.n_query_global as usize, "global")] {
        let mut g = onnx::builder::GraphBuilder::new(&format!("resampler_{tag}"));
        npu::deepseekocr2_topology::build_resampler_graph(&ecfg, &w, n_query, local, decoder_hidden, &mut g);
        let bytes = g.finish();
        assert!(bytes.len() > 500, "{tag}: onnx export suspiciously small: {} bytes", bytes.len());

        let model = onnx::decode_model(&bytes).unwrap_or_else(|e| panic!("{tag}: export must decode as a valid ONNX ModelProto: {e}"));
        let graph = model.graph.expect("model has a graph");
        assert!(graph.node.len() > 20, "{tag}: expected a real multi-layer graph, got {} nodes", graph.node.len());
        assert!(graph.input.iter().any(|v| v.name == "sam_tokens"), "{tag}: missing sam_tokens input");
        assert!(graph.output.iter().any(|v| v.name == "projected"), "{tag}: missing projected output");
        // A prefix-LM mask that never appears is a graph that silently fell
        // back to plain causal or plain bidirectional attention.
        assert!(graph.initializer.iter().any(|i| i.name == "prefix_mask"), "{tag}: prefix_mask initializer missing");
    }
}

#[test]
fn decoder_onnx_graph_is_well_formed_dense_and_moe() {
    let cfg = DeepseekV2Config::tiny();
    let w = fill(&cfg.param_list(), 0xDECADE);
    const SEQ: usize = 6;

    let mut g = onnx::builder::GraphBuilder::new("deepseek2_decoder");
    npu::deepseek2_topology::build_deepseek2_graph(&cfg, &w, SEQ, &mut g);
    let bytes = g.finish();
    assert!(bytes.len() > 500, "onnx export suspiciously small: {} bytes", bytes.len());

    let model = onnx::decode_model(&bytes).expect("export must decode as a valid ONNX ModelProto");
    let graph = model.graph.expect("model has a graph");
    assert!(graph.node.len() > 20, "expected a real multi-layer graph, got {} nodes", graph.node.len());
    assert!(graph.input.iter().any(|v| v.name == "inputs_embeds"), "missing inputs_embeds input");
    assert!(graph.output.iter().any(|v| v.name == "logits"), "missing logits output");
    // `tiny()` has a dense layer 0 AND an MoE layer 1 - both op shapes must
    // actually appear, not just one of the two dispatched paths.
    assert!(graph.node.iter().any(|n| n.op_type == "TopK"), "no TopK found - the MoE layer's router never ran");
}
