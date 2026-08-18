// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity gate for the Nemotron FastConformer encoder **head** (stages 2–3: the
//! macaron Conformer blocks + prompt/encoder projectors) as an OpenVINO ONNX graph.
//! Self-contained: a tiny random-weight encoder is run through both
//! `nemotronasr::reference::encode_pooler` (the host f32 oracle) and the ONNX graph
//! compiled on the OpenVINO **CPU** device, and the two poolers must agree on the
//! valid rows (padding rows are allowed to differ — a fully-masked padding query's
//! softmax is uniform in ONNX vs zeroed in the reference, but causal/ masked ops
//! keep VALID outputs independent of padding).
//!
//! Skips cleanly without an OpenVINO runtime. Run:
//!   LD_LIBRARY_PATH=<openvino/libs> cargo test -p brain-npu --test nemotron_encoder -- --nocapture

use std::collections::HashMap;

use nemotronasr::config::NemotronConfig;
use nemotronasr::reference::encode_pooler;
use npu::openvino::{available_devices, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};
use npu::{build_nemotron_head, NemotronTopo};

/// Deterministic small pseudo-random fill.
fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    // The unified deterministic LCG (audit F39/F40) — one premix keeps
    // distinct seeds decorrelated, as the old local copy did.
    let mut l = data::rng::Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
    (0..n).map(|_| l.scaled(scale)).collect()
}

/// A tiny but structurally-faithful config + matching topology.
fn tiny() -> (NemotronConfig, NemotronTopo, u32, u32, u32) {
    let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
    cfg.hidden = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 4; // head_dim 16
    cfg.intermediate = 128;
    cfg.conv_kernel = 9;
    cfg.sliding_window = 5; // left ctx 4
    cfg.default_lookahead = 1; // right 1
    cfg.decoder_hidden = 24;
    cfg.num_prompts = 8;
    cfg.prompt_intermediate = 32;
    let topo = NemotronTopo {
        num_mel_bins: cfg.num_mel_bins,
        hidden: cfg.hidden,
        subsampling_channels: cfg.subsampling_channels,
        subsampling_kernel: cfg.subsampling_kernel,
        subsampling_stride: cfg.subsampling_stride,
        subsampling_stages: cfg.subsampling_stages(),
        n_layers: cfg.n_layers,
        n_heads: cfg.n_heads,
        head_dim: cfg.head_dim(),
        intermediate: cfg.intermediate,
        conv_kernel: cfg.conv_kernel,
        left_ctx: cfg.sliding_window - 1,
        right_ctx: cfg.default_lookahead,
        ln_eps: cfg.ln_eps,
        num_prompts: cfg.num_prompts,
        prompt_intermediate: cfg.prompt_intermediate,
        decoder_hidden: cfg.decoder_hidden,
    };
    let (t, valid, prompt_id) = (8u32, 6u32, 1u32);
    (cfg, topo, t, valid, prompt_id)
}

/// Random weights for every name `encode_pooler` (blocks + projectors) reads, at
/// the sizes the reference and the ONNX builder both expect. Norm gammas ≈ 1 so
/// activations stay O(1) through the stack; everything else is small.
fn weights(cfg: &NemotronConfig) -> HashMap<String, Vec<f32>> {
    let (c, ffn, np, pi, dh, k) = (
        cfg.hidden as usize,
        cfg.intermediate as usize,
        cfg.num_prompts as usize,
        cfg.prompt_intermediate as usize,
        cfg.decoder_hidden as usize,
        cfg.conv_kernel as usize,
    );
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut seed = 1u64;
    let mut put = |w: &mut HashMap<String, Vec<f32>>, name: String, n: usize, scale: f32| {
        seed += 1;
        w.insert(name, fill(seed, n, scale));
    };
    let gamma = |w: &mut HashMap<String, Vec<f32>>, name: String| {
        w.insert(name, vec![1.0f32; c]); // layernorm weight ≈ 1
    };
    for b in 0..cfg.n_layers {
        let p = format!("encoder.layers.{b}");
        gamma(&mut w, format!("{p}.norm_feed_forward1.weight"));
        put(&mut w, format!("{p}.norm_feed_forward1.bias"), c, 0.02);
        put(&mut w, format!("{p}.feed_forward1.linear1.weight"), ffn * c, 0.05);
        put(&mut w, format!("{p}.feed_forward1.linear2.weight"), c * ffn, 0.05);
        gamma(&mut w, format!("{p}.norm_self_att.weight"));
        put(&mut w, format!("{p}.norm_self_att.bias"), c, 0.02);
        for pr in ["q_proj", "k_proj", "v_proj", "relative_k_proj", "o_proj"] {
            put(&mut w, format!("{p}.self_attn.{pr}.weight"), c * c, 0.05);
        }
        put(&mut w, format!("{p}.self_attn.bias_u"), c, 0.05);
        put(&mut w, format!("{p}.self_attn.bias_v"), c, 0.05);
        gamma(&mut w, format!("{p}.norm_conv.weight"));
        put(&mut w, format!("{p}.norm_conv.bias"), c, 0.02);
        put(&mut w, format!("{p}.conv.pointwise_conv1.weight"), 2 * c * c, 0.05);
        put(&mut w, format!("{p}.conv.depthwise_conv.weight"), c * k, 0.05);
        gamma(&mut w, format!("{p}.conv.norm.weight"));
        put(&mut w, format!("{p}.conv.norm.bias"), c, 0.02);
        put(&mut w, format!("{p}.conv.pointwise_conv2.weight"), c * c, 0.05);
        gamma(&mut w, format!("{p}.norm_feed_forward2.weight"));
        put(&mut w, format!("{p}.norm_feed_forward2.bias"), c, 0.02);
        put(&mut w, format!("{p}.feed_forward2.linear1.weight"), ffn * c, 0.05);
        put(&mut w, format!("{p}.feed_forward2.linear2.weight"), c * ffn, 0.05);
        gamma(&mut w, format!("{p}.norm_out.weight"));
        put(&mut w, format!("{p}.norm_out.bias"), c, 0.02);
    }
    put(&mut w, "prompt_projector.linear_1.weight".into(), pi * (c + np), 0.05);
    put(&mut w, "prompt_projector.linear_1.bias".into(), pi, 0.02);
    put(&mut w, "prompt_projector.linear_2.weight".into(), c * pi, 0.05);
    put(&mut w, "prompt_projector.linear_2.bias".into(), c, 0.02);
    put(&mut w, "encoder_projector.weight".into(), dh * c, 0.05);
    put(&mut w, "encoder_projector.bias".into(), dh, 0.02);
    w
}

#[test]
fn encoder_head_matches_reference_on_cpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    if available_devices().map(|d| d.is_empty()).unwrap_or(true) {
        brain_testutil::skip_unavailable("no OpenVINO runtime");
        return;
    }
    let (cfg, topo, t, valid, prompt_id) = tiny();
    let c = cfg.hidden as usize;
    let dh = cfg.decoder_hidden as usize;
    let w = weights(&cfg);

    // subsampled input [t, hidden] — O(1) features (they hit a LayerNorm first).
    let sub = fill(0xABCD, t as usize * c, 1.0);

    // host oracle
    let reference = encode_pooler(&sub, &w, &cfg, t as usize, valid as usize, prompt_id as usize);
    assert_eq!(reference.len(), t as usize * dh);

    // ONNX head: sub_in [t, C] -> pooler [t, dh]
    let mut g = onnx::GraphBuilder::new("nemotron_head");
    g.input_f32("sub_in", &[t as i64, c as i64]);
    build_nemotron_head(&mut g, &topo, &w, t, valid, prompt_id, "sub_in", "pooler");
    g.output_f32("pooler", &[t as i64, dh as i64]);
    let bytes = g.finish_with(onnx::DEFAULT_OPSET, onnx::DEFAULT_IR_VERSION);

    let cfgv = NpuConfig { device: NpuDevice::Cpu, perf_hint: PerfHint::Latency, allow_fallback: true, ..Default::default() };
    // NOT a skip. The test already established above that an OpenVINO runtime
    // is present, and this compiles OUR OWN emitted ONNX onto the always-present
    // CPU plugin with fallback allowed - so a failure here is a malformed graph
    // out of brain's exporter, not an unavailable machine. Swallowing it as a
    // skip is exactly how a broken exporter reports a green suite.
    let mut graph = NpuGraph::compile_bytes(&bytes, &cfgv).unwrap_or_else(|e| {
        panic!("OpenVINO is present but refused brain's emitted nemotron encoder head ONNX graph: {e:?}")
    });
    let out = graph.run(&[("sub_in", Feed::F32(&sub, vec![t as i64, c as i64]))]).expect("run head");
    let (_n, shape, data) = &out[0];
    eprintln!("pooler out shape {shape:?} ({} elems)", data.len());
    assert_eq!(data.len(), reference.len(), "pooler shape mismatch");

    // compare the VALID rows only (rows 0..valid), each dh long.
    let nvalid = valid as usize * dh;
    let (a, b) = (&data[..nvalid], &reference[..nvalid]);
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (na * nb + 1e-12);
    let maxdiff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("nemotron encoder-head ONNX(cpu) vs reference: cosine {cosine:.6}, maxdiff {maxdiff:.3e} (valid rows {valid}/{t})");
    assert!(cosine > 0.999, "encoder-head parity cosine {cosine} too low");
    assert!(maxdiff < 5e-2, "encoder-head parity maxdiff {maxdiff} too high");
}

/// Structural gate (no OpenVINO): the FULL encoder graph (subsampling → 24 blocks
/// → projectors) decodes to a well-formed ONNX model with the `mel` input, the
/// `pooler` output, and one attention Softmax per Conformer block — i.e. the
/// stage-1 (validated by nemotron_subsampling) and stage-2/3 (validated above)
/// composition wires up.
#[test]
fn full_encoder_graph_is_well_formed() {
    let (cfg, topo, _t, _v, prompt_id) = tiny();
    let mut w = weights(&cfg);
    // stage-1 (subsampling) weights
    let (ch, k, hidden) = (topo.subsampling_channels as usize, topo.subsampling_kernel as usize, topo.hidden as usize);
    let mut seed = 9000u64;
    let mut put = |w: &mut HashMap<String, Vec<f32>>, name: String, n: usize| {
        seed += 1;
        w.insert(name, fill(seed, n, 0.05));
    };
    put(&mut w, "encoder.subsampling.conv_in.weight".into(), ch * k * k);
    put(&mut w, "encoder.subsampling.conv_in.bias".into(), ch);
    for i in 0..(topo.subsampling_stages as usize - 1) {
        put(&mut w, format!("encoder.subsampling.layers.{i}.depthwise_conv.weight"), ch * k * k);
        put(&mut w, format!("encoder.subsampling.layers.{i}.depthwise_conv.bias"), ch);
        put(&mut w, format!("encoder.subsampling.layers.{i}.pointwise_conv.weight"), ch * ch);
        put(&mut w, format!("encoder.subsampling.layers.{i}.pointwise_conv.bias"), ch);
    }
    let flat = ch * topo.out_freq() as usize;
    put(&mut w, "encoder.subsampling.linear.weight".into(), hidden * flat);
    put(&mut w, "encoder.subsampling.linear.bias".into(), hidden);

    let (mel_t, mel_valid) = (32u32, 24u32);
    let bytes = npu::nemotron_export::build_nemotron_bytes(&w, &topo, mel_t, mel_valid, prompt_id);
    let model = onnx::decode_model(&bytes).expect("valid ONNX ModelProto");
    let g = model.graph.expect("model has a graph");
    assert!(g.input.iter().any(|v| v.name == "mel"), "missing mel input");
    assert!(g.output.iter().any(|v| v.name == "pooler"), "missing pooler output");
    let ops: Vec<&str> = g.node.iter().map(|n| n.op_type.as_str()).collect();
    for op in ["Conv", "MatMul", "Softmax", "Sigmoid", "Relu", "Pad", "Slice", "Transpose", "Concat"] {
        assert!(ops.contains(&op), "expected a {op} node in the full encoder");
    }
    let softmaxes = ops.iter().filter(|&&o| o == "Softmax").count();
    assert_eq!(softmaxes, topo.n_layers as usize, "one attention Softmax per Conformer block");
}
