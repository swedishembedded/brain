// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structural validation of the Qwen3-TTS **codec decoder** ONNX export.
//!
//! OpenVINO is not installed in this environment, so we cannot *run* the graph;
//! instead we build it from a tiny synthetic codec, serialize it, decode the
//! proto back, and assert it is well-formed and contains the expected op
//! vocabulary — in particular the new `ConvTranspose` (SEANet upsampling) and the
//! `Sin`/`Erf` primitives that compose SnakeBeta / exact-GELU. The `init_f32`
//! shape checks in the builder also validate every weight's declared shape.

use data::rng::Lcg;
use std::collections::HashMap;

use mimi::CodecConfig;

/// A deliberately tiny decoder config exercising every stage.
fn tiny_cfg() -> CodecConfig {
    CodecConfig {
        num_quantizers: 4, // 1 semantic + 3 acoustic
        num_semantic_quantizers: 1,
        codebook_size: 8,
        semantic_codebook_size: 8,
        codebook_dim: 8, // per-codebook gather dim = 4, quantizer out = 8
        latent_dim: 12,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        sliding_window: 72,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        layer_scale_initial_scale: 0.01,
        decoder_dim: 16,
        upsample_rates: vec![2, 2],
        upsampling_ratios: vec![2],
        input_sample_rate: 24000,
        output_sample_rate: 24000,
        decode_upsample_rate: 8,
        ..Default::default()
    }
}

/// Build a synthetic weight map with every tensor the decode graph reads, sized
/// to `cfg`. Values are small deterministic noise (shapes are what matter).
fn synth_weights(cfg: &CodecConfig) -> HashMap<String, Vec<f32>> {
    let dim = (cfg.codebook_dim / 2) as usize;
    let lat = cfg.codebook_dim as usize;
    let hidden = cfg.hidden_size as usize;
    let latent = cfg.latent_dim as usize;
    let ff = cfg.intermediate_size as usize;
    let hd = cfg.head_dim as usize;
    let nh = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let hq = nh * hd;
    let hkv = nkv * hd;
    let nq = cfg.num_quantizers as usize;
    let cb = cfg.codebook_size as usize;
    let dec = cfg.decoder_dim as usize;

    let mut m: HashMap<String, Vec<f32>> = HashMap::new();
    let mut seed = Lcg::new(1);
    let mut put = |m: &mut HashMap<String, Vec<f32>>, name: &str, numel: usize| {
        m.insert(name.to_string(), seed.vec_scaled(numel, 0.05));
    };

    // RVQ.
    put(&mut m, "quantizer.rvq_first.vq.layers.0.table", cb * dim);
    put(&mut m, "quantizer.rvq_first.output_proj.weight", lat * dim);
    for i in 0..(nq - 1) {
        put(&mut m, &format!("quantizer.rvq_rest.vq.layers.{i}.table"), cb * dim);
    }
    put(&mut m, "quantizer.rvq_rest.output_proj.weight", lat * dim);

    // pre_conv (lat -> latent, k3).
    put(&mut m, "pre_conv.conv.weight", latent * lat * 3);
    put(&mut m, "pre_conv.conv.bias", latent);

    // transformer.
    put(&mut m, "pre_transformer.input_proj.weight", hidden * latent);
    put(&mut m, "pre_transformer.input_proj.bias", hidden);
    for l in 0..cfg.num_hidden_layers as usize {
        let p = |s: &str| format!("pre_transformer.layers.{l}.{s}");
        put(&mut m, &p("input_layernorm.weight"), hidden);
        put(&mut m, &p("self_attn.q_proj.weight"), hq * hidden);
        put(&mut m, &p("self_attn.k_proj.weight"), hkv * hidden);
        put(&mut m, &p("self_attn.v_proj.weight"), hkv * hidden);
        put(&mut m, &p("self_attn.o_proj.weight"), hidden * hq);
        put(&mut m, &p("self_attn_layer_scale.scale"), hidden);
        put(&mut m, &p("post_attention_layernorm.weight"), hidden);
        put(&mut m, &p("mlp.gate_proj.weight"), ff * hidden);
        put(&mut m, &p("mlp.up_proj.weight"), ff * hidden);
        put(&mut m, &p("mlp.down_proj.weight"), hidden * ff);
        put(&mut m, &p("mlp_layer_scale.scale"), hidden);
    }
    put(&mut m, "pre_transformer.norm.weight", hidden);
    put(&mut m, "pre_transformer.output_proj.weight", latent * hidden);
    put(&mut m, "pre_transformer.output_proj.bias", latent);

    // upsample stages (ConvNeXt). channel = latent.
    for (u, &factor) in cfg.upsampling_ratios.iter().enumerate() {
        let f = factor as usize;
        put(&mut m, &format!("upsample.{u}.0.conv.weight"), latent * latent * f); // [Cin,Cout,K]
        put(&mut m, &format!("upsample.{u}.0.conv.bias"), latent);
        let p = format!("upsample.{u}.1");
        put(&mut m, &format!("{p}.dwconv.conv.weight"), latent * 7); // depthwise
        put(&mut m, &format!("{p}.dwconv.conv.bias"), latent);
        put(&mut m, &format!("{p}.norm.weight"), latent);
        put(&mut m, &format!("{p}.norm.bias"), latent);
        put(&mut m, &format!("{p}.pwconv1.weight"), (4 * latent) * latent);
        put(&mut m, &format!("{p}.pwconv1.bias"), 4 * latent);
        put(&mut m, &format!("{p}.pwconv2.weight"), latent * (4 * latent));
        put(&mut m, &format!("{p}.pwconv2.bias"), latent);
        put(&mut m, &format!("{p}.gamma"), latent);
    }

    // SEANet decoder.
    put(&mut m, "decoder.0.conv.weight", dec * latent * 7);
    put(&mut m, "decoder.0.conv.bias", dec);
    for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
        let r = rate as usize;
        let in_dim = dec >> i;
        let out_dim = dec >> (i + 1);
        let bp = format!("decoder.{}", i + 1);
        put(&mut m, &format!("{bp}.block.0.alpha"), in_dim);
        put(&mut m, &format!("{bp}.block.0.beta"), in_dim);
        put(&mut m, &format!("{bp}.block.1.conv.weight"), in_dim * out_dim * (2 * r)); // [Cin,Cout,K]
        put(&mut m, &format!("{bp}.block.1.conv.bias"), out_dim);
        for j in [2usize, 3, 4] {
            let rp = format!("{bp}.block.{j}");
            put(&mut m, &format!("{rp}.act1.alpha"), out_dim);
            put(&mut m, &format!("{rp}.act1.beta"), out_dim);
            put(&mut m, &format!("{rp}.conv1.conv.weight"), out_dim * out_dim * 7);
            put(&mut m, &format!("{rp}.conv1.conv.bias"), out_dim);
            put(&mut m, &format!("{rp}.act2.alpha"), out_dim);
            put(&mut m, &format!("{rp}.act2.beta"), out_dim);
            put(&mut m, &format!("{rp}.conv2.conv.weight"), out_dim * out_dim);
            put(&mut m, &format!("{rp}.conv2.conv.bias"), out_dim);
        }
    }
    let final_out = dec >> cfg.upsample_rates.len();
    put(&mut m, "decoder.5.alpha", final_out);
    put(&mut m, "decoder.5.beta", final_out);
    put(&mut m, "decoder.6.conv.weight", final_out * 7);
    put(&mut m, "decoder.6.conv.bias", 1);
    m
}

#[test]
fn codec_onnx_graph_is_well_formed_with_convtranspose() {
    let cfg = tiny_cfg();
    let w = synth_weights(&cfg);
    let t = 3usize;

    let mut g = onnx::GraphBuilder::new("codec_test");
    npu::codec_topology::build_codec_graph(&cfg, &w, t, &mut g);
    let bytes = g.finish();
    assert!(!bytes.is_empty());

    // Proto must decode (structurally valid ONNX).
    let model = onnx::decode_model(&bytes).expect("codec ONNX must decode");
    let graph = model.graph.expect("graph");

    // Op-type histogram.
    let mut hist: HashMap<&str, usize> = HashMap::new();
    for n in &graph.node {
        *hist.entry(n.op_type.as_str()).or_default() += 1;
    }
    eprintln!("codec ONNX op histogram: {hist:?}");

    // Expected op vocabulary.
    // ConvTranspose: 1 upsample stage + 2 SEANet block.1 = 3.
    assert_eq!(hist.get("ConvTranspose").copied().unwrap_or(0), 3, "ConvTranspose count");
    // Conv (causal/depthwise): pre_conv + dwconv + decoder.0 + per-block conv1/conv2.
    assert!(hist.get("Conv").copied().unwrap_or(0) >= 10, "Conv nodes present");
    // SnakeBeta -> Sin; exact GELU -> Erf.
    assert!(hist.get("Sin").copied().unwrap_or(0) >= 10, "SnakeBeta Sin present");
    assert_eq!(hist.get("Erf").copied().unwrap_or(0), 1, "GELU Erf (one ConvNeXt)");
    // transformer.
    assert!(hist.get("MatMul").copied().unwrap_or(0) >= 8, "MatMul present");
    assert!(hist.get("Softmax").copied().unwrap_or(0) >= 2, "attention softmax present");
    // tail clamp.
    assert_eq!(hist.get("Clip").copied().unwrap_or(0), 1, "tail Clip");

    // Graph IO.
    assert!(graph.input.iter().any(|i| i.name == "codes"));
    assert!(graph.output.iter().any(|o| o.name == "waveform"));

    // The output spatial length must be T * Π ratios * Π rates.
    let l: usize = t
        * cfg.upsampling_ratios.iter().product::<u32>() as usize
        * cfg.upsample_rates.iter().product::<u32>() as usize;
    let wav = graph.output.iter().find(|o| o.name == "waveform").unwrap();
    let dims: Vec<i64> = wav
        .r#type
        .as_ref()
        .unwrap()
        .tensor_type
        .as_ref()
        .unwrap()
        .shape
        .as_ref()
        .unwrap()
        .dim
        .iter()
        .map(|d| d.dim_value)
        .collect();
    assert_eq!(dims, vec![1, 1, l as i64], "waveform shape [1,1,T*upsample]");
}

/// The exported `pre_transformer` mask must be SLIDING-WINDOW causal, not
/// plain causal. This graph is the front the default streaming serve path runs
/// (`qwen3tts::serve` -> `npu_gen::NpuStreamCodec`), and it has to carry the
/// same mask as the two brain-side implementations of the same transformer
/// (`mimi::Codec::transformer`, `mimi::decode_stream::front`) or a long clip
/// decodes to a different waveform depending only on which backend served it.
///
/// The mask is a materialized initializer, so its content is checkable here
/// even though OpenVINO is absent and the graph cannot be RUN - the assertion
/// is on the real exported bytes (decoded back out of the proto), not on a
/// re-derivation of the builder's own logic.
#[test]
fn codec_onnx_attention_mask_is_sliding_window_causal() {
    let mut cfg = tiny_cfg();
    cfg.sliding_window = 3;
    let w = synth_weights(&cfg);
    let t = 10usize; // > sliding_window, so the window is not degenerate here
    let window = cfg.sliding_window as usize;

    let mut g = onnx::GraphBuilder::new("codec_mask_test");
    npu::codec_topology::build_codec_graph(&cfg, &w, t, &mut g);
    let model = onnx::decode_model(&g.finish()).expect("codec ONNX must decode");
    let graph = model.graph.expect("graph");
    let inits = onnx::initializers(&graph).expect("initializers must decode");
    let mask = inits.get("tf_mask").expect("tf_mask initializer must exist");
    assert_eq!(mask.shape, vec![1, 1, t, t], "mask shape");

    let mut live = 0usize;
    for i in 0..t {
        for j in 0..t {
            let v = mask.data[i * t + j];
            let want_masked = j > i || i - j >= window;
            if want_masked {
                assert!(v < -1.0e8, "mask[{i},{j}] must be masked out, got {v}");
            } else {
                assert_eq!(v, 0.0, "mask[{i},{j}] must be live");
                live += 1;
            }
        }
    }
    // Row `i` keeps min(i+1, window) keys, so the live count is a closed form:
    // a per-row bound that a plain causal mask (i+1 keys per row) cannot hit.
    let want_live: usize = (0..t).map(|i| (i + 1).min(window)).sum();
    assert_eq!(live, want_live, "live-key count is not the sliding-window one");
    let plain_causal: usize = (0..t).map(|i| i + 1).sum();
    assert_ne!(want_live, plain_causal, "test config is degenerate: window >= t");
}
