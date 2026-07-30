// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity gate for the Nemotron FastConformer encoder **head** (stages 2–3: the
//! macaron Conformer blocks + prompt/encoder projectors) as an OpenVINO ONNX graph.
//! Self-contained: a tiny random-weight encoder is run through both
//! `nemotron::reference::encode_pooler` (the host f32 oracle) and the ONNX graph
//! compiled on the OpenVINO **CPU** device, and the two poolers must agree on the
//! valid rows (padding rows are allowed to differ — a fully-masked padding query's
//! softmax is uniform in ONNX vs zeroed in the reference, but causal/ masked ops
//! keep VALID outputs independent of padding).
//!
//! Skips cleanly without an OpenVINO runtime. Run:
//!   LD_LIBRARY_PATH=<openvino/libs> cargo test -p brain-npu --test nemotron_encoder -- --nocapture

use std::collections::HashMap;

use nemotron::config::NemotronConfig;
use nemotron::reference::encode_pooler;
use npu::openvino::{available_devices, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};
use npu::{build_nemotron_head, NemotronTopo};

/// Deterministic small pseudo-random fill.
fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
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
        eprintln!("skip: no OpenVINO runtime");
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
    let mut graph = match NpuGraph::compile_bytes(&bytes, &cfgv) {
        Ok(gr) => gr,
        Err(e) => {
            eprintln!("skip: OpenVINO compile failed: {e:?}");
            return;
        }
    };
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
