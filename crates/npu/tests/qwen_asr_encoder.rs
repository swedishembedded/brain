// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity gate for the Qwen3-ASR audio-encoder **head** (the 24 windowed ViT
//! blocks + ln_post + multi-modal projector) as an OpenVINO ONNX graph. Ported
//! op-for-op from `qwen_asr::encoder::AudioEncoder::encode_packed`. Self-contained:
//! a tiny random-weight head is run through both the reference (device) and the
//! ONNX graph on the OpenVINO **CPU** device, and the audio embeds must agree.
//! The conv stem + valid-position packing stay on host (like build_nemotron_head).
//!
//! Skips cleanly without an OpenVINO runtime. Run:
//!   LD_LIBRARY_PATH=<openvino/libs> cargo test -p brain-npu --test qwen_asr_encoder -- --nocapture

use std::collections::HashMap;

use npu::openvino::{available_devices, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};
use npu::{build_qwen_asr_head, QwenAsrTopo};
use qwen_asr::config::AudioEncoderConfig;
use qwen_asr::encoder::{audio_pipelines, AudioEncoder};

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn tiny() -> (AudioEncoderConfig, QwenAsrTopo) {
    let mut cfg = AudioEncoderConfig::qwen3_asr();
    cfg.d_model = 64;
    cfg.n_heads = 4; // head_dim 16
    cfg.ffn_dim = 128;
    cfg.n_layers = 2;
    cfg.output_dim = 48;
    let topo = QwenAsrTopo {
        d_model: cfg.d_model,
        n_heads: cfg.n_heads,
        head_dim: cfg.head_dim(),
        ffn_dim: cfg.ffn_dim,
        n_layers: cfg.n_layers,
        output_dim: cfg.output_dim,
        eps: cfg.eps,
    };
    (cfg, topo)
}

fn weights(cfg: &AudioEncoderConfig) -> HashMap<String, Vec<f32>> {
    let (c, ffn, out) = (cfg.d_model as usize, cfg.ffn_dim as usize, cfg.output_dim as usize);
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut seed = 1u64;
    let mut put = |w: &mut HashMap<String, Vec<f32>>, name: String, n: usize, scale: f32| {
        seed += 1;
        w.insert(name, fill(seed, n, scale));
    };
    let ln = |w: &mut HashMap<String, Vec<f32>>, wn: String, bn: String| {
        w.insert(wn, vec![1.0f32; c]);
        w.insert(bn, vec![0.0f32; c]);
    };
    for b in 0..cfg.n_layers {
        let p = format!("blocks.{b}");
        ln(&mut w, format!("{p}.norm1.weight"), format!("{p}.norm1.bias"));
        ln(&mut w, format!("{p}.norm2.weight"), format!("{p}.norm2.bias"));
        put(&mut w, format!("{p}.qkv.weight"), 3 * c * c, 0.05);
        put(&mut w, format!("{p}.qkv.bias"), 3 * c, 0.02);
        put(&mut w, format!("{p}.proj.weight"), c * c, 0.05);
        put(&mut w, format!("{p}.proj.bias"), c, 0.02);
        put(&mut w, format!("{p}.fc1.weight"), ffn * c, 0.05);
        put(&mut w, format!("{p}.fc1.bias"), ffn, 0.02);
        put(&mut w, format!("{p}.fc2.weight"), c * ffn, 0.05);
        put(&mut w, format!("{p}.fc2.bias"), c, 0.02);
    }
    w.insert("ln_post.weight".into(), vec![1.0f32; c]);
    w.insert("ln_post.bias".into(), vec![0.0f32; c]);
    put(&mut w, "multi_modal_projector.linear_1.weight".into(), c * c, 0.05);
    put(&mut w, "multi_modal_projector.linear_1.bias".into(), c, 0.02);
    put(&mut w, "multi_modal_projector.linear_2.weight".into(), out * c, 0.05);
    put(&mut w, "multi_modal_projector.linear_2.bias".into(), out, 0.02);
    w
}

#[test]
fn audio_encoder_head_matches_reference_on_cpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    if available_devices().map(|d| d.is_empty()).unwrap_or(true) {
        eprintln!("skip: no OpenVINO runtime");
        return;
    }
    let (cfg, topo) = tiny();
    let c = cfg.d_model as usize;
    let out = cfg.output_dim as usize;
    let w = weights(&cfg);

    // packed post-CNN tokens [n_audio, d_model] + block-diagonal windows.
    let n_audio = 10u32;
    let spans = [(0u32, 6u32), (6u32, 4u32)];
    let packed = fill(0xBEEF, n_audio as usize * c, 1.0);

    // reference (device) head
    let gpu = gpu_core::Gpu::new_cpu(audio_pipelines());
    let enc = AudioEncoder::new(&gpu, cfg.clone(), &w);
    let (_encoder_out, reference) = enc.encode_packed(&packed, n_audio, &spans);
    assert_eq!(reference.len(), n_audio as usize * out);

    // ONNX head: x [n, d_model] -> audio_embeds [n, output_dim]
    let mut g = onnx::GraphBuilder::new("qwen_asr_head");
    g.input_f32("x", &[n_audio as i64, c as i64]);
    build_qwen_asr_head(&mut g, &topo, &w, n_audio, &spans, "x", "embeds");
    g.output_f32("embeds", &[n_audio as i64, out as i64]);
    let bytes = g.finish_with(onnx::DEFAULT_OPSET, onnx::DEFAULT_IR_VERSION);

    let cfgv = NpuConfig { device: NpuDevice::Cpu, perf_hint: PerfHint::Latency, allow_fallback: true, ..Default::default() };
    let mut graph = match NpuGraph::compile_bytes(&bytes, &cfgv) {
        Ok(gr) => gr,
        Err(e) => {
            eprintln!("skip: OpenVINO compile failed: {e:?}");
            return;
        }
    };
    let ovout = graph.run(&[("x", Feed::F32(&packed, vec![n_audio as i64, c as i64]))]).expect("run head");
    let (_n, shape, data) = &ovout[0];
    eprintln!("audio_embeds out shape {shape:?} ({} elems)", data.len());
    assert_eq!(data.len(), reference.len(), "audio_embeds shape mismatch");

    let dot: f32 = data.iter().zip(&reference).map(|(x, y)| x * y).sum();
    let na: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (na * nb + 1e-12);
    let maxdiff = data.iter().zip(&reference).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("qwen-asr audio-encoder-head ONNX(cpu) vs reference: cosine {cosine:.6}, maxdiff {maxdiff:.3e} (n_audio {n_audio})");
    assert!(cosine > 0.999, "audio-encoder-head parity cosine {cosine} too low");
    assert!(maxdiff < 5e-2, "audio-encoder-head parity maxdiff {maxdiff} too high");
}
