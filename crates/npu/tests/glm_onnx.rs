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
//! is gated separately (see docs/models/glm/npu.md); it is not asserted here.

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
    let path = dir.join("glm.safetensors");
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

/// Numerical parity: run the exported GLM ONNX through OpenVINO (CPU device) and
/// compare per-position logits to brain's own forward. Validates the whole MLA +
/// dense-expert MoE graph math. Gated on BRAIN_OV_PROBE (needs OpenVINO) and
/// MOE_SKIP_GPU_TESTS (brain's reference forward needs a backend).
#[test]
fn glm_onnx_matches_brain_forward() {
    if std::env::var("BRAIN_OV_PROBE").is_err() || std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use glm::model::Glm;
    use npu::openvino::{DecoderSession, NpuConfig, NpuDevice};

    let cfg = GlmConfig::tiny(); // layer 0 dense, layer 1 MoE
    let vocab = cfg.vocab as usize;
    let init = glm::init_weights(&cfg, 7);
    let model = Glm::new(cfg.clone(), 1, cfg.block_size, &init);
    let ids: Vec<u32> = (0..6).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let reference = model.logits_all(&ids); // [6 * vocab]

    let dir = std::env::temp_dir().join(format!("brain-glm-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wpath = dir.join("tiny.safetensors");
    model.save(wpath.to_str().unwrap());

    let (bytes, _) = npu::glm_export::build_glm_fp32_bytes(wpath.to_str().unwrap(), ids.len()).unwrap();
    let mut sess = DecoderSession::load_bytes(
        &bytes,
        &NpuConfig { device: NpuDevice::Cpu, allow_fallback: true, ..Default::default() },
    )
    .expect("compile GLM decoder on OpenVINO CPU");

    let ids64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let got = sess.run_ids(&ids64).expect("run GLM decoder");
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(got.len(), reference.len(), "logit count");
    let argmax = |s: &[f32]| (0..s.len()).max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap()).unwrap();
    let mut max_abs = 0f32;
    for pos in 0..ids.len() {
        let r = &reference[pos * vocab..(pos + 1) * vocab];
        let gg = &got[pos * vocab..(pos + 1) * vocab];
        for (a, b) in r.iter().zip(gg) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert_eq!(argmax(r), argmax(gg), "argmax disagree at position {pos} (max_abs so far {max_abs})");
    }
    eprintln!("GLM ONNX vs brain: max_abs={max_abs:.5} (device {})", sess.device());
    assert!(max_abs < 1e-2, "logit mismatch too large: {max_abs}");
}

/// Compile + run the exported GLM ONNX on the actual Intel **NPU** device (with
/// CPU fallback) and check parity vs brain. The NPU plugin may reject some ops
/// (TopK / Scatter) and fall back — the test reports the device it landed on.
/// Gated on BRAIN_OV_PROBE + a working backend.
#[test]
fn glm_onnx_runs_on_npu() {
    if std::env::var("BRAIN_OV_PROBE").is_err() || std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use glm::model::Glm;
    use npu::openvino::{DecoderSession, NpuConfig, NpuDevice};

    let cfg = GlmConfig::tiny();
    let vocab = cfg.vocab as usize;
    let init = glm::init_weights(&cfg, 7);
    let model = Glm::new(cfg.clone(), 1, cfg.block_size, &init);
    let ids: Vec<u32> = (0..6).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let reference = model.logits_all(&ids);

    let dir = std::env::temp_dir().join(format!("brain-glm-npu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wpath = dir.join("tiny.safetensors");
    model.save(wpath.to_str().unwrap());
    let (bytes, _) = npu::glm_export::build_glm_fp32_bytes(wpath.to_str().unwrap(), ids.len()).unwrap();

    let sess = DecoderSession::load_bytes(
        &bytes,
        &NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() },
    );
    let mut sess = match sess {
        Ok(s) => s,
        Err(e) => {
            eprintln!("GLM on NPU: compile failed ({e}); skipping (NPU op support gap)");
            std::fs::remove_dir_all(&dir).ok();
            return;
        }
    };
    let ids64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let got = sess.run_ids(&ids64).expect("run GLM decoder on NPU/fallback");
    std::fs::remove_dir_all(&dir).ok();

    let argmax = |s: &[f32]| (0..s.len()).max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap()).unwrap();
    let mut max_abs = 0f32;
    for pos in 0..ids.len() {
        let r = &reference[pos * vocab..(pos + 1) * vocab];
        let gg = &got[pos * vocab..(pos + 1) * vocab];
        for (a, b) in r.iter().zip(gg) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert_eq!(argmax(r), argmax(gg), "argmax disagree at position {pos}");
    }
    eprintln!("GLM ONNX on NPU-requested: max_abs={max_abs:.5} (ran on {})", sess.device());
    assert!(max_abs < 2e-2, "logit mismatch too large: {max_abs}");
}

/// The INT8 weight-only GLM export compiles + runs on OpenVINO and roughly tracks
/// brain (INT8 is lossy, so we check finiteness + a loose bound, not strict
/// parity). Validates the in-graph DequantizeLinear path. Gated on BRAIN_OV_PROBE.
#[test]
fn glm_onnx_int8_runs() {
    if std::env::var("BRAIN_OV_PROBE").is_err() || std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use glm::model::Glm;
    use npu::openvino::{DecoderSession, NpuConfig, NpuDevice};

    let cfg = GlmConfig::tiny();
    let init = glm::init_weights(&cfg, 7);
    let model = Glm::new(cfg.clone(), 1, cfg.block_size, &init);
    let ids: Vec<u32> = (0..6).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let reference = model.logits_all(&ids);

    let dir = std::env::temp_dir().join(format!("brain-glm-int8-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wpath = dir.join("tiny.safetensors");
    model.save(wpath.to_str().unwrap());
    let (bytes, _) = npu::glm_export::build_glm_int8_bytes(wpath.to_str().unwrap(), ids.len()).unwrap();
    let mut sess = DecoderSession::load_bytes(
        &bytes,
        &NpuConfig { device: NpuDevice::Cpu, allow_fallback: true, ..Default::default() },
    )
    .expect("compile INT8 GLM decoder");
    let ids64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let got = sess.run_ids(&ids64).expect("run INT8 GLM decoder");
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(got.len(), reference.len());
    assert!(got.iter().all(|v| v.is_finite()), "INT8 logits must be finite");
    let mut max_abs = 0f32;
    for (a, b) in reference.iter().zip(&got) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!("GLM INT8 ONNX vs brain: max_abs={max_abs:.4} (device {})", sess.device());
    assert!(max_abs < 2.0, "INT8 logits wildly off: {max_abs}"); // loose: lossy quant on a random tiny model
}
