// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder ONNX export tests.
//!
//! Structural (always runs): tiny checkpoint → export → decode → op/shape
//! assertions, at a materialized S and at an S above the chunking threshold
//! (the statically-unrolled query-chunked attention).
//!
//! Numerical parity vs brain's own forward runs the graph through OpenVINO's
//! CPU plugin — gated on BRAIN_OV_PROBE (needs the OpenVINO runtime) and
//! MOE_SKIP_GPU_TESTS unset (brain's reference forward needs a backend). NPU
//! compilation itself only happens on a machine with the Intel NPU driver.

use std::collections::HashMap;

use lfm2::config::LfmConfig;

fn write_tiny_ckpt(dir: &std::path::Path) -> (std::path::PathBuf, LfmConfig) {
    let cfg = LfmConfig::tiny(); // conv, attention, conv — both mixers
    let init: HashMap<String, Vec<f32>> = lfm2::init::init_weights(&cfg, 3);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(n, numel)| (n.clone(), vec![numel as u64], init[&n].clone()))
        .collect();
    let path = dir.join("lfm.safetensors");
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
    (path, cfg)
}

fn assert_well_formed(bytes: &[u8], expect_chunks: bool) {
    let model = onnx::decode_model(bytes).expect("valid ONNX ModelProto");
    let g = model.graph.expect("model has a graph");
    assert!(g.input.iter().any(|v| v.name == "ids"), "missing ids input");
    assert!(g.input.iter().any(|v| v.name == "kmask"), "missing kmask input");
    assert!(g.output.iter().any(|v| v.name == "hidden"), "missing hidden output");
    let ops: Vec<&str> = g.node.iter().map(|n| n.op_type.as_str()).collect();
    for op in ["Gather", "MatMul", "Softmax", "Conv", "Sigmoid", "Mul", "Transpose"] {
        assert!(ops.contains(&op), "expected a {op} node");
    }
    let softmaxes = ops.iter().filter(|&&o| o == "Softmax").count();
    if expect_chunks {
        assert!(softmaxes > 1, "chunked attention should emit one Softmax per query chunk, got {softmaxes}");
    } else {
        assert_eq!(softmaxes, 1, "tiny config has one attention layer");
    }
}

#[test]
fn lfm_onnx_graph_is_well_formed() {
    let dir = std::env::temp_dir().join(format!("brain-lfm-onnx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (path, _cfg) = write_tiny_ckpt(&dir);
    let (bytes, _) = npu::lfm_export::build_lfm_bytes(path.to_str().unwrap(), 8, false).unwrap();
    assert_well_formed(&bytes, false);
    // Above the chunk threshold: the same graph shape but query-chunked.
    let (bytes, _) = npu::lfm_export::build_lfm_bytes(path.to_str().unwrap(), 4096, false).unwrap();
    assert_well_formed(&bytes, true);
    std::fs::remove_dir_all(&dir).ok();
}

/// Parity: exported graph through OpenVINO (CPU plugin) vs brain's forward.
#[test]
fn lfm_onnx_matches_brain_forward() {
    if std::env::var("BRAIN_OV_PROBE").is_err() || std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("set BRAIN_OV_PROBE (and a working backend) for OpenVINO parity");
        return;
    }
    use npu::openvino::{LfmSession, NpuConfig, NpuDevice};

    let dir = std::env::temp_dir().join(format!("brain-lfm-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (path, cfg) = write_tiny_ckpt(&dir);

    let s = 8usize;
    let ids: Vec<u32> = (0..s as u32).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let model = lfm2::Lfm::new(cfg.clone(), 1, s as u32, &checkpoint::load(path.to_str().unwrap()).by_role(""));
    model.set_tokens(&ids);
    model.forward();
    let reference = model.read_hidden();
    drop(model);

    let (bytes, _) = npu::lfm_export::build_lfm_bytes(path.to_str().unwrap(), s, false).unwrap();
    let mut sess = LfmSession::load_bytes(
        &bytes,
        &NpuConfig { device: NpuDevice::Cpu, allow_fallback: true, ..Default::default() },
    )
    .expect("compile LFM encoder on OpenVINO CPU");
    let ids64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let got = sess.run(&ids64, &vec![0.0; s]).expect("run LFM encoder");
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(got.len(), reference.len());
    let (mut dot, mut na, mut nb, mut max_abs) = (0f64, 0f64, 0f64, 0f32);
    for (a, b) in reference.iter().zip(&got) {
        dot += *a as f64 * *b as f64;
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!("LFM ONNX vs brain: cosine={cos:.6} max_abs={max_abs:.5} (device {})", sess.device());
    assert!(cos > 0.9999, "cosine {cos}");
    assert!(max_abs < 1e-2, "max_abs {max_abs}");
}
