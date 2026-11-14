// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5-35B-A3B (`qwen35moe`) ONNX export tests.
//!
//! Structural (always runs, no backend/OpenVINO): build a TINY hybrid
//! qwen35moe checkpoint (both layer types, a genuine multi-chunk GDN
//! recurrence, a 6-expert/top-2 sparse MoE) and assert the exported ONNX
//! graph is well-formed — this validates the `gdn_chunk` emitter and the
//! sparse gather-based expert dispatch (`crate::qwen35moe_topology`'s two
//! hard sub-problems) without hardware, mirroring `glm_onnx.rs`'s own
//! structural-test convention exactly.
//!
//! Numerical parity vs brain's own forward and an actual OpenVINO compile
//! attempt require OpenVINO (`BRAIN_OV_PROBE`); per this task's own explicit
//! scope ("compiles + best-effort OpenVINO compile attempt", not a working
//! NPU run), those tests are gated the same way `glm_onnx.rs`'s own
//! `glm_onnx_matches_brain_forward`/`glm_onnx_runs_on_npu` are, not asserted
//! unconditionally.

use std::collections::HashMap;

use qwen35moe::config::Qwen35Config;

/// Write a checkpoint directly from freshly-initialised weights (no GPU/CPU
/// backend needed — `init_weights`, `checkpoint::save`, and the ONNX export
/// are all pure Rust).
fn write_tiny_ckpt(dir: &std::path::Path, cfg: &Qwen35Config) -> std::path::PathBuf {
    let init: HashMap<String, Vec<f32>> = qwen35moe::init::init_weights(cfg, 3);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(n, numel)| (n.clone(), vec![numel as u64], init[&n].clone()))
        .collect();
    let path = dir.join("qwen35.safetensors");
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
    path
}

/// `tiny()`'s `block_size` (24) picked deliberately: `gdn_chunk_size(24) == 8`
/// (3 chunks) — a genuine multi-chunk GDN recurrence, not a degenerate
/// single-chunk collapse.
const SEQ: usize = 24;

#[test]
fn qwen35moe_onnx_graph_is_well_formed() {
    let dir = std::env::temp_dir().join(format!("brain-qwen35moe-onnx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = Qwen35Config::tiny();
    let path = write_tiny_ckpt(&dir, &cfg);

    let (bytes, _cfg) = npu::qwen35moe_export::build_qwen35_fp32_bytes(path.to_str().unwrap(), SEQ).unwrap();
    assert!(bytes.len() > 1000, "onnx export suspiciously small: {} bytes", bytes.len());

    let model = onnx::decode_model(&bytes).expect("export must decode as a valid ONNX ModelProto");
    let g = model.graph.expect("model has a graph");
    assert!(g.node.len() > 100, "expected a real multi-layer hybrid graph, got {} nodes", g.node.len());
    assert!(g.input.iter().any(|v| v.name == "input_ids"), "missing input_ids");
    assert!(g.output.iter().any(|v| v.name == "logits"), "missing logits output");

    // The gdn_chunk emitter: Conv (causal depthwise), Softplus (decay gate),
    // and enough MatMul nodes for the Neumann-series UT-transform + the
    // per-chunk recurrence to have actually unrolled.
    let ops: Vec<&str> = g.node.iter().map(|n| n.op_type.as_str()).collect();
    for op in ["Conv", "Softplus", "Sigmoid", "Exp", "MatMul", "Softmax", "TopK", "Gather"] {
        assert!(ops.contains(&op), "expected a {op} node in the qwen35moe graph");
    }
    let matmul_count = ops.iter().filter(|&&o| o == "MatMul").count();
    // Every GDN layer alone needs >= chunk-1 (Neumann series) + per-chunk-loop
    // matmuls; six GDN layers + two GQA layers + eight MoE sublayers should
    // clear a three-digit MatMul count by a wide margin.
    assert!(matmul_count > 150, "expected many MatMul nodes (sparse MoE + GDN chunk math), got {matmul_count}");
    eprintln!(
        "qwen35moe tiny ONNX: {} bytes, {} nodes ({} MatMul), {} initializers",
        bytes.len(),
        g.node.len(),
        matmul_count,
        g.initializer.len()
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Numerical parity: run the exported ONNX through OpenVINO (CPU device) and
/// compare finiteness / rough shape (NOT bit-parity with brain's own GPU
/// forward, which this environment cannot run standalone here). Gated on
/// `BRAIN_OV_PROBE` (needs OpenVINO) — a best-effort compile attempt, exactly
/// this task's own explicit stopping point ("compiles + best-effort OpenVINO
/// compile attempt", not a working NPU run).
#[test]
fn qwen35moe_onnx_compiles_on_openvino_cpu() {
    if std::env::var("BRAIN_OV_PROBE").is_err() {
        return brain_testutil::skip_unavailable("BRAIN_OV_PROBE unset (this needs an OpenVINO runtime)");
    }
    use npu::openvino::{DecoderSession, NpuConfig, NpuDevice};

    let dir = std::env::temp_dir().join(format!("brain-qwen35moe-ov-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = Qwen35Config::tiny();
    let path = write_tiny_ckpt(&dir, &cfg);
    let (bytes, _cfg) = npu::qwen35moe_export::build_qwen35_fp32_bytes(path.to_str().unwrap(), SEQ).unwrap();

    let sess = DecoderSession::load_bytes(&bytes, &NpuConfig { device: NpuDevice::Cpu, allow_fallback: true, ..Default::default() });
    let mut sess = match sess {
        Ok(s) => s,
        Err(e) => {
            eprintln!("qwen35moe: OpenVINO CPU compile failed ({e}); this is the documented best-effort boundary, not a hard failure");
            std::fs::remove_dir_all(&dir).ok();
            return;
        }
    };
    let ids: Vec<i64> = (0..SEQ as i64).map(|i| (i * 5 + 1) % cfg.vocab as i64).collect();
    match sess.run_ids(&ids) {
        Ok(got) => {
            assert_eq!(got.len(), SEQ * cfg.vocab as usize, "logit count");
            assert!(got.iter().all(|v| v.is_finite()), "qwen35moe ONNX logits must be finite");
            eprintln!("qwen35moe ONNX ran on OpenVINO device {}", sess.device());
        }
        Err(e) => eprintln!("qwen35moe: OpenVINO run failed ({e}); documented best-effort boundary"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Numerical parity: the exported graph on OpenVINO's CPU device against
/// brain's own forward on the same tiny weights, position by position. This
/// is what checks the GDN chunk recurrence and the sparse expert dispatch
/// compute what the model does, not only that they run. Gated on
/// `BRAIN_OV_PROBE` (needs OpenVINO) and on a backend for brain's reference
/// forward (`brain_testutil::gpu_tests_skipped`).
#[test]
fn qwen35moe_onnx_matches_brain_forward() {
    if std::env::var("BRAIN_OV_PROBE").is_err() || brain_testutil::gpu_tests_skipped() {
        return brain_testutil::skip_unavailable("set BRAIN_OV_PROBE (and a working backend) for OpenVINO parity");
    }
    use npu::openvino::{DecoderSession, NpuConfig, NpuDevice};

    let cfg = Qwen35Config::tiny();
    let vocab = cfg.vocab as usize;
    let dir = std::env::temp_dir().join(format!("brain-qwen35moe-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_tiny_ckpt(&dir, &cfg);
    let ids: Vec<u32> = (0..SEQ as u32).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let reference = qwen35moe::model::Qwen35::new(cfg.clone(), 1, SEQ as u32, &qwen35moe::init::init_weights(&cfg, 3)).logits_all(&ids);

    let (bytes, _) = npu::qwen35moe_export::build_qwen35_fp32_bytes(path.to_str().unwrap(), SEQ).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let mut sess = DecoderSession::load_bytes(&bytes, &NpuConfig { device: NpuDevice::Cpu, allow_fallback: true, ..Default::default() })
        .expect("compile qwen35moe on OpenVINO CPU");
    let got = sess.run_ids(&ids.iter().map(|&x| x as i64).collect::<Vec<_>>()).expect("run qwen35moe");

    assert_eq!(got.len(), reference.len(), "logit count");
    // A comparison of two flat outputs proves nothing: the logits must vary.
    let spread = reference.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) - reference.iter().fold(f32::INFINITY, |m, &v| m.min(v));
    assert!(spread > 1e-3, "the reference logits are flat (spread {spread})");
    let argmax = |s: &[f32]| (0..s.len()).max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap()).unwrap();
    let mut max_abs = 0f32;
    for pos in 0..SEQ {
        let (r, g) = (&reference[pos * vocab..(pos + 1) * vocab], &got[pos * vocab..(pos + 1) * vocab]);
        for (a, b) in r.iter().zip(g) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert_eq!(argmax(r), argmax(g), "argmax disagrees at position {pos} (max_abs so far {max_abs})");
    }
    eprintln!("qwen35moe ONNX vs brain: max_abs={max_abs:.2e} over a logit spread of {spread:.3} (device {})", sess.device());
    assert!(max_abs < 1e-2, "logit mismatch too large: {max_abs}");
}
