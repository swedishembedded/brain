// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Validate the Qwen ONNX decoder export against brain's own forward: a tiny
//! model is run through brain (CPU/WGSL) and through OpenVINO (CPU device) from
//! the same checkpoint; per-position argmax must agree and logits match closely.
//! Proves the ONNX graph math (RMSNorm/QK-norm/RoPE/GQA/SwiGLU/tied head) is
//! correct independent of model size. Gated on BRAIN_OV_PROBE (needs OpenVINO).

#[test]
fn tiny_onnx_matches_brain_cpu() {
    if std::env::var("BRAIN_OV_PROBE").is_err() {
        return;
    }
    use npu::openvino::{DecoderSession, NpuConfig, NpuDevice};
    use qwen::{Qwen, QwenConfig};

    let cfg = QwenConfig::tiny();
    let block = cfg.block_size;
    let init = qwen::init_weights(&cfg, 7);
    let model = Qwen::new(cfg.clone(), 1, block, &init);
    let ids: Vec<u32> = (0..6).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let reference = model.logits_all(&ids); // [6 * vocab]
    let vocab = cfg.vocab as usize;

    let dir = std::env::temp_dir().join(format!("brain_qwen_onnx_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wpath = dir.join("tiny.safetensors");
    model.save(wpath.to_str().unwrap());

    let (bytes, _) = npu::qwen_export::build_qwen_fp32_bytes(wpath.to_str().unwrap(), ids.len()).unwrap();
    let mut sess = DecoderSession::load_bytes(
        &bytes,
        &NpuConfig { device: NpuDevice::Cpu, allow_fallback: true, ..Default::default() },
    )
    .expect("compile decoder on OpenVINO CPU");

    let ids64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let got = sess.run_ids(&ids64).expect("run decoder");
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(got.len(), reference.len(), "logit count");
    let argmax = |s: &[f32]| (0..s.len()).max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap()).unwrap();
    let mut max_abs = 0f32;
    for p in 0..ids.len() {
        let r = &reference[p * vocab..(p + 1) * vocab];
        let g = &got[p * vocab..(p + 1) * vocab];
        assert_eq!(argmax(r), argmax(g), "argmax disagree at position {p}");
        for (a, b) in r.iter().zip(g) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    eprintln!("tiny ONNX vs brain CPU: max_abs={max_abs:.5} (device {})", sess.device());
    assert!(max_abs < 1e-2, "logit mismatch too large: {max_abs}");
}
