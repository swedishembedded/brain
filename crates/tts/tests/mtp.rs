// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Env-gated MTP real-checkpoint import + forward.
use tts::MtpModel;

#[test]
fn mtp_real_import_and_forward() {
    let Ok(dir) = std::env::var("BRAIN_TTS_CKPT") else {
        return;
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let out = std::env::temp_dir().join("brain_mtp_test.weights");
    let out = out.to_str().unwrap();
    tts::import::import_mtp(&dir, out).expect("mtp import");
    let m = MtpModel::load_inference_on(gpu_core::testgpu::dev(tts::mtp::PIPELINES), out);
    assert_eq!(m.cfg.vocab, 2048);
    assert_eq!(m.cfg.num_code_groups, 16);
    let d = m.cfg.d_model as usize;
    let th = vec![0.1f32; d];
    let cb0 = vec![-0.1f32; d];
    let residual: Vec<u32> = (0..(m.cfg.num_code_groups as usize - 2))
        .map(|i| (i * 7 % 2048) as u32)
        .collect();
    let embeds = m.assemble(&th, &cb0, &residual);
    let logits = m.logits(&embeds);
    assert_eq!(logits.len(), (m.cfg.num_code_groups as usize - 1) * 2048);
    assert!(logits.iter().all(|x| x.is_finite()));
    let _ = std::fs::remove_file(out);
}
