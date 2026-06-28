// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Talker forward + gradient-check integration tests.
//!
//! The Talker decoder is a Qwen3 dense decoder (untied codec embedding/head), so
//! the gradient check exercises GQA + per-head QK-norm + half-split RoPE-base +
//! SwiGLU + the (untied) codec head backprop through the shared `model::block`
//! builders — the same parity-exact path as `crate::qwen`.

use tts::{TalkerConfig, TalkerModel};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

#[test]
fn talker_forward_finite_correct_shape() {
    if gpu_disabled() {
        return;
    }
    let cfg = TalkerConfig::tiny(); // vocab 23, d 16, L2, GQA 4/2, head_dim 8
    let vocab = cfg.vocab as usize;
    let model = TalkerModel::new_trainable(cfg, 1, 8, 7);
    let codec: Vec<u32> = (0..6).map(|i| (i * 3 % 23) as u32).collect();
    let logits = model.logits_all(&codec);
    assert_eq!(
        logits.len(),
        codec.len() * vocab,
        "logits must be [T, vocab]"
    );
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits must be finite"
    );
}

/// Local newtype so we can implement the external `CheckModel` trait (orphan
/// rule) — delegates to the Talker's public forward/backward surface.
struct Checkable(TalkerModel);

impl gradcheck::CheckModel for Checkable {
    fn param_names(&self) -> Vec<String> {
        self.0.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.forward()
    }
    fn zero_grads(&self) {
        self.0.zero_grads();
    }
    fn backward(&self) {
        self.0.backward();
    }
}

#[test]
fn talker_analytic_grads_match_finite_differences() {
    if gpu_disabled() {
        return;
    }
    let model = TalkerModel::new_trainable(TalkerConfig::tiny(), 2, 6, 7);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_codec_batch(&x, &y);
    let report = gradcheck::directional_check(&Checkable(model), 5e-3, 4, 0x7abc);
    report.print();
    // fp32 directional finite differences on the software GPU: the framework's
    // proven combined tolerance (same as every other brain model's gate).
    let (atol, rtol) = (4e-3, 8e-2);
    let fails = report.failures(atol, rtol);
    assert!(
        fails.is_empty(),
        "Talker gradient check failed for {:?}",
        fails
            .iter()
            .map(|c| (&c.param, c.abs_err, c.rel_err))
            .collect::<Vec<_>>()
    );
}

/// Env-gated real-checkpoint import + forward (set `BRAIN_TTS_CKPT` to the
/// `Qwen3-TTS-12Hz-0.6B-Base` dir). Verifies the import consumes every Talker
/// tensor with matching shapes and produces finite `[T, 3072]` codebook-0 logits.
#[test]
fn talker_real_import_and_forward() {
    let Ok(dir) = std::env::var("BRAIN_TTS_CKPT") else {
        return;
    };
    if gpu_disabled() {
        return;
    }
    let out = std::env::temp_dir().join("brain_talker_test.weights");
    let out = out.to_str().unwrap();
    tts::import::import_talker(&dir, out).expect("talker import");
    let model = TalkerModel::load_inference(out, 1, 32);
    assert_eq!(model.vocab(), 3072);
    assert!(model.text.is_some(), "text_projection should load");
    let codec: Vec<u32> = vec![2149, 0, 5, 100, 2150];
    let logits = model.logits_all(&codec);
    assert_eq!(logits.len(), codec.len() * 3072);
    assert!(logits.iter().all(|x| x.is_finite()));
    let _ = std::fs::remove_file(out);
}
