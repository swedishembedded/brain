// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The saved-activation policy (`model::saved`) on the qwen3 trainer: holding the layer-boundary residuals
//! in bf16 or FP8 (one scale per row) trains the adapters the fp32-boundary build does to the format's
//! precision, rounds the stream at every boundary (so the loss is not bitwise the same), and keeps a smaller
//! boundary stream than the fp32 pair-per-layer build. `Keep` has no implementation in this trainer and is
//! refused rather than silently replaced.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::provider::library::GemmPrecision;
use model::saved::{SavedFormat, SavedPolicy};
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

const TOKENS: [u32; 12] = [1, 5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2];
const TARGETS: [u32; 12] = [5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2, 1];

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

fn config() -> QwenConfig {
    QwenConfig { block_size: 12, lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() }
}

fn weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 21);
    let mut rng = Rng::random();
    for (name, v) in w.iter_mut() {
        if name.ends_with(".lora_b") {
            v.iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * 0.3);
        }
    }
    w
}

fn build(cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>, saved: SavedPolicy) -> Qwen {
    Qwen::new_lora_dt_saved(cfg.clone(), 1, 12, init, Dtype::F32, GemmPrecision::Fp32Reference, saved)
}

fn loss_and_grads(m: &Qwen) -> (f32, Vec<(String, Vec<f32>)>) {
    m.set_batch(&TOKENS, &TARGETS);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    let grads = model::Model::optimized_params(m).unwrap().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, grads)
}

#[test]
fn quantized_boundaries_train_the_adapters_the_fp32_boundaries_do_to_the_formats_precision() {
    let cfg = config();
    let init = weights(&cfg);
    let (want_loss, want) = loss_and_grads(&build(&cfg, &init, SavedPolicy::Recompute));
    for (format, loss_tol, min_cos) in [(SavedFormat::Bf16, 2e-2, 0.99), (SavedFormat::Fp8, 1.5e-1, 0.95)] {
        let (loss, got) = loss_and_grads(&build(&cfg, &init, SavedPolicy::Quantized(format)));
        assert_ne!(loss, want_loss, "{format:?}: the stream is rounded at every boundary");
        assert!((loss - want_loss).abs() < loss_tol * want_loss.abs().max(1.0), "{format:?}: loss {loss} vs {want_loss}");
        let mut nonzero = 0;
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            if w.iter().all(|x| *x == 0.0) {
                continue;
            }
            nonzero += 1;
            let c = cosine(g, w);
            assert!(c > min_cos, "{format:?} {name}: adapter gradient cosine {c}");
        }
        assert!(nonzero >= 4, "{format:?}: the adapters received gradient ({nonzero} tensors)");
    }
}

#[test]
fn a_quantized_stream_holds_fewer_device_bytes_than_the_fp32_one() {
    memauth::publish_limits(Some(64 << 30), Some(64 << 30));
    // Many layers and rows so the boundary stream dominates the fixed buffers.
    let cfg = QwenConfig { n_layers: 12, block_size: 96, max_position_embeddings: 96, ..config() };
    let init = qwen3::init_weights(&cfg, 5);
    let bytes = |saved| Qwen::new_lora_dt_saved(cfg.clone(), 1, 96, &init, Dtype::F32, GemmPrecision::Fp32Reference, saved).gpu.charged_bytes();
    let (fp32, bf16, fp8) = (bytes(SavedPolicy::Recompute), bytes(SavedPolicy::Quantized(SavedFormat::Bf16)), bytes(SavedPolicy::Quantized(SavedFormat::Fp8)));
    assert!(fp8 < bf16 && bf16 < fp32, "fp32 {fp32} B, bf16 {bf16} B, fp8 {fp8} B");
}

#[test]
#[should_panic(expected = "SavedPolicy::Keep is not available")]
fn keeping_every_activation_is_refused_by_a_trainer_that_always_recomputes() {
    let cfg = config();
    build(&cfg, &weights(&cfg), SavedPolicy::Keep);
}
