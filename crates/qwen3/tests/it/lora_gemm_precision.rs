// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA fine-tuning of large
// language models on a single GPU for its clients. If your team needs
// expertise in making a tensor-core training path agree with its fp32
// reference then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The TF32 tensor-core tier trains the adapters the exact fp32 tier does, to
//! TF32 rounding: on one tiny LoRA model the loss and every adapter gradient of
//! `GemmPrecision::Auto` agree with `Fp32Reference` (loss to 1e-3,
//! gradient cosine above 0.999), the reference tier never reports TF32, and a
//! device without the vendor library runs the fp32 kernels and the comparison
//! is exact. The adapters' own products stay fp32 in both tiers.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::provider::library::GemmPrecision;
use qwen3::{LoraCfg, Qwen, QwenConfig};

const TOKENS: [u32; 12] = [1, 5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2];
const TARGETS: [u32; 12] = [5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2, 1];
/// Rows of the wide-head model: the flash trio's shortest training sequence, so attention takes it.
const WIDE_T: usize = 256;

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

fn loss_and_grads(m: &Qwen) -> (f32, Vec<(String, Vec<f32>)>) {
    loss_and_grads_on(m, &TOKENS, &TARGETS)
}

fn loss_and_grads_on(m: &Qwen, tokens: &[u32], targets: &[u32]) -> (f32, Vec<(String, Vec<f32>)>) {
    m.set_batch(tokens, targets);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    let grads = model::Model::optimized_params(m).unwrap().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, grads)
}

#[test]
fn the_tf32_tier_trains_like_the_fp32_reference() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let cfg = config();
    let init = weights(&cfg);
    let (want_loss, want) = loss_and_grads(&Qwen::new(cfg.clone(), 1, 12, &init));

    let shard = || qwen3::Shard::whole(cfg.n_layers as usize);
    let tf32 = Qwen::new_shard_gemm(cfg.clone(), 1, 12, &init, true, shard(), GemmPrecision::Auto);
    let (_, on_tf32) = tf32.gemm_report();
    let reference = Qwen::new_shard_gemm(cfg.clone(), 1, 12, &init, true, shard(), GemmPrecision::Fp32Reference);
    assert!(!reference.gemm_report().1, "the pinned reference tier must not report TF32");
    if !on_tf32 {
        brain_testutil::skip_unavailable(&format!("lora_gemm_precision: no TF32 tier on this device ({:?})", tf32.gemm_report().0));
        return;
    }
    let (loss, got) = loss_and_grads(&tf32);
    assert!((loss - want_loss).abs() <= 1e-3 * want_loss.abs().max(1.0), "loss {loss} vs fp32 {want_loss}");
    assert_eq!(got.len(), want.len());
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        assert!(cosine(g, w) > 0.999, "{name}: gradient cosine {} against the fp32 tier", cosine(g, w));
    }
}

/// Qwen3's head width (128) over a sequence long enough for the flash trio: the training forward then runs
/// its attention on fp16 tensor cores as well as its linears on TF32, and the loss and every adapter gradient
/// still agree with the exact fp32 tier (loss to 2e-3, gradient cosine above 0.999).
#[test]
fn the_default_tier_trains_a_qwen3_wide_head_like_the_fp32_reference() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let cfg = QwenConfig { vocab: 64, block_size: WIDE_T as u32, d_model: 256, n_heads: 4, n_kv_heads: 2, head_dim: 128, d_ff: 512, max_position_embeddings: WIDE_T as u32, lora: Some(LoraCfg::attn(4, 8.0)), ..QwenConfig::tiny() };
    let init = weights(&cfg);
    let tokens: Vec<u32> = (0..WIDE_T as u32).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..WIDE_T as u32).map(|i| (i * 7 + 10) % cfg.vocab).collect();
    let shard = || qwen3::Shard::whole(cfg.n_layers as usize);
    let reference = Qwen::new_shard_gemm(cfg.clone(), 1, WIDE_T as u32, &init, true, shard(), GemmPrecision::Fp32Reference);
    let tuned = Qwen::new_shard_gemm(cfg.clone(), 1, WIDE_T as u32, &init, true, shard(), GemmPrecision::Auto);
    if !tuned.gemm_report().1 {
        brain_testutil::skip_unavailable(&format!("lora_gemm_precision: no tensor-core tier on this device ({:?})", tuned.gemm_report().0));
        return;
    }
    let (want_loss, want) = loss_and_grads_on(&reference, &tokens, &targets);
    let (loss, got) = loss_and_grads_on(&tuned, &tokens, &targets);
    assert!((loss - want_loss).abs() <= 2e-3 * want_loss.abs().max(1.0), "loss {loss} vs fp32 {want_loss}");
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        assert!(cosine(g, w) > 0.999, "{name}: gradient cosine {} against the fp32 tier", cosine(g, w));
    }
}
