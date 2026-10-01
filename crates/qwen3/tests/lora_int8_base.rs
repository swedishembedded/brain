// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements parameter-efficient fine-tuning on
// consumer GPUs for its clients. If your team needs expertise in training
// large models within a fixed memory budget then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A LoRA model whose frozen base is held as int8 weights (one byte per
//! weight and a scale per 32) reads them through the matrix kernels, which
//! decode each weight as they load it and multiply it with fp32 activations:
//! no activation is quantised. Built over weights that are exactly
//! representable in that format, the model is the fp32 model - the loss and
//! every adapter gradient agree - and a fresh adapter (B = 0) leaves the
//! base model's loss as it was.

use std::collections::HashMap;

use data::rng::Rng;
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

/// `n` tokens and the targets that follow them.
fn batch(n: usize) -> (Vec<u32>, Vec<u32>) {
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % 22 + 1).collect();
    let targets = (0..n).map(|i| tokens[(i + 1) % n]).collect();
    (tokens, targets)
}

/// A model shape every linear of which has a contraction width that is a
/// multiple of the quantisation group (32).
fn config(block: u32, d_model: u32, d_ff: u32) -> QwenConfig {
    QwenConfig { block_size: block, d_model, n_heads: 4, n_kv_heads: 2, head_dim: d_model / 4, d_ff, lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() }
}

/// Weights that survive the int8 round trip unchanged: each linear is
/// quantised and dequantised once, so building the int8 model from them
/// loses nothing further.
fn weights(cfg: &QwenConfig, adapter_b: f32) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 31);
    let mut rng = Rng::new(5);
    for (name, v) in w.iter_mut() {
        if name.ends_with(".lora_b") {
            v.iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * adapter_b);
        } else if qwen3::q8::Q8::is_i8_linear(name) {
            let k = if name.ends_with("down.weight") { cfg.d_ff } else if name.ends_with("wo.weight") { cfg.q_dim() } else { cfg.d_model } as usize;
            let n = v.len() / k;
            let (packed, scales) = model::int8::quantize_weight(v, n, k);
            *v = model::int8::dequantize_weight(&packed, &scales, n, k);
        }
    }
    w
}

fn loss_and_grads(m: &Qwen, tokens: &[u32], targets: &[u32]) -> (f32, Vec<(String, Vec<f32>)>) {
    m.set_batch(tokens, targets);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    let grads = model::Model::optimized_params(m).unwrap().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, grads)
}

/// The int8 build against the fp32 one on weights both hold exactly, at a shape
/// of `block` tokens (the tiled kernels take over from the plain ones once the
/// rows fill a 128-tile).
fn the_adapters_train_alike(block: u32, d_model: u32, d_ff: u32) {
    let cfg = config(block, d_model, d_ff);
    let init = weights(&cfg, 0.3);
    let (tokens, targets) = batch(block as usize);
    let (want_loss, want) = loss_and_grads(&Qwen::new(cfg.clone(), 1, block, &init), &tokens, &targets);
    let int8 = Qwen::new_lora_dt(cfg, 1, block, &init, Dtype::I8);
    if int8.linear_dtype() != Some(Dtype::I8) {
        return; // no int8 storage path on this device
    }
    let (loss, got) = loss_and_grads(&int8, &tokens, &targets);
    let used = |r: &gpu_core::cost::CostReport, k: &str| r.by_kernel.keys().chain(r.uncovered.keys()).any(|n| n.starts_with(k) && n.ends_with("#w=w8"));
    assert!(used(&int8.cost_fwd(), "matmul"), "the forward reads the int8 weights through the weight-only kernels");
    assert!(used(&int8.cost_bwd(), "matmul_dx"), "so does the input-gradient pass");
    assert!((loss - want_loss).abs() < 1e-4 * want_loss.abs().max(1.0), "loss {loss} vs {want_loss}");
    assert_eq!(got.len(), want.len());
    let mut nonzero = 0;
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        if w.iter().all(|x| *x == 0.0) {
            continue;
        }
        nonzero += 1;
        let c = cosine(g, w);
        assert!(c > 0.9999, "{name}: adapter gradient cosine {c}");
    }
    assert!(nonzero >= 4, "the adapters received gradient ({nonzero} tensors)");
}

#[test]
fn the_int8_base_trains_the_adapters_the_fp32_base_does() {
    if gpu_disabled() {
        return;
    }
    the_adapters_train_alike(12, 64, 128);
}

/// 128 rows: the register-tiled forward and input-gradient kernels.
#[test]
fn the_tiled_kernels_read_the_int8_base_the_same_way() {
    if gpu_disabled() {
        return;
    }
    the_adapters_train_alike(128, 128, 256);
}

#[test]
fn a_fresh_adapter_leaves_the_int8_base_loss_unchanged() {
    if gpu_disabled() {
        return;
    }
    let cfg = config(12, 64, 128);
    let (tokens, targets) = batch(12);
    let init = weights(&cfg, 0.0);
    let int8 = Qwen::new_lora_dt(cfg.clone(), 1, 12, &init, Dtype::I8);
    if int8.linear_dtype() != Some(Dtype::I8) {
        return;
    }
    int8.set_batch(&tokens, &targets);
    let with_adapter = int8.forward();
    let base_cfg = QwenConfig { lora: None, ..cfg };
    let base_init: HashMap<String, Vec<f32>> = init.iter().filter(|(n, _)| !n.contains(".lora_")).map(|(n, v)| (n.clone(), v.clone())).collect();
    let base = Qwen::new_shard_dt(base_cfg.clone(), 1, 12, &base_init, qwen3::Shard::whole(base_cfg.n_layers as usize), Dtype::I8);
    base.set_batch(&tokens, &targets);
    // The inference build quantises its activations; the training build does
    // not, so only the loose agreement of two different tiers is expected.
    let plain = base.forward();
    assert!((with_adapter - plain).abs() < 0.05 * plain.abs().max(1.0), "{with_adapter} vs {plain}");
}
