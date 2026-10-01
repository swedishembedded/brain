// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements parameter-efficient fine-tuning on
// consumer GPUs for its clients. If your team needs expertise in training
// large models within a fixed memory budget then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A LoRA model whose frozen base is held in bf16 (half the bytes of the
//! fp32 base) trains the same adapters the fp32 build does. The base values
//! are bf16-representable, so the only difference between the two builds is
//! the storage tier of the base matmuls: the loss and every adapter gradient
//! agree, and a fresh adapter (B = 0) leaves the base model's loss exactly
//! as it was.

use std::collections::HashMap;

use data::rng::Rng;

use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

const TOKENS: [u32; 12] = [1, 5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2];
const TARGETS: [u32; 12] = [5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2, 1];

fn weights(cfg: &QwenConfig, adapter_b: f32) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 21);
    let mut rng = Rng::new(5);
    for (name, v) in w.iter_mut() {
        if name.ends_with(".lora_b") {
            v.iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * adapter_b);
        } else if !name.contains(".lora_") {
            v.iter_mut().for_each(|x| *x = round_bf16(*x));
        }
    }
    w
}

fn config() -> QwenConfig {
    QwenConfig { block_size: 12, lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() }
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
fn the_bf16_base_trains_the_adapters_the_fp32_base_does() {
    if gpu_disabled() {
        return;
    }
    let cfg = config();
    let init = weights(&cfg, 0.3);
    let (want_loss, want) = loss_and_grads(&Qwen::new(cfg.clone(), 1, 12, &init));
    let half = Qwen::new_lora_dt(cfg, 1, 12, &init, Dtype::BF16);
    if half.linear_dtype() != Some(Dtype::BF16) {
        return; // no bf16 storage path on this device
    }
    let (loss, got) = loss_and_grads(&half);
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
fn a_fresh_adapter_leaves_the_bf16_base_loss_unchanged() {
    if gpu_disabled() {
        return;
    }
    let cfg = config();
    let init = weights(&cfg, 0.0);
    let half = Qwen::new_lora_dt(cfg.clone(), 1, 12, &init, Dtype::BF16);
    if half.linear_dtype() != Some(Dtype::BF16) {
        return;
    }
    half.set_batch(&TOKENS, &TARGETS);
    let with_adapter = half.forward();
    let base_cfg = QwenConfig { lora: None, ..cfg };
    let base_init: HashMap<String, Vec<f32>> = init.iter().filter(|(n, _)| !n.contains(".lora_")).map(|(n, v)| (n.clone(), v.clone())).collect();
    let base = Qwen::new_shard_dt(base_cfg.clone(), 1, 12, &base_init, qwen3::Shard::whole(base_cfg.n_layers as usize), Dtype::BF16);
    base.set_batch(&TOKENS, &TARGETS);
    assert_eq!(with_adapter, base.forward(), "B = 0 adds exactly nothing to the bf16 base");
}

/// The token embedding and the LM head are frozen in a LoRA build, so a bf16
/// base holds them in bf16 too (half the bytes, and a 152k-token head then
/// fits one storage binding): the loss and every adapter gradient still agree
/// with the fp32 build, for a tied and for an untied head.
#[test]
fn the_head_and_embedding_are_held_in_bf16_and_train_the_same_adapters() {
    if gpu_disabled() {
        return;
    }
    for tie in [true, false] {
        let cfg = QwenConfig { tie_embeddings: tie, ..config() };
        let init = weights(&cfg, 0.3);
        let (want_loss, want) = loss_and_grads(&Qwen::new(cfg.clone(), 1, 12, &init));
        let half = Qwen::new_lora_dt(cfg, 1, 12, &init, Dtype::BF16);
        if half.linear_dtype() != Some(Dtype::BF16) {
            return;
        }
        assert_eq!(half.head_dtype(), Dtype::BF16, "tie={tie}: the head table is held in bf16");
        let (loss, got) = loss_and_grads(&half);
        assert!((loss - want_loss).abs() < 1e-4 * want_loss.abs().max(1.0), "tie={tie}: loss {loss} vs {want_loss}");
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            if w.iter().any(|x| *x != 0.0) {
                assert!(cosine(g, w) > 0.9999, "tie={tie} {name}: adapter gradient cosine {}", cosine(g, w));
            }
        }
        let host = half.read_weight(half.cfg.head_weight());
        assert_eq!(host, init[half.cfg.head_weight()], "tie={tie}: the packed table reads back as the values it was built from");
    }
}
