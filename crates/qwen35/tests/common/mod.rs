// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Helpers shared by the LoRA-on-a-narrow-base tests: a tiny config that has
//! both mixer types and every targetable leaf, bf16-representable base
//! weights, and a loss+gradient reader.

#![allow(dead_code)]

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{Qwen35, TrainOpts};

pub const T: usize = 12;
pub const TOKENS: [u32; T] = [1, 5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2];
pub const TARGETS: [u32; T] = [5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2, 1];

pub fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

/// 4 layers (3 GDN + 1 GQA), all 12 LoRA-targetable leaves, rank 3 (coprime
/// with every model dimension so a transposed adapter shape cannot hide).
pub fn lora_config() -> Qwen35Config {
    Qwen35Config { block_size: T as u32, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() }
}

/// Base weights rounded to bf16 (so an fp32 and a bf16 build multiply by the
/// same values) and adapters with a non-zero `B` of magnitude `adapter_b`.
pub fn weights(cfg: &Qwen35Config, adapter_b: f32) -> HashMap<String, Vec<f32>> {
    // seed: any fixed init works; the test only needs a reproducible model
    let mut w = qwen35::init::init_weights(cfg, 21);
    let mut rng = Rng::random();
    for (name, v) in w.iter_mut() {
        if name.ends_with(".lora_b") {
            v.iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * adapter_b);
        } else if !name.contains(".lora_") {
            v.iter_mut().for_each(|x| *x = round_bf16(*x));
        }
    }
    w
}

pub fn opts(base: Dtype) -> TrainOpts {
    TrainOpts { base: TierPolicy::uniform(base), ..TrainOpts::default() }
}

pub fn loss_and_grads(m: &Qwen35) -> (f32, Vec<(String, Vec<f32>)>) {
    m.set_batch(&TOKENS, &TARGETS);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    let grads = m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, grads)
}

/// Assert `got` agrees with `want` on loss (relative `loss_tol`) and on every
/// adapter gradient that is non-zero in `want` (cosine above `min_cos`).
pub fn assert_same_training_signal(got: (f32, Vec<(String, Vec<f32>)>), want: (f32, Vec<(String, Vec<f32>)>), loss_tol: f32, min_cos: f64) {
    let ((loss, got), (want_loss, want)) = (got, want);
    assert!((loss - want_loss).abs() < loss_tol * want_loss.abs().max(1.0), "loss {loss} vs {want_loss}");
    assert_eq!(got.len(), want.len());
    let mut nonzero = 0;
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        if w.iter().all(|x| *x == 0.0) {
            assert!(g.iter().all(|x| *x == 0.0), "{name}: gradient where the reference has none");
            continue;
        }
        nonzero += 1;
        let c = cosine(g, w);
        assert!(c > min_cos, "{name}: adapter gradient cosine {c}");
    }
    assert!(nonzero >= got.len() / 2, "the adapters received gradient ({nonzero} of {} tensors)", got.len());
}
