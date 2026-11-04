// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of encoder models for its clients. If your team needs
// expertise in training long-context encoders and decision heads within a fixed device budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Activation recompute for the ModernBERT trunk and Laya's decision head, held to the contract
//! `model::saved::check_recompute` states: a trainer whose layers share one set of intermediates and rebuild a
//! layer's from its input boundary before its backward gives the objective's gradients of the trainer that keeps
//! every layer, bit for bit, holding fewer device bytes. Narrow formats and going back to keeping are refused by
//! name. One test in its own binary, so nothing else allocates while the device bytes are compared.

use modernbert::config::ModernBertConfig;
use modernbert::kern::PIPELINES;
use modernbert::laya::{tensor_manifest, LayaConfig, LayaHead};
use modernbert::model::ModernBert;
use model::saved::{SavedFormat, SavedPolicy};

/// Ragged spans past the local window, so both attention rungs and the no-norm layer 0 run.
const SPANS: &[(u32, u32)] = &[(0, 9), (9, 6), (15, 4)];
/// One question per span, each of a different type; absolute packed rows of the option markers grouped per question.
const QTYPE: &[u32] = &[0, 1, 2];
const MARKER_ROWS: &[u32] = &[1, 4, 7, 10, 13, 17];
const ARITY: &[usize] = &[3, 2, 1];

/// The gradient bits of every parameter, sorted by name, and the device bytes the trainer holds after the policy
/// is set.
type Measured = (Vec<(String, Vec<u32>)>, u64);

fn trunk_config() -> ModernBertConfig {
    let mut cfg = ModernBertConfig::tiny();
    cfg.n_layers = 6;
    cfg.layer_types = (0..6).map(|l| if l % 2 == 0 { modernbert::config::LayerAttn::Full } else { modernbert::config::LayerAttn::Local }).collect();
    cfg
}

fn rows() -> u32 {
    SPANS.iter().map(|&(_, l)| l).sum()
}

fn max_span() -> u32 {
    SPANS.iter().map(|&(_, l)| l).max().unwrap()
}

fn trunk_step(init: &std::collections::HashMap<String, Vec<f32>>, policy: SavedPolicy) -> Measured {
    let cfg = trunk_config();
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let mut m = ModernBert::new_train_on(gpu.share(), cfg.clone(), rows(), max_span(), init);
    m.set_saved_policy(policy).unwrap();
    let held = gpu.memory_usage().unwrap().live - before;
    let ids: Vec<u32> = (0..rows()).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    m.set_batch(&ids, SPANS);
    m.prepare_reverse();
    let seed: Vec<f32> = (0..rows() * cfg.d_model).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
    m.gpu.write_f32(m.seed_buf(), &seed);
    m.zero_grads();
    m.forward();
    m.backward_seeded();
    m.poll_wait();
    let mut names: Vec<String> = cfg.tensor_manifest().into_iter().map(|(n, _)| n).collect();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), m.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    (grads, held)
}

fn head_step(trunk_init: &std::collections::HashMap<String, Vec<f32>>, head_init: &std::collections::HashMap<String, Vec<f32>>, policy: SavedPolicy) -> Measured {
    let cfg = trunk_config();
    let laya = LayaConfig::new(cfg.d_model);
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut enc = ModernBert::new_train_on(gpu.share(), cfg.clone(), rows(), max_span(), trunk_init);
    let before = gpu.memory_usage().unwrap().live;
    let mut head = LayaHead::new_train_on(gpu.share(), laya.clone(), rows(), max_span(), MARKER_ROWS.len() as u32, SPANS.len() as u32, head_init);
    head.set_saved_policy(policy).unwrap();
    let held = gpu.memory_usage().unwrap().live - before;
    let ids: Vec<u32> = (0..rows()).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    enc.set_batch(&ids, SPANS);
    enc.prepare_reverse();
    head.set_call_train(enc.hidden_buf(), enc.seed_buf(), SPANS, QTYPE, MARKER_ROWS, ARITY);
    head.zero_grads();
    enc.forward();
    enc.poll_wait();
    head.forward();
    let d_logits: Vec<f32> = (0..MARKER_ROWS.len()).map(|i| 0.2 - 0.1 * i as f32).collect();
    let d_act: Vec<f32> = (0..SPANS.len() * laya.n_act as usize).map(|i| 0.05 * (i as f32 - 3.0)).collect();
    head.backward(&d_logits, &d_act);
    head.poll_wait();
    let mut names: Vec<String> = tensor_manifest(&laya).into_iter().map(|(n, _)| n).collect();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), head.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    (grads, held)
}

fn assert_same(tag: &str, keep: &Measured, recompute: &Measured) {
    assert!(keep.0.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "{tag}: no parameter received a gradient");
    assert_eq!(keep.0.len(), recompute.0.len());
    for ((n, a), (_, b)) in keep.0.iter().zip(&recompute.0) {
        assert!(a == b, "{tag}: {n}: gradient differs recomputing");
    }
    assert!(recompute.1 < keep.1, "{tag}: recomputing holds {} device bytes, keeping {}", recompute.1, keep.1);
}

#[test]
fn modernbert_recomputes_its_layers_exactly() {
    let seed = data::rng::random_seed();
    let trunk_init = modernbert::init::init_weights(&trunk_config(), seed);
    let head_init = modernbert::init::init_weights_laya(&LayaConfig::new(trunk_config().d_model), seed ^ 1);

    assert_same("trunk", &trunk_step(&trunk_init, SavedPolicy::Keep), &trunk_step(&trunk_init, SavedPolicy::Recompute));
    assert_same("head", &head_step(&trunk_init, &head_init, SavedPolicy::Keep), &head_step(&trunk_init, &head_init, SavedPolicy::Recompute));

    // Narrow saved formats do not apply, and a recomputing trainer does not go back to keeping.
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut m = ModernBert::new_train_on(gpu.share(), trunk_config(), rows(), max_span(), &trunk_init);
    assert!(m.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("modernbert"));
    m.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(m.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("modernbert"));
    let mut head = LayaHead::new_train_on(gpu.share(), LayaConfig::new(trunk_config().d_model), rows(), max_span(), MARKER_ROWS.len() as u32, SPANS.len() as u32, &head_init);
    assert!(head.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Fp8)).unwrap_err().contains("laya"));
    head.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(head.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("laya"));
}
