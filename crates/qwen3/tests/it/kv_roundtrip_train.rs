// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on the exact numerics they are served with,
// for its clients. If your team needs expertise in quantization-aware reinforcement learning against an inference
// engine then you can procure our services by sending an email to info@swedishembedded.com.

//! The trainer's attention can read its keys and values through an int8 or fp8 KV cache's encoding
//! ([`qwen3::Qwen::set_kv_roundtrip`]), so a rollout sampled by a server with that cache is scored with the numerics it was
//! sampled under. Gated on a tiny plain decoder on the CPU: the round trip changes what the forward computes (it is not the
//! identity) and the log-probabilities move by the size of the cache's rounding, the backward still differentiates the
//! adapters (the straight-through gradient is the unrounded one, so it matches the plain gradient to within the rounding,
//! it is neither dropped nor non-finite), and switching it off restores the plain forward exactly.

use data::rng::Rng;
use model::kv_tier::KvElement;
use model::{training, Batch, Model};
use qwen3::{Qwen, QwenConfig};

const T: usize = 12;

fn tiny() -> (Qwen, Vec<u32>, Vec<u32>) {
    let cfg = QwenConfig { lora: Some(qwen3::config::LoraCfg::new(2, 4.0, qwen3::finetune::default_lora_targets())), ..QwenConfig::tiny() };
    let init = <Qwen as Model>::init_weights(&cfg, 5);
    let m = Qwen::new(cfg, 1, T as u32, &init);
    let mut rng = Rng::new(9);
    for name in Model::param_names(&m).into_iter().filter(|n| n.contains(".lora_")) {
        let w: Vec<f32> = (0..Model::read_weight(&m, &name).len()).map(|_| 0.8 * (rng.next_f32() - 0.5)).collect();
        Model::write_weight(&m, &name, &w);
    }
    let tokens: Vec<u32> = vec![1, 2, 3, 9, 10, 11, 12, 4, 5, 6, 7, 8];
    let mut targets = vec![model::IGNORE; T];
    targets[4..T - 1].copy_from_slice(&tokens[5..T]);
    (m, tokens, targets)
}

fn log_probs_and_grads(m: &Qwen, tokens: &[u32], targets: &[u32]) -> (Vec<f32>, Vec<f32>) {
    Model::set_batch(m, Batch::Lm { tokens, targets });
    m.zero_grads();
    let _ = m.forward();
    let lp = training::objectives(m).batch_token_logprobs();
    m.backward();
    m.poll_wait();
    let grads: Vec<f32> = Model::param_names(m).iter().filter(|n| n.contains(".lora_")).flat_map(|n| Model::read_grad(m, n)).collect();
    (lp, grads)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn a_kv_round_trip_changes_the_forward_by_the_caches_rounding_and_leaves_the_gradient_straight_through() {
    let (mut m, tokens, targets) = tiny();
    let (plain_lp, plain_grads) = log_probs_and_grads(&m, &tokens, &targets);
    for element in [KvElement::Int8, KvElement::Fp8E4m3] {
        m.set_kv_roundtrip(Some(element)).unwrap();
        let (lp, grads) = log_probs_and_grads(&m, &tokens, &targets);
        let moved = lp.iter().zip(&plain_lp).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(moved > 1e-6, "{element:?}: the round trip changed nothing");
        assert!(moved < 0.5, "{element:?}: a rounding of a 127- or 448-level grid moved a log-probability by {moved}");
        assert!(grads.iter().all(|g| g.is_finite()), "{element:?}: the backward went non-finite");
        let cos = cosine(&grads, &plain_grads);
        assert!(cos > 0.97, "{element:?}: the straight-through gradient points away from the plain one (cosine {cos})");
        let norm_ratio = grads.iter().map(|g| f64::from(*g).powi(2)).sum::<f64>().sqrt() / plain_grads.iter().map(|g| f64::from(*g).powi(2)).sum::<f64>().sqrt();
        assert!((0.8..1.25).contains(&norm_ratio), "{element:?}: the gradient's norm is {norm_ratio} of the plain one");
    }
    m.set_kv_roundtrip(None).unwrap();
    let (back_lp, back_grads) = log_probs_and_grads(&m, &tokens, &targets);
    assert_eq!(back_lp, plain_lp, "switching it off restores the plain forward exactly");
    assert_eq!(back_grads, plain_grads);
    assert!(m.set_kv_roundtrip(Some(KvElement::Bf16)).is_err() && m.set_kv_roundtrip(Some(KvElement::F32)).is_err(), "the tiers that are not lossy here are refused by name");
}
