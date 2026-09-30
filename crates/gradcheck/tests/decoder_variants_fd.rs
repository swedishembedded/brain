// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gradient checks for every decoder variant the qwen3
//! decoder serves and trains: Llama (full multi-head and grouped-query
//! attention, no QK-norm, no bias, an untied head), Qwen2 (q/k/v bias, no
//! QK-norm, untied), and a non-default RMSNorm epsilon. Each variant is
//! checked fully trainable and as a LoRA fine-tune over all seven
//! projections; the parameters whose gradient is easiest to get subtly wrong
//! (biases, norm gains, a tied embedding table that collects the head's
//! gradient too) are also checked element by element.
//!
//! The base Qwen3 config, the Qwen2 variant with a tied head and the scaled
//! RoPE tables are checked in the crate's own unit tests.

use gradcheck::{directional_check, elementwise_check, Report};
use model::Model;
use qwen3::{LoraCfg, Qwen, QwenConfig};

/// The workspace gradient gate: every tensor within tolerance, and no
/// gradient that comes back exactly zero where the loss clearly moves.
fn assert_gate(report: &Report, what: &str) {
    report.print();
    let fails = report.failures(4e-3, 8e-2);
    assert!(fails.is_empty(), "{what}: {:?}", fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
    let dead = report.dead_gradients();
    assert!(dead.is_empty(), "{what}: dead gradients {:?}", dead.iter().map(|c| (&c.param, c.numeric)).collect::<Vec<_>>());
}

fn llama_mha() -> QwenConfig {
    QwenConfig { n_heads: 2, n_kv_heads: 2, head_dim: 8, qk_norm: false, attn_bias: false, tie_embeddings: false, ..QwenConfig::tiny() }
}

fn llama_gqa() -> QwenConfig {
    QwenConfig { qk_norm: false, attn_bias: false, tie_embeddings: false, ..QwenConfig::tiny() }
}

fn qwen2_gqa_untied() -> QwenConfig {
    QwenConfig { qk_norm: false, attn_bias: true, tie_embeddings: false, ..QwenConfig::tiny() }
}

/// A large epsilon, so a kernel that ignores the configured one (and uses a
/// hard-coded 1e-6) moves both the loss and the gradient visibly.
fn rms_eps() -> QwenConfig {
    QwenConfig { rms_eps: 1e-2, ..QwenConfig::tiny() }
}

const VARIANTS: [(&str, fn() -> QwenConfig); 4] =
    [("llama (MHA, untied)", llama_mha), ("llama (GQA, untied)", llama_gqa), ("qwen2 (GQA, bias, untied)", qwen2_gqa_untied), ("rms_eps 1e-2", rms_eps)];

/// The model at `cfg` with a fixed batch where every position is supervised.
fn built(cfg: QwenConfig, seed: u64) -> Qwen {
    let init = qwen3::init_weights(&cfg, seed);
    let model = Qwen::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    model
}

#[test]
fn every_decoder_variant_matches_finite_differences() {
    for (name, cfg) in VARIANTS {
        let model = built(cfg(), 7);
        assert_gate(&directional_check(&model, 5e-3, 4, 7 ^ 0x1234), name);
    }
}

/// The LoRA twin of each variant: adapters on all seven projections, moved
/// off their zero-`B` start by a few steps so both factors carry gradient.
///
/// Those steps leave the loss sharply curved along the adapters: on the
/// Qwen2 variant the directional difference for `wv.lora_b` misses by 0.150
/// at a 5e-3 step, 0.039 at 2.5e-3 and 0.010 at 1.25e-3 while the analytic
/// value holds still - quadratic truncation error, not a wrong gradient. The
/// check therefore steps 1.25e-3, which f32 resolves with room to spare.
#[test]
fn every_decoder_variant_lora_matches_finite_differences() {
    for (name, cfg) in VARIANTS {
        let targets = qwen3::finetune::LORA_TARGETS.iter().map(|t| t.to_string()).collect();
        let model = built(QwenConfig { lora: Some(LoraCfg { rank: 2, alpha: 4.0, targets }), ..cfg() }, 7);
        for step in 1..=5 {
            model.zero_grads();
            model.forward();
            model.backward();
            model.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
            model.poll_wait();
        }
        assert_gate(&directional_check(&model, 1.25e-3, 4, 7 ^ 0x1234), &format!("{name}, LoRA"));
    }
}

/// Element by element: the q/k/v biases and a layer's norm gains of the
/// Qwen2 variant, the untied head of a Llama, and the tied table of the
/// base config, which collects both the embedding's and the head's gradient.
#[test]
fn biases_norms_and_the_tied_table_match_element_by_element() {
    let cases: [(fn() -> QwenConfig, &[&str]); 3] = [
        (qwen2_gqa_untied, &["blocks.0.attn.wq.bias", "blocks.0.attn.wk.bias", "blocks.1.attn.wv.bias", "blocks.0.ln1.weight", "blocks.1.ln2.weight", "norm.weight"]),
        (llama_mha, &["lm_head.weight"]),
        (QwenConfig::tiny, &["tok.weight"]),
    ];
    for (cfg, names) in cases {
        let model = built(cfg(), 11);
        let params = model.param_names();
        for name in names {
            assert!(params.iter().any(|p| p == name), "{name} is not a parameter of this config: {params:?}");
            assert_gate(&elementwise_check(&model, name, 5e-3), name);
        }
    }
}
