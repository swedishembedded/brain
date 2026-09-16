// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for Florence-2's BART text encoder-decoder training
//! graph (`gradcheck::florence2`) - the M6 (LoRA + full fine-tune training)
//! milestone.

use gradcheck::Report;

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn gate(report: Report, what: &str) {
    report.print();
    let fails = report.failures(ATOL, RTOL);
    assert!(
        fails.is_empty(),
        "{what}: gradient check failed for {:?}",
        fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
    );
    let dead = report.dead_gradients();
    assert!(dead.is_empty(), "{what}: dead (identically zero) gradients for {:?}", dead.iter().map(|c| &c.param).collect::<Vec<_>>());
}

/// Full fine-tune: every base tensor (both towers' attention, FFN,
/// LayerNorms, position tables, and the tied `shared.weight`/
/// `final_logits_bias`) is trainable.
#[test]
fn florence2_full_finetune_analytic_grads_match_finite_differences() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_florence2: MOE_SKIP_GPU_TESTS set");
        return;
    }
    gate(gradcheck::check_florence2(7), "florence2 full fine-tune");
}

/// LoRA: base attention/FFN weights frozen, only the fresh `.lora_a`/
/// `.lora_b` adapters (and the untouched LayerNorms/embeddings, which stay
/// `Role::Frozen` too) are trainable.
#[test]
fn florence2_lora_analytic_grads_match_finite_differences() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_florence2_lora: MOE_SKIP_GPU_TESTS set");
        return;
    }
    gate(gradcheck::check_florence2_lora(11), "florence2 LoRA");
}
