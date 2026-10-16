// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The trajectory gate for the tensor-core GEMM tier: 30 optimiser steps of the
//! tiny hybrid model on fixed data from one set of weights, once on the fp32
//! reference tier and once on the automatic tier. A drifted reduction or a
//! rounding that biases the gradient shows as a loss curve that parts company
//! with the reference over steps, which a one-step gradient comparison cannot
//! see. The curve must stay within 2% of the reference at every step, fall over
//! windows of five steps, and the trained adapters must point the same way.
//!
//! On a device without the library the automatic tier is the reference tier and
//! the comparison is exact; the reported path says which one ran.

mod common;

use common::*;
use gpu_core::provider::library::{GemmPath, GemmPrecision};
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const STEPS: u32 = 30;

/// Loss curve and final adapter weights of `STEPS` AdamW steps on `gpu`.
fn train(gpu: &Gpu, init: &std::collections::HashMap<String, Vec<f32>>, precision: GemmPrecision, gdn_segment: Option<u32>) -> (Vec<f32>, Vec<(String, Vec<f32>)>, GemmPath) {
    let opts = TrainOpts { base: TierPolicy::uniform(Dtype::BF16), recompute: true, gemm_precision: precision, gdn_segment_tokens: gdn_segment, ..TrainOpts::default() };
    let m = Qwen35::new_train_opts_on(gpu.share(), lora_config(), 1, T as u32, init, &opts);
    m.set_batch(&TOKENS, &TARGETS);
    let mut curve = Vec::new();
    for step in 1..=STEPS {
        m.zero_grads();
        curve.push(m.forward());
        m.backward();
        m.adamw_step(step, 1e-2, 0.0, Default::default(), Some(1.0), 1.0);
    }
    let weights = m.param_names().into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect();
    (curve, weights, m.gemm_path().clone())
}

#[test]
fn the_tensor_core_tier_follows_the_reference_loss_curve() {
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        eprintln!("train_tier_trajectory: no CUDA device on this box - nothing to run");
        return;
    };
    let init = weights(&lora_config(), 0.3);
    // Whole-sequence and 4-token GDN segments (the tiny model's chunk is 4).
    for segment in [None, Some(4)] {
        let (want, want_w, _) = train(&gpu, &init, GemmPrecision::Fp32Reference, segment);
        let (got, got_w, path) = train(&gpu, &init, GemmPrecision::Auto, segment);
        eprintln!("train_tier_trajectory: segment {segment:?}: {path:?}, loss {:.4} -> {:.4} (reference {:.4})", got[0], got[got.len() - 1], want[want.len() - 1]);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 0.02 * w.abs(), "segment {segment:?} step {}: loss {g} vs reference {w}", i + 1);
        }
        for window in got.windows(5) {
            assert!(window[4] < window[0], "segment {segment:?}: the loss did not fall over five steps: {window:?}");
        }
        let flat = |v: &[(String, Vec<f32>)]| v.iter().flat_map(|(_, x)| x.iter().copied()).collect::<Vec<f32>>();
        let cos = cosine(&flat(&got_w), &flat(&want_w));
        assert!(cos >= 0.99, "segment {segment:?}: the trained adapters have cosine {cos} to the reference");
    }
}
