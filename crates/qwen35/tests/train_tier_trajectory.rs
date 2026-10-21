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
//! see. The first step must agree tightly, the curve must stay within a measured
//! band of the reference at every step and at the last one, the loss must fall
//! over the run, and the trained adapters must point the same way.
//!
//! On a device without the library the automatic tier is the reference tier and
//! the comparison is exact; the reported path says which one ran.

mod common;

use common::*;
use gpu_core::provider::library::{GemmPath, GemmPrecision};
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use model::saved::SavedPolicy;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const STEPS: u32 = 30;

/// What a training run reports: its loss curve, the final adapter weights and the GEMM path taken.
type TrainOutcome = (Vec<f32>, Vec<(String, Vec<f32>)>, GemmPath);

/// Loss curve and final adapter weights of `STEPS` AdamW steps on `gpu`.
fn train(gpu: &Gpu, init: &std::collections::HashMap<String, Vec<f32>>, precision: GemmPrecision, gdn_segment: Option<u32>) -> TrainOutcome {
    let opts = TrainOpts { base: TierPolicy::uniform(Dtype::BF16), saved: SavedPolicy::Recompute, gemm_precision: precision, gdn_segment_tokens: gdn_segment, ..TrainOpts::default() };
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
        // The first step runs no update yet, so it measures the tensor-core forward alone, and agrees to rounding. Later
        // steps do not stay that close: AdamW turns the sign of every near-zero gradient component into a full-size
        // step, so bf16 noise in those components moves the trajectory, and the deviation of the loss curve is
        // heavy-tailed. The bounds below sit above the worst of 48 (seed, segment) runs on a GH200 (the test prints each
        // run's deviations under --nocapture) and far below what a wrong gradient produces.
        let dev = |g: &f32, w: &f32| (g - w).abs() / w.abs();
        let worst = got.iter().zip(&want).map(|(g, w)| dev(g, w)).fold(0.0f32, f32::max);
        let (first, last) = (dev(&got[0], &want[0]), dev(&got[got.len() - 1], &want[want.len() - 1]));
        eprintln!("train_tier_trajectory: segment {segment:?}: first-step deviation {first:.5}, worst per-step deviation {worst:.5}, last-step deviation {last:.5}");
        assert!(first <= 1e-3, "segment {segment:?}: the first step's loss differs by {first} from the reference");
        assert!(worst <= 0.10, "segment {segment:?}: the loss curve strays {worst} from the reference");
        assert!(last <= 0.03, "segment {segment:?}: the final loss differs by {last} from the reference");
        // The curve is not monotone (an update can overshoot for a few steps); what must hold is that training
        // learns: the loss fell by at least a fifth over the run in every measurement.
        assert!(got[got.len() - 1] < 0.9 * got[0], "segment {segment:?}: the loss did not fall over the run: {got:?}");
        let flat = |v: &[(String, Vec<f32>)]| v.iter().flat_map(|(_, x)| x.iter().copied()).collect::<Vec<f32>>();
        let cos = cosine(&flat(&got_w), &flat(&want_w));
        eprintln!("train_tier_trajectory: segment {segment:?}: adapter cosine {cos:.6}");
        // AdamW turns the sign of every near-zero gradient component into a full-size step, so bf16 tensor-core
        // noise in those components moves the trained adapters even when the loss curve agrees: over 14 random
        // seeds on a GH200 the cosine was 0.987-0.9993 (mean 0.9965) and a fixed
        // 0.99 failed one run in thirteen. A wrong gradient lands far below this (the loss-curve check above is the
        // primary guard), so the bound sits under the observed minimum.
        assert!(cos >= 0.98, "segment {segment:?}: the trained adapters have cosine {cos} to the reference");
    }
}
