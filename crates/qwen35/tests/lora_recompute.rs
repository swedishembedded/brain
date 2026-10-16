// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Activation recompute: the trainer keeps only the layer-boundary residuals
//! through the forward pass and rebuilds each layer's activations inside the
//! backward. It must compute exactly what the keep-everything trainer does
//! (the recomputed forward is the same deterministic kernels over the same
//! inputs), through both mixer types and a GDN sequence of several chunks.

mod common;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn build(cfg: &Qwen35Config, init: &std::collections::HashMap<String, Vec<f32>>, base: Dtype, recompute: bool) -> Qwen35 {
    let opts = TrainOpts { base: TierPolicy::uniform(base), recompute, ..TrainOpts::default() };
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, T as u32, init, &opts)
}

#[test]
fn recompute_gives_the_same_loss_and_gradients_as_keeping_every_activation() {
    // t = 12 gives the GDN layers 3 chunks of 4: the cross-chunk recurrence is recomputed too.
    let cfg = lora_config();
    let init = weights(&cfg, 0.3);
    for base in [Dtype::F32, Dtype::BF16] {
        let (want_loss, want) = loss_and_grads(&build(&cfg, &init, base, false));
        let (loss, got) = loss_and_grads(&build(&cfg, &init, base, true));
        assert_eq!(loss, want_loss, "{base:?}: loss");
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            assert_eq!(g, w, "{base:?}: gradient of {name}");
        }
        assert!(want.iter().any(|(_, w)| w.iter().any(|x| *x != 0.0)), "the reference has gradient to compare");
    }
}

#[test]
fn recompute_keeps_no_per_layer_activations_across_the_forward() {
    let cfg = lora_config();
    let init = weights(&cfg, 0.3);
    for (recompute, kept) in [(false, cfg.n_layers as usize), (true, 0)] {
        let m = build(&cfg, &init, Dtype::BF16, recompute);
        m.set_batch(&TOKENS, &TARGETS);
        m.forward();
        assert_eq!(m.retained_layer_activations(), kept, "recompute={recompute}");
        m.backward();
        assert_eq!(m.retained_layer_activations(), 0, "backward consumes what the forward kept");
    }
}

#[test]
fn recompute_can_be_switched_between_steps() {
    let cfg = lora_config();
    let init = weights(&cfg, 0.3);
    let m = build(&cfg, &init, Dtype::F32, false);
    let want = loss_and_grads(&m);
    m.set_recompute(true);
    let got = loss_and_grads(&m);
    assert_eq!(got.0, want.0);
    assert_eq!(got.1, want.1);
}

/// The MLP is row-wise, so it can run in token chunks whose activations are
/// rebuilt when their gradient is taken: loss and gradients are those of the
/// whole-sequence MLP (the adapter gradients are summed chunk by chunk, so to
/// rounding rather than bitwise). 5 rows does not divide the 12 tokens.
#[test]
fn a_chunked_mlp_gives_the_loss_and_gradients_of_the_whole_one() {
    let cfg = lora_config();
    let init = weights(&cfg, 0.3);
    for base in [Dtype::F32, Dtype::BF16] {
        let whole = loss_and_grads(&build(&cfg, &init, base, true));
        let opts = TrainOpts { base: TierPolicy::uniform(base), recompute: true, mlp_rows: Some(5), ..TrainOpts::default() };
        let chunked = Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, T as u32, &init, &opts);
        assert_same_training_signal(loss_and_grads(&chunked), whole, 1e-6, 0.999999);
    }
}
