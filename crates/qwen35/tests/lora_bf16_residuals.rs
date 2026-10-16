// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The layer-boundary residual stream held as bf16: half the bytes of what
//! dominates training memory once a layer's activations are recomputed. The
//! stream is rounded to bf16 at every boundary (forward and backward alike),
//! so loss and adapter gradients stay close to the fp32-stream build's, and
//! the per-token memory of a deep stack drops.

mod common;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn build(init: &std::collections::HashMap<String, Vec<f32>>, bf16_residuals: bool, recompute: bool) -> Qwen35 {
    let opts = TrainOpts { bf16_residuals, recompute, ..opts(Dtype::BF16) };
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), lora_config(), 1, T as u32, init, &opts)
}

#[test]
fn a_bf16_residual_stream_trains_the_adapters_the_fp32_stream_does_to_bf16_precision() {
    let init = weights(&lora_config(), 0.3);
    let want = loss_and_grads(&build(&init, false, true));
    for recompute in [true, false] {
        let got = loss_and_grads(&build(&init, true, recompute));
        assert_ne!(got.0, want.0, "the residual stream is rounded, so the loss is not bitwise the fp32 stream's");
        // Four layers of bf16 rounding (2^-8 relative) on the stream.
        assert_same_training_signal(got, want.clone(), 2e-2, 0.99);
    }
}

#[test]
fn a_bf16_residual_stream_still_overfits_a_fixed_batch() {
    let init = weights(&lora_config(), 0.0);
    let m = build(&init, true, true);
    m.set_batch(&TOKENS, &TARGETS);
    let before = m.forward();
    for step in 1..=120 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 3e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    assert!(m.forward() < before * 0.8, "no learning from {before}");
}

fn deep() -> Qwen35Config {
    Qwen35Config { n_layers: 16, block_size: 512, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() }
}

#[test]
#[ignore = "child process helper, driven by bf16_residuals_halve_the_per_token_memory_of_the_boundary_stream"]
fn child_charged_bytes_per_residual_mode() {
    publish_ceiling();
    let cfg = deep();
    let init = qwen35::init::init_weights(&cfg, 5); // seed: only the shapes matter here
    for (name, bf16_residuals) in [("fp32", false), ("bf16", true)] {
        let at = |t: u32| {
            let opts = TrainOpts { bf16_residuals, recompute: true, ..TrainOpts::default() };
            Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), Qwen35Config { block_size: t, ..cfg.clone() }, 1, t, &init, &opts).gpu.charged_bytes()
        };
        mark(&format!("{name}_slope"), (at(512) - at(256)) / 256);
    }
}

#[test]
fn bf16_residuals_halve_the_per_token_memory_of_the_boundary_stream() {
    let m = measure_in_child("child_charged_bytes_per_residual_mode");
    let cfg = deep();
    // fp32: (L + 1) boundaries of d floats; bf16: the same count of half-size
    // packed buffers plus the two fp32 working buffers.
    let boundaries = cfg.n_layers as u64 + 1;
    let saved = 4 * boundaries * cfg.d_model as u64 - (2 * boundaries * cfg.d_model as u64 + 2 * 4 * cfg.d_model as u64);
    let (fp32, bf16) = (m["fp32_slope"], m["bf16_slope"]);
    assert!(fp32 > bf16 && fp32 - bf16 >= saved * 9 / 10, "per-token request fell from {fp32} to {bf16} B; the boundary stream should save {saved} B");
}
