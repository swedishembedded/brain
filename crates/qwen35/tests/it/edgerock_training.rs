// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

//! EdgeRock training behaviour that finite differences cannot state: a batch of two
//! different sequences is exactly two single-sequence batches (the memory's kernels,
//! the rope tables and the banks never cross a sequence boundary), a real training
//! loop reduces the loss, and the memory is load-bearing in it.

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen35::config::{AnchorMemoryCfg, Qwen35Config};
use model::saved::SavedPolicy;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const T: u32 = 12;

fn cfg() -> Qwen35Config {
    Qwen35Config {
        n_heads: 4,
        n_kv_heads: 2,
        swa_window: Some(5),
        anchor_memory: Some(AnchorMemoryCfg { capacity: 3, commit_chunk: 2, admission_rank: 4, writer_intermediate: 8 }),
        ..Qwen35Config::tiny()
    }
}

/// `init_weights` with the memory (and the attention beside it) drawn wide, so it carries signal.
fn weights(cfg: &Qwen35Config) -> HashMap<String, Vec<f32>> {
    let mut w = qwen35::init::init_weights(cfg, 11);
    let mut rng = data::rng::Lcg::new(0x3d0f_1234); // seed: fixed so a failure reproduces
    for (name, v) in w.iter_mut() {
        if (name.contains(".memory.") && !name.ends_with("_norm.weight")) || name == "tok.weight" || name.starts_with("blocks.3.self_attn.") && name.ends_with("_proj.weight") {
            v.iter_mut().for_each(|x| *x = rng.scaled(if name.ends_with("admission_up.weight") { 2.0 } else { 0.4 }));
        }
    }
    w
}

fn batch(seqs: u32, offset: usize) -> (Vec<u32>, Vec<u32>) {
    let n = (T * seqs) as usize;
    let x = (0..n).map(|i| ((i * 5 + 1 + (i / T as usize + offset) * 7) % 29) as u32).collect();
    let y = (0..n).map(|i| ((i * 5 + 2 + (i / T as usize + offset) * 7) % 29) as u32).collect();
    (x, y)
}

/// The fp32 kernels whatever the device would pick by default: these tests compare gradients and losses of runs
/// that differ in shape, which TF32 products would blur.
fn fp32() -> TrainOpts {
    TrainOpts { gemm_precision: gpu_core::provider::library::GemmPrecision::Fp32Reference, ..TrainOpts::default() }
}

fn model(cfg: &Qwen35Config, init: &HashMap<String, Vec<f32>>, b: u32, x: &[u32], y: &[u32]) -> Qwen35 {
    let m = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg.clone(), b, T, init, &fp32());
    m.set_batch(x, y);
    m
}

#[test]
fn a_batch_of_two_sequences_is_two_single_sequence_batches() {
    let cfg = cfg();
    let init = weights(&cfg);
    let ((x0, y0), (x1, y1)) = (batch(1, 0), batch(1, 1));
    let (x2, y2) = ([x0.clone(), x1.clone()].concat(), [y0.clone(), y1.clone()].concat());

    let both = model(&cfg, &init, 2, &x2, &y2);
    let (first, second) = (model(&cfg, &init, 1, &x0, &y0), model(&cfg, &init, 1, &x1, &y1));
    let loss_both = both.forward();
    let (loss_a, loss_b) = (first.forward(), second.forward());
    assert!((loss_both - (loss_a + loss_b) / 2.0).abs() < 1e-5, "batched loss {loss_both} vs mean of singles {}", (loss_a + loss_b) / 2.0);
    for m in [&both, &first, &second] {
        m.zero_grads();
    }
    both.forward();
    both.backward();
    first.forward();
    first.backward();
    second.forward();
    second.backward();

    // Every sequence has as many targets, so the batch gradient is the mean of the two.
    let mut compared = 0;
    for name in ["blocks.3.memory.q_proj.weight", "blocks.3.memory.k_proj.weight", "blocks.3.memory.v_proj.weight", "blocks.3.memory.o_proj.weight", "blocks.3.memory.write_up_proj.weight", "blocks.3.memory.read_scale", "blocks.3.memory.read_gate_weight", "blocks.3.memory.input_norm.weight", "blocks.3.self_attn.q_proj.weight", "tok.weight"] {
        let (gb, ga, gc) = (both.read_grad(name), first.read_grad(name), second.read_grad(name));
        let scale = gb.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = gb.iter().zip(ga.iter().zip(&gc)).fold(0f32, |m, (b, (a, c))| m.max((b - (a + c) / 2.0).abs()));
        assert!(worst < 2e-4 * scale + 1e-7, "{name}: batched gradient differs from the mean of the singles by {worst} (scale {scale})");
        compared += usize::from(scale > 1e-5);
    }
    assert!(compared >= 8, "most compared gradients must be live, got {compared}");
}

#[test]
fn training_reduces_the_loss_and_the_memory_is_load_bearing() {
    let cfg = cfg();
    let init = weights(&cfg);
    let (x, y) = batch(2, 3);
    let m = model(&cfg, &init, 2, &x, &y);
    let first = m.forward();
    for step in 1..=60 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 3e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    let trained = m.forward();
    assert!(trained < 0.6 * first, "60 AdamW steps on a fixed batch must cut the loss by 40 percent: {first} -> {trained}");

    // The trained memory matters: silencing its read gate (tanh(0) = 0) moves the loss.
    let scale = m.read_weight("blocks.3.memory.read_scale");
    assert!(scale.iter().any(|v| v.abs() > 1e-3), "training moved the read gate off zero or it started there");
    m.write_weight("blocks.3.memory.read_scale", &vec![0.0; scale.len()]);
    let silenced = m.forward();
    assert!((silenced - trained).abs() > 1e-3, "zeroing read_scale left the loss at {trained} -> {silenced}: the memory is not load-bearing");
}

#[test]
fn the_admission_scorer_is_never_optimised() {
    let cfg = cfg();
    let init = weights(&cfg);
    let (x, y) = batch(1, 0);
    let m = model(&cfg, &init, 1, &x, &y);
    let before = m.read_weight("blocks.3.memory.admission_down.weight");
    for step in 1..=3 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 3e-2, 0.1, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    assert_eq!(m.read_weight("blocks.3.memory.admission_down.weight"), before, "no gradient reaches a discrete selection, and weight decay must not either");
    assert!(!m.param_names().iter().any(|n| n.contains(".memory.admission_")));
}

/// The saved-activation policy: rebuilding every layer's activations (the memory's included) from its input in the
/// backward gives the gradients the kept activations give.
#[test]
fn recomputing_the_memory_in_the_backward_gives_the_same_gradients() {
    let cfg = cfg();
    let init = weights(&cfg);
    let (x, y) = batch(2, 5);
    let kept = model(&cfg, &init, 2, &x, &y);
    let recomputed = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg.clone(), 2, T, &init, &TrainOpts { saved: SavedPolicy::Recompute, ..fp32() });
    recomputed.set_batch(&x, &y);
    for m in [&kept, &recomputed] {
        m.zero_grads();
        m.forward();
        m.backward();
    }
    let mut live = 0;
    for name in ["blocks.3.memory.q_proj.weight", "blocks.3.memory.v_proj.weight", "blocks.3.memory.write_gate_proj.weight", "blocks.3.memory.read_gate_weight", "blocks.3.memory.q_norm.weight", "blocks.3.self_attn.o_proj.weight", "blocks.1.ln1.weight"] {
        let (a, b) = (kept.read_grad(name), recomputed.read_grad(name));
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(&b).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        assert!(worst < 1e-4 * scale + 1e-8, "{name}: recomputed gradient differs from the kept one by {worst} (scale {scale})");
        live += usize::from(scale > 1e-6);
    }
    assert!(live >= 6, "the comparison must be over live gradients, got {live}");
}
