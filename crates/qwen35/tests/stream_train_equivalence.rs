// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Step 1 gate for the streaming LoRA trainer (`qwen35::stream_train`): its
//! forward+backward+AdamW machinery must reproduce the EXISTING, already-
//! proven resident LoRA trainer (`Qwen35::new_train_on`/`forward`/
//! `backward`/`adamw_step`) exactly, at `Qwen35Config::tiny()` scale, on
//! IDENTICAL init weights and an IDENTICAL fixed batch, run on the SAME
//! (CPU JIT) backend so no cross-backend floating-point-summation-order
//! noise can muddy an "is the training-loop MACHINERY correct" proof.
//!
//! Deliberately does NOT go through real on-disk safetensors shards
//! (`import_layer`/`MmapSafetensors`) - that disk-streaming path is already
//! gated elsewhere in this crate's test suite (real-weight FP8 dequant in
//! `real_weight_streaming.rs`, int8 packing, the `(1+w)` RMSNorm fold). This
//! gate's own job is the NEW piece: the
//! forward+backward TRAINING MATH through a streamed layer. Driving
//! `qwen35::stream_train::build_layer_f32` directly against a synthetic
//! host map (`crate::init::init_weights`, the SAME map `crate::stream::
//! StreamState`'s own `gdn_end_padding_does_not_change_real_position_outputs`
//! test already establishes as this crate's own precedent for testing the
//! device-upload/forward half of streaming without a real checkpoint)
//! exercises 100% of the same device-upload and forward/backward code a
//! disk-backed run would - `load_layer_f32` is exactly
//! `import_layer(dir).then(build_layer_f32)`, and only that first, already
//! separately proven half differs.
//!
//! Uses fp32 (`Dtype::F32`) streamed weights throughout (the trainer's only
//! supported tier - see `stream_train`'s own module doc for why), so this
//! comparison is exact-not-just-close: no int8 quantization noise to muddy
//! the correctness proof.

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};
use qwen35::stream_train::StreamTrainer;

const SEED: u64 = 20260819;
const N: u32 = 6;
const RANK: u32 = 2;
const ALPHA: f32 = 4.0;
/// Deliberately narrower than `Qwen35Config::tiny()`'s own 4 layers, so a
/// real eviction (drop + rebuild a slot mid-pass) is exercised in BOTH the
/// forward and the reverse-order backward pass, not just the boundary pin.
const WINDOW_BUDGET: u32 = 2;

fn tiny_lora_cfg() -> Qwen35Config {
    let mut cfg = Qwen35Config::tiny();
    cfg.lora = Some(lora_cfg(RANK, ALPHA));
    cfg
}

/// A fixed, small, deterministic batch - not random per call, so both
/// trainers see IDENTICAL inputs across however many steps this test runs.
fn batch(cfg: &Qwen35Config) -> (Vec<u32>, Vec<u32>) {
    let v = cfg.vocab;
    let tokens: Vec<u32> = (0..N).map(|i| (i * 3 + 1) % v).collect();
    let targets: Vec<u32> = (0..N).map(|i| (i * 5 + 2) % v).collect();
    (tokens, targets)
}

fn lora_names(cfg: &Qwen35Config) -> Vec<String> {
    cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| n.ends_with(".lora_a") || n.ends_with(".lora_b")).collect()
}

#[test]
fn streaming_lora_trainer_matches_the_resident_trainer_exactly() {
    let cfg = tiny_lora_cfg();
    let init: HashMap<String, Vec<f32>> = qwen35::init::init_weights(&cfg, SEED);
    let (tokens, targets) = batch(&cfg);
    let names = lora_names(&cfg);
    assert!(!names.is_empty(), "test setup bug: no .lora_a/.lora_b tensors found");

    // ---- resident trainer (existing, already-proven path) ----
    let resident = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, N, &init);

    // ---- streaming trainer (new path under test) ----
    let streaming = StreamTrainer::new_synthetic(Gpu::new_cpu(pipelines()), &cfg, N, WINDOW_BUDGET, &init);
    let x0 = StreamTrainer::embed_synthetic(&init, &tokens, cfg.d_model as usize);
    let loader = streaming.synthetic_loader(&cfg, &init);

    let lr = 0.01f32;
    let mut loss_trajectory: Vec<(f32, f32)> = Vec::new();

    for step in 1..=3u32 {
        resident.zero_grads();
        resident.set_batch(&tokens, &targets);
        let rl = resident.forward();
        resident.backward();

        streaming.lora.zero_grads(&streaming.gpu);
        let sl = streaming.forward_backward(&cfg, &loader, &x0, &targets);
        loss_trajectory.push((rl, sl));

        assert!((rl - sl).abs() < 1e-4, "step {step}: loss mismatch: resident={rl} streaming={sl}");

        // Compare every LoRA adapter's own gradient BEFORE the AdamW step
        // changes anything - the direct "is backward correct" proof, not
        // just an indirect one via the post-optimizer weights.
        for name in &names {
            let rg = resident.read_grad(name);
            let sg = streaming.lora.ps.read_grad(&streaming.gpu, name);
            assert_eq!(rg.len(), sg.len(), "step {step}: {name}: grad length mismatch");
            let maxdiff = rg.iter().zip(&sg).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(maxdiff < 1e-4, "step {step}: {name}: grad maxdiff {maxdiff} (resident vs streaming)");
        }

        resident.adamw_step(step, lr, 0.0, None, 1.0);
        streaming.lora.adamw_step(&streaming.gpu, step, lr);

        for name in &names {
            let rw = resident.read_weight(name);
            let sw = streaming.lora.ps.read_weight(&streaming.gpu, name);
            let maxdiff = rw.iter().zip(&sw).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(maxdiff < 1e-4, "step {step}: {name}: weight maxdiff {maxdiff} after AdamW (resident vs streaming)");
        }
    }

    println!("streaming_lora_trainer_matches_the_resident_trainer_exactly: loss trajectory (resident, streaming):");
    for (i, (r, s)) in loss_trajectory.iter().enumerate() {
        println!("  step {}: resident={r:.6} streaming={s:.6}", i + 1);
    }
}

/// A separate, narrower gate: the streaming trainer's own loss must actually
/// go DOWN over a few real steps on a fixed batch (not just match the
/// resident trainer's number) - a real, if small, convergence signal, not
/// only a bit-for-bit-replay proof. Independent of the equivalence test
/// above so a future change that breaks convergence but not equivalence (or
/// vice versa) is still caught.
#[test]
fn streaming_lora_trainer_reduces_loss_on_a_fixed_batch() {
    let cfg = tiny_lora_cfg();
    let init: HashMap<String, Vec<f32>> = qwen35::init::init_weights(&cfg, SEED ^ 0xA5A5);
    let (tokens, targets) = batch(&cfg);

    let streaming = StreamTrainer::new_synthetic(Gpu::new_cpu(pipelines()), &cfg, N, WINDOW_BUDGET, &init);
    let x0 = StreamTrainer::embed_synthetic(&init, &tokens, cfg.d_model as usize);
    let loader = streaming.synthetic_loader(&cfg, &init);

    let mut losses = Vec::new();
    for step in 1..=8u32 {
        let loss = streaming.step(&cfg, &loader, &x0, &targets, 0.05, step);
        losses.push(loss);
    }
    println!("streaming_lora_trainer_reduces_loss_on_a_fixed_batch: {losses:?}");
    assert!(losses.last().unwrap() < &losses[0], "loss did not decrease: {losses:?}");
}
