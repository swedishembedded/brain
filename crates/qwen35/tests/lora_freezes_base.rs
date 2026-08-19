// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Focused test for the LoRA wiring's core promise (M8): with `cfg.lora` set,
//! a [`Qwen35::new_train_on`] build must train ONLY the `.lora_a`/`.lora_b`
//! adapter tensors; every frozen base weight (the 12 targeted linears' bases,
//! plus every never-targeted weight: norms, embeddings, `A_log`/`dt_bias`,
//! `conv1d.weight`) must come out of a real training loop bit-identical to
//! where it started. Not a gradcheck (that's `gradcheck::check_qwen35_lora`,
//! which validates the ADAPTER gradients are numerically correct), this is
//! the complementary "did the freeze actually hold" check: a LoRA wiring bug
//! that accidentally left the base `Role::Trainable` (or silently no-opped
//! the adapter update) would still gradient-check fine but corrupt/waste the
//! whole point of LoRA; only THIS test would catch it. Mirrors
//! `qwen35moe/tests/lora_freezes_base.rs` exactly.

use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn run(gpu: Gpu) {
    let mut cfg = Qwen35Config::tiny();
    // Both layer types (n_layers=4, full_attention_interval=4 -> layer 3 is
    // GQA, the rest GDN) and every one of the 12 targetable leaves.
    cfg.lora = Some(lora_cfg(2, 4.0));
    let b = 1;
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 7);
    let all_names: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
    let lora_leaf_count = all_names.iter().filter(|n| n.ends_with(".lora_a")).count();
    assert!(lora_leaf_count > 0, "tiny() + lora_cfg must target at least one leaf");

    let m = Qwen35::new_train_on(gpu, cfg.clone(), b, t, &init);

    // `param_names()` (the surface gradcheck's `directional_check` walks)
    // must list ONLY the trainable adapters -- never the frozen base, and
    // never an untargeted weight.
    let names = m.param_names();
    assert!(!names.is_empty(), "a LoRA build must have at least one trainable tensor");
    assert!(
        names.iter().all(|n| n.ends_with(".lora_a") || n.ends_with(".lora_b")),
        "LoRA build's param_names must list only adapter tensors, got: {names:?}"
    );

    // Snapshot every real parameter (base + adapters) before training.
    let before: Vec<(String, Vec<f32>)> = all_names.iter().map(|n| (n.clone(), m.read_weight(n))).collect();

    let x: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 3 + 2) % cfg.vocab).collect();
    m.set_batch(&x, &y);
    for step in 1..=3 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 1e-1, 0.0, Some(1.0), 1.0);
        m.poll_wait();
    }

    let mut any_lora_changed = false;
    for (name, before_w) in &before {
        let after_w = m.read_weight(name);
        if name.ends_with(".lora_a") || name.ends_with(".lora_b") {
            if after_w != *before_w {
                any_lora_changed = true;
            }
        } else {
            assert_eq!(&after_w, before_w, "frozen base weight {name} changed under LoRA training");
        }
    }
    assert!(any_lora_changed, "no LoRA adapter weight changed after training steps -- the adapter update is silently a no-op");
}

#[test]
fn lora_training_only_updates_adapters_cpu() {
    run(Gpu::new_cpu(pipelines()));
}

/// `Gpu::new` honours `BRAIN_DEVICE` when set and defaults to the wgpu
/// backend otherwise -- run both, since a barrier-crossing kernel can
/// silently misbehave on exactly one backend.
#[test]
fn lora_training_only_updates_adapters_default_backend() {
    run(Gpu::new(pipelines()));
}
