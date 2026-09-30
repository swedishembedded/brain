// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A LoRA adapter applied at runtime on a resident base (`Qwen::attach_adapter`)
//! must be the same model as the adapter folded into the base weights, and
//! detaching it must give back exactly the base. Runs on whichever backend
//! `BRAIN_DEVICE` selects (`cpu`, `vulkan`).

use std::collections::HashMap;

use qwen3::{LoraCfg, Qwen, QwenConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-lora-runtime-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every linear a Qwen3 block has, so the MLP sites are exercised as well as
/// the attention ones.
fn all_targets(rank: u32, alpha: f32) -> LoraCfg {
    LoraCfg { rank, alpha, targets: ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect() }
}

/// Train a small adapter off its `B = 0` init so its delta is non-trivial,
/// save it, and return the trained LoRA build, the frozen base tensors and
/// the adapter file.
/// The fixed training batch: 12 inputs and their next-token targets.
fn batch() -> (Vec<u32>, Vec<u32>) {
    ((0..12).map(|i| (i * 7 + 1) % 23).collect(), (0..12).map(|i| (i * 7 + 2) % 23).collect())
}

fn trained_adapter(tag: &str) -> (Qwen, HashMap<String, Vec<f32>>, String) {
    let lora_cfg = QwenConfig { lora: Some(all_targets(3, 6.0)), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&lora_cfg, 21);
    let (x, y) = batch();
    let trained = Qwen::new(lora_cfg, 1, 12, &init);
    trained.set_batch(&x, &y);
    for step in 1..=8 {
        trained.zero_grads();
        trained.forward();
        trained.backward();
        trained.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
        trained.poll_wait();
    }
    let path = tmp(tag).join("adapter.safetensors").to_string_lossy().into_owned();
    qwen3::lora::save_adapter(&path, &trained, "test/adapter", "test/base", None).unwrap();
    let base: HashMap<String, Vec<f32>> = trained
        .ps
        .params
        .iter()
        .map(|(name, _)| name)
        .filter(|name| !(name.ends_with(".lora_a") || name.ends_with(".lora_b")))
        .map(|name| (name.clone(), trained.read_weight(name)))
        .collect();
    (trained, base, path)
}

const PROMPT: [u32; 10] = [3, 17, 5, 9, 22, 1, 14, 8, 11, 6];

/// Decode `PROMPT` from an empty cache, returning every position's logits.
fn decode_logits(m: &Qwen) -> Vec<f32> {
    m.reset_cache();
    let mut out = Vec::new();
    for &tok in &PROMPT {
        m.step(tok);
        out.extend(m.decode_logits());
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// fp32 tolerance: the fold and the runtime correction sum the same products
/// in a different order, so they agree to rounding, relative to the logit
/// scale.
fn assert_fp32_close(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    let diff = max_abs_diff(got, want);
    assert!(diff <= 1e-4 * scale, "{what}: max |diff| {diff:e} exceeds 1e-4 x logit scale {scale:e}");
}

#[test]
fn attached_adapter_matches_the_fold_and_detach_restores_the_base_exactly() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let (_trained, base, adapter) = trained_adapter("decode");
    let mut folded_tensors = base.clone();
    qwen3::lora::fold_adapter_into(&mut folded_tensors, &adapter).unwrap();
    let folded = Qwen::from_tensors_decode(QwenConfig::tiny(), &folded_tensors, 32);
    let want = decode_logits(&folded);

    let mut m = Qwen::from_tensors_decode(QwenConfig::tiny(), &base, 32);
    let base_logits = decode_logits(&m);
    // The adapter must actually move the model, or parity below proves nothing.
    let moved = max_abs_diff(&base_logits, &want);
    assert!(moved > 1e-2, "the trained adapter barely changes the logits ({moved:e}); the parity gate would be vacuous");

    m.attach_adapter(&adapter).unwrap();
    assert_eq!(m.attached_adapter(), Some((3, 6.0)));
    assert_fp32_close(&decode_logits(&m), &want, "attached vs folded (decode)");

    assert!(m.detach_adapter());
    assert!(!m.detach_adapter(), "a second detach has nothing to remove");
    assert_eq!(decode_logits(&m), base_logits, "after detach the model must be exactly the base again");

    // Re-attaching serves the adapter again from the same resident base.
    m.attach_adapter(&adapter).unwrap();
    assert_fp32_close(&decode_logits(&m), &want, "re-attached vs folded (decode)");
}

#[test]
fn attached_adapter_applies_in_the_batched_forward() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let (_trained, base, adapter) = trained_adapter("batched");
    let mut folded_tensors = base.clone();
    qwen3::lora::fold_adapter_into(&mut folded_tensors, &adapter).unwrap();
    let shard = qwen3::Shard::whole(QwenConfig::tiny().n_layers as usize);
    let folded = Qwen::new_shard(QwenConfig::tiny(), 1, 12, &folded_tensors, false, shard.clone());
    let mut m = Qwen::new_shard(QwenConfig::tiny(), 1, 12, &base, false, shard);
    let base_logits = m.logits_all(&PROMPT);

    m.attach_adapter(&adapter).unwrap();
    assert_fp32_close(&m.logits_all(&PROMPT), &folded.logits_all(&PROMPT), "attached vs folded (batched)");
    m.detach_adapter();
    assert_eq!(m.logits_all(&PROMPT), base_logits, "after detach the batched forward must be exactly the base again");
}

#[test]
fn a_lora_build_decodes_with_its_own_trainable_adapter() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let (trained, base, adapter) = trained_adapter("trainable");
    let mut folded_tensors = base;
    qwen3::lora::fold_adapter_into(&mut folded_tensors, &adapter).unwrap();
    let folded = Qwen::from_tensors_decode(QwenConfig::tiny(), &folded_tensors, 32);
    // The training build decodes through its live, unfolded adapter.
    assert_fp32_close(&decode_logits(&trained), &decode_logits(&folded), "LoRA build decode vs folded");
}

#[test]
fn attach_refuses_an_adapter_for_another_shape_and_keeps_serving_the_base() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let (_trained, _base, adapter) = trained_adapter("refuse");
    // Same names, wider model: every pair disagrees with the base's shape.
    let wide = QwenConfig { d_model: 24, ..QwenConfig::tiny() };
    let wide_init = qwen3::init_weights(&wide, 5);
    let mut m = Qwen::from_tensors_decode(wide.clone(), &wide_init, 16);
    let before = decode_logits(&m);
    let err = m.attach_adapter(&adapter).unwrap_err();
    assert!(err.contains("blocks.0."), "the refusal names the mismatched linear: {err}");
    assert_eq!(m.attached_adapter(), None);
    assert_eq!(decode_logits(&m), before, "a refused attach must leave the base untouched");

    // A tensor that is not one of the model's linears is refused by name.
    let mut stray = qwen3::lora::read_adapter(&adapter).unwrap();
    stray.sites[0].base = "blocks.0.ln1.weight".into();
    let mut m = Qwen::from_tensors_decode(QwenConfig::tiny(), &qwen3::init_weights(&QwenConfig::tiny(), 5), 16);
    let err = m.attach_adapter_pairs(&stray).unwrap_err();
    assert!(err.contains("blocks.0.ln1.weight"), "{err}");
    assert_eq!(m.attached_adapter(), None);
}

/// A LoRA training build is itself a resident base: it serves its own live
/// adapter, any attached one, or none, and trains again once the override is
/// gone - one copy of the base weights for all of it. The training tape it
/// resubmits every step must come through the inference in between intact:
/// the loss afterwards is exactly the loss before.
#[test]
fn a_training_build_serves_any_adapter_or_none_and_trains_after_detach() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let (mut trained, base, adapter) = trained_adapter("shared");
    let (x, y) = batch();
    trained.set_batch(&x, &y);
    let loss_before = trained.forward();
    let own = decode_logits(&trained);
    let base_logits = decode_logits(&Qwen::from_tensors_decode(QwenConfig::tiny(), &base, 32));

    trained.bypass_adapter();
    assert_eq!(trained.attached_adapter(), None);
    assert_eq!(decode_logits(&trained), base_logits, "bypassed, a LoRA build must be exactly its base");

    // Another adapter in place of the trainable one: the trained pair scaled
    // down, written through the host-side form a trainer would hand over.
    let mut other = qwen3::lora::read_adapter(&adapter).unwrap();
    other.alpha *= 0.5;
    let mut folded = base.clone();
    for site in &other.sites {
        let w = folded.get_mut(&site.base).unwrap();
        let (out, inn) = site.dims(other.rank);
        for o in 0..out {
            for i in 0..inn {
                let delta: f32 = (0..other.rank as usize).map(|k| site.b[o * other.rank as usize + k] * site.a[k * inn + i]).sum();
                w[o * inn + i] += other.scale() * delta;
            }
        }
    }
    trained.attach_adapter_pairs(&other).unwrap();
    assert_eq!(trained.attached_adapter(), Some((3, 3.0)));
    let want = decode_logits(&Qwen::from_tensors_decode(QwenConfig::tiny(), &folded, 32));
    assert_fp32_close(&decode_logits(&trained), &want, "LoRA build with another adapter attached vs that adapter folded");

    // Training under an override would differentiate parameters the forward
    // did not use, so it is refused.
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trained.zero_grads();
        trained.forward();
        trained.backward();
    }));
    assert!(refused.is_err(), "backward must refuse while an adapter is attached");

    assert!(trained.detach_adapter());
    assert_eq!(decode_logits(&trained), own, "detached, a LoRA build serves its own trainable adapter again");
    trained.set_batch(&x, &y);
    trained.zero_grads();
    let loss = trained.forward();
    assert_eq!(loss, loss_before, "the recorded training forward must be untouched by the inference in between");
    trained.backward();
    trained.poll_wait();
}
