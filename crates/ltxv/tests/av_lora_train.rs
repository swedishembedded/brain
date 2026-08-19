// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The three parts of the LoRA gate for the audio+video LTX DiT - the AV
//! twin of `crates/ltxv/tests/lora_train.rs`: **exact no-op at init**,
//! **measured descent**, and **fold-vs-apply bit-equality**. Plus the base
//! staying frozen and a save/load round trip.
//!
//! Bit-equality is the right bar for the same reason the video-only path's
//! own doc gives: the AV DiT fuses nothing either (every LoRA pair covers a
//! whole `[out,in]` tensor at offset 0), so `apply` and `fold_into_tensors`
//! perform the identical additions in the identical order.

use std::collections::HashMap;

use ltxv::av_lora::{LoraAdapter, LoraCfg};
use ltxv::av_modelgrad::{grads, make_av_flow_batch, AvBatch, AvCfg, AvModelWeights};
use ltxv::dit::av_dit_tensor_manifest;
use ltxv::LtxAvDitConfig;
use vae::blocks::Tensors;

fn tiny_av() -> LtxAvDitConfig {
    LtxAvDitConfig::tiny()
}

fn synthetic_weights(cfg: &LtxAvDitConfig) -> Tensors {
    let mut t: Tensors = HashMap::new();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for (name, shape) in av_dit_tensor_manifest(cfg) {
        let n: usize = shape.iter().product();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(0.2 * (((state >> 33) as u32) as f32 / (1u64 << 31) as f32 - 0.5));
        }
        if name.contains("q_norm") || name.contains("k_norm") {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        t.insert(name, (shape, v));
    }
    t
}

fn fixed_batch(cfg: &AvCfg) -> AvBatch<f32> {
    let v_x0: Vec<f32> = (0..cfg.tv * cfg.v_in_channels).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let a_x0: Vec<f32> = (0..cfg.ta * cfg.a_in_channels).map(|i| ((i % 17) as f32 / 17.0 - 0.5) * 0.9).collect();
    let v_noise: Vec<f32> = (0..v_x0.len()).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
    let a_noise: Vec<f32> = (0..a_x0.len()).map(|i| ((i % 11) as f32 / 11.0 - 0.5) * 0.7).collect();
    let v_ctx: Vec<f32> = (0..cfg.v_context_len * cfg.vdim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    let a_ctx: Vec<f32> = (0..cfg.a_context_len * cfg.adim).map(|i| ((i % 5) as f32 / 5.0 - 0.5) * 1.2).collect();
    make_av_flow_batch(cfg, &v_x0, &a_x0, &v_ctx, &a_ctx, 0.5, 0.6, &v_noise, &a_noise)
}

#[test]
fn av_lora_is_a_no_op_at_init_then_descends_with_the_base_frozen() {
    let cfg = AvCfg::tiny();
    let ts = synthetic_weights(&tiny_av());
    let base = AvModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));

    // 1. exact no-op at init: B = 0, so W_eff is the base BIT for BIT.
    assert!(ad.apply(&base) == base, "a fresh AV adapter must not change a single weight");

    let b = fixed_batch(&cfg);
    let (l0, _) = grads(&cfg, &ad.apply(&base), &b);
    let mut last = l0;
    for step in 0..40 {
        let w_eff = ad.apply(&base);
        let (l, g) = grads(&cfg, &w_eff, &b);
        ad.step(&g, 3e-3);
        if step % 10 == 0 {
            println!("  av lora step {step:>3}  loss {l:.6}");
        }
        last = l;
    }
    println!("av lora: loss {l0:.6} -> {last:.6} over 40 steps (rank 4, lr 3e-3)");
    assert!(last < l0 * 0.9, "AV LoRA training must descend: {l0} -> {last}");

    // 2. the base is frozen: `apply` clones, so the original is untouched.
    let base_again = AvModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    assert!(base == base_again, "the base AV weights must not move during LoRA training");
}

#[test]
fn folding_into_the_av_inference_tensors_equals_applying_to_the_training_weights() {
    let cfg = AvCfg::tiny();
    let av_cfg = tiny_av();
    let ts = synthetic_weights(&av_cfg);
    let base = AvModelWeights::from_tensors(&cfg, &ts).expect("host weights");

    // Train a few steps so B is non-zero - folding zeros would prove nothing.
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));
    let b = fixed_batch(&cfg);
    for _ in 0..3 {
        let (_l, g) = grads(&cfg, &ad.apply(&base), &b);
        ad.step(&g, 5e-3);
    }

    let applied = ad.apply(&base);
    let mut folded_ts = ts.clone();
    ad.fold_into_tensors(&mut folded_ts).expect("fold");
    let folded = AvModelWeights::from_tensors(&cfg, &folded_ts).expect("host weights");
    assert!(applied == folded, "fold-into-tensors and apply-to-weights must be bit-equal");
    assert!(applied != base, "3 AV LoRA steps must have moved the effective weights");

    // A missing base tensor is an error BY NAME, not a silent skip.
    let mut broken = ts.clone();
    broken.remove("transformer_blocks.1.audio_to_video_attn.to_v.weight");
    let e = ad.fold_into_tensors(&mut broken).expect_err("a missing tensor must fail");
    assert!(e.contains("transformer_blocks.1.audio_to_video_attn.to_v.weight"), "{e}");
}

#[test]
fn an_av_adapter_round_trips_through_the_checkpoint_container() {
    let cfg = AvCfg::tiny();
    let ts = synthetic_weights(&tiny_av());
    let base = AvModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));
    let b = fixed_batch(&cfg);
    for _ in 0..2 {
        let (_l, g) = grads(&cfg, &ad.apply(&base), &b);
        ad.step(&g, 5e-3);
    }
    let dir = std::env::temp_dir().join(format!("ltxv-av-lora-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("adapter.brain");
    let p = path.to_str().expect("utf-8 path");
    ltxv::av_lora::save_adapter(p, &ad);
    let back = ltxv::av_lora::load_adapter(p, &cfg).expect("reload");
    assert_eq!(back.rank(), ad.rank());
    assert!(ad.apply(&base) == back.apply(&base), "a reloaded AV adapter must produce the same effective weights");
    let _ = std::fs::remove_dir_all(&dir);
}
