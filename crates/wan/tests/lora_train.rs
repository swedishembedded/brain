// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The three parts of the LoRA gate: **exact no-op at init**, **measured
//! descent**, and **fold-vs-apply bit-equality**. Plus the base staying frozen
//! and a save/load round trip.
//!
//! Bit-equality (not "close") is the right bar here because Wan fuses nothing:
//! every adapter pair covers a whole `[out, in]` tensor at offset 0, so
//! `apply` (add into the host training weights) and `fold_into_tensors` (add
//! into the inference tensor map) perform the identical additions in the
//! identical order. Any difference at all would mean the two walks disagree
//! about which tensor is which - the defect that trains `k` into `q` and still
//! produces plausible video.

use std::collections::HashMap;

use wan::config::WanConfig;
use wan::import::dit_manifest;
use wan::lora::{LoraAdapter, LoraCfg};
use wan::model::Tensors;
use wan::modelgrad::{grads, make_flow_batch, Batch, Cfg, ModelWeights};

fn tiny_wan(c: &Cfg) -> WanConfig {
    WanConfig {
        name: "tiny-lora",
        dim: c.dim,
        ffn_dim: c.ffn_dim,
        num_heads: c.n_heads,
        num_layers: c.n_layers,
        in_channels: c.in_channels,
        out_channels: c.out_channels,
        text_dim: c.text_dim,
        text_len: c.text_len,
        freq_dim: c.freq_dim,
        ..WanConfig::t2v_1_3b()
    }
}

fn synthetic_weights(cfg: &WanConfig) -> Tensors {
    let mut t: Tensors = HashMap::new();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for (name, shape) in dit_manifest(cfg) {
        let n: usize = shape.iter().product();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(0.2 * (((state >> 33) as u32) as f32 / (1u64 << 31) as f32 - 0.5));
        }
        if name.contains("norm_q") || name.contains("norm_k") || name.ends_with("norm3.weight") {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        t.insert(name, (shape, v));
    }
    t
}

fn fixed_batch(cfg: &Cfg) -> Batch<f32> {
    let x0: Vec<f32> = (0..cfg.latent_len()).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
    let rows = cfg.text_len - 1;
    let ctx: Vec<f32> = (0..rows * cfg.text_dim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    make_flow_batch(cfg, &x0, &ctx, rows, 0.5, &noise)
}

#[test]
fn lora_is_a_no_op_at_init_then_descends_with_the_base_frozen() {
    let cfg = Cfg::tiny();
    let ts = synthetic_weights(&tiny_wan(&cfg));
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));

    // 1. exact no-op at init: B = 0, so W_eff is the base BIT for BIT.
    assert!(ad.apply(&base) == base, "a fresh adapter must not change a single weight");

    let b = fixed_batch(&cfg);
    let (l0, _) = grads(&cfg, &ad.apply(&base), &b);
    let mut last = l0;
    for step in 0..40 {
        let w_eff = ad.apply(&base);
        let (l, g) = grads(&cfg, &w_eff, &b);
        ad.step(&g, 3e-3);
        if step % 10 == 0 {
            println!("  lora step {step:>3}  loss {l:.6}");
        }
        last = l;
    }
    println!("lora: loss {l0:.6} -> {last:.6} over 40 steps (rank 4, lr 3e-3)");
    assert!(last < l0 * 0.9, "LoRA training must descend: {l0} -> {last}");

    // 2. the base is frozen: `apply` clones, so the original is untouched.
    let base_again = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    assert!(base == base_again, "the base weights must not move during LoRA training");
}

#[test]
fn folding_into_the_inference_tensors_equals_applying_to_the_training_weights() {
    let cfg = Cfg::tiny();
    let wc = tiny_wan(&cfg);
    let ts = synthetic_weights(&wc);
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");

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
    let folded = ModelWeights::from_tensors(&cfg, &folded_ts).expect("host weights");
    assert!(applied == folded, "fold-into-tensors and apply-to-weights must be bit-equal");
    // ...and the adapter really did move something (guards against a
    // vacuously-true comparison of two untouched copies).
    assert!(applied != base, "3 LoRA steps must have moved the effective weights");

    // A missing base tensor is an error BY NAME, not a silent skip.
    let mut broken = ts.clone();
    broken.remove("blocks.1.cross_attn.v.weight");
    let e = ad.fold_into_tensors(&mut broken).expect_err("a missing tensor must fail");
    assert!(e.contains("blocks.1.cross_attn.v.weight"), "{e}");
}

#[test]
fn an_adapter_round_trips_through_the_checkpoint_container() {
    let cfg = Cfg::tiny();
    let ts = synthetic_weights(&tiny_wan(&cfg));
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));
    let b = fixed_batch(&cfg);
    for _ in 0..2 {
        let (_l, g) = grads(&cfg, &ad.apply(&base), &b);
        ad.step(&g, 5e-3);
    }
    let dir = std::env::temp_dir().join(format!("wan-lora-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("adapter.brain");
    let p = path.to_str().expect("utf-8 path");
    wan::lora::save_adapter(p, &ad);
    let back = wan::lora::load_adapter(p, &cfg).expect("reload");
    assert_eq!(back.rank(), ad.rank());
    assert!(ad.apply(&base) == back.apply(&base), "a reloaded adapter must produce the same effective weights");
    let _ = std::fs::remove_dir_all(&dir);
}
