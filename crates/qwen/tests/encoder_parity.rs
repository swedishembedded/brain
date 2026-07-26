// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-4B text-encoder parity: brain's penultimate hidden state
//! (`res[n_layers-1]`, the Z-Image/FLUX.2 caption features) vs the transformers
//! reference `hidden_states[-2]`.
//!
//! Golden fixture (`tests/fixtures/qwen3_4b_encoder.safetensors`, committed): a
//! fixed token sequence + Qwen3-4B `hidden_states[-2]` `[8,2560]`, baked by
//! `resources/image-models/_goldens/gen_qwen_encoder.py` from the SAME Comfy
//! single-file weights brain imports. Fixed token ids isolate forward parity
//! from tokenizer parity. The 8 GB weights are NOT committed — set
//! `BRAIN_QWEN3_4B` (a resources default is tried); skips if absent. Runs on the
//! CPU backend (`BRAIN_DEVICE=cpu` recommended; deterministic).

use std::path::Path;

use qwen::{QwenConfig, Qwen};

const DEFAULT_WEIGHTS: &str = "/data/workspace/resources/image-models/common/qwen3-4b-text-encoder/split_files/text_encoders/qwen_3_4b.safetensors";
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/qwen3_4b_encoder.safetensors");

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den).sqrt()
}

#[test]
fn qwen3_4b_penultimate_hidden_matches_transformers() {
    let wpath = std::env::var("BRAIN_QWEN3_4B").unwrap_or_else(|_| DEFAULT_WEIGHTS.to_string());
    if !Path::new(&wpath).exists() {
        eprintln!("SKIP: Qwen3-4B weights not found at {wpath} (set BRAIN_QWEN3_4B)");
        return;
    }

    // Golden.
    let fx = checkpoint::safetensors::read(FIXTURE).expect("read fixture");
    let tokens_i: &Vec<f32> = &fx.iter().find(|t| t.name == "tokens").unwrap().data;
    let tokens: Vec<u32> = tokens_i.iter().map(|&x| x as u32).collect();
    let want = &fx.iter().find(|t| t.name == "hidden").unwrap().data;

    // brain: import the same weights, build the encoder, run it.
    let cfg = QwenConfig::qwen3_4b();
    let tensors = checkpoint::safetensors::read(&wpath).expect("read weights");
    let init = qwen::import::brain_init_from_hf(tensors, &cfg).expect("brain_init_from_hf");
    let model = Qwen::new(cfg.clone(), 1, tokens.len() as u32, &init);
    let got = model.encode(&tokens);

    assert_eq!(got.len(), want.len(), "hidden len {} != golden {}", got.len(), want.len());
    let cos = cosine(&got, want);
    let rl2 = rel_l2(&got, want);
    let max_abs = got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!(
        "Qwen3-4B encoder parity: cosine={cos:.6}  rel_l2={rl2:.5}  max_abs={max_abs:.3}  (|want|max≈{:.0})",
        want.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
    );
    assert!(cos >= 0.9999, "cosine {cos:.6} < 0.9999");
    assert!(rl2 <= 0.02, "rel_l2 {rl2:.5} > 0.02");
}
