// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving fine-tuned adapters on
// very large models at interactive speed for its clients. If your team needs
// expertise in low-rank adaptation inference you can procure our services by
// sending an email to info@swedishembedded.com.

//! The KV-cached decode path (`prefill_chunked` + `step`) of a LoRA build must
//! apply the adapter exactly as the whole-sequence forward (`logits_all`) does.
//!
//! * a random non-zero `lora_b` on every target: decode logits equal the full
//!   forward's, at every position;
//! * an all-zero `lora_b`: decode is bit-identical to a build without LoRA.

mod common;

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

// seed: the tolerances below was measured on this weight draw
const SEED: u64 = 7;
const PREFIX: usize = 9;

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn tokens(cfg: &Qwen35Config) -> Vec<u32> {
    (0..cfg.block_size).map(|i| (i * 5 + 3) % cfg.vocab).collect()
}

/// A LoRA build (rank 5, coprime with the tiny dims) over the base `init_weights(.., SEED)`;
/// every `lora_b` is drawn non-zero when `nonzero_b`, left at its zero init otherwise.
fn lora_model(gpu: Gpu, nonzero_b: bool) -> Qwen35 {
    let base_cfg = Qwen35Config::tiny();
    let cfg = Qwen35Config { lora: Some(lora_cfg(5, 8.0)), ..Qwen35Config::tiny() };
    let mut init: HashMap<String, Vec<f32>> = qwen35::init::init_weights(&cfg, SEED);
    init.extend(qwen35::init::init_weights(&base_cfg, SEED));
    if nonzero_b {
        // seed: the tolerances below were measured on this adapter draw
        let mut rng = Rng::new(99);
        // Large enough that the low-rank delta clearly moves the logits (lora_a's
        // init std is 0.02, so a unit-scale lora_b alone would be lost in rounding).
        for (name, w) in init.iter_mut().filter(|(n, _)| n.ends_with(".lora_a") || n.ends_with(".lora_b")) {
            if name.ends_with(".lora_b") {
                assert!(w.iter().all(|&x| x == 0.0), "{name}: expected the zero init");
                w.iter_mut().for_each(|x| *x = 2.0 * (rng.next_f32() - 0.5));
            } else {
                w.iter_mut().for_each(|x| *x *= 20.0);
            }
        }
    }
    Qwen35::new_train_on(gpu, cfg.clone(), 1, cfg.block_size, &init)
}

/// Logits of one final-norm hidden state under the host copy of the LM head `[vocab, d_model]`.
fn logits_of(m: &Qwen35, head: &[f32], hidden: &[f32]) -> Vec<f32> {
    let d = m.cfg.d_model as usize;
    (0..m.cfg.vocab as usize).map(|r| head[r * d..(r + 1) * d].iter().zip(hidden).map(|(a, b)| a * b).sum()).collect()
}

/// Decode of `m` (any tier) against its own whole-sequence forward, within `tol` on the logits.
fn assert_decode_matches_full_forward(m: &Qwen35, head: &[f32], tol: f32) -> f32 {
    let toks = tokens(&m.cfg);
    let v = m.cfg.vocab as usize;
    let full = m.logits_all(&toks);

    // The chunked prefill's last row, then one KV-cached step per remaining token.
    m.reset_decode_cache();
    let mut hidden = m.prefill_chunked(&toks[..PREFIX], 4);
    let mut worst = maxabs(&logits_of(m, head, &hidden), &full[(PREFIX - 1) * v..PREFIX * v]);
    for (i, &tok) in toks.iter().enumerate().skip(PREFIX) {
        hidden = m.step(tok);
        worst = worst.max(maxabs(&logits_of(m, head, &hidden), &full[i * v..(i + 1) * v]));
    }
    assert!(worst < tol, "decode with the adapter vs full forward with the adapter: maxabs {worst}");
    worst
}

fn decode_matches_full_forward(gpu: Gpu) {
    let m = lora_model(gpu, true);
    let worst = assert_decode_matches_full_forward(&m, &m.read_weight(m.cfg.head_weight()), 2e-3);
    let toks = tokens(&m.cfg);
    let full = m.logits_all(&toks);

    // Guard against a vacuous pass: the adapter must actually move the logits.
    let plain_cfg = Qwen35Config::tiny();
    let plain = Qwen35::new_on(Gpu::new_cpu(pipelines()), plain_cfg.clone(), 1, plain_cfg.block_size, &qwen35::init::init_weights(&plain_cfg, SEED));
    let moved = maxabs(&plain.logits_all(&toks), &full);
    assert!(moved > 10.0 * worst.max(1e-4), "the adapter changes logits by only {moved}; the test cannot tell applied from ignored");
}

#[test]
fn lora_decode_matches_full_forward_cpu() {
    decode_matches_full_forward(Gpu::new_cpu(pipelines()));
}

#[test]
fn zero_lora_b_decodes_bit_identically_to_a_build_without_lora() {
    let lora = lora_model(Gpu::new_cpu(pipelines()), false);
    let plain_cfg = Qwen35Config::tiny();
    let plain = Qwen35::new_on(Gpu::new_cpu(pipelines()), plain_cfg.clone(), 1, plain_cfg.block_size, &qwen35::init::init_weights(&plain_cfg, SEED));
    let toks = tokens(&plain_cfg);
    lora.reset_decode_cache();
    plain.reset_decode_cache();
    assert_eq!(lora.prefill_chunked(&toks[..PREFIX], 4), plain.prefill_chunked(&toks[..PREFIX], 4), "prefill hidden differs");
    for &tok in &toks[PREFIX..] {
        assert_eq!(lora.step(tok), plain.step(tok), "step hidden differs");
    }
}

/// The same equivalence on the default device (the NF4 tier's CUDA agreement is
/// checked by the real-checkpoint runs; this covers the device GEMM route of the
/// adapter products).
#[test]
#[ignore = "needs CUDA"]
fn lora_decode_matches_full_forward_default_backend() {
    decode_matches_full_forward(Gpu::new(pipelines()));
}

/// The NF4 tier: the adapter rides on an NF4-held base (with the embedding table held at bf16,
/// outside the fp32 store), in the decode step as in the forward. Both paths run the same
/// dequantising kernels, so the gap is fp32 reduction order and the activation rounding of a
/// 1-row against a many-row GEMM.
fn nf4_decode_matches_full_forward(gpu: Gpu) {
    let cfg = Qwen35Config { lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny_i8() };
    let init = common::weights(&cfg, 0.5);
    let m = Qwen35::new_train_opts_on(gpu, cfg.clone(), 1, cfg.block_size, &init, &common::opts(Dtype::NF4));
    assert_eq!(m.linear_dtype("blocks.0.mlp.gate.weight"), Some(Dtype::NF4));
    assert_eq!(m.linear_dtype("tok.weight"), Some(Dtype::BF16));
    assert_decode_matches_full_forward(&m, &init[cfg.head_weight()], 5e-2);
}

#[test]
fn lora_decode_matches_full_forward_nf4_tier_cpu() {
    nf4_decode_matches_full_forward(Gpu::new_cpu(pipelines()));
}

#[test]
#[ignore = "needs CUDA"]
fn lora_decode_matches_full_forward_nf4_tier() {
    nf4_decode_matches_full_forward(Gpu::new(pipelines()));
}
