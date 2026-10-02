// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A Qwen3.8 paged engine whose KV cache is stored `bf16` or per-row `int8`
//! must produce the logits of the `f32` cache, to a tolerance the tier earns,
//! across the whole serving path: chunked prefill over several rounds, then
//! batched decode of sequences at different lengths sharing one pool.
//!
//! Swedish Embedded AB implements long-context inference serving for clients
//! whose GPU memory decides how many users one card carries. If your team needs
//! expertise in fitting more concurrent sequences into the same device memory
//! without giving up accuracy, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The comparison is on LOGITS, not on sampled tokens: a randomly initialised
//! model has near-flat logits, so its argmax flips on perturbations far below
//! anything a trained model would notice, and "same tokens" would measure the
//! initialisation. The gates are therefore the relative L2 error of the full
//! logits vector (prefill) and of the top-k logits (decode), each against the
//! tier's own storage rounding: `bf16` keeps 8 significant bits, `int8` keeps
//! 7 bits per element against its row's absmax. The bounds are regression
//! gates set a few times above what the tiers measured (tiny: bf16 3e-6, int8
//! 1e-5; real dims: bf16 2e-4, int8 9e-4), not derived limits; the rounding
//! bounds themselves are gated per kernel in `crates/model/tests/kv_tier.rs`.
//! The trained-model agreement (top-1 on a real prompt) is gated in
//! `gguf_kv_tier_real.rs`.
//!
//! Two shapes: the tiny hybrid (head_dim 40, the unfused triad everywhere) and
//! the real Qwen3.8-27B attention dims (head_dim 256, where a GPU takes the
//! fused prefill kernel).

use std::collections::HashMap;

use model::kv_tier::KvTier;
use model::paged::BlockTable;
use qwen35::config::Qwen35Config;
use qwen35::serve::Engine;

/// Two sequences sharing one pool: lengths chosen so the first needs two
/// prefill rounds (256 + 14) and the second a single short one.
const PROMPT_LENS: [usize; 2] = [270, 40];
const DECODE_STEPS: usize = 6;

fn prompt(seed: u32, len: usize, vocab: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 7 + seed * 13 + 3) % vocab).collect()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn rel_l2(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    l2(&want.iter().zip(got).map(|(w, g)| w - g).collect::<Vec<_>>()) / l2(want).max(1e-12)
}

/// What one tier's engine produced: the full logits after each sequence's
/// prefill, and the top-k `(token, logit)` rows of each decode step. Decode
/// inputs are the `f32` run's own greedy tokens, so every tier sees the same
/// token stream and only the cache differs.
struct Run {
    prefill_logits: Vec<Vec<f32>>,
    decode_topk: Vec<Vec<Vec<(u32, f32)>>>,
}

fn run(cfg: &Qwen35Config, weights: &HashMap<String, Vec<f32>>, tier: KvTier, inputs: Option<&[Vec<u32>]>) -> (Run, Vec<Vec<u32>>) {
    use model::serve::PagedDecoder;
    let max_seq = (PROMPT_LENS[0] + DECODE_STEPS + 1) as u32;
    let mut engine = Engine::from_map_kv(cfg.clone(), weights, max_seq, 2, tier);
    let mut tables: Vec<BlockTable> = (0..2).map(|_| BlockTable::new()).collect();
    let mut prefill_logits = Vec::new();
    let mut next: Vec<u32> = Vec::new();
    for (s, table) in tables.iter_mut().enumerate() {
        let hidden = engine.prefill(table, &prompt(s as u32, PROMPT_LENS[s], cfg.vocab));
        let logits = engine.logits(&hidden);
        next.push(logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap());
        prefill_logits.push(logits);
    }
    let (mut decode_topk, mut fed) = (Vec::new(), Vec::new());
    for step in 0..DECODE_STEPS {
        let cur: Vec<u32> = inputs.map(|i| i[step].clone()).unwrap_or_else(|| next.clone());
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        let rows = engine.forward_batched_topk(&mut refs, &cur, engine.topk_capacity().min(cfg.vocab as usize));
        next = rows.iter().map(|r| r[0].0).collect();
        fed.push(cur);
        decode_topk.push(rows);
    }
    (Run { prefill_logits, decode_topk }, fed)
}

/// Relative L2 error of `got`'s top-k logits against `want`'s, over the tokens
/// `want` ranks in its top-k (a token `got` ranks outside its own top-k is a
/// miss, counted as an error of the whole logit).
fn topk_rel_l2(want: &[(u32, f32)], got: &[(u32, f32)]) -> f32 {
    let got: HashMap<u32, f32> = got.iter().copied().collect();
    let (mut num, mut den) = (0f32, 0f32);
    for &(tok, w) in want {
        let g = got.get(&tok).copied().unwrap_or(0.0);
        num += (w - g) * (w - g);
        den += w * w;
    }
    (num / den.max(1e-12)).sqrt()
}

fn gate(cfg: &Qwen35Config, label: &str, bound_prefill: f32, bound_decode: f32, tier: KvTier) {
    let weights = qwen35::init::init_weights(cfg, 11);
    let (want, fed) = run(cfg, &weights, KvTier::F32, None);
    let (got, _) = run(cfg, &weights, tier, Some(&fed));

    let mut worst_prefill = 0f32;
    for (s, (w, g)) in want.prefill_logits.iter().zip(&got.prefill_logits).enumerate() {
        assert!(g.iter().all(|x| x.is_finite()), "{label} {tier}: sequence {s}'s prefill logits are not finite");
        let e = rel_l2(w, g);
        worst_prefill = worst_prefill.max(e);
        assert!(e <= bound_prefill, "{label} {tier}: sequence {s} prefill logits differ from f32 by rel-L2 {e} (> {bound_prefill})");
    }
    let mut worst_decode = 0f32;
    for (step, (w, g)) in want.decode_topk.iter().zip(&got.decode_topk).enumerate() {
        for (s, (wr, gr)) in w.iter().zip(g).enumerate() {
            let e = topk_rel_l2(wr, gr);
            worst_decode = worst_decode.max(e);
            assert!(e <= bound_decode, "{label} {tier}: decode step {step}, sequence {s}: top-k logits differ from f32 by rel-L2 {e} (> {bound_decode})");
        }
    }
    assert!(worst_prefill > 0.0, "{label} {tier}: logits are bit-identical to f32, so the compact cache was not the one read");
    eprintln!("{label} {tier}: worst rel-L2 vs f32 - prefill logits {worst_prefill:.3e}, decode top-k logits {worst_decode:.3e}");
}

fn tiny() -> Qwen35Config {
    Qwen35Config { block_size: 288, max_position_embeddings: 288, ..Qwen35Config::tiny_i8() }
}

/// The real model's attention dims (24 heads / 4 kv heads / head_dim 256) at a
/// reduced depth and vocabulary.
fn real_dims() -> Qwen35Config {
    Qwen35Config { vocab: 320, n_layers: 4, block_size: 288, max_position_embeddings: 288, ..Qwen35Config::qwen38_27b() }
}

#[test]
fn bf16_kv_tracks_f32_logits_on_the_tiny_hybrid() {
    gate(&tiny(), "tiny", 5e-5, 5e-5, KvTier::Bf16);
}

#[test]
fn int8_kv_tracks_f32_logits_on_the_tiny_hybrid() {
    gate(&tiny(), "tiny", 1e-4, 1e-4, KvTier::Int8);
}

#[test]
fn bf16_kv_tracks_f32_logits_at_the_real_attention_dims() {
    gate(&real_dims(), "real-dims", 1e-3, 1e-3, KvTier::Bf16);
}

#[test]
fn int8_kv_tracks_f32_logits_at_the_real_attention_dims() {
    gate(&real_dims(), "real-dims", 4e-3, 4e-3, KvTier::Int8);
}

#[test]
fn a_compact_pool_is_reported_at_the_bytes_it_occupies() {
    let cfg = tiny();
    let weights = qwen35::init::init_weights(&cfg, 11);
    let bytes = |tier| Engine::from_map_kv(cfg.clone(), &weights, 64, 2, tier).kv_pool_bytes();
    let (f32b, bf16b, int8b) = (bytes(KvTier::F32), bytes(KvTier::Bf16), bytes(KvTier::Int8));
    assert!(bf16b < f32b && int8b < bf16b, "pool bytes must shrink with the tier: f32 {f32b}, bf16 {bf16b}, int8 {int8b}");
}
