// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured low-precision tiers for the inference engines of its clients. If your team
// needs expertise in deciding what a narrower KV cache costs a model's answers then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec of the engine's bf16 K/V tier (`EngineOptions::kv_tier = KvTier::BF16`, `brain serve --kv-cache bf16`):
//!
//! - the engine passes the generic serving conformance suite with every K/V row (paged planes and rings) in bf16;
//! - its precision against the f32 tier is measured as the KL divergence of the served next-token distributions and
//!   their top-1 agreement at every position of a teacher-forced sequence, on the tiny reference checkpoint and on the
//!   released weights, and held to a ceiling here (the measured values are in the inkling2 ledger, I3.4).

use std::path::PathBuf;

use inkling2::serve::{Engine, EngineOptions};
use inkling2::Inkling2;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::conformance::{self, Conformance};
use model::serve::visit_state::VisitPoolSizing;
use model::serve::{PagedDecoder, PrefillProgress};

const REAL: &str = "thinkingmachines/Inkling-2-Mini";

/// KL(p || q) of the softmax distributions of two logit rows, in nats, accumulated in f64.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let log_softmax = |row: &[f32]| {
        let max = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
        let lse = max + row.iter().map(|&x| (f64::from(x) - max).exp()).sum::<f64>().ln();
        row.iter().map(|&x| f64::from(x) - lse).collect::<Vec<f64>>()
    };
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
}

/// How far the bf16 tier's served rows are from the f32 tier's on the same positions.
#[derive(Debug)]
struct Spread {
    mean_kl: f64,
    max_kl: f64,
    top1: f64,
}

fn spread(reference: &[Vec<f32>], narrow: &[Vec<f32>]) -> Spread {
    assert_eq!(reference.len(), narrow.len());
    let kls: Vec<f64> = reference.iter().zip(narrow).map(|(p, q)| kl(p, q)).collect();
    let agree = reference.iter().zip(narrow).filter(|(p, q)| argmax(p) == argmax(q)).count();
    Spread { mean_kl: kls.iter().sum::<f64>() / kls.len() as f64, max_kl: kls.iter().fold(0.0, |m: f64, &k| m.max(k)), top1: agree as f64 / reference.len() as f64 }
}

/// The served logit rows at every position of `seq`: its first token prefilled, then each next one decoded
/// teacher-forced, so every row but the first reads K/V the engine stored at its tier.
fn served_rows(e: &mut Engine, seq: &[u32]) -> Vec<Vec<f32>> {
    let mut table = BlockTable::new();
    let hidden = loop {
        if let PrefillProgress::Done(h) = e.prefill_chunk(&mut table, &seq[..1], 1) {
            break h;
        }
    };
    let mut rows = vec![e.logits(&hidden)];
    for &t in &seq[1..seq.len() - 1] {
        rows.push(e.forward_batched_logit_rows(&mut [&mut table], &[t]).remove(0));
    }
    e.release_table(&mut table);
    rows
}

fn tiny_dir() -> Option<PathBuf> {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    if dir.join("model.safetensors").is_file() {
        Some(dir)
    } else {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        None
    }
}

fn tiny_options(kv_tier: KvTier) -> EngineOptions {
    EngineOptions { kv_tier, sizing: VisitPoolSizing { block_size: 8, num_blocks: 64, live_rows: 4, snapshot_rows: 4, undo_rows: false }, max_seq_len: 128, prefill_chunk: 7 }
}

fn bf16_engine() -> Engine {
    Engine::open(crate::tiny_parity::ops(), &tiny_dir().expect("checked"), tiny_options(KvTier::BF16)).expect("the tiny engine")
}

#[test]
fn the_bf16_tier_passes_the_serving_conformance_suite() {
    if tiny_dir().is_none() {
        return;
    }
    conformance::run_all(&Conformance { make: bf16_engine, prefix_reuse: true, prefix_prompt_tokens: 20, load_adapter: None });
}

#[test]
fn the_bf16_tier_stays_within_its_measured_spread_of_f32_on_the_tiny_checkpoint() {
    let Some(dir) = tiny_dir() else { return };
    // Past the window (4) and the relative-bias extent (6) of the tiny config, so rings wrap and every bias distance
    // reads a stored row.
    let seq: Vec<u32> = (0..40u32).map(|i| (i * 37 + 11) % 60 + 1).collect();
    let mut f32_engine = Engine::open(crate::tiny_parity::ops(), &dir, tiny_options(KvTier::F32)).unwrap();
    let reference = served_rows(&mut f32_engine, &seq);
    let mut bf16 = Engine::new(f32_engine.into_model(), tiny_options(KvTier::BF16)).unwrap();
    let narrow = served_rows(&mut bf16, &seq);
    let s = spread(&reference, &narrow);
    eprintln!("tiny, bf16 K/V vs f32 over {} positions: {s:?}", reference.len());
    // Measured 2026-11-22 (CPU): mean KL 1.2e-4, max 6.5e-4, top-1 37 of 39. The tiny config caps its logits at 1.5,
    // so its distributions are flat and near ties common: the flips are ties, not a broken cache.
    assert!(s.mean_kl < 5e-4 && s.max_kl < 2e-3, "{s:?}");
    assert!(s.top1 >= 0.9, "{s:?}");
}

/// The released weights: three chat prompts and their f32-greedy continuations, every position served at f32 and at
/// bf16 K/V from one upload of the weights (about 25 GB resident).
#[test]
fn the_bf16_tier_stays_within_its_measured_spread_of_f32_on_the_released_weights() {
    let Some(dir) = brain_testutil::model_dir(REAL).map(PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file()) else {
        brain_testutil::skip_unavailable(&format!("{REAL} is not in the model store"));
        return;
    };
    let gold_dir = brain_testutil::testdata_path("golden/inkling2/real-f32");
    if !gold_dir.join("golden.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/real-f32 absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py real-f32)");
        return;
    }
    let gold: std::collections::HashMap<String, Vec<f32>> = checkpoint::safetensors::read(&gold_dir.join("golden.safetensors").to_string_lossy()).unwrap().into_iter().map(|t| (t.name, t.data)).collect();
    let options = |kv_tier| EngineOptions { kv_tier, sizing: VisitPoolSizing { block_size: 16, num_blocks: 8, live_rows: 1, snapshot_rows: 0, undo_rows: false }, max_seq_len: 128, prefill_chunk: 16 };
    const NEW: usize = 8;
    let mut f32_engine = Engine::open(crate::tiny_parity::ops(), &dir, options(KvTier::F32)).unwrap();
    let mut seqs = Vec::new();
    for p in 0..3 {
        let mut seq: Vec<u32> = gold[&format!("p{p}.tokens")].iter().map(|&t| t as u32).collect();
        let mut table = BlockTable::new();
        let hidden = f32_engine.prefill(&mut table, &seq);
        let mut next = f32_engine.admit_greedy(&hidden);
        for _ in 0..NEW {
            seq.push(next);
            next = f32_engine.forward_batched_greedy(&mut [&mut table], &[next])[0];
        }
        f32_engine.release_table(&mut table);
        seqs.push(seq);
    }
    let reference: Vec<Vec<f32>> = seqs.iter().flat_map(|s| served_rows(&mut f32_engine, s)).collect();
    let model: Inkling2 = f32_engine.into_model();
    let mut bf16 = Engine::new(model, options(KvTier::BF16)).unwrap();
    let narrow: Vec<Vec<f32>> = seqs.iter().flat_map(|s| served_rows(&mut bf16, s)).collect();
    let s = spread(&reference, &narrow);
    eprintln!("released weights, bf16 K/V vs f32 over {} positions: {s:?}", reference.len());
    // Measured 2026-11-22 (CPU, 97 positions): mean KL 1.4e-3 nats, max 2.0e-2, top-1 agreement 1.0.
    assert!(s.mean_kl < 3e-3 && s.max_kl < 5e-2, "{s:?}");
    assert!(s.top1 >= 0.95, "{s:?}");
}
